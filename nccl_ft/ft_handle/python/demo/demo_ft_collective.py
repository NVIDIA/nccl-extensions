#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault-Tolerant Collective Handle — Python Demo

Demonstrates the FTCollectiveHandle Python bindings by running fault-tolerant
AllGather and AllReduce collectives across multiple GPUs on a single node.

Each rank is a separate Python process (spawned via multiprocessing).
The NCCL unique ID is exchanged through a shared-memory Array so no file
system I/O is required.

Usage:
    python demo_ft_collective.py [--nranks N] [--count C] [--timeout-us T]

    --nranks N      Number of GPUs / ranks to use (default: all visible GPUs,
                    up to 8).
    --count C       Number of float elements per rank (default: 256).
    --timeout-us T  Timeout in microseconds for the normal runs (default: 10 s).

Requirements:
    - libft_collective.so built and on LD_LIBRARY_PATH (or FT_COLLECTIVE_LIB set)
    - NCCL_HOME pointing to an NCCL installation with Device API support
    - LSA (NVLink, single node) or GIN (RDMA, multi-node) connectivity
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time

# ---------------------------------------------------------------------------
# Make sure the local ft_collective package is importable even if not installed.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)   # python/ (parent of demo/) holds ft_collective/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from ft_collective import (
    FTLibrary, NCCLBaseLibrary, CUDARuntime,
    FT_OK, FT_TIMEOUT,
    ncclUniqueId,
)


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def rank_worker(rank: int, nranks: int, count: int, timeout_us: int,
                shared_uid: mp.Array, ready_barrier: mp.Barrier,
                result_dict: dict) -> None:
    """Entry point for each rank process.

    Args:
        rank:          This rank's index (0 … nranks-1).
        nranks:        Total number of ranks.
        count:         Float elements per rank for the collectives.
        timeout_us:    Timeout for normal runs (microseconds).
        shared_uid:    ctypes c_byte Array(128) — rank 0 writes the unique ID,
                       all others read it after the barrier.
        ready_barrier: mp.Barrier synchronising unique-ID broadcast.
        result_dict:   mp.Manager dict for collecting per-rank results.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Set CUDA device and load libraries
        # ------------------------------------------------------------------
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        nccl = NCCLBaseLibrary()
        ft   = FTLibrary()
        cuda = CUDARuntime()

        # ------------------------------------------------------------------
        # 2. Exchange NCCL unique ID via shared memory
        # ------------------------------------------------------------------
        uid = ncclUniqueId()
        if rank == 0:
            uid = nccl.get_unique_id()
            for i in range(128):
                shared_uid[i] = uid.internal[i]

        ready_barrier.wait()   # all ranks wait until rank 0 has written

        if rank != 0:
            for i in range(128):
                uid.internal[i] = shared_uid[i]

        # ------------------------------------------------------------------
        # 3. Initialise NCCL communicator
        # ------------------------------------------------------------------
        comm = nccl.comm_init_rank(nranks, uid, rank)

        # ------------------------------------------------------------------
        # 4. Create FTCollectiveHandle with full initial mask
        # ------------------------------------------------------------------
        full_mask = [True] * nranks
        handle    = ft.handle_create(comm, nranks, full_mask)

        # ------------------------------------------------------------------
        # 5. AllGather — normal run (generous timeout)
        # ------------------------------------------------------------------
        if rank == 0:
            print(f"\n=== AllGather: normal run (timeout {timeout_us // 1_000_000} s) ===")

        ag_ctx = ft.allgather_create(handle, count)

        send_data = [float(rank * 1000 + i) for i in range(count)]
        ft.allgather_launch(ag_ctx, send_data, count, timeout_us=timeout_us)
        cuda.stream_synchronize(0)   # wait for kernel before reading results

        ag_mask   = ft.handle_get_result_mask(handle, nranks)
        ag_status = ft.handle_get_error(handle)
        gathered  = ft.allgather_get_result(ag_ctx, nranks, count)

        # Verify: gathered[r][i] should equal r*1000 + i
        errors = []
        for r in range(nranks):
            if not ag_mask[r]:
                continue
            for i in range(min(count, 8)):   # spot-check first 8 elements
                expected = float(r * 1000 + i)
                if gathered[r][i] != expected:
                    errors.append(f"  [r={r}][i={i}]: got {gathered[r][i]}, want {expected}")

        if rank == 0:
            status_str = "FT_OK" if ag_status == FT_OK else "FT_TIMEOUT"
            print(f"  AllGather mask:   {[int(b) for b in ag_mask]}")
            print(f"  AllGather status: {status_str}")
            print(f"  Data check:       {'PASSED' if not errors else 'FAILED'}")
            for e in errors[:5]:
                print(e)

        ft.handle_clear_error(handle)

        # ------------------------------------------------------------------
        # 6. AllGather — timeout demonstration (1-cycle budget)
        # ------------------------------------------------------------------
        if rank == 0:
            print(f"\n=== AllGather: timeout demonstration (1-cycle budget) ===")

        ft.allgather_launch(ag_ctx, send_data, count, timeout_us=1)   # effectively 0 cycles
        cuda.stream_synchronize(0)

        to_mask   = ft.handle_get_result_mask(handle, nranks)
        to_status = ft.handle_get_error(handle)

        if rank == 0:
            status_str = "FT_TIMEOUT" if to_status == FT_TIMEOUT else "FT_OK"
            print(f"  Returned mask:    {[int(b) for b in to_mask]}")
            print(f"  Error status:     {status_str}")
            print(f"  Own-rank bit set: {'YES (correct)' if to_mask[rank] else 'NO (BUG)'}")
            print(f"  No deadlock:      YES")

        ft.handle_set_mask(handle, to_mask, nranks)
        ft.handle_clear_error(handle)

        ft.allgather_destroy(ag_ctx)

        # ------------------------------------------------------------------
        # 7. AllReduce — normal run
        # ------------------------------------------------------------------
        ft.handle_set_mask(handle, full_mask, nranks)
        if rank == 0:
            print(f"\n=== AllReduce: normal run (timeout {timeout_us // 1_000_000} s) ===")

        ar_ctx = ft.allreduce_create(handle, count)

        ft.allreduce_launch(ar_ctx, send_data, count, timeout_us=timeout_us)
        cuda.stream_synchronize(0)   # wait for kernel before reading results

        ar_mask   = ft.handle_get_result_mask(handle, nranks)
        ar_status = ft.handle_get_error(handle)
        reduced   = ft.allreduce_get_result(ar_ctx, count)

        if rank == 0:
            # Expected: reduced[i] = sum_r(r*1000 + i) for r in 0..nranks-1
            rank_sum = nranks * (nranks - 1) / 2.0
            ar_errors = []
            for i in range(min(count, 8)):
                expected = 1000.0 * rank_sum + i * nranks
                if abs(reduced[i] - expected) > 1e-3:
                    ar_errors.append(f"  [i={i}]: got {reduced[i]}, want {expected}")

            status_str = "FT_OK" if ar_status == FT_OK else "FT_TIMEOUT"
            print(f"  AllReduce mask:   {[int(b) for b in ar_mask]}")
            print(f"  AllReduce status: {status_str}")
            print(f"  Data check:       {'PASSED' if not ar_errors else 'FAILED'}")
            for e in ar_errors[:5]:
                print(e)

        ft.handle_clear_error(handle)
        ft.allreduce_destroy(ar_ctx)

        # ------------------------------------------------------------------
        # 8. Cleanup
        # ------------------------------------------------------------------
        ft.handle_destroy(handle)
        nccl.comm_finalize(comm)
        nccl.comm_destroy(comm)

        result_dict[rank] = "OK"

    except Exception as exc:
        result_dict[rank] = f"ERROR: {exc}"
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nranks",     type=int, default=0,
                        help="Number of ranks (0 = all visible GPUs, up to 8)")
    parser.add_argument("--count",      type=int, default=256,
                        help="Float elements per rank (default: 256)")
    parser.add_argument("--timeout-us", type=int, default=10_000_000,
                        help="Timeout for normal runs in µs (default: 10 000 000)")
    args = parser.parse_args()

    # Determine number of GPUs to use.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ngpu = len(visible.split(","))
    else:
        # Try to count GPUs via nvidia-smi.
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            )
            ngpu = len(out.strip().splitlines())
        except Exception:
            ngpu = 1

    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 1:
        print("ERROR: no GPUs found", file=sys.stderr)
        sys.exit(1)

    print(f"Fault-Tolerant Collective Handle Demo")
    print(f"  ranks   : {nranks}")
    print(f"  count   : {args.count} floats/rank")
    print(f"  timeout : {args.timeout_us // 1_000_000} s ({args.timeout_us} µs)")

    # Shared state between processes.
    ctx         = mp.get_context("spawn")
    shared_uid  = ctx.Array(ctypes.c_byte, 128)
    barrier     = ctx.Barrier(nranks)
    manager     = ctx.Manager()
    result_dict = manager.dict()

    procs = []
    for r in range(nranks):
        p = ctx.Process(
            target=rank_worker,
            args=(r, nranks, args.count, args.timeout_us,
                  shared_uid, barrier, result_dict),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\n=== Summary ===")
    all_ok = True
    for r in range(nranks):
        status = result_dict.get(r, "DID NOT REPORT")
        print(f"  Rank {r}: {status}")
        if status != "OK":
            all_ok = False

    print(f"\nOverall: {'PASSED' if all_ok else 'FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
