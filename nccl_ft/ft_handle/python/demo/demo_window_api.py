#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault-Tolerant Window API Demo

Demonstrates the window-based ftAllGatherLSA / ftAllReduceLSA Python API.
Windows are created and owned by the caller using nccl4py, which provides:
  - nccl.core.mem_alloc()        — symmetric memory allocation (ncclMemAlloc)
  - Communicator.register_window() — window registration (ncclCommWindowRegister)
  - RegisteredWindowHandle.handle  — raw ncclWindow_t pointer passed to ft_collective

The FTCollectiveHandle manages: devComm, readywin, mask buffers, error status.
The caller manages: send/recv/flags windows, zeroing before each call.

Usage:
    python demo_window_api.py [--nranks N] [--count C] [--timeout-us T]

Requirements:
    - nccl4py installed (pip install nccl4py[cu12] or cu13)
    - libft_collective.so built and on LD_LIBRARY_PATH (or FT_COLLECTIVE_LIB set)
    - All GPUs connected via NVLink (same LSA team, nLsaTeams == 1)
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)   # python/ (parent of demo/) holds ft_collective/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from ft_collective import FTLibrary, CUDARuntime, FT_OK, FT_TIMEOUT


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def rank_worker(rank: int, nranks: int, count: int, timeout_us: int,
                shared_uid_bytes: mp.Array, barrier: mp.Barrier,
                result_dict: dict) -> None:
    try:
        # ------------------------------------------------------------------
        # 1. Assign GPU and load libraries
        # ------------------------------------------------------------------
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        from nccl.core import mem_alloc, mem_free
        from nccl.core.communicator import Communicator
        from nccl.core.constants import WindowFlag
        from nccl.core.utils import UniqueId, get_unique_id

        ft   = FTLibrary()
        cuda = CUDARuntime()

        # ------------------------------------------------------------------
        # 2. Exchange NCCL unique ID via shared memory
        # ------------------------------------------------------------------
        if rank == 0:
            uid = get_unique_id()
            uid_bytes = bytes(uid)
            for i, b in enumerate(uid_bytes):
                shared_uid_bytes[i] = b

        barrier.wait()

        if rank != 0:
            uid = UniqueId.from_bytes(bytes(shared_uid_bytes))

        # ------------------------------------------------------------------
        # 3. Create NCCL communicator (nccl4py)
        # ------------------------------------------------------------------
        comm = Communicator.init(nranks, uid, rank)
        if rank == 0:
            print(f"  Communicators initialized ({nranks} ranks)")

        # ------------------------------------------------------------------
        # 4. Allocate symmetric buffers with nccl4py mem_alloc
        # ------------------------------------------------------------------
        send_bytes  = count * 4                  # float32
        recv_bytes  = nranks * count * 4
        flags_bytes = nranks * 8                 # uint64 per rank

        buf_send  = mem_alloc(send_bytes)
        buf_recv  = mem_alloc(recv_bytes)
        buf_flags = mem_alloc(flags_bytes)

        # ------------------------------------------------------------------
        # 5. Register symmetric windows (collective operation — all ranks)
        # ------------------------------------------------------------------
        win_send  = comm.register_window(buf_send,  WindowFlag.CollSymmetric)
        win_recv  = comm.register_window(buf_recv,  WindowFlag.CollSymmetric)
        win_flags = comm.register_window(buf_flags, WindowFlag.CollSymmetric)

        # ------------------------------------------------------------------
        # 6. Create FTCollectiveHandle (window-based API needs only this)
        # ------------------------------------------------------------------
        full_mask = [True] * nranks
        handle = ft.handle_create(comm.ptr, nranks, full_mask)

        # Helper: load send data and zero recv + flags
        def prepare(send_data):
            arr = (ctypes.c_float * count)(*send_data)
            cuda.memcpy_h2d(buf_send.ptr, arr, send_bytes)
            cuda.memset(buf_recv.ptr,  0, recv_bytes)
            cuda.memset(buf_flags.ptr, 0, flags_bytes)

        # Helper: read back gathered data as list-of-lists [rank][elem]
        def read_gathered():
            out = (ctypes.c_float * (nranks * count))()
            cuda.memcpy_d2h(out, buf_recv.ptr, recv_bytes)
            return [[out[r * count + i] for i in range(count)] for r in range(nranks)]

        send_data = [float(rank * 1000 + i) for i in range(count)]

        # ------------------------------------------------------------------
        # 7. AllGather — normal run
        # ------------------------------------------------------------------
        if rank == 0:
            print(f"\n=== AllGather: normal run (timeout {timeout_us // 1_000_000} s) ===")

        prepare(send_data)
        ft.allgather(handle,
                     win_send.handle,  0,
                     win_recv.handle,  0,
                     win_flags.handle,
                     count, timeout_us=timeout_us)
        cuda.stream_synchronize(0)   # wait for kernel before reading results

        mask   = ft.handle_get_result_mask(handle, nranks)
        status = ft.handle_get_error(handle)

        if rank == 0:
            gathered = read_gathered()
            errors = [
                f"  [r={r}][i={i}]: got {gathered[r][i]}, want {float(r*1000+i)}"
                for r in range(nranks) if mask[r]
                for i in range(min(count, 8))
                if gathered[r][i] != float(r * 1000 + i)
            ]
            print(f"  Result mask:  {[int(b) for b in mask]}")
            print(f"  Error status: {'FT_OK' if status == FT_OK else 'FT_TIMEOUT'}")
            print(f"  Data check:   {'PASSED' if not errors else 'FAILED'}")
            for e in errors[:5]:
                print(e)

        ft.handle_clear_error(handle)

        # ------------------------------------------------------------------
        # 8. AllGather — timeout demonstration
        # ------------------------------------------------------------------
        if rank == 0:
            print(f"\n=== AllGather: timeout demonstration (1-cycle budget) ===")

        prepare(send_data)
        ft.allgather(handle,
                     win_send.handle,  0,
                     win_recv.handle,  0,
                     win_flags.handle,
                     count, timeout_us=1)
        cuda.stream_synchronize(0)

        to_mask   = ft.handle_get_result_mask(handle, nranks)
        to_status = ft.handle_get_error(handle)

        if rank == 0:
            print(f"  Result mask:    {[int(b) for b in to_mask]}")
            print(f"  Error status:   {'FT_TIMEOUT' if to_status == FT_TIMEOUT else 'FT_OK'}")
            print(f"  Own-rank bit:   {'set (correct)' if to_mask[rank] else 'NOT SET (bug)'}")
            print(f"  No deadlock:    YES")

        ft.handle_set_mask(handle, full_mask, nranks)
        ft.handle_clear_error(handle)

        # ------------------------------------------------------------------
        # 9. Cleanup — destroy handle, then deregister windows, free memory
        # ------------------------------------------------------------------
        ft.handle_destroy(handle)

        # Windows are deregistered automatically when the Communicator is
        # destroyed, but we can also close them explicitly here.
        win_flags.close()
        win_recv.close()
        win_send.close()

        mem_free(buf_flags)
        mem_free(buf_recv)
        mem_free(buf_send)

        comm.finalize()
        comm.destroy()

        result_dict[rank] = "OK"

    except Exception as exc:
        import traceback
        result_dict[rank] = f"ERROR: {exc}\n{traceback.format_exc()}"
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

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ngpu = len(visible.split(","))
    else:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            ngpu = len(out.strip().splitlines())
        except Exception:
            ngpu = 1

    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 1:
        print("ERROR: no GPUs found", file=sys.stderr)
        sys.exit(1)

    print(f"Fault-Tolerant Window API Demo")
    print(f"  ranks   : {nranks}")
    print(f"  count   : {args.count} floats/rank")
    print(f"  timeout : {args.timeout_us // 1_000_000} s ({args.timeout_us} µs)")

    ctx             = mp.get_context("spawn")
    shared_uid      = ctx.Array(ctypes.c_byte, 128)
    barrier         = ctx.Barrier(nranks)
    manager         = ctx.Manager()
    result_dict     = manager.dict()

    procs = [
        ctx.Process(target=rank_worker,
                    args=(r, nranks, args.count, args.timeout_us,
                          shared_uid, barrier, result_dict))
        for r in range(nranks)
    ]
    for p in procs:
        p.start()
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
