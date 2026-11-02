#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault-Tolerant Collective Demo (AllGather / AllReduce)

Starts one process per GPU rank and runs a collective once per second.  Kill
any rank from a second terminal to observe the behaviour difference between the
two backends:

  ft_nccl backend — survives rank failure: after the configured timeout the
                    kernel returns a partial result + mask.  Subsequent
                    iterations exclude the dead rank and succeed immediately.

  nccl backend    — does NOT survive rank failure: after the watchdog timeout
                    all remaining ranks receive an exception and exit.

The default count (131072) puts each transfer above the multi-CTA threshold,
so the FT path exercises the cooperative multi-CTA kernel; the header prints
which kernel path is active.  Every iteration validates ALL result elements
(not just element 0) and prints CHECK ok / CHECK FAIL — so a multi-CTA
grid-partitioning or grid.sync bug surfaces immediately, with or without a
fault.  Use a small --count (e.g. 16) to test the single-CTA path instead.

Usage
-----
  # FT AllGather, multi-CTA path (default count crosses the threshold):
  python demo_fault_tolerance.py --backend ft_nccl --nranks 8

  # FT AllReduce, multi-CTA path:
  python demo_fault_tolerance.py --backend ft_nccl --collective allreduce --nranks 8

  # Single-CTA path (small count):
  python demo_fault_tolerance.py --backend ft_nccl --count 16

  # Standard NCCL backend (to show the contrast):
  python demo_fault_tolerance.py --backend nccl --nranks 8

  # Faster timeout for demos:
  python demo_fault_tolerance.py --backend ft_nccl --timeout-s 5

To kill a rank, look for the PID printed at startup and run:
  kill <PID>

Requirements
------------
  - libft_collective.so built and on LD_LIBRARY_PATH (or FT_COLLECTIVE_LIB)
  - nccl4py not required; ft_collective Python package installed
  - At least 2 GPUs on a single node
"""

import argparse
import datetime
import os
import signal
import socket
import sys
import time
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)   # python/ (parent of demo/) holds ft_collective/


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _count_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return len(visible.split(","))
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return len(out.strip().splitlines())
    except Exception:
        return 0


def _verify(mode, ar_tensor, output_list, mask, nranks, active, my_rank=0):
    """Validate ALL elements of the result; return (ok, detail_str).

    Run on rank 0 only.  Checking every element (not just [0]) is the point of
    the multi-CTA test: a grid-partitioning or grid.sync bug shows up as wrong
    values in some elements, which a single-element peek would miss.

    mask is None for the standard NCCL backend (all ranks assumed active).
    """
    import torch
    if mode == "allreduce":
        # Each responding rank r contributed value r; the sum reduces over the
        # ranks that arrived (== active), so every element must equal sum(active).
        expected = float(sum(active))
        ok = bool(torch.all(ar_tensor == expected).item())
        return ok, f"sum={ar_tensor[0].item():.0f} (expect {expected:.0f})"
    if mode == "alltoall":
        # ROUTING-SENSITIVE check: source s filled its chunk-for-dest-d with
        # (s*nranks + d), so this rank's slot s (received from source s) must be
        # uniformly (s*nranks + my_rank).  A wrong send offset / transposition /
        # slot mixup produces a different constant, which this catches.
        ok   = True
        vals = []
        for s in range(nranks):
            if mask is None or mask[s]:
                expect = float(s * nranks + my_rank)
                if not bool(torch.all(output_list[s] == expect).item()):
                    ok = False
                vals.append(f"{output_list[s][0].item():.0f}/{expect:.0f}")
            else:
                vals.append("—")
        return ok, f"slot=got/expect {vals}"
    # allgather: each active rank r's slice must be uniformly r; "—" if absent.
    ok   = True
    vals = []
    for r in range(nranks):
        if mask is None or mask[r]:
            if not bool(torch.all(output_list[r] == float(r)).item()):
                ok = False
            vals.append(f"{output_list[r][0].item():.0f}")
        else:
            vals.append("—")
    return ok, f"values={vals}"


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def _rank_worker(rank: int, nranks: int, args, store_port: int) -> None:
    """Entry point executed in each rank's subprocess.

    The process is intentionally killable with a plain ``kill <PID>`` or
    ``kill -9 <PID>`` — no special cleanup is expected after death.
    """
    # Workers ignore SIGINT so Ctrl+C on the terminal only fires in the
    # parent, which then terminates children in an orderly fashion.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Pin this process to one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)   # only GPU 0 is visible (= physical GPU <rank>)

    # Print PID early so the user can copy it before collectives start.
    print(f"  rank {rank}  pid {os.getpid():7d}  gpu {rank}", flush=True)

    # ------------------------------------------------------------------ #
    #  Rendezvous store                                                    #
    # ------------------------------------------------------------------ #
    store = dist.TCPStore(
        "127.0.0.1", store_port, nranks,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=120),
    )

    # ------------------------------------------------------------------ #
    #  Init process group and allocate buffers                            #
    # ------------------------------------------------------------------ #
    # Pass device_id so NCCL does not guess from global rank.
    # Each rank pins one GPU as cuda:0 via CUDA_VISIBLE_DEVICES; without
    # device_id NCCL picks device <global_rank> which may not exist.
    _device_id = torch.device("cuda:0")

    _DTYPE_MAP = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype  = _DTYPE_MAP[args.dtype]
    is_ar  = (args.collective == "allreduce")
    is_a2a = (args.collective == "alltoall")

    # Buffers per collective.  send_tensor/ar_tensor must be symmetric (from
    # pg.empty) on the FT path so the kernel can use the symmetric-memory route.
    # The allgather output is one contiguous block (views) so the FT scatter
    # coalesces into a single strided copy — the realistic layout.
    send_tensor = None   # allgather/alltoall input (symmetric on FT path)
    ar_tensor   = None   # allreduce in/out (symmetric on FT path)
    output_list = None   # allgather/alltoall output (contiguous views)
    out_flat    = None   # alltoall flat output tensor (for all_to_all_single)

    if args.backend == "ft_nccl":
        # Add ft_collective package to path and register the backend.
        if _PKG_ROOT not in sys.path:
            sys.path.insert(0, _PKG_ROOT)
        from ft_collective import get_ft_process_group   # registers "ft_nccl"

        try:
            dist.init_process_group(
                "ft_nccl", store=store, rank=rank, world_size=nranks,
                device_id=_device_id,
            )
        except TypeError:
            dist.init_process_group(
                "ft_nccl", store=store, rank=rank, world_size=nranks,
            )
        # _get_default_group() returns the C-level ProcessGroup wrapper which
        # cannot access Python-only methods (empty, get_result_mask).
        # Use get_ft_process_group() to retrieve the Python FTProcessGroup.
        pg = get_ft_process_group()

        if is_ar:
            ar_tensor = pg.empty(args.count, dtype=dtype)
        elif is_a2a:
            # per-destination chunks: send holds nranks*count (chunk d -> rank d)
            send_tensor = pg.empty(nranks * args.count, dtype=dtype)
            out_flat    = torch.zeros(nranks * args.count, dtype=dtype, device="cuda")
            output_list = list(out_flat.chunk(nranks))
        else:
            send_tensor = pg.empty(args.count, dtype=dtype)
            _out_buf    = torch.zeros(nranks * args.count, dtype=dtype, device="cuda")
            output_list = list(_out_buf.chunk(nranks))

    else:
        # Standard NCCL — set a watchdog timeout so we see a clean error
        # rather than hanging forever.
        try:
            dist.init_process_group(
                "nccl", store=store, rank=rank, world_size=nranks,
                timeout=datetime.timedelta(seconds=args.timeout_s + 5),
                device_id=_device_id,
            )
        except TypeError:
            dist.init_process_group(
                "nccl", store=store, rank=rank, world_size=nranks,
                timeout=datetime.timedelta(seconds=args.timeout_s + 5),
            )
        pg = None
        if is_ar:
            ar_tensor = torch.zeros(args.count, dtype=dtype, device="cuda")
        elif is_a2a:
            send_tensor = torch.zeros(nranks * args.count, dtype=dtype, device="cuda")
            out_flat    = torch.zeros(nranks * args.count, dtype=dtype, device="cuda")
            output_list = list(out_flat.chunk(nranks))
        else:
            send_tensor = torch.zeros(args.count, dtype=dtype, device="cuda")
            _out_buf    = torch.zeros(nranks * args.count, dtype=dtype, device="cuda")
            output_list = list(_out_buf.chunk(nranks))

    # ------------------------------------------------------------------ #
    #  Print header (rank 0 only)                                         #
    # ------------------------------------------------------------------ #
    if rank == 0:
        elem_size      = torch.tensor([], dtype=dtype).element_size()
        per_rank_bytes = args.count * elem_size
        threshold      = int(os.environ.get("FT_MULTICTA_THRESHOLD_BYTES", 262144))
        # Approximate: on a single-node NVLink (LSA) box with cooperative launch
        # support, the FT path takes the multi-CTA kernel when per-rank bytes
        # reach the threshold.  (Does not account for max_coop_blocks/LSA, which
        # hold on the target hardware.)
        multicta = (args.backend == "ft_nccl") and (per_rank_bytes >= threshold)
        sep = "=" * 64
        print(f"\n{sep}", flush=True)
        print(f"  Backend    : {args.backend}", flush=True)
        print(f"  Collective : {args.collective}", flush=True)
        print(f"  Ranks      : {nranks}", flush=True)
        print(f"  Count      : {args.count} × {args.dtype} / rank "
              f"({per_rank_bytes} B)", flush=True)
        if args.backend == "ft_nccl":
            print(f"  Kernel path: {'multi-CTA (cooperative)' if multicta else 'single-CTA'}"
                  f"  [threshold {threshold} B/rank]", flush=True)
        print(f"  Timeout    : {args.timeout_s} s", flush=True)
        print(f"  Interval   : {args.interval} s", flush=True)
        print(f"{sep}", flush=True)
        if args.backend == "ft_nccl":
            print(
                "  Kill any non-zero rank with  kill <PID>  and watch the\n"
                f"  mask shrink — {args.collective} keeps running with the\n"
                "  survivors, and CHECK stays ok (result reduces/gathers over\n"
                "  the surviving ranks).",
                flush=True,
            )
        else:
            print(
                "  Kill any rank with  kill <PID>  — after the watchdog\n"
                "  timeout all remaining ranks will receive an error and stop.",
                flush=True,
            )
        print(f"{sep}\n", flush=True)

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #
    iteration = 0
    prev_active = list(range(nranks))   # rank-0 display: detect newly-dropped ranks
    while True:
        iteration += 1

        # Each rank's send value is its rank index so results are easy to read.
        if is_ar:
            ar_tensor.fill_(float(rank))
        elif is_a2a:
            # routing-sensitive fill: this rank's chunk for destination d holds
            # (rank*nranks + d), so the receiver can verify the RIGHT chunk from
            # the RIGHT source landed in the RIGHT slot.
            for d in range(nranks):
                send_tensor[d * args.count:(d + 1) * args.count].fill_(
                    float(rank * nranks + d))
        else:
            send_tensor.fill_(float(rank))

        try:
            # ---------------------------------------------------------- #
            #  FT backend                                                 #
            # ---------------------------------------------------------- #
            if args.backend == "ft_nccl":
                # Pre-sync: zero d_ready synchronously on this rank's GPU,
                # then barrier all ranks so no rank starts the next kernel
                # until every peer's d_ready is confirmed 0.  Prevents the
                # stale-readywin race that causes early-iteration timeouts.
                # ft_converge runs the fault-tolerant store barrier: it
                # re-synchronizes survivors and agrees a single surviving set
                # (a killed rank is dropped here, consistently across ranks).
                pg.ft_converge()
                # The collective dispatches through our Python FTProcessGroup
                # override (pg is the Python object); work.wait() is internal.
                if is_ar:
                    dist.all_reduce(ar_tensor, group=pg)
                elif is_a2a:
                    dist.all_to_all_single(out_flat, send_tensor, group=pg)
                else:
                    dist.all_gather(output_list, send_tensor, group=pg)

                # get_result_mask() syncs the stream and returns this rank's
                # per-iteration observation of who delivered (used for CHECK).
                mask = pg.get_result_mask()

                if rank == 0:
                    active   = [r for r in range(nranks) if mask[r]]
                    dropped  = sorted(set(prev_active) - set(active))
                    ok, detail = _verify(args.collective, ar_tensor, output_list,
                                         mask, nranks, active, my_rank=rank)
                    tag = f"DROPPED {dropped}" if dropped else "OK     "
                    chk = "CHECK ok" if ok else "CHECK FAIL"
                    print(
                        f"  [iter {iteration:4d}]  {tag}  {chk}"
                        f"  active={active}  {detail}",
                        flush=True,
                    )
                    prev_active = active

            # ---------------------------------------------------------- #
            #  Standard NCCL backend                                      #
            # ---------------------------------------------------------- #
            else:
                if is_ar:
                    dist.all_reduce(ar_tensor)
                elif is_a2a:
                    dist.all_to_all_single(out_flat, send_tensor)
                else:
                    dist.all_gather(output_list, send_tensor)

                if rank == 0:
                    active = list(range(nranks))
                    ok, detail = _verify(args.collective, ar_tensor, output_list,
                                         None, nranks, active, my_rank=rank)
                    chk = "CHECK ok" if ok else "CHECK FAIL"
                    print(
                        f"  [iter {iteration:4d}]  OK      {chk}"
                        f"  active={active}  {detail}",
                        flush=True,
                    )

        # ---------------------------------------------------------------- #
        #  Error handling                                                   #
        # ---------------------------------------------------------------- #
        except dist.DistBackendError as exc:
            print(
                f"\n  [iter {iteration:4d}]  rank {rank}  NCCL ERROR — "
                f"watchdog timeout after {args.timeout_s + 5} s:\n"
                f"    {exc}\n"
                f"  The process group is now unusable.  Exiting.",
                flush=True,
            )
            break

        except Exception as exc:
            print(
                f"\n  [iter {iteration:4d}]  rank {rank}  UNEXPECTED ERROR:\n"
                f"    {type(exc).__name__}: {exc}\n"
                f"  Exiting.",
                flush=True,
            )
            break

        time.sleep(args.interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["nccl", "ft_nccl"],
        default="ft_nccl",
        help="Collective backend (default: ft_nccl)",
    )
    parser.add_argument(
        "--nranks",
        type=int,
        default=0,
        help="Number of ranks / GPUs (0 = all visible, up to 8; default: 0)",
    )
    parser.add_argument(
        "--collective",
        choices=["allgather", "allreduce", "alltoall"],
        default="allgather",
        help="Collective to exercise (default: allgather)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=131072,
        help=(
            "Elements per rank per collective (default: 131072 = 512 KiB/rank "
            "for float32, above the default FT_MULTICTA_THRESHOLD_BYTES so the "
            "cooperative multi-CTA kernel is exercised).  Use a small count "
            "(e.g. 16) to test the single-CTA path instead."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help=(
            "Tensor element dtype (default: float32).  All three are supported "
            "by both FT AllGather and FT AllReduce (the reduction accumulates "
            "in float32 internally)."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=10,
        help=(
            "Collective timeout in seconds (default: 10).  "
            "For ft_nccl this is the FT kernel timeout; "
            "for nccl this is the NCCL watchdog timeout."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between AllGather iterations (default: 1.0)",
    )
    args = parser.parse_args()

    # Determine number of GPUs.
    ngpu = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 2:
        print(
            "ERROR: at least 2 GPUs are required for a meaningful demo.\n"
            "       Use --nranks or set CUDA_VISIBLE_DEVICES.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pass timeout to FTProcessGroup via env var (factory signature is fixed
    # by torch.distributed and doesn't forward extra kwargs).
    os.environ["FT_TIMEOUT_US"] = str(args.timeout_s * 1_000_000)

    store_port = _find_free_port()

    print(f"Starting {nranks} rank processes on port {store_port} …", flush=True)

    ctx   = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_rank_worker,
            args=(r, nranks, args, store_port),
            name=f"rank-{r}",
            daemon=False,
        )
        for r in range(nranks)
    ]

    for p in procs:
        p.start()

    # Let the parent wait for all children, forwarding Ctrl+C as SIGTERM.
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\nCtrl+C — sending SIGTERM to all ranks …", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        sys.exit(0)

    # Report exit codes.
    failed = [(p.name, p.exitcode) for p in procs
              if p.exitcode not in (0, None, -signal.SIGTERM, -signal.SIGKILL)]
    if failed:
        for name, code in failed:
            print(f"  {name} exited with code {code}", flush=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
