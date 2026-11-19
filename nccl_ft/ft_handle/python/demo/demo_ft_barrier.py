#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo/test for the fault-tolerant FTProcessGroup.barrier().

Two modes:

  Kill-test (default): loops dist.barrier(group=pg) once per --interval seconds.
  Kill any non-zero rank with `kill <PID>` and watch the barrier NOT hang — after
  the FT timeout every survivor drops the dead rank and reports the SAME new
  active set (consistent membership). A plain NCCL barrier would hang forever.

  Timing (--timing N): runs N back-to-back barriers and reports per-call latency
  (mean/p50/p95/p99) after --warmup. Compare --barrier-mode store vs collective
  to see the steady-state cost of a fault-tolerant barrier.

Examples:
  python demo_ft_barrier.py --nranks 8                      # kill-test, store mode
  python demo_ft_barrier.py --nranks 8 --barrier-mode collective
  python demo_ft_barrier.py --nranks 8 --timing 300         # perf micro-timing
  python demo_ft_barrier.py --nranks 8 --timing 300 --barrier-mode collective

Requirements: libft_collective.so on LD_LIBRARY_PATH (or FT_COLLECTIVE_LIB),
ft_collective Python package installed.
"""

import argparse
import datetime
import os
import signal
import socket
import statistics
import sys
import time
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)   # python/ (parent of demo/) holds ft_collective/


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _rank_worker(rank: int, nranks: int, args, store_port: int) -> None:
    """Per-rank subprocess: init an FTProcessGroup and loop the FT barrier."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["FT_BARRIER_MODE"] = args.barrier_mode
    os.environ["FT_TIMEOUT_US"]   = str(args.timeout_s * 1_000_000)

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)   # only GPU 0 is visible (= physical GPU <rank>)
    print(f"  rank {rank}  pid {os.getpid():7d}  gpu {rank}", flush=True)

    store = dist.TCPStore(
        "127.0.0.1", store_port, nranks,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=120),
    )

    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    from ft_collective import get_ft_process_group   # registers "ft_nccl"

    _device_id = torch.device("cuda:0")
    try:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=nranks, device_id=_device_id)
    except TypeError:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=nranks)
    pg = get_ft_process_group()

    if rank == 0:
        sep = "=" * 66
        kind = (f"timing: {args.timing} back-to-back barriers"
                if args.timing > 0 else
                f"kill-test: 1 barrier / {args.interval}s (kill a PID to test)")
        print(f"\n{sep}\n  FT barrier demo — mode={args.barrier_mode}  "
              f"nranks={nranks}\n  {kind}\n{sep}\n", flush=True)

    prev_active = list(range(nranks))
    lat = []
    n_iter = args.timing if args.timing > 0 else 10 ** 9

    for i in range(n_iter):
        t0 = time.perf_counter_ns()
        dist.barrier(group=pg)                     # -> FTProcessGroup.barrier -> ft_converge
        us = (time.perf_counter_ns() - t0) / 1_000.0
        lat.append(us)

        # Membership after this barrier (this rank's agreed surviving set).
        active  = pg.get_active_mask()
        act_idx = [r for r in range(nranks) if active[r]]
        if act_idx != prev_active:
            dropped = sorted(set(prev_active) - set(act_idx))
            print(f"  rank {rank}: DROPPED {dropped}  ->  active={act_idx}",
                  flush=True)
            prev_active = act_idx

        if args.timing == 0:
            if rank == 0:
                print(f"  [iter {i + 1:4d}]  barrier {us:8.1f} us  "
                      f"active={act_idx}", flush=True)
            time.sleep(args.interval)

    if args.timing > 0 and rank == 0:
        s = sorted(lat[args.warmup:]) or sorted(lat)
        def pct(p):
            return s[min(len(s) - 1, int(len(s) * p))]
        print(f"\n  {args.barrier_mode} barrier — {len(s)} calls "
              f"(after {args.warmup} warmup):", flush=True)
        print(f"    mean {statistics.mean(s):8.1f} us   p50 {pct(0.5):8.1f}   "
              f"p95 {pct(0.95):8.1f}   p99 {pct(0.99):8.1f}   "
              f"max {max(s):8.1f}", flush=True)

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nranks", type=int, default=0,
                        help="Number of ranks / GPUs (0 = all visible, up to 8)")
    parser.add_argument("--barrier-mode", choices=["store", "collective"],
                        default="store",
                        help="FT_BARRIER_MODE: store (TCPStore) or collective "
                             "(store-free FT AllGather). Default: store")
    parser.add_argument("--timeout-s", type=int, default=10,
                        help="FT barrier timeout in seconds (default: 10)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between barriers in kill-test mode")
    parser.add_argument("--timing", type=int, default=0,
                        help="If >0, run N back-to-back barriers and report "
                             "latency stats, then exit (perf mode)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup barriers to skip in timing mode")
    args = parser.parse_args()

    ngpu   = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 2:
        print("ERROR: at least 2 GPUs are required.", file=sys.stderr)
        sys.exit(1)

    store_port = _find_free_port()
    print(f"Starting {nranks} rank processes on port {store_port} …", flush=True)

    ctx   = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_rank_worker, args=(r, nranks, args, store_port),
                    name=f"rank-{r}", daemon=False)
        for r in range(nranks)
    ]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\nCtrl+C — terminating ranks …", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        sys.exit(0)

    failed = [(p.name, p.exitcode) for p in procs
              if p.exitcode not in (0, None, -signal.SIGTERM, -signal.SIGKILL)]
    for name, code in failed:
        print(f"  {name} exited with code {code}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
