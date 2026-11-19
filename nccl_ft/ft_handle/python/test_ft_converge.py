#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agreement test for the collective-mode FT membership barrier (ft_converge).

The property under test is AGREEMENT: after a barrier, every survivor must
install the IDENTICAL surviving set. A result-mask-only barrier can violate this
at the timeout boundary (rank A sees a slow peer C, rank B does not -> split
brain -> the next collective deadlocks). The belief-exchange + intersection
barrier is supposed to converge survivors to one set.

Each iteration: run ft_converge() in collective mode, then post this rank's
active mask to the store and read every believed-alive peer's mask; PASS iff they
are all identical to ours (survivors agree). Uses the store (off-GPU, reliable)
for the cross-rank comparison so it stays valid even mid-failure.

Modes:
  default: healthy loop — every rank must agree on the full set each iteration.
  --kill : kill a non-zero PID; survivors must still AGREE on the same surviving
           set (the split-brain check). Prints DISAGREE loudly if they don't.

Usage:
  python test_ft_converge.py --nranks 8
  python test_ft_converge.py --nranks 8 --kill --timeout-s 5
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


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0)); port = s.getsockname()[1]; s.close()
    return port


def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _mask_str(mask) -> str:
    return "".join("1" if b else "0" for b in mask)


def _agree_check(pg, store, it: int, timeout_s: float):
    """Post my active mask; read every believed-alive peer's; return (ok, mask).

    ok = all believed-alive survivors posted the IDENTICAL mask to mine (agreement).
    """
    n = pg._size
    mask = pg.get_active_mask()
    mine = _mask_str(mask)
    store.set(f"agree/{it}/{pg._rank}", mine)
    survivors = [r for r in range(n) if mask[r] and r != pg._rank]
    deadline = time.monotonic() + timeout_s
    ok = True
    for r in survivors:
        k = f"agree/{it}/{r}"
        while not store.check([k]):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        if store.check([k]):
            if store.get(k).decode() != mine:      # a survivor disagrees with me
                ok = False
        # a believed-alive peer that never posts is itself a disagreement
        else:
            ok = False
    return ok, mask


def _rank_worker(rank: int, nranks: int, args, store_port: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["FT_BARRIER_MODE"] = "collective"        # exercise the collective barrier
    os.environ["FT_TIMEOUT_US"] = str(args.timeout_s * 1_000_000)

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(0)
    print(f"  rank {rank}  pid {os.getpid():7d}  gpu {rank}", flush=True)

    store = dist.TCPStore("127.0.0.1", store_port, nranks,
                          is_master=(rank == 0),
                          timeout=datetime.timedelta(seconds=120))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from ft_collective import get_ft_process_group   # registers "ft_nccl"

    _device_id = torch.device("cuda:0")
    try:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=nranks, device_id=_device_id)
    except TypeError:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=nranks)
    pg = get_ft_process_group()
    n = nranks

    if not args.kill:
        ok_all = True
        for it in range(args.iters):
            pg.ft_converge()
            ok, mask = _agree_check(pg, store, it, args.timeout_s)
            if not ok or sum(mask) != n:
                ok_all = False
        tag = "AGREE ok" if ok_all else "AGREE FAIL"
        print(f"  rank {rank}: {tag}  ({args.iters} healthy barriers)", flush=True)
        dist.barrier(group=pg)
        dist.destroy_process_group()
        sys.exit(0 if ok_all else 1)

    # ---- kill-loop mode ----
    if rank == 0:
        print("\n  Loop mode: kill a non-zero PID; survivors must AGREE on the "
              "same surviving set (no split brain).\n", flush=True)
    it = 0
    prev = list(range(n))
    while True:
        it += 1
        try:
            pg.ft_converge()
            ok, mask = _agree_check(pg, store, it, args.timeout_s)
            if rank == 0:
                active = [r for r in range(n) if mask[r]]
                newly = sorted(set(prev) - set(active))
                tag = f"DROPPED {newly}" if newly else "OK     "
                chk = "AGREE ok" if ok else "AGREE FAIL <<< SPLIT BRAIN"
                print(f"  [iter {it:4d}]  {tag}  {chk}  active={active}", flush=True)
                prev = active
        except Exception as e:
            print(f"  rank {rank}: exiting ({type(e).__name__})", flush=True)
            break
        time.sleep(args.interval)
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nranks", type=int, default=0)
    p.add_argument("--iters", type=int, default=20,
                   help="healthy-mode barrier iterations (default 20)")
    p.add_argument("--kill", action="store_true",
                   help="loop for kill-testing (verify survivors agree)")
    p.add_argument("--timeout-s", type=int, default=5)
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    ngpu   = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 2:
        print("ERROR: at least 2 GPUs are required.", file=sys.stderr)
        sys.exit(1)

    store_port = _find_free_port()
    print(f"Starting {nranks} ranks on port {store_port} …", flush=True)
    ctx   = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_worker, args=(r, nranks, args, store_port),
                         name=f"rank-{r}", daemon=False) for r in range(nranks)]
    for pr in procs:
        pr.start()
    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        for pr in procs:
            pr.join(timeout=5)
        sys.exit(0)
    codes = [pr.exitcode for pr in procs]
    sys.exit(1 if any(c not in (0, None, -signal.SIGTERM, -signal.SIGKILL) for c in codes) else 0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
