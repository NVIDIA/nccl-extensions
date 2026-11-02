#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate ft_rejoin(): a stalled-but-alive rank rejoins WITHOUT changing the
communicator.

Scenario
--------
  1. All ranks run warm-up collectives together (generation counters in lockstep).
  2. The victim rank STALLS (sleeps) and skips a round of work; the survivors run a
     collective (which times the victim out) and then ft_converge(), which DROPS
     the victim from their active set. The victim's on-device generation is now
     frozen behind the survivors' — a generation-gated handshake could never let it
     back in.
  3. The victim wakes. Every rank calls ft_rejoin() (a full-world collective): a
     generation max-consensus re-synchronizes the drifted counter, then a presence
     round re-admits the responders.
  4. Several NORMAL collectives run across the FULL set — each must include every
     rank in its result mask (a stale/unsynced generation would time the returnee
     out) and compute the correct sum. This is the generation-resync assertion:
     the group is only truly back in lockstep if repeated normal collectives run
     correctly with no one dropped.

PASS iff: survivors had actually dropped the victim before rejoin, ft_rejoin
returns the full set on every rank, and every post-rejoin AllReduce runs across
the complete set with the correct result.

A TCPStore barrier is used only to orchestrate *when* the collective ft_rejoin is
called (production would trigger it on an app signal / periodic cadence) — the
rejoin itself is pure on-GPU collective, no store.

Usage
-----
  python test_ft_rejoin.py --nranks 8
  python test_ft_rejoin.py --nranks 8 --victim 3 --timeout-s 5
"""

import argparse
import os
import signal
import sys
import time
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))


def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _mask_str(mask) -> str:
    return "".join("1" if b else "0" for b in mask)


def _store_barrier(store, tag: str, n: int, rank: int, timeout: float) -> None:
    """Test-only rendezvous: block until all n ranks reach `tag`."""
    store.set(f"{tag}/{rank}", "1")
    deadline = time.monotonic() + timeout
    for r in range(n):
        k = f"{tag}/{r}"
        while not store.check([k]):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"barrier {tag} timed out waiting for rank {r}")
            time.sleep(0.02)


def _rank_worker(rank: int, world: int, local_rank: int,
                 master_addr: str, master_port: int, args) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["FT_BARRIER_MODE"] = "collective"
    os.environ["FT_TIMEOUT_US"] = str(int(args.timeout_s * 1_000_000))

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)   # multi-node: this rank's GPU on THIS host

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import ft_launch
    store = ft_launch.make_store(rank, world, master_addr, master_port)
    from ft_collective import get_ft_process_group   # registers "ft_nccl"

    dev = torch.device(f"cuda:{local_rank}")
    try:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=world, device_id=dev)
    except TypeError:
        dist.init_process_group("ft_nccl", store=store, rank=rank, world_size=world)
    pg = get_ft_process_group()
    n = world
    victim = args.victim if args.victim >= 0 else n - 1
    buf = pg.empty(args.count)

    def allreduce_ones():
        buf.fill_(1.0)
        pg.allreduce([buf]).wait()
        return pg.get_result_mask()

    # 1. warm-up — all present, generations in lockstep
    for _ in range(args.warmup):
        allreduce_ones()

    # 2. stall + drop
    if rank == victim:
        time.sleep(args.stall_s)                 # unresponsive: skip this round of work
    else:
        allreduce_ones()                         # times the victim out (it is absent)
        pg.ft_converge()                         # survivors DROP the victim from _active

    # snapshot membership before rejoin (survivors should show the victim excluded)
    before = pg.get_active_mask()

    # 3. rendezvous, then the collective rejoin (store = orchestration only)
    _store_barrier(store, "pre_rejoin", n, rank, timeout=args.stall_s + 60)
    mask = pg.ft_rejoin()
    gen_after = pg.get_generation()      # synchronized generation (should match everywhere)

    # 4. Assert the generation is re-synced by running several NORMAL collectives
    #    across the FULL set. If the returnee's generation were still drifted, its
    #    push would look stale and survivors would time it out -> the result mask
    #    would drop it and/or the sum would be wrong. Requiring a COMPLETE result
    #    mask and a correct sum for every post-rejoin iteration is the resync check.
    coll_ok = True
    got = expected = 0.0
    for it in range(args.post_iters):
        buf.fill_(float(rank + 1 + it))
        pg.allreduce([buf]).wait()
        res_mask = pg.get_result_mask()
        expected = float(sum(r + 1 + it for r in range(n)))
        got = float(buf[0].item())
        if abs(got - expected) > 1e-3 or sum(res_mask) != n:
            coll_ok = False
            break

    rejoined_full = (sum(mask) == n)
    dropped_before = (not before[victim]) if rank != victim else True
    ok = rejoined_full and coll_ok and dropped_before

    store.set(f"rj/{rank}",
              f"{'OK' if ok else 'FAIL'}|gen={gen_after}|before={_mask_str(before)}|"
              f"after={_mask_str(mask)}|got={got:.1f}/{expected:.1f}")

    if rank == 0:
        print("\n" + "=" * 72, flush=True)
        print(f"  FT_REJOIN  nranks={n}  victim=rank{victim}  timeout={args.timeout_s}s",
              flush=True)
        print("=" * 72, flush=True)
        deadline = time.monotonic() + 60
        rows = {}
        all_ok = True
        for r in range(n):
            k = f"rj/{r}"
            while not store.check([k]) and time.monotonic() < deadline:
                time.sleep(0.02)
            rows[r] = store.get(k).decode() if store.check([k]) else "MISSING"
            if not rows[r].startswith("OK"):
                all_ok = False
        for r in range(n):
            tag = "victim" if r == victim else "      "
            print(f"  rank {r} {tag}  {rows[r]}", flush=True)
        print("-" * 72, flush=True)
        import re
        gens = []
        for r in range(n):
            m = re.search(r"gen=(\d+)", rows[r])
            gens.append(int(m.group(1)) if m else -1)
        gen_ok = (len(set(gens)) == 1 and gens[0] >= 0)
        print(f"  generations: {gens}  -> "
              f"{'ALL EQUAL (synced)' if gen_ok else 'MISMATCH — rejoin did not sync d_gen'}",
              flush=True)
        verdict = ("PASS — victim dropped then rejoined the full set"
                   if all_ok else "FAIL (see rows)")
        print(f"  VERDICT: {verdict}", flush=True)
        print("=" * 72 + "\n", flush=True)
        store.set("rj/verdict", "PASS" if all_ok else "FAIL")

    try:
        dist.barrier(group=pg)
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nranks", type=int, default=0)
    p.add_argument("--victim", type=int, default=-1, help="rank to stall (default: last)")
    p.add_argument("--warmup", type=int, default=5, help="lockstep warm-up collectives")
    p.add_argument("--count", type=int, default=1024, help="allreduce elements")
    p.add_argument("--timeout-s", type=float, default=5.0, help="FT / rejoin timeout")
    p.add_argument("--stall-s", type=float, default=3.0, help="victim stall duration")
    p.add_argument("--post-iters", type=int, default=10,
                   help="normal collectives after rejoin (generation-resync check)")
    args = p.parse_args()

    if args.victim == 0:
        print("ERROR: victim must be non-zero (rank 0 hosts the store).", file=sys.stderr)
        sys.exit(2)

    import ft_launch
    ngpu = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else ft_launch.world_size_hint(min(ngpu, 8))
    if nranks < 2:
        print("ERROR: at least 2 GPUs are required.", file=sys.stderr)
        sys.exit(1)

    if not ft_launch.launched():
        print(f"ft_rejoin demo: {nranks} ranks (single host)", flush=True)
    # Under torchrun/srun this does not return (each process is one rank and
    # sys.exit()s); single-host returns per-rank exit codes to aggregate.
    codes = ft_launch.launch(_rank_worker, nranks, args)
    if codes is not None:
        ok = all(c == 0 for c in codes)
        print(f"exit codes {codes} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
