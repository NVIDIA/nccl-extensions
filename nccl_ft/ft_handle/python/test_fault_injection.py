#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault-injection test: kill a rank and attribute where survivors fault.

Motivation
----------
The FT LSA kernels touch peer memory at three points:
  Step 1  readiness push  -- an UNGATED remote store into every active-mask peer's
                             slot (gated by the input mask, NOT by liveness).
  Step 2  liveness poll   -- reads only LOCAL slots with a timeout (safe).
  Step 4  data + flag push-- GATED by s_can_push[r] (only peers that answered).

"Posted store never stalls on a dead peer" is true, but posted prevents *stalling*,
not *faulting*: once the driver has reclaimed a SIGKILLed rank's windows
(ncclMemAlloc ties the memory to the process), a survivor's store into that freed
window is an illegal access that poisons the CUDA context (sticky 715 / 700).

Empirically, teardown is NOT instantaneous: the first collective right after the
kill often SURVIVES (the store still lands on mapped memory) and the fault only
appears on a LATER access once reclamation completes. So this test (a) waits a
settle window so the fault is deterministic, and (b) runs three ATTRIBUTED probes
after the kill so we can see exactly which access faults:

  probe A  first data collective after the kill (the raw Step-1 store into freed mem)
  probe B  ft_converge()   (its all-to-all also pushes to the dead peer)
  probe C  data collective AFTER ft_converge excluded the victim (should be clean)

Modes
-----
  pre       : victim SIGKILLs itself, survivors WAIT for confirmed-dead +
              --teardown-wait-ms, THEN run the probes. Deterministic reproducer.
  during    : victim launches the collective and SIGKILLs itself BEFORE syncing,
              so its context is torn down mid-kernel while survivors are still in
              the collective (sub-case #2, timing-sampled); survivors then run
              B/C against the dead peer.
  straggler : victim stays ALIVE but sleeps past the timeout, and stays alive
              through all probes. Memory stays mapped -> survivors must SURVIVE
              every probe. CONTROL isolating fault=teardown, not timeout.

Each survivor classifies A/B/C as OK / FAULT-<detail> / HANG (the real exception
text is preserved, not just a code), posted to the store; rank 0 prints the table
plus a details section. --expect survive asserts all-OK (regression gate).

Usage
-----
  python test_fault_injection.py --nranks 8 --mode pre  --collective allgather
  python test_fault_injection.py --nranks 8 --mode straggler
  python test_fault_injection.py --nranks 8 --mode pre --expect survive   # gate
"""

import argparse
import datetime
import os
import re
import signal
import socket
import sys
import threading
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


def _self_kill():
    # SIGKILL is uncatchable: tears down the CUDA context abruptly, like a real crash.
    os.kill(os.getpid(), signal.SIGKILL)


def _fault_tag(exc: Exception):
    """Return (short_tag, full_detail) preserving the REAL error text."""
    s = " ".join(str(exc).split())            # collapse whitespace/newlines
    low = s.lower()
    if "illegal memory access" in low:
        tag = "FAULT-illegal-mem"
    elif "illegal instruction" in low or "715" in s:
        tag = "FAULT-illegal-instr(715)"
    elif "unspecified launch failure" in low or "719" in s:
        tag = "FAULT-launch(719)"
    elif "an illegal" in low or "700" in s:
        tag = "FAULT-700"
    else:
        m = re.search(r"\b(\d{3})\b", s)
        tag = f"FAULT-{m.group(1)}" if m else "FAULT-other"
    return tag, (f"{type(exc).__name__}: {s}")[:200]


class _Watchdog:
    """Post HANG to the store and hard-exit if the guarded block overruns."""
    def __init__(self, seconds, store, key):
        self._sec = seconds; self._store = store; self._key = key
        self._ev = threading.Event()

    def __enter__(self):
        def run():
            if not self._ev.wait(self._sec):
                try:
                    self._store.set(self._key, "HANG")
                    self._store.set(self._key + ".detail", "watchdog fired")
                except Exception:
                    pass
                os._exit(2)
        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._ev.set()
        return False


def _run_collective(pg, coll: str, buf, out):
    """One collective + forced stream sync (raises on a CUDA fault). Returns mask."""
    if coll == "allreduce":
        buf.fill_(1.0)
        pg.allreduce([buf]).wait()
    else:  # allgather
        buf.fill_(float(pg._rank + 1))
        pg.allgather_into_tensor(out, buf).wait()
    return pg.get_result_mask()   # blocks host until kernel done; raises on fault


def _probe(pg, store, key, watchdog_s, fn):
    """Run fn() under a watchdog; record OK / FAULT-<detail> / HANG to the store."""
    try:
        with _Watchdog(watchdog_s, store, key):
            fn()
        store.set(key, "OK"); store.set(key + ".detail", "")
        return True
    except Exception as e:
        tag, detail = _fault_tag(e)
        store.set(key, tag); store.set(key + ".detail", detail)
        return False


def _wait_key(store, key, deadline):
    while not store.check([key]):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _rank_worker(rank: int, nranks: int, args, store_port: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["FT_BARRIER_MODE"] = "collective"
    os.environ["FT_TIMEOUT_US"] = str(int(args.timeout_s * 1_000_000))

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(0)

    store = dist.TCPStore("127.0.0.1", store_port, nranks,
                          is_master=(rank == 0),
                          timeout=datetime.timedelta(seconds=180))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from ft_collective import get_ft_process_group   # registers "ft_nccl"

    dev = torch.device("cuda:0")
    try:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=nranks, device_id=dev)
    except TypeError:
        dist.init_process_group("ft_nccl", store=store, rank=rank, world_size=nranks)
    pg = get_ft_process_group()
    n = nranks

    victim = args.victim if args.victim >= 0 else n - 1
    count  = args.count
    buf = pg.empty(count)
    out = pg.empty(count * n) if args.collective == "allgather" else None
    coll = args.collective

    def A(): _run_collective(pg, coll, buf, out)
    def C(): _run_collective(pg, coll, buf, out)

    # --- healthy iters (all present) ---
    for _ in range(args.kill_iter):
        _run_collective(pg, coll, buf, out)

    # --- injection ---
    if rank == victim:
        if args.mode == "pre":
            store.set("fi/dead", "1")
            _self_kill()                                   # never returns
        elif args.mode == "during":
            # TRUE mid-collective death: launch the FT collective (async kernel)
            # and SIGKILL BEFORE the get_result_mask sync, so the victim's CUDA
            # context is torn down while its kernel is still in flight and the
            # survivors are mid-collective (sub-case #2).  The old timer approach
            # blocked on the sync, so the victim finished probe A and exited
            # cleanly (exit 0) — it never actually died mid-kernel.
            if coll == "allreduce":
                buf.fill_(1.0)
                pg.allreduce([buf]).wait()      # async launch; .wait()=stream order, no CPU block
            else:
                buf.fill_(float(rank + 1))
                pg.allgather_into_tensor(out, buf).wait()
            if args.kill_delay_ms > 0:
                time.sleep(args.kill_delay_ms / 1000.0)   # optional: sample later in the window
            _self_kill()                        # die mid-kernel, before any sync
        elif args.mode == "straggler":
            time.sleep(args.timeout_s * 1.5)               # timed out, but ALIVE + mapped...
            # ...and STAY alive (memory mapped) through the survivors' probes.
            try:
                _wait_key(store, "fi/done", time.monotonic() + args.watchdog_s * 4)
            except Exception:
                pass                                       # store server (rank 0) may have exited
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            return

    # --- survivors ---
    if args.mode == "pre":
        # Wait for confirmed death, then let teardown/reclamation settle so the
        # first probe deterministically hits freed memory.
        _wait_key(store, "fi/dead", time.monotonic() + 30)
        time.sleep(args.teardown_wait_ms / 1000.0)

    ka, kb, kc = f"fi/{rank}/A", f"fi/{rank}/B", f"fi/{rank}/C"
    okA = _probe(pg, store, ka, args.watchdog_s, A)                   # raw Step-1 store
    if okA:
        okB = _probe(pg, store, kb, args.watchdog_s, pg.ft_converge) # converge over dead peer
    else:
        store.set(kb, "SKIP"); store.set(kb + ".detail", "prior probe faulted"); okB = False
    if okB:
        okC = _probe(pg, store, kc, args.watchdog_s, C)              # post-converge collective
    else:
        store.set(kc, "SKIP"); store.set(kc + ".detail", "prior probe faulted"); okC = False
    my_clean = okA and okB and okC

    # --- rank 0 aggregates + prints ---
    if rank == 0:
        survivors = [r for r in range(n) if r != victim]
        deadline = time.monotonic() + max(30.0, args.watchdog_s * 3)
        cols = ("A", "B", "C")
        res = {r: {} for r in survivors}
        det = {r: {} for r in survivors}
        for r in survivors:
            for c in cols:
                k = f"fi/{r}/{c}"
                _wait_key(store, k, deadline)
                res[r][c] = store.get(k).decode() if store.check([k]) else "MISSING"
                dk = k + ".detail"
                det[r][c] = store.get(dk).decode() if store.check([dk]) else ""

        print("\n" + "=" * 74, flush=True)
        print(f"  FAULT INJECTION  mode={args.mode}  collective={coll}  "
              f"victim=rank{victim}  timeout={args.timeout_s}s  "
              f"settle={args.teardown_wait_ms}ms", flush=True)
        print("  probes: A=first collective  B=ft_converge  C=post-converge collective",
              flush=True)
        print("=" * 74, flush=True)
        print(f"  {'rank':>5}  {'A':>22}  {'B':>10}  {'C':>10}", flush=True)
        for r in survivors:
            print(f"  {r:>5}  {res[r]['A']:>22}  {res[r]['B']:>10}  {res[r]['C']:>10}",
                  flush=True)

        # details for anything that wasn't a clean OK
        shown = False
        for r in survivors:
            for c in cols:
                if res[r][c] not in ("OK", "SKIP") and det[r][c]:
                    if not shown:
                        print("-" * 74 + "\n  details:", flush=True); shown = True
                    print(f"    rank{r} {c}: {det[r][c]}", flush=True)

        clean = all(res[r][c] == "OK" for r in survivors for c in cols)
        print("-" * 74, flush=True)
        aset = sorted({res[r]['A'] for r in survivors})
        print(f"  probe A outcomes: {aset}", flush=True)
        print(f"  VERDICT: {'CLEAN (all probes OK)' if clean else 'FAULT (see table/details)'}",
              flush=True)
        print("=" * 74 + "\n", flush=True)
        store.set("fi/done", "1")            # release a straggler victim

    try:
        dist.destroy_process_group()
    except Exception:
        pass
    # Propagate this survivor's outcome via the EXIT CODE (0 = all probes OK,
    # 4 = faulted).  main() aggregates exit codes rather than reading the store
    # post-mortem — rank 0 (the store host) has exited by then, so a store read
    # would spuriously return UNKNOWN and break --expect survive gating.
    os._exit(0 if my_clean else 4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nranks", type=int, default=0)
    p.add_argument("--mode", choices=["pre", "during", "straggler"], default="pre")
    p.add_argument("--collective", choices=["allreduce", "allgather"], default="allreduce")
    p.add_argument("--victim", type=int, default=-1, help="rank to kill (default: last)")
    p.add_argument("--kill-iter", type=int, default=3, help="healthy iters before injection")
    p.add_argument("--count", type=int, default=262144, help="elements per rank")
    p.add_argument("--timeout-s", type=float, default=5.0, help="FT kernel timeout")
    p.add_argument("--kill-delay-ms", type=float, default=0.0,
                   help="during mode: ms to wait after launching the collective before "
                        "SIGKILL (0 = mid-kernel; larger samples later in the window)")
    p.add_argument("--teardown-wait-ms", type=float, default=500.0,
                   help="pre mode: settle window after death before probe A")
    p.add_argument("--watchdog-s", type=float, default=30.0,
                   help="host watchdog for a hung probe")
    p.add_argument("--expect", choices=["any", "survive"], default="any",
                   help="'survive' asserts CLEAN (regression gate); 'any' just reports")
    args = p.parse_args()

    if args.victim == 0:
        print("ERROR: victim must be non-zero (rank 0 is the store master).",
              file=sys.stderr)
        sys.exit(2)

    ngpu   = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 2:
        print("ERROR: at least 2 GPUs are required.", file=sys.stderr)
        sys.exit(1)

    store_port = _find_free_port()
    print(f"Fault injection: {nranks} ranks, mode={args.mode}, "
          f"collective={args.collective}, port={store_port}", flush=True)
    ctx   = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_worker, args=(r, nranks, args, store_port),
                         name=f"rank-{r}", daemon=False) for r in range(nranks)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    # Verdict from SURVIVOR exit codes (robust — no post-mortem store read).
    # Survivor codes: 0 = all probes OK, 4 = faulted, 2 = watchdog HANG.
    # The victim is expected to die (SIGKILL -> negative code), so it is excluded.
    victim = args.victim if args.victim >= 0 else nranks - 1
    codes  = [pr.exitcode for pr in procs]
    survivor_codes = {r: codes[r] for r in range(nranks) if r != victim}
    clean = all(c == 0 for c in survivor_codes.values())
    verdict = "CLEAN" if clean else "FAULT"
    print(f"survivor exit codes {survivor_codes}  (victim rank{victim}={codes[victim]})",
          flush=True)

    if args.expect == "survive":
        print(f"[expect=survive] verdict={verdict} -> {'PASS' if clean else 'FAIL'}",
              flush=True)
        sys.exit(0 if clean else 1)
    print(f"[expect=any] verdict={verdict} (characterization; not gating)", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
