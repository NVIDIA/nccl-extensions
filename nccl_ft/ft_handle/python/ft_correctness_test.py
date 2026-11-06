#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical correctness of every FT collective (full active set — no failures).

Consolidates the former per-collective tests into one launched sweep. For each
(collective, dtype, count, mode) case a symmetric input is filled with a known,
rank-dependent pattern and the FT result is checked against the closed-form
expected value. Fault behavior is covered separately (test_ft_converge.py,
test_fault_injection.py, test_ft_rejoin.py); here every rank is present.

Coverage
--------
- Collectives: allreduce (SUM), allgather (list-form packed path),
  reduce_scatter (SUM), allgatherv (uniform), reduce_scatterv (uniform).
- Dtypes: float32, float16, bfloat16.
- Sizes: a sweep chosen to straddle BOTH kernel-path thresholds
  (FT_MULTICTA_THRESHOLD_BYTES, FT_REDUCE_SCATTER_THRESHOLD_BYTES) plus a
  non-16B-aligned count for the tail path — so single-CTA, multi-CTA, and
  reduce-scatter kernels are all exercised.
- Modes: eager and CUDA-graph-captured (replayed).

PASS iff every case matches within a dtype-appropriate tolerance.

Usage
-----
  python ft_correctness_test.py --nranks 8
  python ft_correctness_test.py --nranks 8 --no-graph        # eager only
  python ft_correctness_test.py --nranks 8 --collective allreduce
"""

import argparse
import os
import signal
import sys
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))

# Sizes (elements) straddling the default thresholds: 256 KiB multi-CTA (=65536
# f32), 2 MiB reduce-scatter (=524288 f32). 1006 is the tail case: 4024 B (f32) /
# 2012 B (f16/bf16) — not a multiple of 16 B, but still divisible by 4 elements.
SIZES = [1024, 65536, 262144, 524288, 2097152, 4194304, 1006]


def _tol(dtype):
    import torch
    return {torch.float32: (1e-4, 1e-4),
            torch.float16: (5e-2, 5e-2),
            torch.bfloat16: (8e-2, 8e-2)}[dtype]


def _rank_worker(rank, world, local_rank, master_addr, master_port, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("FT_NCCL_MAX_COUNT", str(max(SIZES)))

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import ft_launch
    store = ft_launch.make_store(rank, world, master_addr, master_port)
    from ft_collective import get_ft_process_group   # registers "ft_nccl"
    try:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=world, device_id=dev)
    except TypeError:
        dist.init_process_group("ft_nccl", store=store, rank=rank, world_size=world)
    pg = get_ft_process_group()
    n = world

    dtypes = ([getattr(torch, args.dtype)] if args.dtype != "all"
              else [torch.float32, torch.float16, torch.bfloat16])
    colls = ([args.collective] if args.collective != "all"
             else ["allreduce", "allgather", "reduce_scatter", "allgatherv",
                   "reduce_scatterv"])
    modes = ["eager"] + ([] if args.no_graph else ["graph"])

    # pg.empty() is a collective alloc + window registration that lives until
    # teardown, so allocate ONE symmetric buffer per dtype (sized for the largest
    # RS/RSV input) and hand out prefix views, which stay FT-eligible.
    sym = {dt: pg.empty(max(SIZES) * n, dtype=dt) for dt in dtypes}

    # ---- per-collective case: returns (ok, detail) ----
    def run_case(coll, dtype, count, mode):
        big = sym[dtype]
        rtol, atol = _tol(dtype)

        def check(got, expected):
            exp = (expected if torch.is_tensor(expected)
                   else torch.full_like(got, float(expected)))
            return bool(torch.allclose(got.float(), exp.float(), rtol=rtol, atol=atol))

        # Build the op as a closure that (re)fills inputs and runs the collective,
        # writing the result into `res` (so graph capture/replay is self-contained).
        if coll == "allreduce":
            x = big[:count]
            res = x
            def op():
                x.fill_(float(rank + 1))
                pg.allreduce([x]).wait()
            expected = n * (n + 1) / 2                       # sum_r (r+1)

        elif coll == "allgather":
            send = big[:count]
            out = torch.empty(count * n, dtype=dtype, device=dev)
            res = out
            def op():
                send.fill_(float(rank))
                dist.all_gather(list(out.chunk(n)), send, group=pg)
            # slot r holds rank r's contribution == r
            expected = torch.repeat_interleave(
                torch.arange(n, device=dev, dtype=torch.float32), count)

        elif coll == "reduce_scatter":
            inp = big[:count * n]
            out = torch.empty(count, dtype=dtype, device=dev)
            res = out
            def op():
                inp.fill_(float(rank + 1))
                # dist.reduce_scatter_tensor reaches the same FT code through
                # _reduce_scatter_base; calling the group directly keeps the
                # test independent of torch's dispatch.
                pg.reduce_scatter_tensor(out, inp).wait()
            expected = n * (n + 1) / 2                       # sum_r (r+1)

        elif coll == "allgatherv":                          # uniform shares
            send = big[:count]
            out = torch.empty(count * n, dtype=dtype, device=dev)
            res = out
            def op():
                send.fill_(float(rank))
                pg.all_gatherv(send, out, count, count)[0].wait()
            expected = torch.repeat_interleave(
                torch.arange(n, device=dev, dtype=torch.float32), count)

        elif coll == "reduce_scatterv":                     # uniform shards
            inp = big[:count * n]
            out = torch.empty(count, dtype=dtype, device=dev)
            res = out
            def op():
                inp.fill_(float(rank + 1))
                pg.reduce_scatterv(inp, out, [count] * n).wait()
            expected = n * (n + 1) / 2
        else:
            return True, "skip"

        try:
            if mode == "eager":
                op()
                torch.cuda.synchronize()
            else:                                           # CUDA-graph capture+replay
                op(); torch.cuda.synchronize()              # warm up allocations
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    op()                                    # input fills are captured too
                # Poison the result so a no-op replay can't pass on the warm-up value.
                res.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
            ok = check(res, expected)
            return ok, ("ok" if ok else "MISMATCH")
        except Exception as e:                              # noqa: BLE001
            # First line only: CUDA errors carry a long multi-line hint.
            return False, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"

    # ---- run the matrix ----
    results = []           # (label, ok, detail)
    for coll in colls:
        for dtype in dtypes:
            for count in SIZES:
                for mode in modes:
                    ok, detail = run_case(coll, dtype, count, mode)
                    label = f"{coll}/{str(dtype).split('.')[-1]}/{count}/{mode}"
                    results.append((label, ok, detail))

    n_fail = sum(0 if ok else 1 for _, ok, _ in results)
    if n_fail:     # written before cx/{rank} so rank 0 never sees a summary without it
        fails = "\n".join(f"{lb}: {d}" for lb, ok, d in results if not ok)
        store.set(f"cxf/{rank}", fails)
    store.set(f"cx/{rank}", f"{len(results) - n_fail}/{len(results)}")

    if rank == 0:
        print("\n" + "=" * 72, flush=True)
        print(f"  FT CORRECTNESS  nranks={n}  cases/rank={len(results)}", flush=True)
        print("=" * 72, flush=True)
        import time
        deadline = time.monotonic() + 60
        all_ok = True
        first_fails = ""
        for r in range(n):
            k = f"cx/{r}"
            while not store.check([k]) and time.monotonic() < deadline:
                time.sleep(0.02)
            summary = store.get(k).decode() if store.check([k]) else "MISSING"
            fk = f"cxf/{r}"
            fails = store.get(fk).decode() if store.check([fk]) else ""
            ok = summary != "MISSING" and not fails
            all_ok = all_ok and ok
            tag = "ok " if ok else "FAIL"
            print(f"  rank {r}: {summary} passed  [{tag}]", flush=True)
            if fails and (r == 0 or fails != first_fails):
                # Print the full list for rank 0, and for any rank that differs.
                for line in fails.split("\n"):
                    print(f"    {line}", flush=True)
            if r == 0:
                first_fails = fails
        print("-" * 72, flush=True)
        print(f"  VERDICT: {'PASS' if all_ok else 'FAIL'}", flush=True)
        print("=" * 72 + "\n", flush=True)

    try:
        dist.barrier(group=pg)
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(0 if n_fail == 0 else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nranks", type=int, default=0)
    p.add_argument("--collective", default="all",
                   choices=["all", "allreduce", "allgather", "reduce_scatter",
                            "allgatherv", "reduce_scatterv"])
    p.add_argument("--dtype", default="all",
                   choices=["all", "float32", "float16", "bfloat16"])
    p.add_argument("--no-graph", action="store_true", help="eager only (skip CUDA graph)")
    args = p.parse_args()

    import ft_launch
    try:
        import torch
        ngpu = torch.cuda.device_count()
    except Exception:
        ngpu = 0
    nranks = args.nranks if args.nranks > 0 else ft_launch.world_size_hint(min(ngpu, 8))
    if nranks < 2:
        print("ERROR: at least 2 GPUs are required.", file=sys.stderr)
        sys.exit(1)
    if not ft_launch.launched():
        print(f"ft correctness: {nranks} ranks (single host)", flush=True)
    codes = ft_launch.launch(_rank_worker, nranks, args)
    if codes is not None:
        ok = all(c == 0 for c in codes)
        print(f"exit codes {codes} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
