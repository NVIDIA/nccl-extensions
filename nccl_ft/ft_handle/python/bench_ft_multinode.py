#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-node FT collective benchmark — latency + algorithmic bandwidth.

Runs under SLURM/srun (or torchrun) via ft_launch; falls back to single-host
mp.spawn with no launcher env. Built for GB200 NVL where the NVLink domain (and
the LSA collectives) spans many hosts of ~4 GPUs, so 32 GPUs = 8 nodes x 4.

Both backends are measured through the SAME process group:
  - ft_nccl : symmetric tensors from pg.empty()  -> the FT LSA kernels
  - nccl    : regular tensors via dist.*(group=pg) -> ProcessGroupNCCL fallback
Per-rank per-iteration wall-clock (with a device sync) gives the latency
distribution; timings are MAX-reduced across ranks (collective time = slowest
rank). algbw = message_bytes / p50.

Writes a CSV in the same schema as benchmark_ft_collective.py.

SLURM:
  srun --nodes=8 --ntasks-per-node=4 --gpus-per-task=1 \
       python3 bench_ft_multinode.py --collective both --backend both \
       --min-count 64 --max-count 4194304 --pow2 --iters 100 \
       --csv data/scale_n32.csv
Single host:
  python3 bench_ft_multinode.py --nranks 4 --collective both --backend both
"""

import argparse
import contextlib
import csv
import os
import statistics
import sys
import time
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))


def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _sizes(min_count: int, max_count: int, pow2: bool, steps: int):
    if pow2:
        out, c = [], max(1, min_count)
        while c <= max_count:
            out.append(c); c *= 2
        if out and out[-1] != max_count:
            out.append(max_count)
        return out
    # geometric steps
    if steps < 2:
        return [max_count]
    r = (max_count / min_count) ** (1.0 / (steps - 1))
    return sorted({int(round(min_count * (r ** i))) for i in range(steps)})


def _rank_worker(rank: int, world: int, local_rank: int,
                 master_addr: str, master_port: int, args) -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["FT_TIMEOUT_US"] = str(int(args.timeout_s * 1_000_000))
    os.environ.setdefault("FT_NCCL_MAX_COUNT", str(args.max_count))

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import ft_launch
    store = ft_launch.make_store(rank, world, master_addr, master_port, timeout_s=300)
    from ft_collective import get_ft_process_group   # registers "ft_nccl"
    try:
        dist.init_process_group("ft_nccl", store=store, rank=rank,
                                world_size=world, device_id=dev)
    except TypeError:
        dist.init_process_group("ft_nccl", store=store, rank=rank, world_size=world)
    pg = get_ft_process_group()

    backends   = ["nccl", "ft_nccl"] if args.backend == "both" else [args.backend]
    collectives = (["allgather", "allreduce"] if args.collective == "both"
                   else [args.collective])
    sizes = _sizes(args.min_count, args.max_count, args.pow2, args.steps)
    esize = 4  # float32

    # Pre-allocate max-size buffers once and slice per size (avoid repeated
    # collective window registration on the FT path).
    ft_send = pg.empty(args.max_count)
    nc_send = torch.ones(args.max_count, device=dev)
    ft_send.fill_(1.0)
    # AllGather uses the LIST-form dist.all_gather with a contiguous output chunked
    # into per-rank views. On the FT backend a symmetric input then takes the packed
    # fast path (one padded->packed strided copy) — matching benchmark_ft_collective.py.
    # List-form dist.all_gather is the verified FT fast path; tensor-form dispatch
    # goes through torch's _allgather_base. The output need not be symmetric, so one
    # regular buffer serves both backends; the nccl backend's non-symmetric input
    # falls back to NCCL cleanly (uniform sizes never hit the variable-allgather
    # raise path).
    ag_out = (torch.zeros(args.max_count * world, device=dev)
              if "allgather" in collectives else None)

    rows = []
    for coll in collectives:
        for backend in backends:
            for count in sizes:
                send = (ft_send if backend == "ft_nccl" else nc_send)[:count]
                if coll == "allreduce":
                    if backend == "ft_nccl":
                        run = lambda s=send: pg.allreduce([s]).wait()
                    else:
                        run = lambda s=send: dist.all_reduce(s, group=pg)
                else:  # allgather: list-form -> FT packed fast path / NCCL fallback
                    outs = list(ag_out[:count * world].chunk(world))
                    run = lambda o=outs, s=send: dist.all_gather(o, s, group=pg)

                # The nccl backend intentionally uses non-symmetric tensors so they
                # fall back to NCCL — silence that expected per-call warning here. The
                # ft_nccl backend is NOT suppressed, so a genuine fallback (a symmetric
                # buffer not recognized -> silently measuring NCCL) would still warn.
                supp = (warnings.catch_warnings() if backend == "nccl"
                        else contextlib.nullcontext())
                with supp:
                    if backend == "nccl":
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*(not symmetric|not supported by FT kernels).*")
                    for _ in range(args.warmup):
                        run()
                    torch.cuda.synchronize()
                    dts = []
                    for _ in range(args.iters):
                        t0 = time.perf_counter()
                        run()
                        torch.cuda.synchronize()
                        dts.append((time.perf_counter() - t0) * 1e6)

                dts.sort()
                mean = statistics.fmean(dts)
                p50  = dts[len(dts) // 2]
                p95  = dts[min(len(dts) - 1, int(len(dts) * 0.95))]
                p99  = dts[min(len(dts) - 1, int(len(dts) * 0.99))]

                # collective time = slowest rank -> MAX-reduce. FT kernels are SUM-only,
                # so this stats reduce intentionally uses NCCL; silence just this call
                # (a real fallback in the TIMED collective above would still warn).
                t = torch.tensor([mean, p50, p95, p99], device=dev)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pg)
                torch.cuda.synchronize()
                mean, p50, p95, p99 = (float(x) for x in t.tolist())

                nbytes = count * esize                    # per-rank input (CSV 'bytes')
                # algbw message size: allreduce = count*elem; allgather = the full
                # gathered output = count*world*elem (nccl-tests convention; matches
                # the original benchmark so ft-vs-nccl and old-vs-new are comparable).
                msg   = nbytes * world if coll == "allgather" else nbytes
                algbw = msg / p50 / 1000.0 if p50 > 0 else 0.0   # GB/s from p50
                if rank == 0:
                    rows.append([backend, coll, world, 1, count, nbytes,
                                 f"{mean:.3f}", f"{p50:.3f}", f"{p95:.3f}", f"{p99:.3f}",
                                 f"{mean:.3f}", f"{algbw:.4f}", f"{algbw:.4f}"])
                    print(f"  {backend:8s} {coll:10s} n={world} count={count:>9} "
                          f"p50={p50:8.2f}us algbw={algbw:8.2f}GB/s", flush=True)
                try:
                    dist.barrier(group=pg)
                except Exception:
                    pass

    if rank == 0 and args.csv:
        header = ["backend", "collective", "nranks", "batch", "count", "bytes",
                  "mean_us", "p50_us", "p95_us", "p99_us", "per_coll_us",
                  "algbw_GBs", "batch_GBs"]
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}", flush=True)

    try:
        dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nranks", type=int, default=0, help="single-host only (else from launcher)")
    p.add_argument("--backend", choices=["ft_nccl", "nccl", "both"], default="both")
    p.add_argument("--collective", choices=["allgather", "allreduce", "both"], default="both")
    p.add_argument("--min-count", type=int, default=64)
    p.add_argument("--max-count", type=int, default=4194304)
    p.add_argument("--pow2", action="store_true", help="powers of 2 (else --steps geometric)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--csv", default="")
    args = p.parse_args()

    import ft_launch
    ngpu = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else ft_launch.world_size_hint(min(ngpu, 8))
    if nranks < 2:
        print("ERROR: at least 2 GPUs are required.", file=sys.stderr)
        sys.exit(1)
    if not ft_launch.launched():
        print(f"multi-node bench: {nranks} ranks (single host)", flush=True)
    codes = ft_launch.launch(_rank_worker, nranks, args)
    if codes is not None:
        sys.exit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
