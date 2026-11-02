#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: FTProcessGroup (ft_nccl) vs standard NCCL.

Measures AllGather and AllReduce latency and bandwidth across a sweep of
message sizes.  Each backend is benchmarked in a fresh set of processes to
avoid state leakage between runs.

Timing model
------------
Every variant is timed apples-to-apples: the timed region is exactly
{batch x collective + one torch.cuda.synchronize()} with NO per-iteration host
barrier, so ft and nccl measure the same thing (the collective kernel, not an
8-rank host rendezvous).

- nccl: stock NCCL, dispatch to torch.cuda.synchronize() return.
- ft_nccl (default FT variant): FT backend; after the synchronize, completion is
  confirmed with check_and_clear_error() (a pinned-flag read).
- ft_nccl_sync (opt-in, --with-sync): FT backend; completion via the full
  get_result_mask() D2H membership read. Runs in addition to ft_nccl so the
  cost of the mask read can be isolated.

Batch mode (--batch N)
----------------------
Queues N collectives on the stream before calling the single sync point.
Each iteration launches N back-to-back kernels, then one sync.  This models
real training loops where multiple collectives are in flight before results
are inspected.

Reported metrics:
- mean_us     : wall time for the entire batch (N collectives + 1 sync)
- per_coll_us : mean_us / N  (amortized per-collective latency)
- algbw_GBs   : bandwidth based on per_coll_us (single-collective equivalent)
- batch_GBs   : aggregate throughput (N x data_per_collective / mean_us)

Bandwidth definitions
---------------------
- AllGather algbw : (nranks x count x 4 B) / per_coll_us
- AllReduce algbw : (count x 4 B) / per_coll_us
- All-to-all (alltoall / alltoallv) algbw : the true bytes a rank SENDS per
  call / per_coll_us (algorithm bandwidth). alltoall = uniform count to each of
  nRanks dests; alltoallv = asymmetric split sizes. Both a2a modes share this
  convention so their bandwidths are directly comparable.

Usage
-----
  # Compare both backends, all collectives (default):
  python benchmark_ft_collective.py

  # FT only, AllGather, write CSV:
  python benchmark_ft_collective.py --backend ft_nccl --collective allgather --csv out.csv

  # Batch of 8 collectives per sync:
  python benchmark_ft_collective.py --batch 8 --csv batch8.csv

  # Sweep batch sizes 1,2,4,8,16 at a fixed message size:
  python benchmark_ft_collective.py --min-count 65536 --max-count 65536 --steps 1 \
      --batch 1 --csv b1.csv
  python benchmark_ft_collective.py --min-count 65536 --max-count 65536 --steps 1 \
      --batch 16 --csv b16.csv
"""

import argparse
import csv
import datetime
import math
import os
import signal
import socket
import sys
import time
import warnings
import multiprocessing as mp
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
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


def _size_sweep(min_count: int, max_count: int, steps: int,
                pow2: bool = False) -> List[int]:
    """Log-2 spaced element counts from min_count to max_count, deduplicated.

    With ``pow2=True`` the sweep is every exact power of two in
    [min_count, max_count] (``--steps`` is ignored).  This guarantees each
    count is 16-byte aligned, so the FT kernels' 128-bit vectorized copy path
    engages on every point — useful for clean bandwidth comparisons.
    """
    if pow2:
        lo = math.ceil(math.log2(max(min_count, 1)))
        hi = math.floor(math.log2(max_count))
        return [1 << k for k in range(lo, hi + 1)]
    if min_count >= max_count or steps <= 1:
        pts = {min_count, max_count}
        return sorted(pts)
    lo = math.log2(max(min_count, 1))
    hi = math.log2(max_count)
    seen: set = set()
    result: List[int] = []
    for i in range(steps):
        val = int(round(2 ** (lo + (hi - lo) * i / (steps - 1))))
        val = max(min_count, min(val, max_count))
        if val not in seen:
            seen.add(val)
            result.append(val)
    if max_count not in seen:
        result.append(max_count)
    return sorted(result)


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sd = sorted(data)
    n  = len(sd)
    idx = (p / 100.0) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sd[lo] * (1.0 - (idx - lo)) + sd[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Collective helpers (called inside worker process)
# ---------------------------------------------------------------------------

def _alloc(backend: str, pg, nranks: int, count: int, dtype=None,
           collective: str = "allgather", rank: int = 0):
    """Allocate send tensor and output for one size step.

    alltoall: `send` holds nRanks per-destination chunks (nranks*count) and the
    output is the flat receive tensor for all_to_all_single. alltoallv: `count`
    is the base unit; asymmetric per-(src,dst) splits, a fixed max-sized symmetric
    send window (a view of it is passed), and (out_flat, in_splits, out_splits).
    Others: `send` is `count`, output is a list of nRanks views.
    """
    import torch
    if dtype is None:
        dtype = torch.float32
    if collective == "allgatherv":
        # Uniform sizes (each rank contributes `count`) so ft all_gatherv and
        # nccl all_gather_into_tensor move identical data — apples-to-apples.
        if backend == "ft_nccl":
            send = pg.empty(count, dtype=dtype)
        else:
            send = torch.zeros(count, dtype=dtype, device="cuda")
        out = torch.zeros(nranks * count, dtype=dtype, device="cuda")
        return send, out
    if collective == "reduce_scatterv":
        # Uniform shards: ft reduce_scatterv vs nccl reduce_scatter_tensor.
        if backend == "ft_nccl":
            send = pg.empty(nranks * count, dtype=dtype)   # full input
        else:
            send = torch.zeros(nranks * count, dtype=dtype, device="cuda")
        out = torch.zeros(count, dtype=dtype, device="cuda")
        return send, out
    if collective == "alltoallv":
        unit = count
        in_splits  = [_a2av_send_count(rank, d, unit) for d in range(nranks)]
        out_splits = [_a2av_send_count(s, rank, unit) for s in range(nranks)]
        total_send, total_recv = sum(in_splits), sum(out_splits)
        max_total = nranks * unit * 5     # same on every rank -> symmetric window
        if backend == "ft_nccl":
            send_full = pg.empty(max_total, dtype=dtype)
        else:
            send_full = torch.zeros(max_total, dtype=dtype, device="cuda")
        send    = send_full[:total_send]   # view keeps send_full (window) alive
        out_flat = torch.zeros(total_recv, dtype=dtype, device="cuda")
        return send, (out_flat, in_splits, out_splits)
    send_count = nranks * count if collective == "alltoall" else count
    if backend == "ft_nccl":
        # pg.empty() uses ncclMemAlloc — symmetric VA required by FT kernels.
        send = pg.empty(send_count, dtype=dtype)
    else:
        send = torch.zeros(send_count, dtype=dtype, device="cuda")
    # Contiguous output block; per-rank slices are views into one allocation so
    # the FT path can coalesce the scatter into a single strided copy.
    out_buf = torch.zeros(nranks * count, dtype=dtype, device="cuda")
    if collective == "alltoall":
        return send, out_buf                    # flat output for all_to_all_single
    return send, list(out_buf.chunk(nranks))


# Timing is always apples-to-apples: every variant times exactly
# {batch x collective + one torch.cuda.synchronize()} with NO per-iteration host
# barrier in the timed window, so ft and nccl measure the same thing (the
# collective kernel, not an 8-rank host rendezvous).
#
# Variants (reported as the CSV "backend"):
#   nccl          - stock NCCL.
#   ft_nccl       - FT backend, completion checked with check_and_clear_error()
#                   (a pinned-flag read).  The default FT measurement.
#   ft_nccl_sync  - FT backend, completion via the full get_result_mask() D2H
#                   membership read.  Opt-in via --with-sync.


def _a2av_send_count(r: int, d: int, unit: int) -> int:
    """Deterministic ASYMMETRIC per-(src,dst) count (matches test_alltoallv)."""
    return unit * (1 + ((r * 7 + d * 3) % 5))


def _a2a_send_bytes(collective: str, nranks: int, count: int,
                    element_size: int):
    """True bytes a rank sends per call for an all-to-all mode (algorithm
    bandwidth), or None for non-a2a collectives that keep their own convention.

    Computed for the reporting rank (rank 0), including the local (self) chunk,
    so both a2a modes report a consistent send-volume bandwidth:
      alltoall  : uniform `count` to each of nRanks destinations
      alltoallv : asymmetric per-dest split sizes
    """
    if collective == "alltoall":
        return nranks * count * element_size
    if collective == "alltoallv":
        return sum(_a2av_send_count(0, d, count) for d in range(nranks)) * element_size
    return None


def _run_alltoallv(variant: str, pg, send, packed, batch: int) -> None:
    """Time one unit: `batch` variable all_to_all_single calls then one sync.

    `packed` = (out_flat, input_splits, output_splits) from _alloc.  Non-empty
    split sizes -> FTProcessGroup.alltoall_base's variable (all_to_allv) FT path.
    """
    import torch
    import torch.distributed as dist
    out_flat, in_splits, out_splits = packed
    if variant == "nccl":
        for _ in range(batch):
            dist.all_to_all_single(out_flat, send, out_splits, in_splits)
        torch.cuda.synchronize()
        return
    for _ in range(batch):
        dist.all_to_all_single(out_flat, send, out_splits, in_splits, group=pg)
    torch.cuda.synchronize()
    if variant == "ft_nccl_sync":
        pg.get_result_mask()
    else:
        pg.check_and_clear_error()


def _run_allgatherv(variant: str, pg, send, output, batch: int) -> None:
    """Time one unit for variable all-gather (uniform sizes here).

    ft: pg.all_gatherv (packed, sizes discovered). nccl baseline:
    all_gather_into_tensor (uniform). Both gather nRanks*count rows.
    """
    import torch
    import torch.distributed as dist
    if variant == "nccl":
        for _ in range(batch):
            dist.all_gather_into_tensor(output, send)
        torch.cuda.synchronize()
        return
    count = send.numel()
    for _ in range(batch):
        work, _ = pg.all_gatherv(send, output, count, count)
        work.wait()
    torch.cuda.synchronize()
    if variant == "ft_nccl_sync":
        pg.get_result_mask()
    else:
        pg.check_and_clear_error()


def _run_reduce_scatterv(variant: str, pg, send, output, batch: int) -> None:
    """Time one unit for variable reduce-scatter (uniform shards here).

    ft: pg.reduce_scatterv (SUM). nccl baseline: reduce_scatter_tensor (uniform).
    """
    import torch
    import torch.distributed as dist
    if variant == "nccl":
        for _ in range(batch):
            dist.reduce_scatter_tensor(output, send)
        torch.cuda.synchronize()
        return
    counts = [output.numel()] * pg._size
    for _ in range(batch):
        work = pg.reduce_scatterv(send, output, counts)
        work.wait()
    torch.cuda.synchronize()
    if variant == "ft_nccl_sync":
        pg.get_result_mask()
    else:
        pg.check_and_clear_error()


def _run_alltoall(variant: str, pg, send, out, batch: int) -> None:
    """Time one unit: `batch` uniform all_to_all_single calls then one sync.

    `send` holds nRanks per-destination chunks; `out` is the flat receive tensor.
    Uniform (no split sizes) -> takes FTProcessGroup.alltoall_base's FT path.
    """
    import torch
    import torch.distributed as dist
    if variant == "nccl":
        for _ in range(batch):
            dist.all_to_all_single(out, send)
        torch.cuda.synchronize()
        return
    for _ in range(batch):
        dist.all_to_all_single(out, send, group=pg)
    torch.cuda.synchronize()
    if variant == "ft_nccl_sync":
        pg.get_result_mask()
    else:
        pg.check_and_clear_error()


def _run_allgather(variant: str, pg, send, outputs, batch: int) -> None:
    """Time one unit: `batch` AllGather calls then one torch.cuda.synchronize()."""
    import torch
    import torch.distributed as dist
    if variant == "nccl":
        for _ in range(batch):
            dist.all_gather(outputs, send)
        torch.cuda.synchronize()
        return
    for _ in range(batch):
        dist.all_gather(outputs, send, group=pg)
    torch.cuda.synchronize()
    if variant == "ft_nccl_sync":
        pg.get_result_mask()
    else:
        pg.check_and_clear_error()


def _run_allreduce(variant: str, pg, send, batch: int) -> None:
    """Time one unit: `batch` AllReduce calls then one torch.cuda.synchronize().

    Uses pg.allreduce() directly for the FT variants so the dispatch path is
    transparent (no implicit work.wait() from dist.all_reduce()).
    """
    import torch
    import torch.distributed as dist
    if variant == "nccl":
        for _ in range(batch):
            dist.all_reduce(send)
        torch.cuda.synchronize()
        return
    for _ in range(batch):
        work = pg.allreduce([send])
        work.wait()   # GPU-side ordering only
    torch.cuda.synchronize()
    if variant == "ft_nccl_sync":
        pg.get_result_mask()
    else:
        pg.check_and_clear_error()


def _run(variant: str, collective: str, pg, send, outputs, batch: int) -> None:
    if collective == "allgather":
        _run_allgather(variant, pg, send, outputs, batch)
    elif collective == "alltoall":
        _run_alltoall(variant, pg, send, outputs, batch)
    elif collective == "alltoallv":
        _run_alltoallv(variant, pg, send, outputs, batch)
    elif collective == "allgatherv":
        _run_allgatherv(variant, pg, send, outputs, batch)
    elif collective == "reduce_scatterv":
        _run_reduce_scatterv(variant, pg, send, outputs, batch)
    else:
        _run_allreduce(variant, pg, send, batch)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _compute_stats(
    backend: str,
    collective: str,
    nranks: int,
    count: int,
    batch: int,
    latencies_us: List[float],
    element_size: int = 4,
    data_per_coll_override: Optional[int] = None,
) -> Dict:
    mean_us      = sum(latencies_us) / len(latencies_us)
    p50_us       = _percentile(latencies_us, 50)
    p95_us       = _percentile(latencies_us, 95)
    p99_us       = _percentile(latencies_us, 99)
    per_coll_us  = mean_us / batch
    nbytes       = count * element_size

    if data_per_coll_override is not None:
        # Caller supplies the true bytes moved per call (e.g. alltoallv, where
        # the payload is the asymmetric per-peer split sizes — not
        # count * element_size).
        data_per_coll = data_per_coll_override
    elif collective in ("allgather", "allgatherv", "reduce_scatterv"):
        # Gather: nRanks*count rows gathered. Reduce-scatter: the full nRanks*count
        # input is reduced per rank. Both scale the reported volume by nRanks.
        data_per_coll = nranks * nbytes
    else:
        # AllReduce: data processed per rank per call
        data_per_coll = nbytes

    # Single-collective equivalent bandwidth (uses amortized per_coll_us)
    algbw_GBs   = data_per_coll / (per_coll_us * 1e-6) / 1e9
    # Aggregate batch throughput (total data across all N collectives / batch time)
    batch_GBs   = (batch * data_per_coll) / (mean_us * 1e-6) / 1e9

    return {
        "backend":      backend,
        "collective":   collective,
        "nranks":       nranks,
        "batch":        batch,
        "count":        count,
        "bytes":        data_per_coll if data_per_coll_override is not None else nbytes,
        "mean_us":      round(mean_us,     2),
        "p50_us":       round(p50_us,      2),
        "p95_us":       round(p95_us,      2),
        "p99_us":       round(p99_us,      2),
        "per_coll_us":  round(per_coll_us, 2),
        "algbw_GBs":    round(algbw_GBs,   4),
        "batch_GBs":    round(batch_GBs,   4),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_HDR = (
    f"  {'count':>10}  {'bytes':>11}  "
    f"{'batch_us':>10}  {'p50_us':>10}  "
    f"{'p95_us':>10}  {'p99_us':>10}  "
    f"{'per_coll_us':>12}  {'algbw_GBs':>11}  {'batch_GBs':>10}"
)


def _print_section_header(backend: str, collective: str, nranks: int,
                          batch: int) -> None:
    sep = "=" * 92
    print(f"\n{sep}", flush=True)
    print(
        f"  backend={backend}  collective={collective}  "
        f"nranks={nranks}  batch={batch}",
        flush=True,
    )
    print(sep, flush=True)
    print(_HDR, flush=True)
    print("  " + "-" * 90, flush=True)


def _print_data_row(row: Dict) -> None:
    print(
        f"  {row['count']:>10d}  {row['bytes']:>11d}  "
        f"{row['mean_us']:>10.1f}  {row['p50_us']:>10.1f}  "
        f"{row['p95_us']:>10.1f}  {row['p99_us']:>10.1f}  "
        f"{row['per_coll_us']:>12.1f}  {row['algbw_GBs']:>11.3f}  "
        f"{row['batch_GBs']:>10.3f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def _rank_worker(
    rank: int,
    nranks: int,
    backend: str,
    collectives: List[str],
    counts: List[int],
    warmup: int,
    iters: int,
    batch: int,
    timeout_s: int,
    store_port: int,
    result_queue,
    dtype_str: str = "float32",
    with_sync: bool = False,
) -> None:
    """Entry point for each rank subprocess."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The nccl baseline intentionally uses non-symmetric tensors; silence only
    # the expected FT-fallback messages there. The ft_nccl process is NOT
    # filtered, so a genuine fallback (silently measuring NCCL) still warns.
    if backend == "nccl":
        warnings.filterwarnings(
            "ignore", message=r".*(not symmetric|not supported by FT kernels).*")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    import torch
    import torch.distributed as dist

    _DTYPE_MAP = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = _DTYPE_MAP[dtype_str]
    element_size = dtype.itemsize

    torch.cuda.set_device(0)

    store = dist.TCPStore(
        "127.0.0.1", store_port, nranks,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=300),
    )

    # ---- Backend-specific initialisation ----
    if backend == "ft_nccl":
        ft_pkg = os.path.join(_HERE, "ft_collective")
        parent = os.path.dirname(ft_pkg)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from ft_collective import get_ft_process_group
        os.environ["FT_TIMEOUT_US"] = str(timeout_s * 1_000_000)
        # Size the FT scratch to the largest count in the sweep so every point
        # takes the FT path (the default 4M limit would silently fall back to
        # NCCL above 4M). Respect an explicit FT_NCCL_MAX_COUNT if the user set
        # one.  counts is in float32 elements; the pg scratch is float32-sized.
        # alltoallv sends up to 5*count to a single peer (_a2av_send_count).
        if counts:
            per_peer = max(counts) * (5 if "alltoallv" in collectives else 1)
            preset = int(os.environ.get("FT_NCCL_MAX_COUNT", "0") or 0)
            if preset < per_peer:
                # A smaller preset (e.g. exported for another run) would push the
                # largest points onto the NCCL fallback while still labeled ft.
                if preset and rank == 0:
                    print(f"note: raising FT_NCCL_MAX_COUNT {preset} -> {per_peer} "
                          f"so every point in the sweep is FT-eligible", flush=True)
                os.environ["FT_NCCL_MAX_COUNT"] = str(per_peer)

    pg_timeout = datetime.timedelta(seconds=max(timeout_s + 30, 120))
    # Pass device_id so NCCL does not guess from global rank.
    # Each worker pins one GPU as cuda:0 via CUDA_VISIBLE_DEVICES; without
    # device_id NCCL picks device <global_rank> which may not exist.
    try:
        dist.init_process_group(
            backend, store=store, rank=rank, world_size=nranks,
            timeout=pg_timeout, device_id=torch.device("cuda:0"),
        )
    except TypeError:
        dist.init_process_group(
            backend, store=store, rank=rank, world_size=nranks,
            timeout=pg_timeout,
        )

    pg = get_ft_process_group() if backend == "ft_nccl" else None

    rows: List[Dict] = []

    # Reporting variants to run in this process.  The nccl process reports
    # "nccl"; the ft process reports "ft_nccl" (default) and, only with
    # --with-sync, additionally "ft_nccl_sync" (the get_result_mask path).
    if backend == "ft_nccl":
        variants = ["ft_nccl"] + (["ft_nccl_sync"] if with_sync else [])
    else:
        variants = ["nccl"]

    for collective in collectives:
        for variant in variants:
            if rank == 0:
                _print_section_header(variant, collective, nranks, batch)

            for count in counts:
                # Allocate buffers — collective call for ft_nccl (ncclMemAlloc).
                send, outputs = _alloc(backend, pg, nranks, count, dtype,
                                       collective, rank)

                # ---- Warmup ----  (also detects an FT point falling back to NCCL,
                # so a fallback is never reported as an FT measurement)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for _ in range(warmup):
                        _run(variant, collective, pg, send, outputs, batch)
                    torch.cuda.synchronize()
                fell_back = backend == "ft_nccl" and any(
                    "using NCCL (not fault-tolerant)" in str(w.message) for w in caught)
                label = f"{variant}(nccl-fallback)" if fell_back else variant
                if fell_back and rank == 0:
                    print(f"WARNING: {collective} count={count} fell back to NCCL: "
                          f"{caught[0].message}", flush=True)

                # Synchronise all ranks before entering the timed region.  This
                # is OUTSIDE the timed window; the timed region itself carries no
                # host barrier so the measurement is the collective kernel, not
                # an 8-rank rendezvous.  The first barrier also pays the one-time
                # NCCL ring-setup (~300 ms) here rather than in the first sample.
                dist.barrier()
                torch.cuda.synchronize()

                # One extra un-timed run to re-warm GPU clocks and leave the FT
                # stream dependency-free immediately before timing begins.
                _run(variant, collective, pg, send, outputs, batch)
                dist.barrier()
                torch.cuda.synchronize()

                # ---- Timed iterations: batch collectives + 1 sync, no barrier.
                latencies_us: List[float] = []
                for _ in range(iters):
                    t0 = time.perf_counter_ns()
                    _run(variant, collective, pg, send, outputs, batch)
                    t1 = time.perf_counter_ns()
                    latencies_us.append((t1 - t0) / 1_000.0)

                # Synchronise before moving to the next size.
                dist.barrier()

                if rank == 0:
                    # All-to-all bandwidth = the bytes a rank actually SENDS per
                    # call (algorithm bandwidth), not count*element_size. Report
                    # the true send volume so both a2a modes are consistent
                    # with each other. (allgather/allreduce keep their own
                    # established conventions and pass no override.)
                    override = _a2a_send_bytes(
                        collective, nranks, count, element_size)
                    row = _compute_stats(
                        label, collective, nranks, count, batch,
                        latencies_us, element_size, override)
                    _print_data_row(row)
                    rows.append(row)

                # Release buffers.  For ft_nccl, ncclMemFree happens in
                # FTProcessGroup.__del__ when the pg is destroyed.
                del send, outputs

        # End-of-collective barrier.
        dist.barrier()

    dist.destroy_process_group()

    if rank == 0 and rows:
        result_queue.put(rows)


# ---------------------------------------------------------------------------
# Backend runner — spawns one fresh process pool per backend
# ---------------------------------------------------------------------------

def _run_backend(
    backend: str,
    collectives: List[str],
    counts: List[int],
    nranks: int,
    warmup: int,
    iters: int,
    batch: int,
    timeout_s: int,
    dtype_str: str = "float32",
    with_sync: bool = False,
) -> List[Dict]:
    """Spawn nranks workers for one backend and collect results from rank 0."""
    store_port   = _find_free_port()
    ctx          = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = [
        ctx.Process(
            target=_rank_worker,
            args=(
                r, nranks, backend, collectives, counts,
                warmup, iters, batch, timeout_s,
                store_port, result_queue, dtype_str, with_sync,
            ),
            name=f"{backend}-rank-{r}",
            daemon=False,
        )
        for r in range(nranks)
    ]

    for p in procs:
        p.start()

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\nCtrl+C — terminating workers …", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        return []

    failed = [
        (p.name, p.exitcode) for p in procs
        if p.exitcode not in (0, None, -signal.SIGTERM, -signal.SIGKILL)
    ]
    for name, code in failed:
        print(f"  WARN: {name} exited with code {code}", file=sys.stderr)

    return result_queue.get() if not result_queue.empty() else []


# ---------------------------------------------------------------------------
# Comparison summary (printed when both backends are benchmarked)
# ---------------------------------------------------------------------------

def _print_comparison(all_rows: List[Dict]) -> None:
    """Side-by-side latency and overhead table for nccl vs ft_nccl / ft_nccl_sync."""
    # Index by (collective, batch, count) → {backend: row}
    index: Dict = {}
    for row in all_rows:
        key = (row["collective"], row["batch"], row["count"])
        index.setdefault(key, {})[row["backend"]] = row

    pairs = {k: v for k, v in index.items()
             if "nccl" in v and "ft_nccl" in v}
    if not pairs:
        return

    have_sync = any("ft_nccl_sync" in v for v in pairs.values())

    for collective in sorted({c for c, _, _ in pairs}):
        for batch in sorted({b for c, b, _ in pairs if c == collective}):
            if have_sync:
                sep = "=" * 100
                print(f"\n{sep}", flush=True)
                print(
                    f"  Comparison: {collective}  batch={batch}  —  "
                    f"nccl vs ft_nccl vs ft_nccl_sync  (+overhead = ft slower)",
                    flush=True,
                )
                print(sep, flush=True)
                print(
                    f"  {'count':>10}  {'nccl_us':>10}  {'ft_us':>10}  {'ft_sync_us':>10}"
                    f"  {'Δft_us':>9}  {'Δft_%':>8}"
                    f"  {'Δsync_us':>9}  {'Δsync_%':>8}"
                    f"  {'nccl_GBs':>10}  {'ft_GBs':>10}  {'ft_sync_GBs':>11}",
                    flush=True,
                )
                print("  " + "-" * 98, flush=True)
            else:
                sep = "=" * 78
                print(f"\n{sep}", flush=True)
                print(
                    f"  Comparison: {collective}  batch={batch}  —  "
                    f"nccl vs ft_nccl  (+overhead = ft slower)",
                    flush=True,
                )
                print(sep, flush=True)
                print(
                    f"  {'count':>10}  {'nccl_coll_us':>13}  {'ft_coll_us':>11}  "
                    f"{'delta_us':>10}  {'delta_%':>10}  "
                    f"{'nccl_GBs':>10}  {'ft_GBs':>10}",
                    flush=True,
                )
                print("  " + "-" * 76, flush=True)

            for count in sorted(n for c, b, n in pairs
                                 if c == collective and b == batch):
                d    = pairs[(collective, batch, count)]
                nccl = d["nccl"]
                ft   = d["ft_nccl"]
                base = nccl["per_coll_us"]
                # Compare amortized per-collective latency for a fair comparison.
                dft_us  = ft["per_coll_us"] - base
                dft_pct = 100.0 * dft_us / base if base else 0.0
                sft = "+" if dft_us >= 0 else ""

                if have_sync and "ft_nccl_sync" in d:
                    syn = d["ft_nccl_sync"]
                    dsy_us  = syn["per_coll_us"] - base
                    dsy_pct = 100.0 * dsy_us / base if base else 0.0
                    ssy = "+" if dsy_us >= 0 else ""
                    print(
                        f"  {count:>10d}  {base:>10.1f}  {ft['per_coll_us']:>10.1f}"
                        f"  {syn['per_coll_us']:>10.1f}"
                        f"  {sft}{dft_us:>8.1f}  {sft}{dft_pct:>7.1f}%"
                        f"  {ssy}{dsy_us:>8.1f}  {ssy}{dsy_pct:>7.1f}%"
                        f"  {nccl['algbw_GBs']:>10.3f}  {ft['algbw_GBs']:>10.3f}"
                        f"  {syn['algbw_GBs']:>11.3f}",
                        flush=True,
                    )
                else:
                    sign = "+" if dft_us >= 0 else ""
                    print(
                        f"  {count:>10d}  {base:>13.1f}  "
                        f"{ft['per_coll_us']:>11.1f}  "
                        f"  {sign}{dft_us:>8.1f}  {sign}{dft_pct:>9.1f}%  "
                        f"{nccl['algbw_GBs']:>10.3f}  {ft['algbw_GBs']:>10.3f}",
                        flush=True,
                    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=["nccl", "ft_nccl", "both"], default="both",
        help="Backend(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--collective", choices=["allgather", "allreduce", "alltoall", "alltoallv",
                                 "allgatherv", "reduce_scatterv",
                                 "both", "all"], default="both",
        help="Collective(s) to benchmark. 'both' = allgather+allreduce (default); "
             "'all' = every collective.",
    )
    parser.add_argument(
        "--nranks", type=int, default=0,
        help="Number of ranks / GPUs (0 = all visible GPUs, up to 8; default: 0)",
    )
    parser.add_argument(
        "--min-count", type=int, default=64,
        help=(
            "Minimum float32 elements per rank (default: 64 = 256 B).  "
            "The FT kernel has fixed per-call overhead (readywin barrier, "
            "stream events, memset, scatter memcpy) that dominates below "
            "~64 elements, making the comparison with NCCL misleading."
        ),
    )
    parser.add_argument(
        "--max-count", type=int, default=1_048_576,
        help="Maximum float32 elements per rank (default: 1 048 576 = 4 MiB)",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of log-2 size steps in sweep (default: 20)",
    )
    parser.add_argument(
        "--pow2", action="store_true",
        help="Sweep exact powers of two in [min-count, max-count] (ignores "
             "--steps); guarantees every count is 16-byte aligned for the "
             "vectorized FT copy path",
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Warmup iterations per (backend, collective, size) (default: 20)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Timed iterations per (backend, collective, size) (default: 100)",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help=(
            "Number of collectives to queue before each sync (default: 1).  "
            "Batch > 1 measures amortized per-collective latency when N kernels "
            "are pipelined on the stream before a single host sync."
        ),
    )
    parser.add_argument(
        "--timeout-s", type=int, default=30,
        help="FT collective timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
        help=(
            "Element dtype for send/recv tensors (default: float32).  "
            "All three dtypes use the FT kernel for both AllGather (byte-level "
            "copy) and AllReduce (reduction accumulates in float32 internally)."
        ),
    )
    parser.add_argument(
        "--csv", type=str, default="",
        help="Write all results to this CSV file (optional)",
    )
    parser.add_argument(
        "--with-sync", action="store_true",
        help="Also benchmark the ft_nccl_sync variant (FT backend with the full "
             "get_result_mask() D2H membership read) alongside the default "
             "ft_nccl (pinned-flag completion check). Off by default.",
    )
    args = parser.parse_args()

    ngpu   = _count_gpus()
    nranks = args.nranks if args.nranks > 0 else min(ngpu, 8)
    if nranks < 2:
        print(
            "ERROR: at least 2 GPUs are required.\n"
            "       Use --nranks or set CUDA_VISIBLE_DEVICES.",
            file=sys.stderr,
        )
        sys.exit(1)

    backends    = ["nccl", "ft_nccl"] if args.backend    == "both" else [args.backend]
    if args.collective == "both":
        collectives = ["allgather", "allreduce"]
    elif args.collective == "all":
        collectives = ["allgather", "allreduce", "alltoall", "alltoallv",
                       "allgatherv", "reduce_scatterv"]
    else:
        collectives = [args.collective]
    counts      = _size_sweep(args.min_count, args.max_count, args.steps, args.pow2)

    if args.batch < 1:
        print("ERROR: --batch must be >= 1.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Benchmark  nranks={nranks}  batch={args.batch}  dtype={args.dtype}  "
        f"sizes={args.min_count}..{args.max_count} ({len(counts)} points)  "
        f"warmup={args.warmup}  iters={args.iters}",
        flush=True,
    )

    all_rows: List[Dict] = []

    for backend in backends:
        rows = _run_backend(
            backend, collectives, counts, nranks,
            args.warmup, args.iters, args.batch, args.timeout_s,
            args.dtype, args.with_sync,
        )
        all_rows.extend(rows)

    # Side-by-side comparison when both backends were measured.
    if len(backends) == 2 and all_rows:
        _print_comparison(all_rows)

    if args.csv and all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults written to: {args.csv}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
