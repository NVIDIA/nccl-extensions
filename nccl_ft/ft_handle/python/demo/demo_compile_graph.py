#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch.compile and CUDA Graph Demo for FT Collectives

Demonstrates that the FT collectives — AllGather, AllReduce, AllToAllv,
AllGatherv, and ReduceScatterv — can be used inside:
  1. torch.compile() — the custom ops appear as opaque nodes; no graph break.
  2. CUDA graphs     — the ops are captured and replayed without CPU involvement.

Both modes require:
  - pg.set_active() before entering the compiled/captured region.
  - pg.ft_converge()   outside the compiled/captured region (it contains a CPU
                    barrier which is incompatible with both modes).
  - Warmup before CUDA graph capture (triggers lazy devComm init).
  - DEVICE-resident send/recv counts, so one captured graph serves any routing
    (the kernel reads the counts at replay).

The process exits non-zero if any rank fails or a correctness check fails.

Usage
-----
  python demo_compile_graph.py --nranks 2
  python demo_compile_graph.py --nranks 4 --count 1024 --iters 20

Requirements
------------
  - libft_collective.so built and on LD_LIBRARY_PATH (or FT_COLLECTIVE_LIB)
  - ft_collective Python package importable (installed, or in the parent
    python/ directory, which this script adds to sys.path)
  - At least --nranks GPUs
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


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def _rank_worker(rank: int, nranks: int, args, store_port: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    # Quiet teardown: stop the NCCL watchdog/heartbeat monitor from polling the
    # TCPStore during shutdown, which otherwise logs "recvValue failed / should
    # dump" once rank 0's store server exits before the other ranks. Set before
    # torch is imported / the PG is created.
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)

    # ------------------------------------------------------------------ #
    #  Init process group                                                  #
    # ------------------------------------------------------------------ #
    store = dist.TCPStore(
        "127.0.0.1", store_port, nranks,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=120),
    )

    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    from ft_collective import get_ft_process_group

    try:
        dist.init_process_group(
            "ft_nccl", store=store, rank=rank, world_size=nranks,
            device_id=torch.device("cuda:0"),
        )
    except TypeError:
        dist.init_process_group(
            "ft_nccl", store=store, rank=rank, world_size=nranks,
        )

    pg = get_ft_process_group()

    # Set this pg as the active one for the custom ops.
    pg.set_active()

    count = args.count

    # Symmetric tensors for FT ops (separate send buffers for AllGather and
    # AllReduce, since AllReduce is in-place).
    send_ag = pg.empty(count, dtype=torch.float32)
    send_ar = pg.empty(count, dtype=torch.float32)

    # Output tensors (non-symmetric is fine for allgather output).
    output_ag = torch.zeros(nranks, count, dtype=torch.float32, device="cuda")

    # ---- Variable all-to-allv (auto-recv-sizing) buffers ----
    # Uniform pattern for an easy check: every rank sends `per` tokens to each
    # destination; rank r's payload is the value r, so receiver slot s == s.
    per = max(1, count // nranks)
    send_av = pg.empty(nranks * per, dtype=torch.float32)       # symmetric send
    out_av  = torch.zeros(nranks * per, dtype=torch.float32, device="cuda")
    sc_av   = torch.full((nranks,), per, dtype=torch.int32, device="cuda")
    rc_av   = torch.zeros(nranks, dtype=torch.int32, device="cuda")

    def _fill_av():
        send_av.fill_(float(rank))          # rank r sends value r to every dest

    def _check_av() -> bool:
        rc = rc_av.cpu().tolist()
        ok = all(rc[s] == per for s in range(nranks))
        for s in range(nranks):
            seg = out_av[s * per:(s + 1) * per]
            ok = ok and bool(torch.all(seg == float(s)).item())
        return ok

    # ---- Variable all-gather (uniform sizes for an easy check) ----
    # Every rank contributes `per` rows of value r -> packed slot s == value s.
    send_gv = pg.empty(per, dtype=torch.float32)
    out_gv  = torch.zeros(nranks * per, dtype=torch.float32, device="cuda")
    sc_gv   = torch.full((nranks,), per, dtype=torch.int32, device="cuda")
    rc_gv   = torch.zeros(nranks, dtype=torch.int32, device="cuda")

    # ---- Variable reduce-scatter (uniform shards) ----
    # rank c contributes value (c+1) everywhere -> each output element == sum(1..n).
    send_rs = pg.empty(nranks * per, dtype=torch.float32)          # full input
    out_rs  = torch.zeros(per, dtype=torch.float32, device="cuda")
    cnt_rs  = torch.full((nranks,), per, dtype=torch.int32, device="cuda")
    rs_stride = ((per * 4 + 15) // 16) * 16
    rs_sum  = float(nranks * (nranks + 1) // 2)

    def _fill_gv():
        send_gv.fill_(float(rank))

    def _fill_rs():
        send_rs.fill_(float(rank + 1))

    def _check_gv() -> bool:
        rc = rc_gv.cpu().tolist()
        ok = all(rc[s] == per for s in range(nranks))
        for s in range(nranks):
            seg = out_gv[s * per:(s + 1) * per]
            ok = ok and bool(torch.all(seg == float(s)).item())
        return ok

    def _check_rs() -> bool:
        return bool(torch.all(out_rs == rs_sum).item())

    failures = []                          # correctness checks that failed (rank 0)

    def _verdict(ok: bool) -> str:
        if not ok:
            failures.append(1)
        return "PASS" if ok else "FAIL"

    if rank == 0:
        print(f"\n[rank 0] FT ProcessGroup initialized. "
              f"count={count}, nranks={nranks}, a2av per={per}\n", flush=True)

    # ------------------------------------------------------------------ #
    #  1. torch.compile demo                                               #
    # ------------------------------------------------------------------ #

    if rank == 0:
        print("=" * 60, flush=True)
        print(" torch.compile demo", flush=True)
        print("=" * 60, flush=True)

    @torch.compile(fullgraph=True)
    def compiled_allgather(send: torch.Tensor, output: torch.Tensor) -> None:
        torch.ops.ft_collective.allgather(send, output)

    @torch.compile(fullgraph=True)
    def compiled_allreduce(tensor: torch.Tensor) -> None:
        torch.ops.ft_collective.allreduce(tensor)

    # Warmup / compilation pass (also exercises correctness).
    for it in range(3):
        send_ag.fill_(float(rank + 1))
        send_ar.fill_(float(rank + 1))
        output_ag.zero_()

        pg.ft_converge()

        compiled_allgather(send_ag, output_ag)
        pg._cuda.stream_synchronize(pg._stream)

        pg.ft_converge()

        compiled_allreduce(send_ar)
        pg._cuda.stream_synchronize(pg._stream)

    # Verify correctness on rank 0.
    if rank == 0:
        expected_ag = torch.tensor(
            [[float(r + 1)] * count for r in range(nranks)],
            dtype=torch.float32, device="cuda"
        )
        ag_ok = torch.allclose(output_ag, expected_ag)
        expected_sum = float(sum(range(1, nranks + 1)))
        ar_ok = torch.allclose(send_ar, torch.full((count,), expected_sum, device="cuda"))
        print(f"  AllGather correctness: {_verdict(ag_ok)}", flush=True)
        print(f"  AllReduce correctness: {_verdict(ar_ok)}", flush=True)

    # Timed loop.
    iters = args.iters
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        send_ag.fill_(float(rank + 1))
        pg.ft_converge()
        compiled_allgather(send_ag, output_ag)
        pg._cuda.stream_synchronize(pg._stream)
    t1 = time.perf_counter()

    if rank == 0:
        ms = (t1 - t0) / iters * 1e3
        print(f"  torch.compile AllGather: {ms:.3f} ms/iter ({iters} iters)", flush=True)

    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        send_ar.fill_(float(rank + 1))
        pg.ft_converge()
        compiled_allreduce(send_ar)
        pg._cuda.stream_synchronize(pg._stream)
    t1 = time.perf_counter()

    if rank == 0:
        ms = (t1 - t0) / iters * 1e3
        print(f"  torch.compile AllReduce: {ms:.3f} ms/iter ({iters} iters)\n", flush=True)

    # ---- torch.compile all-to-allv (auto-recv-sizing) ----
    @torch.compile(fullgraph=True)
    def compiled_alltoallv(send, output, sc, rc) -> None:
        torch.ops.ft_collective.alltoallv(send, output, sc, rc, per, 1)

    @torch.compile(fullgraph=True)
    def compiled_allgatherv(send, output, sc, rc) -> None:
        torch.ops.ft_collective.allgatherv(send, output, sc, rc, per, 1)

    @torch.compile(fullgraph=True)
    def compiled_reduce_scatterv(send, output, cnt) -> None:
        torch.ops.ft_collective.reduce_scatterv(send, output, cnt, rs_stride, per)

    for _ in range(3):
        _fill_av(); out_av.zero_(); rc_av.zero_()
        pg.ft_converge()
        compiled_alltoallv(send_av, out_av, sc_av, rc_av)
        pg._cuda.stream_synchronize(pg._stream)
        _fill_gv(); out_gv.zero_(); rc_gv.zero_()
        pg.ft_converge()
        compiled_allgatherv(send_gv, out_gv, sc_gv, rc_gv)
        pg._cuda.stream_synchronize(pg._stream)
        _fill_rs(); out_rs.zero_()
        pg.ft_converge()
        compiled_reduce_scatterv(send_rs, out_rs, cnt_rs)
        pg._cuda.stream_synchronize(pg._stream)

    if rank == 0:
        print(f"  torch.compile AllToAllv        correctness: "
              f"{_verdict(_check_av())}", flush=True)
        print(f"  torch.compile AllGatherv       correctness: "
              f"{_verdict(_check_gv())}", flush=True)
        print(f"  torch.compile ReduceScatterv   correctness: "
              f"{_verdict(_check_rs())}", flush=True)
        print("", flush=True)

    # ------------------------------------------------------------------ #
    #  2. CUDA graph demo                                                  #
    # ------------------------------------------------------------------ #

    if rank == 0:
        print("=" * 60, flush=True)
        print(" CUDA graph demo", flush=True)
        print("=" * 60, flush=True)

    # FT ops launch on pg._stream.  We must capture on the same stream.
    ft_stream = torch.cuda.ExternalStream(pg._stream)

    # Warmup: __init__ already ran one allgather, but run a couple more to
    # ensure all lazy initialization is complete before capture.
    for _ in range(2):
        send_ag.fill_(1.0)
        send_ar.fill_(1.0)
        output_ag.zero_()
        pg.ft_converge()
        torch.ops.ft_collective.allgather(send_ag, output_ag)
        pg._cuda.stream_synchronize(pg._stream)
        pg.ft_converge()
        torch.ops.ft_collective.allreduce(send_ar)
        pg._cuda.stream_synchronize(pg._stream)
        _fill_av(); out_av.zero_(); rc_av.zero_()
        pg.ft_converge()
        torch.ops.ft_collective.alltoallv(send_av, out_av, sc_av, rc_av, per, 1)
        pg._cuda.stream_synchronize(pg._stream)
        _fill_gv(); out_gv.zero_(); rc_gv.zero_()
        pg.ft_converge()
        torch.ops.ft_collective.allgatherv(send_gv, out_gv, sc_gv, rc_gv, per, 1)
        pg._cuda.stream_synchronize(pg._stream)
        _fill_rs(); out_rs.zero_()
        pg.ft_converge()
        torch.ops.ft_collective.reduce_scatterv(send_rs, out_rs, cnt_rs, rs_stride, per)
        pg._cuda.stream_synchronize(pg._stream)

    dist.barrier()

    # --- Capture AllGather graph ---
    # Use torch.cuda.stream() to make ft_stream the current stream during
    # capture, and thread_local mode so only this thread's stream operations
    # are captured (avoids issues with other CUDA internal streams).
    g_ag = torch.cuda.CUDAGraph()
    send_ag.fill_(float(rank + 1))
    output_ag.zero_()
    with torch.cuda.stream(ft_stream):
        with torch.cuda.graph(g_ag, stream=ft_stream,
                              capture_error_mode='thread_local'):
            torch.ops.ft_collective.allgather(send_ag, output_ag)

    # --- Capture AllReduce graph ---
    g_ar = torch.cuda.CUDAGraph()
    send_ar.fill_(float(rank + 1))
    with torch.cuda.stream(ft_stream):
        with torch.cuda.graph(g_ar, stream=ft_stream,
                              capture_error_mode='thread_local'):
            torch.ops.ft_collective.allreduce(send_ar)

    if rank == 0:
        print("  CUDA graphs captured.", flush=True)

    # Replay and verify.
    send_ag.fill_(float(rank + 1))
    output_ag.zero_()
    pg.ft_converge()
    g_ag.replay()
    pg._cuda.stream_synchronize(pg._stream)

    send_ar.fill_(float(rank + 1))
    pg.ft_converge()
    g_ar.replay()
    pg._cuda.stream_synchronize(pg._stream)

    if rank == 0:
        expected_ag = torch.tensor(
            [[float(r + 1)] * count for r in range(nranks)],
            dtype=torch.float32, device="cuda"
        )
        ag_ok = torch.allclose(output_ag, expected_ag)
        expected_sum = float(sum(range(1, nranks + 1)))
        ar_ok = torch.allclose(send_ar, torch.full((count,), expected_sum, device="cuda"))
        print(f"  AllGather correctness: {_verdict(ag_ok)}", flush=True)
        print(f"  AllReduce correctness: {_verdict(ar_ok)}", flush=True)

    # --- Capture + replay AllToAllv graph (device counts -> one graph, any routing) ---
    g_av = torch.cuda.CUDAGraph()
    _fill_av(); out_av.zero_(); rc_av.zero_()
    with torch.cuda.stream(ft_stream):
        with torch.cuda.graph(g_av, stream=ft_stream,
                              capture_error_mode='thread_local'):
            torch.ops.ft_collective.alltoallv(send_av, out_av, sc_av, rc_av, per, 1)
    _fill_av(); out_av.zero_(); rc_av.zero_()
    pg.ft_converge()
    g_av.replay()
    pg._cuda.stream_synchronize(pg._stream)
    if rank == 0:
        print(f"  CUDA graph AllToAllv      correctness: "
              f"{_verdict(_check_av())}", flush=True)

    # --- Capture + replay AllGatherv graph ---
    g_gv = torch.cuda.CUDAGraph()
    _fill_gv(); out_gv.zero_(); rc_gv.zero_()
    with torch.cuda.stream(ft_stream):
        with torch.cuda.graph(g_gv, stream=ft_stream,
                              capture_error_mode='thread_local'):
            torch.ops.ft_collective.allgatherv(send_gv, out_gv, sc_gv, rc_gv, per, 1)
    _fill_gv(); out_gv.zero_(); rc_gv.zero_()
    pg.ft_converge()
    g_gv.replay()
    pg._cuda.stream_synchronize(pg._stream)
    if rank == 0:
        print(f"  CUDA graph AllGatherv     correctness: "
              f"{_verdict(_check_gv())}", flush=True)

    # --- Capture + replay ReduceScatterv graph ---
    g_rs = torch.cuda.CUDAGraph()
    _fill_rs(); out_rs.zero_()
    with torch.cuda.stream(ft_stream):
        with torch.cuda.graph(g_rs, stream=ft_stream,
                              capture_error_mode='thread_local'):
            torch.ops.ft_collective.reduce_scatterv(send_rs, out_rs, cnt_rs, rs_stride, per)
    _fill_rs(); out_rs.zero_()
    pg.ft_converge()
    g_rs.replay()
    pg._cuda.stream_synchronize(pg._stream)
    if rank == 0:
        print(f"  CUDA graph ReduceScatterv correctness: "
              f"{_verdict(_check_rs())}", flush=True)

    # Timed replay loop.
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        send_ag.fill_(float(rank + 1))
        pg.ft_converge()
        g_ag.replay()
        pg._cuda.stream_synchronize(pg._stream)
    t1 = time.perf_counter()

    if rank == 0:
        ms = (t1 - t0) / iters * 1e3
        print(f"  CUDA graph AllGather replay: {ms:.3f} ms/iter ({iters} iters)", flush=True)

    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        send_ar.fill_(float(rank + 1))
        pg.ft_converge()
        g_ar.replay()
        pg._cuda.stream_synchronize(pg._stream)
    t1 = time.perf_counter()

    if rank == 0:
        ms = (t1 - t0) / iters * 1e3
        print(f"  CUDA graph AllReduce replay: {ms:.3f} ms/iter ({iters} iters)\n", flush=True)

    dist.barrier()
    if rank == 0:
        print("Done.", flush=True)

    # ------------------------------------------------------------------ #
    #  Clean teardown                                                      #
    # ------------------------------------------------------------------ #
    # Drop the CUDA graphs, then destroy the PG on
    # every rank so the NCCL monitor is joined before rank 0's TCPStore server
    # exits.  The trailing barrier keeps all ranks together up to destroy so no
    # rank polls a store that has already shut down.
    g_ag = g_ar = g_av = g_gv = g_rs = None
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


# ---------------------------------------------------------------------------
# Main — spawn one process per rank
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nranks", type=int, default=None,
                        help="Number of ranks/GPUs (default: all available)")
    parser.add_argument("--count",  type=int, default=256,
                        help="Elements per rank (default: 256)")
    parser.add_argument("--iters",  type=int, default=10,
                        help="Timed iterations per operation (default: 10)")
    args = parser.parse_args()

    ngpus = _count_gpus()
    if ngpus == 0:
        print("No GPUs found. Exiting.", flush=True)
        sys.exit(1)

    nranks = args.nranks or ngpus
    if nranks > ngpus:
        print(f"Requested {nranks} ranks but only {ngpus} GPUs available.", flush=True)
        sys.exit(1)

    store_port = _find_free_port()
    print(f"Spawning {nranks} rank(s) on {nranks} GPU(s). "
          f"Rendezvous port: {store_port}", flush=True)

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(nranks):
        p = ctx.Process(target=_rank_worker,
                        args=(rank, nranks, args, store_port),
                        daemon=False)
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()

    codes = [p.exitcode for p in procs]
    if any(c != 0 for c in codes):
        print(f"FAIL: rank exit codes {codes}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
