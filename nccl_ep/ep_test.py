#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

"""Python EP test — replicates ep_test.cu using the nccl.ep Pythonic API.

Usage:
    mpirun -np <N> python ep_test.py [OPTIONS]

Options:
    -a {ll,ht}                          Algorithm mode (default: ll)
    -L {em,rm,fl}                       Layout (default: em for LL, fl for HT)
    -q                                  HT eager query-then-allocate
    -s {none,dispatch,combine,both}      Send-only mode (default: none)
    -c                                   Enable cached mode (HT only)
    --backward                          Exercise HT backward dispatch/combine
    --update                            Refresh a handle with changed routing
    --overflow-drop                     Exercise HT overflow DROP
    --ht-flow {auto,push-pull,pull-push} HT dispatch/combine flow
    --meta-exchange-policy {auto,require-fused,force-scan}
                                        HT metadata exchange policy
    --expert-id-kind {auto,local,global}
    --alignment NUM                     HT expert-major slot alignment
    --mask                              Exercise LL active-mask APIs
    -r                                   Enable random mode
    -t NUM                               Number of tokens (default: 50)
    -d NUM                               Hidden dimension size (default: 7168)
    -Q, --quant                          Quantization smoke test (see below)

Quantization smoke test (-Q):
    Exercises the scales-forwarding dispatch recipe on a fixed, fully
    deterministic configuration. Only -a is honoured; -s/-c/-r/-t/-d are
    ignored. Dispatch only -- combine is not run.

    Wire types match ep_bench's
        --dispatch-quantization scales-forward
        --scales-forward-token-dtype fp8e4m3 --scales-forward-scale-dtype fp32
    i.e. fp8e4m3 tokens [128, 7168] and fp32 scales [128, 56].
"""

from __future__ import annotations

import argparse
import ctypes
import random
import struct
import sys

import numpy as np
from cuda.bindings import runtime as cudart
from cuda.core import Device
from mpi4py import MPI

import nccl.core as nccl_core
import nccl.ep as nccl_ep
from nccl._extensions.bindings import nccl_ep as _ep_bindings


# ---------------------------------------------------------------------------
# Custom allocator callbacks. These wrap cudaMalloc/cudaFree — functionally
# equivalent to using the default allocator path (NCCL EP falls back to
# cudaMalloc/cudaFree when alloc_fn is NULL), but exercise the pluggable
# allocator hooks. The decorated functions MUST stay alive at module scope for
# the lifetime of any nccl_ep.Group that referenced them; if GC'd, NCCL EP's stored
# function pointers become dangling.
# ---------------------------------------------------------------------------

@nccl_ep.AllocFn
def _alloc_fn(out_ptr, size, _context):
    err, ptr = cudart.cudaMalloc(size)
    out_ptr[0] = ctypes.c_void_p(int(ptr))
    return int(err)


@nccl_ep.FreeFn
def _free_fn(ptr, _context):
    err, = cudart.cudaFree(ptr)
    return int(err)


_ALLOC_FN_ADDR = ctypes.cast(_alloc_fn, ctypes.c_void_p).value
_FREE_FN_ADDR  = ctypes.cast(_free_fn,  ctypes.c_void_p).value


# ---------------------------------------------------------------------------
# Host <-> Device transfers via cudaMemcpyAsync against raw device pointers.
# ---------------------------------------------------------------------------

_H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
_D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


def _check_cuda(err) -> None:
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def h2d(dev_ptr: int, src_arr: np.ndarray, stream) -> None:
    err, = cudart.cudaMemcpyAsync(
        dev_ptr, src_arr.ctypes.data, src_arr.nbytes, _H2D, int(stream.handle),
    )
    _check_cuda(err)


def d2h(dst_arr: np.ndarray, dev_ptr: int, stream) -> None:
    err, = cudart.cudaMemcpyAsync(
        dst_arr.ctypes.data, dev_ptr, dst_arr.nbytes, _D2H, int(stream.handle),
    )
    _check_cuda(err)


# ---------------------------------------------------------------------------
# Device tensor helper: pairs a raw cudaMalloc allocation with its nccl_ep.NDTensor.
# Sized at create-time so the host side can compute h2d/d2h byte counts
# without re-deriving from sizes.
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    nccl_core.INT8: 1,
    nccl_core.UINT8: 1,
    nccl_core.INT32: 4,
    nccl_core.UINT32: 4,
    nccl_core.INT64: 8,
    nccl_core.UINT64: 8,
    nccl_core.FLOAT16: 2,
    nccl_core.BFLOAT16: 2,
    nccl_core.FLOAT32: 4,
    nccl_core.FLOAT8E4M3: 1,
}


class DevTensor:
    """Owning pair of a cudaMalloc'd device buffer and its ``nccl_ep.Tensor``."""

    def __init__(self, ndim: int, datatype, *sizes: int) -> None:
        nbytes = _DTYPE_BYTES[datatype]
        for s in sizes:
            nbytes *= s
        err, dev_ptr = cudart.cudaMalloc(nbytes)
        _check_cuda(err)
        self.data = int(dev_ptr)
        self.nbytes = nbytes
        self.tensor: nccl_ep.Tensor | None = nccl_ep.Tensor(
            self.data, dtype=int(datatype), shape=sizes,
        )

    def destroy(self) -> None:
        if self.tensor is not None:
            self.tensor = None  # release the Python-owned descriptor
        if self.data:
            cudart.cudaFree(self.data)
            self.data = 0


def make_tensor(ndim: int, datatype, *sizes: int) -> DevTensor:
    return DevTensor(ndim, datatype, *sizes)


def free_tensor(t: DevTensor | None) -> None:
    if t is not None:
        t.destroy()


# ---------------------------------------------------------------------------
# bfloat16 conversion (matches the C++ float_to_bf16 exactly)
# ---------------------------------------------------------------------------

def float_to_bf16(f: float) -> int:
    x = struct.unpack("I", struct.pack("f", f))[0]
    rounding_bias = 0x00007FFF + ((x >> 16) & 1)
    return ((x + rounding_bias) >> 16) & 0xFFFF


# ---------------------------------------------------------------------------
# Quantization smoke test (-Q)
#
# Fixed configuration, deterministic routing: every rank sends
# QUANT_NUM_TOKENS tokens and routes all of them to a single global expert,
# so the one rank owning that expert receives the whole job's tokens and
# every other rank receives none. Dispatch runs under the scales-forwarding
# recipe, which byte-forwards the token payload and its companion scale row.
#
# Slot order inside an expert's zone is NOT deterministic (LL assigns slots
# with an atomicAdd, HT documents its zone as "internal ordering"), and
# expert-major populates no recv_topk_idx, so a received row cannot be
# identified by position. Every row therefore carries its own identity and
# verification is content-addressed: decode (src_rank, token) from the token
# payload, then require the *paired* scale row to carry the same identity.
# That is what actually proves a row's scales travelled with the row.
# ---------------------------------------------------------------------------

QUANT_NUM_TOKENS = 128
QUANT_HIDDEN = 7168
QUANT_TOP_K = 2
QUANT_SCALE_BLOCK = 128  # one scale element per 128 token elements
QUANT_TARGET_EXPERT = 0  # every token routes here; its owner receives everything
# Wire types, matching ep_bench's
#   --dispatch-quantization scales-forward
#   --scales-forward-token-dtype fp8e4m3 --scales-forward-scale-dtype fp32
# FP8 is one byte per element and is NOT dimension-packed (only fp4x2 halves
# the hidden dim), so the physical token shape stays [B, hidden].
QUANT_TOKEN_ELEM_BYTES = 1  # fp8e4m3
QUANT_SCALE_ELEM_BYTES = 4  # fp32


def _quant_global_ids(src_rank: int) -> np.ndarray:
    """Globally unique ids for one rank's tokens: rank-major, contiguous."""
    return src_rank * QUANT_NUM_TOKENS + np.arange(QUANT_NUM_TOKENS, dtype=np.int64)


def _quant_token_rows(gids: np.ndarray, hidden: int) -> np.ndarray:
    """[N, hidden] uint8 fp8e4m3 wire bytes; bytes 0..2 tag the row's identity.

    Mirrors the byte pattern ep_bench.cu uses for this recipe
    (``scalesForwardTokenByte``): rank in byte 0, token index in bytes 1..2,
    a per-element hash after that. The recipe forwards physical bytes, so the
    payload is opaque and compared byte-exactly -- byte values that happen to
    be FP8 NaN are harmless because nothing interprets them numerically.
    """
    gids = np.asarray(gids, dtype=np.int64)
    ranks = gids // QUANT_NUM_TOKENS
    tokens = gids % QUANT_NUM_TOKENS
    j = np.arange(hidden, dtype=np.int64)
    rows = ((ranks[:, None] * 131 + tokens[:, None] * 17 + j[None, :]) & 0xFF).astype(np.uint8)
    rows[:, 0] = ranks.astype(np.uint8)            # source rank
    rows[:, 1] = (tokens // 256).astype(np.uint8)  # token index, high byte
    rows[:, 2] = (tokens % 256).astype(np.uint8)   # token index, low byte
    return rows


def _quant_scale_rows(gids: np.ndarray, scales_per_token: int) -> np.ndarray:
    """[N, S] float32 scales keyed to the same identity as the token row.

    Integer-valued and well under 2**24, so every entry is exact in FP32 and
    equality comparison is safe.
    """
    gids = np.asarray(gids, dtype=np.int64)
    s = np.arange(scales_per_token, dtype=np.int64)
    return (gids[:, None] * 64 + s[None, :]).astype(np.float32)


def _quant_decode_ids(token_rows: np.ndarray, n_ranks: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover (tag_ok, global_id) per received row from its own payload.

    A row whose tag is out of range never came from this job -- an untouched
    or garbage slot -- so it is reported before the payload comparison.
    """
    ranks = token_rows[:, 0].astype(np.int64)
    tokens = token_rows[:, 1].astype(np.int64) * 256 + token_rows[:, 2].astype(np.int64)
    tag_ok = (ranks < n_ranks) & (tokens < QUANT_NUM_TOKENS)
    return tag_ok, ranks * QUANT_NUM_TOKENS + tokens


def run_quant_smoke_test(comm, stream, my_rank: int, n_ranks: int, algorithm) -> None:
    """Dispatch-only smoke + correctness test for the scales-forwarding recipe."""
    is_ll = algorithm == nccl_ep.Algorithm.LOW_LATENCY
    hidden = QUANT_HIDDEN
    num_tokens = QUANT_NUM_TOKENS
    top_k = QUANT_TOP_K
    scales_per_token = hidden // QUANT_SCALE_BLOCK

    num_experts = min(256, top_k * n_ranks)
    if num_experts % n_ranks != 0:
        if my_rank == 0:
            print(f"Error: num_experts ({num_experts}) must be divisible by nRanks ({n_ranks})")
        sys.exit(1)
    num_local_experts = num_experts // n_ranks

    # Global expert QUANT_TARGET_EXPERT lives on exactly one rank; that rank
    # receives every token in the job and all others receive none.
    local_target = QUANT_TARGET_EXPERT - my_rank * num_local_experts
    is_owner = 0 <= local_target < num_local_experts

    # Three distinct row counts, easy to conflate:
    #
    #  recv_budget   worst-case recv SLOTS per rank. Expert-major expands a
    #                token to one slot per local expert it targets, so the
    #                budget carries a top_k factor. HT enforces this:
    #                ncclEpInitHandle asserts
    #                max_recv_tokens >= nRanks * max_dispatch * num_topk for
    #                em_slot staging, and dispatch requires the recv tensor to
    #                cover the full budget under fixed sizing.
    #  ll_zone_rows  LL expert-major's per-expert zone stride, which the kernel
    #                fixes at nRanks * max_dispatch_tokens_per_rank with no
    #                top_k factor (see the layout contract in ep_enums.h).
    #  expected_recv rows this particular routing actually delivers. Every
    #                token names one real expert (slot 1 is -1), so it is one
    #                row per token -- well under the budget.
    recv_budget = num_tokens * n_ranks * top_k
    ll_zone_rows = num_tokens * n_ranks
    expected_recv = num_tokens * n_ranks if is_owner else 0

    # Physical row sizes, used below to offset into the recv tensors.
    token_bytes = hidden * QUANT_TOKEN_ELEM_BYTES            # fp8e4m3 payload
    scale_bytes = scales_per_token * QUANT_SCALE_ELEM_BYTES  # fp32 scales

    # max_token_bytes is an upper bound on one token row including its scales,
    # not an exact fit. Size it to the unquantized bf16 envelope, as ep_bench
    # does (it takes max(hidden * token_dtype_bytes, recipe_payload_bytes)), so
    # the group is configured for the model's native token width and the
    # quantized fp8 + scales payload simply fits inside it.
    max_token_bytes = QUANT_HIDDEN * 2
    assert max_token_bytes >= token_bytes + scale_bytes, "budget must cover tokens + scales"
    # Each physical row, and the HT budget, must be 16-byte aligned (docs/documentation/quantization.md).
    assert token_bytes % 16 == 0, "token row must be 16-byte aligned"
    assert scale_bytes % 16 == 0, "scale row must be 16-byte aligned"
    assert max_token_bytes % 16 == 0, "max_token_bytes must be 16-byte aligned for HT"

    algorithm_name = "LOW_LATENCY" if is_ll else "HIGH_THROUGHPUT"
    if my_rank == 0:
        print(f"Quantization smoke test: algorithm={algorithm_name}, layout=EXPERT_MAJOR, "
              f"recipe=scales-forward, tokens=fp8e4m3, scales=fp32, "
              f"tokens={num_tokens}, hidden={hidden}, top_k={top_k}, "
              f"scales/token={scales_per_token}, experts={num_experts}, "
              f"max_token_bytes={max_token_bytes}")

    config = nccl_ep.GroupConfig(
        algorithm=algorithm,
        num_experts=num_experts,
        max_dispatch_tokens_per_rank=num_tokens,
        max_recv_tokens_per_rank=recv_budget,
        max_token_bytes=max_token_bytes,
        num_topk=top_k,
        alloc=nccl_ep.AllocConfig(alloc_fn=_ALLOC_FN_ADDR, free_fn=_FREE_FN_ADDR),
    )
    ep_group = nccl_ep.Group.create(comm, config)

    # -- routing: slot 0 -> the target expert, slot 1 unused ----------------
    # A negative entry marks an unused top-k slot; both backends skip it.
    topk_idx = make_tensor(2, nccl_core.INT64, num_tokens, top_k)
    topk_idx_host = np.full((num_tokens, top_k), -1, dtype=np.int64)
    topk_idx_host[:, 0] = QUANT_TARGET_EXPERT
    h2d(topk_idx.data, topk_idx_host, stream)

    # -- inputs -------------------------------------------------------------
    input_tokens = make_tensor(2, nccl_core.FLOAT8E4M3, num_tokens, hidden)
    input_scales = make_tensor(2, nccl_core.FLOAT32, num_tokens, scales_per_token)
    topk_weights = make_tensor(2, nccl_core.FLOAT32, num_tokens, top_k)

    gids = _quant_global_ids(my_rank)
    h2d(input_tokens.data, _quant_token_rows(gids, hidden), stream)
    h2d(input_scales.data, _quant_scale_rows(gids, scales_per_token), stream)
    tw_host = np.zeros((num_tokens, top_k), dtype=np.float32)
    tw_host[:, 0] = 1.0  # the unused slot's weight is never read
    h2d(topk_weights.data, tw_host, stream)

    # -- outputs ------------------------------------------------------------
    # LL expert-major is 3D [num_local_experts, ll_zone_rows, ...] -- the zone
    # stride the kernel assumes. HT is 2D over recv slots and must cover the
    # whole budget. Scales mirror the token tensor's leading dimensions with S
    # as the final one.
    expert_counters = recv_total = output_topk_weights = None
    if is_ll:
        output_tokens = make_tensor(3, nccl_core.FLOAT8E4M3, num_local_experts, ll_zone_rows, hidden)
        output_scales = make_tensor(3, nccl_core.FLOAT32, num_local_experts, ll_zone_rows, scales_per_token)
        expert_counters = make_tensor(1, nccl_core.INT32, num_local_experts)
    else:
        output_tokens = make_tensor(2, nccl_core.FLOAT8E4M3, recv_budget, hidden)
        output_scales = make_tensor(2, nccl_core.FLOAT32, recv_budget, scales_per_token)
        # HT expert-major forward dispatch requires a 1D weight per recv slot.
        output_topk_weights = make_tensor(1, nccl_core.FLOAT32, recv_budget)
        recv_total = make_tensor(1, nccl_core.INT32, 1)

    # -- handle -------------------------------------------------------------
    handle_layout_info = None if is_ll else nccl_ep.LayoutInfo(
        recv_total_counter=recv_total.tensor,
    )
    ep_handle = ep_group.create_handle(
        nccl_ep.Layout.EXPERT_MAJOR, topk_idx.tensor,
        layout_info=handle_layout_info,
        config=nccl_ep.HandleConfig(),  # alignment 0 -> unpadded expert zones
        stream=stream,
    )
    stream.sync()

    # -- dispatch -----------------------------------------------------------
    # round_scales must stay 0 under the scales-forwarding recipe.
    dispatch_config = nccl_ep.DispatchConfig(
        send_only=0,
        round_scales=0,
        quantization_recipe=nccl_ep.DispatchQuantizationRecipe.FWD,
    )
    if is_ll:
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=input_tokens.tensor,
            scales=input_scales.tensor,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=output_tokens.tensor,
            scales=output_scales.tensor,
        )
        dispatch_layout = nccl_ep.LayoutInfo(expert_counters=expert_counters.tensor)
    else:
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=input_tokens.tensor,
            topk_weights=topk_weights.tensor,
            scales=input_scales.tensor,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=output_tokens.tensor,
            topk_weights=output_topk_weights.tensor,
            scales=output_scales.tensor,
        )
        dispatch_layout = None

    print(f"Rank {my_rank}: Testing quantized dispatch (scales forwarding)")
    ep_handle.dispatch(
        dispatch_inputs, dispatch_outputs,
        layout_info=dispatch_layout,
        config=dispatch_config,
        stream=stream,
    )
    ep_handle.complete(stream=stream)
    stream.sync()

    # -- how many rows actually landed --------------------------------------
    errors: list[str] = []
    if is_ll:
        counts_host = np.empty(num_local_experts, dtype=np.int32)
        d2h(counts_host, expert_counters.data, stream)
        stream.sync()
        n_valid = int(counts_host[local_target]) if is_owner else 0
        for e in range(num_local_experts):
            if e == local_target:
                continue
            if int(counts_host[e]) != 0:
                errors.append(f"local expert {e} received {int(counts_host[e])} tokens, expected 0")
    else:
        total_host = np.empty(1, dtype=np.int32)
        d2h(total_host, recv_total.data, stream)
        stream.sync()
        n_valid = int(total_host[0])

    if n_valid != expected_recv:
        errors.append(f"recv count is {n_valid}, expected {expected_recv}")

    # -- content-addressed verification of the received rows ----------------
    # Both layouts place the target expert's rows contiguously from the start
    # of its zone; expert 0 is the first zone, and no alignment padding was
    # requested, so the zone base is row 0 of the tensor.
    zone_capacity = ll_zone_rows if is_ll else recv_budget
    n_check = min(n_valid, zone_capacity)
    if is_owner and n_check > 0:
        zone_row = local_target * ll_zone_rows if is_ll else 0
        token_rows = np.empty((n_check, hidden), dtype=np.uint8)
        scale_rows = np.empty((n_check, scales_per_token), dtype=np.float32)
        d2h(token_rows, output_tokens.data + zone_row * token_bytes, stream)
        d2h(scale_rows, output_scales.data + zone_row * scale_bytes, stream)
        stream.sync()

        tag_ok, gids_seen = _quant_decode_ids(token_rows, n_ranks)
        bad_tags = np.flatnonzero(~tag_ok)
        for row in bad_tags[:5]:
            errors.append(
                f"row {int(row)} has an out-of-range identity tag "
                f"(rank byte {int(token_rows[row, 0])}, token bytes "
                f"{int(token_rows[row, 1])}/{int(token_rows[row, 2])})"
            )

        # Full-row payload check against the identity the row claims.
        expected_tokens = _quant_token_rows(gids_seen, hidden)
        bad_tokens = np.flatnonzero((token_rows != expected_tokens).any(axis=1))
        for row in bad_tokens[:5]:
            col = int(np.flatnonzero(token_rows[row] != expected_tokens[row])[0])
            errors.append(
                f"token payload mismatch at row {int(row)} (gid {int(gids_seen[row])}) "
                f"byte {col}: expected 0x{int(expected_tokens[row, col]):02x}, "
                f"got 0x{int(token_rows[row, col]):02x}"
            )

        # The pairing proof: the scale row must carry the identity that the
        # token row claims, not merely some valid identity.
        expected_scales = _quant_scale_rows(gids_seen, scales_per_token)
        bad_scales = np.flatnonzero((scale_rows != expected_scales).any(axis=1))
        for row in bad_scales[:5]:
            col = int(np.flatnonzero(scale_rows[row] != expected_scales[row])[0])
            errors.append(
                f"scales not paired with their token at row {int(row)} "
                f"(token claims gid {int(gids_seen[row])}) element {col}: "
                f"expected {float(expected_scales[row, col])}, got {float(scale_rows[row, col])}"
            )

        # Completeness: exactly one row per (source rank, token), no gaps or
        # duplicates.
        if not np.array_equal(np.sort(gids_seen), np.arange(expected_recv, dtype=np.int64)):
            seen = set(int(g) for g in gids_seen)
            missing = sorted(set(range(expected_recv)) - seen)
            errors.append(
                f"received token set is wrong: {len(seen)} distinct of {expected_recv} expected"
                + (f", first missing gid {missing[0]}" if missing else "")
            )

    if errors:
        print(f"Rank {my_rank}: Quantization verification FAILED ({len(errors)} problems)")
        for message in errors[:10]:
            print(f"Rank {my_rank}:   {message}")
    else:
        print(f"Rank {my_rank}: Quantization verification PASSED "
              f"({n_valid} rows, tokens and scales paired)")

    # -- cleanup ------------------------------------------------------------
    free_tensor(topk_idx)
    free_tensor(input_tokens)
    free_tensor(input_scales)
    free_tensor(topk_weights)
    free_tensor(output_tokens)
    free_tensor(output_scales)
    free_tensor(expert_counters)
    free_tensor(output_topk_weights)
    free_tensor(recv_total)

    ep_handle.destroy()
    ep_group.destroy()
    comm.destroy()

    if errors:
        sys.exit(1)
    print(f"[MPI Rank {my_rank}] Success ")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

ELEMENTS_TESTED_PER_TOKEN = 10


def main():  # noqa: C901 — kept as a single function to mirror ep_test.cu
    mpi_comm = MPI.COMM_WORLD
    my_rank = mpi_comm.Get_rank()
    n_ranks = mpi_comm.Get_size()

    parser = argparse.ArgumentParser(description="EP Test (Python)")
    parser.add_argument("-a", choices=["ll", "ht"], default="ll", help="Algorithm mode")
    parser.add_argument("-L", choices=["em", "rm", "fl"], help="Output layout (default: em for LL, fl for HT)")
    parser.add_argument("-q", action="store_true", help="HT eager query-then-allocate mode")
    parser.add_argument("-s", choices=["none", "dispatch", "combine", "both"], default="none",
                        help="Send-only mode")
    parser.add_argument("-c", action="store_true", help="Enable cached mode (HT only)")
    parser.add_argument("--backward", action="store_true", help="Exercise HT backward dispatch/combine")
    parser.add_argument("--update", action="store_true", help="Update the handle with changed routing")
    parser.add_argument("--overflow-drop", action="store_true", help="Exercise HT overflow DROP policy")
    parser.add_argument("--ht-flow", choices=["auto", "push-pull", "pull-push"], default="auto",
                        help="HT dispatch/combine flow")
    parser.add_argument("--meta-exchange-policy",
                        choices=["auto", "require-fused", "force-scan"], default="auto",
                        help="HT metadata exchange policy")
    parser.add_argument("--expert-id-kind", choices=["auto", "local", "global"], default="auto",
                        help="Numbering for received expert indices")
    parser.add_argument("--alignment", type=int, default=1,
                        help="HT expert-major per-expert output alignment")
    parser.add_argument("--mask", action="store_true", help="Exercise LL active-mask APIs")
    parser.add_argument("-r", action="store_true", help="Enable random mode")
    parser.add_argument("-t", type=int, default=50, help="Number of tokens")
    parser.add_argument("-d", type=int, default=7168, help="Hidden dimension size")
    parser.add_argument("-Q", "--quant", action="store_true",
                        help="Quantization smoke test (scales forwarding); only -a is honoured")
    args = parser.parse_args()

    algorithm = nccl_ep.Algorithm.LOW_LATENCY if args.a == "ll" else nccl_ep.Algorithm.HIGH_THROUGHPUT
    quant_mode = args.quant
    if quant_mode and my_rank == 0:
        ignored = [flag for flag, used in (
            ("-s", args.s != "none"), ("-c", args.c), ("-r", args.r),
            ("-t", args.t != 50), ("-d", args.d != 7168),
        ) if used]
        if ignored:
            print(f"Note: -Q uses a fixed configuration; ignoring {' '.join(ignored)}")
    dispatch_send_only = 1 if args.s in ("dispatch", "both") else 0
    combine_send_only = 1 if args.s in ("combine", "both") else 0
    cached_mode = args.c
    random_mode = args.r
    num_tokens = args.t
    hidden = args.d
    layout_name = args.L or ("em" if args.a == "ll" else "fl")

    if args.a == "ll" and layout_name not in ("em", "rm"):
        parser.error("LL supports only -L em or -L rm")
    if args.a == "ht" and layout_name not in ("fl", "em"):
        parser.error("HT supports only -L fl or -L em")
    if args.a == "ht" and args.s != "none":
        parser.error("send-only/complete mode is supported only by LL")
    if num_tokens <= 0:
        parser.error("-t must be positive")
    if hidden < ELEMENTS_TESTED_PER_TOKEN:
        parser.error(f"-d must be at least {ELEMENTS_TESTED_PER_TOKEN}")
    if args.a == "ht" and (hidden * 2) % 16:
        parser.error("HT requires -d * sizeof(bfloat16) to be a multiple of 16 bytes")
    if args.q and args.a != "ht":
        parser.error("-q is supported only by HT")
    if args.q and random_mode:
        parser.error("-q requires deterministic routing so every rank has a non-empty output")
    if args.backward and args.a != "ht":
        parser.error("--backward is supported only by HT")
    if args.overflow_drop and (args.a != "ht" or layout_name != "fl" or args.q):
        parser.error("--overflow-drop requires statically-sized HT flat mode")
    if args.mask and args.a != "ll":
        parser.error("--mask is supported only by LL")
    if args.expert_id_kind != "auto" and layout_name not in ("rm", "fl"):
        parser.error("--expert-id-kind applies only to LL rank-major and HT flat")
    if args.alignment < 1 or args.alignment & (args.alignment - 1):
        parser.error("--alignment must be a positive power of two")
    if args.alignment != 1 and not (args.a == "ht" and layout_name == "em"):
        parser.error("--alignment applies only to HT expert-major")
    if args.c and (args.a != "ht" or layout_name != "fl"):
        parser.error("-c currently requires HT flat layout")
    if args.overflow_drop and (args.backward or args.c):
        parser.error("--overflow-drop cannot be combined with --backward or -c")
    if args.a != "ht" and (args.ht_flow != "auto" or args.meta_exchange_policy != "auto"):
        parser.error("--ht-flow and --meta-exchange-policy apply only to HT")
    if args.update and (args.overflow_drop or random_mode):
        parser.error("--update cannot be combined with --overflow-drop or -r")

    if n_ranks not in (2, 4) and n_ranks % 8 != 0:
        if my_rank == 0:
            print("Error: nRanks must be 2, 4 or multiple of 8 for this test")
        sys.exit(1)

    top_k = min(8, n_ranks)
    num_experts = min(256, top_k * n_ranks)
    num_local_experts = num_experts // n_ranks
    local_experts_start = num_local_experts * my_rank
    local_experts_end = local_experts_start + num_local_experts

    if not quant_mode and num_experts % n_ranks != 0:
        if my_rank == 0:
            print(f"Error: num_experts ({num_experts}) must be divisible by nRanks ({n_ranks})")
        sys.exit(1)
    if not quant_mode and top_k > num_local_experts:
        if my_rank == 0:
            print(f"Error: top_k ({top_k}) must be <= num_local_experts ({num_local_experts})")
        sys.exit(1)

    version = nccl_ep.get_lib_version()
    lib_path = nccl_ep.get_lib_path()
    if my_rank == 0:
        print(f"NCCL EP library version {version} ({lib_path or 'path unavailable'})")

    # Local rank = rank within the per-node sub-communicator.
    local_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()

    device = Device(local_rank)
    device.set_current()
    stream = device.create_stream()

    # NCCL communicator: rank 0 generates a unique ID, MPI broadcasts it,
    # then every rank initializes its own Communicator.
    unique_id = nccl_core.get_unique_id() if my_rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)
    comm = nccl_core.Communicator.init(nranks=n_ranks, rank=my_rank, unique_id=unique_id)

    # The quantization mode owns its whole configuration (and its teardown),
    # so it runs instead of the standard dispatch/combine sequence below.
    if quant_mode:
        run_quant_smoke_test(comm, stream, my_rank, n_ranks, algorithm)
        return

    is_ll = algorithm == nccl_ep.Algorithm.LOW_LATENCY
    is_em = layout_name == "em"
    layout_map = {
        "em": nccl_ep.Layout.EXPERT_MAJOR,
        "rm": nccl_ep.Layout.RANK_MAJOR,
        "fl": nccl_ep.Layout.FLAT,
    }
    handle_layout = layout_map[layout_name]
    expert_kind_map = {
        "auto": nccl_ep.ExpertIdKind.AUTO,
        "local": nccl_ep.ExpertIdKind.LOCAL,
        "global": nccl_ep.ExpertIdKind.GLOBAL,
    }
    ht_flow_map = {
        "auto": nccl_ep.HtFlow.AUTO,
        "push-pull": nccl_ep.HtFlow.PUSH_PULL,
        "pull-push": nccl_ep.HtFlow.PULL_PUSH,
    }
    meta_exchange_policy_map = {
        "auto": nccl_ep.MetaExchangePolicy.AUTO,
        "require-fused": nccl_ep.MetaExchangePolicy.REQUIRE_FUSED,
        "force-scan": nccl_ep.MetaExchangePolicy.FORCE_SCAN,
    }
    static_recv_slots = num_tokens * n_ranks * (top_k if is_em else 1)
    if not is_ll and is_em and args.alignment > 1:
        # Padding is per local-expert zone, so reserve the maximum aggregate
        # padding in addition to the unpadded worst-case token slots.
        static_recv_slots += (args.alignment - 1) * num_local_experts
    max_recv_slots = 0 if args.q else (num_tokens if args.overflow_drop else static_recv_slots)

    # -- EP group -----------------------------------------------------------
    config = nccl_ep.GroupConfig(
        algorithm=algorithm,
        num_experts=num_experts,
        max_dispatch_tokens_per_rank=num_tokens,
        max_recv_tokens_per_rank=max_recv_slots if not is_ll else 0,
        max_token_bytes=hidden * 2,  # bfloat16
        enable_mask=args.mask,
        ht_flow=ht_flow_map[args.ht_flow],
        meta_exchange_policy=meta_exchange_policy_map[args.meta_exchange_policy],
        overflow_policy=(
            nccl_ep.OverflowPolicy.DROP
            if args.overflow_drop else nccl_ep.OverflowPolicy.AUTO
        ),
        num_topk=top_k,
        alloc=nccl_ep.AllocConfig(alloc_fn=_ALLOC_FN_ADDR, free_fn=_FREE_FN_ADDR),
    )

    algorithm_name = "LOW_LATENCY" if algorithm == nccl_ep.Algorithm.LOW_LATENCY else "HIGH_THROUGHPUT"
    extra = ""
    print(f"Rank {my_rank}: Testing ncclEpCreateGroup with algorithm: {algorithm_name}{extra}")

    ep_group = nccl_ep.Group.create(comm, config)

    # -- topk_idx tensor [num_tokens, top_k] int64 --------------------------
    topk_idx = make_tensor(2, nccl_core.INT64, num_tokens, top_k)

    if random_mode:
        random.seed(my_rank + 42)
        first_experts = np.array(
            [random.randint(0, num_experts - 1) for _ in range(num_tokens)],
            dtype=np.int64,
        )
        # Each row: [first, first+1, ..., first+top_k-1] modulo num_experts.
        offsets = np.arange(top_k, dtype=np.int64)
        topk_idx_host = ((first_experts[:, None] + offsets) % num_experts).astype(np.int64)
        if my_rank == 0:
            print("Random mode enabled: first expert random, rest deterministic (no repetitions)")
    elif args.overflow_drop:
        # Concentrate traffic on rank 0 so its true recv count exceeds the
        # deliberately small static capacity.
        topk_idx_host = np.tile(
            np.arange(top_k, dtype=np.int64), (num_tokens, 1),
        )
    else:
        topk_idx_host = np.empty((num_tokens, top_k), dtype=np.int64)
        for i in range(num_tokens):
            for j in range(top_k):
                topk_idx_host[i, j] = (local_experts_end + j) % num_experts

    h2d(topk_idx.data, topk_idx_host, stream)

    # -- EP handle ----------------------------------------------------------
    print(f"Rank {my_rank}: Testing ncclEpCreateHandle")
    handle_recv_total: DevTensor | None = None
    handle_expert_counters: DevTensor | None = None
    handle_expert_offsets: DevTensor | None = None
    handle_layout_info = None
    if not is_ll:
        handle_recv_total = make_tensor(1, nccl_core.INT32, 1)
        if is_em:
            # HT preprocessing requires every integer output tensor in one
            # LayoutInfo to use the same dtype.
            handle_expert_counters = make_tensor(1, nccl_core.INT32, num_local_experts)
            handle_expert_offsets = make_tensor(1, nccl_core.INT32, num_local_experts)
        handle_layout_info = nccl_ep.LayoutInfo(
            expert_counters=handle_expert_counters.tensor if handle_expert_counters else None,
            expert_offsets=handle_expert_offsets.tensor if handle_expert_offsets else None,
            recv_total_counter=handle_recv_total.tensor,
            recv_topk_idx_kind=expert_kind_map[args.expert_id_kind],
        )
    ep_handle = ep_group.create_handle(
        handle_layout, topk_idx.tensor,
        layout_info=handle_layout_info,
        config=nccl_ep.HandleConfig(
            dispatch_output_per_expert_alignment=args.alignment if is_em and not is_ll else 0,
        ),
        stream=stream,
    )
    stream.sync()

    actual_recv_slots = 0
    if not is_ll:
        recv_total_host = np.empty(1, dtype=np.int32)
        d2h(recv_total_host, handle_recv_total.data, stream)
        stream.sync()
        actual_recv_slots = int(recv_total_host[0])
        if actual_recv_slots < 0:
            raise AssertionError(f"rank {my_rank}: negative recv_total_counter")
    if args.q:
        num_recv_tokens = actual_recv_slots
        print(f"Rank {my_rank}: eager recv_total_counter={num_recv_tokens} "
              f"(static worst case {static_recv_slots})")
    elif is_ll:
        num_recv_tokens = num_tokens * n_ranks
    else:
        num_recv_tokens = max_recv_slots
    assert num_recv_tokens > 0
    if not is_ll and is_em:
        counts_host = np.empty(num_local_experts, dtype=np.int32)
        offsets_host = np.empty(num_local_experts, dtype=np.int32)
        d2h(counts_host, handle_expert_counters.data, stream)
        d2h(offsets_host, handle_expert_offsets.data, stream)
        stream.sync()
        expected_offsets = np.concatenate((
            np.array([0], dtype=np.int32),
            np.cumsum(counts_host[:-1], dtype=np.int32),
        ))
        if not np.array_equal(offsets_host, expected_offsets):
            raise AssertionError(
                f"rank {my_rank}: expert offsets {offsets_host} do not match counts {counts_host}"
            )
        if args.alignment > 1 and np.any(counts_host % args.alignment):
            raise AssertionError(
                f"rank {my_rank}: expert counts are not aligned to {args.alignment}: {counts_host}"
            )
        if int(counts_host.sum()) > num_recv_tokens:
            raise AssertionError("HT expert-major metadata exceeds dispatch output capacity")

    dispatch_config = nccl_ep.DispatchConfig(send_only=dispatch_send_only, round_scales=0)

    # -- input/output tensors for dispatch ----------------------------------
    input_tokens = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)
    topk_weights = make_tensor(2, nccl_core.FLOAT32, num_tokens, top_k)

    if is_ll and is_em:
        output_tokens = make_tensor(
            3, nccl_core.BFLOAT16,
            num_local_experts, config.max_dispatch_tokens_per_rank * n_ranks, hidden,
        )
    elif is_ll:
        output_tokens = make_tensor(
            3, nccl_core.BFLOAT16, n_ranks, num_tokens, hidden,
        )
    else:
        output_tokens = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)

    local_tensor_recv_count: DevTensor | None = None
    if is_ll:
        local_tensor_recv_count = make_tensor(
            1, nccl_core.INT32, num_local_experts if is_em else n_ranks,
        )

    output_topk_weights: DevTensor | None = None
    output_topk_idx: DevTensor | None = None
    if not is_ll:
        output_topk_weights = (
            make_tensor(1, nccl_core.FLOAT32, num_recv_tokens)
            if is_em else
            make_tensor(2, nccl_core.FLOAT32, num_recv_tokens, top_k)
        )
        if not is_em:
            output_topk_idx = make_tensor(2, nccl_core.INT64, num_recv_tokens, top_k)
    elif not is_em:
        output_topk_weights = make_tensor(
            3, nccl_core.FLOAT32, n_ranks, num_tokens, top_k,
        )
        output_topk_idx = make_tensor(
            3, nccl_core.INT32, n_ranks, num_tokens, top_k,
        )

    # Fill input tokens: first ELEMENTS_TESTED_PER_TOKEN values = 0x1000 + my_rank
    input_host = np.zeros((num_tokens, hidden), dtype=np.uint16)
    input_host[:, :ELEMENTS_TESTED_PER_TOKEN] = 0x1000 + my_rank
    h2d(input_tokens.data, input_host, stream)

    # Fill topk_weights: 1.0 / top_k for every entry.
    tw_host = np.full(num_tokens * top_k, 1.0 / top_k, dtype=np.float32)
    h2d(topk_weights.data, tw_host, stream)

    # Build the named-struct ABI bundles for dispatch.
    if is_ll:
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=input_tokens.tensor,
            topk_weights=topk_weights.tensor if not is_em else None,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=output_tokens.tensor,
            topk_weights=output_topk_weights.tensor if not is_em else None,
            topk_idx=output_topk_idx.tensor if not is_em else None,
        )
        dispatch_layout = nccl_ep.LayoutInfo(
            expert_counters=local_tensor_recv_count.tensor if is_em else None,
            src_rank_counters=local_tensor_recv_count.tensor if not is_em else None,
            recv_topk_idx_kind=expert_kind_map[args.expert_id_kind],
        )
    else:
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=input_tokens.tensor,
            topk_weights=topk_weights.tensor,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=output_tokens.tensor,
            topk_weights=output_topk_weights.tensor,
            topk_idx=output_topk_idx.tensor if output_topk_idx else None,
        )
        dispatch_layout = (
            nccl_ep.LayoutInfo(recv_topk_idx_kind=expert_kind_map[args.expert_id_kind])
            if not is_em else None
        )

    print(f"Rank {my_rank}: Testing dispatch (send_only={bool(dispatch_send_only)})")
    ep_handle.dispatch(
        dispatch_inputs, dispatch_outputs,
        layout_info=dispatch_layout,
        config=dispatch_config,
        stream=stream,
    )

    if dispatch_send_only:
        print(f"Rank {my_rank}: Testing complete (after dispatch)")
        ep_handle.complete(stream=stream)
    stream.sync()

    # Read recv_count for verification.
    recv_count_host: np.ndarray | None = None
    if is_ll:
        recv_count_host = np.empty(num_local_experts if is_em else n_ranks, dtype=np.int32)
        d2h(recv_count_host, local_tensor_recv_count.data, stream)
        stream.sync()
    elif args.overflow_drop:
        true_recv = np.empty(1, dtype=np.int32)
        d2h(true_recv, handle_recv_total.data, stream)
        stream.sync()
        expected_true = n_ranks * num_tokens if my_rank == 0 else 0
        if int(true_recv[0]) != expected_true:
            raise AssertionError(
                f"rank {my_rank}: expected true recv count {expected_true}, got {int(true_recv[0])}"
            )
        if my_rank == 0 and int(true_recv[0]) <= num_recv_tokens:
            raise AssertionError("overflow DROP case did not exceed its output capacity")
        print(f"Rank {my_rank}: overflow DROP true_recv={int(true_recv[0])}, "
              f"capacity={num_recv_tokens}")

    recv_from_expert_start = (local_experts_start + num_experts - num_local_experts) % num_experts
    recv_rank = recv_from_expert_start // num_local_experts

    # Verify dispatch output (deterministic mode only).
    dispatch_check_passed = True

    if not random_mode and not args.overflow_drop and is_ll and is_em and recv_count_host is not None:
        total_elems = num_local_experts * config.max_dispatch_tokens_per_rank * n_ranks * hidden
        output_host = np.empty(total_elems, dtype=np.uint16)
        d2h(output_host, output_tokens.data, stream)
        stream.sync()

        max_t = config.max_dispatch_tokens_per_rank * n_ranks
        for e in range(num_local_experts):
            if recv_count_host[e] != num_tokens:
                print(f"Recv_count check failed! Rank {my_rank}, expert {e}: "
                      f"expected {num_tokens}, got {int(recv_count_host[e])}")
                dispatch_check_passed = False
                break
            for t in range(min(int(recv_count_host[e]), max_t)):
                token_off = (e * max_t + t) * hidden
                for j in range(ELEMENTS_TESTED_PER_TOKEN):
                    expected = 0x1000 + recv_rank
                    actual = int(output_host[token_off + j])
                    if actual != expected:
                        print(f"Dispatch data check failed! Rank {my_rank}, expert {e}, "
                              f"token {t}, element {j}: expected {expected}, got {actual}")
                        dispatch_check_passed = False
                        break
                if not dispatch_check_passed:
                    break
            if not dispatch_check_passed:
                break

    elif not random_mode and not args.overflow_drop and is_ll:
        if recv_count_host is not None:
            if int(recv_count_host.sum()) != num_tokens:
                print(f"Recv_count check failed! Rank {my_rank}: "
                      f"expected total {num_tokens}, got {int(recv_count_host.sum())}")
                dispatch_check_passed = False
        output_host = np.empty((n_ranks, num_tokens, hidden), dtype=np.uint16)
        d2h(output_host, output_tokens.data, stream)
        recv_idx_host = np.empty((n_ranks, num_tokens, top_k), dtype=np.int32)
        d2h(recv_idx_host, output_topk_idx.data, stream)
        stream.sync()
        for src_rank, count in enumerate(recv_count_host):
            for token in range(int(count)):
                if not np.all(
                    output_host[src_rank, token, :ELEMENTS_TESTED_PER_TOKEN]
                    == 0x1000 + src_rank
                ):
                    print(f"Rank-major dispatch payload check failed on rank {my_rank}, "
                          f"source {src_rank}, token {token}")
                    dispatch_check_passed = False
                    break
            if not dispatch_check_passed:
                break
        valid_idx = recv_idx_host[recv_idx_host != -1]
        if args.expert_id_kind == "global":
            idx_ok = np.all((valid_idx >= local_experts_start) & (valid_idx < local_experts_end))
        else:
            idx_ok = np.all((valid_idx >= 0) & (valid_idx < num_local_experts))
        if not idx_ok:
            print(f"Rank {my_rank}: rank-major recv_topk_idx numbering check failed")
            dispatch_check_passed = False

    elif not random_mode and not args.overflow_drop:
        output_host = np.empty(num_recv_tokens * hidden, dtype=np.uint16)
        d2h(output_host, output_tokens.data, stream)
        stream.sync()
        check_count = min(num_tokens, num_recv_tokens)
        for i in range(check_count):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                expected = 0x1000 + recv_rank
                actual = int(output_host[i * hidden + j])
                if actual != expected:
                    print(f"Dispatch check failed! Rank {my_rank}, token {i}, "
                          f"element {j}: expected {expected}, got {actual}")
                    dispatch_check_passed = False
                    break
            if not dispatch_check_passed:
                break

    # Verify HT recv_topk_weights / recv_topk_idx.
    if not random_mode and not args.overflow_drop and not is_ll:
        recv_tw_elems = num_recv_tokens if is_em else num_recv_tokens * top_k
        recv_tw = np.empty(recv_tw_elems, dtype=np.float32)
        d2h(recv_tw, output_topk_weights.data, stream)
        recv_ti = None
        if output_topk_idx is not None:
            recv_ti = np.empty(num_recv_tokens * top_k, dtype=np.int64)
            d2h(recv_ti, output_topk_idx.data, stream)
        stream.sync()

        # Only the first num_tokens rows are meaningful for the per-rank check.
        expected_weight = np.float32(1.0 / top_k)
        window_tw = recv_tw[:num_tokens * (1 if is_em else top_k)]
        w_bad = window_tw != expected_weight
        weight_errors = int(w_bad.sum())
        idx_errors = 0
        i_bad = np.array([], dtype=bool)
        window_ti = np.array([], dtype=np.int64)
        if recv_ti is not None:
            window_ti = recv_ti[:num_tokens * top_k]
            if args.expert_id_kind == "global":
                i_bad = ((window_ti != -1) &
                         ((window_ti < local_experts_start) | (window_ti >= local_experts_end)))
            else:
                i_bad = ((window_ti != -1) &
                         ((window_ti < 0) | (window_ti >= num_local_experts)))
            idx_errors = int(i_bad.sum())

        for off in np.flatnonzero(w_bad)[:5]:
            i, j = int(off) // top_k, int(off) % top_k
            print(f"Rank {my_rank}: recv_topk_weights[{i}][{j}] = {window_tw[off]}, "
                  f"expected {expected_weight}")
        for off in np.flatnonzero(i_bad)[:5]:
            i, j = int(off) // top_k, int(off) % top_k
            print(f"Rank {my_rank}: recv_topk_idx[{i}][{j}] = {int(window_ti[off])}, "
                  f"expected range [0, {num_experts})")

        if weight_errors:
            print(f"Rank {my_rank}: recv_topk_weights verification failed with {weight_errors} errors")
        if idx_errors:
            print(f"Rank {my_rank}: recv_topk_idx verification failed with {idx_errors} errors")
        if weight_errors == 0 and idx_errors == 0:
            print(f"Rank {my_rank}: {algorithm_name} recv_topk_weights / recv_topk_idx verification passed")
        else:
            dispatch_check_passed = False

    if random_mode:
        print(f"Rank {my_rank}: {algorithm_name} Dispatch flow completed (random mode, checks skipped)")
    elif dispatch_check_passed:
        print(f"Rank {my_rank}: {algorithm_name} Dispatch flow passed successfully")
    else:
        print(f"Rank {my_rank}: Exiting test due to dispatch failure")
        sys.exit(1)

    # ===================================================================
    # Combine
    # ===================================================================
    print(f"Rank {my_rank}: Testing {algorithm_name} Combine flow")

    if is_ll and is_em:
        expert_outputs = make_tensor(
            3, nccl_core.BFLOAT16,
            num_local_experts, config.max_dispatch_tokens_per_rank * n_ranks, hidden,
        )
        eo_host = np.zeros(config.max_dispatch_tokens_per_rank * hidden, dtype=np.uint16)
        for t in range(config.max_dispatch_tokens_per_rank):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                eo_host[t * hidden + j] = float_to_bf16(float((j + 1) * 2))
        stride_bytes = config.max_dispatch_tokens_per_rank * hidden * n_ranks * 2
        for e in range(num_local_experts):
            h2d(expert_outputs.data + e * stride_bytes, eo_host, stream)
    elif is_ll:
        expert_outputs = make_tensor(
            3, nccl_core.BFLOAT16, n_ranks, num_tokens, hidden,
        )
        eo_host = np.zeros((n_ranks, num_tokens, hidden), dtype=np.uint16)
        eo_host[:, :, :ELEMENTS_TESTED_PER_TOKEN] = np.array(
            [float_to_bf16(float((j + 1) * 2)) for j in range(ELEMENTS_TESTED_PER_TOKEN)],
            dtype=np.uint16,
        )
        h2d(expert_outputs.data, eo_host, stream)
    else:
        expert_outputs = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)
        eo_host = np.zeros(num_recv_tokens * hidden, dtype=np.uint16)
        for t in range(num_recv_tokens):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                eo_host[t * hidden + j] = float_to_bf16(float((j + 1) * 2))
        h2d(expert_outputs.data, eo_host, stream)

    combined_output = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)

    if is_ll:
        combine_inputs = nccl_ep.CombineInputs(tokens=expert_outputs.tensor)
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=combined_output.tensor,
            # Expert-major applies the original per-token routing weights on
            # the receive side. Rank-major is already locally reduced.
            topk_weights=topk_weights.tensor if is_em else None,
        )
    else:
        combine_inputs = nccl_ep.CombineInputs(tokens=expert_outputs.tensor)
        combine_outputs = nccl_ep.CombineOutputs(tokens=combined_output.tensor)

    print(f"Rank {my_rank}: Testing combine (send_only={bool(combine_send_only)})")
    ep_handle.combine(
        combine_inputs, combine_outputs,
        config=nccl_ep.CombineConfig(send_only=combine_send_only),
        stream=stream,
    )

    if combine_send_only:
        print(f"Rank {my_rank}: Testing complete (after combine)")
        ep_handle.complete(stream=stream)
    stream.sync()

    # Verify combine output.
    combine_errors = 0
    if not random_mode and not args.overflow_drop:
        co_host = np.empty(num_tokens * hidden, dtype=np.uint16)
        d2h(co_host, combined_output.data, stream)
        stream.sync()
        # HT expert-major combine sums one received slot per top-k route.
        # Other layouts either reduce locally or apply routing weights.
        combine_scale = top_k if not is_ll and is_em else 1
        for i in range(num_tokens):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                expected = float_to_bf16(float(combine_scale * (j + 1) * 2))
                actual = int(co_host[i * hidden + j])
                if actual != expected:
                    print(f"Combine check failed! Rank {my_rank}, token {i}, "
                          f"element {j}: expected {expected}, got {actual}")
                    combine_errors += 1
                    if combine_errors >= 5:
                        break
            if combine_errors >= 5:
                break

    if args.overflow_drop:
        print(f"Rank {my_rank}: Combine completed after overflow DROP")
    elif random_mode:
        print(f"Rank {my_rank}: Combine flow completed (random mode, checks skipped)")
    elif combine_errors == 0:
        print(f"Rank {my_rank}: Combine verification PASSED! "
              f"All {num_tokens} tokens with {hidden} elements each correctly combined")
    else:
        print(f"Rank {my_rank}: Combine verification FAILED with {combine_errors} errors")
        sys.exit(1)

    # ===================================================================
    # HT backward pass. Forward dispatch's recv weights are the backward
    # combine weights, which makes this a round-trip correctness check.
    # ===================================================================
    extra_tensors: list[DevTensor] = []
    if args.backward:
        bwd_recv_tokens = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)
        bwd_combined_tokens = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)
        bwd_combined_weights = make_tensor(2, nccl_core.FLOAT32, num_tokens, top_k)
        extra_tensors.extend([bwd_recv_tokens, bwd_combined_tokens, bwd_combined_weights])

        ep_handle.dispatch(
            nccl_ep.DispatchInputs(tokens=input_tokens.tensor),
            nccl_ep.DispatchOutputs(tokens=bwd_recv_tokens.tensor),
            config=nccl_ep.DispatchConfig(pass_direction=nccl_ep.PassDir.BWD),
            stream=stream,
        )
        stream.sync()
        ep_handle.combine(
            nccl_ep.CombineInputs(
                tokens=output_tokens.tensor,
                topk_weights=output_topk_weights.tensor,
            ),
            nccl_ep.CombineOutputs(
                tokens=bwd_combined_tokens.tensor,
                topk_weights=bwd_combined_weights.tensor,
            ),
            config=nccl_ep.CombineConfig(pass_direction=nccl_ep.PassDir.BWD),
            stream=stream,
        )
        stream.sync()
        bwd_recv_host = np.empty(actual_recv_slots * hidden, dtype=np.uint16)
        fwd_recv_host = np.empty(actual_recv_slots * hidden, dtype=np.uint16)
        d2h(bwd_recv_host, bwd_recv_tokens.data, stream)
        d2h(fwd_recv_host, output_tokens.data, stream)
        recovered_weights = np.empty(num_tokens * top_k, dtype=np.float32)
        d2h(recovered_weights, bwd_combined_weights.data, stream)
        stream.sync()
        if not np.array_equal(bwd_recv_host, fwd_recv_host):
            raise AssertionError(f"rank {my_rank}: HT backward dispatch token scatter failed")
        if not np.allclose(recovered_weights, tw_host, rtol=0, atol=1e-6):
            raise AssertionError(f"rank {my_rank}: HT backward top-k weight round-trip failed")
        print(f"Rank {my_rank}: HT backward dispatch/combine verification PASSED")

    # ===================================================================
    # Cached mode (HT only): repeat dispatch+combine and compare outputs.
    # ===================================================================
    cached_tensors: list[DevTensor] = []
    if cached_mode:
        if is_ll:
            print(f"Rank {my_rank}: Error - cached mode is only supported in HT modes (not LL)")
            sys.exit(1)

        print(f"Rank {my_rank}: Testing cached mode ({algorithm_name})")

        # save first-phase outputs
        first_d0 = first_dw = first_di = first_co = None
        if not random_mode:
            first_d0 = np.empty(actual_recv_slots * hidden, dtype=np.uint16)
            d2h(first_d0, output_tokens.data, stream)
            first_dw = np.empty(actual_recv_slots * top_k, dtype=np.float32)
            d2h(first_dw, output_topk_weights.data, stream)
            first_di = np.empty(actual_recv_slots * top_k, dtype=np.int64)
            d2h(first_di, output_topk_idx.data, stream)

            first_co = np.empty(num_tokens * hidden, dtype=np.uint16)
            d2h(first_co, combined_output.data, stream)
            stream.sync()

        # New output tensors for the second dispatch / combine.
        cached_out_tokens = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)
        cached_out_weights = make_tensor(2, nccl_core.FLOAT32, num_recv_tokens, top_k)
        cached_out_idx = make_tensor(2, nccl_core.INT64, num_recv_tokens, top_k)
        cached_combined_output = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)
        cached_tensors.extend([
            cached_out_tokens,
            cached_out_weights,
            cached_out_idx,
            cached_combined_output,
        ])

        print(f"Rank {my_rank}: Testing cached mode - second dispatch "
              f"(send_only={bool(dispatch_send_only)})")
        ep_handle.dispatch(
            dispatch_inputs,
            nccl_ep.DispatchOutputs(
                tokens=cached_out_tokens.tensor,
                topk_weights=cached_out_weights.tensor,
                topk_idx=cached_out_idx.tensor,
            ),
            config=dispatch_config,
            stream=stream,
        )
        stream.sync()

        print(f"Rank {my_rank}: Testing cached mode - second combine "
              f"(send_only={bool(combine_send_only)})")
        ep_handle.combine(
            combine_inputs,
            nccl_ep.CombineOutputs(tokens=cached_combined_output.tensor),
            config=nccl_ep.CombineConfig(),
            stream=stream,
        )
        stream.sync()

        # Compare first vs second phase.
        cached_dispatch_errors = 0
        cached_combine_errors = 0

        if not random_mode:
            sec_d0 = np.empty(actual_recv_slots * hidden, dtype=np.uint16)
            d2h(sec_d0, cached_out_tokens.data, stream)
            sec_dw = np.empty(actual_recv_slots * top_k, dtype=np.float32)
            d2h(sec_dw, cached_out_weights.data, stream)
            sec_di = np.empty(actual_recv_slots * top_k, dtype=np.int64)
            d2h(sec_di, cached_out_idx.data, stream)

            sec_co = np.empty(num_tokens * hidden, dtype=np.uint16)
            d2h(sec_co, cached_combined_output.data, stream)
            stream.sync()

            d0_diff = first_d0 != sec_d0
            dw_diff = first_dw != sec_dw
            di_diff = first_di != sec_di
            co_diff = first_co != sec_co

            cached_dispatch_errors = int(d0_diff.sum()) + int(dw_diff.sum()) + int(di_diff.sum())
            cached_combine_errors = int(co_diff.sum())

            for off in np.flatnonzero(d0_diff)[:5]:
                print(f"Rank {my_rank}: Cached dispatch output mismatch at {int(off)}: "
                      f"first={int(first_d0[off])}, second={int(sec_d0[off])}")
            for off in np.flatnonzero(co_diff)[:5]:
                print(f"Rank {my_rank}: Cached combine output mismatch at {int(off)}: "
                      f"first={int(first_co[off])}, second={int(sec_co[off])}")

        if random_mode:
            print(f"Rank {my_rank}: Cached mode completed (random mode, checks skipped)")
        elif cached_dispatch_errors == 0 and cached_combine_errors == 0:
            print(f"Rank {my_rank}: Cached mode verification PASSED")
        else:
            print(f"Rank {my_rank}: Cached mode verification FAILED - "
                  f"dispatch errors: {cached_dispatch_errors}, combine errors: {cached_combine_errors}")
            sys.exit(1)

    # Explicitly refresh routing in-place and run another dispatch. The
    # alternate route preserves the per-rank receive cardinality, including
    # eager output sizes, while changing the source/destination mapping.
    if args.update:
        updated_topk_idx = make_tensor(2, nccl_core.INT64, num_tokens, top_k)
        extra_tensors.append(updated_topk_idx)
        updated_host = np.empty_like(topk_idx_host)
        target_start = local_experts_start
        for i in range(num_tokens):
            for j in range(top_k):
                updated_host[i, j] = target_start + j
        h2d(updated_topk_idx.data, updated_host, stream)
        ep_handle.update(
            updated_topk_idx.tensor,
            layout_info=handle_layout_info if not is_ll else None,
            stream=stream,
        )
        stream.sync()
        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=dispatch_layout,
            config=dispatch_config,
            stream=stream,
        )
        if dispatch_send_only:
            ep_handle.complete(stream=stream)
        stream.sync()
        updated_payload = np.empty(output_tokens.nbytes // 2, dtype=np.uint16)
        d2h(updated_payload, output_tokens.data, stream)
        stream.sync()
        if is_ll and not is_em:
            start = (my_rank * num_tokens) * hidden
        else:
            start = 0
        checked = updated_payload.reshape(-1)[
            start:start + ELEMENTS_TESTED_PER_TOKEN
        ]
        if not np.all(checked == 0x1000 + my_rank):
            raise AssertionError(
                f"rank {my_rank}: Handle.update dispatch did not follow changed routing"
            )
        ep_handle.combine(
            combine_inputs,
            combine_outputs,
            config=nccl_ep.CombineConfig(send_only=combine_send_only),
            stream=stream,
        )
        if combine_send_only:
            ep_handle.complete(stream=stream)
        stream.sync()
        updated_combined = np.empty(num_tokens * hidden, dtype=np.uint16)
        d2h(updated_combined, combined_output.data, stream)
        stream.sync()
        expected_prefix = np.array(
            [
                float_to_bf16(float(combine_scale * (j + 1) * 2))
                for j in range(ELEMENTS_TESTED_PER_TOKEN)
            ],
            dtype=np.uint16,
        )
        if not np.array_equal(updated_combined[:ELEMENTS_TESTED_PER_TOKEN], expected_prefix):
            raise AssertionError(f"rank {my_rank}: Handle.update combine verification failed")
        print(f"Rank {my_rank}: Handle.update changed-routing dispatch/combine PASSED")

    # The mask smoke deliberately avoids timeout injection: it validates every
    # newly bound API deterministically, then restores the group to all-active.
    if args.mask:
        mask_status = make_tensor(1, nccl_core.INT32, n_ranks)
        extra_tensors.append(mask_status)

        def query_mask() -> np.ndarray:
            _ep_bindings.mask_query(ep_group.ptr, mask_status.data, int(stream.handle))
            result = np.empty(n_ranks, dtype=np.int32)
            d2h(result, mask_status.data, stream)
            stream.sync()
            return result

        if not np.array_equal(query_mask(), np.ones(n_ranks, dtype=np.int32)):
            raise AssertionError(f"rank {my_rank}: initial active mask is not all-active")
        requested_mask = np.ones(n_ranks, dtype=np.int32)
        requested_mask[-1] = 0
        _ep_bindings.mask_update(
            ep_group.ptr, requested_mask.ctypes.data, int(stream.handle),
        )
        stream.sync()
        if not np.array_equal(query_mask(), requested_mask):
            raise AssertionError(f"rank {my_rank}: mask update/query mismatch")
        if _ep_bindings.get_async_error(ep_group.ptr) != 0:
            raise AssertionError("mask update unexpectedly raised the async timeout flag")
        all_active = np.ones(n_ranks, dtype=np.int32)
        _ep_bindings.mask_update(
            ep_group.ptr, all_active.ctypes.data, int(stream.handle),
        )
        stream.sync()
        if not np.array_equal(query_mask(), all_active):
            raise AssertionError(f"rank {my_rank}: mask restore update failed")
        _ep_bindings.mask_clean(ep_group.ptr, int(stream.handle))
        _ep_bindings.error_clear(ep_group.ptr)
        stream.sync()
        if not np.array_equal(query_mask(), np.ones(n_ranks, dtype=np.int32)):
            raise AssertionError(f"rank {my_rank}: mask clean did not restore all ranks")
        if _ep_bindings.get_async_error(ep_group.ptr) != 0:
            raise AssertionError("async error flag was not clear after mask recovery")
        print(f"Rank {my_rank}: active-mask API lifecycle PASSED")

    # ===================================================================
    # Cleanup
    # ===================================================================
    for t in cached_tensors:
        free_tensor(t)
    for t in extra_tensors:
        free_tensor(t)
    free_tensor(expert_outputs)
    free_tensor(topk_weights)
    free_tensor(combined_output)
    free_tensor(topk_idx)
    free_tensor(input_tokens)
    free_tensor(output_tokens)
    free_tensor(output_topk_weights)
    free_tensor(output_topk_idx)
    free_tensor(local_tensor_recv_count)
    free_tensor(handle_recv_total)
    free_tensor(handle_expert_counters)
    free_tensor(handle_expert_offsets)

    ep_handle.destroy()
    ep_group.destroy()
    comm.destroy()
    stream.close()
    local_comm.Free()
    print(f"[MPI Rank {my_rank}] Success ")


if __name__ == "__main__":
    main()
