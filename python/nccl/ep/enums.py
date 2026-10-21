# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Pure-Python enum definitions mirroring the NCCL EP C enums.

Defined here (rather than re-exported from a generated bindings module) so
the public API does not depend on any code-generation tool's naming
conventions and so each member can carry a docstring describing its
semantics. Values mirror the corresponding C enums in ``ep_enums.h``
(the quantization recipes live in ``nccl_ep.h``).
"""

from enum import IntEnum

__all__ = [
    "Algorithm",
    "CombineQuantizationRecipe",
    "DispatchQuantizationRecipe",
    "ExpertIdKind",
    "HtFlow",
    "Layout",
    "MetaExchangePolicy",
    "OverflowPolicy",
    "PassDir",
    "ZeroCopyMode",
]


class Algorithm(IntEnum):
    """EP communication algorithm, mirroring :c:type:`ncclEpAlgorithm_t`.

    Set on :py:attr:`GroupConfig.algorithm` before calling
    :py:meth:`Group.create` to select the dispatch/combine path.
    """

    LOW_LATENCY = 0
    """Low-Latency (LL) algorithm. Tuned for minimal per-step latency."""

    HIGH_THROUGHPUT = 1
    """High-Throughput (HT) algorithm. Tuned for peak aggregate bandwidth."""


class Layout(IntEnum):
    """Receive-buffer layout for dispatch/combine, mirroring :c:type:`ncclEpLayout_t`.

    Passed as the ``layout`` argument of :py:meth:`Group.create_handle`
    to control the shape of the dispatch output tensors and the expected
    shape of the combine input tensors. Each layout is supported by a
    subset of the algorithms.
    """

    UNSET = 0
    """Zero-init sentinel — must be overridden. The library does not
    auto-resolve; leaving this value trips an assertion at handle-init
    time."""

    EXPERT_MAJOR = 1
    """LL and HT. Dispatch output ``tokens``:
    ``[num_local_experts, num_ranks * max_dispatch_tokens_per_rank, hidden]``
    under LL, or ``[num_recv_slots, hidden]`` with internal expert grouping
    under HT. Combine accumulates up to ``num_topk`` expert contributions
    per slot; HT expects the caller to have applied the top-k weights
    beforehand, LL applies them in the kernel."""

    RANK_MAJOR = 2
    """LL only. Dispatch output ``tokens``:
    ``[num_ranks, max_dispatch_tokens_per_rank, hidden]``. The caller
    pre-reduces across local experts before combine."""

    FLAT = 3
    """HT only. Dispatch output ``tokens``: ``[num_recv_slots, hidden]`` —
    a single contiguous sequence with no rank or expert structure. The
    caller routes each slot via ``recv_topk_idx`` and pre-reduces before
    combine."""


class PassDir(IntEnum):
    """Pass direction for HT dispatch/combine, mirroring :c:type:`ncclEpPassDir_t`.

    Set on :py:attr:`DispatchConfig.pass_direction` /
    :py:attr:`CombineConfig.pass_direction` to select forward or
    backward pass semantics. HT-only; LL does not distinguish the two and
    ignores the field.
    """

    FWD = 0
    """Forward pass (default)."""

    BWD = 1
    """Backward pass."""


class OverflowPolicy(IntEnum):
    """Recv-overflow policy, mirroring :c:type:`ncclEpOverflowPolicy_t`.

    Set on :py:attr:`GroupConfig.overflow_policy` to choose what happens
    when a rank receives more tokens than ``max_recv_tokens_per_rank``.
    HT only; ignored by LL.
    """

    AUTO = 0
    """Zero-init default. Resolves to :py:attr:`TRAP`."""

    TRAP = 1
    """Device trap (process abort) on recv overflow. Safe for
    capacity-planned deployments."""

    DROP = 2
    """Drop the overflowing tokens and continue. The per-rank true
    (pre-drop) recv total is reported via
    :py:attr:`LayoutInfo.recv_total_counter` when provided. Not available
    in eager mode (``max_recv_tokens_per_rank=0``)."""


class ZeroCopyMode(IntEnum):
    """Token-staging mode, mirroring :c:type:`ncclEpZeroCopyMode_t`.

    Set on :py:attr:`GroupConfig.zero_copy` to control whether
    library-owned staging buffers may be bypassed in favor of
    window-backed user tensors.
    """

    AUTO = 0
    """Zero-init default. Equivalent to :py:attr:`OFF` today: staging
    stays available and compatible tensor windows may still be used
    directly."""

    OFF = 1
    """Same as :py:attr:`AUTO`; set explicitly to pin the behavior."""

    ON = 2
    """Requires windows for the supported direct paths and elides their
    staging: HT needs dispatch-output and combine-input token windows; LL
    can write dispatch output directly but its combine remains staged.
    :py:attr:`DispatchQuantizationRecipe.FWD` requires paired token and
    scale windows."""


class HtFlow(IntEnum):
    """HT dispatch/combine flow, mirroring :c:type:`ncclEpHtFlow_t`."""

    AUTO = 0
    """Use push/pull unless the legacy environment override requests pull/push."""

    PUSH_PULL = 1
    """Push dispatch followed by pull combine. Supports flat and expert-major layouts."""

    PULL_PUSH = 2
    """Pull dispatch followed by push combine. Expert-major and one LSA domain only."""


class MetaExchangePolicy(IntEnum):
    """HT metadata exchange policy, mirroring :c:type:`ncclEpMetaExchangePolicy_t`."""

    AUTO = 0
    """Fuse metadata with dispatch when supported; otherwise use the existing fallback."""

    REQUIRE_FUSED = 1
    """Reject handles or updates that cannot keep metadata exchange fused with dispatch."""

    FORCE_SCAN = 2
    """Force the legacy routing-map exchange and scan path."""


class ExpertIdKind(IntEnum):
    """Numbering of dispatch's ``recv_topk_idx`` output, mirroring
    :c:type:`ncclEpExpertIdKind_t`.

    Set on :py:attr:`LayoutInfo.recv_topk_idx_kind`. Applies to the
    layouts that populate ``recv_topk_idx`` (LL rank-major, HT flat).
    Slots not routed to this rank are ``-1`` under either numbering.
    """

    AUTO = 0
    """Zero-init default; resolves to :py:attr:`LOCAL` today. The
    resolved value may change in a future release without an ABI break —
    pin :py:attr:`LOCAL` or :py:attr:`GLOBAL` for a stable contract."""

    LOCAL = 1
    """Expert id in ``[0, num_local_experts)``, i.e.
    ``global_id - rank * num_local_experts``."""

    GLOBAL = 2
    """Unmodified wire-format global expert id, in
    ``[rank * num_local_experts, (rank + 1) * num_local_experts)`` for
    this rank's local experts."""


class DispatchQuantizationRecipe(IntEnum):
    """Dispatch quantization recipe, mirroring :c:type:`ncclEpDispQuant_t`.

    Set on :py:attr:`DispatchConfig.quantization_recipe`. The contracts
    apply to both HT and LL; only the layouts differ. See
    ``nccl_ep/include/nccl_ep.h`` for the full per-recipe rules.
    """

    NONE = 0
    """Zero-init default. The token tensor is transported in its declared
    dtype; both ``scales`` tensors must be absent."""

    FWD = 1
    """Forward the caller's scales as-is rather than generating them
    (unrelated to :py:class:`PassDir` — this applies to both passes).
    ``inputs.tokens`` and ``inputs.scales`` are 2D and their physical
    bytes are forwarded without conversion: output dtypes and row widths
    must match the inputs, ``round_scales`` must be 0, and rows plus their
    storage base (or window offset) must be 16-byte aligned. Tokens may be
    FP32/FP16/BF16/FP8 or packed FP4 (``ncclFloat4x2``); scales
    additionally accept raw ``uint8``. Packed FP4 tokens are declared as
    ``[num_tokens, H/2]`` for logical hidden size ``H``, so the alignment
    rule requires ``H`` to be a multiple of 32."""

    DS_FP8E3M4 = 2
    """DeepSeek FP8 recipe; LL-only internal quantization. The caller
    supplies BF16 tokens; dispatch emits E4M3 token bytes and generated
    FP32 scales, one per 128 token elements. Hidden must be divisible by
    512. ``inputs.scales`` must be absent and ``outputs.scales`` is
    required."""


class CombineQuantizationRecipe(IntEnum):
    """Combine quantization recipe, mirroring :c:type:`ncclEpCombQuant_t`.

    Set on :py:attr:`CombineConfig.quantization_recipe`.
    """

    NONE = 0
    """Zero-init default. Unquantized transport."""

    NVFP4 = 1
    """EXPERIMENTAL: LL-only BF16 expert-output transport. Its API
    contract, supported shapes, and numerical behavior may change before
    graduation. The caller supplies FP32 global scales through
    :py:attr:`CombineInputs.scales`; the kernel follows the DeepEP-LL
    NVFP4 pack/dequantize contract."""

    MXFP8 = 2
    """HT expert-major local-permute transport for a single LSA team.
    The caller supplies BF16 tokens and no scales; NCCL EP internally
    quantizes to FP8 E4M3 with E8M0 scales (block size 32), transports the
    packed row over NVLink, and produces BF16 output. Hidden size must be a
    multiple of 512 and the library must be built with CUDA 12.8 or newer."""
