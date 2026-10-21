# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""EP group lifecycle: ``Group`` and its configuration ``GroupConfig``."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from nccl._extensions._binding_helpers import binding_dataclass

from nccl.core.cuda import get_stream_ptr
from nccl.core.typing import NcclInvalid, NcclStreamSpec

from nccl._extensions.bindings import nccl_ep as _ep_bindings
from nccl.ep.allocator import AllocConfig
from nccl.ep.enums import (
    Algorithm,
    HtFlow,
    Layout,
    MetaExchangePolicy,
    OverflowPolicy,
    ZeroCopyMode,
)
from nccl.ep.handle import Handle, HandleConfig, HandleState, LayoutInfo

if TYPE_CHECKING:
    from nccl.core import Communicator
    from nccl.ep.tensor import Tensor


__all__ = ["Group", "GroupConfig"]


@binding_dataclass(_ep_bindings.GroupConfig)
class GroupConfig:
    """Pythonic configuration for :py:meth:`Group.create`.

    Mirrors :c:struct:`ncclEpGroupConfig_t`. Fields left at their
    defaults (``0`` or ``LOW_LATENCY``/``AUTO``) forward as
    ``NCCL_EP_AUTO`` where applicable.

    Attributes:
        algorithm: Dispatch/combine algorithm. Default: ``LOW_LATENCY``.
        num_experts: Total number of experts across all ranks. Required.
        max_dispatch_tokens_per_rank: Maximum tokens any single rank will
            dispatch. Required for both LL and HT: must be > 0 and
            identical on every rank.
        max_token_bytes: Upper bound on per-token row bytes, covering
            dispatch and combine. Under a quantization recipe the
            quantized token data *and* its scales must fit in this
            budget. HT requires a multiple of 16. Required.
        rdma_buffer_size: LL RDMA staging buffer size in bytes. 0 lets
            the library size it, possibly lazily at handle creation. An
            explicit value is allocated exactly and never grown — handle
            creation fails if the requested layout does not fit.
        num_qp_per_rank: Number of QPs per rank. 0 selects auto.
        num_channels: Channels per rank. 0 selects auto. In HT each
            channel occupies 2 SMs.
        max_recv_tokens_per_rank: Per-rank recv-slot budget. An explicit
            HT value must be ``>= max_dispatch_tokens_per_rank``, and
            expert-major must also account for a token duplicated to
            several local experts. 0 selects HT eager mode, where the
            caller sizes dispatch outputs from
            :py:attr:`LayoutInfo.recv_total_counter`; eager mode supports
            neither ``OverflowPolicy.DROP`` nor CUDA Graph capture of
            dispatch. LL ignores this field.
        max_num_sms: Shared dispatch/combine SM budget. 0 selects an
            algorithm-dependent default.
        dispatch_num_sms: Dispatch SM budget. 0 inherits the resolved
            shared budget. NCCL_EP_DISPATCH_SMS overrides it.
        combine_num_sms: Combine SM budget. 0 inherits the resolved
            shared budget. NCCL_EP_COMBINE_SMS overrides it.
        num_attn_ranks: Number of leading attention ranks for AFD. Non-zero
            values require universal LL kernels, rank-major layout, and a
            square attention-to-expert-rank partition. Default: 0.
        alloc: Device allocator hooks. Default
            :class:`AllocConfig` selects ``cudaMalloc``/``cudaFree``.
        enable_mask: Enable active-mask support for fault tolerance
            (LL mode only). When ``True``, a per-rank mask buffer is
            allocated; remote ranks that time out during dispatch or
            combine are skipped rather than tripping a GPU trap, and a
            host-visible error flag is set. The mask and async-error
            APIs that inspect it (``ncclEpMaskQuery`` and friends) are not
            wrapped in Python. Default: ``False``.
        timeout_ns: GPU-side wait-loop timeout in nanoseconds. ``0``
            selects the library default (~100 s). Setting too low risks
            false positives. The ``NCCL_EP_TIMEOUT_MS`` env var
            overrides this field at group creation.
        zero_copy: Whether library-owned dispatch/combine staging may be
            bypassed in favor of window-backed tensors; see
            :py:class:`ZeroCopyMode`. Default ``AUTO``.
        overflow_policy: Policy applied when a rank receives more tokens
            than ``max_recv_tokens_per_rank``. HT only; ignored by LL.
            ``DROP`` is unavailable in eager mode. Default ``AUTO``
            resolves to ``TRAP``.
        ht_flow: HT dispatch/combine flow. An explicit value takes precedence
            over legacy environment controls; under ``AUTO``, the environment
            may override the built-in push-pull default. All ranks must resolve
            to the same flow. Default ``AUTO``.
        meta_exchange_policy: Whether HT metadata exchange is automatic, required
            to be fused with dispatch, or forced to scan. An explicit policy
            takes precedence over legacy environment controls; all ranks must
            resolve to the same policy. Default ``AUTO``.
        num_topk: Upper bound on per-token top-k across all handles of
            this group. Required for HT eager mode with Expert-Major
            layout; 0 means unset.

    See Also:
        NCCL EP ``ncclEpGroupConfig_t``:
        ``nccl_ep/include/nccl_ep.h``
    """

    algorithm: Algorithm = Algorithm.LOW_LATENCY
    num_experts: int = 0
    max_dispatch_tokens_per_rank: int = 0
    max_recv_tokens_per_rank: int = 0
    max_token_bytes: int = 0
    rdma_buffer_size: int = 0
    num_qp_per_rank: int = 0
    num_channels: int = 0
    max_num_sms: int = 0
    dispatch_num_sms: int = 0
    combine_num_sms: int = 0
    ht_flow: HtFlow = HtFlow.AUTO
    meta_exchange_policy: MetaExchangePolicy = MetaExchangePolicy.AUTO
    num_attn_ranks: int = 0
    alloc: AllocConfig = field(default_factory=AllocConfig)
    enable_mask: bool = False
    timeout_ns: int = 0
    zero_copy: ZeroCopyMode = ZeroCopyMode.AUTO
    overflow_policy: OverflowPolicy = OverflowPolicy.AUTO
    num_topk: int = 0


class Group:
    """A NCCL EP group built on top of an existing :class:`Communicator`.

    Construct via :meth:`create`; release with :meth:`destroy`.
    """

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    @classmethod
    def create(
        cls,
        comm: Communicator,
        config: GroupConfig,
    ) -> Group:
        """Collectively create an EP group across all ranks of *comm*.

        Args:
            comm: An initialized :class:`nccl.core.Communicator`.
            config: Filled-in :class:`GroupConfig` describing the
                group. Custom allocators live in ``config.alloc``
                (:class:`AllocConfig`) — see :mod:`nccl.ep.allocator`
                for usage and lifetime requirements.

        See Also:
            :meth:`destroy`.
        """
        ptr = _ep_bindings.create_group(comm.ptr, config._lowpp.ptr)  # type: ignore[attr-defined]
        return cls(ptr)

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: Group is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpGroup_t`` address."""
        self._check_valid("read ptr")
        return self._ptr

    def create_handle(
        self,
        layout: Layout,
        topk_idx: Tensor,
        *,
        layout_info: LayoutInfo | None = None,
        config: HandleConfig | None = None,
        stream: NcclStreamSpec,
    ) -> Handle:
        """Collectively create and initialize a :class:`Handle` over this group.

        HT mode performs metadata exchange as part of this call.

        Args:
            layout: Receive-buffer layout. Required — must not be
                :py:attr:`Layout.UNSET`. HT supports ``FLAT`` /
                ``EXPERT_MAJOR``; LL supports ``EXPERT_MAJOR`` /
                ``RANK_MAJOR``.
            topk_idx: Top-k expert indices for this step
                (shape ``[num_tokens, top_k]``, int64).
            layout_info: Optional :class:`LayoutInfo`. HT: supply
                ``recv_total_counter`` to learn the actual recv count —
                required to size dispatch outputs in eager mode
                (``max_recv_tokens_per_rank=0``); expert-major
                additionally reports ``expert_counters`` and
                ``expert_offsets``. LL mode: must be ``None``.
            config: Optional :class:`HandleConfig`; ``None`` forwards
                NULL (library defaults).
            stream: CUDA stream for the launch.
        """
        self._check_valid("create_handle")
        ptr = _ep_bindings.create_handle(
            self._ptr,
            int(layout),
            topk_idx.ptr,
            layout_info._lowpp.ptr if layout_info is not None else 0,  # type: ignore[attr-defined]
            config._lowpp.ptr if config is not None else 0,  # type: ignore[attr-defined]
            get_stream_ptr(stream),
        )
        return Handle(ptr)

    def handle_mem_size(
        self,
        layout: Layout,
        *,
        config: HandleConfig | None = None,
        num_topk: int = -1,
    ) -> int:
        """Return device bytes required for caller-owned handle memory.

        ``num_topk`` must be positive for LL and may remain ``-1`` for HT.
        Allocate a one-dimensional ``ncclUint8`` :class:`Tensor` with at
        least this many elements before calling :meth:`init_handle`.
        """
        self._check_valid("query handle memory size")
        return int(
            _ep_bindings.handle_mem_size(
                self._ptr,
                int(layout),
                config._lowpp.ptr if config is not None else 0,  # type: ignore[attr-defined]
                int(num_topk),
            )
        )

    def init_handle(
        self,
        layout: Layout,
        *,
        config: HandleConfig | None = None,
        num_topk: int = -1,
        handle_mem: Tensor | None = None,
    ) -> Handle:
        """Allocate routing state without binding this step's top-k tensor.

        Call :meth:`Handle.update` before dispatch/combine. Passing
        ``handle_mem=None`` lets NCCL EP allocate the routing buffer; otherwise
        the supplied 1D ``ncclUint8`` tensor is borrowed and retained by the
        returned handle. With an auto-sized group RDMA buffer this call may be
        collective and must be made in rank-consistent order.
        """
        self._check_valid("initialize handle")
        ptr = _ep_bindings.init_handle(
            self._ptr,
            int(layout),
            config._lowpp.ptr if config is not None else 0,  # type: ignore[attr-defined]
            int(num_topk),
            handle_mem.ptr if handle_mem is not None else 0,
        )
        return Handle(ptr, handle_mem=handle_mem)

    def import_handle(
        self,
        layout: Layout,
        state: HandleState,
        handle_mem: Tensor,
        *,
        config: HandleConfig | None = None,
    ) -> Handle:
        """Import an HT handle from an already-prepared routing buffer.

        The buffer must have been prepared with matching group, layout,
        configuration, and :class:`HandleState`. Import is local: it performs
        no CUDA launch, metadata scan, or device copy. The returned handle
        retains ``handle_mem`` for its lifetime.
        """
        self._check_valid("import handle")
        ptr = _ep_bindings.import_handle(
            self._ptr,
            int(layout),
            config._lowpp.ptr if config is not None else 0,  # type: ignore[attr-defined]
            state._lowpp.ptr,  # type: ignore[attr-defined]
            handle_mem.ptr,
        )
        return Handle(ptr, handle_mem=handle_mem)

    def destroy(self) -> None:
        """Release the group. Subsequent operations on this object are invalid."""
        if self._ptr:
            _ep_bindings.group_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        if self._ptr:
            return f"<Group ptr={self._ptr:#x}>"
        return "<Group destroyed>"
