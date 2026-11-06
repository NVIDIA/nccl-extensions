# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault-tolerant torch.distributed backend.

``FTProcessGroup`` uses a separate NCCL communicator for device-side FT
collectives. Eligible symmetric tensors take that path; other calls use the
inherited NCCL implementation and emit a warning. Importing the package
registers the ``ft_nccl`` backend. See the package README for usage and the
eligibility rules below for supported types and layouts.
"""

import ctypes
import datetime
import os
import sys
import warnings
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.library

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from .ft_wrapper import (
    FTLibrary, NCCLBaseLibrary, CUDARuntime,
    FT_OK, ncclUniqueId,
    FT_DTYPE_FLOAT32, FT_DTYPE_FLOAT16, FT_DTYPE_BFLOAT16,
)


# ---------------------------------------------------------------------------
# Warning helper
# ---------------------------------------------------------------------------

def _ft_warn(op: str, reason: str) -> None:
    """Emit a UserWarning when a collective uses the NCCL (non-FT) path."""
    warnings.warn(
        f"FTProcessGroup.{op}(): {reason}; using NCCL (not fault-tolerant).",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# DLPack wrapper for ncclMemAlloc pointers
# ---------------------------------------------------------------------------

_DLDEVICE_CUDA = 2   # kDLCUDA

# Dtypes supported by the FT AllGather and AllReduce paths.
_FT_AG_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})
_FT_AR_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})

# DLPack dtype encoding: (kDL* code, bits). Integer types support symmetric
# allocations but are not eligible for the typed AllGather/AllReduce paths.
_TORCH_TO_DL: dict = {
    torch.float32:  (2, 32),
    torch.float16:  (2, 16),
    torch.bfloat16: (4, 16),
    torch.int64:    (0, 64),
    torch.int32:    (0, 32),
    torch.int16:    (0, 16),
    torch.int8:     (0, 8),
    torch.uint8:    (1, 8),
}

# Map torch dtype → FTDataType enum value for AllReduce dispatch.
_TORCH_TO_FT_AR_DTYPE: dict = {
    torch.float32:  FT_DTYPE_FLOAT32,
    torch.float16:  FT_DTYPE_FLOAT16,
    torch.bfloat16: FT_DTYPE_BFLOAT16,
}

# ---------------------------------------------------------------------------
# Module-level active process group for custom ops
# ---------------------------------------------------------------------------
# torch.library.custom_op cannot accept arbitrary Python objects as arguments,
# so the active FTProcessGroup is stored here and retrieved by the custom ops.

_active_pg: Optional['FTProcessGroup'] = None


def set_active_pg(pg: Optional['FTProcessGroup']) -> None:
    """Set the module-level active FTProcessGroup used by custom ops.

    Must be called before using ``torch.ops.ft_collective.allgather`` or
    ``torch.ops.ft_collective.allreduce`` (e.g. inside a torch.compiled
    function or CUDA graph).
    """
    global _active_pg
    _active_pg = pg


# ---------------------------------------------------------------------------
# Custom ops — allow torch.compile and CUDA graphs to treat FT collectives
# as opaque nodes, preventing graph breaks.
#
# Both ops are asynchronous (fire-and-forget on pg._stream).  The caller must
# call pg._cuda.stream_synchronize(pg._stream) or use pg._ft_work().wait()
# before reading the output tensor.
# ---------------------------------------------------------------------------

@torch.library.custom_op("ft_collective::allgather", mutates_args=("output",))
def ft_allgather_op(send: torch.Tensor, output: torch.Tensor) -> None:
    """FT AllGather: gather ``send`` (symmetric) from all ranks into ``output``.

    ``send``   : symmetric tensor from ``pg.empty()`` — shape (numel,)
    ``output`` : pre-allocated tensor of shape (nranks, numel)

    The active FTProcessGroup must be set via ``set_active_pg()`` before use.
    Launches asynchronously on ``pg._stream``; call
    ``pg._cuda.stream_synchronize(pg._stream)`` before reading ``output``.
    """
    pg = _active_pg
    if pg is None:
        raise RuntimeError(
            "ft_collective::allgather: no active FTProcessGroup. "
            "Call set_active_pg(pg) first."
        )
    t = send
    if not pg._is_ft_eligible_ag(t):
        raise RuntimeError(
            f"ft_collective::allgather: tensor is not FT-eligible: "
            f"{pg._fallback_reason(t, 'allgather')}"
        )
    win_send, send_offset = pg._window_spec(t)
    chunk_bytes = t.numel() * t.element_size()
    float_equiv = chunk_bytes // 4
    # Per-rank recv slice is padded to a 16-byte stride (matches the kernel's
    # FT_RECV_STRIDE) so the device copies stay vectorized for any count.
    stride_bytes = ((chunk_bytes + 15) // 16) * 16
    # d_recv is intentionally NOT zeroed: every present rank fully overwrites
    # its slice, so the memset (which scaled with nranks*count) was dead work
    # on the critical path.  After a fault an absent rank's slice is stale;
    # callers consult the result mask to know which slices are valid.
    pg._order_after_current()   # wait for the caller\'s stream
    pg._cuda.memset_async(pg._d_flags, 0, pg._size * 8,           pg._stream)
    pg._ft.allgather(
        pg._handle,
        win_send,        send_offset,
        pg._win_recv,    0,
        pg._win_flags,
        float_equiv,
        timeout_us=pg._timeout_us,
        stream=pg._stream,
    )
    # output is a contiguous (nranks, numel) tensor → gather the padded recv
    # buffer into it with a single strided 2D copy when layout permits, else
    # fall back to per-rank copies.
    base = output[0].data_ptr()
    contiguous = all(
        output[r].data_ptr() == base + r * chunk_bytes
        and output[r].numel() * output[r].element_size() == chunk_bytes
        for r in range(pg._size)
    )
    if contiguous:
        pg._cuda.memcpy_2d_d2d_async(
            base,         chunk_bytes,    # dst, dpitch (packed)
            pg._d_recv,   stride_bytes,   # src, spitch (padded)
            chunk_bytes,  pg._size,        # width, height
            pg._stream,
        )
    else:
        for r in range(pg._size):
            pg._cuda.memcpy_d2d_async(
                output[r].data_ptr(),
                pg._d_recv + r * stride_bytes,
                chunk_bytes,
                pg._stream,
            )


@ft_allgather_op.register_fake
def _ft_allgather_op_fake(send: torch.Tensor, output: torch.Tensor) -> None:
    pass


@torch.library.custom_op("ft_collective::alltoall", mutates_args=("output",))
def ft_alltoall_op(send: torch.Tensor, output: torch.Tensor) -> None:
    """FT uniform all-to-all (Phase 0): chunk j of ``send`` goes to rank j.

    ``send``   : symmetric tensor from ``pg.empty()`` — nRanks contiguous
                 per-destination chunks (shape (nranks, per_dest) or flat).
    ``output`` : pre-allocated (nranks, per_dest); slot i = chunk from rank i.

    The active FTProcessGroup must be set via ``set_active_pg()``. Launches
    asynchronously on ``pg._stream``; sync before reading ``output``.
    """
    pg = _active_pg
    if pg is None:
        raise RuntimeError(
            "ft_collective::alltoall: no active FTProcessGroup. "
            "Call set_active_pg(pg) first."
        )
    t = send
    if not pg._is_ft_eligible_a2a(t):
        raise RuntimeError(
            "ft_collective::alltoall: send is not FT-eligible (must be a "
            "symmetric tensor whose per-destination count <= max_count)."
        )
    n = pg._size
    win_send, send_offset = pg._window_spec(t)
    per_dest_bytes = (t.numel() // n) * t.element_size()
    float_equiv    = per_dest_bytes // 4
    # Per-source recv slice padded to a 16-byte stride (matches FT_RECV_STRIDE).
    stride_bytes   = ((per_dest_bytes + 15) // 16) * 16
    # d_recv is NOT zeroed: every present rank overwrites its slot; absent-rank
    # slots are stale and reported via the result mask.
    pg._order_after_current()   # wait for the caller\'s stream
    pg._cuda.memset_async(pg._d_flags, 0, pg._size * 8, pg._stream)
    pg._ft.alltoall(
        pg._handle,
        win_send,       send_offset,
        pg._win_recv,   0,
        pg._win_flags,
        float_equiv,
        timeout_us=pg._timeout_us,
        stream=pg._stream,
    )
    # Scatter the padded recv slots into the packed (nranks, per_dest) output —
    # identical layout to AllGather's recv, so the same strided 2D copy applies.
    base = output[0].data_ptr()
    contiguous = all(
        output[r].data_ptr() == base + r * per_dest_bytes
        and output[r].numel() * output[r].element_size() == per_dest_bytes
        for r in range(n)
    )
    if contiguous:
        pg._cuda.memcpy_2d_d2d_async(
            base,          per_dest_bytes,   # dst, dpitch (packed)
            pg._d_recv,    stride_bytes,     # src, spitch (padded)
            per_dest_bytes, n,               # width, height
            pg._stream,
        )
    else:
        for r in range(n):
            pg._cuda.memcpy_d2d_async(
                output[r].data_ptr(),
                pg._d_recv + r * stride_bytes,
                per_dest_bytes,
                pg._stream,
            )


@ft_alltoall_op.register_fake
def _ft_alltoall_op_fake(send: torch.Tensor, output: torch.Tensor) -> None:
    pass


@torch.library.custom_op("ft_collective::allreduce", mutates_args=("tensor",))
def ft_allreduce_op(tensor: torch.Tensor) -> None:
    """FT AllReduce (in-place SUM) on a symmetric tensor from ``pg.empty()``.

    ``tensor`` : symmetric tensor from ``pg.empty()`` — result written in-place.

    The active FTProcessGroup must be set via ``set_active_pg()`` before use.
    Launches asynchronously on ``pg._stream``; call
    ``pg._cuda.stream_synchronize(pg._stream)`` before reading ``tensor``.
    """
    pg = _active_pg
    if pg is None:
        raise RuntimeError(
            "ft_collective::allreduce: no active FTProcessGroup. "
            "Call set_active_pg(pg) first."
        )
    t = tensor
    if not pg._is_ft_eligible_ar(t):
        raise RuntimeError(
            f"ft_collective::allreduce: tensor is not FT-eligible: "
            f"{pg._fallback_reason(t, 'allreduce')}"
        )
    win_send, send_offset = pg._window_spec(t)
    ft_dtype    = _TORCH_TO_FT_AR_DTYPE[t.dtype]
    # d_recv is intentionally NOT zeroed: present ranks fully overwrite their
    # slices and the reduction skips absent ranks via the result mask, so the
    # memset (which scaled with nranks*count) was dead work on the hot path.
    pg._order_after_current()   # wait for the caller\'s stream
    pg._cuda.memset_async(pg._d_flags, 0, pg._size * 8,           pg._stream)
    pg._ft.allreduce(
        pg._handle,
        win_send,        send_offset,
        pg._win_recv,    0,
        pg._win_flags,
        t.data_ptr(),
        t.numel(),
        timeout_us=pg._timeout_us,
        dtype=ft_dtype,
        stream=pg._stream,
    )


@ft_allreduce_op.register_fake
def _ft_allreduce_op_fake(tensor: torch.Tensor) -> None:
    pass


@torch.library.custom_op("ft_collective::alltoallv",
                         mutates_args=("output", "recv_counts"))
def ft_alltoallv_op(send: torch.Tensor, output: torch.Tensor,
                    send_counts: torch.Tensor, recv_counts: torch.Tensor,
                    recv_capacity: int, row_size: int) -> None:
    """FT variable all-to-allv, auto-recv-sizing (device-resident, opaque).

    ``send``        : symmetric tensor from ``pg.empty()`` (token-major payload).
    ``output``      : DEVICE tensor; source ``s`` packed at ``s*recv_capacity``
                      rows (capacity mode) — mutated.
    ``send_counts`` : DEVICE int32[nRanks], tokens/rows to each destination.
    ``recv_counts`` : DEVICE int32[nRanks], DISCOVERED tokens per source —
                      mutated (0 for a timed-out source; consult the mask).
    ``recv_capacity``: max rows any one source may deliver (output slot stride).
    ``row_size``    : elements per row (hidden); 1 = element counts.

    Device-resident (no host sync): displacements are derived on-device and the
    receiver discovers the recv counts. Set the active pg via ``set_active_pg()``.
    Async on ``pg._stream``; sync before reading ``output`` / ``recv_counts``.
    """
    pg = _active_pg
    if pg is None:
        raise RuntimeError("ft_collective::alltoallv: no active FTProcessGroup. "
                           "Call set_active_pg(pg) first.")
    if not pg._is_registered_tensor(send):
        raise RuntimeError("ft_collective::alltoallv: send is not symmetric "
                           "(use pg.empty() / pg.register_symmetric_tensor()).")
    win_send, send_offset = pg._window_spec(send)
    dtype_code = _TORCH_TO_FT_AR_DTYPE[send.dtype]
    # Fixed per-source scratch slot stride (holds any dtype up to max_count elems).
    recv_stride_bytes = ((pg._max_count * 4 + 15) // 16) * 16
    pg._order_after_current()   # wait for the caller\'s stream
    pg._cuda.memset_async(pg._d_flags, 0, pg._size * 8, pg._stream)
    pg._ft.alltoallv(
        pg._handle,
        win_send,       send_offset,
        pg._win_recv,   0,
        pg._win_flags,
        send_counts.data_ptr(), 0,          # d_send_counts, d_send_displs=NULL (derive)
        recv_stride_bytes,
        recv_capacity,                      # max_send_count hint = capacity (no host reduce)
        output.data_ptr(),
        0, 0,                               # d_recv_counts/displs (packed mode; unused)
        recv_counts.data_ptr(), recv_capacity, row_size,   # auto mode: discover + capacity + rows
        dtype_code,
        timeout_us=pg._timeout_us,
        stream=pg._stream,
    )


@ft_alltoallv_op.register_fake
def _ft_alltoallv_op_fake(send: torch.Tensor, output: torch.Tensor,
                          send_counts: torch.Tensor, recv_counts: torch.Tensor,
                          recv_capacity: int, row_size: int) -> None:
    pass


@torch.library.custom_op("ft_collective::allgatherv",
                         mutates_args=("output", "recv_counts"))
def ft_allgatherv_op(send: torch.Tensor, output: torch.Tensor,
                     send_counts: torch.Tensor, recv_counts: torch.Tensor,
                     recv_capacity: int, row_size: int) -> None:
    """FT variable all-gather, auto-recv-sizing + packed output (opaque).

    ``send``        : symmetric tensor from ``pg.empty()``; this rank contributes
                      ``send_counts[myRank]`` rows (token-major).
    ``output``      : DEVICE tensor; the PACKED concatenation of every rank's chunk
                      in rank order (>= sum of counts rows) — mutated.
    ``send_counts`` : DEVICE int32[nRanks]; only ``[myRank]`` is read (this rank's
                      row count). Other ranks' sizes are DISCOVERED, not required.
    ``recv_counts`` : DEVICE int32[nRanks], discovered rows per source — mutated
                      (0 for a timed-out source; consult the mask).
    ``recv_capacity``: max rows any one source may deliver (scratch slot bound).
    ``row_size``    : elements per row (hidden); 1 = element counts.

    Device-resident, no host sync. Set the active pg via ``set_active_pg()``.
    Async on ``pg._stream``; sync before reading ``output`` / ``recv_counts``.
    """
    pg = _active_pg
    if pg is None:
        raise RuntimeError("ft_collective::allgatherv: no active FTProcessGroup. "
                           "Call set_active_pg(pg) first.")
    if not pg._is_registered_tensor(send):
        raise RuntimeError("ft_collective::allgatherv: send is not symmetric "
                           "(use pg.empty() / pg.register_symmetric_tensor()).")
    win_send, send_offset = pg._window_spec(send)
    row_bytes = row_size * send.element_size()
    recv_stride = ((recv_capacity * row_bytes + 15) // 16) * 16
    slot_max  = ((pg._max_count * 4 + 15) // 16) * 16
    if recv_stride > slot_max:
        raise RuntimeError(
            f"ft_collective::allgatherv: recv_capacity*row_bytes ({recv_stride} B) "
            f"exceeds the handle scratch slot ({slot_max} B); lower recv_capacity "
            f"or raise FT_NCCL_MAX_COUNT.")
    pg._order_after_current()   # wait for the caller\'s stream
    pg._cuda.memset_async(pg._d_flags, 0, pg._size * 8, pg._stream)
    pg._ft.allgatherv(
        pg._handle,
        win_send,      send_offset,
        pg._win_recv,  0,
        pg._win_flags,
        send_counts.data_ptr(),
        output.data_ptr(),
        recv_counts.data_ptr(),
        recv_capacity, row_bytes, recv_stride,
        recv_capacity,                      # max_send_count hint = capacity (rows)
        timeout_us=pg._timeout_us,
        stream=pg._stream,
    )


@ft_allgatherv_op.register_fake
def _ft_allgatherv_op_fake(send: torch.Tensor, output: torch.Tensor,
                           send_counts: torch.Tensor, recv_counts: torch.Tensor,
                           recv_capacity: int, row_size: int) -> None:
    pass


@torch.library.custom_op("ft_collective::reduce_scatterv", mutates_args=("output",))
def ft_reduce_scatterv_op(send: torch.Tensor, output: torch.Tensor,
                          counts: torch.Tensor,
                          recv_stride: int, max_shard_count: int) -> None:
    """FT variable reduce-scatter, typed SUM (opaque, device-resident).

    ``send``   : symmetric FULL input from ``pg.empty()`` (sum-of-counts elems).
    ``output`` : DEVICE tensor; this rank's reduced shard (counts[myRank] elems),
                 summed over surviving contributors — mutated.
    ``counts`` : DEVICE int32[nRanks] shard-size partition (known to all ranks).
    ``recv_stride``: per-contributor scratch slot stride in BYTES (>= max_shard *
                 element_size, 16-byte multiple).
    ``max_shard_count``: max shard size (elements) — grid-size hint.

    Shard offsets are derived on-device. Set the active pg via ``set_active_pg()``.
    Async on ``pg._stream``; sync before reading ``output``.
    """
    pg = _active_pg
    if pg is None:
        raise RuntimeError("ft_collective::reduce_scatterv: no active "
                           "FTProcessGroup. Call set_active_pg(pg) first.")
    if not pg._is_registered_tensor(send):
        raise RuntimeError("ft_collective::reduce_scatterv: send is not symmetric "
                           "(use pg.empty() / pg.register_symmetric_tensor()).")
    win_send, send_offset = pg._window_spec(send)
    dtype_code = _TORCH_TO_FT_AR_DTYPE[send.dtype]
    pg._order_after_current()   # wait for the caller\'s stream
    pg._cuda.memset_async(pg._d_flags, 0, pg._size * 8, pg._stream)
    pg._ft.reduce_scatterv(
        pg._handle,
        win_send,      send_offset,
        pg._win_recv,  0,
        pg._win_flags,
        counts.data_ptr(), 0,               # d_counts, d_displs=NULL (derive)
        output.data_ptr(),
        recv_stride, max_shard_count, dtype_code,
        timeout_us=pg._timeout_us,
        stream=pg._stream,
    )


@ft_reduce_scatterv_op.register_fake
def _ft_reduce_scatterv_op_fake(send: torch.Tensor, output: torch.Tensor,
                                counts: torch.Tensor,
                                recv_stride: int, max_shard_count: int) -> None:
    pass


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data",        ctypes.c_void_p),
        ("device",      _DLDevice),
        ("ndim",        ctypes.c_int),
        ("dtype",       _DLDataType),
        ("shape",       ctypes.POINTER(ctypes.c_int64)),
        ("strides",     ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DeleterFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor",   _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter",     _DeleterFn),
    ]


class _SymTensor:
    """DLPack producer wrapping an ncclMemAlloc device pointer.

    Call ``torch.from_dlpack(sym)`` to obtain a 1-D torch.Tensor with the
    given dtype.  The underlying memory is NOT freed on GC; FTProcessGroup
    owns the lifecycle.  Instances must be stored in
    ``FTProcessGroup._sym_objects`` to stay alive as long as any derived
    tensor exists.
    """

    def __init__(self, ptr: int, numel: int, device_id: int,
                 dtype: torch.dtype = torch.float32) -> None:
        dl_code, dl_bits = _TORCH_TO_DL[dtype]
        self._shape_arr = (ctypes.c_int64 * 1)(numel)

        self._managed = _DLManagedTensor()
        dl = self._managed.dl_tensor
        dl.data        = ptr
        dl.device      = _DLDevice(_DLDEVICE_CUDA, device_id)
        dl.ndim        = 1
        dl.dtype       = _DLDataType(dl_code, dl_bits, 1)
        dl.shape       = self._shape_arr
        dl.strides     = None
        dl.byte_offset = 0

        @_DeleterFn
        def _noop(_ctx):
            pass

        self._deleter_fn = _noop
        self._managed.deleter = self._deleter_fn

    def __dlpack__(self, stream=None):
        ctypes.pythonapi.PyCapsule_New.restype  = ctypes.py_object
        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p
        ]
        return ctypes.pythonapi.PyCapsule_New(
            ctypes.addressof(self._managed), b"dltensor", None
        )

    def __dlpack_device__(self):
        return (_DLDEVICE_CUDA, self._managed.dl_tensor.device.device_id)


# ---------------------------------------------------------------------------
# FTWork
# ---------------------------------------------------------------------------

class FTWork(dist.Work):
    """Represents an in-flight FT collective.

    ``wait()`` uses stream ordering — it makes the current CUDA stream wait
    for the FT kernel's completion event without blocking the host CPU.  This
    matches the semantics of standard ProcessGroupNCCL: subsequent GPU
    operations on the current stream are guaranteed to see the result, but the
    host thread is not blocked.

    Completion includes partial results after a peer timeout. Inspect the
    process group's error flag and result mask for peer participation.
    """

    def __init__(self, cuda: CUDARuntime, event: int, owner) -> None:
        super().__init__()
        self._cuda = cuda
        self._event = event
        self._owner = owner  # Keep stream and collective resources alive.

    def wait(self, timeout=None) -> bool:
        if self._event is not None:
            current = torch.cuda.current_stream(self._owner._device).cuda_stream
            self._cuda.stream_wait_event(current, self._event)
        return True

    def is_completed(self) -> bool:
        if self._event is None:
            return True
        return self._cuda.event_query(self._event)

    def is_success(self) -> bool:
        # Peer timeouts produce valid partial results; their status is per group.
        return self.is_completed()

    def __del__(self):
        try:
            if self._event is not None:
                self._cuda.event_destroy(self._event)
                self._event = None
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FTProcessGroup
# ---------------------------------------------------------------------------

class FTProcessGroup(dist.ProcessGroupNCCL):
    """Fault-tolerant torch.distributed ProcessGroup.

    Inherits standard NCCL collectives and dispatches eligible symmetric
    tensors to the device-side FT kernels.

    See module docstring for full details.

    Args:
        store:      torch.distributed Store (passed by torch.distributed).
        rank:       This process's rank.
        world_size: Total number of ranks.
        timeout:    Ignored (kept for API compatibility).
        max_count:  Maximum elements per rank for pre-allocated FT scratch
                    buffers (applies to all dtypes; scratch is sized for
                    float32 = 4 bytes/element, so float16/bfloat16 tensors
                    up to ``max_count`` elements always fit).  Larger tensors
                    fall back to NCCL.  Overridden by the
                    ``FT_NCCL_MAX_COUNT`` environment variable when set.
                    Default: 4 194 304 (16 MiB/rank at float32).
        timeout_us: FT kernel timeout in microseconds.  Overridden by the
                    ``FT_TIMEOUT_US`` environment variable when set.
                    Default: 10 000 000 (10 s).
    """

    def __init__(self, store, rank: int, world_size: int,
                 timeout=None,
                 max_count: int = 4_194_304,
                 timeout_us: int = 10_000_000) -> None:

        self._rank      = rank
        self._size      = world_size
        self._max_count = int(os.environ.get("FT_NCCL_MAX_COUNT", str(max_count)))
        self._timeout_us = int(os.environ.get("FT_TIMEOUT_US", str(timeout_us)))

        # Adopt the device the caller already pinned (the standard LOCAL_RANK
        # convention used by torchrun / vLLM / Megatron), so FTProcessGroup
        # honors external placement instead of imposing a rank%ngpu heuristic.
        # Fall back to rank%ngpu only when nothing set a non-default current
        # device, to avoid every rank silently colliding on GPU 0.
        self._device_id = torch.cuda.current_device()
        if (self._device_id == 0 and world_size > 1
                and torch.cuda.device_count() > 1):
            self._device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(self._device_id)
        self._device = torch.device(f"cuda:{self._device_id}")

        # ---- Init ProcessGroupNCCL (provides all NCCL fallback collectives) ----
        # Set its watchdog timeout slightly longer than the FT kernel timeout so
        # the FT path can exhaust its budget without NCCL killing the process group.
        # ProcessGroupNCCL.Options().timeout is only a settable Python attribute in
        # some PyTorch builds; fall back gracefully when it is not available.
        nccl_timeout_s = max(self._timeout_us // 1_000_000 + 5, 60)
        nccl_timeout   = datetime.timedelta(seconds=nccl_timeout_s)
        nccl_opts = dist.ProcessGroupNCCL.Options()
        if hasattr(nccl_opts, "timeout"):
            nccl_opts.timeout = nccl_timeout

        # Pass device_id explicitly so NCCL does not guess from global rank.
        # When CUDA_VISIBLE_DEVICES remaps device ordinals (e.g. each process
        # pins one GPU as cuda:0) the guess is wrong and causes a hang.
        device = torch.device("cuda", self._device_id)
        try:
            super().__init__(store, rank, world_size, nccl_opts, device_id=device)
        except TypeError:
            # Older PyTorch builds do not accept device_id; also try timeout kwarg.
            try:
                super().__init__(store, rank, world_size, nccl_opts,
                                 timeout=nccl_timeout)
            except TypeError:
                super().__init__(store, rank, world_size, nccl_opts)

        # ---- Load FT libraries ----
        self._nccl = NCCLBaseLibrary()
        self._ft   = FTLibrary()
        self._cuda = CUDARuntime()

        # ---- Bootstrap a second NCCL comm for the FT handle ----
        # ProcessGroupNCCL already consumed the store for its own comm.
        # We reuse the same store with a distinct key to exchange a second UID.
        uid = ncclUniqueId()
        if rank == 0:
            uid = self._nccl.get_unique_id()
            store.set("ft_nccl_uid_ft", bytes(uid.internal))
        else:
            raw = store.get("ft_nccl_uid_ft")
            for i, b in enumerate(raw):
                uid.internal[i] = b

        self._comm_ft = self._nccl.comm_init_rank(world_size, uid, rank)

        # The optional store barrier uses epoch keys; the default exchanges masks.
        self._store    = store
        self._active   = [True] * world_size   # agreed surviving set
        self._epoch    = 0                     # monotonic barrier epoch

        # ---- FT handle (auto-selects LSA or GIN) ----
        full_mask    = [True] * world_size
        self._handle = self._ft.handle_create(self._comm_ft, world_size, full_mask)

        # Membership is installed by ft_converge() or set_active_mask(), rather
        # than adapting independently from each rank's local observation.
        self._ft.handle_set_self_adapt(self._handle, False)

        # ---- Dedicated CUDA stream for FT kernel launches ----
        self._stream = self._cuda.stream_create()
        # Persistent event for UPSTREAM stream ordering: before each collective we
        # record it on the caller's current stream and make the FT stream wait, so
        # the FT kernel consumes work the caller just produced (e.g. the send
        # buffer) without a host synchronize.  FTWork.wait() covers the downstream
        # direction (caller's next op waits for the FT kernel).
        self._up_event = self._cuda.event_create()

        # Preallocate the all_to_allv metadata buffers now (not lazily) so the FT
        # path performs no per-call allocation — required for CUDA-graph capture
        # on the first iteration and for stable addresses. Pinned host staging +
        # device buffers for send/recv counts (int32) and displacements (int64).
        _n = world_size
        # Pinned sources are cached per split value (see reduce_scatterv below).
        self._a2av_counts_cache = {}
        self._a2av_d_counts  = torch.empty(_n, dtype=torch.int32, device=self._device)
        self._a2av_d_displs  = torch.empty(_n, dtype=torch.int64, device=self._device)
        self._a2av_d_rcounts = torch.empty(_n, dtype=torch.int32, device=self._device)
        self._a2av_d_rdispls = torch.empty(_n, dtype=torch.int64, device=self._device)
        # all_gatherv: this rank's own send count (device int32[nRanks]; [myRank]
        # used) + discovered per-source counts. Persistent -> alloc-free / graph-safe.
        self._agv_d_send_counts = torch.zeros(_n, dtype=torch.int32, device=self._device)
        self._agv_d_recv_counts = torch.zeros(_n, dtype=torch.int32, device=self._device)
        self._agv_counts_cache = {}   # send_count value -> stable pinned int32[nRanks]
        # reduce_scatterv: shard-size partition. The pinned SOURCE is cached PER
        # VALUE (tuple(counts) -> pinned int32[nRanks]); using one shared pinned
        # buffer is a multi-CUDA-graph bug — capturing a 2nd graph overwrites it,
        # so the 1st graph's captured H2D copies the wrong counts on replay. The
        # device buffer can stay shared: each graph's H2D (from its own stable
        # per-value pinned source) refreshes it before that graph's reduce.
        self._rsv_counts_cache = {}
        self._rsv_d_counts = torch.zeros(_n, dtype=torch.int32, device=self._device)

        # ---- Pre-allocate symmetric scratch buffers (collective!) ----
        # Each rank's recv slice is padded to a 16-byte stride (matching the
        # kernel's FT_RECV_STRIDE) so the device-side copies stay 128-bit
        # vectorized for any count.  Size the buffer from the EFFECTIVE limit
        # self._max_count (which honors FT_NCCL_MAX_COUNT) — the same value the
        # eligibility check uses — so any FT-eligible tensor fits.  Using the
        # bare constructor arg here would undersize the scratch whenever the env
        # raises the limit, and the gather kernel + 2D scatter would then run off
        # the end of the buffer (cudaErrorInvalidValue).
        recv_stride_max = ((self._max_count * 4 + 15) // 16) * 16
        recv_bytes  = world_size * recv_stride_max  # float32 gather scratch
        flags_bytes = world_size * 8                # uint64 flags per rank

        self._d_recv  = self._nccl.mem_alloc(recv_bytes)
        self._d_flags = self._nccl.mem_alloc(flags_bytes)
        self._win_recv  = self._nccl.window_register(
            self._comm_ft, self._d_recv,  recv_bytes)
        self._win_flags = self._nccl.window_register(
            self._comm_ft, self._d_flags, flags_bytes)

        # ---- One-time devComm init and CUDA JIT warm-up ----
        # ftAllGatherCreate calls ncclDevCommCreate (all permanent symmetric
        # windows — win_ready, _win_recv, _win_flags — are registered above,
        # satisfying the ordering requirement) and cudaDeviceSynchronize to
        # drain NCCL-internal streams.  A tiny AllGather kernel is launched
        # immediately afterwards to trigger CUDA PTX→SASS JIT compilation for
        # libft_collective.so while we are still in __init__.
        #
        # Without this, the lazy ncclDevCommCreate + JIT both fire inside the
        # first benchmark collective, causing a multi-minute startup stall that
        # is absent from C++ ft_handle_bench (which calls ftAllGatherCreate
        # explicitly before the timed loop).
        #
        # Note: ncclMemAlloc / ncclCommWindowRegister inside ftAllGatherCreate
        # are collective — all ranks must reach this point simultaneously.
        # They do because __init__ is called from the same
        # dist.init_process_group barrier on every rank.
        _ctx = self._ft.allgather_create(self._handle, 1)
        self._ft.allgather_launch(
            _ctx, [0.0], 1,
            timeout_us=self._timeout_us,
            stream=self._stream,
        )
        self._cuda.stream_synchronize(self._stream)
        self._ft.allgather_destroy(_ctx)

        # ---- Symmetric tensor registry ----
        self._windows:     Dict[int, int] = {}     # data_ptr → ncclWindow_t
        self._win_offsets: Dict[int, int] = {}     # data_ptr → byte offset in window
        self._win_sizes:   Dict[int, int] = {}     # registered bytes from data_ptr
        self._sym_objects: List[_SymTensor] = []   # keep _SymTensor alive
        self._sym_ptrs:    List[int] = []          # for cleanup (owned allocs from empty())
        # base_ptr → end_ptr for every *primary* window (from empty() or
        # register_symmetric_tensor).  Used for containment/overlap checks.
        self._win_extents: Dict[int, int] = {}
        # data_ptrs that alias a parent window (no window_deregister on cleanup).
        self._win_derived: set = set()

        # Register barrier buffers now, while all ranks are present at init.
        # Registration (ncclCommWindowRegister via self.empty()) is COLLECTIVE, but
        # ft_converge() is called precisely *because* a rank died — so its first
        # invocation is typically AFTER a failure, when a fresh registration would
        # deadlock / raise ncclRemoteError on the dead peer. Pre-registering here
        # keeps the recovery path allocation-free.
        n = self._size
        self._bar_send = self.empty(n * n, dtype=torch.float32)
        self._bar_out  = torch.zeros(n * n, dtype=torch.float32, device=self._device)

    # ------------------------------------------------------------------
    # Factory: allocate a symmetric tensor
    # ------------------------------------------------------------------

    def empty(self, *shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate a symmetric device tensor backed by ncclMemAlloc.

        COLLECTIVE: all ranks must call simultaneously with matching shape and dtype.

        float32/float16/bfloat16 tensors take the FT AllGather/AllReduce kernels.
        Integer dtypes (int64/int32/int16/int8/uint8) are also allocatable as
        symmetric memory for byte-oriented paths; they are not AllGather/AllReduce
        eligible and would fall back to NCCL there.

        Returns a ``torch.Tensor`` backed by a registered symmetric window.
        """
        if dtype not in _TORCH_TO_DL:
            raise NotImplementedError(
                f"FTProcessGroup.empty() does not support {dtype}; "
                f"supported: {', '.join(str(d) for d in _TORCH_TO_DL)}"
            )

        numel = 1
        for d in shape:
            numel *= d
        size_bytes = numel * dtype.itemsize

        ptr = self._nccl.mem_alloc(size_bytes)
        win = self._nccl.window_register(self._comm_ft, ptr, size_bytes)

        self._windows[ptr] = win
        self._win_offsets[ptr] = 0
        self._win_sizes[ptr] = size_bytes
        self._win_extents[ptr] = ptr + size_bytes
        self._sym_ptrs.append(ptr)

        sym = _SymTensor(ptr, numel, self._device_id, dtype)
        self._sym_objects.append(sym)

        return torch.from_dlpack(sym)

    def register_symmetric_tensor(
        self,
        tensor: Optional[torch.Tensor] = None,
        *,
        base_ptr: Optional[int] = None,
        size_bytes: Optional[int] = None,
    ) -> None:
        """Register a symmetric tensor (or raw allocation) for FT collectives.

        Use this when memory was allocated symmetrically outside of
        ``pg.empty()`` (e.g. via ``ncclMemAlloc`` / nccl4py) and you want it
        to take the FT kernel path.

        Two calling conventions are supported:

        1. **Tensor** — derive the pointer and byte size from *tensor*::

               pg.register_symmetric_tensor(my_tensor)

        2. **Explicit range** — register a raw symmetric allocation by address
           and size, typically for a large slab that will later be sub-divided
           into views::

               pg.register_symmetric_tensor(base_ptr=ptr, size_bytes=N)

        **Sub-region reuse**: if the range being registered is fully contained
        within an already-registered window the existing window is reused
        (no second ``ncclWindowRegister`` call is issued).  This is safe
        because the parent window already covers the same physical memory.
        Partial overlaps (neither contained nor containing) raise
        ``ValueError`` because overlapping NCCL window registrations have
        undefined behavior.

        COLLECTIVE: all ranks must call simultaneously with matching ranges
        at the same symmetric virtual address.

        The caller retains ownership of the memory.  FTProcessGroup will
        deregister its own NCCL windows on shutdown but will **not** free
        the underlying pointer.

        Args:
            tensor:     A ``torch.Tensor`` whose backing memory is symmetric.
                        Mutually exclusive with *base_ptr* / *size_bytes*.
            base_ptr:   Device pointer (int) of the symmetric allocation.
                        Requires *size_bytes*.
            size_bytes: Byte size of the allocation.  Required with *base_ptr*.

        Raises:
            ValueError: On already-registered, partial overlap, or bad args.
        """
        if tensor is not None and base_ptr is None:
            if not tensor.is_contiguous():
                raise ValueError("registered FT tensors must be contiguous")
            ptr = tensor.data_ptr()
            sz  = tensor.numel() * tensor.element_size()
        elif base_ptr is not None and tensor is None:
            if size_bytes is None:
                raise ValueError(
                    "size_bytes is required when base_ptr is specified."
                )
            ptr = base_ptr
            sz  = size_bytes
        else:
            raise ValueError(
                "Provide either tensor or base_ptr+size_bytes, not both (or neither)."
            )

        end = ptr + sz

        # Already registered (exact match)?
        if ptr in self._windows:
            kind = "derived alias" if ptr in self._win_derived else "primary window"
            raise ValueError(
                f"0x{ptr:x} is already registered as a {kind} with this "
                f"FTProcessGroup."
            )

        # Check existing primary windows for containment or partial overlap.
        for base, base_end in self._win_extents.items():
            fully_contained = (base <= ptr and end <= base_end)
            overlaps        = (ptr < base_end and end > base)

            if fully_contained:
                # Reuse the parent window — no new ncclWindowRegister needed.
                self._windows[ptr] = self._windows[base]
                self._win_offsets[ptr] = ptr - base
                self._win_sizes[ptr] = sz
                self._win_derived.add(ptr)
                return

            if overlaps:
                raise ValueError(
                    f"New range [0x{ptr:x}, 0x{end:x}) partially overlaps "
                    f"existing window [0x{base:x}, 0x{base_end:x}). "
                    f"Only full containment within an existing window is "
                    f"supported; partial overlaps have undefined NCCL behavior."
                )

        # No existing coverage — register a new primary window.
        win = self._nccl.window_register(self._comm_ft, ptr, sz)
        self._windows[ptr] = win
        self._win_offsets[ptr] = 0
        self._win_sizes[ptr] = sz
        self._win_extents[ptr] = end
        # Intentionally NOT added to self._sym_ptrs — caller owns the memory.

    def unregister_symmetric_tensor(
        self,
        tensor: Optional[torch.Tensor] = None,
        *,
        base_ptr: Optional[int] = None,
    ) -> None:
        """Unregister a tensor (or raw pointer) previously registered with
        ``register_symmetric_tensor``.

        COLLECTIVE: all ranks must call simultaneously.

        For **derived** registrations (sub-regions that reused a parent
        window) only the alias mapping is removed; the parent window is left
        intact.

        For **primary** registrations the NCCL window is deregistered and any
        derived aliases that referenced the same window are automatically
        removed.

        The underlying memory is never freed.

        Args:
            tensor:   Tensor whose registration should be removed.
                      Mutually exclusive with *base_ptr*.
            base_ptr: Raw device pointer identifying the registration.
                      Mutually exclusive with *tensor*.

        Raises:
            ValueError: If the pointer is not registered, was allocated by
                        ``pg.empty()`` (use teardown instead), or if both /
                        neither arguments are supplied.
        """
        if tensor is not None and base_ptr is None:
            ptr = tensor.data_ptr()
        elif base_ptr is not None and tensor is None:
            ptr = base_ptr
        else:
            raise ValueError(
                "Provide either tensor or base_ptr, not both (or neither)."
            )

        if ptr not in self._windows:
            raise ValueError(
                f"0x{ptr:x} is not registered with this FTProcessGroup."
            )

        if ptr in self._win_derived:
            # Derived alias — remove the mapping only, parent window unchanged.
            self._windows.pop(ptr)
            self._win_offsets.pop(ptr)
            self._win_sizes.pop(ptr)
            self._win_derived.discard(ptr)
            return

        if ptr in self._sym_ptrs:
            # pg.empty() owns this window and frees its allocation on teardown.
            raise ValueError(
                f"0x{ptr:x} was allocated by pg.empty() and cannot be "
                f"unregistered via unregister_symmetric_tensor."
            )

        # Primary external window — cascade-remove derived aliases first.
        win = self._windows[ptr]
        for p in [p for p in self._win_derived if self._windows.get(p) == win]:
            self._windows.pop(p, None)
            self._win_offsets.pop(p, None)
            self._win_sizes.pop(p, None)
            self._win_derived.discard(p)

        self._windows.pop(ptr)
        self._win_offsets.pop(ptr)
        self._win_sizes.pop(ptr)
        self._win_extents.pop(ptr)
        self._nccl.window_deregister(self._comm_ft, win)

    def set_active(self) -> None:
        """Register this instance as the module-level active FTProcessGroup.

        Required before using ``torch.ops.ft_collective.allgather`` or
        ``torch.ops.ft_collective.allreduce`` (e.g. inside a torch.compiled
        function or CUDA graph).
        """
        set_active_pg(self)

    def ft_converge(self) -> None:
        """Agree on a surviving set and synchronize before further collectives.

        The process group disables the C handle's local mask adaptation. Call
        this after a failure to exclude unresponsive peers consistently, or use
        ``set_active_mask`` when membership is managed externally. Without either
        step, later collectives keep polling the old active set and may time out.
        ``FT_BARRIER_MODE`` selects the collective (default) or TCPStore barrier.
        """
        self._cuda.stream_synchronize(self._stream)
        # FT_BARRIER_MODE selects the membership/re-sync mechanism:
        #   "collective" (default) — store-free on-GPU belief-exchange consensus
        #                  (fixed poll set + intersection; see _ft_barrier_collective).
        #   "store"      — reliable off-GPU TCPStore barrier.
        if os.environ.get("FT_BARRIER_MODE", "collective") == "store":
            self._ft_barrier()
        else:
            self._ft_barrier_collective()

    def ft_rejoin(self) -> List[bool]:
        """Collective rejoin — GROW the surviving set to re-admit stalled-but-alive
        ranks, WITHOUT changing the NCCL communicator. Sibling to ``ft_converge()``.

        Where ``ft_converge()`` only *shrinks* (it polls the current ``_active``
        set, so a dropped rank can never come back), ``ft_rejoin()`` polls the
        **full world** — the caller assumes every live rank is at least trying to
        rejoin. A rank that was removed from ``_active`` (e.g. it stalled past the
        timeout and others dropped it) re-appears here once it is responsive again.

        Mechanism: a generation **max-consensus** first re-synchronizes the
        on-device generation counter of a returnee whose ``d_gen`` drifted while it
        was out (a generation-gated handshake alone would reject it); then a normal
        generation-gated presence round finds the responders. Still-dead ranks time
        out and stay excluded. The agreed set is installed via ``set_active_mask``.

        COLLECTIVE: every live rank must call it together. Returns the new active
        set (also available via ``get_active_mask()``).

        Caveats: this restores *participation* only — a returnee's DATA is stale
        (it missed collectives while out) and the application must resync it (e.g.
        re-broadcast weights). A membership change also invalidates any captured
        CUDA graph, so re-capture after a rejoin.
        """
        self._cuda.stream_synchronize(self._stream)
        self._ft.rejoin(self._handle, timeout_us=self._timeout_us,
                        stream=self._stream)
        mask = self.get_result_mask()      # syncs; responders (kernel also set d_gen)
        self._active = mask
        self.set_active_mask(mask)         # install the grown set for the next collectives
        return list(mask)

    def get_generation(self) -> int:
        """Read the on-device generation counter. Diagnostic: after ft_rejoin all
        survivors should report an identical value (the synchronized generation)."""
        self._cuda.stream_synchronize(self._stream)
        return self._ft.handle_get_generation(self._handle)

    def _ft_barrier(self) -> None:
        """Timeout-bounded store barrier; updates the agreed surviving set.

        Uses non-blocking ``store.check()`` polling rather than the blocking
        ``store.wait()``: a blocking wait on a dead rank's key sits on the socket
        for the full timeout, which makes the c10d TCPStore client log noisy
        "waitForInput: poll ... likely a timeout" warnings.  Polling avoids that
        while keeping identical semantics (bounded wait; a peer whose key never
        appears is declared dead, consistently across all ranks).
        """
        import time

        self._epoch += 1
        e = self._epoch

        def _key(r: int) -> str:
            return f"ft_bar/{e}/{r}"

        # Announce arrival at this epoch.
        self._store.set(_key(self._rank), "1")

        peers = [r for r in range(self._size)
                 if r != self._rank and self._active[r]]
        keys = [_key(r) for r in peers]
        deadline = time.monotonic() + self._timeout_us / 1e6

        # Fast path: one combined existence check per poll; returns as soon as
        # every active peer has arrived.
        arrived = bool(keys) and self._store.check(keys)
        while not arrived:
            if not keys:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
            arrived = self._store.check(keys)

        if not arrived and keys:
            # Deadline passed: identify the specific non-responders (dead).
            newly_dead = [r for r in peers if not self._store.check([_key(r)])]
            if newly_dead:
                for r in newly_dead:
                    self._active[r] = False
                # Push the agreed surviving set into the FT handle so the next
                # kernel only signals/polls live peers.
                self._ft.set_active_mask(self._handle, self._active, self._size)

    def _ft_barrier_collective(self, rounds: int = None) -> None:
        """Exchange and intersect survivor masks over a fixed number of rounds.

        The poll set stays fixed while beliefs shrink. Every participant runs
        the same number of rounds so readiness generations remain aligned.
        See ``docs/FT_BARRIER_COLLECTIVE.md`` for the agreement argument.
        """
        import torch
        n = self._size
        if rounds is None:
            rounds = int(os.environ.get("FT_BARRIER_ROUNDS", "3"))
        rounds = max(2, rounds)   # >=2 so a boundary peer's exclusion propagates

        # Symmetric [nRanks, nRanks] send (my belief -> every destination) and a
        # flat [nRanks, nRanks] recv (row s = source s's belief).
        # FIXED poll set for the whole barrier -> readiness generations stay in
        # lockstep (every rank polls the same peers, the same number of rounds).
        poll_set = list(self._active)
        self._ft.set_active_mask(self._handle, poll_set, n)

        belief = list(self._active)
        sendv = self._bar_send.view(n, n)
        me = self._rank
        for _ in range(rounds):
            bt = torch.tensor([1.0 if b else 0.0 for b in belief],
                              dtype=torch.float32, device=self._device)
            sendv.copy_(bt.unsqueeze(0).expand(n, n))   # send my belief to every dest
            self.alltoall_base(self._bar_out, self._bar_send, [], [])
            resp = self.get_result_mask()               # who answered (syncs stream)
            recv = (self._bar_out.view(n, n) > 0.5).cpu().tolist()   # recv[s][r] = s's belief
            responders = [s for s in range(n) if resp[s] or s == me]
            new_belief = [False] * n
            for r in range(n):
                if r == me:
                    new_belief[r] = True                # never drop self from my belief
                elif resp[r] and all(recv[s][r] for s in responders):
                    new_belief[r] = True                # survived every responder's belief
            belief = new_belief

        belief[me] = True
        self._active = belief
        # Commit the agreed surviving set to the FT handle for the next kernel.
        self._ft.set_active_mask(self._handle, self._active, n)

    def get_error(self) -> int:
        """Read the sticky FT status without synchronizing.

        While kernels are pending, a zero status may still change. ``wait()``
        orders GPU work without waiting on the host; synchronize the FT stream
        or device for a definitive read.
        """
        return self._ft.handle_get_error(self._handle)

    def clear_error(self) -> None:
        """Reset the sticky FT error flag to ``FT_OK``.

        Only call when no FT kernel is in flight. ``work.wait()`` alone does
        not establish host completion.
        """
        self._ft.handle_clear_error(self._handle)

    def check_and_clear_error(self) -> int:
        """Read and clear FT status after the caller synchronizes queued work.

        This method does not synchronize. Clearing while a kernel is running
        can lose an error; use ``get_error()`` to poll pending work.
        """
        error = self._ft.handle_get_error(self._handle)
        self._ft.handle_clear_error(self._handle)
        return error

    def get_result_mask(self) -> List[bool]:
        """Return the result mask from the last FT collective, syncing if needed.

        Blocks the host CPU until the FT kernel is complete, then returns this
        rank's per-iteration observation of which ranks responded.

        It intentionally does NOT mutate the input mask.  Membership is owned by
        the fault-tolerant barrier in ``ft_converge()`` (which agrees a single
        surviving set across all ranks); letting this per-rank GPU observation
        self-heal the input mask would re-introduce the cross-rank divergence the
        barrier exists to prevent.  The sticky error is cleared so the next
        iteration starts fresh.

        Safe to call at any time after the collective is launched.  If called
        before ``work.wait()``, it also satisfies the stream-ordering guarantee
        for subsequent operations on the current stream.
        """
        self._cuda.stream_synchronize(self._stream)
        self._ft.handle_clear_error(self._handle)
        return self._ft.handle_get_result_mask(self._handle, self._size)

    def get_active_mask(self) -> List[bool]:
        """Return the agreed surviving set (membership owned by the FT barrier).

        This is the consistent, cross-rank-agreed view of who is alive, updated
        by ``ft_converge()``.  Prefer this over ``get_result_mask()`` for deciding
        membership; ``get_result_mask()`` is a per-iteration GPU observation.
        """
        return list(self._active)

    def set_active_mask(self, mask: List[bool]) -> None:
        """Manually install the active-membership mask (framework-managed override).

        The symmetric setter for :meth:`get_active_mask`.  Overrides the surviving
        set that :meth:`ft_converge` would agree — the CALLER takes ownership of
        membership.  Use this for a *framework-managed* recovery mode: the app
        decides who is alive (e.g. after an external health check) and installs it
        directly, instead of the PG-managed consensus barrier.

        COORDINATED: every rank must install the SAME mask on the same step, or the
        next collective will split membership and can deadlock (some ranks wait on
        a peer others have dropped).  The FT backend runs with device
        self-adaptation disabled, so the mask you set persists until the next
        ``set_active_mask()`` or ``ft_converge()``.

        Args:
            mask: host bool list of length ``world_size``; ``True`` = rank participates.
        """
        if len(mask) != self._size:
            raise ValueError(
                f"active mask length {len(mask)} != world size {self._size}")
        self._active = [bool(b) for b in mask]
        self._ft.set_active_mask(self._handle, self._active, self._size)

    def _is_registered_tensor(self, tensor: torch.Tensor) -> bool:
        ptr = tensor.data_ptr()
        return (ptr in self._windows and tensor.is_contiguous() and
                tensor.numel() * tensor.element_size() <= self._win_sizes[ptr])

    def _window_spec(self, tensor: torch.Tensor):
        """Return the registered window and this tensor's byte offset in it."""
        ptr = tensor.data_ptr()
        return self._windows[ptr], self._win_offsets[ptr]

    def _is_ft_eligible(self, tensor: torch.Tensor) -> bool:
        """True if tensor fits its symmetric window and pre-allocated scratch."""
        return (self._is_registered_tensor(tensor) and
                tensor.numel() <= self._max_count)

    def _is_ft_eligible_ag(self, tensor: torch.Tensor) -> bool:
        """True if tensor can take the FT AllGather path."""
        byte_count = tensor.numel() * tensor.element_size()
        return (self._is_ft_eligible(tensor) and
                tensor.dtype in _FT_AG_DTYPES and
                byte_count % 4 == 0)

    def _is_ft_eligible_ar(self, tensor: torch.Tensor) -> bool:
        """True if tensor can take the FT AllReduce path."""
        return (self._is_ft_eligible(tensor) and
                tensor.dtype in _FT_AR_DTYPES)

    def _is_ft_eligible_a2a(self, send: torch.Tensor) -> bool:
        """True if send can take the FT uniform all-to-all path.

        send holds nRanks per-destination chunks (numel = nRanks * per_dest), so
        the binding size limit is the PER-DESTINATION count — the recv scratch
        holds one such chunk per source, exactly like AllGather."""
        if not self._is_registered_tensor(send):
            return False
        if send.numel() % self._size != 0:
            return False
        per_dest = send.numel() // self._size
        per_dest_bytes = per_dest * send.element_size()
        return (per_dest <= self._max_count and
                send.dtype in _FT_AG_DTYPES and
                per_dest_bytes % 4 == 0)

    def _is_ft_eligible_a2av(self, send: torch.Tensor,
                             input_splits, output_splits) -> bool:
        """True if send can take the FT variable all-to-allv path (Phase 1).

        float32/float16/bfloat16 (counts are native element counts; the kernel is
        a byte-copy dispatched per dtype). Every per-(src,dst) chunk must fit one
        scratch slot, i.e. max split <= max_count."""
        if not self._is_registered_tensor(send):
            return False
        if send.dtype not in _FT_AR_DTYPES:
            return False
        n = self._size
        if len(input_splits) != n or len(output_splits) != n:
            return False
        send_counts = [int(x) for x in input_splits]
        recv_counts = [int(x) for x in output_splits]
        if any(x < 0 or x > self._max_count for x in send_counts + recv_counts):
            return False
        if sum(send_counts) != send.numel():
            return False
        return True

    def _order_after_current(self) -> None:
        """Make the FT stream wait for the caller's current stream (upstream
        ordering).  Call at the start of every FT collective so the kernel sees
        work the caller just enqueued (send buffer, counts) with no host sync.
        Graph-capturable: event record + stream wait are captured."""
        current = torch.cuda.current_stream(self._device).cuda_stream
        self._cuda.event_record(self._up_event, current)
        self._cuda.stream_wait_event(self._stream, self._up_event)

    def _ft_work(self) -> FTWork:
        """Record an event on the FT stream and return an FTWork for ordering."""
        event = self._cuda.event_create()
        self._cuda.event_record(event, self._stream)
        return FTWork(self._cuda, event, self)

    def _fallback_reason(self, tensor: torch.Tensor, op: str = "allgather") -> str:
        if tensor.data_ptr() not in self._windows:
            return ("input is not symmetric "
                    "(use pg.empty() or pg.register_symmetric_tensor() for FT)")
        if not self._is_registered_tensor(tensor):
            return "input is strided or exceeds its registered window"
        if tensor.numel() > self._max_count:
            return f"numel {tensor.numel()} exceeds max_count {self._max_count}"
        if op == "allgather":
            if tensor.dtype not in _FT_AG_DTYPES:
                return f"dtype {tensor.dtype} is not supported by FT AllGather"
            byte_count = tensor.numel() * tensor.element_size()
            if byte_count % 4 != 0:
                return (f"numel {tensor.numel()} × itemsize {tensor.element_size()} "
                        f"= {byte_count} bytes is not divisible by 4")
        else:  # allreduce
            if tensor.dtype not in _FT_AR_DTYPES:
                return (f"dtype {tensor.dtype} is not supported by FT AllReduce "
                        f"(supported: float32, float16, bfloat16)")
        return f"numel {tensor.numel()} exceeds max_count {self._max_count}"

    # ------------------------------------------------------------------
    # allreduce — FT path or NCCL fallback
    # ------------------------------------------------------------------

    def allreduce(self, tensors: List[torch.Tensor],
                  opts=None) -> dist.Work:
        """In-place AllReduce.

        Uses the FT kernel when ``tensors[0]`` is a symmetric tensor from
        ``pg.empty()`` and the op is SUM.  Supported dtypes: float32, float16,
        bfloat16 (reduction always accumulates in float32 internally).
        Non-SUM ops fall back to ``ProcessGroupNCCL.allreduce()`` with a
        ``UserWarning``.

        Uses stream ordering on the FT path: ``work.wait()`` makes the current
        stream wait for the kernel without blocking the CPU.  Call
        ``get_result_mask()`` when you need to inspect which ranks responded.
        """
        t  = tensors[0]
        op = getattr(opts, "reduceOp", dist.ReduceOp.SUM)

        if self._is_ft_eligible_ar(t) and op == dist.ReduceOp.SUM:
            win_send, send_offset = self._window_spec(t)
            ft_dtype    = _TORCH_TO_FT_AR_DTYPE[t.dtype]
            # d_recv is intentionally NOT zeroed: present ranks fully overwrite
            # their slices and the reduction skips absent ranks via the result
            # mask, so the memset (scaled with nranks*count) was dead hot-path
            # work.  Per-rank slices use a padded 16-byte stride in the kernel.
            self._order_after_current()   # upstream stream ordering
            self._cuda.memset_async(
                self._d_flags, 0, self._size * 8,           self._stream)
            self._ft.allreduce(
                self._handle,
                win_send,        send_offset,
                self._win_recv,  0,
                self._win_flags,
                t.data_ptr(),
                t.numel(),
                timeout_us=self._timeout_us,
                dtype=ft_dtype,
                stream=self._stream,
            )
            return self._ft_work()

        if op != dist.ReduceOp.SUM:
            reason = f"reduceOp {op} is not supported by FT kernels"
        else:
            reason = self._fallback_reason(t, "allreduce")
        _ft_warn("allreduce", reason)
        return super().allreduce(tensors, opts)

    # ------------------------------------------------------------------
    # allgather — FT path or NCCL fallback
    # ------------------------------------------------------------------

    def allgather(self, output_tensor_lists: List[List[torch.Tensor]],
                  input_tensor_list: List[torch.Tensor],
                  opts=None) -> dist.Work:
        """AllGather.

        Uses the FT kernel when ``input_tensor_list[0]`` is a symmetric tensor
        from ``pg.empty()``.  Supported dtypes: float32, float16, bfloat16
        (any dtype whose ``numel * element_size`` is divisible by 4).  Output
        tensors do not need to be symmetric.  Otherwise falls back to
        ``ProcessGroupNCCL.allgather()`` with a ``UserWarning``.

        Uses stream ordering on the FT path: ``work.wait()`` makes the current
        stream wait for the kernel without blocking the CPU.  Call
        ``get_result_mask()`` when you need to inspect which ranks responded.
        """
        t       = input_tensor_list[0]
        outputs = output_tensor_lists[0]
        n       = self._size

        # Variable all-gather: the per-rank output slots differ in size — route to
        # the FT all_gatherv kernel (NCCL cannot do variable all-gather at all).
        # Detect "variable" from the OUTPUT slot sizes, which are identical on
        # every rank (the gather layout), so all ranks make the SAME FT-vs-NCCL
        # choice — a split would deadlock (some ranks in the FT collective, others
        # in NCCL).
        sizes = [o.numel() for o in outputs]
        is_variable = (len(outputs) == n and n >= 2
                       and any(s != sizes[0] for s in sizes)
                       and t.dtype in _FT_AR_DTYPES
                       and 1 <= max(sizes) <= self._max_count)
        if is_variable:
            if (not self._is_registered_tensor(t)
                    or not all(o.is_contiguous() and o.dtype == t.dtype
                               for o in outputs)):
                # Every rank agreed this is the FT path; a non-symmetric input here
                # can't take it, and falling back to NCCL would deadlock the ranks
                # that did. Fail loudly instead. (A 0-length contribution must be a
                # VIEW of a symmetric buffer, e.g. pg.empty().reshape(-1)[:0].)
                raise RuntimeError(
                    "FTProcessGroup.allgather: variable all-gather requires a "
                    "symmetric, contiguous input (a view of pg.empty()) on EVERY "
                    "rank — even a 0-length contribution. Got a non-symmetric or "
                    "non-contiguous input; NCCL fallback would deadlock.")
            tflat = t.reshape(-1)                      # flat: row_size = 1
            cap   = max(sizes)
            if tflat.numel() > cap or tflat.numel() != sizes[self._rank]:
                raise ValueError("variable all-gather input size must match this rank's output slot")
            esize = t.element_size()
            # Packed fast path: the slots are contiguous views of ONE buffer in
            # rank order -> the kernel packs straight into it, zero extra copy.
            base, off, packed = outputs[0].data_ptr(), 0, True
            for r in range(n):
                if outputs[r].data_ptr() != base + off * esize:
                    packed = False
                    break
                off += sizes[r]
            if packed:
                work, _ = self.all_gatherv(
                    tflat, outputs[0], tflat.numel(), cap,
                    _output_rows=sum(sizes))
                return work
            # Non-packed (uncommon): gather into a scratch, then scatter to the
            # slots using the DISCOVERED counts — correct under failure, but one
            # host sync (this is the slow, non-contiguous layout).
            scratch = torch.empty(sum(sizes), dtype=t.dtype, device=self._device)
            work, d_recv = self.all_gatherv(
                tflat, scratch, tflat.numel(), cap,
                _output_rows=sum(sizes))
            work.wait()
            self._cuda.stream_synchronize(self._stream)
            rc, o = d_recv.cpu().tolist(), 0
            for s in range(n):
                cnt = int(rc[s])
                if cnt:
                    outputs[s].view(-1)[:cnt].copy_(scratch[o:o + cnt])
                o += cnt
            return work

        if self._is_ft_eligible_ag(t):
            if (len(outputs) != n or
                    not all(o.is_contiguous() and o.dtype == t.dtype and
                            o.numel() == t.numel() for o in outputs)):
                raise ValueError("FT allgather needs one contiguous, matching output per rank")
            win_send, send_offset = self._window_spec(t)
            chunk_bytes = t.numel() * t.element_size()
            # ftAllGather count is in float32-element units; convert byte count.
            float_equiv = chunk_bytes // 4
            # Per-rank recv slice is padded to a 16-byte stride (matches the
            # kernel's FT_RECV_STRIDE) so the device copies stay vectorized.
            stride_bytes = ((chunk_bytes + 15) // 16) * 16
            # d_recv is intentionally NOT zeroed: every present rank fully
            # overwrites its slice, so the memset (scaled with nranks*count) was
            # dead hot-path work.  Absent-rank slices are reported via the mask.
            self._order_after_current()   # upstream stream ordering
            self._cuda.memset_async(
                self._d_flags, 0, self._size * 8,           self._stream)
            self._ft.allgather(
                self._handle,
                win_send,        send_offset,
                self._win_recv,  0,
                self._win_flags,
                float_equiv,
                timeout_us=self._timeout_us,
                stream=self._stream,
            )
            # Scatter scratch recv → output tensors.  When the outputs form one
            # contiguous, equally-sized block (the common all_gather_into_tensor
            # layout), do the padded→packed gather in a single strided 2D copy
            # instead of nranks separate launches — this removes most of the
            # per-call Python/launch overhead that dominates at small sizes.
            base = outputs[0].data_ptr()
            contiguous = all(
                outputs[r].data_ptr() == base + r * chunk_bytes
                and outputs[r].numel() * outputs[r].element_size() == chunk_bytes
                for r in range(self._size)
            )
            if contiguous:
                self._cuda.memcpy_2d_d2d_async(
                    base,            chunk_bytes,    # dst, dpitch (packed)
                    self._d_recv,    stride_bytes,   # src, spitch (padded)
                    chunk_bytes,     self._size,      # width, height
                    self._stream,
                )
            else:
                for r in range(self._size):
                    self._cuda.memcpy_d2d_async(
                        outputs[r].data_ptr(),
                        self._d_recv + r * stride_bytes,
                        chunk_bytes,
                        self._stream,
                    )
            return self._ft_work()

        _ft_warn("allgather", self._fallback_reason(t, "allgather"))
        return super().allgather(output_tensor_lists, input_tensor_list, opts)

    # ------------------------------------------------------------------
    # NCCL-fallback wrappers with warnings
    # ------------------------------------------------------------------

    def broadcast(self, tensors, opts=None):
        _ft_warn("broadcast", "not fault-tolerant")
        return super().broadcast(tensors, opts)

    def reduce(self, tensors, opts=None):
        _ft_warn("reduce", "not fault-tolerant")
        return super().reduce(tensors, opts)

    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        _ft_warn("reduce_scatter", "not fault-tolerant (use reduce_scatter_tensor "
                 "with a symmetric input for the FT path)")
        return super().reduce_scatter(output_tensors, input_tensors, opts)

    def _allgather_base(self, output, input, opts=None):
        """AllGather into one tensor — FT path via all_gatherv when eligible.

        ``dist.all_gather_into_tensor`` dispatches here. Takes the FT variable
        all-gather kernel (uniform here) when ``input`` is a symmetric tensor from
        ``pg.empty()`` and ``output`` is ``nRanks × input``; otherwise falls back
        to ``ProcessGroupNCCL`` with a ``UserWarning``.
        """
        n = self._size
        # all_gatherv treats the last dim as the row, so rows = numel / row width.
        row_size = input.shape[-1] if input.dim() > 1 else 1
        rows = input.numel() // row_size if row_size else 0
        if (rows > 0 and self._is_registered_tensor(input)
                and input.dtype in _FT_AR_DTYPES
                and output.is_contiguous()
                and output.dtype == input.dtype
                and output.numel() == input.numel() * n
                and rows <= self._max_count):
            work, _ = self.all_gatherv(input, output, rows, rows)
            return work
        _ft_warn("allgather_into_tensor",
                 "not fault-tolerant (non-symmetric input, unsupported dtype, or "
                 "output != nRanks × input)")
        return super()._allgather_base(
            output, input, opts if opts is not None else dist.AllgatherOptions())

    def allgather_into_tensor(self, output, input, opts=None):
        """Alias of :meth:`_allgather_base` for direct calls on the group."""
        return self._allgather_base(output, input, opts)

    def _reduce_scatter_base(self, output, input, opts=None):
        """Reduce-scatter into one tensor — FT path via reduce_scatterv (SUM).

        ``dist.reduce_scatter_tensor`` dispatches here. Uses FT for SUM on a
        symmetric input sized ``nRanks × output``. Other operations or layouts
        use NCCL with a warning.
        """
        n = self._size
        op = getattr(opts, "reduceOp", dist.ReduceOp.SUM)
        if (op == dist.ReduceOp.SUM
                and self._is_registered_tensor(input)
                and input.dtype in _FT_AR_DTYPES
                and output.is_contiguous()
                and output.dtype == input.dtype
                and input.numel() == output.numel() * n
                and output.numel() <= self._max_count):
            return self.reduce_scatterv(input, output, [output.numel()] * n)
        _ft_warn("reduce_scatter_tensor",
                 "not fault-tolerant (requires SUM and matching symmetric input/output layouts)")
        return super()._reduce_scatter_base(
            output, input, opts if opts is not None else dist.ReduceScatterOptions())

    def reduce_scatter_tensor(self, output, input, opts=None):
        """Alias of :meth:`_reduce_scatter_base` for direct calls on the group."""
        return self._reduce_scatter_base(output, input, opts)

    def allreduce_coalesced(self, tensors, opts=None):
        _ft_warn("allreduce_coalesced", "not fault-tolerant")
        return super().allreduce_coalesced(tensors, opts)

    def alltoall(self, output_tensors, input_tensors, opts=None):
        _ft_warn("alltoall", "not fault-tolerant")
        return super().alltoall(output_tensors, input_tensors, opts)

    def alltoall_base(self, output, input, output_splits, input_splits, opts=None):
        """Uniform all-to-all takes the FT kernel; ragged (split sizes) falls back.

        FT path requires: no output/input split sizes (uniform), a symmetric
        ``input`` whose per-destination count <= max_count, and a contiguous
        ``output`` of equal numel.  ``input``'s chunk j (offset j*per_dest) is
        sent to rank j; ``output``'s slot i receives rank i's chunk.
        """
        if (not output_splits and not input_splits
                and output.is_contiguous()
                and output.dtype == input.dtype
                and output.numel() == input.numel()
                and self._is_ft_eligible_a2a(input)):
            n              = self._size
            win_send, send_offset = self._window_spec(input)
            per_dest_bytes = (input.numel() // n) * input.element_size()
            float_equiv    = per_dest_bytes // 4
            stride_bytes   = ((per_dest_bytes + 15) // 16) * 16
            self._order_after_current()   # upstream stream ordering
            self._cuda.memset_async(self._d_flags, 0, self._size * 8, self._stream)
            self._ft.alltoall(
                self._handle,
                win_send,        send_offset,
                self._win_recv,  0,
                self._win_flags,
                float_equiv,
                timeout_us=self._timeout_us,
                stream=self._stream,
            )
            # output is contiguous [n * per_dest]; scatter the padded recv slots
            # into it with one strided 2D copy (same as AllGather's scatter).
            self._cuda.memcpy_2d_d2d_async(
                output.data_ptr(), per_dest_bytes,   # dst, dpitch (packed)
                self._d_recv,      stride_bytes,     # src, spitch (padded)
                per_dest_bytes,    n,                # width, height
                self._stream,
            )
            return self._ft_work()

        # Variable (all_to_allv): caller-supplied split sizes; f32/f16/bf16.
        if (output_splits and input_splits and output.is_contiguous()
                and output.dtype == input.dtype
                and sum(int(x) for x in output_splits) == output.numel()
                and self._is_ft_eligible_a2av(input, input_splits, output_splits)):
            n            = self._size
            win_send, send_offset = self._window_spec(input)
            esize        = input.element_size()
            dtype_code   = _TORCH_TO_FT_AR_DTYPE[input.dtype]
            send_counts  = [int(x) for x in input_splits]
            recv_counts  = [int(x) for x in output_splits]
            send_displs  = [0] * n
            recv_displs  = [0] * n
            acc = 0
            for i in range(n):
                send_displs[i] = acc; acc += send_counts[i]
            acc = 0
            for i in range(n):
                recv_displs[i] = acc; acc += recv_counts[i]
            max_send_count = max(send_counts) if send_counts else 0
            # Fixed per-source scratch slot stride: win_recv holds nRanks slots of
            # recv_stride_max bytes (>= max_count*4 >= max_count*esize), so any
            # recv chunk (<= max_count elems) fits regardless of dtype.
            recv_stride_bytes = ((self._max_count * 4 + 15) // 16) * 16

            # Count/displ metadata (native-element units) is uploaded with an
            # async H2D on the FT stream; the kernel compacts recv slots into
            # `output` itself. The pinned sources are cached PER VALUE: a shared
            # pinned buffer would be overwritten by a later call (or a second
            # captured graph) before an earlier queued H2D had read it.
            key = (tuple(send_counts), tuple(send_displs),
                   tuple(recv_counts), tuple(recv_displs))
            srcs = self._a2av_counts_cache.get(key)
            if srcs is None:
                srcs = []
                for vals, dt in ((send_counts, torch.int32), (send_displs, torch.int64),
                                 (recv_counts, torch.int32), (recv_displs, torch.int64)):
                    h = torch.empty(n, dtype=dt, pin_memory=True)
                    h.numpy()[:] = vals
                    srcs.append(h)
                self._a2av_counts_cache[key] = srcs
            for hbuf, dbuf in zip(srcs, (self._a2av_d_counts, self._a2av_d_displs,
                                         self._a2av_d_rcounts, self._a2av_d_rdispls)):
                self._cuda.memcpy_h2d_async(dbuf.data_ptr(), hbuf.data_ptr(),
                                            n * hbuf.element_size(), self._stream)

            self._order_after_current()   # upstream stream ordering
            self._cuda.memset_async(self._d_flags, 0, self._size * 8, self._stream)
            self._ft.alltoallv(
                self._handle,
                win_send,        send_offset,
                self._win_recv,  0,
                self._win_flags,
                self._a2av_d_counts.data_ptr(), self._a2av_d_displs.data_ptr(),
                recv_stride_bytes, max_send_count,
                output.data_ptr(), self._a2av_d_rcounts.data_ptr(),
                self._a2av_d_rdispls.data_ptr(),
                0, 0, 1,                     # packed mode: no discovery/capacity; element counts
                dtype_code,
                timeout_us=self._timeout_us,
                stream=self._stream,
            )
            return self._ft_work()

        _ft_warn("alltoall_base",
                 "not fault-tolerant (unsupported split sizes / dtype or non-symmetric input)")
        return super().alltoall_base(output, input, output_splits, input_splits, opts)

    def all_gatherv(self, send: torch.Tensor, output: torch.Tensor,
                    send_count, recv_capacity: int, recv_counts=None,
                    *, _output_rows: Optional[int] = None):
        """Variable all-gather: contribute a variable-size chunk, receive all.

        This rank contributes ``send_count`` rows of the symmetric tensor ``send``;
        every rank receives the PACKED concatenation of all ranks' chunks (in rank
        order) into ``output``. Sizes are DISCOVERED (count-in-flag) — no separate
        size all-gather — and returned device-resident.

        Args:
            send: symmetric ``[rows, hidden]`` (or 1-D) tensor from ``pg.empty()`` /
                ``register_symmetric_tensor``.
            output: LOCAL device tensor holding the packed concatenation; size it
                for the worst case (``nRanks * recv_capacity`` rows).
            send_count: this rank's row count — an ``int`` (staged, convenient) or a
                contiguous device ``int32[nRanks]`` whose ``[myRank]`` is this rank's
                count (graph-safe; other entries are ignored).
            recv_capacity: max rows any one source may contribute (scratch bound).
            recv_counts: optional device ``int32[nRanks]`` to receive the discovered
                per-source counts; if ``None`` an internal buffer is used.

        Returns:
            ``(work, d_recv_counts)`` — an :class:`FTWork` and the device
            ``int32[nRanks]`` discovered counts (``0`` for a timed-out source). The
            packed output occupies the first ``sum(d_recv_counts)`` rows. No host
            synchronize; poll ``get_error()`` for failures.
        """
        if not self._is_registered_tensor(send):
            raise ValueError("all_gatherv: send is not symmetric (use pg.empty() / "
                             "pg.register_symmetric_tensor()).")
        if not output.is_contiguous() or output.dtype != send.dtype:
            raise ValueError("all_gatherv: output must be contiguous with the send dtype")
        row_size  = send.shape[-1] if send.dim() > 1 else 1
        row_bytes = row_size * send.element_size()
        required_rows = (self._size * recv_capacity if _output_rows is None
                         else _output_rows)
        available_bytes = output.numel() * output.element_size()
        if _output_rows is not None and required_rows * row_bytes > available_bytes:
            # A packed list-form gather passes the first view of a larger tensor.
            available_bytes = (output.untyped_storage().nbytes() -
                               output.storage_offset() * output.element_size())
        if required_rows < 0 or required_rows * row_bytes > available_bytes:
            raise ValueError("all_gatherv: output storage is too small for packed rows")
        recv_stride = ((recv_capacity * row_bytes + 15) // 16) * 16
        slot_max = ((self._max_count * 4 + 15) // 16) * 16
        if recv_stride > slot_max:
            raise ValueError(
                f"all_gatherv: recv_capacity*row_bytes ({recv_stride} B) exceeds the "
                f"handle scratch slot ({slot_max} B); lower recv_capacity or raise "
                f"FT_NCCL_MAX_COUNT.")
        if isinstance(send_count, torch.Tensor):
            if (not send_count.is_cuda or send_count.dtype != torch.int32
                    or send_count.numel() != self._size
                    or not send_count.is_contiguous()):
                raise ValueError("device send_count must be a contiguous int32[nRanks]")
            if recv_capacity * row_bytes > send.numel() * send.element_size():
                raise ValueError("all_gatherv: send tensor must hold recv_capacity rows")
            d_send_counts = send_count.data_ptr()
        else:
            # Per-value pinned source, refreshed each call into the shared device
            # buffer — graph-safe across multiple captured graphs (a shared pinned
            # buffer would be overwritten by a later capture; see reduce_scatterv).
            key = int(send_count)
            if key < 0 or key > recv_capacity or key * row_bytes > send.numel() * send.element_size():
                raise ValueError("all_gatherv: send_count exceeds send tensor or recv_capacity")
            h_sc = self._agv_counts_cache.get(key)
            if h_sc is None:
                h_sc = torch.zeros(self._size, dtype=torch.int32, pin_memory=True)
                h_sc[self._rank] = key
                self._agv_counts_cache[key] = h_sc
            d_send_counts = self._agv_d_send_counts.data_ptr()
        rc = recv_counts if recv_counts is not None else self._agv_d_recv_counts
        win_send, send_offset = self._window_spec(send)
        # Order after the caller's stream BEFORE any FT-stream work, so the count
        # upload is part of a captured graph (and of the joined fork) too.
        self._order_after_current()
        if not isinstance(send_count, torch.Tensor):
            self._cuda.memcpy_h2d_async(d_send_counts, h_sc.data_ptr(),
                                        self._size * 4, self._stream)
        self._cuda.memset_async(self._d_flags, 0, self._size * 8, self._stream)
        self._ft.allgatherv(
            self._handle,
            win_send,       send_offset,
            self._win_recv, 0,
            self._win_flags,
            d_send_counts, output.data_ptr(), rc.data_ptr(),
            recv_capacity, row_bytes, recv_stride,
            recv_capacity,                      # max_send_count hint (rows)
            timeout_us=self._timeout_us,
            stream=self._stream,
        )
        return self._ft_work(), rc

    # ------------------------------------------------------------------
    # reduce_scatterv — variable reduce-scatter (typed SUM)
    # ------------------------------------------------------------------

    def reduce_scatterv(self, send: torch.Tensor, output: torch.Tensor, counts):
        """Variable reduce-scatter (SUM): sum every rank's full input, keep a shard.

        Every rank supplies its full input ``send`` (``sum(counts)`` elements);
        this rank receives shard ``[displs[myRank], counts[myRank])`` summed over
        all surviving contributors into ``output`` (``counts[myRank]`` elements).
        The shard sizes are a known partition (an input, not discovered).

        Args:
            send: symmetric full-input tensor from ``pg.empty()`` /
                ``register_symmetric_tensor`` (float32/float16/bfloat16).
            output: LOCAL device tensor of ``counts[myRank]`` elements — written.
            counts: the shard-size partition — a host sequence of ``nRanks`` ints
                (shard offsets are derived on-device). Must be identical on every
                rank (it is the output partition).

        Returns:
            An :class:`FTWork` for stream ordering. Survivor-sum: a timed-out
            contributor drops out; poll ``get_error()`` and consult
            ``get_result_mask()`` after a flagged failure.
        """
        if not self._is_registered_tensor(send):
            raise ValueError("reduce_scatterv: send is not symmetric (use "
                             "pg.empty() / pg.register_symmetric_tensor()).")
        if send.dtype not in _FT_AR_DTYPES:
            raise ValueError("reduce_scatterv supports float32/float16/bfloat16.")
        counts_list = ([int(c) for c in counts.tolist()]
                       if isinstance(counts, torch.Tensor) else [int(c) for c in counts])
        if len(counts_list) != self._size:
            raise ValueError("counts must have nRanks entries.")
        if any(c < 0 for c in counts_list) or sum(counts_list) != send.numel():
            raise ValueError("reduce_scatterv: counts must partition the send tensor")
        if (not output.is_contiguous() or output.dtype != send.dtype or
                output.numel() != counts_list[self._rank]):
            raise ValueError("reduce_scatterv: output size must match this rank's shard")
        max_shard = max(counts_list) if counts_list else 0
        if max_shard < 1:
            raise ValueError("reduce_scatterv: all shard counts are zero.")
        esize = send.element_size()
        recv_stride = ((max_shard * esize + 15) // 16) * 16
        slot_max = ((self._max_count * 4 + 15) // 16) * 16
        if recv_stride > slot_max:
            raise ValueError(
                f"reduce_scatterv: max_shard*elem ({recv_stride} B) exceeds the "
                f"handle scratch slot ({slot_max} B); raise FT_NCCL_MAX_COUNT.")
        dtype_code = _TORCH_TO_FT_AR_DTYPE[send.dtype]
        # Per-value pinned source (stable across graph captures — see __init__).
        key = tuple(counts_list)
        h_counts = self._rsv_counts_cache.get(key)
        if h_counts is None:
            h_counts = torch.empty(self._size, dtype=torch.int32, pin_memory=True)
            h_counts.numpy()[:] = counts_list
            self._rsv_counts_cache[key] = h_counts
        win_send, send_offset = self._window_spec(send)
        self._order_after_current()
        self._cuda.memcpy_h2d_async(self._rsv_d_counts.data_ptr(),
                                    h_counts.data_ptr(),
                                    self._size * 4, self._stream)
        self._cuda.memset_async(self._d_flags, 0, self._size * 8, self._stream)
        self._ft.reduce_scatterv(
            self._handle,
            win_send,       send_offset,
            self._win_recv, 0,
            self._win_flags,
            self._rsv_d_counts.data_ptr(), 0,   # d_counts, d_displs=NULL (derive)
            output.data_ptr(),
            recv_stride, max_shard, dtype_code,
            timeout_us=self._timeout_us,
            stream=self._stream,
        )
        return self._ft_work()

    def send(self, tensors, dst, tag):
        _ft_warn("send", "not fault-tolerant")
        return super().send(tensors, dst, tag)

    def recv(self, tensors, src, tag):
        _ft_warn("recv", "not fault-tolerant")
        return super().recv(tensors, src, tag)

    def barrier(self, opts=None):
        """Fault-tolerant barrier.

        Instead of ``ProcessGroupNCCL.barrier()`` (a plain NCCL barrier with no
        timeout that hangs forever on a killed rank), this runs the FT
        membership/re-sync barrier via ``ft_converge()`` — the same timeout-bounded
        rendezvous the collectives use, honoring ``FT_BARRIER_MODE`` (TCPStore or
        the store-free FT-AllGather barrier).  A non-responding peer is declared
        dead and dropped from the agreed surviving set; the survivors continue.

        Note: this updates membership as a side effect (that is the point of an FT
        barrier).  A slow-but-alive rank is safe as long as it responds within
        ``FT_TIMEOUT_US``.  Returns a completed FTWork (the barrier is CPU-side).
        """
        self.ft_converge()
        return self._ft_work()

    def recv_anysource(self, tensors, tag):
        raise NotImplementedError(
            "FTProcessGroup: recv_anysource is not supported by NCCL"
        )

    # ------------------------------------------------------------------
    # Cleanup — only FT-specific resources; ProcessGroupNCCL owns the rest
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        try:
            self._cuda.stream_synchronize(self._stream)
        except Exception:
            pass
        try:
            self._cuda.event_destroy(self._up_event)
        except Exception:
            pass
        try:
            for ptr, win in list(self._windows.items()):
                if ptr not in self._win_derived:   # skip aliases — same win as parent
                    self._nccl.window_deregister(self._comm_ft, win)
            for ptr in self._sym_ptrs:
                self._nccl.mem_free(ptr)
            self._nccl.window_deregister(self._comm_ft, self._win_flags)
            self._nccl.window_deregister(self._comm_ft, self._win_recv)
            self._nccl.mem_free(self._d_flags)
            self._nccl.mem_free(self._d_recv)
            self._ft.handle_destroy(self._handle)
            self._cuda.stream_destroy(self._stream)
            self._nccl.comm_finalize(self._comm_ft)
            self._nccl.comm_destroy(self._comm_ft)
            # _comm_world is managed by ProcessGroupNCCL — do not touch it here.
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registry — lets callers retrieve the Python FTProcessGroup after
# dist.init_process_group(), since _get_default_group() returns the C-level
# ProcessGroup wrapper which has no access to Python-only methods.
# ---------------------------------------------------------------------------

# This registry OWNS the Python FTProcessGroup. torch.distributed keeps only the
# C++ backend after the factory returns, so without this strong reference the
# Python object (and the C++ instance it holds) is collected mid-init and torch
# dereferences a dangling backend (segfault in init_process_group).
_registry: Dict[int, FTProcessGroup] = {}   # rank -> FTProcessGroup


def get_ft_process_group(rank: Optional[int] = None) -> Optional[FTProcessGroup]:
    """Return the FTProcessGroup instance created in this process.

    ``dist.distributed_c10d._get_default_group()`` returns the C-level
    ``ProcessGroup`` wrapper, which cannot access Python-only methods such as
    ``empty()`` and ``get_result_mask()``.  Use this function instead.

    Args:
        rank: Rank key to look up.  When ``None`` (default) returns the first
              registered instance, which is correct for the single-PG case.

    Returns:
        The ``FTProcessGroup`` Python object, or ``None`` if none was created
        in this process yet.
    """
    if rank is None:
        return next(iter(_registry.values()), None)
    return _registry.get(rank)


# ---------------------------------------------------------------------------
# Register "ft_nccl" backend with torch.distributed
# ---------------------------------------------------------------------------

def _ft_nccl_pg_factory(store, rank: int, size: int, timeout) -> FTProcessGroup:
    pg = FTProcessGroup(store, rank, size, timeout=timeout)
    _registry[rank] = pg
    return pg


if dist.is_available():
    dist.Backend.register_backend(
        "ft_nccl",
        _ft_nccl_pg_factory,
        devices=["cuda"],
    )
