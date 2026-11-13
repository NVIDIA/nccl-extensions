# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python ctypes wrapper for the FTCollectiveHandle fault-tolerant collective API.

Loads libft_collective.so and libnccl.so at runtime — no compilation needed from
the Python side beyond building the .so with the provided Makefile.

Library search order (for libft_collective.so):
  1. FT_COLLECTIVE_LIB environment variable (explicit path)
  2. LD_LIBRARY_PATH / system default

Library search order (for libnccl.so):
  1. NCCL_HOME/lib/libnccl.so
  2. LD_LIBRARY_PATH / system default

ASYNCHRONOUS LAUNCHES
---------------------
All collective launch functions (allgather_launch, allreduce_launch, allgather,
allreduce) return immediately after enqueueing work on the CUDA stream.  The
caller MUST call cuda.stream_synchronize(stream) before:
  - Reading handle_get_error() or handle_get_result_mask()
  - Reading any recv/output buffer contents

Example::

    ft.allgather_launch(ctx, data, count, timeout_us=10_000_000)
    cuda.stream_synchronize(0)                        # wait for kernel
    status = ft.handle_get_error(handle)
    mask   = ft.handle_get_result_mask(handle, nranks)
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Basic NCCL / CUDA types
# ---------------------------------------------------------------------------

ncclResult_t  = ctypes.c_int
ncclComm_t    = ctypes.c_void_p
cudaStream_t  = ctypes.c_void_p

# ncclUniqueId is a 128-byte opaque struct.
class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]

# ---------------------------------------------------------------------------
# FT-collective types
# ---------------------------------------------------------------------------

FTStatus_t              = ctypes.c_int
FTCollectiveHandle_p    = ctypes.c_void_p   # FTCollectiveHandle*
FTAllGatherCtx_p        = ctypes.c_void_p   # FTAllGatherCtx*
FTAllReduceCtx_p        = ctypes.c_void_p   # FTAllReduceCtx*

# ncclWindow_t is an opaque pointer (ncclWindow_vidmem*).
# Pass nccl4py's RegisteredWindowHandle.handle (a Python int) wherever this is expected.
ncclWindow_t            = ctypes.c_void_p   # ncclWindow_t

# FTStatus enum values
FT_OK                = 0
FT_TIMEOUT           = 1
FT_CAPACITY_EXCEEDED = 2   # a2av send_count > recv_capacity (push was clamped)

# FTDataType enum values (must match ft_collective.h)
FT_DTYPE_FLOAT32  = 0
FT_DTYPE_FLOAT16  = 1
FT_DTYPE_BFLOAT16 = 2

# ncclCommWindowRegister flags (from nccl.h)
NCCL_WIN_DEFAULT        = 0x00
NCCL_WIN_COLL_SYMMETRIC = 0x01   # symmetric virtual address across all ranks
NCCL_WIN_STRICT_ORDERING = 0x02


# ---------------------------------------------------------------------------
# Function descriptor helper (mirrors nccl_ep pattern)
# ---------------------------------------------------------------------------

class CUDARuntime:
    """Minimal libcudart.so wrapper for device memory and stream operations."""

    def __init__(self) -> None:
        for candidate in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
            try:
                self._lib = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        else:
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            self._lib = ctypes.CDLL(os.path.join(cuda_home, "lib64", "libcudart.so"))

        self._lib.cudaMemset.restype  = ctypes.c_int
        self._lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self._lib.cudaMemcpy.restype  = ctypes.c_int
        self._lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_size_t, ctypes.c_int]
        self._lib.cudaStreamSynchronize.restype  = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_size_t]

        self._lib.cudaStreamCreate.restype  = ctypes.c_int
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        self._lib.cudaStreamCreateWithFlags.restype  = ctypes.c_int
        self._lib.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint
        ]
        self._lib.cudaStreamDestroy.restype  = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_size_t]

        self._lib.cudaMemsetAsync.restype  = ctypes.c_int
        self._lib.cudaMemsetAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t
        ]
        self._lib.cudaMemcpyAsync.restype  = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_size_t,
        ]
        # cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
        self._lib.cudaMemcpy2DAsync.restype  = ctypes.c_int
        self._lib.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t,
        ]

        # cudaEventDisableTiming avoids perf overhead of timestamp collection.
        _cudaEventDisableTiming = 0x02
        self._lib.cudaEventCreateWithFlags.restype  = ctypes.c_int
        self._lib.cudaEventCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint
        ]
        self._lib.cudaEventDestroy.restype  = ctypes.c_int
        self._lib.cudaEventDestroy.argtypes = [ctypes.c_size_t]
        self._lib.cudaEventRecord.restype  = ctypes.c_int
        self._lib.cudaEventRecord.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self._lib.cudaEventQuery.restype  = ctypes.c_int
        self._lib.cudaEventQuery.argtypes = [ctypes.c_size_t]
        self._lib.cudaStreamWaitEvent.restype  = ctypes.c_int
        self._lib.cudaStreamWaitEvent.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint
        ]
        self._cuda_event_disable_timing = _cudaEventDisableTiming

    def memset(self, ptr: int, value: int, size: int) -> None:
        """Zero or fill device memory (cudaMemset)."""
        ret = self._lib.cudaMemset(ctypes.c_void_p(ptr), ctypes.c_int(value),
                                   ctypes.c_size_t(size))
        if ret != 0:
            raise RuntimeError(f"cudaMemset failed: {ret}")

    def memcpy_h2d(self, dst: int, src_arr: ctypes.Array, size: int) -> None:
        """Copy from host ctypes array to device pointer (cudaMemcpyHostToDevice=1)."""
        ret = self._lib.cudaMemcpy(ctypes.c_void_p(dst), src_arr,
                                   ctypes.c_size_t(size), ctypes.c_int(1))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed: {ret}")

    def memcpy_d2h(self, dst_arr: ctypes.Array, src: int, size: int) -> None:
        """Copy from device pointer to host ctypes array (cudaMemcpyDeviceToHost=2)."""
        ret = self._lib.cudaMemcpy(dst_arr, ctypes.c_void_p(src),
                                   ctypes.c_size_t(size), ctypes.c_int(2))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed: {ret}")

    def stream_synchronize(self, stream: int = 0) -> None:
        """Block until all work on the given CUDA stream is complete.

        Must be called after any collective launch before reading error
        status, result masks, or output buffers.

        Args:
            stream: CUDA stream handle (int).  0 = default stream.
        """
        ret = self._lib.cudaStreamSynchronize(ctypes.c_size_t(stream or 0))
        if ret != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed: {ret}")

    def stream_create(self) -> int:
        """Create a non-blocking CUDA stream and return its handle (int).

        Uses cudaStreamNonBlocking (0x01) so the stream does not implicitly
        synchronize with the legacy/default CUDA stream.  This is required for
        CUDA graph capture: blocking streams cause cudaErrorStreamCaptureImplicit
        because CUDA automatically makes the legacy stream depend on all blocking
        streams, which is forbidden inside a capture region.
        """
        _CUDA_STREAM_NON_BLOCKING = ctypes.c_uint(0x01)
        s = ctypes.c_size_t()
        ret = self._lib.cudaStreamCreateWithFlags(
            ctypes.byref(s), _CUDA_STREAM_NON_BLOCKING
        )
        if ret != 0:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed: {ret}")
        return s.value

    def stream_destroy(self, stream: int) -> None:
        """Destroy a CUDA stream created with stream_create()."""
        ret = self._lib.cudaStreamDestroy(ctypes.c_size_t(stream))
        if ret != 0:
            raise RuntimeError(f"cudaStreamDestroy failed: {ret}")

    def memset_async(self, ptr: int, value: int, size: int, stream: int) -> None:
        """Asynchronously fill device memory on the given CUDA stream."""
        ret = self._lib.cudaMemsetAsync(
            ctypes.c_void_p(ptr), ctypes.c_int(value),
            ctypes.c_size_t(size), ctypes.c_size_t(stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemsetAsync failed: {ret}")

    def memcpy_d2d_async(self, dst: int, src: int, size: int, stream: int) -> None:
        """Asynchronously copy device-to-device memory on the given CUDA stream."""
        _DEVICE_TO_DEVICE = 3
        ret = self._lib.cudaMemcpyAsync(
            ctypes.c_void_p(dst), ctypes.c_void_p(src),
            ctypes.c_size_t(size), ctypes.c_int(_DEVICE_TO_DEVICE),
            ctypes.c_size_t(stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2D failed: {ret}")

    def memcpy_h2d_async(self, dst: int, src: int, size: int, stream: int) -> None:
        """Asynchronously copy host-to-device on the given CUDA stream.

        Ordered on ``stream``, so a kernel launched afterwards on the same stream
        sees the data without a host synchronize. Use a PINNED host source for a
        truly non-blocking copy (pageable is ordered but may briefly block)."""
        _HOST_TO_DEVICE = 1
        ret = self._lib.cudaMemcpyAsync(
            ctypes.c_void_p(dst), ctypes.c_void_p(src),
            ctypes.c_size_t(size), ctypes.c_int(_HOST_TO_DEVICE),
            ctypes.c_size_t(stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed: {ret}")

    def memcpy_2d_d2d_async(self, dst: int, dpitch: int, src: int, spitch: int,
                            width: int, height: int, stream: int) -> None:
        """Strided device-to-device 2D copy on the given stream.

        Copies ``height`` rows of ``width`` bytes each, advancing ``dpitch`` in
        the destination and ``spitch`` in the source per row.  Used to gather a
        padded-stride recv buffer (spitch = padded slice stride) into a packed
        contiguous output (dpitch = width) in a single launch.
        """
        _DEVICE_TO_DEVICE = 3
        ret = self._lib.cudaMemcpy2DAsync(
            ctypes.c_void_p(dst), ctypes.c_size_t(dpitch),
            ctypes.c_void_p(src), ctypes.c_size_t(spitch),
            ctypes.c_size_t(width), ctypes.c_size_t(height),
            ctypes.c_int(_DEVICE_TO_DEVICE), ctypes.c_size_t(stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync D2D failed: {ret}")

    def event_create(self) -> int:
        """Create a timing-disabled CUDA event and return its handle (int)."""
        e = ctypes.c_size_t()
        ret = self._lib.cudaEventCreateWithFlags(
            ctypes.byref(e), ctypes.c_uint(self._cuda_event_disable_timing)
        )
        if ret != 0:
            raise RuntimeError(f"cudaEventCreateWithFlags failed: {ret}")
        return e.value

    def event_destroy(self, event: int) -> None:
        """Destroy a CUDA event created with event_create()."""
        ret = self._lib.cudaEventDestroy(ctypes.c_size_t(event))
        if ret != 0:
            raise RuntimeError(f"cudaEventDestroy failed: {ret}")

    def event_record(self, event: int, stream: int) -> None:
        """Record a CUDA event on the given stream."""
        ret = self._lib.cudaEventRecord(
            ctypes.c_size_t(event), ctypes.c_size_t(stream)
        )
        if ret != 0:
            raise RuntimeError(f"cudaEventRecord failed: {ret}")

    def event_query(self, event: int) -> bool:
        """Query completion without blocking; raise on CUDA errors."""
        ret = self._lib.cudaEventQuery(ctypes.c_size_t(event))
        if ret == 0:
            return True
        if ret == 600:            # cudaErrorNotReady
            return False
        raise RuntimeError(f"cudaEventQuery failed: {ret}")

    def stream_wait_event(self, stream: int, event: int) -> None:
        """Make stream wait for event before executing subsequent work.

        Does not block the CPU — inserts a GPU-side dependency only.
        """
        ret = self._lib.cudaStreamWaitEvent(
            ctypes.c_size_t(stream), ctypes.c_size_t(event), ctypes.c_uint(0)
        )
        if ret != 0:
            raise RuntimeError(f"cudaStreamWaitEvent failed: {ret}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


# ---------------------------------------------------------------------------
# NCCLBaseLibrary — thin wrapper around libnccl.so for comm creation
# ---------------------------------------------------------------------------

class NCCLBaseLibrary:
    """Minimal libnccl.so wrapper for communicator management."""

    _exported = [
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        Function("ncclGetUniqueId",    ncclResult_t,    [ctypes.POINTER(ncclUniqueId)]),
        Function("ncclCommInitRank",   ncclResult_t,    [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int
        ]),
        Function("ncclCommFinalize",   ncclResult_t,    [ncclComm_t]),
        Function("ncclCommDestroy",    ncclResult_t,    [ncclComm_t]),

        # Symmetric memory allocation (collective — all ranks must call together)
        Function("ncclMemAlloc", ncclResult_t, [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
        ]),
        Function("ncclMemFree", ncclResult_t, [ctypes.c_void_p]),

        # Window registration / deregistration
        Function("ncclCommWindowRegister", ncclResult_t, [
            ncclComm_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ncclWindow_t), ctypes.c_int,
        ]),
        Function("ncclCommWindowDeregister", ncclResult_t, [ncclComm_t, ncclWindow_t]),
    ]

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = self._find()
        self._lib = ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
        self._funcs = {}
        for f in self._exported:
            fn = getattr(self._lib, f.name)
            fn.restype  = f.restype
            fn.argtypes = f.argtypes
            self._funcs[f.name] = fn

    @staticmethod
    def _find() -> str:
        nccl_home = os.environ.get("NCCL_HOME", "")
        for candidate in [
            os.path.join(nccl_home, "lib", "libnccl.so.2"),
            os.path.join(nccl_home, "lib", "libnccl.so"),
        ]:
            if os.path.exists(candidate):
                return candidate
        for name in ["nccl.2", "nccl"]:
            lib = ctypes.util.find_library(name)
            if lib:
                return lib
        raise RuntimeError(
            "Cannot find libnccl.so. Set NCCL_HOME or add the library to LD_LIBRARY_PATH."
        )

    def check(self, result: int) -> None:
        if result != 0:
            msg = self._funcs["ncclGetErrorString"](result)
            raise RuntimeError(f"NCCL error {result}: {msg.decode()}")

    def get_unique_id(self) -> ncclUniqueId:
        uid = ncclUniqueId()
        self.check(self._funcs["ncclGetUniqueId"](ctypes.byref(uid)))
        return uid

    def comm_init_rank(self, nranks: int, uid: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.check(self._funcs["ncclCommInitRank"](
            ctypes.byref(comm), nranks, uid, rank
        ))
        return comm

    def comm_finalize(self, comm: ncclComm_t) -> None:
        self.check(self._funcs["ncclCommFinalize"](comm))

    def comm_destroy(self, comm: ncclComm_t) -> None:
        self.check(self._funcs["ncclCommDestroy"](comm))

    def mem_alloc(self, size: int) -> int:
        """Allocate symmetric device memory (ncclMemAlloc).

        COLLECTIVE: all ranks must call simultaneously with the same size.

        Args:
            size: Allocation size in bytes.

        Returns:
            Device pointer as a Python int (same virtual address on every rank).
        """
        ptr = ctypes.c_void_p()
        self.check(self._funcs["ncclMemAlloc"](ctypes.byref(ptr), size))
        return ptr.value

    def mem_free(self, ptr: int) -> None:
        """Free symmetric device memory allocated with mem_alloc (ncclMemFree).

        COLLECTIVE: all ranks must call simultaneously.
        """
        self.check(self._funcs["ncclMemFree"](ctypes.c_void_p(ptr)))

    def window_register(self, comm: ncclComm_t, ptr: int, size: int,
                        flags: int = NCCL_WIN_COLL_SYMMETRIC) -> int:
        """Register a symmetric device buffer as an ncclWindow_t.

        COLLECTIVE: all ranks must call simultaneously.

        Args:
            comm:  NCCL communicator.
            ptr:   Device pointer (from mem_alloc or ncclMemAlloc).
            size:  Buffer size in bytes.
            flags: Window flags (default: NCCL_WIN_COLL_SYMMETRIC).

        Returns:
            ncclWindow_t handle as a Python int.
        """
        win = ncclWindow_t()
        self.check(self._funcs["ncclCommWindowRegister"](
            comm, ctypes.c_void_p(ptr), size, ctypes.byref(win), flags
        ))
        return win.value

    def window_deregister(self, comm: ncclComm_t, win: int) -> None:
        """Deregister an ncclWindow_t (ncclCommWindowDeregister).

        COLLECTIVE: all ranks must call simultaneously.
        """
        self.check(self._funcs["ncclCommWindowDeregister"](comm, ncclWindow_t(win)))


# ---------------------------------------------------------------------------
# FTLibrary — wrapper for libft_collective.so
# ---------------------------------------------------------------------------

class FTLibrary:
    """ctypes wrapper for libft_collective.so.

    Exposes the FTCollectiveHandle API:
      - Handle lifecycle:  create / destroy / set_mask / get_result_mask /
                           get_error / clear_error
      - AllGather context: create / destroy / set_data / launch / get_result
      - AllReduce context: create / destroy / set_data / launch / get_result
      - Window-based API:  allgather / allreduce (caller supplies nccl4py windows)

    The handle automatically selects LSA or GIN kernels at creation time based
    on communicator properties (LSA preferred; GIN used as fallback).

    All launch functions are ASYNCHRONOUS.  The caller must call
    cuda.stream_synchronize(stream) before reading results or error status.
    """

    _exported = [
        # --- Handle ---
        Function("ftHandleCreate", ncclResult_t, [
            ncclComm_t, ctypes.c_int,
            ctypes.POINTER(ctypes.c_bool),          # initial_mask[nRanks]
            ctypes.POINTER(FTCollectiveHandle_p),   # out
        ]),
        Function("ftHandleDestroy",       None,          [FTCollectiveHandle_p]),
        Function("ftHandleSetActiveMask", None,          [FTCollectiveHandle_p,
                                                          ctypes.POINTER(ctypes.c_bool)]),
        Function("ftHandleSetMask",       None,          [FTCollectiveHandle_p,
                                                          ctypes.POINTER(ctypes.c_bool)]),
        Function("ftHandleSetSelfAdapt",  None,          [FTCollectiveHandle_p,
                                                          ctypes.c_bool]),
        Function("ftHandleGetResultMask", ctypes.POINTER(ctypes.c_bool),
                                                         [FTCollectiveHandle_p]),
        Function("ftHandleGetError",      FTStatus_t,    [FTCollectiveHandle_p]),
        Function("ftHandleClearError",    None,          [FTCollectiveHandle_p]),
        Function("ftRejoin",              ncclResult_t,  [FTCollectiveHandle_p,
                                                          cudaStream_t,
                                                          ctypes.c_uint64]),
        Function("ftHandleGetGeneration", ctypes.c_uint64, [FTCollectiveHandle_p]),

        # --- AllGather context ---
        Function("ftAllGatherCreate", ncclResult_t, [
            FTCollectiveHandle_p, ctypes.c_size_t,
            ctypes.POINTER(FTAllGatherCtx_p),
        ]),
        Function("ftAllGatherDestroy",   None,         [FTAllGatherCtx_p]),
        Function("ftAllGatherSetData",   ncclResult_t, [
            FTAllGatherCtx_p, ctypes.POINTER(ctypes.c_float)
        ]),
        Function("ftAllGatherLaunch",    ncclResult_t, [
            FTAllGatherCtx_p, cudaStream_t, ctypes.c_uint64
        ]),
        Function("ftAllGatherGetResult", ncclResult_t, [
            FTAllGatherCtx_p, ctypes.POINTER(ctypes.c_float)
        ]),

        # --- AllReduce context ---
        Function("ftAllReduceCreate", ncclResult_t, [
            FTCollectiveHandle_p, ctypes.c_size_t,
            ctypes.POINTER(FTAllReduceCtx_p),
        ]),
        Function("ftAllReduceDestroy",   None,         [FTAllReduceCtx_p]),
        Function("ftAllReduceSetData",   ncclResult_t, [
            FTAllReduceCtx_p, ctypes.POINTER(ctypes.c_float)
        ]),
        Function("ftAllReduceLaunch",    ncclResult_t, [
            FTAllReduceCtx_p, cudaStream_t, ctypes.c_uint64
        ]),
        Function("ftAllReduceGetResult", ncclResult_t, [
            FTAllReduceCtx_p, ctypes.POINTER(ctypes.c_float)
        ]),

        # --- Window-based API ---
        # win_flags is ignored when the handle uses the GIN kernel.
        Function("ftAllGather", ncclResult_t, [
            FTCollectiveHandle_p,
            ncclWindow_t, ctypes.c_size_t,   # win_send, sendoffset
            ncclWindow_t, ctypes.c_size_t,   # win_recv, recvoffset
            ncclWindow_t,                    # win_flags (ignored for GIN)
            ctypes.c_size_t,                 # count
            cudaStream_t,                    # stream
            ctypes.c_uint64,                 # timeout_us
        ]),
        Function("ftAllToAll", ncclResult_t, [
            FTCollectiveHandle_p,
            ncclWindow_t, ctypes.c_size_t,   # win_send, sendoffset
            ncclWindow_t, ctypes.c_size_t,   # win_recv, recvoffset
            ncclWindow_t,                    # win_flags
            ctypes.c_size_t,                 # count (per-destination)
            cudaStream_t,                    # stream
            ctypes.c_uint64,                 # timeout_us
        ]),
        Function("ftAllToAllv", ncclResult_t, [
            FTCollectiveHandle_p,
            ncclWindow_t, ctypes.c_size_t,   # win_send, sendoffset
            ncclWindow_t, ctypes.c_size_t,   # win_recv, recvoffset
            ncclWindow_t,                    # win_flags
            ctypes.c_void_p,                 # d_send_counts (device int32[nRanks])
            ctypes.c_void_p,                 # d_send_displs (device int64[nRanks])
            ctypes.c_size_t,                 # recv_stride_bytes
            ctypes.c_size_t,                 # max_send_count (coop-grid hint)
            ctypes.c_void_p,                 # output (recv, device ptr)
            ctypes.c_void_p,                 # d_recv_counts (packed mode, or NULL)
            ctypes.c_void_p,                 # d_recv_displs (packed mode, or NULL)
            ctypes.c_void_p,                 # d_recv_counts_out (auto mode, or NULL)
            ctypes.c_size_t,                 # recv_capacity (auto: slot stride, else 0)
            ctypes.c_size_t,                 # row_size (elems/row; 1 = element counts)
            ctypes.c_int,                    # dtype (FTDataType)
            cudaStream_t,                    # stream
            ctypes.c_uint64,                 # timeout_us
        ]),
        Function("ftAllGatherv", ncclResult_t, [
            FTCollectiveHandle_p,
            ncclWindow_t, ctypes.c_size_t,   # win_send, sendoffset
            ncclWindow_t, ctypes.c_size_t,   # win_recv, recvoffset
            ncclWindow_t,                    # win_flags
            ctypes.c_void_p,                 # d_send_counts (device int32[nRanks])
            ctypes.c_void_p,                 # output (device, packed)
            ctypes.c_void_p,                 # d_recv_counts_out (device int32[nRanks])
            ctypes.c_size_t,                 # recv_capacity (rows)
            ctypes.c_size_t,                 # row_bytes
            ctypes.c_size_t,                 # recv_stride (bytes)
            ctypes.c_size_t,                 # max_send_count (rows)
            cudaStream_t,                    # stream
            ctypes.c_uint64,                 # timeout_us
        ]),
        Function("ftReduceScatterv", ncclResult_t, [
            FTCollectiveHandle_p,
            ncclWindow_t, ctypes.c_size_t,   # win_send, sendoffset
            ncclWindow_t, ctypes.c_size_t,   # win_recv, recvoffset
            ncclWindow_t,                    # win_flags
            ctypes.c_void_p,                 # d_counts  (device int32[nRanks])
            ctypes.c_void_p,                 # d_displs  (device int64[nRanks] or NULL)
            ctypes.c_void_p,                 # d_output
            ctypes.c_size_t,                 # recv_stride (bytes)
            ctypes.c_size_t,                 # max_shard_count (elements)
            ctypes.c_int,                    # dtype (FTDataType)
            cudaStream_t,                    # stream
            ctypes.c_uint64,                 # timeout_us
        ]),
        Function("ftAllReduce", ncclResult_t, [
            FTCollectiveHandle_p,
            ncclWindow_t, ctypes.c_size_t,   # win_send, sendoffset
            ncclWindow_t, ctypes.c_size_t,   # win_recv, recvoffset
            ncclWindow_t,                    # win_flags (ignored for GIN)
            ctypes.c_void_p,                 # d_output (device pointer)
            ctypes.c_size_t,                 # count
            ctypes.c_int,                    # dtype (FTDataType enum)
            cudaStream_t,                    # stream
            ctypes.c_uint64,                 # timeout_us
        ]),
    ]

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = self._find()
        self._lib   = ctypes.CDLL(so_file)
        self._funcs: dict = {}
        for f in self._exported:
            fn = getattr(self._lib, f.name)
            fn.restype  = f.restype
            fn.argtypes = f.argtypes
            self._funcs[f.name] = fn

    @staticmethod
    def _find() -> str:
        explicit = os.environ.get("FT_COLLECTIVE_LIB")
        if explicit and os.path.exists(explicit):
            return explicit
        lib = ctypes.util.find_library("ft_collective")
        if lib:
            return lib
        raise RuntimeError(
            "Cannot find libft_collective.so. "
            "Build it with 'make' in ft_handle/ and set FT_COLLECTIVE_LIB or LD_LIBRARY_PATH."
        )

    def _check(self, result: int) -> None:
        if result != 0:
            raise RuntimeError(f"ft_collective error: ncclResult_t = {result}")

    # ------------------------------------------------------------------
    # Handle API
    # ------------------------------------------------------------------

    def handle_create(self, comm: ncclComm_t, nranks: int,
                      initial_mask: Sequence[bool]) -> FTCollectiveHandle_p:
        """Create an FTCollectiveHandle.

        Automatically selects LSA or GIN kernel backend based on communicator
        properties (LSA preferred; GIN used as fallback).

        Args:
            comm:         NCCL communicator (ncclComm_t opaque pointer value).
            nranks:       Total number of ranks.
            initial_mask: Boolean sequence of length nranks (True = active).

        Returns:
            Opaque handle pointer (pass to all subsequent API calls).

        Raises:
            RuntimeError: If neither LSA nor GIN is available on the communicator.
        """
        mask_arr = (ctypes.c_bool * nranks)(*initial_mask)
        handle   = FTCollectiveHandle_p()
        self._check(self._funcs["ftHandleCreate"](
            comm, nranks, mask_arr, ctypes.byref(handle)
        ))
        return handle

    def handle_destroy(self, handle: FTCollectiveHandle_p) -> None:
        self._funcs["ftHandleDestroy"](handle)

    def set_active_mask(self, handle: FTCollectiveHandle_p,
                        mask: Sequence[bool], nranks: int) -> None:
        """Replace the active-membership mask for subsequent launches.

        COORDINATED: every rank running the next collective together must install
        the SAME mask, or the collective can deadlock on a split membership.
        """
        mask_arr = (ctypes.c_bool * nranks)(*mask)
        self._funcs["ftHandleSetActiveMask"](handle, mask_arr)

    def handle_set_mask(self, handle: FTCollectiveHandle_p,
                        mask: Sequence[bool], nranks: int) -> None:
        """Deprecated alias for set_active_mask()."""
        self.set_active_mask(handle, mask, nranks)

    def handle_set_self_adapt(self, handle: FTCollectiveHandle_p,
                              enable: bool) -> None:
        """Enable/disable device-side self-adaptation of the input mask."""
        self._funcs["ftHandleSetSelfAdapt"](handle, ctypes.c_bool(enable))

    def rejoin(self, handle: FTCollectiveHandle_p, timeout_us: int,
               stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue a collective rejoin (full-world). ASYNCHRONOUS.

        Re-admits stalled-but-alive ranks without changing the communicator.
        After ``cuda.stream_synchronize()``, read the responder set with
        ``handle_get_result_mask()`` and install it via ``set_active_mask()``;
        the kernel also synchronizes the on-device generation counter.
        """
        self._check(self._funcs["ftRejoin"](
            handle,
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def handle_get_generation(self, handle: FTCollectiveHandle_p) -> int:
        """Read the on-device generation counter (D2H). After rejoin, every
        survivor should report the same value."""
        return int(self._funcs["ftHandleGetGeneration"](handle))

    def handle_get_result_mask(self, handle: FTCollectiveHandle_p,
                               nranks: int) -> List[bool]:
        """Copy d_mask_out → h_mask_out and return the result mask.

        Must be called after cuda.stream_synchronize().
        """
        ptr = self._funcs["ftHandleGetResultMask"](handle)
        return [bool(ptr[r]) for r in range(nranks)]

    def handle_get_error(self, handle: FTCollectiveHandle_p) -> int:
        """Return FT_OK (0) or FT_TIMEOUT (1).

        Reads mapped pinned memory — no device copy.
        Must be called after cuda.stream_synchronize().
        """
        return int(self._funcs["ftHandleGetError"](handle))

    def handle_clear_error(self, handle: FTCollectiveHandle_p) -> None:
        """Reset error status to FT_OK.  Call only when no kernel is in flight."""
        self._funcs["ftHandleClearError"](handle)

    # ------------------------------------------------------------------
    # AllGather context API
    # ------------------------------------------------------------------

    def allgather_create(self, handle: FTCollectiveHandle_p,
                         count: int) -> FTAllGatherCtx_p:
        """Allocate symmetric buffers and register windows for AllGather.

        For LSA handles: allocates send, recv, and flags windows.
        For GIN handles: allocates send and recv windows only (no flags).
        """
        ctx = FTAllGatherCtx_p()
        self._check(self._funcs["ftAllGatherCreate"](
            handle, ctypes.c_size_t(count), ctypes.byref(ctx)
        ))
        return ctx

    def allgather_destroy(self, ctx: FTAllGatherCtx_p) -> None:
        self._funcs["ftAllGatherDestroy"](ctx)

    def allgather_launch(self, ctx: FTAllGatherCtx_p,
                         data: Sequence[float], count: int,
                         timeout_us: int,
                         stream: Optional[cudaStream_t] = None) -> None:
        """Copy data into the send buffer and enqueue the AllGather kernel.

        ASYNCHRONOUS: returns before the kernel completes.
        Call cuda.stream_synchronize(stream) before reading results.

        Args:
            ctx:        Context returned by allgather_create().
            data:       Float sequence of length count (host data to send).
            count:      Number of float elements.
            timeout_us: Timeout in microseconds for the collective.
            stream:     CUDA stream handle (None = default stream 0).
        """
        arr = (ctypes.c_float * count)(*data)
        self._check(self._funcs["ftAllGatherSetData"](ctx, arr))
        self._check(self._funcs["ftAllGatherLaunch"](
            ctx,
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def allgather_get_result(self, ctx: FTAllGatherCtx_p,
                             nranks: int, count: int) -> List[List[float]]:
        """Copy gathered data to host.  Returns list-of-lists [rank][elem].

        Must be called after cuda.stream_synchronize().
        """
        total = nranks * count
        out   = (ctypes.c_float * total)()
        self._check(self._funcs["ftAllGatherGetResult"](ctx, out))
        return [[out[r * count + i] for i in range(count)] for r in range(nranks)]

    # ------------------------------------------------------------------
    # AllReduce context API
    # ------------------------------------------------------------------

    def allreduce_create(self, handle: FTCollectiveHandle_p,
                         count: int) -> FTAllReduceCtx_p:
        """Allocate symmetric buffers and register windows for AllReduce.

        For LSA handles: allocates send, recv, and flags windows.
        For GIN handles: allocates send and recv windows only (no flags).
        """
        ctx = FTAllReduceCtx_p()
        self._check(self._funcs["ftAllReduceCreate"](
            handle, ctypes.c_size_t(count), ctypes.byref(ctx)
        ))
        return ctx

    def allreduce_destroy(self, ctx: FTAllReduceCtx_p) -> None:
        self._funcs["ftAllReduceDestroy"](ctx)

    def allreduce_launch(self, ctx: FTAllReduceCtx_p,
                         data: Sequence[float], count: int,
                         timeout_us: int,
                         stream: Optional[cudaStream_t] = None) -> None:
        """Copy data into the send buffer and enqueue the AllReduce kernel.

        ASYNCHRONOUS: returns before the kernel completes.
        Call cuda.stream_synchronize(stream) before reading results.
        """
        arr = (ctypes.c_float * count)(*data)
        self._check(self._funcs["ftAllReduceSetData"](ctx, arr))
        self._check(self._funcs["ftAllReduceLaunch"](
            ctx,
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def allreduce_get_result(self, ctx: FTAllReduceCtx_p,
                             count: int) -> List[float]:
        """Copy reduced output to host.  Returns float[count].

        Must be called after cuda.stream_synchronize().
        """
        out = (ctypes.c_float * count)()
        self._check(self._funcs["ftAllReduceGetResult"](ctx, out))
        return list(out)

    # ------------------------------------------------------------------
    # Window-based API
    #
    # Windows are created and owned by the caller using nccl4py:
    #
    #   from nccl.core import mem_alloc
    #   from nccl.core.constants import WindowFlag
    #   buf  = mem_alloc(size_bytes)
    #   win  = comm.register_window(buf, WindowFlag.CollSymmetric)
    #
    #   ft.allgather(handle, win_send.handle, 0, win_recv.handle, 0,
    #                win_flags.handle, count, timeout_us=10_000_000)
    #   cuda.stream_synchronize(0)   # REQUIRED before reading results
    #
    # For GIN handles, win_flags is ignored (pass None or any value).
    # The caller must zero win_recv (and win_flags for LSA) before each call.
    # The handle resets its own readywin (LSA) or GIN signals internally.
    # ------------------------------------------------------------------

    def allgather(self, handle: FTCollectiveHandle_p,
                  win_send: int, sendoffset: int,
                  win_recv: int, recvoffset: int,
                  win_flags: Optional[int],
                  count: int,
                  timeout_us: int,
                  stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue fault-tolerant AllGather with caller-supplied nccl4py windows.

        ASYNCHRONOUS: returns before the kernel completes.
        Call cuda.stream_synchronize(stream) before reading results.

        Args:
            handle:      FTCollectiveHandle created with handle_create().
            win_send:    RegisteredWindowHandle.handle — symmetric send buffer.
            sendoffset:  Byte offset into win_send.
            win_recv:    RegisteredWindowHandle.handle — symmetric recv buffer.
                         Must be zeroed by caller before each call.
            recvoffset:  Byte offset into win_recv for slot 0.
            win_flags:   RegisteredWindowHandle.handle — symmetric flags buffer
                         (LSA only, nRanks uint64_t). Must be zeroed before
                         each call. Ignored for GIN handles (pass None).
            count:       Float elements per rank.
            timeout_us:  Timeout in microseconds.
            stream:      CUDA stream (None = default stream 0).
        """
        self._check(self._funcs["ftAllGather"](
            handle,
            ncclWindow_t(win_send),   ctypes.c_size_t(sendoffset),
            ncclWindow_t(win_recv),   ctypes.c_size_t(recvoffset),
            ncclWindow_t(win_flags),  # ignored for GIN
            ctypes.c_size_t(count),
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def alltoall(self, handle: FTCollectiveHandle_p,
                 win_send: int, sendoffset: int,
                 win_recv: int, recvoffset: int,
                 win_flags: Optional[int],
                 count: int,
                 timeout_us: int,
                 stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue fault-tolerant uniform all-to-all (Phase 0), window-based.

        ASYNCHRONOUS. `count` is the per-destination float-element count: win_send
        holds nRanks*count elements (chunk j -> rank j), win_recv holds nRanks
        padded slots (slot i = data from rank i). LSA only — raises on a GIN
        handle (ncclInvalidUsage) so the caller can fall back to stock NCCL.
        """
        self._check(self._funcs["ftAllToAll"](
            handle,
            ncclWindow_t(win_send),   ctypes.c_size_t(sendoffset),
            ncclWindow_t(win_recv),   ctypes.c_size_t(recvoffset),
            ncclWindow_t(win_flags),
            ctypes.c_size_t(count),
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def alltoallv(self, handle: FTCollectiveHandle_p,
                  win_send: int, sendoffset: int,
                  win_recv: int, recvoffset: int,
                  win_flags: Optional[int],
                  d_send_counts: int, d_send_displs: int,
                  recv_stride_bytes: int, max_send_count: int,
                  output: int, d_recv_counts: int, d_recv_displs: int,
                  d_recv_counts_out: int, recv_capacity: int, row_size: int,
                  dtype: int, timeout_us: int,
                  stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue fault-tolerant variable all-to-all (Phase 1), window-based.

        ASYNCHRONOUS. d_send_counts/d_send_displs are device pointers to nRanks
        int32/int64 arrays in NATIVE element units of `dtype`; chunk d is sent to
        rank d's scratch slot[myRank]. The kernel then COMPACTS the per-source
        slots into `output` (device ptr) using d_recv_counts/d_recv_displs (also
        native-element device arrays). recv_stride_bytes = fixed slot stride;
        max_send_count = single- vs multi-CTA hint; dtype = FTDataType. LSA only.
        """
        self._check(self._funcs["ftAllToAllv"](
            handle,
            ncclWindow_t(win_send),   ctypes.c_size_t(sendoffset),
            ncclWindow_t(win_recv),   ctypes.c_size_t(recvoffset),
            ncclWindow_t(win_flags),
            ctypes.c_void_p(d_send_counts),
            ctypes.c_void_p(d_send_displs),
            ctypes.c_size_t(recv_stride_bytes),
            ctypes.c_size_t(max_send_count),
            ctypes.c_void_p(output),
            ctypes.c_void_p(d_recv_counts),
            ctypes.c_void_p(d_recv_displs),
            ctypes.c_void_p(d_recv_counts_out),
            ctypes.c_size_t(recv_capacity),
            ctypes.c_size_t(row_size),
            ctypes.c_int(dtype),
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def allgatherv(self, handle: FTCollectiveHandle_p,
                   win_send: int, sendoffset: int,
                   win_recv: int, recvoffset: int,
                   win_flags: Optional[int],
                   d_send_counts: int, output: int, d_recv_counts_out: int,
                   recv_capacity: int, row_bytes: int, recv_stride: int,
                   max_send_count: int, timeout_us: int,
                   stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue variable all-gather (all_gatherv), auto-recv-sizing + packed.

        ASYNCHRONOUS. Each rank contributes ONE variable chunk (d_send_counts is a
        device int32[nRanks]; this rank pushes d_send_counts[myRank] rows at
        win_send+sendoffset) which every rank receives, concatenated in rank order
        into `output` (device, packed). Count-in-flag: the receiver DISCOVERS each
        source's size (no separate size all-gather) into d_recv_counts_out (device
        int32[nRanks]) and packs on-device. recv_capacity = max rows any source may
        send; row_bytes = bytes/row; recv_stride = per-source scratch slot stride
        (>= recv_capacity*row_bytes); max_send_count = grid hint (rows). LSA only.
        """
        self._check(self._funcs["ftAllGatherv"](
            handle,
            ncclWindow_t(win_send), ctypes.c_size_t(sendoffset),
            ncclWindow_t(win_recv), ctypes.c_size_t(recvoffset),
            ncclWindow_t(win_flags),
            ctypes.c_void_p(d_send_counts),
            ctypes.c_void_p(output),
            ctypes.c_void_p(d_recv_counts_out),
            ctypes.c_size_t(recv_capacity),
            ctypes.c_size_t(row_bytes),
            ctypes.c_size_t(recv_stride),
            ctypes.c_size_t(max_send_count),
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def reduce_scatterv(self, handle: FTCollectiveHandle_p,
                        win_send: int, sendoffset: int,
                        win_recv: int, recvoffset: int,
                        win_flags: Optional[int],
                        d_counts: int, d_displs: int, d_output: int,
                        recv_stride: int, max_shard_count: int, dtype: int,
                        timeout_us: int,
                        stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue variable reduce-scatter (typed SUM), window-based.

        ASYNCHRONOUS. Every rank supplies its full input at win_send+sendoffset
        (sum-of-counts elements); rank r receives shard [displs[r], counts[r])
        summed over surviving contributors into d_output (counts[myRank] elems).
        d_counts is a DEVICE int32[nRanks] shard-size partition; d_displs a DEVICE
        int64[nRanks] offset array (0/NULL = derive on-device). recv_stride = the
        per-contributor scratch slot stride in bytes (>= max_shard*sizeof, 16-byte
        multiple); max_shard_count = grid hint; dtype = FTDataType. LSA only.
        """
        self._check(self._funcs["ftReduceScatterv"](
            handle,
            ncclWindow_t(win_send), ctypes.c_size_t(sendoffset),
            ncclWindow_t(win_recv), ctypes.c_size_t(recvoffset),
            ncclWindow_t(win_flags),
            ctypes.c_void_p(d_counts),
            ctypes.c_void_p(d_displs),
            ctypes.c_void_p(d_output),
            ctypes.c_size_t(recv_stride),
            ctypes.c_size_t(max_shard_count),
            ctypes.c_int(dtype),
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))

    def allreduce(self, handle: FTCollectiveHandle_p,
                  win_send: int, sendoffset: int,
                  win_recv: int, recvoffset: int,
                  win_flags: Optional[int],
                  d_output: int,
                  count: int,
                  timeout_us: int,
                  dtype: int = FT_DTYPE_FLOAT32,
                  stream: Optional[cudaStream_t] = None) -> None:
        """Enqueue fault-tolerant AllReduce with caller-supplied nccl4py windows.

        ASYNCHRONOUS: returns before the kernel completes.
        Call cuda.stream_synchronize(stream) before reading results.

        Args:
            handle:      FTCollectiveHandle created with handle_create().
            win_send:    RegisteredWindowHandle.handle — symmetric send buffer.
            sendoffset:  Byte offset into win_send.
            win_recv:    RegisteredWindowHandle.handle — symmetric gather buffer.
                         Must be zeroed by caller before each call.
            recvoffset:  Byte offset into win_recv for slot 0.
            win_flags:   Flags window (LSA only). Ignored for GIN (pass None).
            d_output:    Raw device pointer (int) for reduction result (count elements).
            count:       Number of elements per rank (in dtype units).
            timeout_us:  Timeout in microseconds.
            dtype:       FT_DTYPE_FLOAT32 / FT_DTYPE_FLOAT16 / FT_DTYPE_BFLOAT16.
                         Reduction accumulates in float32 internally (Option-B).
            stream:      CUDA stream (None = default stream 0).
        """
        self._check(self._funcs["ftAllReduce"](
            handle,
            ncclWindow_t(win_send),   ctypes.c_size_t(sendoffset),
            ncclWindow_t(win_recv),   ctypes.c_size_t(recvoffset),
            ncclWindow_t(win_flags),  # ignored for GIN
            ctypes.c_void_p(d_output),
            ctypes.c_size_t(count),
            ctypes.c_int(dtype),
            stream if stream is not None else cudaStream_t(None),
            ctypes.c_uint64(timeout_us),
        ))
