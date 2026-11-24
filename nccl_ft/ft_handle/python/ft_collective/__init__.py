# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ft_collective — Python ctypes bindings for the FTCollectiveHandle API.

Provides fault-tolerant LSA AllGather and AllReduce collectives with:
  - Per-rank readywin barrier (no all-or-nothing deadlock)
  - Configurable timeout that excludes slow/crashed ranks from the output mask
  - Sticky FTStatus error flag readable from host with zero copy

Typical usage::

    from ft_collective import FTLibrary, NCCLBaseLibrary, FT_OK, FT_TIMEOUT

    nccl = NCCLBaseLibrary()
    ft   = FTLibrary()

    comm = nccl.comm_init_rank(nranks, uid, rank)
    mask = [True] * nranks

    handle = ft.handle_create(comm, nranks, mask)
    ag_ctx = ft.allgather_create(handle, count)

    ft.allgather_launch(ag_ctx, my_data, count, timeout_us=10_000_000)   # 10 s

    result_mask = ft.handle_get_result_mask(handle, nranks)
    gathered    = ft.allgather_get_result(ag_ctx, nranks, count)

    if ft.handle_get_error(handle) == FT_TIMEOUT:
        ft.handle_set_mask(handle, result_mask, nranks)
        ft.handle_clear_error(handle)

    ft.allgather_destroy(ag_ctx)
    ft.handle_destroy(handle)
    nccl.comm_finalize(comm)
    nccl.comm_destroy(comm)
"""

from ._version import __version__

from .ft_wrapper import (
    FTLibrary,
    NCCLBaseLibrary,
    CUDARuntime,
    FT_OK,
    FT_TIMEOUT,
    ncclUniqueId,
    ncclComm_t,
    ncclWindow_t,
    cudaStream_t,
    FTStatus_t,
    FTCollectiveHandle_p,
    FTAllGatherCtx_p,
    FTAllReduceCtx_p,
    NCCL_WIN_DEFAULT,
    NCCL_WIN_COLL_SYMMETRIC,
    NCCL_WIN_STRICT_ORDERING,
)

# Importing FTProcessGroup registers the "ft_nccl" torch.distributed backend.
from .ft_process_group import FTProcessGroup, get_ft_process_group, set_active_pg

__all__ = [
    "__version__",
    # Core wrappers
    "FTLibrary",
    "NCCLBaseLibrary",
    "CUDARuntime",
    # Status
    "FT_OK",
    "FT_TIMEOUT",
    # Types
    "ncclUniqueId",
    "ncclComm_t",
    "ncclWindow_t",
    "cudaStream_t",
    "FTStatus_t",
    "FTCollectiveHandle_p",
    "FTAllGatherCtx_p",
    "FTAllReduceCtx_p",
    # Window flags
    "NCCL_WIN_DEFAULT",
    "NCCL_WIN_COLL_SYMMETRIC",
    "NCCL_WIN_STRICT_ORDERING",
    # torch backend
    "FTProcessGroup",
    "get_ft_process_group",
    "set_active_pg",
]
