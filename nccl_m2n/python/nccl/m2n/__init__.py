# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""NCCL M2N: Pythonic wrappers for ``libnccl_m2n.so``.

This module is packaged separately from NCCL4Py as the ``nccl.m2n`` package.
It wraps the global NCCL M2N C API from
``nccl_m2n.h`` while reusing NCCL4Py communicator, window, stream, and buffer
conventions.
"""

from nccl.core.typing import NcclDataType, NcclInvalid
from nccl.m2n.bindings.nccl_m2n import NCCLReshardError as NcclReshardError
from nccl.m2n.api import (
    finalize,
    init,
    reshard,
    xdtensor_reshard,
)
from nccl.m2n.config import Config
from nccl.m2n.constants import MAX_TENSOR_DIMS, MESH_NDIMS
from nccl.m2n.handle import Handle
from nccl.m2n.mesh import Mesh
from nccl.m2n.placement import Replicate, Shard
from nccl.m2n.tensor import DistTensor

__all__ = [
    "MAX_TENSOR_DIMS",
    "MESH_NDIMS",
    "Config",
    "DistTensor",
    "Handle",
    "Mesh",
    "NcclDataType",
    "NcclInvalid",
    "NcclReshardError",
    "Replicate",
    "Shard",
    "finalize",
    "init",
    "reshard",
    "xdtensor_reshard",
]
