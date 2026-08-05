# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated $version_span. Do not modify it directly.

$external_imports


###############################################################################
# Types and constants
###############################################################################

# The generated declarations encode the M2N ABI, including these public
# constants, so wheel builds do not require nccl_m2n.h at build time.
cdef extern from *:
    """
    #include <limits.h>
    enum {
      NCCL_RESHARD_MESH_NDIMS = 2,
      NCCL_RESHARD_MAX_TENSOR_DIMS = 3,
      NCCL_RESHARD_REPLICATE = -1,
      NCCL_M2N_CONFIG_UNDEF_INT = INT_MIN,
      NCCL_M2N_API_MAGIC = 0x4d324e32u,
      NCCL_M2N_API_VERSION = 2u
    };
    """
    enum:
        NCCL_RESHARD_MESH_NDIMS
        NCCL_RESHARD_MAX_TENSOR_DIMS
        NCCL_RESHARD_REPLICATE
        NCCL_M2N_CONFIG_UNDEF_INT
        NCCL_M2N_API_MAGIC
        NCCL_M2N_API_VERSION

# enums
$enum_decls

# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaError_t 'cudaError_t'

$type_decls


###############################################################################
# Functions
###############################################################################

$func_decls

# Keep the error-detail query on the original no-throw Cython contract.
cdef const char* ncclM2nGetLastError() noexcept nogil
