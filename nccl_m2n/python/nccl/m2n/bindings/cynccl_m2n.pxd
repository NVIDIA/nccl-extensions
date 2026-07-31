# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

###############################################################################
# Types and functions
###############################################################################

cdef extern from "nccl_m2n.h":
    enum:
        NCCL_RESHARD_MESH_NDIMS
        NCCL_RESHARD_MAX_TENSOR_DIMS
        NCCL_RESHARD_REPLICATE
        NCCL_M2N_CONFIG_UNDEF_INT
        NCCL_M2N_API_MAGIC
        NCCL_M2N_API_VERSION

    ctypedef enum ncclResult_t 'ncclResult_t':
        _NCCLRESULT_T_INTERNAL_LOADING_ERROR '((ncclResult_t)-42)' = -42
    ctypedef enum ncclDataType_t 'ncclDataType_t':
        pass
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef void* ncclComm_t 'ncclComm_t'
    ctypedef void* ncclWindow_t 'ncclWindow_t'
    ctypedef void* ncclM2nHandle_t 'ncclM2nHandle_t'

    ctypedef struct ncclMesh_t 'ncclMesh_t':
        int dims[2]
        int startRank

    ctypedef struct ncclDistTensor_t 'ncclDistTensor_t':
        void* dataPtr
        size_t localShape[3]
        int ndims
        ncclDataType_t dtype
        const ncclMesh_t* mesh
        int placements[2]

    ctypedef struct ncclM2nConfig_t 'ncclM2nConfig_t':
        size_t size
        unsigned int magic
        unsigned int version
        int maxCta


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclM2nFinalize(ncclM2nHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef const char* ncclM2nGetLastError() noexcept nogil
cdef ncclResult_t ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
