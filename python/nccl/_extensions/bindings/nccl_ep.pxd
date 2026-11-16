# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.2.0. Do not modify it directly.



# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t


# <<<< END OF PREAMBLE CONTENT >>>>

from .cynccl_ep cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t Comm
ctypedef ncclWindow_t Window
ctypedef ncclEpGroup_t Group
ctypedef ncclEpHandle_t Handle
ctypedef ncclEpAllocFn_t AllocFn
ctypedef ncclEpFreeFn_t FreeFn

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef ncclEpAlgorithm_t _Algorithm
ctypedef ncclEpOverflowPolicy_t _OverflowPolicy
ctypedef ncclEpLayout_t _Layout
ctypedef ncclEpPassDir_t _PassDir
ctypedef ncclEpZeroCopyMode_t _ZeroCopyMode
ctypedef ncclEpHtFlow_t _HtFlow
ctypedef ncclEpMetaExchangePolicy_t _MetaExchangePolicy
ctypedef ncclEpExpertIdKind_t _ExpertIdKind
ctypedef ncclEpDispQuant_t _DispQuant
ctypedef ncclEpCombQuant_t _CombQuant


###############################################################################
# Functions
###############################################################################

cpdef int get_version() except? -1
cpdef tensor_alloc(intptr_t tensor, unsigned int ndim, ncclDataType_t datatype, intptr_t sizes, intptr_t config)
cpdef tensor_destroy(intptr_t tensor)
cpdef intptr_t create_group(intptr_t comm, intptr_t config) except? 0
cpdef group_destroy(intptr_t ep_group)
cpdef size_t handle_mem_size(intptr_t ep_group, int layout, intptr_t config, int num_topk) except? -1
cpdef intptr_t init_handle(intptr_t ep_group, int layout, intptr_t config, int num_topk, intptr_t handle_mem) except? 0
cpdef intptr_t import_handle(intptr_t ep_group, int layout, intptr_t config, intptr_t state, intptr_t handle_mem) except? 0
cpdef update_handle(intptr_t handle, intptr_t topk_idx, intptr_t layout_info, intptr_t stream)
cpdef intptr_t create_handle(intptr_t ep_group, int layout, intptr_t topk_idx, intptr_t layout_info, intptr_t config, intptr_t stream) except? 0
cpdef handle_destroy(intptr_t handle)
cpdef dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream)
cpdef combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream)
cpdef complete(intptr_t handle, intptr_t config, intptr_t stream)
cpdef mask_query(intptr_t ep_group, intptr_t mask_status, intptr_t stream)
cpdef mask_update(intptr_t ep_group, intptr_t mask, intptr_t stream)
cpdef mask_clean(intptr_t ep_group, intptr_t stream)
cpdef int get_async_error(intptr_t ep_group) except? -1
cpdef error_clear(intptr_t ep_group)
cpdef af_dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream)
cpdef af_combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream)
cpdef af_recv(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream)
cpdef af_return(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream)
cpdef object get_library_path()
