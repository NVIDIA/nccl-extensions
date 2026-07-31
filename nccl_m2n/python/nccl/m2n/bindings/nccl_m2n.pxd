# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

from libc.stdint cimport intptr_t

from .cynccl_m2n cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t Comm
ctypedef ncclWindow_t Window
ctypedef ncclM2nHandle_t Handle
ctypedef cudaStream_t Stream


cdef class Config:
    cdef:
        ncclM2nConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly
    cdef intptr_t _get_ptr(self)


cdef class Mesh:
    cdef:
        ncclMesh_t *_ptr
        object _owner
        bint _owned
        bint _readonly
    cdef intptr_t _get_ptr(self)


cdef class DistTensor:
    cdef:
        ncclDistTensor_t *_ptr
        object _owner
        bint _owned
        bint _readonly
    cdef intptr_t _get_ptr(self)


###############################################################################
# Functions
###############################################################################

cpdef intptr_t init(intptr_t config) except? 0
cpdef finalize(intptr_t handle)
cpdef reshard_with_window(intptr_t handle, intptr_t comm, intptr_t window, intptr_t src, intptr_t dst, intptr_t stream)
cpdef reshard(intptr_t handle, intptr_t comm, intptr_t src, intptr_t dst, intptr_t stream)
