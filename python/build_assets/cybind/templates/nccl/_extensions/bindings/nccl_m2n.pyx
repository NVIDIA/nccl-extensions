# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated $version_span. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr, get_buffer_pointer,
                              get_resource_ptr, get_nested_resource_ptr)

from enum import IntEnum as _IntEnum

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

from libc.stdlib cimport calloc, free, malloc
from cython cimport view
cimport cpython.buffer
cimport cpython.memoryview
cimport cpython
from libc.string cimport memcmp, memcpy
import numpy as _numpy


cdef __from_data(data, dtype_name, expected_dtype, lowpp_type):
    # _numpy.recarray is a subclass of _numpy.ndarray, so implicitly handled here.
    if isinstance(data, lowpp_type):
        return data
    if not isinstance(data, _numpy.ndarray):
        raise TypeError("data argument must be a NumPy ndarray")
    if data.size != 1:
        raise ValueError("data array must have a size of 1")
    if data.dtype != expected_dtype:
        raise ValueError(f"data array must be of dtype {dtype_name}")
    return lowpp_type.from_ptr(data.ctypes.data, not data.flags.writeable, data)


cdef __from_buffer(buffer, size, lowpp_type):
    cdef Py_buffer view
    if cpython.PyObject_GetBuffer(buffer, &view, cpython.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, view.readonly, buffer)
    finally:
        cpython.PyBuffer_Release(&view)


cdef __getbuffer(object self, cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
    buffer.buf = <char *>ptr
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = 1
    buffer.len = size
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = readonly
    buffer.shape = &buffer.len
    buffer.strides = &buffer.itemsize
    buffer.suboffsets = NULL


###############################################################################
# POD
###############################################################################

$pod_defs


MESH_NDIMS = NCCL_RESHARD_MESH_NDIMS
MAX_TENSOR_DIMS = NCCL_RESHARD_MAX_TENSOR_DIMS
REPLICATE = NCCL_RESHARD_REPLICATE


###############################################################################
# Error handling
###############################################################################

from nccl.bindings.nccl import NCCLError as _NCCLError
from nccl._extensions._runtime import NATIVE_CALL_LOCK as _NATIVE_CALL_LOCK


class NCCLReshardError(_NCCLError):

    def __init__(self, status, detail=None):
        self.status = int(status)
        self.detail = detail
        message = f"NCCL Reshard error code {self.status}"
        if detail:
            message += f": {detail}"
        Exception.__init__(self, message)

    def __reduce__(self):
        return (type(self), (self.status, self.detail))


@cython.profile(False)
cpdef inline check_status(int status):
    cdef const char* detail = NULL
    cdef bytes detail_bytes
    cdef object detail_text = None
    if status != 0:
        detail = ncclM2nGetLastError()
        if detail != NULL:
            detail_bytes = <bytes>detail
            if detail_bytes:
                detail_text = detail_bytes.decode("utf-8", "replace")
        raise NCCLReshardError(status, detail_text)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t init(intptr_t config) except? 0:
    cdef Handle handle
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nInit(&handle, <const ncclM2nConfig_t*>config)
        check_status(status)
    return <intptr_t>handle


cpdef finalize(intptr_t handle):
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclM2nFinalize(<Handle>handle)
        check_status(status)


cpdef reshard_with_window(intptr_t handle, intptr_t comm, intptr_t window, intptr_t src, intptr_t dst, intptr_t stream):
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclReshardWithWindow(<Handle>handle, <Comm>comm, <Window>window, <const ncclDistTensor_t*>src, <const ncclDistTensor_t*>dst, <Stream>stream)
        check_status(status)


cpdef reshard(intptr_t handle, intptr_t comm, intptr_t src, intptr_t dst, intptr_t stream):
    cdef ncclResult_t status
    with _NATIVE_CALL_LOCK:
        with nogil:
            status = ncclReshard(<Handle>handle, <Comm>comm, <const ncclDistTensor_t*>src, <const ncclDistTensor_t*>dst, <Stream>stream)
        check_status(status)
