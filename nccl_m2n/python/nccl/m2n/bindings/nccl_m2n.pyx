# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

cimport cython

from libc.stdint cimport intptr_t
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcmp, memcpy

cimport cpython
cimport cpython.buffer

import numpy as _numpy

from nccl.m2n._runtime import NATIVE_CALL_LOCK as _NATIVE_CALL_LOCK


MESH_NDIMS = NCCL_RESHARD_MESH_NDIMS
MAX_TENSOR_DIMS = NCCL_RESHARD_MAX_TENSOR_DIMS
REPLICATE = NCCL_RESHARD_REPLICATE


cdef __from_data(data, dtype_name, expected_dtype, lowpp_type):
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
    cdef cpython.Py_buffer view
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


cdef class Config:
    """Initialize an instance of ``ncclM2nConfig_t`` using C defaults."""

    def __init__(self):
        self._ptr = <ncclM2nConfig_t *>calloc(1, sizeof(ncclM2nConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Config")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclM2nConfig_t)
        self._ptr[0].magic = NCCL_M2N_API_MAGIC
        self._ptr[0].version = NCCL_M2N_API_VERSION
        self._ptr[0].maxCta = NCCL_M2N_CONFIG_UNDEF_INT

    def __dealloc__(self):
        cdef ncclM2nConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Config object at {hex(id(self))}>"

    @property
    def ptr(self):
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Config other_
        if not isinstance(other, Config):
            return False
        other_ = other
        return memcmp(<void *>self._ptr, <void *>other_._ptr, sizeof(ncclM2nConfig_t)) == 0

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclM2nConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    @property
    def maxCta(self):
        return self._ptr[0].maxCta

    @maxCta.setter
    def maxCta(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].maxCta = val

    @staticmethod
    def from_buffer(buffer):
        return __from_buffer(buffer, sizeof(ncclM2nConfig_t), Config)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Config obj = Config.__new__(Config)
        if owner is None:
            obj._ptr = <ncclM2nConfig_t *>malloc(sizeof(ncclM2nConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Config")
            memcpy(<void *>obj._ptr, <void *>ptr, sizeof(ncclM2nConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclM2nConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_mesh_dtype_offsets():
    cdef ncclMesh_t pod
    return _numpy.dtype({
        'names': ['dims', 'startRank'],
        'formats': [
            (_numpy.int32, NCCL_RESHARD_MESH_NDIMS),
            _numpy.int32,
        ],
        'offsets': [
            (<intptr_t>&(pod.dims)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.startRank)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclMesh_t),
    })


mesh_dtype = _get_mesh_dtype_offsets()


cdef class Mesh:
    """Initialize an instance of ``ncclMesh_t``."""

    def __init__(self):
        self._ptr = <ncclMesh_t *>calloc(1, sizeof(ncclMesh_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Mesh")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclMesh_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Mesh object at {hex(id(self))}>"

    @property
    def ptr(self):
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Mesh other_
        if not isinstance(other, Mesh):
            return False
        other_ = other
        return memcmp(<void *>self._ptr, <void *>other_._ptr, sizeof(ncclMesh_t)) == 0

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclMesh_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    @property
    def dims(self):
        return (self._ptr[0].dims[0], self._ptr[0].dims[1])

    @dims.setter
    def dims(self, values):
        if self._readonly:
            raise ValueError("This Mesh instance is read-only")
        if len(values) != NCCL_RESHARD_MESH_NDIMS:
            raise ValueError(f"dims must have length {NCCL_RESHARD_MESH_NDIMS}")
        self._ptr[0].dims[0] = values[0]
        self._ptr[0].dims[1] = values[1]

    @property
    def startRank(self):
        return self._ptr[0].startRank

    @startRank.setter
    def startRank(self, val):
        if self._readonly:
            raise ValueError("This Mesh instance is read-only")
        self._ptr[0].startRank = val

    @staticmethod
    def from_buffer(buffer):
        return __from_buffer(buffer, sizeof(ncclMesh_t), Mesh)

    @staticmethod
    def from_data(data):
        return __from_data(data, "mesh_dtype", mesh_dtype, Mesh)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Mesh obj = Mesh.__new__(Mesh)
        if owner is None:
            obj._ptr = <ncclMesh_t *>malloc(sizeof(ncclMesh_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Mesh")
            memcpy(<void *>obj._ptr, <void *>ptr, sizeof(ncclMesh_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclMesh_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dist_tensor_dtype_offsets():
    cdef ncclDistTensor_t pod
    return _numpy.dtype({
        'names': ['dataPtr', 'localShape', 'ndims', 'dtype', 'mesh', 'placements'],
        'formats': [
            _numpy.intp,
            (_numpy.uintp, NCCL_RESHARD_MAX_TENSOR_DIMS),
            _numpy.int32,
            _numpy.dtype(('V', sizeof(int))),
            _numpy.intp,
            (_numpy.int32, NCCL_RESHARD_MESH_NDIMS),
        ],
        'offsets': [
            (<intptr_t>&(pod.dataPtr)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.localShape)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ndims)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dtype)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.mesh)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.placements)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclDistTensor_t),
    })


dist_tensor_dtype = _get_dist_tensor_dtype_offsets()


cdef class DistTensor:
    """Initialize an instance of ``ncclDistTensor_t``."""

    def __init__(self):
        self._ptr = <ncclDistTensor_t *>calloc(1, sizeof(ncclDistTensor_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DistTensor")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclDistTensor_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.DistTensor object at {hex(id(self))}>"

    @property
    def ptr(self):
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DistTensor other_
        if not isinstance(other, DistTensor):
            return False
        other_ = other
        return memcmp(<void *>self._ptr, <void *>other_._ptr, sizeof(ncclDistTensor_t)) == 0

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDistTensor_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    @property
    def dataPtr(self):
        return <intptr_t>(self._ptr[0].dataPtr)

    @dataPtr.setter
    def dataPtr(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].dataPtr = <void *><intptr_t>val

    @property
    def localShape(self):
        return (
            self._ptr[0].localShape[0],
            self._ptr[0].localShape[1],
            self._ptr[0].localShape[2],
        )

    @localShape.setter
    def localShape(self, values):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        if len(values) != NCCL_RESHARD_MAX_TENSOR_DIMS:
            raise ValueError(f"localShape must have length {NCCL_RESHARD_MAX_TENSOR_DIMS}")
        self._ptr[0].localShape[0] = values[0]
        self._ptr[0].localShape[1] = values[1]
        self._ptr[0].localShape[2] = values[2]

    @property
    def ndims(self):
        return self._ptr[0].ndims

    @ndims.setter
    def ndims(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].ndims = val

    @property
    def dtype(self):
        return self._ptr[0].dtype

    @dtype.setter
    def dtype(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].dtype = val

    @property
    def mesh(self):
        return <intptr_t>(self._ptr[0].mesh)

    @mesh.setter
    def mesh(self, val):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        self._ptr[0].mesh = <const ncclMesh_t *><intptr_t>val

    @property
    def placements(self):
        return (self._ptr[0].placements[0], self._ptr[0].placements[1])

    @placements.setter
    def placements(self, values):
        if self._readonly:
            raise ValueError("This DistTensor instance is read-only")
        if len(values) != NCCL_RESHARD_MESH_NDIMS:
            raise ValueError(f"placements must have length {NCCL_RESHARD_MESH_NDIMS}")
        self._ptr[0].placements[0] = values[0]
        self._ptr[0].placements[1] = values[1]

    @staticmethod
    def from_buffer(buffer):
        return __from_buffer(buffer, sizeof(ncclDistTensor_t), DistTensor)

    @staticmethod
    def from_data(data):
        return __from_data(data, "dist_tensor_dtype", dist_tensor_dtype, DistTensor)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DistTensor obj = DistTensor.__new__(DistTensor)
        if owner is None:
            obj._ptr = <ncclDistTensor_t *>malloc(sizeof(ncclDistTensor_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DistTensor")
            memcpy(<void *>obj._ptr, <void *>ptr, sizeof(ncclDistTensor_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDistTensor_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


###############################################################################
# Error handling
###############################################################################

class NCCLReshardError(Exception):

    def __init__(self, status, detail=None):
        self.status = int(status)
        self.detail = detail
        message = f"NCCL Reshard error code {self.status}"
        if detail:
            message += f": {detail}"
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (self.status, self.detail))


@cython.profile(False)
cpdef inline check_status(int status):
    cdef const char* detail = NULL
    if status != 0:
        detail = ncclM2nGetLastError()
        raise NCCLReshardError(status, (<bytes>detail).decode() if detail != NULL else None)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t init(intptr_t config) except? 0:
    cdef Handle handle
    with _NATIVE_CALL_LOCK:
        with nogil:
            __status__ = ncclM2nInit(&handle, <const ncclM2nConfig_t*>config)
        check_status(__status__)
    return <intptr_t>handle


cpdef finalize(intptr_t handle):
    with _NATIVE_CALL_LOCK:
        with nogil:
            __status__ = ncclM2nFinalize(<Handle>handle)
        check_status(__status__)


cpdef reshard_with_window(intptr_t handle, intptr_t comm, intptr_t window, intptr_t src, intptr_t dst, intptr_t stream):
    with _NATIVE_CALL_LOCK:
        with nogil:
            __status__ = ncclReshardWithWindow(<Handle>handle, <Comm>comm, <Window>window, <const ncclDistTensor_t*>src, <const ncclDistTensor_t*>dst, <Stream>stream)
        check_status(__status__)


cpdef reshard(intptr_t handle, intptr_t comm, intptr_t src, intptr_t dst, intptr_t stream):
    with _NATIVE_CALL_LOCK:
        with nogil:
            __status__ = ncclReshard(<Handle>handle, <Comm>comm, <const ncclDistTensor_t*>src, <const ncclDistTensor_t*>dst, <Stream>stream)
        check_status(__status__)
