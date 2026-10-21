# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.2.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

cimport cpython as _cyb_cpython
cimport cpython.buffer as _cyb_cpython_buffer
from cython cimport view as _cyb_view
from libc.stdint cimport intptr_t
from libc.stdlib cimport (
    calloc as _cyb_calloc,
    free as _cyb_free,
    malloc as _cyb_malloc,
)
from libc.string cimport (
    memcmp as _cyb_memcmp,
    memcpy as _cyb_memcpy,
)

from enum import IntEnum as _cyb_IntEnum

import numpy as _numpy

cdef _cyb___getbuffer(object self, _cyb_cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
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

cdef _cyb_from_buffer(buffer, size, lowpp_type):
    cdef _cyb_cpython.Py_buffer view
    if _cyb_cpython.PyObject_GetBuffer(buffer, &view, _cyb_cpython_buffer.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, view.readonly, buffer)
    finally:
        _cyb_cpython.PyBuffer_Release(&view)

cdef _cyb_from_data(data, dtype_name, expected_dtype, lowpp_type):
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


# <<<< END OF PREAMBLE CONTENT >>>>

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr,
                              get_resource_ptr, get_nested_resource_ptr)

_version_span = "with version 0.2.0"
__version__ = _version_span.split()[-1]

# NCCL_VERSION(X,Y,Z) = X*10000 + Y*100 + Z (NCCL >= 2.9).
_version_parts = __version__.split(".")
__version_code__ = (
    int(_version_parts[0]) * 10000 + int(_version_parts[1]) * 100 + int(_version_parts[2])
)


###############################################################################
# POD
###############################################################################

cdef _get_tensor_dtype_offsets():
    cdef ncclEpTensor_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'ndim_', 'datatype', 'data_', 'win_hdl', 'win_offset', 'sizes'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.dtype(('V', sizeof(ncclDataType_t))), _numpy.intp, _numpy.intp, _numpy.uint64, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ndim)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.datatype)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.data)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.win_hdl)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.win_offset)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.sizes)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpTensor_t),
    })

tensor_dtype = _get_tensor_dtype_offsets()

cdef class Tensor:
    """Initialize an instance of `ncclEpTensor_t` using configured defaults.


    .. seealso:: `ncclEpTensor_t`
    """
    cdef:
        ncclEpTensor_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpTensor_t *>_cyb_calloc(1, sizeof(ncclEpTensor_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Tensor")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpTensor_t)
        self._ptr[0].magic = 0xCAFECAFE

    def __dealloc__(self):
        cdef ncclEpTensor_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.Tensor object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Tensor other_
        if not isinstance(other, Tensor):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpTensor_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpTensor_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpTensor_t *>_cyb_malloc(sizeof(ncclEpTensor_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Tensor")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpTensor_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].magic = val

    @property
    def ndim_(self):
        """int: """
        return self._ptr[0].ndim

    @ndim_.setter
    def ndim_(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].ndim = val

    @property
    def datatype(self):
        """: """
        return self._ptr[0].datatype

    @datatype.setter
    def datatype(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].datatype = val

    @property
    def data_(self):
        """int: """
        return <intptr_t>(self._ptr[0].data)

    @data_.setter
    def data_(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].data = <void *><intptr_t>val

    @property
    def win_hdl(self):
        """int: """
        return <intptr_t>(self._ptr[0].win_hdl)

    @win_hdl.setter
    def win_hdl(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].win_hdl = <ncclWindow_t><intptr_t>val

    @property
    def win_offset(self):
        """int: """
        return self._ptr[0].win_offset

    @win_offset.setter
    def win_offset(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].win_offset = val

    @property
    def sizes(self):
        """int: """
        return <intptr_t>(self._ptr[0].sizes)

    @sizes.setter
    def sizes(self, val):
        if self._readonly:
            raise ValueError("This Tensor instance is read-only")
        self._ptr[0].sizes = <size_t*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an Tensor instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpTensor_t), Tensor)

    @staticmethod
    def from_data(data):
        """Create an Tensor instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `tensor_dtype` holding the data.
        """
        return _cyb_from_data(data, "tensor_dtype", tensor_dtype, Tensor)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Tensor instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Tensor obj = Tensor.__new__(Tensor)
        if owner is None:
            obj._ptr = <ncclEpTensor_t *>_cyb_malloc(sizeof(ncclEpTensor_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Tensor")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpTensor_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpTensor_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_tensor_alloc_config_dtype_offsets():
    cdef ncclEpTensorAllocConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic'],
        'formats': [_numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpTensorAllocConfig_t),
    })

tensor_alloc_config_dtype = _get_tensor_alloc_config_dtype_offsets()

cdef class TensorAllocConfig:
    """Initialize an instance of `ncclEpTensorAllocConfig_t` using configured defaults.


    .. seealso:: `ncclEpTensorAllocConfig_t`
    """
    cdef:
        ncclEpTensorAllocConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpTensorAllocConfig_t *>_cyb_calloc(1, sizeof(ncclEpTensorAllocConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating TensorAllocConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpTensorAllocConfig_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpTensorAllocConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.TensorAllocConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef TensorAllocConfig other_
        if not isinstance(other, TensorAllocConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpTensorAllocConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpTensorAllocConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpTensorAllocConfig_t *>_cyb_malloc(sizeof(ncclEpTensorAllocConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating TensorAllocConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpTensorAllocConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This TensorAllocConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This TensorAllocConfig instance is read-only")
        self._ptr[0].magic = val

    @staticmethod
    def from_buffer(buffer):
        """Create an TensorAllocConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpTensorAllocConfig_t), TensorAllocConfig)

    @staticmethod
    def from_data(data):
        """Create an TensorAllocConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `tensor_alloc_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "tensor_alloc_config_dtype", tensor_alloc_config_dtype, TensorAllocConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an TensorAllocConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef TensorAllocConfig obj = TensorAllocConfig.__new__(TensorAllocConfig)
        if owner is None:
            obj._ptr = <ncclEpTensorAllocConfig_t *>_cyb_malloc(sizeof(ncclEpTensorAllocConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating TensorAllocConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpTensorAllocConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpTensorAllocConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_alloc_config_dtype_offsets():
    cdef ncclEpAllocConfig_t pod
    return _numpy.dtype({
        'names': ['alloc_fn', 'free_fn', 'context'],
        'formats': [_numpy.dtype(('V', sizeof(ncclEpAllocFn_t))), _numpy.dtype(('V', sizeof(ncclEpFreeFn_t))), _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.alloc_fn)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.free_fn)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.context)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpAllocConfig_t),
    })

alloc_config_dtype = _get_alloc_config_dtype_offsets()

cdef class AllocConfig:
    """Empty-initialize an instance of `ncclEpAllocConfig_t`.


    .. seealso:: `ncclEpAllocConfig_t`
    """
    cdef:
        ncclEpAllocConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpAllocConfig_t *>_cyb_calloc(1, sizeof(ncclEpAllocConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating AllocConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpAllocConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.AllocConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef AllocConfig other_
        if not isinstance(other, AllocConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpAllocConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpAllocConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpAllocConfig_t *>_cyb_malloc(sizeof(ncclEpAllocConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating AllocConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpAllocConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def alloc_fn(self):
        return <intptr_t>(self._ptr[0].alloc_fn)

    @alloc_fn.setter
    def alloc_fn(self, val):
        if self._readonly:
            raise ValueError("This AllocConfig instance is read-only")
        self._ptr[0].alloc_fn = <ncclEpAllocFn_t><intptr_t>val

    @property
    def free_fn(self):
        return <intptr_t>(self._ptr[0].free_fn)

    @free_fn.setter
    def free_fn(self, val):
        if self._readonly:
            raise ValueError("This AllocConfig instance is read-only")
        self._ptr[0].free_fn = <ncclEpFreeFn_t><intptr_t>val

    @property
    def context(self):
        """int: """
        return <intptr_t>(self._ptr[0].context)

    @context.setter
    def context(self, val):
        if self._readonly:
            raise ValueError("This AllocConfig instance is read-only")
        self._ptr[0].context = <void *><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an AllocConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpAllocConfig_t), AllocConfig)

    @staticmethod
    def from_data(data):
        """Create an AllocConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `alloc_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "alloc_config_dtype", alloc_config_dtype, AllocConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an AllocConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef AllocConfig obj = AllocConfig.__new__(AllocConfig)
        if owner is None:
            obj._ptr = <ncclEpAllocConfig_t *>_cyb_malloc(sizeof(ncclEpAllocConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating AllocConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpAllocConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpAllocConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_handle_config_dtype_offsets():
    cdef ncclEpHandleConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'dispatch_output_per_expert_alignment'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dispatch_output_per_expert_alignment)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpHandleConfig_t),
    })

handle_config_dtype = _get_handle_config_dtype_offsets()

cdef class HandleConfig:
    """Initialize an instance of `ncclEpHandleConfig_t` using configured defaults.


    .. seealso:: `ncclEpHandleConfig_t`
    """
    cdef:
        ncclEpHandleConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpHandleConfig_t *>_cyb_calloc(1, sizeof(ncclEpHandleConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating HandleConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpHandleConfig_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpHandleConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.HandleConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef HandleConfig other_
        if not isinstance(other, HandleConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpHandleConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpHandleConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpHandleConfig_t *>_cyb_malloc(sizeof(ncclEpHandleConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating HandleConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpHandleConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This HandleConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This HandleConfig instance is read-only")
        self._ptr[0].magic = val

    @property
    def dispatch_output_per_expert_alignment(self):
        """int: """
        return self._ptr[0].dispatch_output_per_expert_alignment

    @dispatch_output_per_expert_alignment.setter
    def dispatch_output_per_expert_alignment(self, val):
        if self._readonly:
            raise ValueError("This HandleConfig instance is read-only")
        self._ptr[0].dispatch_output_per_expert_alignment = val

    @staticmethod
    def from_buffer(buffer):
        """Create an HandleConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpHandleConfig_t), HandleConfig)

    @staticmethod
    def from_data(data):
        """Create an HandleConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `handle_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "handle_config_dtype", handle_config_dtype, HandleConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an HandleConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef HandleConfig obj = HandleConfig.__new__(HandleConfig)
        if owner is None:
            obj._ptr = <ncclEpHandleConfig_t *>_cyb_malloc(sizeof(ncclEpHandleConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating HandleConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpHandleConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpHandleConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_handle_state_dtype_offsets():
    cdef ncclEpHandleState_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'num_tokens', 'num_topk', 'topk_idx_datatype'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.dtype(('V', sizeof(ncclDataType_t)))],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_topk)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_idx_datatype)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpHandleState_t),
    })

handle_state_dtype = _get_handle_state_dtype_offsets()

cdef class HandleState:
    """Initialize an instance of `ncclEpHandleState_t` using configured defaults.


    .. seealso:: `ncclEpHandleState_t`
    """
    cdef:
        ncclEpHandleState_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpHandleState_t *>_cyb_calloc(1, sizeof(ncclEpHandleState_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating HandleState")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpHandleState_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpHandleState_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.HandleState object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef HandleState other_
        if not isinstance(other, HandleState):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpHandleState_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpHandleState_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpHandleState_t *>_cyb_malloc(sizeof(ncclEpHandleState_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating HandleState")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpHandleState_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This HandleState instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This HandleState instance is read-only")
        self._ptr[0].magic = val

    @property
    def num_tokens(self):
        """int: """
        return self._ptr[0].num_tokens

    @num_tokens.setter
    def num_tokens(self, val):
        if self._readonly:
            raise ValueError("This HandleState instance is read-only")
        self._ptr[0].num_tokens = val

    @property
    def num_topk(self):
        """int: """
        return self._ptr[0].num_topk

    @num_topk.setter
    def num_topk(self, val):
        if self._readonly:
            raise ValueError("This HandleState instance is read-only")
        self._ptr[0].num_topk = val

    @property
    def topk_idx_datatype(self):
        """: """
        return self._ptr[0].topk_idx_datatype

    @topk_idx_datatype.setter
    def topk_idx_datatype(self, val):
        if self._readonly:
            raise ValueError("This HandleState instance is read-only")
        self._ptr[0].topk_idx_datatype = val

    @staticmethod
    def from_buffer(buffer):
        """Create an HandleState instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpHandleState_t), HandleState)

    @staticmethod
    def from_data(data):
        """Create an HandleState instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `handle_state_dtype` holding the data.
        """
        return _cyb_from_data(data, "handle_state_dtype", handle_state_dtype, HandleState)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an HandleState instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef HandleState obj = HandleState.__new__(HandleState)
        if owner is None:
            obj._ptr = <ncclEpHandleState_t *>_cyb_malloc(sizeof(ncclEpHandleState_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating HandleState")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpHandleState_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpHandleState_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dispatch_config_dtype_offsets():
    cdef ncclEpDispatchConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'send_only', 'round_scales', 'pass_direction', 'quant_recipe'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.send_only)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.round_scales)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.pass_direction)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.quant_recipe)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpDispatchConfig_t),
    })

dispatch_config_dtype = _get_dispatch_config_dtype_offsets()

cdef class DispatchConfig:
    """Initialize an instance of `ncclEpDispatchConfig_t` using configured defaults.


    .. seealso:: `ncclEpDispatchConfig_t`
    """
    cdef:
        ncclEpDispatchConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpDispatchConfig_t *>_cyb_calloc(1, sizeof(ncclEpDispatchConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DispatchConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpDispatchConfig_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpDispatchConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.DispatchConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DispatchConfig other_
        if not isinstance(other, DispatchConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpDispatchConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpDispatchConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpDispatchConfig_t *>_cyb_malloc(sizeof(ncclEpDispatchConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DispatchConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpDispatchConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This DispatchConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This DispatchConfig instance is read-only")
        self._ptr[0].magic = val

    @property
    def send_only(self):
        """int: """
        return self._ptr[0].send_only

    @send_only.setter
    def send_only(self, val):
        if self._readonly:
            raise ValueError("This DispatchConfig instance is read-only")
        self._ptr[0].send_only = val

    @property
    def round_scales(self):
        """int: """
        return self._ptr[0].round_scales

    @round_scales.setter
    def round_scales(self, val):
        if self._readonly:
            raise ValueError("This DispatchConfig instance is read-only")
        self._ptr[0].round_scales = val

    @property
    def pass_direction(self):
        """int: """
        return <int>(self._ptr[0].pass_direction)

    @pass_direction.setter
    def pass_direction(self, val):
        if self._readonly:
            raise ValueError("This DispatchConfig instance is read-only")
        self._ptr[0].pass_direction = <ncclEpPassDir_t><int>val

    @property
    def quant_recipe(self):
        """int: """
        return <int>(self._ptr[0].quant_recipe)

    @quant_recipe.setter
    def quant_recipe(self, val):
        if self._readonly:
            raise ValueError("This DispatchConfig instance is read-only")
        self._ptr[0].quant_recipe = <ncclEpDispQuant_t><int>val

    @staticmethod
    def from_buffer(buffer):
        """Create an DispatchConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpDispatchConfig_t), DispatchConfig)

    @staticmethod
    def from_data(data):
        """Create an DispatchConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dispatch_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "dispatch_config_dtype", dispatch_config_dtype, DispatchConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DispatchConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DispatchConfig obj = DispatchConfig.__new__(DispatchConfig)
        if owner is None:
            obj._ptr = <ncclEpDispatchConfig_t *>_cyb_malloc(sizeof(ncclEpDispatchConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DispatchConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpDispatchConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpDispatchConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_combine_config_dtype_offsets():
    cdef ncclEpCombineConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'send_only', 'pass_direction', 'quant_recipe'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.send_only)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.pass_direction)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.quant_recipe)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCombineConfig_t),
    })

combine_config_dtype = _get_combine_config_dtype_offsets()

cdef class CombineConfig:
    """Initialize an instance of `ncclEpCombineConfig_t` using configured defaults.


    .. seealso:: `ncclEpCombineConfig_t`
    """
    cdef:
        ncclEpCombineConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCombineConfig_t *>_cyb_calloc(1, sizeof(ncclEpCombineConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CombineConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpCombineConfig_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpCombineConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.CombineConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CombineConfig other_
        if not isinstance(other, CombineConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCombineConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCombineConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpCombineConfig_t *>_cyb_malloc(sizeof(ncclEpCombineConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CombineConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCombineConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This CombineConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This CombineConfig instance is read-only")
        self._ptr[0].magic = val

    @property
    def send_only(self):
        """int: """
        return self._ptr[0].send_only

    @send_only.setter
    def send_only(self, val):
        if self._readonly:
            raise ValueError("This CombineConfig instance is read-only")
        self._ptr[0].send_only = val

    @property
    def pass_direction(self):
        """int: """
        return <int>(self._ptr[0].pass_direction)

    @pass_direction.setter
    def pass_direction(self, val):
        if self._readonly:
            raise ValueError("This CombineConfig instance is read-only")
        self._ptr[0].pass_direction = <ncclEpPassDir_t><int>val

    @property
    def quant_recipe(self):
        """int: """
        return <int>(self._ptr[0].quant_recipe)

    @quant_recipe.setter
    def quant_recipe(self, val):
        if self._readonly:
            raise ValueError("This CombineConfig instance is read-only")
        self._ptr[0].quant_recipe = <ncclEpCombQuant_t><int>val

    @staticmethod
    def from_buffer(buffer):
        """Create an CombineConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpCombineConfig_t), CombineConfig)

    @staticmethod
    def from_data(data):
        """Create an CombineConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `combine_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "combine_config_dtype", combine_config_dtype, CombineConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CombineConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CombineConfig obj = CombineConfig.__new__(CombineConfig)
        if owner is None:
            obj._ptr = <ncclEpCombineConfig_t *>_cyb_malloc(sizeof(ncclEpCombineConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CombineConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCombineConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCombineConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_complete_config_dtype_offsets():
    cdef ncclEpCompleteConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic'],
        'formats': [_numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCompleteConfig_t),
    })

complete_config_dtype = _get_complete_config_dtype_offsets()

cdef class CompleteConfig:
    """Initialize an instance of `ncclEpCompleteConfig_t` using configured defaults.


    .. seealso:: `ncclEpCompleteConfig_t`
    """
    cdef:
        ncclEpCompleteConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCompleteConfig_t *>_cyb_calloc(1, sizeof(ncclEpCompleteConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CompleteConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpCompleteConfig_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpCompleteConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.CompleteConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CompleteConfig other_
        if not isinstance(other, CompleteConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCompleteConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCompleteConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpCompleteConfig_t *>_cyb_malloc(sizeof(ncclEpCompleteConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CompleteConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCompleteConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This CompleteConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This CompleteConfig instance is read-only")
        self._ptr[0].magic = val

    @staticmethod
    def from_buffer(buffer):
        """Create an CompleteConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpCompleteConfig_t), CompleteConfig)

    @staticmethod
    def from_data(data):
        """Create an CompleteConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `complete_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "complete_config_dtype", complete_config_dtype, CompleteConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CompleteConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CompleteConfig obj = CompleteConfig.__new__(CompleteConfig)
        if owner is None:
            obj._ptr = <ncclEpCompleteConfig_t *>_cyb_malloc(sizeof(ncclEpCompleteConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CompleteConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCompleteConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCompleteConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_layout_info_dtype_offsets():
    cdef ncclEpLayoutInfo_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'expert_counters', 'src_rank_counters', 'expert_offsets', 'recv_total_counter', 'recv_topk_idx_kind', 'padding_v2'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp, _numpy.intp, _numpy.int32, (_numpy.uint8, 4)],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.expert_counters)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.src_rank_counters)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.expert_offsets)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.recv_total_counter)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.recv_topk_idx_kind)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.padding_v2)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpLayoutInfo_t),
    })

layout_info_dtype = _get_layout_info_dtype_offsets()

cdef class LayoutInfo:
    """Initialize an instance of `ncclEpLayoutInfo_t` using configured defaults.


    .. seealso:: `ncclEpLayoutInfo_t`
    """
    cdef:
        ncclEpLayoutInfo_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpLayoutInfo_t *>_cyb_calloc(1, sizeof(ncclEpLayoutInfo_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating LayoutInfo")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpLayoutInfo_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpLayoutInfo_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.LayoutInfo object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef LayoutInfo other_
        if not isinstance(other, LayoutInfo):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpLayoutInfo_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpLayoutInfo_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpLayoutInfo_t *>_cyb_malloc(sizeof(ncclEpLayoutInfo_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating LayoutInfo")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpLayoutInfo_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].magic = val

    @property
    def expert_counters(self):
        """int: """
        return <intptr_t>(self._ptr[0].expert_counters)

    @expert_counters.setter
    def expert_counters(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].expert_counters = <ncclEpTensor_t*><intptr_t>val

    @property
    def src_rank_counters(self):
        """int: """
        return <intptr_t>(self._ptr[0].src_rank_counters)

    @src_rank_counters.setter
    def src_rank_counters(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].src_rank_counters = <ncclEpTensor_t*><intptr_t>val

    @property
    def expert_offsets(self):
        """int: """
        return <intptr_t>(self._ptr[0].expert_offsets)

    @expert_offsets.setter
    def expert_offsets(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].expert_offsets = <ncclEpTensor_t*><intptr_t>val

    @property
    def recv_total_counter(self):
        """int: """
        return <intptr_t>(self._ptr[0].recv_total_counter)

    @recv_total_counter.setter
    def recv_total_counter(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].recv_total_counter = <ncclEpTensor_t*><intptr_t>val

    @property
    def recv_topk_idx_kind(self):
        """int: """
        return <int>(self._ptr[0].recv_topk_idx_kind)

    @recv_topk_idx_kind.setter
    def recv_topk_idx_kind(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        self._ptr[0].recv_topk_idx_kind = <ncclEpExpertIdKind_t><int>val

    @property
    def padding_v2(self):
        """~_numpy.uint8: (array of length 4)."""
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(unsigned char), format="B", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].padding_v2))
        return _numpy.asarray(arr)

    @padding_v2.setter
    def padding_v2(self, val):
        if self._readonly:
            raise ValueError("This LayoutInfo instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field padding_v2, got {len(val)}")
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(unsigned char), format="B", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint8)
        _cyb_memcpy(<void *>(&(self._ptr[0].padding_v2)), <void *>(arr.data), sizeof(unsigned char) * len(val))

    @staticmethod
    def from_buffer(buffer):
        """Create an LayoutInfo instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpLayoutInfo_t), LayoutInfo)

    @staticmethod
    def from_data(data):
        """Create an LayoutInfo instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `layout_info_dtype` holding the data.
        """
        return _cyb_from_data(data, "layout_info_dtype", layout_info_dtype, LayoutInfo)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an LayoutInfo instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef LayoutInfo obj = LayoutInfo.__new__(LayoutInfo)
        if owner is None:
            obj._ptr = <ncclEpLayoutInfo_t *>_cyb_malloc(sizeof(ncclEpLayoutInfo_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating LayoutInfo")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpLayoutInfo_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpLayoutInfo_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dispatch_inputs_dtype_offsets():
    cdef ncclEpDispatchInputs_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'tokens', 'topk_weights', 'scales'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.scales)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpDispatchInputs_t),
    })

dispatch_inputs_dtype = _get_dispatch_inputs_dtype_offsets()

cdef class DispatchInputs:
    """Initialize an instance of `ncclEpDispatchInputs_t` using configured defaults.


    .. seealso:: `ncclEpDispatchInputs_t`
    """
    cdef:
        ncclEpDispatchInputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpDispatchInputs_t *>_cyb_calloc(1, sizeof(ncclEpDispatchInputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DispatchInputs")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpDispatchInputs_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpDispatchInputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.DispatchInputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DispatchInputs other_
        if not isinstance(other, DispatchInputs):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpDispatchInputs_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpDispatchInputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpDispatchInputs_t *>_cyb_malloc(sizeof(ncclEpDispatchInputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DispatchInputs")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpDispatchInputs_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This DispatchInputs instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This DispatchInputs instance is read-only")
        self._ptr[0].magic = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This DispatchInputs instance is read-only")
        self._ptr[0].tokens = <ncclEpTensor_t*><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This DispatchInputs instance is read-only")
        self._ptr[0].topk_weights = <ncclEpTensor_t*><intptr_t>val

    @property
    def scales(self):
        """int: """
        return <intptr_t>(self._ptr[0].scales)

    @scales.setter
    def scales(self, val):
        if self._readonly:
            raise ValueError("This DispatchInputs instance is read-only")
        self._ptr[0].scales = <ncclEpTensor_t*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an DispatchInputs instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpDispatchInputs_t), DispatchInputs)

    @staticmethod
    def from_data(data):
        """Create an DispatchInputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dispatch_inputs_dtype` holding the data.
        """
        return _cyb_from_data(data, "dispatch_inputs_dtype", dispatch_inputs_dtype, DispatchInputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DispatchInputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DispatchInputs obj = DispatchInputs.__new__(DispatchInputs)
        if owner is None:
            obj._ptr = <ncclEpDispatchInputs_t *>_cyb_malloc(sizeof(ncclEpDispatchInputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DispatchInputs")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpDispatchInputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpDispatchInputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dispatch_outputs_dtype_offsets():
    cdef ncclEpDispatchOutputs_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'tokens', 'topk_weights', 'scales', 'topk_idx'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.scales)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_idx)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpDispatchOutputs_t),
    })

dispatch_outputs_dtype = _get_dispatch_outputs_dtype_offsets()

cdef class DispatchOutputs:
    """Initialize an instance of `ncclEpDispatchOutputs_t` using configured defaults.


    .. seealso:: `ncclEpDispatchOutputs_t`
    """
    cdef:
        ncclEpDispatchOutputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpDispatchOutputs_t *>_cyb_calloc(1, sizeof(ncclEpDispatchOutputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DispatchOutputs")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpDispatchOutputs_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpDispatchOutputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.DispatchOutputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DispatchOutputs other_
        if not isinstance(other, DispatchOutputs):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpDispatchOutputs_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpDispatchOutputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpDispatchOutputs_t *>_cyb_malloc(sizeof(ncclEpDispatchOutputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DispatchOutputs")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpDispatchOutputs_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This DispatchOutputs instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This DispatchOutputs instance is read-only")
        self._ptr[0].magic = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This DispatchOutputs instance is read-only")
        self._ptr[0].tokens = <ncclEpTensor_t*><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This DispatchOutputs instance is read-only")
        self._ptr[0].topk_weights = <ncclEpTensor_t*><intptr_t>val

    @property
    def scales(self):
        """int: """
        return <intptr_t>(self._ptr[0].scales)

    @scales.setter
    def scales(self, val):
        if self._readonly:
            raise ValueError("This DispatchOutputs instance is read-only")
        self._ptr[0].scales = <ncclEpTensor_t*><intptr_t>val

    @property
    def topk_idx(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_idx)

    @topk_idx.setter
    def topk_idx(self, val):
        if self._readonly:
            raise ValueError("This DispatchOutputs instance is read-only")
        self._ptr[0].topk_idx = <ncclEpTensor_t*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an DispatchOutputs instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpDispatchOutputs_t), DispatchOutputs)

    @staticmethod
    def from_data(data):
        """Create an DispatchOutputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dispatch_outputs_dtype` holding the data.
        """
        return _cyb_from_data(data, "dispatch_outputs_dtype", dispatch_outputs_dtype, DispatchOutputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DispatchOutputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DispatchOutputs obj = DispatchOutputs.__new__(DispatchOutputs)
        if owner is None:
            obj._ptr = <ncclEpDispatchOutputs_t *>_cyb_malloc(sizeof(ncclEpDispatchOutputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DispatchOutputs")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpDispatchOutputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpDispatchOutputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_combine_inputs_dtype_offsets():
    cdef ncclEpCombineInputs_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'tokens', 'topk_weights', 'scales'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.scales)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCombineInputs_t),
    })

combine_inputs_dtype = _get_combine_inputs_dtype_offsets()

cdef class CombineInputs:
    """Initialize an instance of `ncclEpCombineInputs_t` using configured defaults.


    .. seealso:: `ncclEpCombineInputs_t`
    """
    cdef:
        ncclEpCombineInputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCombineInputs_t *>_cyb_calloc(1, sizeof(ncclEpCombineInputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CombineInputs")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpCombineInputs_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpCombineInputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.CombineInputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CombineInputs other_
        if not isinstance(other, CombineInputs):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCombineInputs_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCombineInputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpCombineInputs_t *>_cyb_malloc(sizeof(ncclEpCombineInputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CombineInputs")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCombineInputs_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This CombineInputs instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This CombineInputs instance is read-only")
        self._ptr[0].magic = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This CombineInputs instance is read-only")
        self._ptr[0].tokens = <ncclEpTensor_t*><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This CombineInputs instance is read-only")
        self._ptr[0].topk_weights = <ncclEpTensor_t*><intptr_t>val

    @property
    def scales(self):
        """int: """
        return <intptr_t>(self._ptr[0].scales)

    @scales.setter
    def scales(self, val):
        if self._readonly:
            raise ValueError("This CombineInputs instance is read-only")
        self._ptr[0].scales = <ncclEpTensor_t*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an CombineInputs instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpCombineInputs_t), CombineInputs)

    @staticmethod
    def from_data(data):
        """Create an CombineInputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `combine_inputs_dtype` holding the data.
        """
        return _cyb_from_data(data, "combine_inputs_dtype", combine_inputs_dtype, CombineInputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CombineInputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CombineInputs obj = CombineInputs.__new__(CombineInputs)
        if owner is None:
            obj._ptr = <ncclEpCombineInputs_t *>_cyb_malloc(sizeof(ncclEpCombineInputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CombineInputs")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCombineInputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCombineInputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_combine_outputs_dtype_offsets():
    cdef ncclEpCombineOutputs_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'tokens', 'topk_weights'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCombineOutputs_t),
    })

combine_outputs_dtype = _get_combine_outputs_dtype_offsets()

cdef class CombineOutputs:
    """Initialize an instance of `ncclEpCombineOutputs_t` using configured defaults.


    .. seealso:: `ncclEpCombineOutputs_t`
    """
    cdef:
        ncclEpCombineOutputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCombineOutputs_t *>_cyb_calloc(1, sizeof(ncclEpCombineOutputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CombineOutputs")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpCombineOutputs_t)
        self._ptr[0].magic = 0xC00FFFEE

    def __dealloc__(self):
        cdef ncclEpCombineOutputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.CombineOutputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CombineOutputs other_
        if not isinstance(other, CombineOutputs):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCombineOutputs_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCombineOutputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpCombineOutputs_t *>_cyb_malloc(sizeof(ncclEpCombineOutputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CombineOutputs")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCombineOutputs_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This CombineOutputs instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This CombineOutputs instance is read-only")
        self._ptr[0].magic = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This CombineOutputs instance is read-only")
        self._ptr[0].tokens = <ncclEpTensor_t*><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This CombineOutputs instance is read-only")
        self._ptr[0].topk_weights = <ncclEpTensor_t*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an CombineOutputs instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpCombineOutputs_t), CombineOutputs)

    @staticmethod
    def from_data(data):
        """Create an CombineOutputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `combine_outputs_dtype` holding the data.
        """
        return _cyb_from_data(data, "combine_outputs_dtype", combine_outputs_dtype, CombineOutputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CombineOutputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CombineOutputs obj = CombineOutputs.__new__(CombineOutputs)
        if owner is None:
            obj._ptr = <ncclEpCombineOutputs_t *>_cyb_malloc(sizeof(ncclEpCombineOutputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CombineOutputs")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCombineOutputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCombineOutputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_group_config_dtype_offsets():
    cdef ncclEpGroupConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'algorithm', 'num_experts', 'max_dispatch_tokens_per_rank', 'max_recv_tokens_per_rank', 'max_token_bytes', 'rdma_buffer_size', 'num_qp_per_rank', 'num_channels', 'max_num_sms', 'alloc', 'enable_mask', 'timeout_ns', 'zero_copy', 'overflow_policy', 'num_topk', 'padding_v2', 'dispatch_num_sms', 'combine_num_sms', 'ht_flow', 'meta_exchange_policy', 'num_attn_ranks', 'padding_v4'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.dtype(('V', sizeof(unsigned long int))), _numpy.uint32, _numpy.uint32, _numpy.uint32, alloc_config_dtype, _numpy.uint32, _numpy.uint64, _numpy.int32, _numpy.int32, _numpy.uint32, (_numpy.uint8, 4), _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.uint32, (_numpy.uint8, 4)],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.algorithm)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_experts)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_dispatch_tokens_per_rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_recv_tokens_per_rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_token_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.rdma_buffer_size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_qp_per_rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_channels)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_num_sms)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.alloc)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.enable_mask)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.timeout_ns)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.zero_copy)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.overflow_policy)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_topk)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.padding_v2)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dispatch_num_sms)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.combine_num_sms)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ht_flow)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.meta_exchange_policy)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_attn_ranks)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.padding_v4)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpGroupConfig_t),
    })

group_config_dtype = _get_group_config_dtype_offsets()

cdef class GroupConfig:
    """Initialize an instance of `ncclEpGroupConfig_t` using configured defaults.


    .. seealso:: `ncclEpGroupConfig_t`
    """
    cdef:
        ncclEpGroupConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpGroupConfig_t *>_cyb_calloc(1, sizeof(ncclEpGroupConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating GroupConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclEpGroupConfig_t)
        self._ptr[0].magic = 0xC00FFFEE
        # = NCCL_EP_API_VERSION; bump with the header.
        self._ptr[0].version = 4

    def __dealloc__(self):
        cdef ncclEpGroupConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.GroupConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef GroupConfig other_
        if not isinstance(other, GroupConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpGroupConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpGroupConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclEpGroupConfig_t *>_cyb_malloc(sizeof(ncclEpGroupConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating GroupConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpGroupConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def alloc(self):
        """AllocConfig: """
        return AllocConfig.from_ptr(
            <intptr_t>&(self._ptr[0].alloc),
            readonly=self._readonly,
            owner=self,
        )

    @alloc.setter
    def alloc(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        cdef AllocConfig val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].alloc), <void *>(val_._get_ptr()), sizeof(ncclEpAllocConfig_t) * 1)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].version = val

    @property
    def algorithm(self):
        """int: """
        return <int>(self._ptr[0].algorithm)

    @algorithm.setter
    def algorithm(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].algorithm = <ncclEpAlgorithm_t><int>val

    @property
    def num_experts(self):
        """int: """
        return self._ptr[0].num_experts

    @num_experts.setter
    def num_experts(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].num_experts = val

    @property
    def max_dispatch_tokens_per_rank(self):
        """int: """
        return self._ptr[0].max_dispatch_tokens_per_rank

    @max_dispatch_tokens_per_rank.setter
    def max_dispatch_tokens_per_rank(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].max_dispatch_tokens_per_rank = val

    @property
    def max_recv_tokens_per_rank(self):
        """int: """
        return self._ptr[0].max_recv_tokens_per_rank

    @max_recv_tokens_per_rank.setter
    def max_recv_tokens_per_rank(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].max_recv_tokens_per_rank = val

    @property
    def max_token_bytes(self):
        """int: """
        return self._ptr[0].max_token_bytes

    @max_token_bytes.setter
    def max_token_bytes(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].max_token_bytes = val

    @property
    def rdma_buffer_size(self):
        """: """
        return self._ptr[0].rdma_buffer_size

    @rdma_buffer_size.setter
    def rdma_buffer_size(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].rdma_buffer_size = val

    @property
    def num_qp_per_rank(self):
        """int: """
        return self._ptr[0].num_qp_per_rank

    @num_qp_per_rank.setter
    def num_qp_per_rank(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].num_qp_per_rank = val

    @property
    def num_channels(self):
        """int: """
        return self._ptr[0].num_channels

    @num_channels.setter
    def num_channels(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].num_channels = val

    @property
    def max_num_sms(self):
        """int: """
        return self._ptr[0].max_num_sms

    @max_num_sms.setter
    def max_num_sms(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].max_num_sms = val

    @property
    def enable_mask(self):
        """int: """
        return self._ptr[0].enable_mask

    @enable_mask.setter
    def enable_mask(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].enable_mask = val

    @property
    def timeout_ns(self):
        """int: """
        return self._ptr[0].timeout_ns

    @timeout_ns.setter
    def timeout_ns(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].timeout_ns = val

    @property
    def zero_copy(self):
        """int: """
        return <int>(self._ptr[0].zero_copy)

    @zero_copy.setter
    def zero_copy(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].zero_copy = <ncclEpZeroCopyMode_t><int>val

    @property
    def overflow_policy(self):
        """int: """
        return <int>(self._ptr[0].overflow_policy)

    @overflow_policy.setter
    def overflow_policy(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].overflow_policy = <ncclEpOverflowPolicy_t><int>val

    @property
    def num_topk(self):
        """int: """
        return self._ptr[0].num_topk

    @num_topk.setter
    def num_topk(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].num_topk = val

    @property
    def padding_v2(self):
        """~_numpy.uint8: (array of length 4)."""
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(unsigned char), format="B", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].padding_v2))
        return _numpy.asarray(arr)

    @padding_v2.setter
    def padding_v2(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field padding_v2, got {len(val)}")
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(unsigned char), format="B", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint8)
        _cyb_memcpy(<void *>(&(self._ptr[0].padding_v2)), <void *>(arr.data), sizeof(unsigned char) * len(val))

    @property
    def dispatch_num_sms(self):
        """int: """
        return self._ptr[0].dispatch_num_sms

    @dispatch_num_sms.setter
    def dispatch_num_sms(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].dispatch_num_sms = val

    @property
    def combine_num_sms(self):
        """int: """
        return self._ptr[0].combine_num_sms

    @combine_num_sms.setter
    def combine_num_sms(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].combine_num_sms = val

    @property
    def ht_flow(self):
        """int: """
        return <int>(self._ptr[0].ht_flow)

    @ht_flow.setter
    def ht_flow(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].ht_flow = <ncclEpHtFlow_t><int>val

    @property
    def meta_exchange_policy(self):
        """int: """
        return <int>(self._ptr[0].meta_exchange_policy)

    @meta_exchange_policy.setter
    def meta_exchange_policy(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].meta_exchange_policy = <ncclEpMetaExchangePolicy_t><int>val

    @property
    def num_attn_ranks(self):
        """int: Number of leading attention ranks for AFD; zero disables AFD."""
        return self._ptr[0].num_attn_ranks

    @num_attn_ranks.setter
    def num_attn_ranks(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        self._ptr[0].num_attn_ranks = val

    @property
    def padding_v4(self):
        """~_numpy.uint8: (array of length 4)."""
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(unsigned char), format="B", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].padding_v4))
        return _numpy.asarray(arr)

    @padding_v4.setter
    def padding_v4(self, val):
        if self._readonly:
            raise ValueError("This GroupConfig instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field padding_v4, got {len(val)}")
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(unsigned char), format="B", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint8)
        _cyb_memcpy(<void *>(&(self._ptr[0].padding_v4)), <void *>(arr.data), sizeof(unsigned char) * len(val))

    @staticmethod
    def from_buffer(buffer):
        """Create an GroupConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclEpGroupConfig_t), GroupConfig)

    @staticmethod
    def from_data(data):
        """Create an GroupConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `group_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "group_config_dtype", group_config_dtype, GroupConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an GroupConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef GroupConfig obj = GroupConfig.__new__(GroupConfig)
        if owner is None:
            obj._ptr = <ncclEpGroupConfig_t *>_cyb_malloc(sizeof(ncclEpGroupConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating GroupConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpGroupConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpGroupConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


###############################################################################
# Enum
###############################################################################

class Algorithm(_cyb_IntEnum):
    """
    See `ncclEpAlgorithm_t`.
    """
    LOW_LATENCY = NCCL_EP_ALGO_LOW_LATENCY
    HIGH_THROUGHPUT = NCCL_EP_ALGO_HIGH_THROUGHPUT

class OverflowPolicy(_cyb_IntEnum):
    """
    See `ncclEpOverflowPolicy_t`.
    """
    OVERFLOW_AUTO = NCCL_EP_OVERFLOW_AUTO
    OVERFLOW_TRAP = NCCL_EP_OVERFLOW_TRAP
    OVERFLOW_DROP = NCCL_EP_OVERFLOW_DROP

class Layout(_cyb_IntEnum):
    """
    See `ncclEpLayout_t`.
    """
    UNSET = NCCL_EP_LAYOUT_UNSET
    EXPERT_MAJOR = NCCL_EP_LAYOUT_EXPERT_MAJOR
    RANK_MAJOR = NCCL_EP_LAYOUT_RANK_MAJOR
    FLAT = NCCL_EP_LAYOUT_FLAT

class PassDir(_cyb_IntEnum):
    """
    See `ncclEpPassDir_t`.
    """
    FWD_PASS = NCCL_EP_FWD_PASS
    BWD_PASS = NCCL_EP_BWD_PASS

class ZeroCopyMode(_cyb_IntEnum):
    """
    See `ncclEpZeroCopyMode_t`.
    """
    ZERO_COPY_AUTO = NCCL_EP_ZERO_COPY_AUTO
    ZERO_COPY_OFF = NCCL_EP_ZERO_COPY_OFF
    ZERO_COPY_ON = NCCL_EP_ZERO_COPY_ON

class HtFlow(_cyb_IntEnum):
    """
    See `ncclEpHtFlow_t`.
    """
    AUTO = NCCL_EP_HT_FLOW_AUTO
    PUSH_PULL = NCCL_EP_HT_FLOW_PUSH_PULL
    PULL_PUSH = NCCL_EP_HT_FLOW_PULL_PUSH

class MetaExchangePolicy(_cyb_IntEnum):
    """
    See `ncclEpMetaExchangePolicy_t`.
    """
    META_EXCHANGE_AUTO = NCCL_EP_META_EXCHANGE_AUTO
    META_EXCHANGE_REQUIRE_FUSED = NCCL_EP_META_EXCHANGE_REQUIRE_FUSED
    META_EXCHANGE_FORCE_SCAN = NCCL_EP_META_EXCHANGE_FORCE_SCAN

class ExpertIdKind(_cyb_IntEnum):
    """
    See `ncclEpExpertIdKind_t`.
    """
    EXPERT_ID_AUTO = NCCL_EP_EXPERT_ID_AUTO
    EXPERT_ID_LOCAL = NCCL_EP_EXPERT_ID_LOCAL
    EXPERT_ID_GLOBAL = NCCL_EP_EXPERT_ID_GLOBAL

class DispQuant(_cyb_IntEnum):
    """
    See `ncclEpDispQuant_t`.
    """
    NONE = NCCL_EP_DISP_QUANT_NONE
    FWD = NCCL_EP_DISP_QUANT_FWD
    DS_FP8E3M4 = NCCL_EP_DISP_QUANT_DS_FP8E3M4

class CombQuant(_cyb_IntEnum):
    """
    See `ncclEpCombQuant_t`.
    """
    NONE = NCCL_EP_COMB_QUANT_NONE
    NVFP4 = NCCL_EP_COMB_QUANT_NVFP4
    MXFP8 = NCCL_EP_COMB_QUANT_MXFP8


###############################################################################
# Error handling
###############################################################################

class NCCLEpError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = f"NCCL EP error code {status}"
        super(NCCLEpError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NCCLEpError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef int get_version() except? -1:
    cdef int version
    with nogil:
        __status__ = ncclEpGetVersion(&version)
    check_status(__status__)
    return version


cpdef tensor_alloc(intptr_t tensor, unsigned int ndim, ncclDataType_t datatype, intptr_t sizes, intptr_t config):
    with nogil:
        __status__ = ncclEpTensorAlloc(<ncclEpTensor_t**>tensor, ndim, datatype, <const size_t*>sizes, <const ncclEpTensorAllocConfig_t*>config)
    check_status(__status__)


cpdef tensor_destroy(intptr_t tensor):
    with nogil:
        __status__ = ncclEpTensorDestroy(<ncclEpTensor_t*>tensor)
    check_status(__status__)


cpdef intptr_t create_group(intptr_t comm, intptr_t config) except? 0:
    cdef Group ep_group
    with nogil:
        __status__ = ncclEpCreateGroup(&ep_group, <Comm>comm, <const ncclEpGroupConfig_t*>config)
    check_status(__status__)
    return <intptr_t>ep_group


cpdef group_destroy(intptr_t ep_group):
    with nogil:
        __status__ = ncclEpGroupDestroy(<Group>ep_group)
    check_status(__status__)


cpdef size_t handle_mem_size(intptr_t ep_group, int layout, intptr_t config, int num_topk) except? -1:
    cdef size_t size_out
    with nogil:
        __status__ = ncclEpHandleMemSize(<Group>ep_group, <_Layout>layout, <const ncclEpHandleConfig_t*>config, &size_out, num_topk)
    check_status(__status__)
    return size_out


cpdef intptr_t init_handle(intptr_t ep_group, int layout, intptr_t config, int num_topk, intptr_t handle_mem) except? 0:
    cdef Handle handle
    with nogil:
        __status__ = ncclEpInitHandle(&handle, <Group>ep_group, <_Layout>layout, <const ncclEpHandleConfig_t*>config, num_topk, <const ncclEpTensor_t*>handle_mem)
    check_status(__status__)
    return <intptr_t>handle


cpdef intptr_t import_handle(intptr_t ep_group, int layout, intptr_t config, intptr_t state, intptr_t handle_mem) except? 0:
    cdef Handle handle
    with nogil:
        __status__ = ncclEpImportHandle(&handle, <Group>ep_group, <_Layout>layout, <const ncclEpHandleConfig_t*>config, <const ncclEpHandleState_t*>state, <const ncclEpTensor_t*>handle_mem)
    check_status(__status__)
    return <intptr_t>handle


cpdef update_handle(intptr_t handle, intptr_t topk_idx, intptr_t layout_info, intptr_t stream):
    with nogil:
        __status__ = ncclEpUpdateHandle(<Handle>handle, <const ncclEpTensor_t*>topk_idx, <const ncclEpLayoutInfo_t*>layout_info, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_handle(intptr_t ep_group, int layout, intptr_t topk_idx, intptr_t layout_info, intptr_t config, intptr_t stream) except? 0:
    cdef Handle handle
    with nogil:
        __status__ = ncclEpCreateHandle(&handle, <Group>ep_group, <_Layout>layout, <const ncclEpTensor_t*>topk_idx, <const ncclEpLayoutInfo_t*>layout_info, <const ncclEpHandleConfig_t*>config, <Stream>stream)
    check_status(__status__)
    return <intptr_t>handle


cpdef handle_destroy(intptr_t handle):
    with nogil:
        __status__ = ncclEpHandleDestroy(<Handle>handle)
    check_status(__status__)


cpdef dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpDispatch(<Handle>handle, <const ncclEpDispatchInputs_t*>inputs, <const ncclEpDispatchOutputs_t*>outputs, <const ncclEpLayoutInfo_t*>layout_info, <const ncclEpDispatchConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpCombine(<Handle>handle, <const ncclEpCombineInputs_t*>inputs, <const ncclEpCombineOutputs_t*>outputs, <const ncclEpCombineConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef complete(intptr_t handle, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpComplete(<Handle>handle, <const ncclEpCompleteConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef mask_query(intptr_t ep_group, intptr_t mask_status, intptr_t stream):
    with nogil:
        __status__ = ncclEpMaskQuery(<Group>ep_group, <int*>mask_status, <Stream>stream)
    check_status(__status__)


cpdef mask_update(intptr_t ep_group, intptr_t mask, intptr_t stream):
    with nogil:
        __status__ = ncclEpMaskUpdate(<Group>ep_group, <const int*>mask, <Stream>stream)
    check_status(__status__)


cpdef mask_clean(intptr_t ep_group, intptr_t stream):
    with nogil:
        __status__ = ncclEpMaskClean(<Group>ep_group, <Stream>stream)
    check_status(__status__)


cpdef int get_async_error(intptr_t ep_group) except? -1:
    cdef int error_out
    with nogil:
        __status__ = ncclEpGetAsyncError(<Group>ep_group, &error_out)
    check_status(__status__)
    return error_out


cpdef error_clear(intptr_t ep_group):
    with nogil:
        __status__ = ncclEpErrorClear(<Group>ep_group)
    check_status(__status__)


cpdef af_dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpAfDispatch(<Handle>handle, <const ncclEpDispatchInputs_t*>inputs, <const ncclEpDispatchOutputs_t*>outputs, <const ncclEpLayoutInfo_t*>layout_info, <const ncclEpDispatchConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef af_combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpAfCombine(<Handle>handle, <const ncclEpCombineInputs_t*>inputs, <const ncclEpCombineOutputs_t*>outputs, <const ncclEpCombineConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef af_recv(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpAfRecv(<Handle>handle, <const ncclEpDispatchInputs_t*>inputs, <const ncclEpDispatchOutputs_t*>outputs, <const ncclEpLayoutInfo_t*>layout_info, <const ncclEpDispatchConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef af_return(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpAfReturn(<Handle>handle, <const ncclEpCombineInputs_t*>inputs, <const ncclEpCombineOutputs_t*>outputs, <const ncclEpCombineConfig_t*>config, <Stream>stream)
    check_status(__status__)


# Hand-written: reports the .so the symbols resolved to; not a C entry point.
cpdef object get_library_path():
    from ._internal.nccl_ep import _inspect_loaded_library_path
    return _inspect_loaded_library_path()
del _cyb_IntEnum
