# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.2.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

cdef extern from * nogil:
    """
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>
        static __forceinline int atomic_int_load(int *p) {
            int v = *(int volatile *)p; _ReadBarrier(); return v;
        }
        static __forceinline void atomic_int_store(int *p, int v) {
            _WriteBarrier(); *(int volatile *)p = v;
        }
    #elif defined(__cplusplus)
        /* GCC/Clang __atomic builtins work in any C++ standard without headers */
        static inline int atomic_int_load(int *p) {
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_int_store(int *p, int v) {
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        }
    #else
        #include <stdatomic.h>
        static inline int atomic_int_load(int *p) {
            return (int)atomic_load_explicit((atomic_int *)p, memory_order_acquire);
        }
        static inline void atomic_int_store(int *p, int v) {
            atomic_store_explicit((atomic_int *)p, v, memory_order_release);
        }
    #endif

    """
    cdef int _cyb_atomic_int_load "atomic_int_load"(int *p) nogil
    cdef void _cyb_atomic_int_store "atomic_int_store"(int *p, int v) nogil

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

from libc.stdint cimport intptr_t

import threading as _cyb_threading

cdef int _cyb___py_nccl_ep_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from .utils import FunctionNotFoundError, NotSupportedError

import os
from nccl._extensions._runtime import bundled_library


cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()

    enum:
        RTLD_NOW
        RTLD_GLOBAL

    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase
        const char* dli_sname
        void* dli_saddr
    int dladdr(const void*, Dl_info*)


###############################################################################
# Library resolution. libnccl_ep.so is not an NVIDIA wheel library, so it is
# located here instead of through cuda.pathfinder.
###############################################################################

# The matching CUDA variant ships under this library's facade package.
# _resolve_library_path() runs on the first call that needs a symbol, not at
# import; after that the generated init guard holds the resolved pointers.
def _resolve_library_path() -> str:
    # 1. CUDA-specific nccl-extensions package path.
    pkg_lib = bundled_library("nccl_ep")
    if pkg_lib is not None:
        return pkg_lib

    # 2. CONDA_PREFIX/lib[64]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for sub in ("lib", "lib64"):
            candidate = os.path.join(conda_prefix, sub, "libnccl_ep.so")
            if os.path.exists(candidate):
                return candidate

    # 3. CUDA_HOME / CUDA_PATH lib[64]
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for sub in ("lib", "lib64"):
                candidate = os.path.join(root, sub, "libnccl_ep.so")
                if os.path.exists(candidate):
                    return candidate

    # 4. SONAME fallback — let dlopen perform its own search across
    # LD_LIBRARY_PATH, /etc/ld.so.cache, and /lib, /usr/lib, /lib64,
    # /usr/lib64. If it fails the caller surfaces a clear error.
    return "libnccl_ep.so"


###############################################################################
# Wrapper init
###############################################################################

cdef void* __ncclEpGetVersion = NULL
cdef void* __ncclEpTensorAlloc = NULL
cdef void* __ncclEpTensorDestroy = NULL
cdef void* __ncclEpCreateGroup = NULL
cdef void* __ncclEpGroupDestroy = NULL
cdef void* __ncclEpHandleMemSize = NULL
cdef void* __ncclEpInitHandle = NULL
cdef void* __ncclEpImportHandle = NULL
cdef void* __ncclEpUpdateHandle = NULL
cdef void* __ncclEpCreateHandle = NULL
cdef void* __ncclEpHandleDestroy = NULL
cdef void* __ncclEpDispatch = NULL
cdef void* __ncclEpCombine = NULL
cdef void* __ncclEpComplete = NULL
cdef void* __ncclEpMaskQuery = NULL
cdef void* __ncclEpMaskUpdate = NULL
cdef void* __ncclEpMaskClean = NULL
cdef void* __ncclEpGetAsyncError = NULL
cdef void* __ncclEpErrorClear = NULL
cdef void* __ncclEpAfDispatch = NULL
cdef void* __ncclEpAfCombine = NULL
cdef void* __ncclEpAfRecv = NULL
cdef void* __ncclEpAfReturn = NULL

cdef int _init_nccl_ep() except -1 nogil:
    global _cyb___py_nccl_ep_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_nccl_ep_init: return 0

        global __ncclEpGetVersion
        __ncclEpGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpGetVersion')
        if __ncclEpGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpGetVersion = _cyb_dlsym(handle, 'ncclEpGetVersion')

        global __ncclEpTensorAlloc
        __ncclEpTensorAlloc = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpTensorAlloc')
        if __ncclEpTensorAlloc == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpTensorAlloc = _cyb_dlsym(handle, 'ncclEpTensorAlloc')

        global __ncclEpTensorDestroy
        __ncclEpTensorDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpTensorDestroy')
        if __ncclEpTensorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpTensorDestroy = _cyb_dlsym(handle, 'ncclEpTensorDestroy')

        global __ncclEpCreateGroup
        __ncclEpCreateGroup = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpCreateGroup')
        if __ncclEpCreateGroup == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpCreateGroup = _cyb_dlsym(handle, 'ncclEpCreateGroup')

        global __ncclEpGroupDestroy
        __ncclEpGroupDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpGroupDestroy')
        if __ncclEpGroupDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpGroupDestroy = _cyb_dlsym(handle, 'ncclEpGroupDestroy')

        global __ncclEpHandleMemSize
        __ncclEpHandleMemSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpHandleMemSize')
        if __ncclEpHandleMemSize == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpHandleMemSize = _cyb_dlsym(handle, 'ncclEpHandleMemSize')

        global __ncclEpInitHandle
        __ncclEpInitHandle = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpInitHandle')
        if __ncclEpInitHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpInitHandle = _cyb_dlsym(handle, 'ncclEpInitHandle')

        global __ncclEpImportHandle
        __ncclEpImportHandle = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpImportHandle')
        if __ncclEpImportHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpImportHandle = _cyb_dlsym(handle, 'ncclEpImportHandle')

        global __ncclEpUpdateHandle
        __ncclEpUpdateHandle = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpUpdateHandle')
        if __ncclEpUpdateHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpUpdateHandle = _cyb_dlsym(handle, 'ncclEpUpdateHandle')

        global __ncclEpCreateHandle
        __ncclEpCreateHandle = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpCreateHandle')
        if __ncclEpCreateHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpCreateHandle = _cyb_dlsym(handle, 'ncclEpCreateHandle')

        global __ncclEpHandleDestroy
        __ncclEpHandleDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpHandleDestroy')
        if __ncclEpHandleDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpHandleDestroy = _cyb_dlsym(handle, 'ncclEpHandleDestroy')

        global __ncclEpDispatch
        __ncclEpDispatch = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpDispatch')
        if __ncclEpDispatch == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpDispatch = _cyb_dlsym(handle, 'ncclEpDispatch')

        global __ncclEpCombine
        __ncclEpCombine = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpCombine')
        if __ncclEpCombine == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpCombine = _cyb_dlsym(handle, 'ncclEpCombine')

        global __ncclEpComplete
        __ncclEpComplete = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpComplete')
        if __ncclEpComplete == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpComplete = _cyb_dlsym(handle, 'ncclEpComplete')

        global __ncclEpMaskQuery
        __ncclEpMaskQuery = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpMaskQuery')
        if __ncclEpMaskQuery == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpMaskQuery = _cyb_dlsym(handle, 'ncclEpMaskQuery')

        global __ncclEpMaskUpdate
        __ncclEpMaskUpdate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpMaskUpdate')
        if __ncclEpMaskUpdate == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpMaskUpdate = _cyb_dlsym(handle, 'ncclEpMaskUpdate')

        global __ncclEpMaskClean
        __ncclEpMaskClean = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpMaskClean')
        if __ncclEpMaskClean == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpMaskClean = _cyb_dlsym(handle, 'ncclEpMaskClean')

        global __ncclEpGetAsyncError
        __ncclEpGetAsyncError = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpGetAsyncError')
        if __ncclEpGetAsyncError == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpGetAsyncError = _cyb_dlsym(handle, 'ncclEpGetAsyncError')

        global __ncclEpErrorClear
        __ncclEpErrorClear = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpErrorClear')
        if __ncclEpErrorClear == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpErrorClear = _cyb_dlsym(handle, 'ncclEpErrorClear')

        global __ncclEpAfDispatch
        __ncclEpAfDispatch = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpAfDispatch')
        if __ncclEpAfDispatch == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpAfDispatch = _cyb_dlsym(handle, 'ncclEpAfDispatch')

        global __ncclEpAfCombine
        __ncclEpAfCombine = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpAfCombine')
        if __ncclEpAfCombine == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpAfCombine = _cyb_dlsym(handle, 'ncclEpAfCombine')

        global __ncclEpAfRecv
        __ncclEpAfRecv = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpAfRecv')
        if __ncclEpAfRecv == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpAfRecv = _cyb_dlsym(handle, 'ncclEpAfRecv')

        global __ncclEpAfReturn
        __ncclEpAfReturn = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclEpAfReturn')
        if __ncclEpAfReturn == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpAfReturn = _cyb_dlsym(handle, 'ncclEpAfReturn')

        _cyb_atomic_int_store(<int *>&_cyb___py_nccl_ep_init, 1)
        return 0

cdef inline int _check_or_init_nccl_ep() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_nccl_ep_init):
        return 0

    return _init_nccl_ep()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nccl_ep()
    cdef dict data = {}
    global __ncclEpGetVersion
    data["__ncclEpGetVersion"] = <intptr_t>__ncclEpGetVersion

    global __ncclEpTensorAlloc
    data["__ncclEpTensorAlloc"] = <intptr_t>__ncclEpTensorAlloc

    global __ncclEpTensorDestroy
    data["__ncclEpTensorDestroy"] = <intptr_t>__ncclEpTensorDestroy

    global __ncclEpCreateGroup
    data["__ncclEpCreateGroup"] = <intptr_t>__ncclEpCreateGroup

    global __ncclEpGroupDestroy
    data["__ncclEpGroupDestroy"] = <intptr_t>__ncclEpGroupDestroy

    global __ncclEpHandleMemSize
    data["__ncclEpHandleMemSize"] = <intptr_t>__ncclEpHandleMemSize

    global __ncclEpInitHandle
    data["__ncclEpInitHandle"] = <intptr_t>__ncclEpInitHandle

    global __ncclEpImportHandle
    data["__ncclEpImportHandle"] = <intptr_t>__ncclEpImportHandle

    global __ncclEpUpdateHandle
    data["__ncclEpUpdateHandle"] = <intptr_t>__ncclEpUpdateHandle

    global __ncclEpCreateHandle
    data["__ncclEpCreateHandle"] = <intptr_t>__ncclEpCreateHandle

    global __ncclEpHandleDestroy
    data["__ncclEpHandleDestroy"] = <intptr_t>__ncclEpHandleDestroy

    global __ncclEpDispatch
    data["__ncclEpDispatch"] = <intptr_t>__ncclEpDispatch

    global __ncclEpCombine
    data["__ncclEpCombine"] = <intptr_t>__ncclEpCombine

    global __ncclEpComplete
    data["__ncclEpComplete"] = <intptr_t>__ncclEpComplete

    global __ncclEpMaskQuery
    data["__ncclEpMaskQuery"] = <intptr_t>__ncclEpMaskQuery

    global __ncclEpMaskUpdate
    data["__ncclEpMaskUpdate"] = <intptr_t>__ncclEpMaskUpdate

    global __ncclEpMaskClean
    data["__ncclEpMaskClean"] = <intptr_t>__ncclEpMaskClean

    global __ncclEpGetAsyncError
    data["__ncclEpGetAsyncError"] = <intptr_t>__ncclEpGetAsyncError

    global __ncclEpErrorClear
    data["__ncclEpErrorClear"] = <intptr_t>__ncclEpErrorClear

    global __ncclEpAfDispatch
    data["__ncclEpAfDispatch"] = <intptr_t>__ncclEpAfDispatch

    global __ncclEpAfCombine
    data["__ncclEpAfCombine"] = <intptr_t>__ncclEpAfCombine

    global __ncclEpAfRecv
    data["__ncclEpAfRecv"] = <intptr_t>__ncclEpAfRecv

    global __ncclEpAfReturn
    data["__ncclEpAfReturn"] = <intptr_t>__ncclEpAfReturn
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    # libnccl_ep.so has NEEDED libnccl.so.2. Forcing nccl4py's loader to run
    # maps that SONAME RTLD_GLOBAL first, so the NEEDED resolves without a
    # filesystem search and nccl4py stays the one place that locates libnccl.
    from nccl.bindings._internal import nccl as _nccl_loader
    _nccl_loader._inspect_function_pointers()

    cdef bytes path_bytes = _resolve_library_path().encode()
    cdef void* handle = dlopen(path_bytes, RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise RuntimeError(
            f'Failed to dlopen libnccl_ep ({err_msg.decode()}); '
            f'tried path {path_bytes.decode()!r}'
        )
    return handle


cdef object __nccl_ep_loaded_so_path = None


cpdef object _inspect_loaded_library_path():
    import os
    # Path of the .so backing the loaded symbols, via dladdr() on a
    # resolved entry point. None if it cannot be determined.
    global __nccl_ep_loaded_so_path
    if __nccl_ep_loaded_so_path is not None:
        return __nccl_ep_loaded_so_path

    cdef dict ptrs = _inspect_function_pointers()
    # Any resolved symbol maps to the same .so.
    cdef intptr_t addr = 0
    for value in ptrs.values():
        if value:
            addr = value
            break

    cdef Dl_info info
    if addr == 0:
        return None
    if dladdr(<void*>addr, &info) == 0 or info.dli_fname == NULL:
        return None
    __nccl_ep_loaded_so_path = os.fsdecode(<bytes>info.dli_fname)
    return __nccl_ep_loaded_so_path


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t _ncclEpGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpGetVersion
    _check_or_init_nccl_ep()
    if __ncclEpGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpGetVersion is not found")
    return (<ncclResult_t (*)(int*) noexcept nogil>__ncclEpGetVersion)(
        version)


cdef ncclResult_t _ncclEpTensorAlloc(ncclEpTensor_t** tensor, unsigned int ndim, ncclDataType_t datatype, const size_t* sizes, const ncclEpTensorAllocConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpTensorAlloc
    _check_or_init_nccl_ep()
    if __ncclEpTensorAlloc == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpTensorAlloc is not found")
    return (<ncclResult_t (*)(ncclEpTensor_t**, unsigned int, ncclDataType_t, const size_t*, const ncclEpTensorAllocConfig_t*) noexcept nogil>__ncclEpTensorAlloc)(
        tensor, ndim, datatype, sizes, config)


cdef ncclResult_t _ncclEpTensorDestroy(ncclEpTensor_t* tensor) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpTensorDestroy
    _check_or_init_nccl_ep()
    if __ncclEpTensorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpTensorDestroy is not found")
    return (<ncclResult_t (*)(ncclEpTensor_t*) noexcept nogil>__ncclEpTensorDestroy)(
        tensor)


cdef ncclResult_t _ncclEpCreateGroup(ncclEpGroup_t* ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpCreateGroup
    _check_or_init_nccl_ep()
    if __ncclEpCreateGroup == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpCreateGroup is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t*, ncclComm_t, const ncclEpGroupConfig_t*) noexcept nogil>__ncclEpCreateGroup)(
        ep_group, comm, config)


cdef ncclResult_t _ncclEpGroupDestroy(ncclEpGroup_t ep_group) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpGroupDestroy
    _check_or_init_nccl_ep()
    if __ncclEpGroupDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpGroupDestroy is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t) noexcept nogil>__ncclEpGroupDestroy)(
        ep_group)


cdef ncclResult_t _ncclEpHandleMemSize(ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpHandleMemSize
    _check_or_init_nccl_ep()
    if __ncclEpHandleMemSize == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpHandleMemSize is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t, ncclEpLayout_t, const ncclEpHandleConfig_t*, size_t*, int) noexcept nogil>__ncclEpHandleMemSize)(
        ep_group, layout, config, size_out, num_topk)


cdef ncclResult_t _ncclEpInitHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, int num_topk, const ncclEpTensor_t* handle_mem) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpInitHandle
    _check_or_init_nccl_ep()
    if __ncclEpInitHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpInitHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t*, ncclEpGroup_t, ncclEpLayout_t, const ncclEpHandleConfig_t*, int, const ncclEpTensor_t*) noexcept nogil>__ncclEpInitHandle)(
        handle, ep_group, layout, config, num_topk, handle_mem)


cdef ncclResult_t _ncclEpImportHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, const ncclEpHandleState_t* state, const ncclEpTensor_t* handle_mem) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpImportHandle
    _check_or_init_nccl_ep()
    if __ncclEpImportHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpImportHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t*, ncclEpGroup_t, ncclEpLayout_t, const ncclEpHandleConfig_t*, const ncclEpHandleState_t*, const ncclEpTensor_t*) noexcept nogil>__ncclEpImportHandle)(
        handle, ep_group, layout, config, state, handle_mem)


cdef ncclResult_t _ncclEpUpdateHandle(ncclEpHandle_t handle, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpUpdateHandle
    _check_or_init_nccl_ep()
    if __ncclEpUpdateHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpUpdateHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpTensor_t*, const ncclEpLayoutInfo_t*, cudaStream_t) noexcept nogil>__ncclEpUpdateHandle)(
        handle, topk_idx, layout_info, stream)


cdef ncclResult_t _ncclEpCreateHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, const ncclEpHandleConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpCreateHandle
    _check_or_init_nccl_ep()
    if __ncclEpCreateHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpCreateHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t*, ncclEpGroup_t, ncclEpLayout_t, const ncclEpTensor_t*, const ncclEpLayoutInfo_t*, const ncclEpHandleConfig_t*, cudaStream_t) noexcept nogil>__ncclEpCreateHandle)(
        handle, ep_group, layout, topk_idx, layout_info, config, stream)


cdef ncclResult_t _ncclEpHandleDestroy(ncclEpHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpHandleDestroy
    _check_or_init_nccl_ep()
    if __ncclEpHandleDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpHandleDestroy is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t) noexcept nogil>__ncclEpHandleDestroy)(
        handle)


cdef ncclResult_t _ncclEpDispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpDispatch
    _check_or_init_nccl_ep()
    if __ncclEpDispatch == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpDispatch is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpDispatchInputs_t*, const ncclEpDispatchOutputs_t*, const ncclEpLayoutInfo_t*, const ncclEpDispatchConfig_t*, cudaStream_t) noexcept nogil>__ncclEpDispatch)(
        handle, inputs, outputs, layout_info, config, stream)


cdef ncclResult_t _ncclEpCombine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpCombine
    _check_or_init_nccl_ep()
    if __ncclEpCombine == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpCombine is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpCombineInputs_t*, const ncclEpCombineOutputs_t*, const ncclEpCombineConfig_t*, cudaStream_t) noexcept nogil>__ncclEpCombine)(
        handle, inputs, outputs, config, stream)


cdef ncclResult_t _ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpComplete
    _check_or_init_nccl_ep()
    if __ncclEpComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpComplete is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpCompleteConfig_t*, cudaStream_t) noexcept nogil>__ncclEpComplete)(
        handle, config, stream)


cdef ncclResult_t _ncclEpMaskQuery(ncclEpGroup_t ep_group, int* mask_status, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpMaskQuery
    _check_or_init_nccl_ep()
    if __ncclEpMaskQuery == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpMaskQuery is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t, int*, cudaStream_t) noexcept nogil>__ncclEpMaskQuery)(
        ep_group, mask_status, stream)


cdef ncclResult_t _ncclEpMaskUpdate(ncclEpGroup_t ep_group, const int* mask, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpMaskUpdate
    _check_or_init_nccl_ep()
    if __ncclEpMaskUpdate == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpMaskUpdate is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t, const int*, cudaStream_t) noexcept nogil>__ncclEpMaskUpdate)(
        ep_group, mask, stream)


cdef ncclResult_t _ncclEpMaskClean(ncclEpGroup_t ep_group, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpMaskClean
    _check_or_init_nccl_ep()
    if __ncclEpMaskClean == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpMaskClean is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t, cudaStream_t) noexcept nogil>__ncclEpMaskClean)(
        ep_group, stream)


cdef ncclResult_t _ncclEpGetAsyncError(ncclEpGroup_t ep_group, int* error_out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpGetAsyncError
    _check_or_init_nccl_ep()
    if __ncclEpGetAsyncError == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpGetAsyncError is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t, int*) noexcept nogil>__ncclEpGetAsyncError)(
        ep_group, error_out)


cdef ncclResult_t _ncclEpErrorClear(ncclEpGroup_t ep_group) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpErrorClear
    _check_or_init_nccl_ep()
    if __ncclEpErrorClear == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpErrorClear is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t) noexcept nogil>__ncclEpErrorClear)(
        ep_group)


cdef ncclResult_t _ncclEpAfDispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpAfDispatch
    _check_or_init_nccl_ep()
    if __ncclEpAfDispatch == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpAfDispatch is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpDispatchInputs_t*, const ncclEpDispatchOutputs_t*, const ncclEpLayoutInfo_t*, const ncclEpDispatchConfig_t*, cudaStream_t) noexcept nogil>__ncclEpAfDispatch)(
        handle, inputs, outputs, layout_info, config, stream)


cdef ncclResult_t _ncclEpAfCombine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpAfCombine
    _check_or_init_nccl_ep()
    if __ncclEpAfCombine == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpAfCombine is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpCombineInputs_t*, const ncclEpCombineOutputs_t*, const ncclEpCombineConfig_t*, cudaStream_t) noexcept nogil>__ncclEpAfCombine)(
        handle, inputs, outputs, config, stream)


cdef ncclResult_t _ncclEpAfRecv(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpAfRecv
    _check_or_init_nccl_ep()
    if __ncclEpAfRecv == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpAfRecv is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpDispatchInputs_t*, const ncclEpDispatchOutputs_t*, const ncclEpLayoutInfo_t*, const ncclEpDispatchConfig_t*, cudaStream_t) noexcept nogil>__ncclEpAfRecv)(
        handle, inputs, outputs, layout_info, config, stream)


cdef ncclResult_t _ncclEpAfReturn(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpAfReturn
    _check_or_init_nccl_ep()
    if __ncclEpAfReturn == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpAfReturn is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpCombineInputs_t*, const ncclEpCombineOutputs_t*, const ncclEpCombineConfig_t*, cudaStream_t) noexcept nogil>__ncclEpAfReturn)(
        handle, inputs, outputs, config, stream)
