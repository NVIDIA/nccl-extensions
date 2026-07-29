# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.

from libc.stdint cimport intptr_t, uint64_t, uintptr_t

import os
import threading

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

${snippet_linux_externs_pxd}

cdef extern from "<dlfcn.h>" nogil:
    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase
        const char* dli_sname
        void* dli_saddr
    int dladdr(const void*, Dl_info*)


###############################################################################
# Library resolution (mirrors cuda.pathfinder.load_nvidia_dynamic_lib precedence,
# adapted for libnccl_ep.so which is not registered as an NVIDIA pip wheel.)
###############################################################################

# Resolved at first import via _resolve_library_path() below. Path lookup runs
# once, then dlopen handle is cached in the lowpp ${libname} init guard.
_PACKAGE_LIB_RELPATH = os.path.join("ep", "lib", "lib${libname}.so")


def _resolve_library_path() -> str:
    # 1. nccl-extensions package path (replaces cuda.pathfinder's NVIDIA-pip-wheel
    #    step). libnccl_ep.so is at nccl_extensions/ep/lib/; this file lives in
    #    nccl_extensions/bindings/_internal/, so go up two dirs to reach
    #    nccl_extensions/.
    pkg_lib = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", _PACKAGE_LIB_RELPATH
    ))
    if os.path.exists(pkg_lib):
        return pkg_lib

    # 2. CONDA_PREFIX/lib[64]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for sub in ("lib", "lib64"):
            candidate = os.path.join(conda_prefix, sub, "lib${libname}.so")
            if os.path.exists(candidate):
                return candidate

    # 3. CUDA_HOME / CUDA_PATH lib[64]
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for sub in ("lib", "lib64"):
                candidate = os.path.join(root, sub, "lib${libname}.so")
                if os.path.exists(candidate):
                    return candidate

    # 4. SONAME fallback — let dlopen perform its own search across
    # LD_LIBRARY_PATH, /etc/ld.so.cache, and /lib, /usr/lib, /lib64,
    # /usr/lib64. If it fails the caller surfaces a clear error.
    return "lib${libname}.so"


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_${libname}_init = False

$wrapper_init


cdef void* load_library() except* with gil:
    # libnccl_ep.so has NEEDED libnccl.so.2. Pre-load it with RTLD_GLOBAL so the
    # SONAME is already mapped when libnccl_ep.so's NEEDED is resolved,
    # without depending on filesystem search.
    load_nvidia_dynamic_lib("nccl")

    cdef bytes path_bytes = _resolve_library_path().encode()
    cdef void* handle = dlopen(path_bytes, RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise RuntimeError(
            f'Failed to dlopen lib${libname} ({err_msg.decode()}); '
            f'tried path {path_bytes.decode()!r}'
        )
    return handle


cdef int _check_or_init_${libname}() except -1 nogil:
    global __py_${libname}_init
    if __py_${libname}_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_${libname}_init:
            return 0

        # Load function
${set_wrapper}
        __py_${libname}_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_${libname}()
    cdef dict data = {}

${set_functor}

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


cdef object __${libname}_loaded_so_path = None


cpdef object _inspect_loaded_library_path():
    # Path of the .so backing the loaded symbols, via dladdr() on a
    # resolved entry point. None if it cannot be determined.
    global __${libname}_loaded_so_path
    if __${libname}_loaded_so_path is not None:
        return __${libname}_loaded_so_path

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
    __${libname}_loaded_so_path = os.fsdecode(<bytes>info.dli_fname)
    return __${libname}_loaded_so_path


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs
