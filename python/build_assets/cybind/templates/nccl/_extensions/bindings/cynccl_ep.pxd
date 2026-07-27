# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.


from libc.stdint cimport uint64_t

$external_imports


###############################################################################
# Types (structs, enums, ...)
###############################################################################

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
