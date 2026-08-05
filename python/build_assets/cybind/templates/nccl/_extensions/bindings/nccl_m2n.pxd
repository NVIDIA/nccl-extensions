# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated $version_span. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cy${libname} cimport *


###############################################################################
# Types
###############################################################################

$type_decls

ctypedef ncclComm_t Comm
ctypedef ncclWindow_t Window
ctypedef cudaStream_t Stream


###############################################################################
# Functions
###############################################################################

cpdef intptr_t init(intptr_t config) except? 0
cpdef finalize(intptr_t handle)
cpdef reshard_with_window(intptr_t handle, intptr_t comm, intptr_t window, intptr_t src, intptr_t dst, intptr_t stream)
cpdef reshard(intptr_t handle, intptr_t comm, intptr_t src, intptr_t dst, intptr_t stream)
