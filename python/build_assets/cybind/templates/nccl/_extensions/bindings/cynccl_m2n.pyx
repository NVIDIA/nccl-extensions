# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated $version_span. Do not modify it directly.

from ._internal cimport $libname as _$libname


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs


cdef const char* ncclM2nGetLastError() noexcept nogil:
    return _$libname._ncclM2nGetLastError()
