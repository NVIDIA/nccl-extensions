/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <nccl.h>
#include "ep_enums.h"

#include <cstdio>
#include <cstdlib>

namespace nccl_ep {
namespace jit {

// ============================================================================
// Shared source-literal helpers: host-side enums / flags -> the C++ source
// token emitted into JIT kernel source (or a variant_name cache-key suffix).
// Shared by the LL and HT source builders so the two cannot drift.
// ============================================================================

// bool -> the literal used as a compile-time template argument.
inline const char* bool_literal(bool value) {
    return value ? "true" : "false";
}

// ncclDataType_t -> enumerator name for the kTokenDtype template argument.
// Kernels only specialize on BF16/FP16/FP32; anything else maps to the BF16
// wire default.
inline const char* token_dtype_literal(ncclDataType_t dt) {
    switch (dt) {
    case ncclFloat32:
        return "ncclFloat32";
    case ncclFloat16:
        return "ncclFloat16";
    default:
        return "ncclBfloat16";
    }
}

// ncclEpLayout_t -> enumerator name for the kLayout template argument. A switch
// (not a ternary) so a newly-introduced layout aborts at JIT-build time instead
// of being silently miscompiled as another layout.
inline const char* layout_literal(ncclEpLayout_t layout) {
    switch (layout) {
    case NCCL_EP_LAYOUT_EXPERT_MAJOR:
        return "NCCL_EP_LAYOUT_EXPERT_MAJOR";
    case NCCL_EP_LAYOUT_RANK_MAJOR:
        return "NCCL_EP_LAYOUT_RANK_MAJOR";
    case NCCL_EP_LAYOUT_FLAT:
        return "NCCL_EP_LAYOUT_FLAT";
    default:
        std::fprintf(stderr, "[nccl_ep jit] unsupported ncclEpLayout_t %d in layout_literal\n",
                     static_cast<int>(layout));
        std::abort();
    }
}

// ncclEpLayout_t -> short suffix folded into a JIT variant_name so distinct
// layouts get distinct cache keys. Same switch/default contract as above.
inline const char* layout_name_tag(ncclEpLayout_t layout) {
    switch (layout) {
    case NCCL_EP_LAYOUT_EXPERT_MAJOR:
        return "_em";
    case NCCL_EP_LAYOUT_RANK_MAJOR:
        return "_rm";
    case NCCL_EP_LAYOUT_FLAT:
        return "_fl";
    default:
        std::fprintf(stderr, "[nccl_ep jit] unsupported ncclEpLayout_t %d in layout_name_tag\n",
                     static_cast<int>(layout));
        std::abort();
    }
}

} // namespace jit
} // namespace nccl_ep
