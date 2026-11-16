/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <nccl.h>
#include "ep_enums.h"
#include "nccl_ep.h"  // ncclEpDispQuant_t (host-only helper; not JIT-included)

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

inline const char* routing_weight_type_literal(ncclDataType_t dtype) {
    switch (dtype) {
    case ncclFloat32:
        return "float";
    case ncclBfloat16:
        return "__nv_bfloat16";
    default:
        std::fprintf(stderr, "[nccl_ep jit] unsupported routing weight dtype %d\n", static_cast<int>(dtype));
        std::abort();
    }
}

// ncclDataType_t -> enumerator name for the kTokenDtype template argument.
// LL dispatch may carry an FP8 wire dtype (MXFP8); HT combine only ever passes
// BF16/FP16/FP32, so the FP8 cases are simply unused there. Anything unlisted
// maps to the BF16 wire default.
inline const char* token_dtype_literal(ncclDataType_t dt) {
    switch (dt) {
    case ncclFloat32:
        return "ncclFloat32";
    case ncclFloat16:
        return "ncclFloat16";
    case ncclFloat8e4m3:
        return "ncclFloat8e4m3";
    case ncclFloat8e5m2:
        return "ncclFloat8e5m2";
    case ncclFloat4x2:
        return "ncclFloat4x2";
    default:
        return "ncclBfloat16";
    }
}

// ncclDataType_t -> short suffix folded into a JIT variant_name so distinct wire
// dtypes get distinct cache keys (the dtype twin of layout_name_tag). Shared by
// the LL and HT variant-name builders. FP8 tags are used by LL; HT combine only
// ever passes BF16/FP16/FP32.
inline const char* token_dtype_name_tag(ncclDataType_t dt) {
    switch (dt) {
    case ncclFloat32:
        return "_tfp32";
    case ncclFloat16:
        return "_tfp16";
    case ncclFloat8e4m3:
        return "_te4m3";
    case ncclFloat8e5m2:
        return "_te5m2";
    case ncclFloat4x2:
        return "_tf4x2";
    default:
        return "_tbf16";
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

// ncclEpCombQuant_t -> enumerator name for the kRecipe / kCombineRecipe
// template argument emitted into the LL and HT combine JIT source.
inline const char* combine_recipe_literal(ncclEpCombQuant_t recipe) {
    switch (recipe) {
    case NCCL_EP_COMB_QUANT_NONE:
        return "NCCL_EP_COMB_QUANT_NONE";
    case NCCL_EP_COMB_QUANT_NVFP4:
        return "NCCL_EP_COMB_QUANT_NVFP4";
    case NCCL_EP_COMB_QUANT_MXFP8:
        return "NCCL_EP_COMB_QUANT_MXFP8";
    default:
        return nullptr;
    }
}

// ncclEpCombQuant_t -> short tag folded into a JIT variant_name so distinct recipes get
// distinct cache keys (the recipe twin of token_dtype_name_tag / layout_name_tag, and the
// combine counterpart of the dispatch spec's recipe_cache_tag). Shared by the LL and HT
// combine launchers. NONE contributes nothing, keeping unquantized variant names as they
// were before recipes existed. Covers the same set as combine_recipe_literal(): callers
// reject a null literal before naming, so a null tag cannot reach an ostringstream.
inline const char* combine_recipe_name_tag(ncclEpCombQuant_t recipe) {
    switch (recipe) {
    case NCCL_EP_COMB_QUANT_NONE:
        return "";
    case NCCL_EP_COMB_QUANT_NVFP4:
        return "_nvfp4";
    case NCCL_EP_COMB_QUANT_MXFP8:
        return "_mxfp8";
    default:
        return nullptr;
    }
}

// ncclEpDispQuant_t -> enumerator name for the kDispatchRecipe
// template argument emitted into the HT dispatch JIT source.
inline const char* dispatch_recipe_literal(ncclEpDispQuant_t recipe) {
    switch (recipe) {
    case NCCL_EP_DISP_QUANT_FWD:
        return "NCCL_EP_DISP_QUANT_FWD";
    default:
        return "NCCL_EP_DISP_QUANT_NONE";
    }
}

} // namespace jit
} // namespace nccl_ep
