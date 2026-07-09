/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "nccl_ep.h"

#include <cstdio>
#include <cstdint>

namespace nccl_ep {

// Fully resolve the selected recipe for the current dispatch byte-copy kernel.
// A future recipe that transforms or interprets values must add its own
// explicit specialization here; it cannot inherit integer-payload
// normalization.
struct DispatchKernelSpec {
    ncclDataType_t wire_token_dtype;
    unsigned int payload_bytes;
    const char* wire_dtype_literal;
    const char* payload_type_literal;
    const char* payload_cache_tag;
    const char* recipe_source_literal;
    const char* recipe_cache_tag;
};

inline ncclResult_t resolveDispatchKernelSpec(
    ncclEpDispatchQuantizationRecipe_t quantization_recipe,
    ncclDataType_t token_dtype,
    DispatchKernelSpec* spec) {
    if (spec == nullptr) {
        std::fprintf(stderr, "NCCL EP warning: null dispatch kernel specification output\n");
        return ncclInvalidArgument;
    }

    spec->wire_token_dtype = token_dtype;
    auto resolve_byte_copy_payload = [&]() -> ncclResult_t {
        switch (token_dtype) {
            case ncclFloat8e4m3:
                spec->wire_dtype_literal = "ncclFloat8e4m3";
                spec->payload_bytes = sizeof(uint8_t);
                spec->payload_type_literal = "uint8_t";
                spec->payload_cache_tag = "u8";
                return ncclSuccess;
            case ncclFloat8e5m2:
                spec->wire_dtype_literal = "ncclFloat8e5m2";
                spec->payload_bytes = sizeof(uint8_t);
                spec->payload_type_literal = "uint8_t";
                spec->payload_cache_tag = "u8";
                return ncclSuccess;
            case ncclFloat16:
            case ncclBfloat16:
                spec->wire_dtype_literal = token_dtype == ncclFloat16 ? "ncclFloat16" : "ncclBfloat16";
                spec->payload_bytes = sizeof(uint16_t);
                spec->payload_type_literal = "uint16_t";
                spec->payload_cache_tag = "u16";
                return ncclSuccess;
            case ncclFloat32:
                spec->wire_dtype_literal = "ncclFloat32";
                spec->payload_bytes = sizeof(uint32_t);
                spec->payload_type_literal = "uint32_t";
                spec->payload_cache_tag = "u32";
                return ncclSuccess;
            default:
                std::fprintf(stderr,
                             "NCCL EP warning: dispatch recipe %d cannot use token dtype %d\n",
                             static_cast<int>(quantization_recipe), static_cast<int>(token_dtype));
                return ncclInvalidArgument;
        }
    };

    switch (quantization_recipe) {
        case NCCL_EP_DISPATCH_QUANT_NONE:
            spec->recipe_source_literal = "NCCL_EP_DISPATCH_QUANT_NONE";
            spec->recipe_cache_tag = "none";
            return resolve_byte_copy_payload();
        case NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD:
            spec->recipe_source_literal = "NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD";
            spec->recipe_cache_tag = "scales_forward";
            return resolve_byte_copy_payload();
        case NCCL_EP_DISPATCH_QUANT_DS_FP8E3M4:
            spec->wire_token_dtype = ncclFloat8e4m3;
            spec->wire_dtype_literal = "ncclFloat8e4m3";
            spec->payload_bytes = sizeof(uint8_t);
            spec->payload_type_literal = "uint8_t";
            spec->payload_cache_tag = "u8";
            spec->recipe_source_literal = "NCCL_EP_DISPATCH_QUANT_DS_FP8E3M4";
            spec->recipe_cache_tag = "ds_fp8e3m4";
            return ncclSuccess;
        default:
            std::fprintf(stderr, "NCCL EP warning: unsupported dispatch quantization recipe %d\n",
                         static_cast<int>(quantization_recipe));
            return ncclInvalidArgument;
    }
}

} // namespace nccl_ep
