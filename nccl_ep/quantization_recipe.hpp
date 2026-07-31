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
    const char* scale_type_literal;
    const char* scale_cache_tag;
    const char* recipe_source_literal;
    const char* recipe_cache_tag;
};

inline ncclResult_t resolveDispatchKernelSpec(
    ncclEpDispQuant_t recipe,
    ncclDataType_t token_dtype,
    ncclDataType_t scale_dtype,
    DispatchKernelSpec* spec) {
    if (spec == nullptr) {
        std::fprintf(stderr, "NCCL EP warning: null dispatch kernel specification output\n");
        return ncclInvalidArgument;
    }

    spec->wire_token_dtype = token_dtype;
    spec->scale_type_literal = "uint8_t";
    spec->scale_cache_tag = "none";
    auto resolve_byte_copy_payload = [&]() -> ncclResult_t {
        switch (token_dtype) {
            case ncclFloat8e4m3:
                spec->wire_dtype_literal = "ncclFloat8e4m3";
                spec->payload_bytes = sizeof(uint8_t);
                spec->payload_type_literal = "uint8_t";
                spec->payload_cache_tag = "u8";
                return ncclSuccess;
            case ncclFloat8e5m2:
            case ncclFloat4x2:
                spec->wire_dtype_literal = "ncclFloat8e5m2";
                if (token_dtype == ncclFloat4x2) spec->wire_dtype_literal = "ncclFloat4x2";
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
                             static_cast<int>(recipe), static_cast<int>(token_dtype));
                return ncclInvalidArgument;
        }
    };

    switch (recipe) {
        case NCCL_EP_DISP_QUANT_NONE:
            spec->recipe_source_literal = "NCCL_EP_DISP_QUANT_NONE";
            spec->recipe_cache_tag = "none";
            return resolve_byte_copy_payload();
        case NCCL_EP_DISP_QUANT_FWD:
            spec->recipe_source_literal = "NCCL_EP_DISP_QUANT_FWD";
            spec->recipe_cache_tag = "scales_forward";
            switch (scale_dtype) {
                case ncclFloat32: spec->scale_type_literal = "float"; spec->scale_cache_tag = "fp32"; break;
                case ncclFloat16: spec->scale_type_literal = "__half"; spec->scale_cache_tag = "fp16"; break;
                case ncclBfloat16: spec->scale_type_literal = "nv_bfloat16"; spec->scale_cache_tag = "bf16"; break;
                case ncclFloat8e4m3: spec->scale_type_literal = "__nv_fp8_e4m3"; spec->scale_cache_tag = "e4m3"; break;
                case ncclFloat8e5m2: spec->scale_type_literal = "__nv_fp8_e5m2"; spec->scale_cache_tag = "e5m2"; break;
                case ncclUint8: spec->scale_type_literal = "uint8_t"; spec->scale_cache_tag = "u8"; break;
                default:
                    std::fprintf(stderr, "NCCL EP warning: SCALES_FORWARD cannot use scale dtype %d\n",
                                 static_cast<int>(scale_dtype));
                    return ncclInvalidArgument;
            }
            return resolve_byte_copy_payload();
        case NCCL_EP_DISP_QUANT_DS_FP8E3M4:
            spec->wire_token_dtype = ncclFloat8e4m3;
            spec->wire_dtype_literal = "ncclFloat8e4m3";
            spec->payload_bytes = sizeof(uint8_t);
            spec->payload_type_literal = "uint8_t";
            spec->payload_cache_tag = "u8";
            spec->scale_type_literal = "float";
            spec->scale_cache_tag = "fp32";
            spec->recipe_source_literal = "NCCL_EP_DISP_QUANT_DS_FP8E3M4";
            spec->recipe_cache_tag = "ds_fp8e3m4";
            return ncclSuccess;
        default:
            std::fprintf(stderr, "NCCL EP warning: unsupported dispatch quantization recipe %d\n",
                         static_cast<int>(recipe));
            return ncclInvalidArgument;
    }
}

} // namespace nccl_ep
