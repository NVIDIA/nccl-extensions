/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <type_traits>
#include "../device_primitives.cuh"

namespace nccl_ep {
namespace ll {

// Device-side representation for each implemented dispatch recipe. Keep the
// source vector, transport vector, and scale element type together: recipes
// that use another scale encoding or do value-aware transport must define
// their own specialization rather than inheriting an unrelated default.
template <ncclEpDispQuant_t kRecipe, typename ScaleT>
struct DispatchRecipeDeviceTypes;

template <typename ScaleT>
struct DispatchRecipeDeviceTypes<NCCL_EP_DISP_QUANT_NONE, ScaleT> {
    using source_vec_t = int4;
    using transport_vec_t = int4;
    using scale_t = float;  // Unused: NONE has no scale payload.

    template <ncclDataType_t kTokenDtype>
    __host__ __device__ static constexpr size_t token_payload_bytes(size_t tensor_hidden) {
        return tensor_hidden * size_u8<kTokenDtype>();
    }
};

template <typename ScaleT>
struct DispatchRecipeDeviceTypes<NCCL_EP_DISP_QUANT_FWD, ScaleT> {
    // tensor_hidden is physical width; ncclFloat4x2 tensors are already H/2.
    using source_vec_t = int4;
    using transport_vec_t = int4;
    using scale_t = ScaleT;

    template <ncclDataType_t kTokenDtype>
    __host__ __device__ static constexpr size_t token_payload_bytes(size_t tensor_hidden) {
        return tensor_hidden * size_u8<kTokenDtype>();
    }
};

template <typename ScaleT>
struct DispatchRecipeDeviceTypes<NCCL_EP_DISP_QUANT_DS_FP8E3M4, ScaleT> {
    static_assert(std::is_same_v<ScaleT, float>,
                  "DS_FP8E3M4 always generates FP32 inverse scales");
    using source_vec_t = int4;
    using transport_vec_t = int2;
    using scale_t = float;

    template <ncclDataType_t>
    __host__ __device__ static constexpr size_t token_payload_bytes(size_t tensor_hidden) {
        // This recipe generates one FP8 byte for each BF16 input element.
        return tensor_hidden * sizeof(uint8_t);
    }
};

// Geometry of the DS-FP8 recipe's int4 input and half-warp scale reduction.
namespace ds_fp8 {
constexpr int kLanesPerScale = kDsFp8E3M4ElementsPerScale / kBf16PerInt4;
constexpr int kElementsPerWarp = kWarpSize * kBf16PerInt4;
constexpr int kScalesPerWarp = kElementsPerWarp / kDsFp8E3M4ElementsPerScale;
static_assert(kLanesPerScale == kWarpSize / 2, "DS-FP8 scales require half-warp groups");
} // namespace ds_fp8

// Quantize eight BF16 elements per lane. All 32 lanes must participate;
// each half warp shares one scale for 128 elements. Memory placement and
// synchronization for overlapping input/output belong to the caller.
__forceinline__ __device__ int2 quantizeDsFp8Vector(int4 packed, bool roundScale, float& scaleInv) {
    constexpr int kElementsPerRead = ::nccl_ep::kBf16PerInt4;
    EP_STATIC_ASSERT(kElementsPerRead * ds_fp8::kLanesPerScale == kDsFp8E3M4ElementsPerScale,
                     "Invalid DS_FP8E3M4 vectorization");
    const auto* bf16 = reinterpret_cast<const nv_bfloat16*>(&packed);
    float values[kElementsPerRead];
    float amax = kFP8Margin;
#pragma unroll
    for (int j = 0; j < kElementsPerRead; ++j) {
        values[j] = static_cast<float>(bf16[j]);
        amax = fmaxf(amax, fabsf(values[j]));
    }
    amax = warp_reduce_max<ds_fp8::kLanesPerScale>(amax);
    float scale;
    calculate_fp8_scales(amax, scale, scaleInv, roundScale);
    int2 quantized;
    auto* fp8 = reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized);
#pragma unroll
    for (int j = 0; j < kElementsPerRead; j += 2) {
        const float2 scaled = {values[j] * scale, values[j + 1] * scale};
        fp8[j / 2] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    return quantized;
}

} // namespace ll
} // namespace nccl_ep
