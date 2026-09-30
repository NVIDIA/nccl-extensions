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

template <ncclEpDispQuant_t kRecipe, typename ScaleT>
__forceinline__ __device__ void castAndWriteToSendBuf(
    const typename DispatchRecipeDeviceTypes<kRecipe, ScaleT>::source_vec_t* srcData,
    typename DispatchRecipeDeviceTypes<kRecipe, ScaleT>::transport_vec_t* sendBufVec,
    typename DispatchRecipeDeviceTypes<kRecipe, ScaleT>::scale_t* sendBufScales,
    int threadId,
    int numThreads,
    int laneId,
    int hiddenBf16Int4,
    bool roundScale,
    const uint8_t* inScales = nullptr,  // QUANT_FWD: raw scale bytes for this token
    int scaleBytes = 0) {               // QUANT_FWD: total bytes to copy

    // Generated and unquantized paths retain their full-warp vector contract.
    if constexpr (kRecipe != NCCL_EP_DISP_QUANT_FWD) {
        EP_DEVICE_ASSERT(hiddenBf16Int4 % 32 == 0);
    }
#pragma unroll
    for (int i = threadId; i < hiddenBf16Int4; i += numThreads) {
        if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
            sendBufVec[i] = __ldg(srcData + i);
        } else if constexpr (kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4) {
            constexpr int kElementsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
            auto dataInt4 = __ldg(srcData + i);
            auto bf16Data = reinterpret_cast<nv_bfloat16*>(&dataInt4);
            float fp32Data[kElementsPerRead];
            float amax = kFP8Margin;
            float scale;
            float scaleInv;
#pragma unroll
            for (int j = 0; j < kElementsPerRead; ++j) {
                fp32Data[j] = static_cast<float>(bf16Data[j]);
                amax = fmaxf(amax, fabsf(fp32Data[j]));
            }

            EP_STATIC_ASSERT(kElementsPerRead * 32 / kDsFp8E3M4ElementsPerScale == 2,
                             "Invalid DS_FP8E3M4 vectorization");
            amax = warp_reduce_max<16>(amax);
            calculate_fp8_scales(amax, scale, scaleInv, roundScale);
            if (laneId == 0 || laneId == 16) {
                sendBufScales[i * kElementsPerRead / kDsFp8E3M4ElementsPerScale] = scaleInv;
            }

            int2 dataInt2;
            auto fp8x2Data = reinterpret_cast<__nv_fp8x2_storage_t*>(&dataInt2);
#pragma unroll
            for (int j = 0; j < kElementsPerRead; j += 2) {
                const float2 fp32x2 = {fp32Data[j] * scale, fp32Data[j + 1] * scale};
                fp8x2Data[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
            }
            sendBufVec[i] = dataInt2;
        } else {
            auto dataInt4 = __ldg(srcData + i);
            sendBufVec[i] = *reinterpret_cast<int4*>(&dataInt4);
        }
    }
    if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
        auto* dstBytes = reinterpret_cast<uint8_t*>(sendBufScales);
        for (int i = threadId; i < scaleBytes; i += numThreads) dstBytes[i] = inScales[i];
    }
}

} // namespace ll
} // namespace nccl_ep

