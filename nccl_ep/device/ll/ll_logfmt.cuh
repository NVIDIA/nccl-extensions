/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "../device_primitives.cuh"

namespace nccl_ep {

namespace ll {

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmtEncode(void* buffer, nv_bfloat162* sharedAmaxmin, const int& laneId) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4Data[kNumSendUnrolls];
    const auto& uint32Data = reinterpret_cast<uint32_t*>(int4Data);
    const auto& bf162Data = reinterpret_cast<nv_bfloat162*>(int4Data);

    // Calculate lane offset
    const auto& loadBuf =
        reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + laneId * (kNumSendUnrolls * sizeof(int4)));
    const auto& storeBuf = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(buffer) + laneId * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    auto bf162Amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162Amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t localSigns = 0;
#pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++k) {
        uint32Data[k] = loadBuf[k];
        localSigns |= ((uint32Data[k] >> 15) & 1) << (k * 2);
        localSigns |= ((uint32Data[k] >> 31) & 1) << (k * 2 + 1);
        uint32Data[k] &= 0x7fff7fff;

        bf162Amax = __hmax2(bf162Amax, bf162Data[k]);
        bf162Amin = __hmin2(bf162Amin, bf162Data[k]);
    }

    // Reduce per 128 channels
    auto amax = std::max(static_cast<float>(bf162Amax.x), static_cast<float>(bf162Amax.y));
    auto amin = std::min(static_cast<float>(bf162Amin.x), static_cast<float>(bf162Amin.y));
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (sharedAmaxmin != nullptr) *sharedAmaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto& logAmax = log2f_approx(amax);
    const auto& logAmin = fmaxf(log2f_approx(amin), logAmax - kMinClip);
    const bool& enableCast = warp_reduce_and<kNumLanesToReduce, true>(logAmax < kLogThreshold and logAmin < logAmax);

    // Case into LogFMT-10 if satisfied
    if (enableCast) {
        const auto step = (logAmax - logAmin) / static_cast<float>(kNumValues - 2);
        const auto stepInv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * stepInv;
        const auto fusedRounding = rounding - logAmin * stepInv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encodedData[kNumElemsPerInt4 * 2];
#pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++i) {
#pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++k) {
                const auto& [x, y] = __bfloat1622float2(bf162Data[i * kNumElemsPerInt4 + k]);
                encodedData[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * stepInv + fusedRounding, 0));
                encodedData[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * stepInv + fusedRounding, 0));
            }
            storeBuf[i * 5 + 0] =
                (encodedData[0] >> 0) | (encodedData[1] << 9) | (encodedData[2] << 18) | (encodedData[3] << 27);
            storeBuf[i * 5 + 1] = (encodedData[3] >> 5) | (encodedData[4] << 4) | (encodedData[5] << 13) |
                                  (encodedData[6] << 22) | (encodedData[7] << 31);
            storeBuf[i * 5 + 2] =
                (encodedData[7] >> 1) | (encodedData[8] << 8) | (encodedData[9] << 17) | (encodedData[10] << 26);
            storeBuf[i * 5 + 3] = (encodedData[10] >> 6) | (encodedData[11] << 3) | (encodedData[12] << 12) |
                                  (encodedData[13] << 21) | (encodedData[14] << 30);
            storeBuf[i * 5 + 4] = (encodedData[14] >> 2) | (encodedData[15] << 7) |
                                  ((i == 0) ? (localSigns << 16) : (localSigns & 0xffff0000u));
        }
    }

    // Min/max metadata is written even when the payload stays uncompressed.
    // Publish every lane's generic shared-memory writes to the async proxy,
    // then converge before the elected lane issues the TMA copies.
    tma_store_fence();
    __syncwarp();

    // Return TMA copy bytes
    return enableCast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)) :
                        (32 * (kNumSendUnrolls * sizeof(int4)));
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmtCheckAmaxmin(
    uint8_t* metaBuffer,
    float2* sharedLogAmax,
    float2* sharedLogAmin,
    int* sharedCastInfo,
    const int laneId) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`

    bool enableCast = true;
    if (laneId < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxminData = reinterpret_cast<uint64_t*>(metaBuffer)[laneId];
        const auto& bf162Amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxminData);
        float logAmax[2], logAmin[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            auto amax = static_cast<float>(bf162Amaxmin[i].x);
            auto amin = static_cast<float>(bf162Amaxmin[i].y);
            logAmax[i] = log2f_approx(amax);
            logAmin[i] = amin == 0 ? logAmax[i] - kMinClip : fmaxf(log2f_approx(amin), logAmax[i] - kMinClip);
            enableCast = enableCast and logAmax[i] < kLogThreshold and logAmin[i] < logAmax[i];
        }
        sharedLogAmax[laneId] = make_float2(logAmax[0], logAmax[1]);
        sharedLogAmin[laneId] = make_float2(logAmin[0], logAmin[1]);
    }

    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enableCast) ? 1u << (laneId / kNumRecvUnrolls) : 0u;
    const auto& numCastedPrefix =
        __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (laneId / kNumRecvUnrolls)) - 1));

    if (laneId < kNumLanes and laneId % kNumRecvUnrolls == 0)
        sharedCastInfo[laneId / kNumRecvUnrolls] = (numCastedPrefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

// Decodes a LogFMT-10 encoded chunk (the inverse of logfmtEncode above: same
// step/logAmin math, 10-bit/5-way-packed layout, and sign bits) and
// accumulates the weighted values into `accum`.
template <int kNumRecvUnrolls>
__forceinline__ __device__ void logfmtDecodeAccumulate(
    uint32_t* ldBuffer,
    float* accum,
    const float& logAmax,
    const float& logAmin,
    const float& weight) {
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    const auto& step = (logAmax - logAmin) / static_cast<float>(kNumValues - 2);
    auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
        const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + logAmin);
        return sign ? -decoded : decoded;
    };

    EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
#pragma unroll
    for (int i = 0; i < kNumRecvUnrolls / 2; ++i) {
        uint32_t concatData[6];
        concatData[0] = ldBuffer[i * 5];
#pragma unroll
        for (int k = 1; k < 5; ++k)
            concatData[k] = (ldBuffer[i * 5 + k - 1] >> (32 - k * 5)) | (ldBuffer[i * 5 + k] << (k * 5));
        concatData[5] = ldBuffer[i * 5 + 4] >> 7;

        const uint32_t& localSigns = ldBuffer[i * 5 + 4] >> 16;
#pragma unroll
        for (int k = 0; k < 5; ++k) {
            accum[i * 16 + k * 3 + 0] +=
                decode((concatData[k] >> 0) & 0x1ff, (localSigns >> (k * 3 + 0)) & 1) * weight;
            accum[i * 16 + k * 3 + 1] +=
                decode((concatData[k] >> 9) & 0x1ff, (localSigns >> (k * 3 + 1)) & 1) * weight;
            accum[i * 16 + k * 3 + 2] +=
                decode((concatData[k] >> 18) & 0x1ff, (localSigns >> (k * 3 + 2)) & 1) * weight;
        }
        accum[i * 16 + 15] += decode(concatData[5] & 0x1ff, (localSigns >> 15) & 1) * weight;
    }
}

} // namespace ll

} // namespace nccl_ep
