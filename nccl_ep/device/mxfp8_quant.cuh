/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

// MXFP8 device helpers: FP8 E4M3 + E8M0 block-scale decode, shared by HT/LL combine kernels.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "cuda_fp_features.h"

namespace nccl_ep {
namespace mxfp8 {

// Decode two packed E4M3 FP8 bytes (low byte = element 0) to float2.
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t v) {
    __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
    return __half22float2(reinterpret_cast<const __half2&>(h2));
}

// E8M0 scale byte -> 2^(E-127). Bit-cast trick: sign=0, exp=E, mantissa=0 -> exact power of two.
__device__ __forceinline__ float e8m0_to_scale(uint8_t e) {
    return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

// Decode a packed FP8x2 pair with a precomputed block scale (E8M0 decoded once per vector
// by the caller, so it is not re-decoded per pair).
__device__ __forceinline__ float2 dequant_x2_scaled(uint16_t fp8x2, float block_scale) {
    float2 v = e4m3x2_to_float2(fp8x2);
    v.x *= block_scale;
    v.y *= block_scale;
    return v;
}

// ---------------------------------------------------------------------------
// Encode helpers (FP32 -> MXFP8)
// ---------------------------------------------------------------------------

// FP8 E4M3 max representable value.
static constexpr float kE4M3Max = 448.0f;
static constexpr float kE4M3MaxRcp = 1.0f / kE4M3Max;

// Compute E8M0 biased exponent for block amax using ceiling-log2.
// E8M0 e => scale = 2^(e-127). We want the smallest scale >= amax/448.
__device__ __forceinline__ uint8_t float_to_e8m0(float amax) {
    const float scaled = amax * kE4M3MaxRcp;
    const uint32_t bits = __float_as_uint(scaled);
    uint32_t exp = (bits >> 23) & 0xFFu;
    if (bits & 0x7FFFFFu) exp++;          // ceiling: bump if mantissa is nonzero
    return static_cast<uint8_t>(min(exp, 0xFEu)); // 0xFF = NaN in E8M0; cap at 0xFE
}

// E8M0 scale byte -> 2^(127-E), the reciprocal of e8m0_to_scale. Hoisted out of
// the per-element path: every element in a block shares the scale.
//
// At e = 0xFE (the cap from float_to_e8m0, avoiding the 0xFF NaN code) the exact
// value is 2^-127: not a normal FP32 (smallest normal is 2^-126) but the
// subnormal 0x00400000. The naive
//     __uint_as_float((254u - e) << 23)
// underflows to +0.0 there. On an Inf-carrying block (amax = Inf -> e = 0xFE)
// that turned Inf * 0 = NaN, SATFINITE encoded E4M3 NaN, and dequant decoded
// NaN instead of saturating Inf to 448.
//
// CUDA 12.8+ uses the runtime __nv_fp8_e8m0 -> float cast (same pattern as
// TransformerEngine / torchao mxfp8_quantize.cuh :: reciprocal_scale; FTZ off).
// Older toolkits keep a software encoding of that subnormal so this header
// still compiles; host_build_supports_mxfp8() rejects MXFP8 on those builds.
__device__ __forceinline__ float e8m0_to_scale_inv(uint8_t e) {
#if NCCL_EP_HAS_CUDA_E8M0_TYPE
    __nv_fp8_e8m0 tmp;
    tmp.__x = static_cast<__nv_fp8_storage_t>(254u - e);
    return static_cast<float>(tmp);
#else
    if (e == 0xFEu) return __uint_as_float(0x00400000u);
    return __uint_as_float((254u - e) << 23);
#endif
}

// Quantize two floats to a packed E4M3 pair (low byte = element 0) with a
// precomputed reciprocal block scale.
__device__ __forceinline__ uint16_t pack_e4m3x2_scaled(float2 v, float scale_inv) {
    const float2 scaled = make_float2(v.x * scale_inv, v.y * scale_inv);
    return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3));
}

// Max |x| over four packed pairs (8 elements).
__device__ __forceinline__ float amax_x8(const float2* v) {
    float amax = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; p++) {
        amax = fmaxf(amax, fmaxf(fabsf(v[p].x), fabsf(v[p].y)));
    }
    return amax;
}

// Quantize 8 elements (four pairs) to 8 packed E4M3 bytes with a reciprocal block scale.
__device__ __forceinline__ uint2 pack_e4m3x8_scaled(const float2* v, float scale_inv) {
    uint2 out;
    uint16_t* const pairs = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
    for (int p = 0; p < 4; p++) {
        pairs[p] = pack_e4m3x2_scaled(v[p], scale_inv);
    }
    return out;
}
} // namespace mxfp8
} // namespace nccl_ep
