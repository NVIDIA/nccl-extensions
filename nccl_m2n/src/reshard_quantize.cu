/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Block-wise FP8 (E4M3) quantize / dequantize.
 *
 * One CUDA block handles one scale block: it reduces the block's amax, derives
 * a scale, and writes the quantized bytes plus one inverse scale.  Dequantize
 * is the exact inverse and needs no reduction.
 *
 * Numerics follow nccl_ep (device_primitives.cuh calculate_fp8_scales) so the
 * two libraries agree on what "E4M3 with block scales" means:
 *   scale_inv = amax / 448, scale = 448 / amax
 * or, with roundScales, both rounded to a power of two.
 *
 * Scales are stored either as FP32 or, for MX, as E8M0 — the biased exponent
 * byte of a power-of-two scale.  E8M0 implies rounding, so selecting it forces
 * the rounded path regardless of the caller's flag.
 ************************************************************************/

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "m2n_checks.h"
#include "reshard_quantize.cuh"

namespace {

/* Matches nccl_ep's kFP8Margin / kFinfoAmaxE4M3. The margin keeps an
 * all-zero block from producing a zero or infinite scale. */
constexpr float kFp8Margin = 1e-4f;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1.0f / 448.0f;

constexpr int kQuantThreads = 256;

__device__ __forceinline__ float loadAsFloat(const void* base, ncclDataType_t dtype, size_t index) {
  switch (dtype) {
  case ncclBfloat16:
    return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]);
  case ncclFloat16:
    return __half2float(static_cast<const __half*>(base)[index]);
  default:
    return static_cast<const float*>(base)[index];
  }
}

__device__ __forceinline__ void storeFromFloat(void* base, ncclDataType_t dtype, size_t index, float value) {
  switch (dtype) {
  case ncclBfloat16:
    static_cast<__nv_bfloat16*>(base)[index] = __float2bfloat16(value);
    break;
  case ncclFloat16:
    static_cast<__half*>(base)[index] = __float2half(value);
    break;
  default:
    static_cast<float*>(base)[index] = value;
    break;
  }
}

/* scale converts payload -> fp8 domain; scaleInv converts back. */
__device__ __forceinline__ void calculateFp8Scales(float amax, bool roundScales, float* scale, float* scaleInv) {
  if (roundScales) {
    /* exp = ceil(log2(amax/448)), so the scale is an exact power of two and
     * the exponent survives the round trip.  frexpf yields ratio = m * 2^exp
     * with m in [0.5, 1), so exp already equals ceil(log2(ratio)) EXCEPT when
     * ratio is itself a power of two (m == 0.5), where ceil is exp - 1.
     * Without that correction an exact power of two would be scaled one
     * binade too far and lose a bit of mantissa. */
    const float ratio = amax * kFinfoAmaxInvE4M3;
    int exponent = 0;
    const float mantissa = frexpf(ratio, &exponent);
    if (mantissa == 0.5f) {
      exponent -= 1;
    }
    *scaleInv = ldexpf(1.0f, exponent);
    *scale = ldexpf(1.0f, -exponent);
  } else {
    *scaleInv = amax * kFinfoAmaxInvE4M3;
    *scale = kFinfoAmaxE4M3 / amax;
  }
}

/* E8M0 keeps only the biased exponent of a power-of-two value, which is the
 * MX shared-scale encoding.  Matches nccl_ep's
 * extract_required_scale_format<kIsUE8M0>. */
__device__ __forceinline__ uint8_t encodeE8m0(float value) {
  return static_cast<uint8_t>(__float_as_uint(value) >> 23);
}

__device__ __forceinline__ float decodeE8m0(uint8_t exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

__device__ __forceinline__ void storeScale(void* scales, ncclDataType_t scaleDtype, size_t index, float scaleInv) {
  if (scaleDtype == ncclUint8) {
    static_cast<uint8_t*>(scales)[index] = encodeE8m0(scaleInv);
  } else {
    static_cast<float*>(scales)[index] = scaleInv;
  }
}

__device__ __forceinline__ float loadScale(const void* scales, ncclDataType_t scaleDtype, size_t index) {
  if (scaleDtype == ncclUint8) {
    return decodeE8m0(static_cast<const uint8_t*>(scales)[index]);
  }
  return static_cast<const float*>(scales)[index];
}

/* Decompose the flat block id into (a, blockInExtent, c) and return the element
 * offset of `lane` within that block. */
__device__ __forceinline__ size_t elementOffset(const ReshardQuantGeometry geometry, size_t blockId, size_t lane) {
  const size_t perA = geometry.blocksPerRow * geometry.outerC;
  const size_t a = blockId / perA;
  const size_t rem = blockId % perA;
  const size_t blk = rem / geometry.outerC;
  const size_t c = rem % geometry.outerC;
  return a * geometry.blockExtent * geometry.outerC + (blk * geometry.blockSize + lane) * geometry.outerC + c;
}

__global__ void reshardQuantizeKernel(const void* __restrict__ src, ncclDataType_t dtype,
                                      __nv_fp8_storage_t* __restrict__ quantized, void* __restrict__ scales,
                                      ncclDataType_t scaleDtype, const ReshardQuantGeometry geometry,
                                      bool roundScales) {
  const size_t blockId = blockIdx.x;

  __shared__ float sharedAmax[kQuantThreads];

  float localAmax = kFp8Margin;
  for (size_t i = threadIdx.x; i < geometry.blockSize; i += blockDim.x) {
    const float value = loadAsFloat(src, dtype, elementOffset(geometry, blockId, i));
    localAmax = fmaxf(localAmax, fabsf(value));
  }
  sharedAmax[threadIdx.x] = localAmax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sharedAmax[threadIdx.x] = fmaxf(sharedAmax[threadIdx.x], sharedAmax[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float amax = sharedAmax[0];
  float scale = 0.0f;
  float scaleInv = 0.0f;
  calculateFp8Scales(amax, roundScales, &scale, &scaleInv);

  if (threadIdx.x == 0) {
    storeScale(scales, scaleDtype, blockId, scaleInv);
  }

  for (size_t i = threadIdx.x; i < geometry.blockSize; i += blockDim.x) {
    const size_t offset = elementOffset(geometry, blockId, i);
    const float value = loadAsFloat(src, dtype, offset) * scale;
    quantized[offset] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void reshardDequantizeKernel(const __nv_fp8_storage_t* __restrict__ quantized,
                                        const void* __restrict__ scales, ncclDataType_t scaleDtype,
                                        void* __restrict__ dst, ncclDataType_t dtype,
                                        const ReshardQuantGeometry geometry) {
  const size_t blockId = blockIdx.x;
  const float scaleInv = loadScale(scales, scaleDtype, blockId);

  for (size_t i = threadIdx.x; i < geometry.blockSize; i += blockDim.x) {
    const size_t offset = elementOffset(geometry, blockId, i);
    __nv_fp8_e4m3 encoded;
    encoded.__x = quantized[offset];
    storeFromFloat(dst, dtype, offset, static_cast<float>(encoded) * scaleInv);
  }
}

bool quantDtypeSupported(ncclDataType_t dtype) {
  return dtype == ncclBfloat16 || dtype == ncclFloat16 || dtype == ncclFloat32;
}

bool scaleDtypeSupported(ncclDataType_t dtype) {
  return dtype == ncclFloat32 || dtype == ncclUint8;
}

} // namespace

ncclResult_t reshardLaunchQuantize(const void* src, ncclDataType_t dtype, void* quantized, void* scales,
                                   ncclDataType_t scaleDtype, const ReshardQuantGeometry& geometry, bool roundScales,
                                   cudaStream_t stream) {
  NCCL_M2N_CHECK_ARG(src != nullptr && quantized != nullptr && scales != nullptr, -1,
                     "reshardLaunchQuantize: buffers must be non-null");
  NCCL_M2N_CHECK_ARG(quantDtypeSupported(dtype), -1, "reshardLaunchQuantize: unsupported payload dtype %d",
                     (int)dtype);
  NCCL_M2N_CHECK_ARG(scaleDtypeSupported(scaleDtype), -1, "reshardLaunchQuantize: unsupported scale dtype %d",
                     (int)scaleDtype);
  /* E8M0 stores only an exponent, so a non-power-of-two scale would be
   * silently truncated.  Force rounding rather than lose the mantissa. */
  const bool effectiveRound = roundScales || scaleDtype == ncclUint8;
  const size_t totalBlocks = geometry.outerA * geometry.blocksPerRow * geometry.outerC;
  if (totalBlocks == 0 || geometry.blockSize == 0) {
    return ncclSuccess;
  }
  reshardQuantizeKernel<<<static_cast<unsigned int>(totalBlocks), kQuantThreads, 0, stream>>>(
    src, dtype, static_cast<__nv_fp8_storage_t*>(quantized), scales, scaleDtype, geometry, effectiveRound);
  NCCL_M2N_CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t reshardLaunchDequantize(const void* quantized, const void* scales, ncclDataType_t scaleDtype, void* dst,
                                     ncclDataType_t dtype, const ReshardQuantGeometry& geometry,
                                     cudaStream_t stream) {
  NCCL_M2N_CHECK_ARG(quantized != nullptr && scales != nullptr && dst != nullptr, -1,
                     "reshardLaunchDequantize: buffers must be non-null");
  NCCL_M2N_CHECK_ARG(quantDtypeSupported(dtype), -1, "reshardLaunchDequantize: unsupported payload dtype %d",
                     (int)dtype);
  NCCL_M2N_CHECK_ARG(scaleDtypeSupported(scaleDtype), -1, "reshardLaunchDequantize: unsupported scale dtype %d",
                     (int)scaleDtype);
  const size_t totalBlocks = geometry.outerA * geometry.blocksPerRow * geometry.outerC;
  if (totalBlocks == 0 || geometry.blockSize == 0) {
    return ncclSuccess;
  }
  reshardDequantizeKernel<<<static_cast<unsigned int>(totalBlocks), kQuantThreads, 0, stream>>>(
    static_cast<const __nv_fp8_storage_t*>(quantized), scales, scaleDtype, dst, dtype, geometry);
  NCCL_M2N_CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
