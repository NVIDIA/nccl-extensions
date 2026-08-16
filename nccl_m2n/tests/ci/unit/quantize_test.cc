/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * quantize_test.cc — block-wise E4M3 quantize/dequantize numerics.
 *
 * These kernels need no communicator, so unlike the reshard paths they can be
 * exercised on a single GPU.  Covered:
 *   - round-trip error stays within the E4M3 relative-precision bound
 *   - per-block scaling: a block with a large outlier does not degrade the
 *     neighbouring block
 *   - roundScales gives exactly-representable power-of-two scales
 *   - degenerate blocks (all zeros) survive without NaN/Inf
 *   - every supported payload dtype round-trips
 ************************************************************************/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_m2n.h"
#include "reshard_quantize.cuh"

namespace {

bool cudaAvailable() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace

/* The vendored googletest predates GTEST_SKIP, so mirror the guarded pattern
 * the MPI suite uses rather than silently passing on a host with no GPU. */
#if defined(GTEST_SKIP)
#define QUANT_REQUIRE_CUDA()                     \
  do {                                           \
    if (!cudaAvailable()) GTEST_SKIP() << "no CUDA device"; \
  } while (0)
#else
#define QUANT_REQUIRE_CUDA()                                        \
  do {                                                              \
    if (!cudaAvailable()) {                                         \
      GTEST_LOG_(WARNING) << "no CUDA device; skipping";            \
      return;                                                       \
    }                                                               \
  } while (0)
#endif

namespace {

/* E4M3 carries 3 mantissa bits, so within a block the relative error of a
 * value near the block maximum is bounded by 2^-4 = 6.25%.  Values far below
 * the block max lose proportionally more, which is inherent to block scaling
 * and is why the tests below compare against the block max, not the value. */
constexpr float kE4m3RelStep = 1.0f / 16.0f;

struct QuantRoundTrip {
  std::vector<float> output;
  std::vector<float> scales;
};

/* Runs src -> quantize -> dequantize -> output entirely on device. */
QuantRoundTrip roundTrip(const std::vector<float>& input, size_t outer, size_t blocksPerRow, size_t blockSize,
                         bool roundScales, ncclDataType_t dtype = ncclFloat32) {
  ReshardQuantGeometry geometry{};
  geometry.outerA = outer;
  geometry.blockExtent = blocksPerRow * blockSize;
  geometry.outerC = 1;
  geometry.blocksPerRow = blocksPerRow;
  geometry.blockSize = blockSize;

  const size_t elements = outer * blocksPerRow * blockSize;
  const size_t totalBlocks = outer * blocksPerRow;
  const size_t elementSize = (dtype == ncclFloat32) ? 4 : 2;

  void* devSrc = nullptr;
  void* devDst = nullptr;
  void* devQuant = nullptr;
  float* devScales = nullptr;
  EXPECT_EQ(cudaSuccess, cudaMalloc(&devSrc, elements * elementSize));
  EXPECT_EQ(cudaSuccess, cudaMalloc(&devDst, elements * elementSize));
  EXPECT_EQ(cudaSuccess, cudaMalloc(&devQuant, elements));
  EXPECT_EQ(cudaSuccess, cudaMalloc(&devScales, totalBlocks * sizeof(float)));

  /* Stage the input through the same conversion the kernel will use, so the
   * comparison isolates quantization error rather than dtype conversion. */
  if (dtype == ncclFloat32) {
    EXPECT_EQ(cudaSuccess, cudaMemcpy(devSrc, input.data(), elements * 4, cudaMemcpyHostToDevice));
  } else {
    /* Host-side bf16 encode is a plain truncation of the fp32 bit pattern.
     * fp16 needs a real converter and is not exercised by this helper. */
    EXPECT_EQ(ncclBfloat16, dtype) << "roundTrip() supports ncclFloat32 and ncclBfloat16 only";
    std::vector<uint16_t> narrow(elements);
    for (size_t i = 0; i < elements; i++) {
      uint32_t bits = 0;
      std::memcpy(&bits, &input[i], 4);
      narrow[i] = static_cast<uint16_t>(bits >> 16);
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(devSrc, narrow.data(), elements * 2, cudaMemcpyHostToDevice));
  }

  EXPECT_EQ(ncclSuccess, reshardLaunchQuantize(devSrc, dtype, devQuant, devScales, ncclFloat32, geometry, roundScales, nullptr));
  EXPECT_EQ(ncclSuccess, reshardLaunchDequantize(devQuant, devScales, ncclFloat32, devDst, dtype, geometry, nullptr));
  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  QuantRoundTrip result;
  result.output.resize(elements);
  result.scales.resize(totalBlocks);
  if (dtype == ncclFloat32) {
    EXPECT_EQ(cudaSuccess, cudaMemcpy(result.output.data(), devDst, elements * 4, cudaMemcpyDeviceToHost));
  } else {
    std::vector<uint16_t> narrow(elements);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(narrow.data(), devDst, elements * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elements; i++) {
      uint32_t bits = static_cast<uint32_t>(narrow[i]) << 16;
      std::memcpy(&result.output[i], &bits, 4);
    }
  }
  EXPECT_EQ(cudaSuccess, cudaMemcpy(result.scales.data(), devScales, totalBlocks * sizeof(float),
                                    cudaMemcpyDeviceToHost));

  cudaFree(devSrc);
  cudaFree(devDst);
  cudaFree(devQuant);
  cudaFree(devScales);
  return result;
}

} // namespace

TEST(QuantizeTest, RoundTripStaysWithinE4m3Precision) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 128;
  std::vector<float> input(kBlockSize);
  for (size_t i = 0; i < kBlockSize; i++) {
    input[i] = static_cast<float>(i + 1) * 0.37f;
  }

  const QuantRoundTrip result = roundTrip(input, 1, 1, kBlockSize, /*roundScales=*/false);

  float blockMax = 0.0f;
  for (float v : input) blockMax = std::fmax(blockMax, std::fabs(v));

  for (size_t i = 0; i < kBlockSize; i++) {
    /* Absolute error is bounded by the block max times the E4M3 step, since
     * every value shares one scale. */
    EXPECT_LE(std::fabs(result.output[i] - input[i]), blockMax * kE4m3RelStep)
      << "index " << i << " in=" << input[i] << " out=" << result.output[i];
  }
}

TEST(QuantizeTest, BlocksAreScaledIndependently) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 64;
  constexpr size_t kBlocks = 2;
  std::vector<float> input(kBlockSize * kBlocks);
  for (size_t i = 0; i < kBlockSize; i++) {
    input[i] = 1.0f + static_cast<float>(i) * 0.01f; /* block 0: small values */
  }
  for (size_t i = 0; i < kBlockSize; i++) {
    input[kBlockSize + i] = 10000.0f; /* block 1: a huge outlier block */
  }

  const QuantRoundTrip result = roundTrip(input, 1, kBlocks, kBlockSize, /*roundScales=*/false);

  ASSERT_EQ(2u, result.scales.size());
  EXPECT_LT(result.scales[0], result.scales[1]) << "each block must derive its own scale";

  /* The small block must NOT be degraded by the huge neighbouring block; that
   * is the entire point of per-block scaling. A single tensor-wide scale would
   * quantize these to zero. */
  for (size_t i = 0; i < kBlockSize; i++) {
    EXPECT_LE(std::fabs(result.output[i] - input[i]), 1.1f * kE4m3RelStep) << "index " << i;
    EXPECT_GT(result.output[i], 0.9f) << "small block collapsed toward zero at index " << i;
  }
}

TEST(QuantizeTest, RoundScalesProducesPowersOfTwo) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 32;
  std::vector<float> input(kBlockSize);
  for (size_t i = 0; i < kBlockSize; i++) {
    input[i] = static_cast<float>(i) * 3.3f - 17.0f;
  }

  const QuantRoundTrip result = roundTrip(input, 1, 1, kBlockSize, /*roundScales=*/true);

  ASSERT_EQ(1u, result.scales.size());
  const float scaleInv = result.scales[0];
  int exponent = 0;
  const float mantissa = std::frexp(scaleInv, &exponent);
  EXPECT_FLOAT_EQ(0.5f, mantissa) << "roundScales must yield an exact power of two, got " << scaleInv;
}

TEST(QuantizeTest, RoundScalesHandlesExactPowerOfTwoAmax) {
  QUANT_REQUIRE_CUDA();

  /* amax/448 lands exactly on a power of two.  frexp returns mantissa 0.5
   * here, so without the ceil correction the scale would be one binade too
   * large and needlessly lose a mantissa bit. */
  constexpr size_t kBlockSize = 32;
  std::vector<float> input(kBlockSize, 0.0f);
  input[0] = 448.0f * 4.0f; /* ratio == 4.0 exactly */

  const QuantRoundTrip result = roundTrip(input, 1, 1, kBlockSize, /*roundScales=*/true);

  ASSERT_EQ(1u, result.scales.size());
  EXPECT_FLOAT_EQ(4.0f, result.scales[0]) << "expected the tight power-of-two scale, not 8.0";
  /* With the tight scale the maximum is representable exactly. */
  EXPECT_FLOAT_EQ(input[0], result.output[0]);
}

TEST(QuantizeTest, AllZeroBlockIsFiniteAndZero) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 64;
  const std::vector<float> input(kBlockSize, 0.0f);

  const QuantRoundTrip result = roundTrip(input, 1, 1, kBlockSize, /*roundScales=*/false);

  ASSERT_EQ(1u, result.scales.size());
  EXPECT_TRUE(std::isfinite(result.scales[0])) << "zero block must not produce a zero or infinite scale";
  for (size_t i = 0; i < kBlockSize; i++) {
    EXPECT_TRUE(std::isfinite(result.output[i]));
    EXPECT_FLOAT_EQ(0.0f, result.output[i]);
  }
}

TEST(QuantizeTest, NegativeValuesSurviveRoundTrip) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 64;
  std::vector<float> input(kBlockSize);
  for (size_t i = 0; i < kBlockSize; i++) {
    input[i] = (i % 2 == 0 ? 1.0f : -1.0f) * (1.0f + static_cast<float>(i));
  }

  const QuantRoundTrip result = roundTrip(input, 1, 1, kBlockSize, /*roundScales=*/false);

  float blockMax = 0.0f;
  for (float v : input) blockMax = std::fmax(blockMax, std::fabs(v));
  for (size_t i = 0; i < kBlockSize; i++) {
    EXPECT_EQ(input[i] < 0.0f, result.output[i] < 0.0f) << "sign flipped at index " << i;
    EXPECT_LE(std::fabs(result.output[i] - input[i]), blockMax * kE4m3RelStep);
  }
}

TEST(QuantizeTest, MultipleRowsUseIndependentScales) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 32;
  constexpr size_t kRows = 4;
  std::vector<float> input(kRows * kBlockSize);
  for (size_t r = 0; r < kRows; r++) {
    const float magnitude = std::ldexp(1.0f, static_cast<int>(r) * 5);
    for (size_t i = 0; i < kBlockSize; i++) {
      input[r * kBlockSize + i] = magnitude * (1.0f + static_cast<float>(i) / kBlockSize);
    }
  }

  const QuantRoundTrip result = roundTrip(input, kRows, 1, kBlockSize, /*roundScales=*/false);

  ASSERT_EQ(kRows, result.scales.size());
  for (size_t r = 1; r < kRows; r++) {
    EXPECT_GT(result.scales[r], result.scales[r - 1]) << "row " << r << " should scale up with its magnitude";
  }
  for (size_t r = 0; r < kRows; r++) {
    float rowMax = 0.0f;
    for (size_t i = 0; i < kBlockSize; i++) rowMax = std::fmax(rowMax, std::fabs(input[r * kBlockSize + i]));
    for (size_t i = 0; i < kBlockSize; i++) {
      const size_t idx = r * kBlockSize + i;
      EXPECT_LE(std::fabs(result.output[idx] - input[idx]), rowMax * kE4m3RelStep) << "row " << r << " index " << i;
    }
  }
}

TEST(QuantizeTest, Bfloat16PayloadRoundTrips) {
  QUANT_REQUIRE_CUDA();

  constexpr size_t kBlockSize = 64;
  std::vector<float> input(kBlockSize);
  for (size_t i = 0; i < kBlockSize; i++) {
    /* Powers of two are exact in bf16, so the comparison isolates the E4M3
     * step from bf16 rounding. */
    input[i] = std::ldexp(1.0f, static_cast<int>(i % 4));
  }

  const QuantRoundTrip result = roundTrip(input, 1, 1, kBlockSize, /*roundScales=*/true, ncclBfloat16);

  float blockMax = 0.0f;
  for (float v : input) blockMax = std::fmax(blockMax, std::fabs(v));
  for (size_t i = 0; i < kBlockSize; i++) {
    EXPECT_LE(std::fabs(result.output[i] - input[i]), blockMax * kE4m3RelStep) << "index " << i;
  }
}

TEST(QuantizeTest, NonInnermostBlockDimStridesCorrectly) {
  QUANT_REQUIRE_CUDA();

  /* [outerA=2, blockExtent=8, outerC=3] with blockSize=4: a block spans four
   * elements strided by 3, not four contiguous ones.  Each (a, c) pair gets its
   * own magnitude so a stride bug shows up as cross-contamination rather than
   * as a small numeric error. */
  constexpr size_t kOuterA = 2, kExtent = 8, kOuterC = 3, kBlockSize = 4;
  constexpr size_t kElements = kOuterA * kExtent * kOuterC;
  constexpr size_t kBlocksPerRow = kExtent / kBlockSize;
  constexpr size_t kTotalBlocks = kOuterA * kBlocksPerRow * kOuterC;

  std::vector<float> input(kElements);
  for (size_t a = 0; a < kOuterA; a++) {
    for (size_t j = 0; j < kExtent; j++) {
      for (size_t c = 0; c < kOuterC; c++) {
        /* magnitude depends only on (a, c); powers of two are exact in E4M3
         * once scaled, so a correct round trip is bit-exact here. */
        input[a * kExtent * kOuterC + j * kOuterC + c] = std::ldexp(1.0f, static_cast<int>(a * kOuterC + c));
      }
    }
  }

  ReshardQuantGeometry geometry{};
  geometry.outerA = kOuterA;
  geometry.blockExtent = kExtent;
  geometry.outerC = kOuterC;
  geometry.blocksPerRow = kBlocksPerRow;
  geometry.blockSize = kBlockSize;

  void* devSrc = nullptr;
  void* devDst = nullptr;
  void* devQuant = nullptr;
  float* devScales = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devSrc, kElements * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devDst, kElements * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devQuant, kElements));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devScales, kTotalBlocks * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMemset(devDst, 0, kElements * 4));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(devSrc, input.data(), kElements * 4, cudaMemcpyHostToDevice));

  ASSERT_EQ(ncclSuccess,
            reshardLaunchQuantize(devSrc, ncclFloat32, devQuant, devScales, ncclFloat32, geometry, /*roundScales=*/true, nullptr));
  ASSERT_EQ(ncclSuccess, reshardLaunchDequantize(devQuant, devScales, ncclFloat32, devDst, ncclFloat32, geometry, nullptr));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<float> output(kElements, -1.0f);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), devDst, kElements * 4, cudaMemcpyDeviceToHost));

  /* Every element must be visited exactly once and reconstructed exactly. A
   * stride error would leave some elements at 0 (never written) or mix
   * magnitudes between (a, c) lanes. */
  for (size_t i = 0; i < kElements; i++) {
    EXPECT_FLOAT_EQ(input[i], output[i]) << "index " << i;
  }

  cudaFree(devSrc);
  cudaFree(devDst);
  cudaFree(devQuant);
  cudaFree(devScales);
}

/* ncclReshardQuantized validates its config before touching the communicator,
 * so the reject paths are exercisable without one. */
namespace {

struct QuantApiFixture {
  ncclMesh_t srcMesh{};
  ncclMesh_t dstMesh{};
  ncclDistTensor_t src{};
  ncclDistTensor_t dst{};
  ncclReshardQuantConfig_t quant = NCCL_M2N_QUANT_CONFIG_INITIALIZER;
  int payload = 0;

  QuantApiFixture() {
    srcMesh.dims[0] = 1;
    srcMesh.dims[1] = 2;
    srcMesh.startRank = 0;
    dstMesh = srcMesh;

    src.dataPtr = &payload;
    src.localShape[0] = 16;
    src.localShape[1] = 256;
    src.ndims = 2;
    src.dtype = ncclBfloat16;
    src.mesh = &srcMesh;
    src.placements[0] = NCCL_RESHARD_REPLICATE;
    src.placements[1] = NCCL_RESHARD_SHARD(0);
    dst = src;
    dst.mesh = &dstMesh;

    quant.recipe = NCCL_M2N_QUANT_FP8E4M3;
    quant.blockDim = 1;
    quant.blockSize = 128;
  }

  ncclResult_t call() { return ncclReshardQuantized(nullptr, nullptr, &src, &dst, &quant, nullptr); }
};

} // namespace

TEST(QuantizedApiTest, RejectsMalformedConfig) {
  QuantApiFixture badSize;
  badSize.quant.size = sizeof(ncclReshardQuantConfig_t) - 1;
  EXPECT_EQ(ncclInvalidArgument, badSize.call());

  QuantApiFixture badMagic;
  badMagic.quant.magic = 0xdeadbeefu;
  EXPECT_EQ(ncclInvalidArgument, badMagic.call());
}

TEST(QuantizedApiTest, RejectsUnknownRecipe) {
  QuantApiFixture f;
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  f.quant.recipe = static_cast<ncclM2nQuantRecipe_t>(9);
  EXPECT_EQ(ncclInvalidArgument, f.call());
}

TEST(QuantizedApiTest, NoneRequiresEveryOtherFieldCleared) {
  QuantApiFixture f;
  f.quant.recipe = NCCL_M2N_QUANT_NONE; /* blockDim/blockSize still set */
  EXPECT_EQ(ncclInvalidArgument, f.call());
}

TEST(QuantizedApiTest, RejectsIntegerPayloadDtypes) {
  for (ncclDataType_t dtype : {ncclInt8, ncclUint8, ncclInt32, ncclInt64, ncclFloat64}) {
    QuantApiFixture f;
    f.src.dtype = dtype;
    f.dst.dtype = dtype;
    EXPECT_EQ(ncclInvalidArgument, f.call()) << "dtype " << static_cast<int>(dtype);
  }
}

TEST(QuantizedApiTest, RejectsMismatchedPayloadDtypes) {
  QuantApiFixture f;
  f.dst.dtype = ncclFloat32;
  EXPECT_EQ(ncclInvalidArgument, f.call());
}

TEST(QuantizedApiTest, RejectsBadBlockGeometry) {
  QuantApiFixture badDim;
  badDim.quant.blockDim = 5;
  EXPECT_EQ(ncclInvalidArgument, badDim.call());

  QuantApiFixture zeroSize;
  zeroSize.quant.blockSize = 0;
  EXPECT_EQ(ncclInvalidArgument, zeroSize.call());

  /* 256 is not a multiple of 96: a shard boundary would fall inside a block. */
  QuantApiFixture indivisible;
  indivisible.quant.blockSize = 96;
  EXPECT_EQ(ncclInvalidArgument, indivisible.call());
}

TEST(QuantizeTest, E8m0ScalesRoundTripExactly) {
  QUANT_REQUIRE_CUDA();

  /* MX shared scales are a single exponent byte, so the scale must survive as
   * an exact power of two.  Inputs are powers of two, making a correct round
   * trip bit-exact. */
  constexpr size_t kBlockSize = 32;
  constexpr size_t kBlocks = 4;
  constexpr size_t kElements = kBlockSize * kBlocks;

  std::vector<float> input(kElements);
  for (size_t b = 0; b < kBlocks; b++) {
    for (size_t i = 0; i < kBlockSize; i++) {
      /* Different magnitude per block so each gets a distinct exponent. */
      input[b * kBlockSize + i] = std::ldexp(1.0f, static_cast<int>(b) * 6);
    }
  }

  ReshardQuantGeometry geometry{};
  geometry.outerA = 1;
  geometry.blockExtent = kElements;
  geometry.outerC = 1;
  geometry.blocksPerRow = kBlocks;
  geometry.blockSize = kBlockSize;

  void* devSrc = nullptr;
  void* devDst = nullptr;
  void* devQuant = nullptr;
  void* devScales = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devSrc, kElements * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devDst, kElements * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devQuant, kElements));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devScales, kBlocks)); /* one byte per block */
  ASSERT_EQ(cudaSuccess, cudaMemcpy(devSrc, input.data(), kElements * 4, cudaMemcpyHostToDevice));

  /* roundScales deliberately false: E8M0 must force rounding on its own,
   * otherwise the stored exponent would silently discard a mantissa. */
  ASSERT_EQ(ncclSuccess, reshardLaunchQuantize(devSrc, ncclFloat32, devQuant, devScales, ncclUint8, geometry,
                                               /*roundScales=*/false, nullptr));
  ASSERT_EQ(ncclSuccess,
            reshardLaunchDequantize(devQuant, devScales, ncclUint8, devDst, ncclFloat32, geometry, nullptr));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<float> output(kElements, -1.0f);
  std::vector<uint8_t> scaleBytes(kBlocks, 0);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), devDst, kElements * 4, cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(scaleBytes.data(), devScales, kBlocks, cudaMemcpyDeviceToHost));

  for (size_t b = 0; b < kBlocks; b++) {
    /* A valid E8M0 byte decodes to a normal power of two. */
    const uint32_t bits = static_cast<uint32_t>(scaleBytes[b]) << 23;
    float decoded = 0.0f;
    std::memcpy(&decoded, &bits, 4);
    int exponent = 0;
    EXPECT_FLOAT_EQ(0.5f, std::frexp(decoded, &exponent)) << "block " << b << " scale is not a power of two";
  }
  for (size_t i = 0; i < kElements; i++) {
    EXPECT_FLOAT_EQ(input[i], output[i]) << "index " << i;
  }

  cudaFree(devSrc);
  cudaFree(devDst);
  cudaFree(devQuant);
  cudaFree(devScales);
}

TEST(QuantizedApiTest, RejectsMxfp8WithoutRoundScalesOrWrongBlockSize) {
  QuantApiFixture noRound;
  noRound.quant.recipe = NCCL_M2N_QUANT_MXFP8;
  noRound.quant.blockSize = 32;
  noRound.quant.roundScales = 0;
  EXPECT_EQ(ncclInvalidArgument, noRound.call()) << "E8M0 cannot represent a non-power-of-two scale";

  QuantApiFixture wrongBlock;
  wrongBlock.quant.recipe = NCCL_M2N_QUANT_MXFP8;
  wrongBlock.quant.blockSize = 128;
  wrongBlock.quant.roundScales = 1;
  EXPECT_EQ(ncclInvalidArgument, wrongBlock.call()) << "MX defines a 32-element shared scale";
}

TEST(QuantizedApiTest, RejectsDestinationDtypeThatSelectsNoMode) {
  QuantApiFixture f;
  f.dst.dtype = ncclFloat32; /* neither src->dtype (bf16) nor the wire dtype */
  EXPECT_EQ(ncclInvalidArgument, f.call());
}

TEST(QuantizedApiTest, KeepQuantizedRequiresDstScales) {
  QuantApiFixture f;
  f.dst.dtype = ncclFloat8e4m3; /* keep quantized */
  f.quant.dstScales = nullptr;
  EXPECT_EQ(ncclInvalidArgument, f.call()) << "an active destination must supply a scale output buffer";
}

TEST(QuantizedApiTest, DequantizeRejectsStrayDstScales) {
  int scratch = 0;
  QuantApiFixture f; /* dst->dtype == src->dtype, so dequantize */
  f.quant.dstScales = &scratch;
  EXPECT_EQ(ncclInvalidArgument, f.call()) << "scales are internal when dequantizing";
}
