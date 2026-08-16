/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * scale_plane_test.cc — coupled (payload, scales) descriptor validation.
 *
 * Covered:
 *   reshardScalePlaneActive   — NULL and NONE both mean "no scale pass"
 *   validateReshardScalePlane — ABI guard, closed recipe set, dtype set,
 *                               block geometry, buffer coupling, per-side
 *                               shape consistency, cross-side global
 *                               agreement, and the shard-boundary rule
 *   buildReshardScaleTensors  — topology inherited, scale fields overridden
 *
 * The shard-boundary case is the one that motivates the feature: resharding a
 * payload and its scales as two independent calls cannot detect it.
 ************************************************************************/

#include <gtest/gtest.h>

#include <string>

#include "nccl_m2n.h"
#include "reshard_internal.h"
#include "reshard_limits.h"

namespace {

constexpr size_t kTokens = 256;
constexpr size_t kHidden = 512;
constexpr size_t kBlockSize = 128;

/* A 2-D [tokens, hidden] payload sharded on dim 0 across a 1x4 mesh, with a
 * companion scale plane blocked along the hidden dimension. */
struct ScaleFixture {
  ncclMesh_t srcMesh{};
  ncclMesh_t dstMesh{};
  ncclDistTensor_t src{};
  ncclDistTensor_t dst{};
  ncclReshardScalePlane_t scales = NCCL_M2N_SCALE_PLANE_INITIALIZER;

  /* Storage the descriptors point at; contents are never dereferenced by the
   * validator, only compared against NULL. */
  int payloadBuf = 0;
  int scaleBuf = 0;

  ScaleFixture(int srcShards = 4, int dstShards = 2, int shardTensorDim = 0, int blockDim = 1,
               size_t blockSize = kBlockSize) {
    srcMesh.dims[0] = 1;
    srcMesh.dims[1] = srcShards;
    srcMesh.startRank = 0;
    dstMesh.dims[0] = 1;
    dstMesh.dims[1] = dstShards;
    dstMesh.startRank = 0;

    src.dataPtr = &payloadBuf;
    src.ndims = 2;
    src.dtype = ncclFloat8e4m3;
    src.mesh = &srcMesh;
    src.placements[0] = NCCL_RESHARD_REPLICATE;
    src.placements[1] = NCCL_RESHARD_SHARD(shardTensorDim);
    src.localShape[0] = kTokens;
    src.localShape[1] = kHidden;

    dst = src;
    dst.mesh = &dstMesh;
    dst.dataPtr = &payloadBuf;

    /* Shard dim is divided per side; every other dim is the full extent. */
    src.localShape[shardTensorDim] /= static_cast<size_t>(srcShards);
    dst.localShape[shardTensorDim] /= static_cast<size_t>(dstShards);

    scales.recipe = NCCL_M2N_SCALE_FWD;
    scales.srcDataPtr = &scaleBuf;
    scales.dstDataPtr = &scaleBuf;
    scales.dtype = ncclFloat32;
    scales.blockDim = blockDim;
    scales.blockSize = blockSize;
    for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
      scales.srcLocalShape[d] = src.localShape[d];
      scales.dstLocalShape[d] = dst.localShape[d];
    }
    scales.srcLocalShape[blockDim] /= blockSize;
    scales.dstLocalShape[blockDim] /= blockSize;
  }

  ncclResult_t validate() { return validateReshardScalePlane("unit", &src, &dst, &scales); }
};

} // namespace

TEST(ScalePlaneActive, NullAndNoneAreInactive) {
  EXPECT_FALSE(reshardScalePlaneActive(nullptr));
  ncclReshardScalePlane_t plane = NCCL_M2N_SCALE_PLANE_INITIALIZER;
  EXPECT_FALSE(reshardScalePlaneActive(&plane));
  plane.recipe = NCCL_M2N_SCALE_FWD;
  EXPECT_TRUE(reshardScalePlaneActive(&plane));
}

TEST(ScalePlaneValidate, AcceptsWellFormedPlane) {
  ScaleFixture f;
  EXPECT_EQ(ncclSuccess, f.validate());
}

TEST(ScalePlaneValidate, AcceptsEveryDocumentedScaleDtype) {
  const ncclDataType_t allowed[] = {ncclFloat32,     ncclFloat16,     ncclBfloat16,
                                    ncclFloat8e4m3, ncclFloat8e5m2, ncclUint8};
  for (ncclDataType_t dtype : allowed) {
    ScaleFixture f;
    f.scales.dtype = dtype;
    EXPECT_EQ(ncclSuccess, f.validate()) << "dtype " << static_cast<int>(dtype);
  }
}

TEST(ScalePlaneValidate, RejectsPayloadOnlyDtypes) {
  /* getNcclDtSize accepts these, but a scale plane holding int64 or float64
   * scaling factors is far more likely a caller mistake than an intent. */
  const ncclDataType_t rejected[] = {ncclInt8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64};
  for (ncclDataType_t dtype : rejected) {
    ScaleFixture f;
    f.scales.dtype = dtype;
    EXPECT_EQ(ncclInvalidArgument, f.validate()) << "dtype " << static_cast<int>(dtype);
  }
}

TEST(ScalePlaneValidate, RejectsMalformedAbiGuard) {
  ScaleFixture badSize;
  badSize.scales.size = sizeof(ncclReshardScalePlane_t) - 1;
  EXPECT_EQ(ncclInvalidArgument, badSize.validate());

  ScaleFixture badMagic;
  badMagic.scales.magic = 0xdeadbeefu;
  EXPECT_EQ(ncclInvalidArgument, badMagic.validate());
}

TEST(ScalePlaneValidate, RejectsUnknownRecipe) {
  ScaleFixture f;
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  f.scales.recipe = static_cast<ncclM2nScaleRecipe_t>(7);
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

TEST(ScalePlaneValidate, NoneRequiresEveryFieldCleared) {
  ScaleFixture f;
  f.scales.recipe = NCCL_M2N_SCALE_NONE;
  EXPECT_EQ(ncclInvalidArgument, f.validate()) << "NONE must reject leftover scale state";

  ncclReshardScalePlane_t clean = NCCL_M2N_SCALE_PLANE_INITIALIZER;
  EXPECT_EQ(ncclSuccess, validateReshardScalePlane("unit", &f.src, &f.dst, &clean));
}

TEST(ScalePlaneValidate, RejectsBlockDimOutOfRange) {
  for (int blockDim : {-1, 2, 99}) {
    ScaleFixture f;
    f.scales.blockDim = blockDim;
    EXPECT_EQ(ncclInvalidArgument, f.validate()) << "blockDim " << blockDim;
  }
}

TEST(ScalePlaneValidate, RejectsZeroBlockSize) {
  ScaleFixture f;
  f.scales.blockSize = 0;
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

TEST(ScalePlaneValidate, RejectsOneSidedBuffers) {
  ScaleFixture missingSrc;
  missingSrc.scales.srcDataPtr = nullptr;
  EXPECT_EQ(ncclInvalidArgument, missingSrc.validate());

  ScaleFixture missingDst;
  missingDst.scales.dstDataPtr = nullptr;
  EXPECT_EQ(ncclInvalidArgument, missingDst.validate());

  /* An inactive side must supply neither plane. */
  ScaleFixture inactive;
  inactive.src.dataPtr = nullptr;
  inactive.scales.srcDataPtr = nullptr;
  EXPECT_EQ(ncclSuccess, inactive.validate());
}

TEST(ScalePlaneValidate, RejectsNonBlockDimShapeMismatch) {
  ScaleFixture f;
  f.scales.srcLocalShape[0] += 1;
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

TEST(ScalePlaneValidate, RejectsScaleExtentNotCoveringPayload) {
  ScaleFixture f;
  f.scales.srcLocalShape[1] += 1;
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

TEST(ScalePlaneValidate, RejectsBlockSizeNotDividingPayload) {
  /* hidden=512 is not a multiple of 384, so no integral scale extent covers it. */
  ScaleFixture f(4, 2, 0, 1, 384);
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

TEST(ScalePlaneValidate, RejectsPerSideBlockSizeDisagreement) {
  /* Right blockSize for src, wrong one for dst.  Caught by dst's own extent
   * check rather than the cross-side comparison, since one blockSize cannot
   * satisfy both sides at once. */
  ScaleFixture f;
  f.scales.dstLocalShape[1] = f.dst.localShape[1] / (kBlockSize / 2);
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

TEST(ScalePlaneValidate, RejectsDisagreeingGlobalScaleShapes) {
  /* Each side is internally consistent, but they describe different global
   * payload tensors, so their derived global scale shapes disagree.  This is
   * what the cross-side check exists for; the reshard path would otherwise
   * report it later without mentioning blockSize. */
  ScaleFixture f;
  f.dst.localShape[1] = kHidden * 2;
  f.scales.dstLocalShape[1] = f.dst.localShape[1] / kBlockSize;
  EXPECT_EQ(ncclInvalidArgument, f.validate());
}

/* The motivating case: the shard dimension IS the block dimension. */
TEST(ScalePlaneValidate, AcceptsBlockAlignedShardOfBlockDim) {
  /* hidden=512 sharded 4 ways gives 128 per shard, exactly one block. */
  ScaleFixture f(4, 2, /*shardTensorDim=*/1, /*blockDim=*/1, kBlockSize);
  EXPECT_EQ(ncclSuccess, f.validate());
}

TEST(ScalePlaneValidate, RejectsShardOfBlockDimInteriorToABlock) {
  /* hidden=512 sharded 8 ways gives 64 per shard — half a 128-element block,
   * so every shard boundary falls inside a scale block.  Submitting the two
   * planes as independent reshards would silently mis-split here. */
  ScaleFixture f(8, 2, /*shardTensorDim=*/1, /*blockDim=*/1, kBlockSize);
  EXPECT_EQ(ncclInvalidArgument, f.validate());

  /* Assert the actionable diagnostic, not just the error code.  The generic
   * extent check rejects the same input, so an ordering regression would keep
   * this test green while losing the message that explains the failure. */
  const std::string detail = ncclM2nGetLastError();
  EXPECT_NE(std::string::npos, detail.find("scale block")) << detail;
  EXPECT_NE(std::string::npos, detail.find("shards block dim")) << detail;
}

TEST(ScalePlaneValidate, RejectsNullArguments) {
  ScaleFixture f;
  EXPECT_EQ(ncclInvalidArgument, validateReshardScalePlane("unit", nullptr, &f.dst, &f.scales));
  EXPECT_EQ(ncclInvalidArgument, validateReshardScalePlane("unit", &f.src, nullptr, &f.scales));
  EXPECT_EQ(ncclInvalidArgument, validateReshardScalePlane("unit", &f.src, &f.dst, nullptr));
}

TEST(ScalePlaneBuild, InheritsTopologyAndOverridesScaleFields) {
  ScaleFixture f;
  ncclDistTensor_t srcScale{};
  ncclDistTensor_t dstScale{};
  ASSERT_EQ(ncclSuccess, buildReshardScaleTensors(&f.src, &f.dst, &f.scales, &srcScale, &dstScale));

  /* Topology must be inherited verbatim — that shared topology is what lets
   * both planes bin together for fusion. */
  EXPECT_EQ(f.src.mesh, srcScale.mesh);
  EXPECT_EQ(f.dst.mesh, dstScale.mesh);
  EXPECT_EQ(f.src.ndims, srcScale.ndims);
  EXPECT_EQ(f.src.placements[0], srcScale.placements[0]);
  EXPECT_EQ(f.src.placements[1], srcScale.placements[1]);

  /* Scale-owned fields must be overridden. */
  EXPECT_EQ(&f.scaleBuf, srcScale.dataPtr);
  EXPECT_EQ(&f.scaleBuf, dstScale.dataPtr);
  EXPECT_EQ(ncclFloat32, srcScale.dtype);
  EXPECT_EQ(kHidden / kBlockSize, srcScale.localShape[1]);
  EXPECT_EQ(f.src.localShape[0], srcScale.localShape[0]);
}

TEST(ScalePlaneBuild, RejectsNullArguments) {
  ScaleFixture f;
  ncclDistTensor_t out{};
  EXPECT_EQ(ncclInvalidArgument, buildReshardScaleTensors(nullptr, &f.dst, &f.scales, &out, &out));
  EXPECT_EQ(ncclInvalidArgument, buildReshardScaleTensors(&f.src, &f.dst, &f.scales, nullptr, &out));
  EXPECT_EQ(ncclInvalidArgument, buildReshardScaleTensors(&f.src, &f.dst, nullptr, &out, &out));
}
