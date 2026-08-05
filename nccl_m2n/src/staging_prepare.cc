/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_prepare.cc — host-side descriptor builders for ncclReshard.
 *
 * Builds the per-rank staging transfer descriptors, delegating mesh and
 * load-balance analysis to the reshard_mesh.cc + reshard_loadbalance.cc
 * helpers.
 *
 * The initial implementation uses a pure RDMA all-to-all descriptor.
 ************************************************************************/

#include "staging_types.h"
#include "staging_buffer.h"

#include "reshard_internal.h"
#include "reshard_types.h"
#include "m2n_log.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"

#include <cstdio>
#include <algorithm>
#include <cstring>
#include "nccl.h"

/* ======================================================================
 * Helper: copy a window/staging plan from ncclReshardTransferPlan into the
 * staging path's StagingTransferPlan layout.
 * ====================================================================*/
static ncclResult_t fillStagingPlan(StagingTransferPlan* sp, const ncclReshardTransferPlan& plan, int ndims) {
  sp->numOuterLoops = plan.numOuterLoops;
  sp->srcBaseOffset = plan.srcBaseOffset;
  sp->dstBaseOffset = plan.dstBaseOffset;
  sp->innerSize = plan.innerSize;
  sp->totalInnerTransfers = plan.totalInnerTransfers;
  sp->isContiguous = (plan.totalInnerTransfers == 1);
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(plan.totalInnerTransfers, plan.innerSize, &sp->totalBytes), -1,
                     "[STAGING] plan totalBytes overflow: transfers=%zu innerSize=%zu", plan.totalInnerTransfers,
                     plan.innerSize);
  for (int d = 0; d < ndims; d++) {
    sp->outerCounts[d] = plan.outerCounts[d];
    sp->outerSrcStrides[d] = plan.outerSrcStrides[d];
    sp->outerDstStrides[d] = plan.outerDstStrides[d];
  }
  return ncclSuccess;
}

/* Helper to derive both meshes' local dims/strides from the half the
 * caller actually provided, matching prepareReshardParams's behaviour
 * for asymmetric ranks. */
static ncclResult_t resolveLocalDims(const size_t* srcTensorDims, const size_t* dstTensorDims, int ndims,
                                     int srcShardDim, int srcShardCount, int dstShardDim, int dstShardCount,
                                     bool isSource, bool isDest, size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                     size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                     size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                     size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  if (isSource) {
    for (int d = 0; d < ndims; d++) {
      srcDims[d] = srcTensorDims[d];
    }
    NCCL_M2N_CHECK(computeStridesChecked(srcDims, ndims, srcStrides));
  }
  if (isDest) {
    for (int d = 0; d < ndims; d++) {
      dstDims[d] = dstTensorDims[d];
    }
    NCCL_M2N_CHECK(computeStridesChecked(dstDims, ndims, dstStrides));
  }
  if (isSource && !isDest) {
    for (int d = 0; d < ndims; d++) {
      size_t globalSize = srcTensorDims[d];
      if (d == srcShardDim) {
        NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcTensorDims[d], (size_t)srcShardCount, &globalSize), -1,
                           "[STAGING] source global dim overflow at dim %d: local=%zu shardCount=%d", d,
                           srcTensorDims[d], srcShardCount);
      }
      dstDims[d] = (d == dstShardDim) ? globalSize / dstShardCount : globalSize;
    }
    NCCL_M2N_CHECK(computeStridesChecked(dstDims, ndims, dstStrides));
  }
  if (isDest && !isSource) {
    for (int d = 0; d < ndims; d++) {
      size_t globalSize = dstTensorDims[d];
      if (d == dstShardDim) {
        NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dstTensorDims[d], (size_t)dstShardCount, &globalSize), -1,
                           "[STAGING] destination global dim overflow at dim %d: local=%zu shardCount=%d", d,
                           dstTensorDims[d], dstShardCount);
      }
      srcDims[d] = (d == srcShardDim) ? globalSize / srcShardCount : globalSize;
    }
    NCCL_M2N_CHECK(computeStridesChecked(srcDims, ndims, srcStrides));
  }
  return ncclSuccess;
}

static ncclResult_t deriveGlobalDimsFromLocal(int worldRank, const char* side, const size_t* localDims, int ndims,
                                              int shardTensorDim, int shardCount,
                                              size_t globalDims[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  NCCL_M2N_CHECK_ARG(localDims != nullptr, worldRank,
                     "validateStagingPlanLimits: %s local shape array must be non-null", side);
  NCCL_M2N_CHECK_ARG(shardCount > 0, worldRank, "validateStagingPlanLimits: %s shard count must be positive", side);
  for (int d = 0; d < ndims; d++) {
    NCCL_M2N_CHECK_ARG(localDims[d] > 0, worldRank,
                       "validateStagingPlanLimits: %s localShape[%d]=%zu must be positive, including on inactive "
                       "ranks",
                       side, d, localDims[d]);
    if (d == shardTensorDim) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(localDims[d], static_cast<size_t>(shardCount), &globalDims[d]), worldRank,
                         "validateStagingPlanLimits: %s localShape[%d]=%zu overflows global shape with shard count %d",
                         side, d, localDims[d], shardCount);
    } else {
      globalDims[d] = localDims[d];
    }
  }
  return ncclSuccess;
}

static ncclResult_t compareGlobalDims(int worldRank, const size_t srcGlobalDims[], const size_t dstGlobalDims[],
                                      int ndims) {
  for (int d = 0; d < ndims; d++) {
    NCCL_M2N_CHECK_ARG(srcGlobalDims[d] == dstGlobalDims[d], worldRank,
                       "validateStagingPlanLimits: src/dst global shape mismatch at dim %d (%zu != %zu)", d,
                       srcGlobalDims[d], dstGlobalDims[d]);
  }
  return ncclSuccess;
}

static ncclResult_t deriveLocalDimsFromGlobal(int worldRank, const char* side, const size_t globalDims[], int ndims,
                                              int shardTensorDim, int shardCount,
                                              size_t localDims[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  NCCL_M2N_CHECK_ARG(shardCount > 0, worldRank, "validateStagingPlanLimits: %s shard count must be positive", side);
  for (int d = 0; d < ndims; d++) {
    if (d == shardTensorDim) {
      NCCL_M2N_CHECK_ARG(globalDims[d] % static_cast<size_t>(shardCount) == 0, worldRank,
                         "validateStagingPlanLimits: %s global shape dim %d (%zu) is not divisible by shard count %d",
                         side, d, globalDims[d], shardCount);
      localDims[d] = globalDims[d] / static_cast<size_t>(shardCount);
    } else {
      localDims[d] = globalDims[d];
    }
  }
  return ncclSuccess;
}

static ncclResult_t resolvePreflightDims(int worldRank, const size_t* srcTensorDims, const size_t* dstTensorDims,
                                         int ndims, int srcShardDim, int srcShardCount, int dstShardDim,
                                         int dstShardCount, size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                         size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                         size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                         size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  size_t srcGlobalDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstGlobalDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t globalDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};

  NCCL_M2N_CHECK(deriveGlobalDimsFromLocal(worldRank, "src", srcTensorDims, ndims, srcShardDim, srcShardCount,
                                           srcGlobalDims));
  NCCL_M2N_CHECK(deriveGlobalDimsFromLocal(worldRank, "dst", dstTensorDims, ndims, dstShardDim, dstShardCount,
                                           dstGlobalDims));
  NCCL_M2N_CHECK(compareGlobalDims(worldRank, srcGlobalDims, dstGlobalDims, ndims));

  for (int d = 0; d < ndims; d++) {
    globalDims[d] = srcGlobalDims[d];
  }

  NCCL_M2N_CHECK(deriveLocalDimsFromGlobal(worldRank, "src", globalDims, ndims, srcShardDim, srcShardCount, srcDims));
  NCCL_M2N_CHECK(deriveLocalDimsFromGlobal(worldRank, "dst", globalDims, ndims, dstShardDim, dstShardCount, dstDims));
  NCCL_M2N_CHECK(computeStridesChecked(srcDims, ndims, srcStrides));
  NCCL_M2N_CHECK(computeStridesChecked(dstDims, ndims, dstStrides));
  return ncclSuccess;
}

static int countOverlappingDstShards(const size_t srcDims[], const size_t srcStrides[], int srcShardDim,
                                     int srcShardIdx, const size_t dstDims[], const size_t dstStrides[],
                                     int dstShardDim, int dstShardCount, int ndims) {
  int count = 0;
  for (int dstShard = 0; dstShard < dstShardCount; dstShard++) {
    ncclReshardTransferPlan plan;
    computeTransferPlan(srcDims, srcStrides, srcShardDim, srcShardIdx, dstDims, dstStrides, dstShardDim, dstShard,
                        ndims, /*elementsPerChunk=*/1, &plan);
    if (plan.totalInnerTransfers > 0) {
      count++;
    }
  }
  return count;
}

static int countOverlappingSrcShards(const size_t srcDims[], const size_t srcStrides[], int srcShardDim,
                                     int srcShardCount, const size_t dstDims[], const size_t dstStrides[],
                                     int dstShardDim, int dstShardIdx, int ndims) {
  int count = 0;
  for (int srcShard = 0; srcShard < srcShardCount; srcShard++) {
    ncclReshardTransferPlan plan;
    computeTransferPlan(srcDims, srcStrides, srcShardDim, srcShard, dstDims, dstStrides, dstShardDim, dstShardIdx,
                        ndims, /*elementsPerChunk=*/1, &plan);
    if (plan.totalInnerTransfers > 0) {
      count++;
    }
  }
  return count;
}

static ncclResult_t validateRankStagingCounts(
  int worldRank, int rank, const ncclDistTensor_t* srcTensor, const ncclReshardMeshGroupInfo* fullSrcInfo,
  const ncclDistTensor_t* dstTensor, const ncclReshardMeshGroupInfo* fullDstInfo, const size_t srcDims[],
  const size_t srcStrides[], const size_t dstDims[], const size_t dstStrides[], int ndims,
  const ncclReshardRepLoadBalancer* lb, ReshardCopyAlgorithm copyAlgo, int gpusPerDomain, size_t* maxPeerGroupSize) {
  const bool isSource = reshardRankInMesh(srcTensor->mesh, rank);
  const bool isDest = reshardRankInMesh(dstTensor->mesh, rank);
  if (!isSource && !isDest) {
    return ncclSuccess;
  }

  size_t numTargets = 0;
  int numSources = 0;

  auto addTargets = [&](size_t count) -> ncclResult_t {
    size_t updated = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(numTargets, count, &updated), worldRank,
                       "validateStagingPlanLimits: target count overflows size_t for rank %d", rank);
    numTargets = updated;
    return ncclSuccess;
  };

  if (isSource) {
    ncclReshardMeshGroupInfo srcInfo;
    computeMeshGroupInfo(srcTensor, rank, &srcInfo);
    int targetRepStart = 0;
    int targetRepEnd = 0;
    getTargetRepRange(lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);
    const int targetRepCount = targetRepEnd - targetRepStart;
    if (targetRepCount > 0) {
      const int overlapDstCount =
        countOverlappingDstShards(srcDims, srcStrides, fullSrcInfo->shardTensorDim, srcInfo.shardIdx, dstDims,
                                  dstStrides, fullDstInfo->shardTensorDim, fullDstInfo->shardCount, ndims);
      if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
        size_t directTargets = 0;
        NCCL_M2N_CHECK_ARG(
          m2nCheckedMulSize(static_cast<size_t>(overlapDstCount), static_cast<size_t>(targetRepCount), &directTargets),
          worldRank, "validateStagingPlanLimits: direct target count overflows size_t for rank %d", rank);
        NCCL_M2N_CHECK(addTargets(directTargets));
      } else {
        NCCL_M2N_CHECK(addTargets(static_cast<size_t>(overlapDstCount)));
      }
    }
  }

  if (isDest) {
    ncclReshardMeshGroupInfo dstInfo;
    computeMeshGroupInfo(dstTensor, rank, &dstInfo);
    numSources = countOverlappingSrcShards(srcDims, srcStrides, fullSrcInfo->shardTensorDim, fullSrcInfo->shardCount,
                                           dstDims, dstStrides, fullDstInfo->shardTensorDim, dstInfo.shardIdx, ndims);

    if (copyAlgo != RESHARD_COPY_ALGO_DIRECT && numSources > 0) {
      int sourceRep = getSourceRepForDest(lb, dstInfo.repIdx);
      int targetRepStart = 0;
      int targetRepEnd = 0;
      getTargetRepRange(lb, sourceRep, &targetRepStart, &targetRepEnd);

      int numLocalReps = 0;
      int myLocalRepIdx = -1;
      const int myNode = rank / gpusPerDomain;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
        if (repRank / gpusPerDomain != myNode) continue;
        if (rep == dstInfo.repIdx) myLocalRepIdx = numLocalReps;
        numLocalReps++;
      }

      NCCL_M2N_CHECK_ARG(numLocalReps <= MAX_LOCAL_FOLLOWERS + 1, worldRank,
                         "validateStagingPlanLimits: local destination fan-out exceeds capacity at rank %d, "
                         "dstShard %d, domain %d (localRepCount=%d > MAX_LOCAL_FOLLOWERS+1=%d)",
                         rank, dstInfo.shardIdx, myNode, numLocalReps, MAX_LOCAL_FOLLOWERS + 1);
      NCCL_M2N_CHECK_ARG(myLocalRepIdx >= 0, worldRank,
                         "validateStagingPlanLimits: destination rank %d has no local handler in target rep range "
                         "[%d, %d)",
                         rank, targetRepStart, targetRepEnd);

      int firstRepNodeDest = -1;
      int firstNodeLocalReps = 0;
      if (targetRepStart < targetRepEnd) {
        firstRepNodeDest = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, targetRepStart) / gpusPerDomain;
        for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
          int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
          if (repRank / gpusPerDomain == firstRepNodeDest) firstNodeLocalReps++;
        }
      }

      int sourceRepSlots = firstNodeLocalReps > 0 ? firstNodeLocalReps : numLocalReps;
      int activeSourceSlots = sourceRepSlots;
      for (int node = firstRepNodeDest; node >= 0 && node <= myNode; node++) {
        int nodeLocalReps = 0;
        for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
          int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
          if (repRank / gpusPerDomain == node) nodeLocalReps++;
        }
        if (nodeLocalReps > 0 && nodeLocalReps < activeSourceSlots) activeSourceSlots = nodeLocalReps;
      }

      int mySourceStart = 0;
      int mySourceEnd = 0;
      int mySourceSlotStart = myLocalRepIdx;
      int mySourceSlotEnd = myLocalRepIdx + 1;
      if (activeSourceSlots > 0 && myLocalRepIdx == activeSourceSlots - 1 && sourceRepSlots > activeSourceSlots) {
        mySourceSlotEnd = sourceRepSlots;
      }
      if (sourceRepSlots > 0 && mySourceSlotStart < activeSourceSlots) {
        int sourcesPerRep = numSources / sourceRepSlots;
        int extra = numSources % sourceRepSlots;
        int threshold = extra * (sourcesPerRep + 1);
        mySourceStart = (mySourceSlotStart < extra) ? mySourceSlotStart * (sourcesPerRep + 1)
                                                   : threshold + (mySourceSlotStart - extra) * sourcesPerRep;
        if (mySourceSlotEnd >= sourceRepSlots) mySourceEnd = numSources;
        else if (mySourceSlotEnd < extra) mySourceEnd = mySourceSlotEnd * (sourcesPerRep + 1);
        else mySourceEnd = threshold + (mySourceSlotEnd - extra) * sourcesPerRep;
      }

      bool hasRingNext = false;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
        if (repRank / gpusPerDomain > myNode) {
          hasRingNext = true;
          break;
        }
      }

      const int handledSources = mySourceEnd - mySourceStart;
      size_t localTargets = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(static_cast<size_t>(handledSources),
                                           static_cast<size_t>(numLocalReps - 1), &localTargets),
                         worldRank, "validateStagingPlanLimits: local target count overflows size_t for rank %d", rank);
      NCCL_M2N_CHECK(addTargets(localTargets));
      if (hasRingNext) NCCL_M2N_CHECK(addTargets(static_cast<size_t>(handledSources)));
    }
  }

  if (numTargets > static_cast<size_t>(MAX_TARGETS)) {
    NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                  "validateStagingPlanLimits: target list would exceed capacity for rank %d "
                  "(targets=%zu > MAX_TARGETS=%d). Fix: increase MAX_TARGETS in reshard_limits.h.",
                  rank, numTargets, MAX_TARGETS);
  }
  if (numSources > MAX_SOURCES) {
    NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                  "validateStagingPlanLimits: source list would exceed capacity for rank %d "
                  "(sources=%d > MAX_SOURCES=%d). Fix: increase MAX_SOURCES in reshard_limits.h.",
                  rank, numSources, MAX_SOURCES);
  }
  if (maxPeerGroupSize != nullptr) {
    *maxPeerGroupSize = std::max(*maxPeerGroupSize, std::max(numTargets, static_cast<size_t>(numSources)));
  }
  return ncclSuccess;
}

ncclResult_t validateStagingPlanLimits(int worldRank, const ncclDistTensor_t* srcTensor, const size_t* srcTensorDims,
                                       const ncclDistTensor_t* dstTensor, const size_t* dstTensorDims,
                                       ReshardCopyAlgorithm copyAlgo, int gpusPerDomain, size_t* maxPeerGroupSize) {
  NCCL_M2N_CHECK_ARG(srcTensor != nullptr, worldRank, "validateStagingPlanLimits: src tensor must be non-null");
  NCCL_M2N_CHECK_ARG(dstTensor != nullptr, worldRank, "validateStagingPlanLimits: dst tensor must be non-null");
  NCCL_M2N_CHECK_ARG(srcTensor->mesh != nullptr, worldRank, "validateStagingPlanLimits: src mesh must be non-null");
  NCCL_M2N_CHECK_ARG(dstTensor->mesh != nullptr, worldRank, "validateStagingPlanLimits: dst mesh must be non-null");
  NCCL_M2N_CHECK_ARG(srcTensorDims != nullptr, worldRank,
                     "validateStagingPlanLimits: src shape array must be non-null");
  NCCL_M2N_CHECK_ARG(dstTensorDims != nullptr, worldRank,
                     "validateStagingPlanLimits: dst shape array must be non-null");
  NCCL_M2N_CHECK_ARG(srcTensor->ndims == dstTensor->ndims, worldRank,
                     "validateStagingPlanLimits: src and dst ndims must match");
  NCCL_M2N_CHECK_ARG(gpusPerDomain > 0, worldRank, "validateStagingPlanLimits: gpusPerDomain must be positive");

  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;
  ReshardMeshInterval srcInterval{};
  ReshardMeshInterval dstInterval{};
  NCCL_M2N_CHECK(computeReshardMeshInterval(srcMesh, worldRank, &srcInterval));
  NCCL_M2N_CHECK(computeReshardMeshInterval(dstMesh, worldRank, &dstInterval));
  ncclReshardMeshGroupInfo fullSrcInfo;
  ncclReshardMeshGroupInfo fullDstInfo;
  computeMeshGroupInfo(srcTensor, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstTensor, dstMesh->startRank, &fullDstInfo);

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(resolvePreflightDims(worldRank, srcTensorDims, dstTensorDims, srcTensor->ndims,
                                      fullSrcInfo.shardTensorDim, fullSrcInfo.shardCount, fullDstInfo.shardTensorDim,
                                      fullDstInfo.shardCount, srcDims, dstDims, srcStrides, dstStrides));

  ncclReshardRepLoadBalancer lb = {};
  lb.srcRepCount = fullSrcInfo.repCount;
  lb.dstRepCount = fullDstInfo.repCount;
  lb.dstGpusPerDomain = gpusPerDomain;
  lb.dstRepStartRank = dstMesh->startRank;
  lb.dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1;
  lb.mode = reshardGetLoadBalanceMode();

  const int firstRank = std::min(srcInterval.startRank, dstInterval.startRank);
  const int lastRank = std::max(srcInterval.endRank, dstInterval.endRank);
  size_t maxPeerGroup = 1;
  for (int rank = firstRank; rank < lastRank; rank++) {
    NCCL_M2N_CHECK(validateRankStagingCounts(worldRank, rank, srcTensor, &fullSrcInfo, dstTensor, &fullDstInfo, srcDims,
                                             srcStrides, dstDims, dstStrides, srcTensor->ndims, &lb, copyAlgo,
                                             gpusPerDomain,
                                             &maxPeerGroup));
  }
  if (maxPeerGroupSize != nullptr) {
    *maxPeerGroupSize = maxPeerGroup;
  }
  return ncclSuccess;
}

/* ======================================================================
 * buildStagingDirectTransferDescriptor — direct (no LSA, no ring).
 * ====================================================================*/
ncclResult_t buildStagingDirectTransferDescriptor(ncclComm_t globalComm, void* srcBuffer, const size_t* srcTensorDims,
                                                  int ndims, const ncclDistTensor_t* srcTensor, void* dstBuffer,
                                                  const size_t* dstTensorDims, const ncclDistTensor_t* dstTensor,
                                                  int gpusPerDomain, int nodeLocalRank,
                                                  StagingTransferDescriptor* desc) {
  if (!desc) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1, "buildStagingDirectTransferDescriptor: output descriptor must be non-null");
  }
  memset(desc, 0, sizeof(*desc));

  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;

  int worldRank = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(globalComm, &worldRank));

  bool isSource = reshardRankInMesh(srcMesh, worldRank);
  bool isDest = reshardRankInMesh(dstMesh, worldRank);

  desc->myWorldRank = worldRank;
  desc->myLocalRank = nodeLocalRank;
  desc->isSource = isSource;
  desc->isDest = isDest;
  desc->srcBuffer = srcBuffer;
  desc->dstBuffer = dstBuffer;
  desc->ndims = ndims;

  ncclReshardMeshGroupInfo fullSrcInfo, fullDstInfo;
  computeMeshGroupInfo(srcTensor, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstTensor, dstMesh->startRank, &fullDstInfo);

  int srcShardDim = fullSrcInfo.shardTensorDim;
  int dstShardDim = fullDstInfo.shardTensorDim;
  int srcShardCount = fullSrcInfo.shardCount;
  int dstShardCount = fullDstInfo.shardCount;

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(resolveLocalDims(srcTensorDims, dstTensorDims, ndims, srcShardDim, srcShardCount, dstShardDim,
                                  dstShardCount, isSource, isDest, srcDims, dstDims, srcStrides, dstStrides));

  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  if (isSource) {
    computeMeshGroupInfo(srcTensor, worldRank, &srcInfo);
  }
  if (isDest) {
    computeMeshGroupInfo(dstTensor, worldRank, &dstInfo);
  }

  for (int d = 0; d < ndims; d++) {
    desc->srcDims[d] = srcDims[d];
    desc->dstDims[d] = dstDims[d];
    desc->srcStrides[d] = srcStrides[d];
    desc->dstStrides[d] = dstStrides[d];
  }

  ncclReshardRepLoadBalancer lb = {};
  lb.srcRepCount = fullSrcInfo.repCount;
  lb.dstRepCount = fullDstInfo.repCount;
  lb.dstGpusPerDomain = gpusPerDomain;
  lb.dstRepStartRank = dstMesh->startRank;
  lb.dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1;
  lb.mode = reshardGetLoadBalanceMode();

  const size_t epc_dummy = 1;

  if (isSource) {
    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);

    desc->numTargets = 0;
    bool targetsTruncated = false;

    for (int dstShard = 0; dstShard < dstShardCount; dstShard++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardDim, srcInfo.shardIdx, dstDims, dstStrides,
                                                dstShardDim, dstShard, ndims, epc_dummy, &plan));

      if (plan.totalInnerTransfers == 0) {
        continue;
      }
      if (targetRepStart >= targetRepEnd) {
        continue;
      }

      int numSourcesToDstShard = 0;
      int myPosition = 0;
      for (int ss = 0; ss < srcShardCount; ss++) {
        ncclReshardTransferPlan check;
        NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardDim, ss, dstDims, dstStrides,
                                                  dstShardDim, dstShard, ndims, epc_dummy, &check));
        if (check.totalInnerTransfers > 0) {
          if (ss < srcInfo.shardIdx) {
            myPosition++;
          }
          numSourcesToDstShard++;
        }
      }

      for (int dstRep = targetRepStart; dstRep < targetRepEnd; dstRep++) {
        if (desc->numTargets >= MAX_TARGETS) {
          targetsTruncated = true;
          break;
        }
        int dstRank = getMeshRank(dstTensor, &fullDstInfo, dstShard, dstRep);

        int ti = desc->numTargets++;
        StagingPeerDescriptor* td = &desc->targets[ti];
        td->peerWorldRank = dstRank;
        td->peerShardIdx = dstShard;
        td->isRdma = true;
        td->peerLocalRank = -1;
        NCCL_M2N_CHECK(fillStagingPlan(&td->plan, plan, ndims));

        desc->destNumSources[ti] = numSourcesToDstShard;
        desc->sourceIndexOnDest[ti] = myPosition;
      }
      if (targetsTruncated) {
        break;
      }
    }

    if (targetsTruncated) {
      NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                    "buildStagingDirectTransferDescriptor: target list truncated (%d / MAX_TARGETS=%d); increase "
                    "MAX_TARGETS",
                    desc->numTargets, MAX_TARGETS);
    }
  }

  if (isDest) {
    int sourceRep = getSourceRepForDest(&lb, dstInfo.repIdx);
    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, sourceRep, &targetRepStart, &targetRepEnd);
    int numTargetReps = targetRepEnd - targetRepStart;

    desc->numSources = 0;

    for (int ss = 0; ss < srcShardCount; ss++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardDim, ss, dstDims, dstStrides, dstShardDim,
                                                dstInfo.shardIdx, ndims, epc_dummy, &plan));

      if (plan.totalInnerTransfers == 0) {
        continue;
      }
      if (desc->numSources >= MAX_SOURCES) {
        NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                      "buildStagingDirectTransferDescriptor: source list truncated (%d / MAX_SOURCES=%d); increase "
                      "MAX_SOURCES",
                      desc->numSources, MAX_SOURCES);
      }

      int srcRank = getMeshRank(srcTensor, &fullSrcInfo, ss, sourceRep);

      int si = desc->numSources++;
      StagingPeerDescriptor* sd = &desc->sources[si];
      sd->peerWorldRank = srcRank;
      sd->peerShardIdx = ss;
      sd->isRdma = true;
      sd->peerLocalRank = -1;
      NCCL_M2N_CHECK(fillStagingPlan(&sd->plan, plan, ndims));

      int overlappingBeforeMe = 0;
      int totalOverlapping = 0;
      for (int ds = 0; ds < dstShardCount; ds++) {
        ncclReshardTransferPlan check;
        NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardDim, ss, dstDims, dstStrides,
                                                  dstShardDim, ds, ndims, epc_dummy, &check));
        if (check.totalInnerTransfers > 0) {
          if (ds < dstInfo.shardIdx) {
            overlappingBeforeMe++;
          }
          totalOverlapping++;
        }
      }

      desc->targetIndexOnSource[si] = overlappingBeforeMe * numTargetReps + (dstInfo.repIdx - targetRepStart);
      desc->sourceNumTargets[si] = totalOverlapping * numTargetReps;
    }
  }

  desc->numRingTargets = 0;
  return ncclSuccess;
}
