/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Copy/Staging Path
 *
 * Public C entry point `ncclReshard(handle, comm, src, dst, stream)` that
 * takes arbitrary device buffers (no symmetric-window contract) and runs
 * the staging-buffer-backed kernel.
 *
 * Implementation notes for ncclReshard:
 *   1. No NVL domain detection — gpus_per_domain derived from
 *      devComm->lsaSize, matching the window API.
 *   2. No separate devComm for the staging path — uses the main comm's
 *      devComm and caches via the existing findCachedDevComm pattern.
 *   3. Per-comm staging buffer pool (StagingBufferPoolEntry) with
 *      event-based cross-stream ordering matching the PACK staging pool.
 *
 * Algorithm selection: NCCL_RESHARD_COPY_ALGORITHM env var
 *   {default, direct, pipe}.
 ************************************************************************/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "reshard_call_setup.h"
#include "reshard_internal.h"
#include "m2n_log.h"
#include "reshard_split.h"
#include "reshard_types.h"
#include "staging_buffer.h"
#include "staging_profile.h"
#include "staging_types.h"

/* Advances the persistent flow-control generation for each reshard call. */
static std::atomic<uint64_t> gStagingEpoch{0};

#ifdef NCCL_M2N_TESTING
static std::atomic<ReshardCopyAlgorithm> gLastCompletedCopyAlgorithm{RESHARD_COPY_ALGO_DIRECT};

ReshardCopyAlgorithm reshardGetLastCompletedCopyAlgorithmForTest() {
  return gLastCompletedCopyAlgorithm.load(std::memory_order_relaxed);
}
#endif

static size_t parseEnvSize(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 0);
  if (parsed == 0) {
    return 0;
  }
  if (end != nullptr && end[0] != '\0') {
    bool kib = (end[0] == 'k' || end[0] == 'K') &&
               (end[1] == '\0' || ((end[1] == 'b' || end[1] == 'B') && end[2] == '\0') ||
                ((end[1] == 'i' || end[1] == 'I') && (end[2] == 'b' || end[2] == 'B') && end[3] == '\0'));
    bool mib = (end[0] == 'm' || end[0] == 'M') &&
               (end[1] == '\0' || ((end[1] == 'b' || end[1] == 'B') && end[2] == '\0') ||
                ((end[1] == 'i' || end[1] == 'I') && (end[2] == 'b' || end[2] == 'B') && end[3] == '\0'));
    if (kib) {
      parsed *= 1024ULL;
    } else if (mib) {
      parsed *= 1024ULL * 1024ULL;
    }
  }
  return (size_t)parsed;
}

static int getPipeTmaTileSize() {
  size_t tileSize = parseEnvSize(getenv("NCCL_RESHARD_PIPE_TMA_TILE_SIZE"));
  if (tileSize == 0) {
    return 64 * 1024;
  }
  switch (tileSize) {
  case 8 * 1024:
  case 16 * 1024:
  case 32 * 1024:
  case 64 * 1024:
    return (int)tileSize;
  default:
    RESHARD_WARN(-1, "unsupported NCCL_RESHARD_PIPE_TMA_TILE_SIZE=%zu; using 64KiB", tileSize);
    return 64 * 1024;
  }
}

static ReshardStagingTensorSignature stagingProcessTensorSignature(const ncclDistTensor_t* src,
                                                                   const ncclDistTensor_t* dst) {
  ReshardStagingTensorSignature signature{};
  signature.srcNdims = src->ndims;
  signature.srcDtype = src->dtype;
  signature.dstNdims = dst->ndims;
  signature.dstDtype = dst->dtype;
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    signature.srcLocalShape[d] = (d < src->ndims) ? src->localShape[d] : 0;
    signature.dstLocalShape[d] = (d < dst->ndims) ? dst->localShape[d] : 0;
  }
  return signature;
}

static ReshardStagingChannelSignature stagingProcessChannelSignature(int activeChannelCount) {
  ReshardStagingChannelSignature signature{};
  signature.activeChannelCount = activeChannelCount;
  return signature;
}

static ReshardStagingMeshSignature stagingProcessMeshSignature(const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                                               int srcGpusPerDomain, int dstGpusPerDomain,
                                                               bool splitStrided, int splitNumInjectionDomains,
                                                               int splitDomainsPerRep) {
  ReshardStagingMeshSignature signature{};
  signature.srcStartRank = src->mesh->startRank;
  signature.dstStartRank = dst->mesh->startRank;
  for (int d = 0; d < NCCL_RESHARD_MESH_NDIMS; d++) {
    signature.srcMeshDims[d] = src->mesh->dims[d];
    signature.srcPlacements[d] = src->placements[d];
    signature.dstMeshDims[d] = dst->mesh->dims[d];
    signature.dstPlacements[d] = dst->placements[d];
  }
  signature.srcGpusPerDomain = srcGpusPerDomain;
  signature.dstGpusPerDomain = dstGpusPerDomain;
  signature.loadBalanceMode = reshardEffectiveLbMode(src, dst);
  signature.splitStrided = splitStrided;
  signature.splitNumInjectionDomains = splitNumInjectionDomains;
  signature.splitDomainsPerRep = splitDomainsPerRep;
  return signature;
}

/* DIRECT has no persistent per-shape state in raw staging storage. Reuse a
 * buffer across shapes with the same channel capacity; its completion event
 * orders successive launches on different streams. */
static uint64_t stagingDirectReusablePoolKey(int capacityChannels) {
  return 0x73746167696e6744ULL ^ static_cast<uint64_t>(capacityChannels); /* "stagingD" */
}

static uint64_t stagingPipeReusablePoolKey() {
  return 0x73746167696e6750ULL; /* "stagingP" */
}

static int stagingComputeCtaHeuristicPeerCount(const StagingTransferDescriptor* desc, const ncclDistTensor_t* src,
                                               const ncclDistTensor_t* dst) {
  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  computeMeshGroupInfo(src, src->mesh->startRank, &srcInfo);
  computeMeshGroupInfo(dst, dst->mesh->startRank, &dstInfo);

  const int srcShardDim = srcInfo.shardTensorDim;
  const int dstShardDim = dstInfo.shardTensorDim;
  const int srcShardCount = std::max(1, srcInfo.shardCount);
  const int dstShardCount = std::max(1, dstInfo.shardCount);
  int maxActivePeers = 1;
  bool foundActive = false;
  const size_t elementsPerChunk = 1;

  for (int ss = 0; ss < srcShardCount; ss++) {
    int active = 0;
    for (int ds = 0; ds < dstShardCount; ds++) {
      ncclReshardTransferPlan plan{};
      computeTransferPlan(desc->srcDims, desc->srcStrides, srcShardDim, ss, desc->dstDims, desc->dstStrides,
                          dstShardDim, ds, desc->ndims, elementsPerChunk, &plan);
      if (plan.totalInnerTransfers > 0) {
        active++;
      }
    }
    if (active > 0) {
      maxActivePeers = std::max(maxActivePeers, active);
      foundActive = true;
    }
  }

  for (int ds = 0; ds < dstShardCount; ds++) {
    int active = 0;
    for (int ss = 0; ss < srcShardCount; ss++) {
      ncclReshardTransferPlan plan{};
      computeTransferPlan(desc->srcDims, desc->srcStrides, srcShardDim, ss, desc->dstDims, desc->dstStrides,
                          dstShardDim, ds, desc->ndims, elementsPerChunk, &plan);
      if (plan.totalInnerTransfers > 0) {
        active++;
      }
    }
    if (active > 0) {
      maxActivePeers = std::max(maxActivePeers, active);
      foundActive = true;
    }
  }

  return foundActive ? std::max(1, maxActivePeers) : 1;
}

static int stagingComputePeerGroupSizeBound(int activePeerCount, const ncclDistTensor_t* dst, int gpusPerDomain,
                                            ReshardCopyAlgorithm copyAlgo) {
  ncclReshardMeshGroupInfo dstInfo{};
  computeMeshGroupInfo(dst, dst->mesh->startRank, &dstInfo);

  int bound = std::max(1, activePeerCount);
  if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
    bound *= std::max(1, dstInfo.repCount);
  } else {
    const int localDstReps = std::max(1, std::min(std::max(1, dstInfo.repCount), std::max(1, gpusPerDomain)));
    bound *= localDstReps;
  }
  return std::max(1, bound);
}

static int stagingComputeHierarchyPeerGroupSizeBound(int activePeerCount, const ncclDistTensor_t* dst,
                                                     int dstGpusPerDomain, int dstNodeAnchor) {
  ncclReshardMeshGroupInfo dstInfo{};
  computeMeshGroupInfo(dst, dst->mesh->startRank, &dstInfo);

  const int active = std::max(1, activePeerCount);
  const int dstShards = std::max(1, dstInfo.shardCount);
  const int dstReps = std::max(1, dstInfo.repCount);
  const int gpd = std::max(1, dstGpusPerDomain);
  auto rankNode = [&](int rank) -> int { return (rank - dstNodeAnchor) / gpd; };

  int maxLocalReps = 1;
  for (int shard = 0; shard < dstShards; shard++) {
    for (int rep = 0; rep < dstReps; rep++) {
      const int rank = getMeshRank(dst, &dstInfo, shard, rep);
      const int node = rankNode(rank);
      int localReps = 0;
      for (int other = 0; other < dstReps; other++) {
        const int otherRank = getMeshRank(dst, &dstInfo, shard, other);
        if (rankNode(otherRank) == node) {
          localReps++;
        }
      }
      maxLocalReps = std::max(maxLocalReps, localReps);
    }
  }

  const int maxAssignedSources = (active + maxLocalReps - 1) / maxLocalReps;
  const int maxHandlerTargets = maxAssignedSources * maxLocalReps;
  return std::max(active, maxHandlerTargets);
}

static int tensorRepCount(const ncclDistTensor_t* tensor) {
  ncclReshardMeshGroupInfo info{};
  computeMeshGroupInfo(tensor, tensor->mesh->startRank, &info);
  return std::max(1, info.repCount);
}

static int splitParentRankToA(const ReshardSplitComms* sc, int parentRank) {
  if (sc == nullptr) {
    return parentRank;
  }
  if (parentRank >= sc->srcStartRank && parentRank < sc->srcStartRank + sc->srcMeshSize) {
    return parentRank - sc->srcStartRank;
  }
  const int genLimit = sc->dstStartRank + sc->numInjectionDomains * sc->lsaSize;
  if (parentRank >= sc->dstStartRank && parentRank < genLimit) {
    return sc->srcMeshSize + (parentRank - sc->dstStartRank);
  }
  return -1;
}

static int splitParentRankToB(const ReshardSplitComms* sc, int parentRank) {
  if (sc == nullptr) {
    return parentRank;
  }
  if (parentRank >= sc->dstStartRank && parentRank < sc->dstStartRank + sc->dstMeshSize) {
    return parentRank - sc->dstStartRank;
  }
  return -1;
}

static bool splitParentRankIsSource(const ReshardSplitComms* sc, int parentRank) {
  return sc != nullptr && parentRank >= sc->srcStartRank && parentRank < sc->srcStartRank + sc->srcMeshSize;
}

static void offsetPipeFlowCtrl(StagingFlowCtrl* fc, int signalOffset, int counterOffset) {
  fc->localTailSignal += signalOffset;
  fc->localHeadSignal += signalOffset;
  fc->remoteTailSignal += signalOffset;
  fc->remoteHeadSignal += signalOffset;
  fc->localPutCounter += counterOffset;
}

static void applyPipeRdmaOffsets(StagingKernelParams* params, const ReshardSplitComms* splitComms, int signalOffsetA,
                                 int counterOffsetA, int signalOffsetB, int counterOffsetB) {
  const bool split = splitComms != nullptr;
  const bool thisRankIsSource = split && splitParentRankIsSource(splitComms, params->myRank);
  for (int ch = 0; ch < params->numChannels; ch++) {
    for (int t = 0; t < params->numRdmaTargets; t++) {
      StagingPeerInfo* target = &params->rdmaTargets[ch][t];
      if (!target->active) {
        continue;
      }
      const bool useCommB = split && !thisRankIsSource;
      const int signalOffset = useCommB ? signalOffsetB : signalOffsetA;
      const int counterOffset = useCommB ? counterOffsetB : counterOffsetA;
      offsetPipeFlowCtrl(&target->fc, signalOffset, counterOffset);
      offsetPipeFlowCtrl(&params->localRdmaFc[ch][t], signalOffset, counterOffset);
    }
    for (int s = 0; s < params->numRdmaSources; s++) {
      StagingPeerInfo* source = &params->rdmaSources[ch][s];
      if (source->active) {
        const bool useCommB = split && !splitParentRankIsSource(splitComms, source->peerWorldRank);
        offsetPipeFlowCtrl(&source->fc, useCommB ? signalOffsetB : signalOffsetA,
                           useCommB ? counterOffsetB : counterOffsetA);
      }
    }
  }
}

static ncclResult_t translateSplitPipePeer(StagingPeerInfo* peer, const ReshardSplitComms* sc, bool useCommB,
                                           int parentRankForLog) {
  if (peer == nullptr || !peer->active) {
    return ncclSuccess;
  }
  const int parentPeer = peer->peerWorldRank;
  if (useCommB) {
    const int rankB = splitParentRankToB(sc, parentPeer);
    if (rankB < 0) {
      NCCL_M2N_FAIL(ncclInvalidArgument, parentRankForLog, "PIPE split translate: parent peer %d is not in commB",
                    parentPeer);
    }
    peer->peerWorldRank = rankB;
    peer->fc.remoteRank = rankB;
    peer->rdmaTransport = STAGING_RDMA_TRANSPORT_SPLIT_B;
  } else {
    const int rankA = splitParentRankToA(sc, parentPeer);
    if (rankA < 0) {
      NCCL_M2N_FAIL(ncclInvalidArgument, parentRankForLog, "PIPE split translate: parent peer %d is not in commA",
                    parentPeer);
    }
    peer->peerWorldRank = rankA;
    peer->fc.remoteRank = rankA;
    peer->rdmaTransport = STAGING_RDMA_TRANSPORT_SPLIT_A;
  }
  return ncclSuccess;
}

static ncclResult_t applySplitPipeParams(StagingKernelParams* params, const ReshardSplitComms* sc, ncclWindow_t windowA,
                                         ncclWindow_t windowB, int signalsPerControlSlot, int countersPerControlSlot,
                                         int signalsPerSlotB, int countersPerSlotB, int persistentControlSlot) {
  NCCL_M2N_CHECK_ARG(params != nullptr && sc != nullptr && sc->active, -1,
                     "applySplitPipeParams: active split state and params are required");
  params->rdmaWindow = windowA;
  params->rdmaWindowB = windowB;
  params->lsaWindow = windowB;

  const ncclWindow_t localRdmaWindow = (sc->inA && windowA != nullptr) ? windowA : windowB;
  for (int ch = 0; ch < params->numChannels; ch++) {
    params->rdmaRegions[ch].window = localRdmaWindow;
    params->lsaRegions[ch].window = windowB;
  }

  const int signalOffsetA = persistentControlSlot * signalsPerControlSlot;
  const int counterOffsetA = persistentControlSlot * countersPerControlSlot;
  const int signalOffsetB = sc->slotIdx * signalsPerSlotB + signalOffsetA;
  const int counterOffsetB = sc->slotIdx * countersPerSlotB + counterOffsetA;
  const bool thisRankIsSource = splitParentRankIsSource(sc, params->myRank);

  applyPipeRdmaOffsets(params, sc, signalOffsetA, counterOffsetA, signalOffsetB, counterOffsetB);

  /* stagingPrepareTransfer builds parent-rank peers; translate them after
   * assigning their graph-slot offsets. */
  for (int ch = 0; ch < params->numChannels; ch++) {
    for (int t = 0; t < params->numRdmaTargets; t++) {
      const bool useCommB = !thisRankIsSource;
      NCCL_M2N_CHECK(translateSplitPipePeer(&params->rdmaTargets[ch][t], sc, useCommB, params->myRank));
    }
    for (int s = 0; s < params->numRdmaSources; s++) {
      StagingPeerInfo* source = &params->rdmaSources[ch][s];
      const bool useCommB = source->active && !splitParentRankIsSource(sc, source->peerWorldRank);
      NCCL_M2N_CHECK(translateSplitPipePeer(source, sc, useCommB, params->myRank));
    }
  }
  return ncclSuccess;
}

static void stagingStripPipeTransferPlans(StagingKernelParams* params) {
  for (int ch = 0; ch < STAGING_MAX_CHANNELS; ch++) {
    for (int t = 0; t < MAX_TARGETS; t++) {
      memset(&params->rdmaTargets[ch][t].plan, 0, sizeof(params->rdmaTargets[ch][t].plan));
      memset(&params->lsaTargets[ch][t].plan, 0, sizeof(params->lsaTargets[ch][t].plan));
    }
    for (int s = 0; s < MAX_SOURCES; s++) {
      memset(&params->rdmaSources[ch][s].plan, 0, sizeof(params->rdmaSources[ch][s].plan));
      memset(&params->lsaSources[ch][s].plan, 0, sizeof(params->lsaSources[ch][s].plan));
    }
  }
}

static void stagingNormalizePipeStaticPlan(StagingKernelParams* params) {
  /* After stagingBuildPipeDevicePlan extracts edge/layout metadata, keep
   * only immutable launch state in cached StagingKernelParams.  Per-call
   * buffers, debug epoch, and full copy plans are supplied separately or via
   * the compact StagingPipeDevicePlan. */
  params->srcBuffer = nullptr;
  params->dstBuffer = nullptr;
  params->stagingBuffer = nullptr;
  params->ginSignalCount = 0;
  params->ginCounterCount = 0;
  params->epoch = 0;

  memset(params->srcDims, 0, sizeof(params->srcDims));
  memset(params->dstDims, 0, sizeof(params->dstDims));
  memset(params->srcStrides, 0, sizeof(params->srcStrides));
  memset(params->dstStrides, 0, sizeof(params->dstStrides));
  params->ndims = 0;

  stagingStripPipeTransferPlans(params);
}

static void stagingMakePipeCopyLayout(const StagingTransferPlan* plan, bool pack, StagingPipeCopyLayout* layout) {
  memset(layout, 0, sizeof(*layout));
  if (plan == nullptr) {
    return;
  }

  layout->numOuterLoops = plan->numOuterLoops;
  layout->baseOffset = pack ? plan->srcBaseOffset : plan->dstBaseOffset;
  layout->innerSize = plan->innerSize;
  layout->isContiguous = plan->isContiguous;
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    layout->outerCounts[d] = plan->outerCounts[d];
    layout->outerStrides[d] = pack ? plan->outerSrcStrides[d] : plan->outerDstStrides[d];
  }
}

static void stagingMakePipePeerEdge(const StagingPeerInfo* peer, int copyLayoutIndex, StagingPipePeerEdge* edge) {
  memset(edge, 0, sizeof(*edge));
  if (peer == nullptr) {
    edge->peerWorldRank = -1;
    edge->peerLocalRank = -1;
    return;
  }

  edge->peerWorldRank = peer->peerWorldRank;
  edge->peerLocalRank = peer->peerLocalRank;
  edge->rdmaTransport = peer->rdmaTransport;
  edge->copyLayoutIndex = copyLayoutIndex;
  edge->active = peer->active && peer->channelCount > 0;
  edge->totalBytes = peer->totalBytes;
  edge->chunkStart = peer->chunkStart;
  edge->chunkEnd = peer->chunkEnd;
  edge->fc = peer->fc;
}

static void stagingBuildPipeDevicePlan(const StagingKernelParams* params, StagingPipeDevicePlan* plan) {
  memset(plan, 0, sizeof(*plan));
  for (int ch = 0; ch < STAGING_MAX_CHANNELS; ch++) {
    bool hasRdmaTarget = false;
    bool hasRdmaSource = false;
    bool hasLsaSource = false;
    for (int t = 0; t < MAX_TARGETS; t++) {
      const StagingPeerInfo* target = &params->rdmaTargets[ch][t];
      stagingMakePipePeerEdge(target, t, &plan->rdmaTargets[ch][t].peer);
      hasRdmaTarget = hasRdmaTarget || plan->rdmaTargets[ch][t].peer.active;
      plan->rdmaTargets[ch][t].localFc = params->localRdmaFc[ch][t];
      stagingMakePipeCopyLayout(&target->plan, true, &plan->rdmaTargetLayouts[ch][t]);

      stagingMakePipePeerEdge(&params->lsaTargets[ch][t], t, &plan->lsaTargets[ch][t]);
    }
    for (int s = 0; s < MAX_SOURCES; s++) {
      const StagingPeerInfo* rdmaSource = &params->rdmaSources[ch][s];
      stagingMakePipePeerEdge(rdmaSource, s, &plan->rdmaSources[ch][s]);
      hasRdmaSource = hasRdmaSource || plan->rdmaSources[ch][s].active;
      stagingMakePipeCopyLayout(&rdmaSource->plan, false, &plan->rdmaSourceLayouts[ch][s]);

      const StagingPeerInfo* lsaSource = &params->lsaSources[ch][s];
      stagingMakePipePeerEdge(lsaSource, s, &plan->lsaSources[ch][s]);
      hasLsaSource = hasLsaSource || plan->lsaSources[ch][s].active;
      stagingMakePipeCopyLayout(&lsaSource->plan, false, &plan->lsaSourceLayouts[ch][s]);
    }
    if (hasRdmaTarget && plan->numTrainerRdmaLaunchChannels < STAGING_MAX_CHANNELS) {
      plan->trainerRdmaLaunchChannels[plan->numTrainerRdmaLaunchChannels++] = ch;
    }
    if (hasRdmaSource && plan->numGeneratorRdmaLaunchChannels < STAGING_MAX_CHANNELS) {
      plan->generatorRdmaLaunchChannels[plan->numGeneratorRdmaLaunchChannels++] = ch;
    }
    if (hasLsaSource && plan->numGeneratorLsaLaunchChannels < STAGING_MAX_CHANNELS) {
      plan->generatorLsaLaunchChannels[plan->numGeneratorLsaLaunchChannels++] = ch;
    }
  }
}

static ncclResult_t stagingEnsurePipePlanEntryStorage(StagingPipePlanCacheEntry* entry, int rank) {
  NCCL_M2N_CHECK_ARG(entry != nullptr, rank, "[STAGING] PIPE plan cache requested with null entry");
  if (entry->hostParams == nullptr) {
    entry->hostParams = new (std::nothrow) StagingKernelParams;
  }
  if (entry->hostPlan == nullptr) {
    entry->hostPlan = new (std::nothrow) StagingPipeDevicePlan;
  }
  if (entry->hostParams == nullptr || entry->hostPlan == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, rank, "[STAGING] failed to allocate cached PIPE host plan");
  }
  if (entry->devParams == nullptr) {
    NCCL_M2N_CUDACHECK(cudaMalloc(&entry->devParams, sizeof(StagingKernelParams)));
  }
  if (entry->devPlan == nullptr) {
    NCCL_M2N_CUDACHECK(cudaMalloc(&entry->devPlan, sizeof(StagingPipeDevicePlan)));
  }
  return ncclSuccess;
}

static bool stagingPlanMatches(const StagingPipePlanCacheEntry& entry, const ReshardStagingMeshSignature& meshSignature,
                               const ReshardStagingChannelSignature& channelSignature,
                               const ReshardStagingTensorSignature& tensorSignature) {
  return entry.hostValid && entry.meshSignature == meshSignature && entry.channelSignature == channelSignature &&
         entry.tensorSignature == tensorSignature;
}

static void stagingInvalidatePipePlanEntry(StagingPipePlanCacheEntry* entry,
                                           const ReshardStagingMeshSignature& meshSignature,
                                           const ReshardStagingChannelSignature& channelSignature,
                                           const ReshardStagingTensorSignature& tensorSignature) {
  entry->meshSignature = meshSignature;
  entry->channelSignature = channelSignature;
  entry->tensorSignature = tensorSignature;
  entry->hostValid = false;
  entry->deviceValid = false;
}

static ncclResult_t stagingGetPipePlanEntry(StagingBufferState* staging,
                                            const ReshardStagingMeshSignature& meshSignature,
                                            const ReshardStagingChannelSignature& channelSignature,
                                            const ReshardStagingTensorSignature& tensorSignature, int preferredSlot,
                                            int rank, StagingPipePlanCacheEntry** outEntry, bool* cacheMiss) {
  NCCL_M2N_CHECK_ARG(staging != nullptr && outEntry != nullptr && cacheMiss != nullptr, rank,
                     "[STAGING] PIPE plan cache lookup requires staging, entry, and miss outputs");
  for (int i = 0; i < STAGING_PIPE_CONTROL_SLOTS; i++) {
    StagingPipePlanCacheEntry& entry = staging->pipePlanCache[i];
    if (stagingPlanMatches(entry, meshSignature, channelSignature, tensorSignature)) {
      *outEntry = &entry;
      *cacheMiss = false;
      return ncclSuccess;
    }
  }

  int slot = -1;
  if (preferredSlot >= 0 && preferredSlot < STAGING_PIPE_CONTROL_SLOTS &&
      !staging->pipePlanCache[preferredSlot].hostValid) {
    slot = preferredSlot;
  }
  if (slot < 0) {
    for (int i = 0; i < STAGING_PIPE_CONTROL_SLOTS; i++) {
      if (!staging->pipePlanCache[i].hostValid) {
        slot = i;
        break;
      }
    }
  }
  if (slot < 0) {
    slot = staging->pipePlanCacheNextVictim;
    staging->pipePlanCacheNextVictim = (staging->pipePlanCacheNextVictim + 1) % STAGING_PIPE_CONTROL_SLOTS;
  }

  StagingPipePlanCacheEntry& entry = staging->pipePlanCache[slot];
  NCCL_M2N_CHECK(stagingEnsurePipePlanEntryStorage(&entry, rank));
  stagingInvalidatePipePlanEntry(&entry, meshSignature, channelSignature, tensorSignature);
  *outEntry = &entry;
  *cacheMiss = true;
  return ncclSuccess;
}

static ncclResult_t stagingFinalizePipeHostPlan(StagingPipePlanCacheEntry* entry,
                                                const ReshardStagingMeshSignature& meshSignature,
                                                const ReshardStagingChannelSignature& channelSignature,
                                                const ReshardStagingTensorSignature& tensorSignature, int rank) {
  NCCL_M2N_CHECK_ARG(entry != nullptr && entry->hostParams != nullptr && entry->hostPlan != nullptr, rank,
                     "[STAGING] cannot finalize missing PIPE host plan");

  /* Build the compact device plan before stripping bulky transfer plans from
   * hostParams; this keeps warm launches to two cached device pointers plus a
   * small by-value StagingPipeCallParams. */
  stagingBuildPipeDevicePlan(entry->hostParams, entry->hostPlan);
  stagingNormalizePipeStaticPlan(entry->hostParams);
  entry->meshSignature = meshSignature;
  entry->channelSignature = channelSignature;
  entry->tensorSignature = tensorSignature;
  entry->hostValid = true;
  entry->deviceValid = false;
  return ncclSuccess;
}

static ncclResult_t stagingEnsurePipeDevicePlan(StagingPipePlanCacheEntry* entry,
                                                const ReshardStagingMeshSignature& meshSignature,
                                                const ReshardStagingChannelSignature& channelSignature,
                                                const ReshardStagingTensorSignature& tensorSignature,
                                                cudaStream_t stream, int rank) {
  NCCL_M2N_CHECK_ARG(entry != nullptr && entry->hostParams != nullptr && entry->hostPlan != nullptr &&
                       entry->devParams != nullptr && entry->devPlan != nullptr && entry->hostValid &&
                       entry->meshSignature == meshSignature && entry->channelSignature == channelSignature &&
                       entry->tensorSignature == tensorSignature,
                     rank, "[STAGING] cannot upload missing PIPE device plan");
  if (entry->deviceValid && entry->meshSignature == meshSignature && entry->channelSignature == channelSignature &&
      entry->tensorSignature == tensorSignature) {
    return ncclSuccess;
  }

  /* This upload happens only on a shape-cache miss.  Warm calls pass mutable
   * src/dst pointers and toggles as kernel arguments and do not update this
   * cached device state. */
  NCCL_M2N_CUDACHECK(cudaMemcpyAsync(entry->devParams, entry->hostParams, sizeof(*entry->hostParams),
                                     cudaMemcpyHostToDevice, stream));
  NCCL_M2N_CUDACHECK(cudaMemcpyAsync(entry->devPlan, entry->hostPlan, sizeof(*entry->hostPlan), cudaMemcpyHostToDevice,
                                     stream));
  entry->deviceValid = true;
  return ncclSuccess;
}

class StagingBufferEventGuard {
public:
  StagingBufferEventGuard(ncclComm_t comm, uint64_t poolKey, cudaStream_t stream)
    : comm_(comm), poolKey_(poolKey), stream_(stream) {}
  StagingBufferEventGuard(const StagingBufferEventGuard&) = delete;
  StagingBufferEventGuard& operator=(const StagingBufferEventGuard&) = delete;
  ~StagingBufferEventGuard() {
    if (bActive_) {
      NCCL_M2N_CHECK_WARN(complete());
    }
  }

  ncclResult_t complete() {
    if (!bActive_) {
      return ncclSuccess;
    }
    bActive_ = false;
    return stagingBufferPoolRecordEvent(comm_, poolKey_, stream_);
  }

private:
  ncclComm_t comm_;
  uint64_t poolKey_;
  cudaStream_t stream_;
  bool bActive_ = true;
};

/* ======================================================================
 * ncclReshard — copy/staging-based public entry.
 * ====================================================================*/

extern "C" ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src,
                                    const ncclDistTensor_t* dst, cudaStream_t stream) {
  if (m2nGroupIsActive()) {
    return m2nGroupEnqueueReshard(handle, comm, src, dst, stream);
  }
  M2nApiLock apiLock;
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(comm != nullptr, -1, "ncclReshard: comm must be non-null");
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr, -1,
                     "ncclReshard: src and dst tensor descriptors must both be non-null on every rank");
  ReshardTensorSetup tensorSetup;
  NCCL_M2N_CHECK(reshardPrepareTensorSetup("ncclReshard", src, dst, &tensorSetup));
  const int ndims = tensorSetup.ndims;
  const size_t element_size = tensorSetup.elementSize;
  void* const srcBuffer = tensorSetup.srcTensor.dataPtr;
  void* const dstBuffer = tensorSetup.dstTensor.dataPtr;
  const size_t* const src_tensor_dims = tensorSetup.srcTensor.localShape;
  const size_t* const dst_tensor_dims = tensorSetup.dstTensor.localShape;
  ncclDistTensor_t& src_local = tensorSetup.srcTensor;
  ncclDistTensor_t& dst_local = tensorSetup.dstTensor;
  const ncclMesh_t* const src_mesh = &tensorSetup.srcMesh;
  const ncclMesh_t* const dst_mesh = &tensorSetup.dstMesh;
  const ncclDistTensor_t* const srcTensor = &src_local;
  const ncclDistTensor_t* const dstTensor = &dst_local;
  const ncclMesh_t* const srcMesh = src_mesh;
  const ncclMesh_t* const dstMesh = dst_mesh;
  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));
  /* A caller comm created with ncclConfig_t.blocking=0 may still be
   * initializing; drive it to readiness before the first communicator query.
   * Run under M2nApiUnlock so a peer rank in this process can progress it. */
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  const ReshardCopyAlgorithm copyAlgo = reshardGetCopyAlgorithm();
  int parentCommSize = 0;
  NCCL_M2N_CHECK(ncclCommCount(comm, &parentCommSize));
  reshardResolveAdaptiveScaleConfig(parentCommSize,
                                    copyAlgo == RESHARD_COPY_ALGO_PACK || copyAlgo == RESHARD_COPY_ALGO_PIPE);

  int world_rank = 0, world_size = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &world_rank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &world_size));
  NCCL_M2N_CHECK(validateReshardMeshBounds(src_mesh, dst_mesh, world_size, world_rank));
  NCCL_M2N_CHECK(reshardValidateActiveBuffers("ncclReshard", world_rank, &src_local, &dst_local));

  auto check_shard_global_size = [&](const char* side, const ncclDistTensor_t* tensor,
                                     const size_t* dims) -> ncclResult_t {
    ncclReshardMeshGroupInfo info;
    computeMeshGroupInfo(tensor, tensor->mesh->startRank, &info);
    if (info.shardTensorDim < 0) {
      return ncclSuccess;
    }
    size_t global_dim = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dims[info.shardTensorDim], (size_t)info.shardCount, &global_dim), world_rank,
                       "ncclReshard: %s shard dimension overflows global extent: dim=%d local=%zu shardCount=%d", side,
                       info.shardTensorDim, dims[info.shardTensorDim], info.shardCount);
    return ncclSuccess;
  };
  NCCL_M2N_CHECK(check_shard_global_size("source", &src_local, src_tensor_dims));
  NCCL_M2N_CHECK(check_shard_global_size("destination", &dst_local, dst_tensor_dims));

  auto check_local_bytes = [&](const size_t* dims, const char* side) -> ncclResult_t {
    size_t total = element_size;
    for (int d = 0; d < ndims; d++) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(total, dims[d], &total), world_rank,
                         "ncclReshard: %s local byte size overflow at dim %d: current=%zu dim=%zu", side, d, total,
                         dims[d]);
    }
    return ncclSuccess;
  };
  NCCL_M2N_CHECK(check_local_bytes(src_tensor_dims, "source"));
  NCCL_M2N_CHECK(check_local_bytes(dst_tensor_dims, "destination"));

  int currentCudaDev = 0;
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclSuccess;
  NCCL_M2N_CHECK(reshardMatchCommCudaDevice(comm, &currentCudaDev, &commProps, &propsResult));
  NCCL_M2N_CHECK(reshardRejectGraphCapture("ncclReshard", stream));

  /* Stream pool for default-stream callers. */
  ReshardWorkStream work{};
  ncclResult_t setupResult = reshardSetupWorkStream(comm, stream, currentCudaDev, propsResult, &commProps, &work);
  if (setupResult != ncclSuccess) {
    return setupResult;
  }
  ReshardWorkStreamCompletion workCompletion(stream, &work);
  cudaStream_t workStream = work.stream;
  const uint64_t callEpoch = gStagingEpoch.fetch_add(1, std::memory_order_relaxed);
  /* PACK: full per-dest-contiguous CE pack + hierarchical
   * user-window kernel + CE unpack.  Reuses the transpose/transfer
   * buffer and reshardKernelUserWindow; bypasses the chunk-ring
   * staging kernel below. */
  if (copyAlgo == RESHARD_COPY_ALGO_PACK) {
    NCCL_M2N_CHECK(reshardStartWorkStream(stream, &work));
    NCCL_M2N_CHECK(reshardCopyPackNormalized(comm, srcTensor, dstTensor, workStream));
    NCCL_M2N_CHECK(workCompletion.complete());
#ifdef NCCL_M2N_TESTING
    gLastCompletedCopyAlgorithm.store(RESHARD_COPY_ALGO_PACK, std::memory_order_relaxed);
#endif
    return ncclSuccess;
  }

  const bool debugLogging = reshardGetLogLevel() >= RESHARD_LOG_DEBUG;
  auto profile = debugLogging ? stagingProfileCreate() : std::unique_ptr<StagingProfile>{};

  /* Convert dims to bytes (last dim absorbs element size). */
  size_t src_dims_bytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dst_dims_bytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(reshardDimsToBytes(world_rank, "ncclReshard:", ndims, element_size, src_tensor_dims, dst_tensor_dims,
                                    src_dims_bytes, dst_dims_bytes));

  int lsa_size_from_comm = 0;
  const int gpus_per_node_cfg = reshardGetGpusPerNode();
  int src_gpus_per_domain = (gpus_per_node_cfg > 0) ? gpus_per_node_cfg : 1;
  int dst_gpus_per_domain = src_gpus_per_domain;
  int node_local_rank = world_rank % dst_gpus_per_domain;
  ncclDevComm probeDevComm{};
  ReshardDevCommUse probeDevCommUse;
  bool parentTopologyReady = false;
  auto ensureParentDomainTopology = [&]() -> ncclResult_t {
    if (parentTopologyReady) {
      return ncclSuccess;
    }
    /* The probe only reads the parent DevComm's LSA size/rank. Match split
     * setup's zero-resource RAIL probe instead of allocating a full-GIN
     * world barrier before the actual transfer requirements are known. */
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_PROBE_DEV_COMM);
    NCCL_M2N_CHECK(reshardGetOrCreateDevCommWithRequirements(comm, 0, 0, 0, RESHARD_DEVCOMM_BARRIER_WORLD, 1,
                                                             NCCLM2N_GIN_RAIL_CONNECTION, workStream, &probeDevComm,
                                                             &probeDevCommUse));
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&probeDevCommUse, workStream));
    lsa_size_from_comm = (probeDevComm.lsaSize > 0) ? probeDevComm.lsaSize : 0;
    NCCL_M2N_CHECK(resolveReshardDomainSizes(world_rank, RESHARD_ALGO_RING, lsa_size_from_comm, lsa_size_from_comm,
                                             &src_gpus_per_domain, &dst_gpus_per_domain));
    node_local_rank = (probeDevComm.lsaRank >= 0 && probeDevComm.lsaRank < dst_gpus_per_domain) ?
                        probeDevComm.lsaRank :
                        world_rank % dst_gpus_per_domain;
    parentTopologyReady = true;
    return ncclSuccess;
  };

  if (copyAlgo != RESHARD_COPY_ALGO_PIPE) {
    NCCL_M2N_CHECK(ensureParentDomainTopology());
  }

  StagingTransferDescriptor desc;
  size_t maxPeerGroupSize = 1;
  bool splitMetaAvailable = false;
  auto buildCurrentStagingDescriptor = [&](StagingTransferDescriptor* outDesc, bool splitStrided = false,
                                           int splitNumInjectionDomains = 0, int splitDomainsPerRep = 1,
                                           bool nodeAnchorAtMeshStart = false) -> ncclResult_t {
    size_t localMaxPeerGroupSize = 1;
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_BUILD_DESCRIPTOR);
    NCCL_M2N_CHECK(validateStagingPlanLimits(world_rank, srcTensor, src_dims_bytes, dstTensor, dst_dims_bytes, copyAlgo,
                                             dst_gpus_per_domain, &localMaxPeerGroupSize, splitStrided,
                                             splitNumInjectionDomains, splitDomainsPerRep, nodeAnchorAtMeshStart));
    maxPeerGroupSize = localMaxPeerGroupSize;
    memset(outDesc, 0, sizeof(*outDesc));
    if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
      NCCL_M2N_CHECK(buildStagingDirectTransferDescriptor(comm, srcBuffer, src_dims_bytes, ndims, srcTensor, dstBuffer,
                                                          dst_dims_bytes, dstTensor, dst_gpus_per_domain,
                                                          node_local_rank, outDesc));
    } else {
      NCCL_M2N_CHECK(buildStagingTransferDescriptor(comm, srcBuffer, src_dims_bytes, ndims, srcTensor, dstBuffer,
                                                    dst_dims_bytes, dstTensor, src_gpus_per_domain, dst_gpus_per_domain,
                                                    node_local_rank, outDesc, splitStrided, splitNumInjectionDomains,
                                                    splitDomainsPerRep, nodeAnchorAtMeshStart));
    }
    return ncclSuccess;
  };

  auto finalizeCurrentStagingDescriptor = [&]() -> int {
    int activePeerCount = stagingComputeCtaHeuristicPeerCount(&desc, srcTensor, dstTensor);
    desc.ctaHeuristicPeerCount = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? activePeerCount : 0;
    if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
      desc.peerGroupSizeBound =
        stagingComputePeerGroupSizeBound(activePeerCount, dstTensor, dst_gpus_per_domain, copyAlgo);
    } else {
      const int dstNodeAnchor = splitMetaAvailable ? dstMesh->startRank : 0;
      desc.peerGroupSizeBound =
        stagingComputeHierarchyPeerGroupSizeBound(activePeerCount, dstTensor, dst_gpus_per_domain, dstNodeAnchor);
    }
    return stagingResolveNumChannelsForTransfer(&desc);
  };

  int requestedStagingNumCtas = stagingResolveNumChannelsForTransfer(nullptr);
  ReshardSplitComms splitComms{};
  const bool pipeMayAttemptSplit =
    copyAlgo == RESHARD_COPY_ALGO_PIPE && reshardShouldAttemptPipeSplitComms(srcTensor, dstTensor);
  if (pipeMayAttemptSplit) {
    const int srcRepCount = tensorRepCount(srcTensor);
    const int dstRepCount = tensorRepCount(dstTensor);
    const bool dstRepStrided =
      dstTensor->placements[1] == NCCL_RESHARD_REPLICATE && dstMesh->dims[0] > 1 && dstMesh->dims[1] > 1;
    NCCL_M2N_CHECK(reshardGetOrCreateSplitComms(comm, srcMesh, dstMesh, srcRepCount, dstRepCount, dstRepStrided,
                                                requestedStagingNumCtas, workStream, &splitComms));
    splitMetaAvailable = splitComms.valid && splitComms.active && splitComms.lsaSize > 0;

    if (splitMetaAvailable) {
      lsa_size_from_comm = splitComms.lsaSize;
      NCCL_M2N_CHECK(resolveReshardDomainSizes(world_rank, RESHARD_ALGO_RING, splitComms.srcLsaSize, splitComms.lsaSize,
                                               &src_gpus_per_domain, &dst_gpus_per_domain));
      if (reshardRankInMesh(dstMesh, world_rank)) {
        node_local_rank = (world_rank - dstMesh->startRank) % dst_gpus_per_domain;
      } else if (reshardRankInMesh(srcMesh, world_rank)) {
        node_local_rank = (world_rank - srcMesh->startRank) % src_gpus_per_domain;
      } else {
        node_local_rank = world_rank % dst_gpus_per_domain;
      }

      StagingTransferDescriptor splitDesc;
      NCCL_M2N_CHECK(buildCurrentStagingDescriptor(&splitDesc, splitComms.strided, splitComms.numInjectionDomains,
                                                   splitComms.domainsPerRep, true));
      desc = splitDesc;
      requestedStagingNumCtas = finalizeCurrentStagingDescriptor();
    }
  }

  if (copyAlgo == RESHARD_COPY_ALGO_PIPE && !splitMetaAvailable) {
    NCCL_M2N_CHECK(ensureParentDomainTopology());
    NCCL_M2N_CHECK(buildCurrentStagingDescriptor(&desc));
    requestedStagingNumCtas = finalizeCurrentStagingDescriptor();
  } else if (copyAlgo != RESHARD_COPY_ALGO_PIPE) {
    NCCL_M2N_CHECK(buildCurrentStagingDescriptor(&desc));
    requestedStagingNumCtas = finalizeCurrentStagingDescriptor();
  }

  const bool splitStagingActive = (copyAlgo == RESHARD_COPY_ALGO_PIPE) && splitMetaAvailable;
  const ReshardStagingMeshSignature stagingMeshSignature = stagingProcessMeshSignature(
    srcTensor, dstTensor, src_gpus_per_domain, dst_gpus_per_domain, splitStagingActive && splitComms.strided,
    splitStagingActive ? splitComms.numInjectionDomains : 0, splitStagingActive ? splitComms.domainsPerRep : 0);
  const ReshardStagingTensorSignature stagingTensorSignature = stagingProcessTensorSignature(srcTensor, dstTensor);
  const ReshardStagingChannelSignature stagingChannelSignature =
    stagingProcessChannelSignature(requestedStagingNumCtas);
  const int stagingControlSlotCount =
    (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? STAGING_PIPE_CONTROL_SLOTS : STAGING_DEFAULT_CONTROL_SLOTS;
  int persistentControlSlot = 0;
  bool persistentControlSlotValid = false;
  if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
    NCCL_M2N_CHECK(reshardGetOrCreatePersistentControlSlot(comm, stagingMeshSignature, stagingChannelSignature,
                                                           world_rank, &persistentControlSlot));
    NCCL_M2N_CHECK_ARG(persistentControlSlot >= 0 && persistentControlSlot < stagingControlSlotCount, world_rank,
                       "PIPE persistent control slot %d exceeds control slot count %d", persistentControlSlot,
                       stagingControlSlotCount);
    persistentControlSlotValid = true;
    desc.controlSlot = persistentControlSlot;
  }
  StagingBufferState* staging = nullptr;
  const StagingBufferConfig stagingConfig = stagingBufferConfigFromEnv();
  const int stagingPoolCapacityChannels = (copyAlgo == RESHARD_COPY_ALGO_PIPE && !stagingConfig.numChannelsExplicit) ?
                                            STAGING_MAX_CHANNELS :
                                            requestedStagingNumCtas;
  /* PIPE reuses one staging allocation per communicator. Its persistent
   * control slots are shared across split and parent launches so cursor
   * slices cannot alias when a model mixes the two paths. */
  const uint64_t stagingPoolKey = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ?
                                    stagingPipeReusablePoolKey() :
                                    stagingDirectReusablePoolKey(stagingPoolCapacityChannels);
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_ENSURE_BUFFER);
    NCCL_M2N_CHECK(ensureStagingBufferPool(comm, stagingPoolKey, workStream, requestedStagingNumCtas,
                                           stagingPoolCapacityChannels, stagingControlSlotCount, &staging));
  }
  const int staging_num_ctas = staging->numChannels;
  StagingBufferEventGuard stagingEvent(comm, stagingPoolKey, workStream);

  if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
    desc.pipeGinPeerCapacity = reshardGetPipeGinPeersPerSlot();
    desc.pipeGinChannelsPerPeer = reshardGetPipeGinChannelsPerPeer();
  }

  ncclWindow_t staging_window = nullptr;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_REGISTER_WINDOW);
    if (!splitStagingActive) {
      ncclWindow_t* cached_win =
        findCachedInternalWindowByPtr(comm, staging->buffer, staging->totalSize, RESHARD_INTERNAL_WINDOW_STAGING);
      if (cached_win != nullptr) {
        staging_window = *cached_win;
      } else {
        {
          M2nApiUnlock apiUnlock;
          NCCL_M2N_CHECK(ncclCommWindowRegister(comm, staging->buffer, staging->totalSize, &staging_window,
                                                NCCL_WIN_COLL_SYMMETRIC));
          NCCL_M2N_CHECK(m2nWaitCommReady(comm));
        }
        NCCL_M2N_CHECK(cacheInternalWindow(comm, staging->buffer, staging->totalSize, RESHARD_INTERNAL_WINDOW_STAGING,
                                           staging_window));
      }
    }
  }

  std::unique_ptr<StagingKernelParams> directParams;
  const int pipeTmaTileSize = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? getPipeTmaTileSize() : 0;
  StagingPipeCallParams pipeCall{};
  StagingPipePlanCacheEntry* pipePlanEntry = nullptr;
  const StagingKernelParams* pipeLaunchParams = nullptr;
  bool pipePlanCacheMiss = false;

  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_PREPARE_PARAMS);
    if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
      NCCL_M2N_CHECK(stagingGetPipePlanEntry(staging, stagingMeshSignature, stagingChannelSignature,
                                             stagingTensorSignature, persistentControlSlot, world_rank, &pipePlanEntry,
                                             &pipePlanCacheMiss));
      if (pipePlanCacheMiss) {
        NCCL_M2N_CHECK(stagingPrepareTransfer(staging, &desc, staging_window, staging_window,
                                              pipePlanEntry->hostParams));

        if (pipePlanEntry->hostParams->numLsaSources > 0) {
          size_t offsetDiff =
            pipePlanEntry->hostParams->lsaRegions[0].dataOffset - pipePlanEntry->hostParams->rdmaRegions[0].dataOffset;
          for (int ch = 0; ch < pipePlanEntry->hostParams->numChannels; ch++) {
            for (int s = 0; s < pipePlanEntry->hostParams->numLsaSources; s++) {
              if (!pipePlanEntry->hostParams->lsaSources[ch][s].active) {
                continue;
              }
              pipePlanEntry->hostParams->lsaSources[ch][s].fc.peerDataOffset -= offsetDiff;
            }
          }
        }
      }
      pipeLaunchParams = pipePlanEntry->hostParams;
      pipeCall.srcBuffer = srcBuffer;
      pipeCall.dstBuffer = dstBuffer;
      pipeCall.epoch = callEpoch;
      pipeCall.splitComm = false;
      pipeCall.splitCommBContextBase = 0;
      pipeCall.splitCommBContextCount = 0;
    } else {
      directParams.reset(new (std::nothrow) StagingKernelParams{});
      NCCL_M2N_CHECK_ARG(directParams != nullptr, world_rank, "ncclReshard: failed to allocate staging params");
      NCCL_M2N_CHECK(stagingPrepareTransfer(staging, &desc, staging_window, staging_window, directParams.get()));
      directParams->epoch = callEpoch;
    }
  }

  int resourceGinSignalCount = 0;
  int resourceGinCounterCount = 0;
  int activePipeRdmaPeers = 0;
  const int activeStagingChannels = (copyAlgo == RESHARD_COPY_ALGO_PIPE && pipeLaunchParams != nullptr) ?
                                      pipeLaunchParams->numChannels :
                                      directParams->numChannels;
  /* Buffer capacity is reusable allocation headroom. DevComm GIN resources
   * are collective and use the algorithm's rank-uniform resource plan. */
  const int resourceStagingChannels = activeStagingChannels;
  const int pipeGinPeersPerSlot = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? reshardGetPipeGinPeersPerSlot() : 0;
  const int pipeGinChannelsPerPeer = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? reshardGetPipeGinChannelsPerPeer() : 0;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_RESOLVE_GIN_COUNTS);
    ReshardMeshInterval srcInterval{};
    ReshardMeshInterval dstInterval{};
    NCCL_M2N_CHECK(computeReshardMeshInterval(srcMesh, world_rank, &srcInterval));
    NCCL_M2N_CHECK(computeReshardMeshInterval(dstMesh, world_rank, &dstInterval));
    size_t maxPeersBound = std::max<size_t>(1, maxPeerGroupSize);
    maxPeersBound = std::max(maxPeersBound, static_cast<size_t>(desc.peerGroupSizeBound));
    NCCL_M2N_CHECK_ARG(static_cast<size_t>(desc.numTargets) <= maxPeersBound &&
                         static_cast<size_t>(desc.numSources) <= maxPeersBound,
                       world_rank,
                       "ncclReshard: local staging peer count exceeds rank-uniform staging bound "
                       "(targets=%d sources=%d bound=%zu)",
                       desc.numTargets, desc.numSources, maxPeersBound);
    if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
      activePipeRdmaPeers = std::max(pipeLaunchParams->numRdmaTargets, pipeLaunchParams->numRdmaSources);
      NCCL_M2N_CHECK_ARG(activePipeRdmaPeers <= pipeGinPeersPerSlot, world_rank,
                         "ncclReshard: PIPE RDMA peer count %d exceeds fixed per-slot capacity %d; "
                         "set NCCL_RESHARD_PIPE_GIN_PEERS_PER_SLOT to a sufficient bound",
                         activePipeRdmaPeers, pipeGinPeersPerSlot);
      NCCL_M2N_CHECK_ARG(pipeGinChannelsPerPeer > 0, world_rank,
                         "ncclReshard: PIPE GIN channels-per-peer capacity must be positive");
      NCCL_M2N_CHECK(reshardComputeStagingGinCounts(world_rank, pipeGinChannelsPerPeer, pipeGinPeersPerSlot,
                                                    &resourceGinSignalCount, &resourceGinCounterCount));
    } else {
      NCCL_M2N_CHECK(reshardComputeStagingGinCounts(world_rank, activeStagingChannels, maxPeersBound,
                                                    &resourceGinSignalCount, &resourceGinCounterCount));
      directParams->ginSignalCount = resourceGinSignalCount;
      directParams->ginCounterCount = resourceGinCounterCount;
    }
  }

  ncclDevComm activeDevComm{};
  ncclDevComm* devCommPtr = nullptr;
  ReshardDevCommUse devCommUse;
  ncclDevComm splitDevCommA{};
  ncclDevComm splitDevCommB{};
  ReshardDevCommUse splitDevCommAUse;
  ReshardDevCommUse splitDevCommBUse;
  ncclDevComm launchDevCommA{};
  ncclDevComm launchDevCommB{};
  const int configuredGinContextCount = std::max(1, reshardGetGinContextCount());
  const int stagingGinContextCount =
    (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? configuredGinContextCount : staging_num_ctas;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_GET_DEV_COMM);
    if (splitStagingActive) {
      const int maxConcurrency = reshardGetSplitSlotCount();
      const int persistentControlSlots = reshardGetPersistentControlSlotCount();
      const int allocatedPersistentControlSlots = persistentControlSlots;
      NCCL_M2N_CHECK_ARG(persistentControlSlotValid, world_rank,
                         "ncclReshard: split PIPE persistent control slot was not initialized");
      NCCL_M2N_CHECK_ARG(persistentControlSlots > 0 && persistentControlSlot >= 0 &&
                           persistentControlSlot < persistentControlSlots,
                         world_rank, "ncclReshard: invalid persistent control slot %d/%d", persistentControlSlot,
                         persistentControlSlots);
      NCCL_M2N_CHECK_ARG(allocatedPersistentControlSlots > 0 &&
                           allocatedPersistentControlSlots <= persistentControlSlots &&
                           persistentControlSlot < allocatedPersistentControlSlots,
                         world_rank, "ncclReshard: invalid allocated persistent control-slot count %d for slot %d/%d",
                         allocatedPersistentControlSlots, persistentControlSlot, persistentControlSlots);
      const int signalsPerControlSlot = resourceGinSignalCount;
      const int countersPerControlSlot = resourceGinCounterCount;
      NCCL_M2N_CHECK_ARG(signalsPerControlSlot <= std::numeric_limits<int>::max() / allocatedPersistentControlSlots &&
                           countersPerControlSlot <= std::numeric_limits<int>::max() / allocatedPersistentControlSlots,
                         world_rank,
                         "ncclReshard: split PIPE persistent-control GIN resource count overflows int "
                         "(signalStride=%d counterStride=%d slots=%d)",
                         signalsPerControlSlot, countersPerControlSlot, allocatedPersistentControlSlots);
      const int ginSignalCountA = signalsPerControlSlot * allocatedPersistentControlSlots;
      const int ginCounterCountA = countersPerControlSlot * allocatedPersistentControlSlots;
      const int signalsPerSlotB = signalsPerControlSlot * allocatedPersistentControlSlots;
      const int countersPerSlotB = std::max(1, countersPerControlSlot * allocatedPersistentControlSlots);
      const int ctxPerSlotB = configuredGinContextCount;
      ncclWindow_t windowA = nullptr;
      ncclWindow_t windowB = nullptr;
      NCCL_M2N_CHECK(reshardSplitEnsureResources(
        &splitComms, staging->buffer, staging->totalSize, resourceStagingChannels, ginSignalCountA, ginCounterCountA,
        signalsPerSlotB, countersPerSlotB, ctxPerSlotB, maxConcurrency, workStream, &windowA, &windowB, &splitDevCommA,
        &splitDevCommAUse, &splitDevCommB, &splitDevCommBUse));
      if (pipePlanCacheMiss) {
        NCCL_M2N_CHECK(applySplitPipeParams(pipePlanEntry->hostParams, &splitComms, windowA, windowB,
                                            signalsPerControlSlot, countersPerControlSlot, signalsPerSlotB,
                                            countersPerSlotB, persistentControlSlot));
      }
      pipeCall.splitComm = true;
      pipeCall.splitCommBContextBase = splitComms.slotIdx * ctxPerSlotB;
      pipeCall.splitCommBContextCount = ctxPerSlotB;
      launchDevCommA = splitComms.inA ? splitDevCommA : splitDevCommB;
      launchDevCommB = splitComms.inB ? splitDevCommB : splitDevCommA;
    } else if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
      const int persistentControlSlots = STAGING_PIPE_CONTROL_SLOTS;
      NCCL_M2N_CHECK_ARG(resourceGinSignalCount <= std::numeric_limits<int>::max() / persistentControlSlots &&
                           resourceGinCounterCount <= std::numeric_limits<int>::max() / persistentControlSlots,
                         world_rank,
                         "ncclReshard: non-split PIPE persistent-control GIN resource count overflows int "
                         "(signalStride=%d counterStride=%d slots=%d)",
                         resourceGinSignalCount, resourceGinCounterCount, persistentControlSlots);
      const int totalGinSignalCount = resourceGinSignalCount * persistentControlSlots;
      const int totalGinCounterCount = resourceGinCounterCount * persistentControlSlots;
      NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, resourceStagingChannels, totalGinSignalCount, totalGinCounterCount,
                                               RESHARD_DEVCOMM_BARRIER_WORLD, stagingGinContextCount, workStream,
                                               &activeDevComm, &devCommUse));
      devCommPtr = &activeDevComm;
      if (pipePlanCacheMiss) {
        applyPipeRdmaOffsets(pipePlanEntry->hostParams, nullptr, persistentControlSlot * resourceGinSignalCount,
                             persistentControlSlot * resourceGinCounterCount, 0, 0);
      }
    } else {
      NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, resourceStagingChannels, resourceGinSignalCount,
                                               resourceGinCounterCount, RESHARD_DEVCOMM_BARRIER_WORLD,
                                               stagingGinContextCount, workStream, &activeDevComm, &devCommUse));
      devCommPtr = &activeDevComm;
    }

    if (copyAlgo == RESHARD_COPY_ALGO_PIPE && pipePlanCacheMiss) {
      NCCL_M2N_CHECK(stagingFinalizePipeHostPlan(pipePlanEntry, stagingMeshSignature, stagingChannelSignature,
                                                 stagingTensorSignature, world_rank));
      pipeLaunchParams = pipePlanEntry->hostParams;
    }
  }

  RESHARD_INFO(world_rank,
               "ncclReshard: copy_algo=%d num_ctas=%d staging_channels=%d resource_channels=%d "
               "gin_signal=%d gin_counter=%d "
               "gpus_per_domain=%d lsa_size=%d split_pipe=%d pipe_tma_tile=%d "
               "src_buf=%p dst_buf=%p staging=%p",
               (int)copyAlgo, staging_num_ctas, activeStagingChannels, resourceStagingChannels, resourceGinSignalCount,
               resourceGinCounterCount, dst_gpus_per_domain, lsa_size_from_comm, (int)splitStagingActive,
               pipeTmaTileSize, srcBuffer, dstBuffer, staging->buffer);

  if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
    NCCL_M2N_CUDACHECK(cudaMemcpyAsync(staging->devParams, directParams.get(), sizeof(*directParams),
                                       cudaMemcpyHostToDevice, workStream));
  }
  NCCL_M2N_CHECK(reshardStartWorkStream(stream, &work));

  ncclResult_t launchResult = ncclSuccess;
  auto launchPipe = [&](bool splitLaunch) -> ncclResult_t {
    ncclResult_t result = stagingEnsurePipeDevicePlan(pipePlanEntry, stagingMeshSignature, stagingChannelSignature,
                                                      stagingTensorSignature, workStream, world_rank);
    if (result == ncclSuccess) {
      if (splitLaunch) {
        result = launchStagingReshardPipeSplit(pipeLaunchParams, &pipeCall, pipePlanEntry->devParams,
                                               pipePlanEntry->devPlan, &launchDevCommA, &launchDevCommB,
                                               staging_num_ctas, pipeTmaTileSize, workStream);
      } else {
        result = launchStagingReshardPipe(pipeLaunchParams, &pipeCall, pipePlanEntry->devParams, pipePlanEntry->devPlan,
                                          devCommPtr, staging_num_ctas, pipeTmaTileSize, workStream);
      }
    }
    return result;
  };
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_LAUNCH_KERNEL);
    const bool verbose = debugLogging;
    if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
      launchResult = launchStagingReshardDirect(staging->devParams, devCommPtr, staging_num_ctas, workStream, verbose);
    } else if (splitStagingActive) {
      launchResult = launchPipe(true);
    } else {
      launchResult = launchPipe(false);
    }
  }
  if (splitStagingActive) {
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&splitDevCommAUse, workStream));
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&splitDevCommBUse, workStream));
  } else {
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&devCommUse, workStream));
  }

  if (launchResult != ncclSuccess) {
    NCCL_M2N_CHECK_WARN(stagingEvent.complete());
    NCCL_M2N_CHECK_WARN(workCompletion.complete());
    return launchResult;
  }

  if (profile != nullptr) {
    profile->log(world_rank);
  }

  /* Record event on the staging buffer for cross-stream ordering
   * (Change 3 — per-comm buffer with events). */
  NCCL_M2N_CHECK(stagingEvent.complete());

  NCCL_M2N_CHECK(workCompletion.complete());
#ifdef NCCL_M2N_TESTING
  gLastCompletedCopyAlgorithm.store(copyAlgo, std::memory_order_relaxed);
#endif

  return ncclSuccess;
}
