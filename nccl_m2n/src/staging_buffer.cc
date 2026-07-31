/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_buffer.cc — host-side implementation of the staging-buffer
 * lifecycle that backs ncclReshard.
 *
 * Logging is routed through the project-wide RESHARD_LOG facility and
 * keyed on desc->myWorldRank.  This TU has no CUDA-kernel code;
 * cudaMemset / cudaMalloc are runtime host APIs and link cleanly when
 * compiled by the host C++ frontend.
 ************************************************************************/

#include "staging_buffer.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "m2n_env_parse.h"
#include "reshard_limits.h"
#include "m2n_log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <cuda_runtime.h>
#include "nccl.h"

/* ======================================================================
 * Verbose-flag shim (preserves the public stagingSetVerbose API).  In
 * the new repo we mostly drive logging through reshardSetLogLevel, but
 * the staging-only knob is useful in the short-term while the kernel is
 * being stabilised.
 * ====================================================================*/
static bool g_staging_verbose = false;

void stagingSetVerbose(bool verbose) {
  g_staging_verbose = verbose;
  if (verbose && reshardGetLogLevel() < RESHARD_LOG_DEBUG) {
    reshardSetLogLevel(RESHARD_LOG_DEBUG);
  }
}

#define STAGING_LOG(rank, fmt, ...) \
  do { \
    if (g_staging_verbose || reshardGetLogLevel() >= RESHARD_LOG_DEBUG) { \
      RESHARD_DEBUG((rank), "[STAGING] " fmt, ##__VA_ARGS__); \
    } \
  } while (0)

/* ======================================================================
 * Env helpers
 * ====================================================================*/

static int readPositiveEnvInt(const char* name, int defaultVal) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time env read during staging pool creation.
  const char* value = getenv(name);
  int parsed = 0;
  if (parseM2nPositiveEnvInt(value, &parsed)) {
    return parsed;
  }
  return defaultVal;
}

static size_t readPositiveEnvSize(const char* name, size_t defaultVal) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time env read during staging pool creation.
  const char* value = getenv(name);
  size_t parsed = 0;
  if (parseM2nEnvSize(value, &parsed, false)) {
    return parsed;
  }
  return defaultVal;
}

StagingBufferConfig stagingBufferConfigFromEnv() {
  StagingBufferConfig config{};
  config.numChannels = readPositiveEnvInt("NCCL_RESHARD_STAGING_NUM_CHANNELS", STAGING_DEFAULT_NUM_CHANNELS);
  config.channelSize = readPositiveEnvSize("NCCL_RESHARD_STAGING_CHANNEL_SIZE", STAGING_DEFAULT_CHANNEL_SIZE);
  config.chunkSize = readPositiveEnvSize("NCCL_RESHARD_STAGING_CHUNK_SIZE", STAGING_DEFAULT_CHUNK_SIZE);
  return config;
}

/* ======================================================================
 * stagingBufferInit
 * ====================================================================*/

ncclResult_t stagingBufferInit(StagingBufferState* state) {
  NCCL_M2N_CHECK_ARG(state != nullptr, -1, "[STAGING] stagingBufferInit called with null state");
  memset(state, 0, sizeof(*state));

  StagingBufferConfig config = stagingBufferConfigFromEnv();
  int numChannels = config.numChannels;
  size_t channelSize = config.channelSize;
  size_t chunkSize = config.chunkSize;

  STAGING_LOG(-1, "stagingBufferInit() ENTRY");
  STAGING_LOG(-1, "  numChannels=%d channelSize=%zu (%zuMB) chunkSize=%zu (%zuKB)", numChannels, channelSize,
              channelSize / (1024 * 1024), chunkSize, chunkSize / 1024);

  if (numChannels > STAGING_MAX_CHANNELS) {
    RESHARD_WARN(-1, "[STAGING] numChannels %d exceeds STAGING_MAX_CHANNELS %d, clamping", numChannels,
                 STAGING_MAX_CHANNELS);
    numChannels = STAGING_MAX_CHANNELS;
  }

  /* Each channel holds a control region + two equal data halves
   * (RDMA + LSA), so we need at least ctrl + 2*chunk bytes of room. */
  size_t chunkPairSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(2, chunkSize, &chunkPairSize), -1,
                     "[STAGING] chunkSize %zu overflows staging channel sizing", chunkSize);
  size_t minChannelSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedAddSize((size_t)STAGING_CTRL_REGION_SIZE, chunkPairSize, &minChannelSize), -1,
                     "[STAGING] chunkSize %zu overflows staging channel sizing", chunkSize);
  NCCL_M2N_CHECK_ARG(channelSize >= minChannelSize, -1,
                     "[STAGING] channelSize %zu too small (min %zu with chunkSize %zu)", channelSize, minChannelSize,
                     chunkSize);

  size_t totalSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)numChannels, channelSize, &totalSize), -1,
                     "[STAGING] total staging size overflows: channels=%d channelSize=%zu", numChannels, channelSize);
  size_t ctrlRegionSize = STAGING_CTRL_REGION_SIZE;
  size_t dataRegionSize = (channelSize - ctrlRegionSize) / 2;
  size_t slotsPerRegionSize = dataRegionSize / chunkSize;
  NCCL_M2N_CHECK_ARG(slotsPerRegionSize <= (size_t)INT_MAX, -1, "[STAGING] slots/region %zu exceeds INT_MAX",
                     slotsPerRegionSize);
  int slotsPerRegion = (int)slotsPerRegionSize;

  STAGING_LOG(-1, "  per-channel: ctrl=%zuB rdma_data=%zuB lsa_data=%zuB slots=%d", ctrlRegionSize, dataRegionSize,
              dataRegionSize, slotsPerRegion);
  STAGING_LOG(-1, "  total alloc=%zu bytes (%zuMB)", totalSize, totalSize / (1024 * 1024));

  void* buffer = nullptr;
  NCCL_M2N_CHECK(ncclMemAlloc(&buffer, totalSize));

  /* From here on, state->initialized is still false, so a later
   * stagingBufferFinalize would early-out without reclaiming `buffer`.
   * Free it explicitly on any failure between here and `initialized=true`. */
  if (cudaError_t e = cudaMemset(buffer, 0, totalSize); e != cudaSuccess) {
    fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    ncclMemFree(buffer);
    return ncclInternalError;
  }

  state->buffer = buffer;
  state->totalSize = totalSize;
  state->numChannels = numChannels;
  state->channelSize = channelSize;
  state->chunkSize = chunkSize;
  state->devParams = nullptr;
  if (cudaError_t e = cudaMalloc(&state->devParams, sizeof(StagingKernelParams)); e != cudaSuccess) {
    fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    ncclMemFree(buffer);
    state->buffer = nullptr;
    return ncclInternalError;
  }
  state->initialized = true;

  RESHARD_INFO(-1,
               "[STAGING] init complete: %d channels x %zuMB = %zuMB total, "
               "chunkSize=%zuKB slots/region=%d",
               numChannels, channelSize / (1024 * 1024), totalSize / (1024 * 1024), chunkSize / 1024, slotsPerRegion);
  return ncclSuccess;
}

/* ======================================================================
 * stagingPrepareTransfer — local helpers
 * ====================================================================*/

static void initFlowCtrl(StagingFlowCtrl* fc) {
  memset(fc, 0, sizeof(*fc));
  fc->remoteRank = -1;
  fc->isLocal = false;
  fc->useGinSignal = false;
}

static void setFcDataRegion(StagingFlowCtrl* fc, size_t peerDataOffset, int peerNumSlots, size_t peerChunkSize) {
  fc->peerDataOffset = peerDataOffset;
  fc->peerNumSlots = peerNumSlots;
  fc->peerChunkSize = peerChunkSize;
}

static void setFcLsaProducer(StagingFlowCtrl* fc, size_t channelBase, int myTargetIdx, int mySourceIdxOnDest,
                             int myWorldRank, int peerWorldRank, int channelId) {
  fc->useGinSignal = false;
  fc->isLocal = false;
  fc->localHeadOffset = channelBase + (size_t)myTargetIdx * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_HEAD;
  fc->remoteTailOffset = channelBase + (size_t)mySourceIdxOnDest * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_TAIL;
  /* Bases default to 0. */
  fc->lsaTailBase = 0;
  fc->lsaHeadBase = 0;
#ifdef STAGING_KERNEL_TRACE
  fprintf(stdout,
          "[FC_PROD] my=%d -> follower_world=%d follower_local=%d ch=%d "
          "ch_base=%zu myTargetIdx=%d mySrcIdxOnDest=%d "
          "local_head_off=%zu remote_tail_off=%zu\n",
          myWorldRank, peerWorldRank, fc->remoteRank, channelId, channelBase, myTargetIdx, mySourceIdxOnDest,
          fc->localHeadOffset, fc->remoteTailOffset);
  fflush(stdout);
#else
  (void)myWorldRank;
  (void)peerWorldRank;
  (void)channelId;
#endif
}

static void setFcLsaConsumer(StagingFlowCtrl* fc, size_t channelBase, int mySourceIdx, int sourceTargetIdxForMe,
                             int myWorldRank, int peerWorldRank, int channelId) {
  fc->useGinSignal = false;
  fc->isLocal = false;
  fc->localTailOffset = channelBase + (size_t)mySourceIdx * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_TAIL;
  fc->remoteHeadOffset = channelBase + (size_t)sourceTargetIdxForMe * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_HEAD;
  /* Bases default to 0. */
  fc->lsaTailBase = 0;
  fc->lsaHeadBase = 0;
#ifdef STAGING_KERNEL_TRACE
  fprintf(stdout,
          "[FC_CONS] my=%d <- src_world=%d src_local=%d ch=%d "
          "ch_base=%zu mySourceIdx=%d src_target_idx_for_me=%d "
          "local_tail_off=%zu remote_head_off=%zu\n",
          myWorldRank, peerWorldRank, fc->remoteRank, channelId, channelBase, mySourceIdx, sourceTargetIdxForMe,
          fc->localTailOffset, fc->remoteHeadOffset);
  fflush(stdout);
#else
  (void)myWorldRank;
  (void)peerWorldRank;
  (void)channelId;
#endif
}

static void setFcRdmaSignals(StagingFlowCtrl* fc, int channelId, int numPeers, int myLocalPeerIdx, int myRemotePeerIdx,
                             int remoteNumPeers) {
  fc->useGinSignal = true;
  fc->isLocal = false;

  int localTailId = (channelId * numPeers + myLocalPeerIdx) * 2;
  int localHeadId = localTailId + 1;
  fc->localTailSignal = (ncclGinSignal_t)localTailId;
  fc->localHeadSignal = (ncclGinSignal_t)localHeadId;

  int remoteTailId = (channelId * remoteNumPeers + myRemotePeerIdx) * 2;
  int remoteHeadId = remoteTailId + 1;
  fc->remoteTailSignal = (ncclGinSignal_t)remoteTailId;
  fc->remoteHeadSignal = (ncclGinSignal_t)remoteHeadId;

  fc->tailSignalBase = 0;
  fc->headSignalBase = 0;
}

static void setFcLocalPipeline(StagingFlowCtrl* fc, int myRank, size_t channelBase, int ctrlEntryIndex,
                               size_t peerDataOffset, int peerNumSlots, size_t chunkSize, int ctrlFieldTail,
                               int ctrlFieldHead) {
  fc->remoteRank = myRank;
  fc->isLocal = true;
  fc->useGinSignal = false;

  size_t entryBase = channelBase + (size_t)ctrlEntryIndex * STAGING_CTRL_ENTRY_SIZE;

  fc->localTailOffset = entryBase + ctrlFieldTail;
  fc->localHeadOffset = entryBase + ctrlFieldHead;
  fc->remoteTailOffset = fc->localTailOffset;
  fc->remoteHeadOffset = fc->localHeadOffset;

  setFcDataRegion(fc, peerDataOffset, peerNumSlots, chunkSize);
}

static size_t getMaxPeerGroupSize(const StagingTransferDescriptor* desc) {
  size_t maxPeers = 1;
  if (desc->isDest && desc->numSources > 0) {
    maxPeers = std::max(maxPeers, (size_t)desc->numSources);
  }
  if (desc->isSource && desc->numTargets > 0) {
    maxPeers = std::max(maxPeers, (size_t)desc->numTargets);
  }
  for (int j = 0; j < desc->numTargets; j++) {
    if (desc->destNumSources[j] > 0) {
      maxPeers = std::max(maxPeers, (size_t)desc->destNumSources[j]);
    }
  }
  return maxPeers;
}

ncclResult_t stagingResolveEffectiveChunkSize(const StagingBufferState* state, size_t maxPeerGroupSize,
                                              size_t* effectiveChunkSize) {
  const bool stateReady = state != nullptr && state->initialized;
  NCCL_M2N_CHECK_ARG(stateReady, -1, "[STAGING] effective chunk requested with uninitialized state");
  NCCL_M2N_CHECK_ARG(effectiveChunkSize != nullptr, -1, "[STAGING] effective chunk output must be non-null");
  NCCL_M2N_CHECK_ARG(maxPeerGroupSize > 0, -1, "[STAGING] maximum peer group must be positive");

  const size_t ctrlSize = STAGING_CTRL_REGION_SIZE;
  NCCL_M2N_CHECK_ARG(state->chunkSize > 0 && state->channelSize > ctrlSize, -1,
                     "[STAGING] invalid staging sizes: channel=%zu ctrl=%zu chunk=%zu", state->channelSize, ctrlSize,
                     state->chunkSize);
  const size_t dataRegionSize = (state->channelSize - ctrlSize) / 2U;
  const size_t maxChunkSize = dataRegionSize / maxPeerGroupSize;
  NCCL_M2N_CHECK_ARG(maxChunkSize > 0, -1, "[STAGING] channel data region too small (%zu B / %zu peers)",
                     dataRegionSize, maxPeerGroupSize);
  *effectiveChunkSize = std::min(state->chunkSize, maxChunkSize);
  return ncclSuccess;
}

/* ======================================================================
 * stagingPrepareTransfer
 * ====================================================================*/

ncclResult_t stagingPrepareTransfer(const StagingBufferState* state, const StagingTransferDescriptor* desc,
                                    ncclWindow_t rdmaWindow, ncclWindow_t lsaWindow, size_t effectiveChunkSize,
                                    StagingKernelParams* params) {
  const bool stateReady = state != nullptr && state->initialized;
  NCCL_M2N_CHECK_ARG(stateReady, -1, "[STAGING] stagingPrepareTransfer called with uninitialized state");
  NCCL_M2N_CHECK_ARG(desc != nullptr, -1, "[STAGING] stagingPrepareTransfer called with null descriptor");
  NCCL_M2N_CHECK_ARG(params != nullptr, -1, "[STAGING] stagingPrepareTransfer called with null params");

  memset(params, 0, sizeof(*params));

  const int R = desc->myWorldRank;
  const int numChannels = state->numChannels;
  const size_t channelSize = state->channelSize;
  const size_t C = STAGING_CTRL_REGION_SIZE;
  const bool validStagingSizes = state->chunkSize > 0 && channelSize > C;
  NCCL_M2N_CHECK_ARG(validStagingSizes, R, "[STAGING] invalid staging sizes: channel=%zu ctrl=%zu chunk=%zu",
                     channelSize, C, state->chunkSize);
  const size_t dataRegionSize = (channelSize - C) / 2;

  const bool validPeerCounts = desc->numTargets >= 0 && desc->numTargets <= MAX_TARGETS && desc->numSources >= 0 &&
                               desc->numSources <= MAX_SOURCES;
  NCCL_M2N_CHECK_ARG(validPeerCounts, R, "[STAGING] invalid peer counts: targets=%d/%d sources=%d/%d", desc->numTargets,
                     MAX_TARGETS, desc->numSources, MAX_SOURCES);

  const size_t localMaxPeerGroupSize = getMaxPeerGroupSize(desc);
  const size_t localMaxChunkSize = dataRegionSize / localMaxPeerGroupSize;
  NCCL_M2N_CHECK_ARG(
    effectiveChunkSize > 0 && effectiveChunkSize <= state->chunkSize && effectiveChunkSize <= localMaxChunkSize, R,
    "[STAGING] effective chunk %zu is incompatible with local geometry "
    "(configured=%zu localMax=%zu peers=%zu)",
    effectiveChunkSize, state->chunkSize, localMaxChunkSize, localMaxPeerGroupSize);
  const size_t chunkSize = effectiveChunkSize;

  STAGING_LOG(R,
              "stagingPrepareTransfer() ENTRY is_src=%d is_dst=%d "
              "numTargets=%d numSources=%d numChannels=%d",
              desc->isSource, desc->isDest, desc->numTargets, desc->numSources, numChannels);

  /* ----------------------------------------------------------------
   * 1. Classify peers into RDMA / LSA buckets.
   * ---------------------------------------------------------------- */
  int numRdmaTargets = 0;
  int numLsaTargets = 0;
  int numRdmaSources = 0;
  int numLsaSources = 0;

  int targetRdmaIdx[MAX_TARGETS];
  int targetLsaIdx[MAX_TARGETS];
  int sourceRdmaIdx[MAX_SOURCES];
  int sourceLsaIdx[MAX_SOURCES];

  memset(targetRdmaIdx, -1, sizeof(targetRdmaIdx));
  memset(targetLsaIdx, -1, sizeof(targetLsaIdx));
  memset(sourceRdmaIdx, -1, sizeof(sourceRdmaIdx));
  memset(sourceLsaIdx, -1, sizeof(sourceLsaIdx));

  for (int j = 0; j < desc->numTargets; j++) {
    if (desc->targets[j].isRdma) {
      targetRdmaIdx[j] = numRdmaTargets++;
    } else {
      targetLsaIdx[j] = numLsaTargets++;
    }
  }

  if (desc->isDest) {
    for (int j = 0; j < desc->numSources; j++) {
      if (desc->sources[j].isRdma) {
        sourceRdmaIdx[j] = numRdmaSources++;
      } else {
        sourceLsaIdx[j] = numLsaSources++;
      }
    }
  }

  STAGING_LOG(R, "  classify: rdma_t=%d lsa_t=%d rdma_s=%d lsa_s=%d", numRdmaTargets, numLsaTargets, numRdmaSources,
              numLsaSources);

  NCCL_M2N_CHECK_ARG(stagingLsaFollowersFitLaneCapacity(desc->numLsaFollowers), R,
                     "[STAGING] LSA fan-out followers exceed kernel lane capacity: followers=%d max=%d",
                     desc->numLsaFollowers, STAGING_LSA_FANOUT_MAX_FOLLOWERS);
  NCCL_M2N_CHECK_ARG(stagingLsaFanoutFitsTargetCapacity(numRdmaSources, desc->numLsaFollowers), R,
                     "[STAGING] LSA fan-out target count exceeds staging target capacity: rdmaSources=%d "
                     "followers=%d max=%d",
                     numRdmaSources, desc->numLsaFollowers, MAX_TARGETS);
  NCCL_M2N_CHECK_ARG(stagingLsaFanoutHasTargetDescriptors(numRdmaSources, desc->numLsaFollowers, numLsaTargets), R,
                     "[STAGING] LSA fan-out descriptor list is incomplete: rdmaSources=%d followers=%d "
                     "lsaTargets=%d",
                     numRdmaSources, desc->numLsaFollowers, numLsaTargets);

  /* ----------------------------------------------------------------
   * 2. Signal/counter budgets — overall-index based so both sides
   *    derive the same IDs without an exchange.
   * ---------------------------------------------------------------- */
  int numAllPeers = std::max(desc->numTargets, desc->numSources);
  int ginSignalCount = numChannels * numAllPeers * 2;
  int ginCounterCount = numChannels * numAllPeers;

  /* ----------------------------------------------------------------
   * 3. Top-level params.
   * ---------------------------------------------------------------- */
  params->numChannels = numChannels;
  params->myRank = desc->myWorldRank;
  params->myLocalRank = desc->myLocalRank;
  params->isSource = desc->isSource;
  params->isDest = desc->isDest;
  params->srcBuffer = desc->srcBuffer;
  params->dstBuffer = desc->dstBuffer;
  params->stagingBuffer = state->buffer;
  params->rdmaWindow = rdmaWindow;
  params->lsaWindow = lsaWindow;
  params->chunkSize = chunkSize;
  params->ginSignalCount = ginSignalCount;
  params->ginCounterCount = ginCounterCount;
  params->numRdmaTargets = numRdmaTargets;
  params->numLsaTargets = numLsaTargets;
  params->numRdmaSources = numRdmaSources;
  params->numLsaSources = numLsaSources;
  params->numLsaFollowers = desc->numLsaFollowers;
  params->numRingTargets = desc->numRingTargets;
  params->ndims = desc->ndims;

  for (int d = 0; d < desc->ndims; d++) {
    params->srcDims[d] = desc->srcDims[d];
    params->dstDims[d] = desc->dstDims[d];
    params->srcStrides[d] = desc->srcStrides[d];
    params->dstStrides[d] = desc->dstStrides[d];
  }

  /* ----------------------------------------------------------------
   * 4. Per-source / per-target sub-region sizing.
   * ---------------------------------------------------------------- */
  size_t perSourceSize = 0;
  int perSourceSlots = 0;
  if (desc->isDest && desc->numSources > 0) {
    perSourceSize = dataRegionSize / desc->numSources;
    perSourceSlots = (int)(perSourceSize / chunkSize);
    NCCL_M2N_CHECK_ARG(perSourceSlots >= 1, R,
                       "[STAGING] per-source sub-region too small (%zu B / %d sources, chunk=%zu)", perSourceSize,
                       desc->numSources, chunkSize);
  }

  size_t perTargetSize = 0;
  int perTargetSlots = 0;
  if (desc->isSource && desc->numTargets > 0) {
    perTargetSize = dataRegionSize / desc->numTargets;
    perTargetSlots = (int)(perTargetSize / chunkSize);
    NCCL_M2N_CHECK_ARG(perTargetSlots >= 1, R,
                       "[STAGING] per-target sub-region too small (%zu B / %d targets, chunk=%zu)", perTargetSize,
                       desc->numTargets, chunkSize);
  }
  int sourceNumSlots = (int)(dataRegionSize / chunkSize);

  /* ----------------------------------------------------------------
   * 5. Per-channel descriptors.
   * ---------------------------------------------------------------- */
  for (int ch = 0; ch < numChannels; ch++) {
    size_t channelBase = (size_t)ch * channelSize;
    size_t rdmaRegionStart = channelBase + C;
    size_t lsaRegionStart = channelBase + C + dataRegionSize;

    StagingRegion* rdmaReg = &params->rdmaRegions[ch];
    rdmaReg->dataOffset = rdmaRegionStart;
    rdmaReg->regionSize = dataRegionSize;
    rdmaReg->chunkSize = chunkSize;
    rdmaReg->numSlots = sourceNumSlots;
    rdmaReg->window = rdmaWindow;

    StagingRegion* lsaReg = &params->lsaRegions[ch];
    lsaReg->dataOffset = lsaRegionStart;
    lsaReg->regionSize = dataRegionSize;
    lsaReg->chunkSize = chunkSize;
    lsaReg->numSlots = sourceNumSlots;
    lsaReg->window = lsaWindow;

    /* ---------- 5a. Dest-side per-source flow control ---------- */
    if (desc->isDest) {
      for (int j = 0; j < desc->numSources; j++) {
        if (!desc->sources[j].isRdma) {
          continue;
        }
        int rdmaJ = sourceRdmaIdx[j];

        StagingPeerInfo* pi = &params->rdmaSources[ch][rdmaJ];
        pi->peerWorldRank = desc->sources[j].peerWorldRank;
        pi->peerShardIdx = desc->sources[j].peerShardIdx;
        pi->peerLocalRank = -1;
        pi->plan = desc->sources[j].plan;

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->sources[j].peerWorldRank;

        size_t peerDataOff = rdmaRegionStart + (size_t)j * perSourceSize;
        setFcDataRegion(fc, peerDataOff, perSourceSlots, chunkSize);

        int myTargetIdxOnSource = desc->targetIndexOnSource[j];
        int srcNumTargets = desc->sourceNumTargets[j];

        setFcRdmaSignals(fc, ch, desc->numSources, j, myTargetIdxOnSource, srcNumTargets);
      }

      for (int j = 0; j < desc->numSources; j++) {
        if (desc->sources[j].isRdma) {
          continue;
        }
        int lsaJ = sourceLsaIdx[j];

        StagingPeerInfo* pi = &params->lsaSources[ch][lsaJ];
        pi->peerWorldRank = desc->sources[j].peerWorldRank;
        pi->peerShardIdx = desc->sources[j].peerShardIdx;
        pi->peerLocalRank = desc->sources[j].peerLocalRank;
        pi->plan = desc->sources[j].plan;

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->sources[j].peerLocalRank;

        size_t peerDataOff = lsaRegionStart + (size_t)j * perSourceSize;
        setFcDataRegion(fc, peerDataOff, perSourceSlots, chunkSize);

        int sourceTargetIdxForMe = desc->targetIndexOnSource[j];
        setFcLsaConsumer(fc, channelBase, j, sourceTargetIdxForMe, desc->myWorldRank, desc->sources[j].peerWorldRank,
                         ch);
      }
    }

    /* ---------- 5b. Per-target flow control (src + dst fan-out) ---------- */
    if (desc->numTargets > 0) {
      for (int j = 0; j < desc->numTargets; j++) {
        if (!desc->targets[j].isRdma) {
          continue;
        }
        int rdmaJ = targetRdmaIdx[j];

        StagingPeerInfo* pi = &params->rdmaTargets[ch][rdmaJ];
        pi->peerWorldRank = desc->targets[j].peerWorldRank;
        pi->peerShardIdx = desc->targets[j].peerShardIdx;
        pi->peerLocalRank = -1;
        pi->plan = desc->targets[j].plan;

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->targets[j].peerWorldRank;

        int destNumSrc = desc->destNumSources[j];
        size_t remotePerSrcSize = (destNumSrc > 0) ? dataRegionSize / destNumSrc : dataRegionSize;
        int remotePerSrcSlots = (int)(remotePerSrcSize / chunkSize);
        int mySrcIdxOnDest = desc->sourceIndexOnDest[j];

        size_t peerDataOff = rdmaRegionStart + (size_t)mySrcIdxOnDest * remotePerSrcSize;
        setFcDataRegion(fc, peerDataOff, remotePerSrcSlots, chunkSize);

        setFcRdmaSignals(fc, ch, desc->numTargets, j, mySrcIdxOnDest, destNumSrc);
        fc->localPutCounter = ch * numRdmaTargets + rdmaJ;
      }

      for (int j = 0; j < desc->numTargets; j++) {
        if (desc->targets[j].isRdma) {
          continue;
        }
        int lsaJ = targetLsaIdx[j];

        StagingPeerInfo* pi = &params->lsaTargets[ch][lsaJ];
        pi->peerWorldRank = desc->targets[j].peerWorldRank;
        pi->peerShardIdx = desc->targets[j].peerShardIdx;
        pi->peerLocalRank = desc->targets[j].peerLocalRank;
        pi->plan = desc->targets[j].plan;

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->targets[j].peerLocalRank;

        int destNumSrc = desc->destNumSources[j];
        size_t remotePerSrcSize = (destNumSrc > 0) ? dataRegionSize / destNumSrc : dataRegionSize;
        int remotePerSrcSlots = (int)(remotePerSrcSize / chunkSize);
        int mySrcIdxOnDest = desc->sourceIndexOnDest[j];

        size_t peerDataOff = lsaRegionStart + (size_t)mySrcIdxOnDest * remotePerSrcSize;
        setFcDataRegion(fc, peerDataOff, remotePerSrcSlots, chunkSize);

        setFcLsaProducer(fc, channelBase, j, mySrcIdxOnDest, desc->myWorldRank, desc->targets[j].peerWorldRank, ch);
      }
    }

    /* ---------- 5c. Per-target local pipeline FC (source side only) ---------- */
    if (desc->isSource) {
      for (int j = 0; j < desc->numTargets; j++) {
        int ctrlIdx = STAGING_LOCAL_FC_BASE + j;

        if (desc->targets[j].isRdma) {
          int rdmaJ = targetRdmaIdx[j];
          StagingFlowCtrl* lrf = &params->localRdmaFc[ch][rdmaJ];
          initFlowCtrl(lrf);

          size_t targetDataOff = rdmaRegionStart + (size_t)j * perTargetSize;
          setFcLocalPipeline(lrf, desc->myWorldRank, channelBase, ctrlIdx, targetDataOff, perTargetSlots, chunkSize,
                             CTRL_FIELD_RDMA_TAIL, CTRL_FIELD_RDMA_HEAD);
          lrf->localPutCounter = ch * numRdmaTargets + rdmaJ;
        } else {
          int lsaJ = targetLsaIdx[j];
          StagingFlowCtrl* llf = &params->localLsaFc[ch][lsaJ];
          initFlowCtrl(llf);

          size_t targetDataOff = lsaRegionStart + (size_t)j * perTargetSize;
          setFcLocalPipeline(llf, desc->myWorldRank, channelBase, ctrlIdx, targetDataOff, perTargetSlots, chunkSize,
                             CTRL_FIELD_LSA_TAIL, CTRL_FIELD_LSA_HEAD);
        }
      }
    }
  }

  STAGING_LOG(R,
              "stagingPrepareTransfer() EXIT (success): "
              "rdma_t=%d lsa_t=%d rdma_s=%d lsa_s=%d gin_sig=%d gin_cnt=%d ring_t=%d",
              numRdmaTargets, numLsaTargets, numRdmaSources, numLsaSources, ginSignalCount, ginCounterCount,
              desc->numRingTargets);
  return ncclSuccess;
}

/* ======================================================================
 * stagingBufferFinalize
 * ====================================================================*/

ncclResult_t stagingBufferFinalize(StagingBufferState* state) {
  if (!state || !state->initialized) {
    return ncclSuccess;
  }

  STAGING_LOG(-1, "stagingBufferFinalize() ENTRY buffer=%p total=%zu", state->buffer, state->totalSize);

  if (state->buffer) {
    NCCL_M2N_CHECK(ncclMemFree(state->buffer));
    state->buffer = nullptr;
  }
  if (state->devParams) {
    NCCL_M2N_CUDACHECK(cudaFree(state->devParams));
    state->devParams = nullptr;
  }

  state->totalSize = 0;
  state->numChannels = 0;
  state->channelSize = 0;
  state->chunkSize = 0;
  state->initialized = false;
  return ncclSuccess;
}
