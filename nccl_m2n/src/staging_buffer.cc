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
 * This TU has no CUDA-kernel code; cudaMemset / cudaMalloc are runtime
 * host APIs and link cleanly when compiled by the host C++ frontend.
 ************************************************************************/

#include "staging_buffer.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "reshard_internal.h"
#include "reshard_limits.h"
#include "m2n_log.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
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
static bool gStagingVerbose = false;

void stagingSetVerbose(bool verbose) {
  gStagingVerbose = verbose;
  if (verbose && reshardGetLogLevel() < RESHARD_LOG_DEBUG) {
    reshardSetLogLevel(RESHARD_LOG_DEBUG);
  }
}

#define STAGING_LOG(rank, fmt, ...) \
  do { \
    if (gStagingVerbose || reshardGetLogLevel() >= RESHARD_LOG_DEBUG) { \
      RESHARD_DEBUG((rank), "[STAGING] " fmt, ##__VA_ARGS__); \
    } \
  } while (0)

#define STAGING_NCCLCHECK(cmd) \
  do { \
    ncclResult_t r = (cmd); \
    if (r != ncclSuccess) { \
      fprintf(stderr, "[STAGING] NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      return r; \
    } \
  } while (0)

#define STAGING_CUDACHECK(cmd) \
  do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return ncclInternalError; \
    } \
  } while (0)

/* ======================================================================
 * Env helpers
 * ====================================================================*/

static bool parsePositiveInt(const char* value, int* out) {
  if (value == nullptr || out == nullptr) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  long parsed = strtol(value, &end, 10);
  if (end == value || errno == ERANGE || parsed <= 0 || parsed > INT_MAX) {
    return false;
  }
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
    end++;
  }
  if (*end != '\0') {
    return false;
  }
  *out = (int)parsed;
  return true;
}

static bool parseNonNegativeInt(const char* value, int* out) {
  if (value == nullptr || out == nullptr) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  long parsed = strtol(value, &end, 10);
  if (end == value || errno == ERANGE || parsed < 0 || parsed > INT_MAX) {
    return false;
  }
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
    end++;
  }
  if (*end != '\0') {
    return false;
  }
  *out = (int)parsed;
  return true;
}

static bool parsePositiveSize(const char* value, size_t* out) {
  if (value == nullptr || out == nullptr) {
    return false;
  }
  const char* scan = value;
  while (*scan != '\0' && std::isspace(static_cast<unsigned char>(*scan))) {
    scan++;
  }
  if (*scan == '-') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (end == value || errno == ERANGE || parsed == 0 ||
      parsed > (unsigned long long)std::numeric_limits<size_t>::max()) {
    return false;
  }
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
    end++;
  }
  if (*end != '\0') {
    return false;
  }
  *out = (size_t)parsed;
  return true;
}

static int readEnvNonNegativeInt(const char* name, int defaultVal) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time env read during staging pool creation.
  const char* val = getenv(name);
  int parsed = 0;
  if (parseNonNegativeInt(val, &parsed)) {
    return parsed;
  }
  return defaultVal;
}

static int stagingTargetCtasFromEnv(bool* found) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time env read during staging pool creation.
  const char* val = getenv("NCCL_RESHARD_STAGING_TARGET_CTAS");
  int parsed = 0;
  if (parsePositiveInt(val, &parsed)) {
    if (found != nullptr) {
      *found = true;
    }
    return parsed;
  }
  if (found != nullptr) {
    *found = false;
  }
  return 0;
}

static int stagingDefaultTargetCtasForPeerGroups(int peerGroupCount) {
  if (peerGroupCount <= 0) {
    return 6;
  }
  int ctasPerPeerGroup = 1;
  if (peerGroupCount <= 8) {
    ctasPerPeerGroup = 4;
  } else if (peerGroupCount <= 16) {
    ctasPerPeerGroup = 2;
  }
  return std::max(6, peerGroupCount * ctasPerPeerGroup);
}

static int readEnvOptionalInt(const char* name, bool* found) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time env read during staging pool creation.
  const char* val = getenv(name);
  int parsed = 0;
  if (parsePositiveInt(val, &parsed)) {
    if (found != nullptr) {
      *found = true;
    }
    return parsed;
  }
  if (found != nullptr) {
    *found = false;
  }
  return 0;
}

static size_t readEnvSize(const char* name, size_t defaultVal) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time env read during staging pool creation.
  const char* val = getenv(name);
  size_t parsed = 0;
  if (parsePositiveSize(val, &parsed)) {
    return parsed;
  }
  return defaultVal;
}

StagingBufferConfig stagingBufferConfigFromEnv() {
  StagingBufferConfig config{};
  config.numChannels = readEnvOptionalInt("NCCL_RESHARD_STAGING_NUM_CHANNELS", &config.numChannelsExplicit);
  config.channelSize = readEnvSize("NCCL_RESHARD_STAGING_CHANNEL_SIZE", STAGING_DEFAULT_CHANNEL_SIZE);
  config.chunkSize = readEnvSize("NCCL_RESHARD_STAGING_CHUNK_SIZE", STAGING_DEFAULT_CHUNK_SIZE);
  config.peersPerChannel = readEnvNonNegativeInt("NCCL_RESHARD_STAGING_PEERS_PER_CHANNEL", 1);
  return config;
}

static int clampNumChannels(int numChannels) {
  if (numChannels <= 0) {
    numChannels = STAGING_DEFAULT_NUM_CHANNELS;
  }
  if (numChannels > STAGING_MAX_CHANNELS) {
    RESHARD_WARN(-1, "[STAGING] numChannels %d exceeds STAGING_MAX_CHANNELS %d, clamping", numChannels,
                 STAGING_MAX_CHANNELS);
    numChannels = STAGING_MAX_CHANNELS;
  }
  return numChannels;
}

/* ======================================================================
 * stagingBufferInit
 * ====================================================================*/

static ncclResult_t stagingBufferInitInternal(StagingBufferState* state, int numChannelsOverride,
                                              int controlSlotCountOverride) {
  NCCL_M2N_CHECK_ARG(state != nullptr, -1, "[STAGING] stagingBufferInit called with null state");
  memset(state, 0, sizeof(*state));

  StagingBufferConfig config = stagingBufferConfigFromEnv();
  int numChannels = (numChannelsOverride > 0) ?
                      numChannelsOverride :
                      (config.numChannelsExplicit ? config.numChannels : STAGING_DEFAULT_NUM_CHANNELS);
  numChannels = clampNumChannels(numChannels);
  int controlSlotCount = (controlSlotCountOverride > 0) ? controlSlotCountOverride : STAGING_DEFAULT_CONTROL_SLOTS;
  size_t channelSize = config.channelSize;
  size_t chunkSize = config.chunkSize;
  int peersPerChannel = config.peersPerChannel;

  STAGING_LOG(-1, "stagingBufferInit() ENTRY");
  STAGING_LOG(-1,
              "  numChannels=%d%s channelSize=%zu (%zuMB) controlSlots=%d chunkSize=%zu (%zuKB) "
              "peersPerChannel=%d",
              numChannels, config.numChannelsExplicit ? " explicit" : "", channelSize, channelSize / (1024 * 1024),
              controlSlotCount, chunkSize, chunkSize / 1024, peersPerChannel);

  /* Each channel holds a control region + two equal data halves
   * (RDMA + LSA), so we need at least ctrl + 2*chunk bytes of room. */
  size_t chunkPairSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(2, chunkSize, &chunkPairSize), -1,
                     "[STAGING] chunkSize %zu overflows staging channel sizing", chunkSize);
  size_t ctrlRegionSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)controlSlotCount, (size_t)STAGING_CTRL_REGION_SIZE, &ctrlRegionSize), -1,
                     "[STAGING] controlSlotCount %d overflows staging channel sizing", controlSlotCount);
  size_t minChannelSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(ctrlRegionSize, chunkPairSize, &minChannelSize), -1,
                     "[STAGING] chunkSize %zu overflows staging channel sizing", chunkSize);
  NCCL_M2N_CHECK_ARG(channelSize >= minChannelSize, -1,
                     "[STAGING] channelSize %zu too small (min %zu with chunkSize %zu)", channelSize, minChannelSize,
                     chunkSize);

  size_t totalSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)numChannels, channelSize, &totalSize), -1,
                     "[STAGING] total staging size overflows: channels=%d channelSize=%zu", numChannels, channelSize);
  size_t dataRegionSize = (channelSize - ctrlRegionSize) / 2;
  size_t slotsPerRegionSize = dataRegionSize / chunkSize;
  NCCL_M2N_CHECK_ARG(slotsPerRegionSize <= (size_t)INT_MAX, -1, "[STAGING] slots/region %zu exceeds INT_MAX",
                     slotsPerRegionSize);
  int slotsPerRegion = (int)slotsPerRegionSize;

  STAGING_LOG(-1, "  per-channel: ctrl=%zuB rdma_data=%zuB lsa_data=%zuB slots=%d", ctrlRegionSize, dataRegionSize,
              dataRegionSize, slotsPerRegion);
  STAGING_LOG(-1, "  total alloc=%zu bytes (%zuMB)", totalSize, totalSize / (1024 * 1024));

  void* buffer = nullptr;
  STAGING_NCCLCHECK(ncclMemAlloc(&buffer, totalSize));

  /* From here on, state->initialized is still false, so a later
   * stagingBufferFinalize would early-out without reclaiming `buffer`.
   * Free it explicitly on any failure between here and `initialized=true`. */
  if (cudaError_t e = cudaMemset2D(buffer, channelSize, 0, ctrlRegionSize, numChannels); e != cudaSuccess) {
    fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    ncclMemFree(buffer);
    return ncclInternalError;
  }

  state->buffer = buffer;
  state->totalSize = totalSize;
  state->numChannels = numChannels;
  state->channelSize = channelSize;
  state->controlSlotCount = controlSlotCount;
  state->controlRegionSize = ctrlRegionSize;
  state->chunkSize = chunkSize;
  state->peersPerChannel = peersPerChannel;
  state->devParams = nullptr;
  state->pipePlanCacheNextVictim = 0;
  if (cudaError_t e = cudaMalloc(&state->devParams, sizeof(StagingKernelParams)); e != cudaSuccess) {
    fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    ncclMemFree(buffer);
    state->buffer = nullptr;
    return ncclInternalError;
  }
  state->initialized = true;

  RESHARD_INFO(-1,
               "[STAGING] init complete: %d channels x %zuMB = %zuMB total, "
               "chunkSize=%zuKB controlSlots=%d slots/region=%d peersPerChannel=%d",
               numChannels, channelSize / (1024 * 1024), totalSize / (1024 * 1024), chunkSize / 1024, controlSlotCount,
               slotsPerRegion, peersPerChannel);
  return ncclSuccess;
}

ncclResult_t stagingBufferInit(StagingBufferState* state) {
  return stagingBufferInitInternal(state, 0, STAGING_DEFAULT_CONTROL_SLOTS);
}

ncclResult_t stagingBufferInitWithNumChannels(StagingBufferState* state, int numChannels) {
  return stagingBufferInitInternal(state, numChannels, STAGING_DEFAULT_CONTROL_SLOTS);
}

ncclResult_t stagingBufferInitWithNumChannelsAndControlSlots(StagingBufferState* state, int numChannels,
                                                             int controlSlotCount) {
  return stagingBufferInitInternal(state, numChannels, controlSlotCount);
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
  /* Persistent-counter PIPE keeps LSA bases at zero and tracks progress
   * with per-edge cursors. */
  fc->lsaTailBase = 0;
  fc->lsaHeadBase = 0;
#ifdef STAGING_KERNEL_TRACE
  fprintf(stdout,
          "[FC_PROD] my=%d -> follower_world=%d follower_local=%d ch=%d "
          "ch_base=%zu my_target_idx=%d my_src_idx_on_dest=%d "
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
  /* Persistent-counter PIPE keeps LSA bases at zero and tracks progress
   * with per-edge cursors. */
  fc->lsaTailBase = 0;
  fc->lsaHeadBase = 0;
#ifdef STAGING_KERNEL_TRACE
  fprintf(stdout,
          "[FC_CONS] my=%d <- src_world=%d src_local=%d ch=%d "
          "ch_base=%zu my_source_idx=%d src_target_idx_for_me=%d "
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

struct StagingRdmaSignalLayout {
  bool dense;
  int channelsPerPeer;
};

static int stagingRdmaSignalId(const StagingRdmaSignalLayout& layout, int channelId, int channelRank, int peerIndex,
                               int peerCount) {
  /* PIPE translates the active edge to its fixed (peer, channel) slot. The
   * resulting signal ID is stored in StagingFlowCtrl, so kernels do not need
   * the translation metadata. */
  const int slot = layout.dense ? peerIndex * layout.channelsPerPeer + channelRank : channelId * peerCount + peerIndex;
  return slot * 2;
}

static void setFcRdmaSignals(StagingFlowCtrl* fc, const StagingRdmaSignalLayout& layout, int channelId, int channelRank,
                             int numPeers, int myLocalPeerIdx, int myRemotePeerIdx, int remoteNumPeers) {
  fc->useGinSignal = true;
  fc->isLocal = false;

  int localTailId = stagingRdmaSignalId(layout, channelId, channelRank, myLocalPeerIdx, numPeers);
  int localHeadId = localTailId + 1;
  fc->localTailSignal = (ncclGinSignal_t)localTailId;
  fc->localHeadSignal = (ncclGinSignal_t)localHeadId;

  int remoteTailId = stagingRdmaSignalId(layout, channelId, channelRank, myRemotePeerIdx, remoteNumPeers);
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

static int ceilDivInt(int numerator, int denominator) {
  if (denominator <= 0) {
    return 0;
  }
  return (numerator + denominator - 1) / denominator;
}

static int ceilDivSizeClamped(size_t numerator, size_t denominator, int clamp) {
  if (denominator == 0 || clamp <= 0) {
    return 0;
  }
  size_t quotient = numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
  if (quotient > (size_t)clamp) {
    return clamp;
  }
  return (int)quotient;
}

static void setPeerChunkRange(StagingPeerInfo* peer, size_t chunkSize) {
  if (peer == nullptr || chunkSize == 0 || peer->channelCount <= 0) {
    return;
  }
  const size_t totalChunks = (peer->plan.totalBytes + chunkSize - 1) / chunkSize;
  const size_t rank = (size_t)peer->channelRank;
  const size_t count = (size_t)peer->channelCount;
  peer->totalBytes = peer->plan.totalBytes;
  peer->chunkStart = (totalChunks * rank) / count;
  peer->chunkEnd = (totalChunks * (rank + 1)) / count;
}

static int positiveMod(int value, int modulus) {
  int result = value % modulus;
  return result < 0 ? result + modulus : result;
}

static int channelsInPeerGroup(int numChannels, int peerGroupCount, int group) {
  if (peerGroupCount <= 0 || group < 0 || group >= peerGroupCount || group >= numChannels) {
    return 0;
  }
  return ((numChannels - 1 - group) / peerGroupCount) + 1;
}

static int channelRankInPeerGroup(int channel, int peerGroupCount) {
  return peerGroupCount > 0 ? channel / peerGroupCount : channel;
}

static int channelPeerGroup(int channel, int peerGroupCount) {
  return peerGroupCount > 0 ? channel % peerGroupCount : 0;
}

static int edgePeerGroup(int sourceIdxOnDest, int targetIdxOnSource, int peerGroupCount) {
  if (peerGroupCount <= 1) {
    return 0;
  }
  return positiveMod(sourceIdxOnDest + targetIdxOnSource, peerGroupCount);
}

static bool channelHasEdge(int channel, int sourceIdxOnDest, int targetIdxOnSource, int peerGroupCount) {
  if (peerGroupCount <= 1) {
    return true;
  }
  return channelPeerGroup(channel, peerGroupCount) == edgePeerGroup(sourceIdxOnDest, targetIdxOnSource, peerGroupCount);
}

static int channelTargetIndexForSource(const StagingTransferDescriptor* desc, int sourceIdx) {
  if (sourceIdx >= 0 && sourceIdx < desc->numSources) {
    return desc->channelTargetIndexOnSource[sourceIdx];
  }
  return 0;
}

static int targetIndexForTarget(const StagingTransferDescriptor* desc, int targetIdx) {
  if (desc->isDest && !desc->isSource) {
    int sourceIdx = desc->sourceIndexOnDest[targetIdx];
    if (sourceIdx >= 0 && sourceIdx < desc->numSources) {
      return channelTargetIndexForSource(desc, sourceIdx);
    }
  }
  return targetIdx;
}

static void channelEdgeKeyForSource(const StagingTransferDescriptor* desc, int sourceIdx, int* sourceKey,
                                    int* targetKey) {
  if (sourceKey == nullptr || targetKey == nullptr) {
    return;
  }
  *sourceKey = sourceIdx;
  *targetKey = channelTargetIndexForSource(desc, sourceIdx);
}

static void channelEdgeKeyForTarget(const StagingTransferDescriptor* desc, int targetIdx, int* sourceKey,
                                    int* targetKey) {
  if (sourceKey == nullptr || targetKey == nullptr) {
    return;
  }
  *sourceKey = desc->sourceIndexOnDest[targetIdx];
  *targetKey = targetIndexForTarget(desc, targetIdx);
}

static int countSourcesForGroup(const StagingTransferDescriptor* desc, int group, int peerGroupCount) {
  int count = 0;
  for (int j = 0; j < desc->numSources; j++) {
    int sourceKey = 0;
    int targetKey = 0;
    channelEdgeKeyForSource(desc, j, &sourceKey, &targetKey);
    if (edgePeerGroup(sourceKey, targetKey, peerGroupCount) == group) {
      count++;
    }
  }
  return count;
}

static int countTargetsForGroup(const StagingTransferDescriptor* desc, int group, int peerGroupCount) {
  int count = 0;
  for (int j = 0; j < desc->numTargets; j++) {
    int sourceKey = 0;
    int targetKey = 0;
    channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
    if (edgePeerGroup(sourceKey, targetKey, peerGroupCount) == group) {
      count++;
    }
  }
  return count;
}

static size_t maxPeersInChannelGroup(const StagingTransferDescriptor* desc, int peerGroupCount) {
  size_t maxPeers = 1;
  for (int group = 0; group < peerGroupCount; group++) {
    maxPeers = std::max(maxPeers, (size_t)countSourcesForGroup(desc, group, peerGroupCount));
    maxPeers = std::max(maxPeers, (size_t)countTargetsForGroup(desc, group, peerGroupCount));
  }
  return maxPeers;
}

static int targetRankInGroup(const StagingTransferDescriptor* desc, int targetIdx, int group, int peerGroupCount) {
  int rank = 0;
  for (int j = 0; j < targetIdx; j++) {
    int sourceKey = 0;
    int targetKey = 0;
    channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
    if (edgePeerGroup(sourceKey, targetKey, peerGroupCount) == group) {
      rank++;
    }
  }
  return rank;
}

static int sourceRankInGroup(int sourceIdxOnDest, int targetIdxOnSource, int group, int peerGroupCount) {
  int rank = 0;
  for (int source = 0; source < sourceIdxOnDest; source++) {
    if (edgePeerGroup(source, targetIdxOnSource, peerGroupCount) == group) {
      rank++;
    }
  }
  return rank;
}

static size_t getMaxPeerGroupSize(const StagingTransferDescriptor* desc) {
  size_t maxPeers = 1;
  if (desc->peerGroupSizeBound > 0) {
    maxPeers = std::max(maxPeers, (size_t)desc->peerGroupSizeBound);
  }
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

int stagingResolveNumChannelsForTransfer(const StagingTransferDescriptor* desc) {
  StagingBufferConfig config = stagingBufferConfigFromEnv();
  if (config.numChannelsExplicit) {
    return clampNumChannels(config.numChannels);
  }
  if (desc != nullptr && config.peersPerChannel > 0) {
    size_t maxPeerGroupSize = getMaxPeerGroupSize(desc);
    int peerGroupCount = ceilDivInt((int)maxPeerGroupSize, config.peersPerChannel);
    int requested = peerGroupCount;
    if (desc->ctaHeuristicPeerCount > 0) {
      bool targetCtasExplicit = false;
      int targetCtas = stagingTargetCtasFromEnv(&targetCtasExplicit);
      if (!targetCtasExplicit) {
        targetCtas = stagingDefaultTargetCtasForPeerGroups(peerGroupCount);
      }
      int requestedCtas = std::max(peerGroupCount, targetCtas);
      if (desc->hasLocalFanout && desc->maxEdgeBytes > 0 && config.chunkSize > 0 &&
          config.channelSize > STAGING_CTRL_REGION_SIZE) {
        size_t dataRegionSize = (config.channelSize - STAGING_CTRL_REGION_SIZE) / 2;
        int maxChannelPeerCount = ceilDivInt((int)maxPeerGroupSize, peerGroupCount);
        size_t peerRegionSize = maxChannelPeerCount > 0 ? dataRegionSize / (size_t)maxChannelPeerCount : 0;
        size_t slotsPerPeer = peerRegionSize / config.chunkSize;
        if (slotsPerPeer > 0) {
          size_t maxEdgeChunks = 1 + (desc->maxEdgeBytes - 1) / config.chunkSize;
          int maxCtasPerPeerGroup = std::max(1, STAGING_MAX_CHANNELS / std::max(1, peerGroupCount));
          int creditSafeCtas = ceilDivSizeClamped(maxEdgeChunks, slotsPerPeer, maxCtasPerPeerGroup);
          requestedCtas = std::max(requestedCtas, peerGroupCount * creditSafeCtas);
        }
      }
      requested = requestedCtas;
    }
    return requested > STAGING_MAX_CHANNELS ? STAGING_MAX_CHANNELS : clampNumChannels(requested);
  }
  return clampNumChannels(STAGING_DEFAULT_NUM_CHANNELS);
}

/* ======================================================================
 * stagingPrepareTransfer
 * ====================================================================*/

ncclResult_t stagingPrepareTransfer(const StagingBufferState* state, const StagingTransferDescriptor* desc,
                                    ncclWindow_t rdmaWindow, ncclWindow_t lsaWindow, StagingKernelParams* params) {
  NCCL_M2N_CHECK_ARG(state != nullptr && state->initialized, -1,
                     "[STAGING] stagingPrepareTransfer called with uninitialized state");
  NCCL_M2N_CHECK_ARG(desc != nullptr, -1, "[STAGING] stagingPrepareTransfer called with null descriptor");
  NCCL_M2N_CHECK_ARG(params != nullptr, -1, "[STAGING] stagingPrepareTransfer called with null params");

  memset(params, 0, sizeof(*params));

  const int R = desc->myWorldRank;
  const int numChannels = state->numChannels;
  const size_t channelSize = state->channelSize;
  const size_t C = state->controlRegionSize;
  const int controlSlot = desc->controlSlot;
  if (numChannels <= 0) {
    RESHARD_WARN(R, "[STAGING] invalid numChannels=%d", numChannels);
    return ncclInvalidArgument;
  }
  NCCL_M2N_CHECK_ARG(state->chunkSize > 0 && channelSize > C, R,
                     "[STAGING] invalid staging sizes: channel=%zu ctrl=%zu chunk=%zu", channelSize, C,
                     state->chunkSize);
  NCCL_M2N_CHECK_ARG(state->controlSlotCount > 0 && controlSlot >= 0 && controlSlot < state->controlSlotCount, R,
                     "[STAGING] invalid controlSlot=%d for controlSlotCount=%d", controlSlot, state->controlSlotCount);
  const size_t dataRegionSize = (channelSize - C) / 2;

  NCCL_M2N_CHECK_ARG(desc->numTargets >= 0 && desc->numTargets <= MAX_TARGETS && desc->numSources >= 0 &&
                       desc->numSources <= MAX_SOURCES,
                     R, "[STAGING] invalid peer counts: targets=%d/%d sources=%d/%d", desc->numTargets, MAX_TARGETS,
                     desc->numSources, MAX_SOURCES);
  NCCL_M2N_CHECK_ARG(stagingLsaFollowersFitKernelCapacity(desc->numLsaFollowers), R,
                     "[STAGING] LSA fan-out followers exceed kernel capacity: followers=%d max=%d",
                     desc->numLsaFollowers, STAGING_LSA_FANOUT_MAX_FOLLOWERS);
  int rdmaSourceCount = 0;
  int lsaTargetCount = 0;
  for (int j = 0; j < desc->numTargets; j++) {
    if (!desc->targets[j].isRdma) {
      lsaTargetCount++;
    }
  }
  if (desc->isDest) {
    for (int j = 0; j < desc->numSources; j++) {
      if (desc->sources[j].isRdma) {
        rdmaSourceCount++;
      }
    }
  }
  NCCL_M2N_CHECK_ARG(stagingLsaFanoutFitsTargetCapacity(rdmaSourceCount, desc->numLsaFollowers), R,
                     "[STAGING] LSA fan-out target count exceeds staging target capacity: rdmaSources=%d "
                     "followers=%d max=%d",
                     rdmaSourceCount, desc->numLsaFollowers, MAX_TARGETS);
  NCCL_M2N_CHECK_ARG(stagingLsaFanoutHasTargetDescriptors(rdmaSourceCount, desc->numLsaFollowers, lsaTargetCount), R,
                     "[STAGING] LSA fan-out target list incomplete: rdmaSources=%d followers=%d lsaTargets=%d",
                     rdmaSourceCount, desc->numLsaFollowers, lsaTargetCount);

  const size_t maxPeerGroupSize = getMaxPeerGroupSize(desc);
  // Only PIPE kernels consume the compact peer-to-channel map. DIRECT uses
  // the channel-major staging map and needs every channel populated.
  const bool groupPeersByChannel = reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE && state->peersPerChannel > 0;
  int peerGroupCount = 1;
  size_t maxChannelPeerCount = maxPeerGroupSize;
  if (groupPeersByChannel) {
    peerGroupCount = ceilDivInt((int)maxPeerGroupSize, state->peersPerChannel);
    NCCL_M2N_CHECK_ARG(peerGroupCount > 0, R, "[STAGING] invalid peerGroupCount=0 for max peers %zu", maxPeerGroupSize);
    if (peerGroupCount > numChannels) {
      STAGING_LOG(R,
                  "peersPerChannel=%d requests %d peer groups for %zu peers, "
                  "capping to numChannels=%d",
                  state->peersPerChannel, peerGroupCount, maxPeerGroupSize, numChannels);
      peerGroupCount = numChannels;
    }
    maxChannelPeerCount =
      std::max((size_t)ceilDivInt((int)maxPeerGroupSize, peerGroupCount), maxPeersInChannelGroup(desc, peerGroupCount));
  }

  if (maxChannelPeerCount == 0) {
    RESHARD_WARN(R, "[STAGING] invalid maxChannelPeerCount=0");
    return ncclInvalidArgument;
  }
  const size_t maxChunkSize = dataRegionSize / maxChannelPeerCount;
  NCCL_M2N_CHECK_ARG(maxChunkSize > 0, R, "[STAGING] channel data region too small (%zu B / %zu peers)", dataRegionSize,
                     maxChannelPeerCount);
  size_t chunkSize = std::min(state->chunkSize, maxChunkSize);
  if (chunkSize != state->chunkSize) {
    STAGING_LOG(R,
                "  chunkSize adjusted: requested=%zu effective=%zu max_peer_group=%zu "
                "max_channel_peer_group=%zu data_region=%zu peersPerChannel=%d peer_groups=%d",
                state->chunkSize, chunkSize, maxPeerGroupSize, maxChannelPeerCount, dataRegionSize,
                state->peersPerChannel, peerGroupCount);
  }

  STAGING_LOG(R,
              "stagingPrepareTransfer() ENTRY is_src=%d is_dst=%d "
              "numTargets=%d numSources=%d numChannels=%d controlSlot=%d/%d peersPerChannel=%d peer_groups=%d",
              desc->isSource, desc->isDest, desc->numTargets, desc->numSources, numChannels, controlSlot,
              state->controlSlotCount, state->peersPerChannel, peerGroupCount);

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

  /* ----------------------------------------------------------------
   * 2. Signal/counter budgets.  GIN is RDMA-only: LSA fanout uses the
   * staging-buffer control region, so it must not inflate this stride.
   * ---------------------------------------------------------------- */
  int numRdmaPeers = std::max(numRdmaTargets, numRdmaSources);
  StagingRdmaSignalLayout rdmaSignalLayout{};
  if (desc->pipeGinPeerCapacity != 0 || desc->pipeGinChannelsPerPeer != 0) {
    NCCL_M2N_CHECK_ARG(desc->pipeGinPeerCapacity > 0 && desc->pipeGinChannelsPerPeer > 0, R,
                       "[STAGING] PIPE GIN map requires positive peer and channel capacities (peers=%d channels=%d)",
                       desc->pipeGinPeerCapacity, desc->pipeGinChannelsPerPeer);
    NCCL_M2N_CHECK_ARG(numRdmaPeers <= desc->pipeGinPeerCapacity, R,
                       "[STAGING] PIPE RDMA peer count %d exceeds GIN peer capacity %d", numRdmaPeers,
                       desc->pipeGinPeerCapacity);
    rdmaSignalLayout = {true, desc->pipeGinChannelsPerPeer};
  }
  const int ginPeerCount = rdmaSignalLayout.dense ? desc->pipeGinPeerCapacity : numRdmaPeers;
  const int ginChannelsPerPeer = rdmaSignalLayout.dense ? rdmaSignalLayout.channelsPerPeer : numChannels;
  const size_t ginCounterCount = (size_t)ginChannelsPerPeer * (size_t)ginPeerCount;
  NCCL_M2N_CHECK_ARG(ginCounterCount <= (size_t)std::numeric_limits<int>::max() / 2U, R,
                     "[STAGING] GIN map exceeds NCCL int capacity (peers=%d channels=%d)", ginPeerCount,
                     ginChannelsPerPeer);
  const size_t ginSignalCount = ginCounterCount * 2U;

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
  params->rdmaWindowB = rdmaWindow;
  params->lsaWindow = lsaWindow;
  params->chunkSize = chunkSize;
  params->ginSignalCount = (int)ginSignalCount;
  params->ginCounterCount = (int)ginCounterCount;
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
  const size_t groupedPeerSize = groupPeersByChannel ? dataRegionSize / maxChannelPeerCount : 0;
  const int groupedPeerSlots = groupPeersByChannel ? (int)(groupedPeerSize / chunkSize) : 0;
  if (groupPeersByChannel) {
    NCCL_M2N_CHECK_ARG(groupedPeerSlots >= 1, R,
                       "[STAGING] grouped peer sub-region too small (%zu B / %zu peers per channel, chunk=%zu)",
                       groupedPeerSize, maxChannelPeerCount, chunkSize);
  }
  int sourceNumSlots = (int)(dataRegionSize / chunkSize);

  /* ----------------------------------------------------------------
   * 5. Per-channel descriptors.
   * ---------------------------------------------------------------- */
  for (int ch = 0; ch < numChannels; ch++) {
    size_t channelBase = (size_t)ch * channelSize;
    size_t ctrlSlotBase = channelBase + (size_t)controlSlot * STAGING_CTRL_REGION_SIZE;
    size_t rdmaRegionStart = channelBase + C;
    size_t lsaRegionStart = channelBase + C + dataRegionSize;
    int peerGroup = groupPeersByChannel ? channelPeerGroup(ch, peerGroupCount) : 0;
    int channelPeerRank = groupPeersByChannel ? channelRankInPeerGroup(ch, peerGroupCount) : ch;
    int channelPeerCount =
      groupPeersByChannel ? channelsInPeerGroup(numChannels, peerGroupCount, peerGroup) : numChannels;
    if (rdmaSignalLayout.dense && numRdmaPeers > 0) {
      NCCL_M2N_CHECK_ARG(channelPeerCount <= rdmaSignalLayout.channelsPerPeer, R,
                         "[STAGING] PIPE channels per peer group %d exceeds GIN capacity %d", channelPeerCount,
                         rdmaSignalLayout.channelsPerPeer);
    }

    int activeSourceCount =
      groupPeersByChannel ? countSourcesForGroup(desc, peerGroup, peerGroupCount) : desc->numSources;
    size_t perSourceSize = 0;
    int perSourceSlots = 0;
    if (desc->isDest && desc->numSources > 0) {
      perSourceSize = groupPeersByChannel ? groupedPeerSize : dataRegionSize / desc->numSources;
      perSourceSlots = (int)(perSourceSize / chunkSize);
      NCCL_M2N_CHECK_ARG(perSourceSlots >= 1, R,
                         "[STAGING] per-source sub-region too small (%zu B / %d active sources, chunk=%zu)",
                         perSourceSize, activeSourceCount, chunkSize);
    }

    int activeTargetCount =
      groupPeersByChannel ? countTargetsForGroup(desc, peerGroup, peerGroupCount) : desc->numTargets;
    size_t perTargetSize = 0;
    int perTargetSlots = 0;
    if (desc->isSource && desc->numTargets > 0) {
      perTargetSize = groupPeersByChannel ? groupedPeerSize : dataRegionSize / desc->numTargets;
      perTargetSlots = (int)(perTargetSize / chunkSize);
      NCCL_M2N_CHECK_ARG(perTargetSlots >= 1, R,
                         "[STAGING] per-target sub-region too small (%zu B / %d active targets, chunk=%zu)",
                         perTargetSize, activeTargetCount, chunkSize);
    }

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
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForSource(desc, j, &sourceKey, &targetKey);
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int rdmaJ = sourceRdmaIdx[j];

        StagingPeerInfo* pi = &params->rdmaSources[ch][rdmaJ];
        pi->peerWorldRank = desc->sources[j].peerWorldRank;
        pi->peerShardIdx = desc->sources[j].peerShardIdx;
        pi->peerLocalRank = -1;
        pi->rdmaTransport = STAGING_RDMA_TRANSPORT_PARENT;
        pi->plan = desc->sources[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->sources[j].peerWorldRank;

        int edgePeerGroupId = edgePeerGroup(sourceKey, targetKey, peerGroupCount);
        int sourceRank =
          groupPeersByChannel ? sourceRankInGroup(sourceKey, targetKey, edgePeerGroupId, peerGroupCount) : j;
        NCCL_M2N_CHECK_ARG((size_t)sourceRank < maxChannelPeerCount, R,
                           "[STAGING] RDMA source rank %d exceeds grouped peer slots %zu", sourceRank,
                           maxChannelPeerCount);
        size_t peerDataOffset = rdmaRegionStart + (size_t)sourceRank * perSourceSize;
        setFcDataRegion(fc, peerDataOffset, perSourceSlots, chunkSize);

        int myTargetIdxOnSource = desc->rdmaTargetIndexOnSource[j];
        int srcNumTargets = desc->sourceNumRdmaTargets[j];
        NCCL_M2N_CHECK_ARG(myTargetIdxOnSource >= 0 && myTargetIdxOnSource < srcNumTargets, R,
                           "[STAGING] invalid remote RDMA target ordinal %d/%d for source %d", myTargetIdxOnSource,
                           srcNumTargets, j);

        setFcRdmaSignals(fc, rdmaSignalLayout, ch, channelPeerRank, numRdmaSources, rdmaJ, myTargetIdxOnSource,
                         srcNumTargets);
        fc->cursorHeadOffset = ctrlSlotBase + (size_t)j * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_HEAD;
      }

      for (int j = 0; j < desc->numSources; j++) {
        if (desc->sources[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForSource(desc, j, &sourceKey, &targetKey);
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int lsaJ = sourceLsaIdx[j];

        StagingPeerInfo* pi = &params->lsaSources[ch][lsaJ];
        pi->peerWorldRank = desc->sources[j].peerWorldRank;
        pi->peerShardIdx = desc->sources[j].peerShardIdx;
        pi->peerLocalRank = desc->sources[j].peerLocalRank;
        pi->plan = desc->sources[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->sources[j].peerLocalRank;

        int edgePeerGroupId = edgePeerGroup(sourceKey, targetKey, peerGroupCount);
        int sourceRank =
          groupPeersByChannel ? sourceRankInGroup(sourceKey, targetKey, edgePeerGroupId, peerGroupCount) : j;
        NCCL_M2N_CHECK_ARG((size_t)sourceRank < maxChannelPeerCount, R,
                           "[STAGING] LSA source rank %d exceeds grouped peer slots %zu", sourceRank,
                           maxChannelPeerCount);
        size_t peerDataOffset = lsaRegionStart + (size_t)sourceRank * perSourceSize;
        setFcDataRegion(fc, peerDataOffset, perSourceSlots, chunkSize);

        int sourceLsaHeadIdxForMe = desc->sourceLsaHeadIndexOnProvider[j];
        setFcLsaConsumer(fc, ctrlSlotBase, j, sourceLsaHeadIdxForMe, desc->myWorldRank, desc->sources[j].peerWorldRank,
                         ch);
        fc->cursorHeadOffset = ctrlSlotBase + (size_t)j * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_HEAD;
      }
    }

    /* ---------- 5b. Per-target flow control (src + dst fan-out) ---------- */
    if (desc->numTargets > 0) {
      for (int j = 0; j < desc->numTargets; j++) {
        if (!desc->targets[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
        int mySrcIdxOnDest = desc->sourceIndexOnDest[j];
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int rdmaJ = targetRdmaIdx[j];

        StagingPeerInfo* pi = &params->rdmaTargets[ch][rdmaJ];
        pi->peerWorldRank = desc->targets[j].peerWorldRank;
        pi->peerShardIdx = desc->targets[j].peerShardIdx;
        pi->peerLocalRank = -1;
        pi->rdmaTransport = STAGING_RDMA_TRANSPORT_PARENT;
        pi->plan = desc->targets[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->targets[j].peerWorldRank;

        int destNumSrc = desc->destNumSources[j];
        size_t remotePerSrcSize =
          groupPeersByChannel ? groupedPeerSize : ((destNumSrc > 0) ? dataRegionSize / destNumSrc : dataRegionSize);
        int remotePerSrcSlots = (int)(remotePerSrcSize / chunkSize);
        int edgePeerGroupId = edgePeerGroup(sourceKey, targetKey, peerGroupCount);
        int sourceRank = groupPeersByChannel ?
                           sourceRankInGroup(sourceKey, targetKey, edgePeerGroupId, peerGroupCount) :
                           mySrcIdxOnDest;
        NCCL_M2N_CHECK_ARG(remotePerSrcSlots >= 1, R, "[STAGING] remote RDMA sub-region too small (%zu B, chunk=%zu)",
                           remotePerSrcSize, chunkSize);
        NCCL_M2N_CHECK_ARG((size_t)sourceRank < maxChannelPeerCount, R,
                           "[STAGING] remote RDMA source rank %d exceeds grouped peer slots %zu", sourceRank,
                           maxChannelPeerCount);

        size_t peerDataOffset = rdmaRegionStart + (size_t)sourceRank * remotePerSrcSize;
        setFcDataRegion(fc, peerDataOffset, remotePerSrcSlots, chunkSize);

        int rdmaSrcIdxOnDest = desc->rdmaSourceIndexOnDest[j];
        int destNumRdmaSrc = desc->destNumRdmaSources[j];
        NCCL_M2N_CHECK_ARG(rdmaSrcIdxOnDest >= 0 && rdmaSrcIdxOnDest < destNumRdmaSrc, R,
                           "[STAGING] invalid remote RDMA source ordinal %d/%d for target %d", rdmaSrcIdxOnDest,
                           destNumRdmaSrc, j);
        setFcRdmaSignals(fc, rdmaSignalLayout, ch, channelPeerRank, numRdmaTargets, rdmaJ, rdmaSrcIdxOnDest,
                         destNumRdmaSrc);
        fc->localPutCounter = ch * numRdmaTargets + rdmaJ;
        fc->cursorTailOffset =
          ctrlSlotBase + (size_t)(STAGING_LOCAL_FC_BASE + j) * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_TAIL;
      }

      for (int j = 0; j < desc->numTargets; j++) {
        if (desc->targets[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
        int mySrcIdxOnDest = desc->sourceIndexOnDest[j];
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int lsaJ = targetLsaIdx[j];

        StagingPeerInfo* pi = &params->lsaTargets[ch][lsaJ];
        pi->peerWorldRank = desc->targets[j].peerWorldRank;
        pi->peerShardIdx = desc->targets[j].peerShardIdx;
        pi->peerLocalRank = desc->targets[j].peerLocalRank;
        pi->plan = desc->targets[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->targets[j].peerLocalRank;

        int destNumSrc = desc->destNumSources[j];
        size_t remotePerSrcSize =
          groupPeersByChannel ? groupedPeerSize : ((destNumSrc > 0) ? dataRegionSize / destNumSrc : dataRegionSize);
        int remotePerSrcSlots = (int)(remotePerSrcSize / chunkSize);
        int edgePeerGroupId = edgePeerGroup(sourceKey, targetKey, peerGroupCount);
        int sourceRank = groupPeersByChannel ?
                           sourceRankInGroup(sourceKey, targetKey, edgePeerGroupId, peerGroupCount) :
                           mySrcIdxOnDest;
        NCCL_M2N_CHECK_ARG(remotePerSrcSlots >= 1, R, "[STAGING] remote LSA sub-region too small (%zu B, chunk=%zu)",
                           remotePerSrcSize, chunkSize);
        NCCL_M2N_CHECK_ARG((size_t)sourceRank < maxChannelPeerCount, R,
                           "[STAGING] remote LSA source rank %d exceeds grouped peer slots %zu", sourceRank,
                           maxChannelPeerCount);

        size_t peerDataOffset = lsaRegionStart + (size_t)sourceRank * remotePerSrcSize;
        setFcDataRegion(fc, peerDataOffset, remotePerSrcSlots, chunkSize);

        setFcLsaProducer(fc, ctrlSlotBase, lsaJ, mySrcIdxOnDest, desc->myWorldRank, desc->targets[j].peerWorldRank, ch);
        fc->cursorTailOffset =
          ctrlSlotBase + (size_t)(STAGING_LOCAL_FC_BASE + j) * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_TAIL;
      }
    }

    /* ---------- 5c. Per-target local pipeline FC (source side only) ---------- */
    if (desc->isSource) {
      for (int j = 0; j < desc->numTargets; j++) {
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int targetRank = groupPeersByChannel ? targetRankInGroup(desc, j, peerGroup, peerGroupCount) : j;
        NCCL_M2N_CHECK_ARG((size_t)targetRank < maxChannelPeerCount, R,
                           "[STAGING] local target rank %d exceeds grouped peer slots %zu", targetRank,
                           maxChannelPeerCount);
        int ctrlIdx = STAGING_LOCAL_FC_BASE + j;

        if (desc->targets[j].isRdma) {
          int rdmaJ = targetRdmaIdx[j];
          StagingFlowCtrl* lrf = &params->localRdmaFc[ch][rdmaJ];
          initFlowCtrl(lrf);

          size_t targetDataOffset = rdmaRegionStart + (size_t)targetRank * perTargetSize;
          setFcLocalPipeline(lrf, desc->myWorldRank, ctrlSlotBase, ctrlIdx, targetDataOffset, perTargetSlots, chunkSize,
                             CTRL_FIELD_RDMA_TAIL, CTRL_FIELD_RDMA_HEAD);
          lrf->localPutCounter = ch * numRdmaTargets + rdmaJ;
        } else {
          int lsaJ = targetLsaIdx[j];
          StagingFlowCtrl* llf = &params->localLsaFc[ch][lsaJ];
          initFlowCtrl(llf);

          size_t targetDataOffset = lsaRegionStart + (size_t)targetRank * perTargetSize;
          setFcLocalPipeline(llf, desc->myWorldRank, ctrlSlotBase, ctrlIdx, targetDataOffset, perTargetSlots, chunkSize,
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
    STAGING_NCCLCHECK(ncclMemFree(state->buffer));
    state->buffer = nullptr;
  }
  if (state->devParams) {
    STAGING_CUDACHECK(cudaFree(state->devParams));
    state->devParams = nullptr;
  }
  for (int i = 0; i < STAGING_PIPE_CONTROL_SLOTS; i++) {
    StagingPipePlanCacheEntry& entry = state->pipePlanCache[i];
    delete entry.hostParams;
    entry.hostParams = nullptr;
    delete entry.hostPlan;
    entry.hostPlan = nullptr;
    if (entry.devParams) {
      STAGING_CUDACHECK(cudaFree(entry.devParams));
      entry.devParams = nullptr;
    }
    if (entry.devPlan) {
      STAGING_CUDACHECK(cudaFree(entry.devPlan));
      entry.devPlan = nullptr;
    }
    entry = {};
  }

  state->totalSize = 0;
  state->numChannels = 0;
  state->channelSize = 0;
  state->controlSlotCount = 0;
  state->controlRegionSize = 0;
  state->chunkSize = 0;
  state->peersPerChannel = 0;
  state->pipePlanCacheNextVictim = 0;
  state->initialized = false;
  return ncclSuccess;
}
