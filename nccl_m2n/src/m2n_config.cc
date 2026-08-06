/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*
 * Library configuration sources, in increasing precedence:
 *   1. Built-in defaults (the inline initializers in reshard_internal.h).
 *   2. The ncclM2nConfig_t copied into the first ncclM2nHandle_t in
 *      a process-lifetime init/finalize epoch.
 *   3. Environment variables (always win when set; honors the
 *      "env-overrides-everything" convention used elsewhere in NCCL).
 *
 * resetReshardRuntimeConfig(), applyReshardConfig(), and applyReshardEnv() are
 * called during the first runtime initialization in each epoch, in that order.
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "m2n_checks.h"
#include "m2n_env_parse.h"
#include "reshard_internal.h"
#include "m2n_log.h"
#include "reshard_types.h"

namespace {

bool parseAlgorithmEnv(const char* s, ReshardAlgorithm* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "AUTO") == 0) {
    *out = RESHARD_ALGO_AUTO;
    return true;
  }
  if (strcasecmp(s, "RING") == 0) {
    *out = RESHARD_ALGO_RING;
    return true;
  }
  if (strcasecmp(s, "DIRECT") == 0) {
    *out = RESHARD_ALGO_DIRECT;
    return true;
  }
  return false;
}

bool parseCopyAlgorithmEnv(const char* s, ReshardCopyAlgorithm* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "DIRECT") == 0) {
    *out = RESHARD_COPY_ALGO_DIRECT;
    return true;
  }
  if (strcasecmp(s, "PACKWINDOW") == 0) {
    *out = RESHARD_COPY_ALGO_PACKWINDOW;
    return true;
  }
  /* TMAPULL remains reserved until it has a dispatched implementation. */
  return false;
}

bool parseLbModeEnv(const char* s, ReshardLoadBalanceMode* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "UNIFORM") == 0) {
    *out = RESHARD_LB_UNIFORM;
    return true;
  }
  if (strcasecmp(s, "NODE_AWARE") == 0) {
    *out = RESHARD_LB_NODE_AWARE;
    return true;
  }
  return false;
}

bool parseBoolEnv(const char* s, bool* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "1") == 0 || strcasecmp(s, "true") == 0 || strcasecmp(s, "yes") == 0 || strcasecmp(s, "on") == 0) {
    *out = true;
    return true;
  }
  if (strcasecmp(s, "0") == 0 || strcasecmp(s, "false") == 0 || strcasecmp(s, "no") == 0 || strcasecmp(s, "off") == 0) {
    *out = false;
    return true;
  }
  return false;
}

} // namespace

void applyReshardConfig(const ncclM2nConfig_t* config) {
  if (config == nullptr) return;

  if (config->maxCta != NCCL_M2N_CONFIG_UNDEF_INT) {
    if (config->maxCta <= 0) RESHARD_WARN(-1, "ncclM2nInit: ignoring config.maxCta=%d (must be > 0).", config->maxCta);
    else gReshardMaxCta = config->maxCta;
  }
}

void resetReshardRuntimeConfig() {
  gReshardGpusPerNode = DEFAULT_GPUS_PER_NODE;
  gReshardSrcDomainSize = 0;
  gReshardDstDomainSize = 0;
  gReshardAlgorithm = RESHARD_ALGO_AUTO;
  gReshardLbMode = RESHARD_LB_UNIFORM;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PACKWINDOW;
  gReshardAutoUniformBcast = true;
  gReshardAutoUniformBcastSet = false;
  gReshardSplitComm = true;
  gReshardSplitCommSet = false;
  gReshardSplitSingleRepInject = false;
  gReshardSplitSingleRepInjectSet = false;
  gReshardSplitAutoParentThreshold = 200;
  gReshardCommBForceRail = false;
  gReshardLbModeSet = false;
  gReshardAdaptiveCallConfig = {};
  gReshardAdaptiveCallConfigValid = false;
  gReshardMaxCta = 0;
  gReshardNumCtasOverride = 0;
  gReshardNumCtas = DEFAULT_NUM_CTAS;
  gReshardElementsPerChunk = DEFAULT_ELEMENTS_PER_CHUNK;
  gReshardGinContextCount = DEFAULT_GIN_CONTEXT_COUNT;
  gReshardCrossDimTransposeThresholdBytes = CROSS_DIM_TRANSPOSE_THRESHOLD_BYTES;
  gReshardUseInternalStreams = true;
  gReshardChunkSizeBytes = 0;
  gReshardStagingWatermarkBytes = 256ULL * 1024ULL * 1024ULL;
  gReshardStagingBucketCount = 0;
}

ncclResult_t validateReshardConfigHeader(const ncclM2nConfig_t* config) {
  if (config == nullptr) return ncclSuccess;

  if (config->size != sizeof(ncclM2nConfig_t) || config->magic != NCCL_M2N_API_MAGIC) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "NCCL M2N config: rejecting malformed ncclM2nConfig_t "
                  "(size=%zu, magic=0x%x, version=%u). Use NCCL_M2N_CONFIG_INITIALIZER.",
                  config->size, config->magic, config->version);
  }

  if (config->version != NCCL_M2N_API_VERSION) {
    RESHARD_WARN(-1,
                 "ncclM2nInit: ncclM2nConfig_t.version=%u, library API_VERSION=%u; "
                 "behavior may differ across versions.",
                 config->version, NCCL_M2N_API_VERSION);
  }

  return ncclSuccess;
}

ncclResult_t resolveReshardDomainSizes(int worldRank, ReshardAlgorithm algo, int srcLsaSize, int dstLsaSize,
                                       int* srcGpusPerDomain, int* dstGpusPerDomain) {
  NCCL_M2N_CHECK_ARG(srcGpusPerDomain != nullptr && dstGpusPerDomain != nullptr, worldRank,
                     "resolveReshardDomainSizes: output pointers must be non-null");

  const int srcOverride = reshardGetSrcDomainSize();
  const int dstOverride = reshardGetDstDomainSize();
  if (algo == RESHARD_ALGO_RING) {
    NCCL_M2N_CHECK_ARG(dstOverride <= 0 || dstLsaSize <= 0 || dstOverride <= dstLsaSize, worldRank,
                       "NCCL_RESHARD_DST_DOMAIN_SIZE=%d exceeds the NCCL DevComm LSA team size %d", dstOverride,
                       dstLsaSize);
  }

  const int gpusPerNode = reshardGetGpusPerNode();
  *srcGpusPerDomain =
    srcOverride > 0 ? srcOverride : (srcLsaSize > 0 ? srcLsaSize : (gpusPerNode > 0 ? gpusPerNode : 1));
  *dstGpusPerDomain =
    dstOverride > 0 ? dstOverride : (dstLsaSize > 0 ? dstLsaSize : (gpusPerNode > 0 ? gpusPerNode : 1));
  return ncclSuccess;
}

// `getenv` is the only POSIX path to read process env vars; there is no portable
// thread-safe alternative (`secure_getenv` is glibc-only). Library init runs
// once on the first calling thread before ncclM2nInit returns, so concurrent
// env mutation by user code is the caller's problem — scope the
// concurrency-mt-unsafe suppression to just this function.
//
// NOLINTBEGIN(concurrency-mt-unsafe)
void applyReshardEnv() {
  ReshardLogLevel lvl;
  if (reshardLogLevelFromStr(getenv("NCCL_RESHARD_LOG_LEVEL"), &lvl)) reshardSetLogLevel(lvl);

  ReshardAlgorithm algo;
  if (parseAlgorithmEnv(getenv("NCCL_RESHARD_ALGORITHM"), &algo)) gReshardAlgorithm = algo;

  const char* copyAlgorithm = getenv("NCCL_RESHARD_COPY_ALGORITHM");
  ReshardCopyAlgorithm copyAlgo;
  if (copyAlgorithm != nullptr) {
    if (parseCopyAlgorithmEnv(copyAlgorithm, &copyAlgo)) {
      gReshardCopyAlgorithm = copyAlgo;
    } else {
      RESHARD_WARN(-1,
                   "NCCL_RESHARD_COPY_ALGORITHM=\"%s\" is not supported; accepted values are DIRECT and PACKWINDOW",
                   copyAlgorithm);
    }
  }

  ReshardLoadBalanceMode lb;
  if (parseLbModeEnv(getenv("NCCL_RESHARD_LB_MODE"), &lb)) {
    gReshardLbMode = lb;
    gReshardLbModeSet = true;
  }

  int n;
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_NUM_CTAS"), &n)) {
    gReshardNumCtasOverride = n;
    gReshardNumCtas = n;
  }
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_GIN_CONTEXT_COUNT"), &n)) {
    gReshardGinContextCount = n;
  }

  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_SRC_DOMAIN_SIZE"), &n)) gReshardSrcDomainSize = n;
  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_DST_DOMAIN_SIZE"), &n)) gReshardDstDomainSize = n;

  size_t sizeValue;
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_ELEMENTS_PER_CHUNK"), &sizeValue, false)) {
    gReshardElementsPerChunk = sizeValue;
  }

  /* Cache chunk-size override so prepareReshardParams doesn't touch
   * getenv on the hot path. */
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_CHUNK_SIZE"), &sizeValue, false)) {
    gReshardChunkSizeBytes = sizeValue;
  }

  if (parseM2nEnvSize(getenv("NCCL_RESHARD_CROSS_DIM_TRANSPOSE_THRESHOLD"), &sizeValue, true)) {
    gReshardCrossDimTransposeThresholdBytes = sizeValue;
  }
  if (parseM2nEnvSize(getenv("NCCL_RESHARD_STAGING_WATERMARK_BYTES"), &sizeValue, false)) {
    gReshardStagingWatermarkBytes = sizeValue;
  }

  bool useInternalStreams;
  if (parseBoolEnv(getenv("NCCL_RESHARD_USE_INTERNAL_STREAMS"), &useInternalStreams)) {
    gReshardUseInternalStreams = useInternalStreams;
  }

  bool splitComm;
  if (parseBoolEnv(getenv("NCCL_RESHARD_SPLIT_COMM"), &splitComm)) {
    gReshardSplitComm = splitComm;
    gReshardSplitCommSet = true;
  }

  bool singleRepInject;
  if (parseBoolEnv(getenv("NCCL_RESHARD_SPLIT_SINGLE_REP_INJECT"), &singleRepInject)) {
    gReshardSplitSingleRepInject = singleRepInject;
    gReshardSplitSingleRepInjectSet = true;
  }

  if (parseM2nPositiveEnvInt(getenv("NCCL_RESHARD_SPLIT_AUTO_PARENT_THRESHOLD"), &n)) {
    gReshardSplitAutoParentThreshold = n;
  }

  bool commBForceRail;
  if (parseBoolEnv(getenv("NCCL_RESHARD_COMMB_FORCE_RAIL"), &commBForceRail)) {
    gReshardCommBForceRail = commBForceRail;
  }

  bool autoUniformBcast;
  if (parseBoolEnv(getenv("NCCL_RESHARD_AUTO_UNIFORM_BCAST"), &autoUniformBcast)) {
    gReshardAutoUniformBcast = autoUniformBcast;
    gReshardAutoUniformBcastSet = true;
  }

  /* Optional bucketed staging pool: "size:slots,size:slots,..." (size in bytes).
   * Unset/empty/malformed -> count stays 0 -> per-comm staging (unchanged).
   * Parsed buckets are kept ascending by size so best-fit is a forward scan. */
  const char* buckets = getenv("NCCL_RESHARD_STAGING_BUCKETS");
  if (buckets != nullptr && buckets[0] != '\0') {
    gReshardStagingBucketCount = 0;
    bool valid = true;
    int totalSlots = 0;
    const char* p = buckets;
    while (*p != '\0') {
      if (gReshardStagingBucketCount >= kMaxStagingBuckets) {
        valid = false;
        break;
      }
      p = skipM2nEnvSpaces(p);
      if (!isM2nEnvDigit(*p)) {
        valid = false;
        break;
      }
      char* end = nullptr;
      errno = 0;
      unsigned long long sz = strtoull(p, &end, 10);
      if (errno == ERANGE || end == p || *end != ':' || sz > std::numeric_limits<size_t>::max()) {
        valid = false;
        break;
      }
      p = skipM2nEnvSpaces(end + 1);
      if (!isM2nEnvDigit(*p)) {
        valid = false;
        break;
      }
      errno = 0;
      long slots = strtol(p, &end, 10);
      if (errno == ERANGE || end == p || sz == 0 || slots <= 0 || slots > MAX_SPLIT_CONCURRENCY ||
          totalSlots > MAX_SPLIT_CONCURRENCY - slots) {
        valid = false;
        break;
      }
      gReshardStagingBuckets[gReshardStagingBucketCount].size = static_cast<size_t>(sz);
      gReshardStagingBuckets[gReshardStagingBucketCount].numSlots = static_cast<int>(slots);
      gReshardStagingBucketCount++;
      totalSlots += static_cast<int>(slots);
      p = end;
      if (*p == ',') {
        p++;
        if (*p == '\0') valid = false;
      } else if (*p != '\0') {
        valid = false;
        break;
      }
    }
    if (!valid) gReshardStagingBucketCount = 0;
    for (int i = 1; i < gReshardStagingBucketCount; i++) {
      ReshardStagingBucketCfg key = gReshardStagingBuckets[i];
      int j = i - 1;
      while (j >= 0 && gReshardStagingBuckets[j].size > key.size) {
        gReshardStagingBuckets[j + 1] = gReshardStagingBuckets[j];
        j--;
      }
      gReshardStagingBuckets[j + 1] = key;
    }
    for (int i = 1; i < gReshardStagingBucketCount; i++) {
      if (gReshardStagingBuckets[i - 1].size == gReshardStagingBuckets[i].size) {
        valid = false;
        gReshardStagingBucketCount = 0;
        break;
      }
    }
    if (valid && gReshardStagingBucketCount > 0) {
      RESHARD_INFO(-1, "staging-bucket pool enabled: %d buckets, %d total slots", gReshardStagingBucketCount,
                   reshardStagingTotalSlots());
    } else {
      RESHARD_WARN(-1, "NCCL_RESHARD_STAGING_BUCKETS=\"%s\" parsed to 0 buckets; staging-bucket pool disabled",
                   buckets);
    }
  }
}
// NOLINTEND(concurrency-mt-unsafe)
