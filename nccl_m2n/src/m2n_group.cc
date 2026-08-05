/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstdio>
#include <new>
#include <utility>
#include <vector>

#include "m2n_checks.h"
#include "nccl_m2n.h"
#include "reshard_internal.h"

namespace {

struct M2nGroupEntry {
  size_t originalIndex = 0;
  M2nGroupReshardKind kind = M2nGroupReshardKind::Staging;
  ncclM2nHandle_t handle = nullptr;
  ncclComm_t comm = nullptr;
  ncclWindow_t window = nullptr;
  cudaStream_t stream = nullptr;
  bool hasSrc = false;
  bool hasDst = false;
  bool hasSrcMesh = false;
  bool hasDstMesh = false;
  ncclDistTensor_t src{};
  ncclDistTensor_t dst{};
  ncclMesh_t srcMesh{};
  ncclMesh_t dstMesh{};
};

struct M2nGroupState {
  bool active = false;
  size_t depth = 0;
  ncclResult_t error = ncclSuccess;
  char errorDetail[M2N_LAST_ERROR_BYTES] = {};
  std::vector<M2nGroupEntry> entries;
};

thread_local M2nGroupState gM2nGroupState;

void clearGroupState(M2nGroupState* state) {
  state->active = false;
  state->depth = 0;
  state->error = ncclSuccess;
  state->errorDetail[0] = '\0';
  state->entries.clear();
}

ncclResult_t failGroup(ncclResult_t result, const char* detail) {
  gM2nGroupState.error = result;
  (void)snprintf(gM2nGroupState.errorDetail, sizeof(gM2nGroupState.errorDetail), "%s", detail);
  m2nSetLastError(gM2nGroupState.errorDetail);
  RESHARD_WARN(-1, "%s", gM2nGroupState.errorDetail);
  return result;
}

cudaStream_t normalizeGroupStream(cudaStream_t stream) {
  return stream == nullptr || stream == cudaStreamLegacy ? nullptr : stream;
}

bool groupContextMatches(const M2nGroupEntry& first, const M2nGroupEntry& next) {
  return first.kind == next.kind && first.handle == next.handle && first.comm == next.comm &&
         first.window == next.window && normalizeGroupStream(first.stream) == normalizeGroupStream(next.stream);
}

ncclResult_t indexGroupError(ncclResult_t result, size_t originalIndex, size_t groupSize) {
  const char* detail = ncclM2nGetLastError();
  char currentDetail[M2N_LAST_ERROR_BYTES];
  (void)snprintf(currentDetail, sizeof(currentDetail), "%s", detail != nullptr ? detail : "");
  char indexedDetail[M2N_LAST_ERROR_BYTES];
  (void)snprintf(indexedDetail, sizeof(indexedDetail), "ncclM2nGroupEnd: entry %zu of %zu failed: %s", originalIndex,
                 groupSize, currentDetail);
  m2nSetLastError(indexedDetail);
  return result;
}

ncclResult_t partitionGroupEntries(std::vector<M2nGroupEntry>&& entries,
                                   std::vector<std::vector<M2nGroupEntry>>* buckets) {
  try {
    buckets->reserve(entries.size());
    for (M2nGroupEntry& entry : entries) {
      auto bucket = buckets->begin();
      // ponytail: linear lookup preserves first occurrence; add a context
      // index if large mixed groups become measurable.
      while (bucket != buckets->end() && !groupContextMatches(bucket->front(), entry)) {
        ++bucket;
      }
      if (bucket == buckets->end()) {
        buckets->emplace_back();
        bucket = buckets->end() - 1;
      }
      bucket->push_back(std::move(entry));
    }
  } catch (const std::bad_alloc&) {
    return failGroup(ncclSystemError, "ncclM2n group failed to allocate context bucket storage");
  }
  return ncclSuccess;
}

ncclResult_t executeGroupBucket(const std::vector<M2nGroupEntry>& entries, size_t groupSize) {
  bool canFuse = entries.size() > 1 && entries.size() <= kM2nGroupMaxFusionEntries &&
                 entries.front().kind == M2nGroupReshardKind::Staging;
  for (const M2nGroupEntry& entry : entries) {
    canFuse = canFuse && entry.hasSrc && entry.hasDst;
  }
  if (canFuse) {
    std::vector<ncclDistTensor_t> srcs;
    std::vector<ncclDistTensor_t> dsts;
    std::vector<size_t> originalIndices;
    try {
      srcs.resize(entries.size());
      dsts.resize(entries.size());
      originalIndices.resize(entries.size());
    } catch (const std::bad_alloc&) {
      return failGroup(ncclSystemError, "ncclM2n group failed to allocate fusion descriptor storage");
    }
    for (size_t i = 0; i < entries.size(); i++) {
      srcs[i] = entries[i].src;
      dsts[i] = entries[i].dst;
      originalIndices[i] = entries[i].originalIndex;
      srcs[i].mesh = entries[i].hasSrcMesh ? &entries[i].srcMesh : nullptr;
      dsts[i].mesh = entries[i].hasDstMesh ? &entries[i].dstMesh : nullptr;
    }

    bool handled = false;
    size_t failedOriginalIndex = entries.front().originalIndex;
    ncclResult_t result = reshardTryExecuteStagingGroup(entries.front().handle, entries.front().comm, srcs.data(),
                                                        dsts.data(), originalIndices.data(), entries.size(),
                                                        entries.front().stream, &handled, &failedOriginalIndex);
    if (result != ncclSuccess) {
      return indexGroupError(result, failedOriginalIndex, groupSize);
    }
    if (handled) {
      return ncclSuccess;
    }
  }

  for (const M2nGroupEntry& entry : entries) {
    ncclDistTensor_t src = entry.src;
    ncclDistTensor_t dst = entry.dst;
    src.mesh = entry.hasSrcMesh ? &entry.srcMesh : nullptr;
    dst.mesh = entry.hasDstMesh ? &entry.dstMesh : nullptr;
    const ncclDistTensor_t* srcPtr = entry.hasSrc ? &src : nullptr;
    const ncclDistTensor_t* dstPtr = entry.hasDst ? &dst : nullptr;
    ncclResult_t result = ncclSuccess;
    if (entry.kind == M2nGroupReshardKind::Window) {
      result = ncclReshardWithWindow(entry.handle, entry.comm, entry.window, srcPtr, dstPtr, entry.stream);
    } else {
      result = ncclReshard(entry.handle, entry.comm, srcPtr, dstPtr, entry.stream);
    }
    if (result != ncclSuccess) {
      return indexGroupError(result, entry.originalIndex, groupSize);
    }
  }
  return ncclSuccess;
}

} // namespace

bool m2nGroupIsActive() {
  return gM2nGroupState.active;
}

ncclResult_t m2nGroupEnqueueReshard(M2nGroupReshardKind kind, ncclM2nHandle_t handle, ncclComm_t comm,
                                    ncclWindow_t window, const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                    cudaStream_t stream) {
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(gM2nGroupState.active, -1, "m2nGroupEnqueueReshard: no group is active on this host thread");
  if (gM2nGroupState.error != ncclSuccess) {
    m2nSetLastError(gM2nGroupState.errorDetail);
    return gM2nGroupState.error;
  }

  M2nGroupEntry entry;
  entry.originalIndex = gM2nGroupState.entries.size();
  entry.kind = kind;
  entry.handle = handle;
  entry.comm = comm;
  entry.window = window;
  entry.stream = stream;
  entry.hasSrc = src != nullptr;
  entry.hasDst = dst != nullptr;
  if (entry.hasSrc) {
    entry.src = *src;
    entry.hasSrcMesh = src->mesh != nullptr;
    if (entry.hasSrcMesh) {
      entry.srcMesh = *src->mesh;
    }
  }
  if (entry.hasDst) {
    entry.dst = *dst;
    entry.hasDstMesh = dst->mesh != nullptr;
    if (entry.hasDstMesh) {
      entry.dstMesh = *dst->mesh;
    }
  }

  try {
    gM2nGroupState.entries.push_back(entry);
  } catch (const std::bad_alloc&) {
    return failGroup(ncclSystemError, "ncclM2n group failed to allocate descriptor storage");
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclM2nGroupStart(void) {
  m2nClearLastError();
  if (gM2nGroupState.active) {
    gM2nGroupState.depth++;
    return ncclSuccess;
  }
  clearGroupState(&gM2nGroupState);
  gM2nGroupState.active = true;
  gM2nGroupState.depth = 1;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclM2nGroupEnd(void) {
  m2nClearLastError();
  if (!gM2nGroupState.active) {
    NCCL_M2N_FAIL(ncclInvalidUsage, -1, "ncclM2nGroupEnd: no group is active on this host thread");
  }

  if (gM2nGroupState.depth > 1) {
    gM2nGroupState.depth--;
    return ncclSuccess;
  }

  if (gM2nGroupState.error != ncclSuccess) {
    const ncclResult_t result = gM2nGroupState.error;
    char detail[sizeof(gM2nGroupState.errorDetail)];
    (void)snprintf(detail, sizeof(detail), "%s", gM2nGroupState.errorDetail);
    clearGroupState(&gM2nGroupState);
    m2nSetLastError(detail);
    return result;
  }

  std::vector<M2nGroupEntry> entries;
  entries.swap(gM2nGroupState.entries);
  clearGroupState(&gM2nGroupState);

  const size_t groupSize = entries.size();
  std::vector<std::vector<M2nGroupEntry>> buckets;
  NCCL_M2N_CHECK(partitionGroupEntries(std::move(entries), &buckets));
  for (const std::vector<M2nGroupEntry>& bucket : buckets) {
    NCCL_M2N_CHECK(executeGroupBucket(bucket, groupSize));
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclM2nGroupAbort(void) {
  m2nClearLastError();
  clearGroupState(&gM2nGroupState);
  std::vector<M2nGroupEntry>().swap(gM2nGroupState.entries);
  return ncclSuccess;
}

