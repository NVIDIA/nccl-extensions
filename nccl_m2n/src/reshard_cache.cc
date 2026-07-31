/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <vector>
#include "nccl.h"

// Host-only TU: pull host-visible types/decls from nccl_device.h. Scope
// the macro to this include so it doesn't leak to subsequent headers.
#define NCCL_HOSTLIB_ONLY
#include "nccl_device.h"
#undef NCCL_HOSTLIB_ONLY

#include "reshard_types.h"
#include "m2n_log.h"
#include "m2n_checks.h"
#include "reshard_internal.h"
#include "staging_buffer.h"

struct DevCommCacheEntry {
  ncclComm_t comm;
  int numCtas;
  int ginSignalCount;
  cudaStream_t stream;
  bool valid;
  ncclDevComm devComm;
};

/* One pool entry per (comm, dev) — a non-blocking stream paired with
 * caller-to-work readiness and work-to-caller completion events.  The events
 * are reused across calls so we avoid cudaEvent{Create,Destroy} on every
 * reshard. */
struct StreamPoolEntry {
  ncclComm_t comm;
  int dev;
  cudaStream_t stream = nullptr;
  cudaEvent_t readyEvent = nullptr;
  cudaEvent_t doneEvent = nullptr;
};

static WindowCache gInternalWindowCache = {};
static DevCommCacheEntry gDevcommCache[MAX_DEVCOMM_CACHE_ENTRIES];
static int gDevcommCacheCount = 0;
static int gDevcommCacheNextIdx = 0;

/* CUDA handle lifetimes are tied to the last finalize call in the
 * process-lifetime init/finalize epoch (so we can destroy them before the CUDA
 * context tears down) — see cacheFinalize.
 * The vector itself just owns memory; handles inside it are released
 * explicitly there before the vector is cleared. */
static std::vector<StreamPoolEntry> gStreamPool;

static ncclWindow_t* findCachedWindowByPtr(WindowCache* cache, ncclComm_t comm, void* buffer, size_t size) {
  for (int i = 0; i < cache->count; i++) {
    WindowCacheEntry& e = cache->entries[i];
    if (e.valid && e.comm == comm && e.windowBuffer == buffer && e.windowSize == size) return &e.window;
  }
  return nullptr;
}

static ncclResult_t cacheWindow(WindowCache* cache, ncclComm_t comm, void* windowBuffer, size_t windowSize,
                                ncclWindow_t window) {
  int idx;
  if (cache->count >= MAX_WINDOW_CACHE_ENTRIES) {
    idx = cache->nextIdx;
    WindowCacheEntry& old = cache->entries[idx];
    RESHARD_WARN(-1, "Window cache full (%d entries), replacing entry at index %d", MAX_WINDOW_CACHE_ENTRIES, idx);
    if (old.valid) NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(old.comm, old.window));
    cache->nextIdx = (cache->nextIdx + 1) % MAX_WINDOW_CACHE_ENTRIES;
  } else {
    idx = cache->count++;
  }
  WindowCacheEntry& e = cache->entries[idx];
  e.comm = comm;
  e.windowBuffer = windowBuffer;
  e.windowSize = windowSize;
  e.window = window;
  e.valid = true;
  return ncclSuccess;
}

ncclWindow_t* findCachedInternalWindowByPtr(ncclComm_t comm, void* buffer, size_t size) {
  return findCachedWindowByPtr(&gInternalWindowCache, comm, buffer, size);
}

ncclResult_t cacheInternalWindow(ncclComm_t comm, void* buffer, size_t size, ncclWindow_t window) {
  return cacheWindow(&gInternalWindowCache, comm, buffer, size, window);
}

ncclDevComm* findCachedDevComm(ncclComm_t comm, int numCtas, int signalCount, cudaStream_t stream) {
  for (int i = 0; i < gDevcommCacheCount; i++) {
    DevCommCacheEntry& e = gDevcommCache[i];
    if (e.valid && e.comm == comm && e.numCtas == numCtas && e.ginSignalCount == signalCount && e.stream == stream)
      return &e.devComm;
  }
  return nullptr;
}

ncclResult_t cacheDevComm(ncclComm_t comm, int numCtas, int signalCount, const ncclDevComm* devComm,
                          cudaStream_t stream) {
  int idx;
  if (gDevcommCacheCount >= MAX_DEVCOMM_CACHE_ENTRIES) {
    idx = gDevcommCacheNextIdx;
    DevCommCacheEntry& old = gDevcommCache[idx];
    RESHARD_WARN(-1, "DevComm cache full (%d entries), replacing entry at index %d", MAX_DEVCOMM_CACHE_ENTRIES, idx);
    if (old.valid) NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(old.comm, &old.devComm));
    gDevcommCacheNextIdx = (gDevcommCacheNextIdx + 1) % MAX_DEVCOMM_CACHE_ENTRIES;
  } else {
    idx = gDevcommCacheCount++;
  }
  DevCommCacheEntry& e = gDevcommCache[idx];
  e.comm = comm;
  e.numCtas = numCtas;
  e.ginSignalCount = signalCount;
  e.stream = stream;
  e.devComm = *devComm;
  e.valid = true;
  return ncclSuccess;
}

/* ======================================================================
 * Staging buffer pool -- per-comm entries, mirroring TransposeBufferEntry.
 * ====================================================================*/

static StagingBufferPoolEntry gStagingPool[MAX_STAGING_BUFFER_ENTRIES];
static int gStagingPoolCount = 0;

static StagingBufferPoolEntry* findStagingPoolEntry(ncclComm_t comm) {
  for (int i = 0; i < gStagingPoolCount; i++) {
    if (gStagingPool[i].comm == comm && gStagingPool[i].allocated) {
      return &gStagingPool[i];
    }
  }
  return nullptr;
}

ncclResult_t ensureStagingBufferPool(ncclComm_t comm, cudaStream_t stream, StagingBufferState** outState) {
  NCCL_M2N_CHECK_ARG(outState != nullptr, -1, "ensureStagingBufferPool called with null outState");
  *outState = nullptr;

  StagingBufferPoolEntry* entry = findStagingPoolEntry(comm);
  if (entry != nullptr) {
    if (entry->stream != stream) {
      NCCL_M2N_CHECK_ARG(entry->event != nullptr, -1, "Staging buffer pool entry has no ordering event");
      NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, entry->event, 0));
      entry->stream = stream;
    }
    *outState = entry->state;
    return ncclSuccess;
  }
  if (gStagingPoolCount >= MAX_STAGING_BUFFER_ENTRIES) {
    RESHARD_WARN(-1, "Staging buffer pool full (%d entries)", MAX_STAGING_BUFFER_ENTRIES);
    return ncclSystemError;
  }

  cudaEvent_t event = nullptr;
  NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

  std::unique_ptr<StagingBufferState> state(new (std::nothrow) StagingBufferState());
  if (state == nullptr) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    RESHARD_WARN(-1, "Failed to allocate staging buffer state");
    return ncclSystemError;
  }

  ncclResult_t r = stagingBufferInit(state.get());
  if (r != ncclSuccess) {
    stagingBufferFinalize(state.get()); // reclaim anything init allocated before failing
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    return r;
  }

  StagingBufferPoolEntry& e = gStagingPool[gStagingPoolCount++];
  e.comm = comm;
  e.stream = stream;
  e.event = event;
  e.state = state.release();
  e.allocated = true;
  RESHARD_DEBUG(-1, "Staging buffer pool: allocated for comm %p (%zu bytes) [slot %d]", (void*)comm,
                e.state->totalSize, gStagingPoolCount - 1);
  *outState = e.state;
  return ncclSuccess;
}

ncclResult_t stagingBufferPoolRecordEvent(ncclComm_t comm, cudaStream_t stream) {
  StagingBufferPoolEntry* e = findStagingPoolEntry(comm);
  if (e != nullptr) {
    NCCL_M2N_CUDACHECK(cudaEventRecord(e->event, stream));
  }
  return ncclSuccess;
}

void stagingBufferPoolFinalize() {
  for (int i = 0; i < gStagingPoolCount; i++) {
    if (gStagingPool[i].event != nullptr) {
      // Sync before destroy/free: a finalize racing an in-flight staging kernel
      // would otherwise free the buffer under a running kernel (use-after-free).
      // Warn-only: teardown must not abort.
      NCCL_M2N_CUDACHECK_WARN(cudaEventSynchronize(gStagingPool[i].event));
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(gStagingPool[i].event));
    }
    if (gStagingPool[i].state != nullptr) {
      stagingBufferFinalize(gStagingPool[i].state);
      delete gStagingPool[i].state;
    }
    gStagingPool[i] = {};
  }
  gStagingPoolCount = 0;
}

void cacheFinalize() {
  for (int i = 0; i < gInternalWindowCache.count; i++) {
    WindowCacheEntry& e = gInternalWindowCache.entries[i];
    if (e.valid) {
      NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(e.comm, e.window));
      e.valid = false;
    }
  }
  gInternalWindowCache.count = 0;
  gInternalWindowCache.nextIdx = 0;

  for (int i = 0; i < gDevcommCacheCount; i++) {
    DevCommCacheEntry& e = gDevcommCache[i];
    if (e.valid) {
      NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(e.comm, &e.devComm));
      e.valid = false;
    }
  }
  gDevcommCacheCount = 0;

  for (StreamPoolEntry& e : gStreamPool) {
    if (e.doneEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(e.doneEvent));
    }
    if (e.readyEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(e.readyEvent));
    }
    if (e.stream != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(e.stream));
    }
  }
  gStreamPool.clear();

  stagingBufferPoolFinalize();
}

ncclResult_t streamPoolAcquire(ncclComm_t comm, int dev, cudaStream_t* outStream, cudaEvent_t* outReadyEvent,
                               cudaEvent_t* outDoneEvent) {
  if (outStream == nullptr || outReadyEvent == nullptr || outDoneEvent == nullptr) {
    return ncclInvalidArgument;
  }
  /* Pool disabled (NCCL_RESHARD_STREAM_POOL_SIZE <= 0) — caller
   * should have gated on reshardGetStreamPoolSize() > 0; defend
   * anyway so a forgotten gate doesn't UB. */
  const int maxEntries = reshardGetStreamPoolSize();
  if (maxEntries <= 0) {
    return ncclInvalidArgument;
  }

  /* Find existing entry for (comm, dev). */
  for (StreamPoolEntry& e : gStreamPool) {
    if (e.comm == comm && e.dev == dev) {
      *outStream = e.stream;
      *outReadyEvent = e.readyEvent;
      *outDoneEvent = e.doneEvent;
      return ncclSuccess;
    }
  }
  /* Pool full — soft fall-through.  Caller checks *outStream ==
   * nullptr and runs on the user's default stream for this call. */
  if ((int)gStreamPool.size() >= maxEntries) {
    RESHARD_WARN(-1,
                 "Stream pool full (%d entries, "
                 "NCCL_RESHARD_STREAM_POOL_SIZE=%d); "
                 "falling through to the caller's default stream for this (comm, "
                 "dev) pair.  Bump NCCL_RESHARD_STREAM_POOL_SIZE if your "
                 "workload "
                 "uses more distinct (comm, dev) pairs.",
                 (int)gStreamPool.size(), maxEntries);
    *outStream = nullptr;
    *outReadyEvent = nullptr;
    *outDoneEvent = nullptr;
    return ncclSuccess;
  }
  /* Lazy-create stream + events for the new (comm, dev). */
  StreamPoolEntry fresh;
  fresh.comm = comm;
  fresh.dev = dev;
  if (cudaStreamCreateWithFlags(&fresh.stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&fresh.readyEvent, cudaEventDisableTiming) != cudaSuccess ||
      cudaEventCreateWithFlags(&fresh.doneEvent, cudaEventDisableTiming) != cudaSuccess) {
    if (fresh.doneEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.doneEvent));
    }
    if (fresh.readyEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.readyEvent));
    }
    if (fresh.stream != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(fresh.stream));
    }
    return ncclSystemError;
  }
  try {
    gStreamPool.push_back(fresh);
  } catch (...) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.doneEvent));
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.readyEvent));
    NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(fresh.stream));
    return ncclSystemError;
  }
  *outStream = fresh.stream;
  *outReadyEvent = fresh.readyEvent;
  *outDoneEvent = fresh.doneEvent;
  return ncclSuccess;
}
