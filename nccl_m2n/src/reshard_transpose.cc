/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "cuda_runtime.h"
#include <algorithm>
#include <mutex>
#include "nccl.h"
#include "reshard_types.h"
#include "m2n_checks.h"
#include "m2n_log.h"
#include "reshard_internal.h"

/* ======================================================================
 * Optional bucketed staging pool.
 *
 * Buffers are fixed-size best-fit buckets. A communicator keeps one stable
 * slot per bucket so window registration remains rank-uniform across calls.
 * ====================================================================*/

/* The slot memo is intentionally stable for the lifetime of a communicator.
 * Every rank must register its symmetric staging window in lockstep. If a
 * communicator changed slots between reshard sequences, ranks taking
 * different sequences could enter ncclCommWindowRegister for different
 * windows and desynchronize the collective. The slot index itself need not
 * be the same on every rank. */
struct StagingSlot {
  void* buffer;
  size_t size;
  cudaEvent_t doneEvent;
  cudaStream_t lastStream;
  int bucketIdx;
  int packWindowPreviousPeerCount;
  int packWindowPreviousPeers[MAX_DIRECT_TARGETS];
  bool eventRecorded;
  bool packWindowRmaWarmed;
  bool reserved;
  bool poisoned;
};

struct StagingBucketRT {
  size_t size;
  int numSlots;
  int numAssigned;
  StagingSlot slots[MAX_SPLIT_CONCURRENCY];
  bool allocated;
};

struct CommBucketSlot {
  ncclComm_t comm;
  int bucketIdx;
  int slotIdx;
};

struct StagingDevicePool {
  int cudaDev;
  StagingBucketRT buckets[kMaxStagingBuckets];
  int bucketCount;
  bool built;
  CommBucketSlot commSlots[MAX_SPLIT_CONCURRENCY * kMaxStagingBuckets];
  int commSlotCount;
  TransposeBufferEntry transposeEntries[MAX_TRANSPOSE_BUFFER_ENTRIES];
  int transposeEntryCount;
};

static StagingDevicePool gStagingDevicePools[MAX_TRANSPOSE_BUFFER_ENTRIES];
static int gStagingDevicePoolCount = 0;
static std::mutex gStagingPoolMutex;
static thread_local ncclComm_t gCurrentStagingComm = nullptr;
static thread_local StagingSlot* gCurrentStagingSlot = nullptr;
static thread_local TransposeBufferEntry* gCurrentTransposeEntry = nullptr;

/* ======================================================================
 * Per-comm symmetric transpose buffer pool.
 *
 * One buffer per ncclComm_t, reused across sequential collective calls.
 * When a different stream reuses the same comm's buffer, a per-entry
 * cudaEvent_t + cudaStreamWaitEvent serializes access.
 *
 * The fallback allocation is fixed at the first allocation, with a minimum
 * configured watermark, so cached window registrations never change size.
 * ====================================================================*/

static TransposeBufferEntry* findPoolEntry(StagingDevicePool& pool, ncclComm_t comm) {
  for (int i = 0; i < pool.transposeEntryCount; i++)
    if (pool.transposeEntries[i].comm == comm && pool.transposeEntries[i].allocated) return &pool.transposeEntries[i];
  return nullptr;
}

static StagingDevicePool* findStagingDevicePool(int cudaDev) {
  for (int i = 0; i < gStagingDevicePoolCount; i++) {
    if (gStagingDevicePools[i].cudaDev == cudaDev) return &gStagingDevicePools[i];
  }
  return nullptr;
}

static ncclResult_t acquireStagingDevicePool(StagingDevicePool** outPool) {
  int cudaDev = -1;
  NCCL_M2N_CUDACHECK(cudaGetDevice(&cudaDev));
  *outPool = findStagingDevicePool(cudaDev);
  if (*outPool != nullptr) return ncclSuccess;
  NCCL_M2N_CHECK_ARG(gStagingDevicePoolCount < MAX_TRANSPOSE_BUFFER_ENTRIES, -1,
                     "Staging device pool table full (%d entries); increase MAX_TRANSPOSE_BUFFER_ENTRIES",
                     MAX_TRANSPOSE_BUFFER_ENTRIES);
  *outPool = &gStagingDevicePools[gStagingDevicePoolCount++];
  (*outPool)->cudaDev = cudaDev;
  return ncclSuccess;
}

static int lookupCommBucketSlot(const StagingDevicePool& pool, ncclComm_t comm, int bucketIdx) {
  for (int i = 0; i < pool.commSlotCount; i++) {
    if (pool.commSlots[i].comm == comm && pool.commSlots[i].bucketIdx == bucketIdx) {
      return pool.commSlots[i].slotIdx;
    }
  }
  return -1;
}

static bool recordCommBucketSlot(StagingDevicePool* pool, ncclComm_t comm, int bucketIdx, int slotIdx) {
  if (pool->commSlotCount >= (int)(sizeof(pool->commSlots) / sizeof(pool->commSlots[0]))) return false;
  pool->commSlots[pool->commSlotCount++] = {comm, bucketIdx, slotIdx};
  return true;
}

static void buildStagingPoolMeta(StagingDevicePool* pool) {
  if (pool->built) return;
  pool->bucketCount = gReshardStagingBucketCount;
  for (int i = 0; i < pool->bucketCount; i++) {
    StagingBucketRT& bucket = pool->buckets[i];
    bucket.size = gReshardStagingBuckets[i].size;
    bucket.numSlots = gReshardStagingBuckets[i].numSlots;
    bucket.numAssigned = 0;
    bucket.allocated = false;
    for (int slot = 0; slot < bucket.numSlots; slot++) {
      bucket.slots[slot] = {};
      bucket.slots[slot].size = bucket.size;
      bucket.slots[slot].bucketIdx = i;
    }
  }
  pool->built = true;
}

static ncclResult_t ensureBucketAllocated(StagingBucketRT* bucket) {
  if (bucket->allocated) return ncclSuccess;
  for (int slot = 0; slot < bucket->numSlots; slot++) {
    NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&bucket->slots[slot].doneEvent, cudaEventDisableTiming));
    ncclResult_t result = ncclMemAlloc(&bucket->slots[slot].buffer, bucket->size);
    if (result != ncclSuccess) {
      for (int cleanup = 0; cleanup <= slot; cleanup++) {
        if (bucket->slots[cleanup].buffer != nullptr) ncclMemFree(bucket->slots[cleanup].buffer);
        if (bucket->slots[cleanup].doneEvent != nullptr) cudaEventDestroy(bucket->slots[cleanup].doneEvent);
        bucket->slots[cleanup] = {};
      }
      return result;
    }
  }
  bucket->allocated = true;
  RESHARD_DEBUG(-1, "staging bucket allocated: %d slots x %zu B", bucket->numSlots, bucket->size);
  return ncclSuccess;
}

/* A reservation is abandoned when it outlived the call that took it.
 *
 * A slot is reserved by ensureTransposeBuffer and released by
 * transposeBufferRecordEvent. Any error between the two returns early and would
 * otherwise leave the slot reserved for good -- and a reserved slot is never
 * reacquired, so the communicator fails every later call with a spurious
 * concurrency error and stops entering collectives its peers still enter.
 *
 * Ownership is deliberately NOT thread-local: the public API lock serializes
 * reshard calls but does not pin them to a thread, so a retry after a failure
 * can arrive on a different thread than the one that abandoned the slot. What
 * does hold is that a reservation with no M2N call in flight besides this one
 * cannot belong to a live call -- so treat it as abandoned.
 *
 * Poison it rather than reusing it: the completion event was never recorded, so
 * neither the buffer contents nor any in-flight device work are known. */
static bool stagingReservationIsAbandoned() {
  return !m2nApiHasConcurrentCalls();
}

static void reclaimAbandonedSlot(StagingSlot* slot) {
  slot->reserved = false;
  slot->eventRecorded = false;
  slot->poisoned = true;
}

static void reclaimAbandonedEntry(TransposeBufferEntry* entry) {
  entry->reserved = false;
  entry->eventRecorded = false;
  entry->poisoned = true;
}


static ncclResult_t acquireStagingSlot(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream,
                                       StagingSlot** outSlot) {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingDevicePool* pool = nullptr;
  NCCL_M2N_CHECK(acquireStagingDevicePool(&pool));
  buildStagingPoolMeta(pool);
  StagingBucketRT* bucket = nullptr;
  int bucketIdx = -1;
  for (int i = 0; i < pool->bucketCount; i++) {
    if (pool->buckets[i].size >= requiredBytes) {
      bucket = &pool->buckets[i];
      bucketIdx = i;
      break;
    }
  }
  if (bucket == nullptr) {
    const size_t largest = pool->bucketCount > 0 ? pool->buckets[pool->bucketCount - 1].size : 0;
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "staging request %zu B exceeds the largest configured bucket %zu B; add or enlarge a bucket in "
                  "NCCL_RESHARD_STAGING_BUCKETS or raise NCCL_RESHARD_STAGING_WATERMARK_BYTES",
                  requiredBytes, largest);
  }
  NCCL_M2N_CHECK(ensureBucketAllocated(bucket));

  int slotIdx = lookupCommBucketSlot(*pool, comm, bucketIdx);
  if (slotIdx < 0) {
    NCCL_M2N_CHECK_ARG(bucket->numAssigned < bucket->numSlots, -1,
                       "staging bucket %d has no free communicator slots (%d configured)", bucketIdx,
                       bucket->numSlots);
    slotIdx = bucket->numAssigned++;
    NCCL_M2N_CHECK_ARG(recordCommBucketSlot(pool, comm, bucketIdx, slotIdx), -1,
                       "staging bucket communicator memo is full");
  }
  StagingSlot* slot = &bucket->slots[slotIdx];
  /* Reclaim before the poison check, not after: reclaiming marks the slot
   * poisoned, and a check that already ran would let this very call reuse a
   * slot it just declared unusable. */
  if (slot->reserved && stagingReservationIsAbandoned()) {
    reclaimAbandonedSlot(slot);
  }
  if (slot->poisoned) {
    NCCL_M2N_FAIL(ncclUnhandledCudaError, -1, "Staging bucket %d slot %d is unavailable", bucketIdx, slotIdx);
  }
  NCCL_M2N_CHECK_ARG(!slot->reserved, -1,
                     "Concurrent reshard operations on communicator %p are unsupported; serialize calls at the caller",
                     (void*)comm);
  if (slot->eventRecorded && slot->lastStream != stream) {
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, slot->doneEvent, 0));
  }
  slot->reserved = true;
  gCurrentStagingComm = comm;
  gCurrentStagingSlot = slot;
  *outSlot = slot;
  return ncclSuccess;
}

ncclResult_t ensureTransposeBuffer(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream) {
  if (reshardStagingBucketsEnabled()) {
    StagingSlot* slot = nullptr;
    return acquireStagingSlot(comm, requiredBytes, stream, &slot);
  }

  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingDevicePool* pool = nullptr;
  NCCL_M2N_CHECK(acquireStagingDevicePool(&pool));
  TransposeBufferEntry* entry = findPoolEntry(*pool, comm);

  if (entry != nullptr) {
    /* Reclaim first, so the poison it sets is honoured by the check below. */
    if (entry->reserved && stagingReservationIsAbandoned()) {
      reclaimAbandonedEntry(entry);
    }
    if (entry->reserved) {
      NCCL_M2N_FAIL(ncclInvalidUsage, -1,
                    "Concurrent reshard operations on communicator %p are unsupported; serialize calls at the caller",
                    (void*)comm);
    }
    if (entry->poisoned) {
      NCCL_M2N_FAIL(ncclUnhandledCudaError, -1, "Transpose buffer for comm %p is unavailable", (void*)comm);
    }
    if (entry->eventRecorded && entry->stream != stream) {
      NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, entry->event, 0));
      entry->stream = stream;
    }

    if (entry->capacity < requiredBytes) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                    "staging request %zu B exceeds comm %p's fixed capacity %zu B; process the comm's largest "
                    "tensor first or raise NCCL_RESHARD_STAGING_WATERMARK_BYTES",
                    requiredBytes, (void*)comm, entry->capacity);
    }
    entry->reserved = true;
    gCurrentTransposeEntry = entry;
    return ncclSuccess;
  }

  /* New comm — allocate the event and buffer into locals, then commit the
   * pool slot only after both succeed.  Committing first (incrementing
   * transposeEntryCount / setting allocated=true) would leave a half-initialized
   * entry on any allocation failure, which getTransposeBuffer would later
   * surface as a null buffer. */
  if (pool->transposeEntryCount >= MAX_TRANSPOSE_BUFFER_ENTRIES) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "Transpose buffer pool full (%d entries); increase MAX_TRANSPOSE_BUFFER_ENTRIES",
                  MAX_TRANSPOSE_BUFFER_ENTRIES);
  }

  cudaEvent_t event = nullptr;
  NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

  void* buffer = nullptr;
  const size_t allocBytes = std::max(requiredBytes, gReshardStagingWatermarkBytes);
  ncclResult_t r = ncclMemAlloc(&buffer, allocBytes);
  if (r != ncclSuccess) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    NCCL_M2N_FAIL(r, -1, "Failed to allocate %zu-byte transpose buffer: %s", allocBytes, ncclGetErrorString(r));
  }

  TransposeBufferEntry& e = pool->transposeEntries[pool->transposeEntryCount++];
  e.comm = comm;
  e.stream = stream;
  e.event = event;
  e.buffer = buffer;
  e.capacity = allocBytes;
  e.eventRecorded = false;
  e.reserved = true;
  e.poisoned = false;
  e.allocated = true;
  /* Track it like the reuse path above, so an error before the completion
   * event is recorded releases this reservation on the next call instead of
   * wedging the communicator on its very first reshard. */
  gCurrentTransposeEntry = &e;
  RESHARD_DEBUG(-1, "Transpose buffer allocated for comm %p: %zu bytes (req=%zu floor=%zu, %p) [slot %d]", (void*)comm,
                allocBytes, requiredBytes, gReshardStagingWatermarkBytes,
                e.buffer, pool->transposeEntryCount - 1);
  return ncclSuccess;
}

void* getTransposeBuffer(ncclComm_t comm) {
  if (reshardStagingBucketsEnabled()) {
    return (comm == gCurrentStagingComm && gCurrentStagingSlot != nullptr) ? gCurrentStagingSlot->buffer : nullptr;
  }
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingDevicePool* pool = nullptr;
  if (acquireStagingDevicePool(&pool) != ncclSuccess) return nullptr;
  TransposeBufferEntry* e = findPoolEntry(*pool, comm);
  return (e != nullptr) ? e->buffer : nullptr;
}

size_t getTransposeBufferCapacity(ncclComm_t comm) {
  if (reshardStagingBucketsEnabled()) {
    return (comm == gCurrentStagingComm && gCurrentStagingSlot != nullptr) ? gCurrentStagingSlot->size : 0;
  }
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingDevicePool* pool = nullptr;
  if (acquireStagingDevicePool(&pool) != ncclSuccess) return 0;
  TransposeBufferEntry* e = findPoolEntry(*pool, comm);
  return (e != nullptr) ? e->capacity : 0;
}

ncclResult_t getTransposeBufferPackWindowState(ncclComm_t comm, bool* rmaWarmed, int* previousPeerCount,
                                               int previousPeers[MAX_DIRECT_TARGETS]) {
  NCCL_M2N_CHECK_ARG(rmaWarmed != nullptr && previousPeerCount != nullptr && previousPeers != nullptr, -1,
                     "PACKWINDOW staging state output must be non-null");
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  if (reshardStagingBucketsEnabled()) {
    StagingSlot* slot = (comm == gCurrentStagingComm) ? gCurrentStagingSlot : nullptr;
    NCCL_M2N_CHECK_ARG(slot != nullptr, -1, "PACKWINDOW staging slot is unavailable for communicator %p",
                       (void*)comm);
    *rmaWarmed = slot->packWindowRmaWarmed;
    *previousPeerCount = slot->packWindowPreviousPeerCount;
    for (int i = 0; i < *previousPeerCount; i++) {
      previousPeers[i] = slot->packWindowPreviousPeers[i];
    }
    return ncclSuccess;
  }

  StagingDevicePool* pool = nullptr;
  NCCL_M2N_CHECK(acquireStagingDevicePool(&pool));
  TransposeBufferEntry* entry = findPoolEntry(*pool, comm);
  NCCL_M2N_CHECK_ARG(entry != nullptr, -1, "PACKWINDOW transpose buffer is unavailable for communicator %p",
                     (void*)comm);
  *rmaWarmed = entry->packWindowRmaWarmed;
  *previousPeerCount = entry->packWindowPreviousPeerCount;
  for (int i = 0; i < *previousPeerCount; i++) {
    previousPeers[i] = entry->packWindowPreviousPeers[i];
  }
  return ncclSuccess;
}

ncclResult_t setTransposeBufferPackWindowState(ncclComm_t comm, bool rmaWarmed, int previousPeerCount,
                                               const int previousPeers[MAX_DIRECT_TARGETS]) {
  NCCL_M2N_CHECK_ARG(previousPeerCount >= 0 && previousPeerCount <= MAX_DIRECT_TARGETS, -1,
                     "PACKWINDOW previous-peer count %d exceeds capacity %d", previousPeerCount,
                     MAX_DIRECT_TARGETS);
  NCCL_M2N_CHECK_ARG(previousPeerCount == 0 || previousPeers != nullptr, -1,
                     "PACKWINDOW previous-peer input must be non-null");
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  if (reshardStagingBucketsEnabled()) {
    StagingSlot* slot = (comm == gCurrentStagingComm) ? gCurrentStagingSlot : nullptr;
    NCCL_M2N_CHECK_ARG(slot != nullptr, -1, "PACKWINDOW staging slot is unavailable for communicator %p",
                       (void*)comm);
    slot->packWindowRmaWarmed = rmaWarmed;
    slot->packWindowPreviousPeerCount = previousPeerCount;
    for (int i = 0; i < previousPeerCount; i++) {
      slot->packWindowPreviousPeers[i] = previousPeers[i];
    }
    return ncclSuccess;
  }

  StagingDevicePool* pool = nullptr;
  NCCL_M2N_CHECK(acquireStagingDevicePool(&pool));
  TransposeBufferEntry* entry = findPoolEntry(*pool, comm);
  NCCL_M2N_CHECK_ARG(entry != nullptr, -1, "PACKWINDOW transpose buffer is unavailable for communicator %p",
                     (void*)comm);
  entry->packWindowRmaWarmed = rmaWarmed;
  entry->packWindowPreviousPeerCount = previousPeerCount;
  for (int i = 0; i < previousPeerCount; i++) {
    entry->packWindowPreviousPeers[i] = previousPeers[i];
  }
  return ncclSuccess;
}

int getStagingBucketIndex(ncclComm_t comm) {
  if (!reshardStagingBucketsEnabled()) return 0;
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingSlot* slot = (comm == gCurrentStagingComm) ? gCurrentStagingSlot : nullptr;
  return (slot != nullptr) ? slot->bucketIdx : -1;
}

ncclResult_t transposeBufferRecordEvent(ncclComm_t comm, cudaStream_t stream) {
  if (reshardStagingBucketsEnabled()) {
    std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
    if (comm == gCurrentStagingComm && gCurrentStagingSlot != nullptr) {
      StagingSlot* slot = gCurrentStagingSlot;
      /* Release the reservation whether or not the record succeeds. A slot left
       * reserved is never reacquired, so the communicator reports spurious
       * concurrent use on every later call and can strand its peers in a
       * collective the failing rank never enters. */
      const ncclResult_t result =
        reshardRecordCompletionEvent(slot->doneEvent, stream, "staging slot", &slot->poisoned);
      slot->lastStream = stream;
      slot->eventRecorded = (result == ncclSuccess);
      slot->reserved = false;
      gCurrentStagingComm = nullptr;
      gCurrentStagingSlot = nullptr;
      NCCL_M2N_CHECK(result);
    }
    return ncclSuccess;
  }
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingDevicePool* pool = nullptr;
  NCCL_M2N_CHECK(acquireStagingDevicePool(&pool));
  TransposeBufferEntry* e = findPoolEntry(*pool, comm);
  if (e != nullptr) {
    const ncclResult_t result =
      reshardRecordCompletionEvent(e->event, stream, "transpose buffer", &e->poisoned);
    e->stream = stream;
    e->eventRecorded = (result == ncclSuccess);
    e->reserved = false;
    gCurrentTransposeEntry = nullptr;
    NCCL_M2N_CHECK(result);
  }
  return ncclSuccess;
}

void transposeBufferSynchronize() {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  int currentDevice = -1;
  NCCL_M2N_CUDACHECK_WARN(cudaGetDevice(&currentDevice));
  for (int device = 0; device < gStagingDevicePoolCount; device++) {
    StagingDevicePool& pool = gStagingDevicePools[device];
    NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(pool.cudaDev));
    for (int entryIdx = 0; entryIdx < pool.transposeEntryCount; entryIdx++) {
      TransposeBufferEntry& entry = pool.transposeEntries[entryIdx];
      if (entry.eventRecorded) NCCL_M2N_CUDACHECK_WARN(cudaEventSynchronize(entry.event));
    }
    for (int bucketIdx = 0; bucketIdx < pool.bucketCount; bucketIdx++) {
      StagingBucketRT& bucket = pool.buckets[bucketIdx];
      if (!bucket.allocated) continue;
      for (int slotIdx = 0; slotIdx < bucket.numSlots; slotIdx++) {
        StagingSlot& slot = bucket.slots[slotIdx];
        if (slot.eventRecorded) NCCL_M2N_CUDACHECK_WARN(cudaEventSynchronize(slot.doneEvent));
      }
    }
  }
  if (currentDevice >= 0) NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(currentDevice));
}

void transposeBufferFinalize() {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  int currentDevice = -1;
  NCCL_M2N_CUDACHECK_WARN(cudaGetDevice(&currentDevice));
  for (int device = 0; device < gStagingDevicePoolCount; device++) {
    StagingDevicePool& pool = gStagingDevicePools[device];
    bool hasAllocation = false;
    for (int entryIdx = 0; entryIdx < pool.transposeEntryCount && !hasAllocation; entryIdx++) {
      hasAllocation = pool.transposeEntries[entryIdx].allocated;
    }
    for (int bucketIdx = 0; bucketIdx < pool.bucketCount && !hasAllocation; bucketIdx++) {
      hasAllocation = pool.buckets[bucketIdx].allocated;
    }
    if (!pool.built && !hasAllocation) continue;
    if (hasAllocation) NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(pool.cudaDev));
    for (int entryIdx = 0; entryIdx < pool.transposeEntryCount; entryIdx++) {
      if (pool.transposeEntries[entryIdx].event != nullptr) cudaEventDestroy(pool.transposeEntries[entryIdx].event);
      if (pool.transposeEntries[entryIdx].buffer != nullptr) ncclMemFree(pool.transposeEntries[entryIdx].buffer);
      pool.transposeEntries[entryIdx] = {};
    }
    pool.transposeEntryCount = 0;
    for (int bucketIdx = 0; bucketIdx < pool.bucketCount; bucketIdx++) {
      StagingBucketRT& bucket = pool.buckets[bucketIdx];
      for (int slotIdx = 0; slotIdx < bucket.numSlots; slotIdx++) {
        if (bucket.slots[slotIdx].doneEvent != nullptr) cudaEventDestroy(bucket.slots[slotIdx].doneEvent);
        if (bucket.slots[slotIdx].buffer != nullptr) {
          ncclMemFree(bucket.slots[slotIdx].buffer);
        }
      }
      bucket = {};
    }
    pool = {};
  }
  if (currentDevice >= 0) NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(currentDevice));
  gStagingDevicePoolCount = 0;
  gCurrentStagingComm = nullptr;
  gCurrentStagingSlot = nullptr;
  gCurrentTransposeEntry = nullptr;
}

bool shouldTransposeForCrossDim(const size_t* srcDimsBytes, const size_t* dstDimsBytes, int ndims, int srcShardDim,
                                int dstShardDim, int srcShardCount, int dstShardCount, int* swapDimA, int* swapDimB) {
  const size_t transposeThreshold = reshardGetCrossDimTransposeThresholdBytes();

  // 2D case: replicated src (or shard on dim 0) -> dst shards innermost dim
  if (ndims == 2 && dstShardDim == 1 && srcShardDim != 1) {
    const size_t* dims = (dstDimsBytes[0] > 0) ? dstDimsBytes : srcDimsBytes;
    size_t innerSize = dims[1];
    if (innerSize < transposeThreshold) {
      *swapDimA = 0;
      *swapDimB = 1;
      return true;
    }
    return false;
  }

  if (ndims != 3) return false;
  if (srcShardDim < 0 || dstShardDim < 0) return false;
  if (srcShardDim == dstShardDim) return false;

  const size_t* dims = (srcDimsBytes[0] > 0) ? srcDimsBytes : dstDimsBytes;

  size_t globalInner = dims[ndims - 1];
  if (srcShardDim == ndims - 1) globalInner *= srcShardCount;
  if (dstShardDim == ndims - 1) globalInner *= dstShardCount;

  size_t innerSize = globalInner;
  if (dstShardDim == ndims - 1) innerSize = globalInner / dstShardCount;
  if (srcShardDim == ndims - 1) innerSize = globalInner / srcShardCount;

  if (innerSize >= transposeThreshold) return false;

  int freeDim = -1;
  for (int d = 0; d < ndims; d++) {
    if (d != srcShardDim && d != dstShardDim) {
      freeDim = d;
      break;
    }
  }
  if (freeDim < 0) return false;

  if (dstShardDim == ndims - 1 && srcShardDim != ndims - 2) {
    *swapDimA = ndims - 2;
    *swapDimB = ndims - 1;
    return true;
  }

  return false;
}
