/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Internal Function Declarations
 *
 * Cross-TU function declarations shared by the split library modules.
 * Each module (reshard_config, reshard_cache, reshard_mesh, etc.)
 * exports its public-to-the-library functions here.
 *
 * Not part of the public C API — intentionally kept inside src/.
 ************************************************************************/

#ifndef NCCL_RESHARD_INTERNAL_H_
#define NCCL_RESHARD_INTERNAL_H_

#include <cstdint>
#include <limits>
#include <memory>

#include "m2n_checks.h"
#include "m2n_handle.h"
#include "m2n_log.h"
#include <mutex>

#include "reshard_types.h"

struct ncclDevComm;

/* ======================================================================
 * Public host-API serialization
 *
 * Mutable process-global M2N host state (the handle table, the runtime epoch,
 * and the DevComm/window caches) is serialized by one process-wide mutex.
 * Device work stays asynchronous: the lock covers host bookkeeping only, and
 * is released before a public call returns.
 *
 * Blocking NCCL collectives must NOT run while the lock is held. Calls such as
 * ncclCommWindowRegister and ncclDevCommCreate only complete once every rank in
 * the communicator has entered them, so a process hosting two ranks of the same
 * communicator would deadlock: the first rank would hold the lock inside the
 * collective while the second waited for the lock to enter it. Wrap each such
 * call in an M2nApiUnlock scope, which drops the lock for the duration and
 * retakes it on scope exit.
 * ====================================================================== */

std::mutex& getM2nApiMutex();

/* True when more than one public M2N call is in flight in this process. Used to
 * decide whether a cache eviction may run its collective teardown now or must
 * be deferred: eviction is data-dependent per rank, so releasing the lock to
 * deregister a window that only one rank is evicting would turn a deadlock into
 * an unmatched collective. */
bool m2nApiHasConcurrentCalls();

/* Takes the API lock for the duration of one public entry point and publishes
 * it so a nested M2nApiUnlock can find it. Not reentrant: one per call. */
class M2nApiLock {
 public:
  M2nApiLock();
  ~M2nApiLock();

  M2nApiLock(const M2nApiLock&) = delete;
  M2nApiLock& operator=(const M2nApiLock&) = delete;

 private:
  std::unique_lock<std::mutex> lock_;
};

/* Temporarily releases the API lock held by the enclosing M2nApiLock so peer
 * ranks in this process can enter their own public calls and join the
 * collective. A no-op when no lock is held on this thread. */
class M2nApiUnlock {
 public:
  M2nApiUnlock();
  ~M2nApiUnlock();

  M2nApiUnlock(const M2nApiUnlock&) = delete;
  M2nApiUnlock& operator=(const M2nApiUnlock&) = delete;

 private:
  bool unlocked_ = false;
};

/* ======================================================================
 * Global configuration (inline — getters fold into a single load).
 *
 * Initial values are library defaults.  Runtime initialization applies the
 * first config for a process-lifetime epoch and then env vars in
 * m2n_config.cc.  Env vars always win.
 * ====================================================================*/

inline int gReshardGpusPerNode = DEFAULT_GPUS_PER_NODE;
inline int gReshardSrcDomainSize = 0;
inline int gReshardDstDomainSize = 0;
inline ReshardAlgorithm gReshardAlgorithm = RESHARD_ALGO_AUTO;
inline ReshardLoadBalanceMode gReshardLbMode = RESHARD_LB_UNIFORM;

/* Upper bound on pickNumCtas() output.  0 = unset (use DEFAULT_NUM_CTAS). */
inline int gReshardMaxCta = 0;

/* Direct CTA-count override from NCCL_RESHARD_NUM_CTAS. 0 = unset; when set,
 * this wins over config.maxCta. */
inline int gReshardNumCtasOverride = 0;

/* Resolved CTA count, computed once during runtime initialization from
 * gReshardNumCtasOverride / gReshardMaxCta / DEFAULT_NUM_CTAS. pickNumCtas reads this directly - no
 * per-call branch. */
inline int gReshardNumCtas = DEFAULT_NUM_CTAS;

/* Stream execution mode populated at first ncclM2nInit from
 * NCCL_RESHARD_USE_INTERNAL_STREAMS. Internal streams are the default; false
 * keeps work on caller streams with ordered DevComm reuse. */
inline bool gReshardUseInternalStreams = true;

/* Byte-level chunk size used by the RING prepare path. Default is
 * CHUNK_SIZE_BYTES; overridable via NCCL_RESHARD_CHUNK_SIZE.
 * Parsed once at first init in applyReshardEnv — keeps
 * prepareReshardParams off the getenv path on every call. 0 means
 * "use the compile-time default". */
inline size_t gReshardChunkSizeBytes = 0;

inline ReshardAlgorithm reshardGetAlgorithm() {
  return gReshardAlgorithm;
}
inline int reshardGetGpusPerNode() {
  return gReshardGpusPerNode;
}
inline int reshardGetSrcDomainSize() {
  return gReshardSrcDomainSize;
}
inline int reshardGetDstDomainSize() {
  return gReshardDstDomainSize;
}
/* Resolve per-side domain sizes after topology discovery. RING treats
 * destination-domain peers as LSA-local, so its destination override must not
 * exceed dstLsaSize. An LSA size of 0 retains the legacy gpus-per-node
 * fallback. */
ncclResult_t resolveReshardDomainSizes(int worldRank, ReshardAlgorithm algo, int srcLsaSize, int dstLsaSize,
                                       int* srcGpusPerDomain, int* dstGpusPerDomain);
inline ReshardLoadBalanceMode reshardGetLoadBalanceMode() {
  return gReshardLbMode;
}
inline bool reshardUseInternalStreams() {
  return gReshardUseInternalStreams;
}
/* ======================================================================
 * m2n_config.cc — configuration appliers
 *
 * Applied in order from ncclM2nInit; env always overrides config.
 * ====================================================================*/
void resetReshardRuntimeConfig();
void applyReshardConfig(const ncclM2nConfig_t* config);
ncclResult_t validateReshardConfigHeader(const ncclM2nConfig_t* config);
void applyReshardEnv();

/* Validate an explicit handle token, or lazily create the internal default for
 * a NULL token. The returned state keeps its runtime alive for the call. */
ncclResult_t acquireM2nHandle(ncclM2nHandle_t token, std::shared_ptr<ncclM2nHandleState>* handle);

/* Element-size lookup for the dtypes accepted by ncclReshardWithWindow.
 * Returns 0 for unsupported dtypes (the API rejects them at call time). */
inline size_t getNcclDtSize(ncclDataType_t t) {
  switch (t) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return 0;
  }
}

/* ======================================================================
 * Picker stubs for numCtas / elementsPerChunk
 *
 * Currently constant — return the value resolved once at
 * ncclM2nInit.  Signature intentionally future-aware
 * (`bytesPerRank`, `algo`) so an input-aware heuristic can drop in
 * without a caller change.
 * ====================================================================*/

inline int pickNumCtas(size_t bytesPerRank, ReshardAlgorithm algo) {
  (void)bytesPerRank;
  (void)algo;
  return gReshardNumCtas;
}

inline size_t pickElementsPerChunk(size_t bytesPerRank, ReshardAlgorithm algo) {
  (void)bytesPerRank;
  (void)algo;
  return DEFAULT_ELEMENTS_PER_CHUNK;
}

/* ======================================================================
 * reshard_cache.cc — DevComm and Window caches
 * ====================================================================*/

ncclDevComm* findCachedDevComm(ncclComm_t comm, int numCtas, int signalCount, cudaStream_t stream = nullptr);

ncclResult_t cacheDevComm(ncclComm_t comm, int numCtas, int signalCount, const ncclDevComm* devComm,
                          cudaStream_t stream = nullptr);

ncclWindow_t* findCachedInternalWindowByPtr(ncclComm_t comm, void* buffer, size_t size);

ncclResult_t cacheInternalWindow(ncclComm_t comm, void* buffer, size_t size, ncclWindow_t window);

/* Acquire a library-owned (stream, ready event, done event) tuple for callers that pass
 * the default stream (nullptr / cudaStreamLegacy / cudaStreamPerThread).
 * 1:1 mapping per (comm, dev) — lazy-creates the tuple on first use;
 * subsequent calls for the same pair return the same handles.  Both
 * events are owned by the cache and freed by cacheFinalize().  They
 * are reused across calls so we don't pay cudaEvent{Create,
 * Destroy} per reshard.
 *
 * Resource creation fails loudly; there is no unordered fallback to the
 * caller stream. */
ncclResult_t streamPoolAcquire(ncclComm_t comm, int dev, cudaStream_t* outStream, cudaEvent_t* outReadyEvent,
                               cudaEvent_t* outDoneEvent);

void cacheFinalize();

/* ======================================================================
 * reshard_mesh.cc — Mesh analysis helpers
 * ====================================================================*/

struct ReshardMeshInterval {
  int startRank;
  int endRank;
  int size;
};

ncclResult_t computeReshardMeshInterval(const ncclMesh_t* mesh, int logRank, ReshardMeshInterval* interval);

ncclResult_t computeStridesChecked(const size_t dims[], int ndims, size_t strides[]);
void computeStrides(const size_t dims[], int ndims, size_t strides[]);

/* Validate that both meshes have positive dimensions before any mesh-group
 * math divides by them.  Host-only; returns ncclInvalidArgument on a null
 * mesh or a non-positive dim. */
ncclResult_t validateReshardMeshDims(const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh);
ncclResult_t validateReshardMeshBounds(const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh, int commSize,
                                      int logRank);

inline bool reshardRankInMesh(const ncclMesh_t* mesh, int worldRank) {
  if (mesh == nullptr || mesh->dims[0] <= 0 || mesh->dims[1] <= 0 || mesh->startRank < 0 || worldRank < 0) {
    return false;
  }
  const int64_t meshSize = static_cast<int64_t>(mesh->dims[0]) * static_cast<int64_t>(mesh->dims[1]);
  const int64_t meshEnd = static_cast<int64_t>(mesh->startRank) + meshSize;
  return static_cast<int64_t>(worldRank) >= static_cast<int64_t>(mesh->startRank) &&
         static_cast<int64_t>(worldRank) < meshEnd;
}

ncclResult_t validateReshardPlacement(const ncclDistTensor_t* tensor, const char* apiName, const char* fieldName);

inline ncclResult_t computeReshardMeshSize(const ncclMesh_t* mesh, int logRank, size_t* outMeshSize) {
  if (mesh == nullptr || outMeshSize == nullptr) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: computeReshardMeshSize called with null argument");
  }
  if (mesh->dims[0] <= 0 || mesh->dims[1] <= 0) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: mesh dims must be positive, got [%d, %d]", mesh->dims[0],
                  mesh->dims[1]);
  }

  size_t dim0 = static_cast<size_t>(mesh->dims[0]);
  size_t dim1 = static_cast<size_t>(mesh->dims[1]);
  if (dim0 > std::numeric_limits<size_t>::max() / dim1) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: mesh size overflows size_t for dims [%d, %d]",
                  mesh->dims[0], mesh->dims[1]);
  }

  *outMeshSize = dim0 * dim1;
  return ncclSuccess;
}

/* GIN's ncclDevCommRequirements::ginSignalCount is an int, and kernel signal
 * IDs are 32-bit.  Keeping source signal IDs relative to srcMesh->startRank
 * lets a non-zero source mesh start at signal slot 0 instead of silently
 * indexing past srcMeshSize * numCtas. */
inline ncclResult_t computeReshardGinSignalCount(const ncclMesh_t* srcMesh, int numCtas, int logRank,
                                                 int* outSignalCount) {
  if (outSignalCount == nullptr) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: computeReshardGinSignalCount called with null output");
  }
  if (numCtas <= 0) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: numCtas must be positive, got %d", numCtas);
  }

  size_t srcTotal = 0;
  ncclResult_t result = computeReshardMeshSize(srcMesh, logRank, &srcTotal);
  if (result != ncclSuccess) return result;

  size_t ctas = static_cast<size_t>(numCtas);
  if (srcTotal > static_cast<size_t>(std::numeric_limits<int>::max()) / ctas) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank,
                  "reshard: GIN signal count overflows NCCL int field "
                  "(srcRanks=%zu, numCtas=%d)",
                  srcTotal, numCtas);
  }

  *outSignalCount = static_cast<int>(srcTotal * ctas);
  return ncclSuccess;
}

inline ncclResult_t computeReshardSignalBase(const ncclMesh_t* srcMesh, int srcRank, int numCtas, int logRank,
                                             unsigned int* outSignalBase) {
  if (outSignalBase == nullptr) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: computeReshardSignalBase called with null output");
  }
  if (numCtas <= 0) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank, "reshard: numCtas must be positive, got %d", numCtas);
  }

  size_t srcTotal = 0;
  ncclResult_t result = computeReshardMeshSize(srcMesh, logRank, &srcTotal);
  if (result != ncclSuccess) return result;

  int64_t relativeRank = static_cast<int64_t>(srcRank) - static_cast<int64_t>(srcMesh->startRank);
  if (relativeRank < 0 || static_cast<size_t>(relativeRank) >= srcTotal) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank,
                  "reshard: source rank %d is outside source mesh "
                  "(startRank=%d, size=%zu)",
                  srcRank, srcMesh->startRank, srcTotal);
  }

  size_t relativeRankSize = static_cast<size_t>(relativeRank);
  size_t ctas = static_cast<size_t>(numCtas);
  size_t maxSignal = static_cast<size_t>(std::numeric_limits<unsigned int>::max());
  if (relativeRankSize > maxSignal / ctas) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank,
                  "reshard: source signal base overflows 32-bit GIN signal id "
                  "(relativeRank=%zu, numCtas=%d)",
                  relativeRankSize, numCtas);
  }

  size_t signalBase = relativeRankSize * ctas;
  if (signalBase > maxSignal - (ctas - 1)) {
    NCCL_M2N_FAIL(ncclInvalidArgument, logRank,
                  "reshard: source signal range overflows 32-bit GIN signal id "
                  "(base=%zu, numCtas=%d)",
                  signalBase, numCtas);
  }

  *outSignalBase = static_cast<unsigned int>(signalBase);
  return ncclSuccess;
}

inline unsigned int computeReshardSignalBaseUnchecked(const ncclMesh_t* srcMesh, int srcRank, int numCtas) {
  size_t relativeRank = static_cast<size_t>(static_cast<int64_t>(srcRank) - static_cast<int64_t>(srcMesh->startRank));
  return static_cast<unsigned int>(relativeRank * static_cast<size_t>(numCtas));
}

void computeMeshGroupInfo(const ncclDistTensor_t* tensor, int worldRank, ncclReshardMeshGroupInfo* info);

int getMeshRank(const ncclDistTensor_t* tensor, const ncclReshardMeshGroupInfo* info, int shardIdx, int repIdx);

void computeGlobalRange(const size_t localDims[], int ndims, int shardTensorDim, int shardIdx, size_t globalStart[],
                        size_t globalEnd[]);

bool computeOverlap(const size_t srcStart[], const size_t srcEnd[], const size_t dstStart[], const size_t dstEnd[],
                    int ndims, size_t overlapStart[], size_t overlapEnd[]);

void computeTransferPlan(const size_t srcDims[], const size_t srcStrides[], int srcShardDim, int srcShardIdx,
                         const size_t dstDims[], const size_t dstStrides[], int dstShardDim, int dstShardIdx, int ndims,
                         size_t elementsPerChunk, ncclReshardTransferPlan* plan);
ncclResult_t computeTransferPlanChecked(const size_t srcDims[], const size_t srcStrides[], int srcShardDim,
                                        int srcShardIdx, const size_t dstDims[], const size_t dstStrides[],
                                        int dstShardDim, int dstShardIdx, int ndims, size_t elementsPerChunk,
                                        ncclReshardTransferPlan* plan);

/* ======================================================================
 * reshard_loadbalance.cc — Replication load balancer
 * ====================================================================*/

int getNodeOfDestRep(const ncclReshardRepLoadBalancer* lb, int dstRepIdx);
int getNumDestNodes(const ncclReshardRepLoadBalancer* lb);

void getDestRepsOnNode(const ncclReshardRepLoadBalancer* lb, int targetNode, int* repStart, int* repEnd);

void getDestRepsOnNodeRange(const ncclReshardRepLoadBalancer* lb, int firstNode, int lastNode, int* repStart, int* repEnd);

void getTargetRepRange(const ncclReshardRepLoadBalancer* lb, int srcRepIdx, int* repStart, int* repEnd);

int getSourceRepForDest(const ncclReshardRepLoadBalancer* lb, int dstRepIdx);

/* ======================================================================
 * reshard_prepare.cc — Kernel parameter builders
 * ====================================================================*/

ncclResult_t prepareReshardParams(int worldRank, const ncclDistTensor_t* src, const size_t srcTensorDims[],
                                  const ncclDistTensor_t* dst, const size_t dstTensorDims[], ncclWindow_t window,
                                  size_t elementsPerChunk, int numCtas, unsigned int mySignalBase,
                                  int srcGpusPerDomain, int dstGpusPerDomain, const size_t* allWindowOffsets,
                                  ncclReshardParams* outParams);

ncclResult_t prepareDirectReshardParams(int worldRank, const ncclDistTensor_t* src, const size_t srcTensorDims[],
                                        const ncclDistTensor_t* dst, const size_t dstTensorDims[],
                                        ncclWindow_t window, size_t elementsPerChunk, int numCtas,
                                        unsigned int mySignalBase, int dstGpusPerDomain,
                                        const size_t* allWindowOffsets, ncclReshardDirectParams* outParams);

ncclResult_t validateReshardPlanLimits(int worldRank, const ncclDistTensor_t* src, const size_t srcTensorDims[],
                                       const ncclDistTensor_t* dst, const size_t dstTensorDims[],
                                       size_t elementsPerChunk, ReshardAlgorithm algo, int dstGpusPerDomain);

/* ======================================================================
 * reshard_transpose.cc — Cross-dim transpose buffer
 * ====================================================================*/

bool shouldTransposeForCrossDim(const size_t* srcDimsBytes, const size_t* dstDimsBytes, int ndims, int srcShardDim,
                                int dstShardDim, int srcShardCount, int dstShardCount, int* swapDimA, int* swapDimB);

ncclResult_t ensureTransposeBuffer(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream);
void* getTransposeBuffer(ncclComm_t comm);
size_t getTransposeBufferCapacity(ncclComm_t comm);
void transposeBufferFinalize();
ncclResult_t transposeBufferRecordEvent(ncclComm_t comm, cudaStream_t stream);

/* ======================================================================
 * staging_prepare.cc -- host-side descriptor builders for ncclReshard.
 * ====================================================================*/

struct StagingTransferDescriptor;
struct StagingKernelParams;

ncclResult_t launchStagingReshardDirect(const StagingKernelParams* hostParams, StagingKernelParams* devParams,
                                        ncclDevComm* devComm, int numCtas, cudaStream_t stream, bool verbose);

ncclResult_t validateStagingPlanLimits(int worldRank, const ncclDistTensor_t* srcTensor,
                                       const size_t* srcTensorDims, const ncclDistTensor_t* dstTensor,
                                       const size_t* dstTensorDims, int gpusPerDomain,
                                       size_t* maxPeerGroupSize = nullptr);

ncclResult_t buildStagingDirectTransferDescriptor(ncclComm_t globalComm, void* srcBuffer, const size_t* srcTensorDims,
                                                  int ndims, const ncclDistTensor_t* srcTensor, void* dstBuffer,
                                                  const size_t* dstTensorDims, const ncclDistTensor_t* dstTensor,
                                                  int gpusPerDomain, int nodeLocalRank,
                                                  StagingTransferDescriptor* desc);

/* ======================================================================
 * reshard_cache.cc -- staging buffer pool
 * ====================================================================*/

struct StagingBufferState;

ncclResult_t ensureStagingBufferPool(ncclComm_t comm, cudaStream_t stream, StagingBufferState** outState);

ncclResult_t stagingBufferPoolRecordEvent(ncclComm_t comm, cudaStream_t stream);

void stagingBufferPoolFinalize();

#endif /* NCCL_RESHARD_INTERNAL_H_ */
