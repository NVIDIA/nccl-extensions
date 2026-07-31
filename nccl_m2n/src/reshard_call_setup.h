/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_RESHARD_CALL_SETUP_H_
#define NCCL_M2N_RESHARD_CALL_SETUP_H_

#include <limits>

#include "cuda_runtime.h"
#include "nccl.h"

#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "nccl_m2n.h"
#include "reshard_internal.h"
#include "reshard_types.h"

struct ReshardWorkStream {
  cudaStream_t stream;
  cudaEvent_t readyEvent;
  cudaEvent_t doneEvent;
};

static inline bool reshardAcquiredPoolSlot(const ReshardWorkStream* work) {
  return work->doneEvent != nullptr;
}

static inline ncclResult_t reshardFixFullyReplicated(ncclMesh_t* mesh,
                                                     int placements[NCCL_RESHARD_MESH_NDIMS]) {
  if (placements[0] == NCCL_RESHARD_REPLICATE && placements[1] == NCCL_RESHARD_REPLICATE) {
    ReshardMeshInterval interval{};
    NCCL_M2N_CHECK(computeReshardMeshInterval(mesh, -1, &interval));
    mesh->dims[0] = interval.size;
    mesh->dims[1] = 1;
    placements[1] = NCCL_RESHARD_SHARD(0);
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardValidateActiveBuffers(const char* apiName, int worldRank,
                                                        const ncclDistTensor_t* src, const ncclDistTensor_t* dst) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr && src->mesh != nullptr && dst->mesh != nullptr, worldRank,
                     "[%s] tensor descriptors and meshes must be non-null", apiName);
  NCCL_M2N_CHECK_ARG(!reshardRankInMesh(src->mesh, worldRank) || src->dataPtr != nullptr, worldRank,
                     "[%s] src->dataPtr must be non-null on active source rank %d", apiName, worldRank);
  NCCL_M2N_CHECK_ARG(!reshardRankInMesh(dst->mesh, worldRank) || dst->dataPtr != nullptr, worldRank,
                     "[%s] dst->dataPtr must be non-null on active destination rank %d", apiName, worldRank);
  return ncclSuccess;
}

static inline ncclResult_t reshardMatchCommCudaDevice(ncclComm_t comm, int* currentCudaDev,
                                                      ncclCommProperties* commProps, ncclResult_t* propsResult) {
  NCCL_M2N_CUDACHECK(cudaGetDevice(currentCudaDev));
  *commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  *propsResult = ncclCommQueryProperties(comm, commProps);
  if (*propsResult == ncclSuccess && *currentCudaDev != commProps->cudaDev) {
    NCCL_M2N_CUDACHECK(cudaSetDevice(commProps->cudaDev));
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardSetupWorkStream(ncclComm_t comm, cudaStream_t callerStream, int currentCudaDev,
                                                  ncclResult_t propsResult, const ncclCommProperties* commProps,
                                                  ReshardWorkStream* work) {
  work->stream = callerStream;
  work->readyEvent = nullptr;
  work->doneEvent = nullptr;

  const bool isDefaultStream =
    (callerStream == nullptr || callerStream == cudaStreamLegacy || callerStream == cudaStreamPerThread);
  const bool wantPool = isDefaultStream && reshardGetStreamPoolSize() > 0;
  if (wantPool) {
    const int dev = (propsResult == ncclSuccess) ? commProps->cudaDev : currentCudaDev;
    NCCL_M2N_CHECK(streamPoolAcquire(comm, dev, &work->stream, &work->readyEvent, &work->doneEvent));
    if (work->stream == nullptr) {
      work->stream = callerStream;
    } else {
      NCCL_M2N_CUDACHECK(cudaEventRecord(work->readyEvent, callerStream));
      NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(work->stream, work->readyEvent, 0));
    }
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardCompleteWorkStream(cudaStream_t callerStream, const ReshardWorkStream* work) {
  if (reshardAcquiredPoolSlot(work)) {
    NCCL_M2N_CUDACHECK(cudaEventRecord(work->doneEvent, work->stream));
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(callerStream, work->doneEvent, 0));
  }
  return ncclSuccess;
}

class ReshardWorkStreamCompletion {
public:
  ReshardWorkStreamCompletion(cudaStream_t callerStream, const ReshardWorkStream* work)
    : callerStream_(callerStream), work_(work) {}
  ReshardWorkStreamCompletion(const ReshardWorkStreamCompletion&) = delete;
  ReshardWorkStreamCompletion& operator=(const ReshardWorkStreamCompletion&) = delete;
  ~ReshardWorkStreamCompletion() {
    if (bActive_) {
      // Preserve the caller's primary error; success paths return complete() directly.
      (void)complete();
    }
  }

  ncclResult_t complete() {
    if (!bActive_) {
      return ncclSuccess;
    }
    bActive_ = false;
    return reshardCompleteWorkStream(callerStream_, work_);
  }

private:
  cudaStream_t callerStream_;
  const ReshardWorkStream* work_;
  bool bActive_ = true;
};

static inline ncclResult_t reshardDimsToBytes(int logRank, const char* apiName, int ndims, size_t elementSize,
                                              const size_t* srcDims, const size_t* dstDims,
                                              size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                              size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  NCCL_M2N_CHECK_ARG(srcDims != nullptr && dstDims != nullptr, logRank,
                     "%s source and destination shape metadata must be present on every rank", apiName);
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    srcDimsBytes[d] = 0;
    dstDimsBytes[d] = 0;
  }
  for (int d = 0; d < ndims; d++) {
    srcDimsBytes[d] = srcDims[d];
    dstDimsBytes[d] = dstDims[d];
  }
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcDimsBytes[ndims - 1], elementSize, &srcDimsBytes[ndims - 1]), logRank,
                     "%s source last dimension byte-size overflow: dim=%zu elementSize=%zu", apiName,
                     srcDims[ndims - 1], elementSize);
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dstDimsBytes[ndims - 1], elementSize, &dstDimsBytes[ndims - 1]), logRank,
                     "%s destination last dimension byte-size overflow: dim=%zu elementSize=%zu", apiName,
                     dstDims[ndims - 1], elementSize);
  return ncclSuccess;
}

static inline ncclResult_t reshardComputeStagingGinCounts(int logRank, int numCtas, size_t maxPeers,
                                                          int* signalCount, int* counterCount) {
  NCCL_M2N_CHECK_ARG(signalCount != nullptr && counterCount != nullptr, logRank,
                     "reshard: staging GIN count outputs must be non-null");
  NCCL_M2N_CHECK_ARG(numCtas > 0 && maxPeers > 0, logRank,
                     "reshard: staging GIN counts require positive numCtas and maxPeers "
                     "(numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);

  size_t counters = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(static_cast<size_t>(numCtas), maxPeers, &counters) &&
                       counters <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     logRank, "reshard: staging GIN counter count overflows NCCL int field "
                              "(numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);
  size_t signals = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(counters, static_cast<size_t>(2), &signals) &&
                       signals <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     logRank, "reshard: staging GIN signal count overflows NCCL int field "
                              "(numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);

  *signalCount = static_cast<int>(signals);
  *counterCount = static_cast<int>(counters);
  return ncclSuccess;
}

#endif // NCCL_M2N_RESHARD_CALL_SETUP_H_
