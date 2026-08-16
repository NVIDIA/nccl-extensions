/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "reshard_call_setup.h"

#include <limits>

static ncclResult_t reshardFixFullyReplicated(ncclMesh_t* mesh, int placements[NCCL_RESHARD_MESH_NDIMS]) {
  if (placements[0] == NCCL_RESHARD_REPLICATE && placements[1] == NCCL_RESHARD_REPLICATE) {
    ReshardMeshInterval interval{};
    NCCL_M2N_CHECK(computeReshardMeshInterval(mesh, -1, &interval));
    mesh->dims[0] = interval.size;
    mesh->dims[1] = 1;
    placements[1] = NCCL_RESHARD_SHARD(0);
  }
  return ncclSuccess;
}

ncclResult_t reshardPrepareTensorSetup(const char* apiName, const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                       ReshardTensorSetup* setup) {
  NCCL_M2N_CHECK_ARG(src->mesh != nullptr && dst->mesh != nullptr, -1,
                     "%s: src->mesh and dst->mesh must both be non-null on every rank", apiName);
  NCCL_M2N_CHECK(validateReshardMeshDims(src->mesh, dst->mesh));
  NCCL_M2N_CHECK_ARG(src->ndims == dst->ndims, -1, "%s: src->ndims (%d) and dst->ndims (%d) must match", apiName,
                     src->ndims, dst->ndims);
  NCCL_M2N_CHECK_ARG(src->dtype == dst->dtype, -1, "%s: src->dtype (%d) and dst->dtype (%d) must match", apiName,
                     (int)src->dtype, (int)dst->dtype);
  NCCL_M2N_CHECK_ARG(src->ndims >= 1 && src->ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS, -1,
                     "%s: ndims (%d) out of range [1, %d]", apiName, src->ndims, NCCL_RESHARD_MAX_TENSOR_DIMS);

  setup->ndims = src->ndims;
  setup->elementSize = getNcclDtSize(src->dtype);
  NCCL_M2N_CHECK_ARG(setup->elementSize != 0, -1, "%s: unsupported data type %d", apiName, (int)src->dtype);

  setup->srcMesh = *src->mesh;
  setup->dstMesh = *dst->mesh;
  setup->srcTensor = *src;
  setup->dstTensor = *dst;
  setup->srcTensor.mesh = &setup->srcMesh;
  setup->dstTensor.mesh = &setup->dstMesh;
  NCCL_M2N_CHECK(reshardFixFullyReplicated(&setup->srcMesh, setup->srcTensor.placements));
  NCCL_M2N_CHECK(reshardFixFullyReplicated(&setup->dstMesh, setup->dstTensor.placements));
  NCCL_M2N_CHECK(validateReshardPlacement(&setup->srcTensor, apiName, "src"));
  NCCL_M2N_CHECK(validateReshardPlacement(&setup->dstTensor, apiName, "dst"));
  return ncclSuccess;
}

ncclResult_t reshardValidateActiveBuffers(const char* apiName, int worldRank, const ncclDistTensor_t* src,
                                          const ncclDistTensor_t* dst) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr && src->mesh != nullptr && dst->mesh != nullptr, worldRank,
                     "%s: tensor descriptors and meshes must be non-null", apiName);
  NCCL_M2N_CHECK_ARG(!reshardRankInMesh(src->mesh, worldRank) || src->dataPtr != nullptr, worldRank,
                     "%s: src->dataPtr must be non-null on active source rank %d", apiName, worldRank);
  NCCL_M2N_CHECK_ARG(!reshardRankInMesh(dst->mesh, worldRank) || dst->dataPtr != nullptr, worldRank,
                     "%s: dst->dataPtr must be non-null on active destination rank %d", apiName, worldRank);
  return ncclSuccess;
}

ncclResult_t reshardComputeLocalBytes(int logRank, const char* apiPrefix, const char* side, const void* buffer,
                                      const size_t* dims, int ndims, size_t elementSize, size_t* bytes) {
  *bytes = 0;
  if (buffer == nullptr) {
    return ncclSuccess;
  }
  size_t total = elementSize;
  for (int d = 0; d < ndims; d++) {
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(total, dims[d], &total), logRank,
                       "%s %s local byte size overflow at dim %d: current=%zu dim=%zu", apiPrefix, side, d, total,
                       dims[d]);
  }
  *bytes = total;
  return ncclSuccess;
}

ncclResult_t reshardComputeStagingGinCounts(int logRank, int numCtas, size_t maxPeers, int* signalCount,
                                            int* counterCount) {
  NCCL_M2N_CHECK_ARG(signalCount != nullptr && counterCount != nullptr, logRank,
                     "reshard: staging GIN count outputs must be non-null");
  NCCL_M2N_CHECK_ARG(numCtas > 0 && maxPeers > 0, logRank,
                     "reshard: staging GIN counts require positive numCtas and maxPeers (numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);

  size_t counters = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(static_cast<size_t>(numCtas), maxPeers, &counters) &&
                       counters <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     logRank, "reshard: staging GIN counter count overflows NCCL int field (numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);
  size_t signals = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(counters, static_cast<size_t>(2), &signals) &&
                       signals <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     logRank, "reshard: staging GIN signal count overflows NCCL int field (numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);

  *signalCount = static_cast<int>(signals);
  *counterCount = static_cast<int>(counters);
  return ncclSuccess;
}

ncclResult_t reshardGetOrCreateDevComm(ncclComm_t comm, int numCtas, int ginSignalCount, int ginCounterCount,
                                       ReshardDevCommBarrierKind barrierKind, int ginContextCount, cudaStream_t stream,
                                       ncclDevComm* activeDevComm, ReshardDevCommUse* use) {
  NCCL_M2N_CHECK_ARG(activeDevComm != nullptr && use != nullptr, -1,
                     "reshardGetOrCreateDevComm: output DevComm and use token must be non-null");
  cudaEvent_t completionEvent = nullptr;
  std::shared_ptr<ReshardDevCommUseState> useState;
  const ReshardDevCommCacheKey key = {comm, numCtas, ginSignalCount, ginCounterCount, ginContextCount, barrierKind};
  ncclDevComm* devComm = findCachedDevComm(key, &completionEvent, &useState);
  if (devComm != nullptr) {
    *activeDevComm = *devComm;
    return reshardPrepareDevCommUse(completionEvent, useState, stream, use);
  }

  ncclDevComm localDevComm = {};
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (barrierKind == RESHARD_DEVCOMM_BARRIER_WORLD) {
    reqs.worldGinBarrierCount = numCtas;
  } else {
    reqs.barrierCount = numCtas;
  }
  reqs.ginSignalCount = ginSignalCount;
  reqs.ginCounterCount = ginCounterCount;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginContextCount = ginContextCount;

  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclDevCommCreate(comm, &reqs, &localDevComm));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  ncclResult_t cacheResult = cacheDevComm(key, &localDevComm);
  if (cacheResult != ncclSuccess) {
    {
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(comm, &localDevComm));
    }
    return cacheResult;
  }
  devComm = findCachedDevComm(key, &completionEvent, &useState);
  NCCL_M2N_CHECK_ARG(devComm != nullptr, -1, "reshardGetOrCreateDevComm: newly cached DevComm was not found");
  *activeDevComm = *devComm;
  return reshardPrepareDevCommUse(completionEvent, useState, stream, use);
}
