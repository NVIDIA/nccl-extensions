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
 *      event-based cross-stream ordering, mirroring the transpose
 *      buffer in the window API.
 ************************************************************************/

#include <algorithm>
#ifdef NCCL_M2N_TESTING
#include <atomic>
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "reshard_call_setup.h"
#include "reshard_internal.h"
#include "m2n_log.h"
#include "reshard_types.h"
#include "staging_buffer.h"
#include "staging_profile.h"
#include "staging_types.h"

static void CUDART_CB releaseStagingKernelParams(void* data) {
  delete static_cast<StagingKernelParams*>(data);
}

#ifdef NCCL_M2N_TESTING
static std::atomic<ReshardCopyAlgorithm> gLastCompletedCopyAlgorithm{RESHARD_COPY_ALGO_DIRECT};

ReshardCopyAlgorithm reshardGetLastCompletedCopyAlgorithmForTest() {
  return gLastCompletedCopyAlgorithm.load(std::memory_order_relaxed);
}
#endif

/* ======================================================================
 * ncclReshard — copy/staging-based public entry.
 * ====================================================================*/

extern "C" ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src,
                                    const ncclDistTensor_t* dst, cudaStream_t stream) {
  if (m2nGroupIsActive()) {
    return m2nGroupEnqueueReshard(M2nGroupReshardKind::Staging, handle, comm, nullptr, src, dst, stream);
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
  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));
  int world_rank = 0, world_size = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &world_rank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &world_size));
  reshardResolveAdaptiveScaleConfig(world_size, reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PACKWINDOW);
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

  if (reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PACKWINDOW) {
    NCCL_M2N_CHECK(reshardCopyPackWindowNormalized(comm, &src_local, &dst_local, workStream));
    NCCL_M2N_CHECK(workCompletion.complete());
#ifdef NCCL_M2N_TESTING
    gLastCompletedCopyAlgorithm.store(RESHARD_COPY_ALGO_PACKWINDOW, std::memory_order_relaxed);
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

  StagingBufferState* staging = nullptr;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_ENSURE_BUFFER);
    NCCL_M2N_CHECK(ensureStagingBufferPool(comm, workStream, &staging));
  }

  /* ----------------------------------------------------------------
   * Register a window on the staging buffer for this comm (cached
   * by the internal window cache — same pattern as the transpose
   * buffer in the window API).
   * ---------------------------------------------------------------- */
  ncclWindow_t staging_window = nullptr;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_REGISTER_WINDOW);
    ncclWindow_t* cached_win = findCachedInternalWindowByPtr(comm, staging->buffer, staging->totalSize);
    if (cached_win != nullptr) {
      staging_window = *cached_win;
    } else {
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclCommWindowRegister(comm, staging->buffer, staging->totalSize, &staging_window,
                                              NCCL_WIN_COLL_SYMMETRIC));
      }
      NCCL_M2N_CHECK(cacheInternalWindow(comm, staging->buffer, staging->totalSize, staging_window));
    }
  }

  /* ----------------------------------------------------------------
   * We need a devComm to read lsaSize. Create one with minimal
   * resources first (the staging kernel's devComm is created below
   * with the correct gin counts). Use the same devComm cache.
   * ---------------------------------------------------------------- */
  int staging_num_ctas = 0;
  int lsa_size_from_comm = 0;
  int gpus_per_domain = 1;
  int node_local_rank = 0;
  ncclDevComm* probeDevComm = nullptr;
  ncclDevComm probeLocalDevComm;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_PROBE_DEV_COMM);
    staging_num_ctas = staging->numChannels;

    const ReshardDevCommCacheKey probeKey = {
      comm, staging_num_ctas, 0, 0, staging_num_ctas, RESHARD_DEVCOMM_BARRIER_WORLD
    };
    probeDevComm = findCachedDevComm(probeKey);
    if (probeDevComm == nullptr) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
      ncclDevCommRequirements probeReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
      ncclDevCommRequirements probeReqs = {};
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
      probeReqs.worldGinBarrierCount = staging_num_ctas;
#else
      probeReqs.lsaBarrierCount = staging_num_ctas;
      probeReqs.railGinBarrierCount = staging_num_ctas;
#endif
      probeReqs.ginSignalCount = 0;
      probeReqs.ginCounterCount = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
      probeReqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      probeReqs.ginForceEnable = true;
#endif
      probeReqs.ginContextCount = staging_num_ctas;

      memset(&probeLocalDevComm, 0, sizeof(probeLocalDevComm));
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclDevCommCreate(comm, &probeReqs, &probeLocalDevComm));
      }
      NCCL_M2N_CHECK(cacheDevComm(probeKey, &probeLocalDevComm));
      probeDevComm = findCachedDevComm(probeKey);
      if (probeDevComm == nullptr) {
        probeDevComm = &probeLocalDevComm;
      }
    }

    /* Derive gpus_per_domain from lsaSize (Change 1). */
    lsa_size_from_comm = (probeDevComm->lsaSize > 0) ? probeDevComm->lsaSize : 0;
    const int gpus_per_node_cfg = reshardGetGpusPerNode();
    if (lsa_size_from_comm > 0) {
      gpus_per_domain = lsa_size_from_comm;
    } else {
      gpus_per_domain = (gpus_per_node_cfg > 0) ? gpus_per_node_cfg : 1;
    }

    node_local_rank = world_rank % gpus_per_domain;
  }

  StagingTransferDescriptor desc;
  size_t maxPeerGroupSize = 1;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_BUILD_DESCRIPTOR);
    NCCL_M2N_CHECK(validateStagingPlanLimits(world_rank, &src_local, src_dims_bytes, &dst_local, dst_dims_bytes,
                                             reshardGetCopyAlgorithm(), gpus_per_domain, &maxPeerGroupSize));
    memset(&desc, 0, sizeof(desc));
    NCCL_M2N_CHECK(buildStagingDirectTransferDescriptor(comm, srcBuffer, src_dims_bytes, ndims, &src_local, dstBuffer,
                                                        dst_dims_bytes, &dst_local, gpus_per_domain, node_local_rank,
                                                        &desc));
  }

  size_t effectiveChunkSize = 0;
  NCCL_M2N_CHECK(stagingResolveEffectiveChunkSize(staging, maxPeerGroupSize, &effectiveChunkSize));

  /* ----------------------------------------------------------------
   * Pack the descriptor into device-ready params.
   * Use the same window for both RDMA and LSA (Change 2 — single
   * window on the main comm).
   * ---------------------------------------------------------------- */
  std::unique_ptr<StagingKernelParams> paramsOwner(new (std::nothrow) StagingKernelParams);
  if (paramsOwner == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, world_rank, "Failed to allocate host memory for staging kernel parameters");
  }
  StagingKernelParams& params = *paramsOwner;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_PREPARE_PARAMS);
    NCCL_M2N_CHECK(
      stagingPrepareTransfer(staging, &desc, staging_window, staging_window, effectiveChunkSize, &params));

  }

  /* ----------------------------------------------------------------
   * Deterministic upper-bound on gin counts — replaces the former
   * AllReduce(MAX) with a locally-computable bound so that every
   * rank derives the same ginSignalCount without a collective.
   *
   * max_peers_bound >= max(numTargets, numSources) on ANY rank:
   *   src_total — bounds a dest rank's numSources
   *   dst_total — bounds a source rank's numTargets
   *   gpus_per_domain — bounds a dest leader's LSA fan-out targets
   * This is now O(1).
  * ---------------------------------------------------------------- */
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_RESOLVE_GIN_COUNTS);
    ReshardMeshInterval srcInterval{};
    ReshardMeshInterval dstInterval{};
    NCCL_M2N_CHECK(computeReshardMeshInterval(src_mesh, world_rank, &srcInterval));
    NCCL_M2N_CHECK(computeReshardMeshInterval(dst_mesh, world_rank, &dstInterval));
    const size_t maxPeersBound = std::max({static_cast<size_t>(srcInterval.size),
                                           static_cast<size_t>(dstInterval.size),
                                           static_cast<size_t>(gpus_per_domain)});
    NCCL_M2N_CHECK(reshardComputeStagingGinCounts(world_rank, params.numChannels, maxPeersBound,
                                                  &params.ginSignalCount, &params.ginCounterCount));
    RESHARD_DEBUG(world_rank,
                  "[STAGING] gin deterministic bound: src_total=%d dst_total=%d "
                  "gpus_per_domain=%d max_peers_bound=%zu "
                  "gin_signal=%d gin_counter=%d",
                  srcInterval.size, dstInterval.size, gpus_per_domain, maxPeersBound, params.ginSignalCount,
                  params.ginCounterCount);
  }

  /* Get/create a devComm sized for the staging kernel's actual resource needs. */
  ncclDevComm* devCommPtr = nullptr;
  ncclDevComm localDevComm;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_GET_DEV_COMM);
    const ReshardDevCommCacheKey devCommKey = {
      comm, staging_num_ctas, params.ginSignalCount, params.ginCounterCount, staging_num_ctas,
      RESHARD_DEVCOMM_BARRIER_WORLD
    };
    devCommPtr = findCachedDevComm(devCommKey);
    if (devCommPtr == nullptr) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
      ncclDevCommRequirements reqs = {};
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
      reqs.worldGinBarrierCount = staging_num_ctas;
#else
      reqs.lsaBarrierCount = staging_num_ctas;
      reqs.railGinBarrierCount = staging_num_ctas;
#endif
      reqs.ginSignalCount = params.ginSignalCount;
      reqs.ginCounterCount = params.ginCounterCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
      reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs.ginForceEnable = true;
#endif
      reqs.ginContextCount = staging_num_ctas;

      memset(&localDevComm, 0, sizeof(localDevComm));
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclDevCommCreate(comm, &reqs, &localDevComm));
      }
      NCCL_M2N_CHECK(cacheDevComm(devCommKey, &localDevComm));
      devCommPtr = findCachedDevComm(devCommKey);
      if (devCommPtr == nullptr) {
        devCommPtr = &localDevComm;
      }
    }
  }

  RESHARD_INFO(world_rank,
               "ncclReshard: num_ctas=%d gin_signal=%d gin_counter=%d "
               "gpus_per_domain=%d lsa_size=%d src_buf=%p dst_buf=%p staging=%p",
               staging_num_ctas, params.ginSignalCount, params.ginCounterCount, gpus_per_domain, lsa_size_from_comm,
               srcBuffer, dstBuffer, staging->buffer);

  /* Launch the staging kernel. */
  ncclResult_t launchResult = ncclSuccess;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_LAUNCH_KERNEL);
    const bool verbose = debugLogging;
    launchResult =
      launchStagingReshardDirect(&params, staging->devParams, devCommPtr, staging_num_ctas, workStream, verbose);
  }

  cudaError_t releaseResult = cudaLaunchHostFunc(workStream, releaseStagingKernelParams, paramsOwner.get());
  if (releaseResult != cudaSuccess) {
    paramsOwner.release();
    RESHARD_WARN(world_rank,
                 "Unable to enqueue staging parameter cleanup (%s); retaining the parameter block until process "
                 "exit",
                 cudaGetErrorString(releaseResult));
  } else {
    paramsOwner.release();
  }

  if (launchResult != ncclSuccess) {
    NCCL_M2N_CHECK_WARN(stagingBufferPoolRecordEvent(comm, workStream));
    NCCL_M2N_CHECK_WARN(workCompletion.complete());
    return launchResult;
  }

  if (profile != nullptr) {
    profile->log(world_rank);
  }

  /* Record event on the staging buffer for cross-stream ordering
   * (Change 3 — per-comm buffer with events). */
  NCCL_M2N_CHECK(stagingBufferPoolRecordEvent(comm, workStream));

  NCCL_M2N_CHECK(workCompletion.complete());
#ifdef NCCL_M2N_TESTING
  gLastCompletedCopyAlgorithm.store(RESHARD_COPY_ALGO_DIRECT, std::memory_order_relaxed);
#endif

  return ncclSuccess;
}
