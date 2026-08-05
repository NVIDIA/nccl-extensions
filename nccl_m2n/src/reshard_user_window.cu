/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — User-Window Path
 *
 * Single-shot resharding entry point that takes a caller-registered
 * ncclWindow_t and runs the RING or DIRECT algorithm.
 *
 * Caller contract (single-offset, symmetric, single window):
 *   - On every rank that participates, src/dst buffers must lie within
 *     the registered window, and srcBuffer and dstBuffer must share
 *     the same offset within it (single-offset assumption — the kernel
 *     uses one params.myWindowOffset field per rank).
 *   - All ranks must agree on that offset (symmetric assumption).
 *   - Window registered on the input comm (NOT a node-local sub-comm).
 *   - LSA fan-out walks the input comm's LSA team.
 *
 * Algorithm selection follows the NCCL_RESHARD_ALGORITHM env var:
 *   RING   -> reshardKernelUserWindow
 *   DIRECT -> directReshardKernelUserWindow
 *   AUTO   -> RING
 ************************************************************************/

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "reshard_types.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "m2n_log.h"
#include "reshard_call_setup.h"
#include "reshard_internal.h"
#include "reshard_kernels.cuh"

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
#define NCCL_RESHARD_GIN_FINAL_FENCE (ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)
#else
#define NCCL_RESHARD_GIN_FINAL_FENCE ncclGinFenceLevel::Relaxed
#endif

// ============================================================================
// Byte-level transpose kernel: [D0, D1, D2] -> [D0, D2, D1]  (row-major)
// ============================================================================

__global__ void uwTranspose2DInnerKernel(const char* __restrict__ in, char* __restrict__ out, size_t D0, size_t D1,
                                         size_t D2) {
  size_t total = D0 * D1 * D2;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;

  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = i % D2;
    size_t rem = i / D2;
    size_t d1 = rem % D1;
    size_t d0 = rem / D1;

    size_t j = d0 * (D2 * D1) + d2 * D1 + d1;
    out[j] = in[i];
  }
}

static void uwLaunchTranspose2DInner(const void* in, void* out, size_t D0, size_t D1, size_t D2, cudaStream_t stream) {
  size_t total = D0 * D1 * D2;
  int blockSize = 256;
  size_t needed = (total + blockSize - 1) / blockSize;
  int numBlocks = (int)std::min(needed, (size_t)65535);
  uwTranspose2DInnerKernel<<<numBlocks, blockSize, 0, stream>>>((const char*)in, (char*)out, D0, D1, D2);
}

// ============================================================================
// RING (hierarchical) Kernel — User-Window Variant
//
// Uses the GLOBAL window for local-buffer access and LSA fan-out, resolving
// peer pointers via world-rank arithmetic.
// ============================================================================

__global__ __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) void reshardKernelUserWindow(ncclReshardParams params,
                                                                                          struct ncclDevComm devComm) {
  int numContexts = min((int)gridDim.x, (int)devComm.ginContextCount);
  int ctasPerContext = (int)gridDim.x / numContexts;
  int ginContext = (int)blockIdx.x / ctasPerContext;
  ncclGin gin{devComm, ginContext};

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);

  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  // [USER-WINDOW] Local pointer comes from the GLOBAL window.  Offset is
  // zero by contract but
  // we still pass params.myWindowOffset for symmetry.
  char* localBuffer = (char*)ncclGetLocalPointer(params.window, params.myWindowOffset);

  __shared__ uint64_t initialSignals[MAX_SOURCES];
  // Compile-time guard: shared-array dim must be at least the prep-side cap
  // on params.numSources.
  static_assert(sizeof(initialSignals) / sizeof(initialSignals[0]) >= MAX_SOURCES,
                "initialSignals[] must be sized at least MAX_SOURCES — "
                "kernel reads initialSignals[i] for i < params.numSources, "
                "and prepareReshardParams caps params.numSources at MAX_SOURCES");

  // Read initial signals (dest ranks only)
  if (params.isDest && params.numSources > 0) {
    if (threadIdx.x < params.numSources) {
      unsigned int signalIdx = params.sources[threadIdx.x].signalBase + blockIdx.x;
      initialSignals[threadIdx.x] = gin.readSignal(signalIdx);
    }
  }
  __syncthreads();

  // Initial barrier
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // SOURCE: send data to targets
  if (params.isSource) {
    int activeSrcWarps = min(MAX_SRC_WARPS, (int)(blockDim.x / 32));
    if (warpId < activeSrcWarps) {
      for (int t = warpId; t < params.numTargets; t += activeSrcWarps) {
        ncclReshardTargetInfo& target = params.targets[t];
        ncclReshardTransferPlan& plan = target.plan;

        if (plan.totalInnerTransfers == 0) continue;

        unsigned int signalIdx = params.mySignalBase + blockIdx.x;

        if (target.isContiguous) {
          size_t totalSize = target.totalBytes;
          size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
          size_t myStart = blockIdx.x * bytesPerCta;
          size_t myEnd = min(myStart + bytesPerCta, totalSize);

          if (myStart < totalSize && laneId == 0) {
            gin.put(world, target.dstWorldRank, params.window, target.windowOffset + plan.dstBaseOffset + myStart,
                    params.window, params.myWindowOffset + plan.srcBaseOffset + myStart, myEnd - myStart,
                    ncclGin_SignalInc{signalIdx});
          }
        } else {
          size_t itersPerCta = (plan.totalInnerTransfers + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, plan.totalInnerTransfers);
          size_t myTotalBytes = (myIterEnd - myIterStart) * plan.innerSize;
          const size_t csb = params.chunkSizeBytes;
          size_t numChunks = (myTotalBytes + csb - 1) / csb;

          for (size_t chunk = 0; chunk < numChunks; chunk++) {
            size_t byteStart = chunk * csb;
            size_t remaining = myTotalBytes - byteStart;
            size_t thisBytes = (csb < remaining) ? csb : remaining;

            if (laneId == 0) {
              emitStridedChunkPuts(gin, world, target.dstWorldRank, params.window, plan, params.ndims, myIterStart,
                                   byteStart, thisBytes, signalIdx,
                                   /*useDstAsSrc=*/false, params.myWindowOffset, target.windowOffset);
            }
          }
        }
      }
    }
  }

  // DEST: receive and replicate
  if (params.isDest && params.numSources > 0) {
    int warpsPerCta = blockDim.x / 32;
    int activeSources = min(params.numSources, min(warpsPerCta, (int)MAX_WARP_GROUPS));
    int warpsPerSource = warpsPerCta / activeSources;
    if (warpsPerSource < 1) warpsPerSource = 1;

    int mySourceGroup = warpId / warpsPerSource;
    int warpInGroup = warpId % warpsPerSource;
    int groupStartWarp = mySourceGroup * warpsPerSource;
    bool isActive = (mySourceGroup < activeSources);

    // [USER-WINDOW] lsaStartRank = world rank of LSA-rank-0 on this
    // rank's LSA team within the input comm.  Used to translate world
    // ranks to LSA ranks for ncclGetLsaPointer on the global window.
    const int lsaStartRank = world.rank - lsa.rank;

    for (int srcOffset = mySourceGroup; srcOffset < params.numSources && isActive; srcOffset += activeSources) {
      ncclReshardSourceInfo& source = params.sources[srcOffset];
      ncclReshardTransferPlan& plan = source.plan;

      int barrierId = mySourceGroup;
      ncclCoopWarpSpan warps(groupStartWarp, warpsPerSource, barrierId);

      unsigned int signalIdx = source.signalBase + blockIdx.x;

      if (source.isContiguous) {
        size_t totalSize = source.totalBytes;
        size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
        size_t myStart = blockIdx.x * bytesPerCta;
        size_t myEnd = min(myStart + bytesPerCta, totalSize);

        if (warpInGroup == 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + 1);
        warps.sync();

        // Ring forward
        if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
          if (myStart < totalSize) {
            gin.put(world, params.ringNextWorldRank, params.window,
                    params.ringNextWindowOffset + plan.dstBaseOffset + myStart, params.window,
                    params.myWindowOffset + plan.dstBaseOffset + myStart, myEnd - myStart,
                    ncclGin_SignalInc{signalIdx});
          }
        }

        // [USER-WINDOW] LSA fan-out via the global window keyed by
        // world-rank arithmetic.
        if (params.numLocalFollowers > 0 && myStart < totalSize) {
          int threadsInGroup = warpsPerSource * 32;
          int threadInGroup = warpInGroup * 32 + laneId;

          char* srcPtr = localBuffer + plan.dstBaseOffset + myStart;
          size_t chunkSize = myEnd - myStart;
          size_t dstByteOffset = plan.dstBaseOffset + myStart;

          lsaReplicateChunk(srcPtr, chunkSize, dstByteOffset,
                            /*fallbackWindow=*/params.window, params.localFollowerWorldRanks,
                            params.localFollowerWindowOffsets, lsaStartRank, params.numLocalFollowers, threadsInGroup,
                            threadInGroup);
        }

        warps.sync();
      } else {
        size_t itersPerCta = (plan.totalInnerTransfers + params.totalCtas - 1) / params.totalCtas;
        size_t myIterStart = blockIdx.x * itersPerCta;
        size_t myIterEnd = min(myIterStart + itersPerCta, plan.totalInnerTransfers);
        size_t myTotalBytes = (myIterEnd - myIterStart) * plan.innerSize;
        const size_t csb = params.chunkSizeBytes;
        size_t numChunks = (myTotalBytes + csb - 1) / csb;

        for (size_t chunk = 0; chunk < numChunks; chunk++) {
          size_t byteStart = chunk * csb;
          size_t remaining = myTotalBytes - byteStart;
          size_t thisBytes = (csb < remaining) ? csb : remaining;

          if (warpInGroup == 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + chunk + 1);
          warps.sync();

          // Ring forward this chunk
          if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
            emitStridedChunkPuts(gin, world, params.ringNextWorldRank, params.window, plan, params.ndims, myIterStart,
                                 byteStart, thisBytes, signalIdx,
                                 /*useDstAsSrc=*/true, params.myWindowOffset, params.ringNextWindowOffset);
          }

          // [USER-WINDOW] LSA fan-out via the global window for
          // strided chunks.
          if (params.numLocalFollowers > 0 && thisBytes > 0) {
            int threadsInGroup = warpsPerSource * 32;
            int threadInGroup = warpInGroup * 32 + laneId;

            const size_t inner = plan.innerSize;
            size_t lsaIter = byteStart / inner;
            size_t lsaOffInIter = byteStart % inner;
            size_t lsaRemaining = thisBytes;

            while (lsaRemaining > 0) {
              size_t avail = inner - lsaOffInIter;
              size_t piece = (avail < lsaRemaining) ? avail : lsaRemaining;

              size_t srcOff, dstOff;
              computeTransferOffset(plan, myIterStart + lsaIter, params.ndims, &srcOff, &dstOff);

              char* piecePtr = localBuffer + dstOff + lsaOffInIter;
              size_t dstByteOffset = dstOff + lsaOffInIter;

              lsaReplicateChunk(piecePtr, piece, dstByteOffset,
                                /*fallbackWindow=*/params.window, params.localFollowerWorldRanks,
                                params.localFollowerWindowOffsets, lsaStartRank, params.numLocalFollowers,
                                threadsInGroup, threadInGroup);

              lsaRemaining -= piece;
              lsaIter++;
              lsaOffInIter = 0;
            }
          }
        }
      }
    }
  }

  __threadfence_system();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
  __syncthreads();
#else
  // Older GIN barriers do not drain puts; flush also synchronizes the CTA.
  gin.flush(ncclCoopCta());
#endif

  // Final barrier
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCL_RESHARD_GIN_FINAL_FENCE);
}

// ============================================================================
// DIRECT Algorithm Kernel — User-Window Variant
// ============================================================================

// clang-format off
__global__
__launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) void
directReshardKernelUserWindow(
    ncclReshardDirectParams params,
    struct ncclDevComm      devComm)
// clang-format on
{
  int numContexts = min((int)gridDim.x, (int)devComm.ginContextCount);
  int ctasPerContext = (int)gridDim.x / numContexts;
  int ginContext = (int)blockIdx.x / ctasPerContext;
  ncclGin gin{devComm, ginContext};

  ncclTeam world = ncclTeamWorld(devComm);

  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  __shared__ uint64_t initialSignals[MAX_DIRECT_SOURCES];
  // Compile-time guard: shared-array dim must be at least the prep-side cap
  // on params.numSources.
  static_assert(sizeof(initialSignals) / sizeof(initialSignals[0]) >= MAX_DIRECT_SOURCES,
                "initialSignals[] must be sized at least MAX_DIRECT_SOURCES — "
                "kernel reads initialSignals[i] for i < params.numSources, "
                "and prepareDirectReshardParams caps params.numSources at "
                "MAX_DIRECT_SOURCES");

  if (params.isDest && params.numSources > 0) {
    if (threadIdx.x < params.numSources) {
      unsigned int signalIdx = params.sources[threadIdx.x].signalBase + blockIdx.x;
      initialSignals[threadIdx.x] = gin.readSignal(signalIdx);
    }
  }
  __syncthreads();

  // Initial barrier
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
#else
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
#endif
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (params.isSource && params.numTargets > 0) {
    bool isGinWarp = (warpId == 0);

    if (isGinWarp) {
      for (int t = 0; t < params.numTargets; t++) {
        ncclReshardDirectTargetInfo& target = params.targets[t];

        if (target.isContiguous) {
          size_t totalBytes = target.totalBytes;
          size_t bytesPerCta = (totalBytes + params.totalCtas - 1) / params.totalCtas;
          size_t myByteStart = blockIdx.x * bytesPerCta;
          size_t myByteEnd = min(myByteStart + bytesPerCta, totalBytes);

          if (myByteStart < totalBytes) {
            size_t myBytes = myByteEnd - myByteStart;
            size_t srcOffset = params.myWindowOffset + target.plan.srcBaseOffset + myByteStart;
            size_t dstOffset = target.windowOffset + target.plan.dstBaseOffset + myByteStart;

            unsigned int signalIdx = params.mySignalBase + blockIdx.x;

            if (laneId == 0) {
              gin.put(world, target.dstWorldRank, params.window, dstOffset, params.window, srcOffset, myBytes,
                      ncclGin_SignalInc{signalIdx});
            }
          }
        } else {
          size_t totalIters = target.plan.totalInnerTransfers;
          size_t itersPerCta = (totalIters + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, totalIters);

          if (myIterStart < totalIters) {
            for (size_t iter = myIterStart; iter < myIterEnd; iter++) {
              size_t srcOffset, dstOffset;
              computeTransferOffset(target.plan, iter, params.ndims, &srcOffset, &dstOffset);
              srcOffset += params.myWindowOffset;
              dstOffset += target.windowOffset;

              unsigned int signalIdx = params.mySignalBase + blockIdx.x;

              if (laneId == 0) {
                gin.put(world, target.dstWorldRank, params.window, dstOffset, params.window, srcOffset,
                        target.plan.innerSize, ncclGin_SignalInc{signalIdx});
              }
            }
          }
        }
      }
    }
  }

  if (params.isDest && params.numSources > 0) {
    if (warpId == 0) {
      for (int s = 0; s < params.numSources; s++) {
        ncclReshardDirectSourceInfo& source = params.sources[s];
        unsigned int signalIdx = source.signalBase + blockIdx.x;
        uint64_t initialSignal = initialSignals[s];

        size_t mySignals;
        if (source.isContiguous) {
          size_t totalBytes = source.totalBytes;
          size_t bytesPerCta = (totalBytes + params.totalCtas - 1) / params.totalCtas;
          size_t myByteStart = blockIdx.x * bytesPerCta;
          mySignals = (myByteStart < totalBytes) ? 1 : 0;
        } else {
          size_t totalIters = source.plan.totalInnerTransfers;
          size_t itersPerCta = (totalIters + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, totalIters);
          mySignals = (myIterStart < totalIters) ? (myIterEnd - myIterStart) : 0;
        }

        if (mySignals > 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignal + mySignals);
      }
    }
  }

  __syncthreads();
  gin.flush(ncclCoopCta());

  // Final barrier
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCL_RESHARD_GIN_FINAL_FENCE);
}

// ============================================================================
// Host: ncclReshardWithWindow
// ============================================================================

extern "C" ncclResult_t ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window,
                                               const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                               cudaStream_t stream) {
  M2nApiLock apiLock;
  m2nClearLastError();
  /* Required handles. */
  NCCL_M2N_CHECK_ARG(comm != nullptr && window != nullptr, -1,
                     "ncclReshardWithWindow: comm and window must both be non-null");
  /* Both descriptors required — each carries one side's mesh and the
     library reads both meshes on every rank.  A rank that does not
     have data on a given side still passes a fully-formed descriptor
     with dataPtr=NULL (the same convention PyTorch DTensor uses with
     a size-0 local tensor on non-participating ranks). */
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr, -1,
                     "ncclReshardWithWindow: src and dst tensor descriptors must both be non-null on every rank "
                     "(use dataPtr=NULL on the side this rank doesn't participate in)");
  NCCL_M2N_CHECK_ARG(src->mesh != nullptr && dst->mesh != nullptr, -1,
                     "ncclReshardWithWindow: src->mesh and dst->mesh must both be non-null on every rank");
  NCCL_M2N_CHECK(validateReshardMeshDims(src->mesh, dst->mesh));
  NCCL_M2N_CHECK_ARG(src->ndims == dst->ndims, -1,
                     "ncclReshardWithWindow: src->ndims (%d) and dst->ndims (%d) must match", src->ndims,
                     dst->ndims);
  NCCL_M2N_CHECK_ARG(src->dtype == dst->dtype, -1,
                     "ncclReshardWithWindow: src->dtype (%d) and dst->dtype (%d) must match", (int)src->dtype,
                     (int)dst->dtype);
  int ndims = src->ndims;
  ncclDataType_t dtype = src->dtype;
  void* srcBuffer = src->dataPtr;
  void* dstBuffer = dst->dataPtr;
  const size_t* srcTensorDims = src->localShape;
  const size_t* dstTensorDims = dst->localShape;
  // Use mutable local descriptor copies for internal rewrites while
  // keeping the public caller-owned descriptors unchanged.
  ncclMesh_t srcMeshLocal = {{src->mesh->dims[0], src->mesh->dims[1]}, src->mesh->startRank};
  ncclMesh_t dstMeshLocal = {{dst->mesh->dims[0], dst->mesh->dims[1]}, dst->mesh->startRank};
  ncclDistTensor_t srcTensorLocal = *src;
  ncclDistTensor_t dstTensorLocal = *dst;
  srcTensorLocal.mesh = &srcMeshLocal;
  dstTensorLocal.mesh = &dstMeshLocal;
  NCCL_M2N_CHECK_ARG(ndims >= 1 && ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS, -1,
                     "ncclReshardWithWindow: ndims (%d) out of range [1, %d]", ndims,
                     NCCL_RESHARD_MAX_TENSOR_DIMS);
  size_t elementSize = getNcclDtSize(dtype);
  NCCL_M2N_CHECK_ARG(elementSize != 0, -1, "ncclReshardWithWindow: unsupported data type %d", (int)dtype);

  NCCL_M2N_CHECK(reshardFixFullyReplicated(&srcMeshLocal, srcTensorLocal.placements));
  NCCL_M2N_CHECK(reshardFixFullyReplicated(&dstMeshLocal, dstTensorLocal.placements));
  NCCL_M2N_CHECK(validateReshardPlacement(&srcTensorLocal, "ncclReshardWithWindow", "src"));
  NCCL_M2N_CHECK(validateReshardPlacement(&dstTensorLocal, "ncclReshardWithWindow", "dst"));
  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));

  const ncclDistTensor_t* srcTensor = &srcTensorLocal;
  const ncclDistTensor_t* dstTensor = &dstTensorLocal;
  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;

  int worldRank, worldSize;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));
  NCCL_M2N_CHECK(validateReshardMeshBounds(srcMesh, dstMesh, worldSize, worldRank));
  NCCL_M2N_CHECK(reshardValidateActiveBuffers("ncclReshardWithWindow", worldRank, srcTensor, dstTensor));

  int currentCudaDev = 0;
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclSuccess;
  NCCL_M2N_CHECK(reshardMatchCommCudaDevice(comm, &currentCudaDev, &commProps, &propsResult));

  // Internal-stream mode routes every caller through a library-owned
  // non-blocking stream. Readiness and completion events preserve the caller
  // stream's ordering; NCCL_RESHARD_USE_INTERNAL_STREAMS=0 uses the caller stream.
  ReshardWorkStream work{};
  ncclResult_t setupResult = reshardSetupWorkStream(comm, stream, currentCudaDev, propsResult, &commProps, &work);
  if (setupResult != ncclSuccess) {
    return setupResult;
  }
  ReshardWorkStreamCompletion workCompletion(stream, &work);
  cudaStream_t workStream = work.stream;

  // ------------------------------------------------------------------
  // Single-offset contract.  Each rank verifies that srcBuffer and
  // dstBuffer (when both present) share the same offset within the
  // registered window — the kernel uses one params.myWindowOffset
  // field per rank, so the two sides must agree.  Cross-rank symmetry
  // (every rank computes the same offset) is trusted under this
  // release; the opt-in §1d diagnostic verifies it when needed.
  //
  // localOffset stays in scope through the rest of the function and
  // gets threaded into the kernel params at launch.
  // ------------------------------------------------------------------
  intptr_t localOffset = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2)
  {
    void* winUserPtr = nullptr;
    NCCL_M2N_CHECK(ncclWinGetUserPtr(comm, window, &winUserPtr));
    NCCL_M2N_CHECK_ARG(winUserPtr != nullptr, worldRank,
                       "ncclReshardWithWindow: ncclWinGetUserPtr returned null for the supplied window; "
                       "a symmetric-memory window (NCCL_WIN_COLL_SYMMETRIC) is required. This is a collective API; "
                       "all ranks must satisfy the window-offset contract or peers may wait indefinitely");

    const bool hasSrc = (srcBuffer != nullptr);
    const bool hasDst = (dstBuffer != nullptr);
    const uintptr_t windowAddress = reinterpret_cast<uintptr_t>(winUserPtr);
    const uintptr_t srcAddress = reinterpret_cast<uintptr_t>(srcBuffer);
    const uintptr_t dstAddress = reinterpret_cast<uintptr_t>(dstBuffer);
    NCCL_M2N_CHECK_ARG(!hasSrc || srcAddress >= windowAddress, worldRank,
                       "ncclReshardWithWindow: srcBuffer lies before the window base "
                       "(srcBuffer=%p, window_user_ptr=%p). This is a collective API; all ranks must satisfy the "
                       "window-offset contract or peers may wait indefinitely",
                       srcBuffer, winUserPtr);
    NCCL_M2N_CHECK_ARG(!hasDst || dstAddress >= windowAddress, worldRank,
                       "ncclReshardWithWindow: dstBuffer lies before the window base "
                       "(dstBuffer=%p, window_user_ptr=%p). This is a collective API; all ranks must satisfy the "
                       "window-offset contract or peers may wait indefinitely",
                       dstBuffer, winUserPtr);

    const uintptr_t maxWindowOffset = static_cast<uintptr_t>(INTPTR_MAX);
    NCCL_M2N_CHECK_ARG(!hasSrc || srcAddress - windowAddress <= maxWindowOffset, worldRank,
                       "ncclReshardWithWindow: srcBuffer offset exceeds intptr_t capacity "
                       "(srcBuffer=%p, window_user_ptr=%p). This is a collective API; all ranks must satisfy the "
                       "window-offset contract or peers may wait indefinitely",
                       srcBuffer, winUserPtr);
    NCCL_M2N_CHECK_ARG(!hasDst || dstAddress - windowAddress <= maxWindowOffset, worldRank,
                       "ncclReshardWithWindow: dstBuffer offset exceeds intptr_t capacity "
                       "(dstBuffer=%p, window_user_ptr=%p). This is a collective API; all ranks must satisfy the "
                       "window-offset contract or peers may wait indefinitely",
                       dstBuffer, winUserPtr);

    const intptr_t srcOff = hasSrc ? static_cast<intptr_t>(srcAddress - windowAddress) : 0;
    const intptr_t dstOff = hasDst ? static_cast<intptr_t>(dstAddress - windowAddress) : 0;
    NCCL_M2N_CHECK_ARG(!hasSrc || !hasDst || srcOff == dstOff, worldRank,
                       "ncclReshardWithWindow: srcBuffer and dstBuffer must share the same offset within the "
                       "registered window (single-offset contract); got srcOff=%lld, dstOff=%lld "
                       "(srcBuffer=%p, dstBuffer=%p, window_user_ptr=%p). This is a collective API; all ranks must "
                       "satisfy the window-offset contract or peers may wait indefinitely",
                       (long long)srcOff, (long long)dstOff, srcBuffer, dstBuffer, winUserPtr);

    // For fully-inactive ranks (both buffers null), 0 is a safe
    // placeholder — the kernel won't read params.myWindowOffset.
    localOffset = hasSrc ? srcOff : (hasDst ? dstOff : 0);
  }
#endif // NCCL_VERSION_CODE >= 2.29.2

  // Resolve algorithm.  AUTO falls through to RING.
  ReshardAlgorithm algo = reshardGetAlgorithm();
  if (algo == RESHARD_ALGO_AUTO) algo = RESHARD_ALGO_RING;

  // Derive the picker input from descriptor metadata on every rank so all
  // participants select the same collective geometry.
  auto localBytes = [&](const size_t* dims, const char* side, size_t* bytes) -> ncclResult_t {
    size_t total = elementSize;
    for (int d = 0; d < ndims; d++) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(total, dims[d], &total), worldRank,
                         "ncclReshardWithWindow: %s local byte size overflow at dim %d", side, d);
    }
    *bytes = total;
    return ncclSuccess;
  };
  size_t srcBytes = 0;
  size_t dstBytes = 0;
  NCCL_M2N_CHECK(localBytes(srcTensorDims, "source", &srcBytes));
  NCCL_M2N_CHECK(localBytes(dstTensorDims, "destination", &dstBytes));
  size_t bytesPerRank = (srcBytes > dstBytes) ? srcBytes : dstBytes;

  int numCtas = pickNumCtas(bytesPerRank, algo);
  size_t elementsPerChunk = pickElementsPerChunk(bytesPerRank, algo);

  int ginSignalCount = 0;
  NCCL_M2N_CHECK(computeReshardGinSignalCount(srcMesh, numCtas, worldRank, &ginSignalCount));

  size_t srcTotal = 0;
  NCCL_M2N_CHECK(computeReshardMeshSize(srcMesh, worldRank, &srcTotal));
  unsigned int mySignalBase = 0;
  int64_t srcRankOffset = static_cast<int64_t>(worldRank) - static_cast<int64_t>(srcMesh->startRank);
  if (srcRankOffset >= 0 && static_cast<size_t>(srcRankOffset) < srcTotal) {
    NCCL_M2N_CHECK(computeReshardSignalBase(srcMesh, worldRank, numCtas, worldRank, &mySignalBase));
  }

  // ------------------------------------------------------------------
  // Get-or-create the global devComm on this stream.  This both gives us
  // the LSA team size (used as gpusPerDomain below) and is what gets
  // passed to the kernel launch.
  // ------------------------------------------------------------------
  ncclDevComm* devCommPtr = findCachedDevComm(comm, numCtas, ginSignalCount, workStream);
  ncclDevComm localDevComm;
  if (devCommPtr == nullptr) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
    ncclDevCommRequirements reqs = {};
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
    if (algo == RESHARD_ALGO_DIRECT) reqs.worldGinBarrierCount = numCtas;
    else reqs.barrierCount = numCtas;
#else
    reqs.lsaBarrierCount = numCtas;
    reqs.railGinBarrierCount = numCtas;
#endif
    reqs.ginSignalCount = ginSignalCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    reqs.ginForceEnable = true;
#endif
    reqs.ginContextCount = DEFAULT_GIN_CONTEXT_COUNT;

    memset(&localDevComm, 0, sizeof(localDevComm));
{
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK(ncclDevCommCreate(comm, &reqs, &localDevComm));
    }
    NCCL_M2N_CHECK(cacheDevComm(comm, numCtas, ginSignalCount, &localDevComm, workStream));
    devCommPtr = findCachedDevComm(comm, numCtas, ginSignalCount, workStream);
    if (devCommPtr == nullptr) devCommPtr = &localDevComm;
  }

  // Domain-size resolution: override > LSA team size > gpusPerNode.
  const int lsaSizeFromComm = (devCommPtr->lsaSize > 0) ? devCommPtr->lsaSize : 0;
  const int gpusPerNode = reshardGetGpusPerNode();
  const int srcOverride = reshardGetSrcDomainSize();
  const int dstOverride = reshardGetDstDomainSize();

  int srcGpusPerDomain;
  if (srcOverride > 0) srcGpusPerDomain = srcOverride;
  else if (lsaSizeFromComm > 0) srcGpusPerDomain = lsaSizeFromComm;
  else srcGpusPerDomain = (gpusPerNode > 0) ? gpusPerNode : 1;

  int dstGpusPerDomain;
  if (dstOverride > 0) dstGpusPerDomain = dstOverride;
  else if (lsaSizeFromComm > 0) dstGpusPerDomain = lsaSizeFromComm;
  else dstGpusPerDomain = (gpusPerNode > 0) ? gpusPerNode : 1;

  RESHARD_INFO(worldRank,
               "algo=%s, lsa_size=%d, srcGpusPerDomain=%d, dstGpusPerDomain=%d, "
               "(srcOverride=%d, dstOverride=%d, gpusPerNode=%d), numCtas=%d, "
               "ginSignalCount=%d",
               algo == RESHARD_ALGO_RING ? "RING" : "DIRECT", lsaSizeFromComm, srcGpusPerDomain, dstGpusPerDomain,
               srcOverride, dstOverride, gpusPerNode, numCtas, ginSignalCount);

  // ------------------------------------------------------------------
  // Convert dims to bytes (matches ncclReshardWithWindow's contract for
  // prepareReshardParams / prepareDirectReshardParams).
  // ------------------------------------------------------------------
  size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "ncclReshardWithWindow:", ndims, elementSize, srcTensorDims,
                                    dstTensorDims, srcDimsBytes, dstDimsBytes));

  // ------------------------------------------------------------------
  // Cross-dim transpose optimisation (3D only).
  //
  // When src and dst shard different dimensions and the dst shard dim
  // is innermost, each GIN put is tiny.  Transposing the last two
  // tensor dims makes the large unshard dim innermost, boosting RDMA
  // throughput.  The kernel itself is unchanged — only the buffer
  // layout and mesh placements are rewritten around it.
  // ------------------------------------------------------------------
  int srcShardTensorDim = -1, dstShardTensorDim = -1;
  int srcShardCountForXpose = 1, dstShardCountForXpose = 1;
  for (int i = 0; i < NCCL_RESHARD_MESH_NDIMS; i++) {
    if (isShardPlacement(srcTensor->placements[i])) {
      srcShardTensorDim = getShardTensorDim(srcTensor->placements[i]);
      srcShardCountForXpose = srcMesh->dims[i];
    }
    if (isShardPlacement(dstTensor->placements[i])) {
      dstShardTensorDim = getShardTensorDim(dstTensor->placements[i]);
      dstShardCountForXpose = dstMesh->dims[i];
    }
  }

  int swapA = -1, swapB = -1;
  bool doTranspose = shouldTransposeForCrossDim(srcDimsBytes, dstDimsBytes, ndims, srcShardTensorDim, dstShardTensorDim,
                                                srcShardCountForXpose, dstShardCountForXpose, &swapA, &swapB);

  // Effective dims / tensors / buffers / window — may be overwritten by
  // the transpose path below, otherwise equal to the originals.
  size_t effSrcDims[NCCL_RESHARD_MAX_TENSOR_DIMS], effDstDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) {
    effSrcDims[d] = srcDimsBytes[d];
    effDstDims[d] = dstDimsBytes[d];
  }
  ncclMesh_t effSrcMesh = *srcMesh;
  ncclMesh_t effDstMesh = *dstMesh;
  ncclDistTensor_t effSrcTensor = *srcTensor;
  ncclDistTensor_t effDstTensor = *dstTensor;
  effSrcTensor.mesh = &effSrcMesh;
  effDstTensor.mesh = &effDstMesh;
  void* effSrcBuffer = srcBuffer;
  void* effDstBuffer = dstBuffer;
  ncclWindow_t effWindow = window;

  size_t dstMeshSize = 0;
  NCCL_M2N_CHECK(computeReshardMeshSize(dstMesh, worldRank, &dstMeshSize));
  int64_t dstRankOffset = static_cast<int64_t>(worldRank) - static_cast<int64_t>(dstMesh->startRank);
  bool isSource = srcRankOffset >= 0 && static_cast<size_t>(srcRankOffset) < srcTotal;
  bool isDest = dstRankOffset >= 0 && static_cast<size_t>(dstRankOffset) < dstMeshSize;

  if (doTranspose) {
    RESHARD_INFO(worldRank,
                 "Cross-dim transpose: swapping dims %d and %d "
                 "(srcShard=%d, dstShard=%d)",
                 swapA, swapB, srcShardTensorDim, dstShardTensorDim);

    // 1. Swap the last two dims in effective dims
    std::swap(effSrcDims[swapA], effSrcDims[swapB]);
    std::swap(effDstDims[swapA], effDstDims[swapB]);

    // 2. Rewrite mesh placements to match swapped layout
    for (int i = 0; i < NCCL_RESHARD_MESH_NDIMS; i++) {
      if (isShardPlacement(effSrcTensor.placements[i])) {
        int td = getShardTensorDim(effSrcTensor.placements[i]);
        if (td == swapA) effSrcTensor.placements[i] = NCCL_RESHARD_SHARD(swapB);
        else if (td == swapB) effSrcTensor.placements[i] = NCCL_RESHARD_SHARD(swapA);
      }
      if (isShardPlacement(effDstTensor.placements[i])) {
        int td = getShardTensorDim(effDstTensor.placements[i]);
        if (td == swapA) effDstTensor.placements[i] = NCCL_RESHARD_SHARD(swapB);
        else if (td == swapB) effDstTensor.placements[i] = NCCL_RESHARD_SHARD(swapA);
      }
    }

    NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, &effSrcTensor, effSrcDims, &effDstTensor, effDstDims,
                                             elementsPerChunk, algo, dstGpusPerDomain));

    // 3. Allocate / grow the transpose buffer.
    //    ncclCommWindowRegister is collective — all ranks must hit or
    //    miss the cache together.  Reconstruct global dims (uniform
    //    across all ranks) then derive both local sizes so every rank
    //    requests the same buffer size regardless of src/dst role.
    //    (Same pattern as prepareReshardParams globalDims logic.)
    size_t globalDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
    for (int d = 0; d < ndims; d++) {
      if (isSource && srcDimsBytes[d] > 0) {
        globalDims[d] = srcDimsBytes[d];
        if (d == srcShardTensorDim) {
          NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(globalDims[d], (size_t)srcShardCountForXpose, &globalDims[d]),
                             worldRank,
                             "ncclReshardWithWindow: transpose source global dim overflow at dim %d: local=%zu "
                             "shardCount=%d",
                             d, srcDimsBytes[d], srcShardCountForXpose);
        }
      } else if (isDest && dstDimsBytes[d] > 0) {
        globalDims[d] = dstDimsBytes[d];
        if (d == dstShardTensorDim) {
          NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(globalDims[d], (size_t)dstShardCountForXpose, &globalDims[d]),
                             worldRank,
                             "ncclReshardWithWindow: transpose destination global dim overflow at dim %d: local=%zu "
                             "shardCount=%d",
                             d, dstDimsBytes[d], dstShardCountForXpose);
        }
      } else {
        globalDims[d] = 1;
      }
    }
    size_t srcLocal = 1, dstLocal = 1;
    for (int d = 0; d < ndims; d++) {
      size_t srcDim = (d == srcShardTensorDim && srcShardCountForXpose > 1) ? globalDims[d] / srcShardCountForXpose
                                                                            : globalDims[d];
      size_t dstDim = (d == dstShardTensorDim && dstShardCountForXpose > 1) ? globalDims[d] / dstShardCountForXpose
                                                                            : globalDims[d];
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcLocal, srcDim, &srcLocal), worldRank,
                         "ncclReshardWithWindow: transpose source local size overflow at dim %d: current=%zu dim=%zu",
                         d, srcLocal, srcDim);
      NCCL_M2N_CHECK_ARG(
        m2nCheckedMulSize(dstLocal, dstDim, &dstLocal), worldRank,
        "ncclReshardWithWindow: transpose destination local size overflow at dim %d: current=%zu dim=%zu", d, dstLocal,
        dstDim);
    }
    size_t myLocalSize = std::max(srcLocal, dstLocal);
    NCCL_M2N_CHECK(ensureTransposeBuffer(comm, myLocalSize, workStream));

    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        NCCL_M2N_FAIL(ncclSystemError, worldRank, "CUDA error after ensureTransposeBuffer: %s",
                      cudaGetErrorString(err));
      }
      RESHARD_DEBUG(worldRank, "ensureTransposeBuffer: size=%zu, buf=%p", myLocalSize, getTransposeBuffer(comm));
    }

    // 4. PACK (source side): transpose user buffer -> transpose buffer
    if (isSource) {
      if (ndims == 2) {
        uwLaunchTranspose2DInner(srcBuffer, getTransposeBuffer(comm), 1, srcDimsBytes[0], srcDimsBytes[1], workStream);
      } else {
        uwLaunchTranspose2DInner(srcBuffer, getTransposeBuffer(comm), srcDimsBytes[0], srcDimsBytes[swapA],
                                 srcDimsBytes[swapB], workStream);
      }

      {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          NCCL_M2N_FAIL(ncclSystemError, worldRank, "CUDA error after transpose pack: %s",
                        cudaGetErrorString(err));
        }
        RESHARD_DEBUG(worldRank, "transpose pack: ndims=%d, D0=%zu, D1=%zu, D2=%zu", ndims, srcDimsBytes[0],
                      srcDimsBytes[swapA], srcDimsBytes[swapB]);
      }

      effSrcBuffer = getTransposeBuffer(comm);
    }

    // 5. DEST: kernel writes into transpose buffer; unpack afterwards
    if (isDest) effDstBuffer = getTransposeBuffer(comm);

    // 6. Register the transpose buffer as a window on comm (cached).
    //    This is collective — all ranks reach it because
    //    ncclReshardWithWindow is itself collective.
    ncclWindow_t* cached =
      findCachedInternalWindowByPtr(comm, getTransposeBuffer(comm), getTransposeBufferCapacity(comm));
    if (cached != nullptr) {
      effWindow = *cached;
    } else {
      ncclWindow_t xposeWin;
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclCommWindowRegister(comm, getTransposeBuffer(comm), getTransposeBufferCapacity(comm), &xposeWin,
                                            NCCL_WIN_COLL_SYMMETRIC));
      }
      NCCL_M2N_CHECK(cacheInternalWindow(comm, getTransposeBuffer(comm), getTransposeBufferCapacity(comm), xposeWin));
      effWindow = xposeWin;
    }

    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        NCCL_M2N_FAIL(ncclSystemError, worldRank, "CUDA error after window registration: %s",
                      cudaGetErrorString(err));
      }
      RESHARD_DEBUG(worldRank, "window register: buf=%p, cap=%zu, effWindow=%p", getTransposeBuffer(comm),
                    getTransposeBufferCapacity(comm), (void*)effWindow);
    }

    RESHARD_DEBUG(worldRank,
                  "Transpose: buf=%p, cap=%zu, effSrcDims=[%zu,%zu,%zu], "
                  "effDstDims=[%zu,%zu,%zu]",
                  getTransposeBuffer(comm), getTransposeBufferCapacity(comm), effSrcDims[0],
                  ndims >= 2 ? effSrcDims[1] : (size_t)0, ndims >= 3 ? effSrcDims[2] : (size_t)0, effDstDims[0],
      ndims >= 2 ? effDstDims[1] : (size_t)0, ndims >= 3 ? effDstDims[2] : (size_t)0);
  }

  if (!doTranspose) {
    NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, &effSrcTensor, effSrcDims, &effDstTensor, effDstDims,
                                             elementsPerChunk, algo, dstGpusPerDomain));
  }

  int threadsPerCta = DEFAULT_KERNEL_MAX_NTHREADS;

  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      NCCL_M2N_FAIL(ncclSystemError, worldRank, "CUDA error before reshard kernel launch: %s",
                    cudaGetErrorString(err));
    }
    RESHARD_DEBUG(worldRank,
                  "pre-launch: doTranspose=%d, algo=%s, numCtas=%d, threads=%d, "
                  "effWindow=%p, eff_src=%p, eff_dst=%p",
                  (int)doTranspose, algo == RESHARD_ALGO_RING ? "RING" : "DIRECT", numCtas, threadsPerCta,
                  (void*)effWindow, effSrcBuffer, effDstBuffer);
  }

  // ------------------------------------------------------------------
  // Build params + launch.  Under the single-offset symmetric contract,
  // every rank's per-peer windowOffset equals its own localOffset, so
  // we fill allWindowOffsets[] uniformly and the prep helpers index
  // into it the same way they do for the multi-window paths.
  //
  // doTranspose path: effWindow is the internal transpose-buffer
  // window whose base IS the transpose buffer, so the effective offset
  // is always 0 regardless of the user's localOffset.
  //
  // For the RING path, all LSA fan-out uses the global user window.
  // ------------------------------------------------------------------
  const size_t kernelOffset = doTranspose ? 0 : (size_t)localOffset;
  std::vector<size_t> allWindowOffsets(worldSize, kernelOffset);
  effSrcTensor.dataPtr = effSrcBuffer;
  effDstTensor.dataPtr = effDstBuffer;
  if (algo == RESHARD_ALGO_DIRECT) {
    ncclReshardDirectParams directParams{};
    NCCL_M2N_CHECK(prepareDirectReshardParams(worldRank, &effSrcTensor, effSrcDims, &effDstTensor, effDstDims,
                                              effWindow, elementsPerChunk, numCtas, mySignalBase, dstGpusPerDomain,
                                              allWindowOffsets.data(), &directParams));
    directParams.myWindowOffset = kernelOffset;

    directReshardKernelUserWindow<<<numCtas, threadsPerCta, 0, workStream>>>(directParams, *devCommPtr);
  } else {
    ncclReshardParams ringParams{};
    NCCL_M2N_CHECK(prepareReshardParams(worldRank, &effSrcTensor, effSrcDims, &effDstTensor, effDstDims, effWindow,
                                        elementsPerChunk, numCtas, mySignalBase, srcGpusPerDomain, dstGpusPerDomain,
                                        allWindowOffsets.data(), &ringParams));

    ringParams.myWindowOffset = kernelOffset;
    ringParams.ringNextWindowOffset = kernelOffset;
    for (int f = 0; f < ringParams.numLocalFollowers; f++) ringParams.localFollowerWindowOffsets[f] = kernelOffset;

    reshardKernelUserWindow<<<numCtas, threadsPerCta, 0, workStream>>>(ringParams, *devCommPtr);
  }

  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "reshard kernel launch failed: %s [algo=%s, numCtas=%d]",
                  cudaGetErrorString(launchErr), algo == RESHARD_ALGO_RING ? "RING" : "DIRECT", numCtas);
  }

  // ------------------------------------------------------------------
  // Transpose UNPACK (dest side): reverse-transpose from the transpose
  // buffer back into the user's dstBuffer.
  // ------------------------------------------------------------------
  if (doTranspose && isDest && dstBuffer != nullptr) {
    if (ndims == 2) {
      uwLaunchTranspose2DInner(getTransposeBuffer(comm), dstBuffer, 1, effDstDims[0], effDstDims[1], workStream);
    } else {
      uwLaunchTranspose2DInner(getTransposeBuffer(comm), dstBuffer, effDstDims[0], effDstDims[swapA], effDstDims[swapB],
                               workStream);
    }
  }

  if (doTranspose) {
    NCCL_M2N_CHECK(transposeBufferRecordEvent(comm, workStream));
  }

  return workCompletion.complete();
}
