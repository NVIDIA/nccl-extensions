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
#include <new>
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
#include "reshard_split.h"

#ifdef NCCL_M2N_TESTING
static thread_local size_t gFusedSubmissionCount = 0;

void reshardResetFusedSubmissionCountForTest() {
  gFusedSubmissionCount = 0;
}

size_t reshardGetFusedSubmissionCountForTest() {
  return gFusedSubmissionCount;
}
#endif

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

struct TransposeBufferEventGuard {
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;

  ~TransposeBufferEventGuard() {
    if (comm != nullptr) {
      NCCL_M2N_CHECK_WARN(transposeBufferRecordEvent(comm, stream));
    }
  }

  void arm(ncclComm_t guardedComm, cudaStream_t guardedStream) {
    comm = guardedComm;
    stream = guardedStream;
  }

  ncclResult_t record() {
    ncclResult_t result = transposeBufferRecordEvent(comm, stream);
    if (result == ncclSuccess) {
      comm = nullptr;
    }
    return result;
  }
};

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

          if (laneId == 0) {
            if (myStart < totalSize) {
              gin.put(world, target.dstWorldRank, params.window, target.windowOffset + plan.dstBaseOffset + myStart,
                      params.window, params.myWindowOffset + plan.srcBaseOffset + myStart, myEnd - myStart,
                      ncclGin_SignalInc{signalIdx});
            } else {
              gin.signal(world, target.dstWorldRank, ncclGin_SignalInc{signalIdx});
            }
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
          } else {
            gin.signal(world, params.ringNextWorldRank, ncclGin_SignalInc{signalIdx});
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
  if (m2nGroupIsActive()) {
    return m2nGroupEnqueueReshard(M2nGroupReshardKind::Window, handle, comm, window, src, dst, stream);
  }
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
  ReshardTensorSetup tensorSetup;
  NCCL_M2N_CHECK(reshardPrepareTensorSetup("ncclReshardWithWindow", src, dst, &tensorSetup));
  const int ndims = tensorSetup.ndims;
  const size_t elementSize = tensorSetup.elementSize;
  void* const srcBuffer = tensorSetup.srcTensor.dataPtr;
  void* const dstBuffer = tensorSetup.dstTensor.dataPtr;
  const size_t* const srcTensorDims = tensorSetup.srcTensor.localShape;
  const size_t* const dstTensorDims = tensorSetup.dstTensor.localShape;
  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));

  const ncclDistTensor_t* const srcTensor = &tensorSetup.srcTensor;
  const ncclDistTensor_t* const dstTensor = &tensorSetup.dstTensor;
  const ncclMesh_t* const srcMesh = &tensorSetup.srcMesh;
  const ncclMesh_t* const dstMesh = &tensorSetup.dstMesh;

  int worldRank, worldSize;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));
  reshardResolveAdaptiveScaleConfig(worldSize, /*splitCapable=*/false);
  NCCL_M2N_CHECK(validateReshardMeshBounds(srcMesh, dstMesh, worldSize, worldRank));
  NCCL_M2N_CHECK(reshardValidateActiveBuffers("ncclReshardWithWindow", worldRank, srcTensor, dstTensor));

  int currentCudaDev = 0;
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclSuccess;
  NCCL_M2N_CHECK(reshardMatchCommCudaDevice(comm, &currentCudaDev, &commProps, &propsResult));
  NCCL_M2N_CHECK(reshardRejectGraphCapture("ncclReshardWithWindow", stream));

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
  // Get-or-create the global devComm. This both gives us
  // the LSA team size (used as gpusPerDomain below) and is what gets
  // passed to the kernel launch.
  // ------------------------------------------------------------------
  const ReshardDevCommBarrierKind barrierKind =
    (algo == RESHARD_ALGO_DIRECT) ? RESHARD_DEVCOMM_BARRIER_WORLD : RESHARD_DEVCOMM_BARRIER_HYBRID;
  const ReshardDevCommCacheKey devCommKey = {
    comm, numCtas, ginSignalCount, 0, reshardGetGinContextCount(), barrierKind
  };
  ncclDevComm* devCommPtr = findCachedDevComm(devCommKey);
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
    reqs.ginContextCount = reshardGetGinContextCount();

    memset(&localDevComm, 0, sizeof(localDevComm));
{
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK(ncclDevCommCreate(comm, &reqs, &localDevComm));
    }
    NCCL_M2N_CHECK(cacheDevComm(devCommKey, &localDevComm));
    devCommPtr = findCachedDevComm(devCommKey);
    if (devCommPtr == nullptr) devCommPtr = &localDevComm;
  }

  const int lsaSizeFromComm = (devCommPtr->lsaSize > 0) ? devCommPtr->lsaSize : 0;
  const int gpusPerNode = reshardGetGpusPerNode();
  const int srcOverride = reshardGetSrcDomainSize();
  const int dstOverride = reshardGetDstDomainSize();

  int srcGpusPerDomain = 0;
  int dstGpusPerDomain = 0;
  NCCL_M2N_CHECK(resolveReshardDomainSizes(worldRank, algo, lsaSizeFromComm, lsaSizeFromComm, &srcGpusPerDomain,
                                           &dstGpusPerDomain));

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
  TransposeBufferEventGuard transposeEvent;

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
    transposeEvent.arm(comm, workStream);

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
    NCCL_M2N_CHECK(transposeEvent.record());
  }

  return workCompletion.complete();
}

// ============================================================================
// PACKWINDOW copy mode
//
// Pack each destination's bytes contiguously into the reused transpose/
// transfer buffer (CE cudaMemcpy3D), transfer with the hierarchical
// user-window kernel over contiguous plans, then unpack into the user dst
// buffer (CE cudaMemcpy3D).
//
// Two plans per (src,dst) pair:
//   * the original plan drives pack with source strides and unpack with
//     destination strides;
//   * a synthesized contiguous plan drives the transfer kernel.
//
// Receive offsets are keyed by source shard so every destination replica
// places a given source's bytes at the same staging offset.
// ============================================================================

static inline bool pwPlanPairBytes(const ncclReshardTransferPlan& plan, size_t* bytes) {
  return m2nCheckedMulSize(plan.totalInnerTransfers, plan.innerSize, bytes);
}

static cudaMemcpy3DParms pwBuildCopy(void* srcPtr, void* dstPtr, const ncclReshardTransferPlan& plan,
                                     const size_t* srcStrides, const size_t* dstStrides) {
  const size_t width = plan.innerSize;
  size_t height = 1;
  size_t depth = 1;
  if (plan.numOuterLoops == 1) {
    height = plan.outerCounts[0];
  } else if (plan.numOuterLoops >= 2) {
    height = plan.outerCounts[1];
    depth = plan.outerCounts[0];
  }

  size_t srcPitch = width;
  size_t srcSliceHeight = height;
  if (srcStrides != nullptr) {
    srcPitch = (plan.numOuterLoops >= 1) ? srcStrides[plan.numOuterLoops - 1] : width;
    srcSliceHeight = (depth > 1 && srcStrides[1] > 0) ? srcStrides[0] / srcStrides[1] : height;
  }
  size_t dstPitch = width;
  size_t dstSliceHeight = height;
  if (dstStrides != nullptr) {
    dstPitch = (plan.numOuterLoops >= 1) ? dstStrides[plan.numOuterLoops - 1] : width;
    dstSliceHeight = (depth > 1 && dstStrides[1] > 0) ? dstStrides[0] / dstStrides[1] : height;
  }

  cudaMemcpy3DParms copy = {};
  copy.kind = cudaMemcpyDeviceToDevice;
  copy.srcPtr = make_cudaPitchedPtr(srcPtr, srcPitch, width, srcSliceHeight);
  copy.dstPtr = make_cudaPitchedPtr(dstPtr, dstPitch, width, dstSliceHeight);
  copy.extent = make_cudaExtent(width, height, depth);
  return copy;
}

template <typename Params>
static ncclResult_t pwPackedRxOffset(const Params& params, int dstShard, int srcShardLimit, size_t rxBase,
                                     size_t* offset) {
  size_t current = rxBase;
  for (int srcShard = 0; srcShard < srcShardLimit; srcShard++) {
    ncclReshardTransferPlan plan;
    NCCL_M2N_CHECK(computeTransferPlanChecked(
      params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard, params.dstDims, params.dstStrides,
      params.dstShardTensorDim, dstShard, params.ndims, params.elementsPerChunk, &plan));
    size_t pairBytes = 0;
    NCCL_M2N_CHECK_ARG(pwPlanPairBytes(plan, &pairBytes) && m2nCheckedAddSize(current, pairBytes, &current), -1,
                       "reshardCopyPackWindow: packed receive offset overflow for dstShard=%d srcShard=%d",
                       dstShard, srcShard);
  }
  *offset = current;
  return ncclSuccess;
}

static ncclResult_t pwLsaHputWarmup(ncclComm_t comm, int worldRank, int worldSize, cudaStream_t stream) {
  if (worldSize < 2) return ncclSuccess;
  const int next = (worldRank + 1) % worldSize;
  const int previous = (worldRank - 1 + worldSize) % worldSize;
  M2nApiUnlock apiUnlock;
  NCCL_M2N_CHECK(ncclSignal(next, 0, 0, 0, comm, stream));
  ncclWaitSignalDesc_t wait = {1, previous, 0, 0};
  NCCL_M2N_CHECK(ncclWaitSignal(1, &wait, comm, stream));
  return ncclSuccess;
}

static ncclResult_t pwGetOrRegisterStagingWindow(ncclComm_t comm, void* staging, size_t capacity,
                                                 ncclWindow_t* outWindow) {
  NCCL_M2N_CHECK_ARG(outWindow != nullptr, -1, "PACKWINDOW staging-window output must be non-null");
  ncclWindow_t* cachedWindow = findCachedInternalWindowByPtr(comm, staging, capacity);
  if (cachedWindow != nullptr) {
    *outWindow = *cachedWindow;
    return ncclSuccess;
  }

  ncclWindow_t window = nullptr;
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclCommWindowRegister(comm, staging, capacity, &window, NCCL_WIN_COLL_SYMMETRIC));
  }
  NCCL_M2N_CHECK(cacheInternalWindow(comm, staging, capacity, window));
  *outWindow = window;
  return ncclSuccess;
}

static ncclResult_t reshardCopyPackWindowLsaHput(ncclComm_t comm, const ncclDistTensor_t* src,
                                                 const size_t srcDimsBytes[], const ncclDistTensor_t* dst,
                                                 const size_t dstDimsBytes[], void* staging, size_t stagingCapacity,
                                                 ncclWindow_t stagingWindow, size_t rxBase, size_t elementsPerChunk,
                                                 int worldRank, int worldSize, cudaStream_t workStream) {
  std::vector<size_t> allWindowOffsets(worldSize, 0);
  ncclReshardDirectParams params{};
  NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, src, srcDimsBytes, dst, dstDimsBytes, elementsPerChunk,
                                           RESHARD_ALGO_DIRECT, worldSize));
  NCCL_M2N_CHECK(prepareDirectReshardParams(worldRank, src, srcDimsBytes, dst, dstDimsBytes, stagingWindow,
                                            elementsPerChunk, 1, 0, worldSize, allWindowOffsets.data(), &params));
  params.myWindowOffset = 0;

  ncclReshardTransferPlan targetPlans[MAX_DIRECT_TARGETS];
  size_t txOffset[MAX_DIRECT_TARGETS] = {0};
  size_t txBytes[MAX_DIRECT_TARGETS] = {0};
  size_t peerRx[MAX_DIRECT_TARGETS] = {0};
  size_t shardTxOffset[MAX_DIRECT_TARGETS];
  bool packTarget[MAX_DIRECT_TARGETS] = {false};
  for (size_t& offset : shardTxOffset) offset = SIZE_MAX;

  if (params.isSource) {
    size_t cursor = 0;
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      const ncclReshardDirectTargetInfo& target = params.targets[targetIdx];
      NCCL_M2N_CHECK_ARG(target.dstShardIdx >= 0 && target.dstShardIdx < MAX_DIRECT_TARGETS, worldRank,
                         "PACKWINDOW LSA target shard index %d exceeds capacity %d", target.dstShardIdx,
                         MAX_DIRECT_TARGETS);
      targetPlans[targetIdx] = target.plan;
      NCCL_M2N_CHECK_ARG(pwPlanPairBytes(targetPlans[targetIdx], &txBytes[targetIdx]), worldRank,
                         "PACKWINDOW LSA target %d byte count overflow", targetIdx);
      if (shardTxOffset[target.dstShardIdx] == SIZE_MAX) {
        shardTxOffset[target.dstShardIdx] = cursor;
        packTarget[targetIdx] = true;
        size_t txEnd = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(cursor, txBytes[targetIdx], &txEnd) && txEnd <= rxBase, worldRank,
                           "PACKWINDOW LSA packed source range exceeds %zu bytes", rxBase);
        cursor = txEnd;
      }
      txOffset[targetIdx] = shardTxOffset[target.dstShardIdx];
      NCCL_M2N_CHECK(
        pwPackedRxOffset(params, target.dstShardIdx, params.mySrcShardIdx, rxBase, &peerRx[targetIdx]));
      size_t peerEnd = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(peerRx[targetIdx], txBytes[targetIdx], &peerEnd) &&
                           peerEnd <= stagingCapacity,
                         worldRank, "PACKWINDOW LSA target %d peer range exceeds %zu bytes", targetIdx,
                         stagingCapacity);
    }
  }

  bool rmaWarmed = false;
  int previousPeerCount = 0;
  int previousPeers[MAX_DIRECT_TARGETS] = {0};
  NCCL_M2N_CHECK(getTransposeBufferPackWindowState(comm, &rmaWarmed, &previousPeerCount, previousPeers));
  if (!rmaWarmed) {
    NCCL_M2N_CHECK(pwLsaHputWarmup(comm, worldRank, worldSize, workStream));
    rmaWarmed = true;
    NCCL_M2N_CHECK(setTransposeBufferPackWindowState(comm, rmaWarmed, previousPeerCount, previousPeers));
  }

  if (previousPeerCount > 0) {
    ncclWaitSignalDesc_t drains[MAX_DIRECT_TARGETS];
    for (int i = 0; i < previousPeerCount; i++) drains[i] = {1, previousPeers[i], 0, 0};
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclWaitSignal(previousPeerCount, drains, comm, workStream));
  }

  if (params.isSource && src->dataPtr != nullptr) {
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      if (!packTarget[targetIdx] || txBytes[targetIdx] == 0) continue;
      cudaMemcpy3DParms copy =
        pwBuildCopy((char*)src->dataPtr + targetPlans[targetIdx].srcBaseOffset,
                    (char*)staging + txOffset[targetIdx], targetPlans[targetIdx],
                    targetPlans[targetIdx].outerSrcStrides, nullptr);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&copy, workStream));
    }
  }

  int currentPeerCount = 0;
  int currentPeers[MAX_DIRECT_TARGETS] = {0};
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclGroupStart());
    ncclResult_t putResult = ncclSuccess;
    if (params.isSource) {
      for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
        if (txBytes[targetIdx] == 0) continue;
        putResult = ncclPutSignal((char*)staging + txOffset[targetIdx], txBytes[targetIdx], ncclUint8,
                                  params.targets[targetIdx].dstWorldRank, stagingWindow, peerRx[targetIdx], 0, 0, 0,
                                  comm, workStream);
        if (putResult != ncclSuccess) break;
        currentPeers[currentPeerCount++] = params.targets[targetIdx].dstWorldRank;
      }
    }
    const ncclResult_t groupResult = ncclGroupEnd();
    NCCL_M2N_CHECK(putResult);
    NCCL_M2N_CHECK(groupResult);
  }
  NCCL_M2N_CHECK(setTransposeBufferPackWindowState(comm, rmaWarmed, currentPeerCount, currentPeers));

  if (params.isDest && params.numSources > 0) {
    ncclWaitSignalDesc_t arrivals[MAX_DIRECT_SOURCES];
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      const int sourceRank = src->mesh->startRank + (int)params.sources[sourceIdx].signalBase;
      arrivals[sourceIdx] = {1, sourceRank, 0, 0};
    }
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclWaitSignal(params.numSources, arrivals, comm, workStream));
  }

  if (params.isDest && dst->dataPtr != nullptr) {
    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard,
                                                params.dstDims, params.dstStrides, params.dstShardTensorDim,
                                                params.myDstShardIdx, params.ndims, params.elementsPerChunk, &plan));
      size_t pairBytes = 0;
      NCCL_M2N_CHECK_ARG(pwPlanPairBytes(plan, &pairBytes), worldRank,
                         "PACKWINDOW LSA unpack shard %d byte count overflow", srcShard);
      if (pairBytes == 0) continue;
      size_t rxOffset = 0;
      size_t rxEnd = 0;
      NCCL_M2N_CHECK(pwPackedRxOffset(params, params.myDstShardIdx, srcShard, rxBase, &rxOffset));
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(rxOffset, pairBytes, &rxEnd) && rxEnd <= stagingCapacity, worldRank,
                         "PACKWINDOW LSA unpack shard %d range exceeds %zu bytes", srcShard, stagingCapacity);
      cudaMemcpy3DParms copy = pwBuildCopy((char*)staging + rxOffset, (char*)dst->dataPtr + plan.dstBaseOffset, plan,
                                           nullptr, plan.outerDstStrides);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&copy, workStream));
    }
  }

  if (params.isDest && params.numSources > 0) {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclGroupStart());
    ncclResult_t signalResult = ncclSuccess;
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      const int sourceRank = src->mesh->startRank + (int)params.sources[sourceIdx].signalBase;
      signalResult = ncclSignal(sourceRank, 0, 0, 0, comm, workStream);
      if (signalResult != ncclSuccess) break;
    }
    const ncclResult_t groupResult = ncclGroupEnd();
    NCCL_M2N_CHECK(signalResult);
    NCCL_M2N_CHECK(groupResult);
  }
  return ncclSuccess;
}

ncclResult_t reshardCopyPackWindowNormalized(ncclComm_t comm, const ncclDistTensor_t* src,
                                             const ncclDistTensor_t* dst, cudaStream_t workStream) {
  const int ndims = src->ndims;
  void* srcBuffer = src->dataPtr;
  void* dstBuffer = dst->dataPtr;
  const size_t elementSize = getNcclDtSize(src->dtype);
  const ncclMesh_t* srcMesh = src->mesh;
  const ncclMesh_t* dstMesh = dst->mesh;

  int worldRank = 0;
  int worldSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));

  size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "reshardCopyPackWindow:", ndims, elementSize, src->localShape,
                                    dst->localShape, srcDimsBytes, dstDimsBytes));

  const int srcMeshSize = srcMesh->dims[0] * srcMesh->dims[1];
  const bool isSource = worldRank >= srcMesh->startRank && worldRank < srcMesh->startRank + srcMeshSize;
  int srcShardTensorDim = -1;
  int dstShardTensorDim = -1;
  int srcShardCount = 1;
  int dstShardCount = 1;
  for (int i = 0; i < NCCL_RESHARD_MESH_NDIMS; i++) {
    if (isShardPlacement(src->placements[i])) {
      srcShardTensorDim = getShardTensorDim(src->placements[i]);
      srcShardCount = srcMesh->dims[i];
    }
    if (isShardPlacement(dst->placements[i])) {
      dstShardTensorDim = getShardTensorDim(dst->placements[i]);
      dstShardCount = dstMesh->dims[i];
    }
  }

  size_t globalDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) {
    globalDims[d] = srcDimsBytes[d];
    if (d == srcShardTensorDim) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(globalDims[d], (size_t)srcShardCount, &globalDims[d]), worldRank,
                         "reshardCopyPackWindow: source global size overflow at dim %d", d);
    }
  }

  size_t srcLocal = 1;
  size_t dstLocal = 1;
  for (int d = 0; d < ndims; d++) {
    size_t srcDim = (d == srcShardTensorDim && srcShardCount > 1) ? globalDims[d] / srcShardCount : globalDims[d];
    size_t dstDim = (d == dstShardTensorDim && dstShardCount > 1) ? globalDims[d] / dstShardCount : globalDims[d];
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcLocal, srcDim, &srcLocal), worldRank,
                       "reshardCopyPackWindow: source staging size overflow at dim %d", d);
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dstLocal, dstDim, &dstLocal), worldRank,
                       "reshardCopyPackWindow: destination staging size overflow at dim %d", d);
  }

  size_t regionBytes = std::max(srcLocal, dstLocal);
  if (regionBytes < 2048) regionBytes = 2048;
  size_t areaBytes = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(regionBytes, (size_t)2, &areaBytes), worldRank,
                     "reshardCopyPackWindow: staging region size overflow");
  const size_t rxBase = regionBytes;
  const int numCtas = pickNumCtas(areaBytes, RESHARD_ALGO_RING);
  const size_t elementsPerChunk = pickElementsPerChunk(areaBytes, RESHARD_ALGO_RING);
  int ginSignalCount = 0;
  NCCL_M2N_CHECK(computeReshardGinSignalCount(srcMesh, numCtas, worldRank, &ginSignalCount));
  unsigned int mySignalBase = 0;
  if (isSource) {
    NCCL_M2N_CHECK(computeReshardSignalBase(srcMesh, worldRank, numCtas, worldRank, &mySignalBase));
  }

  ncclCommProperties commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t queryResult = ncclSuccess;
  {
    M2nApiUnlock apiUnlock;
    queryResult = ncclCommQueryProperties(comm, &commProperties);
  }
  NCCL_M2N_CHECK(queryResult);
  int cudaDriverVersion = 0;
  NCCL_M2N_CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  int ncclVersion = 0;
  NCCL_M2N_CHECK(ncclGetVersion(&ncclVersion));
  const bool lsaCePlanFits =
    worldSize <= MAX_DIRECT_TARGETS && srcShardCount <= MAX_DIRECT_SOURCES && dstShardCount <= MAX_DIRECT_TARGETS;
  const bool hostRmaOrderingSupport = ncclVersion >= NCCL_VERSION(2, 30, 7);
  const bool forwardMeshOrder = srcMesh->startRank < dstMesh->startRank;
  const bool useLsaHput = commProperties.deviceApiSupport && commProperties.hostRmaSupport &&
                          commProperties.nLsaTeams == 1 && cudaDriverVersion >= 12050 && hostRmaOrderingSupport &&
                          forwardMeshOrder && lsaCePlanFits;

  NCCL_M2N_CHECK(ensureTransposeBuffer(comm, areaBytes, workStream));
  TransposeBufferEventGuard stagingEvent;
  stagingEvent.arm(comm, workStream);
  void* staging = getTransposeBuffer(comm);
  const size_t stagingCapacity = getTransposeBufferCapacity(comm);

  if (useLsaHput) {
    ncclWindow_t stagingWindow = nullptr;
    NCCL_M2N_CHECK(pwGetOrRegisterStagingWindow(comm, staging, stagingCapacity, &stagingWindow));
    RESHARD_INFO(worldRank, "packwindow-lsa-hput: ranks=%d srcShards=%d dstShards=%d areaBytes=%zu", worldSize,
                 srcShardCount, dstShardCount, areaBytes);
    NCCL_M2N_CHECK(reshardCopyPackWindowLsaHput(comm, src, srcDimsBytes, dst, dstDimsBytes, staging,
                                                stagingCapacity, stagingWindow, rxBase, elementsPerChunk, worldRank,
                                                worldSize, workStream));
    NCCL_M2N_CHECK(stagingEvent.record());
    return ncclSuccess;
  }

  if (commProperties.nLsaTeams == 1 && !hostRmaOrderingSupport) {
    RESHARD_INFO(worldRank,
                 "packwindow-lsa-hput: NCCL %d.%d.%d predates ordered host-RMA data/signal launch; using existing "
                 "kernel",
                 ncclVersion / 10000, (ncclVersion % 10000) / 100, ncclVersion % 100);
  } else if (commProperties.nLsaTeams == 1 && cudaDriverVersion < 12050) {
    RESHARD_INFO(worldRank,
                 "packwindow-lsa-hput: CUDA driver %d.%d is below the NCCL host-RMA minimum 12.5; using existing "
                 "kernel",
                 cudaDriverVersion / 1000, (cudaDriverVersion % 1000) / 10);
  } else if (commProperties.nLsaTeams == 1 && !forwardMeshOrder) {
    RESHARD_INFO(worldRank,
                 "packwindow-lsa-hput: reversed source/destination mesh ordering is not enabled; using existing "
                 "kernel");
  } else if (commProperties.nLsaTeams == 1 && !lsaCePlanFits) {
    RESHARD_INFO(worldRank,
                 "packwindow-lsa-hput: plan exceeds conservative limits (ranks=%d srcShards=%d dstShards=%d); "
                 "using existing kernel",
                 worldSize, srcShardCount, dstShardCount);
  }

  ReshardSplitComms splitComms;
  memset(&splitComms, 0, sizeof(splitComms));
  bool splitActive = false;
  if (reshardGetSplitCommEnabled() && reshardEffectiveLbMode(src, dst) == RESHARD_LB_NODE_AWARE) {
    int srcRepCount = 1;
    int dstRepCount = 1;
    for (int i = 0; i < NCCL_RESHARD_MESH_NDIMS; i++) {
      if (src->placements[i] == NCCL_RESHARD_REPLICATE) srcRepCount = srcMesh->dims[i];
      if (dst->placements[i] == NCCL_RESHARD_REPLICATE) dstRepCount = dstMesh->dims[i];
    }
    const bool dstRepStrided =
      dst->placements[1] == NCCL_RESHARD_REPLICATE && dstMesh->dims[0] > 1 && dstMesh->dims[1] > 1;
    NCCL_M2N_CHECK(reshardGetOrCreateSplitComms(comm, srcMesh, dstMesh, srcRepCount, dstRepCount, dstRepStrided,
                                                numCtas, workStream, &splitComms));
    splitActive = splitComms.active;
    const int configuredDstDomain = reshardGetDstDomainSize();
    if (splitActive && configuredDstDomain > 0 && configuredDstDomain != splitComms.lsaSize) {
      RESHARD_INFO(worldRank,
                   "split-comm: NCCL_RESHARD_DST_DOMAIN_SIZE=%d differs from probed commB lsaSize=%d -> fallback",
                   configuredDstDomain, splitComms.lsaSize);
      splitActive = false;
    }
    if (splitActive && reshardPlanHasCrossRailRingHop(src, dst, splitComms.lsaSize, dstMesh->startRank,
                                                      splitComms.strided, splitComms.numInjectionDomains,
                                                      splitComms.domainsPerRep)) {
      RESHARD_INFO(worldRank,
                   "split-comm: rail-incompatible ring hop for this layout (commB is RAIL-only) -> fallback to "
                   "non-split path");
      splitActive = false;
    }
  }

  ncclWindow_t stagingWindow = nullptr;
  if (!splitActive) {
    NCCL_M2N_CHECK(pwGetOrRegisterStagingWindow(comm, staging, stagingCapacity, &stagingWindow));
  }

  const ReshardDevCommCacheKey devCommKey = {
    comm, numCtas, ginSignalCount, 0, reshardGetGinContextCount(), RESHARD_DEVCOMM_BARRIER_HYBRID
  };
  ncclDevComm* devComm = nullptr;
  ncclDevComm localDevComm;
  int srcLsaSize = 0;
  int dstLsaSize = 0;
  if (splitActive) {
    srcLsaSize = splitComms.srcLsaSize;
    dstLsaSize = splitComms.lsaSize;
  } else {
    devComm = findCachedDevComm(devCommKey);
  }
  if (!splitActive && devComm == nullptr) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
    ncclDevCommRequirements requirements = {};
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
    requirements.barrierCount = numCtas;
#else
    requirements.lsaBarrierCount = numCtas;
    requirements.railGinBarrierCount = numCtas;
#endif
    requirements.ginSignalCount = ginSignalCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    requirements.ginForceEnable = true;
#endif
    requirements.ginContextCount = reshardGetGinContextCount();

    memset(&localDevComm, 0, sizeof(localDevComm));
    {
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK(ncclDevCommCreate(comm, &requirements, &localDevComm));
    }
    NCCL_M2N_CHECK(cacheDevComm(devCommKey, &localDevComm));
    devComm = findCachedDevComm(devCommKey);
    if (devComm == nullptr) devComm = &localDevComm;
  }

  if (!splitActive) {
    srcLsaSize = devComm->lsaSize > 0 ? devComm->lsaSize : 0;
    dstLsaSize = srcLsaSize;
  }
  int srcGpusPerDomain = 0;
  int dstGpusPerDomain = 0;
  NCCL_M2N_CHECK(resolveReshardDomainSizes(worldRank, RESHARD_ALGO_RING, srcLsaSize, dstLsaSize, &srcGpusPerDomain,
                                           &dstGpusPerDomain));

  std::vector<size_t> allWindowOffsets(worldSize, 0);
  const bool splitStrided = splitActive && splitComms.strided;
  const int splitNumInjectionDomains = splitActive ? splitComms.numInjectionDomains : -1;
  const int splitDomainsPerRep = splitActive ? splitComms.domainsPerRep : 1;
  ncclReshardParams params{};
  NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, src, srcDimsBytes, dst, dstDimsBytes, elementsPerChunk,
                                           RESHARD_ALGO_RING, dstGpusPerDomain));
  NCCL_M2N_CHECK(prepareReshardParams(worldRank, src, srcDimsBytes, dst, dstDimsBytes, stagingWindow,
                                      elementsPerChunk, numCtas, mySignalBase, srcGpusPerDomain, dstGpusPerDomain,
                                      allWindowOffsets.data(), &params, splitStrided,
                                      /*nodeAnchorAtMeshStart=*/splitActive, splitNumInjectionDomains,
                                      splitDomainsPerRep));
  params.window = stagingWindow;
  params.myWindowOffset = 0;
  params.ringNextWindowOffset = 0;
  for (int follower = 0; follower < params.numLocalFollowers; follower++) {
    params.localFollowerWindowOffsets[follower] = 0;
  }

  RESHARD_INFO(worldRank,
               "reshardCopyPackWindow: isSource=%d isDest=%d numTargets=%d numSources=%d numCtas=%d "
               "ginSignal=%d areaBytes=%zu staging=%p lsa=%d",
               (int)params.isSource, (int)params.isDest, params.numTargets, params.numSources, numCtas, ginSignalCount,
               areaBytes, staging, dstLsaSize);

  ncclReshardTransferPlan originalTargetPlans[MAX_TARGETS];
  size_t txOffset[MAX_TARGETS] = {0};
  size_t txBytes[MAX_TARGETS] = {0};
  size_t peerRx[MAX_TARGETS] = {0};
  size_t rxOffset[MAX_SOURCES] = {0};
  size_t rxBytes[MAX_SOURCES] = {0};

  if (params.isSource) {
    size_t cursor = 0;
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      originalTargetPlans[targetIdx] = params.targets[targetIdx].plan;
      NCCL_M2N_CHECK_ARG(pwPlanPairBytes(originalTargetPlans[targetIdx], &txBytes[targetIdx]), worldRank,
                         "reshardCopyPackWindow: target %d byte count overflow", targetIdx);
      txOffset[targetIdx] = cursor;
      NCCL_M2N_CHECK(pwPackedRxOffset(params, params.targets[targetIdx].dstShardIdx, params.mySrcShardIdx, rxBase,
                                      &peerRx[targetIdx]));
      size_t txEnd = 0;
      size_t peerEnd = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(cursor, txBytes[targetIdx], &txEnd) && txEnd <= stagingCapacity, worldRank,
                         "reshardCopyPackWindow: target %d staging range exceeds %zu bytes", targetIdx,
                         stagingCapacity);
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(peerRx[targetIdx], txBytes[targetIdx], &peerEnd) &&
                           peerEnd <= stagingCapacity,
                         worldRank, "reshardCopyPackWindow: target %d peer range exceeds %zu bytes", targetIdx,
                         stagingCapacity);
      cursor = txEnd;
    }
  }

  if (params.isDest) {
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      NCCL_M2N_CHECK_ARG(pwPlanPairBytes(params.sources[sourceIdx].plan, &rxBytes[sourceIdx]), worldRank,
                         "reshardCopyPackWindow: source %d byte count overflow", sourceIdx);
      NCCL_M2N_CHECK(pwPackedRxOffset(params, params.myDstShardIdx, params.sources[sourceIdx].srcShardIdx, rxBase,
                                      &rxOffset[sourceIdx]));
      size_t rxEnd = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(rxOffset[sourceIdx], rxBytes[sourceIdx], &rxEnd) &&
                           rxEnd <= stagingCapacity,
                         worldRank, "reshardCopyPackWindow: source %d staging range exceeds %zu bytes", sourceIdx,
                         stagingCapacity);
    }
  }

  if (params.isSource && srcBuffer != nullptr) {
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      if (txBytes[targetIdx] == 0) continue;
      cudaMemcpy3DParms copy =
        pwBuildCopy((char*)srcBuffer + originalTargetPlans[targetIdx].srcBaseOffset,
                    (char*)staging + txOffset[targetIdx], originalTargetPlans[targetIdx],
                    originalTargetPlans[targetIdx].outerSrcStrides, nullptr);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&copy, workStream));
    }
  }

  if (params.isSource) {
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      ncclReshardTargetInfo& target = params.targets[targetIdx];
      target.windowOffset = 0;
      target.isContiguous = true;
      target.totalBytes = txBytes[targetIdx];
      target.plan.srcBaseOffset = txOffset[targetIdx];
      target.plan.dstBaseOffset = peerRx[targetIdx];
      if (target.plan.totalInnerTransfers == 0) target.plan.totalInnerTransfers = 1;
    }
  }
  if (params.isDest) {
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      ncclReshardSourceInfo& source = params.sources[sourceIdx];
      source.isContiguous = true;
      source.totalBytes = rxBytes[sourceIdx];
      source.plan.dstBaseOffset = rxOffset[sourceIdx];
      if (source.plan.totalInnerTransfers == 0) source.plan.totalInnerTransfers = 1;
    }
  }

  const int threadsPerCta = DEFAULT_KERNEL_MAX_NTHREADS;
  if (splitActive) {
    NCCL_M2N_CHECK(reshardLaunchPackWindowSplit(&splitComms, staging, stagingCapacity, &params, numCtas, workStream));
  } else {
    reshardKernelUserWindow<<<numCtas, threadsPerCta, 0, workStream>>>(params, *devComm);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
      NCCL_M2N_FAIL(ncclSystemError, worldRank, "reshardCopyPackWindow kernel launch failed: %s [numCtas=%d]",
                    cudaGetErrorString(launchError), numCtas);
    }
  }

  if (params.isDest && dstBuffer != nullptr) {
    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard,
                                                params.dstDims, params.dstStrides, params.dstShardTensorDim,
                                                params.myDstShardIdx, params.ndims, params.elementsPerChunk, &plan));
      size_t pairBytes = 0;
      NCCL_M2N_CHECK_ARG(pwPlanPairBytes(plan, &pairBytes), worldRank,
                         "reshardCopyPackWindow: unpack shard %d byte count overflow", srcShard);
      if (pairBytes == 0) continue;
      size_t rx = 0;
      size_t rxEnd = 0;
      NCCL_M2N_CHECK(pwPackedRxOffset(params, params.myDstShardIdx, srcShard, rxBase, &rx));
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(rx, pairBytes, &rxEnd) && rxEnd <= stagingCapacity, worldRank,
                         "reshardCopyPackWindow: unpack shard %d staging range exceeds %zu bytes", srcShard,
                         stagingCapacity);
      cudaMemcpy3DParms copy = pwBuildCopy((char*)staging + rx, (char*)dstBuffer + plan.dstBaseOffset, plan, nullptr,
                                           plan.outerDstStrides);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&copy, workStream));
    }
  }

  NCCL_M2N_CHECK(stagingEvent.record());
  return ncclSuccess;
}
// Cross-tensor PACKWINDOW group fusion.
//
// All entries share one normalized mesh/placement signature.  Each entry is
// prepared independently, then its per-(src shard, dst shard) fragment is
// packed into a source-major batch layout:
//
//   [src shard 0: tensor 0..N][src shard 1: tensor 0..N]...
//
// A source therefore sends one contiguous payload to each destination and the
// existing PACKWINDOW kernel performs one barrier/signal epoch for the group.
// ============================================================================

static bool pwGroupSameTopology(const ReshardTensorSetup& first, const ReshardTensorSetup& entry) {
  return m2nSameTensorTopology(first.srcTensor, entry.srcTensor) &&
         m2nSameTensorTopology(first.dstTensor, entry.dstTensor);
}

static ncclResult_t pwGroupTensorBytes(int worldRank, const ReshardTensorSetup& setup, size_t* srcBytes,
                                       size_t* dstBytes) {
  // Match the single-entry PACKWINDOW path: reconstruct the global shape,
  // then derive both canonical local regions from mesh shard counts.
  size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "ncclM2nGroupEnd:", setup.ndims, setup.elementSize,
                                    setup.srcTensor.localShape, setup.dstTensor.localShape, srcDimsBytes,
                                    dstDimsBytes));

  int srcShardDim = -1;
  int dstShardDim = -1;
  int srcShardCount = 1;
  int dstShardCount = 1;
  for (int i = 0; i < NCCL_RESHARD_MESH_NDIMS; i++) {
    if (isShardPlacement(setup.srcTensor.placements[i])) {
      srcShardDim = getShardTensorDim(setup.srcTensor.placements[i]);
      srcShardCount = setup.srcMesh.dims[i];
    }
    if (isShardPlacement(setup.dstTensor.placements[i])) {
      dstShardDim = getShardTensorDim(setup.dstTensor.placements[i]);
      dstShardCount = setup.dstMesh.dims[i];
    }
  }

  *srcBytes = 1;
  *dstBytes = 1;
  for (int d = 0; d < setup.ndims; d++) {
    size_t globalDim = srcDimsBytes[d];
    if (d == srcShardDim) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(globalDim, (size_t)srcShardCount, &globalDim), worldRank,
                         "ncclM2nGroupEnd: source global size overflow at dim %d", d);
    }
    size_t dstGlobalDim = dstDimsBytes[d];
    if (d == dstShardDim) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dstGlobalDim, (size_t)dstShardCount, &dstGlobalDim), worldRank,
                         "ncclM2nGroupEnd: destination global size overflow at dim %d", d);
    }
    NCCL_M2N_CHECK_ARG(globalDim == dstGlobalDim, worldRank,
                       "ncclM2nGroupEnd: source and destination global shapes differ at dim %d", d);
    const size_t srcDim = (d == srcShardDim && srcShardCount > 1) ? globalDim / srcShardCount : globalDim;
    const size_t dstDim = (d == dstShardDim && dstShardCount > 1) ? globalDim / dstShardCount : globalDim;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(*srcBytes, srcDim, srcBytes), worldRank,
                       "ncclM2nGroupEnd: source local byte size overflow at dim %d", d);
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(*dstBytes, dstDim, dstBytes), worldRank,
                       "ncclM2nGroupEnd: destination local byte size overflow at dim %d", d);
  }
  return ncclSuccess;
}

struct PwGroupBin {
  std::vector<size_t> entries;
  size_t srcBytes = 0;
  size_t dstBytes = 0;
  size_t areaBytes = 0;
};

static size_t groupOriginalIndex(const size_t* originalIndices, size_t entry) {
  return originalIndices == nullptr ? entry : originalIndices[entry];
}

static bool pwGroupAreaBytes(size_t srcBytes, size_t dstBytes, size_t* areaBytes) {
  size_t regionBytes = std::max(srcBytes, dstBytes);
  if (regionBytes < 2048) {
    regionBytes = 2048;
  }
  return m2nCheckedMulSize(regionBytes, (size_t)2, areaBytes);
}

static ncclResult_t pwGroupPairOffset(const size_t* pairBytes, size_t count, int srcShardCount, int dstShardCount,
                                      int srcShard, int dstShard, size_t entry, size_t rxBase, size_t* offset) {
  size_t result = rxBase;
  for (int s = 0; s < srcShard; s++) {
    for (size_t e = 0; e < count; e++) {
      const size_t index = (e * (size_t)srcShardCount + (size_t)s) * (size_t)dstShardCount + (size_t)dstShard;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(result, pairBytes[index], &result), -1,
                         "ncclM2nGroupEnd: packed receive offset overflow");
    }
  }
  for (size_t e = 0; e < entry; e++) {
    const size_t index = (e * (size_t)srcShardCount + (size_t)srcShard) * (size_t)dstShardCount + (size_t)dstShard;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(result, pairBytes[index], &result), -1,
                       "ncclM2nGroupEnd: packed receive offset overflow");
  }
  *offset = result;
  return ncclSuccess;
}

static ncclResult_t reshardCopyPackWindowGroupNormalized(ncclComm_t comm, const ReshardTensorSetup* setups,
                                                         const size_t* entries, const size_t* originalIndices,
                                                         size_t count, cudaStream_t workStream,
                                                         size_t* failedOriginalIndex) {
  *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[0]);
  int worldRank = 0;
  int worldSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));

  size_t srcBatchBytes = 0;
  size_t dstBatchBytes = 0;
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
    const ReshardTensorSetup& setup = setups[entries[e]];
    size_t srcBytes = 0;
    size_t dstBytes = 0;
    NCCL_M2N_CHECK(pwGroupTensorBytes(worldRank, setup, &srcBytes, &dstBytes));
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(srcBatchBytes, srcBytes, &srcBatchBytes) &&
                         m2nCheckedAddSize(dstBatchBytes, dstBytes, &dstBatchBytes),
                       worldRank, "ncclM2nGroupEnd: aggregate local byte size overflow at entry %zu", entries[e]);
  }

  size_t regionBytes = std::max(srcBatchBytes, dstBatchBytes);
  if (regionBytes < 2048) {
    regionBytes = 2048;
  }
  size_t areaBytes = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(regionBytes, (size_t)2, &areaBytes), worldRank,
                     "ncclM2nGroupEnd: staging region size overflow");
  const size_t rxBase = regionBytes;
  const int numCtas = pickNumCtas(areaBytes, RESHARD_ALGO_RING);
  const size_t elementsPerChunk = pickElementsPerChunk(areaBytes, RESHARD_ALGO_RING);

  const ncclMesh_t* srcMesh = &setups[entries[0]].srcMesh;
  const bool isSource = reshardRankInMesh(srcMesh, worldRank);
  int ginSignalCount = 0;
  NCCL_M2N_CHECK(computeReshardGinSignalCount(srcMesh, numCtas, worldRank, &ginSignalCount));
  unsigned int mySignalBase = 0;
  if (isSource) {
    NCCL_M2N_CHECK(computeReshardSignalBase(srcMesh, worldRank, numCtas, worldRank, &mySignalBase));
  }

  NCCL_M2N_CHECK(ensureTransposeBuffer(comm, areaBytes, workStream));
  TransposeBufferEventGuard stagingEvent;
  stagingEvent.arm(comm, workStream);
  void* staging = getTransposeBuffer(comm);
  const size_t stagingCapacity = getTransposeBufferCapacity(comm);

  ncclWindow_t stagingWindow = nullptr;
  ncclWindow_t* cachedWindow = findCachedInternalWindowByPtr(comm, staging, stagingCapacity);
  if (cachedWindow != nullptr) {
    stagingWindow = *cachedWindow;
  } else {
    {
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK(
        ncclCommWindowRegister(comm, staging, stagingCapacity, &stagingWindow, NCCL_WIN_COLL_SYMMETRIC));
    }
    NCCL_M2N_CHECK(cacheInternalWindow(comm, staging, stagingCapacity, stagingWindow));
  }

  ncclDevComm activeDevComm;
  NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, numCtas, ginSignalCount, 0, RESHARD_DEVCOMM_BARRIER_HYBRID,
                                           reshardGetGinContextCount(), workStream, &activeDevComm));
  const int lsaSize = activeDevComm.lsaSize > 0 ? activeDevComm.lsaSize : 0;
  int srcGpusPerDomain = 0;
  int dstGpusPerDomain = 0;
  NCCL_M2N_CHECK(resolveReshardDomainSizes(worldRank, RESHARD_ALGO_RING, lsaSize, lsaSize, &srcGpusPerDomain,
                                           &dstGpusPerDomain));

  size_t planSlots = 0;
  size_t pairSlots = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(count, (size_t)MAX_TARGETS, &planSlots), worldRank,
                     "ncclM2nGroupEnd: target plan count overflow");

  std::unique_ptr<ncclReshardParams> aggregate(new (std::nothrow) ncclReshardParams);
  std::unique_ptr<ncclReshardParams> entryParams(new (std::nothrow) ncclReshardParams);
  std::unique_ptr<ncclReshardTransferPlan[]> targetPlans(
    new (std::nothrow) ncclReshardTransferPlan[planSlots]);
  std::unique_ptr<size_t[]> targetBytes(new (std::nothrow) size_t[planSlots]);
  if (aggregate == nullptr || entryParams == nullptr || targetPlans == nullptr || targetBytes == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate host planning storage");
  }

  std::unique_ptr<size_t[]> allWindowOffsets(new (std::nothrow) size_t[(size_t)worldSize]());
  if (allWindowOffsets == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate window-offset storage");
  }
  std::unique_ptr<ncclReshardTransferPlan[]> unpackPlans;
  std::unique_ptr<size_t[]> pairBytes;
  int srcShardCount = 0;
  int dstShardCount = 0;

  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
    const ReshardTensorSetup& setup = setups[entries[e]];
    size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
    size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
    NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "ncclM2nGroupEnd:", setup.ndims, setup.elementSize,
                                      setup.srcTensor.localShape, setup.dstTensor.localShape, srcDimsBytes,
                                      dstDimsBytes));
    NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, &setup.srcTensor, srcDimsBytes, &setup.dstTensor,
                                             dstDimsBytes, elementsPerChunk, RESHARD_ALGO_RING, dstGpusPerDomain));
    memset(entryParams.get(), 0, sizeof(*entryParams));
    NCCL_M2N_CHECK(prepareReshardParams(worldRank, &setup.srcTensor, srcDimsBytes, &setup.dstTensor, dstDimsBytes,
                                        stagingWindow, elementsPerChunk, numCtas, mySignalBase, srcGpusPerDomain,
                                        dstGpusPerDomain, allWindowOffsets.get(), entryParams.get()));

    if (e == 0) {
      *aggregate = *entryParams;
      srcShardCount = aggregate->srcShardCount;
      dstShardCount = aggregate->dstShardCount;
      size_t unpackSlots = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(count, (size_t)srcShardCount, &unpackSlots) &&
                           m2nCheckedMulSize(unpackSlots, (size_t)dstShardCount, &pairSlots),
                         worldRank, "ncclM2nGroupEnd: source plan count overflow");
      unpackPlans.reset(new (std::nothrow) ncclReshardTransferPlan[unpackSlots]);
      pairBytes.reset(new (std::nothrow) size_t[pairSlots]);
      if (unpackPlans == nullptr || pairBytes == nullptr) {
        NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate host planning storage");
      }
    } else {
      NCCL_M2N_CHECK_ARG(entryParams->numTargets == aggregate->numTargets &&
                         entryParams->numSources == aggregate->numSources &&
                           entryParams->srcShardCount == srcShardCount && entryParams->dstShardCount == dstShardCount,
                         worldRank, "ncclM2nGroupEnd: entry %zu produced a different peer topology", entries[e]);
      for (int t = 0; t < aggregate->numTargets; t++) {
        NCCL_M2N_CHECK_ARG(entryParams->targets[t].dstWorldRank == aggregate->targets[t].dstWorldRank &&
                             entryParams->targets[t].dstShardIdx == aggregate->targets[t].dstShardIdx,
                           worldRank, "ncclM2nGroupEnd: entry %zu produced a different target order", entries[e]);
      }
      for (int s = 0; s < aggregate->numSources; s++) {
        NCCL_M2N_CHECK_ARG(entryParams->sources[s].srcShardIdx == aggregate->sources[s].srcShardIdx, worldRank,
                           "ncclM2nGroupEnd: entry %zu produced a different source order", entries[e]);
      }
    }

    for (int t = 0; t < aggregate->numTargets; t++) {
      const size_t index = e * (size_t)MAX_TARGETS + (size_t)t;
      targetPlans[index] = entryParams->targets[t].plan;
      NCCL_M2N_CHECK_ARG(pwPlanPairBytes(targetPlans[index], &targetBytes[index]), worldRank,
                         "ncclM2nGroupEnd: target byte count overflow at entry %zu", entries[e]);
    }

    for (int s = 0; s < srcShardCount; s++) {
      ncclReshardTransferPlan& plan = unpackPlans[e * (size_t)srcShardCount + (size_t)s];
      memset(&plan, 0, sizeof(plan));
      if (entryParams->isDest) {
        NCCL_M2N_CHECK(computeTransferPlanChecked(entryParams->srcDims, entryParams->srcStrides,
                                                  entryParams->srcShardTensorDim, s, entryParams->dstDims,
                                                  entryParams->dstStrides, entryParams->dstShardTensorDim,
                                                  entryParams->myDstShardIdx, entryParams->ndims,
                                                  entryParams->elementsPerChunk, &plan));
      }
      for (int d = 0; d < dstShardCount; d++) {
        ncclReshardTransferPlan pairPlan;
        NCCL_M2N_CHECK(computeTransferPlanChecked(entryParams->srcDims, entryParams->srcStrides,
                                                  entryParams->srcShardTensorDim, s, entryParams->dstDims,
                                                  entryParams->dstStrides, entryParams->dstShardTensorDim, d,
                                                  entryParams->ndims, entryParams->elementsPerChunk, &pairPlan));
        const size_t index = (e * (size_t)srcShardCount + (size_t)s) * (size_t)dstShardCount + (size_t)d;
        NCCL_M2N_CHECK_ARG(pwPlanPairBytes(pairPlan, &pairBytes[index]), worldRank,
                           "ncclM2nGroupEnd: pair byte count overflow at entry %zu", entries[e]);
      }
    }
  }

  aggregate->window = stagingWindow;
  aggregate->myWindowOffset = 0;
  aggregate->ringNextWindowOffset = 0;
  for (int f = 0; f < aggregate->numLocalFollowers; f++) {
    aggregate->localFollowerWindowOffsets[f] = 0;
  }

  size_t txCursor = 0;
  for (int t = 0; t < aggregate->numTargets; t++) {
    size_t total = 0;
    for (size_t e = 0; e < count; e++) {
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(total, targetBytes[e * (size_t)MAX_TARGETS + (size_t)t], &total),
                         worldRank, "ncclM2nGroupEnd: target payload overflow");
    }
    size_t txEnd = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(txCursor, total, &txEnd) && txEnd <= stagingCapacity, worldRank,
                       "ncclM2nGroupEnd: source staging range exceeds %zu bytes", stagingCapacity);
    size_t peerOffset = 0;
    NCCL_M2N_CHECK(pwGroupPairOffset(pairBytes.get(), count, srcShardCount, dstShardCount,
                                     aggregate->mySrcShardIdx, aggregate->targets[t].dstShardIdx, 0, rxBase,
                                     &peerOffset));
    size_t peerEnd = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(peerOffset, total, &peerEnd) && peerEnd <= stagingCapacity, worldRank,
                       "ncclM2nGroupEnd: destination staging range exceeds %zu bytes", stagingCapacity);

    size_t entryCursor = txCursor;
    if (aggregate->isSource) {
      for (size_t e = 0; e < count; e++) {
        *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
        const size_t index = e * (size_t)MAX_TARGETS + (size_t)t;
        if (targetBytes[index] != 0) {
          const ncclReshardTransferPlan& plan = targetPlans[index];
          cudaMemcpy3DParms cp = pwBuildCopy((char*)setups[entries[e]].srcTensor.dataPtr + plan.srcBaseOffset,
                                             (char*)staging + entryCursor, plan, plan.outerSrcStrides, nullptr);
          NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
        }
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(entryCursor, targetBytes[index], &entryCursor), worldRank,
                           "ncclM2nGroupEnd: source packing offset overflow");
      }
    }

    ncclReshardTargetInfo& target = aggregate->targets[t];
    target.windowOffset = 0;
    target.isContiguous = true;
    target.totalBytes = total;
    target.plan.srcBaseOffset = txCursor;
    target.plan.dstBaseOffset = peerOffset;
    target.plan.totalInnerTransfers = 1;
    txCursor = txEnd;
  }

  if (aggregate->isDest) {
    for (int i = 0; i < aggregate->numSources; i++) {
      ncclReshardSourceInfo& source = aggregate->sources[i];
      size_t total = 0;
      for (size_t e = 0; e < count; e++) {
        const size_t index = (e * (size_t)srcShardCount + (size_t)source.srcShardIdx) *
                               (size_t)dstShardCount +
                             (size_t)aggregate->myDstShardIdx;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(total, pairBytes[index], &total), worldRank,
                           "ncclM2nGroupEnd: source payload overflow");
      }
      size_t offset = 0;
      NCCL_M2N_CHECK(pwGroupPairOffset(pairBytes.get(), count, srcShardCount, dstShardCount, source.srcShardIdx,
                                       aggregate->myDstShardIdx, 0, rxBase, &offset));
      source.isContiguous = true;
      source.totalBytes = total;
      source.plan.dstBaseOffset = offset;
      source.plan.totalInnerTransfers = 1;
    }
  }

  RESHARD_INFO(worldRank,
               "ncclM2nGroupEnd: count=%zu numTargets=%d numSources=%d numCtas=%d areaBytes=%zu staging=%p",
               count, aggregate->numTargets, aggregate->numSources, numCtas, areaBytes, staging);

  reshardKernelUserWindow<<<numCtas, DEFAULT_KERNEL_MAX_NTHREADS, 0, workStream>>>(*aggregate, activeDevComm);
  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd kernel launch failed: %s [numCtas=%d]",
                  cudaGetErrorString(launchError), numCtas);
  }
  if (aggregate->isDest) {
    for (int s = 0; s < srcShardCount; s++) {
      for (size_t e = 0; e < count; e++) {
        *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
        const size_t pairIndex = (e * (size_t)srcShardCount + (size_t)s) * (size_t)dstShardCount +
                                 (size_t)aggregate->myDstShardIdx;
        if (pairBytes[pairIndex] == 0) {
          continue;
        }
        const ncclReshardTransferPlan& plan = unpackPlans[e * (size_t)srcShardCount + (size_t)s];
        size_t offset = 0;
        NCCL_M2N_CHECK(pwGroupPairOffset(pairBytes.get(), count, srcShardCount, dstShardCount, s,
                                         aggregate->myDstShardIdx, e, rxBase, &offset));
        size_t end = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(offset, pairBytes[pairIndex], &end) && end <= stagingCapacity, worldRank,
                           "ncclM2nGroupEnd: unpack range exceeds %zu bytes", stagingCapacity);
        cudaMemcpy3DParms cp =
          pwBuildCopy((char*)staging + offset,
                      (char*)setups[entries[e]].dstTensor.dataPtr + plan.dstBaseOffset, plan, nullptr,
                      plan.outerDstStrides);
        NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
      }
    }
  }

  NCCL_M2N_CHECK(stagingEvent.record());
  return ncclSuccess;
}

ncclResult_t reshardTryExecuteStagingGroup(ncclM2nHandle_t handle, ncclComm_t comm,
                                           const ncclDistTensor_t* srcs, const ncclDistTensor_t* dsts,
                                           const size_t* originalIndices, size_t count, cudaStream_t stream,
                                           bool* handled, size_t* failedOriginalIndex) {
  *handled = false;
  if (count < 2 || count > kM2nGroupMaxFusionEntries ||
      reshardGetCopyAlgorithm() != RESHARD_COPY_ALGO_PACKWINDOW) {
    return ncclSuccess;
  }
  *failedOriginalIndex = groupOriginalIndex(originalIndices, 0);

  M2nApiLock apiLock;
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(comm != nullptr, -1, "ncclM2nGroupEnd: comm must be non-null");
  NCCL_M2N_CHECK_ARG(srcs != nullptr && dsts != nullptr, -1,
                     "ncclM2nGroupEnd: srcs and dsts must both be non-null");

  std::unique_ptr<ReshardTensorSetup[]> setups(new (std::nothrow) ReshardTensorSetup[count]);
  if (setups == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nGroupEnd: failed to allocate descriptor storage");
  }
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
    NCCL_M2N_CHECK(reshardPrepareTensorSetup("ncclM2nGroupEnd", &srcs[e], &dsts[e], &setups[e]));
  }

  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));
  int parentCommSize = 0;
  NCCL_M2N_CHECK(ncclCommCount(comm, &parentCommSize));
  reshardResolveAdaptiveScaleConfig(parentCommSize, /*splitCapable=*/false);
  if (reshardGetSplitCommEnabled()) {
    return ncclSuccess;
  }

  int worldRank = 0;
  int worldSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
    NCCL_M2N_CHECK(validateReshardMeshBounds(&setups[e].srcMesh, &setups[e].dstMesh, worldSize, worldRank));
    NCCL_M2N_CHECK(
      reshardValidateActiveBuffers("ncclM2nGroupEnd", worldRank, &setups[e].srcTensor, &setups[e].dstTensor));
  }

  int currentCudaDev = 0;
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclSuccess;
  NCCL_M2N_CHECK(reshardMatchCommCudaDevice(comm, &currentCudaDev, &commProps, &propsResult));
  ncclResult_t captureResult = reshardRejectGraphCapture("ncclReshard", stream);
  if (captureResult == ncclInvalidUsage) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK(captureResult);

  std::unique_ptr<size_t[]> srcEntryBytes(new (std::nothrow) size_t[count]);
  std::unique_ptr<size_t[]> dstEntryBytes(new (std::nothrow) size_t[count]);
  if (srcEntryBytes == nullptr || dstEntryBytes == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate bin sizing storage");
  }
  size_t largestSingleArea = 0;
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
    NCCL_M2N_CHECK(pwGroupTensorBytes(worldRank, setups[e], &srcEntryBytes[e], &dstEntryBytes[e]));
    size_t areaBytes = 0;
    NCCL_M2N_CHECK_ARG(pwGroupAreaBytes(srcEntryBytes[e], dstEntryBytes[e], &areaBytes), worldRank,
                       "ncclM2nGroupEnd: entry %zu staging size overflow", e);
    largestSingleArea = std::max(largestSingleArea, areaBytes);
  }

  struct GroupBufferRange {
    uintptr_t begin;
    uintptr_t end;
    size_t entry;
  };
  std::vector<GroupBufferRange> ranges;
  try {
    ranges.reserve(count * 2);
    for (size_t e = 0; e < count; e++) {
      *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
      if (setups[e].srcTensor.dataPtr != nullptr && srcEntryBytes[e] > 0) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(setups[e].srcTensor.dataPtr);
        size_t end = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(begin, srcEntryBytes[e], &end), worldRank,
                           "ncclM2nGroupEnd: source range overflow at entry %zu",
                           groupOriginalIndex(originalIndices, e));
        ranges.push_back({begin, end, groupOriginalIndex(originalIndices, e)});
      }
      if (setups[e].dstTensor.dataPtr != nullptr && dstEntryBytes[e] > 0) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(setups[e].dstTensor.dataPtr);
        size_t end = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(begin, dstEntryBytes[e], &end), worldRank,
                           "ncclM2nGroupEnd: destination range overflow at entry %zu",
                           groupOriginalIndex(originalIndices, e));
        ranges.push_back({begin, end, groupOriginalIndex(originalIndices, e)});
      }
    }
    std::sort(ranges.begin(), ranges.end(), [](const GroupBufferRange& a, const GroupBufferRange& b) {
      return a.begin < b.begin;
    });
    for (size_t i = 0; i < ranges.size(); i++) {
      for (size_t j = i + 1; j < ranges.size() && ranges[j].begin < ranges[i].end; j++) {
        const size_t firstEntry = std::min(ranges[i].entry, ranges[j].entry);
        const size_t secondEntry = std::max(ranges[i].entry, ranges[j].entry);
        *failedOriginalIndex = secondEntry;
        NCCL_M2N_CHECK_ARG(ranges[i].entry == ranges[j].entry, worldRank,
                           "ncclM2nGroupEnd: grouped entries %zu and %zu have overlapping buffers",
                           firstEntry, secondEntry);
      }
    }
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate overlap validation storage");
  }

  size_t stagingBudget = getTransposeBufferCapacity(comm);
  if (reshardStagingBucketsEnabled()) {
    stagingBudget = gReshardStagingBuckets[gReshardStagingBucketCount - 1].size;
    for (int i = 0; i < gReshardStagingBucketCount; i++) {
      if (gReshardStagingBuckets[i].size >= largestSingleArea) {
        stagingBudget = gReshardStagingBuckets[i].size;
        break;
      }
    }
  } else if (stagingBudget == 0) {
    stagingBudget = std::max(gReshardStagingWatermarkBytes, largestSingleArea);
  }

  std::vector<PwGroupBin> bins;
  try {
    for (size_t e = 0; e < count; e++) {
      *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
      bool added = false;
      for (PwGroupBin& bin : bins) {
        if (!pwGroupSameTopology(setups[bin.entries.front()], setups[e])) {
          continue;
        }
        size_t srcBytes = 0;
        size_t dstBytes = 0;
        size_t areaBytes = 0;
        if (!m2nCheckedAddSize(bin.srcBytes, srcEntryBytes[e], &srcBytes) ||
            !m2nCheckedAddSize(bin.dstBytes, dstEntryBytes[e], &dstBytes) ||
            !pwGroupAreaBytes(srcBytes, dstBytes, &areaBytes) || areaBytes > stagingBudget) {
          continue;
        }
        bin.entries.push_back(e);
        bin.srcBytes = srcBytes;
        bin.dstBytes = dstBytes;
        bin.areaBytes = areaBytes;
        added = true;
        break;
      }
      if (!added) {
        PwGroupBin bin;
        bin.entries.push_back(e);
        bin.srcBytes = srcEntryBytes[e];
        bin.dstBytes = dstEntryBytes[e];
        NCCL_M2N_CHECK_ARG(pwGroupAreaBytes(bin.srcBytes, bin.dstBytes, &bin.areaBytes), worldRank,
                           "ncclM2nGroupEnd: entry %zu staging size overflow", e);
        bins.push_back(std::move(bin));
      }
    }
    std::stable_sort(bins.begin(), bins.end(),
                     [](const PwGroupBin& a, const PwGroupBin& b) { return a.areaBytes > b.areaBytes; });
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate fusion bins");
  }

  ReshardWorkStream work{};
  NCCL_M2N_CHECK(reshardSetupWorkStream(comm, stream, currentCudaDev, propsResult, &commProps, &work));
  ReshardWorkStreamCompletion workCompletion(stream, &work);
  size_t fusedBins = 0;
  size_t maxBinEntries = 0;
  for (const PwGroupBin& bin : bins) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, bin.entries.front());
    maxBinEntries = std::max(maxBinEntries, bin.entries.size());
    if (bin.entries.size() == 1) {
      const ReshardTensorSetup& setup = setups[bin.entries.front()];
      NCCL_M2N_CHECK(
        reshardCopyPackWindowNormalized(comm, &setup.srcTensor, &setup.dstTensor, work.stream));
      continue;
    }
    NCCL_M2N_CHECK(reshardCopyPackWindowGroupNormalized(comm, setups.get(), bin.entries.data(), originalIndices,
                                                        bin.entries.size(), work.stream, failedOriginalIndex));
    fusedBins++;
#ifdef NCCL_M2N_TESTING
    gFusedSubmissionCount++;
#endif
  }
  RESHARD_INFO(worldRank, "ncclM2nGroupEnd: entries=%zu bins=%zu fusedBins=%zu maxBinEntries=%zu budget=%zu", count,
               bins.size(), fusedBins, maxBinEntries, stagingBudget);
  *handled = true;
  return workCompletion.complete();
}
