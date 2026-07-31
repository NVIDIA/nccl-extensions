/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Staging Reshard Kernel — Multi-Warp Implementation
 *
 * This file contains the pure-RDMA DIRECT kernel and launch wrapper.
 *
 * Source warps pack into local staging memory and issue RDMA puts. Destination
 * warps poll the staging slots and unpack into the user buffer. A rank that is
 * both source and destination executes those phases sequentially.
 *
 * Dependencies:
 *   staging_buffer.h      — StagingKernelParams, StagingFlowCtrl, etc.
 *   staging_primitives.cuh — lsa_rdma_wait_for_credits, rdma_wait_for_credits,
 *                            rdma_signal, staging_poll, rdma_release,
 *                            staging_copy_*
 *   nccl.h / nccl_device.h — ncclGin, ncclDevComm, ncclBarrierSession
 ************************************************************************/

#include "nccl.h"
#include "nccl_device.h"
#include "cuda_runtime.h"

#include "m2n_checks.h"
#include "staging_types.h"
#include "staging_primitives.cuh"

#include <cstdio>

// ============================================================================
// Error-check macros (host-side only)
// ============================================================================

#define SK_CUDACHECK_IMPL(cmd, errorVar)                                                                        \
  do {                                                                                                          \
    cudaError_t errorVar = (cmd);                                                                               \
    if (errorVar != cudaSuccess) {                                                                              \
      NCCL_M2N_FAIL(ncclInternalError, -1, "CUDA operation %s failed: %s", #cmd, cudaGetErrorString(errorVar)); \
    }                                                                                                           \
  } while (0)
#define SK_CUDACHECK(cmd) SK_CUDACHECK_IMPL(cmd, NCCL_M2N_UNIQUE(skCudaError_))

// Named barrier for a subset of threads within a CTA.
// __barrier_sync(id) syncs ALL CTA threads; to sync only `count` threads
// we need the PTX "bar.sync id, count" instruction directly.
__device__ __forceinline__ void barrier_sync_subset(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// ============================================================================
// Direct-Algorithm Kernel — StagingReshardKernel_Direct
// ============================================================================
//
// Pure RDMA with no LSA fan-out or ring forwarding.
//
// Warp layout:
//   Source side: Type 1 (pack) + Type 4 (RDMA send)
//   Dest side:   Type 6 (RDMA recv + unpack), pure copy groups only
//
// ============================================================================

#define DIRECT_TYPE1_NUM_WARPS 8
#define DIRECT_TYPE1_THREADS (DIRECT_TYPE1_NUM_WARPS * 32)
#define DIRECT_TYPE1_BARRIER_ID 1

#define DIRECT_TYPE4_NUM_WARPS 4
#define DIRECT_TYPE4_WARP_START DIRECT_TYPE1_NUM_WARPS
#define DIRECT_SOURCE_TOTAL_WARPS (DIRECT_TYPE1_NUM_WARPS + DIRECT_TYPE4_NUM_WARPS)

#define DIRECT_TYPE6_NUM_GROUPS 4
#define DIRECT_TYPE6_WARPS_PER_GROUP 2
#define DIRECT_TYPE6_TOTAL_WARPS (DIRECT_TYPE6_NUM_GROUPS * DIRECT_TYPE6_WARPS_PER_GROUP)
#define DIRECT_TYPE6_THREADS_PER_GROUP (DIRECT_TYPE6_WARPS_PER_GROUP * 32)
#define DIRECT_TYPE6_BARRIER_ID_BASE 2

#define DIRECT_TOTAL_WARPS \
  (DIRECT_SOURCE_TOTAL_WARPS > DIRECT_TYPE6_TOTAL_WARPS ? DIRECT_SOURCE_TOTAL_WARPS : DIRECT_TYPE6_TOTAL_WARPS)

/* __launch_bounds__ caps register usage to keep the per-CTA register
 * budget below the 65536 SM limit. */
__global__ __launch_bounds__(DIRECT_TOTAL_WARPS * 32, 1) void StagingReshardKernel_Direct(
  StagingKernelParams* __restrict__ params, ncclDevComm devComm) {
  const int channel_id = (int)blockIdx.x;

  int numContexts = min((int)gridDim.x, (int)devComm.ginContextCount);
  int ctasPerContext = (int)gridDim.x / numContexts;
  int ginContext = (int)blockIdx.x / ctasPerContext;
  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const bool isSource = params->isSource;
  const bool isDest = params->isDest;
  const int numRdmaTargets = params->numRdmaTargets;
  const int numRdmaSources = params->numRdmaSources;
  const size_t chunkSize = params->chunkSize;
  const int numChannels = (int)gridDim.x;

  __shared__ int type1_slot;
  __shared__ int type4_num_new[DIRECT_TYPE4_NUM_WARPS];
  __shared__ size_t type4_first_recv_offset[DIRECT_TYPE4_NUM_WARPS];
  __shared__ int type6_num_new[DIRECT_TYPE6_NUM_GROUPS];
  __shared__ size_t type6_first_recv_offset[DIRECT_TYPE6_NUM_GROUPS];

  __shared__ uint64_t rdma_head_bases[MAX_TARGETS];
  __shared__ uint64_t rdma_tail_bases[MAX_SOURCES];
  __shared__ uint64_t local_pipeline_tail_base[MAX_TARGETS];
  __shared__ uint64_t local_pipeline_head_base[MAX_TARGETS];
  __shared__ uint64_t local_put_counter_base[MAX_TARGETS];

  // ========================================================================
  // PROLOGUE: Read GIN signal base values + local pipeline base values
  // ========================================================================
  if (numRdmaTargets > 0 && (int)threadIdx.x < numRdmaTargets) {
    StagingFlowCtrl& fc = params->rdmaTargets[channel_id][threadIdx.x].fc;
    if (fc.useGinSignal) {
      rdma_head_bases[threadIdx.x] = gin.readSignal(fc.localHeadSignal);
    } else {
      rdma_head_bases[threadIdx.x] = 0;
    }
  }
  if (isSource && (int)threadIdx.x < numRdmaTargets) {
    StagingFlowCtrl& local_fc = params->localRdmaFc[channel_id][threadIdx.x];
    StagingRegion& region = params->rdmaRegions[channel_id];
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    local_pipeline_tail_base[threadIdx.x] =
      ld_acquire((const uint64_t*)(staging_base + local_fc.localTailOffset), local_fc.isLocal);
    local_pipeline_head_base[threadIdx.x] =
      ld_acquire((const uint64_t*)(staging_base + local_fc.localHeadOffset), local_fc.isLocal);
    local_put_counter_base[threadIdx.x] = gin.readCounter(local_fc.localPutCounter);
  }
  if (isDest && (int)threadIdx.x < numRdmaSources) {
    StagingFlowCtrl& fc = params->rdmaSources[channel_id][threadIdx.x].fc;
    if (fc.useGinSignal) {
      rdma_tail_bases[threadIdx.x] = gin.readSignal(fc.localTailSignal);
    } else {
      rdma_tail_bases[threadIdx.x] = 0;
    }
  }
  __syncthreads();

  // ========================================================================
  // Initial barrier (all ranks, all CTAs)
  // ========================================================================
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
#else
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
#endif
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // ========================================================================
  // SOURCE SIDE — Type 1 (multi-warp pack) + Type 4 (RDMA send)
  // ========================================================================
  if (isSource && numRdmaTargets > 0) {
    StagingRegion rdma_region = params->rdmaRegions[channel_id];

    // ================================================================
    // Type 1: Multi-Warp Source Pack
    // ================================================================
    if (warp_id < DIRECT_TYPE1_NUM_WARPS) {
      const bool is_root_warp = (warp_id == 0);
      const int thread_in_group = warp_id * 32 + lane_id;

      for (int t = 0; t < numRdmaTargets; t++) {
        StagingFlowCtrl fc = params->localRdmaFc[channel_id][t];
        if (is_root_warp) {
          fc.shadowTail = local_pipeline_tail_base[t];
          fc.lastTailVal = local_pipeline_tail_base[t];
          fc.localHeadVal = local_pipeline_head_base[t];
        }
        const StagingTransferPlan plan = params->rdmaTargets[channel_id][t].plan;
        size_t total_chunks = (plan.totalBytes + chunkSize - 1) / chunkSize;
        size_t my_chunk_start = (total_chunks * (size_t)channel_id) / (size_t)numChannels;
        size_t my_chunk_end = (total_chunks * (size_t)(channel_id + 1)) / (size_t)numChannels;
        size_t my_num_chunks = my_chunk_end - my_chunk_start;

        for (size_t chunk = 0; chunk < my_num_chunks; chunk++) {
          size_t global_chunk = my_chunk_start + chunk;

          if (is_root_warp && lane_id == 0) {
            uint64_t base_offset = local_pipeline_tail_base[t] - local_put_counter_base[t];
            lsa_rdma_wait_for_credits(gin, fc, base_offset);
            type1_slot = (int)(fc.shadowTail % (uint64_t)fc.peerNumSlots);
          }

          barrier_sync_subset(DIRECT_TYPE1_BARRIER_ID, DIRECT_TYPE1_THREADS);

          int slot = type1_slot;
          char* staging_base = (char*)ncclGetLocalPointer(rdma_region.window, 0);
          char* staging_dst = staging_base + fc.peerDataOffset + (size_t)slot * fc.peerChunkSize;
          const char* user_src = (const char*)params->srcBuffer;

          {
            size_t byte_start = global_chunk * chunkSize;
            size_t remaining = plan.totalBytes - byte_start;
            size_t this_bytes = (chunkSize < remaining) ? chunkSize : remaining;

            if (plan.isContiguous) {
              staging_copy_contig(staging_dst, user_src + plan.srcBaseOffset + byte_start, this_bytes,
                                  DIRECT_TYPE1_THREADS, thread_in_group);
            } else {
              staging_copy_pack(staging_dst, user_src, plan, byte_start, this_bytes, DIRECT_TYPE1_THREADS,
                                thread_in_group);
            }
          }

          barrier_sync_subset(DIRECT_TYPE1_BARRIER_ID, DIRECT_TYPE1_THREADS);

          if (is_root_warp && lane_id == 0) {
            rdma_signal(gin, world, rdma_region, fc);
          }
        }
      }
    }

    // ================================================================
    // Type 4: Multi-Warp RDMA Send
    // ================================================================
    else if (warp_id >= DIRECT_TYPE4_WARP_START && warp_id < DIRECT_TYPE4_WARP_START + DIRECT_TYPE4_NUM_WARPS) {
      const int type4_idx = warp_id - DIRECT_TYPE4_WARP_START;

      const int my_tgt_start = (numRdmaTargets * type4_idx) / DIRECT_TYPE4_NUM_WARPS;
      const int my_tgt_end = (numRdmaTargets * (type4_idx + 1)) / DIRECT_TYPE4_NUM_WARPS;

      for (int t = my_tgt_start; t < my_tgt_end; t++) {
        StagingFlowCtrl local_fc = params->localRdmaFc[channel_id][t];
        local_fc.shadowTail = local_pipeline_tail_base[t];
        local_fc.lastTailVal = local_pipeline_tail_base[t];
        local_fc.localHeadVal = local_pipeline_head_base[t];
        StagingFlowCtrl rdma_fc = params->rdmaTargets[channel_id][t].fc;
        rdma_fc.headSignalBase = rdma_head_bases[t];

        const StagingTransferPlan& plan = params->rdmaTargets[channel_id][t].plan;
        const int target_rank = params->rdmaTargets[channel_id][t].peerWorldRank;

        size_t total_chunks = (plan.totalBytes + chunkSize - 1) / chunkSize;
        size_t my_chunk_start = (total_chunks * (size_t)channel_id) / (size_t)numChannels;
        size_t my_chunk_end = (total_chunks * (size_t)(channel_id + 1)) / (size_t)numChannels;
        size_t my_num_chunks = my_chunk_end - my_chunk_start;

        size_t chunks_done = 0;
        while (chunks_done < my_num_chunks) {
          size_t first_recv_offset = 0;
          int num_new = 0;
          if (lane_id == 0) {
            while ((num_new = staging_poll(rdma_region, local_fc, &first_recv_offset)) == 0) {
            }
          }
          if (lane_id == 0) {
            type4_num_new[type4_idx] = num_new;
            type4_first_recv_offset[type4_idx] = first_recv_offset;
          }
          __syncwarp();
          num_new = type4_num_new[type4_idx];
          first_recv_offset = type4_first_recv_offset[type4_idx];

          int batch = num_new;
          if ((size_t)batch > my_num_chunks - chunks_done) {
            batch = (int)(my_num_chunks - chunks_done);
          }

          int first_slot = (local_fc.peerChunkSize > 0) ?
                             (int)((first_recv_offset - local_fc.peerDataOffset) / local_fc.peerChunkSize) :
                             0;

          for (int bi = 0; bi < batch; bi++) {
            size_t chunk = chunks_done + (size_t)bi;
            size_t global_chunk = my_chunk_start + chunk;

            int this_slot = (first_slot + bi) % local_fc.peerNumSlots;
            size_t local_recv_offset = local_fc.peerDataOffset + (size_t)this_slot * local_fc.peerChunkSize;

            size_t byte_start = global_chunk * chunkSize;
            size_t remaining = plan.totalBytes - byte_start;
            size_t chunk_bytes = (chunkSize < remaining) ? chunkSize : remaining;

            if (lane_id == 0) {
              rdma_wait_for_credits(gin, rdma_region, rdma_fc);

              int remote_slot = (int)(rdma_fc.shadowTail % (uint64_t)rdma_fc.peerNumSlots);
              size_t remote_offset = rdma_fc.peerDataOffset + (size_t)remote_slot * rdma_fc.peerChunkSize;

              gin.put(world, target_rank, params->rdmaWindow, remote_offset, params->rdmaWindow, local_recv_offset,
                      chunk_bytes, ncclGin_None{}, ncclGin_CounterInc{local_fc.localPutCounter});
              rdma_signal(gin, world, rdma_region, rdma_fc);
            }
            __syncwarp();
          }

          chunks_done += (size_t)batch;
        }
      }
    }

  } // end source side

  // ========================================================================
  // DEST SIDE — Type 6 (Multi-Group RDMA Receive + Unpack)
  // ========================================================================
  if (isDest && numRdmaSources > 0) {
    StagingRegion rdma_region = params->rdmaRegions[channel_id];

    if (warp_id < DIRECT_TYPE6_TOTAL_WARPS) {
      const int group_id = warp_id / DIRECT_TYPE6_WARPS_PER_GROUP;
      const int warp_in_group = warp_id % DIRECT_TYPE6_WARPS_PER_GROUP;
      const bool is_root_warp = (warp_in_group == 0);
      const int group_barrier = DIRECT_TYPE6_BARRIER_ID_BASE + group_id;
      const int dst_copy_thread = warp_in_group * 32 + lane_id;
      const int dst_copy_threads = DIRECT_TYPE6_WARPS_PER_GROUP * 32;

      char* rdma_staging_base = (char*)ncclGetLocalPointer(rdma_region.window, 0);
      char* user_dst = (char*)params->dstBuffer;

      const int my_src_start = (numRdmaSources * group_id) / DIRECT_TYPE6_NUM_GROUPS;
      const int my_src_end = (numRdmaSources * (group_id + 1)) / DIRECT_TYPE6_NUM_GROUPS;

      for (int s = my_src_start; s < my_src_end; s++) {
        StagingFlowCtrl rdma_fc = params->rdmaSources[channel_id][s].fc;
        if (is_root_warp) {
          rdma_fc.tailSignalBase = rdma_tail_bases[s];
        }

        const StagingTransferPlan& plan = params->rdmaSources[channel_id][s].plan;

        size_t total_chunks = (plan.totalBytes + chunkSize - 1) / chunkSize;
        size_t my_chunk_start = (total_chunks * (size_t)channel_id) / (size_t)numChannels;
        size_t my_chunk_end = (total_chunks * (size_t)(channel_id + 1)) / (size_t)numChannels;
        size_t my_num_chunks = my_chunk_end - my_chunk_start;

        size_t chunks_done = 0;
        while (chunks_done < my_num_chunks) {
          size_t first_recv_offset = 0;
          int num_new = 0;
          if (is_root_warp && lane_id == 0) {
            while ((num_new = rdma_poll(gin, rdma_region, rdma_fc, &first_recv_offset)) == 0) {
            }
            type6_num_new[group_id] = num_new;
            type6_first_recv_offset[group_id] = first_recv_offset;
          }

          barrier_sync_subset(group_barrier, DIRECT_TYPE6_THREADS_PER_GROUP);

          num_new = type6_num_new[group_id];
          first_recv_offset = type6_first_recv_offset[group_id];

          int batch = num_new;
          if ((size_t)batch > my_num_chunks - chunks_done) {
            batch = (int)(my_num_chunks - chunks_done);
          }

          int first_slot = (rdma_fc.peerChunkSize > 0) ?
                             (int)((first_recv_offset - rdma_fc.peerDataOffset) / rdma_fc.peerChunkSize) :
                             0;

          for (int bi = 0; bi < batch; bi++) {
            int this_slot = (first_slot + bi) % rdma_fc.peerNumSlots;
            size_t recv_offset = rdma_fc.peerDataOffset + (size_t)this_slot * rdma_fc.peerChunkSize;

            size_t global_chunk = my_chunk_start + chunks_done + (size_t)bi;
            size_t byte_start = global_chunk * chunkSize;
            size_t remaining = plan.totalBytes - byte_start;
            size_t this_bytes = (chunkSize < remaining) ? chunkSize : remaining;

            const char* staging_src = rdma_staging_base + recv_offset;

            if (plan.isContiguous) {
              staging_copy_contig(user_dst + plan.dstBaseOffset + byte_start, staging_src, this_bytes, dst_copy_threads,
                                  dst_copy_thread);
            } else {
              staging_copy_unpack(user_dst, staging_src, plan, byte_start, this_bytes, dst_copy_threads,
                                  dst_copy_thread);
            }
          }

          barrier_sync_subset(group_barrier, DIRECT_TYPE6_THREADS_PER_GROUP);

          if (is_root_warp && lane_id == 0) {
            rdma_release_flush(gin, world, rdma_region, rdma_fc);
          }

          chunks_done += (size_t)batch;
        }
      }
    }

  } // end dest side

  // ========================================================================
  // Final barrier (all ranks, all CTAs)
  // ========================================================================
  __syncthreads();
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
}

// ============================================================================
// Host-Side Launch Wrapper — Direct Algorithm
// ============================================================================

ncclResult_t launchStagingReshardDirect(const StagingKernelParams* hostParams, StagingKernelParams* devParams,
                                        struct ncclDevComm* devComm, int numCtas, cudaStream_t stream, bool verbose) {
  SK_CUDACHECK(cudaMemcpyAsync(devParams, hostParams, sizeof(StagingKernelParams), cudaMemcpyHostToDevice, stream));

  const int threads_per_cta = DIRECT_TOTAL_WARPS * 32;

  if (verbose) {
    printf("[STAGING_KERNEL_DIRECT] Launching: grid=%d, block=%d "
           "(Type1=%d, Type4=%d, Type6=%dx%d), stream=%p\n",
           numCtas, threads_per_cta, DIRECT_TYPE1_NUM_WARPS, DIRECT_TYPE4_NUM_WARPS, DIRECT_TYPE6_NUM_GROUPS,
           DIRECT_TYPE6_WARPS_PER_GROUP, (void*)stream);
    fflush(stdout);
  }

  StagingReshardKernel_Direct<<<numCtas, threads_per_cta, 0, stream>>>(devParams, *devComm);

  SK_CUDACHECK(cudaGetLastError());

  if (verbose) {
    printf("[STAGING_KERNEL_DIRECT] Kernel enqueued successfully\n");
    fflush(stdout);
  }

  return ncclSuccess;
}
