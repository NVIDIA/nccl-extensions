/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// Largest NUM_ACC the RED accumulator loops can keep fully register-resident.
// Each thread holds float2 acc_token_fp32[NUM_ACC], i.e. 2*NUM_ACC 32-bit registers,
// against a 255-register-per-thread budget; 64 leaves 128 for accumulators and the
// rest for addresses, barriers and dequant temporaries.
//
// NUM_ACC = ceil((H/2) / THRDS_PER_PIPELINE), THRDS_PER_PIPELINE = (RED_WARPS/P)*32.
// H=7168 1-node P=2 → 56 (fits). H=16384 1-node P=2 → 128 (would spill to local memory).
//
// Enforced on the host in ht_ep_adapter.cu, which lowers the default pipeline count
// until NUM_ACC fits (H=16384: P=2 → P=1 → NUM_ACC=64). Pinning
// NCCL_EP_COMBINE_NUM_PIPELINES bypasses that and may spill.
#ifndef NCCL_EP_COMBINE_MAX_ACC
#define NCCL_EP_COMBINE_MAX_ACC 64
#endif

#include "nccl_ep.h"
#include "common.hpp"
#include "device_primitives.cuh"
#include "lsa_completion.cuh"
#include "ht_ep_configs.cuh"
#include "mxfp8_quant.cuh"
#include "quant_recipe.cuh"
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda/ptx>
#include "nccl_device.h"
#include "cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x

// Per-warp transfer stage for pull dispatch: read one remote row once, store it to
// each of the K local EM slots (read-once-scatter-many). The row is TMA'd
// (cp.async.bulk) into a per-warp smem buffer, then TMA-stored to each EM slot; the
// pull kernel is a thin driver over this stage.
//
// Contract (all calls warp-collective, 32 lanes; kRowBytes must be a multiple of 16):
//   using P = PullTmaStage<kRowBytes>;
//   P p; p.init(lane, warp_smem);          // per-warp setup; mbarrier_init here
//   P::block_init_fence();                 // driver: barrier+fence after ALL warps init
//   for (recv token t) {
//     p.load(remote_row_ptr);              // ingest the remote row
//     for (a < cnt) p.store(em_row_ptr);   // scatter to each EM slot
//     bulk_wait_reads(lane);               // release the smem buffer for the next row
//   }
//   bulk_drain(lane);                      // once, before the block exits
// Driver lays out dynamic smem as warpId * P::smem_bytes_per_warp (16B-aligned base).
namespace ht_ep {

// Release the staged smem buffers for reuse: waits only until the bulk stores have
// *read* their source, so writes keep draining while the next remote row is in flight.
__device__ __forceinline__ void bulk_wait_reads(int lane) {
    if (lane == 0) cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
    __syncwarp();
}
// Full completion of every bulk store issued by this warp; once before block exit.
__device__ __forceinline__ void bulk_drain(int lane) {
    if (lane == 0) cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    __syncwarp();
}

// TMA (cp.async.bulk) read -> per-warp smem row, then TMA-store scatter to each EM
// slot. Used for both the token row and the QUANT_FWD scale row.
template <int kRowBytes>
struct PullTmaStage {
    static_assert(kRowBytes > 0 && (kRowBytes % 16) == 0,
                  "PullTmaStage: row must be a non-zero multiple of 16B (cp.async.bulk granularity)");
    // One 16B-aligned row buffer + one mbarrier per warp.
    static constexpr int smem_bytes_per_warp = kRowBytes + 16;

    int lane_;
    uint32_t phase_;
    uint8_t* buf_;
    uint64_t* mbar_;

    __device__ __forceinline__ void init(int lane, uint8_t* warp_smem) {
        lane_ = lane;
        phase_ = 0;
        buf_ = warp_smem;
        mbar_ = reinterpret_cast<uint64_t*>(warp_smem + kRowBytes);
        if (lane_ == 0) cuda::ptx::mbarrier_init(mbar_, 1);
    }
    // Make the mbarrier inits visible to the async proxy before any load; call once
    // per block after all warps have run init().
    static __device__ __forceinline__ void block_init_fence() {
        __syncthreads();
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        __syncthreads();
    }
    // Split issue/wait lets the driver overlap other work (e.g. the weight read)
    // with the in-flight async row transfer.
    __device__ __forceinline__ void issue(const void* __restrict__ remote_row) {
        if (lane_ == 0) {
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                                     buf_, remote_row, static_cast<uint32_t>(kRowBytes), mbar_);
            cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta,
                                                 cuda::ptx::space_shared, mbar_,
                                                 static_cast<uint32_t>(kRowBytes));
        }
    }
    __device__ __forceinline__ void wait() {
        __syncwarp();
        while (!cuda::ptx::mbarrier_try_wait_parity(mbar_, phase_)) {}
        phase_ ^= 1;
        __syncwarp();
    }
    __device__ __forceinline__ void load(const void* __restrict__ remote_row) {
        issue(remote_row);
        wait();
    }
    // Commits to the per-thread bulk group; released by bulk_wait_reads()/bulk_drain().
    __device__ __forceinline__ void store(void* __restrict__ em_row) {
        if (lane_ == 0) {
            cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                                     em_row, buf_, static_cast<uint32_t>(kRowBytes));
            cuda::ptx::cp_async_bulk_commit_group();
        }
    }
};

// Conditionally allocate compile-time arrays only when enabled.
template <bool ENABLE, int N>
struct acc_prob_storage_t {};

template <int N>
struct acc_prob_storage_t<true, N> {
    float data[N];
};

// Generic warp group for warp-specializaion.
template <int NUM_WARPS, int STARTING_WARPS>
struct warp_group {
    __host__ __device__ static constexpr int size() {
        return 32 * NUM_WARPS;
    }
    __host__ __device__ static constexpr int warp_size() {
        return NUM_WARPS;
    }

    __host__ __device__ static int thread_rank() {
        return threadIdx.x - (32 * STARTING_WARPS);
    }
    __host__ __device__ static int warp_rank() {
        return thread_rank() / 32;
    }
};

// Memory region info structs for GIN (gin-deepep style)
// All buffers are part of a single large gin_base_ptr buffer
// Offsets are relative to gin_base_ptr (stored as size_t for offset calculation)
struct dispatch_memory_region_info_t {
    size_t attn_input_token_offset;           // Offset of token staging buffer from gin_base_ptr
    size_t attn_input_prob_offset;             // Offset of prob staging buffer from gin_base_ptr
    size_t attn_input_scaling_factor_offset;   // Offset of scaling factor staging buffer
  // Batched RDMA staging (packed layout: token+prob+sf per entry)
    size_t gin_send_staging_offset;           // Offset of per-destination staging buffer
    size_t gin_recv_staging_offset; // Offset of packed receive buffer (token+prob+sf per entry)
    size_t guard_offset; // Offset of RDMA sync-guard readiness flags (LSA_TEAMS uint64 slots)
    size_t bytes_per_entry; // Size of packed entry (token + prob + sf)
    size_t max_tokens_per_dest; // Max tokens that can be staged per destination
    // Streaming RDMA signals
    unsigned signals_tail_base; // Base signal ID for tail tracking (sender -> receiver)
    // Streaming buffer configuration
    int num_max_rdma_chunked_send_tokens; // Batch size per RDMA put (default: 6)
} __attribute__((__aligned__(8)));

// Tail-signal id for the (src_lteam -> dst_lteam) edge of (local_rank, chunk);
// namespace [src][dst][local_rank][chunk].
__forceinline__ __device__ unsigned dispatch_tail_signal_id(
    unsigned signals_tail_base,
    int src_lteam,
    int dst_lteam,
    int local_rank,
    int cidx,
    int lsa_teams,
    int ranks_per_lsa_team,
    int max_chunks_per_rank) {
    return signals_tail_base +
           ((src_lteam * lsa_teams + dst_lteam) * ranks_per_lsa_team + local_rank) * max_chunks_per_rank + cidx;
}

// Byte offset (from the packed inter-lsa-team receive region start) of one source slot's chunk.
// Packed layout is [remote_slot][token-in-slot], each entry bytes_per_entry (token + prob + sf).
__forceinline__ __device__ size_t
dispatch_packed_entry_offset(const dispatch_memory_region_info_t* mr, int remote_slot, int chunk_first_token) {
    return mr->gin_recv_staging_offset +
           (static_cast<size_t>(remote_slot) * mr->max_tokens_per_dest +
            static_cast<size_t>(chunk_first_token)) *
               mr->bytes_per_entry;
}

struct combine_memory_region_info_t {
    size_t combine_red_token_offset; // Offset of combine LSA-team-reduced token buffer
    size_t combine_g2s_token_offset; // Offset of combine cross-LSA-team (N2N RDMA) token buffer
    size_t combine_red_prob_offset; // Offset of combine LSA-team-reduced prob buffer
    size_t combine_g2s_prob_offset; // Offset of combine cross-LSA-team (N2N RDMA) prob buffer
    size_t guard_offset; // RDMA sync-guard: offset of combine's internal-buffer readiness flags
} __attribute__((__aligned__(8)));

// ============================================================================
// Warp-parallel memory copy helper for RDMA staging
// All 32 threads participate using int4 (16-byte) loads/stores for maximum bandwidth
// ============================================================================
template <int STRIDE = 32>
__device__ __forceinline__ void
warp_copy_int4(void* __restrict__ dst, const void* __restrict__ src, size_t bytes, int lane_id) {
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    const int count = bytes / sizeof(int4);

#pragma unroll 4
    for (int i = lane_id; i < count; i += STRIDE) {
        dst4[i] = __ldg(src4 + i);
    }
    __syncwarp();
}

struct dispatch_config_t {
    int num_of_stages;
    int num_of_in_flight_s2g;
    int num_of_tokens_per_chunk;
    int num_of_blocks;
    bool forward_dispatch;
    bool device_side_sync;
    int s2d_inner_dim; // flat: lsa_team_size, expert-major: num_topk
    int num_pipelines;
    int stages_per_pipeline;
    int sf_bytes_per_token; // total scale bytes per token (pre-computed on host)
};

struct combine_config_t {
    int num_of_stages_g2s;
    int num_of_stages_s2g;
    int num_pipelines;
    int num_of_tokens_per_chunk;
    int num_of_tokens_per_group;
    int num_of_blocks;
    bool backward_combine;
    bool device_side_sync;
};

struct model_config_t {
    int hidden_dim;
    int max_num_of_tokens_per_rank;
    int num_of_experts_per_rank;
    int ranks_per_lsa_team;
    int num_lsa_teams;
};

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
struct dispatch_warp_timing_entry_t {
    long long start_clock;
    long long end_clock;
};
struct combine_warp_timing_entry_t {
    long long work_start_clock;
    long long work_end_clock;
};
struct combine_block_timing_entry_t {
    long long head_sync_start_clock;
    long long head_sync_end_clock;
};
#endif

// Expert-major S2D entry: int32 packed as [31:22]=rank_id (10b, <= 1024), [21:0]=slot (22b).
// -1 = no entry.  Only used when expert-major layout is active.
constexpr int EM_S2D_RANK_BITS = 10;
constexpr int EM_S2D_SLOT_BITS = 32 - EM_S2D_RANK_BITS;
constexpr int EM_S2D_MAX_RANKS = 1 << EM_S2D_RANK_BITS; // 1024
constexpr uint32_t EM_S2D_SLOT_MASK = (1u << EM_S2D_SLOT_BITS) - 1u;
// Slot field must hold values up to MAX_SUPPORTED_TOKENS_PER_RANK; -1 sentinel must not collide.
static_assert(
    MAX_SUPPORTED_TOKENS_PER_RANK < (1 << EM_S2D_SLOT_BITS),
    "MAX_SUPPORTED_TOKENS_PER_RANK exceeds em_s2d slot field width");

// s2d-map double-buffer depth: consume one stage while prefetching the next (chunk, LSA team) row.
constexpr int S2D_MAP_RING_STAGES = 2;

__host__ __device__ __forceinline__ int32_t em_s2d_pack(int rank_id, int slot) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(rank_id) << EM_S2D_SLOT_BITS) | (static_cast<uint32_t>(slot) & EM_S2D_SLOT_MASK));
}
__host__ __device__ __forceinline__ int em_s2d_unpack_rank(int32_t v) {
    return static_cast<int>(static_cast<uint32_t>(v) >> EM_S2D_SLOT_BITS);
}
__host__ __device__ __forceinline__ int em_s2d_unpack_slot(int32_t v) {
    return static_cast<int>(static_cast<uint32_t>(v) & EM_S2D_SLOT_MASK);
}

// NCCL_EP_OVERFLOW_DROP helpers: a recv slot at/above capacity would index past
// the recv buffer, so the drop policy remaps it to the sentinel -1.
__host__ __device__ __forceinline__ bool slot_overflows(int32_t slot, int capacity) {
    return slot >= capacity;
}
__host__ __device__ __forceinline__ int32_t drop_overflow_slot(int32_t slot, bool drop, int capacity) {
    return (drop && slot_overflows(slot, capacity)) ? -1 : slot;
}

// Combine kernels pad each LSA team's rdma_to_attn_map row to a multiple of 16 bools;
// producers of the map must use the same stride.
__host__ __device__ __forceinline__ int rdma_to_attn_row_stride(int tokens_per_rank) {
    return (((tokens_per_rank - 1) / 16) + 1) * 16;
}

// EM unfused-combine path: s2d entries are lex-sorted (dest_rank, k), so duplicates
// within a row are adjacent. Returns true when this lane's entry has the same
// dest_rank as the lane immediately upstream -- the peer rank's local_reduce_kernel
// has already merged it into the primary slot, so this entry should be skipped.
// Must be called with all 32 lanes active (uses __shfl_up_sync(0xffffffff, ...)).
template <ncclEpLayout_t kLayout>
__device__ __forceinline__ bool is_em_secondary_entry(int32_t s2d_val, int lane_id, bool enabled) {
    if constexpr (kLayout != NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        return false;
    }
    const int32_t prev = __shfl_up_sync(0xffffffff, s2d_val, 1);
    return enabled && lane_id > 0 && s2d_val != -1 && prev != -1 &&
           em_s2d_unpack_rank(s2d_val) == em_s2d_unpack_rank(prev);
}

struct combine_smem_layout_t {
    uint16_t* lsa_token_G2S_buffer;
    uint16_t* lsa_token_S2G_buffer;
    uint16_t* cross_lsa_token_G2S_buffer;
    uint16_t* cross_lsa_token_S2G_buffer;
    float* lsa_prob_G2S_buffer;
    float* lsa_prob_S2G_buffer;
    float* cross_lsa_prob_G2S_buffer;
    float* cross_lsa_prob_S2G_buffer;
    uint64_t* lsa_mbarrier_G2S_buffer;
    uint64_t* cross_lsa_mbarrier_G2S_buffer;
    uint64_t* lsa_to_rdma_mbarrier_buffer;
    bool* lsa_flag_G2S_buffer;
    bool* cross_lsa_flag_G2S_buffer;

    // No separate scale staging: MXFP8 E8M0 rows always ride in the token stage tail, so a
    // stage base plus HIDDEN_DIM locates them.
    // Shared by the intra-LSA and cross-LSA G2S rings, which carry the same row:
    // packed [FP8 H | E8M0 H/32] for MXFP8, dtype width for NONE.
    int token_G2S_stage_stride; // elements (not bytes)
    int token_S2G_stage_stride; // elements (not bytes)
    int prob_G2S_stage_stride; // elements (not bytes)
    int prob_S2G_stage_stride; // intra-LSA elements (not bytes)
    int prob_S2G_inter_stage_stride; // cross-LSA-team elements (not bytes)
    combine_memory_region_info_t* combine_memory_region_info;

    // Streaming overlap: reduction warp -> RDMA warp within a chunk
    uint32_t* rdma_streaming_counter; // [1] cumulative tokens produced for the current chunk

    int s2d_inner_dim; // Inner dimension of unified S2D map (ranks per LSA team or num_topk)

    // Accessor methods for staged buffers
    __device__ __forceinline__ uint16_t* get_lsa_token_G2S(int stage) const {
        return lsa_token_G2S_buffer + stage * token_G2S_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_lsa_token_S2G(int stage) const {
        return lsa_token_S2G_buffer + stage * token_S2G_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_cross_lsa_token_G2S(int stage) const {
        return cross_lsa_token_G2S_buffer + stage * token_G2S_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_cross_lsa_token_S2G(int stage) const {
        return cross_lsa_token_S2G_buffer + stage * token_S2G_stage_stride;
    }
    __device__ __forceinline__ float* get_lsa_prob_G2S(int stage) const {
        return lsa_prob_G2S_buffer + stage * prob_G2S_stage_stride;
    }
    __device__ __forceinline__ float* get_lsa_prob_S2G(int stage) const {
        return lsa_prob_S2G_buffer + stage * prob_S2G_stage_stride;
    }
    __device__ __forceinline__ float* get_cross_lsa_prob_G2S(int stage) const {
        return cross_lsa_prob_G2S_buffer + stage * prob_G2S_stage_stride;
    }
    __device__ __forceinline__ float* get_cross_lsa_prob_S2G(int stage) const {
        return cross_lsa_prob_S2G_buffer + stage * prob_S2G_inter_stage_stride;
    }
    // Accessor methods for mbarrier buffers (producer = stage*2, consumer = stage*2+1)
    __device__ __forceinline__ uint64_t* get_lsa_mbarrier_G2S_producer(int stage) const {
        return lsa_mbarrier_G2S_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_lsa_mbarrier_G2S_consumer(int stage) const {
        return lsa_mbarrier_G2S_buffer + stage * 2 + 1;
    }
    __device__ __forceinline__ uint64_t* get_cross_lsa_mbarrier_G2S_producer(int stage) const {
        return cross_lsa_mbarrier_G2S_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_cross_lsa_mbarrier_G2S_consumer(int stage) const {
        return cross_lsa_mbarrier_G2S_buffer + stage * 2 + 1;
    }
};

struct dispatch_smem_layout_t {
    void* lsa_token_buffer;
    float* lsa_prob_buffer;
    uint8_t* lsa_scaling_factor_buffer;
    int32_t* sparse_to_dense_map_buffer;
    bool* attn_to_rdma_map_buffer;
    uint64_t* lsa_mbarrier_buffer;
    uint64_t* sparse_to_dense_map_mbarrier_buffer;
    uint64_t* S2G_group_mbarrier_buffer;
    // Single TMA staging slot used by the PAD warp to broadcast a zeroed token
    // row to padding slots (expert-major only; nullptr otherwise).
    void* pad_tma_buffer;

    int token_buffer_stage_stride; // bytes
    int prob_buffer_stage_stride; // bytes
    int sf_buffer_stage_stride; // bytes
    int s2d_map_stage_stride; // bytes (flat: tokens * ranks, expert-major: tokens * topk)
    int pad_tma_slot_bytes; // bytes (= padded hidden_dim * sizeof(token))
    int s2d_inner_dim; // flat: lsa_team_size, expert-major: num_topk
    int num_pipelines;
    int stages_per_pipeline;
    dispatch_memory_region_info_t* dispatch_memory_region_info;

    // Flat stage accessors (used when pipeline_id is already folded into stage)
    __device__ __forceinline__ void* get_token_buffer(int stage) const {
        return reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(lsa_token_buffer) + stage * token_buffer_stage_stride);
    }
    __device__ __forceinline__ float* get_prob_buffer(int stage) const {
        return reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(lsa_prob_buffer) + stage * prob_buffer_stage_stride);
    }
    __device__ __forceinline__ void* get_sf_buffer(int stage) const {
        return reinterpret_cast<void*>(lsa_scaling_factor_buffer + stage * sf_buffer_stage_stride);
    }
    __device__ __forceinline__ uint64_t* get_lsa_mbarrier_producer(int stage) const {
        return lsa_mbarrier_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_lsa_mbarrier_consumer(int stage) const {
        return lsa_mbarrier_buffer + stage * 2 + 1;
    }

    // Pipeline-indexed stage accessors: translate (pipeline_id, local_stage) to absolute stage
    __device__ __forceinline__ void* get_token_buffer(int pipeline_id, int local_stage) const {
        return get_token_buffer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ float* get_prob_buffer(int pipeline_id, int local_stage) const {
        return get_prob_buffer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ void* get_sf_buffer(int pipeline_id, int local_stage) const {
        return get_sf_buffer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ uint64_t* get_lsa_mbarrier_producer(int pipeline_id, int local_stage) const {
        return get_lsa_mbarrier_producer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ uint64_t* get_lsa_mbarrier_consumer(int pipeline_id, int local_stage) const {
        return get_lsa_mbarrier_consumer(pipeline_id * stages_per_pipeline + local_stage);
    }

    // Per-pipeline s2d_map accessors: each pipeline has its own S2D_MAP_RING_STAGES ping-pong stages
    __device__ __forceinline__ int32_t* get_s2d_map_buffer(int pipeline_id, int stage, int token_idx) const {
        int abs_stage = pipeline_id * S2D_MAP_RING_STAGES + stage;
        return reinterpret_cast<int32_t*>(
                   reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + abs_stage * s2d_map_stage_stride) +
               token_idx * s2d_inner_dim;
    }
    __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int pipeline_id, int stage) const {
        int abs_stage = pipeline_id * S2D_MAP_RING_STAGES + stage;
        return reinterpret_cast<int32_t*>(
            reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + abs_stage * s2d_map_stage_stride);
    }
    // Legacy s2d accessors (pipeline_id=0)
    __device__ __forceinline__ int32_t* get_s2d_map_buffer(int stage, int token_idx) const {
        return get_s2d_map_buffer(0, stage, token_idx);
    }
    __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int stage) const {
        return get_s2d_map_buffer_base(0, stage);
    }

    // Per-pipeline s2d_map mbarrier: each pipeline has S2D_MAP_RING_STAGES ping-pong mbarriers
    __device__ __forceinline__ uint64_t* get_s2d_map_mbar(int pipeline_id, int stage) const {
        return sparse_to_dense_map_mbarrier_buffer + pipeline_id * S2D_MAP_RING_STAGES + stage;
    }
    // Per-pipeline S2G group mbarrier
    __device__ __forceinline__ uint64_t* get_S2G_group_mbar(int pipeline_id) const {
        return S2G_group_mbarrier_buffer + pipeline_id;
    }

    __device__ __forceinline__ void* get_pad_tma_slot() const {
        return pad_tma_buffer;
    }
};

template <ncclEpLayout_t kLayout, int kTokenSize>
__device__ dispatch_smem_layout_t create_dispatch_smem_layout(
    dispatch_smem_layout_t& layout,
    void* smem_base,
    const dispatch_config_t& config,
    const model_config_t& model) {
    static_assert(kTokenSize > 0, "token size must be positive");
    size_t offset = 0;
    const int num_pipelines = config.num_pipelines;
    layout.num_pipelines = num_pipelines;
    layout.stages_per_pipeline = config.stages_per_pipeline;

    // Token buffer (aligned to 128B for TMA) -- total stages unchanged
    layout.token_buffer_stage_stride = model.hidden_dim * kTokenSize;
    layout.token_buffer_stage_stride = (layout.token_buffer_stage_stride + 127) & ~127;
    layout.lsa_token_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * layout.token_buffer_stage_stride;

    // Sparse to dense map buffer: S2D_MAP_RING_STAGES ping-pong stages PER PIPELINE (128B aligned)
    // Inner dim is mode-dependent: flat = lsa_team_size, expert-major = num_topk.
    layout.s2d_inner_dim = config.s2d_inner_dim;
    layout.s2d_map_stage_stride = config.num_of_tokens_per_chunk * config.s2d_inner_dim * sizeof(int32_t);
    layout.s2d_map_stage_stride = (layout.s2d_map_stage_stride + 127) & ~127;
    layout.sparse_to_dense_map_buffer = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += S2D_MAP_RING_STAGES * num_pipelines * layout.s2d_map_stage_stride;

    // Prob buffer (only if forward dispatch, 16B aligned) -- total stages unchanged
    if (config.forward_dispatch) {
        layout.prob_buffer_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
        layout.prob_buffer_stage_stride = (layout.prob_buffer_stage_stride + 15) & ~15;
        layout.lsa_prob_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_stages * layout.prob_buffer_stage_stride;
    } else {
        layout.lsa_prob_buffer = nullptr;
        layout.prob_buffer_stage_stride = 0;
    }

    // Scaling factor buffer (only if quantized, 16B aligned) -- total stages unchanged
    if (config.sf_bytes_per_token > 0) {
        layout.sf_buffer_stage_stride = config.sf_bytes_per_token;
        layout.sf_buffer_stage_stride = (layout.sf_buffer_stage_stride + 15) & ~15;
        layout.lsa_scaling_factor_buffer = reinterpret_cast<uint8_t*>(smem_base) + offset;
        offset += config.num_of_stages * layout.sf_buffer_stage_stride;
    } else {
        layout.lsa_scaling_factor_buffer = nullptr;
        layout.sf_buffer_stage_stride = 0;
    }

    // attn_to_rdma_map buffer (16B aligned, only if multi_lsa, shared across pipelines)
    if (model.num_lsa_teams > 1) {
        offset = (offset + 15) & ~15;
        layout.attn_to_rdma_map_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_tokens_per_chunk * (model.num_lsa_teams - 1) * sizeof(bool);
    } else {
        layout.attn_to_rdma_map_buffer = nullptr;
    }

    // Mbarrier buffers (8B aligned) -- total stages unchanged (producer+consumer per stage)
    offset = (offset + 7) & ~7;
    layout.lsa_mbarrier_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * 2 * sizeof(uint64_t);

    // Per-pipeline s2d_map mbarriers: 2 per pipeline
    layout.sparse_to_dense_map_mbarrier_buffer =
        reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += 2 * num_pipelines * sizeof(uint64_t);

    // Per-pipeline S2G group mbarrier: 1 per pipeline
    layout.S2G_group_mbarrier_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_pipelines * sizeof(uint64_t);

    if (model.num_lsa_teams > 1) {
        offset = (offset + 7) & ~7;
        layout.dispatch_memory_region_info =
            reinterpret_cast<dispatch_memory_region_info_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_lsa_teams - 1) * sizeof(dispatch_memory_region_info_t);
    } else {
        layout.dispatch_memory_region_info = nullptr;
    }

    // PAD warp TMA slot: one zeroed token/scale row, broadcast to padding slots.
    // Only allocated for expert-major; flat leaves the pointer null.
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        const int token_bytes = model.hidden_dim * kTokenSize;
        int pad_bytes = token_bytes > config.sf_bytes_per_token ? token_bytes : config.sf_bytes_per_token;
        layout.pad_tma_slot_bytes = (pad_bytes + 127) & ~127;
        offset = (offset + 127) & ~127;
        layout.pad_tma_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += layout.pad_tma_slot_bytes;
    } else {
        layout.pad_tma_buffer = nullptr;
        layout.pad_tma_slot_bytes = 0;
    }

    return layout;
}
// Bytes added to the dispatch SMEM layout by a SINGLE pipeline stage. The
// token, probability, scaling-factor and mbarrier buffers are each replicated
// once per stage, so the stage-scaled portion of the layout is exactly
// config.num_of_stages * this value. Every per-stage stride is individually
// aligned (128B token, 16B prob/sf, 8B mbarrier), so the returned size is a
// multiple of 16B and never perturbs the 8B/16B alignment of the
// stage-independent sections that follow it. kLayout is irrelevant here: no
// per-stage buffer depends on the layout mode.
template <int kTokenSize>
static size_t calc_disp_per_stage_smem(const dispatch_config_t& config, const model_config_t& model) {
    static_assert(kTokenSize > 0, "token size must be positive");
    size_t stage_size = 0;
    // Token buffer (aligned to 128B for TMA)
    int token_buffer_stage_stride = model.hidden_dim * kTokenSize;
    token_buffer_stage_stride = (token_buffer_stage_stride + 127) & ~127;
    stage_size += token_buffer_stage_stride;
    // Prob buffer (16B aligned per stage)
    if (config.forward_dispatch) {
        int prob_buffer_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
        prob_buffer_stage_stride = (prob_buffer_stage_stride + 15) & ~15;
        stage_size += prob_buffer_stage_stride;
    }
    // Scaling factor buffer (16B aligned per stage, only if quantized)
    if (config.sf_bytes_per_token > 0) {
        int sf_buffer_stage_stride = config.sf_bytes_per_token;
        sf_buffer_stage_stride = (sf_buffer_stage_stride + 15) & ~15;
        stage_size += sf_buffer_stage_stride;
    }
    // Mbarrier buffers: producer + consumer per stage (8B aligned)
    stage_size += 2 * sizeof(uint64_t);
    return stage_size;
}

// Bytes added to the dispatch SMEM layout by a SINGLE pipeline. Each pipeline
// owns S2D_MAP_RING_STAGES ping-pong S2D map slots (128B aligned), 2 S2D map
// mbarriers and 1 S2G group mbarrier, so the pipeline-scaled portion of the
// layout is exactly config.num_pipelines * this value. The three buffers live
// in separate sections of the real layout, but the S2D slot stride is 128B
// aligned and the mbarriers are 8B multiples, so summing them here and scaling
// by the pipeline count reproduces the per-section totals without introducing
// alignment error.
static size_t calc_disp_per_pipeline_smem(const dispatch_config_t& config) {
    size_t pipeline_size = 0;
    // Sparse to dense map buffer: S2D_MAP_RING_STAGES ping-pong stages (128B aligned)
    // Inner dim is mode-dependent: flat = lsa_team_size, expert-major = num_topk.
    int s2d_map_stage_stride = config.num_of_tokens_per_chunk * config.s2d_inner_dim * sizeof(int32_t);
    s2d_map_stage_stride = (s2d_map_stage_stride + 127) & ~127;
    pipeline_size += S2D_MAP_RING_STAGES * s2d_map_stage_stride;
    // s2d_map mbarriers: 2 per pipeline (8B aligned)
    pipeline_size += 2 * sizeof(uint64_t);
    // S2G group mbarrier: 1 per pipeline (8B aligned)
    pipeline_size += sizeof(uint64_t);
    return pipeline_size;
}

// Bytes of the dispatch SMEM layout that depend on neither the pipeline count
// nor the stage count: the shared attn_to_rdma map, the dispatch region info,
// and (expert-major only) the PAD warp TMA slot. attn_to_rdma is rounded to 8B
// to absorb the alignment padding the real layout inserts before the following
// mbarriers, keeping the composed total exact for flat layouts.
template <ncclEpLayout_t kLayout, int kTokenSize>
static size_t calc_disp_fixed_smem(const dispatch_config_t& config, const model_config_t& model) {
    static_assert(kTokenSize > 0, "token size must be positive");
    size_t shared_size = 0;
    // attn_to_rdma_map buffer (only if multi_lsa, shared). Rounded to 8B so the
    // per-pipeline mbarriers that follow it in the real layout stay 8B aligned.
    if (model.num_lsa_teams > 1) {
        int attn_bytes = config.num_of_tokens_per_chunk * (model.num_lsa_teams - 1) * sizeof(bool);
        shared_size += (attn_bytes + 7) & ~7;
    }
    // Dispatch memory region info buffer (only if multi_lsa)
    if (model.num_lsa_teams > 1) {
        shared_size += (model.num_lsa_teams - 1) * sizeof(dispatch_memory_region_info_t);
    }
    // PAD warp TMA slot (expert-major only, 128B aligned)
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        const int token_bytes = model.hidden_dim * kTokenSize;
        int pad_bytes = token_bytes > config.sf_bytes_per_token ? token_bytes : config.sf_bytes_per_token;
        pad_bytes = (pad_bytes + 127) & ~127;
        shared_size = (shared_size + 127) & ~127;
        shared_size += pad_bytes;
    }
    return shared_size;
}

// Bilinear coefficients of the dispatch SMEM layout: total bytes are
//   fixed + num_pipelines * per_pipeline + num_of_stages * per_stage
// rounded up to 128B (see calc_disp_smem). None of the three
// terms depend on the pipeline or stage counts, so the host fitter can compute
// them once and then solve for (pipelines, stages) with plain arithmetic
// instead of re-invoking the full size calculation per candidate.
struct disp_smem_cost_t {
    size_t fixed;         // stage- and pipeline-independent bytes (shared)
    size_t per_pipeline;  // bytes per pipeline
    size_t per_stage;     // bytes per stage
};

// Compose the three terms into a total, matching the layout's alignment exactly.
inline size_t calc_disp_smem(
    const disp_smem_cost_t& terms, int num_pipelines, int num_of_stages) {
    size_t total_size = terms.fixed +
                        static_cast<size_t>(num_pipelines) * terms.per_pipeline +
                        static_cast<size_t>(num_of_stages) * terms.per_stage;
    // Add padding for alignment. (For expert-major, folding the PAD slot's 128B
    // alignment into the shared term can add up to 127B versus the fully
    // interleaved layout; the result stays an exact-or-over upper bound.)
    return (total_size + 127) & ~127;
}

template <ncclEpLayout_t kLayout, int kTokenSize>
static disp_smem_cost_t calc_disp_smem_cost(
    const dispatch_config_t& config, const model_config_t& model) {
    static_assert(kTokenSize > 0, "token size must be positive");
    return disp_smem_cost_t{
        calc_disp_fixed_smem<kLayout, kTokenSize>(config, model),
        calc_disp_per_pipeline_smem(config),
        calc_disp_per_stage_smem<kTokenSize>(config, model),
    };
}

// Host-side companion that maps a runtime layout + token width to the bilinear
// SMEM coefficients. The JIT specializes the device kernel on TOKEN_DATA_TYPE;
// this wrapper uses the same byte width. Returns {0,0,0} for an unsupported
// token size (per_stage == 0 flags the failure).
inline disp_smem_cost_t calc_disp_smem_cost(
    ncclEpLayout_t layout,
    unsigned int token_size,
    const dispatch_config_t& config,
    const model_config_t& model) {
    if (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        switch (token_size) {
            case 1: return calc_disp_smem_cost<NCCL_EP_LAYOUT_EXPERT_MAJOR, 1>(config, model);
            case 2: return calc_disp_smem_cost<NCCL_EP_LAYOUT_EXPERT_MAJOR, 2>(config, model);
            case 4: return calc_disp_smem_cost<NCCL_EP_LAYOUT_EXPERT_MAJOR, 4>(config, model);
            default: return disp_smem_cost_t{0, 0, 0};
        }
    }
    switch (token_size) {
        case 1: return calc_disp_smem_cost<NCCL_EP_LAYOUT_FLAT, 1>(config, model);
        case 2: return calc_disp_smem_cost<NCCL_EP_LAYOUT_FLAT, 2>(config, model);
        case 4: return calc_disp_smem_cost<NCCL_EP_LAYOUT_FLAT, 4>(config, model);
        default: return disp_smem_cost_t{0, 0, 0};
    }
}

// kTokenSize drives the per-stage token-buffer stride; everything else
// (probabilities, mbarriers, scales) is element-width-invariant.
template <ncclDataType_t kTokenDtype,
          ncclEpCombQuant_t kCombineRecipe>
__device__ combine_smem_layout_t create_combine_smem_layout(
    combine_smem_layout_t& layout,
    void* smem_base,
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_chunk,
    bool backward_combine,
    const model_config_t& model) {
    size_t offset = 0;
    const uintptr_t smem_base_addr = reinterpret_cast<uintptr_t>(smem_base);
    auto align_offset = [&](size_t alignment) {
        const size_t mask = alignment - 1;
        const size_t misalignment = (smem_base_addr + offset) & mask;
        if (misalignment != 0) {
            offset += alignment - misalignment;
        }
    };

    // In the single-LSA-team case (num_lsa_teams == 1), the combine kernel does not use the
    // intra-LSA staging buffers. Skipping these buffers can cut SMEM roughly in half.
    const bool multi_lsa = (model.num_lsa_teams > 1);

    // Per-token wire size: bytes for buffer offsets, uint16_t units for stage strides (the
    // token buffer base is uint16_t*). FP32 doubles both vs BF16/FP16.
    // S2G output stays at the caller's output dtype width (2H for BF16).
    const int token_bytes = model.hidden_dim * nccl_ep::size_u8<kTokenDtype>();
    const int token_stride_u16 = model.hidden_dim * nccl_ep::size_u16<kTokenDtype>();
    // G2S stages carry the recipe's packed row: MXFP8 [FP8 H | E8M0 H/32], NONE = dtype
    // width. Scales live in the token stage tail — no separate scale SMEM.
    using Recipe = nccl_ep::combine_recipe_traits<kCombineRecipe>;
    const int g2s_stage_bytes_raw =
        Recipe::template packed_bytes<kTokenDtype>(model.hidden_dim);
    // Match calc_comb_per_g2s_stage_smem: round stage stride up to 128B for TMA alignment
    // (H+H/32 is not always 128B-aligned, e.g. H=7168 → 7392).
    const int g2s_stage_bytes = (g2s_stage_bytes_raw + 127) & ~127;
    const int g2s_stage_stride_u16 = g2s_stage_bytes / (int)sizeof(uint16_t);
    // Both G2S rings carry the same packed row, so they share one stage stride.
    layout.token_G2S_stage_stride = g2s_stage_stride_u16;
    layout.token_S2G_stage_stride = token_stride_u16;
    layout.prob_G2S_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team;
    layout.prob_S2G_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team;
    layout.prob_S2G_inter_stage_stride = layout.prob_S2G_stage_stride * model.num_lsa_teams;

    // lsa_token_* buffers (128B aligned, multi-LSA-team only). Stage stride scales
    // with the on-wire element width (2 B for BF16/FP16, 4 B for FP32).
    if (multi_lsa) {
        align_offset(128);
        layout.lsa_token_G2S_buffer =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * g2s_stage_bytes;

        align_offset(128);
        layout.lsa_token_S2G_buffer =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * token_bytes;
    } else {
        layout.lsa_token_G2S_buffer = nullptr;
        layout.lsa_token_S2G_buffer = nullptr;
    }

    // cross_lsa_token_G2S_buffer (128B aligned). Must advance by the same packed stage stride
    // the accessor indexes with, or stage k>0 would overrun the buffers that follow.
    align_offset(128);
    layout.cross_lsa_token_G2S_buffer = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * g2s_stage_bytes;

    // cross_lsa_token_S2G_buffer (128B aligned)
    align_offset(128);
    layout.cross_lsa_token_S2G_buffer = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_s2g * token_bytes;

    // Prob buffers (only if backward_combine, 16B aligned)
    if (backward_combine) {
        if (multi_lsa) {
            // lsa_prob_G2S_buffer
            align_offset(16);
            layout.lsa_prob_G2S_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_g2s * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);

            // lsa_prob_S2G_buffer
            align_offset(16);
            layout.lsa_prob_S2G_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_s2g * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
        } else {
            layout.lsa_prob_G2S_buffer = nullptr;
            layout.lsa_prob_S2G_buffer = nullptr;
        }

        // cross_lsa_prob_G2S_buffer
        align_offset(16);
        layout.cross_lsa_prob_G2S_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);

        // cross_lsa_prob_S2G_buffer
        align_offset(16);
        layout.cross_lsa_prob_S2G_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * model.num_of_experts_per_rank * model.ranks_per_lsa_team * model.num_lsa_teams *
                  sizeof(float);
    } else {
        layout.lsa_prob_G2S_buffer = nullptr;
        layout.lsa_prob_S2G_buffer = nullptr;
        layout.cross_lsa_prob_G2S_buffer = nullptr;
        layout.cross_lsa_prob_S2G_buffer = nullptr;
    }

    // Mbarrier buffers (8B aligned)
    // lsa_mbarrier_G2S_buffer (multi-LSA-team only)
    if (multi_lsa) {
        align_offset(8);
        layout.lsa_mbarrier_G2S_buffer =
            reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * 2 * sizeof(uint64_t);
    } else {
        layout.lsa_mbarrier_G2S_buffer = nullptr;
    }

    // cross_lsa_mbarrier_G2S_buffer
    align_offset(8);
    layout.cross_lsa_mbarrier_G2S_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * 2 * sizeof(uint64_t);

    // lsa_to_rdma_mbarrier_buffer (only if multi-LSA-team)
    if (model.num_lsa_teams > 1) {
        int max_num_of_chunks_per_rank =
            (model.max_num_of_tokens_per_rank + num_of_tokens_per_chunk - 1) / num_of_tokens_per_chunk;
        align_offset(8);
        layout.lsa_to_rdma_mbarrier_buffer =
            reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_lsa_teams - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
    } else {
        layout.lsa_to_rdma_mbarrier_buffer = nullptr;
    }

    if (model.num_lsa_teams > 1) {
        align_offset(8);
        layout.combine_memory_region_info =
            reinterpret_cast<combine_memory_region_info_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_lsa_teams - 1) * sizeof(combine_memory_region_info_t);
    } else {
        layout.combine_memory_region_info = nullptr;
    }

    // Flag buffers (no special alignment needed)
    if (multi_lsa) {
        layout.lsa_flag_G2S_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * sizeof(bool);
    } else {
        layout.lsa_flag_G2S_buffer = nullptr;
    }

    layout.cross_lsa_flag_G2S_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * sizeof(bool);

    // Streaming overlap fields (multi-LSA-team only, 4B aligned)
    if (multi_lsa) {
        align_offset(4);
        layout.rdma_streaming_counter = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += sizeof(uint32_t);
    } else {
        layout.rdma_streaming_counter = nullptr;
    }

    return layout;
}

// Bytes added to the combine SMEM layout by a SINGLE G2S stage. G2S stages
// carry four buffer families: token, prob (backward combine only), mbarrier and
// flag. For each family the cross_lsa_* buffer is always present and the lsa_*
// buffer is present only for multi-LSA-team. The token stride is rounded to 128B
// (TMA) and the prob stride to 16B: this reproduces the per-buffer start
// alignment exactly when the raw strides already meet it (the case in practice,
// since hidden_dim * elem_width is a multiple of 128) and over-estimates safely
// otherwise. kTokenDtype selects the wire element width (2 B BF16/FP16, 4 B FP32).
// kCombineRecipe: MXFP8 G2S token stages use packed_bytes (FP8+E8M0);
// separate per-stage scale buffers are not allocated.
// For MXFP8, both G2S token buffers hold the packed record; S2G stays output-dtype width.
template <ncclDataType_t kTokenDtype,
          ncclEpCombQuant_t kCombineRecipe>
static size_t calc_comb_per_g2s_stage_smem(
    int num_lsa_teams, const combine_config_t& config, const model_config_t& model) {
    const bool multi_lsa = (num_lsa_teams > 1);
    using Recipe = nccl_ep::combine_recipe_traits<kCombineRecipe>;
    // Both G2S rings carry the recipe's packed row, so one stride sizes both.
    int g2s_stage_bytes = Recipe::template packed_bytes<kTokenDtype>(model.hidden_dim);
    g2s_stage_bytes = (g2s_stage_bytes + 127) & ~127;
    int prob_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
    prob_stride = (prob_stride + 15) & ~15;
    size_t stage_size = 0;
    // Token buffers (128B aligned): cross_lsa always, lsa_* only when multi-team.
    stage_size += g2s_stage_bytes;
    if (multi_lsa) stage_size += g2s_stage_bytes;
    // Prob buffers (16B aligned, backward combine only).
    if (config.backward_combine) {
        stage_size += prob_stride;
        if (multi_lsa) stage_size += prob_stride;
    }
    // Mbarriers: 2 per stage (8B aligned). cross_lsa always, lsa_* when multi-team.
    stage_size += 2 * sizeof(uint64_t);
    if (multi_lsa) stage_size += 2 * sizeof(uint64_t);
    // Flags: 1 byte per stage. cross_lsa always, lsa_* when multi-team.
    stage_size += sizeof(bool);
    if (multi_lsa) stage_size += sizeof(bool);
    return stage_size;
}

// Bytes added to the combine SMEM layout by a SINGLE S2G stage. S2G stages carry
// only token and (backward combine only) prob buffers -- no mbarriers or flags
// scale with the S2G stage count. Note cross_lsa_prob_S2G spans all LSA teams
// (num_lsa_teams x the per-team prob width), unlike the G2S prob buffers. Same
// 128B/16B stride rounding as the G2S helper.
template <ncclDataType_t kTokenDtype = ncclBfloat16>
static size_t calc_comb_per_s2g_stage_smem(
    int num_lsa_teams, const combine_config_t& config, const model_config_t& model) {
    const bool multi_lsa = (num_lsa_teams > 1);
    int token_stride = model.hidden_dim * nccl_ep::size_u8<kTokenDtype>();
    token_stride = (token_stride + 127) & ~127;
    const int prob_bytes = model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
    int prob_stride = (prob_bytes + 15) & ~15;
    int cross_prob_stride = (prob_bytes * num_lsa_teams + 15) & ~15;
    size_t stage_size = 0;
    // Token buffers (128B aligned): cross_lsa always, lsa_* only when multi-team.
    stage_size += token_stride;
    if (multi_lsa) stage_size += token_stride;
    // Prob buffers (16B aligned, backward combine only).
    if (config.backward_combine) {
        stage_size += cross_prob_stride;          // cross_lsa_prob_S2G spans all LSA teams
        if (multi_lsa) stage_size += prob_stride; // lsa_prob_S2G is per-team width
    }
    return stage_size;
}

// Bytes of the combine SMEM layout that scale with neither stage count: the
// LSA->RDMA mbarriers ((teams-1) x chunks), the combine region info ((teams-1)),
// and the RDMA streaming counter. All three exist only for multi-LSA-team.
static size_t calc_comb_fixed_smem(
    int max_num_of_tokens_per_rank, int num_lsa_teams, const combine_config_t& config) {
    if (num_lsa_teams <= 1) return 0;
    const int max_num_of_chunks_per_rank =
        (max_num_of_tokens_per_rank + config.num_of_tokens_per_chunk - 1) / config.num_of_tokens_per_chunk;
    size_t fixed_size = 0;
    // lsa_to_rdma_mbarrier_buffer [(teams-1)][chunks] (8B aligned)
    fixed_size += (num_lsa_teams - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
    // combine_memory_region_info [(teams-1)] (8B aligned)
    fixed_size += (num_lsa_teams - 1) * sizeof(combine_memory_region_info_t);
    // rdma_streaming_counter (4B aligned)
    fixed_size += sizeof(uint32_t);
    return fixed_size;
}

// Bilinear coefficients of the combine SMEM layout: total bytes are
//   fixed + num_of_stages_g2s * per_g2s_stage + num_of_stages_s2g * per_s2g_stage
// rounded up to 128B (see calc_comb_smem). None of the three terms
// depend on either stage count, so the host fitter can compute them once and
// solve for (g2s, s2g) stages directly.
struct comb_smem_cost_t {
    size_t fixed;          // stage-independent bytes
    size_t per_g2s_stage;  // bytes per G2S stage
    size_t per_s2g_stage;  // bytes per S2G stage
};

// Compose the three terms into a total, matching the layout's alignment exactly.
inline size_t calc_comb_smem(
    const comb_smem_cost_t& cost, int num_of_stages_g2s, int num_of_stages_s2g) {
    size_t total_size = cost.fixed +
                        static_cast<size_t>(num_of_stages_g2s) * cost.per_g2s_stage +
                        static_cast<size_t>(num_of_stages_s2g) * cost.per_s2g_stage;
    // Round up to 128B. Matches the per-buffer TMA alignment and absorbs the
    // small (<128B) alignment padding the fully interleaved layout inserts
    // between buffers (including the streaming counter's 4B pad); the result
    // stays an exact-or-over upper bound on the real layout.
    return (total_size + 127) & ~127;
}

template <ncclDataType_t kTokenDtype,
          ncclEpCombQuant_t kCombineRecipe>
static comb_smem_cost_t calc_comb_smem_cost(
    int max_num_of_tokens_per_rank,
    int num_lsa_teams,
    const combine_config_t& config,
    const model_config_t& model) {
    return comb_smem_cost_t{
        calc_comb_fixed_smem(max_num_of_tokens_per_rank, num_lsa_teams, config),
        calc_comb_per_g2s_stage_smem<kTokenDtype, kCombineRecipe>(num_lsa_teams, config, model),
        calc_comb_per_s2g_stage_smem<kTokenDtype>(num_lsa_teams, config, model),
    };
}

// Count-exchange dispatch payload (kernel-side mirror of nccl_ep::ht::DispatchPushCountParams).
// When active, the MAP warps build the rank-major s2d and receiver EM tables inside
// dispatch from the sender-published count rows; false leaves the comm path untouched.
struct dispatch_push_count_kparams_t {
    bool active = false;                                    // gate: run MAP-warp count path
    // Fused path: this rank's own count row, written by the histogram into per-handle scratch
    // and fanned out to the LSA peers at the dispatch head (see head warp 2).
    const int32_t* own_row = nullptr;
    // Unfused path: every sender's count row, AllGathered into a per-handle buffer in
    // UpdateHandle. When set, the MAP warps read it in place of the head fan-out landing.
    const int32_t* cached_cnt_rows = nullptr;               // [nRanks * (nRanks + num_experts)]
    bool fused_meta_dispatch = true;
    int32_t* s2d_out = nullptr;                             // writable s2d [max_tokens, lsa_team_size], pre-set to 0xFF by host
    // Byte offset of the sender-published count rows ([cnt_rank(nRanks) | cnt_expert(num_experts)])
    // inside each peer's count-table region; written by the histogram, read after the head barrier.
    size_t published_offset = 0;
    const int32_t* per_src_lteam_chunk_rank = nullptr;       // [num_chunks, lsa_team_size] per-chunk send counts
    const int64_t* cached_topk_idx = nullptr;               // cached sender topk_idx [max_tokens, num_topk]
    int32_t* num_recv_out = nullptr;                        // FLAT recv count for this rank (num_tokens_for_experts)
    const int32_t* per_src_lteam_num_tokens = nullptr;       // [1] real token count for the sender's stream
    int num_topk = 0;
    int num_src_ranks = 0;                                  // nRanks (== lsa_team_size in count mode)
    int lsa_team_size = 0;
    const uint64_t* token_dst_rank_bitmap = nullptr;        // [max_tokens] per-token global dest-rank bitmap
    uint8_t* const* recv_tables_ptrs = nullptr;             // [lsa_team_size] peer count-table regions
    int recv_max_slots = 0;                                 // slots per sub-table in the peer region
    // When set, the fused meta build clamps published per-expert offsets/counts to
    // recv_max_slots so over-budget tokens drop; the pre-drop total stays in num_recv_out.
    bool allow_overflow_drop = false;
    // Receiver EM tables published by block 0 (zone offsets, unpadded counts, caller outputs).
    int64_t* expert_token_offsets_out = nullptr;            // [experts_per_rank + 1]
    int32_t* per_expert_counts_out = nullptr;               // [experts_per_rank] unpadded counts
    void* caller_offsets = nullptr;
    void* caller_counts = nullptr;
    void* caller_recv_total = nullptr;                      // recv_total_counter scalar (pre-drop padded total), nullable
    bool caller_out_is_int64 = false;
    int em_alignment = 1;
};

// Fixed-size part of dispatch kernel parameters. Peer pointer arrays are appended
// by dispatch_kernel_param_t<..., LSA_TEAM_SZ> for JIT-specialized kernels.
template <typename TOKEN_DATA_TYPE>
struct dispatch_kernel_param_base_t {
    int hidden_dim;
    int experts_per_rank;
    int ranks_per_lsa_team;
    // Input buffers. These buffers are local buffers.
    const TOKEN_DATA_TYPE* attn_input_token;
    const float* attn_input_prob; // Needed by expert layer, so only valid in forward dispatch.
    const uint8_t* attn_input_token_scaling_factor; // QUANT_FWD input bytes.
    // Internal temp buffers. These buffers are local buffers.
    uint64_t* gin_G2S_flags; // For RDMA Atomic flags.
    uint32_t* lsa_S2G_flags; // For intra-LSA S2G write completion notification.
    // Metadata buffers. These buffers are local buffers.
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // flat: lsa_team_size, expert-major: num_topk
    // Expert-major zero-padding inputs for the PAD warp.
    // PAD warp is a no-op when pad_alignment <= 1; pointers may be null then.
    const int32_t* pad_actual_counts; // [experts_per_rank] unpadded token counts
    const int64_t* pad_expert_token_offsets; // [experts_per_rank] zone start offsets
    int pad_alignment; // per-expert zone alignment in tokens (<=1 = no padding)
    // Device-resident expected counters. Initialized at bootstrap; bumped in
    // this kernel's tail so CUDA-graph capture+replay self-sequences.
    uint64_t* expected_gin_flag_val;
    uint32_t* expected_lsa_flag_val;
    int local_rank;
    int lsa_team;
    // The number of token output by attn layer on a rank/GPU.
    int num_of_tokens_per_rank;
    // NCCL GIN context
    ncclDevComm dcomm; // Device communicator
    ncclWindow_t token_window; // Source window handle for token data
    ncclWindow_t prob_window; // Source window handle for probability data
    ncclWindow_t sf_window; // Source window handle for scaling-factor data
    ncclWindow_t dest_window; // Destination window handle
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    // Memory Region info
    struct dispatch_memory_region_info_t mr_info;
    // Grid barrier counter for fused device_sync in dispatch tail (per-rank, not IPC-shared)
    uint32_t* dispatch_grid_barrier_counter;
    // When true (EM layout with local fanout enabled),
    // sender S2G dedups consecutive same-dest entries; secondary slots are filled
    // afterwards by the local_dup kernel.
    bool local_dup_enabled;
    // Cross-round WAR sync-guards: LSA (intra-LSA staging) uses the NCCL LSA barrier; RDMA
    // (cross-LSA-team staging) is hand-rolled. Only the enable flags are needed on the device now.
    bool guard_enabled; // cross-round WAR guard (LSA + RDMA share one enable)
    // Backstop bound for recv slot indices; the scan never publishes slots at or
    // above it (DROP masks the rest), so the S2G assert only fires on corrupted
    // or stale routing maps.
    int max_recv_tokens_per_rank;
    // Count-exchange payload (fused-meta-dispatch). See dispatch_push_count_kparams_t.
    dispatch_push_count_kparams_t dispatch_push_count;
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    dispatch_warp_timing_entry_t* warp_timing;
#endif
};

// Data structure for JIT dispatch kernel parameters.
template <typename TOKEN_DATA_TYPE, int LSA_TEAM_SZ>
struct dispatch_kernel_param_t : dispatch_kernel_param_base_t<TOKEN_DATA_TYPE> {
    // Output buffers. These buffers are both local and remote buffers.
    // Keep embedded arrays here to avoid device-side pointer-table indirection.
    TOKEN_DATA_TYPE* expert_output_token[LSA_TEAM_SZ];
    float* expert_output_prob[LSA_TEAM_SZ]; // Only valid in forward dispatch.
    uint8_t* expert_output_scaling_factor[LSA_TEAM_SZ]; // Only valid for QUANT_FWD.
};

// Fixed-size part of combine kernel parameters. Peer pointer arrays are appended
// by combine_kernel_param_t<LSA_TEAM_SZ> for JIT-specialized kernels.
struct combine_kernel_param_base_t {
    int hidden_dim;
    int experts_per_rank;
    int ranks_per_lsa_team;
    // EM unfused-combine: when true, the cross-LSA-team G2S warp group skips local-dup
    // secondary em_slots (primaries already carry the pre-reduced weighted sum
    // written by the local_reduce kernel). Default false => fused fanout.
    bool combine_local_reduce_enabled;
    // Output buffers. These buffers are local buffers.
    uint16_t* attn_output_token;
    float* attn_output_prob;
    // Internal temp buffers. These buffers are local buffers.
    uint16_t* combine_gin_RED_tokens;
    float* combine_gin_RED_prob;
    const uint16_t* combine_gin_G2S_tokens;
    const float* combine_gin_G2S_prob;
    uint64_t* gin_G2S_flags;
    uint32_t* lsa_S2G_flags; // For intra-LSA src ready notification.
    // Metadata buffers. These buffers are local buffers.
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // flat: lsa_team_size, expert-major: num_topk
    // Device-resident expected counters. Initialized at bootstrap; bumped in
    // this kernel's tail so CUDA-graph capture+replay self-sequences.
    uint64_t* expected_gin_flag_val;
    uint32_t* expected_lsa_flag_val;
    int local_rank;
    int lsa_team;
    // Stride for routing-map indexing (= max_tokens_per_rank).
    int num_of_tokens_per_rank;
    // Actual token count; gates the cross_lsa_red TMA store.
    int num_real_tokens;
    // Per-rank grid-barrier counter that elects the last block at the combine tail.
    uint32_t* combine_grid_barrier_counter;
    // NCCL GIN context
    ncclDevComm_t* dcomms; // Device communicators array (1 element, on device)
    ncclWindow_t token_window; // Source window handle for token data
    ncclWindow_t prob_window; // Source window handle for probability data
    ncclWindow_t dest_window; // Destination window handle
    int num_gin_comms; // Number of GIN communicators (1)
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    unsigned signals_base; // Base signal ID
    unsigned combine_signal_offset; // Signal offset for combine operations
    // qp info and mr info
    struct combine_memory_region_info_t mr_info;
    // Cross-round WAR sync-guards: LSA (intra-LSA staging) uses the NCCL LSA barrier; RDMA
    // (cross-LSA-team staging) is hand-rolled. Only the enable flags are needed on the device now.
    bool guard_enabled; // cross-round WAR guard (LSA + RDMA share one enable)
    uint32_t combine_barrier_offset; // Reserved LSA barrier session index, not a byte offset
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    combine_warp_timing_entry_t* warp_timing;
    combine_block_timing_entry_t* block_timing;
#endif
};

// Data structure for JIT combine kernel parameters.
template <int LSA_TEAM_SZ>
struct combine_kernel_param_t : combine_kernel_param_base_t {
    // Input buffers. These buffers are both local and remote buffers.
    // Keep embedded arrays here to avoid device-side pointer-table indirection.
    uint16_t* expert_input_token[LSA_TEAM_SZ];
    float* expert_input_prob[LSA_TEAM_SZ];
};

// Each CUDA block has sixteen named barriers numbered 0..15.
// __syncthreads(); will use the 0 named barriers, so we want to avoid that.
// We want to use 1 for intra-LSA reduction warp group, >= 2 for cross-LSA-team reduction warp group,
// RDMA warp group currently only contains 1 warp so does not use named bar yet, if it need to use, it should use 2 + NUM_OF_DATA_PIPELINE_PER_BLOCK.
__forceinline__ __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id = 0) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

// Map a data channel onto its GIN context, skipping the reserved contexts.
__forceinline__ __device__ int get_data_ctx(int channel, int num_ctx_per_comm) {
    return NCCL_EP_HT_RESERVED_GIN_GPU_CTXS + (channel % num_ctx_per_comm);
}

// Helper to compute communicator index and context index from global channel
// Used for 6-comm x 4-ctx GIN configuration (6 communicators with 4 contexts each = 24 total channels)
__forceinline__ __device__ void get_comm_ctx(int global_channel, int num_ctx_per_comm, int& comm_idx, int& ctx_idx) {
    comm_idx = global_channel / num_ctx_per_comm;
    ctx_idx = get_data_ctx(global_channel, num_ctx_per_comm);
}

// Advance a ring-buffer slot; on wrap (slot == num_slots) reset to 0 and flip phase parity.
// Shared by producer/consumer FIFO traversals.
template <typename SlotT>
__forceinline__ __device__ void ring_advance(SlotT& slot, uint32_t& parity, int num_slots) {
    if (++slot == static_cast<SlotT>(num_slots)) {
        slot = 0;
        parity ^= 1;
    }
}

// Put one token's bundle (token, +prob if FWD, +sf if quantized) to a remote LSA team, packed from dst_offset.
template <typename TOKEN_DATA_TYPE, bool FORWARD_DISPATCH, bool HAS_SF, int LSA_TEAMS>
__forceinline__ __device__ void dispatch_n2n_put_token(
    ncclGin& net,
    const ncclTeam& rail,
    int remote_lteam_id,
    ncclWindow_t internal_window,
    size_t dst_offset,
    const struct dispatch_memory_region_info_t* mr_info,
    ncclWindow_t token_window,
    ncclWindow_t prob_window,
    ncclWindow_t sf_window,
    int token_idx,
    size_t token_bytes,
    size_t prob_bytes,
    int sf_bytes_per_token) {
    size_t token_src = mr_info->attn_input_token_offset + token_idx * token_bytes;
    net.put(
        rail,
        remote_lteam_id,
        internal_window,
        dst_offset,
        token_window,
        token_src,
        token_bytes,
        ncclGin_None{},
        ncclGin_None{},
        ncclCoopThread(),
        ncclGin_None{},
        cuda::thread_scope_thread,
        cuda::thread_scope_device,
        ncclGinOptFlagsAggregateRequests);
    if constexpr (FORWARD_DISPATCH) {
        size_t prob_src = mr_info->attn_input_prob_offset + (token_idx * LSA_TEAMS + remote_lteam_id) * prob_bytes;
        net.put(
            rail,
            remote_lteam_id,
            internal_window,
            dst_offset + token_bytes,
            prob_window,
            prob_src,
            prob_bytes,
            ncclGin_None{},
            ncclGin_None{},
            ncclCoopThread(),
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
    }
    if constexpr (HAS_SF) {
        size_t sf_src = mr_info->attn_input_scaling_factor_offset + token_idx * sf_bytes_per_token;
        net.put(
            rail,
            remote_lteam_id,
            internal_window,
            dst_offset + token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0),
            sf_window,
            sf_src,
            sf_bytes_per_token,
            ncclGin_None{},
            ncclGin_None{},
            ncclCoopThread(),
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
    }
}

// Resolved G2S source for one (LSA team, chunk): packed-remote entry base, or strided-local base pointers.
template <typename TOKEN_DATA_TYPE>
struct g2s_source_t {
    bool use_packed;
    const uint8_t* packed_base;
    const TOKEN_DATA_TYPE* token_base;
    const float* prob_base;
    const uint8_t* sf_base;
};

// Resolve a (LSA team, chunk) source: packed-remote base (after waiting the RDMA arrival signal) or strided-local bases.
template <
    typename TOKEN_DATA_TYPE,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int MAX_TOKENS_PER_RANK,
    int TOKENS_PER_CHUNK,
    int NBLOCKS,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ g2s_source_t<TOKEN_DATA_TYPE> dispatch_g2s_resolve_source(
    const TOKEN_DATA_TYPE* attn_input_token,
    const float* attn_input_prob,
    const uint8_t* attn_input_token_scaling_factor,
    const int lteam_id,
    const int my_lteam,
    const int local_rank,
    const int cidx,
    const uint64_t expected_flag_value,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    const struct dispatch_memory_region_info_t* mr_info) {
    g2s_source_t<TOKEN_DATA_TYPE> src;
    src.use_packed = false;
    src.packed_base = nullptr;
    src.token_base = nullptr;
    src.prob_base = nullptr;
    src.sf_base = nullptr;

    if (lteam_id != my_lteam) {
        // Remote: wait for the RDMA arrival signal, then point at this tile+chunk in the packed buffer.
        constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
        unsigned tail_signal_id = dispatch_tail_signal_id(
            mr_info->signals_tail_base,
            lteam_id,
            my_lteam,
            local_rank,
            cidx,
            LSA_TEAMS,
            LSA_TEAM_SZ,
            MAX_CHUNKS_PER_RANK);
        constexpr int N2N_WARPS = (LSA_TEAMS == 1) ? 1 : NCCL_EP_HT_DISPATCH_N2N_WARPS;
        int signal_channel = cidx % (NBLOCKS * N2N_WARPS);

        int ctx_idx = get_data_ctx(signal_channel, num_ctx_per_comm);
        ncclGin net(dcomm, ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
        net.waitSignal(ncclCoopThread(), tail_signal_id, expected_flag_value);

        const int remote_slot = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
        const int chunk_first_token = cidx * TOKENS_PER_CHUNK;

        src.use_packed = true;
        src.packed_base = reinterpret_cast<const uint8_t*>(gin_base_ptr) +
                          dispatch_packed_entry_offset(mr_info, remote_slot, chunk_first_token);
    } else {
        // Local: strided bases into this rank's own global mem arrays for this chunk.
        int chunk_first_token = cidx * TOKENS_PER_CHUNK;
        src.token_base = attn_input_token + chunk_first_token * HIDDEN_DIM;
        if constexpr (FORWARD_DISPATCH) {
            // attn_input_prob is laid out per token as LSA_TEAMS blocks of experts_per_lsa_team floats.
            const int experts_per_lsa_team = experts_per_rank * LSA_TEAM_SZ;
            const int prob_row_stride = experts_per_lsa_team * LSA_TEAMS;
            src.prob_base = attn_input_prob + chunk_first_token * prob_row_stride;
        }
        if constexpr (HAS_SF) {
            src.sf_base = attn_input_token_scaling_factor + chunk_first_token * sf_bytes_per_token;
        }
    }
    return src;
}

enum class copy_dir {
    to_smem,
    to_gmem
}; // load into SMEM / store to gmem

// One field copy: to_smem loads (gmem->smem, mbar form), to_gmem stores (smem->gmem). Returns bytes copied.
template <copy_dir DIR>
__forceinline__ __device__ uint32_t bulk_copy(void* smem_ptr, const void* gmem_ptr, uint32_t bytes, uint64_t* mbar) {
    if constexpr (DIR == copy_dir::to_smem) {
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            /*dst=*/smem_ptr,
            /*src=*/gmem_ptr,
            bytes,
            mbar);
    } else {
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_global,
            cuda::ptx::space_shared,
            /*dst=*/const_cast<void*>(gmem_ptr),
            /*src=*/smem_ptr,
            bytes);
    }
    return bytes;
}

// Copy a token bundle (token, +prob if FWD, +sf if quantized) between this stage's SMEM and gmem. Returns total bytes.
template <typename TOKEN_DATA_TYPE, typename SMEM_TYPE, bool FORWARD_DISPATCH, bool HAS_SF, copy_dir DIR>
__forceinline__ __device__ uint32_t copy_token_bundle(
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const int stage,
    const void* token_gmem_ptr,
    const void* prob_gmem_ptr,
    const void* sf_gmem_ptr,
    const uint32_t token_bytes,
    const uint32_t prob_bytes,
    const uint32_t sf_bytes,
    uint64_t* mbar) {
    uint32_t tx =
        bulk_copy<DIR>(smem_buffer_ptr->get_token_buffer(pipeline_rank, stage), token_gmem_ptr, token_bytes, mbar);
    if constexpr (FORWARD_DISPATCH) {
        tx += bulk_copy<DIR>(
            reinterpret_cast<void*>(smem_buffer_ptr->get_prob_buffer(pipeline_rank, stage)),
            prob_gmem_ptr,
            prob_bytes,
            mbar);
    }
    if constexpr (HAS_SF) {
        tx += bulk_copy<DIR>(
            reinterpret_cast<void*>(smem_buffer_ptr->get_sf_buffer(pipeline_rank, stage)),
            sf_gmem_ptr,
            sf_bytes,
            mbar);
    }
    return tx;
}

// TMA-copy one token (+prob if FWD, +sf if quantized) from a packed-remote or strided-local source into its SMEM stage, then publish.
template <
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ void dispatch_g2s_issue_token(
    const g2s_source_t<TOKEN_DATA_TYPE>& src,
    const int cur_tokid,
    const int packed_dense_idx,
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const int stage,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const int my_lteam,
    const struct dispatch_memory_region_info_t* mr_info) {
    uint64_t* mbar = smem_buffer_ptr->get_lsa_mbarrier_producer(pipeline_rank, stage);
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SZ) * sizeof(float));
    const uint32_t sf_bytes = (uint32_t)sf_bytes_per_token;
    uint32_t tx_bytes;

    if (src.use_packed) {
        // Packed entry is contiguous [token | prob | sf].
        const uint8_t* packed_src_base = src.packed_base + packed_dense_idx * mr_info->bytes_per_entry;
        const void* token_src = packed_src_base;
        const void* prob_src = packed_src_base + token_bytes;
        const void* sf_src = packed_src_base + token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0);
        tx_bytes = copy_token_bundle<TOKEN_DATA_TYPE, SMEM_TYPE, FORWARD_DISPATCH, HAS_SF, copy_dir::to_smem>(
            smem_buffer_ptr,
            pipeline_rank,
            stage,
            token_src,
            prob_src,
            sf_src,
            token_bytes,
            prob_bytes,
            sf_bytes,
            mbar);
    } else {
        // Strided-local: each field has its own base.
        const void* token_src = src.token_base + (cur_tokid * HIDDEN_DIM);
        const void* prob_src = nullptr;
        const void* sf_src = nullptr;
        if constexpr (FORWARD_DISPATCH) {
            // Advance by whole token rows, then pick this LSA team's expert slice within the row.
            const int experts_per_lsa_team = experts_per_rank * LSA_TEAM_SZ;
            const int prob_row_stride = experts_per_lsa_team * LSA_TEAMS;
            prob_src = src.prob_base + cur_tokid * prob_row_stride + my_lteam * experts_per_lsa_team;
        }
        if constexpr (HAS_SF) {
            sf_src = src.sf_base + cur_tokid * sf_bytes_per_token;
        }
        tx_bytes = copy_token_bundle<TOKEN_DATA_TYPE, SMEM_TYPE, FORWARD_DISPATCH, HAS_SF, copy_dir::to_smem>(
            smem_buffer_ptr,
            pipeline_rank,
            stage,
            token_src,
            prob_src,
            sf_src,
            token_bytes,
            prob_bytes,
            sf_bytes,
            mbar);
    }

    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release,
        cuda::ptx::scope_cta,
        cuda::ptx::space_shared,
        mbar,
        tx_bytes);
}

// One destination decoded from a single s2d-map entry.
struct s2g_dest_t {
    bool issue; // false for an empty entry or an EM secondary duplicate
    int remote_rank_id;
    int output_buffer_index;
};

// TMA-load one (LSA team, chunk)'s s2d-map slice into the SMEM stage and publish. Caller elects one lane.
template <typename SMEM_TYPE, int NUM_OF_TOKENS_PER_CHUNK>
__forceinline__ __device__ void dispatch_s2g_prefetch_s2d_map(
    const int32_t* sparse_to_dense_map,
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const uint32_t s2d_map_stage,
    const int lsa_team_id,
    const int chunk_id,
    const int chunk_size,
    const int num_of_tokens_per_rank,
    const int s2d_inner_dim) {
    const int32_t* s2d_base =
        sparse_to_dense_map + (lsa_team_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * s2d_inner_dim;
    void* smem_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_s2d_map_buffer_base(pipeline_rank, s2d_map_stage));
    uint64_t* mbar = smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, s2d_map_stage);
    // cp.async.bulk needs a 16B-multiple size: round up. Safe because the source S2D buffer
    // is over-allocated for max_tokens_per_rank and the smem dest stage is padded to 128B.
    uint32_t copy_bytes = (uint32_t)(chunk_size * s2d_inner_dim * sizeof(int32_t));
    copy_bytes = (copy_bytes + 15u) & ~15u;
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        smem_dst,
        reinterpret_cast<const void*>(s2d_base),
        copy_bytes,
        mbar);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release,
        cuda::ptx::scope_cta,
        cuda::ptx::space_shared,
        mbar,
        copy_bytes);
}

// Decode one s2d-map entry into its remote destination. `issue` is false for an empty entry (-1)
// or an EM secondary duplicate (the receiver's local_dup kernel fills it from the primary slot).
template <ncclEpLayout_t kLayout>
__forceinline__ __device__ s2g_dest_t
dispatch_s2g_resolve_dest(const int32_t* s2d_row, const int flat_idx, const bool local_dup_enabled) {
    s2g_dest_t dst;
    dst.issue = false;
    dst.remote_rank_id = -1;
    dst.output_buffer_index = -1;

    const int32_t entry_val = s2d_row[flat_idx];
    if (entry_val == -1) {
        return dst;
    }

    // Rank-major: entry_idx=rank, value=slot. Expert-major: value packs (rank,slot).
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        dst.remote_rank_id = em_s2d_unpack_rank(entry_val);
        dst.output_buffer_index = em_s2d_unpack_slot(entry_val);
        if (local_dup_enabled && flat_idx > 0) {
            const int32_t prev_val = s2d_row[flat_idx - 1];
            if (prev_val != -1 && em_s2d_unpack_rank(prev_val) == dst.remote_rank_id) {
                return dst; // secondary dup: issue stays false
            }
        }
    } else {
        dst.remote_rank_id = flat_idx;
        dst.output_buffer_index = entry_val;
    }
    dst.issue = true;
    return dst;
}

// TMA-store one token (+prob if FWD, +sf if quantized) from this stage's SMEM to one resolved remote destination.
template <typename TOKEN_DATA_TYPE, typename SMEM_TYPE, int LSA_TEAM_SIZE, bool FORWARD_DISPATCH, bool HAS_SF>
__forceinline__ __device__ void dispatch_s2g_issue_token(
    const s2g_dest_t& dst,
    SMEM_TYPE* smem_buffer_ptr,
    TOKEN_DATA_TYPE* const* remote_expert_output_token,
    float* const* remote_expert_output_prob,
    uint8_t* const* remote_expert_output_scaling_factor,
    const int pipeline_rank,
    const int stage,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int max_recv_tokens_per_rank,
    const int experts_per_rank) {
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SIZE) * sizeof(float));
    const uint32_t sf_bytes = (uint32_t)sf_bytes_per_token;

    // Remote fan-out: each field has its own output array, indexed [rank]+slot*stride.
    // The scan never publishes slots at or above the bound, so this catches
    // corrupted or stale routing maps before they overrun a recv buffer.
    EP_DEVICE_ASSERT(dst.output_buffer_index < max_recv_tokens_per_rank);

    // Cast slot to size_t to prevent int32 overflow in the offset multiplication.
    const size_t slot = static_cast<size_t>(dst.output_buffer_index);
    const void* token_dst = remote_expert_output_token[dst.remote_rank_id] + (slot * HIDDEN_DIM);
    const void* prob_dst = nullptr;
    const void* sf_dst = nullptr;
    if constexpr (FORWARD_DISPATCH) {
        prob_dst = remote_expert_output_prob[dst.remote_rank_id] +
                   (slot * (experts_per_rank * LSA_TEAM_SIZE));
    }
    if constexpr (HAS_SF) {
        sf_dst = remote_expert_output_scaling_factor[dst.remote_rank_id] + slot * sf_bytes_per_token;
    }
    copy_token_bundle<TOKEN_DATA_TYPE, SMEM_TYPE, FORWARD_DISPATCH, HAS_SF, copy_dir::to_gmem>(
        smem_buffer_ptr,
        pipeline_rank,
        stage,
        token_dst,
        prob_dst,
        sf_dst,
        token_bytes,
        prob_bytes,
        sf_bytes,
        /*mbar=*/nullptr);
}

// Device function for cross-LSA-team node2node(RDMA) warp for dispatch kernel.
template <
    typename GIN_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_STAGES,
    int TOKENS_PER_CHUNK,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int NBLOCKS,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ void dispatch_N2N_warp(
    // INPUT
    const bool* attn_to_rdma_map,
    // CONFIG
    const int local_rank,
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    ncclWindow_t nccl_token_window,
    ncclWindow_t nccl_prob_window,
    ncclWindow_t nccl_sf_window,
    ncclWindow_t nccl_internal_window,
    const struct dispatch_memory_region_info_t* mr_info,
    SMEM_TYPE* smem_buffer_ptr) {
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);

    static_assert(GIN_GROUP::size() >= LSA_TEAMS - 1, "mr_info should be loaded at once.");
    static_assert(TOKENS_PER_CHUNK % 32 == 0, "TOKENS_PER_CHUNK must be multiple of 32.");
    static_assert(
        MAX_TOKENS_PER_RANK % TOKENS_PER_CHUNK == 0,
        "MAX_TOKENS_PER_RANK must be multiple of TOKENS_PER_CHUNK.");

    // Load mr_info into shared memory for faster access in Put calls.
    int lane_id = GIN_GROUP::thread_rank() % 32;
    struct dispatch_memory_region_info_t* smem_mr_info_ptr = nullptr;
    if constexpr (LSA_TEAMS != 1) {
        smem_mr_info_ptr = smem_buffer_ptr->dispatch_memory_region_info;
        if (lane_id == 0) {
            smem_mr_info_ptr[0] = mr_info[0];
        }
        __syncwarp();
    }

    constexpr int N2N_WARPS = GIN_GROUP::size() / 32;
    int n2n_warp_id = GIN_GROUP::thread_rank() / 32;
    size_t token_bytes = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
    size_t prob_bytes = (experts_per_rank * LSA_TEAM_SZ) * sizeof(float);
    constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    constexpr int NUM_REMOTE_LSA = LSA_TEAMS - 1;

    // GIN device side setup. Single communicator; ctx_idx spreads QP traffic.
    int global_channel = blockIdx.x * N2N_WARPS + n2n_warp_id;
    int ctx_idx = get_data_ctx(global_channel, num_ctx_per_comm);

    ncclGin net(dcomm, ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam rail = ncclTeamRail(dcomm);

    for (int cidx = blockIdx.x * N2N_WARPS + n2n_warp_id; cidx < MAX_CHUNKS_PER_RANK;
         cidx += NBLOCKS * N2N_WARPS) {
        int chunk_first_token_idx = cidx * TOKENS_PER_CHUNK;
        int csize = 0;
        if (cidx < num_of_chunks_per_rank) {
            csize = TOKENS_PER_CHUNK;
            if (chunk_first_token_idx + csize > num_of_tokens_per_rank) {
                csize = num_of_tokens_per_rank - chunk_first_token_idx;
            }
        }

        for (int j = 0; j < NUM_REMOTE_LSA; ++j) {
            // Skip-self ring over the other LSA teams, load-balanced start at my_lteam.
            int remote_idx = (j + my_lteam) % NUM_REMOTE_LSA;
            int remote_lteam_id = remote_idx < my_lteam ? remote_idx : remote_idx + 1;
            int remote_slot = remote_idx < my_lteam ? my_lteam - 1 : my_lteam;

            size_t dense_dst_offset =
                dispatch_packed_entry_offset(smem_mr_info_ptr, remote_slot, chunk_first_token_idx);
            const size_t entry_bytes = smem_mr_info_ptr->bytes_per_entry;

            // Create a bitmask of tokens that need to be written. One word per 32 tokens.
            int32_t need_write_bitmask[TOKENS_PER_CHUNK / 32];
            for (int i = 0; i < TOKENS_PER_CHUNK / 32; i++) {
                int token_idx_in_chunk = i * 32 + ncclCoopWarp().thread_rank();
                bool need_write =
                    token_idx_in_chunk < csize &&
                    attn_to_rdma_map[((token_idx_in_chunk + chunk_first_token_idx) * NUM_REMOTE_LSA) + remote_idx];
                need_write_bitmask[i] = __ballot_sync(~0u, need_write);
            }

            for (int token_idx_in_chunk = ncclCoopWarp().thread_rank(); token_idx_in_chunk < csize;
                 token_idx_in_chunk += ncclCoopWarp().size()) {
                const int word = token_idx_in_chunk / 32;
                const int lane_in_word = token_idx_in_chunk % 32;
                size_t dst_offset = dense_dst_offset;

                // Compact: skip space for earlier tokens in this word that aren't written.
                if (lane_in_word > 0) {
                    uint32_t writes_before = need_write_bitmask[word] & ((1u << lane_in_word) - 1);
                    dst_offset += __popc(writes_before) * entry_bytes;
                }

                bool need_write =
                    attn_to_rdma_map[((chunk_first_token_idx + token_idx_in_chunk) * NUM_REMOTE_LSA) + remote_idx];
                if (need_write) {
                    int token_idx = chunk_first_token_idx + token_idx_in_chunk;
                    dispatch_n2n_put_token<TOKEN_DATA_TYPE, FORWARD_DISPATCH, HAS_SF, LSA_TEAMS>(
                        net,
                        rail,
                        remote_lteam_id,
                        nccl_internal_window,
                        dst_offset,
                        smem_mr_info_ptr,
                        nccl_token_window,
                        nccl_prob_window,
                        nccl_sf_window,
                        token_idx,
                        token_bytes,
                        prob_bytes,
                        sf_bytes_per_token);
                }
                // Advance the compacted base past all written tokens in this word.
                dense_dst_offset += __popc(need_write_bitmask[word]) * entry_bytes;
            }

            // Signal chunk completion on the SAME put comm: same-QP ordering makes all
            // preceding puts visible at the remote before this signal arrives.
            unsigned tail_signal_id = dispatch_tail_signal_id(
                smem_mr_info_ptr->signals_tail_base,
                my_lteam,
                remote_lteam_id,
                local_rank,
                cidx,
                LSA_TEAMS,
                LSA_TEAM_SZ,
                MAX_CHUNKS_PER_RANK);
            net.signal(
                rail,
                remote_lteam_id,
                ncclGin_SignalAdd{tail_signal_id, 1},
                ncclCoopWarp(),
                ncclGin_None{},
                cuda::thread_scope_thread,
                cuda::thread_scope_thread,
                ncclGinOptFlagsDefault);
        }
    }
    // GIN flush with coopWarp includes syncwarp at the end
    net.flush(ncclCoopWarp(), cuda::memory_order_acquire);
}

// Count-mode: wait for MAP warp (unit % NUM_MAP_WARPS)'s (unit / NUM_MAP_WARPS)-th
// publish (must match dispatch_push_map_warp's split). Shared by S2G (s2d prefetch) and
// G2S (rdma_to_attn_map read, which build_s2d drop-clears while building the same unit) --
// both must wait on the same counter or one can read a value the other already sees cleared.
template <int NUM_MAP_WARPS>
__forceinline__ __device__ void s2d_unit_wait(const int* s2d_chunks_ready, int u) {
    // Relaxed spin, then one CTA-acquire fence once the count passes, pairing with the
    // MAP warp's st_release_cta so the built s2d rows are visible without a per-iteration acquire.
    while (nccl_ep::ld_relaxed_cta(s2d_chunks_ready + u % NUM_MAP_WARPS) <= u / NUM_MAP_WARPS) {
    }
    cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_block);
    nccl_ep::fence_proxy_async();
}

// Dispatch intra-LSA S2G warp group. With NUM_PIPELINES > 1, each warp is an
// independent pipeline consumer paired with the G2S warp of the same pipeline_rank.
template <
    typename LSA_S2G_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_STAGES,
    int IN_FLIGHT_S2G,
    int TOKENS_PER_CHUNK,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int NBLOCKS,
    int NUM_PIPELINES,
    bool FORWARD_DISPATCH,
    bool HAS_SF,
    ncclEpLayout_t kLayout,
    int NUM_MAP_WARPS,
    bool COUNT_MODE>
__forceinline__ __device__ void dispatch_S2G_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const int32_t* sparse_to_dense_map,
    // OUTPUT
    TOKEN_DATA_TYPE* const* remote_expert_output_token,
    float* const* remote_expert_output_prob,
    uint8_t* const* remote_expert_output_scaling_factor,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const bool local_dup_enabled,
    const int max_recv_tokens_per_rank,
    SMEM_TYPE* smem_buffer_ptr,
    // Count-mode: MAP-warp unit-completion counters (smem, one per MAP warp); gate each
    // chunk's s2d prefetch. Unit ordinal u is built by warp u % NUM_MAP_WARPS as its
    // (u / NUM_MAP_WARPS)-th publish (must match dispatch_push_map_warp's split).
    const int* s2d_chunks_ready = nullptr) {
    constexpr int STAGES_PER_PIPELINE = NUM_STAGES / NUM_PIPELINES;
    static_assert(
        IN_FLIGHT_S2G < STAGES_PER_PIPELINE,
        "IN_FLIGHT_S2G must be smaller than STAGES_PER_PIPELINE.");
    using routing_loads_t = uint4;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(
        TOKENS_PER_CHUNK % sizeof(routing_loads_t) == 0,
        "TOKENS_PER_CHUNK must be multiple of routing_loads_t.");
    constexpr int TOKENS_PER_ROUTING_LOAD = sizeof(routing_loads_t) / sizeof(bool);
    constexpr int ROUTING_LOADS_PER_CHUNK = TOKENS_PER_CHUNK / TOKENS_PER_ROUTING_LOAD;

    // S2D inner dim: mode-dependent, carried by SMEM layout struct.
    const int s2d_inner_dim = smem_buffer_ptr->s2d_inner_dim;

    const int pipeline_rank = LSA_S2G_GROUP::warp_rank();
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    // TOKENS_PER_ROUTING_LOAD must match the producer's pad in scan_kernel.cuh
    const int routing_map_lsa_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
    int in_flight_s2g = 0;
    int stage = 0;
    uint32_t producer_parity = 0;
    uint32_t s2d_stage = 0;
    uint32_t s2d_parity = 0;

    // S2G on all 32 lanes (warp-uniform state); cp_async_bulk striped by lane=flat_idx (up to s2d_inner_dim stores/token).
    const int s2g_lane = LSA_S2G_GROUP::thread_rank() % 32;

    // Scan mode prefetches each pipeline's first s2d map into SMEM (single TMA load, lane 0 only).
    // Count mode reads the MAP-warp-built s2d rows directly from GMEM (gated on s2d_chunks_ready),
    // so it skips the TMA staging entirely.
    if constexpr (!COUNT_MODE) if (s2g_lane == 0) {
        int chunk_iter = 0;
        for (int chunk_idx = blockIdx.x; chunk_idx < num_of_chunks_per_rank; chunk_idx += NBLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) == pipeline_rank) {
                int current_chunk_size;
                if (rem_chunk_sz != 0 && chunk_idx == num_of_chunks_per_rank - 1) {
                    current_chunk_size = rem_chunk_sz;
                } else {
                    current_chunk_size = TOKENS_PER_CHUNK;
                }
                dispatch_s2g_prefetch_s2d_map<SMEM_TYPE, TOKENS_PER_CHUNK>(
                    sparse_to_dense_map,
                    smem_buffer_ptr,
                    pipeline_rank,
                    s2d_stage,
                    my_lteam,
                    chunk_idx,
                    current_chunk_size,
                    num_of_tokens_per_rank,
                    s2d_inner_dim);
                break;
            }
        }
    }
    __syncwarp();

    {
        int chunk_iter = 0;
        for (int cidx = blockIdx.x; cidx < num_of_chunks_per_rank; cidx += NBLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) continue;

            int routing_loads_in_chunk;
            int csize;
            if (rem_chunk_sz != 0 && cidx == num_of_chunks_per_rank - 1) {
                routing_loads_in_chunk = nccl_ep::ceil_div(rem_chunk_sz, (int)sizeof(routing_loads_t));
                csize = rem_chunk_sz;
            } else {
                routing_loads_in_chunk = ROUTING_LOADS_PER_CHUNK;
                csize = TOKENS_PER_CHUNK;
            }
            for (int j = 0; j < LSA_TEAMS; j++) {
                // Per-pipeline self-sync (arrival count = 1, trivially satisfied); lane 0 only.
                if (s2g_lane == 0) {
                    uint64_t state_token =
                        cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_S2G_group_mbar(pipeline_rank));
                    while (!cuda::ptx::mbarrier_try_wait(
                        smem_buffer_ptr->get_S2G_group_mbar(pipeline_rank),
                        state_token)) {
                    }
                }
                __syncwarp();

                // Prefetch next (chunk, LSA team) s2d map for THIS pipeline (single TMA load, lane 0 only).
                // Scan mode only; count mode reads GMEM and gates on s2d_chunks_ready below.
                if constexpr (!COUNT_MODE) if (s2g_lane == 0) {
                    int next_chunk_id;
                    int next_lsa_id;
                    int next_lsa_iter = j + 1;
                    if (next_lsa_iter < LSA_TEAMS) {
                        next_chunk_id = cidx;
                        next_lsa_id = (my_lteam + LSA_TEAMS - next_lsa_iter) % LSA_TEAMS;
                    } else {
                        // Find the next chunk this pipeline will process
                        int future_chunk_iter = chunk_iter; // chunk_iter was already incremented for current chunk
                        next_chunk_id = -1;
                        for (int fi = cidx + NBLOCKS; fi < num_of_chunks_per_rank; fi += NBLOCKS) {
                            if ((future_chunk_iter++ % NUM_PIPELINES) == pipeline_rank) {
                                next_chunk_id = fi;
                                break;
                            }
                        }
                        next_lsa_id = my_lteam;
                    }

                    if (next_chunk_id >= 0 && next_chunk_id < num_of_chunks_per_rank) {
                        int next_chunk_size;
                        if (rem_chunk_sz != 0 && next_chunk_id == num_of_chunks_per_rank - 1) {
                            next_chunk_size = rem_chunk_sz;
                        } else {
                            next_chunk_size = TOKENS_PER_CHUNK;
                        }
                        dispatch_s2g_prefetch_s2d_map<SMEM_TYPE, TOKENS_PER_CHUNK>(
                            sparse_to_dense_map,
                            smem_buffer_ptr,
                            pipeline_rank,
                            s2d_stage ^ 1,
                            next_lsa_id,
                            next_chunk_id,
                            next_chunk_size,
                            num_of_tokens_per_rank,
                            s2d_inner_dim);
                    }
                }

                // Count mode: wait for the in-kernel MAP warp to build this (chunk, LSA team)
                // unit's s2d slice in GMEM before the fan-out reads it. The proxy fence in
                // s2d_unit_wait orders the acquire ahead of the reads.
                if constexpr (COUNT_MODE) {
                    s2d_unit_wait<NUM_MAP_WARPS>(s2d_chunks_ready, (chunk_iter - 1) * LSA_TEAMS + j);
                }
                // Walk LSA teams backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int lteam_id = (my_lteam + LSA_TEAMS - j) % LSA_TEAMS;
                const routing_loads_t* routing_map_ptr = reinterpret_cast<const routing_loads_t*>(
                    rdma_to_attn_map + (lteam_id * routing_map_lsa_stride + cidx * TOKENS_PER_CHUNK));

                // Scan mode waits for the TMA-staged s2d row; count mode reads GMEM (no staging).
                if constexpr (!COUNT_MODE) {
                    uint64_t* wait_mbar = smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, s2d_stage);
                    nccl_ep::mbarrier_wait_parity(wait_mbar, s2d_parity);
                }

                for (int load_idx = 0; load_idx < routing_loads_in_chunk; load_idx++) {
                    routing_loads_t routing_flags = routing_map_ptr[load_idx];
#pragma unroll
                    for (int token_in_load = 0; token_in_load < TOKENS_PER_ROUTING_LOAD; token_in_load++) {
                        int cur_tokid = load_idx * TOKENS_PER_ROUTING_LOAD + token_in_load;
                        if (cur_tokid >= csize) {
                            break;
                        }
                        bool token_needed = *(reinterpret_cast<bool*>(&routing_flags) + token_in_load);
                        if (token_needed) {
                            // Scan mode reads the TMA-staged s2d row from SMEM; count mode reads the
                            // MAP-warp-built row directly from GMEM (gated by s2d_chunks_ready above).
                            const int32_t* s2d_smem_row;
                            if constexpr (COUNT_MODE) {
                                s2d_smem_row =
                                    sparse_to_dense_map +
                                    ((size_t)lteam_id * num_of_tokens_per_rank + cidx * TOKENS_PER_CHUNK +
                                     cur_tokid) *
                                        s2d_inner_dim;
                            } else {
                                s2d_smem_row =
                                    smem_buffer_ptr->get_s2d_map_buffer(pipeline_rank, s2d_stage, cur_tokid);
                            }
                            nccl_ep::mbarrier_wait_parity(
                                smem_buffer_ptr->get_lsa_mbarrier_producer(pipeline_rank, stage),
                                producer_parity);

                            // Per-entry parallel issue (lane handles flat_idx=lane,lane+32,...); empty/EM-dup entries resolve to issue=false.
                            for (int flat_idx = s2g_lane; flat_idx < s2d_inner_dim; flat_idx += 32) {
                                s2g_dest_t dst =
                                    dispatch_s2g_resolve_dest<kLayout>(s2d_smem_row, flat_idx, local_dup_enabled);
                                if (dst.issue) {
                                    dispatch_s2g_issue_token<
                                        TOKEN_DATA_TYPE,
                                        SMEM_TYPE,
                                        LSA_TEAM_SZ,
                                        FORWARD_DISPATCH,
                                        HAS_SF>(
                                        dst,
                                        smem_buffer_ptr,
                                        remote_expert_output_token,
                                        remote_expert_output_prob,
                                        remote_expert_output_scaling_factor,
                                        pipeline_rank,
                                        stage,
                                        HIDDEN_DIM,
                                        sf_bytes_per_token,
                                        max_recv_tokens_per_rank,
                                        experts_per_rank);
                                }
                            }
                            // S1: only issuing lanes commit/wait — idle lanes skip the empty pair.
                            if (s2g_lane < s2d_inner_dim) {
                                cuda::ptx::cp_async_bulk_commit_group();
                            }
                            in_flight_s2g += 1;
                            if (in_flight_s2g > IN_FLIGHT_S2G) {
                                if (s2g_lane < s2d_inner_dim) {
                                    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<IN_FLIGHT_S2G>{});
                                }
                                __syncwarp();
                                in_flight_s2g -= 1;
                                int notify_stage = (stage - IN_FLIGHT_S2G) >= 0 ?
                                                       (stage - IN_FLIGHT_S2G) :
                                                       (stage - IN_FLIGHT_S2G + STAGES_PER_PIPELINE);
                                if (s2g_lane == 0) {
                                    cuda::ptx::mbarrier_arrive(
                                        smem_buffer_ptr->get_lsa_mbarrier_consumer(pipeline_rank, notify_stage));
                                }
                            }

                            ring_advance(stage, producer_parity, STAGES_PER_PIPELINE);
                        }
                    }
                }
                ring_advance(s2d_stage, s2d_parity, S2D_MAP_RING_STAGES);
            }
        }
        // Drain in-flight TMA S2G writes before returning (each lane drains its own commit groups).
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        // Drain TMA stores into the generic memory path so the dispatch-tail barrier sees them.
        nccl_ep::fence_proxy_async();
    }
}

// Dispatch intra-LSA G2S warp group. With NUM_PIPELINES > 1, each warp is an independent
// pipeline processing disjoint chunks through its own partition of the shared-memory FIFO.
template <
    typename LSA_G2S_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_STAGES,
    int TOKENS_PER_CHUNK,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int NBLOCKS,
    int NUM_PIPELINES,
    bool FORWARD_DISPATCH,
    bool HAS_SF,
    int NUM_MAP_WARPS = 1>
__forceinline__ __device__ void dispatch_G2S_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const TOKEN_DATA_TYPE* attn_input_token,
    const float* attn_input_prob,
    const uint8_t* attn_input_token_scaling_factor,
    // OUTPUT
    uint64_t* gin_G2S_flags,
    // CONFIG
    const int local_rank,
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const uint64_t expected_flag_value,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    const struct dispatch_memory_region_info_t* mr_info,
    SMEM_TYPE* smem_buffer_ptr) {
    using routing_loads_t = uint4;

    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(
        TOKENS_PER_CHUNK % sizeof(routing_loads_t) == 0,
        "TOKENS_PER_CHUNK must be multiple of routing_loads_t.");
    static_assert(
        MAX_TOKENS_PER_RANK % TOKENS_PER_CHUNK == 0,
        "MAX_TOKENS_PER_RANK must be multiple of TOKENS_PER_CHUNK.");

    constexpr int TOKENS_PER_ROUTING_LOAD = sizeof(routing_loads_t) / sizeof(bool);
    constexpr int ROUTING_LOADS_PER_CHUNK = TOKENS_PER_CHUNK / TOKENS_PER_ROUTING_LOAD;
    constexpr int STAGES_PER_PIPELINE = NUM_STAGES / NUM_PIPELINES;

    const int pipeline_rank = LSA_G2S_GROUP::warp_rank();
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int max_num_of_chunks_per_rank = nccl_ep::ceil_div(MAX_TOKENS_PER_RANK, TOKENS_PER_CHUNK);
    // TOKENS_PER_ROUTING_LOAD must match the producer's pad in scan_kernel.cuh
    const int routing_map_lsa_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
    int stage = 0;
    uint32_t consumer_parity = 1;
    int tokens_produced = 0;

    if (cuda::ptx::elect_sync(~0)) {
        int chunk_iter = 0;
        for (int cidx = blockIdx.x; cidx < num_of_chunks_per_rank; cidx += NBLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) continue;

            int routing_loads_in_chunk;
            int csize;
            if (rem_chunk_sz != 0 && cidx == num_of_chunks_per_rank - 1) {
                routing_loads_in_chunk = nccl_ep::ceil_div(rem_chunk_sz, (int)sizeof(routing_loads_t));
                csize = rem_chunk_sz;
            } else {
                routing_loads_in_chunk = ROUTING_LOADS_PER_CHUNK;
                csize = TOKENS_PER_CHUNK;
            }

            for (int j = 0; j < LSA_TEAMS; j++) {
                // Walk LSA teams backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int lteam_id = (my_lteam + LSA_TEAMS - j) % LSA_TEAMS;

                g2s_source_t<TOKEN_DATA_TYPE> src = dispatch_g2s_resolve_source<
                    TOKEN_DATA_TYPE,
                    LSA_TEAMS,
                    LSA_TEAM_SZ,
                    MAX_TOKENS_PER_RANK,
                    TOKENS_PER_CHUNK,
                    NBLOCKS,
                    FORWARD_DISPATCH,
                    HAS_SF>(
                    attn_input_token,
                    attn_input_prob,
                    attn_input_token_scaling_factor,
                    lteam_id,
                    my_lteam,
                    local_rank,
                    cidx,
                    expected_flag_value,
                    HIDDEN_DIM,
                    sf_bytes_per_token,
                    experts_per_rank,
                    dcomm,
                    num_ctx_per_comm,
                    gin_base_ptr,
                    mr_info);

                const routing_loads_t* routing_map_ptr = reinterpret_cast<const routing_loads_t*>(
                    rdma_to_attn_map + (lteam_id * routing_map_lsa_stride) + (cidx * TOKENS_PER_CHUNK));

                int packed_dense_idx = 0;
                for (int load_idx = 0; load_idx < routing_loads_in_chunk; load_idx++) {
                    routing_loads_t routing_flags = routing_map_ptr[load_idx];

#pragma unroll
                    for (int token_in_load = 0; token_in_load < TOKENS_PER_ROUTING_LOAD; token_in_load++) {
                        int cur_tokid = load_idx * TOKENS_PER_ROUTING_LOAD + token_in_load;
                        if (cur_tokid >= csize) {
                            break;
                        }

                        bool token_needed = *(reinterpret_cast<bool*>(&routing_flags) + token_in_load);
                        if (token_needed) {
                            if (tokens_produced >= STAGES_PER_PIPELINE) {
                                uint64_t* mbar =
                                    smem_buffer_ptr->get_lsa_mbarrier_consumer(pipeline_rank, stage);
                                nccl_ep::mbarrier_wait_parity(mbar, consumer_parity);
                            }

                            dispatch_g2s_issue_token<
                                TOKEN_DATA_TYPE,
                                SMEM_TYPE,
                                LSA_TEAMS,
                                LSA_TEAM_SZ,
                                FORWARD_DISPATCH,
                                HAS_SF>(
                                src,
                                cur_tokid,
                                packed_dense_idx,
                                smem_buffer_ptr,
                                pipeline_rank,
                                stage,
                                HIDDEN_DIM,
                                sf_bytes_per_token,
                                experts_per_rank,
                                my_lteam,
                                mr_info);

                            if (src.use_packed) {
                                packed_dense_idx++;
                            }

                            tokens_produced += 1;
                            ring_advance(stage, consumer_parity, STAGES_PER_PIPELINE);
                        }
                    }
                }
            }
        }
    }
    // Update residue flags (only pipeline 0 does this to avoid duplicate writes).
    if (LSA_G2S_GROUP::warp_rank() == 0) {
        int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;

        for (int lteam_id = blockIdx.x; lteam_id < LSA_TEAMS - 1; lteam_id += gridDim.x) {
            uint64_t* residue_flag_base_ptr =
                gin_G2S_flags + (lteam_id * max_num_of_chunks_per_rank) + num_of_chunks_per_rank;
            if (LSA_G2S_GROUP::thread_rank() < residue_flag_count) {
                residue_flag_base_ptr[LSA_G2S_GROUP::thread_rank()] = expected_flag_value;
            }
        }
    }
}

// Shared single-entry G2S issue for both the intra- and cross-LSA-team combine
// G2S warps. CROSS_LSA selects the cross-LSA-team staged-buffer accessors and
// flag buffer; otherwise the intra-LSA ones (both sets live on the same
// combine smem layout, so one helper covers both tiers).
//   - derive stage_idx + parity from (global_offset + rank_in_batch)
//   - wait for consumer to free the stage (mbarrier_try_wait_parity)
//   - cp_async_bulk the token (and the prob under BACKWARD_COMBINE). MXFP8 E8M0 rows
//     ride in the token copy: every source is packed [FP8 H | E8M0 H/32].
//   - optionally write <tier>_flag_G2S_buffer[stage_idx]
//   - mbarrier_arrive_expect_tx with the cumulative tx size
//
// The caller has already computed the token (and prob) source pointers
// and the `is_last_entry` boolean (only consulted when WRITE_LAST_FLAG).
template <bool CROSS_LSA, bool BACKWARD_COMBINE, bool WRITE_LAST_FLAG, typename SMEM_TYPE>
__forceinline__ __device__ void issue_g2s_entry(
    SMEM_TYPE* smem_buffer_ptr,
    int global_offset,
    int rank_in_batch,
    int starting_G2S_index,
    int ring_len,
    const void* token_src,
    uint32_t token_bytes,
    const float* prob_src,
    uint32_t prob_bytes,
    bool is_last_entry) {
    const int my_abs_offset = global_offset + rank_in_batch;
    const int stage_idx = starting_G2S_index + (my_abs_offset % ring_len);
    const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

    uint64_t* consumer_mbar;
    uint64_t* producer_mbar;
    void* token_dst;
    if constexpr (CROSS_LSA) {
        consumer_mbar = smem_buffer_ptr->get_cross_lsa_mbarrier_G2S_consumer(stage_idx);
        producer_mbar = smem_buffer_ptr->get_cross_lsa_mbarrier_G2S_producer(stage_idx);
        token_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_cross_lsa_token_G2S(stage_idx));
    } else {
        consumer_mbar = smem_buffer_ptr->get_lsa_mbarrier_G2S_consumer(stage_idx);
        producer_mbar = smem_buffer_ptr->get_lsa_mbarrier_G2S_producer(stage_idx);
        token_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_lsa_token_G2S(stage_idx));
    }

    while (!cuda::ptx::mbarrier_try_wait_parity(consumer_mbar, parity)) {
    }

    uint32_t total_tx_size = 0;
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        token_dst,
        token_src,
        token_bytes,
        producer_mbar);
    total_tx_size += token_bytes;

    if constexpr (BACKWARD_COMBINE) {
        void* prob_dst;
        if constexpr (CROSS_LSA) {
            prob_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_cross_lsa_prob_G2S(stage_idx));
        } else {
            prob_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_lsa_prob_G2S(stage_idx));
        }
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            prob_dst,
            reinterpret_cast<const void*>(prob_src),
            prob_bytes,
            producer_mbar);
        total_tx_size += prob_bytes;
    }

    if constexpr (WRITE_LAST_FLAG) {
        if constexpr (CROSS_LSA) {
            smem_buffer_ptr->cross_lsa_flag_G2S_buffer[stage_idx] = is_last_entry;
        } else {
            smem_buffer_ptr->lsa_flag_G2S_buffer[stage_idx] = is_last_entry;
        }
    }

    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release,
        cuda::ptx::scope_cta,
        cuda::ptx::space_shared,
        producer_mbar,
        total_tx_size);
}

// Warp-cooperative scan of one sparse_to_dense_map row plus its inline
// broadcast-issue, shared by the intra-LSA G2S warp and the LOCAL tier of
// the cross-LSA-team G2S warp (the two differ only in CROSS_LSA, starting_G2S_index,
// and which s2d row / FIFO slice they target -- all passed in).
//
//  * Process s2d entries in WARP_SIZE steps
//  * Each lane loads an s2d entry and participates in valid entry filtering
//  * Next a single lane performs TMA loads of the token and probability data
//    * NOTE: Using single lane allows to ensure the ordering of TMA loads so
//      they are aligned with the RED process.
template <
    bool CROSS_LSA,
    bool BACKWARD_COMBINE,
    ncclEpLayout_t kLayout,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    typename SMEM_TYPE>
__forceinline__ __device__ void issue_local_g2s_row(
    SMEM_TYPE* smem_buffer_ptr,
    const int32_t* sparse_to_dense_row,
    int s2d_entries,
    int& global_offset,
    int starting_G2S_index,
    int ring_len,
    int lane_id,
    const void* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    uint32_t token_bytes,
    uint32_t prob_bytes,
    int experts_per_rank,
    int ranks_per_lsa_team,
    bool combine_local_reduce_enabled) {
    constexpr int WARP_SIZE = 32;
    int total_valid_count = 0;
    int valid_seen = 0;
    bool have_pending = false;
    int32_t pending_s2d = -1;
    int pending_entry_idx = -1;

    // Resolve the s2d entry and issue TMA load
    auto issue_pending = [&](bool is_last_entry) {
        int rank_id;
        int slot;
        if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
            rank_id = em_s2d_unpack_rank(pending_s2d);
            slot = em_s2d_unpack_slot(pending_s2d);
        } else {
            rank_id = pending_entry_idx;
            slot = pending_s2d;
        }
        // Avoid int32-overflow issue by casting slot to size_t
        const size_t slot_st = static_cast<size_t>(slot);
        // Byte-addressed offset: token_bytes already encodes the per-slot byte count
        // for all dtypes (BF16=2*H, FP8=H, FP32=4*H), so no per-recipe branching needed.
        const void* token_src =
            static_cast<const uint8_t*>(remote_expert_input_token[rank_id]) + slot_st * token_bytes;
        const float* prob_src = nullptr;
        if constexpr (BACKWARD_COMBINE) {
            prob_src = remote_expert_input_prob[rank_id] + (slot_st * (experts_per_rank * ranks_per_lsa_team));
        }
        issue_g2s_entry<CROSS_LSA, BACKWARD_COMBINE, /*WRITE_LAST_FLAG=*/true>(
            smem_buffer_ptr,
            global_offset,
            valid_seen,
            starting_G2S_index,
            ring_len,
            token_src,
            token_bytes,
            prob_src,
            prob_bytes,
            is_last_entry);
    };

    for (int entry_base = 0; entry_base < s2d_entries; entry_base += WARP_SIZE) {
        const int entry_idx = entry_base + lane_id;
        const bool lane_active = (entry_idx < s2d_entries);
        const int32_t s2d_val = lane_active ? sparse_to_dense_row[entry_idx] : -1;
        const bool is_secondary = is_em_secondary_entry<kLayout>(s2d_val, lane_id, combine_local_reduce_enabled);

        const unsigned mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1 && !is_secondary);
        total_valid_count += __popc(mask);

        // All 32 lanes iterate set bits of this batch's mask in lockstep. The lane
        // at `src_lane` broadcasts its s2d_val to every lane via __shfl_sync; lane
        // 0 takes the value and issues the TMA. The issue itself is deferred by one
        // entry (kept in pending_*) so the final issue knows it's last.
        unsigned m = mask;
        while (m != 0) {
            const int src_lane = __ffs((int)m) - 1; // 0..31
            m &= (m - 1);
            const int32_t bcast_s2d = __shfl_sync(0xffffffff, s2d_val, src_lane);

            if (lane_id == 0) {
                if (have_pending) {
                    // Flush the previously-buffered entry; not last because we just
                    // discovered another to issue.
                    issue_pending(/*is_last_entry=*/false);
                    valid_seen++;
                }
                pending_s2d = bcast_s2d;
                pending_entry_idx = entry_base + src_lane;
                have_pending = true;
            }
        }
    }

    // Final flush: issue the last pending entry (if any) with is_last_entry=true.
    // Only lane 0 ever sets have_pending, so this is a single-thread tail call.
    if (lane_id == 0 && have_pending) {
        issue_pending(/*is_last_entry=*/true);
    }

    global_offset += total_valid_count;
}

// Decode a block's flat chunk index into its destination LSA team, per-LSA-team chunk id, and token
// count. Emit order (chunk c for LSA team+1..LSA team-1, then c+1) matches the RDMA consume order.
// Shared by the combine warp groups.
struct comb_chunk_meta_t {
    int lteam_id;
    int chunk_id;
    int csize;
};

template <int LSA_TEAMS, int TOKENS_PER_CHUNK>
__forceinline__ __device__ comb_chunk_meta_t
combine_chunk_meta(int cidx, int my_lteam, int cpr, int rem_chunk_sz) {
    comb_chunk_meta_t c;
    c.lteam_id = (cidx % (LSA_TEAMS - 1) + (my_lteam + 1)) % LSA_TEAMS;
    c.chunk_id = cidx / (LSA_TEAMS - 1);
    c.csize = (rem_chunk_sz != 0 && c.chunk_id == cpr - 1) ? rem_chunk_sz : TOKENS_PER_CHUNK;
    return c;
}

// Intra-LSA G2S warp for the combine kernel. Exactly one such warp per block.
template <
    typename SMEM_TYPE,
    int STAGES_G2S,
    int TOKENS_PER_CHUNK,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int NBLOCKS,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclEpLayout_t kLayout,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe>
__forceinline__ __device__ void combine_G2S_intra_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const int32_t* sparse_to_dense_map,
    const void* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    const bool combine_local_reduce_enabled,
    SMEM_TYPE* smem_buffer_ptr) {
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = (LSA_TEAMS - 1) * cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per LSA team.
    const int rdma_per_lsa_sz = nccl_ep::align(num_of_tokens_per_rank, 16);

    // Warp-cooperative G2S: lanes scan sparse_to_dense_map in parallel and issue TMA to distinct
    // stages; global_offset counts total stages filled so RED consumes them in lockstep.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    constexpr int ring_len = STAGES_G2S;
    // Every MXFP8 source is packed [FP8 H | E8M0 H/32]: the prologue emits it that way and
    // the RDMA leg carries BF16, so one TMA per token covers values and scales alike.
    using Recipe = nccl_ep::combine_recipe_traits<kCombineRecipe>;
    const uint32_t token_bytes = (uint32_t)Recipe::template packed_bytes<kTokenDtype>(HIDDEN_DIM);
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SZ) * sizeof(float));

    // EM unfused-combine dedup uses __shfl_up_sync(1); requires s2d_inner_dim <= WARP_SIZE.
    if (combine_local_reduce_enabled && lane_id == 0 && smem_buffer_ptr->s2d_inner_dim > WARP_SIZE) {
        __trap();
    }

    int global_offset = 0;

    for (int cidx = blockIdx.x; cidx < total_chunks; cidx += NBLOCKS) {
        const auto meta = combine_chunk_meta<LSA_TEAMS, TOKENS_PER_CHUNK>(cidx, my_lteam, cpr, rem_chunk_sz);

        const bool* rdma_row_base =
            rdma_to_attn_map + (meta.lteam_id * rdma_per_lsa_sz + meta.chunk_id * TOKENS_PER_CHUNK);

        // S2D inner dim: mode-dependent, carried by SMEM layout struct.
        const int s2d_entries_g2s = smem_buffer_ptr->s2d_inner_dim;
        const int32_t* s2d_row_base =
            sparse_to_dense_map +
            (meta.lteam_id * num_of_tokens_per_rank + meta.chunk_id * TOKENS_PER_CHUNK) * s2d_entries_g2s;

        for (int cur_tokid = 0; cur_tokid < meta.csize; cur_tokid++) {
            // Skip dst tokens this LSA team doesn't need.
            if (!rdma_row_base[cur_tokid]) {
                continue;
            }

            const int32_t* sparse_to_dense_row = s2d_row_base + cur_tokid * s2d_entries_g2s;

            // Warp-cooperative s2d-row scan with inline broadcast-issue; advances global_offset
            // by the number of entries issued. See issue_local_g2s_row.
            issue_local_g2s_row</*CROSS_LSA=*/false, BACKWARD_COMBINE, kLayout, HIDDEN_DIM, kTokenDtype>(
                smem_buffer_ptr,
                sparse_to_dense_row,
                s2d_entries_g2s,
                global_offset,
                /*starting_G2S_index=*/0,
                ring_len,
                lane_id,
                remote_expert_input_token,
                remote_expert_input_prob,
                token_bytes,
                prob_bytes,
                experts_per_rank,
                LSA_TEAM_SZ,
                combine_local_reduce_enabled);
        }
    }
}

// MXFP8 element<->accumulator mapping for the RED groups.
//
// Strided ownership (element = n*THRDS + tid) gives each thread one FP8 pair per accumulator,
// so MXFP8 pays the same trip count as BF16 plus a per-iteration E8M0 load/broadcast. Contiguous
// ownership below makes one vector (= kPairs pairs) share a single E8M0 block: one wide SMEM
// load, one scale read, no __shfl. Load and store must use the same map.
template <int NUM_ACC>
struct mxfp8_red_map {
    // Widest power-of-two vector that tiles NUM_ACC. 8 pairs = one 16B load; must also divide
    // the 16-pair E8M0 group so a vector never straddles two scale blocks.
    static constexpr int kPairs =
        (NUM_ACC % 8 == 0) ? 8 : ((NUM_ACC % 4 == 0) ? 4 : ((NUM_ACC % 2 == 0) ? 2 : 1));
    static_assert(16 % kPairs == 0, "vector must tile the 16-pair E8M0 group");
    static constexpr int kLanesPerScale = 16 / kPairs;

    __device__ static constexpr int vec_base(int n, int thrds, int tid) {
        return (n / kPairs) * (kPairs * thrds) + tid * kPairs;
    }
    __device__ static constexpr int pair_id(int n, int thrds, int tid) {
        return vec_base(n, thrds, tid) + (n % kPairs);
    }
};

// Wide shared load of kPairs FP8 pairs. Alignment: stage bases are 128B-aligned and vec_base
// is a multiple of kPairs, so the byte address is a multiple of 2*kPairs (16B when kPairs==8).
// Copied via a local POD so we do not type-pun the uint16_t array itself.
template <int kPairs>
__device__ __forceinline__ void ld_fp8_pairs_shared(uint16_t (&dst)[kPairs], const uint16_t* src) {
    static_assert(kPairs > 0 && kPairs <= 8 && (kPairs & (kPairs - 1)) == 0,
                  "kPairs must be a power of two up to 8, so the run is one machine load");
    using vec_t = typename nccl_ep::VecInt<kPairs * static_cast<int>(sizeof(uint16_t))>::vec_t;
    const vec_t v = *reinterpret_cast<const vec_t*>(src);
    __builtin_memcpy(dst, &v, sizeof(v));
}

// Wide shared store of kPairs consecutive token pairs: the store-side mirror of
// ld_fp8_pairs_shared. vec_base hands each thread kPairs consecutive pair ids per group, so a
// group is one contiguous run and leaves as 16B stores instead of kPairs narrow ones.
// Alignment: the byte offset is first_pair * pair_bytes with first_pair a multiple of kPairs,
// hence a multiple of the run width, and stage bases are 128B aligned.
template <int kPairs, ncclDataType_t kTokenDtype>
__device__ __forceinline__ void st_token_pairs_shared(void* base, int first_pair, const float2* src) {
    constexpr int kPairBytes = 2 * nccl_ep::size_u8<kTokenDtype>();
    constexpr int kRunBytes = kPairs * kPairBytes;
    // Encode into a local POD first so the wide move is a plain copy of the run.
    alignas(16) uint8_t staging[kRunBytes];
#pragma unroll
    for (int t = 0; t < kPairs; t++) {
        nccl_ep::st_token_pair<kTokenDtype>(staging, t, src[t]);
    }

    // 16 B is the widest machine store, so a run wider than that leaves as a short loop of
    // them and a narrower one as a single store of its own width.
    constexpr int kChunkBytes = kRunBytes < 16 ? kRunBytes : 16;
    static_assert(kRunBytes % kChunkBytes == 0, "unexpected token pair run width");
    using chunk_t = typename nccl_ep::VecInt<kChunkBytes>::vec_t;

    uint8_t* dst = static_cast<uint8_t*>(base) + static_cast<size_t>(first_pair) * kPairBytes;
#pragma unroll
    for (int b = 0; b < kRunBytes; b += kChunkBytes) {
        chunk_t v;
        __builtin_memcpy(&v, staging + b, sizeof(v));
        *reinterpret_cast<chunk_t*>(dst + b) = v;
    }
}

// Reduce all G2S source-token contributions for one destination token into FP32 registers
// (+prob into SMEM for backward). Advances the G2S stage cursor/parity to the next dst token.
template <
    typename RED_GROUP,
    int STAGES_G2S,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe,
    typename SMEM_TYPE,
    int NUM_ACC>
__forceinline__ __device__ void combine_reduce_dst_token(
    SMEM_TYPE* smem_buffer_ptr,
    int& token_stage,
    uint32_t& token_producer_parity,
    float2 (&acc_token_fp32)[NUM_ACC],
    float* acc_prob_ptr,
    int prob_vec_per_thread,
    int prob_dim) {
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

#pragma unroll
    for (int n = 0; n < NUM_ACC; n++) {
        acc_token_fp32[n].x = 0.0f;
        acc_token_fp32[n].y = 0.0f;
    }
    if constexpr (BACKWARD_COMBINE) {
#pragma unroll
        for (int n = 0; n < prob_vec_per_thread; n++) {
            acc_prob_ptr[n] = 0.0f;
        }
    }

    // Consume source tokens for this dst token until the producer marks the last one.
    bool last_src_token = false;
    do {
        __nv_bfloat162* load_token_base_ptr =
            reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_lsa_token_G2S(token_stage));
        float* load_prob_base_ptr;
        if constexpr (BACKWARD_COMBINE) {
            load_prob_base_ptr = smem_buffer_ptr->get_lsa_prob_G2S(token_stage);
        }

        // Warp 0 waits for the producer; then the whole reduction group reads this stage.
        if (RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    smem_buffer_ptr->get_lsa_mbarrier_G2S_producer(token_stage), token_producer_parity)) {
                }
            }
        }
        arrive_and_wait(RED_GROUP::size(), 1);

// Accumulate the register-resident token. NONE-FP16 reads __half2, NONE-FP32 reads float2 and
// skips precision conversion; predicates are launch-uniform so branching is free.
// MXFP8 stages FP8 pairs plus one E8M0 byte per 32 elements and dequantizes here.
        if constexpr (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) {
            using Map = mxfp8_red_map<NUM_ACC>;
            constexpr int kPairs = Map::kPairs;
            const int tid = (int)RED_GROUP::thread_rank();
            const int thrds = (int)RED_GROUP::size();
            const uint16_t* fp8_data = reinterpret_cast<const uint16_t*>(load_token_base_ptr);
            const uint8_t* scale_data =
                reinterpret_cast<const uint8_t*>(load_token_base_ptr) + HIDDEN_DIM;
#pragma unroll
            for (int base = 0; base < NUM_ACC; base += kPairs) {
                const int pair0 = Map::vec_base(base, thrds, tid);
                if (pair0 + kPairs <= NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                    // Full vector: one wide load, one scale, no shuffle.
                    const float scale_f = nccl_ep::mxfp8::e8m0_to_scale(scale_data[pair0 / 16]);
                    uint16_t fp8x2[kPairs];
                    ld_fp8_pairs_shared<kPairs>(fp8x2, fp8_data + pair0);
#pragma unroll
                    for (int t = 0; t < kPairs; t++) {
                        const float2 src_data_fp32 =
                            nccl_ep::mxfp8::dequant_x2_scaled(fp8x2[t], scale_f);
                        acc_token_fp32[base + t].x += src_data_fp32.x;
                        acc_token_fp32[base + t].y += src_data_fp32.y;
                    }
                } else {
                    // Tail: never wide-load past the FP8 payload into the E8M0 tail.
#pragma unroll
                    for (int t = 0; t < kPairs; t++) {
                        const int pair = pair0 + t;
                        if (pair < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                            const float scale_f =
                                nccl_ep::mxfp8::e8m0_to_scale(scale_data[pair / 16]);
                            const float2 src_data_fp32 =
                                nccl_ep::mxfp8::dequant_x2_scaled(fp8_data[pair], scale_f);
                            acc_token_fp32[base + t].x += src_data_fp32.x;
                            acc_token_fp32[base + t].y += src_data_fp32.y;
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int n = 0; n < NUM_ACC; n++) {
                int element_id = (n * RED_GROUP::size()) + RED_GROUP::thread_rank();
                if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                    float2 src_data_fp32 = nccl_ep::ld_token_pair<kTokenDtype>(load_token_base_ptr, element_id);
                    acc_token_fp32[n].x += src_data_fp32.x;
                    acc_token_fp32[n].y += src_data_fp32.y;
                }
            }
        }
        if constexpr (BACKWARD_COMBINE) {
#pragma unroll
            for (int n = 0; n < prob_vec_per_thread; n++) {
                int prob_element_id = RED_GROUP::thread_rank() + n * RED_GROUP::size();
                if (prob_element_id < prob_dim) {
                    acc_prob_ptr[n] += load_prob_base_ptr[prob_element_id];
                }
            }
        }

        last_src_token = smem_buffer_ptr->lsa_flag_G2S_buffer[token_stage];

        // All reduction threads must finish reading before the producer reuses this stage.
        arrive_and_wait(RED_GROUP::size(), 1);
        if (RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_lsa_mbarrier_G2S_consumer(token_stage));
            }
        }

        token_stage += 1;
        if (token_stage == STAGES_G2S) {
            token_stage = 0;
            token_producer_parity ^= 1;
        }
    } while (!last_src_token);
}

// Store one reduced destination token (+prob) from FP32 registers into an S2G SMEM stage and
// TMA-copy it to the per-destination intra-LSA red buffer. Advances the S2G stage cursor.
// MXFP8 stores the same 2H-byte BF16 row as NONE and emits no scales; only the
// element->accumulator map differs (contiguous vs strided).
// kCombineRecipe placed before deducible SMEM_TYPE/NUM_ACC so callers can specify:
//   combine_store_reduced_token<RED_GROUP,S2G,BWD,H,Dt,kRecipe,SMEM_TYPE,ACC_N>(...)
// SMEM_TYPE and NUM_ACC are also deducible from the function arguments.
template <
    typename RED_GROUP,
    int STAGES_S2G,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe,
    typename SMEM_TYPE,
    int NUM_ACC>
__forceinline__ __device__ void combine_store_reduced_token(
    SMEM_TYPE* smem_buffer_ptr,
    int& dst_token_stage,
    const float2 (&acc_token_fp32)[NUM_ACC],
    const float* acc_prob_ptr,
    int prob_vec_per_thread,
    int prob_dim,
    uint16_t* red_token_base,
    float* red_prob_base,
    int cur_tokid) {
    constexpr int BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

    // Ensure any earlier TMA read from this S2G stage has completed before we overwrite it.
    if (RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<STAGES_S2G - 1>{});
        }
    }
    arrive_and_wait(RED_GROUP::size(), 1);

    __nv_bfloat162* store_token_base_ptr =
        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_lsa_token_S2G(dst_token_stage));

    // Store the register-resident token (NONE-FP16 packs __half2, NONE-FP32 writes float2 verbatim).
    // Must mirror combine_reduce_dst_token: MXFP8 fills the accumulators via the contiguous
    // mxfp8_red_map, so a strided store here would permute the token.
    using Map = mxfp8_red_map<NUM_ACC>;
#pragma unroll
    for (int n = 0; n < NUM_ACC; n++) {
        int element_id;
        if constexpr (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) {
            element_id = Map::pair_id(n, (int)RED_GROUP::size(), (int)RED_GROUP::thread_rank());
        } else {
            element_id = (n * RED_GROUP::size()) + RED_GROUP::thread_rank();
        }
        if (element_id < BF16X2_ELEMENTS_PER_TOKEN) {
            nccl_ep::st_token_pair<kTokenDtype>(store_token_base_ptr, element_id, acc_token_fp32[n]);
        }
    }
    if constexpr (BACKWARD_COMBINE) {
        float* store_prob_base_ptr = smem_buffer_ptr->get_lsa_prob_S2G(dst_token_stage);
#pragma unroll
        for (int n = 0; n < prob_vec_per_thread; n++) {
            int prob_element_id = RED_GROUP::thread_rank() + n * RED_GROUP::size();
            if (prob_element_id < prob_dim) {
                store_prob_base_ptr[prob_element_id] = acc_prob_ptr[n];
            }
        }
    }

    // Publish SMEM writes to the async copy engine, then sync before the TMA launch.
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    arrive_and_wait(RED_GROUP::size(), 1);

    if (RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            // Wire token width scaled into uint16_t units (4 B FP32, 2 B BF16/FP16).
            const size_t red_token_bytes = HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>());
            uint16_t* current_token_addr =
                red_token_base + cur_tokid * red_token_bytes / sizeof(uint16_t);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_global,
                cuda::ptx::space_shared,
                reinterpret_cast<void*>(current_token_addr),
                reinterpret_cast<const void*>(smem_buffer_ptr->get_lsa_token_S2G(dst_token_stage)),
                (uint32_t)(red_token_bytes));

            if constexpr (BACKWARD_COMBINE) {
                float* current_prob_addr = red_prob_base + cur_tokid * prob_dim;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    reinterpret_cast<void*>(current_prob_addr),
                    reinterpret_cast<const void*>(smem_buffer_ptr->get_lsa_prob_S2G(dst_token_stage)),
                    (uint32_t)(prob_dim * sizeof(float)));
            }
            cuda::ptx::cp_async_bulk_commit_group();
        }
    }
    dst_token_stage += 1;
    if (dst_token_stage == STAGES_S2G) {
        dst_token_stage = 0;
    }
}

// Drain outstanding TMA S2G writes and publish the cumulative streaming counter to the RDMA warp.
// CHUNK_END uses a device-scope fence (GDR/NIC visibility); mid-batch uses a block-scope fence.
template <typename RED_GROUP, int STREAMING_BATCH, bool CHUNK_END>
__forceinline__ __device__ void combine_streaming_drain(
    int& streaming_pending,
    int& additional_in_flight_s2g,
    uint32_t& cumulative_produced,
    volatile uint32_t* rdma_streaming_counter) {
    if (RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            if constexpr (STREAMING_BATCH > 0) {
                cumulative_produced += streaming_pending;
                if constexpr (CHUNK_END) {
                    __threadfence(); // device scope: flush L2->VRAM for NIC visibility (GDR)
                } else {
                    __threadfence_block(); // same block; sm_90 SMEM is not cached
                }
                *rdma_streaming_counter = cumulative_produced;
            }
        }
    }
    additional_in_flight_s2g = 0;
    streaming_pending = 0;
}

// Intra-LSA reduction warp group for the combine kernel.
// Reads the packed [FP8 H | E8M0 H/32] intra-LSA wire for MXFP8 and writes a BF16 row to
// combine_gin_RED_tokens. Only reached for multi-LSA-team groups, which MXFP8 does not
// support -- it is rejected at the API, so this warp is never instantiated with it.
template <
    typename RED_GROUP,
    typename SMEM_TYPE,
    int STAGES_G2S,
    int STAGES_S2G,
    int TOKENS_PER_CHUNK,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SZ,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe>
__forceinline__ __device__ void combine_RED_intra_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    // OUTPUT
    uint16_t* combine_gin_RED_tokens,
    float* combine_gin_RED_prob,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    SMEM_TYPE* smem_buffer_ptr) {
    // Routing map is read as vectorized 16B loads; each dst token contributes one bool.
    using routing_loads_t = uint4;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(
        TOKENS_PER_CHUNK % sizeof(routing_loads_t) == 0,
        "TOKENS_PER_CHUNK must be multiple of routing_loads_t.");
    constexpr int ROUTING_LOADS_PER_CHUNK = TOKENS_PER_CHUNK / sizeof(routing_loads_t);
    constexpr int TOKENS_PER_ROUTING_LOAD = sizeof(routing_loads_t) / sizeof(bool);

    // Tokens reduced as BF16x2 in FP32; HIDDEN_DIM must be even.
    constexpr int BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
    constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    constexpr int ACC_ELEM_PER_THRD =
        nccl_ep::ceil_div(BF16X2_ELEMENTS_PER_TOKEN, (int)RED_GROUP::size());
    // Backward-combine prob vectors stay in float (no BF16 packing).
    const int prob_dim = experts_per_rank * LSA_TEAM_SZ;
    const int prob_vec_per_thread = nccl_ep::ceil_div(prob_dim, (int)RED_GROUP::size());
    // Compile-time upper bound sized exactly to this instantiation's LSA team.
    constexpr int MAX_PROB_ELEM_PER_THRD =
        nccl_ep::ceil_div(NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SZ, (int)RED_GROUP::size());

    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = (LSA_TEAMS - 1) * cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per LSA team.
    const int rdma_per_lsa_sz = nccl_ep::align(num_of_tokens_per_rank, 16);

    // G2S FIFO cursor + producer parity (source tokens); S2G FIFO cursor (reduced dst tokens).
    int token_stage = 0;
    uint32_t token_producer_parity = 0;
    int dst_token_stage = 0;

    // Streaming overlap: drain + signal every STREAMING_BATCH dst tokens. The counter is cumulative
    // across chunks (never reset), so the consumer never races a reset.
    constexpr int STREAMING_BATCH = NCCL_EP_HT_COMBINE_RDMA_STREAMING_BATCH;
    int streaming_pending = 0;
    uint32_t cumulative_produced = 0;

    for (int cidx = blockIdx.x; cidx < total_chunks; cidx += NBLOCKS) {
        const auto meta = combine_chunk_meta<LSA_TEAMS, TOKENS_PER_CHUNK>(cidx, my_lteam, cpr, rem_chunk_sz);
        // Compact destination-slot index + token offset in the RDMA reduction buffers.
        const int rdma_tile_id = meta.lteam_id > my_lteam ? meta.lteam_id - 1 : meta.lteam_id;
        const int gin_RED_slot =
            rdma_tile_id * MAX_TOKENS_PER_RANK + meta.chunk_id * TOKENS_PER_CHUNK;
        // Vector loads covering this chunk's routing flags (tail chunk is shorter).
        const int routing_loads_for_chunk = (rem_chunk_sz != 0 && meta.chunk_id == cpr - 1)
                                                 ? nccl_ep::ceil_div(rem_chunk_sz, (int)sizeof(routing_loads_t))
                                                 : ROUTING_LOADS_PER_CHUNK;

        const routing_loads_t* rdma_map_base = reinterpret_cast<const routing_loads_t*>(
            rdma_to_attn_map + (meta.lteam_id * rdma_per_lsa_sz + meta.chunk_id * TOKENS_PER_CHUNK));

        // Per-token stride in gin_RED: Recipe::rdma_bytes, the inter-node row width
        // (dtype-width for NONE), matching the previous
        // HIDDEN_DIM * size_u16<kTokenDtype>() form for every unquantized case.
        const int red_stride =
            nccl_ep::combine_recipe_traits<kCombineRecipe>::template rdma_bytes<kTokenDtype>(HIDDEN_DIM);
        uint16_t* red_token_base = reinterpret_cast<uint16_t*>(
            reinterpret_cast<uint8_t*>(combine_gin_RED_tokens) +
            static_cast<size_t>(gin_RED_slot) * red_stride);
        float* red_prob_base = nullptr;
        if constexpr (BACKWARD_COMBINE) {
            red_prob_base = combine_gin_RED_prob + gin_RED_slot * prob_dim;
        }
        streaming_pending = 0;
        int additional_in_flight_s2g = 0;
        for (int load_idx = 0; load_idx < routing_loads_for_chunk; load_idx++) {
            routing_loads_t routing_data = rdma_map_base[load_idx];
#pragma unroll
            for (int token_in_load = 0; token_in_load < TOKENS_PER_ROUTING_LOAD; token_in_load++) {
                int cur_tokid = load_idx * TOKENS_PER_ROUTING_LOAD + token_in_load;
                // Tail chunk: stop once past the real token count.
                if (cur_tokid >= meta.csize) {
                    break;
                }
                // Skip dst tokens this LSA team doesn't need.
                if (!*(reinterpret_cast<bool*>(&routing_data) + token_in_load)) {
                    continue;
                }

                float2 acc_token_fp32[ACC_ELEM_PER_THRD];
                // acc_prob storage instantiated only in backward specializations.
                using acc_prob_storage_type =
                    acc_prob_storage_t<BACKWARD_COMBINE, MAX_PROB_ELEM_PER_THRD>;
                [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
                [[maybe_unused]] float* acc_prob_ptr = nullptr;
                if constexpr (BACKWARD_COMBINE) {
                    acc_prob_ptr = acc_prob_storage.data;
                }

                combine_reduce_dst_token<
                    RED_GROUP,
                    STAGES_G2S,
                    BACKWARD_COMBINE,
                    HIDDEN_DIM,
                    kTokenDtype,
                    kCombineRecipe>(
                    smem_buffer_ptr,
                    token_stage,
                    token_producer_parity,
                    acc_token_fp32,
                    acc_prob_ptr,
                    prob_vec_per_thread,
                    prob_dim);

                combine_store_reduced_token<
                    RED_GROUP,
                    STAGES_S2G,
                    BACKWARD_COMBINE,
                    HIDDEN_DIM,
                    kTokenDtype,
                    kCombineRecipe,
                    SMEM_TYPE,
                    ACC_ELEM_PER_THRD>(
                    smem_buffer_ptr,
                    dst_token_stage,
                    acc_token_fp32,
                    acc_prob_ptr,
                    prob_vec_per_thread,
                    prob_dim,
                    red_token_base,
                    red_prob_base,
                    cur_tokid);

                additional_in_flight_s2g += 1;
                streaming_pending++;
                if constexpr (STREAMING_BATCH > 0) {
                    if (streaming_pending >= STREAMING_BATCH) {
                        combine_streaming_drain<RED_GROUP, STREAMING_BATCH, /*CHUNK_END=*/false>(
                            streaming_pending,
                            additional_in_flight_s2g,
                            cumulative_produced,
                            (volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter);
                    }
                }
            }
        }

        // End of chunk: drain remaining TMA writes + signal the streaming counter.
        if (streaming_pending > 0 || additional_in_flight_s2g > 0) {
            combine_streaming_drain<RED_GROUP, STREAMING_BATCH, /*CHUNK_END=*/true>(
                streaming_pending,
                additional_in_flight_s2g,
                cumulative_produced,
                (volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter);
        }

        // Chunk-complete mbarrier (parity tracking).
        if constexpr (LSA_TEAMS != 1) {
            if (RED_GROUP::warp_rank() == 0) {
                if (cuda::ptx::elect_sync(~0)) {
                    cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->lsa_to_rdma_mbarrier_buffer
                                                    [rdma_tile_id * MAX_CHUNKS_PER_RANK + meta.chunk_id]);
                }
            }
        }
    }
}

// RDMA-put the active tokens (per rdma_to_attn_map) of one chunk to the remote LSA team, coalescing
// contiguous runs into batches of at most MAX_BATCH (token + optional prob).
// TOKEN_BYTES = Recipe::rdma_bytes. Every recipe that reaches here has a scale-free
// inter-node wire, so one put per batch always suffices.
template <
    bool STREAMING,
    bool BACKWARD_COMBINE,
    int MAX_BATCH,
    int MAX_TOKENS_PER_RANK,
    size_t TOKEN_BYTES>
__forceinline__ __device__ void combine_n2n_put_active_tokens(
    ncclGin& net,
    ncclTeam rail,
    int lteam_id,
    ncclWindow_t nccl_token_window,
    ncclWindow_t nccl_prob_window,
    ncclWindow_t nccl_internal_window,
    const struct combine_memory_region_info_t* smem_mr_info_ptr,
    const bool* rdma_to_attn_map,
    int chunk_base_token_idx,
    int chunk_first_token,
    int token_range,
    int rdma_tile_id,
    int rank_in_remote,
    int prob_dim,
    volatile uint32_t* streaming_counter,
    uint32_t& cumulative_sent) {
    int batch_start_in_chunk = -1;
    int batch_count = 0;
    for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
        bool need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
        bool is_last = (token_idx_in_chunk == token_range - 1);
        if (need_write) {
            if (batch_count == 0) batch_start_in_chunk = token_idx_in_chunk;
            batch_count++;
        }
        bool should_flush = batch_count > 0 && (!need_write || is_last || batch_count >= MAX_BATCH);
        if (should_flush) {
            if constexpr (STREAMING) {
                while (*streaming_counter < (cumulative_sent + batch_count)) {
                }
            }
            int batch_start_token = batch_start_in_chunk + chunk_first_token;
            size_t token_src_offset =
                smem_mr_info_ptr->combine_red_token_offset +
                (rdma_tile_id * MAX_TOKENS_PER_RANK + batch_start_token) * TOKEN_BYTES;
            size_t token_dst_offset =
                smem_mr_info_ptr->combine_g2s_token_offset +
                (rank_in_remote * MAX_TOKENS_PER_RANK + batch_start_token) * TOKEN_BYTES;
            net.put(
                rail,
                lteam_id,
                nccl_internal_window,
                token_dst_offset,
                nccl_token_window,
                token_src_offset,
                batch_count * TOKEN_BYTES,
                ncclGin_None{},
                ncclGin_None{},
                ncclCoopThread());

            if constexpr (BACKWARD_COMBINE) {
                size_t prob_src_offset =
                    smem_mr_info_ptr->combine_red_prob_offset +
                    (rdma_tile_id * MAX_TOKENS_PER_RANK + batch_start_token) * prob_dim * sizeof(float);
                size_t prob_dst_offset =
                    smem_mr_info_ptr->combine_g2s_prob_offset +
                    (rank_in_remote * MAX_TOKENS_PER_RANK + batch_start_token) * prob_dim * sizeof(float);
                net.put(
                    rail,
                    lteam_id,
                    nccl_internal_window,
                    prob_dst_offset,
                    nccl_prob_window,
                    prob_src_offset,
                    batch_count * prob_dim * sizeof(float),
                    ncclGin_None{},
                    ncclGin_None{},
                    ncclCoopThread());
            }

            if constexpr (STREAMING) {
                cumulative_sent += batch_count;
            }
            batch_count = 0;
            batch_start_in_chunk = -1;
        }
    }
}

// Signal the remote LSA team (via ncclGin) that this chunk has been delivered. Lane 0 only.
template <int MAX_TOKENS_PER_RANK, int TOKENS_PER_CHUNK, int LSA_TEAMS>
__forceinline__ __device__ void combine_n2n_signal_remote(
    ncclGin& net,
    ncclTeam rail,
    int lteam_id,
    int local_rank,
    int my_lteam,
    int chunk_id,
    unsigned signals_base,
    unsigned combine_signal_offset) {
    constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    unsigned signal_id = signals_base + combine_signal_offset +
                         local_rank * (LSA_TEAMS * MAX_CHUNKS_PER_RANK) + my_lteam * MAX_CHUNKS_PER_RANK +
                         chunk_id;
    net.signal(
        rail,
        lteam_id,
        ncclGin_SignalAdd{signal_id, 1},
        ncclCoopThread(),
        ncclGin_None{},
        cuda::thread_scope_thread,
        cuda::thread_scope_thread);
}

// Cross-LSA-team N2N (RDMA) warp group for the combine kernel. Exactly one such warp per block;
// uses the ncclGin API (net.put / net.signal).
// The put width is Recipe::rdma_bytes<kTokenDtype>(H); the hop carries no scale tail.
template <
    typename GIN_GROUP,
    typename SMEM_TYPE,
    int TOKENS_PER_CHUNK,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    int LSA_TEAM_SZ,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe>
__forceinline__ __device__ void combine_N2N_inter_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const struct combine_memory_region_info_t* mr_info,
    // SCRATCH
    SMEM_TYPE* smem_buffer_ptr,
    // CONFIG
    const int local_rank,
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    // CONFIG: ncclGin RDMA plumbing
    ncclDevComm_t* dcomms,
    ncclWindow_t nccl_token_window,
    ncclWindow_t nccl_prob_window,
    ncclWindow_t nccl_internal_window,
    int num_gin_comms,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    unsigned signals_base,
    unsigned combine_signal_offset) {
    // Token RDMA offsets/sizes below scale by size_u8 (4 B for FP32, 2 B for BF16/FP16);
    // prob is always float and is unaffected.
    // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
    using routing_loads_t = uint4;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(GIN_GROUP::size() == 32, "GIN_GROUP should be 1 warp.");
    static_assert(GIN_GROUP::size() >= LSA_TEAMS - 1, "mr_info should be loaded at once.");
    static_assert(
        TOKENS_PER_CHUNK % GIN_GROUP::size() == 0,
        "TOKENS_PER_CHUNK must be multiple of 32.");
    static_assert(
        TOKENS_PER_CHUNK % sizeof(routing_loads_t) == 0,
        "TOKENS_PER_CHUNK must be multiple of sizeof(routing_loads_t).");
    // mr_info and the intra-LSA -> rdma mbarrier buffer are staged in shared memory.
    struct combine_memory_region_info_t* smem_mr_info_ptr = nullptr;
    uint64_t* lsa_to_rdma_mbarrier_buffer_ptr = nullptr;
    constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    if constexpr (LSA_TEAMS != 1) {
        smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
        if (GIN_GROUP::thread_rank() == 0) {
            smem_mr_info_ptr[0] = mr_info[0];
        }
        lsa_to_rdma_mbarrier_buffer_ptr = smem_buffer_ptr->lsa_to_rdma_mbarrier_buffer;
    }
    __syncwarp();

    // Chunks this rank produces for the RDMA warps to consume; residue slots pad up to the max.
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = (LSA_TEAMS - 1) * MAX_CHUNKS_PER_RANK;
    // rdma_to_attn_map is padded to 16B (16 bools) per LSA team.
    const int rdma_per_lsa_sz = nccl_ep::align(num_of_tokens_per_rank, 16);
    uint32_t token_consumer_parity = 0;
    uint32_t cumulative_sent = 0; // Cumulative active tokens RDMA-put across all chunks (never reset).
    for (int cidx = blockIdx.x; cidx < total_chunks; cidx += NBLOCKS) {
        // LSA-team/chunk mapping shared with the G2S receiver so signals match.
        const auto meta = combine_chunk_meta<LSA_TEAMS, TOKENS_PER_CHUNK>(cidx, my_lteam, cpr, rem_chunk_sz);
        const int lteam_id = meta.lteam_id;
        const int chunk_id = meta.chunk_id;
        // With rail-scoped GIN comms, LSA-team index maps to rail-team rank.
        const int rank_in_remote = lteam_id < my_lteam ? my_lteam - 1 : my_lteam;
        const bool is_residue = (chunk_id >= cpr);

        // Distribute chunks across comms/contexts for parallelism.
        int total_channels = num_gin_comms * num_ctx_per_comm;
        int global_channel = chunk_id % total_channels;
        int comm_idx, ctx_idx;
        get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
        ncclGin net(dcomms[comm_idx], ctx_idx);
        ncclTeam rail = ncclTeamRail(dcomms[comm_idx]);
        int rdma_tile_id = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
        int chunk_base_token_idx = lteam_id * rdma_per_lsa_sz + chunk_id * TOKENS_PER_CHUNK;
        // Residue chunks carry no tokens; real chunks use the scheduled size (tail = remainder).
        const int token_range = is_residue ? 0 : meta.csize;
        constexpr int STREAMING_BATCH = NCCL_EP_HT_COMBINE_RDMA_STREAMING_BATCH;
        // Multi-node RDMA wire width (dtype-width for NONE). Must be a compile-time constant:
        // combine_n2n_put_active_tokens takes TOKEN_BYTES as a template argument.  The hop
        // carries no scale tail, so there is only ever one put per token batch.
        constexpr size_t token_bytes = static_cast<size_t>(
            nccl_ep::combine_recipe_traits<kCombineRecipe>::template rdma_bytes<kTokenDtype>(HIDDEN_DIM));
        static_assert(nccl_ep::combine_recipe_traits<kCombineRecipe>::template rdma_scale_bytes<kTokenDtype>(HIDDEN_DIM) == 0,
                      "combine RDMA wire must be scale-free");
        if constexpr (STREAMING_BATCH > 0) {
            // ---- STREAMING PATH: process tokens as reduction warp produces them ----
            // cumulative_sent tracks total active tokens across all chunks (no reset).

            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_put_active_tokens</*STREAMING=*/true, BACKWARD_COMBINE, STREAMING_BATCH,
                                              MAX_TOKENS_PER_RANK, token_bytes>(
                    net, rail, lteam_id, nccl_token_window, nccl_prob_window, nccl_internal_window, smem_mr_info_ptr,
                    rdma_to_attn_map, chunk_base_token_idx, chunk_id * TOKENS_PER_CHUNK, token_range,
                    rdma_tile_id, rank_in_remote, experts_per_rank * LSA_TEAM_SZ,
                    (volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter, cumulative_sent);
            }

            // Wait for mbarrier (parity tracking -- reduction warp always arrives)
            if (!is_residue) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    &lsa_to_rdma_mbarrier_buffer_ptr
                        [rdma_tile_id * MAX_CHUNKS_PER_RANK + chunk_id],
                    token_consumer_parity)) {
                }
            }

            // Signal remote
            __syncwarp();
            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_signal_remote<MAX_TOKENS_PER_RANK, TOKENS_PER_CHUNK, LSA_TEAMS>(
                    net, rail, lteam_id, local_rank, my_lteam, chunk_id, signals_base, combine_signal_offset);
            }
            __syncwarp();

            // No consumed handshake needed: cumulative counter never resets.

        } else {
            // ---- FALLBACK PATH (STREAMING_BATCH == 0): mbarrier-first, then put ----
            if (!is_residue) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    &lsa_to_rdma_mbarrier_buffer_ptr
                        [rdma_tile_id * MAX_CHUNKS_PER_RANK + chunk_id],
                    token_consumer_parity)) {
                }
            }

            if (!is_residue && GIN_GROUP::thread_rank() == 0) {
                constexpr int max_batch = NCCL_EP_HT_DISPATCH_RDMA_BATCH_SIZE;
                combine_n2n_put_active_tokens</*STREAMING=*/false, BACKWARD_COMBINE, max_batch,
                                              MAX_TOKENS_PER_RANK, token_bytes>(
                    net, rail, lteam_id, nccl_token_window, nccl_prob_window, nccl_internal_window, smem_mr_info_ptr,
                    rdma_to_attn_map, chunk_base_token_idx, chunk_id * TOKENS_PER_CHUNK, token_range,
                    rdma_tile_id, rank_in_remote, experts_per_rank * LSA_TEAM_SZ,
                    nullptr, cumulative_sent);
            }

            // Signal remote
            __syncwarp();
            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_signal_remote<MAX_TOKENS_PER_RANK, TOKENS_PER_CHUNK, LSA_TEAMS>(
                    net, rail, lteam_id, local_rank, my_lteam, chunk_id, signals_base, combine_signal_offset);
            }
            __syncwarp();
        }
    }
    token_consumer_parity ^= 1;
}

// One warp lane's remote LSA team for the combine cross-LSA-team RDMA tier and whether that LSA team
// needs the current dst token. tile_id indexes the per-remote-LSA-team RDMA buffers.
struct rdma_lane_t {
    bool valid;
    int tile_id;
};

// Map lane_id -> remote LSA team (lane 0 -> my_lteam-1, wrapping past self) and read its routing bit.
template <int LSA_TEAMS>
__forceinline__ __device__ rdma_lane_t
combine_g2s_resolve_rdma_lane(int lane_id, int my_lteam, const bool* attn_to_rdma_addr) {
    const int n = lane_id + 1;
    if (n >= LSA_TEAMS) {
        return rdma_lane_t{false, 0};
    }
    const int lteam_id = (my_lteam + LSA_TEAMS - n) % LSA_TEAMS;
    const int tile_id = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
    return rdma_lane_t{attn_to_rdma_addr[tile_id], tile_id};
}

// Token (+prob) source pointers for one remote token in the RDMA cross-LSA-team group buffers.
// No scale pointer: the RDMA hop is scale-free for every recipe.
struct g2s_src_t {
    const uint16_t* token_src;
    const float* prob_src;
};

template <bool BACKWARD_COMBINE, int HIDDEN_DIM, ncclDataType_t kTokenDtype,
          int MAX_NUM_OF_TOKENS_PER_RANK,
          ncclEpCombQuant_t kCombineRecipe>
__forceinline__ __device__ g2s_src_t combine_g2s_resolve_rdma_source(
    const uint16_t* combine_gin_G2S_tokens,
    const float* combine_gin_G2S_prob,
    int tile_id,
    int flat_token_id,
    int experts_per_rank,
    int lteam_sz) {
    const int rdma_row = tile_id * MAX_NUM_OF_TOKENS_PER_RANK + flat_token_id;
    // rdma_bytes is the inter-node row width (dtype-width for NONE), matching the previous
    // HIDDEN_DIM * size_u16<kTokenDtype>() form for every unquantized case.
    const int rdma_wire_stride =
        nccl_ep::combine_recipe_traits<kCombineRecipe>::template rdma_bytes<kTokenDtype>(HIDDEN_DIM);
    const uint16_t* token_src = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const uint8_t*>(combine_gin_G2S_tokens) +
        static_cast<size_t>(rdma_row) * rdma_wire_stride);
    const float* prob_src = nullptr;
    if constexpr (BACKWARD_COMBINE) {
        prob_src = combine_gin_G2S_prob + rdma_row * (experts_per_rank * lteam_sz);
    }
    return g2s_src_t{token_src, prob_src};
}

// Cooperatively issue every remote LSA team's src token for one dst token into this warp's G2S ring.
// Each lane owns one remote LSA team; valid lanes issue in parallel to distinct stages, batched by
// ring_len so parity resolves cleanly when valid remotes exceed the ring depth (RED consumes
// stages sequentially). Advances global_offset by the number of entries issued. Mirrors the
// LOCAL tier's issue_local_g2s_row.
template <
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    ncclEpCombQuant_t kCombineRecipe,
    typename SMEM_TYPE>
__forceinline__ __device__ void issue_rdma_g2s_row(
    SMEM_TYPE* smem_buffer_ptr,
    const bool* attn_to_rdma_addr,
    const uint16_t* combine_gin_G2S_tokens,
    const float* combine_gin_G2S_prob,
    int flat_token_id,
    int& global_offset,
    int starting_G2S_index,
    int ring_len,
    int lane_id,
    int my_lteam,
    uint32_t token_bytes,
    uint32_t prob_bytes,
    int experts_per_rank,
    int lteam_sz) {
    static_assert(
        LSA_TEAMS <= 33, "LSA_TEAMS must fit in a single warp pass for RDMA parallelization.");
    // Ensure all local TMAs are committed before RDMA.
    __syncwarp(0xffffffff);

    const rdma_lane_t lane = combine_g2s_resolve_rdma_lane<LSA_TEAMS>(lane_id, my_lteam, attn_to_rdma_addr);

    // Valid-entry count/rank -- identical across lanes.
    const unsigned rdma_valid_mask = __ballot_sync(0xffffffff, lane.valid);
    const int rdma_valid_count = __popc(rdma_valid_mask);
    if (rdma_valid_count == 0) {
        return;
    }
    const int rdma_local_rank = __popc(rdma_valid_mask & ((1u << lane_id) - 1));

    int rdma_ranks_issued = 0;
    while (rdma_ranks_issued < rdma_valid_count) {
        const int batch_end = (rdma_ranks_issued + ring_len < rdma_valid_count) ? rdma_ranks_issued + ring_len :
                                                                                  rdma_valid_count;

        const bool in_batch = lane.valid && rdma_local_rank >= rdma_ranks_issued && rdma_local_rank < batch_end;
        if (in_batch) {
            const int rank_in_batch = rdma_local_rank - rdma_ranks_issued;
            const g2s_src_t src =
                combine_g2s_resolve_rdma_source<BACKWARD_COMBINE, HIDDEN_DIM, kTokenDtype,
                                                MAX_TOKENS_PER_RANK, kCombineRecipe>(
                    combine_gin_G2S_tokens,
                    combine_gin_G2S_prob,
                    lane.tile_id,
                    flat_token_id,
                    experts_per_rank,
                    lteam_sz);
            // RDMA tier does not set cross_lsa_flag_G2S_buffer; the RED group reads
            // attn_to_rdma_map to demarcate RDMA entries.
            issue_g2s_entry</*CROSS_LSA=*/true, BACKWARD_COMBINE, /*WRITE_LAST_FLAG=*/false>(
                smem_buffer_ptr,
                global_offset,
                rank_in_batch,
                starting_G2S_index,
                ring_len,
                src.token_src,
                token_bytes,
                src.prob_src,
                prob_bytes,
                /*is_last_entry=*/false);
        }

        global_offset += (batch_end - rdma_ranks_issued);
        rdma_ranks_issued = batch_end;
        // Prevent non-in_batch lanes racing ahead of in_batch lanes' wait_parity+TMA+arrive.
        __syncwarp(0xffffffff);
    }
}

// Cross-LSA-team G2S warp group for the combine kernel.
template <
    typename SMEM_TYPE,
    typename G2S_GROUP,
    int STAGES_G2S,
    int TOKENS_PER_CHUNK,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    int TOKENS_PER_GROUP,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SZ,
    ncclEpLayout_t kLayout,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe>
__forceinline__ __device__ void combine_G2S_inter_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const bool* attn_to_rdma_map,
    const int32_t* sparse_to_dense_map,
    const void* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    const uint16_t* combine_gin_G2S_tokens,
    const float* combine_gin_G2S_prob,
    // OUTPUT
    uint64_t* gin_G2S_flags,
    SMEM_TYPE* smem_buffer_ptr,
    // CONFIG
    const int local_rank,
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    const uint64_t expected_flag_value,
    const bool combine_local_reduce_enabled,
    // CONFIG: ncclGin RDMA plumbing
    ncclDevComm_t* dcomms,
    unsigned signals_base,
    unsigned combine_signal_offset,
    int num_gin_comms,
    int num_ctx_per_comm) {
    // The G2S group is split into single-warp pipelines (warp == pipeline), matching the cross-LSA-team red
    // group; each warp owns an equal slice of the G2S FIFO.
    static_assert(
        STAGES_G2S % G2S_GROUP::warp_size() == 0,
        "STAGES_G2S must be a multiple of the inter-node G2S warp count.");
    constexpr int STAGES_G2S_PER_WARP = STAGES_G2S / G2S_GROUP::warp_size();
    static_assert(
        TOKENS_PER_CHUNK % TOKENS_PER_GROUP == 0,
        "TOKENS_PER_CHUNK must be a multiple of TOKENS_PER_GROUP.");
    constexpr int TOKEN_GROUPS_PER_CHUNK = TOKENS_PER_CHUNK / TOKENS_PER_GROUP;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // Produces the local rank's output chunks in order; RDMA feeds matching chunk IDs from
    // LSA team-1, ..., LSA team+1, so src chunks arrive in the order the RED group consumes them.
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int max_chunks_per_rank = nccl_ep::ceil_div(MAX_TOKENS_PER_RANK, TOKENS_PER_CHUNK);
    const int total_chunks = cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per LSA team.
    const int rdma_per_lsa_sz = nccl_ep::align(num_of_tokens_per_rank, 16);
    // This warp's G2S FIFO sub-range.
    const int starting_G2S_index = STAGES_G2S_PER_WARP * G2S_GROUP::warp_rank();
    const int ending_G2S_index = STAGES_G2S_PER_WARP * (G2S_GROUP::warp_rank() + 1);

    // Unified body for NVLink-only (LSA_TEAMS==1) and RDMA+NVLink (>1).
    //   LOCAL tier: warp-cooperative s2d-row scan -> issue_local_g2s_row.
    //   RDMA tier (>1): chunk-level ncclGin signal pre-wait, warp-parallel per-LSA-team TMA issue, residue flags.
    // Parity: global_offset counts stages filled this warp; entry rank R lands at
    //   stage = starting + (global_offset+R) % ring_len, parity = 1 ^ (((global_offset+R)/ring_len) & 1),
    // matching RED's sequential consumption.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    const int ring_len = ending_G2S_index - starting_G2S_index;
    using Recipe = nccl_ep::combine_recipe_traits<kCombineRecipe>;
    // Intra-leg (local NVLink) rows are always packed [FP8 H | E8M0 H/32] for MXFP8: the
    // prologue emits that layout, so one TMA per token carries values and scales together.
    const uint32_t local_token_bytes =
        (uint32_t)Recipe::template packed_bytes<kTokenDtype>(HIDDEN_DIM);
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SZ) * sizeof(float));

    // EM unfused-combine dedup uses __shfl_up_sync(1), requiring s2d_inner_dim <= WARP_SIZE.
    // combine_local_reduce_enabled is only true under EM unfused combine (no-op for FLAT).
    if (combine_local_reduce_enabled && lane_id == 0 && smem_buffer_ptr->s2d_inner_dim > WARP_SIZE) {
        __trap();
    }

    // Total stages filled across all tokens (local + RDMA).
    int global_offset = 0;

    for (int cidx = 0; cidx < total_chunks; cidx++) {
        const bool is_tail = (rem_chunk_sz != 0 && cidx == cpr - 1);
        const int csize = is_tail ? rem_chunk_sz : TOKENS_PER_CHUNK;
        const int token_groups_for_chunk =
            is_tail ? nccl_ep::ceil_div(rem_chunk_sz, TOKENS_PER_GROUP) : TOKEN_GROUPS_PER_CHUNK;

        const bool* rdma_to_attn_map_base =
            rdma_to_attn_map + (my_lteam * rdma_per_lsa_sz + cidx * TOKENS_PER_CHUNK);
        const int s2d_entries = smem_buffer_ptr->s2d_inner_dim;
        const int32_t* sparse_to_dense_map_base =
            sparse_to_dense_map + (my_lteam * num_of_tokens_per_rank + cidx * TOKENS_PER_CHUNK) * s2d_entries;

        // Chunk-level RDMA pre-wait: wait ncclGin signals once so the per-token RDMA tier issues without
        // per-token waits. Multi-domain only.
        const bool* attn_to_rdma_map_base = nullptr;
        bool rdma_flag_clear[LSA_TEAMS];
        if constexpr (LSA_TEAMS > 1) {
            attn_to_rdma_map_base = attn_to_rdma_map + (cidx * TOKENS_PER_CHUNK) * (LSA_TEAMS - 1);

            if (lane_id == 0) {
                constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
                int total_channels = num_gin_comms * num_ctx_per_comm;
                int global_channel = cidx % total_channels;
                int comm_idx, ctx_idx;
                get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
                ncclGin net(dcomms[comm_idx], ctx_idx);
                for (int n = 1; n < LSA_TEAMS; n++) {
                    int signal_lteam_id = my_lteam >= n ? my_lteam - n : my_lteam + LSA_TEAMS - n;
                    unsigned signal_id = signals_base + combine_signal_offset +
                                         local_rank * (LSA_TEAMS * MAX_CHUNKS_PER_RANK) +
                                         signal_lteam_id * MAX_CHUNKS_PER_RANK + cidx;
                    net.waitSignal(ncclCoopThread(), signal_id, expected_flag_value);
                }
            }
            __syncwarp(0xffffffff);

#pragma unroll
            for (int jj = 0; jj < LSA_TEAMS; ++jj) {
                rdma_flag_clear[jj] = true;
            }
        }

        for (int group_idx = blockIdx.x; group_idx < token_groups_for_chunk; group_idx += NBLOCKS) {
            for (int token_in_group = G2S_GROUP::warp_rank(); token_in_group < TOKENS_PER_GROUP;
                 token_in_group += G2S_GROUP::warp_size()) {
                int cur_tokid = group_idx * TOKENS_PER_GROUP + token_in_group;
                if (cur_tokid >= csize) {
                    break;
                }

                // Uniform across the warp. No early continue: the RDMA tier still runs when false.
                bool reqired_by_local_lsa = rdma_to_attn_map_base[cur_tokid];

                // LOCAL tier: warp-cooperative s2d-row scan with inline broadcast-issue. Advances
                // global_offset by the entries issued, keeping later reads consistent on every lane.
                if (reqired_by_local_lsa) {
                    const int32_t* sparse_to_dense_row =
                        sparse_to_dense_map_base + (group_idx * TOKENS_PER_GROUP + token_in_group) * s2d_entries;
                    issue_local_g2s_row</*CROSS_LSA=*/true, BACKWARD_COMBINE, kLayout, HIDDEN_DIM, kTokenDtype>(
                        smem_buffer_ptr,
                        sparse_to_dense_row,
                        s2d_entries,
                        global_offset,
                        starting_G2S_index,
                        ring_len,
                        lane_id,
                        remote_expert_input_token,
                        remote_expert_input_prob,
                        local_token_bytes,
                        prob_bytes,
                        experts_per_rank,
                        LSA_TEAM_SZ,
                        combine_local_reduce_enabled);
                }

                // RDMA tier: each lane maps to a remote LSA team; valid lanes issue TMAs in parallel to
                // distinct stages, batched by ring_len so parity resolves cleanly.
                if constexpr (LSA_TEAMS > 1) {
                    // Inter-node RDMA leg width (dtype-width for NONE); scale-free, so
                    // issue_rdma_g2s_row takes no scale width. Declared inside the gate:
                    // this warp is instantiated for LSA_TEAMS == 1 as well, and a recipe
                    // with no RDMA leg (MXFP8) defines no rdma_bytes to call.
                    const uint32_t rdma_token_bytes =
                        (uint32_t)Recipe::template rdma_bytes<kTokenDtype>(HIDDEN_DIM);
                    const int flat_token_id =
                        cidx * TOKENS_PER_CHUNK + group_idx * TOKENS_PER_GROUP + token_in_group;
                    const bool* attn_to_rdma_addr =
                        attn_to_rdma_map_base + (group_idx * TOKENS_PER_GROUP + token_in_group) * (LSA_TEAMS - 1);
                    issue_rdma_g2s_row<
                        BACKWARD_COMBINE,
                        HIDDEN_DIM,
                        kTokenDtype,
                        MAX_TOKENS_PER_RANK,
                        LSA_TEAMS,
                        kCombineRecipe>(
                        smem_buffer_ptr,
                        attn_to_rdma_addr,
                        combine_gin_G2S_tokens,
                        combine_gin_G2S_prob,
                        flat_token_id,
                        global_offset,
                        starting_G2S_index,
                        ring_len,
                        lane_id,
                        my_lteam,
                        rdma_token_bytes,
                        prob_bytes,
                        experts_per_rank,
                        LSA_TEAM_SZ);
                }
            }
        }
    }
    if constexpr (LSA_TEAMS > 1) {
        // Update residue flags for the chunks this rank did not produce.
        int residue_flag_count = max_chunks_per_rank - cpr;
        for (int lteam_id = blockIdx.x; lteam_id < LSA_TEAMS - 1; lteam_id += gridDim.x) {
            uint64_t* residue_flag_base =
                gin_G2S_flags + (lteam_id * max_chunks_per_rank + cpr);
            for (int flag_id = G2S_GROUP::thread_rank(); flag_id < residue_flag_count;
                 flag_id += G2S_GROUP::size()) {
                residue_flag_base[flag_id] = expected_flag_value;
            }
        }
    }
}

// Consume one cross-LSA-team G2S stage into the FP32 accumulators: token is accumulated; prob is written
// into LSA-team slot `prob_slot` (accumulated for the local phase, assigned for remote phases). Advances
// this pipeline's G2S cursor/parity. Returns the producer's last-src flag when READ_LAST_FLAG.
template <
    int THRDS_PER_PIPELINE,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    bool READ_LAST_FLAG,
    bool ACCUMULATE_PROB,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe,
    typename SMEM_TYPE,
    int NUM_ACC>
__forceinline__ __device__ bool combine_inter_consume_src(
    SMEM_TYPE* smem_buffer_ptr,
    int& token_stage,
    uint32_t& token_producer_parity,
    int starting_G2S_index,
    int ending_G2S_index,
    int warp_rank_within_pipeline,
    int thread_rank_within_pipeline,
    int pipeline_rank,
    float2 (&acc_token_fp32)[NUM_ACC],
    float* acc_prob_ptr,
    int prob_slot,
    int prob_vec_per_thread,
    int prob_dim) {
    constexpr int BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

    __nv_bfloat162* load_token_base_ptr =
        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_cross_lsa_token_G2S(token_stage));
    float* load_prob_base_ptr;
    if constexpr (BACKWARD_COMBINE) {
        load_prob_base_ptr = smem_buffer_ptr->get_cross_lsa_prob_G2S(token_stage);
    }

    // Wait until this src token is staged in SMEM, then let the whole pipeline read it.
    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            while (!cuda::ptx::mbarrier_try_wait_parity(
                smem_buffer_ptr->get_cross_lsa_mbarrier_G2S_producer(token_stage), token_producer_parity)) {
            }
        }
    }
    arrive_and_wait(THRDS_PER_PIPELINE, 2 + pipeline_rank);

    // MXFP8 stages are always the packed NVLink row: the recipe is single-LSA-team only, and
    // the RDMA consumer below is instantiated only for LSA_TEAMS > 1. BF16/NONE takes the
    // strided load_pair path at the end of this function.
    static_assert(kCombineRecipe != NCCL_EP_COMB_QUANT_MXFP8 || READ_LAST_FLAG,
                  "MXFP8 is NVLink-only; it must never consume an RDMA-staged row");
    if constexpr (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) {
        using Map = mxfp8_red_map<NUM_ACC>;
        // One 16B FP8 load feeds kPairs accumulators and shares a single E8M0 read.
        constexpr int kPairs = Map::kPairs;
        const uint16_t* fp8_data = reinterpret_cast<const uint16_t*>(
            smem_buffer_ptr->get_cross_lsa_token_G2S(token_stage));
        const uint8_t* scale_data =
            reinterpret_cast<const uint8_t*>(smem_buffer_ptr->get_cross_lsa_token_G2S(token_stage)) +
            HIDDEN_DIM;
#pragma unroll
        for (int base = 0; base < NUM_ACC; base += kPairs) {
            const int pair0 =
                Map::vec_base(base, THRDS_PER_PIPELINE, thread_rank_within_pipeline);
            if (pair0 + kPairs <= BF16X2_ELEMENTS_PER_TOKEN) {
                uint16_t fp8x2[kPairs];
                ld_fp8_pairs_shared<kPairs>(fp8x2, fp8_data + pair0);
                const float scale_f = nccl_ep::mxfp8::e8m0_to_scale(scale_data[pair0 / 16]);
#pragma unroll
                for (int t = 0; t < kPairs; t++) {
                    const float2 src_data_fp32 =
                        nccl_ep::mxfp8::dequant_x2_scaled(fp8x2[t], scale_f);
                    acc_token_fp32[base + t].x += src_data_fp32.x;
                    acc_token_fp32[base + t].y += src_data_fp32.y;
                }
            } else {
                // Tail: fewer than kPairs pairs left, so fall back to scalar decode.
#pragma unroll
                for (int t = 0; t < kPairs; t++) {
                    const int pair = pair0 + t;
                    if (pair < BF16X2_ELEMENTS_PER_TOKEN) {
                        const float scale_f =
                            nccl_ep::mxfp8::e8m0_to_scale(scale_data[pair / 16]);
                        const float2 src_data_fp32 =
                            nccl_ep::mxfp8::dequant_x2_scaled(fp8_data[pair], scale_f);
                        acc_token_fp32[base + t].x += src_data_fp32.x;
                        acc_token_fp32[base + t].y += src_data_fp32.y;
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int n = 0; n < NUM_ACC; n++) {
            int element_id = (n * THRDS_PER_PIPELINE) + thread_rank_within_pipeline;
            if (element_id < BF16X2_ELEMENTS_PER_TOKEN) {
                float2 src_data_fp32 =
                    nccl_ep::combine_recipe_traits<kCombineRecipe>::template load_pair<kTokenDtype>(
                        load_token_base_ptr, element_id);
                acc_token_fp32[n].x += src_data_fp32.x;
                acc_token_fp32[n].y += src_data_fp32.y;
            }
        }
    }
    if constexpr (BACKWARD_COMBINE) {
#pragma unroll
        for (int pv = 0; pv < prob_vec_per_thread; pv++) {
            int element_id = thread_rank_within_pipeline + pv * THRDS_PER_PIPELINE;
            if (element_id < prob_dim) {
                float src_data = load_prob_base_ptr[element_id];
                if constexpr (ACCUMULATE_PROB) {
                    acc_prob_ptr[prob_slot * prob_vec_per_thread + pv] += src_data;
                } else {
                    acc_prob_ptr[prob_slot * prob_vec_per_thread + pv] = src_data;
                }
            }
        }
    }

    bool last_src_token = false;
    if constexpr (READ_LAST_FLAG) {
        last_src_token = smem_buffer_ptr->cross_lsa_flag_G2S_buffer[token_stage];
    }

    // All pipeline threads finish reading before the producer reuses this stage.
    arrive_and_wait(THRDS_PER_PIPELINE, 2 + pipeline_rank);
    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_cross_lsa_mbarrier_G2S_consumer(token_stage));
        }
    }

    token_stage += 1;
    if (token_stage == ending_G2S_index) {
        token_stage = starting_G2S_index;
        token_producer_parity ^= 1;
    }
    return last_src_token;
}

// Store one reduced dst token (+per-LSA-team prob) from FP32 registers into an S2G SMEM stage and TMA-copy
// it to the attn output. Advances this pipeline's S2G cursor.
template <
    int THRDS_PER_PIPELINE,
    int S2G_STAGES_PER_PIPELINE,
    int LSA_TEAMS,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe,
    typename SMEM_TYPE,
    int NUM_ACC>
__forceinline__ __device__ void combine_inter_store_token(
    SMEM_TYPE* smem_buffer_ptr,
    int& dst_token_stage,
    int starting_S2G_index,
    int ending_S2G_index,
    int warp_rank_within_pipeline,
    int thread_rank_within_pipeline,
    int pipeline_rank,
    const float2 (&acc_token_fp32)[NUM_ACC],
    const float* acc_prob_ptr,
    int prob_vec_per_thread,
    int prob_dim,
    int my_lteam,
    uint16_t* attn_output_token_base,
    float* attn_output_prob_base,
    size_t out_token_stride_u16,
    int token_in_chunk,
    int absolute_token_id,
    int num_real_tokens) {
    constexpr int BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

    __nv_bfloat162* store_token_base_ptr =
        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_cross_lsa_token_S2G(dst_token_stage));

    // Wait for prior TMA reads of this S2G stage to finish before overwriting it.
    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<S2G_STAGES_PER_PIPELINE - 1>{});
        }
    }
    arrive_and_wait(THRDS_PER_PIPELINE, 2 + pipeline_rank);

    if constexpr (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) {
        // Must mirror combine_inter_consume_src: MXFP8 owns contiguous vectors, else strided.
        // Contiguity means a group of kPairs accumulators is one run, so it goes out as 16B
        // stores rather than kPairs 4B ones.
        using Map = mxfp8_red_map<NUM_ACC>;
        constexpr int kPairs = Map::kPairs;
        static_assert(NUM_ACC % kPairs == 0, "mxfp8_red_map picks kPairs to tile NUM_ACC");
#pragma unroll
        for (int acc_base = 0; acc_base < NUM_ACC; acc_base += kPairs) {
            const int first = Map::vec_base(acc_base, THRDS_PER_PIPELINE, thread_rank_within_pipeline);
            if (first + kPairs <= BF16X2_ELEMENTS_PER_TOKEN) {
                st_token_pairs_shared<kPairs, kTokenDtype>(
                    store_token_base_ptr, first, &acc_token_fp32[acc_base]);
            } else {
                // Tail: fewer than kPairs pairs left, so store them one at a time.
#pragma unroll
                for (int t = 0; t < kPairs; t++) {
                    if (first + t < BF16X2_ELEMENTS_PER_TOKEN) {
                        nccl_ep::st_token_pair<kTokenDtype>(
                            store_token_base_ptr, first + t, acc_token_fp32[acc_base + t]);
                    }
                }
            }
        }

    } else {
#pragma unroll
        for (int n = 0; n < NUM_ACC; n++) {
            const int element_id = (n * THRDS_PER_PIPELINE) + thread_rank_within_pipeline;
            if (element_id < BF16X2_ELEMENTS_PER_TOKEN) {
                nccl_ep::st_token_pair<kTokenDtype>(store_token_base_ptr, element_id, acc_token_fp32[n]);
            }
        }
    }
    if constexpr (BACKWARD_COMBINE) {
        float* store_prob_base_ptr = smem_buffer_ptr->get_cross_lsa_prob_S2G(dst_token_stage);
        // Gather per-source-LSA-team prob into output LSA-team order (my_lteam, my_lteam-1, ...).
#pragma unroll
        for (int n = 0; n < LSA_TEAMS; n++) {
            int output_lteam_id = (my_lteam - n) >= 0 ? my_lteam - n : my_lteam + LSA_TEAMS - n;
            int element_base_id = output_lteam_id * prob_dim;
#pragma unroll
            for (int m = 0; m < prob_vec_per_thread; m++) {
                int element_id = thread_rank_within_pipeline + m * THRDS_PER_PIPELINE;
                if (element_id < prob_dim) {
                    store_prob_base_ptr[element_base_id + element_id] = acc_prob_ptr[n * prob_vec_per_thread + m];
                }
            }
        }
    }

    // Publish SMEM writes to the async proxy, then sync the pipeline before the TMA launch.
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    arrive_and_wait(THRDS_PER_PIPELINE, 2 + pipeline_rank);

    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0) && absolute_token_id < num_real_tokens) {
            uint16_t* current_token_addr = attn_output_token_base + (size_t)token_in_chunk * out_token_stride_u16;
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_global,
                cuda::ptx::space_shared,
                reinterpret_cast<void*>(current_token_addr),
                reinterpret_cast<const void*>(smem_buffer_ptr->get_cross_lsa_token_S2G(dst_token_stage)),
                (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>())));

            if constexpr (BACKWARD_COMBINE) {
                float* current_prob_addr = attn_output_prob_base + token_in_chunk * (prob_dim * LSA_TEAMS);
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    reinterpret_cast<void*>(current_prob_addr),
                    reinterpret_cast<const void*>(smem_buffer_ptr->get_cross_lsa_prob_S2G(dst_token_stage)),
                    (uint32_t)((prob_dim * LSA_TEAMS) * sizeof(float)));
            }
            cuda::ptx::cp_async_bulk_commit_group();
        }
    }

    dst_token_stage += 1;
    if (dst_token_stage == ending_S2G_index) {
        dst_token_stage = starting_S2G_index;
    }
}

// Cross-LSA-team reduction warp group for the combine kernel.
template <
    typename SMEM_TYPE,
    typename RED_GROUP,
    int PIPELINES_PER_BLOCK,
    int STAGES_G2S,
    int STAGES_S2G,
    int TOKENS_PER_CHUNK,
    int LSA_TEAMS,
    int NBLOCKS,
    int TOKENS_PER_GROUP,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SZ,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe>
__forceinline__ __device__ void combine_RED_inter_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const bool* attn_to_rdma_map,
    // OUTPUT
    uint16_t* attn_output_token,
    float* attn_output_prob,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int num_real_tokens,
    const int experts_per_rank,
    SMEM_TYPE* smem_buffer_ptr) {
    // The warp group is split into PIPELINES_PER_BLOCK independent pipelines (matching the
    // cross-LSA-team G2S group); each pipeline owns an equal slice of the G2S/S2G FIFOs.
    static_assert(
        RED_GROUP::warp_size() % PIPELINES_PER_BLOCK == 0,
        "Inter-node red warp count must be a multiple of PIPELINES_PER_BLOCK.");
    constexpr int WARP_SIZE = 32;
    constexpr int THRDS_PER_PIPELINE =
        (RED_GROUP::warp_size() / PIPELINES_PER_BLOCK) * WARP_SIZE;
    static_assert(
        STAGES_G2S % PIPELINES_PER_BLOCK == 0,
        "STAGES_G2S must be a multiple of PIPELINES_PER_BLOCK.");
    constexpr int STAGES_G2S_PER_PIPELINE = STAGES_G2S / PIPELINES_PER_BLOCK;
    static_assert(
        STAGES_S2G % PIPELINES_PER_BLOCK == 0,
        "STAGES_S2G must be a multiple of PIPELINES_PER_BLOCK.");
    constexpr int STAGES_S2G_PER_PIPELINE = STAGES_S2G / PIPELINES_PER_BLOCK;
    // Output chunks are split into token groups striped across blocks (unlike the chunk-per-block stages).
    static_assert(
        TOKENS_PER_CHUNK % TOKENS_PER_GROUP == 0,
        "TOKENS_PER_CHUNK must be a multiple of TOKENS_PER_GROUP.");
    constexpr int TOKEN_GROUPS_PER_CHUNK = TOKENS_PER_CHUNK / TOKENS_PER_GROUP;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // Tokens reduced as BF16x2 in FP32 (HIDDEN_DIM even); prob stays in float.
    constexpr int BF16X2_ELEMS_PER_TOKEN = HIDDEN_DIM / 2;
    constexpr int ACC_ELEMS_PER_THRD =
        nccl_ep::ceil_div(BF16X2_ELEMS_PER_TOKEN, THRDS_PER_PIPELINE);
    const int prob_dim = experts_per_rank * LSA_TEAM_SZ;
    const int prob_vec_per_thread = nccl_ep::ceil_div(prob_dim, THRDS_PER_PIPELINE);
    // Compile-time upper bound sized exactly to this instantiation's LSA team.
    constexpr int MAX_PROB_ELEMS_PER_THRD =
        nccl_ep::ceil_div(NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SZ, THRDS_PER_PIPELINE);

    // Each block produces token groups of the local rank's output chunks in order; the RDMA network feeds
    // matching chunk IDs from LSA team-1, LSA team-2, ..., LSA team+1, so src chunks arrive in the same order.
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per LSA team.
    const int rdma_per_lsa_sz = nccl_ep::align(num_of_tokens_per_rank, 16);
    // Dtype-aware per-token output stride in uint16_t units.
    const size_t out_token_stride_u16 = (size_t)HIDDEN_DIM * nccl_ep::size_u8<kTokenDtype>() / sizeof(uint16_t);

    // This thread's pipeline placement and G2S/S2G FIFO sub-range.
    const int pipeline_rank = RED_GROUP::thread_rank() / THRDS_PER_PIPELINE;
    const int thread_rank_within_pipeline = RED_GROUP::thread_rank() % THRDS_PER_PIPELINE;
    const int warp_rank_within_pipeline = thread_rank_within_pipeline / WARP_SIZE;
    const int starting_G2S_index = STAGES_G2S_PER_PIPELINE * pipeline_rank;
    const int ending_G2S_index = STAGES_G2S_PER_PIPELINE * (pipeline_rank + 1);
    int token_stage = starting_G2S_index;
    uint32_t token_producer_parity = 0;
    const int starting_S2G_index = STAGES_S2G_PER_PIPELINE * pipeline_rank;
    const int ending_S2G_index = STAGES_S2G_PER_PIPELINE * (pipeline_rank + 1);
    int dst_token_stage = starting_S2G_index;

    for (int cidx = 0; cidx < total_chunks; cidx++) {
        const bool is_tail = (rem_chunk_sz != 0 && cidx == cpr - 1);
        const int csize = is_tail ? rem_chunk_sz : TOKENS_PER_CHUNK;
        const int token_groups_for_chunk =
            is_tail ? nccl_ep::ceil_div(rem_chunk_sz, TOKENS_PER_GROUP) : TOKEN_GROUPS_PER_CHUNK;

        const bool* rdma_to_attn_map_base =
            rdma_to_attn_map + (my_lteam * rdma_per_lsa_sz + cidx * TOKENS_PER_CHUNK);
        const bool* attn_to_rdma_map_base = nullptr;
        if constexpr (LSA_TEAMS > 1) {
            attn_to_rdma_map_base = attn_to_rdma_map + (cidx * TOKENS_PER_CHUNK) * (LSA_TEAMS - 1);
        }
        uint16_t* attn_output_token_base =
            attn_output_token + (size_t)(cidx * TOKENS_PER_CHUNK) * out_token_stride_u16;
        float* attn_output_prob_base = nullptr;
        if constexpr (BACKWARD_COMBINE) {
            attn_output_prob_base =
                attn_output_prob + (cidx * TOKENS_PER_CHUNK) * (prob_dim * LSA_TEAMS);
        }

        // Token groups are striped across blocks; each pipeline handles a round-robin slice of dst tokens.
        for (int group_idx = blockIdx.x; group_idx < token_groups_for_chunk; group_idx += NBLOCKS) {
            for (int token_in_group = pipeline_rank; token_in_group < TOKENS_PER_GROUP;
                 token_in_group += PIPELINES_PER_BLOCK) {
                int cur_tokid = group_idx * TOKENS_PER_GROUP + token_in_group;
                if (cur_tokid >= csize) {
                    break;
                }

                // Each dst token accumulates local-LSA-team src tokens (like intra reduction) then remote-LSA-team
                // RDMA src tokens. Prob is gathered per source LSA team: slot 0 = local, 1..LSA-1 = remote.
                float2 acc_token_fp32[ACC_ELEMS_PER_THRD];
                using acc_prob_storage_type =
                    acc_prob_storage_t<BACKWARD_COMBINE, LSA_TEAMS * MAX_PROB_ELEMS_PER_THRD>;
                [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
                [[maybe_unused]] float* acc_prob_ptr = nullptr;
                if constexpr (BACKWARD_COMBINE) {
                    acc_prob_ptr = acc_prob_storage.data;
                }
#pragma unroll
                for (int n = 0; n < ACC_ELEMS_PER_THRD; n++) {
                    acc_token_fp32[n].x = 0.0f;
                    acc_token_fp32[n].y = 0.0f;
                }
                if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                    for (int n = 0; n < LSA_TEAMS; n++) {
                        for (int m = 0; m < prob_vec_per_thread; m++) {
                            acc_prob_ptr[n * prob_vec_per_thread + m] = 0.0f;
                        }
                    }
                }

                // Local-LSA-team accumulation: consume staged src tokens until the producer marks the last one.
                if (rdma_to_attn_map_base[cur_tokid]) {
                    bool last_local_src = false;
                    do {
                        last_local_src = combine_inter_consume_src<
                            THRDS_PER_PIPELINE,
                            BACKWARD_COMBINE,
                            HIDDEN_DIM,
                            /*READ_LAST_FLAG=*/true,
                            /*ACCUMULATE_PROB=*/true,
                            kTokenDtype,
                            kCombineRecipe>(
                            smem_buffer_ptr,
                            token_stage,
                            token_producer_parity,
                            starting_G2S_index,
                            ending_G2S_index,
                            warp_rank_within_pipeline,
                            thread_rank_within_pipeline,
                            pipeline_rank,
                            acc_token_fp32,
                            acc_prob_ptr,
                            /*prob_slot=*/0,
                            prob_vec_per_thread,
                            prob_dim);
                    } while (!last_local_src);
                }

                // Remote-LSA-team accumulation: at most one src token per remote LSA team (LSA team-1, LSA team-2, ..., LSA team+1).
                if constexpr (LSA_TEAMS > 1) {
                    const bool* attn_to_rdma_addr = attn_to_rdma_map_base + cur_tokid * (LSA_TEAMS - 1);
#pragma unroll
                    for (int n = 1; n < LSA_TEAMS; n++) {
                        int lteam_id = my_lteam >= n ? my_lteam - n : my_lteam + LSA_TEAMS - n;
                        int rdma_buffer_tile_id = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
                        if (attn_to_rdma_addr[rdma_buffer_tile_id]) {
                            combine_inter_consume_src<
                                THRDS_PER_PIPELINE,
                                BACKWARD_COMBINE,
                                HIDDEN_DIM,
                                /*READ_LAST_FLAG=*/false,
                                /*ACCUMULATE_PROB=*/false,
                                kTokenDtype,
                                kCombineRecipe>(
                                smem_buffer_ptr,
                                token_stage,
                                token_producer_parity,
                                starting_G2S_index,
                                ending_G2S_index,
                                warp_rank_within_pipeline,
                                thread_rank_within_pipeline,
                                pipeline_rank,
                                acc_token_fp32,
                                acc_prob_ptr,
                                /*prob_slot=*/n,
                                prob_vec_per_thread,
                                prob_dim);
                        }
                    }
                }

                // Every attn dst token was routed in dispatch, so it is always written back.
                combine_inter_store_token<
                    THRDS_PER_PIPELINE,
                    STAGES_S2G_PER_PIPELINE,
                    LSA_TEAMS,
                    BACKWARD_COMBINE,
                    HIDDEN_DIM,
                    kTokenDtype,
                    kCombineRecipe>(
                    smem_buffer_ptr,
                    dst_token_stage,
                    starting_S2G_index,
                    ending_S2G_index,
                    warp_rank_within_pipeline,
                    thread_rank_within_pipeline,
                    pipeline_rank,
                    acc_token_fp32,
                    acc_prob_ptr,
                    prob_vec_per_thread,
                    prob_dim,
                    my_lteam,
                    attn_output_token_base,
                    attn_output_prob_base,
                    out_token_stride_u16,
                    /*token_in_chunk=*/group_idx * TOKENS_PER_GROUP + token_in_group,
                    /*absolute_token_id=*/cidx * TOKENS_PER_CHUNK +
                        group_idx * TOKENS_PER_GROUP + token_in_group,
                    num_real_tokens);
            }
        }
    }
    // Attn output buffers are produced only by the local combine kernel, so the CUDA stream's
    // kernel-boundary sync flushes all outstanding TMA S2G writes; no explicit drain is needed here.
}

// __launch_bounds__(1, 1)
// __global__ void device_sync_kernel(uint32_t* intra_node_remote_flags, const uint32_t* expected_flag_value)
// {
//   // Atomically reduce add 1 to the u32 flag on rank #0 in current NVLink domain.
//   // Need a strong system-scope red to make sure all ranks from current NVLink domain can see the side effect.
//   // But no memory fence(i.e. .release) needed since CUDA stream already do that for us.
//   // red.relaxed.sys.global.add.u32          [a], 1;
//   asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
//                 :
//                 : "l"(__cvta_generic_to_global(intra_node_remote_flags)), "n"(1)
//                 : "memory");

//   // Polling flag value from the u32 flag on rank #0 in current NVLink domain.
//   // Keep polling until reach the expected value.
//   uint32_t flag_data = 0;
//   do {
//       flag_data = 0;
//       // Need a strong system-scope load to observe other ranks' Atomic result.
//       // But no no memory fence(i.e. .aquired) needed since no memory operation behind this.
//       asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
//                     : "=r"(flag_data)
//                     : "l"(__cvta_generic_to_global(intra_node_remote_flags))
//                     : "memory");
//     } while (flag_data != *expected_flag_value);
// }

// ============================================================================
// PAD warp device function (expert-major zero padding, fused into dispatch_kernel)
// ============================================================================

// Zero-init EM alignment padding slots; one warp inside dispatch_kernel, concurrent with N2N/G2S/S2G.
// Zeroes one SMEM row, then 32 lanes cp_async_bulk it to padding slots striped by (block,lane).
template <typename PAD_GROUP, typename TOKEN_DATA_TYPE, typename SMEM_TYPE>
__forceinline__ __device__ void PAD_warp_group_device_function(
    TOKEN_DATA_TYPE* __restrict__ local_buf,
    uint8_t* __restrict__ local_scale_buf,
    const int32_t* __restrict__ actual_counts,
    const int64_t* __restrict__ zone_offsets,
    const int experts_per_rank,
    const int alignment,
    const int hidden_dim,
    const int scale_row_bytes,
    const int num_blocks,
    SMEM_TYPE* smem) {
    // Caller zeroes alignment when not expert-major; alignment<=1 ⇒ no padding work.
    if (alignment <= 1 || local_buf == nullptr || actual_counts == nullptr || zone_offsets == nullptr) return;

    const int lane = PAD_GROUP::thread_rank();
    constexpr int warp_size = 32;
    const uint32_t token_bytes = static_cast<uint32_t>(hidden_dim * sizeof(TOKEN_DATA_TYPE));
    const uint32_t zero_bytes = token_bytes > static_cast<uint32_t>(scale_row_bytes)
        ? token_bytes
        : static_cast<uint32_t>(scale_row_bytes);

    // Cooperatively zero the SMEM staging slot once per kernel invocation.
    auto* smem_u4 = reinterpret_cast<uint4*>(smem->get_pad_tma_slot());
    const int vec_n = zero_bytes / sizeof(uint4);
    const uint4 zero4{0, 0, 0, 0};
    for (int i = lane; i < vec_n; i += warp_size) smem_u4[i] = zero4;
    __syncwarp();

    // Flatten padding rows across all experts and stripe across (blocks × lanes).
    const int global_id = static_cast<int>(blockIdx.x) * warp_size + lane;
    const int global_stride = num_blocks * warp_size;

    int row_idx = 0;
    int my_in_flight = 0;
    for (int e = 0; e < experts_per_rank; e++) {
        int32_t count = actual_counts[e];
        // Empty experts reserve no zone slot; never pad them.
        if (count == 0) continue;
        int32_t rem = count % alignment;
        int32_t pad = rem ? (alignment - rem) : 0;
        for (int p = 0; p < pad; p++, row_idx++) {
            if ((row_idx % global_stride) == global_id) {
                void* dst = reinterpret_cast<void*>(local_buf + (zone_offsets[e] + count + p) * hidden_dim);
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst,
                    smem->get_pad_tma_slot(),
                    token_bytes);
                if (local_scale_buf != nullptr && scale_row_bytes > 0) {
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_global,
                        cuda::ptx::space_shared,
                        local_scale_buf +
                            static_cast<size_t>(zone_offsets[e] + count + p) * scale_row_bytes,
                        smem->get_pad_tma_slot(),
                        scale_row_bytes);
                }
                my_in_flight++;
            }
        }
    }
    if (my_in_flight > 0) {
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    }
    __syncwarp();
}

// Cross-LSA-team RDMA (GIN) cross-round WAR guard, warp-collective (lane i -> peer i), relaxed (WAR only).

// Wait until every rail peer's flag reaches the expected round.
__device__ __forceinline__ void
warp_rdma_guard_wait(const uint64_t* peer_flags, int my_lteam, int lsa_teams, uint64_t expected) {
    for (int peer = (threadIdx.x & 31); peer < lsa_teams; peer += 32) {
        if (peer == my_lteam) continue;
        while (nccl_ep::ld_relaxed_sys_global(&peer_flags[peer]) + 1ull < expected) { /* busy-wait */
        }
    }
}

// Publish the expected round into this rank's slot (my_slot) of every rail peer's window.
// Runs on reserved context 0, so it never shares QP state with a data channel.
__device__ __forceinline__ void warp_rdma_guard_publish(
    ncclDevComm dcomm,
    ncclWindow_t dest_window,
    size_t my_slot,
    int my_lteam,
    int lsa_teams,
    uint64_t expected) {
    ncclGin net(dcomm, /*contextIndex=*/0);
    ncclTeam rail = ncclTeamRail(dcomm);
    for (int peer = (threadIdx.x & 31); peer < lsa_teams; peer += 32) {
        if (peer == my_lteam) continue;
        net.putValue(rail, peer, dest_window, my_slot, expected, ncclGin_None{}, ncclCoopThread());
    }
}

// Shared runtime state for the count-mode MAP-warp phase helpers below. Built
// once in dispatch_push_map_warp; every field mirrors a dispatch_push_map_warp input or a
// value derived from it. Template-derived constexpr (HIT_WORDS, MASK_WORDS, ...)
// are recomputed inside each helper, not stored here. Fields read only by
// dispatch_push_map_publish_outputs live in dispatch_push_map_publish_ctx_t instead, so they
// don't sit live in registers across the build_s2d/build_lerm phases that run before it.
struct dispatch_push_map_ctx_t {
    const int64_t* topk_idx;
    int32_t* s2d;
    const int32_t* per_src_lteam_chunk_rank;
    const uint64_t* token_dst_rank_bitmap;
    uint8_t* const* recv_tables_ptrs;
    int recv_max_slots;
    bool allow_overflow_drop;
    const int32_t* per_src_lteam_num_tokens;
    int num_topk;
    int experts_per_rank;
    int lsa_team_size;
    int my_rank;
    int num_blocks;
    int* s2d_chunks_ready;
    // Derived once in dispatch_push_map_warp.
    int experts_per_lsa_team;
    int dst_off;
    int expert_off;
    int num_chunks;
    size_t count_row_ints;
    const int32_t* count_rows;
    int hit_word_offset;
    int hit_bit_shift;
};

// Fields dispatch_push_map_publish_outputs alone needs (block 0 / MAP warp 0 only); built
// just before that call so they don't extend the live range of the earlier phases.
struct dispatch_push_map_publish_ctx_t {
    int num_src_ranks;
    int32_t* num_recv_out;
    int64_t* expert_token_offsets_out;
    int32_t* per_expert_counts_out;
    void* caller_offsets;
    void* caller_counts;
    void* caller_recv_total;
    bool caller_out_is_int64;
    int em_alignment;
};

// Warp-uniform FLAT slot base per dest rank for one (stream, chunk): the stream
// sender's global-sender prefix plus the stream's earlier-chunk counts. Splits
// the per-rank sum across lanes differently depending on whether dest ranks fit
// in a warp (see the two branches below).
// Rescans [0, c) from scratch each call: O(n^2) in n = num_chunks (chunk_size defaults to
// 64, overridable via NCCL_EP_TOKENS_PER_CHUNK). Measured linear up to 16K tokens/rank at
// chunk_size=32; an incremental running sum would make it O(num_chunks) if it matters.
template <int MAX_DST_RANKS>
__device__ __forceinline__ void dispatch_push_map_chunk_slot_base(
    const int32_t* rank_base, const int32_t* stream_chunk_rank, int c,
    int32_t (&run)[MAX_DST_RANKS], int lane) {
        if constexpr (MAX_DST_RANKS <= 32) {
            constexpr int groups = 32 / MAX_DST_RANKS;
            const int g = lane / MAX_DST_RANKS;
            const int r = lane - g * MAX_DST_RANKS;
            int32_t sum = 0;
            if (g < groups)
                for (int c2 = g; c2 < c; c2 += groups)
                    sum += stream_chunk_rank[(size_t)c2 * MAX_DST_RANKS + r];
            for (int stride = 1; stride < groups; stride <<= 1) {
                const int32_t v = __shfl_down_sync(0xffffffffu, sum, stride * MAX_DST_RANKS);
                if (g + stride < groups) sum += v;
            }
            for (int r2 = 0; r2 < MAX_DST_RANKS; r2++)
                run[r2] = rank_base[r2] + __shfl_sync(0xffffffffu, sum, r2);
        } else {
            constexpr int SLOTS = nccl_ep::ceil_div(MAX_DST_RANKS, 32);
            int32_t part[SLOTS] = {};
            for (int c2 = 0; c2 < c; c2++) {
                const int32_t* row = stream_chunk_rank + (size_t)c2 * MAX_DST_RANKS;
                for (int s = 0; s < SLOTS; s++) {
                    const int rr = lane + s * 32;
                    if (rr < MAX_DST_RANKS) part[s] += row[rr];
                }
            }
            for (int r2 = 0; r2 < MAX_DST_RANKS; r2++) {
                const int32_t v = __shfl_sync(0xffffffffu, part[r2 >> 5], r2 & 31);
                run[r2] = rank_base[r2] + v;
            }
        }
}

// Global-sender prefix per (stream, dest rank): tokens all senders ordered before
// this stream's sender already placed into my LSA team's dest rank. Each MAP warp
// fills its own s_rank_base copy (the builder warps run unsynchronized).
__device__ __forceinline__ void dispatch_push_map_build_rank_base(
    const dispatch_push_map_ctx_t& ctx, int32_t* base, int lane) {
    const int lsa_team_size = ctx.lsa_team_size;
    const int my_rank = ctx.my_rank;
    const int dst_off = ctx.dst_off;
    const size_t count_row_ints = ctx.count_row_ints;
    auto cnt_rank_row = [&](int s) -> const int32_t* {
        return ctx.count_rows + (size_t)s * count_row_ints;
    };
    for (int r = lane; r < lsa_team_size; r += 32) {
        int32_t sum = 0;
        for (int s = 0; s < my_rank; s++) sum += cnt_rank_row(s)[dst_off + r];
        base[r] = sum;
    }
    __syncwarp();
}

// Maps a topk entry to this LSA team's local expert id; -1 if routed off-team or past
// the packed mask width.
template <int EXPERTS_PER_RANK, typename TopkT>
__device__ __forceinline__ int32_t dispatch_push_map_topk_ge(
    const TopkT* row, int k, int expert_off, int experts_per_lsa_team) {
    constexpr int MASK_WORDS = nccl_ep::bit_words(EXPERTS_PER_RANK);
        const int32_t ge = (int32_t)((int64_t)row[k] - expert_off); // LSA-team-local expert id
        if (ge < 0 || ge >= experts_per_lsa_team) return -1;
        if ((ge % EXPERTS_PER_RANK) >= 64 * MASK_WORDS) return -1;
        return ge;
}

// token_dst_rank_bitmap is a packed bitmap over all world dest ranks (HIT_WORDS =
// bit_words(MAX_DST_RANKS) u64 words per token); a receiver only ever needs the bits covering
// its own LSA team's contiguous rank range. Bundles that range's compile-time word geometry
// with its runtime word/bit offset, and the single-bit hit test against a prefetched slice.
template <int MAX_DST_RANKS>
struct rank_slice_t {
    static constexpr int HIT_WORDS = nccl_ep::bit_words(MAX_DST_RANKS);
    // Words a single LSA team's rank slice spans: 1 for lsa_team_size <= 64, else ceil(lsa/64).
    static constexpr int SLICE_WORDS = nccl_ep::bit_words(MAX_DST_RANKS);
    // Single-word fast path is valid only when every LSA team's slice fits inside one 64-bit
    // word, i.e. lsa <= 64 and a divisor of 64 (team bases are lsa-aligned, so no straddle).
    // Any other size uses the general straddle-capable reader below.
    static constexpr bool ONE_WORD = (SLICE_WORDS == 1) && ((64 % MAX_DST_RANKS) == 0);
    // Words to prefetch per token to cover an arbitrarily-aligned, possibly straddling slice.
    static constexpr int PREFETCH_WORDS = ONE_WORD ? 1 : (SLICE_WORDS + 1);

    int word_offset; // first word of this LSA team's slice
    int bit_shift;   // bit offset of the slice within that word

    // Test whether dest rank r (LSA-team-local) is hit, given PREFETCH_WORDS words already
    // loaded starting at word_offset.
    __device__ __forceinline__ bool hit(const uint64_t* prefetched, int r) const {
        if constexpr (ONE_WORD) {
            return (prefetched[0] >> (bit_shift + r)) & 1ull;
        } else {
            return nccl_ep::test_bit(prefetched, bit_shift + r);
        }
    }
};

// Build the rank-major s2d (FLAT slot per (stream, token, dest rank), -1 = not
// routed) that S2G gates on. All MAP warps walk S2G's (chunk x stream) domain
// splitting units round-robin (unit % NUM_MAP_WARPS), publishing each via
// s2d_chunks_ready; units past a stream's token count publish empty.
template <int NUM_OF_TOKENS_PER_CHUNK, int MAX_DST_RANKS, int NUM_MAP_WARPS>
__device__ __forceinline__ void dispatch_push_map_build_s2d(
    const dispatch_push_map_ctx_t& ctx, int map_warp,
    int32_t (*s_rank_base)[MAX_DST_RANKS]) {
    const int lane = (int)(threadIdx.x & 31);
    using RankSlice = rank_slice_t<MAX_DST_RANKS>;
    const RankSlice rank_slice{ctx.hit_word_offset, ctx.hit_bit_shift};
    constexpr int WAVES = nccl_ep::ceil_div(NUM_OF_TOKENS_PER_CHUNK, 32);
    const int num_chunks = ctx.num_chunks;
    const int num_blocks = ctx.num_blocks;
    const int lsa_team_size = ctx.lsa_team_size;
    const int* per_src_lteam_num_tokens = ctx.per_src_lteam_num_tokens;
    const uint64_t* token_dst_rank_bitmap = ctx.token_dst_rank_bitmap;
    int32_t* s2d = ctx.s2d;
    const int32_t* per_src_lteam_chunk_rank = ctx.per_src_lteam_chunk_rank;
    int* s2d_chunks_ready = ctx.s2d_chunks_ready;
    const int stream_tokens = per_src_lteam_num_tokens[0];
    auto chunk_slot_base = [&](const int32_t* rank_base, const int32_t* stream_chunk_rank, int c,
                               int32_t (&run)[MAX_DST_RANKS]) {
        dispatch_push_map_chunk_slot_base<MAX_DST_RANKS>(rank_base, stream_chunk_rank, c, run, lane);
    };
    // Chunks stride across blocks (c += num_blocks); each block's own chunk order then
    // stripes across its MAP warps (unit % NUM_MAP_WARPS), so every (block, warp) owns a
    // disjoint, evenly-spaced slice of the chunks.
    int built = 0, unit = 0;
    for (int c = blockIdx.x; c < num_chunks; c += num_blocks, unit++) {
        if (unit % NUM_MAP_WARPS != map_warp) continue;
        const int t0 = c * NUM_OF_TOKENS_PER_CHUNK;
        if (t0 >= stream_tokens) {
            built++;
            if (lane == 0) nccl_ep::st_release_cta(s2d_chunks_ready + map_warp, built);
            continue;
        }
        const uint64_t* stream_hits = token_dst_rank_bitmap;
        int32_t* stream_s2d = s2d;
        int32_t run[MAX_DST_RANKS];
        chunk_slot_base(s_rank_base[map_warp], per_src_lteam_chunk_rank, c, run);
        // Waves tiled so the hit-bitmap loads issue in parallel ahead of the serial
        // ballot chain (run[] carries across waves).
        constexpr int PF_WAVES = WAVES < 4 ? WAVES : 4;
        for (int wave_base = 0; wave_base < WAVES; wave_base += PF_WAVES) {
            // Non-straddling slice: one word covers it. Otherwise prefetch RankSlice::PREFETCH_WORDS
            // words to cover an arbitrarily-aligned, possibly straddling slice (clamped to HIT_WORDS).
            constexpr int PF_SLICE = RankSlice::PREFETCH_WORDS;
            // Zero-initialized so waves past the per-lane break below (out-of-chunk/out-of-stream)
            // read back as "no hit", matching the old valid-ternary fill without re-checking validity below.
            uint64_t hits_pf[PF_WAVES][PF_SLICE] = {};
#pragma unroll
            for (int pi = 0; pi < PF_WAVES; pi++) {
                const int tc = (wave_base + pi) * 32 + lane; // offset within the chunk
                const int t = t0 + tc;
                // tc/t only grow with pi, so once out of range every later wave is too.
                if (tc >= NUM_OF_TOKENS_PER_CHUNK || t >= stream_tokens) break;
                const uint64_t* slice_hits = stream_hits + (size_t)t * RankSlice::HIT_WORDS + rank_slice.word_offset;
                if constexpr (RankSlice::ONE_WORD) {
                    hits_pf[pi][0] = slice_hits[0];
                } else {
#pragma unroll
                    for (int mw = 0; mw < PF_SLICE; mw++)
                        if (rank_slice.word_offset + mw < RankSlice::HIT_WORDS) hits_pf[pi][mw] = slice_hits[mw];
                }
            }
#pragma unroll
            for (int pi = 0; pi < PF_WAVES; pi++) {
                const int t = t0 + (wave_base + pi) * 32 + lane;
                // FLAT slot assignment per dest rank (dedup: one slot per routed rank).
                for (int r = 0; r < lsa_team_size; r++) {
                    const bool hit = rank_slice.hit(hits_pf[pi], r);
                    const uint32_t bal = __ballot_sync(0xffffffffu, hit);
                    if (hit) {
                        const int32_t slot = run[r] + (int32_t)__popc(bal & ((1u << lane) - 1u));
                        // TRAP: an over-budget slot under a non-DROP overflow policy aborts here,
                        // before this chunk is published, instead of leaving S2G (which may already
                        // be consuming earlier-published chunks concurrently) to hit it blind as a
                        // raw assert with no actionable message.
                        if (!ctx.allow_overflow_drop && slot_overflows(slot, ctx.recv_max_slots)) {
                            printf("ncclEp HT EM count dispatch: recv slot %d exceeds "
                                   "max_recv_tokens_per_rank %d; increase "
                                   "ncclEpGroupConfig_t::max_recv_tokens_per_rank or set "
                                   "NCCL_EP_OVERFLOW_DROP\n", slot, ctx.recv_max_slots);
                            __trap();
                        }
                        // DROP: a FLAT slot at/above the recv budget would overrun the
                        // receiver's FLAT/LERM buffers (sized to recv_max_slots), so mark it
                        // -1 (S2G and build_lerm skip -1). Mirrors scan mode's FLAT clamp.
                        const int32_t es = drop_overflow_slot(slot, ctx.allow_overflow_drop, ctx.recv_max_slots);
                        stream_s2d[(size_t)t * lsa_team_size + r] = es;
                    }
                    run[r] += __popc(bal);
                }
                // The combine gate (rdma_to_attn_map clear for fully-dropped own tokens) is applied
                // in a post-drain kernel phase, so dispatch reads the pristine routing map.
            }
        }
        // Publish the unit: the GPU-scope fence makes the s2d stores visible to the TMA
        // engine (which reads via L2, beyond what a CTA-scope release guarantees), the proxy
        // fence orders them into the async-proxy domain, and the release pairs with S2G's
        // acquire poll on the counter.
        __threadfence();
        nccl_ep::fence_proxy_async();
        __syncwarp();
        built++;
        if (lane == 0) nccl_ep::st_release_cta(s2d_chunks_ready + map_warp, built);
    }
}

// Build the receiver LERM rows (packed per-slot expert bitmap) in the peer IPC
// tables, off the S2G critical path. All MAP warps split chunks round-robin.
// Weights are receiver-local (from the landed prob) and flat2em is assigned by the
// permute, so only LERM is written here.
template <int NUM_OF_TOKENS_PER_CHUNK, int MAX_DST_RANKS, int NUM_MAP_WARPS,
          int EXPERTS_PER_RANK>
__device__ __forceinline__ void dispatch_push_map_build_lerm(
    const dispatch_push_map_ctx_t& ctx, int map_warp) {
    const int lane = (int)(threadIdx.x & 31);
    constexpr int MASK_WORDS = nccl_ep::bit_words(EXPERTS_PER_RANK);
    constexpr int WAVES = nccl_ep::ceil_div(NUM_OF_TOKENS_PER_CHUNK, 32);
    const int num_chunks = ctx.num_chunks;
    const int num_blocks = ctx.num_blocks;
    const int lsa_team_size = ctx.lsa_team_size;
    const int experts_per_rank = ctx.experts_per_rank;
    const int experts_per_lsa_team = ctx.experts_per_lsa_team;
    const int expert_off = ctx.expert_off;
    const int num_topk = ctx.num_topk;
    const int64_t* topk_idx = ctx.topk_idx;
    int32_t* s2d = ctx.s2d;
    uint8_t* const* recv_tables_ptrs = ctx.recv_tables_ptrs;
    const int* per_src_lteam_num_tokens = ctx.per_src_lteam_num_tokens;
    int* s2d_chunks_ready = ctx.s2d_chunks_ready;
    auto map_topk_ge = [&](const auto* row, int k) -> int32_t {
        return dispatch_push_map_topk_ge<EXPERTS_PER_RANK>(row, k, expert_off, experts_per_lsa_team);
    };
    // Single-LSA-team: writes only the LERM rows: weights come from the landed dense prob and
    // flat2em is assigned by the permute. All MAP warps split each chunk at wave
    // granularity; FLAT slots are read back from the built s2d (0xFF-preset, -1 = not routed).
    const int stream_tokens = per_src_lteam_num_tokens[0];
    // LERM rows sit at the base of the receiver table region (weights are receiver-local now).
    const size_t lerm_base = 0;
    int unit = 0;
    for (int c = blockIdx.x; c < num_chunks; c += num_blocks, unit++) {
        const int t0 = c * NUM_OF_TOKENS_PER_CHUNK;
        if (t0 >= stream_tokens) break;
        // The chunk's s2d (same unit numbering as build_s2d) supplies the FLAT slots.
        if (lane == 0)
            while (nccl_ep::ld_acquire_cta(s2d_chunks_ready + unit % NUM_MAP_WARPS) <=
                   unit / NUM_MAP_WARPS) {
            }
        __syncwarp();
        for (int w = 0; w < WAVES; w++) {
            if ((NUM_MAP_WARPS - 1 - (unit * WAVES + w) % NUM_MAP_WARPS) != map_warp) continue;
            const int tc = w * 32 + lane;
            const int t = t0 + tc;
            if (tc >= NUM_OF_TOKENS_PER_CHUNK || t >= stream_tokens) continue;
            // Collect this token's routed dest ranks and packed expert masks in a
            // small list, then write only those rows -- avoids walking every dest
            // rank on wide NVLink domains where lsa_team_size >> num_topk.
            constexpr int kRankListCap = (MAX_DST_RANKS < 64) ? MAX_DST_RANKS : 64;
            int      touched_rank[kRankListCap];
            uint64_t touched_mask[kRankListCap][MASK_WORDS];
            int nrank = 0;
            // Derive each token's per-dest-rank expert mask from its topk row.
            const int64_t* row = topk_idx + (size_t)t * num_topk;
            for (int k = 0; k < num_topk; k++) {
                const int32_t ge = map_topk_ge(row, k);
                if (ge < 0) continue;
                const int r = ge / EXPERTS_PER_RANK;
                const int le = ge - r * EXPERTS_PER_RANK;
                int idx = -1;
                for (int i = 0; i < nrank; i++) if (touched_rank[i] == r) { idx = i; break; }
                if (idx < 0) {
                    // nrank is bounded by distinct dest ranks (<=lsa_team_size<=MAX_DST_RANKS)
                    // and by num_topk (<=kEpCountMaxTopk=64); guard the cross-file invariant.
                    EP_DEVICE_ASSERT(nrank < kRankListCap);
                    idx = nrank;
                    touched_rank[nrank] = r;
#pragma unroll
                    for (int mw = 0; mw < MASK_WORDS; mw++) touched_mask[nrank][mw] = 0;
                    nrank++;
                }
                nccl_ep::set_bit(touched_mask[idx], le);
            }
            // LERM is a packed bitmap: ceil(epr/64) u64 words per slot (the mask is already
            // packed, so copy the words straight through -- no bit->byte expansion).
            const int lerm_words = nccl_ep::bit_words(experts_per_rank);
            for (int i = 0; i < nrank; i++) {
                const int r = touched_rank[i];
                const int32_t slot = s2d[(size_t)t * lsa_team_size + r];
                // Skip un-routed (-1) and any over-budget slot: under a non-DROP overflow
                // policy build_s2d does not clamp, so this bound keeps the write inside the
                // recv_max_slots-sized LERM region and never scribbles a peer's IPC table
                // (publish_outputs traps the run with an actionable message).
                if (slot < 0 || slot >= ctx.recv_max_slots) continue;
                uint64_t* lrow = reinterpret_cast<uint64_t*>(recv_tables_ptrs[r] + lerm_base) +
                                 (size_t)slot * lerm_words;
                for (int mw = 0; mw < lerm_words; mw++) lrow[mw] = touched_mask[i][mw];
            }
        }
    }
}

// Block 0 / warp 0 publishes the FLAT recv count and the receiver EM zone offsets
// (padded counts, caller outputs) for the post-dispatch permute.
template <int EXPERTS_PER_RANK>
__device__ __forceinline__ void dispatch_push_map_publish_outputs(
    const dispatch_push_map_ctx_t& ctx, const dispatch_push_map_publish_ctx_t& pub, int map_warp) {
    const int lane = (int)(threadIdx.x & 31);
    constexpr int MASK_WORDS = nccl_ep::bit_words(EXPERTS_PER_RANK);
    const int num_src_ranks = pub.num_src_ranks;
    const int dst_off = ctx.dst_off;
    const int my_rank = ctx.my_rank;
    const int expert_off = ctx.expert_off;
    const int experts_per_rank = ctx.experts_per_rank;
    const int em_alignment = pub.em_alignment;
    const size_t count_row_ints = ctx.count_row_ints;
    int32_t* num_recv_out = pub.num_recv_out;
    int64_t* expert_token_offsets_out = pub.expert_token_offsets_out;
    int32_t* per_expert_counts_out = pub.per_expert_counts_out;
    void* caller_offsets = pub.caller_offsets;
    void* caller_counts = pub.caller_counts;
    void* caller_recv_total = pub.caller_recv_total;
    const bool caller_out_is_int64 = pub.caller_out_is_int64;
    auto cnt_rank_row = [&](int s) -> const int32_t* {
        return ctx.count_rows + (size_t)s * count_row_ints;
    };
    auto cnt_expert_row = [&](int s) -> const int32_t* {
        return cnt_rank_row(s) + num_src_ranks;
    };
    if (map_warp == 0 && blockIdx.x == 0) {
        if (lane == 0) {
            int32_t tot = 0;
            for (int s = 0; s < num_src_ranks; s++)
                tot += cnt_rank_row(s)[dst_off + my_rank];
            // TRAP: over-budget recv count under a non-DROP overflow policy aborts here with an
            // actionable message before the permute reads the FLAT/LERM buffers (scan-mode parity).
            if (!ctx.allow_overflow_drop && tot > ctx.recv_max_slots) {
                printf("ncclEp HT EM count dispatch: recv tokens %d exceed max_recv_tokens_per_rank "
                       "%d; increase ncclEpGroupConfig_t::max_recv_tokens_per_rank or set "
                       "NCCL_EP_OVERFLOW_DROP\n", tot, (int)ctx.recv_max_slots);
                __trap();
            }
            // DROP: this FLAT recv count drives the permute's per-slot loop over the
            // recv_max_slots-sized FLAT/LERM/weights buffers, so clamp it to the budget when
            // dropping (over-budget FLAT slots were already dropped to -1 in build_s2d).
            if (num_recv_out)
                *num_recv_out =
                    (ctx.allow_overflow_drop && tot > ctx.recv_max_slots) ? ctx.recv_max_slots : tot;
        }
        if (expert_token_offsets_out) {
            __shared__ int64_t s_pub_off[64 * MASK_WORDS + 1];
            __shared__ int32_t s_raw_cnt[64 * MASK_WORDS];
            const int em_align = (em_alignment > 1) ? em_alignment : 1;
            // DROP: clamp published per-expert offsets/counts to the recv budget so
            // over-budget tokens drop; slot bases keep the true prefix and num_recv_out
            // stays the pre-drop recv total.
            const bool drop = ctx.allow_overflow_drop;
            const int64_t cap = ctx.recv_max_slots;
            for (int e = lane; e < experts_per_rank; e += 32) {
                const int ge = expert_off + my_rank * experts_per_rank + e;
                int c = 0;
                for (int s = 0; s < num_src_ranks; s++) c += cnt_expert_row(s)[ge];
                const int64_t padded = nccl_ep::align<int64_t>(c, em_align);
                s_pub_off[e] = padded;
                s_raw_cnt[e] = c;
            }
            __syncwarp();
            if (lane == 0) {
                int64_t cum = 0;
                for (int e = 0; e < experts_per_rank; e++) {
                    const int64_t padded = s_pub_off[e];
                    s_pub_off[e] = cum;
                    cum += padded;
                }
                s_pub_off[experts_per_rank] = cum;
                // TRAP: padded EM total over budget under a non-DROP policy (may overflow via
                // alignment padding even when the FLAT count fits) aborts with scan-mode parity.
                if (!drop && cum > cap) {
                    printf("ncclEp HT EM count dispatch: padded recv tokens %lld exceed "
                           "max_recv_tokens_per_rank %lld; increase "
                           "ncclEpGroupConfig_t::max_recv_tokens_per_rank or set NCCL_EP_OVERFLOW_DROP\n",
                           (long long)cum, (long long)cap);
                    __trap();
                }
                expert_token_offsets_out[experts_per_rank] = (drop && cum > cap) ? cap : cum;
                // recv_total_counter reports the pre-drop padded total (unclamped cum);
                // equals the unpadded FLAT recv total when em_alignment == 1.
                if (caller_recv_total) {
                    if (caller_out_is_int64) *static_cast<int64_t*>(caller_recv_total) = cum;
                    else *static_cast<int32_t*>(caller_recv_total) = (int32_t)cum;
                }
            }
            __syncwarp();
            for (int e = lane; e < experts_per_rank; e += 32) {
                const int64_t base = s_pub_off[e];
                const int64_t padded = s_pub_off[e + 1] - base;
                // Room left in the recv buffer at this expert's zone (0 once full).
                const int64_t room = drop ? (base < cap ? cap - base : 0) : padded;
                const int64_t expert_pub_off = (drop && base > cap) ? cap : base;
                const int32_t out_actual = drop ? (int32_t)((int64_t)s_raw_cnt[e] < room ? s_raw_cnt[e] : room)
                                                : s_raw_cnt[e];
                const int64_t out_padded = drop ? (padded < room ? padded : room) : padded;
                expert_token_offsets_out[e] = expert_pub_off;
                if (per_expert_counts_out) per_expert_counts_out[e] = out_actual;
                if (caller_offsets) {
                    if (caller_out_is_int64) static_cast<int64_t*>(caller_offsets)[e] = expert_pub_off;
                    else static_cast<int32_t*>(caller_offsets)[e] = (int32_t)expert_pub_off;
                }
                if (caller_counts) {
                    if (caller_out_is_int64) static_cast<int64_t*>(caller_counts)[e] = out_padded;
                    else static_cast<int32_t*>(caller_counts)[e] = (int32_t)out_padded;
                }
            }
        }
    }
}

// Count-mode routing-map builder, run by dedicated MAP warps concurrently with
// G2S/N2N. Builds the rank-major s2d that S2G gates on via s2d_chunks_ready, then
// writes the receiver LERM rows; block 0 / warp 0 publishes the recv count and EM
// zone offsets. Single-LSA-team only, so there is no cross-team metadata exchange.
// Phase helpers (_build_s2d / _build_lerm / _publish_outputs) share dispatch_push_map_ctx_t.
template <int NUM_OF_TOKENS_PER_CHUNK, int MAX_DST_RANKS, int NUM_MAP_WARPS,
          int EXPERTS_PER_RANK>
__device__ __forceinline__ void dispatch_push_map_warp(
    const dispatch_push_count_kparams_t& p,
    int experts_per_rank,    // runtime; == EXPERTS_PER_RANK, kept to match dispatch_push_map_ctx_t layout
    int my_rank,             // local (LSA) rank
    int my_lteam,
    int num_blocks,
    int num_tokens_per_rank, // S2G's chunk domain (max tokens); >= any stream's token count
    int map_warp,            // this warp's index in [0, NUM_MAP_WARPS); stripes s2d + LERM
    int* s2d_chunks_ready) {
    const int lane = (int)(threadIdx.x & 31);
    // experts_per_rank is JIT-baked, so the mask width and topk->rank math below
    // are compile-time constants.
    constexpr int MASK_WORDS = nccl_ep::bit_words(EXPERTS_PER_RANK);
    const int experts_per_lsa_team = experts_per_rank * p.lsa_team_size;             // this LSA team's experts
    const int dst_off = my_lteam * p.lsa_team_size;                // my LSA team's global dst-rank base
    // token_dst_rank_bitmap words per token (ceil(world/64)); the receiver reads only the words
    // covering its own LSA team's rank slice (see rank_slice_t above).
    const rank_slice_t<MAX_DST_RANKS> rank_slice{nccl_ep::bit_word(dst_off), dst_off & 63};
    const int expert_off = my_lteam * experts_per_lsa_team;                       // my LSA team's global expert base
    const int num_chunks = nccl_ep::ceil_div(num_tokens_per_rank, NUM_OF_TOKENS_PER_CHUNK);
    // Count landing rows, one per global sender: own-LSA-team senders' rows are fanned out into this
    // rank's own table region at the dispatch head (the head LSA barrier orders them before any read
    // here). All reads are local.
    // Unfused path: the rows were AllGathered into a per-handle buffer in UpdateHandle, so read
    // that in place of the fanned-out landing (no head fan-out, no LSA barrier needed).
    const bool count_unfused = !p.fused_meta_dispatch;
    const int32_t* count_rows =
        count_unfused
            ? p.cached_cnt_rows
            : reinterpret_cast<const int32_t*>(p.recv_tables_ptrs[my_rank] + p.published_offset);
    const size_t count_row_ints = (size_t)p.num_src_ranks + (size_t)experts_per_rank * p.num_src_ranks;
    dispatch_push_map_ctx_t ctx;
    ctx.topk_idx = p.cached_topk_idx;
    ctx.s2d = p.s2d_out;
    ctx.per_src_lteam_chunk_rank = p.per_src_lteam_chunk_rank;
    ctx.token_dst_rank_bitmap = p.token_dst_rank_bitmap;
    ctx.recv_tables_ptrs = p.recv_tables_ptrs;
    ctx.recv_max_slots = p.recv_max_slots;
    ctx.allow_overflow_drop = p.allow_overflow_drop;
    ctx.per_src_lteam_num_tokens = p.per_src_lteam_num_tokens;
    ctx.num_topk = p.num_topk;
    ctx.experts_per_rank = experts_per_rank;
    ctx.lsa_team_size = p.lsa_team_size;
    ctx.my_rank = my_rank;
    ctx.num_blocks = num_blocks;
    ctx.s2d_chunks_ready = s2d_chunks_ready;
    ctx.experts_per_lsa_team = experts_per_lsa_team;
    ctx.dst_off = dst_off;
    ctx.expert_off = expert_off;
    ctx.num_chunks = num_chunks;
    ctx.count_row_ints = count_row_ints;
    ctx.count_rows = count_rows;
    ctx.hit_word_offset = rank_slice.word_offset;
    ctx.hit_bit_shift = rank_slice.bit_shift;
    __shared__ int32_t s_rank_base[NUM_MAP_WARPS][MAX_DST_RANKS];

    // Rank-base prefixes shared by the s2d builders and the LERM pass. All MAP warps
    // build s2d, so every warp needs its own row.
    dispatch_push_map_build_rank_base(ctx, s_rank_base[map_warp], lane);

    // Phase 1: build s2d (S2G-critical).
    dispatch_push_map_build_s2d<NUM_OF_TOKENS_PER_CHUNK, MAX_DST_RANKS, NUM_MAP_WARPS>(ctx, map_warp, s_rank_base);

    // Phase 2: build receiver LERM tables.
    dispatch_push_map_build_lerm<NUM_OF_TOKENS_PER_CHUNK, MAX_DST_RANKS, NUM_MAP_WARPS,
                            EXPERTS_PER_RANK>(ctx, map_warp);

    // Peer-visible table writes must be ordered before this rank's completion-barrier flag.
    __threadfence_system();
    // Phase 3: publish recv count + expert offsets (block 0 / warp 0). pub is built here, not
    // up front, so its fields aren't live registers across the build_s2d/build_lerm phases above.
    dispatch_push_map_publish_ctx_t pub;
    pub.num_src_ranks = p.num_src_ranks;
    pub.num_recv_out = p.num_recv_out;
    pub.expert_token_offsets_out = p.expert_token_offsets_out;
    pub.per_expert_counts_out = p.per_expert_counts_out;
    pub.caller_offsets = p.caller_offsets;
    pub.caller_counts = p.caller_counts;
    pub.caller_recv_total = p.caller_recv_total;
    pub.caller_out_is_int64 = p.caller_out_is_int64;
    pub.em_alignment = p.em_alignment;
    dispatch_push_map_publish_outputs<EXPERTS_PER_RANK>(ctx, pub, map_warp);
}

// Elect the last block to arrive at *counter (result broadcast to all threads in the block).
__device__ __forceinline__ bool elect_last_block(const int* counter, int num_blocks) {
    __syncthreads();
    int arrived = -1;
    if (threadIdx.x == 0) arrived = static_cast<int>(nccl_ep::atomic_add_acqrel_global(counter, 1));
    return __syncthreads_or(arrived == num_blocks - 1);
}

template <
    typename TOKEN_DATA_TYPE,
    typename SCALE_DATA_TYPE,
    ncclEpDispQuant_t kRecipe,
    typename GIN_GROUP,
    typename LSA_G2S_GROUP,
    typename LSA_S2G_GROUP,
    typename MAP_GROUP,
    typename PAD_GROUP,
    typename HEAD_EXTRA_GROUP,
    int NUM_STAGES,
    int IN_FLIGHT_S2G,
    int TOKENS_PER_CHUNK,
    int MAX_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    bool FORWARD_DISPATCH,
    int NUM_PIPELINES,
    int LSA_TEAM_SZ,
    ncclEpLayout_t kLayout,
    int HIDDEN_DIM,
    int SF_BYTES_PER_TOKEN,
    int EXPERTS_PER_RANK>
__device__ __forceinline__ void dispatch_kernel_impl(
    const dispatch_kernel_param_t<TOKEN_DATA_TYPE, LSA_TEAM_SZ>& param,
    uint8_t* smem_bytes) {
    static_assert(kRecipe == NCCL_EP_DISP_QUANT_NONE || kRecipe == NCCL_EP_DISP_QUANT_FWD,
                  "unsupported dispatch quantization recipe");
    // The JIT-selected ScaleT validates the caller's scale element packing.
    static_assert(SF_BYTES_PER_TOKEN == 0 || SF_BYTES_PER_TOKEN % sizeof(SCALE_DATA_TYPE) == 0,
                  "scale payload must contain a whole number of ScaleT elements");
    const int my_lteam = param.lsa_team;
    if constexpr (LSA_TEAMS != 1) {
        static_assert(
            GIN_GROUP::size() % 32 == 0 && GIN_GROUP::size() <= 64,
            "Dispatch kernel supports 1 or 2 N2N warps.");
    }
    static_assert(NUM_STAGES % NUM_PIPELINES == 0, "NUM_STAGES must be divisible by NUM_PIPELINES.");
    constexpr int STAGES_PER_PIPELINE = NUM_STAGES / NUM_PIPELINES;

    using cur_smem_t = dispatch_smem_layout_t;

    cur_smem_t smem_layout;
    dispatch_config_t d_config;
    model_config_t d_model;
    d_config.num_of_stages = NUM_STAGES;
    d_config.num_of_in_flight_s2g = IN_FLIGHT_S2G;
    d_config.num_of_tokens_per_chunk = TOKENS_PER_CHUNK;
    d_config.num_of_blocks = NBLOCKS;
    d_config.forward_dispatch = FORWARD_DISPATCH;
    d_config.sf_bytes_per_token = SF_BYTES_PER_TOKEN;
    d_config.num_pipelines = NUM_PIPELINES;
    d_config.stages_per_pipeline = STAGES_PER_PIPELINE;
    d_config.s2d_inner_dim = param.s2d_inner_dim;
    d_model.hidden_dim = HIDDEN_DIM;
    d_model.max_num_of_tokens_per_rank = MAX_TOKENS_PER_RANK;
    d_model.num_of_experts_per_rank = param.experts_per_rank;
    d_model.ranks_per_lsa_team = param.ranks_per_lsa_team;
    d_model.num_lsa_teams = LSA_TEAMS;
    create_dispatch_smem_layout<kLayout, sizeof(TOKEN_DATA_TYPE)>(smem_layout, smem_bytes, d_config, d_model);
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    using head_init_warp = warp_group<1, 0>;
    using head_rdma_warp = warp_group<1, 1>;
    using head_lsa_warp = warp_group<1, 2>;

    // The head phase runs on warps 0, 1, 2 (mbarrier init / RDMA guard / intra-LSA
    // barrier), so the block must always contain at least 3 warps -- even when the
    // communication groups need fewer (flat layout + single LSA team + 1 pipeline
    // yields only a G2S/S2G pair). HEAD_EXTRA_GROUP is the JIT-sized filler that
    // makes up the difference; it does no communication work.
    static_assert(
        GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size() + MAP_GROUP::size() +
                PAD_GROUP::size() + HEAD_EXTRA_GROUP::size() >= 3 * 32,
        "dispatch head needs 3 warps");
    const int head_tid = (int)threadIdx.x;
    if (head_tid < head_init_warp::size()) {
        // warp 0: per-pipeline mbarrier init (both producer/consumer arrival counts = 1).
        if (head_tid == 0) {
            for (int p = 0; p < NUM_PIPELINES; p++) {
                for (int s = 0; s < STAGES_PER_PIPELINE; s++) {
                    int abs_stage = p * STAGES_PER_PIPELINE + s;
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->lsa_mbarrier_buffer + 2 * abs_stage, 1);
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->lsa_mbarrier_buffer + 2 * abs_stage + 1, 1);
                }
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_s2d_map_mbar(p, 0), 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_s2d_map_mbar(p, 1), 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_S2G_group_mbar(p), 1);
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size()) {
        // warp 1: cross-LSA-team RDMA guard wait.
        if constexpr (LSA_TEAMS != 1) {
            if (param.guard_enabled)
                warp_rdma_guard_wait(
                    reinterpret_cast<const uint64_t*>(
                        reinterpret_cast<const uint8_t*>(param.gin_base_ptr) + param.mr_info.guard_offset),
                    my_lteam,
                    LSA_TEAMS,
                    *param.expected_gin_flag_val);
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size() + head_lsa_warp::size()) {
        // warp 2: intra-LSA LSA barrier. Fused count mode fans this rank's finished
        // count row out to every LSA peer first (self slot always written, even for
        // LSA_TEAM_SZ==1), then runs the barrier with acq_rel so the MAP warps see it
        // after sync. Unfused mode already has rows local (AllGathered in UpdateHandle),
        // so it skips the fan-out and uses a plain guard barrier.
        const bool count_fanout = param.dispatch_push_count.active && param.dispatch_push_count.fused_meta_dispatch;
        if (count_fanout) {
            const int lane = head_tid & 31;
            const int me = my_lteam * LSA_TEAM_SZ + param.local_rank; // global sender-row index
            const size_t row_ints = (size_t)param.dispatch_push_count.num_src_ranks +
                                    (size_t)param.experts_per_rank * param.dispatch_push_count.num_src_ranks;
            // Split by consumer scope: every block ships the cnt_rank prefix (dispatch_push_map_build_rank_base
            // needs it on every block); only block 0 ships the cnt_expert tail
            // (dispatch_push_map_publish_outputs is block 0 / MAP warp 0 only). Idempotent-write is
            // preserved: non-zero blocks write a prefix of what block 0 writes, at identical
            // addresses.
            const size_t n_ship = (blockIdx.x == 0) ? row_ints : (size_t)param.dispatch_push_count.num_src_ranks;
            const int32_t* my_row = param.dispatch_push_count.own_row;
            for (int p = 0; p < LSA_TEAM_SZ; p++) {
                int32_t* dst = reinterpret_cast<int32_t*>(
                    param.dispatch_push_count.recv_tables_ptrs[p] + param.dispatch_push_count.published_offset) + (size_t)me * row_ints;
                for (int i = lane; i < (int)n_ship; i += 32) dst[i] = my_row[i];
            }
            __threadfence_system();
        }
        if constexpr (LSA_TEAM_SZ != 1) {
            // Peer publish/synchronize barrier (a 1-member team needs no barrier; the
            // __threadfence_system above plus the block __syncthreads order the self write).
            if (param.guard_enabled || count_fanout) {
                ncclLsaBarrierSession<ncclCoopWarp> bar(
                    ncclCoopWarp(),
                    param.dcomm,
                    ncclTeamTagLsa(),
                    (uint32_t)blockIdx.x);
                bar.sync(
                    ncclCoopWarp(),
                    count_fanout ? cuda::memory_order_acq_rel : cuda::memory_order_relaxed);
            }
        }
    }

    // Count mode: the dedicated MAP warps build s2d/masks per chunk after the split; S2G polls
    // this counter before each chunk's s2d prefetch, so G2S/N2N start immediately after the head
    // barrier. One s2d unit counter per MAP warp (round-robin units; see dispatch_push_map_warp).
    constexpr int kMapWarps = (MAP_GROUP::size() / 32 > 1) ? MAP_GROUP::size() / 32 : 1;
    __shared__ int s_s2d_chunks_ready[kMapWarps];
    if (param.dispatch_push_count.active && threadIdx.x < kMapWarps) s_s2d_chunks_ready[threadIdx.x] = 0;

    __syncthreads();

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    long long _wt_start = 0;
    if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif
    // LSA-only scan dispatch uses one aligned routing vector per tile. Count mode
    // keeps the transport chunk size because its MAP warps publish whole chunks.
    constexpr int DISPATCH_TILE_TOKENS = LSA_TEAMS == 1 ? sizeof(uint4) : TOKENS_PER_CHUNK;
    constexpr bool HAS_SF = (kRecipe == NCCL_EP_DISP_QUANT_FWD);
    int threadIdx_x_int = (int)threadIdx.x;
    if (threadIdx_x_int < GIN_GROUP::size()) {
        if constexpr (LSA_TEAMS != 1) {
#define DISPATCH_N2N_TEMPLATE \
            dispatch_N2N_warp<GIN_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_STAGES, TOKENS_PER_CHUNK, \
                              MAX_TOKENS_PER_RANK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, FORWARD_DISPATCH, HAS_SF>
            DISPATCH_N2N_TEMPLATE(
                param.attn_to_rdma_map,
                param.local_rank,
                my_lteam,
                param.num_of_tokens_per_rank,
                HIDDEN_DIM,
                SF_BYTES_PER_TOKEN,
                param.experts_per_rank,
                param.dcomm,
                param.num_ctx_per_comm,
                param.token_window,
                param.prob_window,
                param.sf_window,
                param.dest_window,
                &param.mr_info,
                smem_buffer_ptr);
#undef DISPATCH_N2N_TEMPLATE
        }
    } else if (threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size()) {
#define DISPATCH_G2S_TEMPLATE(TILE_TOKENS) \
        dispatch_G2S_warp<LSA_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_STAGES, TILE_TOKENS, \
                            MAX_TOKENS_PER_RANK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, NUM_PIPELINES, \
                            FORWARD_DISPATCH, HAS_SF, kMapWarps>( \
            param.rdma_to_attn_map, param.attn_input_token, param.attn_input_prob, \
            param.attn_input_token_scaling_factor, param.gin_G2S_flags, param.local_rank, \
            my_lteam, param.num_of_tokens_per_rank, HIDDEN_DIM, SF_BYTES_PER_TOKEN, \
            param.experts_per_rank, *param.expected_gin_flag_val, param.dcomm, \
            param.num_ctx_per_comm, param.gin_base_ptr, &param.mr_info, smem_buffer_ptr)
        if (param.dispatch_push_count.active) {
            DISPATCH_G2S_TEMPLATE(TOKENS_PER_CHUNK);
        } else {
            DISPATCH_G2S_TEMPLATE(DISPATCH_TILE_TOKENS);
        }
#undef DISPATCH_G2S_TEMPLATE
    } else if (
        threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size()) {
#define DISPATCH_S2G_TEMPLATE(COUNT_MODE) \
        dispatch_S2G_warp<LSA_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_STAGES, \
                            IN_FLIGHT_S2G, (COUNT_MODE ? TOKENS_PER_CHUNK : DISPATCH_TILE_TOKENS), \
                            LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, NUM_PIPELINES, \
                            FORWARD_DISPATCH, HAS_SF, kLayout, kMapWarps, COUNT_MODE>( \
            param.rdma_to_attn_map, \
            param.sparse_to_dense_map, \
            param.expert_output_token, \
            param.expert_output_prob, \
            param.expert_output_scaling_factor, \
            my_lteam, \
            param.num_of_tokens_per_rank, \
            HIDDEN_DIM, \
            SF_BYTES_PER_TOKEN, \
            param.experts_per_rank, \
            param.local_dup_enabled, \
            param.max_recv_tokens_per_rank, \
            smem_buffer_ptr, \
            param.dispatch_push_count.active ? s_s2d_chunks_ready : nullptr)
        if (param.dispatch_push_count.active) {
            DISPATCH_S2G_TEMPLATE(true);
        } else {
            DISPATCH_S2G_TEMPLATE(false);
        }
#undef DISPATCH_S2G_TEMPLATE
    } else if (
        MAP_GROUP::size() > 0 && threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size() +
                                                       LSA_S2G_GROUP::size() + MAP_GROUP::size()) {
        // MAP warps (count mode only): the MAP warps cooperatively build the count-mode s2d
        // per chunk concurrently with G2S (publishing each chunk via s_s2d_chunks_ready for
        // S2G), then write the receiver LERM rows.
        const int map_warp =
            (threadIdx_x_int -
             (GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size())) /
            32;
        dispatch_push_map_warp<TOKENS_PER_CHUNK, LSA_TEAM_SZ, kMapWarps, EXPERTS_PER_RANK>(
            param.dispatch_push_count,
            param.experts_per_rank,
            param.local_rank,
            my_lteam,
            NBLOCKS,
            param.num_of_tokens_per_rank,
            map_warp,
            s_s2d_chunks_ready);
    } else if (
        PAD_GROUP::size() > 0 && threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size() +
                                                       LSA_S2G_GROUP::size() + MAP_GROUP::size() +
                                                       PAD_GROUP::size()) {
        // PAD warp: zero-init expert-major alignment padding slots concurrently with S2G.
        // No barrier needed against S2G — padding rows live past the actual token rows
        // in each expert's zone, so the two warps target disjoint global memory.
        PAD_warp_group_device_function<PAD_GROUP, TOKEN_DATA_TYPE>(
            param.expert_output_token[param.local_rank],
            SF_BYTES_PER_TOKEN > 0 ? param.expert_output_scaling_factor[param.local_rank] : nullptr,
            param.pad_actual_counts,
            param.pad_expert_token_offsets,
            param.experts_per_rank,
            param.pad_alignment,
            HIDDEN_DIM,
            SF_BYTES_PER_TOKEN,
            NBLOCKS,
            smem_buffer_ptr);
    }
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    if (threadIdx.x % 32 == 0) {
        constexpr int _WT_WARPS = (GIN_GROUP::size() + LSA_G2S_GROUP::size() +
                                   LSA_S2G_GROUP::size() + MAP_GROUP::size() + PAD_GROUP::size() +
                                   HEAD_EXTRA_GROUP::size()) /
                                  32;
        int _warp_id = threadIdx.x / 32;
        int _idx = blockIdx.x * _WT_WARPS + _warp_id;
        param.warp_timing[_idx].start_clock = _wt_start;
        param.warp_timing[_idx].end_clock = clock64();
    }
#endif

    // Count-mode DROP: MAP warps clear the combine gate (rdma_to_attn_map) for own tokens whose
    // every dest slot overflowed. Deferred to here, after G2S/S2G have read the pristine routing
    // map, so dispatch never races the clear (G2S needs no MAP-warp wait). Visible to combine via
    // stream ordering.
    if (param.dispatch_push_count.active && param.dispatch_push_count.allow_overflow_drop) {
        __syncthreads();
        constexpr int kMapBase = GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size();
        if (MAP_GROUP::size() > 0 && threadIdx_x_int >= kMapBase &&
            threadIdx_x_int < kMapBase + (int)MAP_GROUP::size()) {
            const int s2d_inner = smem_buffer_ptr->s2d_inner_dim;
            const int ntok = param.num_of_tokens_per_rank;
            const int num_chunks = nccl_ep::ceil_div(ntok, TOKENS_PER_CHUNK);
            bool* r2a = const_cast<bool*>(param.rdma_to_attn_map);
            const int32_t* s2d = param.sparse_to_dense_map;
            const int map_tid = threadIdx_x_int - kMapBase;
            // Own this block's chunks only (same split as build_s2d), so we read s2d this block built.
            for (int c = blockIdx.x; c < num_chunks; c += NBLOCKS) {
                for (int ti = map_tid; ti < TOKENS_PER_CHUNK; ti += (int)MAP_GROUP::size()) {
                    const int t = c * TOKENS_PER_CHUNK + ti;
                    if (t >= ntok || !r2a[t]) continue;
                    bool kept = false;
                    for (int r = 0; r < s2d_inner; ++r) {
                        if (s2d[(size_t)t * s2d_inner + r] >= 0) { kept = true; break; }
                    }
                    if (!kept) r2a[t] = false;
                }
            }
        }
    }

    // ===== FUSED DEVICE SYNC (dispatch tail) =====
    if (elect_last_block(reinterpret_cast<const int*>(param.dispatch_grid_barrier_counter), NBLOCKS)) {
        using tail_completion_warp = warp_group<1, 0>; // warp 0
        using tail_rdma_warp = warp_group<1, 1>; // warp 1
        const int tail_tid = (int)threadIdx.x;
        if (tail_tid < tail_completion_warp::size()) {
            // warp 0 (thread 0): inter-rank completion barrier, then reset + bump the intra-LSA round
            // (local_dup defers that bump to its own tail).
            if (tail_tid == 0) {
                const uint32_t expected_val = *param.expected_lsa_flag_val;
                nccl_ep::publish_lsa_completion(param.lsa_S2G_flags + param.local_rank, expected_val);
                nccl_ep::wait_lsa_completion(param.lsa_S2G_flags, expected_val, param.ranks_per_lsa_team);
                atomicExch((unsigned int*)param.dispatch_grid_barrier_counter, 0u);
                if (!param.local_dup_enabled)
                    *param.expected_lsa_flag_val += static_cast<uint32_t>(param.ranks_per_lsa_team);
            }
        } else if (tail_tid < tail_completion_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the cross-LSA-team RDMA guard + bump the RDMA round.
            if constexpr (LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_gin_flag_val;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        param.dcomm,
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(my_lteam) * sizeof(uint64_t),
                        my_lteam,
                        LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_gin_flag_val = expected + 1ull;
            }
        }
    }
}

template < // This type represent intra-LSA reduction warp group.
  typename LSA_RED_GROUP,
  // This type represent cross-LSA-team reduction warp group.
  typename CROSS_LSA_RED_GROUP,
  // This type represent intra-LSA G2S warp group.
  typename LSA_G2S_GROUP,
  // This type represent cross-LSA-team G2S warp group.
  typename CROSS_LSA_G2S_GROUP,
  // This type represent cross-LSA-team rdma warp group.
  typename GIN_GROUP,
  // Number of independent data pipeline per CUDA block.
  int PIPELINES_PER_BLOCK,
  // Number of token entry in the shared memory for G2S operations.
  int STAGES_G2S,
  // Number of token entry in the shared memory for S2G operations.
  int STAGES_S2G,
  // Number of token per group in the cross-LSA-team reduction/G2S warp group.
  int TOKENS_PER_GROUP,
  // Size of each chunk.
  int TOKENS_PER_CHUNK,
  // Model configuration.
  int MAX_TOKENS_PER_RANK, 
  // Number of LSA teams.
  int LSA_TEAMS,
  // Number of CUDA block running dispatch kernel.
  int NBLOCKS,
  // Whether the combine kernel is used in backward process. If so, need to transfer the prob for each token as well.
  bool BACKWARD_COMBINE, int HIDDEN_DIM, int LSA_TEAM_SZ, ncclEpLayout_t kLayout,
  // Output dtype, resolved at compile time (JIT literal) so the per-element
  // reduction branches fold away.
  ncclDataType_t kTokenDtype,
  ncclEpCombQuant_t kCombineRecipe>
// Each CUDA block of combine kernel has named warp groups:
// intra/inter reduction, intra/inter G2S, and cross-LSA-team N2N RDMA. Group sizes are
// set by the HT combine warp-count constants and the selected pipeline count.
__device__ __forceinline__ void combine_kernel_impl(const combine_kernel_param_t<LSA_TEAM_SZ>& param,
                                                    uint8_t* smem_bytes) {
    const int my_lteam = param.lsa_team;
    // Compile-time check (only enforce for multi-LSA-team layout).
    if constexpr (LSA_TEAMS != 1) {
        static_assert(
            LSA_G2S_GROUP::size() == 32,
            "Combine kernel only support 1 INTRA_NODE_G2S warp currently.");
        static_assert(
            CROSS_LSA_G2S_GROUP::size() == 32,
            "Combine kernel only support 1 INTER_NODE_G2S warp currently.");
    }
    // The token and its properties should meet size and alignment requirement.
    // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per LSA team to be multiple of 4).
    // We need to add padding or not using TMA for prob, if we want to support other scenario.
    // assert((param.experts_per_rank * param.ranks_per_lsa_team * sizeof(float)) % 16 == 0);
    static_assert((HIDDEN_DIM % 2) == 0, "HIDDEN_DIM must be even for BF16x2.");
    static_assert((HIDDEN_DIM * sizeof(uint16_t)) % 16 == 0, "HIDDEN_DIM must satisfy TMA alignment.");
    static_assert(
        MAX_TOKENS_PER_RANK % TOKENS_PER_CHUNK == 0,
        "MAX_TOKENS_PER_RANK must be multiple of TOKENS_PER_CHUNK.");
    constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    constexpr int _WT_WARPS =
        (LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size() + LSA_G2S_GROUP::size() +
         CROSS_LSA_G2S_GROUP::size() + GIN_GROUP::size()) /
        32;
    long long _wt_head_start = 0;
    long long _wt_head_end = 0;
#endif

    // Shared memory used over 48KB, should use dynamic shared memory.
    using cur_smem_t = combine_smem_layout_t;

    // Initialize the layout struct (each thread has its own copy in registers)
    cur_smem_t smem_layout;
    model_config_t c_model;
    c_model.hidden_dim = HIDDEN_DIM;
    c_model.max_num_of_tokens_per_rank = MAX_TOKENS_PER_RANK;
    c_model.num_of_experts_per_rank = param.experts_per_rank;
    c_model.ranks_per_lsa_team = param.ranks_per_lsa_team;
    c_model.num_lsa_teams = LSA_TEAMS;
    // Layout derives the element width from kTokenDtype (FP32 doubles the per-stage
    // token-buffer bytes vs BF16/FP16).
    create_combine_smem_layout<kTokenDtype, kCombineRecipe>(
        smem_layout,
        smem_bytes,
        STAGES_G2S,
        STAGES_S2G,
        TOKENS_PER_CHUNK,
        BACKWARD_COMBINE,
        c_model);
    smem_layout.s2d_inner_dim = param.s2d_inner_dim;
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    // ===== FUSED DEVICE SYNC (combine head) =====
    using head_init_warp = warp_group<1, 0>; // warp 0
    using head_rdma_warp = warp_group<1, 1>; // warp 1
    const int head_tid = (int)threadIdx.x;
    if (head_tid < head_init_warp::size()) {
        if (head_tid == 0) {
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
            _wt_head_start = clock64();
#endif
            // Block 0 publishes this rank's input readiness; every block waits for all peers.
            const uint32_t expected_val = *param.expected_lsa_flag_val;
            if (blockIdx.x == 0)
                nccl_ep::publish_lsa_completion(param.lsa_S2G_flags + param.local_rank, expected_val);
            nccl_ep::wait_lsa_completion(param.lsa_S2G_flags, expected_val, param.ranks_per_lsa_team);
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
            _wt_head_end = clock64();
            param.block_timing[blockIdx.x].head_sync_start_clock = _wt_head_start;
            param.block_timing[blockIdx.x].head_sync_end_clock = _wt_head_end;
#endif
            // mbarrier init (both producer/consumer arrival counts = 1).
            for (int i = 0; i < STAGES_G2S; i++) {
                if constexpr (LSA_TEAMS != 1) {
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->lsa_mbarrier_G2S_buffer + 2 * i, 1);
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->lsa_mbarrier_G2S_buffer + 2 * i + 1, 1);
                }
                cuda::ptx::mbarrier_init(smem_buffer_ptr->cross_lsa_mbarrier_G2S_buffer + 2 * i, 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->cross_lsa_mbarrier_G2S_buffer + 2 * i + 1, 1);
            }
            if constexpr (LSA_TEAMS != 1) {
                for (int i = 0; i < LSA_TEAMS - 1; i++)
                    for (int j = 0; j < MAX_CHUNKS_PER_RANK; j++)
                        cuda::ptx::mbarrier_init(
                            smem_buffer_ptr->lsa_to_rdma_mbarrier_buffer + i * MAX_CHUNKS_PER_RANK + j,
                            1);
                *(smem_buffer_ptr->rdma_streaming_counter) = 0u;
            }
            cuda::ptx::fence_proxy_async(
                cuda::ptx::space_shared); // make mbarrier init visible to the async (TMA) proxy
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size()) {
        // warp 1: cross-LSA-team RDMA guard wait.
        if constexpr (LSA_TEAMS != 1) {
            if (param.guard_enabled)
                warp_rdma_guard_wait(
                    reinterpret_cast<const uint64_t*>(
                        reinterpret_cast<const uint8_t*>(param.gin_base_ptr) + param.mr_info.guard_offset),
                    my_lteam,
                    LSA_TEAMS,
                    *param.expected_gin_flag_val);
        }
    }

    // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
    __syncthreads();

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    // Measure warp-group work only (starts after combine-head sync and setup barriers).
    long long _wt_start = 0;
    if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif

    // Now warps can become specialized.
    // The input warp group data type must match the warp groups layout.
    // To prevent compiler generate pointless comparison warning.
    int threadIdx_x_int = (int)threadIdx.x;
    if (threadIdx_x_int < LSA_RED_GROUP::size()) {
        if constexpr (LSA_TEAMS != 1) {
            // Intra-LSA reduction warp group.
#define COMBINE_RED_INTRA_TEMPLATE \
            combine_RED_intra_warp<LSA_RED_GROUP, cur_smem_t, STAGES_G2S, STAGES_S2G, \
                                   TOKENS_PER_CHUNK, MAX_TOKENS_PER_RANK, LSA_TEAMS, NBLOCKS, \
                                   BACKWARD_COMBINE, HIDDEN_DIM, LSA_TEAM_SZ, \
                                   kTokenDtype, kCombineRecipe>
            COMBINE_RED_INTRA_TEMPLATE(
                // INPUT
                param.rdma_to_attn_map,
                // OUTPUT
                param.combine_gin_RED_tokens,
                param.combine_gin_RED_prob,
                // CONFIG
                my_lteam,
                param.num_of_tokens_per_rank,
                param.experts_per_rank,
                smem_buffer_ptr);
#undef COMBINE_RED_INTRA_TEMPLATE
        }
    } else if (threadIdx_x_int < LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size()) {
        // Cross-LSA-team reduction warp group.
#define COMBINE_RED_INTER_TEMPLATE \
        combine_RED_inter_warp<cur_smem_t, CROSS_LSA_RED_GROUP, PIPELINES_PER_BLOCK, STAGES_G2S, \
                                STAGES_S2G, TOKENS_PER_CHUNK, LSA_TEAMS, NBLOCKS, TOKENS_PER_GROUP, \
                                BACKWARD_COMBINE, HIDDEN_DIM, LSA_TEAM_SZ, kTokenDtype, kCombineRecipe>
        COMBINE_RED_INTER_TEMPLATE(
            // INPUT
            param.rdma_to_attn_map,
            param.attn_to_rdma_map,
            // OUTPUT
            param.attn_output_token,
            param.attn_output_prob,
            // CONFIG
            my_lteam,
            param.num_of_tokens_per_rank,
            param.num_real_tokens,
            param.experts_per_rank,
            smem_buffer_ptr);
#undef COMBINE_RED_INTER_TEMPLATE
    } else if (
        threadIdx_x_int < LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size() + LSA_G2S_GROUP::size()) {
        // Intra-LSA G2S warp group.
        if constexpr (LSA_TEAMS != 1) {
#define COMBINE_G2S_INTRA_TEMPLATE \
            combine_G2S_intra_warp<cur_smem_t, STAGES_G2S, TOKENS_PER_CHUNK, LSA_TEAMS, \
                                   LSA_TEAM_SZ, NBLOCKS, BACKWARD_COMBINE, HIDDEN_DIM, kLayout, kTokenDtype, \
                                   kCombineRecipe>
            COMBINE_G2S_INTRA_TEMPLATE(
                param.rdma_to_attn_map,
                param.sparse_to_dense_map,
                reinterpret_cast<const void* const*>(param.expert_input_token),
                param.expert_input_prob,
                my_lteam,
                param.num_of_tokens_per_rank,
                param.experts_per_rank,
                param.combine_local_reduce_enabled,
                smem_buffer_ptr);
#undef COMBINE_G2S_INTRA_TEMPLATE
        }
    } else if (
        threadIdx_x_int < LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size() + LSA_G2S_GROUP::size() +
                              CROSS_LSA_G2S_GROUP::size()) {
        // Cross-LSA-team G2S warp group.
#define COMBINE_G2S_INTER_TEMPLATE \
        combine_G2S_inter_warp<cur_smem_t, CROSS_LSA_G2S_GROUP, STAGES_G2S, TOKENS_PER_CHUNK, \
                                MAX_TOKENS_PER_RANK, LSA_TEAMS, NBLOCKS, TOKENS_PER_GROUP, \
                                BACKWARD_COMBINE, HIDDEN_DIM, LSA_TEAM_SZ, kLayout, kTokenDtype, kCombineRecipe>
        COMBINE_G2S_INTER_TEMPLATE(
            // INPUT
            param.rdma_to_attn_map,
            param.attn_to_rdma_map,
            param.sparse_to_dense_map,
            reinterpret_cast<const void* const*>(param.expert_input_token),
            param.expert_input_prob,
            param.combine_gin_G2S_tokens,
            param.combine_gin_G2S_prob,
            // OUTPUT
            param.gin_G2S_flags,
            smem_buffer_ptr,
            // CONFIG
            param.local_rank,
            my_lteam,
            param.num_of_tokens_per_rank,
            param.experts_per_rank,
            *param.expected_gin_flag_val,
            param.combine_local_reduce_enabled,
            param.dcomms,
            param.signals_base,
            param.combine_signal_offset,
            param.num_gin_comms,
            param.num_ctx_per_comm);
#undef COMBINE_G2S_INTER_TEMPLATE
    } else if (
        threadIdx_x_int < LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size() + LSA_G2S_GROUP::size() +
                              CROSS_LSA_G2S_GROUP::size() + GIN_GROUP::size()) {
        // Cross-LSA-team rdma warp group.
        if constexpr (LSA_TEAMS != 1) {
#define COMBINE_N2N_INTER_TEMPLATE \
            combine_N2N_inter_warp<GIN_GROUP, cur_smem_t, TOKENS_PER_CHUNK, \
                                   MAX_TOKENS_PER_RANK, LSA_TEAMS, NBLOCKS, LSA_TEAM_SZ, BACKWARD_COMBINE, \
                                   HIDDEN_DIM, kTokenDtype, kCombineRecipe>
            COMBINE_N2N_INTER_TEMPLATE(
                // INPUT
                param.rdma_to_attn_map,
                &param.mr_info,
                // SCRATCH
                smem_buffer_ptr,
                // CONFIG
                param.local_rank,
                my_lteam,
                param.num_of_tokens_per_rank,
                param.experts_per_rank,
                param.dcomms,
                param.token_window,
                param.prob_window,
                param.dest_window,
                param.num_gin_comms,
                param.num_ctx_per_comm,
                param.gin_base_ptr,
                param.signals_base,
                param.combine_signal_offset);
#undef COMBINE_N2N_INTER_TEMPLATE
        }
    } else {
        // Too many threads, should not goes here.
    }
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    if (threadIdx.x % 32 == 0) {
        int _warp_id = threadIdx.x / 32;
        int _idx = blockIdx.x * _WT_WARPS + _warp_id;
        param.warp_timing[_idx].work_start_clock = _wt_start;
        param.warp_timing[_idx].work_end_clock = clock64();
    }
#endif

    if (elect_last_block(reinterpret_cast<const int*>(param.combine_grid_barrier_counter), NBLOCKS)) {
        using tail_reset_warp = warp_group<1, 0>; // warp 0
        using tail_rdma_warp = warp_group<1, 1>; // warp 1
        using tail_lsa_warp = warp_group<1, 2>; // warp 2
        static_assert(
            LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size() + LSA_G2S_GROUP::size() +
                    CROSS_LSA_G2S_GROUP::size() + GIN_GROUP::size() >=
                3 * 32,
            "combine tail needs 3 warps");
        const int tail_tid = (int)threadIdx.x;
        if (tail_tid < tail_reset_warp::size()) {
            // warp 0: reset the grid counter + bump the intra-LSA round.
            if (tail_tid == 0) {
                atomicExch((unsigned int*)param.combine_grid_barrier_counter, 0u);
                *param.expected_lsa_flag_val += static_cast<uint32_t>(param.ranks_per_lsa_team);
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the cross-LSA-team RDMA guard, then bump the RDMA round.
            if constexpr (LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_gin_flag_val;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        param.dcomms[0],
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(my_lteam) * sizeof(uint64_t),
                        my_lteam,
                        LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_gin_flag_val = expected + 1ull;
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size() + tail_lsa_warp::size()) {
            // warp 2: intra-LSA LSA WAR barrier. Relaxed -- the tail __syncthreads already drained this
            // round's reads. Index = dispatch's block count (disjoint from dispatch's per-block [0, NB)).
            if constexpr (LSA_TEAM_SZ != 1) {
                if (param.guard_enabled) {
                    ncclLsaBarrierSession<ncclCoopWarp> bar(
                        ncclCoopWarp(),
                        param.dcomms[0],
                        ncclTeamTagLsa(),
                        param.combine_barrier_offset);
                    bar.sync(ncclCoopWarp(), cuda::memory_order_relaxed);
                }
            }
        }
    }
}

// ============================================================================
// Fills secondary EM slots from the primary slot in this rank's recv token
// buffer after dispatch (EM-unfused mode only).
// ============================================================================

template <typename T>
struct local_dup_kernel_param_t {
    T* expert_output_token; // [max_recv_tokens, hidden]
    float* expert_output_prob; // [max_recv_tokens, epr * ranks_per_lsa_team]; valid iff FORWARD_DISPATCH
    const int32_t* emuf_group_buf; // [num_groups, group_stride] = [primary, sec0, ..., -1]
    const int32_t* emuf_group_count; // scalar (produced by scan)
    int emuf_group_stride; // = experts_per_rank
    // S2G-completion flag dispatch polls; local_dup re-polls before reading primaries.
    const uint32_t* lsa_S2G_flag;
    // Shared with dispatch. When local_dup_enabled, dispatch defers the bump
    // here so peers only observe the flag move after secondaries are filled.
    uint32_t* expected_lsa_flag_val;
    // Reused from dispatch_grid_barrier_counter (dispatch leaves it at 0).
    uint32_t* grid_barrier_counter;
    int experts_per_rank;
    int ranks_per_lsa_team;
    void* expert_output_scale; // QUANT_FWD: per-token scale row; void* (dtype varies by recipe)
    int scale_row_bytes;
};

// Dynamic shared-memory bytes required by local_dup_kernel_impl for the given
// hidden_dim and token element size (token dtype is BF16/uint16_t).
inline int local_dup_dynamic_smem_bytes(
    int hidden_dim,
    int pipe_depth,
    bool forward_dispatch,
    int experts_per_rank,
    int ranks_per_lsa_team,
    size_t token_elem_bytes,
    int scale_row_bytes) {
    const int token_bytes = hidden_dim * static_cast<int>(token_elem_bytes);
    const int prob_bytes =
        forward_dispatch ? experts_per_rank * ranks_per_lsa_team * static_cast<int>(sizeof(float)) : 0;
    const int rings = pipe_depth * (token_bytes + prob_bytes + scale_row_bytes);
    const int mbar_bytes = pipe_depth * 2 * static_cast<int>(sizeof(uint64_t)) + 8;
    return rings + mbar_bytes;
}

template <typename T, int HIDDEN_DIM, int PIPE_DEPTH, bool FORWARD_DISPATCH,
          ncclEpDispQuant_t kRecipe = NCCL_EP_DISP_QUANT_NONE>
__device__ __forceinline__ void local_dup_kernel_impl(const local_dup_kernel_param_t<T>& p) {
    // Dispatch defers the expected-epoch advance until local duplication finishes.
    if (threadIdx.x == 0)
        nccl_ep::wait_lsa_completion(p.lsa_S2G_flag, *p.expected_lsa_flag_val, p.ranks_per_lsa_team);
    __syncthreads();

    constexpr int kTokenBytes = HIDDEN_DIM * sizeof(T);
    const int prob_floats = FORWARD_DISPATCH ? (p.experts_per_rank * p.ranks_per_lsa_team) : 0;
    const int prob_bytes = prob_floats * static_cast<int>(sizeof(float));

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* smem_ptr = smem_raw;
    T* smem_token[PIPE_DEPTH];
    float* smem_prob[PIPE_DEPTH];
    uint8_t* smem_scale[PIPE_DEPTH];
#pragma unroll
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        smem_token[s] = reinterpret_cast<T*>(smem_ptr);
        smem_ptr += kTokenBytes;
    }
    if constexpr (FORWARD_DISPATCH) {
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            smem_prob[s] = reinterpret_cast<float*>(smem_ptr);
            smem_ptr += prob_bytes;
        }
    }
    if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            smem_scale[s] = smem_ptr;
            smem_ptr += p.scale_row_bytes;
        }
    }
    smem_ptr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_ptr) + 7) & ~uintptr_t(7));
    uint64_t* prod_mbar = reinterpret_cast<uint64_t*>(smem_ptr);
    uint64_t* cons_mbar = prod_mbar + PIPE_DEPTH;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            cuda::ptx::mbarrier_init(&prod_mbar[s], 1);
            cuda::ptx::mbarrier_init(&cons_mbar[s], 1);
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    __shared__ int s_group_count;
    if (threadIdx.x == 0) s_group_count = *p.emuf_group_count;
    __syncthreads();

    const int N = s_group_count;
    if (N == 0) {
        // Dispatch deferred the bump; still owe peers the flag advance.
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t arrived = static_cast<uint32_t>(
                nccl_ep::atomic_add_acqrel_global(reinterpret_cast<const int*>(p.grid_barrier_counter), 1));
            if (arrived == gridDim.x - 1) {
                atomicExch((unsigned int*)p.grid_barrier_counter, 0u);
                *p.expected_lsa_flag_val += static_cast<uint32_t>(p.ranks_per_lsa_team);
            }
        }
        return;
    }

    const int block_id = blockIdx.x;
    const int n_blocks = gridDim.x;
    const int group_stride = p.emuf_group_stride;
    uint32_t total_tx = static_cast<uint32_t>(kTokenBytes + prob_bytes);
    if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD)
        total_tx += static_cast<uint32_t>(p.scale_row_bytes);

    if (warp_id == 0) {
        // Producer (G2S): 1 TMA load of the primary token per group.
        int stage = 0;
        uint32_t consumer_parity = 1;
        int iters_done = 0;
        for (int i = block_id; i < N; i += n_blocks) {
            if (iters_done >= PIPE_DEPTH) {
                while (!cuda::ptx::mbarrier_try_wait_parity(&cons_mbar[stage], consumer_parity)) {
                }
            }
            if (lane == 0) {
                const int primary_em = p.emuf_group_buf[i * group_stride + 0];
                const T* src_token = p.expert_output_token + static_cast<size_t>(primary_em) * HIDDEN_DIM;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_shared,
                    cuda::ptx::space_global,
                    smem_token[stage],
                    src_token,
                    kTokenBytes,
                    &prod_mbar[stage]);
                if constexpr (FORWARD_DISPATCH) {
                    const float* src_prob = p.expert_output_prob + static_cast<size_t>(primary_em) *
                                                                       (p.experts_per_rank * p.ranks_per_lsa_team);
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_shared,
                        cuda::ptx::space_global,
                        smem_prob[stage],
                        src_prob,
                        prob_bytes,
                        &prod_mbar[stage]);
                }
                if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
                    const uint8_t* src_scale =
                        static_cast<const uint8_t*>(p.expert_output_scale) + static_cast<size_t>(primary_em) * p.scale_row_bytes;
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_shared,
                        cuda::ptx::space_global,
                        smem_scale[stage],
                        src_scale,
                        p.scale_row_bytes,
                        &prod_mbar[stage]);
                }
                cuda::ptx::mbarrier_arrive_expect_tx(
                    cuda::ptx::sem_release,
                    cuda::ptx::scope_cta,
                    cuda::ptx::space_shared,
                    &prod_mbar[stage],
                    total_tx);
            }
            iters_done++;
            stage++;
            if (stage == PIPE_DEPTH) {
                stage = 0;
                consumer_parity ^= 1;
            }
        }
    } else if (warp_id == 1) {
        // Consumer (S2G): fan the primary stage out to every secondary in the row.
        constexpr int kMaxInFlightS2G = PIPE_DEPTH - 1;
        int stage = 0;
        uint32_t producer_parity = 0;
        int in_flight_s2g = 0;
        for (int i = block_id; i < N; i += n_blocks) {
            while (!cuda::ptx::mbarrier_try_wait_parity(&prod_mbar[stage], producer_parity)) {
            }
            if (lane == 0) {
                const int32_t* row = p.emuf_group_buf + static_cast<size_t>(i) * group_stride;
                for (int s = 1; s < group_stride; s++) {
                    const int sec = row[s];
                    if (sec < 0) break;
                    T* dst_token = p.expert_output_token + static_cast<size_t>(sec) * HIDDEN_DIM;
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_global,
                        cuda::ptx::space_shared,
                        dst_token,
                        smem_token[stage],
                        kTokenBytes);
                    if constexpr (FORWARD_DISPATCH) {
                        float* dst_prob = p.expert_output_prob +
                                          static_cast<size_t>(sec) * (p.experts_per_rank * p.ranks_per_lsa_team);
                        cuda::ptx::cp_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            dst_prob,
                            smem_prob[stage],
                            prob_bytes);
                    }
                    if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
                        uint8_t* dst_scale = static_cast<uint8_t*>(p.expert_output_scale) + static_cast<size_t>(sec) * p.scale_row_bytes;
                        cuda::ptx::cp_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            dst_scale,
                            smem_scale[stage],
                            p.scale_row_bytes);
                    }
                }
                cuda::ptx::cp_async_bulk_commit_group();
                in_flight_s2g++;
                if (in_flight_s2g > kMaxInFlightS2G) {
                    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<kMaxInFlightS2G>{});
                    in_flight_s2g--;
                    const int notify_stage = stage >= kMaxInFlightS2G
                        ? stage - kMaxInFlightS2G
                        : stage - kMaxInFlightS2G + PIPE_DEPTH;
                    cuda::ptx::mbarrier_arrive(&cons_mbar[notify_stage]);
                }
            }
            stage++;
            if (stage == PIPE_DEPTH) {
                stage = 0;
                producer_parity ^= 1;
            }
        }
        // Drain S2G before peers observe the flag bump below.
        if (lane == 0) {
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            nccl_ep::fence_proxy_async();
        }
    }

    // Last block owns the flag bump so peers only see it after secondaries land.
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        uint32_t arrived = static_cast<uint32_t>(
            nccl_ep::atomic_add_acqrel_global(reinterpret_cast<const int*>(p.grid_barrier_counter), 1));
        if (arrived == gridDim.x - 1) {
            atomicExch((unsigned int*)p.grid_barrier_counter, 0u);
            *p.expected_lsa_flag_val += static_cast<uint32_t>(p.ranks_per_lsa_team);
        }
    }
}

// ============================================================================
// Pre-sums secondary EM slots into the primary slot in expert_input_token
// (plus expert_input_prob for BACKWARD_COMBINE). Runs before combine in
// EM-unfused mode.
// ============================================================================

template <typename T>
struct local_reduce_kernel_param_t {
    T* expert_input_token; // [max_recv_tokens, hidden]
    float* expert_input_prob; // [max_recv_tokens, epr * ranks_per_lsa_team]; valid iff BACKWARD_COMBINE
    const int32_t* emuf_group_buf; // [num_groups, group_stride] = [primary, sec0, ..., -1]
    const int32_t* emuf_group_count; // scalar
    int emuf_group_stride; // = experts_per_rank
    int experts_per_rank;
    int ranks_per_lsa_team;
};

// Dynamic shared-memory bytes required by local_reduce_kernel_impl for the
// given hidden_dim (token dtype is BF16/uint16_t).
inline int local_reduce_dynamic_smem_bytes(int hidden_dim, int token_elem_bytes) {
    constexpr int kPipeDepth = NCCL_EP_HT_LOCAL_REDUCE_PIPE_DEPTH;
    constexpr int kOutStages = NCCL_EP_HT_LOCAL_REDUCE_OUT_STAGES;
    return (kPipeDepth + kOutStages) * hidden_dim * token_elem_bytes +
           2 * kPipeDepth * static_cast<int>(sizeof(uint64_t)) + 8;
}

template <typename T, int HIDDEN_DIM, int BLOCK_DIM, bool BACKWARD_COMBINE, ncclDataType_t kTokenDtype, int MAX_EXPERTS_PER_RANK>
__device__ __forceinline__ void local_reduce_kernel_impl(const local_reduce_kernel_param_t<T>& p) {
    static_assert(HIDDEN_DIM % 8 == 0, "HIDDEN_DIM must be a multiple of 8 (uint4 = 8 elems @2 B / 4 @4 B per vector)");
    static_assert(BLOCK_DIM % 32 == 0 && BLOCK_DIM >= 64, "BLOCK_DIM must be a multiple of 32 and at least 2 warps");

    // 1 producer warp (parallel G2S over lanes 0..n_src-1) + (W-1) consumer warps
    // (FP32 accumulate, BF16 cast, S2G).
    constexpr int kProdWarpCount = 1;
    constexpr int kWarpCount = BLOCK_DIM / 32;
    constexpr int kConsWarpCount = kWarpCount - kProdWarpCount;
    constexpr int kConsThreads = kConsWarpCount * 32;
    constexpr int kConsBarId = 1;

    // uint4 (16 B) holds 8 elems for 2-byte dtypes, 4 for FP32.
    constexpr int VEC_DIM = HIDDEN_DIM * static_cast<int>(sizeof(T)) / 16;
    constexpr int VEC_PER_THREAD = (VEC_DIM + kConsThreads - 1) / kConsThreads;
    constexpr int kTokenBytes = HIDDEN_DIM * sizeof(T);

    constexpr int PIPE_DEPTH = NCCL_EP_HT_LOCAL_REDUCE_PIPE_DEPTH;
    constexpr int kOutStages = NCCL_EP_HT_LOCAL_REDUCE_OUT_STAGES;

    const int N = *p.emuf_group_count;
    if (N == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int stride = p.emuf_group_stride;

    // n_src per group must be <= PIPE_DEPTH (also <= 32 since PIPE_DEPTH <= 32);
    // enforced by __trap in the producer loop once n_src is known.
    const int PROB_DIM = BACKWARD_COMBINE ? (p.experts_per_rank * p.ranks_per_lsa_team) : 0;

    // Shmem layout:
    //   s_in[PIPE_DEPTH] x kTokenBytes        (G2S ring)
    //   s_out[kOutStages] x kTokenBytes        (S2G ring)
    //   prod_mbar[PIPE_DEPTH], cons_mbar[PIPE_DEPTH]
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* smem_ptr = smem_raw;
    T* s_in[PIPE_DEPTH];
#pragma unroll
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        s_in[s] = reinterpret_cast<T*>(smem_ptr);
        smem_ptr += kTokenBytes;
    }
    T* s_out[kOutStages];
#pragma unroll
    for (int s = 0; s < kOutStages; ++s) {
        s_out[s] = reinterpret_cast<T*>(smem_ptr);
        smem_ptr += kTokenBytes;
    }
    smem_ptr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_ptr) + 7) & ~uintptr_t(7));
    uint64_t* prod_mbar = reinterpret_cast<uint64_t*>(smem_ptr);
    uint64_t* cons_mbar = prod_mbar + PIPE_DEPTH;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            cuda::ptx::mbarrier_init(&prod_mbar[s], 1);
            cuda::ptx::mbarrier_init(&cons_mbar[s], 1);
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    const T* token_base = reinterpret_cast<const T*>(p.expert_input_token);
    T* token_base_w = reinterpret_cast<T*>(p.expert_input_token);
    const int n_blocks = gridDim.x;
    const int my_block = blockIdx.x;

    if (warp_id == 0) {
        // PRODUCER WARP (cooperative G2S): lanes 0..n_src-1 issue TMA in parallel for
        // the current group's primary+secondaries. After each group, advance the global
        // stage offset by n_src so the next group's lanes target the next set of stages.
        int global_offset = 0;
        for (int i = my_block; i < N; i += n_blocks) {
            const int32_t* row = p.emuf_group_buf + static_cast<size_t>(i) * stride;
            // Lane 0 scans the row terminator and broadcasts n_src.
            int n_src;
            if (lane == 0) {
                int n = 1; // primary
                for (int s = 1; s < stride; s++) {
                    if (row[s] < 0) break;
                    n++;
                }
                n_src = n;
            }
            n_src = __shfl_sync(0xffffffff, n_src, 0);
            if (lane == 0 && n_src > PIPE_DEPTH) {
                __trap();
            }

            if (lane < n_src) {
                const int absolute = global_offset + lane;
                const int stage = absolute % PIPE_DEPTH;
                const uint32_t parity = 1u ^ (static_cast<uint32_t>(absolute / PIPE_DEPTH) & 1u);
                while (!cuda::ptx::mbarrier_try_wait_parity(&cons_mbar[stage], parity)) {
                }
                const int slot = row[lane]; // row[0]=primary, row[1..n_sec]=secondaries
                const T* src = token_base + static_cast<size_t>(slot) * HIDDEN_DIM;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_shared,
                    cuda::ptx::space_global,
                    s_in[stage],
                    src,
                    kTokenBytes,
                    &prod_mbar[stage]);
                cuda::ptx::mbarrier_arrive_expect_tx(
                    cuda::ptx::sem_release,
                    cuda::ptx::scope_cta,
                    cuda::ptx::space_shared,
                    &prod_mbar[stage],
                    kTokenBytes);
            }
            global_offset += n_src;
        }
    } else {
        // Consumer warps: FP32-accumulate n_src sources, BF16-cast, S2G to primary.
        int stage = 0;
        int absolute = 0;
        int out_slot = 0;
        const int cons_tid = tid - 32;

        for (int i = my_block; i < N; i += n_blocks) {
            const int32_t* row = p.emuf_group_buf + static_cast<size_t>(i) * stride;
            const int primary = row[0];
            int n_sec = 0;
            int secondaries[MAX_EXPERTS_PER_RANK];
#pragma unroll 1
            for (int s = 1; s < stride; s++) {
                int v = row[s];
                if (v < 0) break;
                secondaries[n_sec++] = v;
            }
            const int n_src = 1 + n_sec;

            // FP32 accumulator in registers (4 float2's per uint4 slot -> 8 floats).
            float2 acc[VEC_PER_THREAD][4];
#pragma unroll
            for (int n = 0; n < VEC_PER_THREAD; n++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    acc[n][k].x = 0.f;
                    acc[n][k].y = 0.f;
                }
            }

            for (int k = 0; k < n_src; k++) {
                const uint32_t prod_parity = static_cast<uint32_t>(absolute / PIPE_DEPTH) & 1u;
                while (!cuda::ptx::mbarrier_try_wait_parity(&prod_mbar[stage], prod_parity)) {
                }
                // Consumer-only barrier: producer is racing ahead on later stages.
                arrive_and_wait(kConsThreads, kConsBarId);

                const uint4* in_vec = reinterpret_cast<const uint4*>(s_in[stage]);
#pragma unroll
                for (int n = 0; n < VEC_PER_THREAD; n++) {
                    const int e = n * kConsThreads + cons_tid;
                    if (e < VEC_DIM) {
                        uint4 v = in_vec[e];
                        // uint4 holds 2 FP32 pairs or 4 packed 16-bit pairs; decode each pair
                        // to FP32 and accumulate (the helper picks the per-dtype unpack).
                        constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                        for (int kk = 0; kk < kPairs; kk++) {
                            float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&v, kk);
                            acc[n][kk].x += f.x;
                            acc[n][kk].y += f.y;
                        }
                    }
                }
                arrive_and_wait(kConsThreads, kConsBarId);
                if (tid == 32) {
                    cuda::ptx::mbarrier_arrive(&cons_mbar[stage]);
                }
                absolute++;
                stage++;
                if (stage == PIPE_DEPTH) stage = 0;
            }

            // Drain prior S2G before reusing s_out[out_slot].
            if (tid == 32) {
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<kOutStages - 1>{});
            }
            arrive_and_wait(kConsThreads, kConsBarId);

            uint4* out_vec = reinterpret_cast<uint4*>(s_out[out_slot]);
#pragma unroll
            for (int n = 0; n < VEC_PER_THREAD; n++) {
                const int e = n * kConsThreads + cons_tid;
                if (e < VEC_DIM) {
                    uint4 out;
                    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                    for (int kk = 0; kk < kPairs; kk++) nccl_ep::st_token_pair<kTokenDtype>(&out, kk, acc[n][kk]);
                    out_vec[e] = out;
                }
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            arrive_and_wait(kConsThreads, kConsBarId);

            if (tid == 32) {
                T* dst = token_base_w + static_cast<size_t>(primary) * HIDDEN_DIM;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst,
                    s_out[out_slot],
                    kTokenBytes);
                cuda::ptx::cp_async_bulk_commit_group();
            }
            out_slot ^= 1;

            if constexpr (BACKWARD_COMBINE) {
                float* prim_prob = p.expert_input_prob + static_cast<size_t>(primary) * PROB_DIM;
                for (int e = cons_tid; e < PROB_DIM; e += kConsThreads) {
                    float a = prim_prob[e];
#pragma unroll 1
                    for (int s = 0; s < n_sec; s++) {
                        const float* sec_prob = p.expert_input_prob + static_cast<size_t>(secondaries[s]) * PROB_DIM;
                        a += sec_prob[e];
                    }
                    prim_prob[e] = a;
                }
            }
        }
        if (tid == 32) {
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            nccl_ep::fence_proxy_async();
        }
    }
}

// Local EM permute kernel (HT + EM + zero_copy != ON). Scatters FLAT staging
// rows into per-expert EM zones with zero padding for inactive slots.
//
// Two concurrent warp groups in each block:
//   - kPermuteWarps token-permute warps: load each FLAT row once into
//     registers and scatter to its EM destinations (load-once-scatter-many,
//     one HBM read amortized over up to top_k STGs). The int4 unroll length
//     HiddenVec is JIT'd per HiddenInt4 (see pick_dup_hidden_vec) to trade
//     ILP against register pressure.
//   - kPadWarps pad-fill warps: cp.async.bulk S2G from a smem zero buffer to
//     per-expert pad rows. TMA path doesn't compete with LDG/STG queues.
// Disjoint output rows + disjoint memory paths => the two groups overlap.
//
// Bounds the per-warp smem active-EM list: a token occupies at most top_k EM slots in this
// rank, so this caps top_k -- experts_per_rank is unbounded here.
constexpr int kLocalPermuteMaxActivePerToken = 256;
constexpr int kLocalPermuteThreadsPerSlot = 32;
constexpr int kLocalPermutePermuteWarps = 8;
constexpr int kLocalPermutePadWarps = 1;
constexpr int kLocalPermuteTokensPerBlock = kLocalPermutePermuteWarps;
constexpr int kLocalPermuteThreads =
    kLocalPermuteThreadsPerSlot * (kLocalPermutePermuteWarps + kLocalPermutePadWarps); // 288
// local_permute_reduce: 128 threads per slot, S slots per block. 8 slots
// (1024 threads) puts 8 independent token loads in flight per block for
// better HBM/L2 latency hiding; single block per SM at this size.
constexpr int kLocalPermuteReduceSlotsPerBlock = 8;
constexpr int kLocalPermuteReduceThreads = 128 * kLocalPermuteReduceSlotsPerBlock;
constexpr int kLocalPermuteReduceBlocksPerSM = 1;
// combine_push block-size upper bound for __launch_bounds__ (24 warps). The real block is sized
// at launch to fit the per-warp smem row buffer (see combine_push_slots_per_block); the kernel
// reads blockDim.x at runtime.
constexpr int kCombinePushThreads = 768;
constexpr int kCombinePushMaxSlots = kCombinePushThreads / 32;  // 24, sizes static metadata smem
// Dynamic smem = slots * row bytes (one per-warp row buffer each). Static metadata is separate.
__host__ __device__ constexpr size_t combine_push_dynamic_smem_bytes(int hidden_int4, int slots) {
    return static_cast<size_t>(slots) * hidden_int4 * 16;
}
// Largest slot count (<= kCombinePushMaxSlots) whose per-warp row buffer fits smem_cap, reserving
// headroom for the static metadata arrays. At least 1.
__host__ __device__ constexpr int combine_push_slots_per_block(int hidden_int4, int smem_cap) {
    const int row_bytes = hidden_int4 * 16;
    // static smem headroom: flat2em[MaxTopK<=32] + nvalid + dst ptr + mbarrier, per slot.
    const int meta_reserve = kCombinePushMaxSlots * (32 * 4 + 4 + 8 + 8) + 256;
    int slots = (smem_cap - meta_reserve) / row_bytes;
    if (slots > kCombinePushMaxSlots) slots = kCombinePushMaxSlots;
    if (slots < 1) slots = 1;
    return slots;
}

struct local_permute_dup_param_t {
    void* recv_x_em;
    float* recv_topk_weights_em;
    const void* flat_staging;
    const float* recv_topk_weights_flat;
    const int32_t* flat2em_slot_map;
    const int32_t* num_recv_tokens_dev;
    const int64_t* expert_token_offsets;
    // Handle-scoped write-gate/pad boundary, read by every call (forward and
    // backward). Mutable: see allow_overflow_drop below.
    int32_t* per_expert_counts_active;
    int top_k;
    int experts_per_rank;
    int row_bytes;
    int caller_num_recv_tokens;   // caller recv buffer row capacity
    // QUANT_FWD: per-token scale rows relocated with the same slot map as
    // the tokens. recv_scales_em/flat_scale_staging are null and scale_row_bytes
    // is 0 for NONE mode. scale_row_bytes is 16B-aligned (host-validated).
    void* recv_scales_em;
    const void* flat_scale_staging;
    int scale_row_bytes;
    // Count mode: build the flat2em rows inline from the sender-written LERM rows,
    // assigning em slots via per-expert atomic cursors (zero-seeded by the host).
    // All null on the stock path (pre-built flat2em consumed as-is). lerm_out
    // persists the LERM row into per-handle mem so combine reads a private copy.
    int32_t* flat2em_out;
    const bool* lerm_in;
    bool* lerm_out;
    // [experts_per_rank + 1]: per-expert atomic write cursors, plus one trailing grid
    // arrival counter this kernel's last-arriving block uses for the DROP phantom-row
    // fixup below (both zero-seeded by the host once per dispatch).
    int32_t* em_permute_cursors;
    // DROP only: build_s2d's FLAT-level drop (sender-rank order) and publish_outputs's
    // EM-zone padding drop (expert-major order) can keep different token sets under
    // overflow, so per_expert_counts_active[e] can overstate em_permute_cursors[e]'s true
    // final tally -- the gap is a "phantom" row no permute warp ever writes. When true,
    // the last-arriving block zeroes that gap in-place and corrects
    // per_expert_counts_active[e] to match (no extra kernel). Backward reuses forward's
    // persisted flat2em_slot_map and never reruns publish_outputs, so this correction is
    // also what lets its own pad warp zero the same gap in its own buffer.
    bool allow_overflow_drop;
};

template <int HiddenInt4, int HiddenVec,
          ncclEpDispQuant_t kRecipe = NCCL_EP_DISP_QUANT_NONE>
__device__ __forceinline__ void local_permute_dup(
    uint8_t* __restrict__ recv_x_em,
    float* __restrict__ recv_topk_weights_em,
    const uint8_t* __restrict__ flat_staging,
    const float* __restrict__ recv_topk_weights_flat,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const int64_t* __restrict__ expert_token_offsets,
    int32_t* __restrict__ per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int /*row_bytes*/,
    int caller_num_recv_tokens,
    // QUANT_FWD: relocate a per-token scale row alongside the token row.
    // scale_dst/scale_src null and scale_row_bytes==0 for NONE (scale copy skipped).
    uint8_t* __restrict__ scale_dst,
    const uint8_t* __restrict__ scale_src,
    int scale_row_bytes,
    int32_t* __restrict__ flat2em_out,
    const bool* __restrict__ lerm_in,
    bool* __restrict__ lerm_out,
    int32_t* __restrict__ em_permute_cursors,
    bool allow_overflow_drop) {
    constexpr int kThreadsPerSlot = kLocalPermuteThreadsPerSlot;
    constexpr int kHiddenVec = HiddenVec;
    constexpr int kPermuteWarps = kLocalPermutePermuteWarps;
    constexpr int kPadWarps = kLocalPermutePadWarps;
    // Per-warp cap on active EM slots per token; a token routes to at most top_k local experts.
    constexpr int kMaxActivePerToken = kLocalPermuteMaxActivePerToken;

    const int warp_id = threadIdx.x / kThreadsPerSlot;
    const int lane = threadIdx.x & (kThreadsPerSlot - 1);
    const bool is_pad = warp_id >= kPermuteWarps;
    const int pad_idx = warp_id - kPermuteWarps;

    // Caller recv buffers must hold the full padded EM total. Host checks
    // enforce this per mode (budget or actual rows); this is the backstop.
    const int64_t em_padded_total = expert_token_offsets[experts_per_rank];
    EP_DEVICE_ASSERT(caller_num_recv_tokens >= em_padded_total);
    // Weights relocate as a pair: an EM destination requires the FLAT source.
    EP_DEVICE_ASSERT((recv_topk_weights_em != nullptr) == (recv_topk_weights_flat != nullptr));
    constexpr int row_int4 = HiddenInt4;
    constexpr int row_bytes = HiddenInt4 * 16;

    const int4* src_int4 = reinterpret_cast<const int4*>(flat_staging);
    int4* dst_int4 = reinterpret_cast<int4*>(recv_x_em);

    __shared__ int32_t s_active[kPermuteWarps][kMaxActivePerToken];
    __shared__ int s_count[kPermuteWarps];
    const int num_recv = *num_recv_tokens_dev;

    // Source row for the pad warp's cp.async.bulk S2G.
    extern __shared__ int4 s_zero[];

    const int zero_bytes = row_bytes > scale_row_bytes ? row_bytes : scale_row_bytes;
    const int zero_int4 = zero_bytes / sizeof(int4);
    for (int j = threadIdx.x; j < zero_int4; j += blockDim.x) {
        s_zero[j] = int4{0, 0, 0, 0};
    }
    __syncthreads();

    if (is_pad) {
        // 32 lanes stripe pad slots across the grid; each lane issues
        // cp.async.bulk S2G from s_zero to its assigned slots.
        const int total_pad_lanes = kThreadsPerSlot * kPadWarps * static_cast<int>(gridDim.x);
        const int my_pad_lane = (static_cast<int>(blockIdx.x) * kPadWarps + pad_idx) * kThreadsPerSlot + lane;
        // Host validates row_bytes % 16 == 0; pass it straight to cp.async.bulk.
        assert((row_bytes & 15) == 0);
        const bool zero_weights = (recv_topk_weights_em != nullptr);
        // Running position in the concatenated stream of pad rows across all experts
        // seen so far. Under random routing only a handful of (low-index) experts carry
        // any pad_count at all, and each one is tiny relative to total_pad_lanes -- if
        // every expert's stripe restarted at my_pad_lane, the same few low-numbered lanes
        // would be hit by every expert while the rest of the grid sat idle. Continuing the
        // stripe across experts (global_pos accumulates, never resets) spreads that same
        // total work evenly over all total_pad_lanes lanes instead.
        int64_t global_pos = 0;
        for (int e = 0; e < experts_per_rank; ++e) {
            const int64_t zone_start = expert_token_offsets[e];
            const int64_t zone_end = expert_token_offsets[e + 1];
            const int32_t active = per_expert_counts_active[e];
            const int64_t pad_begin = zone_start + active;
            const int64_t pad_count = zone_end - pad_begin;
            int64_t rel = (my_pad_lane - global_pos) % total_pad_lanes;
            if (rel < 0) rel += total_pad_lanes;
            global_pos += pad_count;
            for (int64_t offs = rel; offs < pad_count; offs += total_pad_lanes) {
                const int64_t slot = pad_begin + offs;
                uint8_t* dst_g = recv_x_em + static_cast<size_t>(slot) * row_bytes;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst_g,
                    reinterpret_cast<uint8_t*>(s_zero),
                    row_bytes);
                if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_global,
                        cuda::ptx::space_shared,
                        scale_dst + static_cast<size_t>(slot) * scale_row_bytes,
                        reinterpret_cast<uint8_t*>(s_zero),
                        scale_row_bytes);
                }
                if (zero_weights) {
                    recv_topk_weights_em[slot] = 0.0f;
                }
            }
        }
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        nccl_ep::fence_proxy_async();
    } else {
        // Permute warps: warp-strided token permute; count mode builds flat2em inline
        // below, other modes use the pre-built map.
        for (int token = static_cast<int>(blockIdx.x) * kPermuteWarps + warp_id;
             token < num_recv;
             token += kPermuteWarps * static_cast<int>(gridDim.x)) {
            const bool copy_weights =
                recv_topk_weights_em != nullptr && recv_topk_weights_flat != nullptr;
            if (flat2em_out != nullptr) {
                // Count mode: assign em slots warp-cooperatively from the LERM row
                // (ascending local expert, matching the FLAT weights order). The
                // persisted row keeps dispatch and combine consistent, so any
                // per-expert slot order is valid.
                // LERM is a packed bitmap: ceil(epr/64) u64 words per slot.
                const int lerm_words = nccl_ep::bit_words(experts_per_rank);
                const uint64_t* lrow =
                    reinterpret_cast<const uint64_t*>(lerm_in) + static_cast<size_t>(token) * lerm_words;
                // Persist the LERM row into per-handle mem for combine (private per dispatch).
                if (lerm_out != nullptr) {
                    uint64_t* lrow_out =
                        reinterpret_cast<uint64_t*>(lerm_out) + static_cast<size_t>(token) * lerm_words;
                    for (int w = lane; w < lerm_words; w += kThreadsPerSlot) lrow_out[w] = lrow[w];
                }
                int32_t* out_row = flat2em_out + static_cast<size_t>(token) * top_k;
                int c = 0;
                for (int base = 0; base < experts_per_rank; base += kThreadsPerSlot) {
                    const int e = base + lane;
                    const bool hit = (e < experts_per_rank) && nccl_ep::test_bit(lrow, e);
                    const uint32_t bal = __ballot_sync(0xffffffffu, hit);
                    if (hit) {
                        const int pos = c + __popc(bal & ((1u << lane) - 1u));
                        // Cursor past the (DROP-clamped) per-expert count overflows the recv
                        // buffer; mark it dropped (-1) so the copy loops and combine skip it.
                        const int cur = atomicAdd(&em_permute_cursors[e], 1);
                        const int32_t es = (cur < per_expert_counts_active[e])
                                               ? static_cast<int32_t>(expert_token_offsets[e]) + cur
                                               : -1;
                        if (pos < kMaxActivePerToken) s_active[warp_id][pos] = es;
                        out_row[pos] = es;
                        if (copy_weights && es >= 0) {
                            recv_topk_weights_em[es] =
                                recv_topk_weights_flat[static_cast<size_t>(token) * top_k + pos];
                        }
                    }
                    c += __popc(bal);
                }
                for (int k = c + lane; k < top_k; k += kThreadsPerSlot) out_row[k] = -1;
                if (lane == 0) s_count[warp_id] = c;
            } else if (lane == 0) {
                int c = 0;
                const int32_t* slot_row = flat2em_slot_map + static_cast<size_t>(token) * top_k;
                for (int k = 0; k < top_k; ++k) {
                    const int32_t es = __ldg(slot_row + k);
                    if (es < 0) continue;
                    // Per-slot capacity backstop: a map entry past the caller buffer
                    // means the scan's drop guard failed — trap instead of an OOB
                    // write (the aggregate offsets check above cannot see this).
                    EP_DEVICE_ASSERT(es < caller_num_recv_tokens);
                    if (c < kMaxActivePerToken) s_active[warp_id][c] = es;
                    if (copy_weights) {
                        recv_topk_weights_em[es] = recv_topk_weights_flat[static_cast<size_t>(token) * top_k + k];
                    }
                    ++c;
                }
                s_count[warp_id] = c;
            }
            __syncwarp();

            const int4* src = src_int4 + static_cast<size_t>(token) * row_int4;
            const int cnt = s_count[warp_id];

            constexpr int kStride = kThreadsPerSlot * kHiddenVec;
            constexpr int j_main_end = (row_int4 / kStride) * kStride;
            for (int j_base = 0; j_base < j_main_end; j_base += kStride) {
                int4 buf[kHiddenVec];
#pragma unroll
                for (int u = 0; u < kHiddenVec; ++u) {
                    buf[u] = src[j_base + u * kThreadsPerSlot + lane];
                }
                for (int a = 0; a < cnt; ++a) {
                    const int32_t as = s_active[warp_id][a];
                    if (as < 0) continue;  // dropped slot
                    int4* dst = dst_int4 + static_cast<size_t>(as) * row_int4;
#pragma unroll
                    for (int u = 0; u < kHiddenVec; ++u) {
                        int4* p = dst + j_base + u * kThreadsPerSlot + lane;
                        nccl_ep::st_cg_global(p, buf[u]);
                    }
                }
            }
            if constexpr (j_main_end < row_int4) {
                for (int j = j_main_end + lane; j < row_int4; j += kThreadsPerSlot) {
                    const int4 v = src[j];
                    for (int a = 0; a < cnt; ++a) {
                        const int32_t as = s_active[warp_id][a];
                        if (as < 0) continue;  // dropped slot
                        int4* dst = dst_int4 + static_cast<size_t>(as) * row_int4;
                        nccl_ep::st_cg_global(&dst[j], v);
                    }
                }
            }

            if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
                const int scale_row_int4 = scale_row_bytes >> 4;
                const int4* ssrc =
                    reinterpret_cast<const int4*>(static_cast<const uint8_t*>(scale_src) + static_cast<size_t>(token) * scale_row_bytes);
                for (int a = 0; a < cnt; ++a) {
                    const int32_t as = s_active[warp_id][a];
                    if (as < 0) continue;  // dropped slot
                    int4* sdst = reinterpret_cast<int4*>(
                        scale_dst + static_cast<size_t>(as) * scale_row_bytes);
                    for (int j = lane; j < scale_row_int4; j += kThreadsPerSlot) {
                        nccl_ep::st_cg_global(&sdst[j], ssrc[j]);
                    }
                }
            }
            __syncwarp();
        }
    }

    // Zero the phantom-row gap, if any (see allow_overflow_drop's doc comment on
    // local_permute_dup_param_t). Elect the last-arriving block, same pattern as
    // elect_last_block's other callers, once every block's cursor contributions land.
    if (allow_overflow_drop && em_permute_cursors != nullptr &&
        elect_last_block(em_permute_cursors + experts_per_rank, static_cast<int>(gridDim.x))) {
        for (int e = warp_id; e < experts_per_rank; e += kPermuteWarps + kPadWarps) {
            const int32_t active = per_expert_counts_active[e];
            const int32_t delivered = em_permute_cursors[e];
            const int32_t true_active = delivered < active ? delivered : active;
            if (true_active >= active) continue;
            // Correct the handle-scoped field in place (see allow_overflow_drop above).
            if (lane == 0) per_expert_counts_active[e] = true_active;
            const int64_t zone_start = expert_token_offsets[e];
            for (int64_t slot = zone_start + true_active; slot < zone_start + active; slot++) {
                int4* tok_row = dst_int4 + static_cast<size_t>(slot) * row_int4;
                for (int j = lane; j < row_int4; j += kThreadsPerSlot) tok_row[j] = int4{0, 0, 0, 0};
                if (recv_topk_weights_em != nullptr && lane == 0) recv_topk_weights_em[slot] = 0.0f;
                if constexpr (kRecipe == NCCL_EP_DISP_QUANT_FWD) {
                    const int scale_row_int4 = scale_row_bytes >> 4;
                    int4* sdst = reinterpret_cast<int4*>(scale_dst + static_cast<size_t>(slot) * scale_row_bytes);
                    for (int j = lane; j < scale_row_int4; j += kThreadsPerSlot) sdst[j] = int4{0, 0, 0, 0};
                }
            }
        }
    }
    __syncthreads(); // pad TMAs / phantom-row fixup must complete before block exits.
}

// Pull EM dispatch kernel (HT + EM + MNNVL). Each receiver reads the source rows it
// needs directly from the source ranks' input buffers over NVLink and scatters them
// into its own EM layout in one pass, via a per-warp PullTmaStage (TMA the row into
// smem, then TMA-store to each EM slot). Weights are pulled from the source's
// per-token topk_weights row (position k aligns with flat2em_slot_map's k).
//
// Two concurrent warp groups per block:
//   - kPullWarps pull warps: one warp per recv token; TMA the remote token row once
//     and scatter it to the token's EM slots. Each warp also pulls that token's
//     topk_weights (and, for QUANT_FWD, its scale row) from the source rank and
//     scatters them to the same slots.
//   - kPadWarps pad warps: STG zero-fill the per-expert pad rows, kept off the
//     TMA engine the pull warps use.
// Output rows are disjoint, so the groups overlap with only the block-exit sync.
//
// Pull warps per CTA. Each warp keeps one remote row in flight; the JIT picks the
// largest count whose per-warp smem staging fits the device, capped here.
constexpr int kPullDispatchMaxPullWarps = 24;
constexpr int kPullDispatchPadWarps = 1;
constexpr int kPullDispatchThreadsPerSlot = 32;  // one warp per recv slot
constexpr int pull_dispatch_threads(int pull_warps) {
    return kPullDispatchThreadsPerSlot * (pull_warps + kPullDispatchPadWarps);
}
// Max EM slots one recv token can scatter to on this rank (== MAX_NUM_TOPK: a token
// routes to at most top_k <= 32 experts, so at most 32 kept copies land here).
constexpr int kPullDispatchMaxActive = 32;

// Max NVLink single-LSA-team ranks. The peer-pointer arrays live inline in the kernel
// param struct so they ride the stream-ordered launch marshaling; a persistent host
// staging + async H2D would race across back-to-back calls.
constexpr int kPullMaxLsaRanks = 128;

// Decode a packed global token id into its (LSA rank, local token) within the team:
// g == src_rank * tokens_per_rank + src_token (mod the team span).
struct decoded_src_t {
    int rank;
    int token;
};
__device__ __forceinline__ decoded_src_t decode_src(int32_t g, int tokens_per_rank, int lsa_team_size) {
    return {(g / tokens_per_rank) % lsa_team_size, g % tokens_per_rank};
}

// Block-uniform LSA grid head gate shared by pull dispatch and push combine.
// TODO: try replacing the ncclLsaBarrier rendezvous with a bare peer-ptr + atomic-flag sync
// (as EM local-permute uses) and measure whether it lowers the head/tail sync overhead.
__device__ __forceinline__ void lsa_grid_head_gate(ncclDevComm_t* dcomms, uint32_t* head_sync_flag) {
    int* head_flag = reinterpret_cast<int*>(head_sync_flag);
    if (blockIdx.x == 0) {
        if (threadIdx.x < 32) {
            ncclLsaBarrierSession<ncclCoopWarp> bar(ncclCoopWarp(), dcomms[0], ncclTeamTagLsa(), 0u);
            bar.sync(ncclCoopWarp(), cuda::memory_order_acq_rel);
        }
        __syncthreads();
        // Sys-scope hand-off: the waiter blocks read remote NVLink peer memory after this gate,
        // so their acquire must invalidate stale peer cache lines (a .gpu acquire only covers the
        // local GPU coherence domain). The paired release keeps the sync morally strong at sys scope.
        if (threadIdx.x == 0) nccl_ep::st_release_sys_global(head_flag, 1);
    } else {
        if (threadIdx.x == 0) {
            while (nccl_ep::ld_relaxed_gpu_global(head_sync_flag) == 0) {}
            (void)nccl_ep::ld_acquire_sys_global(head_flag);
        }
        __syncthreads();
    }
}

// Block-uniform LSA grid tail barrier shared by pull dispatch and push combine.
__device__ __forceinline__ void lsa_grid_tail_barrier(
    ncclDevComm_t* dcomms,
    uint32_t* grid_barrier_counter,
    uint32_t* head_sync_flag) {
    __threadfence_system();
    if (elect_last_block(reinterpret_cast<const int*>(grid_barrier_counter), static_cast<int>(gridDim.x))) {
        if (threadIdx.x < 32) {
            ncclLsaBarrierSession<ncclCoopWarp> bar(ncclCoopWarp(), dcomms[0], ncclTeamTagLsa(), 1u);
            bar.sync(ncclCoopWarp(), cuda::memory_order_acq_rel);
        }
        if (threadIdx.x == 0) {
            atomicExch(grid_barrier_counter, 0u);
            atomicExch(head_sync_flag, 0u);
        }
    }
}

struct dispatch_pull_param_t {
    void* recv_x_em;                            // EM output token buffer
    float* recv_topk_weights_em;                // EM 1D weights out (null if not delivered)
    uint8_t* recv_x_scale_em;                   // QUANT_FWD: EM output scale buffer (null for NONE)
    const int32_t* flat2em_slot_map;            // [num_recv, top_k]
    const int32_t* srcpos_map;                  // [num_recv, top_k] source top-k pos of each hit (weights)
    const int32_t* recv_slot_to_src;            // [num_recv] recv slot -> global src token id
    const void* peer_input_ptrs[kPullMaxLsaRanks];   // [lsa_team_size] source token bases (NVLink)
    const float* peer_weight_ptrs[kPullMaxLsaRanks]; // [lsa_team_size] source topk_weights bases (null if not delivered)
    const void* peer_scale_ptrs[kPullMaxLsaRanks];   // [lsa_team_size] source scale bases (QUANT_FWD; null for NONE)
    const int32_t* num_recv_tokens_dev;
    const int64_t* expert_token_offsets;
    const int32_t* per_expert_counts_active;
    int top_k;
    int experts_per_rank;
    int row_bytes;
    int scale_row_bytes;                        // QUANT_FWD: per-token scale row bytes (0 for NONE)
    int caller_num_recv_tokens;                 // caller EM recv buffer row capacity
    int tokens_per_rank;                        // decode src token = g % tokens_per_rank
    int lsa_team_size;                          // decode src rank within the LSA team
    // Intra-LSA head/tail sync: source ready before any peer read, reads done before reuse.
    ncclDevComm_t* dcomms;
    uint32_t* head_sync_flag;                   // grid head gate (idle counter during dispatch)
    uint32_t* grid_barrier_counter;             // tail elect-last-block
    // When set, head/tail sync runs as separate kernels; this kernel skips the inline sync.
    bool unfused_sync;
};

// Zero-fill the per-expert pad rows (disjoint from real slots) that the pull warps skip. One warp
// per pad row; the 32 lanes stripe zero int4s across the token row (and scale row + weight on FWD).
// Uses plain vectorized st.global (16B int4), not TMA: the source is a constant zero with no smem
// payload to bulk-copy.
template <int HiddenInt4, bool kFwd, int kThreadsPerSlot, int kPadWarps>
__device__ __forceinline__ void dispatch_pull_pad_fill(
    int4* dst_int4, uint8_t* recv_x_scale_em, float* recv_topk_weights_em,
    const int64_t* expert_token_offsets, const int32_t* per_expert_counts_active, int experts_per_rank,
    int scale_row_bytes, int scale_row_int4, int pad_idx, int lane) {
    const int total_pad_warps = kPadWarps * static_cast<int>(gridDim.x);
    const int my_pad_warp = static_cast<int>(blockIdx.x) * kPadWarps + pad_idx;
    const bool zero_weights = (recv_topk_weights_em != nullptr);
    const int4 zero_v = int4{0, 0, 0, 0};
    // Running offset across experts (mirrors local_permute_dup's pad-warp fix):
    // restarting at my_pad_warp per expert repeatedly hits the same warps, leaving
    // the rest of the grid's pad warps idle.
    int64_t global_pos = 0;
    for (int e = 0; e < experts_per_rank; ++e) {
        const int64_t pad_begin = expert_token_offsets[e] + per_expert_counts_active[e];
        const int64_t pad_count = expert_token_offsets[e + 1] - pad_begin;
        int64_t rel = (my_pad_warp - global_pos) % total_pad_warps;
        if (rel < 0) rel += total_pad_warps;
        global_pos += pad_count;
        for (int64_t r = rel; r < pad_count; r += total_pad_warps) {
            const int64_t slot = pad_begin + r;
            int4* row = dst_int4 + static_cast<size_t>(slot) * HiddenInt4;
            for (int j = lane; j < HiddenInt4; j += kThreadsPerSlot) {
                nccl_ep::st_cg_global(&row[j], zero_v);
            }
            if constexpr (kFwd) {
                int4* srow = reinterpret_cast<int4*>(
                    recv_x_scale_em + static_cast<size_t>(slot) * scale_row_bytes);
                for (int j = lane; j < scale_row_int4; j += kThreadsPerSlot) {
                    nccl_ep::st_cg_global(&srow[j], zero_v);
                }
            }
            if (zero_weights && lane == 0) recv_topk_weights_em[slot] = 0.0f;
        }
    }
}

// ScaleInt4PerLane: per-lane int4 capacity for the QUANT_FWD scale row (loaded once
// into registers, scattered to every EM slot). JIT-sized from the actual scale row
// (ceil(scale_row_int4 / 32)), so any scale dtype / quantization block size is covered.
// ScaleTmaRowBytes: when non-zero, the scale row is moved by TMA through smem instead
// (exact row size; 0 keeps the register-held ld/st path).
template <int HiddenInt4, class Policy,
          ncclEpDispQuant_t kRecipe = NCCL_EP_DISP_QUANT_NONE, int ScaleInt4PerLane = 1,
          int ScaleTmaRowBytes = 0, int PullWarps = kPullDispatchMaxPullWarps>
__device__ __forceinline__ void dispatch_pull(
    uint8_t* __restrict__ recv_x_em,
    float* __restrict__ recv_topk_weights_em,
    uint8_t* __restrict__ recv_x_scale_em,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ srcpos_map,
    const int32_t* __restrict__ recv_slot_to_src,
    const void* const* __restrict__ peer_input_ptrs,
    const float* const* __restrict__ peer_weight_ptrs,
    const void* const* __restrict__ peer_scale_ptrs,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const int64_t* __restrict__ expert_token_offsets,
    const int32_t* __restrict__ per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int /*row_bytes*/,
    int scale_row_bytes,
    int caller_num_recv_tokens,
    int tokens_per_rank,
    int lsa_team_size,
    ncclDevComm_t* __restrict__ dcomms,
    uint32_t* __restrict__ head_sync_flag,
    uint32_t* __restrict__ grid_barrier_counter,
    bool unfused_sync) {
    static_assert(kRecipe == NCCL_EP_DISP_QUANT_NONE || kRecipe == NCCL_EP_DISP_QUANT_FWD,
                  "pull dispatch: only NONE and FWD (scales-forward) recipes are supported");
    constexpr bool kFwd = (kRecipe == NCCL_EP_DISP_QUANT_FWD);
    constexpr bool kScaleTma = kFwd && (ScaleTmaRowBytes > 0);
    using ScalePolicy = PullTmaStage<kScaleTma ? ScaleTmaRowBytes : 16>;

    const int warp_id = threadIdx.x / kPullDispatchThreadsPerSlot;
    const int lane = threadIdx.x & (kPullDispatchThreadsPerSlot - 1);
    const bool is_pad = warp_id >= PullWarps;
    const int pad_idx = warp_id - PullWarps;

    const int num_recv = *num_recv_tokens_dev;
    const int64_t em_padded_total = expert_token_offsets[experts_per_rank];
    EP_DEVICE_ASSERT(caller_num_recv_tokens >= em_padded_total);

    int4* dst_int4 = reinterpret_cast<int4*>(recv_x_em);
    // QUANT_FWD: per-token scale row (16B units), forwarded alongside the token row.
    const int scale_row_int4 = kFwd ? (scale_row_bytes >> 4) : 0;
    EP_DEVICE_ASSERT(!kFwd || scale_row_int4 <= 32 * ScaleInt4PerLane);

    __shared__ int32_t s_active[PullWarps][kPullDispatchMaxActive];
    __shared__ int s_count[PullWarps];
    // Per-pull-warp policy staging.
    extern __shared__ uint8_t s_policy[];

    // One transfer policy per pull warp; smem policies publish their mbarrier
    // inits before the block-wide fence, which every warp (incl. pad) must reach.
    Policy p;
    ScalePolicy sp;
    if (!is_pad) {
        p.init(lane, s_policy + static_cast<size_t>(warp_id) * Policy::smem_bytes_per_warp);
        if constexpr (kScaleTma) {
            sp.init(lane, s_policy +
                              static_cast<size_t>(PullWarps) * Policy::smem_bytes_per_warp +
                              static_cast<size_t>(warp_id) * ScalePolicy::smem_bytes_per_warp);
        }
    }
    Policy::block_init_fence();

    // Head sync: every rank's source is ready+visible before any peer read. Block 0
    // rendezvous with every peer's block 0 (slot 0), then publishes the grid flag the
    // other resident blocks spin on. Skipped when a separate head-sync kernel runs it.
    if (!unfused_sync) lsa_grid_head_gate(dcomms, head_sync_flag);

    if (is_pad) {
        dispatch_pull_pad_fill<HiddenInt4, kFwd, kPullDispatchThreadsPerSlot, kPullDispatchPadWarps>(
            dst_int4, recv_x_scale_em, recv_topk_weights_em, expert_token_offsets,
            per_expert_counts_active, experts_per_rank, scale_row_bytes, scale_row_int4, pad_idx, lane);
    } else {
        const bool copy_weights = (recv_topk_weights_em != nullptr && peer_weight_ptrs != nullptr);
        // Transposed read schedule: each wave's warps read recv slots strided by num_waves,
        // not a contiguous window. The scan clusters recv slots by source rank, so striding
        // spreads each wave's reads across all source ranks instead of one.
        const int kNWarps = PullWarps * static_cast<int>(gridDim.x);
        const int gwarp = static_cast<int>(blockIdx.x) * PullWarps + warp_id;
        const int num_waves = (num_recv + kNWarps - 1) / kNWarps;
        for (int wave = 0; wave < num_waves; ++wave) {
            const int token = gwarp * num_waves + wave;
            if (token >= num_recv) continue;

            // Decode the source (rank, token) that routes to this recv slot.
            const int32_t g = recv_slot_to_src[token];
            const decoded_src_t src_id = decode_src(g, tokens_per_rank, lsa_team_size);
            const int src_rank = src_id.rank;
            const int src_token = src_id.token;
            const int4* src = reinterpret_cast<const int4*>(peer_input_ptrs[src_rank]) +
                              static_cast<size_t>(src_token) * HiddenInt4;
            const int32_t* slot_row = flat2em_slot_map + static_cast<size_t>(token) * top_k;

            // Start the token row read, then resolve the destination slot(s) while it
            // is in flight so the control-plane reads hide under the token transfer.
            p.issue(src);

            // QUANT_FWD: ingest the source scale row under the in-flight token TMA, then
            // scatter it to each EM slot alongside the token row. kScaleTma rides the async
            // proxy through a second smem stage; otherwise the row is held in registers.
            // TODO: the register path holds the whole scale row; for very large scale rows, chunk
            // it through a register tile to bound register pressure. Deferred.
            int4 sreg[kScaleTma ? 1 : ScaleInt4PerLane];
            if constexpr (kFwd) {
                const uint8_t* ssrc = static_cast<const uint8_t*>(peer_scale_ptrs[src_rank]) +
                                      static_cast<size_t>(src_token) * scale_row_bytes;
                if constexpr (kScaleTma) {
                    sp.issue(ssrc);
                } else {
#pragma unroll
                    for (int i = 0; i < ScaleInt4PerLane; ++i) {
                        const int idx = i * 32 + lane;
                        if (idx < scale_row_int4) sreg[i] = __ldg(reinterpret_cast<const int4*>(ssrc) + idx);
                    }
                }
            }
            auto scatter_scale = [&](int32_t slot) {
                uint8_t* sd = recv_x_scale_em + static_cast<size_t>(slot) * scale_row_bytes;
                if constexpr (kScaleTma) {
                    sp.store(sd);
                } else {
#pragma unroll
                    for (int i = 0; i < ScaleInt4PerLane; ++i) {
                        const int idx = i * 32 + lane;
                        if (idx < scale_row_int4) nccl_ep::st_cg_global(reinterpret_cast<int4*>(sd) + idx, sreg[i]);
                    }
                }
            };

            // Lane-parallel pass over the top_k slots: each lane reads its flat2em entry,
            // relocates its weight, and compacts the kept slots into s_active via a ballot
            // prefix. Chunked by 32 to support top_k > warp width.
            const float* src_w = copy_weights
                ? peer_weight_ptrs[src_rank] + static_cast<size_t>(src_token) * top_k
                : nullptr;
            const int32_t* srcpos_row = srcpos_map ? srcpos_map + static_cast<size_t>(token) * top_k : nullptr;
            int c = 0;
            for (int base = 0; base < top_k; base += 32) {
                const int k = base + lane;
                const int32_t es = (k < top_k) ? __ldg(slot_row + k) : -1;
                const bool keep = (es >= 0);
                if (keep) {
                    // Per-slot capacity backstop (mirror of local_permute_dup).
                    EP_DEVICE_ASSERT(es < caller_num_recv_tokens);
                    // Read the weight at the source's top-k position (order-preserving),
                    // not the packed flat2em index. srcpos_row aligns 1:1 with slot_row.
                    if (copy_weights) {
                        const int wpos = srcpos_row ? __ldg(srcpos_row + k) : k;
                        recv_topk_weights_em[es] = __ldg(src_w + wpos);
                    }
                }
                const unsigned mask = __ballot_sync(0xffffffffu, keep);
                const int pos = c + __popc(mask & ((1u << lane) - 1));
                if (keep && pos < kPullDispatchMaxActive) s_active[warp_id][pos] = es;
                c += __popc(mask);
            }
            if (lane == 0) s_count[warp_id] = c;
            __syncwarp();

            p.wait(); // token row resident
            if constexpr (kScaleTma) sp.wait();
            const int cnt = s_count[warp_id];
            for (int a = 0; a < cnt; ++a) {
                const int32_t slot = s_active[warp_id][a];
                p.store(dst_int4 + static_cast<size_t>(slot) * HiddenInt4);
                if constexpr (kFwd) scatter_scale(slot);
            }
            bulk_wait_reads(lane);
        }
        bulk_drain(lane);
    }

    __syncthreads(); // policy async stores must complete before the block exits.

    // Tail sync: all ranks finished reading peer source before it can be reused.
    // Skipped when a separate tail-sync kernel runs it.
    if (!unfused_sync) lsa_grid_tail_barrier(dcomms, grid_barrier_counter, head_sync_flag);
}

// Local EM reduce kernel (inverse of local_permute_dup). Sums the top_k EM
// rows that share a FLAT recv slot and writes the result back into FLAT staging:
// the caller's dtype for NONE, packed [FP8 H | E8M0 H/32] for MXFP8.
struct local_permute_reduce_param_t {
    void* flat_staging;
    const void* recv_x_em;
    const int32_t* flat2em_slot_map;
    const int32_t* num_recv_tokens_dev;
    // Optional fused EM to FLAT weight gather. Both null on FWD (token only).
    const float* em_weights_in;
    float* flat_weights_out;
    int top_k;
    // Row width is a template parameter on the device side (HiddenInt4); the host keeps
    // it here because it derives that parameter and the JIT cache key from it.
    int row_bytes;
    int caller_num_recv_tokens;   // caller EM buffer row capacity (slot backstop)
};

// Direct-load reduce: each slot's row is reduced by a 128-thread sub-warp; with
// kSlotsPerBlock=8 a block computes 8 slots in parallel. The 128 threads walk the row in
// strided int4 chunks, accumulating every contributor in FP32 via direct cached global
// loads, then write the row out.
//
// HiddenInt4 counts int4s in the EM *input* row, which both recipes read as kTokenDtype.
// It is a template parameter so the per-thread strided loop gets a compile-time bound.
// The accumulation is therefore recipe-independent; only the epilogue differs, because
// NONE writes the input width back while MXFP8 writes a narrower packed row.
template <
    int MaxTopK,
    int HiddenInt4,
    ncclDataType_t kTokenDtype,
    ncclEpCombQuant_t kCombineRecipe>
__device__ __forceinline__ void local_permute_reduce(
    uint8_t* __restrict__ flat_staging,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const float* __restrict__ em_weights_in,
    float* __restrict__ flat_weights_out,
    int top_k,
    int /*row_bytes*/,
    int caller_num_recv_tokens) {
    static_assert(kCombineRecipe != NCCL_EP_COMB_QUANT_MXFP8 || kTokenDtype == ncclBfloat16,
                  "MXFP8 local reduce must output BF16");
    constexpr int kInRowBytes = HiddenInt4 * 16;

    constexpr int kThreadsPerSlot = 128;
    constexpr int kSlotsPerBlock = kLocalPermuteReduceSlotsPerBlock;
    // int4s this thread covers across the row (ceil for partial tails), and how many it
    // keeps in flight per iteration. The cap of 4 bounds the accumulator set at
    // float2 acc[4][kPairs] (~32 registers), which is what keeps this kernel off the
    // spill path at the 64-register budget of blocks_per_sm==1.
    constexpr int kElemsPerThread = (HiddenInt4 + kThreadsPerSlot - 1) / kThreadsPerSlot;
    constexpr int kHiddenVec = (kElemsPerThread < 4) ? kElemsPerThread : 4;
    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();

    // MXFP8 emits the packed [FP8 H | E8M0 H/32] row; the E8M0 tail starts at out_row + H.
    using Mxfp8Traits = nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>;
    constexpr int kElemsPerInt4 = kPairs * 2;
    constexpr int kHidden = HiddenInt4 * kElemsPerInt4;
    constexpr int kPackedRowBytes = Mxfp8Traits::template packed_bytes<ncclBfloat16>(kHidden);
    // Preconditions the epilogue hardcodes, stated through the traits so a format change
    // fails here rather than silently miscomputing.
    static_assert(kCombineRecipe != NCCL_EP_COMB_QUANT_MXFP8 || Mxfp8Traits::kWireBytesPerElem == 1,
                  "MXFP8 epilogue assumes one wire byte per element");
    static_assert(kCombineRecipe != NCCL_EP_COMB_QUANT_MXFP8 ||
                      kElemsPerInt4 * Mxfp8Traits::kWireBytesPerElem == (int)sizeof(uint2),
                  "MXFP8 epilogue narrows one input int4 into a uint2");
    static_assert(kCombineRecipe != NCCL_EP_COMB_QUANT_MXFP8 ||
                      Mxfp8Traits::kScaleBlock == 4 * kElemsPerInt4,
                  "MXFP8 amax reduction assumes a scale block spans 4 int4 lanes");
    static_assert(kCombineRecipe != NCCL_EP_COMB_QUANT_MXFP8 || kPackedRowBytes % 16 == 0,
                  "packed row must be 16B-aligned");
    constexpr int kOutRowBytes =
        (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) ? kPackedRowBytes : kInRowBytes;

    // Per-slot packed em_slot ids in smem: only valid contributors (the rest
    // of top_k are -1 from non-local experts). Lets the inner loop iterate
    // n_valid instead of top_k, which is the dominant win at top_k > EPR.
    __shared__ int32_t smem_flat2em_slot_map[kSlotsPerBlock][MaxTopK];
    __shared__ int s_nvalid[kSlotsPerBlock];

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
    const int slot_thread = tid % kThreadsPerSlot;

    const int num_recv = *num_recv_tokens_dev;
    const int slot_stride = kSlotsPerBlock * static_cast<int>(gridDim.x);

    for (int s_base = static_cast<int>(blockIdx.x) * kSlotsPerBlock; s_base < num_recv; s_base += slot_stride) {
        const int slot = s_base + slot_in_block;
        const bool slot_valid = (slot < num_recv);

        // Cooperative pack: warp 0 of each slot builds the packed em_slot list
        // from the stored flat2em row. top_k rows wider than a warp are packed in
        // ceil(MaxTopK/32) lane-chunks with a running compaction offset; for
        // MaxTopK <= 32 the loop collapses to a single chunk.
        static_assert(MaxTopK <= 64, "cooperative pack assumes MaxTopK <= 64");
        if (slot_valid && slot_thread < 32) {
            const int32_t* slot_row_global = flat2em_slot_map + static_cast<size_t>(slot) * top_k;
            constexpr int kPackChunks = nccl_ep::ceil_div(MaxTopK, 32);
            int packed = 0;
#pragma unroll
            for (int c = 0; c < kPackChunks; ++c) {
                const int pos = c * 32 + slot_thread;
                const int32_t s = (pos < top_k) ? __ldg(slot_row_global + pos) : -1;
                // Per-slot capacity backstop (mirror of local_permute_dup): a map entry
                // past the caller EM buffer means the scan's drop guard failed.
                EP_DEVICE_ASSERT(s < caller_num_recv_tokens);
                if (em_weights_in != nullptr && pos < top_k) {
                    flat_weights_out[static_cast<size_t>(slot) * top_k + pos] = (s >= 0) ? em_weights_in[s] : 0.0f;
                }
                const unsigned valid = __ballot_sync(0xFFFFFFFFu, s >= 0);
                const int my_pos = packed + __popc(valid & ((1u << slot_thread) - 1));
                if (s >= 0) smem_flat2em_slot_map[slot_in_block][my_pos] = s;
                packed += __popc(valid);
            }
            if (slot_thread == 0) s_nvalid[slot_in_block] = packed;
        }
        __syncthreads();

        if (slot_valid) {
            const int n = s_nvalid[slot_in_block];
            uint8_t* out_row = flat_staging + static_cast<size_t>(slot) * kOutRowBytes;

            for (int nn_base = 0; nn_base < kElemsPerThread; nn_base += kHiddenVec) {
                int js[kHiddenVec];
                bool valid_u[kHiddenVec];
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    const int nn = nn_base + u;
                    js[u] = slot_thread + nn * kThreadsPerSlot;
                    valid_u[u] = (nn < kElemsPerThread) && (js[u] < HiddenInt4);
                }

                float2 acc[kHiddenVec][kPairs] = {};

                for (int k = 0; k < n; k++) {
                    const int32_t em_slot = smem_flat2em_slot_map[slot_in_block][k];
                    const int4* src = reinterpret_cast<const int4*>(
                        recv_x_em + static_cast<size_t>(em_slot) * kInRowBytes);
                    int4 buf[kHiddenVec];
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++)
                        if (valid_u[u]) buf[u] = src[js[u]];
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!valid_u[u]) continue;
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&buf[u], p);
                            acc[u][p].x += f.x;
                            acc[u][p].y += f.y;
                        }
                    }
                }

                if constexpr (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) {
                    // One MXFP8 block spans 32 elements = 4 consecutive BF16 int4s, and
                    // consecutive j land on consecutive lanes, so amax reduces over the lane
                    // pair distances 1 and 2. Scale index is j>>2; the FP8 payload for one
                    // int4 is 8 B at byte offset j*8.
                    //
                    // Quantize one u at a time so the e8m0/out temporaries do not multiply by
                    // kHiddenVec. Every lane runs the shuffles to keep the warp converged.
                    uint8_t* dst_scale = out_row + kHidden;
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        float amax = 0.0f;
                        if (valid_u[u]) {
#pragma unroll
                            for (int p = 0; p < kPairs; p++) {
                                amax = fmaxf(
                                    amax, fmaxf(fabsf(acc[u][p].x), fabsf(acc[u][p].y)));
                            }
                        }
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));

                        if (valid_u[u]) {
                            const int j = js[u];
                            const uint8_t e8m0 = nccl_ep::mxfp8::float_to_e8m0(amax);
                            if ((j & 3) == 0) dst_scale[j >> 2] = e8m0;
                            const float scale_inv = nccl_ep::mxfp8::e8m0_to_scale_inv(e8m0);
                            uint2 out;
                            uint16_t* out_pairs = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
                            for (int p = 0; p < kPairs; p++)
                                out_pairs[p] =
                                    nccl_ep::mxfp8::pack_e4m3x2_scaled(acc[u][p], scale_inv);
                            *reinterpret_cast<uint2*>(out_row + static_cast<size_t>(j) * 8) = out;
                        }
                    }
                } else {
                    int4* dst_int4 = reinterpret_cast<int4*>(out_row);
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!valid_u[u]) continue;
                        int4 out;
#pragma unroll
                        for (int p = 0; p < kPairs; p++)
                            nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                        nccl_ep::st_cg_global(&dst_int4[js[u]], out);
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Push-combine staging row stride. A row stride that is a multiple of 768 bytes camps on the
// same HBM partitions across concurrent comm-SM writes, so pad such strides by 256 bytes.
// combine_push (write), combine_reduce (read), and the staging allocation must all agree.
// reserve_prob (backward) co-locates the prob row in the token row's trailing pad (one remote
// write per slot): reserve a 256B pad; if that lands the stride on the 768-byte camping period,
// bump to 512B to stay off it.
constexpr int kCombStagePadMaxBytes = 512;
__host__ __device__ constexpr int comb_stage_stride_bytes(int row_bytes, bool reserve_prob = false) {
    if (reserve_prob) {
        const int s = row_bytes + 256;
        return (s % 768 == 0) ? s + 256 : s;  // +512 keeps the co-located stride anti-camping
    }
    return row_bytes + ((row_bytes % 768 == 0) ? 256 : 0);
}

// ============================================================================
// Push EM combine (HT + EM + MNNVL). Mirror of pull EM dispatch: each
// expert rank locally gathers+reduces its K em copies of a recv token and PUSHES
// the reduced row into the destination attn-rank's staging over NVLink; a final
// combine_reduce sums the team_size partials per attn token. Head/tail intra-LSA syncs
// are fused into the push kernel. Forward reduces tokens; backward (BACKWARD_COMBINE)
// also scatters input weights into peer prob staging. NONE recipe, single LSA team only.
// ============================================================================

struct combine_push_param_t {
    // Output: per-attn-rank staging bases (peer-writable); reduced row lands at [t*team_size + R].
    void* peer_staging_ptrs[kPullMaxLsaRanks];   // [lsa_team_size]
    // Inputs
    const void* recv_x_em;            // local EM expert-output token buffer
    const int32_t* flat2em_slot_map;  // [num_recv, top_k]
    const int32_t* recv_slot_to_src;  // [num_recv] recv slot -> global src token id
    const int32_t* num_recv_tokens_dev;
    // Backward-only inputs: weights + their source top-k positions (co-located into the staging pad).
    const float* topk_weights_em;     // 1D EM input weights, indexed by em_slot
    const int32_t* srcpos_map;        // [num_recv, top_k] source topk position per copy
    // Intra-LSA sync (head gate + tail barrier)
    ncclDevComm_t* dcomms;
    uint32_t* head_sync_flag;         // grid head gate (idle dispatch_grid_barrier_counter)
    uint32_t* grid_barrier_counter;   // tail elect-last-block (combine_grid_barrier_counter)
    // Scalar config
    int top_k;
    int row_bytes;
    int caller_num_recv_tokens;       // caller EM buffer row capacity (slot backstop)
    int my_lsa_rank;                  // R: this rank's slot in each peer's staging
    int tokens_per_rank;              // decode t = g % tokens_per_rank
    int lsa_team_size;                // decode dst_rank; staging row = t*team_size + R
    // When set, head/tail sync runs as separate kernels; this kernel skips the inline sync.
    bool unfused_sync;
};

// One recv slot per warp (kSlotsPerBlock in parallel), identical
// gather-reduce to local_permute_reduce; the reduced row is assembled in a per-warp smem buffer
// and pushed to the remote destination staging slot [t*team_size + R] with one whole-row TMA store.
template <int MaxTopK, int HiddenInt4, ncclDataType_t kTokenDtype, bool BACKWARD_COMBINE = false>
__device__ __forceinline__ void combine_push(
    void* const* __restrict__ peer_staging_ptrs,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ recv_slot_to_src,
    const int32_t* __restrict__ num_recv_tokens_dev,
    ncclDevComm_t* __restrict__ dcomms,
    uint32_t* __restrict__ head_sync_flag,
    uint32_t* __restrict__ grid_barrier_counter,
    int top_k,
    int /*row_bytes*/,
    int caller_num_recv_tokens,
    int my_lsa_rank,
    int tokens_per_rank,
    int lsa_team_size,
    // Backward-only: sparse-direct prob scatter into the token staging pad (null/unused on forward).
    const float* __restrict__ topk_weights_em = nullptr,
    const int32_t* __restrict__ srcpos_map = nullptr,
    bool unfused_sync = false,
    uint8_t* __restrict__ smem_bytes = nullptr) {
    constexpr int kRowBytes = HiddenInt4 * 16;
    // One warp per recv slot: narrow reduction granularity keeps many rows in flight per
    // block, saturating the NVLink write path. Each token is reduced into a per-warp smem row,
    // then pushed with one whole-row TMA (cp.async.bulk) store. The hidden is walked in
    // kHiddenVec-int4 tiles, so registers stay bounded regardless of hidden.
    constexpr int kThreadsPerSlot = 32;
    const int kSlotsPerBlock = static_cast<int>(blockDim.x) / kThreadsPerSlot;  // sized at launch
    constexpr int kElemsPerThread = (HiddenInt4 + kThreadsPerSlot - 1) / kThreadsPerSlot;

    // ---- HEAD SYNC: gates peer pushes until all ranks reach combine, so a push can't
    // land while the previous iteration's combine_reduce is still reading staging. ----
    // Skipped when a separate head-sync kernel runs it.
    if (!unfused_sync) lsa_grid_head_gate(dcomms, head_sync_flag);

    // Metadata smem is sized to the max block; only the first kSlotsPerBlock rows are used.
    __shared__ int32_t smem_flat2em_slot_map[kCombinePushMaxSlots][MaxTopK];
    __shared__ int s_nvalid[kCombinePushMaxSlots];
    __shared__ int4* s_dst[kCombinePushMaxSlots];
    // Per-warp mbarrier for the opt-in TMA-load source relay (single-copy fast path).
    __shared__ uint64_t s_mbar[kCombinePushMaxSlots];
    // Dynamic smem: one row buffer per warp for the TMA push (each HiddenInt4 int4).
    int4* const s_rows = reinterpret_cast<int4*>(smem_bytes);
    // Backward co-locates the prob in the token staging row's trailing pad (offset kRowBytes),
    // so no separate prob destination is needed.

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
    const int wlane = tid % kThreadsPerSlot;
    // This warp's row staging buffer: [slot_in_block][HiddenInt4].
    int4* const my_row = s_rows + static_cast<size_t>(slot_in_block) * HiddenInt4;

    const int num_recv = *num_recv_tokens_dev;
    // Contiguous per-warp range with SM-spread global ordering: consecutive global warps land on
    // different SMs (even last-wave spread) and each warp streams a contiguous slot range (writes
    // become sequential per peer once slots are dst-grouped). Output-independent.
    const int kNSub = kSlotsPerBlock * static_cast<int>(gridDim.x);
    const int global_warp_idx = slot_in_block * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int num_per_warp = (num_recv + kNSub - 1) / kNSub;
    // Cross-rank rotation: offset each rank's warp->range assignment by its own rank so the
    // concurrent senders begin on different dst regions, balancing the receiver-side NVLink
    // links when slots are dst-grouped. A permutation of the same slots -> output-independent.
    const int rot_warp = (global_warp_idx + my_lsa_rank * (kNSub / lsa_team_size)) % kNSub;
    const int slot_start = rot_warp * num_per_warp;

    // TMA-load relay: init this warp's mbarrier once (1 arrival: the load's complete_tx).
    uint32_t mbar_phase = 0;
    if (wlane == 0) nccl_ep::mbarrier_init(&s_mbar[slot_in_block], 1);
    nccl_ep::fence_barrier_init();
    __syncthreads();

    for (int wave = 0; wave < num_per_warp; ++wave) {
        const int slot = slot_start + wave;
        const bool slot_valid = (slot < num_recv);

        static_assert(MaxTopK <= 32, "cooperative pack assumes MaxTopK <= 32");
        // Backward: per-copy prob metadata (srcpos + input weight), prefetched in the metadata
        // phase but stored only after the token push loop so its load latency hides behind the
        // NVLink token writes. Carried at wave scope across both blocks.
        int32_t prob_p = -1;
        float prob_w = 0.0f;
        if (slot_valid && wlane < 32) {
            const int32_t* slot_row_global = flat2em_slot_map + static_cast<size_t>(slot) * top_k;
            const int32_t s = (wlane < top_k) ? __ldg(slot_row_global + wlane) : -1;
            EP_DEVICE_ASSERT(s < caller_num_recv_tokens);
            if constexpr (BACKWARD_COMBINE) {
                if (wlane < top_k && s >= 0) {
                    prob_p = __ldg(srcpos_map + static_cast<size_t>(slot) * top_k + wlane);
                    prob_w = __ldg(topk_weights_em + s);
                }
            }
            const unsigned valid = __ballot_sync(0xFFFFFFFFu, s >= 0);
            const int my_pos = __popc(valid & ((1u << wlane) - 1));
            if (s >= 0) smem_flat2em_slot_map[slot_in_block][my_pos] = s;
            if (wlane == 0) {
                s_nvalid[slot_in_block] = __popc(valid);
                // Decode the destination attn (rank, token) this recv slot returns to.
                const int32_t g = recv_slot_to_src[slot];
                const decoded_src_t dst_id = decode_src(g, tokens_per_rank, lsa_team_size);
                const int dst_rank = dst_id.rank;
                const int t = dst_id.token;
                const int64_t dst_row = static_cast<int64_t>(t) * lsa_team_size + my_lsa_rank;
                s_dst[slot_in_block] = reinterpret_cast<int4*>(
                    static_cast<uint8_t*>(peer_staging_ptrs[dst_rank]) +
                    static_cast<size_t>(dst_row) * comb_stage_stride_bytes(kRowBytes, BACKWARD_COMBINE));
            }
        }
        __syncwarp();

        if (slot_valid) {
            const int n = s_nvalid[slot_in_block];
            int4* dst_int4 = s_dst[slot_in_block];

            constexpr int kHiddenVec = (kElemsPerThread < 4) ? kElemsPerThread : 4;
            constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
            constexpr int kNTiles = (kElemsPerThread + kHiddenVec - 1) / kHiddenVec;

            // Wait for this warp's prior push to drain before reusing its single row buffer.
            if (wlane == 0) nccl_ep::tma_store_wait<0>();
            __syncwarp();
            auto src_j = [&](int tile, int u) { return wlane + (tile * kHiddenVec + u) * kThreadsPerSlot; };
            // Bound the row index too: when HiddenInt4 < kThreadsPerSlot the extra lanes own no element.
            auto tile_valid = [&](int tile, int u) {
                return (tile * kHiddenVec + u) < kElemsPerThread && src_j(tile, u) < HiddenInt4;
            };

            // Single-copy relay: TMA-load the one source row straight into the smem buffer.
            bool filled = false;
            if (n == 1) {
                const int32_t em0 = smem_flat2em_slot_map[slot_in_block][0];
                const int4* src0 =
                    reinterpret_cast<const int4*>(recv_x_em + static_cast<size_t>(em0) * kRowBytes);
                if (wlane == 0) {
                    nccl_ep::mbarrier_arrive_and_expect_tx(&s_mbar[slot_in_block], kRowBytes);
                    nccl_ep::tma_load_1d(my_row, src0, &s_mbar[slot_in_block], kRowBytes);
                }
                nccl_ep::mbarrier_wait(&s_mbar[slot_in_block], mbar_phase);
                filled = true;
            }

            if (!filled) {
            // Software-pipelined push: prefetch the next tile's first em copy so its read
            // overlaps the posted writes. Registers stay bounded regardless of top_k.
            int4 pf_a[kHiddenVec], pf_b[kHiddenVec];
            int4* pf_cur = pf_a;
            int4* pf_nxt = pf_b;
            auto load_copy0 = [&](int tile, int4* pf) {
                const int32_t em0 = smem_flat2em_slot_map[slot_in_block][0];
                const int4* src =
                    reinterpret_cast<const int4*>(recv_x_em + static_cast<size_t>(em0) * kRowBytes);
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++)
                    if (tile_valid(tile, u)) pf[u] = src[src_j(tile, u)];
            };
            if (n > 0) load_copy0(0, pf_cur);
#pragma unroll
            for (int tile = 0; tile < kNTiles; tile++) {
                float2 acc[kHiddenVec][4];
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++)
#pragma unroll
                    for (int p = 0; p < 4; p++) { acc[u][p].x = 0.0f; acc[u][p].y = 0.0f; }
                if (n > 1) {  // accumulate copy 0 only when more copies follow
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!tile_valid(tile, u)) continue;
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&pf_cur[u], p);
                            acc[u][p].x += f.x;
                            acc[u][p].y += f.y;
                        }
                    }
                }
                if (tile + 1 < kNTiles && n > 0) load_copy0(tile + 1, pf_nxt);  // prefetch
                for (int k = 1; k < n; k++) {  // remaining copies loaded inline
                    const int32_t em_slot = smem_flat2em_slot_map[slot_in_block][k];
                    const int4* src =
                        reinterpret_cast<const int4*>(recv_x_em + static_cast<size_t>(em_slot) * kRowBytes);
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!tile_valid(tile, u)) continue;
                        int4 b = src[src_j(tile, u)];
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&b, p);
                            acc[u][p].x += f.x;
                            acc[u][p].y += f.y;
                        }
                    }
                }
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    if (!tile_valid(tile, u)) continue;
                    int4 out;
                    if (n == 1) {  // single copy: relay verbatim, no repack
                        out = pf_cur[u];
                    } else {
#pragma unroll
                        for (int p = 0; p < kPairs; p++)
                            nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                    }
                    my_row[src_j(tile, u)] = out;  // assemble reduced row in smem
                }
                int4* tmp = pf_cur; pf_cur = pf_nxt; pf_nxt = tmp;
            }
            }  // end if (!filled): scalar reduce path

            // Push the row. The proxy fence makes the scalar smem writes visible to the async
            // (TMA) proxy read.
            nccl_ep::tma_store_fence();
            __syncwarp();
            if (wlane == 0) nccl_ep::tma_store_1d(my_row, dst_int4, kRowBytes);

            // Backward: store the prefetched weight to its srcpos in this (t,R) token row's trailing
            // pad (co-located, one remote write per slot), after the token push has hidden the load.
            if constexpr (BACKWARD_COMBINE) {
                if (prob_p >= 0) {
                    float* prob_dst = reinterpret_cast<float*>(
                        reinterpret_cast<uint8_t*>(dst_int4) + kRowBytes);
                    prob_dst[prob_p] = prob_w;
                }
            }
        }
        __syncwarp();
    }
    // Drain all outstanding bulk pushes (global writes complete) before the tail sync makes them
    // visible to combine_reduce.
    if (wlane == 0) nccl_ep::tma_store_wait_complete<0>();
    __syncwarp();

    // ---- TAIL SYNC: all peer pushes globally visible before combine_reduce reads ----
    // Skipped when a separate tail-sync kernel runs it.
    if (!unfused_sync) lsa_grid_tail_barrier(dcomms, grid_barrier_counter, head_sync_flag);
}

struct combine_reduce_param_t {
    // Outputs
    void* attn_output;                // [num_combined_tokens, hidden]
    float* combined_topk_weights;     // backward-only: [num_combined_tokens, top_k] gathered weights
    // Inputs. Which of the team_size partials exist per token is derived locally from this rank's
    // own routing: token t has a partial from rank R iff one of its top-k experts lives on R.
    const void* staging;              // local staging [tokens_per_rank, team_size, hidden]
    const uint16_t* topk_idx;         // [num_combined_tokens, num_topk] global expert ids (uint16)
    // Scalar config
    int num_topk;                     // experts routed per token
    int experts_per_rank;             // expert -> rank block size (R = expert / experts_per_rank)
    int experts_per_lsa_team;         // in-team expert range; ids >= this (incl. sentinel) skipped
    int num_combined_tokens;          // attn tokens this rank reduces
    int lsa_team_size;                // number of per-token partials to sum
    int row_bytes;
    int top_k;                        // backward-only: combined_topk_weights width
};

// One attn token per 128-thread sub-warp: sum the pushed team_size staging partials
// [t*team_size + 0..team_size) into attn_output[t]. Which partials exist is derived from
// this rank's own routing (topk_idx): token t has a partial from rank R iff one of its
// top-k experts lives on R -- unwritten slots are skipped (no staging zero-init needed).
template <int HiddenInt4, ncclDataType_t kTokenDtype, bool BACKWARD_COMBINE = false>
__device__ __forceinline__ void combine_reduce(
    uint8_t* __restrict__ attn_output,
    const uint8_t* __restrict__ staging,
    const uint16_t* __restrict__ topk_idx,
    int num_topk,
    int experts_per_rank,
    int experts_per_lsa_team,
    int num_combined_tokens,
    int lsa_team_size,
    int /*row_bytes*/,
    // Backward-only: gather each srcpos weight from the token staging pad into combined_topk_weights.
    float* __restrict__ combined_topk_weights = nullptr,
    int top_k = 0) {
    constexpr int kRowBytes = HiddenInt4 * 16;
    constexpr int kThreadsPerSlot = 128;
    constexpr int kSlotsPerBlock = kLocalPermuteReduceSlotsPerBlock;
    constexpr int kElemsPerThread = (HiddenInt4 + kThreadsPerSlot - 1) / kThreadsPerSlot;

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
    const int slot_thread = tid % kThreadsPerSlot; // thread index within its slot (0..kThreadsPerSlot), not a warp lane
    const int slot_stride = kSlotsPerBlock * static_cast<int>(gridDim.x);

    // One expert per warp lane, so num_topk must fit the warp width.
    EP_DEVICE_ASSERT(num_topk <= MAX_NUM_TOPK);

    for (int t_base = static_cast<int>(blockIdx.x) * kSlotsPerBlock; t_base < num_combined_tokens;
         t_base += slot_stride) {
        const int t = t_base + slot_in_block;
        // t grows monotonically with t_base and there is no block-wide barrier below
        // (only full-warp intrinsics, and a warp shares one slot), so bail out.
        if (t >= num_combined_tokens) break;

        // Distinct source ranks that pushed a partial for token t: rank R = expert / experts_per_rank
        // for each of the token's in-team top-k experts. Deduped via a warp match (combine_push already
        // merged a rank's multiple experts into one pushed row). Single LSA team, so ids are in-team
        // (base 0): the e_id < experts_per_lsa_team bound and the /experts_per_rank rank decode assume
        // one team; multi-team would need the global expert count and a team-aware decode. Out-of-range
        // / sentinel ids drop. Each warp lane holds one expert (num_topk <= MAX_NUM_TOPK).
        const int wlane = static_cast<int>(threadIdx.x) & 31;
        const int e_id = (wlane < num_topk)
                             ? static_cast<int>(topk_idx[static_cast<size_t>(t) * num_topk + wlane])
                             : -1;
        const bool rank_valid = (e_id >= 0) && (e_id < experts_per_lsa_team);
        const int src_rank_of_lane = rank_valid ? (e_id / experts_per_rank) : -1;
        const unsigned same_rank = __match_any_sync(0xFFFFFFFFu, src_rank_of_lane);
        const bool rank_leader = rank_valid && ((__ffs(static_cast<int>(same_rank)) - 1) == wlane);
        const unsigned present_ranks = __ballot_sync(0xFFFFFFFFu, rank_leader);

        // Backward: srcpos slot `slot_thread` was pushed by the rank hosting that expert
        // (src_rank_of_lane), co-located in that rank's token staging row pad (offset kRowBytes).
        // Out-of-team / unrouted slots yield zero.
        if constexpr (BACKWARD_COMBINE) {
            if (slot_thread < top_k) {
                float w = 0.0f;
                if (src_rank_of_lane >= 0) {
                    const float* prob_row = reinterpret_cast<const float*>(
                        staging + (static_cast<size_t>(t) * lsa_team_size + src_rank_of_lane) *
                                      comb_stage_stride_bytes(kRowBytes, /*reserve_prob=*/true) + kRowBytes);
                    w = prob_row[slot_thread];
                }
                combined_topk_weights[static_cast<size_t>(t) * top_k + slot_thread] = w;
            }
        }

        int4* dst_int4 = reinterpret_cast<int4*>(attn_output + static_cast<size_t>(t) * kRowBytes);
        const size_t base_row = static_cast<size_t>(t) * lsa_team_size;

        constexpr int kHiddenVec = (kElemsPerThread < 4) ? kElemsPerThread : 4;
        for (int nn_base = 0; nn_base < kElemsPerThread; nn_base += kHiddenVec) {
            int js[kHiddenVec];
            bool valid_u[kHiddenVec];
#pragma unroll
            for (int u = 0; u < kHiddenVec; u++) {
                const int nn = nn_base + u;
                js[u] = slot_thread + nn * kThreadsPerSlot;
                valid_u[u] = (nn < kElemsPerThread) && (js[u] < HiddenInt4);
            }

            float2 acc[kHiddenVec][4];
#pragma unroll
            for (int u = 0; u < kHiddenVec; u++) {
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    acc[u][p].x = 0.0f;
                    acc[u][p].y = 0.0f;
                }
            }

            for (unsigned bl = present_ranks; bl != 0u; bl &= (bl - 1u)) {
                const int r = __shfl_sync(0xFFFFFFFFu, src_rank_of_lane, __ffs(static_cast<int>(bl)) - 1);
                const int4* src =
                    reinterpret_cast<const int4*>(staging + (base_row + r) *
                        comb_stage_stride_bytes(kRowBytes, BACKWARD_COMBINE));
                int4 buf[kHiddenVec];
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    if (valid_u[u]) buf[u] = src[js[u]];
                }
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    if (!valid_u[u]) continue;
                    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                    for (int p = 0; p < kPairs; p++) {
                        float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&buf[u], p);
                        acc[u][p].x += f.x;
                        acc[u][p].y += f.y;
                    }
                }
            }

#pragma unroll
            for (int u = 0; u < kHiddenVec; u++) {
                if (!valid_u[u]) continue;
                int4 out;
                constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                for (int p = 0; p < kPairs; p++) {
                    nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                }
                dst_int4[js[u]] = out;
            }
        }
    }
}

} // namespace ht_ep

#include "scan_kernel.cuh"
#include "count_kernel.cuh"
