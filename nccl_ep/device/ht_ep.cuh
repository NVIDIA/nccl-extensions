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

#include "nccl_ep.h"
#include "common.hpp"
#include "device_primitives.cuh"
#include "ht_ep_configs.cuh"
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda/ptx>
#include "nccl_device.h"
#include "cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x

namespace ht_ep {

template <int NUM_OF_BOOL_TO_REDUCE>
using Reduce_t = typename std::conditional<
    NUM_OF_BOOL_TO_REDUCE % 8 == 0,
    uint64_t,
    typename std::conditional<
        NUM_OF_BOOL_TO_REDUCE % 4 == 0,
        uint32_t,
        typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 2 == 0, uint16_t, uint8_t>::type>::type>::type;

template <int NUM_OF_BYTES_TO_COPY>
using Copy_t = typename std::conditional<
    NUM_OF_BYTES_TO_COPY % 16 == 0,
    uint4,
    typename std::conditional<
        NUM_OF_BYTES_TO_COPY % 8 == 0,
        uint2,
        typename std::conditional<
            NUM_OF_BYTES_TO_COPY % 4 == 0,
            uint32_t,
            typename std::conditional<NUM_OF_BYTES_TO_COPY % 2 == 0, uint16_t, uint8_t>::type>::type>::type>::type;

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
    size_t rdma_send_staging_offset;           // Offset of per-destination staging buffer
    size_t rdma_inter_node_group_packed_offset; // Offset of packed receive buffer (token+prob+sf per entry)
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
    return mr->rdma_inter_node_group_packed_offset +
           (static_cast<size_t>(remote_slot) * mr->max_tokens_per_dest +
            static_cast<size_t>(chunk_first_token)) *
               mr->bytes_per_entry;
}

struct combine_memory_region_info_t {
    size_t rdma_intra_node_red_token_offset; // Offset of intra-node reduced token buffer
    size_t combine_rdma_inter_node_group_token_offset; // Offset of combine rdma token buffer
    size_t rdma_intra_node_red_prob_offset; // Offset of intra-node reduced prob buffer
    size_t combine_rdma_inter_node_group_prob_offset; // Offset of combine rdma prob buffer
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

// Acquire/release lock helpers for shared memory coordination
__device__ __forceinline__ void acquire_lock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
    }
    __threadfence_block();
}

__device__ __forceinline__ void release_lock(int* lock) {
    __threadfence_block();
    atomicExch(lock, 0);
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
    int num_of_tokens_per_chunk;
    int num_of_tokens_per_group;
    int num_of_blocks;
    int num_of_additional_in_flight_s2g;
    bool backward_combine;
    bool device_side_sync;
};

struct model_config_t {
    int hidden_dim;
    int max_num_of_tokens_per_rank;
    int num_of_experts_per_rank;
    int ranks_per_lsa_team;
    int num_of_nodes;
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

// s2d-map double-buffer depth: consume one stage while prefetching the next (chunk, node) row.
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

// Combine kernels pad each node's rdma_to_attn_map row to a multiple of 16 bools;
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

// Popcount of row[start_bit, end_bit). Step 4 uses this to derive em_s2d slot indices atomic-free.
__device__ __forceinline__ int popcount_bit_range(const uint8_t* row, int start_bit, int end_bit) {
    if (end_bit <= start_bit) return 0;
    const int byte_lo = start_bit >> 3;
    const int byte_hi = (end_bit + 7) >> 3;
    const int lo_off = start_bit & 7;
    const int hi_keep = end_bit & 7; // 0 → keep full last byte
    int total = 0;
    for (int b = byte_lo; b < byte_hi; b++) {
        unsigned byte = row[b];
        if (b == byte_lo) byte &= (0xFFu << lo_off) & 0xFFu;
        if (b == byte_hi - 1 && hi_keep != 0) byte &= (1u << hi_keep) - 1u;
        total += __popc(byte);
    }
    return total;
}

// Extract up to 64 contiguous bits from a bit-packed row.
__device__ __forceinline__ uint64_t extract_bits64(const uint8_t* row, int start_bit, int nbits) {
    if (nbits <= 0) return 0;
    const int byte_lo = start_bit >> 3;
    const int byte_hi = (start_bit + nbits + 7) >> 3;
    const int lo_off = start_bit & 7;
    uint64_t out = 0;
    for (int b = byte_lo; b < byte_hi && (b - byte_lo) < 9; b++) {
        out |= static_cast<uint64_t>(row[b]) << ((b - byte_lo) * 8);
    }
    out >>= lo_off;
    if (nbits < 64) out &= (static_cast<uint64_t>(1) << nbits) - 1;
    return out;
}

struct combine_smem_layout_t {
    uint16_t* intra_node_token_G2S_buffer;
    uint16_t* intra_node_token_S2G_buffer;
    uint16_t* inter_node_token_G2S_buffer;
    uint16_t* inter_node_token_S2G_buffer;
    float* intra_node_prob_G2S_buffer;
    float* intra_node_prob_S2G_buffer;
    float* inter_node_prob_G2S_buffer;
    float* inter_node_prob_S2G_buffer;
    uint64_t* intra_node_mbarrier_G2S_buffer;
    uint64_t* inter_node_mbarrier_G2S_buffer;
    uint64_t* intra_node_to_rdma_mbarrier_buffer;
    bool* intra_node_flag_G2S_buffer;
    bool* inter_node_flag_G2S_buffer;

    int token_G2S_stage_stride; // elements (not bytes)
    int token_S2G_stage_stride; // elements (not bytes)
    int prob_G2S_stage_stride; // elements (not bytes)
    int prob_S2G_stage_stride; // intra-node elements (not bytes)
    int prob_S2G_inter_stage_stride; // inter-node elements (not bytes)
    combine_memory_region_info_t* combine_memory_region_info;

    // Streaming overlap: reduction warp -> RDMA warp within a chunk
    uint32_t* rdma_streaming_counter; // [1] cumulative tokens produced for the current chunk

    int s2d_inner_dim; // Inner dimension of unified S2D map (n_ranks_per_node or num_topk)

    // Accessor methods for staged buffers
    __device__ __forceinline__ uint16_t* get_intra_node_token_G2S(int stage) const {
        return intra_node_token_G2S_buffer + stage * token_G2S_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_intra_node_token_S2G(int stage) const {
        return intra_node_token_S2G_buffer + stage * token_S2G_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_inter_node_token_G2S(int stage) const {
        return inter_node_token_G2S_buffer + stage * token_G2S_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_inter_node_token_S2G(int stage) const {
        return inter_node_token_S2G_buffer + stage * token_S2G_stage_stride;
    }
    __device__ __forceinline__ float* get_intra_node_prob_G2S(int stage) const {
        return intra_node_prob_G2S_buffer + stage * prob_G2S_stage_stride;
    }
    __device__ __forceinline__ float* get_intra_node_prob_S2G(int stage) const {
        return intra_node_prob_S2G_buffer + stage * prob_S2G_stage_stride;
    }
    __device__ __forceinline__ float* get_inter_node_prob_G2S(int stage) const {
        return inter_node_prob_G2S_buffer + stage * prob_G2S_stage_stride;
    }
    __device__ __forceinline__ float* get_inter_node_prob_S2G(int stage) const {
        return inter_node_prob_S2G_buffer + stage * prob_S2G_inter_stage_stride;
    }
    // Accessor methods for mbarrier buffers (producer = stage*2, consumer = stage*2+1)
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_G2S_producer(int stage) const {
        return intra_node_mbarrier_G2S_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_G2S_consumer(int stage) const {
        return intra_node_mbarrier_G2S_buffer + stage * 2 + 1;
    }
    __device__ __forceinline__ uint64_t* get_inter_node_mbarrier_G2S_producer(int stage) const {
        return inter_node_mbarrier_G2S_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_inter_node_mbarrier_G2S_consumer(int stage) const {
        return inter_node_mbarrier_G2S_buffer + stage * 2 + 1;
    }
};

struct dispatch_smem_layout_t {
    void* intra_node_token_buffer;
    float* intra_node_prob_buffer;
    uint8_t* intra_node_scaling_factor_buffer;
    int32_t* sparse_to_dense_map_buffer;
    bool* attn_to_rdma_map_buffer;
    uint64_t* intra_node_mbarrier_buffer;
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
            reinterpret_cast<uint8_t*>(intra_node_token_buffer) + stage * token_buffer_stage_stride);
    }
    __device__ __forceinline__ float* get_prob_buffer(int stage) const {
        return reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(intra_node_prob_buffer) + stage * prob_buffer_stage_stride);
    }
    __device__ __forceinline__ void* get_sf_buffer(int stage) const {
        return reinterpret_cast<void*>(intra_node_scaling_factor_buffer + stage * sf_buffer_stage_stride);
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_producer(int stage) const {
        return intra_node_mbarrier_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_consumer(int stage) const {
        return intra_node_mbarrier_buffer + stage * 2 + 1;
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
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_producer(int pipeline_id, int local_stage) const {
        return get_intra_node_mbarrier_producer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_consumer(int pipeline_id, int local_stage) const {
        return get_intra_node_mbarrier_consumer(pipeline_id * stages_per_pipeline + local_stage);
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
    layout.intra_node_token_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
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
        layout.intra_node_prob_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_stages * layout.prob_buffer_stage_stride;
    } else {
        layout.intra_node_prob_buffer = nullptr;
        layout.prob_buffer_stage_stride = 0;
    }

    // Scaling factor buffer (only if quantized, 16B aligned) -- total stages unchanged
    if (config.sf_bytes_per_token > 0) {
        layout.sf_buffer_stage_stride = config.sf_bytes_per_token;
        layout.sf_buffer_stage_stride = (layout.sf_buffer_stage_stride + 15) & ~15;
        layout.intra_node_scaling_factor_buffer = reinterpret_cast<uint8_t*>(smem_base) + offset;
        offset += config.num_of_stages * layout.sf_buffer_stage_stride;
    } else {
        layout.intra_node_scaling_factor_buffer = nullptr;
        layout.sf_buffer_stage_stride = 0;
    }

    // attn_to_rdma_map buffer (16B aligned, only if multinode, shared across pipelines)
    if (model.num_of_nodes > 1) {
        offset = (offset + 15) & ~15;
        layout.attn_to_rdma_map_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
    } else {
        layout.attn_to_rdma_map_buffer = nullptr;
    }

    // Mbarrier buffers (8B aligned) -- total stages unchanged (producer+consumer per stage)
    offset = (offset + 7) & ~7;
    layout.intra_node_mbarrier_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * 2 * sizeof(uint64_t);

    // Per-pipeline s2d_map mbarriers: 2 per pipeline
    layout.sparse_to_dense_map_mbarrier_buffer =
        reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += 2 * num_pipelines * sizeof(uint64_t);

    // Per-pipeline S2G group mbarrier: 1 per pipeline
    layout.S2G_group_mbarrier_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_pipelines * sizeof(uint64_t);

    if (model.num_of_nodes > 1) {
        offset = (offset + 7) & ~7;
        layout.dispatch_memory_region_info =
            reinterpret_cast<dispatch_memory_region_info_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
    } else {
        layout.dispatch_memory_region_info = nullptr;
    }

    // PAD warp TMA slot: one zeroed token row, broadcast to padding slots.
    // Only allocated for expert-major; flat leaves the pointer null.
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        int pad_bytes = model.hidden_dim * kTokenSize;
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
template <ncclEpLayout_t kLayout, int kTokenSize>
static size_t calculate_dispatch_smem_layout_size(const dispatch_config_t& config, const model_config_t& model) {
    static_assert(kTokenSize > 0, "token size must be positive");
    size_t total_size = 0;
    const int num_pipelines = config.num_pipelines;
    // Token buffer (aligned to 128B for TMA) -- total stages unchanged
    int token_buffer_stage_stride = model.hidden_dim * kTokenSize;
    token_buffer_stage_stride = (token_buffer_stage_stride + 127) & ~127;
    total_size += config.num_of_stages * token_buffer_stage_stride;

    // Sparse to dense map buffer: S2D_MAP_RING_STAGES ping-pong stages PER PIPELINE (128B aligned)
    // Inner dim is mode-dependent: flat = lsa_team_size, expert-major = num_topk.
    int s2d_map_stage_stride = config.num_of_tokens_per_chunk * config.s2d_inner_dim * sizeof(int32_t);
    s2d_map_stage_stride = (s2d_map_stage_stride + 127) & ~127;
    total_size += S2D_MAP_RING_STAGES * num_pipelines * s2d_map_stage_stride;

    // Prob buffer (16B aligned per stage) -- total stages unchanged
    if (config.forward_dispatch) {
        int prob_buffer_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
        prob_buffer_stage_stride = (prob_buffer_stage_stride + 15) & ~15;
        total_size += config.num_of_stages * prob_buffer_stage_stride;
    }

    // Scaling factor buffer (16B aligned per stage, only if quantized) -- total stages unchanged
    if (config.sf_bytes_per_token > 0) {
        int sf_buffer_stage_stride = config.sf_bytes_per_token;
        sf_buffer_stage_stride = (sf_buffer_stage_stride + 15) & ~15;
        total_size += config.num_of_stages * sf_buffer_stage_stride;
    }
    // attn_to_rdma_map buffer (aligned to 16B, only if multinode, shared)
    if (model.num_of_nodes > 1) {
        total_size = (total_size + 15) & ~15;
        total_size += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
    }
    // Mbarrier buffers (aligned to 8B) -- total stages unchanged
    total_size = (total_size + 7) & ~7;
    total_size += config.num_of_stages * 2 * sizeof(uint64_t);
    // Per-pipeline s2d_map mbarriers: 2 per pipeline
    total_size = (total_size + 7) & ~7;
    total_size += 2 * num_pipelines * sizeof(uint64_t);
    // Per-pipeline S2G group mbarrier: 1 per pipeline
    total_size = (total_size + 7) & ~7;
    total_size += num_pipelines * sizeof(uint64_t);
    // Dispatch memory region info buffer (aligned to 8B, only if multinode)
    if (model.num_of_nodes > 1) {
        total_size = (total_size + 7) & ~7;
        total_size += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
    }
    // PAD warp TMA slot (expert-major only, 128B aligned)
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        int pad_bytes = model.hidden_dim * kTokenSize;
        pad_bytes = (pad_bytes + 127) & ~127;
        total_size = (total_size + 127) & ~127;
        total_size += pad_bytes;
    }
    // Add padding for alignment
    total_size = (total_size + 127) & ~127;
    return total_size;
}

// Host-side companion to the templated layout calculation above. The JIT
// specializes the device kernel on TOKEN_DATA_TYPE; this wrapper uses that
// same byte width for the launch-time dynamic-SMEM calculation, keeping the
// only host template dispatch next to the template it selects.
inline size_t calculate_dispatch_smem_layout_size(
    ncclEpLayout_t layout,
    unsigned int token_size,
    const dispatch_config_t& config,
    const model_config_t& model) {
    if (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        switch (token_size) {
            case 1: return calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_EXPERT_MAJOR, 1>(config, model);
            case 2: return calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_EXPERT_MAJOR, 2>(config, model);
            case 4: return calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_EXPERT_MAJOR, 4>(config, model);
            default: return 0;
        }
    }
    // Preserve the pre-wrapper host mapping: every non-EM layout uses the
    // flat dispatch SMEM layout. Layout validity is enforced at the API layer.
    switch (token_size) {
        case 1: return calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_FLAT, 1>(config, model);
        case 2: return calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_FLAT, 2>(config, model);
        case 4: return calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_FLAT, 4>(config, model);
        default: return 0;
    }
}

// kTokenSize drives the per-stage token-buffer stride; everything else
// (probabilities, mbarriers, scales) is element-width-invariant.
template <ncclDataType_t kTokenDtype = ncclBfloat16>
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

    // In the single-node case (num_of_nodes == 1), the combine kernel does not use the
    // intra-node staging buffers. Skipping these buffers can cut SMEM roughly in half.
    const bool multinode = (model.num_of_nodes > 1);

    // Per-token wire size: bytes for buffer offsets, uint16_t units for stage strides (the
    // token buffer base is uint16_t*). FP32 doubles both vs BF16/FP16.
    const int token_bytes = model.hidden_dim * nccl_ep::size_u8<kTokenDtype>();
    const int token_stride_u16 = model.hidden_dim * nccl_ep::size_u16<kTokenDtype>();

    // Stage strides in uint16_t units, so FP32 stages advance 2× and don't overlap.
    layout.token_G2S_stage_stride = token_stride_u16;
    layout.token_S2G_stage_stride = token_stride_u16;
    layout.prob_G2S_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team;
    layout.prob_S2G_stage_stride = model.num_of_experts_per_rank * model.ranks_per_lsa_team;
    layout.prob_S2G_inter_stage_stride = layout.prob_S2G_stage_stride * model.num_of_nodes;

    // intra_node_token_* buffers (128B aligned, multi-node only). Stage stride scales
    // with the on-wire element width (2 B for BF16/FP16, 4 B for FP32).
    if (multinode) {
        align_offset(128);
        layout.intra_node_token_G2S_buffer =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * token_bytes;

        align_offset(128);
        layout.intra_node_token_S2G_buffer =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * token_bytes;
    } else {
        layout.intra_node_token_G2S_buffer = nullptr;
        layout.intra_node_token_S2G_buffer = nullptr;
    }

    // inter_node_token_G2S_buffer (128B aligned)
    align_offset(128);
    layout.inter_node_token_G2S_buffer = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * token_bytes;

    // inter_node_token_S2G_buffer (128B aligned)
    align_offset(128);
    layout.inter_node_token_S2G_buffer = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_s2g * token_bytes;

    // Prob buffers (only if backward_combine, 16B aligned)
    if (backward_combine) {
        if (multinode) {
            // intra_node_prob_G2S_buffer
            align_offset(16);
            layout.intra_node_prob_G2S_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_g2s * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);

            // intra_node_prob_S2G_buffer
            align_offset(16);
            layout.intra_node_prob_S2G_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_s2g * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
        } else {
            layout.intra_node_prob_G2S_buffer = nullptr;
            layout.intra_node_prob_S2G_buffer = nullptr;
        }

        // inter_node_prob_G2S_buffer
        align_offset(16);
        layout.inter_node_prob_G2S_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);

        // inter_node_prob_S2G_buffer
        align_offset(16);
        layout.inter_node_prob_S2G_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * model.num_of_experts_per_rank * model.ranks_per_lsa_team * model.num_of_nodes *
                  sizeof(float);
    } else {
        layout.intra_node_prob_G2S_buffer = nullptr;
        layout.intra_node_prob_S2G_buffer = nullptr;
        layout.inter_node_prob_G2S_buffer = nullptr;
        layout.inter_node_prob_S2G_buffer = nullptr;
    }

    // Mbarrier buffers (8B aligned)
    // intra_node_mbarrier_G2S_buffer (multi-node only)
    if (multinode) {
        align_offset(8);
        layout.intra_node_mbarrier_G2S_buffer =
            reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * 2 * sizeof(uint64_t);
    } else {
        layout.intra_node_mbarrier_G2S_buffer = nullptr;
    }

    // inter_node_mbarrier_G2S_buffer
    align_offset(8);
    layout.inter_node_mbarrier_G2S_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * 2 * sizeof(uint64_t);

    // intra_node_to_rdma_mbarrier_buffer (only if multi-node)
    if (model.num_of_nodes > 1) {
        int max_num_of_chunks_per_rank =
            (model.max_num_of_tokens_per_rank + num_of_tokens_per_chunk - 1) / num_of_tokens_per_chunk;
        align_offset(8);
        layout.intra_node_to_rdma_mbarrier_buffer =
            reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_of_nodes - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
    } else {
        layout.intra_node_to_rdma_mbarrier_buffer = nullptr;
    }

    if (model.num_of_nodes > 1) {
        align_offset(8);
        layout.combine_memory_region_info =
            reinterpret_cast<combine_memory_region_info_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_of_nodes - 1) * sizeof(combine_memory_region_info_t);
    } else {
        layout.combine_memory_region_info = nullptr;
    }

    // Flag buffers (no special alignment needed)
    if (multinode) {
        layout.intra_node_flag_G2S_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * sizeof(bool);
    } else {
        layout.intra_node_flag_G2S_buffer = nullptr;
    }

    layout.inter_node_flag_G2S_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * sizeof(bool);

    // Streaming overlap fields (multi-node only, 4B aligned)
    if (multinode) {
        align_offset(4);
        layout.rdma_streaming_counter = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += sizeof(uint32_t);
    } else {
        layout.rdma_streaming_counter = nullptr;
    }

    return layout;
}

template <ncclDataType_t kTokenDtype = ncclBfloat16>
static size_t calculate_combine_smem_layout_size(
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_chunk,
    int max_num_of_tokens_per_rank,
    int num_lsa_teams,
    bool backward_combine,
    const model_config_t& model) {
    // Dynamically computes the size required for combine shared memory layout,
    // mirroring the logic from create_combine_smem_layout
    size_t total_size = 0;

    // Compute max number of chunks per rank
    const int hidden_dim = model.hidden_dim;
    const int token_bytes = hidden_dim * nccl_ep::size_u8<kTokenDtype>(); // per-token wire bytes
    const int max_num_of_chunks_per_rank =
        (max_num_of_tokens_per_rank + num_of_tokens_per_chunk - 1) / num_of_tokens_per_chunk;
    const bool multinode = (num_lsa_teams > 1);

    // Token buffers (128B aligned for TMA). Stage stride scales with the wire element
    // width (2 B for BF16/FP16, 4 B for FP32).
    // intra_node_token_* buffers (multi-node only)
    if (multinode) {
        total_size = (total_size + 127) & ~127;
        total_size += num_of_stages_g2s * token_bytes;

        total_size = (total_size + 127) & ~127;
        total_size += num_of_stages_s2g * token_bytes;
    }

    // inter_node_token_G2S_buffer
    total_size = (total_size + 127) & ~127;
    total_size += num_of_stages_g2s * token_bytes;

    // inter_node_token_S2G_buffer
    total_size = (total_size + 127) & ~127;
    total_size += num_of_stages_s2g * token_bytes;

    // Prob buffers (16B aligned, only if backward_combine)
    if (backward_combine) {
        if (multinode) {
            // intra_node_prob_G2S_buffer
            total_size = (total_size + 15) & ~15;
            total_size +=
                num_of_stages_g2s * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);

            // intra_node_prob_S2G_buffer
            total_size = (total_size + 15) & ~15;
            total_size +=
                num_of_stages_s2g * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);
        }

        // inter_node_prob_G2S_buffer
        total_size = (total_size + 15) & ~15;
        total_size += num_of_stages_g2s * model.num_of_experts_per_rank * model.ranks_per_lsa_team * sizeof(float);

        // inter_node_prob_S2G_buffer
        total_size = (total_size + 15) & ~15;
        total_size += num_of_stages_s2g * model.num_of_experts_per_rank * model.ranks_per_lsa_team * num_lsa_teams *
                      sizeof(float);
    }

    // Mbarrier buffers (8B aligned)
    // intra_node_mbarrier_G2S_buffer [stages][2] (multi-node only)
    if (multinode) {
        total_size = (total_size + 7) & ~7;
        total_size += num_of_stages_g2s * 2 * sizeof(uint64_t);
    }

    // inter_node_mbarrier_G2S_buffer [stages][2]
    total_size = (total_size + 7) & ~7;
    total_size += num_of_stages_g2s * 2 * sizeof(uint64_t);

    // intra_node_to_rdma_mbarrier_buffer [(nodes-1)][chunks] (only if multi-node)
    if (multinode) {
        total_size = (total_size + 7) & ~7;
        total_size += (num_lsa_teams - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
    }

    // combine_memory_region_info [(nodes-1)] (align 8B, only if multi-node)
    if (multinode) {
        total_size = (total_size + 7) & ~7;
        total_size += (num_lsa_teams - 1) * sizeof(combine_memory_region_info_t);
    }

    // Flag buffers (no special alignment needed)
    if (multinode) {
        total_size += num_of_stages_g2s * sizeof(bool);
    }
    total_size += num_of_stages_g2s * sizeof(bool);

    // Streaming overlap fields (multi-node only, 4B aligned)
    if (multinode) {
        total_size = (total_size + 3) & ~3;
        total_size += sizeof(uint32_t); // rdma_streaming_counter
    }

    return total_size;
}
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
    const uint8_t*
        attn_input_token_scaling_factor; // FP8 EXTERN: per-token scales (float* for FP32, uint8_t* for UE8M0 — pure byte transport).
    // Internal temp buffers. These buffers are local buffers.
    uint64_t* rdma_inter_node_group_flags; // For RDMA Atomic flags.
    uint32_t* intra_node_write_completion_flags; // For intra-node S2G write completion notification.
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
    uint64_t* expected_rdma_flag_value;
    uint32_t* expected_intra_node_flag_value;
    int local_rank;
    int node_rank;
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
    // Cross-round WAR sync-guards: LSA (intra-node staging) uses the NCCL LSA barrier; RDMA
    // (inter-node staging) is hand-rolled. Only the enable flags are needed on the device now.
    bool guard_enabled; // cross-round WAR guard (LSA + RDMA share one enable)
    // Backstop bound for recv slot indices; the scan never publishes slots at or
    // above it (DROP masks the rest), so the S2G assert only fires on corrupted
    // or stale routing maps.
    int max_recv_tokens_per_rank;
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
    uint8_t* expert_output_scaling_factor[LSA_TEAM_SZ]; // Only valid for SCALES_FORWARD.
};

// Fixed-size part of combine kernel parameters. Peer pointer arrays are appended
// by combine_kernel_param_t<LSA_TEAM_SZ> for JIT-specialized kernels.
struct combine_kernel_param_base_t {
    int hidden_dim;
    int experts_per_rank;
    int ranks_per_lsa_team;
    // EM unfused-combine: when true, the inter-node G2S warp group skips local-dup
    // secondary em_slots (primaries already carry the pre-reduced weighted sum
    // written by the local_reduce kernel). Default false => fused fanout.
    bool combine_local_reduce_enabled;
    // Output buffers. These buffers are local buffers.
    uint16_t* attn_output_token;
    float* attn_output_prob;
    // Internal temp buffers. These buffers are local buffers.
    uint16_t* rdma_intra_node_red_token;
    float* rdma_intra_node_red_prob;
    const uint16_t* rdma_inter_node_group_token;
    const float* rdma_inter_node_group_prob;
    uint64_t* rdma_inter_node_group_flags;
    uint32_t* intra_node_write_completion_flags; // For intra-node src ready notification.
    // Metadata buffers. These buffers are local buffers.
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // flat: lsa_team_size, expert-major: num_topk
    // Device-resident expected counters. Initialized at bootstrap; bumped in
    // this kernel's tail so CUDA-graph capture+replay self-sequences.
    uint64_t* expected_rdma_flag_value;
    uint32_t* expected_intra_node_flag_value;
    int local_rank;
    int node_rank;
    // Stride for routing-map indexing (= max_tokens_per_rank).
    int num_of_tokens_per_rank;
    // Actual token count; gates the inter_node_red TMA store.
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
    // Cross-round WAR sync-guards: LSA (intra-node staging) uses the NCCL LSA barrier; RDMA
    // (inter-node staging) is hand-rolled. Only the enable flags are needed on the device now.
    bool guard_enabled; // cross-round WAR guard (LSA + RDMA share one enable)
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
// We want to use 1 for intra-node reduction warp group, >= 2 for inter-node reduction warp group,
// RDMA warp group currently only contains 1 warp so does not use named bar yet, if it need to use, it should use 2 + NUM_OF_DATA_PIPELINE_PER_BLOCK.
__forceinline__ __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id = 0) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

// Helper to compute communicator index and context index from global channel
// Used for 6-comm x 4-ctx GIN configuration (6 communicators with 4 contexts each = 24 total channels)
__forceinline__ __device__ void get_comm_ctx(int global_channel, int num_ctx_per_comm, int& comm_idx, int& ctx_idx) {
    comm_idx = global_channel / num_ctx_per_comm;
    ctx_idx = global_channel % num_ctx_per_comm;
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

// Spin until an mbarrier reaches the expected phase parity.
__forceinline__ __device__ void mbarrier_wait(uint64_t* mbar, uint32_t parity) {
    while (!cuda::ptx::mbarrier_try_wait_parity(mbar, parity)) {
    }
}

// Put one token's bundle (token, +prob if FWD, +sf if quantized) to a remote node, packed from dst_offset.
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

// Resolved G2S source for one (node, chunk): packed-remote entry base, or strided-local base pointers.
template <typename TOKEN_DATA_TYPE>
struct g2s_source_t {
    bool use_packed;
    const uint8_t* packed_base;
    const TOKEN_DATA_TYPE* token_base;
    const float* prob_base;
    const uint8_t* sf_base;
};

// Resolve a (node, chunk) source: packed-remote base (after waiting the RDMA arrival signal) or strided-local bases.
template <
    typename TOKEN_DATA_TYPE,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int MAX_NUM_OF_TOKENS_PER_RANK,
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
        constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
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

        int ctx_idx = signal_channel % num_ctx_per_comm;
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
            // attn_input_prob is laid out per token as LSA_TEAMS blocks of experts_per_node floats.
            const int experts_per_node = experts_per_rank * LSA_TEAM_SZ;
            const int prob_row_stride = experts_per_node * LSA_TEAMS;
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
    uint64_t* mbar = smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage);
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
            // Advance by whole token rows, then pick this node's expert slice within the row.
            const int experts_per_node = experts_per_rank * LSA_TEAM_SZ;
            const int prob_row_stride = experts_per_node * LSA_TEAMS;
            prob_src = src.prob_base + cur_tokid * prob_row_stride + my_lteam * experts_per_node;
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

// TMA-load one (node, chunk)'s s2d-map slice into the SMEM stage and publish. Caller elects one lane.
template <typename SMEM_TYPE, int NUM_OF_TOKENS_PER_CHUNK>
__forceinline__ __device__ void dispatch_s2g_prefetch_s2d_map(
    const int32_t* sparse_to_dense_map,
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const uint32_t s2d_map_stage,
    const int node_id,
    const int chunk_id,
    const int chunk_size,
    const int num_of_tokens_per_rank,
    const int s2d_inner_dim) {
    const int32_t* s2d_base =
        sparse_to_dense_map + (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * s2d_inner_dim;
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
    const void* token_dst = remote_expert_output_token[dst.remote_rank_id] + (dst.output_buffer_index * HIDDEN_DIM);
    const void* prob_dst = nullptr;
    const void* sf_dst = nullptr;
    if constexpr (FORWARD_DISPATCH) {
        prob_dst = remote_expert_output_prob[dst.remote_rank_id] +
                   (dst.output_buffer_index * (experts_per_rank * LSA_TEAM_SIZE));
    }
    if constexpr (HAS_SF) {
        sf_dst = remote_expert_output_scaling_factor[dst.remote_rank_id] + dst.output_buffer_index * sf_bytes_per_token;
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

// Device function for inter-node node2node(RDMA) warp for dispatch kernel.
template <
    typename GIN_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_OF_STAGES,
    int TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
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
        MAX_NUM_OF_TOKENS_PER_RANK % TOKENS_PER_CHUNK == 0,
        "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of TOKENS_PER_CHUNK.");

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
    constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    constexpr int NUM_REMOTE_NODES = LSA_TEAMS - 1;

    // GIN device side setup. Single communicator; ctx_idx spreads QP traffic.
    int global_channel = blockIdx.x * N2N_WARPS + n2n_warp_id;
    int ctx_idx = global_channel % num_ctx_per_comm;

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

        for (int j = 0; j < NUM_REMOTE_NODES; ++j) {
            // Skip-self ring over the other nodes, load-balanced start at my_lteam.
            int remote_idx = (j + my_lteam) % NUM_REMOTE_NODES;
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
                    attn_to_rdma_map[((token_idx_in_chunk + chunk_first_token_idx) * NUM_REMOTE_NODES) + remote_idx];
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
                    attn_to_rdma_map[((chunk_first_token_idx + token_idx_in_chunk) * NUM_REMOTE_NODES) + remote_idx];
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

// Dispatch intra-node S2G warp group. With NUM_PIPELINES > 1, each warp is an
// independent pipeline consumer paired with the G2S warp of the same pipeline_rank.
template <
    typename LSA_S2G_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_OF_STAGES,
    int NUM_OF_IN_FLIGHT_S2G,
    int TOKENS_PER_CHUNK,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int NBLOCKS,
    int NUM_PIPELINES,
    bool FORWARD_DISPATCH,
    bool HAS_SF,
    ncclEpLayout_t kLayout>
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
    SMEM_TYPE* smem_buffer_ptr) {
    constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;
    static_assert(
        NUM_OF_IN_FLIGHT_S2G < STAGES_PER_PIPELINE,
        "NUM_OF_IN_FLIGHT_S2G must be smaller than STAGES_PER_PIPELINE.");
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
    const int routing_map_node_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
    int in_flight_s2g = 0;
    int stage = 0;
    uint32_t producer_parity = 0;
    uint32_t s2d_stage = 0;
    uint32_t s2d_parity = 0;

    // S2G on all 32 lanes (warp-uniform state); cp_async_bulk striped by lane=flat_idx (up to s2d_inner_dim stores/token).
    const int s2g_lane = LSA_S2G_GROUP::thread_rank() % 32;

    // Each pipeline prefetches its own first s2d map for its first chunk (single TMA load, lane 0 only).
    if (s2g_lane == 0) {
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

                // Prefetch next (chunk, node) s2d map for THIS pipeline (single TMA load, lane 0 only).
                if (s2g_lane == 0) {
                    int next_chunk_id;
                    int next_node_id;
                    int next_node_iter = j + 1;
                    if (next_node_iter < LSA_TEAMS) {
                        next_chunk_id = cidx;
                        next_node_id = (my_lteam + LSA_TEAMS - next_node_iter) % LSA_TEAMS;
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
                        next_node_id = my_lteam;
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
                            next_node_id,
                            next_chunk_id,
                            next_chunk_size,
                            num_of_tokens_per_rank,
                            s2d_inner_dim);
                    }
                }

                // Walk nodes backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int lteam_id = (my_lteam + LSA_TEAMS - j) % LSA_TEAMS;
                const routing_loads_t* routing_map_ptr = reinterpret_cast<const routing_loads_t*>(
                    rdma_to_attn_map + (lteam_id * routing_map_node_stride + cidx * TOKENS_PER_CHUNK));

                {
                    uint64_t* wait_mbar = smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, s2d_stage);
                    mbarrier_wait(wait_mbar, s2d_parity);
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
                            const int32_t* s2d_smem_row =
                                smem_buffer_ptr->get_s2d_map_buffer(pipeline_rank, s2d_stage, cur_tokid);
                            mbarrier_wait(
                                smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage),
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
                            if (in_flight_s2g > NUM_OF_IN_FLIGHT_S2G) {
                                if (s2g_lane < s2d_inner_dim) {
                                    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_IN_FLIGHT_S2G>{});
                                }
                                __syncwarp();
                                in_flight_s2g -= 1;
                                int notify_stage = (stage - NUM_OF_IN_FLIGHT_S2G) >= 0 ?
                                                       (stage - NUM_OF_IN_FLIGHT_S2G) :
                                                       (stage - NUM_OF_IN_FLIGHT_S2G + STAGES_PER_PIPELINE);
                                if (s2g_lane == 0) {
                                    cuda::ptx::mbarrier_arrive(
                                        smem_buffer_ptr->get_intra_node_mbarrier_consumer(pipeline_rank, notify_stage));
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

// Dispatch intra-node G2S warp group. With NUM_PIPELINES > 1, each warp is an independent
// pipeline processing disjoint chunks through its own partition of the shared-memory FIFO.
template <
    typename LSA_G2S_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_OF_STAGES,
    int TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int LSA_TEAM_SZ,
    int NBLOCKS,
    int NUM_PIPELINES,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ void dispatch_G2S_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const TOKEN_DATA_TYPE* attn_input_token,
    const float* attn_input_prob,
    const uint8_t* attn_input_token_scaling_factor,
    // OUTPUT
    uint64_t* rdma_inter_node_group_flags,
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
        MAX_NUM_OF_TOKENS_PER_RANK % TOKENS_PER_CHUNK == 0,
        "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of TOKENS_PER_CHUNK.");

    constexpr int TOKENS_PER_ROUTING_LOAD = sizeof(routing_loads_t) / sizeof(bool);
    constexpr int ROUTING_LOADS_PER_CHUNK = TOKENS_PER_CHUNK / TOKENS_PER_ROUTING_LOAD;
    constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;

    const int pipeline_rank = LSA_G2S_GROUP::warp_rank();
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int max_num_of_chunks_per_rank = nccl_ep::ceil_div(MAX_NUM_OF_TOKENS_PER_RANK, TOKENS_PER_CHUNK);
    // TOKENS_PER_ROUTING_LOAD must match the producer's pad in scan_kernel.cuh
    const int routing_map_node_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
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
                // Walk nodes backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int lteam_id = (my_lteam + LSA_TEAMS - j) % LSA_TEAMS;

                g2s_source_t<TOKEN_DATA_TYPE> src = dispatch_g2s_resolve_source<
                    TOKEN_DATA_TYPE,
                    LSA_TEAMS,
                    LSA_TEAM_SZ,
                    MAX_NUM_OF_TOKENS_PER_RANK,
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
                    rdma_to_attn_map + (lteam_id * routing_map_node_stride) + (cidx * TOKENS_PER_CHUNK));

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
                                    smem_buffer_ptr->get_intra_node_mbarrier_consumer(pipeline_rank, stage);
                                mbarrier_wait(mbar, consumer_parity);
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
                rdma_inter_node_group_flags + (lteam_id * max_num_of_chunks_per_rank) + num_of_chunks_per_rank;
            if (LSA_G2S_GROUP::thread_rank() < residue_flag_count) {
                residue_flag_base_ptr[LSA_G2S_GROUP::thread_rank()] = expected_flag_value;
            }
        }
    }
}

// Shared single-entry G2S issue for both the intra- and inter-node combine
// G2S warps. INTER_NODE selects the inter-node staged-buffer accessors and
// flag buffer; otherwise the intra-node ones (both sets live on the same
// combine smem layout, so one helper covers both tiers).
//   - derive stage_idx + parity from (global_offset + rank_in_batch)
//   - wait for consumer to free the stage (mbarrier_try_wait_parity)
//   - cp_async_bulk the token (and the prob under BACKWARD_COMBINE)
//   - optionally write <tier>_flag_G2S_buffer[stage_idx]
//   - mbarrier_arrive_expect_tx with the cumulative tx size
//
// The caller has already computed the token (and prob) source pointers
// and the `is_last_entry` boolean (only consulted when WRITE_LAST_FLAG).
template <bool INTER_NODE, bool BACKWARD_COMBINE, bool WRITE_LAST_FLAG, typename SMEM_TYPE>
__forceinline__ __device__ void issue_g2s_entry(
    SMEM_TYPE* smem_buffer_ptr,
    int global_offset,
    int rank_in_batch,
    int starting_G2S_index,
    int ring_len,
    const uint16_t* token_src,
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
    if constexpr (INTER_NODE) {
        consumer_mbar = smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(stage_idx);
        producer_mbar = smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx);
        token_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(stage_idx));
    } else {
        consumer_mbar = smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(stage_idx);
        producer_mbar = smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(stage_idx);
        token_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_token_G2S(stage_idx));
    }

    while (!cuda::ptx::mbarrier_try_wait_parity(consumer_mbar, parity)) {
    }

    uint32_t total_tx_size = 0;
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        token_dst,
        reinterpret_cast<const void*>(token_src),
        token_bytes,
        producer_mbar);
    total_tx_size += token_bytes;

    if constexpr (BACKWARD_COMBINE) {
        void* prob_dst;
        if constexpr (INTER_NODE) {
            prob_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(stage_idx));
        } else {
            prob_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_prob_G2S(stage_idx));
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
        if constexpr (INTER_NODE) {
            smem_buffer_ptr->inter_node_flag_G2S_buffer[stage_idx] = is_last_entry;
        } else {
            smem_buffer_ptr->intra_node_flag_G2S_buffer[stage_idx] = is_last_entry;
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
// broadcast-issue, shared by the intra-node G2S warp and the LOCAL tier of
// the inter-node G2S warp (the two differ only in INTER_NODE, starting_G2S_index,
// and which s2d row / FIFO slice they target -- all passed in).
//
//  * Process s2d entries in WARP_SIZE steps
//  * Each lane loads an s2d entry and participates in valid entry filtering
//  * Next a single lane performs TMA loads of the token and probability data
//    * NOTE: Using single lane allows to ensure the ordering of TMA loads so
//      they are aligned with the RED process.
template <
    bool INTER_NODE,
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
    uint16_t* const* remote_expert_input_token,
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
        const uint16_t* token_src =
            remote_expert_input_token[rank_id] + (slot * HIDDEN_DIM * nccl_ep::size_u16<kTokenDtype>());
        const float* prob_src = nullptr;
        if constexpr (BACKWARD_COMBINE) {
            prob_src = remote_expert_input_prob[rank_id] + (slot * (experts_per_rank * ranks_per_lsa_team));
        }
        issue_g2s_entry<INTER_NODE, BACKWARD_COMBINE, /*WRITE_LAST_FLAG=*/true>(
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

// Decode a block's flat chunk index into its destination node, per-node chunk id, and token
// count. Emit order (chunk c for node+1..node-1, then c+1) matches the RDMA consume order.
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

// Intra-node G2S warp for the combine kernel. Exactly one such warp per block.
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
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void combine_G2S_intra_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const int32_t* sparse_to_dense_map,
    uint16_t* const* remote_expert_input_token,
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
    // rdma_to_attn_map is padded to 16B (16 bools) per node.
    const int rdma_map_size_per_node = nccl_ep::align(num_of_tokens_per_rank, 16);

    // Warp-cooperative G2S: lanes scan sparse_to_dense_map in parallel and issue TMA to distinct
    // stages; global_offset counts total stages filled so RED consumes them in lockstep.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    constexpr int ring_len = STAGES_G2S;
    // Wire token width: 2 B for BF16/FP16 (default), 4 B for FP32 NONE.
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>()));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SZ) * sizeof(float));

    // EM unfused-combine dedup uses __shfl_up_sync(1); requires s2d_inner_dim <= WARP_SIZE.
    if (combine_local_reduce_enabled && lane_id == 0 && smem_buffer_ptr->s2d_inner_dim > WARP_SIZE) {
        __trap();
    }

    int global_offset = 0;

    for (int cidx = blockIdx.x; cidx < total_chunks; cidx += NBLOCKS) {
        const auto meta = combine_chunk_meta<LSA_TEAMS, TOKENS_PER_CHUNK>(cidx, my_lteam, cpr, rem_chunk_sz);

        const bool* rdma_row_base =
            rdma_to_attn_map + (meta.lteam_id * rdma_map_size_per_node + meta.chunk_id * TOKENS_PER_CHUNK);

        // S2D inner dim: mode-dependent, carried by SMEM layout struct.
        const int s2d_entries_g2s = smem_buffer_ptr->s2d_inner_dim;
        const int32_t* s2d_row_base =
            sparse_to_dense_map +
            (meta.lteam_id * num_of_tokens_per_rank + meta.chunk_id * TOKENS_PER_CHUNK) * s2d_entries_g2s;

        for (int cur_tokid = 0; cur_tokid < meta.csize; cur_tokid++) {
            // Skip dst tokens this node doesn't need.
            if (!rdma_row_base[cur_tokid]) {
                continue;
            }

            const int32_t* sparse_to_dense_row = s2d_row_base + cur_tokid * s2d_entries_g2s;

            // Warp-cooperative s2d-row scan with inline broadcast-issue; advances global_offset
            // by the number of entries issued. See issue_local_g2s_row.
            issue_local_g2s_row</*INTER_NODE=*/false, BACKWARD_COMBINE, kLayout, HIDDEN_DIM, kTokenDtype>(
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

// Reduce all G2S source-token contributions for one destination token into FP32 registers
// (+prob into SMEM for backward). Advances the G2S stage cursor/parity to the next dst token.
template <
    typename RED_GROUP,
    int STAGES_G2S,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
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
            reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_G2S(token_stage));
        float* load_prob_base_ptr;
        if constexpr (BACKWARD_COMBINE) {
            load_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_G2S(token_stage);
        }

        // Warp 0 waits for the producer; then the whole reduction group reads this stage.
        if (RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage), token_producer_parity)) {
                }
            }
        }
        arrive_and_wait(RED_GROUP::size(), 1);

// Accumulate the register-resident token. NONE-FP16 reads __half2, NONE-FP32 reads float2 and
// skips precision conversion; predicates are launch-uniform so branching is free.
#pragma unroll
        for (int n = 0; n < NUM_ACC; n++) {
            int element_id = (n * RED_GROUP::size()) + RED_GROUP::thread_rank();
            if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                float2 src_data_fp32 = nccl_ep::ld_token_pair<kTokenDtype>(load_token_base_ptr, element_id);
                acc_token_fp32[n].x += src_data_fp32.x;
                acc_token_fp32[n].y += src_data_fp32.y;
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

        last_src_token = smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage];

        // All reduction threads must finish reading before the producer reuses this stage.
        arrive_and_wait(RED_GROUP::size(), 1);
        if (RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(token_stage));
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
// TMA-copy it to the per-destination intra-node red buffer. Advances the S2G stage cursor.
template <
    typename RED_GROUP,
    int NUM_OF_STAGES_S2G,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
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
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

    __nv_bfloat162* store_token_base_ptr =
        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage));

    // Ensure any earlier TMA read from this S2G stage has completed before we overwrite it.
    if (RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G - 1>{});
        }
    }
    arrive_and_wait(RED_GROUP::size(), 1);

// Store the register-resident token (NONE-FP16 packs __half2, NONE-FP32 writes float2 verbatim).
#pragma unroll
    for (int n = 0; n < NUM_ACC; n++) {
        int element_id = (n * RED_GROUP::size()) + RED_GROUP::thread_rank();
        if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
            nccl_ep::st_token_pair<kTokenDtype>(store_token_base_ptr, element_id, acc_token_fp32[n]);
        }
    }
    if constexpr (BACKWARD_COMBINE) {
        float* store_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_S2G(dst_token_stage);
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
            uint16_t* current_token_addr = red_token_base + cur_tokid * red_token_bytes / sizeof(uint16_t);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_global,
                cuda::ptx::space_shared,
                reinterpret_cast<void*>(current_token_addr),
                reinterpret_cast<const void*>(smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage)),
                (uint32_t)(red_token_bytes));

            if constexpr (BACKWARD_COMBINE) {
                float* current_prob_addr = red_prob_base + cur_tokid * prob_dim;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    reinterpret_cast<void*>(current_prob_addr),
                    reinterpret_cast<const void*>(smem_buffer_ptr->get_intra_node_prob_S2G(dst_token_stage)),
                    (uint32_t)(prob_dim * sizeof(float)));
            }
            cuda::ptx::cp_async_bulk_commit_group();
        }
    }

    dst_token_stage += 1;
    if (dst_token_stage == NUM_OF_STAGES_S2G) {
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

// Intra-node reduction warp group for the combine kernel.
template <
    typename RED_GROUP,
    typename SMEM_TYPE,
    int STAGES_G2S,
    int NUM_OF_STAGES_S2G,
    int TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SZ,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void combine_RED_intra_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    // OUTPUT
    uint16_t* rdma_intra_node_red_token,
    float* rdma_intra_node_red_prob,
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
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
    constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    constexpr int NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA =
        nccl_ep::ceil_div(NUM_OF_BF16X2_ELEMENTS_PER_TOKEN, (int)RED_GROUP::size());
    // Backward-combine prob vectors stay in float (no BF16 packing).
    const int prob_dim = experts_per_rank * LSA_TEAM_SZ;
    const int prob_vec_per_thread = nccl_ep::ceil_div(prob_dim, (int)RED_GROUP::size());
    // Compile-time upper bound sized exactly to this instantiation's LSA team.
    constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
        nccl_ep::ceil_div(NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SZ, (int)RED_GROUP::size());

    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = (LSA_TEAMS - 1) * cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per node.
    const int rdma_map_size_per_node = nccl_ep::align(num_of_tokens_per_rank, 16);

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
        const int rdma_intra_node_red_id =
            rdma_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + meta.chunk_id * TOKENS_PER_CHUNK;
        // Vector loads covering this chunk's routing flags (tail chunk is shorter).
        const int routing_loads_for_chunk = (rem_chunk_sz != 0 && meta.chunk_id == cpr - 1)
                                                 ? nccl_ep::ceil_div(rem_chunk_sz, (int)sizeof(routing_loads_t))
                                                 : ROUTING_LOADS_PER_CHUNK;

        const routing_loads_t* rdma_map_base = reinterpret_cast<const routing_loads_t*>(
            rdma_to_attn_map + (meta.lteam_id * rdma_map_size_per_node + meta.chunk_id * TOKENS_PER_CHUNK));

        // Per-token stride scaled into uint16_t units (HIDDEN_DIM for BF16/FP16, 2*HIDDEN_DIM for FP32).
        uint16_t* red_token_base =
            rdma_intra_node_red_token + rdma_intra_node_red_id * HIDDEN_DIM * nccl_ep::size_u16<kTokenDtype>();
        float* red_prob_base = nullptr;
        if constexpr (BACKWARD_COMBINE) {
            red_prob_base = rdma_intra_node_red_prob + rdma_intra_node_red_id * prob_dim;
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
                // Skip dst tokens this node doesn't need.
                if (!*(reinterpret_cast<bool*>(&routing_data) + token_in_load)) {
                    continue;
                }

                float2 acc_token_fp32[NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA];
                // acc_prob storage instantiated only in backward specializations.
                using acc_prob_storage_type =
                    acc_prob_storage_t<BACKWARD_COMBINE, MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD>;
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
                    kTokenDtype>(
                    smem_buffer_ptr,
                    token_stage,
                    token_producer_parity,
                    acc_token_fp32,
                    acc_prob_ptr,
                    prob_vec_per_thread,
                    prob_dim);

                combine_store_reduced_token<
                    RED_GROUP,
                    NUM_OF_STAGES_S2G,
                    BACKWARD_COMBINE,
                    HIDDEN_DIM,
                    kTokenDtype>(
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
                    cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer
                                                    [rdma_tile_id * MAX_NUM_OF_CHUNKS_PER_RANK + meta.chunk_id]);
                }
            }
        }
    }
}

// RDMA-put the active tokens (per rdma_to_attn_map) of one chunk to the remote node, coalescing
// contiguous runs into batches of at most MAX_BATCH (token + optional prob). Under STREAMING each flush
// first waits for the intra-node reduction warp to publish enough produced tokens (streaming_counter)
// and advances cumulative_sent; otherwise the whole chunk is assumed reduced. Lane 0 only.
template <
    bool STREAMING,
    bool BACKWARD_COMBINE,
    int MAX_BATCH,
    int MAX_NUM_OF_TOKENS_PER_RANK,
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
                smem_mr_info_ptr->rdma_intra_node_red_token_offset +
                (rdma_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * TOKEN_BYTES;
            size_t token_dst_offset =
                smem_mr_info_ptr->combine_rdma_inter_node_group_token_offset +
                (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * TOKEN_BYTES;
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
                    smem_mr_info_ptr->rdma_intra_node_red_prob_offset +
                    (rdma_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * prob_dim * sizeof(float);
                size_t prob_dst_offset =
                    smem_mr_info_ptr->combine_rdma_inter_node_group_prob_offset +
                    (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * prob_dim * sizeof(float);
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

// Signal the remote node (via ncclGin) that this chunk has been delivered. Lane 0 only.
template <int MAX_NUM_OF_TOKENS_PER_RANK, int TOKENS_PER_CHUNK, int LSA_TEAMS>
__forceinline__ __device__ void combine_n2n_signal_remote(
    ncclGin& net,
    ncclTeam rail,
    int lteam_id,
    int local_rank,
    int my_lteam,
    int chunk_id,
    unsigned signals_base,
    unsigned combine_signal_offset) {
    constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
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

// Inter-node N2N (RDMA) warp group for the combine kernel. Exactly one such warp per block;
// uses the ncclGin API (net.put / net.signal).
template <
    typename GIN_GROUP,
    typename SMEM_TYPE,
    int NUM_OF_STAGES_S2G,
    int TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    int LSA_TEAM_SZ,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype>
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
    // mr_info and the intra-node -> rdma mbarrier buffer are staged in shared memory.
    struct combine_memory_region_info_t* smem_mr_info_ptr = nullptr;
    uint64_t* intra_node_to_rdma_mbarrier_buffer_ptr = nullptr;
    constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    if constexpr (LSA_TEAMS != 1) {
        smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
        if (GIN_GROUP::thread_rank() == 0) {
            smem_mr_info_ptr[0] = mr_info[0];
        }
        intra_node_to_rdma_mbarrier_buffer_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;
    }
    __syncwarp();

    // Chunks this rank produces for the RDMA warps to consume; residue slots pad up to the max.
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = (LSA_TEAMS - 1) * MAX_NUM_OF_CHUNKS_PER_RANK;
    // rdma_to_attn_map is padded to 16B (16 bools) per node.
    const int rdma_map_size_per_node = nccl_ep::align(num_of_tokens_per_rank, 16);
    uint32_t token_consumer_parity = 0;
    uint32_t cumulative_sent = 0; // Cumulative active tokens RDMA-put across all chunks (never reset).
    for (int cidx = blockIdx.x; cidx < total_chunks; cidx += NBLOCKS) {
        // Node/chunk mapping shared with the G2S receiver so signals match.
        const auto meta = combine_chunk_meta<LSA_TEAMS, TOKENS_PER_CHUNK>(cidx, my_lteam, cpr, rem_chunk_sz);
        const int lteam_id = meta.lteam_id;
        const int chunk_id = meta.chunk_id;
        // With rail-scoped GIN comms, node index maps to rail-team rank.
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
        int chunk_base_token_idx = lteam_id * rdma_map_size_per_node + chunk_id * TOKENS_PER_CHUNK;
        // Residue chunks carry no tokens; real chunks use the scheduled size (tail = remainder).
        const int token_range = is_residue ? 0 : meta.csize;
        constexpr int STREAMING_BATCH = NCCL_EP_HT_COMBINE_RDMA_STREAMING_BATCH;
        // Per-token wire bytes (compile-time): hidden x element width.
        constexpr size_t token_bytes = static_cast<size_t>(HIDDEN_DIM) * nccl_ep::size_u8<kTokenDtype>();
        if constexpr (STREAMING_BATCH > 0) {
            // ---- STREAMING PATH: process tokens as reduction warp produces them ----
            // cumulative_sent tracks total active tokens across all chunks (no reset).

            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_put_active_tokens</*STREAMING=*/true, BACKWARD_COMBINE, STREAMING_BATCH,
                                              MAX_NUM_OF_TOKENS_PER_RANK, token_bytes>(
                    net, rail, lteam_id, nccl_token_window, nccl_prob_window, nccl_internal_window, smem_mr_info_ptr,
                    rdma_to_attn_map, chunk_base_token_idx, chunk_id * TOKENS_PER_CHUNK, token_range,
                    rdma_tile_id, rank_in_remote, experts_per_rank * LSA_TEAM_SZ,
                    (volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter, cumulative_sent);
            }

            // Wait for mbarrier (parity tracking -- reduction warp always arrives)
            if (!is_residue) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    &intra_node_to_rdma_mbarrier_buffer_ptr
                        [rdma_tile_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id],
                    token_consumer_parity)) {
                }
            }

            // Signal remote
            __syncwarp();
            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_signal_remote<MAX_NUM_OF_TOKENS_PER_RANK, TOKENS_PER_CHUNK, LSA_TEAMS>(
                    net, rail, lteam_id, local_rank, my_lteam, chunk_id, signals_base, combine_signal_offset);
            }
            __syncwarp();

            // No consumed handshake needed: cumulative counter never resets.

        } else {
            // ---- FALLBACK PATH (STREAMING_BATCH == 0): mbarrier-first, then put ----
            if (!is_residue) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    &intra_node_to_rdma_mbarrier_buffer_ptr
                        [rdma_tile_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id],
                    token_consumer_parity)) {
                }
            }

            if (!is_residue && GIN_GROUP::thread_rank() == 0) {
                constexpr int max_batch = NCCL_EP_HT_DISPATCH_RDMA_BATCH_SIZE;
                combine_n2n_put_active_tokens</*STREAMING=*/false, BACKWARD_COMBINE, max_batch,
                                              MAX_NUM_OF_TOKENS_PER_RANK, token_bytes>(
                    net, rail, lteam_id, nccl_token_window, nccl_prob_window, nccl_internal_window, smem_mr_info_ptr,
                    rdma_to_attn_map, chunk_base_token_idx, chunk_id * TOKENS_PER_CHUNK, token_range,
                    rdma_tile_id, rank_in_remote, experts_per_rank * LSA_TEAM_SZ,
                    nullptr, cumulative_sent);
            }

            // Signal remote
            __syncwarp();
            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_signal_remote<MAX_NUM_OF_TOKENS_PER_RANK, TOKENS_PER_CHUNK, LSA_TEAMS>(
                    net, rail, lteam_id, local_rank, my_lteam, chunk_id, signals_base, combine_signal_offset);
            }
            __syncwarp();
        }
    }
    token_consumer_parity ^= 1;
}

// One warp lane's remote node for the combine inter-node RDMA tier and whether that node
// needs the current dst token. tile_id indexes the per-remote-node RDMA buffers.
struct rdma_lane_t {
    bool valid;
    int tile_id;
};

// Map lane_id -> remote node (lane 0 -> my_lteam-1, wrapping past self) and read its routing bit.
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

// Token (+prob) source pointers for one remote token in the RDMA inter-node group buffers.
struct g2s_src_t {
    const uint16_t* token_src;
    const float* prob_src;
};

template <bool BACKWARD_COMBINE, int HIDDEN_DIM, ncclDataType_t kTokenDtype, int MAX_NUM_OF_TOKENS_PER_RANK>
__forceinline__ __device__ g2s_src_t combine_g2s_resolve_rdma_source(
    const uint16_t* rdma_inter_node_group_token,
    const float* rdma_inter_node_group_prob,
    int tile_id,
    int flat_token_id,
    int experts_per_rank,
    int lteam_sz) {
    const int rdma_row = tile_id * MAX_NUM_OF_TOKENS_PER_RANK + flat_token_id;
    const uint16_t* token_src =
        rdma_inter_node_group_token + rdma_row * HIDDEN_DIM * nccl_ep::size_u16<kTokenDtype>();
    const float* prob_src = nullptr;
    if constexpr (BACKWARD_COMBINE) {
        prob_src = rdma_inter_node_group_prob + rdma_row * (experts_per_rank * lteam_sz);
    }
    return g2s_src_t{token_src, prob_src};
}

// Cooperatively issue every remote node's src token for one dst token into this warp's G2S ring.
// Each lane owns one remote node; valid lanes issue in parallel to distinct stages, batched by
// ring_len so parity resolves cleanly when valid remotes exceed the ring depth (RED consumes
// stages sequentially). Advances global_offset by the number of entries issued. Mirrors the
// LOCAL tier's issue_local_g2s_row.
template <
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int LSA_TEAMS,
    typename SMEM_TYPE>
__forceinline__ __device__ void issue_rdma_g2s_row(
    SMEM_TYPE* smem_buffer_ptr,
    const bool* attn_to_rdma_addr,
    const uint16_t* rdma_inter_node_group_token,
    const float* rdma_inter_node_group_prob,
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
                combine_g2s_resolve_rdma_source<BACKWARD_COMBINE, HIDDEN_DIM, kTokenDtype, MAX_NUM_OF_TOKENS_PER_RANK>(
                    rdma_inter_node_group_token,
                    rdma_inter_node_group_prob,
                    lane.tile_id,
                    flat_token_id,
                    experts_per_rank,
                    lteam_sz);
            // RDMA tier does not set inter_node_flag_G2S_buffer; the RED group reads
            // attn_to_rdma_map to demarcate RDMA entries.
            issue_g2s_entry</*INTER_NODE=*/true, BACKWARD_COMBINE, /*WRITE_LAST_FLAG=*/false>(
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

// Inter-node G2S warp group for the combine kernel.
template <
    typename SMEM_TYPE,
    typename G2S_GROUP,
    int STAGES_G2S,
    int TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    int NUM_OF_TOKENS_PER_GROUP,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SZ,
    ncclEpLayout_t kLayout,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void combine_G2S_inter_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const bool* attn_to_rdma_map,
    const int32_t* sparse_to_dense_map,
    uint16_t* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    const uint16_t* rdma_inter_node_group_token,
    const float* rdma_inter_node_group_prob,
    // OUTPUT
    uint64_t* rdma_inter_node_group_flags,
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
    // The G2S group is split into single-warp pipelines (warp == pipeline), matching the inter-node red
    // group; each warp owns an equal slice of the G2S FIFO.
    static_assert(
        STAGES_G2S % G2S_GROUP::warp_size() == 0,
        "STAGES_G2S must be a multiple of the inter-node G2S warp count.");
    constexpr int NUM_OF_STAGES_G2S_PER_WARP = STAGES_G2S / G2S_GROUP::warp_size();
    static_assert(
        TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0,
        "TOKENS_PER_CHUNK must be a multiple of NUM_OF_TOKENS_PER_GROUP.");
    constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // Produces the local rank's output chunks in order; RDMA feeds matching chunk IDs from
    // node-1, ..., node+1, so src chunks arrive in the order the RED group consumes them.
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int max_chunks_per_rank = nccl_ep::ceil_div(MAX_NUM_OF_TOKENS_PER_RANK, TOKENS_PER_CHUNK);
    const int total_chunks = cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per node.
    const int rdma_map_size_per_node = nccl_ep::align(num_of_tokens_per_rank, 16);
    // This warp's G2S FIFO sub-range.
    const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * G2S_GROUP::warp_rank();
    const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * (G2S_GROUP::warp_rank() + 1);

    // Unified body for NVLink-only (LSA_TEAMS==1) and RDMA+NVLink (>1).
    //   LOCAL tier: warp-cooperative s2d-row scan -> issue_local_g2s_row.
    //   RDMA tier (>1): chunk-level ncclGin signal pre-wait, warp-parallel per-node TMA issue, residue flags.
    // Parity: global_offset counts stages filled this warp; entry rank R lands at
    //   stage = starting + (global_offset+R) % ring_len, parity = 1 ^ (((global_offset+R)/ring_len) & 1),
    // matching RED's sequential consumption.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    const int ring_len = ending_G2S_index - starting_G2S_index;
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>()));
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
            is_tail ? nccl_ep::ceil_div(rem_chunk_sz, NUM_OF_TOKENS_PER_GROUP) : NUM_OF_TOKEN_GROUPS_PER_CHUNK;

        const bool* rdma_to_attn_map_base =
            rdma_to_attn_map + (my_lteam * rdma_map_size_per_node + cidx * TOKENS_PER_CHUNK);
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
                constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
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
            for (int token_in_group = G2S_GROUP::warp_rank(); token_in_group < NUM_OF_TOKENS_PER_GROUP;
                 token_in_group += G2S_GROUP::warp_size()) {
                int cur_tokid = group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group;
                if (cur_tokid >= csize) {
                    break;
                }

                // Uniform across the warp. No early continue: the RDMA tier still runs when false.
                bool token_needed_by_this_node = rdma_to_attn_map_base[cur_tokid];

                // LOCAL tier: warp-cooperative s2d-row scan with inline broadcast-issue. Advances
                // global_offset by the entries issued, keeping later reads consistent on every lane.
                if (token_needed_by_this_node) {
                    const int32_t* sparse_to_dense_row =
                        sparse_to_dense_map_base + (group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group) * s2d_entries;
                    issue_local_g2s_row</*INTER_NODE=*/true, BACKWARD_COMBINE, kLayout, HIDDEN_DIM, kTokenDtype>(
                        smem_buffer_ptr,
                        sparse_to_dense_row,
                        s2d_entries,
                        global_offset,
                        starting_G2S_index,
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

                // RDMA tier: each lane maps to a remote node; valid lanes issue TMAs in parallel to
                // distinct stages, batched by ring_len so parity resolves cleanly.
                if constexpr (LSA_TEAMS > 1) {
                    const int flat_token_id =
                        cidx * TOKENS_PER_CHUNK + group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group;
                    const bool* attn_to_rdma_addr =
                        attn_to_rdma_map_base + (group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group) * (LSA_TEAMS - 1);
                    issue_rdma_g2s_row<
                        BACKWARD_COMBINE,
                        HIDDEN_DIM,
                        kTokenDtype,
                        MAX_NUM_OF_TOKENS_PER_RANK,
                        LSA_TEAMS>(
                        smem_buffer_ptr,
                        attn_to_rdma_addr,
                        rdma_inter_node_group_token,
                        rdma_inter_node_group_prob,
                        flat_token_id,
                        global_offset,
                        starting_G2S_index,
                        ring_len,
                        lane_id,
                        my_lteam,
                        token_bytes,
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
                rdma_inter_node_group_flags + (lteam_id * max_chunks_per_rank + cpr);
            for (int flag_id = G2S_GROUP::thread_rank(); flag_id < residue_flag_count;
                 flag_id += G2S_GROUP::size()) {
                residue_flag_base[flag_id] = expected_flag_value;
            }
        }
    }
}

// Consume one inter-node G2S stage into the FP32 accumulators: token is accumulated; prob is written
// into node slot `prob_slot` (accumulated for the local phase, assigned for remote phases). Advances
// this pipeline's G2S cursor/parity. Returns the producer's last-src flag when READ_LAST_FLAG.
template <
    int NUM_OF_THREADS_PER_PIPELINE,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    bool READ_LAST_FLAG,
    bool ACCUMULATE_PROB,
    ncclDataType_t kTokenDtype,
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
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

    __nv_bfloat162* load_token_base_ptr =
        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_G2S(token_stage));
    float* load_prob_base_ptr;
    if constexpr (BACKWARD_COMBINE) {
        load_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_G2S(token_stage);
    }

    // Wait until this src token is staged in SMEM, then let the whole pipeline read it.
    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            while (!cuda::ptx::mbarrier_try_wait_parity(
                smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage), token_producer_parity)) {
            }
        }
    }
    arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

#pragma unroll
    for (int n = 0; n < NUM_ACC; n++) {
        int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
        if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
            float2 src_data_fp32 = nccl_ep::ld_token_pair<kTokenDtype>(load_token_base_ptr, element_id);
            acc_token_fp32[n].x += src_data_fp32.x;
            acc_token_fp32[n].y += src_data_fp32.y;
        }
    }
    if constexpr (BACKWARD_COMBINE) {
#pragma unroll
        for (int pv = 0; pv < prob_vec_per_thread; pv++) {
            int element_id = thread_rank_within_pipeline + pv * NUM_OF_THREADS_PER_PIPELINE;
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
        last_src_token = smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage];
    }

    // All pipeline threads finish reading before the producer reuses this stage.
    arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage));
        }
    }

    token_stage += 1;
    if (token_stage == ending_G2S_index) {
        token_stage = starting_G2S_index;
        token_producer_parity ^= 1;
    }
    return last_src_token;
}

// Store one reduced dst token (+per-node prob) from FP32 registers into an S2G SMEM stage and TMA-copy
// it to the attn output. Advances this pipeline's S2G cursor.
template <
    int NUM_OF_THREADS_PER_PIPELINE,
    int NUM_OF_STAGES_S2G_PER_PIPELINE,
    int LSA_TEAMS,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
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
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

    __nv_bfloat162* store_token_base_ptr =
        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_S2G(dst_token_stage));

    // Wait for prior TMA reads of this S2G stage to finish before overwriting it.
    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G_PER_PIPELINE - 1>{});
        }
    }
    arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

#pragma unroll
    for (int n = 0; n < NUM_ACC; n++) {
        int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
        if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
            nccl_ep::st_token_pair<kTokenDtype>(store_token_base_ptr, element_id, acc_token_fp32[n]);
        }
    }
    if constexpr (BACKWARD_COMBINE) {
        float* store_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_S2G(dst_token_stage);
        // Gather per-source-node prob into output node order (my_lteam, my_lteam-1, ...).
#pragma unroll
        for (int n = 0; n < LSA_TEAMS; n++) {
            int output_lteam_id = (my_lteam - n) >= 0 ? my_lteam - n : my_lteam + LSA_TEAMS - n;
            int element_base_id = output_lteam_id * prob_dim;
#pragma unroll
            for (int m = 0; m < prob_vec_per_thread; m++) {
                int element_id = thread_rank_within_pipeline + m * NUM_OF_THREADS_PER_PIPELINE;
                if (element_id < prob_dim) {
                    store_prob_base_ptr[element_base_id + element_id] = acc_prob_ptr[n * prob_vec_per_thread + m];
                }
            }
        }
    }

    // Publish SMEM writes to the async proxy, then sync the pipeline before the TMA launch.
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

    if (warp_rank_within_pipeline == 0) {
        if (cuda::ptx::elect_sync(~0) && absolute_token_id < num_real_tokens) {
            uint16_t* current_token_addr = attn_output_token_base + (size_t)token_in_chunk * out_token_stride_u16;
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_global,
                cuda::ptx::space_shared,
                reinterpret_cast<void*>(current_token_addr),
                reinterpret_cast<const void*>(smem_buffer_ptr->get_inter_node_token_S2G(dst_token_stage)),
                (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>())));

            if constexpr (BACKWARD_COMBINE) {
                float* current_prob_addr = attn_output_prob_base + token_in_chunk * (prob_dim * LSA_TEAMS);
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    reinterpret_cast<void*>(current_prob_addr),
                    reinterpret_cast<const void*>(smem_buffer_ptr->get_inter_node_prob_S2G(dst_token_stage)),
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

// Inter-node reduction warp group for the combine kernel.
template <
    typename SMEM_TYPE,
    typename RED_GROUP,
    int NUM_OF_DATA_PIPELINE_PER_BLOCK,
    int STAGES_G2S,
    int NUM_OF_STAGES_S2G,
    int TOKENS_PER_CHUNK,
    int LSA_TEAMS,
    int NBLOCKS,
    int NUM_OF_TOKENS_PER_GROUP,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SZ,
    ncclDataType_t kTokenDtype>
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
    // The warp group is split into NUM_OF_DATA_PIPELINE_PER_BLOCK independent pipelines (matching the
    // inter-node G2S group); each pipeline owns an equal slice of the G2S/S2G FIFOs.
    static_assert(
        RED_GROUP::warp_size() % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0,
        "Inter-node red warp count must be a multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_OF_THREADS_PER_PIPELINE =
        (RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK) * WARP_SIZE;
    static_assert(
        STAGES_G2S % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0,
        "STAGES_G2S must be a multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
    constexpr int NUM_OF_STAGES_G2S_PER_PIPELINE = STAGES_G2S / NUM_OF_DATA_PIPELINE_PER_BLOCK;
    static_assert(
        NUM_OF_STAGES_S2G % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0,
        "NUM_OF_STAGES_S2G must be a multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
    constexpr int NUM_OF_STAGES_S2G_PER_PIPELINE = NUM_OF_STAGES_S2G / NUM_OF_DATA_PIPELINE_PER_BLOCK;
    // Output chunks are split into token groups striped across blocks (unlike the chunk-per-block stages).
    static_assert(
        TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0,
        "TOKENS_PER_CHUNK must be a multiple of NUM_OF_TOKENS_PER_GROUP.");
    constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // Tokens reduced as BF16x2 in FP32 (HIDDEN_DIM even); prob stays in float.
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
    constexpr int NUM_OF_ACC_ELEMENTS_PER_THREAD =
        nccl_ep::ceil_div(NUM_OF_BF16X2_ELEMENTS_PER_TOKEN, NUM_OF_THREADS_PER_PIPELINE);
    const int prob_dim = experts_per_rank * LSA_TEAM_SZ;
    const int prob_vec_per_thread = nccl_ep::ceil_div(prob_dim, NUM_OF_THREADS_PER_PIPELINE);
    // Compile-time upper bound sized exactly to this instantiation's LSA team.
    constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
        nccl_ep::ceil_div(NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SZ, NUM_OF_THREADS_PER_PIPELINE);

    // Each block produces token groups of the local rank's output chunks in order; the RDMA network feeds
    // matching chunk IDs from node-1, node-2, ..., node+1, so src chunks arrive in the same order.
    const int rem_chunk_sz = num_of_tokens_per_rank % TOKENS_PER_CHUNK;
    const int cpr = nccl_ep::ceil_div(num_of_tokens_per_rank, TOKENS_PER_CHUNK);
    const int total_chunks = cpr;
    // rdma_to_attn_map is padded to 16B (16 bools) per node.
    const int rdma_map_size_per_node = nccl_ep::align(num_of_tokens_per_rank, 16);
    // Dtype-aware per-token output stride in uint16_t units.
    const size_t out_token_stride_u16 = (size_t)HIDDEN_DIM * nccl_ep::size_u8<kTokenDtype>() / sizeof(uint16_t);

    // This thread's pipeline placement and G2S/S2G FIFO sub-range.
    const int pipeline_rank = RED_GROUP::thread_rank() / NUM_OF_THREADS_PER_PIPELINE;
    const int thread_rank_within_pipeline = RED_GROUP::thread_rank() % NUM_OF_THREADS_PER_PIPELINE;
    const int warp_rank_within_pipeline = thread_rank_within_pipeline / WARP_SIZE;
    const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * pipeline_rank;
    const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * (pipeline_rank + 1);
    int token_stage = starting_G2S_index;
    uint32_t token_producer_parity = 0;
    const int starting_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * pipeline_rank;
    const int ending_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * (pipeline_rank + 1);
    int dst_token_stage = starting_S2G_index;

    for (int cidx = 0; cidx < total_chunks; cidx++) {
        const bool is_tail = (rem_chunk_sz != 0 && cidx == cpr - 1);
        const int csize = is_tail ? rem_chunk_sz : TOKENS_PER_CHUNK;
        const int token_groups_for_chunk =
            is_tail ? nccl_ep::ceil_div(rem_chunk_sz, NUM_OF_TOKENS_PER_GROUP) : NUM_OF_TOKEN_GROUPS_PER_CHUNK;

        const bool* rdma_to_attn_map_base =
            rdma_to_attn_map + (my_lteam * rdma_map_size_per_node + cidx * TOKENS_PER_CHUNK);
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
            for (int token_in_group = pipeline_rank; token_in_group < NUM_OF_TOKENS_PER_GROUP;
                 token_in_group += NUM_OF_DATA_PIPELINE_PER_BLOCK) {
                int cur_tokid = group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group;
                if (cur_tokid >= csize) {
                    break;
                }

                // Each dst token accumulates local-node src tokens (like intra reduction) then remote-node
                // RDMA src tokens. Prob is gathered per source node: slot 0 = local, 1..LSA-1 = remote.
                float2 acc_token_fp32[NUM_OF_ACC_ELEMENTS_PER_THREAD];
                using acc_prob_storage_type =
                    acc_prob_storage_t<BACKWARD_COMBINE, LSA_TEAMS * MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD>;
                [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
                [[maybe_unused]] float* acc_prob_ptr = nullptr;
                if constexpr (BACKWARD_COMBINE) {
                    acc_prob_ptr = acc_prob_storage.data;
                }
#pragma unroll
                for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD; n++) {
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

                // Local-node accumulation: consume staged src tokens until the producer marks the last one.
                if (rdma_to_attn_map_base[cur_tokid]) {
                    bool last_local_src = false;
                    do {
                        last_local_src = combine_inter_consume_src<
                            NUM_OF_THREADS_PER_PIPELINE,
                            BACKWARD_COMBINE,
                            HIDDEN_DIM,
                            /*READ_LAST_FLAG=*/true,
                            /*ACCUMULATE_PROB=*/true,
                            kTokenDtype>(
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

                // Remote-node accumulation: at most one src token per remote node (node-1, node-2, ..., node+1).
                if constexpr (LSA_TEAMS > 1) {
                    const bool* attn_to_rdma_addr = attn_to_rdma_map_base + cur_tokid * (LSA_TEAMS - 1);
#pragma unroll
                    for (int n = 1; n < LSA_TEAMS; n++) {
                        int lteam_id = my_lteam >= n ? my_lteam - n : my_lteam + LSA_TEAMS - n;
                        int rdma_buffer_tile_id = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
                        if (attn_to_rdma_addr[rdma_buffer_tile_id]) {
                            combine_inter_consume_src<
                                NUM_OF_THREADS_PER_PIPELINE,
                                BACKWARD_COMBINE,
                                HIDDEN_DIM,
                                /*READ_LAST_FLAG=*/false,
                                /*ACCUMULATE_PROB=*/false,
                                kTokenDtype>(
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
                    NUM_OF_THREADS_PER_PIPELINE,
                    NUM_OF_STAGES_S2G_PER_PIPELINE,
                    LSA_TEAMS,
                    BACKWARD_COMBINE,
                    HIDDEN_DIM,
                    kTokenDtype>(
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
                    /*token_in_chunk=*/group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group,
                    /*absolute_token_id=*/cidx * TOKENS_PER_CHUNK +
                        group_idx * NUM_OF_TOKENS_PER_GROUP + token_in_group,
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
    const int32_t* __restrict__ actual_counts,
    const int64_t* __restrict__ zone_offsets,
    const int experts_per_rank,
    const int alignment,
    const int hidden_dim,
    const int num_blocks,
    SMEM_TYPE* smem) {
    // Caller zeroes alignment when not expert-major; alignment<=1 ⇒ no padding work.
    if (alignment <= 1 || local_buf == nullptr || actual_counts == nullptr || zone_offsets == nullptr) return;

    const int lane = PAD_GROUP::thread_rank();
    constexpr int warp_size = 32;
    const uint32_t token_bytes = static_cast<uint32_t>(hidden_dim * sizeof(TOKEN_DATA_TYPE));

    // Cooperatively zero the SMEM staging slot once per kernel invocation.
    auto* smem_u4 = reinterpret_cast<uint4*>(smem->get_pad_tma_slot());
    const int vec_n = token_bytes / sizeof(uint4);
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

// Inter-node RDMA (GIN) cross-round WAR guard, warp-collective (lane i -> peer i), relaxed (WAR only).

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
__device__ __forceinline__ void warp_rdma_guard_publish(
    ncclDevComm dcomm,
    ncclWindow_t dest_window,
    size_t my_slot,
    int my_lteam,
    int lsa_teams,
    uint64_t expected) {
    ncclGin net(dcomm, /*contextIndex=*/0, NCCL_GIN_RESOURCE_SHARING_THREAD);
    ncclTeam rail = ncclTeamRail(dcomm);
    for (int peer = (threadIdx.x & 31); peer < lsa_teams; peer += 32) {
        if (peer == my_lteam) continue;
        net.putValue(rail, peer, dest_window, my_slot, expected, ncclGin_None{}, ncclCoopThread());
    }
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
    ncclEpDispatchQuantizationRecipe_t kQuantizationRecipe,
    typename GIN_GROUP,
    typename LSA_G2S_GROUP,
    typename LSA_S2G_GROUP,
    typename PAD_GROUP,
    int NUM_OF_STAGES,
    int NUM_OF_IN_FLIGHT_S2G,
    int TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int LSA_TEAMS,
    int NBLOCKS,
    bool FORWARD_DISPATCH,
    int NUM_PIPELINES,
    int LSA_TEAM_SZ,
    ncclEpLayout_t kLayout,
    int HIDDEN_DIM,
    int SF_BYTES_PER_TOKEN>
__device__ __forceinline__ void dispatch_kernel_impl(
    const dispatch_kernel_param_t<TOKEN_DATA_TYPE, LSA_TEAM_SZ>& param,
    uint8_t* smem_bytes) {
    static_assert(kQuantizationRecipe == NCCL_EP_DISPATCH_QUANT_NONE ||
                      kQuantizationRecipe == NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD,
                  "unsupported dispatch quantization recipe");
    const int my_lteam = param.node_rank;
    if constexpr (LSA_TEAMS != 1) {
        static_assert(
            GIN_GROUP::size() % 32 == 0 && GIN_GROUP::size() <= 64,
            "Dispatch kernel supports 1 or 2 N2N warps.");
    }
    static_assert(NUM_OF_STAGES % NUM_PIPELINES == 0, "NUM_OF_STAGES must be divisible by NUM_PIPELINES.");
    constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;

    using cur_smem_t = dispatch_smem_layout_t;

    cur_smem_t smem_layout;
    dispatch_config_t d_config;
    model_config_t d_model;
    d_config.num_of_stages = NUM_OF_STAGES;
    d_config.num_of_in_flight_s2g = NUM_OF_IN_FLIGHT_S2G;
    d_config.num_of_tokens_per_chunk = TOKENS_PER_CHUNK;
    d_config.num_of_blocks = NBLOCKS;
    d_config.forward_dispatch = FORWARD_DISPATCH;
    d_config.sf_bytes_per_token = SF_BYTES_PER_TOKEN;
    d_config.num_pipelines = NUM_PIPELINES;
    d_config.stages_per_pipeline = STAGES_PER_PIPELINE;
    d_config.s2d_inner_dim = param.s2d_inner_dim;
    d_model.hidden_dim = HIDDEN_DIM;
    d_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    d_model.num_of_experts_per_rank = param.experts_per_rank;
    d_model.ranks_per_lsa_team = param.ranks_per_lsa_team;
    d_model.num_of_nodes = LSA_TEAMS;
    create_dispatch_smem_layout<kLayout, sizeof(TOKEN_DATA_TYPE)>(smem_layout, smem_bytes, d_config, d_model);
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    using head_init_warp = warp_group<1, 0>;
    using head_rdma_warp = warp_group<1, 1>;
    using head_lsa_warp = warp_group<1, 2>;
    static_assert(
        GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size() + PAD_GROUP::size() >=
            3 * 32,
        "dispatch head needs 3 warps");
    const int head_tid = (int)threadIdx.x;
    if (head_tid < head_init_warp::size()) {
        // warp 0: per-pipeline mbarrier init (both producer/consumer arrival counts = 1).
        if (head_tid == 0) {
            for (int p = 0; p < NUM_PIPELINES; p++) {
                for (int s = 0; s < STAGES_PER_PIPELINE; s++) {
                    int abs_stage = p * STAGES_PER_PIPELINE + s;
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_buffer + 2 * abs_stage, 1);
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_buffer + 2 * abs_stage + 1, 1);
                }
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_s2d_map_mbar(p, 0), 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_s2d_map_mbar(p, 1), 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_S2G_group_mbar(p), 1);
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size()) {
        // warp 1: inter-node RDMA guard wait.
        if constexpr (LSA_TEAMS != 1) {
            if (param.guard_enabled)
                warp_rdma_guard_wait(
                    reinterpret_cast<const uint64_t*>(
                        reinterpret_cast<const uint8_t*>(param.gin_base_ptr) + param.mr_info.guard_offset),
                    my_lteam,
                    LSA_TEAMS,
                    *param.expected_rdma_flag_value);
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size() + head_lsa_warp::size()) {
        // warp 2: intra-node LSA barrier.
        if constexpr (LSA_TEAM_SZ != 1) {
            if (param.guard_enabled) {
                ncclLsaBarrierSession<ncclCoopWarp> bar(
                    ncclCoopWarp(),
                    param.dcomm,
                    ncclTeamTagLsa(),
                    (uint32_t)blockIdx.x);
                bar.sync(ncclCoopWarp(), cuda::memory_order_relaxed);
            }
        }
    }

    __syncthreads();

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    long long _wt_start = 0;
    if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif
    constexpr bool HAS_SF = (kQuantizationRecipe == NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD);
    int threadIdx_x_int = (int)threadIdx.x;
    if (threadIdx_x_int < GIN_GROUP::size()) {
        if constexpr (LSA_TEAMS != 1) {
#define DISPATCH_N2N_TEMPLATE \
            dispatch_N2N_warp<GIN_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, TOKENS_PER_CHUNK, \
                              MAX_NUM_OF_TOKENS_PER_RANK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, FORWARD_DISPATCH, HAS_SF>
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
#define DISPATCH_G2S_TEMPLATE \
        dispatch_G2S_warp<LSA_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, TOKENS_PER_CHUNK, \
                            MAX_NUM_OF_TOKENS_PER_RANK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, NUM_PIPELINES, \
                            FORWARD_DISPATCH, HAS_SF>
        DISPATCH_G2S_TEMPLATE(
            param.rdma_to_attn_map,
            param.attn_input_token,
            param.attn_input_prob,
            param.attn_input_token_scaling_factor,
            param.rdma_inter_node_group_flags,
            param.local_rank,
            my_lteam,
            param.num_of_tokens_per_rank,
            HIDDEN_DIM,
            SF_BYTES_PER_TOKEN,
            param.experts_per_rank,
            *param.expected_rdma_flag_value,
            param.dcomm,
            param.num_ctx_per_comm,
            param.gin_base_ptr,
            &param.mr_info,
            smem_buffer_ptr);
#undef DISPATCH_G2S_TEMPLATE
    } else if (
        threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size()) {
#define DISPATCH_S2G_TEMPLATE \
        dispatch_S2G_warp<LSA_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, \
                            NUM_OF_IN_FLIGHT_S2G, TOKENS_PER_CHUNK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, NUM_PIPELINES, \
                            FORWARD_DISPATCH, HAS_SF, kLayout>
        DISPATCH_S2G_TEMPLATE(
            param.rdma_to_attn_map,
            param.sparse_to_dense_map,
            param.expert_output_token,
            param.expert_output_prob,
            param.expert_output_scaling_factor,
            my_lteam,
            param.num_of_tokens_per_rank,
            HIDDEN_DIM,
            SF_BYTES_PER_TOKEN,
            param.experts_per_rank,
            param.local_dup_enabled,
            param.max_recv_tokens_per_rank,
            smem_buffer_ptr);
#undef DISPATCH_S2G_TEMPLATE
    } else if (
        PAD_GROUP::size() > 0 && threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size() +
                                                       LSA_S2G_GROUP::size() + PAD_GROUP::size()) {
        // PAD warp: zero-init expert-major alignment padding slots concurrently with S2G.
        // No barrier needed against S2G — padding rows live past the actual token rows
        // in each expert's zone, so the two warps target disjoint global memory.
        PAD_warp_group_device_function<PAD_GROUP, TOKEN_DATA_TYPE>(
            param.expert_output_token[param.local_rank],
            param.pad_actual_counts,
            param.pad_expert_token_offsets,
            param.experts_per_rank,
            param.pad_alignment,
            HIDDEN_DIM,
            NBLOCKS,
            smem_buffer_ptr);
    }
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    if (threadIdx.x % 32 == 0) {
        constexpr int _WT_WARPS = (GIN_GROUP::size() + LSA_G2S_GROUP::size() +
                                   LSA_S2G_GROUP::size() + PAD_GROUP::size()) /
                                  32;
        int _warp_id = threadIdx.x / 32;
        int _idx = blockIdx.x * _WT_WARPS + _warp_id;
        param.warp_timing[_idx].start_clock = _wt_start;
        param.warp_timing[_idx].end_clock = clock64();
    }
#endif

    // ===== FUSED DEVICE SYNC (dispatch tail) =====
    if (elect_last_block(reinterpret_cast<const int*>(param.dispatch_grid_barrier_counter), NBLOCKS)) {
        using tail_completion_warp = warp_group<1, 0>; // warp 0
        using tail_rdma_warp = warp_group<1, 1>; // warp 1
        const int tail_tid = (int)threadIdx.x;
        if (tail_tid < tail_completion_warp::size()) {
            // warp 0 (thread 0): inter-rank completion barrier, then reset + bump the intra-node round
            // (local_dup defers that bump to its own tail).
            if (tail_tid == 0) {
                const uint32_t expected_val = *param.expected_intra_node_flag_value;
                nccl_ep::red_add_release_sys_global(param.intra_node_write_completion_flags, 1u);
                uint32_t flag_data;
                do {
                    flag_data = nccl_ep::ld_relaxed_sys_global(param.intra_node_write_completion_flags);
                } while (flag_data != expected_val);
                nccl_ep::memory_fence();
                atomicExch((unsigned int*)param.dispatch_grid_barrier_counter, 0u);
                if (!param.local_dup_enabled)
                    *param.expected_intra_node_flag_value += static_cast<uint32_t>(param.ranks_per_lsa_team);
            }
        } else if (tail_tid < tail_completion_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the inter-node RDMA guard + bump the RDMA round.
            if constexpr (LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_rdma_flag_value;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        param.dcomm,
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(my_lteam) * sizeof(uint64_t),
                        my_lteam,
                        LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_rdma_flag_value = expected + 1ull;
            }
        }
    }
}

template < // This type represent intra-node reduction warp group.
  typename LSA_RED_GROUP,
  // This type represent inter-node reduction warp group.
  typename CROSS_LSA_RED_GROUP,
  // This type represent intra-node G2S warp group.
  typename LSA_G2S_GROUP,
  // This type represent inter-node G2S warp group.
  typename CROSS_LSA_G2S_GROUP,
  // This type represent inter-node rdma warp group.
  typename GIN_GROUP,
  // Number of independent data pipeline per CUDA block.
  int NUM_OF_DATA_PIPELINE_PER_BLOCK,
  // Number of token entry in the shared memory for G2S operations.
  int STAGES_G2S,
  // Number of token entry in the shared memory for S2G operations.
  int NUM_OF_STAGES_S2G,
  // Number of token per group in the inter-node reduction/G2S warp group.
  int NUM_OF_TOKENS_PER_GROUP,
  // Size of each chunk.
  int TOKENS_PER_CHUNK,
  // Model configuration.
  int MAX_NUM_OF_TOKENS_PER_RANK, int LSA_TEAMS,
  // Number of CUDA block running dispatch kernel.
  int NBLOCKS,
  // Number of fully in-flight S2G in intra-node reduction warp group.
  int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
  // Whether the combine kernel is used in backward process. If so, need to transfer the prob for each token as well.
  bool BACKWARD_COMBINE, int HIDDEN_DIM, int LSA_TEAM_SZ, ncclEpLayout_t kLayout,
  // NONE output dtype, resolved at compile time (JIT literal) so the per-element
  // reduction branches fold away — BF16 (default) pays zero dtype-branch cost.
  ncclDataType_t kTokenDtype = ncclBfloat16>
// Each CUDA block of combine kernel has named warp groups:
// intra/inter reduction, intra/inter G2S, and inter-node N2N RDMA. Group sizes are
// set by the HT combine warp-count constants and the selected pipeline count.
__device__ __forceinline__ void combine_kernel_impl(const combine_kernel_param_t<LSA_TEAM_SZ>& param,
                                                    uint8_t* smem_bytes) {
    const int my_lteam = param.node_rank;
    // Compile-time check (only enforce for multi-node layout).
    if constexpr (LSA_TEAMS != 1) {
        static_assert(
            LSA_G2S_GROUP::size() == 32,
            "Combine kernel only support 1 INTRA_NODE_G2S warp currently.");
        static_assert(
            CROSS_LSA_G2S_GROUP::size() == 32,
            "Combine kernel only support 1 INTER_NODE_G2S warp currently.");
    }
    // The token and its properties should meet size and alignment requirement.
    // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
    // We need to add padding or not using TMA for prob, if we want to support other scenario.
    // assert((param.experts_per_rank * param.ranks_per_lsa_team * sizeof(float)) % 16 == 0);
    static_assert((HIDDEN_DIM % 2) == 0, "HIDDEN_DIM must be even for BF16x2.");
    static_assert((HIDDEN_DIM * sizeof(uint16_t)) % 16 == 0, "HIDDEN_DIM must satisfy TMA alignment.");
    static_assert(
        MAX_NUM_OF_TOKENS_PER_RANK % TOKENS_PER_CHUNK == 0,
        "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of TOKENS_PER_CHUNK.");
    constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
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
    c_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    c_model.num_of_experts_per_rank = param.experts_per_rank;
    c_model.ranks_per_lsa_team = param.ranks_per_lsa_team;
    c_model.num_of_nodes = LSA_TEAMS;
    // Layout derives the element width from kTokenDtype (FP32 doubles the per-stage
    // token-buffer bytes vs BF16/FP16).
    create_combine_smem_layout<kTokenDtype>(
        smem_layout,
        smem_bytes,
        STAGES_G2S,
        NUM_OF_STAGES_S2G,
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
            // Inter-rank completion barrier: block 0 signals (red.release orders prior stores), every block polls.
            if (blockIdx.x == 0) nccl_ep::red_add_release_sys_global(param.intra_node_write_completion_flags, 1u);
            const uint32_t expected_val = *param.expected_intra_node_flag_value;
            uint32_t flag_data;
            do {
                flag_data = nccl_ep::ld_relaxed_sys_global(param.intra_node_write_completion_flags);
            } while (flag_data != expected_val);
            nccl_ep::memory_fence();
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
            _wt_head_end = clock64();
            param.block_timing[blockIdx.x].head_sync_start_clock = _wt_head_start;
            param.block_timing[blockIdx.x].head_sync_end_clock = _wt_head_end;
#endif
            // mbarrier init (both producer/consumer arrival counts = 1).
            for (int i = 0; i < STAGES_G2S; i++) {
                if constexpr (LSA_TEAMS != 1) {
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_G2S_buffer + 2 * i, 1);
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_G2S_buffer + 2 * i + 1, 1);
                }
                cuda::ptx::mbarrier_init(smem_buffer_ptr->inter_node_mbarrier_G2S_buffer + 2 * i, 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->inter_node_mbarrier_G2S_buffer + 2 * i + 1, 1);
            }
            if constexpr (LSA_TEAMS != 1) {
                for (int i = 0; i < LSA_TEAMS - 1; i++)
                    for (int j = 0; j < MAX_NUM_OF_CHUNKS_PER_RANK; j++)
                        cuda::ptx::mbarrier_init(
                            smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer + i * MAX_NUM_OF_CHUNKS_PER_RANK + j,
                            1);
                *(smem_buffer_ptr->rdma_streaming_counter) = 0u;
            }
            cuda::ptx::fence_proxy_async(
                cuda::ptx::space_shared); // make mbarrier init visible to the async (TMA) proxy
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size()) {
        // warp 1: inter-node RDMA guard wait.
        if constexpr (LSA_TEAMS != 1) {
            if (param.guard_enabled)
                warp_rdma_guard_wait(
                    reinterpret_cast<const uint64_t*>(
                        reinterpret_cast<const uint8_t*>(param.gin_base_ptr) + param.mr_info.guard_offset),
                    my_lteam,
                    LSA_TEAMS,
                    *param.expected_rdma_flag_value);
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
            // Intra-node reduction warp group.
#define COMBINE_RED_INTRA_TEMPLATE \
            combine_RED_intra_warp<LSA_RED_GROUP, cur_smem_t, STAGES_G2S, NUM_OF_STAGES_S2G, \
                                   TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, LSA_TEAMS, NBLOCKS, \
                                   NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE, HIDDEN_DIM, LSA_TEAM_SZ, \
                                   kTokenDtype>
            COMBINE_RED_INTRA_TEMPLATE(
                // INPUT
                param.rdma_to_attn_map,
                // OUTPUT
                param.rdma_intra_node_red_token,
                param.rdma_intra_node_red_prob,
                // CONFIG
                my_lteam,
                param.num_of_tokens_per_rank,
                param.experts_per_rank,
                smem_buffer_ptr);
#undef COMBINE_RED_INTRA_TEMPLATE
        }
    } else if (threadIdx_x_int < LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size()) {
        // Inter-node reduction warp group.
#define COMBINE_RED_INTER_TEMPLATE \
        combine_RED_inter_warp<cur_smem_t, CROSS_LSA_RED_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, STAGES_G2S, \
                                NUM_OF_STAGES_S2G, TOKENS_PER_CHUNK, LSA_TEAMS, NBLOCKS, NUM_OF_TOKENS_PER_GROUP, \
                                BACKWARD_COMBINE, HIDDEN_DIM, LSA_TEAM_SZ, kTokenDtype>
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
        // Intra-node G2S warp group.
        if constexpr (LSA_TEAMS != 1) {
#define COMBINE_G2S_INTRA_TEMPLATE \
            combine_G2S_intra_warp<cur_smem_t, STAGES_G2S, TOKENS_PER_CHUNK, LSA_TEAMS, \
                                   LSA_TEAM_SZ, NBLOCKS, BACKWARD_COMBINE, HIDDEN_DIM, kLayout, kTokenDtype>
            COMBINE_G2S_INTRA_TEMPLATE(
                param.rdma_to_attn_map,
                param.sparse_to_dense_map,
                param.expert_input_token,
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
        // Inter-node G2S warp group.
#define COMBINE_G2S_INTER_TEMPLATE \
        combine_G2S_inter_warp<cur_smem_t, CROSS_LSA_G2S_GROUP, STAGES_G2S, TOKENS_PER_CHUNK, \
                                MAX_NUM_OF_TOKENS_PER_RANK, LSA_TEAMS, NBLOCKS, NUM_OF_TOKENS_PER_GROUP, \
                                BACKWARD_COMBINE, HIDDEN_DIM, LSA_TEAM_SZ, kLayout, kTokenDtype>
        COMBINE_G2S_INTER_TEMPLATE(
            // INPUT
            param.rdma_to_attn_map,
            param.attn_to_rdma_map,
            param.sparse_to_dense_map,
            param.expert_input_token,
            param.expert_input_prob,
            param.rdma_inter_node_group_token,
            param.rdma_inter_node_group_prob,
            // OUTPUT
            param.rdma_inter_node_group_flags,
            smem_buffer_ptr,
            // CONFIG
            param.local_rank,
            my_lteam,
            param.num_of_tokens_per_rank,
            param.experts_per_rank,
            *param.expected_rdma_flag_value,
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
        // Inter-node rdma warp group.
        if constexpr (LSA_TEAMS != 1) {
#define COMBINE_N2N_INTER_TEMPLATE \
            combine_N2N_inter_warp<GIN_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, TOKENS_PER_CHUNK, \
                                   MAX_NUM_OF_TOKENS_PER_RANK, LSA_TEAMS, NBLOCKS, LSA_TEAM_SZ, BACKWARD_COMBINE, \
                                   HIDDEN_DIM, kTokenDtype>
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
            // warp 0: reset the grid counter + bump the intra-node round.
            if (tail_tid == 0) {
                atomicExch((unsigned int*)param.combine_grid_barrier_counter, 0u);
                *param.expected_intra_node_flag_value += static_cast<uint32_t>(param.ranks_per_lsa_team);
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the inter-node RDMA guard, then bump the RDMA round.
            if constexpr (LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_rdma_flag_value;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        param.dcomms[0],
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(my_lteam) * sizeof(uint64_t),
                        my_lteam,
                        LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_rdma_flag_value = expected + 1ull;
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size() + tail_lsa_warp::size()) {
            // warp 2: intra-node LSA WAR barrier. Relaxed -- the tail __syncthreads already drained this
            // round's reads. Index = dispatch's block count (disjoint from dispatch's per-block [0, NB)).
            if constexpr (LSA_TEAM_SZ != 1) {
                if (param.guard_enabled) {
                    ncclLsaBarrierSession<ncclCoopWarp> bar(
                        ncclCoopWarp(),
                        param.dcomms[0],
                        ncclTeamTagLsa(),
                        (uint32_t)NCCL_EP_HT_DISPATCH_NUM_OF_BLOCKS);
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
    const uint32_t* intra_node_write_completion_flag;
    // Shared with dispatch. When local_dup_enabled, dispatch defers the bump
    // here so peers only observe the flag move after secondaries are filled.
    uint32_t* expected_intra_node_flag_value;
    // Reused from dispatch_grid_barrier_counter (dispatch leaves it at 0).
    uint32_t* grid_barrier_counter;
    int experts_per_rank;
    int ranks_per_lsa_team;
};

// Dynamic shared-memory bytes required by local_dup_kernel_impl for the given
// hidden_dim and token element size (token dtype is BF16/uint16_t).
inline int local_dup_dynamic_smem_bytes(
    int hidden_dim,
    int pipe_depth,
    bool forward_dispatch,
    int experts_per_rank,
    int ranks_per_lsa_team,
    size_t token_elem_bytes) {
    const int token_bytes = hidden_dim * static_cast<int>(token_elem_bytes);
    const int prob_bytes =
        forward_dispatch ? experts_per_rank * ranks_per_lsa_team * static_cast<int>(sizeof(float)) : 0;
    const int rings = pipe_depth * (token_bytes + prob_bytes);
    const int mbar_bytes = pipe_depth * 2 * static_cast<int>(sizeof(uint64_t)) + 8;
    return rings + mbar_bytes;
}

// TODO: FP8 token duplication is not yet supported.
template <typename T, int HIDDEN_DIM, int PIPE_DEPTH, bool FORWARD_DISPATCH>
__device__ __forceinline__ void local_dup_kernel_impl(const local_dup_kernel_param_t<T>& p) {
    // Wait until all peers have signaled S2G completion on this rank's recv buffer.
    // Use >= rather than == so a future code path that overshoots the counter
    // (e.g. extra peer arrivals) doesn't hang.
    if (threadIdx.x == 0) {
        const uint32_t expected_val = *p.expected_intra_node_flag_value;
        uint32_t v;
        do {
            v = nccl_ep::ld_relaxed_sys_global(p.intra_node_write_completion_flag);
        } while (v < expected_val);
        nccl_ep::memory_fence();
    }
    __syncthreads();

    constexpr int kTokenBytes = HIDDEN_DIM * sizeof(T);
    const int prob_floats = FORWARD_DISPATCH ? (p.experts_per_rank * p.ranks_per_lsa_team) : 0;
    const int prob_bytes = prob_floats * static_cast<int>(sizeof(float));

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* smem_ptr = smem_raw;
    T* smem_token[PIPE_DEPTH];
    float* smem_prob[PIPE_DEPTH];
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
                *p.expected_intra_node_flag_value += static_cast<uint32_t>(p.ranks_per_lsa_team);
            }
        }
        return;
    }

    const int block_id = blockIdx.x;
    const int n_blocks = gridDim.x;
    const int group_stride = p.emuf_group_stride;
    const uint32_t total_tx = static_cast<uint32_t>(kTokenBytes + prob_bytes);

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
        int stage = 0;
        uint32_t producer_parity = 0;
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
                }
                cuda::ptx::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<PIPE_DEPTH - 1>{});
                cuda::ptx::mbarrier_arrive(&cons_mbar[stage]);
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
            *p.expected_intra_node_flag_value += static_cast<uint32_t>(p.ranks_per_lsa_team);
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
// kLocalPermuteMaxExpertsPerRank bounds the per-warp smem active-EM list.
constexpr int kLocalPermuteMaxExpertsPerRank = 256;
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

struct local_permute_dup_param_t {
    void* recv_x_em;
    float* recv_topk_weights_em;
    const void* flat_staging;
    const float* recv_topk_weights_flat;
    const int32_t* flat2em_slot_map;
    const int32_t* num_recv_tokens_dev;
    const int64_t* expert_token_offsets;
    const int32_t* per_expert_counts_active;
    int top_k;
    int experts_per_rank;
    int row_bytes;
    int caller_num_recv_tokens;   // caller recv buffer row capacity
};

template <int HiddenInt4, int HiddenVec>
__device__ __forceinline__ void local_permute_dup(
    uint8_t* __restrict__ recv_x_em,
    float* __restrict__ recv_topk_weights_em,
    const uint8_t* __restrict__ flat_staging,
    const float* __restrict__ recv_topk_weights_flat,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const int64_t* __restrict__ expert_token_offsets,
    const int32_t* __restrict__ per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int /*row_bytes*/,
    int caller_num_recv_tokens) {
    constexpr int kThreadsPerSlot = kLocalPermuteThreadsPerSlot;
    constexpr int kHiddenVec = HiddenVec;
    constexpr int kPermuteWarps = kLocalPermutePermuteWarps;
    constexpr int kPadWarps = kLocalPermutePadWarps;
    // Per-warp cap on active EM slots; bounded by top_k and experts_per_rank.
    constexpr int kMaxActivePerToken = kLocalPermuteMaxExpertsPerRank;

    const int warp_id = threadIdx.x / kThreadsPerSlot;
    const int lane = threadIdx.x & (kThreadsPerSlot - 1);
    const bool is_pad = warp_id >= kPermuteWarps;
    const int pad_idx = warp_id - kPermuteWarps;

    const int num_recv = *num_recv_tokens_dev;
    // Caller recv buffers must hold the full padded EM total. Host checks
    // enforce this per mode (budget or actual rows); this is the backstop.
    const int64_t em_padded_total = expert_token_offsets[experts_per_rank];
    EP_DEVICE_ASSERT(caller_num_recv_tokens >= em_padded_total);
    constexpr int row_int4 = HiddenInt4;
    constexpr int row_bytes = HiddenInt4 * 16;

    const int4* src_int4 = reinterpret_cast<const int4*>(flat_staging);
    int4* dst_int4 = reinterpret_cast<int4*>(recv_x_em);

    __shared__ int32_t s_active[kPermuteWarps][kMaxActivePerToken];
    __shared__ int s_count[kPermuteWarps];
    // Source row for the pad warp's cp.async.bulk S2G.
    extern __shared__ int4 s_zero[];

    for (int j = threadIdx.x; j < row_int4; j += blockDim.x) {
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
        for (int e = 0; e < experts_per_rank; ++e) {
            const int64_t zone_start = expert_token_offsets[e];
            const int64_t zone_end = expert_token_offsets[e + 1];
            const int32_t active = per_expert_counts_active[e];
            const int64_t pad_begin = zone_start + active;
            const int64_t pad_count = zone_end - pad_begin;
            for (int64_t offs = my_pad_lane; offs < pad_count; offs += total_pad_lanes) {
                const int64_t slot = pad_begin + offs;
                uint8_t* dst_g = recv_x_em + static_cast<size_t>(slot) * row_bytes;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst_g,
                    reinterpret_cast<uint8_t*>(s_zero),
                    row_bytes);
                if (zero_weights) {
                    recv_topk_weights_em[slot] = 0.0f;
                }
            }
        }
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        nccl_ep::fence_proxy_async();
    } else {
        // One warp per token, grid-strided over tokens.
        for (int blk = static_cast<int>(blockIdx.x) * kPermuteWarps; blk < num_recv;
             blk += kPermuteWarps * static_cast<int>(gridDim.x)) {
            const int token = blk + warp_id;
            if (token >= num_recv) continue;

            // Lane 0 packs active em_slots into smem and folds in the
            // topk-weights copy (one scalar store per slot).
            if (lane == 0) {
                const int32_t* slot_row = flat2em_slot_map + static_cast<size_t>(token) * top_k;
                int c = 0;
                const bool copy_weights = recv_topk_weights_em != nullptr && recv_topk_weights_flat != nullptr;
                for (int k = 0; k < top_k; ++k) {
                    const int32_t es = __ldg(slot_row + k);
                    if (es < 0) continue;
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
                    int4* dst = dst_int4 + static_cast<size_t>(s_active[warp_id][a]) * row_int4;
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
                        int4* dst = dst_int4 + static_cast<size_t>(s_active[warp_id][a]) * row_int4;
                        nccl_ep::st_cg_global(&dst[j], v);
                    }
                }
            }
            __syncwarp();
        }
    }

    __syncthreads(); // pad TMAs must complete before block exits.
}

// Local EM reduce kernel (inverse of local_permute_dup). Sums the top_k EM
// rows that share a FLAT recv slot and writes the bf16 result back into FLAT
// staging.
struct local_permute_reduce_param_t {
    void* flat_staging;
    const void* recv_x_em;
    const int32_t* flat2em_slot_map;
    const int32_t* num_recv_tokens_dev;
    // Optional fused EM to FLAT weight gather. Both null on FWD (token only).
    const float* em_weights_in;
    float* flat_weights_out;
    int top_k;
    int row_bytes;
};

// Direct-load reduce: each slot's row is reduced by a 128-thread sub-warp;
// with kSlotsPerBlock=8 a block computes 8 slots in parallel. For each int4
// lane the sub-warp's 128 threads accumulate across top_k contributors via
// direct cached global loads, then write the packed bf16 result back to
// flat_staging. HiddenInt4 = row_bytes / 16 is templated so the per-thread
// strided element loop is a compile-time bound.
template <int MaxTopK, int HiddenInt4, ncclDataType_t kTokenDtype = ncclBfloat16>
__device__ __forceinline__ void local_permute_reduce(
    uint8_t* __restrict__ flat_staging,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const float* __restrict__ em_weights_in,
    float* __restrict__ flat_weights_out,
    int top_k,
    int /*row_bytes*/) {
    constexpr int kRowBytes = HiddenInt4 * 16;

    constexpr int kThreadsPerSlot = 128;
    constexpr int kSlotsPerBlock = kLocalPermuteReduceSlotsPerBlock;
    constexpr int kBlockDim = kThreadsPerSlot * kSlotsPerBlock;
    constexpr int kElemsPerThread = (HiddenInt4 + kThreadsPerSlot - 1) / kThreadsPerSlot;

    // Per-slot packed em_slot ids in smem: only valid contributors (the rest
    // of top_k are -1 from non-local experts). Lets the inner loop iterate
    // n_valid instead of top_k, which is the dominant win at top_k > EPR.
    __shared__ int32_t smem_flat2em_slot_map[kSlotsPerBlock][MaxTopK];
    __shared__ int s_nvalid[kSlotsPerBlock];

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
    const int lane = tid % kThreadsPerSlot;

    const int num_recv = *num_recv_tokens_dev;
    const int slot_stride = kSlotsPerBlock * static_cast<int>(gridDim.x);

    for (int s_base = static_cast<int>(blockIdx.x) * kSlotsPerBlock; s_base < num_recv; s_base += slot_stride) {
        const int slot = s_base + slot_in_block;
        const bool slot_valid = (slot < num_recv);

        // Cooperative pack: warp 0 of each slot reads slot_row[lane] in
        // parallel, ballots valid lanes, and packs via warp scan. Requires
        // MaxTopK <= 32 (true for all current configs).
        static_assert(MaxTopK <= 32, "cooperative pack assumes MaxTopK <= 32");
        if (slot_valid && lane < 32) {
            const int32_t* slot_row_global = flat2em_slot_map + static_cast<size_t>(slot) * top_k;
            const int32_t s = (lane < top_k) ? __ldg(slot_row_global + lane) : -1;
            if (em_weights_in != nullptr && lane < top_k) {
                flat_weights_out[static_cast<size_t>(slot) * top_k + lane] = (s >= 0) ? em_weights_in[s] : 0.0f;
            }
            const unsigned valid = __ballot_sync(0xFFFFFFFFu, s >= 0);
            const int my_pos = __popc(valid & ((1u << lane) - 1));
            if (s >= 0) smem_flat2em_slot_map[slot_in_block][my_pos] = s;
            if (lane == 0) s_nvalid[slot_in_block] = __popc(valid);
        }
        __syncthreads();

        if (slot_valid) {
            const int n = s_nvalid[slot_in_block];

            int4* dst_int4 = reinterpret_cast<int4*>(flat_staging + static_cast<size_t>(slot) * kRowBytes);

            // Process the per-thread hidden-dim int4 indices in groups of
            // kHiddenVec so each iter has kHiddenVec * n LDGs in flight per
            // thread, hiding per-LDG latency. Cap kHiddenVec at
            // kElemsPerThread (JIT-known from HiddenInt4) so at small hidden
            // the dead u-lanes and their float2 accumulators disappear:
            // H=2048 -> kHiddenVec=2 (vs 4) frees 16 float regs per thread.
            constexpr int kHiddenVec = (kElemsPerThread < 4) ? kElemsPerThread : 4;
            for (int nn_base = 0; nn_base < kElemsPerThread; nn_base += kHiddenVec) {
                int js[kHiddenVec];
                bool valid_u[kHiddenVec];
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    const int nn = nn_base + u;
                    js[u] = lane + nn * kThreadsPerSlot;
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

                for (int k = 0; k < n; k++) {
                    const int32_t em_slot = smem_flat2em_slot_map[slot_in_block][k];
                    const int4* src =
                        reinterpret_cast<const int4*>(recv_x_em + static_cast<size_t>(em_slot) * kRowBytes);
                    int4 buf[kHiddenVec];
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (valid_u[u]) buf[u] = src[js[u]];
                    }
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!valid_u[u]) continue;
                        // int4 holds 2 FP32 pairs or 4 packed 16-bit pairs; decode
                        // each pair to FP32 and accumulate.
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
                    // Keep the FLAT recv row in L2 for the host-side D2D
                    // that reads it next.
                    nccl_ep::st_cg_global(&dst_int4[js[u]], out);
                }
            }
        }
        __syncthreads();
    }
}

} // namespace ht_ep

#include "scan_kernel.cuh"
