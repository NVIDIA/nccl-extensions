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
#include "ht_ep_configs.cuh"
#include "ht/ht_gin_signal_layout.cuh"
#include "mxfp8_quant.cuh"
#include "quant_recipe.cuh"
#include "ht/ht_combine_packed.cuh"
#include "ht/warp_group.cuh"
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda/ptx>
#include "nccl_device.h"
#include "cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x
#include "ht/ht_common.cuh"

namespace ht_gin_unordered {

// Compile-time scheduling avoids tail-indexing overhead that can regress coarse-kernel performance.
// Preserve complete coarse waves, subdivide only an additional partial wave.
template <nccl_ep::HtDispatchSchedule Schedule, int Chunk, int Teams>
class dispatch_work_t {
public:
    static constexpr bool kRedistributed = Schedule == nccl_ep::HtDispatchSchedule::Redistributed && Teams == 1;
    static constexpr int kMapStride = (kRedistributed) ? 1 : Chunk;

    __device__ dispatch_work_t(int n, int blocks) {
        count_ = nccl_ep::ceil_div(n, Chunk);
        bulk_chunks_ = count_;
        tail_start_ = 0;
        const int remainder = n % Chunk;
        last_size_ = (remainder != 0) ? remainder : Chunk;
        if constexpr (kRedistributed) {
            const int complete_chunks = nccl_ep::align_down(n / Chunk, blocks);
            const int complete_tokens = complete_chunks * Chunk;
            if (complete_chunks > 0 && n != complete_tokens) {
                bulk_chunks_ = complete_chunks;
                tail_start_ = complete_tokens;
                const int tail_tokens = n - complete_tokens;
                count_ = complete_chunks + nccl_ep::ceil_div(tail_tokens, nccl_ep::kHtDispatchTailChunkTokens);
                const int tail_remainder = tail_tokens % nccl_ep::kHtDispatchTailChunkTokens;
                last_size_ = (tail_remainder != 0) ? tail_remainder : nccl_ep::kHtDispatchTailChunkTokens;
            }
        }
        last_routing_loads_ = nccl_ep::ceil_div(last_size_, kTokensPerRoutingLoad);
    }
    __device__ __forceinline__ int count() const {
        return count_;
    }
    __device__ __forceinline__ int first(int unit) const {
        if constexpr (kRedistributed) {
            if (unit >= bulk_chunks_) {
                return tail_start_ + (unit - bulk_chunks_) * nccl_ep::kHtDispatchTailChunkTokens;
            }
        }
        return unit * Chunk;
    }
    __device__ __forceinline__ int size(int unit) const {
        if (unit == count_ - 1) {
            return last_size_;
        }
        if constexpr (kRedistributed) {
            if (unit >= bulk_chunks_) {
                return nccl_ep::kHtDispatchTailChunkTokens;
            }
        }
        return Chunk;
    }
    __device__ __forceinline__ int map_index(int unit) const {
        if constexpr (kRedistributed) {
            return first(unit);
        } else {
            return unit;
        }
    }
    __device__ __forceinline__ int metadata_chunk(int unit) const {
        if constexpr (kRedistributed) {
            return first(unit) / Chunk;
        } else {
            return unit;
        }
    }
    __device__ __forceinline__ int map_size(int unit) const {
        if constexpr (kRedistributed) {
            return size(unit);
        } else {
            return Chunk;
        }
    }
    __device__ __forceinline__ int routing_loads(int unit) const {
        return (unit == count_ - 1) ? last_routing_loads_ : (size(unit) / kTokensPerRoutingLoad);
    }

private:
    static constexpr int kTokensPerRoutingLoad = sizeof(uint4) / sizeof(bool);
    int bulk_chunks_;
    int count_;
    int tail_start_;
    int last_size_;
    int last_routing_loads_;
};

// Conditionally allocate compile-time arrays only when enabled.
template <bool ENABLE, int N>
struct acc_prob_storage_t {};

template <int N>
struct acc_prob_storage_t<true, N> {
    float data[N];
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
    // Compact FWD payload: original-order FP32 weights plus uint16 expert IDs,
    // padded to16B. wire_prob_floats measures this opaque row in 4-byte words;
    // transport copies all words without performing FP arithmetic on the IDs.
    bool compact_prob;
    int wire_prob_floats;
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
    // Compact BWD payload: round_up(top_k,4) FP32 gradients in original top-k
    // order. No IDs are carried or accumulated by combine.
    bool compact_prob;
    int wire_prob_floats;
};

struct model_config_t {
    int hidden_dim;
    int max_num_of_tokens_per_rank;
    int num_of_experts_per_rank;
    int ranks_per_lsa_team;
    int num_lsa_teams;
};

// Wire prob row width shared by dispatch and combine: dense has one float per
// LSA-team expert; compact has a runtime-width token-scoped row. Host validation
// guarantees a positive width on legs that carry probabilities.
template <typename Config>
__host__ __device__ __forceinline__ int
prob_row_floats(const Config& config, const model_config_t& model) {
    return config.compact_prob ? config.wire_prob_floats
                               : model.num_of_experts_per_rank * model.ranks_per_lsa_team;
}

// Combine wire prob row widths. The staged/reduced row (per LSA team) and the
// final output row differ under the dense format (output = per-team blocks x
// num_lsa_teams) but are the same single merged K4 row under the compact format
// (positions are globally disjoint across teams, so contributions merge).
__host__ __device__ __forceinline__ int
combine_prob_out_row_floats(const combine_config_t& config, const model_config_t& model) {
    return prob_row_floats(config, model) * (config.compact_prob ? 1 : model.num_lsa_teams);
}

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

// s2d-map double-buffer depth: consume one stage while prefetching the next (chunk, LSA team) row.
constexpr int S2D_MAP_RING_STAGES = 2;

using ht_common::em_s2d_unpack_rank;
using ht_common::em_s2d_unpack_slot;
using ht_common::slot_overflows;
using ht_common::drop_overflow_slot;

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

    // Prob buffer (only if forward dispatch, 16B aligned) -- total stages unchanged.
    // Width follows the wire format (dense per-LSA-team row or compact top-k row).
    const int wire_prob_row_floats = prob_row_floats(config, model);
    if (config.forward_dispatch) {
        layout.prob_buffer_stage_stride = wire_prob_row_floats * sizeof(float);
        layout.prob_buffer_stage_stride = nccl_ep::align(layout.prob_buffer_stage_stride, NCCL_EP_HT_TMA_ALIGN);
        layout.lsa_prob_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_stages * layout.prob_buffer_stage_stride;
    } else {
        layout.lsa_prob_buffer = nullptr;
        layout.prob_buffer_stage_stride = 0;
    }

    // Scaling factor buffer (only if quantized, 16B aligned) -- total stages unchanged
    if (config.sf_bytes_per_token > 0) {
        layout.sf_buffer_stage_stride = config.sf_bytes_per_token;
        layout.sf_buffer_stage_stride = nccl_ep::align(layout.sf_buffer_stage_stride, NCCL_EP_HT_TMA_ALIGN);
        layout.lsa_scaling_factor_buffer = reinterpret_cast<uint8_t*>(smem_base) + offset;
        offset += config.num_of_stages * layout.sf_buffer_stage_stride;
    } else {
        layout.lsa_scaling_factor_buffer = nullptr;
        layout.sf_buffer_stage_stride = 0;
    }

    // attn_to_rdma_map buffer (16B aligned, only if multi_lsa, shared across pipelines)
    if (model.num_lsa_teams > 1) {
        offset = nccl_ep::align(offset, static_cast<size_t>(NCCL_EP_HT_TMA_ALIGN));
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
    const model_config_t& model,
    // Wire prob row widths in floats. Dense: per-team row and per-team x
    // num_lsa_teams output row. Compact: both are the
    // same round_up(top_k, 4) merged row.
    int prob_row_floats,
    int prob_inter_row_floats) {
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
    layout.prob_G2S_stage_stride = prob_row_floats;
    layout.prob_S2G_stage_stride = prob_row_floats;
    layout.prob_S2G_inter_stage_stride = prob_inter_row_floats;

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
            offset += num_of_stages_g2s * prob_row_floats * sizeof(float);

            // lsa_prob_S2G_buffer
            align_offset(16);
            layout.lsa_prob_S2G_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_s2g * prob_row_floats * sizeof(float);
        } else {
            layout.lsa_prob_G2S_buffer = nullptr;
            layout.lsa_prob_S2G_buffer = nullptr;
        }

        // cross_lsa_prob_G2S_buffer
        align_offset(16);
        layout.cross_lsa_prob_G2S_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * prob_row_floats * sizeof(float);

        // cross_lsa_prob_S2G_buffer
        align_offset(16);
        layout.cross_lsa_prob_S2G_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * prob_inter_row_floats * sizeof(float);
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
    const int32_t* cached_topk_idx = nullptr;               // cached sender topk_idx [max_tokens, num_topk] (int32 expert ids)
    int32_t* num_recv_out = nullptr;                        // FLAT recv count for this rank (num_tokens_for_experts)
    const int32_t* per_src_lteam_num_tokens = nullptr;       // [1] real token count for the sender's stream
    int num_topk = 0;
    int num_src_ranks = 0;                                  // nRanks (== lsa_team_size in count mode)
    int lsa_team_size = 0;
    const uint64_t* token_dst_rank_bitmap = nullptr;        // [max_tokens] per-token global dest-rank bitmap
    uint8_t* const* recv_tables_ptrs = nullptr;             // [lsa_team_size] peer count-table regions
    int recv_max_slots = 0;                                 // slots per sub-table in the peer region
    // When set, the fused meta build clamps published EM per-expert offsets/counts to
    // recv_max_slots so over-budget tokens drop; the pre-drop total stays in num_recv_out.
    bool allow_overflow_drop = false;
    // Flat layout: no EM zones. The publish writes only num_recv_out and caller_recv_total
    // (rank-level dedup count); per-expert caller counts come from the post-dispatch LERM
    // recount (launch_lerm_flat_copy) and caller_offsets are not written.
    bool is_flat = false;
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
    // Compact FWD rows are opaque original-order weights+IDs payloads, with
    // wire_prob_floats giving their padded width in 4-byte words. Dense rows
    // remain per-LSA-team FP32 expert arrays.
    bool compact_prob;
    int wire_prob_floats;
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
    // Tail round counter; aliases expected_gin_flag_val unless split pass grids keep one per direction.
    uint64_t* expected_tail_signal_val;
    int local_rank;
    int lsa_team;
    // Capacity determines routing and sparse-map strides.
    int num_of_tokens_per_rank;
    int work_tokens_per_rank;
    // NCCL GIN context
    ncclDevComm* dcomm; // Device communicator (device pointer)
    ncclWindow_t token_window; // Source window handle for token data
    ncclWindow_t prob_window; // Source window handle for probability data
    ncclWindow_t sf_window; // Source window handle for scaling-factor data
    ncclWindow_t dest_window; // Destination window handle
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    // Memory Region info
    struct dispatch_memory_region_info_t mr_info;
    // Dispatch tail counter followed by its head-sync counter (per-rank, not IPC-shared).
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
    int output_prob_floats; // Compact caller row width; wire rows keep their padded stride
    int hidden_dim;
    int experts_per_rank;
    int ranks_per_lsa_team;
    // Compact BWD rows carry original-top-k FP32 gradients only; the forward
    // inline ID bytes never enter the combine reduction.
    bool compact_prob;
    int wire_prob_floats;
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
    // Combine tail counter followed by its head-sync counter.
    uint32_t* combine_grid_barrier_counter;
    // NCCL GIN context
    ncclDevComm_t* dcomm; // Device communicator (device pointer)
    ncclWindow_t token_window; // Source window handle for token data
    ncclWindow_t prob_window; // Source window handle for probability data
    ncclWindow_t dest_window; // Destination window handle
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
__forceinline__ __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
    nccl_ep::named_barrier_sync(barrier_id, num_threads);
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
// Every put carries its own weak signal and rings its own doorbell (ncclGinOptFlagsDefault).
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
    int sf_bytes_per_token,
    bool compact_prob, // compact top-k wire format
    unsigned sig_id) {
    size_t token_src = mr_info->attn_input_token_offset + token_idx * token_bytes;
    net.put(
        rail,
        remote_lteam_id,
        internal_window,
        dst_offset,
        token_window,
        token_src,
        token_bytes,
        ncclGin_WeakSignalAdd{sig_id, 1},
        ncclGin_None{},
        ncclCoopThread(),
        ncclGin_None{},
        cuda::thread_scope_thread,
        cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
    if constexpr (FORWARD_DISPATCH) {
        // Dense rows are laid out [token][lsa_team][experts_per_lsa_team] and
        // sliced per destination; the compact row is token-scoped (same
        // original-order weights + IDs for every destination).
        size_t prob_src = compact_prob
            ? mr_info->attn_input_prob_offset + token_idx * prob_bytes
            : mr_info->attn_input_prob_offset + (token_idx * LSA_TEAMS + remote_lteam_id) * prob_bytes;
        net.put(
            rail,
            remote_lteam_id,
            internal_window,
            dst_offset + token_bytes,
            prob_window,
            prob_src,
            prob_bytes,
            ncclGin_WeakSignalAdd{sig_id, 1},
            ncclGin_None{},
            ncclCoopThread(),
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsDefault);
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
            ncclGin_WeakSignalAdd{sig_id, 1},
            ncclGin_None{},
            ncclCoopThread(),
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsDefault);
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
    const int cidx,
    const uint64_t expected_flag_value,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const bool compact_prob,
    const uint32_t prob_bytes,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    const struct dispatch_memory_region_info_t* mr_info,
    int first_token) {
    g2s_source_t<TOKEN_DATA_TYPE> src;
    src.use_packed = false;
    src.packed_base = nullptr;
    src.token_base = nullptr;
    src.prob_base = nullptr;
    src.sf_base = nullptr;

    if (lteam_id != my_lteam) {
        // Remote: wait for the RDMA arrival signal, then point at this tile+chunk in the packed buffer.
        constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
        constexpr int CHUNK_STRIDE = NBLOCKS * NCCL_EP_HT_DISPATCH_N2N_WARPS;
        const ht_gin_signals::gin_chunk_signal_t tail_signal =
            ht_gin_signals::gin_dispatch_chunk_signal(
                lteam_id, my_lteam, cidx, MAX_CHUNKS_PER_RANK, CHUNK_STRIDE, mr_info->signals_tail_base);
        const unsigned tail_signal_id = tail_signal.sig_id;
        ncclGin net(dcomm, tail_signal.ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
        // Already a single lane here.
        // The G2S warp already raised the shadow by the number of puts this chunk carries.
        net.waitSignalMeetShadow(ncclCoopThread(), tail_signal_id);

        const int remote_slot = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
        const int chunk_first_token = cidx * TOKENS_PER_CHUNK;

        src.use_packed = true;
        src.packed_base = reinterpret_cast<const uint8_t*>(gin_base_ptr) +
                          dispatch_packed_entry_offset(mr_info, remote_slot, chunk_first_token);
    } else {
        // Local: strided bases into this rank's own global mem arrays for this chunk.
        int chunk_first_token = first_token;
        src.token_base = attn_input_token + chunk_first_token * HIDDEN_DIM;
        if constexpr (FORWARD_DISPATCH) {
            // Compact has one wire row per token; dense has one per LSA team.
            const size_t prob_row_floats = prob_bytes / sizeof(float);
            const size_t prob_row_stride = prob_row_floats * (compact_prob ? 1 : LSA_TEAMS);
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
    const bool compact_prob,
    const uint32_t prob_bytes,
    const int my_lteam,
    const struct dispatch_memory_region_info_t* mr_info) {
    uint64_t* mbar = smem_buffer_ptr->get_lsa_mbarrier_producer(pipeline_rank, stage);
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
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
            // Compact uses the same row for every destination; dense selects
            // this LSA team's slice within the full token row.
            const size_t prob_row_floats = prob_bytes / sizeof(float);
            const size_t prob_row = compact_prob ? static_cast<size_t>(cur_tokid)
                                                : static_cast<size_t>(cur_tokid) * LSA_TEAMS + my_lteam;
            prob_src = src.prob_base + prob_row * prob_row_floats;
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
    copy_bytes = nccl_ep::align(copy_bytes, static_cast<uint32_t>(NCCL_EP_HT_TMA_ALIGN));
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
template <typename TOKEN_DATA_TYPE, typename SMEM_TYPE, bool FORWARD_DISPATCH, bool HAS_SF>
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
    const uint32_t prob_bytes) {
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
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
        // Recv rows use the same width as the wire for either format.
        const size_t recv_prob_row_floats = prob_bytes / sizeof(float);
        prob_dst = remote_expert_output_prob[dst.remote_rank_id] + (slot * recv_prob_row_floats);
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
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const bool compact_prob,
    const uint32_t prob_bytes,
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
    constexpr int MAX_CHUNKS_PER_RANK = MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK;
    constexpr int NUM_REMOTE_LSA = LSA_TEAMS - 1;

    constexpr int CHUNK_STRIDE = NBLOCKS * N2N_WARPS;
    constexpr int GLOBAL_CHANNELS = CHUNK_STRIDE;
    static_assert(N2N_WARPS == NCCL_EP_HT_DISPATCH_N2N_WARPS,
                  "dispatch channel count must match the host's, which sizes the GIN context pool");

    // GIN device side setup. Single communicator; ctx_idx spreads QP traffic.
    int global_channel = ht_gin_signals::gin_warp_channel(blockIdx.x, n2n_warp_id, N2N_WARPS);
    int ctx_idx = ht_gin_signals::gin_channel_ctx(global_channel, GLOBAL_CHANNELS);

    ncclGin net(dcomm, ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam rail = ncclTeamRail(dcomm);

    // At most GIN_WINDOW chunks per rail peer in flight. Compiled in only when it can engage: the mere
    // presence of wait/flushAsync in this loop slows posting.
    constexpr int GIN_WINDOW = NCCL_EP_HT_DISPATCH_GIN_WINDOW;
    constexpr bool USE_WINDOW = MAX_CHUNKS_PER_RANK > GIN_WINDOW * CHUNK_STRIDE;
    const int n_mine = ht_gin_signals::gin_channel_num_chunks(global_channel, GLOBAL_CHANNELS, MAX_CHUNKS_PER_RANK);
    ncclGinRequest_t chunk_done[GIN_WINDOW][NUM_REMOTE_LSA];

    for (int n_posted = 0; n_posted < n_mine; ++n_posted) {
        const int cidx = ht_gin_signals::gin_chunk_at(global_channel, n_posted, GLOBAL_CHANNELS);
        if constexpr (USE_WINDOW) {
            if (n_posted >= GIN_WINDOW) {
                for (int j = 0; j < NUM_REMOTE_LSA; ++j) {
                    net.wait(chunk_done[n_posted % GIN_WINDOW][j], ncclCoopWarp());
                }
            }
        }
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

            // Signal id of this (chunk, destination) band; the waiter derives it through the same entry point.
            // A chunk past num_of_chunks_per_rank has csize 0, so it sends nothing and signals nothing.
            const unsigned sig_id =
                ht_gin_signals::gin_dispatch_chunk_signal(
                    my_lteam, remote_lteam_id, cidx, MAX_CHUNKS_PER_RANK, CHUNK_STRIDE,
                    smem_mr_info_ptr->signals_tail_base)
                    .sig_id;

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
                        sf_bytes_per_token,
                        compact_prob,
                        sig_id);
                }
                // Advance the compacted base past all written tokens in this word.
                dense_dst_offset += __popc(need_write_bitmask[word]) * entry_bytes;
            }

            if constexpr (USE_WINDOW) {
                if (n_posted + GIN_WINDOW < n_mine) {
                    net.flushAsync(rail, remote_lteam_id, &chunk_done[n_posted % GIN_WINDOW][j], ncclCoopWarp());
                }
            }
        }
    }
    // GIN flush with coopWarp includes syncwarp at the end
    net.flush(ncclCoopWarp(), cuda::memory_order_acquire);
}

// Count-mode: wait for this block's chunk unit u, built by MAP warp u % NUM_MAP_WARPS.
// Each MAP warp publishes its completed-chunk count; S2G waits before reading that chunk's
// s2d rows. The unit numbering must match dispatch_push_map_warp's split.
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
    nccl_ep::HtDispatchSchedule Schedule,
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
    const int work_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const uint32_t prob_bytes,
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

    // S2D inner dim: mode-dependent, carried by SMEM layout struct.
    const int s2d_inner_dim = smem_buffer_ptr->s2d_inner_dim;

    const int pipeline_rank = LSA_S2G_GROUP::warp_rank();
    using work_t = dispatch_work_t<Schedule, TOKENS_PER_CHUNK, LSA_TEAMS>;
    const work_t work(work_tokens_per_rank, NBLOCKS);
    const int num_of_chunks_per_rank = work.count();
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
                int current_chunk_size = work.size(chunk_idx);
                dispatch_s2g_prefetch_s2d_map<SMEM_TYPE, work_t::kMapStride>(
                    sparse_to_dense_map,
                    smem_buffer_ptr,
                    pipeline_rank,
                    s2d_stage,
                    my_lteam,
                    work.map_index(chunk_idx),
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
            if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) {
                continue;
            }

            const int csize = work.size(cidx);
            const int routing_loads_in_chunk = work.routing_loads(cidx);
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
                        int next_chunk_size = work.size(next_chunk_id);
                        dispatch_s2g_prefetch_s2d_map<SMEM_TYPE, work_t::kMapStride>(
                            sparse_to_dense_map,
                            smem_buffer_ptr,
                            pipeline_rank,
                            s2d_stage ^ 1,
                            next_lsa_id,
                            work.map_index(next_chunk_id),
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
                    rdma_to_attn_map + (lteam_id * routing_map_lsa_stride + work.first(cidx)));

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
                                    ((size_t)lteam_id * num_of_tokens_per_rank + work.first(cidx) +
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
                                        prob_bytes);
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
    nccl_ep::HtDispatchSchedule Schedule,
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
    bool HAS_SF>
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
    const int work_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const bool compact_prob,
    const uint32_t prob_bytes,
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
    constexpr int STAGES_PER_PIPELINE = NUM_STAGES / NUM_PIPELINES;

    const int pipeline_rank = LSA_G2S_GROUP::warp_rank();
    const dispatch_work_t<Schedule, TOKENS_PER_CHUNK, LSA_TEAMS> work(work_tokens_per_rank, NBLOCKS);
    const int num_of_chunks_per_rank = work.count();
    const int max_num_of_chunks_per_rank = nccl_ep::ceil_div(MAX_TOKENS_PER_RANK, TOKENS_PER_CHUNK);
    // TOKENS_PER_ROUTING_LOAD must match the producer's pad in scan_kernel.cuh
    const int routing_map_lsa_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
    int stage = 0;
    uint32_t consumer_parity = 1;
    int tokens_produced = 0;

    // All lanes walk the chunks and count the routing map together; lane 0 owns the pipeline state
    // (stage, consumer_parity, tokens_produced) and does the shadow raise, the wait and the copies.
    const int lane = LSA_G2S_GROUP::thread_rank() % 32;
    {
        int chunk_iter = 0;
        for (int cidx = blockIdx.x; cidx < num_of_chunks_per_rank; cidx += NBLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) {
                continue;
            }

            const int csize = work.size(cidx);
            const int routing_loads_in_chunk = work.routing_loads(cidx);

            for (int j = 0; j < LSA_TEAMS; j++) {
                // Walk LSA teams backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int lteam_id = (my_lteam + LSA_TEAMS - j) % LSA_TEAMS;

                const bool* routing_row =
                    rdma_to_attn_map + (lteam_id * routing_map_lsa_stride) + work.first(cidx);
                int tokens_needed = 0;
                if (lteam_id != my_lteam) {
                    // Number of tokens this team sends us in the chunk, 32 tokens per ballot.
                    for (int t0 = 0; t0 < csize; t0 += 32) {
                        const int t = t0 + lane;
                        tokens_needed += __popc(__ballot_sync(~0u, t < csize && routing_row[t]));
                    }
                }

                if (lane == 0) {
                    if (lteam_id != my_lteam) {
                        // Every gin.put of this chunk carries one weak signal.
                        constexpr uint64_t GIN_PUTS_PER_TOKEN = 1 + (FORWARD_DISPATCH ? 1 : 0) + (HAS_SF ? 1 : 0);
                        const ht_gin_signals::gin_chunk_signal_t sig = ht_gin_signals::gin_dispatch_chunk_signal(
                            lteam_id, my_lteam, cidx, MAX_TOKENS_PER_RANK / TOKENS_PER_CHUNK,
                            NBLOCKS * NCCL_EP_HT_DISPATCH_N2N_WARPS, mr_info->signals_tail_base);
                        ncclGin net(dcomm, sig.ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
                        net.increaseSignalShadow(sig.sig_id, GIN_PUTS_PER_TOKEN * static_cast<uint64_t>(tokens_needed));
                    }

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
                        cidx,
                        expected_flag_value,
                        HIDDEN_DIM,
                        sf_bytes_per_token,
                        compact_prob,
                        prob_bytes,
                        dcomm,
                        num_ctx_per_comm,
                        gin_base_ptr,
                        mr_info,
                        work.first(cidx));

                    const routing_loads_t* routing_map_ptr = reinterpret_cast<const routing_loads_t*>(routing_row);

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
                                    compact_prob,
                                    prob_bytes,
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
                __syncwarp();
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
    int wire_prob_dim, // staged prob row width (dense: epr*team; compact: round_up(top_k,4))
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
            prob_src = remote_expert_input_prob[rank_id] + (slot_st * wire_prob_dim);
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

// The receiver inverts the mapping above to recover the sender's cidx.
template <int LSA_TEAMS>
__forceinline__ __device__ int
combine_chunk_meta_inverse(int src_lteam, int dst_lteam, int chunk_id) {
    return chunk_id * (LSA_TEAMS - 1) + ((dst_lteam - src_lteam - 1 + LSA_TEAMS) % LSA_TEAMS);
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
    const int wire_prob_dim, // staged prob row width (dense: epr*team; compact: round_up(top_k,4); 0 = no prob)
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
    const uint32_t prob_bytes = (uint32_t)(wire_prob_dim * sizeof(float));

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
                wire_prob_dim,
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

// Zero the FP32 accumulators for one destination token. Used by the reducer before it adds
// source contributions, and on its own for a reserved slot under OVERFLOW_DROP.
template <bool BACKWARD_COMBINE, int NUM_ACC>
__forceinline__ __device__ void combine_zero_acc(
    float2 (&acc_token_fp32)[NUM_ACC],
    float* acc_prob_ptr,
    int prob_vec_per_thread) {
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

    combine_zero_acc<BACKWARD_COMBINE, NUM_ACC>(
        acc_token_fp32, acc_prob_ptr, prob_vec_per_thread);

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
            uint8_t* red_slot = reinterpret_cast<uint8_t*>(red_token_base) +
                                cur_tokid * ::nccl_ep::ht_packed::slot_bytes(red_token_bytes, BACKWARD_COMBINE, prob_dim);
            uint16_t* current_token_addr = reinterpret_cast<uint16_t*>(red_slot);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_global,
                cuda::ptx::space_shared,
                reinterpret_cast<void*>(current_token_addr),
                reinterpret_cast<const void*>(smem_buffer_ptr->get_lsa_token_S2G(dst_token_stage)),
                (uint32_t)(red_token_bytes));

            if constexpr (BACKWARD_COMBINE) {
                float* current_prob_addr = reinterpret_cast<float*>(red_slot + red_token_bytes);
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
    ncclEpCombQuant_t kCombineRecipe,
    bool OVERFLOW_DROP>
__forceinline__ __device__ void combine_RED_intra_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const bool* rdma_to_attn_map_wo_drop,
    // OUTPUT
    uint16_t* combine_gin_RED_tokens,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    const int wire_prob_dim, // staged prob row width (dense: epr*team; compact: round_up(top_k,4); 0 = no prob)
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
    // Backward-combine prob vectors stay in float (no BF16 packing). Width
    // follows the wire format (compact rows are <= the dense row by the
    // activation guard, so MAX_PROB_ELEM_PER_THRD still bounds the vectors).
    const int prob_dim = wire_prob_dim;
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

        const int rdma_map_offset = meta.lteam_id * rdma_per_lsa_sz + meta.chunk_id * TOKENS_PER_CHUNK;
        const routing_loads_t* rdma_map_base =
            reinterpret_cast<const routing_loads_t*>(rdma_to_attn_map + rdma_map_offset);
        // Slot numbering follows the pristine map, ring traffic the cleared one: the receiver's
        // attn_to_rdma_map keeps a dropped token's bit, so its slot must still be counted.
        const routing_loads_t* rdma_map_base_wo_drop = nullptr;
        if constexpr (OVERFLOW_DROP) {
            rdma_map_base_wo_drop =
                reinterpret_cast<const routing_loads_t*>(rdma_to_attn_map_wo_drop + rdma_map_offset);
        }

        // Per-token stride in gin_RED: Recipe::rdma_bytes, the inter-node row width
        // (dtype-width for NONE), matching the previous
        // HIDDEN_DIM * size_u16<kTokenDtype>() form for every unquantized case.
        const int red_stride =
            nccl_ep::combine_recipe_traits<kCombineRecipe>::template rdma_bytes<kTokenDtype>(HIDDEN_DIM);
        uint16_t* red_token_base = reinterpret_cast<uint16_t*>(
            reinterpret_cast<uint8_t*>(combine_gin_RED_tokens) +
            static_cast<size_t>(gin_RED_slot) * ::nccl_ep::ht_packed::slot_bytes(red_stride, BACKWARD_COMBINE, prob_dim));
        streaming_pending = 0;
        int additional_in_flight_s2g = 0;
        // Packed contiguous layout: the n-th routed token of this chunk goes to slot n. This walk is
        // sequential, so one cursor is the whole sender-side change.
        int packed_slot = 0;
        for (int load_idx = 0; load_idx < routing_loads_for_chunk; load_idx++) {
            routing_loads_t routing_data = rdma_map_base[load_idx];
            routing_loads_t routing_data_wo_drop = routing_data;
            if constexpr (OVERFLOW_DROP) {
                routing_data_wo_drop = rdma_map_base_wo_drop[load_idx];
            }
#pragma unroll
            for (int token_in_load = 0; token_in_load < TOKENS_PER_ROUTING_LOAD; token_in_load++) {
                int cur_tokid = load_idx * TOKENS_PER_ROUTING_LOAD + token_in_load;
                // Tail chunk: stop once past the real token count.
                if (cur_tokid >= meta.csize) {
                    break;
                }
                // Skip dst tokens this LSA team doesn't need. A dropped one keeps its slot.
                const bool routed = *(reinterpret_cast<bool*>(&routing_data_wo_drop) + token_in_load);
                if (!routed) {
                    continue;
                }
                const bool owed = *(reinterpret_cast<bool*>(&routing_data) + token_in_load);

                float2 acc_token_fp32[ACC_ELEM_PER_THRD];
                // acc_prob storage instantiated only in backward specializations.
                using acc_prob_storage_type =
                    acc_prob_storage_t<BACKWARD_COMBINE, MAX_PROB_ELEM_PER_THRD>;
                [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
                [[maybe_unused]] float* acc_prob_ptr = nullptr;
                if constexpr (BACKWARD_COMBINE) {
                    acc_prob_ptr = acc_prob_storage.data;
                }

                // owed alone decides this; the OVERFLOW_DROP conjunct is only there so the
                // non-drop build folds the branch away, since owed is always true here then.
                // A routed-but-dropped token still gets a zero row stored and sent: the NVLink
                // fanout never combined it, because that rank's recv buffer overflowed. The G2S
                // producer gates on the cleared map too, so it has no ring stage and the
                // zero-fill must not consume one.
                if (!OVERFLOW_DROP || owed) {
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
                } else {
                    combine_zero_acc<BACKWARD_COMBINE, ACC_ELEM_PER_THRD>(
                        acc_token_fp32, acc_prob_ptr, prob_vec_per_thread);
                }

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
                    packed_slot);
                packed_slot++;

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
    int token_idx_in_packed_chunk = 0;
    for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
        bool need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
        bool is_last = (token_idx_in_chunk == token_range - 1);
        if (need_write) {
            if (batch_count == 0) batch_start_in_chunk = token_idx_in_packed_chunk;
            batch_count++;
            token_idx_in_packed_chunk++;
        }
        bool should_flush = batch_count > 0 && (is_last || batch_count >= MAX_BATCH);
        if (should_flush) {
            if constexpr (STREAMING) {
                while (*streaming_counter < (cumulative_sent + batch_count)) {
                }
            }
            int batch_start_token = batch_start_in_chunk + chunk_first_token;
            const size_t slot = ::nccl_ep::ht_packed::slot_bytes(TOKEN_BYTES, BACKWARD_COMBINE, prob_dim);
            net.put(
                rail,
                lteam_id,
                nccl_internal_window,
                smem_mr_info_ptr->combine_g2s_token_offset +
                    (rank_in_remote * MAX_TOKENS_PER_RANK + batch_start_token) * slot,
                nccl_internal_window,
                smem_mr_info_ptr->combine_red_token_offset +
                    (rdma_tile_id * MAX_TOKENS_PER_RANK + batch_start_token) * slot,
                batch_count * slot,
                ncclGin_None{},
                ncclGin_None{},
                ncclCoopThread());

            if constexpr (STREAMING) {
                cumulative_sent += batch_count;
            }
            batch_count = 0;
            batch_start_in_chunk = -1;
        }
    }
}

// Signal the remote LSA team (via ncclGin) that this chunk has been delivered. Lane 0 only.
template <int NBLOCKS>
__forceinline__ __device__ void combine_n2n_signal_remote(
    ncclGin& net,
    ncclTeam rail,
    int lteam_id,
    int g,
    unsigned signals_base,
    unsigned combine_signal_offset) {
    unsigned signal_id =
        ht_gin_signals::gin_combine_chunk_signal(g, NBLOCKS, signals_base + combine_signal_offset).sig_id;
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
    ncclEpCombQuant_t kCombineRecipe,
    bool OVERFLOW_DROP>
__forceinline__ __device__ void combine_N2N_inter_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const bool* rdma_to_attn_map_wo_drop,
    const struct combine_memory_region_info_t* mr_info,
    // SCRATCH
    SMEM_TYPE* smem_buffer_ptr,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    const int wire_prob_dim, // reduced prob row width (dense: epr*team; compact: round_up(top_k,4); 0 = no prob)
    // CONFIG: ncclGin RDMA plumbing
    ncclDevComm_t* dcomm,
    ncclWindow_t nccl_internal_window,
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

    // One channel per CTA.
    const int channel = ht_gin_signals::gin_warp_channel(blockIdx.x, 0, 1);
    const int ctx_idx = ht_gin_signals::gin_channel_ctx(channel, NBLOCKS);
    ncclGin net(*dcomm, ctx_idx);

    const int n_mine = ht_gin_signals::gin_channel_num_chunks(channel, NBLOCKS, total_chunks);
    for (int k = 0; k < n_mine; ++k) {
        const int g = ht_gin_signals::gin_chunk_at(channel, k, NBLOCKS);
        // LSA-team/chunk mapping shared with the G2S receiver so signals match.
        const auto meta = combine_chunk_meta<LSA_TEAMS, TOKENS_PER_CHUNK>(g, my_lteam, cpr, rem_chunk_sz);
        const int lteam_id = meta.lteam_id;
        const int chunk_id = meta.chunk_id;
        // With rail-scoped GIN comms, LSA-team index maps to rail-team rank.
        const int rank_in_remote = lteam_id < my_lteam ? my_lteam - 1 : my_lteam;
        const bool is_residue = (chunk_id >= cpr);

        ncclTeam rail = ncclTeamRail(*dcomm);
        int rdma_tile_id = lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
        int chunk_base_token_idx = lteam_id * rdma_per_lsa_sz + chunk_id * TOKENS_PER_CHUNK;
        // Put every token the receiver counts: a reserved slot holds real zeros, and skipping
        // it would leave the receiver reading stale memory.
        const bool* put_gate_map = OVERFLOW_DROP ? rdma_to_attn_map_wo_drop : rdma_to_attn_map;
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
                    net, rail, lteam_id, nccl_internal_window, smem_mr_info_ptr,
                    put_gate_map, chunk_base_token_idx, chunk_id * TOKENS_PER_CHUNK, token_range,
                    rdma_tile_id, rank_in_remote, wire_prob_dim,
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
                combine_n2n_signal_remote<NBLOCKS>(net, rail, lteam_id, g, signals_base, combine_signal_offset);
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
                    net, rail, lteam_id, nccl_internal_window, smem_mr_info_ptr,
                    put_gate_map, chunk_base_token_idx, chunk_id * TOKENS_PER_CHUNK, token_range,
                    rdma_tile_id, rank_in_remote, wire_prob_dim,
                    nullptr, cumulative_sent);
            }

            // Signal remote
            __syncwarp();
            if (GIN_GROUP::thread_rank() == 0) {
                combine_n2n_signal_remote<NBLOCKS>(net, rail, lteam_id, g, signals_base, combine_signal_offset);
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

template <int LSA_TEAMS>
__forceinline__ __device__ int combine_g2s_rdma_tile(int lane_id, int my_lteam) {
    const int n = lane_id + 1;
    const int lteam_id = (my_lteam + LSA_TEAMS - n) % LSA_TEAMS;
    return lteam_id > my_lteam ? lteam_id - 1 : lteam_id;
}

// Map lane_id -> remote LSA team (lane 0 -> my_lteam-1, wrapping past self) and read its routing bit.
template <int LSA_TEAMS>
__forceinline__ __device__ rdma_lane_t
combine_g2s_resolve_rdma_lane(int lane_id, int my_lteam, const bool* attn_to_rdma_addr) {
    const int n = lane_id + 1;
    if (n >= LSA_TEAMS) {
        return rdma_lane_t{false, 0};
    }
    const int tile_id = combine_g2s_rdma_tile<LSA_TEAMS>(lane_id, my_lteam);
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
    int tile_id,
    int flat_token_id,
    int wire_prob_dim) {
    const int rdma_row = tile_id * MAX_NUM_OF_TOKENS_PER_RANK + flat_token_id;
    // rdma_bytes is the inter-node row width (dtype-width for NONE), matching the previous
    // HIDDEN_DIM * size_u16<kTokenDtype>() form for every unquantized case.
    const int rdma_wire_stride =
        nccl_ep::combine_recipe_traits<kCombineRecipe>::template rdma_bytes<kTokenDtype>(HIDDEN_DIM);
    const uint8_t* slot = reinterpret_cast<const uint8_t*>(combine_gin_G2S_tokens) +
                          static_cast<size_t>(rdma_row) *
                              ::nccl_ep::ht_packed::slot_bytes(rdma_wire_stride, BACKWARD_COMBINE, wire_prob_dim);
    const uint16_t* token_src = reinterpret_cast<const uint16_t*>(slot);
    const float* prob_src = nullptr;
    if constexpr (BACKWARD_COMBINE) {
        prob_src = reinterpret_cast<const float*>(slot + rdma_wire_stride);
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
    int flat_token_id,
    int& global_offset,
    int starting_G2S_index,
    int ring_len,
    int lane_id,
    int my_lteam,
    uint32_t token_bytes,
    uint32_t prob_bytes,
    int wire_prob_dim) {
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
                    lane.tile_id,
                    flat_token_id,
                    wire_prob_dim);
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
    // OUTPUT
    uint64_t* gin_G2S_flags,
    SMEM_TYPE* smem_buffer_ptr,
    // CONFIG
    const int my_lteam,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    const int wire_prob_dim, // staged prob row width (dense: epr*team; compact: round_up(top_k,4); 0 = no prob)
    const uint64_t expected_flag_value,
    const bool combine_local_reduce_enabled,
    // CONFIG: ncclGin RDMA plumbing
    ncclDevComm_t* dcomm,
    unsigned signals_base,
    unsigned combine_signal_offset,
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
    const uint32_t prob_bytes = (uint32_t)(wire_prob_dim * sizeof(float));

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
                for (int n = 1; n < LSA_TEAMS; n++) {
                    const int src_lteam = my_lteam >= n ? my_lteam - n : my_lteam + LSA_TEAMS - n;
                    const int global_cidx = combine_chunk_meta_inverse<LSA_TEAMS>(src_lteam, my_lteam, cidx);
                    const ht_gin_signals::gin_chunk_signal_t signal =
                        ht_gin_signals::gin_combine_chunk_signal(
                            global_cidx, NBLOCKS, signals_base + combine_signal_offset);
                    unsigned signal_id = signal.sig_id;
                    ncclGin net(*dcomm, signal.ctx_idx);
                    net.waitSignalMeetShadow(ncclCoopThread(), signal_id);
                }
            }
            __syncwarp(0xffffffff);

#pragma unroll
            for (int jj = 0; jj < LSA_TEAMS; ++jj) {
                rdma_flag_clear[jj] = true;
            }
        }

        // Layout rule and a worked example: build_chunk_mask in ht/ht_combine_packed.cuh.
        nccl_ep::ht_packed::chunk_mask_t<TOKENS_PER_CHUNK> packed_mask{};
        if constexpr (LSA_TEAMS > 1) {
            if (blockIdx.x < (unsigned)token_groups_for_chunk) {
                const int packed_col =
                    (lane_id + 1 < LSA_TEAMS) ? combine_g2s_rdma_tile<LSA_TEAMS>(lane_id, my_lteam) : -1;
                // packed_mask: one bitmap per lane, LSA_TEAMS - 1 live across the warp, one per
                // remote team. Bit t set = token t of this chunk is valid here and needs a slot.
                // 0xB5 = 1011 0101, and bit 0 is token 0 so it reads right to left: tokens
                // 0, 2, 4, 5, 7, which take slots 0, 1, 2, 3, 4.
                packed_mask = nccl_ep::ht_packed::build_chunk_mask<TOKENS_PER_CHUNK>(
                    attn_to_rdma_map_base, LSA_TEAMS - 1, LSA_TEAMS - 1, packed_col, csize);
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
                        wire_prob_dim,
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
                    // The sender's compacted slot for this lane's sender, which differs per lane.
                    // issue_rdma_g2s_row uses it only inside its own lane's address math, so a
                    // divergent value is safe.
                    const int flat_token_id =
                        cidx * TOKENS_PER_CHUNK + nccl_ep::ht_packed::prefix<TOKENS_PER_CHUNK>(packed_mask, cur_tokid);
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
                        flat_token_id,
                        global_offset,
                        starting_G2S_index,
                        ring_len,
                        lane_id,
                        my_lteam,
                        rdma_token_bytes,
                        prob_bytes,
                        wire_prob_dim);
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
    int prob_out_dim, // dense: prob_dim * LSA_TEAMS; compact: caller top-k (unpadded)
    bool compact_prob,
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
        if (compact_prob) {
            // All teams already merged into slot 0. Write the caller's K columns
            // directly; scalar FP32 stores support rows such as K10 (40 bytes),
            // which cannot use a 16-byte-aligned TMA row copy.
#pragma unroll
            for (int m = 0; m < prob_vec_per_thread; m++) {
                int element_id = thread_rank_within_pipeline + m * THRDS_PER_PIPELINE;
                if (element_id < prob_out_dim && absolute_token_id < num_real_tokens) {
                    attn_output_prob_base[static_cast<size_t>(token_in_chunk) * prob_out_dim + element_id] =
                        acc_prob_ptr[m];
                }
            }
        } else {
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
                if (!compact_prob) {
                    float* current_prob_addr = attn_output_prob_base + token_in_chunk * prob_out_dim;
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_global,
                        cuda::ptx::space_shared,
                        reinterpret_cast<void*>(current_prob_addr),
                        reinterpret_cast<const void*>(smem_buffer_ptr->get_cross_lsa_prob_S2G(dst_token_stage)),
                        (uint32_t)(prob_out_dim * sizeof(float)));
                }
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
    const int wire_prob_dim, // staged prob row width (dense: epr*team; compact: round_up(top_k,4); 0 = no prob)
    const bool compact_prob, // compact wire format: all LSA teams merge into ONE row
    const int output_prob_floats, // compact output omits the wire row's alignment padding
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
    const int prob_dim = wire_prob_dim;
    const int prob_vec_per_thread = nccl_ep::ceil_div(prob_dim, THRDS_PER_PIPELINE);
    // Output row width: dense = per-LSA-team blocks; compact = one merged row
    // (positions are globally disjoint across teams, so contributions merge).
    const int prob_out_dim = compact_prob ? output_prob_floats : prob_dim * LSA_TEAMS;
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
            attn_output_prob_base = attn_output_prob + (cidx * TOKENS_PER_CHUNK) * prob_out_dim;
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
                            if (compact_prob) {
                                // Compact: positions are globally disjoint across LSA
                                // teams and each remote team contributes at most once
                                // per dst token, so remote rows MERGE (+=) into the
                                // single slot-0 row instead of landing in per-team
                                // blocks.
                                combine_inter_consume_src<
                                    THRDS_PER_PIPELINE,
                                    BACKWARD_COMBINE,
                                    HIDDEN_DIM,
                                    /*READ_LAST_FLAG=*/false,
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
                            } else {
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
                    prob_out_dim,
                    compact_prob,
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
        if (pad == 0) continue;
        // DROP may truncate the last kept expert zone before the requested
        // alignment boundary. The scan publishes capacity-clamped endpoints,
        // so do not pad past this zone into the next allocation (e.g. prob).
        const int64_t zone_base = zone_offsets[e];
        const int64_t pad_room = zone_offsets[e + 1] - zone_base - count;
        if (pad_room < pad) pad = pad_room > 0 ? static_cast<int32_t>(pad_room) : 0;
        for (int p = 0; p < pad; p++, row_idx++) {
            if ((row_idx % global_stride) == global_id) {
                void* dst = reinterpret_cast<void*>(local_buf + (zone_base + count + p) * hidden_dim);
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
                            static_cast<size_t>(zone_base + count + p) * scale_row_bytes,
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
    const ncclDevComm& dcomm,
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
    const int32_t* topk_idx;
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
    bool is_flat;
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
template <nccl_ep::HtDispatchSchedule Schedule, int NUM_OF_TOKENS_PER_CHUNK, int MAX_DST_RANKS, int NUM_MAP_WARPS>
__device__ __forceinline__ void dispatch_push_map_build_s2d(
    const dispatch_push_map_ctx_t& ctx, int map_warp,
    int32_t (*s_rank_base)[MAX_DST_RANKS],
    const dispatch_work_t<Schedule, NUM_OF_TOKENS_PER_CHUNK, 1>& work) {
    using work_t = dispatch_work_t<Schedule, NUM_OF_TOKENS_PER_CHUNK, 1>;
    const int lane = (int)(threadIdx.x & 31);
    using RankSlice = rank_slice_t<MAX_DST_RANKS>;
    const RankSlice rank_slice{ctx.hit_word_offset, ctx.hit_bit_shift};
    constexpr int WAVES = nccl_ep::ceil_div(NUM_OF_TOKENS_PER_CHUNK, 32);
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
    for (int c = blockIdx.x; c < work.count(); c += num_blocks, unit++) {
        if (unit % NUM_MAP_WARPS != map_warp) {
            continue;
        }
        const int t0 = work.first(c);
        const int work_size = work.map_size(c);
        if (t0 >= stream_tokens) {
            built++;
            if (lane == 0) nccl_ep::st_release_cta(s2d_chunks_ready + map_warp, built);
            continue;
        }
        const uint64_t* stream_hits = token_dst_rank_bitmap;
        int32_t* stream_s2d = s2d;
        int32_t run[MAX_DST_RANKS];
        const int metadata_chunk = work.metadata_chunk(c);
        chunk_slot_base(s_rank_base[map_warp], per_src_lteam_chunk_rank, metadata_chunk, run);
        // MAP cursor run[r] starts after earlier source ranks and metadata chunks.
        // Split tail units also skip hits before t0 within the same metadata chunk,
        // preserving receive slots independently of the other CTAs' progress.
        if constexpr (work_t::kRedistributed) {
            for (int first = metadata_chunk * NUM_OF_TOKENS_PER_CHUNK; first < t0; first += nccl_ep::kWarpSize) {
                const int t = first + lane;
                uint64_t hits[RankSlice::PREFETCH_WORDS] = {};
                if (t < t0 && t < stream_tokens) {
                    const uint64_t* slice = stream_hits + (size_t)t * RankSlice::HIT_WORDS + rank_slice.word_offset;
#pragma unroll
                    for (int word = 0; word < RankSlice::PREFETCH_WORDS; word++) {
                        if (rank_slice.word_offset + word < RankSlice::HIT_WORDS) {
                            hits[word] = slice[word];
                        }
                    }
                }
                for (int r = 0; r < lsa_team_size; r++) {
                    const uint32_t rank_hit_mask = __ballot_sync(0xffffffffu, rank_slice.hit(hits, r));
                    run[r] += __popc(rank_hit_mask);
                }
            }
        }
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
                if (tc >= work_size || t >= stream_tokens) break;
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
                            printf("ncclEp HT count dispatch: recv slot %d exceeds "
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
template <nccl_ep::HtDispatchSchedule Schedule, int NUM_OF_TOKENS_PER_CHUNK, int MAX_DST_RANKS, int NUM_MAP_WARPS,
          int EXPERTS_PER_RANK>
__device__ __forceinline__ void dispatch_push_map_build_lerm(
    const dispatch_push_map_ctx_t& ctx, int map_warp,
    const dispatch_work_t<Schedule, NUM_OF_TOKENS_PER_CHUNK, 1>& work) {
    const int lane = (int)(threadIdx.x & 31);
    constexpr int MASK_WORDS = nccl_ep::bit_words(EXPERTS_PER_RANK);
    constexpr int WAVES = nccl_ep::ceil_div(NUM_OF_TOKENS_PER_CHUNK, 32);
    const int num_blocks = ctx.num_blocks;
    const int lsa_team_size = ctx.lsa_team_size;
    const int experts_per_rank = ctx.experts_per_rank;
    const int experts_per_lsa_team = ctx.experts_per_lsa_team;
    const int expert_off = ctx.expert_off;
    const int num_topk = ctx.num_topk;
    const int32_t* topk_idx = ctx.topk_idx;
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
    for (int c = blockIdx.x; c < work.count(); c += num_blocks, unit++) {
        const int t0 = work.first(c);
        const int unit_size = work.map_size(c);
        if (t0 >= stream_tokens) {
            break;
        }
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
            if (tc >= unit_size || t >= stream_tokens) {
                continue;
            }
            // Collect this token's routed dest ranks and packed expert masks in a
            // small list, then write only those rows -- avoids walking every dest
            // rank on wide NVLink domains where lsa_team_size >> num_topk.
            constexpr int kRankListCap = (MAX_DST_RANKS < 64) ? MAX_DST_RANKS : 64;
            int      touched_rank[kRankListCap];
            uint64_t touched_mask[kRankListCap][MASK_WORDS];
            int nrank = 0;
            // Derive each token's per-dest-rank expert mask from its topk row.
            const int32_t* row = topk_idx + (size_t)t * num_topk;
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
    const bool is_flat = pub.is_flat;
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
                printf("ncclEp HT count dispatch: recv tokens %d exceed max_recv_tokens_per_rank "
                       "%d; increase ncclEpGroupConfig_t::max_recv_tokens_per_rank or set "
                       "overflow_policy = NCCL_EP_OVERFLOW_DROP\n", tot, static_cast<int>(ctx.recv_max_slots));
                __trap();
            }
            // DROP: this FLAT recv count drives the permute's per-slot loop over the
            // recv_max_slots-sized FLAT/LERM/weights buffers, so clamp it to the budget when
            // dropping (over-budget FLAT slots were already dropped to -1 in build_s2d).
            if (num_recv_out)
                *num_recv_out =
                    (ctx.allow_overflow_drop && tot > ctx.recv_max_slots) ? ctx.recv_max_slots : tot;
            // Flat: recv total is the rank-level dedup count (pre-drop; see is_flat in
            // dispatch_push_count_kparams_t), not the per-expert sum below.
            if (caller_recv_total && is_flat) {
                if (caller_out_is_int64) {
                    *static_cast<int64_t*>(caller_recv_total) = tot;
                } else {
                    *static_cast<int32_t*>(caller_recv_total) = tot;
                }
            }
        }
        // EM per-expert zone publish (padded prefix offsets, counts, caller outputs). Flat has
        // no zones: its per-expert counts are recounted from the LERM after dispatch.
        if (!is_flat && (expert_token_offsets_out || caller_offsets || caller_counts)) {
            // Static SMEM: update calc_disp_static_smem.
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
                // alignment padding even when the per-rank count fits) aborts with scan-mode
                // parity.
                if (!drop && cum > cap) {
                    printf("ncclEp HT EM count dispatch: padded recv tokens %lld exceed "
                           "max_recv_tokens_per_rank %lld; increase "
                           "ncclEpGroupConfig_t::max_recv_tokens_per_rank or set NCCL_EP_OVERFLOW_DROP\n",
                           (long long)cum, (long long)cap);
                    __trap();
                }
                if (expert_token_offsets_out) {
                    expert_token_offsets_out[experts_per_rank] = (drop && cum > cap) ? cap : cum;
                    // recv_total_counter reports the pre-drop padded EM total (unclamped cum),
                    // one slot per (token, local-expert) hit. Flat's caller_recv_total is
                    // written from the rank-level tot above instead -- see that comment.
                    if (caller_recv_total) {
                        if (caller_out_is_int64) {
                            *static_cast<int64_t*>(caller_recv_total) = cum;
                        } else {
                            *static_cast<int32_t*>(caller_recv_total) = static_cast<int32_t>(cum);
                        }
                    }
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
                if (expert_token_offsets_out) {
                    expert_token_offsets_out[e] = expert_pub_off;
                    if (per_expert_counts_out) {
                        per_expert_counts_out[e] = out_actual;
                    }
                }
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
template <nccl_ep::HtDispatchSchedule Schedule, int NUM_OF_TOKENS_PER_CHUNK, int MAX_DST_RANKS, int NUM_MAP_WARPS,
          int EXPERTS_PER_RANK>
__device__ __forceinline__ void dispatch_push_map_warp(
    const dispatch_push_count_kparams_t& p,
    int experts_per_rank,    // runtime; == EXPERTS_PER_RANK, kept to match dispatch_push_map_ctx_t layout
    int my_rank,             // local (LSA) rank
    int my_lteam,
    int num_blocks,
    int work_tokens_per_rank, // Same local work domain as G2S/S2G.
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
    const dispatch_work_t<Schedule, NUM_OF_TOKENS_PER_CHUNK, 1> work(work_tokens_per_rank, num_blocks);
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
    ctx.count_row_ints = count_row_ints;
    ctx.count_rows = count_rows;
    ctx.hit_word_offset = rank_slice.word_offset;
    ctx.hit_bit_shift = rank_slice.bit_shift;
    // Static SMEM: update calc_disp_static_smem.
    __shared__ int32_t s_rank_base[NUM_MAP_WARPS][MAX_DST_RANKS];

    // Rank-base prefixes shared by the s2d builders and the LERM pass. All MAP warps
    // build s2d, so every warp needs its own row.
    dispatch_push_map_build_rank_base(ctx, s_rank_base[map_warp], lane);

    // Phase 1: build s2d (S2G-critical).
    dispatch_push_map_build_s2d<Schedule, NUM_OF_TOKENS_PER_CHUNK, MAX_DST_RANKS, NUM_MAP_WARPS>(
        ctx, map_warp, s_rank_base, work);

    // Phase 2: build receiver LERM tables.
    dispatch_push_map_build_lerm<Schedule, NUM_OF_TOKENS_PER_CHUNK, MAX_DST_RANKS, NUM_MAP_WARPS,
                            EXPERTS_PER_RANK>(ctx, map_warp, work);

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
    pub.is_flat = p.is_flat;
    dispatch_push_map_publish_outputs<EXPERTS_PER_RANK>(ctx, pub, map_warp);
}

// Elect the first block to claim *state (result broadcast to all threads in the block).
__device__ __forceinline__ bool elect_first_block(uint32_t* state, uint32_t claimed_state) {
    int elected = 0;
    if (threadIdx.x == 0) elected = atomicCAS(state, 0u, claimed_state) == 0u;
    return __syncthreads_or(elected);
}

// Elect the first block to claim *state (result broadcast within the calling warp).
__device__ __forceinline__ bool elect_first_block_warp(
    uint32_t* state, uint32_t claimed_state, int lane) {
    int elected = 0;
    if (lane == 0) elected = atomicCAS(state, 0u, claimed_state) == 0u;
    return __shfl_sync(0xffffffffu, elected, 0);
}

// Elect the last block to arrive at *counter (result broadcast to all threads in the block).
__device__ __forceinline__ bool elect_last_block(const int* counter, int num_blocks) {
    __syncthreads();
    int arrived = -1;
    if (threadIdx.x == 0) arrived = static_cast<int>(nccl_ep::atomic_add_acqrel_global(counter, 1));
    return __syncthreads_or(arrived == num_blocks - 1);
}

template <
    nccl_ep::HtDispatchSchedule Schedule,
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
    d_config.compact_prob = param.compact_prob;
    d_config.wire_prob_floats = param.wire_prob_floats;
    d_model.hidden_dim = HIDDEN_DIM;
    d_model.max_num_of_tokens_per_rank = MAX_TOKENS_PER_RANK;
    d_model.num_of_experts_per_rank = param.experts_per_rank;
    d_model.ranks_per_lsa_team = LSA_TEAM_SZ;
    d_model.num_lsa_teams = LSA_TEAMS;
    // Resolve the payload width once for all dispatch copy paths. Only source
    // addressing still needs the compact flag; receivers see uniform wire rows.
    const uint32_t prob_bytes = static_cast<uint32_t>(prob_row_floats(d_config, d_model) * sizeof(float));
    create_dispatch_smem_layout<kLayout, sizeof(TOKEN_DATA_TYPE)>(smem_layout, smem_bytes, d_config, d_model);
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    using head_init_warp = warp_group<1, 0>;
    using head_rdma_warp = warp_group<1, 1>;
    using head_lsa_warp = warp_group<1, 2>;

    uint32_t* dispatch_head_sync_counter = param.dispatch_grid_barrier_counter + 1;

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
        // warp 2: one first-admitted block publishes count metadata and runs the
        // intra-LSA barrier, then releases the other local blocks.
        const bool count_fanout =
            param.dispatch_push_count.active && param.dispatch_push_count.fused_meta_dispatch;
        const bool lsa_head_sync = param.guard_enabled || count_fanout;
        if (lsa_head_sync) {
            const int lane = head_lsa_warp::thread_rank();
            const bool head_lsa_leader =
                elect_first_block_warp(dispatch_head_sync_counter, 1u, lane);
            if (head_lsa_leader) {
                if (count_fanout) {
                    const int me =
                        my_lteam * LSA_TEAM_SZ + param.local_rank; // global sender-row index
                    const size_t row_ints =
                        (size_t)param.dispatch_push_count.num_src_ranks +
                        (size_t)param.experts_per_rank *
                            param.dispatch_push_count.num_src_ranks;
                    const int32_t* my_row = param.dispatch_push_count.own_row;
                    for (int p = 0; p < LSA_TEAM_SZ; p++) {
                        int32_t* dst = reinterpret_cast<int32_t*>(
                            param.dispatch_push_count.recv_tables_ptrs[p] +
                            param.dispatch_push_count.published_offset) +
                            (size_t)me * row_ints;
                        for (int i = lane; i < (int)row_ints; i += 32) dst[i] = my_row[i];
                    }
                    // Order every leader-warp count store before lane 0 publishes readiness.
                    __syncwarp();
                }
                if constexpr (LSA_TEAM_SZ != 1) {
                    ncclLsaBarrierSession<ncclCoopWarp> bar(
                        ncclCoopWarp(), *param.dcomm, ncclTeamTagLsa(), 0u);
                    bar.sync(
                        ncclCoopWarp(),
                        count_fanout ? cuda::memory_order_acq_rel : cuda::memory_order_relaxed);
                }
            }
            if (lane == 0) {
                int* counter = reinterpret_cast<int*>(dispatch_head_sync_counter);
                if (head_lsa_leader) {
                    nccl_ep::st_release_gpu_global(counter, 2);
                } else {
                    while (nccl_ep::ld_relaxed_gpu_global(dispatch_head_sync_counter) != 2u) {}
                    (void)nccl_ep::ld_acquire_global(counter);
                }
            }
        }
    }

    // Count mode: the dedicated MAP warps build s2d/masks per chunk after the split; S2G polls
    // this counter before each chunk's s2d prefetch, so G2S/N2N start immediately after the head
    // barrier. One s2d unit counter per MAP warp (round-robin units; see dispatch_push_map_warp).
    constexpr int kMapWarps = (MAP_GROUP::size() / 32 > 1) ? MAP_GROUP::size() / 32 : 1;
    // Static SMEM: update calc_disp_static_smem.
    __shared__ int s_s2d_chunks_ready[kMapWarps];
    if (param.dispatch_push_count.active && threadIdx.x < kMapWarps) s_s2d_chunks_ready[threadIdx.x] = 0;

    __syncthreads();

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    long long _wt_start = 0;
    if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif
    constexpr bool HAS_SF = (kRecipe == NCCL_EP_DISP_QUANT_FWD);
    int threadIdx_x_int = (int)threadIdx.x;
    if (threadIdx_x_int < GIN_GROUP::size()) {
        if constexpr (LSA_TEAMS != 1) {
#define DISPATCH_N2N_TEMPLATE \
            dispatch_N2N_warp<GIN_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_STAGES, TOKENS_PER_CHUNK, \
                              MAX_TOKENS_PER_RANK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, FORWARD_DISPATCH, HAS_SF>
            DISPATCH_N2N_TEMPLATE(
                param.attn_to_rdma_map,
                my_lteam,
                param.num_of_tokens_per_rank,
                HIDDEN_DIM,
                SF_BYTES_PER_TOKEN,
                param.compact_prob,
                prob_bytes,
                *param.dcomm,
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
        dispatch_G2S_warp<Schedule, LSA_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_STAGES, TOKENS_PER_CHUNK, \
                            MAX_TOKENS_PER_RANK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, NUM_PIPELINES, \
                            FORWARD_DISPATCH, HAS_SF>
        DISPATCH_G2S_TEMPLATE(
            param.rdma_to_attn_map,
            param.attn_input_token,
            param.attn_input_prob,
            param.attn_input_token_scaling_factor,
            param.gin_G2S_flags,
            param.local_rank,
            my_lteam,
            param.num_of_tokens_per_rank,
            param.work_tokens_per_rank,
            HIDDEN_DIM,
            SF_BYTES_PER_TOKEN,
            param.compact_prob,
            prob_bytes,
            *param.expected_tail_signal_val,
            *param.dcomm,
            param.num_ctx_per_comm,
            param.gin_base_ptr,
            &param.mr_info,
            smem_buffer_ptr);
#undef DISPATCH_G2S_TEMPLATE
    } else if (
        threadIdx_x_int < GIN_GROUP::size() + LSA_G2S_GROUP::size() + LSA_S2G_GROUP::size()) {
#define DISPATCH_S2G_TEMPLATE(COUNT_MODE) \
        dispatch_S2G_warp<Schedule, LSA_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_STAGES, \
                            IN_FLIGHT_S2G, TOKENS_PER_CHUNK, LSA_TEAMS, LSA_TEAM_SZ, NBLOCKS, NUM_PIPELINES, \
                            FORWARD_DISPATCH, HAS_SF, kLayout, kMapWarps, COUNT_MODE>( \
            param.rdma_to_attn_map, \
            param.sparse_to_dense_map, \
            param.expert_output_token, \
            param.expert_output_prob, \
            param.expert_output_scaling_factor, \
            my_lteam, \
            param.num_of_tokens_per_rank, \
            param.work_tokens_per_rank, \
            HIDDEN_DIM, \
            SF_BYTES_PER_TOKEN, \
            prob_bytes, \
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
        dispatch_push_map_warp<Schedule, TOKENS_PER_CHUNK, LSA_TEAM_SZ, kMapWarps, EXPERTS_PER_RANK>(
            param.dispatch_push_count,
            param.experts_per_rank,
            param.local_rank,
            my_lteam,
            NBLOCKS,
            param.work_tokens_per_rank,
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
            const int ntok = param.work_tokens_per_rank;
            const dispatch_work_t<Schedule, TOKENS_PER_CHUNK, 1> work(ntok, NBLOCKS);
            bool* r2a = const_cast<bool*>(param.rdma_to_attn_map);
            const int32_t* s2d = param.sparse_to_dense_map;
            const int map_tid = threadIdx_x_int - kMapBase;
            // Own this block's chunks only (same split as build_s2d), so we read s2d this block built.
            for (int c = blockIdx.x; c < work.count(); c += NBLOCKS) {
                const int unit_size = work.map_size(c);
                for (int ti = map_tid; ti < unit_size; ti += (int)MAP_GROUP::size()) {
                    const int t = work.first(c) + ti;
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
                nccl_ep::red_add_release_sys_global(param.lsa_S2G_flags, 1u);
                uint32_t flag_data;
                do {
                    flag_data = nccl_ep::ld_relaxed_sys_global(param.lsa_S2G_flags);
                } while (flag_data != expected_val);
                nccl_ep::memory_fence();
                atomicExch((unsigned int*)param.dispatch_grid_barrier_counter, 0u);
                atomicExch(dispatch_head_sync_counter, 0u);
                if (!param.local_dup_enabled)
                    *param.expected_lsa_flag_val += static_cast<uint32_t>(param.ranks_per_lsa_team);
            }
        } else if (tail_tid < tail_completion_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the cross-LSA-team RDMA guard + bump the RDMA round.
            if constexpr (LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_gin_flag_val;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        *param.dcomm,
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(my_lteam) * sizeof(uint64_t),
                        my_lteam,
                        LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) {
                    *param.expected_gin_flag_val = expected + 1ull;
                    if (param.expected_tail_signal_val != param.expected_gin_flag_val) {
                        *param.expected_tail_signal_val += 1ull;
                    }
                }
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
  ncclEpCombQuant_t kCombineRecipe,
  // NCCL_EP_OVERFLOW_DROP: reserve and zero-fill a fully dropped token's slot.
  bool OVERFLOW_DROP>
// Each CUDA block of combine kernel has named warp groups:
// intra/inter reduction, intra/inter G2S, and cross-LSA-team N2N RDMA. Group sizes are
// set by the HT combine warp-count constants and the selected pipeline count.
__device__ __forceinline__ void combine_kernel_impl(const combine_kernel_param_t<LSA_TEAM_SZ>& param,
                                                    uint8_t* smem_bytes,
                                                    const bool* rdma_to_attn_map_wo_drop) {
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
    // Wire prob row widths for this run (used consistently by the SMEM layout
    // and every prob-touching warp group below).
    combine_config_t c_wire_cfg{};
    c_wire_cfg.backward_combine = BACKWARD_COMBINE;
    c_wire_cfg.compact_prob = param.compact_prob;
    c_wire_cfg.wire_prob_floats = param.wire_prob_floats;
    const int wire_prob_dim = prob_row_floats(c_wire_cfg, c_model);
    const int wire_prob_out_dim = combine_prob_out_row_floats(c_wire_cfg, c_model);
    // Layout derives the element width from kTokenDtype (FP32 doubles the per-stage
    // token-buffer bytes vs BF16/FP16).
    create_combine_smem_layout<kTokenDtype, kCombineRecipe>(
        smem_layout,
        smem_bytes,
        STAGES_G2S,
        STAGES_S2G,
        TOKENS_PER_CHUNK,
        BACKWARD_COMBINE,
        c_model,
        wire_prob_dim,
        wire_prob_out_dim);
    smem_layout.s2d_inner_dim = param.s2d_inner_dim;
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    // ===== FUSED DEVICE SYNC (combine head) =====
    uint32_t* combine_head_sync_counter = param.combine_grid_barrier_counter + 1;
    const bool head_leader = elect_first_block(combine_head_sync_counter, 1u);
    using head_init_warp = warp_group<1, 0>; // warp 0
    using head_rdma_warp = warp_group<1, 1>; // warp 1
    const int head_tid = (int)threadIdx.x;
    if (head_tid < head_init_warp::size()) {
        // Every chunk will wait for a signal, so prepare the expectation here.
        if constexpr (LSA_TEAMS != 1) {
            if (head_leader) {
                const int total_wire = (LSA_TEAMS - 1) * MAX_CHUNKS_PER_RANK;
                for (int g = head_tid; g < total_wire; g += head_init_warp::size()) {
                    const ht_gin_signals::gin_chunk_signal_t signal =
                        ht_gin_signals::gin_combine_chunk_signal(
                            g, NBLOCKS, param.signals_base + param.combine_signal_offset);
                    ncclGin net(*param.dcomm, signal.ctx_idx);
                    net.increaseSignalShadow(signal.sig_id, 1ull);
                }
            }
        }
        __syncwarp();
        if (head_tid == 0) {
#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
            _wt_head_start = clock64();
#endif
            // Inter-rank completion barrier: the first admitted block signals
            // (red.release orders prior stores), every block polls.
            if (head_leader) nccl_ep::red_add_release_sys_global(param.lsa_S2G_flags, 1u);
            const uint32_t expected_val = *param.expected_lsa_flag_val;
            uint32_t flag_data;
            do {
                flag_data = nccl_ep::ld_relaxed_sys_global(param.lsa_S2G_flags);
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
                                   kTokenDtype, kCombineRecipe, OVERFLOW_DROP>
            COMBINE_RED_INTRA_TEMPLATE(
                // INPUT
                param.rdma_to_attn_map,
                rdma_to_attn_map_wo_drop,
                // OUTPUT
                param.combine_gin_RED_tokens,
                // CONFIG
                my_lteam,
                param.num_of_tokens_per_rank,
                param.experts_per_rank,
                wire_prob_dim,
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
            wire_prob_dim,
            param.compact_prob,
            param.output_prob_floats,
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
                wire_prob_dim,
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
            // OUTPUT
            param.gin_G2S_flags,
            smem_buffer_ptr,
            // CONFIG
            my_lteam,
            param.num_of_tokens_per_rank,
            param.experts_per_rank,
            wire_prob_dim,
            *param.expected_gin_flag_val,
            param.combine_local_reduce_enabled,
            param.dcomm,
            param.signals_base,
            param.combine_signal_offset,
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
                                   HIDDEN_DIM, kTokenDtype, kCombineRecipe, OVERFLOW_DROP>
            COMBINE_N2N_INTER_TEMPLATE(
                // INPUT
                param.rdma_to_attn_map,
                rdma_to_attn_map_wo_drop,
                &param.mr_info,
                // SCRATCH
                smem_buffer_ptr,
                // CONFIG
                my_lteam,
                param.num_of_tokens_per_rank,
                param.experts_per_rank,
                wire_prob_dim,
                param.dcomm,
                param.dest_window,
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
                atomicExch(combine_head_sync_counter, 0u);
                *param.expected_lsa_flag_val += static_cast<uint32_t>(param.ranks_per_lsa_team);
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the cross-LSA-team RDMA guard, then bump the RDMA round.
            if constexpr (LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_gin_flag_val;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        *param.dcomm,
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(my_lteam) * sizeof(uint64_t),
                        my_lteam,
                        LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_gin_flag_val = expected + 1ull;
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size() + tail_lsa_warp::size()) {
            // warp 2: intra-LSA LSA WAR barrier. Relaxed -- the tail __syncthreads already drained this
            // round's reads. Index = combine_barrier_offset, the same for both passes.
            if constexpr (LSA_TEAM_SZ != 1) {
                if (param.guard_enabled) {
                    ncclLsaBarrierSession<ncclCoopWarp> bar(
                        ncclCoopWarp(),
                        *param.dcomm,
                        ncclTeamTagLsa(),
                        param.combine_barrier_offset);
                    bar.sync(ncclCoopWarp(), cuda::memory_order_relaxed);
                }
            }
        }
    }
}

} // namespace ht_gin_unordered
