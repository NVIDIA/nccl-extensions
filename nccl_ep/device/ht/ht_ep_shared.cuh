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
#include "../ht_ep.cuh"

// The HT kernels and host helpers outside the dispatch and combine kernels. Both
// protocols use them from ht_ep; the unordered kernel replaces only those two.
// Included at the end of ht_ep.cuh, which it builds on.
namespace ht_ep {

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
// =====================================================================
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
using ht_common::flat_route_pack;
using ht_common::flat_route_expert;
using ht_common::flat_route_source_pos;
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
    // Prob buffer (16B aligned per stage); width follows the wire format.
    const int wire_prob_row_floats = prob_row_floats(config, model);
    if (config.forward_dispatch) {
        int prob_buffer_stage_stride = wire_prob_row_floats * sizeof(float);
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
// Static __shared__ of the dispatch kernel; callers subtract it from the dynamic budget.
// Keep in sync with s_rank_base, s_s2d_chunks_ready, s_pub_off and s_raw_cnt. Always counts
// 2 MAP warps: over-reserving costs ~1 KB, an under-count fails at launch.
inline size_t calc_disp_static_smem(const model_config_t& model) {
    constexpr size_t kMapWarps = 2;
    const size_t mask_words = static_cast<size_t>(nccl_ep::bit_words(model.num_of_experts_per_rank));
    const auto align16 = [](size_t n) { return (n + 15) & ~static_cast<size_t>(15); };
    return align16(kMapWarps * model.ranks_per_lsa_team * sizeof(int32_t)) +  // s_rank_base
           align16(kMapWarps * sizeof(int)) +                                 // s_s2d_chunks_ready
           align16((64 * mask_words + 1) * sizeof(int64_t)) +                 // s_pub_off
           align16(64 * mask_words * sizeof(int32_t));                        // s_raw_cnt
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
    // Wire prob width (dense per-team row or compact top-k row).
    int prob_stride = prob_row_floats(config, model) * sizeof(float);
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
    // Wire prob widths: the reduced (per-team) row and the final output row.
    // Compact collapses both to the same K4 row.
    const int prob_bytes = prob_row_floats(config, model) * sizeof(float);
    int prob_stride = (prob_bytes + 15) & ~15;
    int cross_prob_stride = (combine_prob_out_row_floats(config, model) * (int)sizeof(float) + 15) & ~15;
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
// ============================================================================
// Fills secondary EM slots from the primary slot in this rank's recv token
// buffer after dispatch (EM-unfused mode only).
// ============================================================================

template <typename T>
struct local_dup_kernel_param_t {
    T* expert_output_token; // [max_recv_tokens, hidden]
    float* expert_output_prob; // [max_recv_tokens, prob_row_floats]; opaque on compact FWD
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
    // Wire row width in 4-byte words: dense FP32 experts or the compact
    // weights+IDs payload. Primary -> secondary is an opaque byte copy;
    // BWD dispatch carries no probability row.
    int prob_row_floats;
    void* expert_output_scale; // QUANT_FWD: per-token scale row; void* (dtype varies by recipe)
    int scale_row_bytes;
};

// Dynamic shared-memory bytes required by local_dup_kernel_impl for the given
// hidden_dim and token element size (token dtype is BF16/uint16_t).
inline int local_dup_dynamic_smem_bytes(
    int hidden_dim,
    int pipe_depth,
    bool forward_dispatch,
    int prob_row_floats,
    size_t token_elem_bytes,
    int scale_row_bytes) {
    const int token_bytes = hidden_dim * static_cast<int>(token_elem_bytes);
    const int prob_bytes = forward_dispatch ? prob_row_floats * static_cast<int>(sizeof(float)) : 0;
    const int rings = pipe_depth * (token_bytes + prob_bytes + scale_row_bytes);
    const int mbar_bytes = pipe_depth * 2 * static_cast<int>(sizeof(uint64_t)) + 8;
    return rings + mbar_bytes;
}

template <typename T, int HIDDEN_DIM, int PIPE_DEPTH, bool FORWARD_DISPATCH,
          ncclEpDispQuant_t kRecipe = NCCL_EP_DISP_QUANT_NONE>
__device__ __forceinline__ void local_dup_kernel_impl(const local_dup_kernel_param_t<T>& p) {
    // Wait until all peers have signaled S2G completion on this rank's recv buffer.
    // Use >= rather than == so a future code path that overshoots the counter
    // (e.g. extra peer arrivals) doesn't hang.
    if (threadIdx.x == 0) {
        const uint32_t expected_val = *p.expected_lsa_flag_val;
        uint32_t v;
        do {
            v = nccl_ep::ld_relaxed_sys_global(p.lsa_S2G_flag);
        } while (v < expected_val);
        nccl_ep::memory_fence();
    }
    __syncthreads();

    constexpr int kTokenBytes = HIDDEN_DIM * sizeof(T);
    const int prob_floats = FORWARD_DISPATCH ? p.prob_row_floats : 0;
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
                    const float* src_prob = p.expert_output_prob +
                                            static_cast<size_t>(primary_em) * prob_floats;
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
                                          static_cast<size_t>(sec) * prob_floats;
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
    // Wire prob row width in floats: epr * ranks_per_lsa_team (dense) or
    // round_up(top_k, 4) (compact). Ignored unless BACKWARD_COMBINE.
    int prob_dim;
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
    const int PROB_DIM = BACKWARD_COMBINE ? p.prob_dim : 0;

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
// combine_reduce: one warp per attn token; each lane loads kCombineReduceUnroll int4 from each of
// kCombineReduceRankGroup partials before accumulating.
constexpr int kCombineReduceSlotsPerBlock = 32;
constexpr int kCombineReduceThreads = 32 * kCombineReduceSlotsPerBlock;
constexpr int kCombineReduceUnroll = 2;
constexpr int kCombineReduceRankGroup = 4;
// combine_push block-size upper bound for __launch_bounds__ (24 warps). The real block is sized
// at launch to fit the per-warp smem row buffer (see combine_push_slots_per_block); the kernel
// reads blockDim.x at runtime.
constexpr int kCombinePushThreads = 768;
constexpr int kCombinePushMaxSlots = kCombinePushThreads / 32;  // 24, sizes static metadata smem
// Tile geometry of a row: kHiddenVec int4 per lane, 32 lanes per warp, kNTiles tiles per row.
__host__ __device__ constexpr int combine_push_tile_int4(int hidden_int4) {
    const int epr = (hidden_int4 + 31) / 32;         // elems per thread
    return (epr < 4 ? epr : 4) * 32;
}
__host__ __device__ constexpr int combine_push_ntiles(int hidden_int4) {
    const int tile_int4 = combine_push_tile_int4(hidden_int4);
    return (hidden_int4 + tile_int4 - 1) / tile_int4;
}
// Chunk size in int4: requested is rounded up to whole tiles and clamped to the row; unset (<1)
// is the whole row (one chunk). Each chunk is reduced into one per-warp smem buffer and pushed
// with one TMA store.
__host__ __device__ constexpr int combine_push_chunk_size(int hidden_int4, int requested) {
    const int tile_int4 = combine_push_tile_int4(hidden_int4);
    const int row_int4 = combine_push_ntiles(hidden_int4) * tile_int4;
    if (requested < 1) return row_int4;
    const int rounded = (requested + tile_int4 - 1) / tile_int4 * tile_int4;
    return rounded > row_int4 ? row_int4 : rounded;
}
__host__ __device__ constexpr int combine_push_num_chunks(int hidden_int4, int chunk_size) {
    const int size = combine_push_chunk_size(hidden_int4, chunk_size);
    return (hidden_int4 + size - 1) / size;
}
// Per-warp push-buffer width in int4: one chunk, capped at the exact row so a single-chunk
// (whole-row) push is not padded up to a tile multiple.
__host__ __device__ constexpr int combine_push_buf_int4(int hidden_int4, int chunk_size) {
    const int size = combine_push_chunk_size(hidden_int4, chunk_size);
    return size < hidden_int4 ? size : hidden_int4;
}
// Fused MXFP8 per-warp buffer in int4: the loaded BF16 row, quantized in place to the packed record,
// plus a scale scratch that is moved behind the FP8 bytes once the BF16 row is consumed.
__host__ __device__ constexpr int combine_push_mxfp8_fused_int4(int hidden_int4) {
    using Mxfp8Traits = nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>;
    const int hidden = hidden_int4 * (16 / nccl_ep::size_u8<ncclBfloat16>());
    return hidden_int4 + nccl_ep::ceil_div(Mxfp8Traits::scale_bytes_per_token(hidden), 16);
}
// Dynamic smem = slots * (push buffer + ring stages) * buffer bytes. Static metadata separate.
__host__ __device__ constexpr size_t combine_push_dynamic_smem_bytes(
    int hidden_int4, int slots, int chunk_size, int ring) {
    return static_cast<size_t>(slots) * (1 + ring) * combine_push_buf_int4(hidden_int4, chunk_size) * 16;
}
// Static smem reserved for the max block's metadata: flat2em[MaxTopK<=32] + nvalid + dst ptr +
// mbarrier, per slot. s_mbar is declared in every specialization (see combine_push), even one
// that uses no mbarrier. Used by combine_push_slots_per_block to find how much of smem_cap is left
// for the per-warp row/chunk buffer.
__host__ __device__ constexpr int combine_push_meta_reserve_bytes() {
    return kCombinePushMaxSlots * (32 * 4 + 4 + 8 + 8) + 256;
}
// Largest slot count (<= kCombinePushMaxSlots) whose per-warp buffer fits smem_cap, reserving
// headroom for the static metadata arrays. At least 1.
__host__ __device__ constexpr int combine_push_slots_per_block(
    int hidden_int4, int smem_cap, int chunk_size) {
    const int row_bytes = combine_push_buf_int4(hidden_int4, chunk_size) * 16;
    int slots = (smem_cap - combine_push_meta_reserve_bytes()) / row_bytes;
    if (slots > kCombinePushMaxSlots) slots = kCombinePushMaxSlots;
    if (slots < 1) slots = 1;
    return slots;
}
// The combine_push load ring needs a multi-chunk push with a one-tile buffer.
__host__ __device__ constexpr bool combine_push_ring_supported(int hidden_int4, int chunk_size) {
    return combine_push_num_chunks(hidden_int4, chunk_size) > 1 &&
           combine_push_buf_int4(hidden_int4, chunk_size) == combine_push_tile_int4(hidden_int4);
}
// 24 warps x (1 + 3) one-tile buffers = 192 KiB, the most stages that fit the 227 KiB opt-in smem;
// also the most copies per row the ring serves.
constexpr int kCombinePushMaxRing = 3;
// Static smem for the ring's mbarriers beyond combine_push_meta_reserve_bytes, which holds one
// mbarrier per slot; the chunked ring and the single-chunk relay share that per-slot storage.
__host__ __device__ constexpr int combine_push_ring_bar_bytes(int ring) {
    return kCombinePushMaxSlots * (ring > 1 ? ring - 1 : 0) * 8;
}
// Load-ring stages per warp in the chunked combine_push: smem left after the slots' push buffers
// and ring mbarriers, capped. 0 (no ring) for a whole-row or multi-tile chunk, or unless
// top_k / team_size is in [1, depth]. That is the mean copies per row over all destinations, a
// routing heuristic: a row's copies can concentrate on one rank (up to top_k), which the ring still
// reduces. More copies on average favor the LDG gather.
__host__ __device__ constexpr int combine_push_ring_depth(
    int hidden_int4, int smem_cap, int chunk_size, int slots, int top_k, int team_size) {
    if (!combine_push_ring_supported(hidden_int4, chunk_size)) {
        return 0;
    }
    const int buf_int4 = combine_push_buf_int4(hidden_int4, chunk_size);
    const int free_bytes =
        smem_cap - combine_push_meta_reserve_bytes() - combine_push_ring_bar_bytes(kCombinePushMaxRing);
    int ring = free_bytes / (slots * buf_int4 * 16) - 1;
    if (ring > kCombinePushMaxRing) {
        ring = kCombinePushMaxRing;
    }
    if (ring < 0) {
        ring = 0;
    }
    if (top_k < team_size || top_k > ring * team_size) {
        return 0;
    }
    return ring;
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
    // Rebuild EM slots with per-expert atomic cursors, from decoded compact IDs
    // or (dense count mode) the sender-written LERM. Null uses the pre-built map.
    // Count mode also persists LERM in per-handle memory for later operations.
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
          ncclEpDispQuant_t kRecipe, bool kFlatExpertIds>
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
        // Permute warps: compact and dense count mode build flat2em inline;
        // other modes and backward dispatch use the persisted map.
        for (int token = static_cast<int>(blockIdx.x) * kPermuteWarps + warp_id;
             token < num_recv;
             token += kPermuteWarps * static_cast<int>(gridDim.x)) {
            const bool copy_weights =
                recv_topk_weights_em != nullptr && recv_topk_weights_flat != nullptr;
            if (flat2em_out != nullptr) {
                // Build EM slots in the weight row's order: compact uses the decoded
                // local IDs; dense count mode walks LERM in ascending expert order.
                // The persisted map is shared by dispatch and backward combine.
                const int lerm_words = nccl_ep::bit_words(experts_per_rank);
                const uint64_t* lrow = lerm_in == nullptr ? nullptr :
                    reinterpret_cast<const uint64_t*>(lerm_in) + static_cast<size_t>(token) * lerm_words;
                // Persist the LERM row into per-handle mem for combine (private per dispatch).
                if (lerm_out != nullptr) {
                    uint64_t* lrow_out =
                        reinterpret_cast<uint64_t*>(lerm_out) + static_cast<size_t>(token) * lerm_words;
                    for (int w = lane; w < lerm_words; w += kThreadsPerSlot) lrow_out[w] = lrow[w];
                }
                int32_t* out_row = flat2em_out + static_cast<size_t>(token) * top_k;
                int c = 0;
                if constexpr (kFlatExpertIds) {
                    for (int base = 0; base < top_k; base += kThreadsPerSlot) {
                        const int k = base + lane;
                        const int e = k < top_k ? out_row[k] : -1;
                        const bool hit = e >= 0;
                        const uint32_t bal = __ballot_sync(0xffffffffu, hit);
                        if (hit) {
                            EP_DEVICE_ASSERT(e < experts_per_rank);
                            const int cur = atomicAdd(&em_permute_cursors[e], 1);
                            const int32_t es = cur < per_expert_counts_active[e]
                                ? static_cast<int32_t>(expert_token_offsets[e]) + cur : -1;
                            s_active[warp_id][k] = es;
                            out_row[k] = es;
                            if (copy_weights && es >= 0) {
                                recv_topk_weights_em[es] =
                                    recv_topk_weights_flat[static_cast<size_t>(token) * top_k + k];
                            }
                        }
                        c += __popc(bal);
                    }
                } else {
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
// Fused pull-count: two MAP warps build the recv-slot/EM maps in-kernel. Barrier 0 is
// reserved for block-wide sync; the MAP warps use barrier 1.
constexpr int kPullDispatchMapWarps = 2;
// A MAP warp claims its next chunk from a grid counter only while fewer than this many of its
// chunks are unclaimed by pull warps, so the last chunks go to the CTAs that drain first.
constexpr int kPullDispatchChunkLookahead = 2;
// Tokens per MAP chunk (one source rank). A chunk is also one pull-warp ticket "unit".
constexpr int kPullDispatchChunkTokens = 32;
// Recv slots a pull warp claims per ticket; kPullDispatchChunkTokens / kPullDispatchSlotBatch
// pull-warp claims read each chunk's ring entry.
constexpr unsigned kPullDispatchSlotBatch = 4;
static_assert(kPullDispatchChunkTokens % kPullDispatchSlotBatch == 0, "slot batch must divide the chunk");
constexpr unsigned kPullDispatchChunkReaders = kPullDispatchChunkTokens / kPullDispatchSlotBatch;
// Per-MAP-warp slots in the chunk_ranges ring. A MAP warp reuses a slot only after every reader
// of its previous chunk has acked (chunks_read), so the ring size is not a correctness bound; it
// only needs to exceed the lookahead so the ack wait is rarely hit.
constexpr int kPullDispatchChunkRing = 8;
static_assert(kPullDispatchChunkRing >= 2 * kPullDispatchChunkLookahead + 2,
              "chunk_ranges ring must exceed the lookahead so ring reuse rarely waits on readers");
// layout_ready is an array of kPullLayoutReadyWords words (pull_recv_cursor tail), zeroed together.
constexpr int kPullLayoutFlagWord = 0;    // grid-wide "layout published" flag
constexpr int kPullChunkClaimWord = 1;    // grid-wide MAP chunk-claim counter
constexpr int kPullLayoutReadyWords = 2;
__device__ __forceinline__ void pull_layout_ready_reset(int32_t* layout_ready) {
    layout_ready[kPullLayoutFlagWord] = 0;
    layout_ready[kPullChunkClaimWord] = 0;
}
// Static smem of the MAP layout scan; counted in the host smem budget.
constexpr size_t kPullDispatchMapScanBytes = kPullDispatchMapWarps * sizeof(int);
constexpr int kPullMapBarrierId = 1;

// Max NVLink single-LSA-team ranks. The peer-pointer arrays live inline in the kernel
// param struct so they ride the stream-ordered launch marshaling; a persistent host
// staging + async H2D would race across back-to-back calls.
using ::kPullMaxLsaRanks;

// Per-rank stride of a peer's published metadata row (counts + topk snapshot).
__device__ __forceinline__ size_t pull_meta_stride(int topk_off_bytes, int tokens_per_rank) {
    return nccl_ep::align<size_t>(
        static_cast<size_t>(topk_off_bytes) + static_cast<size_t>(tokens_per_rank) * MAX_NUM_TOPK * sizeof(uint16_t), 16);
}

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
struct lsa_head_sync_param_t {
    ncclDevComm_t* dcomm;
    uint32_t* head_sync_flag;
};

struct lsa_tail_sync_param_t {
    ncclDevComm_t* dcomm;
    uint32_t* grid_barrier_counter;
    uint32_t* head_sync_flag;
};

// Standalone fused pull-count metadata publish, run ahead of the unfused head gate.
struct lsa_map_publish_param_t {
    const int32_t* own_row;
    const uint16_t* own_topk_snapshot;
    const uint8_t* meta_ptrs[kPullMaxLsaRanks];
    int topk_off_bytes;
    int tokens_per_rank;
    int top_k;
    int my_rank;
    int lsa_team_size;
    int32_t* layout_ready;
};

__device__ __forceinline__ void lsa_grid_head_gate_elected(
    ncclDevComm_t* dcomm, uint32_t* head_sync_flag, bool leader, int ready) {
    int* head_flag = reinterpret_cast<int*>(head_sync_flag);
    if (leader) {
        if (threadIdx.x < 32) {
            ncclLsaBarrierSession<ncclCoopWarp> bar(ncclCoopWarp(), *dcomm, ncclTeamTagLsa(), 0u);
            bar.sync(ncclCoopWarp(), cuda::memory_order_acq_rel);
        }
        __syncthreads();
        // Sys-scope hand-off: the waiter blocks read remote NVLink peer memory after this gate,
        // so their acquire must invalidate stale peer cache lines (a .gpu acquire only covers the
        // local GPU coherence domain). The paired release keeps the sync morally strong at sys scope.
        if (threadIdx.x == 0) nccl_ep::st_release_sys_global(head_flag, ready);
    } else {
        if (threadIdx.x == 0) {
            while (nccl_ep::ld_relaxed_gpu_global(head_sync_flag) != static_cast<uint32_t>(ready)) {}
            (void)nccl_ep::ld_acquire_sys_global(head_flag);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void lsa_grid_head_gate(ncclDevComm_t* dcomm, uint32_t* head_sync_flag) {
    constexpr int kReady = 1;
    const bool leader = elect_first_block(head_sync_flag, static_cast<uint32_t>(kReady + 1));
    lsa_grid_head_gate_elected(dcomm, head_sync_flag, leader, kReady);
}

// Block-uniform LSA grid tail barrier shared by pull dispatch and push combine.
__device__ __forceinline__ void lsa_grid_tail_barrier(
    ncclDevComm_t* dcomm,
    uint32_t* grid_barrier_counter,
    uint32_t* head_sync_flag) {
    __threadfence_system();
    if (elect_last_block(reinterpret_cast<const int*>(grid_barrier_counter), static_cast<int>(gridDim.x))) {
        if (threadIdx.x < 32) {
            ncclLsaBarrierSession<ncclCoopWarp> bar(ncclCoopWarp(), *dcomm, ncclTeamTagLsa(), 1u);
            bar.sync(ncclCoopWarp(), cuda::memory_order_acq_rel);
        }
        if (threadIdx.x == 0) {
            atomicExch(grid_barrier_counter, 0u);
            atomicExch(head_sync_flag, 0u);
            // combine_push pool counter (combine_head_sync_counter word).
            atomicExch(grid_barrier_counter + 1, 0u);
        }
    }
}

struct dispatch_pull_param_t {
    // Flat specialization: packed local-expert/source-position entries in srcpos_map.
    // These fields are unused by EM; layout is specialized in the JIT entry.
    bool flat;
    int64_t* recv_topk_idx;
    int output_expert_base; // 0 for LOCAL IDs, this rank's first global expert for GLOBAL
    int32_t* retained_slots; // Flat fused DROP destination slots.
    void* recv_x_em;                            // EM output token buffer
    float* recv_topk_weights_em;                // EM 1D weights out (null if not delivered)
    uint8_t* recv_x_scale_em;                   // QUANT_FWD: EM output scale buffer (null for NONE)
    // Fused pull-count builds these during dispatch; scan builds them during UpdateHandle.
    int32_t* flat2em_slot_map;                  // [num_recv, top_k]
    int32_t* srcpos_map;                        // [num_recv, top_k] source top-k pos of each hit (weights)
    int32_t* recv_slot_to_src;                  // [num_recv] recv slot -> global src token id
    const void* peer_input_ptrs[kPullMaxLsaRanks];   // [lsa_team_size] source token bases (NVLink)
    const float* peer_weight_ptrs[kPullMaxLsaRanks]; // [lsa_team_size] source topk_weights bases (null if not delivered)
    const void* peer_scale_ptrs[kPullMaxLsaRanks];   // [lsa_team_size] source scale bases (QUANT_FWD; null for NONE)
    int32_t* num_recv_tokens_dev;                // fused: written by the MAP group's Phase 0
    int64_t* expert_token_offsets;               // fused: written by the MAP group's Phase 0
    int32_t* per_expert_counts_active;           // fused: written by the MAP group's Phase 0
    int top_k;
    int experts_per_rank;
    int row_bytes;
    int scale_row_bytes;                        // QUANT_FWD: per-token scale row bytes (0 for NONE)
    int caller_num_recv_tokens;                 // caller EM recv buffer row capacity
    int tokens_per_rank;                        // decode src token = g % tokens_per_rank
    int lsa_team_size;                          // decode src rank within the LSA team
    // Intra-LSA head/tail sync: source ready before any peer read, reads done before reuse.
    ncclDevComm_t* dcomm;
    uint32_t* head_sync_flag;                   // grid head gate (idle counter during dispatch)
    uint32_t* grid_barrier_counter;             // head (fused) and tail elect-last-block
    // When set, head/tail sync runs as separate kernels; this kernel skips the inline sync.
    bool unfused_sync;

    // Fused pull-count inputs; unused on the scan path.
    const uint8_t* meta_ptrs[kPullMaxLsaRanks];  // peer metadata inboxes (NVLink)
    int topk_off_bytes;                 // byte offset of the uint16 topk half within a slice
    int32_t* rank_cursor;               // [lsa_team_size] per-source-rank recv-slot cursor (zeroed)
    int32_t* expert_cursor;             // [experts_per_rank] per-expert EM-slot cursor (zeroed)
    int32_t* layout_ready;              // [kPullLayoutReadyWords] layout-published flag + chunk-claim counter (zeroed)
    int my_rank;                        // this rank's slot within the LSA team
    int max_recv_tokens_per_rank;       // EM-slot budget (em_slot overflow bound)
    int flat_recv_capacity;             // FLAT recv-slot overflow bound
    int em_alignment;
    bool allow_overflow_drop;
    // MAP chunks beyond each warp's first are claimed dynamically; false keeps static assignment.
    bool claim_dynamic;
    // Publish handle-local snapshots at dispatch so batched UpdateHandles stay independent.
    const int32_t* own_row;             // [lsa_team_size + num_experts]
    const uint16_t* own_topk_snapshot;  // [tokens_per_rank, top_k] this rank's packed topk
    uint8_t* own_meta_staging;          // local inbox with one metadata row per source rank
    const int32_t* cached_cnt_rows;     // optional handle-local AllGather from UpdateHandle
    // Caller's dispatch-time layout_info outputs (mirrors dispatch_push_count_kparams_t); null
    // when the caller didn't request them at this ncclEpDispatch call.
    void* caller_offsets;               // [experts_per_rank] padded per-expert offsets
    void* caller_counts;                // [experts_per_rank] padded per-expert counts
    void* caller_recv_total;            // scalar pre-drop padded total
    bool caller_out_is_int64;
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

// Grid-strided metadata publish shared by the standalone publish kernel and the fused pull dispatch.
// Callers order the stores before the head sync.
__device__ __forceinline__ void dispatch_pull_map_publish_grid(
    const int32_t* __restrict__ own_row, const uint16_t* __restrict__ own_topk,
    const uint8_t* const* __restrict__ meta_ptrs, size_t row_stride, int topk_off_bytes,
    int tokens_per_rank, int top_k, int my_rank, int nRanks) {
    // Use size_t for grid-flat indices to avoid signed overflow on large (tokens_per_rank * top_k * nRanks) products.
    const size_t nthreads = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count_ints = own_row ? static_cast<size_t>(topk_off_bytes) / sizeof(int32_t) : 0;
    for (size_t idx = tid; idx < count_ints * nRanks; idx += nthreads) {
        const int r = static_cast<int>(idx / count_ints);
        const int i = static_cast<int>(idx - static_cast<size_t>(r) * count_ints);
        uint8_t* const row = const_cast<uint8_t*>(meta_ptrs[r]) + my_rank * row_stride;
        reinterpret_cast<volatile int32_t*>(row)[i] = own_row[i];
    }
    const int total = tokens_per_rank * top_k;
    int tail = 0;
    // topk section is 16-byte aligned when topk_off_bytes & 15 == 0, enabling uint4 bulk stores.
    if ((topk_off_bytes & 15) == 0) {
        const size_t chunks = static_cast<size_t>(total) / 8;
        if (chunks > 0) {
            for (size_t idx = tid; idx < chunks * nRanks; idx += nthreads) {
                const int r = static_cast<int>(idx / chunks);
                const int i = static_cast<int>(idx - static_cast<size_t>(r) * chunks);
                const uint4 v = reinterpret_cast<const uint4*>(own_topk)[i];
                int4* dst = reinterpret_cast<int4*>(
                    const_cast<uint8_t*>(meta_ptrs[r]) + my_rank * row_stride + topk_off_bytes);
                nccl_ep::st_na_relaxed(dst + i, reinterpret_cast<const int4&>(v));
            }
            tail = static_cast<int>(chunks * 8);
        }
    }
    const size_t rem = static_cast<size_t>(total - tail);
    if (rem > 0) {
        for (size_t idx = tid; idx < rem * nRanks; idx += nthreads) {
            const int r = static_cast<int>(idx / rem);
            const int i = tail + static_cast<int>(idx - static_cast<size_t>(r) * rem);
            volatile uint16_t* dst = reinterpret_cast<volatile uint16_t*>(
                const_cast<uint8_t*>(meta_ptrs[r]) + my_rank * row_stride + topk_off_bytes);
            dst[i] = own_topk[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        nccl_ep::memory_fence_seq_cst_sys();
    }
}

// Two MAP warps build maps and publish completed chunk ranges.
// Every block derives the layout from the published counts; CTA 0 also writes the global
// outputs and releases them to the pad warps.
__device__ __forceinline__ void dispatch_pull_map_warp(
    int topk_off_bytes,
    int32_t* __restrict__ recv_slot_to_src,
    int32_t* __restrict__ srcpos_map,
    int32_t* __restrict__ flat2em_slot_map,
    int32_t* __restrict__ num_recv_tokens_dev,
    int64_t* __restrict__ expert_token_offsets,
    int32_t* __restrict__ per_expert_counts_active,
    int32_t* __restrict__ rank_cursor,
    int32_t* __restrict__ expert_cursor,
    int32_t* __restrict__ layout_ready,
    int32_t* __restrict__ s_base,
    int32_t* __restrict__ s_expert_base,
    int top_k,
    int experts_per_rank,
    int lsa_team_size,
    int tokens_per_rank,  // == max_tokens (source token stride; matches decode_src's encoding)
    int my_rank,
    int max_recv_tokens_per_rank,
    int flat_recv_capacity,
    int em_alignment,
    bool allow_overflow_drop,
    void* __restrict__ caller_offsets,
    void* __restrict__ caller_counts,
    void* __restrict__ caller_recv_total,
    bool caller_out_is_int64,
    int map_idx,
    int lane,
    const uint8_t* local_meta,
    size_t meta_stride,
    int2* chunk_ranges,
    unsigned* chunks_ready,
    unsigned* chunks_done,
    unsigned* chunks_read,
    unsigned* chunk_next,
    int32_t* chunk_claim,
    const int32_t* cached_cnt_rows,
    bool claim_dynamic) {
    constexpr int nMapThreads = kPullDispatchMapWarps * 32;
    const int tid = map_idx * 32 + lane;
    const int nRanks = lsa_team_size;
    const int epr = experts_per_rank;
    const int expert_base_gid = my_rank * epr;
    auto peer_meta = [&](int r) { return local_meta + r * meta_stride; };
    const uint8_t* count_base = cached_cnt_rows
        ? reinterpret_cast<const uint8_t*>(cached_cnt_rows) : local_meta;
    const size_t count_stride = cached_cnt_rows
        ? static_cast<size_t>(nRanks) * (epr + 1) * sizeof(int32_t) : meta_stride;
    auto peer_count = [&](int r, int i) {
        const int32_t* src = reinterpret_cast<const int32_t*>(count_base + r * count_stride) + i;
        return *src;
    };

    const bool is_layout_writer = (blockIdx.x == 0);
    for (int r = tid; r < nRanks; r += nMapThreads) {
        s_base[r] = peer_count(r, my_rank);
    }

    __shared__ int s_warp_total[kPullDispatchMapWarps];
    static_assert(sizeof(s_warp_total) == kPullDispatchMapScanBytes, "update kPullDispatchMapScanBytes");
    const int align = (em_alignment > 1) ? em_alignment : 1;
    int carry = 0;
    for (int tile = 0; tile < epr; tile += nMapThreads) {
        const int k = tile + tid;
        int c = 0;
        if (k < epr) {
            for (int r = 0; r < nRanks; ++r) {
                c += peer_count(r, nRanks + expert_base_gid + k);
            }
        }
        const int padded = (align > 1 && c > 0) ? nccl_ep::align(c, align) : c;
        // Block-wide inclusive scan of the padded zone sizes.
        int v = padded;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, v, off);
            if (lane >= off) {
                v += n;
            }
        }
        if (lane == 31) {
            s_warp_total[map_idx] = v;
        }
        arrive_and_wait(nMapThreads, kPullMapBarrierId);
        int warp_prefix = carry;
        int tile_total = 0;
        for (int w = 0; w < kPullDispatchMapWarps; ++w) {
            const int warp_total = s_warp_total[w];
            if (w < map_idx) {
                warp_prefix += warp_total;
            }
            tile_total += warp_total;
        }
        const int ecum = warp_prefix + v - padded;
        if (k < epr) {
            s_expert_base[k] = ecum;
        }
        if (is_layout_writer && k < epr) {
            int room = padded;
            if (allow_overflow_drop) {
                room = (ecum < max_recv_tokens_per_rank) ? (max_recv_tokens_per_rank - ecum) : 0;
            }
            const int zone_base =
                (allow_overflow_drop && ecum > max_recv_tokens_per_rank) ? max_recv_tokens_per_rank : ecum;
            expert_token_offsets[k] = zone_base;
            if (per_expert_counts_active) {
                int out_actual = c;
                if (allow_overflow_drop) {
                    out_actual = (c < room) ? c : room;
                }
                per_expert_counts_active[k] = out_actual;
            }
            // Report padded layout sizes; active counts above exclude padding.
            if (caller_offsets || caller_counts) {
                int out_padded = padded;
                if (allow_overflow_drop) {
                    out_padded = (padded < room) ? padded : room;
                }
                if (caller_offsets) {
                    if (caller_out_is_int64) {
                        static_cast<int64_t*>(caller_offsets)[k] = zone_base;
                    } else {
                        static_cast<int32_t*>(caller_offsets)[k] = zone_base;
                    }
                }
                if (caller_counts) {
                    if (caller_out_is_int64) {
                        static_cast<int64_t*>(caller_counts)[k] = out_padded;
                    } else {
                        static_cast<int32_t*>(caller_counts)[k] = out_padded;
                    }
                }
            }
        }
        carry += tile_total;
        // s_warp_total is rewritten by the next tile.
        arrive_and_wait(nMapThreads, kPullMapBarrierId);
    }

    // Rank-major recv slots; tid 0 also publishes the totals.
    if (tid == 0) {
        int cum = 0;
        for (int r = 0; r < nRanks; r++) {
            const int count = s_base[r];
            s_base[r] = cum;
            cum += count;
        }
        const int true_total = cum;
        const bool overflow = true_total > max_recv_tokens_per_rank;
        if (overflow && !allow_overflow_drop) {
            if (is_layout_writer) {
                printf("ncclEpDispatch: HT fused pull-count FLAT recv tokens %d > max_recv_tokens_per_rank %d; "
                       "increase ncclEpGroupConfig_t::max_recv_tokens_per_rank "
                       "or set ncclEpGroupConfig_t::overflow_policy = NCCL_EP_OVERFLOW_DROP\n",
                       true_total, max_recv_tokens_per_rank);
            }
            __trap();
        }
        const int ecum = carry;
        // TRAP: padded EM total over budget under a non-DROP policy (may overflow via
        // alignment padding even when the FLAT count fits) aborts with scan-mode parity.
        if (!allow_overflow_drop && ecum > max_recv_tokens_per_rank) {
            if (is_layout_writer) {
                printf("ncclEpDispatch: HT fused pull-count padded EM total %d > max_recv_tokens_per_rank %d; "
                       "increase ncclEpGroupConfig_t::max_recv_tokens_per_rank "
                       "or set ncclEpGroupConfig_t::overflow_policy = NCCL_EP_OVERFLOW_DROP\n",
                       ecum, max_recv_tokens_per_rank);
            }
            __trap();
        }
        if (is_layout_writer) {
            // Also clamp to flat_recv_capacity: slots beyond it were never written by the MAP phase.
            const int capped_total = overflow ? max_recv_tokens_per_rank : true_total;
            *num_recv_tokens_dev = (capped_total < flat_recv_capacity) ? capped_total : flat_recv_capacity;
            expert_token_offsets[epr] = (ecum > max_recv_tokens_per_rank) ? max_recv_tokens_per_rank : ecum;
            // recv_total_counter reports the pre-drop padded total (unclamped ecum); equals the
            // unpadded FLAT recv total when em_alignment == 1. Matches dispatch_push_map_warp.
            if (caller_recv_total) {
                if (caller_out_is_int64) {
                    *static_cast<int64_t*>(caller_recv_total) = ecum;
                } else {
                    *static_cast<int32_t*>(caller_recv_total) = ecum;
                }
            }
        }
    }
    arrive_and_wait(nMapThreads, kPullMapBarrierId);
    if (is_layout_writer && tid == 0) {
        // Publish the completed layout for the pad warps.
        nccl_ep::st_release_sys_global(layout_ready, 1);
    }

    // Each warp reads 32 contiguous tokens from one source rank. Cycling ranks
    // across warps spreads recv-slot claims over the per-rank cursors.
    const int max_hits = top_k;
    const int num_cblocks = nccl_ep::ceil_div(tokens_per_rank, 32);
    const size_t total_chunks = static_cast<size_t>(nRanks) * num_cblocks;
    // Use vector loads and stores when top-k rows have the required alignment.
    const bool row_vec_ok = (top_k % 8 == 0) && ((topk_off_bytes & 15) == 0);
    const bool map_vec_ok = (top_k % 4 == 0) &&
        (((reinterpret_cast<uintptr_t>(flat2em_slot_map) | reinterpret_cast<uintptr_t>(srcpos_map)) & 15) == 0);
    unsigned built = 0;
    // Each MAP warp's first chunk is static. Later chunks are claimed from the grid counter, or follow
    // the same stride when claiming is off. round is warp-uniform (built is only advanced by lane 0).
    const size_t stride = static_cast<size_t>(gridDim.x) * kPullDispatchMapWarps;
    unsigned round = 0;
    for (;;) {
        size_t chunk = round * stride + static_cast<size_t>(blockIdx.x) * kPullDispatchMapWarps + map_idx;
        if (claim_dynamic && round > 0) {
            int claimed = 0;
            if (lane == 0) {
                cuda::atomic_ref<unsigned, cuda::thread_scope_block> next(*chunk_next);
                while (built * kPullDispatchMapWarps + map_idx >=
                       next.load(cuda::memory_order_relaxed) / kPullDispatchChunkTokens +
                           kPullDispatchChunkLookahead * kPullDispatchMapWarps) {
                }
                claimed = atomicAdd(chunk_claim, 1);
            }
            chunk = stride + static_cast<size_t>(__shfl_sync(0xffffffffu, claimed, 0));
        }
        if (chunk >= total_chunks) {
            break;
        }
        ++round;
        const int cblock = static_cast<int>(chunk / nRanks);
        const int r = static_cast<int>(chunk % nRanks);
        const int t = cblock * 32 + lane;
        const bool valid = t < tokens_per_rank;
        static_assert(kLocalPermuteMaxActivePerToken * MAX_NUM_TOPK <= 65536);
        uint16_t hits[MAX_NUM_TOPK];
        int n = 0;
        if (valid) {
            const uint8_t* row_base = peer_meta(r) + topk_off_bytes + static_cast<size_t>(t) * top_k * sizeof(uint16_t);
            if (row_vec_ok) {
                for (int k = 0; k < top_k; k += 8) {
                    const uint4* src = reinterpret_cast<const uint4*>(row_base + k * sizeof(uint16_t));
                    const uint4 v = *src;
                    const uint16_t* vals = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int ge = static_cast<int>(vals[j]);
                        const int el = ge - expert_base_gid;
                        if (el >= 0 && el < epr && n < max_hits) {
                            hits[n++] = static_cast<uint16_t>(el * MAX_NUM_TOPK + k + j);
                        }
                    }
                }
            } else {
                const uint16_t* row = reinterpret_cast<const uint16_t*>(row_base);
                for (int k = 0; k < top_k; k++) {
                    const int ge = static_cast<int>(row[k]);
                    const int el = ge - expert_base_gid;
                    if (el >= 0 && el < epr && n < max_hits) {
                        hits[n++] = static_cast<uint16_t>(el * MAX_NUM_TOPK + k);
                    }
                }
            }
        }
        // Warp-aggregated recv-slot claim: one ballot+atomic for the whole warp (r is uniform),
        // instead of a per-lane atomic. recv_slot indexes FLAT buffers (bound by flat_recv_capacity).
        const unsigned pmask_r = __ballot_sync(0xffffffffu, n > 0);
        int base_r = 0;
        if (pmask_r != 0) {
            const int leader = __ffs(pmask_r) - 1;
            if (lane == leader) base_r = atomicAdd(&rank_cursor[r], __popc(pmask_r));
            base_r = __shfl_sync(0xffffffffu, base_r, leader);
        }
        const int recv_slot = s_base[r] + base_r + __popc(pmask_r & ((1u << lane) - 1));
        // Overflowing slots are skipped unconditionally (matches the scan oracle's clamp): under
        // no-drop an overflow already trapped in Phase 0, but other blocks race ahead here, so an
        // unguarded write would go out of bounds before the host observes the trap. em_slot indexes
        // the EM budget.
        bool produce = false;
        if (n > 0) {
            if (!slot_overflows(recv_slot, flat_recv_capacity)) {
                produce = true;
                recv_slot_to_src[recv_slot] = r * tokens_per_rank + t;
                for (int j = 0; j < n; j++) {
                    const int hit = hits[j];
                    const int e = hit / MAX_NUM_TOPK;
                    const int em_slot = s_expert_base[e] + atomicAdd(&expert_cursor[e], 1);
                    const bool dropped = allow_overflow_drop && slot_overflows(em_slot, max_recv_tokens_per_rank);
                    const size_t o = static_cast<size_t>(recv_slot) * top_k + j;
                    flat2em_slot_map[o] = dropped ? -1 : em_slot;
                    srcpos_map[o] = dropped ? -1 : hit % MAX_NUM_TOPK;
                }
                int j = n;
                if (map_vec_ok) {
                    for (; j < top_k && (j & 3) != 0; ++j) {
                        const size_t o = static_cast<size_t>(recv_slot) * top_k + j;
                        flat2em_slot_map[o] = -1;
                        srcpos_map[o] = -1;
                    }
                    for (; j < top_k; j += 4) {
                        const size_t o = static_cast<size_t>(recv_slot) * top_k + j;
                        *reinterpret_cast<int4*>(flat2em_slot_map + o) = int4{-1, -1, -1, -1};
                        *reinterpret_cast<int4*>(srcpos_map + o) = int4{-1, -1, -1, -1};
                    }
                }
                for (; j < top_k; ++j) {
                    const size_t o = static_cast<size_t>(recv_slot) * top_k + j;
                    flat2em_slot_map[o] = -1;
                    srcpos_map[o] = -1;
                }
            }
        }
        // Order all lanes' map writes before lane 0 publishes the chunk.
        __syncwarp();
        const unsigned pmask = __ballot_sync(0xffffffffu, produce);
        if (lane == 0) {
            const unsigned ring_slot = (built % kPullDispatchChunkRing) * kPullDispatchMapWarps + map_idx;
            // Before reusing the slot, wait until every reader of its previous chunk has read it.
            cuda::atomic_ref<unsigned, cuda::thread_scope_block> read(chunks_read[ring_slot]);
            const unsigned reads_needed = (built / kPullDispatchChunkRing) * kPullDispatchChunkReaders;
            while (read.load(cuda::memory_order_acquire) < reads_needed) {
            }
            chunk_ranges[ring_slot] =
                make_int2(s_base[r] + base_r, __popc(pmask));
            cuda::atomic_ref<unsigned, cuda::thread_scope_block> ready(chunks_ready[map_idx]);
            ready.store(++built, cuda::memory_order_release);
        }
    }
    if (lane == 0) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_block> done(chunks_done[map_idx]);
        done.store(1, cuda::memory_order_release);
    }
}

// Result of waiting for a MAP chunk: usable, skipped (owner exhausted), or all MAP warps drained.
enum class PullChunkState { kReady, kSkip, kFinished };

// Lane 0 of a pull warp: wait for chunk idx of its owner MAP warp and read its range.
__device__ __forceinline__ PullChunkState dispatch_pull_wait_chunk(
    unsigned* chunks_ready, unsigned* chunks_done, unsigned* chunks_read, const int2* chunk_ranges, int owner,
    unsigned idx, int& first, int& count) {
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> ready(chunks_ready[owner]);
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> done(chunks_done[owner]);
    while (ready.load(cuda::memory_order_acquire) <= idx) {
        if (done.load(cuda::memory_order_acquire) != 0 && ready.load(cuda::memory_order_acquire) <= idx) {
            PullChunkState state = PullChunkState::kFinished;
            for (int w = 0; w < kPullDispatchMapWarps; ++w) {
                cuda::atomic_ref<unsigned, cuda::thread_scope_block> rw(chunks_ready[w]);
                cuda::atomic_ref<unsigned, cuda::thread_scope_block> dw(chunks_done[w]);
                if (dw.load(cuda::memory_order_acquire) == 0 || rw.load(cuda::memory_order_acquire) > idx) {
                    state = PullChunkState::kSkip;
                }
            }
            return state;
        }
    }
    const unsigned ring_slot = (idx % kPullDispatchChunkRing) * kPullDispatchMapWarps + owner;
    const int2 range = chunk_ranges[ring_slot];
    first = range.x;
    count = range.y;
    // Ack the read: the MAP warp may overwrite this slot once all kPullDispatchChunkReaders have.
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> read(chunks_read[ring_slot]);
    read.fetch_add(1, cuda::memory_order_release);
    return PullChunkState::kReady;
}

// ScaleInt4PerLane: per-lane int4 capacity for the QUANT_FWD scale row (loaded once
// into registers, scattered to every EM slot). JIT-sized from the actual scale row
// (ceil(scale_row_int4 / 32)), so any scale dtype / quantization block size is covered.
// ScaleTmaRowBytes: when non-zero, the scale row is moved by TMA through smem instead
// (exact row size; 0 keeps the register-held ld/st path).
// Fused count uses two MAP producers; scan consumes prebuilt maps.
template <int HiddenInt4, class Policy,
          ncclEpDispQuant_t kRecipe = NCCL_EP_DISP_QUANT_NONE, int ScaleInt4PerLane = 1,
          int ScaleTmaRowBytes = 0, int PullWarps = kPullDispatchMaxPullWarps, bool FusedMap = false>
__device__ __forceinline__ void dispatch_pull(
    const dispatch_pull_param_t& dp,
    // FusedMap literal-substitutes these at JIT-source-generation time, so they stay
    // separate (not read from dp) to keep the constant-folding/unrolling win.
    int top_k,
    int experts_per_rank,
    int lsa_team_size) {
    uint8_t* __restrict__ recv_x_em = static_cast<uint8_t*>(dp.recv_x_em);
    float* __restrict__ recv_topk_weights_em = dp.recv_topk_weights_em;
    uint8_t* __restrict__ recv_x_scale_em = dp.recv_x_scale_em;
    int32_t* __restrict__ flat2em_slot_map = dp.flat2em_slot_map;
    int32_t* __restrict__ srcpos_map = dp.srcpos_map;
    int32_t* __restrict__ recv_slot_to_src = dp.recv_slot_to_src;
    const void* const* __restrict__ peer_input_ptrs = dp.peer_input_ptrs;
    const float* const* __restrict__ peer_weight_ptrs = dp.peer_weight_ptrs;
    const void* const* __restrict__ peer_scale_ptrs = dp.peer_scale_ptrs;
    int32_t* __restrict__ num_recv_tokens_dev = dp.num_recv_tokens_dev;
    int64_t* __restrict__ expert_token_offsets = dp.expert_token_offsets;
    int32_t* __restrict__ per_expert_counts_active = dp.per_expert_counts_active;
    const int scale_row_bytes = dp.scale_row_bytes;
    const int caller_num_recv_tokens = dp.caller_num_recv_tokens;
    const int tokens_per_rank = dp.tokens_per_rank;
    ncclDevComm_t* __restrict__ dcomm = dp.dcomm;
    uint32_t* __restrict__ head_sync_flag = dp.head_sync_flag;
    uint32_t* __restrict__ grid_barrier_counter = dp.grid_barrier_counter;
    const bool unfused_sync = dp.unfused_sync;
    // FusedMap inputs only; ignored otherwise.
    const uint8_t* const* __restrict__ meta_ptrs = dp.meta_ptrs;
    const int topk_off_bytes = dp.topk_off_bytes;
    int32_t* __restrict__ rank_cursor = dp.rank_cursor;
    int32_t* __restrict__ expert_cursor = dp.expert_cursor;
    int32_t* __restrict__ layout_ready = dp.layout_ready;
    const int my_rank = dp.my_rank;
    const int max_recv_tokens_per_rank = dp.max_recv_tokens_per_rank;
    const int flat_recv_capacity = dp.flat_recv_capacity;
    const int em_alignment = dp.em_alignment;
    const bool allow_overflow_drop = dp.allow_overflow_drop;
    const int32_t* __restrict__ own_row = dp.own_row;
    const uint16_t* __restrict__ own_topk_snapshot = dp.own_topk_snapshot;
    uint8_t* __restrict__ own_meta_staging = dp.own_meta_staging;
    const int32_t* __restrict__ cached_cnt_rows = dp.cached_cnt_rows;
    void* __restrict__ caller_offsets = dp.caller_offsets;
    void* __restrict__ caller_counts = dp.caller_counts;
    void* __restrict__ caller_recv_total = dp.caller_recv_total;
    const bool caller_out_is_int64 = dp.caller_out_is_int64;
    static_assert(kRecipe == NCCL_EP_DISP_QUANT_NONE || kRecipe == NCCL_EP_DISP_QUANT_FWD,
                  "pull dispatch: only NONE and FWD (scales-forward) recipes are supported");
    constexpr bool kFwd = (kRecipe == NCCL_EP_DISP_QUANT_FWD);
    constexpr bool kScaleTma = kFwd && (ScaleTmaRowBytes > 0);
    constexpr bool kFusedMap = FusedMap;
    constexpr int kMapWarps = FusedMap ? kPullDispatchMapWarps : 0;
    constexpr int kPayloadWarps = PullWarps - kMapWarps;
    static_assert(kPayloadWarps > 0, "dispatch requires a payload warp");
    const size_t meta_stride = pull_meta_stride(topk_off_bytes, tokens_per_rank);
    using ScalePolicy = PullTmaStage<kScaleTma ? ScaleTmaRowBytes : 16>;

    const int warp_id = threadIdx.x / kPullDispatchThreadsPerSlot;
    const int lane = threadIdx.x & (kPullDispatchThreadsPerSlot - 1);
    const bool is_pad = warp_id >= PullWarps;
    const int pad_idx = warp_id - PullWarps;
    const bool is_map = kFusedMap && warp_id < kMapWarps;
    const int map_idx = warp_id;
    const int payload_warp = warp_id - kMapWarps;

    int4* dst_int4 = reinterpret_cast<int4*>(recv_x_em);
    // QUANT_FWD: per-token scale row (16B units), forwarded alongside the token row.
    const int scale_row_int4 = kFwd ? (scale_row_bytes >> 4) : 0;
    EP_DEVICE_ASSERT(!kFwd || scale_row_int4 <= 32 * ScaleInt4PerLane);

    __shared__ int32_t s_active[kPayloadWarps][kPullDispatchMaxActive];
    __shared__ int s_count[kPayloadWarps];
    // Per-pull-warp policy staging.
    extern __shared__ uint8_t s_policy[];

    // One transfer policy per pull warp; smem policies publish their mbarrier
    // inits before the block-wide fence, which every warp (incl. pad) must reach.
    Policy p;
    ScalePolicy sp;
    if (!is_pad && !is_map) {
        p.init(lane, s_policy + static_cast<size_t>(payload_warp) * Policy::smem_bytes_per_warp);
        if constexpr (kScaleTma) {
            sp.init(lane, s_policy +
                              static_cast<size_t>(kPayloadWarps) * Policy::smem_bytes_per_warp +
                              static_cast<size_t>(payload_warp) * ScalePolicy::smem_bytes_per_warp);
        }
    }
    Policy::block_init_fence();

    // Fused pull-count MAP smem: [s_base(lsa_team_size) | s_expert_base(epr)],
    // laid out right after the pull warps' Policy/ScalePolicy staging region. Unused when kMapWarps == 0.
    int32_t* s_map_base = nullptr;
    int32_t* s_map_expert_base = nullptr;
    int2* s_chunk_ranges = nullptr;
    if constexpr (kFusedMap) {
        constexpr size_t kPolicyTotalBytes =
            static_cast<size_t>(kPayloadWarps) * Policy::smem_bytes_per_warp +
            (kScaleTma ? static_cast<size_t>(kPayloadWarps) * ScalePolicy::smem_bytes_per_warp : 0);
        int32_t* s_map = reinterpret_cast<int32_t*>(s_policy + kPolicyTotalBytes);
        s_map_base = s_map;
        s_map_expert_base = s_map_base + lsa_team_size;
        const size_t map_bytes = static_cast<size_t>(lsa_team_size + experts_per_rank) * sizeof(int32_t);
        s_chunk_ranges = reinterpret_cast<int2*>(s_policy + kPolicyTotalBytes +
            ((map_bytes + alignof(int2) - 1) & ~size_t(alignof(int2) - 1)));
    }

    __shared__ unsigned s_chunks_ready[kFusedMap ? kPullDispatchMapWarps : 1];
    __shared__ unsigned s_chunks_done[kFusedMap ? kPullDispatchMapWarps : 1];
    __shared__ unsigned s_chunks_read[kFusedMap ? kPullDispatchChunkRing * kPullDispatchMapWarps : 1];
    __shared__ unsigned s_chunk_next;
    if constexpr (kFusedMap) {
        if (threadIdx.x < kMapWarps) {
            s_chunks_ready[threadIdx.x] = 0;
            s_chunks_done[threadIdx.x] = 0;
        }
        for (int i = threadIdx.x; i < kPullDispatchChunkRing * kPullDispatchMapWarps; i += blockDim.x) {
            s_chunks_read[i] = 0;
        }
        if (threadIdx.x == 0) s_chunk_next = 0;
        __syncthreads();
    }

    const bool copy_weights = (recv_topk_weights_em != nullptr && peer_weight_ptrs != nullptr);
    const int kNWarps = PullWarps * static_cast<int>(gridDim.x);
    const int gwarp = static_cast<int>(blockIdx.x) * PullWarps + warp_id;
    // Read + scatter one recv slot: NVLink token pull + destination resolve + EM store.
    auto process_slot = [&](int token) {
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
            // MAP warps can write these rows during this kernel.
            const int32_t es = (k < top_k) ? slot_row[k] : -1;
            const bool keep = (es >= 0);
            if (keep) {
                // Per-slot capacity backstop (mirror of local_permute_dup).
                EP_DEVICE_ASSERT(es < caller_num_recv_tokens);
                // Read the weight at the source's top-k position (order-preserving),
                // not the packed flat2em index. srcpos_row aligns 1:1 with slot_row.
                if (copy_weights) {
                    const int wpos = srcpos_row ? srcpos_row[k] : k;
                    recv_topk_weights_em[es] = __ldg(src_w + wpos);
                }
            }
            const unsigned mask = __ballot_sync(0xffffffffu, keep);
            const int pos = c + __popc(mask & ((1u << lane) - 1));
            if (keep && pos < kPullDispatchMaxActive) s_active[payload_warp][pos] = es;
            c += __popc(mask);
        }
        if (lane == 0) s_count[payload_warp] = c;
        __syncwarp();

        p.wait(); // token row resident
        if constexpr (kScaleTma) sp.wait();
        const int cnt = s_count[payload_warp];
        for (int a = 0; a < cnt; ++a) {
            const int32_t slot = s_active[payload_warp][a];
            p.store(dst_int4 + static_cast<size_t>(slot) * HiddenInt4);
            if constexpr (kFwd) scatter_scale(slot);
        }
        bulk_wait_reads(lane);
    };

    if constexpr (kFusedMap) {
        // When unfused, a standalone publish kernel and head-sync kernel already did this.
        if (!unfused_sync) {
            constexpr int kMapReady = 2;
            dispatch_pull_map_publish_grid(cached_cnt_rows ? nullptr : own_row, own_topk_snapshot,
                meta_ptrs, meta_stride, topk_off_bytes, tokens_per_rank, top_k, my_rank,
                lsa_team_size);
            const bool head_leader = elect_last_block(reinterpret_cast<const int*>(grid_barrier_counter),
                                                      static_cast<int>(gridDim.x));
            if (head_leader) {
                if (threadIdx.x == 0) {
                    pull_layout_ready_reset(layout_ready);
                    atomicExch(grid_barrier_counter, 0u);
                }
                // Reset the slot cursors (and the trailing arrival counter) before any block claims.
                for (int i = threadIdx.x; i < lsa_team_size; i += blockDim.x) {
                    rank_cursor[i] = 0;
                }
                for (int i = threadIdx.x; i <= experts_per_rank; i += blockDim.x) {
                    expert_cursor[i] = 0;
                }
            }
            lsa_grid_head_gate_elected(dcomm, head_sync_flag, head_leader, kMapReady);
        }
    } else if (!unfused_sync) {
        lsa_grid_head_gate(dcomm, head_sync_flag);
    }

    if constexpr (kFusedMap) {
        if (is_map) {
            dispatch_pull_map_warp(
                topk_off_bytes, recv_slot_to_src, srcpos_map, flat2em_slot_map,
                num_recv_tokens_dev, expert_token_offsets, per_expert_counts_active, rank_cursor,
                expert_cursor, layout_ready, s_map_base, s_map_expert_base, top_k, experts_per_rank,
                lsa_team_size, tokens_per_rank, my_rank,
                max_recv_tokens_per_rank, flat_recv_capacity, em_alignment, allow_overflow_drop,
                caller_offsets, caller_counts, caller_recv_total, caller_out_is_int64,
                map_idx, lane, own_meta_staging, meta_stride, s_chunk_ranges, s_chunks_ready,
                s_chunks_done, s_chunks_read, &s_chunk_next, layout_ready + kPullChunkClaimWord, cached_cnt_rows,
                dp.claim_dynamic);
        } else if (is_pad) {
            if (lane == 0) {
                while (nccl_ep::ld_acquire_sys_global(layout_ready) == 0) {
                }
            }
            __syncwarp();
        }
    }

    // Fused map: each block holds the layout in smem; only the pad warps wait for the global outputs.
    if (!kFusedMap || is_pad) {
        const int64_t em_padded_total = expert_token_offsets[experts_per_rank];
        EP_DEVICE_ASSERT(caller_num_recv_tokens >= em_padded_total);
    }

    if (is_pad) {
        dispatch_pull_pad_fill<HiddenInt4, kFwd, kPullDispatchThreadsPerSlot, kPullDispatchPadWarps>(
            dst_int4, recv_x_scale_em, recv_topk_weights_em, expert_token_offsets,
            per_expert_counts_active, experts_per_rank, scale_row_bytes, scale_row_int4, pad_idx, lane);
    } else {
        if constexpr (kFusedMap) {
            if (!is_map) {
                // Claim small batches from the slot positions of each chunk; skip the unused tail.
                constexpr unsigned kSlotBatch = kPullDispatchSlotBatch;
                for (;;) {
                    unsigned ticket = 0;
                    if (lane == 0) ticket = atomicAdd(&s_chunk_next, kSlotBatch);
                    ticket = __shfl_sync(0xffffffffu, ticket, 0);
#ifdef NCCL_EP_TEST_PULL_TICKET0_STALL_US
                    // Test only: hold the block's first ticket before reading its chunk range, so the
                    // MAP warps run ahead across a ring wrap (see test_ht_pull_chunk_ring).
                    if (lane == 0 && ticket == 0) {
                        for (unsigned waited_us = 0; waited_us < NCCL_EP_TEST_PULL_TICKET0_STALL_US; waited_us += 100) {
                            __nanosleep(100000);
                        }
                    }
                    __syncwarp();
#endif
                    const unsigned unit = ticket / kPullDispatchChunkTokens;
                    const int owner = unit % kMapWarps;
                    const unsigned idx = unit / kMapWarps;
                    int first = 0, count = 0, state = 0;
                    if (lane == 0) {
                        state = static_cast<int>(dispatch_pull_wait_chunk(
                            s_chunks_ready, s_chunks_done, s_chunks_read, s_chunk_ranges, owner, idx, first, count));
                    }
                    first = __shfl_sync(0xffffffffu, first, 0);
                    count = __shfl_sync(0xffffffffu, count, 0);
                    state = __shfl_sync(0xffffffffu, state, 0);
                    __syncwarp();
                    if (state == static_cast<int>(PullChunkState::kFinished)) {
                        break;
                    }
                    const int offset = ticket % kPullDispatchChunkTokens;
                    const int end = min(offset + static_cast<int>(kSlotBatch), count);
                    for (int i = offset; i < end; ++i) process_slot(first + i);
                }
            }
        } else {
            const int num_recv = *num_recv_tokens_dev;
            const int num_waves = (num_recv + kNWarps - 1) / kNWarps;
            for (int wave = 0; wave < num_waves; ++wave) {
                const int token = gwarp * num_waves + wave;
                if (token >= num_recv) continue;
                process_slot(token);
            }
        }
        bulk_drain(lane);
    }

    __syncthreads(); // policy async stores must complete before the block exits.

    if constexpr (kFusedMap) {
        // DROP only: per_expert_counts_active[e]'s pre-scan estimate (dispatch_pull_map_warp's
        // Phase 0) and expert_cursor[e]'s live per-slot tally can disagree under a real
        // FLAT-vs-EM-budget overflow, the same way local_permute_dup's phantom-row fixup
        // handles it on the push side. Elect the last-arriving block (expert_cursor's
        // trailing slot is the arrival counter) to zero the gap in this call's own buffer
        // and correct per_expert_counts_active[e] in place.
        if (allow_overflow_drop && expert_cursor != nullptr &&
            elect_last_block(expert_cursor + experts_per_rank, static_cast<int>(gridDim.x))) {
            const int total_warps = PullWarps + kPullDispatchPadWarps;
            for (int e = warp_id; e < experts_per_rank; e += total_warps) {
                const int32_t active = per_expert_counts_active[e];
                const int32_t delivered = expert_cursor[e];
                const int32_t true_active = delivered < active ? delivered : active;
                if (true_active >= active) continue;
                if (lane == 0) per_expert_counts_active[e] = true_active;
                const int64_t zone_start = expert_token_offsets[e];
                for (int64_t slot = zone_start + true_active; slot < zone_start + active; slot++) {
                    int4* row = dst_int4 + static_cast<size_t>(slot) * HiddenInt4;
                    for (int j = lane; j < HiddenInt4; j += kPullDispatchThreadsPerSlot) {
                        nccl_ep::st_cg_global(&row[j], int4{0, 0, 0, 0});
                    }
                    if (recv_topk_weights_em != nullptr && lane == 0) recv_topk_weights_em[slot] = 0.0f;
                    if constexpr (kFwd) {
                        int4* srow = reinterpret_cast<int4*>(
                            recv_x_scale_em + static_cast<size_t>(slot) * scale_row_bytes);
                        for (int j = lane; j < scale_row_int4; j += kPullDispatchThreadsPerSlot) {
                            nccl_ep::st_cg_global(&srow[j], int4{0, 0, 0, 0});
                        }
                    }
                }
            }
        }
    }

    // Tail sync: all ranks finished reading peer source before it can be reused.
    // Skipped when a separate tail-sync kernel runs it.
    if (!unfused_sync) lsa_grid_tail_barrier(dcomm, grid_barrier_counter, head_sync_flag);
}

// Quantize one lane's kVec reduced BF16 int4s (u-th at j0 + u * kJStride) to MXFP8: 8 FP8 bytes at
// payload_dst0[u * kJStride] and the E8M0 byte at scale_row[j >> 2]. A 32-element block is 4
// consecutive lanes, so amax reduces over lane distances 1 and 2. All lanes must call (shuffles);
// valid[u] only gates the stores.
template <int kVec, int kPairs, int kJStride>
__device__ __forceinline__ void mxfp8_quantize_tile(
    const float2 (&acc)[kVec][kPairs], const bool (&valid)[kVec], int j0, uint2* payload_dst0,
    uint8_t* scale_row) {
    static_assert(kPairs * 2 == static_cast<int>(sizeof(uint2)), "one int4 of BF16 narrows into a uint2");
    float amax[kVec];
#pragma unroll
    for (int u = 0; u < kVec; u++) {
        amax[u] = 0.0f;
        if (valid[u]) {
#pragma unroll
            for (int p = 0; p < kPairs; p++) {
                amax[u] = fmaxf(amax[u], fmaxf(fabsf(acc[u][p].x), fabsf(acc[u][p].y)));
            }
        }
    }
#pragma unroll
    for (int u = 0; u < kVec; u++) {
        amax[u] = fmaxf(amax[u], __shfl_xor_sync(0xFFFFFFFFu, amax[u], 1));
    }
#pragma unroll
    for (int u = 0; u < kVec; u++) {
        amax[u] = fmaxf(amax[u], __shfl_xor_sync(0xFFFFFFFFu, amax[u], 2));
    }
#pragma unroll
    for (int u = 0; u < kVec; u++) {
        if (!valid[u]) {
            continue;
        }
        const int j = j0 + u * kJStride;
        const uint8_t e8m0 = nccl_ep::mxfp8::float_to_e8m0(amax[u]);
        if ((j & 3) == 0) {
            scale_row[j >> 2] = e8m0;
        }
        const float scale_inv = nccl_ep::mxfp8::e8m0_to_scale_inv(e8m0);
        uint2 out;
        uint16_t* out_pairs = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
        for (int p = 0; p < kPairs; p++) {
            out_pairs[p] = nccl_ep::mxfp8::pack_e4m3x2_scaled(acc[u][p], scale_inv);
        }
        payload_dst0[u * kJStride] = out;
    }
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
                    // The FP8 payload for int4 j is 8 B at byte offset j*8.
                    mxfp8_quantize_tile<kHiddenVec, kPairs, kThreadsPerSlot>(
                        acc, valid_u, js[0], reinterpret_cast<uint2*>(out_row) + js[0], out_row + kHidden);
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

// Push staging stride. The packed MXFP8 record is rounded up to 128 B so every pushed row starts
// line-aligned; other recipes use the token row.
__host__ __device__ constexpr int comb_stage_push_stride_bytes(int row_bytes, bool mxfp8, bool reserve_prob) {
    return comb_stage_stride_bytes(mxfp8 ? nccl_ep::align(row_bytes, 128) : row_bytes, reserve_prob);
}

// ============================================================================
// Push EM combine (HT + EM + MNNVL). Mirror of pull EM dispatch: each
// expert rank locally gathers+reduces its K em copies of a recv token and PUSHES
// the reduced row into the destination attn-rank's staging over NVLink; a final
// combine_reduce sums the team_size partials per attn token. Head/tail intra-LSA syncs
// are fused into the push kernel. Forward reduces tokens; backward (BACKWARD_COMBINE)
// also scatters input weights into peer prob staging. NONE recipe, single LSA team only.
// ============================================================================

// Exact n / divisor for 0 <= n < 2^31 by multiply-high; g < 0 falls back to decode_src.
struct tokens_fast_div_t {
    uint32_t divisor;
    uint32_t magic;
    uint32_t shift;
};
__host__ __device__ inline tokens_fast_div_t make_tokens_fast_div(uint32_t divisor) {
    uint32_t shift = 0;
    while ((1ull << shift) < divisor) ++shift;
    return {divisor, static_cast<uint32_t>(((1ull << 32) * ((1ull << shift) - divisor)) / divisor + 1), shift};
}
__device__ __forceinline__ decoded_src_t decode_src_fast(int32_t g, const tokens_fast_div_t& fd, int lsa_team_size) {
    if (g < 0) return decode_src(g, static_cast<int>(fd.divisor), lsa_team_size);
    const uint32_t n = static_cast<uint32_t>(g);
    const uint32_t q = (__umulhi(n, fd.magic) + n) >> fd.shift;
    const uint32_t team = static_cast<uint32_t>(lsa_team_size);
    return {static_cast<int>(q < team ? q : q % team), static_cast<int>(n - q * fd.divisor)};
}

struct combine_push_param_t {
    bool flat;
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
    ncclDevComm_t* dcomm;
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
    // Rows are claimed dynamically only when set; otherwise each warp walks a static range.
    bool claim_dynamic;
    // Byte offset to the per-row pushed flags (0 = off); see combine_reduce_param_t::retained_slots.
    size_t row_flag_offset;
};

// Unpack one int4 of BF16 into four float pairs by bit placement; fewer instructions than
// ld_token_pair<ncclBfloat16> in the fused quantize loop.
__device__ __forceinline__ void combine_push_unpack_bf16x8(const int4& v, float2 (&out)[4]) {
    const uint32_t w[4] = {static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), static_cast<uint32_t>(v.z),
                           static_cast<uint32_t>(v.w)};
#pragma unroll
    for (int p = 0; p < 4; p++) {
        out[p] = make_float2(__uint_as_float(w[p] << 16), __uint_as_float(w[p] & 0xFFFF0000u));
    }
}

// Lane-rotated int4 index within a scale block. Lanes are 64 B apart, so in-order access makes
// 8 lanes of a 128-bit access phase share 2 bank groups (4-way conflict); rot = (lane / 2) % 4
// gives them distinct banks.
__device__ __forceinline__ int combine_push_rot_idx(int q, int rot) {
    return (q + rot) & 3;
}

// Fused MXFP8 push: sum one 32-element scale block (four int4) over a row's copies. Copy 0 is
// read from the smem row buffer, the other copies from global.
template <int MaxTopK, int HiddenInt4>
__device__ __forceinline__ void combine_push_sum_block(
    float2 (&acc)[4][4],
    const uint8_t* __restrict__ row,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ f2e,
    int n,
    int b,
    int rot) {
    // Copies loaded before they are summed: enough loads in flight to hide global latency, while the
    // 8 int4 held per batch plus the 32-register accumulator stay in the per-thread register budget.
    constexpr int kCopyBatch = 2;
#pragma unroll
    for (int q = 0; q < 4; q++) {
#pragma unroll
        for (int p = 0; p < 4; p++) {
            acc[q][p] = make_float2(0.0f, 0.0f);
        }
    }
    if (n > 0) {
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int4 v = *reinterpret_cast<const int4*>(row + static_cast<size_t>(b) * 64 + combine_push_rot_idx(q, rot) * 16);
            combine_push_unpack_bf16x8(v, acc[q]);
        }
    }
#pragma unroll
    for (int kb = 1; kb < MaxTopK; kb += kCopyBatch) {
        if (kb >= n) {
            break;
        }
        int4 buf[kCopyBatch][4];
#pragma unroll
        for (int kk = 0; kk < kCopyBatch; kk++) {
            if (kb + kk >= n || kb + kk >= MaxTopK) {
                continue;
            }
            const uint8_t* src = recv_x_em + static_cast<size_t>(f2e[kb + kk]) * (HiddenInt4 * 16) +
                                 static_cast<size_t>(b) * 64;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                buf[kk][q] = *reinterpret_cast<const int4*>(src + combine_push_rot_idx(q, rot) * 16);
            }
        }
#pragma unroll
        for (int kk = 0; kk < kCopyBatch; kk++) {
            if (kb + kk >= n || kb + kk >= MaxTopK) {
                continue;
            }
#pragma unroll
            for (int q = 0; q < 4; q++) {
                float2 f[4];
                combine_push_unpack_bf16x8(buf[kk][q], f);
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    acc[q][p].x += f[p].x;
                    acc[q][p].y += f[p].y;
                }
            }
        }
    }
}

// Quantize one summed scale block in place: FP8 bytes go to row + 32 * b (over BF16 bytes already
// consumed) and the E8M0 byte to scratch[b]. Lane-rotated chunk order avoids bank conflicts.
__device__ __forceinline__ void combine_push_quant_block(
    const float2 (&acc)[4][4], uint8_t* __restrict__ row, uint8_t* __restrict__ scratch, int b, int rot) {
    float amax = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        amax = fmaxf(amax, nccl_ep::mxfp8::amax_x8(acc[q]));
    }
    const uint8_t e8m0 = nccl_ep::mxfp8::float_to_e8m0(amax);
    scratch[b] = e8m0;
    const float scale_inv = nccl_ep::mxfp8::e8m0_to_scale_inv(e8m0);
#pragma unroll
    for (int q = 0; q < 4; q++) {
        *reinterpret_cast<uint2*>(row + static_cast<size_t>(b) * 32 + combine_push_rot_idx(q, rot) * 8) =
            nccl_ep::mxfp8::pack_e4m3x8_scaled(acc[q], scale_inv);
    }
}

// Fused MXFP8 push row: copy 0 is TMA-loaded into the row buffer and quantized in place, one scale
// block per lane per round. Scales stage in a scratch behind the row and move after the FP8 bytes.
template <int MaxTopK, int HiddenInt4>
__device__ __forceinline__ void combine_push_quant_row(
    uint8_t* __restrict__ row,
    uint64_t* mbar,
    uint32_t& mbar_phase,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ f2e,
    int n,
    int wlane) {
    using Mxfp8Traits = nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>;
    constexpr int kElemsPerInt4 = 16 / nccl_ep::size_u8<ncclBfloat16>();
    constexpr int kRowBytes = HiddenInt4 * 16;
    constexpr int kHidden = HiddenInt4 * kElemsPerInt4;
    constexpr int kBlocks = kHidden / Mxfp8Traits::kScaleBlock;
    constexpr int kScaleBytes = Mxfp8Traits::scale_bytes_per_token(kHidden);
    static_assert(Mxfp8Traits::kScaleBlock == 4 * kElemsPerInt4, "a scale block spans four int4");
    static_assert(HiddenInt4 % 4 == 0, "hidden is a whole number of scale blocks");
    constexpr int kRounds = nccl_ep::ceil_div(kBlocks, 32);
    uint8_t* const scratch = row + kRowBytes;
    if (n > 0) {
        if (wlane == 0) {
            nccl_ep::mbarrier_arrive_and_expect_tx(mbar, kRowBytes);
            nccl_ep::tma_load_1d(row, recv_x_em + static_cast<size_t>(f2e[0]) * kRowBytes, mbar, kRowBytes);
        }
        nccl_ep::mbarrier_wait(mbar, mbar_phase);
    }
    const int rot = (wlane >> 1) & 3;
    for (int m = 0; m < kRounds; m++) {
        const int b = m * 32 + wlane;
        float2 acc[4][4];
        if (b < kBlocks) {
            combine_push_sum_block<MaxTopK, HiddenInt4>(acc, row, recv_x_em, f2e, n, b, rot);
        }
        __syncwarp();  // every lane has read this round before its output overwrites it
        if (b < kBlocks) {
            combine_push_quant_block(acc, row, scratch, b, rot);
        }
    }
    __syncwarp();
    for (int i = wlane; i < kScaleBytes; i += 32) {
        row[kHidden + i] = scratch[i];
    }
}

// Row claiming for combine_push. Dynamic: a block claims rows of its own region one by one (shared
// counter); the last 1/kPoolDiv of every region is a grid-wide pool claimed by ticket t, which maps to
// region (t % grid) at pool position (t / grid). Rows up to kPoolMinRowBytes skip the pool. Static:
// each warp walks a contiguous range. Both rotate regions by rank so concurrent senders start on
// different destinations.
template <bool kDynamic>
struct combine_push_claim_t {
    static constexpr int kPoolDiv = 4;
    static constexpr int kPoolMinRowBytes = 8192;
    int num_recv;
    int grid_blocks;
    int num_per_block;
    int own_per_block;
    int rot_offset;
    int own_start;
    int own_end;
    int num_tickets;
    bool own_done;
    int static_next;
    int static_end;

    __device__ __forceinline__ void init(int num_recv_in, int my_lsa_rank, int lsa_team_size, int row_bytes,
                                         int slots_per_block, int slot_in_block) {
        num_recv = num_recv_in;
        grid_blocks = static_cast<int>(gridDim.x);
        if constexpr (!kDynamic) {
            const int n_sub = slots_per_block * grid_blocks;
            const int warp_idx = slot_in_block * grid_blocks + static_cast<int>(blockIdx.x);
            const int per_warp = nccl_ep::ceil_div(num_recv, n_sub);
            static_next = ((warp_idx + my_lsa_rank * (n_sub / lsa_team_size)) % n_sub) * per_warp;
            static_end = min(static_next + per_warp, num_recv);
            return;
        }
        num_per_block = nccl_ep::ceil_div(num_recv, grid_blocks);
        const int pool_per_block = row_bytes > kPoolMinRowBytes ? num_per_block / kPoolDiv : 0;
        own_per_block = num_per_block - pool_per_block;
        rot_offset = my_lsa_rank * (grid_blocks / lsa_team_size);
        own_start = ((static_cast<int>(blockIdx.x) + rot_offset) % grid_blocks) * num_per_block;
        own_end = min(own_start + own_per_block, num_recv);
        num_tickets = grid_blocks * pool_per_block;
        own_done = false;
    }

    // Warp-uniform. Returns the next row for this warp, or num_recv when none are left.
    __device__ __forceinline__ int next(int* s_claim, uint32_t* pool_counter, int wlane) {
        if constexpr (!kDynamic) {
            return static_next < static_end ? static_next++ : num_recv;
        }
        if (!own_done) {
            int idx = 0;
            if (wlane == 0) {
                idx = atomicAdd(s_claim, 1);
            }
            idx = __shfl_sync(0xFFFFFFFFu, idx, 0);
            if (own_start + idx < own_end) {
                return own_start + idx;
            }
            own_done = true;
        }
        while (true) {
            int ticket = 0;
            if (wlane == 0) {
                ticket = static_cast<int>(atomicAdd(pool_counter, 1u));
            }
            ticket = __shfl_sync(0xFFFFFFFFu, ticket, 0);
            if (ticket >= num_tickets) {
                return num_recv;
            }
            const int region = (ticket % grid_blocks + rot_offset) % grid_blocks;
            const int claimed = region * num_per_block + own_per_block + ticket / grid_blocks;
            if (claimed < num_recv) {
                return claimed;
            }
        }
    }
};

// Overflow drop: mark row dst_row of a peer's staging pushed; the tail sync orders it before combine_reduce.
__device__ __forceinline__ void mark_row_pushed(void* peer_staging, size_t row_flag_offset, int64_t dst_row) {
    reinterpret_cast<int32_t*>(static_cast<uint8_t*>(peer_staging) + row_flag_offset)[dst_row] = 0;
}

// Drop mode: write every pad float so the reduce never reads a stale one.
__device__ __forceinline__ void write_drop_prob_pad(
    float* prob_dst, int32_t prob_p, float prob_w, int top_k, int wlane) {
    float w = 0.0f;
    for (int j = 0; j < top_k; ++j) {
        const int32_t pp = __shfl_sync(0xFFFFFFFFu, prob_p, j);
        const float pw = __shfl_sync(0xFFFFFFFFu, prob_w, j);
        if (pp == wlane) {
            w = pw;
        }
    }
    if (wlane < top_k) {
        prob_dst[wlane] = w;
    }
}

// Re-arm the read pushed-row flag (-1) for the next call.
__device__ __forceinline__ void rearm_row_flag(int32_t* flags, int t, int lsa_team_size, int src_rank) {
    flags[static_cast<size_t>(t) * lsa_team_size + src_rank] = -1;
}

// Adds one int4 of kTokenDtype elements into the FP32 pair accumulator of one combine_push lane.
template <ncclDataType_t kTokenDtype>
__device__ __forceinline__ void combine_push_acc_add(float2 (&acc)[4], const int4& v) {
    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
    for (int p = 0; p < kPairs; p++) {
        const float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&v, p);
        acc[p].x += f.x;
        acc[p].y += f.y;
    }
}

// Per-warp TMA load ring of the multi-chunk one-tile combine_push. Unit u of a row with num_copies
// copies is (chunk u / num_copies, copy u % num_copies); it is loaded into stage u % kRing, kRing
// units ahead of the reduce. stage_parity_bits holds one parity bit per stage and persists across rows.
template <int HiddenInt4, ncclDataType_t kTokenDtype, int kRing>
struct combine_push_ring_t {
    static constexpr int kStageInt4 = combine_push_tile_int4(HiddenInt4);
    static constexpr int kNumChunks = combine_push_ntiles(HiddenInt4);
    const uint8_t* recv_x_em;
    int4* buf;      // kRing stages of kStageInt4 int4
    uint64_t* bar;  // one mbarrier per stage
    uint32_t stage_parity_bits;
    const int32_t* em_rows;  // this warp's smem map of packed EM row indices, rewritten per claimed row
    int num_copies;
    int num_transfers;
    int stage;

    __device__ __forceinline__ void init(const uint8_t* x, int4* ring_buf, uint64_t* ring_bar,
                                         const int32_t* warp_em_rows, int wlane) {
        recv_x_em = x;
        buf = ring_buf;
        bar = ring_bar;
        em_rows = warp_em_rows;
        stage_parity_bits = 0;
        if (wlane < kRing) {
            nccl_ep::mbarrier_init(&bar[wlane], 1);
        }
        nccl_ep::fence_barrier_init();
    }

    // The last chunk of a row that is not a whole number of tiles is a short transfer.
    __device__ __forceinline__ void issue(int u, int s) const {
        const int c = u / num_copies;
        const int first_int4 = c * kStageInt4;
        const int bytes = min(kStageInt4, HiddenInt4 - first_int4) * 16;
        const uint8_t* src = recv_x_em + static_cast<size_t>(em_rows[u - c * num_copies]) * (HiddenInt4 * 16) +
                             static_cast<size_t>(first_int4) * 16;
        nccl_ep::mbarrier_arrive_and_expect_tx(&bar[s], bytes);
        nccl_ep::tma_load_1d(buf + static_cast<size_t>(s) * kStageInt4, src, &bar[s], bytes, true);
    }

    // Starts a row of row_copies copies, already packed in em_rows: issues its first kRing units.
    __device__ __forceinline__ void start_row(int row_copies, int wlane) {
        num_copies = row_copies;
        num_transfers = kNumChunks * row_copies;
        stage = 0;
        if (wlane == 0) {
            for (int u = 0; u < kRing && u < num_transfers; u++) {
                issue(u, u);
            }
        }
    }

    // Sums the copies of one chunk into acc in ascending order, refilling each stage once read.
    template <int kHiddenVec, class SrcJ, class TileValid>
    __device__ __forceinline__ void reduce(float2 (&acc)[kHiddenVec][4], int chunk, int base_j, int wlane,
                                           SrcJ src_j, TileValid tile_valid) {
        for (int k = 0; k < num_copies; k++) {
            nccl_ep::mbarrier_wait<true>(&bar[stage], stage_parity_bits, stage);
            const int4* const b = buf + static_cast<size_t>(stage) * kStageInt4;
#pragma unroll
            for (int u = 0; u < kHiddenVec; u++) {
                if (!tile_valid(chunk, u)) {
                    continue;
                }
                combine_push_acc_add<kTokenDtype>(acc[u], b[src_j(chunk, u) - base_j]);
            }
            // Order generic-proxy reads before lane 0 refills the stage with an async TMA load.
            nccl_ep::tma_store_fence<true>();
            __syncwarp();
            const int next_u = chunk * num_copies + k + kRing;
            if (wlane == 0 && next_u < num_transfers) {
                issue(next_u, stage);
            }
            stage = stage + 1 == kRing ? 0 : stage + 1;
        }
    }
};

// One recv slot per warp (kSlotsPerBlock in parallel), identical
// gather-reduce to local_permute_reduce; the reduced row is assembled in a per-warp smem buffer
// and pushed to the remote destination staging slot [t*team_size + R], one TMA store per row
// chunk (see combine_push_buf_int4).
//   kChunkSize - chunk width in int4 (combine_push_chunk_size, resolved host-side in
//                launch_combine_push); a chunk covering the whole row takes the single-chunk path,
//                whose prefetch-pipelined reduce needs extra registers. The multi-chunk path stays
//                register-frugal and relies on high occupancy to hide gather latency.
//   kCombineRecipe - NONE pushes the reduced caller-dtype row; MXFP8 (whole-row, BF16) quantizes it
//                in place in the smem row buffer and pushes the packed [FP8 H | E8M0 H/32] row.
//   kRing - chunked one-tile path only (combine_push_ring_depth): TMA-load source chunks into kRing
//                smem stages ahead of the reduce instead of LDG gathers. 0 keeps the LDG gather.
template <int MaxTopK, int HiddenInt4, ncclDataType_t kTokenDtype, ncclEpCombQuant_t kCombineRecipe,
          bool BACKWARD_COMBINE = false, int kChunkSize = 0, bool kClaimDynamic, int kRing>
__device__ __forceinline__ void combine_push(
    void* const* __restrict__ peer_staging_ptrs,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ recv_slot_to_src,
    const int32_t* __restrict__ num_recv_tokens_dev,
    ncclDevComm_t* __restrict__ dcomm,
    uint32_t* __restrict__ head_sync_flag,
    uint32_t* __restrict__ grid_barrier_counter,
    int top_k,
    int /*row_bytes*/,
    int caller_num_recv_tokens,
    int my_lsa_rank,
    int tokens_per_rank,
    int lsa_team_size,
    size_t row_flag_offset,
    // Backward-only: sparse-direct prob scatter into the token staging pad (null/unused on forward).
    const float* __restrict__ topk_weights_em = nullptr,
    const int32_t* __restrict__ srcpos_map = nullptr,
    bool unfused_sync = false,
    uint8_t* __restrict__ smem_bytes = nullptr) {
    constexpr int kRowBytes = HiddenInt4 * 16;
    constexpr bool kMxfp8 = (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8);
    static_assert(!kMxfp8 || kTokenDtype == ncclBfloat16, "MXFP8 push combine reduces BF16 rows");
    // Wire row pushed to the peer staging: the caller row for NONE, packed [FP8 H | E8M0 H/32] for MXFP8.
    using Mxfp8Traits = nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>;
    constexpr int kHidden = HiddenInt4 * 8;
    constexpr int kPushRowBytes = kMxfp8 ? Mxfp8Traits::template packed_bytes<ncclBfloat16>(kHidden) : kRowBytes;
    static_assert(!kMxfp8 || (kPushRowBytes % 16 == 0 && HiddenInt4 % 64 == 0),
                  "MXFP8 push row must be 16B-aligned (hidden multiple of 512)");
    static_assert(!kMxfp8 || Mxfp8Traits::kScaleBlock == 32, "MXFP8 scale blocks span four int4");
    static_assert(!kMxfp8 || kChunkSize == 0, "MXFP8 push is whole-row");
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
    if (!unfused_sync) lsa_grid_head_gate(dcomm, head_sync_flag);

    // Metadata smem is sized to the max block; only the first kSlotsPerBlock rows are used.
    __shared__ int32_t smem_flat2em_slot_map[kCombinePushMaxSlots][MaxTopK];
    __shared__ int s_nvalid[kCombinePushMaxSlots];
    __shared__ int4* s_dst[kCombinePushMaxSlots];
    constexpr bool kSingleChunk = kMxfp8 || combine_push_num_chunks(HiddenInt4, kChunkSize) == 1;
    // Per-warp mbarriers: one for the single-chunk path's TMA-load source relay (n==1 fast path, see
    // below) or kRing for the chunked load ring. A specialization using neither still reserves one
    // word per slot (see combine_push_slots_per_block).
    constexpr int kWarpMbars = kSingleChunk || kRing == 0 ? 1 : kRing;
    __shared__ uint64_t s_mbar[kCombinePushMaxSlots * kWarpMbars];
    // Dynamic smem: one per-warp chunk buffer for the TMA push (kBufInt4 int4 each), then the
    // per-warp load ring (kRing stages of kBufInt4 int4).
    int4* const s_rows = reinterpret_cast<int4*>(smem_bytes);
    // Backward co-locates the prob in the token staging row's trailing pad (offset kRowBytes),
    // so no separate prob destination is needed.

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
    const int wlane = tid % kThreadsPerSlot;
    // This warp's push staging buffer (one chunk of the row).
    constexpr int kBufInt4 =
        kMxfp8 ? combine_push_mxfp8_fused_int4(HiddenInt4) : combine_push_buf_int4(HiddenInt4, kChunkSize);
    static_assert(kRing == 0 || (!kMxfp8 && combine_push_ring_supported(HiddenInt4, kChunkSize)),
                  "load ring needs the multi-chunk one-tile path");
    int4* const my_row = s_rows + static_cast<size_t>(slot_in_block) * kBufInt4;

    const int num_recv = *num_recv_tokens_dev;
    // Rows are claimed dynamically (see combine_push_claim_t). The pool counter is the
    // combine_head_sync_counter word (idle in pull/push), cleared by lsa_grid_tail_barrier.
    __shared__ int s_claim;
    // The static smem must fit what combine_push_slots_per_block and combine_push_ring_depth reserve.
    static_assert(sizeof(smem_flat2em_slot_map) + sizeof(s_nvalid) + sizeof(s_dst) + sizeof(s_mbar) +
                          sizeof(s_claim) <=
                      combine_push_meta_reserve_bytes() + combine_push_ring_bar_bytes(kRing),
                  "combine_push static smem exceeds its sizing reserve");
    uint32_t* const pool_counter = grid_barrier_counter + 1;
    combine_push_claim_t<kClaimDynamic> claim;
    claim.init(num_recv, my_lsa_rank, lsa_team_size, kRowBytes, kSlotsPerBlock, slot_in_block);

    // TMA-load relay: init this warp's mbarrier once (1 arrival: the load's complete_tx).
    uint32_t mbar_phase = 0;
    if constexpr (kSingleChunk) {
        if (wlane == 0) nccl_ep::mbarrier_init(&s_mbar[slot_in_block], 1);
        nccl_ep::fence_barrier_init();
    }
    combine_push_ring_t<HiddenInt4, kTokenDtype, kRing> ring;
    if constexpr (kRing > 0) {
        ring.init(recv_x_em,
                  s_rows + static_cast<size_t>(kSlotsPerBlock) * kBufInt4 +
                      static_cast<size_t>(slot_in_block) * kRing * kBufInt4,
                  s_mbar + static_cast<size_t>(slot_in_block) * kRing, smem_flat2em_slot_map[slot_in_block], wlane);
    }
    if (tid == 0) {
        s_claim = 0;
    }
    __syncthreads();

    const tokens_fast_div_t tokens_div = make_tokens_fast_div(static_cast<uint32_t>(tokens_per_rank));
    // Next slot's routing metadata is loaded one wave ahead to hide its latency.
    int32_t next_s = -1;
    int32_t next_g = 0;
    auto load_meta = [&](int sl, bool want) {
        const bool ok = want && sl < num_recv;
        next_s = (ok && wlane < top_k) ? __ldg(flat2em_slot_map + static_cast<size_t>(sl) * top_k + wlane) : -1;
        next_g = (ok && wlane == 0) ? __ldg(recv_slot_to_src + sl) : 0;
    };
    int slot = claim.next(&s_claim, pool_counter, wlane);
    int next_slot = 0;
    load_meta(slot, slot < num_recv);

    for (; slot < num_recv; slot = next_slot) {
        const bool slot_valid = (slot < num_recv);

        static_assert(MaxTopK <= 32, "cooperative pack assumes MaxTopK <= 32");
        // Backward: per-copy prob metadata (srcpos + input weight), prefetched in the metadata
        // phase but stored only after the token push loop so its load latency hides behind the
        // NVLink token writes. Carried at wave scope across both blocks.
        int32_t prob_p = -1;
        float prob_w = 0.0f;
        if (slot_valid && wlane < 32) {
            const int32_t s = next_s;
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
                const decoded_src_t dst_id = decode_src_fast(next_g, tokens_div, lsa_team_size);
                const int dst_rank = dst_id.rank;
                const int t = dst_id.token;
                const int64_t dst_row = static_cast<int64_t>(t) * lsa_team_size + my_lsa_rank;
                s_dst[slot_in_block] = reinterpret_cast<int4*>(
                    static_cast<uint8_t*>(peer_staging_ptrs[dst_rank]) +
                    static_cast<size_t>(dst_row) * comb_stage_push_stride_bytes(kPushRowBytes, kMxfp8, BACKWARD_COMBINE));
                if (row_flag_offset != 0) {
                    mark_row_pushed(peer_staging_ptrs[dst_rank], row_flag_offset, dst_row);
                }
            }
        }
        __syncwarp();
        next_slot = claim.next(&s_claim, pool_counter, wlane);
        load_meta(next_slot, next_slot < num_recv);

        if (slot_valid) {
            const int n = s_nvalid[slot_in_block];
            int4* dst_int4 = s_dst[slot_in_block];

            constexpr int kHiddenVec = (kElemsPerThread < 4) ? kElemsPerThread : 4;
            constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
            constexpr int kNTiles = (kElemsPerThread + kHiddenVec - 1) / kHiddenVec;

            // Wait for this warp's prior push to drain before reusing its single row buffer. The ring
            // path waits just before writing the buffer, so its loads overlap the drain.
            if constexpr (kRing == 0) {
                if (wlane == 0) {
                    nccl_ep::tma_store_wait<0>();
                }
                __syncwarp();
            }
            auto src_j = [&](int tile, int u) { return wlane + (tile * kHiddenVec + u) * kThreadsPerSlot; };
            // Bound the row index too: when HiddenInt4 < kThreadsPerSlot the extra lanes own no element.
            auto tile_valid = [&](int tile, int u) {
                return (tile * kHiddenVec + u) < kElemsPerThread && src_j(tile, u) < HiddenInt4;
            };

            // First int4 index of tile t, clamped to the row width.
            auto tile_j = [&](int t) { const int j = t * kHiddenVec * kThreadsPerSlot; return j < HiddenInt4 ? j : HiddenInt4; };

            // Chunked staging, kNumChunks iterations per token:
            //   - reduce the chunk's tiles into the smem buffer (n copies summed per tile)
            //   - push the chunk with one TMA store
            //   - reuse the same buffer for the next chunk
            // No copy-0 prefetch: at this occupancy the warps hide the gather latency, and dropping
            // it frees registers for the tighter per-thread budget. The reuse-wait (tma_store_wait
            // below) is hidden by other warps' concurrent gathers at this occupancy, not by this
            // warp's own next chunk. n==1 folds through the same reduce (lossless for finite values;
            // NaN payload/-0 sign are not preserved through the float round-trip).
            // combine_push_tile_int4 must match the tile size the kernel computes here.
            static_assert(kHiddenVec * kThreadsPerSlot == combine_push_tile_int4(HiddenInt4));
            static_assert(kNTiles == combine_push_ntiles(HiddenInt4));
            constexpr int kChunkTiles = kBufInt4 / (kHiddenVec * kThreadsPerSlot);
            constexpr int kNumChunks = combine_push_num_chunks(HiddenInt4, kChunkSize);
            if constexpr (kSingleChunk) {
                // Single-copy relay: TMA-load the one source row straight into the smem buffer,
                // bypassing the register round-trip below entirely. n==1 dominates at high EP
                // (mean valid-copies per slot ~= top_k / lsa_team_size), so this matters more as
                // EP grows even though it's a small win at low EP.
                bool filled = false;
                if constexpr (kMxfp8) {
                    // Copy 0 is TMA-loaded and quantized in place; the other copies are summed in.
                    combine_push_quant_row<MaxTopK, HiddenInt4>(
                        reinterpret_cast<uint8_t*>(my_row), &s_mbar[slot_in_block], mbar_phase, recv_x_em,
                        smem_flat2em_slot_map[slot_in_block], n, wlane);
                    filled = true;
                } else if (n == 1) {
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
                // Whole row in one chunk: no multi-chunk occupancy to hide the gather latency,
                // so pipeline copy 0's load with the previous tile's accumulate/store instead.
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
                for (int tile = 0; tile < kNTiles; tile++) {
                    float2 acc[kHiddenVec][4];
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++)
#pragma unroll
                        for (int p = 0; p < 4; p++) { acc[u][p].x = 0.0f; acc[u][p].y = 0.0f; }
                    if (n > 0) {
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
                    for (int k = 1; k < n; k++) {
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
                        if (!tile_valid(tile, u)) {
                            continue;
                        }
                        // n == 1 is handled by the TMA relay above, so n >= 2 here: always repack.
                        int4 out;
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                        }
                        my_row[src_j(tile, u)] = out;
                    }
                    int4* tmp = pf_cur; pf_cur = pf_nxt; pf_nxt = tmp;
                }
                }  // end if (!filled): scalar reduce path
                nccl_ep::tma_store_fence();
                __syncwarp();
                if (wlane == 0) {
                    nccl_ep::tma_store_1d(my_row, dst_int4, kPushRowBytes);
                }
            } else {
                if constexpr (kRing > 0) {
                    ring.start_row(n, wlane);
                }
            for (int chunk = 0; chunk < kNumChunks; chunk++) {
                const int tile0 = chunk * kChunkTiles;
                const int tile1 = tile0 + kChunkTiles < kNTiles ? tile0 + kChunkTiles : kNTiles;
                const int base_j = tile_j(tile0);
                const int end_j = tile_j(tile1);
                if constexpr (kRing == 0) {
                    if (chunk > 0 && wlane == 0) {
                        nccl_ep::tma_store_wait<0>();  // reuse buffer after prior chunk push
                    }
                    __syncwarp();
                }
                for (int tile = tile0; tile < tile1; tile++) {
                    const int32_t* const f2e = smem_flat2em_slot_map[slot_in_block];
                    float2 acc[kHiddenVec][4];
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++)
#pragma unroll
                        for (int p = 0; p < 4; p++) { acc[u][p].x = 0.0f; acc[u][p].y = 0.0f; }
                    if constexpr (kRing > 0) {
                        // One tile per chunk: the prior push drains here, behind the ring loads.
                        ring.reduce(acc, tile, base_j, wlane, src_j, tile_valid);
                        if (wlane == 0) {
                            nccl_ep::tma_store_wait<0>();
                        }
                        __syncwarp();
                    } else {
                        // Copies in ascending order; each batch's loads are all issued before any is summed.
                        constexpr int kCopyBatch = 8 / kHiddenVec;
#pragma unroll
                        for (int kb = 0; kb < MaxTopK; kb += kCopyBatch) {
                            if (kb > 0 && kb >= n) {
                                break;
                            }
                            int4 b[kCopyBatch][kHiddenVec];
#pragma unroll
                            for (int kk = 0; kk < kCopyBatch; kk++) {
                                if (kb + kk >= n) {
                                    continue;
                                }
                                const int4* src = reinterpret_cast<const int4*>(
                                    recv_x_em + static_cast<size_t>(f2e[kb + kk]) * kRowBytes);
#pragma unroll
                                for (int u = 0; u < kHiddenVec; u++) {
                                    if (tile_valid(tile, u)) {
                                        b[kk][u] = src[src_j(tile, u)];
                                    }
                                }
                            }
#pragma unroll
                            for (int kk = 0; kk < kCopyBatch; kk++) {
                                if (kb + kk >= n) {
                                    continue;
                                }
#pragma unroll
                                for (int u = 0; u < kHiddenVec; u++) {
                                    if (!tile_valid(tile, u)) {
                                        continue;
                                    }
                                    combine_push_acc_add<kTokenDtype>(acc[u], b[kk][u]);
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!tile_valid(tile, u)) {
                            continue;
                        }
                        int4 out;
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                        }
                        my_row[src_j(tile, u) - base_j] = out;
                    }
                }
                nccl_ep::tma_store_fence();
                __syncwarp();
                if (wlane == 0) {
                    nccl_ep::tma_store_1d(my_row, dst_int4 + base_j, (end_j - base_j) * 16);
                }
            }
            }

            // Backward: store the prefetched weight to its srcpos in this (t,R) token row's trailing
            // pad (co-located, one remote write per slot), after the token push has hidden the load.
            if constexpr (BACKWARD_COMBINE) {
                float* prob_dst = reinterpret_cast<float*>(
                    reinterpret_cast<uint8_t*>(dst_int4) + kPushRowBytes);
                if (row_flag_offset != 0) {
                    write_drop_prob_pad(prob_dst, prob_p, prob_w, top_k, wlane);
                } else if (prob_p >= 0) {
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
    if (!unfused_sync) lsa_grid_tail_barrier(dcomm, grid_barrier_counter, head_sync_flag);
}

struct combine_reduce_param_t {
    // Optional [source token, destination rank] map; negative = dropped, never read its staging row.
    // Flat: scan-built. Expert-major: pushed-row flags, re-armed by the reduce.
    int32_t* retained_slots;
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
template <int HiddenInt4, ncclDataType_t kTokenDtype, ncclEpCombQuant_t kCombineRecipe, bool BACKWARD_COMBINE,
          bool CheckRetained, bool Flat>
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
    float* __restrict__ combined_topk_weights,
    int top_k,
    int32_t* retained_slots) {
    constexpr int kRowBytes = HiddenInt4 * 16;
    constexpr bool kMxfp8 = (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8);
    static_assert(!kMxfp8 || (kTokenDtype == ncclBfloat16 && !Flat),
                  "MXFP8 push combine reduces BF16 rows in the expert-major layout");
    // Staging row as pushed by combine_push: packed [FP8 H | E8M0 H/32] for MXFP8.
    constexpr int kHidden = HiddenInt4 * 8;
    constexpr int kStageRowBytes = kMxfp8
        ? nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::template packed_bytes<ncclBfloat16>(kHidden)
        : kRowBytes;
    // Flat spreads small batches across CTAs with four warps per token.
    constexpr int kThreadsPerSlot = Flat
        ? kLocalPermuteReduceThreads / kLocalPermuteReduceSlotsPerBlock : 32;
    constexpr int kSlotsPerBlock = Flat ? kLocalPermuteReduceSlotsPerBlock : kCombineReduceSlotsPerBlock;
    constexpr int kElemsPerThread = nccl_ep::ceil_div(HiddenInt4, kThreadsPerSlot);
    constexpr int kFlatUnroll = kElemsPerThread < 4 ? kElemsPerThread : 4;
    constexpr int kUnroll = Flat ? kFlatUnroll : kCombineReduceUnroll;
    constexpr int kRankGroup = Flat ? 1 : kCombineReduceRankGroup;
    constexpr int kChunkInt4 = kUnroll * kThreadsPerSlot;
    constexpr int kNumChunks = nccl_ep::ceil_div(HiddenInt4, kChunkInt4);
    constexpr bool kExactChunks = (HiddenInt4 % kChunkInt4) == 0;
    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
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
        const int slot_thread = Flat ? tid % kThreadsPerSlot : wlane;
        const int e_id = (wlane < num_topk)
                             ? static_cast<int>(topk_idx[static_cast<size_t>(t) * num_topk + wlane])
                             : -1;
        const bool routed = (e_id >= 0) && (e_id < experts_per_lsa_team);
        bool rank_valid = routed;
        if constexpr (CheckRetained) {
            rank_valid = routed &&
                retained_slots[static_cast<size_t>(t) * lsa_team_size + e_id / experts_per_rank] >= 0;
        }
        const int src_rank_of_lane = rank_valid ? (e_id / experts_per_rank) : -1;
        const unsigned same_rank = __match_any_sync(0xFFFFFFFFu, src_rank_of_lane);
        const bool rank_leader = rank_valid && ((__ffs(static_cast<int>(same_rank)) - 1) == wlane);
        const unsigned present_ranks = __ballot_sync(0xFFFFFFFFu, rank_leader);
        if constexpr (CheckRetained && !Flat) {
            if (rank_leader) {
                rearm_row_flag(retained_slots, t, lsa_team_size, src_rank_of_lane);
            }
        }

        // Backward: srcpos slot `slot_thread` was pushed by the rank hosting that expert
        // (src_rank_of_lane), co-located in that rank's token staging row pad (offset kRowBytes).
        // Out-of-team / unrouted slots yield zero.
        if constexpr (BACKWARD_COMBINE) {
            if (slot_thread < top_k) {
                float w = 0.0f;
                if (src_rank_of_lane >= 0) {
                    const float* prob_row = reinterpret_cast<const float*>(
                        staging + (static_cast<size_t>(t) * lsa_team_size + src_rank_of_lane) *
                                      comb_stage_push_stride_bytes(kStageRowBytes, kMxfp8, /*reserve_prob=*/true) + kStageRowBytes);
                    w = prob_row[slot_thread];
                }
                combined_topk_weights[static_cast<size_t>(t) * top_k + slot_thread] = w;
            }
        }

        int4* dst_int4 = reinterpret_cast<int4*>(attn_output + static_cast<size_t>(t) * kRowBytes);
        const size_t base_row = static_cast<size_t>(t) * lsa_team_size;
        const size_t stage_stride = comb_stage_push_stride_bytes(kStageRowBytes, kMxfp8, BACKWARD_COMBINE);

#pragma unroll 1
        for (int c = 0; c < kNumChunks; c++) {
            int js[kUnroll];
            bool valid_u[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; u++) {
                js[u] = (c * kUnroll + u) * kThreadsPerSlot + slot_thread;
                valid_u[u] = kExactChunks || (js[u] < HiddenInt4);
            }

            float2 acc[kUnroll][kPairs];
#pragma unroll
            for (int u = 0; u < kUnroll; u++) {
#pragma unroll
                for (int p = 0; p < kPairs; p++) {
                    acc[u][p].x = 0.0f;
                    acc[u][p].y = 0.0f;
                }
            }

            unsigned bl = present_ranks;
            while (bl != 0u) {
                int4 buf[kRankGroup][kUnroll];
                uint2 qbuf[kRankGroup][kUnroll];
                float qscale[kRankGroup][kUnroll];
                bool have[kRankGroup];
#pragma unroll
                for (int g = 0; g < kRankGroup; g++) {
                    const int lead = __ffs(static_cast<int>(bl)) - 1;
                    have[g] = lead >= 0;
                    const int r = __shfl_sync(0xFFFFFFFFu, src_rank_of_lane, have[g] ? lead : 0);
                    bl &= (bl - 1u);
                    if (have[g]) {
                        const uint8_t* row = staging + (base_row + r) * stage_stride;
#pragma unroll
                        for (int u = 0; u < kUnroll; u++) {
                            if (!valid_u[u]) {
                                continue;
                            }
                            if constexpr (kMxfp8) {
                                qbuf[g][u] = reinterpret_cast<const uint2*>(row)[js[u]];
                                qscale[g][u] = nccl_ep::mxfp8::e8m0_to_scale(row[kHidden + (js[u] >> 2)]);
                            } else {
                                buf[g][u] = reinterpret_cast<const int4*>(row)[js[u]];
                            }
                        }
                    }
                }
#pragma unroll
                for (int g = 0; g < kRankGroup; g++) {
                    if (!have[g]) continue;
#pragma unroll
                    for (int u = 0; u < kUnroll; u++) {
                        if (!valid_u[u]) continue;
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            float2 f;
                            if constexpr (kMxfp8) {
                                f = nccl_ep::mxfp8::dequant_x2_scaled(
                                    reinterpret_cast<const uint16_t*>(&qbuf[g][u])[p], qscale[g][u]);
                            } else {
                                f = nccl_ep::ld_token_pair<kTokenDtype>(&buf[g][u], p);
                            }
                            acc[u][p].x += f.x;
                            acc[u][p].y += f.y;
                        }
                    }
                }
            }

#pragma unroll
            for (int u = 0; u < kUnroll; u++) {
                if (!valid_u[u]) continue;
                int4 out;
#pragma unroll
                for (int p = 0; p < kPairs; p++) {
                    nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                }
                dst_int4[js[u]] = out;
            }
        }
    }
}

// Counts occupy the prefix of each published row, followed by packed top-k.
// Flat appends [source chunk, destination rank] counts to the EM count prefix.
__device__ __forceinline__ const int32_t* flat_count_row(const dispatch_pull_param_t& p, int source) {
    if (p.cached_cnt_rows != nullptr) {
        return p.cached_cnt_rows + static_cast<size_t>(source) * (p.topk_off_bytes / sizeof(int32_t));
    }
    return reinterpret_cast<const int32_t*>(p.own_meta_staging +
        static_cast<size_t>(source) * pull_meta_stride(p.topk_off_bytes, p.tokens_per_rank));
}

template <bool FusedMap>
__device__ __forceinline__ void flat_dispatch_head(const dispatch_pull_param_t& p) {
    if (p.unfused_sync) {
        return;
    }
    if constexpr (FusedMap) {
        dispatch_pull_map_publish_grid(p.cached_cnt_rows ? nullptr : p.own_row, p.own_topk_snapshot,
            p.meta_ptrs, pull_meta_stride(p.topk_off_bytes, p.tokens_per_rank), p.topk_off_bytes,
            p.tokens_per_rank, p.top_k, p.my_rank, p.lsa_team_size);
        const bool leader = elect_last_block(reinterpret_cast<const int*>(p.grid_barrier_counter),
                                             static_cast<int>(gridDim.x));
        if (leader && threadIdx.x == 0) {
            atomicExch(p.grid_barrier_counter, 0u);
        }
        lsa_grid_head_gate_elected(p.dcomm, p.head_sync_flag, leader, 2);
    } else {
        lsa_grid_head_gate(p.dcomm, p.head_sync_flag);
    }
}

__device__ __forceinline__ int flat_receive_count(const dispatch_pull_param_t& p, int* shared_total) {
    if (threadIdx.x == 0) {
        int total = 0;
        for (int r = 0; r < p.lsa_team_size; ++r) {
            total += flat_count_row(p, r)[p.my_rank];
        }
        if (total > p.flat_recv_capacity && !p.allow_overflow_drop) {
            printf("ncclEpDispatch: flat pull recv tokens %d exceed capacity %d\n", total, p.flat_recv_capacity);
            __trap();
        }
        *shared_total = min(total, p.flat_recv_capacity);
        if (blockIdx.x == 0) {
            *p.num_recv_tokens_dev = *shared_total;
            if (p.caller_recv_total != nullptr) {
                if (p.caller_out_is_int64) {
                    *static_cast<int64_t*>(p.caller_recv_total) = total;
                } else {
                    *static_cast<int32_t*>(p.caller_recv_total) = total;
                }
            }
        }
    }
    __syncthreads();
    return *shared_total;
}

// Warp-collective prefix for a source chunk at one destination. Chunk counts
// preserve source-token order without per-token global atomics or a grid barrier.
__device__ __forceinline__ int flat_chunk_base(
    const dispatch_pull_param_t& p, int source, int chunk, int destination, int lane) {
    int prefix = 0;
    for (int r = lane; r < source; r += 32) {
        prefix += flat_count_row(p, r)[destination];
    }
    const int32_t* chunks = flat_count_row(p, source) + p.lsa_team_size * (p.experts_per_rank + 1);
    for (int c = lane; c < chunk; c += 32) {
        prefix += chunks[static_cast<size_t>(c) * p.lsa_team_size + destination];
    }
    for (int d = 16; d > 0; d /= 2) {
        prefix += __shfl_down_sync(0xffffffffu, prefix, d);
    }
    return __shfl_sync(0xffffffffu, prefix, 0);
}

template <int Warps>
__device__ __forceinline__ void flat_map_barrier() {
    // Only MAP threads participate. Order both shared prefix scratch and global
    // routing writes before the leader publishes readiness to payload warps.
    nccl_ep::named_barrier_sync(kPullMapBarrierId, Warps * 32);
}

// Preserve local-expert order and original top-k positions in one pass.
template <int TopK, bool Receiver>
__device__ __forceinline__ int flat_token_routes(
    const uint16_t* routes, int token, int first_expert, int experts_per_rank, int* hits) {
    int count = 0;
#pragma unroll
    for (int k = 0; k < TopK; ++k) {
        const int expert = static_cast<int>(routes[static_cast<size_t>(token) * TopK + k]) - first_expert;
        if (expert < 0 || expert >= experts_per_rank) {
            continue;
        }
        if constexpr (Receiver) {
            const int route = flat_route_pack(expert, k);
            // K is small and JIT-specialized. Sorted insertion preserves
            // flat's expert ordering without scanning every local expert.
            int i = count;
            while (i > 0 && hits[i - 1] > route) {
                hits[i] = hits[i - 1];
                --i;
            }
            hits[i] = route;
        }
        ++count;
        // Source-side DROP validity needs only destination membership.
        if constexpr (!Receiver) {
            break;
        }
    }
    return count;
}

// The MAP warp group builds one deterministic source chunk. Receiver=true writes
// the receive maps; false builds this source's retained destination slots for
// DROP combine. Both use exactly the same slot arithmetic.
template <int TopK, int ChunkTokens, int Warps, bool Receiver, bool Publish>
__device__ __forceinline__ int2 flat_build_chunk(
    const dispatch_pull_param_t& p, int source, int chunk, int destination,
    int* warp_counts, int* shared_base,
    int2* tile_ranges, unsigned* tiles_ready, unsigned first_tile) {
    static_assert(!Publish || ChunkTokens % (Warps * 32) == 0);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (warp == 0) {
        const int base = flat_chunk_base(p, source, chunk, destination, lane);
        if (lane == 0) {
            *shared_base = base;
        }
    }
    flat_map_barrier<Warps>();
    const int base = *shared_base;
    int consumed = 0;
    const int chunk_end = (chunk + 1) * ChunkTokens;
    const int end = min(chunk_end, p.tokens_per_rank);
    const int first_expert = destination * p.experts_per_rank;
    const uint16_t* routes = reinterpret_cast<const uint16_t*>(p.own_meta_staging +
        static_cast<size_t>(source) * pull_meta_stride(p.topk_off_bytes, p.tokens_per_rank) + p.topk_off_bytes);
    const int loop_end = Publish ? chunk_end : end;
    for (int tile = chunk * ChunkTokens; tile < loop_end; tile += Warps * 32) {
        const int token = tile + threadIdx.x;
        int hits[Receiver ? TopK : 1];
        const int count = token < end
            ? flat_token_routes<TopK, Receiver>(routes, token, first_expert, p.experts_per_rank, hits) : 0;
        const unsigned mask = __ballot_sync(0xffffffffu, count > 0);
        if (lane == 0) {
            warp_counts[warp] = __popc(mask);
        }
        flat_map_barrier<Warps>();
        int prefix = consumed;
        for (int w = 0; w < warp; ++w) {
            prefix += warp_counts[w];
        }
        const int slot = base + prefix + __popc(mask & ((1u << lane) - 1));
        if constexpr (Receiver) {
            if (count > 0 && slot < p.flat_recv_capacity) {
                p.recv_slot_to_src[slot] = source * p.tokens_per_rank + token;
#pragma unroll
                for (int k = 0; k < TopK; ++k) {
                    p.srcpos_map[static_cast<size_t>(slot) * TopK + k] = k < count ? hits[k] : -1;
                    if (k < count && p.caller_counts != nullptr) {
                        atomicAdd(static_cast<int32_t*>(p.caller_counts) + flat_route_expert(hits[k]), 1);
                    }
                }
            }
        } else {
            if (token < end) {
                p.retained_slots[static_cast<size_t>(token) * p.lsa_team_size + destination] =
                count > 0 && slot < p.flat_recv_capacity ? slot : -1;
            }
        }
        const int tile_base = base + consumed;
        int tile_count = 0;
        for (int w = 0; w < Warps; ++w) {
            tile_count += warp_counts[w];
        }
        consumed += tile_count;
        // Order routing writes before publication and finish all scratch reads
        // before the next tile. Empty tail tiles also publish a zero-sized range.
        flat_map_barrier<Warps>();
        if constexpr (Publish) {
            if (threadIdx.x == 0) {
                const unsigned unit = first_tile + (tile - chunk * ChunkTokens) / (Warps * 32);
                tile_ranges[unit] = make_int2(tile_base, max(0, min(tile_count, p.flat_recv_capacity - tile_base)));
                nccl_ep::st_release_block_shared(tiles_ready, unit + 1);
            }
        }
    }
    return make_int2(base, max(0, min(consumed, p.flat_recv_capacity - base)));
}

template <int HiddenInt4, bool Scales, bool ScaleTma, int ScaleInt4PerLane,
          typename TokenStage, typename ScaleStage>
__device__ __forceinline__ void dispatch_pull_flat_row(
    const dispatch_pull_param_t& p, int slot, int lane,
    TokenStage& token_stage, ScaleStage& scale_stage) {
    const int scale_int4 = p.scale_row_bytes / 16;
    const decoded_src_t src = decode_src(p.recv_slot_to_src[slot], p.tokens_per_rank, p.lsa_team_size);
    token_stage.issue(static_cast<const int4*>(p.peer_input_ptrs[src.rank]) +
                      static_cast<size_t>(src.token) * HiddenInt4);
    int4 scale_regs[ScaleTma ? 1 : ScaleInt4PerLane];
    if constexpr (Scales) {
        const uint8_t* scales = static_cast<const uint8_t*>(p.peer_scale_ptrs[src.rank]) +
                                static_cast<size_t>(src.token) * p.scale_row_bytes;
        if constexpr (ScaleTma) {
            scale_stage.issue(scales);
        } else {
#pragma unroll
            for (int i = 0; i < ScaleInt4PerLane; ++i) {
                const int j = lane + i * 32;
                if (j < scale_int4) {
                    scale_regs[i] = reinterpret_cast<const int4*>(scales)[j];
                }
            }
        }
    }
    if (p.recv_topk_weights_em != nullptr && lane < p.top_k) {
        const size_t out = static_cast<size_t>(slot) * p.top_k + lane;
        const int32_t route = p.srcpos_map[out];
        const bool valid = route >= 0;
        p.recv_topk_weights_em[out] = valid
            ? p.peer_weight_ptrs[src.rank][static_cast<size_t>(src.token) * p.top_k +
                                           flat_route_source_pos(route)] : 0.0f;
        if (p.recv_topk_idx != nullptr) {
            p.recv_topk_idx[out] = valid ? flat_route_expert(route) + p.output_expert_base : -1;
        }
    }
    token_stage.wait();
    token_stage.store(static_cast<int4*>(p.recv_x_em) + static_cast<size_t>(slot) * HiddenInt4);
    if constexpr (Scales) {
        uint8_t* scales = p.recv_x_scale_em + static_cast<size_t>(slot) * p.scale_row_bytes;
        if constexpr (ScaleTma) {
            scale_stage.wait();
            scale_stage.store(scales);
        } else {
#pragma unroll
            for (int i = 0; i < ScaleInt4PerLane; ++i) {
                const int j = lane + i * 32;
                if (j < scale_int4) {
                    reinterpret_cast<int4*>(scales)[j] = scale_regs[i];
                }
            }
        }
    }
    bulk_wait_reads(lane);
}

template <int TopK, int ChunkTokens, int kMapWarps>
__device__ __forceinline__ void flat_produce_maps(
    const dispatch_pull_param_t& p, int* warp_counts, int* shared_base,
    int2* tile_ranges, unsigned* tiles_ready) {
    constexpr unsigned kTilesPerChunk = ChunkTokens / (kMapWarps * 32);
    const int total_chunks = nccl_ep::ceil_div(p.tokens_per_rank, ChunkTokens) * p.lsa_team_size;
    unsigned built = 0;
    for (int work = blockIdx.x; work < total_chunks; work += gridDim.x) {
        const int source = work % p.lsa_team_size;
        const int chunk = work / p.lsa_team_size;
        flat_build_chunk<TopK, ChunkTokens, kMapWarps, true, true>(
            p, source, chunk, p.my_rank, warp_counts, shared_base,
            tile_ranges, tiles_ready, built * kTilesPerChunk);
        ++built;
        if (p.retained_slots != nullptr) {
            flat_build_chunk<TopK, ChunkTokens, kMapWarps, false, false>(
                p, p.my_rank, chunk, source, warp_counts, shared_base, nullptr, nullptr, 0);
        }
    }
}

template <int ChunkTokens, int kMapWarps, typename ProcessSlot>
__device__ __forceinline__ void flat_consume_maps(
    const dispatch_pull_param_t& p, int lane, int2* tile_ranges,
    unsigned* tiles_ready, unsigned* next_ticket, ProcessSlot process_slot) {
    constexpr unsigned kTileTokens = kMapWarps * 32;
    constexpr unsigned kTilesPerChunk = ChunkTokens / kTileTokens;
    const int total_chunks = nccl_ep::ceil_div(p.tokens_per_rank, ChunkTokens) * p.lsa_team_size;
    // Match EM's small work claims; deterministic prefixes fix output order
    // independently of which payload warp handles each ready row.
    constexpr unsigned kSlotBatch = 4;
    static_assert(kTileTokens % kSlotBatch == 0);
    for (;;) {
        unsigned ticket = 0;
        if (lane == 0) {
            ticket = atomicAdd(next_ticket, kSlotBatch);
        }
        ticket = __shfl_sync(0xffffffffu, ticket, 0);
        const unsigned unit = ticket / kTileTokens;
        if (blockIdx.x + (unit / kTilesPerChunk) * gridDim.x >= total_chunks) {
            break;
        }
        int first_slot = 0, count = 0;
        if (lane == 0) {
            nccl_ep::wait_acquire_block_shared(tiles_ready, unit + 1);
            const int2 range = tile_ranges[unit];
            first_slot = range.x;
            count = range.y;
        }
        first_slot = __shfl_sync(0xffffffffu, first_slot, 0);
        count = __shfl_sync(0xffffffffu, count, 0);
        __syncwarp();
        const int offset = ticket % kTileTokens;
        const int end = min(offset + static_cast<int>(kSlotBatch), count);
        for (int i = offset; i < end; ++i) {
            process_slot(first_slot + i);
        }
    }
}

template <int TopK, int ChunkTokens, int PullWarps, typename ProcessSlot>
__device__ __forceinline__ void flat_map_and_transfer(
    const dispatch_pull_param_t& p, int warp, int* warp_counts, int* shared_base, ProcessSlot process_slot) {
    // With few chunks, use the whole CTA to build a map quickly rather than
    // reserving producers that cannot overlap enough subsequent work.
    const int total_chunks = nccl_ep::ceil_div(p.tokens_per_rank, ChunkTokens) * p.lsa_team_size;
    for (int work = blockIdx.x; work < total_chunks; work += gridDim.x) {
        const int source = work % p.lsa_team_size;
        const int chunk = work / p.lsa_team_size;
        const int2 range = flat_build_chunk<TopK, ChunkTokens, PullWarps, true, false>(
            p, source, chunk, p.my_rank, warp_counts, shared_base, nullptr, nullptr, 0);
        for (int i = warp; i < range.y; i += PullWarps) {
            process_slot(range.x + i);
        }
        __syncthreads();
        if (p.retained_slots != nullptr) {
            flat_build_chunk<TopK, ChunkTokens, PullWarps, false, false>(
                p, p.my_rank, chunk, source, warp_counts, shared_base, nullptr, nullptr, 0);
        }
    }
}

// One warp transfers a receive row. Unlike EM, no output fan-out or padding is
// needed. Metadata and scale loads overlap the outstanding token-row load.
template <int HiddenInt4, ncclEpDispQuant_t Recipe, int ScaleInt4PerLane,
          int ScaleTmaRowBytes, int PullWarps, bool FusedMap, int TopK, int ChunkTokens, bool OverlapMap>
__device__ __forceinline__ void dispatch_pull_flat(const dispatch_pull_param_t& p) {
    static_assert(Recipe == NCCL_EP_DISP_QUANT_NONE || Recipe == NCCL_EP_DISP_QUANT_FWD);
    constexpr int kMapWarps = FusedMap && OverlapMap ? kPullDispatchMapWarps : 0;
    constexpr int kPayloadWarps = PullWarps - kMapWarps;
    static_assert(kPayloadWarps > 0);
    constexpr bool kScales = Recipe == NCCL_EP_DISP_QUANT_FWD;
    constexpr bool kScaleTma = kScales && ScaleTmaRowBytes > 0;
    using TokenStage = PullTmaStage<HiddenInt4 * 16>;
    using ScaleStage = PullTmaStage<kScaleTma ? ScaleTmaRowBytes : 16>;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    extern __shared__ uint8_t smem[];
    TokenStage token_stage;
    ScaleStage scale_stage;
    const bool is_map = warp < kMapWarps;
    const int payload_warp = warp - kMapWarps;
    if (!is_map) {
        token_stage.init(lane, smem + payload_warp * TokenStage::smem_bytes_per_warp);
        if constexpr (kScaleTma) {
            scale_stage.init(lane, smem + kPayloadWarps * TokenStage::smem_bytes_per_warp +
                                       payload_warp * ScaleStage::smem_bytes_per_warp);
        }
    }
    TokenStage::block_init_fence();
    flat_dispatch_head<FusedMap>(p);

    constexpr size_t kPayloadBytes = kPayloadWarps * (TokenStage::smem_bytes_per_warp +
        (kScaleTma ? ScaleStage::smem_bytes_per_warp : 0));
    int2* tile_ranges = reinterpret_cast<int2*>(smem + kPayloadBytes);
    __shared__ unsigned tiles_ready;
    __shared__ unsigned next_ticket;
    constexpr int kScratchWarps = OverlapMap ? kMapWarps : PullWarps;
    __shared__ int warp_counts[FusedMap ? kScratchWarps : 1];
    __shared__ int shared_base;
    __shared__ int shared_total;
    int num_recv;
    if constexpr (FusedMap) {
        if (threadIdx.x == 0) {
            tiles_ready = 0;
            next_ticket = 0;
        }
        num_recv = flat_receive_count(p, &shared_total);
    } else {
        num_recv = *p.num_recv_tokens_dev;
    }
    EP_DEVICE_ASSERT(num_recv <= p.caller_num_recv_tokens);
    const int total_warps = gridDim.x * kPayloadWarps;
    const int rows_per_warp = nccl_ep::ceil_div(num_recv, total_warps);
    const int first = (blockIdx.x * kPayloadWarps + payload_warp) * rows_per_warp;
    auto process_slot = [&](int slot) {
        dispatch_pull_flat_row<HiddenInt4, kScales, kScaleTma, ScaleInt4PerLane>(
            p, slot, lane, token_stage, scale_stage);
    };
    if constexpr (FusedMap && OverlapMap) {
        if (is_map) {
            flat_produce_maps<TopK, ChunkTokens, kMapWarps>(
                p, warp_counts, &shared_base, tile_ranges, &tiles_ready);
        } else {
            flat_consume_maps<ChunkTokens, kMapWarps>(
                p, lane, tile_ranges, &tiles_ready, &next_ticket, process_slot);
        }
    } else if constexpr (FusedMap) {
        flat_map_and_transfer<TopK, ChunkTokens, PullWarps>(p, warp, warp_counts, &shared_base, process_slot);
    } else {
        for (int slot = first; slot < min(first + rows_per_warp, num_recv); ++slot) {
            process_slot(slot);
        }
    }
    if (!is_map) {
        bulk_drain(lane);
    }
    __syncthreads();
    if (!p.unfused_sync) {
        lsa_grid_tail_barrier(p.dcomm, p.grid_barrier_counter, p.head_sync_flag);
    }
}

// Flat expert outputs are already reduced locally by the caller. Relay each row
// with TMA; neither floating-point accumulation nor EM gather metadata is needed.
template <int HiddenInt4, int PushWarps, bool Backward>
__device__ __forceinline__ void combine_push_flat(const combine_push_param_t& p) {
    using Stage = PullTmaStage<HiddenInt4 * 16>;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    extern __shared__ uint8_t smem[];
    Stage stage;
    stage.init(lane, smem + warp * Stage::smem_bytes_per_warp);
    Stage::block_init_fence();
    if (!p.unfused_sync) {
        lsa_grid_head_gate(p.dcomm, p.head_sync_flag);
    }

    const int num_recv = *p.num_recv_tokens_dev;
    EP_DEVICE_ASSERT(num_recv <= p.caller_num_recv_tokens);
    const int total_warps = gridDim.x * PushWarps;
    // Match EM: spread consecutive row ranges across SMs before rotating peers.
    const int global_warp = warp * gridDim.x + blockIdx.x;
    const int rotated_warp = (global_warp +
                             p.my_lsa_rank * (total_warps / p.lsa_team_size)) % total_warps;
    const int rows_per_warp = nccl_ep::ceil_div(num_recv, total_warps);
    const int first = rotated_warp * rows_per_warp;
    const tokens_fast_div_t divisor = make_tokens_fast_div(p.tokens_per_rank);
    constexpr int kRowBytes = HiddenInt4 * 16;
    constexpr int kStride = comb_stage_stride_bytes(kRowBytes, Backward);
    for (int slot = first; slot < min(first + rows_per_warp, num_recv); ++slot) {
        stage.issue(static_cast<const int4*>(p.recv_x_em) + static_cast<size_t>(slot) * HiddenInt4);
        const decoded_src_t dst = decode_src_fast(p.recv_slot_to_src[slot], divisor, p.lsa_team_size);
        uint8_t* row = static_cast<uint8_t*>(p.peer_staging_ptrs[dst.rank]) +
                       (static_cast<size_t>(dst.token) * p.lsa_team_size + p.my_lsa_rank) * kStride;
        // Carry sparse backward weights while the token load is outstanding.
        int source_pos = -1;
        float weight = 0.0f;
        if constexpr (Backward) {
            if (lane < p.top_k) {
                const size_t i = static_cast<size_t>(slot) * p.top_k + lane;
                const int32_t route = p.srcpos_map[i];
                if (route >= 0) {
                    source_pos = flat_route_source_pos(route);
                    weight = p.topk_weights_em[i];
                }
            }
        }
        stage.wait();
        stage.store(row);
        if constexpr (Backward) {
            if (source_pos >= 0) {
                reinterpret_cast<float*>(row + kRowBytes)[source_pos] = weight;
            }
        }
        bulk_wait_reads(lane);
    }
    bulk_drain(lane);
    __syncthreads();
    if (!p.unfused_sync) {
        lsa_grid_tail_barrier(p.dcomm, p.grid_barrier_counter, p.head_sync_flag);
    }
}

} // namespace ht_ep

#include "../scan_kernel.cuh"
#include "../count_kernel.cuh"

namespace ht_ep {

// Shared layout scratch: source-rank bases and expert bases.
__host__ __device__ __forceinline__ size_t dispatch_pull_map_smem_ints(int nRanks, int epr) {
    return static_cast<size_t>(nRanks) + static_cast<size_t>(epr);
}

}  // namespace ht_ep
