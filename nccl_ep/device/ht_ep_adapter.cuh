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
#include "quantization_recipe.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>
#include "nccl_device.h"
#include "ep_enums.h"
#include "common.hpp"
#include "nccl_ep_env.h"

namespace nccl_ep {
namespace ht {

// Count-mode receiver-table row width cap (fixed [flat2em|weights] row stride in the group
// IPC tables). Also caps HT EM local-permute num_topk (local_permute_reduce packs in
// ceil(kEpCountMaxTopk/32) warp chunks).
constexpr int kEpCountMaxTopk = 64;

// ============================================================================
// Runtime constants (from ep_group, not hardcoded):
//   lsa_team_size  - ncclTeamLsa(comm).nRanks  (1..NCCL_EP_HT_MAX_LSA_TEAM_SIZE)
//   rdma_team_size - ncclTeamRail(comm).nRanks
//   num_local_experts, hidden
//
// Kernels are JIT-compiled per (lsa_team_size, num_lsa_teams, ...); templates
// size their register/SMEM arrays exactly to the JIT lsa_team_size, so any
// value in [1, NCCL_EP_HT_MAX_LSA_TEAM_SIZE] works.  The only runtime
// precondition is the S2D cp.async.bulk alignment asserted in dispatch_impl
// (matters when s2d_inner_dim is not a multiple of 4).
// ============================================================================

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
// NCCL API uses sparse format: topk_idx[token][k] = expert_id
// HT uses bitmap format, byte-padded per LSA-team: each team's experts occupy
// their own ceil(experts_per_lsa_team/8)-byte block within the row.
// cached_topk_idx stores expert ids as int32 (narrower than any caller int64 topk_idx; expert
// ids always fit int32).
template <typename TopkIdxT>
void convert_topk_to_routing_map(
    const TopkIdxT* topk_idx,
    uint8_t* routing_bitmap,
    int32_t* cached_topk_idx,  // nullable; when non-null, caches the row in the same pass
    int num_tokens,
    int max_tokens,            // tail bound; rows [num_tokens, max_tokens) are zeroed
    int num_topk,
    int experts_per_lsa_team,        // = LSA_TEAM_SIZE * experts_per_rank
    int experts_per_lsa_team_packed, // = ceil(experts_per_lsa_team / 8)
    int row_bytes,                   // per-token stride = experts_per_lsa_team_packed * num_lsa_teams
    cudaStream_t stream);

// ============================================================================
// Count-exchange (push dispatch): builds per-source-rank count rows from topk_idx
// and reduces them into receiver layout on the unfused path. Push-dispatch only;
// pull dispatch (below) uses its own count-row builder.
// ============================================================================

// Count-exchange path: per-destination-rank and per-expert send-count histogram from topk_idx.
// topk_idx is read in its native width; cached_topk_idx stores expert ids as int32.
template <typename TopkIdxT>
void build_count_metadata(
    const TopkIdxT* topk_idx,
    int32_t* cached_topk_idx,  // [num_tokens, num_topk] cache copy for prob rebuild + combine, nullable
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int num_world_dst_ranks,
    int num_experts,           // num_world_dst_ranks * experts_per_rank
    int32_t* cnt_rank,         // [num_world_dst_ranks], pre-zeroed
    int32_t* cnt_expert,       // [num_experts], pre-zeroed, nullable
    int tokens_per_chunk,      // dispatch source-chunk width
    int32_t* own_chunk_rank,   // own-LSA-team slice [num_chunks, lsa_team_size], pre-zeroed, nullable
    int lsa_team_size,                       // dest ranks per LSA team
    int num_lsa_teams,                       // must be 1 (count mode is single-LSA-team only)
    uint64_t* token_dst_rank_bitmap, // [num_tokens] per-token dest-rank bitmap, nullable
    int32_t* per_src_lteam_num_tokens, // [num_lsa_teams] token counts; only own slot written (single-LSA: [1]), nullable
    int my_lteam,
    bool* rdma_to_attn_map,    // this-LSA-team slice; rdma_to_attn_map[r2a_offset+token], nullable
    int r2a_offset,            // my_lteam * rdma_per_lsa_sz
    int num_sms,               // grid cap = scan preprocessing SM count; kernel grid-strides
    cudaStream_t stream);

// compute_layout_info: unfused count path (EM). Reduces the world-AllGathered
// per-source count rows into per-expert recv counts and writes the caller layout
// tensors. The reduction is order-independent, so it is valid whether count mode's
// current single-LSA-team constraint holds or is later lifted to multiple LSA teams
// (multinode). Counts/offsets exclude dropped tokens; recv_total_counter reports
// the pre-drop total (same DROP contract as scan mode).
void compute_layout_info(
    const int32_t* cached_cnt_rows,    // [num_src_ranks * (num_src_ranks + num_experts)]
    int num_src_ranks,             // nRanks
    int num_experts,               // num_src_ranks * experts_per_rank (world)
    int experts_per_rank,
    int my_rank,
    int em_alignment,              // dispatch_output_per_expert_alignment
    int max_recv_tokens_per_rank,  // recv budget / DROP capacity
    bool allow_overflow_drop,
    bool out_is_int64,             // dtype of caller counts/offsets
    int64_t* em_internal_offsets,  // [experts_per_rank + 1] handle expert_token_offsets
    void* em_padded_out_counts,    // caller expert_counters
    void* em_out_offsets,          // caller expert_offsets, nullable
    int32_t* em_actual_counts_out, // handle authoritative counts, nullable
    void* recv_total_counter,      // caller recv_total_counter, nullable
    int32_t* num_tokens_for_experts, // handle FLAT recv count (unpadded); nullable
    cudaStream_t stream);

// ============================================================================
// Pull dispatch (kPullPush): order-preserving uint16 topk map, replacing the bitmap
// routing map. Pull-side count-exchange helpers (receiver count-row build) belong
// here too, next to pack_topk_idx.
// ============================================================================

// Pull dispatch: order-preserving alternative to the bitmap routing map. Emits each
// token's top-k global expert ids as uint16 (row stride = num_topk), kTopkIdxInvalid
// for empty slots and tail rows [num_tokens, max_tokens). cached_topk_idx mirrors the
// uint16 output into a handle-local snapshot for push combine; nullable.
template <typename TopkIdxT>
void pack_topk_idx(
    const TopkIdxT* topk_idx,
    uint16_t* topk_idx_u16,
    uint16_t* cached_topk_idx,
    int num_tokens,
    int max_tokens,
    int num_topk,
    cudaStream_t stream);

// Fused pull-count: pack the handle-local top-k snapshot to uint16 and accumulate its
// send-count histogram (per-source-rank and per-expert) in one pass, feeding
// dispatch_pull_param_t's own_row/own_topk_snapshot inputs directly.
template <typename TopkIdxT>
void pack_topk_and_count_row(
    const TopkIdxT* topk_idx,
    uint16_t* topk_idx_u16,
    int num_tokens,
    int max_tokens,
    int num_topk,
    int experts_per_rank,
    int num_experts,
    int32_t* cnt_rank,
    int32_t* cnt_expert,
    int num_sms,
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob (for dispatch input)
// ============================================================================
// NCCL API uses sparse format: topk_weights[token][k] = weight, topk_idx[token][k] = expert_id
// HT uses dense format: prob[token][expert] = weight (0 for non-selected)
template <typename TopkIdxT>
void sparse_to_dense_prob(
    const TopkIdxT* topk_idx,     // [num_tokens, topk] - which experts
    const float* topk_weights,    // [num_tokens, topk] - weights for those experts
    float* dense_prob,            // [num_tokens, num_experts] - output dense prob (pre-zeroed)
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob (for combine input)
// ============================================================================
// Used for combine backward pass. Unlike dispatch which has explicit topk_idx,
// combine uses local_expert_routing_map to determine expert positions.
// The order matches dispatch output (dense_to_sparse_prob): experts scanned in order 0,1,2...
// NOTE: No topk_idx needed - expert mapping comes from local_expert_routing_map
void sparse_to_dense_prob_combine(
    const float* topk_weights, // [num_tokens, topk] - sparse weights from combine input
    const bool* local_expert_routing_map, // [num_tokens, experts_per_rank] - from handle
    float* dense_prob, // [num_tokens, experts_per_lsa_team] - output (pre-zeroed)
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_lsa_team, // = experts_per_rank * ranks_per_lsa_team
    int local_rank, // rank within lsa team
    bool lerm_bitmap, // count mode: LERM is a ceil(epr/64)-word packed bitmap per token
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert dense prob output to sparse format (for dispatch output)
// ============================================================================
// HT outputs: dense_prob[recv_token][experts_per_lsa_team]
// HT expects: recv_topk_weights[recv_token][topk], recv_topk_idx[recv_token][topk]
// Uses local_expert_routing_map to know which experts each token is routed to.
// recv_topk_idx numbering is selected by recv_topk_idx_kind:
//   LOCAL  -- per-rank local expert id (0-based within this rank); default.
//   GLOBAL -- wire-format global expert id (= global_expert_offset + local_id),
//             matching the LL rank-major GLOBAL contract.
// recv_topk_idx_kind must be resolved (no AUTO) by the caller.
void dense_to_sparse_prob(
    const float* dense_prob, // [num_recv_tokens, experts_per_lsa_team] - from IPC buffer
    const bool* local_expert_routing_map, // [num_recv_tokens, experts_per_rank] - from preprocessing
    float* recv_topk_weights, // EM: [N]; FLAT/RM: [N, topk]
    int64_t* recv_topk_idx, // [num_recv_tokens, topk] - LOCAL or GLOBAL expert id (see kind); nullptr under EM
    int num_recv_tokens,
    const int32_t* num_recv_tokens_dev, // Flat rows, or padded extent for direct EM.
    int topk,
    int experts_per_rank,
    int experts_per_lsa_team, // = experts_per_rank * ranks_per_lsa_team
    int local_rank, // rank within lsa team
    int global_expert_offset, // = group_rank * experts_per_rank; added to local id under GLOBAL
    ncclEpExpertIdKind_t recv_topk_idx_kind,
    bool expert_major, // true = 1D recv_topk_weights, false = 2D
    bool lerm_bitmap, // count mode: LERM is a ceil(epr/64)-word packed bitmap per token
    cudaStream_t stream);

// Combine BWD output: sparse k-slot writeback keyed by FWD-input cached_topk_idx.
template <typename TopkIdxT>
void dense_to_sparse_prob_combine(
    const float* dense_prob, // [num_tokens, num_experts]
    const TopkIdxT* cached_topk_idx, // [num_tokens, topk]
    float* combined_topk_weights, // [num_tokens, topk]
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream);

// ============================================================================
// Preprocessing wrapper with template parameter resolution
// ============================================================================

// Helper to call metadata_preprocessing with resolved template parameters.
// Caller must provide a pre-allocated scan temp buffer (see get_preprocessing_scan_tmp_size).
// When alignment > 0, the kernel also fuses the expert-major remap pass.
ncclResult_t call_metadata_preprocessing(
    const uint8_t* global_routing_map, // Already allgathered bitmap routing map
    int32_t*
        sparse_to_dense_map, // Output: unified S2D — flat stores token->rank->slot; expert-major stores packed (rank, slot) entries
    bool* rdma_to_attn_map, // Output: which tokens come from RDMA
    bool* attn_to_rdma_map, // Output: which tokens go to RDMA
    void* token_rank_mask, // Scratch rank masks, sized by get_rank_mask_elem_size().
    int32_t* num_tokens_for_experts, // Output: total tokens for local experts
    bool* local_expert_routing_map, // Output: per-expert routing for local tokens
    int32_t* per_expert_token_counts, // Optional output: per-expert counts (nullptr to skip)
    void* ranks_scan_tmp, // Pre-allocated per-rank scan state (from get_preprocessing_scan_tmp_size)
    int lsa_team, // This LSA team's index (0 to num_lsa_teams-1)
    int local_rank, // Rank within lsa team (0 to lsa_team_size-1)
    int num_tokens_per_rank, // Actual tokens per rank this iteration (runtime)
    int num_lsa_teams, // Number of LSA teams (RDMA domain size)
    int lsa_team_size, // Ranks per LSA (NVLink) team
    int experts_per_rank, // Experts per GPU
    bool expert_major = false, // true = expert-major layout (gates fused remap)
    int64_t* internal_offsets = nullptr, // Expert-major: per-expert zone offsets consumed by dispatch
    void* padded_out_counts =
        nullptr, // Expert-major: per-expert padded counts (caller tensor, nullable; int32 or int64)
    void* out_offsets = nullptr, // Expert-major: per-expert offsets (caller tensor, nullable; int32 or int64)
    size_t alignment =
        0, // Per-expert zone alignment in tokens (pow2; 0/1 = no padding). Ignored when expert_major=false.
    int32_t* actual_counts_out = nullptr, // Expert-major: authoritative per-expert dispatch counts
    int s2d_inner_dim = 0, // 0 = flat (ranks_per_lsa_team); >0 = expert-major (top_k)
    void* recv_total_counter = nullptr, // Optional scalar: total recv tokens (int32 or int64; nullable)
    bool out_is_int64 = true, // Shared dtype for the 3 int output tensors above
    int max_recv_tokens_per_rank = 0, // HT recv-budget; __trap on overflow.
    int32_t* emuf_group_buf = nullptr, // Local-fanout dup-groups (nullable; EM only): [num_groups, group_stride]
    int32_t* emuf_group_count = nullptr, // Scalar group counter (zeroed by caller)
    int emuf_group_stride = 0, // Row width (= experts_per_rank)
    int emuf_max_groups = 0, // Row capacity; kernel __trap on overflow.
    int num_blocks = 16, // Number of SMs for the kernel grid
    void* scan_gscratch = nullptr, // EM cooperative scan scratch (required when expert_major)
    // EM-permute mode: when true, FLAT scan replaces scan_em and writes the
    // unified (FLAT-shaped) sparse_to_dense_map + FLAT-shape LERM;
    // em_scan_kernel produces only EM offsets / counts / em_slot table.
    bool em_permute = false,
    int32_t* token_to_recv_slot = nullptr,
    int32_t* flat2em_slot_map = nullptr,
    int em_top_k = 0,
    // NCCL_EP_OVERFLOW_DROP: drop tokens whose recv slot exceeds the budget
    // instead of trapping on overflow.
    bool allow_overflow_drop = false,
    // Pull dispatch: recv slot -> global source token id. Null unless pull is enabled.
    int32_t* recv_slot_to_src = nullptr,
    // Pull dispatch: source top-k position of each hit (parallel to flat2em).
    int32_t* srcpos_map = nullptr,
    // Pull dispatch: order-preserving uint16 topk routing map + gate.
    const uint16_t* global_topk_idx = nullptr,
    bool use_topk_idx = false,
    cudaStream_t stream = 0);

// Returns required size in bytes for the scan temp buffer used by call_metadata_preprocessing.
// Caller must allocate at least this many bytes and pass the pointer to call_metadata_preprocessing.
// num_blocks must be >= the block count the scan is launched with (see the
// preprocessing SM count resolved in ncclEpCreateGroup); size with the device
// SM count to cover any NCCL_EP_PREPROCESS_NUM_SMS override.
size_t get_preprocessing_scan_tmp_size(int num_blocks, int lsa_team_size);

// Returns sizeof(rank_mask_t<ceil(lsa_team_size/64)>) for the given lsa_team_size.
// Formula: ceil(lsa_team_size / 64) * sizeof(uint64_t).
size_t get_rank_mask_elem_size(int lsa_team_size);

// Cooperative kernel that consumes the preprocessing scan outputs (rank mask)
// and builds the EM-layout index tables: S2D, LERM, per-expert counts/offsets,
// em_slot. Used when expert_major=true. Requires lsa_team_size <= 128
// (fits in 2 x uint64 mask words) and experts_per_rank to be a power of two.
void launch_build_em_tables(
    const uint8_t* input_routing_map,
    const void* token_rank_mask,
    int num_mask_words,
    int num_total_attn_tokens,
    int num_tokens_per_rank,
    int lsa_team_size,
    int experts_per_rank,
    int num_lsa_teams,
    int lsa_team,
    int local_rank,
    int s2d_inner_dim,
    int max_recv_tokens_per_rank,
    int em_alignment,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* local_expert_routing_map,
    int32_t* num_tokens_for_experts,
    int64_t* em_internal_offsets,
    void* em_padded_out_counts,
    void* em_out_offsets,
    int32_t* em_actual_counts_out,
    void* recv_total_counter,
    bool out_is_int64,
    int32_t*
        emuf_group_buf, // Local-fanout dup-groups (nullable): [num_groups, group_stride] = [primary, sec0, ..., -1]
    int32_t* emuf_group_count, // Scalar group counter
    int emuf_group_stride, // Row width (= experts_per_rank)
    int emuf_max_groups, // Row capacity; kernel __trap on overflow.
    int32_t* gscratch,
    int num_sms,
    const int32_t* token_to_recv_slot,
    int32_t* flat2em_slot_map,
    int em_top_k,
    bool allow_overflow_drop,
    cudaStream_t stream);

// The size of gscratch (ep_workspace) consumed by the EM scan
size_t get_em_scan_gscratch_size(int lsa_team_size, int experts_per_rank,
                                 int num_sms, bool is_local_permute);

// Copy only device-counted rows from dispatch staging.
enum class DispatchCopyMode : unsigned int { kCe = 0, kSimt = 1, kTma = 2 };
constexpr unsigned int dispatch_copy_tma_sms(unsigned int device_sm, unsigned int device_sm_count) {
    const unsigned int sms = (device_sm == 100 || device_sm == 103) ? 96u : 128u;
    return device_sm_count < sms ? device_sm_count : sms;
}
ncclResult_t configure_dispatch_copy_tma();
void launch_dispatch_copy_epilogue(
    const int32_t* flat_count, const int64_t* em_count,
    size_t capacity, size_t staging_capacity,
    const void* token_src, void* token_dst, size_t token_row_bytes,
    const void* scale_src, void* scale_dst, size_t scale_row_bytes,
    unsigned int copy_sms, DispatchCopyMode mode, cudaStream_t stream);

// Scatter FLAT staging rows into EM zones using flat2em_slot_map (written by
// em_scan_kernel during UpdateHandle); zero-fill per-expert pad rows.
// shuffle_sms=0 picks sm_count. Count mode passes flat2em_out/lerm_in/lerm_out/em_permute_cursors
// (all non-null) to build the flat2em rows inline from the LERM rows instead of consuming a
// pre-built map; em_permute_cursors is a zero-seeded [experts_per_rank + 1] cursor array (see
// local_permute_dup_param_t in ht_ep.cuh) and lerm_out persists the LERM row for combine.
void launch_dispatch_permute(
    void* recv_x_em,
    float* recv_topk_weights_em,
    const void* flat_staging,
    const float* recv_topk_weights_flat,
    const int32_t* flat2em_slot_map,
    const int32_t* num_recv_tokens_dev,
    const int64_t* expert_token_offsets,
    // Mutable: DROP-mode phantom-row fixup corrects it in place (see allow_overflow_drop
    // below and local_permute_dup_param_t in ht_ep.cuh).
    int32_t* per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int row_bytes,
    int sm_count,
    unsigned int shuffle_sms,
    int caller_num_recv_tokens,
    cudaStream_t stream,
    ncclEpDispQuant_t recipe,
    void* recv_scales_em,
    const void* flat_scale_staging,
    int scale_row_bytes,
    int32_t* flat2em_out = nullptr,
    const bool* lerm_in = nullptr,
    bool* lerm_out = nullptr,
    int32_t* em_permute_cursors = nullptr,
    // Count mode + DROP only: gates an in-kernel phantom-row correction (see
    // local_permute_dup_param_t::allow_overflow_drop in ht_ep.cuh for why it's needed).
    // Ignored (no correction) when em_permute_cursors is null.
    bool allow_overflow_drop = false);

// Pull EM dispatch (MNNVL): each receiver reads its source rows and topk weights
// directly from the source ranks' input buffers over NVLink and scatters them into
// EM zones. peer_input_ptrs / peer_weight_ptrs are device arrays [lsa_team_size] of
// source base pointers. shuffle_sms=0 picks sm_count.
ncclResult_t launch_dispatch_pull(
    void* recv_x_em,
    float* recv_topk_weights_em,
    void* recv_x_scale_em,
    int32_t* flat2em_slot_map,
    int32_t* srcpos_map,
    int32_t* recv_slot_to_src,
    const void* const* peer_input_ptrs,
    const float* const* peer_weight_ptrs,
    const void* const* peer_scale_ptrs,
    int32_t* num_recv_tokens_dev,
    int64_t* expert_token_offsets,
    int32_t* per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int row_bytes,
    int scale_row_bytes,
    int caller_num_recv_tokens,
    int tokens_per_rank,
    int lsa_team_size,
    int sm_count,
    unsigned int shuffle_sms,
    ncclEpDispQuant_t recipe,
    ncclDevComm_t* dcomm,
    uint32_t* head_sync_flag,
    uint32_t* grid_barrier_counter,
    cudaStream_t stream,
    // When true, run the intra-LSA head/tail sync as separate kernels around this launch.
    bool unfused_sync = false,
    // Fused pull-count inputs; leave at defaults for the scan path. layout_ready != nullptr
    // selects the MAP-warp-group kernel variant that builds flat2em_slot_map/srcpos_map/
    // recv_slot_to_src in-kernel instead of reading them prebuilt by the scan.
    const uint8_t* const* meta_ptrs = nullptr,
    int topk_off_bytes = 0,
    int32_t* rank_cursor = nullptr,
    int32_t* expert_cursor = nullptr,
    int32_t* layout_ready = nullptr,
    int my_rank = 0,
    int max_recv_tokens_per_rank = 0,
    int flat_recv_capacity = 0,
    int em_alignment = 1,
    bool allow_overflow_drop = false,
    const int32_t* own_row = nullptr,
    const uint16_t* own_topk_snapshot = nullptr,
    uint8_t* own_meta_staging = nullptr,
    int32_t* layout_slot_base = nullptr,
    int32_t* layout_expert_base = nullptr,
    const int32_t* cached_cnt_rows = nullptr,
    // Caller's dispatch-time layout_info outputs (mirrors dispatch_push_count's caller_offsets/
    // caller_counts/caller_recv_total); null when the caller didn't request them.
    void* caller_offsets = nullptr,
    void* caller_counts = nullptr,
    void* caller_recv_total = nullptr,
    bool caller_out_is_int64 = false);

// Minimum shared-memory bytes launch_dispatch_pull requires for a given row width
// (int4 units): what a single pull warp needs, static and dynamic. Callers compare this
// against the device opt-in smem cap; a row that does not fit one warp is unsupported.
size_t dispatch_pull_smem_bytes(int hidden_int4);

// Push-combine staging row stride (bytes) for a given token row width. The combine
// input staging must be sized by this so the buffer matches the stride combine_push
// writes and combine_reduce reads. reserve_prob (backward) co-locates the prob row
// in the token row's trailing pad.
size_t comb_stage_stride_bytes(int row_bytes, bool reserve_prob);

// Push EM combine (mirror of pull EM dispatch). The push kernel does the local
// gather-reduce + remote push with the head/tail intra-LSA syncs fused in; the reduce
// kernel does the final team_size sum. peer_staging_ptrs is a [lsa_team_size] array of
// peer-writable dst staging bases (indexed by dst_rank == the source rank's slot in the team).
ncclResult_t launch_combine_push(
    void* const* peer_staging_ptrs,
    const void* recv_x_em,
    const int32_t* flat2em_slot_map,
    const int32_t* recv_slot_to_src,
    const int32_t* num_recv_tokens_dev,
    ncclDevComm_t* dcomm,
    uint32_t* head_sync_flag,
    uint32_t* grid_barrier_counter,
    int top_k,
    int row_bytes,
    int caller_num_recv_tokens,
    int my_lsa_rank,
    int tokens_per_rank,
    int lsa_team_size,
    int sm_count,
    unsigned int shuffle_sms,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16,
    // Backward combine (sparse-direct prob scatter into the token staging pad); null/false on forward.
    const float* topk_weights_em = nullptr,
    const int32_t* srcpos_map = nullptr,
    bool backward = false,
    // When true, run the intra-LSA head/tail sync as separate kernels around this launch.
    bool unfused_sync = false);

ncclResult_t launch_combine_reduce_stage(
    void* attn_output,
    const void* staging,
    const uint16_t* topk_idx,
    int num_topk,
    int experts_per_rank,
    int experts_per_lsa_team,
    int num_combined_tokens,
    int lsa_team_size,
    int row_bytes,
    int sm_count,
    unsigned int shuffle_sms,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16,
    // Backward combine: gather each srcpos weight from the token staging pad into combined_topk_weights.
    float* combined_topk_weights = nullptr,
    int top_k = 0,
    bool backward = false);

// Inverse of launch_dispatch_permute: gather caller EM combine input into
// FLAT staging by summing the top_k EM rows per FLAT recv slot. Optional
// em_weights_in / flat_weights_out pair fuses a 1D EM to [num_flat, top_k]
// FLAT weight gather (BWD combine); pass nullptr for both on FWD.
ncclResult_t launch_combine_reduce(
    void* flat_staging,
    const void* recv_x_em,
    const int32_t* flat2em_slot_map,
    const int32_t* num_recv_tokens_dev,
    const float* em_weights_in,
    float* flat_weights_out,
    int top_k,
    int hidden,
    int input_row_bytes,
    int caller_num_recv_tokens,
    int sm_count,
    unsigned int shuffle_sms,
    cudaStream_t stream,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t combine_recipe);

// ============================================================================
// Memory region info structs for GIN
// All buffers are part of a single large gin_base_ptr buffer
// Offsets are relative to gin_base_ptr (stored as size_t for offset calculation)
// ============================================================================

struct dispatch_memory_region_info_t {
    // Offsets relative to gin_base_ptr for RDMA operations
    size_t attn_input_token_offset; // Offset of token staging buffer from gin_base_ptr
    size_t attn_input_prob_offset; // Offset of prob staging buffer from gin_base_ptr
    size_t attn_input_scaling_factor_offset; // Offset of scaling factor staging buffer
    // Batched RDMA staging (packed layout: token+prob+sf per entry)
    size_t gin_send_staging_offset; // Offset of per-destination staging buffer
    size_t gin_recv_staging_offset; // Offset of packed receive buffer (token+prob+sf per entry)
    size_t guard_offset; // Offset of RDMA sync-guard readiness flags (NUM_LSA_TEAMS uint64 slots)
    size_t bytes_per_entry; // Size of packed entry (token + prob + sf)
    size_t max_tokens_per_dest; // Max tokens that can be staged per destination
    // Streaming RDMA signals
    unsigned signals_tail_base; // Base signal ID for tail tracking (sender -> receiver)
    // Streaming buffer configuration
    int num_max_rdma_chunked_send_tokens; // Batch size per RDMA put (default: 6)
};

struct combine_memory_region_info_t {
    // Offsets relative to gin_base_ptr for RDMA operations
    size_t combine_red_token_offset; // Offset of combine LSA-team-reduced token buffer
    size_t combine_g2s_token_offset; // Offset of combine cross-LSA-team (N2N RDMA) token buffer
    size_t combine_red_prob_offset; // Offset of combine LSA-team-reduced prob buffer
    size_t combine_g2s_prob_offset; // Offset of combine cross-LSA-team (N2N RDMA) prob buffer
    size_t guard_offset; // Offset of combine's RDMA sync-guard readiness flags
};

// ============================================================================
// Dispatch wrapper with template parameter resolution
// ============================================================================

// Count-exchange dispatch payload: buffers and tunables the MAP warps consume to
// rebuild the receiver s2d/LERM/flat2em tables inside dispatch. Disabled
// (active=false) leaves the scan comm path untouched. Set together in the
// dispatch caller so the top-level DispatchParams stays scannable.
struct DispatchPushCountParams {
    bool active = false;                                    // gate: MAP warps build s2d/LERM in-kernel
    const int32_t* own_row = nullptr;                       // fused-path per-handle own count row (head fan-out source)
    const int32_t* cached_cnt_rows = nullptr;               // unfused: AllGathered count rows (per-handle), nullable
    bool fused_meta_dispatch = true;                        // fused meta delivery (intra-LSA fan-out); false = unfused, MAP reads cached_cnt_rows
    int32_t* s2d_out = nullptr;                             // writable s2d [max_tokens, lsa_team_size], host-set 0xFF
    size_t published_offset = 0;                            // count rows offset in the peer table regions
    const int32_t* per_src_lteam_chunk_rank = nullptr;       // [num_chunks, lsa_team_size] per-chunk send counts
    const int32_t* cached_topk_idx = nullptr;               // cached sender topk_idx [max_tokens, num_topk] (int32 expert ids)
    int32_t* num_recv_out = nullptr;                        // FLAT recv count (num_tokens_for_experts)
    const int32_t* per_src_lteam_num_tokens = nullptr;       // [1] real token count for the sender's stream
    int num_topk = 0;
    int num_src_ranks = 0;                                  // nRanks (== lsa_team_size in count mode)
    int lsa_team_size = 0;
    const uint64_t* token_dst_rank_bitmap = nullptr;        // [max_tokens] per-token global dest-rank bitmap
    uint8_t* const* recv_tables_ptrs = nullptr;             // per-peer EM table region base
    int recv_max_slots = 0;
    bool allow_overflow_drop = false;                       // clamp published per-expert offsets/counts to recv_max_slots
    int64_t* expert_token_offsets_out = nullptr;            // [experts_per_rank + 1]
    int32_t* per_expert_counts_out = nullptr;               // [experts_per_rank] unpadded counts
    void* caller_offsets = nullptr;
    void* caller_counts = nullptr;
    void* caller_recv_total = nullptr;                      // caller recv_total_counter scalar (pre-drop padded total), nullable
    bool caller_out_is_int64 = false;
    int em_alignment = 1;
};

// All parameters needed for dispatch kernel, needed redifine because those in ht_ep.cuh are with template on data type.
struct DispatchParams {
    // User inputs
    int hidden_dim; // Model hidden dimension
    int experts_per_rank; // Experts per GPU
    int lsa_team_size; // Ranks per LSA (NVLink) team
    const void* attn_input_token;
    const float* attn_input_prob; // Forward dispatch only
    const void* attn_input_scaling_factor;     // QUANT_FWD input; runtime-typed storage
    ncclDataType_t scale_dtype;                // QUANT_FWD tensor dtype; selects ScaleT in JIT

    // IPC-mapped output buffers (from ep_group)
    // Pointer tables are expected to be device-resident (d_* arrays).
    void* const* expert_output_token_ptrs; // Array[lsa_team_size]
    float* const* expert_output_prob_ptrs; // Forward only
    void* const* expert_output_scaling_factor_ptrs;    // QUANT_FWD outputs; runtime-typed storage

    // Metadata (from handle->ht preprocessing outputs)
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // Inner dim of unified S2D (num_topk in expert-major, ranks per LSA team in flat).
    ncclEpLayout_t layout; // Output layout (selects kernel template specialization).

    // EM zero-padding inputs for the in-kernel PAD warp (alignment==0 disables; ptrs may be null).
    const int32_t* pad_actual_counts;
    const int64_t* pad_expert_token_offsets;
    int pad_alignment;

    // Device pointers to the expected counters; initialized at bootstrap and
    // bumped by the dispatch kernel tail so CUDA-graph replays self-sequence.
    uint64_t* expected_gin_flag_val;
    uint32_t* expected_lsa_flag_val;
    uint64_t* gin_G2S_flags;
    uint32_t* lsa_S2G_flags;
    // Grid barrier counter for fused device_sync in dispatch tail
    uint32_t* dispatch_grid_barrier_counter;

    // GIN context (from ep_group, multi-LSA-team only)
    ncclDevComm* dcomm; // Device communicator
    ncclWindow_t nccl_token_window; // Source window handle for token data
    ncclWindow_t nccl_prob_window; // Registered window handle for probability data
    ncclWindow_t nccl_sf_window; // Registered window handle for scaling-factor data
    ncclWindow_t nccl_internal_window; // Internal destination window handle
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    dispatch_memory_region_info_t mr_info;

    // Runtime config
    int local_rank;
    int lsa_team;
    int tokens_per_lsa;

    // EM local-fanout: > 0 enables sender S2G dedup + a receiver local_dup kernel.
    int local_dup_num_sms = 0;

    bool guard_enabled = false; // RDMA + LSA buffer guard on/off

    // Backstop bound for recv slot indices (see dispatch_kernel_param_base_t).
    int max_recv_tokens_per_rank = 0;

    // Count-exchange payload (fused-meta-dispatch). See DispatchPushCountParams.
    DispatchPushCountParams dispatch_push_count;
};

// Call dispatch kernel. Recipe and pass-direction enums remain explicit until
// the JIT boundary selects its compile-time specialization.
// Returns ncclInvalidArgument if the dispatch SMEM exceeds the device limit.
ncclResult_t call_dispatch(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank, // Max tokens for buffer sizing (chunk-aligned)
    int num_tokens_per_chunk, // Dispatch/combine tokens per chunk (resolved per group)
    int num_lsa_teams, // Number of LSA teams (RDMA domain size)
    ncclEpDispQuant_t recipe,
    ncclEpPassDir_t pass_direction,
    int num_blocks, // Number of SMs/blocks for the kernel grid
    int max_dynamic_smem, // Cached opt-in dynamic shared-memory limit for this group
    int sf_bytes_per_token, // Total scale bytes per token (pre-computed on host)
    const ncclEpEnvConfig* env, // Group env config; gates rank-0 verbose param dump (may be null)
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16);

// ============================================================================
// Combine wrapper with template parameter resolution
// ============================================================================

// All parameters needed for combine kernel
struct CombineParams {
    // IPC-mapped input buffers (expert outputs from MLP)
    // NOTE: Pass HOST arrays containing device pointers (not device arrays).
    // These pointers are copied into the kernel param struct for fast __grid_constant__ access.
    int hidden_dim; // Model hidden dimension
    int experts_per_rank; // Experts per GPU
    int lsa_team_size; // Ranks per LSA (NVLink) team
    uint16_t* const* expert_input_token_ptrs; // HOST array[lsa_team_size], BF16 only
    float* const* expert_input_prob_ptrs; // HOST array, backward only

    // User output buffers
    void* attn_output_token;
    float* attn_output_prob; // Backward only (dense format for kernel)

    // RDMA buffers (multi-LSA-team only, from ep_group)
    uint16_t* combine_gin_RED_tokens;
    float* combine_gin_RED_prob; // Backward only
    const uint16_t* combine_gin_G2S_tokens;
    const float* combine_gin_G2S_prob; // Backward only

    // Metadata (from handle->ht preprocessing outputs)
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // Inner dim of unified S2D (num_topk in expert-major, ranks per LSA team in flat).
    ncclEpLayout_t layout; // Output layout (selects kernel template specialization).
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map; // For multi-LSA-team RDMA routing
    const bool* local_expert_routing_map; // For backward gradient routing

    // Device pointers to the expected counters; initialized at bootstrap and
    // bumped by the combine kernel tail so CUDA-graph replays self-sequence.
    uint64_t* expected_gin_flag_val;
    uint32_t* expected_lsa_flag_val;
    uint64_t* gin_G2S_flags;
    uint32_t* lsa_S2G_flags;
    // Per-rank grid-barrier counter that elects the last block at the combine tail.
    uint32_t* combine_grid_barrier_counter;

    // GIN context (multi-LSA-team only)
    ncclDevComm_t* dcomm; // Device communicator (device pointer)
    ncclWindow_t nccl_token_window; // Source window handle for token data
    ncclWindow_t nccl_prob_window; // Source window handle for probability data
    ncclWindow_t nccl_internal_window; // Internal destination window handle
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    unsigned signals_base; // Base signal ID
    unsigned combine_signal_offset; // Signal offset for combine operations
    combine_memory_region_info_t mr_info;

    // Runtime config
    int local_rank;
    int lsa_team;
    int tokens_per_lsa; // Stride for map indexing (= max_tokens_per_rank)
    int num_real_tokens; // Actual token count for output write gate
    int num_recv_tokens; // Actual received tokens this rank

    // EM unfused-combine: combine skips secondary em_slots; primaries hold the
    // pre-reduced sum written by local_reduce.
    bool combine_local_reduce_enabled = false;
    ncclDataType_t token_dtype = ncclBfloat16;  // Actual token wire dtype

    bool guard_enabled = false; // RDMA + LSA buffer guard on/off
    uint32_t combine_barrier_offset; // Reserved LSA barrier session index, not a byte offset

    ncclEpCombQuant_t combine_recipe = NCCL_EP_COMB_QUANT_NONE;
};

// Call combine kernel with runtime template parameter resolution
ncclResult_t call_combine(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank, // Max tokens for buffer sizing (chunk-aligned)
    int num_tokens_per_chunk, // Dispatch/combine tokens per chunk (resolved per group)
    int num_lsa_teams, // Number of LSA teams (RDMA domain size)
    bool backward_combine, // True for backward (training), false for forward
    int num_blocks, // Number of SMs/blocks for the kernel grid
    int max_dynamic_smem, // Cached opt-in dynamic shared-memory limit for this group
    const ncclEpEnvConfig* env, // Group env config; gates rank-0 verbose param dump (may be null)
    cudaStream_t stream);

// EM local-fanout: fill secondary em_slots from primaries after dispatch.
void call_local_dup(
    void* expert_output_token, // [recv_total, hidden]
    float* expert_output_prob, // forward: [recv_total, epr * ranks_per_lsa_team]
    const int32_t* emuf_group_buf, // device [num_groups, group_stride]
    const int32_t* emuf_group_count, // device scalar (read by kernel)
    int emuf_group_stride,
    const uint32_t* lsa_S2G_flag,
    uint32_t* expected_lsa_flag_val,
    uint32_t* grid_barrier_counter,
    int hidden_dim,
    int experts_per_rank,
    int ranks_per_lsa_team,
    bool forward_dispatch,
    int num_blocks,
    cudaStream_t stream,
    ncclDataType_t token_dtype,
    ncclEpDispQuant_t recipe,
    void* expert_output_scale,
    int scale_row_bytes);

// EM local-fanout: pre-sum secondaries into primaries before combine.
void call_local_reduce(
    void* expert_input_token, // [recv_total, hidden]
    float* expert_input_prob, // backward: [recv_total, epr * ranks_per_lsa_team]
    const int32_t* emuf_group_buf, // device [num_groups, group_stride]
    const int32_t* emuf_group_count, // device scalar (read by kernel)
    int emuf_group_stride, // row width (= experts_per_rank)
    int hidden_dim,
    int experts_per_rank,
    int ranks_per_lsa_team,
    bool backward_combine,
    int num_blocks,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16);

} // namespace ht
} // namespace nccl_ep
