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

#include "nccl_device.h"
#include "ht_ep_adapter.cuh"
#include "ht_ep_configs.cuh"
#include "common.hpp"
#include "jit/ht_combine_jit.cuh"
#include "jit/ht_dispatch_jit.cuh"
#include "jit/preprocess_jit.cuh"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace nccl_ep {
namespace ht {

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
// cached_topk_idx mirrors topk_idx in its native width (int32 or int64).
template <typename TopkIdxT>
__global__ void convert_topk_to_routing_map_kernel(
    const TopkIdxT* __restrict__ topk_idx,    // [num_tokens, num_topk]
    uint8_t* __restrict__ routing_bitmap,     // [max_tokens, row_bytes] (byte-padded per team)
    TopkIdxT* __restrict__ cached_topk_idx,   // [num_tokens, num_topk]; nullable
    int num_tokens,
    int max_tokens,                           // tail-zero bound (>= num_tokens)
    int num_topk,
    int experts_per_lsa_team,                 // = LSA_TEAM_SIZE * experts_per_rank
    int experts_per_lsa_team_packed,          // = ceil(experts_per_lsa_team / 8)
    int row_bytes                             // per-token stride = experts_per_lsa_team_packed * num_lsa_teams
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= max_tokens) return;

    // Each thread exclusively owns its row -- no atomics needed.
    // Zero the row before OR-ing in bits; the caller does not pre-zero.
    // Threads for tail rows [num_tokens, max_tokens) zero and exit, so the
    // downstream ncclAllGather over max_tokens rows ships clean tail bytes.
    // Byte-padded per LSA-team: each team's experts occupy their own
    // ceil(experts_per_lsa_team/8)-byte block (bit = within-team local expert id).
    uint8_t* row = routing_bitmap + token * row_bytes;
    for (int b = 0; b < row_bytes; b++) row[b] = 0;
    if (token >= num_tokens) return;
    const TopkIdxT* in_row = topk_idx + token * num_topk;
    TopkIdxT* out_row = cached_topk_idx ? cached_topk_idx + token * num_topk : nullptr;
    for (int k = 0; k < num_topk; k++) {
        TopkIdxT expert = in_row[k];
        if (out_row) out_row[k] = expert;
        if (expert >= 0) {
            // Global expert id = team * experts_per_lsa_team + within-team local id.
            const int team = static_cast<int>(expert) / experts_per_lsa_team;
            const int loc  = static_cast<int>(expert) % experts_per_lsa_team;
            row[team * experts_per_lsa_team_packed + loc / 8] |= (1u << (loc % 8));
        }
    }
}

// ============================================================================
// Convert topk to bitmap routing map
// ============================================================================
template <typename TopkIdxT>
void convert_topk_to_routing_map(
    const TopkIdxT* topk_idx,
    uint8_t* routing_bitmap,
    TopkIdxT* cached_topk_idx,
    int num_tokens,
    int max_tokens,
    int num_topk,
    int experts_per_lsa_team,
    int experts_per_lsa_team_packed,
    int row_bytes,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (max_tokens + block_size - 1) / block_size;

    convert_topk_to_routing_map_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx,
        routing_bitmap,
        cached_topk_idx,
        num_tokens,
        max_tokens,
        num_topk,
        experts_per_lsa_team,
        experts_per_lsa_team_packed,
        row_bytes);
}

template void
convert_topk_to_routing_map<int32_t>(const int32_t*, uint8_t*, int32_t*, int, int, int, int, int, int, cudaStream_t);
template void
convert_topk_to_routing_map<int64_t>(const int64_t*, uint8_t*, int64_t*, int, int, int, int, int, int, cudaStream_t);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob
// ============================================================================
template <typename TopkIdxT>
__global__ void sparse_to_dense_prob_kernel(
    const TopkIdxT* __restrict__ topk_idx,     // [num_tokens, topk]
    const float* __restrict__ topk_weights,    // [num_tokens, topk]
    float* __restrict__ dense_prob,            // [num_tokens, num_experts]
    int num_tokens,
    int num_topk,
    int num_experts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token = tid / num_topk;
    int k = tid % num_topk;

    if (token >= num_tokens) return;

    int64_t expert = static_cast<int64_t>(topk_idx[token * num_topk + k]);
    float weight = topk_weights[token * num_topk + k];

    // Scatter weight to the correct expert position
    if (expert >= 0 && expert < num_experts) {
        dense_prob[token * num_experts + expert] = weight;
    }
}

// ============================================================================
// Convert sparse to dense prob
// ============================================================================
template <typename TopkIdxT>
void sparse_to_dense_prob(
    const TopkIdxT* topk_idx,
    const float* topk_weights,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream) {
    int total_elements = num_tokens * num_topk;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    sparse_to_dense_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx,
        topk_weights,
        dense_prob,
        num_tokens,
        num_topk,
        num_experts);
}

template void sparse_to_dense_prob<int32_t>(const int32_t*, const float*, float*, int, int, int, cudaStream_t);
template void sparse_to_dense_prob<int64_t>(const int64_t*, const float*, float*, int, int, int, cudaStream_t);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob for combine input
// ============================================================================
// Used for combine backward pass. Uses local_expert_routing_map to determine
// which experts each token is routed to, matching the order from dispatch output.
// Each thread handles one token.
__global__ void sparse_to_dense_prob_combine_kernel(
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const bool* __restrict__ local_expert_routing_map, // [num_tokens, experts_per_rank]
    float* __restrict__ dense_prob,                   // [num_tokens, experts_per_lsa_team]
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_lsa_team,
    int local_rank) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Scan local experts in order (matches dense_to_sparse_prob output order)
    int k_in = 0;
    for (int e = 0; e < experts_per_rank && k_in < num_topk; e++) {
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // This expert is active for this token - take next weight from sparse input
            float weight = topk_weights[token * num_topk + k_in];

            // Place at correct position in dense matrix
            // Local expert e on local_rank maps to: local_rank * experts_per_rank + e
            int dense_idx = token * experts_per_lsa_team + local_rank * experts_per_rank + e;
            dense_prob[dense_idx] = weight;

            k_in++;
        }
    }
}

// ============================================================================
// Convert sparse to dense prob for combine input
// ============================================================================
void sparse_to_dense_prob_combine(
    const float* topk_weights,
    const bool* local_expert_routing_map,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_lsa_team,
    int local_rank,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    sparse_to_dense_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_weights,
        local_expert_routing_map,
        dense_prob,
        num_tokens,
        num_topk,
        experts_per_rank,
        experts_per_lsa_team,
        local_rank);
}

// ============================================================================
// Kernel: Convert dense prob output to sparse format
// ============================================================================
// One thread per token. Output by layout:
//   FLAT/RM: recv_topk_weights[token, k_out] zero-filled tail; recv_topk_idx parallel.
//   EM:      recv_topk_weights[token] (single scalar; slot = (token, local_expert)); recv_topk_idx unused.
// recv_topk_idx numbering: per-rank local id when kind=LOCAL (default), or
// wire-format global id (= global_expert_offset + local_expert) when kind=GLOBAL.
// kind must be resolved (no AUTO) by the host wrapper.
__global__ void dense_to_sparse_prob_kernel(
    const float* __restrict__ dense_prob,              // [num_recv_tokens, experts_per_lsa_team]
    const bool* __restrict__ local_expert_routing_map, // [num_recv_tokens, experts_per_rank]
    float* __restrict__ recv_topk_weights,             // EM: [N]; FLAT/RM: [N, topk]
    int64_t* __restrict__ recv_topk_idx,               // [num_recv_tokens, topk]; nullptr under EM
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_lsa_team,
    int local_rank,
    int global_expert_offset, // = group_rank * experts_per_rank; added to local id under GLOBAL
    ncclEpExpertIdKind_t recv_topk_idx_kind,
    bool expert_major) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_recv_tokens) return;

    if (expert_major) {
    // Each slot has at most one matching local expert (the one defining the slot).
    // Write the single scalar weight at recv_topk_weights[token]; default 0.
        float weight = 0.0f;
        for (int e = 0; e < experts_per_rank; e++) {
            if (local_expert_routing_map[token * experts_per_rank + e]) {
                int dense_idx = token * experts_per_lsa_team + local_rank * experts_per_rank + e;
                weight = dense_prob[dense_idx];
                break;
            }
        }
        recv_topk_weights[token] = weight;
        return;
    }

    int k_out = 0;

  // Caller must resolve AUTO before kernel launch -- the kernel only
  // understands LOCAL and GLOBAL.
    EP_DEVICE_ASSERT(recv_topk_idx_kind == NCCL_EP_EXPERT_ID_LOCAL || recv_topk_idx_kind == NCCL_EP_EXPERT_ID_GLOBAL);

  // Scan local experts (the ones this rank is responsible for)
    for (int e = 0; e < experts_per_rank && k_out < topk; e++) {
    // Check if this token is routed to expert e
        if (local_expert_routing_map[token * experts_per_rank + e]) {
      // Numbering: LOCAL writes within-rank id e; GLOBAL adds the per-group offset.
            int64_t expert_id = (recv_topk_idx_kind == NCCL_EP_EXPERT_ID_GLOBAL) ?
                                    static_cast<int64_t>(global_expert_offset + e) :
                                    static_cast<int64_t>(e);

      // Get weight from dense output (indexed by local expert within LSA team)
            // dense_prob layout: [token, experts_per_lsa_team] where experts_per_lsa_team = experts_per_rank * ranks_per_lsa_team
            // Local rank's experts are at offset: local_rank * experts_per_rank
            int dense_idx = token * experts_per_lsa_team + local_rank * experts_per_rank + e;
            float weight = dense_prob[dense_idx];

            // Write outputs
            if (recv_topk_idx != nullptr) {
                recv_topk_idx[token * topk + k_out] = expert_id;
            }
            recv_topk_weights[token * topk + k_out] = weight;
            k_out++;
        }
    }

    // Zero-fill remaining topk slots if fewer than topk experts found
    for (; k_out < topk; k_out++) {
        if (recv_topk_idx != nullptr) {
            recv_topk_idx[token * topk + k_out] = -1; // Invalid expert marker
        }
        recv_topk_weights[token * topk + k_out] = 0.0f;
    }
}

// O(top_k) lookup from cached_topk_idx; k-slot order preserves FWD input.
template <typename TopkIdxT>
__global__ void dense_to_sparse_prob_combine_kernel(
    const float* __restrict__ dense_prob, // [num_tokens, num_experts]
    const TopkIdxT* __restrict__ cached_topk_idx, // [num_tokens, topk]
    float* __restrict__ combined_topk_weights, // [num_tokens, topk]
    int num_tokens,
    int topk,
    int num_experts) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    for (int k = 0; k < topk; k++) {
        int64_t e = static_cast<int64_t>(cached_topk_idx[token * topk + k]);
        float weight = (e >= 0 && e < num_experts) ? dense_prob[token * num_experts + e] : 0.0f;
        combined_topk_weights[token * topk + k] = weight;
    }
}

template <typename TopkIdxT>
void dense_to_sparse_prob_combine(
    const float* dense_prob,
    const TopkIdxT* cached_topk_idx,
    float* combined_topk_weights,
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob,
        cached_topk_idx,
        combined_topk_weights,
        num_tokens,
        topk,
        num_experts);
}

template void dense_to_sparse_prob_combine<int32_t>(const float*, const int32_t*, float*, int, int, int, cudaStream_t);
template void dense_to_sparse_prob_combine<int64_t>(const float*, const int64_t*, float*, int, int, int, cudaStream_t);

// ============================================================================
// Dense to sparse prob
// ============================================================================
void dense_to_sparse_prob(
    const float* dense_prob,
    const bool* local_expert_routing_map,
    float* recv_topk_weights,
    int64_t* recv_topk_idx,
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_lsa_team,
    int local_rank,
    int global_expert_offset,
    ncclEpExpertIdKind_t recv_topk_idx_kind,
    bool expert_major,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_recv_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob,
        local_expert_routing_map,
        recv_topk_weights,
        recv_topk_idx,
        num_recv_tokens,
        topk,
        experts_per_rank,
        experts_per_lsa_team,
        local_rank,
        global_expert_offset,
        recv_topk_idx_kind,
        expert_major);
}

// ============================================================================
// Call metadata preprocessing
// ============================================================================
ncclResult_t call_metadata_preprocessing(
    const uint8_t* global_routing_map,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    void* token_rank_mask,
    int32_t* num_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    void* ranks_scan_tmp,
    int lsa_team,
    int local_rank,
    int num_tokens_per_rank,
    int num_lsa_teams,
    int lsa_team_size,
    int experts_per_rank,
    bool expert_major,
    int64_t* internal_offsets,
    void* padded_out_counts,
    void* out_offsets,
    size_t alignment,
    int32_t* actual_counts_out,
    int s2d_inner_dim,
    void* recv_total_counter,
    bool out_is_int64,
    int max_recv_tokens_per_rank,
    int32_t* emuf_group_buf,
    int32_t* emuf_group_count,
    int emuf_group_stride,
    int emuf_max_groups,
    int num_blocks,
    void* scan_gscratch,
    bool em_permute,
    int32_t* token_to_recv_slot,
    int32_t* flat2em_slot_map,
    int em_top_k,
    bool allow_overflow_drop,
    cudaStream_t stream) {
    if (expert_major && per_expert_token_counts == nullptr) {
        std::fprintf(stderr, "[nccl_ep] EXPERT_MAJOR remap requires per_expert_token_counts != nullptr\n");
        return ncclInvalidArgument;
    }
    if (expert_major && scan_gscratch == nullptr) {
        std::fprintf(stderr, "[nccl_ep] EM scan requires scan_gscratch != nullptr\n");
        return ncclInvalidArgument;
    }

    constexpr int NUM_THREADS_PER_BLOCK = NCCL_EP_HT_NUM_THREADS_PER_BLOCK_PREPROCESSING;
    const int NUM_OF_BLOCKS = num_blocks;
    constexpr int NUM_OF_WARPS_PER_BLOCK_SCAN = NUM_THREADS_PER_BLOCK / 32;

    if (expert_major) {
        // The EM scan's gscratch is the shared ep_workspace (NUM_WORKSPACE_BYTES).
        // Verify the selected path's requirement fits before using it.
        const size_t gscratch_needed =
            get_em_scan_gscratch_size(lsa_team_size, experts_per_rank, NUM_OF_BLOCKS, em_permute);
        if (gscratch_needed > NUM_WORKSPACE_BYTES) {
            std::fprintf(stderr,
                         "[nccl_ep] EM scan gscratch (%zu B) exceeds ep_workspace (%zu B) for "
                         "lsa_team_size=%d experts_per_rank=%d num_sms=%d local_permute=%d\n",
                         gscratch_needed, static_cast<size_t>(NUM_WORKSPACE_BYTES), lsa_team_size,
                         experts_per_rank, NUM_OF_BLOCKS, static_cast<int>(em_permute));
            return ncclInvalidUsage;
        }

        if (em_permute) {
            const size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * lsa_team_size * sizeof(::ht_ep::tmp_state_t);
            CUDA_CHECK(cudaMemsetAsync(ranks_scan_tmp, 0, preprocessing_tmp_sz, stream));

            // The shared gscratch (ep_workspace, fit-checked above) doubles as the
            // fused scan's per-expert decoupled-scan state; gscratch_needed is its
            // exact byte size on the local-permute path.
            auto* expert_scan_tmp = reinterpret_cast<::ht_ep::tmp_state_t*>(scan_gscratch);
            CUDA_CHECK(cudaMemsetAsync(expert_scan_tmp, 0, gscratch_needed, stream));

            // Rank region + EM-permute region; sized by the single scan_flat_smem_t layout.
            const int dynamic_smem_bytes = static_cast<int>(::ht_ep::scan_flat_smem_t::byte_size(
                NUM_OF_WARPS_PER_BLOCK_SCAN, lsa_team_size, experts_per_rank,
                /*has_expert_counts=*/false, /*has_em_permute=*/true));

            ::ht_ep::scan_flat_kernel_param_t sp{};
            sp.input_routing_map = global_routing_map;
            sp.tmp = reinterpret_cast<::ht_ep::tmp_state_t*>(ranks_scan_tmp);
            sp.sparse_to_dense_map = sparse_to_dense_map;
            sp.rdma_to_attn_map = rdma_to_attn_map;
            sp.attn_to_rdma_map = attn_to_rdma_map;
            sp.token_rank_mask = token_rank_mask;
            // Initialize Flat parameters
            sp.num_of_tokens_for_experts = num_tokens_for_experts;
            sp.local_expert_routing_map = local_expert_routing_map;
            sp.per_expert_token_counts = nullptr; // unused for Expert-major path
            sp.lsa_team = lsa_team;
            sp.local_rank = local_rank;
            sp.num_of_tokens_per_rank = num_tokens_per_rank;
            sp.experts_per_rank = experts_per_rank;
            sp.recv_total_counter = recv_total_counter;
            sp.out_is_int64 = out_is_int64;
            sp.max_recv_tokens_per_rank = max_recv_tokens_per_rank;
            sp.allow_overflow_drop = allow_overflow_drop;
            sp.token_to_recv_slot = nullptr;  // not needed: recv slot known at emit
            // EM-permute outputs.
            sp.expert_scan_tmp = expert_scan_tmp;
            sp.flat2em_slot_map = flat2em_slot_map;
            sp.em_top_k = em_top_k;
            sp.em_alignment = static_cast<int>(alignment);
            sp.em_internal_offsets = internal_offsets;
            // dtype (int32/int64) is a template parameter
            sp.em_padded_out_counts = padded_out_counts;
            sp.em_out_offsets = out_offsets;
            sp.em_actual_counts_out = actual_counts_out;

            jit::launch_scan_flat(
                NUM_THREADS_PER_BLOCK,
                NUM_OF_BLOCKS,
                num_lsa_teams,
                lsa_team_size,
                experts_per_rank,
                /*enable_per_expert_counts=*/false,
                /*enable_em_permute=*/true,
                out_is_int64,
                sp,
                dynamic_smem_bytes,
                stream);

            (void)s2d_inner_dim;
            (void)emuf_group_buf;
            (void)emuf_group_count;
            (void)emuf_group_stride;
            (void)emuf_max_groups;
            (void)token_to_recv_slot;
            return ncclSuccess;
        } else {
            // nvlink_dup / local_dup EM path: produce only the per-token rank mask + RDMA/attn
            // maps in the scan, then let em_scan_kernel build S2D / LERM / em offsets.
            ::ht_ep::scan_em_kernel_param_t sp;
            sp.input_routing_map = global_routing_map;
            sp.rdma_to_attn_map = rdma_to_attn_map;
            sp.attn_to_rdma_map = attn_to_rdma_map;
            sp.token_rank_mask = token_rank_mask;
            sp.lsa_team = lsa_team;
            sp.local_rank = local_rank;
            sp.num_of_tokens_per_rank = num_tokens_per_rank;
            sp.experts_per_rank = experts_per_rank;

            jit::launch_scan_em(NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, num_lsa_teams, lsa_team_size, sp, stream);
        }

        const int num_mask_words = (lsa_team_size + 63) / 64;
        const int num_total_attn_tokens = num_tokens_per_rank * lsa_team_size * num_lsa_teams;
        launch_build_em_tables(
            global_routing_map,
            token_rank_mask,
            num_mask_words,
            num_total_attn_tokens,
            num_tokens_per_rank,
            lsa_team_size,
            experts_per_rank,
            num_lsa_teams,
            lsa_team,
            local_rank,
            s2d_inner_dim,
            max_recv_tokens_per_rank,
            static_cast<int>(alignment),
            // em-permute: scan_flat_kernel already wrote the unified s2d in
            // FLAT shape; suppress em_scan_kernel's EM-shape writes.
            em_permute ? nullptr : sparse_to_dense_map,
            // Combine gate: em_scan_kernel clears it for fully-dropped send tokens
            // in the non-permute path (where it owns the s2d); the null s2d above
            // disables the clear under em-permute, leaving the gate to the FLAT scan.
            rdma_to_attn_map,
            // em-permute: scan_flat_kernel already wrote the unified FLAT LERM;
            // suppress em_scan_kernel's EM-shape writes.
            em_permute ? nullptr : local_expert_routing_map,
            // em-permute: num_tokens_for_experts already holds FLAT num_recv.
            em_permute ? nullptr : num_tokens_for_experts,
            internal_offsets,
            padded_out_counts,
            out_offsets,
            actual_counts_out,
            recv_total_counter,
            out_is_int64,
            emuf_group_buf,
            emuf_group_count,
            emuf_group_stride,
            emuf_max_groups,
            static_cast<int32_t*>(scan_gscratch),
            NUM_OF_BLOCKS,
            em_permute ? token_to_recv_slot : nullptr,
            em_permute ? flat2em_slot_map : nullptr,
            em_permute ? em_top_k : 0,
            allow_overflow_drop,
            stream);
        return ncclSuccess;
    }

    // FLAT path.
    if (per_expert_token_counts != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(per_expert_token_counts, 0, experts_per_rank * sizeof(int32_t), stream));
    }

    const size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * lsa_team_size * sizeof(::ht_ep::tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(ranks_scan_tmp, 0, preprocessing_tmp_sz, stream));

    // Rank region (+ optional per-expert counts); sized by the single scan_flat_smem_t layout.
    const int dynamic_smem_bytes = static_cast<int>(::ht_ep::scan_flat_smem_t::byte_size(
        NUM_OF_WARPS_PER_BLOCK_SCAN, lsa_team_size, experts_per_rank,
        /*has_expert_counts=*/per_expert_token_counts != nullptr, /*has_em_permute=*/false));

    ::ht_ep::scan_flat_kernel_param_t sp;
    sp.input_routing_map = global_routing_map;
    sp.tmp = reinterpret_cast<::ht_ep::tmp_state_t*>(ranks_scan_tmp);
    sp.sparse_to_dense_map = sparse_to_dense_map;
    sp.rdma_to_attn_map = rdma_to_attn_map;
    sp.attn_to_rdma_map = attn_to_rdma_map;
    sp.token_rank_mask = token_rank_mask;
    sp.num_of_tokens_for_experts = num_tokens_for_experts;
    sp.local_expert_routing_map = local_expert_routing_map;
    sp.per_expert_token_counts = per_expert_token_counts;
    sp.lsa_team = lsa_team;
    sp.local_rank = local_rank;
    sp.num_of_tokens_per_rank = num_tokens_per_rank;
    sp.experts_per_rank = experts_per_rank;
    sp.recv_total_counter = recv_total_counter;
    sp.out_is_int64 = out_is_int64;
    sp.max_recv_tokens_per_rank = max_recv_tokens_per_rank;
    sp.allow_overflow_drop = allow_overflow_drop;
    sp.token_to_recv_slot = nullptr;

    jit::launch_scan_flat(
        NUM_THREADS_PER_BLOCK,
        NUM_OF_BLOCKS,
        num_lsa_teams,
        lsa_team_size,
        experts_per_rank,
        per_expert_token_counts != nullptr,
        /*enable_em_permute=*/false,
        out_is_int64,
        sp,
        dynamic_smem_bytes,
        stream);

    // Suppress unused-parameter warnings for EM-only outputs.
    (void)internal_offsets;
    (void)padded_out_counts;
    (void)out_offsets;
    (void)actual_counts_out;
    (void)alignment;
    (void)s2d_inner_dim;
    (void)scan_gscratch;
    return ncclSuccess;
}

size_t get_preprocessing_scan_tmp_size(int num_blocks, int lsa_team_size) {
    return static_cast<size_t>(num_blocks) * lsa_team_size * sizeof(::ht_ep::tmp_state_t);
}

size_t get_rank_mask_elem_size(int lsa_team_size) {
    return ((lsa_team_size + 63) / 64) * sizeof(uint64_t);
}


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
    int32_t* emuf_group_buf,
    int32_t* emuf_group_count,
    int emuf_group_stride,
    int emuf_max_groups,
    int32_t* gscratch,
    int num_sms,
    const int32_t* token_to_recv_slot,
    int32_t* flat2em_slot_map,
    int em_top_k,
    bool allow_overflow_drop,
    cudaStream_t stream) {
    if (num_total_attn_tokens <= 0 || lsa_team_size <= 0 || experts_per_rank <= 0) return;
    assert((experts_per_rank & (experts_per_rank - 1)) == 0 && "experts_per_rank must be a power of two");
    assert(num_mask_words >= 1 && num_mask_words <= 2 && "lsa_team_size must be <= 128");
    assert(num_sms > 0 && "launch_build_em_tables requires num_sms > 0");
    const int n_dle = lsa_team_size * experts_per_rank;

    constexpr int kNumWarps = jit::kBuildEmTablesBlockDim / 32;
    const size_t smem_bytes = static_cast<size_t>(kNumWarps + 1) * n_dle * sizeof(int32_t);

    ::ht_ep::build_em_tables_param_t p{};
    p.input_routing_map        = input_routing_map;
    p.token_rank_mask_words    = static_cast<const uint64_t*>(token_rank_mask);
    p.num_mask_words           = num_mask_words;
    p.num_total_attn_tokens    = num_total_attn_tokens;
    p.num_tokens_per_rank      = num_tokens_per_rank;
    p.num_lsa_teams            = num_lsa_teams;
    p.lsa_team                = lsa_team;
    p.local_rank               = local_rank;
    p.s2d_inner_dim            = s2d_inner_dim;
    p.max_recv_tokens_per_rank = max_recv_tokens_per_rank;
    p.em_alignment             = em_alignment;
    p.sparse_to_dense_map      = sparse_to_dense_map;
    // Drop policy: em_scan owns the combine gate here; pass the map only when
    // dropping is enabled so the device side clears it for fully-dropped tokens.
    p.rdma_to_attn_map         = allow_overflow_drop ? rdma_to_attn_map : nullptr;
    p.local_expert_routing_map = local_expert_routing_map;
    p.num_tokens_for_experts   = num_tokens_for_experts;
    p.em_internal_offsets      = em_internal_offsets;
    p.em_padded_out_counts_i32 = out_is_int64 ? nullptr : static_cast<int32_t*>(em_padded_out_counts);
    p.em_padded_out_counts_i64 = out_is_int64 ? static_cast<int64_t*>(em_padded_out_counts) : nullptr;
    p.em_out_offsets_i32       = out_is_int64 ? nullptr : static_cast<int32_t*>(em_out_offsets);
    p.em_out_offsets_i64       = out_is_int64 ? static_cast<int64_t*>(em_out_offsets) : nullptr;
    p.em_actual_counts_out     = em_actual_counts_out;
    p.recv_total_counter_i32   = out_is_int64 ? nullptr : static_cast<int32_t*>(recv_total_counter);
    p.recv_total_counter_i64   = out_is_int64 ? static_cast<int64_t*>(recv_total_counter) : nullptr;
    p.out_is_int64             = out_is_int64;
    p.emuf_group_buf           = emuf_group_buf;
    p.emuf_group_count         = emuf_group_count;
    p.emuf_group_stride        = emuf_group_stride;
    p.emuf_max_groups          = emuf_max_groups;
    p.gscratch                 = gscratch;
    p.token_to_recv_slot       = token_to_recv_slot;
    p.flat2em_slot_map         = flat2em_slot_map;
    p.em_top_k                 = em_top_k;
    p.allow_overflow_drop      = allow_overflow_drop;

    jit::launch_build_em_tables_jit(experts_per_rank, lsa_team_size, p, static_cast<int>(smem_bytes),
                                    num_sms, stream);
}

size_t get_em_scan_gscratch_size(int lsa_team_size, int experts_per_rank, int num_sms, bool is_local_permute) {
    assert(num_sms > 0);
    if (is_local_permute) {
        // Fused em-permute scan: per-expert decoupled-scan state
        // expert_scan_tmp[num_sms * experts_per_rank] tmp_state_t (independent of nrpn).
        return static_cast<size_t>(num_sms) * experts_per_rank * sizeof(::ht_ep::tmp_state_t);
    }
    // em_scan_kernel (kLocalDup / nvlink_dup path): block_count[num_sms][nrpn*epr] int32.
    return static_cast<size_t>(num_sms) * lsa_team_size * experts_per_rank * sizeof(int32_t);
}

int get_device_max_dynamic_smem() {
    int device = 0;
    int max_smem = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return max_smem;
}

static int env_or_default(const ncclEpEnvVar* var, int default_value) {
    if (var == nullptr || !var->is_set) return default_value;
    return var->value.ul <= static_cast<unsigned long>(INT_MAX) ? static_cast<int>(var->value.ul) : 0;
}

// Result of fitting the dispatch SMEM config to the device limit.
struct dispatch_smem_fit_t {
    int num_pipelines;
    int num_of_stages;  // total stages (== num_pipelines * stages_per_pipeline)
    size_t smem_size;
    bool feasible;
};

// Choose (pipelines, total stages) that fit within the SMEM budget.
//
// Uses the bilinear coefficients (size = fixed + P*per_pipeline + S*per_stage,
// rounded to 128B) to solve for the stage count directly -- no binary search.
//   - Any env-pinned value is honored exactly and never reduced.
//   - Pipelines are scanned from the requested count down to 1 (skipped when
//     env-pinned); each candidate uses the most stages that fit, capped at the
//     requested target and floored at min_stages_per_pipeline per pipeline.
//   - Among feasible candidates the one wasting the least SMEM wins; ties favor
//     more pipelines (the scan starts high and only a strictly larger footprint
//     replaces the incumbent).
static dispatch_smem_fit_t choose_dispatch_smem_config(
    const ::ht_ep::disp_smem_cost_t& terms,
    size_t budget,
    int target_stages,
    bool stages_fixed,
    int target_pipelines,
    bool pipelines_fixed,
    int min_stages_per_pipeline,
    int max_pipelines) {
    dispatch_smem_fit_t best{0, 0, 0, false};
    const int hi_p = pipelines_fixed ? target_pipelines : std::min(target_pipelines, max_pipelines);
    const int lo_p = pipelines_fixed ? target_pipelines : 1;

    for (int P = hi_p; P >= lo_p && P >= 1; --P) {
        int stages = 0;
        if (stages_fixed) {
            // Env-pinned stage count: use exactly, only if it is valid for this P.
            if (target_stages % P != 0 || target_stages / P < min_stages_per_pipeline) continue;
            stages = target_stages;
        } else {
            // Largest stages-per-pipeline within budget, capped at the target.
            // smem <= budget  <=>  pre-round sum <= (budget rounded down to 128B).
            const size_t base = terms.fixed + static_cast<size_t>(P) * terms.per_pipeline;
            const size_t budget_presum = budget & ~static_cast<size_t>(127);
            if (terms.per_stage == 0 || base > budget_presum) continue;
            const size_t spp_by_budget =
                (budget_presum - base) / (static_cast<size_t>(P) * terms.per_stage);
            const int spp = static_cast<int>(
                std::min<size_t>(spp_by_budget, static_cast<size_t>(target_stages / P)));
            if (spp < min_stages_per_pipeline) continue;
            stages = spp * P;
        }
        const size_t smem = ::ht_ep::calc_disp_smem(terms, P, stages);
        if (smem > budget) continue;
        if (!best.feasible || smem > best.smem_size) {
            best = dispatch_smem_fit_t{P, stages, smem, true};
        }
    }
    return best;
}

// Result of fitting the combine SMEM config to the device limit.
struct combine_smem_fit_t {
    int num_pipelines;
    int num_of_stages_g2s;
    int num_of_stages_s2g;
    size_t smem_size;
    bool feasible;
};

// Choose (pipelines, g2s stages, s2g stages) that fit within the SMEM budget.
//
// Combine performs accumulation, so it prioritizes keeping the pipeline count:
// unlike dispatch there is NO min-waste search across pipeline counts. The scan
// starts at the requested pipeline count and the FIRST value that admits a fit
// wins; a lower count is tried only when the current one cannot fit even at the
// minimum stage depth. Within a pipeline count, both stage axes are capped at
// their requested targets and reduced to fit, shaving G2S first (the deeper
// buffer) and only then S2G. Any env-pinned value is honored exactly.
static combine_smem_fit_t choose_combine_smem_config(
    const ::ht_ep::comb_smem_cost_t& cost,
    size_t budget,
    int req_g2s, bool g2s_fixed,
    int req_s2g, bool s2g_fixed,
    int req_pipelines, bool pipelines_fixed,
    int min_stages_per_pipeline,
    int red_warps) {
    combine_smem_fit_t best{0, 0, 0, 0, false};
    const size_t budget_presum = budget & ~static_cast<size_t>(127);

    // Largest multiple of P in [.., hi] whose axis contribution (var * unit)
    // keeps the footprint within budget while the other axis holds `other_bytes`.
    // May return a value below the caller's lo, signalling "does not fit".
    auto max_multiple = [&](int hi, size_t unit, size_t other_bytes, int P) -> int {
        if (cost.fixed + other_bytes > budget_presum) return -1;
        if (unit == 0) return hi;  // this axis is free -> take the cap
        const size_t room = budget_presum - cost.fixed - other_bytes;
        const int by_budget = static_cast<int>(std::min<size_t>(static_cast<size_t>(hi), room / unit));
        return (by_budget / P) * P;  // round down to a multiple of P
    };

    for (int P = req_pipelines; P >= 1; --P) {
        if (pipelines_fixed && P != req_pipelines) break;
        if (red_warps % P != 0) continue;  // pipelines must divide the reduction warps
        // Pinned stages must be a multiple of this P to be usable.
        if (g2s_fixed && req_g2s % P != 0) continue;
        if (s2g_fixed && req_s2g % P != 0) continue;

        const int g2s_hi = g2s_fixed ? req_g2s : (req_g2s / P) * P;
        const int s2g_hi = s2g_fixed ? req_s2g : (req_s2g / P) * P;
        const int g2s_lo = g2s_fixed ? req_g2s : P * min_stages_per_pipeline;
        const int s2g_lo = s2g_fixed ? req_s2g : P * min_stages_per_pipeline;
        if (g2s_hi < g2s_lo || s2g_hi < s2g_lo) continue;

        // Reduce G2S first, holding S2G at its cap; if even the minimum G2S with
        // full S2G overflows, pin G2S at its minimum and reduce S2G instead.
        int g2s = max_multiple(g2s_hi, cost.per_g2s_stage,
                               static_cast<size_t>(s2g_hi) * cost.per_s2g_stage, P);
        int s2g = s2g_hi;
        if (g2s < g2s_lo) {
            g2s = g2s_lo;
            s2g = max_multiple(s2g_hi, cost.per_s2g_stage,
                               static_cast<size_t>(g2s_lo) * cost.per_g2s_stage, P);
            if (s2g < s2g_lo) continue;  // infeasible for this pipeline count
        }
        const size_t smem = ::ht_ep::calc_comb_smem(cost, g2s, s2g);
        if (smem > budget) continue;
        best = combine_smem_fit_t{P, g2s, s2g, smem, true};
        break;  // first (highest) feasible pipeline count wins
    }
    return best;
}

// ============================================================================
// Dispatch wrapper implementation
// ============================================================================

// Helper to populate the fixed-size dispatch parameter fields from DispatchParams.
template <typename TOKEN_DATA_TYPE>
::ht_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE> build_dispatch_param_base(const DispatchParams& params) {
    ::ht_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE> kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.ranks_per_lsa_team = params.lsa_team_size;
    // User input buffers
    kp.attn_input_token = reinterpret_cast<const TOKEN_DATA_TYPE*>(params.attn_input_token);
    kp.attn_input_prob = params.attn_input_prob;
    kp.attn_input_token_scaling_factor = static_cast<const uint8_t*>(params.attn_input_scaling_factor);

    // Metadata and sync flags
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.pad_actual_counts = params.pad_actual_counts;
    kp.pad_expert_token_offsets = params.pad_expert_token_offsets;
    kp.pad_alignment = params.pad_alignment;
    kp.expected_gin_flag_val = params.expected_gin_flag_val;
    kp.expected_lsa_flag_val = params.expected_lsa_flag_val;
    kp.gin_G2S_flags = params.gin_G2S_flags;
    kp.lsa_S2G_flags = params.lsa_S2G_flags;
    kp.dispatch_grid_barrier_counter = params.dispatch_grid_barrier_counter;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.lsa_team = params.lsa_team;
    kp.num_of_tokens_per_rank = params.tokens_per_lsa;
    kp.local_dup_enabled = (params.local_dup_num_sms > 0);
    kp.guard_enabled = params.guard_enabled;
    kp.max_recv_tokens_per_rank = params.max_recv_tokens_per_rank;

    // Pass device communicators and windows
    kp.dcomm = params.dcomm;
    kp.token_window = params.nccl_token_window;
    kp.prob_window = params.nccl_prob_window;
    kp.sf_window = params.nccl_sf_window;
    kp.dest_window = params.nccl_internal_window;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
        .attn_input_token_offset = params.mr_info.attn_input_token_offset,
        .attn_input_prob_offset = params.mr_info.attn_input_prob_offset,
        .attn_input_scaling_factor_offset = params.mr_info.attn_input_scaling_factor_offset,
        // Batched staging parameters (packed layout)
        .gin_send_staging_offset = params.mr_info.gin_send_staging_offset,
        .gin_recv_staging_offset = params.mr_info.gin_recv_staging_offset,
        .guard_offset = params.mr_info.guard_offset,
        .bytes_per_entry = params.mr_info.bytes_per_entry,
        .max_tokens_per_dest = params.mr_info.max_tokens_per_dest,
        // Streaming signal parameters
        .signals_tail_base = params.mr_info.signals_tail_base,
        .num_max_rdma_chunked_send_tokens = params.mr_info.num_max_rdma_chunked_send_tokens
    };

    return kp;
}

template <typename TOKEN_DATA_TYPE>
std::vector<uint8_t> build_dispatch_arg_buffer(
    const ::ht_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>& kp,
    const DispatchParams& params) {
    using ParamBase = ::ht_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>;
    static_assert(sizeof(ParamBase) % alignof(void*) == 0);

    const size_t base_size = sizeof(ParamBase);
    const size_t token_offset = base_size;
    const size_t prob_offset = token_offset + params.lsa_team_size * sizeof(TOKEN_DATA_TYPE*);
    const size_t sf_offset = prob_offset + params.lsa_team_size * sizeof(float*);
    const size_t total_size = sf_offset + params.lsa_team_size * sizeof(float*);

    std::vector<uint8_t> arg(total_size);
    std::memcpy(arg.data(), &kp, sizeof(kp));

    auto* token_ptrs = reinterpret_cast<TOKEN_DATA_TYPE**>(arg.data() + token_offset);
    auto* prob_ptrs = reinterpret_cast<float**>(arg.data() + prob_offset);
    auto* sf_ptrs = reinterpret_cast<uint8_t**>(arg.data() + sf_offset);
    for (int i = 0; i < params.lsa_team_size; i++) {
        token_ptrs[i] = reinterpret_cast<TOKEN_DATA_TYPE*>(params.expert_output_token_ptrs[i]);
        prob_ptrs[i] = params.expert_output_prob_ptrs ? params.expert_output_prob_ptrs[i] : nullptr;
        sf_ptrs[i] = nullptr;
        if (params.expert_output_scaling_factor_ptrs) {
            sf_ptrs[i] = static_cast<uint8_t*>(params.expert_output_scaling_factor_ptrs[i]);
        }
    }

    return arg;
}

// Host dispatch launcher. The JIT source owns all device-kernel specialization;
// the host only asks ht_ep for the matching dynamic-SMEM size.
ncclResult_t dispatch_impl(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_lsa_teams,
    ncclEpPassDir_t pass_direction,
    int num_blocks,
    int sf_bytes_per_token,
    const ncclEpEnvConfig* env,
    cudaStream_t stream,
    const DispatchKernelSpec& kernel_spec) {
    {
        // env is a required argument: callers pass &group->env, which is never null.
        assert(env != nullptr && "dispatch_impl requires a non-null env config");
        const bool forward_dispatch = (pass_direction == NCCL_EP_FWD_PASS);
        // The dispatch param/arg buffers are pointer-only (wire-width-invariant), so the
        // host packs with one fixed type; the JIT specializes the actual kernel by
        // token_dtype (dispatch_token_data_type_literal in launch_dispatch). No host-side
        // compile-time token-type switch is needed -- rely on the JIT.
        using TOKEN_DATA_TYPE = uint16_t;
        // TMA requires prob buffer (experts_per_lsa_team * sizeof(float)) to be 16B aligned
        // Check alignment at runtime now that experts_per_rank is dynamic
        const int experts_per_lsa_team = params.experts_per_rank * params.lsa_team_size;
        assert(
            (experts_per_lsa_team * sizeof(float)) % 16 == 0 && "experts_per_lsa_team must be multiple of 4 for TMA alignment");
        // 16B cp.async.bulk alignment for the S2D map fetch; matters when s2d_inner_dim < 4.
        assert(
            (static_cast<int64_t>(params.tokens_per_lsa) * params.s2d_inner_dim) % 4 == 0 &&
            "Dispatch S2D cp.async.bulk: num_tokens_per_rank * s2d_inner_dim must be a "
            "multiple of 4 (flat layout with lsa_team_size <= 3 requires even num_tokens_per_rank)");

        auto kp = build_dispatch_param_base<TOKEN_DATA_TYPE>(params);

        ::ht_ep::dispatch_config_t d_config{};
        d_config.num_of_stages = env_or_default(
            &env->dispatch_num_stages, NCCL_EP_HT_DISPATCH_STAGES);
        ::ht_ep::model_config_t d_model;
        d_config.num_pipelines = env_or_default(
            &env->dispatch_num_pipelines, NCCL_EP_HT_DISPATCH_PIPELINES);
        d_config.num_of_in_flight_s2g = NCCL_EP_HT_DISPATCH_IN_FLIGHT_S2G;
        d_config.num_of_tokens_per_chunk = num_tokens_per_chunk;
        d_config.num_of_blocks = num_blocks;
        d_config.forward_dispatch = forward_dispatch;
        d_config.sf_bytes_per_token = sf_bytes_per_token;
        d_config.s2d_inner_dim = kp.s2d_inner_dim;
        if (d_config.num_of_stages <= 0 || d_config.num_pipelines <= 0) {
            std::fprintf(stderr, "[nccl_ep] invalid dispatch config: stages=%d, pipelines=%d\n",
                         d_config.num_of_stages, d_config.num_pipelines);
            return ncclInvalidArgument;
        }

        d_model.hidden_dim = kp.hidden_dim;
        d_model.max_num_of_tokens_per_rank = max_dispatch_tokens_per_rank;
        d_model.num_of_experts_per_rank = kp.experts_per_rank;
        d_model.ranks_per_lsa_team = kp.ranks_per_lsa_team;
        d_model.num_lsa_teams = num_lsa_teams;

        // Requested config (env-pinned values are honored exactly, never reduced).
        const bool stages_from_env = env->dispatch_num_stages.is_set;
        const bool pipelines_from_env = env->dispatch_num_pipelines.is_set;
        const int requested_stages = d_config.num_of_stages;
        const int requested_pipelines = d_config.num_pipelines;
        const int requested_in_flight = d_config.num_of_in_flight_s2g;
        // Absolute floor: a pipeline needs at least this many stages for the FIFO
        // to make progress. The S2G-overlap requirement (stages/pipeline strictly
        // greater than in_flight_s2g) is layered on during fitting, but since
        // in_flight_s2g can itself be reduced, it does not gate validity here.
        constexpr int kMinStagesPerPipeline = 3;

        // A fully env-pinned config must be internally valid on its own -- the
        // fitter reshapes neither count, so it cannot fix a bad combination.
        if (stages_from_env && pipelines_from_env &&
            (requested_stages % requested_pipelines != 0 ||
             requested_stages / requested_pipelines < kMinStagesPerPipeline)) {
            std::fprintf(
                stderr,
                "[nccl_ep] invalid dispatch config: stages=%d must be a multiple of pipelines=%d "
                "with at least %d stages per pipeline.\n",
                requested_stages, requested_pipelines, kMinStagesPerPipeline);
            return ncclInvalidArgument;
        }

        // Warp budget caps the pipeline count (2 warps per pipeline).
        const int fixed_warps =
            (num_lsa_teams != 1 ? NCCL_EP_HT_DISPATCH_N2N_WARPS : 0) +
            (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? 1 : 0);
        const int max_pipelines = (32 - fixed_warps) / 2;
        if (pipelines_from_env && requested_pipelines > max_pipelines) {
            std::fprintf(stderr, "[nccl_ep] dispatch pipelines=%d exceeds block limit; maximum=%d.\n",
                         requested_pipelines, max_pipelines);
            return ncclInvalidArgument;
        }

        // Bilinear SMEM coefficients: size = fixed + P*per_pipeline + S*per_stage.
        const ::ht_ep::disp_smem_cost_t terms = ::ht_ep::calc_disp_smem_cost(
            params.layout, kernel_spec.payload_bytes, d_config, d_model);
        if (terms.per_stage == 0) {
            std::fprintf(stderr, "NCCL EP warning: unsupported HT dispatch token size %u\n",
                         kernel_spec.payload_bytes);
            return ncclInvalidArgument;
        }

        const int max_smem = get_device_max_dynamic_smem();
        // Fit within the SMEM budget. If the requested in_flight_s2g admits no
        // fit (its min-stages floor is too tall), reduce it toward 1 -- lowering
        // in_flight_s2g lowers the required stages/pipeline. in_flight_s2g is
        // never env-pinned, so it is always free to reduce; the largest value
        // that fits is kept, preserving as much S2G overlap as possible.
        dispatch_smem_fit_t fit{0, 0, 0, false};
        int chosen_in_flight = requested_in_flight;
        for (int in_flight = requested_in_flight; in_flight >= 1; --in_flight) {
            const int min_stages_per_pipeline = std::max(kMinStagesPerPipeline, in_flight + 1);
            fit = choose_dispatch_smem_config(
                terms, static_cast<size_t>(max_smem),
                requested_stages, stages_from_env,
                requested_pipelines, pipelines_from_env,
                min_stages_per_pipeline, max_pipelines);
            if (fit.feasible) {
                chosen_in_flight = in_flight;
                break;
            }
        }

        const size_t requested_smem =
            ::ht_ep::calc_disp_smem(terms, requested_pipelines, requested_stages);
        if (!fit.feasible) {
            std::fprintf(
                stderr,
                "[nccl_ep] dispatch shared memory cannot be fit to the device limit: requested "
                "stages=%d, pipelines=%d need %zu bytes, limit=%d bytes%s.\n",
                requested_stages, requested_pipelines, requested_smem, max_smem,
                (stages_from_env || pipelines_from_env) ? " (env-pinned values cannot be reduced)" : "");
            return ncclInvalidArgument;
        }

        d_config.num_of_in_flight_s2g = chosen_in_flight;
        d_config.num_pipelines = fit.num_pipelines;
        d_config.num_of_stages = fit.num_of_stages;
        d_config.stages_per_pipeline = fit.num_of_stages / fit.num_pipelines;
        const size_t smem_size = fit.smem_size;

        // Report the resolved config (once per distinct outcome) when verbose is
        // requested -- always, whether or not auto-fit changed anything.
        if (nccl_ep_env_verbose(*env)) {
            std::ostringstream key;
            key << "ht_dispatch_smem_fit:" << requested_stages << ':' << requested_pipelines << ':'
                << requested_in_flight << ':' << d_config.num_of_stages << ':' << d_config.num_pipelines
                << ':' << d_config.num_of_in_flight_s2g << ':' << smem_size;
            if (::nccl_ep::jit::announce_once(key.str())) {
                std::fprintf(
                    stderr,
                    "[nccl_ep][env] HT dispatch SMEM fit:\n"
                    "[nccl_ep][env]   stages=%d (req %d), pipelines=%d (req %d), in_flight_s2g=%d (req %d)\n"
                    "[nccl_ep][env]   smem=%zu bytes (req %zu, limit %d)\n"
                    "[nccl_ep][env]   cost: fixed=%zu, per_pipeline=%zu, per_stage=%zu bytes\n",
                    d_config.num_of_stages, requested_stages,
                    d_config.num_pipelines, requested_pipelines,
                    d_config.num_of_in_flight_s2g, requested_in_flight,
                    smem_size, requested_smem, max_smem,
                    terms.fixed, terms.per_pipeline, terms.per_stage);
            }
        }

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
        const jit::dispatch_warp_layout_t dispatch_layout =
            jit::compute_dispatch_warp_layout(num_lsa_teams, params.layout, d_config.num_pipelines);
        const int dispatch_wt_total = num_blocks * (dispatch_layout.block_dim / 32);
        ::ht_ep::dispatch_warp_timing_entry_t* d_wt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_wt, dispatch_wt_total * sizeof(::ht_ep::dispatch_warp_timing_entry_t)));
        CUDA_CHECK(
            cudaMemsetAsync(d_wt, 0, dispatch_wt_total * sizeof(::ht_ep::dispatch_warp_timing_entry_t), stream));
        kp.warp_timing = d_wt;
#endif

        std::vector<uint8_t> kernel_arg = build_dispatch_arg_buffer(kp, params);
        if (ncclResult_t r = jit::launch_dispatch(
            d_config,
            max_dispatch_tokens_per_rank,
            num_lsa_teams,
            params.lsa_team_size,
            params.layout,
            kp.hidden_dim,
            sf_bytes_per_token,
            env,
            kernel_arg.data(),
            kernel_arg.size(),
            static_cast<int>(smem_size),
            stream,
            kernel_spec); r != ncclSuccess)
            return r;

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
        jit::dispatch_dump_warp_timing(dispatch_layout, num_blocks, d_wt, stream);
        CUDA_CHECK(cudaFree(d_wt));
#endif
    }
    return ncclSuccess;
}

ncclResult_t call_dispatch(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_lsa_teams,
    ncclEpDispQuant_t recipe,
    ncclEpPassDir_t pass_direction,
    int num_blocks,
    int sf_bytes_per_token,
    const ncclEpEnvConfig* env,
    cudaStream_t stream,
    ncclDataType_t token_dtype) {
    DispatchKernelSpec kernel_spec;
    if (ncclResult_t r = resolveDispatchKernelSpec(recipe, token_dtype, params.scale_dtype, &kernel_spec);
        r != ncclSuccess) {
        return r;
    }
    return dispatch_impl(
        params, max_dispatch_tokens_per_rank, num_tokens_per_chunk, num_lsa_teams,
        pass_direction, num_blocks, sf_bytes_per_token, env, stream, kernel_spec);
}

// ============================================================================
// Combine wrapper implementation
// ============================================================================

// Helper to populate the fixed-size combine parameter fields from CombineParams.
::ht_ep::combine_kernel_param_base_t build_combine_param_base(const CombineParams& params) {
    ::ht_ep::combine_kernel_param_base_t kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.ranks_per_lsa_team = params.lsa_team_size;
    // User output buffers
    kp.attn_output_token = reinterpret_cast<uint16_t*>(params.attn_output_token);
    kp.attn_output_prob = params.attn_output_prob;

    // RDMA buffers (multiple LSA teams only)
    kp.combine_gin_RED_tokens = params.combine_gin_RED_tokens;
    kp.combine_gin_RED_prob = params.combine_gin_RED_prob;
    kp.combine_gin_G2S_tokens = params.combine_gin_G2S_tokens;
    kp.combine_gin_G2S_prob = params.combine_gin_G2S_prob;

    // Metadata
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;

    // Sync flags
    kp.expected_gin_flag_val = params.expected_gin_flag_val;
    kp.expected_lsa_flag_val = params.expected_lsa_flag_val;
    kp.gin_G2S_flags = params.gin_G2S_flags;
    kp.lsa_S2G_flags = params.lsa_S2G_flags;
    kp.combine_grid_barrier_counter = params.combine_grid_barrier_counter;
    kp.guard_enabled = params.guard_enabled;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.lsa_team = params.lsa_team;
    kp.num_of_tokens_per_rank = params.tokens_per_lsa;
    kp.num_real_tokens = params.num_real_tokens;
    kp.combine_local_reduce_enabled = params.combine_local_reduce_enabled;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.token_window = params.nccl_token_window;
    kp.prob_window = params.nccl_prob_window;
    kp.dest_window = params.nccl_internal_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    kp.combine_signal_offset = params.combine_signal_offset;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
        .combine_red_token_offset = params.mr_info.combine_red_token_offset,
        .combine_g2s_token_offset = params.mr_info.combine_g2s_token_offset,
        .combine_red_prob_offset = params.mr_info.combine_red_prob_offset,
        .combine_g2s_prob_offset = params.mr_info.combine_g2s_prob_offset,
        .guard_offset = params.mr_info.guard_offset
    };

    return kp;
}

std::vector<uint8_t> build_combine_arg_buffer(
    const ::ht_ep::combine_kernel_param_base_t& kp,
    const CombineParams& params) {
    using ParamBase = ::ht_ep::combine_kernel_param_base_t;
    static_assert(sizeof(ParamBase) % alignof(void*) == 0);

    const size_t base_size = sizeof(ParamBase);
    const size_t token_offset = base_size;
    const size_t prob_offset = token_offset + params.lsa_team_size * sizeof(uint16_t*);
    const size_t total_size = prob_offset + params.lsa_team_size * sizeof(float*);

    std::vector<uint8_t> arg(total_size);
    std::memcpy(arg.data(), &kp, sizeof(kp));

    auto* token_ptrs = reinterpret_cast<uint16_t**>(arg.data() + token_offset);
    auto* prob_ptrs = reinterpret_cast<float**>(arg.data() + prob_offset);
    for (int i = 0; i < params.lsa_team_size; i++) {
        token_ptrs[i] = params.expert_input_token_ptrs[i];
        prob_ptrs[i] = params.expert_input_prob_ptrs ? params.expert_input_prob_ptrs[i] : nullptr;
    }

    return arg;
}

// Template combine launcher for forward/backward
template <bool BACKWARD_COMBINE>
ncclResult_t combine_impl(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_lsa_teams,
    int num_blocks,
    const ncclEpEnvConfig* env,
    cudaStream_t stream) {
    // env is a required argument: callers pass &group->env, which is never null.
    assert(env != nullptr && "combine_impl requires a non-null env config");
    // TMA requires prob buffer (experts_per_lsa_team * sizeof(float)) to be 16B aligned
    const int experts_per_lsa_team = params.experts_per_rank * params.lsa_team_size;
    assert((experts_per_lsa_team * sizeof(float)) % 16 == 0 && "experts_per_lsa_team must be multiple of 4 for TMA alignment");

    auto kp = build_combine_param_base(params);

    const bool multi_lsa = (num_lsa_teams != 1);
    ::ht_ep::combine_config_t c_config{};
    c_config.num_of_stages_g2s = env_or_default(
        &env->combine_num_stages_g2s,
        multi_lsa ? NCCL_EP_HT_COMBINE_CROSS_LSA_STAGES_G2S :
                    NCCL_EP_HT_COMBINE_LSA_STAGES_G2S);
    c_config.num_of_stages_s2g = env_or_default(
        &env->combine_num_stages_s2g,
        multi_lsa ? NCCL_EP_HT_COMBINE_CROSS_LSA_STAGES_S2G :
                    NCCL_EP_HT_COMBINE_LSA_STAGES_S2G);
    c_config.num_pipelines = env_or_default(
        &env->combine_num_pipelines,
        multi_lsa ? NCCL_EP_HT_COMBINE_CROSS_LSA_PIPELINES :
                    NCCL_EP_HT_COMBINE_LSA_PIPELINES);
    c_config.num_of_tokens_per_chunk = num_tokens_per_chunk;
    c_config.num_of_tokens_per_group = NCCL_EP_HT_COMBINE_TOK_PER_GROUP;
    c_config.num_of_blocks = num_blocks;
    c_config.backward_combine = BACKWARD_COMBINE;
    // Requested config (env-pinned values are honored exactly, never reduced).
    const bool g2s_from_env = env->combine_num_stages_g2s.is_set;
    const bool s2g_from_env = env->combine_num_stages_s2g.is_set;
    const bool pipelines_from_env = env->combine_num_pipelines.is_set;
    const int requested_g2s_stages = c_config.num_of_stages_g2s;
    const int requested_s2g_stages = c_config.num_of_stages_s2g;
    const int requested_pipelines = c_config.num_pipelines;
    constexpr int kMinStagesPerPipeline = 1;  // combine needs >= 1 stage/pipeline per axis

    if (requested_g2s_stages <= 0 || requested_s2g_stages <= 0 || requested_pipelines <= 0) {
        std::fprintf(stderr, "[nccl_ep] invalid combine config: g2s_stages=%d, s2g_stages=%d, pipelines=%d\n",
                     requested_g2s_stages, requested_s2g_stages, requested_pipelines);
        return ncclInvalidArgument;
    }
    // An env-pinned pipeline count must divide the reduction warps, and any
    // env-pinned stage count must be a multiple of it -- the fitter reshapes
    // only non-pinned counts and cannot fix a bad pinned combination.
    if (pipelines_from_env &&
        (NCCL_EP_HT_COMBINE_RED_WARPS % requested_pipelines != 0 ||
         (g2s_from_env && requested_g2s_stages % requested_pipelines != 0) ||
         (s2g_from_env && requested_s2g_stages % requested_pipelines != 0))) {
        std::fprintf(
            stderr,
            "[nccl_ep] invalid combine config: pipelines=%d must divide reduction warps=%d and any "
            "pinned stage count; g2s_stages=%d, s2g_stages=%d.\n",
            requested_pipelines, NCCL_EP_HT_COMBINE_RED_WARPS,
            requested_g2s_stages, requested_s2g_stages);
        return ncclInvalidArgument;
    }

    ::ht_ep::model_config_t model;
    model.hidden_dim = kp.hidden_dim;
    model.max_num_of_tokens_per_rank = max_dispatch_tokens_per_rank;
    model.num_of_experts_per_rank = kp.experts_per_rank;
    model.ranks_per_lsa_team = kp.ranks_per_lsa_team;
    model.num_lsa_teams = num_lsa_teams;

    // Bilinear SMEM coefficients: size = fixed + G2S*per_g2s + S2G*per_s2g.
    // Layout size depends only on element width, so FP16 and BF16 (both 2 B)
    // share the BF16 instantiation; only FP32 (4 B) is distinct.
    const ::ht_ep::comb_smem_cost_t cost = (params.token_dtype == ncclFloat32) ?
        ::ht_ep::calc_comb_smem_cost<ncclFloat32>(
            max_dispatch_tokens_per_rank, num_lsa_teams, c_config, model) :
        ::ht_ep::calc_comb_smem_cost<ncclBfloat16>(
            max_dispatch_tokens_per_rank, num_lsa_teams, c_config, model);

    const int max_smem = get_device_max_dynamic_smem();
    const combine_smem_fit_t fit = choose_combine_smem_config(
        cost, static_cast<size_t>(max_smem),
        requested_g2s_stages, g2s_from_env,
        requested_s2g_stages, s2g_from_env,
        requested_pipelines, pipelines_from_env,
        kMinStagesPerPipeline, NCCL_EP_HT_COMBINE_RED_WARPS);

    const size_t requested_smem =
        ::ht_ep::calc_comb_smem(cost, requested_g2s_stages, requested_s2g_stages);
    if (!fit.feasible) {
        std::fprintf(
            stderr,
            "[nccl_ep] combine shared memory cannot be fit to the device limit: requested g2s_stages=%d, "
            "s2g_stages=%d, pipelines=%d need %zu bytes, limit=%d bytes%s.\n",
            requested_g2s_stages, requested_s2g_stages, requested_pipelines, requested_smem, max_smem,
            (g2s_from_env || s2g_from_env || pipelines_from_env) ? " (env-pinned values cannot be reduced)" : "");
        return ncclInvalidArgument;
    }

    c_config.num_pipelines = fit.num_pipelines;
    c_config.num_of_stages_g2s = fit.num_of_stages_g2s;
    c_config.num_of_stages_s2g = fit.num_of_stages_s2g;
    const size_t smem_size = fit.smem_size;

    // Report the resolved config (once per distinct outcome) when verbose is
    // requested -- always, whether or not auto-fit changed anything.
    if (nccl_ep_env_verbose(*env)) {
        std::ostringstream key;
        key << "ht_combine_smem_fit:" << requested_g2s_stages << ':' << requested_s2g_stages << ':'
            << requested_pipelines << ':' << c_config.num_of_stages_g2s << ':' << c_config.num_of_stages_s2g << ':'
            << c_config.num_pipelines << ':' << smem_size;
        if (::nccl_ep::jit::announce_once(key.str())) {
            std::fprintf(
                stderr,
                "[nccl_ep][env] HT combine SMEM fit:\n"
                "[nccl_ep][env]   g2s_stages=%d (req %d), s2g_stages=%d (req %d), pipelines=%d (req %d)\n"
                "[nccl_ep][env]   smem=%zu bytes (req %zu, limit %d)\n"
                "[nccl_ep][env]   cost: fixed=%zu, per_g2s_stage=%zu, per_s2g_stage=%zu bytes\n",
                c_config.num_of_stages_g2s, requested_g2s_stages,
                c_config.num_of_stages_s2g, requested_s2g_stages,
                c_config.num_pipelines, requested_pipelines,
                smem_size, requested_smem, max_smem,
                cost.fixed, cost.per_g2s_stage, cost.per_s2g_stage);
        }
    }

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    const jit::combine_warp_layout_t combine_layout =
        jit::compute_combine_warp_layout(num_lsa_teams, c_config.num_pipelines);
    const int combine_wt_total = num_blocks * (combine_layout.block_dim / 32);
    ::ht_ep::combine_warp_timing_entry_t* d_wt = nullptr;
    ::ht_ep::combine_block_timing_entry_t* d_bt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wt, combine_wt_total * sizeof(::ht_ep::combine_warp_timing_entry_t)));
    CUDA_CHECK(cudaMalloc(&d_bt, num_blocks * sizeof(::ht_ep::combine_block_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, combine_wt_total * sizeof(::ht_ep::combine_warp_timing_entry_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_bt, 0, num_blocks * sizeof(::ht_ep::combine_block_timing_entry_t), stream));
    kp.warp_timing = d_wt;
    kp.block_timing = d_bt;
#endif

    std::vector<uint8_t> kernel_arg = build_combine_arg_buffer(kp, params);
    jit::launch_combine(
        c_config,
        max_dispatch_tokens_per_rank,
        num_lsa_teams,
        params.lsa_team_size,
        params.layout,
        kp.hidden_dim,
        env,
        kernel_arg.data(),
        kernel_arg.size(),
        static_cast<int>(smem_size),
        stream,
        params.token_dtype);

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    jit::combine_dump_warp_timing(combine_layout, num_blocks, d_wt, d_bt, stream);
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_bt));
#endif
    return ncclSuccess;
}

void call_local_dup(
    void* expert_output_token,
    float* expert_output_prob,
    const int32_t* emuf_group_buf,
    const int32_t* emuf_group_count,
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
    int scale_row_bytes) {
    constexpr int kPipeDepth = NCCL_EP_HT_LOCAL_DUP_PIPE_DEPTH;
    // local_dup is a byte-relocation fan-out: the wire type only sets the per-token
    // width (fp8 -> uint8_t, FP16/BF16 -> uint16_t, FP32 -> uint32_t). QUANT_FWD
    // fans a per-token scale row (scale_row_bytes) out alongside the token.
    auto run = [&](auto tag) {
        using TOKEN_DATA_TYPE = decltype(tag);
        const int smem_bytes = ::ht_ep::local_dup_dynamic_smem_bytes(
            hidden_dim,
            kPipeDepth,
            forward_dispatch,
            experts_per_rank,
            ranks_per_lsa_team,
            sizeof(TOKEN_DATA_TYPE),
            scale_row_bytes);

        ::ht_ep::local_dup_kernel_param_t<TOKEN_DATA_TYPE> pp{};
        pp.expert_output_token = reinterpret_cast<TOKEN_DATA_TYPE*>(expert_output_token);
        pp.expert_output_prob = expert_output_prob;
        pp.emuf_group_buf = emuf_group_buf;
        pp.emuf_group_count = emuf_group_count;
        pp.emuf_group_stride = emuf_group_stride;
        pp.lsa_S2G_flag = lsa_S2G_flag;
        pp.expected_lsa_flag_val = expected_lsa_flag_val;
        pp.grid_barrier_counter = grid_barrier_counter;
        pp.experts_per_rank = experts_per_rank;
        pp.ranks_per_lsa_team = ranks_per_lsa_team;
        pp.expert_output_scale = expert_output_scale;
        pp.scale_row_bytes = scale_row_bytes;
        jit::launch_local_dup<TOKEN_DATA_TYPE>(
            hidden_dim,
            kPipeDepth,
            forward_dispatch,
            recipe,
            num_blocks,
            pp,
            smem_bytes,
            stream);
    };
    if (token_dtype == ncclFloat32) run(uint32_t{});
    else if (token_dtype == ncclFloat8e4m3 || token_dtype == ncclFloat8e5m2 ||
             token_dtype == ncclFloat4x2) run(uint8_t{});
    else run(uint16_t{});
}

void call_local_reduce(
    void* expert_input_token,
    float* expert_input_prob,
    const int32_t* emuf_group_buf,
    const int32_t* emuf_group_count,
    int emuf_group_stride,
    int hidden_dim,
    int experts_per_rank,
    int ranks_per_lsa_team,
    bool backward_combine,
    int num_blocks,
    cudaStream_t stream,
    ncclDataType_t token_dtype) {
    // The reduce decodes/accumulates/re-encodes per token_dtype; the param/sizeof
    // type collapses FP16->uint16_t (layout-identical), FP32 -> uint32_t.
    auto run = [&](auto tag) {
        using T = decltype(tag);
        ::ht_ep::local_reduce_kernel_param_t<T> lp{};
        lp.expert_input_token = reinterpret_cast<T*>(expert_input_token);
        lp.expert_input_prob = expert_input_prob;
        lp.emuf_group_buf = emuf_group_buf;
        lp.emuf_group_count = emuf_group_count;
        lp.emuf_group_stride = emuf_group_stride;
        lp.experts_per_rank = experts_per_rank;
        lp.ranks_per_lsa_team = ranks_per_lsa_team;
        jit::launch_local_reduce<T>(hidden_dim, backward_combine, experts_per_rank, num_blocks, lp, stream, token_dtype);
    };
    if (token_dtype == ncclFloat32) run(uint32_t{});
    else run(uint16_t{});
}

ncclResult_t call_combine(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_lsa_teams,
    bool backward_combine,
    int num_blocks,
    const ncclEpEnvConfig* env,
    cudaStream_t stream) {
    if (backward_combine) {
        return combine_impl<true>(
            params,
            max_dispatch_tokens_per_rank,
            num_tokens_per_chunk,
            num_lsa_teams,
            num_blocks,
            env,
            stream);
    }
    return combine_impl<false>(
        params,
        max_dispatch_tokens_per_rank,
        num_tokens_per_chunk,
        num_lsa_teams,
        num_blocks,
        env,
        stream);
}

// Grid sizing for local-permute kernels: one block per SM. Latency is hidden
// by in-flight loads in dup/reduce, so block-level oversubscription is moot.
static inline unsigned int local_permute_grid(int sm_count, unsigned int shuffle_sms) {
    unsigned int grid = (shuffle_sms != 0) ? shuffle_sms : static_cast<unsigned int>(sm_count);
    if (grid == 0) grid = 1;
    return grid;
}

void launch_dispatch_permute(
    void* recv_x_em,
    float* recv_topk_weights_em,
    const void* flat_staging,
    const float* recv_topk_weights_flat,
    const int32_t* flat2em_slot_map,
    const int32_t* num_recv_tokens_dev,
    const int64_t* expert_token_offsets,
    const int32_t* per_expert_counts_active,
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
    int scale_row_bytes) {
    assert(experts_per_rank > 0 && experts_per_rank <= ::ht_ep::kLocalPermuteMaxExpertsPerRank);
    assert(row_bytes > 0 && (row_bytes % 16) == 0);
    assert(top_k > 0);
    assert(sm_count > 0);
    assert((recv_topk_weights_em == nullptr) == (recv_topk_weights_flat == nullptr));
    // QUANT_FWD: scale rows relocate alongside tokens; 16B-aligned like tokens.
    assert(scale_row_bytes >= 0 && (scale_row_bytes % 16) == 0);
    assert((scale_row_bytes > 0) == (recv_scales_em != nullptr && flat_scale_staging != nullptr));

    const unsigned int grid = local_permute_grid(sm_count, shuffle_sms);

    ::ht_ep::local_permute_dup_param_t p{};
    p.recv_x_em = recv_x_em;
    p.recv_topk_weights_em = recv_topk_weights_em;
    p.flat_staging = flat_staging;
    p.recv_topk_weights_flat = recv_topk_weights_flat;
    p.flat2em_slot_map = flat2em_slot_map;
    p.num_recv_tokens_dev = num_recv_tokens_dev;
    p.expert_token_offsets = expert_token_offsets;
    p.per_expert_counts_active = per_expert_counts_active;
    p.top_k = top_k;
    p.experts_per_rank = experts_per_rank;
    p.row_bytes = row_bytes;
    p.caller_num_recv_tokens = caller_num_recv_tokens;
    p.recv_scales_em = recv_scales_em;
    p.flat_scale_staging = flat_scale_staging;
    p.scale_row_bytes = scale_row_bytes;

    ::nccl_ep::ht::jit::launch_local_permute_dup(static_cast<int>(grid), p, recipe, stream);
}

void launch_combine_reduce(
    void* flat_staging,
    const void* recv_x_em,
    const int32_t* flat2em_slot_map,
    const int32_t* num_recv_tokens_dev,
    const float* em_weights_in,
    float* flat_weights_out,
    int top_k,
    int row_bytes,
    int sm_count,
    unsigned int shuffle_sms,
    cudaStream_t stream,
    ncclDataType_t token_dtype) {
    assert(row_bytes > 0 && (row_bytes % 16) == 0);
    assert(top_k > 0);
    assert(sm_count > 0);
    assert((em_weights_in == nullptr) == (flat_weights_out == nullptr));

    const unsigned int grid = local_permute_grid(sm_count, shuffle_sms);

    ::ht_ep::local_permute_reduce_param_t p{};
    p.flat_staging = flat_staging;
    p.recv_x_em = recv_x_em;
    p.flat2em_slot_map = flat2em_slot_map;
    p.num_recv_tokens_dev = num_recv_tokens_dev;
    p.em_weights_in = em_weights_in;
    p.flat_weights_out = flat_weights_out;
    p.top_k = top_k;
    p.row_bytes = row_bytes;

    ::nccl_ep::ht::jit::launch_local_permute_reduce(
        top_k,
        row_bytes,
        static_cast<int>(grid),
        p,
        stream,
        token_dtype);
}

} // namespace ht
} // namespace nccl_ep
