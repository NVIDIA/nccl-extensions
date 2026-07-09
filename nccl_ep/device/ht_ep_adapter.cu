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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    uint8_t* __restrict__ routing_bitmap,     // [max_tokens, num_experts_packed]
    TopkIdxT* __restrict__ cached_topk_idx,   // [num_tokens, num_topk]; nullable
    int num_tokens,
    int max_tokens,                           // tail-zero bound (>= num_tokens)
    int num_topk,
    int num_experts_packed                    // = ceil(num_experts / 8)
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= max_tokens) return;

    // Each thread exclusively owns its row -- no atomics needed.
    // Zero the row before OR-ing in bits; the caller does not pre-zero.
    // Threads for tail rows [num_tokens, max_tokens) zero and exit, so the
    // downstream ncclAllGather over max_tokens rows ships clean tail bytes.
    uint8_t* row = routing_bitmap + token * num_experts_packed;
    for (int b = 0; b < num_experts_packed; b++) row[b] = 0;
    if (token >= num_tokens) return;
    const TopkIdxT* in_row = topk_idx + token * num_topk;
    TopkIdxT* out_row = cached_topk_idx ? cached_topk_idx + token * num_topk : nullptr;
    for (int k = 0; k < num_topk; k++) {
        TopkIdxT expert = in_row[k];
        if (out_row) out_row[k] = expert;
        if (expert >= 0) {
            row[expert / 8] |= (1u << (expert % 8));
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
    int num_experts_packed,
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
        num_experts_packed);
}

template void
convert_topk_to_routing_map<int32_t>(const int32_t*, uint8_t*, int32_t*, int, int, int, int, cudaStream_t);
template void
convert_topk_to_routing_map<int64_t>(const int64_t*, uint8_t*, int64_t*, int, int, int, int, cudaStream_t);

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
    float* __restrict__ dense_prob,                   // [num_tokens, experts_per_node]
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
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
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
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
    int experts_per_node,
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
        experts_per_node,
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
    const float* __restrict__ dense_prob,              // [num_recv_tokens, experts_per_node]
    const bool* __restrict__ local_expert_routing_map, // [num_recv_tokens, experts_per_rank]
    float* __restrict__ recv_topk_weights,             // EM: [N]; FLAT/RM: [N, topk]
    int64_t* __restrict__ recv_topk_idx,               // [num_recv_tokens, topk]; nullptr under EM
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
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
                int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
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

      // Get weight from dense output (indexed by local expert within node)
            // dense_prob layout: [token, experts_per_node] where experts_per_node = experts_per_rank * ranks_per_node
            // Local rank's experts are at offset: local_rank * experts_per_rank
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
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
    int experts_per_node,
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
        experts_per_node,
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
    int node_rank,
    int local_rank,
    int num_tokens_per_rank,
    int num_nodes,
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
            sp.node_rank = node_rank;
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
                num_nodes,
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
            sp.node_rank = node_rank;
            sp.local_rank = local_rank;
            sp.num_of_tokens_per_rank = num_tokens_per_rank;
            sp.experts_per_rank = experts_per_rank;

            jit::launch_scan_em(NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, num_nodes, lsa_team_size, sp, stream);
        }

        const int num_mask_words = (lsa_team_size + 63) / 64;
        const int num_total_attn_tokens = num_tokens_per_rank * lsa_team_size * num_nodes;
        launch_build_em_tables(
            global_routing_map,
            token_rank_mask,
            num_mask_words,
            num_total_attn_tokens,
            num_tokens_per_rank,
            lsa_team_size,
            experts_per_rank,
            num_nodes,
            node_rank,
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
    sp.node_rank = node_rank;
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
        num_nodes,
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
    int node_rank,
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
    p.node_rank                = node_rank;
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

ncclResult_t check_dispatch_smem_limit(const ::ht_ep::dispatch_config_t& config, size_t smem_size) {
    const int max_smem = get_device_max_dynamic_smem();
    if (smem_size <= static_cast<size_t>(max_smem)) return ncclSuccess;

    std::fprintf(
        stderr,
        "[nccl_ep] dispatch dynamic shared memory exceeds device limit: requested=%zu bytes, "
        "limit=%d bytes. Tune dispatch stages/pipelines; current stages=%d, pipelines=%d.\n",
        smem_size,
        max_smem,
        config.num_of_stages,
        config.num_pipelines);
    return ncclInvalidArgument;
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
    kp.num_of_ranks_per_node = params.lsa_team_size;
    // User input buffers
    kp.attn_input_token = reinterpret_cast<const TOKEN_DATA_TYPE*>(params.attn_input_token);
    kp.attn_input_prob = params.attn_input_prob;
    kp.attn_input_token_scaling_factor = params.attn_input_scaling_factor;

    // Metadata and sync flags
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.pad_actual_counts = params.pad_actual_counts;
    kp.pad_expert_token_offsets = params.pad_expert_token_offsets;
    kp.pad_alignment = params.pad_alignment;
    kp.expected_rdma_flag_value = params.expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.intra_node_write_completion_flags;
    kp.dispatch_grid_barrier_counter = params.dispatch_grid_barrier_counter;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;
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
        .rdma_send_staging_offset = params.mr_info.rdma_send_staging_offset,
        .rdma_inter_node_group_packed_offset = params.mr_info.rdma_inter_node_group_packed_offset,
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
        sf_ptrs[i] = params.expert_output_scaling_factor_ptrs ? params.expert_output_scaling_factor_ptrs[i] : nullptr;
    }

    return arg;
}

// Host dispatch launcher. The JIT source owns all device-kernel specialization;
// the host only asks ht_ep for the matching dynamic-SMEM size.
ncclResult_t dispatch_impl(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_nodes,
    ncclEpPassDir_t pass_direction,
    int num_blocks,
    int sf_bytes_per_token,
    const ncclEpEnvConfig* env,
    cudaStream_t stream,
    const DispatchKernelSpec& kernel_spec) {
    {
        const bool forward_dispatch = (pass_direction == NCCL_EP_FWD_PASS);
        // The dispatch param/arg buffers are pointer-only (wire-width-invariant), so the
        // host packs with one fixed type; the JIT specializes the actual kernel by
        // token_dtype (dispatch_token_data_type_literal in launch_dispatch). No host-side
        // compile-time token-type switch is needed -- rely on the JIT.
        using TOKEN_DATA_TYPE = uint16_t;
        // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
        // Check alignment at runtime now that experts_per_rank is dynamic
        const int experts_per_node = params.experts_per_rank * params.lsa_team_size;
        assert(
            (experts_per_node * sizeof(float)) % 16 == 0 && "experts_per_node must be multiple of 4 for TMA alignment");
        // 16B cp.async.bulk alignment for the S2D map fetch; matters when s2d_inner_dim < 4.
        assert(
            (static_cast<int64_t>(params.num_tokens_per_rank) * params.s2d_inner_dim) % 4 == 0 &&
            "Dispatch S2D cp.async.bulk: num_tokens_per_rank * s2d_inner_dim must be a "
            "multiple of 4 (flat layout with lsa_team_size <= 3 requires even num_tokens_per_rank)");

        auto kp = build_dispatch_param_base<TOKEN_DATA_TYPE>(params);

        // Compute dynamic SMEM size at host (was done inside ht_ep::dispatch).
        ::ht_ep::dispatch_config_t d_config;
        ::ht_ep::model_config_t d_model;
        d_config.num_of_stages = NCCL_EP_HT_DISPATCH_NUM_OF_STAGES;
        d_config.num_of_in_flight_s2g = NCCL_EP_HT_DISPATCH_NUM_OF_IN_FLIGHT_S2G;
        d_config.num_of_tokens_per_chunk = num_tokens_per_chunk;
        d_config.num_of_blocks = num_blocks;
        d_config.forward_dispatch = forward_dispatch;
        d_config.sf_bytes_per_token = sf_bytes_per_token;
        d_config.num_pipelines = NCCL_EP_HT_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
        d_config.stages_per_pipeline = NCCL_EP_HT_DISPATCH_NUM_OF_STAGES / NCCL_EP_HT_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
        d_config.s2d_inner_dim = kp.s2d_inner_dim;
        d_model.hidden_dim = kp.hidden_dim;
        d_model.max_num_of_tokens_per_rank = max_dispatch_tokens_per_rank;
        d_model.num_of_experts_per_rank = kp.experts_per_rank;
        d_model.num_of_ranks_per_node = kp.num_of_ranks_per_node;
        d_model.num_of_nodes = num_nodes;

        const int smem_size = static_cast<int>(::ht_ep::calculate_dispatch_smem_layout_size(
            params.layout, kernel_spec.payload_bytes, d_config, d_model));
        if (smem_size == 0) {
            std::fprintf(stderr, "NCCL EP warning: unsupported HT dispatch token size %u\n",
                         kernel_spec.payload_bytes);
            return ncclInvalidArgument;
        }
        if (ncclResult_t r = check_dispatch_smem_limit(d_config, smem_size); r != ncclSuccess) {
            std::fprintf(stderr, "NCCL EP warning: dispatch shared-memory requirement %d is unsupported\n",
                         smem_size);
            return r;
        }

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
        const jit::dispatch_warp_layout_t dispatch_layout = jit::compute_dispatch_warp_layout(num_nodes, params.layout);
        const int dispatch_wt_total = num_blocks * (dispatch_layout.block_dim / 32);
        ::ht_ep::dispatch_warp_timing_entry_t* d_wt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_wt, dispatch_wt_total * sizeof(::ht_ep::dispatch_warp_timing_entry_t)));
        CUDA_CHECK(
            cudaMemsetAsync(d_wt, 0, dispatch_wt_total * sizeof(::ht_ep::dispatch_warp_timing_entry_t), stream));
        kp.warp_timing = d_wt;
#endif

        std::vector<uint8_t> kernel_arg = build_dispatch_arg_buffer(kp, params);
        if (ncclResult_t r = jit::launch_dispatch(
            NCCL_EP_HT_DISPATCH_NUM_OF_STAGES,
            NCCL_EP_HT_DISPATCH_NUM_OF_IN_FLIGHT_S2G,
            num_tokens_per_chunk,
            max_dispatch_tokens_per_rank,
            num_blocks,
            forward_dispatch,
            num_nodes,
            params.lsa_team_size,
            params.layout,
            kp.hidden_dim,
            sf_bytes_per_token,
            env,
            kernel_arg.data(),
            kernel_arg.size(),
            smem_size,
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
    int num_nodes,
    ncclEpDispatchQuantizationRecipe_t quantization_recipe,
    ncclEpPassDir_t pass_direction,
    int num_blocks,
    int sf_bytes_per_token,
    const ncclEpEnvConfig* env,
    cudaStream_t stream,
    ncclDataType_t token_dtype) {
    DispatchKernelSpec kernel_spec;
    if (ncclResult_t r = resolveDispatchKernelSpec(quantization_recipe, token_dtype, &kernel_spec);
        r != ncclSuccess) {
        return r;
    }
    return dispatch_impl(
        params, max_dispatch_tokens_per_rank, num_tokens_per_chunk, num_nodes,
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
    kp.num_of_ranks_per_node = params.lsa_team_size;
    // User output buffers
    kp.attn_output_token = reinterpret_cast<uint16_t*>(params.attn_output_token);
    kp.attn_output_prob = params.attn_output_prob;

    // RDMA buffers (multi-node only)
    kp.rdma_intra_node_red_token = params.rdma_intra_node_red_token;
    kp.rdma_intra_node_red_prob = params.rdma_intra_node_red_prob;
    kp.rdma_inter_node_group_token = params.combine_rdma_inter_node_group_token;
    kp.rdma_inter_node_group_prob = params.combine_rdma_inter_node_group_prob;

    // Metadata
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;

    // Sync flags
    kp.expected_rdma_flag_value = params.combine_expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.combine_expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.combine_rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.combine_intra_node_write_completion_flags;
    kp.combine_grid_barrier_counter = params.combine_grid_barrier_counter;
    kp.guard_enabled = params.guard_enabled;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;
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
        .rdma_intra_node_red_token_offset = params.mr_info.rdma_intra_node_red_token_offset,
        .combine_rdma_inter_node_group_token_offset = params.mr_info.combine_rdma_inter_node_group_token_offset,
        .rdma_intra_node_red_prob_offset = params.mr_info.rdma_intra_node_red_prob_offset,
        .combine_rdma_inter_node_group_prob_offset = params.mr_info.combine_rdma_inter_node_group_prob_offset,
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
void combine_impl(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_nodes,
    int num_blocks,
    const ncclEpEnvConfig* env,
    cudaStream_t stream) {
    // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
    const int experts_per_node = params.experts_per_rank * params.lsa_team_size;
    assert((experts_per_node * sizeof(float)) % 16 == 0 && "experts_per_node must be multiple of 4 for TMA alignment");

    auto kp = build_combine_param_base(params);

    // Select config based on num_nodes (single-node: 12 stages/2 pipelines, multi-node: 5 stages/1 pipeline)
    const int num_stages_g2s =
        (num_nodes == 1) ? NCCL_EP_HT_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S : NCCL_EP_HT_COMBINE_MULTINODE_NUM_OF_STAGES_G2S;
    const int num_stages_s2g =
        (num_nodes == 1) ? NCCL_EP_HT_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G : NCCL_EP_HT_COMBINE_MULTINODE_NUM_OF_STAGES_S2G;

    ::ht_ep::model_config_t model;
    model.hidden_dim = kp.hidden_dim;
    model.max_num_of_tokens_per_rank = max_dispatch_tokens_per_rank;
    model.num_of_experts_per_rank = kp.experts_per_rank;
    model.num_of_ranks_per_node = kp.num_of_ranks_per_node;
    model.num_of_nodes = num_nodes;
    // Pick the layout-size instantiation by wire dtype; the width is derived inside the
    // template. Layout size depends only on element width, so FP16 and BF16 (both 2 B)
    // share the BF16 instantiation; only FP32 (4 B) is distinct.
    const int smem_size = (params.token_dtype == ncclFloat32) ?
                              static_cast<int>(::ht_ep::calculate_combine_smem_layout_size<ncclFloat32>(
                                  num_stages_g2s,
                                  num_stages_s2g,
                                  num_tokens_per_chunk,
                                  max_dispatch_tokens_per_rank,
                                  num_nodes,
                                  BACKWARD_COMBINE,
                                  model)) :
                              static_cast<int>(::ht_ep::calculate_combine_smem_layout_size<ncclBfloat16>(
                                  num_stages_g2s,
                                  num_stages_s2g,
                                  num_tokens_per_chunk,
                                  max_dispatch_tokens_per_rank,
                                  num_nodes,
                                  BACKWARD_COMBINE,
                                  model));

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    const jit::combine_warp_layout_t combine_layout = jit::compute_combine_warp_layout(num_nodes);
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
        num_stages_g2s,
        num_stages_s2g,
        num_tokens_per_chunk,
        max_dispatch_tokens_per_rank,
        NCCL_EP_HT_COMBINE_NUM_OF_TOKENS_PER_GROUP,
        num_blocks,
        NCCL_EP_HT_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
        BACKWARD_COMBINE,
        num_nodes,
        params.lsa_team_size,
        params.layout,
        kp.hidden_dim,
        env,
        kernel_arg.data(),
        kernel_arg.size(),
        smem_size,
        stream,
        params.token_dtype);

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
    jit::combine_dump_warp_timing(combine_layout, num_blocks, d_wt, d_bt, stream);
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_bt));
#endif
}

void call_local_dup(
    void* expert_output_token,
    float* expert_output_prob,
    const int32_t* emuf_group_buf,
    const int32_t* emuf_group_count,
    int emuf_group_stride,
    const uint32_t* intra_node_write_completion_flag,
    uint32_t* expected_intra_node_flag_value,
    uint32_t* grid_barrier_counter,
    int hidden_dim,
    int experts_per_rank,
    int num_of_ranks_per_node,
    bool forward_dispatch,
    int num_blocks,
    cudaStream_t stream,
    ncclDataType_t token_dtype) {
    constexpr int kPipeDepth = NCCL_EP_HT_LOCAL_DUP_PIPE_DEPTH;
    // local_dup is a byte-relocation fan-out: the wire type only sets the
    // per-token width (FP16/BF16 -> uint16_t, FP32 -> uint32_t). SCALES_FORWARD
    // is rejected upstream on this path.
    auto run = [&](auto tag) {
        using TOKEN_DATA_TYPE = decltype(tag);
        const int smem_bytes = ::ht_ep::local_dup_dynamic_smem_bytes(
            hidden_dim,
            kPipeDepth,
            forward_dispatch,
            experts_per_rank,
            num_of_ranks_per_node,
            sizeof(TOKEN_DATA_TYPE));

        ::ht_ep::local_dup_kernel_param_t<TOKEN_DATA_TYPE> pp{};
        pp.expert_output_token = reinterpret_cast<TOKEN_DATA_TYPE*>(expert_output_token);
        pp.expert_output_prob = expert_output_prob;
        pp.emuf_group_buf = emuf_group_buf;
        pp.emuf_group_count = emuf_group_count;
        pp.emuf_group_stride = emuf_group_stride;
        pp.intra_node_write_completion_flag = intra_node_write_completion_flag;
        pp.expected_intra_node_flag_value = expected_intra_node_flag_value;
        pp.grid_barrier_counter = grid_barrier_counter;
        pp.experts_per_rank = experts_per_rank;
        pp.num_of_ranks_per_node = num_of_ranks_per_node;
        jit::launch_local_dup<TOKEN_DATA_TYPE>(
            hidden_dim,
            kPipeDepth,
            forward_dispatch,
            num_blocks,
            pp,
            smem_bytes,
            stream);
    };
    if (token_dtype == ncclFloat32) run(uint32_t{});
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
    int num_of_ranks_per_node,
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
        lp.num_of_ranks_per_node = num_of_ranks_per_node;
        jit::launch_local_reduce<T>(hidden_dim, backward_combine, experts_per_rank, num_blocks, lp, stream, token_dtype);
    };
    if (token_dtype == ncclFloat32) run(uint32_t{});
    else run(uint16_t{});
}

void call_combine(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_tokens_per_chunk,
    int num_nodes,
    bool backward_combine,
    int num_blocks,
    const ncclEpEnvConfig* env,
    cudaStream_t stream) {
    if (backward_combine) {
        combine_impl<true>(
            params,
            max_dispatch_tokens_per_rank,
            num_tokens_per_chunk,
            num_nodes,
            num_blocks,
            env,
            stream);
    } else {
        combine_impl<false>(
            params,
            max_dispatch_tokens_per_rank,
            num_tokens_per_chunk,
            num_nodes,
            num_blocks,
            env,
            stream);
    }
}

// Grid sizing for local-permute kernels: one block per SM. Latency is hidden
// by in-flight loads in dup/reduce, so block-level oversubscription is moot.
static inline unsigned int local_permute_grid(int sm_count, unsigned int prolog_epilog_sms) {
    unsigned int grid = (prolog_epilog_sms != 0) ? prolog_epilog_sms : static_cast<unsigned int>(sm_count);
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
    unsigned int prolog_epilog_sms,
    int caller_num_recv_tokens,
    cudaStream_t stream) {
    assert(experts_per_rank > 0 && experts_per_rank <= ::ht_ep::kLocalPermuteMaxExpertsPerRank);
    assert(row_bytes > 0 && (row_bytes % 16) == 0);
    assert(top_k > 0);
    assert(sm_count > 0);
    assert((recv_topk_weights_em == nullptr) == (recv_topk_weights_flat == nullptr));

    const unsigned int grid = local_permute_grid(sm_count, prolog_epilog_sms);

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

    ::nccl_ep::ht::jit::launch_local_permute_dup(static_cast<int>(grid), p, stream);
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
    unsigned int prolog_epilog_sms,
    cudaStream_t stream,
    ncclDataType_t token_dtype) {
    assert(row_bytes > 0 && (row_bytes % 16) == 0);
    assert(top_k > 0);
    assert(sm_count > 0);
    assert((em_weights_in == nullptr) == (flat_weights_out == nullptr));

    const unsigned int grid = local_permute_grid(sm_count, prolog_epilog_sms);

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
