/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once
#include "common.hpp"

namespace ht_ep {

// Count-exchange path: per-destination-rank send-count histogram. Topology
// constants (experts_per_rank, lsa_team_size) are baked as template params so the
// per-token divisions fold to compile-time shifts/masks. Single-LSA-team only.
struct build_count_metadata_param_t {
    const void* topk_idx;                               // [num_tokens, num_topk], native int32/int64 (see TopkIdxT)
    int32_t* cached_topk_idx;                           // [num_tokens, num_topk] cache copy; expert ids fit int32
    int32_t* cnt_rank;                                  // [num_world_dst_ranks], pre-zeroed
    int32_t* cnt_expert;                                // [num_experts], pre-zeroed
    int32_t* own_chunk_rank;                            // own-LSA-team slice [num_chunks, lsa_team_size]
    uint64_t* token_dst_rank_bitmap;                    // [num_tokens] per-token dest-rank bitmap
    int32_t* per_src_lteam_num_tokens;                  // [num_lsa_teams] token counts; only own slot written (single-LSA: [1])
    bool* rdma_to_attn_map;                             // this-LSA-team slice; rdma_to_attn_map[r2a_offset + token]
    int num_tokens;
    int num_topk;
    int tokens_per_chunk;                               // dispatch source-chunk width
    int my_lteam;
    int r2a_offset;                                     // my_lteam * rdma_per_lsa_sz
};

template <typename TopkIdxT, int MAX_DST_RANKS, int EXPERTS_PER_RANK, int LSA_TEAM_SZ>
__device__ void build_count_metadata_impl(const build_count_metadata_param_t& p) {
    // Count mode is single-LSA-team only (enforced at handle init); the histogram writes
    // one LSA-team's worth of ranks.
    constexpr int NUM_LSA_TEAMS = 1;
    const int NUM_DST_RANKS = NUM_LSA_TEAMS * LSA_TEAM_SZ;

    const TopkIdxT* __restrict__ topk_idx = static_cast<const TopkIdxT*>(p.topk_idx);
    int32_t* __restrict__ cached_topk_idx = p.cached_topk_idx;
    const int num_tokens = p.num_tokens;
    const int num_topk = p.num_topk;
    int32_t* __restrict__ cnt_rank = p.cnt_rank;
    int32_t* __restrict__ cnt_expert = p.cnt_expert;
    const int tokens_per_chunk = p.tokens_per_chunk;
    int32_t* __restrict__ own_chunk_rank = p.own_chunk_rank;
    uint64_t* __restrict__ token_dst_rank_bitmap = p.token_dst_rank_bitmap;
    int32_t* __restrict__ per_src_lteam_num_tokens = p.per_src_lteam_num_tokens;
    const int my_lteam = p.my_lteam;
    bool* __restrict__ rdma_to_attn_map = p.rdma_to_attn_map;
    const int r2a_offset = p.r2a_offset;

    // Buffer layouts (count mode is single-LSA-team, so lsa_team_size == nRanks):
    //   seen[]                        per-token dest-rank bitmap, ceil(nRanks/64) words
    //   own_chunk_rank                [num_chunks, lsa_team_size]
    //   rdma_to_attn_map              [max_tokens] any-local-rank gate (rdma_rank slice + token)
    const int stride = gridDim.x * blockDim.x;
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token == 0) per_src_lteam_num_tokens[my_lteam] = num_tokens;
    // Grid-strided: the grid is capped at the scan preprocessing SM count, so each block walks
    // multiple tokens rather than launching one block per token.
    for (; token < num_tokens; token += stride) {
        const TopkIdxT* row = topk_idx + (size_t)token * num_topk;
        const int chunk = token / tokens_per_chunk;
        unsigned long long seen[nccl_ep::bit_words(MAX_DST_RANKS)] = {0};       // dedup dest ranks, up to 1024
        for (int k = 0; k < num_topk; k++) {
            TopkIdxT e = row[k];
            // Cache the topk row for the sender-side dense-prob rebuild in dispatch and the BWD combine scatter.
            cached_topk_idx[(size_t)token * num_topk + k] = static_cast<int32_t>(e);
            // Masked/unassigned slots use a negative sentinel; cache it but skip all counting
            // (matches the scan reference and the receiver LERM build).
            if (e < 0) continue;
            int dst_rank = (int)(e / EXPERTS_PER_RANK);
            // Per-expert count: a token routes to a given global expert at most once.
            atomicAdd(&cnt_expert[(int)e], 1);
            if (nccl_ep::test_bit(seen, dst_rank)) continue;
            nccl_ep::set_bit(seen, dst_rank);
            atomicAdd(&cnt_rank[dst_rank], 1);
            // single LSA team: LSA-local dest rank == global dest rank
            const int rl = dst_rank;
            atomicAdd(&own_chunk_rank[(size_t)chunk * LSA_TEAM_SZ + rl], 1);
        }
        // Own-LSA-team routing from the world bitmap -> rdma_to_attn_map (the G2S gate). An LSA team
        // owns the lsa_team_size seen[] bits starting at team*lsa_team_size; that slice fits
        // one word for the <=64 case and may straddle/span words (arbitrary alignment) for wider
        // NVLink domains.
        const int my_slice_lo = my_lteam * LSA_TEAM_SZ;
        rdma_to_attn_map[r2a_offset + token] = nccl_ep::bit_range_any(seen, my_slice_lo, my_slice_lo + LSA_TEAM_SZ);
        // token_dst_rank_bitmap is ceil(nRanks/64) words per token; the receiver reads only the word
        // covering its LSA team's ranks.
        const int hit_words = nccl_ep::bit_words(NUM_DST_RANKS);
        for (int w = 0; w < hit_words; w++)
            token_dst_rank_bitmap[(size_t)token * hit_words + w] = seen[w];
    }
}

} // namespace ht_ep
