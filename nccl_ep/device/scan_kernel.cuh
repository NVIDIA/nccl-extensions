/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once
#include "common.hpp"
#include "device_primitives.cuh"
#include <cooperative_groups.h>

namespace hybrid_ep {

enum scan_state {
    EMPTY = 0,
    PRIV_SUM = 1
};

struct tmp_state_t {
    scan_state state;
    int32_t value;
};

static constexpr int WARP_SIZE = 32;

template <int NUM_MASK_WORDS>
struct rank_mask_t {
    uint64_t words[NUM_MASK_WORDS];

    __device__ __forceinline__ bool test(int r) const {
        return (words[r >> 6] >> (r & 63)) & 1ull;
    }
    __device__ __forceinline__ void set(int r) {
        words[r >> 6] |= 1ull << (r & 63);
    }
    __device__ __forceinline__ bool any() const {
        uint64_t v = 0;
#pragma unroll
        for (int i = 0; i < NUM_MASK_WORDS; i++) v |= words[i];
        return v != 0;
    }
    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < NUM_MASK_WORDS; i++) words[i] = 0;
    }
};

// Shared-memory layout for the scan. Owns BOTH the FLAT rank-scan region and the
// (mutually exclusive) optional add-ons: per-expert counts (FLAT) OR the EM-permute
// region. All offsets are derived sequentially in from_raw() so the sub-regions
// provably do not overlap, and byte_size() is the single source of truth shared by
// the device carve and the host launch-config sizing.
struct scan_flat_smem_t {
    // FLAT rank-scan region.
    int32_t* warp_rank_sums;      // [num_warps * num_ranks]
    int32_t* block_prefix;        // [num_ranks]
    int32_t* warp_prefix;         // [num_warps * num_ranks]
    int32_t* expert_counts;       // [experts_per_rank] iff has_expert_counts, else null
    // EM-permute region (all null unless has_em_permute).
    int32_t* expert_warp_sums;    // [num_warps * experts_per_rank]
    int32_t* expert_block_prefix; // [experts_per_rank]
    int32_t* expert_total;        // [experts_per_rank]
    int32_t* expert_base;         // [experts_per_rank]
    int32_t* warp_expert_prefix;  // [num_warps * experts_per_rank]

    // Total int32 slots the layout needs -- must mirror from_raw()'s advances.
    static __host__ __device__ size_t num_ints(
        int num_warps, int num_ranks, int experts_per_rank, bool has_expert_counts, bool has_em_permute) {
        size_t n = static_cast<size_t>(2 * num_warps + 1) * num_ranks;  // rank region
        if (has_expert_counts) n += static_cast<size_t>(experts_per_rank);
        if (has_em_permute) n += static_cast<size_t>(2 * num_warps + 3) * experts_per_rank;
        return n;
    }
    static __host__ __device__ size_t byte_size(
        int num_warps, int num_ranks, int experts_per_rank, bool has_expert_counts, bool has_em_permute) {
        return num_ints(num_warps, num_ranks, experts_per_rank, has_expert_counts, has_em_permute) * sizeof(int32_t);
    }

    template <bool ENABLE_PER_EXPERT_COUNTS, bool ENABLE_EM_PERMUTE>
    __device__ static scan_flat_smem_t from_raw(
        uint8_t* smem, int num_warps, int num_ranks, int experts_per_rank) {
        scan_flat_smem_t s{};
        int32_t* p = reinterpret_cast<int32_t*>(smem);
        // FLAT rank region.
        s.warp_rank_sums = p; p += num_warps * num_ranks;
        s.block_prefix   = p; p += num_ranks;
        s.warp_prefix    = p; p += num_warps * num_ranks;
        // Mutually exclusive add-ons carved immediately after the rank region.
        if constexpr (ENABLE_PER_EXPERT_COUNTS) { s.expert_counts = p; p += experts_per_rank; }
        if constexpr (ENABLE_EM_PERMUTE) {
            s.expert_warp_sums    = p; p += num_warps * experts_per_rank;
            s.expert_block_prefix = p; p += experts_per_rank;
            s.expert_total        = p; p += experts_per_rank;
            s.expert_base         = p; p += experts_per_rank;
            s.warp_expert_prefix  = p; p += num_warps * experts_per_rank;
        }
        return s;
    }
};

// Shared geometry used by both FLAT and EM scan entries (computed by
// compute_scan_geometry). Passed by reference into the scan helpers.
struct scan_geometry_t {
    int num_of_total_attn_tokens;
    int num_of_tokens_per_thread;
    int num_of_tokens_per_warp;
    int num_of_tokens_per_block;
    int rdma_to_attn_map_size_per_node;
    int experts_per_node_packed;
    int packed_row_bytes;
    int thread_starting_token;
    int warp_id;
    int lane_id;
};

static __device__ __forceinline__ bool
bitmap_range_has_set_bit(const uint8_t* bitmap_row, int bit_begin, int bit_count) {
    if (bit_count <= 0) return false;

    const int bit_end = bit_begin + bit_count;
    const int first_byte = bit_begin >> 3;
    const int last_byte = (bit_end - 1) >> 3;
    const int first_bit = bit_begin & 7;
    const int last_bit = (bit_end - 1) & 7;

    if (first_byte == last_byte) {
        const uint8_t left_mask = static_cast<uint8_t>(0xFFu << first_bit);
        const uint8_t right_mask = static_cast<uint8_t>((last_bit == 7) ? 0xFFu : ((1u << (last_bit + 1)) - 1u));
        return (bitmap_row[first_byte] & static_cast<uint8_t>(left_mask & right_mask)) != 0;
    }

    const uint8_t first_mask = static_cast<uint8_t>(0xFFu << first_bit);
    if ((bitmap_row[first_byte] & first_mask) != 0) return true;

    uint8_t middle_or = 0;
    for (int b = first_byte + 1; b < last_byte; b++) middle_or |= bitmap_row[b];
    if (middle_or != 0) return true;

    const uint8_t last_mask = static_cast<uint8_t>((last_bit == 7) ? 0xFFu : ((1u << (last_bit + 1)) - 1u));
    return (bitmap_row[last_byte] & last_mask) != 0;
}

template <int NUM_MASK_WORDS, int LSA_TEAM_SIZE>
static __device__ __forceinline__ rank_mask_t<NUM_MASK_WORDS>
bitmap_row_to_rank_mask(const uint8_t* bitmap_row, int experts_per_rank) {
    rank_mask_t<NUM_MASK_WORDS> rank_mask;
    rank_mask.clear();
#pragma unroll
    for (int rank = 0; rank < LSA_TEAM_SIZE; rank++) {
        const int bit_begin = rank * experts_per_rank;
        if (bitmap_range_has_set_bit(bitmap_row, bit_begin, experts_per_rank)) {
            rank_mask.set(rank);
        }
    }
    return rank_mask;
}

// Scans the range of assigned tokens (defined by the scan geometry)
// * Typically each thread gets at most one token (512 threads/block x 16 blocks is 8K threads)
// Logic for each token:
// * Calculate `token_rank_mask` indicating which rank from local LSA team receives this token
// * Mark all tokens that are sent from ranks having the same local rank as the caller
//   * This is required to understand what to expect from RDMA rail-partners
// * Accumulate `rank_sums` for each local rank in LSA team (rank-scan prefix sum)
// * When `smem` is provided (FLAT rank-scan path), perform warp-reduce of the
//   per-thread rank_sums
template <int LSA_TEAM_SIZE, int NUM_MASK_WORDS>
__device__ __forceinline__ void extract_lsa_ranks_meta(
    const uint8_t* input_routing_map,
    rank_mask_t<NUM_MASK_WORDS>* token_rank_mask,
    bool* rdma_to_attn_map,
    const scan_geometry_t& g,
    int tokens_per_rank,
    int experts_per_rank,
    int lsa_team_id,
    int local_rank,
    scan_flat_smem_t* smem = nullptr) {
    int32_t rank_sums[LSA_TEAM_SIZE];
#pragma unroll
    for (int i = 0; i < LSA_TEAM_SIZE; i++) rank_sums[i] = 0;

    for (int i = 0; i < g.num_of_tokens_per_thread; i++) {
        int token_id = g.thread_starting_token + i * WARP_SIZE;
        if (token_id >= g.num_of_total_attn_tokens) break;

        const int TOKENS_PER_LSA_TEAM = tokens_per_rank * LSA_TEAM_SIZE;
        int src_lsa_team = token_id / TOKENS_PER_LSA_TEAM;
        int src_local_rank = (token_id % TOKENS_PER_LSA_TEAM) / tokens_per_rank;
        int src_token_id = token_id % tokens_per_rank;
        int rdma_map_id = src_lsa_team * g.rdma_to_attn_map_size_per_node + src_token_id;

        const uint8_t* bitmap_row =
            input_routing_map + token_id * g.packed_row_bytes + lsa_team_id * g.experts_per_node_packed;
        rank_mask_t<NUM_MASK_WORDS> rank_mask =
            bitmap_row_to_rank_mask<NUM_MASK_WORDS, LSA_TEAM_SIZE>(bitmap_row, experts_per_rank);

        // Accumulate unconditionally; discarded by callers that pass no smem.
#pragma unroll
        for (int j = 0; j < LSA_TEAM_SIZE; j++) {
            rank_sums[j] += rank_mask.test(j);
        }
        token_rank_mask[token_id] = rank_mask;

        if (src_local_rank == local_rank) {
            rdma_to_attn_map[rdma_map_id] = rank_mask.any();
        }
    }

    // Reduce the per-thread local rank counts to warp-level sums (FLAT path only).
    if (smem != nullptr) {
#pragma unroll
        for (int rank = 0; rank < LSA_TEAM_SIZE; rank++) {
            int32_t rank_sum = __reduce_add_sync(~0u, rank_sums[rank]);
            if (g.lane_id == 0) {
                smem->warp_rank_sums[g.warp_id * LSA_TEAM_SIZE + rank] = rank_sum;
            }
        }
    }
}


// Inter-block prefix scan for ranks
template <int NUM_THREADS_PER_BLOCK, int NUM_OF_WARPS_PER_BLOCK, int LSA_TEAM_SIZE>
__device__ __forceinline__ void
cross_block_prefix_scan_ranks(scan_flat_smem_t& smem, tmp_state_t* tmp, int block_id) {

    // Publish per-rank block sums using `PRIV_SUM` as an atomic signal for other blocks
    for (int i = threadIdx.x; i < LSA_TEAM_SIZE; i += NUM_THREADS_PER_BLOCK) {
        int32_t rank_acc = 0;
#pragma unroll
        for (int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++) {
            rank_acc += smem.warp_rank_sums[j * LSA_TEAM_SIZE + i];
        }

        tmp_state_t tmp_data{PRIV_SUM, rank_acc};
        uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
        nccl_ep::st_relaxed_gpu_global(reinterpret_cast<uint64_t*>(&tmp[block_id * LSA_TEAM_SIZE + i]), data);
    }

    // Traverse LSA team ranks (assigned to threads). For each rank, scan through all blocks
    // and accumulate the per-rank block sums.
    //   * wait on the `PRIV_SUM` signal to ensure block arrival
    //   * Calculate the per-rank block prefix by adding the accumulated block sums
    for (int i = threadIdx.x; i < LSA_TEAM_SIZE; i += NUM_THREADS_PER_BLOCK) {
        int32_t previous_block_sum_for_current_rank = 0;
        for (int j = 0; j < block_id; j++) {
            tmp_state_t tmp_data{EMPTY, 0};
            tmp_state_t* tmp_src = &tmp[j * LSA_TEAM_SIZE + i];
            do {
                uint64_t data = nccl_ep::ld_relaxed_gpu_global(reinterpret_cast<const uint64_t*>(tmp_src));
                tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
            } while (tmp_data.state != PRIV_SUM);
            previous_block_sum_for_current_rank += tmp_data.value;
        }
        smem.block_prefix[i] = previous_block_sum_for_current_rank;
    }
}

template <int NUM_OF_WARPS_PER_BLOCK, int LSA_TEAM_SIZE>
__device__ __forceinline__ void init_warp_rank_prefixes(scan_flat_smem_t& smem) {
    if (threadIdx.x < LSA_TEAM_SIZE) {
        const int rank = threadIdx.x;
        int32_t prefix = smem.block_prefix[rank];
#pragma unroll
        for (int warp = 0; warp < NUM_OF_WARPS_PER_BLOCK; warp++) {
            smem.warp_prefix[warp * LSA_TEAM_SIZE + rank] = prefix;
            prefix += smem.warp_rank_sums[warp * LSA_TEAM_SIZE + rank];
        }
    }
}

__device__ __forceinline__ int32_t warp_excl_scan(bool participates, int lane_id, int32_t& tile_sum_out) {
    int32_t temp_scan = participates ? 1 : 0;
#pragma unroll
    for (int k = 1; k < WARP_SIZE; k *= 2) {
        int32_t temp = __shfl_up_sync(~0u, temp_scan, k);
        if (lane_id >= k) temp_scan += temp;
    }

    tile_sum_out = __shfl_sync(~0u, temp_scan, WARP_SIZE - 1);
    int32_t exclusive_scan = __shfl_up_sync(~0u, temp_scan, 1);
    return (lane_id >= 1) ? exclusive_scan : 0;
}

template <bool ENABLE_PER_EXPERT_COUNTS, bool ENABLE_EM_PERMUTE = false>
__device__ __forceinline__ void write_local_routing(
    const uint8_t* input_routing_map,
    bool* local_expert_routing_map,
    scan_flat_smem_t& smem,
    const scan_geometry_t& g,
    int current_token_id,
    int token_out_of_bound,
    bool token_needed_by_local_rank,
    int32_t local_rank_slot,
    int node_rank,
    int local_rank,
    int experts_per_rank,
    bool expert_major,
    // EM-permute related params
    int32_t* flat2em_slot_map = nullptr,
    int em_top_k = 0) {
    bool lane_participates = (token_out_of_bound == 0) && token_needed_by_local_rank;
    const uint8_t* local_rank_bitmap_row = nullptr;
    bool* local_expert_routing_map_store_base_addr = nullptr;

    if (lane_participates) {
        local_rank_bitmap_row =
            input_routing_map + current_token_id * g.packed_row_bytes + node_rank * g.experts_per_node_packed;
        if (!expert_major) {
            local_expert_routing_map_store_base_addr = local_expert_routing_map + local_rank_slot * experts_per_rank;
        }
    }

    int em_k = 0;  // number of local-expert hits emitted for this lane's token
    const int local_expert_bit_base = local_rank * experts_per_rank;
    for (int k = 0; k < experts_per_rank; k++) {
        int expert_bit = local_expert_bit_base + k;
        bool routed_to_expert = false;
        if (lane_participates) {
            routed_to_expert = ((local_rank_bitmap_row[expert_bit / 8] >> (expert_bit % 8)) & 1u) != 0;
            if (!expert_major) {
                local_expert_routing_map_store_base_addr[k] = routed_to_expert;
            }
        }

        if constexpr (ENABLE_EM_PERMUTE) {
            // Each thread is processing a unique token (assigned to it by the scan geometry)
            // 1. If thread's current token is routed to the local expert k, 
            //    the ballot_sync establishes its sequential number across all warp threads
            //    routing their tokens to the same local expert
            // 2. `em_k` is used to track the number of local-expert hits emitted for this lane's token
            // 3. `pref_k` is initialized by `init_warp_expert_prefixes()` to it's initial offset
            //    and is advanced every iteration by the number of local-expert hits observed
            //    cumulatively by all warp's lanes: `__popc(expert_mask)`
            const unsigned expert_mask = __ballot_sync(~0u, routed_to_expert);
            const int pref_k = smem.warp_expert_prefix[g.warp_id * experts_per_rank + k];  // read before update
            if (routed_to_expert) {
                const int within = __popc(expert_mask & ((1u << g.lane_id) - 1u));
                const int em_slot = smem.expert_base[k] + pref_k + within;
                if (em_k < em_top_k) {
                    flat2em_slot_map[static_cast<size_t>(local_rank_slot) * em_top_k + em_k] = em_slot;
                }
                em_k++;
            }
            __syncwarp();  // all lanes read pref_k before lane 0 overwrites it
            if (g.lane_id == 0) {
                smem.warp_expert_prefix[g.warp_id * experts_per_rank + k] = pref_k + __popc(expert_mask);
            }
            __syncwarp();  // publish update before the next tile re-reads this slot
        } else if constexpr (ENABLE_PER_EXPERT_COUNTS) {
            unsigned expert_mask = __ballot_sync(~0u, routed_to_expert);
            if (g.lane_id == 0) {
                int warp_expert_count = __popc(expert_mask);
                if (warp_expert_count > 0) {
                    atomicAdd(smem.expert_counts + k, warp_expert_count);
                }
            }
        }
    }

    // Pad trailing flat2em slots with -1 so the copy kernel skips them.
    if constexpr (ENABLE_EM_PERMUTE) {
        if (lane_participates) {
            for (int k = em_k; k < em_top_k; k++) {
                flat2em_slot_map[static_cast<size_t>(local_rank_slot) * em_top_k + k] = -1;
            }
        }
    }
}

template <int NUM_RANK_TILES, int LSA_TEAM_SIZE, int NUM_MASK_WORDS, bool ENABLE_PER_EXPERT_COUNTS,
          bool ENABLE_EM_PERMUTE = false>
__device__ __forceinline__ void assign_recv_slots(
    const uint8_t* input_routing_map,
    int32_t* sparse_to_dense_map,
    rank_mask_t<NUM_MASK_WORDS>* token_rank_mask,
    bool* local_expert_routing_map,
    int32_t* num_of_tokens_for_experts,
    void* recv_total_counter,
    scan_flat_smem_t& smem,
    const scan_geometry_t& g,
    int tokens_per_rank,
    int experts_per_rank,
    int node_rank,
    int local_rank,
    bool expert_major,
    bool out_is_int64,
    int max_recv_tokens_per_rank,
    bool allow_overflow_drop,
    bool* rdma_to_attn_map,
    int32_t* token_to_recv_slot = nullptr,
    // EM-permute outputs (ENABLE_EM_PERMUTE only)
    int32_t* flat2em_slot_map = nullptr,
    int em_top_k = 0) {

    // Scan the range of assigned tokens (defined by the scan geometry)
    for (int i = 0; i < g.num_of_tokens_per_thread; i++) {
        int current_token_id = g.thread_starting_token + i * WARP_SIZE;
        int token_out_of_bound = 0;
        if (current_token_id >= g.num_of_total_attn_tokens) token_out_of_bound = 1;
        if (__all_sync(~0u, token_out_of_bound) != 0) break;

        const int TOKENS_PER_LSA_TEAM = tokens_per_rank * LSA_TEAM_SIZE;
        int src_lsa_team = current_token_id / TOKENS_PER_LSA_TEAM;
        int src_local_rank = (current_token_id % TOKENS_PER_LSA_TEAM) / tokens_per_rank;
        int src_token_id = current_token_id % tokens_per_rank;

        rank_mask_t<NUM_MASK_WORDS> rank_mask;
        rank_mask.clear();
        if (token_out_of_bound == 0)
            rank_mask = token_rank_mask[current_token_id];

        bool token_needed_by_local_rank = false;
        int32_t local_rank_slot = -1;
        int32_t local_rank_prefix_after_scan = 0;
        // Drop policy: track whether any destination slot for this token survived.
        bool token_any_slot_kept = false;

#pragma unroll NUM_RANK_TILES
        for (int tile = 0; tile < NUM_RANK_TILES; tile++) {
            const int rank_base = tile * 32;
            const int tile_width = (LSA_TEAM_SIZE - rank_base < 32) ? (LSA_TEAM_SIZE - rank_base) : 32;

            int32_t previous_token_sum[32];
#pragma unroll
            for (int j = 0; j < tile_width; j++) {
                previous_token_sum[j] = smem.warp_prefix[g.warp_id * LSA_TEAM_SIZE + rank_base + j];
            }

#pragma unroll
            for (int j = 0; j < tile_width; j++) {
                const int rank = rank_base + j;
                bool token_needed_by_this_rank = (rank < LSA_TEAM_SIZE) && rank_mask.test(rank);
                int32_t temp_sum = 0;
                int32_t temp_scan =
                    warp_excl_scan(token_out_of_bound == 0 && token_needed_by_this_rank, g.lane_id, temp_sum);
                int32_t final_ex_scan = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;
                previous_token_sum[j] += temp_sum;

                if (!expert_major && token_out_of_bound == 0 && src_local_rank == local_rank &&
                    rank < LSA_TEAM_SIZE) {
                    // Drop policy: slot at/above capacity OOBs the recv buffer; mark -1.
                    int32_t slot = drop_overflow_slot(final_ex_scan, allow_overflow_drop, max_recv_tokens_per_rank);
                    sparse_to_dense_map
                        [(src_lsa_team * tokens_per_rank + src_token_id) *
                             LSA_TEAM_SIZE +
                         rank] = slot;
                    if (slot != -1) token_any_slot_kept = true;
                }

                if (rank == local_rank) {
                    token_needed_by_local_rank = token_needed_by_this_rank;
                    local_rank_slot = final_ex_scan;
                    local_rank_prefix_after_scan = previous_token_sum[j];
                }
            }

            if (g.lane_id == 0) {
#pragma unroll
                for (int j = 0; j < tile_width; j++) {
                    smem.warp_prefix[g.warp_id * LSA_TEAM_SIZE + rank_base + j] = previous_token_sum[j];
                }
            }
            __syncwarp();
        }

        // Drop policy: clear combine gate for fully-dropped tokens to avoid combine deadlock.
        if (!expert_major && allow_overflow_drop && token_out_of_bound == 0 &&
            src_local_rank == local_rank && rank_mask.any() && !token_any_slot_kept) {
            rdma_to_attn_map[src_lsa_team * g.rdma_to_attn_map_size_per_node +
                             src_token_id] = false;
        }

        write_local_routing<ENABLE_PER_EXPERT_COUNTS, ENABLE_EM_PERMUTE>(
            input_routing_map,
            local_expert_routing_map,
            smem,
            g,
            current_token_id,
            token_out_of_bound,
            token_needed_by_local_rank,
            local_rank_slot,
            node_rank,
            local_rank,
            experts_per_rank,
            expert_major,
            flat2em_slot_map,
            em_top_k);

        // EM-permute updates warp_expert_prefix in smem; keep the warp converged
        // before the next tile re-reads it.
        if constexpr (ENABLE_EM_PERMUTE)
            __syncwarp();

        // em-permute: persist recv slot; -1 if no local hit or dropped.
        if (token_to_recv_slot != nullptr && token_out_of_bound == 0) {
            int32_t slot = drop_overflow_slot(token_needed_by_local_rank ? local_rank_slot : -1,
                                              allow_overflow_drop, max_recv_tokens_per_rank);
            token_to_recv_slot[current_token_id] = slot;
        }

        if (!expert_major && current_token_id == g.num_of_total_attn_tokens - 1) {
            const int32_t true_total = local_rank_prefix_after_scan;
            const bool overflow = true_total > max_recv_tokens_per_rank;
            if (overflow && !allow_overflow_drop) {
                printf(
                    "ncclEpUpdateHandle: HT FLAT actual recv tokens %d > "
                    "max_recv_tokens_per_rank %d on (node %d local %d); "
                    "increase ncclEpGroupConfig_t::max_recv_tokens_per_rank "
                    "or set ncclEpGroupConfig_t::overflow_policy = NCCL_EP_OVERFLOW_DROP\n",
                    true_total,
                    max_recv_tokens_per_rank,
                    node_rank,
                    local_rank);
                __trap();
            }
            // Internal count drives recv processing, so clamp to capacity (slots above
            // it were dropped). recv_total_counter reports the true pre-drop total.
            *num_of_tokens_for_experts = overflow ? max_recv_tokens_per_rank : true_total;
            // EM-permute reports the padded total via recv_total_counter (written
            // in scan_impl_flat's EM block); don't clobber it with the unpadded
            // FLAT count here.
            if constexpr (!ENABLE_EM_PERMUTE) {
                if (recv_total_counter) {
                    if (out_is_int64) {
                        *static_cast<int64_t*>(recv_total_counter) = static_cast<int64_t>(true_total);
                    } else {
                        *static_cast<int32_t*>(recv_total_counter) = true_total;
                    }
                }
            }
        }
    }
}

template <int NUM_LSA_TEAMS, int NUM_THREADS_PER_BLOCK, int NUM_OF_BLOCKS, int LSA_TEAM_SIZE>
__device__ __forceinline__ void fill_attn_to_rdma(
    const uint8_t* input_routing_map,
    bool* attn_to_rdma_map,
    const scan_geometry_t& g,
    int num_of_tokens_per_rank,
    int experts_per_rank,
    int node_rank,
    int local_rank) {
    if constexpr (NUM_LSA_TEAMS == 1) return;

    constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;
    const int num_of_total_token_rows = (NUM_LSA_TEAMS - 1) * num_of_tokens_per_rank;
    const int num_of_token_rows_per_thread = ((num_of_total_token_rows - 1) / NUM_OF_TOTAL_THREADS) + 1;
    int tid = threadIdx.x + blockIdx.x * NUM_THREADS_PER_BLOCK;
    const int experts_per_node = experts_per_rank * LSA_TEAM_SIZE;

    for (int i = 0; i < num_of_token_rows_per_thread; i++) {
        int current_token_id = i * NUM_OF_TOTAL_THREADS + tid;
        if (current_token_id >= num_of_total_token_rows) break;

        int attn_node_id = current_token_id % (NUM_LSA_TEAMS - 1);
        int current_token_node_id = attn_node_id < node_rank ? attn_node_id : attn_node_id + 1;
        int current_token_local_id = current_token_id / (NUM_LSA_TEAMS - 1);

        const uint8_t* bitmap_row =
            input_routing_map +
            ((node_rank * LSA_TEAM_SIZE + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                g.packed_row_bytes +
            current_token_node_id * g.experts_per_node_packed;

        bool* attn_to_rdma_map_base_addr =
            attn_to_rdma_map + (current_token_local_id * (NUM_LSA_TEAMS - 1) + attn_node_id);
        *attn_to_rdma_map_base_addr = bitmap_range_has_set_bit(bitmap_row, 0, experts_per_node);
    }
}

template <int NUM_THREADS_PER_BLOCK, int NUM_OF_BLOCKS, int NUM_LSA_TEAMS, int LSA_TEAM_SIZE>
__device__ __forceinline__ scan_geometry_t
compute_scan_geometry(int num_of_tokens_per_rank, int experts_per_rank) {
    constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;
    constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;

    scan_geometry_t g;
    g.num_of_total_attn_tokens = num_of_tokens_per_rank * LSA_TEAM_SIZE * NUM_LSA_TEAMS;
    g.num_of_tokens_per_thread = ((g.num_of_total_attn_tokens - 1) / NUM_OF_TOTAL_THREADS) + 1;
    g.num_of_tokens_per_warp = g.num_of_tokens_per_thread * WARP_SIZE;
    g.num_of_tokens_per_block = g.num_of_tokens_per_warp * NUM_OF_WARPS_PER_BLOCK;
    g.rdma_to_attn_map_size_per_node = rdma_to_attn_row_stride(num_of_tokens_per_rank);

    const int experts_per_node = experts_per_rank * LSA_TEAM_SIZE;
    g.experts_per_node_packed = (experts_per_node + 7) / 8;
    g.packed_row_bytes = g.experts_per_node_packed * NUM_LSA_TEAMS;

    const int block_starting_token = blockIdx.x * g.num_of_tokens_per_block;
    g.warp_id = threadIdx.x / WARP_SIZE;
    g.lane_id = threadIdx.x % WARP_SIZE;
    const int warp_starting_token = block_starting_token + g.warp_id * g.num_of_tokens_per_warp;
    g.thread_starting_token = warp_starting_token + g.lane_id;
    return g;
}

// ---------------------------------------------------------------------------
// EM-permute fusion helpers: extra phases that let scan_impl_flat also produce the
// em-permute outputs (flat2em_slot_map + per-expert padded zone geometry) that
// em_scan_kernel used to build, so the second cooperative scan can be dropped.
// Gated at the scan_impl_flat call sites by `if constexpr (ENABLE_EM_PERMUTE)`.
// ---------------------------------------------------------------------------

// Scans the range of assigned tokens (defined by the scan geometry)
// * Typically each thread gets at most one token (512 threads/block x 16 blocks is 8K threads)
// Logic for each token:
// * Calculate per-local-expert `expert_sums` indicating how many tokens each local expert receives
// * Warp-reduces `expert_sums` into `smem.expert_warp_sums`
template <int EXPERTS_PER_RANK>
__device__ __forceinline__ void extract_local_experts_meta(
    const uint8_t* input_routing_map,
    scan_flat_smem_t& smem,
    const scan_geometry_t& g,
    int lsa_team_id,
    int local_rank) {
    int32_t expert_sums[EXPERTS_PER_RANK];
#pragma unroll
    for (int k = 0; k < EXPERTS_PER_RANK; k++) expert_sums[k] = 0;

    const int local_expert_bit_base = local_rank * EXPERTS_PER_RANK;
    for (int i = 0; i < g.num_of_tokens_per_thread; i++) {
        const int tok = g.thread_starting_token + i * WARP_SIZE;
        if (tok >= g.num_of_total_attn_tokens) break;
        const uint8_t* row =
            input_routing_map + static_cast<size_t>(tok) * g.packed_row_bytes +
            lsa_team_id * g.experts_per_node_packed;
#pragma unroll
        for (int k = 0; k < EXPERTS_PER_RANK; k++) {
            const int bit = local_expert_bit_base + k;
            if ((row[bit >> 3] >> (bit & 7)) & 1u) expert_sums[k]++;
        }
    }
    // All lanes converge here (loop exit); reduce per-lane sums per expert.
#pragma unroll
    for (int k = 0; k < EXPERTS_PER_RANK; k++) {
        const int32_t s = __reduce_add_sync(~0u, expert_sums[k]);
        if (g.lane_id == 0) smem.expert_warp_sums[g.warp_id * EXPERTS_PER_RANK + k] = s;
    }
}

template <int NUM_THREADS_PER_BLOCK, int NUM_OF_WARPS_PER_BLOCK, int NUM_OF_BLOCKS>
__device__ __forceinline__ void cross_block_prefix_scan_experts(
    scan_flat_smem_t& smem,
    tmp_state_t* expert_scan_tmp,
    int experts_per_rank,
    int block_id) {

    // Publish per-expert block sums using `PRIV_SUM` as an atomic signal for other blocks
    for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
        int32_t acc = 0;
#pragma unroll
        for (int w = 0; w < NUM_OF_WARPS_PER_BLOCK; w++) acc += smem.expert_warp_sums[w * experts_per_rank + e];
        tmp_state_t data{PRIV_SUM, acc};
        uint64_t bits = *reinterpret_cast<uint64_t*>(&data);
        nccl_ep::st_relaxed_gpu_global(
            reinterpret_cast<uint64_t*>(&expert_scan_tmp[block_id * experts_per_rank + e]), bits);
    }

    // Traverse local rank's experts (assigned to threads). For each local expert, scan through all blocks
    // and accumulate the per-expert block sums.
    //   * wait on the `PRIV_SUM` signal to ensure block arrival
    //   * Calculate the per-expert block prefix by adding the accumulated block sums
    //   * Calculate the grand total across ALL blocks by summing the accumulated block sums
    for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
        int32_t prefix = 0;
        int32_t total = 0;
        for (int b = 0; b < NUM_OF_BLOCKS; b++) {
            tmp_state_t data{EMPTY, 0};
            tmp_state_t* src = &expert_scan_tmp[b * experts_per_rank + e];
            do {
                uint64_t bits = nccl_ep::ld_relaxed_gpu_global(reinterpret_cast<const uint64_t*>(src));
                data = *reinterpret_cast<tmp_state_t*>(&bits);
            } while (data.state != PRIV_SUM);
            if (b < block_id) prefix += data.value;
            total += data.value;
        }
        smem.expert_block_prefix[e] = prefix;
        smem.expert_total[e] = total;
    }
}

// Thread 0 turns the per-expert cross-block grand totals (smem.expert_total)
// into padded per-expert zone bases (smem.expert_base, computed by every block),
// and block 0 additionally publishes the global EM count/offset tensors
// (internal offsets, padded/actual counts, out offsets, and the optional padded
// recv_total_counter). If the padded EM total exceeds the recv budget it traps,
// unless allow_overflow_drop is set, in which case reported counts are clamped to
// the budget and recv_total_counter reports the true pre-drop total.
template <typename EM_OUT_T>
__device__ __forceinline__ void em_populate_cnt_tensors(
    scan_flat_smem_t& smem,
    int experts_per_rank,
    int em_alignment,
    int64_t* em_internal_offsets,
    EM_OUT_T* em_padded_out_counts,
    EM_OUT_T* em_out_offsets,
    int32_t* em_actual_counts_out,
    void* recv_total_counter,
    bool out_is_int64,
    int max_recv_tokens_per_rank,
    bool allow_overflow_drop) {
    if (threadIdx.x == 0) {
        const int align = (em_alignment > 1) ? em_alignment : 1;
        int cum = 0;
        for (int k = 0; k < experts_per_rank; k++) {
            const int c = smem.expert_total[k];
            smem.expert_base[k] = cum;
            const int padded = (align > 1 && c > 0) ? ((c + align - 1) / align) * align : c;
            if (blockIdx.x == 0) {
                // Drop policy: clamp reported counts to remaining budget; expert_base / cum
                // (slot assignment) stay unchanged so emit-phase slots still match.
                int rep_actual = c;
                int rep_padded = padded;
                if (allow_overflow_drop) {
                    const int room = (cum < max_recv_tokens_per_rank) ? (max_recv_tokens_per_rank - cum) : 0;
                    rep_actual = (c < room) ? c : room;
                    rep_padded = (padded < room) ? padded : room;
                }
                if (em_internal_offsets) em_internal_offsets[k] = cum;
                if (em_actual_counts_out) em_actual_counts_out[k] = rep_actual;
                if (em_padded_out_counts) em_padded_out_counts[k] = static_cast<EM_OUT_T>(rep_padded);
                if (em_out_offsets) em_out_offsets[k] = static_cast<EM_OUT_T>(cum);
            }
            cum += padded;
        }
        const int true_total = cum;
        const bool overflow = true_total > max_recv_tokens_per_rank;
        if (blockIdx.x == 0) {
            // Internal total drives recv processing, so clamp to capacity (slots above it
            // were dropped); recv_total_counter still reports the true pre-drop padded total.
            if (em_internal_offsets)
                em_internal_offsets[experts_per_rank] = overflow ? max_recv_tokens_per_rank : true_total;
            // EM user-visible recv-token count is the padded total (matches
            // getNumRecvTokens). num_of_tokens_for_experts stays the unpadded
            // FLAT count (the permute/dispatch path relies on it); only the
            // optional user recv_total_counter reports the padded total, and
            // the FLAT branch of assign_recv_slots is suppressed for it.
            if (recv_total_counter) {
                if (out_is_int64) {
                    *static_cast<int64_t*>(recv_total_counter) = static_cast<int64_t>(true_total);
                } else {
                    *static_cast<int32_t*>(recv_total_counter) = true_total;
                }
            }
        }
        if (overflow && !allow_overflow_drop) {
            printf("scan_impl_flat(em): padded EM slots %d > max_recv_tokens_per_rank %d; "
                   "increase ncclEpGroupConfig_t::max_recv_tokens_per_rank "
                   "or set overflow_policy = NCCL_EP_OVERFLOW_DROP\n",
                   true_total, max_recv_tokens_per_rank);
            __trap();
        }
    }
}

// EM-permute counterpart of init_warp_rank_prefixes: seed each warp's per-expert
// prefix with the block predecessor prefix plus the sums of earlier warps.
template <int NUM_OF_WARPS_PER_BLOCK>
__device__ __forceinline__ void init_warp_expert_prefixes(scan_flat_smem_t& smem, int experts_per_rank) {
    if (threadIdx.x < experts_per_rank) {
        const int e = threadIdx.x;
        int32_t prefix = smem.expert_block_prefix[e];
#pragma unroll
        for (int w = 0; w < NUM_OF_WARPS_PER_BLOCK; w++) {
            smem.warp_expert_prefix[w * experts_per_rank + e] = prefix;
            prefix += smem.expert_warp_sums[w * experts_per_rank + e];
        }
    }
}

// FLAT-layout scan, optionally fused with EM-permute. Always produces
// token_rank_mask, rdma/attn maps, sparse_to_dense (rank-major), and recv-slot
// prefixes. Two mutually-exclusive add-ons, selected at compile time:
//   ENABLE_PER_EXPERT_COUNTS -> FLAT per-expert token counts.
//   ENABLE_EM_PERMUTE           -> em-permute outputs: flat2em_slot_map and the
//                               [experts_per_rank(+1)] padded zone geometry
//                               (replaces the standalone em_scan_kernel).
// The EM add-on requires expert_scan_tmp + the em_* outputs + EXPERTS_PER_RANK and
// its smem region carved past the FLAT region; all are ignored when disabled, so
// the FLAT instantiation is byte-identical and pays no register/smem cost.
template <
    int NUM_THREADS_PER_BLOCK,
    int NUM_OF_BLOCKS,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    bool ENABLE_PER_EXPERT_COUNTS,
    bool ENABLE_EM_PERMUTE = false,
    int EXPERTS_PER_RANK = 0,
    typename EM_OUT_T = int32_t>  // dtype of em_padded_out_counts / em_out_offsets
__device__ __forceinline__ void scan_impl_flat(
    const uint8_t* input_routing_map,
    tmp_state_t* tmp,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    rank_mask_t<(LSA_TEAM_SIZE + 63) / 64>* token_rank_mask,
    int32_t* num_of_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    const int node_rank,
    const int local_rank,
    const int num_of_tokens_per_rank,
    const int experts_per_rank,
    void* recv_total_counter,
    bool out_is_int64,
    int max_recv_tokens_per_rank,
    bool allow_overflow_drop,
    int32_t* token_to_recv_slot,
    uint8_t* smem_bytes,
    // EM-permute inputs/outputs; ignored unless ENABLE_EM_PERMUTE.
    tmp_state_t* expert_scan_tmp = nullptr,
    int32_t* flat2em_slot_map = nullptr,
    int em_top_k = 0,
    int em_alignment = 1,
    int64_t* em_internal_offsets = nullptr,
    EM_OUT_T* em_padded_out_counts = nullptr,
    EM_OUT_T* em_out_offsets = nullptr,
    int32_t* em_actual_counts_out = nullptr) {
    static_assert(
        LSA_TEAM_SIZE <= EM_S2D_MAX_RANKS,
        "em_s2d_pack rank field is 10 bits; LSA team size must fit in 1024");
    static_assert(
        LSA_TEAM_SIZE <= NUM_THREADS_PER_BLOCK,
        "scan_impl_flat requires one block thread per LSA rank to initialize warp prefixes");
    static_assert(
        !(ENABLE_PER_EXPERT_COUNTS && ENABLE_EM_PERMUTE),
        "per-expert-counts and EM-permute both drive the expert ballot; enable at most one");

    constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;
    constexpr int NUM_MASK_WORDS = (LSA_TEAM_SIZE + 63) / 64;
    constexpr int NUM_RANK_TILES = (LSA_TEAM_SIZE + 31) / 32;

    // LSA_TEAM_SIZE (compile-time ranks-per-LSA-team) is the runtime
    // num_of_ranks_per_node by construction, so use it directly.
    const scan_geometry_t g = compute_scan_geometry<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_LSA_TEAMS, LSA_TEAM_SIZE>(
        num_of_tokens_per_rank,
        experts_per_rank);

    // One layout owns both the rank region and the (mutually exclusive) per-expert
    // counts / EM-permute regions, so the sub-regions can't overlap. Its EM fields
    // (smem.expert_*) are null unless ENABLE_EM_PERMUTE.
    scan_flat_smem_t smem = scan_flat_smem_t::from_raw<ENABLE_PER_EXPERT_COUNTS, ENABLE_EM_PERMUTE>(
        smem_bytes, NUM_OF_WARPS_PER_BLOCK, LSA_TEAM_SIZE, experts_per_rank);

    if constexpr (ENABLE_PER_EXPERT_COUNTS) {
        for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
            smem.expert_counts[e] = 0;
        }
        __syncthreads();
    }

    // Phase 1: rank tally (+ per-local-expert tally when EM-permute is enabled).
    // Passing &smem drives the warp-reduce of rank counts into smem.warp_rank_sums.
    extract_lsa_ranks_meta<LSA_TEAM_SIZE, NUM_MASK_WORDS>(
        input_routing_map,
        token_rank_mask,
        rdma_to_attn_map,
        g,
        num_of_tokens_per_rank,
        experts_per_rank,
        node_rank,
        local_rank,
        &smem);

    if constexpr (ENABLE_EM_PERMUTE) {
        extract_local_experts_meta<EXPERTS_PER_RANK>(input_routing_map, smem, g, node_rank, local_rank);
    }

    __syncthreads();

    // Cross-block prefixes: ranks (predecessor-only) + experts (predecessor +
    // grand total for padded bases, EM-permute only).
    cross_block_prefix_scan_ranks<NUM_THREADS_PER_BLOCK, NUM_OF_WARPS_PER_BLOCK, LSA_TEAM_SIZE>(
        smem,
        tmp,
        blockIdx.x);
    if constexpr (ENABLE_EM_PERMUTE) {
        cross_block_prefix_scan_experts<NUM_THREADS_PER_BLOCK, NUM_OF_WARPS_PER_BLOCK, NUM_OF_BLOCKS>(
            smem, expert_scan_tmp, experts_per_rank, blockIdx.x);
    }
    __syncthreads();

    init_warp_rank_prefixes<NUM_OF_WARPS_PER_BLOCK, LSA_TEAM_SIZE>(smem);

    if constexpr (ENABLE_EM_PERMUTE) {
        // Padded per-expert zone bases from grand totals; block 0 publishes the
        // global EM offset arrays. Every block computes expert_base locally.
        em_populate_cnt_tensors<EM_OUT_T>(
            smem,
            experts_per_rank,
            em_alignment,
            em_internal_offsets,
            em_padded_out_counts,
            em_out_offsets,
            em_actual_counts_out,
            recv_total_counter,
            out_is_int64,
            max_recv_tokens_per_rank,
            allow_overflow_drop);

        // Seed per-warp expert prefixes: block predecessor prefix + earlier warps.
        init_warp_expert_prefixes<NUM_OF_WARPS_PER_BLOCK>(smem, experts_per_rank);
    }
    __syncthreads();

    // Phase 2: single token pass -> FLAT recv slots + FLAT LERM (+ EM slots).
    assign_recv_slots<NUM_RANK_TILES, LSA_TEAM_SIZE, NUM_MASK_WORDS, ENABLE_PER_EXPERT_COUNTS, ENABLE_EM_PERMUTE>(
        input_routing_map,
        sparse_to_dense_map,
        token_rank_mask,
        local_expert_routing_map,
        num_of_tokens_for_experts,
        recv_total_counter,
        smem,
        g,
        num_of_tokens_per_rank,
        experts_per_rank,
        node_rank,
        local_rank,
        /*expert_major=*/false,
        out_is_int64,
        max_recv_tokens_per_rank,
        allow_overflow_drop,
        rdma_to_attn_map,
        token_to_recv_slot,
        flat2em_slot_map,
        em_top_k);

    if constexpr (ENABLE_PER_EXPERT_COUNTS) {
        __syncthreads();
        for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
            int32_t block_count = smem.expert_counts[e];
            if (block_count > 0) atomicAdd(per_expert_token_counts + e, block_count);
        }
    }

    fill_attn_to_rdma<NUM_LSA_TEAMS, NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, LSA_TEAM_SIZE>(
        input_routing_map,
        attn_to_rdma_map,
        g,
        num_of_tokens_per_rank,
        experts_per_rank,
        node_rank,
        local_rank);
}

// EM-layout pre-scan: produces only what em_scan_kernel (see hybridep_adapter.cu)
// and the rest of the EM pipeline still need from the global routing bitmap --
// the per-token rank mask, rdma_to_attn_map (filled by extract_lsa_ranks_meta), and
// attn_to_rdma_map. The FLAT-only prefix scan / sparse_to_dense / per-expert
// counts are skipped entirely.
template <int NUM_THREADS_PER_BLOCK, int NUM_OF_BLOCKS, int NUM_LSA_TEAMS, int LSA_TEAM_SIZE>
__device__ __forceinline__ void scan_impl_em(
    const uint8_t* input_routing_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    rank_mask_t<(LSA_TEAM_SIZE + 63) / 64>* token_rank_mask,
    const int node_rank,
    const int local_rank,
    const int num_of_tokens_per_rank,
    const int experts_per_rank) {
    static_assert(
        LSA_TEAM_SIZE <= EM_S2D_MAX_RANKS,
        "em_s2d_pack rank field is 10 bits; LSA team size must fit in 1024");

    constexpr int NUM_MASK_WORDS = (LSA_TEAM_SIZE + 63) / 64;

    // LSA_TEAM_SIZE (compile-time ranks-per-LSA-team) equals the runtime
    // num_of_ranks_per_node by construction, so use it directly.
    const scan_geometry_t g = compute_scan_geometry<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_LSA_TEAMS, LSA_TEAM_SIZE>(
        num_of_tokens_per_rank,
        experts_per_rank);

    // EM pre-pass: no smem argument -> rank sums are skipped; only token_rank_mask
    // and rdma_to_attn_map are produced.
    extract_lsa_ranks_meta<LSA_TEAM_SIZE, NUM_MASK_WORDS>(
        input_routing_map,
        token_rank_mask,
        rdma_to_attn_map,
        g,
        num_of_tokens_per_rank,
        experts_per_rank,
        node_rank,
        local_rank);

    fill_attn_to_rdma<NUM_LSA_TEAMS, NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, LSA_TEAM_SIZE>(
        input_routing_map,
        attn_to_rdma_map,
        g,
        num_of_tokens_per_rank,
        experts_per_rank,
        node_rank,
        local_rank);
}

// Parameter packs for the scan JIT entries. Non-templated so the host can build
// them once without knowing LSA_TEAM_SIZE at compile time; the JIT-emitted
// wrappers reinterpret_cast token_rank_mask to rank_mask_t<(LSA_TEAM_SIZE+63)/64>*
// before calling the corresponding scan_impl.
struct scan_flat_kernel_param_t {
    const uint8_t* input_routing_map;
    tmp_state_t* tmp;
    int32_t* sparse_to_dense_map;
    bool* rdma_to_attn_map;
    bool* attn_to_rdma_map;
    void* token_rank_mask;
    int32_t* num_of_tokens_for_experts;
    bool* local_expert_routing_map;
    int32_t* per_expert_token_counts;
    int node_rank;
    int local_rank;
    int num_of_tokens_per_rank;
    int experts_per_rank;
    void* recv_total_counter;
    bool out_is_int64;
    int max_recv_tokens_per_rank;
    bool allow_overflow_drop;     // NCCL_EP_OVERFLOW_DROP: drop overflowing tokens instead of trapping.
    int32_t* token_to_recv_slot;  // em-permute scratch; null otherwise.

    // ---- EM-permute fusion outputs (scan_impl_flat<ENABLE_EM_PERMUTE=true> only; null on FLAT path) ----
    // When these are wired, scan_flat also produces what em_scan_kernel used to
    // build on the em-permute path, so em_scan_kernel can be skipped entirely.
    tmp_state_t* expert_scan_tmp;        // [num_of_blocks * experts_per_rank] decoupled-scan state
    int32_t* flat2em_slot_map;       // [num_recv, em_top_k] FLAT recv slot -> EM slot(s)
    int em_top_k;                    // width of flat2em_slot_map inner dim
    int em_alignment;                // per-expert zone padding multiple (>=1)
    int64_t* em_internal_offsets;    // [experts_per_rank + 1] padded zone base + total
    // count and offset datatypes are templated
    void* em_padded_out_counts;
    void* em_out_offsets;
    int32_t* em_actual_counts_out;   // [experts_per_rank] unpadded per-expert counts
};

struct scan_em_kernel_param_t {
    const uint8_t* input_routing_map;
    bool* rdma_to_attn_map;
    bool* attn_to_rdma_map;
    void* token_rank_mask;
    int node_rank;
    int local_rank;
    int num_of_tokens_per_rank;
    int experts_per_rank;
};

struct build_em_tables_param_t {
    const uint8_t* input_routing_map;
    const uint64_t* token_rank_mask_words;
    int num_mask_words;
    int num_total_attn_tokens;
    int num_tokens_per_rank;
    int num_lsa_teams;
    int node_rank;
    int local_rank;
    int s2d_inner_dim;
    int max_recv_tokens_per_rank;
    int em_alignment;
    int32_t* sparse_to_dense_map;
    // NCCL_EP_OVERFLOW_DROP: combine gate; non-null (only under drop) lets the
    // kernel clear it for fully-dropped send tokens. See allow_overflow_drop.
    bool*    rdma_to_attn_map;
    bool*    local_expert_routing_map;
    int32_t* num_tokens_for_experts;
    int64_t* em_internal_offsets;
    int32_t* em_padded_out_counts_i32;
    int64_t* em_padded_out_counts_i64;
    int32_t* em_out_offsets_i32;
    int64_t* em_out_offsets_i64;
    int32_t* em_actual_counts_out;
    int32_t* recv_total_counter_i32;
    int64_t* recv_total_counter_i64;
    bool     out_is_int64;
    int32_t* emuf_group_buf;
    int32_t* emuf_group_count;
    int      emuf_group_stride;
    int      emuf_max_groups;
    int32_t* gscratch;
    const int32_t* token_to_recv_slot;
    int32_t* flat2em_slot_map;
    int      em_top_k;
    // NCCL_EP_OVERFLOW_DROP: drop overflowing tokens instead of trapping.
    bool     allow_overflow_drop;
};

template<int MAX_EXPERTS_PER_RANK, int LSA_TEAM_SIZE>
__device__ void build_em_tables_impl(const build_em_tables_param_t& p) {
    namespace cg = cooperative_groups;
    extern __shared__ int32_t s_smem[];

    // MAX_EXPERTS_PER_RANK and LSA_TEAM_SIZE are JIT-baked template params.
    const int epr   = MAX_EXPERTS_PER_RANK;
    const int nrpn  = LSA_TEAM_SIZE;
    const int n_dle = nrpn * epr;
    const int packed_row_bytes =
        ((p.num_lsa_teams * nrpn * epr) + 7) / 8;

    int32_t* g_block_count = p.gscratch;

    const int local_per_node_bytes = ((nrpn * epr) + 7) / 8;
    // Matches the combine kernels' rdma_to_attn_map row padding (16 bools/node).
    const int rdma_to_attn_map_size_per_node = rdma_to_attn_row_stride(p.num_tokens_per_rank);
    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp      = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const int B         = blockIdx.x;
    const int N         = gridDim.x;
    const int s_warp_stride = n_dle;
    int32_t* s_warp_state = s_smem;
    int32_t* s_offsets    = s_smem + num_warps * n_dle;

    auto grid = cg::this_grid();

    const int tpb = (p.num_total_attn_tokens + N - 1) / N;
    const int bs  = B * tpb;
    const int be  = min(bs + tpb, p.num_total_attn_tokens);
    const int tpw = (be - bs + num_warps - 1) / num_warps;
    const int ws  = bs + warp * tpw;
    const int we  = min(ws + tpw, be);

    for (int i = tid; i < num_warps * n_dle; i += blockDim.x) s_warp_state[i] = 0;
    __syncthreads();

    const int n_local_bits_ph1  = nrpn * epr;
    const int n_local_words_ph1 = (n_local_bits_ph1 + 63) / 64;
    for (int tok = ws + lane; tok < we; tok += 32) {
        const uint64_t* mw = p.token_rank_mask_words + (size_t)tok * p.num_mask_words;
        const uint64_t mw0 = mw[0];
        const uint64_t mw1 = (p.num_mask_words >= 2) ? mw[1] : 0;
        if (mw0 == 0 && mw1 == 0) continue;
        const uint8_t* row = p.input_routing_map + (size_t)tok * packed_row_bytes
                           + (size_t)p.node_rank * local_per_node_bytes;
        for (int wi = 0; wi < n_local_words_ph1; wi++) {
            const int word_bit_base = wi * 64;
            const int remaining = n_local_bits_ph1 - word_bit_base;
            const int word_bits = remaining >= 64 ? 64 : remaining;
            uint64_t s = nccl_ep::em_ld64(row, wi * 8);
            if (word_bits < 64) s &= (uint64_t{1} << word_bits) - 1;
            while (s) {
                const int b = __ffsll(static_cast<long long>(s)) - 1;
                atomicAdd(&s_warp_state[warp * s_warp_stride + word_bit_base + b], 1);
                s &= s - 1;
            }
        }
    }
    __syncthreads();

    for (int dle = tid; dle < n_dle; dle += blockDim.x) {
        int sum = 0;
        for (int w = 0; w < num_warps; w++) sum += s_warp_state[w * s_warp_stride + dle];
        g_block_count[(size_t)B * n_dle + dle] = sum;
    }

    grid.sync();

    for (int dle = tid; dle < n_dle; dle += blockDim.x) {
        int my_prefix = 0;
        int total = 0;
        for (int b = 0; b < N; b++) {
            const int c = g_block_count[(size_t)b * n_dle + dle];
            if (b < B) my_prefix += c;
            total += c;
        }
        s_offsets[dle] = total;
        int cum = my_prefix;
        for (int w = 0; w < num_warps; w++) {
            const int c = s_warp_state[w * s_warp_stride + dle];
            s_warp_state[w * s_warp_stride + dle] = cum;
            cum += c;
        }
    }
    __syncthreads();

    {
        const int align = (p.em_alignment > 1) ? p.em_alignment : 1;
        for (int d = warp; d < nrpn; d += num_warps) {
            if (lane == 0) {
                const bool write_em = (B == 0 && d == p.local_rank);
                int cum = 0;
                for (int k = 0; k < epr; k++) {
                    const int c = s_offsets[d * epr + k];
                    const int padded = (align > 1 && c > 0) ? ((c + align - 1) / align) * align : c;
                    s_offsets[d * epr + k] = cum;
                    if (write_em) {
                        // Drop policy: clamp counts to budget; offsets unchanged so emit-phase slots still match.
                        int rep_actual = c;
                        int rep_padded = padded;
                        if (p.allow_overflow_drop) {
                            const int room = (cum < p.max_recv_tokens_per_rank)
                                                 ? (p.max_recv_tokens_per_rank - cum) : 0;
                            rep_actual = (c < room) ? c : room;
                            rep_padded = (padded < room) ? padded : room;
                        }
                        if (p.em_internal_offsets) p.em_internal_offsets[k] = cum;
                        if (p.em_actual_counts_out) p.em_actual_counts_out[k] = rep_actual;
                        if (p.out_is_int64) {
                            if (p.em_padded_out_counts_i64) p.em_padded_out_counts_i64[k] = (int64_t)rep_padded;
                            if (p.em_out_offsets_i64) p.em_out_offsets_i64[k] = (int64_t)cum;
                        } else {
                            if (p.em_padded_out_counts_i32) p.em_padded_out_counts_i32[k] = (int32_t)rep_padded;
                            if (p.em_out_offsets_i32) p.em_out_offsets_i32[k] = (int32_t)cum;
                        }
                    }
                    cum += padded;
                }
                const int true_total = cum;
                const bool overflow = true_total > p.max_recv_tokens_per_rank;
                if (overflow && !p.allow_overflow_drop) {
                    printf("build_em_tables: dest %d padded slots %d > "
                           "max_recv_tokens_per_rank %d; "
                           "increase ncclEpGroupConfig_t::max_recv_tokens_per_rank "
                           "or set overflow_policy = NCCL_EP_OVERFLOW_DROP\n",
                           d, true_total, p.max_recv_tokens_per_rank);
                    __trap();
                }
                if (write_em) {
                    // Clamp to capacity for internal processing; report true total externally.
                    const int kept_total = overflow ? p.max_recv_tokens_per_rank : true_total;
                    if (p.em_internal_offsets) p.em_internal_offsets[epr] = (int64_t)kept_total;
                    if (p.num_tokens_for_experts) *p.num_tokens_for_experts = (int32_t)kept_total;
                    if (p.out_is_int64) {
                        if (p.recv_total_counter_i64) *p.recv_total_counter_i64 = (int64_t)true_total;
                    } else {
                        if (p.recv_total_counter_i32) *p.recv_total_counter_i32 = (int32_t)true_total;
                    }
                }
            }
        }
    }
    __syncthreads();

    const int n_local_bits  = nrpn * epr;
    const int n_local_words = (n_local_bits + 63) / 64;
    const int num_tiles     = (we - ws + 31) / 32;
    const int epr_l2        = __ffs(epr) - 1;
    const int epr_mask      = epr - 1;
    for (int tile = 0; tile < num_tiles; tile++) {
        const int tok = ws + tile * 32 + lane;
        const bool valid = (tok < we);
        bool any_hit = false;
        if (valid) {
            const uint64_t* mw = p.token_rank_mask_words + (size_t)tok * p.num_mask_words;
            const uint64_t mw0 = mw[0];
            const uint64_t mw1 = (p.num_mask_words >= 2) ? mw[1] : 0;
            any_hit = (mw0 != 0) || (mw1 != 0);
        }

        const uint8_t* row_local = nullptr;
        int send_idx = 0;
        bool is_our_send = false;
        int src_node = 0;
        int src_local_id = 0;
        if (any_hit) {
            row_local = p.input_routing_map + (size_t)tok * packed_row_bytes
                      + (size_t)p.node_rank * local_per_node_bytes;
            const int sgr = tok / p.num_tokens_per_rank;
            const int sn  = sgr / nrpn;
            const int slr = sgr % nrpn;
            const int lti = tok % p.num_tokens_per_rank;
            send_idx = sn * p.num_tokens_per_rank + lti;
            is_our_send = (slr == p.local_rank);
            src_node = sn;
            src_local_id = lti;
        }

        int my_packed_idx = 0;
        // Drop policy: whether any local-node slot for this send token survived.
        bool token_any_slot_kept = false;
        int primary_em_slot = -1;
        int n_local_sec = 0;
        int local_secondaries[MAX_EXPERTS_PER_RANK];

        int my_local_packed_idx = 0;
        int my_recv_s = -1;
        if (p.flat2em_slot_map && any_hit) {
            my_recv_s = p.token_to_recv_slot[tok];
        }

        for (int wi = 0; wi < n_local_words; wi++) {
            const int word_bit_base = wi * 64;
            const int remaining = n_local_bits - word_bit_base;
            const int word_bits = remaining >= 64 ? 64 : remaining;
            uint64_t my_slice = 0;
            if (any_hit) {
                my_slice = nccl_ep::em_ld64(row_local, wi * 8);
                if (word_bits < 64) my_slice &= (uint64_t{1} << word_bits) - 1;
            }
            const uint32_t any_lo = __reduce_or_sync(0xffffffff, (uint32_t)my_slice);
            const uint32_t any_hi = (word_bits > 32)
                ? __reduce_or_sync(0xffffffff, (uint32_t)(my_slice >> 32)) : 0u;
            uint64_t union_slice = ((uint64_t)any_hi << 32) | (uint64_t)any_lo;

            while (union_slice) {
                const int b = __ffsll(static_cast<long long>(union_slice)) - 1;
                const int dle = word_bit_base + b;
                const int d = dle >> epr_l2;
                const int le = dle & epr_mask;
                const bool my_hit = (my_slice >> b) & 1ull;
                const uint32_t mask = __ballot_sync(0xffffffff, my_hit);
                if (my_hit) {
                    const int within = __popc(mask & ((1u << lane) - 1u));
                    const int em_slot = s_offsets[dle] + s_warp_state[warp * s_warp_stride + dle] + within;
                    // Drop policy: slot at/above capacity would OOB recv buffer; mark dropped.
                    const bool dropped = p.allow_overflow_drop && slot_overflows(em_slot, p.max_recv_tokens_per_rank);
                    if (d == p.local_rank) {
                        if (p.local_expert_routing_map && !dropped) {
                            p.local_expert_routing_map[em_slot * epr + le] = true;
                        }
                        if (p.emuf_group_buf != nullptr && !dropped) {
                            if (primary_em_slot < 0) {
                                primary_em_slot = em_slot;
                            } else {
                                local_secondaries[n_local_sec++] = em_slot;
                            }
                        }
                        if (p.flat2em_slot_map && my_recv_s >= 0 &&
                            my_local_packed_idx < p.em_top_k) {
                            p.flat2em_slot_map[(size_t)my_recv_s * p.em_top_k + my_local_packed_idx] =
                                dropped ? -1 : em_slot;
                            my_local_packed_idx++;
                        }
                    }
                    if (is_our_send) {
                        if (p.sparse_to_dense_map) {
                            p.sparse_to_dense_map[(size_t)send_idx * p.s2d_inner_dim + my_packed_idx] =
                                dropped ? -1 : em_s2d_pack(d, em_slot);
                        }
                        if (!dropped) token_any_slot_kept = true;
                        my_packed_idx++;
                    }
                }
                if (lane == 0) {
                    s_warp_state[warp * s_warp_stride + dle] += __popc(mask);
                }
                union_slice &= union_slice - 1;
            }
        }

        if (p.flat2em_slot_map && my_recv_s >= 0) {
            for (int k = my_local_packed_idx; k < p.em_top_k; k++) {
                p.flat2em_slot_map[(size_t)my_recv_s * p.em_top_k + k] = -1;
            }
        }

        // Drop policy: clear combine gate for fully-dropped send tokens; em-permute delegates this to FLAT scan.
        if (p.allow_overflow_drop && p.rdma_to_attn_map != nullptr && p.sparse_to_dense_map != nullptr &&
            is_our_send && my_packed_idx > 0 && !token_any_slot_kept) {
            p.rdma_to_attn_map[src_node * rdma_to_attn_map_size_per_node + src_local_id] = false;
        }

        if (p.emuf_group_buf != nullptr && n_local_sec > 0) {
            const int grp = atomicAdd(p.emuf_group_count, 1);
            if (grp >= p.emuf_max_groups) {
                __trap();
            }
            int32_t* row = p.emuf_group_buf + (size_t)grp * p.emuf_group_stride;
            row[0] = primary_em_slot;
            for (int s = 0; s < n_local_sec; s++) row[1 + s] = local_secondaries[s];
            if (1 + n_local_sec < p.emuf_group_stride) {
                row[1 + n_local_sec] = -1;
            }
        }
    }
}

} // namespace hybrid_ep
