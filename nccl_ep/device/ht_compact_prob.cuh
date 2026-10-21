/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "nccl_ep.h"
#include "common.hpp"
#include "device/macros.cuh"
#include "ht_compact_prob.h"

#include <cstddef>
#include <cstdint>

namespace nccl_ep {
namespace ht {

struct compact_prob_output_param_t {
    const float* wire_prob;
    const bool* local_expert_routing_map;
    void* wire_prob_pos;
    float* recv_topk_weights;
    int64_t* recv_topk_idx;
    int num_recv_tokens;
    const int32_t* num_recv_tokens_dev;
    int topk;
    int experts_per_rank;
    int wire_prob_floats;
    int global_expert_offset;
    ncclEpExpertIdKind_t recv_topk_idx_kind;
    bool expert_major;
    bool lerm_bitmap;
    const int64_t* expert_token_offsets;
    const int32_t* per_expert_counts_active;
    int32_t* flat_expert_ids;
};

__device__ __forceinline__ int compact_find_pos(const float* row, int topk, int global_expert) {
    // Called only after LERM establishes a real payload. Loading the ID word
    // as float is bit preserving: no FP arithmetic is applied to these bytes.
    for (int k = 0; k < topk; ++k) {
        const uint32_t word = __float_as_uint(row[topk + k / 2]);
        const uint16_t id = static_cast<uint16_t>(word >> (16 * (k & 1)));
        if (id == global_expert) {
            return k;
        }
    }
    // A selected expert missing from its token IDs is a protocol mismatch,
    // not padding: do not silently pretend its real weight was zero.
    EP_DEVICE_ASSERT(false);
    return -1;
}

// Ordering is selected by the JIT specialization.
template <typename PosT, bool SortedTopk>
__device__ __forceinline__ void compact_prob_to_output(const compact_prob_output_param_t& p) {
    const float* __restrict__ wire_prob = p.wire_prob;
    const bool* __restrict__ lerm = p.local_expert_routing_map;
    PosT* __restrict__ positions = static_cast<PosT*>(p.wire_prob_pos);
    float* __restrict__ weights = p.recv_topk_weights;
    int64_t* __restrict__ ids = p.recv_topk_idx;
    const int64_t* __restrict__ offsets = p.expert_token_offsets;
    const int32_t* __restrict__ counts = p.per_expert_counts_active;
    int32_t* __restrict__ flat_ids = p.flat_expert_ids;
    constexpr PosT invalid_pos = static_cast<PosT>(-1);
    const int lerm_words = nccl_ep::bit_words(p.experts_per_rank);
    const auto* lerm_bits = reinterpret_cast<const uint64_t*>(lerm);
    for (size_t token = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         token < static_cast<size_t>(p.num_recv_tokens);
         token += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (token >= static_cast<size_t>(*p.num_recv_tokens_dev)) {
            break;
        }
        const float* row = wire_prob + token * p.wire_prob_floats;
        int k_out = 0;
        const auto routed = [&](int e) {
            return p.lerm_bitmap ? nccl_ep::test_bit(lerm_bits + token * lerm_words, e) :
                lerm[token * p.experts_per_rank + e];
        };
        const auto emit = [&](int e, int source_k) {
            const size_t output = p.expert_major ? token : token * p.topk + k_out;
            positions[output] = static_cast<PosT>(source_k);
            weights[output] = row[source_k];
            if (!p.expert_major) {
                if (ids) {
                    ids[output] = p.recv_topk_idx_kind == NCCL_EP_EXPERT_ID_GLOBAL ? p.global_expert_offset + e : e;
                }
                if constexpr (!SortedTopk) {
                    if (flat_ids) {
                        flat_ids[output] = e;
                    }
                }
            }
            ++k_out;
        };
        if (p.expert_major) {
            positions[token] = invalid_pos;
            weights[token] = 0.0f;
        }
        if constexpr (SortedTopk) {
            // LERM guards unwritten EM padding before the inline-ID search.
            for (int e = 0; e < p.experts_per_rank && k_out < p.topk; ++e) {
                if (!routed(e)) {
                    continue;
                }
                const int source_k = compact_find_pos(row, p.topk, p.global_expert_offset + e);
                if (source_k < 0) {
                    continue; // compact_find_pos diagnosed a protocol mismatch
                }
                emit(e, source_k);
                if (p.expert_major) {
                    break;
                }
            }
        } else {
            // Native EM includes unwritten padding. Locate its expert zone before
            // loading inline IDs; FLAT and EM local-permute need no zone lookup.
            if (p.expert_major) {
                int lo = 0, hi = p.experts_per_rank;
                while (lo < hi) {
                    const int mid = lo + (hi - lo) / 2;
                    if (token >= static_cast<size_t>(offsets[mid + 1])) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if (lo == p.experts_per_rank ||
                    token >= static_cast<size_t>(offsets[lo] + counts[lo])) {
                    continue;
                }
            }
            // Pack local matches in source order, keeping IDs and weights paired.
            for (int source_k = 0; source_k < p.topk; ++source_k) {
                const uint32_t word = __float_as_uint(row[p.topk + source_k / 2]);
                const int global_expert = static_cast<uint16_t>(word >> (16 * (source_k & 1)));
                const int e = global_expert - p.global_expert_offset;
                if (global_expert == kCompactInvalidExpert || e < 0 || e >= p.experts_per_rank || !routed(e)) {
                    continue;
                }
                emit(e, source_k);
                if (p.expert_major) {
                    break;
                }
            }
            EP_DEVICE_ASSERT(k_out > 0); // A live payload must contain a selected local expert.
        }
        if (!p.expert_major) {
            for (; k_out < p.topk; ++k_out) {
                const size_t output = token * p.topk + k_out;
                positions[output] = invalid_pos;
                weights[output] = 0.0f;
                if (ids) {
                    ids[output] = -1;
                }
                if constexpr (!SortedTopk) {
                    if (flat_ids) {
                        flat_ids[output] = -1;
                    }
                }
            }
        }
    }
}

} // namespace ht
} // namespace nccl_ep
