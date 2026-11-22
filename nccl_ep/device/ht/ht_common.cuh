/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once
#include "common.hpp"
#include <cstdint>

// Routing-metadata layout shared by the scan kernels and the HT kernels.
namespace ht_common {

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

// Flat receive rows contain local experts in ascending order. One 32-bit entry
// records both the expert and its original top-k position; -1 marks the tail.
// Kept with the handle so backward combine never consults another handle's routing.
__host__ __device__ constexpr int32_t flat_route_pack(int expert, int source_pos) {
    return expert * MAX_NUM_TOPK + source_pos;
}
__host__ __device__ constexpr int flat_route_expert(int32_t route) {
    return route / MAX_NUM_TOPK;
}
__host__ __device__ constexpr int flat_route_source_pos(int32_t route) {
    return route % MAX_NUM_TOPK;
}

} // namespace ht_common
