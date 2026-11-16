/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

// Per-recipe compile-time policy: geometry (block size, wire bytes, scale-row bytes) and,
// for NONE, load_pair<Dt>() (decode element pair to float2). Kernels template on the recipe
// enum and call these traits; no per-recipe branches in hot loops, SMEM layout, or G2S transport.
// packed_bytes = intra-node NVLink / G2S row (MXFP8: FP8+E8M0). rdma_bytes = inter-node hop,
// defined only for recipes that have one -- MXFP8 is NVLink-only and rejected for multi-LSA-team
// groups at the API, so it deliberately has no rdma_bytes/rdma_scale_bytes and an attempt to
// instantiate the RDMA warps with it will not compile. To add a recipe: add a specialization
// here + API enum + one host validation row.

#pragma once

#include <cstdint>

#include "nccl_ep.h" // ncclEp*QuantizationRecipe_t, ncclDataType_t
#include "device_primitives.cuh" // ld_token_pair
#include "mxfp8_quant.cuh" // kept for consumers that get mxfp8:: helpers via this header

namespace nccl_ep {

// Primary template undefined: unhandled recipe → compile error.
template <ncclEpCombQuant_t kRecipe>
struct combine_recipe_traits;

// NONE — tokens are the wire dtype; decode is a plain widening load, no scales.
template <>
struct combine_recipe_traits<NCCL_EP_COMB_QUANT_NONE> {
    static constexpr int kScaleBlock = 0;
    static constexpr int kWireBytesPerElem = 0; // 0 => derive from kTokenDtype
    __host__ __device__ static constexpr int scale_bytes_per_token(int /*hidden*/) { return 0; }

    template <ncclDataType_t kTokenDtype>
    __host__ __device__ static constexpr int wire_token_bytes(int hidden) {
        return hidden * nccl_ep::size_u8<kTokenDtype>();
    }

    template <ncclDataType_t kTokenDtype>
    __host__ __device__ static constexpr int packed_bytes(int hidden) {
        return wire_token_bytes<kTokenDtype>(hidden);
    }

    // Inter-node RDMA hop is the same dtype-width row as the intra-node packed wire.
    template <ncclDataType_t kTokenDtype>
    __host__ __device__ static constexpr int rdma_bytes(int hidden) {
        return packed_bytes<kTokenDtype>(hidden);
    }
    template <ncclDataType_t /*kTokenDtype*/>
    __host__ __device__ static constexpr int rdma_scale_bytes(int /*hidden*/) {
        return 0;
    }

    template <ncclDataType_t kTokenDtype>
    __device__ static __forceinline__ float2 load_pair(const void* token, int idx) {
        return nccl_ep::ld_token_pair<kTokenDtype>(token, idx);
    }
};

// MXFP8 — FP8 E4M3 data + E8M0 block scales (block=32); decode = fp8*2^(E-127) -> float2.
template <>
struct combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8> {
    static constexpr int kScaleBlock = NCCL_EP_MXFP8_SCALE_BLOCK;
    static constexpr int kWireBytesPerElem = 1; // fp8
    __host__ __device__ static constexpr int scale_bytes_per_token(int hidden) { return hidden / kScaleBlock; }

    template <ncclDataType_t /*kTokenDtype unused: wire type is always fp8*/>
    __host__ __device__ static constexpr int wire_token_bytes(int hidden) { return hidden * kWireBytesPerElem; }

    // Intra-node NVLink / G2S packed row: [FP8 H bytes | E8M0 H/32 bytes].
    template <ncclDataType_t kTokenDtype>
    __host__ __device__ static constexpr int packed_bytes(int hidden) {
        return wire_token_bytes<kTokenDtype>(hidden) + scale_bytes_per_token(hidden);
    }

    // No load_pair: the MXFP8 RED loops decode explicitly via mxfp8_red_map + ld_fp8_pairs_shared
    // so one 16B load shares a single E8M0 read. A per-pair load_pair would defeat that.
};

// dispatch_recipe_traits<R> follows the same pattern for the dispatch direction (geometry only, no load_pair).

// Host-callable: scale row bytes for a given recipe and hidden dim.
// Returns 0 for NONE. Centralises the block-size lookup so callers don't hardcode it.
inline int combine_recipe_scale_row_bytes(ncclEpCombQuant_t recipe, int hidden) {
    switch (recipe) {
        case NCCL_EP_COMB_QUANT_MXFP8:
            return hidden / combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::kScaleBlock;
        default:
            return 0;
    }
}

// Host-callable: bytes of the token row the combine reduce kernel *writes*, given the hidden
// dim and the bytes of the row it *reads*. NONE writes back the row it read; MXFP8 writes the
// packed [FP8 H | E8M0 H/32] row. Callers hold both hidden and the input width, so the packed
// geometry is only ever derived forwards -- no launcher has to invert it.
inline int combine_recipe_packed_row_bytes(ncclEpCombQuant_t recipe, int hidden, int input_row_bytes) {
    switch (recipe) {
        case NCCL_EP_COMB_QUANT_MXFP8:
            return combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::template packed_bytes<ncclBfloat16>(hidden);
        default:
            return input_row_bytes;
    }
}

} // namespace nccl_ep
