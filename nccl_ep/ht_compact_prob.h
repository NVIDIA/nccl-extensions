/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <limits>

namespace nccl_ep {
namespace ht {

// FWD-only opaque payload. Weight and ID in column k belong to
// the same caller top-k slot; neither field is sorted. The dispatch transport
// copies these bytes, it must never apply FP arithmetic to the ID tail.
//
//   [K float32 weights][K uint16 global expert IDs][zero tail to 16 bytes]
//
// The complete row is 16-byte aligned, not each field. In particular K=10
// occupies 64 bytes (40 weights + 20 IDs + 4 padding), not 80 bytes.
constexpr uint16_t kCompactInvalidExpert = 0xffffu;
constexpr int kCompactMaxExperts = 65535; // valid IDs are 0..65534

constexpr int dispatch_compact_prob_row_floats(int topk) {
    const int64_t words = ((int64_t{6} * topk + 15) / 16) * 4;
    return topk < 0 || words > std::numeric_limits<int>::max() ? -1 : static_cast<int>(words);
}

constexpr int64_t compact_prob_id_byte_offset(int topk) {
    return static_cast<int64_t>(topk) * sizeof(float);
}

// Original top-k position saved per returned weight for backward combine.
// Reserve the largest value for padding: K=255 uses positions 0..254 plus
// sentinel255; K=256 needs uint16. Compact's expert-ID limit also bounds K.
constexpr int compact_prob_pos_bytes(int topk) {
    return topk <= 255 ? sizeof(uint8_t) : sizeof(uint16_t);
}

// Dense allocation is retained. Activate only when the full opaque payload,
// not merely its weight prefix, is strictly smaller than that dense row.
// Mode compatibility (for example EM pull-push) is checked by the caller.
constexpr bool compact_prob_fits(int topk, int num_experts, int dense_row_floats) {
    const int words = dispatch_compact_prob_row_floats(topk);
    return topk > 0 && num_experts > 0 && num_experts <= kCompactMaxExperts &&
           words > 0 && words < dense_row_floats;
}

static_assert(sizeof(float) == 4, "compact wire requires 32-bit floats");
static_assert(dispatch_compact_prob_row_floats(8) == 12, "K8 occupies 48 bytes");
static_assert(dispatch_compact_prob_row_floats(10) == 16, "K10 occupies 64 bytes");
static_assert(compact_prob_pos_bytes(255) == 1, "K255 map uses uint8");
static_assert(compact_prob_pos_bytes(256) == 2, "K256 map uses uint16");

} // namespace ht
} // namespace nccl_ep
