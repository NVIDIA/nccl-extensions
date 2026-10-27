/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

// Benchmark-only capacity policy for random, distinct-expert top-k routing.
// This is not a worst-case routing bound; the library's overflow policy still applies.
// UINT_MAX selects headroom sizing; other values, including eager zero, pass through.
// Pass zero local experts for FLAT (no expert padding), and nonzero staging_ranks
// only for legacy duplication modes whose staging requires the full fan-out.
inline std::optional<unsigned int> ep_bench_recv_budget(
    unsigned int requested, unsigned int tokens, unsigned int topk, unsigned int local_experts,
    size_t alignment, unsigned int staging_ranks) {
    if (requested != std::numeric_limits<unsigned int>::max()) {
        return requested;
    }
    constexpr uint64_t limit = std::numeric_limits<int>::max();
    const uint64_t expected = static_cast<uint64_t>(tokens) * topk;
    if (expected == 0 || expected > limit) {
        return std::nullopt;
    }

    // ceil(expected * 1.15), using integers to avoid floating-point rounding.
    uint64_t budget = expected + (expected * 3 + 19) / 20;
    if (staging_ranks != 0) {
        if (staging_ranks > limit / expected) {
            return std::nullopt;
        }
        budget = std::max(budget, expected * staging_ranks);
    }
    if (budget > limit) {
        return std::nullopt;
    }
    if (local_experts != 0 && alignment > 1) {
        const size_t slack_per_expert = alignment - 1;
        if (slack_per_expert > (limit - budget) / local_experts) {
            return std::nullopt;
        }
        budget += static_cast<uint64_t>(local_experts) * slack_per_expert;
    }
    return static_cast<unsigned int>(budget);
}
