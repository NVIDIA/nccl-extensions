/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ep_bench_recv_budget.hpp"

#include <cstdlib>
#include <iostream>

static void check(bool condition) {
    if (!condition) {
        std::cerr << "ep_bench receive-budget check failed\n";
        std::exit(1);
    }
}

int main() {
    constexpr unsigned int automatic = std::numeric_limits<unsigned int>::max();
    // Explicit capacity, including eager zero, bypasses headroom and padding.
    check(ep_bench_recv_budget(1408, 128, 8, 32, 128, 8) == 1408u);
    check(ep_bench_recv_budget(0, 128, 8, 32, 128, 8) == 0u);
    // Both layouts use the same budget without padding. Round fractional slots up.
    for (unsigned int experts : {0u, 32u}) {
        check(ep_bench_recv_budget(automatic, 1, 1, experts, 0, 0) == 2u);
        check(ep_bench_recv_budget(automatic, 20, 1, experts, 1, 0) == 23u);
        check(ep_bench_recv_budget(automatic, 128, 8, experts, 1, 0) == 1178u);
        check(ep_bench_recv_budget(automatic, 4096, 8, experts, 1, 0) == 37684u);
    }
    // One slack allocation per local expert; FLAT ignores alignment.
    check(ep_bench_recv_budget(automatic, 128, 8, 32, 128, 0) == 5242u);
    check(ep_bench_recv_budget(automatic, 128, 8, 0, 128, 0) == 1178u);
    // Preserve the legacy staged-duplication bound, including padded EM zones.
    check(ep_bench_recv_budget(automatic, 4096, 8, 32, 1, 8) == 262144u);
    check(ep_bench_recv_budget(automatic, 4096, 8, 32, 128, 8) == 266208u);

    // All internal row indices must fit a signed int. Reject overflow in each term.
    constexpr unsigned int int_max = std::numeric_limits<int>::max();
    constexpr unsigned int uint_max = std::numeric_limits<unsigned int>::max();
    check(ep_bench_recv_budget(automatic, 1867377084u, 1, 0, 1, 0) == int_max);
    check(!ep_bench_recv_budget(automatic, 1867377085u, 1, 0, 1, 0));
    check(!ep_bench_recv_budget(automatic, int_max, 2, 0, 1, 0));
    check(!ep_bench_recv_budget(automatic, uint_max, uint_max, 0, 1, 0));
    check(!ep_bench_recv_budget(automatic, 128, 8, 32, std::numeric_limits<size_t>::max(), 0));
    check(!ep_bench_recv_budget(automatic, 128, 8, 0, 1, uint_max));
    check(!ep_bench_recv_budget(automatic, 0, 8, 0, 1, 0));
    check(!ep_bench_recv_budget(automatic, 128, 0, 0, 1, 0));
    std::cout << "ep_bench receive-budget checks passed\n";
}
