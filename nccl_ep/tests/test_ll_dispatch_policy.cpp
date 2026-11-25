/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ll_dispatch_policy.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace nccl_ep::ll;

static void check(bool condition) {
    if (!condition) {
        std::fprintf(stderr, "LL dispatch policy check failed\n");
        std::abort();
    }
}

int main() {
    constexpr uint64_t referenceBytes = 7168 * sizeof(uint16_t);
    // Preserve the calibrated K8 choices.
    check(!llUseHierarchicalDispatch(2, 8, 32, 256, 8, referenceBytes, false));
    check(llUseHierarchicalDispatch(2, 8, 64, 256, 8, referenceBytes, false));
    check(!llUseHierarchicalDispatch(4, 8, 256, 512, 8, referenceBytes, false));
    check(llUseHierarchicalDispatch(4, 8, 512, 512, 8, referenceBytes, false));
    check(llUseHierarchicalDispatch(4, 8, 8192, 256, 8, referenceBytes, false));
    check(!llMayUseHierarchicalDispatch(16, 4, 128, 256, 8, referenceBytes, false));

    // AUTO must not promote outside the provisioning limit, even where the
    // unconstrained fit predicts a win at high top-k or token capacity.
    check(llHierarchicalDispatchScore(5, 4, 128, 80, 15, referenceBytes, false) < 0);
    check(llHierarchicalDispatchScore(8, 4, 128, 128, 23, referenceBytes, false) < 0);
    check(llHierarchicalDispatchScore(16, 4, 8192, 256, 32, referenceBytes, false) < 0);
    for (int domains : {5, 8, 16, 32}) {
        const int experts = domains * 4 * 4;
        for (int topk = 1; topk <= 32; ++topk) {
            check(!llUseHierarchicalDispatch(domains, 4, 8192, experts, topk, referenceBytes, false));
            check(!llMayUseHierarchicalDispatch(domains, 4, 8192, experts, topk, referenceBytes, false));
        }
        check(!llMayUseHierarchicalDispatch(domains, 4, 8192, experts, 0, referenceBytes, false));
    }

    for (int domains = 2; domains <= 4; ++domains) {
        for (int ranks = 4; ranks <= 8; ++ranks) {
            for (int localExperts = 8; localExperts <= 16; ++localExperts) {
                const int experts = domains * ranks * localExperts;
                double previous = std::numeric_limits<double>::infinity();
                bool promoted = false;
                for (uint64_t capacity = 1; capacity <= 8192; ++capacity) {
                    const double score = llHierarchicalDispatchScore(
                        domains, ranks, capacity, experts, 8, referenceBytes, false);
                    check(std::isfinite(score));
                    check(score < previous);
                    check(!promoted || score < 0);
                    promoted = score < 0;
                    previous = score;
                }
            }
        }
    }

    // Extrapolation remains finite outside the original topology, size, and expert bounds.
    for (int domains : {2, 3, 4, 8, 16}) {
        for (int ranks : {1, 4, 8, 72}) {
            for (int localExperts : {1, 8, 16}) {
                const int experts = domains * ranks * localExperts;
                if (experts > std::numeric_limits<uint16_t>::max()) {
                    continue;
                }
                for (int topk : {1, 2, 6, 8, 16, 32}) {
                    if (topk > experts) {
                        continue;
                    }
                    for (uint64_t capacity : {uint64_t(1), uint64_t(32), uint64_t(256), uint64_t(512),
                                               uint64_t(8192), uint64_t(65536), uint64_t(std::numeric_limits<int>::max())}) {
                        for (uint64_t bytes : {uint64_t(16), uint64_t(4096), referenceBytes, uint64_t(32768)}) {
                            check(std::isfinite(llHierarchicalDispatchScore(
                                domains, ranks, capacity, experts, topk, bytes, false)));
                            check(!llUseHierarchicalDispatch(domains, ranks, capacity, experts, topk, bytes, true));
                        }
                    }
                }
            }
        }
    }

    // Group reservation and dispatch agree at and beyond the domain boundary.
    for (int domains : {1, 2, 3, 4, 5, 8, 16}) {
        for (unsigned bound : {0u, 1u, 6u, 8u, 32u, std::numeric_limits<unsigned>::max()}) {
            const int experts = domains * 8 * 16;
            const bool reserved = llMayUseHierarchicalDispatch(domains, 8, 512, experts, bound, referenceBytes, false);
            const unsigned limit = bound == 0 ? 32u : std::min(bound, 32u);
            bool any = false;
            for (unsigned topk = 1; topk <= limit; ++topk) {
                any |= llUseHierarchicalDispatch(domains, 8, 512, experts, int(topk), referenceBytes, false);
            }
            check(reserved == any);
        }
    }
    check(!llUseHierarchicalDispatch(2, 8, std::numeric_limits<int>::max(), 256, 1, referenceBytes, false));
    check(llUseHierarchicalDispatch(2, 8, 512, 256, 6, referenceBytes, false));
    check(llUseHierarchicalDispatch(2, 8, 16384, 256, 8, referenceBytes, false));
    for (unsigned normal : {129u, 297u, 513u}) {
        check(llGinContextsRestricted(normal - 1, normal));
        check(!llGinContextsRestricted(normal, normal));
        check(!llGinContextsRestricted(normal + 1, normal));
        check(!llGinContextsRestricted(std::numeric_limits<unsigned>::max(), normal));
    }

    for (int domains : {std::numeric_limits<int>::min(), -1, 0, 1, std::numeric_limits<int>::max()}) {
        for (int ranks : {std::numeric_limits<int>::min(), 0, 1, 8, std::numeric_limits<int>::max()}) {
            check(!llUseHierarchicalDispatch(domains, ranks, 512, 512, 8, referenceBytes, false));
        }
    }
    for (uint64_t capacity : {uint64_t(0), uint64_t(std::numeric_limits<int>::max()) + 1,
                              std::numeric_limits<uint64_t>::max()}) {
        check(!llUseHierarchicalDispatch(2, 8, capacity, 256, 8, referenceBytes, false));
    }
    for (int experts : {std::numeric_limits<int>::min(), -1, 0, 255, 257, std::numeric_limits<int>::max()}) {
        check(!llUseHierarchicalDispatch(2, 8, 512, experts, 8, referenceBytes, false));
    }
    for (int topk : {std::numeric_limits<int>::min(), -1, 0, 33, std::numeric_limits<int>::max()}) {
        check(!llUseHierarchicalDispatch(2, 8, 512, 256, topk, referenceBytes, false));
    }
    check(!llUseHierarchicalDispatch(2, 8, 512, 256, 8, 0, false));
    check(llDestinationHitProbability(16, 8, 1) == .5);
    check(std::abs(llDestinationHitProbability(16, 8, 2) - (1. - 28. / 120.)) < 1e-12);
    check(llDestinationHitProbability(8, 8, 8) == 1);
    check(llDestinationHitProbability(8, 0, 8) == 0);
    check(llDestinationHitProbability(7, 1, 8) == 0);
    check(llDestinationHitProbability(std::numeric_limits<int>::max(), 1, 8) > 0);
    check(llDestinationHitProbability(std::numeric_limits<int>::min(), 1, 8) == 0);
    std::puts("LL dispatch policy checks passed");
}
