/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nccl_ep {
namespace ll {

// Calibration provenance and limitations known and may need improvement
namespace hierarchical_dispatch_fit {
constexpr double kDomainOverheadUs = 24.990092522982042;
constexpr double kForwardUsPerCopy = 0.3344025571653127;
constexpr double kSavedRdmaUsPerCopy = 0.43792670346517215;
constexpr double kBaselineFixedUs = 35.316274260758924;
constexpr double kBaselinePeerUs = 1.9535062967130778;
constexpr double kBaselineGpuCopyUs = 0.2985860760626015;
constexpr double kBaselineSparseUs = 14.900315473823898;
constexpr double kBaselineExtraDomainCopyUs = 0.013426807355760092;
constexpr double kBaselineSparseTokenScale = 32;
constexpr double kReferenceTokenBytes = 7168 * sizeof(uint16_t);
} // namespace hierarchical_dispatch_fit

// AUTO eligibility and promotion margins are independent of the fit.
namespace hierarchical_dispatch_auto_policy {
// Larger-domain wins are unvalidated extrapolations; cap AUTO to avoid
// reserving an expensive GIN pool. Forced hierarchy remains available.
constexpr int kMaxDomains = 4;
constexpr double kMinSavingUs = 5;
constexpr double kMinSavingFraction = .02;
} // namespace hierarchical_dispatch_auto_policy

// Explicit budgets at or above the normal AUTO allocation do not restrict GIN.
inline bool llGinContextsRestricted(unsigned contexts, unsigned autoContexts) {
    return contexts < autoContexts;
}

// Uniform expert sampling without replacement; all intermediates stay in [0, 1].
inline double llDestinationHitProbability(int experts, int destinationExperts, int topk) {
    if (topk <= 0 || topk > std::numeric_limits<uint32_t>::digits ||
        experts < topk || destinationExperts <= 0 || destinationExperts > experts) {
        return 0;
    }
    double miss = 1;
    for (int entry = 0; entry < topk; ++entry) {
        const int remaining = experts - destinationExperts - entry;
        if (remaining <= 0) {
            return 1;
        }
        miss *= double(remaining) / (experts - entry);
    }
    return std::clamp(1 - miss, 0., 1.);
}

// Extrapolate copy costs with payload size and destination probabilities with actual top-k.
inline double llHierarchicalDispatchScore(int domains, int ranksPerDomain, uint64_t capacity,
                                          int experts, int topk, uint64_t tokenBytes, bool contextsRestricted) {
    using namespace hierarchical_dispatch_fit;
    using namespace hierarchical_dispatch_auto_policy;
    if (domains < 2 ||
        ranksPerDomain < 1 ||
        capacity == 0 ||
        capacity > uint64_t(std::numeric_limits<int>::max()) ||
        experts <= 0 ||
        experts > std::numeric_limits<uint16_t>::max() ||
        topk <= 0 ||
        topk > std::numeric_limits<uint32_t>::digits ||
        topk > experts ||
        tokenBytes == 0 ||
        contextsRestricted) {
        return std::numeric_limits<double>::infinity();
    }
    const int64_t rankCount = int64_t(domains) * ranksPerDomain;
    if (rankCount > experts || experts % rankCount != 0) {
        return std::numeric_limits<double>::infinity();
    }
    const int ranks = static_cast<int>(rankCount);
    const double remoteDomains = domains - 1;
    const double remotePeers = remoteDomains * ranksPerDomain;
    const double gpuCopies = remotePeers * llDestinationHitProbability(experts, experts / ranks, topk);
    const double domainCopies = remoteDomains * llDestinationHitProbability(experts, experts / domains, topk);
    const double tokens = double(capacity);
    const double payloadScale = double(tokenBytes) / kReferenceTokenBytes;
    const double baselineUs = kBaselineFixedUs + kBaselinePeerUs * remotePeers +
        payloadScale * kBaselineGpuCopyUs * tokens * gpuCopies +
        kBaselineSparseUs * std::exp(-tokens / kBaselineSparseTokenScale) +
        payloadScale * kBaselineExtraDomainCopyUs * (domains - 2) * tokens * gpuCopies;
    const double newMinusBaselineUs = kDomainOverheadUs * remoteDomains + payloadScale *
        (kForwardUsPerCopy * tokens * domainCopies -
         kSavedRdmaUsPerCopy * tokens * std::max(0., gpuCopies - domainCopies));
    return newMinusBaselineUs + std::max(kMinSavingUs, kMinSavingFraction * baselineUs);
}

inline bool llUseHierarchicalDispatch(int domains, int ranksPerDomain, uint64_t capacity,
                                       int experts, int topk, uint64_t tokenBytes, bool contextsRestricted) {
    return domains <= hierarchical_dispatch_auto_policy::kMaxDomains &&
        llHierarchicalDispatchScore(domains, ranksPerDomain, capacity, experts, topk,
                                   tokenBytes, contextsRestricted) < 0;
}

// Group resources cover any handle allowed by its top-k bound; dispatch uses the actual top-k.
inline bool llMayUseHierarchicalDispatch(int domains, int ranksPerDomain, uint64_t capacity,
                                          int experts, unsigned topkBound, uint64_t tokenBytes,
                                          bool contextsRestricted) {
    unsigned limit = std::numeric_limits<uint32_t>::digits;
    if (topkBound != 0) {
        limit = std::min(limit, topkBound);
    }
    for (unsigned topk = 1; topk <= limit; ++topk) {
        if (llUseHierarchicalDispatch(domains, ranksPerDomain, capacity, experts, int(topk),
                                     tokenBytes, contextsRestricted)) {
            return true;
        }
    }
    return false;
}

} // namespace ll
} // namespace nccl_ep
