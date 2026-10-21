/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "../device_primitives.cuh"

namespace nccl_ep {

namespace ll {

// Selects the epoch (send-buffer bank) for this launch and, on the control
// thread, advances/pairs the send/recv halves of a split-phase launch so a
// send-only call's epoch is picked up by the matching recv-only call.
__device__ __forceinline__ unsigned int selectLowLatencyEpoch(
    LowLatencyEpochState* epochState, int phases, int smId, int threadId) {
    if (phases & LOW_LATENCY_SEND_PHASE) {
        const unsigned int epoch = epochState->epoch;
        if (smId == 0 && threadId == 0) {
            EP_DEVICE_ASSERT(epochState->send_in_flight == 0);
            if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
                epochState->pending_epoch = epoch;
                epochState->send_in_flight = 1;
            }
        }
        return epoch;
    }

    const unsigned int epoch = epochState->pending_epoch;
    if (smId == 0 && threadId == 0) {
        EP_DEVICE_ASSERT(epochState->send_in_flight == 1);
        EP_DEVICE_ASSERT(epoch == epochState->epoch);
        ++epochState->epoch;
        epochState->send_in_flight = 0;
    }
    return epoch;
}

// Advances the epoch for a combined send+recv launch at the caller's
// SEND-completion boundary. The caller establishes required ordering and elects
// one updating thread; this helper does not wait for RECV work to complete.
// Shared by both overloads so the phase-mask check and update stay together.
__device__ __forceinline__ void completeFullLowLatencyEpochImpl(
    LowLatencyEpochState* epochState, int phases, bool shouldUpdate) {
    if ((phases & (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE)) ==
            (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE) &&
        shouldUpdate) {
        ++epochState->epoch;
    }
}

// Elects smId == 0, threadId == 0 to perform the update -- correct for
// callers whose SEND/RECV phase boundary is a real grid-wide convergence
// point (e.g. cg::this_grid().sync()), so every CTA (including smId == 0)
// only reaches this call once every other CTA's SEND-phase work is done.
__device__ __forceinline__ void completeFullLowLatencyEpoch(
    LowLatencyEpochState* epochState, int phases, int smId, int threadId) {
    completeFullLowLatencyEpochImpl(epochState, phases, smId == 0 && threadId == 0);
}

// For callers with no grid-wide SEND/RECV boundary: isElectedCta must be
// true only in the CTA elected after the caller establishes the required
// SEND-completion ordering (e.g. ll_dispatch_lsa.cuh's
// syncAndSendCounts's isLastCta). smId == 0 alone carries no such guarantee
// without a barrier.
__device__ __forceinline__ void completeFullLowLatencyEpoch(
    LowLatencyEpochState* epochState, int phases, bool isElectedCta, int threadId) {
    completeFullLowLatencyEpochImpl(epochState, phases, isElectedCta && threadId == 0);
}

// Enable compiler memory ordering by default; callers can explicitly opt out.
template <bool kMemoryClobber = true>
__device__ __forceinline__ void syncSmGroup(int groupIdx, int nThreads) {
    if constexpr (kMemoryClobber) {
        asm volatile("bar.sync %0, %1;" ::"r"(groupIdx), "r"(nThreads) : "memory");
    } else {
        asm volatile("bar.sync %0, %1;" ::"r"(groupIdx), "r"(nThreads));
    }
}

// Zeroes the next epoch's receive-count/flag buffer bank. Shared by every LL dispatch/combine
// variant (general and LSA); callers differ only in which/how many threads participate, hence
// the explicit offset/stride instead of a hardcoded warp.
__device__ __forceinline__ void cleanNextRecvCntBufStrided(
    int* nextRecvCntBuf, int nextRecvCntBufSize, int offset, int stride) {
#pragma unroll
    for (int i = offset; i < nextRecvCntBufSize; i += stride) {
        nextRecvCntBuf[i] = 0;
    }
}

// Record one rank-major token after the caller has established readiness and
// decoded its output slot and source-info address. All 32 lanes of one warp
// call together with the same header, slot and geometry. This helper owns no
// transport synchronization, count decoding or payload copies.
__device__ __forceinline__ void recordRankMajorRecvTokenMetadata(
    const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>* recvHdr,
    int slot,
    int numTopk,
    int numLocalExperts,
    int currRank,
    int laneId,
    int* recvSrcInfo,
    int32_t* outRecvTopkIdx,
    float* outRecvTopkWeights,
    ncclEpExpertIdKind_t recvTopkIdxKind) {
    const auto rtr = recvHdr->rtr;
    const auto recvSrcTopkInfo = recvSrcInfo + 1;
    const auto globalExpertStartIdx = currRank * numLocalExperts;
    if (laneId == 0) {
        recvSrcInfo[0] = recvHdr->token_id;
        recvSrcTopkInfo[0] = slot;
    }
    if (numTopk > 1) {
        bool matchesCurrRank = (laneId < numTopk) &&
            (getExpertRankIdx((int)rtr[laneId].expert_id, numLocalExperts) == currRank);
        uint32_t currRankMask = __ballot_sync(0xffffffff, matchesCurrRank);
        int j_eff = currRankMask ? (__ffs(currRankMask) - 1) : -1;
        if (laneId == 0) {
            recvSrcTopkInfo[1] = j_eff;
        }
    }

    EP_DEVICE_ASSERT(outRecvTopkIdx != nullptr && outRecvTopkWeights != nullptr);
    EP_DEVICE_ASSERT(recvTopkIdxKind == NCCL_EP_EXPERT_ID_LOCAL || recvTopkIdxKind == NCCL_EP_EXPERT_ID_GLOBAL);
    if (laneId < numTopk) {
        int globalExpertIdx = (int)rtr[laneId].expert_id;
        int localExpertIdx = globalExpertIdx - globalExpertStartIdx;
        bool valid = (localExpertIdx >= 0 && localExpertIdx < numLocalExperts);
        int writeIdx = (recvTopkIdxKind == NCCL_EP_EXPERT_ID_GLOBAL) ? globalExpertIdx : localExpertIdx;
        outRecvTopkIdx[slot * numTopk + laneId] = valid ? (int32_t)writeIdx : (int32_t)-1;
        outRecvTopkWeights[slot * numTopk + laneId] = rtr[laneId].topk_weight;
    }
}

// Warp-cooperative: all 32 lanes must call together.
// Each lane inspects its own topk index and matches it with others via __match_any_sync.
// If this lane is not the first occurrence of its source rank, topkIdxByLane is dropped to -1.
__device__ __forceinline__ int warpFirstTopKOccurence(int topkIdxByLane, int numLocalExperts, int laneId) {
    int rank = getExpertRankIdx(topkIdxByLane, numLocalExperts);
    uint32_t mask = __match_any_sync(0xffffffff, rank);
    bool isFirst = (laneId == (__ffs(mask) - 1));
    return topkIdxByLane * (int)isFirst - (int)(!isFirst);
}

// This function is very efficient and outperforms orignal expert counting
// one (that it is replacing) even though it is more complex.
// TopkIdxT is int32_t or int64_t. When the narrower type is used, the
// caller is responsible for ensuring expert ids do not overflow it.
template <typename TopkIdxT>
__forceinline__ __device__ void countTokensPerRank_mask(
    const TopkIdxT* inTopkIdx,
    int numTokens,
    int numTopk,
    int numLocalExperts,
    int rankBeginIdx,
    int rankEndIdx,
    int* rankCount,
    uint64_t* rankMap,
    int* sharedRankCount,
    int laneId) {
    const int batchSize = 32;
    const int rankRangeSize = rankEndIdx - rankBeginIdx;
    assert(rankRangeSize <= 8 * sizeof(uint64_t));

    // Per lane count

    int batchIdx = 0;

    for (int i = laneId; i < numTokens; i += 32) {
        // Scan all topK indices and store in rankMap
#pragma unroll 8
        for (int k = 0; k < numTopk; k++) {
            auto idx = static_cast<int>(__ldg(inTopkIdx + i * numTopk + k));
            auto rankIdx = getExpertRankIdx(idx, numLocalExperts);
            if (rankIdx < 0) {
                continue;
            }
            bool rankInRange = rankIdx >= rankBeginIdx and rankIdx < rankEndIdx;
            rankMap[batchIdx] |= rankInRange * (1 << (rankIdx - rankBeginIdx));
        }
        batchIdx++;
        if (batchIdx == batchSize) {
            for (int j = 0; j < batchSize; ++j) {
#pragma unroll 8
                for (int k = 0; k < rankRangeSize; k++) {
                    if (rankMap[j] & (1 << k)) {
                        rankCount[k]++;
                    }
                }
                // Clear the rankMap for the next batch
                rankMap[j] = 0;
            }
            batchIdx = 0;
        }
    }

    for (int j = 0; j < batchIdx; ++j) {
#pragma unroll 8
        for (int k = 0; k < rankRangeSize; k++) {
            if (rankMap[j] & (1 << k)) {
                rankCount[k]++;
            }
            // no need to clear the rankMap for the last batch
        }
    }

    // Warp reduce
#pragma unroll
    for (int i = rankBeginIdx; i < rankEndIdx; ++i) {
        auto sum = warp_reduce_sum(rankCount[i - rankBeginIdx]);
        if (laneId == 0) {
            sharedRankCount[i - rankBeginIdx] = sum;
        }
    }
}

} // namespace ll

} // namespace nccl_ep
