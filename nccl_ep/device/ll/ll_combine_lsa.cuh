/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cooperative_groups.h>
#include "../device_primitives.cuh"
#include "nccl_device.h"
#include "../ll_ep_adapter.cuh"
#include "ll_ep_smem.cuh"
#include "ll_common.cuh"
#include "ll_lsa_primitives.cuh"
#include "ll_mask.cuh"
#include "../lamport.cuh"

namespace cg = cooperative_groups;

// LSA-only, unquantized, rank-major-only LL combine. Every destination is
// assumed NVLink-reachable (same LSA team) -- there is no RDMA/GIN fallback
// anywhere in this file, unlike the general combine_kernel_impl in ll_ep.cuh,
// which resolves each peer's P2P pointer at runtime and falls back to GIN
// whenever it comes back null. This is a faithful, behavior-preserving
// extraction/specialization of that general kernel, meant as a clean
// prototyping base, following the same pattern already used for dispatch
// (ll/ll_dispatch_lsa.cuh).
//
// Scope cuts relative to combine_kernel_impl:
//   - kRecipe is fixed to NCCL_EP_COMB_QUANT_NONE: no NVFP4 support at all
//     (no CombineRecipeTraits<NVFP4>/CombineRecipeSendOps<NVFP4>, no
//     nvfp4_* packing helpers, no decodeAndAccumulateNvfp4, no
//     __CUDA_ARCH_FAMILY_SPECIFIC__ NVFP4-unsupported-device guard -- that
//     guard exists purely because of NVFP4, so it goes with it).
//   - kUseLogFMT is fixed to false: no LogFMT encode/decode anywhere. The
//     NONE-recipe wire slot size (numBytesPerSlot) intentionally keeps the
//     same reserved LogFMT-metadata tail bytes the general kernel's NONE
//     recipe always carries (that reservation does not depend on kUseLogFMT
//     even in the general kernel), so the wire format stays byte-compatible;
//     this file simply never writes or reads that tail. Shared-memory
//     regions that exist purely to stage LogFMT metadata (per-division
//     logAmax/logAmin/castInfo buffers) are dropped, since nothing populates
//     or consumes them once LogFMT is gone -- that's a smem-only footprint
//     reduction, not a wire-format change.
//   - kLayout is fixed to NCCL_EP_LAYOUT_RANK_MAJOR: no EXPERT_MAJOR support
///    (matches the dispatch LSA extraction's own layout scope cut). The
//     EXPERT_MAJOR-specific per-topk send loop and its weight lookup are
//     dropped entirely; rank-major's "one send per received slot" loop and
//     its caller-applies-the-weight convention (topkWeight is always 1.0f
//     here) are the only path.
//   - Every P2P pointer this file resolves is asserted nonzero
//     (EP_DEVICE_ASSERT) instead of falling back to a GIN put/signal-wait --
//     sendRankFinishFlagLsa/waitForAllRanksFinishLsa/processAndSendTokenLsa
//     have no RDMA branch, and sendTokenViaRdma has no LSA counterpart at all.
//   - `layoutRange` and the runtime `hidden` parameter are dropped from the
//     signature: neither is referenced anywhere in the general kernel's body
//     either, so there is nothing to extract.
//   - TMA/shared-memory pipelining (combine_smem, PatternVisitor, TMA
//     buffer/barrier setup) is kept as-is: it's a data-movement optimization
//     orthogonal to quantization/RDMA/layout, not something this scope cut
//     touches.
//
// `zeroCopy` (whether the send-phase TMA copy source is the local
// send-staging slot vs `inData` directly) is unrelated to any of the above
// and stays a runtime toggle, same as in the general kernel.

namespace nccl_ep {

namespace ll {

// kMaxLsaCombineRanks/kActiveRankMaskWords/kLsaStaticSmemBytes live in
// combine_smem (ll_ep_smem.cuh) so this file and choose_combine_smem_config
// (ll_ep_adapter.cuh) share one definition.

// Rank-scoped finish-flag publish, used by combine_kernel_impl_2sided_rm_lsa's
// single-shot send completion: writes into `dstRank`'s recvFlagBuf at index
// `currRank` (the sender's own identity) -- mirrors sendPeerTokenCountLsa in
// ll_dispatch_lsa.cuh exactly (send keyed by the sender's rank, receive
// polls by source rank), adapted to a plain ready flag instead of an encoded
// count, since combine's receiver already knows how many tokens to expect
// from topk info -- there is nothing to count, only "am I ready" to signal.
__forceinline__ __device__ void sendRankFinishFlagLsa(
    int dstRank,
    int currRank,
    int* recvFlagBufLocal,
    size_t recvFlagOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {
    const uint64_t recvFlagPtr = reinterpret_cast<uint64_t>(recvFlagBufLocal + currRank);
    const size_t recvFlagOffset = recvFlagOff + static_cast<size_t>(currRank) * sizeof(int);
    const auto dstP2pPtr = ncclGetP2pPtr(recvFlagPtr, recvFlagOffset, currRank, dstRank, windows, devComm);
    if (not isRankMasked(rankMask, dstRank)) {
        EP_DEVICE_ASSERT(dstP2pPtr != 0);
        st_relaxed_sys_global(reinterpret_cast<int*>(dstP2pPtr), 1);
    }
}

// A peer timed out: mask it (a trap without a rank mask, since nothing could skip it) and raise the async error.
__forceinline__ __device__ void maskTimedOutPeer(int* rankMask, int peer, int* asyncErrorFlag) {
    if (rankMask == nullptr) {
        trap();
    }
    atomicExch(rankMask + peer, 0);
    if (asyncErrorFlag != nullptr) {
        atomicExch_system(asyncErrorFlag, 1);
    }
}

// Rank-centric recv-side readiness wait, mirroring ll_dispatch_lsa.cuh's
// syncAndRecvCounts closely: up to numRanks threads each independently
// confirm one active peer's finish flag has landed in parallel (relaxed
// poll, one shared timeout deadline across the loop, same as dispatch) --
// peers' arrival times are independent, so checking them on separate
// threads overlaps their tail latencies instead of stacking them the way a
// single serial thread would. Each participating thread acquire-fences its
// own now-confirmed read(s) (and, transitively, that peer's payload writes
// ordered before it by the peer's own release fence); __syncthreads()
// propagates that visibility to the rest of the CTA, including threads that
// checked a different peer or none at all -- same reliance on
// __syncthreads() propagating another thread's fence as the original
// single-thread version had. No grid-wide barrier needed for cross-rank
// readiness -- every CTA confirms independently. Must be called by every
// thread of the CTA.
//
// Each CTA observes remote signals independently. To avoid cross-CTA race
// condition or performing expensive synchronization, each CTA builds a local
// rank mask based on the existing global mask and uses it during data
// processing.
__forceinline__ __device__ void waitForAllRanksFinishLsa(
    int threadId,
    int numRanks,
    int* rankMask,
    uint32_t* activeRankMask,
    const int* recvFlagBuf,
    uint64_t timeoutCycles,
    int currRank,
    int* asyncErrorFlag,
    int64_t* waitStats) {
    EP_DEVICE_ASSERT(numRanks <= combine_smem::kMaxLsaCombineRanks);
#pragma unroll
    for (int w = threadId; w < combine_smem::kActiveRankMaskWords; w += blockDim.x) {
        activeRankMask[w] = 0;
    }
    // Every thread must see the zero-init before any atomicOr below.
    __syncthreads();

    const auto startTime = clock64();
    bool didWork = false;
    for (int peer = threadId; peer < numRanks; peer += blockDim.x) {
        didWork = true;
        if (isRankMasked(rankMask, peer)) continue;
        uint64_t waitCost = 0;
        bool flagReady;
        do {
            flagReady = ld_relaxed_sys_global(reinterpret_cast<const uint32_t*>(recvFlagBuf + peer)) != 0;
            waitCost = clock64() - startTime;
        } while (!flagReady && waitCost <= timeoutCycles);
        if (waitCost > timeoutCycles) {
            printf("Warning: NCCL EP timeout for combine receive, rank %d, src_rank %d\n", currRank, peer);
            maskTimedOutPeer(rankMask, peer, asyncErrorFlag);
        } else {
            // Confirmed ready by this CTA's own poll.
            atomicOr(&activeRankMask[peer / 32], 1u << (peer % 32));
        }
        if (waitStats != nullptr) {
            atomicAdd(reinterpret_cast<unsigned long long*>(waitStats + peer), waitCost);
        }
    }
    if (didWork) {
        // Acquire-fence this thread's now-confirmed reads (and, transitively,
        // the peer's payload writes ordered before it by the peer's own
        // release fence) before any thread in this CTA touches recvBuf. The
        // relaxed flag read followed by this fence is an acquire pattern
        // pairing with the peer's fence.release.sys + relaxed store; no
        // release half is needed because no later signal depends on this
        // thread's earlier writes.
        memory_fence_acquire_sys();
    }
    __syncthreads();
}

// Unquantized (no LogFMT) recv-side decode/accumulate: casts the raw
// BF16/FP16/FP32 wire bytes and accumulates into `accum`, weighted.
template <int kNumRecvUnrolls, ncclDataType_t kTokenDtype>
__forceinline__ __device__ void decodeAndAccumulateLsa(
    uint32_t* ldBuffer,
    float* accum,
    const float& weight) {
    if constexpr (kTokenDtype == ncclFloat32) {
#pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            accum[k] += *reinterpret_cast<float*>(ldBuffer + k) * weight;
        }
    } else if constexpr (kTokenDtype == ncclFloat16) {
#pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            auto fp16Pack = *reinterpret_cast<__half2*>(ldBuffer + k);
            accum[k * 2 + 0] += static_cast<float>(fp16Pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(fp16Pack.y) * weight;
        }
    } else {
#pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            auto bf16Pack = *reinterpret_cast<__nv_bfloat162*>(ldBuffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16Pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16Pack.y) * weight;
        }
    }
}

// Inverse of decodeAndAccumulateLsa: packs the fp32 accumulators back to the BF16/FP16/FP32 wire words.
template <int kNumRecvUnrolls, ncclDataType_t kTokenDtype>
__forceinline__ __device__ void encodeCombinedLsa(uint32_t* stBuffer, const float* combinedData) {
#pragma unroll
    for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
        uint32_t packed;
        if constexpr (kTokenDtype == ncclFloat32) {
            packed = *reinterpret_cast<const uint32_t*>(&combinedData[k]);
        } else if constexpr (kTokenDtype == ncclFloat16) {
            auto fp16Pack =
                __half2(__float2half(combinedData[k * 2]), __float2half(combinedData[k * 2 + 1]));
            packed = *reinterpret_cast<uint32_t*>(&fp16Pack);
        } else {
            auto bf16Pack = __nv_bfloat162(combinedData[k * 2], combinedData[k * 2 + 1]);
            packed = *reinterpret_cast<uint32_t*>(&bf16Pack);
        }
        stBuffer[k] = packed;
    }
}

// Lamport combine round words: roundWords[p] in this rank's memory is the number of combine rounds peer p has started
// (p writes round + 1 during its round, after its sends). This rank may write round r into p's segment r % 2 only once
// roundWords[p] >= r, i.e. p has started round r - 1 and therefore finished round r - 2, the previous user of that
// segment. See docs/plc/401-ll-combine-lamport.

// Publish "this rank is in `round`" (so its previous round has retired) to every active peer: one relaxed store per
// peer, spread over the grid starting at the highest CTA (peer = (numSms - 1 - smId) + numSms * threadId): the lowest
// CTAs hold the receive vectors of small batches, and their warps must not be delayed by the peer-pointer lookup. Called
// after the sends.
__forceinline__ __device__ void publishLamportRoundLsa(
    int smId,
    int numSms,
    int threadId,
    int numRanks,
    int currRank,
    unsigned int round,
    int* roundWords,
    size_t roundWordsOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {
    const int peer = (numSms - 1 - smId) + numSms * threadId;
    if (peer >= numRanks || isRankMasked(rankMask, peer)) {
        return;
    }
    const uint64_t localPtr = reinterpret_cast<uint64_t>(roundWords + currRank);
    const size_t localOff = roundWordsOff + static_cast<size_t>(currRank) * sizeof(int);
    const uint64_t dstPtr = ncclGetP2pPtr(localPtr, localOff, currRank, peer, windows, devComm);
    st_relaxed_sys_global(reinterpret_cast<int*>(dstPtr), static_cast<int>(round + 1));
}

// Wait until dstRank has started round `round - 1` (roundWords[dstRank] >= round, wrap-safe: lamport::round_reached)
// before the first store into its segment: one warp-coalesced load in the common case. The word is read only here, by
// the sending warps: a read by every thread at kernel entry made the single line a hot spot that cost about a
// microsecond. Timeouts are handled
// like a missing peer in the receive: the destination is masked, the error flag raised and false returned, and the
// caller must not store into its segment (the peer may still be consuming round `round - 2` there). Warp-collective
// (all lanes read the same word); the result is warp-uniform.
__forceinline__ __device__ bool waitLamportRoundLsa(
    const int* roundWord,
    unsigned int round,
    int dstRank,
    int currRank,
    int* rankMask,
    uint64_t timeoutCycles,
    int* asyncErrorFlag,
    int* shTimeoutReports) {
    const auto startTime = clock64();
    bool ready = false;
    bool expired = false;
    do {
        ready = __all_sync(
            0xffffffffu, lamport::round_reached(static_cast<unsigned int>(ld_volatile_global(roundWord)), round));
        expired = !ready && __any_sync(0xffffffffu, static_cast<uint64_t>(clock64() - startTime) > timeoutCycles);
    } while (!ready && !expired);
    if (!ready) {
        if (atomicAdd(shTimeoutReports, 1) == 0) {
            printf("Warning: NCCL EP timeout for Lamport combine round of peer %d, rank %d\n", dstRank, currRank);
        }
        maskTimedOutPeer(rankMask, dstRank, asyncErrorFlag);
    }
    return ready;
}

// Pending round check of a sending warp group (Lamport combine): consumed by processAndSendTokenLsa after the first
// stage load has been issued and before the first bulk store, so that the word's latency overlaps the load. `word`
// is nullptr when nothing is to be checked (2-sided combine, masked destination, or already done).
struct LamportRoundCheck {
    const int* word;
    unsigned int round;
    int dstRank;
    uint64_t timeoutCycles;
    int* asyncErrorFlag;
    int* shTimeoutReports;
    // Set when the wait timed out: the destination may still be consuming the segment's previous round, so this warp
    // issues no store to it for the rest of the kernel. The struct outlives the tokens of the warp group, which is
    // what carries the decision from the first token to the later ones.
    bool closed;
};

// Runs the pending round check once (warp-collective) and records a timeout in `closed`. Returns whether this token may
// store into the destination.
__forceinline__ __device__ bool consumeLamportRoundCheck(LamportRoundCheck& roundCheck, int currRank, int* rankMask) {
    if (roundCheck.word == nullptr) {
        return true;
    }
    const bool dstOpen = waitLamportRoundLsa(
        roundCheck.word, roundCheck.round, roundCheck.dstRank, currRank, rankMask, roundCheck.timeoutCycles,
        roundCheck.asyncErrorFlag, roundCheck.shTimeoutReports);
    roundCheck.word = nullptr;
    roundCheck.closed = !dstOpen;
    __syncwarp();
    return dstOpen;
}

// Send-side hooks of processAndSendTokenLsa: what the Lamport combine adds to the 2-sided send, and what the plain
// 2-sided send leaves out. Both are consumed through the same three calls.
struct PlainSendHooks {
    // The destination was closed by an earlier token (never, here).
    __device__ constexpr bool closed() const {
        return false;
    }
    // Runs once per warp group, after the first stage load is issued; whether this token may store.
    __device__ constexpr bool openDestination(int, int*) {
        return true;
    }
    // Rewrites the stage between its TMA load and the bulk store.
    template <int kNumSendUnrolls>
    __device__ void encodeStage(uint4*, int) const {}
};

// The Lamport combine: the destination's round check, and the sentinel rewrite of each stage in shared memory between
// the TMA load and the bulk store, overlapped with the in-flight load of the next stage and the store of the previous
// one. The round check is consumed once, after the first stage load is issued and before the first bulk store.
template <uint32_t kSanitizedWord>
struct LamportSendHooks {
    LamportRoundCheck roundCheck;

    __device__ bool closed() const {
        return roundCheck.closed;
    }
    __device__ bool openDestination(int currRank, int* rankMask) {
        return consumeLamportRoundCheck(roundCheck, currRank, rankMask);
    }
    template <int kNumSendUnrolls>
    __device__ void encodeStage(uint4* stage, int laneId) const {
        lamport::sanitize_stage<kNumSendUnrolls, kSanitizedWord>(stage, laneId);
        tma_store_fence();
        __syncwarp();
    }
};

// LSA-only counterpart of processAndSendToken: copies one combined token's
// payload via TMA from `srcDataInt4Ptr` (or the local send-staging slot,
// when `zeroCopy` is set) directly into the peer's already-resolved,
// already-nonzero P2P destination -- unquantized, non-LogFMT BF16-family
// wire format only, so there is no smem quantization/packing step and no
// RDMA fallback (dstP2pPtr is asserted nonzero, never checked against 0).
// `hooks` (PlainSendHooks or LamportSendHooks) carry what differs between the plain 2-sided send and the Lamport one.
template <
    int kNumSendUnrolls,
    int kNumStages,
    int kNumPrefetch,
    typename SendHooksT,
    typename TmaBuffersT,
    typename FullBarriersT,
    typename TmaLoadAndArriveT>
__forceinline__ __device__ void processAndSendTokenLsa(
    const int4* srcDataInt4Ptr,
    int4* sendBufInt4Ptr,
    uint64_t dstP2pPtr,
    int hiddenBf16Int4,
    bool zeroCopy,
    TmaBuffersT tmaBuffers,
    FullBarriersT fullBarriers,
    uint32_t& tmaPhase,
    TmaLoadAndArriveT& tmaLoadAndArrive,
    SendHooksT& hooks,
    int currRank,
    int* rankMask,
    int laneId) {
    EP_DEVICE_ASSERT(dstP2pPtr != 0);
    // A destination closed by an earlier token: nothing is loaded or stored (warp-uniform).
    if (hooks.closed()) {
        return;
    }
    const int kNumIters = hiddenBf16Int4 / (32 * kNumSendUnrolls);
    constexpr int kNumTmaBytes = 32 * kNumSendUnrolls * sizeof(int4);

    const auto copySrcPtr = zeroCopy ? sendBufInt4Ptr : srcDataInt4Ptr;
    const auto copyDstPtr = reinterpret_cast<int4*>(dstP2pPtr);

    // Prefetch
    if (elect_one_sync()) {
        tmaLoadAndArrive(0, copySrcPtr, kNumTmaBytes);
    }
    __syncwarp();
    // False once the destination's round check has timed out: its segment may still be in use by round `round - 2`, so
    // no store below may be issued for this token (its loads and barriers still run, which keeps the pipeline state
    // consistent). Later tokens of the warp see hooks.closed() and return before issuing anything. In the shadow of the
    // first stage load: no store below may be issued before the destination has started the previous round.
    const bool dstOpen = hooks.openDestination(currRank, rankMask);

#pragma unroll
    for (int i = laneId * kNumSendUnrolls, iterIdx = 0; i < hiddenBf16Int4; i += 32 * kNumSendUnrolls, ++iterIdx) {
        // Load the next iteration
        const int& stageIdx = iterIdx % kNumStages;
        const int& nextStageIdx = (iterIdx + 1) % kNumStages;
        if (iterIdx + 1 < kNumIters and elect_one_sync()) {
            tma_store_wait<kNumStages - kNumPrefetch - 1>();
            const auto offsetInt4 = i + 32 * kNumSendUnrolls;
            tmaLoadAndArrive(nextStageIdx, copySrcPtr + offsetInt4, kNumTmaBytes);
        }
        __syncwarp();

        // Wait the current TMA arrival
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);

        hooks.template encodeStage<kNumSendUnrolls>(reinterpret_cast<uint4*>(tmaBuffers[stageIdx]), laneId);

        // BF16-family original values, unquantized, no LogFMT packing.
        if (dstOpen && elect_one_sync()) {
            tma_store_1d(tmaBuffers[stageIdx], copyDstPtr + i, kNumTmaBytes);
        }
        __syncwarp();
    }

    // Flush all stores
    tma_store_wait<0>();
    __syncwarp();
}

// Reduction-warp body for one token, used by combine_kernel_impl_2sided_rm_lsa:
// pulls each of this token's up-to-numTopk contributions out of the
// TMA-load warp's ring buffer, decodes+accumulates them weighted, and stores
// the combined result. getDstRank(i) resolves topk slot i's destination rank
// (or a negative sentinel to skip a duplicate-rank/invalid slot), re-derived
// per token from inTopkIdx via warpFirstTopKOccurence. tmaPhase/stageIdx are
// threaded through by reference: this ring buffer state persists across
// tokens, not just within one call.
template <int kNumStages, int kNumElemsPerInt4, int kNumRecvUnrolls, ncclDataType_t kTokenDtype,
    typename FullBarriersT, typename EmptyBarriersT, typename TmaLdBuffersT, typename TmaStBuffersT,
    typename GetDstRankT>
__forceinline__ __device__ void combineReduceTokenLsa(
    int tokenIdx,
    int numTopk,
    int laneId,
    int decodeWarpIdx,
    int kNumBF16PerWarpBytes,
    int64_t hiddenBf16Int4,
    const uint32_t* activeRankMask,
    FullBarriersT fullBarriers,
    EmptyBarriersT emptyBarriers,
    TmaLdBuffersT tmaLdBuffers,
    TmaStBuffersT tmaStBuffers,
    uint32_t& tmaPhase,
    int& stageIdx,
    void* outData,
    GetDstRankT getDstRank) {
    float combinedData[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
    for (int i = 0; i < numTopk; ++i) {
        const int dstRank = getDstRank(i);
        if (dstRank < 0 or not isRankActiveLocal(activeRankMask, dstRank)) continue;
        // Per-rank weight sum is applied by the caller before ncclEpCombine.
        constexpr float topkWeight = 1.0f;

        mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);
        int tmaOffset = kNumBF16PerWarpBytes * decodeWarpIdx;
        decodeAndAccumulateLsa<kNumRecvUnrolls, kTokenDtype>(
            reinterpret_cast<uint32_t*>(
                tmaLdBuffers[stageIdx] + tmaOffset + kNumBF16PerWarpBytes / 32 * laneId),
            combinedData,
            topkWeight);

        // Order generic-proxy reads before releasing the stage to the async TMA producer.
        fence_view_async_shared();
        if (elect_one_sync()) mbarrier_arrive(emptyBarriers[stageIdx]);
        stageIdx = (stageIdx + 1) % kNumStages;
    }
    tma_store_wait<0>();

    encodeCombinedLsa<kNumRecvUnrolls, kTokenDtype>(
        &tmaStBuffers[decodeWarpIdx][kNumRecvUnrolls * 4 * laneId], combinedData);
    tma_store_fence();
    if (elect_one_sync()) {
        const int storeOffsetInt4 = decodeWarpIdx * kNumRecvUnrolls * 32;
        tma_store_1d(
            tmaStBuffers[decodeWarpIdx],
            static_cast<int4*>(outData) + tokenIdx * hiddenBf16Int4 + storeOffsetInt4,
            kNumBF16PerWarpBytes);
    }
    __syncwarp();
}

// CTA-scope barrier, then election of the last-finishing CTA of the grid (one release-scoped atomic per CTA): that CTA
// runs publishOnLast once, after the acquire that pairs with the atomics. Shared by the 2-sided finish handshake
// (syncAndSendFinishLsa) and the Lamport combine, which publishes nothing and only needs the election (round advance).
template <typename PublishFn>
__forceinline__ __device__ bool electLastCtaLsa(
    int warpId, int numWarps, int laneId, int* ctaDone, int numSms, PublishFn&& publishOnLast) {
    // Step 1: CTA-scope barrier (also a full memory fence at CTA scope, per
    // bar.sync semantics) -- every write issued above by any warp of this
    // CTA (this round's token sends) is visible to every thread in this CTA
    // from this point on.
    __syncthreads();

    __shared__ bool shIsLastCta;

    // Steps 2-4: elect the last-finishing CTA in the grid and have it, once,
    // run publishOnLast.
    if (warpId == numWarps - 1) {
        // Step 2: one release-scoped atomic per CTA (not per responsible-expert
        // channel). Every CTA in the grid reaches this exactly once (including
        // ones with no responsible experts this round), so gridDim.x (==
        // numSms) is the correct total.
        int prevDone = 0;
        if (laneId == 0) {
            prevDone = atomic_add_release_global(ctaDone, 1);
        }
        prevDone = __shfl_sync(0xffffffff, prevDone, 0);
        const bool isLastCta = (prevDone + 1 == numSms);
        if (laneId == 0) {
            shIsLastCta = isLastCta;
        }

        if (isLastCta) {
            // Step 3: pairing acquire fence for the atomic release on ctaDone
            // above -- required to observe every other CTA's preceding writes
            // (their token sends) before publishing for them. Acquire-only:
            // the release half belongs to publishOnLast, if it publishes.
            memory_fence_acquire_gpu();
            __syncwarp();

            // Step 4: whatever the caller does once per grid.
            publishOnLast();

            if (laneId == 0) {
                *ctaDone = 0;
            }
        }
    }
    // Propagate shIsLastCta (and, transitively, the electing warp's own
    // fences above) from warpId == numWarps-1 to every other warp of this
    // CTA -- CTA-local only, not a grid-wide barrier.
    __syncthreads();
    return shIsLastCta;
}

// SEND-phase completion for combine_kernel_impl_2sided_rm_lsa, mirroring
// syncAndSendCounts in ll_dispatch_lsa.cuh: a CTA-scope barrier, then
// electing the last-finishing CTA in the grid (one release-scoped atomic per
// CTA) to publish readiness to every peer -- once, instead of every
// responsible-expert channel independently signaling completion via its own
// atomicCleanFlag countdown gate (that per-channel replication exists in the
// general kernel only to spread completion across multiple GIN QPs; plain
// NVLink P2P has no such requirement, so one signal per peer suffices, same
// as dispatch's own syncAndSendCounts). Unlike syncAndSendCounts, there is
// no per-destination count to publish -- combine's receiver already knows
// how many tokens to expect from topk info, so this is a plain readiness
// flag (sendRankFinishFlagLsa), not a value the receiver reads.
__forceinline__ __device__ bool syncAndSendFinishLsa(
    int warpId,
    int numWarps,
    int laneId,
    int* ctaDone,
    int numSms,
    int numRanks,
    int currRank,
    int* recvFlagBuf,
    size_t recvFlagOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {
    // Publish this rank's writes for this round (now provably visible, via
    // the election's acquire) to the whole system before telling peers we're
    // done, then one relaxed store per active peer (see sendRankFinishFlagLsa).
    return electLastCtaLsa(warpId, numWarps, laneId, ctaDone, numSms, [&] {
        memory_fence_release_sys();
#pragma unroll 1
        for (int dstRank = laneId; dstRank < numRanks; dstRank += 32) {
            sendRankFinishFlagLsa(dstRank, currRank, recvFlagBuf, recvFlagOff, rankMask, windows, devComm);
        }
    });
}

// SEND phase of the 2-sided LSA combine kernels. The warp group that owns responsibleExpertIdx TMA-pushes every token
// it received from dstRank into that rank's landing slot (tokenIdx * numTopk + j_eff) of recvBuf, rank-major; the warps
// of the group split the tokens. recvOff is recvBuf's offset in the window (p2p address translation).
// `hooks`: PlainSendHooks, or LamportSendHooks to encode the payload for the Lamport receiver (processAndSendTokenLsa).
template <int kHidden, int kNumTopk, ncclDataType_t kTokenDtype, typename SendHooksT>
__device__ __forceinline__ void combineSendTokensLsa(
    const void* inData,
    const int* srcInfo,
    int* rankMask,
    void* sendBuf,
    void* recvBuf,
    size_t recvOff,
    uint8_t* smemBuffer,
    int warpId,
    int laneId,
    int subWarpId,
    int responsibleExpertIdx,
    int numWarpsPerGroup,
    int maxTokensPerRank,
    int numExperts,
    int numLocalExperts,
    int currRank,
    int numRanks,
    bool zeroCopy,
    ncclDevComm* devComm,
    const ncclWindow_t* windows,
    SendHooksT hooks) {
    constexpr int numTopk = kNumTopk;
    constexpr int kElemBytes = (kTokenDtype == ncclFloat32) ? 4 : 2;
    constexpr int kNumElemsPerInt4 = combine_smem::elements_per_int4(kElemBytes);
    constexpr int kWarpSize = combine_smem::kWarpSize;
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;
    constexpr int kNumSendUnrolls = combine_smem::send_unrolls(kHidden, kElemBytes);
    // Wire slot size, see numBytesPerSlot in the kernels.
    constexpr size_t numBytesPerSlot = combine_smem::none_slot_bytes(kHidden, kElemBytes);

    if (responsibleExpertIdx >= numExperts) {
        return;
    }
    const auto dstRank = responsibleExpertIdx / numLocalExperts;
    // Each pair of rank establish numLocalExperts channels for parallelization
    const auto rankLaneIdx = responsibleExpertIdx % numLocalExperts;

    auto slotsPerToken = numTopk + 1; // token_id + topk_idx's
    // We have numRanks entries with per-rank count first, then the rest is (tokenId + numTopk) entries for each token
    const auto localSrcInfo = srcInfo + numRanks + dstRank * maxTokensPerRank * slotsPerToken;

    // TMA stuffs
    constexpr int kNumTMABufferBytes = sizeof(int4) * kWarpSize * kNumSendUnrolls;
    constexpr int kNumStages = combine_smem::kNumStages;
    constexpr int kNumPrefetch = combine_smem::kNumTmaPrefetch;
    EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

    auto smemPtr =
        smemBuffer + warpId * (kNumStages * (kNumTMABufferBytes + combine_smem::kTmaBarrierAndPaddingBytes));
    uint32_t tmaPhase = 0;
    auto tmaBuffers = PatternVisitor([=](const int& i) {
        return reinterpret_cast<int4*>(
            smemPtr + i * (kNumTMABufferBytes + combine_smem::kTmaBarrierAndPaddingBytes));
    });
    auto fullBarriers = PatternVisitor([=](const int& i) {
        return reinterpret_cast<uint64_t*>(
            smemPtr + i * (kNumTMABufferBytes + combine_smem::kTmaBarrierAndPaddingBytes) + kNumTMABufferBytes);
    });
    EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

    // Initialize m-barriers
    if (laneId < kNumStages) {
        mbarrier_init(fullBarriers[laneId], 1);
        fence_barrier_init();
    }
    __syncwarp();

    auto tmaLoadAndArrive = [&](const int& stageIdx, const int4* gmemPtr, const int& numBytes) {
        tma_load_1d(tmaBuffers[stageIdx], gmemPtr, fullBarriers[stageIdx], numBytes);
        mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numBytes);
    };

    // Issue sends for each token from the responsible dstRank. Rank-major:
    // one send per received slot. srcInfo[0] = linear slot into inData;
    // srcInfo[1] = j_eff (topk return index). The receive slot is
    // (tokenIdx * numTopk + j_eff) * numBytesPerSlot, placing each expert
    // rank's contribution into a distinct position in the recv buffer.
    if (not isRankMasked<true>(rankMask, dstRank)) {
        // Read right before the loop: kept live across the TMA set-up it spilled.
        const int numRecvTokens = __shfl_sync(0xffffffff, __ldg(srcInfo + dstRank), 0);
        for (int i = subWarpId * numLocalExperts + rankLaneIdx; i < numRecvTokens;
             i += numWarpsPerGroup * numLocalExperts) {
            auto localSrcTokenInfo = localSrcInfo + i * slotsPerToken;
            auto localSrcTopkInfo = localSrcTokenInfo + 1;
            int tokenIdx = __shfl_sync(0xffffffff, __ldg(localSrcTokenInfo), 0);

            int slot = __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + 0), 0);
            int j_eff = (numTopk > 1) ? __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + 1), 0) : 0;
            if (j_eff < 0) {  // no local expert found (shouldn't happen)
                continue;
            }

            // Byte offsets scale with slot (= srcRank*maxTokensPerRank + i)
            // and tokenIdx*numTopk, which times numBytesPerSlot can exceed
            // INT_MAX -- use size_t to avoid 32-bit truncation.
            size_t sndTokenOffset = static_cast<size_t>(slot) * numBytesPerSlot;
            size_t rcvTokenOffset = static_cast<size_t>(tokenIdx * numTopk + j_eff) * numBytesPerSlot;

            const auto srcDataInt4Ptr = static_cast<const int4*>(inData) + (int64_t)slot * hiddenBf16Int4;
            const auto sendBufUint8 = static_cast<uint8_t*>(sendBuf) + sndTokenOffset;
            const auto sendBufPtr = reinterpret_cast<int4*>(sendBufUint8);
            const auto recvPtr = reinterpret_cast<uint64_t>(recvBuf) + rcvTokenOffset;
            const auto expectedDstOffset = recvOff + rcvTokenOffset;
            const auto dstP2pPtr =
                ncclGetP2pPtr(recvPtr, expectedDstOffset, currRank, dstRank, windows, devComm);

            processAndSendTokenLsa<kNumSendUnrolls, kNumStages, kNumPrefetch>(
                srcDataInt4Ptr,
                sendBufPtr,
                dstP2pPtr,
                hiddenBf16Int4,
                zeroCopy,
                tmaBuffers,
                fullBarriers,
                tmaPhase,
                tmaLoadAndArrive,
                hooks,
                currRank,
                rankMask,
                laneId);
        }
    }
    // Destroy m-barriers (per-warp, no cross-warp dependency -- each
    // warp owns its own m-barrier region via smemPtr's warpId offset
    // above, so no bar.sync is needed to gate this).
    if (laneId < kNumStages) {
        mbarrier_inval(fullBarriers[laneId]);
        fence_barrier_init();
    }
    __syncwarp();

    // Wait for all stores to complete
    tma_store_wait_complete<0>();
}

// LSA-only, unquantized, rank-major-only LL combine entry point. See the
// scope-cuts doc comment at the top of this file.
template <int kHidden, int kNumTopk, int kNumMaxUnrolls, typename TopkIdxT, ncclDataType_t kTokenDtype>
__device__ __forceinline__ void combine_kernel_impl_2sided_rm_lsa( // INPUT
    const void* inData,
    const int* srcInfo,
    const TopkIdxT* inTopkIdx,
    int* rankMask,
    int* asyncErrorFlag,
    // OUTPUT
    void* outData,
    // INTERMEDIATE
    void* commBuf,
    size_t sendOffBase,
    size_t recvOffBase,
    size_t recvFlagOffBase,
    int* ctaDone,
    int lsaSignalingCleanCount,
    int64_t* waitStats,
    LowLatencyEpochState* epochState,
    size_t payloadSlotStride,
    size_t signalSlotStride,
    // CONFIG
    int numCombinedTokens,
    int maxTokensPerRank,
    int numExperts,
    int currRank,
    int numRanks,
    int numWarpGroups,
    int numWarpsPerGroup,
    int phases,
    bool zeroCopy,
    ncclDevComm* devComm,
    const ncclWindow_t* windows,
    uint64_t timeoutCycles) {
    static constexpr ncclEpLayout_t kLayout = NCCL_EP_LAYOUT_RANK_MAJOR;
    constexpr int numTopk = kNumTopk;
    EP_STATIC_ASSERT(numTopk > 0, "numTopk must be positive");
    // Token dtype derivations
    constexpr int kElemBytes = (kTokenDtype == ncclFloat32) ? 4 : 2;

    const auto smId = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto numSms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto numThreads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warpId = __shfl_sync(0xffffffff, threadId / 32, 0), laneId = get_lane_id();
    const auto numLocalExperts = numExperts / numRanks;
    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    const unsigned int epoch = selectLowLatencyEpoch(epochState, phases, smId, threadId);
    const size_t bank = static_cast<size_t>(epoch & 1U);
    const size_t bank_next = bank ^ 1U;
    const size_t sendOff = sendOffBase + bank * payloadSlotStride;
    const size_t recvOff = recvOffBase + bank * payloadSlotStride;
    const size_t recvFlagOff = recvFlagOffBase + bank * signalSlotStride;
    char* const commBase = static_cast<char*>(commBuf);
    void* const sendBuf = commBase + sendOff;
    void* const recvBuf = commBase + recvOff;
    int* const recvFlagBuf = reinterpret_cast<int*>(commBase + recvFlagOff);
    int* const nextRecvCntBuf = reinterpret_cast<int*>(commBase + recvFlagOffBase + bank_next * signalSlotStride);

    extern __shared__ __align__(1024) uint8_t smemBuffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = combine_smem::elements_per_int4(kElemBytes);
    constexpr int kWarpSize = combine_smem::kWarpSize;
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = combine_smem::send_unrolls(kHidden, kElemBytes);
    constexpr int kNumRecvUnrolls = combine_smem::recv_unrolls(kHidden, kElemBytes);
    EP_STATIC_ASSERT(kNumRecvUnrolls > 0, "hidden too wide: no LL combine receive group fits a CTA");
    EP_STATIC_ASSERT(kHidden % (kWarpSize * kNumRecvUnrolls * kNumElemsPerInt4) == 0,
                     "Partial tiles require universal combine");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hiddenBf16Int4 % kNumSendUnrolls == 0, "Invalid hidden");

    // Message package: the NONE-recipe wire slot size, including its
    // reserved (here, always-unused) LogFMT metadata tail -- see the
    // top-of-file doc comment for why that tail stays.
    constexpr size_t numBytesPerSlot = combine_smem::none_slot_bytes(kHidden, kElemBytes);
    EP_STATIC_ASSERT(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

    // Set by syncAndSendFinishLsa below; declared (and default-initialized)
    // here so it's still in scope, and safely readable, at the RECV label
    // below even for a RECV-only call that jumps straight past the SEND
    // phase via the goto immediately below.
    bool isLastCta = false;

    // Populated by waitForAllRanksFinishLsa below (RECV phase); see its doc
    // comment. Declared ahead of the SEND/RECV goto, like isLastCta above,
    // purely for scope.
    __shared__ uint32_t activeRankMaskShared[combine_smem::kActiveRankMaskWords];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_COMBINE_LSA_RECV;
    }

    // Clean up next buffer: signal + read-done span in one pass, even though 2-sided itself never
    // touches read-done -- it's prepping the bank for whichever op lands there next (see
    // LowLatencyBuffer::lsa_signaling_clean_int_count). ctaDone (syncAndSendFinishLsa's own
    // CTA-arrival counter, below) is untouched here -- nothing to notify. Every thread of this CTA
    // participates (not just one warp): all are still idle here, ahead of their own token-sending
    // work below, and each writes a distinct index, so no synchronization is needed between them.
    if (smId == 0) {
        cleanNextRecvCntBufStrided(nextRecvCntBuf, lsaSignalingCleanCount, threadId, numThreads);
    }

    {
        // Scoped: the SEND-phase goto above must not jump over an initialisation.
        combineSendTokensLsa<kHidden, kNumTopk, kTokenDtype>(
            inData, srcInfo, rankMask, sendBuf, recvBuf, recvOff, smemBuffer, warpId, laneId, subWarpId,
            responsibleExpertIdx, numWarpsPerGroup, maxTokensPerRank, numExperts, numLocalExperts, currRank, numRanks,
            zeroCopy, devComm, windows, PlainSendHooks{});
    }

    // Every CTA that had a responsible expert (or none at all) has now
    // finished its own sends. Elect the last-finishing CTA and have it
    // broadcast readiness to every peer, replacing the old per-channel
    // atomicCleanFlag countdown gate + per-channel sendFinishFlagLsa above.
    isLastCta = syncAndSendFinishLsa(
        warpId, numWarps, laneId, ctaDone, numSms, numRanks, currRank, recvFlagBuf, recvFlagOff, rankMask,
        windows, devComm);

// Receiving phase
LOW_LATENCY_COMBINE_LSA_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
        return;
    }

    // Split invocation (SEND phase didn't run here): syncAndSendFinishLsa
    // never elected anyone, so nominate smId == 0 instead -- same fallback
    // as dispatch_kernel_impl_2sided_rm_lsa.
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        isLastCta = (smId == 0);
    }

    // Cross-rank readiness: every CTA independently confirms every active
    // peer has broadcast readiness (dispatch's syncAndRecvCounts pattern,
    // reused as-is via waitForAllRanksFinishLsa since neither cares about
    // the flag's value, only that it's non-zero) -- no grid-wide barrier
    // needed for this, unlike the old waitForRecvFlagLsa (which was keyed
    // per (srcRank, localExpert) and relied on a cg::this_grid().sync() to
    // propagate readiness). completeFullLowLatencyEpoch below relies on
    // isLastCta (elected above, or the split-invocation fallback) rather
    // than the "smId == 0" convention, which would have no ordering
    // guarantee against other CTAs' epoch reads without a grid-wide barrier.
    waitForAllRanksFinishLsa(
        threadId, numRanks, rankMask, activeRankMaskShared, recvFlagBuf, timeoutCycles, currRank, asyncErrorFlag,
        waitStats);
    completeFullLowLatencyEpoch(epochState, phases, isLastCta, threadId);

    // Reassign warp groups; FP32 doubles SMEM per group, drop to 1 group to stay within limits
    constexpr int kMaxNumGroups = combine_smem::max_recv_groups(kElemBytes);
    constexpr int numDecodeWarps = combine_smem::recv_decode_warps(kHidden, kElemBytes);
    const int warpsPerRecvGroup = numDecodeWarps + combine_smem::kNumRecvTmaWarps;
    const int numGroups = min(kMaxNumGroups, (numThreads / kWarpSize) / warpsPerRecvGroup);
    const int decodeWarpIdx = __shfl_sync(0xffffffff, warpId % warpsPerRecvGroup, 0);
    const int groupIdx = __shfl_sync(0xffffffff, warpId / warpsPerRecvGroup, 0);
    EP_DEVICE_ASSERT(numTopk <= kWarpSize);
    EP_DEVICE_ASSERT(numGroups > 0);

    if (groupIdx < numGroups) {
        constexpr int kNumStages = combine_smem::kNumStages;
        constexpr int kNumTMABufferBytes = combine_smem::recv_tma_buffer_bytes(kHidden, kElemBytes, /*nvfp4=*/false);
        constexpr int kNumBF16PerWarpBytes = kWarpSize * kNumRecvUnrolls * kNumElemsPerInt4 * kElemBytes;
        constexpr int kTokenBytes = kHidden * kElemBytes;
        // No LogFMT metadata region (logAmax/logAmin/castInfo) -- nothing
        // populates or consumes it once LogFMT is gone.
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kTokenBytes;

        // Reallocate shared memory
        const auto smemGroupBuffer = smemBuffer + kNumBytesPerGroup * groupIdx;
        auto fullBarriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes); });
        auto emptyBarriers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint64_t*>(
                smemGroupBuffer + i * kNumTMABufferBytes + combine_smem::kRecvTmaEmptyBarrierOffsetBytes);
        });
        auto tmaLdBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint8_t*>(
                smemGroupBuffer + i * kNumTMABufferBytes + combine_smem::kRecvTmaPayloadOffsetBytes);
        });
        auto tmaStBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(
                smemGroupBuffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });

        uint32_t tmaPhase = 0;
        EP_STATIC_ASSERT(kNumStages < kWarpSize, "Too many stages");
        if (decodeWarpIdx == numDecodeWarps) tmaPhase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decodeWarpIdx == numDecodeWarps and laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            mbarrier_init(emptyBarriers[laneId], numDecodeWarps);
        }
        syncSmGroup(groupIdx + 1, warpsPerRecvGroup * kWarpSize);

        int stageIdx = 0, topkIdxByLane = 0;
        EP_STATIC_ASSERT(
            kNumTopk <= kWarpSize,
            "numTopk must not exceed warp size: warpFirstTopKOccurence relies on active "
            "lanes (0..numTopk-1) having lower IDs than inactive lanes (numTopk..31)");

        if (decodeWarpIdx == numDecodeWarps) {
            // TMA load warp
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens;
                 tokenIdx += numSms * numGroups) {
                if (laneId < numTopk) topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));

                topkIdxByLane = warpFirstTopKOccurence(topkIdxByLane, numLocalExperts, laneId);

                for (int i = 0; i < numTopk; ++i) {
                    int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                    if (topkIdxReg < 0) continue;
                    if (not isRankActiveLocal(activeRankMaskShared, topkIdxReg / numLocalExperts)) continue;

                    mbarrier_wait<true>(emptyBarriers[stageIdx], tmaPhase, stageIdx);
                    auto recvBufData = static_cast<uint8_t*>(recvBuf) + (tokenIdx * numTopk + i) * numBytesPerSlot;

                    if (elect_one_sync()) {
                        const int numTmaBytes = kHidden * kElemBytes;
                        tma_load_1d(tmaLdBuffers[stageIdx], recvBufData, fullBarriers[stageIdx], numTmaBytes);
                        mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numTmaBytes);
                    }
                    __syncwarp();
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens;
                 tokenIdx += numSms * numGroups) {
                if (laneId < numTopk) {
                    topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));
                }
                __syncwarp();

                topkIdxByLane = warpFirstTopKOccurence(topkIdxByLane, numLocalExperts, laneId);

                combineReduceTokenLsa<kNumStages, kNumElemsPerInt4, kNumRecvUnrolls, kTokenDtype>(
                    tokenIdx, numTopk, laneId, decodeWarpIdx, kNumBF16PerWarpBytes, hiddenBf16Int4,
                    activeRankMaskShared, fullBarriers, emptyBarriers, tmaLdBuffers, tmaStBuffers, tmaPhase, stageIdx,
                    outData,
                    [&](int i) {
                        const int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                        return topkIdxReg < 0 ? -1 : topkIdxReg / numLocalExperts;
                    });
            }
        }
    }
}

// Take up to kGroup slots, lowest first, out of the `remaining` bit mask into slots[]; returns the bit mask of the
// taken group positions (bit j set <=> slots[j] valid).
template <int kGroup>
__forceinline__ __device__ unsigned takeCombineSlots(unsigned& remaining, int (&slots)[kGroup]) {
    unsigned taken = 0;
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
        slots[j] = 0;
        if (remaining) {
            slots[j] = __ffs(remaining) - 1;
            remaining &= remaining - 1;
            taken |= 1u << j;
        }
    }
    return taken;
}

// Timeout of a Lamport receive, as in the 2-sided receive: warn (once per CTA), trap without a rank mask, otherwise
// mask the ranks of the `pending` group positions, raise the async error and let the caller go on without them.
template <int kGroup, typename TopkIdxT>
__forceinline__ __device__ void handleLamportRecvTimeout(
    unsigned pending,
    const int (&slots)[kGroup],
    int tokenIdx,
    int numTopk,
    const TopkIdxT* inTopkIdx,
    int numLocalExperts,
    int currRank,
    int* rankMask,
    int* asyncErrorFlag,
    int* shTimeoutReports) {
    if (!pending) {
        return;
    }
    if (atomicAdd(shTimeoutReports, 1) == 0) {
        printf(
            "Warning: NCCL EP timeout for Lamport combine receive, rank %d, token %d, pending slot mask 0x%x\n",
            currRank, tokenIdx, pending);
    }
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
        if ((pending >> j) & 1u) {
            const int peer = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + slots[j])) / numLocalExperts;
            maskTimedOutPeer(rankMask, peer, asyncErrorFlag);
        }
    }
}

// NCCL_EP_COMBINE_LSA=lamport: the 2-sided kernel's send phase into a sentinel-poisoned landing region
// (LamportRegion: at the tail of the buffer, two segments selected by the combine round parity), no finish flag. The
// receive has no barrier: each thread polls one 16-byte vector of every contributing slot until no word is the
// sentinel, accumulates in top-k order and re-arms the slot. Before its first store into a peer's segment a sender
// waits until that peer has started the previous round (round words), which keeps the segment's previous user out.
// Protocol and the sentinel encoding: docs/plc/401-ll-combine-lamport.
template <int kHidden, int kNumTopk, int kNumMaxUnrolls, typename TopkIdxT, ncclDataType_t kTokenDtype>
__device__ __forceinline__ void combine_kernel_impl_2sided_lamport_rm_lsa( // INPUT
    const void* inData,
    const int* srcInfo,
    const TopkIdxT* inTopkIdx,
    int* rankMask,
    int* asyncErrorFlag,
    // OUTPUT
    void* outData,
    // INTERMEDIATE
    void* commBuf,
    int* ctaDone,
    LamportView lamportRegion,
    LowLatencyEpochState* epochState, // only its send_in_flight mark: the epoch itself is not touched
    // CONFIG
    int numCombinedTokens,
    int maxTokensPerRank,
    int numExperts,
    int currRank,
    int numRanks,
    int numWarpGroups,
    int numWarpsPerGroup,
    int phases,
    ncclDevComm* devComm,
    const ncclWindow_t* windows,
    uint64_t timeoutCycles) {
    constexpr int numTopk = kNumTopk;
    EP_STATIC_ASSERT(numTopk > 0, "numTopk must be positive");
    // Token dtype derivations
    constexpr int kElemBytes = (kTokenDtype == ncclFloat32) ? 4 : 2;

    const auto smId = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto numSms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto numThreads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warpId = __shfl_sync(0xffffffff, threadId / 32, 0), laneId = get_lane_id();
    const auto numLocalExperts = numExperts / numRanks;
    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    if (smId == 0 && threadId == 0) {
        pairStagedLaunch(epochState, phases);
    }

    // Combine round, advanced by the last CTA at the end of the receive, so a SEND-only and its RECV-only launch read
    // the same value. Its parity selects the landing segment. The LL epoch (dispatch's and the non-Lamport combine
    // kernels' counter) is neither read nor advanced: this kernel uses no signalling bank and no payload slot, so
    // consecutive dispatches alternate banks among themselves and a fallback combine in between sees the banks it
    // expects.
    const unsigned int round = lamportRegion.state->round;
    const size_t recvOff = lamport::segment_offset(lamportRegion.recv_offset, lamportRegion.segment_stride, round);
    char* const commBase = static_cast<char*>(commBuf);
    void* const recvBuf = commBase + recvOff;
    int* const roundWords = reinterpret_cast<int*>(commBase + lamportRegion.round_offset);
    // Destination of this warp group's sends (rank-major); its round word is read by the sending warps only, inside
    // the send pipeline (LamportRoundCheck), never here by every thread.
    const int sendDstRank = responsibleExpertIdx < numExperts ? responsibleExpertIdx / numLocalExperts : -1;

    extern __shared__ __align__(1024) uint8_t smemBuffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = combine_smem::elements_per_int4(kElemBytes);
    constexpr int kWarpSize = combine_smem::kWarpSize;
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = combine_smem::send_unrolls(kHidden, kElemBytes);
    constexpr int kNumRecvUnrolls = combine_smem::recv_unrolls(kHidden, kElemBytes);
    EP_STATIC_ASSERT(kNumRecvUnrolls > 0, "hidden too wide: no LL combine receive group fits a CTA");
    EP_STATIC_ASSERT(kHidden % (kWarpSize * kNumRecvUnrolls * kNumElemsPerInt4) == 0,
                     "Partial tiles require universal combine");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hiddenBf16Int4 % kNumSendUnrolls == 0, "Invalid hidden");

    // Message package: the NONE-recipe wire slot size, including its
    // reserved (here, always-unused) LogFMT metadata tail -- see the
    // top-of-file doc comment for why that tail stays.
    constexpr size_t numBytesPerSlot = combine_smem::none_slot_bytes(kHidden, kElemBytes);
    EP_STATIC_ASSERT(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

    // Declared ahead of the SEND/RECV goto for scope (see the 2-sided kernel).
    bool isLastCta = false;
    // Timeout warnings printed by this CTA's receive (first one only).
    __shared__ int shTimeoutReports;
    if (threadId == 0) {
        shTimeoutReports = 0;
    }
    __syncthreads();

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_COMBINE_LAMPORT_RECV;
    }

    {
        // Destination round check, consumed inside the send pipeline after its first stage load (see
        // LamportRoundCheck). Scoped: the SEND-phase goto above must not jump over an initialisation.
        LamportSendHooks<lamport::sanitized_word<kTokenDtype>()> hooks{LamportRoundCheck{
            sendDstRank >= 0 ? roundWords + sendDstRank : nullptr, round, sendDstRank, timeoutCycles, asyncErrorFlag,
            &shTimeoutReports, /*closed=*/false}};
        combineSendTokensLsa<kHidden, kNumTopk, kTokenDtype>(
            inData, srcInfo, rankMask, /*sendBuf=*/nullptr, recvBuf, recvOff, smemBuffer, warpId, laneId, subWarpId,
            responsibleExpertIdx, numWarpsPerGroup, maxTokensPerRank, numExperts, numLocalExperts, currRank, numRanks,
            /*zeroCopy=*/false, devComm, windows, hooks);
    }
    // Publish "in round `round`" to the peers after the sends rather than before them: the word only has to witness
    // that the previous round has retired, and here the peer-pointer lookup overlaps the arrival of the first data
    // instead of delaying the sends.
    publishLamportRoundLsa(
        smId, numSms, threadId, numRanks, currRank, round, roundWords, lamportRegion.round_offset, rankMask, windows,
        devComm);

    // No finish flag and no CTA barrier: a warp that finished its sends goes straight on to receiving. The last-CTA
    // election for the round advance is at the end of the kernel.

// Receiving phase
LOW_LATENCY_COMBINE_LAMPORT_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
        return;
    }

    // ---- Lamport-style receive: one 16-byte vector per thread, no barrier ----
    constexpr int kVecBytes = lamport::kVecBytes;
    constexpr int kVecsPerRow = kHidden * kElemBytes / kVecBytes;
    constexpr int kElemsPerVec = kVecBytes / kElemBytes;
    EP_STATIC_ASSERT(kVecsPerRow % kWarpSize == 0, "a warp must stay inside one token row");
    EP_STATIC_ASSERT(numTopk <= kWarpSize, "numTopk must not exceed warp size (warpFirstTopKOccurence)");
    const int64_t totalVecs = static_cast<int64_t>(numCombinedTokens) * kVecsPerRow;
    const int64_t gridStride = static_cast<int64_t>(numSms) * numThreads;
    int topkIdxByLane = 0;

    // Every lane of a warp lies in one token row (kVecsPerRow % 32 == 0): the warp-collective calls below are uniform.
    for (int64_t g = static_cast<int64_t>(smId) * numThreads + threadId; g < totalVecs; g += gridStride) {
        const int tokenIdx = static_cast<int>(g / kVecsPerRow);
        const int v = static_cast<int>(g % kVecsPerRow);

        if (laneId < numTopk) {
            topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));
        }
        topkIdxByLane = warpFirstTopKOccurence(topkIdxByLane, numLocalExperts, laneId);

        // Slots (top-k positions) that deliver a contribution: first occurrence of a valid rank that is not masked.
        const int slotRank = topkIdxByLane < 0 ? -1 : topkIdxByLane / numLocalExperts;
        const bool live = laneId < numTopk && slotRank >= 0 && !isRankMasked(rankMask, slotRank);
        unsigned remaining = __ballot_sync(0xffffffffu, live);

        float acc[kElemsPerVec];
#pragma unroll
        for (int e = 0; e < kElemsPerVec; ++e) {
            acc[e] = 0.0f;
        }

        // Unweighted sum: the caller applies the per-rank weights before ncclEpCombine.
        constexpr float kUnitWeight = 1.0f;
        // Up to kLamportGroup live slots polled concurrently, in top-k order; 8 spilled registers.
        constexpr int kLamportGroup = 4;
        uint8_t* const vecBase = static_cast<uint8_t*>(recvBuf) +
            static_cast<size_t>(tokenIdx) * numTopk * numBytesPerSlot + static_cast<size_t>(v) * kVecBytes;
        while (remaining) {  // warp-uniform
            int slots[kLamportGroup];
            const unsigned takenMask = takeCombineSlots<kLamportGroup>(remaining, slots);
            uint4 buf[kLamportGroup];
            const auto slotPtr = [&](int j) {
                return reinterpret_cast<uint4*>(vecBase + static_cast<size_t>(slots[j]) * numBytesPerSlot);
            };
            const unsigned pending = lamport::poll_vec_per_thread<kLamportGroup>(
                takenMask, timeoutCycles, buf, [&](int j) { return ld_volatile_global(slotPtr(j)); });
            handleLamportRecvTimeout<kLamportGroup>(
                pending, slots, tokenIdx, numTopk, inTopkIdx, numLocalExperts, currRank, rankMask, asyncErrorFlag,
                &shTimeoutReports);
            __syncwarp();  // lanes leave the poll loop at different times; reconverge before the next warp-collective
            // Slots that arrived: re-arm and accumulate in slot order.
#pragma unroll
            for (int j = 0; j < kLamportGroup; ++j) {
                if (((takenMask >> j) & 1u) && !((pending >> j) & 1u)) {
                    st_volatile_global(slotPtr(j), lamport::sentinel_vec());
                    decodeAndAccumulateLsa<1, kTokenDtype>(reinterpret_cast<uint32_t*>(&buf[j]), acc, kUnitWeight);
                }
            }
        }

        uint32_t packed[4];
        encodeCombinedLsa<1, kTokenDtype>(packed, acc);
        reinterpret_cast<uint4*>(outData)[g] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        __syncwarp();
    }

    // The last CTA to finish its receive advances the round. Every CTA read the counter at kernel entry; the election's
    // release/acquire orders those reads before the bump.
    isLastCta = electLastCtaLsa(warpId, numWarps, laneId, ctaDone, numSms, [] {});
    if (isLastCta && threadId == 0) {
        ++lamportRegion.state->round;
    }
}

} // namespace ll

} // namespace nccl_ep

namespace nccl_ep {

namespace ll {

// "Every peer that could have pulled from me has confirmed it's done reading my inData" wait,
// used only by combine_kernel_impl_1sided_rm_lsa right before it returns. Unlike
// waitForAllRanksFinishLsa above (every CTA redundantly re-checks all of numRanks, because each
// CTA's own subsequent pulling depends on it), nothing within this kernel invocation gates on a
// *specific* CTA having observed a given peer's done-flag -- it only needs to be true, grid-wide,
// by the time this kernel retires, which is what the caller relies on via CUDA's stream ordering
// before it may safely let the next round's expert-compute step overwrite this rank's inData. So
// this distributes [0, numRanks) once across the whole grid (every (CTA, thread) pair a disjoint
// slice) instead of covering it numSms times over. No closing __syncthreads()/memory_fence() to
// propagate visibility within the CTA: this is the last thing the kernel does.
__forceinline__ __device__ void waitForAllReadDoneDistributedLsa(
    int smId,
    int numSms,
    int threadId,
    int numThreads,
    int numRanks,
    int* rankMask,
    const int* readDoneBuf,
    uint64_t timeoutCycles,
    int currRank,
    int* asyncErrorFlag) {
    const auto startTime = clock64();
    for (int peer = smId * numThreads + threadId; peer < numRanks; peer += numSms * numThreads) {
        if (isRankMasked(rankMask, peer)) continue;
        uint64_t waitCost = 0;
        while (ld_relaxed_sys_global(reinterpret_cast<const uint32_t*>(readDoneBuf + peer)) == 0 &&
               (waitCost = clock64() - startTime) <= timeoutCycles);
        if (waitCost > timeoutCycles) {
            printf("Warning: NCCL EP timeout for combine read-done ack, rank %d, peer %d\n", currRank, peer);
            maskTimedOutPeer(rankMask, peer, asyncErrorFlag);
        }
    }
}

// LSA-only, unquantized (NONE recipe), rank-major-only, BF16-only LL combine entry point for the
// dispatch-push/combine-pull ("1-sided") mode -- pairs with any 2-sided dispatch kernel that was
// run with kRecordDstInfo=true (ll/ll_dispatch_lsa.cuh), which additionally records each token's
// (dstRank, slotIdx) into dstInfo. This kernel never sends anything itself: it pulls each
// token's already-computed result directly out of the owning rank's inData window (resolved via
// ncclGetPeerPointer(inDataWindow, ...)), decoding with a -1 dstRank sentinel for
// duplicate-rank/invalid topk slots (written unconditionally by dispatch -- see its own doc
// comment). SEND phase is reduced to a single-shot readiness broadcast (telling every peer "my
// inData is ready to be pulled from"); RECV phase additionally must wait, before retiring, for
// every peer that could have pulled from THIS rank's inData to confirm it's done -- the caller's
// own inData buffer must not be overwritten by the next round's expert compute until then, which
// is what the trailing waitForAllReadDoneDistributedLsa call guarantees (see its own doc
// comment).
template <int kHidden, int kNumTopk, int kNumMaxUnrolls, ncclDataType_t kTokenDtype>
__device__ __forceinline__ void combine_kernel_impl_1sided_rm_lsa( // INPUT
    const void* inData,
    ncclWindow_t inDataWindow,
    size_t inDataWindowOffset,
    const int* dstInfo, // (dstRank, slotIdx) pairs written by dispatch's kRecordDstInfo path
    int* rankMask,
    int* asyncErrorFlag,
    // OUTPUT
    void* outData,
    // INTERMEDIATE
    void* commBuf,
    size_t recvFlagOffBase,
    // NCCL_EP_COMBINE_SYNC_COUNT-element array of grid-wide election counters. ctaSync[1] (never
    // ctaSync[0] again) backs the SECOND (RECV-phase read-done) election below: the two elections
    // are separated only by a per-CTA __syncthreads(), not a grid-wide barrier, so reusing
    // ctaSync[0] races (see its doc comment in ll_ep_adapter.cuh).
    int* ctaSync,
    int lsaSignalingCleanCount,
    int64_t* waitStats,
    LowLatencyEpochState* epochState,
    size_t payloadSlotStride,
    size_t signalSlotStride,
    size_t readDoneOffBase,
    // CONFIG
    int numCombinedTokens,
    int maxTokensPerRank,
    int numExperts,
    int currRank,
    int numRanks,
    int numWarpGroups,
    int numWarpsPerGroup,
    int phases,
    ncclDevComm* devComm,
    const ncclWindow_t* windows,
    uint64_t timeoutCycles) {
    constexpr int numTopk = kNumTopk;
    EP_STATIC_ASSERT(numTopk > 0, "numTopk must be positive");
    constexpr int kElemBytes = (kTokenDtype == ncclFloat32) ? 4 : 2;

    const auto smId = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto numSms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto numThreads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warpId = __shfl_sync(0xffffffff, threadId / 32, 0), laneId = get_lane_id();
    // Unlike the 2-sided kernel, dstRank/numLocalExperts routing is never re-derived here (it
    // comes straight from dstInfo); numWarpGroups/numWarpsPerGroup/numExperts are kept in the
    // signature only for call-site symmetry with the 2-sided launch path.
    (void)numExperts;
    (void)numWarpGroups;
    (void)numWarpsPerGroup;
    (void)payloadSlotStride;

    const unsigned int epoch = selectLowLatencyEpoch(epochState, phases, smId, threadId);
    const size_t bank = static_cast<size_t>(epoch & 1U);
    const size_t bank_next = bank ^ 1U;
    const size_t recvFlagOff = recvFlagOffBase + bank * signalSlotStride;
    const size_t readDoneOff = readDoneOffBase + bank * signalSlotStride;
    char* const commBase = static_cast<char*>(commBuf);
    int* const recvFlagBuf = reinterpret_cast<int*>(commBase + recvFlagOff);
    int* const nextRecvCntBuf = reinterpret_cast<int*>(commBase + recvFlagOffBase + bank_next * signalSlotStride);
    int* const readDoneBuf = reinterpret_cast<int*>(commBase + readDoneOff);

    extern __shared__ __align__(1024) uint8_t smemBuffer[];

    constexpr int kNumElemsPerInt4 = combine_smem::elements_per_int4(kElemBytes);
    constexpr int kWarpSize = combine_smem::kWarpSize;
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;

    constexpr int kNumRecvUnrolls = combine_smem::recv_unrolls(kHidden, kElemBytes);
    EP_STATIC_ASSERT(kNumRecvUnrolls > 0, "hidden too wide: no LL combine receive group fits a CTA");
    EP_STATIC_ASSERT(kHidden % (kWarpSize * kNumRecvUnrolls * kNumElemsPerInt4) == 0,
                     "Partial tiles require universal combine");
    EP_STATIC_ASSERT(kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");

    bool isLastCta = false;
    __shared__ uint32_t activeRankMaskShared[combine_smem::kActiveRankMaskWords];

    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_COMBINE_LSA_RECV_1SIDED;
    }

    // Single-shot readiness: the caller already fully wrote this rank's inData (every token this
    // rank is responsible for) before launching this kernel's SEND phase -- there is no payload
    // left to send, only the "ready to be pulled from" broadcast.
    if (smId == 0) {
        cleanNextRecvCntBufStrided(nextRecvCntBuf, lsaSignalingCleanCount, threadId, numThreads);
        __syncthreads();

        if (warpId == 0) {
            memory_fence_release_sys();
            for (int peer = laneId; peer < numRanks; peer += 32) {
                sendRankFinishFlagLsa(peer, currRank, recvFlagBuf, recvFlagOff, rankMask, windows, devComm);
            }
        }
    }

    // Elect the CTA allowed to bump epochState->epoch, same release/acquire election pattern as
    // syncAndSendFinishLsa above, using ctaSync[0].
    __syncthreads();
    __shared__ bool shIsLastCta;
    if (threadId == 0) {
        const int prevDone = atomic_add_release_global(&ctaSync[0], 1);
        shIsLastCta = (prevDone + 1 == numSms);
        if (shIsLastCta) {
            memory_fence_gpu();
            ctaSync[0] = 0;
        }
    }
    __syncthreads();
    isLastCta = shIsLastCta;

LOW_LATENCY_COMBINE_LSA_RECV_1SIDED:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
        return;
    }
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        isLastCta = (smId == 0);
    }

    waitForAllRanksFinishLsa(
        threadId, numRanks, rankMask, activeRankMaskShared, recvFlagBuf, timeoutCycles, currRank, asyncErrorFlag,
        waitStats);
    completeFullLowLatencyEpoch(epochState, phases, isLastCta, threadId);

    constexpr int kMaxNumGroups = combine_smem::max_recv_groups(kElemBytes);
    constexpr int numDecodeWarps = combine_smem::recv_decode_warps(kHidden, kElemBytes);
    const int warpsPerRecvGroup = numDecodeWarps + combine_smem::kNumRecvTmaWarps;
    const int numGroups = min(kMaxNumGroups, (numThreads / kWarpSize) / warpsPerRecvGroup);
    const int decodeWarpIdx = __shfl_sync(0xffffffff, warpId % warpsPerRecvGroup, 0);
    const int groupIdx = __shfl_sync(0xffffffff, warpId / warpsPerRecvGroup, 0);
    EP_DEVICE_ASSERT(numTopk <= kWarpSize);
    EP_DEVICE_ASSERT(numGroups > 0);

    if (groupIdx < numGroups) {
        constexpr int kNumStages = combine_smem::kNumStages;
        constexpr int kNumTMABufferBytes = combine_smem::recv_tma_buffer_bytes(kHidden, kElemBytes, /*nvfp4=*/false);
        constexpr int kNumBF16PerWarpBytes = kWarpSize * kNumRecvUnrolls * kNumElemsPerInt4 * kElemBytes;
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * kElemBytes;

        const auto smemGroupBuffer = smemBuffer + kNumBytesPerGroup * groupIdx;
        auto fullBarriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes); });
        auto emptyBarriers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint64_t*>(
                smemGroupBuffer + i * kNumTMABufferBytes + combine_smem::kRecvTmaEmptyBarrierOffsetBytes);
        });
        auto tmaLdBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint8_t*>(
                smemGroupBuffer + i * kNumTMABufferBytes + combine_smem::kRecvTmaPayloadOffsetBytes);
        });
        auto tmaStBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(
                smemGroupBuffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });

        uint32_t tmaPhase = 0;
        EP_STATIC_ASSERT(kNumStages < kWarpSize, "Too many stages");
        if (decodeWarpIdx == numDecodeWarps) tmaPhase = (1 << kNumStages) - 1;

        if (decodeWarpIdx == numDecodeWarps and laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            mbarrier_init(emptyBarriers[laneId], numDecodeWarps);
        }
        syncSmGroup(groupIdx + 1, warpsPerRecvGroup * kWarpSize);

        int stageIdx = 0;
        EP_STATIC_ASSERT(kNumTopk <= kWarpSize, "numTopk must not exceed warp size");

        if (decodeWarpIdx == numDecodeWarps) {
            // TMA load warp: pulls each topk slot's contribution directly out of the owning
            // rank's inData window, using dstInfo's (dstRank, slotIdx) -- no re-derivation
            // from topk indices needed, and no local recvBuf staging to read from.
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens;
                 tokenIdx += numSms * numGroups) {
                for (int i = 0; i < numTopk; ++i) {
                    const size_t infoIdx = (static_cast<size_t>(tokenIdx) * numTopk + i) * 2;
                    const int dstRank = __ldg(dstInfo + infoIdx + 0);
                    if (dstRank < 0 or not isRankActiveLocal(activeRankMaskShared, dstRank)) continue;
                    const int slotIdx = __ldg(dstInfo + infoIdx + 1);

                    mbarrier_wait<true>(emptyBarriers[stageIdx], tmaPhase, stageIdx);
                    const size_t remoteSlot = static_cast<size_t>(currRank) * maxTokensPerRank + slotIdx;
                    const size_t remoteOffset = inDataWindowOffset + remoteSlot * hiddenBf16Int4 * sizeof(int4);
                    auto recvBufData =
                        static_cast<const uint8_t*>(ncclGetPeerPointer(inDataWindow, remoteOffset, dstRank));

                    if (elect_one_sync()) {
                        const int numTmaBytes = numDecodeWarps * kNumBF16PerWarpBytes;
                        tma_load_1d(tmaLdBuffers[stageIdx], recvBufData, fullBarriers[stageIdx], numTmaBytes);
                        mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numTmaBytes);
                    }
                    __syncwarp();
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
            }
        } else {
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens;
                 tokenIdx += numSms * numGroups) {
                combineReduceTokenLsa<kNumStages, kNumElemsPerInt4, kNumRecvUnrolls, kTokenDtype>(
                    tokenIdx, numTopk, laneId, decodeWarpIdx, kNumBF16PerWarpBytes, hiddenBf16Int4,
                    activeRankMaskShared, fullBarriers, emptyBarriers, tmaLdBuffers, tmaStBuffers, tmaPhase, stageIdx,
                    outData,
                    [&](int i) {
                        const size_t infoIdx = (static_cast<size_t>(tokenIdx) * numTopk + i) * 2;
                        return __ldg(dstInfo + infoIdx + 0);
                    });
            }
        }
    }

    // Every CTA that pulled from any peer has now finished decoding into outData. Elect the
    // last-finishing CTA, via ctaSync[1] (its OWN dedicated counter, never ctaSync[0]) to
    // broadcast "I'm done reading your inData" to every peer -- see ctaSync's doc comment above.
    // A relaxed add is enough: unlike a store, a load can't be "posted" -- this CTA's reads are
    // already fully retired the moment their values were consumed (into outData above), so there's
    // nothing to release/acquire here, only arrivals to count (see the flag send below for why no
    // fence is needed there either).
    __syncthreads();
    __shared__ bool shIsLastCtaRecv;
    if (threadId == 0) {
        const int prevDone = atomicAdd(&ctaSync[1], 1);
        shIsLastCtaRecv = (prevDone + 1 == numSms);
        if (shIsLastCtaRecv) {
            ctaSync[1] = 0;
        }
    }
    __syncthreads();
    if (shIsLastCtaRecv and warpId == 0) {
        // No fence needed: unlike the payload-then-flag stores elsewhere in this file, this flag
        // has no prior write to order against -- our reads from the peer's inData already
        // self-completed (see the comment above), and outData is irrelevant to the peer.
        for (int peer = laneId; peer < numRanks; peer += 32) {
            sendRankFinishFlagLsa(peer, currRank, readDoneBuf, readDoneOff, rankMask, windows, devComm);
        }
    }

    waitForAllReadDoneDistributedLsa(
        smId, numSms, threadId, numThreads, numRanks, rankMask, readDoneBuf, timeoutCycles, currRank,
        asyncErrorFlag);
}

// NCCL_EP_COMBINE_LSA auto mode (LlCombineKernel::kAutoRmLsa): picks 1-sided vs 2-sided on-device
// from dispatch's already-decided p.want1SidedBuf (dispatch itself compared every active peer's
// published count against combineLsaThreshold -- see computeAndPublishWant1SidedLsa in
// ll/ll_dispatch_lsa.cuh), then calls straight into the same impls above -- no JIT-generated
// string logic, this is a real compiled function. p.want1SidedBuf is a plain read: dispatch
// overwrites it fresh each round, so no reset is needed here.
template <int kHidden, int kNumTopk, int kNumMaxUnrolls, typename TopkIdxT, ncclDataType_t kTokenDtype, typename WeightT>
__device__ __forceinline__ void combine_kernel_impl_auto_rm_lsa(const combine_kernel_args_t<WeightT>& p) {
    const bool want1Sided = *p.want1SidedBuf != 0;
    if (want1Sided) {
        combine_kernel_impl_1sided_rm_lsa<kHidden, kNumTopk, kNumMaxUnrolls, kTokenDtype>(
            p.inData, p.inDataWindow, p.inDataWindowOffset, p.dstInfo, p.rankMask, p.asyncErrorFlag,
            p.outData, p.rdmaBuf, p.recvFlagOff, p.combineSync, p.lsaSignalingCleanCount, p.waitStats,
            p.epochState, p.payloadSlotStride, p.signalSlotStride, p.readDoneOffBase,
            p.numCombinedTokens, p.maxTokensPerRank, p.numExperts, p.currRank,
            p.numRanks, p.numWarpGroups, p.numWarpsPerGroup, p.phases, p.devComm, p.windows,
            p.timeoutCycles);
    } else {
        combine_kernel_impl_2sided_rm_lsa<kHidden, kNumTopk, kNumMaxUnrolls, TopkIdxT, kTokenDtype>(
            p.inData, p.srcInfo, static_cast<const TopkIdxT*>(p.inTopkIdx), p.rankMask, p.asyncErrorFlag,
            p.outData, p.rdmaBuf, p.sendOff, p.recvOff, p.recvFlagOff, p.combineSync,
            p.lsaSignalingCleanCount, p.waitStats, p.epochState, p.payloadSlotStride, p.signalSlotStride,
            p.numCombinedTokens, p.maxTokensPerRank, p.numExperts, p.currRank, p.numRanks,
            p.numWarpGroups, p.numWarpsPerGroup, p.phases, p.zeroCopy, p.devComm, p.windows,
            p.timeoutCycles);
    }
}

} // namespace ll

} // namespace nccl_ep
