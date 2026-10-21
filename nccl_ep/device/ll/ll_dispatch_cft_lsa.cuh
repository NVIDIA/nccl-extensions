/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <type_traits>
#include "nccl_ep.h"
#include "nccl_device.h"
#include "../device_primitives.cuh"
#include "../ll_ep_adapter.cuh"
#include "ll_common.cuh"
#include "ll_lsa_primitives.cuh"
#include "ll_mask.cuh"
#include "../ll_ep_cft.cuh"

namespace nccl_ep {
namespace ll {

// Rank-major CFT dispatch; protocol: docs/plc/298-ll-cft-lsa-dispatch/README.md.
constexpr int kLsaCftCtaShift = kLlLsaCftCtaShift;  // packed word: arrivals << 20 | tokens
constexpr unsigned kLsaCftTokenMask = (1u << kLsaCftCtaShift) - 1u;
constexpr int kLsaCftArrivalBits = kLlLsaCftArrivalBits;
constexpr unsigned kLsaCftArrivalMask = (1u << kLsaCftArrivalBits) - 1u;
constexpr unsigned kLsaCftCountedFlag = 1u << 31;  // set by the publishing CTA when this source counts its payload
constexpr int kLsaCftMaxRanks = 72;                 // one NVL72 domain
constexpr int kLsaCftMaxTokWords = kLlLsaCftMaxTokensPerCta / 64;
constexpr int kLsaCftGroupBarrierBase = 2;            // named barrier id of receive group g = base + g
static_assert(kLsaCftMaxTokWords * 64 == kLlLsaCftMaxTokensPerCta, "token mask words");

// Static shared state of the LSA-CFT send phase.
template <int kMaskWords>
struct LsaCftSmem {
    static_assert(kMaskWords == 1 || kMaskWords == 8, "supported CFT token mask widths");
    unsigned long long tokMask[kLsaCftMaxRanks][kMaskWords];  // bit j: token j of this CTA goes to rank r
    int base[kLsaCftMaxRanks];   // first slot of this CTA at destination r
    int cnt[kLsaCftMaxRanks];    // tokens of this CTA to destination r
    int off[kLsaCftMaxRanks];    // header-batch offset (in headers) of destination r
};

// Participants: complete helper warps. Deduplicate every token of this CTA by destination
// rank. list[j*K + pos] = rank, nd[j] = number of distinct destinations, and
// tokMask[r] gets bit j. tokMask must be zero on entry.
template <int kNumTopk, typename TopkIdxT, int kMaskWords>
__forceinline__ __device__ void lsaCftPreprocess(
    const TopkIdxT* inTopkIdx, int numTokens, int numSms, int smId, int nTok, int numLocalExperts,
    int helperIdx, int numHelpers, int laneId, int* list, int* nd, LsaCftSmem<kMaskWords>& cs) {
    for (int j = helperIdx; j < nTok; j += numHelpers) {
        const int tok = smId + j * numSms;
        int rank = -1;
        if (laneId < kNumTopk) {
            const int e = static_cast<int>(ld_read_only_global(inTopkIdx + static_cast<size_t>(tok) * kNumTopk + laneId));
            rank = getExpertRankIdx(e, numLocalExperts);
        }
        const unsigned same = __match_any_sync(0xffffffffu, rank);
        const bool win = rank >= 0 && laneId == (__ffs(same) - 1);
        const unsigned winMask = __ballot_sync(0xffffffffu, win);
        if (win) {
            const int pos = __popc(winMask & ((1u << laneId) - 1u));
            list[j * kNumTopk + pos] = rank;
            atomic_or_relaxed_shared(&cs.tokMask[rank][::nccl_ep::bit_word(j)], 1ull << (j & 63));
        }
        if (laneId == 0) {
            nd[j] = __popc(winMask);
        }
    }
}

// Participants: complete warp 0, after barrier A publishes helper writes.
// Owns base/cnt/off; the trailing warp barrier publishes them within warp 0.
// Helper header writers require the additional CTA reservation barrier.
template <int kMaskWords>
__forceinline__ __device__ void lsaCftReserve(
    LsaCftSmem<kMaskWords>& cs, int* rankSentCnt, int numRanks, int laneId) {
    // Warp-uniform loop over 32-rank chunks: every lane joins every shuffle.
    int carry = 0;  // header-batch offset of the first rank of this chunk
    for (int q = 0; q < numRanks; q += 32) {
        const int r = q + laneId;
        const bool active = r < numRanks;
        int cnt = 0;
        if (active) {
#pragma unroll
            for (int w = 0; w < kMaskWords; ++w) {
                cnt += __popcll(cs.tokMask[r][w]);
            }
        }
        // Inclusive scan of cnt within the chunk (for the header-batch offsets).
        int incl = cnt;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, incl, d);
            if (laneId >= d) {
                incl += up;
            }
        }
        const int chunkTotal = __shfl_sync(0xffffffffu, incl, 31);
        if (active) {
            cs.cnt[r] = cnt;
            cs.off[r] = carry + incl - cnt;
            cs.base[r] = cnt != 0 ? atomic_add_relaxed_global(rankSentCnt + r, cnt) : 0;
        }
        carry += chunkTotal;
    }
    __syncwarp();
}

// Participants: the complete elected publisher warp. It has acquired all
// reservations and sentinel clears; one publisher owns each remote word.
// Uncounted sources must also acquire completed sends before publishing.
__forceinline__ __device__ void lsaCftPublishCounts(
    const int* rankSentCnt, int numSms, int numRanks, int currRank, int* recvCntBuf, size_t recvCntOff,
    const ncclWindow_t* windows, ncclDevComm* devComm, int* asyncErrorFlag, int laneId, bool countedPayload) {
    memory_fence_release_sys();
    const unsigned head = (countedPayload ? kLsaCftCountedFlag : 0u) | (1u << kLsaCftCtaShift);
#pragma unroll 1
    for (int r = laneId; r < numRanks; r += 32) {
        if (r == currRank) {
            continue;
        }
        const unsigned total = static_cast<unsigned>(rankSentCnt[r]);
        const auto p2p = ncclGetP2pPtr(reinterpret_cast<uint64_t>(recvCntBuf + currRank),
                                       recvCntOff + static_cast<size_t>(currRank) * sizeof(int),
                                       currRank, r, windows, devComm);
        if (p2p == 0) {
            cftReportSendFailure(asyncErrorFlag, currRank, r, "count publish (no p2p mapping)");
        } else {
            // One publisher owns this remote word; the fence above supplies release ordering.
            st_relaxed_sys_global(reinterpret_cast<int*>(p2p), static_cast<int>(head | total));
        }
    }
    __syncwarp();
    if (laneId == 0) {
        // All sentinel clears are complete; self also waits for every partial count.
        atomic_add_release_global(recvCntBuf + currRank, numSms << kLsaCftCtaShift);
    }
}

// Participants: complete warp 0 of every CTA after reservation. The last
// release RMW elects the publisher; its acquire joins reservations and clears.
// Counted sources may publish early: their byte counters gate payload reads.
__forceinline__ __device__ void lsaCftPublishCountedEarly(
    const int* rankSentCnt, int* reserveDone, int numSms, int numRanks, int currRank, int* recvCntBuf,
    size_t recvCntOff, const ncclWindow_t* windows, ncclDevComm* devComm, int* asyncErrorFlag, int laneId) {
    int prevDone = 0;
    if (laneId == 0) {
        prevDone = atomic_add_release_global(reserveDone, 1);
    }
    prevDone = __shfl_sync(0xffffffffu, prevDone, 0);
    if (prevDone + 1 != numSms) {
        return;
    }
    memory_fence_gpu();
    __syncwarp();
    lsaCftPublishCounts(rankSentCnt, numSms, numRanks, currRank, recvCntBuf, recvCntOff,
                        windows, devComm, asyncErrorFlag, laneId, /*countedPayload=*/true);
    if (laneId == 0) {
        *reserveDone = 0;
    }
}

// Participants: any lane after reservation state has been acquired; read-only.
// Slot of token j (CTA-local index) at destination r: base[r] plus the number
// of earlier tokens of this CTA that also go to r.
template <int kMaskWords>
__forceinline__ __device__ int lsaCftSlot(const LsaCftSmem<kMaskWords>& cs, int r, int j) {
    if constexpr (kMaskWords == 1) {
        return cs.base[r] + __popcll(cs.tokMask[r][0] & ((1ull << (j & 63)) - 1ull));
    } else {
        int before = 0;
        const int w = ::nccl_ep::bit_word(j);
#pragma unroll
        for (int k = 0; k < kLsaCftMaxTokWords; ++k) {
            if (k < w) {
                before += __popcll(cs.tokMask[r][k]);
            }
        }
        before += __popcll(cs.tokMask[r][w] & ((1ull << (j & 63)) - 1ull));
        return cs.base[r] + before;
    }
}

// Participants: complete warp 0 for lsaCftSendToken. All lanes stage the header into the tile (and
// the self slot), lane 0 waits for the TMA load, prefetches the next token
// and issues the puts from the precomputed destination list.
#if NCCL_EP_CFT_DEVICE_SUPPORTED
// Lane-zero-only uncounted zero-copy issue. Reservation state and rank lists
// are immutable after the existing CTA handoff. Compute each original slot
// directly, so this path has no shared descriptor producer/consumer handoff.
template <int kNumTopk, class Geometry, int kMaskWords>
__forceinline__ __device__ void lsaCftPutTokenScalar(
    CftSendContext<Geometry>& ctx, const cft_dispatch_args_t& cft, const CftSelfDestination& self,
    const CftMessageLayout& msg, const LsaCftSmem<kMaskWords>& cs, const int* list,
    const uint8_t* tile, int j, int nDest, int currRank) {
    static_assert(Geometry::kZeroCopy, "Scalar issue is only used for zero-copy payloads");
    // Keep the original per-token proxy release before any async source read.
    nccl_ep::cft::memFence(ncclCoopThread{}, cuda::memory_order_release, nccl_ep::cft::ProxyType::Generic,
                           nccl_ep::cft::ProxyType::Fabric, nccl_ep::cft::FenceScope::Cta);
#pragma unroll 1
    for (int d = 0; d < nDest; ++d) {
        const int dstRank = list[j * kNumTopk + d];
        const int slotIdx = lsaCftSlot(cs, dstRank, j);
        if (dstRank == currRank) {
            nccl_ep::cft::cft_tma_s2g(
                self.payloadBase + static_cast<size_t>(slotIdx) * msg.payloadBytes,
                tile + Geometry::kPayloadOffset, static_cast<unsigned>(msg.hiddenBytes));
            nccl_ep::cft::cft_bulk_commit();
        } else {
            const uint64_t payloadOff = cft.outputOffset +
                (static_cast<uint64_t>(currRank) * msg.maxTokensPerRank + slotIdx) * msg.hiddenBytes;
            ctx.fabric.put(ncclCoopThread{}, ctx.leMirror[dstRank], payloadOff,
                           tile + Geometry::kPayloadOffset, static_cast<unsigned>(msg.hiddenBytes));
        }
    }
}

// Participants: complete warp 0. Owns this token's descriptor scratch and tile header.
// The previous source reads are complete; warp joins publish headers before issue.
template <int kNumTopk, class Geometry, typename TopkIdxT, int kMaskWords>
__forceinline__ __device__ void prepareTokenHeadersCftLsa(
    const CftSelfDestination& self, const CftMessageLayout& msg,
    const LsaCftSmem<kMaskWords>& cs, const int* list, int* dscratch,
    uint8_t* tile, uint8_t* hdrBatch, int j, int nDest, int tok,
    const TopkIdxT* inTopkIdx, const float* inTopkWeights, int currRank, int laneId) {
    // Destination list -> (rank, slot) pairs for lane 0; find the self slot.
    int selfSlot = -1;
    if (laneId < nDest) {
        const int r = list[j * kNumTopk + laneId];
        const int slot = lsaCftSlot(cs, r, j);
        dscratch[2 * laneId] = r;
        dscratch[2 * laneId + 1] = slot;
        if (r == currRank) {
            selfSlot = slot;
        }
    }
    selfSlot = __reduce_max_sync(0xffffffffu, selfSlot);

    // Header straight into the tile front (the tile's previous reads were
    // drained by cftReleaseTile two tokens ago; the TMA load fills the
    // payload region only).
    auto* hdr = reinterpret_cast<DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(tile);
    if (laneId == 0) {
        hdr->token_id = tok;
    }
    if (laneId < kNumTopk) {
        const int e = static_cast<int>(ld_read_only_global(inTopkIdx + static_cast<size_t>(tok) * kNumTopk + laneId));
        hdr->rtr[laneId].expert_id = static_cast<uint16_t>(e);
        hdr->rtr[laneId].topk_weight = ld_read_only_global(inTopkWeights + static_cast<size_t>(tok) * kNumTopk + laneId);
    }
    __syncwarp();
    const int hdrInt4 = static_cast<int>(msg.hdrBytes / sizeof(int4));
    if (selfSlot >= 0) {
        const int4* hdrSrc = reinterpret_cast<const int4*>(tile);
        int4* selfHdrDst = reinterpret_cast<int4*>(self.recvBase + static_cast<size_t>(selfSlot) * msg.hdrBytes);
        for (int i = laneId; i < hdrInt4; i += 32) {
            st_na_global(selfHdrDst + i, hdrSrc[i]);
        }
    }
    if constexpr (Geometry::kZeroCopy) {
        // Copy the header into the batch slot of every remote destination:
        // position = off[r] + (slot - base[r]) in header units.
        const int4* hdrSrc = reinterpret_cast<const int4*>(tile);
        for (int i = laneId; i < nDest * hdrInt4; i += 32) {
            const int d = i / hdrInt4, q = i - d * hdrInt4;
            const int r = dscratch[2 * d];
            if (r == currRank) {
                continue;
            }
            const int pos = cs.off[r] + (dscratch[2 * d + 1] - cs.base[r]);
            reinterpret_cast<int4*>(hdrBatch + static_cast<size_t>(pos) * msg.hdrBytes)[q] = hdrSrc[q];
        }
    }
    __syncwarp();
}

template <int kNumTopk, bool kCountedPayload, class Geometry, typename TopkIdxT, int kMaskWords>
__forceinline__ __device__ void lsaCftSendToken(
    CftSendContext<Geometry>& ctx, const cft_dispatch_args_t& cft, const CftSelfDestination& self,
    const CftMessageLayout& msg, const LsaCftSmem<kMaskWords>& cs, const int* list, const int* nd, uint8_t* hdrBatch, int j,
    int nTok, int smId, int numSms, const TopkIdxT* inTopkIdx, const float* inTopkWeights, const int4* inDataInt4,
    size_t hiddenInt4, int currRank, int laneId, uint64_t timeoutCycles, int* asyncErrorFlag) {
    const int tok = smId + j * numSms;
    const int rslot = j % Geometry::kRingDepth;
    int* dscratch = ctx.destScratch + (j & 1) * (kCftDestScratchInts / 2);
    uint8_t* tile = ctx.tile(rslot);
    const int nDest = nd[j];

    if constexpr (Geometry::kZeroCopy && !kCountedPayload) {
        // Helpers own routing headers. Only lane zero consumes immutable
        // list/mask/base state below; this path has no descriptor scratch writes.
    } else {
        prepareTokenHeadersCftLsa<kNumTopk, Geometry>(
            self, msg, cs, list, dscratch, tile, hdrBatch, j, nDest, tok,
            inTopkIdx, inTopkWeights, currRank, laneId);
    }

    const uint64_t byteCntrOff = cft.parityOff + cft.bytesCntrBaseOff + static_cast<uint64_t>(currRank) * 256;
    uint32_t ringPhase = static_cast<uint32_t>((j / Geometry::kRingDepth) & 1);
    if (laneId == 0) {
        const int ok = cftAwaitStagedPayload(ctx, rslot, &ringPhase, timeoutCycles, asyncErrorFlag, currRank);
        if (ok && j + 1 < nTok) {
            const int nextTok = smId + (j + 1) * numSms;
            cftStagePayload(ctx, rslot ^ 1, inDataInt4 + static_cast<size_t>(nextTok) * hiddenInt4, msg);
        }
        if (ok && nDest > 0) {
            if constexpr (Geometry::kZeroCopy && !kCountedPayload) {
                lsaCftPutTokenScalar<kNumTopk>(ctx, cft, self, msg, cs, list, tile, j, nDest, currRank);
            } else {
                cftPutToken(ctx, cft, self, msg, dscratch, tile, (1u << nDest) - 1u, byteCntrOff, currRank,
                            /*skipHdr=*/true, /*uncountedPayload=*/!kCountedPayload);
            }
            cftReleaseTile(ctx);
            cftSettleIfOverBudget(ctx, timeoutCycles, asyncErrorFlag, currRank);
        }
    }
}

// Participants: complete helper warps; each warp owns disjoint token headers.
// The reservation barrier publishes base/count/offset and token masks to
// helpers. Each helper warp owns complete token headers, so no two warps
// write one header. No payload tile or sender descriptor scratch is touched.
template <int kNumTopk, typename TopkIdxT, int kMaskWords>
__forceinline__ __device__ void lsaCftPrepareHeaders(
    const CftSelfDestination& self, const CftMessageLayout& msg, const LsaCftSmem<kMaskWords>& cs,
    const int* list, const int* nd, uint8_t* hdrBatch, int nTok, int smId, int numSms,
    const TopkIdxT* inTopkIdx, const float* inTopkWeights, int currRank,
    int helperIdx, int numHelpers, int laneId) {
    for (int j = helperIdx; j < nTok; j += numHelpers) {
        const int tok = smId + j * numSms;
        int expert = 0;
        float weight = 0;
        if (laneId < kNumTopk) {
            expert = static_cast<int>(ld_read_only_global(inTopkIdx + static_cast<size_t>(tok) * kNumTopk + laneId));
            weight = ld_read_only_global(inTopkWeights + static_cast<size_t>(tok) * kNumTopk + laneId);
        }
        const int nDest = nd[j];
        for (int d = 0; d < nDest; ++d) {
            const int r = list[j * kNumTopk + d];
            const int slot = lsaCftSlot(cs, r, j);
            const int pos = cs.off[r] + slot - cs.base[r];
            uint8_t* dst = r == currRank
                ? self.recvBase + static_cast<size_t>(slot) * msg.hdrBytes
                : hdrBatch + static_cast<size_t>(pos) * msg.hdrBytes;
            auto* hdr = reinterpret_cast<DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(dst);
            if (laneId == 0) {
                hdr->token_id = tok;
            }
            if (laneId < kNumTopk) {
                hdr->rtr[laneId].expert_id = static_cast<uint16_t>(expert);
                hdr->rtr[laneId].topk_weight = weight;
            }
        }
    }
}

// Participants: lane 0, after all header writers join (CTA barrier for
// helper-prepared headers; warp sync otherwise). Each destination gets one
// put covering this CTA's contiguous slots. The batch stays live until drain.
// Drain before issue only when the transaction-tracking budget could overflow.
template <bool kCounted, class Geometry, int kMaskWords>
__forceinline__ __device__ void lsaCftPutHeaderBatch(
    CftSendContext<Geometry>& ctx, const cft_dispatch_args_t& cft, const CftMessageLayout& msg, const LsaCftSmem<kMaskWords>& cs,
    const uint8_t* hdrBatch, int numRanks, int currRank, uint64_t timeoutCycles, int* asyncErrorFlag) {
    unsigned units = 0;
    for (int r = 0; r < numRanks; ++r) {
        if (r != currRank) {
            units += static_cast<unsigned>(cs.cnt[r] * msg.hdrBytes / 16);
        }
    }
    if (units == 0) {
        return;
    }
    // Uncounted sources publish only after both transfers complete.
    if (ctx.fabric.pendingTxCount + units > Geometry::kTxSettleUnits) {
        cftDrainPuts(ctx, timeoutCycles, asyncErrorFlag, currRank);
    }
    nccl_ep::cft::memFence(ncclCoopThread{}, cuda::memory_order_release, nccl_ep::cft::ProxyType::Generic,
                           nccl_ep::cft::ProxyType::Fabric, nccl_ep::cft::FenceScope::Cta);
    const uint64_t byteCntrOff = cft.parityOff + cft.bytesCntrBaseOff + static_cast<uint64_t>(currRank) * 256;
    const uint64_t regionOff = cft.parityOff + static_cast<uint64_t>(currRank) * cft.srcRankRegionBytes;
    static_assert(Geometry::kZeroCopy, "Only zero-copy needs a separate header batch");
#pragma unroll 1
    for (int r = 0; r < numRanks; ++r) {
        if (r == currRank || cs.cnt[r] == 0) {
            continue;
        }
        const uint64_t hdrOff = regionOff + static_cast<uint64_t>(cs.base[r]) * msg.hdrBytes;
        const uint8_t* src = hdrBatch + static_cast<size_t>(cs.off[r]) * msg.hdrBytes;
        const unsigned bytes = static_cast<unsigned>(cs.cnt[r] * msg.hdrBytes);
        if constexpr (kCounted) {
            ctx.fabric.putCounted(ncclCoopThread{}, ctx.leMirror[r], hdrOff, byteCntrOff, src, bytes);
        } else {
            ctx.fabric.put(ncclCoopThread{}, ctx.leMirror[r], hdrOff, src, bytes);
        }
    }
    ctx.fabric.submit(ncclCoopThread{});
}
#endif  // NCCL_EP_CFT_DEVICE_SUPPORTED

// Participants: entire CTA after barrier B. Thread 0 releases its self partial
// count before entering; the last warp independently joins the source census.
// The winner acquires completed sends/clears, publishes uncounted counts and
// resets reservations. The final CTA barrier joins both warp roles and shares
// the election; the self receiver separately waits for every partial count.
template <bool kCountedPayload>
__forceinline__ __device__ bool lsaCftFinishAndPublish(
    int warpId, int numWarps, int laneId, int* ctaDone, int numSms, int numRanks, int* rankSentCnt,
    int currRank, int* recvCntBuf, size_t recvCntOff, const ncclWindow_t* windows, ncclDevComm* devComm,
    int* asyncErrorFlag) {
    __shared__ bool shIsLastCtaCft;
    if (warpId == numWarps - 1) {
        int prevDone = 0;
        if (laneId == 0) {
            prevDone = atomic_add_release_global(ctaDone, 1);
        }
        prevDone = __shfl_sync(0xffffffff, prevDone, 0);
        const bool isLastCta = (prevDone + 1 == numSms);
        if (laneId == 0) {
            shIsLastCtaCft = isLastCta;
        }
        if (isLastCta) {
            memory_fence_gpu();
            __syncwarp();
            if constexpr (!kCountedPayload) {
                lsaCftPublishCounts(rankSentCnt, numSms, numRanks, currRank, recvCntBuf, recvCntOff,
                                    windows, devComm, asyncErrorFlag, laneId, /*countedPayload=*/false);
            }
#pragma unroll 1
            for (int r = laneId; r < numRanks; r += 32) {
                rankSentCnt[r] = 0;
            }
            if (laneId == 0) {
                *ctaDone = 0;
            }
        }
    }
    __syncthreads();
    return shIsLastCtaCft;
}

// Participants: any lane after receive readiness; reads the acquired source word.
__forceinline__ __device__ int lsaCftDecodeCount(const int* recvCntBuf, int srcRank) {
    return static_cast<int>(static_cast<unsigned>(recvCntBuf[srcRank]) & kLsaCftTokenMask);
}


#if NCCL_EP_CFT_DEVICE_SUPPORTED
// Participants: entire CTA, including RECV-only launches as in the original
// kernel. Warp 0 owns transport initialization; all threads clear disjoint
// token-mask words. The final CTA barrier publishes both before SEND begins.
// Buffer carving uses the unchanged host allocation. Only the existing
// CftSendContext carries transport state; list/count/header pointers describe
// the additional routing workspace, whose lifetime spans the SEND phase.
template <int kNumTopk, class Geometry, int kMaskWords>
__forceinline__ __device__ void initDispatchCftLsa(
    const dispatch_kernel_args_t& args, const cft_dispatch_args_t& cft,
    const CftMessageLayout& msg, void* recvBuf, CftSendContext<Geometry>& sender,
    CftSelfDestination& self, LsaCftSmem<kMaskWords>& cs,
    int*& list, int*& nd, uint8_t*& hdrBatch,
    int threadId, int warpId, int laneId, int numSms, int numWarps, int nTokMax) {
    EP_DEVICE_ASSERT(cft.peerLeIds != nullptr && cft.recvByteCounters != nullptr && args.rankMask == nullptr &&
                     (Geometry::kZeroCopy || args.recvDataWindow == ncclWindow_t{}));
    EP_DEVICE_ASSERT(args.numRanks <= kLsaCftMaxRanks && nTokMax <= kLsaCftMaxTokWords * 64 &&
                     2 * numSms < (1 << kLsaCftArrivalBits) &&
                     args.maxTokensPerRank < (1 << kLsaCftCtaShift));
    if constexpr (kMaskWords == 1) {
        EP_DEVICE_ASSERT(nTokMax <= 64 && args.numTokens >= 0 && args.numTokens <= args.maxTokensPerRank);
    }
    extern __shared__ __align__(128) uint8_t cft_dyn_smem_[];
    __shared__ nccl_ep::cft::Smem cft_data_smem;
    __shared__ __align__(8) uint64_t cft_tma_mbars[Geometry::kRingDepth];
    sender = cftCarveSendContext<Geometry>(cft_dyn_smem_, &cft_data_smem, cft_tma_mbars, args.numRanks);
    const size_t leMirrorBytes = ::nccl_ep::align<size_t>(static_cast<size_t>(args.numRanks) * sizeof(uint32_t), 16);
    list = reinterpret_cast<int*>(cft_dyn_smem_ + Geometry::kTilesBytes + leMirrorBytes + kCftDestScratchBytes);
    nd = list + nTokMax * kNumTopk;
    hdrBatch = reinterpret_cast<uint8_t*>(
        ::nccl_ep::align<uintptr_t>(reinterpret_cast<uintptr_t>(nd + nTokMax), 16));
    self = cftSelfDestination<Geometry::kZeroCopy>(
        recvBuf, args.outDataBuf, args.currRank, args.numRanks,
        static_cast<size_t>(args.maxTokensPerRank) * msg.msgBytes,
        args.maxTokensPerRank, msg.hdrBytes, msg.hiddenBytes);
    for (int i = threadId; i < kLsaCftMaxRanks * kMaskWords; i += numWarps * 32) {
        (&cs.tokMask[0][0])[i] = 0ull;
    }
    if (warpId == 0) {
        cftInitSender(sender, cft.peerLeIds, args.numRanks, laneId);
    }
    __syncthreads();
}

// Participants: complete helper warps (all warps except warp 0). Each warp
// owns strided token lists; masks are atomic ORs. Sentinel rows are disjoint
// across all helper threads/CTAs. CTA 0 also clears the next count bank.
// The caller's barrier A joins these writes with warp 0 before reservation.
template <int kNumTopk, typename TopkIdxT, int kMaskWords>
__forceinline__ __device__ void preprocessRoutingCftLsa(
    const dispatch_kernel_args_t& args, int smId, int warpId, int laneId,
    int numSms, int numWarps, int numLocalExperts, int nTok, int* list, int* nd,
    int* nextRecvCntBuf, LsaCftSmem<kMaskWords>& cs) {
    lsaCftPreprocess<kNumTopk>(
        static_cast<const TopkIdxT*>(args.inTopkIdx), args.numTokens, numSms, smId, nTok,
        numLocalExperts, warpId - 1, numWarps - 1, laneId, list, nd, cs);
    const size_t totalTopkEntries = static_cast<size_t>(args.numRanks) * args.maxTokensPerRank * kNumTopk;
    const int helperThreads = (numWarps - 1) * 32;
    const int helperThread = (warpId - 1) * 32 + laneId;
    for (size_t i = static_cast<size_t>(smId) * helperThreads + helperThread; i < totalTopkEntries;
         i += static_cast<size_t>(numSms) * helperThreads) {
        args.outRecvTopkIdx[i] = -1;
    }
    EP_DEVICE_ASSERT(numSms > 1);
    if (smId == 0) {
        cleanNextRecvCntBufStrided(
            nextRecvCntBuf, args.lsaSignalingCleanCount, helperThread, helperThreads);
    }
}

// Participants: complete warp 0. Header helpers may run concurrently; they
// never touch payload tiles or descriptor scratch. Per-token source-read waits
// protect the two-slot TMA ring; capacity drains remain inside lsaCftSendToken.
// The final warp barrier joins lane 0's stream before the header-batch handoff.
template <int kNumTopk, bool kCountedPayload, typename TopkIdxT, class Geometry, int kMaskWords>
__forceinline__ __device__ void streamPayloadsCftLsa(
    const dispatch_kernel_args_t& args, const cft_dispatch_args_t& cft,
    CftSendContext<Geometry>& sender, const CftSelfDestination& self, const CftMessageLayout& msg,
    const LsaCftSmem<kMaskWords>& cs, const int* list, const int* nd, uint8_t* hdrBatch,
    int nTok, int smId, int numSms, int laneId) {
    for (int j = 0; j < nTok; ++j) {
        lsaCftSendToken<kNumTopk, kCountedPayload>(
            sender, cft, self, msg, cs, list, nd, hdrBatch, j, nTok, smId, numSms,
            static_cast<const TopkIdxT*>(args.inTopkIdx), args.inTopkWeights,
            static_cast<const int4*>(args.inData), msg.hiddenBytes / sizeof(int4),
            args.currRank, laneId, args.timeoutCycles, args.asyncErrorFlag);
    }
    __syncwarp();
}

// Participants: entire CTA. Warp 0 streams payloads; helper warps preprocess
// routes/clears and, for uncounted zero-copy, prepare independent headers.
// Barrier A joins preprocessing before reservation. The two conditional CTA
// barriers publish reservations to header helpers and join their stores before
// header puts, preserving their overlap with payload streaming.
// Returns the source-wide last-SEND-CTA election after final publication/reset.
template <int kNumTopk, bool kCountedPayload, typename TopkIdxT, class Geometry, int kMaskWords>
__forceinline__ __device__ bool dispatchSendPhaseCftLsa(
    const dispatch_kernel_args_t& args, const cft_dispatch_args_t& cft,
    CftSendContext<Geometry>& sender, const CftSelfDestination& self, const CftMessageLayout& msg,
    LsaCftSmem<kMaskWords>& cs, int* list, int* nd, uint8_t* hdrBatch,
    int* recvCntBuf, int* nextRecvCntBuf, size_t recvCntOff,
    int smId, int warpId, int laneId, int numSms, int numWarps, int numLocalExperts, int nTok) {
    if (warpId == 0) {
        if (laneId == 0) {
            cftCheckBulkAlignment(self, msg);
            if constexpr (Geometry::kZeroCopy) {
                EP_DEVICE_ASSERT(args.recvDataWindow != ncclWindow_t{} && (cft.outputOffset % 16) == 0);
            }
            if (nTok > 0) {
                cftStagePayload(sender, 0, static_cast<const int4*>(args.inData) +
                                static_cast<size_t>(smId) * (msg.hiddenBytes / sizeof(int4)), msg);
            }
        }
    } else {
        preprocessRoutingCftLsa<kNumTopk, TopkIdxT>(
            args, smId, warpId, laneId, numSms, numWarps, numLocalExperts, nTok, list, nd, nextRecvCntBuf, cs);
    }
    __syncthreads();  // barrier A

    if (warpId == 0) {
        lsaCftReserve(cs, args.rankSentCnt, args.numRanks, laneId);
        if constexpr (kCountedPayload) {
            lsaCftPublishCountedEarly(args.rankSentCnt, args.rankDone + 1, numSms, args.numRanks, args.currRank,
                                      recvCntBuf, recvCntOff, args.windows, args.devComm, args.asyncErrorFlag, laneId);
        }
    }
    if constexpr (Geometry::kZeroCopy && !kCountedPayload) {
        __syncthreads();  // reservation handoff to header helpers
        if (warpId != 0) {
            lsaCftPrepareHeaders<kNumTopk>(
                self, msg, cs, list, nd, hdrBatch, nTok, smId, numSms,
                static_cast<const TopkIdxT*>(args.inTopkIdx), args.inTopkWeights, args.currRank,
                warpId - 1, numWarps - 1, laneId);
        }
    }
    if (warpId == 0) {
        streamPayloadsCftLsa<kNumTopk, kCountedPayload, TopkIdxT>(
            args, cft, sender, self, msg, cs, list, nd, hdrBatch, nTok, smId, numSms, laneId);
    }
    if constexpr (Geometry::kZeroCopy && !kCountedPayload) {
        __syncthreads();  // header stores precede Generic->Fabric release
    }
    // Warp 0 lane 0 completes headers, self TMA stores, and Fabric writes.
    // Barrier B joins that completion before self readiness and final publication.
    // Transport timeouts remain fatal before either readiness publication.
    if (warpId == 0) {
        if constexpr (Geometry::kZeroCopy) {
            if (laneId == 0) {
                lsaCftPutHeaderBatch<kCountedPayload>(sender, cft, msg, cs, hdrBatch, args.numRanks,
                                                    args.currRank, args.timeoutCycles, args.asyncErrorFlag);
            }
        }
        if (laneId == 0) {
            cftDrainSender(sender, args.timeoutCycles, args.asyncErrorFlag, args.currRank);
            if constexpr (!kCountedPayload) {
                nccl_ep::cft::memFence(ncclCoopThread{}, cuda::memory_order_release,
                                       nccl_ep::cft::ProxyType::Fabric, nccl_ep::cft::ProxyType::Generic,
                                       nccl_ep::cft::FenceScope::Sys);
            }
        }
    }
    __syncthreads();  // barrier B
    if (threadIdx.x == 0) {
        // One partial arrival and this CTA's self tokens. The publisher adds
        // numSms sentinel-clear arrivals, making self complete at 2*numSms.
        atomic_add_release_global(recvCntBuf + args.currRank,
                                  static_cast<int>((1u << kLsaCftCtaShift) | static_cast<unsigned>(cs.cnt[args.currRank])));
    }
    return lsaCftFinishAndPublish<kCountedPayload>(
        warpId, numWarps, laneId, args.rankDone, numSms, args.numRanks, args.rankSentCnt,
        args.currRank, recvCntBuf, recvCntOff, args.windows, args.devComm, args.asyncErrorFlag);
}

// Participants: the first lane of an active receive group. Owns its byte target.
// Acquires source readiness and local sentinel clears; uncounted targets stay unchanged.
__forceinline__ __device__ void awaitSourceReadyCftLsa(
    const dispatch_kernel_args_t& args, const cft_dispatch_args_t& cft,
    int* recvCntBuf, size_t numBytesPerMsg, int numSms, int numLocalExperts,
    int srcRank, int rankLaneIdx) {
    const bool selfSrc = srcRank == args.currRank;
    const unsigned want = selfSrc ? 2u * static_cast<unsigned>(numSms) : 1u;
    const auto startTime = clock64();
    unsigned w;
    while ((((w = ld_relaxed_sys_global(reinterpret_cast<const uint32_t*>(recvCntBuf + srcRank))) >>
             kLsaCftCtaShift) & kLsaCftArrivalMask) != want) {
        if (static_cast<uint64_t>(clock64() - startTime) > args.timeoutCycles) {
            cftReportRecvFailure(args.asyncErrorFlag, args.currRank, srcRank, -1, "dispatch count (packed word)");
            break;
        }
    }
    if (!selfSrc) {
        // The count publisher marks completion of every local CTA's sentinel clears.
        while (((ld_relaxed_sys_global(reinterpret_cast<const uint32_t*>(recvCntBuf + args.currRank)) >>
                 kLsaCftCtaShift) & kLsaCftArrivalMask) < static_cast<unsigned>(numSms)) {
            if (static_cast<uint64_t>(clock64() - startTime) > args.timeoutCycles) {
                cftReportRecvFailure(args.asyncErrorFlag, args.currRank, args.currRank, -1,
                                     "dispatch count (own sentinel mark)");
                break;
            }
        }
        (void)ld_acquire_sys_global(recvCntBuf + args.currRank);
    }
    (void)ld_acquire_sys_global(recvCntBuf + srcRank);
    compiler_fence();
    if (!selfSrc) {
        const int n = static_cast<int>(w & kLsaCftTokenMask);
        // An uncounted source's packed count is its completion signal.
        // Leave its byte target unchanged so later counted rounds stay aligned.
        const bool srcCounted = (w & kLsaCftCountedFlag) != 0u;
        if (srcCounted) {
            cftAwaitPayload(cftRecvChannel(cft, srcRank, rankLaneIdx, numLocalExperts), n, numBytesPerMsg,
                            args.timeoutCycles, args.asyncErrorFlag, args.currRank, srcRank, rankLaneIdx);
        } else {
            nccl_ep::cft::memFence(ncclCoopThread{}, cuda::memory_order_acquire,
                                   nccl_ep::cft::ProxyType::Fabric, nccl_ep::cft::ProxyType::Generic,
                                   nccl_ep::cft::FenceScope::Sys);
        }
    }
}

// Participants: entire CTA, even when a receive group has no assigned expert.
// Each active group's first lane owns its source/group byte target and polls
// counts. Sources choose completion independently via their packed flag.
// Named barriers release only that group's warps; with >14 groups the final
// CTA-wide barrier MUST include inactive groups. Compiler fences keep generic
// payload/header loads after the acquire and the group/CTA handoff.
__forceinline__ __device__ void syncAndRecvCountsCftLsa(
    const dispatch_kernel_args_t& args, const cft_dispatch_args_t& cft,
    int* recvCntBuf, size_t numBytesPerMsg, int laneId, int numSms, int numLocalExperts,
    int warpGroupId, int subWarpId, int responsibleExpertIdx) {
    const bool perGroupRelease = args.numWarpGroups + kLsaCftGroupBarrierBase <= 16;
    if (responsibleExpertIdx < args.numExperts) {
        const int srcRank = responsibleExpertIdx / numLocalExperts;
        const int rankLaneIdx = responsibleExpertIdx % numLocalExperts;
        if (subWarpId == 0 && laneId == 0) {
            awaitSourceReadyCftLsa(args, cft, recvCntBuf, numBytesPerMsg,
                                   numSms, numLocalExperts, srcRank, rankLaneIdx);
        }
        if (perGroupRelease) {
            syncSmGroup(kLsaCftGroupBarrierBase + warpGroupId, args.numWarpsPerGroup * 32);
        }
    }
    if (!perGroupRelease) {
        __syncthreads();
    }
    compiler_fence();
}
#endif  // NCCL_EP_CFT_DEVICE_SUPPORTED

// Participants: complete warps of active receive groups after readiness.
// Each warp owns disjoint output slots. Count decoding and fabric/split-buffer
// addressing stay here; the shared metadata helper sees only an interpreted
// header and output slot. Zero-copy skips payload copying; self always reads
// its split-layout headers, while staged remote headers precede their payloads.
template <int kNumTopk, bool kZeroCopy>
__forceinline__ __device__ void recordRecvTokensCftLsa(
    const dispatch_kernel_args_t& args, const cft_dispatch_args_t& cft,
    const int* recvCntBuf, const void* recvBuf, size_t numBytesPerMsg,
    size_t dispatch_hdr_sz, size_t hiddenInt4, int laneId, int numLocalExperts,
    int subWarpId, int responsibleExpertIdx) {
    if (responsibleExpertIdx >= args.numExperts) {
        return;
    }
    const int srcRank = responsibleExpertIdx / numLocalExperts;
    const int rankLaneIdx = responsibleExpertIdx % numLocalExperts;
    const int numRecvTokens = isRankMasked(args.rankMask, srcRank) ? 0 : lsaCftDecodeCount(recvCntBuf, srcRank);
    if (laneId == 0 && rankLaneIdx == 0) {
        args.outSrcInfo[srcRank] = numRecvTokens;
        if (args.outRecvRankCounter != nullptr) {
            args.outRecvRankCounter[srcRank] = numRecvTokens;
        }
    }
    const bool remoteSrc = srcRank != args.currRank;
    const uint8_t* const srcRankRecvBase = remoteSrc
        ? cftRecvRegion(cft, srcRank)
        : reinterpret_cast<const uint8_t*>(recvBuf) +
              static_cast<size_t>(relativeRankSlot(srcRank, args.currRank, args.numRanks)) * args.maxTokensPerRank * numBytesPerMsg;
    const size_t hdrStride = remoteSrc && !kZeroCopy ? numBytesPerMsg : dispatch_hdr_sz;
    for (int i = rankLaneIdx * args.numWarpsPerGroup + subWarpId; i < numRecvTokens;
         i += args.numWarpsPerGroup * numLocalExperts) {
        const int slot = srcRank * args.maxTokensPerRank + i;
        const auto* hdr = reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(
            srcRankRecvBase + static_cast<size_t>(i) * hdrStride);
        int* recvSrcInfo = args.outSrcInfo + args.numRanks + slot * (kNumTopk + 1);
        recordRankMajorRecvTokenMetadata(hdr, slot, kNumTopk, numLocalExperts, args.currRank, laneId,
                                        recvSrcInfo, args.outRecvTopkIdx, args.outRecvTopkWeights, args.recvTopkIdxKind);
        if (args.recvDataWindow == ncclWindow_t{}) {
            const int4* src;
            int4* dst;
            if (remoteSrc) {
                src = reinterpret_cast<const int4*>(
                    srcRankRecvBase + static_cast<size_t>(i) * numBytesPerMsg + dispatch_hdr_sz);
                dst = static_cast<int4*>(args.outDataBuf) + static_cast<size_t>(slot) * hiddenInt4;
            } else {
                const int payloadBytes = static_cast<int>(numBytesPerMsg) - static_cast<int>(dispatch_hdr_sz);
                src = reinterpret_cast<const int4*>(
                    srcRankRecvBase + args.maxTokensPerRank * static_cast<int>(dispatch_hdr_sz) + i * payloadBytes);
                dst = static_cast<int4*>(args.outDataBuf) + slot * static_cast<int>(hiddenInt4);
            }
            UNROLLED_WARP_COPY(7, laneId, static_cast<int>(hiddenInt4), dst, src, ld_nc_global, st_na_global);
        }
    }
}

// Rank-major CFT entry point. Initialization and all phase helpers are CTA
// cooperative; SEND returns the elected last sender for the shared epoch rule.
// Split RECV chooses/advances its pending epoch in selectLowLatencyEpoch.
template <int kHidden, int kNumTopk, typename TopkIdxT, ncclDataType_t kTokenDtype,
          bool kCftZeroCopy, bool kCftCountedPayload, int kCftMaskWords>
__device__ __forceinline__ void dispatch_kernel_impl_cft_rm_lsa(const dispatch_kernel_args_t& args) {
    EP_STATIC_ASSERT(kNumTopk > 0 && kNumTopk <= combine_smem::kWarpSize - kLlDispatchControlWarps,
                     "LL dispatch top-k must leave one control warp");
    EP_STATIC_ASSERT(size_u8<kTokenDtype>() == kCftElementBytes, "LSA-CFT dispatch requires two-byte tokens");
#if defined(__CUDA_ARCH__)
    EP_STATIC_ASSERT(NCCL_EP_CFT_DEVICE_SUPPORTED, "CFT dispatch requires device CFT support");
#endif
    const int smId = static_cast<int>(blockIdx.x);
    const int threadId = static_cast<int>(threadIdx.x);
    const int warpId = threadId / 32, laneId = get_lane_id();
    const int numSms = static_cast<int>(gridDim.x);
    const int numWarps = args.numWarpGroups * args.numWarpsPerGroup;
    const int numLocalExperts = args.numExperts / args.numRanks;
    const int warpGroupId = warpId / args.numWarpsPerGroup;
    const int subWarpId = warpId % args.numWarpsPerGroup;
    const int responsibleExpertIdx = smId * args.numWarpGroups + warpGroupId;
    const unsigned int epoch = selectLowLatencyEpoch(args.epochState, args.phases, smId, threadId);
    const size_t bank = static_cast<size_t>(epoch & 1U);
    const size_t bankNext = bank ^ 1U;
    const size_t recvOff = args.recvOff + bank * args.payloadSlotStride;
    const size_t recvCntOff = args.recvCntOff + bank * args.signalSlotStride;
    char* const rdmaBase = static_cast<char*>(args.rdmaBuf);
    void* const recvBuf = rdmaBase + recvOff;
    int* const recvCntBuf = reinterpret_cast<int*>(rdmaBase + recvCntOff);
    int* const nextRecvCntBuf = reinterpret_cast<int*>(rdmaBase + args.recvCntOff + bankNext * args.signalSlotStride);
    const cft_dispatch_args_t cft = args.cft;
    const int nTokMax = ll_lsa_cft_tokens_per_cta(args.maxTokensPerRank, numSms);
    const int nTok = smId < args.numTokens ? ::nccl_ep::ceil_div(args.numTokens - smId, numSms) : 0;
    const size_t hiddenBytes = static_cast<size_t>(kHidden) * size_u8<kTokenDtype>();
    const size_t hdrBytes = get_dispatch_hdr_sz<NCCL_EP_LAYOUT_RANK_MAJOR>(kNumTopk);
    const size_t msgBytes = hdrBytes + hiddenBytes;
    EP_DEVICE_ASSERT(msgBytes % sizeof(int4) == 0);
    bool isLastCta = false;

#if NCCL_EP_CFT_DEVICE_SUPPORTED
    using Geometry = CftTileGeometry<kHidden, kNumTopk, NCCL_EP_LAYOUT_RANK_MAJOR, kCftZeroCopy>;
    __shared__ LsaCftSmem<kCftMaskWords> routing;
    CftSendContext<Geometry> sender;
    CftSelfDestination self{};
    const CftMessageLayout msg{hdrBytes, hiddenBytes, hiddenBytes, msgBytes, args.maxTokensPerRank};
    int* list = nullptr;
    int* nd = nullptr;
    uint8_t* hdrBatch = nullptr;
    initDispatchCftLsa<kNumTopk>(args, cft, msg, recvBuf, sender, self, routing, list, nd, hdrBatch,
                                 threadId, warpId, laneId, numSms, numWarps, nTokMax);

    if (args.phases & LOW_LATENCY_SEND_PHASE) {
        isLastCta = dispatchSendPhaseCftLsa<kNumTopk, kCftCountedPayload, TopkIdxT>(
            args, cft, sender, self, msg, routing, list, nd, hdrBatch, recvCntBuf, nextRecvCntBuf, recvCntOff,
            smId, warpId, laneId, numSms, numWarps, numLocalExperts, nTok);
    }
#endif
    if ((args.phases & LOW_LATENCY_RECV_PHASE) == 0) {
        return;
    }

    // The elected last SEND CTA advances a full invocation exactly once here,
    // before receive readiness. This helper does not wait for RECV completion.
    // Split RECV already advanced through selectLowLatencyEpoch; its phase mask
    // makes completeFullLowLatencyEpoch a no-op, preserving the shared lifecycle.
    if ((args.phases & LOW_LATENCY_SEND_PHASE) == 0) {
        isLastCta = (smId == 0);
    }
    completeFullLowLatencyEpoch(args.epochState, args.phases, isLastCta, threadId);
#if NCCL_EP_CFT_DEVICE_SUPPORTED
    syncAndRecvCountsCftLsa(args, cft, recvCntBuf, msgBytes, laneId, numSms, numLocalExperts,
                             warpGroupId, subWarpId, responsibleExpertIdx);
#endif
    recordRecvTokensCftLsa<kNumTopk, kCftZeroCopy>(
        args, cft, recvCntBuf, recvBuf, msgBytes, hdrBytes, hiddenBytes / sizeof(int4),
        laneId, numLocalExperts, subWarpId, responsibleExpertIdx);
}

}  // namespace ll
}  // namespace nccl_ep
