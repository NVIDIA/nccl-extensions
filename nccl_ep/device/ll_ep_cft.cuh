/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

// LL dispatch protocol over CFT counted fabric writes (NCCL_EP_LL_CFT).
// Transport primitives live in cft.cuh; ll_ep.cuh calls the helpers below.
// See nccl_ep/docs/plc/ll-cft-dispatch/README.md for the protocol.
//
// Warp 0 owns each CTA's sender state. Its lanes prepare routing, slots, and
// headers; lane 0 issues TMA loads/stores, fabric puts, and completion waits.
// No other warp accesses that state, so no CTA barrier is needed. Token counts
// use the existing store channel (cftPublishCountEarly in ll_ep.cuh).
//
// Fabric and TMA operations require NCCL_EP_CFT_DEVICE_SUPPORTED. Plain data
// structures and transport predicates remain available on all targets.

#pragma once

#include <cstdint>

#include "nccl_ep.h"
#include "nccl_device.h"
#include "common.hpp"
#include "device_primitives.cuh"
#include "cft.cuh"
#include "ll_ep_adapter.cuh"  // cft_dispatch_args_t, kCftDestScratchInts

namespace nccl_ep {
namespace ll {

// Failure reporting

// Receive-counter, fabric-completion, and TMA-load waits are bounded by the
// group timeout. On expiry, flag the error and trap: uncancelled async writes
// may still reference this CTA's tile or the receiver's counters.
//
// TODO: fault tolerance needs fabric fault/timeout discrimination, receive
// region quiescence, parity across re-admission, and timed-out tile quarantine.
// Keep failures fatal until those are supported; do not fall back to stores
// for a CFT pair during an operation.
__device__ __forceinline__ void cftReportSendFailure(
    int* asyncErrorFlag, int currRank, int dstRank, const char* what) {
    printf("Warning: NCCL EP CFT %s did not complete, rank %d -> rank %d\n", what, currRank, dstRank);
    // Flag first: it lives in pinned host memory, so it survives the trap's
    // context teardown and the host can still observe the failure.
    if (asyncErrorFlag != nullptr) atomicExch_system(asyncErrorFlag, 1);
    trap();
}

// Receive timeouts are fatal; CFT and rank masking are mutually exclusive.
// The caller must leave the cumulative byte target unchanged on failure.
__device__ __forceinline__ void cftReportRecvFailure(
    int* asyncErrorFlag, int currRank, int srcRank, int localExpertIdx, const char* what) {
    printf("Warning: NCCL EP CFT %s poll expired, rank %d, local_expert_idx %d, src_rank %d\n",
           what, currRank, localExpertIdx, srcRank);
    // Flag first: it lives in pinned host memory, so it survives the trap's
    // context teardown and the host can still observe the failure.
    if (asyncErrorFlag != nullptr) atomicExch_system(asyncErrorFlag, 1);
    trap();
}

// Sender/receiver transport choice

// Shared sender/receiver transport predicate. Strict CFT initialization
// ensures every peer is available; self-sends use local stores because fabric
// loopback puts are unsupported. Keep all pair decisions in this helper.
//
// Host setup requires one LSA team for the P2P count channel. That team may
// span multiple physical nodes in the same NVLink domain.
__device__ __forceinline__ bool cftPairUsesCft(bool cftEnabled, int srcRank, int dstRank) {
    return cftEnabled && (srcRank != dstRank);
}

// Sender state

// JIT tile geometry. Keep the host's dynamic shared-memory sizing in
// ll_ep_adapter.cu consistent with these constants.
template <int kHidden, int kNumTopk, ncclEpLayout_t kLayout, bool kCftZeroCopy>
struct CftTileGeometry {
    static constexpr bool kZeroCopy = kCftZeroCopy;
    // Double buffering overlaps the next payload load with the current put.
    // Only one TMA load is in flight: each iteration waits for its own load
    // before issuing the next, since independent copies need not finish in order.
    static constexpr int kRingDepth = 2;
    // Round the header room to 128 B so each tile remains 128 B aligned.
    static constexpr int kTileHdrBytes =
        static_cast<int>(align<size_t>(get_dispatch_hdr_sz<kLayout>(kNumTopk), 128));
    static constexpr int kTileBytes = kHidden * kCftElementBytes + kTileHdrBytes;
    // Separate puts use a 128 B-aligned payload source; the staged put is contiguous.
    static constexpr size_t kPayloadOffset = kZeroCopy ? kTileHdrBytes : get_dispatch_hdr_sz<kLayout>(kNumTopk);
    static constexpr size_t kTilesBytes = static_cast<size_t>(kRingDepth) * kTileBytes;
    // Completion budget in 16 B units. Reserve one worst-case token below
    // the 2^20-unit phase limit because the check runs after each submit.
    // All puts must be submitted before a drain can make progress; the
    // headroom keeps the next token below the limit before its budget check.
    static constexpr unsigned kTxSettleUnits =
        (1u << 20) - static_cast<unsigned>(kNumTopk > 0 ? kNumTopk : 1) * static_cast<unsigned>(kTileBytes / 16);
};

// Wire message: [header | payload]. CFT uses QUANT_NONE, so payloadBytes
// and hiddenBytes are equal; they describe the slot stride and copy size.
struct CftMessageLayout {
    size_t hdrBytes;      // dispatch header, align16(4 + numTopk * router)
    size_t hiddenBytes;   // token payload the TMA load/store and the puts carry
    size_t payloadBytes;  // hidden + scales (== hidden under CFT)
    size_t msgBytes;      // hdrBytes + payloadBytes: the counted put's size
    int maxTokensPerRank; // source-rank stride in the zero-copy output
};

// Warp-0 sender state. Tiles, endpoint IDs, and destination scratch use
// dynamic shared memory; mbarriers use static shared memory. Fabric parity
// and transaction counts belong to lane 0's register state.
//
// Dynamic layout, also used by the host's allocation:
//   [kTilesBytes]             staging tiles, each 128 B aligned
//   [align16(numRanks * 4)]    cached endpoint IDs
//   [kCftDestScratchBytes]     (rank, slot) pairs in two token-parity halves
// CftSendContext contains device-only fabric state, so it shares the PTX gate.
#if NCCL_EP_CFT_DEVICE_SUPPORTED
template <class Geometry>
struct CftSendContext {
    uint8_t* tiles;      // staging ring: TMA destination and fabric source
    uint32_t* leMirror;  // cached peer endpoint IDs; zero remains a valid ID
    int* destScratch;    // packed (rank, slot) pairs for lane 0
    // Plain TMA barriers, separate from the layout::v1 fabric barrier.
    // Each slot's phase is (iteration / kRingDepth) & 1.
    uint64_t* tmaMbars;
    nccl_ep::cft::Cft<ncclCoopThread> fabric;  // completion tracking owned by lane 0

    __device__ __forceinline__ uint8_t* tile(int slot) const {
        return tiles + static_cast<size_t>(slot) * Geometry::kTileBytes;
    }
};

// Bind the dynamic shared-memory regions and static mbarriers. The kernel
// builds this context only in CFT variants; warp 0 owns the sender state.
template <class Geometry>
__device__ __forceinline__ CftSendContext<Geometry> cftCarveSendContext(
    uint8_t* dynSmem, nccl_ep::cft::Smem* dataSmem, uint64_t* tmaMbars, int numRanks) {
    // Keep the scratch region 16 B aligned after the endpoint ID cache.
    const size_t leMirrorBytes =
        (static_cast<size_t>(numRanks) * sizeof(uint32_t) + 15) & ~static_cast<size_t>(15);
    CftSendContext<Geometry> ctx;
    ctx.tiles = dynSmem;
    ctx.leMirror = reinterpret_cast<uint32_t*>(dynSmem + Geometry::kTilesBytes);
    ctx.destScratch = reinterpret_cast<int*>(dynSmem + Geometry::kTilesBytes + leMirrorBytes);
    ctx.tmaMbars = tmaMbars;
    ctx.fabric.coop = ncclCoopThread{};
    ctx.fabric.smem = dataSmem;
    ctx.fabric.txCount = 0;
    ctx.fabric.phaseParity = 0;
    ctx.fabric.pendingTxCount = 0;
    return ctx;
}
#endif  // NCCL_EP_CFT_DEVICE_SUPPORTED (CftSendContext)

// Self-rank receive destination, using the store path's split layout:
// packed headers followed by payload slots. Headers use lane stores and
// payloads use TMA from the staged tile; fabric loopback is unsupported.
struct CftSelfDestination {
    uint8_t* recvBase;       // header section: slot s at recvBase + s * hdrBytes
    uint8_t* payloadBase;   // payload slot zero, either internal storage or the bound output
};

template <bool kCftZeroCopy>
__device__ __forceinline__ CftSelfDestination cftSelfDestination(
    void* recvBuf, void* outData, int currRank, int numRanks, size_t srcRankRegionBytes,
    int maxTokensPerRank, size_t hdrBytes, size_t hiddenBytes) {
    CftSelfDestination self;
    self.recvBase = static_cast<uint8_t*>(recvBuf) +
        static_cast<size_t>(relativeRankSlot(currRank, currRank, numRanks)) * srcRankRegionBytes;
    self.payloadBase = kCftZeroCopy
        ? static_cast<uint8_t*>(outData) + static_cast<size_t>(currRank) * maxTokensPerRank * hiddenBytes
        : self.recvBase + static_cast<size_t>(maxTokensPerRank) * hdrBytes;
    return self;
}

// All local experts receiving from one source poll the same hardware counter.
// Each keeps a separate cumulative target to avoid concurrent target updates.
struct CftRecvChannel {
    const uint64_t* byteCounter;  // srcRank's hardware byte counter, parity-resolved, never reset
    uint64_t* cumulTarget;        // this channel's running byte target (device word)
};

__device__ __forceinline__ CftRecvChannel cftRecvChannel(
    const cft_dispatch_args_t& cft, int srcRank, int rankLaneIdx, int numLocalExperts) {
    constexpr uint64_t kCntrStride8 = 256 / 8;  // uint64 stride for a 256B slot
    CftRecvChannel ch;
    ch.byteCounter = cft.recvByteCounters + srcRank * kCntrStride8;
    ch.cumulTarget = cft.cumulByteExpect + (srcRank * numLocalExperts + rankLaneIdx);
    return ch;
}

// Fabric receive regions use absolute source-rank indexing and interleaved
// messages. The store path's recvBuf instead uses relativeRankSlot.
__device__ __forceinline__ const uint8_t* cftRecvRegion(const cft_dispatch_args_t& cft, int srcRank) {
    return reinterpret_cast<const uint8_t*>(cft.recvBase) + srcRank * cft.srcRankRegionBytes;
}

#if NCCL_EP_CFT_DEVICE_SUPPORTED

// Sender: one-time init (warp 0)

// Lane 0 initializes the barriers; their wrappers publish to the async proxy.
// The closing warp sync makes the endpoint ID cache visible to lane 0.
// Only warp 0 uses this state.
template <class Geometry>
__device__ __forceinline__ void cftInitSender(
    CftSendContext<Geometry>& ctx, const uint32_t* peerLeIds, int numRanks, int laneId) {
    if (laneId == 0) {
        ctx.fabric.init(ncclCoopThread{});
        for (int s = 0; s < Geometry::kRingDepth; ++s) {
            nccl_ep::cft::cft_tma_mbar_init(&ctx.tmaMbars[s]);
        }
    }
    // Cache endpoint IDs in shared memory for the put loop.
#pragma unroll 1  // Avoid expanding the lane-dependent trip count.
    for (int r = laneId; r < numRanks; r += 32) {
        ctx.leMirror[r] = peerLeIds[r];
    }
    __syncwarp();
}

// TMA requires 16 B-aligned addresses and copy sizes. Header alignment and
// the kernel's hidden-size assertion ensure valid strides; check the local
// receive base once per CTA.
__device__ __forceinline__ void cftCheckBulkAlignment(const CftSelfDestination& self, const CftMessageLayout& msg) {
    EP_DEVICE_ASSERT((msg.hdrBytes % 16) == 0 && (msg.hiddenBytes % 16) == 0 &&
                     (reinterpret_cast<uintptr_t>(self.recvBase) % 16) == 0 &&
                     (reinterpret_cast<uintptr_t>(self.payloadBase) % 16) == 0);
}

// Sender: stage (TMA load into a ring slot)

// Stage the payload after the header room. Routing headers are prepared per
// iteration. The prologue loads the first token; later iterations issue the
// next load only after consuming the current one.
template <class Geometry>
__device__ __forceinline__ void cftStagePayload(
    CftSendContext<Geometry>& ctx, int slot, const int4* srcToken, const CftMessageLayout& msg) {
    nccl_ep::cft::cft_tma_expect_tx(&ctx.tmaMbars[slot], static_cast<unsigned>(msg.hiddenBytes));
    nccl_ep::cft::cft_tma_g2s(ctx.tile(slot) + Geometry::kPayloadOffset, srcToken, static_cast<unsigned>(msg.hiddenBytes),
                              &ctx.tmaMbars[slot]);
}

// Lane 0 waits before rearming a slot. Parity follows the ring-wrap count;
// rearming while the copy is in flight would supply a second barrier arrival.
// Timeout reports and traps; the zero return is defensive.
template <class Geometry>
__device__ __forceinline__ int cftAwaitStagedPayload(
    CftSendContext<Geometry>& ctx, int slot, uint32_t* ringPhase, uint64_t timeoutCycles, int* asyncErrorFlag,
    int currRank) {
    int ok = 1;
    if (nccl_ep::cft::cft_tma_wait(&ctx.tmaMbars[slot], ringPhase, timeoutCycles) == 0) {
        cftReportSendFailure(asyncErrorFlag, currRank, -1, "payload staging copy");
        ok = 0;
    }
    return ok;
}

// Sender: prepare (warp-wide phase of one token)

// All lanes prepare every token. The first top-k entry for each destination
// wins a slot, including self. Return one mask bit per winning lane.
//
// Header writes may precede the TMA wait because header and payload regions
// are disjoint, and the tile's previous reads were drained before reuse.
// A TMA timeout after slot allocation is fatal, so those slots are never sent.
// The final warp sync publishes headers and scratch to lane 0; cftPutToken's
// proxy fence then publishes the header to the async engines.
template <int kNumTopk, typename TopkIdxT>
__device__ __forceinline__ unsigned cftPrepareToken(
    int* dscratch, uint8_t* tile, const CftSelfDestination& self, int tokenIdx, const TopkIdxT* inTopkIdx,
    const uint8_t* sendBufBase, int* rankSentCnt, int numLocalExperts, int currRank, size_t hdrBytes, int laneId) {
    int myRank = -1;
    if (laneId < kNumTopk) {
        const auto e = static_cast<int>(__ldg(inTopkIdx + tokenIdx * kNumTopk + laneId));
        if (e >= 0) myRank = getExpertRankIdx(e, numLocalExperts);
    }
    // Deduplicate destinations, including the self rank.
    bool win = (myRank >= 0);
    // Uniform trip count: every lane joins every shuffle.
    for (int j = 0; j < kNumTopk; ++j) {
        const int rj = __shfl_sync(0xffffffff, myRank, j);
        if (j < laneId && rj >= 0 && rj == myRank) win = false;
    }
    const unsigned winMask = __ballot_sync(0xffffffff, win);

    if (winMask != 0u) {
        // Winning lanes allocate destination slots concurrently. Alternating
        // scratch halves preserve the current token's entries while lane 0
        // issues its puts.
        int selfSlot = -1;
        if (win) {
            const int mySlot = atomicAdd(rankSentCnt + myRank, 1);
            const int pos = __popc(winMask & ((1u << laneId) - 1u));
            dscratch[2 * pos] = myRank;
            dscratch[2 * pos + 1] = mySlot;
            if (myRank == currRank) selfSlot = mySlot;
        }
        // Broadcast the unique self destination slot, or -1 if absent.
        selfSlot = __reduce_max_sync(0xffffffff, selfSlot);
        // Copy the header to the tile and, if routed locally, the self slot.
        // The warp sync orders local header stores before lane 0's grid barrier;
        // the receiver reads after that barrier or the split-phase kernel boundary.
        const int4* hdrSrc = reinterpret_cast<const int4*>(sendBufBase);
        int4* hdrDst = reinterpret_cast<int4*>(tile);
        int4* selfHdrDst = reinterpret_cast<int4*>(self.recvBase + static_cast<size_t>(selfSlot) * hdrBytes);
        const int hdrInt4 = static_cast<int>(hdrBytes / sizeof(int4));
        for (int i = laneId; i < hdrInt4; i += 32) {
            const int4 h = hdrSrc[i];
            hdrDst[i] = h;
            if (selfSlot >= 0) st_na_global(selfHdrDst + i, h);
        }
    }
    __syncwarp();
    return winMask;
}

// Sender: put (lane 0)

// Publish the tile, then send once per destination. Self uses a TMA payload
// store. Zero-copy sends headers to the internal buffer and payloads to the
// output; staged CFT sends a combined message. Completion reports are drained
// by cftSettleIfOverBudget and cftDrainSender.
template <class Geometry>
__device__ __forceinline__ void cftPutToken(
    CftSendContext<Geometry>& ctx, const cft_dispatch_args_t& cft, const CftSelfDestination& self,
    const CftMessageLayout& msg, const int* dscratch, const uint8_t* tile, unsigned winMask, uint64_t byteCntrOff,
    int currRank, bool skipHdr, bool uncountedPayload) {
    // Publish the header before the fabric puts and local TMA store read it.
    nccl_ep::cft::memFence(ncclCoopThread{}, cuda::memory_order_release, nccl_ep::cft::ProxyType::Generic,
                           nccl_ep::cft::ProxyType::Fabric, nccl_ep::cft::FenceScope::Cta);
    const int nDests = __popc(winMask);
    // Keep the destination loop rolled to limit code growth.
#pragma unroll 1
    for (int d = 0; d < nDests; ++d) {
        const int dstRank = dscratch[2 * d];
        const int slotIdx = dscratch[2 * d + 1];
        if (dstRank == currRank) {
            // Reuse the staged payload for self; commit one bulk group per token.
            nccl_ep::cft::cft_tma_s2g(
                self.payloadBase + static_cast<size_t>(slotIdx) * msg.payloadBytes,
                tile + Geometry::kPayloadOffset, static_cast<unsigned>(msg.hiddenBytes));
            nccl_ep::cft::cft_bulk_commit();
        } else {
            const uint64_t regionOff = cft.parityOff + static_cast<uint64_t>(currRank) * cft.srcRankRegionBytes;
            if constexpr (Geometry::kZeroCopy) {
                // Counted sources use the same per-source byte counter for headers and payloads.
                const uint64_t hdrOff = regionOff + static_cast<uint64_t>(slotIdx) * msg.hdrBytes;
                const uint64_t payloadOff = cft.outputOffset +
                    (static_cast<uint64_t>(currRank) * msg.maxTokensPerRank + slotIdx) * msg.hiddenBytes;
                // LSA-CFT header batching sends the headers of a slot range in one put later.
                if (!skipHdr) {
                    ctx.fabric.putCounted(ncclCoopThread{}, ctx.leMirror[dstRank], hdrOff, byteCntrOff, tile,
                                          static_cast<unsigned>(msg.hdrBytes));
                }
                if (uncountedPayload) {
                    // LSA publishes readiness after both payload and routing headers complete.
                    ctx.fabric.put(ncclCoopThread{}, ctx.leMirror[dstRank], payloadOff,
                                   tile + Geometry::kPayloadOffset, static_cast<unsigned>(msg.hiddenBytes));
                } else {
                    ctx.fabric.putCounted(ncclCoopThread{}, ctx.leMirror[dstRank], payloadOff, byteCntrOff,
                                          tile + Geometry::kPayloadOffset, static_cast<unsigned>(msg.hiddenBytes));
                }
            } else {
                const uint64_t dataOff = regionOff + static_cast<uint64_t>(slotIdx) * msg.msgBytes;
                if (uncountedPayload) {
                    // LSA publishes readiness after the complete message arrives.
                    ctx.fabric.put(ncclCoopThread{}, ctx.leMirror[dstRank], dataOff, tile,
                                   static_cast<unsigned>(msg.msgBytes));
                } else {
                    ctx.fabric.putCounted(ncclCoopThread{}, ctx.leMirror[dstRank], dataOff, byteCntrOff, tile,
                                          static_cast<unsigned>(msg.msgBytes));
                }
            }
        }
    }
}

// Sender: tile reuse (lane 0)

// Submit this token's puts, then wait for fabric and local TMA source reads
// before reusing the tile. Local write completion is deferred to cftDrainSender.
// A self-only token has no fabric puts; its submit is empty. The local TMA
// read wait has no bounded form (see cft.cuh).
template <class Geometry>
__device__ __forceinline__ void cftReleaseTile(CftSendContext<Geometry>& ctx) {
    ctx.fabric.submit(ncclCoopThread{});
    ctx.fabric.flushSmem(ncclCoopThread{});
    nccl_ep::cft::cft_bulk_wait_read<0>();
}

// Sender: tracking drain (lane 0)

// Drain submitted puts, advance barrier parity, and clear the completion
// budget. Every put must be submitted first. Timeout reports and traps.
template <class Geometry>
__device__ __forceinline__ void cftDrainPuts(
    CftSendContext<Geometry>& ctx, uint64_t timeoutCycles, int* asyncErrorFlag, int currRank) {
    if (!ctx.fabric.flush(ncclCoopThread{}, timeoutCycles)) {
        cftReportSendFailure(asyncErrorFlag, currRank, -1, "payload puts (batch)");
    }
}

// Check after each token's submit. The reserved headroom in kTxSettleUnits
// keeps the next token within the barrier's 2^20-unit phase limit.
template <class Geometry>
__device__ __forceinline__ void cftSettleIfOverBudget(
    CftSendContext<Geometry>& ctx, uint64_t timeoutCycles, int* asyncErrorFlag, int currRank) {
    if (ctx.fabric.pendingTxCount >= Geometry::kTxSettleUnits) {
        cftDrainPuts(ctx, timeoutCycles, asyncErrorFlag, currRank);
    }
}

// Complete local TMA writes and fence async -> generic, then drain Fabric reports.
// The caller publishes self readiness through a grid/kernel boundary (general)
// or a CTA barrier and self-count release/acquire (LSA). Local-store waits are
// unbounded; the Fabric drain uses the group timeout.
template <class Geometry>
__device__ __forceinline__ void cftDrainSender(
    CftSendContext<Geometry>& ctx, uint64_t timeoutCycles, int* asyncErrorFlag, int currRank) {
    nccl_ep::cft::cft_bulk_wait_all();
    nccl_ep::cft::cft_publish_bulk_stores_global();
    cftDrainPuts(ctx, timeoutCycles, asyncErrorFlag, currRank);
}

// Sender: one token, warp 0

// Warp 0 prepares the token; lane 0 waits for staging, prefetches the next
// payload, issues puts, releases the tile, and checks the completion budget.
// Fabric receivers wait on byte counters; self waits on the grid barrier
// or split-phase kernel boundary. This path does not use rankDone.
template <int kNumTopk, class Geometry, typename TopkIdxT>
__device__ __forceinline__ void cftSendToken(
    CftSendContext<Geometry>& ctx, const cft_dispatch_args_t& cft, const CftSelfDestination& self,
    const CftMessageLayout& msg, int tokenIdx, int numSms, int numTokens, const TopkIdxT* inTopkIdx,
    const int4* inDataInt4, size_t hiddenBf16Int4, const uint8_t* sendBufBase, int* rankSentCnt,
    int numLocalExperts, int currRank, int laneId, uint64_t timeoutCycles, int* asyncErrorFlag) {
    const int cftIt = tokenIdx / numSms;
    const int rslot = cftIt % Geometry::kRingDepth;
    int* dscratch = ctx.destScratch + (cftIt & 1) * (kCftDestScratchInts / 2);
    uint8_t* tile = ctx.tile(rslot);
    const unsigned winMask = cftPrepareToken<kNumTopk>(
        dscratch, tile, self, tokenIdx, inTopkIdx, sendBufBase, rankSentCnt, numLocalExperts, currRank,
        msg.hdrBytes, laneId);

    // Select this rank's counter in each peer's current parity copy.
    // The combine round trip orders dispatches (see ncclEpDispatch's parity flip).
    const uint64_t byteCntrOff = cft.parityOff + cft.bytesCntrBaseOff + static_cast<uint64_t>(currRank) * 256;
    uint32_t ringPhase = static_cast<uint32_t>((cftIt / Geometry::kRingDepth) & 1);

    if (laneId == 0) {
        const int ok = cftAwaitStagedPayload(ctx, rslot, &ringPhase, timeoutCycles, asyncErrorFlag, currRank);
        // Prefetch into the other tile after consuming this load. The wait
        // traps on failure; ok also prevents subsequent work if it returns.
        if (ok) {
            const int nextTok = tokenIdx + numSms;
            if (nextTok < numTokens) {
                cftStagePayload(ctx, rslot ^ 1, inDataInt4 + static_cast<size_t>(nextTok) * hiddenBf16Int4, msg);
            }
        }
        if (ok && winMask != 0u) {
            cftPutToken(ctx, cft, self, msg, dscratch, tile, winMask, byteCntrOff, currRank,
                        /*skipHdr=*/false, /*uncountedPayload=*/false);
            cftReleaseTile(ctx);
            cftSettleIfOverBudget(ctx, timeoutCycles, asyncErrorFlag, currRank);
        }
    }
}

// Receiver: are the bytes here (poll and fence)

// Wait for payload bytes after receiving the token count, which may arrive
// first. One thread per receive warp group polls the cumulative byte counter.
// Save the expected target on success, rather than the observed counter:
// a fast sender may already have contributed bytes for the next round.
// Zero-token rounds skip polling. On timeout, leave the target unchanged,
// consume nothing, and report the failure.
__device__ __forceinline__ int cftAwaitPayload(
    const CftRecvChannel& ch, int numRecvTokens, size_t msgBytes, uint64_t timeoutCycles, int* asyncErrorFlag,
    int currRank, int srcRank, int rankLaneIdx) {
    const uint64_t prevBytes = *ch.cumulTarget;
    const uint64_t expectedBytes = prevBytes + static_cast<uint64_t>(numRecvTokens) * static_cast<uint64_t>(msgBytes);
    bool ok = true;
    if (numRecvTokens > 0) {
        // waitCounted fences fabric writes before this thread reads the payload.
        ok = nccl_ep::cft::waitCounted(ncclCoopThread{}, cuda::memory_order_acquire, nccl_ep::cft::ProxyType::Generic,
                                       ch.byteCounter, expectedBytes, timeoutCycles) == ncclSuccess;
    }
    if (ok) {
        *ch.cumulTarget = expectedBytes;
    } else {
        numRecvTokens = 0;
        cftReportRecvFailure(asyncErrorFlag, currRank, srcRank, rankLaneIdx, "dispatch receive");
    }
    return numRecvTokens;
}

#endif  // NCCL_EP_CFT_DEVICE_SUPPORTED

}  // namespace ll
}  // namespace nccl_ep
