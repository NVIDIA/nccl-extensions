/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nccl_ep::ll::combine_smem {

// These values define the LL combine kernel's dynamic shared-memory layout.
// Keep all host launch accounting and device layout derivations based on them.
constexpr int kWarpSize = 32;
constexpr int kMaxSendUnrolls = 4;
constexpr int kMinSendUnrolls = 2;
constexpr int kNumRecvUnrolls = 2;
constexpr int kNumStages = 3;
constexpr int kNumTmaPrefetch = 1;
constexpr int kNumRecvTmaWarps = 1;
constexpr int kSendTmaBarrierBytes = sizeof(std::uint64_t);
constexpr int kSendTmaPaddingBytes = sizeof(std::uint64_t);
constexpr int kTmaBarrierAndPaddingBytes = kSendTmaBarrierBytes + kSendTmaPaddingBytes;
constexpr int kNumRecvTmaBarriers = 2;
constexpr int kRecvTmaBarrierBytes = kNumRecvTmaBarriers * sizeof(std::uint64_t);
constexpr int kRecvTmaPaddingBytes = sizeof(int4);
constexpr int kRecvTmaEmptyBarrierOffsetBytes = sizeof(std::uint64_t);
constexpr int kRecvTmaPayloadOffsetBytes = kRecvTmaBarrierBytes;
constexpr int kMetaElementBytes = sizeof(std::uint32_t);
constexpr int kLogFmtElementsPerMeta = 128;
constexpr int kLogFmtInputBytesPerChunk = 16;
constexpr int kLogFmtOutputBytesPerChunk = 10;
constexpr int kNumRecvMetaBuffers = 3;
constexpr int kTwoByteElementBytes = sizeof(std::uint16_t);
constexpr int kMaxRecvGroupsForTwoByteElements = 2;
constexpr int kMaxRecvGroupsForOtherElements = 1;
constexpr int kNvfp4ElementsPerPackedByte = 2;
constexpr int kNvfp4ElementsPerScale = 16;
constexpr int kNvfp4GlobalScaleBytes = sizeof(float);

constexpr int elements_per_int4(int elem_bytes) {
    return static_cast<int>(sizeof(int4)) / elem_bytes;
}

constexpr int send_unrolls(int hidden, int elem_bytes) {
    return hidden % (kWarpSize * kMaxSendUnrolls * elements_per_int4(elem_bytes)) == 0 ? kMaxSendUnrolls :
                                                                                           kMinSendUnrolls;
}

constexpr int padded_hidden_int4(int hidden, int elem_bytes) {
    const int unrolls = send_unrolls(hidden, elem_bytes);
    const int hidden_int4 = hidden / elements_per_int4(elem_bytes);
    const int alignment = kWarpSize * unrolls;
    return ((hidden_int4 + alignment - 1) / alignment) * alignment;
}

constexpr int recv_decode_warps(int hidden, int elem_bytes) {
    return padded_hidden_int4(hidden, elem_bytes) / (kNumRecvUnrolls * kWarpSize);
}

constexpr int max_recv_groups(int elem_bytes) {
    return elem_bytes == kTwoByteElementBytes ? kMaxRecvGroupsForTwoByteElements : kMaxRecvGroupsForOtherElements;
}

constexpr int meta_bytes(int hidden) {
    return hidden / kLogFmtElementsPerMeta * kMetaElementBytes;
}

constexpr int align_to_int4(int bytes) {
    return ((bytes + static_cast<int>(sizeof(int4)) - 1) / static_cast<int>(sizeof(int4))) *
           static_cast<int>(sizeof(int4));
}

constexpr int nvfp4_slot_bytes(int hidden) {
    return align_to_int4(
        hidden / kNvfp4ElementsPerPackedByte + hidden / kNvfp4ElementsPerScale + kNvfp4GlobalScaleBytes);
}

constexpr int send_warp_smem_bytes(int hidden, int elem_bytes) {
    const int tma_bytes = kWarpSize * static_cast<int>(sizeof(int4)) * send_unrolls(hidden, elem_bytes) +
                          kTmaBarrierAndPaddingBytes;
    // Each warp owns TMA buffers followed by a 4-byte-granular metadata tail.
    // Round the complete stride so every warp's first TMA buffer remains aligned.
    return align_to_int4(kNumStages * tma_bytes + meta_bytes(hidden));
}

constexpr int send_smem_bytes(int hidden, int elem_bytes, int num_warps) {
    return num_warps * send_warp_smem_bytes(hidden, elem_bytes);
}

constexpr int recv_tma_buffer_bytes(int hidden, int elem_bytes, bool nvfp4) {
    const int padded_token_bytes = padded_hidden_int4(hidden, elem_bytes) * static_cast<int>(sizeof(int4));
    return kRecvTmaBarrierBytes + (nvfp4 ? nvfp4_slot_bytes(hidden) : kRecvTmaPaddingBytes + padded_token_bytes);
}

constexpr int recv_group_smem_bytes(int hidden, int elem_bytes, bool nvfp4) {
    const int tma_bytes = recv_tma_buffer_bytes(hidden, elem_bytes, nvfp4);
    const int padded_token_bytes = padded_hidden_int4(hidden, elem_bytes) * static_cast<int>(sizeof(int4));
    // Every receive group contains TMA barriers and bulk-copy buffers. Keep the
    // group stride int4-aligned so group 1+ has the same alignment as group 0.
    // The metadata tail is only 4-byte granular and can otherwise leave the
    // next group 8 bytes off a TMA-compatible address (for example H=2880).
    return align_to_int4(
        kNumStages * tma_bytes + padded_token_bytes + kNumStages * meta_bytes(hidden) * kNumRecvMetaBuffers);
}

constexpr int dynamic_smem_bytes(int hidden, int elem_bytes, int num_warps, int num_recv_groups, bool nvfp4) {
    const int send_bytes = send_smem_bytes(hidden, elem_bytes, num_warps);
    const int recv_bytes = num_recv_groups * recv_group_smem_bytes(hidden, elem_bytes, nvfp4);
    return send_bytes > recv_bytes ? send_bytes : recv_bytes;
}

// Upper bound on numRanks for combine_kernel_impl_2sided_rm_lsa's CTA-local
// active-rank bitmask (ll/ll_combine_lsa.cuh), which has no other
// compile-time bound. 128 matches the HT path's enforced lsa_team_size cap.
constexpr int kMaxLsaCombineRanks = 128;
constexpr int kActiveRankMaskWords = (kMaxLsaCombineRanks + 31) / 32;

// Static __shared__ footprint of that bitmask -- separate from (and not
// covered by) the dynamic_smem_bytes() budget above; must be reserved out
// of max_dynamic_smem by choose_combine_smem_config (ll_ep_adapter.cuh).
constexpr int kLsaStaticSmemBytes = kActiveRankMaskWords * static_cast<int>(sizeof(std::uint32_t));

} // namespace nccl_ep::ll::combine_smem

// ============================================================================
// Shared-memory budget for the TMA-pipelined LL dispatch SEND phase
// (dispatch_kernel_impl_2sided_rm_lsa_tma, ll/ll_dispatch_lsa.cuh).
//
// Unlike combine_smem::dynamic_smem_bytes above, this budget is NOT divided across warps:
// dispatch_kernel_impl_2sided_rm_lsa_tma uses exactly one warp per CTA to drive the whole
// token-send pipeline (see that file's header comment for why), so the entire per-CTA dynamic
// shared-memory budget is available to that single warp's stage ring.
//
// Compile-time vs runtime split (important): num_stages here is resolved ONCE, host-side, at
// JIT-variant-selection time, from (hidden, dtype, max_dynamic_smem) alone -- the same inputs that
// already key the JIT variant cache. It is NOT re-derived per dispatch call from that call's actual
// token count, because the per-call token count (numTokens) does not participate in JIT variant
// selection today (only hidden/layout/topk/nvlinkOnly/dtype do) -- keying the smem allocation to a
// runtime-varying quantity would mean a new compiled kernel per distinct numTokens/numSms ratio,
// which defeats JIT caching. Instead, num_stages here is an upper bound on allocated stages; the
// kernel itself clamps how many stages it actually *uses* for prefetch to
// min(kNumStages, ceil_div(numTokens, numSms)) at runtime, so a CTA with fewer tokens than stages
// simply prefetches fewer of them -- no wasted work, same fixed allocation.
// ============================================================================

namespace nccl_ep::ll::dispatch_tma_smem {

constexpr int kMinQuantWarps = 2;
constexpr int kMaxQuantWarps = 6;
constexpr int kControlWarps = 1;
constexpr int kCleanupWarps = 1;
constexpr int kQuantReadyBarrier = 1;
constexpr int kQuantReadBarrier = 2;

constexpr int kTmaBarrierBytes = sizeof(std::uint64_t);
constexpr int kTmaPaddingBytes = sizeof(std::uint64_t);
constexpr int kTmaBarrierAndPaddingBytes = kTmaBarrierBytes + kTmaPaddingBytes; // 16

// Upper bound on the number of pipeline stages, independent of smem capacity -- keeps the
// mbarrier array and ring-index arithmetic a small, compile-time-friendly size. This many
// in-flight TMA loads is already enough to hide typical NVLink load latency behind issue-rate;
// going deeper trades shared memory for diminishing returns.
constexpr int kMaxStages = 8;

// Per-stage layout: [payload (hiddenBytes, TMA-loaded from args.inData)]
//                   [header (dispatch_hdr_sz bytes, assembled in-place by lanes, 16B-aligned
//                    already by get_dispatch_hdr_sz's own alignment)]
//                   [mbarrier + padding (kTmaBarrierAndPaddingBytes), tracks the payload load only
//                    -- the header has no load to wait on, it's written directly by lanes and only
//                    needs the warp's own __syncwarp() before being read back out for the store]
__host__ __device__ __forceinline__ int per_stage_bytes(int hidden_bytes, int dispatch_hdr_sz) {
    return hidden_bytes + dispatch_hdr_sz + kTmaBarrierAndPaddingBytes;
}

// Per-CTA (not per-stage) resolved-P2P-pointer cache, one uint64_t per destination rank. See
// dispatchSendPhaseLsaTma's doc comment: ncclGetP2pPtr's result for a given dstRank is loop
// invariant (depends only on currRank/dstRank/recvOff, none of which change across this CTA's
// token range), so it's resolved once into this cache before the send loop instead of once per
// (token, winning lane) -- removing a redundant team-membership check + window-table lookup from
// the hot loop. Placed immediately after the stage ring in the same dynamic smem allocation.
__host__ __device__ __forceinline__ int rank_cache_bytes(int num_ranks) {
    return static_cast<int>(static_cast<size_t>(num_ranks) * sizeof(std::uint64_t));
}

struct DispatchTmaSmemConfig {
    int num_stages;
    int dynamic_smem_bytes; // stage ring + rank-pointer cache, total
    bool feasible;
};

// Account for the kernel's static smem footprint -- separate from (and not covered by) this
// namespace's dynamic smem budget, so the caller must reserve it out of max_dynamic_smem before
// calling choose_config, or a boundary-sized stage ring can push static + dynamic over the
// device's shared-memory limit and fail at launch. Current static usage:
//  * syncAndSendCounts: __shared__ bool shIsLastCta
// Revisit if the TMA kernel's call path gains more static __shared__ state.
constexpr int kStaticSmemBytes = static_cast<int>(sizeof(std::uint32_t));

// hidden_bytes/dispatch_hdr_sz: per the doc comment above, resolved once per JIT variant from
// (hidden, dtype, numTopk) -- the same quantities already baked into that variant's template args.
// max_dynamic_smem: group->max_dynamic_smem (cudaDevAttrMaxSharedMemoryPerBlockOptin), the same
// host-cached value combine's smem config already uses.
// num_ranks: this EP group's world size -- sizes the rank-pointer cache (see rank_cache_bytes
// above); reserved out of the budget before sizing the stage ring, since the cache is a small,
// fixed, per-CTA cost independent of how many stages end up allocated.
__host__ __device__ __forceinline__ DispatchTmaSmemConfig choose_config(
    int hidden_bytes, int dispatch_hdr_sz, int max_dynamic_smem, int num_ranks) {
    const int stage_bytes = per_stage_bytes(hidden_bytes, dispatch_hdr_sz);
    const int cache_bytes = rank_cache_bytes(num_ranks);
    const int stage_budget = max_dynamic_smem - cache_bytes;
    if (stage_bytes <= 0 || stage_budget <= 0) return DispatchTmaSmemConfig{0, 0, false};
    int num_stages = stage_budget / stage_bytes;
    num_stages = num_stages < kMaxStages ? num_stages : kMaxStages;
    if (num_stages <= 0) return DispatchTmaSmemConfig{0, 0, false};
    return DispatchTmaSmemConfig{num_stages, num_stages * stage_bytes + cache_bytes, true};
}

} // namespace nccl_ep::ll::dispatch_tma_smem
