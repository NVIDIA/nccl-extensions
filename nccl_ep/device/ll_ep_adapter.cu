/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/ll_ep_adapter.cuh"
#include "device/macros.cuh"
#include "common.hpp"
#include "jit/jit_runtime.hpp"
#include "jit/ll_dispatch_jit.cuh"
#include "jit/ll_combine_jit.cuh"
#include "jit/ll_clean_jit.cuh"
#include "quantization_recipe.hpp"

#include <algorithm>
#include <cstdio>

namespace nccl_ep {
namespace ll {

// Reuse the host/device sizing helper from common.hpp.
using ::nccl_ep::ceil_div;

bool ll_cft_zero_copy_supported(
    bool nvlinkOnly, ncclEpDispQuant_t recipe, ncclEpLayout_t layout,
    NcclEpDispatchLsaMode lsaMode, int hidden, ncclDataType_t tokenDtype) {
    return jit::ll_dispatch_cft_zero_copy_supported(
        nvlinkOnly, recipe, layout, lsaMode, hidden, tokenDtype);
}

namespace {

int cft_counted_payload(const DispatchParams& params) {
    const int maxTok = ll_lsa_cft_counted_max_tokens();
    const int countedPayload = (maxTok > 0 && params.numTokens <= maxTok) ? 1 : 0;
    static bool announced = false;
    if (maxTok > 0 && !announced && std::getenv("NCCL_EP_DEBUG") != nullptr) {
        announced = true;
        std::fprintf(stderr, "[nccl_ep][ll_cft] rank %d: adaptive put form: counted payload for dispatches with <= %d tokens (this dispatch: %d tokens -> %s)\n",
                     params.currRank, maxTok, params.numTokens, countedPayload ? "counted" : "uncounted");
    }
    return countedPayload;
}

ncclResult_t cft_dispatch_smem_bytes(
    const DispatchParams& params, bool lsaCft, int numSms, int* smemBytes) {
    // Match the device's two-slot stage, peer IDs and destination scratch.
    const int tileHdrBytes =
        static_cast<int>(align<size_t>(get_dispatch_hdr_sz(params.numTopk, params.layout), 128));
    const int tileBytes = params.hidden * kCftElementBytes + tileHdrBytes;
    int bytes = 2 * tileBytes;
    bytes += static_cast<int>(align<size_t>(params.numRanks * sizeof(uint32_t), 16));
    bytes += kCftDestScratchBytes;
    if (lsaCft) {
        const int tokensPerCta = ll_lsa_cft_tokens_per_cta(params.maxTokensPerRank, numSms);
        if (tokensPerCta > kLlLsaCftMaxTokensPerCta || params.maxTokensPerRank >= (1 << kLlLsaCftCtaShift) ||
            2 * numSms >= (1 << kLlLsaCftArrivalBits) || params.numRanks > 72) {
            std::fprintf(stderr,
                         "ncclEpDispatch: LL LSA-CFT limits exceeded (tokens/CTA %d > %d, max tokens %d, SMs %d, ranks %d); "
                         "use staged output with NCCL_EP_LL_CFT_LSA=0\n",
                         tokensPerCta, kLlLsaCftMaxTokensPerCta, params.maxTokensPerRank, numSms, params.numRanks);
            return ncclInvalidArgument;
        }
        bytes += ll_lsa_cft_list_bytes(params.maxTokensPerRank, numSms, params.numTopk);
        bytes = static_cast<int>(align<size_t>(bytes, 16));
        bytes += ll_lsa_cft_hdr_batch_bytes(
            params.maxTokensPerRank, numSms, params.numTopk,
            static_cast<int>(get_dispatch_hdr_sz(params.numTopk, params.layout)));
    }
    if (bytes > params.maxDynamicSmem) {
        std::fprintf(stderr,
                     "ncclEpDispatch: LL CFT requires %d bytes of shared memory, device limit is %d bytes\n",
                     bytes, params.maxDynamicSmem);
        return ncclInvalidArgument;
    }
    *smemBytes = bytes;
    return ncclSuccess;
}

}  // namespace

// ============================================================================
// LL dispatch wrapper
//
//   - validates the workspace + numTopk/numExperts/numDeviceSms constraints
//   - chooses (numSms, numWarps) from the per-rank expert count
//   - packs the params + per-call flags into dispatch_kernel_args_t
//   - hands off to launch_ll_dispatch(), which JIT-compiles the kernel
//     specialised for (recipe, hidden, layout, nvlinkOnly)
// ============================================================================
ncclResult_t call_dispatch(
    const DispatchParams& params,
    ncclEpDispQuant_t recipe,
    cudaStream_t stream,
    bool nucs) {
    if (params.cftZeroCopy &&
        (params.cftPeerLeIds == nullptr || !ll_cft_zero_copy_supported(
            params.nvlinkOnly, recipe, params.layout, params.lsaMode, params.hidden, params.tokenDtype))) {
        std::fprintf(stderr, "ncclEpDispatch: CFT zero-copy requires the LSA dispatch kernel\n");
        return ncclInvalidUsage;
    }
    if (params.numTopk <= 0 || params.numTopk > MAX_NUM_TOPK) {
        std::fprintf(stderr, "ncclEpDispatch: LL top-k %d is outside [1, %d]\n", params.numTopk, MAX_NUM_TOPK);
        return ncclInvalidArgument;
    }
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    int numWarpsPerGroup = combine_smem::kWarpSize / numWarpGroups;
    const bool stageQuant = params.nvlinkOnly && params.layout == NCCL_EP_LAYOUT_RANK_MAJOR &&
        recipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4 &&
        ll_dispatch_stage_quant(params.hidden, params.maxDynamicSmem);
    if (stageQuant) {
        // Shared quantization needs fewer forwarding warps than the direct
        // per-peer path. Keep at least two receive warps per expert group and
        // enough forwarding warps for every top-k slot plus the control warp.
        const int compactWarpsPerGroup = kLlDsFp8CompactWarps / numWarpGroups;
        if (compactWarpsPerGroup >= 2 &&
            compactWarpsPerGroup * numWarpGroups >= params.numTopk + kLlDispatchControlWarps) {
            numWarpsPerGroup = compactWarpsPerGroup;
        }
    }
    if (numWarpGroups <= 0 || numWarpsPerGroup <= 0) {
        return ncclInvalidUsage;
    }

    int numWarps = numWarpGroups * numWarpsPerGroup;
    // One warp group per expert sets the receive grid; sends share that grid.
    // Widening it may displace overlapped model work, so benchmark with that
    // work present before changing the grid.
    const int numSms = ceil_div(params.numExperts, numWarpGroups);
    if (params.numTopk + kLlDispatchControlWarps > numWarps) {
        std::fprintf(stderr, "ncclEpDispatch: LL top-k %d needs %d forwarding/control warps, geometry provides %d\n",
                     params.numTopk, params.numTopk + kLlDispatchControlWarps, numWarps);
        return ncclInvalidUsage;
    }

    dispatch_kernel_args_t args{};
    args.inData = params.inData;
    args.inScalesBuf = params.inScalesBuf;
    args.inTopkIdx = params.inTopkIdx;
    args.inTopkWeights = params.inTopkWeights;
    args.rankMask = params.rankMask;
    args.asyncErrorFlag = params.asyncErrorFlag;
    args.outDataBuf = params.outDataBuf;
    args.outScalesBuf = params.outScalesBuf;
    args.outSrcInfo = params.outSrcInfo;
    args.outDstInfo = params.outDstInfo;
    args.outWant1Sided = params.outWant1Sided;
    args.outRecvRankCounter = params.outRecvRankCounter;
    args.outLayout = params.outLayout;
    args.outCnt = params.outCnt;
    args.outRecvTopkWeights = params.outRecvTopkWeights;
    args.outRecvTopkIdx = params.outRecvTopkIdx;
    args.rdmaBuf = params.rdmaBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvCntOff = params.recvCntOff;
    args.totalTokenOff = params.totalTokenOff;
    args.rankSentCnt = params.rankSentCnt;
    args.rankArrivedCnt = params.rankArrivedCnt;
    args.rankDone = params.rankDone;
    args.lsaSignalingCleanCount = params.lsaSignalingCleanCount;
    args.recvStats = params.recvStats;
    args.waitStats = params.waitStats;
    args.epochState = params.epochState;
    args.payloadSlotStride = params.payloadSlotStride;
    args.signalSlotStride = params.signalSlotStride;
    args.numTokens = params.numTokens;
    args.scalesPerToken = params.scalesPerToken;
    args.maxTokensPerRank = params.maxTokensPerRank;
    args.numExperts = params.numExperts;
    args.currRank = params.currRank;
    args.numRanks = params.numRanks;
    args.numAttnRanks = params.numAttnRanks;
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.roundScale = params.roundScale;
    args.combineLsaThreshold = params.combineLsaThreshold;
    args.recvTopkIdxKind = params.recvTopkIdxKind;
    args.phases = params.phases;
    args.devComm = params.devComm;
    args.windows = params.windows;
    args.signalsBase = params.signalsBase;
    args.timeoutCycles = params.timeoutCycles;
    args.recvDataWindow = params.recvDataWindow;
    args.recvDataOffset = params.recvDataOffset;
    args.rcvScalesWin = params.rcvScalesWin;
    args.rcvScalesOffs = params.rcvScalesOffs;
    args.cft.peerLeIds          = params.cftPeerLeIds;
    args.cft.recvBase           = params.cftRecvBase;
    args.cft.bytesCntrBaseOff   = params.cftBytesCntrBaseOff;
    args.cft.parityOff          = params.cftParityOff;
    args.cft.srcRankRegionBytes = params.cftSrcRankRegionBytes;
    args.cft.recvByteCounters   = params.cftRecvByteCounters;
    args.cft.cumulByteExpect    = params.cftCumulByteExpect;
    args.cft.outputOffset       = params.cftOutputOffset;
    args.cft.countedPayload = cft_counted_payload(params);

    DispatchKernelSpec kernel_spec;
    ncclResult_t r = resolveDispatchKernelSpec(
        recipe, params.tokenDtype, params.scaleDtype, &kernel_spec);
    if (r != ncclSuccess) {
        return r;
    }

    // Select the CFT JIT variant only for supported dispatch configurations.
    const bool cft_active = (params.cftPeerLeIds != nullptr) &&
                            (recipe == NCCL_EP_DISP_QUANT_NONE) &&
                            (params.cftZeroCopy || params.recvDataWindow == ncclWindow_t{}) &&
                            (params.rcvScalesWin == ncclWindow_t{}) &&
                            (params.rankMask == nullptr) &&
                            (params.tokenDtype != ncclFloat32);

    const bool lsa_cft_active = cft_active &&
        jit::ll_dispatch_select_algo(params.nvlinkOnly, recipe, params.layout, /*cftEnabled=*/true, params.lsaMode,
                                     params.hidden, params.tokenDtype) == jit::LlDispatchAlgo::kCftRmLsa;
    if (lsa_cft_active) {
        numWarpsPerGroup = ll_lsa_cft_choose_warps_per_group(
            numWarpsPerGroup, params.maxTokensPerRank, params.numExperts / params.numRanks);
        numWarps = numWarpGroups * numWarpsPerGroup;
        args.numWarpsPerGroup = numWarpsPerGroup;
    }

    // The qualified E384 grid fits at most 43 tokens/CTA in one mask word.
    const bool cftOneWordMask = lsa_cft_active &&
        params.deviceSm == 107 && params.deviceSmCount == 212 &&
        params.numDeviceSms == static_cast<int>(params.deviceSmCount) &&
        params.numRanks == 4 && params.numExperts == 384 && params.hidden == 7168 &&
        params.numTopk == 6 && params.tokenDtype == ncclBfloat16 && params.topkIdxIsInt64 &&
        params.maxTokensPerRank >= 2048 && params.maxTokensPerRank <= 8192 &&
        numWarpGroups == 2 && numSms == 192 && numWarps == 32 && numWarpsPerGroup == 16;

    int cft_smem_bytes = 0;
    if (cft_active) {
        r = cft_dispatch_smem_bytes(params, lsa_cft_active, numSms, &cft_smem_bytes);
        if (r != ncclSuccess) {
            return r;
        }
    }

    return jit::launch_ll_dispatch(
        params.hidden,
        params.deviceSm,
        params.layout,
        params.nvlinkOnly,
        params.topkIdxIsInt64,
        kernel_spec,
        params.tokenDtype,
        recipe,
        params.numTopk,
        numSms,
        numWarps,
        stageQuant,
        cft_smem_bytes,
        params.lsaMode,
        params.tmaQuantWarps,
        params.maxDynamicSmem,
        params.recordDstInfo,
        cft_active && params.cftZeroCopy,
        args,
        stream,
        nucs,
        params.dstInfoDisqualifiedOut,
        cftOneWordMask);
}

// ============================================================================
// LL combine wrapper
//
// Resolves (numSms, numWarps), computes the dynamic SMEM budget, packs args,
// and hands off to launch_ll_combine() for JIT compile + launch.
// ============================================================================
ncclResult_t call_combine(const CombineParams& params, cudaStream_t stream, bool nucs) {
    if (params.numDeviceSms <= 0 || params.numExperts <= 0 || params.numCombinedTokens < 0) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine requires positive device SMs and experts, and non-negative combined tokens: "
            "device_sms=%d, experts=%d, combined_tokens=%d.\n",
            params.numDeviceSms, params.numExperts, params.numCombinedTokens);
        return ncclInvalidArgument;
    }
    if (params.numTopk <= 0 || params.numTopk > MAX_NUM_TOPK || params.numTopk > combine_smem::kWarpSize) {
        std::fprintf(stderr, "ncclEpCombine: LL top-k %d is outside [1, %d]\n", params.numTopk, MAX_NUM_TOPK);
        return ncclInvalidArgument;
    }
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    const int requestedWarpsPerGroup = combine_smem::kWarpSize / numWarpGroups;
    const int numRecvPerSm = ceil_div(params.numCombinedTokens, params.numDeviceSms);
    if (numWarpGroups <= 0 || requestedWarpsPerGroup <= 0 || numRecvPerSm < 0) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine produced an invalid launch configuration: warp_groups=%d, "
            "requested_warps_per_group=%d, recv_per_sm=%d.\n",
            numWarpGroups, requestedWarpsPerGroup, numRecvPerSm);
        return ncclInvalidArgument;
    }

    // Reserve room for the LSA combine kernel's static __shared__ usage,
    // which draws from the same per-block budget as the dynamic portion
    // sized below -- see choose_combine_smem_config's doc comment. Only the
    // LSA path declares that static usage, so only trim the budget when
    // this call is actually eligible for it.
    //
    // call_combine is the sole place either eligibility question is evaluated and a request is
    // resolved to a kernel -- see ll_combine_select_kernel's doc comment. An enforced LSA kernel
    // falls back to kDefault silently when the call isn't LSA-eligible at all (NCCL_EP_COMBINE_LSA
    // only governs the pick within the LSA domain); kInvalid comes back only for an in-domain
    // request it still can't honor (e.g. 1-sided enforced but not oneSidedEligible).
    const bool lsa_eligible = jit::ll_combine_lsa_eligible(
        params.nvlinkOnly, params.quantizationRecipe, params.useLogFmt, params.layout, params.cftEnabled,
        params.hidden, params.tokenDtype, params.lsaMode, params.numAttnRanks);
    // oneSidedEligible implies lsaEligible by construction (AND-ed on top, never asserted
    // independently) -- see CombineParams::groupZeroCopyOn's doc comment for why these two extra
    // conditions live here rather than being precomputed by the caller.
    const bool one_sided_eligible = lsa_eligible && !params.cftEnabled && params.tokenDtype == ncclBfloat16 &&
        params.groupZeroCopyOn && params.inDataWindow != ncclWindow_t{} && !params.dispatch1SidedDisqualified;
    const LlCombineKernel combine_kernel =
        jit::ll_combine_select_kernel(/*enforced_lsa=*/params.enforced, lsa_eligible, one_sided_eligible);
    if (combine_kernel == LlCombineKernel::kInvalid) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine: NCCL_EP_COMBINE_LSA enforced kernel %s could not be honored for this call "
            "(lsa_eligible=%d, one_sided_eligible=%d, bf16=%d, zero_copy=%d, windowed=%d, "
            "dispatch_universal=%d).\n",
            jit::ll_combine_kernel_name(params.enforced), static_cast<int>(lsa_eligible),
            static_cast<int>(one_sided_eligible), static_cast<int>(params.tokenDtype == ncclBfloat16),
            static_cast<int>(params.groupZeroCopyOn), static_cast<int>(params.inDataWindow != ncclWindow_t{}),
            static_cast<int>(params.dispatch1SidedDisqualified));
        return ncclInvalidArgument;
    }

    const bool lsaCombineEligible = combine_kernel == LlCombineKernel::k2SidedRmLsa ||
        combine_kernel == LlCombineKernel::k1SidedRmLsa || combine_kernel == LlCombineKernel::kAutoRmLsa;
    const int max_dynamic_smem_for_combine =
        params.maxDynamicSmem - (lsaCombineEligible ? combine_smem::kLsaStaticSmemBytes : 0);
    const combine_smem_config_t smem_config = choose_combine_smem_config(
        params.hidden,
        params.tokenDtype,
        params.quantizationRecipe,
        numWarpGroups,
        requestedWarpsPerGroup,
        max_dynamic_smem_for_combine);
    if (!smem_config.feasible) {
        const int elem_bytes = params.tokenDtype == ncclFloat32 ? 4 : 2;
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine shared memory cannot fit: hidden=%d, dtype=%d, warp_groups=%d, "
            "requested_warps_per_group=%d, limit=%d bytes (receive group needs %d decode warps + %d TMA warp "
            "at recv_unrolls=%d; the CTA has at most %d warps).\n",
            params.hidden, static_cast<int>(params.tokenDtype), numWarpGroups, requestedWarpsPerGroup,
            max_dynamic_smem_for_combine, combine_smem::recv_decode_warps(params.hidden, elem_bytes),
            combine_smem::kNumRecvTmaWarps, combine_smem::recv_unrolls(params.hidden, elem_bytes),
            numWarpGroups * requestedWarpsPerGroup);
        return ncclInvalidArgument;
    }
    const int numWarpsPerGroup = smem_config.num_warps_per_group;
    const int numWarps = smem_config.num_warps;
    const int smem_size = smem_config.dynamic_smem_bytes;
    if (params.resolvedWarpsPerGroup != nullptr) *params.resolvedWarpsPerGroup = numWarpsPerGroup;
    const int numSms = std::max(
        ceil_div(params.numExperts, numWarpGroups),
        numRecvPerSm == 0 ? 1 : ceil_div(params.numCombinedTokens, numRecvPerSm));

    // combineSync is a dedicated, non-overlapping region computed once at
    // group-creation time (ncclEpCreateGroup) -- forwarded here as-is, never
    // derived via offset math.
    if (params.combineSync == nullptr) {
        std::fprintf(stderr, "[nccl_ep] LL combine requires a non-null combineSync workspace pointer.\n");
        return ncclInvalidArgument;
    }
    if (params.zeroCopy && params.useLogFmt) {
        std::fprintf(stderr, "[nccl_ep] LL combine does not support zero-copy with LogFMT.\n");
        return ncclInvalidArgument;
    }

    auto combineSync = params.combineSync;

    const int hidden = params.hidden;

    combine_kernel_args_t args{};
    args.inData = params.inData;
    args.inGlobalScales = params.inGlobalScales;
    args.srcInfo = params.srcInfo;
    args.dstInfo = params.dstInfo;
    args.inDataWindow = params.inDataWindow;
    args.inDataWindowOffset = params.inDataWindowOffset;
    args.layoutRange = params.layoutRange;
    args.inTopkIdx = params.inTopkIdx;
    args.topkWeights = params.topkWeights;
    args.rankMask = params.rankMask;
    args.asyncErrorFlag = params.asyncErrorFlag;
    args.outData = params.outData;
    args.rdmaBuf = params.rdmaBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvFlagOff = params.recvFlagOff;
    args.readDoneOffBase = params.readDoneOffBase;
    args.combineSync = combineSync;
    args.want1SidedBuf = params.want1SidedBuf;
    args.lsaSignalingCleanCount = params.lsaSignalingCleanCount;
    args.waitStats = params.waitStats;
    args.epochState = params.epochState;
    args.payloadSlotStride = params.payloadSlotStride;
    args.signalSlotStride = params.signalSlotStride;
    args.numCombinedTokens = params.numCombinedTokens;
    args.hidden = hidden;
    args.maxTokensPerRank = params.maxTokensPerRank;
    args.numExperts = params.numExperts;
    args.currRank = params.currRank;
    args.numRanks = params.numRanks;
    args.numAttnRanks = params.numAttnRanks;
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.phases = params.phases;
    args.zeroCopy = params.zeroCopy;
    args.devComm = params.devComm;
    args.windows = params.windows;
    args.signalsBase = params.signalsBase;
    args.timeoutCycles = params.timeoutCycles;

    return jit::launch_ll_combine(
        params.useLogFmt,
        params.quantizationRecipe,
        params.deviceSm,
        hidden,
        params.layout,
        params.topkIdxIsInt64,
        params.tokenDtype,
        params.numTopk,
        numSms,
        numWarps,
        smem_size,
        combine_kernel,
        args,
        stream,
        nucs);
}

// ============================================================================
// LL buffer-clean wrapper
// ============================================================================
ncclResult_t
call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream, bool nucs) {
    clean_low_latency_buffer_kernel_args_t args{};
    args.clean_0 = params.clean_0;
    args.num_clean_int_0 = params.num_clean_int_0;
    args.clean_1 = params.clean_1;
    args.num_clean_int_1 = params.num_clean_int_1;
    args.rankMask = params.rankMask;
    args.syncBuffer = params.syncBuffer;
    args.syncWindow = params.syncWindow;
    args.devComm = params.devComm;
    args.barrierSignalBase = params.barrierSignalBase;
    args.timeoutCycles = params.timeoutCycles;

    return jit::launch_ll_clean_low_latency_buffer(args, params.deviceSm, stream, nucs);
}

} // namespace ll
} // namespace nccl_ep
