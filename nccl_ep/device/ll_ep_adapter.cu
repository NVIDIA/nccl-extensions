/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/ll_ep_adapter.cuh"
#include "device/macros.cuh"
#include "common.hpp"
#include "jit/ll_dispatch_jit.cuh"
#include "jit/ll_combine_jit.cuh"
#include "jit/ll_clean_jit.cuh"
#include "quantization_recipe.hpp"

#include <algorithm>
#include <cstdio>

namespace nccl_ep {
namespace ll {

// Forward-declare the host-side `ceil_div` helper used here. `common.hpp`
// provides templated ceil_div in namespace nccl_ep; we reuse it via ADL.
using ::nccl_ep::ceil_div;

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
    cudaStream_t stream) {
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
    if (numWarpGroups <= 0 || numWarpsPerGroup <= 0) return ncclInvalidUsage;

    const int numWarps = numWarpGroups * numWarpsPerGroup;
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
    args.outRecvRankCounter = params.outRecvRankCounter;
    args.outLayout = params.outLayout;
    args.outCnt = params.outCnt;
    args.outRecvTopkWeights = params.outRecvTopkWeights;
    args.outRecvTopkIdx = params.outRecvTopkIdx;
    args.rdmaBuf = params.rdmaBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvCntOff = params.recvCntOff;
    args.rankSentCnt = params.rankSentCnt;
    args.rankArrivedCnt = params.rankArrivedCnt;
    args.rankDone = params.rankDone;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
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
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.roundScale = params.roundScale;
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

    DispatchKernelSpec kernel_spec;
    ncclResult_t r = resolveDispatchKernelSpec(
        recipe, params.tokenDtype, params.scaleDtype, &kernel_spec);
    if (r != ncclSuccess) {
        return r;
    }

    return jit::launch_ll_dispatch(
        params.hidden,
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
        args,
        stream);
}

// ============================================================================
// LL combine wrapper
//
// Resolves (numSms, numWarps), computes the dynamic SMEM budget, packs args,
// and hands off to launch_ll_combine() for JIT compile + launch.
// ============================================================================
ncclResult_t call_combine(const CombineParams& params, cudaStream_t stream) {
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
    const bool lsaCombineEligible =
        jit::ll_combine_select_algo(params.nvlinkOnly, params.quantizationRecipe, params.useLogFmt, params.layout) ==
        jit::LlCombineAlgo::k2SidedRmLsa;
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
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine shared memory cannot fit: hidden=%d, dtype=%d, warp_groups=%d, "
            "requested_warps_per_group=%d, limit=%d bytes.\n",
            params.hidden, static_cast<int>(params.tokenDtype), numWarpGroups, requestedWarpsPerGroup,
            max_dynamic_smem_for_combine);
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
    args.combineSync = combineSync;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
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
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.phases = params.phases;
    args.zeroCopy = params.zeroCopy;
    args.devComm = params.devComm;
    args.windows = params.windows;
    args.signalsBase = params.signalsBase;
    args.timeoutCycles = params.timeoutCycles;

    return jit::launch_ll_combine(
        params.nvlinkOnly,
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
        args,
        stream);
}

// ============================================================================
// LL buffer-clean wrapper
// ============================================================================
ncclResult_t call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream) {
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

    return jit::launch_ll_clean_low_latency_buffer(args, stream);
}

} // namespace ll
} // namespace nccl_ep
