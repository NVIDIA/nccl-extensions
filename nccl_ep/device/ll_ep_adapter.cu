/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/ll_ep_adapter.cuh"
#include "device/ll_ep.cuh"
#include "device/macros.cuh"
#include "common.hpp"
#include "jit/ll_dispatch_jit.cuh"
#include "jit/ll_combine_jit.cuh"
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
    constexpr int kNumMaxTopK = 9;
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= numWarpGroups * numWarpsPerGroup);

    const int numWarps = numWarpGroups * numWarpsPerGroup;
    const int numSms = ceil_div(params.numExperts, numWarpGroups);
    EP_HOST_ASSERT(params.numTopk <= kNumMaxTopK);

    // Workspace: [rankSentCnt | rankArrivedCnt | rankDone(=expertDone)].
    auto rankCountersBase = static_cast<int*>(params.workspace);
    auto rankDone = rankCountersBase + 2 * params.numRanks;
    EP_HOST_ASSERT((2 * params.numRanks + params.numExperts) * sizeof(int) <= NUM_WORKSPACE_BYTES);

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
    args.rankCountersBase = rankCountersBase;
    args.rankDone = rankDone;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
    args.recvStats = params.recvStats;
    args.waitStats = params.waitStats;
    args.epochState = params.epochState;
    args.payloadSlotStride = params.payloadSlotStride;
    args.signalSlotStride = params.signalSlotStride;
    args.numTokens = params.numTokens;
    args.scalesPerToken = params.scalesPerToken;
    args.maxTokensPerRank = params.maxTokensPerRank;
    args.numTopk = params.numTopk;
    args.numExperts = params.numExperts;
    args.currRank = params.currRank;
    args.numRanks = params.numRanks;
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.roundScale = params.roundScale;
    args.recvTopkIdxKind = params.recvTopkIdxKind;
    args.phases = params.phases;
    args.numComms = params.numComms;
    args.devComms = params.devComms;
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
        numSms,
        numWarps,
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
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    const int requestedWarpsPerGroup = 32 / numWarpGroups;
    const int numRecvPerSm = ceil_div(params.numCombinedTokens, params.numDeviceSms);
    if (numWarpGroups <= 0 || requestedWarpsPerGroup <= 0 || numRecvPerSm < 0) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine produced an invalid launch configuration: warp_groups=%d, "
            "requested_warps_per_group=%d, recv_per_sm=%d.\n",
            numWarpGroups, requestedWarpsPerGroup, numRecvPerSm);
        return ncclInvalidArgument;
    }

    const combine_smem_config_t smem_config = choose_combine_smem_config(
        params.hidden,
        params.tokenDtype,
        params.quantizationRecipe,
        numWarpGroups,
        requestedWarpsPerGroup,
        params.maxDynamicSmem);
    if (!smem_config.feasible) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine shared memory cannot fit: hidden=%d, dtype=%d, warp_groups=%d, "
            "requested_warps_per_group=%d, limit=%d bytes.\n",
            params.hidden, static_cast<int>(params.tokenDtype), numWarpGroups, requestedWarpsPerGroup,
            params.maxDynamicSmem);
        return ncclInvalidArgument;
    }
    const int numWarpsPerGroup = smem_config.num_warps_per_group;
    const int numWarps = smem_config.num_warps;
    if (params.resolvedWarpsPerGroup != nullptr) *params.resolvedWarpsPerGroup = numWarpsPerGroup;
    const int numSms = std::max(
        ceil_div(params.numExperts, numWarpGroups),
        numRecvPerSm == 0 ? 1 : ceil_div(params.numCombinedTokens, numRecvPerSm));

    if (NUM_WORKSPACE_BYTES < sizeof(int) || params.workspace == nullptr) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine requires at least %zu workspace bytes for its atomic flag; available=%d, "
            "workspace=%p.\n",
            sizeof(int), NUM_WORKSPACE_BYTES, params.workspace);
        return ncclInvalidArgument;
    }
    if (params.numTopk < 0 || params.numTopk > jit::kLlCombineMaxTopk) {
        std::fprintf(
            stderr,
            "[nccl_ep] LL combine top-k is unsupported: num_topk=%d, maximum=%d.\n",
            params.numTopk, jit::kLlCombineMaxTopk);
        return ncclInvalidArgument;
    }
    if (params.zeroCopy && params.useLogFmt) {
        std::fprintf(stderr, "[nccl_ep] LL combine does not support zero-copy with LogFMT.\n");
        return ncclInvalidArgument;
    }

    auto atomicCleanFlag = static_cast<int*>(params.workspace);

    const int hidden = params.hidden;
    const int smem_size = smem_config.dynamic_smem_bytes;

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
    args.atomicCleanFlag = atomicCleanFlag;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
    args.waitStats = params.waitStats;
    args.epochState = params.epochState;
    args.payloadSlotStride = params.payloadSlotStride;
    args.signalSlotStride = params.signalSlotStride;
    args.numCombinedTokens = params.numCombinedTokens;
    args.hidden = hidden;
    args.numTopk = params.numTopk;
    args.maxTokensPerRank = params.maxTokensPerRank;
    args.numExperts = params.numExperts;
    args.currRank = params.currRank;
    args.numRanks = params.numRanks;
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.phases = params.phases;
    args.zeroCopy = params.zeroCopy;
    args.numComms = params.numComms;
    args.devComms = params.devComms;
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
        numSms,
        numWarps,
        smem_size,
        args,
        stream);
}

// ============================================================================
// LL buffer-clean kernel (precompiled).
//
// Unlike dispatch/combine, clean has no runtime template parameters, so it is
// statically compiled and launched directly (one cooperative block) instead of
// going through JIT.
// ============================================================================
constexpr int kLlCleanNumThreads = 256;

__launch_bounds__(kLlCleanNumThreads, 1) __global__ void ll_clean_low_latency_buffer_kernel(
    const __grid_constant__ clean_low_latency_buffer_kernel_args_t p) {
    clean_low_latency_buffer_kernel_impl<kLlCleanNumThreads>(
        p.clean_0, p.num_clean_int_0,
        p.clean_1, p.num_clean_int_1,
        p.rankMask,
        p.syncBuffer, p.syncWindow,
        p.devComms, p.barrierSignalBase, p.timeoutCycles);
}

// ============================================================================
// LL buffer-clean wrapper
// ============================================================================
void call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream) {
    clean_low_latency_buffer_kernel_args_t args{};
    args.clean_0 = params.clean_0;
    args.num_clean_int_0 = params.num_clean_int_0;
    args.clean_1 = params.clean_1;
    args.num_clean_int_1 = params.num_clean_int_1;
    args.rankMask = params.rankMask;
    args.syncBuffer = params.syncBuffer;
    args.syncWindow = params.syncWindow;
    args.devComms = params.devComms;
    args.barrierSignalBase = params.barrierSignalBase;
    args.timeoutCycles = params.timeoutCycles;

    SETUP_LAUNCH_CONFIG(1, kLlCleanNumThreads, stream);
    LAUNCH_KERNEL(&cfg, ll_clean_low_latency_buffer_kernel, args);
}

} // namespace ll
} // namespace nccl_ep
