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
#include <atomic>

namespace nccl_ep {
namespace ll {

// ----------------------------------------------------------------------------
// LL P2P signal generations.
//
// The LL count/flag slots are addressed by offsets into the group's rdma_buffer,
// but the parity that selects them (`handle->ll.buffer_idx`) is per-handle. Two
// handles on one group therefore alias the same slots while advancing parity
// independently, so a poll can observe a sibling handle's value -- and because the
// values are workload-determined (finish flags are the constant 1) it is
// indistinguishable from the one being waited for.
//
// Stamping every launch with a monotonically increasing generation, and accepting a
// signal only when its generation matches, makes any foreign or leftover value inert.
// Ranks run the same LL call sequence, so a per-process counter stays consistent
// across them; a SEND-phase launch advances it and a RECV-only launch reuses it.
// ----------------------------------------------------------------------------
namespace {
std::atomic<unsigned> gLlDispatchGen{0};
std::atomic<unsigned> gLlCombineGen{0};
// Never 0, so an all-zero (freshly cleaned) slot can never look like a valid signal.
inline unsigned llNextGen(std::atomic<unsigned>& ctr, int phases) {
    unsigned c = (phases & LOW_LATENCY_SEND_PHASE)
                     ? ctr.fetch_add(1, std::memory_order_relaxed) + 1
                     : ctr.load(std::memory_order_relaxed);
    return (c - 1u) % 0x7fffu + 1u;
}
}  // namespace

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
    args.sendBuf = params.sendBuf;
    args.recvBuf = params.recvBuf;
    args.recvCntBuf = params.recvCntBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvCntOff = params.recvCntOff;
    args.rankCountersBase = rankCountersBase;
    args.rankDone = rankDone;
    args.nextRecvCntBuf = params.nextRecvCntBuf;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
    args.recvStats = params.recvStats;
    args.waitStats = params.waitStats;
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
    args.signalGen = llNextGen(gLlDispatchGen, params.phases);
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
void call_combine(const CombineParams& params, cudaStream_t stream) {
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    const int numRecvPerSm = ceil_div(params.numCombinedTokens, params.numDeviceSms);
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0 and numRecvPerSm >= 0);

    const int numWarps = numWarpGroups * numWarpsPerGroup;
    const int numSms = std::max(
        ceil_div(params.numExperts, numWarpGroups),
        numRecvPerSm == 0 ? 1 : ceil_div(params.numCombinedTokens, numRecvPerSm));

    auto atomicCleanFlag = static_cast<int*>(params.workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(params.numTopk <= jit::kLlCombineMaxTopk);

    // Online cast (LogFMT) is incompatible with zero-copy.
    EP_HOST_ASSERT(not(params.zeroCopy and params.useLogFmt));

    // Per-block SMEM = max(send-side TMA staging, recv-side TMA staging).
    // Send side: numWarps × kNumStages TMA buffers + per-warp LogFMT metadata.
    // Recv side: kMaxNumGroups × (kNumStages TMA buffers + decoded output +
    //            kNumStages × LogFMT decode metadata).
    constexpr int kNumStages = 3;
    // Must mirror the kernel's group count exactly (see combine_kernel_impl
    // kMaxNumGroups): FP32 doubles per-stage token bytes, so it runs 1 group to
    // stay within the device dynamic-SMEM cap; BF16/FP16 run 2. Computing 2 for
    // FP32 would over-request SMEM and fail the func attribute at large hidden.
    const int elemBytes = (params.tokenDtype == ncclFloat32) ? 4 : 2;
    const int kMaxNumGroups = (elemBytes == 2) ? 2 : 1;
    const int hidden = params.hidden;
    const int numMetaBytes = hidden / 128 * 4;
    const int numSendTmaBytes = 32 * static_cast<int>(sizeof(int4)) * jit::kLlCombineMaxUnrolls + 16;
    const int smemSendSize = numWarps * (kNumStages * numSendTmaBytes + numMetaBytes);
    const int numRecvTmaBytes = 16 + hidden * elemBytes;
    const int smemRecvSize =
        kMaxNumGroups * (kNumStages * numRecvTmaBytes + hidden * elemBytes + kNumStages * numMetaBytes * 3);
    const int smem_size = std::max(smemSendSize, smemRecvSize);

    combine_kernel_args_t args{};
    args.inData = params.inData;
    args.srcInfo = params.srcInfo;
    args.layoutRange = params.layoutRange;
    args.inTopkIdx = params.inTopkIdx;
    args.topkWeights = params.topkWeights;
    args.rankMask = params.rankMask;
    args.asyncErrorFlag = params.asyncErrorFlag;
    args.outData = params.outData;
    args.sendBuf = params.sendBuf;
    args.recvBuf = params.recvBuf;
    args.recvFlagBuf = params.recvFlagBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvFlagOff = params.recvFlagOff;
    args.atomicCleanFlag = atomicCleanFlag;
    args.nextRecvCntBuf = params.nextRecvCntBuf;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
    args.waitStats = params.waitStats;
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
    args.signalGen = llNextGen(gLlCombineGen, params.phases);
    args.timeoutCycles = params.timeoutCycles;

    jit::launch_ll_combine(
        // LogFMT compression is not wired into the current code flow; force it
        // off until it is revisited (the template plumbing is kept).
        /*useLogFmt=*/false,
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
