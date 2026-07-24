/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "nccl_ep.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "nccl_device.h"
#include "ep_enums.h"
#include "common.hpp"

namespace nccl_ep {
namespace ll {

// ============================================================================
// Packed kernel parameter structs.
//
// These structs are written by the host-side adapters and consumed by the
// JIT-compiled kernel entry points. The JIT entry receives the struct as
// `const __grid_constant__ <struct> p`, then unpacks `p.foo` into the
// templated `*_kernel_impl(...)` device function defined in ll_ep.cuh.
//
// The struct layouts must match what the kernel expects, so any field added
// here must also be threaded through the matching kernel entry: the JIT sources
// device/jit/ll_dispatch_jit.cuh / ll_combine_jit.cuh for dispatch and combine,
// and the precompiled clean kernel in device/ll_ep_adapter.cu.
// ============================================================================

struct dispatch_kernel_args_t {
    // INPUT
    const void* inData;
    const void* inScalesBuf;      // non-null for SCALES_FORWARD; runtime-typed storage
    const void* inTopkIdx;        // cast to const TopkIdxT* by the JIT entry
    const float* inTopkWeights;
    int* rankMask;
    int* asyncErrorFlag;
    // OUTPUT
    void* outDataBuf;
    void* outScalesBuf;
    int* outSrcInfo;
    int* outRecvRankCounter;
    int64_t* outLayout;
    int* outCnt;
    float* outRecvTopkWeights;
    int32_t* outRecvTopkIdx;
    // INTERMEDIATE
    void* sendBuf;
    void* recvBuf;
    int* recvCntBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvCntOff;
    int* rankCountersBase;
    int* rankDone;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int* recvStats;
    int64_t* waitStats;
    // CONFIG
    int numTokens;
    // Derived from validated inputs->scales->sizes[1].
    int scalesPerToken;
    int maxTokensPerRank;
    int numTopk;
    int numExperts;
    int currRank;
    int numRanks;
    int numWarpGroups;
    int numWarpsPerGroup;
    bool roundScale;
    // recv_topk_idx numbering (LOCAL/GLOBAL); resolved on the host (never AUTO).
    ncclEpExpertIdKind_t recvTopkIdxKind;
    int phases;
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;
    uint64_t timeoutCycles;
    // Non-null windows enable peer payload writes.
    ncclWindow_t recvDataWindow;
    size_t recvDataOffset;
    ncclWindow_t rcvScalesWin;
    size_t rcvScalesOffs;
};

struct combine_kernel_args_t {
    // INPUT
    const void* inData;
    const int* srcInfo;
    const int64_t* layoutRange;
    const void* inTopkIdx;        // cast to const TopkIdxT* by the JIT entry
    const float* topkWeights;
    int* rankMask;
    int* asyncErrorFlag;
    // OUTPUT
    void* outData;
    // INTERMEDIATE
    void* sendBuf;
    void* recvBuf;
    int* recvFlagBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvFlagOff;
    int* atomicCleanFlag;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int64_t* waitStats;
    // CONFIG
    int numCombinedTokens;
    int hidden;
    int numTopk;
    int maxTokensPerRank;
    int numExperts;
    int currRank;
    int numRanks;
    int numWarpGroups;
    int numWarpsPerGroup;
    int phases;
    bool zeroCopy;
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;
    uint64_t timeoutCycles;
};

struct clean_low_latency_buffer_kernel_args_t {
    int* clean_0;
    int num_clean_int_0;
    int* clean_1;
    int num_clean_int_1;
    int* rankMask;
    int* syncBuffer;
    ncclWindow_t* syncWindow;
    ncclDevComm* devComms;
    unsigned barrierSignalBase;
    uint64_t timeoutCycles;
};

// ============================================================================
// Public host-side parameter structs.
//
// Callers fill in a *Params struct once and pass it to call_dispatch /
// call_combine / call_clean_low_latency_buffer along with the few flags that
// vary per call (template-affecting bools, phases, stream). Grouping the many
// arguments into a struct keeps the call sites readable and lets fields default.
// ============================================================================

struct DispatchParams {
    // User inputs
    const void* inData;
    const void* inScalesBuf = nullptr;     // non-null for SCALES_FORWARD; runtime-typed storage
    const void* inTopkIdx;                  // int32_t* or int64_t*; see topkIdxIsInt64
    bool topkIdxIsInt64 = true;             // selects the TopkIdxT kernel specialization
    // Derived from inputs->scales->sizes[1] by ncclEpDispatch.
    int scalesPerToken = 0;
    ncclDataType_t scaleDtype;              // SCALES_FORWARD tensor dtype; selects ScaleT in JIT
    const float* inTopkWeights;

    // User / pre-allocated output buffers
    void* outDataBuf;
    void* outScalesBuf;
    int* outSrcInfo;
    int* outRecvRankCounter;       // rank-major only; nullptr otherwise
    int64_t* outLayout;
    int* outCnt;
    float* outRecvTopkWeights;     // rank-major only; nullptr otherwise
    int32_t* outRecvTopkIdx;       // rank-major only; nullptr otherwise

    // Intermediate RDMA buffers + window-relative offsets
    void* sendBuf;
    void* recvBuf;
    int* recvCntBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvCntOff;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int* recvStats;
    int64_t* waitStats;

    // Sizes / identifiers
    int numTokens;
    int hidden;
    int maxTokensPerRank;
    int numTopk;
    int numExperts;
    int currRank;
    int numRanks;
    ncclEpLayout_t layout;

    // GIN / NCCL device context
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;

    // Runtime workspace + error tracking
    void* workspace;
    int numDeviceSms;
    int* rankMask = nullptr;
    int* asyncErrorFlag = nullptr;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;

    // Per-call behavior toggles that do not affect the quantization recipe.
    bool roundScale = false;
    bool nvlinkOnly = false;
    // recv_topk_idx numbering; the host wrapper resolves AUTO -> LOCAL before
    // launch, so the kernel only ever sees LOCAL or GLOBAL.
    ncclEpExpertIdKind_t recvTopkIdxKind = NCCL_EP_EXPERT_ID_LOCAL;
    int phases = 0;

    // Zero-copy dispatch output (rank-major + nvlinkOnly). Each supplied
    // token or SCALES_FORWARD scale window is written directly to its peer output.
    ncclWindow_t recvDataWindow = ncclWindow_t{};
    size_t recvDataOffset = 0;
    ncclWindow_t rcvScalesWin = ncclWindow_t{};
    size_t rcvScalesOffs = 0;

    // Actual token wire dtype. Selects the kTokenDtype kernel specialization;
    // FP16 and BF16 share the same two-byte copy specialization.
    ncclDataType_t tokenDtype = ncclBfloat16;
};

struct CombineParams {
    // User inputs
    const void* inData;
    const int* srcInfo;
    const int64_t* layoutRange;
    const void* inTopkIdx;        // int32_t* or int64_t*; see topkIdxIsInt64
    bool topkIdxIsInt64 = true;   // selects the TopkIdxT kernel specialization
    const float* topkWeights;

    // User output
    void* outData;

    // Intermediate RDMA buffers + window-relative offsets
    void* sendBuf;
    void* recvBuf;
    int* recvFlagBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvFlagOff;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int64_t* waitStats;

    // Sizes / identifiers
    int numCombinedTokens;
    int hidden;
    int maxTokensPerRank;
    int numTopk;
    int numExperts;
    int currRank;
    int numRanks;
    ncclEpLayout_t layout;

    // GIN / NCCL device context
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;

    // Runtime workspace + error tracking
    void* workspace;
    int numDeviceSms;
    int* rankMask = nullptr;
    int* asyncErrorFlag = nullptr;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;

    // Per-call behavior toggles that select the JIT kernel specialization.
    bool useLogFmt = false;
    bool zeroCopy = false;
    int phases = 0;

    // Token wire dtype (unquantized payload width). Combine decodes + reduces,
    // so BF16/FP16/FP32 are three distinct kernel specializations (FP32 also
    // halves kMaxNumGroups to stay within the dynamic-SMEM cap).
    ncclDataType_t tokenDtype = ncclBfloat16;
};

struct CleanLowLatencyBufferParams {
    int* clean_0;
    int num_clean_int_0;
    int* clean_1;
    int num_clean_int_1;
    int* rankMask;
    int* syncBuffer;
    ncclWindow_t* syncWindow;
    ncclDevComm* devComms;
    unsigned barrierSignalBase;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;
};

// ============================================================================
// Host-side wrappers.
//
// Each wrapper resolves all runtime template parameters (hidden, layout,
// nvlinkOnly and useLogFmt) and dispatches to a per-variant
// JIT-compiled kernel.
// ============================================================================

ncclResult_t call_dispatch(
    const DispatchParams& params,
    ncclEpDispatchQuantizationRecipe_t quantization_recipe,
    cudaStream_t stream = 0);

void call_combine(const CombineParams& params, cudaStream_t stream = 0);

void call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream = 0);

} // namespace ll
} // namespace nccl_ep
