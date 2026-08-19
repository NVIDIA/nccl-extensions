/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "nccl_ep.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "nccl_device.h"
#include "ep_enums.h"
#include "common.hpp"
#include "ll_ep_smem.cuh"

namespace nccl_ep {
namespace ll {

// LL combine uses dynamic shared memory for independent send and receive
// phases. Keep this host-side accounting next to the launch parameters so it
// can be checked before configuring a JIT kernel.
struct combine_smem_config_t {
    int num_warps_per_group;
    int num_warps;
    int num_recv_groups;
    int dynamic_smem_bytes;
    bool feasible;
};

inline int ll_combine_dynamic_smem_bytes(
    int hidden,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t quantization_recipe,
    int num_warps,
    int* num_recv_groups) {
    const int elem_bytes = token_dtype == ncclFloat32 ? 4 : 2;
    const bool nvfp4 = quantization_recipe == NCCL_EP_COMB_QUANT_NVFP4;
    const int decode_warps = combine_smem::recv_decode_warps(hidden, elem_bytes);
    const int recv_groups = std::min(combine_smem::max_recv_groups(elem_bytes), num_warps / (decode_warps + 1));
    if (num_recv_groups != nullptr) *num_recv_groups = recv_groups;
    return combine_smem::dynamic_smem_bytes(hidden, elem_bytes, num_warps, recv_groups, nvfp4);
}

// Keep the requested mapping from warp groups to experts, reducing only the
// per-group parallelism. This preserves correctness while trading throughput
// to fit the device's opt-in dynamic shared-memory limit.
inline combine_smem_config_t choose_combine_smem_config(
    int hidden,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t quantization_recipe,
    int num_warp_groups,
    int requested_warps_per_group,
    int max_dynamic_smem) {
    combine_smem_config_t result{0, 0, 0, 0, false};
    if (hidden <= 0 || num_warp_groups <= 0 || requested_warps_per_group < 2 || max_dynamic_smem <= 0) return result;

    for (int warps_per_group = requested_warps_per_group; warps_per_group >= 2; --warps_per_group) {
        const int num_warps = num_warp_groups * warps_per_group;
        if (num_warps > 32) continue;
        int num_recv_groups = 0;
        const int smem =
            ll_combine_dynamic_smem_bytes(hidden, token_dtype, quantization_recipe, num_warps, &num_recv_groups);
        if (num_recv_groups == 0 || smem > max_dynamic_smem) continue;
        return combine_smem_config_t{warps_per_group, num_warps, num_recv_groups, smem, true};
    }
    return result;
}

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
    const void* inScalesBuf;      // non-null for QUANT_FWD; runtime-typed storage
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
    void* rdmaBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvCntOff;
    int* rankCountersBase;
    int* rankDone;
    int nextRecvCntBufSize;
    int* recvStats;
    int64_t* waitStats;
    LowLatencyEpochState* epochState;
    size_t payloadSlotStride;
    size_t signalSlotStride;
    // CONFIG
    int numTokens;
    // Derived from validated inputs->scales->sizes[1].
    int scalesPerToken;
    int maxTokensPerRank;
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
    const float* inGlobalScales;
    const int* srcInfo;
    const int64_t* layoutRange;
    const void* inTopkIdx;        // cast to const TopkIdxT* by the JIT entry
    const float* topkWeights;
    int* rankMask;
    int* asyncErrorFlag;
    // OUTPUT
    void* outData;
    // INTERMEDIATE
    void* rdmaBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvFlagOff;
    int* atomicCleanFlag;
    int nextRecvCntBufSize;
    int64_t* waitStats;
    LowLatencyEpochState* epochState;
    size_t payloadSlotStride;
    size_t signalSlotStride;
    // CONFIG
    int numCombinedTokens;
    int hidden;
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
    const void* inScalesBuf = nullptr;     // non-null for QUANT_FWD; runtime-typed storage
    const void* inTopkIdx;                  // int32_t* or int64_t*; see topkIdxIsInt64
    bool topkIdxIsInt64 = true;             // selects the TopkIdxT kernel specialization
    // Derived from inputs->scales->sizes[1] by ncclEpDispatch.
    int scalesPerToken = 0;
    ncclDataType_t scaleDtype;              // QUANT_FWD tensor dtype; selects ScaleT in JIT
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
    void* rdmaBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvCntOff;
    int nextRecvCntBufSize;
    int* recvStats;
    int64_t* waitStats;

    LowLatencyEpochState* epochState = nullptr;
    size_t payloadSlotStride = 0;
    size_t signalSlotStride = 0;

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
    // token or QUANT_FWD scale window is written directly to its peer output.
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
    const float* inGlobalScales = nullptr;
    const int* srcInfo;
    const int64_t* layoutRange;
    const void* inTopkIdx;        // int32_t* or int64_t*; see topkIdxIsInt64
    bool topkIdxIsInt64 = true;   // selects the TopkIdxT kernel specialization
    const float* topkWeights;

    // User output
    void* outData;

    // Intermediate RDMA buffers + window-relative offsets
    void* rdmaBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvFlagOff;
    int nextRecvCntBufSize;
    int64_t* waitStats;

    LowLatencyEpochState* epochState = nullptr;
    size_t payloadSlotStride = 0;
    size_t signalSlotStride = 0;

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
    unsigned int deviceSm;
    int maxDynamicSmem;
    int* resolvedWarpsPerGroup = nullptr; // Host-only test/diagnostic output
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
    ncclEpCombQuant_t quantizationRecipe = NCCL_EP_COMB_QUANT_NONE;
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
    ncclEpDispQuant_t recipe,
    cudaStream_t stream = 0);

ncclResult_t call_combine(const CombineParams& params, cudaStream_t stream = 0);

void call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream = 0);

} // namespace ll
} // namespace nccl_ep
