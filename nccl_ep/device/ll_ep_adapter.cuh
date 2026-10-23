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
#include <cstdlib>
#include <cuda_runtime.h>
#include "nccl_device.h"
#include "ep_enums.h"
#include "common.hpp"
#include "ll/ll_ep_smem.cuh"

namespace nccl_ep {
namespace ll {

// Selects which LL dispatch kernel family an LSA-eligible call uses, driven by the
// NCCL_EP_DISPATCH_LSA env var (see nccl_ep_env.h): "simt" -> kSimt (dispatch_kernel_impl_2sided_rm_lsa),
// "tma" -> kTma (dispatch_kernel_impl_2sided_rm_lsa_tma, the default), "universal" -> kUniversal (forces
// the general dispatch_kernel_impl in ll_ep.cuh even when the call would otherwise be LSA-eligible).
// kUniversal always wins regardless of eligibility. Both STORE LSA families support NONE and
// fused DS_FP8E3M4. Eligible CFT calls use their separate LSA kernel in either kSimt or kTma
// mode; non-LSA-eligible calls and partial input warp tiles use the general kernel.
// See ll_dispatch_select_algo in device/jit/ll_dispatch_jit.cuh.
enum class NcclEpDispatchLsaMode { kSimt, kTma, kUniversal };

// Both the NCCL_EP_COMBINE_LSA request for one ncclEpCombine call (see nccl_ep_env.h) and the
// concrete LL combine kernel ll_combine_select_kernel (jit/ll_combine_jit.cuh) resolves it to --
// one enum, since every enforceable request already names a concrete kernel 1:1. Not every value
// is meaningful in both roles:
//   kNone:        valid only as a request -- "2sided"/"1sided"/"universal" unset; defer to
//                 eligibility (ll_combine_select_kernel's kNone case). Never a resolved kernel.
//   kDefault:     the general kernel. Valid as a request ("universal", always honorable) and as
//                 the resolved kernel (ineligible for any LSA-only kernel).
//   k2SidedRmLsa / k1SidedRmLsa: valid as a request ("2sided"/"1sided") and as the resolved
//                 kernel for that request.
//   kAutoRmLsa:   the on-device 1-sided-vs-2-sided auto-pick kernel. Only ever a resolved
//                 kernel (for an unenforced, eligible call) -- nothing can request it directly.
//   kInvalid:     an enforced request eligibility can't honor. Only ever a resolved kernel;
//                 callers must check for it and fail, never launch a kernel.
enum class LlCombineKernel { kNone, kDefault, k2SidedRmLsa, k1SidedRmLsa, kAutoRmLsa, kInvalid };

// Keep shared-quantization eligibility identical in the host geometry, JIT
// launch bounds, and device staging allocation.
// Tuning-derived cutoff: shared staging was validated through H=16384 on
// GB200. This is not a correctness limit; capacity is checked separately.
constexpr int kLlDsFp8SharedHiddenLimit = 16384;
constexpr int kLlDsFp8CompactWarps = 16;

// Conservative allowance for epoch scratch and static-allocation alignment.
// Staging is static shared memory, so do not use the opt-in budget above 48 KiB.
inline bool ll_dispatch_stage_quant(int hidden, int max_shared_bytes) {
    if (hidden <= 0 || hidden > kLlDsFp8SharedHiddenLimit) return false;
    const int required = hidden + (hidden / 128) * sizeof(float) + 32;
    return required <= std::min(max_shared_bytes, 48 * 1024);
}

// CFT currently stages BF16/FP16 payload elements.
constexpr int kCftElementBytes = sizeof(uint16_t);

// NCCL_EP_LL_CFT_LSA (default on): rank-major CFT dispatch runs the LSA-CFT
// kernel (ll/ll_dispatch_cft_lsa.cuh) and pairs with the LSA combine.
// Set it to 0 to force staged general CFT + general combine; CFT zero-copy requires LSA.
inline bool ll_cft_lsa_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("NCCL_EP_LL_CFT_LSA");
        return e == nullptr || *e == '\0' || *e != '0';
    }();
    return v;
}

// Uses the actual dispatch selector so public output binding and launch agree.
// General CFT zero-copy is unsupported; STORE zero-copy is unaffected.
bool ll_cft_zero_copy_supported(
    bool nvlinkOnly, ncclEpDispQuant_t recipe, ncclEpLayout_t layout,
    NcclEpDispatchLsaMode lsaMode, int hidden, ncclDataType_t tokenDtype);

// LSA-CFT sources with at most this many tokens use counted writes and publish counts early.
// Larger sends use ordinary writes and publish after completion. Receivers learn
// each source's form from bit 31 of its packed count word; 0 disables counted sends.
constexpr int kLlLsaCftCountedMaxTokensDefault = 128;
inline int ll_lsa_cft_counted_max_tokens() {
    static const int v = [] {
        const char* e = std::getenv("NCCL_EP_LL_CFT_COUNTED_MAX_TOKENS");
        return (e == nullptr || *e == '\0') ? kLlLsaCftCountedMaxTokensDefault : std::atoi(e);
    }();
    return v;
}

// Size receive warps from capacity: even zero-token ranks may receive peer traffic.
// Keep a sender/helper pair and the validated per-group warp ceiling.
inline int ll_lsa_cft_choose_warps_per_group(
    int currentWarpsPerGroup, int maxTokensPerRank, int numLocalExperts) {
    const int tokensPerGroup = static_cast<int>(
        nccl_ep::ceil_div<int64_t>(maxTokensPerRank, numLocalExperts));
    int requested = 2;
    while (requested < currentWarpsPerGroup && requested < tokensPerGroup) {
        requested *= 2;
    }
    return std::min(currentWarpsPerGroup, requested);
}

// LSA-CFT per-CTA dynamic shared memory for the precomputed destination list:
// list[tokensPerCta][numTopk] destination ranks plus nd[tokensPerCta] counts.
// Shared by the host sizing (ll_ep_adapter.cu) and the device carve.
__host__ __device__ constexpr int ll_lsa_cft_tokens_per_cta(int maxTokensPerRank, int numSms) {
    return nccl_ep::ceil_div(maxTokensPerRank, numSms);
}
__host__ __device__ constexpr int ll_lsa_cft_list_bytes(int maxTokensPerRank, int numSms, int numTopk) {
    return (ll_lsa_cft_tokens_per_cta(maxTokensPerRank, numSms) * numTopk +
            ll_lsa_cft_tokens_per_cta(maxTokensPerRank, numSms)) * static_cast<int>(sizeof(int));
}
// Space for one CTA's routing-header batches, used by zero-copy sends.
__host__ __device__ constexpr int ll_lsa_cft_hdr_batch_bytes(int maxTokensPerRank, int numSms, int numTopk, int hdrBytes) {
    return ll_lsa_cft_tokens_per_cta(maxTokensPerRank, numSms) * numTopk * hdrBytes;
}
constexpr int kLlLsaCftMaxTokensPerCta = 512;   // 8 x 64-bit token masks per destination
constexpr int kLlLsaCftCtaShift = 20;           // packed count word: flag << 31 | arrivals << 20 | tokens
constexpr int kLlLsaCftArrivalBits = 11;        // Remote: 1 arrival; self: 2 * numSms (< 2048).

// Two token-parity banks of 32 (rank, slot) pairs. Shared by host sizing and
// the device layout; allocated only when CFT is active.
constexpr int kCftDestScratchInts = 2 * 2 * 32;
constexpr int kCftDestScratchBytes = kCftDestScratchInts * static_cast<int>(sizeof(int));
static_assert(kCftDestScratchBytes == 512, "CFT dest scratch is 512 B by layout contract");

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
//
// max_dynamic_smem is taken as the exact usable dynamic-smem budget --
// callers that also need to reserve room for a kernel's static __shared__
// usage must subtract it before calling this.
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
// The host adapters pack these structs for the JIT entry points. Dispatch
// implementations take the whole struct; combine and clean entries unpack it.
// Keep their bindings in sync when adding fields.
// ============================================================================

// CFT dispatch parameters, zero-initialized unless collective setup succeeded.
// CFT carries payloads; token counts still use recvCntBuf.
struct cft_dispatch_args_t {
    const uint32_t* peerLeIds;         // device [numRanks]: peer endpoint IDs; 0 is valid, [currRank] is unused
    void*           recvBase;          // this rank's FABRIC recv buffer VA, parity-resolved (receiver reads here)
    uint64_t        bytesCntrBaseOff;  // byte offset to the byte-counter region within each peer's fabric buf
    // Remote puts add this parity offset; host-provided receive pointers
    // already include it. Both use the group-scoped dispatch counter.
    uint64_t        parityOff;
    size_t          srcRankRegionBytes; // bytes per srcRank slot group in the data region
                                        // (tokensPerRank * worst-case slot stride)
    uint64_t*       recvByteCounters;   // this rank's byte counter array [numRanks * (256/8)], parity-resolved
    uint64_t*       cumulByteExpect;    // device [numRanks*numLocalExperts]: cumulative expected bytes, parity-resolved
    uint64_t        outputOffset;       // peer endpoint offset of zero-copy output row zero
    int             countedPayload;      // LSA-CFT: 1 = this rank's payload puts are counted in this dispatch
};

struct dispatch_kernel_args_t {
    // INPUT
    const void* inData;
    const void* inScalesBuf;      // non-null for QUANT_FWD; runtime-typed storage
    const void* inTopkIdx;        // cast to const TopkIdxT* by dispatch_kernel_impl
    const float* inTopkWeights;
    int* rankMask;
    int* asyncErrorFlag;
    // OUTPUT
    void* outDataBuf;
    void* outScalesBuf;
    int* outSrcInfo;
    // Sender-side (dstRank, slotIdx) info for combine_kernel_impl_1sided_rm_lsa, written
    // ADDITIONALLY to outSrcInfo above (see kRecordDstInfo in ll/ll_dispatch_lsa.cuh).
    int* outDstInfo;
    // Handle-owned want-1-sided output for NCCL_EP_COMBINE_LSA auto mode -- dispatch itself compares
    // every active peer's published count against combineLsaThreshold below and publishes the
    // combined 0/1 answer (see computeAndPublishWant1SidedLsa in ll/ll_dispatch_lsa.cuh); same gate
    // as outDstInfo above.
    int* outWant1Sided;
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
    // Per-peer "total tokens I'm dispatching this round" publish region (see
    // LowLatencyBuffer::dispatch_rdma_total_token_buffer_offset's doc comment): buffer-relative
    // offset, banked by epoch parity using signalSlotStride below (shared by every signaling kind
    // -- see its doc comment in common.hpp). Only read/written under the same kRecordDstInfo gate
    // as outDstInfo.
    size_t totalTokenOff;
    int* rankSentCnt;
    int* rankArrivedCnt;
    int* rankDone;
    // Ints to clear covering this bank's signal + read-done span (never total-token); see
    // LowLatencyBuffer::lsa_signaling_clean_int_count's doc comment. Shared by every dispatch
    // kernel's next-bank wipe, LSA and general alike -- the general kernel used to wipe only its
    // own narrower signal-only span (a separate, now-removed nextRecvCntBufSize field), which left
    // read-done stale for a bank a later LSA round reused.
    int lsaSignalingCleanCount;
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
    // AFD attention-rank count. Non-zero calls always use the universal LL kernel.
    int numAttnRanks;
    int numWarpGroups;
    int numWarpsPerGroup;
    bool roundScale;
    // NCCL_EP_COMBINE_LSA_1SIDED_THRESH; only read when outWant1Sided above is non-null. See
    // computeAndPublishWant1SidedLsa in ll/ll_dispatch_lsa.cuh.
    int combineLsaThreshold;
    // recv_topk_idx numbering (LOCAL/GLOBAL); resolved on the host (never AUTO).
    ncclEpExpertIdKind_t recvTopkIdxKind;
    int phases;
    ncclDevComm* devComm;
    const ncclWindow_t* windows;
    unsigned signalsBase;
    uint64_t timeoutCycles;
    // Non-null windows enable peer payload writes.
    ncclWindow_t recvDataWindow;
    size_t recvDataOffset;
    ncclWindow_t rcvScalesWin;
    size_t rcvScalesOffs;
    // LL CFT (counted fabric transport) parameters; see cft_dispatch_args_t.
    cft_dispatch_args_t cft;
};

struct combine_kernel_args_t {
    // INPUT
    const void* inData;
    const float* inGlobalScales;
    const int* srcInfo;
    // Sender-side (dstRank, slotIdx) info read by combine_kernel_impl_1sided_rm_lsa; non-null
    // only when that kernel is selected (see ncclEpCombine's combine_1sided gating).
    const int* dstInfo;
    // inData's own window registration + base offset, resolved by ncclEpCombine for the
    // 1-sided pull path (combine_kernel_impl_1sided_rm_lsa reads other ranks' inData directly
    // via ncclGetPeerPointer(inDataWindow, ...)). ncclWindow_t{} when unused (2-sided path).
    ncclWindow_t inDataWindow;
    size_t inDataWindowOffset;
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
    // combine_kernel_impl_1sided_rm_lsa's read-done signaling region (see
    // LowLatencyBuffer::combine_rdma_read_done_buffer_offset's doc comment); banked using
    // signalSlotStride below, shared by every signaling kind (see its doc comment in common.hpp).
    // Unused by the 2-sided kernel.
    size_t readDoneOffBase;
    // NCCL_EP_COMBINE_SYNC_COUNT-element array of grid-wide election counters; see ctaSync's doc
    // comment in ll_combine_lsa.cuh.
    int* combineSync;
    // NCCL_EP_COMBINE_LSA auto mode's want-1-sided input: dispatch has already compared every
    // active peer's count against combineLsaThreshold and published the combined 0/1 answer (see
    // dispatch_kernel_args_t::outWant1Sided / computeAndPublishWant1SidedLsa). A plain read -- no
    // reset needed, dispatch overwrites it fresh each round.
    int* want1SidedBuf;
    // Ints to clear covering this bank's signal + read-done span (never total-token, which combine
    // never touches anyway); see LowLatencyBuffer::lsa_signaling_clean_int_count's doc comment.
    // Shared by every combine kernel's next-bank wipe, LSA and general alike -- see
    // dispatch_kernel_args_t::lsaSignalingCleanCount's doc comment for why the general kernel can't
    // use a narrower, signal-only count of its own.
    int lsaSignalingCleanCount;
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
    // AFD attention-rank count. Non-zero calls always use the universal LL kernel.
    int numAttnRanks;
    int numWarpGroups;
    int numWarpsPerGroup;
    int phases;
    bool zeroCopy;
    ncclDevComm* devComm;
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
    ncclDevComm* devComm;
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
    int maxDynamicSmem = 0; // Cached device capacity; also honors the test override.
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
    int* outDstInfo = nullptr; // non-null only when 1-sided combine could be selected
    int* outWant1Sided = nullptr; // non-null only when 1-sided combine could be selected
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
    size_t totalTokenOff = 0;      // see dispatch_kernel_args_t::totalTokenOff
    int lsaSignalingCleanCount = 0; // see dispatch_kernel_args_t::lsaSignalingCleanCount
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
    int numAttnRanks = 0;
    ncclEpLayout_t layout;

    // GIN / NCCL device context
    ncclDevComm* devComm;
    const ncclWindow_t* windows;
    unsigned signalsBase;

    // Runtime workspace + error tracking. Each pointer is a dedicated,
    // non-overlapping region computed once at group-creation time
    int* rankSentCnt;
    int* rankArrivedCnt;
    int* rankDone;
    int numDeviceSms;
    unsigned int deviceSm = 0;       // Host-only mask-specialization eligibility.
    unsigned int deviceSmCount = 0;  // Physical SM count, distinct from the requested budget.
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

    // Selects which LSA-eligible LL dispatch kernel family gets JIT-compiled (NCCL_EP_DISPATCH_LSA
    // env var; see NcclEpDispatchLsaMode's doc comment above). A build-time/JIT-variant selector,
    // not data threaded into dispatch_kernel_args_t. When kTma is (effectively) selected, the TMA
    // variant's stage ring (see ll/ll_ep_smem.cuh's dispatch_tma_smem namespace) is sized from
    // maxDynamicSmem above.
    NcclEpDispatchLsaMode lsaMode = NcclEpDispatchLsaMode::kTma;
    // JIT specialization only; does not change launch geometry or shared memory.
    int tmaQuantWarps = dispatch_tma_smem::kMinQuantWarps;

    // True whenever 1-sided combine could be selected for this call
    bool recordDstInfo = false;
    // NCCL_EP_COMBINE_LSA_1SIDED_THRESH; only read when recordDstInfo above is true. See
    // dispatch_kernel_args_t::combineLsaThreshold.
    int combineLsaThreshold = 0;
    // INPUT/OUTPUT: the caller seeds *dstInfoDisqualifiedOut with its own already-known
    // disqualifying reasons (true if it never wants 1-sided enabled this call, regardless of
    // dispatch's kernel selection), and call_dispatch ORs in whether THIS call's actual kernel
    // selection ALSO disqualifies 1-sided combine (see launch_ll_dispatch's dstInfoDisqualifiedOut
    // doc comment) -- the single source of truth ncclEpHandle::ll.dispatch_1sided_disqualified
    // must be set from, instead of re-deriving it from lsaMode alone.
    bool* dstInfoDisqualifiedOut = nullptr;

    // Populated after CFT setup succeeds on every rank.
    const uint32_t* cftPeerLeIds = nullptr;
    void*           cftRecvBase = nullptr;
    uint64_t        cftBytesCntrBaseOff = 0;
    uint64_t        cftParityOff = 0;
    size_t          cftSrcRankRegionBytes = 0;
    uint64_t*       cftRecvByteCounters = nullptr;
    uint64_t*       cftCumulByteExpect = nullptr;
    bool            cftZeroCopy = false; // selects the JIT specialization after collective output binding
    uint64_t        cftOutputOffset = 0;
};

struct CombineParams {
    // User inputs
    const void* inData;
    const float* inGlobalScales = nullptr;
    const int* srcInfo;
    const int* dstInfo = nullptr;      // 1-sided only; see combine_kernel_args_t
    ncclWindow_t inDataWindow = ncclWindow_t{};
    size_t inDataWindowOffset = 0;
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
    size_t readDoneOffBase = 0;    // 1-sided only; see combine_kernel_args_t
    int lsaSignalingCleanCount = 0; // see combine_kernel_args_t::lsaSignalingCleanCount
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
    int numAttnRanks = 0;
    ncclEpLayout_t layout;

    // GIN / NCCL device context
    ncclDevComm* devComm;
    const ncclWindow_t* windows;
    unsigned signalsBase;

    // Runtime workspace + error tracking. Each pointer is a dedicated,
    // non-overlapping region computed once at group-creation time.
    int* combineSync; // NCCL_EP_COMBINE_SYNC_COUNT ints; see combine_kernel_args_t
    int* want1SidedBuf = nullptr; // auto mode only; see combine_kernel_args_t
    int numDeviceSms;
    unsigned int deviceSm;
    int maxDynamicSmem;
    int* resolvedWarpsPerGroup = nullptr; // Host-only test/diagnostic output
    int* rankMask = nullptr;
    int* asyncErrorFlag = nullptr;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;

    // CFT dispatch requires combine to clear each consumed finish flag.
    // The general combine kernel provides that protocol.
    bool cftEnabled = false;
    // Keep forced-general CFT dispatch paired with its consumer-clearing combine.
    NcclEpDispatchLsaMode lsaMode = NcclEpDispatchLsaMode::kTma;

    // Per-call behavior toggles that select the JIT kernel specialization.
    bool useLogFmt = false;
    bool zeroCopy = false;
    int phases = 0;

    // Token wire dtype (unquantized payload width). Combine decodes + reduces,
    // so BF16/FP16/FP32 are three distinct kernel specializations (FP32 also
    // halves kMaxNumGroups to stay within the dynamic-SMEM cap).
    ncclDataType_t tokenDtype = ncclBfloat16;
    ncclEpCombQuant_t quantizationRecipe = NCCL_EP_COMB_QUANT_NONE;

    // True when every destination is NVLink-reachable (the whole world fits in
    // one LSA team). One of the inputs call_combine passes to ll_combine_lsa_eligible;
    // ignored by the general (kDefault) JIT path.
    bool nvlinkOnly = false;

    // NCCL_EP_COMBINE_LSA request for this call, as a kernel (see LlCombineKernel's doc comment
    // above) -- kNone unless "2sided"/"1sided"/"universal" was set.
    LlCombineKernel enforced = LlCombineKernel::kNone;

    // The two extra inputs call_combine needs, beyond the fields above, to compute 1-sided
    // eligibility itself (see ll_combine_lsa_eligible/ll_combine_select_kernel's doc comments).
    bool groupZeroCopyOn = false;
    bool dispatch1SidedDisqualified = false;
};

struct CleanLowLatencyBufferParams {
    unsigned int deviceSm;
    int* clean_0;
    int num_clean_int_0;
    int* clean_1;
    int num_clean_int_1;
    int* rankMask;
    int* syncBuffer;
    ncclWindow_t* syncWindow;
    ncclDevComm* devComm;
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
    cudaStream_t stream = 0,
    bool nucs = false);

ncclResult_t call_combine(const CombineParams& params, cudaStream_t stream = 0, bool nucs = false);

ncclResult_t call_clean_low_latency_buffer(
    const CleanLowLatencyBufferParams& params,
    cudaStream_t stream = 0,
    bool nucs = false);

} // namespace ll
} // namespace nccl_ep
