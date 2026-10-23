/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ll_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_source_literals.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace ll {
namespace jit {

constexpr const char* kLlCombineJitEntryName = "nccl_ep_jit_ll_combine_kernel";

// The inner-loop unroll factor is compile-time fixed; top-k is passed to the
// source generator so each handle receives an exact specialization.
constexpr int kLlCombineMaxUnrolls = 4;

// Which device combine kernel implementation gets JIT-compiled -- see LlCombineKernel's doc
// comment in device/ll_ep_adapter.cuh for what each value means and which role(s) it's valid in.
// Chosen automatically per call by ll_combine_select_kernel() below -- callers never pick one
// directly.
//   kDefault:     the general kernel (every recipe/layout, RDMA/GIN capable). Selected whenever
//                 the call isn't LSA-only, its recipe/layout/LogFMT combination isn't eligible
//                 for the LSA-only kernel below, or "universal" was enforced regardless of
//                 eligibility.
//   k2SidedRmLsa: LSA-only rank-major kernel that stages payload through
//                 RDMA buffers on both send and receive. Selected for every
//                 LSA-only, NONE-recipe, rank-major call paired with STORE or CFT LSA
//                 dispatch and a hidden size divisible by the receive tile.
//   k1SidedRmLsa: LSA-only rank-major kernel that pulls each token's result directly out of the
//                 owning rank's inData window instead of sending it back. Requires everything
//                 k2SidedRmLsa requires plus STORE transport, BF16 tokens and enforced "1sided".
//   kAutoRmLsa:   LSA-only rank-major wrapper that makes the 1-sided-vs-2-sided pick on-device, at
//                 kernel entry, from dispatch's already-decided want1SidedBuf (dispatch itself
//                 compares every active peer's count against combineLsaThreshold) -- no host
//                 round-trip needed. Requires everything k1SidedRmLsa requires except the enforced request;
//                 selected only when nothing was enforced.

// The LSA-only eligibility check shared by both k2SidedRmLsa and (refined further by the caller,
// see ll_combine_select_kernel below) k1SidedRmLsa/kAutoRmLsa. Pulled out of selection so
// ncclEpCombine can evaluate it once, host-side, and pass the result down instead of selection
// re-deriving it from raw params -- that duplication is exactly what let the two diverge before.
inline bool ll_combine_lsa_eligible(
    bool nvlinkOnly, ncclEpCombQuant_t qrecipe, bool useLogFmt, ncclEpLayout_t layout,
    bool cftEnabled, int hidden, ncclDataType_t tokenDtype,
    NcclEpDispatchLsaMode lsaMode, int numAttnRanks = 0) {
    // General CFT does not clear the next bank and must use general combine.
    const bool dispatchWipesNextBank = !cftEnabled ||
        (ll_cft_lsa_enabled() && lsaMode != NcclEpDispatchLsaMode::kUniversal);
    const int elemBytes = tokenDtype == ncclFloat32 ? 4 : 2;
    const int elemsPerTile = combine_smem::kWarpSize * combine_smem::kNumRecvUnrolls *
        combine_smem::elements_per_int4(elemBytes);
    return numAttnRanks == 0 && hidden > 0 && hidden % elemsPerTile == 0 && dispatchWipesNextBank && nvlinkOnly &&
        qrecipe == NCCL_EP_COMB_QUANT_NONE && !useLogFmt && layout == NCCL_EP_LAYOUT_RANK_MAJOR;
}

// Picks the kernel for the given request, trusting the caller's own eligibility (see
// ll_combine_lsa_eligible above) rather than re-deriving it -- the sole place `enforced_lsa` is
// turned into a concrete resolved kernel. oneSidedEligible must already imply lsaEligible (see
// call_combine in device/ll_ep_adapter.cu, the only caller).
//
// enforced_lsa only governs the pick WITHIN the LSA domain (1-sided vs. 2-sided vs. auto) --
// NCCL_EP_COMBINE_LSA says nothing about whether this call is even in that domain. So when the call
// isn't LSA-eligible at all (wrong topology/layout/recipe or general CFT), an enforced LSA kernel falls back to
// kDefault silently, same as an unenforced call would. kInvalid is reserved for a request that IS
// in the LSA domain but still can't be honored (e.g. 1-sided enforced, lsaEligible but not
// oneSidedEligible) -- that's an actionable misconfiguration, not an out-of-domain no-op.
inline LlCombineKernel ll_combine_select_kernel(
    LlCombineKernel enforced_lsa, bool lsaEligible, bool oneSidedEligible) {
    switch (enforced_lsa) {
        case LlCombineKernel::kDefault:
            return LlCombineKernel::kDefault;
        case LlCombineKernel::k1SidedRmLsa:
            if (!lsaEligible) return LlCombineKernel::kDefault;
            return oneSidedEligible ? LlCombineKernel::k1SidedRmLsa : LlCombineKernel::kInvalid;
        case LlCombineKernel::k2SidedRmLsa:
            return lsaEligible ? LlCombineKernel::k2SidedRmLsa : LlCombineKernel::kDefault;
        case LlCombineKernel::kNone:
            if (oneSidedEligible) return LlCombineKernel::kAutoRmLsa;
            return lsaEligible ? LlCombineKernel::k2SidedRmLsa : LlCombineKernel::kDefault;
        case LlCombineKernel::kAutoRmLsa:
        case LlCombineKernel::kInvalid:
            break; // Never valid requests; fall through to the assert below.
    }
    EP_HOST_ASSERT(false && "ll_combine_select_kernel: enforced_lsa must be a request-valid value");
    return LlCombineKernel::kInvalid;
}

// For diagnostics: an LlCombineKernel value printed in an error message should read as a name,
// not an opaque int.
inline const char* ll_combine_kernel_name(LlCombineKernel kernel) {
    switch (kernel) {
        case LlCombineKernel::kNone: return "none";
        case LlCombineKernel::kDefault: return "kDefault";
        case LlCombineKernel::k2SidedRmLsa: return "k2SidedRmLsa";
        case LlCombineKernel::k1SidedRmLsa: return "k1SidedRmLsa";
        case LlCombineKernel::kAutoRmLsa: return "kAutoRmLsa";
        case LlCombineKernel::kInvalid: return "kInvalid";
    }
    return "<unknown LlCombineKernel>";
}

inline std::string ll_combine_jit_source(
    bool useLogFmt,
    const char* recipe_literal,
    int hidden,
    int num_topk,
    ncclEpLayout_t layout,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    LlCombineKernel kernel) {
    // Callers must pass a resolved, concrete kernel -- never the sentinels ll_combine_select_kernel
    // returns for "not yet resolved" (kNone) or "enforcement unreachable" (kInvalid); see its doc
    // comment.
    EP_HOST_ASSERT(kernel != LlCombineKernel::kInvalid && kernel != LlCombineKernel::kNone);
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    const char* topk_type = topkIdxIsInt64 ? "int64_t" : "int32_t";
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(tokenDtype);
    std::ostringstream src;
    src << "#include \"device/ll_ep.cuh\"\n"
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(1024, 1)\n"
        << "__global__ void " << kLlCombineJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::ll::combine_kernel_args_t p) {\n";
    switch (kernel) {
        case LlCombineKernel::kAutoRmLsa:
            // A real compiled function (ll/ll_combine_lsa.cuh), not string-generated logic -- see
            // its own doc comment for the on-device pick.
            src << "  nccl_ep::ll::combine_kernel_impl_auto_rm_lsa<\n"
                << "      " << hidden << ",\n"
                << "      " << num_topk << ",\n"
                << "      " << kLlCombineMaxUnrolls << ",\n"
                << "      " << topk_type << ",\n"
                << "      " << token_dtype_literal << ">(p);\n";
            break;
        case LlCombineKernel::k1SidedRmLsa:
            src << "  nccl_ep::ll::combine_kernel_impl_1sided_rm_lsa<\n"
                << "      " << hidden << ",\n"
                << "      " << num_topk << ",\n"
                << "      " << kLlCombineMaxUnrolls << ",\n"
                << "      " << token_dtype_literal << ">(\n"
                << "      p.inData, p.inDataWindow, p.inDataWindowOffset, p.dstInfo,\n"
                << "      p.rankMask, p.asyncErrorFlag,\n"
                << "      p.outData,\n"
                << "      p.rdmaBuf,\n"
                << "      p.recvFlagOff, p.combineSync, p.lsaSignalingCleanCount,\n"
                << "      p.waitStats, p.epochState, p.payloadSlotStride, p.signalSlotStride,\n"
                << "      p.readDoneOffBase,\n"
                << "      p.numCombinedTokens, p.maxTokensPerRank,\n"
                << "      p.numExperts, p.currRank, p.numRanks,\n"
                << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
                << "      p.phases,\n"
                << "      p.devComm, p.windows, p.timeoutCycles);\n";
            break;
        case LlCombineKernel::k2SidedRmLsa:
            src << "  nccl_ep::ll::combine_kernel_impl_2sided_rm_lsa<\n"
                << "      " << hidden << ",\n"
                << "      " << num_topk << ",\n"
                << "      " << kLlCombineMaxUnrolls << ",\n"
                << "      " << topk_type << ",\n"
                << "      " << token_dtype_literal << ">(\n"
                << "      p.inData, p.srcInfo,\n"
                << "      static_cast<const " << topk_type << "*>(p.inTopkIdx),\n"
                << "      p.rankMask, p.asyncErrorFlag,\n"
                << "      p.outData,\n"
                << "      p.rdmaBuf,\n"
                << "      p.sendOff, p.recvOff, p.recvFlagOff,\n"
                << "      p.combineSync, p.lsaSignalingCleanCount,\n"
                << "      p.waitStats, p.epochState, p.payloadSlotStride, p.signalSlotStride,\n"
                << "      p.numCombinedTokens, p.maxTokensPerRank,\n"
                << "      p.numExperts, p.currRank, p.numRanks,\n"
                << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
                << "      p.phases, p.zeroCopy,\n"
                << "      p.devComm, p.windows, p.timeoutCycles);\n";
            break;
        case LlCombineKernel::kDefault:
            src << "  nccl_ep::ll::combine_kernel_impl<\n"
                << "      " << ::nccl_ep::jit::bool_literal(useLogFmt) << ",\n"
                << "      " << hidden << ",\n"
                << "      " << num_topk << ",\n"
                << "      " << kLlCombineMaxUnrolls << ",\n"
                << "      " << layout_literal << ",\n"
                << "      " << topk_type << ",\n"
                << "      " << token_dtype_literal << ", " << recipe_literal << ">(\n"
                << "      p.inData, p.inGlobalScales, p.srcInfo, p.layoutRange,\n"
                << "      static_cast<const " << topk_type << "*>(p.inTopkIdx), p.topkWeights,\n"
                << "      p.rankMask, p.asyncErrorFlag,\n"
                << "      p.outData,\n"
                << "      p.rdmaBuf,\n"
                << "      p.sendOff, p.recvOff, p.recvFlagOff,\n"
                << "      p.combineSync, p.lsaSignalingCleanCount,\n"
                << "      p.waitStats, p.epochState, p.payloadSlotStride, p.signalSlotStride,\n"
                << "      p.numCombinedTokens, p.hidden, p.maxTokensPerRank,\n"
                << "      p.numExperts, p.currRank, p.numRanks,\n"
                << "      p.numAttnRanks,\n"
                << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
                << "      p.phases, p.zeroCopy,\n"
                << "      p.devComm, p.windows, p.signalsBase, p.timeoutCycles);\n";
            break;
        case LlCombineKernel::kNone:
        case LlCombineKernel::kInvalid:
            // Caught by the EP_HOST_ASSERT above; listed only so the switch covers every value.
            EP_HOST_ASSERT(false && "ll_combine_jit_source: kernel must be resolved and valid");
            break;
    }
    src << "}\n";
    return src.str();
}

inline ncclResult_t launch_ll_combine(
    bool useLogFmt,
    ncclEpCombQuant_t qrecipe,
    unsigned int device_sm,
    int hidden,
    ncclEpLayout_t layout,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    int num_topk,
    int numSms,
    int numWarps,
    int dynamic_smem_bytes,
    LlCombineKernel kernel,
    const combine_kernel_args_t& args,
    cudaStream_t stream,
    bool nucs) {
    // kernel is resolved once by the caller (ll_combine_select_kernel, via call_combine) --
    // never re-derived here, so there's exactly one place eligibility is evaluated.
    const bool twoSidedRmLsa = kernel == LlCombineKernel::k2SidedRmLsa;
    const bool oneSidedRmLsa = kernel == LlCombineKernel::k1SidedRmLsa;
    const bool autoRmLsa = kernel == LlCombineKernel::kAutoRmLsa;
    const bool anyLsa = twoSidedRmLsa || oneSidedRmLsa || autoRmLsa;

    const char* recipe_literal = anyLsa ? "" : ::nccl_ep::jit::combine_recipe_literal(qrecipe);
    if (!anyLsa && recipe_literal == nullptr) {
        std::fprintf(stderr, "ncclEpCombine: unsupported LL combine recipe %d\n", qrecipe);
        return ncclInvalidArgument;
    }
    static const int variant_identity_default = 0;
    static const int variant_identity_2sided_rm_lsa = 0;
    static const int variant_identity_1sided_rm_lsa = 0;
    static const int variant_identity_auto_rm_lsa = 0;

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.device_sm = static_cast<int>(device_sm);
    variant.propagate_cache_errors = true;
    variant.kernel_family = oneSidedRmLsa ? "ll_combine_1sided_rm_lsa" :
        (twoSidedRmLsa ? "ll_combine_2sided_rm_lsa" : (autoRmLsa ? "ll_combine_auto_rm_lsa" : "ll_combine"));
    variant.entry_name = kLlCombineJitEntryName;
    variant.identity = oneSidedRmLsa ? &variant_identity_1sided_rm_lsa :
        (twoSidedRmLsa ? &variant_identity_2sided_rm_lsa :
         (autoRmLsa ? &variant_identity_auto_rm_lsa : &variant_identity_default));
    // Derived from the raw parameters so the warm-cache launch path never has
    // to build the variant-name string.
    std::uint64_t key = ::nccl_ep::jit::kRuntimeKeySeed;
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(hidden));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(layout));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(num_topk));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(qrecipe));
    key = ::nccl_ep::jit::runtime_key_mix(
        key, (useLogFmt ? 1u : 0u) | (topkIdxIsInt64 ? 2u : 0u) | (oneSidedRmLsa ? 4u : 0u) |
            (autoRmLsa ? 8u : 0u));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(tokenDtype));
    variant.runtime_key = key;
    variant.num_blocks = numSms;
    variant.block_dim = numWarps * 32;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;
    variant.nucs = nucs;
    if (qrecipe == NCCL_EP_COMB_QUANT_NVFP4) {
        // Recipe validation has already checked toolkit and device support.
        variant.min_sm = 100;
        variant.target_arch = host_device_fp4_target_arch(device_sm);
    }
    // Always cooperative: kDefault/2sided_rm_lsa need it for their grid-wide
    // SEND/RECV sync.
    variant.cooperative = true;
    // Pair SMs into clusters of 2 when possible to share distributed SMEM.
    variant.cluster_dim_x = (numSms % 2 == 0) ? 2 : 1;

    std::string error;
    // Warm-cache fast path: launches without materializing variant_name/source.
    ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel_cached(
        variant, const_cast<combine_kernel_args_t*>(&args), 0, stream, &error);

    std::string variant_name;
    if (status == ::nccl_ep::jit::JitKernelStatus::kDisabled) {
        // Cache miss: build the name + source and take the compile/load path.
        std::ostringstream name;
        name << "ll_combine"
             << "_hdim" << hidden << ::nccl_ep::jit::layout_name_tag(layout)
             << "_topk" << num_topk
             << (useLogFmt ? "_logfmt" : "")
             << (twoSidedRmLsa ? "_2sidedrmlsa" : "")
             << (oneSidedRmLsa ? "_1sidedrmlsa" : "")
             << (autoRmLsa ? "_autormlsa" : "")
             << (topkIdxIsInt64 ? "_topk64" : "_topk32")
             << ::nccl_ep::jit::token_dtype_name_tag(tokenDtype)
             << (anyLsa ? "" : ::nccl_ep::jit::combine_recipe_name_tag(qrecipe));
        variant_name = name.str();
        const std::string source = ll_combine_jit_source(
            useLogFmt, recipe_literal, hidden, num_topk, layout, topkIdxIsInt64, tokenDtype, kernel);
        variant.variant_name = variant_name;
        variant.source = source;
        status = ::nccl_ep::jit::launch_jit_kernel(
            variant, const_cast<combine_kernel_args_t*>(&args), stream, &error);
    }

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] LL combine JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

} // namespace jit
} // namespace ll
} // namespace nccl_ep
