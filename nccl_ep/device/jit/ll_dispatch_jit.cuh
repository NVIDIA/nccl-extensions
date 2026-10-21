/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/device_primitives.cuh"
#include "device/ll/ll_ep_smem.cuh"
#include "device/ll_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_source_literals.hpp"
#include "quantization_recipe.hpp"

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

constexpr const char* kLlDispatchJitEntryName = "nccl_ep_jit_ll_dispatch_kernel";

// Selects which device dispatch kernel implementation gets JIT-compiled,
// chosen automatically per call by ll_dispatch_select_algo() below -- callers
// never pick a kernel directly.
//   kDefault:     the general kernel (every recipe/layout, RDMA/GIN capable).
//                 Also the CFT fallback: quantized recipes, partial tiles, expert-major,
//                 non-NVLink groups, and NCCL_EP_LL_CFT_LSA=0.
//   k2SidedRmLsa: STORE LSA-only rank-major SIMT kernel, quantization: NONE or
//                 fused DS_FP8E3M4. Token and scale windows independently bypass
//                 receive staging, selected at runtime.
//   kCftRmLsa:    CFT LSA-only rank-major kernel, NONE recipe only. Warp 0 issues
//                 fabric puts. Counted sources publish counts early; ordinary-put
//                 sources publish after completion. Sender completion mode and
//                 zero-copy output are JIT specializations. NCCL_EP_LL_CFT_LSA=0
//                 selects the general CFT kernel instead.
//   k2SidedRmLsaTma: TMA-pipelined SEND, NONE or fused DS_FP8E3M4, with the same
//                 direct-output contract, when CFT is disabled. DSFP8 quantizes
//                 once per token into the existing BF16 stage in place before
//                 issuing peer TMA stores.
enum class LlDispatchAlgo { kDefault, k2SidedRmLsa, k2SidedRmLsaTma, kCftRmLsa };

// Picks the most performant kernel for the configuration; overridable via NCCL_EP_DISPATCH_LSA
// (lsaMode, see NcclEpDispatchLsaMode's doc comment in device/ll_ep_adapter.cuh).
inline LlDispatchAlgo ll_dispatch_select_algo(
    bool nvlinkOnly, ncclEpDispQuant_t recipe, ncclEpLayout_t layout, bool cftEnabled,
    NcclEpDispatchLsaMode lsaMode, int hidden, ncclDataType_t tokenDtype,
    int numAttnRanks = 0) {
    // Explicit override: NCCL_EP_DISPATCH_LSA=universal always forces the general kernel,
    // regardless of eligibility.
    if (lsaMode == NcclEpDispatchLsaMode::kUniversal) {
        return LlDispatchAlgo::kDefault;
    }

    const bool lsaRecipe =
        recipe == NCCL_EP_DISP_QUANT_NONE || recipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4;
    // All LSA paths require complete warp-sized input vector tiles.
    // Partial tiles use the universal kernel, including explicit SIMT/TMA modes.
    const int elemsPerWarp = 32 * static_cast<int>(sizeof(int4)) / (tokenDtype == ncclFloat32 ? 4 : 2);
    // CFT: the LSA-CFT specialization (NONE recipe) unless NCCL_EP_LL_CFT_LSA=0; partial tiles
    // and other recipes keep the general CFT kernel.
    const bool cftUsesLsaKernel = !cftEnabled || (ll_cft_lsa_enabled() && recipe == NCCL_EP_DISP_QUANT_NONE);
    const bool lsaEligible = numAttnRanks == 0 && hidden > 0 && hidden % elemsPerWarp == 0 &&
        cftUsesLsaKernel && nvlinkOnly && lsaRecipe && layout == NCCL_EP_LAYOUT_RANK_MAJOR;

    // CFT owns a separate LSA kernel; both SIMT/TMA overrides retain that
    // selection. STORE keeps its existing SIMT/TMA choice.
    if (lsaEligible && cftEnabled) {
        return LlDispatchAlgo::kCftRmLsa;
    }
    // Prefer TMA for both supported STORE LSA recipes.
    if (lsaEligible && !cftEnabled && lsaMode == NcclEpDispatchLsaMode::kTma) {
        return LlDispatchAlgo::k2SidedRmLsaTma;
    }
    // Explicit SIMT selection uses the STORE SIMT kernel.
    if (lsaEligible) {
        return LlDispatchAlgo::k2SidedRmLsa;
    }

    return LlDispatchAlgo::kDefault;
}

// Zero-copy CFT is implemented only by the LSA kernel. Keep public binding
// validation tied to the same selector used by cold and warm JIT launches.
inline bool ll_dispatch_cft_zero_copy_supported(
    bool nvlinkOnly, ncclEpDispQuant_t recipe, ncclEpLayout_t layout,
    NcclEpDispatchLsaMode lsaMode, int hidden, ncclDataType_t tokenDtype) {
    return (tokenDtype == ncclBfloat16 || tokenDtype == ncclFloat16) &&
        ll_dispatch_select_algo(nvlinkOnly, recipe, layout, /*cftEnabled=*/true,
                                lsaMode, hidden, tokenDtype) == LlDispatchAlgo::kCftRmLsa;
}

inline std::string ll_dispatch_jit_source(
    const DispatchKernelSpec& kernel_spec,
    int hidden,
    int num_topk,
    ncclEpLayout_t layout,
    bool nvlinkOnly,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    LlDispatchAlgo algo,
    bool compactQuant,
    bool stageQuant,
    bool cftEnabled,
    bool rankOwnsExperts,
    bool rankOwnsTokens,
    int numTmaStages,
    int numTmaQuantWarps,
    bool recordDstInfo,
    bool cftZeroCopy,
    bool cftCountedPayload,
    int cftNumWarps,
    bool cftOneWordMask) {
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    const char* topk_type = topkIdxIsInt64 ? "int64_t" : "int32_t";
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(tokenDtype);

    int blockThreads = 1024;
    if (algo == LlDispatchAlgo::kCftRmLsa) {
        blockThreads = cftNumWarps * 32;
    } else if (compactQuant) {
        blockThreads = kLlDsFp8CompactWarps * 32;
    }

    // CFT LSA owns its header; other families use the umbrella include.
    std::ostringstream src;
    src << (algo == LlDispatchAlgo::kCftRmLsa
                ? "#include \"device/ll/ll_dispatch_cft_lsa.cuh\"\n"
                : "#include \"device/ll_ep.cuh\"\n")
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__("
        << blockThreads
        << ", 1)\n"
        << "__global__ void " << kLlDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::ll::dispatch_kernel_args_t p) {\n";
    switch (algo) {
    case LlDispatchAlgo::k2SidedRmLsaTma:
        src << "  nccl_ep::ll::dispatch_kernel_impl_2sided_rm_lsa_tma<\n"
            << "      " << kernel_spec.recipe_source_literal << ",\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << numTmaStages << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ",\n"
            << "      " << numTmaQuantWarps << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(recordDstInfo) << ">(p);\n";
        break;
    case LlDispatchAlgo::k2SidedRmLsa:
        src << "  nccl_ep::ll::dispatch_kernel_impl_2sided_rm_lsa<\n"
            << "      " << kernel_spec.recipe_source_literal << ",\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ", " << (stageQuant ? "true" : "false") << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(recordDstInfo) << ">(p);\n";
        break;
    case LlDispatchAlgo::kCftRmLsa:
        src << "  nccl_ep::ll::dispatch_kernel_impl_cft_rm_lsa<\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(cftZeroCopy) << ", "
            << ::nccl_ep::jit::bool_literal(cftCountedPayload);
        src << ", " << (cftOneWordMask ? 1 : 8) << ">(p);\n";
        break;
    case LlDispatchAlgo::kDefault:
        src << "  nccl_ep::ll::dispatch_kernel_impl<\n"
            << "      " << kernel_spec.recipe_source_literal << ",\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << layout_literal << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(nvlinkOnly) << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ",\n"
            << "      " << kernel_spec.scale_type_literal << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(cftEnabled) << ", "
            << ::nccl_ep::jit::bool_literal(stageQuant) << ", "
            << ::nccl_ep::jit::bool_literal(rankOwnsExperts) << ", "
            << ::nccl_ep::jit::bool_literal(rankOwnsTokens) << ">(p);\n";
        break;
    }
    src << "}\n";
    return src.str();
}

inline ncclResult_t launch_ll_dispatch(
    int hidden,
    ncclEpLayout_t layout,
    bool nvlinkOnly,
    bool topkIdxIsInt64,
    const DispatchKernelSpec& kernel_spec,
    ncclDataType_t tokenDtype,
    ncclEpDispQuant_t recipe,
    int num_topk,
    int numSms,
    int numWarps,
    bool stageQuant,
    int cft_smem_bytes,
    NcclEpDispatchLsaMode lsaMode,
    int tmaQuantWarps,
    int maxDynamicSmem,
    bool recordDstInfo,
    bool cftZeroCopy,
    const dispatch_kernel_args_t& args,
    cudaStream_t stream,
    bool nucs,
    bool* dstInfoDisqualifiedOut,
    bool cftOneWordMask) {
    const bool cftEnabled = cft_smem_bytes != 0;
    const LlDispatchAlgo algo = ll_dispatch_select_algo(
        nvlinkOnly, recipe, layout, cftEnabled, lsaMode, hidden, tokenDtype, args.numAttnRanks);
    // Only STORE LSA kernels record one-sided-combine metadata.
    // Preserve any disqualification already established by the caller.
    *dstInfoDisqualifiedOut = *dstInfoDisqualifiedOut ||
        (algo != LlDispatchAlgo::k2SidedRmLsa && algo != LlDispatchAlgo::k2SidedRmLsaTma);
    const bool twoSidedRmLsa = algo == LlDispatchAlgo::k2SidedRmLsa;
    const bool cftRmLsa = algo == LlDispatchAlgo::kCftRmLsa;
    const bool twoSidedRmLsaTma = algo == LlDispatchAlgo::k2SidedRmLsaTma;
    const bool rankOwnsExperts = args.numAttnRanks == 0 || args.currRank >= args.numAttnRanks;
    const bool rankOwnsTokens = args.numAttnRanks == 0 || args.currRank < args.numAttnRanks;
    const bool tmaQuant = twoSidedRmLsaTma && recipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4;
    // Non-quantizing variants have no TMA quantization workers. Keep their
    // specialization independent of the DSFP8 TMA tuning knob.
    const int numTmaQuantWarps = tmaQuant ? tmaQuantWarps : 0;
    if (tmaQuant && (numTmaQuantWarps < dispatch_tma_smem::kMinQuantWarps ||
                     numTmaQuantWarps > dispatch_tma_smem::kMaxQuantWarps)) {
        std::fprintf(stderr, "[nccl_ep jit] LL TMA quantization worker count must be %d..%d (got %d).\n",
                     dispatch_tma_smem::kMinQuantWarps, dispatch_tma_smem::kMaxQuantWarps, numTmaQuantWarps);
        return ncclInvalidArgument;
    }
    const int requiredTmaWarps = dispatch_tma_smem::kControlWarps + numTmaQuantWarps +
                                dispatch_tma_smem::kCleanupWarps;
    if (tmaQuant && numWarps < requiredTmaWarps) {
        std::fprintf(stderr, "[nccl_ep jit] LL TMA quantization needs %d CTA warps (got %d).\n",
                     requiredTmaWarps, numWarps);
        return ncclInvalidArgument;
    }
    // General CFT zero-copy is separate from the LSA implementation. Reject
    // it before both cache paths, including receive-only continuations.
    if (cftZeroCopy && (!cftEnabled || !cftRmLsa)) {
        return ncclInvalidUsage;
    }
    // Specialize the local sender; each received source still carries its own put form.
    const bool cftCountedPayload = cftRmLsa && args.cft.countedPayload != 0;
    // Keep the compact thread limit with a one-block launch bound: repeated
    // single- and multi-node tuning did not favor the tighter register budget.
    const bool compactQuant = stageQuant && numWarps <= kLlDsFp8CompactWarps;

    // Validate before either cache path. Mask width never changes the grid,
    // warp grouping, launch bound, dynamic carve, or source count mode.
    if (cftOneWordMask &&
        (!cftEnabled || !cftRmLsa || hidden != 7168 || num_topk != 6 ||
         tokenDtype != ncclBfloat16 || !topkIdxIsInt64 || args.numRanks != 4 || args.numExperts != 384 ||
         args.maxTokensPerRank < 2048 || args.maxTokensPerRank > 8192 ||
         args.numTokens < 0 || args.numTokens > args.maxTokensPerRank ||
         numSms != 192 || numWarps != 32 || args.numWarpGroups != 2 || args.numWarpsPerGroup != 16 ||
         ll_lsa_cft_tokens_per_cta(args.maxTokensPerRank, numSms) > 64)) {
        return ncclInvalidArgument;
    }

    // TMA smem budget: stage-ring + rank-pointer-cache sizing is resolved here (host-side,
    // JIT-variant-selection time) from (hidden, dtype, maxDynamicSmem) only -- see
    // ll/ll_ep_smem.cuh's dispatch_tma_smem namespace doc comment for why this must not depend on
    // this call's token count.
    int numTmaStages = 0;
    int tmaDynamicSmemBytes = 0;
    if (twoSidedRmLsaTma) {
        // DSFP8 compacts in place; preserve the BF16 stage size and depth.
        const int hiddenBytes = static_cast<int>(::nccl_ep::token_row_bytes(tokenDtype, hidden));
        const size_t dispatchHdrSz = get_dispatch_hdr_sz(num_topk, layout);
        // Reserve syncAndSendCounts's static __shared__ state out of the budget before sizing the
        // dynamic stage ring -- see dispatch_tma_smem::kStaticSmemBytes's doc comment; same split
        // call_combine already applies for combine_smem::kLsaStaticSmemBytes.
        const int maxDynamicSmemForTma = maxDynamicSmem - dispatch_tma_smem::kStaticSmemBytes;
        const auto smemConfig = dispatch_tma_smem::choose_config(
            hiddenBytes, static_cast<int>(dispatchHdrSz), maxDynamicSmemForTma, args.numRanks);
        if (!smemConfig.feasible) {
            std::fprintf(
                stderr,
                "[nccl_ep jit] LL dispatch TMA shared memory cannot fit: hidden=%d, dtype=%d, numRanks=%d, "
                "limit=%d bytes (device limit=%d bytes, %d reserved for static __shared__ state).\n",
                hidden, static_cast<int>(tokenDtype), args.numRanks, maxDynamicSmemForTma, maxDynamicSmem,
                dispatch_tma_smem::kStaticSmemBytes);
            return ncclInvalidArgument;
        }
        numTmaStages = smemConfig.num_stages;
        tmaDynamicSmemBytes = smemConfig.dynamic_smem_bytes;
    }

    static const int variant_identity_default = 0;
    static const int variant_identity_2sided_rm_lsa = 0;
    static const int variant_identity_2sided_rm_lsa_tma = 0;
    static const int variant_identity_cft_rm_lsa = 0;

    ::nccl_ep::jit::JitKernelVariant variant;
    switch (algo) {
    case LlDispatchAlgo::k2SidedRmLsaTma:
        variant.kernel_family = "ll_dispatch_2sided_rm_lsa_tma";
        variant.identity = &variant_identity_2sided_rm_lsa_tma;
        break;
    case LlDispatchAlgo::k2SidedRmLsa:
        variant.kernel_family = "ll_dispatch_2sided_rm_lsa";
        variant.identity = &variant_identity_2sided_rm_lsa;
        break;
    case LlDispatchAlgo::kCftRmLsa:
        variant.kernel_family = "ll_dispatch_cft_rm_lsa";
        variant.identity = &variant_identity_cft_rm_lsa;
        break;
    case LlDispatchAlgo::kDefault:
    default:
        variant.kernel_family = "ll_dispatch";
        variant.identity = &variant_identity_default;
        break;
    }
    variant.entry_name = kLlDispatchJitEntryName;
    // The family identity separates CFT LSA from STORE and general CFT.
    // Within each family, derive the key from raw parameters so the warm-cache
    // launch path never has to build the variant-name string.
    std::uint64_t key = ::nccl_ep::jit::kRuntimeKeySeed;
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(hidden));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(layout));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(num_topk));
    key = ::nccl_ep::jit::runtime_key_mix(key, kernel_spec.recipe_cache_tag);
    key = ::nccl_ep::jit::runtime_key_mix(key, kernel_spec.payload_cache_tag);
    key = ::nccl_ep::jit::runtime_key_mix(key, kernel_spec.scale_cache_tag);
    key = ::nccl_ep::jit::runtime_key_mix(
        key, (nvlinkOnly ? 1u : 0u) | (topkIdxIsInt64 ? 2u : 0u) | (cftEnabled ? 4u : 0u));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(tokenDtype));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(compactQuant));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(stageQuant));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(rankOwnsExperts));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(rankOwnsTokens));
    key = ::nccl_ep::jit::runtime_key_mix(
        key, twoSidedRmLsaTma ? (1u | (static_cast<std::uint64_t>(numTmaStages) << 2)) : 0u);
    if (tmaQuant) {
        key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(numTmaQuantWarps));
    }
    key = ::nccl_ep::jit::runtime_key_mix(key, recordDstInfo ? 1u : 0u);
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(cftZeroCopy));
    if (cftRmLsa) {
        key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(cftCountedPayload));
        key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(numWarps));
        if (cftOneWordMask) {
            key = ::nccl_ep::jit::runtime_key_mix(key, 1u);
        }
    }
    variant.runtime_key = key;
    variant.num_blocks = numSms;
    variant.block_dim = numWarps * 32;
    // CFT staging and the TMA variant (mutually exclusive -- TMA is only ever selected when CFT is
    // disabled, see ll_dispatch_select_algo) use dynamic shared memory; store-only dispatch needs
    // none.
    variant.dynamic_smem_bytes = cftEnabled ? cft_smem_bytes : tmaDynamicSmemBytes;
    variant.nucs = nucs;
    // Always cooperative: kDefault needs its grid-wide SEND/RECV sync;
    // All LSA families use it for efficient cross-CTA atomic coordination.
    variant.cooperative = true;
    // Pair SMs into clusters of 2 when possible to share distributed SMEM.
    variant.cluster_dim_x = (numSms % 2 == 0) ? 2 : 1;

    std::string error;
    // Warm-cache fast path: launches without materializing variant_name/source.
    // sizeof = 0 means "kernel_param points to a single fixed-size arg struct".
    ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel_cached(
        variant, const_cast<dispatch_kernel_args_t*>(&args), 0, stream, &error);

    std::string variant_name;
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched &&
        status != ::nccl_ep::jit::JitKernelStatus::kLaunchFailed) {
        // Cache miss: build the name + source and take the compile/load path.
        std::ostringstream name;
        name << "ll_dispatch"
             << "_hdim" << hidden << ::nccl_ep::jit::layout_name_tag(layout)
             << "_topk" << num_topk
             << "_recipe" << kernel_spec.recipe_cache_tag
             << "_payload" << kernel_spec.payload_cache_tag
             << "_scale" << kernel_spec.scale_cache_tag
             << (nvlinkOnly ? "_nvlinkonly" : "")
             << (twoSidedRmLsa ? "_2sidedrmlsa" : "")
             << (cftRmLsa ? "_cftrmlsa" : "")
             << (twoSidedRmLsaTma ? "_2sidedrmlsatma" : "")
             << (twoSidedRmLsaTma ? ("_stages" + std::to_string(numTmaStages)) : "")
             << (tmaQuant ? ("_qwarps" + std::to_string(numTmaQuantWarps)) : "")
             << (cftEnabled ? "_cft" : "")
             << (compactQuant ? "_compact" : "")
             << (stageQuant ? "_staged" : "")
             << (rankOwnsExperts ? "_expertowner" : "_nonexpert")
             << (rankOwnsTokens ? "_tokenowner" : "_nontoken")
             << (topkIdxIsInt64 ? "_topk64" : "_topk32")
             << ::nccl_ep::jit::token_dtype_name_tag(tokenDtype)
             << (recordDstInfo ? "_dstinfo1s" : "");
        if (cftZeroCopy) {
            name << "_zcopy";
        }
        if (cftRmLsa) {
            name << (cftCountedPayload ? "_counted" : "_uncounted");
            name << "_warps" << numWarps;
        }
        if (cftRmLsa && cftOneWordMask) {
            name << "_maskw1";
        }
        variant_name = name.str();
        const std::string source = ll_dispatch_jit_source(
            kernel_spec, hidden, num_topk, layout, nvlinkOnly, topkIdxIsInt64, tokenDtype, algo,
            compactQuant, stageQuant, cftEnabled, rankOwnsExperts, rankOwnsTokens,
            numTmaStages, numTmaQuantWarps, recordDstInfo,
            cftZeroCopy, cftCountedPayload, numWarps, cftOneWordMask);
        variant.variant_name = variant_name;
        variant.source = source;
        status = ::nccl_ep::jit::launch_jit_kernel(
            variant, const_cast<dispatch_kernel_args_t*>(&args), stream, &error);
    }

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal LL dispatch JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

} // namespace jit
} // namespace ll
} // namespace nccl_ep
