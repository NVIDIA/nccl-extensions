/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "nccl_ep.h"
#include "device/jit/ll_dispatch_jit.cuh"
#include "device/jit/ll_combine_jit.cuh"

using namespace nccl_ep::ll;
using namespace nccl_ep::ll::jit;

TEST(LlKernelSelection, PartialDispatchTilesUseUniversalInBothModes) {
    for (auto dtype : {ncclBfloat16, ncclFloat16, ncclFloat32}) {
        const int tile = dtype == ncclFloat32 ? 128 : 256;
        for (auto mode : {NcclEpDispatchLsaMode::kSimt, NcclEpDispatchLsaMode::kTma}) {
            for (int hidden : {2880, tile - 16, tile + 16}) {
                EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
                    NCCL_EP_LAYOUT_RANK_MAJOR, false, mode, hidden, dtype), LlDispatchAlgo::kDefault);
            }
            for (int hidden : {tile, 3072, 7168}) {
                EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
                    NCCL_EP_LAYOUT_RANK_MAJOR, false, mode, hidden, dtype),
                    mode == NcclEpDispatchLsaMode::kTma ? LlDispatchAlgo::k2SidedRmLsaTma :
                                                        LlDispatchAlgo::k2SidedRmLsa);
            }
        }
    }
}

TEST(LlKernelSelection, DsFp8DispatchSupportsTmaAndSimt) {
    EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_DS_FP8E3M4,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, NcclEpDispatchLsaMode::kTma, 7168, ncclBfloat16),
        LlDispatchAlgo::k2SidedRmLsaTma);
    EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_DS_FP8E3M4,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, NcclEpDispatchLsaMode::kSimt, 7168, ncclBfloat16),
        LlDispatchAlgo::k2SidedRmLsa);
}

TEST(LlKernelSelection, DispatchPreservesOtherFallbacks) {
    EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_DS_FP8E3M4,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, NcclEpDispatchLsaMode::kTma, 128, ncclBfloat16),
        LlDispatchAlgo::kDefault);
    for (bool nvlink : {false, true}) {
        for (bool cft : {false, true}) {
            EXPECT_EQ(ll_dispatch_select_algo(nvlink, NCCL_EP_DISP_QUANT_NONE,
                NCCL_EP_LAYOUT_RANK_MAJOR, cft, NcclEpDispatchLsaMode::kUniversal, 7168, ncclBfloat16),
                LlDispatchAlgo::kDefault);
            if (!nvlink) {
                EXPECT_EQ(ll_dispatch_select_algo(nvlink, NCCL_EP_DISP_QUANT_NONE,
                    NCCL_EP_LAYOUT_RANK_MAJOR, cft, NcclEpDispatchLsaMode::kTma, 7168, ncclBfloat16),
                    LlDispatchAlgo::kDefault);
            }
        }
    }
}

// Enforcing k2SidedRmLsa only reads lsaEligible, never oneSidedEligible -- pass false for the
// latter throughout, since these cases never exercise the 1-sided/auto paths. NCCL_EP_COMBINE_LSA
// only governs the pick within the LSA domain, so a call that isn't LSA-eligible at all falls back
// to kDefault silently rather than erroring -- see ll_combine_select_kernel's doc comment.

TEST(LlKernelSelection, DispatchJitBurnsInRankRoles) {
    nccl_ep::DispatchKernelSpec spec{};
    ASSERT_EQ(nccl_ep::resolveDispatchKernelSpec(
        NCCL_EP_DISP_QUANT_NONE, ncclBfloat16, ncclFloat32, &spec), ncclSuccess);

    const auto source_for = [&](bool rank_owns_experts, bool rank_owns_tokens) {
        return ll_dispatch_jit_source(
            spec, 2048, 2, NCCL_EP_LAYOUT_RANK_MAJOR, true, false,
            ncclBfloat16, LlDispatchAlgo::kDefault, false, false, false,
            rank_owns_experts, rank_owns_tokens, 0, 0, false, false, false, 4, false);
    };

    EXPECT_NE(source_for(true, true).find("false, false, true, true>(p);"), std::string::npos);
    EXPECT_NE(source_for(false, true).find("false, false, false, true>(p);"), std::string::npos);
    EXPECT_NE(source_for(true, false).find("false, false, true, false>(p);"), std::string::npos);
}

TEST(LlKernelSelection, CombineRequiresFullReceiveTiles) {
    for (auto dtype : {ncclBfloat16, ncclFloat16, ncclFloat32}) {
        const int tile = dtype == ncclFloat32 ? 256 : 512;
        for (int hidden : {2880, tile / 2, tile - 16, tile + 16}) {
            const bool eligible = ll_combine_lsa_eligible(
                true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR, false, hidden, dtype,
                NcclEpDispatchLsaMode::kTma);
            EXPECT_EQ(
                ll_combine_select_kernel(LlCombineKernel::k2SidedRmLsa, eligible, false),
                LlCombineKernel::kDefault);
        }
        for (int hidden : {tile, 3072, 7168}) {
            const bool eligible = ll_combine_lsa_eligible(
                true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR, false, hidden, dtype,
                NcclEpDispatchLsaMode::kTma);
            EXPECT_EQ(
                ll_combine_select_kernel(LlCombineKernel::k2SidedRmLsa, eligible, false),
                LlCombineKernel::k2SidedRmLsa);
        }
    }
    EXPECT_EQ(
        ll_combine_select_kernel(
            LlCombineKernel::k2SidedRmLsa,
            ll_combine_lsa_eligible(
                true, NCCL_EP_COMB_QUANT_NONE, true, NCCL_EP_LAYOUT_RANK_MAJOR, false, 7168, ncclBfloat16,
                NcclEpDispatchLsaMode::kTma),
            false),
        LlCombineKernel::kDefault);
    EXPECT_EQ(
        ll_combine_select_kernel(
            LlCombineKernel::k2SidedRmLsa,
            ll_combine_lsa_eligible(
                true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR, true, 7168, ncclBfloat16,
                NcclEpDispatchLsaMode::kTma),
            false),
        ll_cft_lsa_enabled() ? LlCombineKernel::k2SidedRmLsa : LlCombineKernel::kDefault);
    EXPECT_EQ(
        ll_combine_select_kernel(
            LlCombineKernel::k2SidedRmLsa,
            ll_combine_lsa_eligible(
                false, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR, false, 7168, ncclBfloat16,
                NcclEpDispatchLsaMode::kTma),
            false),
        LlCombineKernel::kDefault);
}

// k1SidedRmLsa, unlike k2SidedRmLsa, still errors when the call IS LSA-eligible but 1-sided
// specifically isn't (oneSidedEligible=false) -- that's an in-domain request the enforcement can't
// honor, not an out-of-domain no-op. Falls back to kDefault only when not even LSA-eligible.
TEST(LlKernelSelection, OneSidedRequiresInDomainEligibility) {
    const bool lsa_eligible = ll_combine_lsa_eligible(
        true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR, false, 7168, ncclBfloat16,
        NcclEpDispatchLsaMode::kTma);
    ASSERT_TRUE(lsa_eligible);
    EXPECT_EQ(
        ll_combine_select_kernel(LlCombineKernel::k1SidedRmLsa, lsa_eligible, false),
        LlCombineKernel::kInvalid);
    EXPECT_EQ(
        ll_combine_select_kernel(LlCombineKernel::k1SidedRmLsa, lsa_eligible, true),
        LlCombineKernel::k1SidedRmLsa);
    const bool not_lsa_eligible = ll_combine_lsa_eligible(
        false, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR, false, 7168, ncclBfloat16,
        NcclEpDispatchLsaMode::kTma);
    ASSERT_FALSE(not_lsa_eligible);
    EXPECT_EQ(
        ll_combine_select_kernel(LlCombineKernel::k1SidedRmLsa, not_lsa_eligible, false),
        LlCombineKernel::kDefault);
}

// run_tests.sh runs these checks with NCCL_EP_LL_CFT_LSA both enabled and disabled.
TEST(LlKernelSelection, CftDispatchSelection) {
    const auto expected = ll_cft_lsa_enabled() ? LlDispatchAlgo::kCftRmLsa : LlDispatchAlgo::kDefault;
    for (auto mode : {NcclEpDispatchLsaMode::kSimt, NcclEpDispatchLsaMode::kTma}) {
        EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
            NCCL_EP_LAYOUT_RANK_MAJOR, true, mode, 7168, ncclBfloat16), expected);
        EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
            NCCL_EP_LAYOUT_RANK_MAJOR, true, mode, 2880, ncclBfloat16), LlDispatchAlgo::kDefault);
        EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_DS_FP8E3M4,
            NCCL_EP_LAYOUT_RANK_MAJOR, true, mode, 7168, ncclBfloat16), LlDispatchAlgo::kDefault);
        EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
            NCCL_EP_LAYOUT_EXPERT_MAJOR, true, mode, 7168, ncclBfloat16), LlDispatchAlgo::kDefault);
    }
}

TEST(LlKernelSelection, CftCombineFollowsDispatchProtocol) {
    const bool eligible = ll_combine_lsa_eligible(
        true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR,
        true, 7168, ncclBfloat16, NcclEpDispatchLsaMode::kTma);
    EXPECT_EQ(ll_combine_select_kernel(LlCombineKernel::kNone, eligible, false),
              ll_cft_lsa_enabled() ? LlCombineKernel::k2SidedRmLsa : LlCombineKernel::kDefault);
    EXPECT_EQ(ll_combine_select_kernel(LlCombineKernel::k1SidedRmLsa, eligible, false),
              ll_cft_lsa_enabled() ? LlCombineKernel::kInvalid : LlCombineKernel::kDefault);

    const bool generalCft = ll_combine_lsa_eligible(
        true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR,
        true, 7168, ncclBfloat16, NcclEpDispatchLsaMode::kUniversal);
    EXPECT_FALSE(generalCft);
    EXPECT_EQ(ll_combine_select_kernel(LlCombineKernel::k2SidedRmLsa, generalCft, false),
              LlCombineKernel::kDefault);
    EXPECT_TRUE(ll_combine_lsa_eligible(
        true, NCCL_EP_COMB_QUANT_NONE, false, NCCL_EP_LAYOUT_RANK_MAJOR,
        false, 7168, ncclBfloat16, NcclEpDispatchLsaMode::kUniversal));
}

TEST(LlKernelSelection, CftZeroCopyRequiresLsaAndHalfPrecision) {
    for (auto dtype : {ncclBfloat16, ncclFloat16}) {
        EXPECT_EQ(ll_dispatch_cft_zero_copy_supported(true, NCCL_EP_DISP_QUANT_NONE,
            NCCL_EP_LAYOUT_RANK_MAJOR, NcclEpDispatchLsaMode::kTma, 7168, dtype), ll_cft_lsa_enabled());
    }
    EXPECT_FALSE(ll_dispatch_cft_zero_copy_supported(true, NCCL_EP_DISP_QUANT_NONE,
        NCCL_EP_LAYOUT_RANK_MAJOR, NcclEpDispatchLsaMode::kTma, 7168, ncclFloat32));
    EXPECT_FALSE(ll_dispatch_cft_zero_copy_supported(true, NCCL_EP_DISP_QUANT_NONE,
        NCCL_EP_LAYOUT_RANK_MAJOR, NcclEpDispatchLsaMode::kUniversal, 7168, ncclBfloat16));
    EXPECT_FALSE(ll_dispatch_cft_zero_copy_supported(true, NCCL_EP_DISP_QUANT_NONE,
        NCCL_EP_LAYOUT_RANK_MAJOR, NcclEpDispatchLsaMode::kTma, 2880, ncclBfloat16));
}

TEST(LlKernelSelection, CftReceiveWarpsFollowCapacity) {
    EXPECT_EQ(ll_lsa_cft_choose_warps_per_group(16, 128, 96), 2);
    EXPECT_EQ(ll_lsa_cft_choose_warps_per_group(16, 256, 96), 4);
    EXPECT_EQ(ll_lsa_cft_choose_warps_per_group(16, 8192, 96), 16);
}

TEST(LlKernelSelection, CftJitSeparatesSenderModes) {
    nccl_ep::DispatchKernelSpec spec{};
    ASSERT_EQ(nccl_ep::resolveDispatchKernelSpec(
        NCCL_EP_DISP_QUANT_NONE, ncclBfloat16, ncclFloat32, &spec), ncclSuccess);
    const std::string counted = ll_dispatch_jit_source(
        spec, 7168, 8, NCCL_EP_LAYOUT_RANK_MAJOR, true, true, ncclBfloat16,
        LlDispatchAlgo::kCftRmLsa, false, false, true, true, true,
        0, 0, false, false, true, 4, false);
    const std::string uncounted = ll_dispatch_jit_source(
        spec, 7168, 8, NCCL_EP_LAYOUT_RANK_MAJOR, true, true, ncclBfloat16,
        LlDispatchAlgo::kCftRmLsa, false, false, true, true, true,
        0, 0, false, false, false, 4, false);
    const std::string zeroCopy = ll_dispatch_jit_source(
        spec, 7168, 8, NCCL_EP_LAYOUT_RANK_MAJOR, true, true, ncclBfloat16,
        LlDispatchAlgo::kCftRmLsa, false, false, true, true, true,
        0, 0, false, true, false, 4, false);
    EXPECT_NE(counted, uncounted);
    EXPECT_NE(uncounted, zeroCopy);
    EXPECT_NE(counted.find("device/ll/ll_dispatch_cft_lsa.cuh"), std::string::npos);
    EXPECT_NE(counted.find("dispatch_kernel_impl_cft_rm_lsa<"), std::string::npos);
}
