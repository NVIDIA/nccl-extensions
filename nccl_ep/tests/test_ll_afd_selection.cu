/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "device/jit/ll_combine_jit.cuh"
#include "device/jit/ll_dispatch_jit.cuh"

namespace {

TEST(LlAfdSelectionTest, AfWrappersReturnErrorForInvalidHandle) {
    EXPECT_EQ(ncclEpAfDispatch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), ncclInvalidArgument);
    EXPECT_EQ(ncclEpAfCombine(nullptr, nullptr, nullptr, nullptr, nullptr), ncclInvalidArgument);
    EXPECT_EQ(ncclEpAfRecv(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), ncclInvalidArgument);
    EXPECT_EQ(ncclEpAfReturn(nullptr, nullptr, nullptr, nullptr, nullptr), ncclInvalidArgument);
}

TEST(LlAfdSelectionTest, SignalBanksCoverEverySourceRank) {
    constexpr int kNumRanks = 4;
    constexpr int kNumAttnRanks = 2;
    constexpr int kNumExperts = 32;
    constexpr int kNumLocalExperts = kNumExperts / (kNumRanks - kNumAttnRanks);

    const nccl_ep::LowLatencyLayout layout(
        /*num_max_dispatch_tokens_per_rank=*/1,
        /*max_token_bytes=*/2048 * sizeof(nv_bfloat16),
        kNumRanks,
        kNumExperts,
        /*num_topk=*/1,
        NCCL_EP_LAYOUT_RANK_MAJOR,
        kNumAttnRanks);

    EXPECT_EQ(layout.buffer.num_clean_int, kNumLocalExperts * kNumRanks);
    EXPECT_GE(layout.signal_slot_stride,
              static_cast<size_t>(kNumLocalExperts * kNumRanks) * sizeof(int));

    const nccl_ep::LowLatencyLayout classicLayout(
        /*num_max_dispatch_tokens_per_rank=*/1,
        /*max_token_bytes=*/2048 * sizeof(nv_bfloat16),
        kNumRanks,
        kNumExperts,
        /*num_topk=*/1,
        NCCL_EP_LAYOUT_RANK_MAJOR,
        /*numAttnRanks=*/0);
    EXPECT_EQ(classicLayout.buffer.num_clean_int, kNumExperts);
}

TEST(LlAfdSelectionTest, AfdIsNotLsaEligibleForDispatch) {
    using namespace nccl_ep::ll;
    using namespace nccl_ep::ll::jit;

    EXPECT_EQ(
        ll_dispatch_select_algo(
            /*nvlinkOnly=*/true,
            NCCL_EP_DISP_QUANT_NONE,
            NCCL_EP_LAYOUT_RANK_MAJOR,
            /*cftEnabled=*/false,
            NcclEpDispatchLsaMode::kTma,
            /*hidden=*/7168,
            ncclBfloat16,
            /*numAttnRanks=*/0),
        LlDispatchAlgo::k2SidedRmLsaTma);
    EXPECT_EQ(
        ll_dispatch_select_algo(
            /*nvlinkOnly=*/true,
            NCCL_EP_DISP_QUANT_NONE,
            NCCL_EP_LAYOUT_RANK_MAJOR,
            /*cftEnabled=*/false,
            NcclEpDispatchLsaMode::kTma,
            /*hidden=*/7168,
            ncclBfloat16,
            /*numAttnRanks=*/1),
        LlDispatchAlgo::kDefault);
}

TEST(LlAfdSelectionTest, AfdDispatchSelectorAlwaysUsesUniversalKernel) {
    using namespace nccl_ep::ll;
    using namespace nccl_ep::ll::jit;

    for (const int numAttnRanks : {1, 2}) {
        for (const auto dtype : {ncclBfloat16, ncclFloat16, ncclFloat32}) {
            for (const auto mode : {NcclEpDispatchLsaMode::kSimt,
                                    NcclEpDispatchLsaMode::kTma,
                                    NcclEpDispatchLsaMode::kUniversal}) {
                for (const bool nvlinkOnly : {false, true}) {
                    for (const bool cftEnabled : {false, true}) {
                        EXPECT_EQ(
                            ll_dispatch_select_algo(
                                nvlinkOnly,
                                NCCL_EP_DISP_QUANT_NONE,
                                NCCL_EP_LAYOUT_RANK_MAJOR,
                                cftEnabled,
                                mode,
                                /*hidden=*/7168,
                                dtype,
                                numAttnRanks),
                            LlDispatchAlgo::kDefault);
                    }
                }
            }
        }
    }
}

TEST(LlAfdSelectionTest, AfdForcesUniversalCombine) {
    using namespace nccl_ep::ll::jit;

    EXPECT_TRUE(
        ll_combine_lsa_eligible(
            /*nvlinkOnly=*/true,
            NCCL_EP_COMB_QUANT_NONE,
            /*useLogFmt=*/false,
            NCCL_EP_LAYOUT_RANK_MAJOR,
            /*cftEnabled=*/false,
            /*hidden=*/7168,
            ncclBfloat16,
            nccl_ep::ll::NcclEpDispatchLsaMode::kTma,
            /*numAttnRanks=*/0));
    const bool afdEligible =
        ll_combine_lsa_eligible(
            /*nvlinkOnly=*/true,
            NCCL_EP_COMB_QUANT_NONE,
            /*useLogFmt=*/false,
            NCCL_EP_LAYOUT_RANK_MAJOR,
            /*cftEnabled=*/false,
            /*hidden=*/7168,
            ncclBfloat16,
            nccl_ep::ll::NcclEpDispatchLsaMode::kTma,
            /*numAttnRanks=*/1);
    EXPECT_FALSE(afdEligible);
    EXPECT_EQ(
        ll_combine_select_kernel(
            nccl_ep::ll::LlCombineKernel::kNone, afdEligible, /*oneSidedEligible=*/false),
        nccl_ep::ll::LlCombineKernel::kDefault);
}

TEST(LlAfdSelectionTest, AfdCombineSelectorAlwaysUsesUniversalKernel) {
    using namespace nccl_ep::ll::jit;

    for (const int numAttnRanks : {1, 2}) {
        for (const auto dtype : {ncclBfloat16, ncclFloat16, ncclFloat32}) {
            for (const bool nvlinkOnly : {false, true}) {
                for (const bool cftEnabled : {false, true}) {
                    const bool eligible =
                        ll_combine_lsa_eligible(
                            nvlinkOnly,
                            NCCL_EP_COMB_QUANT_NONE,
                            /*useLogFmt=*/false,
                            NCCL_EP_LAYOUT_RANK_MAJOR,
                            cftEnabled,
                            /*hidden=*/7168,
                            dtype,
                            nccl_ep::ll::NcclEpDispatchLsaMode::kTma,
                            numAttnRanks);
                    EXPECT_FALSE(eligible);
                    EXPECT_EQ(
                        ll_combine_select_kernel(
                            nccl_ep::ll::LlCombineKernel::kNone,
                            eligible,
                            /*oneSidedEligible=*/false),
                        nccl_ep::ll::LlCombineKernel::kDefault);
                }
            }
        }
    }
}

} // namespace
