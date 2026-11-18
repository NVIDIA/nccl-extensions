/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "nccl_ep.h"
#include "device/ll/ll_hier_delivery.cuh"
#include <cstddef>
#include <initializer_list>
#include <vector>
#include "device/jit/ll_dispatch_jit.cuh"
#include "device/jit/ll_combine_jit.cuh"

using namespace nccl_ep::ll;
using namespace nccl_ep::ll::jit;

TEST(LlHierPolicy, CapacityGeometryHonorsSmBudget) {
  EXPECT_EQ(hierLogicalCtas(148, 148), 384);
  EXPECT_EQ(hierLogicalCtas(32, 148), 32);
  EXPECT_EQ(hierLogicalCtas(80, 80), 240);
  for (int tokens : {0, 1, 4, 5, 8, 9, 16, 17, 32, 33}) {
    const bool direct = hierSelectDirect(hierDirectCutoff(384), tokens);
    const auto geometry = selectHierGeometry(384, tokens);
    EXPECT_EQ(direct, tokens <= 32);
    if (direct) {
      EXPECT_EQ(geometry.blocks, tokens == 0 || (tokens > 8 && tokens <= 16) ? 128 : 256);
      EXPECT_EQ(geometry.threads, tokens > 4 && tokens <= 16 ? 256 : 128);
    } else {
      EXPECT_EQ(geometry.blocks, 384);
      EXPECT_EQ(geometry.numRdmaRecvCtas, 256);
    }
  }
  EXPECT_EQ(selectHierGeometry(32, 1).blocks, 32);
}

TEST(LlHierPolicy, DirectRequiresSparseSourceCount) {
  EXPECT_EQ(hierDirectCutoff(32), 20);
  EXPECT_EQ(hierDirectCutoff(384), 32);
  for (int logicalCtas = 3; logicalCtas <= int(kHierMaxLogicalCtas); ++logicalCtas) {
    SCOPED_TRACE(logicalCtas);
    const int sendCtas = hierNumRdmaSendCtas(logicalCtas);
    const int cutoff = hierDirectCutoff(logicalCtas);
    EXPECT_LT(cutoff, sendCtas);
    for (int tokens : {cutoff, cutoff + 1, sendCtas - 1, sendCtas}) {
      SCOPED_TRACE(tokens);
      const bool direct = hierSelectDirect(cutoff, tokens);
      EXPECT_EQ(direct, tokens <= kHierDirectMaxTokens && tokens < sendCtas);
      const auto geometry = selectHierGeometry(logicalCtas, tokens);
      EXPECT_EQ(geometry.minCtasPerSm < kHierMinCtasPerSm, direct);
      if (direct) {
        EXPECT_LT(tokens, sendCtas);
      } else {
        EXPECT_EQ(geometry.blocks, logicalCtas);
        EXPECT_EQ(geometry.numRdmaRecvCtas, sendCtas);
      }
    }
  }
}

template <typename WeightT>
class LlHierReceivePolicy : public ::testing::Test {};
using LlHierWeightTypes = ::testing::Types<float, __nv_bfloat16>;
TYPED_TEST_SUITE(LlHierReceivePolicy, LlHierWeightTypes);

TYPED_TEST(LlHierReceivePolicy, ReceiveChoiceUsesCapacityAndTopologyOnly) {
  DispatchParams<TypeParam> params{};
  params.roundRobinReceiveSupported = true;
  params.hidden = 7168;
  params.numTopk = 8;
  params.deviceSm = 100;
  params.numRanks = 16;
  params.hier.numLogicalCtas = 384;
  params.maxTokensPerRank = 8192;
  params.tokenDtype = ncclBfloat16;
  params.layout = NCCL_EP_LAYOUT_RANK_MAJOR;
  for (int tokens : {-1, 0, 1, 16, 32, 33, 63, 64, 256, 8192, 8193}) {
    params.numTokens = tokens;
    EXPECT_TRUE(hierSelectRoundRobinReceive(params));
  }
  params.numTokens = 256;
  for (unsigned sm : {80u, 90u, 100u, 103u}) {
    params.deviceSm = sm;
    EXPECT_TRUE(hierSelectRoundRobinReceive(params));
  }
  for (int topk : {0, 1, 6, 7, 8, 32}) {
    params.numTopk = topk;
    EXPECT_EQ(hierSelectRoundRobinReceive(params), topk == 6 || topk == 8);
  }
  params.numTopk = 8;
  params.numRanks = 32;
  EXPECT_FALSE(hierSelectRoundRobinReceive(params));
  params.numRanks = 16;
  params.maxTokensPerRank = 8193;
  EXPECT_FALSE(hierSelectRoundRobinReceive(params));
  params.maxTokensPerRank = 63;
  EXPECT_FALSE(hierSelectRoundRobinReceive(params));
  params.maxTokensPerRank = 64;
  EXPECT_TRUE(hierSelectRoundRobinReceive(params));
  params.maxTokensPerRank = 512;
  params.roundRobinReceiveSupported = false;
  EXPECT_FALSE(hierSelectRoundRobinReceive(params));
  params.roundRobinReceiveSupported = true;
  params.hier.numLogicalCtas = 32;
  EXPECT_FALSE(hierSelectRoundRobinReceive(params));
}

TEST(LlKernelSelection, PartialDispatchTilesUseUniversalInBothModes) {
    struct Case {
        ncclEpDispQuant_t recipe;
        ncclDataType_t dtype;
    };
    // QUANT_FWD also forwards raw 8-bit and packed fp4 rows; a warp tile is 32 lanes * 16 bytes.
    const Case cases[] = {
        {NCCL_EP_DISP_QUANT_NONE, ncclBfloat16}, {NCCL_EP_DISP_QUANT_NONE, ncclFloat16},
        {NCCL_EP_DISP_QUANT_NONE, ncclFloat32}, {NCCL_EP_DISP_QUANT_FWD, ncclBfloat16},
        {NCCL_EP_DISP_QUANT_FWD, ncclFloat8e4m3}, {NCCL_EP_DISP_QUANT_FWD, ncclFloat8e5m2},
        {NCCL_EP_DISP_QUANT_FWD, ncclFloat4x2}};
    for (const auto& c : cases) {
        const int tile = 32 * static_cast<int>(sizeof(int4)) / static_cast<int>(::nccl_ep::ncclTypeSize(c.dtype));
        for (auto mode : {NcclEpDispatchLsaMode::kSimt, NcclEpDispatchLsaMode::kTma}) {
            for (int hidden : {2880, tile - 16, tile + 16}) {
                EXPECT_EQ(ll_dispatch_select_algo(true, c.recipe,
                    NCCL_EP_LAYOUT_RANK_MAJOR, false, mode, hidden, c.dtype), LlDispatchAlgo::kDefault);
            }
            for (int hidden : {tile, 3072, 7168}) {
                EXPECT_EQ(ll_dispatch_select_algo(true, c.recipe,
                    NCCL_EP_LAYOUT_RANK_MAJOR, false, mode, hidden, c.dtype),
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
    // An unassigned dtype value has no wire size (ncclTypeSize == 0) and must not reach the tile division.
    const auto unassignedDtype = static_cast<ncclDataType_t>(15);
    EXPECT_EQ(::nccl_ep::ncclTypeSize(unassignedDtype), 0u);
    EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, NcclEpDispatchLsaMode::kTma, 7168, unassignedDtype),
        LlDispatchAlgo::kDefault);
    for (auto recipe : {NCCL_EP_DISP_QUANT_NONE, NCCL_EP_DISP_QUANT_FWD}) {
        EXPECT_EQ(ll_dispatch_select_algo(true, recipe,
            NCCL_EP_LAYOUT_EXPERT_MAJOR, false, NcclEpDispatchLsaMode::kTma, 7168, ncclBfloat16),
            LlDispatchAlgo::kDefault);
        for (bool nvlink : {false, true}) {
            for (bool cft : {false, true}) {
                EXPECT_EQ(ll_dispatch_select_algo(nvlink, recipe,
                    NCCL_EP_LAYOUT_RANK_MAJOR, cft, NcclEpDispatchLsaMode::kUniversal, 7168, ncclBfloat16),
                    LlDispatchAlgo::kDefault);
                if (!nvlink) {
                    EXPECT_EQ(ll_dispatch_select_algo(nvlink, recipe,
                        NCCL_EP_LAYOUT_RANK_MAJOR, cft, NcclEpDispatchLsaMode::kTma, 7168, ncclBfloat16),
                        LlDispatchAlgo::kDefault);
                }
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
            rank_owns_experts, rank_owns_tokens, 0, 0, false, false, false, 4, false, ncclFloat32);
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

TEST(LlKernelSelection, ForwardedScalesStayOffTheCftKernels) {
    // QUANT_FWD is forwarded by the non-CFT STORE/TMA LSA kernels only; with CFT enabled it keeps the
    // general kernel (the CFT LSA kernel is NONE-recipe only), and CFT zero-copy is never offered for it.
    for (auto mode : {NcclEpDispatchLsaMode::kSimt, NcclEpDispatchLsaMode::kTma}) {
        EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_FWD, NCCL_EP_LAYOUT_RANK_MAJOR, true, mode,
            4096, ncclBfloat16), LlDispatchAlgo::kDefault);
        EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_FWD, NCCL_EP_LAYOUT_RANK_MAJOR, false, mode,
            4096, ncclBfloat16),
            mode == NcclEpDispatchLsaMode::kTma ? LlDispatchAlgo::k2SidedRmLsaTma : LlDispatchAlgo::k2SidedRmLsa);
    }
    for (auto dtype : {ncclBfloat16, ncclFloat16}) {
        EXPECT_FALSE(ll_dispatch_cft_zero_copy_supported(true, NCCL_EP_DISP_QUANT_FWD, NCCL_EP_LAYOUT_RANK_MAJOR,
            NcclEpDispatchLsaMode::kTma, 4096, dtype));
    }
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
        0, 0, false, false, true, 4, false, ncclFloat32);
    const std::string uncounted = ll_dispatch_jit_source(
        spec, 7168, 8, NCCL_EP_LAYOUT_RANK_MAJOR, true, true, ncclBfloat16,
        LlDispatchAlgo::kCftRmLsa, false, false, true, true, true,
        0, 0, false, false, false, 4, false, ncclFloat32);
    const std::string zeroCopy = ll_dispatch_jit_source(
        spec, 7168, 8, NCCL_EP_LAYOUT_RANK_MAJOR, true, true, ncclBfloat16,
        LlDispatchAlgo::kCftRmLsa, false, false, true, true, true,
        0, 0, false, true, false, 4, false, ncclFloat32);
    EXPECT_NE(counted, uncounted);
    EXPECT_NE(uncounted, zeroCopy);
    EXPECT_NE(counted.find("device/ll/ll_dispatch_cft_lsa.cuh"), std::string::npos);
    EXPECT_NE(counted.find("dispatch_kernel_impl_cft_rm_lsa<"), std::string::npos);
}

namespace {
using namespace nccl_ep::ll;

TEST(LlHierProtocol, SenderSelectionIsLocal) {
  EXPECT_TRUE(hierSelectDirect(1, 0));
  EXPECT_TRUE(hierSelectDirect(1, 1));
  EXPECT_FALSE(hierSelectDirect(1, 256));
  EXPECT_FALSE(hierSelectDirect(0, 1));
  // Peers may choose independently in the same dispatch generation.
  const int tokens[] = {0, 1, 2, 256};
  const bool expected[] = {true, true, false, false};
  for (int rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(hierSelectDirect(1, tokens[rank]), expected[rank]);
  }
}

TEST(LlHierProtocol, CompactHeaderWrapsWithoutChangingDeliveryMode) {
  EXPECT_EQ(sizeof(HierDeliveryHeader), 8u);
  EXPECT_EQ(offsetof(HierDeliveryHeader, control), 4u);
  for (uint64_t epoch = 0; epoch < 256; ++epoch) {
    for (bool direct : {false, true}) {
      HierDeliveryHeader header{0xffU, hierPacketControl(epoch, direct)};
      EXPECT_EQ(hierDeliveryIsDirect(&header), direct);
      EXPECT_EQ(header.control & 15U, (epoch + 1) % 16);
      EXPECT_EQ(header.control & ~31U, 0U);
      EXPECT_EQ(header.control, hierPacketControl(epoch + 16, direct));
      // A label match is not a match of the final slot's full ready tag.
      EXPECT_NE(epoch + 1, epoch + 17);
    }
  }
}

TEST(LlHierProtocol, EntryMaskAddressesTwoNvl72Domains) {
  for (int rank = 0; rank < 144; ++rank) {
    // Entry one targets the second expert on any GPU, including GPU 143.
    // Entry zero is a disabled duplicate, so it cannot suppress entry one.
    const uint16_t experts[] = {uint16_t(rank * 2), uint16_t(rank * 2 + 1)};
    const HierDeliveryHeader header{2U, hierPacketControl(0, false)};
    EXPECT_FALSE(hierDeliveryEntryEnabled(&header, 0));
    ASSERT_TRUE(hierDeliveryEntryEnabled(&header, 1));
    EXPECT_EQ(experts[1] / 2, rank);
    EXPECT_FALSE(hierDeliveryEntryEnabled(&header, -1));
    EXPECT_FALSE(hierDeliveryEntryEnabled(&header, 32));
  }
  const HierDeliveryHeader last{uint32_t(1) << 31, 0};
  EXPECT_TRUE(hierDeliveryEntryEnabled(&last, 31));
}

TEST(LlHierProtocol, SparseRelayTerminalIncludesZeroAndRequiresSendCtaZero) {
  for (uint64_t epoch = 0; epoch < 64; ++epoch) {
    EXPECT_FALSE(hierDeliveryRelayComplete(0, epoch, 0));
    EXPECT_TRUE(hierDeliveryRelayComplete(epoch + 1, epoch, 17));
    for (int count : {0, 1, 17, 255}) {
      const auto relay = hierDeliveryCount(epoch, count, false);
      EXPECT_EQ(hierDeliveryDecode(relay, epoch), count);
      EXPECT_TRUE(hierDeliveryRelayComplete(relay, epoch, 0));
      EXPECT_FALSE(hierDeliveryRelayComplete(relay, epoch, 1));
      EXPECT_FALSE(hierDeliveryRelayComplete(relay, epoch + 2, 0));
      // Direct completions additionally require the strong full-generation
      // counter and cannot be consumed as ordinary relay terminals.
      EXPECT_FALSE(hierDeliveryRelayComplete(hierDeliveryCount(epoch, count, true), epoch, 0));
    }
  }
}

TEST(LlHierProtocol, StrongCountersSurviveSkippedDirectEpochsAndPackedEpochWrap) {
  uint64_t history[2] = {}, remote[2] = {};
  for (uint64_t epoch : {uint64_t(0), uint64_t(1), uint64_t(16), uint64_t(17), uint64_t(1) << 30,
                         (uint64_t(1) << 30) + 1, uint64_t(1) << 32, (uint64_t(1) << 32) + 1}) {
    const int bank = epoch & 1;
    const uint64_t tag = epoch + 1;
    remote[bank] += tag - history[bank];
    history[bank] = tag;
    EXPECT_EQ(remote[bank], tag);
  }
  const uint64_t old = hierDeliveryCount(0, 1, true);
  const uint64_t freshEpoch = uint64_t(1) << 30;
  EXPECT_EQ(old, hierDeliveryCount(freshEpoch, 1, true));
  EXPECT_EQ(hierDeliveryDecode(old, freshEpoch), 1);
  EXPECT_NE(uint64_t(1), freshEpoch + 1);  // stale strong counter must fail
}
TEST(LlHierProtocol, StridedPeersPreserveDenseContextOrdinals) {
  for (int domains : {2, 3, 4}) {
    for (int localSize : {1, 4, 8, 72}) {
      const int ranks = domains * localSize;
      for (int stride : {1, domains}) {
        for (int first = 0; first + (localSize - 1) * stride < ranks; ++first) {
          std::vector<bool> local(ranks, false);
          for (int i = 0; i < localSize; ++i) {
            local[first + i * stride] = true;
          }
          int ordinal = 0;
          for (int peer = 0; peer < ranks; ++peer) {
            if (!local[peer]) {
              EXPECT_EQ(hierRemotePeerOrdinal(peer, first, localSize, stride), ordinal);
              ++ordinal;
            }
          }
          EXPECT_EQ(ordinal, ranks - localSize);
          for (int numWarps : {1, 4, 8}) {
            std::vector<int> owners(ranks - localSize, 0);
            for (int warp = 0; warp < numWarps; ++warp) {
              for (int peer = warp; peer < ranks; peer += numWarps) {
                if (!local[peer]) {
                  const int index = hierRemotePeerOrdinal(peer, first, localSize, stride);
                  ASSERT_GE(index, 0);
                  ASSERT_LT(index, owners.size());
                  ++owners[index];
                }
              }
            }
            for (int count : owners) {
              EXPECT_EQ(count, 1);
            }
          }
        }
      }
    }
  }
}
}  // namespace
