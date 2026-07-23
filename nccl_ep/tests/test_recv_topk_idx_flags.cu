/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression coverage for LL rank-major recv_topk_idx sentinel reset. Exercise one and
// two experts per rank, source-specific destinations, mixed top-k routes, partial tails,
// and zero-token dispatch.

#include "test_common.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kFlagMaxTokens = 4;
constexpr int kFlagHidden = 256;

enum class RoutePattern {
    kRankZero,
    kLastRank,
    kSourceRank,
    kSourceRankAndNext,
};

struct RecvTopkIdxSentinelCase {
    int experts_per_rank;
    int num_tokens;
    int topk;
    RoutePattern route_pattern;
};

constexpr RecvTopkIdxSentinelCase kCases[] = {
    {1, 1, 1, RoutePattern::kRankZero},          // One expert per rank.
    {2, 1, 1, RoutePattern::kRankZero},          // Second local expert on rank 0.
    {2, 1, 1, RoutePattern::kLastRank},          // Second local expert on the last rank.
    {2, 1, 2, RoutePattern::kSourceRankAndNext}, // Each source routes to itself and its next rank.
    {2, 2, 1, RoutePattern::kSourceRank},        // Two live slots and a partial tail.
    {2, 0, 1, RoutePattern::kRankZero},          // No live slots; reset the complete receive buffer.
};

class RecvTopkIdxSentinelTest : public ::testing::TestWithParam<RecvTopkIdxSentinelCase> {
protected:
    ncclEpGroup_t group_ = nullptr;
    ncclEpHandle_t handle_ = nullptr;
    int32_t *d_topk_ = nullptr, *d_recv_idx_ = nullptr;
    nv_bfloat16 *d_tokens_ = nullptr, *d_recv_tokens_ = nullptr;
    float *d_weights_ = nullptr, *d_recv_weights_ = nullptr;
    ncclEpTensor_t *t_topk_ = nullptr, *t_tokens_ = nullptr;
    ncclEpTensor_t *t_recv_tokens_ = nullptr, *t_weights_ = nullptr;
    ncclEpTensor_t *t_recv_weights_ = nullptr, *t_recv_idx_ = nullptr;
    std::vector<int32_t> h_topk_;

    const RecvTopkIdxSentinelCase& test_case() const {
        return GetParam();
    }

    std::vector<int32_t> routing_for_source(int src_rank) const {
        const auto& config = test_case();
        std::vector<int32_t> topk_idx(config.num_tokens * config.topk);
        for (int token = 0; token < config.num_tokens; ++token) {
            for (int topk = 0; topk < config.topk; ++topk) {
                int dst_rank = 0;
                switch (config.route_pattern) {
                    case RoutePattern::kRankZero:
                        break;
                    case RoutePattern::kLastRank:
                        dst_rank = g_nranks - 1;
                        break;
                    case RoutePattern::kSourceRank:
                        dst_rank = src_rank;
                        break;
                    case RoutePattern::kSourceRankAndNext:
                        dst_rank = (src_rank + topk) % g_nranks;
                        break;
                }
                const int local_expert = config.route_pattern == RoutePattern::kSourceRankAndNext ?
                                             topk :
                                             config.experts_per_rank - 1;
                topk_idx[token * config.topk + topk] = dst_rank * config.experts_per_rank + local_expert;
            }
        }
        return topk_idx;
    }

    std::vector<int> received_token_indices_for_source(const std::vector<int32_t>& source_topk) const {
        const auto& config = test_case();
        std::vector<int> tokens;
        for (int token = 0; token < config.num_tokens; ++token) {
            for (int topk = 0; topk < config.topk; ++topk) {
                if (source_topk[token * config.topk + topk] / config.experts_per_rank == g_rank) {
                    tokens.push_back(token);
                    break;
                }
            }
        }
        return tokens;
    }

    void SetUp() override {
        const auto& config = test_case();
        ncclEpGroupConfig_t group_cfg = NCCL_EP_GROUP_CONFIG_INIT;
        group_cfg.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
        group_cfg.num_experts = static_cast<unsigned int>(g_nranks * config.experts_per_rank);
        group_cfg.max_dispatch_tokens_per_rank = kFlagMaxTokens;
        group_cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(g_nranks * kFlagMaxTokens);
        group_cfg.max_token_bytes = kFlagHidden * sizeof(nv_bfloat16);
        group_cfg.rdma_buffer_size = NCCL_EP_AUTO;
        group_cfg.num_qp_per_rank = 1;
        group_cfg.num_channels = NCCL_EP_AUTO;
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &group_cfg));

        h_topk_ = routing_for_source(g_rank);
        CUDA_ASSERT(cudaMalloc(&d_topk_, sizeof(int32_t) * std::max<size_t>(1, h_topk_.size())));
        if (!h_topk_.empty()) {
            CUDA_ASSERT(cudaMemcpy(
                d_topk_, h_topk_.data(), h_topk_.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        NCCL_ASSERT(epTensorCreate(&t_topk_, 2, ncclInt32, d_topk_, config.num_tokens, config.topk));
        NCCL_ASSERT(ncclEpCreateHandle(
            &handle_, group_, NCCL_EP_LAYOUT_RANK_MAJOR, t_topk_, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        CUDA_ASSERT(cudaMalloc(
            &d_tokens_, std::max<size_t>(1, config.num_tokens * kFlagHidden) * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_weights_, std::max<size_t>(1, h_topk_.size()) * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_tokens_, g_nranks * kFlagMaxTokens * kFlagHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_recv_weights_, g_nranks * kFlagMaxTokens * config.topk * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_idx_, g_nranks * kFlagMaxTokens * config.topk * sizeof(int32_t)));

        std::vector<nv_bfloat16> h_tokens(config.num_tokens * kFlagHidden, __float2bfloat16(1.0f));
        std::vector<float> h_weights(h_topk_.size(), 1.0f);
        if (!h_tokens.empty()) {
            CUDA_ASSERT(cudaMemcpy(
                d_tokens_, h_tokens.data(), h_tokens.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_ASSERT(cudaMemcpy(d_weights_, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        NCCL_ASSERT(epTensorCreate(&t_tokens_, 2, ncclBfloat16, d_tokens_, config.num_tokens, kFlagHidden));
        NCCL_ASSERT(epTensorCreate(&t_weights_, 2, ncclFloat32, d_weights_, config.num_tokens, config.topk));
        NCCL_ASSERT(epTensorCreate(
            &t_recv_tokens_, 3, ncclBfloat16, d_recv_tokens_, g_nranks, kFlagMaxTokens, kFlagHidden));
        NCCL_ASSERT(epTensorCreate(
            &t_recv_weights_, 3, ncclFloat32, d_recv_weights_, g_nranks, kFlagMaxTokens, config.topk));
        NCCL_ASSERT(epTensorCreate(
            &t_recv_idx_, 3, ncclInt32, d_recv_idx_, g_nranks, kFlagMaxTokens, config.topk));
    }

    void TearDown() override {
        if (handle_) ncclEpHandleDestroy(handle_);
        if (t_recv_idx_) ncclEpTensorDestroy(t_recv_idx_);
        if (t_recv_weights_) ncclEpTensorDestroy(t_recv_weights_);
        if (t_weights_) ncclEpTensorDestroy(t_weights_);
        if (t_recv_tokens_) ncclEpTensorDestroy(t_recv_tokens_);
        if (t_tokens_) ncclEpTensorDestroy(t_tokens_);
        if (t_topk_) ncclEpTensorDestroy(t_topk_);
        if (d_recv_idx_) cudaFree(d_recv_idx_);
        if (d_recv_weights_) cudaFree(d_recv_weights_);
        if (d_tokens_) cudaFree(d_tokens_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_recv_tokens_) cudaFree(d_recv_tokens_);
        if (d_topk_) cudaFree(d_topk_);
        if (group_) ncclEpGroupDestroy(group_);
    }

    void run_case(bool expect_counters) {
        const auto& config = test_case();
        CUDA_ASSERT(cudaMemset(d_recv_idx_, 0x5a, g_nranks * kFlagMaxTokens * config.topk * sizeof(int32_t)));

        int32_t* d_counters = nullptr;
        ncclEpTensor_t* t_counters = nullptr;
        if (expect_counters) {
            CUDA_ASSERT(cudaMalloc(&d_counters, g_nranks * sizeof(int32_t)));
            CUDA_ASSERT(cudaMemset(d_counters, 0xa5, g_nranks * sizeof(int32_t)));
            NCCL_ASSERT(epTensorCreate(&t_counters, 1, ncclInt32, d_counters, g_nranks));
        }

        ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
        inputs.tokens = t_tokens_;
        inputs.topk_weights = t_weights_;
        outputs.tokens = t_recv_tokens_;
        outputs.topk_weights = t_recv_weights_;
        outputs.topk_idx = t_recv_idx_;
        layout_info.src_rank_counters = t_counters;
        NCCL_ASSERT(ncclEpDispatch(handle_, &inputs, &outputs, &layout_info, nullptr, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        std::vector<int32_t> recv_idx(g_nranks * kFlagMaxTokens * config.topk);
        CUDA_ASSERT(cudaMemcpy(
            recv_idx.data(), d_recv_idx_, recv_idx.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

        for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
            // Derive this row's expected tokens from the routing tensor constructed by
            // src_rank, mirroring the local h_topk_ construction in SetUp().
            const std::vector<int32_t> source_topk = routing_for_source(src_rank);
            const std::vector<int> received_tokens = received_token_indices_for_source(source_topk);
            const int expected_per_source = static_cast<int>(received_tokens.size());
            for (int slot = 0; slot < expected_per_source; ++slot) {
                const int token = received_tokens[slot];
                for (int topk = 0; topk < config.topk; ++topk) {
                    const int expert = source_topk[token * config.topk + topk];
                    const int expected_idx = expert / config.experts_per_rank == g_rank ?
                                                 expert % config.experts_per_rank :
                                                 -1;
                    const size_t offset =
                        (static_cast<size_t>(src_rank) * kFlagMaxTokens + slot) * config.topk + topk;
                    EXPECT_EQ(recv_idx[offset], expected_idx)
                        << "source rank=" << src_rank << " slot=" << slot << " topk=" << topk;
                }
            }
            for (int token = expected_per_source; token < kFlagMaxTokens; ++token) {
                for (int topk = 0; topk < config.topk; ++topk) {
                    const size_t offset =
                        (static_cast<size_t>(src_rank) * kFlagMaxTokens + token) * config.topk + topk;
                    EXPECT_EQ(recv_idx[offset], -1)
                        << "source rank=" << src_rank << " token=" << token << " topk=" << topk;
                }
            }
        }

        if (expect_counters) {
            std::vector<int32_t> counters(g_nranks);
            CUDA_ASSERT(cudaMemcpy(counters.data(), d_counters, counters.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
                const int expected_per_source =
                    static_cast<int>(received_token_indices_for_source(routing_for_source(src_rank)).size());
                EXPECT_EQ(counters[src_rank], expected_per_source) << "source rank=" << src_rank;
            }
        }

        if (t_counters) ncclEpTensorDestroy(t_counters);
        if (d_counters) cudaFree(d_counters);
    }
};

TEST_P(RecvTopkIdxSentinelTest, OptionalCountersAreWritten) {
    run_case(true);
}

TEST_P(RecvTopkIdxSentinelTest, SentinelsResetWithoutCounters) {
    run_case(false);
}

// Instantiate both counter modes for every routing configuration in kCases.
INSTANTIATE_TEST_SUITE_P(
    Routes, RecvTopkIdxSentinelTest, ::testing::ValuesIn(kCases));

} // namespace

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_recv_topk_idx_flags_uid")) return 0;
    const int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
