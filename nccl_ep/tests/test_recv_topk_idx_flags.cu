/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression coverage for LL rank-major recv_topk_idx sentinel reset. Every rank sends
// one token to global expert 0, leaving a tail in every source-rank row.

#include "test_common.h"

#include <cstdint>

namespace {

constexpr int kFlagNumTokens = 1;
constexpr int kFlagMaxTokens = 4;
constexpr int kFlagTopK = 1;
constexpr int kFlagHidden = 256;

class RecvTopkIdxSentinelTest : public ::testing::Test {
protected:
    ncclEpGroup_t group_ = nullptr;
    ncclEpHandle_t handle_ = nullptr;
    int32_t *d_topk_ = nullptr, *d_recv_idx_ = nullptr;
    nv_bfloat16 *d_tokens_ = nullptr, *d_recv_tokens_ = nullptr;
    float *d_weights_ = nullptr, *d_recv_weights_ = nullptr;
    ncclEpTensor_t *t_topk_ = nullptr, *t_tokens_ = nullptr;
    ncclEpTensor_t *t_recv_tokens_ = nullptr, *t_weights_ = nullptr;
    ncclEpTensor_t *t_recv_weights_ = nullptr, *t_recv_idx_ = nullptr;

    void SetUp() override {
        ncclEpGroupConfig_t group_cfg = NCCL_EP_GROUP_CONFIG_INIT;
        group_cfg.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
        group_cfg.num_experts = static_cast<unsigned int>(g_nranks);
        group_cfg.max_dispatch_tokens_per_rank = kFlagMaxTokens;
        group_cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(g_nranks * kFlagMaxTokens);
        group_cfg.max_token_bytes = kFlagHidden * sizeof(nv_bfloat16);
        group_cfg.rdma_buffer_size = NCCL_EP_AUTO;
        group_cfg.num_qp_per_rank = 1;
        group_cfg.num_channels = NCCL_EP_AUTO;
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &group_cfg));

        int32_t h_topk[kFlagNumTokens * kFlagTopK] = {0};
        CUDA_ASSERT(cudaMalloc(&d_topk_, sizeof(h_topk)));
        CUDA_ASSERT(cudaMemcpy(d_topk_, h_topk, sizeof(h_topk), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&t_topk_, 2, ncclInt32, d_topk_, kFlagNumTokens, kFlagTopK));
        NCCL_ASSERT(ncclEpCreateHandle(
            &handle_, group_, NCCL_EP_LAYOUT_RANK_MAJOR, t_topk_, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        CUDA_ASSERT(cudaMalloc(&d_tokens_, kFlagNumTokens * kFlagHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_weights_, kFlagNumTokens * kFlagTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_tokens_, g_nranks * kFlagMaxTokens * kFlagHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_recv_weights_, g_nranks * kFlagMaxTokens * kFlagTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_idx_, g_nranks * kFlagMaxTokens * kFlagTopK * sizeof(int32_t)));

        std::vector<nv_bfloat16> h_tokens(kFlagNumTokens * kFlagHidden, __float2bfloat16(1.0f));
        const float h_weights[kFlagNumTokens * kFlagTopK] = {1.0f};
        CUDA_ASSERT(cudaMemcpy(
            d_tokens_, h_tokens.data(), h_tokens.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_weights_, h_weights, sizeof(h_weights), cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&t_tokens_, 2, ncclBfloat16, d_tokens_, kFlagNumTokens, kFlagHidden));
        NCCL_ASSERT(epTensorCreate(&t_weights_, 2, ncclFloat32, d_weights_, kFlagNumTokens, kFlagTopK));
        NCCL_ASSERT(epTensorCreate(
            &t_recv_tokens_, 3, ncclBfloat16, d_recv_tokens_, g_nranks, kFlagMaxTokens, kFlagHidden));
        NCCL_ASSERT(epTensorCreate(
            &t_recv_weights_, 3, ncclFloat32, d_recv_weights_, g_nranks, kFlagMaxTokens, kFlagTopK));
        NCCL_ASSERT(epTensorCreate(
            &t_recv_idx_, 3, ncclInt32, d_recv_idx_, g_nranks, kFlagMaxTokens, kFlagTopK));
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
        CUDA_ASSERT(cudaMemset(d_recv_idx_, 0x5a, g_nranks * kFlagMaxTokens * kFlagTopK * sizeof(int32_t)));

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

        std::vector<int32_t> recv_idx(g_nranks * kFlagMaxTokens * kFlagTopK);
        CUDA_ASSERT(cudaMemcpy(
            recv_idx.data(), d_recv_idx_, recv_idx.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

        const int expected_per_source = g_rank == 0 ? kFlagNumTokens : 0;
        for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
            if (expected_per_source != 0) {
                const size_t live_offset = static_cast<size_t>(src_rank) * kFlagMaxTokens * kFlagTopK;
                EXPECT_EQ(recv_idx[live_offset], 0) << "source rank=" << src_rank;
            }
            for (int token = expected_per_source; token < kFlagMaxTokens; ++token) {
                const size_t offset = (static_cast<size_t>(src_rank) * kFlagMaxTokens + token) * kFlagTopK;
                EXPECT_EQ(recv_idx[offset], -1)
                    << "source rank=" << src_rank << " token=" << token;
            }
        }

        if (expect_counters) {
            std::vector<int32_t> counters(g_nranks);
            CUDA_ASSERT(cudaMemcpy(counters.data(), d_counters, counters.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
                EXPECT_EQ(counters[src_rank], expected_per_source) << "source rank=" << src_rank;
            }
        }

        if (t_counters) ncclEpTensorDestroy(t_counters);
        if (d_counters) cudaFree(d_counters);
    }
};

TEST_F(RecvTopkIdxSentinelTest, OptionalCountersAreWritten) {
    run_case(true);
}

TEST_F(RecvTopkIdxSentinelTest, SentinelsResetWithoutCounters) {
    run_case(false);
}

} // namespace

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_recv_topk_idx_flags_uid")) return 0;
    const int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
