/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression test for the pull-dispatch chunk_ranges ring (HT expert-major, pull-push, fused count).
 *
 *   A MAP warp publishes each 32-token chunk's recv range into a small per-warp ring. A pull
 *   warp must read the range of the chunk its ticket names, even when the MAP warp runs a full
 *   ring ahead of a stalled reader. The other suites use 4 tokens per rank, so the ring never
 *   wraps; this suite dispatches enough tokens to wrap it many times.
 *
 *   The delayed-reader case sets NCCL_EP_TEST_PULL_STALL_US, which makes the first pull ticket
 *   of every block sleep before it reads its range (test-only hook in dispatch_pull). Without a
 *   read acknowledgment before ring reuse, the MAP warp overwrites the unread entry and the
 *   dispatch silently drops some tokens and duplicates others.
 *
 * Every (source rank, token) must be received exactly once on the rank hosting its expert.
 */

#include "test_common.h"
#include <cstdlib>

static constexpr int kStressTokens = 8192;  // enough 32-token chunks per block to wrap the ring

class HtPullChunkRingTest : public ::testing::Test {
protected:
    ncclEpGroup_t group_ = nullptr;

    void SetUp() override {
        if (kNumExperts % g_nranks != 0) GTEST_SKIP() << "needs a rank count dividing kNumExperts";
        const char* pull_push = std::getenv("NCCL_EP_HT_PULL_PUSH");
        if (pull_push == nullptr || pull_push[0] != '1') GTEST_SKIP() << "needs NCCL_EP_HT_PULL_PUSH=1";
        ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
        gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        gcfg.num_experts = kNumExperts;
        gcfg.max_dispatch_tokens_per_rank = kStressTokens;
        gcfg.max_token_bytes = kHidden * sizeof(nv_bfloat16);
        gcfg.rdma_buffer_size = NCCL_EP_AUTO;
        gcfg.num_qp_per_rank = NCCL_EP_AUTO;
        gcfg.num_channels = NCCL_EP_AUTO;
        gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecv());
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));
    }

    void TearDown() override {
        unsetenv("NCCL_EP_TEST_PULL_STALL_US");
        if (group_) ncclEpGroupDestroy(group_);
    }

    // Worst-case per-rank recv budget (every rank's tokens routed to one rank).
    static int kMaxRecv() { return g_nranks * kStressTokens; }
    static int experts_per_rank() { return kNumExperts / g_nranks; }
    static int route(int src_rank, int token) { return (src_rank * kStressTokens + token) % kNumExperts; }

    void run(int stall_us) {
        if (stall_us > 0) setenv("NCCL_EP_TEST_PULL_STALL_US", std::to_string(stall_us).c_str(), 1);

        // Row (src + 1, token / 64 + 1, token % 64 + 1, 0...): small integers are exact in bf16, and no
        // received row is all zero, so zero rows (padding / unwritten) can be told from tokens.
        std::vector<nv_bfloat16> h_tok(static_cast<size_t>(kStressTokens) * kHidden, __float2bfloat16(0.f));
        std::vector<int64_t> h_topk(kStressTokens);
        for (int i = 0; i < kStressTokens; ++i) {
            h_tok[static_cast<size_t>(i) * kHidden + 0] = __float2bfloat16(static_cast<float>(g_rank + 1));
            h_tok[static_cast<size_t>(i) * kHidden + 1] = __float2bfloat16(static_cast<float>(i / 64 + 1));
            h_tok[static_cast<size_t>(i) * kHidden + 2] = __float2bfloat16(static_cast<float>(i % 64 + 1));
            h_topk[i] = route(g_rank, i);
        }
        std::vector<float> h_w(static_cast<size_t>(kStressTokens) * kTopK, 1.0f);

        int64_t* d_topk = nullptr;
        nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
        float *d_w = nullptr, *d_recv_w = nullptr;
        const size_t recv_rows = static_cast<size_t>(kMaxRecv());
        CUDA_ASSERT(cudaMalloc(&d_topk, h_topk.size() * sizeof(int64_t)));
        CUDA_ASSERT(cudaMalloc(&d_tok, h_tok.size() * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv, recv_rows * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_recv_w, recv_rows * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemset(d_recv, 0, recv_rows * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(d_recv_w, 0, recv_rows * sizeof(float)));

        ncclEpTensor_t *t_topk = nullptr, *t_tok = nullptr, *t_w = nullptr, *t_recv = nullptr, *t_recv_w = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_topk, 2, ncclInt64, d_topk, kStressTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kStressTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_w, kStressTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, recv_rows, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, recv_rows));

        ncclEpHandle_t handle = nullptr;
        NCCL_ASSERT(ncclEpCreateHandle(
            &handle, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, t_topk, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = t_tok;
        d_in.topk_weights = t_w;
        d_out.tokens = t_recv;
        d_out.topk_weights = t_recv_w;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        NCCL_ASSERT(ncclEpDispatch(handle, &d_in, &d_out, nullptr, &dcfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        std::vector<nv_bfloat16> h_recv(recv_rows * kHidden);
        CUDA_ASSERT(cudaMemcpy(h_recv.data(), d_recv, h_recv.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

        // Each (src, token) routed to this rank's experts must appear exactly once.
        std::vector<uint8_t> seen(static_cast<size_t>(g_nranks) * kStressTokens, 0);
        size_t duplicates = 0, unexpected = 0;
        for (size_t r = 0; r < recv_rows; ++r) {
            const int src = static_cast<int>(__bfloat162float(h_recv[r * kHidden + 0])) - 1;
            const int hi = static_cast<int>(__bfloat162float(h_recv[r * kHidden + 1])) - 1;
            const int lo = static_cast<int>(__bfloat162float(h_recv[r * kHidden + 2])) - 1;
            if (src < 0 && hi < 0 && lo < 0) continue;  // padding / unwritten row
            const int token = hi * 64 + lo;
            if (src < 0 || src >= g_nranks || token < 0 || token >= kStressTokens ||
                route(src, token) / experts_per_rank() != g_rank) {
                ++unexpected;
                continue;
            }
            uint8_t& s = seen[static_cast<size_t>(src) * kStressTokens + token];
            if (s) ++duplicates;
            s = 1;
        }
        size_t missing = 0;
        for (int src = 0; src < g_nranks; ++src)
            for (int i = 0; i < kStressTokens; ++i)
                if (route(src, i) / experts_per_rank() == g_rank && !seen[static_cast<size_t>(src) * kStressTokens + i])
                    ++missing;
        EXPECT_EQ(missing, 0u) << "rank " << g_rank << ": tokens never received";
        EXPECT_EQ(duplicates, 0u) << "rank " << g_rank << ": tokens received more than once";
        EXPECT_EQ(unexpected, 0u) << "rank " << g_rank << ": rows that match no routed token";

        ncclEpHandleDestroy(handle);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_topk);
        cudaFree(d_recv_w);
        cudaFree(d_recv);
        cudaFree(d_w);
        cudaFree(d_tok);
        cudaFree(d_topk);
    }
};

// The ring wraps many times with readers keeping pace.
TEST_F(HtPullChunkRingTest, RingWraparound) {
    run(0);
}

// The first pull ticket of every block stalls while its block's MAP warps run across a ring wrap.
// The MAP warps must wait for the delayed reader instead of overwriting its entry.
TEST_F(HtPullChunkRingTest, DelayedReaderAcrossRingWrap) {
    run(3000);
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_pull_chunk_ring_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
