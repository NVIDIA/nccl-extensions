/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_common.h"
#include <array>
#include <set>

namespace {
constexpr int capacity = 16384;
constexpr int hidden = 128;
constexpr int topk = 2;

struct Tensor {
    ncclEpTensor_t* t = nullptr;
    void* data = nullptr;
    ~Tensor() {
        if (t) ncclEpTensorDestroy(t);
        if (data) cudaFree(data);
    }
    void init(ncclDataType_t dtype, size_t bytes, size_t rows, size_t cols = 0) {
        CUDA_ASSERT(cudaMalloc(&data, bytes));
        CUDA_ASSERT(cudaMemset(data, 0, bytes));
        if (cols) NCCL_ASSERT(epTensorCreate(&t, 2, dtype, data, rows, cols));
        else NCCL_ASSERT(epTensorCreate(&t, 1, dtype, data, rows));
    }
};

int tokens_for(int rank, int round, int handle) {
    if (round < 3) return ((round + handle) % 2) ? capacity : 128;
    constexpr std::array<int, 16> counts = {
        0, 1, 31, 32, 33, 512, 513, 1024, 1025, 2048, 2049, 4096, 4097, 8192, 8193, capacity};
    return counts[(rank + (round - 3) * 4 + handle * 8) % counts.size()];
}

int expert_for(int src, int token, int slot, int handle) {
    return 2 * ((src + token + slot + handle) % g_nranks) + slot;
}

float value_for(int src, int token, int column, int round, int handle) {
    switch (column % 4) {
        case 0: return src + 1;
        case 1: return token % 128;
        case 2: return token / 128;
        default: return 1 + round * 2 + handle;
    }
}

struct State {
    ncclEpHandle_t handle = nullptr;
    ncclEpTensor_t* mem = nullptr;
    void* storage = nullptr;
    size_t bytes = 0;
    Tensor idx, input, weights, recv, recv_weights, recv_idx, output, grad_weights, total, counts, offsets;
    ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
    ~State() {
        if (handle) ncclEpHandleDestroy(handle);
        if (mem) ncclEpTensorDestroy(mem);
        if (storage) cudaFree(storage);
    }
    void init(ncclEpGroup_t group, bool em) {
        const auto mode = em ? NCCL_EP_LAYOUT_EXPERT_MAJOR : NCCL_EP_LAYOUT_FLAT;
        const int recv_capacity = capacity * g_nranks * topk;
        ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
        cfg.dispatch_output_per_expert_alignment = 1;
        NCCL_ASSERT(ncclEpHandleMemSize(group, mode, &cfg, &bytes, topk));
        CUDA_ASSERT(cudaMalloc(&storage, bytes + 512));
        CUDA_ASSERT(cudaMemset(storage, 0xA5, bytes + 512));
        NCCL_ASSERT(epTensorCreate(&mem, 1, ncclUint8, static_cast<char*>(storage) + 256, bytes));
        NCCL_ASSERT(ncclEpInitHandle(&handle, group, mode, &cfg, topk, mem));
        idx.init(ncclInt64, capacity * topk * sizeof(int64_t), capacity, topk);
        input.init(ncclBfloat16, capacity * hidden * 2, capacity, hidden);
        weights.init(ncclFloat32, capacity * topk * sizeof(float), capacity, topk);
        recv.init(ncclBfloat16, recv_capacity * hidden * 2, recv_capacity, hidden);
        recv_weights.init(ncclFloat32, recv_capacity * topk * sizeof(float), recv_capacity, em ? 0 : topk);
        recv_idx.init(ncclInt64, recv_capacity * topk * sizeof(int64_t), recv_capacity, topk);
        output.init(ncclBfloat16, capacity * hidden * 2, capacity, hidden);
        grad_weights.init(ncclFloat32, capacity * topk * sizeof(float), capacity, topk);
        total.init(ncclInt64, sizeof(int64_t), 1);
        counts.init(ncclInt64, 2 * sizeof(int64_t), 2);
        offsets.init(ncclInt64, 2 * sizeof(int64_t), 2);
        layout.recv_total_counter = total.t;
        if (em) {
            layout.expert_counters = counts.t;
            layout.expert_offsets = offsets.t;
        }
    }
    void update(int round, int id, bool unfused) {
        const int n = tokens_for(g_rank, round, id);
        std::vector<int64_t> routing(capacity * topk, -1);
        std::vector<nv_bfloat16> values(capacity * hidden);
        std::vector<float> probs(capacity * topk, 0.5f);
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < topk; ++k) routing[i * topk + k] = expert_for(g_rank, i, k, id);
            for (int h = 0; h < hidden; ++h)
                values[i * hidden + h] = __float2bfloat16(value_for(g_rank, i, h, round, id));
        }
        CUDA_ASSERT(cudaMemcpy(idx.data, routing.data(), routing.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(input.data, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(weights.data, probs.data(), probs.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (auto* t : {idx.t, input.t, weights.t, output.t, grad_weights.t}) t->sizes[0] = n;
        NCCL_ASSERT(ncclEpUpdateHandle(handle, idx.t, unfused ? &layout : nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }
    void run(bool em, bool backward) {
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        in.tokens = input.t;
        in.topk_weights = backward ? nullptr : weights.t;
        out.tokens = recv.t;
        out.topk_weights = backward ? nullptr : recv_weights.t;
        if (!em && !backward) out.topk_idx = recv_idx.t;
        cfg.pass_direction = backward ? NCCL_EP_BWD_PASS : NCCL_EP_FWD_PASS;
        CUDA_ASSERT(cudaMemsetAsync(recv.data, 0xFF, capacity * g_nranks * topk * hidden * 2, g_stream));
        CUDA_ASSERT(cudaMemsetAsync(output.data, 0xFF, capacity * hidden * 2, g_stream));
        CUDA_ASSERT(cudaMemsetAsync(grad_weights.data, 0xFF, capacity * topk * sizeof(float), g_stream));
        NCCL_ASSERT(ncclEpDispatch(handle, &in, &out, &layout, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle, nullptr, g_stream));
        ncclEpCombineInputs_t ci = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t co = NCCL_EP_COMBINE_OUTPUTS_INIT;
        ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
        ci.tokens = recv.t;
        co.tokens = output.t;
        ccfg.pass_direction = cfg.pass_direction;
        if (backward) {
            ci.topk_weights = recv_weights.t;
            co.topk_weights = grad_weights.t;
        }
        NCCL_ASSERT(ncclEpCombine(handle, &ci, &co, &ccfg, g_stream));
    }
    void check(int round, int id, bool em, bool backward) {
        int64_t total_rows = -1, expert_counts[2], expert_offsets[2];
        CUDA_ASSERT(cudaMemcpy(&total_rows, total.data, sizeof(int64_t), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(expert_counts, counts.data, sizeof(expert_counts), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(expert_offsets, offsets.data, sizeof(expert_offsets), cudaMemcpyDeviceToHost));
        std::multiset<std::array<int, 3>> expected[2];
        for (int src = 0; src < g_nranks; ++src)
            for (int i = 0; i < tokens_for(src, round, id); ++i)
                for (int k = 0; k < topk; ++k)
                    if (expert_for(src, i, k, id) / 2 == g_rank)
                        expected[em ? k : 0].insert({src + 1, i % 128, i / 128});
        ASSERT_EQ(total_rows, expected[0].size() + expected[1].size());
        std::vector<nv_bfloat16> received(total_rows * hidden);
        CUDA_ASSERT(cudaMemcpy(received.data(), recv.data, received.size() * 2, cudaMemcpyDeviceToHost));
        size_t offset = 0;
        for (int e = 0; e < (em ? 2 : 1); ++e) {
            if (em) {
                EXPECT_EQ(expert_counts[e], expected[e].size());
                EXPECT_EQ(expert_offsets[e], offset);
            }
            std::multiset<std::array<int, 3>> actual;
            for (size_t i = 0; i < expected[e].size(); ++i) {
                const auto* row = received.data() + (offset + i) * hidden;
                actual.insert({int(__bfloat162float(row[0])), int(__bfloat162float(row[1])), int(__bfloat162float(row[2]))});
            }
            EXPECT_EQ(actual, expected[e]);
            offset += expected[e].size();
        }
        const int n = tokens_for(g_rank, round, id);
        std::vector<nv_bfloat16> combined(capacity * hidden);
        CUDA_ASSERT(cudaMemcpy(combined.data(), output.data, combined.size() * 2, cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < hidden; ++h)
                errors += __bfloat162float(combined[i * hidden + h]) != topk * value_for(g_rank, i, h, round, id);
        EXPECT_EQ(errors, 0);
        if (backward) {
            std::vector<float> grads(capacity * topk);
            CUDA_ASSERT(cudaMemcpy(grads.data(), grad_weights.data, grads.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n * topk; ++i) EXPECT_EQ(grads[i], 0.5f);
        }
        unsigned char guards[512];
        CUDA_ASSERT(cudaMemcpy(guards, storage, 256, cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(guards + 256, static_cast<char*>(storage) + 256 + bytes, 256, cudaMemcpyDeviceToHost));
        for (auto v : guards) EXPECT_EQ(v, 0xA5);
    }
};

class HtChunkHandle : public ::testing::TestWithParam<int> {};

TEST_P(HtChunkHandle, MixedRanksReuseAndGraphs) {
    const bool em = GetParam() != 0, unfused = GetParam() == 2;
    ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
    cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    cfg.num_experts = 2 * g_nranks;
    cfg.num_topk = topk;
    cfg.max_dispatch_tokens_per_rank = capacity;
    cfg.max_recv_tokens_per_rank = capacity * g_nranks * topk;
    cfg.max_token_bytes = hidden * 2;
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &cfg));
    {
        State states[2];
        for (auto& s : states) s.init(group, em);
        for (int round = 0; round < 6; ++round) {
            SCOPED_TRACE(round);
            for (int id = 0; id < 2; ++id) states[id].update(round, id, unfused || !em);
            for (int id = 0; id < 2; ++id) {
                auto& s = states[id];
                for (bool backward : {false, true}) {
                    SCOPED_TRACE(backward);
                    s.run(em, backward);
                    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
                    s.check(round, id, em, backward);
                    cudaGraph_t graph;
                    cudaGraphExec_t exec;
                    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal));
                    s.run(em, backward);
                    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
                    CUDA_ASSERT(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                    for (int replay = 0; replay < 3; ++replay) {
                        CUDA_ASSERT(cudaGraphLaunch(exec, g_stream));
                        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
                        s.check(round, id, em, backward);
                    }
                    CUDA_ASSERT(cudaGraphExecDestroy(exec));
                    CUDA_ASSERT(cudaGraphDestroy(graph));
                }
            }
        }
    }
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

INSTANTIATE_TEST_SUITE_P(Layouts, HtChunkHandle, ::testing::Values(0, 1, 2));
} // namespace

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_ht_chunk_handle.uid")) return 0;
    int result = RUN_ALL_TESTS();
    ep_teardown();
    return result;
}
