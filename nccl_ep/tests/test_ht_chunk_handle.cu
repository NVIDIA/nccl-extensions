/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_common.h"
#include "ht_sm_policy.hpp"
#include <nccl_device.h>
#include <array>
#include <set>
#include <tuple>

namespace {
constexpr int kDefaultCapacity = 16384;
int capacity = kDefaultCapacity;
constexpr int hidden = 128;
constexpr int topk = 2;
enum class LayoutMode { Flat, ExpertFused, ExpertUnfused };
enum class Direction { Forward, Backward };

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
    if (round == 6) {
        return 0;
    }
    if (round < 3) return ((round + handle) % 2) ? capacity : 128;
    const std::array<int, 24> counts = {
        0, 1, 31, 32, 33, 512, 513, 1024, 1025, 2048, 2049, 2303,
        2304, 2305, 3071, 3072, 3073, 4096, 4097, 4608, 8192, 8193, 9216, capacity};
    return std::min(capacity, counts[(rank + (round - 3) * 4 + handle * 8) % counts.size()]);
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

// Buffers one dispatch/combine pass writes; a queued sequence gives each launch its own set.
struct Outputs {
    Tensor recv, recv_weights, recv_idx, output, grad_weights;
    void init(bool em) {
        const int recv_capacity = capacity * g_nranks * topk;
        recv.init(ncclBfloat16, recv_capacity * hidden * 2, recv_capacity, hidden);
        recv_weights.init(ncclFloat32, recv_capacity * topk * sizeof(float), recv_capacity, em ? 0 : topk);
        recv_idx.init(ncclInt64, recv_capacity * topk * sizeof(int64_t), recv_capacity, topk);
        output.init(ncclBfloat16, capacity * hidden * 2, capacity, hidden);
        grad_weights.init(ncclFloat32, capacity * topk * sizeof(float), capacity, topk);
    }
};

struct State {
    ncclEpHandle_t handle = nullptr;
    ncclEpTensor_t* mem = nullptr;
    void* storage = nullptr;
    size_t bytes = 0;
    Tensor idx, input, weights, total, counts, offsets;
    Outputs bufs;
    ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
    ~State() {
        if (handle) ncclEpHandleDestroy(handle);
        if (mem) ncclEpTensorDestroy(mem);
        if (storage) cudaFree(storage);
    }
    void init(ncclEpGroup_t group, bool em) {
        const auto mode = em ? NCCL_EP_LAYOUT_EXPERT_MAJOR : NCCL_EP_LAYOUT_FLAT;
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
        bufs.init(em);
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
        for (auto* t : {idx.t, input.t, weights.t, bufs.output.t, bufs.grad_weights.t}) {
            t->sizes[0] = n;
        }
        NCCL_ASSERT(ncclEpUpdateHandle(handle, idx.t, unfused ? &layout : nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }
    void relocate(ncclEpGroup_t group, LayoutMode mode) {
        void* snapshot = nullptr;
        ncclEpTensor_t* snapshot_mem = nullptr;
        CUDA_ASSERT(cudaMalloc(&snapshot, bytes + 512));
        CUDA_ASSERT(cudaMemcpy(snapshot, storage, bytes + 512, cudaMemcpyDeviceToDevice));
        NCCL_ASSERT(epTensorCreate(&snapshot_mem, 1, ncclUint8, static_cast<char*>(snapshot) + 256, bytes));
        ncclEpHandleState_t state = NCCL_EP_HANDLE_STATE_INIT;
        state.num_tokens = input.t->sizes[0];
        state.num_topk = topk;
        state.topk_idx_datatype = ncclInt64;
        ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
        cfg.dispatch_output_per_expert_alignment = 1;
        ncclEpHandle_t imported = nullptr;
        const ncclEpLayout_t layout = (mode == LayoutMode::Flat) ? NCCL_EP_LAYOUT_FLAT : NCCL_EP_LAYOUT_EXPERT_MAJOR;
        NCCL_ASSERT(ncclEpImportHandle(&imported, group, layout, &cfg, &state, snapshot_mem));
        NCCL_ASSERT(ncclEpHandleDestroy(handle));
        NCCL_ASSERT(ncclEpTensorDestroy(mem));
        CUDA_ASSERT(cudaFree(storage));
        handle = imported;
        mem = snapshot_mem;
        storage = snapshot;
    }
    void run(bool em, Direction direction, Outputs* outputs) {
        const bool backward = direction == Direction::Backward;
        Outputs& o = outputs ? *outputs : bufs;
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        in.tokens = input.t;
        in.topk_weights = backward ? nullptr : weights.t;
        out.tokens = o.recv.t;
        out.topk_weights = backward ? nullptr : o.recv_weights.t;
        if (!em && !backward) {
            out.topk_idx = o.recv_idx.t;
        }
        cfg.pass_direction = backward ? NCCL_EP_BWD_PASS : NCCL_EP_FWD_PASS;
        CUDA_ASSERT(cudaMemsetAsync(o.recv.data, 0xFF, capacity * g_nranks * topk * hidden * 2, g_stream));
        CUDA_ASSERT(cudaMemsetAsync(o.output.data, 0xFF, capacity * hidden * 2, g_stream));
        CUDA_ASSERT(cudaMemsetAsync(o.grad_weights.data, 0xFF, capacity * topk * sizeof(float), g_stream));
        NCCL_ASSERT(ncclEpDispatch(handle, &in, &out, &layout, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle, nullptr, g_stream));
        ncclEpCombineInputs_t ci = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t co = NCCL_EP_COMBINE_OUTPUTS_INIT;
        ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
        ci.tokens = o.recv.t;
        co.tokens = o.output.t;
        ccfg.pass_direction = cfg.pass_direction;
        if (backward) {
            ci.topk_weights = o.recv_weights.t;
            co.topk_weights = o.grad_weights.t;
        }
        NCCL_ASSERT(ncclEpCombine(handle, &ci, &co, &ccfg, g_stream));
    }
    void check(int round, int id, bool em, Direction direction, Outputs* outputs) {
        const bool backward = direction == Direction::Backward;
        Outputs& o = outputs ? *outputs : bufs;
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
        CUDA_ASSERT(cudaMemcpy(received.data(), o.recv.data, received.size() * 2, cudaMemcpyDeviceToHost));
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
        CUDA_ASSERT(cudaMemcpy(combined.data(), o.output.data, combined.size() * 2, cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < hidden; ++h)
                errors += __bfloat162float(combined[i * hidden + h]) != topk * value_for(g_rank, i, h, round, id);
        EXPECT_EQ(errors, 0);
        if (backward) {
            std::vector<float> grads(capacity * topk);
            CUDA_ASSERT(cudaMemcpy(grads.data(), o.grad_weights.data, grads.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n * topk; ++i) EXPECT_EQ(grads[i], 0.5f);
        }
        unsigned char guards[512];
        CUDA_ASSERT(cudaMemcpy(guards, storage, 256, cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(guards + 256, static_cast<char*>(storage) + 256 + bytes, 256, cudaMemcpyDeviceToHost));
        for (auto v : guards) EXPECT_EQ(v, 0xA5);
    }
};

// Runs two handles with distinct routing through an uneven forward/backward sequence, queued back
// to back eagerly and as graphs. An unbalanced sequence exposes a shared round counter that
// balanced pass counts would hide.
void run_uneven_passes(ncclEpGroup_t group, bool em, bool unfused) {
    constexpr Direction F = Direction::Forward;
    constexpr Direction B = Direction::Backward;
    struct Pass {
        int id;
        Direction direction;
    };
    const std::array<Pass, 12> seq = {{{0, F}, {1, F}, {0, F}, {1, F}, {0, B}, {1, B},
                                       {0, F}, {1, F}, {0, B}, {1, B}, {0, B}, {1, B}}};
    State s[2];
    for (auto& st : s) {
        st.init(group, em);
    }
    // Handle 0 sends full capacity, handle 1 a small count; routing stays fixed across replays.
    for (int id = 0; id < 2; ++id) {
        s[id].update(1, id, unfused || !em);
    }
    // Queue the whole sequence back to back on one stream, eagerly and as graphs, and check only
    // after the last launch. Every launch writes its own output set, so one that is skipped or
    // overwritten by a later launch leaves poisoned rows behind. A backward launch consumes the
    // received weights of its handle's latest forward launch.
    for (bool as_graph : {false, true}) {
        SCOPED_TRACE(as_graph ? "queued graphs" : "queued eager");
        std::vector<Outputs> outs(seq.size());
        std::vector<cudaGraph_t> queued_graphs(seq.size());
        std::vector<cudaGraphExec_t> queued_execs(seq.size());
        for (size_t k = 0; k < seq.size(); ++k) {
            outs[k].init(em);
            for (auto* t : {outs[k].output.t, outs[k].grad_weights.t}) {
                t->sizes[0] = tokens_for(g_rank, 1, seq[k].id);
            }
            if (!as_graph) {
                continue;
            }
            CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal));
            s[seq[k].id].run(em, seq[k].direction, &outs[k]);
            CUDA_ASSERT(cudaStreamEndCapture(g_stream, &queued_graphs[k]));
            CUDA_ASSERT(cudaGraphInstantiate(&queued_execs[k], queued_graphs[k], nullptr, nullptr, 0));
        }
        std::array<size_t, 2> last_forward = {0, 0};
        for (size_t k = 0; k < seq.size(); ++k) {
            const auto& p = seq[k];
            if (p.direction == F) {
                last_forward[p.id] = k;
            } else {
                CUDA_ASSERT(cudaMemcpyAsync(outs[k].recv_weights.data, outs[last_forward[p.id]].recv_weights.data,
                                            capacity * g_nranks * topk * topk * sizeof(float),
                                            cudaMemcpyDeviceToDevice, g_stream));
            }
            if (as_graph) {
                CUDA_ASSERT(cudaGraphLaunch(queued_execs[k], g_stream));
            } else {
                s[p.id].run(em, p.direction, &outs[k]);
            }
        }
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        for (size_t k = 0; k < seq.size(); ++k) {
            SCOPED_TRACE(::testing::Message() << "launch " << k << " id=" << seq[k].id
                         << " direction=" << static_cast<int>(seq[k].direction));
            s[seq[k].id].check(1, seq[k].id, em, seq[k].direction, &outs[k]);
            if (!as_graph) {
                continue;
            }
            CUDA_ASSERT(cudaGraphExecDestroy(queued_execs[k]));
            CUDA_ASSERT(cudaGraphDestroy(queued_graphs[k]));
        }
    }
}

class HtChunkHandle : public ::testing::TestWithParam<std::tuple<LayoutMode, int>> {};

TEST_P(HtChunkHandle, MixedRanksReuseAndGraphs) {
    const LayoutMode mode = std::get<0>(GetParam());
    capacity = std::get<1>(GetParam());
    const bool em = mode != LayoutMode::Flat;
    const bool unfused = mode == LayoutMode::ExpertUnfused;
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
        for (int round = 0; round < 7; ++round) {
            SCOPED_TRACE(round);
            for (int id = 0; id < 2; ++id) states[id].update(round, id, unfused || !em);
            for (int id = 0; id < 2; ++id) {
                auto& s = states[id];
                for (Direction direction : {Direction::Forward, Direction::Backward}) {
                    SCOPED_TRACE(static_cast<int>(direction));
                    s.run(em, direction, nullptr);
                    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
                    s.check(round, id, em, direction, nullptr);
                    cudaGraph_t graph;
                    cudaGraphExec_t exec;
                    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal));
                    s.run(em, direction, nullptr);
                    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
                    CUDA_ASSERT(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                    for (int replay = 0; replay < 3; ++replay) {
                        CUDA_ASSERT(cudaGraphLaunch(exec, g_stream));
                        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
                        s.check(round, id, em, direction, nullptr);
                    }
                    CUDA_ASSERT(cudaGraphExecDestroy(exec));
                    CUDA_ASSERT(cudaGraphDestroy(graph));
                    if (direction == Direction::Forward && (round == 2 || round == 4)) {
                        s.relocate(group, mode);
                    }
                }
            }
        }
    }
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

INSTANTIATE_TEST_SUITE_P(
    Layouts, HtChunkHandle,
    ::testing::Combine(
        ::testing::Values(LayoutMode::Flat, LayoutMode::ExpertFused, LayoutMode::ExpertUnfused),
        ::testing::Values(193, 2304, 3072, 4096, 4608, 6144, 8191, 8192, 9216, 16384)),
    [](const ::testing::TestParamInfo<HtChunkHandle::ParamType>& info) {
        std::string mode;
        switch (std::get<0>(info.param)) {
            case LayoutMode::Flat:
                mode = "Flat";
                break;
            case LayoutMode::ExpertFused:
                mode = "ExpertFused";
                break;
            case LayoutMode::ExpertUnfused:
                mode = "ExpertUnfused";
                break;
        }
        return mode + "Capacity" + std::to_string(std::get<1>(info.param));
    });

// One capacity per layout: the pass sequence does not depend on it.
class HtChunkHandlePasses : public ::testing::TestWithParam<LayoutMode> {};

// On several LSA teams, unequal forward/backward grids split the tail signals and round counters;
// a larger backward grid also raises the LSA barrier count and the combine session index. The
// flat-only case keeps the expert-major grids equal and differs only in the flat grid.
TEST_P(HtChunkHandlePasses, UnevenPassesUnequalGrids) {
    capacity = kDefaultCapacity;
    const bool em = GetParam() != LayoutMode::Flat;
    const bool unfused = GetParam() == LayoutMode::ExpertUnfused;
    if (em && !unfused && ncclTeamLsa(g_comm).nRanks != g_nranks) {
        GTEST_SKIP() << "with several LSA teams dispatch leaves layout counters to ncclEpUpdateHandle; "
                        "ExpertUnfused covers expert-major there";
    }
    struct BudgetCase {
        unsigned int dispatch_fwd, combine_fwd, dispatch_bwd, combine_bwd;
        ncclEpHtFlow_t flow;
        bool flat_only;
    };
    std::vector<BudgetCase> cases = {{32, 32, 12, 48, NCCL_EP_HT_FLOW_PUSH_PULL, false},
                                     {32, 32, 48, 12, NCCL_EP_HT_FLOW_PUSH_PULL, false}};
    if (ncclTeamLsa(g_comm).nRanks == g_nranks) {
        cases.push_back({32, 32, 48, 12, NCCL_EP_HT_FLOW_PULL_PUSH, false});
    }
    if (!em) {
        // Backward expert-major grid at the architecture's AUTO default, so only the flat grid differs.
        int device = 0;
        cudaDeviceProp prop{};
        CUDA_ASSERT(cudaGetDevice(&device));
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));
        const unsigned int device_sm = static_cast<unsigned int>(prop.major * 10 + prop.minor);
        const unsigned int em_default = nccl_ep::ht_default_comm_sms(device_sm, /*expert_major=*/true);
        cases.push_back({0, 0, em_default, 0, NCCL_EP_HT_FLOW_AUTO, true});
    }
    for (const auto& c : cases) {
        SCOPED_TRACE(::testing::Message() << "dispatch_bwd=" << c.dispatch_bwd
                     << " combine_bwd=" << c.combine_bwd << " flow=" << c.flow);
        ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
        cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        cfg.num_experts = 2 * g_nranks;
        cfg.num_topk = topk;
        cfg.max_dispatch_tokens_per_rank = capacity;
        cfg.max_recv_tokens_per_rank = capacity * g_nranks * topk;
        cfg.max_token_bytes = hidden * 2;
        cfg.ht_flow = c.flow;
        cfg.dispatch_num_sms = c.dispatch_fwd;
        cfg.combine_num_sms = c.combine_fwd;
        cfg.dispatch_bwd_num_sms = c.dispatch_bwd;
        cfg.combine_bwd_num_sms = c.combine_bwd;
        ncclEpGroup_t group = nullptr;
        NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &cfg));
        if (c.flat_only) {
            State probe;
            probe.init(group, false);
            unsigned int em_fwd = 0, em_bwd = 0, flat_fwd = 0, flat_bwd = 0;
            ncclEpGroup_test_getSmBudgets(group, NCCL_EP_FWD_PASS, &em_fwd, nullptr);
            ncclEpGroup_test_getSmBudgets(group, NCCL_EP_BWD_PASS, &em_bwd, nullptr);
            ncclEpHandle_test_getSmBudgets(probe.handle, NCCL_EP_FWD_PASS, &flat_fwd, nullptr);
            ncclEpHandle_test_getSmBudgets(probe.handle, NCCL_EP_BWD_PASS, &flat_bwd, nullptr);
            ASSERT_EQ(em_fwd, em_bwd);
            ASSERT_NE(flat_fwd, flat_bwd);
        }
        run_uneven_passes(group, em, unfused);
        NCCL_ASSERT(ncclEpGroupDestroy(group));
    }
}

INSTANTIATE_TEST_SUITE_P(
    Layouts, HtChunkHandlePasses,
    ::testing::Values(LayoutMode::Flat, LayoutMode::ExpertFused, LayoutMode::ExpertUnfused));
} // namespace

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_ht_chunk_handle.uid")) return 0;
    int result = RUN_ALL_TESTS();
    ep_teardown();
    return result;
}
