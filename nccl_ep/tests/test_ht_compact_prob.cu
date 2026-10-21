/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Public-API E2E tests for compact weights + IDs. Independent host routing,
 * weight and gradient expectations cover FWD dispatch and BWD combine.
 * The runner selects count, scan, LOCAL_DUP, NVLINK_DUP or pull-push mode.
 * The custom bootstrap avoids test_common's legacy E8/capacity32 groups.
 */
#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include "../device/ht_ep_adapter.cuh"
#include "../ht_compact_prob.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <strings.h>
#include <tuple>

namespace {
constexpr int kTokens = 4;
constexpr int kWidth = 16;
constexpr int kAlignment = 8;
constexpr int kTokenIdRadix = 256;

// Small integer fields stay exact in BF16, including the 513-token case.
float token_value(int rank, int token, int column) {
    switch (column % 3) {
        case 0:
            return static_cast<float>(rank + 1);
        case 1:
            return static_cast<float>(token / kTokenIdRadix);
        default:
            return static_cast<float>(token % kTokenIdRadix);
    }
}

bool enabled(const char* name) {
    return env_flag_active(name);
}

// Fused count mode applies to both EM local-permute and Flat handles (see
// dispatch_push_count_capable).
bool fused_count_mode() {
    return !enabled("NCCL_EP_HT_EM_LOCAL_DUP") && !enabled("NCCL_EP_HT_EM_NVLINK_DUP") &&
           !enabled("NCCL_EP_HT_AG_SCAN_MODE") &&
           !enabled("NCCL_EP_HT_COUNT_UNFUSED");
}

template <typename T> struct DeviceBuffer {
    T* p = nullptr;
    size_t size = 0;
    ~DeviceBuffer() { if (p) cudaFree(p); }
    void allocate(size_t n) {
        size = n;
        CUDA_ASSERT(cudaMalloc(&p, n * sizeof(T)));
        CUDA_ASSERT(cudaMemset(p, 0, n * sizeof(T)));
    }
    void upload(const std::vector<T>& v) {
        ASSERT_LE(v.size(), size);
        if (!v.empty()) CUDA_ASSERT(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& v, size_t n) const {
        ASSERT_LE(n, size);
        v.resize(n);
        if (n) CUDA_ASSERT(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// Positions 0 and 3 select a same-owner pair in reversed expert order.
// Pair-specific local IDs keep all IDs distinct even at EP4/K10, where owners wrap.
int64_t route(int source, int token, int k, int topk, int experts, int epoch) {
    static const int order8[8] = {1, 3, 5, 0, 2, 4, 7, 6};
    static const int order10[10] = {1, 3, 5, 0, 2, 4, 7, 9, 6, 8};
    // Epoch 8 keeps epoch 0's exact selected-expert set/bitmap but reverses
    // caller top-k order, including padding. The saved position map must change.
    if (epoch & 8) k = topk - 1 - k;
    epoch &= 7;
    if (token == 3 || (token == 1 && k == topk - 1)) return -1;
    // Use only valid IDs and -1, as required by the public routing API.
    if ((epoch & 1) && token == 0 && k == topk - 1) return -1;
    int p = topk == 8 ? order8[k] : order10[k];
    // E8/K8 intentionally selects the automatic dense fallback.
    if (experts == 8) return (p + source + token + epoch) % experts;
    int owner = (source + token + epoch + p / 2) % g_nranks;
    return static_cast<int64_t>(owner * (experts / g_nranks) + p);
}

float source_weight(int source, int token, int k, int epoch, int repeat) {
    return static_cast<float>(1 + source * 128 + token * 16 + k + epoch * 8192 + repeat * 65536) / 128.0f;
}

float independent_gradient(int source, int token, int expert, int epoch) {
    return -static_cast<float>(7 + source * 2048 + token * 512 + expert + epoch * 131072) / 64.0f;
}

struct Snapshot {
    int total = 0;
    std::vector<nv_bfloat16> recv_x, token_gradients, combined_x;
    std::vector<float> recv_w, gradients, combined_w;
    std::vector<int64_t> recv_idx;
};

struct State {
    ncclEpHandle_t handle = nullptr;
    std::vector<ncclEpTensor_t*> tensors;
    ncclEpTensor_t *ids = nullptr, *x = nullptr, *w = nullptr, *rx = nullptr,
                   *rw = nullptr, *ri = nullptr, *cx = nullptr, *cw = nullptr;
    ncclEpTensor_t *counts_tensor = nullptr, *offsets_tensor = nullptr, *total_tensor = nullptr;
    DeviceBuffer<int32_t> ids32;
    DeviceBuffer<int64_t> ids64, recv_ids, counts, offsets, total;
    DeviceBuffer<nv_bfloat16> tokens, recv_tokens, combined_tokens;
    DeviceBuffer<float> weights, recv_weights, combined_weights;
    ncclEpLayoutInfo_t info = NCCL_EP_LAYOUT_INFO_INIT;
    // Hand info to UpdateHandle even in fused count mode (forces the unfused exchange, so
    // the layout tensors are published before dispatch).
    bool layout_info_at_update = false;
    bool em = false, use_i64 = false, topk_order = false;
    int topk = 0, experts = 0, capacity = 0, ntokens = 0, epoch = 0;
    int token_capacity = kTokens;
    std::vector<int> source_tokens;
    Snapshot output;

    ~State() {
        if (handle) ncclEpHandleDestroy(handle);
        for (auto* t : tensors) ncclEpTensorDestroy(t);
    }

    template <typename... Dims>
    void tensor(ncclEpTensor_t** result, unsigned ndim, ncclDataType_t dtype, void* ptr, Dims... dims) {
        NCCL_ASSERT(epTensorCreate(result, ndim, dtype, ptr, dims...));
        tensors.push_back(*result);
    }

    void initialize(ncclEpGroup_t group, bool expert_major, bool int64_ids, int k, int e, int cap,
                    int alignment = kAlignment, bool guard_buffers = false) {
        ASSERT_LE(g_nranks, kTokenIdRadix);
        ASSERT_LE(token_capacity, kTokenIdRadix * kTokenIdRadix);
        em = expert_major; use_i64 = int64_ids; topk = k; experts = e; capacity = cap;
        ASSERT_NO_FATAL_FAILURE(ids32.allocate(token_capacity * topk));
        ASSERT_NO_FATAL_FAILURE(ids64.allocate(token_capacity * topk));
        ASSERT_NO_FATAL_FAILURE(tokens.allocate(token_capacity * kWidth));
        ASSERT_NO_FATAL_FAILURE(weights.allocate(token_capacity * topk));
        ASSERT_NO_FATAL_FAILURE(recv_tokens.allocate(capacity * kWidth + (guard_buffers ? 16 : 0)));
        ASSERT_NO_FATAL_FAILURE(recv_weights.allocate(capacity * (em ? 1 : topk) + (guard_buffers ? 4 : 0)));
        ASSERT_NO_FATAL_FAILURE(recv_ids.allocate(capacity * topk));
        ASSERT_NO_FATAL_FAILURE(combined_tokens.allocate(token_capacity * kWidth));
        ASSERT_NO_FATAL_FAILURE(combined_weights.allocate(token_capacity * topk + 4));
        ASSERT_NO_FATAL_FAILURE(counts.allocate(experts / g_nranks));
        ASSERT_NO_FATAL_FAILURE(offsets.allocate(experts / g_nranks));
        ASSERT_NO_FATAL_FAILURE(total.allocate(1));
        void* ids_ptr = use_i64 ? static_cast<void*>(ids64.p) : static_cast<void*>(ids32.p);
        ASSERT_NO_FATAL_FAILURE(tensor(&ids, 2, use_i64 ? ncclInt64 : ncclInt32, ids_ptr, token_capacity, topk));
        ASSERT_NO_FATAL_FAILURE(tensor(&x, 2, ncclBfloat16, tokens.p, token_capacity, kWidth));
        ASSERT_NO_FATAL_FAILURE(tensor(&w, 2, ncclFloat32, weights.p, token_capacity, topk));
        ASSERT_NO_FATAL_FAILURE(tensor(&rx, 2, ncclBfloat16, recv_tokens.p, capacity, kWidth));
        if (em) {
            ASSERT_NO_FATAL_FAILURE(tensor(&rw, 1, ncclFloat32, recv_weights.p, capacity));
            ASSERT_NO_FATAL_FAILURE(tensor(&counts_tensor, 1, ncclInt64, counts.p, experts / g_nranks));
            ASSERT_NO_FATAL_FAILURE(tensor(&offsets_tensor, 1, ncclInt64, offsets.p, experts / g_nranks));
            info.expert_counters = counts_tensor;
            info.expert_offsets = offsets_tensor;
        } else {
            ASSERT_NO_FATAL_FAILURE(tensor(&rw, 2, ncclFloat32, recv_weights.p, capacity, topk));
            ASSERT_NO_FATAL_FAILURE(tensor(&ri, 2, ncclInt64, recv_ids.p, capacity, topk));
        }
        ASSERT_NO_FATAL_FAILURE(tensor(&cx, 2, ncclBfloat16, combined_tokens.p, token_capacity, kWidth));
        ASSERT_NO_FATAL_FAILURE(tensor(&cw, 2, ncclFloat32, combined_weights.p, token_capacity, topk));
        ASSERT_NO_FATAL_FAILURE(tensor(&total_tensor, 1, ncclInt64, total.p, 1));
        info.recv_total_counter = total_tensor;
        info.recv_topk_idx_kind = NCCL_EP_EXPERT_ID_LOCAL;
        ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
        cfg.dispatch_output_per_expert_alignment = em ? alignment : 0;
        NCCL_ASSERT(ncclEpInitHandle(&handle, group, em ? NCCL_EP_LAYOUT_EXPERT_MAJOR : NCCL_EP_LAYOUT_FLAT,
                                   &cfg, topk, nullptr));
        const int compact_floats = ((6 * topk + 15) & ~15) / 4;
        const int dense_floats = (experts / g_nranks) * ncclEpHandle_test_getNRanksPerNode(handle);
        const bool inline_wire = compact_floats < dense_floats && !enabled("NCCL_EP_HT_PULL_PUSH");
        ASSERT_EQ(ncclEpHandle_test_getCompactProbWireFloats(handle), inline_wire ? compact_floats : 0)
            << "Compact is automatic when smaller; pull-push keeps its existing sparse-ID path";
        const char* order = std::getenv("NCCL_EP_HT_RECV_TOPK_ORDER");
        topk_order = inline_wire && !(order && strcasecmp(order, "expert-ascend") == 0);
    }

    void update(int n, int generation, bool empty_last_rank) {
        source_tokens.assign(g_nranks, n);
        if (empty_last_rank) {
            source_tokens.back() = 0;
        }
        n = source_tokens[g_rank];
        ntokens = n; epoch = generation;
        ids->sizes[0] = x->sizes[0] = w->sizes[0] = cx->sizes[0] = cw->sizes[0] = n;
        // Empty outputs may have no storage, even though the allocation is
        // retained here for the next nonempty update and guard checks.
        cx->data = n ? combined_tokens.p : nullptr;
        cw->data = n ? combined_weights.p : nullptr;
        std::vector<int64_t> host64(n * topk);
        std::vector<int32_t> host32(n * topk);
        for (int t = 0; t < n; ++t) for (int k = 0; k < topk; ++k) {
            host64[t * topk + k] = route(g_rank, t, k, topk, experts, epoch);
            host32[t * topk + k] = static_cast<int32_t>(host64[t * topk + k]);
        }
        if (use_i64) { ASSERT_NO_FATAL_FAILURE(ids64.upload(host64)); }
        else { ASSERT_NO_FATAL_FAILURE(ids32.upload(host32)); }
        info.recv_topk_idx_kind = (epoch & 1) ? NCCL_EP_EXPERT_ID_GLOBAL : NCCL_EP_EXPERT_ID_LOCAL;
        // Passing layout info to UpdateHandle forces count provisioning there.
        // Omit it only for a single-team fused mode, to test dispatch-time map
        // build. Multi-team scan paths must publish their layout at UpdateHandle.
        const bool fused = fused_count_mode() && !layout_info_at_update &&
                           ncclEpHandle_test_getNRanksPerNode(handle) == g_nranks;
        NCCL_ASSERT(ncclEpUpdateHandle(handle, ids, fused ? nullptr : &info, g_stream));
        EXPECT_EQ(ht_count_fused_active(handle), fused);
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    void dispatch(int repeat, bool graph_replay) {
        std::vector<nv_bfloat16> ht(ntokens * kWidth);
        std::vector<float> hw(ntokens * topk);
        for (int t = 0; t < ntokens; ++t) {
            for (int d = 0; d < kWidth; ++d) {
                ht[t * kWidth + d] = __float2bfloat16(token_value(g_rank, t, d));
            }
            for (int k = 0; k < topk; ++k) hw[t * topk + k] = source_weight(g_rank, t, k, epoch, repeat);
        }
        ASSERT_NO_FATAL_FAILURE(tokens.upload(ht));
        ASSERT_NO_FATAL_FAILURE(weights.upload(hw));
        CUDA_ASSERT(cudaMemset(recv_tokens.p, 0, recv_tokens.size * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(recv_weights.p, 0xA5, recv_weights.size * sizeof(float)));
        CUDA_ASSERT(cudaMemset(recv_ids.p, 0xA5, recv_ids.size * sizeof(int64_t)));
        ncclEpDispatchInputs_t input = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t result = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        input.tokens = x; input.topk_weights = w;
        result.tokens = rx; result.topk_weights = rw; result.topk_idx = ri;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        auto launch = [&]() {
            NCCL_ASSERT(ncclEpDispatch(handle, &input, &result, &info, &cfg, g_stream));
            NCCL_ASSERT(ncclEpComplete(handle, nullptr, g_stream));
        };
        if (graph_replay) {
            // The caller warmed the handle before capture. Replay twice so an
            // order-dependent map cannot accidentally pass only its first use.
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
            ASSERT_NO_FATAL_FAILURE(launch());
            CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
            CUDA_ASSERT(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                CUDA_ASSERT(cudaGraphLaunch(executable, g_stream));
                CUDA_ASSERT(cudaStreamSynchronize(g_stream));
            }
            CUDA_ASSERT(cudaGraphExecDestroy(executable));
            CUDA_ASSERT(cudaGraphDestroy(graph));
        } else {
            ASSERT_NO_FATAL_FAILURE(launch());
        }
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        std::vector<int64_t> host_total;
        ASSERT_NO_FATAL_FAILURE(total.download(host_total, 1));
        ASSERT_GE(host_total[0], 0); ASSERT_LE(host_total[0], capacity);
        output.total = static_cast<int>(host_total[0]);
        ASSERT_NO_FATAL_FAILURE(recv_tokens.download(output.recv_x, output.total * kWidth));
        ASSERT_NO_FATAL_FAILURE(recv_weights.download(output.recv_w, output.total * (em ? 1 : topk)));
        if (!em) ASSERT_NO_FATAL_FAILURE(recv_ids.download(output.recv_idx, output.total * topk));
        ASSERT_NO_FATAL_FAILURE(check_forward_and_make_gradients(repeat));
    }

    void check_forward_and_make_gradients(int repeat) {
        std::vector<int64_t> hc, ho;
        int epr = experts / g_nranks;
        if (em) {
            ASSERT_NO_FATAL_FAILURE(counts.download(hc, epr));
            ASSERT_NO_FATAL_FAILURE(offsets.download(ho, epr));
        }
        // BWD receives the full fixed-capacity tensor, not only actual rows.
        // Leave unused capacity, EM alignment pads and FLAT tail columns
        // nonzero so stale/uninitialized saved map entries cannot false-pass.
        output.gradients.assign(recv_weights.size, 12345.0f);
        std::vector<int> seen(g_nranks * token_capacity * topk, 0);
        output.token_gradients.assign(recv_tokens.size, __float2bfloat16(0));
        for (int s = 0; s < output.total; ++s) {
            SCOPED_TRACE(::testing::Message() << "rank=" << g_rank << " epoch=" << epoch << " recv_slot=" << s);
            const float source_rank = __bfloat162float(output.recv_x[s * kWidth]);
            if (source_rank == 0) {
                ASSERT_TRUE(em) << "FLAT has no internal padding in its active receive range";
                for (int d = 0; d < kWidth; ++d) EXPECT_EQ(__bfloat162float(output.recv_x[s * kWidth + d]), 0.0f);
                EXPECT_EQ(bits(output.recv_w[s]), bits(0.0f));
                continue;
            }
            const float token_high = __bfloat162float(output.recv_x[s * kWidth + 1]);
            const float token_low = __bfloat162float(output.recv_x[s * kWidth + 2]);
            ASSERT_GE(source_rank, 1.0f);
            ASSERT_LE(source_rank, static_cast<float>(g_nranks));
            ASSERT_GE(token_high, 0.0f);
            ASSERT_LT(token_high, static_cast<float>(kTokenIdRadix));
            ASSERT_GE(token_low, 0.0f);
            ASSERT_LT(token_low, static_cast<float>(kTokenIdRadix));
            const int source = static_cast<int>(source_rank) - 1;
            const int token = static_cast<int>(token_high) * kTokenIdRadix + static_cast<int>(token_low);
            ASSERT_LT(token, source_tokens[source]);
            for (int d = 0; d < kWidth; ++d) {
                EXPECT_EQ(__bfloat162float(output.recv_x[s * kWidth + d]), token_value(source, token, d));
                output.token_gradients[s * kWidth + d] = __float2bfloat16(1);
            }
            std::vector<std::pair<int, int>> local;
            for (int k = 0; k < topk; ++k) {
                int64_t e = route(source, token, k, topk, experts, epoch);
                if (e >= g_rank * epr && e < (g_rank + 1) * epr) local.emplace_back(static_cast<int>(e), k);
            }
            // Compact keeps top-k encounter order in a packed prefix by default.
            // The explicit sorted option and dense fallback keep ascending order.
            if (!topk_order) {
                std::sort(local.begin(), local.end());
            }
            ASSERT_FALSE(local.empty());
            if (em) {
                int local_e = -1;
                for (int e = 0; e < epr; ++e) if (s >= ho[e] && s < ho[e] + hc[e]) { local_e = e; break; }
                ASSERT_GE(local_e, 0);
                auto hit = std::find_if(local.begin(), local.end(), [&](const std::pair<int, int>& v) {
                    return v.first == g_rank * epr + local_e;
                });
                ASSERT_NE(hit, local.end());
                int k = hit->second;
                EXPECT_EQ(bits(output.recv_w[s]), bits(source_weight(source, token, k, epoch, repeat)));
                output.gradients[s] = independent_gradient(source, token, hit->first, epoch);
                ++seen[(source * token_capacity + token) * topk + k];
            } else {
                for (int j = 0; j < topk; ++j) {
                    if (j >= static_cast<int>(local.size())) {
                        EXPECT_EQ(output.recv_idx[s * topk + j], -1);
                        EXPECT_EQ(bits(output.recv_w[s * topk + j]), bits(0.0f));
                        continue;
                    }
                    int e = local[j].first, k = local[j].second;
                    int expected_id = info.recv_topk_idx_kind == NCCL_EP_EXPERT_ID_GLOBAL ? e : e - g_rank * epr;
                    EXPECT_EQ(output.recv_idx[s * topk + j], expected_id);
                    EXPECT_EQ(bits(output.recv_w[s * topk + j]), bits(source_weight(source, token, k, epoch, repeat)));
                    output.gradients[s * topk + j] = independent_gradient(source, token, e, epoch);
                    ++seen[(source * token_capacity + token) * topk + k];
                }
            }
        }
        // Prove no expert contribution was silently omitted or duplicated.
        for (int r = 0; r < g_nranks; ++r) {
            for (int t = 0; t < source_tokens[r]; ++t) {
                for (int k = 0; k < topk; ++k) {
                    int64_t e = route(r, t, k, topk, experts, epoch);
                    int expected = e >= g_rank * epr && e < (g_rank + 1) * epr ? 1 : 0;
                    EXPECT_EQ(seen[(r * token_capacity + t) * topk + k], expected)
                        << "source=" << r << " token=" << t << " k=" << k;
                }
            }
        }
    }

    void backward() {
        ASSERT_NO_FATAL_FAILURE(recv_weights.upload(output.gradients));
        ASSERT_NO_FATAL_FAILURE(recv_tokens.upload(output.token_gradients));
        CUDA_ASSERT(cudaMemset(combined_weights.p, 0xA5, combined_weights.size * sizeof(float)));
        CUDA_ASSERT(cudaMemset(combined_tokens.p, 0, combined_tokens.size * sizeof(nv_bfloat16)));
        ncclEpCombineInputs_t input = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t result = NCCL_EP_COMBINE_OUTPUTS_INIT;
        input.tokens = rx; input.topk_weights = rw;
        result.tokens = cx; result.topk_weights = cw;
        ncclEpCombineConfig_t cfg = NCCL_EP_COMBINE_CONFIG_INIT;
        cfg.pass_direction = NCCL_EP_BWD_PASS;
        NCCL_ASSERT(ncclEpCombine(handle, &input, &result, &cfg, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        ASSERT_NO_FATAL_FAILURE(combined_tokens.download(output.combined_x, ntokens * kWidth));
        ASSERT_NO_FATAL_FAILURE(combined_weights.download(output.combined_w, ntokens * topk));
        // The fused output store must honor K (including K10's unaligned row
        // width) and the live token count; neither padding nor unused capacity
        // belongs to the caller output.
        std::vector<float> guarded_weights;
        ASSERT_NO_FATAL_FAILURE(combined_weights.download(guarded_weights, combined_weights.size));
        for (size_t i = static_cast<size_t>(ntokens) * topk; i < guarded_weights.size(); ++i) {
            EXPECT_EQ(bits(guarded_weights[i]), 0xA5A5A5A5u) << "output guard column=" << i;
        }
        for (int t = 0; t < ntokens; ++t) for (int k = 0; k < topk; ++k) {
            int64_t e = route(g_rank, t, k, topk, experts, epoch);
            float expected = e >= 0 && e < experts ? independent_gradient(g_rank, t, static_cast<int>(e), epoch) : 0.0f;
            EXPECT_EQ(bits(output.combined_w[t * topk + k]), bits(expected))
                << "independent BWD oracle: token=" << t << " k=" << k << " expert=" << e;
        }
        // FLAT receives one token contribution per owning rank; EM receives
        // one per selected expert. Ones make the BF16 reduction exact.
        for (int t = 0; t < ntokens; ++t) {
            std::vector<bool> owners(g_nranks, false);
            int contributions = 0;
            for (int k = 0; k < topk; ++k) {
                const int64_t e = route(g_rank, t, k, topk, experts, epoch);
                if (e < 0) continue;
                const int owner = static_cast<int>(e) / (experts / g_nranks);
                if (em || !owners[owner]) ++contributions;
                owners[owner] = true;
            }
            for (int d = 0; d < kWidth; ++d)
                EXPECT_EQ(__bfloat162float(output.combined_x[t * kWidth + d]), static_cast<float>(contributions));
        }
    }
};

class HtCompactProbTest : public ::testing::Test {
protected:
    ncclEpGroup_t group_ = nullptr;

    void TearDown() override {
        cudaStreamSynchronize(g_stream);
        if (group_) ncclEpGroupDestroy(group_);
    }

    void create_group(int experts, int topk, int capacity, bool drop, int max_tokens, ncclEpGroup_t* output) {
        ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
        cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        cfg.num_experts = experts;
        cfg.num_topk = topk;
        cfg.max_dispatch_tokens_per_rank = max_tokens;
        cfg.max_recv_tokens_per_rank = capacity;
        if (max_tokens > kTokens) {
            cfg.max_num_sms = 8;
        }
        cfg.max_token_bytes = kWidth * sizeof(nv_bfloat16);
        cfg.rdma_buffer_size = cfg.num_qp_per_rank = cfg.num_channels = NCCL_EP_AUTO;
        if (drop) {
            cfg.overflow_policy = NCCL_EP_OVERFLOW_DROP;
        }
        NCCL_ASSERT(ncclEpCreateGroup(output, g_comm, &cfg));
    }

    void run(bool em, bool int64_ids, int topk, int experts = 0) {
        if (!experts) experts = topk == 8 ? 256 : 512;
        ASSERT_EQ(experts % g_nranks, 0);
        // Cross live chunk boundaries while retaining empty capacity tails.
        const bool overlap_case = !em && int64_ids && topk == 10;
        const int live_tokens = overlap_case ? 513 : kTokens;
        const int max_tokens = overlap_case ? 2048 : kTokens;
        const int cap = std::max(max_tokens,
            (g_nranks * live_tokens * topk + (experts / g_nranks) * (kAlignment - 1) + 127) & ~63);
        ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, cap, false, max_tokens, &group_));
        State a, b;
        a.token_capacity = b.token_capacity = live_tokens;
        ASSERT_NO_FATAL_FAILURE(a.initialize(group_, em, int64_ids, topk, experts, cap));
        ASSERT_NO_FATAL_FAILURE(b.initialize(group_, em, int64_ids, topk, experts, cap));
        ASSERT_NO_FATAL_FAILURE(a.update(live_tokens, 0, false));
        ASSERT_NO_FATAL_FAILURE(b.update(live_tokens, 7, false));

        // A's private receive outputs and saved positions survive B using
        // the same group's staging buffers. Every BWD follows that handle's FWD.
        ASSERT_NO_FATAL_FAILURE(a.dispatch(0, false));
        ASSERT_NO_FATAL_FAILURE(b.dispatch(0, false));
        ASSERT_NO_FATAL_FAILURE(b.dispatch(1, false));
        ASSERT_NO_FATAL_FAILURE(a.backward());
        ASSERT_NO_FATAL_FAILURE(b.backward());

        // Reuse the handle with fewer and then more tokens; the last rank has
        // no source tokens in the smaller step but still receives peer rows.
        // Change routing and LOCAL/GLOBAL output IDs at the same time.
        for (int epoch = 1; epoch <= 2; ++epoch) {
            ASSERT_NO_FATAL_FAILURE(a.update(epoch == 1 ? 2 : live_tokens, epoch, /*empty_last_rank=*/epoch == 1));
            ASSERT_NO_FATAL_FAILURE(a.dispatch(epoch, false));
            ASSERT_NO_FATAL_FAILURE(a.backward());
        }
        // Reverse caller K order without changing its selected-expert bitmap.
        // A bitmap-only/stale position map must not pass this sequence.
        for (int generation : {0, 8}) {
            ASSERT_NO_FATAL_FAILURE(a.update(live_tokens, generation, false));
            ASSERT_NO_FATAL_FAILURE(a.dispatch(3, false));
            ASSERT_NO_FATAL_FAILURE(b.dispatch(3, false));
            ASSERT_NO_FATAL_FAILURE(a.backward());
        }
    }
};

TEST_F(HtCompactProbTest, RejectsInvalidFlatMetadata) {
    constexpr int topk = 8, experts = 256;
    const int capacity = g_nranks * kTokens;
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity, kAlignment, false));
    ASSERT_NO_FATAL_FAILURE(s.update(kTokens, 0, false));
    ncclEpTensor_t* invalid_counts = nullptr;
    ASSERT_NO_FATAL_FAILURE(s.tensor(&invalid_counts, 1, ncclInt64, s.counts.p, experts / g_nranks));
    ncclEpLayoutInfo_t info = NCCL_EP_LAYOUT_INFO_INIT;
    info.expert_counters = invalid_counts;
    EXPECT_EQ(ncclEpUpdateHandle(s.handle, s.ids, &info, g_stream), ncclInvalidArgument);

    ncclEpDispatchInputs_t input = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t output = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    input.tokens = s.x;
    input.topk_weights = s.w;
    output.tokens = s.rx;
    output.topk_weights = s.rw;
    output.topk_idx = s.ri;
    s.ri->datatype = ncclInt32;
    EXPECT_EQ(ncclEpDispatch(s.handle, &input, &output, nullptr, &config, g_stream), ncclInvalidArgument);
    s.ri->datatype = ncclInt64;
    s.ri->sizes[1] = topk - 1;
    EXPECT_EQ(ncclEpDispatch(s.handle, &input, &output, nullptr, &config, g_stream), ncclInvalidArgument);
    s.ri->sizes[1] = topk;
    ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
    ASSERT_NO_FATAL_FAILURE(s.backward());
}

TEST_F(HtCompactProbTest, RejectsBackwardWidthMismatch) {
    constexpr int topk = 8, experts = 256;
    const int capacity = g_nranks * kTokens;
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity, kAlignment, false));
    ASSERT_NO_FATAL_FAILURE(s.update(kTokens, 0, false));
    ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
    ncclEpCombineInputs_t input = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t output = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t config = NCCL_EP_COMBINE_CONFIG_INIT;
    config.pass_direction = NCCL_EP_BWD_PASS;
    input.tokens = s.rx;
    input.topk_weights = s.rw;
    output.tokens = s.cx;
    output.topk_weights = s.cw;
    s.rw->sizes[1] = topk - 1;
    EXPECT_EQ(ncclEpCombine(s.handle, &input, &output, &config, g_stream), ncclInvalidArgument);
    s.rw->sizes[1] = topk;
    s.cw->sizes[1] = topk - 1;
    EXPECT_EQ(ncclEpCombine(s.handle, &input, &output, &config, g_stream), ncclInvalidArgument);
    s.cw->sizes[1] = topk;
    ASSERT_NO_FATAL_FAILURE(s.backward());
}

TEST_F(HtCompactProbTest, FlatFusedMapsRequireMatchingForward) {
    if (!fused_count_mode()) {
        GTEST_SKIP() << "fused metadata only";
    }
    constexpr int topk = 8, experts = 256;
    const int capacity = g_nranks * kTokens;
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity, kAlignment, false));
    if (ncclEpHandle_test_getNRanksPerNode(s.handle) != g_nranks) {
        GTEST_SKIP() << "single LSA team only";
    }
    ncclEpDispatchInputs_t dispatch_input = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dispatch_output = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t dispatch_config = NCCL_EP_DISPATCH_CONFIG_INIT;
    dispatch_input.tokens = s.x;
    dispatch_output.tokens = s.rx;
    dispatch_config.pass_direction = NCCL_EP_BWD_PASS;
    ncclEpCombineInputs_t combine_input = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t combine_output = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t combine_config = NCCL_EP_COMBINE_CONFIG_INIT;
    combine_input.tokens = s.rx;
    combine_output.tokens = s.cx;
    for (int generation : {0, 1}) {
        ASSERT_NO_FATAL_FAILURE(s.update(kTokens, generation, false));
        ASSERT_TRUE(ht_count_fused_active(s.handle));
        EXPECT_EQ(ncclEpDispatch(s.handle, &dispatch_input, &dispatch_output, nullptr,
                                 &dispatch_config, g_stream), ncclInvalidUsage);
        EXPECT_EQ(ncclEpCombine(s.handle, &combine_input, &combine_output,
                                &combine_config, g_stream), ncclInvalidUsage);
        ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
        ASSERT_NO_FATAL_FAILURE(s.backward());
    }
}

TEST_F(HtCompactProbTest, FlatPullCounterWindowAndIgnoredOffsets) {
    if (!ht_pull_push_active() || !fused_count_mode()) {
        GTEST_SKIP() << "fused flat pull metadata only";
    }
    constexpr int topk = 8, experts = 256;
    const int capacity = g_nranks * kTokens;
    const int local_experts = experts / g_nranks;
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity, kAlignment, false));
    if (ncclEpHandle_test_getNRanksPerNode(s.handle) != g_nranks) {
        GTEST_SKIP() << "single LSA team only";
    }
    ASSERT_NO_FATAL_FAILURE(s.update(kTokens, 0, false));
    void* window_counts = nullptr;
    const size_t counter_bytes = local_experts * sizeof(int32_t);
    NCCL_ASSERT(ncclMemAlloc(&window_counts, counter_bytes));
    ncclWindow_t window{};
    NCCL_ASSERT(ncclCommWindowRegister(g_comm, window_counts, counter_bytes,
                                      &window, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_NO_FATAL_FAILURE(s.tensor(&s.counts_tensor, 1, ncclInt32, nullptr, local_experts));
    s.counts_tensor->win_hdl = window;
    ASSERT_NO_FATAL_FAILURE(s.tensor(&s.offsets_tensor, 1, ncclInt64, s.offsets.p, local_experts));
    CUDA_ASSERT(cudaMemset(s.offsets.p, 0xA5, s.offsets.size * sizeof(int64_t)));
    ncclEpLayoutInfo_t info = NCCL_EP_LAYOUT_INFO_INIT;
    info.expert_offsets = s.offsets_tensor;
    NCCL_ASSERT(ncclEpUpdateHandle(s.handle, s.ids, &info, g_stream));
    EXPECT_TRUE(ht_count_fused_active(s.handle));
    info.expert_counters = s.counts_tensor;
    NCCL_ASSERT(ncclEpUpdateHandle(s.handle, s.ids, &info, g_stream));
    EXPECT_EQ(ncclEpHandle_test_getMetadataMode(s.handle), NCCL_EP_TEST_META_SCAN);
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    auto check_counters = [&]() {
        std::vector<int32_t> actual(local_experts), expected(local_experts, 0);
        CUDA_ASSERT(cudaMemcpy(actual.data(), window_counts, actual.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));
        for (int source = 0; source < g_nranks; ++source) {
            for (int token = 0; token < kTokens; ++token) {
                for (int k = 0; k < topk; ++k) {
                    const int64_t expert = route(source, token, k, topk, experts, 0);
                    if (expert >= g_rank * local_experts && expert < (g_rank + 1) * local_experts) {
                        ++expected[expert % local_experts];
                    }
                }
            }
        }
        EXPECT_EQ(actual, expected);
        std::vector<int64_t> offsets;
        ASSERT_NO_FATAL_FAILURE(s.offsets.download(offsets, local_experts));
        for (int64_t offset : offsets) {
            EXPECT_EQ(static_cast<uint64_t>(offset), UINT64_C(0xA5A5A5A5A5A5A5A5));
        }
    };
    ASSERT_NO_FATAL_FAILURE(check_counters());
    ASSERT_NO_FATAL_FAILURE(s.update(kTokens, 0, false));
    CUDA_ASSERT(cudaMemset(window_counts, 0xA5, counter_bytes));
    s.info.expert_counters = s.counts_tensor;
    s.info.expert_offsets = s.offsets_tensor;
    ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
    ASSERT_NO_FATAL_FAILURE(check_counters());
    ASSERT_NO_FATAL_FAILURE(s.backward());
    NCCL_ASSERT(ncclCommWindowDeregister(g_comm, window));
    NCCL_ASSERT(ncclMemFree(window_counts));
}

// Retain the process setting: this test creates differently configured groups
// in one process to catch JIT fast-cache aliasing as well as late getenv reads.
struct ScopedRecvTopkOrderEnv {
    bool existed = std::getenv("NCCL_EP_HT_RECV_TOPK_ORDER") != nullptr;
    std::string old = existed ? std::getenv("NCCL_EP_HT_RECV_TOPK_ORDER") : "";
    ~ScopedRecvTopkOrderEnv() {
        if (existed) {
            setenv("NCCL_EP_HT_RECV_TOPK_ORDER", old.c_str(), 1);
        } else {
            unsetenv("NCCL_EP_HT_RECV_TOPK_ORDER");
        }
    }
};

struct ScopedGroup {
    ncclEpGroup_t value = nullptr;
    ~ScopedGroup() {
        if (value) {
            ncclEpGroupDestroy(value);
        }
    }
};

class HtCompactProbOrderGate : public HtCompactProbTest, public ::testing::WithParamInterface<bool> {};

TEST_P(HtCompactProbOrderGate, CapturedPerGroupAndGraphReplay) {
    if (ht_pull_push_active()) {
        GTEST_SKIP() << "This case exercises the native compact decoder";
    }
    const bool em = GetParam();
    constexpr int topk = 8, experts = 256;
    const int capacity = (g_nranks * kTokens * topk + (experts / g_nranks) * 7 + 127) & ~63;
    ScopedRecvTopkOrderEnv env;
    ScopedGroup default_order, expert_order, source_order;
    State default_state, expert, source;

    unsetenv("NCCL_EP_HT_RECV_TOPK_ORDER");
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &default_order.value));
    ASSERT_NO_FATAL_FAILURE(default_state.initialize(default_order.value, em, true, topk, experts, capacity));
    setenv("NCCL_EP_HT_RECV_TOPK_ORDER", "expert-ascend", 1);
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &expert_order.value));
    ASSERT_NO_FATAL_FAILURE(expert.initialize(expert_order.value, em, true, topk, experts, capacity));
    setenv("NCCL_EP_HT_RECV_TOPK_ORDER", "source-compact", 1);
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, kTokens, &source_order.value));
    ASSERT_NO_FATAL_FAILURE(source.initialize(source_order.value, em, true, topk, experts, capacity));
    EXPECT_TRUE(default_state.topk_order);
    EXPECT_FALSE(expert.topk_order);
    EXPECT_TRUE(source.topk_order);

    // Change the process setting after all groups exist. Their decoder and EM
    // permutation choices must remain distinct, including under graph replay.
    setenv("NCCL_EP_HT_RECV_TOPK_ORDER", "expert-ascend", 1);
    for (State* state : {&default_state, &expert, &source}) {
        ASSERT_NO_FATAL_FAILURE(state->update(4, 0, false));
        ASSERT_NO_FATAL_FAILURE(state->dispatch(0, false));
    }
    for (State* state : {&expert, &default_state, &source, &expert}) {
        ASSERT_NO_FATAL_FAILURE(state->dispatch(1, true));
        ASSERT_NO_FATAL_FAILURE(state->backward());
    }
}

INSTANTIATE_TEST_SUITE_P(
    Layout, HtCompactProbOrderGate, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Em" : "Flat"; });

// Exercise the reviewer's exact example independently of public rank ownership:
// this decoder owns global experts 1 and 2. Also cross the saved-position-width
// boundary in the same process so neither JIT key can alias the other variant.
TEST_F(HtCompactProbTest, ExactGaplessOrderAndSavedColumns) {
    using namespace nccl_ep::ht;
    DeviceBuffer<int32_t> count;
    DeviceBuffer<uint8_t> bool_lerm;
    DeviceBuffer<uint64_t> bitmap_lerm;
    ASSERT_NO_FATAL_FAILURE(count.allocate(1));
    ASSERT_NO_FATAL_FAILURE(count.upload({1}));
    ASSERT_NO_FATAL_FAILURE(bool_lerm.allocate(2));
    ASSERT_NO_FATAL_FAILURE(bool_lerm.upload({1, 1}));
    ASSERT_NO_FATAL_FAILURE(bitmap_lerm.allocate(1));
    ASSERT_NO_FATAL_FAILURE(bitmap_lerm.upload({3}));
    for (int topk : {4, 256}) {
        SCOPED_TRACE(::testing::Message() << "topk=" << topk);
        std::vector<int32_t> ids(topk, -1);
        ids[0] = 2; ids[1] = 6; ids[2] = 7; ids[3] = 1;
        std::vector<float> weights(topk, 0.0f);
        weights[0] = 20.0f; weights[1] = 60.0f; weights[2] = 70.0f; weights[3] = 10.0f;
        DeviceBuffer<int32_t> input_ids, local_ids;
        DeviceBuffer<float> input_weights, wire, output_weights;
        DeviceBuffer<int64_t> output_ids;
        DeviceBuffer<uint8_t> positions8;
        DeviceBuffer<uint16_t> positions16;
        const int wire_floats = dispatch_compact_prob_row_floats(topk);
        ASSERT_NO_FATAL_FAILURE(input_ids.allocate(topk));
        ASSERT_NO_FATAL_FAILURE(input_ids.upload(ids));
        ASSERT_NO_FATAL_FAILURE(input_weights.allocate(topk));
        ASSERT_NO_FATAL_FAILURE(input_weights.upload(weights));
        ASSERT_NO_FATAL_FAILURE(wire.allocate(wire_floats));
        ASSERT_NO_FATAL_FAILURE(output_weights.allocate(topk));
        ASSERT_NO_FATAL_FAILURE(output_ids.allocate(topk));
        ASSERT_NO_FATAL_FAILURE(local_ids.allocate(topk));
        ASSERT_NO_FATAL_FAILURE(positions8.allocate(topk));
        ASSERT_NO_FATAL_FAILURE(positions16.allocate(topk));
        dispatch_prob_input_to_compact(input_ids.p, input_weights.p, wire.p, 1, topk, 512, wire_floats, g_stream);
        const int source_columns[] = {0, 3};
        const int expert_columns[] = {3, 0};
        const int invalid_position = topk <= 255 ? 255 : 65535;
        for (bool sorted : {false, true, false}) {
            const int* expected_columns = sorted ? expert_columns : source_columns;
            for (bool bitmap : {false, true}) {
                for (auto kind : {NCCL_EP_EXPERT_ID_GLOBAL, NCCL_EP_EXPERT_ID_LOCAL}) {
                    void* positions = topk <= 255 ? static_cast<void*>(positions8.p) : positions16.p;
                    const bool* lerm = bitmap ? reinterpret_cast<const bool*>(bitmap_lerm.p)
                                              : reinterpret_cast<const bool*>(bool_lerm.p);
                    dispatch_prob_compact_to_output(wire.p, lerm, positions, output_weights.p, output_ids.p,
                        1, count.p, topk, 2, wire_floats, 1, kind, false, bitmap, nullptr, nullptr,
                        sorted ? nullptr : local_ids.p, !sorted, g_stream);
                    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
                    std::vector<int64_t> actual_ids;
                    std::vector<float> actual_weights;
                    std::vector<uint8_t> actual8;
                    std::vector<uint16_t> actual16;
                    ASSERT_NO_FATAL_FAILURE(output_ids.download(actual_ids, topk));
                    ASSERT_NO_FATAL_FAILURE(output_weights.download(actual_weights, topk));
                    if (topk <= 255) {
                        ASSERT_NO_FATAL_FAILURE(positions8.download(actual8, topk));
                    } else {
                        ASSERT_NO_FATAL_FAILURE(positions16.download(actual16, topk));
                    }
                    for (int k = 0; k < topk; ++k) {
                        const int position = topk <= 255 ? actual8[k] : actual16[k];
                        if (k >= 2) {
                            EXPECT_EQ(actual_ids[k], -1);
                            EXPECT_EQ(bits(actual_weights[k]), bits(0.0f));
                            EXPECT_EQ(position, invalid_position);
                            continue;
                        }
                        const int source_k = expected_columns[k];
                        const int expected_id = kind == NCCL_EP_EXPERT_ID_LOCAL ? ids[source_k] - 1 : ids[source_k];
                        EXPECT_EQ(actual_ids[k], expected_id);
                        EXPECT_EQ(bits(actual_weights[k]), bits(weights[source_k]));
                        EXPECT_EQ(position, source_k);
                    }
                }
            }
        }
    }
}

using RoundTripCase = std::tuple<bool, bool, int>; // expert-major, int64 IDs, top-k
class HtCompactProbRoundTrip : public HtCompactProbTest, public ::testing::WithParamInterface<RoundTripCase> {};

TEST_P(HtCompactProbRoundTrip, ForwardAndBackward) {
    const auto& param = GetParam();
    run(std::get<0>(param), std::get<1>(param), std::get<2>(param));
}

INSTANTIATE_TEST_SUITE_P(
    LayoutAndIds, HtCompactProbRoundTrip,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Values(8, 10)),
    [](const ::testing::TestParamInfo<RoundTripCase>& info) {
        return std::string(std::get<0>(info.param) ? "Em" : "Flat") +
               "K" + std::to_string(std::get<2>(info.param)) +
               (std::get<1>(info.param) ? "Int64" : "Int32");
    });

TEST_F(HtCompactProbTest, DenseFallbackFlat) {
    if (g_nranks > 8) GTEST_SKIP() << "E8 fallback fixture requires EP4 or EP8";
    run(false, false, 8, 8);
}

TEST_F(HtCompactProbTest, DenseFallbackExpertMajor) {
    if (g_nranks > 8) GTEST_SKIP() << "E8 fallback fixture requires EP4 or EP8";
    run(true, true, 8, 8);
}

// FLAT has no per-expert zones, so its receive budget and recv_total_counter are rank-level
// (one slot per destined token). route() sends every token to two experts of each owning rank,
// so the per-expert sum is about twice the unique-token count. With the budget set to exactly the
// busiest rank's unique count, any budget check or total derived from the per-expert sum (as the
// unfused count path once did for FLAT) aborts or over-reports.
TEST_F(HtCompactProbTest, FlatRecvBudgetIsRankLevel) {
    if (ht_pull_push_active()) {
        GTEST_SKIP() << "This case exercises HT Flat count mode";
    }
    // The EM duplication reruns stage with a larger recv budget than this exact-budget fixture.
    if (enabled("NCCL_EP_HT_EM_LOCAL_DUP") || enabled("NCCL_EP_HT_EM_NVLINK_DUP"))
        GTEST_SKIP() << "EM duplication modes need more recv budget than the exact FLAT fixture";
    constexpr int topk = 8, experts = 256, tokens = 4;
    const int epr = experts / g_nranks;
    auto unique_recv = [&](int rank) {
        int count = 0;
        for (int source = 0; source < g_nranks; ++source) for (int t = 0; t < tokens; ++t) {
            bool hit = false;
            for (int k = 0; k < topk; ++k) {
                const int64_t e = route(source, t, k, topk, experts, 0);
                if (e >= 0 && e / epr == rank) hit = true;
            }
            count += hit ? 1 : 0;
        }
        return count;
    };
    int capacity = 0;
    for (int rank = 0; rank < g_nranks; ++rank) capacity = std::max(capacity, unique_recv(rank));
    ASSERT_GT(capacity, 0);
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, tokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity));
    // Hand layout info to UpdateHandle so count mode runs its unfused path and publishes
    // recv_total_counter here (before any dispatch); it must be the rank-level count.
    s.layout_info_at_update = true;
    ASSERT_NO_FATAL_FAILURE(s.update(tokens, 0, false));
    // Scan writes the same rank-level total at UpdateHandle; only multi-team groups differ.
    if (ncclEpHandle_test_getNRanksPerNode(s.handle) == g_nranks) {
        std::vector<int64_t> total_after_update;
        ASSERT_NO_FATAL_FAILURE(s.total.download(total_after_update, 1));
        EXPECT_EQ(total_after_update[0], unique_recv(g_rank)) << "UpdateHandle total, rank=" << g_rank;
    }
    ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
    EXPECT_EQ(s.output.total, unique_recv(g_rank)) << "rank=" << g_rank;
    ASSERT_NO_FATAL_FAILURE(s.backward());
}

// Same exact rank-level budget on the fused path: no layout info at UpdateHandle, so the
// dispatch MAP warps apply the budget check and publish recv_total_counter themselves.
TEST_F(HtCompactProbTest, FlatRecvBudgetIsRankLevelFused) {
    if (enabled("NCCL_EP_HT_EM_LOCAL_DUP") || enabled("NCCL_EP_HT_EM_NVLINK_DUP"))
        GTEST_SKIP() << "EM duplication modes need more recv budget than the exact FLAT fixture";
    if (!fused_count_mode())
        GTEST_SKIP() << "fused count mode only";
    constexpr int topk = 8, experts = 256, tokens = 4;
    const int epr = experts / g_nranks;
    auto unique_recv = [&](int rank) {
        int count = 0;
        for (int source = 0; source < g_nranks; ++source) for (int t = 0; t < tokens; ++t) {
            bool hit = false;
            for (int k = 0; k < topk; ++k) {
                const int64_t e = route(source, t, k, topk, experts, 0);
                if (e >= 0 && e / epr == rank) hit = true;
            }
            count += hit ? 1 : 0;
        }
        return count;
    };
    int capacity = 0;
    for (int rank = 0; rank < g_nranks; ++rank) capacity = std::max(capacity, unique_recv(rank));
    ASSERT_GT(capacity, 0);
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, false, tokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity));
    ASSERT_NO_FATAL_FAILURE(s.update(tokens, 0, false));
    ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
    EXPECT_EQ(s.output.total, unique_recv(g_rank)) << "rank=" << g_rank;
    // Poison the counter so the replay check proves the graph rewrote it rather
    // than reading the eager dispatch's value back.
    CUDA_ASSERT(cudaMemset(s.total.p, 0xFF, sizeof(int64_t)));
    ASSERT_NO_FATAL_FAILURE(s.dispatch(1, true));
    EXPECT_EQ(s.output.total, unique_recv(g_rank)) << "graph replay, rank=" << g_rank;
    ASSERT_NO_FATAL_FAILURE(s.backward());
}

// FLAT expert_counters are per-expert hit counts, not EM zone sizes. Under the DROP policy the
// count path used to clamp each expert against the budget left after the preceding experts'
// hits; with every token hitting two local experts that prefix exceeds the rank-level budget
// even though nothing is dropped, zeroing the later experts. Checked at UpdateHandle (unfused
// publish), after a plain dispatch and after graph replays (LERM recount).
TEST_F(HtCompactProbTest, FlatDropPolicyExpertCountsAreHitCounts) {
    if (ht_pull_push_active()) {
        GTEST_SKIP() << "This case exercises HT Flat count mode";
    }
    if (enabled("NCCL_EP_HT_EM_LOCAL_DUP") || enabled("NCCL_EP_HT_EM_NVLINK_DUP"))
        GTEST_SKIP() << "EM duplication modes need more recv budget than the exact FLAT fixture";
    constexpr int topk = 8, experts = 256, tokens = 4;
    const int epr = experts / g_nranks;
    std::vector<int32_t> expected_hits(epr, 0);
    int capacity = 0;
    for (int rank = 0; rank < g_nranks; ++rank) {
        int unique = 0;
        for (int source = 0; source < g_nranks; ++source) for (int t = 0; t < tokens; ++t) {
            bool hit = false;
            for (int k = 0; k < topk; ++k) {
                const int64_t e = route(source, t, k, topk, experts, 0);
                if (e < 0 || e / epr != rank) continue;
                hit = true;
                if (rank == g_rank) ++expected_hits[static_cast<int>(e % epr)];
            }
            unique += hit ? 1 : 0;
        }
        capacity = std::max(capacity, unique);
    }
    ASSERT_GT(capacity, 0);
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, true, tokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, false, false, topk, experts, capacity));
    DeviceBuffer<int32_t> hits;
    ASSERT_NO_FATAL_FAILURE(hits.allocate(epr));
    ncclEpTensor_t* hits_tensor = nullptr;
    ASSERT_NO_FATAL_FAILURE(s.tensor(&hits_tensor, 1, ncclInt32, hits.p, epr));
    s.info.expert_counters = hits_tensor;
    // The UpdateHandle-time counts need the unfused exchange; the dispatch-time recount is
    // checked on the same handle afterwards.
    s.layout_info_at_update = true;
    const std::vector<int32_t> poison(epr, static_cast<int32_t>(0xA5A5A5A5));
    auto expect_hits = [&](const char* stage) {
        std::vector<int32_t> actual;
        ASSERT_NO_FATAL_FAILURE(hits.download(actual, epr));
        for (int e = 0; e < epr; ++e) {
            EXPECT_EQ(actual[e], expected_hits[e]) << stage << " rank=" << g_rank << " local_expert=" << e;
        }
    };
    ASSERT_NO_FATAL_FAILURE(hits.upload(poison));
    ASSERT_NO_FATAL_FAILURE(s.update(tokens, 0, false));
    ASSERT_NO_FATAL_FAILURE(expect_hits("UpdateHandle"));
    ASSERT_NO_FATAL_FAILURE(hits.upload(poison));
    ASSERT_NO_FATAL_FAILURE(s.dispatch(0, false));
    EXPECT_EQ(s.output.total, capacity) << "rank=" << g_rank;
    if (ht_count_mode_active(s.handle)) {
        ASSERT_NO_FATAL_FAILURE(expect_hits("dispatch"));
        ASSERT_NO_FATAL_FAILURE(hits.upload(poison));
        ASSERT_NO_FATAL_FAILURE(s.dispatch(1, true));
        ASSERT_NO_FATAL_FAILURE(expect_hits("graph replay"));
    }
    ASSERT_NO_FATAL_FAILURE(s.backward());
}

// All push EM modes: ten local hits per token, reversed source columns, and a
// partially retained expert zone. EM-LP must rebuild its gap-less map without
// pairing dropped experts with surviving weights or backward source columns.
// A final expert zone ends midway through its alignment padding: writes past
// capacity used to corrupt the compact rows immediately following staging.
TEST_F(HtCompactProbTest, PartialZoneDrop) {
    if (g_nranks != 4 && g_nranks != 8) GTEST_SKIP() << "Partial-zone fixture requires EP4 or EP8";
    if (ht_pull_push_active()) {
        GTEST_SKIP() << "This case exercises the native compact decoder";
    }

    constexpr int topk = 10, capacity = 160, alignment = 64;
    const int experts = 128 * g_nranks, tokens = 16 / g_nranks;
    // The fixture sends 16 tokens globally. Declare that per-rank bound so
    // EP8 also meets the native-EM staging requirement: ranks * tokens * K.
    ASSERT_NO_FATAL_FAILURE(create_group(experts, topk, capacity, true, tokens, &group_));
    State s;
    ASSERT_NO_FATAL_FAILURE(s.initialize(group_, true, true, topk, experts, capacity, alignment, true));
    s.ids->sizes[0] = s.x->sizes[0] = s.w->sizes[0] = s.cx->sizes[0] = s.cw->sizes[0] = tokens;
    std::vector<int64_t> ids(tokens * topk);
    std::vector<nv_bfloat16> x(tokens * kWidth), initial_recv(s.recv_tokens.size, __float2bfloat16(0));
    std::vector<float> w(tokens * topk), initial_w(s.recv_weights.size, 0.0f);
    for (int t = 0; t < tokens; ++t) {
        for (int d = 0; d < kWidth; ++d) x[t * kWidth + d] = __float2bfloat16(1 + g_rank * tokens + t);
        for (int k = 0; k < topk; ++k) {
            ids[t * topk + k] = topk - 1 - k;
            w[t * topk + k] = source_weight(g_rank, t, k, 0, 0);
        }
    }
    std::fill(initial_recv.end() - 16, initial_recv.end(), __float2bfloat16(77));
    std::fill(initial_w.end() - 4, initial_w.end(), 77.0f);
    ASSERT_NO_FATAL_FAILURE(s.ids64.upload(ids));
    ASSERT_NO_FATAL_FAILURE(s.tokens.upload(x));
    ASSERT_NO_FATAL_FAILURE(s.weights.upload(w));
    ASSERT_NO_FATAL_FAILURE(s.recv_tokens.upload(initial_recv));
    ASSERT_NO_FATAL_FAILURE(s.recv_weights.upload(initial_w));

    NCCL_ASSERT(ncclEpUpdateHandle(s.handle, s.ids, nullptr, g_stream));
    ncclEpDispatchInputs_t di = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dout = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    di.tokens = s.x; di.topk_weights = s.w;
    dout.tokens = s.rx; dout.topk_weights = s.rw;
    ncclEpDispatchConfig_t dc = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(s.handle, &di, &dout, nullptr, &dc, g_stream));
    NCCL_ASSERT(ncclEpComplete(s.handle, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    std::vector<nv_bfloat16> recv;
    std::vector<float> recv_w;
    ASSERT_NO_FATAL_FAILURE(s.recv_tokens.download(recv, s.recv_tokens.size));
    ASSERT_NO_FATAL_FAILURE(s.recv_weights.download(recv_w, s.recv_weights.size));

    // Sixteen rows per expert: E0 at 0..15, E1 at 64..79, E2 at 128..143.
    // Only padding rows 144..159 fit; E3..E9 must be dropped.
    std::vector<float> gradients(s.recv_weights.size, 12345.0f);
    std::vector<int> seen(3 * 16, 0);
    for (int slot = 0; slot < capacity; ++slot) {
        const int expert = slot / alignment;
        const bool active = g_rank == 0 && slot % alignment < 16;
        const float token_id = __bfloat162float(recv[slot * kWidth]);
        if (active) {
            const int source_id = static_cast<int>(token_id) - 1;
            ASSERT_GE(source_id, 0); ASSERT_LT(source_id, 16);
            ASSERT_EQ(token_id, static_cast<float>(source_id + 1));
            EXPECT_EQ(bits(recv_w[slot]), bits(source_weight(source_id / tokens, source_id % tokens,
                                                          topk - 1 - expert, 0, 0)));
            gradients[slot] = independent_gradient(source_id / tokens, source_id % tokens, expert, 0);
            ++seen[expert * 16 + source_id];
        } else {
            if (g_rank == 0) EXPECT_EQ(token_id, 0.0f);
            EXPECT_EQ(bits(recv_w[slot]), bits(0.0f));
        }
        if (g_rank == 0)
            for (int d = 1; d < kWidth; ++d) EXPECT_EQ(__bfloat162float(recv[slot * kWidth + d]), token_id);
    }
    if (g_rank == 0) for (int count : seen) EXPECT_EQ(count, 1);
    for (size_t i = capacity * kWidth; i < recv.size(); ++i) EXPECT_EQ(__bfloat162float(recv[i]), 77.0f);
    for (size_t i = capacity; i < recv_w.size(); ++i) EXPECT_EQ(recv_w[i], 77.0f);

    // Nonzero padding gradients expose stale saved positions; only E0/E1/E2
    // may contribute. Keep the public buffer guard untouched.
    std::fill(gradients.end() - 4, gradients.end(), 77.0f);
    ASSERT_NO_FATAL_FAILURE(s.recv_weights.upload(gradients));
    ncclEpCombineInputs_t ci = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t co = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ci.tokens = s.rx; ci.topk_weights = s.rw;
    co.tokens = s.cx; co.topk_weights = s.cw;
    ncclEpCombineConfig_t cc = NCCL_EP_COMBINE_CONFIG_INIT;
    cc.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(s.handle, &ci, &co, &cc, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    std::vector<float> combined_w;
    std::vector<nv_bfloat16> combined;
    ASSERT_NO_FATAL_FAILURE(s.combined_weights.download(combined_w, tokens * topk));
    ASSERT_NO_FATAL_FAILURE(s.combined_tokens.download(combined, tokens * kWidth));
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < topk; ++k) {
            const int expert = static_cast<int>(ids[t * topk + k]);
            const float expected = expert < 3 ? independent_gradient(g_rank, t, expert, 0) : 0.0f;
            EXPECT_EQ(bits(combined_w[t * topk + k]), bits(expected));
        }
        for (int d = 0; d < kWidth; ++d)
            EXPECT_EQ(__bfloat162float(combined[t * kWidth + d]), 3.0f * __bfloat162float(x[t * kWidth + d]));
    }
}

void bootstrap() {
    int devices = 0;
    CUDA_ASSERT(cudaGetDeviceCount(&devices)); ASSERT_GT(devices, 0);
    CUDA_ASSERT(cudaSetDevice(g_rank % devices));
    ncclUniqueId uid{};
    ASSERT_NO_FATAL_FAILURE(exchange_uid(&uid));
    NCCL_ASSERT(ncclCommInitRank(&g_comm, g_nranks, uid, g_rank));
    CUDA_ASSERT(cudaStreamCreate(&g_stream));
}
} // namespace

int main(int argc, char** argv) {
    ep_parse_args(argc, argv, "nccl_ep_ht_compact_prob_uid");
    ::testing::InitGoogleTest(&argc, argv);
    if (g_nranks != 4 && g_nranks != 8 && g_nranks != 16 && g_nranks != 32) {
        fprintf(stderr, "This test requires EP4, EP8, EP16 or EP32.\n");
        return EXIT_FAILURE;
    }
    bootstrap();
    if (::testing::Test::HasFatalFailure()) return EXIT_FAILURE;
    int result = RUN_ALL_TESTS();
    cudaStreamDestroy(g_stream);
    ncclCommDestroy(g_comm);
    if (g_rank == 0) remove(g_uid_file.c_str());
    return result;
}
