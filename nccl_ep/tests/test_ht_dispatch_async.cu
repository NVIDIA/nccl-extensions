/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_common.h"
#include "device/ht_ep_adapter.cuh"

#include <cuda.h>
#include <cstring>
#include <optional>
#include <vector>

namespace {

constexpr uint16_t kSentinel = 0xA5A5;

TEST(DispatchCopyPolicyTest,ArchitectureGrid) {
    using nccl_ep::ht::dispatch_copy_tma_sms;
    EXPECT_EQ(dispatch_copy_tma_sms(100, 148), 96u);
    EXPECT_EQ(dispatch_copy_tma_sms(100, 152), 96u);
    EXPECT_EQ(dispatch_copy_tma_sms(103, 152), 96u);
    EXPECT_EQ(dispatch_copy_tma_sms(107, 212), 128u);
    EXPECT_EQ(dispatch_copy_tma_sms(110, 212), 128u);
    EXPECT_EQ(dispatch_copy_tma_sms(90, 132), 128u);
    EXPECT_EQ(dispatch_copy_tma_sms(120, 144), 128u);
    EXPECT_EQ(dispatch_copy_tma_sms(100, 64), 64u);
    EXPECT_EQ(dispatch_copy_tma_sms(103, 64), 64u);
    EXPECT_EQ(dispatch_copy_tma_sms(107, 64), 64u);
}

class ScopedEnv {
    const char* name_;
    std::optional<std::string> value_;
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        if (const char* previous = getenv(name)) value_ = previous;
        setenv(name_, value, 1);
    }
    ~ScopedEnv() {
        if (value_) setenv(name_, value_->c_str(), 1);
        else unsetenv(name_);
    }
};

struct DispatchBuffers {
    nv_bfloat16* tokens = nullptr;
    nv_bfloat16* recv = nullptr;
    nv_bfloat16* recv_storage = nullptr;
    float* weights = nullptr;
    float* recv_weights = nullptr;
    int64_t* recv_idx = nullptr;

    ncclEpTensor_t* tokens_tensor = nullptr;
    ncclEpTensor_t* recv_tensor = nullptr;
    ncclEpTensor_t* weights_tensor = nullptr;
    ncclEpTensor_t* recv_weights_tensor = nullptr;
    ncclEpTensor_t* recv_idx_tensor = nullptr;
};

class HtDispatchAsyncTest : public EpTestBase, public ::testing::WithParamInterface<const char*> {
protected:
    static constexpr const char* kCopyEnv = "NCCL_EP_DISPATCH_COPY_MODE";
    ncclEpGroup_t group_ = nullptr;
    bool had_copy_env_ = false;
    std::string saved_copy_env_;
    bool tma_copy_available_ = false;

    bool use_tma_copy() const {
        return tma_copy_available_;
    }
    bool use_sm_copy() const {
        return (GetParam() != nullptr && strcasecmp(GetParam(), "SIMT") == 0) || use_tma_copy();
    }

    void SetUp() override {
        const char* previous = getenv(kCopyEnv);
        had_copy_env_ = previous != nullptr;
        if (previous) saved_copy_env_ = previous;
        if (GetParam()) setenv(kCopyEnv, GetParam(), 1);
        else unsetenv(kCopyEnv);
        EpTestBase::SetUp();
        create_group();
    }

    void create_group(int recv_capacity = kMaxRecvSlots, bool drop = false) {
        if (group_) NCCL_ASSERT(ncclEpGroupDestroy(group_));
        group_ = nullptr;
        ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
        config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        config.num_experts = kNumExperts;
        config.max_dispatch_tokens_per_rank = kNumTokens;
        config.max_token_bytes = kHidden * sizeof(nv_bfloat16);
        config.max_recv_tokens_per_rank = recv_capacity;
        if (drop) config.overflow_policy = NCCL_EP_OVERFLOW_DROP;
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &config));
        tma_copy_available_ = false;
        if (GetParam() == nullptr ||
            (strcasecmp(GetParam(), "CE") != 0 && strcasecmp(GetParam(), "SIMT") != 0)) {
            // Use the library's opt-in check, including runtime and build-time CE fallback.
            const ncclResult_t status = nccl_ep::ht::configure_dispatch_copy_tma();
            ASSERT_TRUE(status == ncclSuccess || status == ncclInvalidUsage);
            tma_copy_available_ = status == ncclSuccess;
        }
    }

    void TearDown() override {
        if (group_) NCCL_ASSERT(ncclEpGroupDestroy(group_));
        EpTestBase::TearDown();
        if (had_copy_env_) setenv(kCopyEnv, saved_copy_env_.c_str(), 1);
        else unsetenv(kCopyEnv);
    }

    ncclEpHandle_t make_handle(const ncclEpHandleConfig_t* config) {
        ncclEpHandle_t handle = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(
            &handle, group_, NCCL_EP_LAYOUT_FLAT, topk_idx_, nullptr, config, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return handle;
    }

    void init_buffers(DispatchBuffers& b, int recv_rows, int recv_offset = 0) {
        CUDA_ASSERT(cudaMalloc(&b.tokens, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&b.recv_storage, (recv_rows * kHidden + recv_offset + 1) * sizeof(nv_bfloat16)));
        b.recv = b.recv_storage + recv_offset;
        CUDA_ASSERT(cudaMemset(b.recv_storage, 0xA5, (recv_rows * kHidden + recv_offset + 1) * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&b.weights, kNumTokens * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&b.recv_weights, kMaxRecvSlots * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&b.recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)));

        std::vector<nv_bfloat16> tokens(kNumTokens * kHidden);
        for (int token = 0; token < kNumTokens; ++token) {
            const float value = static_cast<float>(g_rank * kNumTokens + token + 1);
            for (int hidden = 0; hidden < kHidden; ++hidden) {
                tokens[token * kHidden + hidden] = __float2bfloat16(value);
            }
        }
        std::vector<float> weights(kNumTokens * kTopK, 1.0f);
        CUDA_ASSERT(cudaMemcpy(
            b.tokens,
            tokens.data(),
            tokens.size() * sizeof(nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(
            b.weights,
            weights.data(),
            weights.size() * sizeof(float),
            cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&b.tokens_tensor, 2, ncclBfloat16, b.tokens, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&b.recv_tensor, 2, ncclBfloat16, b.recv, recv_rows, kHidden));
        NCCL_ASSERT(epTensorCreate(&b.weights_tensor, 2, ncclFloat32, b.weights, kNumTokens, kTopK));
        NCCL_ASSERT(
            epTensorCreate(&b.recv_weights_tensor, 2, ncclFloat32, b.recv_weights, kMaxRecvSlots, kTopK));
        NCCL_ASSERT(epTensorCreate(&b.recv_idx_tensor, 2, ncclInt64, b.recv_idx, kMaxRecvSlots, kTopK));
    }

    void destroy_buffers(DispatchBuffers& b) {
        ncclEpTensorDestroy(b.tokens_tensor);
        ncclEpTensorDestroy(b.recv_tensor);
        ncclEpTensorDestroy(b.weights_tensor);
        ncclEpTensorDestroy(b.recv_weights_tensor);
        ncclEpTensorDestroy(b.recv_idx_tensor);
        cudaFree(b.tokens);
        cudaFree(b.recv_storage);
        cudaFree(b.weights);
        cudaFree(b.recv_weights);
        cudaFree(b.recv_idx);
    }

    ncclResult_t dispatch(ncclEpHandle_t handle, const DispatchBuffers& b) {
        ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
        inputs.tokens = b.tokens_tensor;
        inputs.topk_weights = b.weights_tensor;
        outputs.tokens = b.recv_tensor;
        outputs.topk_weights = b.recv_weights_tensor;
        outputs.topk_idx = b.recv_idx_tensor;
        return ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream);
    }

    void reset_outputs(const DispatchBuffers& b) {
        CUDA_ASSERT(cudaMemset(b.recv, 0xA5, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(b.recv_weights, 0xA5, kMaxRecvSlots * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMemset(b.recv_idx, 0xA5, kMaxRecvSlots * kTopK * sizeof(int64_t)));
    }

    void expect_metadata(const DispatchBuffers& b, int expected_rows) {
        std::vector<uint32_t> weights(kMaxRecvSlots * kTopK);
        std::vector<int64_t> indices(kMaxRecvSlots * kTopK);
        CUDA_ASSERT(cudaMemcpy(weights.data(), b.recv_weights, weights.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(indices.data(), b.recv_idx, indices.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
        for (int row = 0; row < kMaxRecvSlots; ++row) {
            SCOPED_TRACE(row);
            if (row < expected_rows) {
                EXPECT_EQ(weights[row], 0x3f800000u); // 1.0f
                EXPECT_GE(indices[row], 0);
                EXPECT_LT(indices[row], kNumExperts / g_nranks);
            } else {
                EXPECT_EQ(weights[row], 0xA5A5A5A5u);
                EXPECT_EQ(static_cast<uint64_t>(indices[row]), 0xA5A5A5A5A5A5A5A5ull);
            }
        }
    }

    void expect_rows(const DispatchBuffers& b, int expected_rows, bool check_tail = true) {
        std::vector<nv_bfloat16> recv(kMaxRecvSlots * kHidden);
        CUDA_ASSERT(cudaMemcpy(
            recv.data(),
            b.recv,
            recv.size() * sizeof(nv_bfloat16),
            cudaMemcpyDeviceToHost));

        for (int row = 0; row < kMaxRecvSlots; ++row) {
            for (int hidden = 0; hidden < kHidden; ++hidden) {
                uint16_t raw;
                memcpy(&raw, &recv[row * kHidden + hidden], sizeof(raw));
                if (row < expected_rows) {
                    EXPECT_NE(raw, kSentinel) << "row " << row << " hidden " << hidden;
                    const float value = __bfloat162float(recv[row * kHidden + hidden]);
                    EXPECT_EQ(value, __bfloat162float(recv[row * kHidden]));
                    EXPECT_GE(value, 1.0f);
                    EXPECT_LE(value, static_cast<float>(g_nranks * kNumTokens));
                } else if (check_tail) {
                    EXPECT_EQ(raw, kSentinel) << "row " << row << " hidden " << hidden;
                }
            }
        }
    }
};

TEST_P(HtDispatchAsyncTest, RejectsUndersizedOutput) {
    ncclEpHandle_t handle = make_handle(nullptr);
    DispatchBuffers b;
    init_buffers(b, kMaxRecvSlots - 1);

    EXPECT_EQ(dispatch(handle, b), ncclInvalidArgument);

    destroy_buffers(b);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_P(HtDispatchAsyncTest, ExactRowsAndGraphReplay) {
    ncclEpHandle_t handle = make_handle(nullptr);
    DispatchBuffers b;
    init_buffers(b, kMaxRecvSlots);

    // Existing groups must retain their policy after an environment change.
    setenv(kCopyEnv, use_sm_copy() ? "CE" : "SIMT", 1);

    reset_outputs(b);
    NCCL_ASSERT(dispatch(handle, b));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_rows(b, 4);
    expect_metadata(b, 4);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
    NCCL_ASSERT(ncclEpUpdateHandle(handle, topk_idx_, nullptr, g_stream));
    NCCL_ASSERT(dispatch(handle, b));
    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
    size_t node_count = 0;
    CUDA_ASSERT(cudaGraphGetNodes(graph, nullptr, &node_count));
    std::vector<cudaGraphNode_t> nodes(node_count);
    CUDA_ASSERT(cudaGraphGetNodes(graph, nodes.data(), &node_count));
    int output_copies = 0;
    int tma_copies = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        CUDA_ASSERT(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeKernel) {
            CUDA_KERNEL_NODE_PARAMS params{};
            ASSERT_EQ(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &params), CUDA_SUCCESS);
            if (params.blockDimX == 32 && params.sharedMemBytes == 128 * 1024) {
                int device;
                cudaDeviceProp prop{};
                CUDA_ASSERT(cudaGetDevice(&device));
                CUDA_ASSERT(cudaGetDeviceProperties(&prop, device));
                EXPECT_EQ(params.gridDimX, nccl_ep::ht::dispatch_copy_tma_sms(
                    prop.major * 10 + prop.minor, prop.multiProcessorCount));
                ++tma_copies;
            }
        }
        if (type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms params{};
        CUDA_ASSERT(cudaGraphMemcpyNodeGetParams(node, &params));
        if (params.dstPtr.ptr == b.recv) ++output_copies;
    }
    EXPECT_EQ(output_copies, use_sm_copy() ? 0 : 1);
    EXPECT_EQ(tma_copies, use_tma_copy() ? 1 : 0);
    CUDA_ASSERT(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    std::vector<int64_t> concentrated(kNumTokens * kTopK, 0);
    CUDA_ASSERT(cudaMemcpy(
        d_topk_,
        concentrated.data(),
        concentrated.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    reset_outputs(b);
    CUDA_ASSERT(cudaGraphLaunch(graph_exec, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_rows(b, g_rank == 0 ? g_nranks * kNumTokens : 0, use_sm_copy());
    expect_metadata(b, g_rank == 0 ? g_nranks * kNumTokens : 0);

    std::vector<int64_t> distributed(kNumTokens * kTopK);
    for (int token = 0; token < kNumTokens; ++token) {
        distributed[token] = expert_for_token(token);
    }
    CUDA_ASSERT(cudaMemcpy(
        d_topk_,
        distributed.data(),
        distributed.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    reset_outputs(b);
    CUDA_ASSERT(cudaGraphLaunch(graph_exec, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_rows(b, 4, use_sm_copy());
    expect_metadata(b, 4);

    CUDA_ASSERT(cudaGraphExecDestroy(graph_exec));
    CUDA_ASSERT(cudaGraphDestroy(graph));
    destroy_buffers(b);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_P(HtDispatchAsyncTest, UnalignedOutputUsesCeUnlessSimtRequested) {
    ncclEpHandle_t handle = make_handle(nullptr);
    for (bool unaligned : {true, false}) {
        SCOPED_TRACE(unaligned);
        DispatchBuffers b;
        init_buffers(b, kMaxRecvSlots, unaligned ? 1 : 0);
        const bool sm_copy = use_sm_copy() && (!unaligned || !use_tma_copy());
        const bool tma_copy = use_tma_copy() && !unaligned;
        reset_outputs(b);
        NCCL_ASSERT(dispatch(handle, b));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        expect_rows(b, 4);
        expect_metadata(b, 4);

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
        NCCL_ASSERT(ncclEpUpdateHandle(handle, topk_idx_, nullptr, g_stream));
        NCCL_ASSERT(dispatch(handle, b));
        CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
        size_t node_count = 0;
        CUDA_ASSERT(cudaGraphGetNodes(graph, nullptr, &node_count));
        std::vector<cudaGraphNode_t> nodes(node_count);
        CUDA_ASSERT(cudaGraphGetNodes(graph, nodes.data(), &node_count));
        int output_copies = 0, tma_copies = 0;
        for (cudaGraphNode_t node : nodes) {
            cudaGraphNodeType type;
            CUDA_ASSERT(cudaGraphNodeGetType(node, &type));
            if (type == cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms params{};
                CUDA_ASSERT(cudaGraphMemcpyNodeGetParams(node, &params));
                if (params.dstPtr.ptr == b.recv) ++output_copies;
            } else if (type == cudaGraphNodeTypeKernel) {
                CUDA_KERNEL_NODE_PARAMS params{};
                ASSERT_EQ(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &params), CUDA_SUCCESS);
                if (params.blockDimX == 32 && params.sharedMemBytes == 128 * 1024) ++tma_copies;
            }
        }
        EXPECT_EQ(output_copies, sm_copy ? 0 : 1);
        EXPECT_EQ(tma_copies, tma_copy ? 1 : 0);
        CUDA_ASSERT(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        for (bool concentrated : {true, false}) {
            std::vector<int64_t> routing(kNumTokens * kTopK, 0);
            if (!concentrated) {
                for (int token = 0; token < kNumTokens; ++token) routing[token] = expert_for_token(token);
            }
            CUDA_ASSERT(cudaMemcpy(d_topk_, routing.data(), routing.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
            reset_outputs(b);
            CUDA_ASSERT(cudaGraphLaunch(graph_exec, g_stream));
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
            const int rows = concentrated ? (g_rank == 0 ? g_nranks * kNumTokens : 0) : 4;
            expect_rows(b, rows, sm_copy);
            expect_metadata(b, rows);
        }
        uint16_t guard;
        if (unaligned) {
            CUDA_ASSERT(cudaMemcpy(&guard, b.recv_storage, sizeof(guard), cudaMemcpyDeviceToHost));
            EXPECT_EQ(guard, kSentinel);
        }
        CUDA_ASSERT(cudaMemcpy(&guard, b.recv + kMaxRecvSlots * kHidden, sizeof(guard), cudaMemcpyDeviceToHost));
        EXPECT_EQ(guard, kSentinel);
        CUDA_ASSERT(cudaGraphExecDestroy(graph_exec));
        CUDA_ASSERT(cudaGraphDestroy(graph));
        destroy_buffers(b);
    }
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_P(HtDispatchAsyncTest, MetadataOverflowDrop) {
    constexpr int kCapacity = kNumTokens;
    create_group(kCapacity, true);
    ncclEpHandle_t handle = make_handle(nullptr);
    DispatchBuffers b;
    init_buffers(b, kMaxRecvSlots);

    reset_outputs(b);
    NCCL_ASSERT(dispatch(handle, b));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_metadata(b, kCapacity);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
    NCCL_ASSERT(ncclEpUpdateHandle(handle, topk_idx_, nullptr, g_stream));
    NCCL_ASSERT(dispatch(handle, b));
    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
    CUDA_ASSERT(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    std::vector<int64_t> concentrated(kNumTokens * kTopK, 0);
    CUDA_ASSERT(cudaMemcpy(d_topk_, concentrated.data(), concentrated.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    reset_outputs(b);
    CUDA_ASSERT(cudaGraphLaunch(graph_exec, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_metadata(b, g_rank == 0 ? kCapacity : 0);

    CUDA_ASSERT(cudaGraphExecDestroy(graph_exec));
    CUDA_ASSERT(cudaGraphDestroy(graph));
    destroy_buffers(b);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_P(HtDispatchAsyncTest, ExpertMajorMetadataPaddingAndTails) {
    constexpr int kAlignment = 8;
    for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(mode); // Count, scan/local-permute, local-dup, NVLink-dup.
        ScopedEnv scan("NCCL_EP_HT_EM_AG_SCAN_MODE", mode == 0 ? "0" : "1");
        ScopedEnv local_dup("NCCL_EP_HT_EM_LOCAL_DUP", mode == 2 ? "1" : "0");
        ScopedEnv nvlink_dup("NCCL_EP_HT_EM_NVLINK_DUP", mode == 3 ? "1" : "0");
        ScopedEnv pull("NCCL_EP_HT_EM_PULL_PUSH", "0");
        ScopedEnv unfused("NCCL_EP_HT_EM_COUNT_UNFUSED", "0");
        create_group();

        std::vector<int64_t> distributed(kNumTokens * kTopK);
        for (int token = 0; token < kNumTokens; ++token) distributed[token] = expert_for_token(token);
        CUDA_ASSERT(cudaMemcpy(d_topk_, distributed.data(), distributed.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        ncclEpHandleConfig_t config = NCCL_EP_HANDLE_CONFIG_INIT;
        config.dispatch_output_per_expert_alignment = kAlignment;
        ncclEpHandle_t handle = nullptr;
        NCCL_ASSERT(ncclEpCreateHandle(&handle, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_, nullptr, &config, g_stream));
        DispatchBuffers b;
        init_buffers(b, kMaxRecvSlots);
        NCCL_ASSERT(ncclEpTensorDestroy(b.recv_weights_tensor));
        NCCL_ASSERT(epTensorCreate(&b.recv_weights_tensor, 1, ncclFloat32, b.recv_weights, kMaxRecvSlots));
        NCCL_ASSERT(ncclEpTensorDestroy(b.recv_idx_tensor));
        b.recv_idx_tensor = nullptr;

        auto check = [&](bool concentrated, bool graph_replay) {
            std::vector<uint32_t> weights(kMaxRecvSlots);
            std::vector<nv_bfloat16> tokens(kMaxRecvSlots * kHidden);
            CUDA_ASSERT(cudaMemcpy(weights.data(), b.recv_weights, weights.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(tokens.data(), b.recv, tokens.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
            const int extent = concentrated ? (g_rank == 0 ? g_nranks * kNumTokens : 0) : 2 * kAlignment;
            for (int row = 0; row < kMaxRecvSlots; ++row) {
                SCOPED_TRACE(row);
                if (row >= extent) {
                    EXPECT_EQ(weights[row], 0xA5A5A5A5u);
                    // Direct EM with CE may copy capacity during replay.
                    if (!(mode >= 2 && graph_replay && !use_sm_copy())) {
                        uint16_t raw;
                        memcpy(&raw, &tokens[row * kHidden], sizeof(raw));
                        EXPECT_EQ(raw, kSentinel);
                    }
                } else {
                    const bool valid = concentrated || row % kAlignment < 2;
                    EXPECT_EQ(weights[row], valid ? 0x3f800000u : 0u);
                    for (int h = 0; h < kHidden; ++h) {
                        const float value = __bfloat162float(tokens[row * kHidden + h]);
                        if (valid) {
                            EXPECT_GE(value, 1.0f);
                            EXPECT_LE(value, static_cast<float>(g_nranks * kNumTokens));
                        } else {
                            EXPECT_EQ(value, 0.0f);
                        }
                    }
                }
            }
        };

        reset_outputs(b);
        NCCL_ASSERT(dispatch(handle, b));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        check(false, false);

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
        NCCL_ASSERT(ncclEpUpdateHandle(handle, topk_idx_, nullptr, g_stream));
        NCCL_ASSERT(dispatch(handle, b));
        CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
        CUDA_ASSERT(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        for (bool concentrated : {true, false}) {
            std::vector<int64_t> routing = concentrated ? std::vector<int64_t>(kNumTokens * kTopK, 0) : distributed;
            CUDA_ASSERT(cudaMemcpy(d_topk_, routing.data(), routing.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
            reset_outputs(b);
            CUDA_ASSERT(cudaGraphLaunch(graph_exec, g_stream));
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
            check(concentrated, true);
        }

        CUDA_ASSERT(cudaGraphExecDestroy(graph_exec));
        CUDA_ASSERT(cudaGraphDestroy(graph));
        destroy_buffers(b);
        NCCL_ASSERT(ncclEpHandleDestroy(handle));
    }
}

TEST_P(HtDispatchAsyncTest, CopiesFp8TokensAndScales) {
    constexpr int kFp8Hidden = 512;
    constexpr int kScalesPerToken = 4;
    constexpr uint8_t kTokenValue = 0x3c;

    uint8_t* tokens = nullptr;
    uint8_t* recv = nullptr;
    float* scales = nullptr;
    float* recv_scales = nullptr;
    float* weights = nullptr;
    float* recv_weights = nullptr;
    int64_t* recv_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&tokens, kNumTokens * kFp8Hidden));
    CUDA_ASSERT(cudaMalloc(&recv, kMaxRecvSlots * kFp8Hidden));
    CUDA_ASSERT(cudaMalloc(&scales, kNumTokens * kScalesPerToken * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&recv_scales, kMaxRecvSlots * kScalesPerToken * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&weights, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&recv_weights, kMaxRecvSlots * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)));

    std::vector<uint8_t> input_tokens(kNumTokens * kFp8Hidden, kTokenValue);
    std::vector<float> input_scales(kNumTokens * kScalesPerToken, 2.0f);
    std::vector<float> input_weights(kNumTokens * kTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(tokens, input_tokens.data(), input_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(
        scales,
        input_scales.data(),
        input_scales.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(
        weights,
        input_weights.data(),
        input_weights.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(recv, 0xA5, kMaxRecvSlots * kFp8Hidden));
    CUDA_ASSERT(cudaMemset(recv_scales, 0xA5, kMaxRecvSlots * kScalesPerToken * sizeof(float)));

    ncclEpTensor_t* tokens_tensor = nullptr;
    ncclEpTensor_t* recv_tensor = nullptr;
    ncclEpTensor_t* scales_tensor = nullptr;
    ncclEpTensor_t* recv_scales_tensor = nullptr;
    ncclEpTensor_t* weights_tensor = nullptr;
    ncclEpTensor_t* recv_weights_tensor = nullptr;
    ncclEpTensor_t* recv_idx_tensor = nullptr;
    NCCL_ASSERT(epTensorCreate(&tokens_tensor, 2, ncclFloat8e4m3, tokens, kNumTokens, kFp8Hidden));
    NCCL_ASSERT(epTensorCreate(&recv_tensor, 2, ncclFloat8e4m3, recv, kMaxRecvSlots, kFp8Hidden));
    NCCL_ASSERT(epTensorCreate(
        &scales_tensor,
        2,
        ncclFloat32,
        scales,
        kNumTokens,
        kScalesPerToken));
    NCCL_ASSERT(epTensorCreate(
        &recv_scales_tensor,
        2,
        ncclFloat32,
        recv_scales,
        kMaxRecvSlots,
        kScalesPerToken));
    NCCL_ASSERT(epTensorCreate(&weights_tensor, 2, ncclFloat32, weights, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(
        &recv_weights_tensor, 2, ncclFloat32, recv_weights, kMaxRecvSlots, kTopK));
    NCCL_ASSERT(epTensorCreate(
        &recv_idx_tensor, 2, ncclInt64, recv_idx, kMaxRecvSlots, kTopK));

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = tokens_tensor;
    inputs.scales = scales_tensor;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    inputs.topk_weights = weights_tensor;
    outputs.tokens = recv_tensor;
    outputs.scales = recv_scales_tensor;
    outputs.topk_weights = recv_weights_tensor;
    outputs.topk_idx = recv_idx_tensor;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    // FP8 scale capacity uses the BF16-equivalent hidden width.
    group_config.max_token_bytes = kFp8Hidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = NCCL_EP_AUTO;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.max_recv_tokens_per_rank = kMaxRecvSlots;
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_FLAT, topk_idx_, nullptr, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    NCCL_ASSERT(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<uint8_t> output_tokens(kMaxRecvSlots * kFp8Hidden);
    std::vector<uint8_t> output_scale_bytes(kMaxRecvSlots * kScalesPerToken * sizeof(float));
    float scale_value = 2.0f;
    uint8_t expected_scale_bytes[sizeof(float)];
    memcpy(expected_scale_bytes, &scale_value, sizeof(float));
    for (bool graph_replay : {false, true}) {
        if (graph_replay) {
            CUDA_ASSERT(cudaMemset(recv, 0xA5, kMaxRecvSlots * kFp8Hidden));
            CUDA_ASSERT(cudaMemset(recv_scales, 0xA5, output_scale_bytes.size()));
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t graph_exec = nullptr;
            CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
            NCCL_ASSERT(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream));
            CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
            CUDA_ASSERT(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
            CUDA_ASSERT(cudaGraphLaunch(graph_exec, g_stream));
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
            CUDA_ASSERT(cudaGraphExecDestroy(graph_exec));
            CUDA_ASSERT(cudaGraphDestroy(graph));
        }
        CUDA_ASSERT(cudaMemcpy(output_tokens.data(), recv, output_tokens.size(), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(
            output_scale_bytes.data(),
            recv_scales,
            output_scale_bytes.size(),
            cudaMemcpyDeviceToHost));
        for (int row = 0; row < kMaxRecvSlots; ++row) {
            if (row >= kNumTokens && graph_replay && !use_sm_copy()) continue;
            const uint8_t expected_token = row < kNumTokens ? kTokenValue : 0xA5;
            for (int byte = 0; byte < kFp8Hidden; ++byte) {
                EXPECT_EQ(output_tokens[row * kFp8Hidden + byte], expected_token);
            }
            for (int byte = 0; byte < kScalesPerToken * static_cast<int>(sizeof(float)); ++byte) {
                const uint8_t actual = output_scale_bytes[row * kScalesPerToken * sizeof(float) + byte];
                const uint8_t expected =
                    row < kNumTokens ? expected_scale_bytes[byte % sizeof(float)] : 0xA5;
                EXPECT_EQ(actual, expected);
            }
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
    ncclEpTensorDestroy(tokens_tensor);
    ncclEpTensorDestroy(recv_tensor);
    ncclEpTensorDestroy(scales_tensor);
    ncclEpTensorDestroy(recv_scales_tensor);
    ncclEpTensorDestroy(weights_tensor);
    ncclEpTensorDestroy(recv_weights_tensor);
    ncclEpTensorDestroy(recv_idx_tensor);
    cudaFree(tokens);
    cudaFree(recv);
    cudaFree(scales);
    cudaFree(recv_scales);
    cudaFree(weights);
    cudaFree(recv_weights);
    cudaFree(recv_idx);
}

INSTANTIATE_TEST_SUITE_P(CopyPolicy, HtDispatchAsyncTest,
                        ::testing::Values(nullptr, "", "CE", "ce", "SIMT", "sImT", "TMA", "tMa", "invalid", "0", "1", "2"));

} // namespace

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_dispatch_async_uid")) return 0;
    const int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
