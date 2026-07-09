/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_common.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

class QuantizationRecipeTest : public EpTestBase {};

struct RecipeTensor {
    ncclEpTensor_t tensor = NCCL_EP_TENSOR_INIT;
    void* data = nullptr;
    size_t sizes[2] = {kNumTokens, kHidden};

    RecipeTensor(ncclDataType_t dtype, size_t rows = kNumTokens, size_t cols = kHidden)
        : sizes{rows, cols} {
        if (cudaMalloc(&data, rows * cols * (dtype == ncclFloat32 ? sizeof(float) : 1)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed while creating quantization recipe test tensor");
        tensor.ndim = 2;
        tensor.datatype = dtype;
        tensor.data = data;
        tensor.sizes = sizes;
    }
    ~RecipeTensor() { if (data) cudaFree(data); }
};

static bool has_nonzero_bytes(const std::vector<uint8_t>& values) {
    return std::any_of(values.begin(), values.end(), [](uint8_t value) { return value != 0; });
}

static bool has_nonzero_scales(const std::vector<float>& values) {
    return std::any_of(values.begin(), values.end(), [](float value) { return value != 0.0f; });
}

TEST_F(QuantizationRecipeTest, ScalesForwardDispatchCompletes) {
    constexpr int kScalesPerToken = 4;

    uint8_t *d_tokens = nullptr, *d_recv_tokens = nullptr;
    float *d_scales = nullptr, *d_recv_scales = nullptr;
    float *d_topk_weights = nullptr, *d_recv_topk_weights = nullptr;
    int64_t* d_recv_topk_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tokens, kNumTokens * kHidden * sizeof(uint8_t)));
    CUDA_ASSERT(cudaMalloc(&d_recv_tokens, kMaxRecvSlots * kHidden * sizeof(uint8_t)));
    CUDA_ASSERT(cudaMalloc(&d_scales, kNumTokens * kScalesPerToken * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_scales, kMaxRecvSlots * kScalesPerToken * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_topk_weights, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_weights, kMaxRecvSlots * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)));

    std::vector<uint8_t> h_tokens(kNumTokens * kHidden);
    std::vector<float> h_scales(kNumTokens * kScalesPerToken);
    std::vector<float> h_topk_weights(kNumTokens * kTopK, 1.0f);
    for (size_t i = 0; i < h_tokens.size(); ++i) {
        h_tokens[i] = static_cast<uint8_t>(1 + g_rank * 17 + i);
    }
    for (size_t i = 0; i < h_scales.size(); ++i) {
        h_scales[i] = static_cast<float>(1 + g_rank * 100 + i);
    }
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_scales, h_scales.data(),
                           h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_topk_weights, h_topk_weights.data(),
                           h_topk_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv_tokens, 0, kMaxRecvSlots * kHidden * sizeof(uint8_t)));
    CUDA_ASSERT(cudaMemset(d_recv_scales, 0, kMaxRecvSlots * kScalesPerToken * sizeof(float)));

    ncclEpTensor_t *tokens = nullptr, *scales = nullptr, *topk_weights = nullptr;
    ncclEpTensor_t *recv_tokens = nullptr, *recv_scales = nullptr;
    ncclEpTensor_t *recv_topk_weights = nullptr, *recv_topk_idx = nullptr;
    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclFloat8e4m3, d_tokens, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&scales, 2, ncclFloat32, d_scales, kNumTokens, kScalesPerToken));
    NCCL_ASSERT(epTensorCreate(&topk_weights, 2, ncclFloat32, d_topk_weights, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&recv_tokens, 2, ncclFloat8e4m3,
                               d_recv_tokens, kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&recv_scales, 2, ncclFloat32,
                               d_recv_scales, kMaxRecvSlots, kScalesPerToken));
    NCCL_ASSERT(epTensorCreate(&recv_topk_weights, 2, ncclFloat32,
                               d_recv_topk_weights, kMaxRecvSlots, kTopK));
    NCCL_ASSERT(epTensorCreate(&recv_topk_idx, 2, ncclInt64,
                               d_recv_topk_idx, kMaxRecvSlots, kTopK));

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = tokens;
    inputs.scales = scales;
    inputs.topk_weights = topk_weights;
    outputs.tokens = recv_tokens;
    outputs.scales = recv_scales;
    outputs.topk_weights = recv_topk_weights;
    outputs.topk_idx = recv_topk_idx;
    config.quantization_recipe = NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD;

    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    std::vector<uint8_t> h_recv_tokens(kMaxRecvSlots * kHidden);
    std::vector<float> h_recv_scales(kMaxRecvSlots * kScalesPerToken);
    CUDA_ASSERT(cudaMemcpy(h_recv_tokens.data(), d_recv_tokens, h_recv_tokens.size(), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_recv_scales.data(), d_recv_scales,
                           h_recv_scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(has_nonzero_bytes(h_recv_tokens));
    EXPECT_TRUE(has_nonzero_scales(h_recv_scales));

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(scales);
    ncclEpTensorDestroy(topk_weights);
    ncclEpTensorDestroy(recv_tokens);
    ncclEpTensorDestroy(recv_scales);
    ncclEpTensorDestroy(recv_topk_weights);
    ncclEpTensorDestroy(recv_topk_idx);
    cudaFree(d_tokens);
    cudaFree(d_recv_tokens);
    cudaFree(d_scales);
    cudaFree(d_recv_scales);
    cudaFree(d_topk_weights);
    cudaFree(d_recv_topk_weights);
    cudaFree(d_recv_topk_idx);
}

TEST_F(QuantizationRecipeTest, DsFp8E3M4DispatchCompletes) {
    constexpr int kDsHidden = 512;
    constexpr int kDsScalesPerToken = kDsHidden / 128;
    ASSERT_EQ(kNumExperts % g_nranks, 0);
    const int num_local_experts = kNumExperts / g_nranks;
    const int recv_slots = g_nranks * kNumTokens;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    group_config.max_token_bytes = kDsHidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = num_local_experts;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.max_recv_tokens_per_rank = kNumTokens;

    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    std::vector<nv_bfloat16> h_tokens(kNumTokens * kDsHidden);
    for (size_t i = 0; i < h_tokens.size(); ++i) {
        h_tokens[i] = __float2bfloat16(static_cast<float>(1 + g_rank * 10 + (i % kDsHidden)));
    }

    nv_bfloat16* d_tokens = nullptr;
    uint8_t* d_recv_tokens = nullptr;
    float* d_recv_scales = nullptr;
    int* d_expert_counters = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv_tokens,
                           static_cast<size_t>(num_local_experts) * recv_slots * kDsHidden));
    CUDA_ASSERT(cudaMalloc(&d_recv_scales,
                           static_cast<size_t>(num_local_experts) * recv_slots *
                               kDsScalesPerToken * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_expert_counters, num_local_experts * sizeof(int)));
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(),
                           h_tokens.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv_tokens, 0,
                           static_cast<size_t>(num_local_experts) * recv_slots * kDsHidden));
    CUDA_ASSERT(cudaMemset(d_recv_scales, 0,
                           static_cast<size_t>(num_local_experts) * recv_slots *
                               kDsScalesPerToken * sizeof(float)));

    ncclEpTensor_t *tokens = nullptr, *recv_tokens = nullptr;
    ncclEpTensor_t *recv_scales = nullptr, *expert_counters = nullptr;
    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclBfloat16, d_tokens, kNumTokens, kDsHidden));
    NCCL_ASSERT(epTensorCreate(&recv_tokens, 3, ncclFloat8e4m3, d_recv_tokens,
                               num_local_experts, recv_slots, kDsHidden));
    NCCL_ASSERT(epTensorCreate(&recv_scales, 3, ncclFloat32, d_recv_scales,
                               num_local_experts, recv_slots, kDsScalesPerToken));
    NCCL_ASSERT(epTensorCreate(&expert_counters, 1, ncclInt32,
                               d_expert_counters, num_local_experts));

    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                   topk_idx_em_, nullptr, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = tokens;
    outputs.tokens = recv_tokens;
    outputs.scales = recv_scales;
    layout_info.expert_counters = expert_counters;
    config.quantization_recipe = NCCL_EP_DISPATCH_QUANT_DS_FP8E3M4;

    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, &layout_info, &config, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    std::vector<int> h_expert_counters(num_local_experts);
    std::vector<float> h_recv_scales(
        static_cast<size_t>(num_local_experts) * recv_slots * kDsScalesPerToken);
    CUDA_ASSERT(cudaMemcpy(h_expert_counters.data(), d_expert_counters,
                           h_expert_counters.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_recv_scales.data(), d_recv_scales,
                           h_recv_scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_GT(std::accumulate(h_expert_counters.begin(), h_expert_counters.end(), 0), 0);
    EXPECT_TRUE(has_nonzero_scales(h_recv_scales));

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(recv_tokens);
    ncclEpTensorDestroy(recv_scales);
    ncclEpTensorDestroy(expert_counters);
    cudaFree(d_tokens);
    cudaFree(d_recv_tokens);
    cudaFree(d_recv_scales);
    cudaFree(d_expert_counters);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(QuantizationRecipeTest, DispatchNoneRejectsScaleTensors) {
    RecipeTensor tokens(ncclBfloat16);
    RecipeTensor output_tokens(ncclBfloat16);
    RecipeTensor scales(ncclFloat32, kNumTokens, kHidden / 128);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &scales.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRequiresInputAndOutputScales) {
    RecipeTensor tokens(ncclFloat8e4m3);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quantization_recipe = NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD;
    inputs.tokens = &tokens.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsNonFloatScales) {
    RecipeTensor tokens(ncclFloat8e4m3);
    RecipeTensor output_tokens(ncclFloat8e4m3);
    RecipeTensor input_scales(ncclUint8, kNumTokens, kHidden / 128);
    RecipeTensor output_scales(ncclFloat32, kNumTokens, kHidden / 128);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quantization_recipe = NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &input_scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, CombineNoneRejectsDispatchWireDtype) {
    RecipeTensor tokens(ncclFloat8e4m3);
    ncclEpCombineInputs_t inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    inputs.tokens = &tokens.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpCombine(handle, &inputs, &outputs, nullptr, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_quantization_recipe_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
