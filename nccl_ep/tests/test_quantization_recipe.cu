/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_common.h"
#include "quantization_recipe.hpp"
#include "../nccl_ep_test_internal.h"
#include "nccl_ep/common.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

class QuantizationRecipeTest : public EpTestBase {};

// Per-token scale count for the recipe tests. kHidden (16) is not divisible by a
// realistic quantization block size, so use a fixed non-zero count: 4 scales/token
// (block size 4), which keeps scale bytes (4*4=16) 16-byte aligned.
static constexpr int kScalesPerToken = 4;

// Bytes per element for the dtypes exercised by the recipe tests. Kept local
// because ncclTypeSize is not exposed to the test translation units.
static size_t recipe_elem_bytes(ncclDataType_t dtype) {
    switch (dtype) {
        case ncclFloat32:
            return 4;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclFloat8e4m3:
        case ncclFloat8e5m2:
        case ncclInt8:
        case ncclUint8:
        case ncclFloat4x2:
            return 1;
        case ncclInt64:
            return 8;
        default:
            throw std::runtime_error("unsupported dtype " + std::to_string(static_cast<int>(dtype)));
    }
}

struct RecipeTensor {
    ncclEpTensor_t tensor = NCCL_EP_TENSOR_INIT;
    void* data = nullptr;
    size_t sizes[2] = {kNumTokens, kHidden};

    RecipeTensor(ncclDataType_t dtype, size_t rows = kNumTokens, size_t cols = kHidden)
        : sizes{rows, cols} {
        if (cudaMalloc(&data, rows * cols * recipe_elem_bytes(dtype)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed while creating quantization recipe test tensor");
        tensor.ndim = 2;
        tensor.datatype = dtype;
        tensor.data = data;
        tensor.sizes = sizes;
    }
    ~RecipeTensor() { if (data) cudaFree(data); }
};

enum class MisalignedRecipeStorage { InputTokens, InputScales, OutputTokens, OutputScales };

static void expect_invalid_misaligned_recipe_storage(
    ncclEpHandle_t handle,
    MisalignedRecipeStorage field) {
    RecipeTensor tokens(ncclFloat8e4m3, kNumTokens, 16);
    RecipeTensor scales(ncclUint8, kNumTokens, 16);
    RecipeTensor output_tokens(ncclFloat8e4m3, kMaxRecvSlots, 16);
    RecipeTensor output_scales(ncclUint8, kMaxRecvSlots, 16);
    ncclEpTensor_t* target = field == MisalignedRecipeStorage::InputTokens ? &tokens.tensor :
        field == MisalignedRecipeStorage::InputScales ? &scales.tensor :
        field == MisalignedRecipeStorage::OutputTokens ? &output_tokens.tensor : &output_scales.tensor;
    target->data = static_cast<uint8_t*>(target->data) + 1;

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
}

static void expect_invalid_ht_scales_forward_output(
    ncclEpHandle_t handle,
    ncclDataType_t output_token_dtype,
    size_t output_token_rows,
    size_t output_token_cols,
    size_t output_scale_rows,
    size_t output_scale_cols) {
    RecipeTensor tokens(ncclFloat8e4m3, kNumTokens, 16);
    RecipeTensor scales(ncclUint8, kNumTokens, 16);
    RecipeTensor topk_weights(ncclFloat32, kNumTokens, kTopK);
    RecipeTensor output_tokens(output_token_dtype, output_token_rows, output_token_cols);
    RecipeTensor output_scales(ncclUint8, output_scale_rows, output_scale_cols);
    RecipeTensor output_topk_weights(ncclFloat32, kMaxRecvSlots, kTopK);
    RecipeTensor output_topk_idx(ncclInt64, kMaxRecvSlots, kTopK);

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &scales.tensor;
    inputs.topk_weights = &topk_weights.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    outputs.topk_weights = &output_topk_weights.tensor;
    outputs.topk_idx = &output_topk_idx.tensor;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
}

static bool has_nonzero_bytes(const std::vector<uint8_t>& values) {
    return std::any_of(values.begin(), values.end(), [](uint8_t value) { return value != 0; });
}

static bool has_nonzero_scales(const std::vector<float>& values) {
    return std::any_of(values.begin(), values.end(), [](float value) { return value != 0.0f; });
}

static bool nvfp4_supported() {
    static const bool supported = [] {
        int device = 0;
        int major = 0;
        int minor = 0;
        return cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
            nccl_ep::host_device_supports_fp4(static_cast<unsigned int>(major * 10 + minor));
    }();
    return supported;
}

TEST(NvFp4SupportTest, MapsDocumentedFamilyTargets) {
    EXPECT_FALSE(nccl_ep::host_device_supports_fp4(90));
    EXPECT_EQ(nccl_ep::host_device_fp4_target_arch(100), "sm_100f");
    EXPECT_EQ(nccl_ep::host_device_fp4_target_arch(103), "sm_100f");
    EXPECT_EQ(nccl_ep::host_device_fp4_target_arch(110), "sm_110f");
    EXPECT_EQ(nccl_ep::host_device_fp4_target_arch(120), "sm_120f");
    EXPECT_EQ(nccl_ep::host_device_fp4_target_arch(121), "sm_120f");
    EXPECT_FALSE(nccl_ep::host_device_supports_fp4(107));
}

// These are exact E2M1 values. Every 16-element block contains +/-6, so the
// per-row global scale below follows the NVFP4 contract exactly.
// Keeping the reference in an exactly representable subset makes it independent
// of the implementation's reciprocal approximation and rounding details.
static float nvfp4_test_value(int token, int channel) {
    constexpr float kValues[] = {
        -6.f, -4.f, -3.f, -2.f, -1.5f, -1.f, -0.5f, 0.f,
        0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -6.f};
    return kValues[(token * 5 + channel) % (sizeof(kValues) / sizeof(kValues[0]))];
}

static float nvfp4_test_global_scale(int token, int hidden) {
    float amax = 0.f;
    for (int channel = 0; channel < hidden; ++channel) {
        amax = std::max(amax, std::abs(nvfp4_test_value(token, channel)));
    }
    return amax == 0.f ? 0.f : 2688.f / amax;
}

static void fill_nvfp4_test_tokens(std::vector<nv_bfloat16>* tokens, int hidden) {
    for (size_t i = 0; i < tokens->size(); ++i) {
        const int token = i / hidden;
        const int channel = i % hidden;
        (*tokens)[i] = __float2bfloat16(nvfp4_test_value(token, channel));
    }
}

static uint8_t ht_em_token_byte(int src_rank, int token, int byte) {
    if (byte == 0) return static_cast<uint8_t>(0x40 + src_rank);
    if (byte == 1) return static_cast<uint8_t>(1 + token);
    return static_cast<uint8_t>(1 + (src_rank * 67 + token * 23 + byte * 3) % 251);
}

static uint8_t ht_em_scale_byte(int src_rank, int token, int byte) {
    if (byte == 0) return static_cast<uint8_t>(0x80 + src_rank);
    if (byte == 1) return static_cast<uint8_t>(1 + token);
    return static_cast<uint8_t>(1 + (src_rank * 43 + token * 31 + byte * 7) % 251);
}

static ncclResult_t attach_recipe_window(
    ncclEpTensor_t* tensor,
    void* data,
    size_t bytes,
    ncclWindow_t* window) {
    ncclResult_t result =
        ncclCommWindowRegister(g_comm, data, bytes, window, NCCL_WIN_COLL_SYMMETRIC);
    if (result != ncclSuccess) return result;
    tensor->win_hdl = *window;
    tensor->win_offset = 0;
    return ncclSuccess;
}

TEST_F(QuantizationRecipeTest, ScalesForwardResolverAcceptsBf16TokenWire) {
    nccl_ep::DispatchKernelSpec spec{};
    EXPECT_EQ(
        nccl_ep::resolveDispatchKernelSpec(
            NCCL_EP_DISP_QUANT_FWD, ncclBfloat16, ncclFloat32, &spec),
        ncclSuccess);
    EXPECT_EQ(spec.wire_token_dtype, ncclBfloat16);
    EXPECT_EQ(spec.payload_bytes, sizeof(uint16_t));
    EXPECT_STREQ(spec.payload_type_literal, "uint16_t");
}

TEST_F(QuantizationRecipeTest, HtGroupRejectsUnalignedMaxTokenBytes) {
    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    group_config.max_recv_tokens_per_rank = kMaxRecvSlots;
    group_config.max_token_bytes = sizeof(int4) * 2 + 1;
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = NCCL_EP_AUTO;
    group_config.num_channels = NCCL_EP_AUTO;

    ncclEpGroup_t group = nullptr;
    EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &group_config), ncclInvalidArgument);
    EXPECT_EQ(group, nullptr);
}

TEST_F(QuantizationRecipeTest, ScalesForwardDispatchCompletes) {
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
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;

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
    uint8_t* d_recv_scales_storage = nullptr;
    float* d_recv_scales = nullptr;
    int* d_expert_counters = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv_tokens,
                           static_cast<size_t>(num_local_experts) * recv_slots * kDsHidden));
    const size_t recv_scale_bytes = static_cast<size_t>(num_local_experts) * recv_slots *
        kDsScalesPerToken * sizeof(float);
    CUDA_ASSERT(cudaMalloc(&d_recv_scales_storage, recv_scale_bytes + sizeof(float)));
    d_recv_scales = reinterpret_cast<float*>(d_recv_scales_storage + sizeof(float));
    CUDA_ASSERT(cudaMalloc(&d_expert_counters, num_local_experts * sizeof(int)));
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(),
                           h_tokens.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv_tokens, 0,
                           static_cast<size_t>(num_local_experts) * recv_slots * kDsHidden));
    CUDA_ASSERT(cudaMemset(d_recv_scales, 0, recv_scale_bytes));

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
    config.quant_recipe = NCCL_EP_DISP_QUANT_DS_FP8E3M4;

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
    cudaFree(d_recv_scales_storage);
    cudaFree(d_expert_counters);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(QuantizationRecipeTest, ScalesForwardDispatchPreservesPackedFp4Bytes) {
    constexpr int kOriginalHidden = 512;
    constexpr int kPackedHidden = kOriginalHidden / 2;
    constexpr int kScalesPerToken = kOriginalHidden / 16;
    ASSERT_EQ(kPackedHidden / 8, kScalesPerToken);
    ASSERT_EQ(kNumExperts % g_nranks, 0);
    const int num_local_experts = kNumExperts / g_nranks;
    const int recv_slots = g_nranks * kNumTokens;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    group_config.max_token_bytes = kOriginalHidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = num_local_experts;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.max_recv_tokens_per_rank = kNumTokens;

    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    std::vector<uint8_t> h_tokens(kNumTokens * kPackedHidden);
    std::vector<uint8_t> h_scales(kNumTokens * kScalesPerToken);
    for (int token = 0; token < kNumTokens; ++token) {
        for (int byte = 0; byte < kPackedHidden; ++byte) {
            h_tokens[token * kPackedHidden + byte] =
                static_cast<uint8_t>((g_rank * 53 + token * 11 + byte) & 0xff);
        }
        for (int scale = 0; scale < kScalesPerToken; ++scale) {
            h_scales[token * kScalesPerToken + scale] =
                static_cast<uint8_t>((g_rank * 29 + token * 7 + scale) & 0xff);
        }
    }

    uint8_t *d_tokens = nullptr, *d_scales = nullptr;
    uint8_t *d_recv_tokens = nullptr, *d_recv_scales = nullptr;
    int* d_expert_counters = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tokens, h_tokens.size()));
    CUDA_ASSERT(cudaMalloc(&d_scales, h_scales.size()));
    CUDA_ASSERT(cudaMalloc(
        &d_recv_tokens, static_cast<size_t>(num_local_experts) * recv_slots * kPackedHidden));
    CUDA_ASSERT(cudaMalloc(
        &d_recv_scales, static_cast<size_t>(num_local_experts) * recv_slots * kScalesPerToken));
    CUDA_ASSERT(cudaMalloc(&d_expert_counters, num_local_experts * sizeof(int)));
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(
        d_recv_tokens, 0, static_cast<size_t>(num_local_experts) * recv_slots * kPackedHidden));
    CUDA_ASSERT(cudaMemset(
        d_recv_scales, 0, static_cast<size_t>(num_local_experts) * recv_slots * kScalesPerToken));

    ncclEpTensor_t *tokens = nullptr, *scales = nullptr;
    ncclEpTensor_t *recv_tokens = nullptr, *recv_scales = nullptr, *expert_counters = nullptr;
    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclFloat4x2, d_tokens, kNumTokens, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&scales, 2, ncclUint8, d_scales, kNumTokens, kScalesPerToken));
    NCCL_ASSERT(epTensorCreate(&recv_tokens, 3, ncclFloat4x2, d_recv_tokens,
                               num_local_experts, recv_slots, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&recv_scales, 3, ncclUint8, d_recv_scales,
                               num_local_experts, recv_slots, kScalesPerToken));
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
    inputs.scales = scales;
    outputs.tokens = recv_tokens;
    outputs.scales = recv_scales;
    layout_info.expert_counters = expert_counters;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;

    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, &layout_info, &config, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    std::vector<uint8_t> h_recv_tokens(
        static_cast<size_t>(num_local_experts) * recv_slots * kPackedHidden);
    std::vector<uint8_t> h_recv_scales(
        static_cast<size_t>(num_local_experts) * recv_slots * kScalesPerToken);
    std::vector<int> h_expert_counters(num_local_experts);
    CUDA_ASSERT(cudaMemcpy(h_recv_tokens.data(), d_recv_tokens, h_recv_tokens.size(), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_recv_scales.data(), d_recv_scales, h_recv_scales.size(), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_expert_counters.data(), d_expert_counters,
                           h_expert_counters.size() * sizeof(int), cudaMemcpyDeviceToHost));

    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        const int global_expert = g_rank * num_local_experts + local_expert;
        std::vector<bool> seen(static_cast<size_t>(g_nranks) * kNumTokens, false);
        int expected_count = 0;
        for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
            for (int token = 0; token < kNumTokens; ++token) {
                if ((src_rank * kNumTokens + token) % kNumExperts == global_expert) {
                    ++expected_count;
                }
            }
        }
        ASSERT_EQ(h_expert_counters[local_expert], expected_count);
        for (int slot = 0; slot < expected_count; ++slot) {
            const size_t recv_token_offset =
                (static_cast<size_t>(local_expert) * recv_slots + slot) * kPackedHidden;
            const size_t recv_scale_offset =
                (static_cast<size_t>(local_expert) * recv_slots + slot) * kScalesPerToken;
            bool matched = false;
            for (int src_rank = 0; src_rank < g_nranks && !matched; ++src_rank) {
                for (int token = 0; token < kNumTokens && !matched; ++token) {
                    if ((src_rank * kNumTokens + token) % kNumExperts != global_expert) continue;
                    const size_t pattern_index = static_cast<size_t>(src_rank) * kNumTokens + token;
                    if (seen[pattern_index]) continue;
                    bool token_matches = true;
                    for (int byte = 0; byte < kPackedHidden; ++byte) {
                        const uint8_t expected =
                            static_cast<uint8_t>((src_rank * 53 + token * 11 + byte) & 0xff);
                        if (h_recv_tokens[recv_token_offset + byte] != expected) {
                            token_matches = false;
                            break;
                        }
                    }
                    if (!token_matches) continue;
                    for (int scale = 0; scale < kScalesPerToken; ++scale) {
                        const uint8_t expected =
                            static_cast<uint8_t>((src_rank * 29 + token * 7 + scale) & 0xff);
                        EXPECT_EQ(h_recv_scales[recv_scale_offset + scale], expected);
                    }
                    seen[pattern_index] = true;
                    matched = true;
                }
            }
            EXPECT_TRUE(matched) << "unexpected packed FP4 payload for expert " << global_expert
                                 << " slot " << slot;
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(scales);
    ncclEpTensorDestroy(recv_tokens);
    ncclEpTensorDestroy(recv_scales);
    ncclEpTensorDestroy(expert_counters);
    cudaFree(d_tokens);
    cudaFree(d_scales);
    cudaFree(d_recv_tokens);
    cudaFree(d_recv_scales);
    cudaFree(d_expert_counters);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

static void run_ht_expert_major_scales_forward_packed_fp4(
    bool windowed_outputs,
    ncclEpZeroCopyMode_t zero_copy_mode) {
    if (g_nranks != 4) GTEST_SKIP() << "requires exactly four ranks";

    constexpr int kHtTokens = 16;
    constexpr int kTopK2 = 2;
    constexpr int kLogicalHidden = 256;
    constexpr int kPackedHidden = kLogicalHidden / 2;
    constexpr int kScaleBytes = kLogicalHidden / 16;
    constexpr int kExpertsPerRank = 2;
    constexpr int kAlignment = 32;
    constexpr int kActivePerExpert = kHtTokens;
    constexpr int kOutputRows = kExpertsPerRank * kAlignment;
    constexpr int kMaxRecvRows = 4 * kHtTokens * kTopK2;
    constexpr int kIterations = 3;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kHtTokens;
    group_config.max_recv_tokens_per_rank = kMaxRecvRows;
    group_config.max_token_bytes = kLogicalHidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = NCCL_EP_AUTO;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.num_topk = kTopK2;
    group_config.zero_copy = zero_copy_mode;

    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    std::vector<int64_t> h_topk(static_cast<size_t>(kHtTokens) * kTopK2);
    std::vector<float> h_topk_weights(static_cast<size_t>(kHtTokens) * kTopK2, 1.0f);
    std::vector<uint8_t> h_tokens(static_cast<size_t>(kHtTokens) * kPackedHidden);
    std::vector<uint8_t> h_scales(static_cast<size_t>(kHtTokens) * kScaleBytes);
    for (int token = 0; token < kHtTokens; ++token) {
        const int dst_rank = token % 4;
        h_topk[token * kTopK2] = dst_rank * kExpertsPerRank;
        h_topk[token * kTopK2 + 1] = dst_rank * kExpertsPerRank + 1;
        for (int byte = 0; byte < kPackedHidden; ++byte) {
            h_tokens[token * kPackedHidden + byte] = ht_em_token_byte(g_rank, token, byte);
        }
        for (int byte = 0; byte < kScaleBytes; ++byte) {
            h_scales[token * kScaleBytes + byte] = ht_em_scale_byte(g_rank, token, byte);
        }
    }

    int64_t* d_topk = nullptr;
    float *d_topk_weights = nullptr, *d_recv_topk_weights = nullptr;
    uint8_t *d_tokens = nullptr, *d_scales = nullptr;
    uint8_t *d_recv_tokens = nullptr, *d_recv_scales = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk, h_topk.size() * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_topk_weights, h_topk_weights.size() * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_weights, kMaxRecvRows * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_tokens, h_tokens.size()));
    CUDA_ASSERT(cudaMalloc(&d_scales, h_scales.size()));
    const size_t recv_token_bytes = static_cast<size_t>(kMaxRecvRows) * kPackedHidden;
    const size_t recv_scale_bytes = static_cast<size_t>(kMaxRecvRows) * kScaleBytes;
    if (windowed_outputs) {
        NCCL_ASSERT(ncclMemAlloc(reinterpret_cast<void**>(&d_recv_tokens), recv_token_bytes));
        NCCL_ASSERT(ncclMemAlloc(reinterpret_cast<void**>(&d_recv_scales), recv_scale_bytes));
    } else {
        CUDA_ASSERT(cudaMalloc(&d_recv_tokens, recv_token_bytes));
        CUDA_ASSERT(cudaMalloc(&d_recv_scales, recv_scale_bytes));
    }
    CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(
        d_topk_weights,
        h_topk_weights.data(),
        h_topk_weights.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));

    ncclEpTensor_t *topk = nullptr, *topk_weights = nullptr, *recv_topk_weights = nullptr;
    ncclEpTensor_t *tokens = nullptr, *scales = nullptr, *recv_tokens = nullptr, *recv_scales = nullptr;
    NCCL_ASSERT(epTensorCreate(&topk, 2, ncclInt64, d_topk, kHtTokens, kTopK2));
    NCCL_ASSERT(epTensorCreate(&topk_weights, 2, ncclFloat32, d_topk_weights, kHtTokens, kTopK2));
    NCCL_ASSERT(epTensorCreate(&recv_topk_weights, 1, ncclFloat32, d_recv_topk_weights, kMaxRecvRows));
    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclFloat4x2, d_tokens, kHtTokens, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&scales, 2, ncclUint8, d_scales, kHtTokens, kScaleBytes));
    NCCL_ASSERT(epTensorCreate(
        &recv_tokens,
        2,
        ncclFloat4x2,
        windowed_outputs ? nullptr : d_recv_tokens,
        kMaxRecvRows,
        kPackedHidden));
    NCCL_ASSERT(epTensorCreate(
        &recv_scales,
        2,
        ncclUint8,
        windowed_outputs ? nullptr : d_recv_scales,
        kMaxRecvRows,
        kScaleBytes));

    ncclWindow_t recv_token_window{};
    ncclWindow_t recv_scale_window{};
    if (windowed_outputs) {
        NCCL_ASSERT(attach_recipe_window(
            recv_tokens, d_recv_tokens, recv_token_bytes, &recv_token_window));
        NCCL_ASSERT(attach_recipe_window(
            recv_scales, d_recv_scales, recv_scale_bytes, &recv_scale_window));
    }

    ncclEpHandleConfig_t handle_config = NCCL_EP_HANDLE_CONFIG_INIT;
    handle_config.dispatch_output_per_expert_alignment = kAlignment;
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(
        &handle,
        group,
        NCCL_EP_LAYOUT_EXPERT_MAJOR,
        topk,
        nullptr,
        &handle_config,
        g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = tokens;
    inputs.scales = scales;
    inputs.topk_weights = topk_weights;
    outputs.tokens = recv_tokens;
    outputs.scales = recv_scales;
    outputs.topk_weights = recv_topk_weights;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;

    std::vector<uint8_t> h_recv_tokens(static_cast<size_t>(kOutputRows) * kPackedHidden);
    std::vector<uint8_t> h_recv_scales(static_cast<size_t>(kOutputRows) * kScaleBytes);
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        SCOPED_TRACE(::testing::Message() << "iteration " << iteration);
        CUDA_ASSERT(cudaMemset(d_recv_tokens, 0xa5, static_cast<size_t>(kMaxRecvRows) * kPackedHidden));
        CUDA_ASSERT(cudaMemset(d_recv_scales, 0x5a, static_cast<size_t>(kMaxRecvRows) * kScaleBytes));
        CUDA_ASSERT(cudaMemset(d_recv_topk_weights, 0x7f, kMaxRecvRows * sizeof(float)));

        ASSERT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclSuccess);
        ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        CUDA_ASSERT(cudaMemcpy(
            h_recv_tokens.data(),
            d_recv_tokens,
            h_recv_tokens.size(),
            cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(
            h_recv_scales.data(),
            d_recv_scales,
            h_recv_scales.size(),
            cudaMemcpyDeviceToHost));

        for (int local_expert = 0; local_expert < kExpertsPerRank; ++local_expert) {
            const int zone_start = local_expert * kAlignment;
            std::vector<bool> seen(static_cast<size_t>(g_nranks) * kHtTokens, false);
            for (int slot = 0; slot < kActivePerExpert; ++slot) {
                const size_t row = static_cast<size_t>(zone_start + slot);
                bool matched = false;
                for (int src_rank = 0; src_rank < g_nranks && !matched; ++src_rank) {
                    for (int token = g_rank; token < kHtTokens && !matched; token += g_nranks) {
                        const size_t pattern_index = static_cast<size_t>(src_rank) * kHtTokens + token;
                        if (seen[pattern_index]) continue;
                        bool token_matches = true;
                        for (int byte = 0; byte < kPackedHidden; ++byte) {
                            if (h_recv_tokens[row * kPackedHidden + byte] !=
                                ht_em_token_byte(src_rank, token, byte)) {
                                token_matches = false;
                                break;
                            }
                        }
                        if (!token_matches) continue;
                        for (int byte = 0; byte < kScaleBytes; ++byte) {
                            EXPECT_EQ(
                                h_recv_scales[row * kScaleBytes + byte],
                                ht_em_scale_byte(src_rank, token, byte))
                                << "expert " << local_expert << " slot " << slot
                                << " scale byte " << byte;
                        }
                        seen[pattern_index] = true;
                        matched = true;
                    }
                }
                EXPECT_TRUE(matched) << "unmatched expert " << local_expert << " slot " << slot;
            }

            for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
                for (int token = g_rank; token < kHtTokens; token += g_nranks) {
                    EXPECT_TRUE(seen[static_cast<size_t>(src_rank) * kHtTokens + token])
                        << "missing source rank " << src_rank << " token " << token
                        << " from expert " << local_expert;
                }
            }

            for (int slot = kActivePerExpert; slot < kAlignment; ++slot) {
                const size_t row = static_cast<size_t>(zone_start + slot);
                for (int byte = 0; byte < kPackedHidden; ++byte) {
                    EXPECT_EQ(h_recv_tokens[row * kPackedHidden + byte], 0)
                        << "expert " << local_expert << " padding slot " << slot
                        << " token byte " << byte;
                }
                for (int byte = 0; byte < kScaleBytes; ++byte) {
                    EXPECT_EQ(h_recv_scales[row * kScaleBytes + byte], 0)
                        << "expert " << local_expert << " padding slot " << slot
                        << " scale byte " << byte;
                }
            }
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    if (windowed_outputs) {
        NCCL_ASSERT(ncclCommWindowDeregister(g_comm, recv_scale_window));
        NCCL_ASSERT(ncclCommWindowDeregister(g_comm, recv_token_window));
    }
    ncclEpTensorDestroy(topk);
    ncclEpTensorDestroy(topk_weights);
    ncclEpTensorDestroy(recv_topk_weights);
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(scales);
    ncclEpTensorDestroy(recv_tokens);
    ncclEpTensorDestroy(recv_scales);
    cudaFree(d_topk);
    cudaFree(d_topk_weights);
    cudaFree(d_recv_topk_weights);
    cudaFree(d_tokens);
    cudaFree(d_scales);
    if (windowed_outputs) {
        NCCL_ASSERT(ncclMemFree(d_recv_tokens));
        NCCL_ASSERT(ncclMemFree(d_recv_scales));
    } else {
        cudaFree(d_recv_tokens);
        cudaFree(d_recv_scales);
    }
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(QuantizationRecipeTest, HtExpertMajorScalesForwardPreservesPackedFp4Rows) {
    run_ht_expert_major_scales_forward_packed_fp4(
        /*windowed_outputs=*/false,
        NCCL_EP_ZERO_COPY_AUTO);
}

TEST_F(QuantizationRecipeTest, HtExpertMajorScalesForwardZeroCopyPreservesPackedFp4RowsAndScales) {
    run_ht_expert_major_scales_forward_packed_fp4(
        /*windowed_outputs=*/true,
        NCCL_EP_ZERO_COPY_ON);
}

TEST_F(QuantizationRecipeTest, HtExpertMajorScalesForwardWindowedAutoPreservesPackedFp4RowsAndScales) {
    run_ht_expert_major_scales_forward_packed_fp4(
        /*windowed_outputs=*/true,
        NCCL_EP_ZERO_COPY_AUTO);
}

// Eager (max_recv_tokens_per_rank = NCCL_EP_AUTO) expert-major scales-forward dispatch
// must accept recv token and scale buffers sized to the per-step recv count, which is
// below the worst-case max_recv_tokens budget.
static void run_ht_em_scales_forward_eager_below_budget() {
    if (g_nranks != 4) GTEST_SKIP() << "requires exactly four ranks";

    constexpr int kHtTokens = 16;
    constexpr int kTopK2 = 2;
    constexpr int kLogicalHidden = 256;
    constexpr int kPackedHidden = kLogicalHidden / 2;
    constexpr int kScaleBytes = kLogicalHidden / 16;
    constexpr int kExpertsPerRank = 2;
    constexpr int kAlignment = 32;
    // Worst-case per-rank recv budget the eager mode derives internally.
    constexpr int kBudget = 4 * kHtTokens * kTopK2;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kHtTokens;
    group_config.max_recv_tokens_per_rank = NCCL_EP_AUTO;  // eager
    group_config.max_token_bytes = kLogicalHidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = NCCL_EP_AUTO;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.num_topk = kTopK2;
    group_config.zero_copy = NCCL_EP_ZERO_COPY_AUTO;

    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    // Balanced routing: every token targets its dst_rank's two local experts, so
    // each rank's padded recv total stays well below the budget.
    std::vector<int64_t> h_topk(static_cast<size_t>(kHtTokens) * kTopK2);
    std::vector<float> h_topk_weights(static_cast<size_t>(kHtTokens) * kTopK2, 1.0f);
    std::vector<uint8_t> h_tokens(static_cast<size_t>(kHtTokens) * kPackedHidden);
    std::vector<uint8_t> h_scales(static_cast<size_t>(kHtTokens) * kScaleBytes);
    for (int token = 0; token < kHtTokens; ++token) {
        const int dst_rank = token % 4;
        h_topk[token * kTopK2] = dst_rank * kExpertsPerRank;
        h_topk[token * kTopK2 + 1] = dst_rank * kExpertsPerRank + 1;
        for (int byte = 0; byte < kPackedHidden; ++byte)
            h_tokens[token * kPackedHidden + byte] = ht_em_token_byte(g_rank, token, byte);
        for (int byte = 0; byte < kScaleBytes; ++byte)
            h_scales[token * kScaleBytes + byte] = ht_em_scale_byte(g_rank, token, byte);
    }

    int64_t* d_topk = nullptr;
    float* d_topk_weights = nullptr;
    uint8_t *d_tokens = nullptr, *d_scales = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk, h_topk.size() * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_topk_weights, h_topk_weights.size() * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_tokens, h_tokens.size()));
    CUDA_ASSERT(cudaMalloc(&d_scales, h_scales.size()));
    CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(
        d_topk_weights, h_topk_weights.data(), h_topk_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));

    ncclEpTensor_t *topk = nullptr, *topk_weights = nullptr, *tokens = nullptr, *scales = nullptr;
    NCCL_ASSERT(epTensorCreate(&topk, 2, ncclInt64, d_topk, kHtTokens, kTopK2));
    NCCL_ASSERT(epTensorCreate(&topk_weights, 2, ncclFloat32, d_topk_weights, kHtTokens, kTopK2));
    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclFloat4x2, d_tokens, kHtTokens, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&scales, 2, ncclUint8, d_scales, kHtTokens, kScaleBytes));

    ncclEpHandleConfig_t handle_config = NCCL_EP_HANDLE_CONFIG_INIT;
    handle_config.dispatch_output_per_expert_alignment = kAlignment;
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk, nullptr, &handle_config, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Per-step recv count the eager caller sizes its recv buffers from.
    unsigned int recv_rows = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(handle, &recv_rows));
    ASSERT_GT(recv_rows, 0u);
    ASSERT_LT(recv_rows, static_cast<unsigned int>(kBudget))
        << "routing must leave the per-step recv count below the budget for this regression";

    uint8_t *d_recv_tokens = nullptr, *d_recv_scales = nullptr;
    float* d_recv_topk_weights = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_recv_tokens, static_cast<size_t>(recv_rows) * kPackedHidden));
    CUDA_ASSERT(cudaMalloc(&d_recv_scales, static_cast<size_t>(recv_rows) * kScaleBytes));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_weights, static_cast<size_t>(recv_rows) * sizeof(float)));

    ncclEpTensor_t *recv_tokens = nullptr, *recv_scales = nullptr, *recv_topk_weights = nullptr;
    NCCL_ASSERT(epTensorCreate(&recv_tokens, 2, ncclFloat4x2, d_recv_tokens, recv_rows, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&recv_scales, 2, ncclUint8, d_recv_scales, recv_rows, kScaleBytes));
    NCCL_ASSERT(epTensorCreate(&recv_topk_weights, 1, ncclFloat32, d_recv_topk_weights, recv_rows));

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = tokens;
    inputs.scales = scales;
    inputs.topk_weights = topk_weights;
    outputs.tokens = recv_tokens;
    outputs.scales = recv_scales;
    outputs.topk_weights = recv_topk_weights;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;

    ASSERT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclSuccess);
    ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(topk);
    ncclEpTensorDestroy(topk_weights);
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(scales);
    ncclEpTensorDestroy(recv_tokens);
    ncclEpTensorDestroy(recv_scales);
    ncclEpTensorDestroy(recv_topk_weights);
    cudaFree(d_topk);
    cudaFree(d_topk_weights);
    cudaFree(d_tokens);
    cudaFree(d_scales);
    cudaFree(d_recv_tokens);
    cudaFree(d_recv_scales);
    cudaFree(d_recv_topk_weights);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(QuantizationRecipeTest, HtExpertMajorScalesForwardEagerAcceptsBelowBudgetRecvScales) {
    run_ht_em_scales_forward_eager_below_budget();
}

// Zero-recv rank under eager + QUANT_FWD: a rank that receives no tokens sizes
// all three recv outputs to 0 rows, so their descriptors are present but empty
// (data == nullptr, as a 0-byte framework allocation hands them over). The scale
// triple passed to the permute launcher must then be fully suppressed —
// otherwise the empty caller buffer pairs with the non-empty internal FLAT scale
// staging and trips the launcher's scale-pairing assert.
static void run_ht_em_scales_forward_eager_zero_recv() {
    if (g_nranks != 4) GTEST_SKIP() << "requires exactly four ranks";

    constexpr int kHtTokens = 16;
    constexpr int kTopK1 = 1;
    constexpr int kLogicalHidden = 256;
    constexpr int kPackedHidden = kLogicalHidden / 2;
    constexpr int kScaleBytes = kLogicalHidden / 16;
    constexpr int kAlignment = 32;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kHtTokens;
    group_config.max_recv_tokens_per_rank = NCCL_EP_AUTO;  // eager
    group_config.max_token_bytes = kLogicalHidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = NCCL_EP_AUTO;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.num_topk = kTopK1;
    group_config.zero_copy = NCCL_EP_ZERO_COPY_AUTO;

    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    // Skewed routing: every token on every rank targets global expert 0, so rank 0
    // receives all 4 * kHtTokens rows and ranks 1-3 receive none.
    std::vector<int64_t> h_topk(static_cast<size_t>(kHtTokens) * kTopK1, 0);
    std::vector<float> h_topk_weights(static_cast<size_t>(kHtTokens) * kTopK1, 1.0f);
    std::vector<uint8_t> h_tokens(static_cast<size_t>(kHtTokens) * kPackedHidden);
    std::vector<uint8_t> h_scales(static_cast<size_t>(kHtTokens) * kScaleBytes);
    for (int token = 0; token < kHtTokens; ++token) {
        for (int byte = 0; byte < kPackedHidden; ++byte)
            h_tokens[token * kPackedHidden + byte] = ht_em_token_byte(g_rank, token, byte);
        for (int byte = 0; byte < kScaleBytes; ++byte)
            h_scales[token * kScaleBytes + byte] = ht_em_scale_byte(g_rank, token, byte);
    }

    int64_t* d_topk = nullptr;
    float* d_topk_weights = nullptr;
    uint8_t *d_tokens = nullptr, *d_scales = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk, h_topk.size() * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_topk_weights, h_topk_weights.size() * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_tokens, h_tokens.size()));
    CUDA_ASSERT(cudaMalloc(&d_scales, h_scales.size()));
    CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(
        d_topk_weights, h_topk_weights.data(), h_topk_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));

    ncclEpTensor_t *topk = nullptr, *topk_weights = nullptr, *tokens = nullptr, *scales = nullptr;
    NCCL_ASSERT(epTensorCreate(&topk, 2, ncclInt64, d_topk, kHtTokens, kTopK1));
    NCCL_ASSERT(epTensorCreate(&topk_weights, 2, ncclFloat32, d_topk_weights, kHtTokens, kTopK1));
    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclFloat4x2, d_tokens, kHtTokens, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&scales, 2, ncclUint8, d_scales, kHtTokens, kScaleBytes));

    ncclEpHandleConfig_t handle_config = NCCL_EP_HANDLE_CONFIG_INIT;
    handle_config.dispatch_output_per_expert_alignment = kAlignment;
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk, nullptr, &handle_config, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    unsigned int recv_rows = ~0u;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(handle, &recv_rows));
    if (g_rank == 0) {
        ASSERT_GT(recv_rows, 0u) << "rank 0 must receive every rank's tokens";
    } else {
        ASSERT_EQ(recv_rows, 0u) << "skewed routing must leave this rank with zero recv rows";
    }

    // Eager sizing: recv buffers are the per-step recv count, so the zero-recv
    // ranks allocate nothing and hand over null-backed empty descriptors.
    uint8_t *d_recv_tokens = nullptr, *d_recv_scales = nullptr;
    float* d_recv_topk_weights = nullptr;
    if (recv_rows > 0) {
        CUDA_ASSERT(cudaMalloc(&d_recv_tokens, static_cast<size_t>(recv_rows) * kPackedHidden));
        CUDA_ASSERT(cudaMalloc(&d_recv_scales, static_cast<size_t>(recv_rows) * kScaleBytes));
        CUDA_ASSERT(cudaMalloc(&d_recv_topk_weights, static_cast<size_t>(recv_rows) * sizeof(float)));
        CUDA_ASSERT(cudaMemset(d_recv_tokens, 0, static_cast<size_t>(recv_rows) * kPackedHidden));
        CUDA_ASSERT(cudaMemset(d_recv_scales, 0, static_cast<size_t>(recv_rows) * kScaleBytes));
    }

    ncclEpTensor_t *recv_tokens = nullptr, *recv_scales = nullptr, *recv_topk_weights = nullptr;
    NCCL_ASSERT(epTensorCreate(&recv_tokens, 2, ncclFloat4x2, d_recv_tokens, recv_rows, kPackedHidden));
    NCCL_ASSERT(epTensorCreate(&recv_scales, 2, ncclUint8, d_recv_scales, recv_rows, kScaleBytes));
    NCCL_ASSERT(epTensorCreate(&recv_topk_weights, 1, ncclFloat32, d_recv_topk_weights, recv_rows));

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = tokens;
    inputs.scales = scales;
    inputs.topk_weights = topk_weights;
    outputs.tokens = recv_tokens;
    outputs.scales = recv_scales;
    outputs.topk_weights = recv_topk_weights;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;

    // Pre-fix, the zero-recv ranks abort here on the launcher's scale-pairing
    // assert (empty caller scales vs. non-empty internal FLAT scale staging).
    ASSERT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclSuccess)
        << "Rank " << g_rank << ": zero-recv QUANT_FWD dispatch must accept empty recv outputs";
    ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": zero-recv QUANT_FWD dispatch must not fault";

    // Rank 0 still receives every sender's token AND scale rows intact: the
    // (src_rank, token) pair is recoverable from bytes 0/1 of each row, so the
    // delivered set is checked without depending on slot order.
    if (recv_rows > 0) {
        std::vector<uint8_t> h_recv_tokens(static_cast<size_t>(recv_rows) * kPackedHidden);
        std::vector<uint8_t> h_recv_scales(static_cast<size_t>(recv_rows) * kScaleBytes);
        CUDA_ASSERT(cudaMemcpy(
            h_recv_tokens.data(), d_recv_tokens, h_recv_tokens.size(), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(
            h_recv_scales.data(), d_recv_scales, h_recv_scales.size(), cudaMemcpyDeviceToHost));

        std::set<std::pair<int, int>> seen;
        for (unsigned int row = 0; row < recv_rows; ++row) {
            const uint8_t* trow = h_recv_tokens.data() + static_cast<size_t>(row) * kPackedHidden;
            const uint8_t* srow = h_recv_scales.data() + static_cast<size_t>(row) * kScaleBytes;
            if (trow[0] == 0 && trow[1] == 0) continue;  // alignment pad row
            const int src_rank = trow[0] - 0x40;
            const int token = trow[1] - 1;
            ASSERT_GE(src_rank, 0);
            ASSERT_LT(src_rank, g_nranks);
            ASSERT_GE(token, 0);
            ASSERT_LT(token, kHtTokens);
            for (int byte = 0; byte < kPackedHidden; ++byte) {
                ASSERT_EQ(trow[byte], ht_em_token_byte(src_rank, token, byte))
                    << "token row " << row << " byte " << byte;
            }
            for (int byte = 0; byte < kScaleBytes; ++byte) {
                ASSERT_EQ(srow[byte], ht_em_scale_byte(src_rank, token, byte))
                    << "scale row " << row << " byte " << byte
                    << " (scales must travel with their token row)";
            }
            seen.insert({src_rank, token});
        }
        EXPECT_EQ(seen.size(), static_cast<size_t>(g_nranks) * kHtTokens)
            << "rank 0 must receive every (rank, token) pair exactly once";
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(topk);
    ncclEpTensorDestroy(topk_weights);
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(scales);
    ncclEpTensorDestroy(recv_tokens);
    ncclEpTensorDestroy(recv_scales);
    ncclEpTensorDestroy(recv_topk_weights);
    cudaFree(d_topk);
    cudaFree(d_topk_weights);
    cudaFree(d_tokens);
    cudaFree(d_scales);
    if (d_recv_tokens) cudaFree(d_recv_tokens);
    if (d_recv_scales) cudaFree(d_recv_scales);
    if (d_recv_topk_weights) cudaFree(d_recv_topk_weights);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(QuantizationRecipeTest, HtExpertMajorScalesForwardEagerAcceptsZeroRecvOutputs) {
    run_ht_em_scales_forward_eager_zero_recv();
}

TEST_F(QuantizationRecipeTest, DispatchNoneRejectsScaleTensors) {
    RecipeTensor tokens(ncclBfloat16);
    RecipeTensor output_tokens(ncclBfloat16);
    RecipeTensor scales(ncclFloat32, kNumTokens, kScalesPerToken);
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
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    inputs.tokens = &tokens.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsUint8TokenDtype) {
    RecipeTensor tokens(ncclUint8, kNumTokens, 16);
    RecipeTensor output_tokens(ncclUint8, kMaxRecvSlots, 16);
    RecipeTensor input_scales(ncclUint8, kNumTokens, 16);
    RecipeTensor output_scales(ncclUint8, kMaxRecvSlots, 16);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &input_scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsMismatchedScaleDtype) {
    RecipeTensor tokens(ncclFloat8e4m3);
    RecipeTensor output_tokens(ncclFloat8e4m3, kMaxRecvSlots, kHidden);
    RecipeTensor input_scales(ncclUint8, kNumTokens, 16);
    RecipeTensor output_scales(ncclFloat32, kMaxRecvSlots, 16);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &input_scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

// These are host-side contract tests: validation rejects the malformed byte
// strides before dispatch can enqueue or launch a device kernel.
TEST_F(QuantizationRecipeTest, ScalesForwardRejectsUnalignedTokenRow) {
    RecipeTensor tokens(ncclFloat8e4m3, kNumTokens, 15);
    RecipeTensor output_tokens(ncclFloat8e4m3);
    RecipeTensor input_scales(ncclUint8, kNumTokens, 16);
    RecipeTensor output_scales(ncclUint8, kMaxRecvSlots, 16);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &input_scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsUnalignedScaleRow) {
    RecipeTensor tokens(ncclFloat8e4m3);
    RecipeTensor output_tokens(ncclFloat8e4m3);
    RecipeTensor input_scales(ncclUint8, kNumTokens, 15);
    RecipeTensor output_scales(ncclUint8, kMaxRecvSlots, 15);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &input_scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsPayloadAboveGroupByteLimit) {
    RecipeTensor tokens(ncclFloat8e4m3, kNumTokens, 16);
    RecipeTensor input_scales(ncclUint8, kNumTokens, 32);
    RecipeTensor output_tokens(ncclFloat8e4m3, kMaxRecvSlots, 16);
    RecipeTensor output_scales(ncclUint8, kMaxRecvSlots, 32);
    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    inputs.tokens = &tokens.tensor;
    inputs.scales = &input_scales.tensor;
    outputs.tokens = &output_tokens.tensor;
    outputs.scales = &output_scales.tensor;
    config.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    ncclEpHandle_t handle = make_handle(nullptr);
    EXPECT_EQ(ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream), ncclInvalidArgument);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsMisalignedTokenStorage) {
    ncclEpHandle_t handle = make_handle(nullptr);
    expect_invalid_misaligned_recipe_storage(handle, MisalignedRecipeStorage::InputTokens);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsMisalignedScaleStorage) {
    ncclEpHandle_t handle = make_handle(nullptr);
    expect_invalid_misaligned_recipe_storage(handle, MisalignedRecipeStorage::InputScales);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsMisalignedOutputTokenStorage) {
    ncclEpHandle_t handle = make_handle(nullptr);
    expect_invalid_misaligned_recipe_storage(handle, MisalignedRecipeStorage::OutputTokens);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, ScalesForwardRejectsMisalignedOutputScaleStorage) {
    ncclEpHandle_t handle = make_handle(nullptr);
    expect_invalid_misaligned_recipe_storage(handle, MisalignedRecipeStorage::OutputScales);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, HtScalesForwardRejectsMismatchedOutputTokenWidth) {
    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    expect_invalid_ht_scales_forward_output(
        handle,
        ncclFloat8e4m3,
        kMaxRecvSlots,
        /*output_token_cols=*/32,
        kMaxRecvSlots,
        /*output_scale_cols=*/16);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, HtScalesForwardRejectsShortOutputCapacity) {
    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    expect_invalid_ht_scales_forward_output(
        handle,
        ncclFloat8e4m3,
        kMaxRecvSlots - 1,
        /*output_token_cols=*/16,
        kMaxRecvSlots - 1,
        /*output_scale_cols=*/16);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, HtScalesForwardRejectsMismatchedOutputRowCapacities) {
    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    expect_invalid_ht_scales_forward_output(
        handle,
        ncclFloat8e4m3,
        kMaxRecvSlots + 1,
        /*output_token_cols=*/16,
        kMaxRecvSlots,
        /*output_scale_cols=*/16);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, HtScalesForwardRejectsMismatchedOutputScaleWidth) {
    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    expect_invalid_ht_scales_forward_output(
        handle,
        ncclFloat8e4m3,
        kMaxRecvSlots,
        /*output_token_cols=*/16,
        kMaxRecvSlots,
        /*output_scale_cols=*/32);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, HtScalesForwardRejectsMismatchedOutputTokenDtype) {
    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    expect_invalid_ht_scales_forward_output(
        handle,
        ncclFloat8e5m2,
        kMaxRecvSlots,
        /*output_token_cols=*/16,
        kMaxRecvSlots,
        /*output_scale_cols=*/16);
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

TEST_F(QuantizationRecipeTest, NvFp4CombineRejectsUnsupportedDeviceBeforeJit) {
    if (nvfp4_supported()) GTEST_SKIP() << "requires a GPU without E2M1 FP4 support";
    RecipeTensor tokens(ncclBfloat16);
    RecipeTensor output_tokens(ncclBfloat16);
    ncclEpCombineInputs_t inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t config = NCCL_EP_COMBINE_CONFIG_INIT;
    inputs.tokens = &tokens.tensor;
    outputs.tokens = &output_tokens.tensor;
    config.quant_recipe = NCCL_EP_COMB_QUANT_NVFP4;
    ncclEpHandle_t handle = make_handle(nullptr);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(ncclEpCombine(handle, &inputs, &outputs, &config, g_stream), ncclInvalidUsage);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
}

TEST_F(QuantizationRecipeTest, NvFp4CombineExpertMajorProducesWeightedBf16) {
    if (!nvfp4_supported()) GTEST_SKIP() << "NVFP4 combine requires E2M1 FP4 support";
    constexpr int kNvFp4Hidden = 4096;
    ASSERT_EQ(kNumExperts % g_nranks, 0);
    const int local_experts = kNumExperts / g_nranks;
    const int recv_slots = g_nranks * kNumTokens;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    group_config.max_token_bytes = kNvFp4Hidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = local_experts;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.max_recv_tokens_per_rank = kNumTokens;
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    const size_t expert_elems = static_cast<size_t>(local_experts) * recv_slots * kNvFp4Hidden;
    nv_bfloat16 *d_input = nullptr, *d_expert = nullptr, *d_combined = nullptr;
    float *d_global_scales = nullptr, *d_topk_weights = nullptr;
    int* d_expert_counters = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_input, static_cast<size_t>(kNumTokens) * kNvFp4Hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_expert, expert_elems * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined, static_cast<size_t>(kNumTokens) * kNvFp4Hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_global_scales, static_cast<size_t>(local_experts) * recv_slots * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_topk_weights, static_cast<size_t>(kNumTokens) * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_expert_counters, local_experts * sizeof(int)));
    std::vector<nv_bfloat16> h_input(static_cast<size_t>(kNumTokens) * kNvFp4Hidden);
    std::vector<float> h_scales(static_cast<size_t>(local_experts) * recv_slots);
    std::vector<float> h_weights(static_cast<size_t>(kNumTokens) * kTopK, 1.f);
    fill_nvfp4_test_tokens(&h_input, kNvFp4Hidden);
    for (size_t slot = 0; slot < h_scales.size(); ++slot) {
        h_scales[slot] = nvfp4_test_global_scale(static_cast<int>(slot % kNumTokens), kNvFp4Hidden);
    }
    CUDA_ASSERT(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_global_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_topk_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *input = nullptr, *expert = nullptr, *combined = nullptr, *scales = nullptr, *weights = nullptr, *counters = nullptr;
    NCCL_ASSERT(epTensorCreate(&input, 2, ncclBfloat16, d_input, kNumTokens, kNvFp4Hidden));
    NCCL_ASSERT(epTensorCreate(&expert, 3, ncclBfloat16, d_expert, local_experts, recv_slots, kNvFp4Hidden));
    NCCL_ASSERT(epTensorCreate(&combined, 2, ncclBfloat16, d_combined, kNumTokens, kNvFp4Hidden));
    NCCL_ASSERT(epTensorCreate(&scales, 3, ncclFloat32, d_global_scales, local_experts, recv_slots, 1));
    NCCL_ASSERT(epTensorCreate(&weights, 2, ncclFloat32, d_topk_weights, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&counters, 1, ncclInt32, d_expert_counters, local_experts));
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream));
    ncclEpDispatchInputs_t dispatch_inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dispatch_outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    dispatch_inputs.tokens = input; dispatch_outputs.tokens = expert; layout_info.expert_counters = counters;
    ASSERT_EQ(ncclEpDispatch(handle, &dispatch_inputs, &dispatch_outputs, &layout_info, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    ncclEpCombineInputs_t combine_inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t combine_outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t combine_config = NCCL_EP_COMBINE_CONFIG_INIT;
    combine_inputs.tokens = expert; combine_inputs.scales = scales;
    combine_outputs.tokens = combined; combine_outputs.topk_weights = weights;
    combine_config.quant_recipe = NCCL_EP_COMB_QUANT_NVFP4;
    ASSERT_EQ(ncclEpCombine(handle, &combine_inputs, &combine_outputs, &combine_config, g_stream), ncclSuccess);
    ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    std::vector<nv_bfloat16> h_combined(static_cast<size_t>(kNumTokens) * kNvFp4Hidden);
    CUDA_ASSERT(cudaMemcpy(h_combined.data(), d_combined, h_combined.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h_combined.size(); ++i) {
        const int token = i / kNvFp4Hidden;
        const int channel = i % kNvFp4Hidden;
        EXPECT_NEAR(__bfloat162float(h_combined[i]), nvfp4_test_value(token, channel), 1e-3f);
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(input); ncclEpTensorDestroy(expert); ncclEpTensorDestroy(combined);
    ncclEpTensorDestroy(scales); ncclEpTensorDestroy(weights); ncclEpTensorDestroy(counters);
    cudaFree(d_input); cudaFree(d_expert); cudaFree(d_combined); cudaFree(d_global_scales); cudaFree(d_topk_weights); cudaFree(d_expert_counters);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(QuantizationRecipeTest, NvFp4CombineRankMajorProducesBf16) {
    if (!nvfp4_supported()) GTEST_SKIP() << "NVFP4 combine requires E2M1 FP4 support";
    constexpr int kNvFp4Hidden = 4096;
    constexpr int kRmTopK = 2;
    const int recv_slots = g_nranks * kNumTokens;
    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    group_config.max_token_bytes = kNvFp4Hidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = kNumExperts / g_nranks;
    group_config.num_channels = NCCL_EP_AUTO;
    group_config.max_recv_tokens_per_rank = kNumTokens;
    group_config.num_topk = kRmTopK;
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

    nv_bfloat16 *d_input = nullptr, *d_expert = nullptr, *d_combined = nullptr;
    float *d_global_scales = nullptr, *d_topk_weights = nullptr, *d_recv_topk_weights = nullptr;
    int64_t* d_topk = nullptr;
    int *d_recv_topk_idx = nullptr, *d_src_rank_counters = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_input, static_cast<size_t>(kNumTokens) * kNvFp4Hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_expert, static_cast<size_t>(recv_slots) * kNvFp4Hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined, static_cast<size_t>(kNumTokens) * kNvFp4Hidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_global_scales, static_cast<size_t>(recv_slots) * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_topk, static_cast<size_t>(kNumTokens) * kRmTopK * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_topk_weights, static_cast<size_t>(kNumTokens) * kRmTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_weights, static_cast<size_t>(recv_slots) * kRmTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_idx, static_cast<size_t>(recv_slots) * kRmTopK * sizeof(int)));
    CUDA_ASSERT(cudaMalloc(&d_src_rank_counters, g_nranks * sizeof(int)));
    std::vector<nv_bfloat16> h_input(static_cast<size_t>(kNumTokens) * kNvFp4Hidden);
    std::vector<float> h_scales(recv_slots);
    std::vector<int64_t> h_topk(static_cast<size_t>(kNumTokens) * kRmTopK);
    std::vector<float> h_weights(static_cast<size_t>(kNumTokens) * kRmTopK, 1.f);
    const int local_experts = kNumExperts / g_nranks;
    fill_nvfp4_test_tokens(&h_input, kNvFp4Hidden);
    for (int slot = 0; slot < recv_slots; ++slot) {
        h_scales[slot] = nvfp4_test_global_scale(static_cast<int>(slot % kNumTokens), kNvFp4Hidden);
    }
    for (int token = 0; token < kNumTokens; ++token) {
        const int first = expert_for_token(token);
        h_topk[token * kRmTopK] = first;
        h_topk[token * kRmTopK + 1] =
            (first / local_experts) * local_experts + (first + 1) % local_experts;
    }
    CUDA_ASSERT(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_global_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_topk_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *input = nullptr, *expert = nullptr, *combined = nullptr, *scales = nullptr;
    ncclEpTensor_t *topk = nullptr, *weights = nullptr, *recv_weights = nullptr, *recv_idx = nullptr, *src_counters = nullptr;
    NCCL_ASSERT(epTensorCreate(&input, 2, ncclBfloat16, d_input, kNumTokens, kNvFp4Hidden));
    NCCL_ASSERT(epTensorCreate(&expert, 3, ncclBfloat16, d_expert, g_nranks, kNumTokens, kNvFp4Hidden));
    NCCL_ASSERT(epTensorCreate(&combined, 2, ncclBfloat16, d_combined, kNumTokens, kNvFp4Hidden));
    NCCL_ASSERT(epTensorCreate(&scales, 3, ncclFloat32, d_global_scales, g_nranks, kNumTokens, 1));
    NCCL_ASSERT(epTensorCreate(&topk, 2, ncclInt64, d_topk, kNumTokens, kRmTopK));
    NCCL_ASSERT(epTensorCreate(&weights, 2, ncclFloat32, d_topk_weights, kNumTokens, kRmTopK));
    NCCL_ASSERT(epTensorCreate(&recv_weights, 3, ncclFloat32, d_recv_topk_weights, g_nranks, kNumTokens, kRmTopK));
    NCCL_ASSERT(epTensorCreate(&recv_idx, 3, ncclInt32, d_recv_topk_idx, g_nranks, kNumTokens, kRmTopK));
    NCCL_ASSERT(epTensorCreate(&src_counters, 1, ncclInt32, d_src_rank_counters, g_nranks));
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_RANK_MAJOR, topk, nullptr, nullptr, g_stream));
    ncclEpDispatchInputs_t dispatch_inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dispatch_outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    dispatch_inputs.tokens = input; dispatch_inputs.topk_weights = weights;
    dispatch_outputs.tokens = expert; dispatch_outputs.topk_weights = recv_weights; dispatch_outputs.topk_idx = recv_idx;
    layout_info.src_rank_counters = src_counters;
    ASSERT_EQ(ncclEpDispatch(handle, &dispatch_inputs, &dispatch_outputs, &layout_info, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);

    ncclEpCombineInputs_t combine_inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t combine_outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t combine_config = NCCL_EP_COMBINE_CONFIG_INIT;
    combine_inputs.tokens = expert;
    combine_inputs.scales = scales;
    combine_outputs.tokens = combined;
    combine_config.quant_recipe = NCCL_EP_COMB_QUANT_NVFP4;
    ASSERT_EQ(ncclEpCombine(handle, &combine_inputs, &combine_outputs, &combine_config, g_stream), ncclSuccess);
    ASSERT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    std::vector<nv_bfloat16> h_combined(static_cast<size_t>(kNumTokens) * kNvFp4Hidden);
    CUDA_ASSERT(cudaMemcpy(h_combined.data(), d_combined, h_combined.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h_combined.size(); ++i) {
        const int token = i / kNvFp4Hidden;
        const int channel = i % kNvFp4Hidden;
        EXPECT_NEAR(__bfloat162float(h_combined[i]), nvfp4_test_value(token, channel), 1e-3f);
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(input); ncclEpTensorDestroy(expert); ncclEpTensorDestroy(combined); ncclEpTensorDestroy(scales);
    ncclEpTensorDestroy(topk); ncclEpTensorDestroy(weights); ncclEpTensorDestroy(recv_weights); ncclEpTensorDestroy(recv_idx); ncclEpTensorDestroy(src_counters);
    cudaFree(d_input); cudaFree(d_expert); cudaFree(d_combined); cudaFree(d_global_scales);
    cudaFree(d_topk); cudaFree(d_topk_weights); cudaFree(d_recv_topk_weights); cudaFree(d_recv_topk_idx); cudaFree(d_src_rank_counters);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_quantization_recipe_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
