/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * HT intra-node MXFP8 combine. Quantization is always internal: the app hands over BF16
 * rows and local_permute_reduce emits packed [FP8 E4M3 | E8M0] before transport, so the
 * recipe is only available on the expert-major local-permute path.
 */

#include "test_common.h"
#include "device/quant_recipe.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time layout contract: MXFP8 wire bytes per element = 1 (FP8 E4M3),
// scale block = 32 (one E8M0 per 32 elements -> scale bytes per token = H/32),
// and NONE recipe has no scale block.
// ---------------------------------------------------------------------------
static_assert(nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::kWireBytesPerElem == 1,
    "MXFP8 wire element must be 1 byte (FP8 E4M3); wire bytes per token = H, not 2H");
static_assert(nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::kScaleBlock == 32,
    "MXFP8 scale block must be 32 elements; scale bytes per token = H/32");
static_assert(NCCL_EP_MXFP8_HIDDEN_ALIGN == 16 * NCCL_EP_MXFP8_SCALE_BLOCK,
    "packed-row 16B alignment is 16 * E8M0 block");
static_assert(nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_NONE>::kWireBytesPerElem == 0,
    "NONE recipe must derive wire bytes from kTokenDtype (kWireBytesPerElem == 0)");
static_assert(nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_NONE>::kScaleBlock == 0,
    "NONE recipe must have no scale block");

static constexpr int kMxHidden = 512;           // must be a multiple of 32 (MXFP8 block)
static constexpr int kMxElemsTested = 10;       // first N elements per token to validate

static ncclEpGroup_t make_mxfp8_ht_group(ncclEpHtFlow_t flow) {
    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts = kNumExperts;
    gcfg.num_topk = kTopK;
    gcfg.ht_flow = flow;
    gcfg.max_dispatch_tokens_per_rank = kNumTokens;
    gcfg.max_token_bytes = kMxHidden * static_cast<unsigned int>(sizeof(nv_bfloat16));
    gcfg.rdma_buffer_size = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank = NCCL_EP_AUTO;
    gcfg.num_channels = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecvSlots);
    ncclEpGroup_t group = nullptr;
    EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &gcfg), ncclSuccess);
    return group;
}

// Fill BF16 expert output rows: first kMxElemsTested elements = (j+1)*2, rest = 0.
__global__ void fill_bf16_expert_rows_kernel(nv_bfloat16* rows, int num_rows, int hidden, int elems) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    nv_bfloat16* r = rows + static_cast<size_t>(row) * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        r[j] = (j < elems) ? __float2bfloat16(static_cast<float>((j + 1) * 2)) : __float2bfloat16(0.0f);
}

static void fill_bf16_expert_rows(nv_bfloat16* d, int num_rows) {
    fill_bf16_expert_rows_kernel<<<num_rows, 128, 0, g_stream>>>(d, num_rows, kMxHidden, kMxElemsTested);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
}

// d_recv_w_out / recv_w_tensor_out are non-null only for the backward case, which feeds the
// forward-delivered recv weights back in as the weight-gradient input; pass nullptr to have
// them released here.
static void run_forward_dispatch_bf16(
    ncclEpHandle_t handle,
    bool expert_major,
    nv_bfloat16** d_recv_out,
    ncclEpTensor_t** recv_tensor_out,
    float** d_recv_w_out,
    ncclEpTensor_t** recv_w_tensor_out) {
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float *d_weights = nullptr, *d_recv_w = nullptr;
    int64_t* d_recv_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,    kNumTokens  * kMxHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,   kMaxRecvSlots * kMxHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights, kNumTokens  * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w, kMaxRecvSlots * kTopK * sizeof(float)));
    if (!expert_major) CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)));

    std::vector<nv_bfloat16> h_tok(kNumTokens * kMxHidden, __float2bfloat16(1.0f));
    std::vector<float>       h_w(kNumTokens * kTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_tok,     h_tok.data(), h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(),   h_w.size()   * sizeof(float),        cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv,    0, kMaxRecvSlots * kMxHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv_w,  0, kMaxRecvSlots * kTopK     * sizeof(float)));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_w = nullptr,
                   *t_recv_w = nullptr, *t_recv_idx = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,  2, ncclBfloat16, d_tok,    kNumTokens,    kMxHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv,   kMaxRecvSlots, kMxHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,    2, ncclFloat32,  d_weights, kNumTokens,   kTopK));
    if (expert_major) {
        NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots));
    } else {
        NCCL_ASSERT(epTensorCreate(&t_recv_w,   2, ncclFloat32, d_recv_w,   kMaxRecvSlots, kTopK));
        NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64,   d_recv_idx, kMaxRecvSlots, kTopK));
    }

    ncclEpDispatchInputs_t d_in   = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t dcfg   = NCCL_EP_DISPATCH_CONFIG_INIT;
    d_in.tokens      = t_tok;
    d_in.topk_weights = t_w;
    d_out.tokens     = t_recv;
    d_out.topk_weights = t_recv_w;
    if (!expert_major) d_out.topk_idx = t_recv_idx;
    EXPECT_EQ(ncclEpDispatch(handle, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_w);
    if (t_recv_idx) ncclEpTensorDestroy(t_recv_idx);
    cudaFree(d_tok); cudaFree(d_weights);
    if (d_recv_idx) cudaFree(d_recv_idx);

    if (recv_w_tensor_out != nullptr) {
        *recv_w_tensor_out = t_recv_w;
        *d_recv_w_out = d_recv_w;
    } else {
        ncclEpTensorDestroy(t_recv_w);
        cudaFree(d_recv_w);
    }
    *d_recv_out      = d_recv;
    *recv_tensor_out = t_recv;
}

// Verify BF16 combine output: expected = slot_count * (j+1)*2 per element.
static void verify_bf16_combine_output(nv_bfloat16* d_combined, int slot_count, float rel_tol = 0.05f) {
    std::vector<nv_bfloat16> h(kNumTokens * kMxHidden);
    CUDA_ASSERT(cudaMemcpy(h.data(), d_combined, h.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kNumTokens; ++i) {
        for (int j = 0; j < kMxElemsTested; ++j) {
            const float expected = static_cast<float>(slot_count * (j + 1) * 2);
            const float actual   = __bfloat162float(h[static_cast<size_t>(i) * kMxHidden + j]);
            EXPECT_NEAR(actual, expected, expected * rel_tol)
                << "token=" << i << " elem=" << j << " rank=" << g_rank;
        }
    }
}

class Mxfp8CombineTest : public EpTestBase {
protected:
    // MXFP8 combine is expert-major only: the quantizing prologue lives in
    // local_permute_reduce, which no other layout runs.
    void run_mxfp8_combine_case() {
        ncclEpGroup_t group = make_mxfp8_ht_group(NCCL_EP_HT_FLOW_PUSH_PULL);
        ASSERT_NE(group, nullptr);

        ncclEpHandle_t handle = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(
            &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream),
            ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        ASSERT_NE(handle, nullptr);

        // Run dispatch first to establish connectivity (tokens and weights).
        nv_bfloat16*   d_recv_bf16 = nullptr;
        ncclEpTensor_t* recv_bf16   = nullptr;
        run_forward_dispatch_bf16(handle, /*expert_major=*/true, &d_recv_bf16, &recv_bf16,
                                  /*d_recv_w_out=*/nullptr, /*recv_w_tensor_out=*/nullptr);
        cudaFree(d_recv_bf16);
        ncclEpTensorDestroy(recv_bf16);

        // Allocate BF16 expert output rows (no scales — NCCL EP quantizes internally).
        nv_bfloat16* d_bf16    = nullptr;
        nv_bfloat16* d_combined = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_bf16,    kMaxRecvSlots * kMxHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_combined, kNumTokens   * kMxHidden * sizeof(nv_bfloat16)));
        fill_bf16_expert_rows(d_bf16, kMaxRecvSlots);

        ncclEpTensor_t *t_bf16 = nullptr, *t_combined = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_bf16,    2, ncclBfloat16, d_bf16,    kMaxRecvSlots, kMxHidden));
        NCCL_ASSERT(epTensorCreate(&t_combined, 2, ncclBfloat16, d_combined, kNumTokens,   kMxHidden));

        ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        ncclEpCombineConfig_t  ccfg  = NCCL_EP_COMBINE_CONFIG_INIT;
        c_in.tokens  = t_bf16;   // BF16 — no scales; NCCL EP quantizes internally
        c_in.scales  = nullptr;
        c_out.tokens = t_combined;
        ccfg.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;

        EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out, &ccfg, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        // local_permute_reduce sums the top-k contributions before quantizing, so each
        // output element carries kTopK copies. 5% tolerance covers MXFP8 quant error.
        verify_bf16_combine_output(d_combined, /*slot_count=*/kTopK, /*rel_tol=*/0.05f);

        ncclEpTensorDestroy(t_bf16);
        ncclEpTensorDestroy(t_combined);
        cudaFree(d_bf16);
        cudaFree(d_combined);
        NCCL_ASSERT(ncclEpHandleDestroy(handle));
        NCCL_ASSERT(ncclEpGroupDestroy(group));
    }
};

TEST_F(Mxfp8CombineTest, ExpertMajorCombineMatchesReference) {
    run_mxfp8_combine_case();
}

// Backward combine takes the same recipe: neither the host gating nor the kernel templates
// tie MXFP8 to the forward pass, and backward reduces gradients through the same
// local_permute_reduce prologue. This is the only case that instantiates the backward MXFP8
// JIT variant, and it pins the split of concerns: the token path is quantized, while the
// FP32 weight gradients must round-trip bit-exactly through it.
TEST_F(Mxfp8CombineTest, ExpertMajorBackwardCombineMatchesReference) {
    ncclEpGroup_t group = make_mxfp8_ht_group(NCCL_EP_HT_FLOW_PUSH_PULL);
    ASSERT_NE(group, nullptr);

    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream),
        ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    ASSERT_NE(handle, nullptr);

    // Forward dispatch delivers recv_topk_weights (1D [num_recv] for EM); feeding those back
    // as the backward-combine weight input must reproduce the weights it was given (all 1.0).
    nv_bfloat16* d_recv_bf16 = nullptr;
    ncclEpTensor_t* recv_bf16 = nullptr;
    float* d_recv_w = nullptr;
    ncclEpTensor_t* t_recv_w = nullptr;
    run_forward_dispatch_bf16(handle, /*expert_major=*/true, &d_recv_bf16, &recv_bf16,
                              &d_recv_w, &t_recv_w);
    cudaFree(d_recv_bf16);
    ncclEpTensorDestroy(recv_bf16);

    // Gradient rows carry the same pattern as the forward case, so the expected reduction is
    // the same: one unweighted copy per routed expert.
    nv_bfloat16* d_grad = nullptr;
    nv_bfloat16* d_combined = nullptr;
    float* d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_grad,     kMaxRecvSlots * kMxHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined, kNumTokens    * kMxHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens  * kTopK     * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));
    fill_bf16_expert_rows(d_grad, kMaxRecvSlots);

    ncclEpTensor_t *t_grad = nullptr, *t_combined = nullptr, *t_combined_w = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_grad,       2, ncclBfloat16, d_grad,     kMaxRecvSlots, kMxHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined,   2, ncclBfloat16, d_combined, kNumTokens,    kMxHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32,  d_combined_w, kNumTokens,  kTopK));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t  ccfg  = NCCL_EP_COMBINE_CONFIG_INIT;
    c_in.tokens = t_grad;
    c_in.scales = nullptr;
    c_in.topk_weights = t_recv_w;   // 1D for EM
    c_out.tokens = t_combined;
    c_out.topk_weights = t_combined_w;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    ccfg.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;

    EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out, &ccfg, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    verify_bf16_combine_output(d_combined, /*slot_count=*/kTopK, /*rel_tol=*/0.05f);

    // Weight gradients never enter the quantized path, so they are exact, not approximate.
    std::vector<float> h_combined_w(kNumTokens * kTopK);
    CUDA_ASSERT(cudaMemcpy(h_combined_w.data(), d_combined_w,
                           h_combined_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kNumTokens; ++i) {
        for (int k = 0; k < kTopK; ++k) {
            EXPECT_FLOAT_EQ(h_combined_w[i * kTopK + k], 1.0f)
                << "MXFP8 backward combine must leave the FP32 weight gradients untouched"
                << " (rank " << g_rank << " token " << i << " k " << k << ")";
        }
    }

    ncclEpTensorDestroy(t_grad);
    ncclEpTensorDestroy(t_combined);
    ncclEpTensorDestroy(t_combined_w);
    ncclEpTensorDestroy(t_recv_w);
    cudaFree(d_grad);
    cudaFree(d_combined);
    cudaFree(d_combined_w);
    cudaFree(d_recv_w);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

// MXFP8 has no quantizing prologue outside the expert-major local-permute path, so a FLAT
// handle must be rejected rather than silently transporting unquantized BF16.
TEST_F(Mxfp8CombineTest, RejectsFlatLayout) {
    ncclEpGroup_t group = make_mxfp8_ht_group(NCCL_EP_HT_FLOW_PUSH_PULL);
    ASSERT_NE(group, nullptr);
    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_FLAT, topk_idx_, nullptr, nullptr, g_stream),
              ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    ASSERT_NE(handle, nullptr);

    nv_bfloat16* d_tok = nullptr; CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * kMxHidden * 2));
    nv_bfloat16* d_out = nullptr; CUDA_ASSERT(cudaMalloc(&d_out, kNumTokens * kMxHidden * 2));

    ncclEpTensor_t *t_tok = nullptr, *t_out = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kMxHidden));
    NCCL_ASSERT(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kNumTokens, kMxHidden));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t  ccfg  = NCCL_EP_COMBINE_CONFIG_INIT;
    c_in.tokens  = t_tok;
    c_in.scales  = nullptr;
    c_out.tokens = t_out;
    ccfg.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;
    EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out, &ccfg, g_stream), ncclInvalidArgument);

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_out);
    cudaFree(d_tok); cudaFree(d_out);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

TEST_F(Mxfp8CombineTest, AutoPullPushRejectsMxfp8) {
    if (ncclTeamLsa(g_comm).nRanks != g_nranks) {
        GTEST_SKIP() << "AUTO pull-push requires one LSA team";
    }
    ScopedEnvVar flow_env("NCCL_EP_HT_PULL_PUSH", nullptr);
    ScopedEnvVar local_dup_env("NCCL_EP_HT_EM_LOCAL_DUP", nullptr);
    ScopedEnvVar nvlink_dup_env("NCCL_EP_HT_EM_NVLINK_DUP", nullptr);
    ncclEpGroup_t group = make_mxfp8_ht_group(NCCL_EP_HT_FLOW_AUTO);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(ncclEpGroup_test_getHtFlow(group), NCCL_EP_HT_FLOW_PULL_PUSH);
    ncclEpHandle_t handle = nullptr;
    ASSERT_EQ(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream), ncclSuccess);
    nv_bfloat16* recv_data = nullptr;
    ncclEpTensor_t* recv = nullptr;
    run_forward_dispatch_bf16(handle, true, &recv_data, &recv, nullptr, nullptr);
    nv_bfloat16* output_data = nullptr;
    CUDA_ASSERT(cudaMalloc(&output_data, kNumTokens * kMxHidden * sizeof(nv_bfloat16)));
    ncclEpTensor_t* output = nullptr;
    NCCL_ASSERT(epTensorCreate(&output, 2, ncclBfloat16, output_data, kNumTokens, kMxHidden));
    ncclEpCombineInputs_t inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t config = NCCL_EP_COMBINE_CONFIG_INIT;
    inputs.tokens = recv;
    outputs.tokens = output;
    config.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;
    EXPECT_EQ(ncclEpCombine(handle, &inputs, &outputs, &config, g_stream), ncclInvalidArgument);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    NCCL_ASSERT(ncclEpTensorDestroy(recv));
    NCCL_ASSERT(ncclEpTensorDestroy(output));
    CUDA_ASSERT(cudaFree(recv_data));
    CUDA_ASSERT(cudaFree(output_data));
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

// Scales are generated internally, so any caller-supplied scale tensor is a contract error.
// Uses the expert-major handle so the rejection can only come from the scales rule.
TEST_F(Mxfp8CombineTest, RejectsWithScales) {
    ncclEpGroup_t group = make_mxfp8_ht_group(NCCL_EP_HT_FLOW_PUSH_PULL);
    ASSERT_NE(group, nullptr);
    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream),
        ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    ASSERT_NE(handle, nullptr);

    nv_bfloat16* d_tok    = nullptr; CUDA_ASSERT(cudaMalloc(&d_tok,    kNumTokens * kMxHidden * 2));
    uint8_t*     d_scales = nullptr; CUDA_ASSERT(cudaMalloc(&d_scales, kNumTokens * (kMxHidden / 32)));
    nv_bfloat16* d_out    = nullptr; CUDA_ASSERT(cudaMalloc(&d_out,    kNumTokens * kMxHidden * 2));

    ncclEpTensor_t *t_tok = nullptr, *t_scales = nullptr, *t_out = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,    2, ncclBfloat16, d_tok,    kNumTokens, kMxHidden));
    NCCL_ASSERT(epTensorCreate(&t_scales, 2, ncclUint8,    d_scales, kNumTokens, kMxHidden / 32));
    NCCL_ASSERT(epTensorCreate(&t_out,    2, ncclBfloat16, d_out,    kNumTokens, kMxHidden));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpCombineConfig_t  ccfg  = NCCL_EP_COMBINE_CONFIG_INIT;
    c_in.tokens  = t_tok;
    c_in.scales  = t_scales;   // should be rejected
    c_out.tokens = t_out;
    ccfg.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;
    EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out, &ccfg, g_stream), ncclInvalidArgument);

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_scales);
    ncclEpTensorDestroy(t_out);
    cudaFree(d_tok); cudaFree(d_scales); cudaFree(d_out);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

// ---------------------------------------------------------------------------
// Wire-byte geometry tests (host-side, no GPU execution required)
// ---------------------------------------------------------------------------

class MxFP8WireGeometryTest : public EpTestBase {};

// Verify that for MXFP8 the per-token wire payload is H bytes (FP8, one byte per
// element) and the per-token scale payload is H/32 bytes (one E8M0 per 32-element
// block).  For NONE, combine_recipe_scale_row_bytes must return 0.
TEST_F(MxFP8WireGeometryTest, WireBytesAreHAndScaleBytesAreHOver32) {
    constexpr int H = kMxHidden;
    // kWireBytesPerElem and kScaleBlock are static constexpr, host-accessible.
    constexpr int kWireElem  = nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::kWireBytesPerElem;
    constexpr int kScaleBlk  = nccl_ep::combine_recipe_traits<NCCL_EP_COMB_QUANT_MXFP8>::kScaleBlock;

    // Token wire bytes = H * 1 = H (not 2H as BF16 would give)
    EXPECT_EQ(kWireElem * H, H);
    EXPECT_NE(kWireElem * H, 2 * H);

    // Scale bytes per token = H / 32
    EXPECT_EQ(H / kScaleBlk, H / 32);

    // Host-callable helper agrees
    EXPECT_EQ(nccl_ep::combine_recipe_scale_row_bytes(NCCL_EP_COMB_QUANT_MXFP8, H), H / 32);
    EXPECT_EQ(nccl_ep::combine_recipe_scale_row_bytes(NCCL_EP_COMB_QUANT_NONE,   H), 0);
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_mxfp8_combine_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
