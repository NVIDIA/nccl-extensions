/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for HT backward-flow paths:
 *
 *   HT_CombineWithGrad_* — ncclEpCombine with topk_weights input
 *                          (backward_combine=true). Reduces grad-recv tokens
 *                          and grad-recv topk_weights back to grad-tokens and
 *                          topk_weight_grad at the source.
 *                          Outputs: combined_x, combined_topk_weights.
 *
 *   HT_DispatchScatter_* — ncclEpDispatch with no topk_idx/topk_weights
 *                          (forward_dispatch=false). Pure token scatter to
 *                          per-expert positions; no weight handling.
 *
 * Both are covered for RM (FLAT, rank-major) and EM (expert-major) layouts.
 *
 *   ImportedHandleSupportsBackwardExpertMajor — forward update+dispatch, snapshot
 *                          handle_mem post-forward, mutate the original buffer,
 *                          ncclEpImportHandle from the snapshot, then BWD dispatch +
 *                          BWD combine on the imported handle must reproduce the
 *                          forward routing and topk-weight grads for both layouts.
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include "nccl_ep/common.hpp"
#include <cstdint>
#include <cuda_fp16.h>
#include <set>

static float bf16_val(nv_bfloat16 v) {
    return __bfloat162float(v);
}

// Pick a unique non-trivial per-(rank, token) topk weight so any routing error in the BWD path
// produces a value mismatch (not a 1.0==1.0 false-pass).
static float weight_for(int rank, int token, int k) {
    return 0.125f * static_cast<float>((rank + 1) * 100 + (token + 1) * 10 + k);
}

// ── Test fixture ──────────────────────────────────────────────────────────────

class HtBwdTest : public EpTestBase {
protected:
    void run_imported_backward(bool expert_major);
    // Forward dispatch on the given handle. Inputs: tokens + topk_weights.
    // Outputs (RM): recv_x, recv_topk_weights, recv_topk_idx.
    // Outputs (EM): recv_x, recv_topk_weights (1D).
    // Returns the device pointers and tensor handles for downstream BWD calls.
    struct FwdState {
        nv_bfloat16* d_tok = nullptr;
        nv_bfloat16* d_recv = nullptr;
        float* d_weights = nullptr;
        float* d_recv_w = nullptr;
        int64_t* d_recv_idx = nullptr;
        ncclEpTensor_t* t_tok = nullptr;
        ncclEpTensor_t* t_recv = nullptr;
        ncclEpTensor_t* t_w = nullptr;
        ncclEpTensor_t* t_recv_w = nullptr;
        ncclEpTensor_t* t_recv_idx = nullptr;
        bool expert_major = false;

        void free_all() {
            if (t_tok) ncclEpTensorDestroy(t_tok);
            if (t_recv) ncclEpTensorDestroy(t_recv);
            if (t_w) ncclEpTensorDestroy(t_w);
            if (t_recv_w) ncclEpTensorDestroy(t_recv_w);
            if (t_recv_idx) ncclEpTensorDestroy(t_recv_idx);
            if (d_tok) cudaFree(d_tok);
            if (d_recv) cudaFree(d_recv);
            if (d_weights) cudaFree(d_weights);
            if (d_recv_w) cudaFree(d_recv_w);
            if (d_recv_idx) cudaFree(d_recv_idx);
        }
    };

    void run_forward_dispatch(ncclEpHandle_t handle, bool expert_major, FwdState& st) {
        st.expert_major = expert_major;

        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }
        std::vector<float> h_w(kNumTokens * kTopK);
        for (int i = 0; i < kNumTokens; ++i)
            for (int k = 0; k < kTopK; ++k) h_w[i * kTopK + k] = weight_for(g_rank, i, k);

        EXPECT_EQ(cudaMalloc(&st.d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_weights, kNumTokens * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_recv_w, kMaxRecvSlots * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)), cudaSuccess);
        EXPECT_EQ(cudaMemset(st.d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(st.d_recv_w, 0, kMaxRecvSlots * kTopK * sizeof(float)), cudaSuccess);

        EXPECT_EQ(
            cudaMemcpy(st.d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);
        EXPECT_EQ(
            cudaMemcpy(st.d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

        EXPECT_EQ(epTensorCreate(&st.t_tok, 2, ncclBfloat16, st.d_tok, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&st.t_recv, 2, ncclBfloat16, st.d_recv, kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&st.t_w, 2, ncclFloat32, st.d_weights, kNumTokens, kTopK), ncclSuccess);
        if (expert_major) {
            EXPECT_EQ(epTensorCreate(&st.t_recv_w, 1, ncclFloat32, st.d_recv_w, kMaxRecvSlots), ncclSuccess);
            st.t_recv_idx = nullptr;
        } else {
            EXPECT_EQ(epTensorCreate(&st.t_recv_w, 2, ncclFloat32, st.d_recv_w, kMaxRecvSlots, kTopK), ncclSuccess);
            EXPECT_EQ(epTensorCreate(&st.t_recv_idx, 2, ncclInt64, st.d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);
        }

        ncclEpDispatchInputs_t d_in_s = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in_s.tokens = st.t_tok;
        d_in_s.topk_weights = st.t_w;
        d_out_s.tokens = st.t_recv;
        d_out_s.topk_weights = st.t_recv_w;
        if (!expert_major) d_out_s.topk_idx = st.t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    }
};

// ── HT CombineWithGrad = ncclEpCombine(backward_combine=true) ────────────────
// Inputs:  x = grad_recv_tokens [num_recv, hidden]
//          topk_weights = grad_recv_topk_weights — shape by layout:
//            RM: 2D [num_recv, top_k]   EM: 1D [num_recv]
// Outputs: combined_x [num_tokens, hidden]
//          combined_topk_weights [num_tokens, top_k]
//
// Test: echo the FWD recv tensors back as BWD combine input; verify the
// round-trip recovers the original source-side topk_weights.

TEST_F(HtBwdTest, CombineWithGradRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/false, fwd);
    nv_bfloat16* d_combined_x = nullptr;
    float* d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32, d_combined_w, kNumTokens, kTopK));

    // BWD-combine input topk_weights must be 2D [num_recv, top_k]; reuse forward output.
    ncclEpCombineInputs_t c_in_s = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens = fwd.t_recv;
    c_in_s.topk_weights = fwd.t_recv_w;
    c_out_s.tokens = t_combined_x;
    c_out_s.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK);
    CUDA_ASSERT(
        cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens * kTopK * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK; ++k) {
            float expected = weight_for(g_rank, i, k);
            EXPECT_NEAR(h_combined_w[i * kTopK + k], expected, 1e-4f)
                << "rank " << g_rank << " token " << i << " k " << k << " expected=" << expected
                << " got=" << h_combined_w[i * kTopK + k];
        }

    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_combined_x);
    cudaFree(d_combined_w);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

TEST_F(HtBwdTest, CombineWithGradExpertMajor) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/true, fwd);

    // EM: FWD recv_topk_weights is 1D [num_recv]; feed that 1D tensor DIRECTLY into
    // BWD combine input — no reshape needed. The output combined_topk_weights stays
    // 2D [num_combined_tokens, source_top_k].
    nv_bfloat16* d_combined_x = nullptr;
    float* d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32, d_combined_w, kNumTokens, kTopK));

    ncclEpCombineInputs_t c_in_s = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens = fwd.t_recv;
    c_in_s.topk_weights = fwd.t_recv_w;   // 1D for EM
    c_out_s.tokens = t_combined_x;
    c_out_s.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK);
    CUDA_ASSERT(
        cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens * kTopK * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK; ++k) {
            float expected = weight_for(g_rank, i, k);
            EXPECT_NEAR(h_combined_w[i * kTopK + k], expected, 1e-4f)
                << "EM rank " << g_rank << " token " << i << " k " << k << " expected=" << expected
                << " got=" << h_combined_w[i * kTopK + k];
        }

    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_combined_x);
    cudaFree(d_combined_w);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── HT DispatchScatter = ncclEpDispatch(forward_dispatch=false) ──────────────
// Inputs:  x [num_tokens, hidden]    (no topk_idx, no topk_weights)
// Outputs: recv_x [num_recv, hidden] (no recv_topk_*)
// Test: prime the handle with a forward dispatch, then re-dispatch new token
// values with no topk_idx; verify the recv set under the same routing.

TEST_F(HtBwdTest, DispatchScatterRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    // First forward dispatch primes the handle.
    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/false, fwd);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));

    // Re-dispatch with new token values; expect identical routing.
    std::vector<nv_bfloat16> h_tok2(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = 100.0f + static_cast<float>(g_rank * kNumTokens + i);
        for (int hh = 0; hh < kHidden; ++hh) h_tok2[i * kHidden + hh] = __float2bfloat16(v);
    }
    nv_bfloat16 *d_tok2, *d_recv2;
    CUDA_ASSERT(cudaMalloc(&d_tok2, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv2, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv2, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemcpy(d_tok2, h_tok2.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok2, *t_recv2;
    NCCL_ASSERT(epTensorCreate(&t_tok2, 2, ncclBfloat16, d_tok2, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv2, 2, ncclBfloat16, d_recv2, kMaxRecvSlots, kHidden));

    ncclEpDispatchInputs_t d_in_s = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens = t_tok2;
    d_out_s.tokens = t_recv2;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    dcfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_recv2(num_recv * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_recv2.data(), d_recv2, num_recv * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    // Expected: same routing as forward — the 4 recv slots come from the same 4 source
    // (rank, token) pairs that contributed in the forward pass. Each source token's
    // new value is 100 + (rank * kNumTokens + token).
    std::set<float> expected;
    if (g_rank == 0) {        // E0,E1 ← ranks 0,2
        expected = {100.f, 101.f, 108.f, 109.f};
    } else if (g_rank == 1) { // E2,E3 ← ranks 0,2
        expected = {102.f, 103.f, 110.f, 111.f};
    } else if (g_rank == 2) { // E4,E5 ← ranks 1,3
        expected = {104.f, 105.f, 112.f, 113.f};
    } else {                  // E6,E7 ← ranks 1,3
        expected = {106.f, 107.f, 114.f, 115.f};
    }
    std::set<float> got;
    for (unsigned s = 0; s < num_recv; ++s) got.insert(bf16_val(h_recv2[s * kHidden]));
    EXPECT_EQ(got, expected) << "rank " << g_rank << " dispatch-scatter recv set mismatch";

    ncclEpTensorDestroy(t_tok2);
    ncclEpTensorDestroy(t_recv2);
    cudaFree(d_tok2);
    cudaFree(d_recv2);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

TEST_F(HtBwdTest, DispatchScatterExpertMajor) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/true, fwd);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));

    std::vector<nv_bfloat16> h_tok2(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = 100.0f + static_cast<float>(g_rank * kNumTokens + i);
        for (int hh = 0; hh < kHidden; ++hh) h_tok2[i * kHidden + hh] = __float2bfloat16(v);
    }
    nv_bfloat16 *d_tok2, *d_recv2;
    CUDA_ASSERT(cudaMalloc(&d_tok2, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv2, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv2, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemcpy(d_tok2, h_tok2.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok2, *t_recv2;
    NCCL_ASSERT(epTensorCreate(&t_tok2, 2, ncclBfloat16, d_tok2, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv2, 2, ncclBfloat16, d_recv2, kMaxRecvSlots, kHidden));

    ncclEpDispatchInputs_t d_in_s = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens = t_tok2;
    d_out_s.tokens = t_recv2;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    dcfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // EM with top-k=1: each local expert receives 2 tokens. Verify the same
    // expected set as RankMajor.
    std::vector<nv_bfloat16> h_recv2(num_recv * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_recv2.data(), d_recv2, num_recv * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    std::set<float> expected;
    if (g_rank == 0) {
        expected = {100.f, 101.f, 108.f, 109.f};
    } else if (g_rank == 1) {
        expected = {102.f, 103.f, 110.f, 111.f};
    } else if (g_rank == 2) {
        expected = {104.f, 105.f, 112.f, 113.f};
    } else {
        expected = {106.f, 107.f, 114.f, 115.f};
    }
    std::set<float> got;
    for (unsigned s = 0; s < num_recv; ++s) got.insert(bf16_val(h_recv2[s * kHidden]));
    EXPECT_EQ(got, expected) << "EM rank " << g_rank << " dispatch-scatter recv set mismatch";

    ncclEpTensorDestroy(t_tok2);
    ncclEpTensorDestroy(t_recv2);
    cudaFree(d_tok2);
    cudaFree(d_recv2);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── HT ImportHandle + Backward — motivating relocated-snapshot path, expert-major ──
// Forward update+dispatch on a caller-owned handle_mem, snapshot handle_mem at the
// post-forward point (the EM map forward dispatch builds), then mutate the original
// buffer as a caller reusing that address for its next iteration would. ncclEpImportHandle
// binds a fresh handle to the snapshot; BWD dispatch (token scatter) and BWD combine
// (topk-weight grad) on that imported handle must reproduce the routing and weights
// the original forward pass established.

void HtBwdTest::run_imported_backward(bool expert_major) {
    const auto group = expert_major ? g_ep_group_em : g_ep_group;
    const auto layout = expert_major ? NCCL_EP_LAYOUT_EXPERT_MAJOR : NCCL_EP_LAYOUT_FLAT;
    auto* topk_idx = expert_major ? topk_idx_em_ : topk_idx_;
    size_t handle_mem_bytes = 0;
    NCCL_ASSERT(
        ncclEpHandleMemSize(group, layout, nullptr, &handle_mem_bytes, kTopK));

    auto make_handle_mem = [&](void** d_mem, ncclEpTensor_t** t) {
        CUDA_ASSERT(cudaMalloc(d_mem, handle_mem_bytes));
        NCCL_ASSERT(epTensorCreate(t, 1, ncclUint8, *d_mem, static_cast<unsigned int>(handle_mem_bytes)));
    };
    void* d_fwd_mem = nullptr;
    void* d_snap_mem = nullptr;
    ncclEpTensor_t* t_fwd_mem = nullptr;
    ncclEpTensor_t* t_snap_mem = nullptr;
    make_handle_mem(&d_fwd_mem, &t_fwd_mem);
    make_handle_mem(&d_snap_mem, &t_snap_mem);

    ncclEpHandle_t fwd_handle = nullptr;
    NCCL_ASSERT(
        ncclEpInitHandle(&fwd_handle, group, layout, nullptr, kTopK, t_fwd_mem));
    NCCL_ASSERT(ncclEpUpdateHandle(fwd_handle, topk_idx, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    FwdState fwd;
    run_forward_dispatch(fwd_handle, expert_major, fwd);

    // Snapshot the prepared handle_mem at the post-forward point.
    CUDA_ASSERT(cudaMemcpyAsync(d_snap_mem, d_fwd_mem, handle_mem_bytes, cudaMemcpyDeviceToDevice, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Mutate the original buffer for a different iteration (e.g. the caller reusing the same
    // address for its next step); the import below must not observe this.
    int64_t* d_topk2 = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk2, kNumTokens * kTopK * sizeof(int64_t)));
    std::vector<int64_t> h_topk2(kNumTokens);
    for (int i = 0; i < kNumTokens; ++i) h_topk2[i] = (expert_for_token(i) + 1) % kNumExperts;
    CUDA_ASSERT(cudaMemcpy(d_topk2, h_topk2.data(), h_topk2.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    ncclEpTensor_t* t_topk2 = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_topk2, 2, ncclInt64, d_topk2, kNumTokens, kTopK));
    NCCL_ASSERT(ncclEpUpdateHandle(fwd_handle, t_topk2, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    NCCL_ASSERT(ncclEpHandleDestroy(fwd_handle));
    ncclEpTensorDestroy(t_topk2);
    CUDA_ASSERT(cudaFree(d_topk2));
    ncclEpTensorDestroy(t_fwd_mem);
    CUDA_ASSERT(cudaFree(d_fwd_mem));

    ncclEpHandleState_t state = NCCL_EP_HANDLE_STATE_INIT;
    state.num_tokens = kNumTokens;
    state.num_topk = kTopK;
    state.topk_idx_datatype = ncclInt64;
    ncclEpHandle_t bwd_handle = nullptr;
    NCCL_ASSERT(ncclEpImportHandle(
        &bwd_handle, group, layout, nullptr, &state, t_snap_mem));

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(bwd_handle, &num_recv));

    // BWD dispatch (token scatter): re-dispatch grad-of-output values through the imported
    // handle; the recv set must match the ORIGINAL forward routing captured above.
    std::vector<nv_bfloat16> h_tok2(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = 100.0f + static_cast<float>(g_rank * kNumTokens + i);
        for (int hh = 0; hh < kHidden; ++hh) h_tok2[i * kHidden + hh] = __float2bfloat16(v);
    }
    nv_bfloat16 *d_tok2, *d_recv2;
    CUDA_ASSERT(cudaMalloc(&d_tok2, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv2, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv2, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(
        cudaMemcpy(d_tok2, h_tok2.data(), h_tok2.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    ncclEpTensor_t *t_tok2, *t_recv2;
    NCCL_ASSERT(epTensorCreate(&t_tok2, 2, ncclBfloat16, d_tok2, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv2, 2, ncclBfloat16, d_recv2, kMaxRecvSlots, kHidden));

    ncclEpDispatchInputs_t d_in_s = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens = t_tok2;
    d_out_s.tokens = t_recv2;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    dcfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpDispatch(bwd_handle, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(bwd_handle, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_recv2(num_recv * kHidden);
    CUDA_ASSERT(
        cudaMemcpy(h_recv2.data(), d_recv2, num_recv * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    std::set<float> expected;
    if (g_rank == 0) {
        expected = {100.f, 101.f, 108.f, 109.f};
    } else if (g_rank == 1) {
        expected = {102.f, 103.f, 110.f, 111.f};
    } else if (g_rank == 2) {
        expected = {104.f, 105.f, 112.f, 113.f};
    } else {
        expected = {106.f, 107.f, 114.f, 115.f};
    }
    std::set<float> got;
    for (unsigned s = 0; s < num_recv; ++s) got.insert(bf16_val(h_recv2[s * kHidden]));
    EXPECT_EQ(got, expected) << "rank " << g_rank << " imported-handle BWD dispatch recv set mismatch";

    // BWD combine (topk-weight grad): feed the ORIGINAL forward recv tensors back in; must
    // recover the exact weight_for() values the forward pass established.
    nv_bfloat16* d_combined_x = nullptr;
    float* d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));
    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32, d_combined_w, kNumTokens, kTopK));

    ncclEpCombineInputs_t c_in_s = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens = fwd.t_recv;
    c_in_s.topk_weights = fwd.t_recv_w;
    c_out_s.tokens = t_combined_x;
    c_out_s.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(bwd_handle, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK);
    CUDA_ASSERT(
        cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens * kTopK * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK; ++k) {
            float expected_w = weight_for(g_rank, i, k);
            EXPECT_NEAR(h_combined_w[i * kTopK + k], expected_w, 1e-4f)
                << "imported-handle rank " << g_rank << " token " << i << " k " << k;
        }

    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    CUDA_ASSERT(cudaFree(d_combined_x));
    CUDA_ASSERT(cudaFree(d_combined_w));
    ncclEpTensorDestroy(t_tok2);
    ncclEpTensorDestroy(t_recv2);
    CUDA_ASSERT(cudaFree(d_tok2));
    CUDA_ASSERT(cudaFree(d_recv2));
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(bwd_handle));
    ncclEpTensorDestroy(t_snap_mem);
    CUDA_ASSERT(cudaFree(d_snap_mem));
}

TEST_F(HtBwdTest, ImportedHandleSupportsBackwardExpertMajor) {
    run_imported_backward(/*expert_major=*/true);
}

TEST_F(HtBwdTest, ImportedHandleSupportsBackwardRankMajor) {
    run_imported_backward(/*expert_major=*/false);
}

// ── top-k=2 fixture (routing identical to TopK2MixedRoutingTest) ─────────────
// T0 → E0, E1   T1 → E2, E3   T2 → E4, E6   T3 → E5, E7
// Routing is sorted within each row so combined_topk_weights (packed in
// ascending expert-id order) matches the input k order.

static constexpr int kTopK2 = 2;

class HtBwdTopK2Test : public ::testing::Test {
protected:
    ncclEpTensor_t* topk_idx2_ = nullptr;
    ncclEpTensor_t* topk_idx2_em_ = nullptr;
    int64_t* d_topk2_ = nullptr;

    static float w2(int rank, int token, int k) {
        return 0.0625f * static_cast<float>((rank + 1) * 100 + (token + 1) * 10 + k);
    }

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk2_, kNumTokens * kTopK2 * sizeof(int64_t)));
        const int64_t routing[kNumTokens * kTopK2] = {
            0,
            1,
            2,
            3,
            4,
            6,
            5,
            7,
        };
        CUDA_ASSERT(cudaMemcpy(d_topk2_, routing, sizeof(routing), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_idx2_, 2, ncclInt64, d_topk2_, kNumTokens, kTopK2));
        NCCL_ASSERT(epTensorCreate(&topk_idx2_em_, 2, ncclInt64, d_topk2_, kNumTokens, kTopK2));
    }

    void TearDown() override {
        if (topk_idx2_em_) ncclEpTensorDestroy(topk_idx2_em_);
        if (topk_idx2_) ncclEpTensorDestroy(topk_idx2_);
        if (d_topk2_) cudaFree(d_topk2_);
    }

    ncclEpHandle_t mk(bool em) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(
            ncclEpCreateHandle(
                &h,
                em ? g_ep_group_em : g_ep_group,
                em ? NCCL_EP_LAYOUT_EXPERT_MAJOR : NCCL_EP_LAYOUT_FLAT,
                em ? topk_idx2_em_ : topk_idx2_,
                nullptr,
                nullptr,
                g_stream),
            ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }

    // RM top-k=2 combine-with-grad round-trip, parametrized on the cached
    // topk_idx datatype. Exercises the int32 and int64 idx paths through
    // CreateHandle and BWD combine; verifies combined_topk_weights recovers
    // the source-side weights.
    void run_combine_with_grad_rm(ncclDataType_t idx_dtype) {
        const int64_t routing[kNumTokens * kTopK2] = {
            0,
            1,
            2,
            3,
            4,
            6,
            5,
            7,
        };
        void* d_idx = nullptr;
        if (idx_dtype == ncclInt32) {
            int32_t routing32[kNumTokens * kTopK2];
            for (int i = 0; i < kNumTokens * kTopK2; ++i) routing32[i] = static_cast<int32_t>(routing[i]);
            CUDA_ASSERT(cudaMalloc(&d_idx, sizeof(routing32)));
            CUDA_ASSERT(cudaMemcpy(d_idx, routing32, sizeof(routing32), cudaMemcpyHostToDevice));
        } else {
            CUDA_ASSERT(cudaMalloc(&d_idx, sizeof(routing)));
            CUDA_ASSERT(cudaMemcpy(d_idx, routing, sizeof(routing), cudaMemcpyHostToDevice));
        }
        ncclEpTensor_t* t_idx = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_idx, 2, idx_dtype, d_idx, kNumTokens, kTopK2));

        ncclEpHandle_t h = nullptr;
        NCCL_ASSERT(ncclEpCreateHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, t_idx, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        ASSERT_NE(h, nullptr);

        // Forward dispatch with unique per-(rank, token, k) weights.
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        std::vector<float> h_w(kNumTokens * kTopK2);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
            for (int k = 0; k < kTopK2; ++k) h_w[i * kTopK2 + k] = w2(g_rank, i, k);
        }

        nv_bfloat16 *d_tok, *d_recv;
        float *d_weights, *d_recv_w;
        int64_t* d_recv_idx;
        CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_weights, kNumTokens * kTopK2 * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_w, kMaxRecvSlots * kTopK2 * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK2 * sizeof(int64_t)));
        CUDA_ASSERT(
            cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), kNumTokens * kTopK2 * sizeof(float), cudaMemcpyHostToDevice));

        ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w, *t_recv_idx;
        NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_weights, kNumTokens, kTopK2));
        NCCL_ASSERT(epTensorCreate(&t_recv_w, 2, ncclFloat32, d_recv_w, kMaxRecvSlots, kTopK2));
        NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64, d_recv_idx, kMaxRecvSlots, kTopK2));

        ncclEpDispatchInputs_t d_in_s = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in_s.tokens = t_tok;
        d_in_s.topk_weights = t_w;
        d_out_s.tokens = t_recv;
        d_out_s.topk_weights = t_recv_w;
        d_out_s.topk_idx = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        // Backward combine: input topk_weights = recv_topk_weights (forward output).
        nv_bfloat16* d_combined_x = nullptr;
        float* d_combined_w = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK2 * sizeof(float)));
        CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK2 * sizeof(float)));

        ncclEpTensor_t *t_combined_x, *t_combined_w;
        NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32, d_combined_w, kNumTokens, kTopK2));

        ncclEpCombineInputs_t c_in_s = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in_s.tokens = t_recv;
        c_in_s.topk_weights = t_recv_w;
        c_out_s.tokens = t_combined_x;
        c_out_s.topk_weights = t_combined_w;
        ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
        ccfg.pass_direction = NCCL_EP_BWD_PASS;
        NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        std::vector<float> h_combined_w(kNumTokens * kTopK2);
        CUDA_ASSERT(
            cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens * kTopK2 * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < kNumTokens; ++i)
            for (int k = 0; k < kTopK2; ++k) {
                float expected = w2(g_rank, i, k);
                EXPECT_NEAR(h_combined_w[i * kTopK2 + k], expected, 1e-4f)
                    << "RM TopK2 idx_dtype " << idx_dtype << " rank " << g_rank << " token " << i << " k " << k
                    << " expected=" << expected << " got=" << h_combined_w[i * kTopK2 + k];
            }

        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_recv_idx);
        ncclEpTensorDestroy(t_combined_x);
        ncclEpTensorDestroy(t_combined_w);
        ncclEpTensorDestroy(t_idx);
        cudaFree(d_tok);
        cudaFree(d_recv);
        cudaFree(d_weights);
        cudaFree(d_recv_w);
        cudaFree(d_recv_idx);
        cudaFree(d_combined_x);
        cudaFree(d_combined_w);
        cudaFree(d_idx);
        NCCL_ASSERT(ncclEpHandleDestroy(h));
    }
};

// CombineWithGrad + RM, top-k=2.
// Expect: combined_topk_weights[t, k] == source-side weight w2(g_rank, t, k).
TEST_F(HtBwdTopK2Test, CombineWithGradRankMajor) {
    run_combine_with_grad_rm(ncclInt64);
}

// Same round-trip with an int32 cached topk_idx (HT int32 support).
TEST_F(HtBwdTopK2Test, CombineWithGradRankMajorInt32) {
    run_combine_with_grad_rm(ncclInt32);
}

// CombineWithGrad + EM, top-k=2.
// EM FWD recv_topk_weights is 1D [num_recv]; feed that 1D tensor directly as
// the BWD combine input. Output combined_topk_weights stays 2D
// [num_combined, top_k], packed in ascending expert-id order.
TEST_F(HtBwdTopK2Test, CombineWithGradExpertMajor) {
    ncclEpHandle_t h = mk(true);
    ASSERT_NE(h, nullptr);

    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    std::vector<float> h_w(kNumTokens * kTopK2);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        for (int k = 0; k < kTopK2; ++k) h_w[i * kTopK2 + k] = w2(g_rank, i, k);
    }

    nv_bfloat16 *d_tok, *d_recv;
    float *d_weights, *d_recv_w_1d;
    CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights, kNumTokens * kTopK2 * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w_1d, kMaxRecvSlots * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv_w_1d, 0, kMaxRecvSlots * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), kNumTokens * kTopK2 * sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w_1d;
    NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_weights, kNumTokens, kTopK2));
    NCCL_ASSERT(epTensorCreate(&t_recv_w_1d, 1, ncclFloat32, d_recv_w_1d, kMaxRecvSlots));

    ncclEpDispatchInputs_t d_in_s = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens = t_tok;
    d_in_s.topk_weights = t_w;
    d_out_s.tokens = t_recv;
    d_out_s.topk_weights = t_recv_w_1d;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Feed the 1D EM recv_topk_weights DIRECTLY into BWD combine as input topk_weights.
    nv_bfloat16* d_combined_x = nullptr;
    float* d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK2 * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK2 * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32, d_combined_w, kNumTokens, kTopK2));

    ncclEpCombineInputs_t c_in_s = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens = t_recv;
    c_in_s.topk_weights = t_recv_w_1d;
    c_out_s.tokens = t_combined_x;
    c_out_s.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK2);
    CUDA_ASSERT(
        cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens * kTopK2 * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK2; ++k) {
            float expected = w2(g_rank, i, k);
            float got = h_combined_w[i * kTopK2 + k];
            if (std::abs(got - expected) > 1e-4f) {
                if (errors < 8) {
                    printf("[Rank %d] EM TopK2 1D-in token %d k %d: expected %.6f got %.6f\n", g_rank, i, k, expected,
                           got);
                }
                errors++;
            }
        }
    EXPECT_EQ(errors, 0) << "EM TopK2 BWD combine with 1D input mismatches";

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv_w_1d);
    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_tok);
    cudaFree(d_recv);
    cudaFree(d_weights);
    cudaFree(d_recv_w_1d);
    cudaFree(d_combined_x);
    cudaFree(d_combined_w);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── chunked combine_push coverage (multi-chunk path, BWD) ────────────────────
//
// combine_push's chunked (multi-chunk) branch needs a row wider than one tile; the shared kHidden=16
// group's row is a single tile. This fixture runs wide rows through the backward combine-with-grad round
// trip CombineWithGradExpertMajor uses: whole-row, forced chunk, default and reused whole-row combines of
// the same inputs must match each other and an analytic reference, combined_x included; each pass checks
// the chunk width the launcher resolved (launch record). Only meaningful under push-combine
// (NCCL_EP_HT_PULL_PUSH); skips otherwise.

// Sets an env var for its scope and restores the prior value (or unsets it) on destruction, even
// if an ASSERT_* mid-scope returns early -- gtest keeps running later tests in the same process.
struct ScopedEnvOverride {
    std::string name;
    bool had_prev = false;
    std::string prev;
    ScopedEnvOverride(const char* n, const char* v) : name(n) {
        if (const char* p = std::getenv(n)) {
            had_prev = true;
            prev = p;
        }
        if (v != nullptr) {
            setenv(n, v, 1);
        } else {
            unsetenv(n);
        }
    }
    ~ScopedEnvOverride() {
        if (had_prev) setenv(name.c_str(), prev.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

static void write_token_elem(void* base, int idx, ncclDataType_t dtype, float v) {
    if (dtype == ncclFloat32) {
        static_cast<float*>(base)[idx] = v;
    } else if (dtype == ncclFloat16) {
        static_cast<__half*>(base)[idx] = __float2half(v);
    } else {
        static_cast<nv_bfloat16*>(base)[idx] = __float2bfloat16(v);
    }
}

static float read_token_elem(const void* base, int idx, ncclDataType_t dtype) {
    if (dtype == ncclFloat32) {
        return static_cast<const float*>(base)[idx];
    }
    if (dtype == ncclFloat16) {
        return __half2float(static_cast<const __half*>(base)[idx]);
    }
    return bf16_val(static_cast<const nv_bfloat16*>(base)[idx]);
}

// Row- and absolute-column-dependent token value, kept a small integer so it is exact in
// bf16, fp16 and fp32 and whole-row and chunked combine reduce to identical bits. The hh/64 and
// hh%13 terms make a reordered chunk or tile change the value.
static float chunk_elem_val(int rank, int token, int hh) {
    const int v = ((rank + 1) * 40 + (token + 1) * 7 + (hh % 13) + (hh / 64)) % 240;
    return static_cast<float>(v);
}

struct CombineChunkCase {
    ncclDataType_t dtype;
    int hidden;
    const char* name;
    int chunk_tiles = 1;  // forced chunk width in tiles
};

struct CombineResult {
    std::vector<float> x;
    std::vector<float> w;
};

class HtBwdChunkedCombineTest : public ::testing::TestWithParam<CombineChunkCase> {
protected:
    ncclEpGroup_t group_ = nullptr;
    ncclEpTensor_t* topk_idx_em_ = nullptr;
    int64_t* d_topk_ = nullptr;

    void SetUp() override {
        if (!ht_pull_push_active()) {
            GTEST_SKIP() << "combine_push's chunked path only exists under NCCL_EP_HT_PULL_PUSH";
            return;
        }
        const CombineChunkCase& c = GetParam();
        ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
        gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        gcfg.num_experts = kNumExperts;
        gcfg.max_dispatch_tokens_per_rank = kNumTokens;
        gcfg.max_token_bytes = static_cast<unsigned int>(c.hidden * nccl_ep::ncclTypeSize(c.dtype));
        gcfg.rdma_buffer_size = NCCL_EP_AUTO;
        gcfg.num_qp_per_rank = NCCL_EP_AUTO;
        gcfg.num_channels = NCCL_EP_AUTO;
        gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecvSlots);
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));

        CUDA_ASSERT(cudaMalloc(&d_topk_, kNumTokens * kTopK * sizeof(int64_t)));
        int64_t h[kNumTokens];
        for (int i = 0; i < kNumTokens; ++i) {
            h[i] = expert_for_token(i);
        }
        CUDA_ASSERT(cudaMemcpy(d_topk_, h, sizeof(h), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_idx_em_, 2, ncclInt64, d_topk_, kNumTokens, kTopK));
    }

    void TearDown() override {
        if (topk_idx_em_) {
            ncclEpTensorDestroy(topk_idx_em_);
        }
        if (d_topk_) {
            cudaFree(d_topk_);
        }
        if (group_) {
            ncclEpGroupDestroy(group_);
        }
    }
};

TEST_P(HtBwdChunkedCombineTest, CombineWithGradExpertMajorChunked) {
    if (!ht_pull_push_active()) {
        return;
    }
    const CombineChunkCase& c = GetParam();
    const int eb = static_cast<int>(nccl_ep::ncclTypeSize(c.dtype));
    const int tile_elems = 2048 / eb;  // one tile = 128 int4
    ASSERT_GT(c.hidden, tile_elems) << c.name << " needs a multi-tile row to split into chunks";
    constexpr int kTileInt4 = 128;
    const int whole_row_int4 = (c.hidden * eb / 16 + kTileInt4 - 1) / kTileInt4 * kTileInt4;
    int cc_major = 0, cc_minor = 0, device = 0;
    CUDA_ASSERT(cudaGetDevice(&device));
    CUDA_ASSERT(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_ASSERT(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
    const unsigned int device_sm = static_cast<unsigned int>(cc_major * 10 + cc_minor);

    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&h, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    ASSERT_NE(h, nullptr);

    // Forward dispatch with row/column-dependent tokens and unique per-(rank, token) weights.
    const size_t tok_elems = static_cast<size_t>(kNumTokens) * c.hidden;
    const size_t recv_bytes = static_cast<size_t>(kMaxRecvSlots) * c.hidden * eb;
    std::vector<uint8_t> h_tok(tok_elems * eb);
    for (int i = 0; i < kNumTokens; ++i) {
        for (int hh = 0; hh < c.hidden; ++hh) {
            write_token_elem(h_tok.data(), i * c.hidden + hh, c.dtype, chunk_elem_val(g_rank, i, hh));
        }
    }
    std::vector<float> h_w(kNumTokens * kTopK);
    for (int i = 0; i < kNumTokens; ++i) {
        for (int k = 0; k < kTopK; ++k) {
            h_w[i * kTopK + k] = weight_for(g_rank, i, k);
        }
    }

    void* d_tok = nullptr;
    void* d_recv = nullptr;
    float* d_weights = nullptr;
    float* d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok, h_tok.size()));
    CUDA_ASSERT(cudaMalloc(&d_recv, recv_bytes));
    CUDA_ASSERT(cudaMalloc(&d_weights, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w, kMaxRecvSlots * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv, 0, recv_bytes));
    CUDA_ASSERT(cudaMemset(d_recv_w, 0, kMaxRecvSlots * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), h_tok.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w;
    NCCL_ASSERT(epTensorCreate(&t_tok, 2, c.dtype, d_tok, kNumTokens, c.hidden));
    NCCL_ASSERT(epTensorCreate(&t_recv, 2, c.dtype, d_recv, kMaxRecvSlots, c.hidden));
    NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_weights, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots));  // EM: 1D

    ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens = t_tok;
    d_in.topk_weights = t_w;
    d_out.tokens = t_recv;
    d_out.topk_weights = t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    void* d_combined_x = nullptr;
    float* d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, tok_elems * eb));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, c.dtype, d_combined_x, kNumTokens, c.hidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32, d_combined_w, kNumTokens, kTopK));

    ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens = t_recv;
    c_in.topk_weights = t_recv_w;  // 1D for EM
    c_out.tokens = t_combined_x;
    c_out.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;

    // One backward combine under a forced chunk size (null: unset); checks the launcher resolved
    // expected_chunk_int4 for this device, and returns combined_x/w as float.
    auto combine_pass = [&](const char* chunk_env, int expected_chunk_int4, CombineResult& out) {
        CUDA_ASSERT(cudaMemset(d_combined_x, 0, tok_elems * eb));
        CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));
        {
            ScopedEnvOverride force_chunk("NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE", chunk_env);
            NCCL_ASSERT(ncclEpCombine(h, &c_in, &c_out, &ccfg, g_stream));
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        }
        int chunk_int4 = 0;
        unsigned int launched_sm = 0;
        ncclEpGroup_test_getLastCombinePushLaunch(group_, &chunk_int4, &launched_sm);
        EXPECT_EQ(chunk_int4, expected_chunk_int4)
            << c.name << " chunk env " << (chunk_env != nullptr ? chunk_env : "(unset)");
        EXPECT_EQ(launched_sm, device_sm) << c.name << " launcher resolved the chunk for the wrong architecture";
        std::vector<uint8_t> raw_x(tok_elems * eb);
        CUDA_ASSERT(cudaMemcpy(raw_x.data(), d_combined_x, raw_x.size(), cudaMemcpyDeviceToHost));
        out.x.resize(tok_elems);
        for (size_t idx = 0; idx < tok_elems; ++idx) {
            out.x[idx] = read_token_elem(raw_x.data(), static_cast<int>(idx), c.dtype);
        }
        out.w.resize(kNumTokens * kTopK);
        CUDA_ASSERT(cudaMemcpy(out.w.data(), d_combined_w, out.w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    };

    // Whole row, forced multi-chunk, the unset default (one tile at this EP size on every
    // architecture), then whole row again: the last pass reuses the first pass's cached JIT
    // variant while alternating the chunk setting, so a stale cached kernel variant would
    // surface as a reuse mismatch.
    const std::string chunk_req = std::to_string(c.chunk_tiles * tile_elems);
    CombineResult whole, forced, by_default, reuse;
    combine_pass("0", whole_row_int4, whole);
    combine_pass(chunk_req.c_str(), c.chunk_tiles * kTileInt4, forced);
    combine_pass(nullptr, kTileInt4, by_default);
    combine_pass("0", whole_row_int4, reuse);

    for (int i = 0; i < kNumTokens; ++i) {
        for (int k = 0; k < kTopK; ++k) {
            EXPECT_NEAR(whole.w[i * kTopK + k], weight_for(g_rank, i, k), 1e-4f)
                << c.name << " whole-row weight token " << i << " k " << k;
        }
    }
    int ref_err = 0;
    for (int i = 0; i < kNumTokens; ++i) {
        for (int hh = 0; hh < c.hidden; ++hh) {
            const int idx = i * c.hidden + hh;
            const float expected = chunk_elem_val(g_rank, i, hh);
            if (std::abs(whole.x[idx] - expected) > 1e-3f) {
                if (ref_err < 8) {
                    printf("[Rank %d] %s whole-row x token %d hh %d: expected %.1f got %.1f\n", g_rank, c.name, i,
                           hh, expected, whole.x[idx]);
                }
                ref_err++;
            }
        }
    }
    EXPECT_EQ(ref_err, 0) << c.name << " combined_x disagrees with analytic reference";

    // Every other variant must reproduce the whole-row tokens and weights exactly.
    auto count_mismatches = [&](const char* label, const CombineResult& got) {
        int bad = 0;
        auto report = [&](const char* what, size_t idx, float want, float have) {
            if (bad++ < 8) {
                printf("[Rank %d] %s %s %s idx %zu: whole %g got %g\n", g_rank, c.name, label, what, idx, want, have);
            }
        };
        for (size_t idx = 0; idx < whole.w.size(); ++idx) {
            if (got.w[idx] != whole.w[idx]) {
                report("weight", idx, whole.w[idx], got.w[idx]);
            }
        }
        for (size_t idx = 0; idx < whole.x.size(); ++idx) {
            if (got.x[idx] != whole.x[idx]) {
                report("x", idx, whole.x[idx], got.x[idx]);
            }
        }
        return bad;
    };
    struct Variant {
        const char* label;
        const CombineResult* result;
    };
    for (const Variant& v : {Variant{"multi-chunk", &forced}, Variant{"default", &by_default},
                             Variant{"reused whole-row (stale JIT variant?)", &reuse}}) {
        EXPECT_EQ(count_mismatches(v.label, *v.result), 0) << c.name << " " << v.label << " != whole-row combine";
    }

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_tok);
    cudaFree(d_recv);
    cudaFree(d_weights);
    cudaFree(d_recv_w);
    cudaFree(d_combined_x);
    cudaFree(d_combined_w);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// Token dtype, row width and forced chunk width: exact-tile-multiple and partial final tile (hidden_int4
// not a multiple of the 128-int4 tile) widths, with one- and two-tile chunks.
INSTANTIATE_TEST_SUITE_P(
    DtypeWidth,
    HtBwdChunkedCombineTest,
    ::testing::Values(
        CombineChunkCase{ncclBfloat16, 4096, "bf16_exact"},
        CombineChunkCase{ncclBfloat16, 4096, "bf16_exact_two_tile_chunk", 2},
        CombineChunkCase{ncclBfloat16, 3584, "bf16_partial"},
        CombineChunkCase{ncclFloat32, 2048, "fp32_exact"},
        CombineChunkCase{ncclFloat32, 1792, "fp32_partial"},
        CombineChunkCase{ncclFloat16, 3584, "fp16_partial"}),
    [](const ::testing::TestParamInfo<CombineChunkCase>& info) { return std::string(info.param.name); });

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_bwd_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
