/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests for the HT recv-overflow policy (ncclEpGroupConfig_t::overflow_policy).
 *
 * Default policy (NCCL_EP_OVERFLOW_AUTO, resolving to TRAP) device-traps when a rank receives
 * more tokens than max_recv_tokens_per_rank. NCCL_EP_OVERFLOW_DROP instead drops
 * the overflowing tokens and lets dispatch + combine continue without a hard
 * error, reporting the true (pre-drop) recv total via recv_total_counter.
 *
 * Recipe (4 ranks, FLAT layout):
 *   - Build a group with overflow_policy = NCCL_EP_OVERFLOW_DROP and a recv
 *     budget (kDropRecvBudget) smaller than the worst-case incoming load.
 *   - Route every token on every rank to global expert 0 (rank 0, local 0), so
 *     rank 0 receives nranks * kNumTokens tokens — far above the budget — while
 *     the other ranks receive none.
 *   - ncclEpUpdateHandle must NOT trap; recv_total_counter on rank 0 reports the
 *     true total, and the internal recv count is clamped to the budget.
 *   - A full dispatch -> complete -> combine cycle must complete with cudaSuccess.
 *
 * The FAIL path is intentionally not exercised here: a device __trap() aborts the
 * whole process, which cannot be asserted in-process.
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"

// Recv budget for the drop group. Must be >= max_dispatch_tokens_per_rank
// (kNumTokens) per ncclEpCreateGroup's HT constraint. Chosen below the
// all-to-expert-0 load (g_nranks * kNumTokens) so the test forces an overflow
// whenever run with >= 3 ranks (run_tests.sh uses 4).
static constexpr unsigned int kDropRecvBudget = 8;

// Drop-policy group, created collectively in main() on g_comm.
static ncclEpGroup_t g_ep_group_drop = nullptr;
// Expert-major drop-policy group forced onto the non-permute (nvlink_dup) EM path,
// so the EM scan owns the s2d and must clear the combine gate itself. Created in
// main() with NCCL_EP_HT_EM_NVLINK_DUP=1.
static ncclEpGroup_t g_ep_group_em_drop = nullptr;

class HtOverflowDropTest : public ::testing::Test {
protected:
    int64_t*        d_topk_ = nullptr;   // [kNumTokens, kTopK], all routed to expert 0
    ncclEpTensor_t* t_topk_ = nullptr;

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk_, kNumTokens * kTopK * sizeof(int64_t)));
        int64_t h[kNumTokens * kTopK] = {0};  // every token -> global expert 0
        CUDA_ASSERT(cudaMemcpy(d_topk_, h, sizeof(h), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&t_topk_, 2, ncclInt64, d_topk_, kNumTokens, kTopK));
    }

    void TearDown() override {
        if (t_topk_) ncclEpTensorDestroy(t_topk_);
        if (d_topk_) cudaFree(d_topk_);
    }
};

// Dispatch + combine continue (no trap / no CUDA error) when a rank overflows
// its recv budget under NCCL_EP_OVERFLOW_DROP, and the drop is accounted for.
TEST_F(HtOverflowDropTest, DispatchCombineContinueOnOverflow) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group_drop, NCCL_EP_LAYOUT_FLAT,
                                 /*config=*/nullptr, kTopK, /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);

    // recv_total_counter: true (pre-drop) per-rank recv total, written by the
    // preprocessing scan in ncclEpUpdateHandle.
    int32_t* d_recv_total = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_recv_total, sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_recv_total, 0, sizeof(int32_t)));
    ncclEpTensor_t* t_recv_total = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_recv_total, 1, ncclInt32, d_recv_total, 1));

    ncclEpLayoutInfo_t li = NCCL_EP_LAYOUT_INFO_INIT;
    li.recv_total_counter = t_recv_total;

    // UpdateHandle runs the scan; with DROP it must not trap on overflow.
    NCCL_ASSERT(ncclEpUpdateHandle(h, t_topk_, &li, g_stream));
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": ncclEpUpdateHandle trapped on recv overflow; "
           "NCCL_EP_OVERFLOW_DROP should drop tokens instead of __trap().";

    // All tokens route to global expert 0 (rank 0). Rank 0 receives every token;
    // all other ranks receive none.
    const int32_t expected_true_total = (g_rank == 0) ? (g_nranks * kNumTokens) : 0;
    // Verify the test actually forces an overflow on rank 0.
    if (g_rank == 0) {
        ASSERT_GT(expected_true_total, static_cast<int32_t>(kDropRecvBudget))
            << "Test misconfigured: recv budget must be below total arriving tokens.";
    }
    const int32_t expected_kept =
        (static_cast<unsigned>(expected_true_total) > kDropRecvBudget)
            ? static_cast<int32_t>(kDropRecvBudget)
            : expected_true_total;

    int32_t h_recv_total = -1;
    CUDA_ASSERT(cudaMemcpy(&h_recv_total, d_recv_total, sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_recv_total, expected_true_total)
        << "Rank " << g_rank << ": recv_total_counter should report the true "
           "pre-drop recv total so callers can detect dropped tokens.";

    // Internal recv count is clamped to the budget (overflow tokens dropped).
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, static_cast<unsigned int>(expected_kept))
        << "Rank " << g_rank << ": internal recv count must be clamped to "
           "max_recv_tokens_per_rank on overflow.";

    // ── Forward dispatch ──────────────────────────────────────────────────────
    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
    }
    std::vector<float> h_w(kNumTokens * kTopK, 1.0f);

    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float       *d_w = nullptr, *d_recv_w = nullptr;
    int64_t     *d_recv_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens       * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,     kDropRecvBudget  * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv, 0,   kDropRecvBudget  * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens       * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   kDropRecvBudget  * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv_w, 0, kDropRecvBudget  * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx, kDropRecvBudget  * kTopK   * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(),
                           kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w, h_w.data(),
                           kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr,
                   *t_w = nullptr, *t_recv_w = nullptr, *t_recv_idx = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,      2, ncclBfloat16, d_tok,      kNumTokens,      kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,     2, ncclBfloat16, d_recv,     kDropRecvBudget, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,        2, ncclFloat32,  d_w,        kNumTokens,      kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w,   2, ncclFloat32,  d_recv_w,   kDropRecvBudget, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64,    d_recv_idx, kDropRecvBudget, kTopK));

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens        = t_tok;
    d_in.topk_weights  = t_w;
    d_out.tokens       = t_recv;
    d_out.topk_weights = t_recv_w;
    d_out.topk_idx     = t_recv_idx;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;

    EXPECT_EQ(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": dispatch must complete cleanly after dropping "
           "overflow tokens (got CUDA error "
        << cudaGetErrorName(cudaGetLastError()) << ").";

    // ── Forward combine (round-trip back to the senders) ──────────────────────
    nv_bfloat16* d_combined = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_combined, 0, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    ncclEpTensor_t* t_combined = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_combined, 2, ncclBfloat16, d_combined, kNumTokens, kHidden));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens   = t_recv;       // expert outputs (identity here)
    c_out.tokens  = t_combined;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;

    EXPECT_EQ(ncclEpCombine(h, &c_in, &c_out, &ccfg, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": combine must complete cleanly after a dropping "
           "dispatch (got CUDA error "
        << cudaGetErrorName(cudaGetLastError()) << ").";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    ncclEpTensorDestroy(t_combined);
    cudaFree(d_combined);
    ncclEpTensorDestroy(t_recv_idx);
    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_tok);
    cudaFree(d_recv_idx);
    cudaFree(d_recv_w);
    cudaFree(d_w);
    cudaFree(d_recv);
    cudaFree(d_tok);
    ncclEpTensorDestroy(t_recv_total);
    cudaFree(d_recv_total);
    (void)ncclEpHandleDestroy(h);
}

// The dispatch output (recv_x) must be at least max_recv_tokens_per_rank rows;
// an undersized buffer is rejected with ncclInvalidArgument before any kernel runs.
TEST_F(HtOverflowDropTest, UndersizedRecvOutputRejected) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group_drop, NCCL_EP_LAYOUT_FLAT,
                                 /*config=*/nullptr, kTopK, /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);
    NCCL_ASSERT(ncclEpUpdateHandle(h, t_topk_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // recv_x with one fewer row than the budget.
    const unsigned int undersized = kDropRecvBudget - 1;
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float       *d_w = nullptr, *d_recv_w = nullptr;
    int64_t     *d_recv_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,     undersized * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   undersized * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx, undersized * kTopK   * sizeof(int64_t)));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr,
                   *t_w = nullptr, *t_recv_w = nullptr, *t_recv_idx = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,      2, ncclBfloat16, d_tok,      kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,     2, ncclBfloat16, d_recv,     undersized, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,        2, ncclFloat32,  d_w,        kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w,   2, ncclFloat32,  d_recv_w,   undersized, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64,    d_recv_idx, undersized, kTopK));

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens        = t_tok;
    d_in.topk_weights  = t_w;
    d_out.tokens       = t_recv;
    d_out.topk_weights = t_recv_w;
    d_out.topk_idx     = t_recv_idx;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;

    EXPECT_EQ(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclInvalidArgument)
        << "Rank " << g_rank << ": dispatch must reject a recv_x smaller than "
           "max_recv_tokens_per_rank.";

    ncclEpTensorDestroy(t_recv_idx);
    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_tok);
    cudaFree(d_recv_idx);
    cudaFree(d_recv_w);
    cudaFree(d_w);
    cudaFree(d_recv);
    cudaFree(d_tok);
    (void)ncclEpHandleDestroy(h);
}

// EM path: an undersized recv_x is rejected with ncclInvalidArgument (same host-side
// validation as FLAT; confirms the check is not gated on overflow_policy).
TEST_F(HtOverflowDropTest, EmUndersizedRecvOutputRejected) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group_em_drop, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                 /*config=*/nullptr, kTopK, /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);
    NCCL_ASSERT(ncclEpUpdateHandle(h, t_topk_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    const unsigned int undersized = kDropRecvBudget - 1;
    const size_t recv_bytes = static_cast<size_t>(undersized) * kHidden * sizeof(nv_bfloat16);
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float       *d_w = nullptr, *d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens * kHidden * sizeof(nv_bfloat16)));
    NCCL_ASSERT(ncclMemAlloc(reinterpret_cast<void**>(&d_recv), recv_bytes));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   undersized           * sizeof(float)));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_w = nullptr, *t_recv_w = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,    2, ncclBfloat16, d_tok,    kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,   2, ncclBfloat16, /*data=*/nullptr, undersized, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,      2, ncclFloat32,  d_w,      kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32,  d_recv_w, undersized));

    ncclWindow_t recv_win{};
    NCCL_ASSERT(ncclCommWindowRegister(g_comm, d_recv, recv_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
    t_recv->win_hdl    = recv_win;
    t_recv->win_offset = 0;

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens       = t_tok;
    d_in.topk_weights = t_w;
    d_out.tokens      = t_recv;
    d_out.topk_weights = t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;

    EXPECT_EQ(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclInvalidArgument)
        << "Rank " << g_rank << ": EM dispatch must reject a recv_x smaller than "
           "max_recv_tokens_per_rank.";

    (void)ncclCommWindowDeregister(g_comm, recv_win);
    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_tok);
    cudaFree(d_recv_w);
    cudaFree(d_w);
    ncclMemFree(d_recv);
    cudaFree(d_tok);
    (void)ncclEpHandleDestroy(h);
}

// Same overflow recipe on the non-permute expert-major path (nvlink_dup, zero_copy):
// the EM scan writes the authoritative s2d and must clear rdma_to_attn_map for any
// send token whose every local-expert slot was dropped, so combine's producer and
// consumer skip it in lockstep instead of deadlocking. recv slots index the
// window-backed recv buffer directly, so a slot past the budget is dropped rather
// than written out of bounds.
TEST_F(HtOverflowDropTest, EmDispatchCombineContinueOnOverflow) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group_em_drop, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                 /*config=*/nullptr, kTopK, /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);

    int32_t* d_recv_total = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_recv_total, sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_recv_total, 0, sizeof(int32_t)));
    ncclEpTensor_t* t_recv_total = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_recv_total, 1, ncclInt32, d_recv_total, 1));

    ncclEpLayoutInfo_t li = NCCL_EP_LAYOUT_INFO_INIT;
    li.recv_total_counter = t_recv_total;

    NCCL_ASSERT(ncclEpUpdateHandle(h, t_topk_, &li, g_stream));
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": EM ncclEpUpdateHandle trapped on recv overflow.";

    // All tokens route to global expert 0 (rank 0, local expert 0). Rank 0's local
    // expert 0 receives every token; the EM padded total (align=1) equals that load.
    const int32_t expected_true_total = (g_rank == 0) ? (g_nranks * kNumTokens) : 0;
    // Verify the test actually forces an overflow on rank 0.
    if (g_rank == 0) {
        ASSERT_GT(expected_true_total, static_cast<int32_t>(kDropRecvBudget))
            << "Test misconfigured: recv budget must be below total arriving tokens.";
    }
    const int32_t expected_kept =
        (static_cast<unsigned>(expected_true_total) > kDropRecvBudget)
            ? static_cast<int32_t>(kDropRecvBudget)
            : expected_true_total;

    int32_t h_recv_total = -1;
    CUDA_ASSERT(cudaMemcpy(&h_recv_total, d_recv_total, sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_recv_total, expected_true_total)
        << "Rank " << g_rank << ": EM recv_total_counter should report the true "
           "pre-drop recv total.";

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, static_cast<unsigned int>(expected_kept))
        << "Rank " << g_rank << ": EM internal recv count must be clamped to the budget.";

    // ── Forward dispatch (expert-major: 1-D recv weights, no topk_idx).
    // recv buffer is a symmetric window (zero_copy requires window-backed recv_x).
    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
    }
    std::vector<float> h_w(kNumTokens * kTopK, 1.0f);

    const size_t recv_bytes = static_cast<size_t>(kDropRecvBudget) * kHidden * sizeof(nv_bfloat16);
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float       *d_w = nullptr, *d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens      * kHidden * sizeof(nv_bfloat16)));
    NCCL_ASSERT(ncclMemAlloc(reinterpret_cast<void**>(&d_recv), recv_bytes));
    CUDA_ASSERT(cudaMemset(d_recv, 0,   recv_bytes));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens      * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   kDropRecvBudget          * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv_w, 0, kDropRecvBudget          * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(),
                           kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w, h_w.data(),
                           kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_w = nullptr, *t_recv_w = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok,    2, ncclBfloat16, d_tok,    kNumTokens,      kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,   2, ncclBfloat16, /*data=*/nullptr, kDropRecvBudget, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,      2, ncclFloat32,  d_w,      kNumTokens,      kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32,  d_recv_w, kDropRecvBudget));

    // Window-register the recv buffer; the same window backs dispatch output and
    // combine input.
    ncclWindow_t recv_win{};
    NCCL_ASSERT(ncclCommWindowRegister(g_comm, d_recv, recv_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
    t_recv->win_hdl    = recv_win;
    t_recv->win_offset = 0;

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens        = t_tok;
    d_in.topk_weights  = t_w;
    d_out.tokens       = t_recv;
    d_out.topk_weights = t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;

    EXPECT_EQ(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": EM dispatch must complete cleanly after dropping "
           "overflow tokens (got CUDA error "
        << cudaGetErrorName(cudaGetLastError()) << ").";

    // ── Forward combine (round-trip back to the senders) ──────────────────────
    nv_bfloat16* d_combined = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_combined, 0, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    ncclEpTensor_t* t_combined = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_combined, 2, ncclBfloat16, d_combined, kNumTokens, kHidden));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens  = t_recv;       // expert outputs (identity here), window-backed
    c_out.tokens = t_combined;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;

    EXPECT_EQ(ncclEpCombine(h, &c_in, &c_out, &ccfg, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": EM combine must complete cleanly (no deadlock) after a "
           "dropping dispatch (got CUDA error "
        << cudaGetErrorName(cudaGetLastError()) << ").";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    ncclEpTensorDestroy(t_combined);
    cudaFree(d_combined);
    (void)ncclCommWindowDeregister(g_comm, recv_win);
    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_tok);
    cudaFree(d_recv_w);
    cudaFree(d_w);
    ncclMemFree(d_recv);
    cudaFree(d_tok);
    ncclEpTensorDestroy(t_recv_total);
    cudaFree(d_recv_total);
    (void)ncclEpHandleDestroy(h);
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_overflow_drop_uid")) return 0;

    // Drop-policy group on the shared communicator. Collective across ranks.
    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm                    = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts                  = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kNumTokens;
    gcfg.max_token_bytes              = kHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size             = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank              = NCCL_EP_AUTO;
    gcfg.num_channels                 = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank     = kDropRecvBudget;
    gcfg.overflow_policy              = NCCL_EP_OVERFLOW_DROP;
    if (ncclEpCreateGroup(&g_ep_group_drop, g_comm, &gcfg) != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclEpCreateGroup (drop) failed.\n", g_rank);
        ep_teardown();
        return 1;
    }

    // Expert-major drop group on the non-permute (nvlink_dup) EM path. zero_copy=ON
    // auto-selects nvlink_dup for lsa>1 and, unlike the library-staged path, lets
    // max_recv_tokens stay below the worst-case load (recv slots index the user
    // window directly), so an overflow can actually occur and be dropped.
    ncclEpGroupConfig_t gcfg_em = gcfg;
    gcfg_em.zero_copy = NCCL_EP_ZERO_COPY_ON;
    ncclResult_t em_ret = ncclEpCreateGroup(&g_ep_group_em_drop, g_comm, &gcfg_em);
    if (em_ret != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclEpCreateGroup (EM drop) failed.\n", g_rank);
        ncclEpGroupDestroy(g_ep_group_drop);
        ep_teardown();
        return 1;
    }

    int ret = RUN_ALL_TESTS();

    ncclEpGroupDestroy(g_ep_group_em_drop);
    ncclEpGroupDestroy(g_ep_group_drop);
    ep_teardown();
    return ret;
}
