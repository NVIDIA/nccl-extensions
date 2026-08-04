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

#include <set>

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

// EM local-permute + alignment overflow: the padding-driven overrun from the OOB
// report (per-expert alignment padding pushes the padded total past capacity while
// the raw FLAT total still fits). The scan must emit -1 for over-capacity EM slots
// so the permute kernel never writes past the caller buffers (canary-verified),
// published counts/offsets must describe exactly the retained zones, and
// recv_total_counter must still report the true pre-drop padded requirement.
//
// Geometry (4 ranks, 8 experts => 2 local experts on rank 0, top-k 1, align 16,
// capacity 16): every rank routes token i -> global expert i%2, so rank 0 gets
// requested per-expert counts [8, 8] -> padded [16, 16] = 32 > 16, while the raw
// FLAT total is exactly 16 (no FLAT drop). Expert 0 keeps its full zone [0, 16)
// with 8 real rows; expert 1's zone is clamped away and all 8 of its assignments
// (em slots 16..23) must be dropped -- pre-fix these were the OOB writes.
TEST_F(HtOverflowDropTest, EmLocalPermuteAlignedDropNoOob) {
    if (g_nranks != 4) {
        GTEST_SKIP() << "zone geometry below assumes 4 ranks (8 experts -> 2 per rank)";
    }
    constexpr unsigned int kCap = 16;       // recv capacity; multiple of kAlign
    constexpr int kAlign = 16;              // per-expert zone alignment
    constexpr unsigned int kSlack = 64;     // canary rows past the caller buffer
    constexpr int kEpr = 2;                 // local experts on each rank

    // Dedicated non-zero-copy drop group (=> HT EM mode resolves to local-permute).
    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm                    = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts                  = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kNumTokens;
    gcfg.max_token_bytes              = kHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size             = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank              = NCCL_EP_AUTO;
    gcfg.num_channels                 = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank     = kCap;
    gcfg.overflow_policy              = NCCL_EP_OVERFLOW_DROP;
    ncclEpGroup_t grp = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&grp, g_comm, &gcfg));

    ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
    hcfg.dispatch_output_per_expert_alignment = kAlign;
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, grp, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, kTopK,
                                 /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);

    // token i -> global expert i%2 (both hosted on rank 0).
    int64_t h_idx[kNumTokens * kTopK];
    for (int i = 0; i < kNumTokens; ++i) h_idx[i] = i % 2;
    int64_t* d_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_idx, sizeof(h_idx)));
    CUDA_ASSERT(cudaMemcpy(d_idx, h_idx, sizeof(h_idx), cudaMemcpyHostToDevice));
    ncclEpTensor_t* t_idx = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_idx, 2, ncclInt64, d_idx, kNumTokens, kTopK));

    // Published-layout outputs: padded per-expert counts, padded offsets, true total.
    int32_t *d_cnt = nullptr, *d_off = nullptr, *d_total = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_cnt, kEpr * sizeof(int32_t)));
    CUDA_ASSERT(cudaMalloc(&d_off, kEpr * sizeof(int32_t)));
    CUDA_ASSERT(cudaMalloc(&d_total, sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_cnt, -1, kEpr * sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_off, -1, kEpr * sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_total, 0, sizeof(int32_t)));
    ncclEpTensor_t *t_cnt = nullptr, *t_off = nullptr, *t_total = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_cnt, 1, ncclInt32, d_cnt, kEpr));
    NCCL_ASSERT(epTensorCreate(&t_off, 1, ncclInt32, d_off, kEpr));
    NCCL_ASSERT(epTensorCreate(&t_total, 1, ncclInt32, d_total, 1));

    ncclEpLayoutInfo_t li = NCCL_EP_LAYOUT_INFO_INIT;
    li.expert_counters = t_cnt;
    li.expert_offsets = t_off;
    li.recv_total_counter = t_total;

    NCCL_ASSERT(ncclEpUpdateHandle(h, t_idx, &li, g_stream));
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": UpdateHandle trapped on padding-driven overflow";

    // Published layout: retained zones only; counter reports the pre-drop total.
    int32_t h_cnt[kEpr], h_off[kEpr], h_total = -1;
    CUDA_ASSERT(cudaMemcpy(h_cnt, d_cnt, sizeof(h_cnt), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_off, d_off, sizeof(h_off), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(&h_total, d_total, sizeof(h_total), cudaMemcpyDeviceToHost));
    if (g_rank == 0) {
        EXPECT_EQ(h_total, 32) << "true pre-drop padded requirement (2 x align(8,16))";
        EXPECT_EQ(h_cnt[0], 16) << "expert 0 keeps its full padded zone";
        EXPECT_EQ(h_cnt[1], 0) << "expert 1's zone is clamped away";
        EXPECT_EQ(h_off[0], 0);
        EXPECT_EQ(h_off[1], 16) << "expert 1's base clamps to capacity";
    } else {
        EXPECT_EQ(h_total, 0);
        EXPECT_EQ(h_cnt[0], 0);
        EXPECT_EQ(h_cnt[1], 0);
    }
    unsigned int num_recv = ~0u;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, (g_rank == 0) ? kCap : 0u);

    // Caller buffers with canary slack: rows [kCap, kCap+kSlack) must survive.
    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    std::vector<float> h_w(kNumTokens * kTopK);
    for (int i = 0; i < kNumTokens; ++i) {
        const float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        h_w[i] = static_cast<float>(g_rank * kNumTokens + i) + 0.5f;
    }
    const size_t recv_rows_alloc = kCap + kSlack;
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float *d_w = nullptr, *d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv, recv_rows_alloc * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w, recv_rows_alloc * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w, h_w.data(), kNumTokens * kTopK * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv, 0xAB, recv_rows_alloc * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv_w, 0xAB, recv_rows_alloc * sizeof(float)));

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_w = nullptr, *t_recv_w = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kCap, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_w, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kCap));

    ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens = t_tok;
    d_in.topk_weights = t_w;
    d_out.tokens = t_recv;
    d_out.topk_weights = t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    EXPECT_EQ(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": dispatch with dropped EM slots must not fault";

    // Readback: rank 0's retained zone has the 8 expert-0 tokens then zero pad;
    // canary rows past the caller buffer are untouched on every rank.
    {
        std::vector<nv_bfloat16> h_recv(recv_rows_alloc * kHidden);
        std::vector<float> h_recv_w(recv_rows_alloc);
        CUDA_ASSERT(cudaMemcpy(h_recv.data(), d_recv,
                               recv_rows_alloc * kHidden * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(h_recv_w.data(), d_recv_w, recv_rows_alloc * sizeof(float),
                               cudaMemcpyDeviceToHost));

        if (g_rank == 0) {
            std::multiset<float> got_vals, got_ws, want_vals, want_ws;
            for (int s = 0; s < 8; ++s) {
                got_vals.insert(__bfloat162float(h_recv[s * kHidden]));
                got_ws.insert(h_recv_w[s]);
            }
            for (int r = 0; r < g_nranks; ++r)
                for (int i = 0; i < kNumTokens; i += 2) {  // expert-0 assignments
                    want_vals.insert(static_cast<float>(r * kNumTokens + i + 1));
                    want_ws.insert(static_cast<float>(r * kNumTokens + i) + 0.5f);
                }
            EXPECT_EQ(got_vals, want_vals) << "retained zone must hold the expert-0 tokens";
            EXPECT_EQ(got_ws, want_ws) << "retained zone weights must match their tokens";
            for (unsigned int s = 8; s < kCap; ++s) {
                EXPECT_EQ(__bfloat162float(h_recv[s * kHidden]), 0.0f) << "pad row " << s;
                EXPECT_EQ(h_recv_w[s], 0.0f) << "pad weight " << s;
            }
        }
        // Canary: any nonzero-vs-0xAB mismatch past the caller buffer is an OOB write.
        const uint8_t* recv_bytes = reinterpret_cast<const uint8_t*>(h_recv.data());
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(h_recv_w.data());
        size_t tok_bad = 0, w_bad = 0;
        for (size_t b = kCap * kHidden * sizeof(nv_bfloat16);
             b < recv_rows_alloc * kHidden * sizeof(nv_bfloat16); ++b) {
            if (recv_bytes[b] != 0xAB) ++tok_bad;
        }
        for (size_t b = kCap * sizeof(float); b < recv_rows_alloc * sizeof(float); ++b) {
            if (w_bytes[b] != 0xAB) ++w_bad;
        }
        EXPECT_EQ(tok_bad, 0u) << "Rank " << g_rank
                               << ": OOB write past recv_tokens (bytes clobbered)";
        EXPECT_EQ(w_bad, 0u) << "Rank " << g_rank
                             << ": OOB write past recv_topk_weights (bytes clobbered)";
    }

    // Combine (identity expert): dropped assignments contribute zero, so token i
    // returns its own value when i is even (expert 0, retained) and 0 when i is
    // odd (expert 1, dropped). Also exercises the reduce over -1 map entries.
    {
        nv_bfloat16* d_out_tok = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_out_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(d_out_tok, 0xCD, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        ncclEpTensor_t* t_out_tok = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_out_tok, 2, ncclBfloat16, d_out_tok, kNumTokens, kHidden));

        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = t_recv;
        c_out.tokens = t_out_tok;
        EXPECT_EQ(ncclEpCombine(h, &c_in, &c_out, /*config=*/nullptr, g_stream), ncclSuccess);
        ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
            << "Rank " << g_rank << ": combine after a dropped dispatch must not fault";

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out_tok, kNumTokens * kHidden * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < kNumTokens; ++i) {
            const float expected =
                (i % 2 == 0) ? static_cast<float>(g_rank * kNumTokens + i + 1) : 0.0f;
            EXPECT_EQ(__bfloat162float(h_out[i * kHidden]), expected)
                << "Rank " << g_rank << " token " << i
                << ": retained assignment round-trips, dropped assignment is zero";
        }
        ncclEpTensorDestroy(t_out_tok);
        cudaFree(d_out_tok);
    }

    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_tok);
    cudaFree(d_recv_w);
    cudaFree(d_w);
    cudaFree(d_recv);
    cudaFree(d_tok);
    ncclEpTensorDestroy(t_total);
    ncclEpTensorDestroy(t_off);
    ncclEpTensorDestroy(t_cnt);
    cudaFree(d_total);
    cudaFree(d_off);
    cudaFree(d_cnt);
    ncclEpTensorDestroy(t_idx);
    cudaFree(d_idx);
    (void)ncclEpHandleDestroy(h);
    NCCL_ASSERT(ncclEpGroupDestroy(grp));
}

// EM local-permute DEEP FLAT overflow: the raw (unpadded) recv count itself
// exceeds capacity, so FLAT-dropped tokens leave holes inside the published
// per-expert row counts ("phantom rows") that neither the permute copy nor the
// pad warp covers — the drop-mode zero-init must turn them into zero rows.
//
// Geometry (4 ranks, 8 experts => 2 local experts on rank 0, top-k 1, align 4,
// capacity 8): every rank routes tokens 0,1 -> expert 0 and tokens 2,3 ->
// expert 1 (both on rank 0). Rank 0's raw FLAT total is 16 > 8, so only ranks
// 0-1's tokens survive the FLAT drop. Expert 0's zone [0, 8) publishes 8 rows
// (clamped requested) but only 4 are delivered (r0t0, r0t1, r1t0, r1t1) —
// rows 4..7 are phantom and must read back zero. Expert 1's zone is clamped
// away; ranks 2-3 are fully dropped and must combine back zeros.
TEST_F(HtOverflowDropTest, EmLocalPermuteDeepFlatOverflowPhantomRowsZeroed) {
    if (g_nranks != 4) {
        GTEST_SKIP() << "zone geometry below assumes 4 ranks (8 experts -> 2 per rank)";
    }
    constexpr unsigned int kCap = 8;
    constexpr int kAlign = 4;
    constexpr unsigned int kSlack = 64;
    constexpr int kEpr = 2;

    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm                    = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts                  = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kNumTokens;
    gcfg.max_token_bytes              = kHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size             = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank              = NCCL_EP_AUTO;
    gcfg.num_channels                 = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank     = kCap;
    gcfg.overflow_policy              = NCCL_EP_OVERFLOW_DROP;
    ncclEpGroup_t grp = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&grp, g_comm, &gcfg));

    ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
    hcfg.dispatch_output_per_expert_alignment = kAlign;
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, grp, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, kTopK,
                                 /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);

    // tokens 0,1 -> expert 0; tokens 2,3 -> expert 1.
    int64_t h_idx[kNumTokens * kTopK];
    for (int i = 0; i < kNumTokens; ++i) h_idx[i] = (i < 2) ? 0 : 1;
    int64_t* d_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_idx, sizeof(h_idx)));
    CUDA_ASSERT(cudaMemcpy(d_idx, h_idx, sizeof(h_idx), cudaMemcpyHostToDevice));
    ncclEpTensor_t* t_idx = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_idx, 2, ncclInt64, d_idx, kNumTokens, kTopK));

    int32_t* d_total = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_total, sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_total, 0, sizeof(int32_t)));
    ncclEpTensor_t* t_total = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_total, 1, ncclInt32, d_total, 1));
    ncclEpLayoutInfo_t li = NCCL_EP_LAYOUT_INFO_INIT;
    li.recv_total_counter = t_total;

    NCCL_ASSERT(ncclEpUpdateHandle(h, t_idx, &li, g_stream));
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": UpdateHandle trapped on deep FLAT overflow";

    int32_t h_total = -1;
    CUDA_ASSERT(cudaMemcpy(&h_total, d_total, sizeof(h_total), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_total, (g_rank == 0) ? 16 : 0)
        << "true pre-drop padded requirement (align(8,4) + align(8,4))";

    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    std::vector<float> h_w(kNumTokens * kTopK);
    for (int i = 0; i < kNumTokens; ++i) {
        const float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        h_w[i] = static_cast<float>(g_rank * kNumTokens + i) + 0.5f;
    }
    const size_t recv_rows_alloc = kCap + kSlack;
    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
    float *d_w = nullptr, *d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv, recv_rows_alloc * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w, recv_rows_alloc * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w, h_w.data(), kNumTokens * kTopK * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv, 0xAB, recv_rows_alloc * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv_w, 0xAB, recv_rows_alloc * sizeof(float)));

    // Declare the recv descriptors LARGER than capacity (the API allows slack):
    // rows [kCap, kCap + kDeclSlack) are caller-owned, inside the canary region,
    // and must never be written — not by the kernels and not by the drop-mode
    // zero-init (which must bound itself to the configured capacity).
    constexpr unsigned int kDeclSlack = 4;
    static_assert(kDeclSlack <= kSlack, "declared slack must stay inside the canary region");
    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_w = nullptr, *t_recv_w = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kCap + kDeclSlack, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_w, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kCap + kDeclSlack));

    ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens = t_tok;
    d_in.topk_weights = t_w;
    d_out.tokens = t_recv;
    d_out.topk_weights = t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    EXPECT_EQ(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
    ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
        << "Rank " << g_rank << ": deep-FLAT-overflow dispatch must not fault";

    {
        std::vector<nv_bfloat16> h_recv(recv_rows_alloc * kHidden);
        std::vector<float> h_recv_w(recv_rows_alloc);
        CUDA_ASSERT(cudaMemcpy(h_recv.data(), d_recv,
                               recv_rows_alloc * kHidden * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(h_recv_w.data(), d_recv_w, recv_rows_alloc * sizeof(float),
                               cudaMemcpyDeviceToHost));

        if (g_rank == 0) {
            // Delivered: the FLAT-retained expert-0 assignments (ranks 0-1, tokens 0-1).
            std::multiset<float> got_vals, got_ws, want_vals, want_ws;
            for (int s = 0; s < 4; ++s) {
                got_vals.insert(__bfloat162float(h_recv[s * kHidden]));
                got_ws.insert(h_recv_w[s]);
            }
            for (int r = 0; r < 2; ++r)
                for (int i = 0; i < 2; ++i) {
                    want_vals.insert(static_cast<float>(r * kNumTokens + i + 1));
                    want_ws.insert(static_cast<float>(r * kNumTokens + i) + 0.5f);
                }
            EXPECT_EQ(got_vals, want_vals) << "delivered rows must be the retained expert-0 tokens";
            EXPECT_EQ(got_ws, want_ws) << "delivered weights must match their tokens";
            // Phantom rows: published count 8 covers them, nothing writes them —
            // the drop-mode zero-init must have cleared the 0xAB canary fill.
            for (unsigned int s = 4; s < kCap; ++s) {
                EXPECT_EQ(__bfloat162float(h_recv[s * kHidden]), 0.0f) << "phantom row " << s;
                EXPECT_EQ(h_recv_w[s], 0.0f) << "phantom weight " << s;
            }
        }
        const uint8_t* recv_bytes = reinterpret_cast<const uint8_t*>(h_recv.data());
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(h_recv_w.data());
        size_t tok_bad = 0, w_bad = 0;
        for (size_t b = kCap * kHidden * sizeof(nv_bfloat16);
             b < recv_rows_alloc * kHidden * sizeof(nv_bfloat16); ++b) {
            if (recv_bytes[b] != 0xAB) ++tok_bad;
        }
        for (size_t b = kCap * sizeof(float); b < recv_rows_alloc * sizeof(float); ++b) {
            if (w_bytes[b] != 0xAB) ++w_bad;
        }
        EXPECT_EQ(tok_bad, 0u) << "Rank " << g_rank << ": OOB write past recv_tokens";
        EXPECT_EQ(w_bad, 0u) << "Rank " << g_rank << ": OOB write past recv_topk_weights";
    }

    // Combine (identity): ranks 0-1 get tokens 0,1 back and zeros for 2,3 (em
    // dropped); ranks 2-3 were fully FLAT-dropped, their combine gate is cleared,
    // so the preset zeros must remain untouched.
    {
        nv_bfloat16* d_out_tok = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_out_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(d_out_tok, 0, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        ncclEpTensor_t* t_out_tok = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_out_tok, 2, ncclBfloat16, d_out_tok, kNumTokens, kHidden));

        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = t_recv;
        c_out.tokens = t_out_tok;
        EXPECT_EQ(ncclEpCombine(h, &c_in, &c_out, /*config=*/nullptr, g_stream), ncclSuccess);
        ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess)
            << "Rank " << g_rank << ": combine after deep FLAT overflow must not fault";

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out_tok, kNumTokens * kHidden * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < kNumTokens; ++i) {
            const float expected = (g_rank < 2 && i < 2)
                ? static_cast<float>(g_rank * kNumTokens + i + 1) : 0.0f;
            EXPECT_EQ(__bfloat162float(h_out[i * kHidden]), expected)
                << "Rank " << g_rank << " token " << i;
        }
        ncclEpTensorDestroy(t_out_tok);
        cudaFree(d_out_tok);
    }

    ncclEpTensorDestroy(t_recv_w);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_tok);
    cudaFree(d_recv_w);
    cudaFree(d_w);
    cudaFree(d_recv);
    cudaFree(d_tok);
    ncclEpTensorDestroy(t_total);
    cudaFree(d_total);
    ncclEpTensorDestroy(t_idx);
    cudaFree(d_idx);
    (void)ncclEpHandleDestroy(h);
    NCCL_ASSERT(ncclEpGroupDestroy(grp));
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
