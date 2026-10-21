/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression test for interleaved (1F1B / PP) HT combine on two live handles.
 *
 *   Under NCCL_EP_HT_PULL_PUSH the push combine's reduce derives which (t,R)
 *   partials exist from this rank's own uint16 topk routing. That routing must be
 *   handle-private: with two handles alive, the later handle's ncclEpUpdateHandle
 *   would otherwise clobber a shared per-rank routing buffer before the earlier
 *   handle's combine runs, so the earlier handle's reduce reads the wrong routing
 *   and gathers the wrong staging rows.
 *
 *   Fixed by snapshotting the uint16 topk map into handle->ht.topk_idx per handle;
 *   combine reads that slot.
 *
 * Test recipe (expert-major; the race only exists under pull-push, other EM modes
 * take a different combine path and pass trivially):
 *   1. Create handle A (routing R_A) then handle B (routing R_B), R_A != R_B per
 *      token. A is created first, so any shared per-rank routing buffer ends up
 *      holding R_B by the time A's combines run.
 *   2. Forward dispatch A and B, retaining each handle's recv tokens + recv
 *      topk_weights and its unique source values.
 *   3. Run two forward combines back-to-back (A then B), then two backward
 *      combines back-to-back (A then B). Forward combine must recover each
 *      handle's original token values; backward combine must recover each
 *      handle's original topk_weights. If A's reduce read R_B, A's results are
 *      wrong.
 *
 * Deterministic: B is the last handle created, so a shared-buffer read is always
 * R_B, not R_A.
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"

static float bf16_val(nv_bfloat16 v) {
    return __bfloat162float(v);
}

class HtCombinePpInterleaveTest : public ::testing::Test {
protected:
    // Distinct per-handle routing so a crossover produces a detectable mismatch.
    static int64_t route_a(int token) { return (g_rank * kNumTokens + token) % kNumExperts; }
    static int64_t route_b(int token) { return (g_rank * kNumTokens + token + 1) % kNumExperts; }

    // Unique per-(handle, rank, token, k) weight so BWD combine round-trips are
    // not a 1.0==1.0 false pass.
    static float weight_for(int which, int token, int k) {
        return 0.125f * static_cast<float>((which + 1) * 1000 + (g_rank + 1) * 100 + (token + 1) * 10 + k);
    }

    // A forward-dispatched handle whose recv tokens/weights are combined later.
    struct Fwd {
        int which = 0;         // 0 = A, 1 = B
        float tok_base = 0.f;  // token i value = tok_base + i
        int64_t* d_topk = nullptr;
        nv_bfloat16* d_tok = nullptr;
        nv_bfloat16* d_recv = nullptr;
        float* d_weights = nullptr;
        float* d_recv_w = nullptr;
        ncclEpTensor_t* t_topk = nullptr;
        ncclEpTensor_t* t_tok = nullptr;
        ncclEpTensor_t* t_recv = nullptr;
        ncclEpTensor_t* t_w = nullptr;
        ncclEpTensor_t* t_recv_w = nullptr;
        ncclEpHandle_t handle = nullptr;

        void free_all() {
            if (t_recv_w) ncclEpTensorDestroy(t_recv_w);
            if (t_w) ncclEpTensorDestroy(t_w);
            if (t_recv) ncclEpTensorDestroy(t_recv);
            if (t_tok) ncclEpTensorDestroy(t_tok);
            if (t_topk) ncclEpTensorDestroy(t_topk);
            if (d_recv_w) cudaFree(d_recv_w);
            if (d_weights) cudaFree(d_weights);
            if (d_recv) cudaFree(d_recv);
            if (d_tok) cudaFree(d_tok);
            if (d_topk) cudaFree(d_topk);
            if (handle) ncclEpHandleDestroy(handle);
        }
    };

    void create_and_dispatch(Fwd& st, int which, int64_t (*route)(int), float tok_base) {
        st.which = which;
        st.tok_base = tok_base;

        int64_t h_topk[kNumTokens];
        for (int i = 0; i < kNumTokens; ++i) h_topk[i] = route(i);
        CUDA_ASSERT(cudaMalloc(&st.d_topk, kNumTokens * kTopK * sizeof(int64_t)));
        CUDA_ASSERT(cudaMemcpy(st.d_topk, h_topk, sizeof(h_topk), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&st.t_topk, 2, ncclInt64, st.d_topk, kNumTokens, kTopK));

        NCCL_ASSERT(ncclEpCreateHandle(
            &st.handle, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, st.t_topk, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        ASSERT_NE(st.handle, nullptr);

        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        std::vector<float> h_w(kNumTokens * kTopK);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = tok_base + static_cast<float>(i);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
            for (int k = 0; k < kTopK; ++k) h_w[i * kTopK + k] = weight_for(which, i, k);
        }

        CUDA_ASSERT(cudaMalloc(&st.d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&st.d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(st.d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&st.d_weights, kNumTokens * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&st.d_recv_w, kMaxRecvSlots * sizeof(float)));
        CUDA_ASSERT(cudaMemset(st.d_recv_w, 0, kMaxRecvSlots * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(st.d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(st.d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&st.t_tok, 2, ncclBfloat16, st.d_tok, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&st.t_recv, 2, ncclBfloat16, st.d_recv, kMaxRecvSlots, kHidden));
        NCCL_ASSERT(epTensorCreate(&st.t_w, 2, ncclFloat32, st.d_weights, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&st.t_recv_w, 1, ncclFloat32, st.d_recv_w, kMaxRecvSlots));

        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = st.t_tok;
        d_in.topk_weights = st.t_w;
        d_out.tokens = st.t_recv;
        d_out.topk_weights = st.t_recv_w;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        NCCL_ASSERT(ncclEpDispatch(st.handle, &d_in, &d_out, nullptr, &dcfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(st.handle, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // Forward combine (identity expert, weight 1.0): combined_x[i] must recover
    // the handle's original token value.
    void forward_combine_and_check(Fwd& st) {
        nv_bfloat16* d_out = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_out, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(d_out, 0, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        ncclEpTensor_t* t_out = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kNumTokens, kHidden));

        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = st.t_recv;
        c_out.tokens = t_out;
        NCCL_ASSERT(ncclEpCombine(st.handle, &c_in, &c_out, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = st.tok_base + static_cast<float>(i);
            EXPECT_NEAR(bf16_val(h_out[i * kHidden]), expected, 1e-3f)
                << "handle " << (st.which == 0 ? "A" : "B") << " rank " << g_rank << " token " << i
                << " FWD combine read stale routing from the other handle (expected " << expected << ", got "
                << bf16_val(h_out[i * kHidden]) << ")";
        }

        ncclEpTensorDestroy(t_out);
        cudaFree(d_out);
    }

    // Backward combine: input topk_weights = recv topk_weights (1D EM);
    // combined_topk_weights[i][k] must recover the handle's source weight.
    void backward_combine_and_check(Fwd& st) {
        nv_bfloat16* d_cx = nullptr;
        float* d_cw = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_cx, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_cw, kNumTokens * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMemset(d_cw, 0, kNumTokens * kTopK * sizeof(float)));
        ncclEpTensor_t *t_cx = nullptr, *t_cw = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_cx, 2, ncclBfloat16, d_cx, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_cw, 2, ncclFloat32, d_cw, kNumTokens, kTopK));

        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = st.t_recv;
        c_in.topk_weights = st.t_recv_w;
        c_out.tokens = t_cx;
        c_out.topk_weights = t_cw;
        ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
        ccfg.pass_direction = NCCL_EP_BWD_PASS;
        NCCL_ASSERT(ncclEpCombine(st.handle, &c_in, &c_out, &ccfg, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        const char* tag = st.which == 0 ? "A" : "B";

        // Reduced grad tokens must recover the handle's original token values.
        std::vector<nv_bfloat16> h_cx(kNumTokens * kHidden);
        CUDA_ASSERT(cudaMemcpy(h_cx.data(), d_cx, kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = st.tok_base + static_cast<float>(i);
            EXPECT_NEAR(bf16_val(h_cx[i * kHidden]), expected, 1e-3f)
                << "handle " << tag << " rank " << g_rank << " token " << i
                << " BWD combine tokens read stale routing from the other handle (expected " << expected << ", got "
                << bf16_val(h_cx[i * kHidden]) << ")";
        }

        // Reduced topk_weights (prob) must recover the handle's source weights.
        std::vector<float> h_cw(kNumTokens * kTopK);
        CUDA_ASSERT(cudaMemcpy(h_cw.data(), d_cw, kNumTokens * kTopK * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < kNumTokens; ++i)
            for (int k = 0; k < kTopK; ++k) {
                float expected = weight_for(st.which, i, k);
                EXPECT_NEAR(h_cw[i * kTopK + k], expected, 1e-4f)
                    << "handle " << tag << " rank " << g_rank << " token " << i << " k " << k
                    << " BWD combine prob read stale routing from the other handle (expected " << expected << ", got "
                    << h_cw[i * kTopK + k] << ")";
            }

        ncclEpTensorDestroy(t_cw);
        ncclEpTensorDestroy(t_cx);
        cudaFree(d_cw);
        cudaFree(d_cx);
    }
};

// Two live handles, two forward combines back to back. A (created first) is the
// victim: a shared per-rank routing buffer would hold B's routing by now.
TEST_F(HtCombinePpInterleaveTest, TwoForwardCombines) {
    Fwd a, b;
    create_and_dispatch(a, /*which=*/0, &route_a, /*tok_base=*/static_cast<float>(g_rank * kNumTokens + 1));
    create_and_dispatch(b, /*which=*/1, &route_b, /*tok_base=*/static_cast<float>(100 + g_rank * kNumTokens));

    forward_combine_and_check(a);
    forward_combine_and_check(b);

    b.free_all();
    a.free_all();
}

// 1F1B interleave: forward-combine the newer handle B, then backward-combine the
// older handle A. A's backward reduce must use A's own snapshotted routing, not
// B's, even though B updated the group after A.
TEST_F(HtCombinePpInterleaveTest, ForwardThenBackwardCombine) {
    Fwd a, b;
    create_and_dispatch(a, /*which=*/0, &route_a, /*tok_base=*/static_cast<float>(g_rank * kNumTokens + 1));
    create_and_dispatch(b, /*which=*/1, &route_b, /*tok_base=*/static_cast<float>(100 + g_rank * kNumTokens));

    forward_combine_and_check(b);
    backward_combine_and_check(a);

    b.free_all();
    a.free_all();
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_combine_pp_interleave_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
