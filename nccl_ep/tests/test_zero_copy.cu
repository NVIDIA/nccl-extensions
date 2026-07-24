/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * zero_copy config coverage. Each test creates its own ep_group with the
 * mode under test. HT covers BF16 round trips, all SCALES_FORWARD scale dtypes,
 * and packed-FP4 zero-copy dispatch;
 * LL checks staged and windowed token/scale payloads as exact bytes.
 */

#include "test_common.h"

static float bf16_val(nv_bfloat16 v) {
    return __bfloat162float(v);
}

namespace {

// Optional overrides on top of NCCL_EP_GROUP_CONFIG_INIT.
struct GroupOpts {
    ncclEpZeroCopyMode_t zero_copy = NCCL_EP_ZERO_COPY_AUTO;
    ncclEpAlgorithm_t algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
};

ncclEpGroupConfig_t base_group_cfg(const GroupOpts& opts) {
    ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
    cfg.algorithm = opts.algorithm;
    cfg.num_experts = kNumExperts;
    cfg.max_dispatch_tokens_per_rank = kNumTokens;
    cfg.max_token_bytes = kHidden * sizeof(nv_bfloat16);
    cfg.rdma_buffer_size = NCCL_EP_AUTO;
    cfg.num_qp_per_rank = NCCL_EP_AUTO;
    cfg.num_channels = NCCL_EP_AUTO;
    cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecvSlots);
    cfg.zero_copy = opts.zero_copy;
    return cfg;
}

// Register `data` as a symmetric window and attach it to `t`.
ncclResult_t attach_symmetric_window(ncclEpTensor_t* t, void* data, size_t bytes, ncclWindow_t* out_win) {
    ncclResult_t r = ncclCommWindowRegister(g_comm, data, bytes, out_win, NCCL_WIN_COLL_SYMMETRIC);
    if (r != ncclSuccess) return r;
    t->win_hdl = *out_win;
    t->win_offset = 0;
    return ncclSuccess;
}

struct ScalesForwardWireCase {
    const char* name;
    ncclDataType_t token_dtype;
    int token_elements;
    ncclDataType_t scale_dtype;
    int scale_elements;
};

constexpr int kFp4LogicalHidden = 512;
constexpr ScalesForwardWireCase kPackedFp4Wire{
    "packed-fp4-u8", ncclUint8, kFp4LogicalHidden / 2,
    ncclUint8, kFp4LogicalHidden / 16};
constexpr ScalesForwardWireCase kBf16Fp16Wire{
    "bf16-token-fp16-scale", ncclBfloat16, 8, ncclFloat16, 8};
constexpr ScalesForwardWireCase kFp8Fp32Wire{
    "fp8-e4m3-token-fp32-scale", ncclFloat8e4m3, 16, ncclFloat32, 4};

size_t wire_dtype_bytes(ncclDataType_t dtype) {
    switch (dtype) {
        case ncclFloat32:
            return 4;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclFloat8e4m3:
        case ncclFloat8e5m2:
        case ncclUint8:
            return 1;
        default:
            return 0;
    }
}

uint8_t ht_scales_forward_token_byte(int src_rank, int token, size_t byte) {
    if (byte == 0) return static_cast<uint8_t>(0x40 + src_rank);
    if (byte == 1) return static_cast<uint8_t>(1 + token);
    return static_cast<uint8_t>((src_rank * 53 + token * 11 + byte) & 0xff);
}

uint8_t ht_scales_forward_scale_byte(int src_rank, int token, size_t byte) {
    if (byte == 0) return static_cast<uint8_t>(0x80 + src_rank);
    if (byte == 1) return static_cast<uint8_t>(1 + token);
    return static_cast<uint8_t>((src_rank * 29 + token * 7 + byte) & 0xff);
}

class ZeroCopyTest : public ::testing::Test {
protected:
    int64_t* d_topk_ = nullptr;
    ncclEpTensor_t* topk_ = nullptr;

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk_, kNumTokens * kTopK * sizeof(int64_t)));
        int64_t h[kNumTokens];
        for (int i = 0; i < kNumTokens; ++i) h[i] = expert_for_token(i);
        CUDA_ASSERT(cudaMemcpy(d_topk_, h, sizeof(h), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_, 2, ncclInt64, d_topk_, kNumTokens, kTopK));
    }

    void TearDown() override {
        if (topk_) ncclEpTensorDestroy(topk_);
        if (d_topk_) cudaFree(d_topk_);
    }

    // Run one HT FLAT dispatch+combine round on `group`. `windowed_dispatch_out`
    // and `windowed_combine_in` independently force the corresponding tensor to
    // be backed by a freshly-registered symmetric NCCL window. Returns the
    // per-token first-hidden-element value (bf16-rounded) for verification, or
    // an empty vector if `expected_dispatch_err` / `expected_combine_err` were
    // hit (the assertion is reported via the standard gtest macros).
    std::vector<float> run_roundtrip(
        ncclEpGroup_t group,
        bool windowed_dispatch_out,
        bool windowed_combine_in,
        ncclResult_t expected_dispatch_err = ncclSuccess,
        ncclResult_t expected_combine_err = ncclSuccess) {
        // -- Allocate host-side input -----------------------------------------
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        // -- Allocate device buffers ------------------------------------------
        // recv buffer (dispatch output / combine input) and combined output use
        // ncclMemAlloc when window-registered so they get a symmetric-friendly
        // allocation; otherwise plain cudaMalloc.
        nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr, *d_out = nullptr;
        float *d_weights = nullptr, *d_recv_w = nullptr;
        int64_t* d_recv_idx = nullptr;
        const size_t recv_bytes = static_cast<size_t>(kMaxRecvSlots) * kHidden * sizeof(nv_bfloat16);

        EXPECT_EQ(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w, kMaxRecvSlots * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)), cudaSuccess);

        const bool need_window = windowed_dispatch_out || windowed_combine_in;
        if (need_window) {
            EXPECT_EQ(ncclMemAlloc(reinterpret_cast<void**>(&d_recv), recv_bytes), ncclSuccess);
            EXPECT_EQ(cudaMemset(d_recv, 0, recv_bytes), cudaSuccess);
        } else {
            EXPECT_EQ(cudaMalloc(&d_recv, recv_bytes), cudaSuccess);
            EXPECT_EQ(cudaMemset(d_recv, 0, recv_bytes), cudaSuccess);
        }

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(
            cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);
        EXPECT_EQ(
            cudaMemcpy(d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

        // -- Build tensor descriptors ----------------------------------------
        ncclEpTensor_t *t_tok, *t_recv, *t_out, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(
            epTensorCreate(
                &t_recv,
                2,
                ncclBfloat16,
                windowed_dispatch_out || windowed_combine_in ? nullptr : d_recv,
                kMaxRecvSlots,
                kHidden),
            ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w, 2, ncclFloat32, d_weights, kNumTokens, kTopK), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_w, 2, ncclFloat32, d_recv_w, kMaxRecvSlots, kTopK), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64, d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);

        // Window-register the recv buffer; the same window covers dispatch
        // output and combine input (caller picks which path uses it).
        ncclWindow_t recv_win{};
        if (need_window) {
            EXPECT_EQ(attach_symmetric_window(t_recv, d_recv, recv_bytes, &recv_win), ncclSuccess);
        }

        // -- Create handle (collective; ep_group lifetime is per-test) -------
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, group, NCCL_EP_LAYOUT_FLAT, topk_, nullptr, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        // -- Dispatch --------------------------------------------------------
        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = t_tok;
        d_in.topk_weights = t_w;
        d_out_s.tokens = t_recv;
        d_out_s.topk_weights = t_recv_w;
        d_out_s.topk_idx = t_recv_idx;

        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        ncclResult_t disp_r = ncclEpDispatch(h, &d_in, &d_out_s, nullptr, &dcfg, g_stream);
        EXPECT_EQ(disp_r, expected_dispatch_err);

        std::vector<float> vals;
        if (disp_r == ncclSuccess && expected_combine_err != ncclInvalidArgument) {
            EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        }

        // -- Combine (only if dispatch succeeded; we keep combine input fresh
        //    from the dispatch output buffer, so the windowed-ness is identical
        //    to windowed_dispatch_out -- windowed_combine_in toggles whether we
        //    *pass* the windowed descriptor to combine).
        if (disp_r == ncclSuccess) {
            ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
            ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;

            // For ZeroCopyRejectsNonWindowCombine we need a non-window descriptor
            // for the combine input even though dispatch wrote into a windowed
            // buffer. Build a sibling descriptor that points at the same data
            // via .data instead of .win_hdl.
            ncclEpTensor_t* t_combine_in = t_recv;
            ncclEpTensor_t* t_combine_in_alias = nullptr;
            if (need_window && !windowed_combine_in) {
                EXPECT_EQ(
                    epTensorCreate(&t_combine_in_alias, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kHidden),
                    ncclSuccess);
                t_combine_in = t_combine_in_alias;
            }

            c_in.tokens = t_combine_in;
            c_out_s.tokens = t_out;
            ncclResult_t comb_r = ncclEpCombine(h, &c_in, &c_out_s, nullptr, g_stream);
            EXPECT_EQ(comb_r, expected_combine_err);

            if (comb_r == ncclSuccess) {
                EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
                std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
                EXPECT_EQ(
                    cudaMemcpy(h_out.data(), d_out, kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost),
                    cudaSuccess);
                vals.resize(kNumTokens);
                for (int i = 0; i < kNumTokens; ++i) vals[i] = bf16_val(h_out[i * kHidden]);
            }
            if (t_combine_in_alias) ncclEpTensorDestroy(t_combine_in_alias);
        }

        // -- Cleanup ---------------------------------------------------------
        EXPECT_EQ(ncclEpHandleDestroy(h), ncclSuccess);
        if (need_window) {
            EXPECT_EQ(ncclCommWindowDeregister(g_comm, recv_win), ncclSuccess);
        }
        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_out);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok);
        cudaFree(d_out);
        cudaFree(d_weights);
        cudaFree(d_recv_w);
        cudaFree(d_recv_idx);
        if (need_window) ncclMemFree(d_recv);
        else cudaFree(d_recv);
        return vals;
    }

    void run_ht_scales_forward_flat_dispatch(
        const ScalesForwardWireCase& wire,
        bool windowed_tokens,
        bool windowed_scales,
        size_t token_window_offset,
        size_t scale_window_offset,
        ncclEpZeroCopyMode_t zero_copy_mode,
        ncclResult_t expected_dispatch_result) {
        const int num_local_experts = kNumExperts / g_nranks;
        const size_t token_bytes_per_row =
            static_cast<size_t>(wire.token_elements) * wire_dtype_bytes(wire.token_dtype);
        const size_t scale_bytes_per_row =
            static_cast<size_t>(wire.scale_elements) * wire_dtype_bytes(wire.scale_dtype);
        const size_t recv_token_bytes = static_cast<size_t>(kMaxRecvSlots) * token_bytes_per_row;
        const size_t recv_scale_bytes = static_cast<size_t>(kMaxRecvSlots) * scale_bytes_per_row;
        const size_t recv_token_allocation_bytes = recv_token_bytes + (token_window_offset ? 16 : 0);
        const size_t recv_scale_allocation_bytes = recv_scale_bytes + (scale_window_offset ? 16 : 0);
        ASSERT_EQ(kNumExperts % g_nranks, 0);

        ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
        group_config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        group_config.num_experts = kNumExperts;
        group_config.max_dispatch_tokens_per_rank = kNumTokens;
        group_config.max_recv_tokens_per_rank = kMaxRecvSlots;
        group_config.max_token_bytes =
            static_cast<unsigned int>((token_bytes_per_row + 15) & ~size_t{15});
        group_config.max_scale_bytes = static_cast<unsigned int>(scale_bytes_per_row);
        group_config.rdma_buffer_size = NCCL_EP_AUTO;
        group_config.num_qp_per_rank = NCCL_EP_AUTO;
        group_config.num_channels = NCCL_EP_AUTO;
        group_config.zero_copy = zero_copy_mode;

        ncclEpGroup_t group = nullptr;
        NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

        std::vector<uint8_t> host_tokens(kNumTokens * token_bytes_per_row);
        std::vector<uint8_t> host_scales(kNumTokens * scale_bytes_per_row);
        std::vector<float> host_weights(kNumTokens * kTopK, 1.0f);
        for (int token = 0; token < kNumTokens; ++token) {
            for (size_t byte = 0; byte < token_bytes_per_row; ++byte) {
                host_tokens[token * token_bytes_per_row + byte] =
                    ht_scales_forward_token_byte(g_rank, token, byte);
            }
            for (size_t byte = 0; byte < scale_bytes_per_row; ++byte) {
                host_scales[token * scale_bytes_per_row + byte] =
                    ht_scales_forward_scale_byte(g_rank, token, byte);
            }
        }

        uint8_t *device_tokens = nullptr, *device_scales = nullptr;
        uint8_t *device_recv_tokens = nullptr, *device_recv_scales = nullptr;
        float *device_weights = nullptr, *device_recv_weights = nullptr;
        int64_t* device_recv_topk_idx = nullptr;
        CUDA_ASSERT(cudaMalloc(&device_tokens, host_tokens.size()));
        CUDA_ASSERT(cudaMalloc(&device_scales, host_scales.size()));
        CUDA_ASSERT(cudaMalloc(&device_weights, host_weights.size() * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&device_recv_weights, kMaxRecvSlots * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&device_recv_topk_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)));
        if (windowed_tokens) {
            NCCL_ASSERT(ncclMemAlloc(
                reinterpret_cast<void**>(&device_recv_tokens), recv_token_allocation_bytes));
        } else {
            CUDA_ASSERT(cudaMalloc(&device_recv_tokens, recv_token_bytes));
        }
        if (windowed_scales) {
            NCCL_ASSERT(ncclMemAlloc(
                reinterpret_cast<void**>(&device_recv_scales), recv_scale_allocation_bytes));
        } else {
            CUDA_ASSERT(cudaMalloc(&device_recv_scales, recv_scale_bytes));
        }

        CUDA_ASSERT(cudaMemcpy(
            device_tokens, host_tokens.data(), host_tokens.size(), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(
            device_scales, host_scales.data(), host_scales.size(), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(
            device_weights,
            host_weights.data(),
            host_weights.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        uint8_t* recv_token_data = device_recv_tokens + (windowed_tokens ? token_window_offset : 0);
        uint8_t* recv_scale_data = device_recv_scales + (windowed_scales ? scale_window_offset : 0);
        CUDA_ASSERT(cudaMemset(recv_token_data, 0xa5, recv_token_bytes));
        CUDA_ASSERT(cudaMemset(recv_scale_data, 0x5a, recv_scale_bytes));
        CUDA_ASSERT(cudaMemset(device_recv_weights, 0, kMaxRecvSlots * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMemset(device_recv_topk_idx, 0xff, kMaxRecvSlots * kTopK * sizeof(int64_t)));

        ncclEpTensor_t *tokens = nullptr, *scales = nullptr, *weights = nullptr;
        ncclEpTensor_t *recv_tokens = nullptr, *recv_scales = nullptr;
        ncclEpTensor_t *recv_weights = nullptr, *recv_topk_idx = nullptr;
        NCCL_ASSERT(epTensorCreate(
            &tokens,
            2,
            wire.token_dtype,
            device_tokens,
            kNumTokens,
            wire.token_elements));
        NCCL_ASSERT(epTensorCreate(
            &scales,
            2,
            wire.scale_dtype,
            device_scales,
            kNumTokens,
            wire.scale_elements));
        NCCL_ASSERT(epTensorCreate(&weights, 2, ncclFloat32, device_weights, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(
            &recv_tokens,
            2,
            wire.token_dtype,
            windowed_tokens ? nullptr : device_recv_tokens,
            kMaxRecvSlots,
            wire.token_elements));
        NCCL_ASSERT(epTensorCreate(
            &recv_scales,
            2,
            wire.scale_dtype,
            windowed_scales ? nullptr : device_recv_scales,
            kMaxRecvSlots,
            wire.scale_elements));
        NCCL_ASSERT(epTensorCreate(
            &recv_weights, 2, ncclFloat32, device_recv_weights, kMaxRecvSlots, kTopK));
        NCCL_ASSERT(epTensorCreate(
            &recv_topk_idx, 2, ncclInt64, device_recv_topk_idx, kMaxRecvSlots, kTopK));

        ncclWindow_t recv_tokens_window{};
        ncclWindow_t recv_scales_window{};
        if (windowed_tokens) {
            NCCL_ASSERT(attach_symmetric_window(
                recv_tokens,
                device_recv_tokens,
                recv_token_allocation_bytes,
                &recv_tokens_window));
            recv_tokens->win_offset = token_window_offset;
        }
        if (windowed_scales) {
            NCCL_ASSERT(attach_symmetric_window(
                recv_scales,
                device_recv_scales,
                recv_scale_allocation_bytes,
                &recv_scales_window));
            recv_scales->win_offset = scale_window_offset;
        }

        ncclEpHandle_t handle = nullptr;
        NCCL_ASSERT(ncclEpCreateHandle(
            &handle, group, NCCL_EP_LAYOUT_FLAT, topk_, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
        inputs.tokens = tokens;
        inputs.scales = scales;
        inputs.topk_weights = weights;
        outputs.tokens = recv_tokens;
        outputs.scales = recv_scales;
        outputs.topk_weights = recv_weights;
        outputs.topk_idx = recv_topk_idx;
        config.quantization_recipe = NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD;

        const ncclResult_t dispatch_result =
            ncclEpDispatch(handle, &inputs, &outputs, nullptr, &config, g_stream);
        EXPECT_EQ(dispatch_result, expected_dispatch_result);
        if (dispatch_result == ncclSuccess) {
            EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
            EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

            std::vector<uint8_t> received_tokens(recv_token_bytes);
            std::vector<uint8_t> received_scales(recv_scale_bytes);
            std::vector<float> received_weights(kMaxRecvSlots * kTopK);
            std::vector<int64_t> received_topk_idx(kMaxRecvSlots * kTopK);
            CUDA_ASSERT(cudaMemcpy(
                received_tokens.data(),
                recv_token_data,
                recv_token_bytes,
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                received_scales.data(),
                recv_scale_data,
                recv_scale_bytes,
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                received_weights.data(),
                device_recv_weights,
                received_weights.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                received_topk_idx.data(),
                device_recv_topk_idx,
                received_topk_idx.size() * sizeof(int64_t),
                cudaMemcpyDeviceToHost));

            std::vector<bool> seen(static_cast<size_t>(g_nranks) * kNumTokens, false);
            int received_count = 0;
            for (int row = 0; row < kMaxRecvSlots; ++row) {
                if (received_topk_idx[row] < 0) continue;
                ++received_count;
                const size_t token_offset = static_cast<size_t>(row) * token_bytes_per_row;
                const size_t scale_offset = static_cast<size_t>(row) * scale_bytes_per_row;
                const int src_rank = static_cast<int>(received_tokens[token_offset]) - 0x40;
                const int token = static_cast<int>(received_tokens[token_offset + 1]) - 1;
                EXPECT_GE(src_rank, 0);
                EXPECT_LT(src_rank, g_nranks);
                EXPECT_GE(token, 0);
                EXPECT_LT(token, kNumTokens);
                if (src_rank < 0 || src_rank >= g_nranks || token < 0 || token >= kNumTokens) continue;

                const size_t pattern_index = static_cast<size_t>(src_rank) * kNumTokens + token;
                EXPECT_FALSE(seen[pattern_index]);
                seen[pattern_index] = true;
                const int expert = (src_rank * kNumTokens + token) % kNumExperts;
                EXPECT_EQ(expert / num_local_experts, g_rank);
                EXPECT_EQ(received_topk_idx[row], expert % num_local_experts);
                EXPECT_FLOAT_EQ(received_weights[row], 1.0f);
                for (size_t byte = 0; byte < token_bytes_per_row; ++byte) {
                    EXPECT_EQ(
                        received_tokens[token_offset + byte],
                        ht_scales_forward_token_byte(src_rank, token, byte));
                }
                for (size_t byte = 0; byte < scale_bytes_per_row; ++byte) {
                    EXPECT_EQ(
                        received_scales[scale_offset + byte],
                        ht_scales_forward_scale_byte(src_rank, token, byte));
                }
            }

            int expected_count = 0;
            for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
                for (int token = 0; token < kNumTokens; ++token) {
                    const int expert = (src_rank * kNumTokens + token) % kNumExperts;
                    const bool expected = expert / num_local_experts == g_rank;
                    EXPECT_EQ(seen[static_cast<size_t>(src_rank) * kNumTokens + token], expected);
                    expected_count += expected;
                }
            }
            EXPECT_EQ(received_count, expected_count);
        }

        NCCL_ASSERT(ncclEpHandleDestroy(handle));
        if (windowed_scales) NCCL_ASSERT(ncclCommWindowDeregister(g_comm, recv_scales_window));
        if (windowed_tokens) NCCL_ASSERT(ncclCommWindowDeregister(g_comm, recv_tokens_window));
        ncclEpTensorDestroy(tokens);
        ncclEpTensorDestroy(scales);
        ncclEpTensorDestroy(weights);
        ncclEpTensorDestroy(recv_tokens);
        ncclEpTensorDestroy(recv_scales);
        ncclEpTensorDestroy(recv_weights);
        ncclEpTensorDestroy(recv_topk_idx);
        cudaFree(device_tokens);
        cudaFree(device_scales);
        cudaFree(device_weights);
        cudaFree(device_recv_weights);
        cudaFree(device_recv_topk_idx);
        if (windowed_tokens) NCCL_ASSERT(ncclMemFree(device_recv_tokens));
        else cudaFree(device_recv_tokens);
        if (windowed_scales) NCCL_ASSERT(ncclMemFree(device_recv_scales));
        else cudaFree(device_recv_scales);
        NCCL_ASSERT(ncclEpGroupDestroy(group));
    }

    void run_scales_forward_rank_major_dispatch(
        const ScalesForwardWireCase& wire,
        bool windowed_tokens,
        bool windowed_scales,
        ncclResult_t expected_dispatch_result,
        ncclEpZeroCopyMode_t zero_copy_mode) {
        constexpr uint8_t kUntouchedTokenByte = 0xa5;
        constexpr uint8_t kUntouchedScaleByte = 0x5a;
        const int num_local_experts = kNumExperts / g_nranks;
        const int recv_rows = g_nranks * kNumTokens;
        const size_t token_element_bytes = wire_dtype_bytes(wire.token_dtype);
        const size_t scale_element_bytes = wire_dtype_bytes(wire.scale_dtype);
        ASSERT_NE(token_element_bytes, 0u);
        ASSERT_NE(scale_element_bytes, 0u);
        const size_t token_bytes_per_row =
            static_cast<size_t>(wire.token_elements) * token_element_bytes;
        const size_t scale_bytes_per_row =
            static_cast<size_t>(wire.scale_elements) * scale_element_bytes;
        ASSERT_EQ(token_bytes_per_row % 16, 0u);
        ASSERT_EQ(scale_bytes_per_row % 16, 0u);
        const size_t recv_token_bytes = static_cast<size_t>(recv_rows) * token_bytes_per_row;
        const size_t recv_scale_bytes = static_cast<size_t>(recv_rows) * scale_bytes_per_row;

        ASSERT_EQ(kNumExperts % g_nranks, 0);

        ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
        group_config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
        group_config.num_experts = kNumExperts;
        group_config.max_dispatch_tokens_per_rank = kNumTokens;
        group_config.max_recv_tokens_per_rank = recv_rows;
        group_config.max_token_bytes =
            static_cast<unsigned int>(token_bytes_per_row + scale_bytes_per_row);
        group_config.max_scale_bytes = static_cast<unsigned int>(scale_bytes_per_row);
        group_config.rdma_buffer_size = NCCL_EP_AUTO;
        group_config.num_qp_per_rank = num_local_experts;
        group_config.num_channels = NCCL_EP_AUTO;
        group_config.zero_copy = zero_copy_mode;

        ncclEpGroup_t group = nullptr;
        NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));

        std::vector<uint8_t> host_tokens(kNumTokens * token_bytes_per_row);
        std::vector<uint8_t> host_scales(kNumTokens * scale_bytes_per_row);
        std::vector<float> host_weights(kNumTokens * kTopK, 1.0f);
        for (int token = 0; token < kNumTokens; ++token) {
            for (size_t byte = 0; byte < token_bytes_per_row; ++byte) {
                host_tokens[token * token_bytes_per_row + byte] =
                    static_cast<uint8_t>((g_rank * 53 + token * 11 + byte) & 0xff);
            }
            for (size_t byte = 0; byte < scale_bytes_per_row; ++byte) {
                host_scales[token * scale_bytes_per_row + byte] =
                    static_cast<uint8_t>((g_rank * 29 + token * 7 + byte) & 0xff);
            }
        }

        uint8_t *device_tokens = nullptr, *device_scales = nullptr;
        uint8_t *device_recv_tokens = nullptr, *device_recv_scales = nullptr;
        float *device_weights = nullptr, *device_recv_weights = nullptr;
        int *device_recv_topk_idx = nullptr, *device_src_rank_counters = nullptr;
        CUDA_ASSERT(cudaMalloc(&device_tokens, host_tokens.size()));
        CUDA_ASSERT(cudaMalloc(&device_scales, host_scales.size()));
        CUDA_ASSERT(cudaMalloc(&device_weights, host_weights.size() * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&device_recv_weights, recv_rows * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&device_recv_topk_idx, recv_rows * kTopK * sizeof(int)));
        CUDA_ASSERT(cudaMalloc(&device_src_rank_counters, g_nranks * sizeof(int)));
        if (windowed_tokens) {
            NCCL_ASSERT(ncclMemAlloc(reinterpret_cast<void**>(&device_recv_tokens), recv_token_bytes));
        } else {
            CUDA_ASSERT(cudaMalloc(&device_recv_tokens, recv_token_bytes));
        }
        if (windowed_scales) {
            NCCL_ASSERT(ncclMemAlloc(reinterpret_cast<void**>(&device_recv_scales), recv_scale_bytes));
        } else {
            CUDA_ASSERT(cudaMalloc(&device_recv_scales, recv_scale_bytes));
        }

        CUDA_ASSERT(cudaMemcpy(
            device_tokens, host_tokens.data(), host_tokens.size(), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(
            device_scales, host_scales.data(), host_scales.size(), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(
            device_weights,
            host_weights.data(),
            host_weights.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemset(device_recv_tokens, kUntouchedTokenByte, recv_token_bytes));
        CUDA_ASSERT(cudaMemset(device_recv_scales, kUntouchedScaleByte, recv_scale_bytes));
        CUDA_ASSERT(cudaMemset(device_recv_weights, 0, recv_rows * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMemset(device_recv_topk_idx, 0xff, recv_rows * kTopK * sizeof(int)));
        CUDA_ASSERT(cudaMemset(device_src_rank_counters, 0, g_nranks * sizeof(int)));

        ncclEpTensor_t *tokens = nullptr, *scales = nullptr, *weights = nullptr;
        ncclEpTensor_t *recv_tokens = nullptr, *recv_scales = nullptr;
        ncclEpTensor_t *recv_weights = nullptr, *recv_topk_idx = nullptr;
        ncclEpTensor_t* src_rank_counters = nullptr;
        NCCL_ASSERT(epTensorCreate(
            &tokens, 2, wire.token_dtype, device_tokens, kNumTokens, wire.token_elements));
        NCCL_ASSERT(epTensorCreate(
            &scales, 2, wire.scale_dtype, device_scales, kNumTokens, wire.scale_elements));
        NCCL_ASSERT(epTensorCreate(&weights, 2, ncclFloat32, device_weights, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(
            &recv_tokens,
            3,
            wire.token_dtype,
            windowed_tokens ? nullptr : device_recv_tokens,
            g_nranks,
            kNumTokens,
            wire.token_elements));
        NCCL_ASSERT(epTensorCreate(
            &recv_scales,
            3,
            wire.scale_dtype,
            windowed_scales ? nullptr : device_recv_scales,
            g_nranks,
            kNumTokens,
            wire.scale_elements));
        NCCL_ASSERT(epTensorCreate(
            &recv_weights, 3, ncclFloat32, device_recv_weights, g_nranks, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(
            &recv_topk_idx, 3, ncclInt32, device_recv_topk_idx, g_nranks, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(
            &src_rank_counters, 1, ncclInt32, device_src_rank_counters, g_nranks));

        ncclWindow_t recv_tokens_window{};
        ncclWindow_t recv_scales_window{};
        if (windowed_tokens) {
            NCCL_ASSERT(attach_symmetric_window(
                recv_tokens, device_recv_tokens, recv_token_bytes, &recv_tokens_window));
        }
        if (windowed_scales) {
            NCCL_ASSERT(attach_symmetric_window(
                recv_scales, device_recv_scales, recv_scale_bytes, &recv_scales_window));
        }

        ncclEpHandle_t handle = nullptr;
        NCCL_ASSERT(ncclEpCreateHandle(
            &handle, group, NCCL_EP_LAYOUT_RANK_MAJOR, topk_, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
        ncclEpDispatchConfig_t dispatch_config = NCCL_EP_DISPATCH_CONFIG_INIT;
        inputs.tokens = tokens;
        inputs.scales = scales;
        inputs.topk_weights = weights;
        outputs.tokens = recv_tokens;
        outputs.scales = recv_scales;
        outputs.topk_weights = recv_weights;
        outputs.topk_idx = recv_topk_idx;
        layout_info.src_rank_counters = src_rank_counters;
        dispatch_config.quantization_recipe = NCCL_EP_DISPATCH_QUANT_SCALES_FORWARD;

        const ncclResult_t dispatch_result =
            ncclEpDispatch(handle, &inputs, &outputs, &layout_info, &dispatch_config, g_stream);
        EXPECT_EQ(dispatch_result, expected_dispatch_result);
        if (dispatch_result == ncclSuccess) {
            EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
            EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

            std::vector<uint8_t> received_tokens(recv_token_bytes);
            std::vector<uint8_t> received_scales(recv_scale_bytes);
            std::vector<float> received_weights(recv_rows * kTopK);
            std::vector<int> received_topk_idx(recv_rows * kTopK);
            std::vector<int> src_rank_counts(g_nranks);
            CUDA_ASSERT(cudaMemcpy(
                received_tokens.data(),
                device_recv_tokens,
                recv_token_bytes,
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                received_scales.data(),
                device_recv_scales,
                recv_scale_bytes,
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                received_weights.data(),
                device_recv_weights,
                received_weights.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                received_topk_idx.data(),
                device_recv_topk_idx,
                received_topk_idx.size() * sizeof(int),
                cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(
                src_rank_counts.data(),
                device_src_rank_counters,
                src_rank_counts.size() * sizeof(int),
                cudaMemcpyDeviceToHost));

            for (int src_rank = 0; src_rank < g_nranks; ++src_rank) {
                std::vector<bool> seen(kNumTokens, false);
                int expected_count = 0;
                for (int token = 0; token < kNumTokens; ++token) {
                    const int expert = (src_rank * kNumTokens + token) % kNumExperts;
                    if (expert / num_local_experts == g_rank) ++expected_count;
                }
                EXPECT_EQ(src_rank_counts[src_rank], expected_count);

                for (int slot = 0; slot < expected_count; ++slot) {
                    const size_t row = static_cast<size_t>(src_rank) * kNumTokens + slot;
                    const size_t token_offset = row * token_bytes_per_row;
                    const size_t scale_offset = row * scale_bytes_per_row;
                    bool matched = false;
                    for (int token = 0; token < kNumTokens && !matched; ++token) {
                        const int expert = (src_rank * kNumTokens + token) % kNumExperts;
                        if (expert / num_local_experts != g_rank || seen[token]) continue;
                        bool token_matches = true;
                        for (size_t byte = 0; byte < token_bytes_per_row; ++byte) {
                            const uint8_t expected =
                                static_cast<uint8_t>((src_rank * 53 + token * 11 + byte) & 0xff);
                            if (received_tokens[token_offset + byte] != expected) {
                                token_matches = false;
                                break;
                            }
                        }
                        if (!token_matches) continue;

                        for (size_t byte = 0; byte < scale_bytes_per_row; ++byte) {
                            const uint8_t expected =
                                static_cast<uint8_t>((src_rank * 29 + token * 7 + byte) & 0xff);
                            EXPECT_EQ(received_scales[scale_offset + byte], expected);
                        }
                        EXPECT_EQ(received_topk_idx[row], expert % num_local_experts);
                        EXPECT_FLOAT_EQ(received_weights[row], 1.0f);
                        seen[token] = true;
                        matched = true;
                    }
                    EXPECT_TRUE(matched) << "rank " << g_rank << " source " << src_rank
                                         << " slot " << slot;
                }

                for (int slot = expected_count; slot < kNumTokens; ++slot) {
                    const size_t row = static_cast<size_t>(src_rank) * kNumTokens + slot;
                    EXPECT_EQ(received_topk_idx[row], -1);
                    for (size_t byte = 0; byte < token_bytes_per_row; ++byte) {
                        EXPECT_EQ(received_tokens[row * token_bytes_per_row + byte], kUntouchedTokenByte);
                    }
                    for (size_t byte = 0; byte < scale_bytes_per_row; ++byte) {
                        EXPECT_EQ(received_scales[row * scale_bytes_per_row + byte], kUntouchedScaleByte);
                    }
                }
            }
        }

        NCCL_ASSERT(ncclEpHandleDestroy(handle));
        if (windowed_scales) NCCL_ASSERT(ncclCommWindowDeregister(g_comm, recv_scales_window));
        if (windowed_tokens) NCCL_ASSERT(ncclCommWindowDeregister(g_comm, recv_tokens_window));
        ncclEpTensorDestroy(tokens);
        ncclEpTensorDestroy(scales);
        ncclEpTensorDestroy(weights);
        ncclEpTensorDestroy(recv_tokens);
        ncclEpTensorDestroy(recv_scales);
        ncclEpTensorDestroy(recv_weights);
        ncclEpTensorDestroy(recv_topk_idx);
        ncclEpTensorDestroy(src_rank_counters);
        cudaFree(device_tokens);
        cudaFree(device_scales);
        cudaFree(device_weights);
        cudaFree(device_recv_weights);
        cudaFree(device_recv_topk_idx);
        cudaFree(device_src_rank_counters);
        if (windowed_tokens) NCCL_ASSERT(ncclMemFree(device_recv_tokens));
        else cudaFree(device_recv_tokens);
        if (windowed_scales) NCCL_ASSERT(ncclMemFree(device_recv_scales));
        else cudaFree(device_recv_scales);
        NCCL_ASSERT(ncclEpGroupDestroy(group));
    }

    static void expect_identity_roundtrip(const std::vector<float>& vals) {
        ASSERT_EQ(vals.size(), static_cast<size_t>(kNumTokens));
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
            EXPECT_NEAR(vals[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
        }
    }
};

// -- Tests ---------------------------------------------------------------------

// zero_copy = AUTO (zero-init default) -> library-owned staging.
TEST_F(ZeroCopyTest, DefaultStaging) {
    GroupOpts opts;
    ncclEpGroupConfig_t cfg = base_group_cfg(opts);
    ncclEpGroup_t g = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&g, g_comm, &cfg));

    auto vals = run_roundtrip(g, /*windowed_dispatch_out=*/false, /*windowed_combine_in=*/false);
    expect_identity_roundtrip(vals);

    NCCL_ASSERT(ncclEpGroupDestroy(g));
}

// zero_copy = ON -- dispatch output and combine input are user-registered windows.
TEST_F(ZeroCopyTest, ZeroCopyWindowedRoundtrip) {
    GroupOpts opts;
    opts.zero_copy = NCCL_EP_ZERO_COPY_ON;
    ncclEpGroupConfig_t cfg = base_group_cfg(opts);
    ncclEpGroup_t g = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&g, g_comm, &cfg));

    auto vals = run_roundtrip(g, /*windowed_dispatch_out=*/true, /*windowed_combine_in=*/true);
    expect_identity_roundtrip(vals);

    NCCL_ASSERT(ncclEpGroupDestroy(g));
}

TEST_F(ZeroCopyTest, HtScalesForwardFlatZeroCopyPreservesPackedBytesAndScales) {
    run_ht_scales_forward_flat_dispatch(
        kPackedFp4Wire,
        /*windowed_tokens=*/true,
        /*windowed_scales=*/true,
        /*token_window_offset=*/0,
        /*scale_window_offset=*/0,
        NCCL_EP_ZERO_COPY_ON,
        /*expected_dispatch_result=*/ncclSuccess);
}

TEST_F(ZeroCopyTest, HtScalesForwardFlatStagedPreservesAllScaleDtypes) {
    constexpr ScalesForwardWireCase cases[] = {
        {"ht-u8-scale", ncclFloat8e4m3, 16, ncclUint8, 16},
        {"ht-fp8-e4m3-scale", ncclFloat8e4m3, 16, ncclFloat8e4m3, 16},
        {"ht-fp8-e5m2-scale", ncclFloat8e4m3, 16, ncclFloat8e5m2, 16},
        {"ht-fp16-scale", ncclFloat8e4m3, 16, ncclFloat16, 8},
        {"ht-bf16-scale", ncclFloat8e4m3, 16, ncclBfloat16, 8},
        {"ht-fp32-scale", ncclFloat8e4m3, 16, ncclFloat32, 4},
    };
    for (const auto& wire : cases) {
        SCOPED_TRACE(wire.name);
        run_ht_scales_forward_flat_dispatch(
            wire,
            /*windowed_tokens=*/false,
            /*windowed_scales=*/false,
            /*token_window_offset=*/0,
            /*scale_window_offset=*/0,
            NCCL_EP_ZERO_COPY_AUTO,
            /*expected_dispatch_result=*/ncclSuccess);
    }
}

TEST_F(ZeroCopyTest, HtScalesForwardFlatRejectsHalfOutputWindows) {
    struct HalfWindowCase {
        const char* name;
        bool windowed_tokens;
        bool windowed_scales;
        ncclEpZeroCopyMode_t zero_copy_mode;
    };
    constexpr HalfWindowCase cases[] = {
        {"on-token-only", true, false, NCCL_EP_ZERO_COPY_ON},
        {"on-scale-only", false, true, NCCL_EP_ZERO_COPY_ON},
        {"auto-token-only", true, false, NCCL_EP_ZERO_COPY_AUTO},
        {"auto-scale-only", false, true, NCCL_EP_ZERO_COPY_AUTO},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_ht_scales_forward_flat_dispatch(
            kPackedFp4Wire,
            test_case.windowed_tokens,
            test_case.windowed_scales,
            /*token_window_offset=*/0,
            /*scale_window_offset=*/0,
            test_case.zero_copy_mode,
            /*expected_dispatch_result=*/ncclInvalidArgument);
    }
}

TEST_F(ZeroCopyTest, HtScalesForwardFlatRejectsMisalignedWindowOffsets) {
    constexpr size_t offsets[][2] = {{1, 0}, {0, 1}};
    for (const auto& offset : offsets) {
        SCOPED_TRACE(::testing::Message() << "token_offset=" << offset[0]
                                          << " scale_offset=" << offset[1]);
        run_ht_scales_forward_flat_dispatch(
            kPackedFp4Wire,
            /*windowed_tokens=*/true,
            /*windowed_scales=*/true,
            offset[0],
            offset[1],
            NCCL_EP_ZERO_COPY_ON,
            /*expected_dispatch_result=*/ncclInvalidArgument);
    }
}

TEST_F(ZeroCopyTest, ScalesForwardPackedFp4RankMajorWindowedPreservesBytesAndScales) {
    run_scales_forward_rank_major_dispatch(
        kPackedFp4Wire,
        /*windowed_tokens=*/true,
        /*windowed_scales=*/true,
        /*expected_dispatch_result=*/ncclSuccess,
        NCCL_EP_ZERO_COPY_ON);
}

TEST_F(ZeroCopyTest, ScalesForwardRankMajorWindowedBf16Fp16PreservesBytes) {
    run_scales_forward_rank_major_dispatch(
        kBf16Fp16Wire,
        /*windowed_tokens=*/true,
        /*windowed_scales=*/true,
        /*expected_dispatch_result=*/ncclSuccess,
        NCCL_EP_ZERO_COPY_ON);
}

TEST_F(ZeroCopyTest, ScalesForwardRankMajorWindowedFp8Fp32PreservesBytes) {
    run_scales_forward_rank_major_dispatch(
        kFp8Fp32Wire,
        /*windowed_tokens=*/true,
        /*windowed_scales=*/true,
        /*expected_dispatch_result=*/ncclSuccess,
        NCCL_EP_ZERO_COPY_ON);
}

TEST_F(ZeroCopyTest, ScalesForwardRankMajorStagedPreservesAllWireDtypes) {
    constexpr ScalesForwardWireCase cases[] = {
        kPackedFp4Wire,
        {"fp8-e4m3", ncclFloat8e4m3, 16, ncclFloat8e4m3, 16},
        {"fp8-e4m3-u8-scale", ncclFloat8e4m3, 16, ncclUint8, 16},
        {"fp8-e5m2", ncclFloat8e5m2, 16, ncclFloat8e5m2, 16},
        {"fp16", ncclFloat16, 8, ncclFloat16, 8},
        {"bf16", ncclBfloat16, 8, ncclBfloat16, 8},
        {"fp32", ncclFloat32, 4, ncclFloat32, 4},
        kFp8Fp32Wire,
        kBf16Fp16Wire,
    };
    for (const auto& wire : cases) {
        SCOPED_TRACE(wire.name);
        run_scales_forward_rank_major_dispatch(
            wire,
            /*windowed_tokens=*/false,
            /*windowed_scales=*/false,
            /*expected_dispatch_result=*/ncclSuccess,
            NCCL_EP_ZERO_COPY_AUTO);
    }
}

TEST_F(ZeroCopyTest, ScalesForwardPackedFp4RankMajorTokenWindowOnlyPreservesBytesAndScales) {
    run_scales_forward_rank_major_dispatch(
        kPackedFp4Wire,
        /*windowed_tokens=*/true,
        /*windowed_scales=*/false,
        /*expected_dispatch_result=*/ncclSuccess,
        NCCL_EP_ZERO_COPY_ON);
}

TEST_F(ZeroCopyTest, ScalesForwardPackedFp4RankMajorScaleWindowOnlyPreservesBytesAndScales) {
    run_scales_forward_rank_major_dispatch(
        kPackedFp4Wire,
        /*windowed_tokens=*/false,
        /*windowed_scales=*/true,
        /*expected_dispatch_result=*/ncclSuccess,
        NCCL_EP_ZERO_COPY_ON);
}

TEST_F(ZeroCopyTest, ScalesForwardPackedFp4RankMajorPartialWindowAutoPreservesBytesAndScales) {
    run_scales_forward_rank_major_dispatch(
        kPackedFp4Wire,
        /*windowed_tokens=*/true,
        /*windowed_scales=*/false,
        /*expected_dispatch_result=*/ncclSuccess,
        NCCL_EP_ZERO_COPY_AUTO);
}

} // namespace

// -- main ----------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_zero_copy_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
