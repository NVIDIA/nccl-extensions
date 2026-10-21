/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for HT handle lifecycle:
 *   SeparatedInitUpdate — ncclEpInitHandle followed by ncclEpUpdateHandle
 *                         (no implicit Update inside Create), then dispatch+combine.
 *                         Same handle reused across two iterations with the same routing.
 *   CallerOwnedHandleMem — caller allocates handle_mem (>= ncclEpHandleMemSize) and
 *                          passes a 1D ncclUint8 tensor to ncclEpInitHandle.
 *                          ncclEpHandleDestroy must not free the caller buffer.
 *   ImportsRelocatedPreparedHandleMem -- ncclEpImportHandle from a handle_mem
 *                          snapshot at a different address than the original,
 *                          which keeps mutating; dispatch+combine on the
 *                          imported handle must still work from the snapshot.
 *   ImportedHandleSelectsDispatchChunkFromNumTokens -- ncclEpImportHandle recomputes
 *                          dispatch_tokens_per_chunk from state.num_tokens (not the
 *                          group default), covering the 32/64 chunk selection boundary.
 *   MultiIterReuse        — single ncclEpCreateHandle followed by 4 back-to-back
 *                          dispatch+combine pairs (buffer ping-pong / signal counter
 *                          correctness across iterations).
 *
 * Setup: same as test_output_layout (4 ranks, 8 experts, top-k=1, FLAT/RM by default).
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include "../device/ht_ep.cuh"
#include "../device/ll_ep_adapter.cuh"

TEST(LlDispatchSmem, StagingRespectsCapacityAndHiddenLimit) {
    using nccl_ep::ll::ll_dispatch_stage_quant;
    constexpr int required = 7168 + 56 * sizeof(float) + 32;
    EXPECT_TRUE(ll_dispatch_stage_quant(7168, required));
    EXPECT_FALSE(ll_dispatch_stage_quant(7168, required - 1));
    EXPECT_FALSE(ll_dispatch_stage_quant(7168, 0));
    EXPECT_FALSE(ll_dispatch_stage_quant(7168, -1));
    EXPECT_TRUE(ll_dispatch_stage_quant(512, 1024));
    EXPECT_TRUE(ll_dispatch_stage_quant(8192, 48 * 1024));
    constexpr int largest_required = 16384 + 128 * sizeof(float) + 32;
    EXPECT_TRUE(ll_dispatch_stage_quant(16384, largest_required));
    EXPECT_FALSE(ll_dispatch_stage_quant(16384, largest_required - 1));
    EXPECT_FALSE(ll_dispatch_stage_quant(16640, 1024 * 1024));
    EXPECT_FALSE(ll_dispatch_stage_quant(0, 48 * 1024));
}

TEST(LlCombineSmem, PreservesRequestedParallelismWhenItFits) {
    const auto config = nccl_ep::ll::choose_combine_smem_config(
        /*hidden=*/4096,
        ncclBfloat16,
        NCCL_EP_COMB_QUANT_NONE,
        /*num_warp_groups=*/1,
        /*requested_warps_per_group=*/32,
        /*max_dynamic_smem=*/1024 * 1024);
    ASSERT_TRUE(config.feasible);
    EXPECT_EQ(config.num_warps_per_group, 32);
    EXPECT_EQ(config.num_warps, 32);
}

TEST(LlCombineSmem, ReducesParallelismToFitSharedMemory) {
    const auto config = nccl_ep::ll::choose_combine_smem_config(
        /*hidden=*/7168,
        ncclBfloat16,
        NCCL_EP_COMB_QUANT_NONE,
        /*num_warp_groups=*/1,
        /*requested_warps_per_group=*/32,
        /*max_dynamic_smem=*/100000);
    ASSERT_TRUE(config.feasible);
    EXPECT_LT(config.num_warps_per_group, 32);
    EXPECT_EQ(config.num_recv_groups, 1);
    EXPECT_LE(config.dynamic_smem_bytes, 100000);
}

TEST(LlCombineSmem, AcceptsExactSharedMemoryBoundary) {
    int recv_groups = 0;
    const int exact_limit = nccl_ep::ll::ll_combine_dynamic_smem_bytes(
        /*hidden=*/4096, ncclBfloat16, NCCL_EP_COMB_QUANT_NONE, /*num_warps=*/32, &recv_groups);
    const auto config = nccl_ep::ll::choose_combine_smem_config(
        /*hidden=*/4096,
        ncclBfloat16,
        NCCL_EP_COMB_QUANT_NONE,
        /*num_warp_groups=*/1,
        /*requested_warps_per_group=*/32,
        exact_limit);
    ASSERT_TRUE(config.feasible);
    EXPECT_EQ(config.num_warps_per_group, 32);
    EXPECT_EQ(config.dynamic_smem_bytes, exact_limit);
}

TEST(LlCombineSmem, PadsGptOssHiddenForPartialDecodeWarp) {
    constexpr int kHidden = 2880;
    constexpr int kElemBytes = nccl_ep::ll::combine_smem::kTwoByteElementBytes;
    constexpr int kPaddedHiddenInt4 = nccl_ep::ll::combine_smem::padded_hidden_int4(kHidden, kElemBytes);

    EXPECT_EQ(kHidden * kElemBytes / sizeof(int4), 360);
    EXPECT_EQ(kPaddedHiddenInt4, 384);
    EXPECT_EQ(nccl_ep::ll::combine_smem::recv_decode_warps(kHidden, kElemBytes), 6);
    const int send_warp_bytes = nccl_ep::ll::combine_smem::send_warp_smem_bytes(kHidden, kElemBytes);
    EXPECT_EQ(send_warp_bytes, 3216);
    EXPECT_EQ(send_warp_bytes % sizeof(int4), 0);
    EXPECT_EQ((2 * send_warp_bytes) % sizeof(int4), 0);
    const int recv_group_bytes =
        nccl_ep::ll::combine_smem::recv_group_smem_bytes(kHidden, kElemBytes, /*nvfp4=*/false);
    EXPECT_EQ(recv_group_bytes, 25472);
    EXPECT_EQ(recv_group_bytes % sizeof(int4), 0);
    EXPECT_EQ((2 * recv_group_bytes) % sizeof(int4), 0);

    const auto config = nccl_ep::ll::choose_combine_smem_config(
        kHidden,
        ncclBfloat16,
        NCCL_EP_COMB_QUANT_NONE,
        /*num_warp_groups=*/1,
        /*requested_warps_per_group=*/32,
        /*max_dynamic_smem=*/1024 * 1024);
    ASSERT_TRUE(config.feasible);
    EXPECT_GT(config.num_recv_groups, 0);

    const nccl_ep::LowLatencyLayout layout(
        /*num_max_dispatch_tokens_per_rank=*/1,
        /*max_token_bytes=*/kHidden * kElemBytes,
        /*num_ranks=*/2,
        /*num_experts=*/128,
        /*num_topk=*/4,
        NCCL_EP_LAYOUT_RANK_MAJOR);
    EXPECT_EQ(layout.buffer.num_bytes_per_combine_msg, 5856);
    EXPECT_EQ(layout.buffer.num_bytes_per_combine_msg % sizeof(int4), 0);
}

TEST(LlCombineSmem, RejectsConfigurationsBelowMinimumFootprint) {
    const auto config = nccl_ep::ll::choose_combine_smem_config(
        /*hidden=*/7168,
        ncclBfloat16,
        NCCL_EP_COMB_QUANT_NONE,
        /*num_warp_groups=*/1,
        /*requested_warps_per_group=*/32,
        /*max_dynamic_smem=*/50000);
    EXPECT_FALSE(config.feasible);
}

TEST(LlCombineSmem, Nvfp4ReceiveBufferMatchesDevicePacketLayout) {
    constexpr int kHidden = 7168;
    constexpr int kPacketBytes =
        kHidden / nccl_ep::ll::combine_smem::kNvfp4ElementsPerPackedByte +
        kHidden / nccl_ep::ll::combine_smem::kNvfp4ElementsPerScale +
        nccl_ep::ll::combine_smem::kNvfp4GlobalScaleBytes;
    constexpr int kAlignedPacketBytes =
        (kPacketBytes + sizeof(int4) - 1) / sizeof(int4) * sizeof(int4);

    EXPECT_EQ(nccl_ep::ll::combine_smem::nvfp4_slot_bytes(kHidden), kAlignedPacketBytes);
    EXPECT_EQ(
        nccl_ep::ll::combine_smem::recv_tma_buffer_bytes(
            kHidden, nccl_ep::ll::combine_smem::kTwoByteElementBytes, /*nvfp4=*/true),
        nccl_ep::ll::combine_smem::kRecvTmaBarrierBytes + kAlignedPacketBytes);
}

// Pure host-side sizing tests for combine_push's chunking (see
// combine_push_chunk_size/combine_push_slots_per_block in device/ht_ep.cuh). smem_cap is passed
// explicitly (never queried from the live device) so these stay deterministic and portable across
// whatever GPU CI happens to run on. hidden_int4=896 (ntiles=7, tile_int4=128) is the shape
// ep_bench -V exercises end to end; hidden_int4=2 covers the single-tile edge case.
struct CombinePushChunkingCase {
    std::string name;
    int hidden_int4;
    int chunk_size;          // requested chunk width in int4; <1 = whole row
    int smem_cap;
    int expected_chunks;     // combine_push_num_chunks's chunk count
    int expected_buf_int4;   // combine_push_buf_int4's per-warp buffer width
    int expected_slots;      // combine_push_slots_per_block's warp count
};

const std::vector<CombinePushChunkingCase> kCombinePushChunkingCases = {
    // A single-tile row is one chunk regardless of the request.
    {"SmallHiddenWholeRow", 2, 0, 232448, 1, 2, 24},
    // Unset is the whole row.
    {"UnsetIsWholeRow", 896, 0, 232448, 1, 7 * 128, 15},
    // A request up to one tile is one tile per chunk.
    {"OneTilePerChunk", 896, 128, 232448, 7, 128, 24},
    {"BelowOneTileRoundsUp", 896, 1, 232448, 7, 128, 24},
    // A request that is not a tile multiple rounds up; the last chunk is short.
    {"TwoTilesPerChunk", 896, 200, 232448, 4, 2 * 128, 24},
    // A request at or above the row clamps to the whole row.
    {"ClampedToWholeRow", 896, 100000, 232448, 1, 7 * 128, 15},
    // Whole row that is not a tile multiple: buffer is the exact row, not tile-rounded (769 pads to
    // 896 and would give 15 slots).
    {"NonAlignedWholeRow769", 769, 0, 232448, 1, 769, 18},
    // 513 pads to 640; at this cap the exact width fits all 24 slots, the padded width only 19.
    {"NonAlignedWholeRow513", 513, 0, 200800, 1, 513, 24},
    // A request at or above a non-aligned row clamps to the exact row.
    {"NonAlignedClampedToWholeRow", 769, 100000, 232448, 1, 769, 18},
    // Multi-chunk on a non-aligned row keeps a tile-multiple reusable buffer.
    {"NonAlignedOneTilePerChunk", 769, 128, 232448, 7, 128, 24},
    // A tight smem cap forces slots down to the floor of 1, never below.
    {"TinyCapClampsToOneSlot", 896, 128, 6000, 7, 128, 1},
};

class CombinePushChunkingTest : public ::testing::TestWithParam<CombinePushChunkingCase> {};

TEST_P(CombinePushChunkingTest, ChunksBufWidthAndSlotsMatchExpected) {
    const auto& c = GetParam();
    EXPECT_EQ(ht_ep::combine_push_num_chunks(c.hidden_int4, c.chunk_size), c.expected_chunks);
    EXPECT_EQ(ht_ep::combine_push_buf_int4(c.hidden_int4, c.chunk_size), c.expected_buf_int4);
    EXPECT_EQ(ht_ep::combine_push_slots_per_block(c.hidden_int4, c.smem_cap, c.chunk_size),
              c.expected_slots);
    // dynamic_smem_bytes is the actual CUDA launch's dynamic-smem request; must stay in lockstep
    // with the slots/buf_int4 it's built from.
    EXPECT_EQ(ht_ep::combine_push_dynamic_smem_bytes(c.hidden_int4, c.expected_slots, c.chunk_size),
              static_cast<size_t>(c.expected_slots) * c.expected_buf_int4 * 16);
}

INSTANTIATE_TEST_SUITE_P(Sizing, CombinePushChunkingTest, ::testing::ValuesIn(kCombinePushChunkingCases),
                          [](const ::testing::TestParamInfo<CombinePushChunkingCase>& info) {
                              return info.param.name;
                          });

static float bf16_val(nv_bfloat16 v) {
    return __bfloat162float(v);
}

class LifecycleTest : public EpTestBase {
protected:
    // Run one dispatch+combine round under FLAT (rank-major) layout on the given handle.
    // Returns the per-token first-hidden-element bf16 value vector of length kNumTokens.
    std::vector<float> run_dispatch_combine_flat(ncclEpHandle_t handle) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv, *d_out;
        float *d_weights, *d_recv_w;
        int64_t* d_recv_idx;
        EXPECT_EQ(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w, kMaxRecvSlots * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(
            cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);
        EXPECT_EQ(
            cudaMemcpy(d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

        ncclEpTensor_t *t_tok, *t_recv, *t_out, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w, 2, ncclFloat32, d_weights, kNumTokens, kTopK), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_w, 2, ncclFloat32, d_recv_w, kMaxRecvSlots, kTopK), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64, d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);

        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = t_tok;
        d_in.topk_weights = t_w;
        d_out_s.tokens = t_recv;
        d_out_s.topk_weights = t_recv_w;
        d_out_s.topk_idx = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = t_recv;
        c_out_s.tokens = t_out;
        EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out_s, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        EXPECT_EQ(
            cudaMemcpy(h_out.data(), d_out, kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost),
            cudaSuccess);

        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_out);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok);
        cudaFree(d_recv);
        cudaFree(d_out);
        cudaFree(d_weights);
        cudaFree(d_recv_w);
        cudaFree(d_recv_idx);

        std::vector<float> vals(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i) vals[i] = bf16_val(h_out[i * kHidden]);
        return vals;
    }
};

TEST_F(LifecycleTest, ResolvesIndependentSmBudgetsAndEnvironmentPrecedence) {
    int device = 0;
    int device_sms = 0;
    CUDA_ASSERT(cudaGetDevice(&device));
    CUDA_ASSERT(cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device));
    ASSERT_GE(device_sms, 8);

    const char* old_comm_ptr = getenv("NCCL_EP_COMM_SMS");
    const char* old_dispatch_ptr = getenv("NCCL_EP_DISPATCH_SMS");
    const char* old_combine_ptr = getenv("NCCL_EP_COMBINE_SMS");
    const bool had_comm = old_comm_ptr != nullptr;
    const bool had_dispatch = old_dispatch_ptr != nullptr;
    const bool had_combine = old_combine_ptr != nullptr;
    const std::string old_comm = had_comm ? old_comm_ptr : "";
    const std::string old_dispatch = had_dispatch ? old_dispatch_ptr : "";
    auto restore_env = [](const char* name, bool was_set, const std::string& old_value) {
        if (was_set) setenv(name, old_value.c_str(), 1);
        else unsetenv(name);
    };
    const std::string old_combine = had_combine ? old_combine_ptr : "";
    unsetenv("NCCL_EP_COMM_SMS");
    unsetenv("NCCL_EP_DISPATCH_SMS");
    unsetenv("NCCL_EP_COMBINE_SMS");

    const unsigned int shared = static_cast<unsigned int>(std::min(device_sms, 8));
    const unsigned int dispatch = shared - 1;
    auto make_group = [&](unsigned int dispatch_cfg, unsigned int combine_cfg) {
        ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
        cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        cfg.num_experts = kNumExperts;
        cfg.max_dispatch_tokens_per_rank = kNumTokens;
        cfg.max_recv_tokens_per_rank = kMaxRecvSlots;
        cfg.max_token_bytes = kHidden * sizeof(nv_bfloat16);
        cfg.rdma_buffer_size = NCCL_EP_AUTO;
        cfg.num_qp_per_rank = NCCL_EP_AUTO;
        cfg.num_channels = NCCL_EP_AUTO;
        cfg.max_num_sms = shared;
        cfg.dispatch_num_sms = dispatch_cfg;
        cfg.combine_num_sms = combine_cfg;
        ncclEpGroup_t group = nullptr;
        EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &cfg), ncclSuccess);
        return group;
    };

    // One operation-specific field does not change the other's shared fallback.
    ncclEpGroup_t group = make_group(dispatch, NCCL_EP_AUTO);
    unsigned int resolved_dispatch = 0, resolved_combine = 0;
    ncclEpGroup_test_getSmBudgets(group, &resolved_dispatch, &resolved_combine);
    EXPECT_EQ(resolved_dispatch, dispatch);
    EXPECT_EQ(resolved_combine, shared);
    NCCL_ASSERT(ncclEpGroupDestroy(group));

    const unsigned int env_shared = shared - 2;
    const unsigned int env_dispatch = shared - 3;
    const std::string env_shared_str = std::to_string(env_shared);
    const std::string env_dispatch_str = std::to_string(env_dispatch);
    const std::string invalid_str = std::to_string(device_sms + 1);
    setenv("NCCL_EP_COMM_SMS", env_shared_str.c_str(), 1);
    setenv("NCCL_EP_DISPATCH_SMS", env_dispatch_str.c_str(), 1);
    setenv("NCCL_EP_COMBINE_SMS", invalid_str.c_str(), 1);
    group = make_group(dispatch, shared - 1);
    ncclEpGroup_test_getSmBudgets(group, &resolved_dispatch, &resolved_combine);
    EXPECT_EQ(resolved_dispatch, env_dispatch);
    EXPECT_EQ(resolved_combine, env_shared);
    NCCL_ASSERT(ncclEpGroupDestroy(group));

    restore_env("NCCL_EP_COMM_SMS", had_comm, old_comm);
    restore_env("NCCL_EP_DISPATCH_SMS", had_dispatch, old_dispatch);
    restore_env("NCCL_EP_COMBINE_SMS", had_combine, old_combine);

}
TEST_F(LifecycleTest, LlCombineFitsMockedSmallSharedMemoryAndPreservesResults) {
    constexpr int kLargeHidden = 7168;
    constexpr int kMockDynamicSmem = 100000;
    const int recv_rows = g_nranks * kNumTokens;

    ncclEpGroupConfig_t group_config = NCCL_EP_GROUP_CONFIG_INIT;
    group_config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    group_config.num_experts = kNumExperts;
    group_config.max_dispatch_tokens_per_rank = kNumTokens;
    group_config.max_recv_tokens_per_rank = recv_rows;
    group_config.max_token_bytes = kLargeHidden * sizeof(nv_bfloat16);
    group_config.rdma_buffer_size = NCCL_EP_AUTO;
    group_config.num_qp_per_rank = kNumExperts / g_nranks;
    group_config.num_channels = NCCL_EP_AUTO;

    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &group_config));
    NCCL_ASSERT(ncclEpGroup_test_setMaxDynamicSmem(group, kMockDynamicSmem));

    nv_bfloat16 *d_tokens = nullptr, *d_recv = nullptr, *d_out = nullptr;
    float *d_weights = nullptr, *d_recv_weights = nullptr;
    int *d_recv_topk_idx = nullptr, *d_src_rank_counters = nullptr;
    ncclEpTensor_t *tokens = nullptr, *recv = nullptr, *out = nullptr, *weights = nullptr;
    ncclEpTensor_t *recv_weights = nullptr, *recv_topk_idx = nullptr, *src_rank_counters = nullptr;
    ncclEpHandle_t handle = nullptr;

    std::vector<nv_bfloat16> host_tokens(kNumTokens * kLargeHidden);
    for (int token = 0; token < kNumTokens; ++token) {
        const float value = static_cast<float>(g_rank * kNumTokens + token + 1);
        for (int h = 0; h < kLargeHidden; ++h) host_tokens[token * kLargeHidden + h] = __float2bfloat16(value);
    }
    std::vector<float> host_weights(kNumTokens * kTopK, 1.0f);

    CUDA_ASSERT(cudaMalloc(&d_tokens, host_tokens.size() * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv, recv_rows * kLargeHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out, host_tokens.size() * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights, host_weights.size() * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_weights, recv_rows * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_topk_idx, recv_rows * kTopK * sizeof(int)));
    CUDA_ASSERT(cudaMalloc(&d_src_rank_counters, g_nranks * sizeof(int)));
    CUDA_ASSERT(cudaMemcpy(
        d_tokens, host_tokens.data(), host_tokens.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_weights, host_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_recv, 0, recv_rows * kLargeHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv_weights, 0, recv_rows * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv_topk_idx, 0xff, recv_rows * kTopK * sizeof(int)));
    CUDA_ASSERT(cudaMemset(d_src_rank_counters, 0, g_nranks * sizeof(int)));

    NCCL_ASSERT(epTensorCreate(&tokens, 2, ncclBfloat16, d_tokens, kNumTokens, kLargeHidden));
    NCCL_ASSERT(epTensorCreate(&recv, 3, ncclBfloat16, d_recv, g_nranks, kNumTokens, kLargeHidden));
    NCCL_ASSERT(epTensorCreate(&out, 2, ncclBfloat16, d_out, kNumTokens, kLargeHidden));
    NCCL_ASSERT(epTensorCreate(&weights, 2, ncclFloat32, d_weights, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&recv_weights, 3, ncclFloat32, d_recv_weights, g_nranks, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&recv_topk_idx, 3, ncclInt32, d_recv_topk_idx, g_nranks, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&src_rank_counters, 1, ncclInt32, d_src_rank_counters, g_nranks));
    NCCL_ASSERT(ncclEpCreateHandle(
        &handle, group, NCCL_EP_LAYOUT_RANK_MAJOR, topk_idx_, nullptr, nullptr, g_stream));

    ncclEpDispatchInputs_t dispatch_inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dispatch_outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    dispatch_inputs.tokens = tokens;
    dispatch_inputs.topk_weights = weights;
    dispatch_outputs.tokens = recv;
    dispatch_outputs.topk_weights = recv_weights;
    dispatch_outputs.topk_idx = recv_topk_idx;
    layout_info.src_rank_counters = src_rank_counters;
    NCCL_ASSERT(ncclEpDispatch(handle, &dispatch_inputs, &dispatch_outputs, &layout_info, nullptr, g_stream));
    NCCL_ASSERT(ncclEpComplete(handle, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    ncclEpCombineInputs_t combine_inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t combine_outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    combine_inputs.tokens = recv;
    combine_outputs.tokens = out;
    NCCL_ASSERT(ncclEpCombine(handle, &combine_inputs, &combine_outputs, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    EXPECT_LT(ncclEpGroup_test_getLastLlCombineWarpsPerGroup(group), 32);

    std::vector<nv_bfloat16> host_out(host_tokens.size());
    CUDA_ASSERT(cudaMemcpy(
        host_out.data(), d_out, host_out.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int token = 0; token < kNumTokens; ++token) {
        EXPECT_NEAR(
            bf16_val(host_out[token * kLargeHidden]), static_cast<float>(g_rank * kNumTokens + token + 1), 0.5f);
    }

    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
    ncclEpTensorDestroy(tokens);
    ncclEpTensorDestroy(recv);
    ncclEpTensorDestroy(out);
    ncclEpTensorDestroy(weights);
    ncclEpTensorDestroy(recv_weights);
    ncclEpTensorDestroy(recv_topk_idx);
    ncclEpTensorDestroy(src_rank_counters);
    cudaFree(d_tokens);
    cudaFree(d_recv);
    cudaFree(d_out);
    cudaFree(d_weights);
    cudaFree(d_recv_weights);
    cudaFree(d_recv_topk_idx);
    cudaFree(d_src_rank_counters);
}

// ── Test: SeparatedInitUpdate — Init then Update, dispatch+combine ────────────
// ncclEpCreateHandle bundles Init+Update; this exercises them separately to
// verify Update alone establishes the routing for subsequent dispatch.

TEST_F(LifecycleTest, SeparatedInitUpdate) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, nullptr));
    ASSERT_NE(h, nullptr);
    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Fused count mode publishes the recv count inside dispatch, so it is only valid after the
    // first dispatch; scan and unfused count wire it at UpdateHandle time.
    unsigned int num_recv = 0;
    if (!ht_count_fused_active(h)) {
        NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
        EXPECT_EQ(num_recv, 4u) << "FLAT recv count for top-k=1 fixed routing";
    }

    auto combined = run_dispatch_combine_flat(h);
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u) << "FLAT recv count after dispatch for top-k=1 fixed routing";
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: SeparatedInitUpdate_TwoUpdates — Init once, Update twice, reuse ─────
// Same handle, second Update with the same topk_idx (re-runs allgather + preprocess).
// Both iterations must yield identical correct results.

TEST_F(LifecycleTest, SeparatedInitUpdateTwoUpdates) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, nullptr));
    ASSERT_NE(h, nullptr);

    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    auto combined_a = run_dispatch_combine_flat(h);

    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    auto combined_b = run_dispatch_combine_flat(h);

    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined_a[i], expected, 0.5f) << "first iter token " << i;
        EXPECT_NEAR(combined_b[i], expected, 0.5f) << "second iter token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// Scan preprocessing is ordered at group scope (its routing-map AllGather is already
// group-owned), so multiple handles share its update-only workspace. Updating h2 before
// dispatching h1 verifies that prepared routing state remains private to each handle.
TEST_F(LifecycleTest, ScanWorkspaceSharedAcrossHandles) {
    // The shared scan workspace exists only on the scan metadata path. Count mode (the
    // AUTO default for Flat and EM local-permute) builds its maps inside dispatch and never
    // allocates it, so pin this group to FORCE_SCAN to test what the name says in every pass.
    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kNumTokens;
    gcfg.max_token_bytes = kHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank = NCCL_EP_AUTO;
    gcfg.num_channels = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecvSlots);
    gcfg.meta_exchange_policy = NCCL_EP_META_EXCHANGE_FORCE_SCAN;
    ncclEpGroup_t scan_group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&scan_group, g_comm, &gcfg));

    ncclEpHandle_t h1 = nullptr;
    ncclEpHandle_t h2 = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h1, scan_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, nullptr));
    NCCL_ASSERT(ncclEpInitHandle(&h2, scan_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, nullptr));

    const void *h1_rank_mask = nullptr, *h1_scan_tmp = nullptr;
    const void *h2_rank_mask = nullptr, *h2_scan_tmp = nullptr;
    const int32_t *h1_token_to_slot = nullptr, *h2_token_to_slot = nullptr;
    ncclEpHandle_test_getScanScratch(h1, &h1_rank_mask, &h1_scan_tmp, &h1_token_to_slot);
    ncclEpHandle_test_getScanScratch(h2, &h2_rank_mask, &h2_scan_tmp, &h2_token_to_slot);
    ASSERT_NE(h1_rank_mask, nullptr);
    ASSERT_NE(h1_scan_tmp, nullptr);
    EXPECT_EQ(h1_rank_mask, h2_rank_mask);
    EXPECT_EQ(h1_scan_tmp, h2_scan_tmp);
    EXPECT_EQ(h1_token_to_slot, nullptr);  // FLAT does not need the EM remap bridge.
    EXPECT_EQ(h2_token_to_slot, nullptr);

    NCCL_ASSERT(ncclEpUpdateHandle(h1, topk_idx_, nullptr, g_stream));
    NCCL_ASSERT(ncclEpUpdateHandle(h2, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    for (ncclEpHandle_t handle : {h1, h2}) {
        auto combined = run_dispatch_combine_flat(handle);
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
            EXPECT_NEAR(combined[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h2));
    NCCL_ASSERT(ncclEpHandleDestroy(h1));

    // The scan-mode EM local-permute path additionally shares its token-to-slot bridge.
    // This branch runs in the suite's NCCL_EP_HT_AG_SCAN_MODE rerun.
    if (ht_ag_scan_mode_active()) {
        ncclEpHandle_t em1 = nullptr;
        ncclEpHandle_t em2 = nullptr;
        NCCL_ASSERT(ncclEpInitHandle(
            &em1, scan_group, NCCL_EP_LAYOUT_EXPERT_MAJOR, nullptr, kTopK, nullptr));
        NCCL_ASSERT(ncclEpInitHandle(
            &em2, scan_group, NCCL_EP_LAYOUT_EXPERT_MAJOR, nullptr, kTopK, nullptr));

        const void *em1_rank_mask = nullptr, *em1_scan_tmp = nullptr;
        const void *em2_rank_mask = nullptr, *em2_scan_tmp = nullptr;
        const int32_t *em1_token_to_slot = nullptr, *em2_token_to_slot = nullptr;
        ncclEpHandle_test_getScanScratch(em1, &em1_rank_mask, &em1_scan_tmp, &em1_token_to_slot);
        ncclEpHandle_test_getScanScratch(em2, &em2_rank_mask, &em2_scan_tmp, &em2_token_to_slot);
        ASSERT_NE(em1_token_to_slot, nullptr);
        EXPECT_EQ(em1_rank_mask, h1_rank_mask);
        EXPECT_EQ(em1_scan_tmp, h1_scan_tmp);
        EXPECT_EQ(em1_rank_mask, em2_rank_mask);
        EXPECT_EQ(em1_scan_tmp, em2_scan_tmp);
        EXPECT_EQ(em1_token_to_slot, em2_token_to_slot);

        NCCL_ASSERT(ncclEpUpdateHandle(em1, topk_idx_, nullptr, g_stream));
        NCCL_ASSERT(ncclEpUpdateHandle(em2, topk_idx_, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        NCCL_ASSERT(ncclEpHandleDestroy(em2));
        NCCL_ASSERT(ncclEpHandleDestroy(em1));
    }
    NCCL_ASSERT(ncclEpGroupDestroy(scan_group));
}

// ── Test: CallerOwnedHandleMem — caller-owned handle_mem buffer ───────────────
// Caller queries the required size, allocates, wraps as ncclUint8 1D tensor,
// passes to ncclEpInitHandle. Verifies roundtrip correctness and that
// ncclEpHandleDestroy does not free the caller buffer (we cudaFree it ourselves).

TEST_F(LifecycleTest, CallerOwnedHandleMem) {
    size_t handle_mem_bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, &handle_mem_bytes, kTopK));
    ASSERT_GT(handle_mem_bytes, 0u);

    void* d_handle_mem = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_handle_mem, handle_mem_bytes));

    ncclEpTensor_t* t_handle_mem = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_handle_mem, 1, ncclUint8, d_handle_mem, static_cast<unsigned int>(handle_mem_bytes)));

    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, t_handle_mem));
    ASSERT_NE(h, nullptr);
    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    auto combined = run_dispatch_combine_flat(h);
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
    // The caller still owns d_handle_mem; free it ourselves.
    ncclEpTensorDestroy(t_handle_mem);
    CUDA_ASSERT(cudaFree(d_handle_mem));
}

// -- Test: ImportsRelocatedPreparedHandleMem -- ncclEpImportHandle from a snapshot --
// Snapshots a prepared handle_mem into a second buffer, then keeps mutating the
// original so the two buffers diverge. Dispatch+combine on the imported handle
// must still complete correctly from the snapshot alone (also covers ncclInt32
// topk_idx_datatype recovery).

TEST_F(LifecycleTest, ImportsRelocatedPreparedHandleMem) {

    auto make_handle_mem = [&](size_t bytes, void** d_mem, ncclEpTensor_t** t) {
        CUDA_ASSERT(cudaMalloc(d_mem, bytes));
        NCCL_ASSERT(epTensorCreate(t, 1, ncclUint8, *d_mem, static_cast<unsigned int>(bytes)));
    };

    size_t handle_mem_bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, &handle_mem_bytes, kTopK));

    void* d_forward_mem = nullptr;
    void* d_restored_mem = nullptr;
    ncclEpTensor_t* t_forward_mem = nullptr;
    ncclEpTensor_t* t_restored_mem = nullptr;
    make_handle_mem(handle_mem_bytes, &d_forward_mem, &t_forward_mem);
    make_handle_mem(handle_mem_bytes, &d_restored_mem, &t_restored_mem);

    std::vector<int32_t> h_topk(kNumTokens * kTopK);
    for (int i = 0; i < kNumTokens; ++i) h_topk[i] = expert_for_token(i);
    int32_t* d_topk = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk, h_topk.size() * sizeof(int32_t)));
    CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    ncclEpTensor_t* t_topk = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_topk, 2, ncclInt32, d_topk, kNumTokens, kTopK));

    ncclEpHandle_t forward_handle = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(
        &forward_handle, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, t_forward_mem));
    NCCL_ASSERT(ncclEpUpdateHandle(forward_handle, t_topk, nullptr, g_stream));
    CUDA_ASSERT(cudaMemcpyAsync(
        d_restored_mem, d_forward_mem, handle_mem_bytes, cudaMemcpyDeviceToDevice, g_stream));

    // Mutate the live handle after the snapshot: the import must not see this.
    for (int i = 0; i < kNumTokens; ++i) h_topk[i] = (expert_for_token(i) + 1) % kNumExperts;
    CUDA_ASSERT(cudaMemcpyAsync(
        d_topk, h_topk.data(), h_topk.size() * sizeof(int32_t), cudaMemcpyHostToDevice, g_stream));
    NCCL_ASSERT(ncclEpUpdateHandle(forward_handle, t_topk, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    ncclEpHandleState_t state = NCCL_EP_HANDLE_STATE_INIT;
    state.num_tokens = kNumTokens;
    state.num_topk = kTopK;
    state.topk_idx_datatype = ncclInt32;
    ncclEpHandle_t restored_handle = nullptr;
    NCCL_ASSERT(ncclEpImportHandle(
        &restored_handle, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, &state, t_restored_mem));

    NCCL_ASSERT(ncclEpHandleDestroy(forward_handle));
    ncclEpTensorDestroy(t_forward_mem);
    ncclEpTensorDestroy(t_topk);
    CUDA_ASSERT(cudaFree(d_forward_mem));
    CUDA_ASSERT(cudaFree(d_topk));

    auto combined = run_dispatch_combine_flat(restored_handle);
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(restored_handle));
    ncclEpTensorDestroy(t_restored_mem);
    CUDA_ASSERT(cudaFree(d_restored_mem));
}

// -- Test: ImportedHandleSelectsDispatchChunkFromNumTokens -------------------
// ncclEpImportHandle must recompute dispatch_tokens_per_chunk from
// state.num_tokens exactly as ncclEpUpdateHandle does, not inherit the group
// default installed by ncclEpInitHandle. A mismatch silently corrupts
// own_chunk_rank in count-mode dispatch (EM+kLocalPermute).
//
// Uses 33+ tokens so the adaptive policy returns 32 on systems with
// dispatch_num_sms in [16,64]; on narrower SM counts the function returns the
// group default regardless, and the test still verifies UpdateHandle and Import
// agree.
TEST_F(LifecycleTest, ImportedHandleSelectsDispatchChunkFromNumTokens) {

    constexpr int kLargeTokens = 48;
    constexpr int kLargeRecvSlots = 4 * kLargeTokens;

    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kLargeTokens;
    gcfg.max_token_bytes = kHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank = NCCL_EP_AUTO;
    gcfg.num_channels = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank = kLargeRecvSlots;

    ncclEpGroup_t local_group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&local_group, g_comm, &gcfg));

    size_t handle_mem_bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(local_group, NCCL_EP_LAYOUT_FLAT, nullptr, &handle_mem_bytes, kTopK));

    void* d_src_mem = nullptr;
    void* d_snap_mem = nullptr;
    ncclEpTensor_t* t_src_mem = nullptr;
    ncclEpTensor_t* t_snap_mem = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_src_mem, handle_mem_bytes));
    CUDA_ASSERT(cudaMalloc(&d_snap_mem, handle_mem_bytes));
    NCCL_ASSERT(epTensorCreate(&t_src_mem, 1, ncclUint8, d_src_mem, (unsigned int)handle_mem_bytes));
    NCCL_ASSERT(epTensorCreate(&t_snap_mem, 1, ncclUint8, d_snap_mem, (unsigned int)handle_mem_bytes));

    std::vector<int32_t> h_topk(kLargeTokens * kTopK);
    for (int i = 0; i < kLargeTokens; ++i) h_topk[i] = (g_rank * kLargeTokens + i) % kNumExperts;
    int32_t* d_topk = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk, h_topk.size() * sizeof(int32_t)));
    CUDA_ASSERT(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    ncclEpTensor_t* t_topk = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_topk, 2, ncclInt32, d_topk, kLargeTokens, kTopK));

    ncclEpHandle_t src_handle = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&src_handle, local_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, t_src_mem));
    NCCL_ASSERT(ncclEpUpdateHandle(src_handle, t_topk, nullptr, g_stream));

    CUDA_ASSERT(cudaMemcpyAsync(
        d_snap_mem, d_src_mem, handle_mem_bytes, cudaMemcpyDeviceToDevice, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    const int expected_chunk = ncclEpHandle_test_getDispatchTokensPerChunk(src_handle);

    ncclEpHandleState_t state = NCCL_EP_HANDLE_STATE_INIT;
    state.num_tokens = kLargeTokens;
    state.num_topk = kTopK;
    state.topk_idx_datatype = ncclInt32;
    ncclEpHandle_t imported_handle = nullptr;
    NCCL_ASSERT(ncclEpImportHandle(
        &imported_handle, local_group, NCCL_EP_LAYOUT_FLAT, nullptr, &state, t_snap_mem));

    EXPECT_EQ(ncclEpHandle_test_getDispatchTokensPerChunk(imported_handle), expected_chunk)
        << "Import must recompute dispatch_tokens_per_chunk from num_tokens, not the group default";

    NCCL_ASSERT(ncclEpHandleDestroy(imported_handle));
    NCCL_ASSERT(ncclEpHandleDestroy(src_handle));
    ncclEpTensorDestroy(t_snap_mem);
    ncclEpTensorDestroy(t_src_mem);
    ncclEpTensorDestroy(t_topk);
    CUDA_ASSERT(cudaFree(d_snap_mem));
    CUDA_ASSERT(cudaFree(d_src_mem));
    CUDA_ASSERT(cudaFree(d_topk));
    NCCL_ASSERT(ncclEpGroupDestroy(local_group));
}

// ── Test: MultiIterHandleReuse — 4 dispatch+combine pairs on one handle ───────
// Exercises buffer ping-pong and signal counter increments across multiple iterations.

TEST_F(LifecycleTest, MultiIterHandleReuse) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    for (int iter = 0; iter < 4; ++iter) {
        auto combined = run_dispatch_combine_flat(h);
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
            EXPECT_NEAR(combined[i], expected, 0.5f) << "iter " << iter << " rank " << g_rank << " token " << i;
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_lifecycle_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
