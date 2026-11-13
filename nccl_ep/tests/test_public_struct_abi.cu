/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Public-struct ABI tests:
 *   - initializers populate the public size/magic/version prefix
 *   - larger future objects are accepted and their unknown tail is ignored
 *   - undersized and wrong-type objects are rejected
 */

#include "test_common.h"
#include <algorithm>
#include <nccl_device.h>
#include "nccl_ep/common.hpp"

// These checks freeze every public-struct field offset in the 64-bit NCCL EP
// ABI. They do not need to run: compilation catches incompatible layout
// changes. When appending a field, add its offset and update only that
// structure's expected size; existing offsets must not change.
void ncclEpTensor_backwards_compat_test() {
    static_assert(offsetof(ncclEpTensor_t, size) == 0);
    static_assert(offsetof(ncclEpTensor_t, magic) == 4);
    static_assert(offsetof(ncclEpTensor_t, ndim) == 8);
    static_assert(offsetof(ncclEpTensor_t, datatype) == 12);
    static_assert(offsetof(ncclEpTensor_t, data) == 16);
    static_assert(offsetof(ncclEpTensor_t, win_hdl) == 24);
    static_assert(offsetof(ncclEpTensor_t, win_offset) == 32);
    static_assert(offsetof(ncclEpTensor_t, sizes) == 40);
    static_assert(sizeof(ncclEpTensor_t) == 48);
}

void ncclEpTensorAllocConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpTensorAllocConfig_t, size) == 0);
    static_assert(offsetof(ncclEpTensorAllocConfig_t, magic) == 4);
    static_assert(sizeof(ncclEpTensorAllocConfig_t) == 8);
}

void ncclEpAllocConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpAllocConfig_t, alloc_fn) == 0);
    static_assert(offsetof(ncclEpAllocConfig_t, free_fn) == 8);
    static_assert(offsetof(ncclEpAllocConfig_t, context) == 16);
    static_assert(sizeof(ncclEpAllocConfig_t) == 24);
}

void ncclEpGroupConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpGroupConfig_t, size) == 0);
    static_assert(offsetof(ncclEpGroupConfig_t, magic) == 4);
    static_assert(offsetof(ncclEpGroupConfig_t, version) == 8);
    static_assert(offsetof(ncclEpGroupConfig_t, algorithm) == 12);
    static_assert(offsetof(ncclEpGroupConfig_t, num_experts) == 16);
    static_assert(
        offsetof(ncclEpGroupConfig_t, max_dispatch_tokens_per_rank) == 20);
    static_assert(
        offsetof(ncclEpGroupConfig_t, max_recv_tokens_per_rank) == 24);
    static_assert(offsetof(ncclEpGroupConfig_t, max_token_bytes) == 28);
    static_assert(offsetof(ncclEpGroupConfig_t, rdma_buffer_size) == 32);
    static_assert(offsetof(ncclEpGroupConfig_t, num_qp_per_rank) == 40);
    static_assert(offsetof(ncclEpGroupConfig_t, num_channels) == 44);
    static_assert(offsetof(ncclEpGroupConfig_t, max_num_sms) == 48);
    static_assert(offsetof(ncclEpGroupConfig_t, alloc) == 56);
    static_assert(offsetof(ncclEpGroupConfig_t, alloc.alloc_fn) == 56);
    static_assert(offsetof(ncclEpGroupConfig_t, alloc.free_fn) == 64);
    static_assert(offsetof(ncclEpGroupConfig_t, alloc.context) == 72);
    static_assert(offsetof(ncclEpGroupConfig_t, enable_mask) == 80);
    static_assert(offsetof(ncclEpGroupConfig_t, timeout_ns) == 88);
    static_assert(offsetof(ncclEpGroupConfig_t, zero_copy) == 96);
    static_assert(offsetof(ncclEpGroupConfig_t, overflow_policy) == 100);
    static_assert(offsetof(ncclEpGroupConfig_t, num_topk) == 104);
    static_assert(offsetof(ncclEpGroupConfig_t, padding_v2) == 108);
    static_assert(offsetof(ncclEpGroupConfig_t, dispatch_num_sms) == 112);
    static_assert(offsetof(ncclEpGroupConfig_t, combine_num_sms) == 116);
    static_assert(offsetof(ncclEpGroupConfig_t, ht_flow) == 120);
    static_assert(offsetof(ncclEpGroupConfig_t, meta_exchange_policy) == 124);
    static_assert(offsetof(ncclEpGroupConfig_t, num_attn_ranks) == 128);
    static_assert(offsetof(ncclEpGroupConfig_t, padding_v4) == 132);
    static_assert(sizeof(ncclEpGroupConfig_t) == 136);
}

void ncclEpLayoutInfo_backwards_compat_test() {
    static_assert(offsetof(ncclEpLayoutInfo_t, size) == 0);
    static_assert(offsetof(ncclEpLayoutInfo_t, magic) == 4);
    static_assert(offsetof(ncclEpLayoutInfo_t, expert_counters) == 8);
    static_assert(offsetof(ncclEpLayoutInfo_t, src_rank_counters) == 16);
    static_assert(offsetof(ncclEpLayoutInfo_t, expert_offsets) == 24);
    static_assert(offsetof(ncclEpLayoutInfo_t, recv_total_counter) == 32);
    static_assert(offsetof(ncclEpLayoutInfo_t, recv_topk_idx_kind) == 40);
    static_assert(offsetof(ncclEpLayoutInfo_t, padding_v2) == 44);
    static_assert(sizeof(ncclEpLayoutInfo_t) == 48);
}

void ncclEpDispatchInputs_backwards_compat_test() {
    static_assert(offsetof(ncclEpDispatchInputs_t, size) == 0);
    static_assert(offsetof(ncclEpDispatchInputs_t, magic) == 4);
    static_assert(offsetof(ncclEpDispatchInputs_t, tokens) == 8);
    static_assert(offsetof(ncclEpDispatchInputs_t, topk_weights) == 16);
    static_assert(offsetof(ncclEpDispatchInputs_t, scales) == 24);
    static_assert(sizeof(ncclEpDispatchInputs_t) == 32);
}

void ncclEpDispatchOutputs_backwards_compat_test() {
    static_assert(offsetof(ncclEpDispatchOutputs_t, size) == 0);
    static_assert(offsetof(ncclEpDispatchOutputs_t, magic) == 4);
    static_assert(offsetof(ncclEpDispatchOutputs_t, tokens) == 8);
    static_assert(offsetof(ncclEpDispatchOutputs_t, topk_weights) == 16);
    static_assert(offsetof(ncclEpDispatchOutputs_t, scales) == 24);
    static_assert(offsetof(ncclEpDispatchOutputs_t, topk_idx) == 32);
    static_assert(sizeof(ncclEpDispatchOutputs_t) == 40);
}

void ncclEpCombineInputs_backwards_compat_test() {
    static_assert(offsetof(ncclEpCombineInputs_t, size) == 0);
    static_assert(offsetof(ncclEpCombineInputs_t, magic) == 4);
    static_assert(offsetof(ncclEpCombineInputs_t, tokens) == 8);
    static_assert(offsetof(ncclEpCombineInputs_t, topk_weights) == 16);
    static_assert(offsetof(ncclEpCombineInputs_t, scales) == 24);
    static_assert(sizeof(ncclEpCombineInputs_t) == 32);
}

void ncclEpCombineOutputs_backwards_compat_test() {
    static_assert(offsetof(ncclEpCombineOutputs_t, size) == 0);
    static_assert(offsetof(ncclEpCombineOutputs_t, magic) == 4);
    static_assert(offsetof(ncclEpCombineOutputs_t, tokens) == 8);
    static_assert(offsetof(ncclEpCombineOutputs_t, topk_weights) == 16);
    static_assert(sizeof(ncclEpCombineOutputs_t) == 24);
}

void ncclEpHandleConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpHandleConfig_t, size) == 0);
    static_assert(offsetof(ncclEpHandleConfig_t, magic) == 4);
    static_assert(
        offsetof(
            ncclEpHandleConfig_t,
            dispatch_output_per_expert_alignment) == 8);
    static_assert(sizeof(ncclEpHandleConfig_t) == 16);
}

void ncclEpHandleState_backwards_compat_test() {
    static_assert(offsetof(ncclEpHandleState_t, size) == 0);
    static_assert(offsetof(ncclEpHandleState_t, magic) == 4);
    static_assert(offsetof(ncclEpHandleState_t, num_tokens) == 8);
    static_assert(offsetof(ncclEpHandleState_t, num_topk) == 12);
    static_assert(offsetof(ncclEpHandleState_t, topk_idx_datatype) == 16);
    static_assert(sizeof(ncclEpHandleState_t) == 20);
}

void ncclEpDispatchConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpDispatchConfig_t, size) == 0);
    static_assert(offsetof(ncclEpDispatchConfig_t, magic) == 4);
    static_assert(offsetof(ncclEpDispatchConfig_t, send_only) == 8);
    static_assert(offsetof(ncclEpDispatchConfig_t, round_scales) == 12);
    static_assert(offsetof(ncclEpDispatchConfig_t, pass_direction) == 16);
    static_assert(
        offsetof(ncclEpDispatchConfig_t, quant_recipe) == 20);
    static_assert(offsetof(ncclEpDispatchConfig_t, scale_layout) == 24);
    static_assert(sizeof(ncclEpDispatchConfig_t) == 28);
}

void ncclEpCombineConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpCombineConfig_t, size) == 0);
    static_assert(offsetof(ncclEpCombineConfig_t, magic) == 4);
    static_assert(offsetof(ncclEpCombineConfig_t, send_only) == 8);
    static_assert(offsetof(ncclEpCombineConfig_t, pass_direction) == 12);
    static_assert(
        offsetof(ncclEpCombineConfig_t, quant_recipe) == 16);
    static_assert(sizeof(ncclEpCombineConfig_t) == 20);
}

void ncclEpCompleteConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpCompleteConfig_t, size) == 0);
    static_assert(offsetof(ncclEpCompleteConfig_t, magic) == 4);
    static_assert(sizeof(ncclEpCompleteConfig_t) == 8);
}

class PublicStructAbiRuntimeTest : public EpTestBase {};

static ncclEpGroupConfig_t htFlowTestConfig() {
    ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
    config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    config.num_experts = kNumExperts;
    config.num_topk = kTopK;
    config.max_dispatch_tokens_per_rank = kNumTokens;
    config.max_recv_tokens_per_rank = kMaxRecvSlots;
    config.max_token_bytes = kHidden * sizeof(nv_bfloat16);
    return config;
}

template <typename T>
static void expectPublicStructPrefix(
    const T& value,
    unsigned int expected_size,
    unsigned int expected_magic) {
    EXPECT_EQ(value.size, expected_size);
    EXPECT_EQ(value.magic, expected_magic);
}

TEST(PublicStructAbiTest, InitializersPopulatePrefix) {
    ncclEpTensor_t tensor = NCCL_EP_TENSOR_INIT;
    expectPublicStructPrefix(tensor, NCCL_EP_TENSOR_SIZE, NCCL_EP_TENSOR_MAGIC);

    ncclEpTensorAllocConfig_t tensor_alloc = NCCL_EP_TENSOR_ALLOC_CONFIG_INIT;
    ncclEpGroupConfig_t group = NCCL_EP_GROUP_CONFIG_INIT;
    ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
    ncclEpDispatchInputs_t dispatch_inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dispatch_outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpCombineInputs_t combine_inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t combine_outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpHandleConfig_t handle = NCCL_EP_HANDLE_CONFIG_INIT;
    ncclEpHandleState_t handle_state = NCCL_EP_HANDLE_STATE_INIT;
    ncclEpDispatchConfig_t dispatch = NCCL_EP_DISPATCH_CONFIG_INIT;
    ncclEpCombineConfig_t combine = NCCL_EP_COMBINE_CONFIG_INIT;
    ncclEpCompleteConfig_t complete = NCCL_EP_COMPLETE_CONFIG_INIT;

    expectPublicStructPrefix(
        tensor_alloc, NCCL_EP_TENSOR_ALLOC_CONFIG_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(group, NCCL_EP_GROUP_CONFIG_SIZE, NCCL_EP_MAGIC);
    EXPECT_EQ(group.version, NCCL_EP_API_VERSION);
    expectPublicStructPrefix(layout, NCCL_EP_LAYOUT_INFO_SIZE, NCCL_EP_MAGIC);
    EXPECT_EQ(group.dispatch_num_sms, NCCL_EP_AUTO);
    EXPECT_EQ(group.combine_num_sms, NCCL_EP_AUTO);
    EXPECT_EQ(group.ht_flow, NCCL_EP_HT_FLOW_AUTO);
    EXPECT_EQ(group.meta_exchange_policy, NCCL_EP_META_EXCHANGE_AUTO);
    EXPECT_EQ(group.num_attn_ranks, 0u);
    expectPublicStructPrefix(
        dispatch_inputs, NCCL_EP_DISPATCH_INPUTS_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        dispatch_outputs, NCCL_EP_DISPATCH_OUTPUTS_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        combine_inputs, NCCL_EP_COMBINE_INPUTS_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        combine_outputs, NCCL_EP_COMBINE_OUTPUTS_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(handle, NCCL_EP_HANDLE_CONFIG_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        handle_state, NCCL_EP_HANDLE_STATE_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        dispatch, NCCL_EP_DISPATCH_CONFIG_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        combine, NCCL_EP_COMBINE_CONFIG_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(
        complete, NCCL_EP_COMPLETE_CONFIG_SIZE, NCCL_EP_MAGIC);
}

TEST(PublicStructAbiTest, TensorAllocConfigSizeCompatibility) {
    // Simulate a future caller: the known V1 prefix reports a larger object
    // and carries an unknown tail. The library must accept and ignore it.
    struct FutureTensorAllocConfig {
        ncclEpTensorAllocConfig_t v1;
        uint64_t future_field;
    } future = {NCCL_EP_TENSOR_ALLOC_CONFIG_INIT, 0x12345678u};
    future.v1.size = sizeof(future);

    const size_t dims[1] = {1};
    ncclEpTensor_t* allocated = nullptr;
    EXPECT_EQ(
        ncclEpTensorAlloc(&allocated, 1, ncclUint8, dims, &future.v1),
        ncclSuccess);
    ASSERT_NE(allocated, nullptr);
    EXPECT_EQ(ncclEpTensorDestroy(allocated), ncclSuccess);

    ncclEpTensorAllocConfig_t too_small = NCCL_EP_TENSOR_ALLOC_CONFIG_INIT;
    too_small.size = NCCL_EP_TENSOR_ALLOC_CONFIG_V1_SIZE - 1;
    allocated = nullptr;
    EXPECT_EQ(
        ncclEpTensorAlloc(&allocated, 1, ncclUint8, dims, &too_small),
        ncclInvalidArgument);
    EXPECT_EQ(allocated, nullptr);

    ncclEpTensorAllocConfig_t wrong_magic = NCCL_EP_TENSOR_ALLOC_CONFIG_INIT;
    wrong_magic.magic ^= 1u;
    EXPECT_EQ(
        ncclEpTensorAlloc(&allocated, 1, ncclUint8, dims, &wrong_magic),
        ncclInvalidArgument);
    EXPECT_EQ(allocated, nullptr);
}

TEST_F(PublicStructAbiRuntimeTest, HtFlowRejectsInvalidAndLlSelections) {
    ncclEpGroup_t group = nullptr;
    ncclEpGroupConfig_t config = htFlowTestConfig();
    config.ht_flow = static_cast<ncclEpHtFlow_t>(NCCL_EP_HT_FLOW_PULL_PUSH + 1);
    EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclInvalidArgument);
    EXPECT_EQ(group, nullptr);

    config.ht_flow = NCCL_EP_HT_FLOW_AUTO;
    config.meta_exchange_policy = static_cast<ncclEpMetaExchangePolicy_t>(
        NCCL_EP_META_EXCHANGE_FORCE_SCAN + 1);
    EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclInvalidArgument);
    EXPECT_EQ(group, nullptr);

    config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    config.meta_exchange_policy = NCCL_EP_META_EXCHANGE_AUTO;
    config.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;
    EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclInvalidUsage);
    EXPECT_EQ(group, nullptr);

    config.ht_flow = NCCL_EP_HT_FLOW_AUTO;
    config.meta_exchange_policy = NCCL_EP_META_EXCHANGE_FORCE_SCAN;
    EXPECT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclInvalidUsage);
    EXPECT_EQ(group, nullptr);
}

TEST_F(PublicStructAbiRuntimeTest, HtDefaultsAndOverridesApplyPerHandleLayout) {
    ScopedEnvVar comm_env("NCCL_EP_COMM_SMS", nullptr);
    ScopedEnvVar dispatch_env("NCCL_EP_DISPATCH_SMS", nullptr);
    ScopedEnvVar combine_env("NCCL_EP_COMBINE_SMS", nullptr);
    ScopedEnvVar chunk_env("NCCL_EP_TOKENS_PER_CHUNK", nullptr);
    ScopedEnvVar dispatch_chunk_env("NCCL_EP_DISPATCH_TOKENS_PER_CHUNK", nullptr);
    int device = 0;
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
    const bool rubin = prop.major * 10 + prop.minor == 107;
    const unsigned int flat_default = 16u;
    const unsigned int em_default = rubin ? 32u : 24u;
    struct SmCase {
        unsigned int shared_config, dispatch_config, combine_config;
        const char* shared_env;
        const char* dispatch_env;
        const char* combine_env;
        unsigned int expected_dispatch, expected_combine;
        unsigned int max_dispatch_tokens, flat_combine_default;
    };
    // Zero expected values select the architecture/layout default.
    const SmCase cases[] = {
        {0, 0, 0, nullptr, nullptr, nullptr, 0, 0, kNumTokens, 16},
        {20, 0, 0, nullptr, nullptr, nullptr, 20, 20, kNumTokens, 16},
        {0, 18, 0, nullptr, nullptr, nullptr, 18, 0, kNumTokens, 16},
        {0, 0, 0, "22", nullptr, nullptr, 22, 22, kNumTokens, 16},
        {0, 0, 0, nullptr, nullptr, "12", 0, 12, kNumTokens, 16},
        {0, 0, 0, "0", "0", nullptr, 0, 0, kNumTokens, 16},
        {20, 18, 19, "22", "12", nullptr, 12, 22, kNumTokens, 16},
        {0, 0, 0, nullptr, nullptr, nullptr, 0, 0, 8191, 16},
        {0, 0, 0, nullptr, nullptr, nullptr, 0, 0, 8192, 32},
        {20, 0, 0, nullptr, nullptr, nullptr, 20, 20, 8192, 32},
        {0, 18, 0, nullptr, nullptr, nullptr, 18, 0, 8192, 32},
        {0, 0, 0, nullptr, nullptr, "12", 0, 12, 8192, 32},
    };
    for (const auto& test_case : cases) {
        ScopedEnvVar shared_override("NCCL_EP_COMM_SMS", test_case.shared_env);
        ScopedEnvVar dispatch_override("NCCL_EP_DISPATCH_SMS", test_case.dispatch_env);
        ScopedEnvVar combine_override("NCCL_EP_COMBINE_SMS", test_case.combine_env);
        ncclEpGroupConfig_t config = htFlowTestConfig();
        config.max_num_sms = test_case.shared_config;
        config.max_dispatch_tokens_per_rank = test_case.max_dispatch_tokens;
        config.max_recv_tokens_per_rank =
            std::max(config.max_recv_tokens_per_rank, test_case.max_dispatch_tokens);
        config.dispatch_num_sms = test_case.dispatch_config;
        config.combine_num_sms = test_case.combine_config;
        ncclEpGroup_t group = nullptr;
        ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);
        for (auto layout : {NCCL_EP_LAYOUT_FLAT, NCCL_EP_LAYOUT_EXPERT_MAJOR}) {
            SCOPED_TRACE(::testing::Message() << "layout=" << layout
                         << " shared=" << test_case.shared_config << " dispatch=" << test_case.dispatch_config);
            const unsigned int fallback = layout == NCCL_EP_LAYOUT_FLAT ? flat_default : em_default;
            const unsigned int expected_d = test_case.expected_dispatch ? test_case.expected_dispatch : fallback;
            const unsigned int combine_fallback = layout == NCCL_EP_LAYOUT_FLAT
                ? test_case.flat_combine_default : em_default;
            const unsigned int expected_c = test_case.expected_combine ? test_case.expected_combine : combine_fallback;
            ncclEpHandle_t handle = nullptr;
            ASSERT_EQ(ncclEpCreateHandle(&handle, group, layout, topk_idx_, nullptr, nullptr, g_stream), ncclSuccess);
            ASSERT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
            unsigned int dispatch_sms = 0, combine_sms = 0;
            ncclEpHandle_test_getSmBudgets(handle, &dispatch_sms, &combine_sms);
            EXPECT_EQ(dispatch_sms, expected_d);
            EXPECT_EQ(combine_sms, expected_c);
            const int expected_chunk = ncclTeamLsa(g_comm).nRanks == g_nranks
                ? static_cast<int>(nccl_ep::align(4 * expected_d, 32u)) : 64;
            EXPECT_EQ(ncclEpHandle_test_getCombineTokensPerChunk(handle), expected_chunk);
            EXPECT_EQ(ncclEpHandleDestroy(handle), ncclSuccess);
        }
        EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
    }
}

TEST_F(PublicStructAbiRuntimeTest, AutoFlowDefaultsToPullPushForBothLayouts) {
    ScopedEnvVar flow_env("NCCL_EP_HT_PULL_PUSH", nullptr);
    ScopedEnvVar local_dup_env("NCCL_EP_HT_EM_LOCAL_DUP", nullptr);
    ScopedEnvVar nvlink_dup_env("NCCL_EP_HT_EM_NVLINK_DUP", nullptr);
    ncclEpGroupConfig_t config = htFlowTestConfig();
    ncclEpGroup_t group = nullptr;
    ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);
    const bool single_lsa = ncclTeamLsa(g_comm).nRanks == g_nranks;
    EXPECT_EQ(ncclEpGroup_test_getHtFlow(group),
              single_lsa ? NCCL_EP_HT_FLOW_PULL_PUSH : NCCL_EP_HT_FLOW_PUSH_PULL);
    for (auto layout : {NCCL_EP_LAYOUT_FLAT, NCCL_EP_LAYOUT_EXPERT_MAJOR}) {
        ncclEpHandle_t handle = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&handle, group, layout, topk_idx_, nullptr, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        if (handle != nullptr) {
            EXPECT_EQ(ncclEpHandleDestroy(handle), ncclSuccess);
        }
    }
    EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
}

TEST_F(PublicStructAbiRuntimeTest, AutoFlowRespectsKnownTopkLimit) {
    ScopedEnvVar flow_env("NCCL_EP_HT_PULL_PUSH", nullptr);
    ScopedEnvVar local_dup_env("NCCL_EP_HT_EM_LOCAL_DUP", nullptr);
    ScopedEnvVar nvlink_dup_env("NCCL_EP_HT_EM_NVLINK_DUP", nullptr);
    constexpr int topk = 64;
    std::vector<int64_t> ids(kNumTokens * topk);
    for (int token = 0; token < kNumTokens; ++token) {
        for (int k = 0; k < topk; ++k) {
            ids[token * topk + k] = k;
        }
    }
    int64_t* device_ids = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ids, ids.size() * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_ids, ids.data(), ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice), cudaSuccess);
    for (unsigned int bound : {0u, 32u, static_cast<unsigned int>(topk)}) {
        ncclEpGroupConfig_t config = htFlowTestConfig();
        config.num_experts = topk * g_nranks;
        config.num_topk = bound;
        config.max_recv_tokens_per_rank = g_nranks * kNumTokens * topk;
        ncclEpGroup_t group = nullptr;
        ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);
        const bool pull = bound == 32 && ncclTeamLsa(g_comm).nRanks == g_nranks;
        EXPECT_EQ(ncclEpGroup_test_getHtFlow(group),
                  pull ? NCCL_EP_HT_FLOW_PULL_PUSH : NCCL_EP_HT_FLOW_PUSH_PULL);
        ncclEpTensor_t* bounded_indices = nullptr;
        ASSERT_EQ(epTensorCreate(&bounded_indices, 2, ncclInt64, device_ids,
                                 kNumTokens, bound == 32 ? 32 : topk), ncclSuccess);
        ncclEpHandle_t handle = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                    bounded_indices, nullptr, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        if (handle != nullptr) {
            EXPECT_EQ(ncclEpHandleDestroy(handle), ncclSuccess);
        }
        EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
        EXPECT_EQ(ncclEpTensorDestroy(bounded_indices), ncclSuccess);
    }
    EXPECT_EQ(cudaFree(device_ids), cudaSuccess);
}

TEST_F(PublicStructAbiRuntimeTest, AutoFlowHonorsPushPullEnvironmentOverride) {
    ScopedEnvVar flow_env("NCCL_EP_HT_PULL_PUSH", "0");
    ncclEpGroupConfig_t config = htFlowTestConfig();
    ncclEpGroup_t group = nullptr;
    ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);
    EXPECT_EQ(ncclEpGroup_test_getHtFlow(group), NCCL_EP_HT_FLOW_PUSH_PULL);
    EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
}

TEST_F(PublicStructAbiRuntimeTest, ExplicitPushPullOverridesEnvironment) {
    ScopedEnvVar pull_push_env("NCCL_EP_HT_PULL_PUSH", "1");
    ncclEpGroupConfig_t config = htFlowTestConfig();
    config.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;

    ncclEpGroup_t group = nullptr;
    ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);
    EXPECT_EQ(ncclEpGroup_test_getHtFlow(group), NCCL_EP_HT_FLOW_PUSH_PULL);

    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(
        ncclEpCreateHandle(
            &handle, group, NCCL_EP_LAYOUT_FLAT, topk_idx_, nullptr, nullptr, g_stream),
        ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    if (handle != nullptr) EXPECT_EQ(ncclEpHandleDestroy(handle), ncclSuccess);
    EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
}

TEST_F(PublicStructAbiRuntimeTest, ExplicitMetaPolicyOverridesLegacyEnvironment) {
    ScopedEnvVar scan_env("NCCL_EP_HT_AG_SCAN_MODE", "1");
    ScopedEnvVar unfused_env("NCCL_EP_HT_COUNT_UNFUSED", "1");
    ncclEpGroupConfig_t config = htFlowTestConfig();
    config.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;
    config.meta_exchange_policy = NCCL_EP_META_EXCHANGE_REQUIRE_FUSED;

    ncclEpGroup_t group = nullptr;
    ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);

    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(
        ncclEpCreateHandle(
            &handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, nullptr, g_stream),
        ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    if (handle != nullptr) EXPECT_EQ(ncclEpHandleDestroy(handle), ncclSuccess);
    EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
}

TEST_F(PublicStructAbiRuntimeTest, ExplicitPullPushSupportsFlatLayout) {
    ScopedEnvVar pull_push_env("NCCL_EP_HT_PULL_PUSH", "0");
    ncclEpGroupConfig_t config = htFlowTestConfig();
    config.ht_flow = NCCL_EP_HT_FLOW_PULL_PUSH;

    ncclEpGroup_t group = nullptr;
    ASSERT_EQ(ncclEpCreateGroup(&group, g_comm, &config), ncclSuccess);
    EXPECT_EQ(ncclEpGroup_test_getHtFlow(group), NCCL_EP_HT_FLOW_PULL_PUSH);

    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(
        ncclEpCreateHandle(
            &handle, group, NCCL_EP_LAYOUT_FLAT, topk_idx_, nullptr, nullptr, g_stream),
        ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    if (handle != nullptr) {
        EXPECT_EQ(ncclEpHandleDestroy(handle), ncclSuccess);
    }
    EXPECT_EQ(ncclEpGroupDestroy(group), ncclSuccess);
}

TEST_F(
    PublicStructAbiRuntimeTest,
    CreateHandleRejectsUndersizedLayoutBeforeAllocation) {
    ncclEpLayoutInfo_t invalid_layout = NCCL_EP_LAYOUT_INFO_INIT;
    invalid_layout.size = NCCL_EP_LAYOUT_INFO_V1_SIZE - 1;

    ncclEpHandle_t handle = nullptr;
    EXPECT_EQ(
        ncclEpCreateHandle(
            &handle,
            g_ep_group,
            NCCL_EP_LAYOUT_FLAT,
            topk_idx_,
            &invalid_layout,
            nullptr,
            g_stream),
        ncclInvalidArgument);
    EXPECT_EQ(handle, nullptr);
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_public_struct_abi_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
