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
    static_assert(sizeof(ncclEpGroupConfig_t) == 112);
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
    static_assert(sizeof(ncclEpCombineInputs_t) == 24);
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

void ncclEpDispatchConfig_backwards_compat_test() {
    static_assert(offsetof(ncclEpDispatchConfig_t, size) == 0);
    static_assert(offsetof(ncclEpDispatchConfig_t, magic) == 4);
    static_assert(offsetof(ncclEpDispatchConfig_t, send_only) == 8);
    static_assert(offsetof(ncclEpDispatchConfig_t, round_scales) == 12);
    static_assert(offsetof(ncclEpDispatchConfig_t, pass_direction) == 16);
    static_assert(
        offsetof(ncclEpDispatchConfig_t, quant_recipe) == 20);
    static_assert(sizeof(ncclEpDispatchConfig_t) == 24);
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
    ncclEpDispatchConfig_t dispatch = NCCL_EP_DISPATCH_CONFIG_INIT;
    ncclEpCombineConfig_t combine = NCCL_EP_COMBINE_CONFIG_INIT;
    ncclEpCompleteConfig_t complete = NCCL_EP_COMPLETE_CONFIG_INIT;

    expectPublicStructPrefix(
        tensor_alloc, NCCL_EP_TENSOR_ALLOC_CONFIG_SIZE, NCCL_EP_MAGIC);
    expectPublicStructPrefix(group, NCCL_EP_GROUP_CONFIG_SIZE, NCCL_EP_MAGIC);
    EXPECT_EQ(group.version, NCCL_EP_API_VERSION);
    expectPublicStructPrefix(layout, NCCL_EP_LAYOUT_INFO_SIZE, NCCL_EP_MAGIC);
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
