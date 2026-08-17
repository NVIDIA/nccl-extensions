/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more information.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <limits>

#include "include/common.hpp"

namespace {

using nccl_ep::LowLatencyEpochState;

struct EpochObservation {
    unsigned int bank;
    LowLatencyEpochState state;
};

__global__ void fullOperation(LowLatencyEpochState* state, EpochObservation* observation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    observation->bank = state->epoch & 1u;
    ++state->epoch;
    observation->state = *state;
}

__global__ void sendOnly(LowLatencyEpochState* state, EpochObservation* observation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    observation->bank = state->epoch & 1u;
    state->pending_epoch = state->epoch;
    state->send_in_flight = 1;
    observation->state = *state;
}

__global__ void recvOnly(LowLatencyEpochState* state, EpochObservation* observation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    observation->bank = state->pending_epoch & 1u;
    ++state->epoch;
    state->send_in_flight = 0;
    observation->state = *state;
}

class LlEpochProtocolTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount = 0;
        ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
        if (deviceCount == 0) GTEST_SKIP() << "No CUDA device available";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&state_, sizeof(*state_)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&observation_, sizeof(*observation_)), cudaSuccess);
    }

    void TearDown() override {
        if (observation_) cudaFree(observation_);
        if (state_) cudaFree(state_);
    }

    LowLatencyEpochState* state_ = nullptr;
    EpochObservation* observation_ = nullptr;
};

TEST_F(LlEpochProtocolTest, FullOperationAdvancesTheLiveEpoch) {
    const LowLatencyEpochState initial{7, 0, 0};
    ASSERT_EQ(cudaMemcpy(state_, &initial, sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);
    fullOperation<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation result{};
    ASSERT_EQ(cudaMemcpy(&result, observation_, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result.bank, 1u);
    EXPECT_EQ(result.state.epoch, 8u);
    EXPECT_EQ(result.state.send_in_flight, 0u);
}

TEST_F(LlEpochProtocolTest, SplitOperationKeepsOneBankUntilRecvCompletes) {
    const LowLatencyEpochState initial{10, 0, 0};
    ASSERT_EQ(cudaMemcpy(state_, &initial, sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);
    sendOnly<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation sent{};
    ASSERT_EQ(cudaMemcpy(&sent, observation_, sizeof(sent), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(sent.bank, 0u);
    EXPECT_EQ(sent.state.epoch, 10u);
    EXPECT_EQ(sent.state.pending_epoch, 10u);
    EXPECT_EQ(sent.state.send_in_flight, 1u);

    ASSERT_EQ(cudaMemcpy(state_, &sent.state, sizeof(sent.state), cudaMemcpyHostToDevice), cudaSuccess);
    recvOnly<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation received{};
    ASSERT_EQ(cudaMemcpy(&received, observation_, sizeof(received), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(received.bank, sent.bank);
    EXPECT_EQ(received.state.epoch, 11u);
    EXPECT_EQ(received.state.send_in_flight, 0u);
}

TEST_F(LlEpochProtocolTest, EpochRolloverPreservesDoubleBufferParity) {
    const LowLatencyEpochState initial{std::numeric_limits<unsigned int>::max(), 0, 0};
    ASSERT_EQ(cudaMemcpy(state_, &initial, sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);
    fullOperation<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation full{};
    ASSERT_EQ(cudaMemcpy(&full, observation_, sizeof(full), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(full.bank, 1u);
    EXPECT_EQ(full.state.epoch, 0u);

    ASSERT_EQ(cudaMemcpy(state_, &initial, sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);
    sendOnly<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation sent{};
    ASSERT_EQ(cudaMemcpy(&sent, observation_, sizeof(sent), cudaMemcpyDeviceToHost), cudaSuccess);
    recvOnly<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation received{};
    ASSERT_EQ(cudaMemcpy(&received, observation_, sizeof(received), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(sent.bank, 1u);
    EXPECT_EQ(received.bank, sent.bank);
    EXPECT_EQ(received.state.epoch, 0u);
}

} // namespace
