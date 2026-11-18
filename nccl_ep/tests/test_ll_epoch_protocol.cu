/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more information.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <limits>

#include "common.hpp"
#include "device/ll/ll_hier_delivery.cuh"

namespace {

using nccl_ep::LowLatencyEpochState;

struct EpochObservation {
    unsigned int bank;
    LowLatencyEpochState state;
};

// PR tier: one GPU, bounded CTA-local stress; no communicator or timing assertion.
__global__ void reuseHierControl(unsigned int* failures) {
  __shared__ uint64_t scratch;
  for (unsigned int generation = 0; generation < 512; ++generation) {
    if (threadIdx.x >= 32 && (generation & 1) == 0) {
      __nanosleep(100);
    }
    const uint64_t expected = (uint64_t(generation) << 32) | (generation & 1);
    const uint64_t observed = nccl_ep::ll::hierBroadcastControl(ncclCoopCta(), scratch, expected);
    if (threadIdx.x >= 32 && (generation & 1) != 0) {
      __nanosleep(100);
    }
    if (observed != expected) {
      atomicAdd(failures, 1U);
    }
  }
}

__global__ void fullOperation(LowLatencyEpochState* state, EpochObservation* observation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    constexpr int phases = LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE;
    observation->bank = nccl_ep::ll::selectLowLatencyEpoch(state, phases, 0, 0) & 1U;
    nccl_ep::ll::completeFullLowLatencyEpoch(state, phases, 0, 0);
    observation->state = *state;
}

__global__ void sendOnly(LowLatencyEpochState* state, EpochObservation* observation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    observation->bank = nccl_ep::ll::selectLowLatencyEpoch(state, LOW_LATENCY_SEND_PHASE, 0, 0) & 1U;
    observation->state = *state;
}

__global__ void recvOnly(LowLatencyEpochState* state, EpochObservation* observation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    observation->bank = nccl_ep::ll::selectLowLatencyEpoch(state, LOW_LATENCY_RECV_PHASE, 0, 0) & 1U;
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

TEST_F(LlEpochProtocolTest, SharedEpochCrossesThe32BitBoundary) {
    const uint64_t last32BitEpoch = std::numeric_limits<uint32_t>::max();
    const LowLatencyEpochState initial{last32BitEpoch, 0, 0};
    ASSERT_EQ(cudaMemcpy(state_, &initial, sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);
    fullOperation<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EpochObservation result{};
    ASSERT_EQ(cudaMemcpy(&result, observation_, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result.bank, 1U);
    EXPECT_EQ(result.state.epoch, last32BitEpoch + 1);
    sendOnly<<<1, 1>>>(state_, observation_);
    recvOnly<<<1, 1>>>(state_, observation_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&result, observation_, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result.bank, 0U);
    EXPECT_EQ(result.state.epoch, last32BitEpoch + 2);
    EXPECT_EQ(result.state.pending_epoch, last32BitEpoch + 1);
    EXPECT_EQ(result.state.send_in_flight, 0U);
}

TEST_F(LlEpochProtocolTest, HierControlReadersFinishBeforeReuse) {
  auto* failures = reinterpret_cast<unsigned int*>(state_);
  ASSERT_EQ(cudaMemset(failures, 0, sizeof(*failures)), cudaSuccess);
  reuseHierControl<<<32, 128>>>(failures);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  unsigned int observed = 0;
  ASSERT_EQ(cudaMemcpy(&observed, failures, sizeof(observed), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(observed, 0U);
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
    const LowLatencyEpochState initial{std::numeric_limits<uint64_t>::max(), 0, 0};
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
