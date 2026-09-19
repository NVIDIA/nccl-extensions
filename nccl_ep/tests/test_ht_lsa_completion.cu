// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "device/lsa_completion.cuh"

namespace {
constexpr int kRanks = 2;
constexpr int kRounds = 256;

__global__ void exchange(uint32_t* slots, uint32_t* payload, uint32_t* errors,
                         int rank, uint32_t initial) {
    uint32_t failures = 0;
    for (int round = 0; round < kRounds; ++round) {
        const uint32_t epoch = initial + kRanks * (round + 1);
        // Alternate the late publisher; keep each round's payload distinct so a
        // fast peer can publish its next epoch before the slow peer finishes polling.
        if (rank == round % kRanks) {
            const auto start = clock64();
            while (clock64() - start < 10000) {}
        }
        payload[round * kRanks + rank] = round * kRanks + rank + 1;
        nccl_ep::publish_lsa_completion(slots + rank, epoch);
        nccl_ep::wait_lsa_completion(slots, epoch, kRanks);
        for (int peer = 0; peer < kRanks; ++peer)
            failures += payload[round * kRanks + peer] != uint32_t(round * kRanks + peer + 1);
    }
    errors[rank] = failures;
}

class LsaCompletionTest : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        if (count < kRanks) GTEST_SKIP() << "Requires two CUDA devices";
        int accessible = 0;
        ASSERT_EQ(cudaDeviceCanAccessPeer(&accessible, 1, 0), cudaSuccess);
        if (!accessible) GTEST_SKIP() << "Requires peer writes from device 1 to device 0";
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        const cudaError_t result = cudaDeviceEnablePeerAccess(0, 0);
        ASSERT_TRUE(result == cudaSuccess || result == cudaErrorPeerAccessAlreadyEnabled);
        enabled_peer_ = result == cudaSuccess;
        if (!enabled_peer_) cudaGetLastError();
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&storage_, (kRanks * (kRounds + 2)) * sizeof(uint32_t)), cudaSuccess);
        ASSERT_EQ(cudaMemset(storage_, 0, (kRanks * (kRounds + 2)) * sizeof(uint32_t)), cudaSuccess);
        const uint32_t initial[kRanks] = {GetParam(), GetParam()};
        ASSERT_EQ(cudaMemcpy(storage_, initial, sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);
        for (int rank = 0; rank < kRanks; ++rank) {
            ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
            ASSERT_EQ(cudaStreamCreate(&streams_[rank]), cudaSuccess);
        }
    }

    void TearDown() override {
        for (int rank = 0; rank < kRanks; ++rank) {
            cudaSetDevice(rank);
            if (streams_[rank]) cudaStreamDestroy(streams_[rank]);
        }
        if (enabled_peer_) cudaDeviceDisablePeerAccess(0);
        cudaSetDevice(0);
        if (storage_) cudaFree(storage_);
    }

    uint32_t* storage_ = nullptr;
    cudaStream_t streams_[kRanks] = {};
    bool enabled_peer_ = false;
};

TEST_P(LsaCompletionTest, PublishesPayloadAcrossPeerWritesAndRepeatedEpochs) {
    auto* payload = storage_ + kRanks;
    auto* errors = payload + kRounds * kRanks;
    for (int rank = 0; rank < kRanks; ++rank) {
        ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
        exchange<<<1, 1, 0, streams_[rank]>>>(storage_, payload, errors, rank, GetParam());
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }
    for (int rank = 0; rank < kRanks; ++rank) {
        ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(streams_[rank]), cudaSuccess);
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    uint32_t failures[kRanks];
    ASSERT_EQ(cudaMemcpy(failures, errors, sizeof(failures), cudaMemcpyDeviceToHost), cudaSuccess);
    for (auto failures_for_rank : failures) EXPECT_EQ(failures_for_rank, 0u);
}

INSTANTIATE_TEST_SUITE_P(Epochs, LsaCompletionTest,
                        ::testing::Values(0u, 0x7ffffff0u, 0xfffffff0u));
} // namespace
