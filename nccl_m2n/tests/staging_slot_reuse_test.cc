/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include <cuda_runtime.h>
#include <nccl.h>

#include "reshard_internal.h"

namespace {

#define ASSERT_NCCL(call) ASSERT_EQ(ncclSuccess, (call))
#define ASSERT_CUDA(call) ASSERT_EQ(cudaSuccess, (call))

constexpr size_t kSlotBytes = 4096;

void CUDART_CB waitForSlotRelease(void* data) {
  auto* release = static_cast<std::atomic<bool>*>(data);
  while (!release->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

class StagingSlotReuseTest : public ::testing::Test {
protected:
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  int savedBucketCount = 0;
  ReshardStagingBucketCfg savedBuckets[kMaxStagingBuckets] = {};

  void SetUp() override {
    ASSERT_CUDA(cudaSetDevice(0));
    int devices[1] = {0};
    ASSERT_NCCL(ncclCommInitAll(&comm, 1, devices));
    ASSERT_CUDA(cudaStreamCreate(&stream));

    transposeBufferFinalize();
    reshardClearResourceQuarantine();
    savedBucketCount = gReshardStagingBucketCount;
    for (int i = 0; i < kMaxStagingBuckets; i++) {
      savedBuckets[i] = gReshardStagingBuckets[i];
      gReshardStagingBuckets[i] = {};
    }
    gReshardStagingBucketCount = 0;
  }

  void TearDown() override {
    transposeBufferFinalize();
    reshardClearResourceQuarantine();
    gReshardStagingBucketCount = savedBucketCount;
    for (int i = 0; i < kMaxStagingBuckets; i++) {
      gReshardStagingBuckets[i] = savedBuckets[i];
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (comm != nullptr) ncclCommDestroy(comm);
  }

  void enableSingleSlotBucket() {
    gReshardStagingBuckets[0] = {kSlotBytes, 1};
    gReshardStagingBucketCount = 1;
  }
};

TEST_F(StagingSlotReuseTest, AssignsStableRoundRobinLanesAcrossCommunicators) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBucketCount = 1;

  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ncclComm_t comm4 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm4, 1, devices));

  ASSERT_NCCL(ensureTransposeBuffer(comm, kSlotBytes, stream));
  void* slot0 = getTransposeBuffer(comm);
  ASSERT_NCCL(transposeBufferRecordEvent(comm, stream));
  ASSERT_NCCL(ensureTransposeBuffer(comm2, kSlotBytes, stream));
  void* slot1 = getTransposeBuffer(comm2);
  ASSERT_NCCL(transposeBufferRecordEvent(comm2, stream));
  ASSERT_NE(slot0, slot1);

  ASSERT_NCCL(ensureTransposeBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(slot0, getTransposeBuffer(comm3));
  ASSERT_NCCL(transposeBufferRecordEvent(comm3, stream));
  ASSERT_NCCL(ensureTransposeBuffer(comm4, kSlotBytes, stream));
  EXPECT_EQ(slot1, getTransposeBuffer(comm4));
  ASSERT_NCCL(transposeBufferRecordEvent(comm4, stream));

  ASSERT_NCCL(ensureTransposeBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(slot0, getTransposeBuffer(comm3));
  ASSERT_NCCL(transposeBufferRecordEvent(comm3, stream));

  transposeBufferFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
  ASSERT_NCCL(ncclCommDestroy(comm4));
}

TEST_F(StagingSlotReuseTest, RejectsOverlappingUsersOfOnePhysicalLane) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));

  std::atomic<bool> slotReady{false};
  std::atomic<bool> releaseOwner{false};
  cudaError_t ownerCudaResult = cudaSuccess;
  ncclResult_t ownerAcquireResult = ncclSuccess;
  ncclResult_t ownerRecordResult = ncclSuccess;
  std::thread owner([&] {
    ownerCudaResult = cudaSetDevice(0);
    if (ownerCudaResult != cudaSuccess) return;
    M2nApiLock apiLock;
    ownerAcquireResult = ensureTransposeBuffer(comm, kSlotBytes, stream);
    slotReady.store(true, std::memory_order_release);
    {
      M2nApiUnlock apiUnlock;
      while (!releaseOwner.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    if (ownerAcquireResult == ncclSuccess) {
      ownerRecordResult = transposeBufferRecordEvent(comm, stream);
    }
  });

  while (!slotReady.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ncclResult_t overlappingResult = ncclSuccess;
  {
    M2nApiLock apiLock;
    overlappingResult = ensureTransposeBuffer(comm2, kSlotBytes, stream);
  }
  releaseOwner.store(true, std::memory_order_release);
  owner.join();

  EXPECT_EQ(cudaSuccess, ownerCudaResult);
  EXPECT_EQ(ncclSuccess, ownerAcquireResult);
  EXPECT_EQ(ncclInvalidUsage, overlappingResult);
  EXPECT_EQ(ncclSuccess, ownerRecordResult);
  ASSERT_NCCL(ensureTransposeBuffer(comm2, kSlotBytes, stream));
  ASSERT_NCCL(transposeBufferRecordEvent(comm2, stream));

  transposeBufferFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(StagingSlotReuseTest, WaitsForPriorCommunicatorOnAnotherStream) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));

  ASSERT_NCCL(ensureTransposeBuffer(comm, kSlotBytes, nullptr));
  cudaStream_t nonblocking = nullptr;
  cudaEvent_t marker = nullptr;
  ASSERT_CUDA(cudaStreamCreateWithFlags(&nonblocking, cudaStreamNonBlocking));
  ASSERT_CUDA(cudaEventCreateWithFlags(&marker, cudaEventDisableTiming));

  std::atomic<bool> release{false};
  ASSERT_CUDA(cudaLaunchHostFunc(nullptr, waitForSlotRelease, &release));
  ASSERT_NCCL(transposeBufferRecordEvent(comm, nullptr));
  ASSERT_NCCL(ensureTransposeBuffer(comm2, kSlotBytes, nonblocking));
  ASSERT_CUDA(cudaEventRecord(marker, nonblocking));
  ASSERT_NCCL(transposeBufferRecordEvent(comm2, nonblocking));

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(cudaErrorNotReady, cudaEventQuery(marker));
  release.store(true, std::memory_order_release);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(nonblocking));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));

  EXPECT_EQ(cudaSuccess, cudaEventDestroy(marker));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(nonblocking));
  transposeBufferFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(StagingSlotReuseTest, KeepsHostRmaWarmupStatePerCommunicator) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));

  ASSERT_NCCL(ensureTransposeBuffer(comm, kSlotBytes, stream));
  void* sharedSlot = getTransposeBuffer(comm);
  ASSERT_NCCL(transposeBufferRecordEvent(comm, stream));
  ASSERT_NCCL(ensureTransposeBuffer(comm2, kSlotBytes, stream));
  EXPECT_EQ(sharedSlot, getTransposeBuffer(comm2));
  ASSERT_NCCL(transposeBufferRecordEvent(comm2, stream));

  ASSERT_NCCL(setPackWindowRmaWarmed(comm, true));
  ASSERT_NCCL(setPackWindowRmaWarmed(comm2, false));
  bool warmed = false;
  ASSERT_NCCL(getPackWindowRmaWarmed(comm, &warmed));
  EXPECT_TRUE(warmed);
  ASSERT_NCCL(getPackWindowRmaWarmed(comm2, &warmed));
  EXPECT_FALSE(warmed);

  transposeBufferFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(StagingSlotReuseTest, NewMappingsSkipPoisonedLanes) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBucketCount = 1;
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices));

  ASSERT_NCCL(ensureTransposeBuffer(comm, kSlotBytes, stream));
  reshardFailNextCompletionEventRecordForTest(/*bFailStreamSynchronize=*/true);
  EXPECT_EQ(ncclSystemError, transposeBufferRecordEvent(comm, stream));
  EXPECT_EQ(ncclUnhandledCudaError, ensureTransposeBuffer(comm, kSlotBytes, stream));

  ASSERT_NCCL(ensureTransposeBuffer(comm2, kSlotBytes, stream));
  void* healthySlot = getTransposeBuffer(comm2);
  ASSERT_NCCL(transposeBufferRecordEvent(comm2, stream));
  ASSERT_NCCL(ensureTransposeBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(healthySlot, getTransposeBuffer(comm3));
  ASSERT_NCCL(transposeBufferRecordEvent(comm3, stream));

  transposeBufferFinalize();
  reshardClearResourceQuarantine();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
}

} // namespace
