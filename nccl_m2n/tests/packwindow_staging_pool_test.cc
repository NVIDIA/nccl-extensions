/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/* Resource, configuration, and ordering coverage for the bounded
 * PACKWINDOW staging pool in src/packwindow_staging.cc. */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include <nccl.h>

#include "reshard_internal.h"

namespace {

#define ASSERT_NCCL(call) ASSERT_EQ(ncclSuccess, (call))
#define ASSERT_CUDA(call) ASSERT_EQ(cudaSuccess, (call))

constexpr size_t kSlotBytes = 4096;
constexpr size_t kUnsatisfiableBytes = static_cast<size_t>(1) << 44;

void CUDART_CB waitForSlotRelease(void* data) {
  auto* release = static_cast<std::atomic<bool>*>(data);
  while (!release->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

class PackWindowStagingPoolTest : public ::testing::Test {
protected:
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  int savedBucketCount = 0;
  bool savedImplicitDefault = true;
  bool hadSavedProfile = false;
  std::string savedProfile;
  ReshardStagingBucketCfg savedBuckets[kMaxStagingBuckets] = {};

  void SetUp() override {
    ASSERT_CUDA(cudaSetDevice(0));
    int devices[1] = {0};
    ASSERT_NCCL(ncclCommInitAll(&comm, 1, devices));
    ASSERT_CUDA(cudaStreamCreate(&stream));

    packWindowStagingFinalize();
    reshardClearResourceQuarantine();
    savedBucketCount = gReshardStagingBucketCount;
    savedImplicitDefault = gReshardStagingBucketsImplicitDefault;
    const char* profile = getenv("NCCL_RESHARD_PACK_BUFFSIZES");
    if (profile != nullptr) {
      hadSavedProfile = true;
      savedProfile = profile;
    }
    unsetenv("NCCL_RESHARD_PACK_BUFFSIZES");
    for (int i = 0; i < kMaxStagingBuckets; i++) {
      savedBuckets[i] = gReshardStagingBuckets[i];
      gReshardStagingBuckets[i] = {};
    }
    gReshardStagingBuckets[0] = {kSlotBytes, 1};
    gReshardStagingBucketCount = 1;
    gReshardStagingBucketsImplicitDefault = false;
  }

  void TearDown() override {
    packWindowStagingFinalize();
    reshardClearResourceQuarantine();
    gReshardStagingBucketCount = savedBucketCount;
    gReshardStagingBucketsImplicitDefault = savedImplicitDefault;
    for (int i = 0; i < kMaxStagingBuckets; i++) {
      gReshardStagingBuckets[i] = savedBuckets[i];
    }
    if (hadSavedProfile) setenv("NCCL_RESHARD_PACK_BUFFSIZES", savedProfile.c_str(), 1);
    else unsetenv("NCCL_RESHARD_PACK_BUFFSIZES");
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (comm != nullptr) ncclCommDestroy(comm);
  }

  void enableSingleSlotBucket() {
    gReshardStagingBuckets[0] = {kSlotBytes, 1};
    gReshardStagingBucketCount = 1;
    gReshardStagingBucketsImplicitDefault = false;
  }
};

TEST_F(PackWindowStagingPoolTest, ParsesConcisePackBufferProfile) {
  ASSERT_EQ(0, setenv("NCCL_RESHARD_PACK_BUFFSIZES", "8k:2,1m", 1));
  applyReshardEnv();
  ASSERT_EQ(2, gReshardStagingBucketCount);
  EXPECT_EQ(8U * 1024U, gReshardStagingBuckets[0].size);
  EXPECT_EQ(2, gReshardStagingBuckets[0].numSlots);
  EXPECT_EQ(1024U * 1024U, gReshardStagingBuckets[1].size);
  EXPECT_EQ(1, gReshardStagingBuckets[1].numSlots);
  EXPECT_FALSE(gReshardStagingBucketsImplicitDefault);
}

TEST_F(PackWindowStagingPoolTest, RejectsProfileBelowMinimumUsableSize) {
  ASSERT_EQ(0, setenv("NCCL_RESHARD_PACK_BUFFSIZES", "1k", 1));
  applyReshardEnv();
  ASSERT_EQ(1, gReshardStagingBucketCount);
  EXPECT_EQ(kDefaultStagingBucketBytes, gReshardStagingBuckets[0].size);
  EXPECT_EQ(kDefaultStagingBucketSlots, gReshardStagingBuckets[0].numSlots);
  EXPECT_TRUE(gReshardStagingBucketsImplicitDefault);
}

TEST_F(PackWindowStagingPoolTest, AllocatesOnlySelectedSlotsLazily) {
  gReshardStagingBuckets[0] = {kSlotBytes, 4};
  gReshardStagingBucketCount = 1;
  gReshardStagingBucketsImplicitDefault = false;

  EXPECT_EQ(0, packWindowStagingAllocationCountForTest());
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_EQ(1, packWindowStagingAllocationCountForTest());
  ASSERT_NCCL(packWindowStagingRecordEvent(comm, stream));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_EQ(1, packWindowStagingAllocationCountForTest());
  ASSERT_NCCL(packWindowStagingRecordEvent(comm, stream));
}

TEST_F(PackWindowStagingPoolTest, AllocationFailureDoesNotCommitMapping) {
  gReshardStagingBuckets[0] = {kSlotBytes, 1};
  gReshardStagingBuckets[1] = {kUnsatisfiableBytes, 1};
  gReshardStagingBucketCount = 2;
  gReshardStagingBucketsImplicitDefault = false;

  EXPECT_NE(ncclSuccess, ensurePackWindowStagingBuffer(comm, kUnsatisfiableBytes, stream));
  EXPECT_EQ(nullptr, getPackWindowStagingBuffer(comm));
  EXPECT_EQ(0U, getPackWindowStagingCapacity(comm));
  EXPECT_EQ(0, packWindowStagingAllocationCountForTest());

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_NE(nullptr, getPackWindowStagingBuffer(comm));
  EXPECT_EQ(kSlotBytes, getPackWindowStagingCapacity(comm));
  EXPECT_EQ(1, packWindowStagingAllocationCountForTest());
  ASSERT_NCCL(packWindowStagingRecordEvent(comm, stream));
}

TEST_F(PackWindowStagingPoolTest, ImplicitDefaultOversizeReportsSizingGuidance) {
  constexpr size_t kLargerRequest = 1U << 20;
  gReshardStagingBuckets[0] = {kSlotBytes, kDefaultStagingBucketSlots};
  gReshardStagingBucketCount = 1;
  gReshardStagingBucketsImplicitDefault = true;

  m2nClearLastError();
  EXPECT_EQ(ncclInvalidArgument, ensurePackWindowStagingBuffer(comm, kLargerRequest, stream));
  const std::string detail = ncclM2nGetLastError();
  EXPECT_NE(std::string::npos, detail.find("implicit default"));
  EXPECT_NE(std::string::npos, detail.find("NCCL_RESHARD_PACK_BUFFSIZES="));
  EXPECT_NE(std::string::npos, detail.find(std::to_string(kLargerRequest)));
  EXPECT_EQ(0, packWindowStagingAllocationCountForTest());
}

TEST_F(PackWindowStagingPoolTest, AssignsStableRoundRobinLanesAcrossCommunicators) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBucketCount = 1;

  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ncclComm_t comm4 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm4, 1, devices));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));
  void* slot0 = getPackWindowStagingBuffer(comm);
  ASSERT_NCCL(packWindowStagingRecordEvent(comm, stream));
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm2, kSlotBytes, stream));
  void* slot1 = getPackWindowStagingBuffer(comm2);
  ASSERT_NCCL(packWindowStagingRecordEvent(comm2, stream));
  ASSERT_NE(slot0, slot1);

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(slot0, getPackWindowStagingBuffer(comm3));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm3, stream));
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm4, kSlotBytes, stream));
  EXPECT_EQ(slot1, getPackWindowStagingBuffer(comm4));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm4, stream));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(slot0, getPackWindowStagingBuffer(comm3));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm3, stream));

  packWindowStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
  ASSERT_NCCL(ncclCommDestroy(comm4));
}

TEST_F(PackWindowStagingPoolTest, RejectsOverlappingUsersOfOnePhysicalLane) {
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
    ownerAcquireResult = ensurePackWindowStagingBuffer(comm, kSlotBytes, stream);
    slotReady.store(true, std::memory_order_release);
    {
      M2nApiUnlock apiUnlock;
      while (!releaseOwner.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    if (ownerAcquireResult == ncclSuccess) {
      ownerRecordResult = packWindowStagingRecordEvent(comm, stream);
    }
  });

  while (!slotReady.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ncclResult_t overlappingResult = ncclSuccess;
  {
    M2nApiLock apiLock;
    overlappingResult = ensurePackWindowStagingBuffer(comm2, kSlotBytes, stream);
  }
  releaseOwner.store(true, std::memory_order_release);
  owner.join();

  EXPECT_EQ(cudaSuccess, ownerCudaResult);
  EXPECT_EQ(ncclSuccess, ownerAcquireResult);
  EXPECT_EQ(ncclInvalidUsage, overlappingResult);
  EXPECT_EQ(ncclSuccess, ownerRecordResult);
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm2, kSlotBytes, stream));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm2, stream));

  packWindowStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(PackWindowStagingPoolTest, WaitsForPriorCommunicatorOnAnotherStream) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, nullptr));
  cudaStream_t nonblocking = nullptr;
  cudaEvent_t marker = nullptr;
  ASSERT_CUDA(cudaStreamCreateWithFlags(&nonblocking, cudaStreamNonBlocking));
  ASSERT_CUDA(cudaEventCreateWithFlags(&marker, cudaEventDisableTiming));

  std::atomic<bool> release{false};
  ASSERT_CUDA(cudaLaunchHostFunc(nullptr, waitForSlotRelease, &release));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm, nullptr));
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm2, kSlotBytes, nonblocking));
  ASSERT_CUDA(cudaEventRecord(marker, nonblocking));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm2, nonblocking));

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(cudaErrorNotReady, cudaEventQuery(marker));
  release.store(true, std::memory_order_release);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(nonblocking));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));

  EXPECT_EQ(cudaSuccess, cudaEventDestroy(marker));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(nonblocking));
  packWindowStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(PackWindowStagingPoolTest, KeepsHostRmaWarmupStatePerCommunicator) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));
  void* sharedSlot = getPackWindowStagingBuffer(comm);
  ASSERT_NCCL(packWindowStagingRecordEvent(comm, stream));
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm2, kSlotBytes, stream));
  EXPECT_EQ(sharedSlot, getPackWindowStagingBuffer(comm2));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm2, stream));

  ASSERT_NCCL(setPackWindowRmaWarmed(comm, true));
  ASSERT_NCCL(setPackWindowRmaWarmed(comm2, false));
  bool warmed = false;
  ASSERT_NCCL(getPackWindowRmaWarmed(comm, &warmed));
  EXPECT_TRUE(warmed);
  ASSERT_NCCL(getPackWindowRmaWarmed(comm2, &warmed));
  EXPECT_FALSE(warmed);

  packWindowStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(PackWindowStagingPoolTest, NewMappingsSkipPoisonedLanes) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBucketCount = 1;
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));
  reshardFailNextCompletionEventRecordForTest(/*bFailStreamSynchronize=*/true);
  EXPECT_EQ(ncclSystemError, packWindowStagingRecordEvent(comm, stream));
  EXPECT_EQ(ncclUnhandledCudaError, ensurePackWindowStagingBuffer(comm, kSlotBytes, stream));

  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm2, kSlotBytes, stream));
  void* healthySlot = getPackWindowStagingBuffer(comm2);
  ASSERT_NCCL(packWindowStagingRecordEvent(comm2, stream));
  ASSERT_NCCL(ensurePackWindowStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(healthySlot, getPackWindowStagingBuffer(comm3));
  ASSERT_NCCL(packWindowStagingRecordEvent(comm3, stream));

  packWindowStagingFinalize();
  reshardClearResourceQuarantine();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
}

} // namespace
