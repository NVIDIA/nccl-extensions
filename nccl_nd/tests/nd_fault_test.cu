/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Fault-injection test for submission failures. One process per GPU on one node:
//   nd_fault_test --rank=N --nranks=N --uid-file=PATH
// tests/run_tests.sh starts the processes. Rank r uses visible GPU r.
//
// This file compiles the library source with NCCL_ND_FAULT_INJECTION, which
// adds test-only hooks that fail the entry barrier launch, one copy submission
// or the exit barrier launch. Every rank injects the same fault, so the
// barriers that were enqueued still match across ranks and the stream
// completes. Each case checks the returned error and the cleared host group,
// then follows step 3 of the README recovery procedure on its own
// communicator: every rank failed alike and injected faults leave the CUDA
// context usable, so once the stream synchronizes without error, destroy the
// handle, deregister its windows and abort.
#define NCCL_ND_FAULT_INJECTION 1
#include "../nccl_nd.cu"
#include "test_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// BF16 [2, 32, heads, 64] -> [2, 32 * P, heads / P, 64]. Payloads are not checked:
// a failed submission leaves the output undefined.
struct Tensor {
  ncclComm_t comm;
  size_t inSizes[4], outSizes[4];
  size_t bytes;
  void* in = nullptr;
  void* out = nullptr;
  ncclWindow_t window = nullptr;
  ncclNdTensor_t input{}, output{};

  Tensor(ncclComm_t c, size_t heads)
      : comm(c),
        inSizes{2, 32, heads, 64},
        outSizes{2, 32 * static_cast<size_t>(gWorld), heads / gWorld, 64},
        bytes(2 * 32 * heads * 64 * 2) {
    NCCL_CHECK(ncclMemAlloc(&in, bytes));
    NCCL_CHECK(ncclCommWindowRegister(comm, in, bytes, &window, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclMemAlloc(&out, bytes));
    input = {4, ncclBfloat16, nullptr, window, 0, inSizes, nullptr};
    output = {4, ncclBfloat16, out, nullptr, 0, outSizes, nullptr};
  }
  ~Tensor() {
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    NCCL_CHECK(ncclMemFree(in));
    NCCL_CHECK(ncclMemFree(out));
  }
  ncclResult_t call(ncclNdHandle_t h, cudaStream_t s) { return ncclNdAllToAll(h, &input, &output, 2, 1, s); }
};

enum class Fault { EntryBarrier, Copy, ExitBarrier };

void runCase(const std::string& path, Fault fault, bool grouped) {
  const std::string name = std::string(grouped ? "group " : "single call ") +
                           (fault == Fault::EntryBarrier ? "entry barrier launch failure"
                            : fault == Fault::Copy       ? "copy failure after earlier copies"
                                                         : "exit barrier launch failure");
  ncclComm_t comm = initComm(path);
  ncclNdHandle_t h;
  NCCL_CHECK(ncclNdCreate(&h, comm));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const size_t p = static_cast<size_t>(gWorld);
  std::vector<std::unique_ptr<Tensor>> tensors;
  tensors.emplace_back(new Tensor(comm, 8 * p));
  if (grouped) {
    tensors.emplace_back(new Tensor(comm, 4 * p));
    tensors.emplace_back(new Tensor(comm, 4 * p));
  }

  nccl_nd::faultPlan = {};
  if (fault == Fault::EntryBarrier) nccl_nd::faultPlan.entryBarrier = true;
  // The last copy in issue order, so every other copy was already submitted.
  if (fault == Fault::Copy) nccl_nd::faultPlan.copyIndex = static_cast<long>(tensors.size() * p) - 1;
  if (fault == Fault::ExitBarrier) nccl_nd::faultPlan.exitBarrier = true;
  ncclResult_t result;
  if (grouped) {
    NCCL_CHECK(ncclNdGroupStart());
    for (auto& t : tensors) NCCL_CHECK(t->call(h, s));
    result = ncclNdGroupEnd();
  } else {
    result = tensors[0]->call(h, s);
  }
  nccl_nd::faultPlan = {};

  check(result == ncclUnhandledCudaError, name + ": returns ncclUnhandledCudaError");
  if (grouped) {
    const ncclResult_t closed = ncclNdGroupEnd();
    const ncclResult_t start = ncclNdGroupStart(), end = ncclNdGroupEnd();
    check(closed == ncclInvalidUsage && start == ncclSuccess && end == ncclSuccess, name + ": host group is cleared");
  }
  // Recovery: no further work on the communicator. The fault hit every rank
  // alike and injected faults leave the context usable, so the stream should
  // complete. Clean up only if it does; otherwise abandon everything.
  const bool completed = cudaStreamSynchronize(s) == cudaSuccess;
  check(completed, name + ": stream completes");
  if (!completed) {
    for (auto& t : tensors) t.release();  // Skip deregistration in ~Tensor.
    return;
  }
  NCCL_CHECK(ncclNdDestroy(h));
  tensors.clear();
  CUDA_CHECK(cudaStreamDestroy(s));
  NCCL_CHECK(ncclCommAbort(comm));
}

}  // namespace

int main(int argc, char** argv) {
  std::string uidFile;
  if (!initRank(argc, argv, &uidFile)) return 2;
  if (gRank == 0) printf("nd_fault_test, P=%d (NCCL ND WARN lines below are expected)\n", gWorld);

  // Each case aborts its communicator, so each gets its own unique ID file.
  std::vector<std::string> paths;
  for (bool grouped : {false, true}) {
    for (Fault fault : {Fault::EntryBarrier, Fault::Copy, Fault::ExitBarrier}) {
      paths.push_back(uidFile + "." + std::to_string(paths.size()));
      runCase(paths.back(), fault, grouped);
    }
  }

  // Sum failures over ranks on a fresh communicator.
  paths.push_back(uidFile + ".sum");
  ncclComm_t comm = initComm(paths.back());
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const int total = reduceFailures(comm, s);
  CUDA_CHECK(cudaStreamDestroy(s));
  NCCL_CHECK(ncclCommDestroy(comm));
  if (gRank == 0)
    for (const std::string& path : paths) unlink(path.c_str());
  return total == 0 ? 0 : 1;
}
