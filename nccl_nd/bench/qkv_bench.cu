/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Q/K/V redistribution latency: three separate ncclNdAllToAll calls versus the
// same calls in one group, issued tensor-major and peer-major. One process per
// GPU on one node; bench/run_bench.sh starts them:
//   qkv_bench --rank=N --nranks=N --uid-file=PATH [--smoke]
//
// Each Q/K/V set is the sequence-sharded to head-sharded redistribution
// [B, T/P, H, D] -> [B, T, H/P, D] of Q (Hq heads), K and V (Hkv heads each),
// BF16, B=1, D=128. Inputs are symmetric windows, outputs plain ncclMemAlloc
// storage. Every mode is captured once as a graph of kInner Q/K/V sets on one
// stream. A sample is one replay, timed with CUDA events and divided by kInner;
// the sample latency is the maximum over ranks. Modes rotate order between
// samples, inputs change every sample, and every sample's outputs are checked
// against an independent per-element reference before the next one.
//
// Grouped tensor-major uses a handle created with NCCL_ND_COPY_ORDER=tensor;
// the other modes use the default (peer-major) handle. Both handles share the
// communicator, stream and buffers.
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "nccl_nd.h"
#include "bench_common.hpp"

namespace {

constexpr int kInner = 20, kWarmup = 5, kSamples = 15, kSmokeSamples = 3;
constexpr size_t kBatch = 1, kDim = 128;
constexpr const char* kModeNames[] = {"separate", "grouped tensor-major", "grouped peer-major"};

ncclComm_t gComm;
cudaStream_t gStream;
int* gErrors;      // [0]: local failures, [1]: sum over ranks
float* gLatency;   // reduced in place: maximum over ranks

__host__ __device__ uint16_t pattern(int rank, size_t index, uint32_t seed) {
  uint64_t x = (static_cast<uint64_t>(seed) << 40) ^ (static_cast<uint64_t>(rank) << 32) ^ index;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint16_t>(x);
}

__global__ void fill(uint16_t* data, size_t n, int rank, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
    data[i] = pattern(rank, i, seed);
}

// Output [B, T, H/P, D] element i came from source rank s = t / (T/P), input
// element [b, t % (T/P), rank * H/P + h, d]. Computed from indices only.
__global__ void verify(const uint16_t* out, size_t n, size_t localS, size_t heads, int rank, int world,
                       uint32_t seed, int* errors) {
  const size_t localH = heads / world, totalS = localS * world;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x) {
    size_t v = i;
    const size_t d = v % kDim;
    v /= kDim;
    const size_t h = v % localH;
    v /= localH;
    const size_t t = v % totalS, b = v / totalS;
    const int source = static_cast<int>(t / localS);
    const size_t index = ((b * localS + t % localS) * heads + rank * localH + h) * kDim + d;
    if (out[i] != pattern(source, index, seed)) atomicAdd(errors, 1);
  }
}

int blocks(size_t n) { return static_cast<int>(std::min<size_t>(1024, (n + 255) / 256)); }

// Sum of failures over ranks; also a rank rendezvous outside any timed region.
int collectiveErrors() {
  NCCL_CHECK(ncclAllReduce(gErrors, gErrors + 1, 1, ncclInt32, ncclSum, gComm, gStream));
  int total = 0;
  CUDA_CHECK(cudaMemcpyAsync(&total, gErrors + 1, sizeof(int), cudaMemcpyDeviceToHost, gStream));
  CUDA_CHECK(cudaMemsetAsync(gErrors, 0, sizeof(int), gStream));
  CUDA_CHECK(cudaStreamSynchronize(gStream));
  return total;
}

struct Tensor {
  size_t heads, localS, elements, bytes;
  size_t inSizes[4], outSizes[4];
  uint16_t* in = nullptr;
  uint16_t* out = nullptr;
  ncclWindow_t window = nullptr;
  ncclNdTensor_t input{}, output{};

  Tensor(size_t totalTokens, size_t h)
      : heads(h),
        localS(totalTokens / gWorld),
        elements(kBatch * localS * h * kDim),
        bytes(elements * sizeof(uint16_t)),
        inSizes{kBatch, localS, h, kDim},
        outSizes{kBatch, totalTokens, h / gWorld, kDim} {
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&in), bytes));
    NCCL_CHECK(ncclCommWindowRegister(gComm, in, bytes, &window, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&out), bytes));
    input = {4, ncclBfloat16, nullptr, window, 0, inSizes, nullptr};
    output = {4, ncclBfloat16, out, nullptr, 0, outSizes, nullptr};
  }
  ~Tensor() {
    NCCL_CHECK(ncclCommWindowDeregister(gComm, window));
    NCCL_CHECK(ncclMemFree(in));
    NCCL_CHECK(ncclMemFree(out));
  }
};

struct Case {
  size_t totalTokens, hq, hkv;
};

class QkvSet {
 public:
  explicit QkvSet(const Case& c) {
    tensors_.emplace_back(new Tensor(c.totalTokens, c.hq));
    tensors_.emplace_back(new Tensor(c.totalTokens, c.hkv));
    tensors_.emplace_back(new Tensor(c.totalTokens, c.hkv));
  }
  ~QkvSet() {
    for (Tensor* t : tensors_) delete t;
  }
  void fillInputs(uint32_t seed) {
    for (size_t i = 0; i < tensors_.size(); ++i) {
      Tensor& t = *tensors_[i];
      fill<<<blocks(t.elements), 256, 0, gStream>>>(t.in, t.elements, gRank, seed + 7919 * uint32_t(i));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemsetAsync(t.out, 0xcd, t.bytes, gStream));
    }
  }
  void submit(ncclNdHandle_t handle, bool grouped) {
    if (grouped) NCCL_CHECK(ncclNdGroupStart());
    for (Tensor* t : tensors_) NCCL_CHECK(ncclNdAllToAll(handle, &t->input, &t->output, 2, 1, gStream));
    if (grouped) NCCL_CHECK(ncclNdGroupEnd());
  }
  void verifyOutputs(uint32_t seed) {
    for (size_t i = 0; i < tensors_.size(); ++i) {
      Tensor& t = *tensors_[i];
      verify<<<blocks(t.elements), 256, 0, gStream>>>(t.out, t.elements, t.localS, t.heads, gRank, gWorld,
                                                      seed + 7919 * uint32_t(i), gErrors);
      CUDA_CHECK(cudaGetLastError());
    }
  }

 private:
  std::vector<Tensor*> tensors_;
};

float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void runCase(const Case& c, int caseIndex, int samples, const std::array<ncclNdHandle_t, 3>& handles) {
  QkvSet set(c);
  std::array<cudaGraphExec_t, 3> exec{};
  for (int m = 0; m < 3; ++m) {
    // Eager correctness first, then capture kInner sets.
    const uint32_t seed = 100 + m;
    set.fillInputs(seed);
    set.submit(handles[m], m > 0);
    set.verifyOutputs(seed);
    if (collectiveErrors() != 0) {
      fprintf(stderr, "[rank %d] %s eager output mismatch\n", gRank, kModeNames[m]);
      exit(1);
    }
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(gStream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < kInner; ++i) set.submit(handles[m], m > 0);
    CUDA_CHECK(cudaStreamEndCapture(gStream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec[m], graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    for (int w = 0; w < kWarmup; ++w) CUDA_CHECK(cudaGraphLaunch(exec[m], gStream));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::array<std::vector<float>, 3> latency;
  for (int sample = 0; sample < samples; ++sample) {
    for (int k = 0; k < 3; ++k) {
      const int m = (sample + caseIndex + k) % 3;  // rotate which mode runs first
      const uint32_t seed = 1000 + 100 * caseIndex + 3 * sample + m;
      set.fillInputs(seed);
      if (collectiveErrors() != 0) exit(1);  // inputs ready on every rank before timing
      CUDA_CHECK(cudaEventRecord(start, gStream));
      CUDA_CHECK(cudaGraphLaunch(exec[m], gStream));
      CUDA_CHECK(cudaEventRecord(stop, gStream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      const float us = 1000.0f * ms / kInner;
      CUDA_CHECK(cudaMemcpyAsync(gLatency, &us, sizeof(float), cudaMemcpyHostToDevice, gStream));
      NCCL_CHECK(ncclAllReduce(gLatency, gLatency, 1, ncclFloat32, ncclMax, gComm, gStream));
      float slowest = 0;
      CUDA_CHECK(cudaMemcpyAsync(&slowest, gLatency, sizeof(float), cudaMemcpyDeviceToHost, gStream));
      set.verifyOutputs(seed);
      if (collectiveErrors() != 0) {
        fprintf(stderr, "[rank %d] %s sample %d output mismatch\n", gRank, kModeNames[m], sample);
        exit(1);
      }
      latency[m].push_back(slowest);
    }
  }
  if (gRank == 0) {
    std::array<float, 3> med{};
    std::string row = "| " + std::to_string(c.hq) + "/" + std::to_string(c.hkv) + ", " +
                      std::to_string(c.totalTokens / 1024) + "K |";
    char cell[96];
    for (int m = 0; m < 3; ++m) {
      med[m] = median(latency[m]);
      const auto range = std::minmax_element(latency[m].begin(), latency[m].end());
      snprintf(cell, sizeof(cell), " %.1f (%.1f-%.1f) |", med[m], *range.first, *range.second);
      row += cell;
    }
    snprintf(cell, sizeof(cell), " %.2fx |", med[0] / med[2]);
    printf("%s%s\n", row.c_str(), cell);
    fflush(stdout);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  for (cudaGraphExec_t e : exec) CUDA_CHECK(cudaGraphExecDestroy(e));
}

// The copy order is read when the handle is created.
ncclNdHandle_t createHandle(const char* copyOrder) {
  if (setenv("NCCL_ND_COPY_ORDER", copyOrder, 1) != 0) {
    perror("setenv");
    exit(2);
  }
  ncclNdHandle_t handle;
  NCCL_CHECK(ncclNdCreate(&handle, gComm));
  unsetenv("NCCL_ND_COPY_ORDER");
  return handle;
}

}  // namespace

int main(int argc, char** argv) {
  std::string uidFile;
  std::vector<std::string> flags;
  if (!initRank(argc, argv, &uidFile, {"--smoke"}, &flags)) return 2;
  const bool smoke = !flags.empty();
  gComm = initComm(uidFile);
  CUDA_CHECK(cudaStreamCreateWithFlags(&gStream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&gErrors, 2 * sizeof(int)));
  CUDA_CHECK(cudaMemset(gErrors, 0, 2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&gLatency, sizeof(float)));
  const ncclNdHandle_t peer = createHandle("peer"), tensor = createHandle("tensor");
  const std::array<ncclNdHandle_t, 3> handles = {peer, tensor, peer};

  // Global token counts; local S = T / P. Head counts must divide by P.
  std::vector<Case> cases = {{1024, 64, 8}, {32768, 64, 8}, {131072, 64, 8}, {32768, 64, 64}};
  if (smoke) cases.resize(1);
  const int samples = smoke ? kSmokeSamples : kSamples;
  if (gRank == 0) {
    int nd = 0, nccl = 0, runtime = 0, driver = 0;
    cudaDeviceProp prop{};
    NCCL_CHECK(ncclNdGetVersion(&nd));
    NCCL_CHECK(ncclGetVersion(&nccl));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gRank));
    printf("qkv_bench: P=%d %s, nccl-nd %d, NCCL %d, CUDA runtime %d, driver %d\n", gWorld, prop.name, nd, nccl,
           runtime, driver);
    printf("BF16 B=%zu D=%zu, sequence->head sharding; CUDA Graph of %d Q/K/V sets per sample, "
           "%d warmup replays, %d samples per mode\n",
           kBatch, kDim, kInner, kWarmup, samples);
    printf("us per Q/K/V set: median over samples (min-max) of the slowest rank; outputs verified every sample\n\n");
    printf("| Hq/Hkv, global tokens | %s | %s | %s | Peer-major speedup vs separate |\n", kModeNames[0],
           kModeNames[1], kModeNames[2]);
    printf("| --- | ---: | ---: | ---: | ---: |\n");
    fflush(stdout);
  }
  for (size_t i = 0; i < cases.size(); ++i) {
    if (cases[i].totalTokens % gWorld != 0 || cases[i].hq % gWorld != 0 || cases[i].hkv % gWorld != 0) {
      fprintf(stderr, "[rank %d] case %zu does not divide by P=%d\n", gRank, i, gWorld);
      return 2;
    }
    runCase(cases[i], static_cast<int>(i), samples, handles);
  }

  NCCL_CHECK(ncclNdDestroy(peer));
  NCCL_CHECK(ncclNdDestroy(tensor));
  CUDA_CHECK(cudaFree(gErrors));
  CUDA_CHECK(cudaFree(gLatency));
  CUDA_CHECK(cudaStreamDestroy(gStream));
  NCCL_CHECK(ncclCommDestroy(gComm));
  if (gRank == 0) {
    unlink(uidFile.c_str());
    printf("\nOVERALL PASS\n");
  }
  return 0;
}
