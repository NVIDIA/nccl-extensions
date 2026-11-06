/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// GPU correctness test for ncclNdAllToAll. One process per GPU on one node:
//   nd_test --rank=N --nranks=N --uid-file=PATH
// tests/run_tests.sh starts the processes. Rank r uses visible GPU r.
// Rank 0 writes the ncclUniqueId to PATH; the other ranks poll it.
//
// Outputs are compared bitwise with a host reference computed independently
// from output element indices. Covered: both directions, FP16/BF16/FP32, 2-D
// and 4-D with B > 1, nonzero window offsets, forward + inverse round trip,
// Q/K/V of different sizes, repeated eager calls and CUDA Graph replays with
// the input rewritten in stream order, and rejected misuse.
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nccl_nd.h"

#define CUDA_CHECK(x)                                                                                        \
  do {                                                                                                       \
    cudaError_t e_ = (x);                                                                                    \
    if (e_ != cudaSuccess) {                                                                                 \
      fprintf(stderr, "[rank %d] CUDA %s at %s:%d: %s\n", gRank, cudaGetErrorString(e_), __FILE__, __LINE__, \
              #x);                                                                                           \
      exit(2);                                                                                               \
    }                                                                                                        \
  } while (0)
#define NCCL_CHECK(x)                                                                                        \
  do {                                                                                                       \
    ncclResult_t r_ = (x);                                                                                   \
    if (r_ != ncclSuccess) {                                                                                 \
      fprintf(stderr, "[rank %d] NCCL %s at %s:%d: %s\n", gRank, ncclGetErrorString(r_), __FILE__, __LINE__, \
              #x);                                                                                           \
      exit(2);                                                                                               \
    }                                                                                                        \
  } while (0)

namespace {

int gRank = -1, gWorld = 0, gFailures = 0;
ncclComm_t gComm;

void check(bool ok, const std::string& name) {
  if (!ok) {
    ++gFailures;
    fprintf(stderr, "[rank %d] FAIL %s\n", gRank, name.c_str());
  } else if (gRank == 0) {
    printf("PASS %s\n", name.c_str());
  }
  fflush(stdout);
}

uint32_t hashValue(int rank, size_t index, uint32_t seed) {
  uint64_t x = (static_cast<uint64_t>(seed) << 40) ^ (static_cast<uint64_t>(rank) << 32) ^ index;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint32_t>(x);
}

size_t count(const std::vector<size_t>& shape) {
  size_t n = 1;
  for (size_t v : shape) n *= v;
  return n;
}

struct Shape {
  std::vector<size_t> in;  // local input shape
  int scatter, gather;
  ncclDataType_t type;

  size_t elementBytes() const { return type == ncclFloat32 ? 4 : 2; }
  size_t bytes() const { return count(in) * elementBytes(); }
  std::vector<size_t> out() const {
    std::vector<size_t> result = in;
    result[scatter] /= gWorld;
    result[gather] *= gWorld;
    return result;
  }
  Shape inverse() const { return {out(), gather, scatter, type}; }
  std::string name() const {
    std::string s = type == ncclFloat32 ? "fp32 [" : type == ncclBfloat16 ? "bf16 [" : "fp16 [";
    for (size_t i = 0; i < in.size(); ++i) s += (i ? "," : "") + std::to_string(in[i]);
    return s + "] (" + std::to_string(scatter) + "," + std::to_string(gather) + ")";
  }
};

// Input bytes of `rank`; payload values are arbitrary bit patterns.
std::vector<uint8_t> hostInput(const Shape& s, int rank, uint32_t seed) {
  const size_t n = count(s.in), e = s.elementBytes();
  std::vector<uint8_t> bytes(n * e);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t v = hashValue(rank, i, seed);
    memcpy(&bytes[i * e], &v, e);
  }
  return bytes;
}

// Y[d] = concat_s(split(X[s], P, scatter)[d], gather) for d = gRank, by output element index.
std::vector<uint8_t> reference(const Shape& s, uint32_t seed) {
  const std::vector<size_t> out = s.out();
  const size_t n = count(out), e = s.elementBytes(), ndim = s.in.size();
  std::vector<uint8_t> bytes(n * e);
  std::vector<size_t> coord(ndim);
  for (size_t o = 0; o < n; ++o) {
    size_t rem = o;
    for (size_t d = ndim; d-- > 0;) {
      coord[d] = rem % out[d];
      rem /= out[d];
    }
    const int source = static_cast<int>(coord[s.gather] / s.in[s.gather]);
    coord[s.gather] %= s.in[s.gather];
    coord[s.scatter] += static_cast<size_t>(gRank) * out[s.scatter];
    size_t index = 0;
    for (size_t d = 0; d < ndim; ++d) index = index * s.in[d] + coord[d];
    const uint32_t v = hashValue(source, index, seed);
    memcpy(&bytes[o * e], &v, e);
  }
  return bytes;
}

// Window-backed input at `offset` bytes and window-backed output, so an output
// can be the input of the inverse call.
struct Buffers {
  Shape shape;
  std::vector<size_t> outSizes;
  size_t bytes, offset;
  void* in = nullptr;
  void* out = nullptr;
  ncclWindow_t inWindow = nullptr, outWindow = nullptr;
  ncclNdTensor_t input{}, output{};

  Buffers(const Shape& s, size_t windowOffset = 0)
      : shape(s), outSizes(s.out()), bytes(s.bytes()), offset(windowOffset) {
    NCCL_CHECK(ncclMemAlloc(&in, bytes + offset));
    NCCL_CHECK(ncclCommWindowRegister(gComm, in, bytes + offset, &inWindow, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclMemAlloc(&out, bytes));
    NCCL_CHECK(ncclCommWindowRegister(gComm, out, bytes, &outWindow, NCCL_WIN_COLL_SYMMETRIC));
    input = {static_cast<unsigned>(s.in.size()), s.type, nullptr, inWindow, offset, shape.in.data(), nullptr};
    output = {static_cast<unsigned>(s.in.size()), s.type, out, nullptr, 0, outSizes.data(), nullptr};
    // Tests use a non-blocking stream, which does not order after this memset.
    CUDA_CHECK(cudaMemset(out, 0xab, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  ~Buffers() {
    NCCL_CHECK(ncclCommWindowDeregister(gComm, inWindow));
    NCCL_CHECK(ncclCommWindowDeregister(gComm, outWindow));
    NCCL_CHECK(ncclMemFree(in));
    NCCL_CHECK(ncclMemFree(out));
  }
  void* inputData() const { return static_cast<uint8_t*>(in) + offset; }
  void writeInput(const std::vector<uint8_t>& h, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(inputData(), h.data(), bytes, cudaMemcpyHostToDevice, s));
  }
  std::vector<uint8_t> readOutput() const {
    std::vector<uint8_t> h(bytes);
    CUDA_CHECK(cudaMemcpy(h.data(), out, bytes, cudaMemcpyDeviceToHost));
    return h;
  }
  ncclResult_t call(ncclNdHandle_t h, cudaStream_t s) {
    return ncclNdAllToAll(h, &input, &output, shape.scatter, shape.gather, s);
  }
};

// Forward against the reference, then the inverse reading the forward output
// through its window must restore the input.
void testRoundTrip(ncclNdHandle_t h, cudaStream_t s, const Shape& shape, size_t offset) {
  Buffers forward(shape, offset), inverse(shape.inverse());
  inverse.input.win_hdl = forward.outWindow;
  const std::vector<uint8_t> input = hostInput(shape, gRank, 1);
  forward.writeInput(input, s);
  NCCL_CHECK(forward.call(h, s));
  NCCL_CHECK(inverse.call(h, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  const std::string name = shape.name() + " offset=" + std::to_string(offset);
  check(forward.readOutput() == reference(shape, 1), name);
  check(inverse.readOutput() == input, name + " inverse");
}

// Q, K and V of different sizes in sequence on one handle and stream.
void testQkv(ncclNdHandle_t h, cudaStream_t s) {
  const size_t heads = 4 * static_cast<size_t>(gWorld);
  const Shape shapes[] = {{{2, 64, 2 * heads, 64}, 2, 1, ncclBfloat16},
                          {{2, 64, heads, 64}, 2, 1, ncclBfloat16},
                          {{2, 32, heads, 128}, 2, 1, ncclBfloat16}};
  std::vector<Buffers*> b;
  for (int i = 0; i < 3; ++i) {
    b.push_back(new Buffers(shapes[i], 64 * i));
    b[i]->writeInput(hostInput(shapes[i], gRank, 10 + i), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
  }
  for (Buffers* buffer : b) NCCL_CHECK(buffer->call(h, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  bool ok = true;
  for (int i = 0; i < 3; ++i) ok = ok && b[i]->readOutput() == reference(shapes[i], 10 + i);
  check(ok, "Q/K/V of different sizes");
  for (Buffers* buffer : b) delete buffer;
}

// Each iteration rewrites the input in stream order right after the previous
// call, so a peer still reading the old payload would corrupt the output.
// Iterations alternate eager calls and replays of one captured graph, so
// barrier epochs must stay in step across both.
void testReuse(ncclNdHandle_t h, cudaStream_t s, const Shape& shape, int iterations) {
  Buffers b(shape, 256);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  NCCL_CHECK(b.call(h, s));
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  std::vector<std::vector<uint8_t>> inputs(iterations), outputs(iterations, std::vector<uint8_t>(b.bytes));
  for (int i = 0; i < iterations; ++i) inputs[i] = hostInput(shape, gRank, 100 + i);
  for (int i = 0; i < iterations; ++i) {
    b.writeInput(inputs[i], s);
    if (i % 2 == 0) {
      NCCL_CHECK(b.call(h, s));
    } else {
      CUDA_CHECK(cudaGraphLaunch(exec, s));
    }
    CUDA_CHECK(cudaMemcpyAsync(outputs[i].data(), b.out, b.bytes, cudaMemcpyDeviceToHost, s));
  }
  CUDA_CHECK(cudaStreamSynchronize(s));
  bool ok = true;
  for (int i = 0; i < iterations; ++i) ok = ok && outputs[i] == reference(shape, 100 + i);
  check(ok, shape.name() + " " + std::to_string(iterations) + " eager calls and graph replays with new inputs");
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

// Every rank takes the same rejection path, so no collective work is pending.
void testMisuse(ncclNdHandle_t h, cudaStream_t s) {
  const Shape shape{{2, 32, 2 * static_cast<size_t>(gWorld), 64}, 2, 1, ncclBfloat16};
  Buffers b(shape);
  b.writeInput(hostInput(shape, gRank, 5), s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  const std::vector<uint8_t> untouched = b.readOutput();
  const auto rejected = [&](ncclResult_t actual, ncclResult_t expected, const char* name) {
    CUDA_CHECK(cudaDeviceSynchronize());
    check(actual == expected && b.readOutput() == untouched, std::string("rejects ") + name);
  };
  ncclNdTensor_t misaligned = b.output, aliased = b.output, pastEnd = b.input;
  misaligned.data = static_cast<uint8_t*>(b.out) + 8;
  aliased.data = b.inputData();
  pastEnd.win_offset = 16;
  rejected(b.call(h, nullptr), ncclInvalidUsage, "null stream");
  rejected(b.call(h, cudaStreamLegacy), ncclInvalidUsage, "legacy stream");
  rejected(ncclNdAllToAll(h, &b.input, &b.output, 1, 1, s), ncclInvalidArgument, "unsupported axis pair");
  rejected(ncclNdAllToAll(h, &b.input, &misaligned, 2, 1, s), ncclInvalidArgument, "misaligned output");
  rejected(ncclNdAllToAll(h, &b.input, &aliased, 2, 1, s), ncclInvalidArgument, "output aliasing input");
  rejected(ncclNdAllToAll(h, &pastEnd, &b.output, 2, 1, s), ncclInvalidArgument, "window range past registration");
  NCCL_CHECK(b.call(h, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  check(b.readOutput() == reference(shape, 5), "call after rejected calls");
}

bool parseInt(const char* text, int* out) {
  char* end = nullptr;
  const long value = strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 || value > 1 << 20) return false;
  *out = static_cast<int>(value);
  return true;
}

void bootstrap(const std::string& path, ncclUniqueId* id) {
  if (gRank == 0) {
    NCCL_CHECK(ncclGetUniqueId(id));
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (f == nullptr || fwrite(id, sizeof(*id), 1, f) != 1 || fclose(f) != 0 || rename(tmp.c_str(), path.c_str())) {
      perror("write unique id");
      exit(2);
    }
    return;
  }
  for (int i = 0; i < 12000; ++i) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f != nullptr) {
      const bool ok = fread(id, sizeof(*id), 1, f) == 1;
      fclose(f);
      if (ok) return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  fprintf(stderr, "[rank %d] timed out waiting for %s\n", gRank, path.c_str());
  exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string uidFile;
  bool valid = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--rank=", 0) == 0) {
      valid = valid && parseInt(argv[i] + 7, &gRank);
    } else if (arg.rfind("--nranks=", 0) == 0) {
      valid = valid && parseInt(argv[i] + 9, &gWorld);
    } else if (arg.rfind("--uid-file=", 0) == 0) {
      uidFile = arg.substr(11);
    } else {
      valid = false;
    }
  }
  if (!valid || gWorld < 2 || gRank < 0 || gRank >= gWorld || uidFile.empty()) {
    fprintf(stderr, "usage: %s --rank=N --nranks=N --uid-file=PATH (nranks >= 2)\n", argv[0]);
    return 2;
  }
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (devices < gWorld) {
    fprintf(stderr, "[rank %d] %d ranks need as many visible GPUs, found %d\n", gRank, gWorld, devices);
    return 2;
  }
  CUDA_CHECK(cudaSetDevice(gRank));
  ncclUniqueId id;
  bootstrap(uidFile, &id);
  NCCL_CHECK(ncclCommInitRank(&gComm, gWorld, id, gRank));

  int version = 0, ncclVersion = 0;
  NCCL_CHECK(ncclNdGetVersion(&version));
  NCCL_CHECK(ncclGetVersion(&ncclVersion));
  if (gRank == 0) printf("nccl-nd %d, NCCL %d, P=%d\n", version, ncclVersion, gWorld);

  ncclNdHandle_t h;
  NCCL_CHECK(ncclNdCreate(&h, gComm));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // Head extents are P * k, so the per-peer width stays 16-byte aligned for every type.
  const size_t p = static_cast<size_t>(gWorld);
  // The inverse of each shape covers the other direction.
  for (ncclDataType_t type : {ncclFloat16, ncclBfloat16, ncclFloat32}) {
    testRoundTrip(h, s, {{64 * p, 8 * p}, 1, 0, type}, 0);
    testRoundTrip(h, s, {{3, 16 * p, 2 * p, 8}, 2, 1, type}, 4096 + 16);
  }
  testQkv(h, s);
  testReuse(h, s, {{2, 128, 4 * p, 64}, 2, 1, ncclBfloat16}, 20);
  testMisuse(h, s);

  // Sum failures over ranks.
  int* failures;
  CUDA_CHECK(cudaMalloc(&failures, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(failures, &gFailures, sizeof(int), cudaMemcpyHostToDevice));
  NCCL_CHECK(ncclAllReduce(failures, failures, 1, ncclInt32, ncclSum, gComm, s));
  int total = 0;
  CUDA_CHECK(cudaMemcpyAsync(&total, failures, sizeof(int), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  if (gRank == 0) printf("%s: %d failure(s) across %d ranks\n", total == 0 ? "ALL PASSED" : "FAILED", total, gWorld);

  NCCL_CHECK(ncclNdDestroy(h));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(failures));
  NCCL_CHECK(ncclCommDestroy(gComm));
  if (gRank == 0) unlink(uidFile.c_str());
  return total == 0 ? 0 : 1;
}
