/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// GPU correctness test for ncclNdAllToAll. One process per GPU on one node:
//   nd_test --rank=N --nranks=N --uid-file=PATH
// tests/run_tests.sh starts the processes; test_common.hpp has the launch helpers.
//
// Outputs are compared bitwise with a host reference computed independently
// from output element indices. Covered: both directions, FP16/BF16/FP32, 2-D
// and 4-D with B > 1, nonzero window offsets, forward + inverse round trip,
// Q/K/V of different sizes, repeated eager calls and CUDA Graph replays with
// the input rewritten in stream order, and rejected misuse. Groups: grouped
// versus separate Q/K/V, nesting, repeated grouped calls and grouped
// capture/replay, tensor-major copy order, and the group contract (binding,
// sticky first error, Create/Destroy rejection, capture boundaries, balanced
// start/end).
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "nccl_nd.h"
#include "test_common.hpp"

namespace {

ncclComm_t gComm;

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
  void resetOutput() {
    CUDA_CHECK(cudaMemset(out, 0xab, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  bool outputUntouched() const { return readOutput() == std::vector<uint8_t>(bytes, 0xab); }
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

Shape qkvShape(int i) {
  const size_t heads = 4 * static_cast<size_t>(gWorld);
  const Shape shapes[] = {{{2, 64, 2 * heads, 64}, 2, 1, ncclBfloat16},
                          {{2, 64, heads, 64}, 2, 1, ncclBfloat16},
                          {{2, 32, heads, 128}, 2, 1, ncclBfloat16}};
  return shapes[i];
}

// Q, K and V of different sizes in one nested group must match the reference
// and the same calls issued separately. The inner end must enqueue nothing.
void testGroupQkv(ncclNdHandle_t h, cudaStream_t s) {
  std::vector<std::unique_ptr<Buffers>> grouped, separate;
  for (int i = 0; i < 3; ++i) {
    grouped.emplace_back(new Buffers(qkvShape(i), 64 * i));
    separate.emplace_back(new Buffers(qkvShape(i), 64 * i));
    grouped[i]->writeInput(hostInput(qkvShape(i), gRank, 20 + i), s);
    separate[i]->writeInput(hostInput(qkvShape(i), gRank, 20 + i), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
  }
  for (auto& b : separate) NCCL_CHECK(b->call(h, s));
  NCCL_CHECK(ncclNdGroupStart());
  NCCL_CHECK(ncclNdGroupStart());
  for (auto& b : grouped) NCCL_CHECK(b->call(h, s));
  NCCL_CHECK(ncclNdGroupEnd());
  CUDA_CHECK(cudaStreamSynchronize(s));
  bool pending = true;
  for (auto& b : grouped) pending = pending && b->outputUntouched();
  NCCL_CHECK(ncclNdGroupEnd());
  CUDA_CHECK(cudaStreamSynchronize(s));
  check(pending, "group: inner end enqueues nothing");
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    const std::vector<uint8_t> out = grouped[i]->readOutput();
    ok = ok && out == reference(qkvShape(i), 20 + i) && out == separate[i]->readOutput();
  }
  check(ok, "group: Q/K/V of different sizes match the reference and separate calls");
}

// Repeated groups alternate with replays of a graph that captured the whole
// group. Inputs are rewritten in stream order after each group.
void testGroupReuse(ncclNdHandle_t h, cudaStream_t s, int iterations) {
  std::vector<std::unique_ptr<Buffers>> b;
  for (int i = 0; i < 3; ++i) b.emplace_back(new Buffers(qkvShape(i), 128 * i));
  const auto group = [&]() {
    NCCL_CHECK(ncclNdGroupStart());
    for (auto& buffer : b) NCCL_CHECK(buffer->call(h, s));
    NCCL_CHECK(ncclNdGroupEnd());
  };
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  group();
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  std::vector<std::vector<std::vector<uint8_t>>> outputs(iterations);
  for (int i = 0; i < iterations; ++i) {
    for (int j = 0; j < 3; ++j) b[j]->writeInput(hostInput(qkvShape(j), gRank, 300 + 3 * i + j), s);
    if (i % 2 == 0) {
      group();
    } else {
      CUDA_CHECK(cudaGraphLaunch(exec, s));
    }
    for (auto& buffer : b) {
      outputs[i].emplace_back(buffer->bytes);
      CUDA_CHECK(cudaMemcpyAsync(outputs[i].back().data(), buffer->out, buffer->bytes, cudaMemcpyDeviceToHost, s));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(s));
  bool ok = true;
  for (int i = 0; i < iterations; ++i)
    for (int j = 0; j < 3; ++j) ok = ok && outputs[i][j] == reference(qkvShape(j), 300 + 3 * i + j);
  check(ok, "group: " + std::to_string(iterations) + " grouped calls and grouped graph replays with new inputs");
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

// NCCL_ND_COPY_ORDER=tensor changes only the grouped copy order; outputs must
// still match. nd_fault_test checks the issued order itself. An invalid value
// warns and keeps the default, so ncclNdCreate still succeeds.
void testCopyOrder(cudaStream_t s) {
  setenv("NCCL_ND_COPY_ORDER", "bogus", 1);
  ncclNdHandle_t h = nullptr;
  const ncclResult_t invalid = ncclNdCreate(&h, gComm);
  check(invalid == ncclSuccess && h != nullptr, "group: invalid NCCL_ND_COPY_ORDER falls back to the default");
  NCCL_CHECK(ncclNdDestroy(h));
  setenv("NCCL_ND_COPY_ORDER", "tensor", 1);
  NCCL_CHECK(ncclNdCreate(&h, gComm));
  unsetenv("NCCL_ND_COPY_ORDER");
  std::vector<std::unique_ptr<Buffers>> b;
  for (int i = 0; i < 3; ++i) {
    b.emplace_back(new Buffers(qkvShape(i), 64 * i));
    b[i]->writeInput(hostInput(qkvShape(i), gRank, 40 + i), s);
  }
  NCCL_CHECK(ncclNdGroupStart());
  for (auto& buffer : b) NCCL_CHECK(buffer->call(h, s));
  NCCL_CHECK(ncclNdGroupEnd());
  CUDA_CHECK(cudaStreamSynchronize(s));
  bool ok = true;
  for (int i = 0; i < 3; ++i) ok = ok && b[i]->readOutput() == reference(qkvShape(i), 40 + i);
  check(ok, "group: tensor-major copy order (NCCL_ND_COPY_ORDER=tensor) matches the reference");
  NCCL_CHECK(ncclNdDestroy(h));
}

// Every rank takes the same rejection path, so no collective work is pending.
// A failed group must enqueue nothing, so outputs stay untouched.
void testGroupMisuse(ncclNdHandle_t h, cudaStream_t s) {
  const Shape shape{{2, 32, 2 * static_cast<size_t>(gWorld), 64}, 2, 1, ncclBfloat16};
  Buffers a(shape), b(shape, 256);
  a.writeInput(hostInput(shape, gRank, 30), s);
  b.writeInput(hostInput(shape, gRank, 31), s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaStream_t s2;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  ncclNdHandle_t h2;
  NCCL_CHECK(ncclNdCreate(&h2, gComm));
  const auto nothingEnqueued = [&](bool ok, const char* name) {
    CUDA_CHECK(cudaDeviceSynchronize());
    check(ok && a.outputUntouched() && b.outputUntouched(), std::string("group: ") + name);
  };
  const auto badCall = [&]() { return ncclNdAllToAll(h, &b.input, &b.output, 1, 1, s); };
  const ncclResult_t ok = ncclSuccess, usage = ncclInvalidUsage, argument = ncclInvalidArgument;

  nothingEnqueued(ncclNdGroupEnd() == usage, "end without start");
  {
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s), other = b.call(h, s2), later = b.call(h, s), end = ncclNdGroupEnd();
    nothingEnqueued(first == ok && other == usage && later == usage && end == usage, "rejects a second stream");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s), other = b.call(h2, s), end = ncclNdGroupEnd();
    nothingEnqueued(first == ok && other == usage && end == usage, "rejects a second handle");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s), bad = badCall(), later = b.call(h, s), other = b.call(h, nullptr);
    const ncclResult_t end = ncclNdGroupEnd();
    nothingEnqueued(first == ok && bad == argument && later == argument && other == argument && end == argument,
                    "keeps the first error, also for a later call with a different error");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    ncclNdHandle_t created = h;
    const ncclResult_t first = a.call(h, s), create = ncclNdCreate(&created, gComm), end = ncclNdGroupEnd();
    nothingEnqueued(first == ok && create == usage && created == nullptr && end == usage, "rejects ncclNdCreate");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s), destroy = ncclNdDestroy(h2), end = ncclNdGroupEnd();
    nothingEnqueued(first == ok && destroy == usage && end == usage, "rejects ncclNdDestroy of another handle");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t destroy = ncclNdDestroy(h), end = ncclNdGroupEnd();
    nothingEnqueued(destroy == usage && end == usage, "rejects ncclNdDestroy before any call");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t bad = badCall(), inner = ncclNdGroupEnd(), outer = ncclNdGroupEnd(), extra = ncclNdGroupEnd();
    nothingEnqueued(bad == argument && inner == ok && outer == argument && extra == usage,
                    "stays balanced after an error and clears at the outermost end");
  }
  {
    cudaGraph_t graph;
    size_t nodes = 1;
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s);
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    const ncclResult_t end = ncclNdGroupEnd();
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    nothingEnqueued(first == ok && end == usage && nodes == 0, "rejects capture begun while open");
  }
  {
    cudaGraph_t graph;
    size_t nodes = 1;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    const ncclResult_t end = ncclNdGroupEnd();
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    nothingEnqueued(first == ok && end == usage && nodes == 0, "rejects capture ended while open");
  }
  {
    cudaGraph_t graphs[2];
    size_t nodes[2] = {1, 1};
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    NCCL_CHECK(ncclNdGroupStart());
    const ncclResult_t first = a.call(h, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graphs[0]));
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    const ncclResult_t second = b.call(h, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graphs[1]));
    const ncclResult_t end = ncclNdGroupEnd();
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaGraphGetNodes(graphs[i], nullptr, &nodes[i]));
      CUDA_CHECK(cudaGraphDestroy(graphs[i]));
    }
    nothingEnqueued(first == ok && second == usage && end == usage && nodes[0] == 0 && nodes[1] == 0,
                    "rejects a second capture session");
  }
  {
    NCCL_CHECK(ncclNdGroupStart());
    int version = 0;
    const ncclResult_t get = ncclNdGetVersion(&version), first = a.call(h, s), second = b.call(h, s);
    const ncclResult_t end = ncclNdGroupEnd();
    CUDA_CHECK(cudaStreamSynchronize(s));
    check(get == ok && version == NCCL_ND_VERSION_CODE && first == ok && second == ok && end == ok &&
              a.readOutput() == reference(shape, 30) && b.readOutput() == reference(shape, 31),
          "group: ncclNdGetVersion inside does not participate; group after failed groups works");
  }
  a.resetOutput();
  NCCL_CHECK(a.call(h2, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  check(a.readOutput() == reference(shape, 30), "group: handle survives the rejected ncclNdDestroy");
  NCCL_CHECK(ncclNdDestroy(h2));
  CUDA_CHECK(cudaStreamDestroy(s2));
}

}  // namespace

int main(int argc, char** argv) {
  std::string uidFile;
  if (!initRank(argc, argv, &uidFile)) return 2;
  gComm = initComm(uidFile);

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
  testGroupQkv(h, s);
  testGroupReuse(h, s, 10);
  testGroupMisuse(h, s);
  testCopyOrder(s);

  const int total = reduceFailures(gComm, s);
  NCCL_CHECK(ncclNdDestroy(h));
  CUDA_CHECK(cudaStreamDestroy(s));
  NCCL_CHECK(ncclCommDestroy(gComm));
  if (gRank == 0) unlink(uidFile.c_str());
  return total == 0 ? 0 : 1;
}
