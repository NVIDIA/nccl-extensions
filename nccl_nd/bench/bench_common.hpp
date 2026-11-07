/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Launch helpers for the benchmarks: one process per GPU on one node, started
// by bench/run_bench.sh with
//   --rank=N --nranks=N --uid-file=PATH [program flags]
// Rank 0 writes the ncclUniqueId to PATH; the other ranks poll it. Each program
// includes this header once, so its definitions are local to that program.
#ifndef NCCL_ND_BENCH_BENCH_COMMON_HPP_
#define NCCL_ND_BENCH_BENCH_COMMON_HPP_

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

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

// Not every program uses every helper; inline and maybe_unused keep -Wall quiet.
[[maybe_unused]] int gRank = -1, gWorld = 0;

inline bool parseInt(const char* text, int* out) {
  char* end = nullptr;
  const long value = strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 || value > 1 << 20) return false;
  *out = static_cast<int>(value);
  return true;
}

// Parse the launcher arguments into gRank, gWorld and *uidFile, accepting the
// program flags listed in `flags` (returned in *given), then make visible GPU
// gRank current. Prints usage and returns false on any error.
inline bool initRank(int argc, char** argv, std::string* uidFile, const std::vector<std::string>& flags = {},
              std::vector<std::string>* given = nullptr) {
  bool valid = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--rank=", 0) == 0) {
      valid = valid && parseInt(argv[i] + 7, &gRank);
    } else if (arg.rfind("--nranks=", 0) == 0) {
      valid = valid && parseInt(argv[i] + 9, &gWorld);
    } else if (arg.rfind("--uid-file=", 0) == 0) {
      *uidFile = arg.substr(11);
    } else {
      bool known = false;
      for (const std::string& flag : flags) known = known || arg == flag;
      valid = valid && known && given != nullptr;
      if (known && given != nullptr) given->push_back(arg);
    }
  }
  if (!valid || gWorld < 2 || gRank < 0 || gRank >= gWorld || uidFile->empty()) {
    std::string extra;
    for (const std::string& flag : flags) extra += " [" + flag + "]";
    fprintf(stderr, "usage: %s --rank=N --nranks=N --uid-file=PATH%s (nranks >= 2)\n", argv[0], extra.c_str());
    return false;
  }
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (devices < gWorld) {
    fprintf(stderr, "[rank %d] %d ranks need as many visible GPUs, found %d\n", gRank, gWorld, devices);
    return false;
  }
  CUDA_CHECK(cudaSetDevice(gRank));
  return true;
}

inline void bootstrap(const std::string& path, ncclUniqueId* id) {
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

inline ncclComm_t initComm(const std::string& uidFile) {
  ncclUniqueId id;
  bootstrap(uidFile, &id);
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, gWorld, id, gRank));
  return comm;
}

}  // namespace

#endif
