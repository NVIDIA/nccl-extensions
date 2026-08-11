/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_mpi — gtest-parametrized multi-process MPI driver for
 * the basic_api matrix. Each TestCase is one gtest parameter (or one
 * TestCase × algorithm parameter when --algorithm all is used).
 *
 * Bootstrap: standard ncclGetUniqueId + MPI_Bcast + ncclCommInitRank.
 * Drives the shared core in basic_api_test_core.h.
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_m2n.h"
#include "basic_api_test_core.h"

namespace {

static MPI_Comm testMpiWorld() {
  return MPI_COMM_WORLD; // NOLINT(bugprone-casting-through-void)
}

static MPI_Datatype testMpiByte() {
  return MPI_BYTE; // NOLINT(bugprone-casting-through-void)
}

static MPI_Datatype testMpiInt() {
  return MPI_INT; // NOLINT(bugprone-casting-through-void)
}

static MPI_Op testMpiMin() {
  return MPI_MIN; // NOLINT(bugprone-casting-through-void)
}

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = (cmd);                                                     \
    if (e != MPI_SUCCESS) {                                            \
      fprintf(stderr, "MPI error %s:%d: %d\n", __FILE__, __LINE__, e); \
      MPI_Abort(testMpiWorld(), 1);                                    \
    }                                                                  \
  } while (0)

/* ======================================================================
 * Bootstrap-specific TestEnv hooks.
 * ====================================================================*/

static void mpiBarrier(TestEnv* env) {
  (void)env;
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

static int mpiAllreduceMinInt(TestEnv* env, int local) {
  (void)env;
  int g = local;
  MPICHECK(MPI_Allreduce(&local, &g, 1, testMpiInt(), testMpiMin(), testMpiWorld()));
  return g;
}

static bool mpiIsRank0Printer(TestEnv* env) {
  return env->rank == 0;
}

/* ======================================================================
 * gtest parameter registry state.
 * ====================================================================*/

struct MpiParam {
  TestCase tc;
  std::string algorithmEnv; /* RING or DIRECT */
  ApiKind api = ApiKind::Window;
  int streamPoolMode = -1; /* -1 = inherited, 0 = user stream, 1 = internal pool */
};

/* gtest pretty-printer hook for MpiParam — found via ADL by
 * INSTANTIATE_TEST_SUITE_P's value-printer. The lookup is template-driven,
 * so plain `-Wunused-function` cannot see the use; mark it accordingly. */
[[maybe_unused]] static void printTo(const MpiParam& param, std::ostream* os) {
  *os << param.algorithmEnv << ":" << (param.api == ApiKind::Default ? "default" : "window");
  if (param.streamPoolMode >= 0) {
    *os << ":" << (param.streamPoolMode == 0 ? "user_stream" : "internal_pool");
  }
  *os << ":" << param.tc.name;
}

static BasicApiCliArgs gCli;
static std::vector<TestCase> gCases;
static int gWorldRank = 0;
static int gWorldSize = 0;
static int gNumDevices = 0;
static int gDevice = 0;
static ncclComm_t gComm = nullptr;
static cudaStream_t gStream = nullptr;
static ncclM2nHandle_t gM2nHandle = nullptr;
static void* gBuffer = nullptr;
static size_t gBufferBytes = 4096;
static void* gCopyBuffer = nullptr;
static size_t gCopyBufferBytes = 0;
static ncclComm_t gGroupCommA = nullptr;
static ncclComm_t gGroupCommB = nullptr;
static cudaStream_t gGroupStreamA = nullptr;
static cudaStream_t gGroupStreamB = nullptr;
static void* gGroupBufferA = nullptr;
static void* gGroupBufferB = nullptr;
static std::string gActiveAlgorithm;
static int gActiveStreamPoolMode = -1;
static bool gInitialUseInternalStreamsEnvSet = false;
static std::string gInitialUseInternalStreamsEnv;
#if !defined(GTEST_SKIP)
static int gSkippedCases = 0;
#endif

static bool runAllAlgorithms() {
  return basicApiRunAllAlgorithms(gCli);
}

static const char* requestedAlgorithmEnv() {
  return basicApiRequestedAlgorithmEnv(gCli, gWorldRank == 0);
}

static std::vector<MpiParam> selectedParams() {
  std::vector<MpiParam> params;
  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);

  auto appendParams = [&](const TestCase& tc, const char* algo, ApiKind api) {
    if (tc.group == "stream_ordering" || tc.group == "stream_churn" || tc.group == "graph_capture") {
      params.push_back(MpiParam{tc, algo, api, 1});
      params.push_back(MpiParam{tc, algo, api, 0});
    } else {
      params.push_back(MpiParam{tc, algo, api, -1});
    }
  };

  std::vector<ApiKind> apis;
  if (basicApiRunAllApis(gCli)) {
    apis = {ApiKind::Window, ApiKind::Default};
  } else {
    apis = {basicApiRequestedApi(gCli)};
  }

  if (runAllAlgorithms()) {
    const char* algos[] = {"RING", "DIRECT"};
    for (const char* algo : algos) {
      for (ApiKind api : apis) {
        for (const TestCase& tc : cases) {
          appendParams(tc, algo, api);
        }
      }
    }
  } else {
    const char* algo = requestedAlgorithmEnv();
    for (ApiKind api : apis) {
      for (const TestCase& tc : cases) {
        appendParams(tc, algo, api);
      }
    }
  }
  return params;
}

static std::string gtestCaseName(const ::testing::TestParamInfo<MpiParam>& info) {
  std::string prefix = info.param.algorithmEnv;
  prefix += (info.param.api == ApiKind::Default) ? "_default" : "_window";
  if (info.param.streamPoolMode >= 0) {
    prefix += (info.param.streamPoolMode == 0) ? "_user_stream" : "_internal_pool";
  }
  return basicApiGtestCaseName(info.param.tc.name, info.index, prefix.c_str());
}

#if !defined(GTEST_SKIP)
static void recordFallbackSkip(const char* reason) {
  basicApiRecordFallbackSkip(&gSkippedCases, reason, gWorldRank == 0);
}
#endif

static void applyStreamPoolMode(int streamPoolMode) {
  if (streamPoolMode == 0) {
    testSetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS", "0");
  } else if (streamPoolMode > 0) {
    testSetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS", "1");
  } else if (gInitialUseInternalStreamsEnvSet) {
    testSetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS", gInitialUseInternalStreamsEnv.c_str());
  } else {
    testUnsetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS");
  }
}

static void activateRuntime(const MpiParam& param, bool bForceReset = false) {
  if (!bForceReset && gActiveAlgorithm == param.algorithmEnv && gActiveStreamPoolMode == param.streamPoolMode) return;

  TEST_CUDACHECK(cudaStreamSynchronize(gStream));
  basicApiConfigureReshardEnv(gCli, param.algorithmEnv.c_str());
  applyStreamPoolMode(param.streamPoolMode);
  TEST_NCCLCHECK(ncclM2nFinalize(gM2nHandle));
  gM2nHandle = nullptr;
  TEST_NCCLCHECK(ncclM2nInit(&gM2nHandle, NULL));
  gActiveAlgorithm = param.algorithmEnv;
  gActiveStreamPoolMode = param.streamPoolMode;
}

class BasicApiMpiTest : public ::testing::TestWithParam<MpiParam> {};

static CaseResult runStreamChurn(const TestCase& tc, TestEnv* env) {
  CaseShape shape;
  const char* skipReason = nullptr;
  if (!caseFeasibleAt(tc, env->worldSize, &shape, &skipReason)) {
    return runOneCase(tc, env);
  }

  constexpr int kStreamCount = 3;
  cudaStream_t freshStreams[kStreamCount] = {};
  const bool bUseFreshStreams = env->rank >= shape.srcTotal;
  if (bUseFreshStreams) {
    // Keep every handle live so CUDA cannot recycle one and hide cache churn.
    for (cudaStream_t& stream : freshStreams) {
      TEST_CUDACHECK(cudaStreamCreate(&stream));
    }
  }

  CaseResult result = makePass();
  for (int i = 0; i < kStreamCount; i++) {
    env->stream = bUseFreshStreams ? freshStreams[i] : gStream;
    result = runOneCase(tc, env);
    if (result.status != CASE_PASS) {
      break;
    }
  }

  if (bUseFreshStreams) {
    for (cudaStream_t stream : freshStreams) {
      TEST_CUDACHECK(cudaStreamDestroy(stream));
    }
  }
  env->stream = gStream;
  return result;
}

static CaseResult runStreamOrdering(const TestCase& tc, TestEnv* env) {
  cudaStream_t alternateStream = nullptr;
  TEST_CUDACHECK(cudaStreamCreateWithFlags(&alternateStream, cudaStreamNonBlocking));
  env->alternateStream = alternateStream;
  CaseResult result = runOneCase(tc, env);
  env->alternateStream = nullptr;
  TEST_CUDACHECK(cudaStreamDestroy(alternateStream));
  return result;
}

TEST(M2nGroupMpiTest, OverlappingCommunicatorsPreserveBucketOrder) {
  if (gCli.filter == nullptr || strcmp(gCli.filter, "group_mixed_context") != 0) {
#if defined(GTEST_SKIP)
    GTEST_SKIP() << "needs --filter group_mixed_context";
#else
    recordFallbackSkip("needs --filter group_mixed_context");
    return;
#endif
  }
  if (gWorldSize < 3) {
#if defined(GTEST_SKIP)
    GTEST_SKIP() << "needs at least 3 ranks";
#else
    recordFallbackSkip("needs at least 3 ranks");
    return;
#endif
  }

  ncclUniqueId ids[2];
  if (gWorldRank == 0) {
    TEST_NCCLCHECK(ncclGetUniqueId(&ids[0]));
    TEST_NCCLCHECK(ncclGetUniqueId(&ids[1]));
  }
  MPICHECK(MPI_Bcast(ids, sizeof(ids), testMpiByte(), 0, testMpiWorld()));

  const bool inA = gWorldRank <= 1;
  const bool inB = gWorldRank == 1 || gWorldRank == 2;
  const int rankA = gWorldRank;
  const int rankB = gWorldRank - 1;
  if (inA) {
    TEST_NCCLCHECK(ncclCommInitRank(&gGroupCommA, 2, ids[0], rankA));
    TEST_CUDACHECK(cudaStreamCreate(&gGroupStreamA));
    TEST_NCCLCHECK(ncclMemAlloc(&gGroupBufferA, 4096));
  }
  if (inB) {
    TEST_NCCLCHECK(ncclCommInitRank(&gGroupCommB, 2, ids[1], rankB));
    TEST_CUDACHECK(cudaStreamCreate(&gGroupStreamB));
    TEST_NCCLCHECK(ncclMemAlloc(&gGroupBufferB, 4096));
  }

  ncclMesh_t srcMesh{};
  srcMesh.dims[0] = 1;
  srcMesh.dims[1] = 1;
  srcMesh.startRank = 0;
  ncclMesh_t dstMesh = srcMesh;
  dstMesh.startRank = 1;
  auto makeTensor = [](void* data, ncclMesh_t* mesh) {
    ncclDistTensor_t tensor{};
    tensor.dataPtr = data;
    tensor.localShape[0] = 32;
    tensor.ndims = 1;
    tensor.dtype = ncclUint8;
    tensor.mesh = mesh;
    tensor.placements[0] = NCCL_RESHARD_REPLICATE;
    tensor.placements[1] = NCCL_RESHARD_REPLICATE;
    return tensor;
  };
  std::array<ncclDistTensor_t, 2> srcA{};
  std::array<ncclDistTensor_t, 2> dstA{};
  std::array<ncclDistTensor_t, 2> srcB{};
  std::array<ncclDistTensor_t, 2> dstB{};
  for (size_t i = 0; i < 2; i++) {
    const size_t offset = i * 64;
    if (inA) {
      srcA[i] = makeTensor(rankA == 0 ? static_cast<char*>(gGroupBufferA) + offset : nullptr, &srcMesh);
      dstA[i] = makeTensor(rankA == 1 ? static_cast<char*>(gGroupBufferA) + offset : nullptr, &dstMesh);
      if (rankA == 0) TEST_CUDACHECK(cudaMemsetAsync(srcA[i].dataPtr, 0x20 + static_cast<int>(i), 32, gGroupStreamA));
      if (rankA == 1) TEST_CUDACHECK(cudaMemsetAsync(dstA[i].dataPtr, 0, 32, gGroupStreamA));
    }
    if (inB) {
      srcB[i] = makeTensor(rankB == 0 ? static_cast<char*>(gGroupBufferB) + offset : nullptr, &srcMesh);
      dstB[i] = makeTensor(rankB == 1 ? static_cast<char*>(gGroupBufferB) + offset : nullptr, &dstMesh);
      if (rankB == 0) TEST_CUDACHECK(cudaMemsetAsync(srcB[i].dataPtr, 0x40 + static_cast<int>(i), 32, gGroupStreamB));
      if (rankB == 1) TEST_CUDACHECK(cudaMemsetAsync(dstB[i].dataPtr, 0, 32, gGroupStreamB));
    }
  }

  TEST_NCCLCHECK(ncclM2nGroupStart());
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[0], &dstA[0], gGroupStreamA));
  if (inB) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommB, &srcB[0], &dstB[0], gGroupStreamB));
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[1], &dstA[1], gGroupStreamA));
  if (inB) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommB, &srcB[1], &dstB[1], gGroupStreamB));
  TEST_NCCLCHECK(ncclM2nGroupEnd());
  if (inA) TEST_CUDACHECK(cudaStreamSynchronize(gGroupStreamA));
  if (inB) TEST_CUDACHECK(cudaStreamSynchronize(gGroupStreamB));

  std::array<unsigned char, 32> actual{};
  auto expectPattern = [&](const ncclDistTensor_t& tensor, int expected) {
    TEST_CUDACHECK(cudaMemcpy(actual.data(), tensor.dataPtr, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual[0]);
    EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [&](unsigned char value) { return value == actual[0]; }));
  };
  for (size_t i = 0; i < 2; i++) {
    if (inA && rankA == 1) expectPattern(dstA[i], 0x20 + static_cast<int>(i));
    if (inB && rankB == 1) expectPattern(dstB[i], 0x40 + static_cast<int>(i));
  }

  if (inA && rankA == 1) {
    TEST_CUDACHECK(cudaMemsetAsync(dstA[0].dataPtr, 0, 32, gGroupStreamA));
    TEST_CUDACHECK(cudaMemsetAsync(dstA[1].dataPtr, 0, 32, gGroupStreamA));
  }
  TEST_NCCLCHECK(ncclM2nGroupStart());
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[0], &dstA[0], gGroupStreamA));
  if (inB) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommB, nullptr, nullptr, gGroupStreamB));
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[1], &dstA[1], gGroupStreamA));
  ncclResult_t bucketOrderResult = ncclM2nGroupEnd();
  EXPECT_EQ(inB ? ncclInvalidArgument : ncclSuccess, bucketOrderResult);
  if (inA) TEST_CUDACHECK(cudaStreamSynchronize(gGroupStreamA));
  if (inA && rankA == 1) {
    expectPattern(dstA[0], 0x20);
    expectPattern(dstA[1], 0x21);
  }
  MPICHECK(MPI_Barrier(testMpiWorld()));

  if (inA && rankA == 1) TEST_CUDACHECK(cudaMemsetAsync(dstA[0].dataPtr, 0, 32, gGroupStreamA));
  TEST_NCCLCHECK(ncclM2nGroupStart());
  if (inA) {
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[0], &dstA[0], gGroupStreamA));
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, nullptr, nullptr, gGroupStreamA));
  }
  ncclResult_t entryOrderResult = ncclM2nGroupEnd();
  EXPECT_EQ(inA ? ncclInvalidArgument : ncclSuccess, entryOrderResult);
  if (inA) TEST_CUDACHECK(cudaStreamSynchronize(gGroupStreamA));
  if (inA && rankA == 1) expectPattern(dstA[0], 0x20);
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

TEST_P(BasicApiMpiTest, Reshard) {
  const MpiParam& param = GetParam();
  SCOPED_TRACE(param.tc.name);
  SCOPED_TRACE(param.algorithmEnv);

  activateRuntime(param, param.tc.group == "graph_capture");

  TestEnv env{};
  env.rank = gWorldRank;
  env.worldSize = gWorldSize;
  env.device = gDevice;
  env.comm = gComm;
  env.stream = gStream;
  env.m2nHandle = gM2nHandle;
  env.buffer = gBuffer;
  env.bufferBytes = gBufferBytes;
  env.copyBuffer = gCopyBuffer;
  env.copyBufferBytes = gCopyBufferBytes;
  env.apiKind = param.api;
  env.expectPackWindow = strcmp(gCli.copyAlgorithm, "packwindow") == 0;
  env.verbose = gCli.verbose;
  env.barrier = mpiBarrier;
  env.allreduceMinInt = mpiAllreduceMinInt;
  env.isRank0Printer = mpiIsRank0Printer;
  env.ctx = nullptr;

  CaseResult res;
  if (param.tc.group == "stream_churn") {
    res = runStreamChurn(param.tc, &env);
  } else if (param.tc.group == "stream_ordering") {
    res = runStreamOrdering(param.tc, &env);
  } else {
    res = runOneCase(param.tc, &env);
  }

  if (res.status == CASE_SKIP) {
    env.barrier(&env);
#if defined(GTEST_SKIP)
    GTEST_SKIP() << ((res.skipReason != nullptr) ? res.skipReason : "skipped");
    return;
#else
    recordFallbackSkip(res.skipReason);
    return;
#endif
  }

  if (res.status == CASE_FAIL) ADD_FAILURE() << ((res.failReason != nullptr) ? res.failReason : "case failed");
  env.barrier(&env);
}

// The mixed-communicator regression intentionally selects no matrix cases.
// Allow the standalone group test to run without GoogleTest treating that as
// an uninstantiated parameterized suite failure.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(BasicApiMpiTest);

INSTANTIATE_TEST_CASE_P(Matrix, BasicApiMpiTest, ::testing::ValuesIn(selectedParams()), gtestCaseName);

static int initMpiRuntime() {
  TEST_CUDACHECK(cudaGetDeviceCount(&gNumDevices));
  MPI_Comm localComm;
  int localRank = 0;
  MPICHECK(MPI_Comm_split_type(testMpiWorld(), MPI_COMM_TYPE_SHARED, gWorldRank, MPI_INFO_NULL, &localComm));
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_free(&localComm));
  gDevice = localRank % (gNumDevices > 0 ? gNumDevices : 1);
  TEST_CUDACHECK(cudaSetDevice(gDevice));

  ncclUniqueId uid;
  if (gWorldRank == 0) TEST_NCCLCHECK(ncclGetUniqueId(&uid));
  MPICHECK(MPI_Bcast(&uid, sizeof(uid), testMpiByte(), 0, testMpiWorld()));

  TEST_NCCLCHECK(ncclCommInitRank(&gComm, gWorldSize, uid, gWorldRank));

  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  gBufferBytes = computeMaxBufferBytes(cases, gWorldSize);

  const char* initialAlgorithm = requestedAlgorithmEnv();
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — captured once during MPI setup
  const char* initialUseInternalStreamsEnv = getenv("NCCL_RESHARD_USE_INTERNAL_STREAMS");
  if (initialUseInternalStreamsEnv != nullptr) {
    gInitialUseInternalStreamsEnvSet = true;
    gInitialUseInternalStreamsEnv = initialUseInternalStreamsEnv;
  }
  basicApiConfigureReshardEnv(gCli, initialAlgorithm);
  TEST_NCCLCHECK(ncclM2nInit(&gM2nHandle, NULL));
  gActiveAlgorithm = initialAlgorithm;
  gActiveStreamPoolMode = -1;

  TEST_CUDACHECK(cudaStreamCreate(&gStream));
  TEST_NCCLCHECK(ncclMemAlloc(&gBuffer, gBufferBytes));

  if (basicApiRunAllApis(gCli) || basicApiRequestedApi(gCli) == ApiKind::Default) {
    gCopyBufferBytes = gBufferBytes;
    TEST_CUDACHECK(cudaMalloc(&gCopyBuffer, gCopyBufferBytes));
  }

  if (gWorldRank == 0) {
    std::vector<MpiParam> params = selectedParams();
    basicApiPrintRuntimeSummary("basic_api_test_mpi (gtest)", gWorldSize, gNumDevices, gCli, gBufferBytes, "num_tests",
                                params.size(), true);
  }
  return 0;
}

static void shutdownMpiRuntime() {
  if (gStream != nullptr) TEST_CUDACHECK(cudaStreamSynchronize(gStream));
  if (gGroupStreamA != nullptr) TEST_CUDACHECK(cudaStreamSynchronize(gGroupStreamA));
  if (gGroupStreamB != nullptr) TEST_CUDACHECK(cudaStreamSynchronize(gGroupStreamB));
  TEST_NCCLCHECK(ncclM2nFinalize(gM2nHandle));
  gM2nHandle = nullptr;
  if (gGroupBufferB != nullptr) TEST_NCCLCHECK(ncclMemFree(gGroupBufferB));
  if (gGroupStreamB != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gGroupStreamB));
  if (gGroupCommB != nullptr) TEST_NCCLCHECK(ncclCommDestroy(gGroupCommB));
  if (gGroupBufferA != nullptr) TEST_NCCLCHECK(ncclMemFree(gGroupBufferA));
  if (gGroupStreamA != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gGroupStreamA));
  if (gGroupCommA != nullptr) TEST_NCCLCHECK(ncclCommDestroy(gGroupCommA));
  if (gCopyBuffer != nullptr) {
    TEST_CUDACHECK(cudaFree(gCopyBuffer));
  }
  gCopyBuffer = nullptr;
  if (gBuffer != nullptr) TEST_NCCLCHECK(ncclMemFree(gBuffer));
  if (gStream != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gStream));
  if (gComm != nullptr) TEST_NCCLCHECK(ncclCommDestroy(gComm));
}

} // namespace

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(testMpiWorld(), &gWorldRank));
  MPICHECK(MPI_Comm_size(testMpiWorld(), &gWorldSize));

  gCli = basicApiParseCli(argc, argv, "mpirun -np <N> %s [options] [--gtest_* flags]", false, true);
  gCases = buildAllTestCases();

  if (gCli.listOnly) {
    basicApiPrintCaseList(gCases, gCli, gWorldRank == 0);
    MPICHECK(MPI_Finalize());
    return 0;
  }

  ::testing::InitGoogleTest(&argc, argv);
  if (gWorldRank != 0) {
    FILE* devnull = freopen("/dev/null", "w", stdout);
    if (devnull == nullptr) fprintf(stderr, "rank %d: failed to redirect stdout\n", gWorldRank);
  }

  if (::testing::GTEST_FLAG(list_tests)) {
    int rc = RUN_ALL_TESTS();
    MPICHECK(MPI_Finalize());
    return rc;
  }

  int initRc = initMpiRuntime();
  if (initRc != 0) {
    MPICHECK(MPI_Finalize());
    return initRc;
  }

  int rc = RUN_ALL_TESTS();
#if !defined(GTEST_SKIP)
  basicApiPrintFallbackSkipSummary(gSkippedCases, gWorldRank == 0);
#endif
  shutdownMpiRuntime();
  MPICHECK(MPI_Finalize());
  return rc;
}
