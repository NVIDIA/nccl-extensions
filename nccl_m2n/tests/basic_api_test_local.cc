/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_local — gtest-parametrized single-process driver for the
 * basic_api matrix. Each TestCase is one gtest parameter; the fixture still
 * runs one pthread per rank over ncclCommInitAll.
 *
 * Useful for dev workstations: no MPI install, no mpirun, no scheduler.
 * Cases requiring more ranks than available GPUs report as SKIP/no-op.
 ************************************************************************/

#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_m2n.h"
#include "basic_api_test_core.h"
#include "reshard_internal.h"

namespace {

/* ======================================================================
 * Bootstrap-specific shared context.
 * ====================================================================*/

struct LocalCtx {
  pthread_barrier_t barrier;
  std::vector<int> aggBuf; /* size == worldSize */
  pthread_mutex_t printMu; /* serializes optional verbose prints */
};

/* ======================================================================
 * TestEnv hooks.
 * ====================================================================*/

static void localBarrier(TestEnv* env) {
  LocalCtx* c = (LocalCtx*)env->ctx;
  pthread_barrier_wait(&c->barrier);
}

static int localAllreduceMinInt(TestEnv* env, int local) {
  LocalCtx* c = (LocalCtx*)env->ctx;
  c->aggBuf[env->rank] = local;
  pthread_barrier_wait(&c->barrier);
  int m = c->aggBuf[0];
  for (int i = 1; i < env->worldSize; ++i)
    if (c->aggBuf[i] < m) m = c->aggBuf[i];
  pthread_barrier_wait(&c->barrier); /* ensure all read aggBuf before next overwrite */
  return m;
}

static bool localIsRank0Printer(TestEnv* env) {
  return env->rank == 0;
}

/* ======================================================================
 * gtest parameter registry state.
 * ====================================================================*/

struct LocalParam {
  TestCase tc;
  ApiKind api = ApiKind::Window;
};

[[maybe_unused]] static void printTo(const LocalParam& param, std::ostream* os) {
  *os << (param.api == ApiKind::Default ? "default" : "window") << ":" << param.tc.name;
}

static BasicApiCliArgs gCli;
static std::vector<TestCase> gCases;
static int gWorldSize = 0;
static int gNumDevices = 0;
static size_t gBufferBytes = 4096;
static std::vector<ncclComm_t> gComms;
static std::vector<ncclM2nHandle_t> gM2nHandles;
#if !defined(GTEST_SKIP)
static int gSkippedCases = 0;
#endif

static void* callInvalidReshard(void* resultPtr) {
  ncclResult_t* result = static_cast<ncclResult_t*>(resultPtr);
  *result = ncclReshard(nullptr, nullptr, nullptr, nullptr, nullptr);
  return nullptr;
}

TEST(NcclReshardStackTest, RejectsInvalidCallOnTwoMiBThread) {
  pthread_attr_t attr;
  ASSERT_EQ(0, pthread_attr_init(&attr));
  ASSERT_EQ(0, pthread_attr_setstacksize(&attr, 2 * 1024 * 1024));

  ncclResult_t result = ncclSuccess;
  pthread_t thread;
  ASSERT_EQ(0, pthread_create(&thread, &attr, callInvalidReshard, &result));
  ASSERT_EQ(0, pthread_attr_destroy(&attr));

  ASSERT_EQ(0, pthread_join(thread, nullptr));
  EXPECT_EQ(ncclInvalidArgument, result);
}

static std::vector<LocalParam> selectedLocalParams() {
  std::vector<LocalParam> params;
  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  std::vector<ApiKind> apis;
  if (basicApiRunAllApis(gCli)) {
    apis = {ApiKind::Window, ApiKind::Default};
  } else {
    apis = {basicApiRequestedApi(gCli)};
  }
  for (ApiKind api : apis) {
    for (const TestCase& tc : cases) {
      params.push_back(LocalParam{tc, api});
    }
  }
  return params;
}

static std::string gtestCaseName(const ::testing::TestParamInfo<LocalParam>& info) {
  std::string prefix = (info.param.api == ApiKind::Default) ? "default" : "window";
  return basicApiGtestCaseName(info.param.tc.name, info.index, prefix.c_str());
}

/* ======================================================================
 * Per-thread payload.
 * ====================================================================*/

struct ThreadArg {
  int rank;
  int worldSize;
  int device;
  ncclComm_t comm;
  ncclM2nHandle_t m2nHandle;
  LocalCtx* ctx;
  const TestCase* tc;
  ApiKind api;
  bool verbose;

  CaseStatus status;
  char skipReason[192];
  char failReason[192];
};

static void* threadMain(void* p) {
  ThreadArg* a = (ThreadArg*)p;

  TEST_CUDACHECK(cudaSetDevice(a->device));

  cudaStream_t stream;
  TEST_CUDACHECK(cudaStreamCreate(&stream));

  void* buffer = nullptr;
  TEST_NCCLCHECK(ncclMemAlloc(&buffer, gBufferBytes));

  void* copyBuffer = nullptr;
  if (a->api == ApiKind::Default) {
    TEST_CUDACHECK(cudaMalloc(&copyBuffer, gBufferBytes));
  }

  TestEnv env{};
  env.rank = a->rank;
  env.worldSize = a->worldSize;
  env.device = a->device;
  env.comm = a->comm;
  env.stream = stream;
  env.m2nHandle = a->m2nHandle;
  env.buffer = buffer;
  env.bufferBytes = gBufferBytes;
  env.copyBuffer = copyBuffer;
  env.copyBufferBytes = (copyBuffer != nullptr) ? gBufferBytes : 0;
  env.apiKind = a->api;
  env.expectPackWindow = strcmp(gCli.copyAlgorithm, "packwindow") == 0;
  env.verbose = a->verbose;
  env.barrier = localBarrier;
  env.allreduceMinInt = localAllreduceMinInt;
  env.isRank0Printer = localIsRank0Printer;
  env.ctx = a->ctx;

  CaseResult res = runOneCase(*a->tc, &env);
  a->status = res.status;
  if (res.skipReason != nullptr) snprintf(a->skipReason, sizeof(a->skipReason), "%s", res.skipReason);
  if (res.failReason != nullptr) snprintf(a->failReason, sizeof(a->failReason), "%s", res.failReason);

  TEST_CUDACHECK(cudaStreamSynchronize(stream));

  /* Keep rank-local buffers alive until every thread has left runOneCase. */
  pthread_barrier_wait(&a->ctx->barrier);

  if (copyBuffer != nullptr) {
    TEST_CUDACHECK(cudaFree(copyBuffer));
  }
  TEST_NCCLCHECK(ncclMemFree(buffer));
  TEST_CUDACHECK(cudaStreamDestroy(stream));
  return nullptr;
}

struct LocalCaseResult {
  CaseStatus status;
  std::string reason;
};

static LocalCaseResult runLocalCase(const TestCase& tc, ApiKind api) {
  LocalCtx ctx{};
  ctx.aggBuf.assign(gWorldSize, 0);
  pthread_mutex_init(&ctx.printMu, nullptr);
  pthread_barrier_init(&ctx.barrier, nullptr, gWorldSize);

  std::vector<pthread_t> tids(gWorldSize);
  std::vector<ThreadArg> args(gWorldSize);
  for (int r = 0; r < gWorldSize; ++r) {
    args[r].rank = r;
    args[r].worldSize = gWorldSize;
    args[r].device = r;
    args[r].comm = gComms[r];
    args[r].m2nHandle = gM2nHandles[r];
    args[r].ctx = &ctx;
    args[r].tc = &tc;
    args[r].api = api;
    args[r].verbose = gCli.verbose;
    args[r].status = CASE_FAIL;
    args[r].skipReason[0] = '\0';
    args[r].failReason[0] = '\0';
    int rc = pthread_create(&tids[r], nullptr, threadMain, &args[r]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create rank=%d failed: %d\n", r, rc);
      _Exit(1);
    }
  }
  for (int r = 0; r < gWorldSize; ++r) pthread_join(tids[r], nullptr);

  pthread_barrier_destroy(&ctx.barrier);
  pthread_mutex_destroy(&ctx.printMu);

  for (const ThreadArg& arg : args) {
    if (arg.status == CASE_FAIL) {
      return LocalCaseResult{
        CASE_FAIL,
        (arg.failReason[0] != '\0') ? arg.failReason : "rank reported failure",
      };
    }
  }
  for (const ThreadArg& arg : args) {
    if (arg.status == CASE_SKIP) {
      return LocalCaseResult{
        CASE_SKIP,
        (arg.skipReason[0] != '\0') ? arg.skipReason : "rank reported skip",
      };
    }
  }
  return LocalCaseResult{CASE_PASS, ""};
}

class BasicApiLocalTest : public ::testing::TestWithParam<LocalParam> {};

/* Exercises cross-rank agreement on the split decision: the two ranks supply
 * deliberately different local values and must receive the same reduced
 * result. The helper it drives is test-only instrumentation, so the whole
 * case compiles away in a build without it. */
#ifdef NCCL_M2N_TESTING
TEST(LocalRankProgressTest, CollectiveSectionsAdmitPeerRanks) {
  if (gCli.filter == nullptr || strcmp(gCli.filter, "local_collective_progress") != 0) {
#if defined(GTEST_SKIP)
    GTEST_SKIP() << "run with --filter local_collective_progress";
#else
    return;
#endif
  }
  if (gWorldSize != 2) {
#if defined(GTEST_SKIP)
    GTEST_SKIP() << "requires exactly two local ranks";
#else
    return;
#endif
  }

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  ncclResult_t results[2] = {};
  int reduced[2] = {};
  cudaError_t cudaResults[2] = {};
  std::thread threads[2];
  for (int rank = 0; rank < 2; rank++) {
    threads[rank] = std::thread([&, rank] {
      cudaResults[rank] = cudaSetDevice(rank);
      cudaStream_t stream = nullptr;
      if (cudaResults[rank] == cudaSuccess) {
        cudaResults[rank] = cudaStreamCreate(&stream);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (cudaResults[rank] == cudaSuccess) {
        M2nApiLock apiLock;
        results[rank] = reshardTestBroadcastMaxInt(gComms[rank], stream, rank, &reduced[rank]);
        cudaResults[rank] = cudaStreamDestroy(stream);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(ncclSuccess, results[0]);
  EXPECT_EQ(ncclSuccess, results[1]);
  EXPECT_EQ(1, reduced[0]);
  EXPECT_EQ(1, reduced[1]);
  EXPECT_EQ(cudaSuccess, cudaResults[0]);
  EXPECT_EQ(cudaSuccess, cudaResults[1]);
}
#endif  /* NCCL_M2N_TESTING */

TEST_P(BasicApiLocalTest, Reshard) {
  const LocalParam& param = GetParam();
  SCOPED_TRACE(param.tc.name);

  LocalCaseResult res = runLocalCase(param.tc, param.api);
  if (res.status == CASE_SKIP) {
#if defined(GTEST_SKIP)
    GTEST_SKIP() << res.reason;
    return;
#else
    basicApiRecordFallbackSkip(&gSkippedCases, res.reason.c_str(), true);
    return;
#endif
  }

  EXPECT_EQ(CASE_PASS, res.status) << res.reason;
}

INSTANTIATE_TEST_CASE_P(Matrix, BasicApiLocalTest, ::testing::ValuesIn(selectedLocalParams()), gtestCaseName);

static int initLocalRuntime() {
  TEST_CUDACHECK(cudaGetDeviceCount(&gNumDevices));
  if (gNumDevices <= 0) {
    fprintf(stderr, "No CUDA devices visible.\n");
    return 2;
  }

  gWorldSize = gCli.requestedRanks > 0 ? gCli.requestedRanks : gNumDevices;
  if (gWorldSize > gNumDevices) {
    fprintf(stderr,
            "Requested %d ranks but only %d CUDA device(s) visible -- "
            "clamping.\n",
            gWorldSize, gNumDevices);
    gWorldSize = gNumDevices;
  }

  basicApiConfigureReshardEnv(gCli, basicApiRequestedAlgorithmEnv(gCli, true));

  std::vector<int> devlist(gWorldSize);
  gComms.assign(gWorldSize, nullptr);
  gM2nHandles.assign(gWorldSize, nullptr);
  for (int i = 0; i < gWorldSize; ++i) devlist[i] = i;

  TEST_NCCLCHECK(ncclCommInitAll(gComms.data(), gWorldSize, devlist.data()));
  for (int i = 0; i < gWorldSize; ++i) {
    TEST_NCCLCHECK(ncclM2nInit(&gM2nHandles[i], NULL));
  }

  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  gBufferBytes = computeMaxBufferBytes(cases, gWorldSize);

  std::vector<LocalParam> localParams = selectedLocalParams();
  basicApiPrintRuntimeSummary("basic_api_test_local (gtest, no MPI)", gWorldSize, gNumDevices, gCli, gBufferBytes,
                              "num_tests", localParams.size(), true);
  return 0;
}

static void shutdownLocalRuntime() {
  for (ncclM2nHandle_t m2nHandle : gM2nHandles) {
    if (m2nHandle != nullptr) {
      TEST_NCCLCHECK(ncclM2nFinalize(m2nHandle));
    }
  }
  gM2nHandles.clear();
  for (ncclComm_t comm : gComms) {
    if (comm != nullptr) {
      TEST_NCCLCHECK(ncclCommDestroy(comm));
    }
  }
  gComms.clear();
}

} // namespace

int main(int argc, char** argv) {
  gCli = basicApiParseCli(argc, argv, "%s [options] [--gtest_* flags]", true, false);
  gCases = buildAllTestCases();

  if (gCli.listOnly) {
    basicApiPrintCaseList(gCases, gCli, true);
    return 0;
  }

  ::testing::InitGoogleTest(&argc, argv);
  if (::testing::GTEST_FLAG(list_tests)) return RUN_ALL_TESTS();

  int initRc = initLocalRuntime();
  if (initRc != 0) return initRc;

  int rc = RUN_ALL_TESTS();
#if !defined(GTEST_SKIP)
  basicApiPrintFallbackSkipSummary(gSkippedCases, true);
#endif
  shutdownLocalRuntime();
  return rc;
}
