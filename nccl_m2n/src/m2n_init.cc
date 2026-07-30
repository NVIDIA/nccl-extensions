/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

#include "nccl.h"
#include "nccl_m2n.h"
#include "m2n_checks.h"
#include "m2n_env_parse.h"
#include "reshard_types.h"
#include "m2n_log.h"
#include "reshard_internal.h"

static std::mutex gM2nRuntimeMutex;
static std::weak_ptr<ncclM2nRuntime> gM2nRuntime;
static std::mutex gM2nDefaultHandleMutex;
static std::shared_ptr<ncclM2nHandleState> gM2nDefaultHandle;
static std::mutex gM2nHandleTableMutex;
static uintptr_t gNextM2nHandleId = 1;
static std::unordered_map<ncclM2nHandle_t, std::shared_ptr<ncclM2nHandleState>> gM2nHandleTable;
static std::atomic<bool> gM2nProcessExiting{false};

struct M2nExitGuard {
  ~M2nExitGuard() {
    gM2nProcessExiting.store(true, std::memory_order_relaxed);
  }
};
static M2nExitGuard gM2nExitGuard;

// NCCL_RESHARD_STREAM_POOL_SIZE: max number of distinct (comm, dev)
// pairs the pool will hold (1:1 stream+event mapping).  Default 4.
// Values <= 0 disable the pool — default-stream callers then run on
// the user's default stream directly (legacy synchronizing behavior).
// Values above STREAM_POOL_MAX_SIZE are capped (with a warning).
// Invalid values likewise disable the pool.

void applyStreamPoolFromEnv() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time, single-thread on the caller
  const char* sizeEnv = getenv("NCCL_RESHARD_STREAM_POOL_SIZE");
  if (sizeEnv == nullptr) {
    return;
  }

  int n = 0;
  if (!parseM2nEnvInt(sizeEnv, &n)) {
    RESHARD_WARN(-1,
                 "NCCL_RESHARD_STREAM_POOL_SIZE='%s' is not a valid integer; "
                 "stream pool disabled — default-stream callers will run on the "
                 "user's default stream directly.",
                 sizeEnv);
    gReshardStreamPoolSize = 0;
    return;
  }

  if (n <= 0) {
    RESHARD_WARN(-1,
                 "NCCL_RESHARD_STREAM_POOL_SIZE='%s' (parsed as %d) <= 0; "
                 "stream "
                 "pool disabled — default-stream callers will run on the user's "
                 "default stream directly.",
                 sizeEnv, n);
    gReshardStreamPoolSize = 0;
  } else if (n > STREAM_POOL_MAX_SIZE) {
    RESHARD_WARN(-1,
                 "NCCL_RESHARD_STREAM_POOL_SIZE='%s' (parsed as %d) exceeds the "
                 "library max %d; capping to %d.",
                 sizeEnv, n, STREAM_POOL_MAX_SIZE, STREAM_POOL_MAX_SIZE);
    gReshardStreamPoolSize = STREAM_POOL_MAX_SIZE;
  } else {
    gReshardStreamPoolSize = n;
  }
}

static ncclResult_t copyHandleConfig(ncclM2nHandleState* handle, const ncclM2nConfig_t* config) {
  if (config == nullptr) {
    handle->config = NCCL_M2N_CONFIG_INITIALIZER;
    return ncclSuccess;
  }

  NCCL_M2N_CHECK(validateReshardConfigHeader(config));
  handle->config = *config;
  return ncclSuccess;
}

static void deleteHandleState(ncclM2nHandleState* handle) {
  // The OS reclaims abandoned process-lifetime state; CUDA teardown is unsafe
  // after main when the CUDA runtime may already be unloading.
  if (gM2nProcessExiting.load(std::memory_order_relaxed)) return;
  std::lock_guard<std::mutex> lock(gM2nRuntimeMutex);
  // Releasing the last state also releases the shared runtime and tears down
  // its process-global resources before a new runtime epoch can be created.
  delete handle;
}

static ncclResult_t createHandle(std::shared_ptr<ncclM2nHandleState>* handle) {
  try {
    *handle = std::shared_ptr<ncclM2nHandleState>(new ncclM2nHandleState(), deleteHandleState);
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t createRuntime(std::shared_ptr<ncclM2nRuntime>* runtime) {
  try {
    *runtime = std::make_shared<ncclM2nRuntime>();
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }
  return ncclSuccess;
}

static void resolveNumCtas() {
  const bool capNumCtas = gReshardMaxCta > 0 && gReshardMaxCta < DEFAULT_NUM_CTAS;
  gReshardNumCtas = capNumCtas ? gReshardMaxCta : DEFAULT_NUM_CTAS;
}

static ncclResult_t initializeRuntimeForFirstHandle(const ncclM2nHandleState* handle) {
  resetReshardRuntimeConfig();
  applyReshardConfig(&handle->config);
  applyReshardEnv();
  applyStreamPoolFromEnv();
  resolveNumCtas();
  return ncclSuccess;
}

static bool configMatches(const ncclM2nConfig_t& a, const ncclM2nConfig_t& b) {
  // Compare every public config field; update this when ncclM2nConfig_t grows.
  return a.size == b.size && a.magic == b.magic && a.version == b.version && a.maxCta == b.maxCta;
}

static ncclResult_t attachRuntime(ncclM2nHandleState* handle, bool warnOnConfigMismatch) {
  std::lock_guard<std::mutex> lock(gM2nRuntimeMutex);
  std::shared_ptr<ncclM2nRuntime> runtime = gM2nRuntime.lock();
  if (runtime == nullptr) {
    NCCL_M2N_CHECK(createRuntime(&runtime));
    runtime->firstConfig = handle->config;
    NCCL_M2N_CHECK(initializeRuntimeForFirstHandle(handle));
    gM2nRuntime = runtime;
  } else if (warnOnConfigMismatch && !configMatches(runtime->firstConfig, handle->config)) {
    RESHARD_WARN(-1, "ncclM2nInit: config ignored because runtime was already "
                     "configured by an earlier ncclM2nInit; finalize all handles "
                     "before reconfiguring.");
  }

  handle->runtime = runtime;
  return ncclSuccess;
}

static ncclResult_t createInitializedHandle(std::shared_ptr<ncclM2nHandleState>* handle,
                                            const ncclM2nConfig_t* config, bool warnOnConfigMismatch) {
  std::shared_ptr<ncclM2nHandleState> newHandle;
  NCCL_M2N_CHECK(createHandle(&newHandle));
  NCCL_M2N_CHECK(copyHandleConfig(newHandle.get(), config));
  NCCL_M2N_CHECK(attachRuntime(newHandle.get(), warnOnConfigMismatch));
  *handle = std::move(newHandle);
  return ncclSuccess;
}

static ncclResult_t registerExplicitHandle(const std::shared_ptr<ncclM2nHandleState>& handle,
                                           ncclM2nHandle_t* token) {
  ncclM2nHandle_t newToken = nullptr;
  try {
    std::lock_guard<std::mutex> lock(gM2nHandleTableMutex);
    if (gNextM2nHandleId == 0) {
      return ncclSystemError;
    }
    newToken = reinterpret_cast<ncclM2nHandle_t>(gNextM2nHandleId++);
    if (!gM2nHandleTable.emplace(newToken, handle).second) {
      return ncclSystemError;
    }
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }
  *token = newToken;
  return ncclSuccess;
}

static ncclResult_t destroyExplicitHandle(ncclM2nHandle_t handle) {
  std::shared_ptr<ncclM2nHandleState> ownedHandle;
  {
    std::lock_guard<std::mutex> lock(gM2nHandleTableMutex);
    auto it = gM2nHandleTable.find(handle);
    if (it == gM2nHandleTable.end()) {
      return ncclInvalidArgument;
    }
    ownedHandle = std::move(it->second);
    gM2nHandleTable.erase(it);
  }
  ownedHandle.reset();
  return ncclSuccess;
}

ncclM2nRuntime::~ncclM2nRuntime() {
  cacheFinalize();
  transposeBufferFinalize();
  resetReshardRuntimeConfig();
}

extern "C" {
ncclResult_t ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) {
  if (handle == nullptr) {
    return ncclInvalidArgument;
  }
  *handle = nullptr;
  std::shared_ptr<ncclM2nHandleState> newHandle;
  NCCL_M2N_CHECK(createInitializedHandle(&newHandle, config, true));
  ncclResult_t result = registerExplicitHandle(newHandle, handle);
  if (result != ncclSuccess) {
    newHandle.reset();
  }
  return result;
}

ncclResult_t ncclM2nFinalize(ncclM2nHandle_t handle) {
  if (handle != nullptr) {
    return destroyExplicitHandle(handle);
  }

  std::shared_ptr<ncclM2nHandleState> defaultHandle;
  {
    std::lock_guard<std::mutex> lock(gM2nDefaultHandleMutex);
    defaultHandle = std::move(gM2nDefaultHandle);
  }
  defaultHandle.reset();
  return ncclSuccess;
}

} // extern "C"

ncclResult_t acquireM2nHandle(ncclM2nHandle_t token, std::shared_ptr<ncclM2nHandleState>* handle) {
  if (handle == nullptr) {
    return ncclInvalidArgument;
  }
  *handle = nullptr;

  if (token != nullptr) {
    std::lock_guard<std::mutex> lock(gM2nHandleTableMutex);
    auto it = gM2nHandleTable.find(token);
    if (it == gM2nHandleTable.end()) {
      return ncclInvalidArgument;
    }
    *handle = it->second;
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(gM2nDefaultHandleMutex);
  if (gM2nDefaultHandle == nullptr) {
    ncclM2nConfig_t config = NCCL_M2N_CONFIG_INITIALIZER;
    NCCL_M2N_CHECK(createInitializedHandle(&gM2nDefaultHandle, &config, false));
  }
  *handle = gM2nDefaultHandle;
  return ncclSuccess;
}
