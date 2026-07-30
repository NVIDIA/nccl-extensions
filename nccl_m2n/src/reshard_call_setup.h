/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_RESHARD_CALL_SETUP_H_
#define NCCL_M2N_RESHARD_CALL_SETUP_H_

#include "cuda_runtime.h"
#include "nccl.h"

#include "m2n_checks.h"
#include "reshard_internal.h"

struct ReshardWorkStream {
  cudaStream_t stream;
  cudaEvent_t readyEvent;
  cudaEvent_t doneEvent;
};

static inline bool reshardAcquiredPoolSlot(const ReshardWorkStream* work) {
  return work->doneEvent != nullptr;
}

static inline ncclResult_t reshardMatchCommCudaDevice(ncclComm_t comm, int* currentCudaDev,
                                                      ncclCommProperties* commProps, ncclResult_t* propsResult) {
  NCCL_M2N_CUDACHECK(cudaGetDevice(currentCudaDev));
  *commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  *propsResult = ncclCommQueryProperties(comm, commProps);
  if (*propsResult == ncclSuccess && *currentCudaDev != commProps->cudaDev) {
    NCCL_M2N_CUDACHECK(cudaSetDevice(commProps->cudaDev));
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardSetupWorkStream(ncclComm_t comm, cudaStream_t callerStream, int currentCudaDev,
                                                  ncclResult_t propsResult, const ncclCommProperties* commProps,
                                                  ReshardWorkStream* work) {
  work->stream = callerStream;
  work->readyEvent = nullptr;
  work->doneEvent = nullptr;

  const bool isDefaultStream =
    (callerStream == nullptr || callerStream == cudaStreamLegacy || callerStream == cudaStreamPerThread);
  const bool wantPool = isDefaultStream && reshardGetStreamPoolSize() > 0;
  if (wantPool) {
    const int dev = (propsResult == ncclSuccess) ? commProps->cudaDev : currentCudaDev;
    NCCL_M2N_CHECK(streamPoolAcquire(comm, dev, &work->stream, &work->readyEvent, &work->doneEvent));
    if (work->stream == nullptr) {
      work->stream = callerStream;
    } else {
      NCCL_M2N_CUDACHECK(cudaEventRecord(work->readyEvent, callerStream));
      NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(work->stream, work->readyEvent, 0));
    }
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardCompleteWorkStream(cudaStream_t callerStream, const ReshardWorkStream* work) {
  if (reshardAcquiredPoolSlot(work)) {
    NCCL_M2N_CUDACHECK(cudaEventRecord(work->doneEvent, work->stream));
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(callerStream, work->doneEvent, 0));
  }
  return ncclSuccess;
}

class ReshardWorkStreamCompletion {
public:
  ReshardWorkStreamCompletion(cudaStream_t callerStream, const ReshardWorkStream* work)
    : callerStream_(callerStream), work_(work) {}
  ReshardWorkStreamCompletion(const ReshardWorkStreamCompletion&) = delete;
  ReshardWorkStreamCompletion& operator=(const ReshardWorkStreamCompletion&) = delete;
  ~ReshardWorkStreamCompletion() {
    if (bActive_) {
      // Preserve the caller's primary error; success paths return complete() directly.
      (void)complete();
    }
  }

  ncclResult_t complete() {
    if (!bActive_) {
      return ncclSuccess;
    }
    bActive_ = false;
    return reshardCompleteWorkStream(callerStream_, work_);
  }

private:
  cudaStream_t callerStream_;
  const ReshardWorkStream* work_;
  bool bActive_ = true;
};

#endif // NCCL_M2N_RESHARD_CALL_SETUP_H_
