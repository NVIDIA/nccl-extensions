/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_buffer.h — host-side API for the staging-buffer infrastructure
 * that backs ncclReshard.
 *
 *   stagingBufferInit       — allocate a symmetric staging pool with
 *                             ncclMemAlloc + zero its control region (NOT
 *                             collective; window registration is the
 *                             caller's responsibility, see reshard_cache).
 *   stagingPrepareTransfer  — translate a host-side
 *                             StagingTransferDescriptor into a packed
 *                             StagingKernelParams ready to upload.
 *   stagingBufferFinalize   — free the pool + matching device-side params
 *                             buffer.
 *
 * No MPI dependency.  Logging picks up `desc->myWorldRank` so we can
 * tag traces without going through MPI_Comm_rank.
 ************************************************************************/

#ifndef NCCL_STAGING_BUFFER_H_
#define NCCL_STAGING_BUFFER_H_

#include "nccl.h"
#include "staging_types.h"

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Host-side handle returned by stagingBufferInit().
 *
 * Holds the staging buffer allocation and a pre-allocated device-side
 * StagingKernelParams scratch slot (ncclReshard host entry uploads the
 * per-call params block here, avoiding a hot-path cudaMalloc).
 * Window registrations live in reshard_cache so that each (comm,
 * staging_buffer) pair gets exactly one collective registration.
 * ====================================================================*/
struct StagingBufferState {
  void* buffer;
  size_t totalSize;
  int numChannels;
  size_t channelSize;
  size_t chunkSize;
  bool initialized;

  StagingKernelParams* devParams; /* cudaMalloc'd once, reused per launch */
};

struct StagingBufferConfig {
  int numChannels;
  size_t channelSize;
  size_t chunkSize;
};

/* Toggle [STAGING] verbose logging at runtime (parallels
 * NCCL_RESHARD_LOG_LEVEL=DEBUG). */
void stagingSetVerbose(bool verbose);

StagingBufferConfig stagingBufferConfigFromEnv();

/* Allocate the staging pool + per-call devParams slot.
 * NOT collective. Sizes default from reshard_limits.h, with internal
 * staging-pool overrides read during initialization. */
ncclResult_t stagingBufferInit(StagingBufferState* state);

/* Derive one rank-uniform chunk size from the configured staging geometry and
 * the exact maximum peer group computed from the shared tensor descriptors. */
ncclResult_t stagingResolveEffectiveChunkSize(const StagingBufferState* state, size_t maxPeerGroupSize,
                                              size_t* effectiveChunkSize);

/* Translate a host descriptor into device-ready kernel params.  The
 * caller is expected to have already registered both windows on the
 * staging buffer (see ncclReshard host entry). */
ncclResult_t stagingPrepareTransfer(const StagingBufferState* state, const StagingTransferDescriptor* desc,
                                    ncclWindow_t rdmaWindow, ncclWindow_t lsaWindow, size_t effectiveChunkSize,
                                    StagingKernelParams* params);

/* Free the staging buffer + devParams slot.  Window deregistration is
 * driven by the global reshard_cache teardown. */
ncclResult_t stagingBufferFinalize(StagingBufferState* state);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_STAGING_BUFFER_H_ */
