/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
#ifndef NCCL_ND_H_
#define NCCL_ND_H_

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define NCCL_ND_API __attribute__((visibility("default")))
#else
#define NCCL_ND_API
#endif

/* Experimental C API; a stable ABI is not yet guaranteed. */
#define NCCL_ND_MAJOR 0
#define NCCL_ND_MINOR 2
#define NCCL_ND_PATCH 0
#define NCCL_ND_VERSION_CODE (NCCL_ND_MAJOR * 10000 + NCCL_ND_MINOR * 100 + NCCL_ND_PATCH)

typedef struct ncclNdHandle* ncclNdHandle_t;

/* Caller-owned descriptor. sizes and optional strides are host arrays in
 * elements; NULL strides mean dense row-major storage. Explicit strides
 * must match that layout.
 * Input: data == NULL; win_hdl is registered on the handle's comm with
 * NCCL_WIN_COLL_SYMMETRIC; win_offset is in bytes.
 * Output: data points to sufficient device storage; win_hdl == NULL and
 * win_offset == 0. Register output storage separately to reuse it as input.
 * Host metadata is read during the call or capture. Keep device storage and
 * windows alive until completion and through any possible Graph replay.
 * Matching metadata and symmetric registration on all ranks are caller
 * requirements; this API cannot verify the window registration flag.
 */
typedef struct ncclNdTensor {
  unsigned int ndim;
  ncclDataType_t datatype;
  void* data;
  ncclWindow_t win_hdl;
  uint64_t win_offset;
  const size_t* sizes;
  const size_t* strides;
} ncclNdTensor_t;

// Return the NCCL ND library version.
// Executes immediately, also inside an open ND group, and does not participate
// in or affect the group.
//
// Arguments:
//   version - [OUT] Receives NCCL_ND_VERSION_CODE.
//
// Returns:
//   ncclResult_t error code.
NCCL_ND_API ncclResult_t ncclNdGetVersion(int* version);

// Create one handle per rank on the current communicator device.
// Call collectively outside CUDA Graph capture. The caller owns comm, which
// must form one LSA team and outlive the handle. No payload is allocated.
// Unsupported inside an open ND group: returns ncclInvalidUsage, sets *handle
// to NULL and invalidates the group (see ncclNdGroupStart).
//
// Arguments:
//   handle - [OUT] Receives the new handle.
//   comm   - [IN]  NCCL communicator shared by all participating ranks.
//
// Returns:
//   ncclResult_t error code.
NCCL_ND_API ncclResult_t ncclNdCreate(ncclNdHandle_t* handle, ncclComm_t comm);

// Destroy a handle after all its work and Graph replays complete.
// Call outside CUDA Graph capture. Caller-owned comm, windows, and storage are
// not freed. Unsupported inside an open ND group, for any handle including
// NULL: returns ncclInvalidUsage, destroys nothing and invalidates the group.
//
// Arguments:
//   handle - [IN] Handle to destroy; NULL is accepted.
//
// Returns:
//   ncclResult_t error code.
NCCL_ND_API ncclResult_t ncclNdDestroy(ncclNdHandle_t handle);

// Redistribute Ulysses sequence/head shards across P >= 2 ranks:
//   Y_d = concat_s(split(X_s, P, scatter_dim)[d], gather_dim).
// For [S, H], supported (scatter_dim, gather_dim) pairs are (1, 0) and (0, 1);
// for [B, S, H, D], they are (2, 1) and (1, 2). Source-rank order, B, and D
// are preserved. Other axes and in-place execution are unsupported.
//
// Arguments:
//   handle      - [IN,OUT] Handle shared by this rank's serialized calls.
//   input       - [IN]     Window-backed input descriptor; data must be NULL.
//   output      - [IN,OUT] Descriptor with preallocated device output storage;
//                          win_hdl must be NULL.
//   scatter_dim - [IN]     Physical tensor axis to split across ranks.
//   gather_dim  - [IN]     Physical tensor axis to concatenate in rank order.
//   stream      - [IN]     Explicit nondefault CUDA stream.
//
// Returns:
//   ncclResult_t error code. Success means enqueued; stream completion makes
//   output ready and all peer reads complete. A rejection before submission
//   enqueues nothing on this rank. A failure during submission
//   (ncclUnhandledCudaError) can leave partial collective work and is fatal
//   for the communicator; there is no rollback or eager fallback.
//   Inside an open ND group, the call is validated and recorded instead;
//   success means recorded, and the outermost ncclNdGroupEnd enqueues the work.
//   A rejected call invalidates the group.
NCCL_ND_API ncclResult_t ncclNdAllToAll(ncclNdHandle_t handle, const ncclNdTensor_t* input, ncclNdTensor_t* output,
                                        int scatter_dim, int gather_dim, cudaStream_t stream);

// Begin recording ncclNdAllToAll calls, e.g. for Q, K and V. Group state is
// local to the calling host thread. Groups nest; only the outermost
// ncclNdGroupEnd enqueues.
//
// The first recorded call binds the group's handle, stream and capture state
// (outside capture, or one capture session). A later call with another handle,
// stream or capture state fails with ncclInvalidUsage; the outermost end checks
// the capture state again. Beginning or ending capture after the first call is
// unsupported, and is detected only while still in effect at a later call or
// the outermost end.
//
// Inside a group, ncclNdAllToAll is validated and recorded, ncclNdGetVersion
// runs immediately, and ncclNdCreate and ncclNdDestroy fail with
// ncclInvalidUsage.
//
// A rejected ncclNdAllToAll, ncclNdCreate or ncclNdDestroy fails the whole
// local group: every later ncclNdAllToAll and the outermost end return the
// first error, and nothing is enqueued. Keep start and end balanced after an
// error. A rejection is local, not a distributed rollback: every rank must
// record the same calls in the same order with matching metadata, or peers
// wait in a barrier.
//
// Grouped calls must be independent: no output may overlap another call's
// input or output (not checked), and no call may consume another's output.
// Other work issued on the stream while the group is open runs before the
// group's work; issue dependent work after the outermost end.
//
// ND groups and native ncclGroupStart/ncclGroupEnd are independent. NCCL
// communication inside an ND group, and ND calls inside an NCCL group, are
// unsupported and not detected.
//
// Calls are validated and their copies resolved when recorded, so descriptors
// and size arrays need not outlive the call.
//
// Returns:
//   ncclResult_t error code.
NCCL_ND_API ncclResult_t ncclNdGroupStart(void);

// Close one group level. An inner end returns ncclSuccess and enqueues
// nothing, even if the group failed. The outermost end, called with the
// handle's device current, enqueues every recorded call on the group's stream
// and clears the group; stream completion makes all outputs ready and inputs
// reusable. Copy order and synchronization within a group are unspecified
// (currently one entry and one exit barrier). An empty group enqueues nothing.
// If the group failed, the outermost end returns the first error and enqueues
// nothing. A submission failure (ncclUnhandledCudaError) can leave partial
// collective work, and the handle and communicator are not reusable; see
// README.md for recovery.
//
// Returns:
//   ncclResult_t error code. ncclInvalidUsage when no group is open or the
//   current device is not the handle's device.
NCCL_ND_API ncclResult_t ncclNdGroupEnd(void);

#ifdef __cplusplus
}
#endif
#endif
