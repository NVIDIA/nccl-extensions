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
#define NCCL_ND_MINOR 1
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
// not freed.
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
//   output ready and all peer reads complete. An enqueue failure can leave
//   partial work; there is no rollback or eager fallback.
NCCL_ND_API ncclResult_t ncclNdAllToAll(ncclNdHandle_t handle, const ncclNdTensor_t* input, ncclNdTensor_t* output,
                                        int scatter_dim, int gather_dim, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
