/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_RESHARD_LIMITS_H_
#define NCCL_RESHARD_LIMITS_H_

#include <cstddef>

#include "nccl_m2n.h"

/*
 * Central definition of all compile-time constants used by the
 * nccl-reshard library.  Every translation unit includes this header
 * instead of defining its own copy.
 *
 * Integer constants are `inline constexpr` (header-only single
 * definition, typed, no preprocessor leakage); function-like helpers
 * are `constexpr` inline functions.  Names stay UPPER_CASE for ABI/
 * diff continuity with the previous `#define` era — clang-tidy
 * (`ConstantCase: aNy_CasE`) permits this.
 *
 * Public placement helpers and tensor-rank limit are owned by the
 * public header `nccl_m2n.h`; they are NOT redefined here.
 */

/* RING (hierarchical) algorithm array sizes. */
inline constexpr int MAX_SOURCES = 16;
inline constexpr int MAX_TARGETS = 64;
inline constexpr int MAX_LOCAL_FOLLOWERS = 128;
inline constexpr int MAX_WARP_GROUPS = 15;
inline constexpr int MAX_SRC_WARPS = 8;

/* DIRECT algorithm array sizes. */
inline constexpr int MAX_DIRECT_SOURCES = 32;
inline constexpr int MAX_DIRECT_TARGETS = 64;

/* Default chunking parameters. */
inline constexpr int DEFAULT_ELEMENTS_PER_CHUNK = 32;
inline constexpr size_t CHUNK_SIZE_BYTES = 256ULL * 1024ULL;

/* Default kernel-launch parameters.
 *
 * DEFAULT_KERNEL_MAX_NTHREADS must match the value baked into each
 * __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) declaration on the
 * resharding kernels — keep them in sync (NCCL v2.30 register-pressure
 * fix from commit 420236f).  __launch_bounds__ accepts constexpr
 * integer constants as well as #define values. */
inline constexpr int DEFAULT_NUM_CTAS = 8;
inline constexpr int DEFAULT_KERNEL_MAX_NTHREADS = 512;
inline constexpr int DEFAULT_GIN_CONTEXT_COUNT = 4;
inline constexpr int DEFAULT_GPUS_PER_NODE = 8;
inline constexpr int DEFAULT_STREAM_POOL_SIZE = 4;

/* Cross-dim transpose threshold (bytes).  If the innermost transfer
   size is below this, the library transparently transposes the last
   two tensor dims to improve RDMA throughput.

   Keep the macro as the build-time override surface
   (`-DCROSS_DIM_TRANSPOSE_THRESHOLD=...`); internal code uses the typed
   constexpr value below. */
#ifndef CROSS_DIM_TRANSPOSE_THRESHOLD
#define CROSS_DIM_TRANSPOSE_THRESHOLD (256ULL * 1024ULL)
#endif
inline constexpr size_t CROSS_DIM_TRANSPOSE_THRESHOLD_BYTES = CROSS_DIM_TRANSPOSE_THRESHOLD;

/* Cache capacities. */
inline constexpr int MAX_WINDOW_CACHE_ENTRIES = 128;
inline constexpr int MAX_DEVCOMM_CACHE_ENTRIES = 64;
inline constexpr int MAX_TRANSPOSE_BUFFER_ENTRIES = 16;

/* Stream pool — used when callers pass the default stream
 * (nullptr / cudaStreamLegacy / cudaStreamPerThread).  Caps how many
 * distinct (ncclComm_t, cuda device) entries the pool will track,
 * one stream + one back-edge event per entry (1:1 mapping).  The
 * effective entry count is driven by NCCL_RESHARD_STREAM_POOL_SIZE
 * (default 4); STREAM_POOL_MAX_SIZE is the hard upper bound on
 * that env-var setting.  The resolved value lives in process-global
 * runtime state for the init/finalize epoch. */
inline constexpr int STREAM_POOL_MAX_SIZE = 256;

/* Placement classifiers — operate on the int placement value stored
 * in ncclDistTensor_t::placements[i] (NCCL_RESHARD_REPLICATE or
 * NCCL_RESHARD_SHARD(d)).  constexpr inline so they evaluate at compile
 * time in template/array-index contexts. */
constexpr inline bool isShardPlacement(int p) {
  return p >= 0;
}
constexpr inline int getShardTensorDim(int p) {
  return p;
}

#endif /* NCCL_RESHARD_LIMITS_H_ */
