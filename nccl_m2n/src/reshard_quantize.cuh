/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Quantize / dequantize launcher declarations.
 *
 * Split from the .cu so host TUs can call the launchers without being
 * compiled by nvcc.
 ************************************************************************/

#ifndef NCCL_M2N_RESHARD_QUANTIZE_CUH_
#define NCCL_M2N_RESHARD_QUANTIZE_CUH_

#include <cstddef>

#include "cuda_runtime.h"
#include "nccl.h"

/* Geometry of one local tile for the quantize/dequantize pass.
 *
 * A dense row-major tile is viewed as [outerA, blockExtent, outerC], where
 * `blockExtent` is the extent of the block dimension, `outerA` is the product
 * of the dims before it, and `outerC` the product of the dims after it.  The
 * block dimension therefore need not be innermost: elements within a block are
 * strided by `outerC`.
 *
 *   offset(a, blk, lane, c) =
 *       a * blockExtent * outerC + (blk * blockSize + lane) * outerC + c
 *
 * One CUDA block handles one scale block, and the scale array is indexed by
 * the flat block id.  That id equals the row-major index into a scale tile of
 * shape [outerA, blockExtent / blockSize, outerC] — exactly the shape the
 * companion scale plane is declared with — so no separate mapping is needed. */
struct ReshardQuantGeometry {
  size_t outerA;       /* product of dims before the block dim */
  size_t blockExtent;  /* extent of the block dim */
  size_t outerC;       /* product of dims after the block dim */
  size_t blocksPerRow; /* blockExtent / blockSize */
  size_t blockSize;    /* elements per scale */
};

/* Scale storage format.
 *
 *   ncclFloat32 — the inverse scale as a plain float.
 *   ncclUint8   — E8M0: the biased exponent byte of a power-of-two inverse
 *                 scale.  This is the MX shared-scale encoding, and it is only
 *                 meaningful when the scale is an exact power of two, so the
 *                 quantizer always rounds when this format is selected.
 *
 * The scale array holds outerA * blocksPerRow * outerC entries either way. */

/* src (dtype) -> quantized E4M3 bytes + inverse scales in `scaleDtype`. */
ncclResult_t reshardLaunchQuantize(const void* src, ncclDataType_t dtype, void* quantized, void* scales,
                                   ncclDataType_t scaleDtype, const ReshardQuantGeometry& geometry, bool roundScales,
                                   cudaStream_t stream);

/* Quantized E4M3 bytes + inverse scales in `scaleDtype` -> dst (dtype). */
ncclResult_t reshardLaunchDequantize(const void* quantized, const void* scales, ncclDataType_t scaleDtype, void* dst,
                                     ncclDataType_t dtype, const ReshardQuantGeometry& geometry, cudaStream_t stream);

#endif /* NCCL_M2N_RESHARD_QUANTIZE_CUH_ */
