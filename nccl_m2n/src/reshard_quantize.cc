/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * On-the-fly wire compression: ncclReshardQuantized.
 *
 * m2n's transport paths cannot host a value transform — PACKWINDOW packs with
 * cudaMemcpy3DAsync and the window path is raw gin.put, neither of which has
 * an ALU stage.  So the transform is staged AROUND an ordinary reshard rather
 * than inside one:
 *
 *   src --quantize--> [FP8 | scales] --ncclReshardScaled--> [FP8 | scales]
 *                                                              --dequantize--> dst
 *
 * The middle step is the coupled reshard from reshard_scale_plane.cc, unchanged,
 * so the scale plane's block-alignment validation and group fusion are
 * inherited rather than duplicated.  This TU adds no transport code.
 *
 * The trailing dequantize is conditional.  When the destination asks for the
 * FP8 dtype it keeps the quantized form and receives the generated scales
 * directly, which is what a consumer computing in FP8 wants — dequantizing
 * only to have it re-quantize would be two lossy conversions to arrive back
 * where the source already was.  nccl_ep behaves the same way: its
 * DS_FP8E3M4 dispatch delivers E4M3 tokens plus scales and never dequantizes.
 ************************************************************************/

#include "nccl_m2n.h"

#include "m2n_checked_math.h"
#include "m2n_checks.h"
#include "m2n_log.h"
#include "reshard_internal.h"
#include "reshard_limits.h"
#include "reshard_quantize.cuh"

namespace {

bool quantPayloadDtypeSupported(ncclDataType_t dtype) {
  /* The quantizer reads and writes through float, so the payload must be a
   * float type.  Integer payloads have no meaningful FP8 representation. */
  return dtype == ncclBfloat16 || dtype == ncclFloat16 || dtype == ncclFloat32;
}

/* The dtype the payload travels as, and which a keep-quantized destination
 * declares.  Both recipes use E4M3; they differ only in scale encoding. */
constexpr ncclDataType_t kQuantWireDtype = ncclFloat8e4m3;

ncclDataType_t recipeScaleDtype(ncclM2nQuantRecipe_t recipe) {
  return recipe == NCCL_M2N_QUANT_MXFP8 ? ncclUint8 : ncclFloat32;
}

size_t recipeScaleElementSize(ncclM2nQuantRecipe_t recipe) {
  return recipe == NCCL_M2N_QUANT_MXFP8 ? 1 : sizeof(float);
}

/* MX defines a shared scale over exactly 32 elements. */
constexpr size_t kMxBlockSize = 32;

/* True when the destination keeps the FP8 form instead of having it
 * reconstructed.  Decided by the destination descriptor's dtype. */
bool keepsQuantized(const ncclDistTensor_t* dst) {
  return dst->dtype == kQuantWireDtype;
}

/* Scratch for one side: the FP8 tile plus its generated scales. */
struct QuantScratch {
  void* quantized;
  void* scales;
  size_t elements;
  size_t blocks;
};

ncclResult_t tileGeometry(const ncclDistTensor_t* tensor, int blockDim, size_t blockSize,
                          ReshardQuantGeometry* geometry, size_t* elements) {
  size_t outerA = 1;
  size_t outerC = 1;
  size_t total = 1;
  for (int d = 0; d < tensor->ndims; d++) {
    const size_t extent = tensor->localShape[d];
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(total, extent, &total), -1,
                       "ncclReshardQuantized: tile element count overflows at dim %d", d);
    if (d < blockDim) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(outerA, extent, &outerA), -1,
                         "ncclReshardQuantized: outer extent overflows at dim %d", d);
    } else if (d > blockDim) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(outerC, extent, &outerC), -1,
                         "ncclReshardQuantized: inner extent overflows at dim %d", d);
    }
  }
  geometry->outerA = outerA;
  geometry->outerC = outerC;
  geometry->blockExtent = tensor->localShape[blockDim];
  geometry->blocksPerRow = geometry->blockExtent / blockSize;
  geometry->blockSize = blockSize;
  *elements = total;
  return ncclSuccess;
}

ncclResult_t allocScratch(const ReshardQuantGeometry& geometry, size_t elements, size_t scaleElementSize,
                          cudaStream_t stream, QuantScratch* scratch) {
  scratch->quantized = nullptr;
  scratch->scales = nullptr;
  scratch->elements = elements;
  scratch->blocks = geometry.outerA * geometry.blocksPerRow * geometry.outerC;
  if (elements == 0) {
    return ncclSuccess;
  }
  /* Stream-ordered so the allocation participates in the same ordering as the
   * reshard it brackets; the caller frees on the same stream after the reshard
   * has been enqueued. */
  NCCL_M2N_CUDACHECK(cudaMallocAsync(&scratch->quantized, elements, stream));
  NCCL_M2N_CUDACHECK(cudaMallocAsync(&scratch->scales, scratch->blocks * scaleElementSize, stream));
  return ncclSuccess;
}

void freeScratch(QuantScratch* scratch, cudaStream_t stream) {
  if (scratch->quantized != nullptr) {
    (void)cudaFreeAsync(scratch->quantized, stream);
    scratch->quantized = nullptr;
  }
  if (scratch->scales != nullptr) {
    (void)cudaFreeAsync(scratch->scales, stream);
    scratch->scales = nullptr;
  }
}

ncclResult_t validateQuantConfig(const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                 const ncclReshardQuantConfig_t* quant) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr, -1,
                     "ncclReshardQuantized: src and dst descriptors must be non-null");
  NCCL_M2N_CHECK_ARG(quant != nullptr, -1, "ncclReshardQuantized: quantization config must be non-null");

  if (quant->size != sizeof(ncclReshardQuantConfig_t) || quant->magic != NCCL_M2N_QUANT_CONFIG_MAGIC) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "ncclReshardQuantized: rejecting malformed ncclReshardQuantConfig_t "
                  "(size=%zu, magic=0x%x, version=%u). Use NCCL_M2N_QUANT_CONFIG_INITIALIZER.",
                  quant->size, quant->magic, quant->version);
  }
  if (quant->version != NCCL_M2N_API_VERSION) {
    RESHARD_WARN(-1,
                 "ncclReshardQuantized: ncclReshardQuantConfig_t.version=%u, library API_VERSION=%u; "
                 "behavior may differ across versions.",
                 quant->version, NCCL_M2N_API_VERSION);
  }

  switch (quant->recipe) {
  case NCCL_M2N_QUANT_NONE:
    NCCL_M2N_CHECK_ARG(quant->blockSize == 0 && quant->blockDim == 0 && quant->roundScales == 0, -1,
                       "ncclReshardQuantized: recipe NCCL_M2N_QUANT_NONE requires every other field to be zero");
    return ncclSuccess;
  case NCCL_M2N_QUANT_FP8E4M3:
    break;
  case NCCL_M2N_QUANT_MXFP8:
    /* E8M0 stores an exponent only, so a non-power-of-two scale cannot be
     * represented.  Require the flag rather than silently overriding it. */
    NCCL_M2N_CHECK_ARG(quant->roundScales != 0, -1,
                       "ncclReshardQuantized: MXFP8 scales are E8M0 (powers of two); set roundScales");
    NCCL_M2N_CHECK_ARG(quant->blockSize == kMxBlockSize, -1,
                       "ncclReshardQuantized: MXFP8 requires blockSize == %zu, got %zu", kMxBlockSize,
                       quant->blockSize);
    break;
  default:
    NCCL_M2N_FAIL(ncclInvalidArgument, -1, "ncclReshardQuantized: unsupported quantization recipe %d",
                  (int)quant->recipe);
  }

  NCCL_M2N_CHECK_ARG(src->ndims == dst->ndims, -1, "ncclReshardQuantized: src->ndims (%d) and dst->ndims (%d) must match",
                     src->ndims, dst->ndims);
  NCCL_M2N_CHECK_ARG(src->ndims >= 1 && src->ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS, -1,
                     "ncclReshardQuantized: ndims (%d) out of range [1, %d]", src->ndims,
                     NCCL_RESHARD_MAX_TENSOR_DIMS);
  /* The destination dtype selects the mode: same as the source means
   * dequantize, the wire dtype means keep the quantized form. */
  NCCL_M2N_CHECK_ARG(dst->dtype == src->dtype || dst->dtype == kQuantWireDtype, -1,
                     "ncclReshardQuantized: dst->dtype (%d) must be either src->dtype (%d) to dequantize, or "
                     "ncclFloat8e4m3 (%d) to keep the quantized form",
                     (int)dst->dtype, (int)src->dtype, (int)kQuantWireDtype);
  NCCL_M2N_CHECK_ARG(quantPayloadDtypeSupported(src->dtype), -1,
                     "ncclReshardQuantized: unsupported payload dtype %d; supported: ncclBfloat16, ncclFloat16, "
                     "ncclFloat32",
                     (int)src->dtype);
  NCCL_M2N_CHECK_ARG(quant->blockDim >= 0 && quant->blockDim < src->ndims, -1,
                     "ncclReshardQuantized: blockDim (%d) out of range [0, %d)", quant->blockDim, src->ndims);
  NCCL_M2N_CHECK_ARG(quant->blockSize >= 1, -1, "ncclReshardQuantized: blockSize must be at least 1");

  /* Same rule the scale plane enforces, checked here too so the error names
   * the quantized entry point rather than surfacing from the inner call. */
  const int bd = quant->blockDim;
  NCCL_M2N_CHECK_ARG(src->localShape[bd] % quant->blockSize == 0, -1,
                     "ncclReshardQuantized: src extent %zu on block dim %d is not a multiple of blockSize=%zu",
                     src->localShape[bd], bd, quant->blockSize);
  NCCL_M2N_CHECK_ARG(dst->localShape[bd] % quant->blockSize == 0, -1,
                     "ncclReshardQuantized: dst extent %zu on block dim %d is not a multiple of blockSize=%zu",
                     dst->localShape[bd], bd, quant->blockSize);

  /* dstScales is the caller's output in keep-quantized mode, and meaningless
   * otherwise; requiring it to be NULL keeps the two modes unambiguous. */
  if (keepsQuantized(dst)) {
    NCCL_M2N_CHECK_ARG(dst->dataPtr == nullptr || quant->dstScales != nullptr, -1,
                       "ncclReshardQuantized: dst->dtype is ncclFloat8e4m3 (keep quantized), so quant->dstScales "
                       "must supply the destination scale buffer on an active destination rank");
  } else {
    NCCL_M2N_CHECK_ARG(quant->dstScales == nullptr, -1,
                       "ncclReshardQuantized: quant->dstScales must be NULL when the destination dequantizes "
                       "(dst->dtype == src->dtype); the scales are internal in that mode");
  }
  return ncclSuccess;
}

} // namespace

extern "C" ncclResult_t ncclReshardQuantized(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src,
                                             const ncclDistTensor_t* dst,
                                             const ncclReshardQuantConfig_t* quant, cudaStream_t stream) {
  if (quant == nullptr) {
    return ncclReshard(handle, comm, src, dst, stream);
  }
  NCCL_M2N_CHECK(validateQuantConfig(src, dst, quant));
  if (quant->recipe == NCCL_M2N_QUANT_NONE) {
    return ncclReshard(handle, comm, src, dst, stream);
  }

  /* Derive both sides' geometry.  A rank inactive on a side has no buffer, but
   * still needs the shape metadata so every rank agrees on the plan — the same
   * rule ncclDistTensor_t::localShape already follows. */
  ReshardQuantGeometry srcGeometry{};
  ReshardQuantGeometry dstGeometry{};
  size_t srcElements = 0;
  size_t dstElements = 0;
  NCCL_M2N_CHECK(tileGeometry(src, quant->blockDim, quant->blockSize, &srcGeometry, &srcElements));
  NCCL_M2N_CHECK(tileGeometry(dst, quant->blockDim, quant->blockSize, &dstGeometry, &dstElements));

  const bool haveSrc = src->dataPtr != nullptr;
  const bool haveDst = dst->dataPtr != nullptr;
  const bool keepQuantized = keepsQuantized(dst);
  const ncclDataType_t scaleDtype = recipeScaleDtype(quant->recipe);
  const size_t scaleElementSize = recipeScaleElementSize(quant->recipe);

  /* The source always needs scratch: the quantized tile is produced here.  The
   * destination only needs scratch when it will be dequantized — in the
   * keep-quantized mode the reshard lands directly in the caller's buffers. */
  QuantScratch srcScratch{};
  QuantScratch dstScratch{};
  NCCL_M2N_CHECK(allocScratch(srcGeometry, haveSrc ? srcElements : 0, scaleElementSize, stream, &srcScratch));
  ncclResult_t result = ncclSuccess;
  if (!keepQuantized) {
    result = allocScratch(dstGeometry, haveDst ? dstElements : 0, scaleElementSize, stream, &dstScratch);
  }

  if (result == ncclSuccess && haveSrc) {
    result = reshardLaunchQuantize(src->dataPtr, src->dtype, srcScratch.quantized, srcScratch.scales, scaleDtype,
                                   srcGeometry, quant->roundScales != 0, stream);
  }

  if (result == ncclSuccess) {
    /* Describe the FP8 tiles and their scales to the coupled reshard.  The
     * quantized payload keeps the payload's shape but is one byte per element;
     * the scale tile is the payload shape with the block dim divided down.
     *
     * The transported dtype is declared as raw bytes rather than
     * ncclFloat8e4m3 because the reshard moves them opaquely, and uint8 is the
     * dtype the coupled reshard's byte-forwarding contract is written against. */
    ncclDistTensor_t quantSrc = *src;
    ncclDistTensor_t quantDst = *dst;
    quantSrc.dataPtr = srcScratch.quantized;
    quantSrc.dtype = ncclUint8;
    quantDst.dataPtr = keepQuantized ? dst->dataPtr : dstScratch.quantized;
    quantDst.dtype = ncclUint8;

    ncclReshardScalePlane_t scales = NCCL_M2N_SCALE_PLANE_INITIALIZER;
    scales.recipe = NCCL_M2N_SCALE_FWD;
    scales.srcDataPtr = srcScratch.scales;
    scales.dstDataPtr = keepQuantized ? quant->dstScales : dstScratch.scales;
    scales.dtype = scaleDtype;
    scales.blockDim = quant->blockDim;
    scales.blockSize = quant->blockSize;
    for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
      scales.srcLocalShape[d] = src->localShape[d];
      scales.dstLocalShape[d] = dst->localShape[d];
    }
    scales.srcLocalShape[quant->blockDim] = src->localShape[quant->blockDim] / quant->blockSize;
    scales.dstLocalShape[quant->blockDim] = dst->localShape[quant->blockDim] / quant->blockSize;

    result = ncclReshardScaled(handle, comm, &quantSrc, &quantDst, &scales, stream);
  }

  /* Only the dequantizing mode has a destination-side pass; keep-quantized
   * has already delivered the payload and scales the consumer wants. */
  if (result == ncclSuccess && haveDst && !keepQuantized) {
    result = reshardLaunchDequantize(dstScratch.quantized, dstScratch.scales, scaleDtype, dst->dataPtr, dst->dtype,
                                     dstGeometry, stream);
  }

  freeScratch(&srcScratch, stream);
  freeScratch(&dstScratch, stream);
  return result;
}
