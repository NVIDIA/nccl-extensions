/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Coupled (payload, scales) resharding.
 *
 * A quantized payload and its companion per-block scale plane must be
 * resharded consistently.  Submitting them as two independent ncclReshard
 * calls moves the right bytes but cannot check that a shard boundary lands on
 * a scale-block boundary, so it silently produces a wrong result when it does
 * not.  This TU adds the missing validation and then delegates all transport
 * to the existing entry points by submitting both planes as one M2N group.
 *
 * There is deliberately no new kernel, plan, or parameter struct here.
 ************************************************************************/

#include "nccl_m2n.h"

#include "m2n_checked_math.h"
#include "m2n_checks.h"
#include "m2n_log.h"
#include "reshard_internal.h"
#include "reshard_limits.h"

namespace {

/* Shard geometry for one side, derived from placements and mesh dims alone.
 * Rank-independent, so it is identical on every rank by construction.
 * shardTensorDim is -1 when the side is fully replicated. */
struct ScalePlaneSideInfo {
  int shardTensorDim;
  size_t shardCount;
};

void scalePlaneSideInfo(const ncclDistTensor_t* tensor, ScalePlaneSideInfo* out) {
  out->shardTensorDim = -1;
  out->shardCount = 1;
  for (int d = 0; d < NCCL_RESHARD_MESH_NDIMS; d++) {
    if (isShardPlacement(tensor->placements[d])) {
      out->shardTensorDim = getShardTensorDim(tensor->placements[d]);
      out->shardCount = static_cast<size_t>(tensor->mesh->dims[d]);
    }
  }
}

bool scalePlaneDtypeSupported(ncclDataType_t dtype) {
  /* Deliberately narrower than getNcclDtSize(), which also accepts the integer
   * and float64 payload types.  A scale plane carries scaling factors; the
   * wider set is far more likely to be a caller mistake than an intent. */
  switch (dtype) {
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
  case ncclUint8:
    return true;
  default:
    return false;
  }
}

/* Validate one side's scale shape against its payload shape, and report the
 * side's derived global scale shape for the cross-side agreement check. */
ncclResult_t validateScaleSideShape(const char* apiName, const char* side, const ncclDistTensor_t* payload,
                                    const size_t scaleShape[NCCL_RESHARD_MAX_TENSOR_DIMS], int blockDim,
                                    size_t blockSize, size_t globalScaleShape[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  ScalePlaneSideInfo info{};
  scalePlaneSideInfo(payload, &info);

  /* The shard-boundary rule, checked BEFORE the generic extent arithmetic.
   * Both reject the same inputs, but only this one names the actual problem:
   * the side shards the block dimension at a boundary interior to a scale
   * block.  Checked second it would be unreachable, because no integral scale
   * extent can cover a payload extent that blockSize does not divide. */
  if (info.shardTensorDim == blockDim) {
    NCCL_M2N_CHECK_ARG(payload->localShape[blockDim] % blockSize == 0, -1,
                       "%s: %s shards block dim %d into %zu shards of %zu elements, which is not a multiple of "
                       "blockSize=%zu; every shard boundary would fall inside a scale block",
                       apiName, side, blockDim, info.shardCount, payload->localShape[blockDim], blockSize);
  }

  for (int d = 0; d < payload->ndims; d++) {
    if (d != blockDim) {
      NCCL_M2N_CHECK_ARG(scaleShape[d] == payload->localShape[d], -1,
                         "%s: %s scale shape must match the payload on every non-block dimension; "
                         "dim %d is %zu but the payload is %zu",
                         apiName, side, d, scaleShape[d], payload->localShape[d]);
    } else {
      size_t covered = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(scaleShape[d], blockSize, &covered), -1,
                         "%s: %s scale extent overflow on block dim %d (scale=%zu, blockSize=%zu)", apiName, side, d,
                         scaleShape[d], blockSize);
      NCCL_M2N_CHECK_ARG(covered == payload->localShape[d], -1,
                         "%s: %s scale extent %zu on block dim %d covers %zu payload elements, but the payload "
                         "extent is %zu; blockSize=%zu must divide the payload extent exactly",
                         apiName, side, scaleShape[d], d, covered, payload->localShape[d], blockSize);
    }
  }

  for (int d = 0; d < payload->ndims; d++) {
    globalScaleShape[d] = scaleShape[d];
  }
  if (info.shardTensorDim >= 0 && info.shardTensorDim < payload->ndims) {
    const int sd = info.shardTensorDim;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(scaleShape[sd], info.shardCount, &globalScaleShape[sd]), -1,
                       "%s: %s global scale shape overflow on shard dim %d (local=%zu, shards=%zu)", apiName, side, sd,
                       scaleShape[sd], info.shardCount);
  }
  return ncclSuccess;
}

} // namespace

bool reshardScalePlaneActive(const ncclReshardScalePlane_t* scales) {
  return scales != nullptr && scales->recipe != NCCL_M2N_SCALE_NONE;
}

ncclResult_t validateReshardScalePlane(const char* apiName, const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                       const ncclReshardScalePlane_t* scales) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr, -1, "%s: src and dst descriptors must be non-null", apiName);
  NCCL_M2N_CHECK_ARG(scales != nullptr, -1, "%s: scale plane descriptor must be non-null", apiName);

  /* ABI guard, mirroring validateReshardConfigHeader: size and magic are hard
   * rejects, a version difference only warns. */
  if (scales->size != sizeof(ncclReshardScalePlane_t) || scales->magic != NCCL_M2N_SCALE_PLANE_MAGIC) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "%s: rejecting malformed ncclReshardScalePlane_t "
                  "(size=%zu, magic=0x%x, version=%u). Use NCCL_M2N_SCALE_PLANE_INITIALIZER.",
                  apiName, scales->size, scales->magic, scales->version);
  }
  if (scales->version != NCCL_M2N_API_VERSION) {
    RESHARD_WARN(-1,
                 "%s: ncclReshardScalePlane_t.version=%u, library API_VERSION=%u; "
                 "behavior may differ across versions.",
                 apiName, scales->version, NCCL_M2N_API_VERSION);
  }

  /* Closed recipe set.  No default fall-through to FWD: an unknown enumerator
   * is a caller error, not something to guess at. */
  switch (scales->recipe) {
  case NCCL_M2N_SCALE_NONE:
    NCCL_M2N_CHECK_ARG(scales->srcDataPtr == nullptr && scales->dstDataPtr == nullptr && scales->blockSize == 0 &&
                         scales->blockDim == 0,
                       -1,
                       "%s: recipe NCCL_M2N_SCALE_NONE requires every scale field to be zero/NULL; "
                       "set recipe to NCCL_M2N_SCALE_FWD to forward a scale plane",
                       apiName);
    return ncclSuccess;
  case NCCL_M2N_SCALE_FWD:
    break;
  default:
    NCCL_M2N_FAIL(ncclInvalidArgument, -1, "%s: unsupported scale recipe %d", apiName, (int)scales->recipe);
  }

  NCCL_M2N_CHECK_ARG(src->mesh != nullptr && dst->mesh != nullptr, -1,
                     "%s: src->mesh and dst->mesh must both be non-null on every rank", apiName);
  NCCL_M2N_CHECK_ARG(src->ndims == dst->ndims, -1, "%s: src->ndims (%d) and dst->ndims (%d) must match", apiName,
                     src->ndims, dst->ndims);
  NCCL_M2N_CHECK_ARG(src->ndims >= 1 && src->ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS, -1,
                     "%s: ndims (%d) out of range [1, %d]", apiName, src->ndims, NCCL_RESHARD_MAX_TENSOR_DIMS);

  NCCL_M2N_CHECK_ARG(scalePlaneDtypeSupported(scales->dtype), -1,
                     "%s: unsupported scale dtype %d; supported: ncclFloat32, ncclFloat16, ncclBfloat16, "
                     "ncclFloat8e4m3, ncclFloat8e5m2, ncclUint8",
                     apiName, (int)scales->dtype);
  NCCL_M2N_CHECK_ARG(scales->blockDim >= 0 && scales->blockDim < src->ndims, -1,
                     "%s: blockDim (%d) out of range [0, %d)", apiName, scales->blockDim, src->ndims);
  NCCL_M2N_CHECK_ARG(scales->blockSize >= 1, -1, "%s: blockSize must be at least 1", apiName);

  /* A rank active on a side must supply both planes for that side, and an
   * inactive side must supply neither.  dataPtr nullness already tracks mesh
   * membership, so this keeps the two planes' participation identical. */
  NCCL_M2N_CHECK_ARG((scales->srcDataPtr != nullptr) == (src->dataPtr != nullptr), -1,
                     "%s: scales->srcDataPtr must be non-null exactly when src->dataPtr is non-null", apiName);
  NCCL_M2N_CHECK_ARG((scales->dstDataPtr != nullptr) == (dst->dataPtr != nullptr), -1,
                     "%s: scales->dstDataPtr must be non-null exactly when dst->dataPtr is non-null", apiName);

  size_t srcGlobalScale[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstGlobalScale[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(validateScaleSideShape(apiName, "src", src, scales->srcLocalShape, scales->blockDim,
                                        scales->blockSize, srcGlobalScale));
  NCCL_M2N_CHECK(validateScaleSideShape(apiName, "dst", dst, scales->dstLocalShape, scales->blockDim,
                                        scales->blockSize, dstGlobalScale));

  /* Both sides must describe the same global scale tensor.  This catches a
   * caller who used the right blockSize for one side and the wrong one for the
   * other, and reports it in terms of blockSize rather than letting the group
   * path fail later with a bare global-shape mismatch. */
  for (int d = 0; d < src->ndims; d++) {
    NCCL_M2N_CHECK_ARG(srcGlobalScale[d] == dstGlobalScale[d], -1,
                       "%s: source and destination global scale shapes differ at dim %d (%zu vs %zu); "
                       "check blockSize=%zu and the per-side scale shapes",
                       apiName, d, srcGlobalScale[d], dstGlobalScale[d], scales->blockSize);
  }

  /* Not an error: m2n moves raw bytes and picks its vector width from runtime
   * pointer alignment, so a narrow scale row is legal.  It is worth surfacing
   * because it costs bandwidth. */
  const size_t scaleElementSize = getNcclDtSize(scales->dtype);
  const size_t srcScaleRowBytes = scales->srcLocalShape[src->ndims - 1] * scaleElementSize;
  if (srcScaleRowBytes != 0 && srcScaleRowBytes < 16) {
    RESHARD_DEBUG(-1, "%s: scale row is %zu bytes (<16); transfers will use a narrow vector width", apiName,
                  srcScaleRowBytes);
  }

  return ncclSuccess;
}

ncclResult_t buildReshardScaleTensors(const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                      const ncclReshardScalePlane_t* scales, ncclDistTensor_t* outSrcScale,
                                      ncclDistTensor_t* outDstScale) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr && scales != nullptr && outSrcScale != nullptr &&
                       outDstScale != nullptr,
                     -1, "buildReshardScaleTensors: called with a null argument");

  /* Copy the payload descriptors so the scale plane inherits mesh, placements
   * and ndims verbatim — that shared topology is the whole point — then
   * override only what the scale plane owns. */
  *outSrcScale = *src;
  *outDstScale = *dst;

  outSrcScale->dataPtr = scales->srcDataPtr;
  outSrcScale->dtype = scales->dtype;
  outDstScale->dataPtr = scales->dstDataPtr;
  outDstScale->dtype = scales->dtype;
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    outSrcScale->localShape[d] = scales->srcLocalShape[d];
    outDstScale->localShape[d] = scales->dstLocalShape[d];
  }
  return ncclSuccess;
}

/* Submit both planes as one group.
 *
 * Group nesting makes this correct in both contexts: with no group active the
 * two entries form a bucket of two and fuse into a single kernel and barrier
 * epoch; inside a caller's group ncclM2nGroupStart only increments the depth,
 * so the entries append to the caller's group and issue with it.
 *
 * ncclM2nGroupAbort is deliberately NOT used on the error path: it clears every
 * nesting level, so calling it while nested inside a caller's group would
 * discard the caller's recorded entries.  Instead we always close the level we
 * opened and let the group's own sticky-error contract prevent a partial issue
 * — a failed enqueue poisons the group, and the outermost ncclM2nGroupEnd then
 * returns that error without issuing anything. */
namespace {

template <typename EnqueueFn>
ncclResult_t reshardScaledSubmit(EnqueueFn&& enqueue, const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                 const ncclDistTensor_t* srcScale, const ncclDistTensor_t* dstScale) {
  NCCL_M2N_CHECK(ncclM2nGroupStart());
  const ncclResult_t payloadResult = enqueue(src, dst);
  ncclResult_t scaleResult = ncclSuccess;
  if (payloadResult == ncclSuccess) {
    scaleResult = enqueue(srcScale, dstScale);
  }
  const ncclResult_t endResult = ncclM2nGroupEnd();

  if (payloadResult != ncclSuccess) return payloadResult;
  if (scaleResult != ncclSuccess) return scaleResult;
  return endResult;
}

} // namespace

extern "C" ncclResult_t ncclReshardScaled(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src,
                                          const ncclDistTensor_t* dst, const ncclReshardScalePlane_t* scales,
                                          cudaStream_t stream) {
  if (!reshardScalePlaneActive(scales)) {
    if (scales != nullptr) {
      NCCL_M2N_CHECK(validateReshardScalePlane("ncclReshardScaled", src, dst, scales));
    }
    return ncclReshard(handle, comm, src, dst, stream);
  }

  NCCL_M2N_CHECK(validateReshardScalePlane("ncclReshardScaled", src, dst, scales));
  ncclDistTensor_t srcScale{};
  ncclDistTensor_t dstScale{};
  NCCL_M2N_CHECK(buildReshardScaleTensors(src, dst, scales, &srcScale, &dstScale));

  return reshardScaledSubmit(
    [&](const ncclDistTensor_t* s, const ncclDistTensor_t* d) -> ncclResult_t {
      return ncclReshard(handle, comm, s, d, stream);
    },
    src, dst, &srcScale, &dstScale);
}

extern "C" ncclResult_t ncclReshardScaledWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window,
                                                    const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                                    const ncclReshardScalePlane_t* scales, cudaStream_t stream) {
  if (!reshardScalePlaneActive(scales)) {
    if (scales != nullptr) {
      NCCL_M2N_CHECK(validateReshardScalePlane("ncclReshardScaledWithWindow", src, dst, scales));
    }
    return ncclReshardWithWindow(handle, comm, window, src, dst, stream);
  }

  NCCL_M2N_CHECK(validateReshardScalePlane("ncclReshardScaledWithWindow", src, dst, scales));
  ncclDistTensor_t srcScale{};
  ncclDistTensor_t dstScale{};
  NCCL_M2N_CHECK(buildReshardScaleTensors(src, dst, scales, &srcScale, &dstScale));

  return reshardScaledSubmit(
    [&](const ncclDistTensor_t* s, const ncclDistTensor_t* d) -> ncclResult_t {
      return ncclReshardWithWindow(handle, comm, window, s, d, stream);
    },
    src, dst, &srcScale, &dstScale);
}
