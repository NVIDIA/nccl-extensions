/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
#ifndef NCCL_ND_LAYOUT_HPP_
#define NCCL_ND_LAYOUT_HPP_

#include "nccl_nd.h"

#include <limits>

namespace nccl_nd {

// Byte geometry for one source peer. Width covers the head slice (including D);
// height covers the sequence slice, and depth preserves independent batches.
struct CopyGeometry {
  size_t inputOffset, outputOffset;
  size_t srcPitch, dstPitch;
  size_t srcHeight, dstHeight;
  size_t width, height, depth;
};

// Pure host metadata; no device resources or application-specific axis names.
struct Layout {
  CopyGeometry copy;
  size_t inputPeerStride;
  size_t outputPeerStride;
  size_t totalBytes;
};

inline bool checkedMultiply(size_t a, size_t b, size_t* output) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
  *output = a * b;
  return true;
}

inline bool canonicalDense(const ncclNdTensor_t* tensor, size_t* elements) {
  size_t stride = 1;
  for (unsigned int dim = tensor->ndim; dim != 0; --dim) {
    const size_t extent = tensor->sizes[dim - 1];
    if (extent == 0 || (tensor->strides != nullptr && tensor->strides[dim - 1] != stride)) return false;
    if (!checkedMultiply(stride, extent, &stride)) return false;
  }
  *elements = stride;
  return true;
}

inline ncclResult_t buildLayout(const ncclNdTensor_t* input, const ncclNdTensor_t* output,
                                int scatter, int gather, int world, Layout* layout) {
  if (input == nullptr || output == nullptr || layout == nullptr || world < 2) {
    return ncclInvalidArgument;
  }
  if ((input->ndim != 2 && input->ndim != 4) || output->ndim != input->ndim || input->sizes == nullptr ||
      output->sizes == nullptr || input->datatype != output->datatype) {
    return ncclInvalidArgument;
  }

  size_t elementBytes;
  switch (input->datatype) {
  case ncclFloat16:
  case ncclBfloat16:
    elementBytes = 2;
    break;
  case ncclFloat32:
    elementBytes = 4;
    break;
  default:
    return ncclInvalidArgument;
  }

  const int sequenceAxis = input->ndim == 2 ? 0 : 1;
  const int headAxis = sequenceAxis + 1;
  if (!((scatter == headAxis && gather == sequenceAxis) ||
        (scatter == sequenceAxis && gather == headAxis)))
    return ncclInvalidArgument;

  size_t inputElements, outputElements;
  if (!canonicalDense(input, &inputElements) || !canonicalDense(output, &outputElements) ||
      inputElements != outputElements)
    return ncclInvalidArgument;
  const size_t peers = static_cast<size_t>(world);
  if (input->sizes[scatter] % peers != 0) return ncclInvalidArgument;
  for (unsigned int dim = 0; dim < input->ndim; ++dim) {
    size_t expected = input->sizes[dim];
    if (dim == static_cast<unsigned int>(scatter)) expected /= peers;
    if (dim == static_cast<unsigned int>(gather) && !checkedMultiply(expected, peers, &expected)) {
      return ncclInvalidArgument;
    }
    if (output->sizes[dim] != expected) return ncclInvalidArgument;
  }

  Layout result{};
  size_t rowUnitBytes;
  const size_t headDim = input->ndim == 4 ? input->sizes[3] : 1;
  if (!checkedMultiply(headDim, elementBytes, &rowUnitBytes) ||
      !checkedMultiply(input->sizes[headAxis], rowUnitBytes, &result.copy.srcPitch) ||
      !checkedMultiply(output->sizes[headAxis], rowUnitBytes, &result.copy.dstPitch) ||
      !checkedMultiply(inputElements, elementBytes, &result.totalBytes))
    return ncclInvalidArgument;
  result.copy.srcHeight = input->sizes[sequenceAxis];
  result.copy.dstHeight = output->sizes[sequenceAxis];
  result.copy.width = result.copy.srcPitch < result.copy.dstPitch ? result.copy.srcPitch : result.copy.dstPitch;
  result.copy.height = result.copy.srcHeight < result.copy.dstHeight ? result.copy.srcHeight : result.copy.dstHeight;
  result.copy.depth = input->ndim == 4 ? input->sizes[0] : 1;
  if (result.copy.width % 16 != 0) return ncclInvalidArgument;
  size_t sliceBytes;
  if (!checkedMultiply(result.copy.width, result.copy.height, &sliceBytes)) return ncclInvalidArgument;
  // Rank selects one input split; source peer selects one output concat slice.
  result.inputPeerStride = scatter == sequenceAxis ? sliceBytes : result.copy.width;
  result.outputPeerStride = gather == sequenceAxis ? sliceBytes : result.copy.width;
  *layout = result;
  return ncclSuccess;
}

// Valid communicator rank/peer indices keep the offsets inside totalBytes.
inline CopyGeometry copyGeometry(const Layout& layout, int rank, int peer) {
  CopyGeometry result = layout.copy;
  result.inputOffset = static_cast<size_t>(rank) * layout.inputPeerStride;
  result.outputOffset = static_cast<size_t>(peer) * layout.outputPeerStride;
  return result;
}

}  // namespace nccl_nd
#endif
