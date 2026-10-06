/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
#include "nccl_nd.h"
#include "nccl_nd_layout.hpp"

#include <cuda_runtime.h>
#include <nccl_device.h>

#include <cstdarg>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>

struct ncclNdHandle {
  ncclComm_t comm;
  ncclDevComm devComm;
  int device;
  int world;
  int rank;
  size_t maxPitch;
};

namespace nccl_nd {

// CUDA failures are reported as ncclResult_t codes; this keeps the underlying
// CUDA status, rank and peer for diagnosis. Failures are rare and fatal for the
// operation, so the message is unconditional.
__attribute__((format(printf, 1, 2))) void warn(const char* format, ...) {
  char message[256];
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(message, sizeof(message), format, arguments);
  va_end(arguments);
  fprintf(stderr, "NCCL ND WARN %s\n", message);
}

inline ncclResult_t checkCapture(cudaStream_t stream, bool allowActive = false) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  const cudaError_t status = cudaStreamIsCapturing(stream, &capture);
  if (status == cudaErrorStreamCaptureImplicit) return ncclInvalidUsage;
  if (status != cudaSuccess) return ncclUnhandledCudaError;
  if (capture == cudaStreamCaptureStatusNone) return ncclSuccess;
  return allowActive && capture == cudaStreamCaptureStatusActive ? ncclSuccess : ncclInvalidUsage;
}

inline bool addressEnd(uintptr_t first, size_t bytes, uintptr_t* end) {
  if (bytes > std::numeric_limits<uintptr_t>::max() - first) return false;
  *end = first + bytes;
  return true;
}

inline bool overlap(uintptr_t first, uintptr_t end, uintptr_t otherFirst, uintptr_t otherEnd) {
  return first < otherEnd && otherFirst < end;
}

ncclResult_t validateStorage(ncclNdHandle_t handle, const ncclNdTensor_t* input, const ncclNdTensor_t* output,
                             const Layout& layout) {
  if (input->win_hdl == nullptr || input->data != nullptr || output->data == nullptr || output->win_hdl != nullptr ||
      output->win_offset != 0)
    return ncclInvalidArgument;
  if (input->win_offset > std::numeric_limits<size_t>::max()) return ncclInvalidArgument;
  const size_t offset = static_cast<size_t>(input->win_offset);
  if (layout.totalBytes > std::numeric_limits<size_t>::max() - offset) return ncclInvalidArgument;

  // Use the communicator's shadow pool to validate window ownership and recover the original registered VA.
  void* userBase = nullptr;
  ncclResult_t status = ncclWinGetUserPtr(handle->comm, input->win_hdl, &userBase);
  if (status != ncclSuccess) return status;
  if (userBase == nullptr) return ncclInvalidArgument;

  // The host pointer accessor checks offset < registered size. Checking both ends of the dense range
  // validates the full input range without accessing payloads. The LSA VA may differ from the userBase alias.
  void* lsaFirst = nullptr;
  status = ncclGetLsaDevicePointer(input->win_hdl, offset, handle->devComm.lsaRank, &lsaFirst);
  if (status != ncclSuccess) return status;
  void* lsaLast = nullptr;
  status = ncclGetLsaDevicePointer(input->win_hdl, offset + layout.totalBytes - 1, handle->devComm.lsaRank, &lsaLast);
  if (status != ncclSuccess) return status;
  if (lsaFirst == nullptr || lsaLast == nullptr) return ncclInvalidArgument;

  const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(userBase);
  if (offset > std::numeric_limits<uintptr_t>::max() - baseAddress) return ncclInvalidArgument;
  const uintptr_t inputAddress = baseAddress + offset;
  const uintptr_t lsaAddress = reinterpret_cast<uintptr_t>(lsaFirst);
  const uintptr_t outputAddress = reinterpret_cast<uintptr_t>(output->data);
  if (inputAddress % 16 != 0 || lsaAddress % 16 != 0 || outputAddress % 16 != 0) return ncclInvalidArgument;
  uintptr_t inputEnd, lsaEnd, outputEnd;
  if (!addressEnd(inputAddress, layout.totalBytes, &inputEnd) || !addressEnd(lsaAddress, layout.totalBytes, &lsaEnd) ||
      !addressEnd(outputAddress, layout.totalBytes, &outputEnd))
    return ncclInvalidArgument;
  if (overlap(outputAddress, outputEnd, inputAddress, inputEnd) ||
      overlap(outputAddress, outputEnd, lsaAddress, lsaEnd))
    return ncclInvalidArgument;

  cudaPointerAttributes attributes{};
  const cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, output->data);
  if (cudaStatus == cudaErrorInvalidValue) return ncclInvalidArgument;
  if (cudaStatus != cudaSuccess) return ncclUnhandledCudaError;
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != handle->device) return ncclInvalidArgument;
  return ncclSuccess;
}

// Entry publishes input readiness; exit makes completion include every peer's
// reads. One CTA per rank participates in barrier slot 0. Stream ordering covers
// the entire copy sequence; there are no payload CTAs to synchronize. Serialized
// operations and graph replays reuse the slot and advance its persistent epoch.
__global__ void rankBarrier(ncclDevComm devComm) {
  const ncclCoopCta coop{};
  ncclLsaBarrierSession<ncclCoopCta> barrier{coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, 0};
  barrier.sync(coop, cuda::memory_order_acq_rel);
}

ncclResult_t enqueueCopies(ncclNdHandle_t handle, const ncclNdTensor_t* input,
                          const ncclNdTensor_t* output, const Layout& layout, cudaStream_t stream) {
  if (layout.copy.srcPitch > handle->maxPitch || layout.copy.dstPitch > handle->maxPitch)
    return ncclInvalidArgument;
  std::vector<cudaMemcpy3DParms> copies;
  try {
    copies.resize(handle->world);
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }
  // Resolve every peer before enqueueing any collective work. These are host
  // metadata queries; AllToAll never allocates or registers device memory.
  const size_t offset = static_cast<size_t>(input->win_offset);
  for (int index = 0; index < handle->world; ++index) {
    const int peer = static_cast<int>((static_cast<size_t>(index) + handle->rank) % handle->world);
    void* source = nullptr;
    void* last = nullptr;
    ncclResult_t status = ncclGetLsaDevicePointer(input->win_hdl, offset, peer, &source);
    if (status != ncclSuccess) return status;
    status = ncclGetLsaDevicePointer(input->win_hdl, offset + layout.totalBytes - 1, peer, &last);
    if (status != ncclSuccess) return status;
    if (source == nullptr || last == nullptr) return ncclInvalidArgument;
    const uintptr_t sourceAddress = reinterpret_cast<uintptr_t>(source);
    uintptr_t sourceEnd;
    if (sourceAddress % 16 != 0 || !addressEnd(sourceAddress, layout.totalBytes, &sourceEnd)) {
      return ncclInvalidArgument;
    }
    const CopyGeometry geometry = copyGeometry(layout, handle->rank, peer);
    // DeviceToDevice treats both LSA pointers as device memory, as the driver-level device copy does.
    auto& copy = copies[index];
    copy.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(sourceAddress + geometry.inputOffset), geometry.srcPitch,
                                      geometry.width, geometry.srcHeight);
    copy.dstPtr = make_cudaPitchedPtr(static_cast<char*>(output->data) + geometry.outputOffset, geometry.dstPitch,
                                      geometry.width, geometry.dstHeight);
    copy.extent = make_cudaExtent(geometry.width, geometry.height, geometry.depth);
    copy.kind = cudaMemcpyDeviceToDevice;
  }
  ncclDevComm devComm = handle->devComm;
  void* arguments[] = {&devComm};
  const auto barrier = reinterpret_cast<const void*>(rankBarrier);
  cudaError_t launch = cudaLaunchKernel(barrier, dim3(1), dim3(32), arguments, 0, stream);
  if (launch != cudaSuccess) {
    // Nothing of this operation is enqueued on this rank, but peers may already wait in the entry barrier.
    warn("rank %d: entry barrier launch failed: %s", handle->rank, cudaGetErrorName(launch));
    return ncclUnhandledCudaError;
  }
  ncclResult_t result = ncclSuccess;
  for (int index = 0; index < handle->world; ++index) {
    const cudaError_t copyStatus = cudaMemcpy3DAsync(&copies[index], stream);
    if (copyStatus != cudaSuccess) {
      // Peers already passed the entry barrier; still enqueue the exit barrier so they do not hang.
      // The output is undefined and the operation is collective-fatal. Do not retry or fall back.
      warn("rank %d: cudaMemcpy3DAsync from peer %d failed: %s after %d of %d copies", handle->rank,
           (handle->rank + index) % handle->world, cudaGetErrorName(copyStatus), index, handle->world);
      result = ncclUnhandledCudaError;
      break;
    }
  }
  launch = cudaLaunchKernel(barrier, dim3(1), dim3(32), arguments, 0, stream);
  if (launch != cudaSuccess) {
    warn("rank %d: exit barrier launch failed: %s", handle->rank, cudaGetErrorName(launch));
    return ncclUnhandledCudaError;
  }
  return result;
}

}  // namespace nccl_nd

extern "C" ncclResult_t ncclNdGetVersion(int* version) {
  if (version == nullptr) return ncclInvalidArgument;
  *version = NCCL_ND_VERSION_CODE;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclNdCreate(ncclNdHandle_t* handle, ncclComm_t comm) {
  using namespace nccl_nd;
  if (handle == nullptr) return ncclInvalidArgument;
  *handle = nullptr;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclResult_t status = checkCapture(nullptr);
  if (status != ncclSuccess) return status;
  int runtimeVersion = 0;
  status = ncclGetVersion(&runtimeVersion);
  if (status != ncclSuccess) return status;
  if (runtimeVersion < NCCL_VERSION_CODE) return ncclInvalidUsage;

  ncclCommProperties_t properties{};
  properties.size = sizeof(ncclCommProperties_t);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  status = ncclCommQueryProperties(comm, &properties);
  if (status != ncclSuccess) return status;
  if (properties.nRanks < 2 || !properties.deviceApiSupport || properties.nLsaTeams != 1) return ncclInvalidUsage;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return ncclUnhandledCudaError;
  if (device != properties.cudaDev) return ncclInvalidUsage;
  cudaDeviceProp deviceProperties{};
  if (cudaGetDeviceProperties(&deviceProperties, device) != cudaSuccess) return ncclUnhandledCudaError;

  auto* result = new (std::nothrow) ncclNdHandle{};
  if (result == nullptr) return ncclSystemError;
  result->comm = comm;
  result->device = device;
  result->world = properties.nRanks;
  result->rank = properties.rank;
  result->maxPitch = deviceProperties.memPitch;
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.lsaBarrierCount = 1;
  status = ncclDevCommCreate(comm, &requirements, &result->devComm);
  if (status != ncclSuccess) {
    delete result;
    return status;
  }
  if (result->devComm.lsaSize != result->world || result->devComm.lsaRank != result->rank) {
    status = ncclDevCommDestroy(comm, &result->devComm);
    delete result;
    return status == ncclSuccess ? ncclInvalidUsage : status;
  }
  *handle = result;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclNdDestroy(ncclNdHandle_t handle) {
  using namespace nccl_nd;
  if (handle == nullptr) return ncclSuccess;
  ncclResult_t status = checkCapture(nullptr);
  if (status != ncclSuccess) return status;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return ncclUnhandledCudaError;
  if (device != handle->device) return ncclInvalidUsage;
  // The caller guarantees completion on every stream. Borrowed communicators, windows, and payloads are untouched.
  status = ncclDevCommDestroy(handle->comm, &handle->devComm);
  if (status != ncclSuccess) return status;
  delete handle;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclNdAllToAll(ncclNdHandle_t handle, const ncclNdTensor_t* input, ncclNdTensor_t* output,
                                       int scatter_dim, int gather_dim, cudaStream_t stream) {
  using namespace nccl_nd;
  if (handle == nullptr) return ncclInvalidArgument;
  if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) return ncclInvalidUsage;
  Layout layout{};
  ncclResult_t status = buildLayout(input, output, scatter_dim, gather_dim, handle->world, &layout);
  if (status != ncclSuccess) return status;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return ncclUnhandledCudaError;
  if (device != handle->device) return ncclInvalidUsage;
  // All resources already exist. Validation only queries host metadata; the launch is captured
  // with parameters copied by CUDA. Invalidated captures must still fail before enqueue.
  status = checkCapture(stream, true);
  if (status != ncclSuccess) return status;
  status = validateStorage(handle, input, output, layout);
  if (status != ncclSuccess) return status;

  return enqueueCopies(handle, input, output, layout, stream);
}
