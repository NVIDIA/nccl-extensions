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

// Capture session of stream: *capturing is false and *id 0 outside capture.
// CUDA gives every capture sequence a unique id, so equal ids mean one session.
inline ncclResult_t captureSession(cudaStream_t stream, bool* capturing, unsigned long long* id) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  unsigned long long captureId = 0;
  const cudaError_t status = cudaStreamGetCaptureInfo(stream, &capture, &captureId);
  if (status == cudaErrorStreamCaptureImplicit) return ncclInvalidUsage;
  if (status != cudaSuccess) return ncclUnhandledCudaError;
  if (capture == cudaStreamCaptureStatusInvalidated) return ncclInvalidUsage;
  *capturing = capture == cudaStreamCaptureStatusActive;
  *id = *capturing ? captureId : 0;
  return ncclSuccess;
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

#ifdef NCCL_ND_FAULT_INJECTION
// Test-only hooks. tests/nd_fault_test.cu compiles this file with the macro to
// fail selected submissions; the shipped library has no hooks.
struct FaultPlan {
  bool entryBarrier = false;
  bool exitBarrier = false;
  long copyIndex = -1;  // issue-order index of the copy to fail; -1 for none
};
FaultPlan faultPlan;
#endif

cudaError_t launchBarrier(ncclDevComm* devComm, cudaStream_t stream, bool exit) {
#ifdef NCCL_ND_FAULT_INJECTION
  if (exit ? faultPlan.exitBarrier : faultPlan.entryBarrier) return cudaErrorLaunchFailure;
#else
  (void)exit;
#endif
  void* arguments[] = {devComm};
  return cudaLaunchKernel(reinterpret_cast<const void*>(rankBarrier), dim3(1), dim3(32), arguments, 0, stream);
}

cudaError_t enqueueCopy(const std::vector<cudaMemcpy3DParms>& copies, size_t index, size_t issued,
                        cudaStream_t stream) {
#ifdef NCCL_ND_FAULT_INJECTION
  if (faultPlan.copyIndex >= 0 && issued == static_cast<size_t>(faultPlan.copyIndex)) return cudaErrorInvalidValue;
#else
  (void)issued;
#endif
  return cudaMemcpy3DAsync(&copies[index], stream);
}

ncclResult_t describePeer(ncclNdHandle_t handle, const ncclNdTensor_t* input, const ncclNdTensor_t* output,
                          const Layout& layout, int peer, cudaMemcpy3DParms* copy) {
  const size_t offset = static_cast<size_t>(input->win_offset);
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
  *copy = cudaMemcpy3DParms{};
  copy->srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(sourceAddress + geometry.inputOffset), geometry.srcPitch,
                                     geometry.width, geometry.srcHeight);
  copy->dstPtr = make_cudaPitchedPtr(static_cast<char*>(output->data) + geometry.outputOffset, geometry.dstPitch,
                                     geometry.width, geometry.dstHeight);
  copy->extent = make_cudaExtent(geometry.width, geometry.height, geometry.depth);
  copy->kind = cudaMemcpyDeviceToDevice;
  return ncclSuccess;
}

// Append one descriptor per peer to copies, in rotated order (rank + index) % world.
// submitCopies relies on this block layout. On failure, copies is left unchanged.
ncclResult_t prepareCopies(ncclNdHandle_t handle, const ncclNdTensor_t* input, const ncclNdTensor_t* output,
                           const Layout& layout, std::vector<cudaMemcpy3DParms>* copies) {
  if (layout.copy.srcPitch > handle->maxPitch || layout.copy.dstPitch > handle->maxPitch)
    return ncclInvalidArgument;
  const size_t first = copies->size();
  try {
    copies->resize(first + handle->world);
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  }
  // Resolve every peer before enqueueing any collective work. These are host
  // metadata queries; AllToAll never allocates or registers device memory.
  for (int index = 0; index < handle->world; ++index) {
    const int peer = static_cast<int>((static_cast<size_t>(index) + handle->rank) % handle->world);
    const ncclResult_t status = describePeer(handle, input, output, layout, peer, &(*copies)[first + index]);
    if (status != ncclSuccess) {
      copies->resize(first);
      return status;
    }
  }
  return ncclSuccess;
}

// Enqueue entry barrier, all copies, and exit barrier. Once the entry barrier is
// enqueued, peers depend on the exit barrier, so it is enqueued even if a copy
// fails. Such a failure is collective-fatal: output is undefined on every rank.
//
// copies holds one block of world descriptors per call, each block in the same
// rotated peer order. Copies are issued peer-major: for each rotation step, the
// copy of every call from that step's peer, in recording order. With several
// calls this keeps one source per rank and step, so ranks that drift apart do
// not converge on the same source GPU at a call boundary. A single call is
// unaffected.
ncclResult_t submitCopies(ncclNdHandle_t handle, const std::vector<cudaMemcpy3DParms>& copies, cudaStream_t stream) {
  const size_t world = static_cast<size_t>(handle->world);
  if (copies.size() % world != 0) return ncclInternalError;
  const size_t calls = copies.size() / world;
  ncclDevComm devComm = handle->devComm;
  cudaError_t launch = launchBarrier(&devComm, stream, false);
  if (launch != cudaSuccess) {
    // Nothing of this operation is enqueued on this rank, but peers may already wait in the entry barrier.
    warn("rank %d: entry barrier launch failed: %s", handle->rank, cudaGetErrorName(launch));
    return ncclUnhandledCudaError;
  }
  ncclResult_t result = ncclSuccess;
  for (size_t step = 0; step < world && result == ncclSuccess; ++step) {
    for (size_t call = 0; call < calls; ++call) {
      const size_t issued = step * calls + call;
      const cudaError_t copyStatus = enqueueCopy(copies, call * world + step, issued, stream);
      if (copyStatus != cudaSuccess) {
        // Peers already passed the entry barrier; still enqueue the exit barrier so they do not hang.
        // The output is undefined and the operation is collective-fatal. Do not retry or fall back.
        warn("rank %d: cudaMemcpy3DAsync from peer %zu for recorded call %zu failed: %s after %zu of %zu copies",
             handle->rank, (static_cast<size_t>(handle->rank) + step) % world, call, cudaGetErrorName(copyStatus),
             issued, copies.size());
        result = ncclUnhandledCudaError;
        break;
      }
    }
  }
  launch = launchBarrier(&devComm, stream, true);
  if (launch != cudaSuccess) {
    warn("rank %d: exit barrier launch failed: %s", handle->rank, cudaGetErrorName(launch));
    return ncclUnhandledCudaError;
  }
  return result;
}

// Calls recorded between ncclNdGroupStart and the outermost ncclNdGroupEnd on
// this host thread. Descriptors are resolved while recording, so caller tensor
// metadata need not outlive the recording call. The first recorded call binds
// the handle, the stream and the stream's capture session.
struct GroupState {
  int depth = 0;
  ncclResult_t error = ncclSuccess;
  ncclNdHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  bool capturing = false;
  unsigned long long captureId = 0;
  std::vector<cudaMemcpy3DParms> copies;
};

thread_local GroupState group;

void clearGroup() {
  group.depth = 0;
  group.error = ncclSuccess;
  group.handle = nullptr;
  group.stream = nullptr;
  group.capturing = false;
  group.captureId = 0;
  group.copies.clear();
}

// Keep the first error; the outermost GroupEnd reports it and enqueues nothing.
ncclResult_t failGroup(ncclResult_t status) {
  if (group.error == ncclSuccess) group.error = status;
  return status;
}

// The stream must stay in the capture session, or outside capture, that the
// first recorded call saw.
ncclResult_t checkGroupSession() {
  bool capturing = false;
  unsigned long long id = 0;
  const ncclResult_t status = captureSession(group.stream, &capturing, &id);
  if (status != ncclSuccess) return status;
  return capturing == group.capturing && id == group.captureId ? ncclSuccess : ncclInvalidUsage;
}

ncclResult_t recordCall(ncclNdHandle_t handle, const ncclNdTensor_t* input, const ncclNdTensor_t* output,
                        const Layout& layout, cudaStream_t stream) {
  if (group.error != ncclSuccess) return group.error;
  // One barrier pair covers the whole group, so every call must share it.
  if (group.handle == nullptr) {
    const ncclResult_t status = captureSession(stream, &group.capturing, &group.captureId);
    if (status != ncclSuccess) return failGroup(status);
    group.handle = handle;
    group.stream = stream;
  } else if (group.handle != handle || group.stream != stream) {
    return failGroup(ncclInvalidUsage);
  } else {
    const ncclResult_t status = checkGroupSession();
    if (status != ncclSuccess) return failGroup(status);
  }
  const ncclResult_t status = prepareCopies(handle, input, output, layout, &group.copies);
  return status == ncclSuccess ? ncclSuccess : failGroup(status);
}

ncclResult_t validateCall(ncclNdHandle_t handle, const ncclNdTensor_t* input, const ncclNdTensor_t* output,
                          int scatter, int gather, cudaStream_t stream, Layout* layout) {
  if (handle == nullptr) return ncclInvalidArgument;
  if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) return ncclInvalidUsage;
  ncclResult_t status = buildLayout(input, output, scatter, gather, handle->world, layout);
  if (status != ncclSuccess) return status;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return ncclUnhandledCudaError;
  if (device != handle->device) return ncclInvalidUsage;
  // All resources already exist. Validation only queries host metadata; the launch is captured
  // with parameters copied by CUDA. Invalidated captures must still fail before enqueue.
  status = checkCapture(stream, true);
  if (status != ncclSuccess) return status;
  return validateStorage(handle, input, output, *layout);
}

}  // namespace nccl_nd

extern "C" ncclResult_t ncclNdGetVersion(int* version) {
  if (version == nullptr) return ncclInvalidArgument;
  *version = NCCL_ND_VERSION_CODE;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclNdCreate(ncclNdHandle_t* handle, ncclComm_t comm) {
  using namespace nccl_nd;
  if (handle != nullptr) *handle = nullptr;
  // Unsupported inside an open group; the rejection also invalidates the batch.
  if (group.depth > 0) return failGroup(ncclInvalidUsage);
  if (handle == nullptr || comm == nullptr) return ncclInvalidArgument;

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
  // Unsupported inside an open group, for any handle; the rejection also invalidates the batch.
  if (group.depth > 0) return failGroup(ncclInvalidUsage);
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
  const bool grouped = group.depth > 0;
  // A failed group returns its first error to every later recorded call, valid or not.
  if (grouped && group.error != ncclSuccess) return group.error;
  Layout layout{};
  ncclResult_t status = validateCall(handle, input, output, scatter_dim, gather_dim, stream, &layout);
  if (status != ncclSuccess) return grouped ? failGroup(status) : status;
  if (grouped) return recordCall(handle, input, output, layout, stream);

  std::vector<cudaMemcpy3DParms> copies;
  status = prepareCopies(handle, input, output, layout, &copies);
  if (status != ncclSuccess) return status;
  return submitCopies(handle, copies, stream);
}

extern "C" ncclResult_t ncclNdGroupStart(void) {
  using namespace nccl_nd;
  if (group.depth == std::numeric_limits<int>::max()) return ncclInvalidUsage;
  ++group.depth;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclNdGroupEnd(void) {
  using namespace nccl_nd;
  if (group.depth == 0) return ncclInvalidUsage;
  // An inner end only closes a level; errors are reported by the outermost end.
  if (--group.depth > 0) return ncclSuccess;
  ncclResult_t status = group.error;
  if (status == ncclSuccess && group.handle != nullptr) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
      status = ncclUnhandledCudaError;
    } else if (device != group.handle->device) {
      status = ncclInvalidUsage;
    } else {
      status = checkGroupSession();
      if (status == ncclSuccess) status = submitCopies(group.handle, group.copies, group.stream);
    }
  }
  clearGroup();
  return status;
}
