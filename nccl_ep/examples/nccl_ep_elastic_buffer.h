/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * Elastic (GPU + CPU) buffer reference allocator for NCCL EP.
 *
 * An "elastic buffer" is a single contiguous virtual-address (VA) range whose
 * physical backing is split into a GPU segment followed by a CPU (HOST_NUMA)
 * segment, mapped end-to-end with the CUDA Virtual Memory Management (cuMem*)
 * APIs. In Expert Parallelism the receive (dispatch-output) token count is
 * data-dependent: the GPU segment is sized to cover the common case (e.g. the
 * 95th-percentile receive count) while the much larger CPU segment absorbs rare
 * outliers — instead of permanently sizing the whole receive buffer for the
 * absolute worst case in scarce GPU memory.
 *
 * Production integrations normally allocate tensors from their framework
 * (PyTorch, etc.); this header is REFERENCE code that such integrators can copy
 * or adapt. It is intentionally self-contained (depends only on <cuda.h> /
 * <cuda_runtime.h>) and header-only — it is NOT part of libnccl_ep and exposes
 * no supported NCCL EP API.
 *
 * Usage (the buffer becomes an EP receive tensor via an NCCL window):
 *
 *     void* base;
 *     ncclEpElasticBuffer buf;
 *     ncclEpElasticAlloc(&base, &buf, gpu_bytes, cpu_bytes, -1, -1);
 *
 *     ncclWindow_t win;
 *     ncclCommWindowRegister(comm, base, ncclEpElasticTotalBytes(&buf), &win, 0);
 *     // ... fill an ncclEpTensor_t with .win_hdl = win, .win_offset = <byte offset>
 *     //     and pass it as the dispatch output (recv) tensor ...
 *     ncclCommWindowDeregister(comm, win);
 *     ncclEpElasticFree(&buf);
 *
 * Window registration of CPU-backed segments requires NCCL_ELASTIC_BUFFER_REGISTER=1
 * (the default). See the "Elastic (GPU + CPU) receive buffers" section of the
 * NCCL EP README for the supported-mode matrix (HT) and limitations (LL).
 *
 * Adapted from test/apitest/device_api/segmented_allocator.h, specialised to two
 * segments and returning errors instead of aborting, so it is usable as real
 * reference code rather than test-only scaffolding.
 */

#ifndef NCCL_EP_ELASTIC_BUFFER_H
#define NCCL_EP_ELASTIC_BUFFER_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum physical segments in an elastic buffer: one GPU + one CPU. */
#define NCCL_EP_ELASTIC_MAX_SEGMENTS 2

/* Owns the cuMem handles + mapped sizes for an elastic allocation.
 * Treat as opaque; pass by pointer to the helpers below. */
typedef struct ncclEpElasticBuffer {
    void*  base;                 /* VA base (start of the GPU segment, or the CPU
                                  *  segment when gpu_bytes == 0). */
    size_t total_bytes;          /* Sum of granularity-aligned segment sizes. */
    size_t gpu_bytes;            /* Aligned size of the GPU segment (0 if none). */
    size_t cpu_bytes;            /* Aligned size of the CPU segment (0 if none). */
    int    num_segments;         /* Number of mapped segments (0, 1, or 2). */
    CUmemGenericAllocationHandle handles[NCCL_EP_ELASTIC_MAX_SEGMENTS];
    size_t seg_sizes[NCCL_EP_ELASTIC_MAX_SEGMENTS]; /* aligned, in map order */
} ncclEpElasticBuffer;

/* ----- internal error-checking helpers (return on first failure) ----- */

#define NCCL_EP_ELASTIC_CUCHECK(cmd) do {                          \
    CUresult _err = (cmd);                                         \
    if (_err != CUDA_SUCCESS) {                                    \
        const char* _s = NULL; cuGetErrorString(_err, &_s);        \
        fprintf(stderr, "nccl_ep_elastic: CUDA driver error at %s:%d - %s\n", \
                __FILE__, __LINE__, _s ? _s : "?");                \
        return _err;                                               \
    }                                                              \
} while (0)

#define NCCL_EP_ELASTIC_CUDACHECK(cmd) do {                        \
    cudaError_t _err = (cmd);                                      \
    if (_err != cudaSuccess) {                                    \
        fprintf(stderr, "nccl_ep_elastic: CUDA runtime error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err));    \
        return CUDA_ERROR_UNKNOWN;                                 \
    }                                                              \
} while (0)

#define NCCL_EP_ELASTIC_ALIGN(size, gran) \
    ((((size_t)(size) + (size_t)(gran) - 1) / (size_t)(gran)) * (size_t)(gran))

#if CUDART_VERSION >= 11030

/* Pick the cuMem handle types this platform supports (POSIX fd, plus FABRIC
 * when available so the buffer is registrable for internode GIN). */
static inline int ncclEpElasticHandleTypes(CUdevice dev) {
    int types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#if CUDART_VERSION >= 12030
    int fabric = 0;
    cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
    if (fabric) types |= CU_MEM_HANDLE_TYPE_FABRIC;
#endif
    return types;
}

/*
 * Allocate an elastic buffer: a single contiguous VA mapping
 *   [gpu_bytes on the device] ++ [cpu_bytes on HOST_NUMA].
 *
 * Either segment size may be 0 (degenerates to device-only or host-only); both
 * 0 is invalid. Each segment is rounded up to the cuMem allocation granularity,
 * so the realised sizes (and total) may exceed the requested bytes — query them
 * via the accessors below and the recorded fields.
 *
 *   ptr        [out] VA base; equals the start of the GPU segment (or the CPU
 *                    segment when gpu_bytes == 0).
 *   buf        [out] Bookkeeping handle; pass to ncclEpElasticFree.
 *   gpu_bytes  [in]  Requested GPU-segment bytes (0 = no GPU segment).
 *   cpu_bytes  [in]  Requested CPU-segment bytes (0 = no CPU segment).
 *   gpu_dev_id [in]  CUDA device ordinal for the GPU segment (-1 = current device).
 *   numa_id    [in]  Host NUMA node for the CPU segment (-1 = the GPU's HOST_NUMA node).
 *
 * Returns CUDA_SUCCESS on success, otherwise the first failing CUresult.
 * The device is granted read/write access to BOTH segments (so device kernels
 * can store into the CPU segment), as are all peer-accessible GPUs; the host is
 * granted access to the CPU segment.
 */
static inline CUresult ncclEpElasticAlloc(
    void** ptr, ncclEpElasticBuffer* buf,
    size_t gpu_bytes, size_t cpu_bytes,
    int gpu_dev_id, int numa_id)
{
    if (ptr == NULL || buf == NULL || (gpu_bytes == 0 && cpu_bytes == 0)) {
        fprintf(stderr, "nccl_ep_elastic: invalid arguments to ncclEpElasticAlloc\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    memset(buf, 0, sizeof(*buf));

    int cudaDev = 0, devCount = 0;
    NCCL_EP_ELASTIC_CUDACHECK(cudaGetDevice(&cudaDev));
    NCCL_EP_ELASTIC_CUDACHECK(cudaGetDeviceCount(&devCount));
    const int gpuDev = (gpu_dev_id >= 0) ? gpu_dev_id : cudaDev;

    CUdevice cuDev;
    NCCL_EP_ELASTIC_CUCHECK(cuDeviceGet(&cuDev, gpuDev));
    int hostNuma = 0;
    NCCL_EP_ELASTIC_CUCHECK(cuDeviceGetAttribute(&hostNuma, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cuDev));
    const int numaNode = (numa_id >= 0) ? numa_id : hostNuma;

    const int handleTypes = ncclEpElasticHandleTypes(cuDev);

    /* Describe the (up to) two segments in map order: GPU first, then CPU. */
    struct { int isDevice; size_t req; } segs[NCCL_EP_ELASTIC_MAX_SEGMENTS];
    int nseg = 0;
    if (gpu_bytes > 0) { segs[nseg].isDevice = 1; segs[nseg].req = gpu_bytes; nseg++; }
    if (cpu_bytes > 0) { segs[nseg].isDevice = 0; segs[nseg].req = cpu_bytes; nseg++; }

    /* Round each segment up to its location's allocation granularity. */
    size_t reserveSize = 0;
    for (int s = 0; s < nseg; s++) {
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = (CUmemAllocationHandleType)handleTypes;
        if (segs[s].isDevice) {
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id   = gpuDev;
        } else {
            prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            prop.location.id   = numaNode;
        }
        size_t gran = 0;
        NCCL_EP_ELASTIC_CUCHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        buf->seg_sizes[s] = NCCL_EP_ELASTIC_ALIGN(segs[s].req, gran);
        reserveSize += buf->seg_sizes[s];
    }

    /* Reserve the contiguous VA range (device granularity for the base). */
    {
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = gpuDev;
        size_t baseGran = 0;
        NCCL_EP_ELASTIC_CUCHECK(cuMemGetAllocationGranularity(&baseGran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        reserveSize = NCCL_EP_ELASTIC_ALIGN(reserveSize, baseGran);
        NCCL_EP_ELASTIC_CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, reserveSize, baseGran, 0, 0));
    }

    /* Create + map each segment into the reservation. */
    CUdeviceptr cursor = (CUdeviceptr)*ptr;
    for (int s = 0; s < nseg; s++) {
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = (CUmemAllocationHandleType)handleTypes;
        if (segs[s].isDevice) {
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id   = gpuDev;
            int rdma = 0;
            cuDeviceGetAttribute(&rdma, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cuDev);
            if (rdma) prop.allocFlags.gpuDirectRDMACapable = 1;
            buf->gpu_bytes = buf->seg_sizes[s];
        } else {
            prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            prop.location.id   = numaNode;
            buf->cpu_bytes = buf->seg_sizes[s];
        }

        CUresult cerr = cuMemCreate(&buf->handles[s], buf->seg_sizes[s], &prop, 0);
#if CUDART_VERSION >= 12030
        if ((handleTypes & CU_MEM_HANDLE_TYPE_FABRIC) &&
            (cerr == CUDA_ERROR_NOT_PERMITTED || cerr == CUDA_ERROR_NOT_SUPPORTED)) {
            /* Retry without FABRIC if the platform advertises but cannot grant it. */
            prop.requestedHandleTypes =
                (CUmemAllocationHandleType)(handleTypes & ~CU_MEM_HANDLE_TYPE_FABRIC);
            cerr = cuMemCreate(&buf->handles[s], buf->seg_sizes[s], &prop, 0);
        }
#endif
        NCCL_EP_ELASTIC_CUCHECK(cerr);
        buf->num_segments = s + 1;  /* track for cleanup-on-error by the caller */
        NCCL_EP_ELASTIC_CUCHECK(cuMemMap(cursor, buf->seg_sizes[s], 0, buf->handles[s], 0));
        cursor += buf->seg_sizes[s];
    }

    /* Grant access: device R/W to every segment for the local + peer GPUs, and
     * host R/W to the CPU segment so it can be touched from the host too. */
    CUmemAccessDesc access[2];
    memset(access, 0, sizeof(access));
    access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[0].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    access[1].location.id   = numaNode;
    access[1].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    for (int dev = 0; dev < devCount; dev++) {
        int canAccess = (dev == gpuDev);
        if (!canAccess) {
            int p2p = 0;
            if (cudaDeviceCanAccessPeer(&p2p, dev, gpuDev) == cudaSuccess && p2p) canAccess = 1;
        }
        if (!canAccess) continue;
        access[0].location.id = dev;
        CUdeviceptr p = (CUdeviceptr)*ptr;
        for (int s = 0; s < nseg; s++) {
            /* Device-only access for the GPU segment; device + host for the CPU segment. */
            unsigned int count = segs[s].isDevice ? 1u : 2u;
            NCCL_EP_ELASTIC_CUCHECK(cuMemSetAccess(p, buf->seg_sizes[s], access, count));
            p += buf->seg_sizes[s];
        }
    }

    buf->base        = *ptr;
    buf->total_bytes = reserveSize;
    return CUDA_SUCCESS;
}

/* Total granularity-aligned mapped bytes — pass this to ncclCommWindowRegister. */
static inline size_t ncclEpElasticTotalBytes(const ncclEpElasticBuffer* buf) {
    return buf ? buf->total_bytes : 0;
}

/* Aligned size of the GPU segment; also the byte offset of the CPU segment. */
static inline size_t ncclEpElasticGpuBytes(const ncclEpElasticBuffer* buf) {
    return buf ? buf->gpu_bytes : 0;
}

/* Unmap + release every segment and free the VA reservation. Idempotent on a
 * zeroed/already-freed handle. */
static inline CUresult ncclEpElasticFree(ncclEpElasticBuffer* buf) {
    if (buf == NULL || buf->base == NULL) return CUDA_SUCCESS;
    CUdeviceptr cursor = (CUdeviceptr)buf->base;
    for (int s = 0; s < buf->num_segments; s++) {
        NCCL_EP_ELASTIC_CUCHECK(cuMemUnmap(cursor, buf->seg_sizes[s]));
        NCCL_EP_ELASTIC_CUCHECK(cuMemRelease(buf->handles[s]));
        cursor += buf->seg_sizes[s];
    }
    NCCL_EP_ELASTIC_CUCHECK(cuMemAddressFree((CUdeviceptr)buf->base, buf->total_bytes));
    memset(buf, 0, sizeof(*buf));
    return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 11030 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NCCL_EP_ELASTIC_BUFFER_H */
