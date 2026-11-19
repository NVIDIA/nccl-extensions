/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "ft_collective.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FT_DEVICE_CTA_COUNT       1
#define FT_DEVICE_THREADS_PER_CTA 512
#define FT_MAX_RANKS              1024

/* Coalesced multi-buffer all_to_allv (feedback #4): move K byte-blob buffers
 * (tokens + expert IDs + source IDs + weights + scales) under ONE routing / flag
 * / discovery.  Counts are in ROWS (tokens), shared by all buffers; each buffer
 * has its own row_bytes.  Passed to the kernel by value. */
#define FT_A2AV_MAX_BUFS 8
struct FTMultiBuf {
    ncclWindow_t sendwin[FT_A2AV_MAX_BUFS];
    size_t       sendoff[FT_A2AV_MAX_BUFS];   /* bytes */
    ncclWindow_t recvwin[FT_A2AV_MAX_BUFS];
    size_t       recvoff[FT_A2AV_MAX_BUFS];   /* bytes */
    void*        output [FT_A2AV_MAX_BUFS];   /* caller output (device, bytes)  */
    size_t       row_bytes  [FT_A2AV_MAX_BUFS];
    size_t       recv_stride[FT_A2AV_MAX_BUFS];/* per-source scratch slot (bytes)*/
    int          nbuf;
};

// =============================================================================
// Internal helpers
// =============================================================================

/*
 * ftCopy — strided device copy of `count` elements of T from src to dst.
 *
 * Uses 128-bit (int4) loads/stores for the bulk when both pointers are 16-byte
 * aligned — up to 4x the bytes per memory transaction versus scalar T-sized
 * access, which is what lets a copy loop approach NVLink/HBM bandwidth — then a
 * scalar tail for the trailing < 16 bytes (no source over-read).  Falls back to
 * fully scalar when the pointers are not 16-byte aligned, so it is safe for
 * arbitrary counts and offsets.
 *
 * Callers get the vectorized path on every count once each per-rank recv slice
 * starts on a 16-byte boundary; see FT_RECV_STRIDE below, which pads the per-
 * rank stride so the slice base is always aligned.
 */
template <typename T>
__device__ __forceinline__ void ftCopy(T* dst, const T* src,
                                        size_t count, int tid, int nthreads) {
    if (((((uintptr_t)dst) | ((uintptr_t)src)) & 0xF) == 0) {
        const size_t bytes = count * sizeof(T);
        const size_t n4    = bytes >> 4;                  /* whole 16B words */
        int4*       d4 = (int4*)dst;
        const int4* s4 = (const int4*)src;
        for (size_t i = tid; i < n4; i += nthreads)
            d4[i] = s4[i];
        /* Scalar tail for the trailing (bytes % 16) bytes. */
        for (size_t i = n4 * (16 / sizeof(T)) + tid; i < count; i += nthreads)
            dst[i] = src[i];
    } else {
        for (size_t i = tid; i < count; i += nthreads)
            dst[i] = src[i];
    }
}

/* Per-rank recv-slice stride in bytes, padded up to a 16-byte multiple so every
 * rank's slice base is 16-byte aligned (a prerequisite for ftCopy's vectorized
 * path on arbitrary counts).  The Python/host side computes the identical value
 * (round_up(count*sizeof(T), 16)) when sizing the recv buffer and reading back
 * per-rank slices, so the layout agrees on both sides without an ABI change. */
#define FT_RECV_STRIDE(chunk_bytes) (((chunk_bytes) + 15) & ~((size_t)15))


// =============================================================================
// LSA AllGather kernel
// =============================================================================

/*
 * FTAllGatherKernel (LSA path)
 *
 * Fault-tolerant AllGather using NVLink symmetric memory.  Direct peer-to-peer
 * writes via ncclGetLsaPointer; per-rank readywin barrier; per-rank flag
 * polling for completion.
 *
 * On timeout, thread 0 sets *d_error = FT_TIMEOUT via system-scope release
 * (sticky: never cleared by the kernel).
 */
template <typename T>
__global__ void FTAllGatherKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    if (tid < devComm.nRanks)
        s_active_mask[tid] = active_mask_in[tid];
    __syncthreads();

    /* Step 1: signal readiness — PUSH our generation into each active peer's
     * slot[myRank] (a posted NVLink store; never stalls on a dead peer). */
    for (int p = tid; p < devComm.nRanks; p += nthreads) {
        if (!s_active_mask[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* Step 2: detect liveness by polling our OWN local slots for >= generation
     * (lane r polls slot r).  No peer memory is read, so a killed peer cannot
     * stall this loop — its slot never reaches `generation` and the bounded
     * timeout fires.  s_can_push[r] gates every later write to peer r. */
    __shared__ bool s_can_push[FT_MAX_RANKS];
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        bool can = false;
        if (s_active_mask[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)   /* expected peer never signaled readiness: flag but keep
                         * going for partial results (matches the GIN barrier). */
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        s_can_push[r] = can;
    }
    __syncthreads();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    /* Step 3: Self-copy + own completion flag. */
    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    ftCopy<T>(localRecvPtr, localSendPtr, count, tid, nthreads);

    if (tid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store(1, cuda::memory_order_release);
    }

    /* Phase 4a: push local send buffer into each ready peer's recv slice. */
    for (int r = 0; r < devComm.nRanks; r++) {
        if (!s_can_push[r]) continue;
        T* peerRecvPtr = (T*)ncclGetLsaPointer(
            recvwin, recvoffset + (size_t)devComm.rank * chunk_stride, r);
        ftCopy<T>(peerRecvPtr, localSendPtr, count, tid, nthreads);
    }
    __syncthreads();

    /* Phase 4b: one system fence to order all NVLink writes before flags. */
    __threadfence_system();

    /* Phase 4c: signal completion to all ready peers (parallel over peers). */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store(1, cuda::memory_order_release);
    }

    /* Step 5: Poll flagswin[r] for each expected inbound push, parallel over
     * peers.  Each lane judges its own peer against the shared absolute
     * deadline, so a timeout on one peer no longer aborts polling of the rest
     * (it would expire at the same deadline anyway) and the per-peer waits
     * overlap.  active_mask_out is initialized cooperatively first. */
    for (int r = tid; r < devComm.nRanks; r += nthreads)
        active_mask_out[r] = (r == devComm.rank);
    __syncthreads();

    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        while (af.load(cuda::memory_order_acquire) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
        } else {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    __syncthreads();
    /* No readywin reset: the PUSH/generation model never resets — peers signal
     * a strictly increasing generation, so a stale prior-iteration value reads
     * as < generation and is correctly treated as "not ready yet". */
}

// =============================================================================
// LSA AllReduce kernel
// =============================================================================

/*
 * FTAllReduceKernel (LSA path)
 *
 * Gather-then-reduce.  Identical fault-tolerance logic to FTAllGatherKernel;
 * adds a cooperative reduction step after all data has arrived.
 */
template <typename T>
__global__ void FTAllReduceKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    T*           output,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    __shared__ bool s_result_mask[FT_MAX_RANKS];
    if (tid < devComm.nRanks) {
        s_active_mask[tid] = active_mask_in[tid];
        s_result_mask[tid] = false;
    }
    __syncthreads();

    /* Signal readiness — PUSH generation into each active peer's slot[myRank]
     * (see FTAllGatherKernel Step 1). */
    for (int p = tid; p < devComm.nRanks; p += nthreads) {
        if (!s_active_mask[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* Liveness: poll OWN local slots for >= generation (see FTAllGatherKernel
     * Step 2) — no peer memory is read, so a dead peer cannot stall the loop. */
    __shared__ bool s_can_push[FT_MAX_RANKS];
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        bool can = false;
        if (s_active_mask[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)   /* expected peer never signaled readiness: flag but keep
                         * going for partial results (matches the GIN barrier). */
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        s_can_push[r] = can;
    }
    __syncthreads();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    ftCopy<T>(localRecvPtr, localSendPtr, count, tid, nthreads);

    if (tid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store(1, cuda::memory_order_release);
    }

    /* Phase 4a: push local send buffer into each ready peer's recv slice. */
    for (int r = 0; r < devComm.nRanks; r++) {
        if (!s_can_push[r]) continue;
        T* peerRecvPtr = (T*)ncclGetLsaPointer(
            recvwin, recvoffset + (size_t)devComm.rank * chunk_stride, r);
        ftCopy<T>(peerRecvPtr, localSendPtr, count, tid, nthreads);
    }
    __syncthreads();

    /* Phase 4b: one system fence to order all NVLink writes before flags. */
    __threadfence_system();

    /* Phase 4c: signal completion to all ready peers (parallel over peers). */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store(1, cuda::memory_order_release);
    }

    /* Step 5: poll inbound flags, parallel over peers (see FTAllGatherKernel).
     * Sets both active_mask_out and s_result_mask (consumed by the reduction). */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        active_mask_out[r] = (r == devComm.rank);
        if (r == devComm.rank) s_result_mask[r] = true;
    }
    __syncthreads();

    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        while (af.load(cuda::memory_order_acquire) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
            s_result_mask[r]   = true;
        } else {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }

    __syncthreads();

    /* Cooperative reduction over arrived ranks.  Vectorized (128-bit) path when
     * the recv base and output are 16-byte aligned: each thread reduces a whole
     * int4 (4 floats / 8 half|bf16) per step, cutting instruction count and
     * improving HBM load coalescing.  Scalar fallback / tail otherwise.
     * NOTE: a single CTA still caps this at one SM's HBM throughput — the large
     * remaining gap to NCCL needs a multi-CTA reduction. */
    T* localRecvBase = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    const size_t stride_elems = chunk_stride / sizeof(T);  /* padded slice stride */
    const int    VEC = (int)(16 / sizeof(T));              /* elements per int4 */
    const bool   canVec =
        ((((uintptr_t)localRecvBase) | ((uintptr_t)output)) & 0xF) == 0;

    if (canVec) {
        const size_t nvec    = count / VEC;
        const size_t vstride = chunk_stride / 16;          /* int4 stride per rank */
        const int4*  baseV   = (const int4*)localRecvBase;
        int4*        outV    = (int4*)output;
        for (size_t v = tid; v < nvec; v += nthreads) {
            float acc[16 / 2];                             /* max VEC (half/bf16) */
            #pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] = 0.0f;
            for (int r = 0; r < devComm.nRanks; r++) {
                if (!s_result_mask[r]) continue;
                int4     packed = baseV[(size_t)r * vstride + v];
                const T* el     = (const T*)&packed;
                #pragma unroll
                for (int k = 0; k < VEC; k++) acc[k] += (float)el[k];
            }
            int4 outp;
            T*   oel = (T*)&outp;
            #pragma unroll
            for (int k = 0; k < VEC; k++) oel[k] = (T)acc[k];
            outV[v] = outp;
        }
        /* Scalar tail for the trailing count % VEC elements. */
        for (size_t i = nvec * VEC + tid; i < count; i += nthreads) {
            float sum = 0.0f;
            for (int r = 0; r < devComm.nRanks; r++)
                if (s_result_mask[r]) sum += (float)localRecvBase[(size_t)r * stride_elems + i];
            output[i] = (T)sum;
        }
    } else {
        for (int i = tid; i < (int)count; i += nthreads) {
            float sum = 0.0f;
            for (int r = 0; r < devComm.nRanks; r++)
                if (s_result_mask[r]) sum += (float)localRecvBase[(size_t)r * stride_elems + i];
            output[i] = (T)sum;
        }
    }

    /* No readywin reset (PUSH/generation model — see FTAllGatherKernel). */
}

// =============================================================================
// LSA multi-CTA kernels (cooperative groups)
// =============================================================================

/*
 * FTAllGatherKernelMultiCTA / FTAllReduceKernelMultiCTA (LSA path, large counts)
 *
 * Same protocol and fault-tolerance semantics as the single-CTA LSA kernels,
 * but spread across a cooperative grid of many CTAs so the data movement (and,
 * for AllReduce, the reduction) uses many SMs instead of one — the only way
 * past the single-SM NVLink/HBM ceiling.  Launched via
 * cudaLaunchCooperativeKernel so cg::grid_group::sync() provides the grid-wide
 * barriers that replace the single-block __syncthreads() phase boundaries.
 *
 * Cross-block state that was per-block shared (s_can_push) lives in the global
 * d_can_push[]; the AllReduce result mask is read back from active_mask_out[].
 *
 * Cross-rank producer ordering (data must be visible to a peer before that
 * peer sees our flag): because the push is partitioned across the whole grid,
 * a single flag-signaling thread's fence is insufficient.  The sequence
 *   push -> grid.sync -> __threadfence_system() [all threads] -> grid.sync -> flag(release)
 * guarantees every thread has flushed its own NVLink stores to system scope
 * before any flag is signaled.
 *
 * Liveness: a dead peer only stalls the bounded ready-/flag-poll loops (they
 * break on timeout), after which every local thread reaches the next
 * grid.sync(); the grid barrier itself never waits on a remote rank.
 */
template <typename T>
__global__ void FTAllGatherKernelMultiCTA(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    /* P0: signal readiness — PUSH our generation into each active peer's
     * slot[myRank] (posted NVLink store; never stalls on a dead peer). */
    for (int p = gtid; p < devComm.nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* P1: liveness — poll OWN local slots for >= generation (one global thread
     * per peer) → d_can_push[] (global).  No peer memory is read, so a killed
     * peer cannot stall this loop; its slot never reaches `generation`. */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        bool can = false;
        if (active_mask_in[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)   /* expected peer never signaled readiness: flag but keep
                         * going for partial results (matches the GIN barrier). */
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[r] = can;
    }
    grid.sync();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    /* P2: self-copy + push, partitioned across the whole grid. */
    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    ftCopy<T>(localRecvPtr, localSendPtr, count, gtid, gnthreads);   /* self-copy (local) */
    if (gtid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store(1, cuda::memory_order_release);
    }
    /* Push to peers with CTAs mapped to DISTINCT peers (task % nRanks), so all
     * outbound NVLink links are driven concurrently — instead of the whole grid
     * copying to one peer at a time, which serialized the egress links.  Each
     * (peer, slice) is copied by one CTA's threads; nSub CTAs split each peer's
     * count.  Inactive/self peers (d_can_push false) are skipped. */
    {
        const int nRanks = devComm.nRanks;
        const int nSub   = (gridDim.x + nRanks - 1) / nRanks;   /* CTAs per peer */
        const int EPV    = (int)(16 / sizeof(T));               /* elems per 16B */
        size_t per = (count + (size_t)nSub - 1) / (size_t)nSub;
        per = ((per + EPV - 1) / EPV) * EPV;   /* 16B-align each slice start so
                                                * ftCopy stays 128-bit vectorized */
        for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
            int r = task % nRanks;
            if (!d_can_push[r]) continue;
            size_t start = (size_t)(task / nRanks) * per;
            if (start >= count) continue;
            size_t len = (count - start < per) ? (count - start) : per;
            T* peerRecvPtr = (T*)ncclGetLsaPointer(
                recvwin, recvoffset + (size_t)devComm.rank * chunk_stride, r);
            ftCopy<T>(peerRecvPtr + start, localSendPtr + start, len,
                      threadIdx.x, blockDim.x);
        }
    }

    /* Producer ordering: every thread flushes its NVLink stores to system
     * scope before any flag is signaled (see kernel header). */
    grid.sync();
    __threadfence_system();
    grid.sync();

    /* P3: signal completion to all ready peers (one thread per peer). */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        if (!d_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store(1, cuda::memory_order_release);
    }

    /* P4: init output mask + poll inbound flags (one thread per peer). */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        active_mask_out[r] = (r == devComm.rank);
        if (r == devComm.rank || !d_can_push[r]) continue;
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        while (af.load(cuda::memory_order_acquire) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
        } else {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    /* No trailing grid.sync: active_mask_out is consumed by the host after the
     * kernel completes (stream sync), and nothing else in-kernel depends on it.
     * No readywin reset (PUSH/generation model — see FTAllGatherKernel). */
}

// =============================================================================
// LSA AllToAll kernels (Phase 0: uniform)
// =============================================================================

/*
 * FTAllToAllKernel / FTAllToAllKernelMultiCTA (LSA path)
 *
 * Uniform all-to-all.  The send buffer holds nRanks contiguous `count`-element
 * chunks; chunk j is destined for rank j and lands in rank j's recv slot[myRank]
 * (16-byte padded stride, identical recv layout to AllGather).  These mirror the
 * AllGather kernels EXACTLY except the push source is per-destination
 * (localSendPtr + peer*count) instead of a single shared chunk — so all
 * readiness / flag / timeout / mask logic is shared verbatim.  See
 * FTAllGatherKernel / FTAllGatherKernelMultiCTA for the annotated original.
 *
 * NOTE: send chunk j starts at j*count elements; if count*sizeof(T) is not a
 * multiple of 16 the source is unaligned and ftCopy falls back to scalar for
 * that chunk (correct, just slower) — Phase 1 pads the send stride.
 */
template <typename T>
__global__ void FTAllToAllKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    if (tid < devComm.nRanks)
        s_active_mask[tid] = active_mask_in[tid];
    __syncthreads();

    /* Step 1: signal readiness (see FTAllGatherKernel). */
    for (int p = tid; p < devComm.nRanks; p += nthreads) {
        if (!s_active_mask[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* Step 2: liveness — poll OWN local slots (see FTAllGatherKernel). */
    __shared__ bool s_can_push[FT_MAX_RANKS];
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        bool can = false;
        if (s_active_mask[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        s_can_push[r] = can;
    }
    __syncthreads();

    /* Give the flag poll its own timeout budget: a dead peer makes the readiness
     * poll above spend the whole budget, which would leave zero slack for the
     * flag poll below and drop live peers whose flags land a hair late.  Only
     * live peers (s_can_push) are polled for flags, so this reset costs ~nothing
     * in the common case.  Matters for the collective FT barrier, which is the
     * one collective that must poll the failing rank itself (no prior ft_converge).*/
    startCycle = clock64();

    /* Step 3: self-copy (our send chunk for ourselves) + own completion flag. */
    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    ftCopy<T>(localRecvPtr, localSendPtr + (size_t)devComm.rank * count,
              count, tid, nthreads);   /* all_to_all: our chunk for ourselves */

    if (tid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store(1, cuda::memory_order_release);
    }

    /* Phase 4a: push per-destination chunk r into peer r's recv slot[myRank]. */
    for (int r = 0; r < devComm.nRanks; r++) {
        if (!s_can_push[r]) continue;
        T* peerRecvPtr = (T*)ncclGetLsaPointer(
            recvwin, recvoffset + (size_t)devComm.rank * chunk_stride, r);
        ftCopy<T>(peerRecvPtr, localSendPtr + (size_t)r * count,
                  count, tid, nthreads);   /* all_to_all: chunk r -> peer r */
    }
    __syncthreads();

    /* Phase 4b: one system fence orders all NVLink writes before flags. */
    __threadfence_system();

    /* Phase 4c: signal completion to all ready peers. */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store(1, cuda::memory_order_release);
    }

    /* Step 5: poll inbound flags + build active_mask_out (see FTAllGatherKernel). */
    for (int r = tid; r < devComm.nRanks; r += nthreads)
        active_mask_out[r] = (r == devComm.rank);
    __syncthreads();

    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        while (af.load(cuda::memory_order_acquire) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
        } else {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    __syncthreads();
}

/*
 * FTAllToAllvKernel (LSA, single-CTA, Phase 1: variable per-destination counts)
 *
 * Variable all-to-all.  Rank r sends send_counts[d] elements (at send_displs[d]
 * in its send buffer) to rank d, landing in rank d's FIXED-STRIDE scratch
 * slot[r] (recv_stride bytes per source, holding up to max_per_dest elements).
 * The receiver then compacts the per-source slots into its packed output on the
 * host side using the (caller-supplied) recv counts/displacements — so the
 * kernel needs no receiver-side displacement metadata (senders write a fixed
 * per-source slot).  Identical readiness / flag / timeout / mask logic to
 * FTAllToAllKernel; only the length + source offset are per-destination and the
 * slot stride is the fixed max stride instead of count*sizeof(T).
 *
 * send_counts[d] : elements this rank sends to rank d (device, nRanks).
 * send_displs[d] : element offset of that chunk in the send buffer (device).
 * recv_stride    : fixed per-source slot stride in BYTES (>= max_per_dest*sizeof(T)).
 */
template <typename T>
__global__ void FTAllToAllvKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    const int*    send_counts,
    const size_t* send_displs,
    size_t        recv_stride,
    T*            output,        /* recv output (compaction target)             */
    const int*    recv_counts,   /* packed mode: elems from each source (nRanks) */
    const size_t* recv_displs,   /* packed mode: packed offset per source       */
    int*          recv_counts_out,/* auto mode: DISCOVERED counts out (or NULL)  */
    size_t        recv_capacity, /* auto mode: >0 -> output slot stride (rows)   */
    size_t        row_size,      /* elements per row/token (1 = element counts)  */
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    /* Auto-recv-sizing (recv_capacity>0): the receiver DISCOVERS how much each
     * source sent it from the completion flag (sender stores count+1), writes it
     * to recv_counts_out, and packs each source into a fixed CAPACITY slot in
     * output.  packed mode (recv_capacity==0): uses caller recv_counts/displs.
     * Counts/displs/capacity are in ROWS of row_size elements (row_size=1 = the
     * original per-element semantics); lengths and offsets are scaled by
     * row_size, discovered counts come back in rows. */
    int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    if (tid < devComm.nRanks)
        s_active_mask[tid] = active_mask_in[tid];
    __syncthreads();

    /* Step 1: signal readiness (see FTAllGatherKernel). */
    for (int p = tid; p < devComm.nRanks; p += nthreads) {
        if (!s_active_mask[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* Step 2: liveness — poll OWN local slots (see FTAllGatherKernel). */
    __shared__ bool s_can_push[FT_MAX_RANKS];
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        bool can = false;
        if (s_active_mask[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        s_can_push[r] = can;
    }
    __syncthreads();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    /* Step 3: self-copy (our chunk for ourselves) into our own slot + own flag. */
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * recv_stride);
    ftCopy<T>(localRecvPtr, localSendPtr + send_displs[devComm.rank] * row_size,
              (size_t)send_counts[devComm.rank] * row_size, tid, nthreads);

    if (tid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store((uint64_t)send_counts[devComm.rank] + 1, cuda::memory_order_release);
    }

    /* Phase 4a: push send_counts[r] rows to peer r's scratch slot[myRank]. */
    for (int r = 0; r < devComm.nRanks; r++) {
        if (!s_can_push[r]) continue;
        T* peerRecvPtr = (T*)ncclGetLsaPointer(
            recvwin, recvoffset + (size_t)devComm.rank * recv_stride, r);
        ftCopy<T>(peerRecvPtr, localSendPtr + send_displs[r] * row_size,
                  (size_t)send_counts[r] * row_size, tid, nthreads);
    }
    __syncthreads();
    __threadfence_system();

    /* Phase 4c: signal completion — the flag also CARRIES the count (count+1) so
     * the receiver can discover recv_counts without a separate exchange. */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store((uint64_t)send_counts[r] + 1, cuda::memory_order_release);
    }

    /* Step 5: poll inbound flags + build active_mask_out (+ discover recv counts). */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        active_mask_out[r] = (r == devComm.rank);
        if (recv_counts_out && r == devComm.rank)
            recv_counts_out[r] = send_counts[r];   /* self contribution */
    }
    __syncthreads();

    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_can_push[r]) {
            /* Not live: report 0 so compaction never packs a stale slot. */
            if (recv_counts_out && r != devComm.rank) recv_counts_out[r] = 0;
            continue;
        }
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        uint64_t fv = 0;
        while ((fv = af.load(cuda::memory_order_acquire)) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
            size_t rc = (size_t)(fv - 1);
            if (recv_capacity > 0 && rc > recv_capacity) rc = recv_capacity;
            if (recv_counts_out) recv_counts_out[r] = (int)rc;
        } else {
            if (recv_counts_out) recv_counts_out[r] = 0;
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    __syncthreads();   /* arrived data + discovered counts visible before compaction. */

    /* Fused compaction (one kernel, no host memcpy loop).  Auto mode
     * (recv_capacity>0): discovered counts, fixed capacity slots.  Packed mode:
     * caller counts/displs.  A stale (dead-source) slot is packed too — the mask
     * reports it and the caller discards / zeros that segment. */
    for (int s = 0; s < devComm.nRanks; s++) {
        size_t rc   = (recv_capacity > 0) ? (size_t)recv_counts_out[s]
                                          : (size_t)recv_counts[s];
        if (rc == 0) continue;
        size_t doff = (recv_capacity > 0) ? (size_t)s * recv_capacity
                                          : recv_displs[s];
        T* slotPtr = (T*)ncclGetLocalPointer(
            recvwin, recvoffset + (size_t)s * recv_stride);
        ftCopy<T>(output + doff * row_size, slotPtr, rc * row_size, tid, nthreads);
    }
}

template <typename T>
__global__ void FTAllToAllKernelMultiCTA(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    /* P0: signal readiness (see FTAllGatherKernelMultiCTA). */
    for (int p = gtid; p < devComm.nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* P1: liveness -> d_can_push (see FTAllGatherKernelMultiCTA). */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        bool can = false;
        if (active_mask_in[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[r] = can;
    }
    grid.sync();

    /* Fresh timeout budget for the flag poll (see FTAllToAllKernel): a dead peer
     * consumes the whole readiness budget above; without this reset the flag
     * poll has zero slack and drops live peers whose flags land late.  Only live
     * peers are polled below, so this is ~free in the healthy case. */
    startCycle = clock64();

    /* P2: self-copy + push, CTAs mapped to DISTINCT peers (peer-concurrency). */
    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    ftCopy<T>(localRecvPtr, localSendPtr + (size_t)devComm.rank * count,
              count, gtid, gnthreads);   /* all_to_all: our chunk for ourselves */
    if (gtid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store(1, cuda::memory_order_release);
    }
    {
        const int nRanks = devComm.nRanks;
        const int nSub   = (gridDim.x + nRanks - 1) / nRanks;   /* CTAs per peer */
        const int EPV    = (int)(16 / sizeof(T));
        size_t per = (count + (size_t)nSub - 1) / (size_t)nSub;
        per = ((per + EPV - 1) / EPV) * EPV;   /* 16B-align each slice start */
        for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
            int r = task % nRanks;
            if (!d_can_push[r]) continue;
            size_t start = (size_t)(task / nRanks) * per;
            if (start >= count) continue;
            size_t len = (count - start < per) ? (count - start) : per;
            T* peerRecvPtr = (T*)ncclGetLsaPointer(
                recvwin, recvoffset + (size_t)devComm.rank * chunk_stride, r);
            ftCopy<T>(peerRecvPtr + start,
                      localSendPtr + (size_t)r * count + start, len,
                      threadIdx.x, blockDim.x);   /* all_to_all: chunk r -> peer r */
        }
    }

    grid.sync();
    __threadfence_system();
    grid.sync();

    /* P3: signal completion to all ready peers. */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        if (!d_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store(1, cuda::memory_order_release);
    }

    /* P4: init output mask + poll inbound flags. */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        active_mask_out[r] = (r == devComm.rank);
        if (r == devComm.rank || !d_can_push[r]) continue;
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        while (af.load(cuda::memory_order_acquire) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
        } else {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    /* No trailing grid.sync (see FTAllGatherKernelMultiCTA). */
}

/* Variable all-to-allv, cooperative multi-CTA (Phase 1 perf).  Same as
 * FTAllToAllvKernel but with the peer-concurrency reorder: CTAs map to DISTINCT
 * peers (task % nRanks) so all outbound links run concurrently; nSub CTAs split
 * each peer's send_counts[r] (variable) into 16B-aligned slices.  Load is even
 * per CTA within a peer but uneven ACROSS peers when counts are skewed (a
 * prefix-sum CTA assignment would balance that — future refinement). */
template <typename T>
__global__ void FTAllToAllvKernelMultiCTA(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    const int*    send_counts,
    const size_t* send_displs,
    size_t        recv_stride,
    T*            output,        /* recv output (compaction target) */
    const int*    recv_counts,   /* packed mode (nRanks) */
    const size_t* recv_displs,   /* packed mode */
    int*          recv_counts_out,/* auto mode: DISCOVERED counts out (or NULL) */
    size_t        recv_capacity, /* auto mode: >0 -> output slot stride (rows) */
    size_t        row_size,      /* elements per row/token (1 = element counts) */
    int*          d_ctr,         /* global work-steal counter (balanced push) */
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    if (gtid == 0) *d_ctr = 0;   /* reset the work-steal counter; P1 grid.sync
                                  * publishes it before the push dispenser. */

    /* P0: signal readiness. */
    for (int p = gtid; p < devComm.nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* P1: liveness -> d_can_push. */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        bool can = false;
        if (active_mask_in[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[r] = can;
    }
    grid.sync();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    /* P2: self-copy + variable per-destination push (peer-concurrency). */
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * recv_stride);
    ftCopy<T>(localRecvPtr, localSendPtr + send_displs[devComm.rank] * row_size,
              (size_t)send_counts[devComm.rank] * row_size, gtid, gnthreads);
    if (gtid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store((uint64_t)send_counts[devComm.rank] + 1, cuda::memory_order_release);
    }
    /* Balanced work-steal push: split every peer's variable count into UNIFORM
     * SLICE-element tasks and let blocks grab the next task atomically, so a
     * block on a small peer immediately steals another instead of idling (fixes
     * the cross-peer imbalance of a fixed CTAs-per-peer split).  Task t is
     * interleaved by peer (t % nRanks) so all outbound links stay concurrent.
     * SLICE targets ~4 real tasks/block; stale/oversized tasks are skipped. */
    {
        const int nRanks = devComm.nRanks;
        const int EPV    = (int)(16 / sizeof(T));
        __shared__ size_t s_slice;
        __shared__ int    s_maxsl;
        if (threadIdx.x == 0) {
            size_t total = 0, cmax = 0;
            for (int r = 0; r < nRanks; r++) {
                if (!d_can_push[r]) continue;
                size_t c = (size_t)send_counts[r] * row_size;   /* elements */
                total += c;
                if (c > cmax) cmax = c;
            }
            size_t slice = (total + (size_t)gridDim.x * 4 - 1) / ((size_t)gridDim.x * 4);
            if (slice < (size_t)EPV) slice = (size_t)EPV;
            slice = ((slice + EPV - 1) / EPV) * EPV;   /* 16B-align slice starts */
            s_slice = slice;
            s_maxsl = (int)((cmax + slice - 1) / slice);
        }
        __syncthreads();
        const size_t SLICE = s_slice;
        const int    ntask = nRanks * s_maxsl;

        __shared__ int s_task;
        for (;;) {
            __syncthreads();                                  /* prev task consumed */
            if (threadIdx.x == 0) s_task = atomicAdd(d_ctr, 1);
            __syncthreads();
            int t = s_task;
            if (t >= ntask) break;
            int r = t % nRanks;
            if (d_can_push[r]) {
                size_t cnt   = (size_t)send_counts[r] * row_size;   /* elements */
                size_t start = (size_t)(t / nRanks) * SLICE;
                if (start < cnt) {
                    size_t len = (cnt - start < SLICE) ? (cnt - start) : SLICE;
                    T* peerRecvPtr = (T*)ncclGetLsaPointer(
                        recvwin, recvoffset + (size_t)devComm.rank * recv_stride, r);
                    ftCopy<T>(peerRecvPtr + start,
                              localSendPtr + send_displs[r] * row_size + start, len,
                              threadIdx.x, blockDim.x);
                }
            }
        }
    }

    grid.sync();
    __threadfence_system();
    grid.sync();

    /* P3: signal completion — flag carries count+1 for recv-count discovery. */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        if (!d_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store((uint64_t)send_counts[r] + 1, cuda::memory_order_release);
    }

    /* P4: init output mask + poll inbound flags (+ discover recv counts). */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        active_mask_out[r] = (r == devComm.rank);
        if (recv_counts_out && r == devComm.rank)
            recv_counts_out[r] = send_counts[r];   /* self contribution */
        if (r == devComm.rank) continue;
        if (!d_can_push[r]) {
            /* Not live: report 0 so compaction never packs a stale slot. */
            if (recv_counts_out) recv_counts_out[r] = 0;
            continue;
        }
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        uint64_t fv = 0;
        while ((fv = af.load(cuda::memory_order_acquire)) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
            size_t rc = (size_t)(fv - 1);
            if (recv_capacity > 0 && rc > recv_capacity) rc = recv_capacity;
            if (recv_counts_out) recv_counts_out[r] = (int)rc;
        } else {
            if (recv_counts_out) recv_counts_out[r] = 0;
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }

    /* Fused compaction: grid-wide barrier so every CTA sees the arrived data +
     * discovered counts, then pack each source's fixed scratch slot into output.
     * Auto mode (recv_capacity>0): discovered counts into fixed capacity slots;
     * packed mode: caller counts/displs.  Stale slots are packed too (mask). */
    grid.sync();
    for (int s = 0; s < devComm.nRanks; s++) {
        size_t rc   = (recv_capacity > 0) ? (size_t)recv_counts_out[s]
                                          : (size_t)recv_counts[s];
        if (rc == 0) continue;
        size_t doff = (recv_capacity > 0) ? (size_t)s * recv_capacity
                                          : recv_displs[s];
        T* slotPtr = (T*)ncclGetLocalPointer(
            recvwin, recvoffset + (size_t)s * recv_stride);
        ftCopy<T>(output + doff * row_size, slotPtr, rc * row_size, gtid, gnthreads);
    }
}

/* Variable all-gather (all_gatherv): each rank contributes ONE variable-size
 * chunk (send_counts[myRank] rows) which every rank receives, concatenated in
 * rank order.  Auto-recv-sizing (count-in-flag): each rank broadcasts its chunk
 * to every peer's slot[myRank] and raises a flag carrying count+1, so receivers
 * DISCOVER each source's size in the flag poll — no separate size all-gather.
 * The receiver then prefix-sums the discovered counts on-device and compacts the
 * per-source slots into a PACKED output.  Byte-oriented (dtype-agnostic), LSA. */
__global__ void FTAllGathervKernel(
    ncclWindow_t sendwin, size_t sendoff,      /* my chunk (bytes offset)         */
    ncclWindow_t recvwin, size_t recvoff,      /* nRanks scratch slots            */
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    const int*   send_counts,      /* rows; this rank pushes send_counts[myRank]  */
    int*         recv_counts_out,  /* DISCOVERED rows per source (nRanks)          */
    size_t*      recv_displs_out,  /* scratch: packed row offsets (in-kernel scan) */
    void*        output,           /* PACKED device output (bytes)                 */
    size_t       recv_capacity,    /* max rows any source may send (slot bound)    */
    size_t       row_bytes,        /* bytes per row                                */
    size_t       recv_stride,      /* per-source scratch slot stride (bytes)       */
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;
    const int nRanks = devComm.nRanks;
    const int myRank = devComm.rank;

    /* Readiness: PUSH generation, poll LOCAL slots -> d_can_push. */
    for (int p = gtid; p < nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == myRank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }
    for (int r = gtid; r < nRanks; r += gnthreads) {
        bool can = false;
        if (active_mask_in[r] && r != myRank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[r] = can;
    }
    grid.sync();
    startCycle = clock64();

    /* My contribution (clamped to capacity; over-capacity is flagged, not fatal). */
    size_t myc = (size_t)send_counts[myRank];
    if (myc > recv_capacity) {
        cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
            .store(FT_CAPACITY_EXCEEDED, cuda::memory_order_release);
        myc = recv_capacity;
    }

    /* Self-copy: my chunk -> my own slot[myRank]. */
    {
        char* ls = (char*)ncclGetLocalPointer(sendwin, sendoff);
        char* lr = (char*)ncclGetLocalPointer(
            recvwin, recvoff + (size_t)myRank * recv_stride);
        ftCopy<char>(lr, ls, myc * row_bytes, gtid, gnthreads);
    }
    if (gtid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store((uint64_t)myc + 1, cuda::memory_order_release);
    }

    /* Broadcast push: my chunk -> every live peer's slot[myRank].  CTAs map to
     * DISTINCT peers (task % nRanks) so all egress links run concurrently; nSub
     * CTAs split my chunk. */
    {
        const int nSub = (gridDim.x + nRanks - 1) / nRanks;
        size_t per = (myc + (size_t)nSub - 1) / (size_t)nSub;
        if (per < 1) per = 1;
        for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
            int p = task % nRanks;
            if (p == myRank || !d_can_push[p]) continue;
            size_t start = (size_t)(task / nRanks) * per;
            if (start >= myc) continue;
            size_t len = (myc - start < per) ? (myc - start) : per;
            char* src = (char*)ncclGetLocalPointer(sendwin, sendoff) + start * row_bytes;
            char* dst = (char*)ncclGetLsaPointer(
                recvwin, recvoff + (size_t)myRank * recv_stride, p) + start * row_bytes;
            ftCopy<char>(dst, src, len * row_bytes, threadIdx.x, blockDim.x);
        }
    }
    grid.sync();
    __threadfence_system();
    grid.sync();

    /* Flag every live peer with my count+1 (count-in-flag). */
    for (int p = gtid; p < nRanks; p += gnthreads) {
        if (p == myRank || !d_can_push[p]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store((uint64_t)myc + 1, cuda::memory_order_release);
    }

    /* Discovery: recv_counts_out[s] = flag[s]-1 (0 if timed out); set mask. */
    for (int s = gtid; s < nRanks; s += gnthreads) {
        active_mask_out[s] = (s == myRank);
        if (s == myRank) { recv_counts_out[s] = (int)myc; continue; }
        if (!d_can_push[s]) { recv_counts_out[s] = 0; continue; }
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)s * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        uint64_t fv = 0;
        while ((fv = af.load(cuda::memory_order_acquire)) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[s] = true;
            size_t rc = (size_t)(fv - 1);
            recv_counts_out[s] = (int)(rc > recv_capacity ? recv_capacity : rc);
        } else {
            recv_counts_out[s] = 0;
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    grid.sync();

    /* Exclusive prefix-sum of discovered counts -> packed row offsets. */
    if (gtid == 0) {
        size_t acc = 0;
        for (int s = 0; s < nRanks; s++) {
            recv_displs_out[s] = acc;
            acc += (size_t)recv_counts_out[s];
        }
    }
    grid.sync();

    /* Compaction: each source's slot -> output[displs[s]] (packed concatenation). */
    for (int s = 0; s < nRanks; s++) {
        size_t rc = (size_t)recv_counts_out[s];
        if (rc == 0) continue;
        char* slot = (char*)ncclGetLocalPointer(
            recvwin, recvoff + (size_t)s * recv_stride);
        char* out  = (char*)output + recv_displs_out[s] * row_bytes;
        ftCopy<char>(out, slot, rc * row_bytes, gtid, gnthreads);
    }
}

/* Single-CTA all_gatherv for small messages (avoids cooperative-launch +
 * grid.sync overhead): one block, __syncthreads barriers, shared can_push. Same
 * broadcast-push + count-in-flag + prefix-sum + packed compaction as the
 * cooperative kernel. Dispatched below FT_MULTICTA_THRESHOLD_BYTES. */
__global__ void FTAllGathervKernelSingle(
    ncclWindow_t sendwin, size_t sendoff,
    ncclWindow_t recvwin, size_t recvoff,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    const int*   send_counts,
    int*         recv_counts_out,
    size_t*      recv_displs_out,
    void*        output,
    size_t       recv_capacity,
    size_t       row_bytes,
    size_t       recv_stride,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;
    const int nRanks = devComm.nRanks;
    const int myRank = devComm.rank;

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    __shared__ bool s_can_push[FT_MAX_RANKS];
    if (tid < nRanks) s_active_mask[tid] = active_mask_in[tid];
    __syncthreads();

    for (int p = tid; p < nRanks; p += nthreads) {
        if (!s_active_mask[p] || p == myRank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }
    for (int r = tid; r < nRanks; r += nthreads) {
        bool can = false;
        if (s_active_mask[r] && r != myRank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        s_can_push[r] = can;
    }
    __syncthreads();
    startCycle = clock64();

    size_t myc = (size_t)send_counts[myRank];
    if (myc > recv_capacity) {
        if (tid == 0)
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_CAPACITY_EXCEEDED, cuda::memory_order_release);
        myc = recv_capacity;
    }

    {
        char* ls = (char*)ncclGetLocalPointer(sendwin, sendoff);
        char* lr = (char*)ncclGetLocalPointer(
            recvwin, recvoff + (size_t)myRank * recv_stride);
        ftCopy<char>(lr, ls, myc * row_bytes, tid, nthreads);
    }
    if (tid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store((uint64_t)myc + 1, cuda::memory_order_release);
    }

    for (int r = 0; r < nRanks; r++) {
        if (r == myRank || !s_can_push[r]) continue;
        char* src = (char*)ncclGetLocalPointer(sendwin, sendoff);
        char* dst = (char*)ncclGetLsaPointer(
            recvwin, recvoff + (size_t)myRank * recv_stride, r);
        ftCopy<char>(dst, src, myc * row_bytes, tid, nthreads);
    }
    __syncthreads();
    __threadfence_system();

    for (int r = tid; r < nRanks; r += nthreads) {
        if (r == myRank || !s_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store((uint64_t)myc + 1, cuda::memory_order_release);
    }

    for (int s = tid; s < nRanks; s += nthreads) {
        active_mask_out[s] = (s == myRank);
        if (s == myRank) { recv_counts_out[s] = (int)myc; continue; }
        if (!s_can_push[s]) { recv_counts_out[s] = 0; continue; }
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)s * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        uint64_t fv = 0;
        while ((fv = af.load(cuda::memory_order_acquire)) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[s] = true;
            size_t rc = (size_t)(fv - 1);
            recv_counts_out[s] = (int)(rc > recv_capacity ? recv_capacity : rc);
        } else {
            recv_counts_out[s] = 0;
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    __syncthreads();

    if (tid == 0) {
        size_t acc = 0;
        for (int s = 0; s < nRanks; s++) {
            recv_displs_out[s] = acc;
            acc += (size_t)recv_counts_out[s];
        }
    }
    __syncthreads();

    for (int s = 0; s < nRanks; s++) {
        size_t rc = (size_t)recv_counts_out[s];
        if (rc == 0) continue;
        char* slot = (char*)ncclGetLocalPointer(
            recvwin, recvoff + (size_t)s * recv_stride);
        char* out  = (char*)output + recv_displs_out[s] * row_bytes;
        ftCopy<char>(out, slot, rc * row_bytes, tid, nthreads);
    }
}

/* Variable reduce-scatter (reduce_scatterv): sum every rank's full input vector,
 * scattering the result so rank r owns shard r = [displs[r], counts[r]).  Unlike
 * all_gatherv the shard sizes are a KNOWN partition (counts/displs are inputs,
 * not discovered).  Reuses the reduce-scatter scatter->round1->reduce handshake
 * but with VARIABLE shards and NO broadcast-back (shards stay distributed).
 * Survivor-sum: a dead contributor drops out of the sum.  Typed (float
 * accumulation); LSA only. */
template <typename T>
__global__ void FTReduceScattervKernel(
    ncclWindow_t sendwin, size_t sendoffset,   /* my full input (sum of counts)  */
    ncclWindow_t recvwin, size_t recvoffset,   /* nRanks contributor slots (mine)*/
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    const int*    counts,       /* shard sizes (elements), nRanks                */
    const size_t* displs,       /* shard base offsets in the input, nRanks       */
    T*           output,        /* my reduced shard (counts[myRank] elements)     */
    size_t       recv_stride,   /* per-contributor slot stride (bytes)            */
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;
    const int nRanks = devComm.nRanks;
    const int myRank = devComm.rank;
    const int    VEC = (int)(16 / sizeof(T));
    const size_t strideElems = recv_stride / sizeof(T);
    const size_t myCount = (size_t)counts[myRank];

    /* Readiness: PUSH generation, poll LOCAL slots (flag-only; d_can_push is set
     * by the round-1 poll below, over the confirmed contributor set). */
    for (int p = gtid; p < nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == myRank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }
    for (int r = gtid; r < nRanks; r += gnthreads) {
        if (active_mask_in[r] && r != myRank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            if (af.load(cuda::memory_order_acquire) < generation)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    grid.sync();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    T* sendLocal    = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* scratchLocal = (T*)ncclGetLocalPointer(recvwin, recvoffset);

    /* Phase 1: scatter — push my slice of each shard o to owner o's slot[myRank].
     * CTAs map to DISTINCT owners (task % nRanks); nSub CTAs split each shard. */
    {
        const int nSub = (gridDim.x + nRanks - 1) / nRanks;
        for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
            int o = task % nRanks;
            if (!active_mask_in[o]) continue;
            size_t olen = (size_t)counts[o];
            if (olen == 0) continue;
            size_t obase = displs[o];
            size_t per = (olen + (size_t)nSub - 1) / (size_t)nSub;
            per = ((per + (size_t)VEC - 1) / (size_t)VEC) * (size_t)VEC;
            size_t start = (size_t)(task / nRanks) * per;
            if (start >= olen) continue;
            size_t len = (olen - start < per) ? (olen - start) : per;
            T* dst;
            if (o == myRank)
                dst = scratchLocal + (size_t)myRank * strideElems;
            else
                dst = (T*)ncclGetLsaPointer(
                    recvwin, recvoffset + (size_t)myRank * recv_stride, o);
            ftCopy<T>(dst + start, sendLocal + obase + start, len,
                      threadIdx.x, blockDim.x);
        }
    }
    grid.sync();
    __threadfence_system();
    grid.sync();

    /* Round-1 signal: tell each owner o "myRank delivered" (flag slot myRank). */
    for (int o = gtid; o < nRanks; o += gnthreads) {
        if (o == myRank || !active_mask_in[o]) continue;
        uint64_t* f = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t), o);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*f)
            .store(1, cuda::memory_order_release);
    }
    /* Round-1 poll: I own shard myRank — record who delivered (the reduction's
     * confirmed contributor set) and the result mask. Self always counts. */
    for (int j = gtid; j < nRanks; j += gnthreads) {
        bool got = (j == myRank);
        if (j != myRank && active_mask_in[j]) {
            uint64_t* f = (uint64_t*)ncclGetLocalPointer(
                flagswin, (size_t)j * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*f);
            bool to = false;
            while (af.load(cuda::memory_order_acquire) < 1) {
                if (clock64() - startCycle >= timeout_cycles) { to = true; break; }
            }
            got = !to;
            if (to) cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                        .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[j]      = got;
        active_mask_out[j] = got;
    }
    grid.sync();

    /* Reduce my shard over confirmed contributors -> output (no broadcast-back). */
    const bool canVec =
        ((((uintptr_t)output) | ((uintptr_t)scratchLocal)) & 0xF) == 0;
    if (canVec) {
        const size_t nvec = myCount / VEC;
        int4* oV = (int4*)output;
        for (size_t v = gtid; v < nvec; v += gnthreads) {
            float acc[16 / 2];
            #pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] = 0.0f;
            for (int j = 0; j < nRanks; j++) {
                if (!d_can_push[j]) continue;
                int4 packed = ((const int4*)(scratchLocal + (size_t)j * strideElems))[v];
                const T* el = (const T*)&packed;
                #pragma unroll
                for (int k = 0; k < VEC; k++) acc[k] += (float)el[k];
            }
            int4 outp; T* oel = (T*)&outp;
            #pragma unroll
            for (int k = 0; k < VEC; k++) oel[k] = (T)acc[k];
            oV[v] = outp;
        }
        for (size_t i = nvec * VEC + gtid; i < myCount; i += gnthreads) {
            float sum = 0.0f;
            for (int j = 0; j < nRanks; j++)
                if (d_can_push[j]) sum += (float)scratchLocal[(size_t)j * strideElems + i];
            output[i] = (T)sum;
        }
    } else {
        for (size_t i = gtid; i < myCount; i += gnthreads) {
            float sum = 0.0f;
            for (int j = 0; j < nRanks; j++)
                if (d_can_push[j]) sum += (float)scratchLocal[(size_t)j * strideElems + i];
            output[i] = (T)sum;
        }
    }
}

/* Single-CTA reduce_scatterv for small messages: one block, __syncthreads,
 * shared can_push; scalar reduce (latency-bound at small sizes). Same
 * scatter->round1->survivor-sum as the cooperative kernel, no broadcast-back. */
template <typename T>
__global__ void FTReduceScattervKernelSingle(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    const int*    counts,
    const size_t* displs,
    T*           output,
    size_t       recv_stride,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid = threadIdx.x, nthreads = blockDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;
    const int nRanks = devComm.nRanks, myRank = devComm.rank;
    const size_t strideElems = recv_stride / sizeof(T);
    const size_t myCount = (size_t)counts[myRank];

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    __shared__ bool s_can_push[FT_MAX_RANKS];
    if (tid < nRanks) s_active_mask[tid] = active_mask_in[tid];
    __syncthreads();

    for (int p = tid; p < nRanks; p += nthreads) {
        if (!s_active_mask[p] || p == myRank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }
    for (int r = tid; r < nRanks; r += nthreads) {
        if (s_active_mask[r] && r != myRank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            if (af.load(cuda::memory_order_acquire) < generation)
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    __syncthreads();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    T* sendLocal    = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* scratchLocal = (T*)ncclGetLocalPointer(recvwin, recvoffset);

    for (int o = 0; o < nRanks; o++) {
        if (!s_active_mask[o]) continue;
        size_t olen = (size_t)counts[o];
        if (olen == 0) continue;
        size_t obase = displs[o];
        T* dst;
        if (o == myRank)
            dst = scratchLocal + (size_t)myRank * strideElems;
        else
            dst = (T*)ncclGetLsaPointer(
                recvwin, recvoffset + (size_t)myRank * recv_stride, o);
        ftCopy<T>(dst, sendLocal + obase, olen, tid, nthreads);
    }
    __syncthreads();
    __threadfence_system();

    for (int o = tid; o < nRanks; o += nthreads) {
        if (o == myRank || !s_active_mask[o]) continue;
        uint64_t* f = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t), o);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*f)
            .store(1, cuda::memory_order_release);
    }
    for (int j = tid; j < nRanks; j += nthreads) {
        bool got = (j == myRank);
        if (j != myRank && s_active_mask[j]) {
            uint64_t* f = (uint64_t*)ncclGetLocalPointer(
                flagswin, (size_t)j * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*f);
            bool to = false;
            while (af.load(cuda::memory_order_acquire) < 1) {
                if (clock64() - startCycle >= timeout_cycles) { to = true; break; }
            }
            got = !to;
            if (to) cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                        .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        s_can_push[j]      = got;
        active_mask_out[j] = got;
    }
    __syncthreads();

    for (size_t i = tid; i < myCount; i += nthreads) {
        float sum = 0.0f;
        for (int j = 0; j < nRanks; j++)
            if (s_can_push[j]) sum += (float)scratchLocal[(size_t)j * strideElems + i];
        output[i] = (T)sum;
    }
}

template <typename T>
__global__ void FTAllReduceKernelMultiCTA(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    T*           output,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;

    /* Signal readiness — PUSH generation into each active peer's slot[myRank]
     * (see FTAllGatherKernelMultiCTA P0). */
    for (int p = gtid; p < devComm.nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == devComm.rank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)devComm.rank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }

    /* Liveness — poll OWN local slots for >= generation (no peer reads). */
    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        bool can = false;
        if (active_mask_in[r] && r != devComm.rank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            can = (af.load(cuda::memory_order_acquire) >= generation);
            if (!can)   /* expected peer never signaled readiness: flag but keep
                         * going for partial results (matches the GIN barrier). */
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[r] = can;
    }
    grid.sync();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    ftCopy<T>(localRecvPtr, localSendPtr, count, gtid, gnthreads);   /* self-copy (local) */
    if (gtid == 0) {
        uint64_t* selfFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*selfFlag)
            .store(1, cuda::memory_order_release);
    }
    /* Push to peers with CTAs mapped to DISTINCT peers (task % nRanks), so all
     * outbound NVLink links run concurrently instead of one peer at a time
     * (same as the multi-CTA AllGather). */
    {
        const int nRanks = devComm.nRanks;
        const int nSub   = (gridDim.x + nRanks - 1) / nRanks;   /* CTAs per peer */
        const int EPV    = (int)(16 / sizeof(T));               /* elems per 16B */
        size_t per = (count + (size_t)nSub - 1) / (size_t)nSub;
        per = ((per + EPV - 1) / EPV) * EPV;   /* 16B-align each slice start */
        for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
            int r = task % nRanks;
            if (!d_can_push[r]) continue;
            size_t start = (size_t)(task / nRanks) * per;
            if (start >= count) continue;
            size_t len = (count - start < per) ? (count - start) : per;
            T* peerRecvPtr = (T*)ncclGetLsaPointer(
                recvwin, recvoffset + (size_t)devComm.rank * chunk_stride, r);
            ftCopy<T>(peerRecvPtr + start, localSendPtr + start, len,
                      threadIdx.x, blockDim.x);
        }
    }

    grid.sync();
    __threadfence_system();
    grid.sync();

    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        if (!d_can_push[r]) continue;
        uint64_t* peerFlag = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)devComm.rank * sizeof(uint64_t), r);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerFlag)
            .store(1, cuda::memory_order_release);
    }

    for (int r = gtid; r < devComm.nRanks; r += gnthreads) {
        active_mask_out[r] = (r == devComm.rank);
        if (r == devComm.rank || !d_can_push[r]) continue;
        uint64_t* myFlag = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)r * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*myFlag);
        bool timed_out = false;
        while (af.load(cuda::memory_order_acquire) == 0) {
            if (clock64() - startCycle >= timeout_cycles) { timed_out = true; break; }
        }
        if (!timed_out) {
            active_mask_out[r] = true;
        } else {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    /* Propagate arrived peer data + final mask to the whole grid before the
     * reduction (the flag-poll acquire established visibility on the polling
     * threads; the grid barrier extends it to every reducing thread). */
    grid.sync();

    /* Reduction: load the arrived-rank mask into per-block shared once, then
     * reduce with the same 128-bit vectorized path as the single-CTA kernel,
     * partitioned across the whole grid. */
    __shared__ bool s_result_mask[FT_MAX_RANKS];
    if (threadIdx.x < devComm.nRanks)
        s_result_mask[threadIdx.x] = active_mask_out[threadIdx.x];
    __syncthreads();

    T* localRecvBase = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    const size_t stride_elems = chunk_stride / sizeof(T);
    const int    VEC = (int)(16 / sizeof(T));
    const bool   canVec =
        ((((uintptr_t)localRecvBase) | ((uintptr_t)output)) & 0xF) == 0;

    if (canVec) {
        const size_t nvec    = count / VEC;
        const size_t vstride = chunk_stride / 16;
        const int4*  baseV   = (const int4*)localRecvBase;
        int4*        outV    = (int4*)output;
        for (size_t v = gtid; v < nvec; v += gnthreads) {
            float acc[16 / 2];
            #pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] = 0.0f;
            for (int r = 0; r < devComm.nRanks; r++) {
                if (!s_result_mask[r]) continue;
                int4     packed = baseV[(size_t)r * vstride + v];
                const T* el     = (const T*)&packed;
                #pragma unroll
                for (int k = 0; k < VEC; k++) acc[k] += (float)el[k];
            }
            int4 outp;
            T*   oel = (T*)&outp;
            #pragma unroll
            for (int k = 0; k < VEC; k++) oel[k] = (T)acc[k];
            outV[v] = outp;
        }
        for (size_t i = nvec * VEC + gtid; i < count; i += gnthreads) {
            float sum = 0.0f;
            for (int r = 0; r < devComm.nRanks; r++)
                if (s_result_mask[r]) sum += (float)localRecvBase[(size_t)r * stride_elems + i];
            output[i] = (T)sum;
        }
    } else {
        for (size_t i = gtid; i < count; i += gnthreads) {
            float sum = 0.0f;
            for (int r = 0; r < devComm.nRanks; r++)
                if (s_result_mask[r]) sum += (float)localRecvBase[(size_t)r * stride_elems + i];
            output[i] = (T)sum;
        }
    }

    /* No readywin reset (PUSH/generation model — see FTAllGatherKernel). */
}

// =============================================================================
// LSA reduce-scatter AllReduce kernel (cooperative, large counts, full set)
// =============================================================================

/*
 * FTAllReduceRSKernelMultiCTA (LSA path, window API only)
 *
 * Bandwidth-optimal AllReduce = reduce-scatter + allgather, O(1) data per
 * element vs gather-then-reduce's O(nRanks).  Runs only above
 * rs_threshold_bytes AND when the FULL set is active (the launch dispatch
 * enforces both); any failure falls back to gather-then-reduce on the next
 * iteration, so this kernel assumes nRanks live owners — shard s is owned by
 * rank s, every rank contributes to and receives every shard.
 *
 * In-place on the symmetric send/output buffer (window API: output aliases
 * sendwin), so no extra result buffer:
 *   Phase 1 (scatter): every rank pushes its slice of shard s into owner s's
 *     recvwin scratch slot[myRank]; owner s reduces all slots -> reduced shard
 *     written into its own output[shard s].
 *   Phase 2 (allgather): every owner broadcasts its reduced shard into every
 *     peer's output[shard s].  After phase 2 output holds the full result.
 * Two flag rounds reuse the nRanks flag slots monotonically (1 then 2): every
 * active peer sends to a given rank exactly once per round, so its flag slot
 * advances 0 -> 1 (scatter) -> 2 (allgather).
 *
 * Producer ordering for both rounds is the proven
 *   push -> grid.sync -> threadfence_system(all) -> grid.sync -> flag(release).
 * A peer that dies mid-kernel trips a bounded ready/flag-poll timeout -> the
 * result is incomplete and FT_TIMEOUT is raised; the caller discards it and the
 * store barrier drops the peer, after which AllReduce falls back to
 * gather-then-reduce over the survivors.
 */

/* Reduce one chunk [cstart, cstart+clen) of the owner's shard from the
 * contributor scratch slots into output[obase+cstart..], summing over the
 * confirmed-contributor set d_can_push.  Distributed over threads [rtid, rtid+rn)
 * (a caller-chosen CTA subset).  Same 128-bit vectorized path as the bulk
 * reduce; cstart/clen are 16B-aligned by the caller so vectorization holds. */
template <typename T>
__device__ __forceinline__ void ftRSReduceChunk(
    T* output, size_t obase, const T* scratchLocal, const bool* d_can_push,
    int nRanks, size_t shardElems, size_t cstart, size_t clen, bool canVec,
    int VEC, size_t rtid, size_t rn)
{
    T* dst = output + obase + cstart;
    if (canVec) {
        const size_t nvec = clen / VEC;
        int4* dV = (int4*)dst;
        for (size_t v = rtid; v < nvec; v += rn) {
            float acc[16 / 2];
            #pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] = 0.0f;
            for (int j = 0; j < nRanks; j++) {
                if (!d_can_push[j]) continue;
                int4 packed = ((const int4*)(scratchLocal + (size_t)j * shardElems + cstart))[v];
                const T* el = (const T*)&packed;
                #pragma unroll
                for (int k = 0; k < VEC; k++) acc[k] += (float)el[k];
            }
            int4 outp; T* oel = (T*)&outp;
            #pragma unroll
            for (int k = 0; k < VEC; k++) oel[k] = (T)acc[k];
            dV[v] = outp;
        }
        for (size_t i = nvec * VEC + rtid; i < clen; i += rn) {
            float sum = 0.0f;
            for (int j = 0; j < nRanks; j++)
                if (d_can_push[j]) sum += (float)scratchLocal[(size_t)j * shardElems + cstart + i];
            dst[i] = (T)sum;
        }
    } else {
        for (size_t i = rtid; i < clen; i += rn) {
            float sum = 0.0f;
            for (int j = 0; j < nRanks; j++)
                if (d_can_push[j]) sum += (float)scratchLocal[(size_t)j * shardElems + cstart + i];
            dst[i] = (T)sum;
        }
    }
}

/* Broadcast one chunk [cstart, cstart+clen) of the owner's reduced shard to
 * every active peer, over broadcaster CTAs [0, nBc) identified by dense id bbid.
 * CTAs map to DISTINCT peers (task % nRanks) so all links run concurrently. */
template <typename T>
__device__ __forceinline__ void ftRSBroadcastChunk(
    ncclWindow_t sendwin, size_t sendoffset, const T* output,
    const bool* active_mask_in, int nRanks, int myRank, size_t obase,
    size_t cstart, size_t clen, int VEC, int bbid, int nBc)
{
    if (nBc <= 0 || clen == 0) return;
    const int nSub = (nBc + nRanks - 1) / nRanks;   /* broadcaster CTAs per peer */
    size_t per = (clen + (size_t)nSub - 1) / (size_t)nSub;
    per = ((per + (size_t)VEC - 1) / (size_t)VEC) * (size_t)VEC;
    for (int task = bbid; task < nRanks * nSub; task += nBc) {
        int p = task % nRanks;
        if (p == myRank || !active_mask_in[p]) continue;
        size_t start = (size_t)(task / nRanks) * per;
        if (start >= clen) continue;
        size_t len = (clen - start < per) ? (clen - start) : per;
        T* dst = (T*)ncclGetLsaPointer(
            sendwin, sendoffset + (obase + cstart + start) * sizeof(T), p);
        ftCopy<T>(dst, output + obase + cstart + start, len,
                  threadIdx.x, blockDim.x);
    }
}

template <typename T>
__global__ void FTAllReduceRSKernelMultiCTA(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin,  size_t recvoffset,
    ncclWindow_t flagswin,
    ncclWindow_t readywin,
    T*           output,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const uint64_t* d_generation,
    const bool* active_mask_in,
    bool*        active_mask_out,
    bool*        d_can_push,
    FTStatus*    d_error,
    int          pipeChunks)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int gtid      = threadIdx.x + blockIdx.x * blockDim.x;
    int gnthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();
    const uint64_t generation = *d_generation;
    const int nRanks = devComm.nRanks;
    const int myRank = devComm.rank;

    /* Readiness: PUSH generation, poll LOCAL slots (see FTAllGatherKernel). */
    for (int p = gtid; p < nRanks; p += gnthreads) {
        if (!active_mask_in[p] || p == myRank) continue;
        uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
            readywin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
            .store(generation, cuda::memory_order_release);
    }
    for (int r = gtid; r < nRanks; r += gnthreads) {
        if (active_mask_in[r] && r != myRank) {
            uint64_t* mySlot = (uint64_t*)ncclGetLocalPointer(
                readywin, (size_t)r * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*mySlot);
            while (af.load(cuda::memory_order_acquire) < generation) {
                if (clock64() - startCycle >= timeout_cycles) break;
            }
            if (af.load(cuda::memory_order_acquire) < generation)
                /* expected peer never signaled readiness: flag but keep going
                 * for partial results (matches the GIN barrier). */
                cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                    .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
    grid.sync();
    startCycle = clock64();   /* fresh budget for the flag poll (see FTAllToAllKernel) */

    /* Shard geometry: shard s = [s*shardElems, ...); shardElems rounded up to a
     * 16-byte multiple so every shard base and scratch slot is 128-bit aligned. */
    const int    VEC = (int)(16 / sizeof(T));
    size_t shardElems = (count + (size_t)nRanks - 1) / (size_t)nRanks;
    shardElems = ((shardElems + (size_t)VEC - 1) / (size_t)VEC) * (size_t)VEC;
    const size_t myBase = (size_t)myRank * shardElems;
    const size_t myLen  = (myBase >= count) ? 0
                        : ((count - myBase < shardElems) ? (count - myBase) : shardElems);

    T* sendLocal    = (T*)ncclGetLocalPointer(sendwin, sendoffset);   /* == output */
    T* scratchLocal = (T*)ncclGetLocalPointer(recvwin, recvoffset);   /* contributor slots */

    /* Phase 1: scatter — push my slice of each shard s to owner s's slot[myRank].
     * CTAs map to DISTINCT owners (task % nRanks) so all outbound links run
     * concurrently instead of the whole grid pushing to one owner at a time;
     * nSub CTAs split each shard, 16B-aligned. */
    {
        const int nSub = (gridDim.x + nRanks - 1) / nRanks;   /* CTAs per shard */
        for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
            int s = task % nRanks;
            size_t sbase = (size_t)s * shardElems;
            if (sbase >= count) continue;
            size_t slen = (count - sbase < shardElems) ? (count - sbase) : shardElems;
            size_t per = (slen + (size_t)nSub - 1) / (size_t)nSub;
            per = ((per + (size_t)VEC - 1) / (size_t)VEC) * (size_t)VEC;
            size_t start = (size_t)(task / nRanks) * per;
            if (start >= slen) continue;
            size_t len = (slen - start < per) ? (slen - start) : per;
            T* dst;
            if (s == myRank)
                dst = scratchLocal + (size_t)myRank * shardElems;
            else
                dst = (T*)ncclGetLsaPointer(
                    recvwin, recvoffset + (size_t)myRank * shardElems * sizeof(T), s);
            ftCopy<T>(dst + start, sendLocal + sbase + start, len,
                      threadIdx.x, blockDim.x);
        }
    }
    grid.sync();
    __threadfence_system();
    grid.sync();

    /* Round-1 signal: tell each owner s "myRank delivered" (flag slot myRank). */
    for (int s = gtid; s < nRanks; s += gnthreads) {
        if (s == myRank || !active_mask_in[s]) continue;
        uint64_t* f = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t), s);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*f)
            .store(1, cuda::memory_order_release);
    }

    /* Round-1 poll: I own shard myRank — wait for every contributor's flag,
     * recording who delivered in d_can_push (the confirmed-contributor set the
     * reduction sums over).  Self always counts (local copy). */
    for (int j = gtid; j < nRanks; j += gnthreads) {
        bool got = (j == myRank);
        if (j != myRank && active_mask_in[j]) {
            uint64_t* f = (uint64_t*)ncclGetLocalPointer(
                flagswin, (size_t)j * sizeof(uint64_t));
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*f);
            bool to = false;
            while (af.load(cuda::memory_order_acquire) < 1) {
                if (clock64() - startCycle >= timeout_cycles) { to = true; break; }
            }
            got = !to;
            if (to) cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                        .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        d_can_push[j] = got;
    }
    grid.sync();

    /* ---- Reduce + broadcast ----
     * pipeChunks<=1 : bulk reduce (whole shard) then bulk broadcast (proven).
     * pipeChunks>1  : chunk my shard and overlap reduce(chunk c+1) on a slice of
     *                 CTAs with broadcast(chunk c) on the rest, hiding the reduce
     *                 phase (compute) + its barriers behind the broadcast NVLink
     *                 traffic.  Cross-rank handshakes above/below are unchanged. */
    const bool canVec =
        ((((uintptr_t)(output + myBase)) | ((uintptr_t)scratchLocal)) & 0xF) == 0;

    if (pipeChunks <= 1) {
        /* Bulk reduce, whole shard. */
        if (canVec) {
            T* dst = output + myBase;
            const size_t nvec = myLen / VEC;
            int4* dV = (int4*)dst;
            for (size_t v = gtid; v < nvec; v += gnthreads) {
                float acc[16 / 2];
                #pragma unroll
                for (int k = 0; k < VEC; k++) acc[k] = 0.0f;
                for (int j = 0; j < nRanks; j++) {
                    if (!d_can_push[j]) continue;
                    int4 packed = ((const int4*)(scratchLocal + (size_t)j * shardElems))[v];
                    const T* el = (const T*)&packed;
                    #pragma unroll
                    for (int k = 0; k < VEC; k++) acc[k] += (float)el[k];
                }
                int4 outp; T* oel = (T*)&outp;
                #pragma unroll
                for (int k = 0; k < VEC; k++) oel[k] = (T)acc[k];
                dV[v] = outp;
            }
            for (size_t i = nvec * VEC + gtid; i < myLen; i += gnthreads) {
                float sum = 0.0f;
                for (int j = 0; j < nRanks; j++)
                    if (d_can_push[j]) sum += (float)scratchLocal[(size_t)j * shardElems + i];
                dst[i] = (T)sum;
            }
        } else {
            T* dst = output + myBase;
            for (size_t i = gtid; i < myLen; i += gnthreads) {
                float sum = 0.0f;
                for (int j = 0; j < nRanks; j++)
                    if (d_can_push[j]) sum += (float)scratchLocal[(size_t)j * shardElems + i];
                dst[i] = (T)sum;
            }
        }
        /* reduce -> broadcast is a LOCAL dependency (both touch this GPU's
         * `output`); grid.sync gives device-scope release/acquire, so no
         * threadfence_system drain is needed here.  The cross-rank flush stays
         * after the broadcast (before the round-2 flags). */
        grid.sync();

        /* Bulk broadcast — CTAs map to DISTINCT peers (task % nRanks) so all
         * outbound links run concurrently; nSub CTAs split my shard, 16B-aligned. */
        {
            const int nSub = (gridDim.x + nRanks - 1) / nRanks;
            size_t per = (myLen + (size_t)nSub - 1) / (size_t)nSub;
            per = ((per + (size_t)VEC - 1) / (size_t)VEC) * (size_t)VEC;
            for (int task = blockIdx.x; task < nRanks * nSub; task += gridDim.x) {
                int p = task % nRanks;
                if (p == myRank || !active_mask_in[p]) continue;
                size_t start = (size_t)(task / nRanks) * per;
                if (start >= myLen) continue;
                size_t len = (myLen - start < per) ? (myLen - start) : per;
                T* dst = (T*)ncclGetLsaPointer(
                    sendwin, sendoffset + (myBase + start) * sizeof(T), p);
                ftCopy<T>(dst, output + myBase + start, len,
                          threadIdx.x, blockDim.x);
            }
        }
        grid.sync();
        __threadfence_system();
        grid.sync();
    } else {
        /* Pipelined reduce<->broadcast.  Partition the grid: nRed CTAs reduce
         * (compute/HBM), the rest broadcast (NVLink) — concurrently.  Reduce is
         * ~nRanks-cheaper than broadcast, so a small reducer slice keeps up. */
        const int NC = pipeChunks;
        const size_t chunkElems =
            ((((myLen + (size_t)NC - 1) / (size_t)NC) + (size_t)VEC - 1)
              / (size_t)VEC) * (size_t)VEC;
        const int    nRed  = (gridDim.x >= 8) ? (gridDim.x / 8) : 1;
        const int    nBc   = gridDim.x - nRed;
        const bool   isRed = (blockIdx.x < nRed);
        const int    bbid  = blockIdx.x - nRed;                 /* broadcaster id */
        const size_t rtid  = (size_t)threadIdx.x + (size_t)blockIdx.x * blockDim.x;
        const size_t rn    = (size_t)blockDim.x * (size_t)nRed;

        /* Prologue: reduce chunk 0 with the whole grid (nothing to broadcast yet). */
        {
            size_t cl = (chunkElems < myLen) ? chunkElems : myLen;
            ftRSReduceChunk<T>(output, myBase, scratchLocal, d_can_push, nRanks,
                               shardElems, 0, cl, canVec, VEC, gtid, gnthreads);
        }
        grid.sync();

        for (int c = 0; c < NC; c++) {
            if (isRed) {
                int nx = c + 1;
                size_t ncs = (size_t)nx * chunkElems;
                if (nx < NC && ncs < myLen) {
                    size_t ncl = (myLen - ncs < chunkElems) ? (myLen - ncs) : chunkElems;
                    ftRSReduceChunk<T>(output, myBase, scratchLocal, d_can_push,
                                       nRanks, shardElems, ncs, ncl, canVec, VEC,
                                       rtid, rn);
                }
            } else {
                size_t ccs = (size_t)c * chunkElems;
                if (ccs < myLen) {
                    size_t ccl = (myLen - ccs < chunkElems) ? (myLen - ccs) : chunkElems;
                    ftRSBroadcastChunk<T>(sendwin, sendoffset, output,
                                          active_mask_in, nRanks, myRank, myBase,
                                          ccs, ccl, VEC, bbid, nBc);
                }
            }
            grid.sync();   /* chunk c broadcast done + chunk c+1 reduce visible */
        }
        __threadfence_system();   /* flush all peer pushes before round-2 flags */
        grid.sync();
    }

    /* Round-2 signal: tell each peer "shard myRank delivered" (flag slot myRank). */
    for (int p = gtid; p < nRanks; p += gnthreads) {
        if (p == myRank || !active_mask_in[p]) continue;
        uint64_t* f = (uint64_t*)ncclGetLsaPointer(
            flagswin, (size_t)myRank * sizeof(uint64_t), p);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*f)
            .store(2, cuda::memory_order_release);
    }

    /* Round-2 poll: wait for every shard owner's broadcast; build active_mask_out. */
    for (int r = gtid; r < nRanks; r += gnthreads)
        active_mask_out[r] = (r == myRank);
    grid.sync();
    for (int j = gtid; j < nRanks; j += gnthreads) {
        if (j == myRank || !active_mask_in[j]) continue;
        uint64_t* f = (uint64_t*)ncclGetLocalPointer(
            flagswin, (size_t)j * sizeof(uint64_t));
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> af(*f);
        bool to = false;
        while (af.load(cuda::memory_order_acquire) < 2) {
            if (clock64() - startCycle >= timeout_cycles) { to = true; break; }
        }
        if (!to) active_mask_out[j] = true;
        else cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                 .store(FT_TIMEOUT, cuda::memory_order_release);
    }
    /* No trailing grid.sync: active_mask_out is consumed by the host after the
     * kernel completes, and nothing else in-kernel depends on it.
     * No readywin reset (PUSH/generation model — see FTAllGatherKernel). */
}

// =============================================================================
// GIN AllGather kernel
// =============================================================================

/*
 * FTAllGatherGINKernel (GIN path)
 *
 * Fault-tolerant AllGather using GPU-Initiated Networking (RDMA).
 *
 * Unlike the LSA kernel, this path:
 *   - Uses ncclGinBarrierSession for the readiness barrier (no readywin window).
 *   - Uses gin.put() for remote data delivery (no ncclGetLsaPointer).
 *   - Uses gin.readSignal() for per-rank completion detection (no flagswin).
 *
 * Signal assignment: signal index r tracks whether rank r's data has arrived.
 * Each rank reads signal baselines before any traffic, then after put-ing its
 * chunk to all peers, polls for baseline+1 from each expected peer.
 *
 * The ncclDevComm must have been created with:
 *   reqs.railGinBarrierCount = FT_DEVICE_CTA_COUNT
 *   reqs.ginSignalCount      = nRanks
 *   reqs.ginConnectionType   = NCCL_GIN_CONNECTION_FULL
 */
template <typename T>
__global__ void FTAllGatherGINKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    if (tid < devComm.nRanks)
        s_active_mask[tid] = active_mask_in[tid];
    __syncthreads();

    int ginContext = 0;
    ncclGin gin { devComm, ginContext };

    /* Step 1: Snapshot signal baselines before any traffic (thread 0). */
    uint64_t signalBase[FT_MAX_RANKS];
    if (tid == 0) {
        for (int r = 0; r < devComm.nRanks; r++)
            signalBase[r] = gin.readSignal((ncclGinSignal_t)r);
    }

    /* Step 2: GIN barrier — all active peers signal readiness over the network.
     * Uses remaining timeout budget so a dead peer does not cause a hang. */
    ncclGinBarrierSession<ncclCoopCta> bar {
        ncclCoopCta(), gin, ncclTeamWorld(devComm),
        devComm.railGinBarrier, blockIdx.x
    };

    uint64_t elapsed = clock64() - startCycle;
    uint64_t barBudget = (elapsed >= timeout_cycles) ? 0
                         : (timeout_cycles - elapsed);
    ncclResult_t barResult = bar.sync(
        ncclCoopCta(), cuda::memory_order_acquire,
        ncclGinFenceLevel::Relaxed, barBudget);

    if (barResult != ncclSuccess) {
        if (tid == 0) {
            for (int r = 0; r < devComm.nRanks; r++)
                active_mask_out[r] = (r == devComm.rank);
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        return;
    }

    /* Step 3: Self-copy — local send → local recv slot. */
    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    for (int i = tid; i < (int)count; i += nthreads)
        localRecvPtr[i] = localSendPtr[i];

    /* Step 4: Issue async GIN puts to all active peers.
     * Threads stripe over destination ranks; each put carries a SignalInc
     * so the receiving rank can detect arrival without polling flagswin. */
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_active_mask[r] || r == devComm.rank) continue;
        gin.put(ncclTeamWorld(devComm), r,
                recvwin,  recvoffset + (size_t)devComm.rank * chunk_stride,
                sendwin,  sendoffset,
                chunk_bytes,
                ncclGin_SignalInc{(ncclGinSignal_t)devComm.rank});
    }
    gin.flush(ncclCoopCta());

    /* Step 5: Poll signals for inbound data; update masks and d_error. */
    if (tid == 0) {
        for (int r = 0; r < devComm.nRanks; r++)
            active_mask_out[r] = false;
        active_mask_out[devComm.rank] = true;

        bool any_timeout = false;
        for (int r = 0; r < devComm.nRanks; r++) {
            if (!s_active_mask[r] || r == devComm.rank) continue;

            /* Each peer is polled against the shared deadline; once it has
             * passed, a peer whose signal already arrived still counts. */
            uint64_t expected = signalBase[r] + 1;
            bool timed_out = false;
            while (gin.readSignal((ncclGinSignal_t)r) < expected) {
                if (clock64() - startCycle >= timeout_cycles) {
                    timed_out = true;
                    break;
                }
            }
            if (!timed_out) {
                active_mask_out[r] = true;
            } else {
                any_timeout = true;
            }
        }

        if (any_timeout) {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }
}

// =============================================================================
// GIN AllReduce kernel
// =============================================================================

/*
 * FTAllReduceGINKernel (GIN path)
 *
 * GIN AllGather + cooperative reduction.  Identical gather logic to
 * FTAllGatherGINKernel; adds per-rank reduction after all data arrives.
 */
template <typename T>
__global__ void FTAllReduceGINKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    T*           output,
    size_t count,
    struct ncclDevComm devComm,
    uint64_t timeout_cycles,
    const bool* active_mask_in,
    bool*        active_mask_out,
    FTStatus*    d_error)
{
    int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    uint64_t startCycle = clock64();

    __shared__ bool s_active_mask[FT_MAX_RANKS];
    __shared__ bool s_result_mask[FT_MAX_RANKS];
    if (tid < devComm.nRanks) {
        s_active_mask[tid] = active_mask_in[tid];
        s_result_mask[tid] = false;
    }
    __syncthreads();

    int ginContext = 0;
    ncclGin gin { devComm, ginContext };

    uint64_t signalBase[FT_MAX_RANKS];
    if (tid == 0) {
        for (int r = 0; r < devComm.nRanks; r++)
            signalBase[r] = gin.readSignal((ncclGinSignal_t)r);
    }

    ncclGinBarrierSession<ncclCoopCta> bar {
        ncclCoopCta(), gin, ncclTeamWorld(devComm),
        devComm.railGinBarrier, blockIdx.x
    };

    uint64_t elapsed = clock64() - startCycle;
    uint64_t barBudget = (elapsed >= timeout_cycles) ? 0
                         : (timeout_cycles - elapsed);
    ncclResult_t barResult = bar.sync(
        ncclCoopCta(), cuda::memory_order_acquire,
        ncclGinFenceLevel::Relaxed, barBudget);

    if (barResult != ncclSuccess) {
        if (tid == 0) {
            for (int r = 0; r < devComm.nRanks; r++)
                active_mask_out[r] = (r == devComm.rank);
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
        /* Zero output so the caller gets a defined value even on barrier fail. */
        for (int i = tid; i < (int)count; i += nthreads)
            output[i] = (T)0;
        return;
    }

    const size_t chunk_bytes  = count * sizeof(T);
    const size_t chunk_stride = FT_RECV_STRIDE(chunk_bytes);
    T* localSendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* localRecvPtr = (T*)ncclGetLocalPointer(
        recvwin, recvoffset + (size_t)devComm.rank * chunk_stride);

    for (int i = tid; i < (int)count; i += nthreads)
        localRecvPtr[i] = localSendPtr[i];

    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        if (!s_active_mask[r] || r == devComm.rank) continue;
        gin.put(ncclTeamWorld(devComm), r,
                recvwin, recvoffset + (size_t)devComm.rank * chunk_stride,
                sendwin, sendoffset,
                chunk_bytes,
                ncclGin_SignalInc{(ncclGinSignal_t)devComm.rank});
    }
    gin.flush(ncclCoopCta());

    if (tid == 0) {
        for (int r = 0; r < devComm.nRanks; r++)
            active_mask_out[r] = false;
        active_mask_out[devComm.rank] = true;
        s_result_mask[devComm.rank]   = true;

        bool any_timeout = false;
        for (int r = 0; r < devComm.nRanks; r++) {
            if (!s_active_mask[r] || r == devComm.rank) continue;

            /* Each peer is polled against the shared deadline; once it has
             * passed, a peer whose signal already arrived still counts. */
            uint64_t expected = signalBase[r] + 1;
            bool timed_out = false;
            while (gin.readSignal((ncclGinSignal_t)r) < expected) {
                if (clock64() - startCycle >= timeout_cycles) {
                    timed_out = true;
                    break;
                }
            }
            if (!timed_out) {
                active_mask_out[r] = true;
                s_result_mask[r]   = true;
            } else {
                any_timeout = true;
            }
        }

        if (any_timeout) {
            cuda::atomic_ref<FTStatus, cuda::thread_scope_system>(*d_error)
                .store(FT_TIMEOUT, cuda::memory_order_release);
        }
    }

    __syncthreads();

    T* localRecvBase = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    const size_t stride_elems = chunk_stride / sizeof(T);  /* padded slice stride */
    for (int i = tid; i < (int)count; i += nthreads) {
        float sum = 0.0f;
        for (int r = 0; r < devComm.nRanks; r++)
            if (s_result_mask[r]) sum += (float)localRecvBase[(size_t)r * stride_elems + i];
        output[i] = (T)sum;
    }
}

// =============================================================================
// FTCollectiveHandle
// =============================================================================

ncclResult_t ftHandleCreate(ncclComm_t comm, int nRanks,
                             const bool* initial_mask,
                             FTCollectiveHandle** out) {
    ncclResult_t ret = ncclSuccess;

    /* Kernels keep per-rank state in single-CTA shared arrays initialized one
     * rank per thread, so nRanks is bounded by the CTA size. */
    if (!comm || !initial_mask || !out) return ncclInvalidArgument;
    if (nRanks < 1 || nRanks > FT_DEVICE_THREADS_PER_CTA) return ncclInvalidArgument;
    {
        int commRanks = 0;
        if (ncclCommCount(comm, &commRanks) != ncclSuccess || commRanks != nRanks)
            return ncclInvalidArgument;
    }

    FTCollectiveHandle* h = (FTCollectiveHandle*)calloc(1, sizeof(*h));
    if (!h) return ncclSystemError;

    h->comm   = comm;
    h->nRanks = nRanks;
    h->self_adapt = true;   /* default on; see ftHandleSetSelfAdapt */

    /* Query communicator capabilities to select kernel backend. */
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ret = ncclCommQueryProperties(comm, &props);
    if (ret != ncclSuccess) goto fail;

    if (!props.deviceApiSupport) {
        ret = ncclInvalidUsage;
        goto fail;
    }

    /* Backend selection: prefer LSA (NVLink) when a team exists, else GIN (IB).
     * GIN is only selected when there is no LSA domain, because GIN operates on
     * the rail (inter-node) dimension — see the degenerate-rail guard in the GIN
     * setup branch below, which rejects GIN when the rail team collapses to
     * {self} (single node, lsaSize == nRanks). */
    if (props.nLsaTeams >= 1) {
        h->kernelType = FT_KERNEL_LSA;
    } else if (props.ginType != NCCL_GIN_TYPE_NONE) {
        h->kernelType = FT_KERNEL_GIN;
    } else {
        /* Neither LSA nor GIN available. */
        ret = ncclInvalidUsage;
        goto fail;
    }

    /* Cache GPU clock rate so launch functions can convert timeout_us →
     * timeout_cycles without calling cudaGetDevice/cudaDeviceGetAttribute
     * on every launch (those sync calls add GPU idle time inside the
     * measured event window).                                               */
    {
        int device = 0;
        int clock_khz = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device) != cudaSuccess ||
            clock_khz < 1000) {
            ret = ncclSystemError; goto fail;   /* a zero rate would zero every timeout */
        }
        h->cycles_per_us = (uint64_t)(clock_khz / 1000);
    }

    /* Input mask. */
    h->h_mask_in = (bool*)malloc((size_t)nRanks * sizeof(bool));
    if (!h->h_mask_in) { ret = ncclSystemError; goto fail; }
    memcpy(h->h_mask_in, initial_mask, (size_t)nRanks * sizeof(bool));
    h->mask_dirty = true;   /* will be uploaded on first launch */

    if (cudaMalloc(&h->d_mask_in, (size_t)nRanks * sizeof(bool)) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }
    if (cudaMemcpy(h->d_mask_in, h->h_mask_in, (size_t)nRanks * sizeof(bool),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }

    /* Output mask. */
    h->h_mask_out = (bool*)calloc((size_t)nRanks, sizeof(bool));
    if (!h->h_mask_out) { ret = ncclSystemError; goto fail; }
    if (cudaMalloc(&h->d_mask_out, (size_t)nRanks * sizeof(bool)) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }

    /* Error status: mapped pinned for zero-copy host reads. */
    if (cudaHostAlloc(&h->h_error, sizeof(FTStatus),
                      cudaHostAllocMapped) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }
    if (cudaHostGetDevicePointer(&h->d_error, h->h_error, 0) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }
    *h->h_error = FT_OK;

    /* Device-side generation counter, bumped on-device by every launch on
     * both backends (GIN kernels ignore it, but the launch path still bumps). */
    if (cudaMalloc(&h->d_gen, sizeof(uint64_t)) != cudaSuccess ||
        cudaMemset(h->d_gen, 0, sizeof(uint64_t)) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }

    if (h->kernelType == FT_KERNEL_LSA) {
        /* LSA path: allocate readywin (one uint64_t per rank's symmetric copy).
         * devComm is NOT created here — it is deferred to ftAllGatherCreate /
         * ftAllReduceCreate so that all symmetric windows (d_send, d_recv,
         * d_flags) are registered before ncclDevCommCreate is called.  This
         * matches the window-first ordering in collectives_bench and ensures
         * the devComm sees the complete symmetric-memory layout. */
        /* readywin: nRanks generation slots (PUSH model — see header).  Zeroed
         * once at creation; never reset thereafter (generations only increase). */
        {
            const size_t ready_bytes = (size_t)nRanks * sizeof(uint64_t);
            ret = ncclMemAlloc(&h->d_ready, ready_bytes);
            if (ret != ncclSuccess) goto fail;
            if (cudaMemset(h->d_ready, 0, ready_bytes) != cudaSuccess) {
                ret = ncclSystemError; goto fail;
            }
            ret = ncclCommWindowRegister(comm, h->d_ready, ready_bytes,
                                         &h->win_ready, NCCL_WIN_COLL_SYMMETRIC);
            if (ret != ncclSuccess) goto fail;

            /* Dedicated rejoin window: 2*nRanks uint64 slots (region A = generation
             * max-consensus, region B = presence).  Pre-registered here while all
             * ranks are present — registration is collective, so it must never be
             * done lazily on a rejoin (a returnee could not join it). */
            const size_t rejoin_bytes = (size_t)2 * nRanks * sizeof(uint64_t);
            ret = ncclMemAlloc(&h->d_rejoin, rejoin_bytes);
            if (ret != ncclSuccess) goto fail;
            if (cudaMemset(h->d_rejoin, 0, rejoin_bytes) != cudaSuccess) {
                ret = ncclSystemError; goto fail;
            }
            ret = ncclCommWindowRegister(comm, h->d_rejoin, rejoin_bytes,
                                         &h->win_rejoin, NCCL_WIN_COLL_SYMMETRIC);
            if (ret != ncclSuccess) goto fail;
        }

        /* Multi-CTA cooperative path (LSA only): grid-global can_push scratch,
         * the per-rank size threshold, and the max cooperative grid (occupancy
         * x #SM).  max_coop_blocks stays 0 if cooperative launch is
         * unsupported, which keeps every launch on the single-CTA kernel. */
        h->multicta_threshold_bytes = (size_t)256 * 1024;   /* 256 KiB / rank */
        {
            const char* env = getenv("FT_MULTICTA_THRESHOLD_BYTES");
            if (env) {
                long long v = atoll(env);
                if (v >= 0) h->multicta_threshold_bytes = (size_t)v;
            }
        }
        /* Reduce-scatter AllReduce threshold (default 2 MiB / rank — the
         * measured crossover where reduce-scatter overtakes the multi-CTA
         * gather-then-reduce path; below it RS's extra phases don't pay off). */
        h->rs_threshold_bytes = (size_t)2 * 1024 * 1024;
        {
            const char* env = getenv("FT_REDUCE_SCATTER_THRESHOLD_BYTES");
            if (env) {
                long long v = atoll(env);
                if (v >= 0) h->rs_threshold_bytes = (size_t)v;
            }
        }
        /* Reduce<->broadcast pipelining for the RS AllReduce (0/1 = off). */
        h->rs_pipeline_chunks = 0;
        {
            const char* env = getenv("FT_RS_PIPELINE_CHUNKS");
            if (env) {
                int v = atoi(env);
                if (v >= 0) h->rs_pipeline_chunks = v;
            }
        }
        if (cudaMalloc(&h->d_can_push, (size_t)nRanks * sizeof(bool)) != cudaSuccess) {
            ret = ncclSystemError; goto fail;
        }
        if (cudaMalloc(&h->d_a2av_ctr, sizeof(int)) != cudaSuccess) {
            ret = ncclSystemError; goto fail;
        }
        if (cudaMalloc(&h->d_a2av_displs, (size_t)nRanks * sizeof(size_t)) != cudaSuccess) {
            ret = ncclSystemError; goto fail;
        }
        {
            int device = 0, coop = 0, numSM = 0, bpAG = 0, bpAR = 0, bpRS = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device);
            if (coop) {
                cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);
                /* Cooperative launch requires the whole grid co-resident, so the
                 * cap must satisfy EVERY cooperative kernel.  The reduce-scatter
                 * kernel uses more registers (lower occupancy) than the gather
                 * kernels — omitting it here let large RS launches exceed its
                 * limit and fail with cudaErrorCooperativeLaunchTooLarge. */
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpAG, FTAllGatherKernelMultiCTA<float>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                /* AllToAll mirrors AllGather's occupancy but include it so the
                 * cooperative cap is valid if register allocation ever differs. */
                int bpAA = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpAA, FTAllToAllKernelMultiCTA<float>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                int bpAAv = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpAAv, FTAllToAllvKernelMultiCTA<float>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                int bpAGv = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpAGv, FTAllGathervKernel,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpAR, FTAllReduceKernelMultiCTA<float>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpRS, FTAllReduceRSKernelMultiCTA<float>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                /* fp16/bf16 RS use acc[8] (more registers) -> lower occupancy;
                 * clamp by the half variant too so those launches also fit. */
                int bpRSh = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpRSh, FTAllReduceRSKernelMultiCTA<__half>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                int bpRSv = 0, bpRSvh = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpRSv, FTReduceScattervKernel<float>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &bpRSvh, FTReduceScattervKernel<__half>,
                    FT_DEVICE_THREADS_PER_CTA, 0);
                int bp = bpAG;
                if (bpAA  < bp) bp = bpAA;
                if (bpAAv < bp) bp = bpAAv;
                if (bpAGv < bp) bp = bpAGv;
                if (bpAR  < bp) bp = bpAR;
                if (bpRS  < bp) bp = bpRS;
                if (bpRSh < bp) bp = bpRSh;
                if (bpRSv  < bp) bp = bpRSv;
                if (bpRSvh < bp) bp = bpRSvh;
                if (bp < 1) bp = 1;
                h->max_coop_blocks = bp * numSM;
            }
        }

    } else {
        /* GIN path: readywin not used (ncclGinBarrierSession handles liveness).
         * devComm needs GIN barrier slots and per-rank signals. */
        h->d_ready   = NULL;
        h->win_ready = NULL;

        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.railGinBarrierCount = FT_DEVICE_CTA_COUNT;  /* one slot per CTA */
        reqs.ginSignalCount      = nRanks;                /* signal r = rank r data */
        reqs.ginConnectionType   = NCCL_GIN_CONNECTION_FULL;
        ret = ncclDevCommCreate(comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) goto fail;

        /* GIN operates on the rail (inter-node) dimension: ncclTeamRail has
         * nRanks = comm.nRanks / comm.lsaSize.  On a single node all ranks share
         * one LSA (NVLink) domain, so lsaSize == nRanks and the rail team
         * collapses to {self} — there are no GIN peers to reach and a GIN
         * collective cannot run.  The GIN kernels' gin.put/bar.sync would then
         * target unmapped remote windows and fault with cudaErrorIllegalAddress
         * (700) mid-launch.  Reject up front with a clear, actionable error. */
        {
            int lsaSz     = (h->devComm.lsaSize > 0) ? h->devComm.lsaSize : 1;
            int railRanks = h->devComm.nRanks / lsaSz;
            if (railRanks < 2) {
                fprintf(stderr,
                    "ft_collective: GIN backend requires a non-degenerate rail "
                    "(inter-node) dimension, but rail team has %d rank(s) "
                    "(nRanks=%d, lsaSize=%d). GIN cannot run on a single node — "
                    "run across >1 node so lsaSize < nRanks, or use LSA.\n",
                    railRanks, h->devComm.nRanks, h->devComm.lsaSize);
                ret = ncclInvalidUsage;
                goto fail;
            }
        }
    }

    *out = h;
    return ncclSuccess;

fail:
    ftHandleDestroy(h);
    return ret;
}

void ftHandleDestroy(FTCollectiveHandle* h) {
    if (!h) return;
    /* devComm.nRanks == 0 means ncclDevCommCreate was never called (LSA deferred
     * path) or already destroyed — skip the destroy to avoid double-free. */
    if (h->comm && h->devComm.nRanks != 0) ncclDevCommDestroy(h->comm, &h->devComm);
    if (h->win_ready) ncclCommWindowDeregister(h->comm, h->win_ready);
    if (h->d_ready)   ncclMemFree(h->d_ready);
    if (h->win_rejoin) ncclCommWindowDeregister(h->comm, h->win_rejoin);
    if (h->d_rejoin)   ncclMemFree(h->d_rejoin);
    if (h->d_can_push) cudaFree(h->d_can_push);
    if (h->d_a2av_ctr) cudaFree(h->d_a2av_ctr);
    if (h->d_gen) cudaFree(h->d_gen);
    if (h->d_a2av_displs) cudaFree(h->d_a2av_displs);
    if (h->h_error)   cudaFreeHost(h->h_error);
    if (h->d_mask_out) cudaFree(h->d_mask_out);
    if (h->h_mask_out) free(h->h_mask_out);
    if (h->d_mask_in)  cudaFree(h->d_mask_in);
    if (h->h_mask_in)  free(h->h_mask_in);
    free(h);
}

void ftHandleSetActiveMask(FTCollectiveHandle* h, const bool* mask) {
    memcpy(h->h_mask_in, mask, (size_t)h->nRanks * sizeof(bool));
    h->mask_dirty = true;
}

/* Deprecated alias for ftHandleSetActiveMask (kept for backward compatibility). */
void ftHandleSetMask(FTCollectiveHandle* h, const bool* mask) {
    ftHandleSetActiveMask(h, mask);
}

void ftHandleSetSelfAdapt(FTCollectiveHandle* h, bool enable) {
    h->self_adapt = enable;
}

const bool* ftHandleGetResultMask(const FTCollectiveHandle* h) {
    cudaMemcpy(h->h_mask_out, h->d_mask_out,
               (size_t)h->nRanks * sizeof(bool), cudaMemcpyDeviceToHost);
    return h->h_mask_out;
}

FTStatus ftHandleGetError(const FTCollectiveHandle* h) {
    return *h->h_error;
}

void ftHandleClearError(FTCollectiveHandle* h) {
    *h->h_error = FT_OK;
}

/* Read the current on-device generation counter (D2H copy).  Diagnostic /
 * membership use: after ftRejoin all survivors should report the same value. */
uint64_t ftHandleGetGeneration(const FTCollectiveHandle* h) {
    uint64_t gen = 0;
    cudaMemcpy(&gen, h->d_gen, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return gen;
}

// =============================================================================
// FTAllGatherCtx
// =============================================================================

ncclResult_t ftAllGatherCreate(FTCollectiveHandle* handle, size_t count,
                                FTAllGatherCtx** out) {
    ncclResult_t ret = ncclSuccess;
    FTAllGatherCtx* ctx = (FTAllGatherCtx*)calloc(1, sizeof(*ctx));
    if (!ctx) return ncclSystemError;

    ctx->handle = handle;
    ctx->count  = count;

    const size_t send_bytes  = count * sizeof(float);
    /* Pad each rank's slice to a 16-byte stride so the kernel's vectorized
     * copies engage on any count (must match FT_RECV_STRIDE in the kernel). */
    const size_t recv_bytes  = FT_RECV_STRIDE(count * sizeof(float))
                               * (size_t)handle->nRanks;

    ret = ncclMemAlloc(&ctx->d_send, send_bytes); if (ret != ncclSuccess) goto fail;
    ret = ncclMemAlloc(&ctx->d_recv, recv_bytes); if (ret != ncclSuccess) goto fail;

    ret = ncclCommWindowRegister(handle->comm, ctx->d_send, send_bytes,
                                 &ctx->win_send, NCCL_WIN_COLL_SYMMETRIC);
    if (ret != ncclSuccess) goto fail;
    ret = ncclCommWindowRegister(handle->comm, ctx->d_recv, recv_bytes,
                                 &ctx->win_recv, NCCL_WIN_COLL_SYMMETRIC);
    if (ret != ncclSuccess) goto fail;

    if (handle->kernelType == FT_KERNEL_LSA) {
        /* Flags window: one uint64_t per rank for per-rank completion signaling. */
        const size_t flags_bytes = (size_t)handle->nRanks * sizeof(uint64_t);
        ret = ncclMemAlloc(&ctx->d_flags, flags_bytes);
        if (ret != ncclSuccess) goto fail;
        ret = ncclCommWindowRegister(handle->comm, ctx->d_flags, flags_bytes,
                                     &ctx->win_flags, NCCL_WIN_COLL_SYMMETRIC);
        if (ret != ncclSuccess) goto fail;

        /* All symmetric windows (d_ready from handle, d_send/d_recv/d_flags
         * from this ctx) are now registered.  Create devComm here so NCCL
         * sees the complete window set — matches collectives_bench ordering. */
        if (handle->devComm.nRanks != 0) {
            ncclDevCommDestroy(handle->comm, &handle->devComm);
            memset(&handle->devComm, 0, sizeof(handle->devComm));
        }
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ret = ncclDevCommCreate(handle->comm, &reqs, &handle->devComm);
        if (ret != ncclSuccess) goto fail;

        /* Drain all CUDA streams in this context — including any NCCL-internal
         * streams left active by ncclDevCommCreate / ncclCommWindowRegister.
         * Without this, the first NULL-stream or cross-stream sync in
         * ftAllGatherLaunch would stall waiting for NCCL's background work,
         * adding ~750 µs of GPU idle time inside the measured event window.  */
        if (cudaDeviceSynchronize() != cudaSuccess) { ret = ncclSystemError; goto fail; }
    }
    /* For GIN: d_flags / win_flags remain NULL (calloc zeroed them).
     * GIN devComm was already created in ftHandleCreate with the right reqs. */

    *out = ctx;
    return ncclSuccess;

fail:
    ftAllGatherDestroy(ctx);
    return ret;
}

void ftAllGatherDestroy(FTAllGatherCtx* ctx) {
    if (!ctx) return;
    if (ctx->win_flags) ncclCommWindowDeregister(ctx->handle->comm, ctx->win_flags);
    if (ctx->win_recv)  ncclCommWindowDeregister(ctx->handle->comm, ctx->win_recv);
    if (ctx->win_send)  ncclCommWindowDeregister(ctx->handle->comm, ctx->win_send);
    if (ctx->d_flags) ncclMemFree(ctx->d_flags);
    if (ctx->d_recv)  ncclMemFree(ctx->d_recv);
    if (ctx->d_send)  ncclMemFree(ctx->d_send);
    free(ctx);
}

ncclResult_t ftAllGatherSetData(FTAllGatherCtx* ctx, const float* data) {
    if (cudaMemcpy(ctx->d_send, data, ctx->count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return ncclSystemError;
    return ncclSuccess;
}

// =============================================================================
// Multi-CTA launch helpers (LSA, cooperative groups)
// =============================================================================

/* True when this transfer should take the cooperative multi-CTA path. */
static inline bool ftUseMultiCTA(const FTCollectiveHandle* h,
                                 size_t count, size_t elem_size) {
    return h->kernelType == FT_KERNEL_LSA
        && h->max_coop_blocks > 1
        && (count * elem_size) >= h->multicta_threshold_bytes;
}

/* True when every rank is currently active (no failures recorded). */
static inline bool ftAllActive(const FTCollectiveHandle* h) {
    for (int r = 0; r < h->nRanks; r++)
        if (!h->h_mask_in[r]) return false;
    return true;
}

/* True when AllReduce should take the reduce-scatter path: LSA window API,
 * cooperative launch available, at/above the RS threshold, >= 2 ranks, and the
 * FULL set active (a masked rank falls back to gather-then-reduce). */
static inline bool ftUseReduceScatter(const FTCollectiveHandle* h,
                                      size_t count, size_t elem_size) {
    return h->kernelType == FT_KERNEL_LSA
        && h->max_coop_blocks > 1
        && h->nRanks >= 2
        && h->rs_threshold_bytes > 0
        && (count * elem_size) >= h->rs_threshold_bytes
        && ftAllActive(h);
}

/* Cooperative grid size: one CTA covers FT_DEVICE_THREADS_PER_CTA 16-byte
 * words; cap to the co-residency limit (max_coop_blocks) and floor at 1. */
static inline int ftCoopGrid(const FTCollectiveHandle* h,
                             size_t count, size_t elem_size) {
    size_t words = (count * elem_size + 15) / 16;
    size_t need  = (words + FT_DEVICE_THREADS_PER_CTA - 1) / FT_DEVICE_THREADS_PER_CTA;
    if (need < 1) need = 1;
    if (need > (size_t)h->max_coop_blocks) need = (size_t)h->max_coop_blocks;
    return (int)need;
}

__global__ void ftBumpGen(uint64_t* g);   /* fwd decl (defined below) */

template <typename T>
static cudaError_t ftLaunchAllGatherMultiCTA(
    FTCollectiveHandle* h, ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset, ncclWindow_t win_flags,
    size_t count, uint64_t timeout_cycles, const uint64_t* d_generation,
    cudaStream_t stream) {
    int blocks = ftCoopGrid(h, count, sizeof(T));
    void* args[] = {
        &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
        &h->win_ready, &count, &h->devComm, &timeout_cycles, &d_generation,
        &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error
    };
    return cudaLaunchCooperativeKernel(
        (void*)FTAllGatherKernelMultiCTA<T>,
        dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream);
}

template <typename T>
static cudaError_t ftLaunchAllToAllMultiCTA(
    FTCollectiveHandle* h, ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset, ncclWindow_t win_flags,
    size_t count, uint64_t timeout_cycles, const uint64_t* d_generation,
    cudaStream_t stream) {
    int blocks = ftCoopGrid(h, count, sizeof(T));
    void* args[] = {
        &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
        &h->win_ready, &count, &h->devComm, &timeout_cycles, &d_generation,
        &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error
    };
    return cudaLaunchCooperativeKernel(
        (void*)FTAllToAllKernelMultiCTA<T>,
        dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream);
}

template <typename T>
static cudaError_t ftLaunchAllToAllvMultiCTA(
    FTCollectiveHandle* h, ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset, ncclWindow_t win_flags,
    const int* d_send_counts, const size_t* d_send_displs, size_t recv_stride,
    T* output, const int* d_recv_counts, const size_t* d_recv_displs,
    int* d_recv_counts_out, size_t recv_capacity, size_t row_size,
    size_t coop_hint, uint64_t timeout_cycles, const uint64_t* d_generation,
    cudaStream_t stream) {
    int blocks = ftCoopGrid(h, coop_hint, sizeof(T));
    void* args[] = {
        &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
        &h->win_ready, &d_send_counts, &d_send_displs, &recv_stride,
        &output, &d_recv_counts, &d_recv_displs,
        &d_recv_counts_out, &recv_capacity, &row_size, &h->d_a2av_ctr,
        &h->devComm, &timeout_cycles, &d_generation,
        &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error
    };
    return cudaLaunchCooperativeKernel(
        (void*)FTAllToAllvKernelMultiCTA<T>,
        dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream);
}

template <typename T>
static cudaError_t ftLaunchAllReduceMultiCTA(
    FTCollectiveHandle* h, ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset, ncclWindow_t win_flags,
    T* output, size_t count, uint64_t timeout_cycles, const uint64_t* d_generation,
    cudaStream_t stream) {
    int blocks = ftCoopGrid(h, count, sizeof(T));
    void* args[] = {
        &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
        &h->win_ready, &output, &count, &h->devComm, &timeout_cycles, &d_generation,
        &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error
    };
    return cudaLaunchCooperativeKernel(
        (void*)FTAllReduceKernelMultiCTA<T>,
        dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream);
}

template <typename T>
static cudaError_t ftLaunchAllReduceRSMultiCTA(
    FTCollectiveHandle* h, ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset, ncclWindow_t win_flags,
    T* output, size_t count, uint64_t timeout_cycles, const uint64_t* d_generation,
    cudaStream_t stream) {
    int blocks = ftCoopGrid(h, count, sizeof(T));
    void* args[] = {
        &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
        &h->win_ready, &output, &count, &h->devComm, &timeout_cycles, &d_generation,
        &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error,
        &h->rs_pipeline_chunks
    };
    return cudaLaunchCooperativeKernel(
        (void*)FTAllReduceRSKernelMultiCTA<T>,
        dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream);
}

template <typename T>
static cudaError_t ftLaunchReduceScattervMultiCTA(
    FTCollectiveHandle* h, ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset, ncclWindow_t win_flags,
    const int* d_counts, const size_t* d_displs, T* output,
    size_t recv_stride, size_t max_shard_count,
    uint64_t timeout_cycles, const uint64_t* d_generation, cudaStream_t stream) {
    int blocks = ftCoopGrid(h, max_shard_count, sizeof(T));
    if (blocks < 1) blocks = 1;
    void* args[] = {
        &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
        &h->win_ready, &d_counts, &d_displs, &output, &recv_stride,
        &h->devComm, &timeout_cycles, &d_generation,
        &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error
    };
    return cudaLaunchCooperativeKernel(
        (void*)FTReduceScattervKernel<T>,
        dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream);
}

ncclResult_t ftAllGatherLaunch(FTAllGatherCtx* ctx,
                                cudaStream_t stream,
                                uint64_t timeout_us) {
    FTCollectiveHandle* h = ctx->handle;
    const size_t mask_bytes  = (size_t)h->nRanks * sizeof(bool);

    /* Upload input mask only when it has changed (set by ftHandleSetActiveMask or
     * at creation).  Skipping the H2D copy on clean iterations avoids a
     * non-pinned cudaMemcpyAsync stall inside the critical GPU timing path. */
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* Reset per-launch state — all async on the benchmark stream.
     * d_mask_out is now async (not NULL-stream) to avoid stalling on any
     * NCCL-internal streams left active in the CUDA context.
     *
     * NOTE: d_recv is intentionally NOT zeroed.  Every present rank writes its
     * own slice (self-copy) and is push-written by every other present rank, so
     * in the all-present case the full recv buffer is overwritten and the memset
     * was pure overhead — and it scaled with nRanks*count (e.g. ~128 MB at the
     * top benchmark size), sitting on the critical timing path.  After a fault,
     * an absent rank's slice is left undefined; callers must consult
     * active_mask_out to know which slices are valid (the FT contract). */
    if (cudaMemsetAsync(h->d_mask_out,  0, mask_bytes, stream) != cudaSuccess) return ncclSystemError;

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);

    if (h->kernelType == FT_KERNEL_LSA) {
        const size_t flags_bytes = (size_t)h->nRanks * sizeof(uint64_t);
        if (cudaMemsetAsync(ctx->d_flags, 0, flags_bytes, stream) != cudaSuccess) return ncclSystemError;

        if (ftUseMultiCTA(h, ctx->count, sizeof(float))) {
            if (ftLaunchAllGatherMultiCTA<float>(
                    h, ctx->win_send, 0, ctx->win_recv, 0, ctx->win_flags,
                    ctx->count, timeout_cycles, h->d_gen, stream) != cudaSuccess)
                return ncclSystemError;
        } else {
        FTAllGatherKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                ctx->win_send, 0,
                ctx->win_recv, 0,
                ctx->win_flags,
                h->win_ready,
                ctx->count,
                h->devComm,
                timeout_cycles,
                h->d_gen,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
        }
    } else {
        FTAllGatherGINKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                ctx->win_send, 0,
                ctx->win_recv, 0,
                ctx->count,
                h->devComm,
                timeout_cycles,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
    }

    /* Self-adaptation: adopt observed survivors as the next collective's input
     * mask so a timed-out peer is excluded automatically (see self_adapt). */
    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out,
                            (size_t)h->nRanks * sizeof(bool),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* ASYNCHRONOUS: caller must cudaStreamSynchronize(stream) before reading
     * results, h_error, or calling ftHandleGetResultMask(). */
    return ncclSuccess;
}

ncclResult_t ftAllGatherGetResult(const FTAllGatherCtx* ctx, float* out) {
    /* d_recv holds one 16-byte-padded slot per rank; unpack to a dense output. */
    const size_t chunk_bytes = ctx->count * sizeof(float);
    if (cudaMemcpy2D(out, chunk_bytes, ctx->d_recv, FT_RECV_STRIDE(chunk_bytes),
                     chunk_bytes, (size_t)ctx->handle->nRanks,
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        return ncclSystemError;
    return ncclSuccess;
}

// =============================================================================
// FTAllReduceCtx
// =============================================================================

ncclResult_t ftAllReduceCreate(FTCollectiveHandle* handle, size_t count,
                                FTAllReduceCtx** out) {
    ncclResult_t ret = ncclSuccess;
    FTAllReduceCtx* ctx = (FTAllReduceCtx*)calloc(1, sizeof(*ctx));
    if (!ctx) return ncclSystemError;

    ctx->handle = handle;
    ctx->count  = count;

    const size_t send_bytes  = count * sizeof(float);
    /* Pad each rank's slice to a 16-byte stride so the kernel's vectorized
     * copies engage on any count (must match FT_RECV_STRIDE in the kernel). */
    const size_t recv_bytes  = FT_RECV_STRIDE(count * sizeof(float))
                               * (size_t)handle->nRanks;

    ret = ncclMemAlloc(&ctx->d_send, send_bytes); if (ret != ncclSuccess) goto fail;
    ret = ncclMemAlloc(&ctx->d_recv, recv_bytes); if (ret != ncclSuccess) goto fail;

    if (cudaMalloc(&ctx->d_output, send_bytes) != cudaSuccess) {
        ret = ncclSystemError; goto fail;
    }

    ret = ncclCommWindowRegister(handle->comm, ctx->d_send, send_bytes,
                                 &ctx->win_send, NCCL_WIN_COLL_SYMMETRIC);
    if (ret != ncclSuccess) goto fail;
    ret = ncclCommWindowRegister(handle->comm, ctx->d_recv, recv_bytes,
                                 &ctx->win_recv, NCCL_WIN_COLL_SYMMETRIC);
    if (ret != ncclSuccess) goto fail;

    if (handle->kernelType == FT_KERNEL_LSA) {
        const size_t flags_bytes = (size_t)handle->nRanks * sizeof(uint64_t);
        ret = ncclMemAlloc(&ctx->d_flags, flags_bytes);
        if (ret != ncclSuccess) goto fail;
        ret = ncclCommWindowRegister(handle->comm, ctx->d_flags, flags_bytes,
                                     &ctx->win_flags, NCCL_WIN_COLL_SYMMETRIC);
        if (ret != ncclSuccess) goto fail;

        /* All symmetric windows are now registered — create devComm here. */
        if (handle->devComm.nRanks != 0) {
            ncclDevCommDestroy(handle->comm, &handle->devComm);
            memset(&handle->devComm, 0, sizeof(handle->devComm));
        }
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ret = ncclDevCommCreate(handle->comm, &reqs, &handle->devComm);
        if (ret != ncclSuccess) goto fail;
        if (cudaDeviceSynchronize() != cudaSuccess) { ret = ncclSystemError; goto fail; }
    }

    *out = ctx;
    return ncclSuccess;

fail:
    ftAllReduceDestroy(ctx);
    return ret;
}

void ftAllReduceDestroy(FTAllReduceCtx* ctx) {
    if (!ctx) return;
    if (ctx->win_flags) ncclCommWindowDeregister(ctx->handle->comm, ctx->win_flags);
    if (ctx->win_recv)  ncclCommWindowDeregister(ctx->handle->comm, ctx->win_recv);
    if (ctx->win_send)  ncclCommWindowDeregister(ctx->handle->comm, ctx->win_send);
    if (ctx->d_output) cudaFree(ctx->d_output);
    if (ctx->d_flags)  ncclMemFree(ctx->d_flags);
    if (ctx->d_recv)   ncclMemFree(ctx->d_recv);
    if (ctx->d_send)   ncclMemFree(ctx->d_send);
    free(ctx);
}

ncclResult_t ftAllReduceSetData(FTAllReduceCtx* ctx, const float* data) {
    if (cudaMemcpy(ctx->d_send, data, ctx->count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return ncclSystemError;
    return ncclSuccess;
}

ncclResult_t ftAllReduceLaunch(FTAllReduceCtx* ctx,
                                cudaStream_t stream,
                                uint64_t timeout_us) {
    FTCollectiveHandle* h = ctx->handle;
    const size_t mask_bytes  = (size_t)h->nRanks * sizeof(bool);

    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* d_recv and d_output are intentionally NOT zeroed.  Present ranks fully
     * overwrite their recv slices, and the reduction writes every output[i]
     * unconditionally, so both memsets were dead work on the critical path (the
     * d_recv memset scaled with nRanks*count).  Absent-rank recv slices are
     * skipped by s_result_mask in the reduction, so staleness is harmless. */
    if (cudaMemsetAsync(h->d_mask_out, 0, mask_bytes,                 stream) != cudaSuccess) return ncclSystemError;

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);

    if (h->kernelType == FT_KERNEL_LSA) {
        const size_t flags_bytes = (size_t)h->nRanks * sizeof(uint64_t);
        if (cudaMemsetAsync(ctx->d_flags, 0, flags_bytes, stream) != cudaSuccess) return ncclSystemError;

        if (ftUseMultiCTA(h, ctx->count, sizeof(float))) {
            if (ftLaunchAllReduceMultiCTA<float>(
                    h, ctx->win_send, 0, ctx->win_recv, 0, ctx->win_flags,
                    ctx->d_output, ctx->count, timeout_cycles, h->d_gen, stream) != cudaSuccess)
                return ncclSystemError;
        } else {
        FTAllReduceKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                ctx->win_send, 0,
                ctx->win_recv, 0,
                ctx->win_flags,
                h->win_ready,
                ctx->d_output,
                ctx->count,
                h->devComm,
                timeout_cycles,
                h->d_gen,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
        }
    } else {
        FTAllReduceGINKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                ctx->win_send, 0,
                ctx->win_recv, 0,
                ctx->d_output,
                ctx->count,
                h->devComm,
                timeout_cycles,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
    }

    /* Self-adaptation: adopt observed survivors as the next collective's input
     * mask so a timed-out peer is excluded automatically (see self_adapt). */
    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out,
                            (size_t)h->nRanks * sizeof(bool),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* ASYNCHRONOUS: caller must cudaStreamSynchronize(stream). */
    return ncclSuccess;
}

ncclResult_t ftAllReduceGetResult(const FTAllReduceCtx* ctx, float* out) {
    if (cudaMemcpy(out, ctx->d_output, ctx->count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return ncclSystemError;
    return ncclSuccess;
}

// =============================================================================
// Window-based API (caller supplies pre-registered windows)
// =============================================================================

ncclResult_t ftAllGather(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          size_t count,
                          cudaStream_t stream,
                          uint64_t timeout_us) {
    /* Lazy devComm init for LSA: ftAllGatherCreate initialises devComm, but
     * callers that use the window-based API directly (e.g. the Python binding)
     * never call ftAllGatherCreate.  By the time this function is reached all
     * symmetric windows are already registered, satisfying the ordering
     * constraint.  The cudaDeviceSynchronize drains any NCCL-internal streams
     * left by ncclDevCommCreate before the first timed launch.              */
    if (h->kernelType == FT_KERNEL_LSA && h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }

    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);

    /* Upload the input mask only when it changed (set at create / by
     * ftHandleSetActiveMask).  Skipping the H2D on stable iterations removes a
     * per-call stream op from the small-message critical path.  No d_mask_out
     * pre-zeroing: the kernel writes every active_mask_out entry. */
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);

    if (h->kernelType == FT_KERNEL_LSA) {
        if (ftUseMultiCTA(h, count, sizeof(float))) {
            if (ftLaunchAllGatherMultiCTA<float>(
                    h, win_send, sendoffset, win_recv, recvoffset, win_flags,
                    count, timeout_cycles, h->d_gen, stream) != cudaSuccess)
                return ncclSystemError;
        } else {
        FTAllGatherKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                win_send, sendoffset,
                win_recv, recvoffset,
                win_flags,
                h->win_ready,
                count,
                h->devComm,
                timeout_cycles,
                h->d_gen,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
        }
    } else {
        /* GIN path: win_flags is ignored. */
        FTAllGatherGINKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                win_send, sendoffset,
                win_recv, recvoffset,
                count,
                h->devComm,
                timeout_cycles,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
    }

    /* Self-adaptation: adopt observed survivors as the next collective's input
     * mask so a timed-out peer is excluded automatically (see self_adapt). */
    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out,
                            (size_t)h->nRanks * sizeof(bool),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* ASYNCHRONOUS. */
    return ncclSuccess;
}

/* Collective rejoin kernel (LSA, single-CTA).  Re-admits stalled-but-alive ranks
 * without touching the communicator.  A dropped rank's d_gen has drifted, so a
 * generation-GATED handshake would never accept it.  Two clean phases over a
 * DEDICATED 2*nRanks window (region A = generation, region B = presence):
 *
 *   Phase 1 (region A, first half of the timeout): every rank CONTINUOUSLY posts
 *     its OWN generation — a STABLE value, not a running max — into region-A
 *     slot[me] of every peer.  Continuous posting covers launch skew (a late
 *     rank's value still lands and is re-posted so peers cannot miss it), and a
 *     stable value avoids the moving-target creep that stalled the earlier
 *     iterative version.  After the phase, region A holds every live rank's
 *     generation; G = max is the same on every rank (all read the same stable
 *     set), so gSync = G+1 is identical everywhere.
 *
 *   Phase 2 (region B, second half): every rank continuously posts gSync into
 *     region-B slot[me] of every peer and marks present[r] once peer r's region-B
 *     slot reaches gSync.  Presence is sticky and region B is separate from region
 *     A, so it never feeds back into the phase-1 max.  A still-dead rank never
 *     posts gSync and stays absent.
 *
 * Poll set is the FULL world.  Commits present -> active_mask_out and gSync ->
 * *d_gen.  Loop bounds are a shared flag (thread 0 owns the clock) so all threads
 * run the same iteration count — a clock-divergent __syncthreads would deadlock. */
__global__ void FTRejoinKernel(
    ncclWindow_t rejoinwin,
    struct ncclDevComm devComm,
    uint64_t*    d_gen,
    uint64_t     timeout_cycles,
    bool*        active_mask_out)
{
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int nR = devComm.nRanks;
    const int me = devComm.rank;
    const uint64_t startCycle = clock64();
    const uint64_t half = timeout_cycles / 2;
    const uint64_t myGen = *d_gen;                 /* stable value we advertise */

    __shared__ bool     s_run;
    __shared__ uint64_t s_gSync;
    __shared__ bool     s_present[FT_MAX_RANKS];
    if (tid == 0) s_run = true;
    if (tid < nR) s_present[tid] = (tid == me);
    __syncthreads();

    /* ---- Phase 1: post STABLE generation into region A ---- */
    while (s_run) {
        for (int p = tid; p < nR; p += nthreads) {
            if (p == me) continue;
            uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
                rejoinwin, (size_t)me * sizeof(uint64_t), p);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
                .store(myGen, cuda::memory_order_release);
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) s_run = (clock64() - startCycle < half);
        __syncthreads();
    }

    /* Reduce region A -> global max G; gSync = G + 1 (identical on every rank). */
    if (tid == 0) {
        uint64_t mx = myGen;
        for (int r = 0; r < nR; r++) {
            if (r == me) continue;
            uint64_t* slot = (uint64_t*)ncclGetLocalPointer(
                rejoinwin, (size_t)r * sizeof(uint64_t));
            uint64_t v = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*slot)
                             .load(cuda::memory_order_acquire);
            if (v > mx) mx = v;
        }
        s_gSync = mx + 1;
        s_run = true;
    }
    __syncthreads();
    const uint64_t gSync = s_gSync;

    /* ---- Phase 2: post gSync into region B, accumulate presence ---- */
    while (s_run) {
        for (int p = tid; p < nR; p += nthreads) {
            if (p == me) continue;
            uint64_t* peerSlot = (uint64_t*)ncclGetLsaPointer(
                rejoinwin, (size_t)(nR + me) * sizeof(uint64_t), p);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peerSlot)
                .store(gSync, cuda::memory_order_release);
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            for (int r = 0; r < nR; r++) {
                if (r == me) continue;
                uint64_t* slot = (uint64_t*)ncclGetLocalPointer(
                    rejoinwin, (size_t)(nR + r) * sizeof(uint64_t));
                uint64_t v = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*slot)
                                 .load(cuda::memory_order_acquire);
                if (v >= gSync) s_present[r] = true;
            }
            s_run = (clock64() - startCycle < timeout_cycles);
        }
        __syncthreads();
    }

    if (tid < nR) active_mask_out[tid] = s_present[tid];
    __syncthreads();
    if (tid == 0) *d_gen = gSync;   /* commit the synchronized generation */
}

/* Collective rejoin C entry (LSA only).  ASYNCHRONOUS: installs the new active
 * set as the input mask on-stream; after syncing, ftHandleGetResultMask reports
 * it.  The kernel also synchronizes the handle generation. */
ncclResult_t ftRejoin(FTCollectiveHandle* h,
                      cudaStream_t stream,
                      uint64_t timeout_us) {
    if (h->kernelType != FT_KERNEL_LSA)
        return ncclInvalidUsage;   /* rejoin is LSA-only */
    if (h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }
    const uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    /* NOTE: no ftBumpGen — the kernel negotiates and commits d_gen itself. */
    FTRejoinKernel<<<1, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
        h->win_rejoin, h->devComm, h->d_gen, timeout_cycles, (bool*)h->d_mask_out);
    /* Commit the negotiated membership on-stream so it reaches d_mask_in through the
     * STABLE device pointer, without relying on the host-launch mask_dirty upload
     * (which CUDA-graph replay bypasses — a graph captured before the rejoin would
     * otherwise keep reading stale membership).  Mirrors the self_adapt
     * d_mask_out -> d_mask_in copy.  The D2H keeps h_mask_in coherent for host reads
     * after the caller syncs the stream; mask_dirty is cleared since d_mask_in is now
     * authoritative. */
    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);
    if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out, mask_bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return ncclSystemError;
    if (cudaMemcpyAsync(h->h_mask_in, h->d_mask_out, mask_bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return ncclSystemError;
    h->mask_dirty = false;
    return ncclSuccess;
}

/* Uniform all-to-all (Phase 0).  `count` = per-destination element count; the
 * send window holds nRanks*count elements (chunk j -> rank j), the recv window
 * holds nRanks padded slots (slot i = data from rank i), same layout as
 * AllGather.  LSA only for now; GIN falls back to the caller (returns
 * ncclInvalidUsage so the Python binding routes to stock NCCL). */
ncclResult_t ftAllToAll(FTCollectiveHandle* h,
                         ncclWindow_t win_send, size_t sendoffset,
                         ncclWindow_t win_recv, size_t recvoffset,
                         ncclWindow_t win_flags,
                         size_t count,
                         cudaStream_t stream,
                         uint64_t timeout_us) {
    if (h->kernelType != FT_KERNEL_LSA)
        return ncclInvalidUsage;   /* GIN all-to-all not implemented yet */

    /* Lazy devComm init (see ftAllGather). */
    if (h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }

    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);

    if (ftUseMultiCTA(h, count, sizeof(float))) {
        if (ftLaunchAllToAllMultiCTA<float>(
                h, win_send, sendoffset, win_recv, recvoffset, win_flags,
                count, timeout_cycles, h->d_gen, stream) != cudaSuccess)
            return ncclSystemError;
    } else {
        FTAllToAllKernel<float>
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                win_send, sendoffset,
                win_recv, recvoffset,
                win_flags,
                h->win_ready,
                count,
                h->devComm,
                timeout_cycles,
                h->d_gen,
                h->d_mask_in,
                (bool*)h->d_mask_out,
                h->d_error);
    }

    /* Self-adaptation: adopt observed survivors (see ftAllGather). */
    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out,
                            (size_t)h->nRanks * sizeof(bool),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* ASYNCHRONOUS. */
    return ncclSuccess;
}

/* Variable all-to-all (Phase 1, single-CTA).  d_send_counts/d_send_displs are
 * device arrays (nRanks) in float32-element units; chunk d (send_displs[d],
 * send_counts[d] elements) goes to rank d's fixed-stride scratch slot[myRank].
 * recv_stride_bytes = the per-source slot stride (>= max_per_dest*4).  The caller
 * compacts the per-source slots into its packed output using the recv counts.
 * LSA only; GIN returns ncclInvalidUsage. */
/* Advance the readiness generation on-device (one thread). Launched on the
 * stream just before a collective so the generation increments PER EXECUTION —
 * including per CUDA-graph replay, where a host-incremented generation would be
 * baked at capture and stay constant, defeating the readiness handshake. */
__global__ void ftBumpGen(uint64_t* g) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *g += 1;
}

/* Exclusive prefix-sum of counts -> displs (nRanks small; one thread). */
__global__ void ftExclusivePrefixSum(const int* counts, size_t* displs, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        size_t acc = 0;
        for (int i = 0; i < n; i++) { displs[i] = acc; acc += (size_t)counts[i]; }
    }
}

ncclResult_t ftAllToAllv(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          const int* d_send_counts, const size_t* d_send_displs,
                          size_t recv_stride_bytes, size_t max_send_count,
                          void* output, const int* d_recv_counts,
                          const size_t* d_recv_displs,
                          int* d_recv_counts_out, size_t recv_capacity,
                          size_t row_size,
                          FTDataType dtype,
                          cudaStream_t stream, uint64_t timeout_us) {
    if (h->kernelType != FT_KERNEL_LSA)
        return ncclInvalidUsage;
    if (row_size == 0) row_size = 1;

    /* Counts-only convenience: if the caller passes NULL send displacements,
     * derive them on-device from the counts (no host round-trip). */
    if (d_send_displs == NULL) {
        ftExclusivePrefixSum<<<1, 32, 0, stream>>>(
            d_send_counts, h->d_a2av_displs, h->nRanks);
        d_send_displs = h->d_a2av_displs;
    }

    if (h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }

    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);

    /* Dispatch on dtype (byte-copy kernel templated for the int4 tail element
     * granularity) and on size (single- vs cooperative multi-CTA, keyed on the
     * largest per-destination count).  counts/displs are in native T-element
     * units; recv_stride_bytes is the fixed per-source slot stride. */
#define _FT_A2AV_LAUNCH(T)                                                       \
    do {                                                                         \
        if (ftUseMultiCTA(h, max_send_count, sizeof(T))) {                       \
            if (ftLaunchAllToAllvMultiCTA<T>(                                    \
                    h, win_send, sendoffset, win_recv, recvoffset, win_flags,    \
                    d_send_counts, d_send_displs, recv_stride_bytes,             \
                    (T*)output, d_recv_counts, d_recv_displs,                    \
                    d_recv_counts_out, recv_capacity, row_size,                  \
                    max_send_count, timeout_cycles, h->d_gen, stream)          \
                != cudaSuccess) return ncclSystemError;                          \
        } else {                                                                 \
            FTAllToAllvKernel<T>                                                 \
                <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>( \
                    win_send, sendoffset, win_recv, recvoffset, win_flags,       \
                    h->win_ready, d_send_counts, d_send_displs,                  \
                    recv_stride_bytes, (T*)output, d_recv_counts, d_recv_displs, \
                    d_recv_counts_out, recv_capacity, row_size,                  \
                    h->devComm, timeout_cycles, h->d_gen,                      \
                    h->d_mask_in, (bool*)h->d_mask_out, h->d_error);             \
        }                                                                        \
    } while (0)

    if      (dtype == FT_DTYPE_FLOAT16)  { _FT_A2AV_LAUNCH(__half); }
    else if (dtype == FT_DTYPE_BFLOAT16) { _FT_A2AV_LAUNCH(__nv_bfloat16); }
    else                                 { _FT_A2AV_LAUNCH(float); }
#undef _FT_A2AV_LAUNCH

    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out,
                            (size_t)h->nRanks * sizeof(bool),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* ASYNCHRONOUS. */
    return ncclSuccess;
}

ncclResult_t ftAllGatherv(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          const int* d_send_counts,   /* device int32[nRanks]; [myRank] used */
                          void* output,               /* PACKED device output */
                          int* d_recv_counts_out,     /* device int32[nRanks] discovered */
                          size_t recv_capacity, size_t row_bytes,
                          size_t recv_stride, size_t max_send_count,
                          cudaStream_t stream, uint64_t timeout_us) {
    if (h->kernelType != FT_KERNEL_LSA) return ncclInvalidUsage;
    if (d_send_counts == NULL || d_recv_counts_out == NULL || output == NULL)
        return ncclInvalidArgument;
    if (recv_capacity < 1 || row_bytes < 1) return ncclInvalidArgument;
    /* The fixed scratch slot must hold a full-capacity chunk. */
    if (recv_capacity * row_bytes > recv_stride) return ncclInvalidArgument;
    if (max_send_count > recv_capacity) return ncclInvalidArgument;

    if (h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }

    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);
    /* Small messages take the single-CTA path (no cooperative launch / grid.sync
     * overhead); large ones the cooperative multi-CTA kernel. */
    if (ftUseMultiCTA(h, max_send_count, row_bytes)) {
        int blocks = ftCoopGrid(h, max_send_count, row_bytes);
        if (blocks < 1) blocks = 1;
        void* args[] = {
            &win_send, &sendoffset, &win_recv, &recvoffset, &win_flags,
            &h->win_ready, &d_send_counts, &d_recv_counts_out, &h->d_a2av_displs,
            &output, &recv_capacity, &row_bytes, &recv_stride,
            &h->devComm, &timeout_cycles, &h->d_gen,
            &h->d_mask_in, &h->d_mask_out, &h->d_can_push, &h->d_error
        };
        if (cudaLaunchCooperativeKernel(
                (void*)FTAllGathervKernel,
                dim3(blocks), dim3(FT_DEVICE_THREADS_PER_CTA), args, 0, stream)
            != cudaSuccess) return ncclSystemError;
    } else {
        FTAllGathervKernelSingle
            <<<1, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                win_send, sendoffset, win_recv, recvoffset, win_flags,
                h->win_ready, d_send_counts, d_recv_counts_out, h->d_a2av_displs,
                output, recv_capacity, row_bytes, recv_stride,
                h->devComm, timeout_cycles, h->d_gen,
                h->d_mask_in, (bool*)h->d_mask_out, h->d_error);
    }

    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out, mask_bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }
    return ncclSuccess;   /* ASYNCHRONOUS */
}

ncclResult_t ftReduceScatterv(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          const int* d_counts, const size_t* d_displs,
                          void* d_output,
                          size_t recv_stride, size_t max_shard_count,
                          FTDataType dtype,
                          cudaStream_t stream, uint64_t timeout_us) {
    if (h->kernelType != FT_KERNEL_LSA) return ncclInvalidUsage;
    if (d_counts == NULL || d_output == NULL) return ncclInvalidArgument;
    if (recv_stride < 1 || (recv_stride % 16) != 0) return ncclInvalidArgument;
    if (max_shard_count < 1) return ncclInvalidArgument;

    if (h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }

    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);
    /* Shard offsets: derive on-device from the counts partition if not given. */
    if (d_displs == NULL) {
        ftExclusivePrefixSum<<<1, 32, 0, stream>>>(
            d_counts, h->d_a2av_displs, h->nRanks);
        d_displs = h->d_a2av_displs;
    }

    /* Small shards take the single-CTA path; large ones the cooperative kernel. */
#define _FT_RSV_LAUNCH(T)                                                         \
        do {                                                                     \
            if (ftUseMultiCTA(h, max_shard_count, sizeof(T))) {                  \
                if (ftLaunchReduceScattervMultiCTA<T>(                           \
                    h, win_send, sendoffset, win_recv, recvoffset, win_flags,    \
                    d_counts, d_displs, (T*)d_output, recv_stride,               \
                    max_shard_count, timeout_cycles, h->d_gen, stream)           \
                != cudaSuccess) return ncclSystemError;                          \
            } else {                                                             \
                FTReduceScattervKernelSingle<T>                                  \
                    <<<1, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>(               \
                        win_send, sendoffset, win_recv, recvoffset, win_flags,   \
                        h->win_ready, d_counts, d_displs, (T*)d_output,          \
                        recv_stride, h->devComm, timeout_cycles, h->d_gen,       \
                        h->d_mask_in, (bool*)h->d_mask_out, h->d_error);         \
            }                                                                    \
        } while (0)
    if      (dtype == FT_DTYPE_FLOAT16)  { _FT_RSV_LAUNCH(__half); }
    else if (dtype == FT_DTYPE_BFLOAT16) { _FT_RSV_LAUNCH(__nv_bfloat16); }
    else                                 { _FT_RSV_LAUNCH(float); }
#undef _FT_RSV_LAUNCH

    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out, mask_bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }
    return ncclSuccess;   /* ASYNCHRONOUS */
}

ncclResult_t ftAllReduce(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          void* d_output,
                          size_t count,
                          FTDataType dtype,
                          cudaStream_t stream,
                          uint64_t timeout_us) {
    /* Lazy devComm init — same rationale as ftAllGather (window-based API). */
    if (h->kernelType == FT_KERNEL_LSA && h->devComm.nRanks == 0) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        ncclResult_t ret = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
        if (ret != ncclSuccess) return ret;
        if (cudaDeviceSynchronize() != cudaSuccess) return ncclSystemError;
    }

    const size_t mask_bytes = (size_t)h->nRanks * sizeof(bool);

    /* Upload the input mask only when it changed (set at create / by
     * ftHandleSetActiveMask).  Skipping the H2D on stable iterations removes a
     * per-call stream op from the small-message critical path.  No d_mask_out
     * pre-zeroing: the kernel writes every active_mask_out entry. */
    if (h->mask_dirty) {
        if (cudaMemcpyAsync(h->d_mask_in, h->h_mask_in, mask_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    uint64_t timeout_cycles = timeout_us * h->cycles_per_us;
    ftBumpGen<<<1, 1, 0, stream>>>(h->d_gen);

    if (h->kernelType == FT_KERNEL_LSA) {
#define _FT_AR_LSA_LAUNCH(T) \
        do { \
            if (ftUseReduceScatter(h, count, sizeof(T))) { \
                /* Large + full set: bandwidth-optimal reduce-scatter (in-place, \
                 * output aliases win_send on this window path). */ \
                if (ftLaunchAllReduceRSMultiCTA<T>( \
                        h, win_send, sendoffset, win_recv, recvoffset, win_flags, \
                        (T*)d_output, count, timeout_cycles, h->d_gen, stream) != cudaSuccess) \
                    return ncclSystemError; \
            } else if (ftUseMultiCTA(h, count, sizeof(T))) { \
                if (ftLaunchAllReduceMultiCTA<T>( \
                        h, win_send, sendoffset, win_recv, recvoffset, win_flags, \
                        (T*)d_output, count, timeout_cycles, h->d_gen, stream) != cudaSuccess) \
                    return ncclSystemError; \
            } else { \
                FTAllReduceKernel<T> \
                    <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>( \
                        win_send, sendoffset, win_recv, recvoffset, win_flags, \
                        h->win_ready, (T*)d_output, count, h->devComm, \
                        timeout_cycles, h->d_gen, h->d_mask_in, (bool*)h->d_mask_out, h->d_error); \
            } \
        } while (0)

        if      (dtype == FT_DTYPE_FLOAT16)  { _FT_AR_LSA_LAUNCH(__half); }
        else if (dtype == FT_DTYPE_BFLOAT16) { _FT_AR_LSA_LAUNCH(__nv_bfloat16); }
        else                                 { _FT_AR_LSA_LAUNCH(float); }
#undef _FT_AR_LSA_LAUNCH
    } else {
        /* GIN path: win_flags is ignored. */
#define _FT_AR_GIN_LAUNCH(T) \
        FTAllReduceGINKernel<T> \
            <<<FT_DEVICE_CTA_COUNT, FT_DEVICE_THREADS_PER_CTA, 0, stream>>>( \
                win_send, sendoffset, win_recv, recvoffset, \
                (T*)d_output, count, h->devComm, \
                timeout_cycles, h->d_mask_in, (bool*)h->d_mask_out, h->d_error)

        if      (dtype == FT_DTYPE_FLOAT16)  { _FT_AR_GIN_LAUNCH(__half); }
        else if (dtype == FT_DTYPE_BFLOAT16) { _FT_AR_GIN_LAUNCH(__nv_bfloat16); }
        else                                 { _FT_AR_GIN_LAUNCH(float); }
#undef _FT_AR_GIN_LAUNCH
    }

    /* Self-adaptation: adopt observed survivors as the next collective's input
     * mask so a timed-out peer is excluded automatically (see self_adapt). */
    if (h->self_adapt) {
        if (cudaMemcpyAsync(h->d_mask_in, h->d_mask_out,
                            (size_t)h->nRanks * sizeof(bool),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return ncclSystemError;
        h->mask_dirty = false;
    }

    /* ASYNCHRONOUS. */
    return ncclSuccess;
}
