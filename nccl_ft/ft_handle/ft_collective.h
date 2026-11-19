/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#pragma once

#include "nccl.h"
#include "nccl_device.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Launch functions enqueue work on the supplied CUDA stream. Synchronize that
 * stream before reading output, the result mask, or the mapped error status.
 * ftHandleCreate selects LSA or GIN from communicator capabilities. */

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Status type
// =============================================================================

/*
 * FTStatus — error code written by the kernel into mapped pinned memory.
 *
 * FT_OK      : no timeout detected during the last launch.
 * FT_TIMEOUT : at least one active rank failed to signal readiness or deliver
 *              its data before timeout_cycles expired.
 * FT_CAPACITY_EXCEEDED : a variable all-to-allv saw a device-resident
 *              send_count larger than recv_capacity. The offending push is
 *              clamped to recv_capacity so no scratch/output memory is
 *              corrupted, but rows beyond capacity are dropped — the caller's
 *              routing kernel should reject such counts before launching.
 *
 * The field is sticky: once set to a non-OK code it stays until the caller
 * calls ftHandleClearError(). The kernel only ever sets error codes, never
 * clears.
 */
typedef enum {
    FT_OK                = 0,
    FT_TIMEOUT           = 1,
    FT_CAPACITY_EXCEEDED = 2,
} FTStatus;

// =============================================================================
// Kernel type
// =============================================================================

/*
 * FTKernelType — selected once at ftHandleCreate() time, not changeable.
 *
 * FT_KERNEL_LSA : Uses NVLink symmetric memory (ncclGetLsaPointer) for direct
 *                 GPU-to-GPU writes and a custom readywin barrier.  Requires
 *                 nLsaTeams >= 1 in the communicator properties.
 *
 * FT_KERNEL_GIN : Uses GPU-Initiated Networking (ncclGin / RDMA) for remote
 *                 writes and ncclGinBarrierSession for the barrier.  Used when
 *                 LSA is unavailable (e.g. multi-node InfiniBand setup).
 *                 Requires ginType != NCCL_GIN_TYPE_NONE.
 */
typedef enum {
    FT_KERNEL_LSA = 0,
    FT_KERNEL_GIN = 1,
} FTKernelType;

// =============================================================================
// FTCollectiveHandle
// =============================================================================

struct FTCollectiveHandle {
    ncclComm_t   comm;        /* borrowed — not freed on destroy             */
    ncclDevComm  devComm;
    int          nRanks;
    FTKernelType kernelType;  /* set at creation, selects LSA or GIN kernels */

    /* Input mask: written by ftHandleSetActiveMask, uploaded before each launch.
     * mask_dirty is set on creation and by ftHandleSetActiveMask; cleared after the
     * first cudaMemcpyAsync upload so subsequent launches skip the H2D copy
     * when the mask is unchanged (the common non-fault case).               */
    bool*        h_mask_in;   /* host   bool[nRanks]  (malloc)               */
    bool*        d_mask_in;   /* device bool[nRanks]  (cudaMalloc)           */
    bool         mask_dirty;  /* true → d_mask_in needs re-upload            */

    /* Output mask: device memory written by kernel, copied to h_mask_out
     * on demand when ftHandleGetResultMask() is called.                     */
    bool*        h_mask_out;  /* host   bool[nRanks]  (malloc)               */
    bool*        d_mask_out;  /* device bool[nRanks]  (cudaMalloc)           */

    /* Self-adaptation (default ON): after each collective the launch copies
     * d_mask_out -> d_mask_in on the stream, so the next collective automatically
     * expects only the ranks that responded — a timed-out peer is excluded with
     * no manual ftHandleSetActiveMask().  This adopts THIS rank's LOCAL observation, so
     * masks may transiently diverge across ranks under a mid-collective failure;
     * for a true crash they converge within a few timeouts (the dead rank never
     * responds again).  Callers needing cross-rank-consistent membership should
     * disable this and drive membership with a consensus step (as FTProcessGroup
     * does via its ft_converge barrier).  Toggle with ftHandleSetSelfAdapt().     */
    bool         self_adapt;

    /* Readywin: liveness signal for the LSA path (LSA only; NULL for GIN).
     *
     * PUSH model with a monotonic generation counter.  This rank's symmetric
     * copy has nRanks uint64 slots; slot[r] holds "the generation rank r last
     * signaled to me".  To signal, a rank WRITES its current generation into
     * every peer's slot[myRank] (a posted NVLink store).  To detect liveness it
     * POLLS only its OWN local slots for >= the current generation.
     *
     * This is what makes a hard peer failure (killed process → freed memory,
     * torn-down NVLink mappings) survivable: the hot poll path only ever READS
     * LOCAL memory, never a peer's, so it cannot stall on a dead peer's
     * unmapped memory the way a pull/load-from-peer poll did.  The monotonic
     * generation (never reset) also removes the stale-readywin race, so no
     * cross-rank pre-launch barrier is needed.                                */
    void*        d_ready;     /* symmetric uint64_t[nRanks] (ncclMemAlloc) or NULL */
    ncclWindow_t win_ready;   /* registered window                         or NULL */
    /* Dedicated rejoin window: 2*nRanks uint64 slots.  Region A [0,nRanks) is the
     * generation max-consensus, region B [nRanks,2*nRanks) is presence-at-gSync.
     * Separate regions keep the two rejoin phases from contaminating each other;
     * a separate window keeps rejoin off the data-collective readiness path.     */
    void*        d_rejoin;    /* symmetric uint64_t[2*nRanks] (ncclMemAlloc) or NULL */
    ncclWindow_t win_rejoin;  /* registered window                           or NULL */
    uint64_t     gen;         /* generation counter, incremented per launch       */

    /* Error status: mapped pinned — kernel writes FT_TIMEOUT via system-scope
     * atomic; host reads h_error directly after cudaStreamSynchronize.      */
    FTStatus*    h_error;     /* host alias  (cudaHostAllocMapped)           */
    FTStatus*    d_error;     /* device alias (cudaHostGetDevicePointer)     */

    /* GPU clock rate cached at creation time: GPU cycles per microsecond.
     * Used to convert timeout_us → timeout_cycles without querying the
     * driver on every ftAllGatherLaunch / ftAllReduceLaunch call.           */
    uint64_t     cycles_per_us;

    /* Multi-CTA (LSA): for transfers at/above multicta_threshold_bytes (per
     * rank) the launch uses a cooperative grid of up to max_coop_blocks CTAs
     * with grid-wide cg::grid_group barriers, instead of the single-CTA kernel.
     * d_can_push is grid-global scratch replacing the per-block shared
     * s_can_push.  max_coop_blocks == 0 disables the path (cooperative launch
     * unsupported or non-LSA).  Tunable via FT_MULTICTA_THRESHOLD_BYTES.      */
    bool*        d_can_push;             /* device bool[nRanks] (cudaMalloc)   */
    int*         d_a2av_ctr;             /* device int, all_to_allv work-steal */
    size_t*      d_a2av_displs;          /* device size_t[nRanks], derived displs */
    uint64_t*    d_gen;                  /* device readiness generation, bumped
                                          * per launch on-device so it advances
                                          * under CUDA-graph replay (a host-baked
                                          * generation would stay constant across
                                          * replays and defeat the readiness
                                          * handshake).                          */
    int          max_coop_blocks;        /* occupancy * #SM, 0 = disabled      */
    size_t       multicta_threshold_bytes;

    /* AllReduce reduce-scatter path (LSA window API only): at/above
     * rs_threshold_bytes (per rank) AND when the full set is active, AllReduce
     * uses a cooperative reduce-scatter + allgather kernel (O(1)/element data
     * movement) instead of gather-then-reduce.  Any masked rank or a smaller
     * size falls back to the gather-then-reduce paths.  Tunable via
     * FT_REDUCE_SCATTER_THRESHOLD_BYTES; 0 disables. */
    size_t       rs_threshold_bytes;

    /* Reduce-scatter AllReduce: number of chunks to pipeline the owner's
     * reduce<->broadcast over, so the reduction of chunk c+1 overlaps the
     * NVLink broadcast of chunk c (hides the reduce phase + its barriers behind
     * the broadcast).  0 or 1 = disabled (bulk reduce then bulk broadcast, the
     * proven path).  Tunable via FT_RS_PIPELINE_CHUNKS. */
    int          rs_pipeline_chunks;
};

/* Create a handle from an existing communicator.
 *
 * Queries ncclCommProperties to select the kernel backend:
 *   - FT_KERNEL_LSA if nLsaTeams >= 1
 *   - FT_KERNEL_GIN if ginType != NCCL_GIN_TYPE_NONE (fallback)
 *   - ncclInvalidUsage if neither is available
 *
 * initial_mask : host bool[nRanks] copied into h_mask_in. */
ncclResult_t ftHandleCreate(ncclComm_t comm, int nRanks,
                             const bool* initial_mask,
                             FTCollectiveHandle** out);

/* Destroy handle and all owned resources. Does not touch the communicator. */
void ftHandleDestroy(FTCollectiveHandle* h);

/* Replace the active-membership mask used for subsequent launches.
 * mask : host bool[nRanks] — true = rank participates in the next collective(s).
 *
 * COORDINATED: every rank that runs the next collective together must install
 * the SAME mask.  A divergent mask splits membership and can deadlock the
 * collective (some ranks wait on a peer others have dropped).  Use this to let
 * a framework own membership (e.g. after an external health check) instead of
 * the on-device timeout / consensus barrier. */
void ftHandleSetActiveMask(FTCollectiveHandle* h, const bool* mask);

/* Deprecated alias for ftHandleSetActiveMask (kept for backward compatibility). */
void ftHandleSetMask(FTCollectiveHandle* h, const bool* mask);

/* Enable/disable self-adaptation (default enabled).  When enabled, each launch
 * copies the result mask (d_mask_out) into the input mask (d_mask_in) on the
 * stream, so the next collective automatically excludes ranks that timed out.
 * See the self_adapt field for the consistency caveat; disable it when a
 * consensus mechanism owns membership. */
void ftHandleSetSelfAdapt(FTCollectiveHandle* h, bool enable);

/* Copy d_mask_out → h_mask_out and return a pointer to h_mask_out.
 *
 * MUST be called after cudaStreamSynchronize(stream) — reads device memory. */
const bool* ftHandleGetResultMask(const FTCollectiveHandle* h);

/* Return the current error status.
 *
 * Reads directly from mapped pinned memory — no device copy needed.
 * MUST be called after cudaStreamSynchronize(stream). */
FTStatus ftHandleGetError(const FTCollectiveHandle* h);

/* Clear the error status back to FT_OK.
 * Must only be called when no kernel referencing this handle is in flight. */
void ftHandleClearError(FTCollectiveHandle* h);

/* Read the current on-device generation counter (D2H copy).  Diagnostic /
 * membership: after ftRejoin all survivors should report an identical value. */
uint64_t ftHandleGetGeneration(const FTCollectiveHandle* h);

/* Collective rejoin (LSA only) — sibling to the consensus barrier that GROWS the
 * membership.  Re-admits stalled-but-alive ranks WITHOUT changing the NCCL
 * communicator: every live rank calls it (full-world poll), a generation
 * max-consensus synchronizes the drifted on-device generation counter, and a
 * generation-gated presence round finds the responders.  The new active set is
 * written to the result mask AND installed as the input mask on-stream (so a
 * previously captured CUDA graph sees it without a host-side upload); the
 * synchronized generation is committed into the handle.
 * Still-dead ranks time out and are excluded.  ASYNCHRONOUS — sync the stream
 * before reading the result.  (Restores participation only; a returnee's DATA is
 * stale and is the caller's responsibility to resync.) */
ncclResult_t ftRejoin(FTCollectiveHandle* h, cudaStream_t stream, uint64_t timeout_us);

// =============================================================================
// FTAllGatherCtx
// =============================================================================

/*
 * Unified AllGather context — works with both FT_KERNEL_LSA and FT_KERNEL_GIN.
 *
 * win_flags / d_flags are allocated and used only for FT_KERNEL_LSA.
 * For FT_KERNEL_GIN they are NULL (GIN uses per-devComm signals instead).
 */
struct FTAllGatherCtx {
    FTCollectiveHandle* handle;   /* borrowed                                 */
    size_t       count;           /* float elements per rank                  */
    void*        d_send;          /* symmetric  count * sizeof(float)         */
    void*        d_recv;          /* symmetric  nRanks slots of roundup16(count*4) B */
    ncclWindow_t win_send;
    ncclWindow_t win_recv;
    /* LSA-only — NULL for GIN: */
    void*        d_flags;         /* symmetric  nRanks * sizeof(uint64_t)     */
    ncclWindow_t win_flags;
};

/* Create an AllGather context attached to handle.
 * Allocates and registers symmetric windows (win_flags only for LSA). */
ncclResult_t ftAllGatherCreate(FTCollectiveHandle* handle, size_t count,
                                FTAllGatherCtx** out);

/* Destroy context and free/deregister all owned windows and buffers. */
void ftAllGatherDestroy(FTAllGatherCtx* ctx);

/* Copy host float[count] into the symmetric send buffer. */
ncclResult_t ftAllGatherSetData(FTAllGatherCtx* ctx, const float* data);

/* Launch the AllGather collective on stream.
 *
 * ASYNCHRONOUS: returns before the kernel finishes.
 * Call cudaStreamSynchronize(stream) before reading results or error status.
 *
 * On timeout, FT_TIMEOUT is set in handle->h_error (sticky until cleared). */
ncclResult_t ftAllGatherLaunch(FTAllGatherCtx* ctx,
                                cudaStream_t stream,
                                uint64_t timeout_us);

/* Copy gathered data to host (float[nRanks * count]).
 * Call after cudaStreamSynchronize(stream). */
ncclResult_t ftAllGatherGetResult(const FTAllGatherCtx* ctx, float* out);

// =============================================================================
// FTAllReduceCtx
// =============================================================================

/*
 * Unified AllReduce context — works with both FT_KERNEL_LSA and FT_KERNEL_GIN.
 *
 * win_flags / d_flags are allocated and used only for FT_KERNEL_LSA.
 */
struct FTAllReduceCtx {
    FTCollectiveHandle* handle;   /* borrowed                                 */
    size_t       count;           /* float elements per rank                  */
    void*        d_send;          /* symmetric  count * sizeof(float)         */
    void*        d_recv;          /* symmetric  nRanks slots of roundup16(count*4) B */
    float*       d_output;        /* non-symmetric reduction output            */
    ncclWindow_t win_send;
    ncclWindow_t win_recv;
    /* LSA-only — NULL for GIN: */
    void*        d_flags;         /* symmetric  nRanks * sizeof(uint64_t)     */
    ncclWindow_t win_flags;
};

ncclResult_t ftAllReduceCreate(FTCollectiveHandle* handle, size_t count,
                                FTAllReduceCtx** out);
void         ftAllReduceDestroy(FTAllReduceCtx* ctx);
ncclResult_t ftAllReduceSetData(FTAllReduceCtx* ctx, const float* data);

/* Launch the AllReduce collective on stream.
 *
 * ASYNCHRONOUS: returns before the kernel finishes.
 * Call cudaStreamSynchronize(stream) before reading results or error status. */
ncclResult_t ftAllReduceLaunch(FTAllReduceCtx* ctx,
                                cudaStream_t stream,
                                uint64_t timeout_us);

/* Copy reduced output to host (float[count]).
 * Call after cudaStreamSynchronize(stream). */
ncclResult_t ftAllReduceGetResult(const FTAllReduceCtx* ctx, float* out);

// =============================================================================
// Element dtype for window-based AllReduce
// =============================================================================

/*
 * FTDataType — element type of the send/recv buffers and the output buffer.
 *
 * Passed to ftAllReduce (window API) to select the correct template
 * instantiation.  The reduction always accumulates in float32 internally
 * (Option-B), so bfloat16 and float16 are supported without precision loss
 * from half-precision accumulation chains.
 */
typedef enum {
    FT_DTYPE_FLOAT32  = 0,
    FT_DTYPE_FLOAT16  = 1,
    FT_DTYPE_BFLOAT16 = 2,
} FTDataType;

// =============================================================================
// Window-based API (caller supplies pre-registered nccl4py windows)
//
// These functions accept caller-owned ncclWindow_t handles directly, allowing
// integration with nccl4py's RegisteredWindowHandle.  The handle owns the
// readywin (LSA) or GIN barrier resources, mask buffers, devComm, and error
// status.
//
// ASYNCHRONOUS: both functions return before the kernel finishes.
// Call cudaStreamSynchronize(stream) before reading results or error status.
//
// Caller responsibilities:
//   - Allocate symmetric memory (ncclMemAlloc) and register windows
//     (ncclCommWindowRegister) via nccl4py before calling.
//   - For FT_KERNEL_LSA: zero win_recv and win_flags before each call.
//   - For FT_KERNEL_GIN: zero win_recv before each call; win_flags is ignored.
//   - Deregister and free windows after use.
// =============================================================================

/* AllGather with caller-supplied windows.
 *
 *   win_send  : symmetric, count * sizeof(float) at sendoffset
 *   win_recv  : symmetric, nRanks slots at recvoffset; each slot is
 *               roundup16(count * sizeof(float)) bytes (slot r = rank r's data)
 *   win_flags : symmetric, nRanks * sizeof(uint64_t) — IGNORED for GIN kernel
 *
 *   count     : Data volume expressed in float32-element units (sizeof = 4).
 *               The kernel copies count * 4 bytes per rank — it is agnostic
 *               to the actual element type.  For non-float32 dtypes, convert:
 *                 count = numel * element_size / sizeof(float)
 *               This requires numel * element_size to be divisible by 4
 *               (float16/bfloat16 need an even numel).
 *
 * Dispatches to FTAllGatherKernel (LSA) or FTAllGatherGINKernel (GIN)
 * based on handle->kernelType. */
ncclResult_t ftAllGather(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          size_t count,
                          cudaStream_t stream,
                          uint64_t timeout_us);

/* Uniform all-to-all with caller-supplied windows (Phase 0).
 *
 *   count     : per-destination element count in float32-element units (see
 *               ftAllGather for the byte/dtype convention).
 *   win_send  : symmetric, holds nRanks * count elements; chunk j (offset
 *               j*count) is sent to rank j.
 *   win_recv  : symmetric, nRanks padded slots (FT_RECV_STRIDE); slot i holds
 *               the chunk received from rank i (same layout as AllGather recv).
 *   win_flags : symmetric, nRanks * sizeof(uint64_t).
 *
 * LSA only for now — returns ncclInvalidUsage on a GIN handle so the caller can
 * fall back to stock NCCL. ASYNCHRONOUS; results valid after stream sync. */
ncclResult_t ftAllToAll(FTCollectiveHandle* h,
                         ncclWindow_t win_send, size_t sendoffset,
                         ncclWindow_t win_recv, size_t recvoffset,
                         ncclWindow_t win_flags,
                         size_t count,
                         cudaStream_t stream,
                         uint64_t timeout_us);

/* Variable all-to-all (Phase 1, single-CTA).
 *
 *   d_send_counts : device array [nRanks], float32-element count sent to each
 *                   destination (chunk d goes to rank d).
 *   d_send_displs : device array [nRanks], float32-element offset of each chunk
 *                   in win_send.
 *   recv_stride_bytes : per-source scratch slot stride in bytes
 *                       (>= max_per_dest * 4); win_recv holds nRanks such slots,
 *                       slot s = data received from rank s.
 *
 * The caller compacts the per-source slots into its packed output using the
 * received counts/displacements (known locally). LSA only — returns
 * ncclInvalidUsage on a GIN handle. ASYNCHRONOUS. */
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
                          cudaStream_t stream, uint64_t timeout_us);

/* Variable all-gather (all_gatherv), auto-recv-sizing + PACKED output.
 *
 * Each rank contributes ONE variable-size chunk (d_send_counts[myRank] rows, at
 * win_send+sendoffset) which every rank receives, concatenated in rank order.
 * Count-in-flag: each rank broadcasts its chunk and raises a flag = count+1, so
 * the receiver DISCOVERS every source's size (no separate size all-gather),
 * prefix-sums them on-device, and compacts into `output` packed. Byte-oriented
 * (dtype-agnostic via row_bytes). LSA only; GIN returns ncclInvalidUsage.
 *
 *   win_recv/recvoffset : handle scratch, nRanks slots of recv_stride bytes.
 *   d_send_counts        : DEVICE int32[nRanks] (rows); this rank uses [myRank].
 *   output               : DEVICE ptr; packed concatenation (>= sum of counts).
 *   d_recv_counts_out    : DEVICE int32[nRanks] discovered rows per source (0 on
 *                          a timed-out source; consult the mask / error flag).
 *   recv_capacity        : max rows any one source may deliver (slot bound, rows).
 *   row_bytes            : bytes per row; recv_stride >= recv_capacity*row_bytes.
 *   max_send_count       : grid-size hint (rows), <= recv_capacity.
 * ASYNCHRONOUS on `stream`. */
ncclResult_t ftAllGatherv(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          const int* d_send_counts,
                          void* output,
                          int* d_recv_counts_out,
                          size_t recv_capacity, size_t row_bytes,
                          size_t recv_stride, size_t max_send_count,
                          cudaStream_t stream, uint64_t timeout_us);

/* Variable reduce-scatter (reduce_scatterv), typed reduction (SUM).
 *
 * Every rank supplies its full input vector (win_send+sendoffset, sum-of-counts
 * elements); the reduction is scattered so rank r receives shard
 * r = [displs[r], counts[r]) summed over all (surviving) contributors, into
 * `d_output` (counts[myRank] elements). Unlike all_gatherv the shard sizes are a
 * KNOWN partition — d_counts/d_displs are DEVICE int32[nRanks]/int64[nRanks]
 * inputs (d_displs=NULL derives them on-device). Survivor-sum: a timed-out
 * contributor drops out of the sum (consult the mask). LSA only.
 *
 *   win_recv/recvoffset : handle scratch, nRanks contributor slots of recv_stride
 *                         bytes (>= max_shard*sizeof(dtype), 16-byte multiple).
 *   recv_stride         : per-contributor slot stride in bytes.
 *   max_shard_count     : grid-size hint (elements), = max shard size.
 *   dtype               : FTDataType (float32/float16/bfloat16; float accum).
 * ASYNCHRONOUS on `stream`. */
ncclResult_t ftReduceScatterv(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          const int* d_counts, const size_t* d_displs,
                          void* d_output,
                          size_t recv_stride, size_t max_shard_count,
                          FTDataType dtype,
                          cudaStream_t stream, uint64_t timeout_us);

/* AllReduce with caller-supplied windows.
 *
 *   d_output  : non-symmetric device pointer (count elements of type dtype);
 *               written by kernel.  Cast to the appropriate pointer type
 *               based on dtype before passing.
 *   count     : number of elements (in dtype units, not bytes).
 *   dtype     : element type of send/recv buffers and d_output.
 *               Reduction always accumulates in float32 internally (Option-B),
 *               so bfloat16/float16 are supported without precision loss.
 *   win_flags : symmetric, nRanks * sizeof(uint64_t) — IGNORED for GIN kernel
 *
 * Results are read from d_output after cudaStreamSynchronize(stream). */
ncclResult_t ftAllReduce(FTCollectiveHandle* h,
                          ncclWindow_t win_send, size_t sendoffset,
                          ncclWindow_t win_recv, size_t recvoffset,
                          ncclWindow_t win_flags,
                          void* d_output,
                          size_t count,
                          FTDataType dtype,
                          cudaStream_t stream,
                          uint64_t timeout_us);

#ifdef __cplusplus
}
#endif
