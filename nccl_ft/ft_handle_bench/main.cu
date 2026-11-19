/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * FT AllGather Latency Benchmark
 *
 * Measures per-call GPU execution time of:
 *   - ftAllGatherLaunch  : context API (handle owns all symmetric windows)
 *   - ftAllGather        : window  API (caller supplies pre-registered ncclWindow_t)
 *   - ncclAllGather      : baseline
 *
 * Timing uses CUDA events recorded around each kernel launch.  Before each FT
 * launch the benchmark issues a 1-element ncclAllGather on the same stream as
 * a GPU-level barrier: all GPUs must commit their pre-barrier work (memsets,
 * d_ready reset) and complete the NCCL exchange before any FT kernel begins
 * polling d_ready on peers.  This mirrors the Python benchmark's dist.barrier()
 * and eliminates the readywin polling inflation caused by OS thread scheduling
 * skew after a CPU-only pthread_barrier_wait.  A CPU barrier (pthread_barrier_wait)
 * with a synchronous stream sync is still used to ensure zeroing completes before
 * the GPU barrier is enqueued.  Rare FT_TIMEOUT events (OS preemption between
 * the GPU barrier and kernel launch) are detected and skipped.
 * Results are reported as per-rank mean / min / max in microseconds.
 *
 * Usage:
 *   NTHREADS=4 ./ft_handle_bench [options]
 *
 * Options:
 *   --count N        Float32 elements per rank (default: 1024)
 *   --warmup N       Warm-up iterations (default: 20)
 *   --iters N        Timed iterations (default: 100)
 *   --timeout-us T   FT kernel timeout µs (default: 10000000)
 *   --csv FILE       Write per-rank results to a CSV file (optional)
 *
 * Environment:
 *   NTHREADS         Number of ranks / GPUs (default: all visible GPUs)
 */

#include "cuda_runtime.h"
#include "nccl.h"
#include "../ft_handle/ft_collective.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// Configuration
// =============================================================================

#define MAX_RANKS 1024

static size_t      g_count      = 1024;
static int         g_warmup     = 20;
static int         g_iters      = 100;
static uint64_t    g_timeout_us = 10000000;
static const char* g_csv_path   = NULL;

// =============================================================================
// Shared state
// =============================================================================

static pthread_barrier_t g_barrier;
static ncclUniqueId      g_nccl_id;
static ncclUniqueId      g_nccl_barrier_id;  /* second comm used only for GPU sync barriers */

static double g_ft_ctx_mean_us[MAX_RANKS];
static double g_ft_ctx_min_us[MAX_RANKS];
static double g_ft_ctx_max_us[MAX_RANKS];
static int    g_ft_ctx_skipped[MAX_RANKS];

static double g_ft_win_mean_us[MAX_RANKS];
static double g_ft_win_min_us[MAX_RANKS];
static double g_ft_win_max_us[MAX_RANKS];
static int    g_ft_win_skipped[MAX_RANKS];

static double g_nccl_mean_us[MAX_RANKS];
static double g_nccl_min_us[MAX_RANKS];
static double g_nccl_max_us[MAX_RANKS];

// =============================================================================
// Error checking macros
// =============================================================================

#define CUDACHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                              \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(e));             \
        exit(1);                                                        \
    }                                                                   \
} while (0)

#define NCCLCHECK(cmd) do {                                             \
    ncclResult_t r = (cmd);                                             \
    if (r != ncclSuccess) {                                             \
        fprintf(stderr, "NCCL error %s:%d: %s\n",                      \
                __FILE__, __LINE__, ncclGetErrorString(r));             \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// =============================================================================
// Barrier helper
// =============================================================================

static void host_barrier(void) {
    int err = pthread_barrier_wait(&g_barrier);
    if (err != 0 && err != PTHREAD_BARRIER_SERIAL_THREAD) {
        fprintf(stderr, "pthread_barrier_wait failed: %d\n", err);
        abort();
    }
}

// =============================================================================
// Thread function
// =============================================================================

typedef struct {
    int rank;
    int nranks;
} thread_args_t;

/* Zero d_ready synchronously then CPU-barrier so every peer sees d_ready==0
 * before any rank enqueues the next FT kernel.  Also zeros any in-stream
 * buffers that were accumulated beforehand (recv, flags) via streamSync. */
#define FT_PRE_SYNC(extra_async_zeroes) do {                                \
    do { extra_async_zeroes } while (0);                                    \
    if (ft_handle->d_ready) {                                               \
        CUDACHECK(cudaMemsetAsync(ft_handle->d_ready, 0,                    \
                                  sizeof(uint64_t), stream));               \
    }                                                                       \
    CUDACHECK(cudaStreamSynchronize(stream));                               \
    host_barrier();                                                         \
} while (0)

static void* bench_thread(void* raw) {
    thread_args_t* ta    = (thread_args_t*)raw;
    int            rank  = ta->rank;
    int            nranks = ta->nranks;

    CUDACHECK(cudaSetDevice(rank));

    // ---- NCCL communicator ----
    ncclUniqueId uid;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&uid));
    if (rank == 0) g_nccl_id = uid;
    host_barrier();   /* [uid-bcast] */
    uid = g_nccl_id;

    if (rank == 0) { printf("[init] ncclCommInitRank …\n"); fflush(stdout); }
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nranks, uid, rank));
    host_barrier();   /* [comm-init-done] */

    if (rank == 0) { printf("[init] ftHandleCreate …\n"); fflush(stdout); }

    /* Second communicator dedicated to GPU-level sync barriers.
     * Kept separate from comm so that user-level ncclAllGather calls cannot
     * corrupt the LSA barrier session counters that ftAllGather uses on comm. */
    ncclUniqueId barrier_uid;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&barrier_uid));
    if (rank == 0) g_nccl_barrier_id = barrier_uid;
    host_barrier();  /* [barrier-uid-bcast] */
    barrier_uid = g_nccl_barrier_id;

    ncclComm_t comm_barrier;
    NCCLCHECK(ncclCommInitRank(&comm_barrier, nranks, barrier_uid, rank));
    host_barrier();  /* [barrier-comm-init-done] */

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_stop;
    CUDACHECK(cudaEventCreate(&ev_start));
    CUDACHECK(cudaEventCreate(&ev_stop));

    // ---- FT handle ----
    bool* initial_mask = (bool*)malloc((size_t)nranks * sizeof(bool));
    for (int r = 0; r < nranks; r++) initial_mask[r] = true;

    FTCollectiveHandle* ft_handle = NULL;
    NCCLCHECK(ftHandleCreate(comm, nranks, initial_mask, &ft_handle));
    free(initial_mask);
    host_barrier();   /* [handle-done] */

    if (rank == 0) {
        const char* kt = (ft_handle->kernelType == FT_KERNEL_LSA) ? "LSA" : "GIN";
        printf("[init] kernel type: %s\n", kt);
        fflush(stdout);
    }

    // ---- Context API buffers ----
    if (rank == 0) { printf("[init] ftAllGatherCreate (context) …\n"); fflush(stdout); }
    FTAllGatherCtx* ft_ctx = NULL;
    NCCLCHECK(ftAllGatherCreate(ft_handle, g_count, &ft_ctx));

    float* h_send = (float*)malloc(g_count * sizeof(float));
    for (size_t i = 0; i < g_count; i++) h_send[i] = (float)rank;
    NCCLCHECK(ftAllGatherSetData(ft_ctx, h_send));

    // ---- Window API buffers ----
    if (rank == 0) { printf("[init] allocating window API buffers …\n"); fflush(stdout); }
    size_t send_bytes  = g_count * sizeof(float);
    /* The kernel pads each rank's recv slice to a 16-byte stride (see
     * FT_RECV_STRIDE in ft_collective.cu), so size the window recv buffer with
     * the padded stride — an unpadded nranks*count*4 buffer would be written
     * out of bounds for any count whose byte size is not a multiple of 16. */
    size_t chunk_stride = (g_count * sizeof(float) + 15) & ~((size_t)15);
    size_t recv_bytes  = (size_t)nranks * chunk_stride;
    size_t flags_bytes = (size_t)nranks * sizeof(uint64_t);

    void* d_win_send;
    void* d_win_recv;
    void* d_win_flags;
    NCCLCHECK(ncclMemAlloc(&d_win_send,  send_bytes));
    NCCLCHECK(ncclMemAlloc(&d_win_recv,  recv_bytes));
    NCCLCHECK(ncclMemAlloc(&d_win_flags, flags_bytes));

    /* Fill the window send buffer with the same data as the context send buf. */
    CUDACHECK(cudaMemcpy(d_win_send, h_send, send_bytes, cudaMemcpyHostToDevice));
    free(h_send);

    ncclWindow_t win_send, win_recv, win_flags;
    NCCLCHECK(ncclCommWindowRegister(comm, d_win_send,  send_bytes,  &win_send,  NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm, d_win_recv,  recv_bytes,  &win_recv,  NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm, d_win_flags, flags_bytes, &win_flags, NCCL_WIN_COLL_SYMMETRIC));

    // ---- NCCL buffers ----
    float* d_send_nccl;
    float* d_recv_nccl;
    CUDACHECK(cudaMalloc(&d_send_nccl, send_bytes));
    CUDACHECK(cudaMalloc(&d_recv_nccl, recv_bytes));
    CUDACHECK(cudaMemset(d_send_nccl, 0, send_bytes));

    host_barrier();  /* [setup-done] — all ranks ready before warmup */
    if (rank == 0) { printf("[init] done\n\n"); fflush(stdout); }

    // ====================================================================
    // Warm-up — Context API
    // ====================================================================

    if (rank == 0)
        printf("Warming up context API (%d iters) …\n", g_warmup);

    for (int i = 0; i < g_warmup; i++) {
        FT_PRE_SYNC(/* no extra zeroes */);
        /* GPU-level barrier: ensures all ranks' memsets are visible across NVLink
         * before any FT kernel starts polling d_ready. */
        NCCLCHECK(ncclAllGather(d_send_nccl, d_recv_nccl, 1, ncclFloat, comm_barrier, stream));
        NCCLCHECK(ftAllGatherLaunch(ft_ctx, stream, g_timeout_us));
        CUDACHECK(cudaStreamSynchronize(stream));
        ftHandleClearError(ft_handle);
    }

    host_barrier(); /* [ctx-warmup-done] */

    // ====================================================================
    // Timed — Context API
    // ====================================================================

    if (rank == 0)
        printf("Timing context API (%d iters) …\n", g_iters);

    double ctx_sum = 0.0, ctx_min = 1e18, ctx_max = 0.0;
    int    ctx_counted = 0, ctx_skipped = 0;

    for (int i = 0; i < g_iters; i++) {
        FT_PRE_SYNC(/* no extra zeroes */);
        /* GPU-level barrier: ev_start fires only after all ranks complete this
         * NCCL exchange, so all FT kernels start with d_ready already zeroed. */
        NCCLCHECK(ncclAllGather(d_send_nccl, d_recv_nccl, 1, ncclFloat, comm_barrier, stream));
        CUDACHECK(cudaEventRecord(ev_start, stream));
        NCCLCHECK(ftAllGatherLaunch(ft_ctx, stream, g_timeout_us));
        CUDACHECK(cudaEventRecord(ev_stop, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        if (ftHandleGetError(ft_handle) == FT_TIMEOUT) {
            ftHandleClearError(ft_handle);
            ctx_skipped++;
            continue;
        }
        ftHandleClearError(ft_handle);

        float ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double us = (double)ms * 1000.0;
        ctx_sum += us;
        if (us < ctx_min) ctx_min = us;
        if (us > ctx_max) ctx_max = us;
        ctx_counted++;
    }

    g_ft_ctx_mean_us[rank] = ctx_counted > 0 ? ctx_sum / (double)ctx_counted : 0.0;
    g_ft_ctx_min_us[rank]  = ctx_counted > 0 ? ctx_min : 0.0;
    g_ft_ctx_max_us[rank]  = ctx_max;
    g_ft_ctx_skipped[rank] = ctx_skipped;

    host_barrier(); /* [ctx-timed-done] */

    // ====================================================================
    // Warm-up — Window API
    // ====================================================================

    if (rank == 0)
        printf("Warming up window API (%d iters) …\n", g_warmup);

    for (int i = 0; i < g_warmup; i++) {
        /* Zero recv and flags in-stream, then zero d_ready + streamSync + barrier. */
        FT_PRE_SYNC(
            CUDACHECK(cudaMemsetAsync(d_win_recv,  0, recv_bytes,  stream));
            CUDACHECK(cudaMemsetAsync(d_win_flags, 0, flags_bytes, stream));
        );
        /* GPU-level barrier: ensures all ranks' memsets are visible before FT kernel. */
        NCCLCHECK(ncclAllGather(d_send_nccl, d_recv_nccl, 1, ncclFloat, comm_barrier, stream));
        NCCLCHECK(ftAllGather(ft_handle,
                              win_send,  0,
                              win_recv,  0,
                              win_flags,
                              g_count, stream, g_timeout_us));
        CUDACHECK(cudaStreamSynchronize(stream));
        ftHandleClearError(ft_handle);
    }

    host_barrier(); /* [win-warmup-done] */

    // ====================================================================
    // Timed — Window API
    // ====================================================================

    if (rank == 0)
        printf("Timing window API (%d iters) …\n", g_iters);

    double win_sum = 0.0, win_min = 1e18, win_max = 0.0;
    int    win_counted = 0, win_skipped = 0;

    for (int i = 0; i < g_iters; i++) {
        FT_PRE_SYNC(
            CUDACHECK(cudaMemsetAsync(d_win_recv,  0, recv_bytes,  stream));
            CUDACHECK(cudaMemsetAsync(d_win_flags, 0, flags_bytes, stream));
        );
        /* GPU-level barrier: ev_start fires only after all ranks complete this
         * NCCL exchange, so all FT kernels start with d_ready already zeroed. */
        NCCLCHECK(ncclAllGather(d_send_nccl, d_recv_nccl, 1, ncclFloat, comm_barrier, stream));
        CUDACHECK(cudaEventRecord(ev_start, stream));
        NCCLCHECK(ftAllGather(ft_handle,
                              win_send,  0,
                              win_recv,  0,
                              win_flags,
                              g_count, stream, g_timeout_us));
        CUDACHECK(cudaEventRecord(ev_stop, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        if (ftHandleGetError(ft_handle) == FT_TIMEOUT) {
            ftHandleClearError(ft_handle);
            win_skipped++;
            continue;
        }
        ftHandleClearError(ft_handle);

        float ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double us = (double)ms * 1000.0;
        win_sum += us;
        if (us < win_min) win_min = us;
        if (us > win_max) win_max = us;
        win_counted++;
    }

    g_ft_win_mean_us[rank] = win_counted > 0 ? win_sum / (double)win_counted : 0.0;
    g_ft_win_min_us[rank]  = win_counted > 0 ? win_min : 0.0;
    g_ft_win_max_us[rank]  = win_max;
    g_ft_win_skipped[rank] = win_skipped;

    host_barrier(); /* [win-timed-done] */

    // ====================================================================
    // Warm-up — NCCL baseline
    // ====================================================================

    if (rank == 0)
        printf("Warming up NCCL (%d iters) …\n", g_warmup);

    for (int i = 0; i < g_warmup; i++) {
        NCCLCHECK(ncclAllGather(d_send_nccl, d_recv_nccl, g_count,
                                ncclFloat, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    host_barrier(); /* [nccl-warmup-done] */

    // ====================================================================
    // Timed — NCCL baseline
    // ====================================================================

    if (rank == 0)
        printf("Timing NCCL (%d iters) …\n", g_iters);

    double nccl_sum = 0.0, nccl_min = 1e18, nccl_max = 0.0;

    for (int i = 0; i < g_iters; i++) {
        CUDACHECK(cudaEventRecord(ev_start, stream));
        NCCLCHECK(ncclAllGather(d_send_nccl, d_recv_nccl, g_count,
                                ncclFloat, comm, stream));
        CUDACHECK(cudaEventRecord(ev_stop, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        float ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double us = (double)ms * 1000.0;
        nccl_sum += us;
        if (us < nccl_min) nccl_min = us;
        if (us > nccl_max) nccl_max = us;
    }

    g_nccl_mean_us[rank] = nccl_sum / (double)g_iters;
    g_nccl_min_us[rank]  = nccl_min;
    g_nccl_max_us[rank]  = nccl_max;

    host_barrier(); /* [nccl-timed-done] */

    // ---- Cleanup ----
    ftAllGatherDestroy(ft_ctx);

    NCCLCHECK(ncclCommWindowDeregister(comm, win_send));
    NCCLCHECK(ncclCommWindowDeregister(comm, win_recv));
    NCCLCHECK(ncclCommWindowDeregister(comm, win_flags));
    NCCLCHECK(ncclMemFree(d_win_send));
    NCCLCHECK(ncclMemFree(d_win_recv));
    NCCLCHECK(ncclMemFree(d_win_flags));

    ftHandleDestroy(ft_handle);

    CUDACHECK(cudaFree(d_send_nccl));
    CUDACHECK(cudaFree(d_recv_nccl));
    CUDACHECK(cudaEventDestroy(ev_start));
    CUDACHECK(cudaEventDestroy(ev_stop));
    CUDACHECK(cudaStreamDestroy(stream));

    NCCLCHECK(ncclCommFinalize(comm_barrier));
    NCCLCHECK(ncclCommDestroy(comm_barrier));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));

    return NULL;
}

#undef FT_PRE_SYNC

// =============================================================================
// Argument parsing
// =============================================================================

static void parse_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            g_count = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            g_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            g_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--timeout-us") == 0 && i + 1 < argc) {
            g_timeout_us = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            g_csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: ft_handle_bench [options]\n");
            printf("  --count N        Float32 elements per rank (default: 1024)\n");
            printf("  --warmup N       Warm-up iterations (default: 20)\n");
            printf("  --iters N        Timed iterations (default: 100)\n");
            printf("  --timeout-us T   FT kernel timeout µs (default: 10000000)\n");
            printf("  --csv FILE       Write per-rank results to a CSV file\n");
            printf("Environment:\n");
            printf("  NTHREADS         Number of ranks / GPUs (default: all visible)\n");
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s  (use --help)\n", argv[i]);
            exit(1);
        }
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    parse_args(argc, argv);

    int num_gpus = 0;
    CUDACHECK(cudaGetDeviceCount(&num_gpus));

    int nranks = num_gpus;
    const char* env = getenv("NTHREADS");
    if (env) nranks = atoi(env);

    if (nranks < 2) {
        fprintf(stderr, "Need at least 2 ranks. Set NTHREADS >= 2 or ensure multiple GPUs are visible.\n");
        return 1;
    }
    if (nranks > num_gpus) {
        fprintf(stderr, "NTHREADS=%d exceeds available GPUs (%d)\n", nranks, num_gpus);
        return 1;
    }
    if (nranks > MAX_RANKS) {
        fprintf(stderr, "NTHREADS=%d exceeds MAX_RANKS=%d\n", nranks, MAX_RANKS);
        return 1;
    }

    printf("FT AllGather Latency Benchmark\n");
    printf("==============================\n");
    printf("Ranks       : %d\n", nranks);
    printf("Count       : %zu floats/rank (%.1f KB send)\n",
           g_count, (double)(g_count * sizeof(float)) / 1024.0);
    printf("Warm-up     : %d\n", g_warmup);
    printf("Iterations  : %d\n", g_iters);
    printf("FT timeout  : %llu µs\n", (unsigned long long)g_timeout_us);
    printf("\n");

    pthread_barrier_init(&g_barrier, NULL, (unsigned)nranks);

    thread_args_t* args = (thread_args_t*)malloc((size_t)nranks * sizeof(thread_args_t));
    pthread_t*     tids = (pthread_t*)malloc((size_t)nranks * sizeof(pthread_t));

    for (int r = 0; r < nranks; r++) {
        args[r].rank   = r;
        args[r].nranks = nranks;
        pthread_create(&tids[r], NULL, bench_thread, &args[r]);
    }

    for (int r = 0; r < nranks; r++)
        pthread_join(tids[r], NULL);

    pthread_barrier_destroy(&g_barrier);
    free(args);
    free(tids);

    // ---- Print results ----
    printf("\n");
    printf("%-6s  %10s %10s %10s  %10s %10s %10s  %10s %10s %10s  %7s %7s\n",
           "rank",
           "ctx_mean", "ctx_min", "ctx_max",
           "win_mean", "win_min", "win_max",
           "nccl_mean", "nccl_min", "nccl_max",
           "ctx_skip", "win_skip");
    printf("%-6s  %10s %10s %10s  %10s %10s %10s  %10s %10s %10s  %7s %7s\n",
           "----",
           "(µs)", "(µs)", "(µs)",
           "(µs)", "(µs)", "(µs)",
           "(µs)", "(µs)", "(µs)",
           "", "");

    double sum_ctx = 0.0, sum_win = 0.0, sum_nccl = 0.0;
    for (int r = 0; r < nranks; r++) {
        printf("%-6d  %10.1f %10.1f %10.1f  %10.1f %10.1f %10.1f  %10.1f %10.1f %10.1f  %7d %7d\n",
               r,
               g_ft_ctx_mean_us[r], g_ft_ctx_min_us[r], g_ft_ctx_max_us[r],
               g_ft_win_mean_us[r], g_ft_win_min_us[r], g_ft_win_max_us[r],
               g_nccl_mean_us[r],   g_nccl_min_us[r],   g_nccl_max_us[r],
               g_ft_ctx_skipped[r], g_ft_win_skipped[r]);
        sum_ctx  += g_ft_ctx_mean_us[r];
        sum_win  += g_ft_win_mean_us[r];
        sum_nccl += g_nccl_mean_us[r];
    }

    printf("\n");
    printf("mean across ranks:  ctx=%.1f µs  win=%.1f µs  nccl=%.1f µs"
           "  (ctx overhead=%.1f µs  win overhead=%.1f µs)\n",
           sum_ctx  / nranks,
           sum_win  / nranks,
           sum_nccl / nranks,
           (sum_ctx  - sum_nccl) / nranks,
           (sum_win  - sum_nccl) / nranks);

    if (g_csv_path) {
        FILE* f = fopen(g_csv_path, "w");
        if (!f) {
            fprintf(stderr, "Cannot open CSV file: %s\n", g_csv_path);
            return 1;
        }
        fprintf(f, "nranks,count,rank,"
                   "ctx_mean_us,ctx_min_us,ctx_max_us,ctx_skipped,"
                   "win_mean_us,win_min_us,win_max_us,win_skipped,"
                   "nccl_mean_us,nccl_min_us,nccl_max_us,"
                   "ctx_overhead_us,win_overhead_us\n");
        for (int r = 0; r < nranks; r++) {
            fprintf(f, "%d,%zu,%d,"
                       "%.3f,%.3f,%.3f,%d,"
                       "%.3f,%.3f,%.3f,%d,"
                       "%.3f,%.3f,%.3f,"
                       "%.3f,%.3f\n",
                    nranks, g_count, r,
                    g_ft_ctx_mean_us[r], g_ft_ctx_min_us[r], g_ft_ctx_max_us[r], g_ft_ctx_skipped[r],
                    g_ft_win_mean_us[r], g_ft_win_min_us[r], g_ft_win_max_us[r], g_ft_win_skipped[r],
                    g_nccl_mean_us[r],   g_nccl_min_us[r],   g_nccl_max_us[r],
                    g_ft_ctx_mean_us[r] - g_nccl_mean_us[r],
                    g_ft_win_mean_us[r] - g_nccl_mean_us[r]);
        }
        fclose(f);
        printf("Results written to: %s\n", g_csv_path);
    }

    return 0;
}
