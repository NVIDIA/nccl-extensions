/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "../device_primitives.cuh"
#include "nccl_device.h"
#include "ll_gin.cuh"
#include "ll_lsa_primitives.cuh"

namespace nccl_ep {

namespace ll {

// Mask convention: 1 = active, 0 = masked/failed. nullptr means masking disabled.
template <bool useWarpSync = false>
__forceinline__ __device__ bool isRankMasked(int* rankMask, int rank) {
    if (rankMask == nullptr) {
        return false;
    }
    if constexpr (useWarpSync) {
        return __shfl_sync(0xffffffff, ld_acquire_global(rankMask + rank), 0) == 0;
    } else {
        return ld_acquire_global(rankMask + rank) == 0;
    }
}

// CTA-local (not grid-global) rank-active bitmask: 1 = this CTA's own poll
// confirmed the rank ready. Avoids the cross-CTA race isRankMasked's
// grid-global rankMask has -- see waitForAllRanksFinishLsa in
// ll_combine_lsa.cuh.
__forceinline__ __device__ bool isRankActiveLocal(const uint32_t* localMask, int rank) {
    return (localMask[rank / 32] & (1u << (rank % 32))) != 0;
}

// ============================================================================
// clean_low_latency_buffer: barrier → zero RDMA buffers → barrier
// Hybrid barrier: NVLink peers use P2P stores, RDMA peers use GIN signals.
// ============================================================================

template <int kNumThreads>
__forceinline__ __device__ void maskAwareBarrier(
    int threadId,
    int myRank,
    int* rankMask,
    int* syncBuffer,
    const ncclWindow_t* syncWindow,
    unsigned barrierSignalBase,
    ncclDevComm* devComm,
    uint64_t timeoutCycles) {
    if (isRankMasked(rankMask, myRank)) return;

    int nRanks = devComm->nRanks;
    EP_DEVICE_ASSERT(kNumThreads >= nRanks);

    // Decrement local sync counter (monotonically decreasing)
    if (threadId == 0) atomicAdd(syncBuffer + myRank, -1);
    __syncthreads();

    int cnt = syncBuffer[myRank];

    // syncBuffer is the entirety of its window — peer-slot offsets are
    // relative to the window base, not nested inside a larger region.
    auto syncWin = syncWindow[0];

    // Publish our counter to each active peer and wait for theirs.
    // NVLink (P2P) peers: direct store + load on the shared syncBuffer.
    // RDMA peers: GIN signal (0-byte put + SignalAdd) instead of putValue.
    if (threadId < nRanks && threadId != myRank) {
        int peer = threadId;
        if (!isRankMasked(rankMask, peer)) {
            size_t peerSlotOffset = myRank * sizeof(int);
            auto p2pPtr = ncclGetP2pPtr(
                reinterpret_cast<uint64_t>(syncBuffer),
                peerSlotOffset,
                myRank,
                peer,
                syncWindow,
                devComm);

            if (p2pPtr == 0) {
                // RDMA peer: use GIN signal
                sendGinBarrierSignal(peer, syncWin, peerSlotOffset, barrierSignalBase, devComm);
            } else {
                // NVLink peer: direct P2P store
                st_release_sys_global(reinterpret_cast<int*>(p2pPtr), cnt);
            }
        }
    }
    __syncthreads();

    // Wait for all active peers
    if (threadId < nRanks && threadId != myRank) {
        int peer = threadId;
        if (!isRankMasked(rankMask, peer)) {
            size_t peerSlotOffset = peer * sizeof(int);
            auto p2pPtr = ncclGetP2pPtr(
                reinterpret_cast<uint64_t>(syncBuffer),
                peerSlotOffset,
                myRank,
                peer,
                syncWindow,
                devComm);

            if (p2pPtr != 0) {
                // NVLink peer: poll syncBuffer directly
                auto startTime = clock64();
                uint64_t elapsed = 0;
                while (ld_acquire_sys_global(syncBuffer + peer) != cnt &&
                       (elapsed = clock64() - startTime) <= timeoutCycles);
                if (elapsed > timeoutCycles) {
                    printf("Warning: NCCL EP clean barrier timeout (P2P), myRank %d, peer %d\n", myRank, peer);
                    atomicExch(rankMask + peer, 0);
                }
            }
        }
    }

    // RDMA peers: wait on GIN signal (thread 0 handles aggregate)
    if (threadId == 0) {
        auto ctxId = myRank % MAX_NCCL_GIN_CTX_PER_COMM;

        int numExpectedSignals = 0;
        for (int r = 0; r < nRanks; r++) {
            if (r == myRank || isRankMasked(rankMask, r)) continue;
            auto p2p = ncclGetP2pPtr(0x01, 0, myRank, r, syncWindow, devComm);
            if (p2p == 0) numExpectedSignals++;
        }

        if (numExpectedSignals > 0) {
            auto startTime = clock64();
            uint64_t elapsed = 0;
            waitGinSignal(
                devComm, ctxId, barrierSignalBase, static_cast<uint64_t>(numExpectedSignals), startTime,
                timeoutCycles, &elapsed);

            if (elapsed > timeoutCycles) {
                printf("Warning: NCCL EP clean barrier timeout (GIN), myRank %d\n", myRank);
                for (int r = 0; r < nRanks; r++) {
                    if (r == myRank || isRankMasked(rankMask, r)) continue;
                    auto p2p = ncclGetP2pPtr(0x01, 0, myRank, r, syncWindow, devComm);
                    if (p2p == 0) atomicExch(rankMask + r, 0);
                }
            }
        }
    }
    __syncthreads();
}

template <int kNumThreads>
__device__ __forceinline__ void clean_low_latency_buffer_kernel_impl(
    int* clean_0,
    int num_clean_int_0,
    int* clean_1,
    int num_clean_int_1,
    int* rankMask,
    int* syncBuffer,
    ncclWindow_t* syncWindow,
    ncclDevComm* devComm,
    unsigned barrierSignalBase,
    uint64_t timeoutCycles) {
    int threadId = static_cast<int>(threadIdx.x);

    // Pre-clean barrier
    if (rankMask == nullptr) {
        ncclGin net(*devComm, 0);
        ncclGinBarrierSession<ncclCoopCta> bar(ncclCoopCta(), net, ncclTeamTagWorld(), blockIdx.x);
        bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    } else {
        maskAwareBarrier<kNumThreads>(
            threadId,
            devComm->rank,
            rankMask,
            syncBuffer,
            syncWindow,
            barrierSignalBase,
            devComm,
            timeoutCycles);
    }

    // Zero out RDMA buffers
    for (int i = threadId; i < num_clean_int_0; i += kNumThreads) clean_0[i] = 0;
    for (int i = threadId; i < num_clean_int_1; i += kNumThreads) clean_1[i] = 0;
    __threadfence_system();

    // Post-clean barrier
    if (rankMask == nullptr) {
        ncclGin net(*devComm, 0);
        ncclGinBarrierSession<ncclCoopCta> bar(ncclCoopCta(), net, ncclTeamTagWorld(), blockIdx.x);
        bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    } else {
        maskAwareBarrier<kNumThreads>(
            threadId,
            devComm->rank,
            rankMask,
            syncBuffer,
            syncWindow,
            barrierSignalBase,
            devComm,
            timeoutCycles);
    }
}

} // namespace ll

} // namespace nccl_ep
