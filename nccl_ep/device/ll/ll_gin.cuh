/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "../device_primitives.cuh"
#include "nccl_device.h"

namespace nccl_ep {

namespace ll {

// A launch's `hashKey` packs both the GIN comm and the ctx within that comm
// (1-comm N-context design: MAX_NCCL_GIN_CTX_PER_COMM contexts per devComm).
__device__ __forceinline__ int getCommId(int hashKey) {
    return hashKey / MAX_NCCL_GIN_CTX_PER_COMM;
}

__device__ __forceinline__ int getCtxId(int hashKey) {
    return hashKey % MAX_NCCL_GIN_CTX_PER_COMM;
}

// Sends `bytes` from (window, srcOffset) to (window, dstOffset) on `dstRank`
// via GIN, carrying `signal` (ncclGin_None{} for a plain, unsignalled put).
// Every LL put site uses the same window for both ends and never sets a
// completion counter, so those are fixed here rather than threaded through.
// `ctxId` selects the GIN context within comm 0 (1-comm N-context design).
template <typename SignalT>
__forceinline__ __device__ void ginPut(
    ncclDevComm* devComm,
    int ctxId,
    int dstRank,
    ncclWindow_t window,
    size_t dstOffset,
    size_t srcOffset,
    size_t bytes,
    SignalT signal) {
    ncclGin net(*devComm, ctxId);
    ncclTeam world = ncclTeamWorld(*devComm);
    net.put(
        world,
        dstRank,
        window,
        dstOffset,
        window,
        srcOffset,
        bytes,
        signal,
        ncclGin_None{}, // no counter
        ncclCoopThread());
}

// Polls a GIN signal until it reaches `threshold` or `timeoutCycles` has
// elapsed since `startTime`, then resets it. Returns the last-read value
// (>= threshold on success) and writes the elapsed cycle count to
// `*elapsedCycles` so the caller can distinguish success from timeout.
__forceinline__ __device__ uint64_t waitGinSignal(
    ncclDevComm* devComm,
    int ctxId,
    ncclGinSignal_t signalId,
    uint64_t threshold,
    uint64_t startTime,
    uint64_t timeoutCycles,
    uint64_t* elapsedCycles) {
    ncclGin net(*devComm, ctxId);
    uint64_t curValue;
    do {
        curValue = net.readSignal(signalId);
    } while (curValue < threshold && (*elapsedCycles = clock64() - startTime) <= timeoutCycles);
    net.resetSignal(signalId);
    return curValue;
}

// RDMA (cross-LSA) peer notify: a 0-byte put carrying a SignalAdd, used by
// the hybrid barrier to signal a peer that isn't reachable over NVLink.
__forceinline__ __device__ void sendGinBarrierSignal(
    int peer,
    ncclWindow_t syncWin,
    size_t peerSlotOffset,
    unsigned barrierSignalBase,
    ncclDevComm* devComm) {
    auto ctxId = peer % MAX_NCCL_GIN_CTX_PER_COMM;
    ginPut(devComm, ctxId, peer, syncWin, peerSlotOffset, /*srcOffset=*/0, /*bytes=*/0,
           ncclGin_SignalAdd{barrierSignalBase, 1});
}

} // namespace ll

} // namespace nccl_ep
