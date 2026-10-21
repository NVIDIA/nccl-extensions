/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "nccl_ep.h"
#include "nccl_device.h"

namespace nccl_ep {

namespace ll {

// Flatten a source rank's token slot into the rank-major receive output.
__forceinline__ __device__ size_t getLsaRecvSlot(int srcRank, int maxTokensPerRank, int slotIdx) {
    return static_cast<size_t>(srcRank) * maxTokensPerRank + slotIdx;
}

// Resolve a token's peer output address, retaining staging when no window is registered.
// The caller must ensure the destination is LSA-reachable and zero-copy is eligible.
template <typename T>
__forceinline__ __device__ T* resolveLsaRecvPointer(
    T* stagingPtr,
    ncclWindow_t window,
    size_t windowOffset,
    size_t recvSlot,
    size_t bytesPerToken,
    int dstRank) {
    if (window != ncclWindow_t{}) {
        return static_cast<T*>(ncclGetPeerPointer(window, windowOffset + recvSlot * bytesPerToken, dstRank));
    }
    return stagingPtr;
}

// Resolves `dstRank`'s peer pointer for the given local window offset, but
// only when `dstRank` is reachable over NVLink (same LSA team as `rank`).
// Returns 0 for same-LSA peers whose window mapping failed and for any
// cross-LSA peer, so the caller can fall back to RDMA.
__forceinline__ __device__ uint64_t ncclGetP2pPtr(
    const uint64_t& dstPtr,
    const size_t& offset,
    const int& rank,
    const int& dstRank,
    const ncclWindow_t* ncclWindows,
    ncclDevComm* devComm) {
    // Local rank, no need for peer mapping
    if (rank == dstRank) {
        return dstPtr;
    }

    // P2P/NVLink only works between ranks on the same LSA team
    // Use NCCL team APIs to check if dstRank is in the same LSA team.
    ncclTeam lsa = ncclTeamLsa(*devComm);
    ncclTeam world = ncclTeamWorld(*devComm);
    if (!ncclTeamRankIsMember(lsa, world, dstRank)) return 0; // Different LSA teams, must use RDMA

    // The window array is indexed per-comm; the 1-comm N-context design pins that to 0.
    constexpr int commId = 0;
    auto const p2pPtr = reinterpret_cast<uint64_t>(ncclGetPeerPointer(ncclWindows[commId], offset, dstRank));

    return p2pPtr ? p2pPtr : 0;
}

} // namespace ll

} // namespace nccl_ep
