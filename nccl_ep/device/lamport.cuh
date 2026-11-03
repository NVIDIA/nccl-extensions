/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
#pragma once

// Lamport-style arrival detection for data pushed by a peer: every landing word is pre-filled with a sentinel the
// payload never contains, a 16-byte vector has arrived when none of its four words is the sentinel, and the consumer
// re-arms the vector for the next round. The sender encodes the payload so that a payload word is never the sentinel
// (sanitize_*).
//
// Host-includable (nccl_ep.cc takes the sentinel byte from here): it uses no device primitives, and the polling loop,
// which needs the device clock, is compiled for nvcc only. Used by the LL Lamport combine
// (device/ll/ll_combine_lsa.cuh, docs/plc/401-ll-combine-lamport).

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>

namespace nccl_ep::lamport {

// Landing-slot word value while unwritten.
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
// The host fills the whole landing region with this byte, so that every word reads as kSentinel.
constexpr int kSentinelByte = 0xFF;
constexpr int kVecBytes = static_cast<int>(sizeof(uint4));

// Replacement for a payload word equal to the sentinel: the canonical NaN of the token width (two 16-bit lanes, or one
// 32-bit lane), so only NaN patterns change.
template <ncclDataType_t kTokenDtype>
__host__ __device__ __forceinline__ constexpr uint32_t sanitized_word() {
    return kTokenDtype == ncclFloat32 ? 0x7FFFFFFFu : 0x7FFF7FFFu;
}

// Byte offset of the landing segment of combine round `round`: two segments alternate, so a round's receivers and the
// next round's senders never touch the same segment (the round words keep the round after next out).
__host__ __device__ __forceinline__ size_t segment_offset(size_t base, size_t stride, unsigned int round) {
    return base + static_cast<size_t>(round & 1U) * stride;
}

// True once a round word (the number of rounds a peer has started) has reached `round`. The round counter and the
// words are 32-bit and wrap, so the test is on the difference modulo 2^32: it holds while the two stay within 2^31 of
// each other, which the protocol keeps (a sender is at most one round ahead of the peer it writes to).
__host__ __device__ __forceinline__ constexpr bool round_reached(unsigned int word, unsigned int round) {
    return static_cast<int>(word - round) >= 0;
}

// Rewrite the sentinel words of one vector. Branch-free: one compare and one select per word.
template <uint32_t kSanitizedWord>
__device__ __forceinline__ void sanitize_vec(uint4& v) {
    v.x = (v.x == kSentinel) ? kSanitizedWord : v.x;
    v.y = (v.y == kSentinel) ? kSanitizedWord : v.y;
    v.z = (v.z == kSentinel) ? kSanitizedWord : v.z;
    v.w = (v.w == kSentinel) ? kSanitizedWord : v.w;
}

// Warp-collective: rewrite the sentinel words of a shared-memory stage of 32 * kVecsPerLane vectors. Lane l owns
// vectors l + 32 * k. The rewrite is plain generic-proxy writes: when an async-proxy (TMA) store reads the stage next,
// the caller orders them first (tma_store_fence, then a warp reconverge).
template <int kVecsPerLane, uint32_t kSanitizedWord>
__device__ __forceinline__ void sanitize_stage(uint4* stage, int laneId) {
#pragma unroll
    for (int k = 0; k < kVecsPerLane; ++k) {
        uint4 v = stage[laneId + 32 * k];
        sanitize_vec<kSanitizedWord>(v);
        stage[laneId + 32 * k] = v;
    }
}

// The vector a consumed landing slot is re-armed with (the consumer is its only reader; the caller stores it).
__host__ __device__ __forceinline__ uint4 sentinel_vec() {
    return make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
}

// Arrived when none of the four words is the sentinel (a partially delivered vector still contains one).
__device__ __forceinline__ bool arrived(const uint4& v) {
    return v.x != kSentinel && v.y != kSentinel && v.z != kSentinel && v.w != kSentinel;
}

// Block-wide start-over of the combine state: re-poison the landing region with the sentinel, zero the per-peer round
// words and restart the round counter. Sizes and addresses are multiples of 16 bytes.
template <int kNumThreads>
__device__ __forceinline__ void reset_state(
    uint4* poison,
    size_t poisonBytes,
    uint4* words,
    size_t wordBytes,
    unsigned int* round,
    int threadId) {
    const uint4 sentinel = sentinel_vec();
    for (size_t i = threadId; i < poisonBytes / sizeof(uint4); i += kNumThreads) {
        poison[i] = sentinel;
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (size_t i = threadId; i < wordBytes / sizeof(uint4); i += kNumThreads) {
        words[i] = zero;
    }
    if (threadId == 0) {
        *round = 0;
    }
}

#if defined(__CUDACC__)
// One vector per thread: poll slot j (for every j in `need`) until all have arrived or timeoutCycles elapsed. load(j)
// reads the current 16 bytes of slot j (a volatile or acquire load: the data is written by a peer). The arrived vectors
// are returned in buf[j]; the result is the mask of slots still pending (0 when all arrived). Does not re-arm.
template <int kGroup, typename LoadFn>
__device__ __forceinline__ unsigned poll_vec_per_thread(
    unsigned need,
    uint64_t timeoutCycles,
    uint4 (&buf)[kGroup],
    LoadFn&& load) {
    const auto startTime = clock64();
    int it = 0;
    while (need) {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
            if ((need >> j) & 1u) {
                buf[j] = load(j);
            }
        }
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
            if (((need >> j) & 1u) && arrived(buf[j])) {
                need &= ~(1u << j);
            }
        }
        if (need && (++it & 63) == 0 && static_cast<uint64_t>(clock64() - startTime) > timeoutCycles) {
            break;
        }
    }
    return need;
}
#endif // __CUDACC__

} // namespace nccl_ep::lamport
