/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more information.
 *
 * Single-process coverage of the Lamport combine's building blocks: the landing-region layout, the double-buffering
 * state, the sentinel encoding for BF16, FP16 and FP32, the arrival test, the re-arm and the segment selection.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "common.hpp"
#include "device/lamport.cuh"

namespace {

using nccl_ep::LamportEpochState;
using nccl_ep::LamportRegion;
using nccl_ep::LowLatencyLayout;

static_assert(std::is_trivially_copyable_v<LamportEpochState>, "the state is memset and read on the device");

TEST(LamportState, StartsAtRoundZero) {
    LamportEpochState state;
    EXPECT_EQ(state.round, 0u);
}

// The gate compares round words with the round counter as a difference modulo 2^32, so the counter may wrap: a peer
// that has started round r - 1 has published r (wrapped to 0 for r == 2^32), and a peer still in round r - 2 has not.
TEST(LamportRound, ReachedIsWrapSafe) {
    using nccl_ep::lamport::round_reached;
    for (unsigned int round : {0u, 1u, 2u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFEu, 0xFFFFFFFFu}) {
        for (int d = -3; d <= 3; ++d) {
            const unsigned int word = round + static_cast<unsigned int>(d);
            EXPECT_EQ(round_reached(word, round), d >= 0) << "round " << round << " word " << word;
        }
    }
    // Start of life: the words are zero, so round 0 passes and round 1 waits for the peer to start round 0.
    EXPECT_TRUE(round_reached(0u, 0u));
    EXPECT_FALSE(round_reached(0u, 1u));
    // Round 2^32 - 1: a peer in the same round has published 0 (wrapped); a peer in round 2^32 - 3 has published
    // 2^32 - 2 and must still be waited for.
    EXPECT_TRUE(round_reached(0u, 0xFFFFFFFFu));
    EXPECT_FALSE(round_reached(0xFFFFFFFEu, 0xFFFFFFFFu));
    // Round 2^32 (counter back at 0): a peer in round 2^32 - 2 has published 2^32 - 1 and must still be waited for.
    EXPECT_FALSE(round_reached(0xFFFFFFFFu, 0u));
    EXPECT_TRUE(round_reached(0u, 0u));
}

constexpr int kMaxTokens = 16;
constexpr size_t kMaxTokenBytes = 4096 * 2;

LamportRegion shape(int ranks, int topk) {
    return nccl_ep::lamport_region_shape(kMaxTokens, kMaxTokenBytes, ranks, topk);
}

TEST(LamportLayout, NoCapacityNoRegion) {
    EXPECT_FALSE(shape(8, 0).valid());
    EXPECT_TRUE(shape(8, 1).valid());
}

TEST(LamportLayout, SlotSizeIsTheLayoutCombineMessageSize) {
    // The region's receive slot must be the slot the handles' layouts address.
    const LowLatencyLayout layout(kMaxTokens, kMaxTokenBytes, 8, 64, 8, NCCL_EP_LAYOUT_RANK_MAJOR);
    EXPECT_EQ(shape(8, 8).slot_bytes, static_cast<size_t>(layout.buffer.num_bytes_per_combine_msg));
}

TEST(LamportLayout, RoundWordsComeFirstAndSegmentsFollowAtACapacityStride) {
    for (int topk : {1, 8, 16}) {
        for (int ranks : {2, 8, 64}) {
            const LamportRegion r = shape(ranks, topk);
            SCOPED_TRACE(::testing::Message() << "topk=" << topk << " ranks=" << ranks);
            EXPECT_EQ(r.capacity_topk, topk);
            EXPECT_EQ(r.segment_stride % 256, 0u);
            // One segment holds the receive slots of a full batch at this top-k.
            EXPECT_GE(r.segment_stride, static_cast<size_t>(kMaxTokens) * r.slot_bytes * topk);
            // The round words (one int per peer) sit at the region start, ahead of the two segments.
            EXPECT_GE(r.recv_offset, static_cast<size_t>(ranks) * sizeof(int));
            EXPECT_EQ(r.bytes, r.recv_offset + 2 * r.segment_stride);
        }
    }
}

TEST(LamportLayout, TheRoundWordsStayPutWhenTheCapacityGrows) {
    // Placed at the tail of buffers of the same size, the words are at the same distance from the end only if the
    // region is the same size; what must hold is that they are at the region start, before any segment.
    const size_t alloc = 1ull << 30;
    const LamportRegion small = nccl_ep::lamport_region_at_tail(shape(8, 2), alloc);
    const LamportRegion large = nccl_ep::lamport_region_at_tail(shape(8, 8), alloc);
    EXPECT_EQ(small.round_offset + small.bytes, alloc);
    EXPECT_EQ(large.round_offset + large.bytes, alloc);
    EXPECT_LT(large.round_offset, small.round_offset);                  // a larger region starts earlier
    EXPECT_EQ(small.recv_offset - small.round_offset, large.recv_offset - large.round_offset); // same word span
    EXPECT_EQ(small.round_offset % 256, 0u);
    EXPECT_EQ(large.recv_offset % 256, 0u);
}

TEST(LamportLayout, TheRegionNeverReachesAHandleLayout) {
    // A handle needs its layout bytes plus the region, so a buffer sized for that places the region past the layout.
    for (int topk : {1, 8, 16}) {
        const LowLatencyLayout layout(kMaxTokens, kMaxTokenBytes, 8, 64, topk, NCCL_EP_LAYOUT_RANK_MAJOR);
        const LamportRegion r = shape(8, topk);
        const size_t alloc = ((layout.total_bytes + r.reserved_bytes + 1023) / 1024) * 1024;
        EXPECT_GE(nccl_ep::lamport_region_at_tail(r, alloc).round_offset, layout.total_bytes) << "topk=" << topk;
    }
}

TEST(LamportLayout, TheRegionStartIsAlignedWhateverTheBufferSize) {
    // A user-specified buffer size is not aligned. The region still starts at a multiple of kAlign, so the round words
    // and both segments are 16-byte aligned, and it fits behind a layout when the buffer has reserved_bytes to spare.
    const LowLatencyLayout layout(kMaxTokens, kMaxTokenBytes, 8, 64, 8, NCCL_EP_LAYOUT_RANK_MAJOR);
    const LamportRegion r = shape(8, 8);
    for (size_t extra : {0, 1, 15, 16, 17, 127, 128, 129, 255, 256}) {
        const size_t alloc = layout.total_bytes + r.reserved_bytes + extra;
        const LamportRegion placed = nccl_ep::lamport_region_at_tail(r, alloc);
        SCOPED_TRACE(::testing::Message() << "extra=" << extra);
        EXPECT_EQ(placed.round_offset % LamportRegion::kAlign, 0u);
        EXPECT_EQ(placed.recv_offset % LamportRegion::kAlign, 0u);
        EXPECT_GE(placed.round_offset, layout.total_bytes);
        EXPECT_LE(placed.round_offset + placed.bytes, alloc);
    }
}

TEST(LamportLayout, ASmallerTopkFitsInALargerCapacity) {
    // A handle with top-k 2 addresses slot (token * 2 + j) inside a segment sized for top-k 8.
    const LamportRegion r = shape(8, 8);
    const size_t last_slot_end = (static_cast<size_t>(kMaxTokens - 1) * 2 + 1 + 1) * r.slot_bytes;
    EXPECT_LE(last_slot_end, r.segment_stride);
}

// ---- device helpers ----

constexpr uint32_t kSentinel = nccl_ep::lamport::kSentinel;

template <ncclDataType_t kDtype>
__global__ void sanitize_kernel(const uint4* in, uint4* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint4 v = in[i];
    nccl_ep::lamport::sanitize_vec<nccl_ep::lamport::sanitized_word<kDtype>()>(v);
    out[i] = v;
}

__global__ void arrived_kernel(const uint4* in, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nccl_ep::lamport::arrived(in[i]) ? 1 : 0;
}

__global__ void rearm_kernel(uint4* p) {
    *p = nccl_ep::lamport::sentinel_vec();
}

__global__ void segment_kernel(size_t base, size_t stride, const unsigned* rounds, size_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nccl_ep::lamport::segment_offset(base, stride, rounds[i]);
}

class LamportDeviceTest : public ::testing::Test {
public:
    void SetUp() override {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        if (count == 0) GTEST_SKIP() << "No CUDA device available";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    template <typename T>
    T* upload(const std::vector<T>& host) {
        T* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, host.size() * sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    }

    template <typename T>
    std::vector<T> download(const T* d, size_t n) {
        std::vector<T> host(n);
        EXPECT_EQ(cudaMemcpy(host.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        return host;
    }
};

// Vectors with the sentinel in every subset of the four word positions, around ordinary words.
std::vector<uint4> sentinel_patterns() {
    std::vector<uint4> v;
    for (unsigned mask = 0; mask < 16; ++mask) {
        auto w = [&](int k) { return ((mask >> k) & 1u) ? kSentinel : 0x3F803F80u + k; };
        v.push_back(make_uint4(w(0), w(1), w(2), w(3)));
    }
    // Words that are NaNs but not the sentinel must pass through unchanged.
    v.push_back(make_uint4(0x7FFF7FFFu, 0xFFFF0000u, 0x0000FFFFu, 0xFFFFFFFEu));
    return v;
}

template <ncclDataType_t kDtype>
void expect_sanitized(LamportDeviceTest& t, const std::vector<uint4>& in) {
    uint4* d_in = t.upload(in);
    uint4* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, in.size() * sizeof(uint4)), cudaSuccess);
    sanitize_kernel<kDtype><<<1, 64>>>(d_in, d_out, static_cast<int>(in.size()));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<uint4> out = t.download(d_out, in.size());
    const uint32_t canonical_nan = nccl_ep::lamport::sanitized_word<kDtype>();
    for (size_t i = 0; i < in.size(); ++i) {
        const uint32_t a[4] = {in[i].x, in[i].y, in[i].z, in[i].w};
        const uint32_t b[4] = {out[i].x, out[i].y, out[i].z, out[i].w};
        for (int k = 0; k < 4; ++k) {
            EXPECT_NE(b[k], kSentinel) << "pattern " << i << " word " << k;
            EXPECT_EQ(b[k], a[k] == kSentinel ? canonical_nan : a[k]) << "pattern " << i << " word " << k;
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);
}

TEST_F(LamportDeviceTest, SanitizeReplacesOnlySentinelWordsWithCanonicalNan) {
    const std::vector<uint4> in = sentinel_patterns();
    expect_sanitized<ncclBfloat16>(*this, in);
    expect_sanitized<ncclFloat16>(*this, in);
    expect_sanitized<ncclFloat32>(*this, in);
    // The replacement is a NaN of the token width (two 16-bit lanes, or one 32-bit lane).
    EXPECT_EQ(nccl_ep::lamport::sanitized_word<ncclBfloat16>(), 0x7FFF7FFFu);
    EXPECT_EQ(nccl_ep::lamport::sanitized_word<ncclFloat16>(), 0x7FFF7FFFu);
    EXPECT_EQ(nccl_ep::lamport::sanitized_word<ncclFloat32>(), 0x7FFFFFFFu);
}

TEST_F(LamportDeviceTest, ArrivedNeedsEveryWordToBeNonSentinel) {
    const std::vector<uint4> in = sentinel_patterns();
    uint4* d_in = upload(in);
    int* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, in.size() * sizeof(int)), cudaSuccess);
    arrived_kernel<<<1, 64>>>(d_in, d_out, static_cast<int>(in.size()));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<int> out = download(d_out, in.size());
    for (unsigned mask = 0; mask < 16; ++mask) {
        EXPECT_EQ(out[mask], mask == 0 ? 1 : 0) << "sentinel in word mask " << mask;
    }
    EXPECT_EQ(out[16], 1); // NaN-like words that are not the sentinel count as arrived
    cudaFree(d_in);
    cudaFree(d_out);
}

TEST_F(LamportDeviceTest, RearmRestoresTheSentinelInEveryWord) {
    uint4* d = upload(std::vector<uint4>{make_uint4(1, 2, 3, 4)});
    rearm_kernel<<<1, 1>>>(d);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const uint4 v = download(d, 1)[0];
    EXPECT_EQ(v.x, kSentinel);
    EXPECT_EQ(v.y, kSentinel);
    EXPECT_EQ(v.z, kSentinel);
    EXPECT_EQ(v.w, kSentinel);
    cudaFree(d);
}

TEST_F(LamportDeviceTest, ConsecutiveRoundsUseDifferentSegmentsAlsoAcrossTheCounterWrap) {
    const size_t base = 4096, stride = 1024;
    const std::vector<unsigned> rounds = {0u, 1u, 2u, 3u, 0xFFFFFFFEu, 0xFFFFFFFFu, 0u};
    unsigned* d_rounds = upload(rounds);
    size_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, rounds.size() * sizeof(size_t)), cudaSuccess);
    segment_kernel<<<1, 32>>>(base, stride, d_rounds, d_out, static_cast<int>(rounds.size()));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<size_t> out = download(d_out, rounds.size());
    EXPECT_EQ(out[0], base);
    EXPECT_EQ(out[1], base + stride);
    EXPECT_EQ(out[2], base);
    EXPECT_EQ(out[3], base + stride);
    EXPECT_EQ(out[4], base);
    EXPECT_EQ(out[5], base + stride);
    EXPECT_EQ(out[6], base); // 0xFFFFFFFF -> 0: different segments
    cudaFree(d_rounds);
    cudaFree(d_out);
}

} // namespace
