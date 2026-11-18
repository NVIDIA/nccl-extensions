/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_DEVICE_HT_COMBINE_PACKED_CUH_
#define NCCL_EP_DEVICE_HT_COMBINE_PACKED_CUH_

#include <cstddef>
#include <cstdint>

#include "common.hpp"

// ============================================================================
// Packed contiguous combine RDMA layout.
//
// The reduction warp writes the n-th routed token of the chunk at slot n, so the
// chunk is one contiguous run and the batcher always emits ceil(n_routed /
// MAX_BATCH) full puts regardless of routing. The layout it replaced wrote each
// token at its own index, leaving a hole for every token the destination did not
// need; a hole ends a batch, so the put count followed the MoE routing pattern
// rather than the amount of data -- one put per token at worst. Used by the
// unordered kernel; the ordered kernel keeps the per-token layout.
//
// The sender needs no help from here: its walk over the chunk is sequential, so
// one counter gives it the slot. The receiver is the hard half and is what this
// header exists for. It does NOT walk a chunk in order -- combine_G2S_inter_warp
// strides its token groups by NBLOCKS and its tokens within a group by the G2S
// warp count -- so no block or warp ever holds a contiguous prefix of a chunk and
// there is no running counter to carry. Instead each lane reconstructs the
// sender's compaction from the routing map it already has: the slot of token t is
// the number of tokens before t in the chunk that this sender routed here, i.e. a
// prefix popcount over one column of the map. One mask per lane per chunk, then
// one popcount per token.
//
// PRECONDITION: the sender's rdma_to_attn_map and the receiver's attn_to_rdma_map
// must be exact transposes. NCCL_EP_OVERFLOW_DROP clears the sender's bit for a
// dropped token, so slots are numbered from the pristine rdma_to_attn_map_wo_drop
// and a dropped token's slot is reserved and zero-filled.
// ============================================================================

namespace nccl_ep {
namespace ht_packed {

// One chunk's worth of routing bits for a single (sender, receiver) pair. Bit t is set iff
// token t of the chunk is routed to this pair: bit 2 set means chunk token 2 is valid here
// and occupies a slot, clear means it is skipped. prefix() popcounts the bits below t to
// turn that into the token's packed slot.
template <int TOKENS_PER_CHUNK>
struct chunk_mask_t {
    using word_t = uint64_t;
    static constexpr int BITS_PER_WORD = 8 * sizeof(word_t);
    static constexpr int WORDS = nccl_ep::ceil_div(TOKENS_PER_CHUNK, BITS_PER_WORD);
    word_t words[WORDS];
};

// Build each lane's column of the chunk's routing map: bit t of lane L's mask is set iff
// token t of the chunk is routed for the pair whose column is L's `my_col`. `map` is
// token-major with `stride` bools per token. Tokens at or past `csize` (the tail chunk)
// stay clear, and a lane passing my_col < 0 keeps a zeroed mask.
//
// WARP-COLLECTIVE: every lane of the warp must call this, because it transposes via
// __ballot_sync. One ballot yields a whole 32-token slice of ONE column, so the warp walks
// the columns and each lane keeps only its own -- 2 * num_cols ballots in place of the
// csize serial single-byte loads per lane that the obvious form costs. At a 64-token chunk
// and 4 LSA teams that is 6 ballots rather than 64 loads per lane, and it is why this is
// warp-collective rather than per-lane: the serial form measured 13.8% slower than the
// sparse layout at 4 nodes, which was the decode cost and not the layout.
//
// Example: TOKENS_PER_CHUNK = 64, LSA_TEAMS = 4, called from combine on node 4 of 4
// (my_lteam = 3), so the columns are the other three nodes, teams 0, 1, 2.
//
// IN: attn_to_rdma_map_base for chunk N, one BYTE per entry. Size: 64 * 3 = 192 B.
//
//    t | team0 team1 team2
//    0 |   1     0     1     <- token 0 came from remote teams 0 and 2
//    1 |   0     1     0
//    2 |   1     1     0
//    3 |   0     0     1
//    4 |   1     0     0
//    5 |   1     1     1
//    6 |   0     1     0
//    7 |   1     0     1
//  ... |  ...   ...   ...    (tokens 8..62 unrouted in this example)
//   63 |   0     1     0
//
// OUT: one bitmap per lane holding ONE column, one BIT per entry -- bit t = token t came
// from that lane's team. 8 B per lane, LSA_TEAMS - 1 = 3 live, so 24 B of registers for
// the 192 B read.
//
//   lane  team  bitmap (64 bit)      low byte       tokens
//    0      2   0x00000000000000A9   0...10101001   0, 3, 5, 7
//    1      1   0x8000000000000066   0...01100110   1, 2, 5, 6, 63
//    2      0   0x00000000000000B5   0...10110101   0, 2, 4, 5, 7
//   3..31   -   0x0000000000000000   0...00000000   none
//
// Bit 0 is token 0, so the binary reads right to left.
template <int TOKENS_PER_CHUNK>
__device__ __forceinline__ chunk_mask_t<TOKENS_PER_CHUNK>
build_chunk_mask(const bool* map, int stride, int num_cols, int my_col, int csize) {
    using mask_t = chunk_mask_t<TOKENS_PER_CHUNK>;
    constexpr int WARP_SIZE = 32;
    // One ballot carries WARP_SIZE bits, so it fills only part of a word.
    constexpr int PARTS_PER_WORD = mask_t::BITS_PER_WORD / WARP_SIZE;
    mask_t m;
#pragma unroll
    for (int i = 0; i < mask_t::WORDS; ++i) {
        m.words[i] = 0;
    }
    const int lane = static_cast<int>(threadIdx.x) & (WARP_SIZE - 1);
    for (int c = 0; c < num_cols; ++c) {
#pragma unroll
        for (int wi = 0; wi < mask_t::WORDS; ++wi) {
#pragma unroll
            for (int part = 0; part < PARTS_PER_WORD; ++part) {
                const int t = wi * mask_t::BITS_PER_WORD + part * WARP_SIZE + lane;
                const bool bit = (t < csize) && map[t * stride + c];
                const unsigned slice = __ballot_sync(0xffffffffu, bit);
                if (c == my_col) {
                    m.words[wi] |= static_cast<typename mask_t::word_t>(slice) << (part * WARP_SIZE);
                }
            }
        }
    }
    return m;
}

// Packed slot of token `tok`: how many tokens before it in this chunk were routed, i.e. sparse
// index -> packed index. Whole words below tok are counted outright; the word holding tok is
// first masked to the bits below it.
//
// 0xB5 = 1011 0101 -> tokens 0, 2, 4, 5, 7 routed:
//   sparse:  [0][-][2][-][4][5][-][7]      slot == token index, holes
//   packed:  [0][2][4][5][7]               slots 0..4
// prefix(0xB5, 5) = 3: token 5 was at sparse slot 5, now at packed slot 3.
template <int TOKENS_PER_CHUNK>
__device__ __forceinline__ int prefix(const chunk_mask_t<TOKENS_PER_CHUNK>& m, int tok) {
    using mask_t = chunk_mask_t<TOKENS_PER_CHUNK>;
    int count = 0;
    const int word = tok / mask_t::BITS_PER_WORD;
#pragma unroll
    for (int i = 0; i < mask_t::WORDS; ++i) {
        if (i < word) {
            count += __popcll(m.words[i]);
        }
    }
    const int bit = tok % mask_t::BITS_PER_WORD;
    if (bit != 0) {
        count += __popcll(m.words[word] & ((typename mask_t::word_t{1} << bit) - 1));
    }
    return count;
}

// One RDMA slot holds a token row and, in backward combine, its prob row right after it, so a
// batch of slots is one contiguous span and travels in a single gin.put.
__device__ __forceinline__ size_t slot_prob_bytes(bool backward, int prob_dim) {
    return backward ? nccl_ep::align<size_t>(static_cast<size_t>(prob_dim) * sizeof(float), 16) : 0;
}

__device__ __forceinline__ size_t slot_bytes(size_t token_bytes, bool backward, int prob_dim) {
    return token_bytes + slot_prob_bytes(backward, prob_dim);
}

}  // namespace ht_packed
}  // namespace nccl_ep

#endif  // NCCL_EP_DEVICE_HT_COMBINE_PACKED_CUH_
