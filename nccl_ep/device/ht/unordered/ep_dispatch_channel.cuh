/*
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_HT_DISPATCH_CHANNEL_CUH_
#define NCCL_EP_HT_DISPATCH_CHANNEL_CUH_

// ============================================================================
// Each channel has a sender warp (N2N) and a symmetric receiver warp (G2S).
//
// Example: 8192 tokens, 48 channels, tokens_per_channel = 176, tokens_per_batch = 64 (default):
//
//   channel 0           channel 1           ...   channel 45          channel 46   channel 47
//   token 0..175        token 176..351            token 7920..8095    token 8096..8191     -
//   [ 64 | 64 | 48 ]    [ 64 | 64 | 48 ]          [ 64 | 64 | 48 ]    [ 64 | 32 ]
//     batches 0..2        batches 0..2              batches 0..2        batches 0..1   (empty)
//
// Why the last channel is smaller: each channel's share is rounded up to 16 tokens (see
// dispatch_tokens_per_channel), so the last channel holding tokens can only take the remainder. Here
// it sends 64 then 32 tokens while the others send 64, 64, 48, so it finishes early. That costs a
// negligible amount of balance.
// ============================================================================

#include "common.hpp"

namespace nccl_ep {
namespace ht_dense {

// Initial even split, the same for every channel, rounded up to 16 tokens: G2S and S2G read the
// routing map 16 entries per 16B load, so the channel size must be a multiple of 16.
__host__ __device__ constexpr int dispatch_tokens_per_channel(int num_tokens, int num_channels) {
    return align(ceil_div(num_tokens, num_channels), 16);
}

// Tokens a channel actually holds, per the split above. The last channel holding tokens is the
// remainder, and a channel after it can start past the end and carry none. E.g. 8192 tokens over
// 48 channels:
//   channel 0:45 = 176 tokens, channel 46 = 96, channel 47 = 0  (176 * 46 + 96 + 0 = 8192)
__host__ __device__ constexpr int dispatch_channel_size(int channel_idx, int fix_tokens_per_channel, int num_tokens) {
    const int left = num_tokens - channel_idx * fix_tokens_per_channel;
    return left <= 0 ? 0 : (left < fix_tokens_per_channel ? left : fix_tokens_per_channel);
}

// Batches a full channel can need.
__host__ __device__ constexpr int dispatch_batches_per_channel(int fix_tokens_per_channel, int fix_tokens_per_batch) {
    return ceil_div(fix_tokens_per_channel, fix_tokens_per_batch);
}

}  // namespace ht_dense
}  // namespace nccl_ep

#endif  // NCCL_EP_HT_DISPATCH_CHANNEL_CUH_
