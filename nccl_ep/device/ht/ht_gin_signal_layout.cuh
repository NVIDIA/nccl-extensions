/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_HT_GIN_SIGNAL_LAYOUT_CUH_
#define NCCL_EP_HT_GIN_SIGNAL_LAYOUT_CUH_

#include "common.hpp"
#include "../ht_ep_configs.cuh"

namespace ht_gin_signals {

// The GIN context a channel's puts and signals ride on.
__forceinline__ __host__ __device__ int gin_channel_ctx(int channel, int channels) {
    return NCCL_EP_HT_RESERVED_GIN_GPU_CTXS + (channel % channels);
}

// Chunk ownership, shared by senders and receivers: chunk cidx is sent by channel
// cidx % channels, as that channel's (cidx / channels)-th chunk.
__forceinline__ __host__ __device__ int gin_warp_channel(int block, int warp, int warps_per_block) {
    return block * warps_per_block + warp;
}

__forceinline__ __host__ __device__ int gin_chunk_at(int channel, int ordinal, int channels) {
    return channel + ordinal * channels;
}

__forceinline__ __host__ __device__ int gin_channel_num_chunks(int channel, int channels, int num_chunks) {
    return channel < num_chunks ? nccl_ep::ceil_div(num_chunks - channel, channels) : 0;
}

__forceinline__ __host__ __device__ int gin_chunk_channel(int cidx, int channels) {
    return cidx % channels;
}

__forceinline__ __host__ __device__ int gin_chunk_ordinal(int cidx, int channels) {
    return cidx / channels;
}

struct gin_chunk_signal_t {
    int ctx_idx;      // GIN context to signal on / wait on
    unsigned sig_id;  // signal id within that context
};

__forceinline__ __host__ __device__ int gin_ids_per_channel(int num_chunks, int channels) {
    return nccl_ep::ceil_div(num_chunks, channels);
}

// Signals one phase needs per context: one id per (source team, chunk ordinal).
__forceinline__ __host__ __device__ int gin_signals_per_phase(int lsa_teams, int max_chunks_per_rank, int channels) {
    return channels > 0 ? (lsa_teams - 1) * gin_ids_per_channel(max_chunks_per_rank, channels) : 0;
}

__forceinline__ __host__ __device__ gin_chunk_signal_t
gin_split_chunk(int cidx, int channels, unsigned id_base) {
    gin_chunk_signal_t s;
    s.ctx_idx = gin_channel_ctx(gin_chunk_channel(cidx, channels), channels);
    s.sig_id = id_base + static_cast<unsigned>(gin_chunk_ordinal(cidx, channels));
    return s;
}

// Best explained via example. cidx = chunk idx, gcidx = global cidx,
// team_idx = the sending team's slot at this receiver.
// gcidx enumerates (cidx, team_idx) pairs in order as below for example LSA_TEAMS=4:
//  cidx0_team0, cidx0_team1, cidx0_team2, cidx1_team0, cidx1_team1 ...
// NBLOCKS=2, DISPATCH_N2N_WARPS=2 (4 warps), max_chunks_per_rank=4, receiver team = team 0:
//
//   warp 0 / gin_ctx_1              warp 1 / gin_ctx_2              warp 2 / gin_ctx_3              warp 3 / gin_ctx_4
//   gcidx team_idx gin_sig_idx      gcidx team_idx gin_sig_idx      gcidx team_idx gin_sig_idx      gcidx team_idx gin_sig_idx
//      0       0           0            3       0           0            6       0           0            9       0           0
//      1       1           1            4       1           1            7       1           1           10       1           1
//      2       2           2            5       2           2            8       2           2           11       2           2
//
// Each warp takes a contiguous run of 3: one chunk's worth of teams.
__forceinline__ __host__ __device__ gin_chunk_signal_t gin_dispatch_chunk_signal(
    int src_lteam, int dst_lteam, int cidx, int max_chunks_per_rank, int channels, unsigned signals_tail_base) {
    const int src_slot = src_lteam > dst_lteam ? src_lteam - 1 : src_lteam;
    const unsigned id_base =
        signals_tail_base + src_slot * gin_ids_per_channel(max_chunks_per_rank, channels);
    return gin_split_chunk(cidx, channels, id_base);
}

// Same enumeration as gin_dispatch_chunk_signal above.
// NBLOCKS=2, COMBINE_N2N_WARPS=1 (2 warps), receiver team = team 0:
//
//   warp 0 / gin_ctx_1              warp 1 / gin_ctx_2
//   gcidx team_idx gin_sig_idx      gcidx team_idx gin_sig_idx
//      0       0           0            1       1           0
//      2       2           1            3       0           1
//      4       1           2            5       2           2
//      6       0           3            7       1           3
//      8       2           4            9       0           4
//     10       1           5           11       2           5
__forceinline__ __host__ __device__ gin_chunk_signal_t
gin_combine_chunk_signal(int cidx, int channels, unsigned combine_signal_offset) {
    return gin_split_chunk(cidx, channels, combine_signal_offset);
}

}  // namespace ht_gin_signals

#endif  // NCCL_EP_HT_GIN_SIGNAL_LAYOUT_CUH_
