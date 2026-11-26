/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_HT_UNORDERED_GIN_DISPATCH_BATCH_CUH_
#define NCCL_EP_HT_UNORDERED_GIN_DISPATCH_BATCH_CUH_

#include "common.hpp"
#include "device/ht/ht_gin_signal_layout.cuh"
#include "ep_dispatch_channel.cuh"

namespace nccl_ep {
namespace ht {

// Unordered dispatch signals once per batch through ht_gin_signals::gin_dispatch_chunk_signal:
// the caller passes gcidx = batch * channels + channel, so each channel signals on its own context
// and each (batch, sending team) gets its own id. team_idx = the sending team's slot at this receiver.
// channels=2, 3 batches per channel, LSA_TEAMS=4 (3 sending teams), receiver team = team 0:
//
//   channel 0 / gin_ctx_1                  channel 1 / gin_ctx_2
//   batch gcidx team_idx gin_sig_idx       batch gcidx team_idx gin_sig_idx
//       0     0        0           0           0     1        0           0
//       1     2        0           1           1     3        0           1
//       2     4        0           2           2     5        0           2
//       0     0        1           3           0     1        1           3
//       1     2        1           4           1     3        1           4
//       2     4        1           5           2     5        1           5
//       0     0        2           6           0     1        2           6
//       1     2        2           7           1     3        2           7
//       2     4        2           8           2     5        2           8
//
// gin_sig_idx = team_idx * batches per channel + batch.

// Maximum signals required per context for the current topology: one per batch, per sending team.
__host__ __device__ constexpr int dispatch_signals_per_ctx(int num_tokens, int num_channels,
                                                        int fix_tokens_per_batch, int remote_lsa_teams) {
    return remote_lsa_teams *
           ht_dense::dispatch_batches_per_channel(ht_dense::dispatch_tokens_per_channel(num_tokens, num_channels),
                                                 fix_tokens_per_batch);
}

}  // namespace ht
}  // namespace nccl_ep

#endif  // NCCL_EP_HT_UNORDERED_GIN_DISPATCH_BATCH_CUH_
