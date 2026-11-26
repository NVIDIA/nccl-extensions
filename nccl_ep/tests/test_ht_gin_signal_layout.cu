/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>
#include "device/ht/ht_gin_signal_layout.cuh"

using namespace ht_gin_signals;

namespace {

// Walk every chunk each sender channel owns and check that it lands exactly once, on the
// context the sender signals on, and that the receiver sees the same context.
template <typename ReceiverSignal>
void check_layout(int teams, int num_chunks, int max_chunks, int blocks, int warps, ReceiverSignal receiver) {
    const int channels = blocks * warps;
    std::vector<int> owner(num_chunks, -1);
    std::set<std::pair<int, unsigned>> ids;
    for (int block = 0; block < blocks; ++block) {
        for (int warp = 0; warp < warps; ++warp) {
            const int channel = gin_warp_channel(block, warp, warps);
            const int ctx = gin_channel_ctx(channel, channels);
            const int n = gin_channel_num_chunks(channel, channels, num_chunks);
            for (int k = 0; k < n; ++k) {
                const int cidx = gin_chunk_at(channel, k, channels);
                ASSERT_LT(cidx, num_chunks);
                ASSERT_EQ(owner[cidx], -1) << "chunk " << cidx << " sent twice";
                owner[cidx] = channel;
                EXPECT_EQ(gin_chunk_channel(cidx, channels), channel);
                EXPECT_EQ(gin_chunk_ordinal(cidx, channels), k);
                for (const gin_chunk_signal_t& s : receiver(cidx, channels)) {
                    EXPECT_EQ(s.ctx_idx, ctx) << "chunk " << cidx;
                    EXPECT_LT(s.sig_id, static_cast<unsigned>(gin_signals_per_phase(teams, max_chunks, channels)));
                    EXPECT_TRUE(ids.insert({s.ctx_idx, s.sig_id}).second)
                        << "ctx " << s.ctx_idx << " id " << s.sig_id << " used twice";
                }
            }
        }
    }
    for (int cidx = 0; cidx < num_chunks; ++cidx) EXPECT_NE(owner[cidx], -1) << "chunk " << cidx << " never sent";
}

}  // namespace

TEST(HtGinSignalLayout, Dispatch) {
    for (int teams : {2, 3, 4, 8}) {
        for (int max_chunks : {1, 3, 4, 17, 64, 128}) {
            for (int blocks : {1, 2, 3, 16, 24}) {
                const int warps = NCCL_EP_HT_DISPATCH_N2N_WARPS;
                // Each receiver team waits on one signal per (source team, chunk).
                for (int dst = 0; dst < teams; ++dst) {
                    auto receiver = [&](int cidx, int channels) {
                        std::vector<gin_chunk_signal_t> v;
                        for (int src = 0; src < teams; ++src)
                            if (src != dst)
                                v.push_back(gin_dispatch_chunk_signal(src, dst, cidx, max_chunks, channels, 0));
                        return v;
                    };
                    check_layout(teams, max_chunks, max_chunks, blocks, warps, receiver);
                }
            }
        }
    }
}

TEST(HtGinSignalLayout, Combine) {
    for (int teams : {2, 3, 4, 8}) {
        for (int max_chunks : {1, 3, 4, 17, 64, 128}) {
            for (int blocks : {1, 2, 3, 16, 32}) {
                // One channel per CTA over the wire index g, (teams - 1) * max_chunks in total.
                auto receiver = [](int g, int channels) {
                    return std::vector<gin_chunk_signal_t>{gin_combine_chunk_signal(g, channels, 0)};
                };
                check_layout(teams, (teams - 1) * max_chunks, max_chunks, blocks, 1, receiver);
            }
        }
    }
}
