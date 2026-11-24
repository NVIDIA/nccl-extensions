/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include "ht_chunk_config.hpp"
#include "ht_sm_policy.hpp"

using nccl_ep::HtDispatchChunkPolicy;
using nccl_ep::HtDispatchSchedule;

TEST(HtChunkConfig, AdaptiveSelectionWithTailScheduling) {
    for (int blocks : {16, 24, 32, 64}) {
        for (int fallback : {128, 256}) {
            const HtDispatchChunkPolicy policy(std::nullopt, fallback, 1, blocks, true);
            EXPECT_EQ(policy.select(64 * blocks), 64);
            for (int tokens : {64 * blocks + 1, 112 * blocks - 1, 112 * blocks, 112 * blocks + 1}) {
                EXPECT_EQ(policy.select(tokens), 128);
            }
            EXPECT_EQ(policy.select(128 * blocks), 128);
            EXPECT_EQ(policy.select(128 * blocks + 1), fallback);
            EXPECT_EQ(HtDispatchChunkPolicy(fallback, fallback, 1, blocks, true).select(96 * blocks), fallback);
            EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, fallback, 2, blocks, true).select(96 * blocks), fallback);
        }
    }
}

TEST(HtChunkConfig, TailScheduleUsesWorkNotCapacity) {
    for (int blocks : {16, 24, 32, 64}) {
        for (int chunk : {32, 64, 128, 256}) {
            const int wave = blocks * chunk;
            for (int tokens : {0, 1, wave - 1, wave, 2 * wave, 3 * wave}) {
                EXPECT_EQ(nccl_ep::ht_dispatch_schedule(tokens, chunk, blocks, 1), HtDispatchSchedule::Coarse);
            }
            for (int tokens : {wave + 1, wave + 16, 2 * wave - 1, 2 * wave + 1}) {
                EXPECT_EQ(nccl_ep::ht_dispatch_schedule(tokens, chunk, blocks, 1), HtDispatchSchedule::Redistributed);
                EXPECT_EQ(nccl_ep::ht_dispatch_schedule(tokens, chunk, blocks, 2), HtDispatchSchedule::Coarse);
            }
        }
    }
}

TEST(HtChunkConfig, TailScheduleSafeArithmetic) {
    const int limit = std::numeric_limits<int>::max();
    for (int invalid : {-1, 0}) {
        EXPECT_EQ(nccl_ep::ht_dispatch_schedule(invalid, 128, 32, 1), HtDispatchSchedule::Coarse);
        EXPECT_EQ(nccl_ep::ht_dispatch_schedule(8192, invalid, 32, 1), HtDispatchSchedule::Coarse);
        EXPECT_EQ(nccl_ep::ht_dispatch_schedule(8192, 128, invalid, 1), HtDispatchSchedule::Coarse);
    }
    EXPECT_EQ(nccl_ep::ht_dispatch_schedule(limit, limit, limit, 1), HtDispatchSchedule::Coarse);
    EXPECT_EQ(nccl_ep::ht_dispatch_schedule(limit, 128, 32, 1), HtDispatchSchedule::Redistributed);
    for (int blocks : {8, 128, limit}) {
        EXPECT_EQ(nccl_ep::ht_dispatch_schedule(32769, 128, blocks, 1), HtDispatchSchedule::Coarse);
    }
    for (int chunk : {16, 96, 192, 512, limit}) {
        EXPECT_EQ(nccl_ep::ht_dispatch_schedule(32769, chunk, 32, 1), HtDispatchSchedule::Coarse);
    }
}

TEST(HtChunkConfig, CombinePreservesDefault) {
    ncclEpEnvConfig cfg;
    for (int sms : {1, 8, 16, 17, 32, 64, 152, 212}) {
        EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, sms, 1), (4 * sms + 31) & ~31);
        EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, sms, 2), 64);
    }
}

TEST(HtChunkConfig, CombineOverrideAndRounding) {
    ncclEpEnvConfig cfg;
    cfg.tokens_per_chunk.is_set = true;
    for (int teams : {1, 2}) {
        for (unsigned long value : {1ul, 31ul, 32ul, 33ul, 64ul, 96ul, 128ul}) {
            cfg.tokens_per_chunk.value.ul = value;
            EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 16, teams), (value + 31) & ~31ul);
        }
    }
    cfg.tokens_per_chunk.value.ul = 0;
    EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 32, 1), 128);
    EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 32, 2), 64);
}

TEST(HtChunkConfig, CombineOverrideRange) {
    ncclEpEnvConfig cfg;
    cfg.tokens_per_chunk.is_set = true;
    const unsigned long int_max = static_cast<unsigned long>(std::numeric_limits<int>::max());
    const unsigned long max_chunk = int_max - 31;
    for (int teams : {1, 2}) {
        for (unsigned long value : {max_chunk - 31, max_chunk - 1, max_chunk}) {
            cfg.tokens_per_chunk.value.ul = value;
            EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 32, teams), max_chunk);
        }
        for (unsigned long value : {max_chunk + 1, int_max, int_max + 1,
                                    std::numeric_limits<unsigned long>::max()}) {
            cfg.tokens_per_chunk.value.ul = value;
            EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 32, teams), teams == 1 ? 128 : 64);
        }
    }
}

TEST(HtChunkConfig, DispatchOverrideDoesNotChangeCombine) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    cfg.dispatch_tokens_per_chunk.value.ul = 32;
    const int combine = nccl_ep_resolve_combine_chunk(cfg, 32, 1);
    EXPECT_EQ(combine, 128);
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, combine, 1), 32);
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, combine, 2), 128);
}

TEST(HtChunkConfig, PreservesDefault) {
    ncclEpEnvConfig cfg;
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 64, 1), 64);
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 128, 2), 128);
}

TEST(HtChunkConfig, ValidOverride) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    for (unsigned long value : {32ul, 64ul, 128ul}) {
        cfg.dispatch_tokens_per_chunk.value.ul = value;
        EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 128, 1), value);
    }
}

TEST(HtChunkConfig, RejectsInvalidAndRdma) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    for (unsigned long value : {0ul, 1ul, 31ul, 48ul, 96ul, 256ul, std::numeric_limits<unsigned long>::max()}) {
        cfg.dispatch_tokens_per_chunk.value.ul = value;
        EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 128, 1), 128);
    }
    cfg.dispatch_tokens_per_chunk.value.ul = 32;
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 64, 2), 64);
}

TEST(HtChunkConfig, ActualTokenBoundary) {
    for (int sms : {16, 32, 64}) {
        for (int tokens : {0, 1, 31, 32, 33, 128, 512, 513, 768, 1024, 2048, 16384}) {
            const int expected = tokens > 32 && tokens <= 32 * sms ? 32 : 64;
            EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, sms, true).select(tokens), expected);
        }
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, sms, true).select(32 * sms), 32);
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, sms, true).select(32 * sms + 1), 64);
    }
}

TEST(HtChunkConfig, AutomaticSelectionScope) {
    for (int sms : {0, 8, 128, std::numeric_limits<int>::max()})
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, sms, true).select(128), 64);
    for (int chunk : {32, 96, 192, 512})
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, chunk, 1, 16, true).select(128), chunk);
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 2, 16, true).select(128), 64);
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, 16, false).select(128), 64);
}

TEST(HtChunkConfig, ExplicitSelectionWins) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    for (unsigned long value : {0ul, 32ul, 48ul, 64ul, std::numeric_limits<unsigned long>::max()}) {
        cfg.dispatch_tokens_per_chunk.value.ul = value;
        const int resolved = nccl_ep_resolve_dispatch_chunk(cfg, 64, 1);
        for (int tokens : {0, 33, 128, 1024})
            EXPECT_EQ(HtDispatchChunkPolicy(resolved, 64, 1, 16, true).select(tokens), resolved);
    }
}

TEST(HtChunkConfig, ScratchCoversEverySelection) {
    for (bool overridden : {false, true}) {
        for (int chunk : {64, 128, 256}) {
            const std::optional<int> explicit_chunk = overridden ? std::optional<int>(chunk) : std::nullopt;
            for (int sms : {8, 16, 32, 64, 128}) {
                const HtDispatchChunkPolicy policy(explicit_chunk, chunk, 1, sms, true);
                const int minimum = policy.minimum();
                EXPECT_EQ(minimum, !overridden && sms >= 16 && sms <= 64 ? 32 : chunk);
                for (int tokens = 0; tokens <= 16384; ++tokens) {
                    const int selected = policy.select(tokens);
                    EXPECT_GE(selected, minimum);
                    EXPECT_LE(selected, chunk);
                    EXPECT_EQ(chunk % selected, 0);
                }
            }
        }
    }
}

TEST(HtChunkConfig, MinimumRespectsSelectionScope) {
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, 16, false).minimum(), 64);
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 2, 16, true).minimum(), 64);
    for (int chunk : {32, 96, 192, 512})
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, chunk, 1, 16, true).minimum(), chunk);
}

TEST(HtChunkConfig, IntermediateBoundaries) {
    for (int fallback : {64, 128, 256}) {
        for (int sms : {16, 32, 64}) {
            const HtDispatchChunkPolicy policy(std::nullopt, fallback, 1, sms, true);
            EXPECT_EQ(policy.minimum(), 32);
            for (int tokens : {std::numeric_limits<int>::min(), 0, 1, 31, 32})
                EXPECT_EQ(policy.select(tokens), fallback);
            EXPECT_EQ(policy.select(33), 32);
            EXPECT_EQ(policy.select(std::numeric_limits<int>::max()), fallback);
            for (int chunk = 32; chunk < fallback; chunk *= 2) {
                EXPECT_EQ(policy.select(chunk * sms - 1), chunk);
                EXPECT_EQ(policy.select(chunk * sms), chunk);
                EXPECT_EQ(policy.select(chunk * sms + 1), chunk * 2);
            }
        }
    }
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 64, 1, 16, true).select(768), 64);
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 128, 1, 32, true).select(2048), 64);
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 128, 1, 32, true).select(8192), 128);
    EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, 256, 1, 64, true).select(8192), 128);
}

TEST(HtChunkConfig, LargerDefaultSafeguards) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    for (int fallback : {128, 256}) {
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, fallback, 2, 32, true).select(128), fallback);
        EXPECT_EQ(HtDispatchChunkPolicy(std::nullopt, fallback, 1, 32, false).minimum(), fallback);
        for (unsigned long value : {0ul, 32ul, 48ul, 64ul, 128ul, 256ul, std::numeric_limits<unsigned long>::max()}) {
            cfg.dispatch_tokens_per_chunk.value.ul = value;
            const int resolved = nccl_ep_resolve_dispatch_chunk(cfg, fallback, 1);
            const HtDispatchChunkPolicy policy(resolved, fallback, 1, 32, true);
            EXPECT_EQ(policy.select(128), resolved);
            EXPECT_EQ(policy.minimum(), resolved);
        }
    }
}

TEST(HtChunkConfig, ExplicitDefaultDisablesSelection) {
    for (int fallback : {64, 128, 256}) {
        for (int sms : {16, 32, 64}) {
            const HtDispatchChunkPolicy automatic(std::nullopt, fallback, 1, sms, true);
            const HtDispatchChunkPolicy explicit_default(fallback, fallback, 1, sms, true);
            EXPECT_EQ(automatic.minimum(), 32);
            EXPECT_EQ(automatic.select(128), 32);
            EXPECT_EQ(explicit_default.minimum(), fallback);
            EXPECT_EQ(explicit_default.select(128), fallback);
        }
    }
}

TEST(HtSmPolicy, ArchitectureAndLayoutDefaults) {
    for (unsigned int sm : {90u, 100u, 103u, 110u, 120u, 121u}) {
        EXPECT_EQ(nccl_ep::ht_default_comm_sms(sm, false), 16u);
        EXPECT_EQ(nccl_ep::ht_default_comm_sms(sm, true), 24u);
    }
    EXPECT_EQ(nccl_ep::ht_default_comm_sms(107, false), 16u);
    EXPECT_EQ(nccl_ep::ht_default_comm_sms(107, true), 32u);
    EXPECT_EQ(nccl_ep::ht_flat_combine_default_sms(8191), 16u);
    EXPECT_EQ(nccl_ep::ht_flat_combine_default_sms(8192), 32u);
}

// resolve_combine_push_chunk returns the pre-tile-rounding width (int4) fed to
// combine_push_chunk_size: 0 is the whole row, 1 (one tile) is the automatic default. SM107
// keeps one tile at every EP size; other architectures switch to the whole row from EP16.
// A malformed override keeps the automatic default and reports kInvalid.
TEST(HtChunkConfig, CombinePushResolveDefaultAndInvalid) {
    for (int ep_size : {4, 15, 16, 32, 64}) {
        const auto rubin = nccl_ep::resolve_combine_push_chunk(107, ep_size, 2, "");
        EXPECT_EQ(rubin.requested_int4, 1);
        EXPECT_EQ(rubin.message, nccl_ep::CombinePushChunkMessage::kNone);
        for (unsigned int sm : {90u, 100u, 103u, 120u}) {
            const auto r = nccl_ep::resolve_combine_push_chunk(sm, ep_size, 2, "");
            EXPECT_EQ(r.requested_int4, ep_size >= 16 ? 0 : 1);
            EXPECT_EQ(r.message, nccl_ep::CombinePushChunkMessage::kNone);
        }
    }
    for (const char* bad : {"abc", "12x", "-1", "600000000", "99999999999"}) {
        const auto rubin = nccl_ep::resolve_combine_push_chunk(107, 16, 2, bad);
        EXPECT_EQ(rubin.requested_int4, 1);
        EXPECT_EQ(rubin.message, nccl_ep::CombinePushChunkMessage::kInvalid);
        const auto other = nccl_ep::resolve_combine_push_chunk(100, 16, 2, bad);
        EXPECT_EQ(other.requested_int4, 0);
        EXPECT_EQ(other.message, nccl_ep::CombinePushChunkMessage::kInvalid);
    }
}

TEST(HtChunkConfig, CombinePushResolveOverride) {
    // 0 forces the whole row; a positive value rounds up to int4 by elem_bytes. Both
    // override the arch/EP default, so SM107 and others resolve identically.
    for (unsigned int sm : {90u, 107u}) {
        for (int ep_size : {15, 16}) {
            const auto zero = nccl_ep::resolve_combine_push_chunk(sm, ep_size, 2, "0");
            EXPECT_EQ(zero.requested_int4, 0);
            EXPECT_EQ(zero.message, nccl_ep::CombinePushChunkMessage::kNone);
        }
    }
    struct Case {
        int elems;
        int elem_bytes;
        int int4s;
    };
    for (const Case c : {Case{1, 2, 1}, Case{16, 2, 2}, Case{17, 2, 3},
                         Case{17, 1, 2}, Case{8, 4, 2}}) {
        for (unsigned int sm : {100u, 107u}) {
            const auto r = nccl_ep::resolve_combine_push_chunk(sm, 15, c.elem_bytes,
                                                               std::to_string(c.elems));
            EXPECT_EQ(r.requested_int4, c.int4s);
            EXPECT_EQ(r.message, nccl_ep::CombinePushChunkMessage::kExplicitWidth);
        }
    }
}
