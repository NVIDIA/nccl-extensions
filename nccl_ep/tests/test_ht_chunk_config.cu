/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <limits>
#include "ht_chunk_config.hpp"

using nccl_ep::HtDispatchChunkPolicy;

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
