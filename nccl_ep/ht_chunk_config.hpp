/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nccl_ep_env.h"
#include "device/ht_ep_configs.cuh"

#include <cstdio>
#include <limits>
#include <optional>

inline int nccl_ep_resolve_combine_chunk(const ncclEpEnvConfig& cfg, int dispatch_num_sms, int num_lsa_teams) {
    // RDMA (multi-LSA-team) configs use the 64-token default; single-LSA configs
    // use NCCL_EP_HT_COMBINE_TOK_PER_GROUP tokens per dispatch SM. Keeping this
    // tied to the dispatch budget prevents combine-grid tuning from changing the
    // dispatch fallback. NCCL_EP_TOKENS_PER_CHUNK overrides either default.
    // Chunks must be multiples of 32 (warp width, uint4 routing-map loads and
    // token-group granularity), so non-conforming values are rounded with a warning.
    auto round_up_32 = [](int v) { return (v + 31) & ~31; };
    int chunk = num_lsa_teams > 1 ? HT_TOKENS_PER_CHUNK_RDMA_DEFAULT
                                 : round_up_32(NCCL_EP_HT_COMBINE_TOK_PER_GROUP * dispatch_num_sms);
    const auto& var = cfg.tokens_per_chunk;
    if (var.is_set && var.value.ul > 0) {
        if (var.value.ul > static_cast<unsigned long>(std::numeric_limits<int>::max() - 31)) {
            std::fprintf(stderr, "[nccl_ep] %s=%lu ignored (value is too large)\n",
                         var.name, var.value.ul);
            return chunk;
        }
        const int requested = static_cast<int>(var.value.ul);
        chunk = round_up_32(requested);
        if (chunk != requested) {
            std::fprintf(stderr, "[nccl_ep] %s=%d rounded up to %d (must be a multiple of 32)\n",
                         var.name, requested, chunk);
        }
    }
    return chunk;
}

inline int nccl_ep_resolve_dispatch_chunk(const ncclEpEnvConfig& cfg, int combine_chunk, int num_lsa_teams) {
    const auto& var = cfg.dispatch_tokens_per_chunk;
    if (!var.is_set) return combine_chunk;
    const unsigned long chunk = var.value.ul;
    // Both kernels use capacity padded to the combine chunk.
    if (num_lsa_teams != 1 || chunk < 32 || chunk % 32 != 0 ||
        chunk > static_cast<unsigned long>(combine_chunk) || combine_chunk % chunk != 0) {
        if (cfg.rank == 0) {
            std::fprintf(stderr, "[nccl_ep] %s=%lu ignored (requires one LSA team and a multiple of 32 dividing %d)\n",
                         var.name, chunk, combine_chunk);
        }
        return combine_chunk;
    }
    return static_cast<int>(chunk);
}

namespace nccl_ep {

class HtDispatchChunkPolicy {
public:
    // An explicit chunk is already resolved and disables automatic selection.
    HtDispatchChunkPolicy(std::optional<int> explicit_chunk, int default_chunk,
                          int num_lsa_teams, int dispatch_num_sms, bool supported_layout)
        : default_chunk_(explicit_chunk.value_or(default_chunk)), dispatch_num_sms_(0) {
        // RDMA keeps the group chunk to preserve shared buffer and signal strides.
        // Other guards bound the validated tuning scope, not hardware support.
        if (supported_layout && !explicit_chunk.has_value() &&
            num_lsa_teams == 1 &&
            (default_chunk == 64 || default_chunk == 128 || default_chunk == 256) &&
            dispatch_num_sms >= 16 && dispatch_num_sms <= 64) {
            dispatch_num_sms_ = dispatch_num_sms;
        }
    }

    // Reserve scratch for any selection, independent of the current batch size.
    int minimum() const { return dispatch_num_sms_ != 0 ? 32 : default_chunk_; }

    int select(int num_tokens) const {
        if (dispatch_num_sms_ == 0 || num_tokens <= 32) return default_chunk_;
        // Target one chunk per CTA without increasing the group fallback.
        for (int chunk = 32; chunk < default_chunk_; chunk *= 2) {
            if (num_tokens <= chunk * dispatch_num_sms_) return chunk;
        }
        return default_chunk_;
    }

private:
    int default_chunk_;
    int dispatch_num_sms_;
};

} // namespace nccl_ep
