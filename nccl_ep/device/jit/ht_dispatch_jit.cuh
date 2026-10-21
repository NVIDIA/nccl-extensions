/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ht_ep.cuh"
#include "device/ht_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_utils.hpp"
#include "device/jit/jit_source_literals.hpp"
#include "nccl_ep_env.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace nccl_ep {
namespace ht {
namespace jit {

constexpr const char* kDispatchJitEntryName = "nccl_ep_jit_ht_dispatch_kernel";

struct dispatch_warp_layout_t {
    int cross_lsa_group_warps;
    int cross_lsa_group_start;
    int lsa_g2s_group_warps;
    int lsa_g2s_group_start;
    int lsa_s2g_group_warps;
    int lsa_s2g_group_start;
    int map_group_warps;
    int map_group_start;
    int pad_group_warps;
    int pad_group_start;
    int head_extra_group_warps;
    int head_extra_group_start;
    int num_pipelines;
    int block_dim;
};

// The dispatch head runs on warps 0, 1, 2 (mbarrier init / RDMA guard / intra-LSA
// barrier), so every block needs at least this many warps even when the
// communication groups need fewer.
inline constexpr int kDispatchHeadWarps = 3;

inline dispatch_warp_layout_t
compute_dispatch_warp_layout(int num_lsa_teams, ncclEpLayout_t layout, int num_pipelines,
                             bool count_map_warp) {
    const bool multi_lsa_layout = (num_lsa_teams != 1);
    dispatch_warp_layout_t L{};
    L.num_pipelines = num_pipelines;
    L.cross_lsa_group_warps = multi_lsa_layout ? NCCL_EP_HT_DISPATCH_N2N_WARPS : 0;
    L.cross_lsa_group_start = 0;
    L.lsa_g2s_group_warps = L.num_pipelines;
    L.lsa_g2s_group_start = multi_lsa_layout ? NCCL_EP_HT_DISPATCH_N2N_WARPS : 0;
    L.lsa_s2g_group_warps = L.num_pipelines;
    L.lsa_s2g_group_start = L.lsa_g2s_group_start + L.lsa_g2s_group_warps;
    // Count mode dispatches FLAT (rank-major) with no EM padding, so it swaps the single EM PAD
    // warp for a dedicated MAP group (the two are mutually exclusive). The MAP warps build the
    // receiver EM tables on round-robin chunks; both warps also build the s2d rows that S2G gates
    // on, so this is the s2d builder count and the floor is 2.
    L.map_group_warps = count_map_warp ? 2 : 0;
    L.map_group_start = L.lsa_s2g_group_start + L.lsa_s2g_group_warps;
    L.pad_group_warps = count_map_warp ? 0 : ((layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? 1 : 0);
    L.pad_group_start = L.map_group_start + L.map_group_warps;
    // Filler warps appended after PAD to guarantee the head always has 3 warps.
    const int comm_warps = L.cross_lsa_group_warps + L.lsa_g2s_group_warps +
                           L.lsa_s2g_group_warps + L.map_group_warps + L.pad_group_warps;
    L.head_extra_group_start = comm_warps;
    L.head_extra_group_warps = (comm_warps >= kDispatchHeadWarps) ? 0 : (kDispatchHeadWarps - comm_warps);
    L.block_dim = 32 * (comm_warps + L.head_extra_group_warps);
    return L;
}

inline std::string dispatch_jit_source(
    int cross_lsa_group_warps,
    int cross_lsa_group_start,
    int lsa_g2s_group_warps,
    int lsa_g2s_group_start,
    int lsa_s2g_group_warps,
    int lsa_s2g_group_start,
    int map_group_warps,
    int map_group_start,
    int pad_group_warps,
    int pad_group_start,
    int head_extra_group_warps,
    int head_extra_group_start,
    int num_of_stages,
    int num_of_in_flight_s2g,
    int num_of_tokens_per_chunk,
    int max_tokens_per_rank,
    int num_lsa_teams,
    int num_of_blocks,
    bool forward_dispatch,
    int num_pipelines,
    int lsa_team_size,
    ncclEpLayout_t layout,
    int hidden_dim,
    int sf_bytes_per_token,
    int experts_per_rank,
    const DispatchKernelSpec& kernel_spec) {
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    std::ostringstream src;
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "using TOKEN_DATA_TYPE = " << kernel_spec.payload_type_literal << ";\n"
        << "using SCALE_DATA_TYPE = " << kernel_spec.scale_type_literal << ";\n"
        << "static constexpr int kSfBytesPerToken = " << sf_bytes_per_token << ";\n"
        << "using GIN_GROUP     = ht_ep::warp_group<" << cross_lsa_group_warps << ", "
        << cross_lsa_group_start << ">;\n"
        << "using LSA_G2S_GROUP = ht_ep::warp_group<" << lsa_g2s_group_warps << ", "
        << lsa_g2s_group_start << ">;\n"
        << "using LSA_S2G_GROUP = ht_ep::warp_group<" << lsa_s2g_group_warps << ", "
        << lsa_s2g_group_start << ">;\n"
        << "using MAP_GROUP            = ht_ep::warp_group<" << map_group_warps << ", " << map_group_start << ">;\n"
        << "using PAD_GROUP            = ht_ep::warp_group<" << pad_group_warps << ", " << pad_group_start << ">;\n"
        << "using HEAD_EXTRA_GROUP     = ht_ep::warp_group<" << head_extra_group_warps << ", "
        << head_extra_group_start << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(GIN_GROUP::size() + LSA_G2S_GROUP::size() + "
           "LSA_S2G_GROUP::size() + MAP_GROUP::size() + PAD_GROUP::size() + HEAD_EXTRA_GROUP::size(), 1)\n"
        << "__global__ void " << kDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ ht_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, " << lsa_team_size
        << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  ht_ep::dispatch_kernel_impl<\n"
        << "      TOKEN_DATA_TYPE,\n"
        << "      SCALE_DATA_TYPE,\n"
        << "      " << kernel_spec.recipe_source_literal << ",\n"
        << "      GIN_GROUP,\n"
        << "      LSA_G2S_GROUP,\n"
        << "      LSA_S2G_GROUP,\n"
        << "      MAP_GROUP,\n"
        << "      PAD_GROUP,\n"
        << "      HEAD_EXTRA_GROUP,\n"
        << "      " << num_of_stages << ",\n"
        << "      " << num_of_in_flight_s2g << ",\n"
        << "      " << num_of_tokens_per_chunk << ",\n"
        << "      " << max_tokens_per_rank << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << ::nccl_ep::jit::bool_literal(forward_dispatch) << ",\n"
        << "      " << num_pipelines << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << hidden_dim << ",\n"
        << "      kSfBytesPerToken,\n"
        << "      " << experts_per_rank << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

inline ncclResult_t launch_dispatch(
    const ::ht_ep::dispatch_config_t& config,
    int max_tokens_per_rank,
    int num_lsa_teams,
    int lsa_team_size,
    ncclEpLayout_t layout,
    int hidden_dim,
    int sf_bytes_per_token,
    int experts_per_rank,
    const ncclEpEnvConfig* env,  // for rank-0-gated verbose param dump; may be null
    void* param,
    size_t param_size,
    int dynamic_smem_bytes,
    cudaStream_t stream,
    const DispatchKernelSpec& kernel_spec,
    bool count_map_warp) {
    const dispatch_warp_layout_t L =
        compute_dispatch_warp_layout(num_lsa_teams, layout, config.num_pipelines, count_map_warp);

    static const int fwd_variant_identity = 0;
    static const int bwd_variant_identity = 0;
    const int& variant_identity = config.forward_dispatch ? fwd_variant_identity : bwd_variant_identity;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "dispatch"
             << "_LSATeams" << num_lsa_teams << "_lsa" << lsa_team_size << "_hdim" << hidden_dim << "_stages"
             << config.num_of_stages << "_pipe" << config.num_pipelines << "_inflt"
             << config.num_of_in_flight_s2g << "_chunk" << config.num_of_tokens_per_chunk << "_maxt"
             << max_tokens_per_rank << "_blocks" << config.num_of_blocks
             << (config.forward_dispatch ? "_fwd" : "_bwd")
             << ::nccl_ep::jit::layout_name_tag(layout)
             << "_recipe" << kernel_spec.recipe_cache_tag
             << "_payload" << kernel_spec.payload_cache_tag
             << "_scale" << kernel_spec.scale_cache_tag
             << "_sf" << sf_bytes_per_token
             << "_epr" << experts_per_rank
             << (count_map_warp ? "_countmap" : "");
        return name.str();
    }();
    const std::string source = dispatch_jit_source(
        L.cross_lsa_group_warps,
        L.cross_lsa_group_start,
        L.lsa_g2s_group_warps,
        L.lsa_g2s_group_start,
        L.lsa_s2g_group_warps,
        L.lsa_s2g_group_start,
        L.map_group_warps,
        L.map_group_start,
        L.pad_group_warps,
        L.pad_group_start,
        L.head_extra_group_warps,
        L.head_extra_group_start,
        config.num_of_stages,
        config.num_of_in_flight_s2g,
        config.num_of_tokens_per_chunk,
        max_tokens_per_rank,
        num_lsa_teams,
        config.num_of_blocks,
        config.forward_dispatch,
        L.num_pipelines,
        lsa_team_size,
        layout,
        hidden_dim,
        sf_bytes_per_token,
        experts_per_rank,
        kernel_spec);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_dispatch";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kDispatchJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = config.num_of_blocks;
    variant.block_dim = L.block_dim;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;
    variant.nucs = env != nullptr && nccl_ep_env_flag_on(env->nucs);

    // Dump the effective kernel parameters once per distinct variant, just before
    // the launch, when the env config asks for verbose output on this rank.
    if (env != nullptr && nccl_ep_env_verbose(*env) && ::nccl_ep::jit::announce_once(variant_name)) {
        std::fprintf(
            stderr,
            "[nccl_ep][env] HT dispatch kernel (%s):\n"
            "[nccl_ep][env]   nodes(lsa_teams)=%d lsa_team_size=%d hidden_dim=%d\n"
            "[nccl_ep][env]   stages=%d in_flight_s2g=%d pipelines=%d tokens_per_chunk=%d max_tokens_per_rank=%d\n"
            "[nccl_ep][env]   num_blocks=%d block_dim=%d dynamic_smem_bytes=%d\n"
            "[nccl_ep][env]   forward=%s layout=%s wire_dtype=%s dispatch_recipe=%s sf_bytes_per_token=%d\n",
            variant_name.c_str(),
            num_lsa_teams,
            lsa_team_size,
            hidden_dim,
            config.num_of_stages,
            config.num_of_in_flight_s2g,
            L.num_pipelines,
            config.num_of_tokens_per_chunk,
            max_tokens_per_rank,
            config.num_of_blocks,
            L.block_dim,
            dynamic_smem_bytes,
            ::nccl_ep::jit::bool_literal(config.forward_dispatch),
            (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "EXPERT_MAJOR" : "FLAT"),
            kernel_spec.wire_dtype_literal,
            kernel_spec.recipe_source_literal,
            sf_bytes_per_token);
    }

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, param, param_size, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal dispatch JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
inline void dispatch_dump_warp_timing(
    const dispatch_warp_layout_t& L,
    int num_of_blocks,
    ::ht_ep::dispatch_warp_timing_entry_t* d_wt,
    cudaStream_t stream) {
    const int wt_warps_per_block = L.block_dim / 32;
    const int wt_total = num_of_blocks * wt_warps_per_block;
    char* pmix_rank_str = std::getenv("PMIX_RANK");
    int pmix_rank = pmix_rank_str ? std::atoi(pmix_rank_str) : -1;
    static int iter_count = 0;
    iter_count++;
    if (pmix_rank != 0 || iter_count != 40) return;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<::ht_ep::dispatch_warp_timing_entry_t> h_wt(wt_total);
    CUDA_CHECK(cudaMemcpy(
        h_wt.data(),
        d_wt,
        wt_total * sizeof(::ht_ep::dispatch_warp_timing_entry_t),
        cudaMemcpyDeviceToHost));
    int _wt_clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&_wt_clock_khz, cudaDevAttrClockRate, 0));
    auto _wt_us = [&](long long cycles) { return (double)cycles * 1000.0 / _wt_clock_khz; };
    auto _wt_print_group = [&](const char* name, int warp_start, int warp_count) {
        if (warp_count == 0) return;
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        int n = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            for (int w = warp_start; w < warp_start + warp_count; w++) {
                long long d = h_wt[b * wt_warps_per_block + w].end_clock - h_wt[b * wt_warps_per_block + w].start_clock;
                if (d < mn) mn = d;
                if (d > mx) mx = d;
                sum += d;
                n++;
            }
        }
        std::printf("  %-9s (%d warp%s x %d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n", name, warp_count,
                    warp_count > 1 ? "s" : " ", num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / n));
    };
    auto _wt_print_block_span = [&]() {
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            long long blk_start = LLONG_MAX;
            long long blk_end = 0;
            for (int w = 0; w < wt_warps_per_block; w++) {
                const auto& e = h_wt[b * wt_warps_per_block + w];
                if (e.start_clock < blk_start) blk_start = e.start_clock;
                if (e.end_clock > blk_end) blk_end = e.end_clock;
            }
            long long d = blk_end - blk_start;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            sum += d;
        }
        std::printf("[DISPATCH BLOCK SPAN TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / num_of_blocks));
    };
    std::printf("[DISPATCH WORK WARP TIMING] (%d blocks, %d warps/block, %d pipelines, clock=%d kHz)\n", num_of_blocks,
                wt_warps_per_block, L.num_pipelines, _wt_clock_khz);
    _wt_print_group("INTER_N2N", L.cross_lsa_group_start, L.cross_lsa_group_warps);
    _wt_print_group("INTRA_G2S", L.lsa_g2s_group_start, L.lsa_g2s_group_warps);
    _wt_print_group("INTRA_S2G", L.lsa_s2g_group_start, L.lsa_s2g_group_warps);
    _wt_print_group("MAP", L.map_group_start, L.map_group_warps);
    _wt_print_group("PAD", L.pad_group_start, L.pad_group_warps);
    _wt_print_block_span();
}
#endif

// ============================================================================
// Local dup JIT (NVLink-dedup mode): fan secondaries from primary EM slots.
// ============================================================================
constexpr const char* kLocalDupJitEntryName = "nccl_ep_jit_ht_local_dup_kernel";

inline std::string local_dup_jit_source(
    int hidden_dim,
    int pipe_depth,
    bool forward_dispatch,
    ncclEpDispQuant_t recipe,
    const char* token_type_literal = "uint16_t") {
    std::ostringstream src;
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "using TOKEN_DATA_TYPE = " << token_type_literal << ";\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(64, 1)\n"
        << "__global__ void " << kLocalDupJitEntryName << "(\n"
        << "    const __grid_constant__ ht_ep::local_dup_kernel_param_t<TOKEN_DATA_TYPE> p) {\n"
        << "  ht_ep::local_dup_kernel_impl<\n"
        << "      TOKEN_DATA_TYPE,\n"
        << "      " << hidden_dim << ",\n"
        << "      " << pipe_depth << ",\n"
        << "      " << ::nccl_ep::jit::bool_literal(forward_dispatch) << ",\n"
        << "      " << ::nccl_ep::jit::dispatch_recipe_literal(recipe) << ">(p);\n"
        << "}\n";
    return src.str();
}

template <typename T>
inline void launch_local_dup(
    int hidden_dim,
    int pipe_depth,
    bool forward_dispatch,
    ncclEpDispQuant_t recipe,
    int num_blocks,
    ::ht_ep::local_dup_kernel_param_t<T>& param,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const char* token_type_literal = sizeof(T) == 4 ? "uint32_t" : sizeof(T) == 1 ? "uint8_t" : "uint16_t";
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "local_dup"
             << "_hdim" << hidden_dim << "_pipe" << pipe_depth << "_b" << static_cast<int>(sizeof(T))
             << (forward_dispatch ? "_fwd" : "_bwd")
             << (recipe != NCCL_EP_DISP_QUANT_NONE ? "_sf" : "");
        return name.str();
    }();
    const std::string source = local_dup_jit_source(hidden_dim, pipe_depth, forward_dispatch, recipe, token_type_literal);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_local_dup";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLocalDupJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key =
        (static_cast<std::uint64_t>(hidden_dim) & 0xFFFFFFu) | (static_cast<std::uint64_t>(pipe_depth & 0xFFu) << 24) |
        (static_cast<std::uint64_t>(forward_dispatch ? 1u : 0u) << 32) | (static_cast<std::uint64_t>(sizeof(T)) << 33) |
        (static_cast<std::uint64_t>(static_cast<uint32_t>(recipe)) << 35);
    variant.num_blocks = num_blocks;
    variant.block_dim = 64;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal duplicate JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

// Local permute (dup) JIT: scatter FLAT staging into EM zones.
constexpr const char* kLocalPermuteDupJitEntryName = "nccl_ep_jit_local_permute_dup_kernel";

// Pick HiddenVec so that hidden_int4 % (32 * HiddenVec) == 0 — eliminates the
// dup main-loop tail. Prefer larger HiddenVec for more in-flight LDGs per lane;
// the {8,4,2,1} ladder keeps H=7168 (hidden_int4=896) at HiddenVec=4 while
// promoting H=2048 (hidden_int4=256) and other H % 4096 == 0 cases to 8.
inline int pick_dup_hidden_vec(int hidden_int4) {
    constexpr int kThreads = ::ht_ep::kLocalPermuteThreadsPerSlot;
    for (int v : {8, 4, 2, 1}) {
        if (hidden_int4 % (kThreads * v) == 0) return v;
    }
    return 1;
}

// 2 blocks/SM at small hidden to hide LDG latency; grid bumped 2x in caller.
inline int pick_dup_blocks_per_sm(int hidden_int4) {
    return (hidden_int4 <= 256) ? 2 : 1;
}

// Pull dispatch blocks/SM. Decoupled from pick_dup_blocks_per_sm (different reason): pull is
// bound by in-flight TMAs per SM (one per pull warp). At small hidden the pull-warp count hits
// the kPullDispatchMaxPullWarps cap below what smem allows, so a 2nd CTA doubles the rows in
// flight and the tiny rows need it to fill NVLink (measured ~29% faster at h=3072 MXFP8). At
// large hidden smem caps the warps and 24 big rows already saturate, so 1 CTA wins.
inline int pick_pull_blocks_per_sm(int hidden_int4) {
    return (hidden_int4 <= 256) ? 2 : 1;
}

inline std::string local_permute_dup_jit_source(int hidden_int4, int hidden_vec, int blocks_per_sm,
                                                ncclEpDispQuant_t recipe, bool flat_expert_ids) {
    std::ostringstream src;
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << ::ht_ep::kLocalPermuteThreads << ", " << blocks_per_sm << ")\n"
        << "__global__ void " << kLocalPermuteDupJitEntryName << "(\n"
        << "    const __grid_constant__ ::ht_ep::local_permute_dup_param_t p) {\n"
        << "  ::ht_ep::local_permute_dup<" << hidden_int4 << ", " << hidden_vec
        << ", " << ::nccl_ep::jit::dispatch_recipe_literal(recipe)
        << ", " << ::nccl_ep::jit::bool_literal(flat_expert_ids) << ">(\n"
        << "      reinterpret_cast<uint8_t*>(p.recv_x_em),\n"
        << "      p.recv_topk_weights_em,\n"
        << "      reinterpret_cast<const uint8_t*>(p.flat_staging),\n"
        << "      p.recv_topk_weights_flat,\n"
        << "      p.flat2em_slot_map,\n"
        << "      p.num_recv_tokens_dev,\n"
        << "      p.expert_token_offsets,\n"
        << "      p.per_expert_counts_active,\n"
        << "      p.top_k,\n"
        << "      p.experts_per_rank,\n"
        << "      p.row_bytes,\n"
        << "      p.caller_num_recv_tokens,\n"
        << "      reinterpret_cast<uint8_t*>(p.recv_scales_em),\n"
        << "      reinterpret_cast<const uint8_t*>(p.flat_scale_staging),\n"
        << "      p.scale_row_bytes,\n"
        << "      p.flat2em_out,\n"
        << "      p.lerm_in,\n"
        << "      p.lerm_out,\n"
        << "      p.em_permute_cursors,\n"
        << "      p.allow_overflow_drop);\n"
        << "}\n";
    return src.str();
}

inline void
launch_local_permute_dup(int num_blocks, ::ht_ep::local_permute_dup_param_t& param,
                         ncclEpDispQuant_t recipe, bool flat_expert_ids, cudaStream_t stream) {
    static const int variant_identity = 0;
    assert((param.row_bytes % 16) == 0);
    const int hidden_int4 = param.row_bytes / 16;
    const int hidden_vec = pick_dup_hidden_vec(hidden_int4);
    const int blocks_per_sm = pick_dup_blocks_per_sm(hidden_int4);
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "local_permute_dup_h" << hidden_int4 << "_v" << hidden_vec << "_b" << blocks_per_sm
             << (recipe != NCCL_EP_DISP_QUANT_NONE ? "_sf" : "")
             << (flat_expert_ids ? "_topk" : "_map");
        return name.str();
    }();
    const std::string source = local_permute_dup_jit_source(hidden_int4, hidden_vec, blocks_per_sm, recipe, flat_expert_ids);

    // Dynamic smem: zero-fill buffer for the pad warp's cp.async.bulk S2G.
    // Must cover both token rows and scale rows (scale_row_bytes may exceed row_bytes).
    const int row_bytes_aligned       = ((param.row_bytes       + 15) >> 4) << 4;
    const int scale_row_bytes_aligned = ((param.scale_row_bytes + 15) >> 4) << 4;
    const int zero_smem_bytes = std::max(row_bytes_aligned, scale_row_bytes_aligned);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "local_permute_dup";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLocalPermuteDupJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = (static_cast<std::uint64_t>(hidden_int4) << 16) |
                          (static_cast<std::uint64_t>(hidden_vec) << 8) | static_cast<std::uint64_t>(blocks_per_sm) |
                          (static_cast<std::uint64_t>(static_cast<uint32_t>(recipe)) << 32) |
                          (static_cast<std::uint64_t>(flat_expert_ids) << 40);
    variant.num_blocks = num_blocks * blocks_per_sm;
    variant.block_dim = ::ht_ep::kLocalPermuteThreads;
    variant.dynamic_smem_bytes = zero_smem_bytes;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal local-permute-dup JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

// ============================ Pull EM dispatch =============================
// Read source rows over NVLink and scatter into EM zones.
constexpr const char* kDispatchPullJitEntryName = "nccl_ep_jit_ht_dispatch_pull_kernel";

// Transfer policy: TMA (cp.async.bulk) read into a per-warp smem row + TMA store to each EM slot.
inline std::string pull_stage_literal(int hidden_int4) {
    std::ostringstream t;
    t << "::ht_ep::PullTmaStage<" << (hidden_int4 * 16) << ">";
    return t.str();
}

// Per-pull-warp staging: token row buffer + mbarrier, plus the scale-row region when
// the QUANT_FWD scale row is moved by TMA (scale_tma_row_bytes != 0).
inline int pull_smem_bytes_per_warp(int hidden_int4, int scale_tma_row_bytes = 0) {
    const int per_warp = hidden_int4 * 16 + 16; // row buffer + mbarrier
    const int scale_per_warp = scale_tma_row_bytes ? (scale_tma_row_bytes + 16) : 0;
    return per_warp + scale_per_warp;
}

// Static (non-dynamic) smem the pull kernel declares per pull warp: the s_active slot
// list plus its count. It shares the device cap with the dynamic staging, so the warp
// count has to be solved against the sum, not against the staging alone.
inline int pull_static_smem_bytes_per_warp() {
    return ::ht_ep::kPullDispatchMaxActive * static_cast<int>(sizeof(std::int32_t)) +
           static_cast<int>(sizeof(int));
}

// Largest pull-warp count (payload + map_only_warps) whose smem fits the device cap. Each
// payload warp holds one remote row in flight, so more warps is strictly better until the
// cap. blocks_per_sm CTAs must be co-resident, so each gets 1/blocks_per_sm of the cap.
// map_only_warps/map_smem_bytes (fused pull-count only) are the fixed MAP-warp-group
// warp count and its smem (layout scratch + chunk-ready counters), taken off the top.
inline int pull_dispatch_warps(int hidden_int4, int scale_tma_row_bytes, size_t smem_optin,
                               int blocks_per_sm, size_t map_smem_bytes, int map_only_warps,
                               bool flat) {
    const size_t total_budget = smem_optin / static_cast<size_t>(std::max(1, blocks_per_sm));
    const size_t counter_bytes = map_only_warps > 0
        ? (::ht_ep::kPullDispatchMapWarps + 1) * sizeof(unsigned) +
          ::ht_ep::kPullDispatchMapScanBytes : 0;
    for (int warps = ::ht_ep::kPullDispatchMaxPullWarps; warps > map_only_warps; --warps) {
        const int payload_warps = warps - map_only_warps;
        size_t static_bytes = payload_warps * pull_static_smem_bytes_per_warp() + counter_bytes;
        if (flat) {
            const int scratch_warps = map_only_warps > 0 ? map_only_warps : payload_warps;
            static_bytes = (scratch_warps + 4) * sizeof(int);
        }
        static_bytes = nccl_ep::align<size_t>(static_bytes, 16);
        const size_t dynamic_bytes = payload_warps * pull_smem_bytes_per_warp(hidden_int4, scale_tma_row_bytes) +
                                      map_smem_bytes;
        if (static_bytes + dynamic_bytes <= total_budget) return warps;
    }
    return 0;
}

inline int pull_dynamic_smem_bytes(int hidden_int4, int scale_tma_row_bytes, int pull_warps) {
    return pull_warps * pull_smem_bytes_per_warp(hidden_int4, scale_tma_row_bytes);
}

// Device opt-in shared-memory cap, queried once per device. Falls back to the 48 KB
// architectural floor if the query fails, so a transient CUDA error degrades to a
// smaller warp count instead of the 1-warp minimum.
inline size_t pull_device_smem_optin() {
    static constexpr size_t kSmemOptinFallback = 48 * 1024;
    static thread_local int cached_dev = -1;
    static thread_local size_t cached_cap = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return kSmemOptinFallback;
    if (dev != cached_dev) {
        int cap = 0;
        if (cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
            cap <= 0) {
            return kSmemOptinFallback;
        }
        cached_cap = static_cast<size_t>(cap);
        cached_dev = dev;
    }
    return cached_cap;
}

// Non-zero => move the QUANT_FWD scale row by TMA through smem instead of a register-held
// ld/st pair. Requires a 16B-multiple row (cp.async.bulk granularity) and enough smem.
inline int pull_scale_tma_row_bytes(int hidden_int4, ncclEpDispQuant_t recipe, int scale_row_bytes,
                                    size_t smem_optin, int blocks_per_sm) {
    if (recipe != NCCL_EP_DISP_QUANT_FWD) return 0;
    if (scale_row_bytes <= 0 || (scale_row_bytes % 16) != 0) return 0;
    const size_t one_warp = static_cast<size_t>(pull_smem_bytes_per_warp(hidden_int4, scale_row_bytes) +
                                                pull_static_smem_bytes_per_warp());
    if (one_warp * static_cast<size_t>(std::max(1, blocks_per_sm)) > smem_optin) return 0;
    return scale_row_bytes;
}

inline std::string dispatch_pull_jit_source(int hidden_int4, int blocks_per_sm, ncclEpDispQuant_t recipe,
                                               int scale_int4_per_lane, int scale_tma_row_bytes,
                                               int pull_warps, bool fused_map,
                                               int top_k, int nRanks, int experts_per_rank, bool flat,
                                               bool overlap_map) {
    std::ostringstream src;
    const int threads = flat ? pull_warps * 32 : ::ht_ep::pull_dispatch_threads(pull_warps);
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << threads << ", " << blocks_per_sm << ")\n"
        << "__global__ void " << kDispatchPullJitEntryName << "(\n"
        << "    const __grid_constant__ ::ht_ep::dispatch_pull_param_t p) {\n";
    if (flat) {
        src << "  ::ht_ep::dispatch_pull_flat<" << hidden_int4 << ", "
            << ::nccl_ep::jit::dispatch_recipe_literal(recipe) << ", " << scale_int4_per_lane
            << ", " << scale_tma_row_bytes << ", " << pull_warps << ", "
            << ::nccl_ep::jit::bool_literal(fused_map) << ", " << top_k << ", "
            << kFlatMetadataChunkTokens << ", " << ::nccl_ep::jit::bool_literal(overlap_map) << ">(p);\n}\n";
        return src.str();
    }
    src << "  ::ht_ep::dispatch_pull<" << hidden_int4 << ", " << pull_stage_literal(hidden_int4)
        << ", " << ::nccl_ep::jit::dispatch_recipe_literal(recipe) << ", " << scale_int4_per_lane
        << ", " << scale_tma_row_bytes << ", " << pull_warps
        << ", " << (fused_map ? "true" : "false") << ">(\n"
        << "      p,\n"
        << "      " << (fused_map ? std::to_string(top_k) : "p.top_k") << ",\n"
        << "      " << (fused_map ? std::to_string(experts_per_rank) : "p.experts_per_rank") << ",\n"
        << "      " << (fused_map ? std::to_string(nRanks) : "p.lsa_team_size") << ");\n"
        << "}\n";
    return src.str();
}

inline ncclResult_t
launch_dispatch_pull_variant(int num_blocks, int blocks_per_sm, ::ht_ep::dispatch_pull_param_t& param,
                             ncclEpDispQuant_t recipe, cudaStream_t stream, bool nucs, bool* retry_residency) {
    static const int variant_identity = 0;
    assert((param.row_bytes % 16) == 0);
    const int hidden_int4 = param.row_bytes / 16;
    // Per-lane int4 capacity for the FWD scale row, sized from the actual scale row so
    // any scale dtype / quantization block size fits (1 for NONE, where scale_row_bytes==0).
    const int scale_int4_per_lane = std::max(1, ((param.scale_row_bytes / 16) + 31) / 32);
    const size_t smem_optin = pull_device_smem_optin();
    const int scale_tma_row_bytes = pull_scale_tma_row_bytes(
        hidden_int4, recipe, param.scale_row_bytes, smem_optin, blocks_per_sm);
    // layout_ready != nullptr is what selects the fused pull-count MAP-warp-group variant;
    // the adapter (launch_dispatch_pull) sets every fused-count field together with it.
    const bool fused_map = param.layout_ready != nullptr;
    const size_t blocks = static_cast<size_t>(num_blocks) * blocks_per_sm;
    const size_t flat_chunks = static_cast<size_t>(param.lsa_team_size) *
        nccl_ep::ceil_div(param.tokens_per_rank, ::nccl_ep::ht::kFlatMetadataChunkTokens);
    // A single wave of flat chunks favors cooperative mapping. With more work,
    // dedicated producers hide map construction behind the payload transfers.
    bool overlap_map = fused_map && param.flat && flat_chunks > blocks;
    int map_warps = fused_map && (!param.flat || overlap_map) ? ::ht_ep::kPullDispatchMapWarps : 0;
    size_t map_smem_bytes = fused_map && !param.flat
        ? ::ht_ep::dispatch_pull_map_smem_ints(param.lsa_team_size, param.experts_per_rank) * sizeof(std::int32_t)
        : 0;
    if (overlap_map) {
        const size_t tiles_per_chunk = ::nccl_ep::ht::kFlatMetadataChunkTokens / (map_warps * 32);
        map_smem_bytes = nccl_ep::ceil_div(flat_chunks, blocks) * tiles_per_chunk * sizeof(int2);
    } else if (fused_map && !param.flat) {
        // CTA0's per-chunk (int2 range) buffer sits after the layout scratch; sized for the
        // worst-case per-CTA chunk count so every MAP warp's chunk_ranges write stays in bounds.
        // Folded into map_smem_bytes before the warp-count search below, or that search would
        // pick a warp count sized only for the layout scratch and leave no room for this buffer.
        const size_t total_chunks = static_cast<size_t>(param.lsa_team_size) * nccl_ep::ceil_div(param.tokens_per_rank, 32);
        const size_t chunk_stride = static_cast<size_t>(num_blocks) * blocks_per_sm * map_warps;
        const size_t max_cta_chunks = (total_chunks / chunk_stride) * map_warps +
            std::min(total_chunks % chunk_stride, static_cast<size_t>(map_warps));
        map_smem_bytes = ((map_smem_bytes + alignof(int2) - 1) & ~size_t(alignof(int2) - 1)) +
            max_cta_chunks * sizeof(int2);
    }
    int pull_warps = pull_dispatch_warps(hidden_int4, scale_tma_row_bytes, smem_optin, blocks_per_sm,
                                              map_smem_bytes, map_warps, param.flat);
    if (pull_warps == 0 && overlap_map) {
        overlap_map = false;
        map_warps = 0;
        map_smem_bytes = 0;
        pull_warps = pull_dispatch_warps(hidden_int4, scale_tma_row_bytes, smem_optin, blocks_per_sm,
                                         map_smem_bytes, map_warps, param.flat);
    }
    if (pull_warps == 0) {
        std::fprintf(stderr, "[nccl_ep jit] dispatch-pull shared memory exceeds the device budget\n");
        return ncclInvalidArgument;
    }

    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "dispatch_pull_h" << hidden_int4 << "_b" << blocks_per_sm
             << "_r" << static_cast<int>(recipe) << "_s" << scale_int4_per_lane
             << "_st" << scale_tma_row_bytes << "_w" << pull_warps
             << "_fm" << (fused_map ? 1 : 0) << (param.flat ? "_flat" : "");
        if (fused_map || param.flat) {
            name << "_k" << param.top_k;
        }
        if (fused_map && !param.flat) {
            name << "_nr" << param.lsa_team_size << "_e" << param.experts_per_rank;
        }
        if (fused_map) {
            name << "_cm" << map_smem_bytes;
        }
        if (param.flat) {
            name << "_om" << overlap_map;
        }
        return name.str();
    }();
    const std::string source = dispatch_pull_jit_source(hidden_int4, blocks_per_sm, recipe, scale_int4_per_lane,
                                                        scale_tma_row_bytes, pull_warps, fused_map, param.top_k,
                                                        param.lsa_team_size, param.experts_per_rank, param.flat, overlap_map);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_dispatch_pull";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kDispatchPullJitEntryName;
    variant.identity = &variant_identity;
    // Hash the variant name: it already carries every specialization parameter, and
    // the cache key is (identity, runtime_key, context, device) only - a hand-packed
    // bitfield would silently alias once a field (e.g. scale_tma_row_bytes) overflows.
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = num_blocks * blocks_per_sm;
    variant.block_dim = param.flat ? pull_warps * 32 : ::ht_ep::pull_dispatch_threads(pull_warps);
    const int payload_warps = pull_warps - map_warps;
    variant.dynamic_smem_bytes = pull_dynamic_smem_bytes(hidden_int4, scale_tma_row_bytes, payload_warps) +
        static_cast<int>(map_smem_bytes);
    variant.nucs = nucs;
    variant.require_full_residency = fused_map && !param.unfused_sync;
    variant.cooperative = variant.require_full_residency;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);
    if (status == ::nccl_ep::jit::JitKernelStatus::kInsufficientResidency && blocks_per_sm > 1) {
        *retry_residency = true;
        return ncclInvalidUsage;
    }

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        // The oversize-hidden smem case (kAttributeFailed) is pre-rejected in ncclEpDispatch
        // via dispatch_pull_smem_bytes, so a failure here is unexpected (compile/driver
        // error); surface it to the caller instead of aborting the process.
        // TODO: add a graceful fallback (local-permute dispatch, or a cap-reduced warp count).
        std::fprintf(stderr, "[nccl_ep jit] dispatch-pull JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

inline ncclResult_t launch_dispatch_pull(
    int num_blocks, ::ht_ep::dispatch_pull_param_t& param, ncclEpDispQuant_t recipe,
    cudaStream_t stream, bool nucs) {
    const int blocks_per_sm = pick_pull_blocks_per_sm(param.row_bytes / 16);
    bool retry_residency = false;
    ncclResult_t result = launch_dispatch_pull_variant(
        num_blocks, blocks_per_sm, param, recipe, stream, nucs, &retry_residency);
    if (retry_residency) {
        retry_residency = false;
        result = launch_dispatch_pull_variant(num_blocks, 1, param, recipe, stream, nucs, &retry_residency);
    }
    return result;
}

} // namespace jit
} // namespace ht
} // namespace nccl_ep
