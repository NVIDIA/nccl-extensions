/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ht_ep.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_utils.hpp"
#include "device/jit/jit_source_literals.hpp"
#include "nccl_ep_env.h"

#include <cassert>
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

constexpr const char* kCombineJitEntryName = "nccl_ep_jit_ht_combine_kernel";

struct combine_warp_layout_t {
    int lsa_red_group_warps;
    int lsa_red_group_start;
    int cross_lsa_red_group_warps;
    int cross_lsa_red_group_start;
    int lsa_g2s_group_warps;
    int lsa_g2s_group_start;
    int cross_lsa_g2s_group_warps;
    int cross_lsa_g2s_group_start;
    int cross_lsa_rdma_group_warps;
    int cross_lsa_rdma_group_start;
    int num_of_data_pipeline_per_block;
    int block_dim;
};

inline combine_warp_layout_t compute_combine_warp_layout(int num_lsa_teams, int num_pipelines) {
    const bool multi_lsa_layout = (num_lsa_teams != 1);
    combine_warp_layout_t L{};
    L.lsa_red_group_warps = multi_lsa_layout ? NCCL_EP_HT_COMBINE_RED_WARPS : 0;
    L.cross_lsa_red_group_warps = NCCL_EP_HT_COMBINE_RED_WARPS;
    L.lsa_g2s_group_warps = multi_lsa_layout ? 1 : 0;
    L.cross_lsa_g2s_group_warps = num_pipelines;
    L.cross_lsa_rdma_group_warps = multi_lsa_layout ? NCCL_EP_HT_COMBINE_N2N_WARPS : 0;
    L.num_of_data_pipeline_per_block = num_pipelines;

    int warp_cursor = 0;
    L.lsa_red_group_start = warp_cursor;
    warp_cursor += L.lsa_red_group_warps;
    L.cross_lsa_red_group_start = warp_cursor;
    warp_cursor += L.cross_lsa_red_group_warps;
    L.lsa_g2s_group_start = warp_cursor;
    warp_cursor += L.lsa_g2s_group_warps;
    L.cross_lsa_g2s_group_start = warp_cursor;
    warp_cursor += L.cross_lsa_g2s_group_warps;
    L.cross_lsa_rdma_group_start = warp_cursor;
    warp_cursor += L.cross_lsa_rdma_group_warps;
    L.block_dim = 32 * warp_cursor;
    return L;
}

inline std::string combine_jit_source(
    const combine_warp_layout_t& L,
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_group,
    int num_of_tokens_per_chunk,
    int max_tokens_per_rank,
    int num_lsa_teams,
    int num_of_blocks,
    bool backward_combine,
    int lsa_team_size,
    ncclEpLayout_t layout,
    int hidden_dim,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t recipe) {
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(token_dtype);
    const char* recipe_literal = ::nccl_ep::jit::combine_recipe_literal(recipe);
    std::ostringstream src;
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "using LSA_RED_GROUP = ht_ep::warp_group<" << L.lsa_red_group_warps << ", "
        << L.lsa_red_group_start << ">;\n"
        << "using CROSS_LSA_RED_GROUP = ht_ep::warp_group<" << L.cross_lsa_red_group_warps << ", "
        << L.cross_lsa_red_group_start << ">;\n"
        << "using LSA_G2S_GROUP = ht_ep::warp_group<" << L.lsa_g2s_group_warps << ", "
        << L.lsa_g2s_group_start << ">;\n"
        << "using CROSS_LSA_G2S_GROUP = ht_ep::warp_group<" << L.cross_lsa_g2s_group_warps << ", "
        << L.cross_lsa_g2s_group_start << ">;\n"
        << "using GIN_GROUP = ht_ep::warp_group<" << L.cross_lsa_rdma_group_warps << ", "
        << L.cross_lsa_rdma_group_start << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(LSA_RED_GROUP::size() + CROSS_LSA_RED_GROUP::size() + "
           "LSA_G2S_GROUP::size() + CROSS_LSA_G2S_GROUP::size() + GIN_GROUP::size(), 1)\n"
        << "__global__ void " << kCombineJitEntryName << "(\n"
        << "    const __grid_constant__ ht_ep::combine_kernel_param_t<" << lsa_team_size << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  ht_ep::combine_kernel_impl<\n"
        << "      LSA_RED_GROUP,\n"
        << "      CROSS_LSA_RED_GROUP,\n"
        << "      LSA_G2S_GROUP,\n"
        << "      CROSS_LSA_G2S_GROUP,\n"
        << "      GIN_GROUP,\n"
        << "      " << L.num_of_data_pipeline_per_block << ",\n"
        << "      " << num_of_stages_g2s << ",\n"
        << "      " << num_of_stages_s2g << ",\n"
        << "      " << num_of_tokens_per_group << ",\n"
        << "      " << num_of_tokens_per_chunk << ",\n"
        << "      " << max_tokens_per_rank << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << ::nccl_ep::jit::bool_literal(backward_combine) << ",\n"
        << "      " << hidden_dim << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << token_dtype_literal << ",\n"
        << "      " << recipe_literal << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

inline ncclResult_t launch_combine(
    const ::ht_ep::combine_config_t& config,
    int max_tokens_per_rank,
    int num_lsa_teams,
    int lsa_team_size,
    ncclEpLayout_t layout,
    int hidden_dim,
    const ncclEpEnvConfig* env,  // for rank-0-gated verbose param dump; may be null
    void* param, // ptr to the packed kernel arguments buffer
    size_t param_size, // size of packed kernel arguments buffer
    int dynamic_smem_bytes,
    cudaStream_t stream,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t recipe) {
    const combine_warp_layout_t L =
        compute_combine_warp_layout(num_lsa_teams, config.num_pipelines);

    static const int fwd_variant_identity = 0;
    static const int bwd_variant_identity = 0;
    const int& variant_identity = config.backward_combine ? bwd_variant_identity : fwd_variant_identity;
    // combine_recipe_literal() returns null for a recipe with no enumerator name; the source
    // builders stream it straight into an ostringstream, where null would be UB.
    if (::nccl_ep::jit::combine_recipe_literal(recipe) == nullptr) {
        std::fprintf(stderr, "[nccl_ep jit] combine: unsupported HT combine recipe %d\n", static_cast<int>(recipe));
        return ncclInvalidArgument;
    }
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "combine"
             << "_LSATeams" << num_lsa_teams << "_lsa" << lsa_team_size << "_hdim" << hidden_dim << "_g2s"
             << config.num_of_stages_g2s << "_s2g" << config.num_of_stages_s2g << "_pipe"
             << config.num_pipelines << "_chunk" << config.num_of_tokens_per_chunk << "_maxt"
             << max_tokens_per_rank << "_group" << config.num_of_tokens_per_group << "_blocks"
             << config.num_of_blocks
             << (config.backward_combine ? "_bwd" : "_fwd")
             << ::nccl_ep::jit::layout_name_tag(layout)
             << ::nccl_ep::jit::token_dtype_name_tag(token_dtype)
             << ::nccl_ep::jit::combine_recipe_name_tag(recipe);
        return name.str();
    }();
    const std::string source = combine_jit_source(
        L,
        config.num_of_stages_g2s,
        config.num_of_stages_s2g,
        config.num_of_tokens_per_group,
        config.num_of_tokens_per_chunk,
        max_tokens_per_rank,
        num_lsa_teams,
        config.num_of_blocks,
        config.backward_combine,
        lsa_team_size,
        layout,
        hidden_dim,
        token_dtype,
        recipe);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_combine";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kCombineJitEntryName;
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
            "[nccl_ep][env] HT combine kernel (%s):\n"
            "[nccl_ep][env]   nodes(lsa_teams)=%d lsa_team_size=%d hidden_dim=%d\n"
            "[nccl_ep][env]   stages_g2s=%d stages_s2g=%d pipelines=%d\n"
            "[nccl_ep][env]   tokens_per_chunk=%d tokens_per_group=%d max_tokens_per_rank=%d\n"
            "[nccl_ep][env]   num_blocks=%d block_dim=%d dynamic_smem_bytes=%d\n"
            "[nccl_ep][env]   backward=%s layout=%s dtype=%s\n",
            variant_name.c_str(),
            num_lsa_teams,
            lsa_team_size,
            hidden_dim,
            config.num_of_stages_g2s,
            config.num_of_stages_s2g,
            L.num_of_data_pipeline_per_block,
            config.num_of_tokens_per_chunk,
            config.num_of_tokens_per_group,
            max_tokens_per_rank,
            config.num_of_blocks,
            L.block_dim,
            dynamic_smem_bytes,
            (config.backward_combine ? "true" : "false"),
            (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "EXPERT_MAJOR" : "FLAT"),
            (token_dtype == ncclFloat32 ? "fp32" :
             token_dtype == ncclFloat16 ? "fp16" :
                                          "bf16"));
    }

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, param, param_size, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] combine JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

#ifdef NCCL_EP_HT_ENABLE_WARP_TIMING
inline void combine_dump_warp_timing(
    const combine_warp_layout_t& L,
    int num_of_blocks,
    ::ht_ep::combine_warp_timing_entry_t* d_wt,
    ::ht_ep::combine_block_timing_entry_t* d_bt,
    cudaStream_t stream) {
    const int wt_warps_per_block = L.block_dim / 32;
    const int wt_total = num_of_blocks * wt_warps_per_block;
    char* pmix_rank_str = std::getenv("PMIX_RANK");
    int pmix_rank = pmix_rank_str ? std::atoi(pmix_rank_str) : -1;
    static int iter_count = 0;
    iter_count++;
    if (pmix_rank != 0 || iter_count != 40) return;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<::ht_ep::combine_warp_timing_entry_t> h_wt(wt_total);
    std::vector<::ht_ep::combine_block_timing_entry_t> h_bt(num_of_blocks);
    CUDA_CHECK(cudaMemcpy(
        h_wt.data(),
        d_wt,
        wt_total * sizeof(::ht_ep::combine_warp_timing_entry_t),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_bt.data(),
        d_bt,
        num_of_blocks * sizeof(::ht_ep::combine_block_timing_entry_t),
        cudaMemcpyDeviceToHost));
    int _wt_clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&_wt_clock_khz, cudaDevAttrClockRate, 0));
    auto _wt_us = [&](long long cycles) { return (double)cycles * 1000.0 / _wt_clock_khz; };
    auto _wt_print_head_sync = [&]() {
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            long long d = h_bt[b].head_sync_end_clock - h_bt[b].head_sync_start_clock;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            sum += d;
        }
        std::printf("[COMBINE HEAD SYNC TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / num_of_blocks));
    };
    auto _wt_print_work_group = [&](const char* name, int warp_start, int warp_count) {
        if (warp_count == 0) return;
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        int n = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            for (int w = warp_start; w < warp_start + warp_count; w++) {
                long long d =
                    h_wt[b * wt_warps_per_block + w].work_end_clock - h_wt[b * wt_warps_per_block + w].work_start_clock;
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
                if (e.work_start_clock < blk_start) blk_start = e.work_start_clock;
                if (e.work_end_clock > blk_end) blk_end = e.work_end_clock;
            }
            long long d = blk_end - blk_start;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            sum += d;
        }
        std::printf("[COMBINE BLOCK SPAN TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / num_of_blocks));
    };
    _wt_print_head_sync();
    std::printf("[COMBINE WORK WARP TIMING] (%d blocks, %d warps/block, %d pipelines, clock=%d kHz)\n", num_of_blocks,
                wt_warps_per_block, L.num_of_data_pipeline_per_block, _wt_clock_khz);
    _wt_print_work_group("INTRA_RED", L.lsa_red_group_start, L.lsa_red_group_warps);
    _wt_print_work_group("INTER_RED", L.cross_lsa_red_group_start, L.cross_lsa_red_group_warps);
    _wt_print_work_group("INTRA_G2S", L.lsa_g2s_group_start, L.lsa_g2s_group_warps);
    _wt_print_work_group("INTER_G2S", L.cross_lsa_g2s_group_start, L.cross_lsa_g2s_group_warps);
    _wt_print_work_group("INTER_N2N", L.cross_lsa_rdma_group_start, L.cross_lsa_rdma_group_warps);
    _wt_print_block_span();
}
#endif

// ============================================================================
// Local reduce JIT (NVLink-dedup mode): cooperative reduction across local
// EM slots that share a primary token.
// ============================================================================
constexpr const char* kLocalReduceJitEntryName = "nccl_ep_jit_ht_local_reduce_kernel";

constexpr int kLocalReduceBlockDim = 128;

inline std::string local_reduce_jit_source(
    int hidden_dim,
    bool backward_combine,
    int experts_per_rank,
    ncclDataType_t token_dtype = ncclBfloat16) {
    // Param/sizeof type collapses FP16->uint16_t (layout-identical); the decode
    // template arg keeps the real dtype so the reduce math is correct.
    const char* token_type_literal = (token_dtype == ncclFloat32) ? "uint32_t" : "uint16_t";
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(token_dtype);
    std::ostringstream src;
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "using TOKEN_DATA_TYPE = " << token_type_literal << ";\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << kLocalReduceBlockDim << ", 1)\n"
        << "__global__ void " << kLocalReduceJitEntryName << "(\n"
        << "    const __grid_constant__ ht_ep::local_reduce_kernel_param_t<TOKEN_DATA_TYPE> p) {\n"
        << "  ht_ep::local_reduce_kernel_impl<\n"
        << "      TOKEN_DATA_TYPE,\n"
        << "      " << hidden_dim << ",\n"
        << "      " << kLocalReduceBlockDim << ",\n"
        << "      " << ::nccl_ep::jit::bool_literal(backward_combine) << ",\n"
        << "      " << token_dtype_literal << ",\n"
        << "      " << experts_per_rank << ">(p);\n"
        << "}\n";
    return src.str();
}

template <typename T>
inline void launch_local_reduce(
    int hidden_dim,
    bool backward_combine,
    int experts_per_rank,
    int num_blocks,
    ::ht_ep::local_reduce_kernel_param_t<T>& param,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "local_reduce"
             << "_hdim" << hidden_dim << "_epr" << experts_per_rank << (backward_combine ? "_bwd" : "_fwd")
             << ::nccl_ep::jit::token_dtype_name_tag(token_dtype);
        return name.str();
    }();
    const std::string source = local_reduce_jit_source(hidden_dim, backward_combine, experts_per_rank, token_dtype);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_local_reduce";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLocalReduceJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = (static_cast<std::uint64_t>(hidden_dim) & 0xFFFFFFu) |
                          (static_cast<std::uint64_t>(experts_per_rank & 0xFFu) << 24) |
                          (static_cast<std::uint64_t>(backward_combine ? 1u : 0u) << 32) |
                          (static_cast<std::uint64_t>(token_dtype) << 33);
    variant.num_blocks = num_blocks;
    variant.block_dim = kLocalReduceBlockDim;
    variant.dynamic_smem_bytes = ::ht_ep::local_reduce_dynamic_smem_bytes(hidden_dim, static_cast<int>(sizeof(T)));

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal local_reduce JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

// ============================================================================
// Local permute (reduce) JIT: gather caller's EM combine input into FLAT
// staging by summing the top_k EM rows per FLAT slot. JIT'd per top_k so the
// per-pair k loop unrolls into a compile-time bound.
// ============================================================================
constexpr const char* kLocalPermuteReduceJitEntryName = "nccl_ep_jit_local_permute_reduce_kernel";

// 2 blocks/SM at small hidden; reg pressure fits without spilling. Grid bumped 2x in caller.
// MXFP8 does not change this budget: the quantize epilogue only adds per-iteration temporaries.
inline int pick_reduce_blocks_per_sm(int hidden_int4) {
    return (hidden_int4 <= 256) ? 2 : ::ht_ep::kLocalPermuteReduceBlocksPerSM;
}

inline std::string local_permute_reduce_jit_source(
    int top_k,
    int hidden_int4,
    int blocks_per_sm,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t recipe) {
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(token_dtype);
    const char* recipe_literal = ::nccl_ep::jit::combine_recipe_literal(recipe);
    std::ostringstream src;
    src << "#include \"device/ht_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << ::ht_ep::kLocalPermuteReduceThreads << ", " << blocks_per_sm
        << ")\n"
        << "__global__ void " << kLocalPermuteReduceJitEntryName << "(\n"
        << "    const __grid_constant__ ::ht_ep::local_permute_reduce_param_t p) {\n"
        << "  ::ht_ep::local_permute_reduce<" << top_k << ", " << hidden_int4 << ", " << token_dtype_literal
        << ", " << recipe_literal << ">(\n"
        << "      reinterpret_cast<uint8_t*>(p.flat_staging),\n"
        << "      reinterpret_cast<const uint8_t*>(p.recv_x_em),\n"
        << "      p.flat2em_slot_map,\n"
        << "      p.num_recv_tokens_dev,\n"
        << "      p.em_weights_in,\n"
        << "      p.flat_weights_out,\n"
        << "      p.top_k,\n"
        << "      p.row_bytes,\n"
        << "      p.caller_num_recv_tokens);\n"
        << "}\n";
    return src.str();
}

inline ncclResult_t launch_local_permute_reduce(
    int top_k,
    int input_row_bytes,
    int num_blocks,
    ::ht_ep::local_permute_reduce_param_t& param,
    cudaStream_t stream,
    ncclDataType_t token_dtype,
    ncclEpCombQuant_t recipe) {
    static const int variant_identity = 0;
    // The caller passes the row the kernel reads, so hidden_int4 needs no per-recipe
    // reasoning: the packed output width is the adapter's business, not this launcher's.
    if ((input_row_bytes % 16) != 0) {
        std::fprintf(stderr,
                     "[nccl_ep jit] local-permute-reduce: %d B input row is not 16B-aligned\n",
                     input_row_bytes);
        return ncclInternalError;
    }
    // Same null contract as launch_combine: the source builder streams the literal directly.
    if (::nccl_ep::jit::combine_recipe_literal(recipe) == nullptr) {
        std::fprintf(
            stderr, "[nccl_ep jit] local-permute-reduce: unsupported combine recipe %d\n", static_cast<int>(recipe));
        return ncclInvalidArgument;
    }
    const int hidden_int4 = input_row_bytes / 16;
    const int blocks_per_sm = pick_reduce_blocks_per_sm(hidden_int4);
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "local_permute_reduce_topk" << top_k << "_h" << hidden_int4 << "_b" << blocks_per_sm
             << ::nccl_ep::jit::token_dtype_name_tag(token_dtype)
             << ::nccl_ep::jit::combine_recipe_name_tag(recipe);
        return name.str();
    }();
    const std::string source = local_permute_reduce_jit_source(top_k, hidden_int4, blocks_per_sm, token_dtype, recipe);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "local_permute_reduce";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLocalPermuteReduceJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = (static_cast<std::uint64_t>(recipe) << 56) |
                          (static_cast<std::uint64_t>(token_dtype) << 48) |
                          (static_cast<std::uint64_t>(hidden_int4) << 32) |
                          (static_cast<std::uint64_t>(blocks_per_sm) << 16) | static_cast<std::uint64_t>(top_k);
    variant.num_blocks = num_blocks * blocks_per_sm;
    variant.block_dim = ::ht_ep::kLocalPermuteReduceThreads;
    variant.dynamic_smem_bytes = 0;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] local-permute-reduce JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

// ======================== Push EM combine + reduce =========================
// Local gather-reduce + remote push (head/tail intra-LSA syncs fused in) and the final
// team_size reduce. See ht_ep.cuh combine_push / combine_reduce.
constexpr const char* kCombinePushJitEntryName = "nccl_ep_jit_ht_combine_push_kernel";
constexpr const char* kCombineReduceJitEntryName = "nccl_ep_jit_ht_combine_epi_reduce_kernel";

inline ncclResult_t launch_combine_push(
    int top_k,
    int row_bytes,
    int num_blocks,
    int ep_size,
    ::ht_ep::combine_push_param_t& param,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16,
    bool backward = false,
    bool nucs = false) {
    static const int variant_identity = 0;
    assert((row_bytes % 16) == 0);
    const int hidden_int4 = row_bytes / 16;
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(token_dtype);
    const char* backward_literal = backward ? "true" : "false";
    // Size the block so the per-warp smem row buffer fits the device opt-in cap, before building
    // the source so __launch_bounds__ matches the actual block (right register budget per block
    // size, not a fixed max that would spill small blocks).
    int smem_cap = 0, dev = 0;
    (void)cudaGetDevice(&dev);
    (void)cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    // NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE is the chunk width in hidden elements. Unset auto-selects
    // the default by EP size: one tile for EP < 16, the whole row for EP >= 16 (chunking wins at
    // small EP but not large). 0 forces the whole row; other values round up to whole tiles,
    // clamped to the row.
    const int requested_int4 = [&] {
        // Flat relays the row without EM gather/reduction accumulators. Use a
        // full-row TMA stage and size the warp count to its shared-memory cost.
        if (param.flat) {
            return 0;
        }
        constexpr int kOneTile = 1;  // rounds up to one tile in combine_push_chunk_size
        constexpr int kAutoNoChunkEpSize = 16;  // EP >= this uses the whole row by default
        const int auto_default = ep_size >= kAutoNoChunkEpSize ? 0 : kOneTile;
        const std::string raw_value = ::nccl_ep::jit::env_value("NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE");
        if (raw_value.empty()) return auto_default;
        char* end = nullptr;
        const long elems = std::strtol(raw_value.c_str(), &end, 10);
        if (end == raw_value.c_str() || *end != '\0' || elems < 0 || elems > INT_MAX / 4) {
            if (::nccl_ep::jit::announce_once("combine_push_chunk_size_bad_h" + std::to_string(hidden_int4))) {
                std::fprintf(stderr,
                             "[nccl_ep][env] NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE=\"%s\" is not a non-negative "
                             "integer; using the default chunk\n",
                             raw_value.c_str());
            }
            return auto_default;
        }
        if (elems == 0) return 0;
        const int elem_bytes = token_dtype == ncclFloat32 ? 4
                               : (token_dtype == ncclFloat8e4m3 || token_dtype == ncclFloat8e5m2 ||
                                  token_dtype == ncclFloat4x2)
                                   ? 1
                                   : 2;
        const int int4s = static_cast<int>((elems * elem_bytes + 15) / 16);
        if (::nccl_ep::jit::announce_once("combine_push_chunk_size_h" + std::to_string(hidden_int4))) {
            std::fprintf(stderr,
                         "[nccl_ep][env] NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE: combine push h%d chunk=%d elems "
                         "(%d int4 before rounding to tiles)\n",
                         hidden_int4, elems, int4s);
        }
        return int4s;
    }();
    const int chunk_size = ::ht_ep::combine_push_chunk_size(hidden_int4, requested_int4);
    const int slots = param.flat
        ? std::min(::ht_ep::kCombinePushMaxSlots, smem_cap / (row_bytes + 16))
        : ::ht_ep::combine_push_slots_per_block(hidden_int4, smem_cap, chunk_size);
    if (slots == 0) {
        std::fprintf(stderr, "[nccl_ep] combine push: a %d-byte row exceeds the %d-byte shared memory budget\n",
                     row_bytes, smem_cap);
        return ncclInvalidArgument;
    }
    const int block_dim = slots * 32;
    const int chunk_bytes = ::ht_ep::combine_push_buf_int4(hidden_int4, chunk_size) * 16;
    if (!param.flat && chunk_bytes > smem_cap - ::ht_ep::combine_push_meta_reserve_bytes()) {
        std::fprintf(stderr,
                     "[nccl_ep] combine push: a %d-byte chunk does not fit the %d-byte shared memory cap; "
                     "set NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE to a smaller chunk\n",
                     chunk_bytes, smem_cap);
        return ncclInvalidArgument;
    }

    // chunk_size is baked into the source and variant name: it sets the per-warp buffer width and
    // the reduce strategy at compile time, so each distinct value is its own JIT compile.
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "combine_push_topk" << top_k << "_h" << hidden_int4
             << ::nccl_ep::jit::token_dtype_name_tag(token_dtype) << (backward ? "_bwd" : "")
             << "_c" << chunk_size << (param.flat ? "_flat" : "") << "_w" << slots;
        return name.str();
    }();
    const std::string source = [&] {
        std::ostringstream src;
        src << "#include \"device/ht_ep.cuh\"\n"
            << "\n"
            << "extern \"C\" __launch_bounds__(" << block_dim << ", 1)\n"
            << "__global__ void " << kCombinePushJitEntryName << "(\n"
            << "    const __grid_constant__ ::ht_ep::combine_push_param_t p) {\n";
        if (param.flat) {
            src << "  ::ht_ep::combine_push_flat<" << hidden_int4 << ", " << slots << ", "
                << backward_literal << ">(p);\n}\n";
            return src.str();
        }
        src << "  extern __shared__ uint8_t smem_bytes[];\n"
            << "  ::ht_ep::combine_push<" << top_k << ", " << hidden_int4 << ", " << token_dtype_literal
            << ", " << backward_literal << ", " << chunk_size << ">(\n"
            << "      p.peer_staging_ptrs,\n"
            << "      reinterpret_cast<const uint8_t*>(p.recv_x_em),\n"
            << "      p.flat2em_slot_map,\n"
            << "      p.recv_slot_to_src,\n"
            << "      p.num_recv_tokens_dev,\n"
            << "      p.dcomm,\n"
            << "      p.head_sync_flag,\n"
            << "      p.grid_barrier_counter,\n"
            << "      p.top_k,\n"
            << "      p.row_bytes,\n"
            << "      p.caller_num_recv_tokens,\n"
            << "      p.my_lsa_rank,\n"
            << "      p.tokens_per_rank,\n"
            << "      p.lsa_team_size,\n"
            << "      p.topk_weights_em,\n"
            << "      p.srcpos_map,\n"
            << "      p.unfused_sync,\n"
            << "      smem_bytes);\n"
            << "}\n";
        return src.str();
    }();

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_combine_push";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kCombinePushJitEntryName;
    variant.identity = &variant_identity;
    // Hash the variant name: it already carries every specialization parameter unmasked, and a
    // hand-packed bitfield would silently alias once a field overflows its allotted bits - see
    // launch_dispatch_pull for the same rationale.
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = num_blocks;
    variant.block_dim = block_dim;
    variant.dynamic_smem_bytes = param.flat ? slots * (row_bytes + 16) : static_cast<int>(
        ::ht_ep::combine_push_dynamic_smem_bytes(hidden_int4, slots, chunk_size));
    variant.nucs = nucs;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] combine-push JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status),
                     error.empty() ? "" : ": ", error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

inline ncclResult_t launch_combine_reduce_stage(
    int row_bytes,
    int num_blocks,
    ::ht_ep::combine_reduce_param_t& param,
    cudaStream_t stream,
    ncclDataType_t token_dtype,
    bool backward,
    bool flat) {
    static const int variant_identity = 0;
    assert((row_bytes % 16) == 0);
    const int hidden_int4 = row_bytes / 16;
    const int block_dim = flat ? ::ht_ep::kLocalPermuteReduceThreads : ::ht_ep::kCombineReduceThreads;
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(token_dtype);
    const char* backward_literal = backward ? "true" : "false";

    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "combine_reduce_h" << hidden_int4 << ::nccl_ep::jit::token_dtype_name_tag(token_dtype)
             << (backward ? "_bwd" : "") << (param.retained_slots ? "_drop" : "");
        if (flat) {
            name << "_flat_k" << param.num_topk << "_e" << param.experts_per_rank
                 << "_n" << param.lsa_team_size;
        }
        return name.str();
    }();
    const std::string source = [&] {
        std::ostringstream src;
        src << "#include \"device/ht_ep.cuh\"\n"
            << "\n"
            << "extern \"C\" __launch_bounds__(" << block_dim << ", 1)\n"
            << "__global__ void " << kCombineReduceJitEntryName << "(\n"
            << "    const __grid_constant__ ::ht_ep::combine_reduce_param_t p) {\n"
            << "  ::ht_ep::combine_reduce<" << hidden_int4 << ", " << token_dtype_literal << ", "
            << backward_literal << ", " << ::nccl_ep::jit::bool_literal(param.retained_slots != nullptr)
            << ", " << ::nccl_ep::jit::bool_literal(flat) << ">(\n"
            << "      reinterpret_cast<uint8_t*>(p.attn_output),\n"
            << "      reinterpret_cast<const uint8_t*>(p.staging),\n"
            << "      p.topk_idx,\n"
            << "      " << (flat ? std::to_string(param.num_topk) : "p.num_topk") << ",\n"
            << "      " << (flat ? std::to_string(param.experts_per_rank) : "p.experts_per_rank") << ",\n"
            << "      " << (flat ? std::to_string(param.experts_per_lsa_team) : "p.experts_per_lsa_team") << ",\n"
            << "      p.num_combined_tokens,\n"
            << "      " << (flat ? std::to_string(param.lsa_team_size) : "p.lsa_team_size") << ",\n"
            << "      p.row_bytes,\n"
            << "      p.combined_topk_weights,\n"
            << "      p.top_k, p.retained_slots);\n"
            << "}\n";
        return src.str();
    }();

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_combine_epi_reduce";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kCombineReduceJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = (static_cast<std::uint64_t>(backward) << 63) |
        (static_cast<std::uint64_t>(param.retained_slots != nullptr) << 62) |
        (static_cast<std::uint64_t>(token_dtype) << 48) | static_cast<std::uint64_t>(hidden_int4);
    // Keep EM dimensions runtime-configurable; specialize only flat rank decoding.
    if (flat) {
        variant.runtime_key |= std::uint64_t{1} << 61;
        for (int value : {param.num_topk, param.experts_per_rank,
                          param.experts_per_lsa_team, param.lsa_team_size}) {
            variant.runtime_key = ::nccl_ep::jit::runtime_key_mix(variant.runtime_key, value);
        }
    }
    variant.num_blocks = num_blocks;
    variant.block_dim = block_dim;
    variant.dynamic_smem_bytes = 0;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] combine-reduce JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status),
                     error.empty() ? "" : ": ", error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

} // namespace jit
} // namespace ht
} // namespace nccl_ep
