/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/hybrid_ep.cuh"
#include "device/jit/jit_runtime.hpp"

#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace hybridep {
namespace jit {

constexpr const char* kScanFlatJitEntryName = "nccl_ep_jit_ht_scan_flat_kernel";
constexpr const char* kScanEmJitEntryName = "nccl_ep_jit_ht_scan_em_kernel";

inline const char* scan_bool_literal(bool value) {
    return value ? "true" : "false";
}

inline std::string scan_flat_jit_source(
    int num_threads_per_block,
    int num_of_blocks,
    int num_lsa_teams,
    int lsa_team_size,
    int experts_per_rank,
    bool enable_per_expert_counts,
    bool enable_em_permute) {
    constexpr int rank_mask_word_bits = CHAR_BIT * static_cast<int>(sizeof(uint64_t));
    const int rank_mask_words = (lsa_team_size + rank_mask_word_bits - 1) / rank_mask_word_bits;
    std::ostringstream src;
    src << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << num_threads_per_block << ", 1)\n"
        << "__global__ void " << kScanFlatJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::scan_flat_kernel_param_t p) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::scan_impl_flat<\n"
        << "      " << num_threads_per_block << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << scan_bool_literal(enable_per_expert_counts) << ",\n"
        << "      " << scan_bool_literal(enable_em_permute) << ",\n"
        << "      " << (enable_em_permute ? experts_per_rank : 0) << ">(\n"
        << "      p.input_routing_map, p.tmp, p.sparse_to_dense_map, p.rdma_to_attn_map, p.attn_to_rdma_map,\n"
        << "      reinterpret_cast<hybrid_ep::rank_mask_t<" << rank_mask_words << ">*>(p.token_rank_mask),\n"
        << "      p.num_of_tokens_for_experts, p.local_expert_routing_map, p.per_expert_token_counts,\n"
        << "      p.node_rank, p.local_rank, p.num_of_tokens_per_rank, p.num_of_ranks_per_node, p.experts_per_rank,\n"
        << "      p.recv_total_counter, p.out_is_int64, p.max_recv_tokens_per_rank,\n"
        << "      p.token_to_recv_slot, smem_bytes";
    if (enable_em_permute) {
        src << ",\n"
            << "      p.expert_scan_tmp, p.flat2em_slot_map, p.em_top_k, p.em_alignment, p.em_internal_offsets,\n"
            << "      p.em_padded_out_counts_i32, p.em_padded_out_counts_i64,\n"
            << "      p.em_out_offsets_i32, p.em_out_offsets_i64, p.em_actual_counts_out";
    }
    src << ");\n"
        << "}\n";
    return src.str();
}

inline std::string
scan_em_jit_source(int num_threads_per_block, int num_of_blocks, int num_lsa_teams, int lsa_team_size) {
    constexpr int rank_mask_word_bits = CHAR_BIT * static_cast<int>(sizeof(uint64_t));
    const int rank_mask_words = (lsa_team_size + rank_mask_word_bits - 1) / rank_mask_word_bits;
    std::ostringstream src;
    src << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << num_threads_per_block << ", 1)\n"
        << "__global__ void " << kScanEmJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::scan_em_kernel_param_t p) {\n"
        << "  hybrid_ep::scan_impl_em<\n"
        << "      " << num_threads_per_block << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << lsa_team_size << ">(\n"
        << "      p.input_routing_map, p.rdma_to_attn_map, p.attn_to_rdma_map,\n"
        << "      reinterpret_cast<hybrid_ep::rank_mask_t<" << rank_mask_words << ">*>(p.token_rank_mask),\n"
        << "      p.node_rank, p.local_rank, p.num_of_tokens_per_rank, p.num_of_ranks_per_node, p.experts_per_rank);\n"
        << "}\n";
    return src.str();
}

inline void launch_scan_flat(
    int num_threads_per_block,
    int num_of_blocks,
    int num_lsa_teams,
    int lsa_team_size,
    int experts_per_rank,
    bool enable_per_expert_counts,
    bool enable_em_permute,
    ::hybrid_ep::scan_flat_kernel_param_t& param,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "scan_flat"
             << "_nodes" << num_lsa_teams << "_lsa" << lsa_team_size << "_threads" << num_threads_per_block << "_blocks"
             << num_of_blocks << (enable_per_expert_counts ? "_pec" : "_nopec") << (enable_em_permute ? "_em" : "_noem");
        if (enable_em_permute) name << "_epr" << experts_per_rank;
        return name.str();
    }();
    const std::string source = scan_flat_jit_source(
        num_threads_per_block,
        num_of_blocks,
        num_lsa_teams,
        lsa_team_size,
        experts_per_rank,
        enable_per_expert_counts,
        enable_em_permute);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_scan_flat";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kScanFlatJitEntryName;
    variant.identity = &variant_identity;
    // The FLAT (nopec/pec) and EM-permute instantiations share this launcher's
    // single variant_identity; key the fast in-process cache on the variant name
    // so they don't alias (an EM handle must not reuse the cached FLAT kernel).
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = num_of_blocks;
    variant.block_dim = num_threads_per_block;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal scan-flat JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

inline void launch_scan_em(
    int num_threads_per_block,
    int num_of_blocks,
    int num_lsa_teams,
    int lsa_team_size,
    ::hybrid_ep::scan_em_kernel_param_t& param,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "scan_em"
             << "_nodes" << num_lsa_teams << "_lsa" << lsa_team_size << "_threads" << num_threads_per_block << "_blocks"
             << num_of_blocks;
        return name.str();
    }();
    const std::string source = scan_em_jit_source(num_threads_per_block, num_of_blocks, num_lsa_teams, lsa_team_size);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_scan_em";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kScanEmJitEntryName;
    variant.identity = &variant_identity;
    // Distinct geometries (lsa/threads/blocks) must not alias in the fast cache.
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = num_of_blocks;
    variant.block_dim = num_threads_per_block;
    variant.dynamic_smem_bytes = 0;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal scan-em JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

constexpr const char* kBuildEmTablesJitEntryName = "nccl_ep_jit_build_em_tables_kernel";
constexpr int kBuildEmTablesBlockDim = 1024;

inline std::string build_em_tables_jit_source(int experts_per_rank) {
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << kBuildEmTablesBlockDim << ", 1)\n"
        << "__global__ void " << kBuildEmTablesJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::build_em_tables_param_t p) {\n"
        << "  hybrid_ep::build_em_tables_impl<" << experts_per_rank << ">(p);\n"
        << "}\n";
    return src.str();
}

inline void launch_build_em_tables_jit(
    int experts_per_rank,
    ::hybrid_ep::build_em_tables_param_t& param,
    int dynamic_smem_bytes,
    int num_blocks,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "build_em_tables_epr" << experts_per_rank;
        return name.str();
    }();
    const std::string source = build_em_tables_jit_source(experts_per_rank);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_build_em_tables";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kBuildEmTablesJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(experts_per_rank);
    variant.num_blocks = num_blocks;
    variant.block_dim = kBuildEmTablesBlockDim;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;
    variant.cooperative = true;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(
            stderr,
            "[nccl_ep jit] fatal build-em-tables JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
