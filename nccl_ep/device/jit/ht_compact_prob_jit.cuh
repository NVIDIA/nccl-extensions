/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "device/ht_compact_prob.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_source_literals.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace ht {
namespace jit {

constexpr const char* kCompactProbOutputEntry = "nccl_ep_jit_dispatch_prob_compact_to_output_kernel";
constexpr int kCompactProbOutputThreads = 256;

inline std::string compact_prob_output_source(bool wide_position, bool sorted_topk) {
    std::ostringstream src;
    src << "#include \"device/ht_compact_prob.cuh\"\n"
        << "extern \"C\" __launch_bounds__(" << kCompactProbOutputThreads << ")\n"
        << "__global__ void " << kCompactProbOutputEntry << "(\n"
        << "    const __grid_constant__ ::nccl_ep::ht::compact_prob_output_param_t p) {\n"
        << "  ::nccl_ep::ht::compact_prob_to_output<" << (wide_position ? "uint16_t" : "uint8_t")
        << ", " << ::nccl_ep::jit::bool_literal(sorted_topk) << ">(p);\n"
        << "}\n";
    return src.str();
}

inline void launch_compact_prob_output(int num_blocks, bool wide_position, bool sorted_topk,
                                      compact_prob_output_param_t& param, cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string name = std::string("compact_prob_output_") + (wide_position ? "u16" : "u8") +
                             (sorted_topk ? "_sorted" : "_topk");
    // JitKernelVariant borrows its source until launch_jit_kernel returns.
    const std::string source = compact_prob_output_source(wide_position, sorted_topk);
    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_compact_prob_output";
    variant.variant_name = name;
    variant.source = source;
    variant.entry_name = kCompactProbOutputEntry;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<uint64_t>(wide_position) | (static_cast<uint64_t>(sorted_topk) << 1);
    variant.num_blocks = num_blocks;
    variant.block_dim = kCompactProbOutputThreads;
    variant.dynamic_smem_bytes = 0;

    std::string error;
    const auto status = ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal compact probability decode failure for %s: %s%s%s\n",
                     name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status),
                     error.empty() ? "" : ": ", error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace ht
} // namespace nccl_ep
