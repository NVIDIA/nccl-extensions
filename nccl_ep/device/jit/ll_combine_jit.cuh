/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ll_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_source_literals.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace ll {
namespace jit {

constexpr const char* kLlCombineJitEntryName = "nccl_ep_jit_ll_combine_kernel";

// Compile-time bounds the combine kernel is specialized on: the maximum topk
// it supports and the inner-loop unroll factor.
constexpr int kLlCombineMaxTopk = 9;
constexpr int kLlCombineMaxUnrolls = 4;

inline std::string ll_combine_jit_source(
    bool useLogFmt,
    int hidden,
    ncclEpLayout_t layout,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype) {
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    const char* topk_type = topkIdxIsInt64 ? "int64_t" : "int32_t";
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(tokenDtype);

    std::ostringstream src;
    src << "#include \"device/ll_ep.cuh\"\n"
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(1024, 1)\n"
        << "__global__ void " << kLlCombineJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::ll::combine_kernel_args_t p) {\n"
        << "  nccl_ep::ll::combine_kernel_impl<\n"
        << "      " << ::nccl_ep::jit::bool_literal(useLogFmt) << ",\n"
        << "      " << hidden << ",\n"
        << "      " << kLlCombineMaxTopk << ",\n"
        << "      " << kLlCombineMaxUnrolls << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << topk_type << ",\n"
        << "      " << token_dtype_literal << ">(\n"
        << "      p.inData, p.srcInfo, p.layoutRange,\n"
        << "      static_cast<const " << topk_type << "*>(p.inTopkIdx), p.topkWeights,\n"
        << "      p.rankMask, p.asyncErrorFlag,\n"
        << "      p.outData,\n"
        << "      p.sendBuf, p.recvBuf, p.recvFlagBuf,\n"
        << "      p.sendOff, p.recvOff, p.recvFlagOff,\n"
        << "      p.atomicCleanFlag,\n"
        << "      p.nextRecvCntBuf, p.nextRecvCntBufSize,\n"
        << "      p.waitStats,\n"
        << "      p.numCombinedTokens, p.hidden, p.numTopk, p.maxTokensPerRank,\n"
        << "      p.numExperts, p.currRank, p.numRanks,\n"
        << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
        << "      p.phases, p.zeroCopy, p.numComms,\n"
        << "      p.devComms, p.windows, p.signalsBase, p.timeoutCycles);\n"
        << "}\n";
    return src.str();
}

inline void launch_ll_combine(
    bool useLogFmt,
    int hidden,
    ncclEpLayout_t layout,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    int numSms,
    int numWarps,
    int dynamic_smem_bytes,
    const combine_kernel_args_t& args,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "ll_combine"
             << "_hdim" << hidden << ::nccl_ep::jit::layout_name_tag(layout)
             << (useLogFmt ? "_logfmt" : "") << (topkIdxIsInt64 ? "_topk64" : "_topk32")
             << ::nccl_ep::jit::token_dtype_name_tag(tokenDtype);
        return name.str();
    }();
    const std::string source = ll_combine_jit_source(useLogFmt, hidden, layout, topkIdxIsInt64, tokenDtype);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ll_combine";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLlCombineJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = numSms;
    variant.block_dim = numWarps * 32;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;
    // Cooperative launch is required for the grid-wide sync between the SEND
    // and RECV phases.
    variant.cooperative = true;
    // Pair SMs into clusters of 2 when possible to share distributed SMEM.
    variant.cluster_dim_x = (numSms % 2 == 0) ? 2 : 1;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, const_cast<combine_kernel_args_t*>(&args), stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal LL combine JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace ll
} // namespace nccl_ep
