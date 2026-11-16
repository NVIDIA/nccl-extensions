/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ht_ep.cuh"
#include "device/jit/jit_runtime.hpp"

#include <cstdio>
#include <cstdint>
#include <functional>
#include <string>

namespace nccl_ep {
namespace ht {
namespace jit {

constexpr int kLsaSyncNumThreads = 32;
constexpr const char* kLsaHeadSyncEntryName = "nccl_ep_jit_ht_lsa_head_sync_kernel";
constexpr const char* kLsaTailSyncEntryName = "nccl_ep_jit_ht_lsa_tail_sync_kernel";

inline ncclResult_t launch_lsa_sync_jit(
    const char* family,
    const char* entry_name,
    const char* source,
    const void* identity,
    void* args,
    cudaStream_t stream) {
    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = family;
    variant.variant_name = family;
    variant.source = source;
    variant.entry_name = entry_name;
    variant.identity = identity;
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(family));
    variant.num_blocks = 1;
    variant.block_dim = kLsaSyncNumThreads;
    variant.dynamic_smem_bytes = 0;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, args, stream, &error);
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] HT LSA sync JIT launch failure: %s%s%s\n",
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

inline ncclResult_t launch_lsa_head_sync(
    ncclDevComm_t* dcomm,
    uint32_t* head_sync_flag,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    static const std::string source =
        "#include \"device/ht_ep.cuh\"\n"
        "extern \"C\" __launch_bounds__(32, 1)\n"
        "__global__ void nccl_ep_jit_ht_lsa_head_sync_kernel(\n"
        "    const __grid_constant__ ::ht_ep::lsa_head_sync_param_t p) {\n"
        "  ::ht_ep::lsa_grid_head_gate(p.dcomm, p.head_sync_flag);\n"
        "}\n";
    ::ht_ep::lsa_head_sync_param_t args{dcomm, head_sync_flag};
    return launch_lsa_sync_jit(
        "ht_lsa_head_sync", kLsaHeadSyncEntryName, source.c_str(), &variant_identity, &args, stream);
}

inline ncclResult_t launch_lsa_tail_sync(
    ncclDevComm_t* dcomm,
    uint32_t* grid_barrier_counter,
    uint32_t* head_sync_flag,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    static const std::string source =
        "#include \"device/ht_ep.cuh\"\n"
        "extern \"C\" __launch_bounds__(32, 1)\n"
        "__global__ void nccl_ep_jit_ht_lsa_tail_sync_kernel(\n"
        "    const __grid_constant__ ::ht_ep::lsa_tail_sync_param_t p) {\n"
        "  ::ht_ep::lsa_grid_tail_barrier(p.dcomm, p.grid_barrier_counter, p.head_sync_flag);\n"
        "}\n";
    ::ht_ep::lsa_tail_sync_param_t args{dcomm, grid_barrier_counter, head_sync_flag};
    return launch_lsa_sync_jit(
        "ht_lsa_tail_sync", kLsaTailSyncEntryName, source.c_str(), &variant_identity, &args, stream);
}

} // namespace jit
} // namespace ht
} // namespace nccl_ep
