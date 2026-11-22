/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Compiles the generated HT dispatch and combine JIT sources with a host pass.
// The runtime JIT builds with --cubin, which does not diagnose undefined names
// in templates it never instantiates, so a broken header can still run.
#include <gtest/gtest.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include "device/jit/ht_dispatch_jit.cuh"
#include "device/jit/ht_combine_jit.cuh"
#include "device/jit/nvcc_compiler.hpp"
#include "quantization_recipe.hpp"

using nccl_ep::ht::HtKernelImpl;
namespace fs = std::filesystem;

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? std::string(v) : fallback;
}

// Same include roots and flags as the runtime JIT, with --cuda in place of --cubin.
std::string compile_host_pass(const std::string& source, const std::string& tag, std::string* log) {
    nccl_ep::jit::JitCompileConfig config;
    config.target_arch = "sm_90";
    config.source_dir = env_or("NCCL_EP_JIT_SOURCE_DIR", NCCL_EP_TEST_JIT_SOURCE_DIR);
    config.build_include_dir = env_or("NCCL_EP_JIT_BUILD_INCLUDE_DIR", NCCL_EP_TEST_NCCL_INCDIR);
    config.cuda_include_dir = fs::path(env_or("CUDA_HOME", "/usr/local/cuda")) / "include";
    const std::vector<std::string> options = nccl_ep::jit::NvccCompiler().compile_options(config);

    const fs::path dir = fs::temp_directory_path() / ("nccl_ep_jit_compile_" + std::to_string(getpid()));
    fs::create_directories(dir);
    const fs::path src = dir / (tag + ".cu");
    std::ofstream(src) << source;

    std::string cmd = env_or("NCCL_EP_JIT_NVCC", env_or("CUDA_HOME", "/usr/local/cuda") + "/bin/nvcc");
    cmd += " --cuda " + src.string() + " -o /dev/null";
    for (const std::string& o : options) cmd += " '" + o + "'";
    cmd += " 2>&1";

    log->clear();
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) return "popen failed";
    std::array<char, 4096> buf;
    while (fgets(buf.data(), buf.size(), pipe) != nullptr) *log += buf.data();
    const int status = pclose(pipe);
    return status == 0 ? std::string() : cmd;
}

struct Shape {
    const char* name;
    ncclEpLayout_t layout;
    bool backward;
};

constexpr int kLsaTeams = 2;
constexpr int kLsaTeamSize = 8;
constexpr int kHidden = 7168;
constexpr int kMaxTokens = 4096;
constexpr int kChunk = 64;
constexpr int kBlocks = 24;
constexpr int kExpertsPerRank = 16;

const Shape kShapes[] = {
    {"flat_fwd", NCCL_EP_LAYOUT_FLAT, false},
    {"flat_bwd", NCCL_EP_LAYOUT_FLAT, true},
    {"em_fwd", NCCL_EP_LAYOUT_EXPERT_MAJOR, false},
};

std::string dispatch_source(const Shape& s, HtKernelImpl impl, nccl_ep::HtDispatchSchedule schedule) {
    using namespace nccl_ep::ht::jit;
    nccl_ep::DispatchKernelSpec spec{};
    EXPECT_EQ(nccl_ep::resolveDispatchKernelSpec(NCCL_EP_DISP_QUANT_NONE, ncclBfloat16, ncclFloat32, &spec),
              ncclSuccess);
    const dispatch_warp_layout_t L = compute_dispatch_warp_layout(kLsaTeams, s.layout, NCCL_EP_HT_DISPATCH_PIPELINES, false);
    return dispatch_jit_source(
        L.cross_lsa_group_warps, L.cross_lsa_group_start, L.lsa_g2s_group_warps, L.lsa_g2s_group_start,
        L.lsa_s2g_group_warps, L.lsa_s2g_group_start, L.map_group_warps, L.map_group_start,
        L.pad_group_warps, L.pad_group_start, L.head_extra_group_warps, L.head_extra_group_start,
        NCCL_EP_HT_DISPATCH_STAGES, NCCL_EP_HT_DISPATCH_IN_FLIGHT_S2G, kChunk, kMaxTokens, kLsaTeams, kBlocks,
        /*forward_dispatch=*/!s.backward, L.num_pipelines, kLsaTeamSize, s.layout, kHidden,
        /*sf_bytes_per_token=*/0, kExpertsPerRank, schedule, spec, impl);
}

std::string combine_source(const Shape& s, HtKernelImpl impl) {
    using namespace nccl_ep::ht::jit;
    const combine_warp_layout_t L = compute_combine_warp_layout(kLsaTeams, NCCL_EP_HT_COMBINE_CROSS_LSA_PIPELINES);
    return combine_jit_source(
        L, NCCL_EP_HT_COMBINE_CROSS_LSA_STAGES_G2S, NCCL_EP_HT_COMBINE_CROSS_LSA_STAGES_S2G,
        NCCL_EP_HT_COMBINE_TOK_PER_GROUP, kChunk,
        kMaxTokens, kLsaTeams, kBlocks, /*backward_combine=*/s.backward, kLsaTeamSize, s.layout, kHidden,
        ncclBfloat16, NCCL_EP_COMB_QUANT_NONE, impl);
}

class HtJitCompile : public ::testing::TestWithParam<HtKernelImpl> {};

const char* impl_name(HtKernelImpl impl) {
    return impl == HtKernelImpl::kUnordered ? "unordered" : "ordered";
}

TEST_P(HtJitCompile, Dispatch) {
    for (const Shape& s : kShapes) {
        for (const auto schedule : {nccl_ep::HtDispatchSchedule::Coarse, nccl_ep::HtDispatchSchedule::Redistributed}) {
            std::string log;
            const std::string tag = std::string("dispatch_") + impl_name(GetParam()) + "_" + s.name +
                                    (schedule == nccl_ep::HtDispatchSchedule::Redistributed ? "_tail" : "_coarse");
            const std::string failed = compile_host_pass(dispatch_source(s, GetParam(), schedule), tag, &log);
            EXPECT_TRUE(failed.empty()) << tag << "\n" << failed << "\n" << log;
        }
    }
}

TEST_P(HtJitCompile, Combine) {
    for (const Shape& s : kShapes) {
        std::string log;
        const std::string tag = std::string("combine_") + impl_name(GetParam()) + "_" + s.name;
        const std::string failed = compile_host_pass(combine_source(s, GetParam()), tag, &log);
        EXPECT_TRUE(failed.empty()) << tag << "\n" << failed << "\n" << log;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Impl, HtJitCompile, ::testing::Values(HtKernelImpl::kOrdered, HtKernelImpl::kUnordered),
    [](const ::testing::TestParamInfo<HtKernelImpl>& info) { return std::string(impl_name(info.param)); });

} // namespace
