/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda.h>
#include <cstring>
#include <unistd.h>

#include "nccl_ep_cft.h"

#if CUDA_VERSION >= 13030
namespace {

const char* const kFunctions[] = {
    "cuLogicalEndpointIdReserve", "cuLogicalEndpointIdRelease",
    "cuLogicalEndpointCreate", "cuLogicalEndpointDestroy",
    "cuLogicalEndpointQuery", "cuLogicalEndpointGetLimits",
    "cuLogicalEndpointExport", "cuLogicalEndpointImport",
    "cuLogicalEndpointBindAddr", "cuLogicalEndpointUnbind",
};

enum class Failure { None, Missing, Version, NullPointer, RuntimeError };
Failure failure = Failure::None;
const char* failed_symbol = nullptr;
int lookup_count = 0;
int limits_count = 0;
int clear_error_count = 0;

void unused_endpoint() {
    ADD_FAILURE() << "Platform probing must not create or release endpoints";
}

CUresult CUDAAPI get_limits(cuuint64_t* alignment, cuuint64_t* size,
                           const CUlogicalEndpointProp* prop) {
    ++limits_count;
    EXPECT_EQ(prop->type, CU_LOGICAL_ENDPOINT_TYPE_UNICAST);
    EXPECT_EQ(prop->flags, CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS);
    *alignment = 65536;
    *size = 0;
    return CUDA_SUCCESS;
}

void check_query(bool supported) {
    ncclEpCftPlatformFacts facts;
    nccl_ep_cft_platform_query(0, &facts);
    EXPECT_EQ(facts.driver_api_available, supported);
    EXPECT_EQ(limits_count, supported ? 1 : 0);
    ncclEpCftPlatformFloors floors;
    floors.min_jit_nvcc = 0;  // This test is about driver entry points, not nvcc discovery.
    const auto verdict = nccl_ep_cft_platform_judge(facts, floors);
    EXPECT_EQ(verdict.failed, supported ? 0u : uint32_t(NCCL_EP_CFT_FAIL_DRIVER_API));
    if (!supported) {
        EXPECT_EQ(verdict.support_level, 0);
        EXPECT_EQ(nccl_ep_cft_platform_reasons(facts, verdict, floors),
                  "CUDA driver does not provide the logical endpoint APIs required for CFT");
    }
}

// Each child starts with an uninitialized process-wide API cache. No GPU is used.
TEST(CftDriverApi, CompleteApiIsCached) {
    EXPECT_EXIT({
        check_query(true);
        EXPECT_EQ(lookup_count, 10);
        const int resolved = lookup_count;
        ncclEpCftPlatformFacts facts;
        nccl_ep_cft_platform_query(0, &facts);
        EXPECT_TRUE(facts.driver_api_available);
        EXPECT_EQ(lookup_count, resolved);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
    }, ::testing::ExitedWithCode(0), "");
}

class MissingCftDriverApi : public ::testing::TestWithParam<const char*> {};

TEST_P(MissingCftDriverApi, RejectsIncompleteApiIncludingCleanup) {
    EXPECT_EXIT({
        failure = Failure::Missing;
        failed_symbol = GetParam();
        check_query(false);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
    }, ::testing::ExitedWithCode(0), "");
}
INSTANTIATE_TEST_SUITE_P(EveryEntryPoint, MissingCftDriverApi, ::testing::ValuesIn(kFunctions));

class FailedCftDriverLookup : public ::testing::TestWithParam<Failure> {};

TEST_P(FailedCftDriverLookup, RejectsInvalidLookupResult) {
    EXPECT_EXIT({
        failure = GetParam();
        failed_symbol = "cuLogicalEndpointGetLimits";
        check_query(false);
        EXPECT_EQ(clear_error_count, failure == Failure::RuntimeError ? 1 : 0);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
    }, ::testing::ExitedWithCode(0), "");
}
INSTANTIATE_TEST_SUITE_P(ResultChecks, FailedCftDriverLookup,
    ::testing::Values(Failure::Version, Failure::NullPointer, Failure::RuntimeError));

}  // namespace

extern "C" cudaError_t CUDARTAPI __wrap_cudaGetDriverEntryPointByVersion(
    const char* symbol, void** function, unsigned int version,
    unsigned long long flags, cudaDriverEntryPointQueryResult* status) {
    ++lookup_count;
    EXPECT_EQ(version, 13030u);
    EXPECT_EQ(flags, cudaEnableDefault);
    bool known = false;
    for (const char* name : kFunctions) known |= std::strcmp(symbol, name) == 0;
    EXPECT_TRUE(known) << symbol;
    *function = std::strcmp(symbol, "cuLogicalEndpointGetLimits") == 0
        ? reinterpret_cast<void*>(&get_limits) : reinterpret_cast<void*>(&unused_endpoint);
    *status = cudaDriverEntryPointSuccess;
    if (failed_symbol && std::strcmp(symbol, failed_symbol) == 0) {
        switch (failure) {
        case Failure::Missing:
            *status = cudaDriverEntryPointSymbolNotFound;
            return cudaSuccess;
        case Failure::Version:
            *status = cudaDriverEntryPointVersionNotSufficent;
            return cudaSuccess;
        case Failure::NullPointer:
            *function = nullptr;
            return cudaSuccess;
        case Failure::RuntimeError:
            return cudaErrorNotSupported;
        case Failure::None:
            break;
        }
    }
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaGetLastError() {
    ++clear_error_count;
    return cudaErrorNotSupported;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaRuntimeGetVersion(int* version) {
    *version = 13040;
    return cudaSuccess;
}

extern "C" CUresult CUDAAPI __wrap_cuDriverGetVersion(int* version) {
    *version = 13040;
    return CUDA_SUCCESS;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaDeviceGetAttribute(
    int* value, cudaDeviceAttr attribute, int) {
    *value = attribute == cudaDevAttrComputeCapabilityMajor ? 10 : 0;
    return cudaSuccess;
}

extern "C" CUresult CUDAAPI __wrap_cuDeviceGet(CUdevice* device, int ordinal) {
    *device = ordinal;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI __wrap_cuDeviceGetAttribute(int* value, CUdevice_attribute, CUdevice) {
    *value = 1;
    return CUDA_SUCCESS;
}
#else
TEST(CftDriverApi, RequiresCftBuild) {
    GTEST_SKIP() << "Logical endpoint API lookup requires CUDA 13.3 headers";
}
#endif
