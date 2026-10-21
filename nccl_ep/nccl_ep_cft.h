/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_CFT_H_
#define NCCL_EP_CFT_H_

// LL CFT sends dispatch payloads to FABRIC-backed receive buffers and tracks
// arrival with hardware byte counters. Token counts still use recvCntBuf.
// Setup is collective: any rank's failure fails group creation on every rank.
// The two parity buffers use 2 * nRanks hardware counters per GPU; per-expert
// progress is tracked separately in cumul_byte_expect_dev.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

// Per-group LL CFT state, embedded in ncclEpGroup as ll_cft_buffers.
struct ncclEpLlCftBuffers {
    // Set on every rank only after collective setup succeeds.
    bool initialized = false;
    void* recv_base = nullptr;              // mapped FABRIC recv buffer VA (this rank owns/consumes)
    size_t recv_buffer_size = 0;            // total FABRIC buffer size (bindAlign multiple)
    unsigned long long recv_va = 0;         // CUdeviceptr cast to ull (for cuMem* calls)
    unsigned long long recv_mem_handle = 0; // CUmemGenericAllocationHandle
    // Track reserve, create and bind separately so partial setup can be released.
    uint32_t local_le = 0;
    bool local_le_reserved = false;
    bool local_le_valid = false;
    bool local_le_bound = false;
    // Imported endpoint IDs [nRanks]; ID 0 is valid and [rank] is unused.
    // cftPairUsesCft() decides whether a pair uses CFT.
    uint32_t* peer_le_dev = nullptr;
    // Every imported endpoint, including those that failed the readiness check.
    std::vector<uint32_t> imported_le_ids;
    // Offset of the 256B-aligned per-source byte counters after each parity's data.
    uint64_t byte_cntr_base_off = 0;
    // Bytes between the two copies of the data region and hardware counters.
    uint64_t parity_stride = 0;
    // Next dispatch's parity, shared across handles and advanced only by dispatch.
    // This keeps consecutive dispatches separate even when handles alternate.
    int dispatch_parity = 0;
    // Per-source capacity: max_dispatch_tokens_per_rank * slot_stride_bound.
    size_t src_rank_region_bytes = 0;
    // Maximum header + payload stride, checked by ncclEpDispatch.
    size_t slot_stride_bound = 0;
    // Targets [2][nRanks * numLocalExperts], one per receiver warp group to avoid
    // concurrent updates. Each parity's hardware counters accumulate across dispatches.
    uint64_t* cumul_byte_expect_dev = nullptr;
    size_t cumul_slots_per_parity = 0;      // nRanks * numLocalExperts

    // Zero-copy keeps two bindings on local_le: internal headers/counters
    // (local_le_bound) and caller-owned payload output (zc_bound).
    // The output window and allocation must stay alive until group destruction.
    uint64_t bind_align = 0;
    uint64_t zc_reserve_bytes = 0;
    uint64_t zc_bind_off = 0;
    bool zc_active = false;          // collective bind succeeded on every rank
    bool zc_bound = false;           // this rank owns a binding to release
    size_t zc_bound_size = 0;
    uint64_t zc_data_off = 0;        // endpoint offset of output row zero
    ncclWindow_t zc_bound_win{};
    void* zc_bound_data = nullptr;
    size_t zc_bound_shape[3] = {};
    ncclDataType_t zc_bound_datatype = ncclBfloat16;
};

// Defined in nccl_ep.cc.
void ncclAllGatherHost(
    void* host_buffer, size_t element_size, int rank, int nRanks, ncclComm_t comm, cudaStream_t stream);

// Platform validation runs before endpoint setup when NCCL_EP_LL_CFT is enabled.
// Query records each rank's capabilities; judge evaluates them without GPU calls.
// Group creation allgathers the facts and floors so every rank reaches the same
// verdict and reports the same failures.
//
// CFT requires unicast endpoints, counted operations and fabric handles, plus
// compatible build, runtime, driver, GPU and JIT compiler versions. Multicast
// support is reported for comparison with NCCL's ncclGpuCftSupport probe but
// is not required by this extension.

// What one rank's device reports. Fixed-width, trivially copyable: it is
// allgathered as part of the NCCL_EP_LL_CFT vote. -1 in an attribute field
// means "not reported by this driver" (cuDeviceGetAttribute failed).
struct ncclEpCftPlatformFacts {
    int32_t cuda_device_id = -1;
    int32_t toolkit_version = 0;           // CUDA_VERSION the library was compiled with
    uint8_t built_with_cft = 0;            // CUDA_VERSION >= 13030 at compile time
    uint8_t query_ok = 0;                  // cuDeviceGet succeeded
    uint8_t limits_ok = 0;                 // cuLogicalEndpointGetLimits(UNICAST|FABRIC|COUNTED_OPS) succeeded
    uint8_t driver_api_available = 0;      // all required logical endpoint entry points resolved
    int32_t query_cu_result = 0;           // CUresult of cuDeviceGet
    int32_t runtime_version = 0;           // cudaRuntimeGetVersion
    int32_t driver_version = 0;            // cuDriverGetVersion (driver API level, e.g. 13030)
    int32_t cc_major = 0;
    int32_t cc_minor = 0;
    int32_t le_unicast_supported = -1;     // attribute 153 (required)
    int32_t le_multicast_supported = -1;   // attribute 154 (reported only; NCCL requires it)
    int32_t le_counted_ops_supported = -1; // attribute 155 (required when reported)
    int32_t fabric_handle_supported = -1;  // CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED (required)
    int32_t limits_cu_result = 0;          // CUresult of cuLogicalEndpointGetLimits
    uint64_t bind_alignment = 0;           // from GetLimits; nccl_ep_cft_init cannot size the buffer when 0
    uint64_t max_size = 0;                 // from GetLimits; 0 is read as "no advertised cap" (as in init)
    // JIT nvcc version in CUDA_VERSION format (13040 = 13.4); 0 = unavailable.
    int32_t jit_nvcc_version = 0;
    int32_t reserved2_ = 0;
};

// Minimum versions and capabilities, allgathered with each rank's facts.
// Defaults require Blackwell, CUDA 13.3 and driver API level 13030.
struct ncclEpCftPlatformFloors {
    int32_t min_toolkit = 13030;
    int32_t min_runtime = 13030;
    int32_t min_driver = 13030;    // May be raised by NCCL_EP_LL_CFT_DEBUG_MIN_DRIVER.
    int32_t min_cc_major = 10;
    int32_t min_jit_nvcc = 13030;
};

// One bit per requirement, in the order the message lists them.
enum ncclEpCftPlatformFail : uint32_t {
    NCCL_EP_CFT_FAIL_TOOLKIT = 1u << 0,        // built without CFT support / toolkit too old
    NCCL_EP_CFT_FAIL_RUNTIME = 1u << 1,        // CUDA runtime too old
    NCCL_EP_CFT_FAIL_DRIVER = 1u << 2,         // driver too old
    NCCL_EP_CFT_FAIL_CC = 1u << 3,             // compute capability below Blackwell
    NCCL_EP_CFT_FAIL_LE_UNICAST = 1u << 4,     // unicast logical endpoints not supported
    NCCL_EP_CFT_FAIL_COUNTED_OPS = 1u << 5,    // counted ops: GetLimits failed, zero bind alignment, or attribute 155 = 0
    NCCL_EP_CFT_FAIL_FABRIC_HANDLE = 1u << 6,  // fabric memory handles not supported
    NCCL_EP_CFT_FAIL_QUERY = 1u << 7,          // the device could not be queried at all
    NCCL_EP_CFT_FAIL_JIT_NVCC = 1u << 8,       // the JIT's nvcc is older than the floor or unavailable
    NCCL_EP_CFT_FAIL_DRIVER_API = 1u << 9,     // a required logical endpoint entry point is unavailable
};

struct ncclEpCftPlatformVerdict {
    uint32_t failed = 0;               // OR of ncclEpCftPlatformFail; 0 = supported
    int32_t support_level = 0;         // extension analogue of NCCL's gpuCftSupport: min(toolkit, driver) or 0
    int32_t nccl_gpu_cft_support = 0;  // what NCCL's ncclGpuCftSupport would compute for this device
};

// Requires cudaSetDevice to have initialized the driver, as in ncclEpCreateGroup.
void nccl_ep_cft_platform_query(int cuda_device_id, ncclEpCftPlatformFacts* facts);

// Pure: identical inputs produce the same verdict on every rank.
ncclEpCftPlatformVerdict nccl_ep_cft_platform_judge(
    const ncclEpCftPlatformFacts& facts, const ncclEpCftPlatformFloors& floors);

// Semicolon-separated failure reasons; empty when supported.
std::string nccl_ep_cft_platform_reasons(
    const ncclEpCftPlatformFacts& facts,
    const ncclEpCftPlatformVerdict& verdict,
    const ncclEpCftPlatformFloors& floors);

// One-line platform summary for NCCL_EP_DEBUG diagnostics.
std::string nccl_ep_cft_platform_summary(
    const ncclEpCftPlatformFacts& facts, const ncclEpCftPlatformVerdict& verdict);

// Build and exchange endpoints, then agree collectively on setup success.
// On failure, all ranks release their resources and return ncclInvalidUsage.
ncclResult_t nccl_ep_cft_init(
    ncclEpLlCftBuffers* lb,
    ncclComm_t comm,
    int rank,
    int nRanks,
    int cuda_device_id,
    int num_local_experts,
    size_t max_dispatch_tokens_per_rank,
    size_t max_token_bytes,
    bool zero_copy,
    const char* env_var_name,
    bool verbose,
    cudaStream_t stream);

// Bind the output collectively before the first zero-copy dispatch. A failure
// returns an error on every rank. The binding stays fixed until group destruction.
ncclResult_t nccl_ep_cft_bind_output(
    ncclEpLlCftBuffers* lb,
    ncclComm_t comm,
    int rank,
    int nRanks,
    int cuda_device_id,
    bool have_window,
    void* data,
    size_t tensor_bytes,
    bool verbose,
    cudaStream_t stream);

// Release all acquired resources, including those from partial setup.
ncclResult_t nccl_ep_cft_destroy(ncclEpLlCftBuffers* lb, int cuda_device_id);

#endif  // NCCL_EP_CFT_H_
