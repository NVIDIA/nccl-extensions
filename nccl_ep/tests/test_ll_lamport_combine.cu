/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multi-rank coverage of the Lamport combine (NCCL_EP_COMBINE_LSA=lamport) on a rank-major LL group, with an identity
 * "expert": the combine input is the dispatch output, so a source rank recovers (number of distinct destination ranks
 * of the token) * (token value).
 *
 *   - Back-to-back combines (the round-gate case): one dispatch followed by several combines with no dispatch in
 *     between, under dense routing (every rank receives from two peers) and sparse routing (every rank receives from
 *     one peer only, so a rank can run rounds ahead of a peer it never hears from). One rank is delayed by a spin
 *     kernel before the combines so the others get ahead.
 *   - Payload words equal to the sentinel 0xFFFFFFFF in BF16, FP16 and FP32: the combine must complete and return NaN
 *     for those words and the exact sums for the others.
 *   - Alternating dispatch and combine bursts, and ncclEpMaskClean followed by more rounds.
 *   - Mixed sequences on one group: a rank-major handle (Lamport combine) and an expert-major handle (the combine falls
 *     back to the default kernel) interleaved as D(rm) C(rm)+ D(em) C(em) ..., with runs of several Lamport combines
 *     between fallback combines and the other way round. The Lamport combine keeps its own epoch state, so the LL epoch
 *     and every bank must stay consistent whichever combine kernel ran last.
 *   - A hidden size that is not a multiple of the combine tile: the Lamport request falls back to the default combine,
 *     after a dispatch that is not the LSA one, and the same sequences must still be exact.
 */

#include "test_common.h"
#include "common.hpp"
#include "device/jit/ll_combine_jit.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kLamMaxTokens = 4;
constexpr int kLamTopk = 2;
constexpr int kBurst = 5; // combines per dispatch in the back-to-back sequences

// kFar: slot k of every token goes to rank (source + 1 + k), so a rank receives combine contributions from the two ranks
// after it and sends to the two before it.
enum class Routing { kDense, kSparse, kFar };

struct LamportCase {
    ncclDataType_t dtype;
    int hidden;
    const char* name;
};

// 4096 is a whole number of combine tiles for every dtype; 2880 is not (it makes the Lamport request fall back).
const LamportCase kCases[] = {
    {ncclBfloat16, 4096, "Bf16"},
    {ncclFloat16, 4096, "Fp16"},
    {ncclFloat32, 4096, "Fp32"},
    {ncclBfloat16, 2880, "Bf16PartialTileFallback"},
};

using nccl_ep::ll::LlCombineKernel;

const char* kernel_name(int kernel) {
    return nccl_ep::ll::jit::ll_combine_kernel_name(static_cast<LlCombineKernel>(kernel));
}

int elem_bytes(ncclDataType_t dtype) {
    return dtype == ncclFloat32 ? 4 : 2;
}

// Small non-negative integers are exact in every dtype and so are their sums.
uint32_t encode_pair(ncclDataType_t dtype, float value) {
    if (dtype == ncclFloat32) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
    uint16_t half_bits;
    if (dtype == ncclFloat16) {
        const __half h = __float2half(value);
        std::memcpy(&half_bits, &h, sizeof(half_bits));
    } else {
        const nv_bfloat16 b = __float2bfloat16(value);
        std::memcpy(&half_bits, &b, sizeof(half_bits));
    }
    return (static_cast<uint32_t>(half_bits) << 16) | half_bits;
}

// Element `e` of a row stored as 32-bit words.
bool element_is_nan(ncclDataType_t dtype, const uint32_t* words, int e) {
    if (dtype == ncclFloat32) {
        const uint32_t bits = words[e];
        return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0;
    }
    const uint16_t bits = reinterpret_cast<const uint16_t*>(words)[e];
    if (dtype == ncclFloat16) return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0;
    return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0;
}

float element_value(ncclDataType_t dtype, const uint32_t* words, int e) {
    if (dtype == ncclFloat32) {
        float f;
        std::memcpy(&f, &words[e], sizeof(f));
        return f;
    }
    const uint16_t bits = reinterpret_cast<const uint16_t*>(words)[e];
    if (dtype == ncclFloat16) {
        __half h;
        std::memcpy(&h, &bits, sizeof(h));
        return __half2float(h);
    }
    nv_bfloat16 b;
    std::memcpy(&b, &bits, sizeof(b));
    return __bfloat162float(b);
}

__global__ void spin_kernel(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

int destination_rank(Routing routing, int source, int slot) {
    switch (routing) {
        case Routing::kDense: return (source + slot) % g_nranks;
        case Routing::kFar: return (source + 1 + slot) % g_nranks;
        default: return (source + 1) % g_nranks;
    }
}

int distinct_destinations(Routing routing) {
    return routing == Routing::kSparse ? 1 : std::min(kLamTopk, g_nranks);
}

__global__ void shift_kernel(const nv_bfloat16* src, nv_bfloat16* dst, size_t n, float add) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dst[i] = __float2bfloat16(__bfloat162float(src[i]) + add);
    }
}

__device__ float to_float(nv_bfloat16 v) { return __bfloat162float(v); }
__device__ float to_float(__half v) { return __half2float(v); }
__device__ float to_float(float v) { return v; }
template <typename T> __device__ T from_float(float v);
template <> __device__ nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ float from_float<float>(float v) { return v; }

// dst = src + add per element; NaN elements are copied bit for bit, so the sentinel-pattern words of the sentinel
// cases still reach the combine unchanged.
template <typename T>
__global__ void shift_rows_kernel(const T* src, T* dst, size_t n, float add) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float v = to_float(src[i]);
        dst[i] = isnan(v) ? src[i] : from_float<T>(v + add);
    }
}

float token_value(int iteration, int source, int token) {
    return static_cast<float>(1 + (source * 7 + token * 3 + iteration) % 13);
}

class LamportCombineTest : public ::testing::TestWithParam<LamportCase> {
protected:
    // The group reads the environment once at creation.
    ScopedEnvVar lamport_env_{"NCCL_EP_COMBINE_LSA", "lamport"};
    ncclEpGroup_t group_ = nullptr;
    ncclEpHandle_t handle_ = nullptr;
    void* d_input_ = nullptr;
    void* d_recv_ = nullptr;
    int32_t* d_topk_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_recv_weights_ = nullptr;
    int32_t* d_recv_idx_ = nullptr;
    void* d_out_[kBurst] = {};
    // Per-combine inputs of a burst: combine k contributes the dispatched rows plus k * kBurstShift, so a combine that
    // is overwritten by (or overwrites) another of the burst returns a visibly wrong value.
    void* d_burst_in_[kBurst] = {};
    ncclEpTensor_t* t_burst_in_[kBurst] = {};
    static constexpr float kBurstShift = 10.0f;
    ncclEpTensor_t* t_input_ = nullptr;
    ncclEpTensor_t* t_recv_ = nullptr;
    ncclEpTensor_t* t_topk_ = nullptr;
    ncclEpTensor_t* t_weights_ = nullptr;
    ncclEpTensor_t* t_recv_weights_ = nullptr;
    ncclEpTensor_t* t_recv_idx_ = nullptr;
    ncclEpTensor_t* t_out_[kBurst] = {};
    Routing routing_ = Routing::kDense;

    // Expert-major handle on the same group (same top-k as the rank-major one, which the Lamport region requires): its combine never takes the Lamport kernel.
    struct ExpertMajorLayer {
        ncclEpHandle_t handle = nullptr;
        int64_t* d_topk = nullptr;
        nv_bfloat16 *d_input = nullptr, *d_dispatched = nullptr, *d_combined = nullptr;
        float* d_weights = nullptr;
        int32_t* d_counts = nullptr;
        ncclEpTensor_t *topk = nullptr, *input = nullptr, *dispatched = nullptr, *combined = nullptr;
        ncclEpTensor_t *weights = nullptr, *counts = nullptr;
    } em_;

    // The kernel the rank-major handle's combine resolves to: Lamport, except for the partial-tile case, whose combine
    // falls back to the default kernel. The expert-major handle's combine always does.
    LlCombineKernel rank_major_kernel() const {
        const bool eligible = nccl_ep::ll::jit::ll_combine_lsa_eligible(
            /*nvlinkOnly=*/true, NCCL_EP_COMB_QUANT_NONE, /*useLogFmt=*/false, NCCL_EP_LAYOUT_RANK_MAJOR,
            /*cftEnabled=*/false, hidden(), dtype(), nccl_ep::ll::NcclEpDispatchLsaMode::kTma);
        return eligible ? LlCombineKernel::k2SidedLamportRmLsa : LlCombineKernel::kDefault;
    }

    // The combine just issued must have resolved to `expected`; the sequence of kernels is recorded for the log.
    void expect_combine_kernel(LlCombineKernel expected) {
        const int got = ncclEpGroup_test_getLastLlCombineKernel(group_);
        EXPECT_EQ(got, static_cast<int>(expected))
            << "resolved " << kernel_name(got) << ", expected " << kernel_name(static_cast<int>(expected));
        kernel_sequence_ += expected == LlCombineKernel::k2SidedLamportRmLsa ? 'L' : 'D';
    }
    std::string kernel_sequence_;

    ncclDataType_t dtype() const { return GetParam().dtype; }
    int hidden() const { return GetParam().hidden; }
    size_t row_words() const { return static_cast<size_t>(hidden()) * elem_bytes(dtype()) / sizeof(uint32_t); }
    size_t row_bytes() const { return static_cast<size_t>(hidden()) * elem_bytes(dtype()); }

    ncclResult_t try_create_group(size_t rdma_buffer_size, unsigned int num_topk) {
        ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
        cfg.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
        cfg.num_experts = static_cast<unsigned int>(g_nranks);
        cfg.max_dispatch_tokens_per_rank = kLamMaxTokens;
        cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(g_nranks * kLamMaxTokens);
        cfg.max_token_bytes = row_bytes();
        cfg.rdma_buffer_size = rdma_buffer_size;
        cfg.num_topk = num_topk; // the group's top-k bound, 0 when the group declares none
        cfg.num_qp_per_rank = NCCL_EP_AUTO;
        cfg.num_channels = NCCL_EP_AUTO;
        cfg.enable_mask = true; // ncclEpMaskClean needs it
        return ncclEpCreateGroup(&group_, g_comm, &cfg);
    }

    // An explicit rdma_buffer_size that holds a handle of either layout at `topk` plus the Lamport region at `topk`.
    size_t explicit_buffer_bytes(int topk) {
        size_t layout_bytes = 0;
        for (const ncclEpLayout_t layout : {NCCL_EP_LAYOUT_EXPERT_MAJOR, NCCL_EP_LAYOUT_RANK_MAJOR}) {
            layout_bytes = std::max(
                layout_bytes,
                nccl_ep::LowLatencyLayout(kLamMaxTokens, row_bytes(), g_nranks, g_nranks, topk, layout, 0).total_bytes);
        }
        const size_t bytes =
            layout_bytes + nccl_ep::lamport_region_shape(kLamMaxTokens, row_bytes(), g_nranks, topk).reserved_bytes;
        return (bytes + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES * NUM_BUFFER_ALIGNMENT_BYTES;
    }

    void create_group(size_t rdma_buffer_size, unsigned int num_topk) {
        NCCL_ASSERT(try_create_group(rdma_buffer_size, num_topk));
    }

    // Replaces the group (and the handle made on it) with a new one.
    void recreate_group(size_t rdma_buffer_size, unsigned int num_topk) {
        if (handle_) {
            ncclEpHandleDestroy(handle_);
            handle_ = nullptr;
        }
        NCCL_ASSERT(ncclEpGroupDestroy(group_));
        group_ = nullptr;
        create_group(rdma_buffer_size, num_topk);
    }

    void SetUp() override {
        create_group(NCCL_EP_AUTO, 0);

        const size_t recv_rows = static_cast<size_t>(g_nranks) * kLamMaxTokens;
        CUDA_ASSERT(cudaMalloc(&d_input_, kLamMaxTokens * row_bytes()));
        CUDA_ASSERT(cudaMalloc(&d_recv_, recv_rows * row_bytes()));
        CUDA_ASSERT(cudaMalloc(&d_topk_, kLamMaxTokens * kLamTopk * sizeof(int32_t)));
        CUDA_ASSERT(cudaMalloc(&d_weights_, kLamMaxTokens * kLamTopk * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_weights_, recv_rows * kLamTopk * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_idx_, recv_rows * kLamTopk * sizeof(int32_t)));
        for (int k = 0; k < kBurst; ++k) {
            CUDA_ASSERT(cudaMalloc(&d_out_[k], kLamMaxTokens * row_bytes()));
            CUDA_ASSERT(cudaMalloc(&d_burst_in_[k], recv_rows * row_bytes()));
        }
        std::vector<float> ones(kLamMaxTokens * kLamTopk, 1.0f);
        CUDA_ASSERT(cudaMemcpy(d_weights_, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&t_input_, 2, dtype(), d_input_, kLamMaxTokens, hidden()));
        NCCL_ASSERT(epTensorCreate(&t_recv_, 3, dtype(), d_recv_, g_nranks, kLamMaxTokens, hidden()));
        NCCL_ASSERT(epTensorCreate(&t_topk_, 2, ncclInt32, d_topk_, kLamMaxTokens, kLamTopk));
        NCCL_ASSERT(epTensorCreate(&t_weights_, 2, ncclFloat32, d_weights_, kLamMaxTokens, kLamTopk));
        NCCL_ASSERT(epTensorCreate(&t_recv_weights_, 3, ncclFloat32, d_recv_weights_, g_nranks, kLamMaxTokens, kLamTopk));
        NCCL_ASSERT(epTensorCreate(&t_recv_idx_, 3, ncclInt32, d_recv_idx_, g_nranks, kLamMaxTokens, kLamTopk));
        for (int k = 0; k < kBurst; ++k) {
            NCCL_ASSERT(epTensorCreate(&t_out_[k], 2, dtype(), d_out_[k], kLamMaxTokens, hidden()));
            NCCL_ASSERT(epTensorCreate(&t_burst_in_[k], 3, dtype(), d_burst_in_[k], g_nranks, kLamMaxTokens, hidden()));
        }
    }

    void TearDown() override {
        if (handle_) ncclEpHandleDestroy(handle_);
        if (em_.handle) ncclEpHandleDestroy(em_.handle);
        for (auto* t : {em_.topk, em_.input, em_.dispatched, em_.combined, em_.weights, em_.counts}) {
            if (t) ncclEpTensorDestroy(t);
        }
        for (auto* p : {static_cast<void*>(em_.d_topk), static_cast<void*>(em_.d_input),
                        static_cast<void*>(em_.d_dispatched), static_cast<void*>(em_.d_combined),
                        static_cast<void*>(em_.d_weights), static_cast<void*>(em_.d_counts)}) {
            if (p) cudaFree(p);
        }
        for (auto*& t : t_out_) if (t) ncclEpTensorDestroy(t);
        for (auto*& t : t_burst_in_) if (t) ncclEpTensorDestroy(t);
        for (auto* t : {t_input_, t_recv_, t_topk_, t_weights_, t_recv_weights_, t_recv_idx_}) {
            if (t) ncclEpTensorDestroy(t);
        }
        for (auto* p : {d_input_, d_recv_, static_cast<void*>(d_topk_), static_cast<void*>(d_weights_),
                        static_cast<void*>(d_recv_weights_), static_cast<void*>(d_recv_idx_)}) {
            if (p) cudaFree(p);
        }
        for (auto* p : d_out_) if (p) cudaFree(p);
        for (auto* p : d_burst_in_) if (p) cudaFree(p);
        if (group_) ncclEpGroupDestroy(group_);
    }

    // A new handle (routing is fixed per handle) for the given pattern.
    void create_handle(Routing routing) {
        if (handle_) {
            ncclEpHandleDestroy(handle_);
            handle_ = nullptr;
        }
        routing_ = routing;
        std::vector<int32_t> topk(kLamMaxTokens * kLamTopk);
        for (int t = 0; t < kLamMaxTokens; ++t) {
            for (int k = 0; k < kLamTopk; ++k) topk[t * kLamTopk + k] = destination_rank(routing, g_rank, k);
        }
        CUDA_ASSERT(cudaMemcpy(d_topk_, topk.data(), topk.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        NCCL_ASSERT(ncclEpCreateHandle(&handle_, group_, NCCL_EP_LAYOUT_RANK_MAJOR, t_topk_, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    static constexpr int kEmSlots = 64; // per local expert, an upper bound on g_nranks * kLamMaxTokens

    void create_expert_major_handle() {
        const size_t slots = static_cast<size_t>(g_nranks) * kLamMaxTokens;
        ASSERT_LE(slots, static_cast<size_t>(kEmSlots));
        CUDA_ASSERT(cudaMalloc(&em_.d_topk, kLamMaxTokens * kLamTopk * sizeof(int64_t)));
        CUDA_ASSERT(cudaMalloc(&em_.d_input, kLamMaxTokens * hidden() * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&em_.d_dispatched, slots * hidden() * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&em_.d_combined, kLamMaxTokens * hidden() * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&em_.d_weights, kLamMaxTokens * kLamTopk * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&em_.d_counts, sizeof(int32_t)));
        std::vector<int64_t> topk(kLamMaxTokens * kLamTopk);
        std::vector<float> weights(kLamMaxTokens * kLamTopk, 1.0f / kLamTopk);
        for (int t = 0; t < kLamMaxTokens; ++t) {
            for (int k = 0; k < kLamTopk; ++k) topk[t * kLamTopk + k] = (g_rank * kLamMaxTokens + t + k) % g_nranks;
        }
        CUDA_ASSERT(cudaMemcpy(em_.d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(em_.d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&em_.topk, 2, ncclInt64, em_.d_topk, kLamMaxTokens, kLamTopk));
        NCCL_ASSERT(epTensorCreate(&em_.input, 2, ncclBfloat16, em_.d_input, kLamMaxTokens, hidden()));
        NCCL_ASSERT(epTensorCreate(&em_.dispatched, 3, ncclBfloat16, em_.d_dispatched, 1, slots, hidden()));
        NCCL_ASSERT(epTensorCreate(&em_.combined, 2, ncclBfloat16, em_.d_combined, kLamMaxTokens, hidden()));
        NCCL_ASSERT(epTensorCreate(&em_.weights, 2, ncclFloat32, em_.d_weights, kLamMaxTokens, kLamTopk));
        NCCL_ASSERT(epTensorCreate(&em_.counts, 1, ncclInt32, em_.d_counts, 1));
        NCCL_ASSERT(ncclEpCreateHandle(
            &em_.handle, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, em_.topk, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // One expert-major dispatch and combine (identity expert, weights 1 / top-k): the combine falls back to the default kernel.
    void run_expert_major(int iteration) {
        std::vector<nv_bfloat16> host(kLamMaxTokens * hidden());
        for (int t = 0; t < kLamMaxTokens; ++t) {
            std::fill_n(host.begin() + t * hidden(), hidden(), __float2bfloat16(token_value(iteration + 100, g_rank, t)));
        }
        CUDA_ASSERT(cudaMemcpy(em_.d_input, host.data(), host.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemset(em_.d_combined, 0, kLamMaxTokens * hidden() * sizeof(nv_bfloat16)));
        ncclEpDispatchInputs_t din = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t dout = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
        din.tokens = em_.input;
        dout.tokens = em_.dispatched;
        layout.expert_counters = em_.counts;
        NCCL_ASSERT(ncclEpDispatch(em_.handle, &din, &dout, &layout, nullptr, g_stream));
        ncclEpCombineInputs_t cin = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t cout = NCCL_EP_COMBINE_OUTPUTS_INIT;
        cin.tokens = em_.dispatched;
        cout.tokens = em_.combined;
        cout.topk_weights = em_.weights;
        NCCL_ASSERT(ncclEpCombine(em_.handle, &cin, &cout, nullptr, g_stream));
        expect_combine_kernel(LlCombineKernel::kDefault);
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        std::vector<nv_bfloat16> out(host.size());
        CUDA_ASSERT(cudaMemcpy(out.data(), em_.d_combined, out.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        for (int t = 0; t < kLamMaxTokens; ++t) {
            const float expected = token_value(iteration + 100, g_rank, t);
            for (int e = 0; e < hidden(); ++e) {
                ASSERT_EQ(__bfloat162float(out[t * hidden() + e]), expected)
                    << "expert-major token " << t << " element " << e << " iteration " << iteration;
            }
        }
    }

    // With `with_sentinels`, every even 32-bit word of a row is the sentinel 0xFFFFFFFF.
    void upload_input(int iteration, bool with_sentinels) {
        std::vector<uint32_t> host(kLamMaxTokens * row_words());
        for (int t = 0; t < kLamMaxTokens; ++t) {
            const uint32_t value_word = encode_pair(dtype(), token_value(iteration, g_rank, t));
            for (size_t w = 0; w < row_words(); ++w) {
                host[t * row_words() + w] = (with_sentinels && w % 2 == 0) ? 0xFFFFFFFFu : value_word;
            }
        }
        CUDA_ASSERT(cudaMemcpy(d_input_, host.data(), host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    void dispatch() {
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
        in.tokens = t_input_;
        in.topk_weights = t_weights_;
        out.tokens = t_recv_;
        out.topk_weights = t_recv_weights_;
        out.topk_idx = t_recv_idx_;
        NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out, &layout, nullptr, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    }

    void combine(int burst_index, ncclEpTensor_t* input = nullptr) {
        ncclEpCombineInputs_t in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        in.tokens = input != nullptr ? input : t_recv_;
        out.tokens = t_out_[burst_index];
        NCCL_ASSERT(ncclEpCombine(handle_, &in, &out, nullptr, g_stream));
        expect_combine_kernel(rank_major_kernel());
    }

    // Fills d_burst_in_[burst_index] with the dispatched rows plus burst_index * kBurstShift.
    void shift_dispatched(int burst_index) {
        const size_t n = static_cast<size_t>(g_nranks) * kLamMaxTokens * hidden();
        const float add = kBurstShift * burst_index;
        switch (dtype()) {
            case ncclFloat16:
                shift_rows_kernel<<<64, 256, 0, g_stream>>>(
                    static_cast<const __half*>(d_recv_), static_cast<__half*>(d_burst_in_[burst_index]), n, add);
                break;
            case ncclFloat32:
                shift_rows_kernel<<<64, 256, 0, g_stream>>>(
                    static_cast<const float*>(d_recv_), static_cast<float*>(d_burst_in_[burst_index]), n, add);
                break;
            default:
                shift_rows_kernel<<<64, 256, 0, g_stream>>>(
                    static_cast<const nv_bfloat16*>(d_recv_), static_cast<nv_bfloat16*>(d_burst_in_[burst_index]), n,
                    add);
                break;
        }
        CUDA_ASSERT(cudaGetLastError());
    }

    void check_output(int burst_index, int iteration, bool with_sentinels) {
        std::vector<uint32_t> host(kLamMaxTokens * row_words());
        CUDA_ASSERT(cudaMemcpy(host.data(), d_out_[burst_index], host.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const int per_word = static_cast<int>(sizeof(uint32_t)) / elem_bytes(dtype());
        for (int t = 0; t < kLamMaxTokens; ++t) {
            const uint32_t* row = host.data() + t * row_words();
            const float expected =
                distinct_destinations(routing_) * (token_value(iteration, g_rank, t) + kBurstShift * burst_index);
            for (int e = 0; e < hidden(); ++e) {
                const bool sentinel_word = with_sentinels && (e / per_word) % 2 == 0;
                if (sentinel_word) {
                    ASSERT_TRUE(element_is_nan(dtype(), row, e))
                        << "token " << t << " element " << e << " burst " << burst_index << " iteration " << iteration;
                } else {
                    ASSERT_EQ(element_value(dtype(), row, e), expected)
                        << "token " << t << " element " << e << " burst " << burst_index << " iteration " << iteration;
                }
            }
        }
    }

    // One dispatch followed by kBurst combines with no dispatch in between. One rank (rotating) is delayed before the
    // first combine so the others can run ahead of it; the combines take distinct inputs (shift_dispatched) and write
    // distinct outputs, and nothing synchronises until the end.
    void run_burst(int iteration, bool with_sentinels, int combines = kBurst) {
        upload_input(iteration, with_sentinels);
        CUDA_ASSERT(cudaMemset(d_recv_, 0, static_cast<size_t>(g_nranks) * kLamMaxTokens * row_bytes()));
        for (auto* p : d_out_) CUDA_ASSERT(cudaMemset(p, 0, kLamMaxTokens * row_bytes()));
        dispatch();
        for (int k = 0; k < combines; ++k) shift_dispatched(k);
        if (g_rank == iteration % g_nranks) spin_kernel<<<1, 1, 0, g_stream>>>(400000LL);
        for (int k = 0; k < combines; ++k) {
            if (k == 2 && g_rank == (iteration + 1) % g_nranks) spin_kernel<<<1, 1, 0, g_stream>>>(200000LL);
            combine(k, t_burst_in_[k]);
        }
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        for (int k = 0; k < combines; ++k) {
            check_output(k, iteration, with_sentinels);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
};

} // namespace

TEST_P(LamportCombineTest, BackToBackCombinesDense) {
    create_handle(Routing::kDense);
    for (int iteration = 0; iteration < 6; ++iteration) {
        run_burst(iteration, false);
        if (HasFatalFailure()) return;
    }
}

TEST_P(LamportCombineTest, BackToBackCombinesSparse) {
    create_handle(Routing::kSparse);
    for (int iteration = 0; iteration < 6; ++iteration) {
        run_burst(iteration, false);
        if (HasFatalFailure()) return;
    }
}

TEST_P(LamportCombineTest, SentinelPayloadWords) {
    for (Routing routing : {Routing::kDense, Routing::kSparse}) {
        create_handle(routing);
        for (int iteration = 0; iteration < 3; ++iteration) {
            run_burst(iteration, true);
            if (HasFatalFailure()) return;
        }
    }
}

TEST_P(LamportCombineTest, AlternatingBurstLengthsAndMaskClean) {
    create_handle(Routing::kSparse);
    for (int iteration = 0; iteration < 3; ++iteration) {
        run_burst(iteration, false);
        if (HasFatalFailure()) return;
    }
    // Collective: re-poisons the landing region and restarts the round protocol on every rank.
    NCCL_ASSERT(ncclEpMaskClean(group_, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    for (int iteration = 3; iteration < 6; ++iteration) {
        run_burst(iteration, iteration % 2 == 0);
        if (HasFatalFailure()) return;
    }
    // A fresh handle on the same group after the clean (a Lamport round counter that is not at its initial value
    // must keep working with the new handle).
    create_handle(Routing::kDense);
    run_burst(6, false);
}

// One group, two handles: D(rm) C(rm)^k with the Lamport kernel (when the call is eligible) interleaved with D(em) C(em),
// whose combine always runs the default kernel. 'R<k>' is a rank-major dispatch followed by k back-to-back combines, 'E'
// an expert-major dispatch and combine.
TEST_P(LamportCombineTest, MixedLamportAndFallbackCombinesOnOneGroup) {
    if (dtype() != ncclBfloat16) GTEST_SKIP() << "the expert-major handle is BF16";
    create_handle(Routing::kSparse);
    create_expert_major_handle();
    if (HasFatalFailure()) return;
    const std::vector<std::vector<int>> sequences = {
        {1, 0, 1, 0, 1, 0},       // R1 E R1 E R1 E
        {3, 0, 3, 0},             // R3 E R3 E
        {0, 3, 0, 0, 3},          // E R3 E E R3
        {0, 0, 5, 0, 5, 1, 0, 1}, // E E R5 E R5 R1 E R1
    };
    int iteration = 0;
    for (const auto& sequence : sequences) {
        for (int step : sequence) {
            if (step == 0) {
                run_expert_major(iteration);
            } else {
                run_burst(iteration, false, step);
            }
            ++iteration;
            if (HasFatalFailure()) return;
        }
    }
    if (g_rank == 0) {
        std::printf("combine kernels in order (L = Lamport, D = default): %s\n", kernel_sequence_.c_str());
    }
    RecordProperty("combine_kernels", kernel_sequence_);
    // The same on a dense rank-major handle.
    create_handle(Routing::kDense);
    for (int step : {3, 0, 2, 0, 0, 4}) {
        if (step == 0) {
            run_expert_major(iteration);
        } else {
            run_burst(iteration, false, step);
        }
        ++iteration;
        if (HasFatalFailure()) return;
    }
}

// Handles of different top-k share the group's Lamport region: it is sized for the largest top-k of the group, not
// something a handle's layout fixes. The top-1 handle is created first, so the region is created at capacity 1; the
// top-2 handle then grows the capacity (the group buffer is reallocated, the region re-created) and the top-1 handle
// must keep working in the larger region. Both are then interleaved, with several back-to-back combines each.
TEST_P(LamportCombineTest, HandlesOfDifferentTopkShareTheRegion) {
    if (dtype() != ncclBfloat16 || rank_major_kernel() != LlCombineKernel::k2SidedLamportRmLsa) {
        GTEST_SKIP() << "needs the BF16 Lamport case";
    }
    // Top-1 handle with its own tensors (the fixture's are sized for top-2).
    const size_t recv_rows = static_cast<size_t>(g_nranks) * kLamMaxTokens;
    ncclEpHandle_t h1 = nullptr;
    int32_t *d_topk1 = nullptr, *d_idx1 = nullptr;
    float *d_w1 = nullptr, *d_rw1 = nullptr;
    void *d_recv1 = nullptr, *d_out1 = nullptr;
    ncclEpTensor_t *t_topk1 = nullptr, *t_w1 = nullptr, *t_rw1 = nullptr, *t_idx1 = nullptr, *t_recv1 = nullptr,
                   *t_out1 = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk1, kLamMaxTokens * sizeof(int32_t)));
    CUDA_ASSERT(cudaMalloc(&d_idx1, recv_rows * sizeof(int32_t)));
    CUDA_ASSERT(cudaMalloc(&d_w1, kLamMaxTokens * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_rw1, recv_rows * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv1, recv_rows * row_bytes()));
    CUDA_ASSERT(cudaMalloc(&d_out1, kLamMaxTokens * row_bytes()));
    std::vector<int32_t> topk1(kLamMaxTokens, destination_rank(Routing::kSparse, g_rank, 0));
    std::vector<float> ones(kLamMaxTokens, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_topk1, topk1.data(), topk1.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w1, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));
    NCCL_ASSERT(epTensorCreate(&t_topk1, 2, ncclInt32, d_topk1, kLamMaxTokens, 1));
    NCCL_ASSERT(epTensorCreate(&t_w1, 2, ncclFloat32, d_w1, kLamMaxTokens, 1));
    NCCL_ASSERT(epTensorCreate(&t_rw1, 3, ncclFloat32, d_rw1, g_nranks, kLamMaxTokens, 1));
    NCCL_ASSERT(epTensorCreate(&t_idx1, 3, ncclInt32, d_idx1, g_nranks, kLamMaxTokens, 1));
    NCCL_ASSERT(epTensorCreate(&t_recv1, 3, dtype(), d_recv1, g_nranks, kLamMaxTokens, hidden()));
    NCCL_ASSERT(epTensorCreate(&t_out1, 2, dtype(), d_out1, kLamMaxTokens, hidden()));
    NCCL_ASSERT(ncclEpCreateHandle(&h1, group_, NCCL_EP_LAYOUT_RANK_MAJOR, t_topk1, nullptr, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // One burst on the top-1 handle: a dispatch and `combines` back-to-back combines, each checked.
    auto top1_burst = [&](int iteration, int combines) {
        upload_input(iteration, false);
        CUDA_ASSERT(cudaMemset(d_recv1, 0, recv_rows * row_bytes()));
        ncclEpDispatchInputs_t din = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t dout = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
        din.tokens = t_input_;
        din.topk_weights = t_w1;
        dout.tokens = t_recv1;
        dout.topk_weights = t_rw1;
        dout.topk_idx = t_idx1;
        NCCL_ASSERT(ncclEpDispatch(h1, &din, &dout, &layout, nullptr, g_stream));
        NCCL_ASSERT(ncclEpComplete(h1, nullptr, g_stream));
        for (int k = 0; k < combines; ++k) {
            CUDA_ASSERT(cudaMemset(d_out1, 0, kLamMaxTokens * row_bytes()));
            ncclEpCombineInputs_t cin = NCCL_EP_COMBINE_INPUTS_INIT;
            ncclEpCombineOutputs_t cout = NCCL_EP_COMBINE_OUTPUTS_INIT;
            cin.tokens = t_recv1;
            cout.tokens = t_out1;
            NCCL_ASSERT(ncclEpCombine(h1, &cin, &cout, nullptr, g_stream));
            expect_combine_kernel(LlCombineKernel::k2SidedLamportRmLsa);
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
            std::vector<uint32_t> host(kLamMaxTokens * row_words());
            CUDA_ASSERT(cudaMemcpy(host.data(), d_out1, host.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (int t = 0; t < kLamMaxTokens; ++t) {
                for (int e = 0; e < hidden(); ++e) {
                    ASSERT_EQ(element_value(dtype(), host.data() + t * row_words(), e), token_value(iteration, g_rank, t))
                        << "top-1 token " << t << " element " << e << " combine " << k << " iteration " << iteration;
                }
            }
        }
    };

    top1_burst(0, 3);
    if (!HasFatalFailure()) {
        create_handle(Routing::kSparse); // top-2: grows the region's capacity
        for (int round = 1; round < 7 && !HasFatalFailure(); ++round) {
            if (round % 2 == 1) {
                run_burst(round, false, 1 + round % 4);
            } else {
                top1_burst(round, 1 + round % 3);
            }
        }
    }
    if (g_rank == 0) {
        std::printf("combine kernels in order (L = Lamport, D = default): %s\n", kernel_sequence_.c_str());
    }
    ncclEpHandleDestroy(h1);
    for (auto* t : {t_topk1, t_w1, t_rw1, t_idx1, t_recv1, t_out1}) if (t) ncclEpTensorDestroy(t);
    for (auto* p : {static_cast<void*>(d_topk1), static_cast<void*>(d_idx1), static_cast<void*>(d_w1),
                    static_cast<void*>(d_rw1), d_recv1, d_out1}) {
        if (p) cudaFree(p);
    }
}

// A group with a user-specified rdma_buffer_size is allocated and registered at creation, so its first Lamport handle
// finds a live buffer that earlier handles have already used. The region is created in place at the tail of that
// buffer, with no reallocation: the expert-major handle that has run before it, and the Lamport handles after it, must
// keep working.
TEST_P(LamportCombineTest, FixedSizeGroupCreatesTheRegionInPlace) {
    if (dtype() != ncclBfloat16 || rank_major_kernel() != LlCombineKernel::k2SidedLamportRmLsa) {
        GTEST_SKIP() << "needs the BF16 Lamport case";
    }
    recreate_group(explicit_buffer_bytes(kLamTopk), 0);
    create_expert_major_handle();
    if (HasFatalFailure()) return;
    int iteration = 0;
    for (int i = 0; i < 3; ++i) {
        run_expert_major(iteration++);
        if (HasFatalFailure()) return;
    }
    const void* segment0 = nullptr;
    size_t stride = 0, slot_bytes = 0;
    EXPECT_EQ(ncclEpGroup_test_getLamportSegments(group_, &segment0, &stride, &slot_bytes), ncclInvalidUsage);

    create_handle(Routing::kDense);
    NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0, &stride, &slot_bytes));
    for (int step : {3, 0, 2, 0, 4}) {
        if (step == 0) {
            run_expert_major(iteration);
        } else {
            run_burst(iteration, false, step);
        }
        ++iteration;
        if (HasFatalFailure()) return;
    }
    // A second rank-major handle of the same top-k finds the region large enough and leaves it alone.
    const void* segment0_again = nullptr;
    create_handle(Routing::kSparse);
    NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0_again, &stride, &slot_bytes));
    EXPECT_EQ(segment0_again, segment0);
    for (int step : {2, 0, 3}) {
        if (step == 0) {
            run_expert_major(iteration);
        } else {
            run_burst(iteration, false, step);
        }
        ++iteration;
        if (HasFatalFailure()) return;
    }
}

// A group that declares a top-k bound (ncclEpGroupConfig_t::num_topk) gets the Lamport region at that capacity from the
// start: in a user-specified buffer at group creation, in an auto-sized one at its first Lamport handle. A handle of a
// smaller top-k then finds the region large enough, and the region does not move.
TEST_P(LamportCombineTest, DeclaredTopkSizesTheRegionFromTheStart) {
    if (dtype() != ncclBfloat16 || rank_major_kernel() != LlCombineKernel::k2SidedLamportRmLsa) {
        GTEST_SKIP() << "needs the BF16 Lamport case";
    }
    const unsigned int kDeclared = static_cast<unsigned int>(g_nranks); // the group has g_nranks experts
    const nccl_ep::LamportRegion expected =
        nccl_ep::lamport_region_shape(kLamMaxTokens, row_bytes(), g_nranks, kDeclared);
    const void* segment0 = nullptr;
    size_t stride = 0, slot_bytes = 0;

    // User-specified buffer: the region exists before any handle.
    recreate_group(explicit_buffer_bytes(kDeclared), kDeclared);
    NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0, &stride, &slot_bytes));
    EXPECT_EQ(stride, expected.segment_stride);
    create_handle(Routing::kDense);
    const void* segment0_after = nullptr;
    NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0_after, &stride, &slot_bytes));
    EXPECT_EQ(segment0_after, segment0);
    EXPECT_EQ(stride, expected.segment_stride);
    run_burst(0, false, 3);
    if (HasFatalFailure()) return;

    // Auto-sized buffer: no region until the first Lamport handle, which creates it at the declared capacity.
    recreate_group(NCCL_EP_AUTO, kDeclared);
    EXPECT_EQ(ncclEpGroup_test_getLamportSegments(group_, &segment0, &stride, &slot_bytes), ncclInvalidUsage);
    create_handle(Routing::kDense);
    NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0, &stride, &slot_bytes));
    EXPECT_EQ(stride, expected.segment_stride);
    create_handle(Routing::kSparse);
    NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0_after, &stride, &slot_bytes));
    EXPECT_EQ(segment0_after, segment0);
    run_burst(1, false, 3);
}

// A handle above the group's declared top-k bound is rejected.
TEST_P(LamportCombineTest, HandleAboveTheDeclaredTopkIsRejected) {
    if (dtype() != ncclBfloat16 || rank_major_kernel() != LlCombineKernel::k2SidedLamportRmLsa) {
        GTEST_SKIP() << "needs the BF16 Lamport case";
    }
    recreate_group(NCCL_EP_AUTO, kLamTopk - 1);
    CUDA_ASSERT(cudaMemset(d_topk_, 0, kLamMaxTokens * kLamTopk * sizeof(int32_t)));
    ncclEpHandle_t rejected = nullptr;
    EXPECT_EQ(
        ncclEpCreateHandle(&rejected, group_, NCCL_EP_LAYOUT_RANK_MAJOR, t_topk_, nullptr, nullptr, g_stream),
        ncclInvalidUsage);
}

// A user-specified buffer that cannot hold the region at the declared top-k is rejected at group creation.
TEST_P(LamportCombineTest, UserBufferTooSmallForTheDeclaredRegionIsRejected) {
    if (dtype() != ncclBfloat16 || rank_major_kernel() != LlCombineKernel::k2SidedLamportRmLsa) {
        GTEST_SKIP() << "needs the BF16 Lamport case";
    }
    const unsigned int kDeclared = static_cast<unsigned int>(g_nranks); // the group has g_nranks experts
    const nccl_ep::LamportRegion shape =
        nccl_ep::lamport_region_shape(kLamMaxTokens, row_bytes(), g_nranks, kDeclared);
    NCCL_ASSERT(ncclEpGroupDestroy(group_));
    group_ = nullptr;
    EXPECT_EQ(try_create_group(shape.reserved_bytes - 1, kDeclared), ncclInvalidUsage);
    EXPECT_EQ(group_, nullptr);
}

// Fault injection: a destination whose round check times out must receive no stores at all from that warp, not only
// from its first token. Rank 2 is delayed, so rank 0 (which receives from ranks 1 and 2) stays in round 0's receive;
// rank 1 runs ahead with a short timeout, and its third combine (round 2) cannot pass the gate for rank 0, whose
// segment 0 still holds round 0's data. Each of rank 1's combines contributes a different value, so what sits in
// rank 0's segment 0 after the timeout tells whether round 2's tokens were stored over round 0's.
TEST_P(LamportCombineTest, TimedOutRoundGateStoresNothing) {
    if (g_nranks != 4 || dtype() != ncclBfloat16 || rank_major_kernel() != LlCombineKernel::k2SidedLamportRmLsa) {
        GTEST_SKIP() << "needs four ranks and the BF16 Lamport case";
    }
    constexpr int kReceiver = 0, kEarly = 1, kLate = 2, kRounds = 3;
    create_handle(Routing::kFar);
    run_burst(0, false, kRounds); // warm-up: every kernel is compiled and the round counter has advanced
    if (HasFatalFailure()) return;

    const size_t recv_elems = static_cast<size_t>(g_nranks) * kLamMaxTokens * hidden();
    nv_bfloat16* d_shift[kRounds] = {};
    ncclEpTensor_t* t_shift[kRounds] = {};
    int* d_barrier = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_barrier, sizeof(int)));
    CUDA_ASSERT(cudaMemset(d_barrier, 0, sizeof(int)));
    upload_input(1, false);
    CUDA_ASSERT(cudaMemset(d_recv_, 0, recv_elems * sizeof(nv_bfloat16)));
    dispatch();
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    for (int k = 0; k < kRounds; ++k) {
        CUDA_ASSERT(cudaMalloc(&d_shift[k], recv_elems * sizeof(nv_bfloat16)));
        shift_kernel<<<64, 256, 0, g_stream>>>(static_cast<const nv_bfloat16*>(d_recv_), d_shift[k], recv_elems, 10.0f * k);
        NCCL_ASSERT(epTensorCreate(&t_shift[k], 3, ncclBfloat16, d_shift[k], g_nranks, kLamMaxTokens, hidden()));
    }
    for (auto* p : d_out_) CUDA_ASSERT(cudaMemset(p, 0, kLamMaxTokens * row_bytes()));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    int clock_khz = 0;
    CUDA_ASSERT(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const auto cycles = [&](double ms) { return static_cast<unsigned long long>(ms * clock_khz); };
    const unsigned long long default_cycles = ncclEpGroup_test_getTimeoutCycles(group_);
    // Per-rank timeouts: the early rank gives up quickly; the others wait longer than the late rank's delay, and
    // only a few seconds for data that a closed gate will never deliver (the default would be 97 s).
    NCCL_ASSERT(ncclEpGroup_test_setTimeoutCycles(group_, g_rank == kEarly ? cycles(20) : cycles(5000)));
    // All ranks are past their set-up before anyone launches (host-side barrier).
    NCCL_ASSERT(ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclSum, g_comm, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    if (g_rank == kLate) spin_kernel<<<1, 1, 0, g_stream>>>(static_cast<long long>(cycles(600)));
    for (int k = 0; k < kRounds; ++k) combine(k, t_shift[k]);

    if (g_rank == kReceiver) {
        // The late rank's data has not arrived: the receiver is still in the first round of the scenario, and rank 1 has
        // long since given up on its third. The scenario's rounds are counted from the warm-up's: the first one uses
        // segment (kRounds & 1), the second the other segment, the third the first again, which is the one the closed
        // gate protects. That segment must still hold rank 1's first value; the other one holds rank 1's second
        // value, which passed its gate (a control that the sampling reads real data).
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const void* segment0 = nullptr;
        size_t stride = 0, slot_bytes = 0;
        NCCL_ASSERT(ncclEpGroup_test_getLamportSegments(group_, &segment0, &stride, &slot_bytes));
        cudaStream_t sampler;
        CUDA_ASSERT(cudaStreamCreateWithFlags(&sampler, cudaStreamNonBlocking));
        const int protected_segment = kRounds & 1, control_segment = (kRounds + 1) & 1;
        for (int t = 0; t < kLamMaxTokens; ++t) {
            // Rank 1 is slot 0 of rank 0's token t (destinations are ranks 1 and 2).
            const size_t slot_offset = static_cast<size_t>(t * kLamTopk) * slot_bytes;
            auto sample = [&](int segment) {
                nv_bfloat16 element{};
                const char* slot = static_cast<const char*>(segment0) + segment * stride + slot_offset;
                EXPECT_EQ(cudaMemcpyAsync(&element, slot, sizeof(element), cudaMemcpyDeviceToHost, sampler), cudaSuccess);
                EXPECT_EQ(cudaStreamSynchronize(sampler), cudaSuccess);
                return __bfloat162float(element);
            };
            EXPECT_EQ(sample(protected_segment), token_value(1, g_rank, t))
                << "token " << t << ": the protected segment was written after the round gate timed out";
            EXPECT_EQ(sample(control_segment), token_value(1, g_rank, t) + 10.0f)
                << "token " << t << ": rank 1's second round did not arrive (the sampling is not reading real data)";
        }
        cudaStreamDestroy(sampler);
    }
    CUDA_ASSERT(cudaStreamSynchronize(g_stream)); // the receiver's round 2 times out waiting for rank 1
    if (g_rank == kReceiver) check_output(0, 1, false);

    NCCL_ASSERT(ncclEpGroup_test_setTimeoutCycles(group_, default_cycles));
    NCCL_ASSERT(ncclEpMaskClean(group_, g_stream)); // collective: unmask, re-poison, restart the rounds
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    for (auto* t : t_shift) if (t) ncclEpTensorDestroy(t);
    for (auto* p : d_shift) if (p) cudaFree(p);
    cudaFree(d_barrier);
}

INSTANTIATE_TEST_SUITE_P(
    Lamport, LamportCombineTest, ::testing::ValuesIn(kCases),
    [](const ::testing::TestParamInfo<LamportCase>& info) { return std::string(info.param.name); });

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_ll_lamport_combine_uid")) return 0;
    const int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
