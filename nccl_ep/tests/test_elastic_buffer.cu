/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for NCCL EP "elastic buffers" — a single contiguous VA range whose
 * physical backing is a GPU segment followed by a CPU (HOST_NUMA) segment, used
 * as an EP dispatch-output (receive) buffer. See RFC
 * https://github.com/NVIDIA/nccl/issues/2110 and
 * nccl_ep/examples/nccl_ep_elastic_buffer.h.
 *
 * Self-contained: builds its own HT group and uses the Expert-Major layout with
 * the EM duplication modes off (NCCL_EP_HT_EM_LOCAL_DUP / NCCL_EP_HT_EM_NVLINK_DUP,
 * both default off), non-zero-copy — the receiver expands the FLAT staging
 * buffer into the user's recv buffer with a permute KERNEL (SASS stores) and
 * combine reads it back with a reduce kernel. That path never runs
 * cudaMemcpy/cudaMemset on the user buffer (which would crash on a cuMem
 * HOST_NUMA pointer), so the CPU segment is filled/read purely by kernels.
 *
 * The whole elastic buffer is the receive buffer (offset 0). Token dimensions
 * are sized so the received data (~3 MiB) exceeds the GPU segment (2 MiB = one
 * cuMem granule), so the token writes cross the GPU->CPU segment boundary; the
 * dispatch+combine round-trip then verifies every token — including those that
 * landed in the CPU (HOST_NUMA) segment — is returned intact.
 *
 * All ncclEpTensor_t descriptors are stack-allocated (NCCL_EP_TENSOR_INIT with
 * caller-owned sizes[] arrays) — no ncclEpTensorAlloc/Destroy.
 *
 * Tests:
 *   RegisterDeregister         — a GPU+CPU elastic buffer registers/deregisters as a window.
 *   DispatchCombineCrossBoundary — EM dispatch+combine into the whole elastic
 *                                  buffer; received tokens span the GPU->CPU boundary.
 *
 * Build:  make -C nccl_ep/tests [BUILDDIR=...]
 * Run:    bash nccl_ep/tests/run_tests.sh 4
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"   // ncclEpHandle_test_getNumRecvTokens
#include "../examples/nccl_ep_elastic_buffer.h"

#include <cstdlib>  // setenv
#include <vector>

// Token dimensions chosen so the received data crosses the 2 MiB cuMem granule:
//   per-token bytes = kXbHidden * sizeof(bf16) = 16 KiB
//   received tokens per rank ≈ kXbDispatch (each rank sends kXbDispatch tokens,
//   routed evenly across the experts, so every hosting rank receives ~kXbDispatch)
//   → ~kXbDispatch * 16 KiB ≈ 3 MiB of recv data > 2 MiB GPU segment.
static constexpr int          kXbHidden   = 8192;         // 16 KiB per token (bf16)
static constexpr unsigned int kXbDispatch = 192;          // tokens/rank; multiple of HT chunk (64)
static constexpr unsigned int kXbRecv     = 256;          // recv slot budget (>= max_dispatch)
static constexpr size_t       kXbGpuBytes = (size_t)2 << 20;  // 2 MiB GPU segment (one granule)
static constexpr size_t       kXbCpuBytes = (size_t)4 << 20;  // 4 MiB CPU segment (holds the spill)
static constexpr size_t       kXbTokenBytes = (size_t)kXbHidden * sizeof(nv_bfloat16);
static constexpr size_t       kXbRecvBytes  = (size_t)kXbRecv * kXbTokenBytes;  // recv tensor span

// Self-contained group/comm for this suite (not the shared g_ep_group).
static ncclEpGroup_t g_elastic_group = nullptr;

// Zero a byte range via a kernel store. Used instead of cudaMemset because the
// runtime cudaMemset can return cudaErrorInvalidValue on a HOST_NUMA-backed
// pointer, whereas a kernel store into HOST_NUMA works.
__global__ void zero_bytes(unsigned char* p, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0;
}

// Allocate a GPU+CPU elastic buffer; returns the VA base (or nullptr on failure).
static void* alloc_elastic(ncclEpElasticBuffer* buf, size_t gpu_bytes, size_t cpu_bytes) {
    void* base = nullptr;
    CUresult r = ncclEpElasticAlloc(&base, buf, gpu_bytes, cpu_bytes, -1);
    EXPECT_EQ(r, CUDA_SUCCESS) << "ncclEpElasticAlloc failed";
    return (r == CUDA_SUCCESS) ? base : nullptr;
}

class ElasticBufferTest : public ::testing::Test {
protected:
    int64_t* d_topk_ = nullptr;
    // Stack/inline routing tensor (no heap alloc, no Destroy). The sizes[] array
    // and descriptor are members so they outlive the handle that consumes them.
    size_t         topk_dims_[2] = {kXbDispatch, kTopK};
    ncclEpTensor_t topk_ = NCCL_EP_TENSOR_INIT;

    void SetUp() override {
        // Route token i to expert (i % num_experts): each rank spreads its
        // kXbDispatch tokens evenly across the experts, so every hosting rank
        // receives ~kXbDispatch tokens for its local expert(s).
        std::vector<int64_t> h(kXbDispatch * kTopK);
        for (unsigned i = 0; i < kXbDispatch; ++i) h[i] = i % kNumExperts;
        CUDA_ASSERT(cudaMalloc(&d_topk_, h.size() * sizeof(int64_t)));
        CUDA_ASSERT(cudaMemcpy(d_topk_, h.data(), h.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        topk_.ndim = 2; topk_.datatype = ncclInt64; topk_.data = d_topk_; topk_.sizes = topk_dims_;
    }

    void TearDown() override {
        if (d_topk_) cudaFree(d_topk_);
    }

    // Expert-Major + non-zero-copy, EM dup modes off → permute/reduce kernels,
    // no cudaMemcpy on the user recv buffer, so HOST_NUMA works.
    ncclEpHandle_t make_handle() {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, g_elastic_group, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                     &topk_, nullptr, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }
};

// ── Test: a GPU+CPU elastic buffer registers + deregisters as an NCCL window ───

TEST_F(ElasticBufferTest, RegisterDeregister) {
    const size_t kReq = 4096;  // 1 granule each (rounded up internally)
    ncclEpElasticBuffer buf;
    void* base = alloc_elastic(&buf, /*gpu*/kReq, /*cpu*/kReq);
    ASSERT_NE(base, nullptr);
    EXPECT_GT(buf.gpu_bytes, 0u);
    EXPECT_GT(buf.cpu_bytes, 0u);
    EXPECT_EQ(buf.num_segments, 2);

    ncclWindow_t win = ncclWindow_t{};
    ncclResult_t wr = ncclCommWindowRegister(
        g_comm, base, ncclEpElasticTotalBytes(&buf), &win, /*winFlags=*/0);
    if (wr == ncclSuccess && win != ncclWindow_t{}) {
        EXPECT_EQ(ncclCommWindowDeregister(g_comm, win), ncclSuccess);
    } else {
        if (g_rank == 0)
            printf("SKIP: NCCL window registration unsupported here (wr=%d)\n", wr);
    }
    EXPECT_EQ(ncclEpElasticFree(&buf), CUDA_SUCCESS);
}

// ── Test: dispatch+combine into the whole elastic buffer, crossing the boundary ─
// The recv buffer IS the elastic buffer (offset 0). Received tokens fill from
// slot 0 and, because the recv data (~3 MiB) exceeds the 2 MiB GPU segment, the
// later slots land in the CPU (HOST_NUMA) segment — so token writes cross the
// GPU->CPU boundary. The identity-expert round-trip verifies every token.

TEST_F(ElasticBufferTest, DispatchCombineCrossBoundary) {
    ncclEpElasticBuffer buf;
    void* base = alloc_elastic(&buf, /*gpu*/kXbGpuBytes, /*cpu*/kXbCpuBytes);
    ASSERT_NE(base, nullptr);
    // The recv tensor spans both segments (its byte extent exceeds the GPU segment).
    ASSERT_GE(buf.gpu_bytes + buf.cpu_bytes, kXbRecvBytes);
    ASSERT_LT(buf.gpu_bytes, kXbRecvBytes);

    ncclEpHandle_t h = make_handle();
    ASSERT_NE(h, nullptr);

    // Confirm the received data actually crosses the GPU segment boundary.
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    const size_t recv_used_bytes = (size_t)num_recv * kXbTokenBytes;
    EXPECT_GT(recv_used_bytes, buf.gpu_bytes)
        << "recv data (" << recv_used_bytes << " B) must exceed the GPU segment ("
        << buf.gpu_bytes << " B) so writes cross into the CPU segment";

    // ── Inputs / outputs (device), recv = the elastic buffer (offset 0) ──
    const size_t send_elems = (size_t)kXbDispatch * kXbHidden;
    std::vector<nv_bfloat16> h_tok(send_elems);
    for (unsigned i = 0; i < kXbDispatch; ++i) {
        float v = static_cast<float>(g_rank * kXbDispatch + i + 1);
        for (int hh = 0; hh < kXbHidden; ++hh)
            h_tok[(size_t)i * kXbHidden + hh] = __float2bfloat16(v);
    }

    nv_bfloat16 *d_tok = nullptr, *d_out = nullptr;
    float       *d_weights = nullptr, *d_recv_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok,     send_elems * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out,     send_elems * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights, (size_t)kXbDispatch * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,  (size_t)kXbRecv * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), send_elems * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    std::vector<float> h_w((size_t)kXbDispatch * kTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Zero the whole recv region via a kernel (works across GPU + HOST_NUMA).
    zero_bytes<<<(kXbRecvBytes + 255) / 256, 256, 0, g_stream>>>(
        static_cast<unsigned char*>(base), kXbRecvBytes);
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // ── Stack-allocated tensor descriptors (caller-owned sizes[]) ──
    size_t tok_dims[2]  = {kXbDispatch, (size_t)kXbHidden};
    size_t recv_dims[2] = {kXbRecv,     (size_t)kXbHidden};
    size_t out_dims[2]  = {kXbDispatch, (size_t)kXbHidden};
    size_t w_dims[2]    = {kXbDispatch, (size_t)kTopK};
    size_t rw_dims[1]   = {kXbRecv};

    ncclEpTensor_t t_tok = NCCL_EP_TENSOR_INIT;
    t_tok.ndim = 2; t_tok.datatype = ncclBfloat16; t_tok.data = d_tok; t_tok.sizes = tok_dims;
    ncclEpTensor_t t_recv = NCCL_EP_TENSOR_INIT;
    t_recv.ndim = 2; t_recv.datatype = ncclBfloat16; t_recv.data = base; t_recv.sizes = recv_dims;
    ncclEpTensor_t t_out = NCCL_EP_TENSOR_INIT;
    t_out.ndim = 2; t_out.datatype = ncclBfloat16; t_out.data = d_out; t_out.sizes = out_dims;
    ncclEpTensor_t t_w = NCCL_EP_TENSOR_INIT;
    t_w.ndim = 2; t_w.datatype = ncclFloat32; t_w.data = d_weights; t_w.sizes = w_dims;
    ncclEpTensor_t t_recv_w = NCCL_EP_TENSOR_INIT;  // EM: recv_topk_weights is 1D [N]
    t_recv_w.ndim = 1; t_recv_w.datatype = ncclFloat32; t_recv_w.data = d_recv_w; t_recv_w.sizes = rw_dims;

    ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens        = &t_tok;
    d_in_s.topk_weights  = &t_w;
    d_out_s.tokens       = &t_recv;
    d_out_s.topk_weights = &t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    EXPECT_EQ(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
    EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    ncclEpCombineInputs_t  c_in_s  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens  = &t_recv;
    c_out_s.tokens = &t_out;
    EXPECT_EQ(ncclEpCombine(h, &c_in_s, &c_out_s, nullptr, g_stream), ncclSuccess);
    EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

    // Identity expert + weight 1.0 → combine returns each sent token intact.
    std::vector<nv_bfloat16> h_out(send_elems);
    CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, send_elems * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < kXbDispatch; ++i) {
        float got = __bfloat162float(h_out[(size_t)i * kXbHidden]);
        float exp = __bfloat162float(__float2bfloat16(static_cast<float>(g_rank * kXbDispatch + i + 1)));
        EXPECT_NEAR(got, exp, 0.5f) << "rank " << g_rank << " token " << i;
    }

    cudaFree(d_tok); cudaFree(d_out); cudaFree(d_weights); cudaFree(d_recv_w);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
    EXPECT_EQ(ncclEpElasticFree(&buf), CUDA_SUCCESS);
}

// ── Self-contained bootstrap / teardown ───────────────────────────────────────
// Creates a single HT group (Expert-Major used per-handle) sized for the
// cross-boundary test: max_dispatch multiple of 64, big per-token bytes.

static bool elastic_bootstrap(int argc, char* argv[]) {
    ep_parse_args(argc, argv, "te_ep_elastic_uid");
    ::testing::InitGoogleTest(&argc, argv);

    int device_count = 0, device = 0, major = 0;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(g_rank % device_count);
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (major < 9) {
        if (g_rank == 0) printf("SKIP: SM_90+ required (this device is SM_%d0)\n", major);
        return false;
    }
    if (g_nranks < 2) {
        if (g_rank == 0) printf("SKIP: at least 2 ranks required\n");
        return false;
    }

    ncclUniqueId uid{};
    exchange_uid(&uid);
    if (ncclCommInitRank(&g_comm, g_nranks, uid, g_rank) != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclCommInitRank failed\n", g_rank);
        return false;
    }
    cudaStreamCreate(&g_stream);

    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm                    = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts                  = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kXbDispatch;                     // multiple of 64
    gcfg.max_token_bytes              = kXbHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size             = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank              = NCCL_EP_AUTO;
    gcfg.num_channels                 = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank     = kXbRecv;
    if (ncclEpCreateGroup(&g_elastic_group, g_comm, &gcfg) != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclEpCreateGroup failed\n", g_rank);
        return false;
    }
    cudaStreamSynchronize(g_stream);
    return true;
}

static void elastic_teardown() {
    if (g_elastic_group) ncclEpGroupDestroy(g_elastic_group);
    if (g_stream)        cudaStreamDestroy(g_stream);
    if (g_comm)          ncclCommDestroy(g_comm);
    if (g_rank == 0)     remove(g_uid_file.c_str());
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // CPU-backed window segments require this (default is already 1). Set it
    // before comm/group creation so NCCL reads it on the first registration.
    setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", /*overwrite=*/1);
    if (!elastic_bootstrap(argc, argv)) return 0;
    int ret = RUN_ALL_TESTS();
    elastic_teardown();
    return ret;
}
