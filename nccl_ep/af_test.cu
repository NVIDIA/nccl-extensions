/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

// AFD (Attention-FFN Disaggregation) round-trip test for NCCL EP.
//
// Low-Latency algorithm, rank-major layout. The communicator's ranks are
// partitioned (config.num_attn_ranks):
//   ranks [0, num_attn_ranks)        -> A-side (attention): own no experts,
//                                       dispatch tokens, receive combined results.
//   ranks [num_attn_ranks, nRanks)   -> F-side (experts): own all experts,
//                                       receive dispatched tokens, return outputs.
//
// Each iteration: A-ranks dispatch tokens to the F-ranks; F-ranks run an
// identity "expert" (echo recv_x) and return. F-ranks validate the dispatched
// payload and routing metadata; A-ranks validate the combined result. Rank-major
// combine returns one echoed contribution per distinct destination F-rank.
//
// Run (2 attention + 2 expert ranks, 8 experts):
//   mpirun -np 4 ./af_test -A 2 -e 8

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_ep.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <vector>

static cudaError_t torchMalloc(void** ptr, size_t size, void* /*context*/) { return cudaMalloc(ptr, size); }
static cudaError_t torchFree(void* ptr, void* /*context*/) { return cudaFree(ptr); }

static size_t epDtypeBytes(ncclDataType_t dt) {
    switch (dt) {
        case ncclInt8:    case ncclUint8:                   return 1;
        case ncclFloat16: case ncclBfloat16:                return 2;
        case ncclFloat32: case ncclInt32: case ncclUint32:  return 4;
        case ncclInt64:   case ncclUint64: case ncclFloat64:return 8;
        default: return 0;
    }
}

// cudaMalloc the backing buffer and wrap it in a heap-allocated descriptor.
// A zero-element tensor (s0==0) gets a 1-byte backing allocation so the
// descriptor has a valid (non-null) data pointer.
static ncclResult_t epMakeTensor(ncclEpTensor_t** out_tensor, unsigned int ndim,
                                 ncclDataType_t dt,
                                 unsigned int s0, unsigned int s1 = 1, unsigned int s2 = 1) {
    if (out_tensor == nullptr) return ncclInvalidArgument;
    *out_tensor = nullptr;
    size_t dims[3] = {s0, s1, s2};
    size_t total = 1;
    for (unsigned int i = 0; i < ndim; i++) total *= dims[i];
    void* data = nullptr;
    cudaError_t e = cudaMalloc(&data, (total ? total : 1) * epDtypeBytes(dt));
    if (e != cudaSuccess) return ncclSystemError;
    ncclResult_t r = ncclEpTensorAlloc(out_tensor, ndim, dt, dims, /*config=*/nullptr);
    if (r != ncclSuccess) { cudaFree(data); return r; }
    (*out_tensor)->data = data;
    return ncclSuccess;
}

static void epFreeTensor(ncclEpTensor_t** field) {
    if (field == nullptr || *field == nullptr) return;
    void* data = (*field)->data;
    ncclEpTensorDestroy(*field);
    *field = nullptr;
    if (data) cudaFree(data);
}

// Collectively register one dispatch-output window. F-ranks contribute their
// full receive allocation; A-ranks contribute only a one-byte placeholder.
// NCCL window registration supports asymmetric sizes, while still requiring
// every communicator rank to participate in the same registration sequence.
static ncclResult_t epMakeAsymmetricRecvWindow(
    ncclEpTensor_t** out_tensor,
    void** out_data,
    ncclWindow_t* out_window,
    bool is_attn,
    unsigned int nRanks,
    unsigned int max_tokens,
    unsigned int hidden,
    ncclComm_t comm) {
    if (out_tensor == nullptr || out_data == nullptr || out_window == nullptr)
        return ncclInvalidArgument;
    *out_tensor = nullptr;
    *out_data = nullptr;
    *out_window = ncclWindow_t{};

    const size_t recv_bytes = static_cast<size_t>(nRanks) * max_tokens * hidden * sizeof(uint16_t);
    const size_t local_bytes = is_attn ? 1 : recv_bytes;
    ncclResult_t r = ncclMemAlloc(out_data, local_bytes);
    if (r != ncclSuccess) return r;
    r = ncclCommWindowRegister(
        comm, *out_data, local_bytes, out_window, NCCL_WIN_COLL_SYMMETRIC);
    if (r != ncclSuccess) {
        ncclMemFree(*out_data);
        *out_data = nullptr;
        return r;
    }

    // F-ranks expose the real rank-major receive shape. A-ranks only carry the
    // collective window handle used by their send path to address F-rank peers;
    // the placeholder itself is never a dispatch destination.
    size_t dims[3] = {is_attn ? 1u : nRanks, is_attn ? 1u : max_tokens, is_attn ? 1u : hidden};
    const unsigned int ndim = is_attn ? 1u : 3u;
    const ncclDataType_t dtype = is_attn ? ncclUint8 : ncclBfloat16;
    r = ncclEpTensorAlloc(out_tensor, ndim, dtype, dims, /*config=*/nullptr);
    if (r != ncclSuccess) {
        ncclCommWindowDeregister(comm, *out_window);
        ncclMemFree(*out_data);
        *out_window = ncclWindow_t{};
        *out_data = nullptr;
        return r;
    }
    (*out_tensor)->win_hdl = *out_window;
    (*out_tensor)->win_offset = 0;
    return ncclSuccess;
}

static ncclResult_t epFreeAsymmetricRecvWindow(
    ncclEpTensor_t** tensor,
    void** data,
    ncclWindow_t* window,
    ncclComm_t comm) {
    if (tensor != nullptr && *tensor != nullptr) {
        ncclResult_t r = ncclEpTensorDestroy(*tensor);
        if (r != ncclSuccess) return r;
        *tensor = nullptr;
    }
    if (window != nullptr && *window != ncclWindow_t{}) {
        ncclResult_t r = ncclCommWindowDeregister(comm, *window);
        if (r != ncclSuccess) return r;
        *window = ncclWindow_t{};
    }
    if (data != nullptr && *data != nullptr) {
        ncclResult_t r = ncclMemFree(*data);
        if (r != ncclSuccess) return r;
        *data = nullptr;
    }
    return ncclSuccess;
}

#define MPICHECK(cmd) do { int e = cmd; if (e != MPI_SUCCESS) { \
    printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); exit(EXIT_FAILURE); } } while(0)
#define CUDACHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
    printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while(0)
#define NCCLCHECK(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { \
    printf("Failed: NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); exit(EXIT_FAILURE); } } while(0)

static uint64_t getHostHash(const char* string) {
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) result = ((result << 5) + result) + string[c];
    return result;
}

static inline uint16_t float_to_bf16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t rounding_bias = 0x00007FFF + ((x >> 16) & 1);
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

static inline float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Stable routing that every rank can reconstruct without exchanging a CPU
// reference. Consecutive experts deliberately exercise same-rank deduplication.
static inline int routedExpert(int src_rank, int token, int topk, unsigned int num_experts) {
    uint32_t x = static_cast<uint32_t>(src_rank + 1) * 0x9e3779b9u;
    x ^= static_cast<uint32_t>(token + 1) * 0x85ebca6bu;
    x ^= x >> 16;
    return static_cast<int>((x % num_experts + static_cast<unsigned int>(topk)) % num_experts);
}

static inline float routingWeight(int src_rank, int token, int topk) {
    return static_cast<float>(src_rank * 4096 + token * 64 + topk + 1);
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) if (hostname[i] == '.') { hostname[i] = '\0'; return; }
}

// Per-(rank, token, hidden) sentinel value. Identity round-trip must return the
// exact same bf16 bits, so any function works; this one varies across all three
// indices to catch transposition / misrouting bugs.
static inline uint16_t sentinelBits(int rank, int iteration, int tok, int h) {
    // Keep successive test iterations far apart in BF16 so a stale epoch bank
    // cannot reproduce the next iteration's expected payload.
    float v = static_cast<float>(iteration * 4096 + rank * 131 + tok * 7 + h % 257);
    return float_to_bf16(v);
}

static void printUsage(const char* prog, int myRank) {
    if (myRank != 0) return;
    printf("Usage: %s [options]   (run under mpirun/srun)\n", prog);
    printf("  -A <num>   Number of attention (A-side) ranks (default: nRanks/2)\n");
    printf("  -e <num>   Total number of experts (default: N_f * 4; must be divisible by N_f)\n");
    printf("  -k <num>   top_k (default: 1)\n");
    printf("  -t <num>   Tokens per attention rank (default: 32)\n");
    printf("  -d <num>   Hidden dimension (default: 2048; LL supports 2048/2560/4096/5120/6144/7168/8192)\n");
    printf("  -i <num>   Iterations (default: 1)\n");
    printf("  -z         Exercise AUTO zero-copy with asymmetric A/F receive-window sizes\n");
    printf("  -h         Help\n");
}

int main(int argc, char* argv[]) {
    int myRank, nRanks, localRank = 0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    int num_attn_ranks = nRanks / 2;
    unsigned int num_experts = 0;     // 0 = auto
    unsigned int top_k = 1;
    unsigned int num_tokens = 32;
    unsigned int hidden = 2048;   // LL supports {2048,2560,4096,5120,6144,7168,8192}
    int iters = 1;
    bool zero_copy = false;

    int opt;
    while ((opt = getopt(argc, argv, "A:e:k:t:d:i:zh")) != -1) {
        switch (opt) {
            case 'A': num_attn_ranks = atoi(optarg); break;
            case 'e': num_experts = static_cast<unsigned int>(atoi(optarg)); break;
            case 'k': top_k = static_cast<unsigned int>(atoi(optarg)); break;
            case 't': num_tokens = static_cast<unsigned int>(atoi(optarg)); break;
            case 'd': hidden = static_cast<unsigned int>(atoi(optarg)); break;
            case 'i': iters = atoi(optarg); break;
            case 'z': zero_copy = true; break;
            case 'h': default: printUsage(argv[0], myRank); MPI_Finalize(); return (opt == 'h') ? 0 : 1;
        }
    }

    if (num_attn_ranks <= 0 || num_attn_ranks >= nRanks) {
        if (myRank == 0) printf("Error: num_attn_ranks (%d) must be in [1, nRanks=%d)\n", num_attn_ranks, nRanks);
        MPI_Finalize(); return 1;
    }
    const int N_f = nRanks - num_attn_ranks;
    if (num_experts == 0) num_experts = static_cast<unsigned int>(N_f) * 4u;
    if (num_experts % static_cast<unsigned int>(N_f) != 0) {
        if (myRank == 0) printf("Error: num_experts (%u) must be divisible by N_f (%d)\n", num_experts, N_f);
        MPI_Finalize(); return 1;
    }
    if (hidden % 128 != 0) {
        if (myRank == 0) printf("Error: hidden (%u) must be a multiple of 128\n", hidden);
        MPI_Finalize(); return 1;
    }
    if (top_k < 1 || top_k > num_experts) {
        if (myRank == 0) printf("Error: top_k (%u) must be in [1, num_experts=%u]\n", top_k, num_experts);
        MPI_Finalize(); return 1;
    }

    const unsigned int num_local_experts = num_experts / static_cast<unsigned int>(N_f);
    const bool is_attn = (myRank < num_attn_ranks);
    const unsigned int my_tokens = is_attn ? num_tokens : 0u;

    if (myRank == 0) {
        printf("AFD test: nRanks=%d  A-ranks=%d  F-ranks=%d  experts=%u (%u/F-rank)  top_k=%u  tokens/A=%u  hidden=%u  iters=%d  zero-copy=%s\n",
               nRanks, num_attn_ranks, N_f, num_experts, num_local_experts, top_k, num_tokens, hidden, iters,
               zero_copy ? "AUTO/asymmetric" : "staged");
    }

    // ---- localRank for GPU selection ----
    uint64_t* hostHashs = new uint64_t[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) { if (p == myRank) break; if (hostHashs[p] == hostHashs[myRank]) localRank++; }
    delete[] hostHashs;

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // ---- AFD communicator ----
    ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
    config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    config.num_experts = num_experts;
    config.max_dispatch_tokens_per_rank = num_tokens;
    config.max_recv_tokens_per_rank = NCCL_EP_AUTO;
    config.max_token_bytes = hidden * 2;                 // bf16
    config.rdma_buffer_size = NCCL_EP_AUTO;
    config.num_qp_per_rank = NCCL_EP_AUTO;
    config.num_channels = NCCL_EP_AUTO;
    config.num_attn_ranks = static_cast<unsigned int>(num_attn_ranks);
    config.alloc.alloc_fn = torchMalloc;
    config.alloc.free_fn  = torchFree;
    config.alloc.context  = nullptr;

    ncclEpGroup_t ep_group;
    NCCLCHECK(ncclEpCreateGroup(&ep_group, comm, &config));

    // ---- routing (topk_idx) ----
    ncclEpTensor_t* topk_idx = nullptr;
    NCCLCHECK(epMakeTensor(&topk_idx, 2, ncclInt64, my_tokens, top_k));
    if (is_attn) {
        std::vector<int64_t> h(static_cast<size_t>(my_tokens) * top_k);
        for (unsigned int t = 0; t < my_tokens; t++) {
            // top_k distinct global experts in [0, num_experts).
            for (unsigned int j = 0; j < top_k; j++)
                h[t * top_k + j] = routedExpert(myRank, static_cast<int>(t), static_cast<int>(j), num_experts);
        }
        CUDACHECK(cudaMemcpy(topk_idx->data, h.data(), h.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    }

    ncclEpHandle_t handle;
    NCCLCHECK(ncclEpCreateHandle(&handle, ep_group, NCCL_EP_LAYOUT_RANK_MAJOR, topk_idx,
                                 /*layout_info=*/nullptr, /*config=*/nullptr, s));
    CUDACHECK(cudaStreamSynchronize(s));

    int total_mismatches = 0;
    if (!is_attn) {
        // Both expert-side wrappers must reject a handle that claims source
        // tokens: those rows belong exclusively to attention ranks in AFD.
        ncclEpTensor_t* invalid_topk_idx = nullptr;
        NCCLCHECK(epMakeTensor(&invalid_topk_idx, 2, ncclInt64, 1u, top_k));
        ncclEpHandle_t invalid_handle;
        NCCLCHECK(ncclEpCreateHandle(&invalid_handle, ep_group, NCCL_EP_LAYOUT_RANK_MAJOR,
                                     invalid_topk_idx, /*layout_info=*/nullptr,
                                     /*config=*/nullptr, s));
        if (ncclEpAfRecv(invalid_handle, nullptr, nullptr, nullptr, nullptr, s) != ncclInvalidUsage)
            total_mismatches++;
        if (ncclEpAfReturn(invalid_handle, nullptr, nullptr, nullptr, s) != ncclInvalidUsage)
            total_mismatches++;
        NCCLCHECK(ncclEpHandleDestroy(invalid_handle));
        epFreeTensor(&invalid_topk_idx);
    }

    // ---- dispatch input tensors (A-side carries real data) ----
    ncclEpTensor_t* input_tokens = nullptr;     // [my_tokens, hidden] bf16
    ncclEpTensor_t* topk_weights_in = nullptr;  // [my_tokens, top_k] f32
    NCCLCHECK(epMakeTensor(&input_tokens, 2, ncclBfloat16, my_tokens, hidden));
    if (is_attn) NCCLCHECK(epMakeTensor(&topk_weights_in, 2, ncclFloat32, my_tokens, top_k));

    std::vector<uint16_t> sent_host(static_cast<size_t>(my_tokens) * hidden);
    if (is_attn) {
        std::vector<float> w(static_cast<size_t>(my_tokens) * top_k);
        for (unsigned int token = 0; token < my_tokens; ++token)
            for (unsigned int j = 0; j < top_k; ++j)
                w[token * top_k + j] = routingWeight(myRank, static_cast<int>(token), static_cast<int>(j));
        CUDACHECK(cudaMemcpy(topk_weights_in->data, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ---- dispatch output (rank-major recv) tensors — F-side only ----
    ncclEpTensor_t* recv_x = nullptr;            // [nRanks, max_tok, hidden] bf16
    ncclEpTensor_t* recv_topk_weights = nullptr; // [nRanks, max_tok, top_k] f32
    ncclEpTensor_t* recv_topk_idx = nullptr;     // [nRanks, max_tok, top_k] i32
    ncclEpTensor_t* src_rank_counter = nullptr;  // [nRanks] i32
    void* recv_x_window_data = nullptr;
    ncclWindow_t recv_x_window{};
    if (zero_copy) {
        NCCLCHECK(epMakeAsymmetricRecvWindow(
            &recv_x, &recv_x_window_data, &recv_x_window, is_attn,
            static_cast<unsigned int>(nRanks), num_tokens, hidden, comm));
    } else if (!is_attn) {
        NCCLCHECK(epMakeTensor(&recv_x, 3, ncclBfloat16, static_cast<unsigned int>(nRanks), num_tokens, hidden));
    }
    if (!is_attn) {
        NCCLCHECK(epMakeTensor(&recv_topk_weights, 3, ncclFloat32, static_cast<unsigned int>(nRanks), num_tokens, top_k));
        NCCLCHECK(epMakeTensor(&recv_topk_idx, 3, ncclInt32, static_cast<unsigned int>(nRanks), num_tokens, top_k));
        NCCLCHECK(epMakeTensor(&src_rank_counter, 1, ncclInt32, static_cast<unsigned int>(nRanks)));
    }

    // ---- combine tensors ----
    // F-side combine input is the (echoed) recv_x, rank-major [nRanks, max_tok, hidden].
    // A-ranks send no expert output and omit this tensor.
    ncclEpTensor_t* combine_in = nullptr;
    ncclEpTensor_t* combined_out = nullptr;      // A-side result [my_tokens, hidden]
    if (!is_attn)
        NCCLCHECK(epMakeTensor(&combine_in, 3, ncclBfloat16, static_cast<unsigned int>(nRanks), num_tokens, hidden));
    NCCLCHECK(epMakeTensor(&combined_out, 2, ncclBfloat16, my_tokens, hidden));

    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    layout_info.src_rank_counters = src_rank_counter;

    ncclEpDispatchInputs_t  din  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dout = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    din.tokens = input_tokens;
    din.topk_weights = topk_weights_in;
    dout.tokens = recv_x;
    dout.topk_weights = recv_topk_weights;
    dout.topk_idx = recv_topk_idx;

    ncclEpCombineInputs_t  cin  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t cout = NCCL_EP_COMBINE_OUTPUTS_INIT;
    cin.tokens = combine_in;
    cout.tokens = combined_out;

    // AFD's public contract is unquantized-only. Check the underlying APIs, not
    // just the role wrappers, and verify rejection happens before protocol state
    // advances so the normal round trip below remains valid.
    ncclEpDispatchConfig_t quantized_dispatch = NCCL_EP_DISPATCH_CONFIG_INIT;
    quantized_dispatch.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    if (ncclEpDispatch(handle, &din, &dout, &layout_info, &quantized_dispatch, s) != ncclInvalidUsage)
        total_mismatches++;

    ncclEpCombineConfig_t quantized_combine = NCCL_EP_COMBINE_CONFIG_INIT;
    quantized_combine.quant_recipe = NCCL_EP_COMB_QUANT_MXFP8;
    if (ncclEpCombine(handle, &cin, &cout, &quantized_combine, s) != ncclInvalidUsage)
        total_mismatches++;

    for (int it = 0; it < iters; it++) {
        // Change and re-upload the payload every iteration. The two-iteration
        // CI runs can therefore detect either side reading the previous epoch bank.
        if (is_attn) {
            for (unsigned int t = 0; t < my_tokens; t++)
                for (unsigned int hh = 0; hh < hidden; hh++)
                    sent_host[t * hidden + hh] =
                        sentinelBits(myRank, it, static_cast<int>(t), static_cast<int>(hh));
            CUDACHECK(cudaMemcpy(input_tokens->data, sent_host.data(),
                                 sent_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        }

        // ---- DISPATCH (collective): A sends, F receives ----
        if (is_attn) NCCLCHECK(ncclEpAfDispatch(handle, &din, &dout, &layout_info, nullptr, s));
        else         NCCLCHECK(ncclEpAfRecv(handle, &din, &dout, &layout_info, nullptr, s));
        CUDACHECK(cudaStreamSynchronize(s));

        // Validate the public F-side dispatch outputs. Slot order may vary with
        // GPU scheduling, so match each received payload to the deterministic
        // source token before checking its routing metadata.
        if (!is_attn) {
            const size_t recv_rows = static_cast<size_t>(nRanks) * num_tokens;
            std::vector<uint16_t> got_x(recv_rows * hidden);
            std::vector<float> got_weights(recv_rows * top_k);
            std::vector<int32_t> got_idx(recv_rows * top_k);
            std::vector<int32_t> got_counts(nRanks);
            const void* recv_x_data = zero_copy ? recv_x_window_data : recv_x->data;
            CUDACHECK(cudaMemcpy(got_x.data(), recv_x_data, got_x.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy(got_weights.data(), recv_topk_weights->data,
                                 got_weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy(got_idx.data(), recv_topk_idx->data,
                                 got_idx.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy(got_counts.data(), src_rank_counter->data,
                                 got_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

            const int local_f_rank = myRank - num_attn_ranks;
            int mism = 0;
            for (int src = 0; src < num_attn_ranks; ++src) {
                std::vector<unsigned char> expected(num_tokens, 0);
                int expected_count = 0;
                for (unsigned int token = 0; token < num_tokens; ++token) {
                    for (unsigned int j = 0; j < top_k; ++j) {
                        const int expert = routedExpert(src, static_cast<int>(token), static_cast<int>(j), num_experts);
                        if (expert / static_cast<int>(num_local_experts) == local_f_rank) {
                            expected[token] = 1;
                            expected_count++;
                            break;
                        }
                    }
                }
                if (got_counts[src] != expected_count) mism++;
                std::vector<unsigned char> matched(num_tokens, 0);
                const int received_count = got_counts[src] < 0 ? 0 :
                    (got_counts[src] > static_cast<int>(num_tokens) ? static_cast<int>(num_tokens) : got_counts[src]);
                for (int slot = 0; slot < received_count; ++slot) {
                    const size_t row = static_cast<size_t>(src) * num_tokens + slot;
                    int token_match = -1;
                    for (unsigned int token = 0; token < num_tokens && token_match < 0; ++token) {
                        if (!expected[token] || matched[token]) continue;
                        bool payload_matches = true;
                        for (unsigned int h = 0; h < hidden; ++h) {
                            if (got_x[row * hidden + h] !=
                                sentinelBits(src, it, static_cast<int>(token), static_cast<int>(h))) {
                                payload_matches = false;
                                break;
                            }
                        }
                        if (payload_matches) token_match = static_cast<int>(token);
                    }
                    if (token_match < 0) {
                        mism++;
                        continue;
                    }
                    matched[token_match] = 1;
                    for (unsigned int j = 0; j < top_k; ++j) {
                        const int expert = routedExpert(src, token_match, static_cast<int>(j), num_experts);
                        const int expected_idx =
                            expert / static_cast<int>(num_local_experts) == local_f_rank
                                ? expert % static_cast<int>(num_local_experts) : -1;
                        if (got_idx[row * top_k + j] != expected_idx) mism++;
                        if (got_weights[row * top_k + j] != routingWeight(src, token_match, static_cast<int>(j)))
                            mism++;
                    }
                }
                for (unsigned int token = 0; token < num_tokens; ++token)
                    if (expected[token] != matched[token]) mism++;
            }
            if (mism) printf("Rank %d iter %d: F-side dispatch validation found %d mismatch(es)\n", myRank, it, mism);
            total_mismatches += mism;
        }

        // ---- F-side identity "expert": echo recv_x into combine_in ----
        if (!is_attn) {
            const void* recv_x_data = zero_copy ? recv_x_window_data : recv_x->data;
            CUDACHECK(cudaMemcpy(combine_in->data, recv_x_data,
                                 static_cast<size_t>(nRanks) * num_tokens * hidden * sizeof(uint16_t),
                                 cudaMemcpyDeviceToDevice));
        }

        // ---- COMBINE (collective): F returns, A receives ----
        if (is_attn) NCCLCHECK(ncclEpAfCombine(handle, &cin, &cout, nullptr, s));
        else         NCCLCHECK(ncclEpAfReturn(handle, &cin, &cout, nullptr, s));
        CUDACHECK(cudaStreamSynchronize(s));

        // ---- A-side validation ----
        // Rank-major combine adds one caller-preweighted result per distinct
        // destination rank. This identity expert does not preweight, so the
        // expected value is the input multiplied by that rank count.
        if (is_attn) {
            std::vector<uint16_t> got(static_cast<size_t>(my_tokens) * hidden);
            CUDACHECK(cudaMemcpy(got.data(), combined_out->data, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            int mism = 0;
            for (unsigned int token = 0; token < my_tokens; ++token) {
                std::vector<unsigned char> seen_f_rank(N_f, 0);
                int distinct_f_ranks = 0;
                for (unsigned int j = 0; j < top_k; ++j) {
                    const int expert = routedExpert(myRank, static_cast<int>(token), static_cast<int>(j), num_experts);
                    const int f_rank = expert / static_cast<int>(num_local_experts);
                    if (!seen_f_rank[f_rank]) {
                        seen_f_rank[f_rank] = 1;
                        distinct_f_ranks++;
                    }
                }
                for (unsigned int h = 0; h < hidden; ++h) {
                    const size_t i = static_cast<size_t>(token) * hidden + h;
                    const uint16_t expected = float_to_bf16(bf16_to_float(sent_host[i]) * distinct_f_ranks);
                    if (got[i] != expected) mism++;
                }
            }
            if (mism) {
                printf("Rank %d iter %d: MISMATCH %d / %zu elements\n", myRank, it, mism, got.size());
            }
            total_mismatches += mism;
        }
    }

    // ---- reduce pass/fail across ranks ----
    int global_mismatches = 0;
    MPICHECK(MPI_Allreduce(&total_mismatches, &global_mismatches, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    int rc = 0;
    if (myRank == 0) {
        if (global_mismatches == 0) {
            printf("AFD test: PASSED (dispatch metadata and round-trip exact across "
                   "%d A-rank(s), top_k=%u, %d iter(s)).\n",
                   num_attn_ranks, top_k, iters);
        } else {
            printf("AFD test: FAILED (%d total mismatches).\n", global_mismatches);
            rc = 1;
        }
    }
    MPICHECK(MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // ---- cleanup ----
    epFreeTensor(&topk_idx);
    epFreeTensor(&input_tokens);
    epFreeTensor(&topk_weights_in);
    if (zero_copy)
        NCCLCHECK(epFreeAsymmetricRecvWindow(&recv_x, &recv_x_window_data, &recv_x_window, comm));
    else
        epFreeTensor(&recv_x);
    epFreeTensor(&recv_topk_weights);
    epFreeTensor(&recv_topk_idx);
    epFreeTensor(&src_rank_counter);
    epFreeTensor(&combine_in);
    epFreeTensor(&combined_out);
    NCCLCHECK(ncclEpHandleDestroy(handle));
    NCCLCHECK(ncclEpGroupDestroy(ep_group));
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaStreamDestroy(s));
    MPICHECK(MPI_Finalize());
    return rc;
}
