/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// ============================================================================
// Configuration constants
// ============================================================================

#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128
#define MAX_HIDDEN_DIM 16384
#define MAX_NUM_TOPK 32
// Sentinel for empty/padding slots in the uint16 topk routing map (pull dispatch).
static constexpr uint16_t kTopkIdxInvalid = 0xFFFFu;
static constexpr int kPullMaxLsaRanks = 128;
#define MAX_NCCL_GIN_CTX_PER_COMM 4
#define NUM_WAIT_NANOSECONDS 500
#define MAX_SUPPORTED_TOKENS_PER_RANK 8192  // Must match kernel template in ht_ep_adapter.cu
#define FINISHED_SUM_TAG (MAX_SUPPORTED_TOKENS_PER_RANK * 2)
// Two dedicated grid-wide election counters for combine_kernel_impl_1sided_rm_lsa (see ctaSync's
// doc comment in ll/ll_combine_lsa.cuh) -- never shared, since they're not separated by a
// grid-wide barrier.
#define NCCL_EP_COMBINE_SYNC_COUNT 2

// Timeout for GPU-side wait loops. When exceeded, the peer is masked (if active-mask
// is enabled) or the kernel traps. Setting this too low risks false positives: a rank
// that is merely slow may be marked as failed. Asymmetric timeouts across ranks can
// produce inconsistent masks (rank A masks rank B, but rank B does not mask rank A).
// Mask consistency is a framework-level concern -- the application should query the
// mask after detecting an error and reconcile as needed (e.g., via EPLB rebalance).
#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// NCCL GIN Configuration

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__ // NOLINT(*-reserved-identifier)
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

// ============================================================================
// Standard includes
// ============================================================================
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <cuda_bf16.h>
#if defined(__has_include)
#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#define NCCL_EP_HAS_CUDA_FP4_TYPES 1
#endif
#endif
#ifndef NCCL_EP_HAS_CUDA_FP4_TYPES
#define NCCL_EP_HAS_CUDA_FP4_TYPES 0
#endif
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "device/macros.cuh"
#include "ep_enums.h"
#include "cuda_fp_features.h"

// Library-side NCCLCHECK: for ncclResult_t-returning functions. The
// app/exit-on-error flavor used by ep_bench/ep_test is defined locally in
// those TUs (they exit rather than return).
#ifndef NCCLCHECK
#define NCCLCHECK(cmd) \
    do { \
        ncclResult_t _ncclcheck_res = (cmd); \
        if (_ncclcheck_res != ncclSuccess) { \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(_ncclcheck_res)); \
            return _ncclcheck_res; \
        } \
    } while (0)
#endif

namespace nccl_ep {

inline constexpr std::string_view host_device_fp4_target_arch(unsigned int sm) {
    switch (sm) {
    case 100:
    case 103: return "sm_100f";
    case 107: return "sm_107f";
    case 110: return "sm_110f";
    case 120:
    case 121: return "sm_120f";
    default: return {};
    }
}

inline constexpr bool host_device_supports_fp4(unsigned int sm) {
    return !host_device_fp4_target_arch(sm).empty();
}

// cuda_fp4.h first ships in CUDA 12.8, but NVFP4 family targets require CUDA 12.9.
inline constexpr bool host_build_supports_fp4() {
    return NCCL_EP_HAS_CUDA_FP4_TYPES && CUDART_VERSION >= 12090;
}

// cuda_fp8.h ships E4M3 from CUDA 11.8; __nv_fp8_e8m0 (and therefore the
// preferred MXFP8 reciprocal-scale path) requires CUDA 12.8. Device headers
// still compile on older toolkits via a software fallback; the API rejects
// MXFP8 there so the fallback is compile-only.
inline constexpr bool host_build_supports_mxfp8() {
    return NCCL_EP_HAS_CUDA_E8M0_TYPE;
}

constexpr int kDsFp8E3M4ElementsPerScale = 128;

// low-latency kernels
namespace ll {

// LL dispatch/send maps one 32-thread warp to each forwarding top-k entry and
// reserves one control warp. For send/recv phase synchronization, its warps
// split into groups of at least two; named barriers currently limit this to 14.
constexpr int kLlDispatchControlWarps = 1;
constexpr int kLlDispatchMaxWarpGroups = 14;

// Helper function for alignment (host/device compatible)
template <typename dtype_t>
__host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
    return ((a + b - 1) / b) * b;
}

// Expert / rank index arithmetic of the LL dispatch protocol, kept next to the
// header layout it addresses so other device headers and host code can reuse it.
__host__ __device__ __forceinline__ int getLocalExpertIdx(int expertIdx, int numLocalExperts) {
    return (expertIdx >= 0) ? expertIdx % numLocalExperts : -1;
}

// Sender's slot relative to the receiver around the rank ring, in [0, numRanks).
// Relative indexing distributes the NVLink staging addresses across receivers.
__host__ __device__ __forceinline__ int relativeRankSlot(int senderRank, int receiverRank, int numRanks) {
    return (senderRank >= receiverRank) ? (senderRank - receiverRank)
                                         : (senderRank - receiverRank + numRanks);
}

__host__ __device__ __forceinline__ int getExpertRankIdx(int expertIdx, int numLocalExperts) {
    return (expertIdx >= 0) ? expertIdx / numLocalExperts : -1;
}

// Per-hop routing entry in the dispatch message header.
// Rank-major carries the topk weight on the wire so the receiver can write
// outRecvTopkWeights without an extra round-trip. Expert-major omits it,
// keeping the per-entry overhead at 2 bytes regardless of num_topk.
template <ncclEpLayout_t kLayout>
struct DispatchRouter;

template <>
struct DispatchRouter<NCCL_EP_LAYOUT_RANK_MAJOR> {
    float topk_weight;  // written to outRecvTopkWeights on the receive side
    uint16_t expert_id;
    // sizeof = 8: float(4) + uint16_t(2) + 2 bytes implicit pad
};

template <>
struct DispatchRouter<NCCL_EP_LAYOUT_EXPERT_MAJOR> {
    uint16_t expert_id;
    // sizeof = 2
};

static_assert(sizeof(DispatchRouter<NCCL_EP_LAYOUT_RANK_MAJOR>) == 8, "unexpected rank-major router size");
static_assert(sizeof(DispatchRouter<NCCL_EP_LAYOUT_EXPERT_MAJOR>) == 2, "unexpected expert-major router size");

// Dispatch message header: token identity + per-topk routing entries.
// alignas(16): header objects are 16-byte aligned in the RDMA buffer.
template <ncclEpLayout_t kLayout>
struct alignas(16) DispatchHdr {
    int token_id;
    DispatchRouter<kLayout> rtr[];  // Flexible array member (Note: this is not a C++ standard, but a CUDA extension)
};

static_assert(offsetof(DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>, rtr) == 4, "unexpected rank-major rtr offset");
static_assert(offsetof(DispatchHdr<NCCL_EP_LAYOUT_EXPERT_MAJOR>, rtr) == 4, "unexpected expert-major rtr offset");

// Dispatch header wire size: token_id prefix through last rtr entry, rounded up
// to int4 (16-byte) boundary for vectorized RDMA access.
template <ncclEpLayout_t kLayout>
__host__ __device__ __forceinline__ constexpr size_t get_dispatch_hdr_sz(int num_topk) {
    const size_t base_sz =
        offsetof(DispatchHdr<kLayout>, rtr) + static_cast<size_t>(num_topk) * sizeof(DispatchRouter<kLayout>);
    return align<size_t>(base_sz, sizeof(int4));
}

// Runtime overload: selects the correct template specialisation based on layout.
__host__ __forceinline__ size_t get_dispatch_hdr_sz(int num_topk, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_RANK_MAJOR ? get_dispatch_hdr_sz<NCCL_EP_LAYOUT_RANK_MAJOR>(num_topk) :
                                                 get_dispatch_hdr_sz<NCCL_EP_LAYOUT_EXPERT_MAJOR>(num_topk);
}
} // namespace ll

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

// Packed bitmap stored as little-endian 64-bit words: word count, word index, single-bit test/set.
__host__ __device__ constexpr int bit_words(int n) { return ceil_div(n, 64); }
__host__ __device__ constexpr int bit_word(int i) { return i >> 6; }
template <typename WordT>
__host__ __device__ __forceinline__ bool test_bit(const WordT* words, int i) {
    return (words[i >> 6] >> (i & 63)) & WordT{1};
}
template <typename WordT>
__host__ __device__ __forceinline__ void set_bit(WordT* words, int i) {
    words[i >> 6] |= (WordT{1} << (i & 63));
}
// OR-reduces bits [lo, hi) of a packed bitmap; lo/hi need not be word-aligned or word-sized
// (partial words at either end are masked off before the OR). The first word keeps bits
// [lo%64, 64), interior words are taken whole, and the last word keeps bits [0, hi%64).
template <typename WordT>
__host__ __device__ __forceinline__ bool bit_range_any(const WordT* words, int lo, int hi) {
    if (lo == hi) return false;
    const WordT whole_word_mask = ~WordT{0};
    const WordT first_word_mask = ~((WordT{1} << (lo & 63)) - WordT{1});
    const WordT last_word_mask = ((hi & 63) == 0) ? whole_word_mask : ((WordT{1} << (hi & 63)) - WordT{1});
    WordT acc = 0;
    WordT cur_mask = first_word_mask;
    int wi = bit_word(lo);
    for (; wi < bit_words(hi) - 1; wi++) {
        acc |= words[wi] & cur_mask;
        cur_mask = whole_word_mask;
    }
    cur_mask &= last_word_mask;
    return (acc | (words[wi] & cur_mask)) != 0;
}

// Buffer-relative offsets into rdma_buffer. Pointers are resolved at use time
// against the group's current rdma_buffer base, so the buffer can be
// reallocated (e.g., grown for a larger layout) without invalidating any
// cached layout state on live handles.
struct LowLatencyBuffer {
    int num_clean_int = 0;

    size_t dispatch_rdma_send_buffer_offset = 0;
    size_t dispatch_rdma_recv_data_buffer_offset = 0;
    size_t dispatch_rdma_recv_count_buffer_offset = 0;

    size_t combine_rdma_send_buffer_offset = 0;
    size_t combine_rdma_recv_data_buffer_offset = 0;
    size_t combine_rdma_recv_flag_buffer_offset = 0;

    size_t combine_rdma_send_buffer_data_start_offset = 0;
    size_t num_bytes_per_combine_msg = 0;

    // "Every peer confirms it's done reading my inData", for combine_kernel_impl_1sided_rm_lsa
    // (ll/ll_combine_lsa.cuh). Allocated unconditionally (numRanks ints), packed immediately after
    // the signal region within the same per-bank span -- see LowLatencyLayout::signal_slot_stride's
    // doc comment for why, and lsa_signaling_clean_int_count below for its clear size.
    size_t combine_rdma_read_done_buffer_offset = 0;

    // Per-peer "total tokens dispatched this round" publish, for NCCL_EP_COMBINE_LSA's auto-mode
    // want-1-sided decision (see ll::computeAndPublishWant1SidedLsa). Same rationale as above, packed right
    // after combine_rdma_read_done_buffer_offset -- but excluded from
    // lsa_signaling_clean_int_count, since it's plain-overwritten every round and never needs
    // clearing.
    size_t dispatch_rdma_total_token_buffer_offset = 0;

    // (count-buffer offset, count) for clean_low_latency_buffer. Dispatch
    // recv-count and combine recv-flag share the same signaling slot.
    std::pair<size_t, int> clean_meta_offset() const {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer_offset == combine_rdma_recv_flag_buffer_offset);
        return {dispatch_rdma_recv_count_buffer_offset, num_clean_int};
    }

    // Ints to clear in one pass to wipe a bank's entire signal + read-done span (never
    // total-token, see its offset field's doc comment above) -- dispatch and combine both pre-wipe
    // their OWN next bank's signal region today; this lets either one wipe read-done in the same
    // pass, which is required since dispatch never otherwise touches read-done at all (see
    // LowLatencyLayout::signal_slot_stride's doc comment for why one shared bank is used by both).
    int lsa_signaling_clean_int_count = 0;
};

struct LowLatencyEpochState {
    unsigned int epoch = 0;
    unsigned int pending_epoch = 0;
    unsigned int send_in_flight = 0;
};

struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffer;
    size_t payload_slot_stride = 0;
    // Shared bank-to-bank stride for every signaling buffer kind (recv-count/recv-flag,
    // read-done, total-token -- see their offset fields' doc comments in LowLatencyBuffer): they're
    // packed bank-major, one combined span per bank ([signal][read_done][total_token], then the
    // same three again for bank 1), not kind-major with independent strides. That's what lets
    // dispatch and combine each clear their next bank's signal+read-done region in a single pass
    // (lsa_signaling_clean_int_count) regardless of which op actually reads each sub-buffer --
    // dispatch never touches read-done itself, but its own next-bank wipe is what prepares it for
    // combine, exactly as it already does for the shared signal region (every launch wipes the
    // bank the *next* launch -- of either op -- will land on, not its own future occurrence).
    size_t signal_slot_stride = 0;

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    static out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    LowLatencyLayout(
        int num_max_dispatch_tokens_per_rank,
        size_t max_token_bytes,
        int num_ranks,
        int num_experts,
        int num_topk,
        ncclEpLayout_t layout,
        int num_attn_ranks = 0) {
        // Two physical payload and signaling banks are addressed from one base
        // layout by the device-selected epoch parity.

        // Per-slot sizes for buffer allocation are datatype-agnostic at the API;
        // max_token_bytes upper-bounds the entire dispatch payload. Recipe
        // validation ensures the selected token and scale payload fits that bound.
        // Combine reserves its existing per-128-token-element metadata space.
        const size_t num_scales = max_token_bytes / sizeof(nv_bfloat16) / 128;
        size_t scale_metadata_bytes = num_scales * sizeof(nv_bfloat162);

        size_t disp_hdr_sz = ll::get_dispatch_hdr_sz(num_topk, layout);
        size_t num_bytes_per_dispatch_msg = disp_hdr_sz + max_token_bytes;
        size_t num_bytes_per_combine_msg =
            align<size_t>(scale_metadata_bytes + max_token_bytes, sizeof(int4));

        // Per-op send and recv sizes.
        // Combine send buffer holds one slot per outbound message produced on
        // the sender side: expert-major fans tokens out per expert (num_experts
        // slots), while rank-major fans them out per destination rank
        // (num_ranks slots). Sizing per layout avoids over-allocating in the
        // rank-major case (typically num_ranks < num_experts).
        size_t combine_send_slots =
            (layout == NCCL_EP_LAYOUT_RANK_MAJOR) ? static_cast<size_t>(num_ranks) : static_cast<size_t>(num_experts);
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_send_buffer_bytes =
            combine_send_slots * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;

        // Symmetric receive buffers: the RDMA wire is always rank-major regardless of user-facing layout.
        size_t dispatch_recv_data_buffer_bytes = static_cast<size_t>(num_ranks) *
                                                 static_cast<size_t>(num_max_dispatch_tokens_per_rank) *
                                                 num_bytes_per_dispatch_msg;
        size_t combine_recv_buffer_bytes = static_cast<size_t>(num_max_dispatch_tokens_per_rank) *
                                           num_bytes_per_combine_msg * static_cast<size_t>(num_topk);

        // All four buffers feed vectorized int4 loads/stores; their starts (and
        // therefore their sizes, since they share slots) must be 16-byte aligned.
        EP_HOST_ASSERT(dispatch_send_buffer_bytes % sizeof(int4) == 0);
        EP_HOST_ASSERT(dispatch_recv_data_buffer_bytes % sizeof(int4) == 0);
        EP_HOST_ASSERT(combine_send_buffer_bytes % sizeof(int4) == 0);
        EP_HOST_ASSERT(combine_recv_buffer_bytes % sizeof(int4) == 0);

        // Dispatch and combine never run concurrently and don't carry state in
        // their recv buffers across the boundary (each kernel copies received
        // payloads out to user tensors before exit). So one shared slot can
        // hold either op's (send + recv) pair, sized to the worst case.
        // Inside a slot:
        //   - send_buffer always starts at slot_base.
        //   - dispatch_recv starts at slot_base + dispatch_send_buffer_bytes.
        //   - combine_recv  starts at slot_base + combine_send_buffer_bytes.
        // Two slots for double-buffering.
        size_t dispatch_combo_bytes = dispatch_send_buffer_bytes + dispatch_recv_data_buffer_bytes;
        size_t combine_combo_bytes = combine_send_buffer_bytes + combine_recv_buffer_bytes;
        size_t slot_bytes = std::max(dispatch_combo_bytes, combine_combo_bytes);
        payload_slot_stride = slot_bytes;
        total_bytes += payload_slot_stride * 2;

        // Dispatch addresses each epoch bank as [localExpert][worldSrcRank],
        // which requires L * R counters. Combine needs one flag per expert (E).
        // Both operations reuse the same bank, so size it for max(L * R, E).
        // In classic EP every rank owns experts, hence L * R == E; AFD has only
        // F expert ranks, so L = E / F and the dispatch bank grows to L * R.
        const int num_expert_ranks = num_ranks - num_attn_ranks;
        // Public group validation establishes this before any layout is built.
        assert(num_expert_ranks > 0 && num_experts % num_expert_ranks == 0);
        const size_t num_dispatch_signals =
            static_cast<size_t>(num_experts / num_expert_ranks) * num_ranks;
        size_t dispatch_recv_count_buffer_bytes = num_dispatch_signals * sizeof(int);
        size_t combine_recv_flag_buffer_bytes = static_cast<size_t>(num_experts) * sizeof(int);
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);

        // Read-done (combine_kernel_impl_1sided_rm_lsa) and total-token (NCCL_EP_COMBINE_LSA auto
        // mode) buffers -- see their offset fields' doc comments above. Packed immediately after
        // the signal region within the same per-bank span (signal_slot_stride's doc comment
        // explains why), both sized by num_ranks and fixed for the group's lifetime (ranks never
        // change once a group is created), so sharing one stride introduces no later-growth
        // coupling. total_token goes last, outside lsa_signaling_clean_int_count's range.
        size_t read_done_buffer_bytes = static_cast<size_t>(num_ranks) * sizeof(int);
        size_t total_token_buffer_bytes = static_cast<size_t>(num_ranks) * sizeof(int);
        size_t signaling_bank_bytes = signaling_buffer_bytes + read_done_buffer_bytes + total_token_buffer_bytes;
        signal_slot_stride = align<size_t>(signaling_bank_bytes, 128);
        total_bytes += signal_slot_stride * 2;

        // Bank-0 offsets within the shared signaling span; the device derives bank 1 by adding
        // signal_slot_stride to each.
        const size_t read_done_base = signaling_buffer_bytes;
        const size_t total_token_base = read_done_base + read_done_buffer_bytes;
        const size_t slot_base = signal_slot_stride * 2;
        buffer = LowLatencyBuffer{
            .num_clean_int = static_cast<int>(signaling_buffer_bytes / sizeof(int)),
            .dispatch_rdma_send_buffer_offset = slot_base,
            .dispatch_rdma_recv_data_buffer_offset = slot_base + dispatch_send_buffer_bytes,
            .dispatch_rdma_recv_count_buffer_offset = 0,
            .combine_rdma_send_buffer_offset = slot_base,
            .combine_rdma_recv_data_buffer_offset = slot_base + combine_send_buffer_bytes,
            .combine_rdma_recv_flag_buffer_offset = 0,
            .combine_rdma_send_buffer_data_start_offset = slot_base,
            .num_bytes_per_combine_msg = num_bytes_per_combine_msg,
            .combine_rdma_read_done_buffer_offset = read_done_base,
            .dispatch_rdma_total_token_buffer_offset = total_token_base,
            .lsa_signaling_clean_int_count =
                static_cast<int>((signaling_buffer_bytes + read_done_buffer_bytes) / sizeof(int)),
        };
    }
};

// Size for a specific LL layout, rounded up to the buffer alignment.
inline unsigned long int get_low_latency_rdma_size_hint_for_layout(
    int num_max_dispatch_tokens_per_rank,
    size_t max_token_bytes,
    int num_ranks,
    int num_experts,
    ncclEpLayout_t layout,
    int num_attn_ranks = 0) {
    auto num_bytes = LowLatencyLayout(
                         num_max_dispatch_tokens_per_rank,
                         max_token_bytes,
                         num_ranks,
                         num_experts,
                         MAX_NUM_TOPK,
                         layout,
                         num_attn_ranks)
                         .total_bytes;
    return ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

// Worst case across all LL layouts (upper bound on what may be needed once the
// per-handle layout is known). Used as the upper limit for the configured
// rdma_buffer_size; the actual allocation may start smaller and grow.
inline unsigned long int get_low_latency_rdma_size_hint(
    int num_max_dispatch_tokens_per_rank,
    size_t max_token_bytes,
    int num_ranks,
    int num_experts,
    int num_attn_ranks = 0) {
    auto bytes_for = [&](ncclEpLayout_t layout) {
        return get_low_latency_rdma_size_hint_for_layout(
            num_max_dispatch_tokens_per_rank,
            max_token_bytes,
            num_ranks,
            num_experts,
            layout,
            num_attn_ranks);
    };
    return std::max(bytes_for(NCCL_EP_LAYOUT_EXPERT_MAJOR), bytes_for(NCCL_EP_LAYOUT_RANK_MAJOR));
}

} // namespace nccl_ep
