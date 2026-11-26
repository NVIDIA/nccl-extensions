/*
 * SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_HT_LOCAL_DISPATCH_PACK_CUH_
#define NCCL_EP_HT_LOCAL_DISPATCH_PACK_CUH_

// ============================================================================
// Unordered HT dispatch pack prologue.
//
// Packs each selected token's [token][prob][sf] entry into gin_send_staging, so N2N
// ships a channel to a destination as whole batches instead of up to three puts
// per token. Runs before the dispatch kernel on the same stream.
//
//   e.g. 2 nodes, 2 channels of 3 tokens; t1, t2, t3 and t5 routed to the remote node
//   (for simplicity only; in practice there are many more channels and tokens per channel)
//
//   input, also used as is for the local node:
//
//     attn_input_token  [     t0     ][     t1     ][     t2     ][     t3     ][     t4     ][     t5     ]
//     attn_input_prob   [     p0     ][     p1     ][     p2     ][     p3     ][     p4     ][     p5     ]
//     attn_input_sf     [     s0     ][     s1     ][     s2     ][     s3     ][     s4     ][     s5     ]
//                       |--------------- channel 0 ---------------|-------------- channel 1 ---------------|
//
//   remote node send staging, one (token, prob, sf) entry per routed token:
//
//     slot                    0             1             2             3             4             5
//                       [(t1, p1, s1)][(t2, p2, s2)][   unused   ][(t3, p3, s3)][(t5, p5, s5)][   unused   ]
//                       |--------------- channel 0 ---------------|-------------- channel 1 ---------------|
//
// The local node uses the original three arrays: its tokens never cross the network and
// already sit on the right node, so they only need the fanup (scatter to the NVLink peers
// within the node). Remote tokens first need the fanout (scatter from node to node), which
// this packing serves, and then the fanup.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda/ptx>

#include "device/cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x
#include "device/device_primitives.cuh"
#include "device/ht_ep_configs.cuh"
#include "ep_dispatch_channel.cuh"
#include "device/ht/warp_group.cuh"

namespace nccl_ep {
namespace ht_dense {

// Byte offset of one (destination, token slot) entry from the region base.
__host__ __device__ __forceinline__ size_t entry_offset(
    size_t region_base, int remote_idx, int token_idx, int max_tokens_per_dest, size_t bytes_per_entry) {
    return region_base +
           (static_cast<size_t>(remote_idx) * static_cast<size_t>(max_tokens_per_dest) +
            static_cast<size_t>(token_idx)) *
               bytes_per_entry;
}

// Cooperative 16B copy. Token, prob and sf rows are 16B multiples and 16B aligned, as dispatch's TMA requires.
__device__ __forceinline__ void copy_bytes(
    uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, size_t bytes, int lane, int lanes) {
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    for (size_t i = lane; i < bytes / sizeof(int4); i += lanes) dst4[i] = src4[i];
}

struct pack_args_t {
    uint8_t* send_staging;
    const uint8_t* attn_input_token;
    const uint8_t* attn_input_prob;
    const uint8_t* attn_input_sf;
    const bool* attn_to_rdma_map;
    const uint64_t* guard_flags;
    const uint64_t* expected_gin_flag_val;
    size_t bytes_per_entry;
    size_t token_bytes;
    size_t prob_bytes;
    bool compact_prob;
    size_t sf_bytes;
    int max_tokens_per_dest;
    int num_of_tokens_per_rank;
    int max_tokens_per_rank;
    int num_channels;
    int lsa_teams;
    int my_lteam;
};

using pack_group = ::ht_gin_unordered::warp_group<8, 0>;

// Packs this rank's remote-bound tokens into the RDMA-registered send staging. It runs ahead of the
// dispatch kernel on the same stream, so its time adds directly to dispatch; for simplicity it uses
// the preprocessing SM count (all SMs by default) to finish as fast as possible.
//   1. Stage: thread 0 TMA-loads the token row and the SF row (if any) from HBM into SMEM.
//   2. Fan out: one warp per destination writes that destination's packed entry with direct copies:
//      the token and SF rows from SMEM, and the prob row (different per destination) from HBM.
// Being HBM-bandwidth bound, it gets faster with a faster HBM.
// To cut the SM footprint, the SMEM -> HBM part could use TMA too (end to end), with a deeper SMEM
// ring so fewer SMs keep enough copies in flight. That is more complex, so it is kept simple for now.
template <bool FORWARD_DISPATCH, bool HAS_SF, bool GUARD_ENABLED>
__device__ __forceinline__ void local_dispatch_pack(const pack_args_t& a) {
    const int num_remote_lsa = a.lsa_teams - 1;
    const int tid = pack_group::thread_rank();
    const int nthreads = pack_group::size();
    const int lane = tid % 32;
    const int warp = pack_group::warp_rank();
    const int warps = pack_group::warp_size();

    // Local packing overwrites dispatch send staging. The dispatch-entry RDMA guard waits
    // for all rail peers to receive the previous round's data before reusing it, protecting
    // staging from overwrite while in-flight RDMA on the data GIN contexts still reads it.
    if constexpr (GUARD_ENABLED) {
        if (warp == 0) {
            const uint64_t expected = *a.expected_gin_flag_val;
            for (int peer = lane; peer < a.lsa_teams; peer += 32) {
                if (peer == a.my_lteam) continue;
                while (nccl_ep::ld_relaxed_sys_global(&a.guard_flags[peer]) + 1ull < expected) {}
            }
        }
        __syncthreads();
    }

    // SMEM: one mbarrier, then the token and scale staging the whole block fans out from.
    extern __shared__ uint8_t smem[];
    uint64_t* const mbar = reinterpret_cast<uint64_t*>(smem);
    uint8_t* const tok_s = smem + NCCL_EP_HT_TMA_ALIGN;
    uint8_t* const sf_s = tok_s + a.token_bytes;
    const uint32_t stage_bytes = static_cast<uint32_t>(a.token_bytes + (HAS_SF ? a.sf_bytes : 0));

    if (tid == 0) {
        cuda::ptx::mbarrier_init(mbar, 1);
        // Publish the init to the async proxy the bulk copy completes through.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    // Each channel's N2N warp sends one channel's range.
    const int fix_tokens_per_channel = dispatch_tokens_per_channel(a.max_tokens_per_rank, a.num_channels);
    // Initialised once; the phase carries across tokens.
    int parity = 0;
    for (int token_idx = blockIdx.x; token_idx < a.num_of_tokens_per_rank; token_idx += gridDim.x) {
        bool any = false;
        for (int remote_idx = tid; remote_idx < num_remote_lsa; remote_idx += nthreads) {
            if (a.attn_to_rdma_map[token_idx * num_remote_lsa + remote_idx]) any = true;
        }
        if (__syncthreads_or(any ? 1 : 0) == 0) continue;

        if (tid == 0) {
            // Order the other warps' generic reads of the previous row before the async-proxy overwrite.
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                                     tok_s, a.attn_input_token + token_idx * a.token_bytes,
                                     static_cast<uint32_t>(a.token_bytes), mbar);
            if constexpr (HAS_SF) {
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                                         sf_s, a.attn_input_sf + token_idx * a.sf_bytes,
                                         static_cast<uint32_t>(a.sf_bytes), mbar);
            }
            // No prob stage: the dense prob row differs per peer node, so each destination copies its own below.
            cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta,
                                                 cuda::ptx::space_shared, mbar, stage_bytes);
            while (!cuda::ptx::mbarrier_try_wait_parity(mbar, parity)) {}
        }
        // Only thread 0 waited; the barrier hands the staged row to the other warps.
        __syncthreads();
        parity ^= 1;

        // One warp per destination; slot = channel base + earlier tokens of the channel routed there.
        const int channel_first_token_idx = (token_idx / fix_tokens_per_channel) * fix_tokens_per_channel;
        for (int remote_idx = warp; remote_idx < num_remote_lsa; remote_idx += warps) {
            if (!a.attn_to_rdma_map[token_idx * num_remote_lsa + remote_idx]) continue;

            const int cur_idx_in_channel = token_idx - channel_first_token_idx;
            int packed_idx = 0;
            for (int i = 0; i * 32 < cur_idx_in_channel; i++) {
                int token_idx_in_channel = i * 32 + lane;
                bool need_write =
                    token_idx_in_channel < cur_idx_in_channel &&
                    a.attn_to_rdma_map[((token_idx_in_channel + channel_first_token_idx) * num_remote_lsa) +
                                       remote_idx];
                packed_idx += __popc(__ballot_sync(~0u, need_write));
            }

            const int remote_lteam_id = remote_idx < a.my_lteam ? remote_idx : remote_idx + 1;
            uint8_t* const entry =
                a.send_staging + entry_offset(0, remote_idx, channel_first_token_idx + packed_idx,
                                              a.max_tokens_per_dest, a.bytes_per_entry);
            copy_bytes(entry, tok_s, a.token_bytes, lane, 32);
            // Prob is per destination (unless compact), so it is not staged.
            if constexpr (FORWARD_DISPATCH) {
                copy_bytes(entry + a.token_bytes,
                           a.attn_input_prob +
                               (a.compact_prob ? token_idx : token_idx * a.lsa_teams + remote_lteam_id) * a.prob_bytes,
                           a.prob_bytes, lane, 32);
            }
            if constexpr (HAS_SF) {
                copy_bytes(entry + a.token_bytes + (FORWARD_DISPATCH ? a.prob_bytes : 0), sf_s, a.sf_bytes, lane, 32);
            }
        }
        __syncthreads();
    }
}

}  // namespace ht_dense
}  // namespace nccl_ep

#endif  // NCCL_EP_HT_LOCAL_DISPATCH_PACK_CUH_
