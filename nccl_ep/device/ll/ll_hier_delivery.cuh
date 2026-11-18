/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "device/ll_ep_adapter.cuh"
#ifdef __CUDACC__
#include "device/device_primitives.cuh"
#include "device/ll/ll_common.cuh"
#include "nccl_device/coop.h"
#endif
#include "nccl_device/gin.h"
#include <algorithm>
#include <limits>
#include <type_traits>
namespace nccl_ep {
namespace ll {
// Shared launch policy and packet/completion protocol.

constexpr unsigned kHierMaxLogicalCtas = 384;
constexpr int kHierThreads = 256;
constexpr int kHierRdmaCtaShare = 2;
constexpr int kHierLsaCtaShare = 1;
constexpr int kHierMinCtasPerSm = 3;
constexpr int kHierSlotAlignment = 128;
constexpr int kHierDirectMaxTokens = 32;
// Small-capacity physical launch sizes; independent of wire layout.
constexpr int kHierDirectTinyMaxTokens = 4;
constexpr int kHierDirectSmallMaxTokens = 8;
constexpr int kHierDirectMediumMaxTokens = 16;
constexpr int kHierDirectGridCtas = 256;
constexpr int kHierDirectReducedGridCtas = 128;
constexpr int kHierDirectThreads = 128;
constexpr int kHierDirectGridQuantum = 32;

// Capacity/topology eligibility; incoming work is decoded at execution time.
namespace hier_receive_policy {
constexpr int kMinTokens = 64;
constexpr int kMaxCapacity = 8192;
constexpr int kRanks = 16;
constexpr int kLsaRanks = 8;
constexpr int kRdmaDomains = 2;
constexpr int kHidden = 7168;
constexpr int kFirstTopk = 6;
constexpr int kSecondTopk = 8;
} // namespace hier_receive_policy

template <typename WeightT>
inline bool hierSelectRoundRobinReceive(const DispatchParams<WeightT>& params) {
  if (!params.roundRobinReceiveSupported ||
      params.maxTokensPerRank < hier_receive_policy::kMinTokens ||
      params.maxTokensPerRank > hier_receive_policy::kMaxCapacity ||
      params.numRanks != hier_receive_policy::kRanks ||
      params.hidden != hier_receive_policy::kHidden ||
      params.hier.numLogicalCtas != int(kHierMaxLogicalCtas) ||
      params.tokenDtype != ncclBfloat16 ||
      params.layout != NCCL_EP_LAYOUT_RANK_MAJOR ||
      params.nvlinkOnly) {
    return false;
  }
  return params.numTopk == hier_receive_policy::kFirstTopk || params.numTopk == hier_receive_policy::kSecondTopk;
}

inline int hierLogicalCtas(unsigned smBudget, unsigned smCount) {
  const unsigned limit = smBudget == smCount ? kHierMinCtasPerSm * smCount : smBudget;
  return std::min(kHierMaxLogicalCtas, limit);
}

// Logical send-CTA counts fix token strides and completion indices across rank-local launches.
__host__ __device__ constexpr int hierNumRdmaSendCtas(int numLogicalCtas) {
  return numLogicalCtas * kHierRdmaCtaShare / (kHierRdmaCtaShare + kHierLsaCtaShare);
}

static_assert(kHierDirectMaxTokens < hierNumRdmaSendCtas(kHierMaxLogicalCtas),
              "Direct sends require sparse source-count completion");

// Direct local producers publish only N done words, so CTA zero must encode N < S.
__host__ __device__ constexpr int hierDirectCutoff(int numLogicalCtas) {
  const int sparseMaxTokens = hierNumRdmaSendCtas(numLogicalCtas) - 1;
  return sparseMaxTokens < kHierDirectMaxTokens ? sparseMaxTokens : kHierDirectMaxTokens;
}

struct HierLaunchGeometry {
  int blocks;
  int threads;
  int numRdmaRecvCtas;
  int minCtasPerSm;
};

inline HierLaunchGeometry selectHierGeometry(int numLogicalCtas, int capacity) {
  const int cutoff = hierDirectCutoff(numLogicalCtas);
  const bool direct = capacity <= cutoff;
  HierLaunchGeometry result{numLogicalCtas, kHierThreads, hierNumRdmaSendCtas(numLogicalCtas), kHierMinCtasPerSm};
  if (!direct) {
    return result;
  }
  int requested = kHierDirectGridCtas;
  if (capacity == 0 || (capacity > kHierDirectSmallMaxTokens && capacity <= kHierDirectMediumMaxTokens)) {
    requested = kHierDirectReducedGridCtas;
  }
  const int active = std::min(hierNumRdmaSendCtas(numLogicalCtas), std::max(1, capacity));
  result.blocks = std::min(numLogicalCtas, std::max(requested, nccl_ep::align(active, kHierDirectGridQuantum)));
  result.threads = kHierDirectThreads;
  if (capacity > kHierDirectTinyMaxTokens && capacity <= kHierDirectMediumMaxTokens) {
    result.threads = kHierThreads;
  }
  result.numRdmaRecvCtas = result.blocks / 2;
  result.minCtasPerSm = result.blocks <= kHierDirectReducedGridCtas ? 1 : 2;
  return result;
}

// Payloads overlay ordinary LL; persistent hierarchy controls follow both payload banks.
inline void hierOverlayLayout(nccl_ep::LowLatencyLayout& layout, HierDispatchResources& resources,
                              size_t tokens, int numRanks, size_t previousPayloadStride) {
  const int ctas = resources.numLogicalCtas;
  const int sendCtas = hierNumRdmaSendCtas(ctas);
  const size_t capacity = nccl_ep::ceil_div(tokens, size_t(sendCtas));
  const size_t payloadBase = layout.buffer.dispatch_rdma_send_buffer_offset;
  resources.relayOffset = payloadBase + tokens * resources.slotBytes;
  resources.finalOffset = resources.relayOffset + numRanks * sendCtas * capacity * resources.slotBytes;
  const size_t payloadBytes = resources.finalOffset - payloadBase + numRanks * tokens * resources.slotBytes;
  layout.payload_slot_stride = nccl_ep::align(
      std::max({layout.payload_slot_stride, payloadBytes, previousPayloadStride}), size_t(256));
  resources.readyOffset = payloadBase + 2 * layout.payload_slot_stride;
  resources.doneOffset = resources.readyOffset + numRanks * tokens * sizeof(uint64_t);
  resources.arrivalsOffset = resources.doneOffset + numRanks * ctas * sizeof(uint64_t);
  resources.flatCompletionOffset = resources.arrivalsOffset + sizeof(uint64_t);
  resources.controlBankStride = nccl_ep::align(
      resources.flatCompletionOffset + numRanks * ctas * sizeof(uint64_t) - resources.readyOffset, size_t(256));
  layout.total_bytes = resources.readyOffset + 2 * resources.controlBankStride;
}

// Each bit names an existing top-k ROUTING ENTRY, never a world GPU rank.
// Expert IDs in those entries identify GPUs, including LSA domains of 72+ GPUs.
// The packet generation is only a compact label. Full-width ready/done/counter
// tags remain the readiness authority; dormant slots are not cleared on wrap.
struct alignas(uint64_t) HierDeliveryHeader {
  uint32_t deliveryEntries;
  uint32_t control;
};
static_assert(sizeof(HierDeliveryHeader) == sizeof(uint64_t), "Unexpected delivery header size");
constexpr int kHierPacketEpochBits = 4;
constexpr uint32_t kHierDeliveryDirect = 1U << kHierPacketEpochBits;
constexpr uint32_t kHierPacketEpochMask = kHierDeliveryDirect - 1;
constexpr int kHierCountBits = std::numeric_limits<uint32_t>::digits;
constexpr uint32_t kHierCountEpochMask = 0x3fffffffU;
constexpr uint32_t kHierCountBias = 2; // Reserve zero/uninitialized and the dense terminal.
constexpr uint32_t kHierDenseTerminal = 1;
constexpr uint64_t kHierCountMarker = uint64_t(1) << 63;
constexpr uint64_t kHierDirectMarker = uint64_t(1) << 62;

// Encodes the packet's compact generation tag and direct-delivery flag.
__host__ __device__ constexpr uint32_t hierPacketControl(uint64_t epoch, bool direct) {
  uint32_t control = uint32_t((epoch + 1) & kHierPacketEpochMask);
  if (direct) {
    control |= kHierDeliveryDirect;
  }
  return control;
}

// Reports whether this packet was sent directly to its final GPU.
__host__ __device__ constexpr bool hierDeliveryIsDirect(const HierDeliveryHeader* header) {
  return (header->control & kHierDeliveryDirect) != 0;
}

// Checks that a routing entry is in range and enabled by the delivery mask.
__host__ __device__ constexpr bool hierDeliveryEntryEnabled(const HierDeliveryHeader* header, int entry) {
  if (entry < 0) {
    return false;
  }
  constexpr int kEntryBits = std::numeric_limits<decltype(header->deliveryEntries)>::digits;
  if (entry >= kEntryBits) {
    return false;
  }
  const uint32_t entryMask = uint32_t(1) << entry;
  return (header->deliveryEntries & entryMask) != 0;
}

// Selects direct delivery when the local token count fits the cutoff.
__host__ __device__ constexpr bool hierSelectDirect(int cutoff, int tokens) {
  if (cutoff <= 0) {
    return false;
  }
  return tokens <= cutoff;
}

// Encodes the source count, truncated generation and sender mode in a completion word.
__host__ __device__ constexpr uint64_t hierDeliveryCount(uint64_t epoch, int count, bool direct) {
  uint64_t word = kHierCountMarker;
  if (direct) {
    word |= kHierDirectMarker;
  }
  const uint64_t generation = epoch & kHierCountEpochMask;
  const uint32_t encodedCount = uint32_t(count) + kHierCountBias;
  word |= generation << kHierCountBits;
  word |= encodedCount;
  return word;
}

// Returns the source count, or -1 for an unmarked, stale or reserved completion word.
__host__ __device__ constexpr int hierDeliveryDecode(uint64_t word, uint64_t epoch) {
  if (!(word & kHierCountMarker)) {
    return -1;
  }
  const uint64_t generation = (word >> kHierCountBits) & kHierCountEpochMask;
  if (generation != (epoch & kHierCountEpochMask)) {
    return -1;
  }
  const uint32_t encodedCount = uint32_t(word);
  if (encodedCount < kHierCountBias) {
    return -1;
  }
  return int(encodedCount - kHierCountBias);
}

// Accepts a relay's full generation tag or send-CTA-zero packed sparse completion.
__host__ __device__ constexpr bool hierDeliveryRelayComplete(uint64_t word, uint64_t epoch, int sendCtaIdx) {
  if (word == epoch + 1) {
    return true;
  }
  if (sendCtaIdx != 0) {
    return false;
  }
  if (word & kHierDirectMarker) {
    return false;
  }
  return hierDeliveryDecode(word, epoch) >= 0;
}

// Subtract local ranks preceding this remote peer to keep context indices dense.
__host__ __device__ inline int hierRemotePeerOrdinal(int peer, int localFirst, int localSize, int localStride) {
  int numLocalBeforePeer = 0;
  if (peer > localFirst) {
    numLocalBeforePeer = ::nccl_ep::ceil_div(peer - localFirst, localStride);
    if (numLocalBeforePeer > localSize) {
      numLocalBeforePeer = localSize;
    }
  }
  return peer - numLocalBeforePeer;
}
#ifdef __CUDACC__
// Control and payload banks share LL epoch parity but have independent strides.
template <typename WeightT>
__device__ __forceinline__ size_t hierPayloadBank(const dispatch_kernel_args_t<WeightT>& args, size_t controlBank) {
  return controlBank == 0 ? 0 : args.payloadSlotStride;
}
template <typename WeightT>
__device__ __forceinline__ bool hierUseDirect(const dispatch_kernel_args_t<WeightT>& args) {
  const int cutoff = hierDirectCutoff(args.hier.numLogicalCtas);
  return hierSelectDirect(cutoff, args.numTokens);
}
__device__ __forceinline__ const HierDeliveryHeader* hierDeliveryHeader(const uint8_t* bytes, int topk) {
  return reinterpret_cast<const HierDeliveryHeader*>(bytes + get_dispatch_hdr_sz<NCCL_EP_LAYOUT_RANK_MAJOR>(topk));
}
// All members inspect routing entries; the leader stamps the header before the caller's publication join.
template <typename Coop, typename WeightT>
__device__ __forceinline__ void hierSetDeliveryHeader(Coop coop, const dispatch_kernel_args_t<WeightT>& args, uint8_t* bytes,
                                                     uint64_t epoch, int topk) {
  static_assert(std::is_same_v<Coop, ncclCoopThread> || std::is_same_v<Coop, ncclCoopWarp>,
                "Delivery header construction requires one thread or a complete warp");
  uint32_t entries = 0;
  const auto* hdr = reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(bytes);
  for (int base = 0; base < topk; base += coop.size()) {
    const int entry = base + coop.thread_rank();
    bool enabled = false;
    if (entry < topk) {
      const int rank = getExpertRankIdx(hdr->rtr[entry].expert_id, args.numExperts / args.numRanks);
      enabled = rank >= 0 && rank < args.numRanks;
    }
    uint32_t batch = uint32_t(enabled);
    if constexpr (!ncclCoopIsThread(Coop{})) {
      batch = __ballot_sync(0xffffffff, enabled);
    }
    entries |= batch << base;
  }
  if (coop.thread_rank() == 0) {
    auto* meta = const_cast<HierDeliveryHeader*>(hierDeliveryHeader(bytes, topk));
    meta->control = hierPacketControl(epoch, hierUseDirect(args));
    meta->deliveryEntries = entries;
  }
}
template <typename WeightT>
__device__ __forceinline__ const uint64_t* hierDirectCounter(const dispatch_kernel_args_t<WeightT>& args, size_t bank, int srcRank,
                                                             int sendCtaIdx) {
  return reinterpret_cast<const uint64_t*>(args.hier.buffer + bank + args.hier.flatCompletionOffset) +
         size_t(srcRank) * args.hier.numLogicalCtas + sendCtaIdx;
}
template <typename WeightT>
__device__ __forceinline__ bool hierSendCtaComplete(const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch,
                                                    int srcRank, int sendCtaIdx) {
  const auto* done = reinterpret_cast<const uint64_t*>(args.hier.buffer + bank + args.hier.doneOffset) +
                     size_t(srcRank) * args.hier.numLogicalCtas + sendCtaIdx;
  const uint64_t word = ld_acquire_sys_global<true>(done);
  if (hierDeliveryRelayComplete(word, epoch, sendCtaIdx)) {
    return true;
  }
  return (word & kHierDirectMarker) && hierDeliveryDecode(word, epoch) >= 0 &&
         ld_acquire_sys_global<true>(hierDirectCounter(args, bank, srcRank, sendCtaIdx)) == uint64_t(epoch) + 1;
}
// Indexed relay signals must be posted, polled, and reset on the same context.
template <typename WeightT>
__device__ __forceinline__ ncclGin hierGin(const dispatch_kernel_args_t<WeightT>& args, int producer,
                                          ncclGinResourceSharingMode sharing) {
  return ncclGin(*args.devComm, producer % args.devComm->ginContextCount, sharing);
}
template <typename WeightT>
__device__ __forceinline__ int hierRelayTaskIndex(const dispatch_kernel_args_t<WeightT>& args, int source, int sendCta) {
  return source * hierNumRdmaSendCtas(args.hier.numLogicalCtas) + sendCta;
}
template <typename WeightT>
__device__ __forceinline__ unsigned hierRelaySignal(const dispatch_kernel_args_t<WeightT>& args, uint64_t epoch,
                                                   int source, int sendCta) {
  const unsigned bank = (epoch & 1U) * args.numRanks * hierNumRdmaSendCtas(args.hier.numLogicalCtas);
  return args.hier.signalBase + bank + hierRelayTaskIndex(args, source, sendCta);
}
// Capped budgets share contexts through NCCL's GPU-safe producer arbitration.
template <typename WeightT>
__device__ __forceinline__ ncclGin hierSendGin(const dispatch_kernel_args_t<WeightT>& args, int sendCtaIdx) {
  const int activeSendCtas = max(1, min(hierNumRdmaSendCtas(args.hier.numLogicalCtas), args.numTokens));
  const auto sharing = activeSendCtas <= args.devComm->ginContextCount ?
    NCCL_GIN_RESOURCE_SHARING_THREAD : NCCL_GIN_RESOURCE_SHARING_GPU;
  return hierGin(args, sendCtaIdx, sharing);
}

template <typename WeightT>
__device__ __forceinline__ void hierSendDirectPeer(
  ncclCoopWarp coop, ncclGin& net, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch, int sendCtaIdx,
  int peer, int hidden, int topk, uint64_t delta) {
  const auto& resources = args.hier;
  const auto world = ncclTeamWorld(*args.devComm);
  const uint64_t tag = epoch + 1;
  // One token per producer bounds deferral; shared queues keep immediate progress.
  uint32_t packetFlags = ncclGinOptFlagsDefault;
  if (net.resourceSharingMode == NCCL_GIN_RESOURCE_SHARING_THREAD &&
      args.numTokens <= hierNumRdmaSendCtas(resources.numLogicalCtas)) {
    packetFlags = ncclGinOptFlagsAggregateRequests;
  }
  for (int token = sendCtaIdx; token < args.numTokens; token += hierNumRdmaSendCtas(resources.numLogicalCtas)) {
    const size_t src = hierPayloadBank(args, bank) + args.sendOff + size_t(token) * resources.slotBytes;
    const auto* hdr = reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(resources.buffer + src);
    bool target = false;
    for (int j = coop.thread_rank(); j < topk; j += coop.size()) {
      target |= getExpertRankIdx(hdr->rtr[j].expert_id, args.numExperts / args.numRanks) == peer;
    }
    target = __any_sync(0xffffffff, target);
    if (!target || coop.thread_rank() != 0) {
      continue;
    }
    const size_t slot = size_t(args.currRank) * args.maxTokensPerRank + token;
    const size_t dst = hierPayloadBank(args, bank) + resources.finalOffset + slot * resources.slotBytes;
    net.put(world, peer, resources.window, dst, resources.window, src,
            resources.headerBytes + size_t(hidden) * sizeof(nv_bfloat16),
            ncclGin_None{},
            ncclGin_None{}, ncclCoopThread(), ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, packetFlags);
    net.putValue(world, peer, resources.window, bank + resources.readyOffset + slot * sizeof(uint64_t), tag,
                 ncclGin_None{}, ncclCoopThread(),
                 ncclGin_None{}, cuda::thread_scope_device, cuda::thread_scope_device, packetFlags);
  }
  if (coop.thread_rank() == 0) {
    const size_t index = size_t(args.currRank) * resources.numLogicalCtas + sendCtaIdx;
    net.putValue(world, peer, resources.window, bank + resources.doneOffset + index * sizeof(uint64_t),
                 hierDeliveryCount(epoch, args.numTokens, true),
                 ncclGin_StrongVASignalAdd{
                   resources.window, bank + resources.flatCompletionOffset + index * sizeof(uint64_t), delta},
                 ncclCoopThread(), ncclGin_None{}, cuda::thread_scope_device,
                 cuda::thread_scope_device, ncclGinOptFlagsDefault);
  }
}
// Every CTA member calls together after staged data and the delivery header
// are complete. Each peer producer owns a distinct context; THREAD sharing
// never has concurrent producers on the same context. The wire is unchanged.
template <typename WeightT>
__device__ __forceinline__ void hierSendDirect(ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                          uint64_t epoch, int sendCtaIdx, int hidden, int topk) {
  const auto& resources = args.hier;
  const auto world = ncclTeamWorld(*args.devComm);
  const auto lsa = ncclTeamLsa(*args.devComm);
  const int remotePeers = args.numRanks - lsa.nRanks;
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = coop.thread_rank() / warp.size();
  const int numActiveSendCtas = max(1, min(hierNumRdmaSendCtas(resources.numLogicalCtas), args.numTokens));
  const bool parallelPeers = numActiveSendCtas * remotePeers <= args.devComm->ginContextCount;
  const uint64_t tag = uint64_t(epoch) + 1;
  const uint64_t lastPostedDirectTag = *hierDirectCounter(args, bank, args.currRank, sendCtaIdx);
  const uint64_t delta = tag - lastPostedDirectTag;
  const int localFirst = ncclTeamRankToWorld(*args.devComm, lsa, 0);
  // Separate warps keep peer-dependent GIN progress from diverging within a warp.
  // With fewer contexts, warp zero visits every peer on the send CTA's context.
  const int numPostingWarps = parallelPeers ? coop.size() / warp.size() : 1;
  if (warpIdx < numPostingWarps) {
    for (int peer = warpIdx; peer < args.numRanks; peer += numPostingWarps) {
      if (ncclTeamRankIsMember(lsa, world, peer)) {
        continue;
      }
      const int peerOrdinal = hierRemotePeerOrdinal(peer, localFirst, lsa.nRanks, lsa.stride);
      auto net = parallelPeers ? ncclGin(*args.devComm, sendCtaIdx * remotePeers + peerOrdinal,
                                       NCCL_GIN_RESOURCE_SHARING_THREAD) : hierSendGin(args, sendCtaIdx);
      // Publish staged writes before claiming device release on this producer's puts.
      if (warp.thread_rank() == 0) {
        __threadfence();
      }
      hierSendDirectPeer(warp, net, args, bank, epoch, sendCtaIdx, peer, hidden, topk, delta);
    }
  }
  // This rank's own source slot is sender-only history; local LSA delivery
  // uses ordinary done words and never consumes this counter.
  coop.sync();
  if (coop.thread_rank() == 0) {
    *const_cast<uint64_t*>(hierDirectCounter(args, bank, args.currRank, sendCtaIdx)) = tag;
  }
}

// One relay rank owns each source-rank/send-CTA/domain tuple, including empty token subsets.
// Under FULL connectivity, use the first GPU needed by this send CTA.
// With 256 tokens and RDMA send CTAs, each send CTA has at most one token, which
// lands on an actual destination. With more tokens per send CTA, later packets
// keep that relay rank to preserve one terminal signal and LSA completion per send CTA.
template <typename WeightT>
__device__ __forceinline__ int hierChooseRelayRank(ncclCoopWarp coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                int sendCtaIdx, int domain, int topk) {
  const auto& resources = args.hier;
  const auto lsa = ncclTeamLsa(*args.devComm);
  for (int token = sendCtaIdx; token < args.numTokens; token += hierNumRdmaSendCtas(resources.numLogicalCtas)) {
    const auto* hdr =
      reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(
        resources.buffer + hierPayloadBank(args, bank) + args.sendOff + size_t(token) * resources.slotBytes);
    for (int base = 0; base < topk; base += coop.size()) {
      const int entry = base + coop.thread_rank();
      int rank = -1;
      if (entry < topk) {
        rank = getExpertRankIdx(hdr->rtr[entry].expert_id, args.numExperts / args.numRanks);
      }
      const bool target = rank >= 0 && rank < args.numRanks && rank / lsa.nRanks == domain;
      const unsigned matches = __ballot_sync(0xffffffff, target);
      if (matches) {
        // The first matching entry preserves serial relay ownership.
        return ncclCoopBcast(coop, rank, __ffs(matches) - 1, false);
      }
    }
  }
  // Empty FULL token subsets also need a relay rank for their completion signal.
  return domain * lsa.nRanks + lsa.rank;
}

template <typename WeightT>
__device__ __forceinline__ void hierSendRelay(ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                    uint64_t epoch, int sendCtaIdx, int hidden, int topk) {
  const auto& resources = args.hier;
  const int numRdmaSendCtas = hierNumRdmaSendCtas(resources.numLogicalCtas);
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = coop.thread_rank() / warp.size();
  coop.sync(); // Include all staged payload/header writes and the header stamp.
  if (warpIdx == 0 && sendCtaIdx < max(1, min(numRdmaSendCtas, args.numTokens))) {
    if (warp.thread_rank() == 0) {
      __threadfence();
    }
    const auto rail = ncclTeamRail(*args.devComm);
    const auto team = ncclTeamWorld(*args.devComm);
    const auto lsa = ncclTeamLsa(*args.devComm);
    const int srcRankIdx = args.currRank;
    const int capacity = nccl_ep::ceil_div(args.maxTokensPerRank, numRdmaSendCtas);
    auto net = hierSendGin(args, sendCtaIdx);
    const uint32_t payloadFlags = net.resourceSharingMode == NCCL_GIN_RESOURCE_SHARING_THREAD ?
      ncclGinOptFlagsAggregateRequests : ncclGinOptFlagsDefault;
    for (int domain = 0; domain < rail.nRanks; ++domain) {
      if (domain == rail.rank) {
        continue;
      }
      const int peer = hierChooseRelayRank(warp, args, bank, sendCtaIdx, domain, topk);
      const unsigned signal = hierRelaySignal(args, epoch, srcRankIdx, sendCtaIdx);
      int sent = 0;
      for (int token = sendCtaIdx; token < args.numTokens; token += numRdmaSendCtas) {
        const size_t src = hierPayloadBank(args, bank) + args.sendOff + size_t(token) * resources.slotBytes;
        const auto* hdr = reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(resources.buffer + src);
        bool target = false;
        for (int j = warp.thread_rank(); j < topk; j += warp.size()) {
          const int rank = getExpertRankIdx(hdr->rtr[j].expert_id, args.numExperts / args.numRanks);
          target |= rank >= 0 && rank < args.numRanks && rank / lsa.nRanks == domain;
        }
        target = __any_sync(0xffffffff, target);
        if (!target) {
          continue;
        }
        if (warp.thread_rank() == 0) {
          const size_t dst =
            hierPayloadBank(args, bank) + resources.relayOffset +
            (size_t(hierRelayTaskIndex(args, srcRankIdx, sendCtaIdx)) * capacity + sent) * resources.slotBytes;
          net.put(team, peer, resources.window, dst + resources.headerBytes, resources.window,
                  src + resources.headerBytes, hidden * sizeof(nv_bfloat16), ncclGin_None{},
                  ncclGin_None{}, ncclCoopThread(), ncclGin_None{}, cuda::thread_scope_device,
                  cuda::thread_scope_device, payloadFlags);
          const size_t headerBytes = get_dispatch_hdr_sz<NCCL_EP_LAYOUT_RANK_MAJOR>(topk) + sizeof(HierDeliveryHeader);
          net.put(team, peer, resources.window, dst, resources.window, src, headerBytes,
                  ncclGin_StrongSignalAdd{unsigned(signal), 1}, ncclGin_None{}, ncclCoopThread(),
                  ncclGin_None{}, cuda::thread_scope_device);
        }
        ++sent;
      }
      if (warp.thread_rank() == 0) {
        net.signal(team, peer,
                   ncclGin_StrongSignalAdd{
                     unsigned(signal),
                     (args.numTokens >= numRdmaSendCtas ? uint64_t(kHierDenseTerminal) :
                       uint64_t(args.numTokens) + kHierCountBias) << kHierCountBits},
                   ncclCoopThread(), ncclGin_None{}, cuda::thread_scope_device);
      }
    }
  }
  coop.sync();
}

// Cooperative LSA forwarding.


// Finish every reader before a leader can reuse the shared control cell.
template <typename Coop, typename T>
__device__ __forceinline__ T hierBroadcastControl(Coop coop, T& scratch, T value) {
  if (coop.thread_rank() == 0) {
    scratch = value;
  }
  coop.sync();
  const T snapshot = scratch;
  coop.sync();
  return snapshot;
}

// Concurrent cooperative groups own separate destination lists.
template <int K>
struct HierForwardScratch {
  uint32_t* destinations[K];
  uint32_t* headers[K];
  uint64_t* ready[K];
  int count;
};

template <int K, typename Idx, typename Coop, typename WeightT>
__device__ __forceinline__ void hierMakeHeader(Coop coop, uint8_t* bytes, const dispatch_kernel_args_t<WeightT>& args,
                                                   int token) {
  auto* hdr = reinterpret_cast<DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(bytes);
  const int worker = coop.thread_rank();
  if (worker == 0) {
    hdr->token_id = token;
  }
  for (int j = worker; j < K; j += coop.size()) {
    hdr->rtr[j].expert_id = static_cast<uint16_t>(static_cast<const Idx*>(args.inTopkIdx)[token * K + j]);
    hdr->rtr[j].topk_weight = static_cast<float>(args.inTopkWeights[token * K + j]);
  }
  // The caller chooses the synchronization boundary before consuming it.
}

template <int K, typename Coop, typename WeightT>
__device__ __forceinline__ void hierBuildForwardDestinations(Coop coop, HierForwardScratch<K>& scratch,
                                                             const dispatch_kernel_args_t<WeightT>& args, size_t bank, int srcRank,
                                                             const uint8_t* header) {
  static_assert(std::is_same_v<Coop, ncclCoopWarp> || std::is_same_v<Coop, ncclCoopCta>,
                "Forward destination discovery requires a complete warp or CTA");
  static_assert(K > 0 && K <= MAX_NUM_TOPK, "Unsupported top-k");
  const auto& resources = args.hier;
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = coop.thread_rank() / warp.size();
  const int lane = warp.thread_rank();
  const auto* hdr = reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(header);
  const ncclTeam lsa = ncclTeamLsa(*args.devComm);
  const ncclTeam world = ncclTeamWorld(*args.devComm);
  const size_t slot = static_cast<size_t>(srcRank) * args.maxTokensPerRank + hdr->token_id;
  const size_t off = hierPayloadBank(args, bank) + resources.finalOffset + slot * resources.slotBytes;
  if constexpr (K <= ::nccl_ep::kWarpSize) {
    if (warpIdx == 0) {
      int rank = -1;
      if (lane < K) {
        rank = getExpertRankIdx(hdr->rtr[lane].expert_id, args.numExperts / args.numRanks);
      }
      const bool valid =
        lane < K && (srcRank == args.currRank || hierDeliveryEntryEnabled(hierDeliveryHeader(header, K), lane)) &&
        rank >= 0 && rank < args.numRanks && ncclTeamRankIsMember(lsa, world, rank) &&
        (srcRank == args.currRank || !hierDeliveryIsDirect(hierDeliveryHeader(header, K)) || rank == args.currRank);
      // All lanes participate. Keep the first expert for each rank,
      // preserving the old list order while removing its nested loop.
      const unsigned sameRank = __match_any_sync(0xffffffff, rank);
      const unsigned earlier = (1U << lane) - 1U;
      const unsigned candidates = __ballot_sync(0xffffffff, valid);
      const bool owner = valid && !(sameRank & earlier & candidates);
      const unsigned owners = __ballot_sync(0xffffffff, owner);
      if (lane == 0) {
        scratch.count = __popc(owners);
      }
      if (owner) {
        const int index = __popc(owners & earlier);
        auto* peer = static_cast<uint8_t*>(ncclGetPeerPointer(resources.window, 0, rank));
        scratch.headers[index] = reinterpret_cast<uint32_t*>(peer + off);
        scratch.destinations[index] = reinterpret_cast<uint32_t*>(peer + off + resources.headerBytes);
        scratch.ready[index] = reinterpret_cast<uint64_t*>(peer + bank + resources.readyOffset + slot * sizeof(uint64_t));
      }
    }
  }
  coop.sync();
}

template <int Hidden, int K, typename Coop>
__device__ __forceinline__ void hierCopyForwardDestinations(
  Coop coop, const HierForwardScratch<K>& scratch, uint64_t tag, const uint8_t* header, const uint8_t* payload) {
  const int worker = coop.thread_rank();
  const int count = scratch.count;
  coop.sync(); // Empty lists must finish their readers before scratch reuse.
  if (!count) {
    return; // Uniform within this cooperative group.
  }
  const int words = get_dispatch_hdr_sz<NCCL_EP_LAYOUT_RANK_MAJOR>(K) / sizeof(uint32_t);
  constexpr int kPayloadVectors = Hidden * sizeof(nv_bfloat16) / sizeof(int4);
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = worker / warp.size();
  const auto copy = [&](const auto* source, uint32_t* const* destinations, int elements) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
    const int elementsPerWarp = nccl_ep::ceil_div(elements, coop.size() / warp.size());
    const int first = min(warpIdx * elementsPerWarp, elements);
    const int copyElements = min(elementsPerWarp, elements - first);
    auto* destination = reinterpret_cast<T*>(destinations[0]);
    auto load = [](const T* ptr) { return *ptr; };
    auto store = [&](T* ptr, const T& value) {
      const auto offset = ptr - destination;
      for (int i = 0; i < count; ++i) {
        reinterpret_cast<T*>(destinations[i])[offset] = value;
      }
    };
    UNROLLED_WARP_COPY(4, warp.thread_rank(), copyElements,
                       destination + first, source + first, load, store);
  };
  copy(reinterpret_cast<const uint32_t*>(header), scratch.headers, words);
  copy(reinterpret_cast<const int4*>(payload), scratch.destinations, kPayloadVectors);
  coop.sync();
  for (int i = worker; i < count; i += coop.size()) {
    *scratch.ready[i] = tag;
  }
  coop.sync();
  // Send-CTA completion is published separately, after every producer group
  // has finished. Ready tags alone do not release peer payload writes.
}

template <int Hidden, int K, typename Coop, typename WeightT>
__device__ __forceinline__ void hierForward(Coop coop, HierForwardScratch<K>& scratch,
                                                const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t tag, int srcRank,
                                                const uint8_t* header, const uint8_t* payload) {
  hierBuildForwardDestinations<K>(coop, scratch, args, bank, srcRank, header);
  hierCopyForwardDestinations<Hidden, K>(coop, scratch, tag, header, payload);
}

template <typename Coop, typename WeightT>
__device__ __forceinline__ void hierPublishDone(Coop coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                    uint64_t tag, int srcRank, int sendCtaIdx) {
  static_assert(std::is_same_v<Coop, ncclCoopWarp> || std::is_same_v<Coop, ncclCoopCta>,
                "Completion publication requires a complete warp or CTA");
  coop.sync();
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = coop.thread_rank() / warp.size();
  if (warpIdx == 0) {
    memory_fence_release_sys();
    const auto lsa = ncclTeamLsa(*args.devComm);
    for (int i = warp.thread_rank(); i < lsa.nRanks; i += warp.size()) {
      const int peer = ncclTeamRankToWorld(*args.devComm, lsa, i);
      const size_t doneOffset = bank + args.hier.doneOffset +
        (static_cast<size_t>(srcRank) * args.hier.numLogicalCtas + sendCtaIdx) * sizeof(uint64_t);
      auto* done = static_cast<uint64_t*>(ncclGetPeerPointer(args.hier.window, doneOffset, peer));
      st_relaxed_sys_global(done, tag);
    }
  }
  coop.sync();
}

#endif
} // namespace ll
} // namespace nccl_ep
