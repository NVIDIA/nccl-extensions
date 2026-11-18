/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "device/ll_ep_adapter.cuh"
#include "device/ll/ll_common.cuh"
#include "device/ll/ll_hier_delivery.cuh"
#include <cooperative_groups.h>

namespace nccl_ep {
namespace ll {

// All CTAs call after receive work; warp zero distributes final cleanup.
template <typename WeightT>
__device__ __forceinline__ void completeFullLowLatencyEpochHier(
  ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank) {
  coop.sync();
  auto warp = ncclCoopCoalesced(coop);
  if (coop.thread_rank() < warp.size()) {
    const auto& resources = args.hier;
    auto* arrivals = reinterpret_cast<unsigned*>(resources.buffer + bank + resources.arrivalsOffset);
    int last = 0;
    if (warp.thread_rank() == 0) {
      last = atomicAdd(arrivals, 1U) == gridDim.x - 1;
    }
    last = ncclCoopBcast(warp, last, 0);
    if (last) {
      auto* done = reinterpret_cast<uint64_t*>(resources.buffer + bank + resources.doneOffset);
      for (int source = warp.thread_rank(); source < args.numRanks; source += warp.size()) {
        st_relaxed_sys_global(done + size_t(source) * resources.numLogicalCtas, uint64_t(0));
      }
      if (warp.thread_rank() < args.numRanks) {
        memory_fence_release_sys();
      }
      warp.sync();
      if (warp.thread_rank() == 0) {
        *arrivals = 0;
        completeFullLowLatencyEpoch(args.epochState, args.phases, true, 0);
      }
    }
  }
}

// Fixed token-indexed final slots avoid remote slot-allocation atomics. Send-CTA
// completion makes both the ready tags and payload visible; the ordinary LSA
// receive step compacts live rows into the existing rank-major output contract.
__device__ inline void hierCheckTimeout(uint64_t start, uint64_t timeout) {
  if (clock64() - start > timeout) {
    printf("hierarchical dispatch timeout: rank-local CTA %u\n", blockIdx.x);
    trap();
  }
}

// Send CTA zero's done word carries the count; no control array or
// extra peer broadcast is allocated. Initial zero is invalid even at epoch 0.
__device__ __forceinline__ uint64_t hierCountDone(uint64_t epoch, int tokens) {
  return hierDeliveryCount(epoch, tokens, false);
}
template <typename WeightT>
__device__ __forceinline__ int hierReadDoneCount(const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch,
                                                 int srcRank, bool* sourceDirect) {
  if (sourceDirect) {
    *sourceDirect = false;
  }
  const auto* done =
    reinterpret_cast<const uint64_t*>(args.hier.buffer + bank + args.hier.doneOffset) + size_t(srcRank) * args.hier.numLogicalCtas;
  const uint64_t value = ld_acquire_sys_global<true>(done);
  if (value == uint64_t(epoch) + 1) {
    return args.maxTokensPerRank;
  }
  const int count = hierDeliveryDecode(value, epoch);
  if (count < 0) {
    return -1;
  }
  if ((value & kHierDirectMarker) &&
      ld_acquire_sys_global<true>(hierDirectCounter(args, bank, srcRank, 0)) != uint64_t(epoch) + 1) {
    return -1;
  }
  // A valid send-CTA-zero control word identifies this source rank's mode for
  // the whole dispatch, not completion of its other send CTAs.
  if (sourceDirect) {
    *sourceDirect = (value & kHierDirectMarker) != 0;
  }
  return count;
}
template <typename Coop, typename WeightT>
__device__ __forceinline__ int hierWaitDoneCount(Coop coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                 uint64_t epoch, int srcRank) {
  int tokens = -1;
  if (coop.thread_rank() == 0) {
    const uint64_t start = clock64();
    while ((tokens = hierReadDoneCount(args, bank, epoch, srcRank, nullptr)) < 0) {
      hierCheckTimeout(start, args.timeoutCycles);
    }
  }
  // Warp broadcasts shuffle registers but do not order peer memory loads.
  // Hand the leader's system acquire to every consuming group member.
  coop.sync();
  return ncclCoopBcast(coop, tokens, 0);
}
template <int Hidden, int K, typename WeightT>
__device__ __forceinline__ void hierForward(ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t tag, int srcRank,
                                            const uint8_t* header, const uint8_t* payload) {
  __shared__ HierForwardScratch<K> scratch;
  hierForward<Hidden, K>(coop, scratch, args, bank, tag, srcRank, header, payload);
}
template <typename WeightT>
__device__ __forceinline__ void hierPublishDone(ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t tag, int srcRank,
                                                int sendCtaIdx, int numSrcTokens) {
  // hierDirectCutoff guarantees N < S for Direct local producers, which publish only N done words.
  // Relay sources also need this sparse encoding whenever N < S, even above the Direct cutoff.
  const uint64_t value =
    sendCtaIdx == 0 && numSrcTokens < hierNumRdmaSendCtas(args.hier.numLogicalCtas) ? hierCountDone(tag - 1, numSrcTokens) : tag;
  hierPublishDone(coop, args, bank, value, srcRank, sendCtaIdx);
}

// One warp copies the tokens assigned to one source rank and send CTA.
template <int Hidden, int K, typename WeightT>
__device__ __forceinline__ void hierCopyFinalTokens(ncclCoopWarp coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                    uint64_t tag, int srcRank, int sendCtaIdx, int numSendCtas,
                                                    int tokenLimit) {
  constexpr int kPayloadVectors = Hidden / ::nccl_ep::kBf16PerInt4;
  const auto& resources = args.hier;
  const int lane = coop.thread_rank();
  // Every lane probes a different slot, then the whole warp copies each hit.
  // The caller acquired completion for this source rank and send CTA.
  for (int baseToken = sendCtaIdx; baseToken < tokenLimit; baseToken += coop.size() * numSendCtas) {
    const int probe = baseToken + lane * numSendCtas;
    const size_t probeSlot = size_t(srcRank) * args.maxTokensPerRank + probe;
    const auto* ready = reinterpret_cast<const uint64_t*>(resources.buffer + bank + resources.readyOffset);
    const bool present = probe < tokenLimit && ready[probeSlot] == tag;
    unsigned mask = __ballot_sync(0xffffffff, present);
    while (mask) {
      const int token = baseToken + (__ffs(mask) - 1) * numSendCtas;
      const size_t srcSlot = size_t(srcRank) * args.maxTokensPerRank + token;
      int slot = lane == 0 ? atomicAdd(args.outSrcInfo + srcRank, 1) : 0;
      slot = __shfl_sync(0xffffffff, slot, 0);
      if (lane == 0 && args.outRecvRankCounter) {
        atomicAdd(args.outRecvRankCounter + srcRank, 1);
      }
      slot += srcRank * args.maxTokensPerRank;
      const auto* msg = resources.buffer + hierPayloadBank(args, bank) + resources.finalOffset + srcSlot * resources.slotBytes;
      recordRankMajorRecvTokenMetadata(reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(msg), slot, K,
                                       args.numExperts / args.numRanks, args.currRank, lane,
                                       args.outSrcInfo + args.numRanks + slot * (K + 1), args.outRecvTopkIdx,
                                       args.outRecvTopkWeights, args.recvTopkIdxKind);
      auto load = [](const int4* ptr) { return *ptr; };
      auto store = [](int4* ptr, const int4& value) { *ptr = value; };
      UNROLLED_WARP_COPY(4, lane, kPayloadVectors,
        static_cast<int4*>(args.outDataBuf) + size_t(slot) * kPayloadVectors,
        reinterpret_cast<const int4*>(msg + resources.headerBytes), load, store);
      mask &= mask - 1;
    }
  }
}

// All Coop members call together; metadata uses its first complete warp.
template <int Hidden, int K, typename Coop, typename WeightT>
__device__ __forceinline__ void hierCopyFinalSingleton(Coop coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank,
                                                       uint64_t tag, int srcRank) {
  static_assert(std::is_same_v<Coop, ncclCoopWarp> || std::is_same_v<Coop, ncclCoopCta>,
                "Final singleton metadata requires a complete warp or CTA");
  constexpr int kPayloadVectors = Hidden / ::nccl_ep::kBf16PerInt4;
  const auto& resources = args.hier;
  const int worker = coop.thread_rank();
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = worker / warp.size();
  const size_t srcSlot = static_cast<size_t>(srcRank) * args.maxTokensPerRank;
  const auto* ready = reinterpret_cast<const uint64_t*>(resources.buffer + bank + resources.readyOffset) + srcSlot;
  int present = worker == 0 ? int(*ready == tag) : 0;
  present = ncclCoopBcast(coop, present, 0);
  if (!present) {
    return;
  }
  int slot = worker == 0 ? atomicAdd(args.outSrcInfo + srcRank, 1) : 0;
  slot = ncclCoopBcast(coop, slot, 0);
  if (worker == 0 && args.outRecvRankCounter) {
    atomicAdd(args.outRecvRankCounter + srcRank, 1);
  }
  slot += srcRank * args.maxTokensPerRank;
  const auto* msg = resources.buffer + hierPayloadBank(args, bank) + resources.finalOffset + srcSlot * resources.slotBytes;
  if (warpIdx == 0) {
    recordRankMajorRecvTokenMetadata(reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(msg), slot, K,
                                     args.numExperts / args.numRanks, args.currRank, warp.thread_rank(),
                                     args.outSrcInfo + args.numRanks + slot * (K + 1), args.outRecvTopkIdx, args.outRecvTopkWeights,
                                     args.recvTopkIdxKind);
  }
  auto load = [](const int4* ptr) { return *ptr; };
  auto store = [](int4* ptr, const int4& value) { *ptr = value; };
  const int vectorsPerWarp = nccl_ep::ceil_div(kPayloadVectors, coop.size() / warp.size());
  const int firstVector = min(warpIdx * vectorsPerWarp, kPayloadVectors);
  const int copyVectors = min(vectorsPerWarp, kPayloadVectors - firstVector);
  UNROLLED_WARP_COPY(4, warp.thread_rank(), copyVectors,
    static_cast<int4*>(args.outDataBuf) + size_t(slot) * kPayloadVectors + firstVector,
    reinterpret_cast<const int4*>(msg + resources.headerBytes) + firstVector, load, store);
}

struct HierSourceControl {
  int count;
  bool direct;
};

template <typename WeightT>
__device__ __forceinline__ HierSourceControl hierReadSourceControl(
  ncclCoopThread, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch, int source) {
  HierSourceControl value{-1, false};
  value.count = hierReadDoneCount(args, bank, epoch, source, &value.direct);
  return value;
}

template <int NumRdmaSendCtas, typename WeightT>
__device__ __forceinline__ uint64_t hierWaitRelayPublication(
  ncclCoopThread, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch,
  int source, int sendCta, ncclGin& net, int consumed) {
  const unsigned signal = hierRelaySignal(args, epoch, source, sendCta);
  const uint64_t waitStart = clock64();
  while (true) {
    const uint64_t publication = net.readSignal(signal);
    if (static_cast<uint32_t>(publication) > consumed || publication >> kHierCountBits) {
      return publication;
    }
    if (hierSendCtaComplete(args, bank, epoch, source, sendCta)) {
      return 0; // Direct delivery or another relay rank completed this task.
    }
    if (sendCta != 0) {
      const int count = hierReadDoneCount(args, bank, epoch, source, nullptr);
      if (count >= 0 && sendCta >= max(1, min(NumRdmaSendCtas, count))) {
        return publication;
      }
    }
    hierCheckTimeout(waitStart, args.timeoutCycles);
  }
}

// The leader polls; the CTA forwards every published packet together.
template <int Hidden, int K, int NumRdmaSendCtas, typename WeightT>
__device__ __forceinline__ void hierDrainRelay(
  ncclCoopCta ctaCoop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch,
  int source, int sendCta) {
  const auto& resources = args.hier;
  const int capacity = nccl_ep::ceil_div(args.maxTokensPerRank, NumRdmaSendCtas);
  const int taskIndex = hierRelayTaskIndex(args, source, sendCta);
  auto net = hierGin(args, sendCta, NCCL_GIN_RESOURCE_SHARING_GPU);
  __shared__ uint64_t publicationScratch;
  uint64_t publication = 0;
  int consumed = 0;
  while (true) {
    uint64_t nextPublication = 0;
    if (ctaCoop.thread_rank() == 0) {
      nextPublication = hierWaitRelayPublication<NumRdmaSendCtas>(
        ncclCoopThread(), args, bank, epoch, source, sendCta, net, consumed);
    }
    publication = hierBroadcastControl(ctaCoop, publicationScratch, nextPublication);
    if (publication == 0) {
      return; // Inactive or already completed by another producer.
    }
    const int available = static_cast<uint32_t>(publication);
    for (; consumed < available; ++consumed) {
      const auto* message = resources.buffer + hierPayloadBank(args, bank) + resources.relayOffset +
        (static_cast<size_t>(taskIndex) * capacity + consumed) * resources.slotBytes;
      hierForward<Hidden, K>(ctaCoop, args, bank, epoch + 1, source, message, message + resources.headerBytes);
    }
    if (publication >> kHierCountBits) {
      break;
    }
    ctaCoop.sync();
  }
  if (ctaCoop.thread_rank() == 0) {
    net.resetSignal(hierRelaySignal(args, epoch, source, sendCta));
  }
  hierPublishDone(ctaCoop, args, bank, epoch + 1, source, sendCta,
                  (publication >> kHierCountBits) == kHierDenseTerminal ?
                    args.maxTokensPerRank : int(publication >> kHierCountBits) - int(kHierCountBias));
}

template <int Hidden, int K, int NumRdmaSendCtas, int NumRdmaRecvCtas, typename WeightT>
__device__ __forceinline__ void hierReceiveRelay(ncclCoopCta ctaCoop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch) {
  const int cta = blockIdx.x;
  const auto lsa = ncclTeamLsa(*args.devComm);

  // FULL relay staging can land on any GPU in this LSA domain. All local
  // GPUs visit source ranks and send CTAs in ascending global order.
  // Only the signaled relay rank forwards; other GPUs observe its LSA done.
  // This order avoids circular ownership waits even when
  // peers launch different physical receive-CTA counts.
  const int numSources = args.numRanks;
  // Cache only acquired, current-generation source counts. Discovery is
  // nonblocking: this CTA may itself own relay work needed for count publication.
  int cachedSource = -1;
  int cachedSourceCount = -1;
  bool cachedSourceDirect = false;
  __shared__ HierSourceControl sourceScratch;
  // Rotate receive ownership without changing source-rank/send-CTA order.
  // Every CTA visits ascending source ranks and send CTAs, including mixed-mode peers.
  // Spread send-CTA-zero ownership across source ranks.
  for (int srcRankIdx = 0; srcRankIdx < numSources; ++srcRankIdx) {
    const int ownerOffset = srcRankIdx % NumRdmaRecvCtas;
    const int firstSendCtaIdx = (cta + NumRdmaRecvCtas - ownerOffset) % NumRdmaRecvCtas;
    for (int sendCtaIdx = firstSendCtaIdx; sendCtaIdx < NumRdmaSendCtas; sendCtaIdx += NumRdmaRecvCtas) {
      const int srcRank = srcRankIdx;
      if (ncclTeamRankIsMember(lsa, ncclTeamWorld(*args.devComm), srcRank) ||
          sendCtaIdx >= max(1, min(NumRdmaSendCtas, args.maxTokensPerRank))) {
        continue;
      }
      if (cachedSource != srcRank) {
        cachedSourceCount = -1;
        cachedSource = srcRank;
      }
      if (cachedSourceCount < 0) {
        HierSourceControl value{-1, false};
        if (ctaCoop.thread_rank() == 0) {
          value = hierReadSourceControl(ncclCoopThread(), args, bank, epoch, srcRank);
        }
        const auto discovered = hierBroadcastControl(ctaCoop, sourceScratch, value);
        cachedSourceCount = discovered.count;
        cachedSourceDirect = discovered.direct;
      }
      // This source sent every packet directly this generation.
      // Final-copy CTAs still acquire each send CTA separately.
      // Relay receive CTAs have no forwarding work for this source rank.
      if (cachedSourceDirect) {
        break;
      }

      // Send CTA zero remains mandatory even for an empty source rank.
      if (cachedSourceCount >= 0 && sendCtaIdx >= max(1, min(NumRdmaSendCtas, cachedSourceCount))) {
        continue;
      }

      hierDrainRelay<Hidden, K, NumRdmaSendCtas>(ctaCoop, args, bank, epoch, srcRank, sendCtaIdx);
    }
  }
}

template <int Hidden, int K, int NumRdmaSendCtas, int NumRdmaRecvCtas, typename WeightT>
__device__ __forceinline__ void hierReceiveFinal(ncclCoopCta ctaCoop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch) {
  const auto& resources = args.hier;
  const int cta = blockIdx.x;
  auto warp = ncclCoopCoalesced(ctaCoop);
  const int warpIdx = ctaCoop.thread_rank() / warp.size();
  const int numWarps = ctaCoop.size() / warp.size();
  const int lane = warp.thread_rank();
  const int numLsaRecvCtas = gridDim.x - NumRdmaRecvCtas;
  const int numLsaSendCtas = min(resources.numLogicalCtas, args.maxTokensPerRank);
  const auto rail = ncclTeamRail(*args.devComm);
  const auto lsa = ncclTeamLsa(*args.devComm);
  const uint64_t tag = epoch + 1;

  // One source per CTA before assigning a second worker CTA.
  // This spreads singleton copies across SMs instead of packing
  // four independent source copies into one 128-thread CTA.
  const int sourceWorkers = min(numLsaRecvCtas, args.numRanks);
  const int receiver = cta - NumRdmaRecvCtas;
  for (int visit = receiver % sourceWorkers; visit < args.numRanks; visit += sourceWorkers) {
    int srcRank = visit + rail.rank * lsa.nRanks;
    if (srcRank >= args.numRanks) {
      srcRank -= args.numRanks;
    }
    const int numSendCtas = srcRank / lsa.nRanks == rail.rank ? numLsaSendCtas : NumRdmaSendCtas;
    const int tokens = hierWaitDoneCount(warp, args, bank, epoch, srcRank);
    const int ctaWorker = numLsaRecvCtas >= args.numRanks ? receiver / args.numRanks : 0;
    const int ctaWorkers = numLsaRecvCtas >= args.numRanks ? 1 + (numLsaRecvCtas - 1 - visit) / args.numRanks : 1;
    if (tokens <= 1) {
      if (tokens && ctaWorker == 0) {
        hierCopyFinalSingleton<Hidden, K>(ctaCoop, args, bank, tag, srcRank);
      }
      ctaCoop.sync();
      continue;
    }
    const int recvWarpIdx = ctaWorker * numWarps + warpIdx;
    const int numRecvWarps = ctaWorkers * numWarps;
    for (int sendCtaIdx = recvWarpIdx; sendCtaIdx < min(numSendCtas, tokens); sendCtaIdx += numRecvWarps) {
      if (sendCtaIdx != 0 && lane == 0) {
        const uint64_t waitStart = clock64();
        while (!hierSendCtaComplete(args, bank, epoch, srcRank, sendCtaIdx)) {
          hierCheckTimeout(waitStart, args.timeoutCycles);
        }
      }
      warp.sync();
      hierCopyFinalTokens<Hidden, K>(warp, args, bank, tag, srcRank, sendCtaIdx, numSendCtas, tokens);
    }
    // Other warps may still consume this source when the leader
    // enters the next visit's CTA broadcast. Join at that boundary.
    ctaCoop.sync();
  }
}

template <int Hidden, int K, typename Idx, int NumRdmaSendCtas, bool Direct, typename WeightT>
__device__ __forceinline__ void hierStageAndPost(ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch) {
  constexpr int kPayloadVectors = Hidden / ::nccl_ep::kBf16PerInt4;
  const auto& resources = args.hier;
  const int cta = blockIdx.x;
  auto warp = ncclCoopCoalesced(coop);
  const int warpIdx = coop.thread_rank() / warp.size();
  const int numWarps = coop.size() / warp.size();
  // A singleton warp stages first, then stamps delivery metadata itself.
  const int firstCopyWarp = numWarps > 1 ? 1 : 0;
  if (cta < max(1, min(NumRdmaSendCtas, args.numTokens))) {
    const int copyWarpIdx = warpIdx - firstCopyWarp;
    const int copyThread = copyWarpIdx * warp.size() + warp.thread_rank();
    const int numCopyThreads = (numWarps - firstCopyWarp) * warp.size();
    for (int token = cta; token < args.numTokens; token += NumRdmaSendCtas) {
      auto* stage = resources.buffer + hierPayloadBank(args, bank) + args.sendOff + static_cast<size_t>(token) * resources.slotBytes;
      if (warpIdx >= firstCopyWarp) {
        for (int i = copyThread; i < resources.headerBytes / int(sizeof(uint32_t)); i += numCopyThreads) {
          reinterpret_cast<uint32_t*>(stage)[i] = 0;
        }
      }
      coop.sync();
      if (warpIdx >= firstCopyWarp) {
        if (copyWarpIdx == 0) {
          hierMakeHeader<K, Idx>(warp, stage, args, token);
        }
        auto load = [](const int4* ptr) { return *ptr; };
        auto store = [](int4* ptr, const int4& value) { *ptr = value; };
        const int vectorsPerWarp = nccl_ep::ceil_div(kPayloadVectors, numWarps - firstCopyWarp);
        const int firstVector = min(copyWarpIdx * vectorsPerWarp, kPayloadVectors);
        const int copyVectors = min(vectorsPerWarp, kPayloadVectors - firstVector);
        UNROLLED_WARP_COPY(4, warp.thread_rank(), copyVectors,
          reinterpret_cast<int4*>(stage + resources.headerBytes) + firstVector,
          static_cast<const int4*>(args.inData) + size_t(token) * kPayloadVectors + firstVector, load, store);
      }
      coop.sync();
      if (warpIdx == 0) {
        hierSetDeliveryHeader(warp, args, stage, epoch, K);
      }
    }
    if constexpr (Direct) {
      // The final header stamp is written by thread zero after
      // the staging barrier. Hand it to every peer producer too.
      coop.sync();
      hierSendDirect(coop, args, bank, epoch, cta, Hidden, K);
    } else {
      hierSendRelay(coop, args, bank, epoch, cta, Hidden, K);
    }
  }
}

// One lane retains state for one uniquely owned source/send-CTA pair.
struct HierReceiveTask {
  int source;
  int sendCta;
  int numSendCtas;
  int sourceTokens;
  int consumed;
  uint64_t publication;
  bool sourceDirect;
  bool relayDone;
  bool finalDone;
};
constexpr unsigned kHierRelayReady = 1U;
constexpr unsigned kHierFinalReady = 2U;

// Never wait for a particular item: return its currently executable stages.
template <typename WeightT>
__device__ __forceinline__ unsigned hierProbeReceiveTask(
  const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch, ncclGin& net, HierReceiveTask& task) {
  if (task.relayDone && task.finalDone) {
    return 0;
  }
  if (task.sourceTokens < 0) {
    task.sourceTokens = hierReadDoneCount(args, bank, epoch, task.source, &task.sourceDirect);
  }
  if (task.sourceTokens >= 0 && task.sendCta >= max(1, min(task.numSendCtas, task.sourceTokens))) {
    task.relayDone = true;
    task.finalDone = true;
    return 0;
  }
  unsigned ready = 0;
  if (hierSendCtaComplete(args, bank, epoch, task.source, task.sendCta)) {
    task.relayDone = true;
    if (!task.finalDone) {
      ready |= kHierFinalReady;
    }
  }
  if (task.sourceDirect) {
    task.relayDone = true;
  }
  if (!task.relayDone) {
    const unsigned signal = hierRelaySignal(args, epoch, task.source, task.sendCta);
    task.publication = net.readSignal(signal);
    if (uint32_t(task.publication) > task.consumed || task.publication >> kHierCountBits) {
      ready |= kHierRelayReady;
    }
  }
  return ready;
}

// Warp-owned scratch permits independent forwarding while other warps compact output.
template <int Hidden, int K, typename WeightT>
__device__ __forceinline__ void hierReceiveTaskRelay(
  ncclCoopWarp warp, HierForwardScratch<K>& scratch, const dispatch_kernel_args_t<WeightT>& args,
  size_t bank, uint64_t epoch, int source, int sendCta, int consumed, uint64_t publication) {
  const int sendCtas = hierNumRdmaSendCtas(args.hier.numLogicalCtas);
  const int capacity = nccl_ep::ceil_div(args.maxTokensPerRank, sendCtas);
  const int signalIdx = hierRelayTaskIndex(args, source, sendCta);
  if (consumed < uint32_t(publication)) {
    const auto& resources = args.hier;
    const auto* message = resources.buffer + hierPayloadBank(args, bank) + resources.relayOffset +
      (size_t(signalIdx) * capacity + consumed) * resources.slotBytes;
    hierForward<Hidden, K>(warp, scratch, args, bank, epoch + 1, source, message, message + resources.headerBytes);
    return;
  }
  if (warp.thread_rank() == 0) {
    auto net = hierGin(args, sendCta, NCCL_GIN_RESOURCE_SHARING_GPU);
    net.resetSignal(hierRelaySignal(args, epoch, source, sendCta));
  }
  int tokens = args.maxTokensPerRank;
  if ((publication >> kHierCountBits) != kHierDenseTerminal) {
    tokens = int(publication >> kHierCountBits) - int(kHierCountBias);
  }
  uint64_t done = epoch + 1;
  if (sendCta == 0 && tokens < sendCtas) {
    done = hierCountDone(epoch, tokens);
  }
  hierPublishDone(warp, args, bank, done, source, sendCta);
}

template <int Hidden, int K, typename WeightT>
__device__ __forceinline__ void hierReceiveRoundRobin(
  ncclCoopCta coop, const dispatch_kernel_args_t<WeightT>& args, size_t bank, uint64_t epoch) {
  constexpr int kMaxWarps = kHierThreads / ::nccl_ep::kWarpSize;
  __shared__ HierForwardScratch<K> scratch[kMaxWarps];
  auto warp = ncclCoopCoalesced(coop);
  const int lane = warp.thread_rank();
  const int warpIdx = coop.thread_rank() / warp.size();
  const int numWarps = coop.size() / warp.size();
  const int totalWarps = int(gridDim.x) * numWarps;
  const int work = int(blockIdx.x) * numWarps + warpIdx + lane * totalWarps;
  const int source = work / args.hier.numLogicalCtas;
  const int sendCta = work % args.hier.numLogicalCtas;
  const auto lsa = ncclTeamLsa(*args.devComm);
  const bool remote = source < args.numRanks && !ncclTeamRankIsMember(lsa, ncclTeamWorld(*args.devComm), source);
  const int sendCtas = hierNumRdmaSendCtas(args.hier.numLogicalCtas);
  const int numSendCtas = remote ? sendCtas : min(args.hier.numLogicalCtas, args.maxTokensPerRank);
  const bool valid = source < args.numRanks && sendCta < max(1, min(numSendCtas, args.maxTokensPerRank));
  HierReceiveTask task{source, sendCta, numSendCtas, -1, 0, 0, false, !valid || !remote, !valid};
  auto net = hierGin(args, sendCta, NCCL_GIN_RESOURCE_SHARING_GPU);
  int cursor = 0;
  uint64_t waitStart = clock64();
  while (true) {
    const unsigned readiness = hierProbeReceiveTask(args, bank, epoch, net, task);
    const unsigned pending = __ballot_sync(0xffffffff, !task.relayDone || !task.finalDone);
    if (!pending) {
      return;
    }
    const unsigned relay = __ballot_sync(0xffffffff, readiness & kHierRelayReady);
    const unsigned final = __ballot_sync(0xffffffff, readiness & kHierFinalReady);
    const unsigned ready = relay | final;
    if (!ready) {
      if (lane == 0) {
        hierCheckTimeout(waitStart, args.timeoutCycles);
      }
      continue;
    }
    unsigned candidates = ready & (0xffffffffU << cursor);
    if (!candidates) {
      candidates = ready;
    }
    const int owner = __ffs(candidates) - 1;
    const unsigned selected = 1U << owner;
    const bool copyFinal = (final & selected) != 0;
    warp.sync(); // Hand the selected lane's acquire to every payload reader.
    const int selectedSource = ncclCoopBcast(warp, task.source, owner, false);
    const int selectedSendCta = ncclCoopBcast(warp, task.sendCta, owner, false);
    if (copyFinal) {
      const int count = ncclCoopBcast(warp, task.sourceTokens, owner, false);
      const int stride = ncclCoopBcast(warp, task.numSendCtas, owner, false);
      const int limit = count < 0 ? args.maxTokensPerRank : count;
      if (limit == 1) {
        hierCopyFinalSingleton<Hidden, K>(warp, args, bank, epoch + 1, selectedSource);
      } else if (limit > 1) {
        hierCopyFinalTokens<Hidden, K>(warp, args, bank, epoch + 1, selectedSource, selectedSendCta, stride, limit);
      }
      if (lane == owner) {
        task.finalDone = true;
      }
    } else {
      const int consumed = ncclCoopBcast(warp, task.consumed, owner, false);
      const uint64_t publication = ncclCoopBcast(warp, task.publication, owner, false);
      hierReceiveTaskRelay<Hidden, K>(warp, scratch[warpIdx], args, bank, epoch,
                                     selectedSource, selectedSendCta, consumed, publication);
      if (lane == owner) {
        if (consumed < uint32_t(publication)) {
          ++task.consumed;
        } else {
          task.relayDone = true;
        }
      }
    }
    cursor = (owner + 1) % warp.size();
    waitStart = clock64();
  }
}

// RoundRobinReceive is capacity/topology-selected; consider runtime selection from incoming work,
// provided switching preserves unique receive-task ownership.
template <int Hidden, int K, typename Idx, int NumRdmaSendCtas, int NumRdmaRecvCtas, bool RoundRobinReceive, typename WeightT>
__device__ void dispatch_kernel_impl_hier(const dispatch_kernel_args_t<WeightT>& args) {
  const bool directSend = hierUseDirect(args);
  const auto& resources = args.hier;
  auto coop = ncclCoopCta();
  auto warp = ncclCoopCoalesced(coop);
  const int cta = blockIdx.x, tid = coop.thread_rank();
  const int warpIdx = tid / warp.size();
  // Both sender specializations share logical send-CTA indices. Physical
  // CTA ownership and the direct/hierarchical choice remain rank-local.
  const int numLsaSendCtas = min(resources.numLogicalCtas, args.maxTokensPerRank);
  // Group creation validates the common context prefix on the host.
  const uint64_t epoch = selectLowLatencyEpoch(args.epochState, args.phases, cta, tid);
  const uint64_t tag = static_cast<uint64_t>(epoch) + 1;
  const size_t bank = (epoch & 1U) * resources.controlBankStride;
  __shared__ __align__(alignof(DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>))
  uint8_t localHeader[sizeof(DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>) +
                      K * sizeof(DispatchRouter<NCCL_EP_LAYOUT_RANK_MAJOR>)];

  // Warp zero resets only local output metadata. Incoming peer banks
  // retain their tags: peers may already be writing this epoch.
  if (warpIdx == 0) {
    const int worker = cta * warp.size() + warp.thread_rank();
    const int workers = int(gridDim.x) * warp.size();
    for (int i = worker; i < args.numRanks; i += workers) {
      args.outSrcInfo[i] = 0;
      if (args.outRecvRankCounter) {
        args.outRecvRankCounter[i] = 0;
      }
    }
  }
  // Low-token direct sends use disjoint producer CTAs at the two ends
  // of the physical grid: RDMA at the front, local LSA at the back.
  // Local headers are independent of RDMA staging, so no prefix join is needed.
  const bool parallelLocal = directSend && args.numTokens > 0 && args.numTokens < NumRdmaSendCtas &&
                             args.numTokens <= int(gridDim.x) / 2;
  if (parallelLocal && cta >= int(gridDim.x) - args.numTokens) {
    const int token = cta - (int(gridDim.x) - args.numTokens);
    for (int i = tid; i < sizeof(localHeader) / sizeof(uint32_t); i += coop.size()) {
      reinterpret_cast<uint32_t*>(localHeader)[i] = 0;
    }
    coop.sync();
    hierMakeHeader<K, Idx>(coop, localHeader, args, token);
    coop.sync();
    hierForward<Hidden, K>(coop, args, bank, tag, args.currRank, localHeader,
                           static_cast<const uint8_t*>(args.inData) + size_t(token) * Hidden * sizeof(nv_bfloat16));
    hierPublishDone(coop, args, bank, tag, args.currRank, token, args.numTokens);
  }
  // Active RDMA send CTAs stage and post their tokens. Empty CTAs
  // reset outputs and join the common prefix boundary.
  if (directSend) {
    hierStageAndPost<Hidden, K, Idx, NumRdmaSendCtas, true>(coop, args, bank, epoch);
  } else {
    hierStageAndPost<Hidden, K, Idx, NumRdmaSendCtas, false>(coop, args, bank, epoch);
  }
  // All threads clear output indices; incoming traffic targets separate staging banks.
  for (size_t i = size_t(cta) * coop.size() + tid; i < size_t(args.numRanks) * args.maxTokensPerRank * K;
       i += size_t(gridDim.x) * coop.size()) {
    args.outRecvTopkIdx[i] = -1;
  }
  // Join local staging/output initialization before receive work.
  // This is a local grid join, not a wait for remote NIC delivery.
  cooperative_groups::this_grid().sync();

  // The independent local send CTAs already published completion.
  // Other local token counts forward after remote submission.
  if (!parallelLocal) {
    for (int sendCtaIdx = (cta + args.numTokens) % int(gridDim.x);
          sendCtaIdx < (args.numTokens >= NumRdmaSendCtas ? numLsaSendCtas : max(1, min(numLsaSendCtas, args.numTokens)));
          sendCtaIdx += int(gridDim.x)) {
      // RDMA senders can also forward local tokens after posting their puts.
      // Both sender specializations publish the same send-CTA completions and tags.
      for (int token = sendCtaIdx; token < args.numTokens; token += numLsaSendCtas) {
        const uint8_t* header = resources.buffer + hierPayloadBank(args, bank) + args.sendOff + static_cast<size_t>(token) * resources.slotBytes;
        hierForward<Hidden, K>(coop, args, bank, tag, args.currRank, header,
                               static_cast<const uint8_t*>(args.inData) + static_cast<size_t>(token) * Hidden * sizeof(nv_bfloat16));
      }
      hierPublishDone(coop, args, bank, tag, args.currRank, sendCtaIdx, args.numTokens);
    }
  }
  coop.sync();

  bool receiveComplete = false;
  if constexpr (RoundRobinReceive) {
    const bool supported = ncclTeamLsa(*args.devComm).nRanks == hier_receive_policy::kLsaRanks &&
      ncclTeamRail(*args.devComm).nRanks == hier_receive_policy::kRdmaDomains &&
      args.devComm->ginContextCount >= NumRdmaSendCtas &&
      args.numRanks * resources.numLogicalCtas <= int(gridDim.x) * coop.size() &&
      coop.size() <= kHierThreads;
    if (supported) {
      hierReceiveRoundRobin<Hidden, K>(coop, args, bank, epoch);
      receiveComplete = true;
    }
  }
  if (!receiveComplete) {
    if (cta < NumRdmaRecvCtas) {
      hierReceiveRelay<Hidden, K, NumRdmaSendCtas, NumRdmaRecvCtas>(coop, args, bank, epoch);
    } else {
      hierReceiveFinal<Hidden, K, NumRdmaSendCtas, NumRdmaRecvCtas>(coop, args, bank, epoch);
    }
  }
  completeFullLowLatencyEpochHier(coop, args, bank);
}
} // namespace ll
} // namespace nccl_ep
