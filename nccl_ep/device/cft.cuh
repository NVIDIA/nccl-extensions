/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

// CFT (Counted Fabric Transport) and local TMA helpers for LL dispatch.
//
// The fabric wrappers mirror NCCL 2.32.1's device CFT API. They are defined
// locally because the pinned NCCL 2.30.7 headers lack that API. LL dispatch
// uses Cft<ncclCoopThread>; the wider coops are retained for API compatibility.
// The local extensions are a bounded flush and pendingTxCount, which tracks
// submitted 16 B units until the caller drains the completion phase.
//
// Fabric and TMA-load waits return a timeout status that callers must treat
// as fatal (see ll_ep_cft.cuh). Local TMA-store waits have no bounded form.
// Fabric PTX requires Blackwell and CUDA 13.3; call sites must use the same
// NCCL_EP_CFT_DEVICE_SUPPORTED gate. NCCL's NCCL_CFT_ENABLE is undefined at
// the end of its header and cannot be reused here.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/atomic>      // cuda::memory_order
#include "nccl_device.h"    // ncclCoopThread

// Blackwell (sm_100+) + CUDA 13.3+ gate for the fabric put/fault PTX + builtin.
#if defined(__CUDA_ARCH__) && (CUDA_VERSION >= 13030) && (__CUDA_ARCH__ >= 1000)
#define NCCL_EP_CFT_DEVICE_SUPPORTED 1
#else
#define NCCL_EP_CFT_DEVICE_SUPPORTED 0
#endif

namespace nccl_ep {
namespace cft {

// Vocabulary (mirrors nccl_device/core.h, cft.h, impl/cft__types.h).

// Logical endpoint ID (ncclCftLeId); zero is a valid ID.
using LeId = uint32_t;

// Shared tracking mbarrier (ncclCftSmem), aligned for fabric completion reports.
// Phase parity and transaction counts live in the issuing thread's Cft state.
struct Smem {
    alignas(16) uint64_t bar;
};

enum class ProxyType : uint32_t {   // ncclMemProxyType
    Generic = 1u << 0,
    Fabric = 1u << 1,
};

enum class FenceScope : uint32_t {  // ncclMemFenceScope
    Cta = 1u << 0,
    Sys = 1u << 1,
};

// Wait budgets use the caller's clock64() timeout, derived from the group's
// timeout_ns or NCCL_EP_TIMEOUT_MS. Nonzero compile-time overrides support
// timeout testing; zero selects the caller's budget.
#ifndef NCCL_EP_CFT_PUT_WAIT_CYCLES
#define NCCL_EP_CFT_PUT_WAIT_CYCLES 0ull   /* 0 = use the caller's timeoutCycles */
#endif
#ifndef NCCL_EP_CFT_TMA_WAIT_CYCLES
#define NCCL_EP_CFT_TMA_WAIT_CYCLES 0ull   /* 0 = use the caller's timeoutCycles */
#endif
__device__ __forceinline__ unsigned long long cft_put_wait_cycles(unsigned long long callerCycles) {
    return (NCCL_EP_CFT_PUT_WAIT_CYCLES != 0ull) ? (unsigned long long)NCCL_EP_CFT_PUT_WAIT_CYCLES : callerCycles;
}
__device__ __forceinline__ unsigned long long cft_tma_wait_cycles(unsigned long long callerCycles) {
    return (NCCL_EP_CFT_TMA_WAIT_CYCLES != 0ull) ? (unsigned long long)NCCL_EP_CFT_TMA_WAIT_CYCLES : callerCycles;
}

// A zero caller budget still bounds failed polls; it does not mean unlimited.
// ncclEpCreateGroup clamps positive timeouts to at least one GPU clock cycle.

// Generic->shared 32-bit address.
__device__ __forceinline__ unsigned cft_smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

#if NCCL_EP_CFT_DEVICE_SUPPORTED

namespace internal {

// Select the issuing thread using NCCL's coop conventions. LL dispatch uses
// ncclCoopThread; warp and wider coops are retained for API compatibility.
template <typename Coop>
__device__ __forceinline__ bool elected(Coop coop) {
    return coop.thread_rank() == 0;
}
template <>
__device__ __forceinline__ bool elected(ncclCoopThread) {
    return true;
}
template <>
__device__ __forceinline__ bool elected(ncclCoopWarp) {
    int chosen;
    asm volatile(
        "{\n\t.reg .pred p;\n\telect.sync _|p, %1;\n\tselp.b32 %0, 1, 0, p;\n}"
        : "=r"(chosen)
        : "r"(0xFFFFFFFFu));
    return chosen != 0;
}

// Wait for an mbarrier phase, returning 1 on completion or 0 on timeout.
// The timeout starts after the first unsuccessful poll. Fabric tracking uses
// the conditional phase; TMA uses the plain form.
// TODO: reconcile conditional-phase waiting with NCCL's primary-phase wait
// and fabric report handling (see the LL CFT design document).
template <bool kConditional>
__device__ __forceinline__ unsigned tryWaitParityBounded(unsigned mb, unsigned par, unsigned long long budget) {
    long long start = 0;
    bool started = false;
    int done = 0;
    for (;;) {
        if constexpr (kConditional) {
            asm volatile(
                "{ .reg .pred p; mbarrier.try_wait.parity.phase_type::conditional.shared::cta.b64 p, [%1], %2; "
                "selp.u32 %0, 1, 0, p; }"
                : "=r"(done)
                : "r"(mb), "r"(par)
                : "memory");
        } else {
            asm volatile(
                "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                : "=r"(done)
                : "r"(mb), "r"(par)
                : "memory");
        }
        if (done) break;
        if (!started) { start = clock64(); started = true; continue; }
        if ((unsigned long long)(clock64() - start) > budget) break;
    }
    return (unsigned)done;
}

}  // namespace internal

// Mirror ncclMemFence: sync before a release and after an acquire.
// LL dispatch uses Generic -> Fabric at CTA scope to publish staged tiles,
// and Fabric -> Generic at system scope after the receiver's counter wait.
// Restaging a tile requires another release fence before any put reads it.
template <typename Coop>
__device__ __forceinline__ void memFence(
    Coop coop, cuda::memory_order order, ProxyType producer, ProxyType consumer, FenceScope scope) {
    if (order == cuda::memory_order_relaxed) return;
    const bool rel = order == cuda::memory_order_release || order == cuda::memory_order_acq_rel ||
                     order == cuda::memory_order_seq_cst;
    const bool acq = order == cuda::memory_order_acquire || order == cuda::memory_order_acq_rel ||
                     order == cuda::memory_order_seq_cst;
    if (rel) coop.sync();
    if (internal::elected(coop)) {
        if (scope == FenceScope::Cta) {
            if (rel) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        } else if (producer == ProxyType::Generic && consumer == ProxyType::Fabric) {
            if (acq) asm volatile("fence.proxy.fabric::generic.alias.acquire.sys;" ::: "memory");
            if (rel) asm volatile("fence.proxy.fabric::generic.alias.release.sys;" ::: "memory");
        } else if (producer == ProxyType::Fabric && consumer == ProxyType::Fabric) {
            if (acq) asm volatile("fence.proxy.fabric::fabric.alias.acquire.sys;" ::: "memory");
            if (rel) asm volatile("fence.proxy.fabric::fabric.alias.release.sys;" ::: "memory");
        } else if (producer == ProxyType::Fabric && consumer == ProxyType::Generic) {
            // PTX proxy directions are to::from: generic::fabric makes fabric
            // writes visible to generic loads.
            if (acq) asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
            if (rel) asm volatile("fence.proxy.generic::fabric.alias.release.sys;" ::: "memory");
        } else {  // Generic -> Generic
            if (order == cuda::memory_order_acquire) {
                asm volatile("fence.acquire.sys;" ::: "memory");
            } else if (order == cuda::memory_order_release) {
                asm volatile("fence.release.sys;" ::: "memory");
            } else {
                asm volatile("membar.sys;" ::: "memory");
            }
        }
    }
    if (acq) coop.sync();
}

// Free-function form for receivers that do not own a Cft instance.
template <typename OpCoop>
__device__ __forceinline__ ncclResult_t waitCounted(
    OpCoop opCoop, cuda::memory_order order, ProxyType consumer, const uint64_t* counter, size_t expected,
    uint64_t timeoutCycles);

// NCCL-compatible fabric operations with deferred completion tracking.
// Each put adds 16 B units to txCount. submit() arms the barrier with those
// units and accumulates pendingTxCount; flush() supplies the phase's single
// arrival and drains completion reports. A report may arrive before expect_tx
// and temporarily make the transaction count negative, but the phase cannot
// complete until flush() arrives.
//
// Explicit init() lets only the elected thread initialize the shared barrier.
// The CTA exits after the final flush, so no barrier invalidation is needed.
template <typename Coop>
struct Cft {
    Coop coop;
    Smem* smem;               // shared tracking mbarrier (layout::v1)
    uint32_t txCount;         // 16 B units queued since the last submit
    uint32_t phaseParity;     // phase the next flush waits on
    uint32_t pendingTxCount;  // 16 B units submitted since the last flush

    // Initialize once, then reuse the barrier by parity. Reinitializing it
    // between puts could overwrite an in-flight completion report. The fence
    // publishes initialization to the fabric proxy before the first report.
    template <typename OpCoop>
    __device__ __forceinline__ void init(OpCoop opCoop) {
        if (internal::elected(opCoop)) {
            asm volatile("mbarrier.init.layout::v1.shared::cta.b64 [%0], 1;" ::"r"(cft_smem_u32(&smem->bar))
                         : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        opCoop.sync();
    }

    // Uncounted unicast put, retained for NCCL API compatibility.
    // LL dispatch uses putCounted so receivers can wait on a byte counter.
    template <typename OpCoop>
    __device__ __forceinline__ void put(OpCoop opCoop, LeId leId, size_t leOffset, const void* smemSource,
                                        uint32_t bytes) {
        opCoop.sync();
        if (internal::elected(opCoop)) {
            asm volatile(
                "fabric.try_put.async.shared::cta"
                ".mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
                "[%0, %1], [%2], %3, [%4];" ::"r"(leId),
                "l"(static_cast<unsigned long long>(leOffset)),
                "r"(cft_smem_u32(smemSource)),
                "r"(bytes),
                "r"(cft_smem_u32(&smem->bar))
                : "memory");
            txCount += bytes / 16;
        }
    }

    // Counted unicast put. The data and byte counter belong to the same
    // endpoint; hardware updates the counter as bytes arrive. The caller
    // owns the source fence, submit, barrier arrival, and completion wait.
    template <typename OpCoop>
    __device__ __forceinline__ void putCounted(OpCoop opCoop, LeId leId, size_t leOffset, size_t leCounterOffset,
                                               const void* smemSource, uint32_t bytes) {
        opCoop.sync();
        if (internal::elected(opCoop)) {
            asm volatile(
                "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric."
                "counted::bytes.relaxed.sys.b128 [%0, %1, %2], [%3], %4, [%5];" ::"r"(leId),
                "l"(static_cast<unsigned long long>(leOffset)),
                "l"(static_cast<unsigned long long>(leCounterOffset)),
                "r"(cft_smem_u32(smemSource)),
                "r"(bytes),
                "r"(cft_smem_u32(&smem->bar))
                : "memory");
            txCount += bytes / 16;
        }
    }

    // Submit queued puts and arm their tracking barrier in 16 B units.
    // pendingTxCount accumulates across submits until flush(). Call flushSmem()
    // before reusing a source tile; submit alone is not that API guarantee.
    template <typename OpCoop>
    __device__ __forceinline__ void submit(OpCoop opCoop) {
        if (internal::elected(opCoop)) {
            asm volatile("fabric.submit;" ::: "memory");
            asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(cft_smem_u32(&smem->bar)),
                         "r"(txCount)
                         : "memory");
            pendingTxCount += txCount;
            txCount = 0;
        }
    }

    // Wait for submitted puts to finish reading shared-memory sources.
    // This is the tile-release point, even where submit subsumes the wait.
    template <typename OpCoop>
    __device__ __forceinline__ void flushSmem(OpCoop opCoop) {
        asm volatile("fabric.wait.sync_restrict::reads;" ::: "memory");
        opCoop.sync();
    }

    // Bounded completion drain. Call once per phase, after submitting all
    // puts: a second arrival could complete the phase early. The parity and
    // budget advance even on timeout, which the caller must treat as fatal.
    // Unlike NCCL's flush, this returns completion status, not a report byte.
    template <typename OpCoop>
    __device__ __forceinline__ bool flush(OpCoop opCoop, uint64_t timeoutCycles) {
        const unsigned mb = cft_smem_u32(&smem->bar);
        unsigned done = 1;
        if (internal::elected(opCoop)) {
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(mb) : "memory");
            done = internal::tryWaitParityBounded<true>(mb, phaseParity & 1u, cft_put_wait_cycles(timeoutCycles));
            phaseParity ^= 1u;
            pendingTxCount = 0;
        }
        return done != 0;
    }

    // Member form retained for NCCL API compatibility.
    template <typename OpCoop>
    __device__ __forceinline__ ncclResult_t waitCounted(OpCoop opCoop, cuda::memory_order order, ProxyType consumer,
                                                        const uint64_t* counter, size_t expected,
                                                        uint64_t timeoutCycles) {
        return cft::waitCounted(opCoop, order, consumer, counter, expected, timeoutCycles);
    }
};

// Wait for the hardware byte counter with relaxed loads and a wraparound-safe
// comparison, then fence Fabric -> consumer at system scope. The fence makes
// fabric-written payload visible to generic loads; putting acquire loads in
// the poll would repeatedly invalidate L1. As in NCCL, the fence is issued
// on timeout too. Only the elected thread returns ncclTimeout; LL dispatch
// uses ncclCoopThread.

// Optional iteration limit for timeout testing, in addition to the clock
// budget. Zero disables it. A ready counter succeeds even with a nonzero cap.
#ifndef NCCL_EP_CFT_POLL_MAX_SPINS
#define NCCL_EP_CFT_POLL_MAX_SPINS 0ull   /* 0 = no iteration bound; the clock bounds it */
#endif

template <typename OpCoop>
__device__ __forceinline__ ncclResult_t waitCounted(
    OpCoop opCoop, cuda::memory_order order, ProxyType consumer, const uint64_t* counter, size_t expected,
    uint64_t timeoutCycles) {
    ncclResult_t ret = ncclSuccess;
    if (internal::elected(opCoop)) {
        unsigned long long got = 0;
        long long start = 0;
        bool started = false;
#if NCCL_EP_CFT_POLL_MAX_SPINS != 0
        unsigned long long spins = 0;
#endif
        for (;;) {
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(got) : "l"(counter) : "memory");
            // A signed-distance comparison tolerates counter wraparound.
            if ((got - static_cast<unsigned long long>(expected)) <= (~0ull >> 1)) break;
#if NCCL_EP_CFT_POLL_MAX_SPINS != 0
            if (++spins >= (unsigned long long)NCCL_EP_CFT_POLL_MAX_SPINS) { ret = ncclTimeout; break; }
#endif
            // Start the timeout after the first unsuccessful poll.
            if (!started) { start = clock64(); started = true; continue; }
            if ((unsigned long long)(clock64() - start) > timeoutCycles) { ret = ncclTimeout; break; }
        }
    }
    opCoop.sync();
    memFence(opCoop, nccl::utility::acquireOrderOf(order), ProxyType::Fabric, consumer, FenceScope::Sys);
    return ret;
}

// Local TMA staging for fabric puts, which require a shared-memory source.
// The TMA barrier is separate from the layout::v1 fabric barrier: its
// transaction count is in bytes, while the fabric barrier counts 16 B units.

// One-time init of a plain (non-fabric) mbarrier used for TMA completion.
__device__ __forceinline__ void cft_tma_mbar_init(uint64_t* mbar) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(cft_smem_u32(mbar)) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Arm the barrier for tx bytes and supply its single arrival. Issue once
// per phase, before the copy; a second arrival could complete the phase early.
__device__ __forceinline__ void cft_tma_expect_tx(uint64_t* mbar, unsigned tx) {
    asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(cft_smem_u32(mbar)),
                 "r"(tx)
                 : "memory");
}

// Bulk async global->shared copy of `bytes` (multiple of 16, both addresses
// 16 B-aligned), completing into `mbar` as `bytes` transactions.
__device__ __forceinline__ void cft_tma_g2s(void* dst_smem, const void* src_gmem, unsigned bytes, uint64_t* mbar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     cft_smem_u32(dst_smem)),
                 "l"(src_gmem),
                 "r"(bytes),
                 "r"(cft_smem_u32(mbar))
                 : "memory");
}

// Wait for TMA completion and flip the phase. Return 1 on success, 0 on
// timeout. The caller must trap on timeout: the copy cannot be canceled and
// may still be writing the tile when the CTA would otherwise release it.
__device__ __forceinline__ unsigned cft_tma_wait(uint64_t* mbar, unsigned* phase, unsigned long long timeoutCycles) {
    const unsigned done = internal::tryWaitParityBounded<false>(cft_smem_u32(mbar), *phase & 1u,
                                                                cft_tma_wait_cycles(timeoutCycles));
    *phase ^= 1u;
    return done;
}

// Local TMA stores reuse the staged payload for the self-rank destination;
// fabric loopback puts are unsupported. The source uses the async proxy, so
// the same shared-memory release fence covers both puts and local stores.
// Headers use generic stores; the local receive buffer has a split layout.
//
// Bulk-group completion replaces mbarrier accounting here. Read completion
// releases the tile. After the token loop, wait for write completion and
// fence async -> generic before the grid barrier (or kernel boundary)
// publishes the payload to the receiver.
//
// cp.async.bulk.wait_group has no bounded form. These local-store waits are
// the only waits on the CFT path without a timeout.

// Bulk async shared->global copy of `bytes` (multiple of 16, both addresses
// 16 B-aligned) into the current bulk-group. Pair with cft_bulk_commit.
__device__ __forceinline__ void cft_tma_s2g(void* dst_gmem, const void* src_smem, unsigned bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_gmem),
                 "r"(cft_smem_u32(src_smem)),
                 "r"(bytes)
                 : "memory");
}

// Close the current bulk-group (all cft_tma_s2g since the previous commit).
__device__ __forceinline__ void cft_bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

// Wait until at most N bulk groups still read their shared-memory sources.
// This releases source tiles but does not imply write completion.
template <int N>
__device__ __forceinline__ void cft_bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// Wait for all bulk-group writes to complete and become visible to this thread.
__device__ __forceinline__ void cft_bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

// Order completed async global stores before generic operations on this
// thread. Call after cft_bulk_wait_all; this fence has no acquire/release form.
__device__ __forceinline__ void cft_publish_bulk_stores_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

#endif  // NCCL_EP_CFT_DEVICE_SUPPORTED

}  // namespace cft
}  // namespace nccl_ep
