/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "nccl_ep_cft.h"

#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "common.hpp"
#include "device/jit/nvcc_compiler.hpp"

#if CUDA_VERSION >= 13030
namespace {

template <typename Fn>
bool load_cft_function(const char* symbol, Fn& function) {
    void* address = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    const cudaError_t result = cudaGetDriverEntryPointByVersion(
        symbol, &address, 13030, cudaEnableDefault, &status);
    if (result != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (status != cudaDriverEntryPointSuccess || address == nullptr) return false;
    function = reinterpret_cast<Fn>(address);
    return true;
}

// A newer build toolkit must not make CFT entry points mandatory at library load.
struct CftDriverApi {
    PFN_cuLogicalEndpointIdReserve_v13030 cuLogicalEndpointIdReserve = nullptr;
    PFN_cuLogicalEndpointIdRelease_v13030 cuLogicalEndpointIdRelease = nullptr;
    PFN_cuLogicalEndpointCreate_v13030 cuLogicalEndpointCreate = nullptr;
    PFN_cuLogicalEndpointDestroy_v13030 cuLogicalEndpointDestroy = nullptr;
    PFN_cuLogicalEndpointQuery_v13030 cuLogicalEndpointQuery = nullptr;
    PFN_cuLogicalEndpointGetLimits_v13030 cuLogicalEndpointGetLimits = nullptr;
    PFN_cuLogicalEndpointExport_v13030 cuLogicalEndpointExport = nullptr;
    PFN_cuLogicalEndpointImport_v13030 cuLogicalEndpointImport = nullptr;
    PFN_cuLogicalEndpointBindAddr_v13030 cuLogicalEndpointBindAddr = nullptr;
    PFN_cuLogicalEndpointUnbind_v13030 cuLogicalEndpointUnbind = nullptr;
    bool available = false;

    CftDriverApi() {
        available =
            load_cft_function("cuLogicalEndpointIdReserve", cuLogicalEndpointIdReserve) &&
            load_cft_function("cuLogicalEndpointIdRelease", cuLogicalEndpointIdRelease) &&
            load_cft_function("cuLogicalEndpointCreate", cuLogicalEndpointCreate) &&
            load_cft_function("cuLogicalEndpointDestroy", cuLogicalEndpointDestroy) &&
            load_cft_function("cuLogicalEndpointQuery", cuLogicalEndpointQuery) &&
            load_cft_function("cuLogicalEndpointGetLimits", cuLogicalEndpointGetLimits) &&
            load_cft_function("cuLogicalEndpointExport", cuLogicalEndpointExport) &&
            load_cft_function("cuLogicalEndpointImport", cuLogicalEndpointImport) &&
            load_cft_function("cuLogicalEndpointBindAddr", cuLogicalEndpointBindAddr) &&
            load_cft_function("cuLogicalEndpointUnbind", cuLogicalEndpointUnbind);
    }
};

const CftDriverApi& cft_driver_api() {
    static const CftDriverApi api;
    return api;
}

// Include alignment slack for an output allocation and an offset within it.
bool cft_output_reserve_size(
    size_t tokens,
    size_t max_token_bytes,
    size_t ranks,
    cuuint64_t bind_align,
    size_t recv_buffer_size,
    size_t& reserve_size) {
    if (max_token_bytes == 0 || tokens > SIZE_MAX / max_token_bytes ||
        tokens * max_token_bytes > SIZE_MAX / ranks) {
        return false;
    }
    const size_t bytes = ranks * tokens * max_token_bytes;
    if (bind_align > SIZE_MAX / 2 || bytes > SIZE_MAX - 2 * bind_align) {
        return false;
    }
    reserve_size = nccl_ep::align<size_t>(bytes, bind_align) + bind_align;
    return recv_buffer_size <= SIZE_MAX - reserve_size;
}

}  // namespace
#endif

// Release acquired resources in reverse setup order, including partial setup.
ncclResult_t nccl_ep_cft_destroy(ncclEpLlCftBuffers* lb, int cuda_device_id) {
#if CUDA_VERSION >= 13030
    const auto& api = cft_driver_api();
    if (lb->cumul_byte_expect_dev != nullptr) {
        cudaFree(lb->cumul_byte_expect_dev);
        lb->cumul_byte_expect_dev = nullptr;
    }
    if (lb->peer_le_dev != nullptr) {
        cudaFree(lb->peer_le_dev);
        lb->peer_le_dev = nullptr;
    }
    // Release both the endpoint and its ID reservation; ID 0 is valid.
    for (uint32_t peer_le : lb->imported_le_ids) {
        api.cuLogicalEndpointDestroy(static_cast<CUlogicalEndpointId>(peer_le));
        api.cuLogicalEndpointIdRelease(static_cast<CUlogicalEndpointId>(peer_le), 1);
    }
    lb->imported_le_ids.clear();
    // Release the caller-owned output before destroying its endpoint.
    if (lb->zc_bound) {
        api.cuLogicalEndpointUnbind(
            static_cast<CUlogicalEndpointId>(lb->local_le),
            static_cast<CUdevice>(cuda_device_id),
            lb->zc_bind_off,
            lb->zc_bound_size);
        lb->zc_bound = false;
    }
    lb->zc_bound_size = 0;
    lb->zc_active = false;
    lb->zc_data_off = 0;
    lb->zc_bound_win = ncclWindow_t{};
    lb->zc_bound_data = nullptr;
    for (auto& dim : lb->zc_bound_shape) {
        dim = 0;
    }
    lb->zc_bound_datatype = ncclBfloat16;
    // Each setup step needs its own cleanup, even if a later step failed.
    if (lb->local_le_bound) {
        api.cuLogicalEndpointUnbind(
            static_cast<CUlogicalEndpointId>(lb->local_le),
            static_cast<CUdevice>(cuda_device_id),
            0,
            lb->recv_buffer_size);
        lb->local_le_bound = false;
    }
    if (lb->local_le_valid) {
        api.cuLogicalEndpointDestroy(static_cast<CUlogicalEndpointId>(lb->local_le));
        lb->local_le_valid = false;
    }
    if (lb->local_le_reserved) {
        api.cuLogicalEndpointIdRelease(static_cast<CUlogicalEndpointId>(lb->local_le), 1);
        lb->local_le_reserved = false;
    }
    lb->local_le = 0;
    if (lb->recv_va != 0) {
        cuMemUnmap(static_cast<CUdeviceptr>(lb->recv_va), lb->recv_buffer_size);
        cuMemAddressFree(static_cast<CUdeviceptr>(lb->recv_va), lb->recv_buffer_size);
        lb->recv_va = 0;
    }
    if (lb->recv_mem_handle != 0) {
        cuMemRelease(static_cast<CUmemGenericAllocationHandle>(lb->recv_mem_handle));
        lb->recv_mem_handle = 0;
    }
    lb->recv_base = nullptr;
    lb->recv_buffer_size = 0;
    lb->initialized = false;
    lb->src_rank_region_bytes = 0;
    lb->slot_stride_bound = 0;
    lb->parity_stride = 0;
    lb->cumul_slots_per_parity = 0;
    lb->zc_reserve_bytes = 0;
    lb->zc_bind_off = 0;
    lb->bind_align = 0;
#else
    (void)lb;
    (void)cuda_device_id;
#endif
    return ncclSuccess;
}

#if CUDA_VERSION >= 13030
static bool ll_cft_poll_ready(CUlogicalEndpointId le) {
    const auto& api = cft_driver_api();
    int st = 0;
    for (int i = 0; i < 200; i++) {
        if (api.cuLogicalEndpointQuery(le, 1, &st) == CUDA_SUCCESS && st != 0) return true;
        usleep(10 * 1000);
    }
    return false;
}
#endif

// Allocate receive buffers, build endpoints and agree collectively on success.
// Record local failures until the verdict allgather so every rank participates.
// All ranks must use the same transport; setup failure cannot fall back locally.
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
    cudaStream_t stream) {
    lb->initialized = false;
#if CUDA_VERSION >= 13030
    const auto& api = cft_driver_api();
    auto cu_ok = [&](CUresult r, const char* what) -> bool {
        if (r != CUDA_SUCCESS) {
            if (verbose) {
                const char* s = nullptr;
                cuGetErrorString(r, &s);
                fprintf(stderr, "[nccl_ep][ll_cft] rank %d: %s -> %s; CFT disabled\n",
                        rank, what, s ? s : "?");
            }
            return false;
        }
        return true;
    };
    // Avoid throwing on local CUDA errors before the collective verdict.
    auto cuda_rt_ok = [&](cudaError_t e, const char* what) -> bool {
        if (e != cudaSuccess) {
            fprintf(stderr, "[nccl_ep][ll_cft] rank %d: %s failed: %s; CFT disabled\n",
                    rank, what, cudaGetErrorString(e));
            return false;
        }
        return true;
    };

    CUdevice dev = 0;
    bool local_ok = api.available && cu_ok(cuDeviceGet(&dev, cuda_device_id), "cuDeviceGet");

    // Query the endpoint's binding requirements.
    CUlogicalEndpointProp prop{};
    cuuint64_t bindAlign = 0, maxSize = 0;
    if (local_ok) {
        prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
        prop.unicast.device = dev;
        prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
        prop.flags = CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS;
        local_ok = cu_ok(api.cuLogicalEndpointGetLimits(&bindAlign, &maxSize, &prop),
                         "cuLogicalEndpointGetLimits");
    }

    // Size each source region from its dispatch bound, not the total receive bound.
    // Top-k and layout are chosen per handle, so reserve the largest supported
    // header and align slots to 256 B. ncclEpDispatch checks the actual message size.
    constexpr int kNumMaxTopK = MAX_NUM_TOPK - 1;
    const size_t maxDispatchHdrBytes =
        nccl_ep::ll::get_dispatch_hdr_sz(kNumMaxTopK, NCCL_EP_LAYOUT_RANK_MAJOR);
    const size_t slotStrideBound =
        nccl_ep::align<size_t>(maxDispatchHdrBytes + max_token_bytes, 256);
    // Retain the legacy header-signal area after the message slots for layout compatibility.
    const size_t srcRankRegionBytes = max_dispatch_tokens_per_rank * slotStrideBound +
        nccl_ep::align<size_t>(max_dispatch_tokens_per_rank * maxDispatchHdrBytes, 256);
    const size_t data_region = static_cast<size_t>(nRanks) * srcRankRegionBytes;
    static constexpr uint64_t kCntrStride = 256;  // 256B-aligned counters
    const size_t numCumulSlots = static_cast<size_t>(nRanks) * num_local_experts;
    const uint64_t byte_cntr_base_off  = nccl_ep::align<uint64_t>(data_region, 256);
    // Separate data and counters for consecutive dispatches.
    const uint64_t parity_stride =
        nccl_ep::align<uint64_t>(byte_cntr_base_off + static_cast<uint64_t>(nRanks) * kCntrStride, 256);
    size_t SZ = 0;
    size_t zc_reserve = 0;
    if (local_ok && bindAlign > 0) {
        const size_t need = static_cast<size_t>(parity_stride) * 2;
        SZ = nccl_ep::align<size_t>(need, bindAlign);
        if (SZ == 0) {
            SZ = bindAlign;
        }
        // Reserve space for an output allocation, including an offset within it.
        if (zero_copy) {
            local_ok = cft_output_reserve_size(
                max_dispatch_tokens_per_rank, max_token_bytes, static_cast<size_t>(nRanks),
                bindAlign, SZ, zc_reserve);
            if (!local_ok) {
                fprintf(stderr, "[nccl_ep][ll_cft] rank %d: zero-copy output size exceeds addressable range\n", rank);
            }
        }
        // Report an oversized buffer before BindAddr; 0 means no advertised cap.
        if (local_ok && maxSize != 0 && SZ + zc_reserve > static_cast<size_t>(maxSize)) {
            fprintf(stderr,
                    "[nccl_ep][ll_cft] rank %d: CFT recv buffer needs %zu B but the logical "
                    "endpoint caps bindable size at %llu B (nRanks=%d, tokens/rank=%zu, "
                    "slot=%zu B); CFT disabled\n",
                    rank, SZ + zc_reserve, static_cast<unsigned long long>(maxSize), nRanks, max_dispatch_tokens_per_rank,
                    slotStrideBound);
            local_ok = false;
        }
        if (local_ok) {
            prop.size = SZ + zc_reserve;
        }
    }

    // Create + map the FABRIC recv buffer.
    CUmemGenericAllocationHandle memh = 0;
    CUdeviceptr rptr = 0;
    if (local_ok) {
        CUmemAllocationProp mp{};
        mp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        mp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        mp.location.id = cuda_device_id;
        mp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        local_ok = cu_ok(cuMemCreate(&memh, SZ, &mp, 0), "cuMemCreate(FABRIC)");
        if (local_ok) lb->recv_mem_handle = static_cast<unsigned long long>(memh);
        local_ok = local_ok && cu_ok(cuMemAddressReserve(&rptr, SZ, 0, 0, 0), "cuMemAddressReserve");
        if (local_ok) {
            lb->recv_va = static_cast<unsigned long long>(rptr);
            lb->recv_buffer_size = SZ;
        }
        local_ok = local_ok && cu_ok(cuMemMap(rptr, SZ, 0, memh, 0), "cuMemMap");
        if (local_ok) {
            CUmemAccessDesc ad{};
            ad.location = mp.location;
            ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            local_ok = cu_ok(cuMemSetAccess(rptr, SZ, &ad, 1), "cuMemSetAccess");
        }
        if (local_ok) lb->recv_base = reinterpret_cast<void*>(rptr);
    }

    // Create the UNICAST endpoint and bind the recv buffer.
    CUlogicalEndpointId le = 0;
    if (local_ok) {
        local_ok = cu_ok(api.cuLogicalEndpointIdReserve(&le, 1), "cuLogicalEndpointIdReserve");
        if (local_ok) {
            // Record ownership before the next operation can fail.
            lb->local_le = static_cast<uint32_t>(le);
            lb->local_le_reserved = true;
            local_ok = cu_ok(api.cuLogicalEndpointCreate(le, &prop), "cuLogicalEndpointCreate");
        }
        if (local_ok) {
            lb->local_le_valid = true;
            // CUDA requires a successful readiness query before endpoint use.
            if (!ll_cft_poll_ready(le)) {
                fprintf(stderr,
                        "[nccl_ep][ll_cft] rank %d: owner endpoint never reported ready (pre-bind "
                        "query); CFT disabled on this rank\n",
                        rank);
                local_ok = false;
            }
        }
        if (local_ok) {
            local_ok = cu_ok(
                api.cuLogicalEndpointBindAddr(le, dev, 0, reinterpret_cast<void*>(rptr), SZ, 0),
                "cuLogicalEndpointBindAddr");
            if (local_ok) lb->local_le_bound = true;
        }
        if (local_ok) {
            local_ok = cuda_rt_ok(cudaMemset(reinterpret_cast<void*>(rptr), 0, SZ),
                                  "cudaMemset(fabric recv buffer)");
        }
        if (local_ok) {
            // Check the bound endpoint before exporting it to peers.
            if (!ll_cft_poll_ready(le)) {
                fprintf(stderr,
                        "[nccl_ep][ll_cft] rank %d: owner endpoint never reported ready (post-bind "
                        "query); CFT disabled on this rank\n",
                        rank);
                local_ok = false;
            }
        }
    }

    // Leave a zero handle on local failure so peers can identify it.
    std::vector<CUlogicalEndpointFabricHandle> handles(nRanks);
    std::memset(handles.data(), 0, handles.size() * sizeof(CUlogicalEndpointFabricHandle));
    if (local_ok)
        local_ok = cu_ok(
            api.cuLogicalEndpointExport(&handles[rank], le, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC),
            "cuLogicalEndpointExport");

    // All ranks participate in the allgather (avoids deadlock on partial failure).
    ncclAllGatherHost(handles.data(), sizeof(CUlogicalEndpointFabricHandle),
                      rank, nRanks, comm, stream);

    // Import all peer endpoints; any failure rejects setup for the whole group.
    std::vector<uint32_t> peer_le(nRanks, 0);
    int failed_peer = -1;  // first peer this rank could not import; -1 = none
    // Count ready endpoints separately from the cleanup list.
    int usable_peers = 0;
    for (int w = 0; local_ok && w < nRanks; w++) {
        if (w == rank) continue;
        // A zero handle indicates that the peer's setup failed.
        bool handle_valid = false;
        for (int b = 0; b < static_cast<int>(sizeof(CUlogicalEndpointFabricHandle)); b++) {
            if (reinterpret_cast<const uint8_t*>(&handles[w])[b] != 0) {
                handle_valid = true;
                break;
            }
        }
        if (!handle_valid) {
            failed_peer = w;
            break;
        }

        CUlogicalEndpointId cand = 0;
        if (api.cuLogicalEndpointIdReserve(&cand, 1) != CUDA_SUCCESS) {
            failed_peer = w;
            break;
        }
        if (api.cuLogicalEndpointImport(cand, &handles[w], CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC)
                != CUDA_SUCCESS) {
            api.cuLogicalEndpointIdRelease(cand, 1);
            if (verbose)
                fprintf(stderr, "[nccl_ep][ll_cft] rank %d: failed to import endpoint from peer=%d\n",
                        rank, w);
            failed_peer = w;
            break;
        }
        // Retain ownership even if the readiness check fails.
        lb->imported_le_ids.push_back(static_cast<uint32_t>(cand));
        if (!ll_cft_poll_ready(cand)) {
            fprintf(stderr,
                    "[nccl_ep][ll_cft] rank %d: imported endpoint from peer=%d never reported ready\n",
                    rank, w);
            failed_peer = w;
            break;
        }
        peer_le[w] = static_cast<uint32_t>(cand);
        usable_peers++;
    }

    // Allocate device state before the verdict and record each allocation for cleanup.
    // ncclAllGatherHost can still throw internally if its own CUDA operations fail.
    bool alloc_ok = true;
    if (local_ok && failed_peer < 0) {
        auto cuda_ok = [&](cudaError_t e, const char* what) -> bool {
            if (e == cudaSuccess) return true;
            fprintf(stderr, "[nccl_ep][ll_cft] rank %d: %s failed: %s\n", rank, what,
                    cudaGetErrorString(e));
            return false;
        };
        // A non-null device array enables CFT in the kernel; its [rank] entry is unused.
        const size_t le_bytes = static_cast<size_t>(nRanks) * sizeof(uint32_t);
        uint32_t* dev_le = nullptr;
        alloc_ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dev_le), le_bytes),
                           "cudaMalloc(peer LE ids)");
        if (alloc_ok) {
            lb->peer_le_dev = dev_le;
            alloc_ok = cuda_ok(cudaMemcpy(dev_le, peer_le.data(), le_bytes, cudaMemcpyHostToDevice),
                               "cudaMemcpy(peer LE ids)");
        }
        // Private targets per receiver warp group and parity avoid concurrent updates.
        const size_t cumul_bytes = 2 * numCumulSlots * sizeof(uint64_t);
        uint64_t* cumul_byte = nullptr;
        if (alloc_ok) {
            alloc_ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cumul_byte), cumul_bytes),
                               "cudaMalloc(cumulative byte targets)");
            if (alloc_ok) lb->cumul_byte_expect_dev = cumul_byte;
        }
        if (alloc_ok)
            alloc_ok = cuda_ok(cudaMemset(cumul_byte, 0, cumul_bytes),
                               "cudaMemset(cumulative byte targets)");
    }

    // Every rank participates, even after a local failure. Success requires all
    // peer endpoints to be ready, not merely present in the cleanup list.
    struct CftInitStatus {
        uint8_t ok;      // 1 = fully initialised
        uint8_t reason;  // kCftInit*
        int16_t peer;    // offending peer for kCftInitPeer, else -1
    };
    enum : uint8_t { kCftInitOk = 0, kCftInitLocal = 1, kCftInitPeer = 2, kCftInitAlloc = 3 };
    const bool imports_complete = (usable_peers == nRanks - 1);
    std::vector<CftInitStatus> status(nRanks, CftInitStatus{0, kCftInitLocal, -1});
    if (!local_ok) {
        status[rank] = {0, kCftInitLocal, -1};
    } else if (failed_peer >= 0 || !imports_complete) {
        status[rank] = {0, kCftInitPeer, static_cast<int16_t>(failed_peer)};
    } else if (!alloc_ok) {
        status[rank] = {0, kCftInitAlloc, -1};
    } else {
        status[rank] = {1, kCftInitOk, -1};
    }
    ncclAllGatherHost(status.data(), sizeof(CftInitStatus), rank, nRanks, comm, stream);

    std::string failures;
    for (int r = 0; r < nRanks; r++) {
        if (status[r].ok) continue;
        if (!failures.empty()) failures += "; ";
        failures += "rank " + std::to_string(r) + " ";
        if (status[r].reason == kCftInitLocal) {
            failures += "could not build its own fabric endpoint";
        } else if (status[r].reason == kCftInitAlloc) {
            failures += "could not allocate its CFT device state";
        } else if (status[r].peer >= 0) {
            failures += "could not import peer " + std::to_string(status[r].peer) + "'s endpoint";
        } else {
            failures += "did not import an endpoint for every peer";
        }
    }
    if (!failures.empty()) {
        fprintf(stderr,
                "NCCL EP: %s was requested but counted-fabric-transport setup failed: %s. "
                "%s is strict — it selects the LL dispatch WIRE PROTOCOL, so the group cannot "
                "run with some ranks on the fabric path and some on the store path. Unset %s "
                "to use the store path everywhere. (CFT requires Blackwell-class hardware, a "
                "recent driver and a running Fabric Manager; the [ll_cft] lines above carry the "
                "driver-level reason.)\n",
                env_var_name, failures.c_str(), env_var_name, env_var_name);
        nccl_ep_cft_destroy(lb, cuda_device_id);
        return ncclInvalidUsage;
    }

    // Publish the initialized state only after every rank succeeds.
    lb->byte_cntr_base_off  = byte_cntr_base_off;
    lb->parity_stride       = parity_stride;
    lb->src_rank_region_bytes = srcRankRegionBytes;
    lb->slot_stride_bound = slotStrideBound;
    lb->cumul_slots_per_parity = numCumulSlots;
    lb->bind_align = bindAlign;
    lb->zc_reserve_bytes = zc_reserve;
    lb->zc_bind_off = SZ;
    lb->zc_active = false;
    lb->dispatch_parity = 0;
    lb->initialized = true;
    if (verbose)
        fprintf(stderr,
                "[nccl_ep][ll_cft] rank %d: initialized (SZ=%zu B, data=%zu MB/parity, "
                "slot=%zu B, tokens/src=%zu, parity_stride=%llu, zc_reserve=%zu B)\n",
                rank, SZ, data_region >> 20, slotStrideBound, max_dispatch_tokens_per_rank,
                static_cast<unsigned long long>(parity_stride), zc_reserve);
#else
    (void)comm;
    (void)rank;
    (void)nRanks;
    (void)cuda_device_id;
    (void)num_local_experts;
    (void)max_dispatch_tokens_per_rank;
    (void)max_token_bytes;
    (void)zero_copy;
    (void)env_var_name;
    (void)verbose;
    (void)stream;
#endif
    return ncclSuccess;
}

#if CUDA_VERSION >= 13030
namespace {

enum class OutputBindStatus : uint64_t { Ready, MissingWindow, Unaligned, TooLarge, BindFailed, NotReady, NoAllocation };

struct OutputBinding {
    OutputBindStatus status;
    uint64_t offset;
    uint64_t bytes;
};

OutputBindStatus bind_local_output(
    ncclEpLlCftBuffers* lb, int rank, int cuda_device_id, bool have_window,
    void* data, size_t tensor_bytes, OutputBinding* binding) {
    if (!have_window) {
        return OutputBindStatus::MissingWindow;
    }
    if (!lb->initialized || lb->zc_bound) {
        return OutputBindStatus::BindFailed;
    }
    const uint64_t alignment = lb->bind_align;
    if (alignment == 0 || lb->zc_reserve_bytes == 0 || tensor_bytes == 0) {
        return OutputBindStatus::TooLarge;
    }

    // Preserve the tensor's row-zero offset within its containing allocation.
    CUdeviceptr base = 0;
    size_t allocation_bytes = 0;
    if (cuMemGetAddressRange(&base, &allocation_bytes, reinterpret_cast<CUdeviceptr>(data)) != CUDA_SUCCESS) {
        return OutputBindStatus::NoAllocation;
    }
    binding->offset = reinterpret_cast<CUdeviceptr>(data) - base;
    if (base % alignment != 0 || binding->offset % 16 != 0) {
        return OutputBindStatus::Unaligned;
    }
    if (binding->offset > allocation_bytes || tensor_bytes > allocation_bytes - binding->offset ||
        binding->offset + tensor_bytes > SIZE_MAX - (alignment - 1)) {
        return OutputBindStatus::TooLarge;
    }
    const size_t bind_size = nccl_ep::align<size_t>(binding->offset + tensor_bytes, alignment);
    if (bind_size > lb->zc_reserve_bytes || bind_size > allocation_bytes) {
        return OutputBindStatus::TooLarge;
    }

    const auto& api = cft_driver_api();
    const CUresult result = api.cuLogicalEndpointBindAddr(
        static_cast<CUlogicalEndpointId>(lb->local_le), static_cast<CUdevice>(cuda_device_id),
        lb->zc_bind_off, reinterpret_cast<void*>(base), bind_size, 0);
    if (result != CUDA_SUCCESS) {
        const char* error = nullptr;
        cuGetErrorString(result, &error);
        fprintf(stderr, "[nccl_ep][ll_cft] rank %d: binding zero-copy output failed: %s\n",
                rank, error ? error : "unknown CUDA error");
        return OutputBindStatus::BindFailed;
    }
    lb->zc_bound = true;
    lb->zc_bound_size = bind_size;
    return ll_cft_poll_ready(static_cast<CUlogicalEndpointId>(lb->local_le))
        ? OutputBindStatus::Ready : OutputBindStatus::NotReady;
}

const char* output_binding_failure(const OutputBinding& binding, const OutputBinding& reference) {
    static const char* const reasons[] = {
        "", "requires a window-backed, unquantized rank-major output", "output allocation or offset is misaligned",
        "output exceeds its allocation or reserved endpoint range", "endpoint binding failed",
        "endpoint did not become ready", "output is not backed by a cuMem allocation"};
    if (binding.status != OutputBindStatus::Ready) {
        return reasons[static_cast<uint64_t>(binding.status)];
    }
    if (binding.offset != reference.offset) {
        return "output offset differs from rank 0";
    }
    if (binding.bytes != reference.bytes) {
        return "output size differs from rank 0";
    }
    return nullptr;
}

void rollback_output_binding(ncclEpLlCftBuffers* lb, int rank, int cuda_device_id) {
    if (!lb->zc_bound) {
        return;
    }
    const CUresult result = cft_driver_api().cuLogicalEndpointUnbind(
        static_cast<CUlogicalEndpointId>(lb->local_le), static_cast<CUdevice>(cuda_device_id),
        lb->zc_bind_off, lb->zc_bound_size);
    if (result == CUDA_SUCCESS) {
        lb->zc_bound = false;
        lb->zc_bound_size = 0;
    } else {
        // Retain ownership so group destruction can retry cleanup.
        fprintf(stderr, "[nccl_ep][ll_cft] rank %d: zero-copy output rollback failed (CUDA %d)\n",
                rank, static_cast<int>(result));
    }
}

}  // namespace
#endif

// Failed ranks still join the collective before any direct payload writes.
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
    cudaStream_t stream) {
#if CUDA_VERSION >= 13030
    std::vector<OutputBinding> bindings(nRanks, OutputBinding{});
    auto& mine = bindings[rank];
    mine.bytes = tensor_bytes;
    mine.status = bind_local_output(lb, rank, cuda_device_id, have_window, data, tensor_bytes, &mine);
    ncclAllGatherHost(bindings.data(), sizeof(OutputBinding), rank, nRanks, comm, stream);
    bool all_ready = true;
    for (int r = 0; r < nRanks; ++r) {
        const char* reason = output_binding_failure(bindings[r], bindings[0]);
        if (reason == nullptr) {
            continue;
        }
        all_ready = false;
        if (rank == 0 || verbose) {
            fprintf(stderr, "[nccl_ep][ll_cft] zero-copy output on rank %d: %s\n", r, reason);
        }
    }
    if (!all_ready) {
        rollback_output_binding(lb, rank, cuda_device_id);
        return ncclInvalidUsage;
    }
    lb->zc_active = true;
    lb->zc_data_off = lb->zc_bind_off + mine.offset;
    if (verbose) {
        fprintf(stderr, "[nccl_ep][ll_cft] rank %d: zero-copy ACTIVE (bound=%zu B, output_offset=%llu)\n",
                rank, lb->zc_bound_size, static_cast<unsigned long long>(lb->zc_data_off));
    }
    return ncclSuccess;
#else
    (void)lb;
    (void)comm;
    (void)rank;
    (void)nRanks;
    (void)cuda_device_id;
    (void)have_window;
    (void)data;
    (void)tensor_bytes;
    (void)verbose;
    (void)stream;
    return ncclInvalidUsage;
#endif
}

namespace {

// 13040 -> "13.4"
std::string cuda_version_string(int32_t v) {
    if (v <= 0) return "unknown";
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

std::string attr_value_string(int32_t v) {
    return v < 0 ? std::string("not reported") : std::to_string(v);
}

void append_clause(std::string* out, const std::string& clause) {
    if (!out->empty()) *out += "; ";
    *out += clause;
}

// Parse the release from the JIT compiler's cached --version output; 0 if unavailable.
int32_t jit_nvcc_release() {
    const std::string id = nccl_ep::jit::NvccCompiler{}.compiler_id();
    const size_t p = id.find("release ");
    if (p == std::string::npos) return 0;
    int major = 0, minor = 0;
    if (std::sscanf(id.c_str() + p, "release %d.%d", &major, &minor) != 2) return 0;
    return major * 1000 + minor * 10;
}

#if CUDA_VERSION >= 13030
// CUDA 13.3 headers may lack the name for attribute 155; verify it with newer headers.
constexpr CUdevice_attribute kAttrLeCountedOpsSupported = static_cast<CUdevice_attribute>(155);
#if CUDA_VERSION >= 13040
static_assert(kAttrLeCountedOpsSupported == CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
              "attribute 155 is no longer LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED");
#endif
#endif

}  // namespace

void nccl_ep_cft_platform_query(int cuda_device_id, ncclEpCftPlatformFacts* facts) {
    *facts = ncclEpCftPlatformFacts{};
    facts->cuda_device_id = cuda_device_id;
    facts->toolkit_version = CUDA_VERSION;
    // Versions and compute capability come from calls every toolkit has, so
    // the message is complete even when the library was built without CFT.
    int version = 0;
    if (cudaRuntimeGetVersion(&version) == cudaSuccess) facts->runtime_version = version;
    version = 0;
    if (cuDriverGetVersion(&version) == CUDA_SUCCESS) facts->driver_version = version;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cuda_device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cuda_device_id) == cudaSuccess) {
        facts->cc_major = major;
        facts->cc_minor = minor;
    }
    facts->jit_nvcc_version = jit_nvcc_release();
#if CUDA_VERSION >= 13030
    facts->built_with_cft = 1;
    const auto& api = cft_driver_api();
    facts->driver_api_available = api.available;
    CUdevice dev = 0;
    const CUresult dev_res = cuDeviceGet(&dev, cuda_device_id);
    facts->query_cu_result = static_cast<int32_t>(dev_res);
    if (dev_res != CUDA_SUCCESS) return;
    facts->query_ok = 1;
    auto read_attr = [dev](CUdevice_attribute attr, int32_t* out) {
        int value = 0;
        *out = (cuDeviceGetAttribute(&value, attr, dev) == CUDA_SUCCESS) ? value : -1;
    };
    read_attr(CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED, &facts->le_unicast_supported);
    read_attr(CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED, &facts->le_multicast_supported);
    read_attr(kAttrLeCountedOpsSupported, &facts->le_counted_ops_supported);
    read_attr(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, &facts->fabric_handle_supported);
    if (!api.available) return;
    // Probe the same endpoint properties used by nccl_ep_cft_init.
    CUlogicalEndpointProp prop{};
    prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    prop.unicast.device = dev;
    prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    prop.flags = CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS;
    cuuint64_t bind_alignment = 0, max_size = 0;
    const CUresult lim_res = api.cuLogicalEndpointGetLimits(&bind_alignment, &max_size, &prop);
    facts->limits_cu_result = static_cast<int32_t>(lim_res);
    if (lim_res == CUDA_SUCCESS) {
        facts->limits_ok = 1;
        facts->bind_alignment = static_cast<uint64_t>(bind_alignment);
        facts->max_size = static_cast<uint64_t>(max_size);
    }
#endif
}

ncclEpCftPlatformVerdict nccl_ep_cft_platform_judge(
    const ncclEpCftPlatformFacts& f, const ncclEpCftPlatformFloors& floors) {
    ncclEpCftPlatformVerdict v{};
    if (!f.built_with_cft || f.toolkit_version < floors.min_toolkit) v.failed |= NCCL_EP_CFT_FAIL_TOOLKIT;
    if (f.runtime_version < floors.min_runtime) v.failed |= NCCL_EP_CFT_FAIL_RUNTIME;
    if (f.driver_version < floors.min_driver) v.failed |= NCCL_EP_CFT_FAIL_DRIVER;
    if (f.cc_major < floors.min_cc_major) v.failed |= NCCL_EP_CFT_FAIL_CC;
    // Validate the JIT compiler now rather than failing at the first dispatch.
    if (f.jit_nvcc_version < floors.min_jit_nvcc) v.failed |= NCCL_EP_CFT_FAIL_JIT_NVCC;
    if (f.built_with_cft) {
        if (!f.driver_api_available) v.failed |= NCCL_EP_CFT_FAIL_DRIVER_API;
        if (!f.query_ok) {
            v.failed |= NCCL_EP_CFT_FAIL_QUERY;
        } else {
            if (f.le_unicast_supported != 1) v.failed |= NCCL_EP_CFT_FAIL_LE_UNICAST;
            // GetLimits probes counted operations when the attribute is unavailable.
            // An explicit unsupported attribute or unusable bind alignment still fails.
            if (f.driver_api_available &&
                (!f.limits_ok || f.bind_alignment == 0 || f.le_counted_ops_supported == 0))
                v.failed |= NCCL_EP_CFT_FAIL_COUNTED_OPS;
            if (f.fabric_handle_supported != 1) v.failed |= NCCL_EP_CFT_FAIL_FABRIC_HANDLE;
        }
    }
    const int32_t level = f.toolkit_version < f.driver_version ? f.toolkit_version : f.driver_version;
    if (v.failed == 0) v.support_level = level;
    // Mirror NCCL's ncclGpuCftSupport probe, independent of the test floors.
    if (f.built_with_cft && f.query_ok && f.driver_version >= 13030 && f.le_unicast_supported == 1 &&
        f.le_multicast_supported == 1)
        v.nccl_gpu_cft_support = level;
    return v;
}

std::string nccl_ep_cft_platform_reasons(
    const ncclEpCftPlatformFacts& f,
    const ncclEpCftPlatformVerdict& v,
    const ncclEpCftPlatformFloors& floors) {
    std::string out;
    if (v.failed & NCCL_EP_CFT_FAIL_TOOLKIT) {
        if (!f.built_with_cft)
            append_clause(&out,
                          "built without CFT support (libnccl_ep was compiled against CUDA " +
                              cuda_version_string(f.toolkit_version) + ", CFT needs " +
                              cuda_version_string(floors.min_toolkit) + " or later)");
        else
            append_clause(&out,
                          "build toolkit CUDA " + cuda_version_string(f.toolkit_version) + " below the floor " +
                              cuda_version_string(floors.min_toolkit));
    }
    if (v.failed & NCCL_EP_CFT_FAIL_RUNTIME)
        append_clause(&out,
                      "CUDA runtime " + cuda_version_string(f.runtime_version) + " (" +
                          std::to_string(f.runtime_version) + ") below the floor " +
                          cuda_version_string(floors.min_runtime) + " (" + std::to_string(floors.min_runtime) + ")");
    if (v.failed & NCCL_EP_CFT_FAIL_DRIVER)
        append_clause(&out,
                      "driver API level " + std::to_string(f.driver_version) + " (CUDA " +
                          cuda_version_string(f.driver_version) + ") below the required " +
                          std::to_string(floors.min_driver) +
                          " -- the CFT floor is driver 610.43.02 = driver API 13030 (nvidia-smi shows the driver)");
    if (v.failed & NCCL_EP_CFT_FAIL_DRIVER_API)
        append_clause(&out, "CUDA driver does not provide the logical endpoint APIs required for CFT");
    if (v.failed & NCCL_EP_CFT_FAIL_CC)
        append_clause(&out,
                      "compute capability " + std::to_string(f.cc_major) + "." + std::to_string(f.cc_minor) +
                          " below " + std::to_string(floors.min_cc_major) + ".0 (CFT needs Blackwell or later)");
    if (v.failed & NCCL_EP_CFT_FAIL_QUERY)
        append_clause(&out,
                      "CUDA device " + std::to_string(f.cuda_device_id) + " could not be queried (cuDeviceGet returned CUresult " +
                          std::to_string(f.query_cu_result) + ")");
    if (v.failed & NCCL_EP_CFT_FAIL_LE_UNICAST)
        append_clause(&out,
                      "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED = " +
                          attr_value_string(f.le_unicast_supported) + " (needs 1)");
    if (v.failed & NCCL_EP_CFT_FAIL_COUNTED_OPS) {
        if (!f.limits_ok)
            append_clause(&out,
                          "cuLogicalEndpointGetLimits(UNICAST, FABRIC, COUNTED_OPS) failed with CUresult " +
                              std::to_string(f.limits_cu_result) + " (counted fabric writes unavailable)");
        else if (f.bind_alignment == 0)
            append_clause(&out, "cuLogicalEndpointGetLimits returned bind alignment 0 (the CFT recv buffer cannot be sized)");
        if (f.le_counted_ops_supported == 0)
            append_clause(&out, "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED = 0 (needs 1)");
    }
    if (v.failed & NCCL_EP_CFT_FAIL_FABRIC_HANDLE)
        append_clause(&out,
                      "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = " +
                          attr_value_string(f.fabric_handle_supported) +
                          " (needs 1: the CFT recv buffer is a FABRIC allocation and endpoints are exchanged as fabric handles)");
    if (v.failed & NCCL_EP_CFT_FAIL_JIT_NVCC) {
        if (f.jit_nvcc_version == 0)
            append_clause(&out,
                          "no usable nvcc for the JIT (NCCL_EP_JIT_NVCC, NVCC, $CUDA_HOME/bin/nvcc or PATH); the LL "
                          "kernels carrying the fabric PTX are compiled at runtime and need nvcc " +
                              cuda_version_string(floors.min_jit_nvcc) + " or later");
        else
            append_clause(&out,
                          "JIT nvcc " + cuda_version_string(f.jit_nvcc_version) + " below the floor " +
                              cuda_version_string(floors.min_jit_nvcc) +
                              " (the LL kernels carrying the fabric PTX are compiled at runtime by the nvcc the JIT "
                              "resolves: NCCL_EP_JIT_NVCC, NVCC, $CUDA_HOME/bin/nvcc, PATH)");
    }
    return out;
}

std::string nccl_ep_cft_platform_summary(
    const ncclEpCftPlatformFacts& f, const ncclEpCftPlatformVerdict& v) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "device=%d toolkit=%d built_with_cft=%d driver_api=%d runtime=%d driver=%d cc=%d.%d le_unicast=%d "
                  "le_multicast=%d le_counted_ops=%d fabric_handle=%d limits=%s(CUresult %d) bind_align=%llu "
                  "max_size=%llu jit_nvcc=%d failed=0x%x support_level=%d nccl_gpu_cft_support=%d",
                  f.cuda_device_id, f.toolkit_version, f.built_with_cft, f.driver_api_available,
                  f.runtime_version, f.driver_version,
                  f.cc_major, f.cc_minor, f.le_unicast_supported, f.le_multicast_supported,
                  f.le_counted_ops_supported, f.fabric_handle_supported, f.limits_ok ? "ok" : "failed",
                  f.limits_cu_result, static_cast<unsigned long long>(f.bind_alignment),
                  static_cast<unsigned long long>(f.max_size), f.jit_nvcc_version, v.failed, v.support_level,
                  v.nccl_gpu_cft_support);
    return std::string(buf);
}
