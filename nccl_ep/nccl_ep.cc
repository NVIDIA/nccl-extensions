/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <cstring>
#include <string>
#include <vector>
#include <nccl.h>
#include <nccl_device.h>
#include "nccl_ep.h"
#include "nccl_ep_env.h"
#include "common.hpp"

// HT (High Throughput) includes
#include "device/ht_ep_adapter.cuh"
#include "device/ht_ep_configs.cuh"
#include "device/ll_ep_adapter.cuh"

// NCCLCHECK macro for public API error checking
// Undef any existing definition from internal NCCL headers to avoid using ncclDebugLog
#ifdef NCCLCHECK
#undef NCCLCHECK
#endif
#define NCCLCHECK(cmd) \
    do { \
        ncclResult_t res = cmd; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            return res; \
        } \
    } while (0)

// Forward declarations for HT functions
static ncclResult_t
init_ht_intranode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_ht_intranode(ncclEpGroup_t ep_group);
static ncclResult_t
init_ht_internode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_ht_internode(ncclEpGroup_t ep_group);

// Forward declaration for LL helper (defined alongside ll_init_handle).
static ncclResult_t ll_resize_rdma_buffer(ncclEpGroup_t ep_group, size_t new_size);

// Forward declarations used by mega-buffer sizing (definitions live later in this file).
static bool is_internode_available(ncclEpGroup_t ep_group);

// Define NCCL_CHECK_RESULT macro for NCCL error checking
#ifndef NCCL_CHECK_RESULT
#define NCCL_CHECK_RESULT(cmd) \
    do { \
        ncclResult_t res = cmd; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error %d at %s:%d\n", res, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

// Append-only public-struct ABI. A caller may be older (the V1 prefix) or
// newer (a larger struct); this library validates the known prefix and ignores
// any unknown future tail. Size is the memory-safety boundary.
static ncclResult_t epValidateStruct(
    const void* object,
    unsigned int size,
    unsigned int magic,
    unsigned int min_size,
    unsigned int expected_magic,
    const char* struct_name) {
    if (object == nullptr) {
        fprintf(stderr, "NCCL EP: required %s is NULL\n", struct_name);
        return ncclInvalidArgument;
    }
    if (size < min_size) {
        fprintf(
            stderr,
            "NCCL EP: %s size too small: got %u, expected at least %u\n",
            struct_name,
            size,
            min_size);
        return ncclInvalidArgument;
    }
    if (magic != expected_magic) {
        fprintf(
            stderr,
            "NCCL EP: %s magic mismatch: got 0x%x, expected 0x%x\n",
            struct_name,
            magic,
            expected_magic);
        return ncclInvalidArgument;
    }
    return ncclSuccess;
}

#define EP_VALIDATE_STRUCT(ptr_, base_) \
    do { \
        const auto* ep_struct_ptr_ = (ptr_); \
        const ncclResult_t ep_struct_result_ = epValidateStruct( \
            ep_struct_ptr_, \
            ep_struct_ptr_ != nullptr ? ep_struct_ptr_->size : 0, \
            ep_struct_ptr_ != nullptr ? ep_struct_ptr_->magic : 0, \
            base_ ## _V1_SIZE, \
            NCCL_EP_MAGIC, \
            #ptr_); \
        if (ep_struct_result_ != ncclSuccess) return ep_struct_result_; \
    } while (0)

template <typename T>
static T epDecodeStruct(const T* source, T defaults) {
    // The caller must validate source against the corresponding V1 boundary
    // before decoding it.
    T decoded = defaults;
    memcpy(
        &decoded,
        source,
        std::min<size_t>(source->size, sizeof(decoded)));
    // Internal code always observes the current library's normalized prefix.
    decoded.size = defaults.size;
    decoded.magic = defaults.magic;
    return decoded;
}

// Resolve AUTO -- the public sentinel meaning "library default" -- to the
// concrete numbering the kernels need. AUTO maps to LOCAL today; pinning a
// non-AUTO value passes straight through.
static inline ncclEpExpertIdKind_t resolveRecvTopkIdxKind(ncclEpExpertIdKind_t k) {
    return (k == NCCL_EP_EXPERT_ID_AUTO) ? NCCL_EP_EXPERT_ID_LOCAL : k;
}

// Helper function to convert ncclDataType_t to cudaDataType_t
static cudaDataType_t ncclDataTypeToCudaDataType(ncclDataType_t nccl_type) {
    switch (nccl_type) {
    case ncclFloat16:
        return CUDA_R_16F;
    case ncclFloat32:
        return CUDA_R_32F;
    case ncclFloat64:
        return CUDA_R_64F;
    case ncclBfloat16:
        return CUDA_R_16BF;
    case ncclInt8:
        return CUDA_R_8I;
    case ncclInt32:
        return CUDA_R_32I;
    case ncclInt64:
        return CUDA_R_64I;
    case ncclUint8:
        return CUDA_R_8U;
    case ncclUint32:
        return CUDA_R_32U;
    case ncclUint64:
        return CUDA_R_64U;
    default:
        assert(false && "Unsupported ncclDataType_t for conversion to cudaDataType_t");
        return CUDA_R_16BF; // Default fallback
    }
}

static size_t ncclTypeSize(ncclDataType_t nccl_type) {
    switch (nccl_type) {
    case ncclInt8:
    case ncclUint8:
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
    case ncclFloat4x2:
        return 1;
    case ncclFloat16:
    case ncclBfloat16:
        return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
        return 8;
    default:
        assert(false && "Unsupported ncclDataType_t for size query");
        return 0;
    }
}

// QUANT_FWD accepts byte-preserving raw wire dtypes.
static bool validate_dtype(ncclDataType_t dt) {
    return dt == ncclBfloat16 || dt == ncclFloat16 || dt == ncclFloat32;
}
static bool validate_scales_forward_token_dtype(ncclDataType_t dt) {
    return dt == ncclFloat32 || dt == ncclFloat16 || dt == ncclBfloat16 ||
        dt == ncclFloat8e4m3 || dt == ncclFloat8e5m2 || dt == ncclFloat4x2;
}

static bool validate_scales_forward_scale_dtype(ncclDataType_t dt) {
    return dt == ncclFloat32 || dt == ncclFloat16 || dt == ncclBfloat16 ||
        dt == ncclFloat8e4m3 || dt == ncclFloat8e5m2 || dt == ncclUint8;
}

// Dynamic NDTensor allocation
// Dynamic allocation is used for tensors that are returned by ncclEpTensorAlloc
// and must be released with ncclEpTensorDestroy.

// Internal magic cookie for tensors returned by ncclEpTensorAlloc. Kept out
// of the public header so callers can only spell the STATIC cookie
// (NCCL_EP_TENSOR_MAGIC) via NCCL_EP_TENSOR_INIT.
#define NCCL_EP_TENSOR_ALLOC_STATIC NCCL_EP_TENSOR_MAGIC
#define NCCL_EP_TENSOR_ALLOC_DYNAMIC 0xBEEFBEEF

#define NCCL_EP_TENSOR_INIT_DYNAMIC \
    ((ncclEpTensor_t){.size = (unsigned int)sizeof(ncclEpTensor_t), .magic = NCCL_EP_TENSOR_ALLOC_DYNAMIC})

typedef struct ncclEpTensorInternal_s {
    // Shadow of pub.sizes captured by ncclEpTensorAlloc; lets the library
    // detect callers that overwrote the sizes pointer on a dynamic descriptor.
    size_t* sizes_shadow;

    // Public field exposed to the user.
    ncclEpTensor_t pub;
} ncclEpTensorInternal_t;

// Recover the outer wrapper from a pointer to its embedded public descriptor.
static ncclEpTensorInternal_t* _getInternalTensor(ncclEpTensor_t* tensor) {
    return reinterpret_cast<ncclEpTensorInternal_t*>(
        reinterpret_cast<char*>(tensor) - offsetof(ncclEpTensorInternal_t, pub));
}
static const ncclEpTensorInternal_t* _getInternalTensor(const ncclEpTensor_t* tensor) {
    return reinterpret_cast<const ncclEpTensorInternal_t*>(
        reinterpret_cast<const char*>(tensor) - offsetof(ncclEpTensorInternal_t, pub));
}

// True if the tensor's `magic` cookie marks it as properly initialised by
// either NCCL_EP_TENSOR_INIT (static) or ncclEpTensorAlloc (dynamic).
// For dynamic tensors, also cross-check the public `sizes` pointer against
// the shadow captured at allocation — catches callers that overwrote the
// descriptor's library-owned sizes array.
static inline bool tensorIsInitialised(const ncclEpTensor_t* t) {
    if (t == nullptr) return false;
    if (t->size < NCCL_EP_TENSOR_V1_SIZE) return false;
    if (t->magic == NCCL_EP_TENSOR_ALLOC_STATIC) return true;
    if (t->magic == NCCL_EP_TENSOR_ALLOC_DYNAMIC) {
        return _getInternalTensor(t)->sizes_shadow == t->sizes;
    }
    return false;
}

// True when the tensor is zero-extent in at least one dimension. Empty tensors
// carry no element to read or write, so the dispatch/combine kernels' per-token
// loops degrade to no-ops and the host side has no storage to address. The
// library treats them as a valid no-op input at the API boundary even though
// they legitimately lack a storage binding (PyTorch/NumPy hand out
// data_ptr() == 0 for such buffers).
static inline bool tensorIsEmpty(const ncclEpTensor_t* t) {
    if (t->sizes == nullptr) return false;
    for (unsigned int i = 0; i < t->ndim; ++i) {
        if (t->sizes[i] == 0) return true;
    }
    return false;
}

// True when the tensor advertises a storage binding -- either a device pointer
// or a window handle (offset may be 0, so we only check `win_hdl`). A tensor
// with neither is unusable by the library.
static inline bool tensorHasBinding(const ncclEpTensor_t* t) {
    return t->data != nullptr || t->win_hdl != ncclWindow_t{};
}

// Combined boundary check used by tensor_ptr / tensor_required: the descriptor
// must be initialised AND carry a storage binding. Empty tensors skip the
// binding check because they have no element to address; the kernels handle
// num_tokens == 0 as a no-op on the sender side and still participate in the
// collective on the receiver side.
static inline void tensorAssertValid(const ncclEpTensor_t* t) {
    assert(tensorIsInitialised(t) && "ncclEpTensor_t not initialised — use NCCL_EP_TENSOR_INIT or ncclEpTensorAlloc");
    assert(
        (tensorIsEmpty(t) || tensorHasBinding(t)) &&
        "ncclEpTensor_t has no storage binding — set `.data` or (`.win_hdl` [+ `.win_offset`])");
}

// Validate that a non-null tensor pointer points at an initialised descriptor.
// Returns the pointer unchanged when null (absent / optional field).
// Aborts with a clear message when the caller passed garbage or a zero-filled
// struct — catching the bug at the API boundary before any field is touched.
static inline ncclEpTensor_t* tensor_ptr(ncclEpTensor_t* t) {
    if (t == nullptr) return nullptr;
    tensorAssertValid(t);
    return t;
}
static inline const ncclEpTensor_t* tensor_ptr(const ncclEpTensor_t* t) {
    if (t == nullptr) return nullptr;
    tensorAssertValid(t);
    return t;
}

// Same as tensor_ptr but for tensors the library knows must be present (e.g.,
// inputs->tokens). Aborts when the pointer is null OR the cookie is wrong.
static inline ncclEpTensor_t* tensor_required(ncclEpTensor_t* t) {
    assert(t != nullptr && "required tensor field is NULL");
    tensorAssertValid(t);
    return t;
}
static inline const ncclEpTensor_t* tensor_required(const ncclEpTensor_t* t) {
    assert(t != nullptr && "required tensor field is NULL");
    tensorAssertValid(t);
    return t;
}

struct DispatchRecipeLaunchContext {
    ncclEpAlgorithm_t algorithm;
    ncclEpLayout_t layout;
    int hidden;
    int num_local_experts;
    int num_ranks;
    int max_tokens_per_rank;
    size_t max_token_bytes;
};

// Recipe validation is deliberately centralized. Algorithm branches may rely on
// the selected recipe and never infer it from an optional tensor or dtype.
static ncclResult_t validateDispatchRecipe(
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpDispatchConfig_t* config,
    const DispatchRecipeLaunchContext& launch) {
    const auto recipe = config ? config->quant_recipe : NCCL_EP_DISP_QUANT_NONE;
    const ncclEpTensor_t* tokens = tensor_required(inputs->tokens);
    const ncclEpTensor_t* output_tokens = tensor_ptr(outputs->tokens);
    const ncclEpTensor_t* input_scales = tensor_ptr(inputs->scales);
    const ncclEpTensor_t* output_scales = tensor_ptr(outputs->scales);
    auto fail = [&](const char* message) -> ncclResult_t {
        fprintf(stderr, "NCCL EP warning: dispatch recipe %d: %s\n",
                static_cast<int>(recipe), message);
        return ncclInvalidArgument;
    };
    switch (recipe) {
        case NCCL_EP_DISP_QUANT_NONE:
            if (!validate_dtype(tokens->datatype)) {
                return fail("tokens has unsupported dtype");
            }
            if (output_tokens == nullptr) {
                return fail("outputs->tokens is required");
            }
            if (output_tokens->datatype != tokens->datatype) {
                return fail("outputs->tokens dtype must match inputs->tokens");
            }
            if (input_scales != nullptr) {
                return fail("inputs->scales must be null for NONE");
            }
            if (output_scales != nullptr) {
                return fail("outputs->scales must be null for NONE");
            }
            return ncclSuccess;
        case NCCL_EP_DISP_QUANT_FWD: {
            auto storage_aligned = [](const ncclEpTensor_t* tensor) {
                return (tensor->data == nullptr ||
                        reinterpret_cast<std::uintptr_t>(tensor->data) % sizeof(int4) == 0) &&
                    (tensor->win_hdl == ncclWindow_t{} || tensor->win_offset % sizeof(int4) == 0);
            };
            if (config != nullptr && config->round_scales != 0) {
                return fail("round_scales must be zero when scales are forwarded");
            }
            if (!validate_scales_forward_token_dtype(tokens->datatype)) {
                return fail("tokens must use a supported raw wire dtype");
            }
            if (tokens->ndim != 2) {
                return fail("tokens must be 2D [tokens, hidden]");
            }
            const size_t token_bytes = tokens->sizes[1] * ncclTypeSize(tokens->datatype);
            if (token_bytes == 0 || token_bytes % sizeof(int4) != 0) {
                return fail("QUANT_FWD token rows must be 16-byte aligned for int4 transport");
            }
            if (input_scales == nullptr) {
                return fail("inputs->scales is required");
            }
            if (output_scales == nullptr) {
                return fail("outputs->scales is required");
            }
            if (output_tokens == nullptr) {
                return fail("outputs->tokens is required");
            }
            if (!storage_aligned(tokens) || !storage_aligned(input_scales) ||
                !storage_aligned(output_tokens) || !storage_aligned(output_scales)) {
                return fail("token and scale storage pointers/window offsets must be 16-byte aligned");
            }
            if (input_scales->ndim != 2) {
                return fail("inputs->scales must be 2D [tokens, scales]");
            }
            if (!validate_scales_forward_scale_dtype(input_scales->datatype)) {
                return fail("inputs->scales must use a supported raw scale dtype");
            }
            if (input_scales->sizes[0] != tokens->sizes[0]) {
                return fail("inputs->scales dimension 0 must equal the token count");
            }
            if (input_scales->sizes[1] == 0) {
                return fail("inputs->scales dimension 1 must be non-zero");
            }
            const size_t scale_bytes = input_scales->sizes[1] * ncclTypeSize(input_scales->datatype);
            if (scale_bytes == 0 || scale_bytes % sizeof(int4) != 0) {
                return fail("scale bytes per token must be non-zero and 16-byte aligned");
            }
            if (token_bytes + scale_bytes > launch.max_token_bytes) {
                return fail("token bytes plus scale bytes exceed the group token-byte limit");
            }
            if (launch.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
                size_t expected_rows = 0;
                size_t expected_slots = 0;
                if (launch.layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                    expected_rows = static_cast<size_t>(launch.num_ranks);
                    expected_slots = static_cast<size_t>(launch.max_tokens_per_rank);
                } else {
                    expected_rows = static_cast<size_t>(launch.num_local_experts);
                    expected_slots = static_cast<size_t>(launch.max_tokens_per_rank) * launch.num_ranks;
                }
                if (output_tokens->datatype != tokens->datatype) {
                    return fail("LL outputs->tokens dtype must match inputs->tokens");
                }
                if (output_tokens->ndim != 3) {
                    return fail("LL outputs->tokens must be 3D");
                }
                if (output_tokens->sizes[0] != expected_rows || output_tokens->sizes[1] != expected_slots ||
                    output_tokens->sizes[2] != tokens->sizes[1]) {
                    return fail("LL outputs->tokens dimensions do not match the selected layout");
                }
                if (output_scales->datatype != input_scales->datatype) {
                    return fail("LL outputs->scales dtype must match inputs->scales");
                }
                if (output_scales->ndim != 3) {
                    return fail("LL outputs->scales must be 3D");
                }
                if (output_scales->sizes[0] != expected_rows || output_scales->sizes[1] != expected_slots ||
                    output_scales->sizes[2] != input_scales->sizes[1]) {
                    return fail("LL outputs->scales dimensions do not match the selected layout and input scales");
                }
            } else {
                if (output_tokens->datatype != tokens->datatype || output_tokens->ndim != 2 ||
                    output_tokens->sizes[1] != tokens->sizes[1]) {
                    return fail("HT outputs->tokens must match the input dtype and physical row width");
                }
                if (output_scales->ndim != 2 || output_scales->datatype != input_scales->datatype) {
                    return fail("HT outputs->scales must match the input scale dtype and be 2D");
                }
                if (output_scales->sizes[0] != output_tokens->sizes[0] ||
                    output_scales->sizes[1] != input_scales->sizes[1]) {
                    return fail("HT outputs->scales must match the token-row capacity and input scale count");
                }
            }
            return ncclSuccess;
        }
        case NCCL_EP_DISP_QUANT_DS_FP8E3M4: {
            if (launch.algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
                return fail("DS_FP8E3M4 is supported only in LL mode");
            }
            if (tokens->datatype != ncclBfloat16) {
                return fail("DS_FP8E3M4 requires BF16 input tokens");
            }
            if (output_tokens == nullptr) {
                return fail("DS_FP8E3M4 requires outputs->tokens");
            }
            if (output_tokens->datatype != ncclFloat8e4m3) {
                return fail("DS_FP8E3M4 requires E4M3 output tokens");
            }
            if (tokens->ndim != 2 ||
                tokens->sizes[1] % (4 * nccl_ep::kDsFp8E3M4ElementsPerScale) != 0) {
                return fail("DS_FP8E3M4 tokens must be 2D with hidden divisible by 512");
            }
            if (input_scales != nullptr) {
                return fail("DS_FP8E3M4 does not accept inputs->scales");
            }
            if (output_scales == nullptr) {
                return fail("DS_FP8E3M4 requires outputs->scales");
            }
            const size_t expected_scales =
                static_cast<size_t>(launch.hidden / nccl_ep::kDsFp8E3M4ElementsPerScale);
            size_t expected_rows = 0;
            size_t expected_slots = 0;
            if (launch.layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                expected_rows = static_cast<size_t>(launch.num_ranks);
                expected_slots = static_cast<size_t>(launch.max_tokens_per_rank);
            } else {
                expected_rows = static_cast<size_t>(launch.num_local_experts);
                expected_slots = static_cast<size_t>(launch.max_tokens_per_rank) * launch.num_ranks;
            }
            if (output_scales->ndim != 3 || output_scales->datatype != ncclFloat32 ||
                output_scales->sizes[0] != expected_rows ||
                output_scales->sizes[1] != expected_slots ||
                output_scales->sizes[2] != expected_scales) {
                return fail("DS_FP8E3M4 outputs->scales must be FP32 3D matching the selected layout "
                            "and hidden / 128");
            }
            if (static_cast<size_t>(launch.hidden) + expected_scales * sizeof(float) > launch.max_token_bytes) {
                return fail("DS_FP8E3M4 token bytes plus scale bytes exceed the group token-byte limit");
            }
            return ncclSuccess;
        }
        default:
            return fail("recipe is not implemented");
    }
}

static ncclResult_t validateCombineRecipe(
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    ncclEpAlgorithm_t algorithm,
    unsigned int device_sm,
    bool internode) {
    const auto recipe = config ? config->quant_recipe : NCCL_EP_COMB_QUANT_NONE;
    const ncclEpTensor_t* tokens = tensor_required(inputs->tokens);
    const ncclEpTensor_t* scales = tensor_ptr(inputs->scales);
    // Recipe validation must be able to reject an invalid input contract before
    // later execution validation requires an output descriptor.  This preserves
    // the public invalid-argument behavior for malformed NONE calls.
    const ncclEpTensor_t* output_tokens = tensor_ptr(outputs->tokens);
    auto fail = [&](const char* message) -> ncclResult_t {
        fprintf(stderr, "NCCL EP warning: combine recipe %d: %s\n",
                static_cast<int>(recipe), message);
        return ncclInvalidArgument;
    };
    switch (recipe) {
        case NCCL_EP_COMB_QUANT_NONE:
            if (scales != nullptr) {
                return fail("inputs->scales must be null for NONE");
            }
            return validate_dtype(tokens->datatype)
                ? ncclSuccess : fail("tokens has unsupported dtype");
        case NCCL_EP_COMB_QUANT_NVFP4:
            // LL-only, as the enum documents: the HT combine kernels have no NVFP4
            // combine_recipe_traits specialization, so an HT call would fail JIT
            // compilation. Reject before the HT path is entered.
            if (algorithm != NCCL_EP_ALGO_LOW_LATENCY)
                return fail("NVFP4 combine is supported only in LL mode");
            if (!nccl_ep::host_build_supports_fp4()) {
                fprintf(stderr, "NCCL EP warning: NVFP4 combine requires CUDA 12.9+ with cuda_fp4.h\n");
                return ncclInvalidUsage;
            }
            if (!nccl_ep::host_device_supports_fp4(device_sm)) {
                fprintf(stderr,
                        "NCCL EP warning: NVFP4 combine is unsupported on sm_%u; "
                        "requires an E2M1 FP4 family target\n",
                        device_sm);
                return ncclInvalidUsage;
            }
            if (output_tokens == nullptr || tokens->datatype != ncclBfloat16 || output_tokens->datatype != ncclBfloat16) {
                return fail("NVFP4 requires BF16 input and output tokens");
            }
            if (scales == nullptr || scales->datatype != ncclFloat32 || scales->ndim != 3) {
                return fail("NVFP4 requires FP32 3D inputs->scales global scales");
            }
            return ncclSuccess;
        case NCCL_EP_COMB_QUANT_MXFP8: {
            // HT-only: LL has no MXFP8 CombineRecipeTraits specialization, so an LL call
            // would fail JIT compilation. Reject before the LL path is entered.
            if (algorithm == NCCL_EP_ALGO_LOW_LATENCY)
                return fail("MXFP8 combine is supported only in HT mode");
            // NVLink-only. The packed [FP8 H | E8M0 H/32] row is an intra-node wire format and
            // there is no quantized inter-node transport, so reject a multi-LSA-team group here
            // rather than accounting for an RDMA leg everywhere downstream.
            if (internode)
                return fail("MXFP8 combine is single-LSA-team (NVLink) only; "
                            "this group spans multiple LSA teams");
            if (!nccl_ep::host_build_supports_mxfp8()) {
                fprintf(stderr, "NCCL EP warning: MXFP8 combine requires CUDA 12.8+ (__nv_fp8_e8m0)\n");
                return ncclInvalidUsage;
            }
            // Quantization is internal and fused into local_permute_reduce, so the caller
            // always hands over BF16 rows and never a scale tensor. The layout requirement
            // that follows from that is checked in ncclEpCombine, where the layout is known.
            // Packed row is H + H/32 bytes; 16-byte alignment requires
            // H % NCCL_EP_MXFP8_HIDDEN_ALIGN == 0, which also implies scale row
            // H/32 >= 16 B for TMA when H > 0. Hidden 0 (empty-rank [0, 0]) is a
            // valid no-op (0 is a multiple of the align).
            const ncclEpTensor_t* out = tensor_required(outputs->tokens);
            if (tokens->ndim != 2)
                return fail("MXFP8 combine tokens must be 2D [recv_tokens, hidden]");
            if (tokens->sizes[1] % NCCL_EP_MXFP8_HIDDEN_ALIGN != 0)
                return fail("MXFP8 combine hidden dim must be a multiple of 512 "
                            "(packed row H+H/32 must be 16-byte aligned)");
            if (out->datatype != ncclBfloat16)
                return fail("MXFP8 combine output tokens must be BF16");
            if (inputs->scales != nullptr)
                return fail("MXFP8 combine generates scales internally; inputs->scales must be null");
            if (tokens->datatype != ncclBfloat16)
                return fail("MXFP8 combine tokens must be BF16");
            return ncclSuccess;
        }
        default:
            return fail("recipe is not implemented");
    }
}

// Make a temporary stack copy of `src` for short-lived library-internal use
// (e.g. window-binding scratch). The copy carries the STATIC magic regardless
// of `src`'s magic. The copy shares `src`'s `sizes` pointer; the caller must
// keep `src->sizes` alive for `dst`'s lifetime.
static inline void tensor_temp_copy(ncclEpTensor_t* dst, const ncclEpTensor_t* src) {
    *dst = NCCL_EP_TENSOR_INIT;
    memcpy(dst, src, std::min<size_t>(src->size, sizeof(*dst)));
    dst->size = NCCL_EP_TENSOR_SIZE;
    dst->magic = NCCL_EP_TENSOR_MAGIC;
}

// Make a permanent (library-owned) copy of `src` into `dst`. Like
// tensor_temp_copy plus a heap-allocated `sizes` array deep-copied from
// `src->sizes`. Reuses `dst`'s existing sizes buffer when the shape (ndim)
// is unchanged.
static inline void tensor_permanent_copy(ncclEpTensor_t* dst, const ncclEpTensor_t* src) {
    size_t* sizes_buf = dst->sizes;
    const bool shape_matches = (sizes_buf != nullptr && dst->ndim == src->ndim);
    if (!shape_matches) {
        delete[] sizes_buf;  // no-op on nullptr
        sizes_buf = new size_t[src->ndim];
    }
    for (unsigned int i = 0; i < src->ndim; i++) {
        sizes_buf[i] = src->sizes[i];
    }
    tensor_temp_copy(dst, src);
    dst->sizes = sizes_buf;
}

ncclResult_t ncclEpTensorAlloc(
    ncclEpTensor_t** tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    const size_t* sizes,
    const ncclEpTensorAllocConfig_t* config) {
    if (config != nullptr) {
        EP_VALIDATE_STRUCT(config, NCCL_EP_TENSOR_ALLOC_CONFIG);
    }
    if (tensor == nullptr || sizes == nullptr || ndim == 0) {
        return ncclInvalidArgument;
    }

    ncclEpTensorInternal_t* internal = new ncclEpTensorInternal_t();
    internal->pub = NCCL_EP_TENSOR_INIT_DYNAMIC;
    internal->pub.ndim = ndim;
    internal->pub.datatype = datatype;

    size_t* sizes_copy = new size_t[ndim];
    for (unsigned int i = 0; i < ndim; i++) sizes_copy[i] = sizes[i];
    internal->pub.sizes = sizes_copy;
    internal->sizes_shadow = sizes_copy;

    *tensor = &internal->pub;
    return ncclSuccess;
}

ncclResult_t ncclEpTensorDestroy(ncclEpTensor_t* tensor) {
    if (tensor == nullptr) {
        return ncclSuccess;
    }
    if (!tensorIsInitialised(tensor) || tensor->magic != NCCL_EP_TENSOR_ALLOC_DYNAMIC) {
        return ncclInvalidArgument;
    }
    ncclEpTensorInternal_t* internal = _getInternalTensor(tensor);
    delete[] internal->sizes_shadow;
    delete internal;
    return ncclSuccess;
}

// Allgather on host memory using NCCL (used once for hostname exchange).
// This operates on a single in-place host buffer, unlike batchAllGatherIpcHandles
// which batches multiple IPC handles via a packed device buffer.
// Each rank contributes element_size bytes at offset rank * element_size.
static void
ncclAllGatherHost(void* host_buffer, size_t element_size, int rank, int nRanks, ncclComm_t comm, cudaStream_t stream) {
    const size_t total_size = element_size * nRanks;
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        static_cast<uint8_t*>(host_buffer) + rank * element_size,
        element_size,
        cudaMemcpyHostToDevice,
        stream));
    NCCL_CHECK_RESULT(ncclAllGather(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        d_buffer,
        element_size,
        ncclUint8,
        comm,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(host_buffer, d_buffer, total_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_buffer));
}

// NCCL barrier using AllReduce.
// If workspace is provided, it is used directly (must be at least sizeof(int) device bytes).
// Otherwise a temporary cudaMalloc/cudaFree pair is used as fallback.
static ncclResult_t ncclBarrier(ncclComm_t comm, cudaStream_t stream, void* workspace = nullptr) {
    int* nccl_barrier_var = nullptr;
    bool owns_memory = false;
    if (workspace) {
        nccl_barrier_var = static_cast<int*>(workspace);
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nccl_barrier_var), sizeof(int)));
        owns_memory = true;
    }
    CUDA_CHECK(cudaMemset(nccl_barrier_var, 0, sizeof(int)));
    NCCL_CHECK_RESULT(ncclAllReduce(nccl_barrier_var, nccl_barrier_var, 1, ncclInt, ncclSum, comm, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (owns_memory) {
        CUDA_CHECK(cudaFree(nccl_barrier_var));
    }
    return ncclSuccess;
}

// Opaque struct definitions
struct ncclEpGroup {
    ncclComm_t comm;

    int nRanks;               // Total number of ranks (from ncclCommCount)
    int rank;                 // This rank's ID
    int nNodes;               // Number of nodes

    void* ep_workspace;       // Device workspace for EP operations
    int cuda_device_id;       // CUDA device ID
    int lsa_team_size;        // LSA team size: ncclTeamLsa(comm).nRanks
    int lsa_rank;             // Rank within LSA team: ncclTeamLsa(comm).rank
    int rdma_team_size;       // RDMA ranks
    int rdma_rank;            // RDMA rank
    void* rdma_buffer;
    // Bytes currently backing rdma_buffer. Layouts on live handles store
    // offsets only and are resolved against this base at use time, so the
    // buffer can grow safely while handles are live.
    //
    // Sizing is controlled by config.rdma_buffer_size:
    //   - NCCL_EP_AUTO: the buffer is lazily allocated on the first LL
    //     handle init using the actual (layout, num_topk), and grown later
    //     if a subsequent handle needs more.
    //   - any other positive value: the buffer is allocated at group time
    //     to exactly that size. Handle init rejects (returns an error) any
    //     handle whose layout does not fit; no growth is performed.
    size_t rdma_buffer_size_alloc;
    nccl_ep::LowLatencyEpochState* ll_epoch_state = nullptr;
    ncclEpGroupConfig_t config;         // Stored configuration

    // Environment-variable configuration, populated once at group creation
    // (nccl_ep_env_init) and read by this group's handles via the
    // nccl_ep_env_* accessors.
    ncclEpEnvConfig env;

    // HT cross-LSA-team RDMA stride: max_dispatch_tokens_per_rank rounded up to a whole
    // number of dispatch chunks so a partial final chunk stays within an LSA team's region.
    int ht_aligned_max_tokens = 0;

    // Resolved HT dispatch/combine tokens-per-chunk for this group. RDMA configs
    // default to HT_TOKENS_PER_CHUNK_RDMA_DEFAULT (64); LSA-only configs default to
    // a dispatch-grid-proportional size (NUM_OF_TOKENS_PER_GROUP times the dispatch
    // budget, rounded up to a multiple of 32). Either may be overridden by NCCL_EP_TOKENS_PER_CHUNK.
    int ht_tokens_per_chunk = 0;

    struct {
        // Device communicator (single comm, multiple contexts)
        // HT cross-LSA-team comms use ncclTeamRail on the base communicator
        ncclDevComm_t* dcomms = nullptr;       // Host array of device communicators
        ncclDevComm_t* d_dcomms = nullptr;     // Device array of device communicators
        int num_comms = 0;                     // Number of communicators (always 1)
        int num_dcomms = 0;                    // Number of device comms
        int qps_per_rank = 0;                  // Total QPs (connections) per rank
        int num_ctx_per_comm = 0;              // Number of contexts per communicator

        // GIN memory base pointer and window
        void* gin_base_ptr = nullptr;         // Base pointer for all GIN memory
        ncclWindow_t nccl_window = {};        // Single registered window handle (pointer-sized)
        unsigned signals_base = 0;            // Base signal ID for dispatch
        unsigned combine_signal_offset = 0;   // Signal offset for combine operations
        int num_total_signals = 0;            // Total number of signals

        // Used by kernels to calculate actual addresses for RDMA puts
        size_t combine_red_token_offset = 0;
        size_t combine_g2s_token_offset = 0;
        size_t combine_red_prob_offset = 0;
        size_t combine_g2s_prob_offset = 0;
        size_t token_staging_offset = 0;
        size_t dense_prob_offset = 0;

        // Layout: [NUM_LSA_TEAMS-1][BATCH_SIZE * bytes_per_entry]
        // bytes_per_entry = max_token_bytes (token + optional scales) + prob_size
        size_t gin_send_staging_offset = 0;
        size_t gin_recv_staging_offset = 0;  // Packed receive buffer (token+prob+sf)

        // RDMA sync-guard readiness-flag regions (per direction)
        size_t dispatch_guard_offset = 0;
        size_t combine_guard_offset = 0;

        unsigned signals_tail_base = 0;         // Base signal ID for tail tracking (sender -> receiver)
        int num_max_rdma_chunked_send_tokens = NCCL_EP_HT_DISPATCH_RDMA_BATCH_SIZE;

    } gin_config;

    int num_local_experts;    // Number of local experts (num_experts / comm->nRanks)
    int max_recv_tokens;      // Resolved per-rank IPC slot budget (= config.max_recv_tokens_per_rank).

    // SM-count configuration, all resolved once at ncclEpCreateGroup.
    unsigned int device_sm;
    unsigned int device_sm_count;   // Number of SMs on the device
    int max_dynamic_smem;           // Opt-in dynamic shared-memory limit per block
    int last_ll_combine_warps_per_group; // Test/diagnostic record of the resolved LL launch shape
    // Hardware opt-in dynamic shared memory cap per block (sharedMemPerBlockOptin), cached
    // once so the dispatch path can reject a pull kernel whose per-warp staging would exceed
    // it (see ncclEpDispatch) without re-querying the device on every call.
    // NOTE: consumed by the EM pull-push path only for now; other EM modes do not yet use it.
    size_t device_smem_optin;
    unsigned int dispatch_num_sms; // Resolved dispatch SM budget.
    unsigned int combine_num_sms;  // Resolved combine SM budget.
    unsigned int shuffle_sms; // Resolved SM count for the shuffle kernels (local_dup, local_reduce, push-combine reduce).
    unsigned int
        preprocess_num_sms; // Resolved SM count for the preprocessing scan kernels (NCCL_EP_PREPROCESS_NUM_SMS).

  // HT EM dispatch/combine code path. Resolved once at ncclEpCreateGroup.
    enum class HtEmMode : uint8_t {
        kLocalPermute = 0, // FLAT-dispatch + local_permute kernels
        kLocalDup = 1, // dispatch dedup + local_dup/local_reduce
        kNvlinkDup = 2, // in-kernel Expert-major scatter (sender duplicates over NVLink)
        kPullPush = 3, // NVLink pull-dispatch + push-combine (single LSA team, expert-major only)
    };
    HtEmMode ht_em_mode;

    ncclEpAllocConfig_t alloc;

  // Active-mask buffer for FT support (LL only)
    int* mask_buffer = nullptr; // Device: int[nRanks], null if !enable_mask
    int* async_error_flag = nullptr; // Host-pinned: 0 = ok, 1 = timeout occurred
    uint64_t timeout_cycles = NUM_TIMEOUT_CYCLES; // GPU clock cycles for wait-loop timeout

  // Physical node properties (CUDA device assignment, IPC between co-located GPUs)
    int gpus_per_node; // Physical GPUs per node (nRanks / nNodes)
    int rank_in_node; // Per-node CUDA device ordinal (= cuda_device_id)
    int node_id; // Physical node index (rank / gpus_per_node)

  // NCCL device API
    size_t num_nccl_comms;
    std::vector<ncclComm_t> nccl_comms;
    ncclDevComm_t* nccl_dev_comms;
    ncclWindow_t* nccl_wins;
    int num_dispatch_signals;
    unsigned clean_barrier_signal_base;

  // Cross-rank sync region used by clean_low_latency_buffer.
    void* sync_buffer = nullptr; // device buffer, int[nRanks] aligned
    ncclWindow_t* sync_window = nullptr; // device ptr to the registered window handle

  // HT buffers for intra-LSA communication
    struct {
    // IPC-mapped buffer pointer arrays (fixed-size, indexed by local NVL rank)
    // Host arrays for population and cleanup
        void** dispatch_expert_output_token_buffer_ptrs;
        float** dispatch_expert_output_prob_buffer_ptrs;
        void** dispatch_expert_output_scaling_factor_buffer_ptrs;
        uint16_t** combine_expert_input_token_buffer_ptrs;
        float** combine_expert_input_prob_buffer_ptrs;
        uint8_t** dispatch_push_count_meta_table_ptrs;

    // Local buffers (owned by this rank)
        void* expert_output_token;
        float* expert_output_prob;
        void* expert_output_scaling_factor;
        uint8_t* expert_output_meta_tables;
        uint16_t* expert_input_token;
        float* expert_input_prob;

    // Slot capacity of expert_output_token / expert_input_token.
        size_t token_staging_slots;

    // Sync flags (rank 0 allocates, others IPC-map)
        uint32_t* dispatch_lsa_S2G_flags;
        uint32_t* combine_lsa_S2G_flags;
    // Single device-resident counter block (one alloc, one free) backing
    // the grid-barrier counters and the per-kernel expected-flag counters.
    // The expected counters are initialized at bootstrap and bumped in the
    // dispatch/combine kernel tails, so the bump is captured into any
    // enclosing CUDA graph and replays correctly.
        void* dev_counter_block = nullptr;
        uint32_t* dispatch_grid_barrier_counter = nullptr;
        uint32_t* combine_grid_barrier_counter = nullptr;
        uint64_t* dispatch_expected_gin_flag_val = nullptr;
        uint32_t* dispatch_expected_lsa_flag_val = nullptr;
        uint64_t* combine_expected_gin_flag_val = nullptr;
        uint32_t* combine_expected_lsa_flag_val = nullptr;

    // Group-shared count-mode / EM local-permute scratch. Written-before-read within each
    // dispatch/combine, so group-scoping stays PP-safe. Managed by alloc/free helpers.
        struct CountScratch {
        // Per-expert EM permute write cursors; zeroed and consumed within a single dispatch.
            int32_t* em_permute_cursors = nullptr;
        // FLAT recv-slot weights (EM local-permute only). Sized by kEpCountMaxTopk.
            float* recv_topk_weights_flat = nullptr;
        } count_scratch;

    // RDMA buffers (multi-LSA-team only)
        uint64_t* dispatch_gin_G2S_flags;
        uint16_t* combine_gin_RED_tokens;
        float* combine_gin_RED_prob;
        uint16_t* combine_gin_G2S_tokens;
        float* combine_gin_G2S_prob;
        uint64_t* combine_gin_G2S_flags;

    // Pre-registered dispatch buffers (group-level, allocated during Group Create)
    // These are pre-registered with GIN to avoid ~60ms registration overhead during dispatch
        void* token_staging_buffer; // Pre-registered staging buffer for user tokens
        float* dense_prob_buffer; // Pre-registered buffer for sparse→dense prob conversion

    // Group-scoped routing bitmap; shared by all handles on this group.
        uint8_t* global_routing_map = nullptr;
        size_t global_routing_map_size = 0;
    // Group-scoped uint16 topk routing map (pull dispatch only); order-preserving
    // alternative to the bitmap. Allocated only when pull is enabled.
        uint16_t* global_topk_idx = nullptr;

    // Merged IPC buffer (single cudaMalloc for all IPC-shared buffers)
        void* ipc_mega_buffer = nullptr;
        size_t ipc_mega_buffer_size = 0;
        ncclWindow_t intranode_mega_window = {};
        size_t ipc_dispatch_token_offset = 0;
        size_t ipc_dispatch_prob_offset = 0;
        size_t ipc_dispatch_scaling_factor_offset = 0;  // QUANT_FWD per-block output-scales region in the mega buffer
        size_t ipc_dispatch_meta_tables_offset = 0;   // count-mode receiver EM tables: [LERM meta] + published count rows
        size_t published_offset = 0; // sender-published count rows offset within the tables region
        size_t ipc_combine_token_offset = 0;
        size_t ipc_combine_prob_offset = 0;

        // Merged completion flags
        uint32_t* completion_flags_base = nullptr;
        ncclWindow_t completion_flags_window = {};

        void* host_ptr_block = nullptr; // Single cudaHostAlloc for all pointer arrays

        // Config
        bool initialized;
        bool internode_initialized;
    } ht_buffers;

    // HT eager recv sizing (config.max_recv_tokens_per_rank == NCCL_EP_AUTO):
    // max_recv_tokens holds the derived internal bound; the caller sizes
    // dispatch recv buffers to the actual recv count of the current routing.
    bool eager_mode;

    // Constructor to properly initialize all members
    ncclEpGroup()
        : comm(nullptr), nRanks(0), rank(0), nNodes(0), ep_workspace(nullptr), cuda_device_id(0), lsa_team_size(0),
          lsa_rank(0), rdma_team_size(0), rdma_rank(0), rdma_buffer(nullptr), rdma_buffer_size_alloc(0), config{},
          num_local_experts(0), max_recv_tokens(0), device_sm(0), device_sm_count(0), max_dynamic_smem(0),
          last_ll_combine_warps_per_group(0), device_smem_optin(0), dispatch_num_sms(0), combine_num_sms(0), shuffle_sms(0),
          preprocess_num_sms(0), ht_em_mode(HtEmMode::kLocalPermute), alloc{}, gpus_per_node(0), rank_in_node(0),
          node_id(0), num_nccl_comms(0), nccl_comms{}, nccl_dev_comms(nullptr), nccl_wins(nullptr),
          num_dispatch_signals(0), clean_barrier_signal_base(0), ht_buffers{}, eager_mode(false) {}
};

// The intra-LSA mega buffer is the one allocation sized directly by the resolved
// per-rank recv budget. With max_recv_tokens_per_rank=NCCL_EP_AUTO that budget is
// the theoretical worst case, so a failure here is usually the budget rather than
// a genuinely undersized device.
static void epWarnMegaBufferAllocFailed(ncclEpGroup_t group, size_t bytes) {
    fprintf(
        stderr,
        "NCCL EP: failed to allocate the HT intra-LSA staging buffer (%.2f MiB) for a recv "
        "budget of %u slots per rank\n",
        static_cast<double>(bytes) / (1024.0 * 1024.0),
        group->config.max_recv_tokens_per_rank);
    if (!group->eager_mode) return;
    fprintf(
        stderr,
        "NCCL EP: that budget was derived from max_recv_tokens_per_rank=NCCL_EP_AUTO as "
        "nRanks * max_dispatch_tokens_per_rank * max(num_topk, 1) = %d * %u * %u, the worst "
        "case in which every rank routes every token to this rank. If your routing never "
        "reaches that bound, set max_recv_tokens_per_rank explicitly to a measured peak.\n",
        group->nRanks,
        group->config.max_dispatch_tokens_per_rank,
        group->config.num_topk > 0 ? group->config.num_topk : 1u);
}

// For tensors w/o external window, lazily bind the internal GIN window and offset.
// Tensors created from a user window already carry their own window; resolve
// their local data pointer here once a group/comm is available.
static ncclResult_t
resolveTensorWindowBinding(const ncclEpGroup_t ep_group, ncclEpTensor_t* tensor, uint64_t default_offset) {
    if (tensor == nullptr) {
        return ncclInvalidArgument;
    }

    // Empty tensors have no storage to bind. Leave data == nullptr (and the
    // window unbound) -- consumers must already treat empty tensors as
    // no-ops at the per-element level. Without this guard, the
    // ncclWinGetUserPtr call below would fail on a NULL win_hdl.
    if (tensorIsEmpty(tensor)) {
        return ncclSuccess;
    }

    const bool internode_initialized = ep_group->ht_buffers.internode_initialized;
    if (internode_initialized) {
        if (tensor->win_hdl == ncclWindow_t{}) {
            tensor->win_hdl = ep_group->gin_config.nccl_window;
            tensor->win_offset = default_offset;
        }
    }

    if (tensor->data != nullptr) {
        return ncclSuccess;
    }

    void* base_ptr = nullptr;
    ncclResult_t result = ncclWinGetUserPtr(ep_group->comm, tensor->win_hdl, &base_ptr);
    if (result != ncclSuccess) {
        return result;
    }
    if (base_ptr == nullptr) {
        return ncclInvalidUsage;
    }

    tensor->data = static_cast<void*>(static_cast<char*>(base_ptr) + tensor->win_offset);
    return ncclSuccess;
}

// Const-input overload: decides internally whether the tensor needs mutation
// (win_hdl assignment or data resolution). If so, copies to local_storage,
// resolves there, and sets *out to local_storage. Otherwise sets *out to the
// original tensor unchanged. Call sites always get back a fully resolved pointer
// without managing the copy themselves.
static ncclResult_t resolveTensorWindowBinding(
    const ncclEpGroup_t ep_group,
    const ncclEpTensor_t* tensor,
    ncclEpTensor_t* local_storage,
    uint64_t default_offset,
    const ncclEpTensor_t** out) {
    const bool internode_initialized = ep_group->ht_buffers.internode_initialized;
    const bool needs_win = internode_initialized && tensor->win_hdl == ncclWindow_t{};
    const bool needs_data = tensor->data == nullptr;
    if (!needs_win && !needs_data) {
        *out = tensor;
        return ncclSuccess;
    }
    tensor_temp_copy(local_storage, tensor);
    NCCLCHECK(resolveTensorWindowBinding(ep_group, local_storage, default_offset));
    *out = local_storage;
    return ncclSuccess;
}

// A tensor is on the zero-copy path when its window is user-provided,
// not the group-owned internal GIN window.
static bool tensorUsesExternalWindow(const ncclEpGroup_t ep_group, const ncclEpTensor_t* tensor) {
    // No window yet means a regular tensor - it may be lazily bound to the
    // internal window later, so do not classify it as external.
    if (tensor->win_hdl == ncclWindow_t{}) {
        return false;
    }

    return tensor->win_hdl != ep_group->gin_config.nccl_window;
}

// Build per-LSA-rank pointers for a window-backed tensor so kernels can write
// directly to same-node peer buffers. The local pointer comes from tensor->data;
// peer pointers are resolved from the NCCL window plus the stored offset.
template <typename T>
static ncclResult_t
buildIntranodePtrArray(const ncclEpGroup_t group, const ncclEpTensor_t* tensor, std::vector<T*>& out_ptrs) {
    if (tensor == nullptr || tensor->win_hdl == ncclWindow_t{}) {
        return ncclInvalidUsage;
    }
    ncclEpTensor_t local;
    const ncclEpTensor_t* resolved;
    NCCLCHECK(resolveTensorWindowBinding(group, tensor, &local, 0, &resolved));

    ncclTeam lsa_team = ncclTeamLsa(group->comm);
    out_ptrs.resize(group->lsa_team_size, nullptr);
    out_ptrs[group->lsa_rank] = static_cast<T*>(resolved->data);

    for (int i = 0; i < group->lsa_team_size; i++) {
        if (i == group->lsa_rank) continue;

        int peer_global = ncclTeamRankToWorld(group->comm, lsa_team, i);
        void* peer_ptr = nullptr;
        NCCL_CHECK_RESULT(ncclGetPeerDevicePointer(resolved->win_hdl, resolved->win_offset, peer_global, &peer_ptr));

        out_ptrs[i] = static_cast<T*>(peer_ptr);
    }
    return ncclSuccess;
}

// True when dispatch writes EM staging by em_slot (max_recv_tokens-sized).
// Only kLocalDup/kNvlinkDup with non-zero-copy; reachable via env override since
// auto selection pairs these modes with zero_copy=ON (no staging allocated).
static bool em_staging_indexed_by_em_slot(ncclEpGroup_t group);
static ncclResult_t ht_query_num_recv_tokens(ncclEpHandle_t handle, cudaStream_t stream, unsigned int* num_recv_tokens);

// Allocate group-shared count-mode / EM local-permute scratch buffers.
static void alloc_ht_count_scratch(ncclEpGroup_t ep_group) {
    // +1 trailing int: DROP-mode phantom-row fixup's grid arrival counter (see
    // local_permute_dup_param_t::allow_overflow_drop in ht_ep.cuh).
    CUDA_CHECK(ep_group->alloc.alloc_fn(
        reinterpret_cast<void**>(&ep_group->ht_buffers.count_scratch.em_permute_cursors),
        static_cast<size_t>(ep_group->num_local_experts + 1) * sizeof(int32_t), ep_group->alloc.context));
    // FLAT recv-slot weights only exist for EM local-permute; sized by the fixed cap.
    if (ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalPermute) {
        CUDA_CHECK(ep_group->alloc.alloc_fn(
            reinterpret_cast<void**>(&ep_group->ht_buffers.count_scratch.recv_topk_weights_flat),
            static_cast<size_t>(ep_group->max_recv_tokens) * nccl_ep::ht::kEpCountMaxTopk * sizeof(float),
            ep_group->alloc.context));
    }
}

// Free group-shared count-mode / EM local-permute scratch buffers.
static void free_ht_count_scratch(ncclEpGroup_t ep_group) {
    if (ep_group->ht_buffers.count_scratch.em_permute_cursors) {
        ep_group->alloc.free_fn(ep_group->ht_buffers.count_scratch.em_permute_cursors, ep_group->alloc.context);
        ep_group->ht_buffers.count_scratch.em_permute_cursors = nullptr;
    }
    if (ep_group->ht_buffers.count_scratch.recv_topk_weights_flat) {
        ep_group->alloc.free_fn(ep_group->ht_buffers.count_scratch.recv_topk_weights_flat, ep_group->alloc.context);
        ep_group->ht_buffers.count_scratch.recv_topk_weights_flat = nullptr;
    }
}

// HT Intra-LSA Initialization (adapted for public NCCL APIs)
static ncclResult_t
init_ht_intranode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* in_config, cudaStream_t stream) {
    ncclComm_t comm = ep_group->comm;
    // HT topology uses NCCL team semantics (rail/lsa)
    int lsa_ranks = ep_group->lsa_team_size;
    int lsa_rank = ep_group->lsa_rank;
    ncclTeam lsa_team = ncclTeamLsa(comm);
    size_t max_token_bytes = ep_group->config.max_token_bytes;
    int num_local_experts = ep_group->num_local_experts;
    int max_recv_tokens = ep_group->max_recv_tokens;

    ep_group->ht_buffers.initialized = false;

    // =========================================================================
    // Phase 1: Allocate all buffers upfront
    // =========================================================================

    // Consolidated intra-LSA mega-buffer: single allocation for all 4 shared buffers.
    // Expert-prob buffers are sized by HT inner-domain cardinality (LSA team size).
    auto align_ipc = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };

    // max_recv_tokens is the resolved per-rank slot budget (see ncclEpCreateGroup).
    size_t max_output_slots = static_cast<size_t>(max_recv_tokens);
    // Token staging slots per direction; zero_copy elides both regions.
    // kLocalPermute uses rank-major writes, kLocalDup and kNvlinkDup use em_slot.
    // Eager mode: max_output_slots covers the raw worst case but no per-expert
    // pad slack (alignment is per-handle, unknown here). A dup-mode routing whose
    // padded total exceeds it traps at the scan instead of overrunning staging.
    const size_t flat_slots = static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * ep_group->nRanks;
    size_t token_staging_slots = em_staging_indexed_by_em_slot(ep_group) ? max_output_slots : flat_slots;
    ep_group->ht_buffers.token_staging_slots = token_staging_slots;
    // kPullPush (expert-major only) sizes the dispatch token + prob staging to just per-rank
    // capacity: both back the non-window pull fallback (token rows and forward topk_weights that
    // peers pull from). The combine prob region is unused (presence derived locally).
    const bool pull_push = ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kPullPush;
    const size_t per_rank_tokens = static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank);
    size_t expert_output_token_sz = (pull_push ? per_rank_tokens : token_staging_slots) * max_token_bytes;
    size_t expert_output_prob_sz = pull_push
        ? per_rank_tokens * MAX_NUM_TOPK * sizeof(float)
        : max_output_slots * num_local_experts * lsa_ranks * sizeof(float);
    // Push-combine (kPullPush) writes at a padded per-row stride (anti-camping pad plus a
    // co-located backward prob row), so size its staging to match. Other EM modes index at
    // the natural row_bytes and never touch the pad, so they keep the unpadded size.
    size_t expert_input_token_sz =
        pull_push ? token_staging_slots * nccl_ep::ht::comb_stage_stride_bytes(
                        static_cast<int>(max_token_bytes), /*reserve_prob=*/true)
                  : token_staging_slots * max_token_bytes;
    size_t expert_input_prob_sz = pull_push
        ? 0
        : max_output_slots * num_local_experts * lsa_ranks * sizeof(float);

    // Output scale byte storage, sized for the largest QUANT_FWD row that
    // the group's token-byte budget permits.
    size_t expert_output_scaling_factor_sz = (pull_push ? per_rank_tokens : max_output_slots) * max_token_bytes;

    // zero_copy elides both token regions (windowed tensors required). Under kPullPush the token
    // staging still backs the non-window pull fallback (input windowing is opt-in independently of
    // zero_copy, which only requires the output window), so keep it.
    const bool zero_copy = ep_group->config.zero_copy == NCCL_EP_ZERO_COPY_ON;
    const bool skip_token_staging = zero_copy && !pull_push;
    size_t dispatch_token_aligned = skip_token_staging ? 0 : align_ipc(expert_output_token_sz);
    size_t dispatch_prob_aligned = align_ipc(expert_output_prob_sz);
    size_t dispatch_sf_aligned = align_ipc(expert_output_scaling_factor_sz);
    // Count-mode receiver EM tables: LERM bitmap rows, then one [cnt_rank | cnt_expert] row
    // per sender. Single-LSA-team only; per-token weights are receiver-local and stay out
    // of this IPC region.
    // LERM is a packed bitmap: ceil(num_local_experts/64) u64 words per slot.
    const size_t lerm_words = nccl_ep::ceil_div<size_t>(static_cast<size_t>(num_local_experts), 64);
    // Group half of dispatch_push_count_capable(); when false, the LERM/count-row region below
    // is dead space and skipped.
    const bool group_count_capable =
        !nccl_ep_env_flag_on(ep_group->env.ht_em_ag_scan_mode) &&
        !is_internode_available(ep_group) &&
        ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalPermute;
    // LERM bitmap rows, then one published [cnt_rank | cnt_expert] count row per sender.
    // Both regions are skipped when the group can't take the count path.
    size_t published_offset = 0;
    size_t dispatch_meta_tables_aligned = 0;
    if (group_count_capable) {
        const size_t count_row_ints =
            static_cast<size_t>(ep_group->nRanks) + ep_group->config.num_experts;
        published_offset = align_ipc(max_output_slots * lerm_words * sizeof(uint64_t));
        dispatch_meta_tables_aligned =
            published_offset +
            align_ipc(static_cast<size_t>(ep_group->nRanks) * count_row_ints * sizeof(int32_t));
    }
    ep_group->ht_buffers.published_offset = published_offset;
    size_t combine_token_aligned = skip_token_staging ? 0 : align_ipc(expert_input_token_sz);
    size_t combine_prob_aligned = align_ipc(expert_input_prob_sz);

    struct StagingSegment {
        const char* name;
        size_t size;
    };
    const StagingSegment staging_segments[] = {
        {"dispatch token", dispatch_token_aligned},
        {"dispatch probability", dispatch_prob_aligned},
        {"dispatch scaling factor", dispatch_sf_aligned},
        {"combine token", combine_token_aligned},
        {"combine probability", combine_prob_aligned},
    };
    for (const StagingSegment& segment : staging_segments) {
        if ((segment.size & 255) != 0) {
            std::fprintf(
                stderr,
                "NCCL EP: HT %s staging segment size is not 256-byte aligned (%zu bytes)\n",
                segment.name,
                segment.size);
            return ncclInternalError;
        }
    }

    size_t mega_sz = dispatch_token_aligned + dispatch_prob_aligned + dispatch_sf_aligned +
                     dispatch_meta_tables_aligned + combine_token_aligned + combine_prob_aligned;
    {
        ncclResult_t mega_res = ncclMemAlloc(&ep_group->ht_buffers.ipc_mega_buffer, mega_sz);
        if (mega_res != ncclSuccess) epWarnMegaBufferAllocFailed(ep_group, mega_sz);
        NCCL_CHECK_RESULT(mega_res);
    }
    ep_group->ht_buffers.ipc_mega_buffer_size = mega_sz;

    uint8_t* mega_base = static_cast<uint8_t*>(ep_group->ht_buffers.ipc_mega_buffer);
    if ((reinterpret_cast<uintptr_t>(mega_base) & 255) != 0) {
        std::fprintf(
            stderr,
            "NCCL EP: HT staging mega-buffer is not 256-byte aligned (%p)\n",
            static_cast<void*>(mega_base));
        return ncclInternalError;
    }
    // Chain each region offset off the previous one so the layout stays correct if a
    // segment size changes: offset[n] = offset[n-1] + size[n-1].
    ep_group->ht_buffers.ipc_dispatch_token_offset = 0;
    ep_group->ht_buffers.expert_output_token = skip_token_staging ? nullptr : mega_base;

    ep_group->ht_buffers.ipc_dispatch_prob_offset =
        ep_group->ht_buffers.ipc_dispatch_token_offset + dispatch_token_aligned;
    ep_group->ht_buffers.expert_output_prob =
        reinterpret_cast<float*>(mega_base + ep_group->ht_buffers.ipc_dispatch_prob_offset);

    // QUANT_FWD output-scales region (after token+prob; shifts combine offsets by dispatch_sf_aligned).
    ep_group->ht_buffers.ipc_dispatch_scaling_factor_offset =
        ep_group->ht_buffers.ipc_dispatch_prob_offset + dispatch_prob_aligned;
    ep_group->ht_buffers.expert_output_scaling_factor =
        mega_base + ep_group->ht_buffers.ipc_dispatch_scaling_factor_offset;

    ep_group->ht_buffers.ipc_dispatch_meta_tables_offset =
        ep_group->ht_buffers.ipc_dispatch_scaling_factor_offset + dispatch_sf_aligned;
    ep_group->ht_buffers.expert_output_meta_tables =
        mega_base + ep_group->ht_buffers.ipc_dispatch_meta_tables_offset;
    // One-time zero of the count landing rows (bootstrap only; overwritten per round).
    CUDA_CHECK(cudaMemsetAsync(
        ep_group->ht_buffers.expert_output_meta_tables + published_offset, 0,
        dispatch_meta_tables_aligned - published_offset, stream));

    ep_group->ht_buffers.ipc_combine_token_offset =
        ep_group->ht_buffers.ipc_dispatch_meta_tables_offset + dispatch_meta_tables_aligned;
    ep_group->ht_buffers.expert_input_token =
        skip_token_staging ? nullptr :
                    reinterpret_cast<uint16_t*>(mega_base + ep_group->ht_buffers.ipc_combine_token_offset);

    ep_group->ht_buffers.ipc_combine_prob_offset =
        ep_group->ht_buffers.ipc_combine_token_offset + combine_token_aligned;
    ep_group->ht_buffers.expert_input_prob =
        reinterpret_cast<float*>(mega_base + ep_group->ht_buffers.ipc_combine_prob_offset);

    // Host pointer arrays indexed by HT local rank within LSA team.
    size_t host_block_sz = sizeof(void*) * lsa_ranks + sizeof(float*) * lsa_ranks // dispatch prob
                           + sizeof(void*) * lsa_ranks     // dispatch scales
                           + sizeof(uint8_t*) * lsa_ranks  // dispatch count tables (count mode)
                           + sizeof(uint16_t*) * lsa_ranks + sizeof(float*) * lsa_ranks;
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.host_ptr_block, host_block_sz, cudaHostAllocMapped));

    uint8_t* hptr = static_cast<uint8_t*>(ep_group->ht_buffers.host_ptr_block);
    ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs = reinterpret_cast<void**>(hptr);
    hptr += sizeof(void*) * lsa_ranks;
    ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);
    hptr += sizeof(float*) * lsa_ranks;
    ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs = reinterpret_cast<void**>(hptr);
    hptr += sizeof(void*) * lsa_ranks;
    ep_group->ht_buffers.dispatch_push_count_meta_table_ptrs = reinterpret_cast<uint8_t**>(hptr);
    hptr += sizeof(uint8_t*) * lsa_ranks;
    ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs = reinterpret_cast<uint16_t**>(hptr);
    hptr += sizeof(uint16_t*) * lsa_ranks;
    ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);


    // Merged completion flags: allocate on all ranks as we will window register is collective
    NCCL_CHECK_RESULT(
        ncclMemAlloc(reinterpret_cast<void**>(&ep_group->ht_buffers.completion_flags_base), 2 * lsa_ranks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ht_buffers.completion_flags_base, 0, 2 * lsa_ranks * sizeof(uint32_t), stream));
    ep_group->ht_buffers.dispatch_lsa_S2G_flags = ep_group->ht_buffers.completion_flags_base;
    ep_group->ht_buffers.combine_lsa_S2G_flags = ep_group->ht_buffers.completion_flags_base + lsa_ranks;

    // Per-rank (not IPC-shared) device counter block. Layout (offsets in bytes):
    //   [ 0..8)  dispatch_expected_gin_flag_val  (uint64_t)
    //   [ 8..16) combine_expected_gin_flag_val   (uint64_t)
    //   [16..20) dispatch_expected_lsa_flag_val (uint32_t)
    //   [20..24) combine_expected_lsa_flag_val  (uint32_t)
    //   [24..28) dispatch_grid_barrier_counter (uint32_t)
    //   [28..32) combine_grid_barrier_counter  (uint32_t)
    {
        constexpr size_t kCounterBlockBytes = 32;
        void* block = nullptr;
        CUDA_CHECK(ep_group->alloc.alloc_fn(&block, kCounterBlockBytes, ep_group->alloc.context));
        ep_group->ht_buffers.dev_counter_block = block;
        auto* base = reinterpret_cast<uint8_t*>(block);
        ep_group->ht_buffers.dispatch_expected_gin_flag_val = reinterpret_cast<uint64_t*>(base + 0);
        ep_group->ht_buffers.combine_expected_gin_flag_val = reinterpret_cast<uint64_t*>(base + 8);
        ep_group->ht_buffers.dispatch_expected_lsa_flag_val = reinterpret_cast<uint32_t*>(base + 16);
        ep_group->ht_buffers.combine_expected_lsa_flag_val = reinterpret_cast<uint32_t*>(base + 20);
        ep_group->ht_buffers.dispatch_grid_barrier_counter = reinterpret_cast<uint32_t*>(base + 24);
        ep_group->ht_buffers.combine_grid_barrier_counter = reinterpret_cast<uint32_t*>(base + 28);

        // Initialize the expected counters to the first-invocation value
        // (grid-barrier counters stay at 0). Bootstrap is outside any user
        // CUDA-graph capture, so a one-shot H2D memcpy + stream sync is safe.
        const bool is_single_node = (ep_group->nNodes == 1);
        const uint64_t init_rdma = is_single_node ? 0ull : 1ull;
        const uint32_t init_intra = static_cast<uint32_t>(ep_group->lsa_team_size);
        uint8_t host_init[kCounterBlockBytes] = {0};
        std::memcpy(host_init + 0, &init_rdma, sizeof(init_rdma));
        std::memcpy(host_init + 8, &init_rdma, sizeof(init_rdma));
        std::memcpy(host_init + 16, &init_intra, sizeof(init_intra));
        std::memcpy(host_init + 20, &init_intra, sizeof(init_intra));
        CUDA_CHECK(cudaMemcpyAsync(block, host_init, kCounterBlockBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Group-shared count-mode / EM local-permute scratch (zeroed/written per dispatch).
    alloc_ht_count_scratch(ep_group);

    // =========================================================================
    // Phase 2: Register windows for shared intra-LSA regions
    // Consolidated registration: mega buffer (token+prob+combine) & completion flags
    // =========================================================================
    // Register the mega buffer
    NCCL_CHECK_RESULT(ncclCommWindowRegister(
        comm,
        ep_group->ht_buffers.ipc_mega_buffer,
        mega_sz,
        &ep_group->ht_buffers.intranode_mega_window,
        NCCL_WIN_COLL_SYMMETRIC));

    // Register the completion flags
    NCCL_CHECK_RESULT(ncclCommWindowRegister(
        comm,
        ep_group->ht_buffers.completion_flags_base,
        2 * lsa_ranks * sizeof(uint32_t),
        &ep_group->ht_buffers.completion_flags_window,
        NCCL_WIN_COLL_SYMMETRIC));

    // =========================================================================
    // Phase 3: Resolve LSA-team peer pointers from NCCL windows.
    // Indexed by HT local rank (LSA team rank)
    // =========================================================================

    for (int i = 0; i < lsa_ranks; i++) {
        if (i == lsa_rank) {
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] = ep_group->ht_buffers.expert_output_token;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] = ep_group->ht_buffers.expert_output_prob;
            ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_output_scaling_factor;
            ep_group->ht_buffers.dispatch_push_count_meta_table_ptrs[i] = ep_group->ht_buffers.expert_output_meta_tables;
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] = ep_group->ht_buffers.expert_input_token;
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] = ep_group->ht_buffers.expert_input_prob;
        } else {
            int peer_global = ncclTeamRankToWorld(comm, lsa_team, i);
            void* peer_base = nullptr;
            NCCL_CHECK_RESULT(
                ncclGetPeerDevicePointer(ep_group->ht_buffers.intranode_mega_window, 0, peer_global, &peer_base));
            uint8_t* pb = static_cast<uint8_t*>(peer_base);
            if ((reinterpret_cast<uintptr_t>(pb) & 255) != 0) {
                std::fprintf(
                    stderr,
                    "NCCL EP: HT staging peer pointer for rank %d is not 256-byte aligned (%p)\n",
                    peer_global,
                    peer_base);
                return ncclInternalError;
            }
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] =
                skip_token_staging ? nullptr : pb + ep_group->ht_buffers.ipc_dispatch_token_offset;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_dispatch_prob_offset);
            ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[i] =
                pb + ep_group->ht_buffers.ipc_dispatch_scaling_factor_offset;
            ep_group->ht_buffers.dispatch_push_count_meta_table_ptrs[i] =
                pb + ep_group->ht_buffers.ipc_dispatch_meta_tables_offset;
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] =
                skip_token_staging ? nullptr : reinterpret_cast<uint16_t*>(pb + ep_group->ht_buffers.ipc_combine_token_offset);
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_combine_prob_offset);
        }
    }

    // Two arrays of single-writer epochs, one slot per LSA rank; resolve rank 0 storage.
    if (lsa_rank != 0) {
        int lsa_rank0_global = ncclTeamRankToWorld(comm, lsa_team, 0);
        void* ptr = nullptr;
        NCCL_CHECK_RESULT(
            ncclGetPeerDevicePointer(ep_group->ht_buffers.completion_flags_window, 0, lsa_rank0_global, &ptr));
        ep_group->ht_buffers.dispatch_lsa_S2G_flags = static_cast<uint32_t*>(ptr);
        ep_group->ht_buffers.combine_lsa_S2G_flags = static_cast<uint32_t*>(ptr) + lsa_ranks;
    }

    ep_group->ht_buffers.initialized = true;
    CUDA_CHECK(cudaDeviceSynchronize());

    return ncclSuccess;
}

// HT Intra-LSA Cleanup
static ncclResult_t destroy_ht_intranode(ncclEpGroup_t ep_group) {
    if (!ep_group->ht_buffers.initialized) return ncclSuccess;

    if (ep_group->ht_buffers.intranode_mega_window != ncclWindow_t{}) {
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, ep_group->ht_buffers.intranode_mega_window));
        ep_group->ht_buffers.intranode_mega_window = {};
    }
    if (ep_group->ht_buffers.completion_flags_window != ncclWindow_t{}) {
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, ep_group->ht_buffers.completion_flags_window));
        ep_group->ht_buffers.completion_flags_window = {};
    }

    // Free consolidated intra-LSA mega-buffer (replaces 4 individual cudaFree calls)
    if (ep_group->ht_buffers.ipc_mega_buffer) {
        NCCL_CHECK_RESULT(ncclMemFree(ep_group->ht_buffers.ipc_mega_buffer));
        ep_group->ht_buffers.ipc_mega_buffer = nullptr;
        ep_group->ht_buffers.ipc_mega_buffer_size = 0;
        ep_group->ht_buffers.expert_output_token = nullptr;
        ep_group->ht_buffers.expert_output_prob = nullptr;
        ep_group->ht_buffers.expert_output_scaling_factor = nullptr;
        ep_group->ht_buffers.expert_output_meta_tables = nullptr;
        ep_group->ht_buffers.expert_input_token = nullptr;
        ep_group->ht_buffers.expert_input_prob = nullptr;
    }
    // Free the consolidated counter block (grid barriers + expected counters).
    if (ep_group->ht_buffers.dev_counter_block) {
        ep_group->alloc.free_fn(ep_group->ht_buffers.dev_counter_block, ep_group->alloc.context);
        ep_group->ht_buffers.dev_counter_block = nullptr;
        ep_group->ht_buffers.dispatch_grid_barrier_counter = nullptr;
        ep_group->ht_buffers.combine_grid_barrier_counter = nullptr;
        ep_group->ht_buffers.dispatch_expected_gin_flag_val = nullptr;
        ep_group->ht_buffers.dispatch_expected_lsa_flag_val = nullptr;
        ep_group->ht_buffers.combine_expected_gin_flag_val = nullptr;
        ep_group->ht_buffers.combine_expected_lsa_flag_val = nullptr;
    }

    // Free group-shared count-mode / EM local-permute scratch
    free_ht_count_scratch(ep_group);

    // Free merged completion flags local allocation
    if (ep_group->ht_buffers.completion_flags_base) {
        NCCL_CHECK_RESULT(ncclMemFree(ep_group->ht_buffers.completion_flags_base));
        ep_group->ht_buffers.completion_flags_base = nullptr;
    }
    ep_group->ht_buffers.dispatch_lsa_S2G_flags = nullptr;
    ep_group->ht_buffers.combine_lsa_S2G_flags = nullptr;

    // Free consolidated host pointer block
    if (ep_group->ht_buffers.host_ptr_block) {
        cudaFreeHost(ep_group->ht_buffers.host_ptr_block);
        ep_group->ht_buffers.host_ptr_block = nullptr;
    }

    ep_group->ht_buffers.initialized = false;
    return ncclSuccess;
}

// CUDACHECK_RET macro for CUDA calls in functions returning ncclResult_t
#ifndef CUDACHECK_RET
#define CUDACHECK_RET(cmd) \
    do { \
        cudaError_t err = cmd; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return ncclInternalError; \
        } \
    } while (0)
#endif

// Constants for GIN configuration
static constexpr int NCCL_EP_HT_GIN_MAX_CONTEXTS = 32;
static constexpr int NCCL_EP_HT_GIN_CTXS_PER_COMM = 4;
static constexpr int MAX_BARRIER_SESSIONS = 32;

static ncclResult_t
init_ht_internode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* in_config, cudaStream_t stream) {
    // Initialize using public NCCL APIs
    if (!in_config || !ep_group) {
        fprintf(stderr, "init_ht_internode: null config or ep_group\n");
        return ncclInvalidArgument;
    }

    int rdma_team_size = ep_group->rdma_team_size;
    // HT cross-LSA-team comms use NCCL team semantics: outer domain=rail (cross-LSA-team), inner domain=lsa.
    int lsa_team_size = ep_group->lsa_team_size;
    ep_group->ht_buffers.internode_initialized = false;

    if (rdma_team_size <= 1) {
        // Single HT outer-domain LSA team — no cross-LSA-team RDMA, but the LSA guard needs a minimal devComm.
        ep_group->gin_config.num_dcomms = 1;
        ep_group->gin_config.dcomms = new ncclDevComm_t[1];
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        // Dispatch indexes one LSA barrier session per CTA with blockIdx.x.
        // Combine synchronizes through a single tail CTA at the next session,
        // so its SM budget does not increase the required session count.
        // Keep the compile-time dispatch range available for specialized kernels.
        reqs.lsaBarrierCount =
            std::max<int>(ep_group->dispatch_num_sms, NCCL_EP_HT_DISPATCH_BLOCKS) + 1;
        NCCLCHECK(ncclDevCommCreate(ep_group->comm, &reqs, &ep_group->gin_config.dcomms[0]));
        CUDACHECK_RET(cudaMalloc(
            reinterpret_cast<void**>(&ep_group->gin_config.d_dcomms),
            sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms));
        CUDACHECK_RET(cudaMemcpy(
            ep_group->gin_config.d_dcomms,
            ep_group->gin_config.dcomms,
            sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms,
            cudaMemcpyHostToDevice));
        return ncclSuccess;
    }

    // =========================================================================
    // Phase 1: All local allocations (no collectives)
    // ncclMemAlloc + buffer partitioning moved here from after ncclDevCommCreate
    // to remove them from the collective critical path.
    // =========================================================================

    constexpr size_t GIN_ALIGNMENT = 4096;
    auto align_size = [](size_t sz, size_t alignment) { return (sz + alignment - 1) & ~(alignment - 1); };

    // Chunk-aligned stride for the cross-LSA-team RDMA buffers, resolved once at group
    // creation (ep_group->ht_aligned_max_tokens / ht_tokens_per_chunk); the
    // dispatch/combine kernels must use the same stride and chunk size.
    const int ht_max_tokens = ep_group->ht_aligned_max_tokens;
    const int ht_tokens_per_chunk = ep_group->ht_tokens_per_chunk;
    size_t combine_gin_RED_tokens_sz = align_size(
        static_cast<size_t>(ht_max_tokens) * (rdma_team_size - 1) * ep_group->config.max_token_bytes,
        GIN_ALIGNMENT);
    size_t combine_gin_G2S_tokens_sz = combine_gin_RED_tokens_sz;
    size_t combine_gin_RED_prob_sz = align_size(
        static_cast<size_t>(ht_max_tokens) * (rdma_team_size - 1) * (ep_group->num_local_experts * lsa_team_size) *
            sizeof(float),
        GIN_ALIGNMENT);
    size_t combine_gin_G2S_prob_sz = combine_gin_RED_prob_sz;

    int max_chunks_per_rank = (ht_max_tokens + ht_tokens_per_chunk - 1) / ht_tokens_per_chunk;
    size_t flags_sz =
        align_size(static_cast<size_t>(rdma_team_size - 1) * max_chunks_per_rank * sizeof(uint64_t), GIN_ALIGNMENT);
    // RDMA sync-guard: NUM_LSA_TEAMS uint64 internal-buffer readiness slots per direction.
    size_t guard_sz = align_size(static_cast<size_t>(rdma_team_size) * sizeof(uint64_t), GIN_ALIGNMENT);
    size_t token_staging_sz = align_size(
        static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * ep_group->config.max_token_bytes,
        GIN_ALIGNMENT);
    size_t dense_prob_sz = align_size(
        static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * ep_group->config.num_experts *
            sizeof(float),
        GIN_ALIGNMENT);
    size_t bytes_per_token_entry = ep_group->config.max_token_bytes;
    size_t bytes_per_prob_entry = (ep_group->num_local_experts * lsa_team_size) * sizeof(float);
    size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry;
    size_t rdma_send_staging_sz = align_size(
        static_cast<size_t>(rdma_team_size - 1) * ep_group->config.max_dispatch_tokens_per_rank * bytes_per_entry,
        GIN_ALIGNMENT);
    size_t rdma_recv_packed_sz = align_size(
        static_cast<size_t>(rdma_team_size - 1) * ep_group->config.max_dispatch_tokens_per_rank * bytes_per_entry,
        GIN_ALIGNMENT);

    size_t total_gin_buffer_size = 0;
    total_gin_buffer_size += combine_gin_RED_tokens_sz;
    total_gin_buffer_size += combine_gin_G2S_tokens_sz;
    total_gin_buffer_size += combine_gin_RED_prob_sz;
    total_gin_buffer_size += combine_gin_G2S_prob_sz;
    total_gin_buffer_size += flags_sz * 2;
    total_gin_buffer_size += guard_sz * 2;
    total_gin_buffer_size += token_staging_sz;
    total_gin_buffer_size += dense_prob_sz;
    total_gin_buffer_size += rdma_send_staging_sz;
    total_gin_buffer_size += rdma_recv_packed_sz;

    NCCLCHECK(ncclMemAlloc(&ep_group->gin_config.gin_base_ptr, total_gin_buffer_size));

    // Partition the buffer into individual regions
    uint8_t* ptr = reinterpret_cast<uint8_t*>(ep_group->gin_config.gin_base_ptr);
    size_t offset = 0;

    ep_group->ht_buffers.combine_gin_RED_tokens = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += combine_gin_RED_tokens_sz;

    ep_group->ht_buffers.combine_gin_G2S_tokens = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += combine_gin_G2S_tokens_sz;

    ep_group->ht_buffers.combine_gin_RED_prob = reinterpret_cast<float*>(ptr + offset);
    offset += combine_gin_RED_prob_sz;

    ep_group->ht_buffers.combine_gin_G2S_prob = reinterpret_cast<float*>(ptr + offset);
    offset += combine_gin_G2S_prob_sz;

    ep_group->ht_buffers.dispatch_gin_G2S_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.dispatch_gin_G2S_flags, 0, flags_sz));
    offset += flags_sz;

    ep_group->ht_buffers.combine_gin_G2S_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.combine_gin_G2S_flags, 0, flags_sz));
    offset += flags_sz;

    // RDMA sync-guard regions (dispatch, combine): addressed by window+offset only.
    CUDACHECK_RET(cudaMemset(ptr + offset, 0, guard_sz));
    offset += guard_sz;
    CUDACHECK_RET(cudaMemset(ptr + offset, 0, guard_sz));
    offset += guard_sz;

    ep_group->ht_buffers.token_staging_buffer = reinterpret_cast<void*>(ptr + offset);
    offset += token_staging_sz;

    ep_group->ht_buffers.dense_prob_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += dense_prob_sz;

    offset += rdma_send_staging_sz;

    // Calculate offsets for kernel mr_info
    size_t cur_offset = 0;
    ep_group->gin_config.combine_red_token_offset = cur_offset;
    cur_offset += combine_gin_RED_tokens_sz;

    ep_group->gin_config.combine_g2s_token_offset = cur_offset;
    cur_offset += combine_gin_G2S_tokens_sz;

    ep_group->gin_config.combine_red_prob_offset = cur_offset;
    cur_offset += combine_gin_RED_prob_sz;

    ep_group->gin_config.combine_g2s_prob_offset = cur_offset;
    cur_offset += combine_gin_G2S_prob_sz;

    cur_offset += flags_sz * 2;

    ep_group->gin_config.dispatch_guard_offset = cur_offset;
    cur_offset += guard_sz;
    ep_group->gin_config.combine_guard_offset = cur_offset;
    cur_offset += guard_sz;

    ep_group->gin_config.token_staging_offset = cur_offset;
    cur_offset += token_staging_sz;

    ep_group->gin_config.dense_prob_offset = cur_offset;
    cur_offset += dense_prob_sz;

    ep_group->gin_config.gin_send_staging_offset = cur_offset;
    cur_offset += rdma_send_staging_sz;

    ep_group->gin_config.gin_recv_staging_offset = cur_offset;
    cur_offset += rdma_recv_packed_sz;

    // =========================================================================
    // Phase 2: configure cross-LSA-team GIN resources
    // =========================================================================
    // Verify that configured HT LSA-team count matches NCCL rail team size.
    ncclTeam rail_team = ncclTeamRail(ep_group->comm);
    if (rail_team.nRanks != rdma_team_size) {
        fprintf(stderr, "[HT GIN] Error: rail team size (%d) must equal number of LSA domains (%d)\n", rail_team.nRanks,
                rdma_team_size);
        return ncclInvalidUsage;
    }

    int qps_per_rank = ep_group->config.num_qp_per_rank;
    // Only dispatch has dedicated cross-node warps per CTA. Combine selects
    // channels from the resulting shared context pool, so its SM budget does
    // not impose an additional per-CTA minimum here.
    int min_required_ctx =
        NCCL_EP_HT_RESERVED_GIN_GPU_CTXS + (ep_group->dispatch_num_sms * NCCL_EP_HT_DISPATCH_N2N_WARPS);
    if (qps_per_rank == 0) qps_per_rank = min_required_ctx;
    if (qps_per_rank < min_required_ctx) {
        fprintf(stderr,
                "[HT GIN] Error: num_qp_per_rank(%d) must be >= %d for reserved + dedicated N2N warp contexts\n",
                qps_per_rank, min_required_ctx);
        return ncclInvalidUsage;
    }
    ep_group->gin_config.qps_per_rank = qps_per_rank;
    ep_group->gin_config.num_comms = 1;
    // num_qp_per_rank is the total context budget; the data range is what's left after the reserved ones.
    ep_group->gin_config.num_ctx_per_comm = qps_per_rank - NCCL_EP_HT_RESERVED_GIN_GPU_CTXS;

    int dispatch_signals = lsa_team_size * rdma_team_size * max_chunks_per_rank;
    int combine_signals = lsa_team_size * rdma_team_size * max_chunks_per_rank;
    int streaming_tail_signals = rdma_team_size * rdma_team_size * lsa_team_size * max_chunks_per_rank;
    int streaming_head_signals = rdma_team_size * rdma_team_size * lsa_team_size;
    ep_group->gin_config.num_total_signals =
        dispatch_signals + combine_signals + streaming_tail_signals + streaming_head_signals + MAX_BARRIER_SESSIONS;
    ep_group->gin_config.signals_base = 0;
    ep_group->gin_config.combine_signal_offset = dispatch_signals;
    ep_group->gin_config.signals_tail_base = dispatch_signals + combine_signals;

    // =========================================================================
    // Phase 3: comm setup (DevCommCreate + WindowRegister)
    // =========================================================================
    ep_group->gin_config.num_dcomms = 1;
    ep_group->gin_config.dcomms = new ncclDevComm_t[1];

    {
        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCL_CHECK_RESULT(ncclCommQueryProperties(ep_group->comm, &props));
        if (props.railedGinType == NCCL_GIN_TYPE_NONE) {
            fprintf(stderr, "[HT GIN] Error: NCCL EP internode requires GIN, but GIN is not supported\n");
            return ncclInvalidUsage;
        }
    }

    {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.ginSignalCount = ep_group->gin_config.num_total_signals;
        reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
        reqs.ginContextCount = ep_group->gin_config.qps_per_rank; // reserved + data contexts
        reqs.ginQueueDepth = 3 * ht_tokens_per_chunk + 1;
        // Dispatch uses sessions [0, dispatch_num_sms); all combine CTAs rendezvous
        // through one tail CTA at session dispatch_num_sms. Therefore
        // combine_num_sms does not affect this resource count.
        reqs.lsaBarrierCount =
            std::max<int>(ep_group->dispatch_num_sms, NCCL_EP_HT_DISPATCH_BLOCKS) + 1;
        NCCLCHECK(ncclDevCommCreate(ep_group->comm, &reqs, &ep_group->gin_config.dcomms[0]));
    }

    CUDACHECK_RET(cudaMalloc(
        reinterpret_cast<void**>(&ep_group->gin_config.d_dcomms),
        sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms));
    CUDACHECK_RET(cudaMemcpy(
        ep_group->gin_config.d_dcomms,
        ep_group->gin_config.dcomms,
        sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms,
        cudaMemcpyHostToDevice));

    // WindowRegister
    NCCLCHECK(ncclCommWindowRegister(
        ep_group->comm,
        ep_group->gin_config.gin_base_ptr,
        total_gin_buffer_size,
        &ep_group->gin_config.nccl_window,
        0));

    ep_group->ht_buffers.internode_initialized = true;
    return ncclSuccess;
}

static ncclResult_t destroy_ht_internode(ncclEpGroup_t ep_group) {
    if (!ep_group->ht_buffers.internode_initialized) return ncclSuccess;

    // =========================================================================
    // Cleanup using public NCCL APIs
    // =========================================================================

    // Destroy device communicator
    if (ep_group->gin_config.dcomms != nullptr) {
        ncclResult_t res = ncclDevCommDestroy(ep_group->comm, &ep_group->gin_config.dcomms[0]);
        if (res != ncclSuccess) {
            fprintf(stderr, "[HT GIN] Warning: Failed to destroy device comm: %s\n", ncclGetErrorString(res));
        }
        delete[] ep_group->gin_config.dcomms;
        ep_group->gin_config.dcomms = nullptr;
    }
    // Free device memory for dcomms
    if (ep_group->gin_config.d_dcomms != nullptr) {
        cudaFree(ep_group->gin_config.d_dcomms);
        ep_group->gin_config.d_dcomms = nullptr;
    }

    // Deregister the window
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclCommWindowDeregister(ep_group->comm, ep_group->gin_config.nccl_window);
        ep_group->gin_config.nccl_window = {};
    }

    // Free the single GIN buffer (contains all RDMA regions)
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclResult_t res = ncclMemFree(ep_group->gin_config.gin_base_ptr);
        if (res != ncclSuccess) {
            fprintf(stderr, "[HT GIN] Warning: Failed to free GIN memory: %s\n", ncclGetErrorString(res));
        }
        ep_group->gin_config.gin_base_ptr = nullptr;

        // Clear buffer pointers (they pointed into gin_base_ptr)
        ep_group->ht_buffers.combine_gin_RED_tokens = nullptr;
        ep_group->ht_buffers.combine_gin_G2S_tokens = nullptr;
        ep_group->ht_buffers.combine_gin_RED_prob = nullptr;
        ep_group->ht_buffers.combine_gin_G2S_prob = nullptr;
        ep_group->ht_buffers.dispatch_gin_G2S_flags = nullptr;
        ep_group->ht_buffers.combine_gin_G2S_flags = nullptr;
        ep_group->ht_buffers.token_staging_buffer = nullptr;
        ep_group->ht_buffers.dense_prob_buffer = nullptr;
    }

    ep_group->gin_config.num_comms = 0;

    ep_group->ht_buffers.internode_initialized = false;
    return ncclSuccess;
}

static cudaError_t default_alloc_fn(void** ptr, size_t size, void* /*context*/) {
    return cudaMalloc(ptr, size);
}
static cudaError_t default_free_fn(void* ptr, void* /*context*/) {
    return cudaFree(ptr);
}

ncclResult_t ncclEpGetVersion(int* version) {
    if (version == nullptr) return ncclInvalidArgument;
    *version = NCCL_EP_VERSION_CODE;
    return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
// CUDA_MAJOR/CUDA_MINOR are injected at build time by makefiles/common.mk and by the top-level
// CMakeLists.txt — they reflect the CUDA toolkit this library was BUILT against.
#define NCCL_EP_STR2(v) #v
#define NCCL_EP_STR(v) NCCL_EP_STR2(v)
#define NCCL_EP_VERSION_STRING \
    "NCCL EP version " NCCL_EP_STR(NCCL_EP_MAJOR) "." NCCL_EP_STR(NCCL_EP_MINOR) "." NCCL_EP_STR( \
        NCCL_EP_PATCH) "+cuda" NCCL_EP_STR(CUDA_MAJOR) "." NCCL_EP_STR(CUDA_MINOR)

static void showVersion() {
    static std::once_flag once;
    std::call_once(once, []() {
        // No EP group exists at library-load time, so read a transient config
        // just to honor NCCL_EP_DEBUG for the banner. Per-group state is
        // populated later in ncclEpCreateGroup.
        ncclEpEnvConfig cfg;
        nccl_ep_env_init(&cfg);
        if (nccl_ep_env_flag_on(cfg.debug)) {
            fprintf(stderr, "%s\n", NCCL_EP_VERSION_STRING);
        }
    });
}

// Print the version banner when libnccl_ep.so is loaded (respects NCCL_EP_DEBUG).
__attribute__((constructor)) static void nccl_ep_lib_init() {
    showVersion();
}

ncclResult_t ncclEpCreateGroup(ncclEpGroup_t* out_ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* in_config) {
    // Parameter validation
    assert(out_ep_group != nullptr);
    int nRanks;
    assert(comm != nullptr && ncclCommCount(comm, &nRanks) == ncclSuccess && nRanks > 0);
    EP_VALIDATE_STRUCT(in_config, NCCL_EP_GROUP_CONFIG);

    // Decode the caller-owned object into current library storage. A future
    // caller may be larger, while a future library may receive this frozen V1
    // prefix; in both directions the copy is bounded and new library fields
    // retain initializer defaults.
    ncclEpGroupConfig_t parsed_config =
        epDecodeStruct(in_config, NCCL_EP_GROUP_CONFIG_INIT);
    in_config = &parsed_config;
    assert(
        (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY || in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) &&
        "ncclEpCreateGroup: invalid algorithm, supported: low_latency, high_throughput");

    bool low_latency_mode = (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY);
    bool ht_mode = (in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);
    assert(in_config->num_experts > 0 && "ncclEpCreateGroup: num_experts must be greater than 0");
    assert(in_config->max_token_bytes > 0 && "ncclEpCreateGroup: max_token_bytes must be greater than 0");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (ht_mode && in_config->max_token_bytes % sizeof(int4) != 0) {
        fprintf(
            stderr,
            "NCCL EP: HT max_token_bytes (%u) must be a multiple of %zu bytes\n",
            in_config->max_token_bytes,
            sizeof(int4));
        cudaStreamDestroy(stream);
        return ncclInvalidArgument;
    }
    assert(
        !(in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY && in_config->max_dispatch_tokens_per_rank == 0) &&
        "ncclEpCreateGroup: max_dispatch_tokens_per_rank must be greater than 0 for low latency mode");
    assert(
        !(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && in_config->max_dispatch_tokens_per_rank == 0) &&
        "ncclEpCreateGroup: max_dispatch_tokens_per_rank must be set for HT backend");
    // HT max_dispatch_tokens_per_rank need not be a chunk multiple; the cross-LSA-team
    // stride is chunk-aligned internally (ep_group->ht_aligned_max_tokens).
    // Create teams: LSA and Rail
    ncclTeam lsa_team = ncclTeamLsa(comm);
    ncclTeam rail_team = ncclTeamRail(comm);

    // Allocate EP group structure
    void* raw_memory = malloc(sizeof(ncclEpGroup));
    assert(raw_memory != nullptr && "Failed to malloc for ncclEpGroup");
    *out_ep_group = new (raw_memory) ncclEpGroup();
    ncclEpGroup_t ep_group = *out_ep_group;

    // Store configuration
    ep_group->comm = comm;
    ep_group->config = *in_config;

    // Populate this group's environment configuration once.
    nccl_ep_env_init(&ep_group->env);

    ep_group->alloc.alloc_fn = default_alloc_fn;
    ep_group->alloc.free_fn = default_free_fn;
    if (in_config->alloc.alloc_fn || in_config->alloc.free_fn) {
        if (!(in_config->alloc.alloc_fn && in_config->alloc.free_fn)) {
            fprintf(stderr, "NCCL EP: Failed to create group: Both alloc and free callbacks must be provided\n");
            return ncclInvalidUsage;
        }
        ep_group->alloc.alloc_fn = in_config->alloc.alloc_fn;
        ep_group->alloc.free_fn = in_config->alloc.free_fn;
        ep_group->alloc.context = in_config->alloc.context;
    }

    NCCL_CHECK_RESULT(ncclCommCount(comm, &ep_group->nRanks));
    NCCL_CHECK_RESULT(ncclCommUserRank(comm, &ep_group->rank));
    NCCL_CHECK_RESULT(ncclCommCuDevice(comm, &ep_group->cuda_device_id));

    // Stamp this process's rank into the env config
    nccl_ep_env_set_rank(&ep_group->env, ep_group->rank);

    if (in_config->version > NCCL_EP_API_VERSION && nccl_ep_env_verbose(ep_group->env)) {
        fprintf(
            stderr,
            "NCCL EP WARN: the application uses API version %u, but the loaded "
            "libnccl_ep.so supports version %u; features introduced by newer API "
            "versions are unsupported and will be ignored.\n",
            in_config->version,
            (unsigned int)NCCL_EP_API_VERSION);
    }

    // Dump the resolved environment configuration once if requested
    if (nccl_ep_env_verbose(ep_group->env)) nccl_ep_env_print(ep_group->env);

    CUDA_CHECK(cudaSetDevice(ep_group->cuda_device_id));
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, ep_group->cuda_device_id));
    ep_group->device_sm = static_cast<unsigned int>(device_prop.major * 10 + device_prop.minor);
    ep_group->device_sm_count = device_prop.multiProcessorCount;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &ep_group->max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, ep_group->cuda_device_id));
    ep_group->device_smem_optin = device_prop.sharedMemPerBlockOptin;

    // Resolve the shared budget first. New operation-specific config fields
    // inherit it when AUTO, preserving the V1/V2 behavior exactly.
    const unsigned int default_comm_sms =
        in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT ?
            NCCL_EP_HT_DFLT_NUM_SMS :
            ep_group->device_sm_count;
    unsigned int shared_num_sms = default_comm_sms;
    if (in_config->max_num_sms != NCCL_EP_AUTO) {
        if (in_config->max_num_sms > ep_group->device_sm_count) {
            fprintf(stderr, "Error: NCCL EP requires max_num_sms <= device_sm_count\n");
            return ncclInvalidUsage;
        }
        shared_num_sms = in_config->max_num_sms;
    }
    // Resolve an operation-specific config value, inheriting the already
    // resolved legacy/default budget when the new field is left as AUTO.
    auto resolve_config_sms = [dev_sms = ep_group->device_sm_count](
                                  const char* name, unsigned int requested, unsigned int fallback,
                                  unsigned int* resolved) -> ncclResult_t {
        if (requested == NCCL_EP_AUTO) {
            *resolved = fallback;
            return ncclSuccess;
        }
        if (requested > dev_sms) {
            fprintf(stderr, "Error: NCCL EP requires %s <= device_sm_count (%u)\n", name, dev_sms);
            return ncclInvalidUsage;
        }
        *resolved = requested;
        return ncclSuccess;
    };
    NCCL_CHECK_RESULT(resolve_config_sms(
        "dispatch_num_sms", in_config->dispatch_num_sms, shared_num_sms, &ep_group->dispatch_num_sms));
    NCCL_CHECK_RESULT(resolve_config_sms(
        "combine_num_sms", in_config->combine_num_sms, shared_num_sms, &ep_group->combine_num_sms));
    ep_group->shuffle_sms =
        in_config->max_num_sms == NCCL_EP_AUTO ? ep_group->device_sm_count : in_config->max_num_sms;
    // Preprocessing defaults to all device SMs so a small communication budget
    // does not throttle its scan kernels.
    ep_group->preprocess_num_sms = ep_group->device_sm_count;

    // Apply env overrides. Precedence is operation env > legacy shared env >
    // operation config > legacy shared config > algorithm default.
    const unsigned int dev_sms = ep_group->device_sm_count;
    // Apply one environment override without discarding the lower-precedence
    // value when the override is invalid.
    auto apply_sms_override = [dev_sms, &env = ep_group->env](const ncclEpEnvVar& var, unsigned int& target) {
        if (!var.is_set) return;
        if (var.value.ul >= 1 && var.value.ul <= dev_sms) {
            target = static_cast<unsigned int>(var.value.ul);
        } else {
            fprintf(stderr, "[nccl_ep] %s=%lu out of range (must be in [1, %u]); using %u\n", var.name, var.value.ul,
                    dev_sms, target);
            nccl_ep_env_print(env);
        }
    };
    apply_sms_override(ep_group->env.comm_num_sms, ep_group->dispatch_num_sms);
    apply_sms_override(ep_group->env.comm_num_sms, ep_group->combine_num_sms);
    apply_sms_override(ep_group->env.dispatch_num_sms, ep_group->dispatch_num_sms);
    apply_sms_override(ep_group->env.combine_num_sms, ep_group->combine_num_sms);
    apply_sms_override(ep_group->env.shuffle_sms, ep_group->shuffle_sms);
    apply_sms_override(ep_group->env.preprocess_num_sms, ep_group->preprocess_num_sms);

    // LL warp-group bound applies independently to both operation grids.
    if (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        constexpr int llMaxWarpGroupsLimit = nccl_ep::ll::kLlDispatchMaxWarpGroups;
        const int num_experts = ep_group->config.num_experts;
        const struct {
            const char* operation;
            unsigned int sms;
        } budgets[] = {
            {"dispatch", ep_group->dispatch_num_sms},
            {"combine", ep_group->combine_num_sms},
        };
        for (const auto& budget : budgets) {
            const int numWarpGroups = (num_experts + static_cast<int>(budget.sms) - 1) / budget.sms;
            if (numWarpGroups > llMaxWarpGroupsLimit) {
                const int required_sms = (num_experts + llMaxWarpGroupsLimit - 1) / llMaxWarpGroupsLimit;
                fprintf(stderr,
                        "Error: insufficient %s SM count for Low-Latency mode: %u SMs with %d "
                        "experts gives %d warp groups (max %d). Need >= %d SMs.\n",
                        budget.operation, budget.sms, num_experts, numWarpGroups,
                        llMaxWarpGroupsLimit, required_sms);
                return ncclInvalidUsage;
            }
        }
    }

    // Determine number of nodes by gathering hostnames and counting unique ones
    constexpr size_t HOSTNAME_LEN = 256;
    std::vector<char> all_hostnames(HOSTNAME_LEN * ep_group->nRanks, 0);
    gethostname(all_hostnames.data() + ep_group->rank * HOSTNAME_LEN, HOSTNAME_LEN);
    ncclAllGatherHost(all_hostnames.data(), HOSTNAME_LEN, ep_group->rank, ep_group->nRanks, comm, stream);
    std::set<std::string> unique_hosts;
    for (int i = 0; i < ep_group->nRanks; ++i) {
        unique_hosts.insert(std::string(all_hostnames.data() + i * HOSTNAME_LEN));
    }
    ep_group->nNodes = static_cast<int>(unique_hosts.size());

    ep_group->num_local_experts = ep_group->config.num_experts / ep_group->nRanks;
    EP_HOST_ASSERT(
        !(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && ep_group->config.max_recv_tokens_per_rank != 0 &&
          ep_group->config.max_recv_tokens_per_rank < ep_group->config.max_dispatch_tokens_per_rank) &&
        "ncclEpCreateGroup: HT mode requires max_recv_tokens_per_rank >= max_dispatch_tokens_per_rank");
    EP_HOST_ASSERT(
        !(in_config->num_topk != 0 && in_config->num_topk > in_config->num_experts) &&
        "ncclEpCreateGroup: num_topk must be <= num_experts");

    // Resolve the per-rank recv slot budget.
    //   HT explicit value: fixed budget; callers size recv buffers to it.
    //   HT AUTO/0: eager mode; the bound only sizes internal buffers and callers
    //   size recv buffers to the actual recv count per routing. Expert-Major
    //   expands each token to up to num_topk slots, hence the config.num_topk
    //   factor. Eager relies on TRAP semantics, so DROP requires a fixed budget.
    //   LL AUTO/0: nRanks * max_dispatch_tokens_per_rank (layout-agnostic).
    ep_group->eager_mode = ht_mode && (ep_group->config.max_recv_tokens_per_rank == NCCL_EP_AUTO);
    if (ep_group->eager_mode) {
        EP_HOST_ASSERT(
            ep_group->config.overflow_policy != NCCL_EP_OVERFLOW_DROP &&
            "ncclEpCreateGroup: eager mode (max_recv_tokens_per_rank = NCCL_EP_AUTO) "
            "does not support NCCL_EP_OVERFLOW_DROP");
        const size_t bound = static_cast<size_t>(ep_group->nRanks) * ep_group->config.max_dispatch_tokens_per_rank *
                             (ep_group->config.num_topk > 0 ? ep_group->config.num_topk : 1);
        EP_HOST_ASSERT(bound <= UINT_MAX && "ncclEpCreateGroup: eager recv bound overflows unsigned int");
        ep_group->config.max_recv_tokens_per_rank = static_cast<unsigned int>(bound);
    } else if (ep_group->config.max_recv_tokens_per_rank == 0) {
        ep_group->config.max_recv_tokens_per_rank = ep_group->nRanks * ep_group->config.max_dispatch_tokens_per_rank;
    }
    ep_group->max_recv_tokens = static_cast<int>(ep_group->config.max_recv_tokens_per_rank);

    // Collective: all ranks must agree on the resolved budget (IPC buffers are sized from it).
    {
        std::vector<unsigned int> all_budgets(ep_group->nRanks, 0);
        all_budgets[ep_group->rank] = ep_group->config.max_recv_tokens_per_rank;
        ncclAllGatherHost(all_budgets.data(), sizeof(unsigned int), ep_group->rank, ep_group->nRanks, comm, stream);
        for (int r = 1; r < ep_group->nRanks; ++r) {
            EP_HOST_ASSERT(
                all_budgets[r] == all_budgets[0] &&
                "ncclEpCreateGroup: max_recv_tokens_per_rank must be identical across ranks");
        }
    }

    // Apply default values for auto-configured fields (when set to NCCL_EP_AUTO)
    if (ep_group->config.num_channels == NCCL_EP_AUTO) {
        ep_group->config.num_channels = 10;
    }

    if (ep_group->config.num_qp_per_rank == NCCL_EP_AUTO) {
        // Dispatch dedicates N2N warps per CTA; combine reuses this pool.
        ep_group->config.num_qp_per_rank =
            NCCL_EP_HT_RESERVED_GIN_GPU_CTXS + (ep_group->dispatch_num_sms * NCCL_EP_HT_DISPATCH_N2N_WARPS);
    }

    // Resolve timeout_cycles: env var > config field > compile-time default
    {
        int dev;
        int clock_khz_int;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz_int, cudaDevAttrClockRate, dev));
        uint64_t clock_khz = static_cast<uint64_t>(clock_khz_int);

        uint64_t resolved = NUM_TIMEOUT_CYCLES;
        const char* source = "compile-time default";
        const uint64_t env_ms = static_cast<uint64_t>(ep_group->env.timeout_ms.value.ul);
        // Only a positive timeout overrides the default.
        const bool have_env_ms = ep_group->env.timeout_ms.is_set && env_ms > 0;

        if (have_env_ms) {
            resolved = clock_khz * 1000ULL * env_ms / 1000ULL;
            source = "NCCL_EP_TIMEOUT_MS env var";
            if (ep_group->config.timeout_ns != 0 && ep_group->rank == 0)
                fprintf(stderr, "NCCL EP: NCCL_EP_TIMEOUT_MS=%lu overrides config.timeout_ns=%lu\n",
                        (unsigned long)env_ms, (unsigned long)ep_group->config.timeout_ns);
        } else if (ep_group->config.timeout_ns != 0) {
            resolved = clock_khz * 1000ULL * (ep_group->config.timeout_ns / 1000000ULL) / 1000ULL;
            source = "config.timeout_ns";
        }

        ep_group->timeout_cycles = resolved;
        if (ep_group->rank == 0) {
            uint64_t timeout_ms = resolved / (clock_khz * 1000ULL / 1000ULL);
            char env_str[32];
            if (have_env_ms) snprintf(env_str, sizeof(env_str), "%llu", (unsigned long long)env_ms);
            else snprintf(env_str, sizeof(env_str), "unset");
            fprintf(stderr, "NCCL EP: using timeout=%llums (env=%s, config.timeout_ns=%llu, source=%s)\n",
                    (unsigned long long)timeout_ms, env_str, (unsigned long long)ep_group->config.timeout_ns, source);
        }
    }

    // Physical node properties. rank_in_node must lie in [0, gpus_per_node)
    // so the peer-access loop below can skip the self-device. Using the
    // within-comm rank (rather than the physical cuda_device_id) keeps this
    // invariant when multiple EP comms colocate on one physical node (e.g.
    // DP × EP mesh where ranks 4..7 form a second EP group on the same box).
    ep_group->gpus_per_node = ep_group->nRanks / ep_group->nNodes;
    ep_group->rank_in_node = ep_group->rank % ep_group->gpus_per_node;
    ep_group->node_id = ep_group->rank / ep_group->gpus_per_node;
    ep_group->lsa_team_size = lsa_team.nRanks;
    ep_group->lsa_rank = lsa_team.rank;
    if (ht_mode) {
        // HT uses rail-domain decomposition.
        ep_group->rdma_team_size = rail_team.nRanks;
        ep_group->rdma_rank = rail_team.rank;
    } else {
        // Preserve legacy semantics.
        // TODO: are we using this in LL?
        ep_group->rdma_team_size = ep_group->nRanks;
        ep_group->rdma_rank = ep_group->rank;
    }
    if (ht_mode) {
        assert(
            ep_group->rdma_team_size > 0 && ep_group->lsa_team_size > 0 &&
            "ncclEpCreateGroup: invalid HT team cardinalities");
        assert(
            ep_group->rdma_team_size * ep_group->lsa_team_size == ep_group->nRanks &&
            "ncclEpCreateGroup: HT requires rdma_team_size * lsa_team_size == nRanks");

        // Pull dispatch and push combine run over NVLink within a single LSA team for
        // now; multi-node (inter-LSA-team) support is a future extension. Fail fast at
        // group creation on an inter-LSA-team topology until then.
        if (nccl_ep_env_flag_on(ep_group->env.ht_em_pull_push) && ep_group->rdma_team_size > 1) {
            fprintf(stderr,
                    "ncclEpCreateGroup: NCCL_EP_HT_EM_PULL_PUSH is "
                    "single-LSA-team only for now; inter-LSA-team (rdma_team_size=%d > 1) is not yet supported.\n",
                    ep_group->rdma_team_size);
            return ncclInvalidUsage;
        }
    }

    // HT EM mode: env override wins, else auto-pick from (zero_copy, number of LSA teams).
    //   pull_push (any zero_copy)   -> kPullPush     (NVLink pull dispatch + push combine)
    //   non zero_copy               -> kLocalPermute (FLAT dispatch + permute kernels)
    //   zero_copy, multiple teams   -> kNvlinkDup    (sender duplicates per-expert over NVLink)
    //   zero_copy, single team      -> kLocalDup     (receiver-side fan-out via local_dup)
    {
        const bool want_local_dup = nccl_ep_env_flag_on(ep_group->env.ht_em_local_dup);
        const bool want_nvlink_dup = nccl_ep_env_flag_on(ep_group->env.ht_em_nvlink_dup);
        const bool want_pull_push = nccl_ep_env_flag_on(ep_group->env.ht_em_pull_push);
        if (want_local_dup && want_nvlink_dup) {
            fprintf(stderr, "NCCL EP: NCCL_EP_HT_EM_LOCAL_DUP and NCCL_EP_HT_EM_NVLINK_DUP are mutually exclusive\n");
            return ncclInvalidUsage;
        }
        // Pull-push and the dup modes are distinct EM recipes. Reject the explicit
        // conflict instead of silently dropping the pull-push request.
        if (want_pull_push && (want_local_dup || want_nvlink_dup)) {
            fprintf(stderr, "NCCL EP: NCCL_EP_HT_EM_PULL_PUSH is mutually exclusive with "
                            "NCCL_EP_HT_EM_LOCAL_DUP / NCCL_EP_HT_EM_NVLINK_DUP\n");
            return ncclInvalidUsage;
        }
        if (want_nvlink_dup) {
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kNvlinkDup;
        } else if (want_local_dup) {
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kLocalDup;
        } else if (want_pull_push) {
            // Checked before the zero_copy default: pull-push runs with either zero_copy
            // setting (zero_copy just elides the dispatch input stage copy), so an explicit
            // request must win instead of falling back to a dup mode.
            // Single-LSA-team enforced by the reject at group creation above.
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kPullPush;
        } else if (in_config->zero_copy == NCCL_EP_ZERO_COPY_ON) {
            ep_group->ht_em_mode =
                (ep_group->rdma_team_size > 1) ? ncclEpGroup::HtEmMode::kNvlinkDup : ncclEpGroup::HtEmMode::kLocalDup;
        } else {
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kLocalPermute;
        }
        // Unfused head/tail sync is only wired for the pull-push EM path.
        if (nccl_ep_env_flag_on(ep_group->env.ht_unfused_sync) &&
            ep_group->ht_em_mode != ncclEpGroup::HtEmMode::kPullPush) {
            fprintf(stderr, "NCCL EP: NCCL_EP_HT_UNFUSED_SYNC is only supported with "
                            "NCCL_EP_HT_EM_PULL_PUSH\n");
            return ncclInvalidUsage;
        }
    }

    ep_group->rdma_buffer = nullptr;

    CUDA_CHECK(ep_group->alloc.alloc_fn(&ep_group->ep_workspace, NUM_WORKSPACE_BYTES, ep_group->alloc.context));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ep_workspace, 0, NUM_WORKSPACE_BYTES, stream));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK_RESULT(ncclCommQueryProperties(ep_group->comm, &props));
    if (!props.deviceApiSupport) {
        fprintf(stderr, "Error: NCCL EP requires NCCL Device API support, but Device API is not supported\n");
        return ncclInvalidUsage;
    }

    // Initialize HT intra-LSA buffers (windows, completion flags, etc.)
    if (ht_mode) {
        // Resolve the dispatch/combine tokens-per-chunk for this group. RDMA
        // (multi-LSA-team) configs use the tuned 64-token default; LSA-only configs
        // use a dispatch-grid-proportional size so one chunk is one wave across
        // the dispatch SMs (NUM_OF_TOKENS_PER_GROUP tokens per SM). This keeps
        // combine grid tuning from changing dispatch scheduling. Either may be overridden by
        // NCCL_EP_TOKENS_PER_CHUNK. The chunk must be a multiple of 32 (warp width;
        // also covers the uint4 routing-map-load and token-group granularities), so
        // non-conforming values are rounded up with a warning.
        auto round_up_32 = [](int v) { return (v + 31) & ~31; };
        int chunk =
            (ep_group->rdma_team_size > 1) ?
                HT_TOKENS_PER_CHUNK_RDMA_DEFAULT :
                round_up_32(
                    NCCL_EP_HT_COMBINE_TOK_PER_GROUP * static_cast<int>(ep_group->dispatch_num_sms));
        if (ep_group->env.tokens_per_chunk.is_set && ep_group->env.tokens_per_chunk.value.ul > 0) {
            const int requested = static_cast<int>(ep_group->env.tokens_per_chunk.value.ul);
            chunk = round_up_32(requested);
            if (chunk != requested) {
                fprintf(stderr, "[nccl_ep] %s=%d rounded up to %d (must be a multiple of 32)\n",
                        ep_group->env.tokens_per_chunk.name, requested, chunk);
            }
        }
        ep_group->ht_tokens_per_chunk = chunk;
        // Chunk-aligned cross-LSA-team RDMA stride; also the max-tokens template arg for
        // the dispatch/combine kernels (guarantees MAX_NUM_OF_TOKENS_PER_RANK % chunk == 0).
        ep_group->ht_aligned_max_tokens = ((ep_group->config.max_dispatch_tokens_per_rank + chunk - 1) / chunk) * chunk;

        NCCL_CHECK_RESULT(init_ht_intranode(ep_group, in_config, stream));
        NCCL_CHECK_RESULT(init_ht_internode(ep_group, in_config, stream));

        // kPullPush replaces the routing bitmap with the order-preserving uint16 topk
        // map (the scan consumes global_topk_idx instead), so only one of the two is
        // ever allocated per group.
        if (ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kPullPush) {
            // Group-scoped; sized by the MAX_NUM_TOPK cap so it is handle-independent
            // (runtime row stride is the handle's num_topk <= MAX_NUM_TOPK).
            const size_t topk_idx_bytes = static_cast<size_t>(ep_group->nRanks) *
                                        ep_group->config.max_dispatch_tokens_per_rank *
                                        MAX_NUM_TOPK * sizeof(uint16_t);
            CUDA_CHECK(ep_group->alloc.alloc_fn(
                reinterpret_cast<void**>(&ep_group->ht_buffers.global_topk_idx),
                topk_idx_bytes,
                ep_group->alloc.context));
        } else {
            // Group-scoped routing bitmap shared by all handles on this group.
            // Per-token row is byte-padded per LSA-team: each team gets its own
            // ceil(experts_per_lsa_team/8)-byte block (== ceil(num_experts/8) for one team).
            const int rm_experts_per_lsa_team = ep_group->lsa_team_size * ep_group->num_local_experts;
            const size_t rm_row_bytes = static_cast<size_t>((rm_experts_per_lsa_team + 7) / 8)
                                      * ep_group->rdma_team_size;
            const size_t routing_bytes = static_cast<size_t>(ep_group->nRanks) *
                                         ep_group->config.max_dispatch_tokens_per_rank *
                                         rm_row_bytes;
            CUDA_CHECK(ep_group->alloc.alloc_fn(
                reinterpret_cast<void**>(&ep_group->ht_buffers.global_routing_map),
                routing_bytes,
                ep_group->alloc.context));
            ep_group->ht_buffers.global_routing_map_size = routing_bytes;
            // No init memset: AllGather in preprocessing overwrites every byte the kernel reads.
        }
    }

    if (low_latency_mode) {
        ep_group->num_nccl_comms = 0; // no split comms created

        // Cleaning up any pending CUDA error
        CUDA_CHECK(cudaGetLastError());

        // Create device communicator on ep_group->comm with all GIN contexts.
        // This depends only on group-level parameters (num_experts, nRanks),
        // not on the per-handle layout/num_topk, so it stays at group time.
        ncclDevComm_t* nccl_dev_comms_host = new ncclDevComm_t[1];
        nccl_dev_comms_host[0] = ncclDevComm_t{};
        ep_group->num_dispatch_signals = ep_group->num_local_experts * ep_group->nRanks;
        int num_total_signals = ep_group->num_dispatch_signals;
        ep_group->clean_barrier_signal_base = 2 * num_total_signals;

        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCLCHECK(ncclCommQueryProperties(ep_group->comm, &props));
        if (props.nLsaTeams > 1 && props.ginType == NCCL_GIN_TYPE_NONE) {
            fprintf(stderr, "[LL] Error: NCCL EP requires GIN, but GIN is not supported\n");
            return ncclInvalidUsage;
        }

        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        if (props.nLsaTeams > 1) {
            reqs.ginContextCount = ep_group->config.num_qp_per_rank; // all contexts in single comm
            // Signal layout: bank 0 [0, N), bank 1 [N, 2N), clean barrier [2N]
            reqs.ginSignalCount = 2 * num_total_signals + 1;
            reqs.ginForceEnable = true;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
            reqs.worldGinBarrierCount = 1;
        }
        NCCL_CHECK_RESULT(ncclDevCommCreate(ep_group->comm, &reqs, &nccl_dev_comms_host[0]));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_dev_comms), sizeof(ncclDevComm_t)));
        CUDA_CHECK(
            cudaMemcpy(ep_group->nccl_dev_comms, nccl_dev_comms_host, sizeof(ncclDevComm_t), cudaMemcpyHostToDevice));

        delete[] nccl_dev_comms_host;
        nccl_dev_comms_host = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->ll_epoch_state), sizeof(nccl_ep::LowLatencyEpochState)));
        CUDA_CHECK(cudaMemset(ep_group->ll_epoch_state, 0, sizeof(nccl_ep::LowLatencyEpochState)));

        if (ep_group->config.enable_mask) {
            // Allocate the cross-rank sync buffer used by clean_low_latency_buffer.
            // Its size depends only on nRanks (a group-level constant), so it is
            // sized and allocated here once, in its own registered NCCL window.
            // Keeping it out of rdma_buffer means lazy/grow reallocation of
            // rdma_buffer doesn't have to touch it, and ncclEpMaskClean can run
            // even before any LL handle has been created.
            const size_t sync_buffer_bytes = ((static_cast<size_t>(ep_group->nRanks) * sizeof(int) + 127) / 128) * 128;
            NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->sync_buffer, sync_buffer_bytes));
            CUDA_CHECK(cudaMemset(ep_group->sync_buffer, 0, sync_buffer_bytes));
            ncclWindow_t sync_win_host;
            NCCL_CHECK_RESULT(
                ncclCommWindowRegister(ep_group->comm, ep_group->sync_buffer, sync_buffer_bytes, &sync_win_host, 0));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->sync_window), sizeof(ncclWindow_t)));
            CUDA_CHECK(cudaMemcpy(ep_group->sync_window, &sync_win_host, sizeof(ncclWindow_t), cudaMemcpyHostToDevice));
        }

        // If the user passed an explicit rdma_buffer_size (anything other than
        // NCCL_EP_AUTO), honor it verbatim now. ll_init_handle will then only
        // verify the requested layout fits; growth is not performed. With
        // NCCL_EP_AUTO, allocation is deferred until ll_init_handle so it can
        // size the buffer with the actual (layout, num_topk).
        if (ep_group->config.rdma_buffer_size != NCCL_EP_AUTO) {
            NCCLCHECK(ll_resize_rdma_buffer(ep_group, ep_group->config.rdma_buffer_size));
        }
    }

    // Allocate mask buffer and async error flag for active-mask support
    if (ep_group->config.enable_mask && ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        size_t mask_bytes = ep_group->nRanks * sizeof(int);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->mask_buffer), mask_bytes));
        // Initialize all ranks as active (1 = active, 0 = masked/failed)
        std::vector<int> all_active(ep_group->nRanks, 1);
        CUDA_CHECK(
            cudaMemcpyAsync(ep_group->mask_buffer, all_active.data(), mask_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaHostAlloc(reinterpret_cast<void**>(&ep_group->async_error_flag), sizeof(int), cudaHostAllocMapped));
        *ep_group->async_error_flag = 0;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ncclSuccess;
}

ncclResult_t ncclEpGroupDestroy(ncclEpGroup_t ep_group) {
    if (ep_group == nullptr) {
        return ncclSuccess;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up HT intra-LSA resources
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && ep_group->ht_buffers.initialized) {
        destroy_ht_intranode(ep_group);
    }
    // Clean up HT cross-LSA-team resources (GIN deregistration must happen before ncclCommDestroy)
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && ep_group->ht_buffers.internode_initialized) {
        destroy_ht_internode(ep_group);
    }
    // Clean up mask buffer and async error flag
    if (ep_group->mask_buffer != nullptr) {
        CUDA_CHECK(cudaFree(ep_group->mask_buffer));
        ep_group->mask_buffer = nullptr;
    }
    if (ep_group->async_error_flag != nullptr) {
        CUDA_CHECK(cudaFreeHost(ep_group->async_error_flag));
        ep_group->async_error_flag = nullptr;
    }

    // Clean up workspace memory
    if (ep_group->ep_workspace != nullptr) {
        CUDA_CHECK(ep_group->alloc.free_fn(ep_group->ep_workspace, ep_group->alloc.context));
    }
    if (ep_group->ht_buffers.global_routing_map != nullptr) {
        CUDA_CHECK(ep_group->alloc.free_fn(ep_group->ht_buffers.global_routing_map, ep_group->alloc.context));
        ep_group->ht_buffers.global_routing_map = nullptr;
    }
    if (ep_group->ht_buffers.global_topk_idx != nullptr) {
        CUDA_CHECK(ep_group->alloc.free_fn(ep_group->ht_buffers.global_topk_idx, ep_group->alloc.context));
        ep_group->ht_buffers.global_topk_idx = nullptr;
    }

    // Clean up RDMA resources (single-comm path: 1 window, 1 devcomm on ep_group->comm).
    // Gate on algorithm only — config.rdma_buffer_size may be NCCL_EP_AUTO
    // (== 0) for LL groups where the buffer is sized lazily.
    if (NCCL_EP_ALGO_LOW_LATENCY == ep_group->config.algorithm) {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        NCCL_CHECK_RESULT(ncclBarrier(ep_group->comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        // Deregister single NCCL window if it was ever registered. The
        // rdma_buffer and its window are allocated lazily on the first LL
        // handle init, so a group that was created but never had an LL
        // handle still has nccl_wins == nullptr here.
        if (ep_group->nccl_wins != nullptr) {
            ncclWindow_t win_host;
            CUDA_CHECK(cudaMemcpy(&win_host, ep_group->nccl_wins, sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
            NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, win_host));
            CUDA_CHECK(cudaFree(ep_group->nccl_wins));
            ep_group->nccl_wins = nullptr;
        }

        // Free RDMA buffer (after window deregistered)
        if (ep_group->ll_epoch_state != nullptr) {
            CUDA_CHECK(cudaFree(ep_group->ll_epoch_state));
            ep_group->ll_epoch_state = nullptr;
        }

        if (ep_group->rdma_buffer) {
            NCCL_CHECK_RESULT(ncclMemFree(ep_group->rdma_buffer));
            ep_group->rdma_buffer = nullptr;
        }

        // Tear down the standalone sync buffer + its window.
        if (ep_group->sync_window != nullptr) {
            ncclWindow_t sync_win_host;
            CUDA_CHECK(cudaMemcpy(&sync_win_host, ep_group->sync_window, sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
            NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, sync_win_host));
            CUDA_CHECK(cudaFree(ep_group->sync_window));
            ep_group->sync_window = nullptr;
        }
        if (ep_group->sync_buffer != nullptr) {
            NCCL_CHECK_RESULT(ncclMemFree(ep_group->sync_buffer));
            ep_group->sync_buffer = nullptr;
        }

        // Destroy single NCCL device communicator (copy back from device, destroy on ep_group->comm)
        ncclDevComm_t dc_host;
        CUDA_CHECK(cudaMemcpy(&dc_host, ep_group->nccl_dev_comms, sizeof(ncclDevComm_t), cudaMemcpyDeviceToHost));
        NCCL_CHECK_RESULT(ncclDevCommDestroy(ep_group->comm, &dc_host));
        CUDA_CHECK(cudaFree(ep_group->nccl_dev_comms));
        ep_group->nccl_dev_comms = nullptr;

        // No split comms to destroy (using ep_group->comm directly)
    }
    // Invoke destructor explicitly (placement new was used)
    ep_group->~ncclEpGroup();

    // Free the group structure
    free(ep_group);

    return ncclSuccess;
}

struct ncclEpHandle {
    ncclEpGroup_t group;

    ncclEpLayout_t layout;

    // User-owned (do not free). LL reads directly; HT uses cached ht.topk_idx.
    ncclEpTensor_t topk_idx;
    int num_tokens, num_topk;
    // True once `num_tokens` has been authoritatively populated by
    // `ncclEpUpdateHandle` (or `ncclEpCreateHandle` via its inner update call).
    // Distinguishes a legitimate zero-token configuration from the
    // "not yet bound" state -- the dispatch / combine lazy branches must not
    // treat a real 0 as "uninitialised" and re-fetch sizes[0] from inputs,
    // since the input tensors on zero-token ranks are empty (and now legally
    // carry data == nullptr per tensorHasBinding's empty-tensor relaxation).
    bool num_tokens_set;

    bool cached_mode;
    int num_scales;
    int hidden_int4;

    union {
        struct {
            // packed tensors for LL (descriptors only; data pointers reference handle_mem)
            ncclEpTensor_t expert_recv_source_indices;
            ncclEpTensor_t expert_dispatch_layout;

            // Persistent backing storage for the per-tensor `sizes` arrays above.
            size_t expert_recv_source_indices_sizes[1];
            size_t expert_dispatch_layout_sizes[2];

            // Backing storage for the two tensors above. Layout matches ll_handle_mem_size().
            void* handle_mem;
            bool owns_handle_mem;

            std::function<ncclResult_t(unsigned int)> continue_fn;
            nccl_ep::LowLatencyLayout layout;
        } ll;
        struct {
            // Global routing map lives on ep_group->ht_buffers (group-scoped).

            // =================================================================================
            // PREPROCESSING OUTPUTS - Computed once per iteration, used by dispatch & combine
            // =================================================================================

            // Sparse-to-dense map: maps each (token, source_rank) pair to its position in
            // the destination rank's expert buffer. Used by NVLink (intra-LSA) warps.
            // Value of -1 indicates token is not routed to that rank.
            // dtype: int32_t
            // layout: [num_lsa_teams * max_dispatch_tokens_per_rank, lsa_team_size]
            // usage: dispatch S2G warp group, combine G2S warp group
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: no direct equivalent.
            //   - HT: uses is_token_in_rank (local tokens only) + atomics to compute positions
            //   - HT: precomputes positions for ALL LSA teams' tokens, -1 sentinel for not routed
            int32_t* sparse_to_dense_map;

            // RDMA-to-attention map: boolean mask indicating which tokens this LSA team needs
            // to RECEIVE from RDMA (cross-LSA-team). Indexed by [node_id, token_id].
            // Primarily used during combine to know which remote tokens to wait for.
            // dtype: bool
            // layout: [num_lsa_teams, max_dispatch_tokens_per_rank_padded_to_16]
            //        (padding to 16 required for TMA alignment)
            // usage: dispatch G2S warp (polling), combine cross-LSA-team G2S/reduction warps
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: inverse perspective of is_token_in_rank.
            //   - is_token_in_rank: outbound - "where do MY tokens go?" [my_token, dest_rank]
            //   - rdma_to_attn_map: inbound - "which remote tokens do I receive?" [src_node, token]
            bool* rdma_to_attn_map;

            // Attention-to-RDMA map: boolean mask indicating which local tokens need to be
            // SENT via RDMA (cross-LSA-team) to each remote LSA team.
            // Only allocated when num_lsa_teams > 1.
            // dtype: bool
            // layout: [max_dispatch_tokens_per_rank, num_lsa_teams - 1]
            // usage: dispatch N2N (RDMA) warp group
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: closest equivalent to is_token_in_rank for cross-LSA-team RDMA.
            //   - is_token_in_rank: per-rank granularity [num_tokens, num_ranks]
            //   - attn_to_rdma_map: per-LSA-team granularity [num_tokens, num_lsa_teams-1] (RDMA only)
            bool* attn_to_rdma_map;

            // Per-token per-rank bitmask cache produced during preprocessing.
            // dtype: rank_mask_t<ceil(lsa_team_size/64)> (one uint64_t word per 64 ranks)
            // layout: [num_lsa_teams * max_send_tokens_per_rank * ranks_per_lsa_team]
            void* token_rank_mask;

            // Local expert routing map: per-expert routing for tokens in this rank's buffer.
            // Used by subsequent expert MLP layers to route tokens to correct experts.
            // dtype: bool
            // layout: [max_recv_tokens, experts_per_rank]
            //        where max_recv_tokens = num_ranks * max_dispatch_tokens_per_rank
            // usage: passed to expert computation layers (not used by dispatch/combine directly)
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: similar purpose to num_tokens_per_expert but more detailed
            bool* local_expert_routing_map;

            // Number of tokens routed to local experts (total across all local experts).
            // Each token counted once even if routed to multiple local experts.
            // dtype: int32_t
            // layout: [1]
            // usage: buffer sizing, iteration control
            // lifetime: valid after metadata_preprocessing
            int32_t* num_tokens_for_experts;

            // =================================================================================
            // CONVERSION BUFFERS - Pre-allocated to avoid dispatch/combine-time malloc
            // =================================================================================

            // Dense prob buffer: shared scratch for sparse↔dense conversions
            // dtype: float
            // layout: [max_dispatch_tokens_per_rank, num_experts]
            // usage:
            //   - dispatch forward: sparse→dense input topk_weights conversion
            //   - combine backward: dense→sparse output prob conversion
            // lifetime: allocated at handle creation, freed at handle destroy
            // note: dispatch and combine are sequential, so one buffer suffices
            // For multi-LSA-team: points to group-level pre-registered buffer
            // For single-LSA-team: handle-owned buffer
            float* dense_prob_buffer;

            // Token staging buffer: pre-registered buffer to avoid GIN registration during dispatch
            // User tokens are copied here during dispatch, then this buffer is used for RDMA
            // dtype: uint32_t, uint16_t, or uint8_t according to raw wire width
            // layout: [max_dispatch_tokens_per_rank, hidden]
            // usage: copy user tokens → use for cross-LSA-team RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            void* token_staging_buffer; // Pointer to group-level buffer (not handle-owned)

            // RDMA inter-node group flags: atomic completion flags for each remote LSA team.
            // Remote ranks increment via RDMA atomic fetch-add to signal chunk completion.
            // Only allocated when num_lsa_teams > 1.
            // dtype: uint64_t
            // layout: [num_lsa_teams - 1]
            // usage: dispatch N2N warp (signaling), dispatch G2S warp (polling)
            // lifetime: reset to 0 at init, incremented by remote RDMA atomics
            // uint64_t* dispatch_gin_G2S_flags;

            // Per-handle preprocessing block (single allocation for all preprocessing buffers)
            void* preprocessing_block;
            bool owns_handle_mem; // false = caller-owned (user path); destroy skips free
            size_t preprocessing_zero_region_size;
            size_t preprocessing_s2d_size;
            void* preprocessing_scan_tmp;

            // Expert-major fields (alignment set in InitHandle; offsets/counts set in UpdateHandle)
            size_t dispatch_output_per_expert_alignment;
            int64_t* expert_token_offsets; // [experts_per_rank] written by remap kernel
            int32_t* per_expert_counts_active; // alias to authoritative counts buffer

            // Count mode, always used (both fused and unfused meta dispatch): per-handle scratch
            // that persists UpdateHandle -> Dispatch, grouped so the count metadata stays one unit.
            struct {
                // own_chunk_rank holds this rank's per-source-chunk per-dest-rank send counts so the
                // s2d build assigns slots deterministically (rank-major, sender token order) without
                // a global atomic.
                int32_t*  own_chunk_rank;             // [max_src_chunks * lsa_team_size]
                int32_t*  per_src_lteam_num_tokens;    // [num_lsa_teams] per-source-LSA-team token counts
                uint64_t* token_dst_rank_bitmap;      // [max_tokens, ceil(nRanks/64)] per-token dest-rank bitmap
                bool*     lerm_landing;               // IPC LERM landing zone (sender-written, permute input)
                // Fused meta dispatch: histogram writes this rank's own count row here (comm-free);
                // the dispatch head fans it out to the LSA peers, so UpdateHandle does no exchange.
                int32_t*  own_row;                    // [nRanks + num_experts] own count row
                // Unfused meta dispatch (caller requests layout_info): UpdateHandle AllGathers every
                // sender's count row into this per-handle buffer (PP-safe); dispatch MAP warps read it
                // in place of the fused fan-out. fused_meta_dispatch=false marks it populated this round.
                int32_t*  cached_cnt_rows;            // [nRanks * (nRanks + num_experts)]
                bool      fused_meta_dispatch;
            } dispatch_push_count;

            // EM local-fanout dup-groups (allocated iff mode == kLocalDup). Written by EM
            // scan; consumed by local_dup (post-dispatch) and local_reduce (pre-combine).
            int32_t* emuf_group_buf;
            int32_t* emuf_group_count;
            int emuf_group_stride;
            int emuf_max_groups;

            // Cached per-rank topk_idx [max_tokens, num_topk], persists across dispatch/combine.
            // Width: int32/int64 (native) outside kPullPush; uint16 snapshot under kPullPush
            // so push combine reads a handle-private slot instead of ht_buffers.global_topk_idx.
            void* topk_idx;

            // FLAT-dispatch + local-permute scratch (EM-permute path only).
            // flat2em_slot_map and token_to_recv_slot are null when em_local_permute_enabled()
            // is false. In em-permute mode sparse_to_dense_map is FLAT-shape (inner=lsa_team_size).
            //
            // Mapping from the local-node FLAT slot index to the Expert-major
            // slot in the output tensor. Shape [max_recv_tokens, top_k]
            // (invalid positions are -1). Written by em_scan_kernel during
            // UpdateHandle, read by the local-permute kernels.
            int32_t* flat2em_slot_map;
            // [num_total_attn_tokens] FLAT recv slot per global attention
            // token; -1 when the token has no local-rank hit. Populated by
            // scan_impl_flat in em-permute mode; consumed by em_scan_kernel
            // to populate the recv-indexed em_slot table.
            int32_t* token_to_recv_slot;
            // [max_recv_tokens] recv slot -> global source token id; inverse of
            // token_to_recv_slot. Null unless em_pull_enabled(). Consumed by the
            // pull dispatch kernel to read each source row over NVLink.
            int32_t* recv_slot_to_src;
            // [max_recv_tokens, num_topk] source top-k position of each flat2em hit
            // (parallel to flat2em_slot_map). Null unless em_pull. Lets pull read the
            // source's topk_weights at the correct position (order-preserving weights).
            int32_t* srcpos_map;

        } ht;
    };

    ncclEpHandle()
        : group(nullptr), layout(NCCL_EP_LAYOUT_UNSET), topk_idx(NCCL_EP_TENSOR_INIT), num_tokens(0), num_topk(0),
          num_tokens_set(false), cached_mode(false), num_scales(0), hidden_int4(0) {
        constexpr size_t union_size = std::max(sizeof(ll), sizeof(ht));
        memset(static_cast<void*>(&ll), 0, union_size);
    }

    ~ncclEpHandle() {}
};

static bool is_internode_available(ncclEpGroup_t ep_group) {
    // True when there are multiple HT outer-domain LSA teams
    return ep_group->rdma_team_size > 1;
}

// EM permute infrastructure (FLAT slot maps, per-expert offsets, EM staging).
// Shared by kLocalPermute (zero_copy != ON) and kPullPush (any zero_copy). HT-only.
static bool em_local_permute_enabled(ncclEpGroup_t group, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_EXPERT_MAJOR &&
           (group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalPermute ||
            group->ht_em_mode == ncclEpGroup::HtEmMode::kPullPush);
}
static inline bool em_local_permute_enabled(ncclEpGroup_t group, ncclEpHandle_t handle) {
    return em_local_permute_enabled(group, handle->layout);
}

// Fused-meta-dispatch (count exchange) gate: replaces the bitmap AllGather + global scan
// with a device-side histogram push and receiver-local EM rebuild. EM local-permute,
// single-LSA-team only; disabled by NCCL_EP_HT_EM_AG_SCAN_MODE. Layout sizing and handle
// activation share this gate so they never disagree.
static inline bool dispatch_push_count_capable(ncclEpGroup_t group, ncclEpLayout_t layout) {
    // kPullPush is also EM local-permute but always uses the scan path (order-preserving
    // uint16 map), so it is excluded here.
    return !nccl_ep_env_flag_on(group->env.ht_em_ag_scan_mode) &&
           !is_internode_available(group) &&
           em_local_permute_enabled(group, layout) &&
           group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalPermute;
}
static inline bool dispatch_push_count_active(ncclEpGroup_t group, ncclEpHandle_t handle) {
    return dispatch_push_count_capable(group, handle->layout);
}

// handle->ht.topk_idx mirrors the caller's native width on the scan path, but count mode's
// cache (build_count_metadata) is always widened to int64.
static inline bool topk_idx_cache_is_int32(ncclEpGroup_t group, ncclEpHandle_t handle) {
    return handle->topk_idx.datatype == ncclInt32 && !dispatch_push_count_active(group, handle);
}

// Pull EM dispatch: a variant of em_permute that pulls source rows over NVLink
// instead of push + FLAT staging. Part of the kPullPush recipe (pull dispatch +
// push combine); pull dispatch is never used on its own.
static bool em_pull_enabled(ncclEpGroup_t group, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_EXPERT_MAJOR &&
           group->ht_em_mode == ncclEpGroup::HtEmMode::kPullPush;
}
static inline bool em_pull_enabled(ncclEpGroup_t group, ncclEpHandle_t handle) {
    return em_pull_enabled(group, handle->layout);
}

// Push EM combine: a variant of em_permute that pushes each expert rank's locally
// reduced token rows into the destination attn-ranks over NVLink instead of the
// attn-ranks pulling from peer expert buffers. Part of the kPullPush recipe.
static bool em_comb_push_enabled(ncclEpGroup_t group, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_EXPERT_MAJOR &&
           group->ht_em_mode == ncclEpGroup::HtEmMode::kPullPush;
}
static inline bool em_comb_push_enabled(ncclEpGroup_t group, ncclEpHandle_t handle) {
    return em_comb_push_enabled(group, handle->layout);
}

// EM + mode == kLocalDup. HT-only by design.
static bool em_local_dup_active(ncclEpGroup_t group, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_EXPERT_MAJOR && group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalDup;
}

static bool em_staging_indexed_by_em_slot(ncclEpGroup_t group) {
    return group->ht_em_mode == ncclEpGroup::HtEmMode::kNvlinkDup ||
           group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalDup;
}

// Returns the total buffer size (in bytes) for a LL handle_mem block.
// Used by ncclEpHandleMemSize (public) and ncclEpInitHandle (internal).
static size_t ll_handle_mem_size(ncclEpGroup_t ep_group, int num_topk) {
    auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
    const size_t local_experts = static_cast<size_t>(ep_group->num_local_experts);
    const size_t nRanks = static_cast<size_t>(ep_group->nRanks);
    const size_t max_tokens = static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank);
    // Layout: nRanks per-rank counts + nRanks * max_tokens * (num_topk+1) token entries
    size_t sz_recv_src = align256(nRanks * (1 + static_cast<size_t>(num_topk + 1) * max_tokens) * sizeof(int32_t));
    size_t sz_dispatch = align256(local_experts * nRanks * sizeof(int64_t));
    return sz_recv_src + sz_dispatch;
}

// All individual buffer sizes for a HT handle_mem block plus derived totals.
// Single source of truth shared by ht_handle_mem_size() and ht_init_handle().
struct HtBlockLayout {
    // global_routing_map is group-scoped (ep_group->ht_buffers); not part of this block.
    size_t sz_r2a, sz_a2r, sz_ler, sz_ntfe;
    size_t sz_s2d, sz_rank_mask, sz_scan_tmp, sz_prob;
    size_t sz_topk_idx; // cached topk_idx (uint16 under pull-push, native int64 elsewhere)
    size_t sz_pec_active; // EM only
    size_t sz_eto; // EM only
    // Local-dup local-fanout scratch.
    size_t sz_emuf_group_buf, sz_emuf_group_count;
    int emuf_group_stride; // row width actually allocated; <= experts_per_rank
    int emuf_max_groups; // capacity in rows; kernel must not exceed
    // EM-permute scratch (handle mem).
    size_t sz_flat2em_slot_map;
    size_t sz_token_to_recv_slot;
    // Count-mode scratch (EM-permute count mode): sender chunk/rank/expert counts, token bitmap,
    // own + gathered count rows.
    size_t sz_count_scratch;
    size_t sz_recv_slot_to_src; // pull dispatch only
    size_t sz_srcpos_map;       // pull dispatch only
    size_t zero_region, no_memset_region, total;

    static HtBlockLayout compute(ncclEpGroup_t ep_group, ncclEpLayout_t layout, int num_topk = 0) {
        auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
        const int num_experts = ep_group->config.num_experts;
        const int max_tokens = ep_group->config.max_dispatch_tokens_per_rank;
        const int lsa_team_size = ep_group->lsa_team_size;
        const int rdma_team_size = ep_group->rdma_team_size;
        const int experts_per_rank = ep_group->num_local_experts;
        const int max_recv_tokens = ep_group->max_recv_tokens;
        const int padded_max_tokens = ((max_tokens + 15) / 16) * 16;
        const bool has_expert_major = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
        const bool em_permute = em_local_permute_enabled(ep_group, layout);
        // Pull recipe (dispatch + push combine). Adds recv_slot_to_src / srcpos_map,
        // and drops the FLAT dense_prob / recv_topk_weights_flat scratch the push
        // path needs but pull does not.
        const bool needs_pull_buffers = em_pull_enabled(ep_group, layout);

        // Fused-meta-dispatch sizing gate (same helper the runtime path uses). When capable,
        // weights alias the own-rank IPC table region (see ht_init_handle) and flat2em/LERM live
        // in per-handle scratch; the scan-only token_to_recv_slot is not allocated.
        const bool count_mode_capable = dispatch_push_count_capable(ep_group, layout);

        HtBlockLayout L = {};
        L.sz_r2a = align256(static_cast<size_t>(rdma_team_size) * padded_max_tokens * sizeof(bool));
        L.sz_a2r =
            (rdma_team_size > 1) ? align256(static_cast<size_t>(max_tokens) * (rdma_team_size - 1) * sizeof(bool)) : 0;
        // Count mode persists LERM as a packed bitmap (ceil(epr/64) u64 words per slot); scan
        // mode as one bool per expert. Per-handle so combine reads a private copy (the FWD
        // permute writes it here from the sender-written IPC landing zone).
        L.sz_ler = count_mode_capable ?
                       align256(static_cast<size_t>(ep_group->max_recv_tokens) *
                                nccl_ep::bit_words(experts_per_rank) * sizeof(uint64_t)) :
                       align256(static_cast<size_t>(ep_group->max_recv_tokens) * experts_per_rank * sizeof(bool));
        L.sz_ntfe = align256(sizeof(int32_t));
        // S2D inner_dim: num_topk for nvlink_dup/local_dup EM (packed rank/slot); lsa_team_size
        // for FLAT and for EM-permute (unified FLAT-shape s2d — em_scan_kernel's
        // EM-shape writes are suppressed in em-permute mode).
        const int s2d_inner_dim = (has_expert_major && !em_permute) ? num_topk : lsa_team_size;
        // pull-push never touches s2d (dispatch passes nullptr, push combine ignores it).
        L.sz_s2d = needs_pull_buffers
                       ? 0
                       : align256(static_cast<size_t>(rdma_team_size) * max_tokens * s2d_inner_dim * sizeof(int32_t));
        L.sz_rank_mask = align256(
            static_cast<size_t>(rdma_team_size) * max_tokens * lsa_team_size *
            nccl_ep::ht::get_rank_mask_elem_size(lsa_team_size));
        // Size for device_sm_count blocks: the upper bound on the preprocessing
        // scan's block count (ep_group->preprocess_num_sms, incl. any
        // NCCL_EP_PREPROCESS_NUM_SMS override) so the buffer always fits.
        L.sz_scan_tmp = align256(
            nccl_ep::ht::get_preprocessing_scan_tmp_size(
                static_cast<int>(ep_group->device_sm_count),
                lsa_team_size));
        // dense_prob_buffer: unused under pull dispatch + push combine (dispatch
        // pulls weights per source token; push combine uses sparse-direct prob
        // staging). Single-node non-pull only.
        L.sz_prob = (!is_internode_available(ep_group) && !needs_pull_buffers) ?
                        align256(static_cast<size_t>(max_tokens) * num_experts * sizeof(float)) :
                        0;
        // Pull-push stores a uint16 snapshot for push combine; other modes cache the native
        // int32/int64 for the FLAT dense-prob rebuild and BWD dense-to-sparse scatter.
        const size_t topk_idx_elem = needs_pull_buffers ? sizeof(uint16_t) : sizeof(int64_t);
        L.sz_topk_idx = (num_topk > 0) ? align256(static_cast<size_t>(max_tokens) * num_topk * topk_idx_elem) : 0;
        L.sz_pec_active = has_expert_major ? align256(static_cast<size_t>(experts_per_rank) * sizeof(int32_t)) : 0;
        // [experts_per_rank + 1]: em_scan_kernel publishes the EM-padded total at `experts_per_rank` index.
        L.sz_eto = has_expert_major ? align256(static_cast<size_t>(experts_per_rank + 1) * sizeof(int64_t)) : 0;
        // EM local-fanout dup-groups. Bounds: num_groups <= max_recv_tokens / 2
        // (each multi-hit group occupies >= 2 em_slots); row width <= min(num_topk, experts_per_rank).
        const bool emuf_enabled = has_expert_major && ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalDup;
        const int emuf_row_width = (num_topk > 0) ? std::min(num_topk, experts_per_rank) : experts_per_rank;
        const size_t emuf_max_groups = static_cast<size_t>(ep_group->max_recv_tokens) / 2;
        const size_t emuf_group_entries = emuf_enabled ? emuf_max_groups * static_cast<size_t>(emuf_row_width) : 0;
        L.emuf_group_stride = emuf_enabled ? emuf_row_width : 0;
        L.emuf_max_groups = emuf_enabled ? static_cast<int>(emuf_max_groups) : 0;
        L.sz_emuf_group_buf = align256(emuf_group_entries * sizeof(int32_t));
        L.sz_emuf_group_count = emuf_enabled ? align256(sizeof(int32_t)) : 0;
        // Recv-token row capacity: one row per received token, before the per-token
        // num_topk fan-out. Eager mode sizes max_recv_tokens to the EM-copy capacity
        // (a num_topk factor larger), so the pull token-indexed maps cap at the token
        // count to avoid an extra num_topk factor.
        const size_t max_flat_recv_tokens =
            std::min<size_t>(static_cast<size_t>(max_recv_tokens),
                             static_cast<size_t>(max_tokens) * lsa_team_size * rdma_team_size);
        // EM-permute scratch: flat2em is per-handle for scan, count, and pull. recv-slot
        // weights are group-shared (ht_buffers.count_scratch.recv_topk_weights_flat). pull-push caps rows
        // at the recv-token count; the other EM paths keep full max_recv_tokens sizing.
        // TODO(em-local-permute): baseline over-sizes by a num_topk factor; cap at
        // max_flat_recv_tokens once its flat2em indexing is confirmed recv-token-bounded.
        L.sz_flat2em_slot_map =
            (em_permute && num_topk > 0)
                ? align256((needs_pull_buffers ? max_flat_recv_tokens : static_cast<size_t>(max_recv_tokens)) *
                           num_topk * sizeof(int32_t))
                : 0;
        // [num_total_attn_tokens] FLAT recv slot per global attention token, filled by
        // scan_impl_flat (scan and pull paths; count never runs the scan that fills it).
        // num_total_attn_tokens = max_tokens * lsa_team_size * rdma_team_size.
        L.sz_token_to_recv_slot =
            (em_permute && !count_mode_capable) ?
                align256(static_cast<size_t>(max_tokens) * lsa_team_size * rdma_team_size * sizeof(int32_t)) :
                0;
        // Count-mode scratch (EM-permute): sender-local per-chunk per-dest-rank counts and the
        // token bitmap, followed by the own/gathered count rows. The receiver count-landing rows
        // live in the IPC table region. Only allocated in count mode.
        const size_t max_src_chunks = (static_cast<size_t>(max_tokens) + ep_group->ht_tokens_per_chunk - 1) /
                                  ep_group->ht_tokens_per_chunk;
        // token_dst_rank_bitmap is ceil(nRanks/64) u64 words per token (matches the histogram's
        // hit_words = (lsa_team_size + 63) / 64); >1 word once nRanks exceeds 64.
        const size_t cnt_hit_words = nccl_ep::ceil_div<size_t>(static_cast<size_t>(ep_group->nRanks), 64);
        L.sz_count_scratch =
            count_mode_capable
                ? align256((max_src_chunks * lsa_team_size + rdma_team_size) * sizeof(int32_t)) +
                      align256(static_cast<size_t>(max_tokens) * cnt_hit_words * sizeof(uint64_t)) +
                      // Per-handle own count row [cnt_rank(nRanks) | cnt_expert(num_experts)]:
                      // the histogram writes it here (comm-free), the dispatch head fans it out.
                      align256(static_cast<size_t>(ep_group->nRanks + num_experts) * sizeof(int32_t)) +
                      // Per-handle AllGather landing for every sender's count row (unfused path).
                      align256(static_cast<size_t>(ep_group->nRanks) *
                               (ep_group->nRanks + num_experts) * sizeof(int32_t))
                : 0;
        // Pull dispatch inverse map, indexed by recv slot. Written only for kept
        // slots by the scan; no zero-init needed (unread slots aren't pulled).
        L.sz_recv_slot_to_src =
            needs_pull_buffers ? align256(max_flat_recv_tokens * sizeof(int32_t)) : 0;
        L.sz_srcpos_map =
            (needs_pull_buffers && num_topk > 0)
                ? align256(max_flat_recv_tokens * num_topk * sizeof(int32_t))
                : 0;
        L.zero_region = L.sz_r2a + L.sz_a2r + L.sz_ler + L.sz_ntfe;
        L.no_memset_region = L.sz_rank_mask + L.sz_scan_tmp + L.sz_prob + L.sz_topk_idx + L.sz_pec_active + L.sz_eto +
                             L.sz_emuf_group_buf + L.sz_emuf_group_count + L.sz_flat2em_slot_map +
                             L.sz_token_to_recv_slot + L.sz_count_scratch + L.sz_recv_slot_to_src +
                             L.sz_srcpos_map;
        L.total = L.zero_region + L.sz_s2d + L.no_memset_region;
        return L;
    }
};

static size_t ht_handle_mem_size(ncclEpGroup_t ep_group, ncclEpLayout_t layout, int num_topk) {
    return HtBlockLayout::compute(ep_group, layout, num_topk).total;
}

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    size_t* size_out,
    int num_topk) {
    assert(ep_group != nullptr && size_out != nullptr);
    if (config != nullptr) {
        EP_VALIDATE_STRUCT(config, NCCL_EP_HANDLE_CONFIG);
    }
    EP_HOST_ASSERT(layout != NCCL_EP_LAYOUT_UNSET && "ncclEpHandleMemSize: layout must be set explicitly");
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        assert(num_topk > 0 && "HT mode requires num_topk > 0 for ncclEpHandleMemSize");
        *size_out = ht_handle_mem_size(ep_group, layout, num_topk);
    } else if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        assert(num_topk > 0 && "LL mode requires num_topk > 0 for ncclEpHandleMemSize");
        *size_out = ll_handle_mem_size(ep_group, num_topk);
    } else {
        return ncclInvalidUsage;
    }
    return ncclSuccess;
}

// Collectively (re-)size rdma_buffer to `new_size`. Two modes:
//   - First allocation (rdma_buffer == nullptr): allocate, register, stash
//     window handle in a newly-allocated device slot.
//   - Grow (rdma_buffer != nullptr): cross-rank fence, deregister window,
//     free, reallocate, re-register, update existing device slot.
// In both cases the buffer is zeroed and rdma_buffer_size_alloc is updated.
// Must be called by all ranks in the comm with the same new_size.
static ncclResult_t ll_resize_rdma_buffer(ncclEpGroup_t ep_group, size_t new_size) {
    EP_HOST_ASSERT(new_size > ep_group->rdma_buffer_size_alloc);
    // With NCCL_EP_AUTO, config.rdma_buffer_size is 0 (the AUTO sentinel) and
    // imposes no cap. With a user-specified value, the caller (ll_init_handle)
    // rejects layouts that don't fit before reaching here, and the group-time
    // path passes exactly config.rdma_buffer_size.
    EP_HOST_ASSERT(ep_group->config.rdma_buffer_size == NCCL_EP_AUTO || new_size <= ep_group->config.rdma_buffer_size);

    const bool first_alloc = (ep_group->rdma_buffer == nullptr);

    if (!first_alloc) {
        // Cross-rank fence before teardown: cudaDeviceSynchronize drains
        // local work, but a remote rank may still be touching this rank's
        // window via RDMA. ncclMemFree/ncclCommWindowDeregister/
        // ncclMemAlloc/ncclCommWindowRegister are themselves collective,
        // so no further barrier is needed afterward.
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        NCCL_CHECK_RESULT(ncclBarrier(ep_group->comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ncclWindow_t win_host;
        CUDA_CHECK(cudaMemcpy(&win_host, ep_group->nccl_wins, sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, win_host));
        NCCL_CHECK_RESULT(ncclMemFree(ep_group->rdma_buffer));
        ep_group->rdma_buffer = nullptr;
    }

    NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->rdma_buffer, new_size));
    CUDA_CHECK(cudaMemset(ep_group->rdma_buffer, 0, new_size));
    ep_group->rdma_buffer_size_alloc = new_size;

    ncclWindow_t win_host;
    NCCL_CHECK_RESULT(ncclCommWindowRegister(ep_group->comm, ep_group->rdma_buffer, new_size, &win_host, 0));
    if (first_alloc) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_wins), sizeof(ncclWindow_t)));
    }
    CUDA_CHECK(cudaMemcpy(ep_group->nccl_wins, &win_host, sizeof(ncclWindow_t), cudaMemcpyHostToDevice));

    return ncclSuccess;
}

static ncclResult_t validate_ll_geometry(const ncclEpGroup_t ep_group, int num_topk) {
    if (num_topk <= 0 || num_topk > ep_group->config.num_experts || num_topk > MAX_NUM_TOPK) {
        fprintf(stderr,
                "ncclEpInitHandle: LL top-k %d must be in [1, min(num_experts=%u, MAX_NUM_TOPK=%d)]\n",
                num_topk, ep_group->config.num_experts, MAX_NUM_TOPK);
        return ncclInvalidArgument;
    }

    const int num_device_sms = static_cast<int>(ep_group->dispatch_num_sms);
    if (num_device_sms <= 0) return ncclInvalidUsage;
    const int num_warp_groups = (ep_group->config.num_experts + num_device_sms - 1) / num_device_sms;
    if (num_warp_groups <= 0 || num_warp_groups > nccl_ep::ll::kLlDispatchMaxWarpGroups) {
        fprintf(stderr, "ncclEpInitHandle: LL warp-group geometry is unsupported: num_experts=%u, dispatch_sms=%d, warp_groups=%d\n",
                ep_group->config.num_experts, num_device_sms, num_warp_groups);
        return ncclInvalidUsage;
    }
    const int num_warps_per_group = nccl_ep::ll::combine_smem::kWarpSize / num_warp_groups;
    const int num_warps = num_warp_groups * num_warps_per_group;
    if (num_warps_per_group <= 0 || num_topk + nccl_ep::ll::kLlDispatchControlWarps > num_warps) {
        fprintf(stderr,
                "ncclEpInitHandle: LL top-k geometry is unsupported: top-k=%d, num_experts=%u, dispatch_sms=%d, "
                "warp_groups=%d, launched_warps=%d, forwarding_warps=%d\n",
                num_topk, ep_group->config.num_experts, num_device_sms, num_warp_groups, num_warps,
                std::max(0, num_warps - nccl_ep::ll::kLlDispatchControlWarps));
        return ncclInvalidUsage;
    }
    return ncclSuccess;
}

static ncclResult_t
ll_init_handle(ncclEpHandle_t handle, ncclEpGroup_t ep_group, const ncclEpTensor_t* handle_mem, int num_topk) {
    NCCLCHECK(validate_ll_geometry(ep_group, num_topk));

    auto layout = nccl_ep::LowLatencyLayout(
        ep_group->config.max_dispatch_tokens_per_rank,
        ep_group->config.max_token_bytes,
        ep_group->nRanks,
        ep_group->config.num_experts,
        num_topk,
        handle->layout);

    // Ensure rdma_buffer is large enough for this handle's actual layout.
    // - NCCL_EP_AUTO: lazily allocate (first handle) or grow (later handle
    //   needing more, e.g. different layout or larger num_topk). Stored
    //   layouts on other live handles are pure offsets, so reallocation
    //   does not invalidate them.
    // - User-specified size: the buffer was already allocated to that exact
    //   value at group create. Reject the handle if it doesn't fit.
    if (layout.total_bytes > ep_group->rdma_buffer_size_alloc) {
        if (ep_group->config.rdma_buffer_size != NCCL_EP_AUTO) {
            fprintf(
                stderr,
                "ncclEpInitHandle: requested layout needs %zu bytes but user-specified "
                "rdma_buffer_size is %zu bytes; either increase rdma_buffer_size or use "
                "NCCL_EP_AUTO to allow lazy sizing\n",
                layout.total_bytes,
                static_cast<size_t>(ep_group->config.rdma_buffer_size));
            return ncclInvalidUsage;
        }
        const size_t new_size = ((layout.total_bytes + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES) *
                                NUM_BUFFER_ALIGNMENT_BYTES;
        NCCLCHECK(ll_resize_rdma_buffer(ep_group, new_size));
    }
    assert(layout.total_bytes <= ep_group->rdma_buffer_size_alloc);

    if (handle_mem != nullptr) {
        assert(handle_mem->ndim == 1 && handle_mem->datatype == ncclUint8);
        assert(
            handle_mem->sizes[0] >= ll_handle_mem_size(ep_group, num_topk) &&
            "handle_mem too small; use ncclEpHandleMemSize to query required size");
        handle->ll.handle_mem = handle_mem->data;
        handle->ll.owns_handle_mem = false;
    } else {
        CUDA_CHECK(ep_group->alloc.alloc_fn(
            &handle->ll.handle_mem,
            ll_handle_mem_size(ep_group, num_topk),
            ep_group->alloc.context));
        handle->ll.owns_handle_mem = true;
    }
    char* base = static_cast<char*>(handle->ll.handle_mem);

    const size_t recv_src_count =
        static_cast<size_t>(ep_group->nRanks) *
        (1 + static_cast<size_t>(num_topk + 1) * ep_group->config.max_dispatch_tokens_per_rank);
    {
        handle->ll.expert_recv_source_indices_sizes[0] = recv_src_count;
        handle->ll.expert_recv_source_indices = (ncclEpTensor_t){
            NCCL_EP_TENSOR_INIT_INLINE,
            .ndim = 1,
            .datatype = ncclInt32,
            .data = base,
            .sizes = handle->ll.expert_recv_source_indices_sizes,
        };
    }

    {
        auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
        const size_t recv_src_bytes = align256(recv_src_count * sizeof(int32_t));
        handle->ll.expert_dispatch_layout_sizes[0] = static_cast<size_t>(ep_group->num_local_experts);
        handle->ll.expert_dispatch_layout_sizes[1] = static_cast<size_t>(ep_group->nRanks);
        handle->ll.expert_dispatch_layout = (ncclEpTensor_t){
            NCCL_EP_TENSOR_INIT_INLINE,
            .ndim = 2,
            .datatype = ncclInt64,
            .data = base + recv_src_bytes,
            .sizes = handle->ll.expert_dispatch_layout_sizes,
        };
    }
    handle->num_topk = num_topk;
    handle->ll.layout = layout;
    return ncclSuccess;
}

// Lays out the count-mode per-handle scratch fields (see ht.dispatch_push_count docstring) at ptr + offset,
// or nulls them out when the handle has no count scratch (sz_count_scratch == 0).
static void ht_assign_count_scratch(ncclEpHandle_t handle, ncclEpGroup_t ep_group, char* ptr, size_t offset,
                                     size_t sz_count_scratch) {
    if (sz_count_scratch == 0) {
        handle->ht.dispatch_push_count.own_chunk_rank = nullptr;
        handle->ht.dispatch_push_count.per_src_lteam_num_tokens = nullptr;
        handle->ht.dispatch_push_count.token_dst_rank_bitmap = nullptr;
        handle->ht.dispatch_push_count.own_row = nullptr;
        handle->ht.dispatch_push_count.cached_cnt_rows = nullptr;
        handle->ht.dispatch_push_count.fused_meta_dispatch = true;
        handle->ht.dispatch_push_count.lerm_landing = nullptr;
        return;
    }
    const int lsa_team_size = ep_group->lsa_team_size;
    const int num_experts = ep_group->config.num_experts;
    const int num_lsa_teams = ep_group->rdma_team_size;
    const size_t max_chunks_per_rank =
        (static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) + ep_group->ht_tokens_per_chunk - 1) /
        ep_group->ht_tokens_per_chunk;
    int32_t* cs = reinterpret_cast<int32_t*>(ptr + offset);
    handle->ht.dispatch_push_count.own_chunk_rank = cs;
    handle->ht.dispatch_push_count.per_src_lteam_num_tokens = handle->ht.dispatch_push_count.own_chunk_rank + max_chunks_per_rank * lsa_team_size;
    const size_t cnt_ints_bytes =
        ((max_chunks_per_rank * lsa_team_size + num_lsa_teams) * sizeof(int32_t) + 255) & ~size_t(255);
    handle->ht.dispatch_push_count.own_row = reinterpret_cast<int32_t*>(ptr + offset + cnt_ints_bytes);
    const size_t own_row_bytes =
        (static_cast<size_t>(ep_group->nRanks + num_experts) * sizeof(int32_t) + 255) & ~size_t(255);
    handle->ht.dispatch_push_count.token_dst_rank_bitmap =
        reinterpret_cast<uint64_t*>(ptr + offset + cnt_ints_bytes + own_row_bytes);
    // ceil(nRanks/64) u64 words per token, matching the histogram's hit_words.
    const size_t bitmap_words = nccl_ep::ceil_div<size_t>(static_cast<size_t>(ep_group->nRanks), 64);
    const size_t bitmap_bytes =
        (static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * bitmap_words * sizeof(uint64_t) +
         255) &
        ~size_t(255);
    handle->ht.dispatch_push_count.cached_cnt_rows =
        reinterpret_cast<int32_t*>(ptr + offset + cnt_ints_bytes + own_row_bytes + bitmap_bytes);
    handle->ht.dispatch_push_count.fused_meta_dispatch = true;
}

static ncclResult_t
ht_init_handle(ncclEpHandle_t handle, ncclEpGroup_t ep_group, const ncclEpTensor_t* handle_mem, int num_topk) {
    assert(ep_group->config.max_dispatch_tokens_per_rank > 0 && "HT requires max_dispatch_tokens_per_rank > 0");

    if(num_topk <= 0) {
        fprintf(stderr, "HT mode requires num_topk > 0 (pass top_k to ncclEpInitHandle)\n");
        return ncclInvalidUsage;
    }

    // Eager groups size internal buffers from config.num_topk; enforce it as the
    // per-handle upper bound, and require it for Expert-Major (per-expert slot
    // expansion is unbounded without it).
    if (ep_group->config.num_topk > 0) {
        if(static_cast<unsigned int>(num_topk) > ep_group->config.num_topk) {
            fprintf(stderr, "NCCL EP: num_topk exceeds ncclEpGroupConfig_t::num_topk\n");
            return ncclInvalidUsage;
        }
    } else if (ep_group->eager_mode && handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        fprintf(stderr,
            "NCCL EP: eager mode (max_recv_tokens_per_rank = NCCL_EP_AUTO) requires "
            "ncclEpGroupConfig_t::num_topk for the Expert-Major layout\n");
        return ncclInvalidUsage;
    }
    if (em_local_permute_enabled(ep_group, handle->layout)) {
        // EM local-permute packs each received token's top-k into a fixed-width row and the
        // group recv_topk_weights_flat is sized by kEpCountMaxTopk; a larger num_topk overruns it.
        if (num_topk > static_cast<int>(nccl_ep::ht::kEpCountMaxTopk)) {
            fprintf(stderr,
                "NCCL EP: HT Expert-Major local-permute requires num_topk <= %d\n",
                nccl_ep::ht::kEpCountMaxTopk);
            return ncclInvalidUsage;
        }
    }
    // Count mode's single-LSA-team precondition is already enforced by dispatch_push_count_capable()
    // (which includes !is_internode_available); internode EM falls back to scan through the same
    // gate, so no redundant handle-init assert is needed here.
    // kNvlinkDup/kLocalDup write the staging buffer in the same per-expert
    // expanded shape as the dispatch output (each token can occupy up to
    // num_topk slots), so max_recv_tokens must cover the worst case
    // nRanks * max_dispatch_tokens_per_rank * num_topk.
    if (em_staging_indexed_by_em_slot(ep_group) && ep_group->config.zero_copy != NCCL_EP_ZERO_COPY_ON) {
        const size_t needed =
            static_cast<size_t>(ep_group->nRanks) * ep_group->config.max_dispatch_tokens_per_rank * num_topk;
        EP_HOST_ASSERT(
            static_cast<size_t>(ep_group->max_recv_tokens) >= needed &&
            "ncclEpInitHandle: max_recv_tokens too small for em_slot staging");
    }
    handle->num_topk = num_topk;
    const auto L = HtBlockLayout::compute(ep_group, handle->layout, num_topk);

    if (handle_mem != nullptr) {
        assert(handle_mem->ndim == 1 && handle_mem->datatype == ncclUint8);
        assert(
            handle_mem->sizes[0] >= L.total && "handle_mem too small; use ncclEpHandleMemSize to query required size");
        ncclEpTensor_t handle_mem_local;
        const ncclEpTensor_t* resolved_handle_mem;
        NCCLCHECK(resolveTensorWindowBinding(ep_group, handle_mem, &handle_mem_local, 0, &resolved_handle_mem));
        handle->ht.preprocessing_block = resolved_handle_mem->data;
        handle->ht.owns_handle_mem = false;
    } else {
        CUDA_CHECK(ep_group->alloc.alloc_fn(&handle->ht.preprocessing_block, L.total, ep_group->alloc.context));
        handle->ht.owns_handle_mem = true;
    }
    handle->ht.preprocessing_zero_region_size = L.zero_region;
    handle->ht.preprocessing_s2d_size = L.sz_s2d;

    char* ptr = static_cast<char*>(handle->ht.preprocessing_block);
    size_t offset = 0;

    // global_routing_map: read directly from ep_group->ht_buffers (not duplicated on the handle).
    handle->ht.rdma_to_attn_map = reinterpret_cast<bool*>(ptr + offset);
    offset += L.sz_r2a;
    handle->ht.attn_to_rdma_map = (ep_group->nNodes > 1) ? reinterpret_cast<bool*>(ptr + offset) : nullptr;
    offset += L.sz_a2r;
    handle->ht.local_expert_routing_map = reinterpret_cast<bool*>(ptr + offset);
    offset += L.sz_ler;
    handle->ht.num_tokens_for_experts = reinterpret_cast<int32_t*>(ptr + offset);
    offset += L.sz_ntfe;
    // --- end of zero_region (memset 0x00) ---
    handle->ht.sparse_to_dense_map = (L.sz_s2d > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_s2d;
    // --- end of s2d region (memset 0xFF) ---
    handle->ht.token_rank_mask = ptr + offset;
    offset += L.sz_rank_mask;
    handle->ht.preprocessing_scan_tmp = reinterpret_cast<void*>(ptr + offset);
    offset += L.sz_scan_tmp;
    // Null unless allocated (internode uses the group buffer, wired below; pull
    // drops it). Guard on the region size so the offset math stays consistent.
    handle->ht.dense_prob_buffer = (L.sz_prob > 0) ? reinterpret_cast<float*>(ptr + offset) : nullptr;
    offset += L.sz_prob;
    handle->ht.topk_idx = (L.sz_topk_idx > 0) ? reinterpret_cast<void*>(ptr + offset) : nullptr;
    offset += L.sz_topk_idx;
    // EM remap counts/offsets live in handle_mem (EM only; FLAT must not read them).
    handle->ht.per_expert_counts_active =
        (L.sz_pec_active > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_pec_active;
    handle->ht.expert_token_offsets = (L.sz_eto > 0) ? reinterpret_cast<int64_t*>(ptr + offset) : nullptr;
    offset += L.sz_eto;
    if (L.sz_emuf_group_buf > 0) {
        handle->ht.emuf_group_buf = reinterpret_cast<int32_t*>(ptr + offset);
        offset += L.sz_emuf_group_buf;
        handle->ht.emuf_group_count = reinterpret_cast<int32_t*>(ptr + offset);
        offset += L.sz_emuf_group_count;
        handle->ht.emuf_group_stride = L.emuf_group_stride;
        handle->ht.emuf_max_groups = L.emuf_max_groups;
    } else {
        handle->ht.emuf_group_buf = nullptr;
        handle->ht.emuf_group_count = nullptr;
        handle->ht.emuf_group_stride = 0;
        handle->ht.emuf_max_groups = 0;
    }
    handle->ht.flat2em_slot_map =
        (L.sz_flat2em_slot_map > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_flat2em_slot_map;
    handle->ht.token_to_recv_slot =
        (L.sz_token_to_recv_slot > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_token_to_recv_slot;
    // Layout order matters: own_row must follow own_chunk_rank/per_src_lteam_num_tokens so
    // ht_update_handle_count_mode can zero all three with one memset (see its call site).
    ht_assign_count_scratch(handle, ep_group, ptr, offset, L.sz_count_scratch);
    offset += L.sz_count_scratch;
    // lerm_landing aliases the IPC table (cross-rank LERM rows); the permute copies it into
    // local_expert_routing_map so combine sees a private copy.
    if (L.sz_count_scratch > 0) {
        handle->ht.dispatch_push_count.lerm_landing =
            reinterpret_cast<bool*>(ep_group->ht_buffers.expert_output_meta_tables);
    }
    handle->ht.recv_slot_to_src =
        (L.sz_recv_slot_to_src > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_recv_slot_to_src;
    handle->ht.srcpos_map =
        (L.sz_srcpos_map > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_srcpos_map;
    handle->ht.dispatch_output_per_expert_alignment = 0;

    if (is_internode_available(ep_group)) {
        handle->ht.dense_prob_buffer = ep_group->ht_buffers.dense_prob_buffer;
        handle->ht.token_staging_buffer = ep_group->ht_buffers.token_staging_buffer;
    } else {
        handle->ht.token_staging_buffer = nullptr;
    }
    return ncclSuccess;
}

// No collective; allocates routing buffers only.
// handle_mem == nullptr → alloc_fn owns the block; freed on destroy.
// handle_mem != nullptr → wraps caller buffer; destroy frees only the struct.
ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t* out_handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    int num_topk,
    const ncclEpTensor_t* handle_mem) {
    assert(ep_group != nullptr && out_handle != nullptr);
    assert(ep_group->comm != nullptr);
    if (config != nullptr) {
        EP_VALIDATE_STRUCT(config, NCCL_EP_HANDLE_CONFIG);
    }
    ncclEpHandleConfig_t parsed_config = NCCL_EP_HANDLE_CONFIG_INIT;
    if (config != nullptr) {
        parsed_config = epDecodeStruct(config, NCCL_EP_HANDLE_CONFIG_INIT);
        config = &parsed_config;
    }
    handle_mem = tensor_ptr(handle_mem); // NULL passthrough; otherwise validates magic
    EP_HOST_ASSERT(layout != NCCL_EP_LAYOUT_UNSET && "ncclEpInitHandle: layout must be set explicitly");
    EP_HOST_ASSERT(
        !(ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && layout != NCCL_EP_LAYOUT_FLAT &&
          layout != NCCL_EP_LAYOUT_EXPERT_MAJOR) &&
        "ncclEpInitHandle: HT mode supports flat and expert-major layouts");
    // Pull dispatch + push combine only stages expert-major traffic; its shared
    // intra-LSA buffers are sized for that path, so reject FLAT handles up-front.
    EP_HOST_ASSERT(
        !(ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kPullPush &&
          layout != NCCL_EP_LAYOUT_EXPERT_MAJOR) &&
        "ncclEpInitHandle: NCCL_EP_HT_EM_PULL_PUSH requires expert-major layout");
    EP_HOST_ASSERT(
        !(ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY && layout != NCCL_EP_LAYOUT_EXPERT_MAJOR &&
          layout != NCCL_EP_LAYOUT_RANK_MAJOR) &&
        "ncclEpInitHandle: LL mode supports only expert-major and rank-major layouts");
    assert(ep_group->config.num_experts > 0);
    assert(ep_group->config.num_experts % ep_group->nRanks == 0);

    // Validate EM padding alignment up-front (pow2 required) before any allocation.
    const bool is_ht_em =
        ep_group->config.algorithm != NCCL_EP_ALGO_LOW_LATENCY && layout == NCCL_EP_LAYOUT_EXPERT_MAJOR;
    const size_t em_align = (is_ht_em && config && config->dispatch_output_per_expert_alignment > 1) ?
                                config->dispatch_output_per_expert_alignment :
                                1;
    assert((em_align & (em_align - 1)) == 0 && "dispatch_output_per_expert_alignment must be a power of two");

    *out_handle = new ncclEpHandle();
    ncclEpHandle_t handle = *out_handle;
    handle->group = ep_group;
    handle->layout = layout;

    ncclResult_t res;
    if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        res = ll_init_handle(handle, ep_group, handle_mem, num_topk);
    } else {
        res = ht_init_handle(handle, ep_group, handle_mem, num_topk);
        if (res == ncclSuccess && is_ht_em) {
            handle->ht.dispatch_output_per_expert_alignment = em_align;
        }
    }

    return res;
}

// Fused-meta-dispatch UpdateHandle path: publishes this rank's count row via the device
// histogram; the unfused path (caller supplies layout_info) AllGathers rows here instead.
static ncclResult_t ht_update_handle_count_mode(
    ncclEpGroup_t ep_group,
    ncclEpHandle_t handle,
    bool out_is_int64,
    void* padded_out_counts,
    void* out_offsets,
    void* recv_total_counter,
    int32_t* per_expert_counts_device,
    int max_recv_tpr,
    cudaStream_t stream) {
    const int num_experts = ep_group->config.num_experts;
    const int max_tokens = ep_group->config.max_dispatch_tokens_per_rank;
    const int num_lsa_teams = ep_group->rdma_team_size;
    const int experts_per_rank = ep_group->num_local_experts;

    // Histogram writes this rank's send counts into its own count row (comm-free); the
    // dispatch head fans it out to LSA peers, which rebuild the EM tables from the landed rows.
    const int lsa_team_size = ep_group->lsa_team_size;
    const size_t count_row_ints = (size_t)ep_group->nRanks + (size_t)num_experts;
    // Own row lands in per-handle scratch (comm-free, multi-handle-safe); fan-out happens at
    // the dispatch head, not here.
    int32_t* cnt_rank_pub = handle->ht.dispatch_push_count.own_row;
    int32_t* cnt_expert_pub = cnt_rank_pub + ep_group->nRanks;
    // Per-chunk per-dest-rank counts give the dispatch MAP warp its deterministic chunk slot
    // bases.
    const int tokens_per_chunk = ep_group->ht_tokens_per_chunk;
    const int padded_max_tokens = nccl_ep::align(max_tokens, 16);
    const int r2a_offset = ep_group->rdma_rank * padded_max_tokens;
    // The histogram writes its own-LSA-team outputs to per-handle scratch, so batched-ahead
    // UpdateHandles can't clobber another handle's in-flight metadata.
    int32_t* chunk_rank_own = handle->ht.dispatch_push_count.own_chunk_rank;
    uint64_t* bitmap_own = handle->ht.dispatch_push_count.token_dst_rank_bitmap;
    int32_t* stream_ntok = handle->ht.dispatch_push_count.per_src_lteam_num_tokens;
    // One memset clears the contiguous zero-required span [own_chunk_rank | per_src_lteam | own_row]:
    // the histogram accumulates into own_chunk_rank and own_row (both need zeroing), while the
    // bitmap that follows is written per token and needs none.
    const size_t count_zero_bytes =
        (size_t)(reinterpret_cast<const uint8_t*>(cnt_rank_pub) -
                 reinterpret_cast<const uint8_t*>(chunk_rank_own)) +
        count_row_ints * sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(chunk_rank_own, 0, count_zero_bytes, stream));
    // topk_idx is read in its caller-native width; the cache it writes is always int64.
    if (handle->topk_idx.datatype == ncclInt32) {
        nccl_ep::ht::build_count_metadata(
            static_cast<const int32_t*>(handle->topk_idx.data),
            static_cast<int64_t*>(handle->ht.topk_idx),
            handle->num_tokens, handle->num_topk, experts_per_rank, ep_group->nRanks,
            num_experts, cnt_rank_pub, cnt_expert_pub,
            tokens_per_chunk, chunk_rank_own,
            lsa_team_size, num_lsa_teams,
            bitmap_own,
            stream_ntok,
            ep_group->rdma_rank, handle->ht.rdma_to_attn_map, r2a_offset,
            ep_group->preprocess_num_sms, stream);
    } else {
        nccl_ep::ht::build_count_metadata(
            static_cast<const int64_t*>(handle->topk_idx.data),
            static_cast<int64_t*>(handle->ht.topk_idx),
            handle->num_tokens, handle->num_topk, experts_per_rank, ep_group->nRanks,
            num_experts, cnt_rank_pub, cnt_expert_pub,
            tokens_per_chunk, chunk_rank_own,
            lsa_team_size, num_lsa_teams,
            bitmap_own,
            stream_ntok,
            ep_group->rdma_rank, handle->ht.rdma_to_attn_map, r2a_offset,
            ep_group->preprocess_num_sms, stream);
    }

    // Unfused path (caller supplies layout_info): AllGather count rows now and derive the
    // caller tensors immediately; dispatch MAP warps then read these gathered rows instead
    // of the fused per-peer landing.
    const bool provide_layout_info =
        (padded_out_counts != nullptr) || (out_offsets != nullptr) || (recv_total_counter != nullptr);
    // Eager mode needs the recv count at UpdateHandle time, before the fused path would
    // publish it, so it always runs the unfused count exchange.
    const bool run_unfused_count =
        provide_layout_info || ep_group->eager_mode || nccl_ep_env_flag_on(ep_group->env.ht_em_count_unfused);
    handle->ht.dispatch_push_count.fused_meta_dispatch = true;
    if (run_unfused_count) {
        NCCL_CHECK_RESULT(ncclAllGather(
            cnt_rank_pub,
            handle->ht.dispatch_push_count.cached_cnt_rows,
            count_row_ints,
            ncclInt32,
            ep_group->comm,
            stream));
        // Derive the caller layout tensors from the gathered rows. Count mode is EM-only
        // (gated by em_local_permute_enabled). Counts/offsets exclude dropped tokens;
        // recv_total_counter reports the pre-drop total.
        nccl_ep::ht::compute_layout_info(
            handle->ht.dispatch_push_count.cached_cnt_rows,
            ep_group->nRanks,
            num_experts,
            experts_per_rank,
            ep_group->rank,
            static_cast<int>(handle->ht.dispatch_output_per_expert_alignment),
            max_recv_tpr,
            ep_group->config.overflow_policy == NCCL_EP_OVERFLOW_DROP,
            out_is_int64,
            handle->ht.expert_token_offsets,
            padded_out_counts,
            out_offsets,
            per_expert_counts_device,
            recv_total_counter,
            stream);
        handle->ht.dispatch_push_count.fused_meta_dispatch = false;
    }
    // Per-expert offsets/counts are receiver-local (known only after dispatch); the caller
    // tensors are wired from ncclEpDispatch's layout_info and written by the permute post-build.
    return ncclSuccess;
}

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    cudaStream_t stream) {
    assert(handle != nullptr);
    if (layout_info != nullptr) {
        EP_VALIDATE_STRUCT(layout_info, NCCL_EP_LAYOUT_INFO);
    }
    ncclEpLayoutInfo_t parsed_layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    if (layout_info != nullptr) {
        parsed_layout_info = epDecodeStruct(layout_info, NCCL_EP_LAYOUT_INFO_INIT);
        layout_info = &parsed_layout_info;
    }
    topk_idx = tensor_required(topk_idx);
    assert(topk_idx->ndim == 2);

    ncclEpGroup_t ep_group = handle->group;
    assert(ep_group != nullptr);

    // LL and HT both accept ncclInt32 or ncclInt64; the cached idx keeps the
    // caller's native width.
    assert(
        (topk_idx->datatype == ncclInt32 || topk_idx->datatype == ncclInt64) &&
        "topk_idx must be ncclInt32 or ncclInt64");

    // Take ownership of the descriptor with a library-owned tensor to
    // ensure no dependency on the caller's descriptor.
    tensor_permanent_copy(&handle->topk_idx, topk_idx);

    handle->num_tokens = static_cast<int>(handle->topk_idx.sizes[0]);
    handle->num_tokens_set = true;

    if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        assert(
            static_cast<int>(topk_idx->sizes[1]) == handle->num_topk &&
            "LL: num_topk mismatch between ncclEpInitHandle and ncclEpUpdateHandle");
        assert(layout_info == nullptr && "LL mode does not accept local tensors in ncclEpUpdateHandle");
        return ncclSuccess;
    }
    if (handle->topk_idx.win_hdl != ncclWindow_t{}) {
        NCCLCHECK(resolveTensorWindowBinding(ep_group, &handle->topk_idx, 0));
    }

    int num_topk = static_cast<int>(topk_idx->sizes[1]);
    if (handle->num_topk > 0)
        assert(
            handle->num_topk == num_topk &&
            "Given topk_idx has unmatched num_topk that ncclEpHandle was created with!");
    else handle->num_topk = num_topk;

    assert(
        handle->num_tokens <= static_cast<int>(ep_group->config.max_dispatch_tokens_per_rank) &&
        "Token count exceeds HT buffer capacity");

    const ncclEpTensor_t* recv_expert_counter = layout_info ? tensor_ptr(layout_info->expert_counters) : nullptr;

    const int num_experts = ep_group->config.num_experts;
    const int max_tokens = ep_group->config.max_dispatch_tokens_per_rank;
    const int n_ranks_per_node = ep_group->lsa_team_size;
    const int nNodes = ep_group->rdma_team_size;
    const int experts_per_rank = ep_group->num_local_experts;
    // Routing map is byte-padded per LSA-team: row stride =
    // ceil(experts_per_lsa_team/8) * num_lsa_teams, each team block byte-aligned.
    const int experts_per_lsa_team = n_ranks_per_node * experts_per_rank;
    const int experts_per_lsa_team_packed = (experts_per_lsa_team + 7) / 8;
    const int routing_row_bytes = experts_per_lsa_team_packed * nNodes;
    (void)num_experts;

    // Count-exchange path selector (stable per group/handle); computed once and reused below.
    const bool count_mode = dispatch_push_count_active(ep_group, handle);

    // Zero the entire preprocessing zero region (routing, r2a, a2r, ler, ntfe) in one call.
    // Buffers are allocated at max_tokens capacity, so this clears beyond the active num_tokens
    // region — safe because allgather/preprocessing will overwrite the relevant portions.
    // Count mode only needs r2a cleared here; ntfe, LERM, and a2r are handled elsewhere
    // (dispatch, the IPC table, and the count block below).
    if (count_mode) {
        const int padded_max_tokens = nccl_ep::align(max_tokens, 16);
        CUDA_CHECK(cudaMemsetAsync(
            handle->ht.rdma_to_attn_map, 0, (size_t)nNodes * padded_max_tokens * sizeof(bool), stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(
            handle->ht.preprocessing_block, 0, handle->ht.preprocessing_zero_region_size, stream));
    }
    // Pull dispatch consumes the uint16 topk map (produced below) instead of the bitmap;
    // skip the bitmap convert + gather entirely when pull is enabled.
    const bool em_pull = em_pull_enabled(ep_group, handle);
    // sparse_to_dense_map (unified S2D) uses 0xFF sentinel (not zero).
    // Pull dispatch never reads the S2D map, so skip its init under pull.
    if (!em_pull && handle->ht.preprocessing_s2d_size > 0) {
        CUDA_CHECK(cudaMemsetAsync(
            handle->ht.sparse_to_dense_map,
            0xFF,
            handle->ht.preprocessing_s2d_size,
            stream));
    }

    const bool use_topk_idx_scan = em_pull && ep_group->ht_buffers.global_topk_idx != nullptr;

    // The routing bitmap is not allocated under pull (the scan reads global_topk_idx),
    // so resolve the send pointer only on the bitmap path.
    uint8_t* global_routing_map = ep_group->ht_buffers.global_routing_map;

    // Count mode skips the bitmap here too, using a count AllGather instead, same as pull
    // dispatch (which scans the order-preserving uint16 topk map).
    // ===== Step 1: Convert sparse topk_idx to bitmap routing map =====
    if (!count_mode && !use_topk_idx_scan) {
        uint8_t* local_routing_send_ptr = global_routing_map + (max_tokens * routing_row_bytes) * ep_group->rank;
        // Pass max_tokens so the kernel zeroes the tail rows in the local send slot;
        // ncclAllGather below ships max_tokens rows and stale tail bits would otherwise
        // be interpreted as live routing by peers.
        // Cache the idx in the caller's native width (int32 or int64).
        if (handle->topk_idx.datatype == ncclInt32) {
            nccl_ep::ht::convert_topk_to_routing_map(
                static_cast<const int32_t*>(handle->topk_idx.data),
                local_routing_send_ptr,
                static_cast<int32_t*>(handle->ht.topk_idx),
                handle->num_tokens,
                max_tokens,
                handle->num_topk,
                experts_per_lsa_team,
                experts_per_lsa_team_packed,
                routing_row_bytes,
                stream);
        } else {
            nccl_ep::ht::convert_topk_to_routing_map(
                static_cast<const int64_t*>(handle->topk_idx.data),
                local_routing_send_ptr,
                static_cast<int64_t*>(handle->ht.topk_idx),
                handle->num_tokens,
                max_tokens,
                handle->num_topk,
                experts_per_lsa_team,
                experts_per_lsa_team_packed,
                routing_row_bytes,
                stream);
        }

        // ===== Step 2: Allgather bitmap routing maps =====
        NCCL_CHECK_RESULT(ncclAllGather(
            local_routing_send_ptr,
            global_routing_map,
            static_cast<size_t>(max_tokens) * routing_row_bytes,
            ncclUint8,
            ep_group->comm,
            stream));
    }
    // Count mode caches topk_idx inside build_count_metadata, so no separate copy here.
    // The cache is always widened to int64 regardless of the caller's native width.
    // TODO: narrow to int16 once the device receiver read is width-parametrized.

    // Pull dispatch: produce + gather the order-preserving uint16 topk map (row stride =
    // num_topk); the pull scan consumes this instead of the bitmap. The pack kernel also
    // writes a handle-local snapshot into handle->ht.topk_idx for push combine (see its
    // field comment).
    if (use_topk_idx_scan) {
        EP_HOST_ASSERT(handle->num_topk <= MAX_NUM_TOPK);
        // uint16 topk map: expert ids are packed as uint16, and kTopkIdxInvalid
        // (0xFFFF) marks empty slots, so the id space must stay below it.
        EP_HOST_ASSERT(ep_group->config.num_experts <= kTopkIdxInvalid);
        const size_t topk_idx_rank_stride = static_cast<size_t>(max_tokens) * handle->num_topk;
        uint16_t* topk_idx_send_ptr =
            ep_group->ht_buffers.global_topk_idx + topk_idx_rank_stride * ep_group->rank;
        assert(handle->ht.topk_idx != nullptr);
        uint16_t* topk_idx_snapshot = static_cast<uint16_t*>(handle->ht.topk_idx);
        if (handle->topk_idx.datatype == ncclInt32) {
            nccl_ep::ht::pack_topk_idx(
                static_cast<const int32_t*>(handle->topk_idx.data), topk_idx_send_ptr, topk_idx_snapshot,
                handle->num_tokens, max_tokens, handle->num_topk, stream);
        } else {
            nccl_ep::ht::pack_topk_idx(
                static_cast<const int64_t*>(handle->topk_idx.data), topk_idx_send_ptr, topk_idx_snapshot,
                handle->num_tokens, max_tokens, handle->num_topk, stream);
        }
        // AllGather moves raw bytes; NCCL has no uint16 type, so ship as uint8.
        NCCL_CHECK_RESULT(ncclAllGather(
            topk_idx_send_ptr, ep_group->ht_buffers.global_topk_idx,
            topk_idx_rank_stride * sizeof(uint16_t), ncclUint8, ep_group->comm, stream));
    }

    // ===== Step 3: Run metadata_preprocessing =====
    const bool expert_major = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

    // layout_info->expert_counters: per-expert counts (HT flat unpadded int32; EM padded int32/64).
    // layout_info->expert_offsets: EM-only per-expert offsets (int32/64).
    const ncclEpTensor_t* recv_expert_offsets_tensor = layout_info ? tensor_ptr(layout_info->expert_offsets) : nullptr;
    auto check_int32_or_int64 = [&](const ncclEpTensor_t* t, const char* name) {
        assert(t->ndim == 1 && "tensor must be 1D");
        assert((t->datatype == ncclInt32 || t->datatype == ncclInt64) && "tensor must be ncclInt32 or ncclInt64");
        assert(
            t->sizes[0] >= static_cast<size_t>(ep_group->num_local_experts) &&
            "tensor size must be >= num_local_experts");
        assert(t->data != nullptr && "tensor data must not be null");
        (void)name;
    };
    // Caller must use the same int dtype across the 3 preprocessing output tensors.
    bool out_is_int64 = true;
    bool out_dtype_set = false;
    auto track_out_dtype = [&](const ncclEpTensor_t* t) {
        const bool is64 = (t->datatype == ncclInt64);
        if (!out_dtype_set) {
            out_is_int64 = is64;
            out_dtype_set = true;
        } else assert(is64 == out_is_int64 && "all preprocessing int output tensors must share dtype");
    };
    void* padded_out_counts = nullptr;
    if (expert_major && recv_expert_counter != nullptr) {
        check_int32_or_int64(recv_expert_counter, "expert_counters");
        padded_out_counts = recv_expert_counter->data;
        track_out_dtype(recv_expert_counter);
    }
    void* out_offsets = nullptr;
    if (expert_major && recv_expert_offsets_tensor != nullptr) {
        check_int32_or_int64(recv_expert_offsets_tensor, "expert_offsets");
        out_offsets = recv_expert_offsets_tensor->data;
        track_out_dtype(recv_expert_offsets_tensor);
    }

    // EM: counts/offsets buffers live in handle_mem (wired by ht_init_handle).
    // FLAT: authoritative counts go to caller's recv_expert_counter when provided.
    int32_t* per_expert_counts_device = nullptr;
    if (expert_major) {
        per_expert_counts_device = handle->ht.per_expert_counts_active;
    } else if (recv_expert_counter != nullptr) {
        assert(recv_expert_counter->ndim == 1 && "recv_expert_counter must be 1D");
        assert(recv_expert_counter->datatype == ncclInt32 && "HT flat: recv_expert_counter must be ncclInt32");
        assert(
            recv_expert_counter->sizes[0] >= static_cast<unsigned int>(ep_group->num_local_experts) &&
            "recv_expert_counter size must be >= num_local_experts");
        ncclEpTensor_t recv_expert_counter_local;
        const ncclEpTensor_t* resolved_recv_expert_counter;
        NCCLCHECK(resolveTensorWindowBinding(
            ep_group,
            recv_expert_counter,
            &recv_expert_counter_local,
            0,
            &resolved_recv_expert_counter));
        assert(resolved_recv_expert_counter->data != nullptr && "recv_expert_counter data must not be null");
        per_expert_counts_device = static_cast<int32_t*>(resolved_recv_expert_counter->data);
    }
    // ht.expert_token_offsets is already the handle_mem slot for EM; FLAT path never reads it.

    // layout_info->recv_total_counter: scalar total recv tokens (size 1, int32/64), written by metadata.
    void* recv_total_counter = nullptr;
    {
        const ncclEpTensor_t* recv_total_counter_tensor =
            layout_info ? tensor_ptr(layout_info->recv_total_counter) : nullptr;
        if (recv_total_counter_tensor != nullptr) {
            assert(recv_total_counter_tensor->ndim == 1);
            assert(recv_total_counter_tensor->sizes[0] >= 1);
            assert(
                recv_total_counter_tensor->datatype == ncclInt32 || recv_total_counter_tensor->datatype == ncclInt64);
            assert(recv_total_counter_tensor->data != nullptr);
            recv_total_counter = recv_total_counter_tensor->data;
            track_out_dtype(recv_total_counter_tensor);
        }
    }

    // Zero the EM-unfused dup-group counter before scan (when allocated).
    if (handle->ht.emuf_group_count != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(handle->ht.emuf_group_count, 0, sizeof(int32_t), stream));
    }

    const bool em_permute_active = em_local_permute_enabled(ep_group, handle);
    // Overflow-drop EM-permute: zero the caller's counter buffer here so the scan's
    // per-block atomicAdd of the post-drop delivered count has a race-free base.
    if (expert_major && em_permute_active && ep_group->config.overflow_policy == NCCL_EP_OVERFLOW_DROP &&
        per_expert_counts_device != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(per_expert_counts_device, 0, experts_per_rank * sizeof(int32_t), stream));
    }
    int max_recv_tpr = static_cast<int>(ep_group->config.max_recv_tokens_per_rank);
    const int alignment = static_cast<int>(handle->ht.dispatch_output_per_expert_alignment);
    if (ep_group->eager_mode && em_permute_active && (alignment > 0)) {
        // Eager local-permute: padded zones live in the caller buffer
        // Extend the zone-overflow budget by the worst-case padding.
        max_recv_tpr += experts_per_rank * (alignment - 1);
    }

    if (count_mode) {
        return ht_update_handle_count_mode(
            ep_group, handle, out_is_int64, padded_out_counts, out_offsets, recv_total_counter,
            per_expert_counts_device, max_recv_tpr, stream);
    }

    NCCLCHECK(
        nccl_ep::ht::call_metadata_preprocessing(
            global_routing_map,
            // Pull dispatch never reads S2D or LERM; skip these scan stores under pull.
            em_pull ? nullptr : handle->ht.sparse_to_dense_map,
            handle->ht.rdma_to_attn_map,
            handle->ht.attn_to_rdma_map,
            handle->ht.token_rank_mask,
            handle->ht.num_tokens_for_experts,
            em_pull ? nullptr : handle->ht.local_expert_routing_map,
            per_expert_counts_device,
            handle->ht.preprocessing_scan_tmp,
            ep_group->rdma_rank,
            ep_group->lsa_rank,
            max_tokens,
            nNodes,
            n_ranks_per_node,
            experts_per_rank,
            expert_major,
            expert_major ? handle->ht.expert_token_offsets : nullptr,
            padded_out_counts,
            out_offsets,
            expert_major ? handle->ht.dispatch_output_per_expert_alignment : size_t(0),
            // Remap kernel writes authoritative per-expert counts (scan overcounts secondary hits).
            expert_major ? per_expert_counts_device : nullptr,
            expert_major ? handle->num_topk : 0,
            recv_total_counter,
            out_is_int64,
            max_recv_tpr,
            handle->ht.emuf_group_buf,
            handle->ht.emuf_group_count,
            handle->ht.emuf_group_stride,
            handle->ht.emuf_max_groups,
            ep_group->preprocess_num_sms,
            // EM cooperative scan scratch carved from ep_workspace.
            expert_major ? ep_group->ep_workspace : nullptr,
            // EM-permute mode: FLAT scan replaces scan_em and writes the unified
            // sparse_to_dense_map (now FLAT-shaped) + FLAT-shape LERM; em_scan_kernel
            // runs only for EM-only offsets / counts / alignment.
            em_permute_active,
            // EM-permute bridge: scan_impl_flat fills token_to_recv_slot with the FLAT
            // recv slot per global token, and em_scan_kernel consumes it to
            // populate the recv-indexed em_slot table.
            em_permute_active ? handle->ht.token_to_recv_slot : nullptr,
            em_permute_active ? handle->ht.flat2em_slot_map : nullptr,
            em_permute_active ? handle->num_topk : 0,
            ep_group->config.overflow_policy == NCCL_EP_OVERFLOW_DROP,
            // Inverse recv-slot map: consumed by pull dispatch and push combine.
            em_pull ? handle->ht.recv_slot_to_src : nullptr,
            // Pull dispatch: scan emits the source top-k position of each hit for weights.
            em_pull ? handle->ht.srcpos_map : nullptr,
            // Pull dispatch: order-preserving uint16 topk map consumed by the scan instead
            // of the bitmap. Both gated on pull (em-permute single-team path).
            ep_group->ht_buffers.global_topk_idx,
            em_pull,
            stream));

    return ncclSuccess;
}

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* out_handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream) {
    topk_idx = tensor_required(topk_idx);
    assert(out_handle != nullptr);
    if (layout_info != nullptr) {
        EP_VALIDATE_STRUCT(layout_info, NCCL_EP_LAYOUT_INFO);
    }
    // Propagate validation errors (e.g. unsupported eager-mode combinations)
    // instead of exiting the process.
    NCCLCHECK(ncclEpInitHandle(
        out_handle,
        ep_group,
        layout,
        config,
        static_cast<int>(topk_idx->sizes[1]),
        /*handle_mem=*/nullptr));
    return ncclEpUpdateHandle(*out_handle, topk_idx, layout_info, stream);
}

ncclResult_t ncclEpHandleDestroy(ncclEpHandle_t handle) {
    if (!handle) return ncclSuccess;

    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        if (handle->ll.owns_handle_mem && handle->ll.handle_mem) {
            handle->group->alloc.free_fn(handle->ll.handle_mem, handle->group->alloc.context);
            handle->ll.handle_mem = nullptr;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        if (handle->ht.owns_handle_mem) {
            if (handle->ht.preprocessing_block) {
                handle->group->alloc.free_fn(handle->ht.preprocessing_block, handle->group->alloc.context);
                handle->ht.preprocessing_block = nullptr;
            }
        }
    }

    delete[] handle->topk_idx.sizes;
    delete handle;
    return ncclSuccess;
}

// EP Operations

// Stream-ordered variant: the readback runs after prior work on `stream`
// (e.g. the ncclEpUpdateHandle scan), so callers on the same stream need no
// separate synchronization.
static ncclResult_t ht_query_num_recv_tokens(ncclEpHandle_t handle, cudaStream_t stream, unsigned int* num_recv_tokens) {
    if (handle->group->config.algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        return ncclInvalidUsage;
    }
    // EM modes: total padded slots is the user-visible recv-token count.
    // em_scan_kernel publishes it as em_internal_offsets[experts_per_rank].
    // FLAT mode: num_tokens_for_experts holds the raw recv count.
    if (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        int64_t em_padded_total;
        CUDA_CHECK(cudaMemcpyAsync(
            &em_padded_total,
            handle->ht.expert_token_offsets + handle->group->num_local_experts,
            sizeof(em_padded_total),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        assert(em_padded_total >= 0);
        *num_recv_tokens = static_cast<unsigned int>(em_padded_total);
        return ncclSuccess;
    }
    int32_t actual_recv_tokens;
    CUDA_CHECK(cudaMemcpyAsync(
        &actual_recv_tokens,
        handle->ht.num_tokens_for_experts,
        sizeof(actual_recv_tokens),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    assert(actual_recv_tokens >= 0);
    *num_recv_tokens = static_cast<unsigned int>(actual_recv_tokens);
    return ncclSuccess;
}

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream) {
    EP_VALIDATE_STRUCT(inputs, NCCL_EP_DISPATCH_INPUTS);
    EP_VALIDATE_STRUCT(outputs, NCCL_EP_DISPATCH_OUTPUTS);
    if (layout_info != nullptr) {
        EP_VALIDATE_STRUCT(layout_info, NCCL_EP_LAYOUT_INFO);
    }
    if (config != nullptr) {
        EP_VALIDATE_STRUCT(config, NCCL_EP_DISPATCH_CONFIG);
    }

    ncclEpDispatchInputs_t parsed_inputs =
        epDecodeStruct(inputs, NCCL_EP_DISPATCH_INPUTS_INIT);
    ncclEpDispatchOutputs_t parsed_outputs =
        epDecodeStruct(outputs, NCCL_EP_DISPATCH_OUTPUTS_INIT);
    inputs = &parsed_inputs;
    outputs = &parsed_outputs;

    ncclEpLayoutInfo_t parsed_layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    if (layout_info != nullptr) {
        parsed_layout_info = epDecodeStruct(layout_info, NCCL_EP_LAYOUT_INFO_INIT);
        layout_info = &parsed_layout_info;
    }

    ncclEpDispatchConfig_t parsed_config = NCCL_EP_DISPATCH_CONFIG_INIT;
    if (config != nullptr) {
        parsed_config = epDecodeStruct(config, NCCL_EP_DISPATCH_CONFIG_INIT);
        config = &parsed_config;
    }

    const unsigned int send_only = config ? config->send_only : 0;
    const ncclEpPassDir_t pass_direction = config ? config->pass_direction : NCCL_EP_FWD_PASS;
    const ncclEpDispQuant_t recipe =
        config ? config->quant_recipe : NCCL_EP_DISP_QUANT_NONE;
    ncclEpGroup_t group = handle->group;
    const ncclEpTensor_t* recipe_tokens = tensor_required(inputs->tokens);
    if (recipe_tokens->ndim != 2) return ncclInvalidArgument;
    const DispatchRecipeLaunchContext recipe_launch{
        .algorithm = group->config.algorithm,
        .layout = handle->layout,
        .hidden = static_cast<int>(recipe_tokens->sizes[1]),
        .num_local_experts = group->num_local_experts,
        .num_ranks = group->nRanks,
        .max_tokens_per_rank = static_cast<int>(group->config.max_dispatch_tokens_per_rank),
        .max_token_bytes = group->config.max_token_bytes,
    };
    NCCLCHECK(validateDispatchRecipe(inputs, outputs, config, recipe_launch));
    if (pass_direction != NCCL_EP_FWD_PASS && handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        fprintf(stderr, "ncclEpDispatch: backward pass (pass_direction=%d) is not supported in LL mode\n",
                (int)pass_direction);
        return ncclInvalidUsage;
    }

    // Lazy num_tokens for callers that skip UpdateHandle (e.g. backward reusing forward's handle_mem).
    // Guard on `num_tokens_set` -- a real zero-token configuration is valid
    // and must not trigger a re-fetch from inputs->tokens (whose data pointer
    // may be NULL on the zero-token rank per the empty-tensor relaxation).
    if (!handle->num_tokens_set) {
        const ncclEpTensor_t* t = tensor_required(inputs->tokens);
        assert(t->ndim > 0);
        handle->num_tokens = static_cast<int>(t->sizes[0]);
        handle->num_tokens_set = true;
    }

    if (group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        assert(x->ndim == 2);
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_dispatch_tokens_per_rank);
        // QUANT_FWD transports the physical byte row. Multi-byte wire
        // dtypes therefore need byte, rather than element-count, alignment.
        if (recipe == NCCL_EP_DISP_QUANT_FWD) {
            assert((x->sizes[1] * ncclTypeSize(x->datatype)) % sizeof(int4) == 0);
        } else {
            assert(x->sizes[1] % sizeof(int4) == 0);
        }
        assert(x->sizes[1] * ncclTypeSize(x->datatype) <= group->config.max_token_bytes);
        const int hidden = static_cast<int>(x->sizes[1]);

        // Find and validate output tensors
        const ncclEpTensor_t* recv_x = tensor_required(outputs->tokens);
        const ncclEpTensor_t* scales = tensor_ptr(outputs->scales);
        // Resolve window-only descriptors before staged fallback.
        ncclEpTensor_t recv_x_local = NCCL_EP_TENSOR_INIT;
        ncclEpTensor_t scales_local = NCCL_EP_TENSOR_INIT;
        NCCLCHECK(resolveTensorWindowBinding(group, recv_x, &recv_x_local, 0, &recv_x));
        if (scales != nullptr) {
            NCCLCHECK(resolveTensorWindowBinding(group, scales, &scales_local, 0, &scales));
        }
        assert(recv_x->ndim > 0);


        // Read rank-major-specific tensors unconditionally so we can assert
        // their presence (rank-major) or absence (expert-major) in the switch below.
        const ncclEpTensor_t* topk_weights_in = tensor_ptr(inputs->topk_weights);
        const ncclEpTensor_t* recv_topk_weights = tensor_ptr(outputs->topk_weights);
        const ncclEpTensor_t* recv_topk_idx = tensor_ptr(outputs->topk_idx);
        const ncclEpTensor_t* src_rank_counter = layout_info ? tensor_ptr(layout_info->src_rank_counters) : nullptr;

        const unsigned num_recv_tokens =
            static_cast<unsigned>(group->nRanks) * group->config.max_dispatch_tokens_per_rank;
        switch (handle->layout) {
        case NCCL_EP_LAYOUT_RANK_MAJOR:
            // Rank-major output is 3D so the per-rank, per-slot structure is
            // explicit in the descriptor — the caller must acknowledge it.
            assert(recv_x->ndim == 3);
            assert(recv_x->sizes[0] == static_cast<unsigned>(group->nRanks));
            assert(recv_x->sizes[1] == group->config.max_dispatch_tokens_per_rank);
            assert(recv_x->sizes[2] == static_cast<unsigned>(hidden));
            assert(topk_weights_in != nullptr);
            assert(recv_topk_weights != nullptr);
            assert(recv_topk_idx != nullptr);
            assert(topk_weights_in->ndim == 2);
            assert(topk_weights_in->datatype == ncclFloat32);
            assert(topk_weights_in->sizes[0] == static_cast<unsigned>(handle->num_tokens));
            assert(topk_weights_in->sizes[1] == static_cast<unsigned>(handle->num_topk));
            assert(recv_topk_weights->ndim == 3);
            assert(recv_topk_weights->datatype == ncclFloat32);
            assert(recv_topk_weights->sizes[0] == static_cast<unsigned>(group->nRanks));
            assert(recv_topk_weights->sizes[1] == group->config.max_dispatch_tokens_per_rank);
            assert(recv_topk_weights->sizes[2] == static_cast<unsigned>(handle->num_topk));
            assert(recv_topk_idx->ndim == 3);
            assert(recv_topk_idx->datatype == ncclInt32);
            assert(recv_topk_idx->sizes[0] == static_cast<unsigned>(group->nRanks));
            assert(recv_topk_idx->sizes[1] == group->config.max_dispatch_tokens_per_rank);
            assert(recv_topk_idx->sizes[2] == static_cast<unsigned>(handle->num_topk));
            if (src_rank_counter != nullptr) {
                assert(src_rank_counter->ndim == 1);
                assert(src_rank_counter->datatype == ncclInt32);
                assert(src_rank_counter->sizes[0] == static_cast<unsigned>(group->nRanks));
            }
            break;
        default:
            assert(recv_x->ndim == 3);
            assert(recv_x->sizes[0] == group->num_local_experts);
            assert(recv_x->sizes[1] == num_recv_tokens);
            assert(recv_x->sizes[2] == static_cast<unsigned>(hidden));
            assert(topk_weights_in == nullptr);
            assert(recv_topk_weights == nullptr);
            assert(recv_topk_idx == nullptr);
            assert(src_rank_counter == nullptr);
            break;
        }

        const ncclEpTensor_t* in_scales_outer = tensor_ptr(inputs->scales);

        // RECV_EXPERT_COUNTER_DEVICE is required for expert-major (per-expert atomic slot allocator)
        // and must be absent for rank-major (outCnt is unused in the rank-major kernel path).
        const ncclEpTensor_t* recv_count = layout_info ? tensor_ptr(layout_info->expert_counters) : nullptr;
        if (handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
            assert(recv_count == nullptr);
        } else {
            assert(recv_count != nullptr);
            assert(recv_count->ndim == 1);
            assert(recv_count->datatype == ncclInt32);
            assert(recv_count->sizes[0] == group->num_local_experts);
        }

        const auto& buffer = handle->ll.layout.buffer;
        const auto next_clean_meta = buffer.clean_meta_offset();

        unsigned signal_base = 0;
        // LL rank-major zero-copy writes each available peer payload window directly.
        const bool nvlink_only = (group->lsa_team_size == group->nRanks);
        const bool recipe_zcopy_ok =
            recipe == NCCL_EP_DISP_QUANT_NONE || recipe == NCCL_EP_DISP_QUANT_FWD;
        const bool zcopy_ok =
            nvlink_only &&
            handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR &&
            recipe_zcopy_ok;
        const bool zcopy_rcv_x = zcopy_ok && recv_x->win_hdl != ncclWindow_t{};
        const bool zcopy_rcv_scales =
            zcopy_ok && recipe == NCCL_EP_DISP_QUANT_FWD &&
            scales != nullptr && scales->win_hdl != ncclWindow_t{};
        const bool zcopy_any = zcopy_rcv_x || zcopy_rcv_scales;
        const char* zcopy_reason =
            !nvlink_only ? "requires nvlink-only topology (lsa_team_size == nRanks)" :
            handle->layout != NCCL_EP_LAYOUT_RANK_MAJOR ? "requires NCCL_EP_LAYOUT_RANK_MAJOR" :
            !recipe_zcopy_ok ? "is unsupported by this LL dispatch recipe" :
            "requires outputs->tokens or outputs->scales to be backed by an NCCL window";
        const char* zcopy_detail = zcopy_reason;
        if (zcopy_rcv_x) {
            zcopy_detail = zcopy_rcv_scales ? "token and scale windows selected" : "token window selected";
        } else if (zcopy_rcv_scales) {
            zcopy_detail = "scale window selected";
        }
        if (nccl_ep_env_flag_on(group->env.debug)) {
            fprintf(
                stderr,
                "[nccl_ep][debug][rank %d] LL dispatch zero-copy %s: %s "
                "(recipe=%d, requested=%d, nvlink_only=%d, rank_major=%d, token_window=%d, scale_window=%d)\n",
                group->rank,
                zcopy_any ? "enabled" : "disabled",
                zcopy_detail,
                static_cast<int>(recipe),
                static_cast<int>(group->config.zero_copy),
                static_cast<int>(nvlink_only),
                static_cast<int>(handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR),
                static_cast<int>(recv_x->win_hdl != ncclWindow_t{}),
                static_cast<int>(zcopy_rcv_scales));
        }
        // Strict zero_copy=ON contract: at least one eligible payload window
        // must be present. AUTO/OFF stay opportunistic
        // (matches pre-enum behavior; missing windows just stage through
        // library buffers).
        if (group->config.zero_copy == NCCL_EP_ZERO_COPY_ON && !zcopy_any) {
            fprintf(stderr, "NCCL EP: zero_copy=ON on LL dispatch %s\n", zcopy_reason);
            return ncclInvalidArgument;
        }
        const ncclWindow_t recv_data_window = zcopy_rcv_x ? recv_x->win_hdl : ncclWindow_t{};
        const size_t recv_data_offset = zcopy_rcv_x ? static_cast<size_t>(recv_x->win_offset) : 0;
        const ncclWindow_t recv_scales_window = zcopy_rcv_scales ? scales->win_hdl : ncclWindow_t{};
        const size_t recv_scales_offset = zcopy_rcv_scales ? static_cast<size_t>(scales->win_offset) : 0;
        const bool round_scale = config ? config->round_scales : false;
        const ncclEpExpertIdKind_t recv_topk_idx_kind =
            resolveRecvTopkIdxKind(
                layout_info != nullptr
                    ? layout_info->recv_topk_idx_kind
                    : NCCL_EP_EXPERT_ID_AUTO);
        auto dispatch_fn = [=](int phases) -> ncclResult_t {
            void* const rdma_buf = group->rdma_buffer;
            // Prepare data pointers
            auto* recv_x_data = recv_x->data;
            auto* scales_data = scales ? scales->data : nullptr;
            auto* expert_recv_source_indices_data = static_cast<int*>(handle->ll.expert_recv_source_indices.data);
            auto* src_rank_counter_data = src_rank_counter ? static_cast<int*>(src_rank_counter->data) : nullptr;
            auto* expert_dispatch_layout_data = static_cast<int64_t*>(handle->ll.expert_dispatch_layout.data);
            auto* recv_count_data = recv_count ? static_cast<int*>(recv_count->data) : nullptr;
            auto* x_data = x->data;
            auto* topk_weights_in_data = topk_weights_in ? static_cast<const float*>(topk_weights_in->data) : nullptr;
            auto* recv_topk_weights_data = recv_topk_weights ? static_cast<float*>(recv_topk_weights->data) : nullptr;
            auto* recv_topk_idx_data = recv_topk_idx ? static_cast<int32_t*>(recv_topk_idx->data) : nullptr;

            // LL accepts int32 or int64 topk_idx; the cached dtype picks the JIT
            // kernel specialization (TopkIdxT). The lambda packs the shared
            // DispatchParams and threads recipe-specific state through.
            auto launch_dispatch = [&](auto* topk_idx_data, bool topk_is_int64) -> ncclResult_t {
                auto* in_scales_data =
                    (recipe == NCCL_EP_DISP_QUANT_FWD)
                    ? static_cast<const uint8_t*>(in_scales_outer->data) : nullptr;
                nccl_ep::ll::DispatchParams params{};
                params.inData = x_data;
                params.inScalesBuf = in_scales_data;
                params.inTopkIdx = topk_idx_data;
                params.topkIdxIsInt64 = topk_is_int64;
                // This is a launch stride derived from the validated 2D input
                // tensor, never an independent user recipe parameter.
                params.scalesPerToken =
                    recipe == NCCL_EP_DISP_QUANT_FWD
                    ? static_cast<int>(in_scales_outer->sizes[1])
                    : 0;
                params.scaleDtype = (recipe == NCCL_EP_DISP_QUANT_FWD)
                    ? in_scales_outer->datatype : ncclUint8;
                params.inTopkWeights = topk_weights_in_data;
                params.outDataBuf = recv_x_data;
                params.outScalesBuf = scales_data;
                params.outSrcInfo = expert_recv_source_indices_data;
                params.outRecvRankCounter = src_rank_counter_data;
                params.outLayout = expert_dispatch_layout_data;
                params.outCnt = recv_count_data;
                params.outRecvTopkWeights = recv_topk_weights_data;
                params.outRecvTopkIdx = recv_topk_idx_data;
                params.rdmaBuf = rdma_buf;
                params.sendOff = buffer.dispatch_rdma_send_buffer_offset;
                params.recvOff = buffer.dispatch_rdma_recv_data_buffer_offset;
                params.recvCntOff = buffer.dispatch_rdma_recv_count_buffer_offset;
                params.nextRecvCntBufSize = next_clean_meta.second;
                params.recvStats = nullptr;
                params.waitStats = nullptr;
                params.epochState = group->ll_epoch_state;
                params.payloadSlotStride = handle->ll.layout.payload_slot_stride;
                params.signalSlotStride = handle->ll.layout.signal_slot_stride;
                params.numTokens = handle->num_tokens;
                params.hidden = hidden;
                params.maxTokensPerRank = group->config.max_dispatch_tokens_per_rank;
                params.numTopk = handle->num_topk;
                params.numExperts = group->config.num_experts;
                params.currRank = group->rank;
                params.numRanks = group->nRanks;
                params.layout = handle->layout;
                params.numComms = group->num_nccl_comms;
                params.devComms = group->nccl_dev_comms;
                params.windows = group->nccl_wins;
                params.signalsBase = signal_base;
                params.workspace = group->ep_workspace;
                params.numDeviceSms = group->dispatch_num_sms;
                params.rankMask = group->mask_buffer;
                params.asyncErrorFlag = group->async_error_flag;
                params.timeoutCycles = group->timeout_cycles;
                params.roundScale = round_scale;
                params.nvlinkOnly = nvlink_only;
                params.recvDataWindow = recv_data_window;
                params.recvDataOffset = recv_data_offset;
                params.rcvScalesWin = recv_scales_window;
                params.rcvScalesOffs = recv_scales_offset;
                params.recvTopkIdxKind = recv_topk_idx_kind;
                params.phases = phases;
                params.tokenDtype = x->datatype;
                // Keep recipe dispatch explicit at the kernel launch boundary. The
                // selected branch becomes the JIT kernel's quantization variant.
                switch (recipe) {
                    case NCCL_EP_DISP_QUANT_NONE:
                        return nccl_ep::ll::call_dispatch(
                            params, NCCL_EP_DISP_QUANT_NONE, stream);
                    case NCCL_EP_DISP_QUANT_FWD:
                        return nccl_ep::ll::call_dispatch(
                            params, NCCL_EP_DISP_QUANT_FWD, stream);
                    case NCCL_EP_DISP_QUANT_DS_FP8E3M4:
                        return nccl_ep::ll::call_dispatch(
                            params, NCCL_EP_DISP_QUANT_DS_FP8E3M4, stream);
                    default:
                        std::fprintf(stderr,
                                     "NCCL EP warning: unsupported LL dispatch recipe %d\n",
                                     static_cast<int>(recipe));
                        return ncclInvalidArgument;
                }
            };
            switch (handle->topk_idx.datatype) {
            case ncclInt32:
                    return launch_dispatch(
                        static_cast<const int32_t*>(handle->topk_idx.data), /*topk_is_int64=*/false);
            case ncclInt64:
                    return launch_dispatch(
                        static_cast<const int64_t*>(handle->topk_idx.data), /*topk_is_int64=*/true);
            default:
                    std::fprintf(stderr, "NCCL EP warning: LL topk_idx has unsupported dtype %d\n",
                                 static_cast<int>(handle->topk_idx.datatype));
                    return ncclInvalidArgument;
            }
        };

        // Execute dispatch with appropriate phase flags
        const int dispatch_phases =
            send_only ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        NCCLCHECK(dispatch_fn(dispatch_phases));

        if (send_only) {
            handle->ll.continue_fn = dispatch_fn;
        }
    } else { // HT

        bool is_lsa_only = !is_internode_available(group);

        const bool expert_major = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        const ncclEpTensor_t* topk_weights = tensor_ptr(inputs->topk_weights);
        const ncclEpTensor_t* scales = tensor_ptr(inputs->scales);
        // Local copies for tensors that need window resolution; only populated when used.
        ncclEpTensor_t x_local, scales_local, topk_weights_local;
        ncclEpTensor_t recv_x_local, recv_scales_local, recv_topk_weights_local, recv_topk_idx_local;

        if (x->ndim == 0) {
            return ncclInvalidArgument;
        }
        assert(x->ndim == 2);
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_dispatch_tokens_per_rank);
        assert(
            x->sizes[1] * ncclTypeSize(x->datatype) <= group->config.max_token_bytes &&
            "HT dispatch token bytes must not exceed group's max_token_bytes");
        const int hidden = static_cast<int>(x->sizes[1]);
        const size_t token_bytes = static_cast<size_t>(hidden) * ncclTypeSize(x->datatype);
        const bool scales_forward = recipe == NCCL_EP_DISP_QUANT_FWD;
        const size_t token_staging_region_bytes =
            static_cast<size_t>(group->config.max_dispatch_tokens_per_rank) * token_bytes;
        NCCLCHECK(resolveTensorWindowBinding(
            group,
            x,
            &x_local,
            static_cast<uint64_t>(group->gin_config.token_staging_offset),
            &x));

        // For multi-LSA-team: copy user buffers to pre-registered staging buffers
        // The staging buffers were allocated and GIN-registered during Group Create
        // This avoids ~60ms GIN registration overhead on the dispatch hot path
        void* token_ptr = x->data; // Default: use user buffer directly
        const bool token_uses_external_window = tensorUsesExternalWindow(group, x);
        const bool staging_available =
            !is_lsa_only && handle->ht.token_staging_buffer != nullptr;
        const bool stage_regular_tokens =
            staging_available && !scales_forward && !token_uses_external_window;
        if (stage_regular_tokens) {
            // Copy user tokens to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t token_size = x->sizes[0] * x->sizes[1] * ncclTypeSize(x->datatype);
            CUDA_CHECK(cudaMemcpyAsync(
                handle->ht.token_staging_buffer,
                x->data,
                token_size,
                cudaMemcpyDeviceToDevice,
                stream));
            token_ptr = handle->ht.token_staging_buffer;
        }
        int scale_elem_bytes = 0;
        bool scales_use_external_window = false;
        if (scales_forward) {
            scale_elem_bytes = static_cast<int>(ncclTypeSize(scales->datatype));
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                scales,
                &scales_local,
                group->gin_config.token_staging_offset + token_staging_region_bytes,
                &scales));
            scales_use_external_window = tensorUsesExternalWindow(group, scales);
        }

        void* scales_ptr = nullptr;
        if (scales_forward) {
            scales_ptr = scales->data;
        }
        // HT dispatch kernel uses TMA for token/prob/scaling-factor payloads.
        // Keep these constraints at API-entry to fail fast on unsupported shapes.
        const int experts_per_lsa_team = group->num_local_experts * group->lsa_team_size;
        assert(
            (experts_per_lsa_team * static_cast<int>(sizeof(float))) % 16 == 0 &&
            "HT dispatch requires experts_per_lsa_team to be multiple of 4 (16B prob TMA alignment)");

        assert(
            (token_bytes % 16) == 0 &&
            "HT dispatch requires token bytes per token to be 16B aligned for TMA");

        int num_scales_per_token = 0;
        size_t forwarded_scale_bytes_per_token = 0;
        if (scales_forward) {
            num_scales_per_token = static_cast<int>(scales->sizes[1]);
            forwarded_scale_bytes_per_token =
                static_cast<size_t>(num_scales_per_token) * scale_elem_bytes;
        }

        const bool stage_forward_tokens =
            staging_available && scales_forward && !token_uses_external_window;
        const bool stage_forward_scales =
            staging_available && scales_forward && !scales_use_external_window;
        if (stage_forward_tokens || stage_forward_scales) {
            uint8_t* staging = static_cast<uint8_t*>(handle->ht.token_staging_buffer);
            uint8_t* scale_staging = staging + token_staging_region_bytes;

            if (stage_forward_tokens) {
                CUDA_CHECK(cudaMemcpyAsync(
                    staging, x->data, x->sizes[0] * token_bytes, cudaMemcpyDeviceToDevice, stream));
                token_ptr = staging;
            }
            if (stage_forward_scales) {
                CUDA_CHECK(cudaMemcpyAsync(
                    scale_staging,
                    scales->data,
                    x->sizes[0] * forwarded_scale_bytes_per_token,
                    cudaMemcpyDeviceToDevice,
                    stream));
                scales_ptr = scale_staging;
            }
        }

        // Output tensors
        const ncclEpTensor_t* recv_x = tensor_required(outputs->tokens);
        const ncclEpTensor_t* recv_topk_weights = tensor_ptr(outputs->topk_weights);
        const ncclEpTensor_t* recv_topk_idx = tensor_ptr(outputs->topk_idx);
        const ncclEpTensor_t* recv_scales = nullptr;
        if (recv_x->ndim == 0) {
            return ncclInvalidArgument;
        }
        const bool recv_x_external_window = tensorUsesExternalWindow(group, recv_x);
        NCCLCHECK(resolveTensorWindowBinding(group, recv_x, &recv_x_local, 0, &recv_x));
        if (recv_x->ndim < 2) {
            return ncclInvalidArgument;
        }
        // Eager mode sizes recv buffers per routing, which requires the actual
        // recv count on host and is unavailable during CUDA Graph capture.
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        const bool is_capturing = (capture_status == cudaStreamCaptureStatusActive);
        if (group->eager_mode && is_capturing) {
            fprintf(
                stderr,
                "NCCL EP: eager mode (max_recv_tokens_per_rank = NCCL_EP_AUTO) does not support "
                "CUDA Graph capture of ncclEpDispatch; use a fixed max_recv_tokens_per_rank\n");
            return ncclInvalidUsage;
        }
        // Fixed budget: dispatch indexes recv slots up to max_recv_tokens; recv_x
        // must cover it in full. Eager mode is checked against the actual recv
        // count below, before any kernel writes caller memory.
        if (!group->eager_mode && recv_x->sizes[0] < static_cast<unsigned>(group->max_recv_tokens)) {
            return ncclInvalidArgument;
        }
        if (recipe == NCCL_EP_DISP_QUANT_FWD) {
            recv_scales = tensor_ptr(outputs->scales);
            if (recv_scales == nullptr) {
                return ncclInvalidArgument;
            }
            const bool recv_scales_external_window = tensorUsesExternalWindow(group, recv_scales);
            if (recv_x_external_window != recv_scales_external_window) {
                fprintf(stderr,
                        "NCCL EP: HT scales-forward requires outputs->tokens and outputs->scales "
                        "to be either both window-backed or both ordinary tensors\n");
                return ncclInvalidArgument;
            }
            NCCLCHECK(resolveTensorWindowBinding(group, recv_scales, &recv_scales_local, 0, &recv_scales));
        }

        // Pass direction is the source of truth (default FWD via zero-init).
        // FWD: topk_weights required (routing live). BWD: topk_weights forbidden.
        const bool forward_dispatch = (pass_direction == NCCL_EP_FWD_PASS);

        if (forward_dispatch) {
            if (topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            assert(topk_weights->ndim == 2 && topk_weights->datatype == ncclFloat32);
            assert(topk_weights->sizes[0] == handle->num_tokens);
            assert(topk_weights->sizes[1] == handle->num_topk);
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                topk_weights,
                &topk_weights_local,
                static_cast<uint64_t>(group->gin_config.dense_prob_offset),
                &topk_weights));
        } else {
            if (topk_weights != nullptr || recv_topk_weights != nullptr || recv_topk_idx != nullptr) {
                return ncclInvalidArgument;
            }
        }

        // Pull EM dispatch reads source tokens + weights over NVLink and writes EM
        // directly, so it skips the push kernel, the FLAT staging, and the FLAT
        // dense-prob / weight compute below (weights are pulled per source token).
        const bool em_pull_active = em_pull_enabled(group, handle);

        // Pull dispatch forwards per-token scales (NONE and FWD recipes only).
        if (em_pull_active && recipe != NCCL_EP_DISP_QUANT_NONE &&
            recipe != NCCL_EP_DISP_QUANT_FWD) {
            fprintf(stderr,
                    "ncclEpDispatch: NCCL_EP_HT_EM_PULL_PUSH pull dispatch "
                    "supports only NCCL_EP_DISP_QUANT_NONE and NCCL_EP_DISP_QUANT_FWD (recipe=%d unsupported).\n",
                    static_cast<int>(recipe));
            return ncclInvalidArgument;
        }

        // FWD only: rebuild dense prob from cached topk_idx + caller topk_weights.
        float* dense_prob = handle->ht.dense_prob_buffer;
        if (forward_dispatch && !em_pull_active) {
            assert(
                handle->ht.topk_idx != nullptr &&
                "HT FWD dispatch: ht.topk_idx missing (ncclEpUpdateHandle not called?)");
            size_t dense_prob_size =
                static_cast<size_t>(handle->num_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_prob, 0, dense_prob_size, stream));

            if (topk_idx_cache_is_int32(group, handle)) {
                nccl_ep::ht::sparse_to_dense_prob(
                    static_cast<const int32_t*>(handle->ht.topk_idx),
                    static_cast<const float*>(topk_weights->data),
                    dense_prob,
                    handle->num_tokens,
                    handle->num_topk,
                    group->config.num_experts,
                    stream);
            } else {
                nccl_ep::ht::sparse_to_dense_prob(
                    static_cast<const int64_t*>(handle->ht.topk_idx),
                    static_cast<const float*>(topk_weights->data),
                    dense_prob,
                    handle->num_tokens,
                    handle->num_topk,
                    group->config.num_experts,
                    stream);
            }
        }

        /* ===== Build DispatchParams ===== */
        // DispatchParams encapsulates all buffers and metadata needed by HT dispatch kernel:
        //   - Input buffers: attn_input_token, attn_input_prob, attn_input_scaling_factor
        //   - Intra-LSA output buffers: expert_output_token_ptrs, expert_output_prob_ptrs (per-rank pointers)
        //   - RDMA staging buffers: dispatch_gin_G2S_flags (for multi-LSA-team only)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map
        //   - Sync flags: expected_*_flag_val, lsa_S2G_flags
        nccl_ep::ht::DispatchParams params{};
        params.hidden_dim = hidden;
        params.experts_per_rank = group->num_local_experts;
        params.lsa_team_size = group->lsa_team_size;
        params.attn_input_token = token_ptr;
        params.attn_input_prob = forward_dispatch ? dense_prob : nullptr;
        params.attn_input_scaling_factor =
            recipe == NCCL_EP_DISP_QUANT_FWD ? static_cast<const uint8_t*>(scales_ptr) : nullptr;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access.
        // For external output tensors with windows, resolve full per-rank output pointers
        // (local + same-node peers) so all writers target user buffers directly.
        std::vector<void*> dispatch_output_token_ptrs;
        const bool rcv_x_zcopy = recv_x_external_window;
        if (group->config.zero_copy == NCCL_EP_ZERO_COPY_ON && !rcv_x_zcopy) {
            fprintf(
                stderr,
                "NCCL EP: zero_copy requires ncclEpDispatch outputs->tokens to be backed by a "
                "user-registered NCCL window (ncclCommWindowRegister)\n");
            return ncclInvalidArgument;
        }
        // em_permute always writes to staging (the permute kernel below copies
        // staging -> recv_x->data); external-window vs. plain recv_x is irrelevant
        // since the permute kernel just dereferences recv_x->data.
        const bool em_permute_active = em_local_permute_enabled(group, handle);
        // Count-exchange path selector (stable per group/handle); computed once and reused below.
        const bool count_mode = dispatch_push_count_active(group, handle);

        if (rcv_x_zcopy && !em_permute_active) {
            NCCLCHECK(buildIntranodePtrArray<void>(group, recv_x, dispatch_output_token_ptrs));
            params.expert_output_token_ptrs = dispatch_output_token_ptrs.data();
        } else {
            params.expert_output_token_ptrs = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs;
        }
        params.expert_output_prob_ptrs = group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs;
        std::vector<void*> dispatch_output_sf_ptrs;
        const bool rcv_scales_zcopy =
            recipe == NCCL_EP_DISP_QUANT_FWD &&
            tensorUsesExternalWindow(group, recv_scales) && !em_permute_active;
        if (rcv_scales_zcopy) {
            NCCLCHECK(buildIntranodePtrArray<void>(group, recv_scales, dispatch_output_sf_ptrs));
            params.expert_output_scaling_factor_ptrs = dispatch_output_sf_ptrs.data();
        } else {
            params.expert_output_scaling_factor_ptrs = recipe == NCCL_EP_DISP_QUANT_FWD
                ? group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs : nullptr;
        }
        if (nccl_ep_env_flag_on(group->env.debug)) {
            const bool token_direct = rcv_x_zcopy && !em_permute_active;
            const bool dispatch_direct = token_direct &&
                (recipe != NCCL_EP_DISP_QUANT_FWD || rcv_scales_zcopy);
            const char* reason = dispatch_direct
                ? (recipe == NCCL_EP_DISP_QUANT_FWD
                       ? "token and scale windows selected"
                       : "token window selected")
                : em_permute_active && rcv_x_zcopy
                    ? "expert-major local_permute stages before writing caller windows"
                    : !rcv_x_zcopy
                        ? "outputs->tokens is not backed by a compatible external window"
                        : "outputs->scales is not backed by a compatible external window";
            fprintf(
                stderr,
                "[nccl_ep][debug][rank %d] HT dispatch zero-copy %s: %s "
                "(recipe=%d, requested=%d, token_window=%d, scale_window=%d)\n",
                group->rank,
                dispatch_direct ? "enabled" : "disabled",
                reason,
                static_cast<int>(recipe),
                static_cast<int>(group->config.zero_copy),
                static_cast<int>(rcv_x_zcopy),
                static_cast<int>(rcv_scales_zcopy));
        }

        bool zcopy_only = rcv_x_zcopy;
        if (recipe != NCCL_EP_DISP_QUANT_NONE) {
            zcopy_only = zcopy_only && rcv_scales_zcopy;
        }
        const bool need_recv_counts = !is_capturing && !em_permute_active && !zcopy_only;

        // Required caller recv capacity (full budget in fixed mode, routed count in eager).
        unsigned int recv_copy_rows = static_cast<unsigned int>(group->max_recv_tokens);
        if (em_permute_active) {
            // EM: kernels read the routed count from device, so recv_x's own row count is the
            // caller capacity here and no device->host sync is needed.
            recv_copy_rows = static_cast<unsigned int>(recv_x->sizes[0]);
        } else if (group->eager_mode || need_recv_counts) {
            // FLAT: the routed count drives the dense->sparse copy below, so query it.
            NCCLCHECK(ht_query_num_recv_tokens(handle, stream, &recv_copy_rows));
            if (recv_x->sizes[0] < recv_copy_rows) {
                fprintf(
                    stderr,
                    "NCCL EP: eager dispatch recv buffer too small: %u tokens < %u required "
                    "(size from ncclEpLayoutInfo_t::recv_total_counter)\n",
                    static_cast<unsigned>(recv_x->sizes[0]),
                    recv_copy_rows);
                return ncclInvalidArgument;
            }
        }
        if (recipe == NCCL_EP_DISP_QUANT_FWD &&
            recv_scales->sizes[0] < recv_copy_rows) {
            fprintf(
                stderr,
                "NCCL EP: dispatch recv scale buffer too small: %u rows < %u required\n",
                static_cast<unsigned>(recv_scales->sizes[0]),
                recv_copy_rows);
            return ncclInvalidArgument;
        }
        // Zero-recv FWD (eager): recv outputs may be legally empty (data == nullptr),
        // but token and weight outputs must agree on emptiness — they are rows of the
        // same recv set.
        if (forward_dispatch && recv_topk_weights != nullptr &&
            tensorIsEmpty(recv_topk_weights) != tensorIsEmpty(recv_x)) {
            return ncclInvalidArgument;
        }

        // EM-permute path runs the dispatch kernel in FLAT layout and reshuffles
        // FLAT staging into EM zones via the local permute kernel below. BWD
        // reuses the FWD-populated handle maps; only the FWD-only weight scatter
        // is skipped (recv_topk_weights is null for BWD).
        params.rdma_to_attn_map = handle->ht.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->ht.attn_to_rdma_map;
        params.sparse_to_dense_map = handle->ht.sparse_to_dense_map;
        params.s2d_inner_dim = (expert_major && !em_permute_active) ? handle->num_topk : group->lsa_team_size;
        params.layout = em_permute_active ? NCCL_EP_LAYOUT_RANK_MAJOR : handle->layout;
        // s2d_inner_dim must pair with layout (mismatch → OOB in combine reduction).
        assert(
            (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? (params.s2d_inner_dim == handle->num_topk) :
                                                             (params.s2d_inner_dim == group->lsa_team_size));
        // Expert-major zero-padding inputs for the in-kernel PAD warp; suppressed
        // in EM-permute mode (permute kernel writes pads after dispatch).
        const bool fused_em_pad = expert_major && !em_permute_active;
        params.pad_actual_counts = fused_em_pad ? handle->ht.per_expert_counts_active : nullptr;
        params.pad_expert_token_offsets = fused_em_pad ? handle->ht.expert_token_offsets : nullptr;
        params.pad_alignment =
            fused_em_pad ? static_cast<int>(handle->ht.dispatch_output_per_expert_alignment) : 0;
        // Padded eager EM recv budget, matching ncclEpUpdateHandle's compute_layout_info
        // (max_recv_tpr). Used below both as count mode's recv_max_slots and as the S2G
        // backstop bound (params.max_recv_tokens_per_rank), so legitimately-padded slots
        // don't trip S2G's assert.
        int max_recv_tokens_per_rank = group->max_recv_tokens;
        if (group->eager_mode && em_permute_active && handle->ht.dispatch_output_per_expert_alignment > 0) {
            max_recv_tokens_per_rank += group->num_local_experts *
                (static_cast<int>(handle->ht.dispatch_output_per_expert_alignment) - 1);
        }
        // Fused-meta-dispatch path: build the rank-major s2d inside the dispatch kernel from the
        // histogram-published per-rank count rows (FWD only; BWD reuses the FWD-populated s2d).
        // No host collective or kernel runs between the histogram and dispatch.
        const bool count_mode_dispatch = count_mode && forward_dispatch;
        params.dispatch_push_count.active = count_mode_dispatch;
        if (count_mode_dispatch) {
            params.dispatch_push_count.own_row = handle->ht.dispatch_push_count.own_row;
            // Unfused path: MAP warps read the per-handle AllGather buffer instead of the
            // fanned-out landing (set in UpdateHandle when the caller supplies layout_info).
            params.dispatch_push_count.fused_meta_dispatch = handle->ht.dispatch_push_count.fused_meta_dispatch;
            params.dispatch_push_count.cached_cnt_rows = handle->ht.dispatch_push_count.cached_cnt_rows;
            params.dispatch_push_count.s2d_out = handle->ht.sparse_to_dense_map;
            params.dispatch_push_count.published_offset = group->ht_buffers.published_offset;
            params.dispatch_push_count.per_src_lteam_chunk_rank = handle->ht.dispatch_push_count.own_chunk_rank;
            params.dispatch_push_count.cached_topk_idx = static_cast<const int64_t*>(handle->ht.topk_idx);
            params.dispatch_push_count.num_recv_out = handle->ht.num_tokens_for_experts;
            params.dispatch_push_count.per_src_lteam_num_tokens = handle->ht.dispatch_push_count.per_src_lteam_num_tokens;
            params.dispatch_push_count.num_topk = handle->num_topk;
            params.dispatch_push_count.num_src_ranks = group->nRanks;
            params.dispatch_push_count.lsa_team_size = group->lsa_team_size;
            params.dispatch_push_count.token_dst_rank_bitmap = handle->ht.dispatch_push_count.token_dst_rank_bitmap;
            params.dispatch_push_count.recv_tables_ptrs =
                reinterpret_cast<uint8_t* const*>(group->ht_buffers.dispatch_push_count_meta_table_ptrs);
            params.dispatch_push_count.recv_max_slots = max_recv_tokens_per_rank;
            // Only NCCL_EP_OVERFLOW_DROP sets this; TRAP still traps on overflow (see the
            // DROP-contract doc). When set, the fused meta build clamps published per-expert
            // offsets/counts to the recv budget so over-budget tokens drop instead of trapping.
            params.dispatch_push_count.allow_overflow_drop = group->config.overflow_policy == NCCL_EP_OVERFLOW_DROP;
            params.dispatch_push_count.expert_token_offsets_out = handle->ht.expert_token_offsets;
            params.dispatch_push_count.per_expert_counts_out = handle->ht.per_expert_counts_active;
            // Info-layout at dispatch time is only wired for HT EM local-permute; other HT paths
            // still write it in UpdateHandle. Remove once every HT mode supports it here.
            EP_HOST_ASSERT(em_permute_active && "HT dispatch info-layout wiring is only supported for EM local permute");
            // Caller's padded per-expert offsets/counts, wired from the dispatch-time
            // layout_info when provided; the permute writes them post-build (receiver-local).
            const ncclEpTensor_t* caller_offsets_tensor =
                layout_info ? tensor_ptr(layout_info->expert_offsets) : nullptr;
            const ncclEpTensor_t* count_caller_counts_tensor =
                layout_info ? tensor_ptr(layout_info->expert_counters) : nullptr;
            auto check_count_caller_tensor = [&](const ncclEpTensor_t* t) {
                EP_HOST_ASSERT(t->ndim == 1 && "tensor must be 1D");
                EP_HOST_ASSERT((t->datatype == ncclInt32 || t->datatype == ncclInt64) && "tensor must be ncclInt32 or ncclInt64");
                EP_HOST_ASSERT(
                    t->sizes[0] >= static_cast<size_t>(group->num_local_experts) &&
                    "tensor size must be >= num_local_experts");
                EP_HOST_ASSERT(t->data != nullptr && "tensor data must not be null");
            };
            if (caller_offsets_tensor) check_count_caller_tensor(caller_offsets_tensor);
            if (count_caller_counts_tensor) check_count_caller_tensor(count_caller_counts_tensor);
            // Offsets and counts must share dtype; default int64 when neither tensor is given.
            if (caller_offsets_tensor && count_caller_counts_tensor)
                EP_HOST_ASSERT(
                    caller_offsets_tensor->datatype == count_caller_counts_tensor->datatype &&
                    "expert_offsets and expert_counters must share dtype");
            params.dispatch_push_count.caller_offsets = caller_offsets_tensor ? caller_offsets_tensor->data : nullptr;
            params.dispatch_push_count.caller_counts = count_caller_counts_tensor ? count_caller_counts_tensor->data : nullptr;
            // Caller's scalar recv_total_counter (pre-drop padded total), written by the fused
            // permute alongside the per-expert counts. Its int width must match counts/offsets.
            const ncclEpTensor_t* recv_total_tensor =
                layout_info ? tensor_ptr(layout_info->recv_total_counter) : nullptr;
            // Infer the shared dtype from whichever optional output the caller actually gave;
            // default int64 only when none of the three were requested.
            params.dispatch_push_count.caller_out_is_int64 =
                count_caller_counts_tensor    ? (count_caller_counts_tensor->datatype == ncclInt64)
                : caller_offsets_tensor ? (caller_offsets_tensor->datatype == ncclInt64)
                : recv_total_tensor     ? (recv_total_tensor->datatype == ncclInt64)
                                              : true;
            params.dispatch_push_count.em_alignment =
                static_cast<int>(handle->ht.dispatch_output_per_expert_alignment);
            if (recv_total_tensor) {
                EP_HOST_ASSERT(recv_total_tensor->ndim == 1 && "recv_total_counter must be 1D");
                EP_HOST_ASSERT(recv_total_tensor->sizes[0] >= 1 && "recv_total_counter size must be >= 1");
                EP_HOST_ASSERT((recv_total_tensor->datatype == ncclInt32 || recv_total_tensor->datatype == ncclInt64) &&
                       "recv_total_counter must be ncclInt32 or ncclInt64");
                EP_HOST_ASSERT(recv_total_tensor->data != nullptr && "recv_total_counter data must not be null");
                EP_HOST_ASSERT((recv_total_tensor->datatype == ncclInt64) == params.dispatch_push_count.caller_out_is_int64 &&
                       "recv_total_counter dtype must match expert_counters/offsets");
            }
            params.dispatch_push_count.caller_recv_total = recv_total_tensor ? recv_total_tensor->data : nullptr;
            // Always publish whatever the caller requested via this call's layout_info: it may
            // name different tensors than ncclEpUpdateHandle's layout_info (or none at all), so
            // UpdateHandle having already published its own tensors doesn't cover this call's.
            // The recompute is deterministic from the same count rows, so re-publishing into a
            // tensor UpdateHandle already wrote is redundant but not wrong.
        }
        // Always pass a valid device pointer — the kernel unconditionally
        // dereferences this even in single-LSA-team mode (the value is just unused).
        params.expected_gin_flag_val = group->ht_buffers.dispatch_expected_gin_flag_val;
        params.gin_G2S_flags = is_lsa_only ? nullptr : group->ht_buffers.dispatch_gin_G2S_flags;
        params.expected_lsa_flag_val = group->ht_buffers.dispatch_expected_lsa_flag_val;
        params.lsa_S2G_flags = group->ht_buffers.dispatch_lsa_S2G_flags;
        params.dispatch_grid_barrier_counter = group->ht_buffers.dispatch_grid_barrier_counter;
        params.guard_enabled = !nccl_ep_env_flag_on(group->env.disable_guard);
        // Pass device communicators and windows
        // Always pass a valid devComm (single-LSA-team too): the HT LSA sync-guard uses the NCCL LSA
        // barrier (needs comm.lsaBarrier). GIN/RDMA paths stay if-constexpr-gated (out single-LSA-team).
        // TODO: remove multiple gin comm notion from group
        params.dcomm = group->gin_config.dcomms[0];
        params.nccl_token_window = x->win_hdl;
        params.nccl_prob_window = forward_dispatch ? group->gin_config.nccl_window : ncclWindow_t{};
        params.nccl_sf_window = ncclWindow_t{};
        if (scales_forward) {
            params.nccl_sf_window = scales->win_hdl;
        }
        params.nccl_internal_window = group->gin_config.nccl_window;
        params.num_ctx_per_comm = is_lsa_only ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_lsa_only ? nullptr : group->gin_config.gin_base_ptr;
        // Use offsets relative to gin_base_ptr
        // All buffers are part of one large registered window
        // Calculate bytes_per_entry for batched staging
        size_t bytes_per_token_entry = group->config.max_token_bytes; // token data
        size_t bytes_per_prob_entry = (group->num_local_experts * group->lsa_team_size) * sizeof(float); // prob data
        size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry;

        params.mr_info.attn_input_token_offset = 0;
        if (!is_lsa_only) {
            params.mr_info.attn_input_token_offset = x->win_offset;
        }
        params.mr_info.attn_input_prob_offset =
            (is_lsa_only || !forward_dispatch) ? 0 : group->gin_config.dense_prob_offset;
        params.mr_info.attn_input_scaling_factor_offset = 0;
        if (!is_lsa_only && scales_forward) {
            params.mr_info.attn_input_scaling_factor_offset = scales->win_offset;
        }
        params.mr_info.gin_send_staging_offset =
            is_lsa_only ? 0 : group->gin_config.gin_send_staging_offset;
        params.mr_info.gin_recv_staging_offset =
            is_lsa_only ? 0 : group->gin_config.gin_recv_staging_offset;
        params.mr_info.guard_offset = is_lsa_only ? 0 : group->gin_config.dispatch_guard_offset;
        params.mr_info.bytes_per_entry = bytes_per_entry;
        params.mr_info.max_tokens_per_dest = static_cast<size_t>(group->config.max_dispatch_tokens_per_rank);
        params.mr_info.signals_tail_base =
            is_lsa_only ? 0 : static_cast<unsigned>(group->gin_config.signals_tail_base);
        params.mr_info.num_max_rdma_chunked_send_tokens =
            is_lsa_only ? 0 : group->gin_config.num_max_rdma_chunked_send_tokens;
        params.local_rank = group->lsa_rank;
        params.lsa_team = group->rdma_rank;
        params.tokens_per_lsa = group->config.max_dispatch_tokens_per_rank;
        // EM local-fanout: dispatch dedups S2G; receiver local_dup fills secondaries.
        const bool em_unfused_active = em_local_dup_active(group, handle->layout);
        params.local_dup_num_sms = em_unfused_active ? static_cast<int>(group->shuffle_sms) : 0;
        // Device-side backstop bound for recv slot indices: the fixed budget or the derived
        // eager bound (computed above). The scan/count build produces slots below it (DROP
        // masks the rest), so the S2G assert only fires on corrupted or stale routing maps.
        params.max_recv_tokens_per_rank = max_recv_tokens_per_rank;

        // Call dispatch kernel
        int sf_bytes_per_token = 0;
        if (scales_forward) {
            sf_bytes_per_token = num_scales_per_token * scale_elem_bytes;
            params.scale_dtype = scales->datatype;
        }
        // Pull writes EM directly from the source buffers; the push dispatch (FLAT
        // staging) is unused. Inputs are read straight from the source windows.
        if (!em_pull_active) {
            NCCLCHECK(
                nccl_ep::ht::call_dispatch(
                    params,
                    group->ht_aligned_max_tokens,
                    group->ht_tokens_per_chunk,
                    group->rdma_team_size,
                    recipe,
                    pass_direction,
                    static_cast<int>(group->dispatch_num_sms),
                    group->max_dynamic_smem,
                    sf_bytes_per_token,
                    &group->env,
                    stream,
                    x->datatype));
        }

        // Fan primaries out to secondary em_slots in this rank's recv buffer.
        // Use the resolved output array (user window under zero_copy, IPC staging otherwise).
        if (em_unfused_active) {
            nccl_ep::ht::call_local_dup(
                /*expert_output_token=*/
                params.expert_output_token_ptrs[group->lsa_rank],
                /*expert_output_prob=*/
                forward_dispatch ? group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[group->lsa_rank] : nullptr,
                handle->ht.emuf_group_buf,
                handle->ht.emuf_group_count,
                handle->ht.emuf_group_stride,
                group->ht_buffers.dispatch_lsa_S2G_flags,
                /*expected_lsa_flag_val=*/params.expected_lsa_flag_val,
                /*grid_barrier_counter=*/params.dispatch_grid_barrier_counter,
                params.hidden_dim,
                params.experts_per_rank,
                params.lsa_team_size,
                forward_dispatch,
                params.local_dup_num_sms,
                stream,
                x->datatype,
                recipe,
                /*expert_output_scale=*/
                (recipe == NCCL_EP_DISP_QUANT_FWD)
                    ? static_cast<uint8_t*>(params.expert_output_scaling_factor_ptrs[group->lsa_rank])
                    : nullptr,
                /*scale_row_bytes=*/sf_bytes_per_token);
        }
        /* ===== Copy intra-LSA staging → caller outputs ===== */
        // External-window outputs are written directly by the kernel; regular tensors
        // need a D2D copy from the shared intra-LSA staging buffers. On the
        // EM-permute path (HT + EM + zero_copy != ON), the copy is deferred until
        // after dense_to_sparse_prob has populated recv_topk_idx_flat, since the
        // permute kernel uses that table to map FLAT slots to EM zones.
        assert(recv_x->ndim == 2);
        const int caller_num_recv_tokens = static_cast<int>(recv_x->sizes[0]);
        if (!rcv_x_zcopy && !em_permute_active) {
            if (recv_x->sizes[0] < recv_copy_rows) {
                return ncclInvalidArgument;
            }
            // Clamp to staging capacity (see token_staging_slots).
            const unsigned int clamped_rows =
                static_cast<unsigned int>(std::min<size_t>(recv_copy_rows, group->ht_buffers.token_staging_slots));
            size_t copy_size = static_cast<size_t>(clamped_rows) * recv_x->sizes[1] * ncclTypeSize(recv_x->datatype);
            void* src = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->lsa_rank];
            CUDA_CHECK(cudaMemcpyAsync(recv_x->data, src, copy_size, cudaMemcpyDeviceToDevice, stream));
        }

        /* ===== Convert dense output → sparse format ===== */
        if (forward_dispatch) {
            // recv_topk_weights required; shape depends on layout:
            //   EM: 1D [N] (each slot is per-(token, local_expert), at most 1 weight).
            //   FLAT: 2D [N, top_k] paired with required recv_topk_idx [N, top_k].
            if (recv_topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            NCCLCHECK(
                resolveTensorWindowBinding(group, recv_topk_weights, &recv_topk_weights_local, 0, &recv_topk_weights));
            const bool em = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
            if (em) {
                if (recv_topk_idx != nullptr) {
                    return ncclInvalidArgument;
                }
            } else {
                if (recv_topk_idx == nullptr) {
                    return ncclInvalidArgument;
                }
                NCCLCHECK(resolveTensorWindowBinding(group, recv_topk_idx, &recv_topk_idx_local, 0, &recv_topk_idx));
            }
            assert(recv_topk_weights->datatype == ncclFloat32);
            if (em) {
                assert(recv_topk_weights->ndim == 1 && "HT EM recv_topk_weights must be 1D [num_recv_tokens]");
            } else {
                assert(recv_topk_weights->ndim == 2 && "HT FLAT recv_topk_weights must be 2D [num_recv_tokens, top_k]");
                assert(recv_topk_idx->ndim == 2);
                // dense_to_sparse_prob writes recv_topk_idx as int64, so the caller buffer must be int64.
                assert(recv_topk_idx->datatype == ncclInt64);
                if (recv_topk_weights->sizes[0] != recv_topk_idx->sizes[0]) {
                    return ncclInvalidArgument;
                }
            }
            if (recv_topk_weights->sizes[0] < recv_copy_rows) {
                return ncclInvalidArgument;
            }

            // Non-permute paths write the caller buffer, so bound the row count to
            // recv_copy_rows. The em-permute path writes internal scratch here and
            // the caller buffer later in launch_dispatch_permute.
            int num_recv_tokens = em_permute_active ? static_cast<int>(group->max_recv_tokens)
                                                    : static_cast<int>(recv_copy_rows);
            int experts_per_lsa_team = group->num_local_experts * group->lsa_team_size;
            // recv_topk_idx numbering selector (matches LL rank-major path).
            // The normalized layout_info supplies AUTO when the caller did not
            // provide a value.
            const ncclEpExpertIdKind_t recv_topk_idx_kind =
                resolveRecvTopkIdxKind(
                    layout_info != nullptr
                        ? layout_info->recv_topk_idx_kind
                        : NCCL_EP_EXPERT_ID_AUTO);
            const int global_expert_offset = group->rank * group->num_local_experts;

            // Fused-meta-dispatch path: the sender-side table work is LERM only (weights
            // extracted from the landed prob below, flat2em built by the permute), so the
            // dense->sparse extraction runs as in scan mode on all topologies.
            // em-permute path: dense_to_sparse_prob runs in FLAT shape.
            // Weights are written to FLAT scratch (recv_topk_weights_flat) and
            // the topk_idx write is skipped to preserve the em_slot values
            // published by em_scan_kernel. The local permute kernel below
            // reshuffles FLAT scratch into the caller's EM recv buffer.
            {
                const bool dsp_em = em && !em_permute_active;
                float* dsp_topk_weights = em_permute_active ? group->ht_buffers.count_scratch.recv_topk_weights_flat :
                                                              static_cast<float*>(recv_topk_weights->data);
                int64_t* dsp_topk_idx =
                    em_permute_active ? nullptr : (recv_topk_idx ? static_cast<int64_t*>(recv_topk_idx->data) : nullptr);

                // Pull relocates weights directly from the source topk_weights row, so it
                // needs no FLAT weight staging (prob buffer is not populated under pull).
                if (!em_pull_active)
                    nccl_ep::ht::dense_to_sparse_prob(
                        group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[group->lsa_rank],
                        // Count mode reads the IPC LERM landing zone here (pre-permute); the
                        // per-handle copy is not populated until the permute runs below.
                        count_mode ? handle->ht.dispatch_push_count.lerm_landing
                                   : handle->ht.local_expert_routing_map,
                        dsp_topk_weights,
                        dsp_topk_idx,
                        num_recv_tokens,
                        handle->num_topk,
                        group->num_local_experts,
                        experts_per_lsa_team,
                        group->lsa_rank,
                        global_expert_offset,
                        recv_topk_idx_kind,
                        dsp_em,
                        count_mode, // count mode: LERM is a packed bitmap
                        stream);
            }
        }

        // EM-permute path: FLAT staging now holds rank-major token rows, the
        // FLAT scratch holds recv_topk_idx/weights. Run the local permute kernel
        // to write the caller's EM recv_x + EM 1D recv_topk_weights with the
        // correct per-expert zone offsets and zero-pad rows.
        if (em_permute_active) {
            // BWD passes recv_topk_weights == nullptr (weights are FWD-only).
            assert(forward_dispatch ? (recv_topk_weights != nullptr) : (recv_topk_weights == nullptr));
            // Zero-recv FWD (eager): the caller's outputs are legally empty
            // (data == nullptr; emptiness consistency validated pre-kernel above).
            // There are no weights to deliver, so drop BOTH halves of the
            // FLAT->EM weight pair.
            const bool deliver_weights = forward_dispatch && !tensorIsEmpty(recv_x);
            if (recipe == NCCL_EP_DISP_QUANT_NONE) {
                if (!validate_dtype(recv_x->datatype)) {
                    // local_permute_dup is a byte-relocation kernel: any NONE-mode wire
                    // dtype (bf16/fp16/fp32) works; QUANT_FWD rows are 1B (fp8) or
                    // 4B (fp32) and relocate the same way via row_bytes.
                    return ncclInvalidArgument;
                }
            }
            const int row_bytes = static_cast<int>(recv_x->sizes[1]) * ncclTypeSize(recv_x->datatype);
            if (row_bytes <= 0 || (row_bytes % 16) != 0) {
                return ncclInvalidArgument; // int4-vectorized row copy requires 16B-aligned row
            }
            // QUANT_FWD: relocate the FLAT scale rows into EM order alongside the
            // tokens (replaces the straight FLAT->caller scale copy below).
            void* perm_recv_scales_em = nullptr;
            const void* perm_flat_scale_staging = nullptr;
            int perm_scale_row_bytes = 0;
            // Zero-recv FWD (eager): recv_scales is legally empty (data == nullptr)
            // whenever recv_x is — the recipe validation ties their row counts — so
            // there are no scale rows to relocate.
            const bool deliver_scales =
                recipe == NCCL_EP_DISP_QUANT_FWD && !tensorIsEmpty(recv_x);
            if (deliver_scales) {
                if (recv_scales->sizes[0] < recv_copy_rows) {
                    return ncclInvalidArgument; // EM scale slots, like recv_x
                }
                perm_recv_scales_em = recv_scales->data;
                perm_flat_scale_staging =
                    group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[group->lsa_rank];
                perm_scale_row_bytes =
                    static_cast<int>(recv_scales->sizes[1]) * ncclTypeSize(recv_scales->datatype);
            }
            if (em_pull_active) {
                // Pull staging holds one full smem row per warp. The launch already
                // scales its warp count down to the device opt-in smem cap, so only a
                // hidden size whose single row exceeds the cap is unsupported. Reject
                // cleanly here rather than letting the JIT launch abort the process.
                // TODO: fall back to the local-permute (kLocalPermute) dispatch path for
                // this hidden size instead of failing the dispatch under NCCL_EP_HT_EM_PULL_PUSH.
                const size_t pull_smem =
                    nccl_ep::ht::dispatch_pull_smem_bytes(row_bytes / 16);
                if (pull_smem > group->device_smem_optin) {
                    fprintf(stderr,
                            "ncclEpDispatch: NCCL_EP_HT_EM_PULL_PUSH pull dispatch needs %zu B "
                            "shared memory for a single pull warp (hidden row = %d B) but the "
                            "device opt-in cap is %zu B; disable NCCL_EP_HT_EM_PULL_PUSH for this "
                            "hidden size.\n",
                            pull_smem, row_bytes, group->device_smem_optin);
                    return ncclInvalidArgument;
                }
                // Pull: each receiver reads the source token + topk-weight rows from peers
                // over NVLink into EM zones. Peer base pointers ride the kernel launch arg
                // (marshaled per call), so there is no shared host staging to race.
                const int lsa_ranks = group->lsa_team_size;
                std::vector<void*> win_tokens;
                std::vector<float*> win_weights;
                std::vector<void*> win_scales;
                const void* const* peer_in = nullptr;
                const float* const* peer_w = nullptr;
                const void* const* peer_scale = nullptr;
                if (x->win_hdl != ncclWindow_t{}) {
                    // Pull reads the token, weight and scale rows from the same peer-visible
                    // source, so every delivered operand must share x's window backing. Input
                    // windowing is per-tensor opt-in (even under zero_copy=ON, which only
                    // requires the output window), so reject a windowed x paired with a plain
                    // topk_weights/scales instead of reading a non-shared peer address.
                    if (deliver_weights && (topk_weights == nullptr || topk_weights->win_hdl == ncclWindow_t{})) {
                        fprintf(stderr,
                                "ncclEpDispatch: NCCL_EP_HT_EM_PULL_PUSH needs topk_weights window-registered "
                                "when tokens are (mixed windowed/non-windowed pull inputs are unsupported)\n");
                        return ncclInvalidArgument;
                    }
                    if (deliver_scales && (scales == nullptr || scales->win_hdl == ncclWindow_t{})) {
                        fprintf(stderr,
                                "ncclEpDispatch: NCCL_EP_HT_EM_PULL_PUSH needs scales window-registered "
                                "when tokens are (mixed windowed/non-windowed pull inputs are unsupported)\n");
                        return ncclInvalidArgument;
                    }
                    // Window-backed input: peers read it in place (zero-copy, no staging).
                    NCCLCHECK(buildIntranodePtrArray<void>(group, x, win_tokens));
                    peer_in = win_tokens.data();
                    if (deliver_weights) {
                        NCCLCHECK(buildIntranodePtrArray<float>(group, topk_weights, win_weights));
                        peer_w = win_weights.data();
                    }
                    if (deliver_scales) {
                        // QUANT_FWD: peers read each token's scale row over NVLink too.
                        NCCLCHECK(buildIntranodePtrArray<void>(group, scales, win_scales));
                        peer_scale = win_scales.data();
                    }
                } else {
                    // Non-symmetric input: stage into the peer-accessible FLAT dispatch
                    // buffers and point pull at them. The memcpy is stream-ordered before the
                    // kernel and lsa_grid_head_gate rendezvous with every peer before any peer
                    // read, so no peer observes this rank's staging early.
                    assert(group->ht_buffers.expert_output_token != nullptr);
                    CUDA_CHECK(cudaMemcpyAsync(group->ht_buffers.expert_output_token, x->data,
                                               static_cast<size_t>(handle->num_tokens) * row_bytes,
                                               cudaMemcpyDeviceToDevice, stream));
                    peer_in = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs;
                    // A source must stage its input weights/scales for peers to pull even
                    // when it receives zero tokens (deliver_weights/deliver_scales gate only
                    // this rank's own recv output). Stage whenever there is input to forward.
                    if (forward_dispatch && topk_weights != nullptr) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            group->ht_buffers.expert_output_prob, topk_weights->data,
                            static_cast<size_t>(handle->num_tokens) * handle->num_topk * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
                    }
                    if (deliver_weights) {
                        peer_w = group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs;
                    }
                    if (recipe == NCCL_EP_DISP_QUANT_FWD && scales != nullptr) {
                        // Stage input scales into the (pull-unused) FLAT scale output buffer.
                        assert(group->ht_buffers.expert_output_scaling_factor != nullptr);
                        const int send_scale_row_bytes =
                            static_cast<int>(scales->sizes[1]) * ncclTypeSize(scales->datatype);
                        CUDA_CHECK(cudaMemcpyAsync(
                            group->ht_buffers.expert_output_scaling_factor, scales->data,
                            static_cast<size_t>(handle->num_tokens) * send_scale_row_bytes,
                            cudaMemcpyDeviceToDevice, stream));
                    }
                    if (deliver_scales) {
                        peer_scale = const_cast<const void* const*>(
                            group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs);
                    }
                }
                NCCLCHECK(nccl_ep::ht::launch_dispatch_pull(
                    recv_x->data,
                    deliver_weights ? static_cast<float*>(recv_topk_weights->data) : nullptr,
                    deliver_scales ? perm_recv_scales_em : nullptr,
                    handle->ht.flat2em_slot_map,
                    handle->ht.srcpos_map,
                    handle->ht.recv_slot_to_src,
                    peer_in,
                    deliver_weights ? peer_w : nullptr,
                    deliver_scales ? peer_scale : nullptr,
                    handle->ht.num_tokens_for_experts,
                    handle->ht.expert_token_offsets,
                    handle->ht.per_expert_counts_active,
                    handle->num_topk,
                    group->num_local_experts,
                    row_bytes,
                    deliver_scales ? perm_scale_row_bytes : 0,
                    caller_num_recv_tokens,
                    static_cast<int>(group->config.max_dispatch_tokens_per_rank),
                    lsa_ranks,
                    // Pull is the dispatch, so it uses the dispatch SM budget
                    // (dispatch_num_sms), not the shuffle or combine budget.
                    static_cast<int>(group->dispatch_num_sms),
                    0u,
                    recipe,
                    group->gin_config.d_dcomms,
                    group->ht_buffers.combine_grid_barrier_counter,  // head gate (idle during dispatch)
                    group->ht_buffers.dispatch_grid_barrier_counter, // tail elect-last-block
                    stream,
                    nccl_ep_env_flag_on(group->env.ht_unfused_sync)));
            } else {
                // Count mode: the permute builds the flat2em rows inline from the
                // MAP-written LERM (FWD only; BWD reuses the rows persisted by the FWD
                // dispatch). The em cursors are zero-seeded per dispatch.
                const bool count_lean_build = forward_dispatch && count_mode;
                int32_t* em_permute_cursors = nullptr;
                if (count_lean_build) {
                    em_permute_cursors = group->ht_buffers.count_scratch.em_permute_cursors;
                    // +1: the phantom-row fixup's grid arrival counter (trailing slot).
                    CUDA_CHECK(cudaMemsetAsync(
                        em_permute_cursors, 0,
                        static_cast<size_t>(group->num_local_experts + 1) * sizeof(int32_t), stream));
                }
                nccl_ep::ht::launch_dispatch_permute(
                    recv_x->data,
                    deliver_weights ? static_cast<float*>(recv_topk_weights->data) : nullptr,
                    group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->lsa_rank],
                    deliver_weights ? group->ht_buffers.count_scratch.recv_topk_weights_flat : nullptr,
                    handle->ht.flat2em_slot_map,
                    handle->ht.num_tokens_for_experts,
                    handle->ht.expert_token_offsets,
                    handle->ht.per_expert_counts_active,
                    handle->num_topk,
                    group->num_local_experts,
                    row_bytes,
                    static_cast<int>(group->device_sm_count),
                    group->shuffle_sms,
                    caller_num_recv_tokens,
                    stream,
                    recipe,
                    perm_recv_scales_em,
                    perm_flat_scale_staging,
                    perm_scale_row_bytes,
                    count_lean_build ? handle->ht.flat2em_slot_map : nullptr,
                    count_lean_build ? handle->ht.dispatch_push_count.lerm_landing : nullptr,
                    count_lean_build ? handle->ht.local_expert_routing_map : nullptr,
                    em_permute_cursors,
                    count_lean_build && group->config.overflow_policy == NCCL_EP_OVERFLOW_DROP);
            }
        }

        // QUANT_FWD output scales (async D2D, sized by caller). On the EM-permute
        // path the scales are already relocated into EM order by local_permute_dup above.
        if (recipe == NCCL_EP_DISP_QUANT_FWD && !em_permute_active) {
            assert(recv_scales->ndim == 2);
            if (!rcv_scales_zcopy) {
                if (recv_scales->sizes[0] < recv_copy_rows) {
                    return ncclInvalidArgument;
                }
                size_t copy_size =
                    static_cast<size_t>(recv_copy_rows) * recv_scales->sizes[1] * ncclTypeSize(recv_scales->datatype);
                CUDA_CHECK(cudaMemcpyAsync(
                    recv_scales->data,
                    group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[group->lsa_rank],
                    copy_size,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
    }
    return ncclSuccess;
}

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream) {
    EP_VALIDATE_STRUCT(inputs, NCCL_EP_COMBINE_INPUTS);
    EP_VALIDATE_STRUCT(outputs, NCCL_EP_COMBINE_OUTPUTS);
    if (config != nullptr) {
        EP_VALIDATE_STRUCT(config, NCCL_EP_COMBINE_CONFIG);
    }

    ncclEpCombineInputs_t parsed_inputs =
        epDecodeStruct(inputs, NCCL_EP_COMBINE_INPUTS_INIT);
    ncclEpCombineOutputs_t parsed_outputs =
        epDecodeStruct(outputs, NCCL_EP_COMBINE_OUTPUTS_INIT);
    inputs = &parsed_inputs;
    outputs = &parsed_outputs;

    ncclEpCombineConfig_t parsed_config = NCCL_EP_COMBINE_CONFIG_INIT;
    if (config != nullptr) {
        parsed_config = epDecodeStruct(config, NCCL_EP_COMBINE_CONFIG_INIT);
        config = &parsed_config;
    }

    const unsigned int send_only = config ? config->send_only : 0;
    const ncclEpPassDir_t pass_direction = config ? config->pass_direction : NCCL_EP_FWD_PASS;
    const ncclEpCombQuant_t quantization_recipe =
        config ? config->quant_recipe : NCCL_EP_COMB_QUANT_NONE;
    NCCLCHECK(validateCombineRecipe(
        inputs, outputs, config, handle->group->config.algorithm, handle->group->device_sm,
        is_internode_available(handle->group)));
    if (pass_direction != NCCL_EP_FWD_PASS && handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        fprintf(stderr, "ncclEpCombine: backward pass (pass_direction=%d) is not supported in LL mode\n",
                (int)pass_direction);
        return ncclInvalidUsage;
    }

    // Lazy num_tokens for callers that skip UpdateHandle (e.g. handle relocation between prepare and combine).
    // Guard on `num_tokens_set` so a legitimate zero-token configuration is
    // preserved (the outputs->tokens descriptor for a zero-token rank may
    // carry data == nullptr, which is now legal per tensorHasBinding's
    // empty-tensor relaxation but would still fail the assert on dim 0).
    if (!handle->num_tokens_set) {
        const ncclEpTensor_t* lazy_combined = tensor_required(outputs->tokens);
        assert(lazy_combined->ndim > 0);
        handle->num_tokens = static_cast<int>(lazy_combined->sizes[0]);
        handle->num_tokens_set = true;
    }

    // Consolidated token-dtype gate for the NONE recipe on all combine paths (LL/HT):
    // bf16/fp16/fp32. Quantized recipes state their own, stricter dtype contract in
    // validateCombineRecipe above -- MXFP8 requires BF16 tokens and a null scales tensor --
    // so they skip this check rather than repeat it. Checked once here.
    {
        const ncclEpCombQuant_t gate_recipe =
            config ? config->quant_recipe : NCCL_EP_COMB_QUANT_NONE;
        const ncclEpTensor_t* xt = tensor_required(inputs->tokens);
        if (gate_recipe == NCCL_EP_COMB_QUANT_NONE && !validate_dtype(xt->datatype)) {
            fprintf(stderr, "NCCL EP: combine unsupported token dtype %d\n", static_cast<int>(xt->datatype));
            return ncclInvalidArgument;
        }
    }

    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // Find and validate input tensors
        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        const ncclEpTensor_t* global_scales = tensor_ptr(inputs->scales);
        assert(x->ndim > 0);

        const ncclEpTensor_t* topk_idx = &handle->topk_idx;
        const ncclEpTensor_t* src_info = &handle->ll.expert_recv_source_indices;
        const ncclEpTensor_t* layout_range = &handle->ll.expert_dispatch_layout;

        // topk_weights: expert-major requires it in outputs; rank-major does not
        // (weights are applied by the caller in preReduceRankMajor before ncclEpCombine).
        const ncclEpTensor_t* topk_weights;
        if (handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
            topk_weights = nullptr;
        } else {
            topk_weights = tensor_ptr(outputs->topk_weights);
            EP_HOST_ASSERT(topk_weights != nullptr && "expert-major combine requires topk_weights in outputs");
        }

        // Extract configuration values
        const int num_experts = handle->group->config.num_experts;
        const int num_ranks = handle->group->nRanks;
        const int num_max_dispatch_tokens_per_rank = handle->group->config.max_dispatch_tokens_per_rank;

        // Extract hidden dimension (index differs by layout).
        int hidden;
        switch (handle->layout) {
        case NCCL_EP_LAYOUT_RANK_MAJOR:
            // Rank-major input is 3D so the per-rank, per-slot structure is
            // explicit in the descriptor — the caller must acknowledge it.
            assert(x->ndim == 3);
            assert(x->sizes[0] == static_cast<unsigned>(num_ranks));
            assert(x->sizes[1] == static_cast<unsigned>(num_max_dispatch_tokens_per_rank));
            assert(x->sizes[2] % sizeof(int4) == 0);
            assert(x->sizes[2] % 128 == 0);
            hidden = static_cast<int>(x->sizes[2]);
            break;
        default:
            assert(x->ndim == 3);
            assert(x->sizes[0] == num_experts / num_ranks);
            assert(x->sizes[1] == static_cast<unsigned>(num_ranks) * num_max_dispatch_tokens_per_rank);
            assert(x->sizes[2] % sizeof(int4) == 0);
            assert(x->sizes[2] % 128 == 0);
            hidden = static_cast<int>(x->sizes[2]);
            break;
        }

        if (quantization_recipe == NCCL_EP_COMB_QUANT_NVFP4) {
            constexpr int kNvfp4WarpSize = 32;
            constexpr int kNvfp4MaxSendUnrolls = 4;
            constexpr int kNvfp4ElementsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
            const int nvfp4_send_unrolls =
                hidden % (kNvfp4WarpSize * kNvfp4MaxSendUnrolls * kNvfp4ElementsPerInt4) == 0
                ? kNvfp4MaxSendUnrolls : 2;
            const int nvfp4_send_iteration_elements =
                kNvfp4WarpSize * nvfp4_send_unrolls * kNvfp4ElementsPerInt4;
            const size_t expected_slots = static_cast<size_t>(num_ranks) * num_max_dispatch_tokens_per_rank;
            const bool expert_major_scales =
                global_scales != nullptr && global_scales->ndim == 3 &&
                global_scales->sizes[0] == static_cast<size_t>(num_experts / num_ranks) &&
                global_scales->sizes[1] == expected_slots;
            const bool rank_major_scales =
                global_scales != nullptr && global_scales->ndim == 3 &&
                global_scales->sizes[0] == static_cast<size_t>(num_ranks) &&
                global_scales->sizes[1] == static_cast<size_t>(num_max_dispatch_tokens_per_rank);
            const bool valid_scales = handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR
                ? expert_major_scales : rank_major_scales;
            if (hidden % nvfp4_send_iteration_elements != 0 ||
                global_scales->sizes[2] != 1 || !valid_scales) {
                fprintf(stderr, "NCCL EP: LL NVFP4 combine requires a hidden dimension with whole send iterations and FP32 scales "
                                "[local_experts, recv_slots, 1] (expert-major) or [ranks, max_tokens, 1] (rank-major)\n");
                return ncclInvalidArgument;
            }
        }

        // Validate topk_idx tensor (LL: int32 or int64; dtype must
        // match what ncclEpUpdateHandle saw).
        assert(topk_idx->ndim == 2);
        assert(
            (topk_idx->datatype == ncclInt32 || topk_idx->datatype == ncclInt64) &&
            "LL topk_idx must be ncclInt32 or ncclInt64");

        // Validate src_info tensor
        assert(src_info->ndim == 1);
        assert(src_info->datatype == ncclInt32);

        // Validate topk_weights tensor (expert-major only; rank-major applies weights before combine)
        if (topk_weights != nullptr) {
            assert(topk_weights->ndim == 2);
            assert(topk_weights->sizes[0] == topk_idx->sizes[0]);
            assert(topk_weights->sizes[1] == topk_idx->sizes[1]);
            assert(topk_weights->datatype == ncclFloat32);
            assert(topk_weights->sizes[0] <= num_max_dispatch_tokens_per_rank);
        }

        // Extract dimensions (hidden already set in the layout switch above)
        const int num_topk = static_cast<int>(topk_idx->sizes[1]);
        const int num_combined_tokens = static_cast<int>(topk_idx->sizes[0]);

        // Manage double-buffering
        const auto& buffer = handle->ll.layout.buffer;
        const auto next_clean_meta = buffer.clean_meta_offset();

        // Validate buffer layout
        assert(handle->ll.layout.total_bytes <= handle->group->rdma_buffer_size_alloc);

        // Find and validate output tensor
        const ncclEpTensor_t* out = tensor_required(outputs->tokens);

        assert(out->ndim > 0);
        assert(out->ndim == 2);
        assert(out->sizes[0] == num_combined_tokens);
        assert(out->sizes[1] == hidden);
        assert(out->datatype == x->datatype);

        if (nccl_ep_env_flag_on(handle->group->env.debug)) {
            fprintf(
                stderr,
                "[nccl_ep][debug][rank %d] LL combine zero-copy disabled: "
                "LL combine does not expose a direct-window path (requested=%d)\n",
                handle->group->rank,
                static_cast<int>(handle->group->config.zero_copy));
        }

        // Define combine lambda
        unsigned signal_base = 0;
        auto combine_fn = [=](int phases) -> ncclResult_t {
            void* const rdma_buf = handle->group->rdma_buffer;
            // Prepare data pointers
            auto* out_data = out->data;
            auto* x_data = x->data;
            auto* topk_weights_data = topk_weights ? static_cast<float*>(topk_weights->data) : nullptr;
            auto* src_info_data = static_cast<int*>(src_info->data);
            auto* layout_range_data = static_cast<int64_t*>(layout_range->data);

            // LL accepts int32 or int64 topk_idx; the cached dtype picks the JIT
            // combine specialization (TopkIdxT), same as dispatch. zeroCopy stays
            // false: combine reads inData directly and uses windows only to
            // translate peer recv-buffer ptrs; the kernel's zeroCopy mode is
            // dispatch-shaped, so config.zero_copy is a dispatch-only switch.
            auto launch_combine = [&](auto* topk_idx_data, bool topk_is_int64) {
                nccl_ep::ll::CombineParams params{};
                params.inData = x_data;
                params.inGlobalScales = global_scales ? static_cast<const float*>(global_scales->data) : nullptr;
                params.srcInfo = src_info_data;
                params.layoutRange = layout_range_data;
                params.inTopkIdx = topk_idx_data;
                params.topkIdxIsInt64 = topk_is_int64;
                params.topkWeights = topk_weights_data;
                params.outData = out_data;
                params.rdmaBuf = rdma_buf;
                params.sendOff = buffer.combine_rdma_send_buffer_offset;
                params.recvOff = buffer.combine_rdma_recv_data_buffer_offset;
                params.recvFlagOff = buffer.combine_rdma_recv_flag_buffer_offset;
                params.nextRecvCntBufSize = next_clean_meta.second;
                params.waitStats = nullptr;
                params.epochState = handle->group->ll_epoch_state;
                params.payloadSlotStride = handle->ll.layout.payload_slot_stride;
                params.signalSlotStride = handle->ll.layout.signal_slot_stride;
                params.numCombinedTokens = num_combined_tokens;
                params.hidden = hidden;
                params.maxTokensPerRank = num_max_dispatch_tokens_per_rank;
                params.numTopk = num_topk;
                params.numExperts = num_experts;
                params.currRank = handle->group->rank;
                params.numRanks = handle->group->nRanks;
                params.layout = handle->layout;
                params.numComms = handle->group->num_nccl_comms;
                params.devComms = handle->group->nccl_dev_comms;
                params.windows = handle->group->nccl_wins;
                params.signalsBase = signal_base;
                params.workspace = handle->group->ep_workspace;
                params.numDeviceSms = handle->group->combine_num_sms;
                params.deviceSm = handle->group->device_sm;
                params.maxDynamicSmem = handle->group->max_dynamic_smem;
                params.resolvedWarpsPerGroup = &handle->group->last_ll_combine_warps_per_group;
                params.rankMask = handle->group->mask_buffer;
                params.asyncErrorFlag = handle->group->async_error_flag;
                params.timeoutCycles = handle->group->timeout_cycles;
                // LogFMT compression is not wired into the LL combine API or call flow;
                // keep the retained JIT template path disabled until that integration exists.
                params.useLogFmt = false;
                params.zeroCopy = false;
                params.phases = phases;
                params.tokenDtype = x->datatype;
                params.quantizationRecipe = quantization_recipe;
                return nccl_ep::ll::call_combine(params, stream);
            };
            switch (topk_idx->datatype) {
            case ncclInt32:
                return launch_combine(static_cast<const int32_t*>(topk_idx->data), /*topk_is_int64=*/false);
            case ncclInt64:
                return launch_combine(static_cast<const int64_t*>(topk_idx->data), /*topk_is_int64=*/true);
            default:
                    std::fprintf(stderr, "NCCL EP warning: LL topk_idx has unsupported dtype %d\n",
                                 static_cast<int>(topk_idx->datatype));
                    return ncclInvalidArgument;
            }
        };

        // Execute combine with appropriate phase flags
        const int combine_phases =
            send_only ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        NCCLCHECK(combine_fn(combine_phases));

        if (send_only) {
            handle->ll.continue_fn = combine_fn;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // ===== HT mode =====
        // Combine: gather expert outputs back to original token positions
        // Forward combine: just tokens (inference)
        // Backward combine: tokens + gradients (training, topk_weights provided)
        ncclEpGroup_t group = handle->group;
        bool is_lsa_only = !is_internode_available(group);
        // assert(is_lsa_only && "HT mode only supports single-node");

        /* ===== Inputs validation ===== */
        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        if (x->ndim == 0) {
            return ncclInvalidArgument;
        }
        assert(x->ndim == 2);
        // Local copies for tensors that need window resolution; only populated when used.
        ncclEpTensor_t x_local, topk_weights_local, combined_topk_weights_local, combined_x_local;
        NCCLCHECK(resolveTensorWindowBinding(
            group,
            x,
            &x_local,
            static_cast<uint64_t>(group->gin_config.combine_red_token_offset),
            &x));

        // Get dimensions from input tensor
        auto num_tokens = static_cast<int>(x->sizes[0]);
        auto hidden = static_cast<int>(x->sizes[1]);

        const ncclEpCombQuant_t combine_recipe =
            config ? config->quant_recipe : NCCL_EP_COMB_QUANT_NONE;
        const bool mxfp8_combine = (combine_recipe == NCCL_EP_COMB_QUANT_MXFP8);

        // Validate int4 alignment for TMA
        assert((hidden * ncclTypeSize(x->datatype)) % sizeof(int4) == 0);

        // Number of tokens to combine back (original token count from this rank)
        auto num_combined_tokens = handle->num_tokens;

        // Top-k checks (for backward mode)
        // Output combined_topk_weights is always 2D [num_combined_tokens, source_top_k].
        // Input topk_weights shape MUST match the FWD recv_topk_weights shape by layout:
        //   FLAT/RM: 2D [num_recv_tokens, source_top_k]
        //   EM:      1D [num_recv_tokens]
        // The scatter kernel sparse_to_dense_prob_combine_kernel scans local experts per
        // recv slot; under EM each slot maps to exactly one local expert, so the inner
        // stride is 1.
        int num_topk = 0;
        int input_topk_stride = 0;
        const ncclEpTensor_t* topk_weights = tensor_ptr(inputs->topk_weights);
        const ncclEpTensor_t* combined_topk_weights = tensor_ptr(outputs->topk_weights);

        // Pass direction is the source of truth (default FWD via zero-init).
        // BWD combine requires inputs->topk_weights and outputs->topk_weights;
        // FWD combine forbids inputs->topk_weights (outputs->topk_weights is unused).
        const bool backward_combine = (pass_direction == NCCL_EP_BWD_PASS);
        const bool expert_major_in = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

        if (backward_combine) {
            if (topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            if (combined_topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            assert(combined_topk_weights->ndim == 2);
            assert(combined_topk_weights->sizes[0] == num_combined_tokens);
            assert(combined_topk_weights->datatype == ncclFloat32);
            num_topk = static_cast<int>(combined_topk_weights->sizes[1]);
            // Input shape validation by layout — must match FWD recv_topk_weights.
            assert(topk_weights->datatype == ncclFloat32);
            // Zero-recv BWD (eager): grad rows and their weight grads are rows of
            // the same recv set, so their emptiness must agree; a mismatch is a
            // malformed call, not a zero-work one.
            if (tensorIsEmpty(topk_weights) != tensorIsEmpty(x)) {
                return ncclInvalidArgument;
            }
            if (expert_major_in) {
                assert(topk_weights->ndim == 1 && "HT EM BWD combine: input topk_weights must be 1D [num_recv_tokens]");
                input_topk_stride = 1;
            } else {
                assert(
                    topk_weights->ndim == 2 &&
                    "HT FLAT/RM BWD combine: input topk_weights must be 2D [num_recv, top_k]");
                assert(
                    static_cast<int>(topk_weights->sizes[1]) == num_topk &&
                    "HT FLAT/RM BWD combine: input top_k must equal output top_k");
                input_topk_stride = num_topk;
            }
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                topk_weights,
                &topk_weights_local,
                static_cast<uint64_t>(group->gin_config.combine_red_prob_offset),
                &topk_weights));
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                combined_topk_weights,
                &combined_topk_weights_local,
                0,
                &combined_topk_weights));
        } else {
            // FWD combine: input topk_weights forbidden. outputs->topk_weights is unused
            // by the kernel here and left to caller bookkeeping (not validated).
            if (topk_weights != nullptr) {
                return ncclInvalidArgument;
            }
        }

        /* ===== Output tensors ===== */
        const ncclEpTensor_t* combined_x = tensor_required(outputs->tokens);
        if (combined_x->ndim == 0) {
            return ncclInvalidArgument;
        }
        assert(combined_x->ndim == 2);
        assert(combined_x->sizes[0] == num_combined_tokens); // Output should match original token count
        assert(combined_x->sizes[1] == hidden); // Should match input hidden dimension
        NCCLCHECK(resolveTensorWindowBinding(group, combined_x, &combined_x_local, 0, &combined_x));

        /* ===== Copy input to IPC staging buffers ===== */
        // Expert MLP output needs to be in IPC buffer so other ranks can read it
        const bool combine_x_uses_external_window = tensorUsesExternalWindow(group, x);
        if (group->config.zero_copy == NCCL_EP_ZERO_COPY_ON && !combine_x_uses_external_window) {
            fprintf(
                stderr,
                "NCCL EP: zero_copy requires ncclEpCombine inputs->tokens to be backed by a "
                "user-registered NCCL window (ncclCommWindowRegister)\n");
            return ncclInvalidArgument;
        }
        // em-permute path: gather caller EM x into FLAT staging via
        // local_permute_reduce so combine can run with FLAT layout. The kernel
        // also fuses a 1D EM to [num_flat, top_k] FLAT weight gather on BWD.
        const bool em_permute_combine = em_local_permute_enabled(group, handle);
        const bool direct_window_combine =
            combine_x_uses_external_window && !em_permute_combine;
        if (nccl_ep_env_flag_on(group->env.debug)) {
            fprintf(
                stderr,
                "[nccl_ep][debug][rank %d] HT combine zero-copy %s: %s "
                "(requested=%d, token_window=%d, expert_major_permute=%d)\n",
                group->rank,
                direct_window_combine ? "enabled" : "disabled",
                direct_window_combine
                    ? "inputs->tokens external window selected"
                    : combine_x_uses_external_window && em_permute_combine
                    ? "expert-major local_permute gathers the external window into staging"
                    : "inputs->tokens is not backed by a compatible external window",
                static_cast<int>(group->config.zero_copy),
                static_cast<int>(combine_x_uses_external_window),
                static_cast<int>(em_permute_combine));
        }
        // Zero-recv BWD: no weight grads to gather — drop BOTH halves of the
        // EM->FLAT pair so the launcher's null-pairing assert keeps guarding
        // half-pairs on nonempty calls. The FLAT scratch's only consumer is the
        // dense-prob scatter below, which is a no-op at num_tokens == 0.
        const bool em_permute_bwd_weights =
            em_permute_combine && backward_combine && expert_major_in && !tensorIsEmpty(x);
        // em-permute combine always runs the local_permute_reduce gather
        // (caller EM x -> FLAT staging), regardless of external-window — x->data
        // is the resolved device pointer either way.
        // Hoisted above the push/pull branch: used by the BWD scatter after call_combine.
        float* dense_output_prob = handle->ht.dense_prob_buffer;
        const bool comb_push = em_comb_push_enabled(group, handle);
        // Push combine is single-LSA-team only (kPullPush enforces this at group creation).
        // Fail loudly if a multi-team handle reaches here instead of silently running a
        // different combine path.
        if (comb_push && !is_lsa_only) {
            fprintf(stderr, "ncclEpCombine: NCCL_EP_HT_EM_PULL_PUSH push combine is single-LSA-team only\n");
            return ncclInvalidUsage;
        }
        // MXFP8 quantization is fused into local_permute_reduce, which only runs in
        // expert-major local-permute mode. Any other layout or EM mode has no kernel that
        // could produce FP8+E8M0, so reject instead of silently transporting BF16.
        // kPullPush satisfies em_permute_combine but routes through combine_push /
        // combine_reduce_stage, neither of which quantizes, so reject it here too --
        // before the branch split, so both paths are covered.
        if (mxfp8_combine && (!em_permute_combine || comb_push)) {
            fprintf(
                stderr,
                "NCCL EP error: MXFP8 combine requires the expert-major local-permute path "
                "(layout=%d, em_permute=%d, push_combine=%d)\n",
                static_cast<int>(handle->layout),
                static_cast<int>(em_permute_combine),
                static_cast<int>(comb_push));
            return ncclInvalidArgument;
        }
        if (comb_push) {
            // Push EM combine (FWD, NONE, single LSA team): each expert rank locally
            // reduces its K em copies and pushes the row into the destination attn
            // rank's staging over NVLink; a final team_size reduce writes attn_output.
            const int row_bytes = hidden * ncclTypeSize(x->datatype);
            if (row_bytes <= 0 || (row_bytes % 16) != 0) {
                return ncclInvalidArgument; // int4-vectorized row copy requires 16B-aligned row
            }
            const int team_size = group->lsa_team_size;
            void* staging = group->ht_buffers.expert_input_token;
            // Which (t,R) partials exist is derived locally in the reduce from this rank's
            // own routing (per-handle uint16 topk snapshot), so no per-(t,R) valid flag is
            // pushed over NVLink and no flag zero-init is needed. Unwritten staging rows
            // are still skipped, so the full staging memset is avoided.
            assert(handle->ht.topk_idx != nullptr &&
                   "HT push combine: topk_idx snapshot missing (ncclEpUpdateHandle not called?)");
            const uint16_t* self_topk_idx = static_cast<const uint16_t*>(handle->ht.topk_idx);
            // TODO(em-pp): forward combine_reduce could read present_ranks straight from
            // this rank's slice of handle->ht.token_rank_mask and skip the per-lane topk
            // decode + __match_any_sync dedup entirely; backward still needs the per-lane
            // src_rank_of_lane for the srcpos pad lookup.
            if (backward_combine) {
                // The reduce writes combined_topk_weights with stride handle->num_topk.
                assert(num_topk == handle->num_topk &&
                       "push-combine backward: combined_topk_weights width must equal handle->num_topk");
            }
            // Peer staging + flag (+ backward prob-staging) base pointers ride the kernel
            // launch arg (marshaled per call), so there is no shared host staging to race.
            NCCLCHECK(nccl_ep::ht::launch_combine_push(
                reinterpret_cast<void* const*>(group->ht_buffers.combine_expert_input_token_buffer_ptrs),
                x->data,
                handle->ht.flat2em_slot_map,
                handle->ht.recv_slot_to_src,
                handle->ht.num_tokens_for_experts,
                group->gin_config.d_dcomms,
                group->ht_buffers.dispatch_grid_barrier_counter, // head gate (idle during combine)
                group->ht_buffers.combine_grid_barrier_counter,  // tail elect-last-block
                handle->num_topk,
                row_bytes,
                num_tokens,  // caller EM buffer rows: slot backstop in the kernel
                group->lsa_rank,
                static_cast<int>(group->config.max_dispatch_tokens_per_rank),
                team_size,
                // Push combine is a comm kernel (NVLink push), so it uses the combine SM budget.
                static_cast<int>(group->combine_num_sms),
                0u,
                stream,
                x->datatype,
                backward_combine ? static_cast<const float*>(topk_weights->data) : nullptr,
                backward_combine ? handle->ht.srcpos_map : nullptr,
                backward_combine,
                nccl_ep_env_flag_on(group->env.ht_unfused_sync)));
            NCCLCHECK(nccl_ep::ht::launch_combine_reduce_stage(
                combined_x->data,
                staging,
                self_topk_idx,
                handle->num_topk,
                group->num_local_experts,
                group->lsa_team_size * group->num_local_experts,
                num_combined_tokens,
                team_size,
                row_bytes,
                static_cast<int>(group->device_sm_count),
                group->shuffle_sms,
                stream,
                x->datatype,
                backward_combine ? static_cast<float*>(combined_topk_weights->data) : nullptr,
                backward_combine ? handle->num_topk : 0,
                backward_combine));
        } else {
            if (em_permute_combine) {
                // MXFP8: app provides BF16; local_permute_reduce accumulates and quantizes to
                // packed [FP8 H | E8M0 H/32] (same layout as the RDMA wire) in expert_input,
                // and the combine kernel dequantizes in the RED warp.
                // NONE: output row matches the caller's dtype (bf16/fp16/fp32).
                // Bytes of the row the reduce kernel reads. What it writes is the recipe's
                // packed width, derived in the adapter where the device geometry lives.
                const int input_row_bytes = hidden * static_cast<int>(ncclTypeSize(x->datatype));
                if (input_row_bytes <= 0 || (input_row_bytes % 16) != 0) {
                    return ncclInvalidArgument;
                }
                // Count mode reads the per-handle flat2em built by the dispatch permute, same
                // as the scan path.
                NCCLCHECK(nccl_ep::ht::launch_combine_reduce(
                    group->ht_buffers.expert_input_token,
                    x->data,
                    handle->ht.flat2em_slot_map,
                    handle->ht.num_tokens_for_experts,
                    em_permute_bwd_weights ? static_cast<const float*>(topk_weights->data) : nullptr,
                    em_permute_bwd_weights ? group->ht_buffers.count_scratch.recv_topk_weights_flat : nullptr,
                    handle->num_topk,
                    hidden,
                    input_row_bytes,
                    num_tokens,  // caller EM buffer rows: slot backstop in the kernel
                    static_cast<int>(group->device_sm_count),
                    group->shuffle_sms,
                    stream,
                    x->datatype,
                    mxfp8_combine ? NCCL_EP_COMB_QUANT_MXFP8 : NCCL_EP_COMB_QUANT_NONE));
            } else if (!combine_x_uses_external_window) {
                // Clamp to staging capacity (nvlink_dup/local_dup EM only).
                const size_t clamped_tokens =
                    std::min<size_t>(static_cast<size_t>(num_tokens), group->ht_buffers.token_staging_slots);
                size_t token_copy_size = clamped_tokens * hidden * ncclTypeSize(x->datatype);
                CUDA_CHECK(cudaMemcpyAsync(
                    group->ht_buffers.expert_input_token,
                    x->data,
                    token_copy_size,
                    cudaMemcpyDeviceToDevice,
                    stream));
                // No scale copy: MXFP8 is rejected above unless em_permute_combine, and its
                // E8M0 scales ride in the packed token row, not a separate buffer.
            }

            /* ===== Convert sparse topk_weights to dense prob for backward combine ===== */
            // For backward combine, convert sparse input weights to dense format for HT kernel
            if (backward_combine) {
                int experts_per_lsa_team = group->num_local_experts * group->lsa_team_size;
                size_t dense_prob_size = static_cast<size_t>(num_tokens) * experts_per_lsa_team * sizeof(float);

                // Zero-initialize the dense prob buffer before scattering
                CUDA_CHECK(cudaMemsetAsync(
                    group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->lsa_rank],
                    0,
                    dense_prob_size,
                    stream));

                // Scatter sparse weights into dense prob [num_recv, experts_per_lsa_team].
                // em-permute uses FLAT inputs (gathered above for EM, or passed
                // through for FLAT) keyed by the FLAT main LERM; nvlink_dup/local_dup EM keeps
                // the 1D EM input keyed by the EM-shape LERM scratch.
                const bool use_flat_inputs = em_permute_combine;
                const float* prob_input = (use_flat_inputs && expert_major_in) ?
                                              group->ht_buffers.count_scratch.recv_topk_weights_flat :
                                              static_cast<const float*>(topk_weights->data);
                const int prob_stride = use_flat_inputs ? num_topk : input_topk_stride;
                const bool* lerm_for_combine = handle->ht.local_expert_routing_map;
                // Count mode persists LERM as a packed bitmap; scan mode as one bool per expert.
                const bool lerm_bitmap = dispatch_push_count_active(group, handle);
                nccl_ep::ht::sparse_to_dense_prob_combine(
                    prob_input,
                    lerm_for_combine,
                    group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->lsa_rank],
                    num_tokens,
                    prob_stride,
                    group->num_local_experts, // experts_per_rank
                    experts_per_lsa_team,
                    group->lsa_rank,
                    lerm_bitmap,
                    stream);
            }

            // Use pre-allocated dense prob buffer for backward combine
            if (backward_combine) {
                size_t dense_output_prob_size =
                    static_cast<size_t>(num_combined_tokens) * group->config.num_experts * sizeof(float);
                CUDA_CHECK(cudaMemsetAsync(dense_output_prob, 0, dense_output_prob_size, stream));
            }

            /* ===== Build CombineParams ===== */
            // CombineParams encapsulates all buffers and metadata needed by HT combine kernel:
            //   - IPC input buffers: expert_input_token_ptrs, expert_input_prob_ptrs (per-rank pointers)
            //   - Output buffers: attn_output_token, attn_output_prob (user-provided)
            //   - RDMA buffers: combine_gin_RED_*, combine_gin_G2S_* (for multi-LSA-team)
            //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, local_expert_routing_map
            //   - Sync flags: expected_*_flag_val, lsa_S2G_flags
            nccl_ep::ht::CombineParams params;
            params.hidden_dim = hidden;
            params.experts_per_rank = group->num_local_experts;
            params.lsa_team_size = group->lsa_team_size;
            // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access
            std::vector<uint16_t*> combine_input_token_ptrs;
            // em-permute combine always reads the FLAT staging, not the caller tensor.
            if (combine_x_uses_external_window && !em_permute_combine) {
                NCCLCHECK(buildIntranodePtrArray<uint16_t>(group, x, combine_input_token_ptrs));
                params.expert_input_token_ptrs = combine_input_token_ptrs.data();
            } else {
                params.expert_input_token_ptrs = group->ht_buffers.combine_expert_input_token_buffer_ptrs;
            }
            params.expert_input_prob_ptrs =
                backward_combine ? group->ht_buffers.combine_expert_input_prob_buffer_ptrs : nullptr;
            params.combine_recipe = mxfp8_combine ? NCCL_EP_COMB_QUANT_MXFP8 : NCCL_EP_COMB_QUANT_NONE;
            params.attn_output_token = combined_x->data;
            params.attn_output_prob = backward_combine ? dense_output_prob : nullptr;
            params.combine_gin_RED_tokens = is_lsa_only ? nullptr : group->ht_buffers.combine_gin_RED_tokens;
            params.combine_gin_RED_prob =
                (!is_lsa_only && backward_combine) ? group->ht_buffers.combine_gin_RED_prob : nullptr;
            params.combine_gin_G2S_tokens =
                is_lsa_only ? nullptr : group->ht_buffers.combine_gin_G2S_tokens;
            params.combine_gin_G2S_prob =
                (!is_lsa_only && backward_combine) ? group->ht_buffers.combine_gin_G2S_prob : nullptr;
            // Unified s2d: FLAT-shape for FLAT layout and EM em-permute mode;
            // EM-shape (packed rank/slot) only for EM kNvlinkDup / kLocalDup modes.
            params.sparse_to_dense_map = handle->ht.sparse_to_dense_map;
            const bool expert_major = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
            params.s2d_inner_dim = (expert_major && !em_permute_combine) ? handle->num_topk : group->lsa_team_size;
            params.layout = em_permute_combine ? NCCL_EP_LAYOUT_RANK_MAJOR : handle->layout;
            assert(
                (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? (params.s2d_inner_dim == handle->num_topk) :
                                                                 (params.s2d_inner_dim == group->lsa_team_size));
            params.rdma_to_attn_map = handle->ht.rdma_to_attn_map;
            params.attn_to_rdma_map = handle->ht.attn_to_rdma_map;
            params.local_expert_routing_map = handle->ht.local_expert_routing_map;
            // Always pass a valid device pointer — see dispatch path comment.
            params.expected_gin_flag_val = group->ht_buffers.combine_expected_gin_flag_val;
            params.gin_G2S_flags =
                is_lsa_only ? nullptr : group->ht_buffers.combine_gin_G2S_flags;
            params.expected_lsa_flag_val = group->ht_buffers.combine_expected_lsa_flag_val;
            params.combine_grid_barrier_counter = group->ht_buffers.combine_grid_barrier_counter;
            params.lsa_S2G_flags = group->ht_buffers.combine_lsa_S2G_flags;
            params.guard_enabled = !nccl_ep_env_flag_on(group->env.disable_guard);
            params.combine_barrier_offset = group->dispatch_num_sms;
            const ncclWindow_t combine_token_window =
                !combine_x_uses_external_window ? x->win_hdl : group->gin_config.nccl_window;
            const size_t combine_token_offset =
                is_lsa_only ? 0 :
                                 (!combine_x_uses_external_window ? static_cast<size_t>(x->win_offset) :
                                                                    group->gin_config.combine_red_token_offset);
            // Pass device communicators and windows
            // Always pass the devComm (single-LSA-team too): the HT LSA sync-guard now uses the
            // NCCL LSA barrier (needs comm.lsaBarrier). RDMA paths stay if-constexpr-gated.
            params.dcomms = group->gin_config.d_dcomms;
            params.nccl_token_window = combine_token_window;
            params.nccl_prob_window = !backward_combine ? ncclWindow_t{} : group->gin_config.nccl_window;
            params.nccl_internal_window = group->gin_config.nccl_window;
            params.num_gin_comms = is_lsa_only ? 0 : group->gin_config.num_comms;
            params.num_ctx_per_comm = is_lsa_only ? 0 : group->gin_config.num_ctx_per_comm;
            params.gin_base_ptr = is_lsa_only ? nullptr : group->gin_config.gin_base_ptr;
            params.signals_base = group->gin_config.signals_base;
            params.combine_signal_offset = group->gin_config.combine_signal_offset;
            // Use offsets relative to gin_base_ptr
            params.mr_info = {
                .combine_red_token_offset = combine_token_offset,
                .combine_g2s_token_offset =
                    is_lsa_only ? 0 : group->gin_config.combine_g2s_token_offset,
                .combine_red_prob_offset = is_lsa_only ? 0 : group->gin_config.combine_red_prob_offset,
                .combine_g2s_prob_offset =
                    is_lsa_only ? 0 : group->gin_config.combine_g2s_prob_offset,
                .guard_offset = is_lsa_only ? 0 : group->gin_config.combine_guard_offset,
            };
            params.local_rank = group->lsa_rank;
            params.lsa_team = group->rdma_rank;
            params.tokens_per_lsa = group->config.max_dispatch_tokens_per_rank;
            params.num_real_tokens = num_combined_tokens;
            params.num_recv_tokens = num_tokens;
            params.combine_local_reduce_enabled = em_local_dup_active(group, handle->layout);

            // Pre-sum secondary em_slots into primaries before combine reads them.
            if (params.combine_local_reduce_enabled) {
                assert(
                    handle->ht.emuf_group_buf != nullptr && handle->ht.emuf_group_count != nullptr &&
                    "unfused combine requires emuf dup-group buf from dispatch scan");
                nccl_ep::ht::call_local_reduce(
                    /*expert_input_token=*/
                    params.expert_input_token_ptrs[group->lsa_rank],
                    /*expert_input_prob=*/
                    backward_combine ? params.expert_input_prob_ptrs[group->lsa_rank] : nullptr,
                    handle->ht.emuf_group_buf,
                    handle->ht.emuf_group_count,
                    handle->ht.emuf_group_stride,
                    params.hidden_dim,
                    params.experts_per_rank,
                    params.lsa_team_size,
                    backward_combine,
                    static_cast<int>(group->shuffle_sms),
                    stream,
                    x->datatype);
            }

            /* ===== Call combine kernel ===== */
            // MXFP8: kTokenDtype drives the BF16 output path; the FP8 wire format is
            // driven by params.combine_recipe, not x->datatype.
            params.token_dtype = mxfp8_combine ? ncclBfloat16 : x->datatype;
            NCCLCHECK(
                nccl_ep::ht::call_combine(
                    params,
                    group->ht_aligned_max_tokens, // chunk-aligned stride
                    group->ht_tokens_per_chunk, // tokens per dispatch/combine chunk
                    group->rdma_team_size, // num_lsa_teams (RDMA domain size)
                    backward_combine, // backward mode flag
                    static_cast<int>(group->combine_num_sms),
                    group->max_dynamic_smem,
                    &group->env,
                    stream));
        } // end pull/push combine branch

        // BWD combine: scatter dense output back to FWD k-slot ordering via ht.topk_idx.
        // Push combine writes combined_topk_weights directly (sparse-direct via srcpos),
        // so this dense->sparse step only runs on the pull path.
        if (backward_combine && !comb_push) {
            assert(
                handle->ht.topk_idx != nullptr &&
                "HT BWD combine: ht.topk_idx missing (ncclEpUpdateHandle not called?)");
            if (topk_idx_cache_is_int32(group, handle)) {
                nccl_ep::ht::dense_to_sparse_prob_combine(
                    dense_output_prob,
                    static_cast<const int32_t*>(handle->ht.topk_idx),
                    static_cast<float*>(combined_topk_weights->data),
                    num_combined_tokens,
                    num_topk,
                    group->config.num_experts,
                    stream);
            } else {
                nccl_ep::ht::dense_to_sparse_prob_combine(
                    dense_output_prob,
                    static_cast<const int64_t*>(handle->ht.topk_idx),
                    static_cast<float*>(combined_topk_weights->data),
                    num_combined_tokens,
                    num_topk,
                    group->config.num_experts,
                    stream);
            }
        }

        handle->cached_mode = true;
    }

    return ncclSuccess;
}

ncclResult_t ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) {
    if (config != nullptr) {
        EP_VALIDATE_STRUCT(config, NCCL_EP_COMPLETE_CONFIG);
    }
    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        if (handle->ll.continue_fn) {
                NCCLCHECK(handle->ll.continue_fn(LOW_LATENCY_RECV_PHASE));
            handle->ll.continue_fn = nullptr;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // HT mode - no continue needed (synchronous)
    }
    return ncclSuccess;
}

ncclResult_t ncclEpMaskQuery(ncclEpGroup_t ep_group, int* mask_status, cudaStream_t stream) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->mask_buffer != nullptr && "ncclEpMaskQuery: enable_mask must be true");
    EP_HOST_ASSERT(mask_status != nullptr);
    CUDA_CHECK(cudaMemcpyAsync(
        mask_status,
        ep_group->mask_buffer,
        ep_group->nRanks * sizeof(int),
        cudaMemcpyDeviceToDevice,
        stream));
    return ncclSuccess;
}

ncclResult_t ncclEpMaskUpdate(ncclEpGroup_t ep_group, const int* mask, cudaStream_t stream) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->mask_buffer != nullptr && "ncclEpMaskUpdate: enable_mask must be true");
    EP_HOST_ASSERT(mask != nullptr);
    CUDA_CHECK(
        cudaMemcpyAsync(ep_group->mask_buffer, mask, ep_group->nRanks * sizeof(int), cudaMemcpyHostToDevice, stream));
    return ncclSuccess;
}

ncclResult_t ncclEpMaskClean(ncclEpGroup_t ep_group, cudaStream_t stream) {
    EP_HOST_ASSERT(ep_group != nullptr);

    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->mask_buffer != nullptr && "ncclEpMaskClean: enable_mask must be true");
    EP_HOST_ASSERT(ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY);
    EP_HOST_ASSERT(
        ep_group->rdma_buffer != nullptr &&
        "ncclEpMaskClean: rdma_buffer not yet allocated; create at least one LL handle first");
    EP_HOST_ASSERT(ep_group->sync_buffer != nullptr && ep_group->sync_window != nullptr);

    // Reset internal RDMA recv-count/flag counters for both double-buffer slots
    // via a GPU kernel that includes a cross-rank barrier, then clear the mask.
    // Query the layout to read the clean_meta_offset values.
    nccl_ep::LowLatencyLayout layout(
        ep_group->config.max_dispatch_tokens_per_rank,
        ep_group->config.max_token_bytes,
        ep_group->nRanks,
        ep_group->config.num_experts,
        MAX_NUM_TOPK,
        NCCL_EP_LAYOUT_RANK_MAJOR);
    char* rdma_base = static_cast<char*>(ep_group->rdma_buffer);
    const auto clean_0 = layout.buffer.clean_meta_offset();
    const auto clean_1 = std::pair{clean_0.first + layout.signal_slot_stride, clean_0.second};
    int* clean_0_ptr = reinterpret_cast<int*>(rdma_base + clean_0.first);
    int* clean_1_ptr = reinterpret_cast<int*>(rdma_base + clean_1.first);

    nccl_ep::ll::CleanLowLatencyBufferParams clean_params{};
    clean_params.clean_0 = clean_0_ptr;
    clean_params.num_clean_int_0 = clean_0.second;
    clean_params.clean_1 = clean_1_ptr;
    clean_params.num_clean_int_1 = clean_1.second;
    clean_params.rankMask = ep_group->mask_buffer;
    clean_params.syncBuffer = static_cast<int*>(ep_group->sync_buffer);
    clean_params.syncWindow = ep_group->sync_window;
    clean_params.devComms = ep_group->nccl_dev_comms;
    clean_params.barrierSignalBase = ep_group->clean_barrier_signal_base;
    clean_params.timeoutCycles = ep_group->timeout_cycles;

    nccl_ep::ll::call_clean_low_latency_buffer(clean_params, stream);

    // Reset all ranks to active (1 = active).
    // Sync the stream before returning so all_active outlives the async copy.
    std::vector<int> all_active(ep_group->nRanks, 1);
    CUDA_CHECK(cudaMemcpyAsync(
        ep_group->mask_buffer,
        all_active.data(),
        ep_group->nRanks * sizeof(int),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return ncclSuccess;
}

ncclResult_t ncclEpGetAsyncError(ncclEpGroup_t ep_group, int* error_out) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->async_error_flag != nullptr && "ncclEpGetAsyncError: enable_mask must be true");
    EP_HOST_ASSERT(error_out != nullptr);
    *error_out = __atomic_load_n(ep_group->async_error_flag, __ATOMIC_ACQUIRE);
    return ncclSuccess;
}

ncclResult_t ncclEpErrorClear(ncclEpGroup_t ep_group) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->async_error_flag != nullptr && "ncclEpErrorClear: enable_mask must be true");
    __atomic_store_n(ep_group->async_error_flag, 0, __ATOMIC_RELEASE);
    return ncclSuccess;
}
// ── Test-only internal helpers (declared in nccl_ep_test_internal.h) ─────────
// Exposed via a separate test header so library consumers cannot accidentally
// depend on these implementation details.

const int32_t* ncclEpHandle_test_getSparseToDenseMap(ncclEpHandle_t handle) {
    return handle->ht.sparse_to_dense_map;
}

int ncclEpHandle_test_getNumTopk(ncclEpHandle_t handle) {
    return handle->num_topk;
}

int ncclEpHandle_test_getMaxTokensPerRank(ncclEpHandle_t handle) {
    return static_cast<int>(handle->group->config.max_dispatch_tokens_per_rank);
}

int ncclEpHandle_test_getNRanksPerNode(ncclEpHandle_t handle) {
    return handle->group->lsa_team_size;
}

int ncclEpHandle_test_getExpertsPerRank(ncclEpHandle_t handle) {
    return handle->group->num_local_experts;
}


ncclResult_t ncclEpHandle_test_getNumRecvTokens(ncclEpHandle_t handle, unsigned int* num_recv_tokens) {
    return ht_query_num_recv_tokens(handle, /*stream=*/nullptr, num_recv_tokens);
}

void ncclEpHandle_test_clearTopkIdx(ncclEpHandle_t handle) {
    handle->topk_idx.data = nullptr;
}

ncclResult_t ncclEpGroup_test_setMaxDynamicSmem(ncclEpGroup_t group, int max_dynamic_smem) {
    if (group == nullptr || max_dynamic_smem <= 0) return ncclInvalidArgument;
    group->max_dynamic_smem = max_dynamic_smem;
    group->last_ll_combine_warps_per_group = 0;
    return ncclSuccess;
}

int ncclEpGroup_test_getLastLlCombineWarpsPerGroup(ncclEpGroup_t group) {
    return group == nullptr ? 0 : group->last_ll_combine_warps_per_group;
}

void ncclEpGroup_test_getSmBudgets(
    ncclEpGroup_t group, unsigned int* dispatch_num_sms, unsigned int* combine_num_sms) {
    if (group == nullptr) return;
    if (dispatch_num_sms != nullptr) *dispatch_num_sms = group->dispatch_num_sms;
    if (combine_num_sms != nullptr) *combine_num_sms = group->combine_num_sms;
}
