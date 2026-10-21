# API Reference

Complete reference for the NCCL EP public C API. Every entry point exported by
`nccl_ep.h` is listed here.

Behavioral topics are covered in dedicated guides rather than repeated per
function: [Quantization](quantization.md), [Zero-Copy](zero_copy.md),
[Recv Overflow Policy](overflow_policy.md), and [HT Eager Mode](eager_mode.md).

## Contents

| Group | Functions |
|---|---|
| [Library](#library) | `ncclEpGetVersion` |
| [Group Management](#group-management) | `ncclEpCreateGroup`, `ncclEpGroupDestroy` |
| [Tensor Descriptors](#tensor-descriptors) | `ncclEpTensorAlloc`, `ncclEpTensorDestroy` |
| [Handle Management](#handle-management) | `ncclEpCreateHandle`, `ncclEpInitHandle`, `ncclEpUpdateHandle`, `ncclEpHandleMemSize`, `ncclEpImportHandle`, `ncclEpHandleDestroy` |
| [Communication Operations](#communication-operations) | `ncclEpDispatch`, `ncclEpCombine`, `ncclEpAfDispatch`, `ncclEpAfRecv`, `ncclEpAfCombine`, `ncclEpAfReturn`, `ncclEpComplete` |
| [Fault Tolerance](#fault-tolerance) | `ncclEpMaskQuery`, `ncclEpMaskUpdate`, `ncclEpMaskClean`, `ncclEpGetAsyncError`, `ncclEpErrorClear` |

## Library

### `ncclEpGetVersion()`

```c
// Return the NCCL_EP_VERSION_CODE of the NCCL EP library in the supplied integer.
// The value encodes MAJOR, MINOR and PATCH as MAJOR*10000 + MINOR*100 + PATCH.
//
// Arguments:
//   version - [OUT] Pointer to receive the library's NCCL_EP_VERSION_CODE value
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetVersion(int* version);
```

Compare against `NCCL_EP_API_VERSION` before relying on a struct field
introduced in a later release; see the ABI notes in the main
[README](../../README.md#core-concepts).

## Group Management

### `ncclEpCreateGroup()`

```c
// Create an EP group from an NCCL communicator
//   This call is collective and must be invoked by all ranks in the group.
//   Any custom device-memory allocator is supplied via config->alloc
//   (see ncclEpAllocConfig_t); zero-init falls back to cudaMalloc/cudaFree.
//
// Arguments:
//   ep_group   - [OUT] Pointer to newly created EP group
//   comm       - [IN]  Existing NCCL communicator
//   config     - [IN]  Pointer to EP configuration structure
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config
);
```

### `ncclEpGroupDestroy()`

```c
// Destroy an EP group and release associated resources.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group
);
```

## Handle Management

An NCCL EP handle represents the state that connects related dispatch and combine
operations — a dispatch/combine pair on the LL path, or both the forward and
backward passes in HT mode. The handle holds only the metadata-related state; the
staging buffers are owned by the EP group.

### Multiple handles per group

Both HT and LL support several handles being active on the same EP group, as a
1F1B / pipeline-parallel schedule requires. Creating or updating one handle does
not disturb another handle's pending dispatch or combine.

### `ncclEpCreateHandle()`

```c
// Create and initialize an EP handle.
// Performs dispatch setup and (in HT mode only) metadata exchange.
// This call is collective and must be invoked by all ranks in the group.
// The routing carried by `topk_idx` is cached on the handle; subsequent
// ncclEpDispatch / ncclEpCombine calls reuse it until ncclEpUpdateHandle
// replaces it with new routing.
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   layout              - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//                                HT supports FLAT / EXPERT_MAJOR; LL supports EXPERT_MAJOR / RANK_MAJOR.
//   topk_idx            - [IN]  Pointer to a caller-owned tensor descriptor holding
//                               top-K expert indices (2D [num_tokens, top_k]; int32 or int64).
//   layout_info         - [IN/OUT, optional] Named-struct pointer carrying device-side
//                         metadata tensor pointers. See ncclEpLayoutInfo_t for the fields
//                         populated at handle time and the layouts each applies to.
//                         NULL = no metadata.
//   config              - [IN]  Optional handle configuration (e.g. expert-major
//                               alignment via dispatch_output_per_expert_alignment);
//                               NULL = defaults.
//   stream              - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream
);
```

### `ncclEpHandleDestroy()`

```c
// Destroy an EP handle and release all associated resources.
//
// Arguments:
//   handle         - [IN]  EP handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
);
```

### `ncclEpInitHandle()`

`ncclEpCreateHandle` is the fused form of `ncclEpInitHandle` +
`ncclEpUpdateHandle`. Use the split form when handle allocation and per-step
routing happen at different points — for example to allocate all handles before
a CUDA graph capture begins.

```c
// Allocate handle buffers without performing any collective.
// Call ncclEpUpdateHandle before the first ncclEpDispatch/ncclEpCombine.
//
// handle_mem == NULL:  NCCL EP allocates via alloc_fn; handle owns the memory.
// handle_mem != NULL:  wraps a caller-owned 1D ncclUint8 tensor
//                      (>= ncclEpHandleMemSize); handle owns no memory and
//                      ncclEpHandleDestroy frees only the struct.
//
// Arguments:
//   handle     - [OUT] Newly created handle
//   ep_group   - [IN]  A valid EP group
//   layout     - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//   config     - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   num_topk   - [IN]  Required for LL (> 0); pass -1 for HT
//   handle_mem - [IN]  NULL = internal alloc; non-NULL = caller-owned device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    int num_topk,
    const ncclEpTensor_t* handle_mem);
```

**Collective under `rdma_buffer_size = NCCL_EP_AUTO`.** With auto-sized RDMA
buffers this call *may* be collective: if the requested `(layout, num_topk)`
needs more memory than the current allocation, it reallocates. All ranks must
then call it in lockstep with the same `(layout, num_topk)`, with no other
communication in flight, and it must not appear inside a CUDA graph capture.
An explicit `rdma_buffer_size` makes it purely local. See the
[`rdma_buffer_size` notes](../../README.md#algorithm-related-configurations).

### `ncclEpUpdateHandle()`

```c
// Per-step collective: prepare the handle for the given top-k routing decisions.
// Must be called after ncclEpInitHandle and before ncclEpDispatch. The routing is
// cached on the handle and reused by every dispatch until the next update.
//
// Arguments:
//   handle      - [IN]  Handle from ncclEpInitHandle
//   topk_idx    - [IN]  [num_tokens, top_k]; ncclInt32 or ncclInt64. The caller
//                       ensures expert ids fit in the chosen width.
//   layout_info - [IN/OUT, optional] Named local tensors (NULL = none provided).
//                       See ncclEpLayoutInfo_t for the fields populated at handle
//                       time and the layouts each applies to. LL mode: must be NULL.
//   stream      - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    cudaStream_t stream);
```

In HT this is where the metadata scan runs, so it is also where a recv overflow
is detected and where `recv_total_counter` is written. See
[Recv Overflow Policy](overflow_policy.md).

### `ncclEpHandleMemSize()`

```c
// Query the device bytes required for a handle's persistent routing state, for
// callers that want to supply handle memory themselves via the handle_mem argument
// of ncclEpInitHandle / ncclEpCreateHandle. Group-owned update-only scan workspace
// is allocated lazily by ncclEpInitHandle and is not included in size_out.
//
// Arguments:
//   ep_group  - [IN]  A valid EP group
//   layout    - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//   config    - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   size_out  - [OUT] Required bytes for handle_mem
//   num_topk  - [IN]  Required for LL (> 0); optional for HT
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    size_t* size_out,
    int num_topk);
```

### `ncclEpImportHandle()` (HT only)

Binds a handle to a `handle_mem` buffer already prepared by `ncclEpUpdateHandle`
on a *different* handle, without rerunning the collective. Intended for callers
that snapshot a prepared `handle_mem` and later rebind it at a different address
— `ncclEpHandleState_t` carries the host metadata needed to reinterpret that
snapshot.

```c
// Import an HT handle from a caller-owned handle buffer already prepared by
// ncclEpUpdateHandle with matching group, layout, config, and state metadata.
// Local only: no device-state copy, CUDA launch, or metadata scan. Dispatch
// and Combine can use the returned handle without another UpdateHandle.
//
// Arguments:
//   handle     - [OUT] Newly imported handle
//   ep_group   - [IN]  The EP group used to prepare handle_mem
//   layout     - [IN]  The layout used to prepare handle_mem
//   config     - [IN]  Handle configuration used to prepare handle_mem. NULL = defaults.
//   state      - [IN]  Host metadata describing the prepared state
//   handle_mem - [IN]  Caller-owned prepared 1D ncclUint8 device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpImportHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    const ncclEpHandleState_t* state,
    const ncclEpTensor_t* handle_mem);
```

`ncclEpHandleState_t` (`num_tokens`, `num_topk`, `topk_idx_datatype`) is
caller-constructed from the values used to prepare the snapshot; initialize it
with `NCCL_EP_HANDLE_STATE_INIT`.

## Tensor Descriptors

`ncclEpTensor_t` is a plain value struct.  The common path is to allocate on the
stack (or as a struct member), zero-initialise it with `NCCL_EP_TENSOR_INIT`, and
fill fields directly:

```c
int N = 128;
int H = 2048;
size_t dims[2] = { N, H };
ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
t.ndim = 2; t.datatype = ncclFloat16; t.data = data_ptr;
t.sizes = dims;   // caller owns; must outlive the descriptor's use
```

For window-backed tensors set `win_hdl` / `win_offset` instead of `data`.

When a heap-allocated descriptor with a library-owned `sizes` copy is more
convenient, use `ncclEpTensorAlloc` to obtain one and `ncclEpTensorDestroy`
to release it (the backing data buffer remains caller-owned).

All user-facing tensor fields (`data`, `ndim`, `datatype`, `sizes`, `win_hdl`,
`win_offset`) are directly accessible as struct members.  Public structs
(`ncclEpDispatchInputs_t`, `ncclEpLayoutInfo_t`, …) hold `ncclEpTensor_t*`
pointers, so callers can mix stack-, static-, and heap-allocated descriptors.

### `ncclEpTensorAlloc()`

```c
// Allocate a tensor descriptor sufficient to represent the requested shape.
// The library copies `sizes` into its own storage, so the caller's array need
// not outlive the call. The backing data buffer remains caller-owned.
//
// Arguments:
//   tensor   - [OUT] On success, receives a pointer to the new descriptor.
//   ndim     - [IN]  Number of dimensions (> 0).
//   datatype - [IN]  Element type.
//   sizes    - [IN]  Array of `ndim` dimension sizes.
//   config   - [IN]  Optional allocation configuration. NULL = defaults.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorAlloc(
    ncclEpTensor_t** tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    const size_t* sizes,
    const ncclEpTensorAllocConfig_t* config);
```

### `ncclEpTensorDestroy()`

```c
// Release a descriptor previously returned by ncclEpTensorAlloc.
// Does not free the descriptor's data buffer, which is caller-owned.
//
// Arguments:
//   tensor - [IN] Pointer returned by ncclEpTensorAlloc. NULL is accepted.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorDestroy(ncclEpTensor_t* tensor);
```

## Communication Operations

### `ncclEpDispatch()`

Perform EP dispatch: send tokens to experts according to routing decisions.

```c
// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Routing (topk_idx) is taken from the handle — supply it once via
//     ncclEpCreateHandle / ncclEpUpdateHandle.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t / ncclEpLayoutInfo_t),
//     each field a `ncclEpTensor_t*` to a caller-owned descriptor.
//
// Arguments:
//   handle        - [IN,OUT] EP handle
//   inputs        - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                            other fields (topk_weights, scales) are optional and depend on
//                            algorithm/layout.
//   outputs       - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                            other fields (topk_weights, topk_idx, scales) are optional and
//                            depend on algorithm/layout. For HT, outputs are 2D [N(r), data_size].
//                            For LL expert-major, outputs are 3D [num_local_experts, N(r), data_size].
//                            For LL rank-major,   outputs->tokens is 3D
//                            [num_ranks, max_dispatch_tokens_per_rank, data_size].
//   layout_info   - [IN,OUT, optional] Named-struct pointer for device-side metadata tensors.
//                            LL mode: optional `expert_counters` (1D ncclInt32 / ncclInt64,
//                            size = num_local_experts) receives per-expert recv counts.
//                            NULL = no metadata.
//   config        - [IN]     Dispatch configuration.
//   stream        - [IN]     CUDA stream. If ncclEpDispatch is called on a different stream than
//                            the stream used in `ncclEpCreateHandle`, it is the responsibility
//                            of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream
);
```

### `ncclEpCombine()`

Perform EP combine: gather expert outputs and return in original token order.

```c
// Perform EP combine
//   * Gathers outputs from experts and returns them to their source in original token order.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpCombineInputs_t / ncclEpCombineOutputs_t).
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation.
//   inputs           - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                               LL expert-major: 3D [num_local_experts, N(r), data_size].
//                               LL rank-major:   3D [num_ranks, max_dispatch_tokens_per_rank, data_size].
//                               HT:              2D [N(r), data_size].
//                               Backward combine: also set `inputs->topk_weights`
//                               [N(r), top_k] (HT only).
//   outputs          - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                               2D [num_tokens, data_size] in original token order.
//                               LL expert-major: `outputs->topk_weights` [num_tokens, top_k]
//                               is used by the receive-side reduction.
//                               HT backward combine: `outputs->topk_weights` [num_tokens, top_k]
//                               receives per-token routing weights.
//   config           - [IN]     Combine configuration (e.g. `send_only` for LL staged mode).
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than
//                               the stream used in `ncclEpCreateHandle()`, it is the responsibility
//                               of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream
);
```

### AFD role wrappers

Attention-FFN Disaggregation (AFD) partitions an LL rank-major group into a
leading prefix of attention ranks and a trailing set of expert ranks, selected
by `ncclEpGroupConfig_t::num_attn_ranks`. The four role-named APIs validate the
calling rank and then forward to the ordinary dispatch or combine operation.
They do not create separate collectives: every rank must call its matching side
of each row below in the same order.

| Collective | Attention ranks | Expert ranks |
|---|---|---|
| Dispatch | `ncclEpAfDispatch` | `ncclEpAfRecv` |
| Combine | `ncclEpAfCombine` | `ncclEpAfReturn` |

```c
ncclResult_t ncclEpAfDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream);

ncclResult_t ncclEpAfRecv(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream);

ncclResult_t ncclEpAfCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream);

ncclResult_t ncclEpAfReturn(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream);
```

On attention ranks, dispatch receive tensors and combine `inputs->tokens` are
omitted. On expert ranks, dispatch `inputs->topk_weights` is omitted, while the
rank-major dispatch receive tensors and combine input tensor are required. A
wrong-role call returns `ncclInvalidUsage`; an invalid handle returns
`ncclInvalidArgument`.

AFD follows the ordinary LL `AUTO` zero-copy contract: window use must be
consistent across the collective. When dispatch output tokens are window-backed,
every rank participates in the same `ncclCommWindowRegister` call and attaches
the resulting handle to `outputs->tokens`. F-ranks register the full receive
allocation. A-ranks may register a minimal placeholder allocation because NCCL
supports asymmetric window sizes; their local allocation is not written. If no
rank attaches a window, dispatch uses library staging as usual. Mixing windowed
and non-windowed descriptors across ranks is unsupported, as in classic LL EP.

Python exposes the same role checks as `Handle.af_dispatch`, `Handle.af_recv`,
`Handle.af_combine`, and `Handle.af_return`.

AFD currently supports unquantized communication only. Both sides of dispatch
must use `NCCL_EP_DISP_QUANT_NONE`, and both sides of combine must use
`NCCL_EP_COMB_QUANT_NONE`. Other quantization recipes are not supported with
`num_attn_ranks > 0`.

### `ncclEpComplete()` (LL mode only)

Must be the first NCCL EP operation to be executed after dispatch or combine,
no other op is allowed on handle or group in between.

```c
// Continues a staged EP operation to completion.
//   * This should be called after a prior `ncclEpDispatch()` or `ncclEpCombine()` call with `send_only` flag set.
//
// Arguments:
//   handle     - [IN,OUT] EP handle used in the preceding staged operation
//   config     - [IN]     Completion configuration (see ncclEpCompleteConfig_t). NULL = defaults.
//   stream     - [IN]     CUDA stream
//
// Notes:
//   - If `ncclEpComplete()` is called on a different stream than the operation initiation call
//     (i.e., `ncclEpDispatch()` or `ncclEpCombine()`), it is the responsibility of the user to
//     synchronize between streams to ensure correctness.
//   - Only LL mode is supported.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpComplete(
    ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
);
```

## Fault Tolerance

These APIs require `ncclEpGroupConfig_t::enable_mask = 1` and are **LL mode
only**. With masking enabled, a remote rank that times out during dispatch or
combine is masked (skipped) rather than trapping the GPU, and a host-visible
error flag is raised.

The timeout is `ncclEpGroupConfig_t::timeout_ns` (0 = default, ~100 s), which
`NCCL_EP_TIMEOUT_MS` overrides. Setting it too low risks masking merely slow
ranks.

A typical recovery loop polls `ncclEpGetAsyncError`, inspects which ranks were
masked with `ncclEpMaskQuery`, continues in degraded mode or re-admits ranks with
`ncclEpMaskClean`, and then re-arms detection with `ncclEpErrorClear`.

### `ncclEpMaskQuery()`

```c
// Query the active-mask status of all ranks.
//   Copies the mask buffer to a user-provided device tensor.
//
// Arguments:
//   ep_group     - [IN]  EP group with masking enabled
//   mask_status  - [OUT] Device pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream       - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskQuery(ncclEpGroup_t ep_group, int* mask_status, cudaStream_t stream);
```

### `ncclEpMaskUpdate()`

```c
// Set the mask for all ranks at once.
//
// Arguments:
//   ep_group   - [IN] EP group with masking enabled
//   mask       - [IN] Host pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream     - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskUpdate(ncclEpGroup_t ep_group, const int* mask, cudaStream_t stream);
```

### `ncclEpMaskClean()`

```c
// Reset masks and RDMA buffers so previously masked ranks can re-join.
//   Collective: all surviving ranks must call simultaneously.
//   Resets RDMA buffers via a cross-rank barrier and sets all masks to active.
//   Does NOT reset the async error flag - call ncclEpErrorClear() separately.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//   stream   - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskClean(ncclEpGroup_t ep_group, cudaStream_t stream);
```

This re-admits a *delayed* rank within the same communicator. Replacing a rank
requires a new communicator (e.g. `ncclCommGrow`) and a new EP group.

### `ncclEpGetAsyncError()`

```c
// Poll for asynchronous errors (e.g., rank timeout).
//   Lightweight host-side check - reads a pinned CPU flag, no GPU sync required.
//   The flag is set by the kernel when a timeout masks a rank; clear it
//   explicitly via ncclEpErrorClear().
//
// Arguments:
//   ep_group  - [IN]  EP group with masking enabled
//   error_out - [OUT] 0 = no error, 1 = timeout occurred (one or more ranks masked)
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetAsyncError(ncclEpGroup_t ep_group, int* error_out);
```

### `ncclEpErrorClear()`

```c
// Clear the async error flag.
//   Lightweight host-side reset - writes zero to the pinned CPU flag.
//   Use after detecting an error (via ncclEpGetAsyncError) to re-arm the flag
//   for detecting new failures. Call after ncclEpMaskClean (full recovery) or
//   standalone when surviving ranks continue in degraded mode.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpErrorClear(ncclEpGroup_t ep_group);
```
