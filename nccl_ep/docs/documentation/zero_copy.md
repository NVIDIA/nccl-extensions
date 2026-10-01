# Zero-Copy

By default NCCL EP moves payloads through library-owned staging buffers: dispatch
writes tokens into an internal buffer that peers read, and combine does the
mirror. Zero-copy may remove that hop for certain algorithms as peers can read and
write the caller's own input/output buffers directly.

The enabler is an **NCCL window**. A tensor whose memory is registered as a window
is addressable by remote peers, so the library can point them at it instead of at
its own staging.

## Preparing a tensor for zero-copy

Register the buffer with `ncclCommWindowRegister`, then attach the resulting
handle to the `ncclEpTensor_t` descriptor via `win_hdl` / `win_offset` instead of
setting `data`:

```c
size_t dims[2] = { num_recv_slots, hidden };
void*  buf     = nullptr;
ncclMemAlloc(&buf, num_recv_slots * hidden * sizeof(nv_bfloat16));

ncclWindow_t win;
ncclCommWindowRegister(comm, buf, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);

ncclEpTensor_t recv = NCCL_EP_TENSOR_INIT;
recv.ndim = 2;
recv.datatype = ncclBfloat16;
recv.sizes = dims;
recv.win_hdl = win;        // instead of recv.data
recv.win_offset = 0;       // byte offset into the window
```

Notes:

- Registration is **collective** — every rank in the communicator must register
  its corresponding buffer.
- `win_offset` lets several tensors share one registration; give each its byte
  offset within the window.
- Only *user-registered* windows count. The library's own internal window does not
  put a tensor on the zero-copy path.
- A descriptor with no window is an ordinary tensor and simply stages.

Rows and the window offset must satisfy the same 16-byte alignment rules as any
other EP tensor; see [Quantization](quantization.md) for the quantized cases.

## What is supported today

Direct access is used **wherever the path supports it, in any mode** — attaching
a window is enough. Support differs by algorithm.

### High Throughput

Both directions are supported:

| Call             | Tensor            |
|------------------|-------------------|
| `ncclEpDispatch` | `outputs->tokens` |
| `ncclEpCombine`  | `inputs->tokens`  |

Under `NCCL_EP_DISP_QUANT_FWD` the scale tensors participate in the same pairing.
Dispatch **inputs** may also be windowed; that is an independent per-tensor
choice, not something the group flag governs.

Expert-major dispatch on the permute path — what you get under `AUTO`/`OFF` —
stages into the recv buffer even when the tensor is windowed, because the permute
kernel, not the peers, writes the caller's tensor. Setting `ON` selects a
different expert-major mode that does write peer buffers directly; see the
algorithm switch below.

### Low Latency

Zero-copy is **dispatch-only**. LL combine reads its input directly and uses
windows only to translate peer receive-buffer pointers, so it always stages.

LL dispatch takes the direct path only when all of these hold:

- **NVLink-only topology** — `lsa_team_size == nRanks`, i.e. no RDMA leg;
- **`NCCL_EP_LAYOUT_RANK_MAJOR`**;
- recipe is `NCCL_EP_DISP_QUANT_NONE` or `NCCL_EP_DISP_QUANT_FWD`.

Then `outputs->tokens` is written directly when windowed, and under `QUANT_FWD`
`outputs->scales` independently as well — either, both, or neither.

## The group-wide `zero_copy` flag

`ncclEpGroupConfig_t::zero_copy` allows users to inform the library that the
tensors on the direct paths — `ncclEpDispatch` outputs and `ncclEpCombine`
inputs — will have a NCCL window attached. Having zero-copy guaranteed allows
NCCL EP to optimize memory consumption by allocating only the required staging
buffer space.

| Value                    | Behavior                                                                                               |
|--------------------------|--------------------------------------------------------------------------------------------------------|
| `NCCL_EP_ZERO_COPY_AUTO` | Opportunistic: windows are used where supported, missing windows stage.                                |
| `NCCL_EP_ZERO_COPY_OFF`  | Identical to `AUTO`.                                                                                   |
| `NCCL_EP_ZERO_COPY_ON`   | Windows become required. A missing window is an error, not a fallback. Token staging is not allocated. |

Under `ON`:

- **HT** rejects a plain `ncclEpDispatch` `outputs->tokens` or `ncclEpCombine`
  `inputs->tokens` with `ncclInvalidArgument`.
- **LL** requires at least one eligible payload window on dispatch and fails with
  `ncclInvalidArgument` naming the unmet condition otherwise.

### `ON` changes the expert-major algorithm

A token routed to *k* of this rank's local experts needs *k* copies in the
expert-major output. **Who makes those copies, and when**, is the expert-major
recipe, and it is auto-selected from `zero_copy` and the topology:

| `zero_copy` | LSA teams | Expert-major recipe                                                                         |
|-------------|-----------|---------------------------------------------------------------------------------------------|
| not `ON`    | any       | Tokens are staged deduplicated; a permute kernel then expands them into the caller's buffer |
| `ON`        | > 1       | The sender writes one copy per destination expert directly over NVLink                      |
| `ON`        | 1         | The receiver fans each token out to its local experts                                       |

So `ON` is a performance decision as well as a memory one: it selects a different
implementation with different staging and different behavior.

**`ON` and the permute recipe are mutually exclusive.** Setting `zero_copy = ON`
always selects one of the two duplicating recipes, and there is no way to require
windows while keeping the permute recipe. Expert-major uses permute only when
`zero_copy` is not `ON` *and* neither of the env overrides below is set.

The overrides work in one direction only: `NCCL_EP_HT_EM_NVLINK_DUP` and
`NCCL_EP_HT_EM_LOCAL_DUP` force the sender-side and receiver-side recipes
respectively, even under `AUTO`/`OFF` — but nothing forces the permute recipe
back on. So choosing `ON` for its window enforcement or its memory saving also
commits you to a different expert-major implementation; the two decisions cannot
be made separately.

> **Note:** the mapping from `zero_copy` to expert-major recipe, and the fact
> that the two are coupled, may change in future releases. Do not depend on a
> particular recipe being selected.

### Memory

`ON` elides both token staging regions in the intra-LSA buffer — the dispatch and
combine token regions are simply not allocated. There is no dedicated scale
region: `QUANT_FWD` scales are carved from each token slot's tail slack, so when
the token regions are elided the caller must window `outputs->scales` too. The
per-expert probability regions are still allocated.

## Debugging

`NCCL_EP_DEBUG=1` reports, per dispatch, whether zero-copy was selected and — when
it was not — which condition failed: the recipe, the requested mode, whether the
topology is NVLink-only, the layout, and which tensors carried windows. That is
the fastest way to find out why a group is still staging.
