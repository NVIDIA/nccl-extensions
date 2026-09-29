# NCCL EP (Expert Parallelism) API

NCCL EP is a high-performance NCCL API extension for efficient Mixture-of-Experts (MoE) communication.
It provides optimized dispatch and combine primitives for Expert Parallelism (EP) across distributed GPU systems
implemented on top of NCCL Device API: Load-Store Accessible (LSA) and GPU-Initiated Networking (GIN) operations.

# Maintainers

| GitHub         | Areas                       |
|----------------|-----------------------------|
| @artpol84      | APIs, new features, layouts |
| @kwen2501      | APIs, integration           |
| @sb17v         | Kernels, build systems      |
| @nv-lschneider | Kernels, mnnvl              |
| @kgioioso      | GIN, NCCL                   |

# Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
  - [Running](#running)
- [Core Concepts](#core-concepts)
  - [Key Data Structures](#key-data-structures)
  - [Algorithm-related configurations](#algorithm-related-configurations)
  - [HT-specific modifiers](#ht-specific-modifiers)
  - [LL-specific modifiers](#ll-specific-modifiers)
  - [Zero-copy](#zero-copy)
  - [Custom Allocators](#custom-allocators)
  - [Elastic (GPU + CPU) receive buffers](#elastic-gpu--cpu-receive-buffers)
  - [API Reference](#api-reference)
- [Execution Modes](#execution-modes)
- [Usage Examples](#usage-examples)

**Guides**

- [API Reference](docs/documentation/api_reference.md) — every public entry point
- [JIT Kernel Compilation](docs/documentation/jit.md) — runtime `nvcc` knobs, paths, and cache
- [Quantization](docs/documentation/quantization.md) — dispatch and combine recipes
- [Zero-Copy](docs/documentation/zero_copy.md) — direct peer access to caller buffers
- [Recv Overflow Policy](docs/documentation/overflow_policy.md) — HT trap / drop behavior
- [HT Eager Mode](docs/documentation/eager_mode.md) — per-routing recv sizing
- [Execution Modes](docs/documentation/execution_modes.md) — synchronous and staged semantics
- [Usage Examples](docs/documentation/usage_examples.md) — end-to-end HT and LL walkthroughs
- [Custom Allocators](docs/documentation/custom_allocators.md) — routing internal allocations through a caller pool
- [Elastic Receive Buffers](docs/documentation/elastic_buffers.md) — GPU + CPU backed recv tensors

# Overview

The NCCL EP API extends NCCL with native support for efficient MoE communication patterns. It provides optimized implementations for token "dispatch" and expert output "combine" operations, which are an important components of modern large language models employing sparse MoE architectures.

NCCL EP brings the performance benefits of modern device-initiated MoE libraries into the NCCL ecosystem with a unified API. NCCL EP provides unified `ncclEpDispatch` and `ncclEpCombine` primitives that allow selecting the appropriate algorithm based on workload characteristics.

Currently, two distinct communication algorithms tailored for different workload characteristics are supported:

* **Low-Latency (LL)** - optimized for small batch sizes and latency-sensitive workloads (i.e., LLM inference). To minimize latency, it uses direct point-to-point all-to-all communication with experts.

* **High-Throughput (HT)** - optimized for training and inference prefilling with large batch sizes. HT mode implements hierarchical communication patterns, relying on NVLink for intra-node aggregation, and on RDMA for inter-node communication. The implementation leverages Hopper architecture features, including warp-specialized pipelines, and TMA (Tensor Memory Accelerator) operations.

NCCL EP relies on NCCL Device API, using GIN `put`/`signal` operations for RDMA and LSA load/store operations for NVLink communication, eliminating CPU involvement in the critical path while inheriting NCCL topology detection and plugin architecture.

## Key Features

- **Staged Execution** (LL mode only): Enable computation-communication overlap through a `send-only` flag.
- **Automatic Tuning**: Let the API auto-tune buffer sizes, queue pairs, and channels.

## Quick Start

### C API

```c
// Group management. Custom allocator (if any) is set via config.alloc
// (ncclEpAllocConfig_t); zero-init uses cudaMalloc/cudaFree.
ncclEpCreateGroup(&ep_group, comm, &config);
ncclEpGroupDestroy(ep_group);

// Handle management. Both calls take the routing as `const ncclEpTensor_t*`.
// Here `topk_idx` is a caller-owned descriptor passed by address; a heap
// descriptor from ncclEpTensorAlloc would be passed directly instead. The
// routing it carries is cached in the handle and reused by all dispatches
// until ncclEpUpdateHandle is called with new routing.
// `layout_info` is an optional ncclEpLayoutInfo_t* whose fields advertise
// device-side metadata tensors (expert_counters, src_rank_counters,
// expert_offsets, recv_total_counter).
ncclEpCreateHandle(&handle, ep_group, layout, &topk_idx, layout_info, handle_cfg, stream);
ncclEpUpdateHandle(handle, &new_topk_idx, layout_info, stream);  // optional: refresh routing
ncclEpHandleDestroy(handle);

// Communication operations. inputs / outputs are named-struct pointers
// (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t /
// ncclEpCombineInputs_t / ncclEpCombineOutputs_t); each cross-boundary
// tensor lives in a named field as a `ncclEpTensor_t*`.
ncclEpDispatch(handle, &dispatch_in, &dispatch_out, layout_info, &dispatch_cfg, stream);
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);
ncclEpComplete(handle, config, stream);  // LL mode only
```

### Python API

Install nccl4py, which includes the NCCL EP Python bindings as `nccl.ep`. Only CUDA 13 is supported as of now.

```bash
$ pip install nccl4py[cu13]
```

Import and use NCCL EP in a python application
```python
from nccl.ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

nccl_lib = NCCLLibrary()
# Use nccl_lib.ncclEpDispatch, ncclEpCombine, etc.
```

### Benchmarking

For microbenchmarking, NCCL EP provides the performance evaluation tool [`ep_bench`](ep_bench.cu).
Run `ep_bench --help` for the full option list. For debugging, `--disable-token-dropping`
makes Low-Latency runs route every token (no random `-1` sentinels in the topk table),
giving deterministic, drop-free routing.

### Common scenarios

This section provides a high-level overview of the input, output, and layout
metadata tensors expected by the API for common scenarios. Each cross-boundary
tensor lives in a named field of one of the API's struct types
(`ncclEpDispatchInputs_t`, `ncclEpDispatchOutputs_t`, `ncclEpLayoutInfo_t`,
`ncclEpCombineInputs_t`, `ncclEpCombineOutputs_t`); the **Struct** column
names the struct and the **Field** column names the field within it.

#### Used notation

**Dimensions:**
* B = batch size
* H = hidden dimension
* L = number of local experts
* K = top K
* R = number of ranks (nRanks)
* N(r) = number of tokens targeting rank r

For quantization recipes, tensor contracts, and `max_token_bytes` sizing, see
[Quantization](docs/documentation/quantization.md).


#### LL mode, expert-major layout (same data type)

| Operation | Struct           | Field           | Dims              |
|:---------:|:-----------------|:----------------|:-----------------:|
| Dispatch  | dispatch_inputs  | tokens          | [B x H]           |
|           | dispatch_outputs | tokens          | [L x (R x B) x H] |
|           | layout_info      | expert_counters | [L]               |
| Combine   | combine_inputs   | tokens          | [L x (R x B) x H] |
|           | combine_outputs  | tokens          | [B x H]           |
|           | combine_outputs  | topk_weights    | [B x K]           |

The combine kernel applies the per-token routing weights on the receive side, so
`combine_outputs.topk_weights` is required.

#### LL mode, rank-major layout (same data type)

Tokens arrive grouped by source rank with no expert dimension, so dispatch also
returns per-slot routing metadata and the caller pre-reduces across local experts
before combine.

| Operation | Struct           | Field             | Dims        |
|:---------:|:-----------------|:------------------|:-----------:|
| Dispatch  | dispatch_inputs  | tokens            | [B x H]     |
|           | dispatch_inputs  | topk_weights      | [B x K]     |
|           | dispatch_outputs | tokens            | [R x B x H] |
|           | dispatch_outputs | topk_weights      | [R x B x K] |
|           | dispatch_outputs | topk_idx          | [R x B x K] |
|           | layout_info      | src_rank_counters | [R]         |
| Combine   | combine_inputs   | tokens            | [R x B x H] |
|           | combine_outputs  | tokens            | [B x H]     |

Note `src_rank_counters`, not `expert_counters`. The caller applies the weights
during its own pre-reduction, so `combine_outputs.topk_weights` is left unset.


#### HT mode (same data type)

HT mode supports `NCCL_EP_LAYOUT_FLAT` and `NCCL_EP_LAYOUT_EXPERT_MAJOR`.
With `NCCL_EP_LAYOUT_FLAT`, dispatch output is a contiguous 2D sequence of N(r) received tokens with no rank-major or expert-major structure.
With `NCCL_EP_LAYOUT_EXPERT_MAJOR`, dispatch output is grouped by local expert, optionally padded via `dispatch_output_per_expert_alignment`.

**Handle creation**

| Operation | Struct      | Field           | Dims |
|-----------|:------------|:----------------|:----:|
| Create    | layout_info | expert_counters | [L]  |


**Forward pass**

`topk_idx` is supplied once via `ncclEpCreateHandle` (or refreshed via
`ncclEpUpdateHandle`) and cached on the handle; subsequent dispatches reuse it.

| Operation | Struct           | Field        | Dims       |
|:---------:|:-----------------|:-------------|:----------:|
| Dispatch  | dispatch_inputs  | tokens       | [B x H]    |
|           | dispatch_inputs  | topk_weights | [B x K]    |
|           | dispatch_outputs | tokens       | [N(r) x H] |
|           | dispatch_outputs | topk_weights | [N(r) x K] |
|           | dispatch_outputs | topk_idx     | [N(r) x K] |
| Combine   | combine_inputs   | tokens       | [N(r) x H] |
|           | combine_outputs  | tokens       | [B x H]    |

**Backward pass**

Compared to the Forward pass, the Backward pass requires per-token routing
weights to be passed as `combine_inputs.topk_weights` and returned via
`combine_outputs.topk_weights`.

| Operation | Struct           | Field            | Dims       |
|:---------:|:-----------------|:-----------------|:----------:|
| Dispatch  | dispatch_inputs  | tokens           | [B x H]    |
|           | dispatch_inputs  | topk_weights     | [B x K]    |
|           | dispatch_outputs | tokens           | [N(r) x H] |
|           | dispatch_outputs | topk_weights     | [N(r) x K] |
|           | dispatch_outputs | topk_idx         | [N(r) x K] |
| Combine   | combine_inputs   | tokens           | [N(r) x H] |
|           | combine_inputs   | **topk_weights** | [N(r) x K] |
|           | combine_outputs  | tokens           | [B x H]    |
|           | combine_outputs  | **topk_weights** | [B x K]    |


# Usage

## Prerequisites

### Dependencies

| Component | Version                    | Notes                             |
|-----------|----------------------------|-----------------------------------|
| CUDA      | 13+                        | Required                          |
| NCCL      | 2.29+                      | With Device API and GIN support   |
| MPI       | Any (OpenMPI, MPICH, etc.) | Required for multi-process launch |
| GPU       | Hopper (H100) or Blackwell | Tested configurations             |

### Discover compute capabilities

Use `nvidia-smi` command to detect the compute capabilities of your NVIDIA GPU.

For example, on Hopper system with `compute_cap` of `90`, the output looks like below:
```bash
$ nvidia-smi --query-gpu=compute_cap --format=csv
compute_cap
9.0
...
```

### Set the environment

```
export COMPUTE_CAP=<discovered compute_cap>
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export PATH="${CUDA_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

## Building

### Step 1: Get NCCL

This repo vendors a compatible NCCL build via git submodule at
`third_party/nccl`. Build it once:

```bash
make nccl-submodule   # -> third_party/nccl/build/{include,lib}
```

The EP unit tests use the independently pinned GoogleTest submodule at
`third_party/googletest`; they do not rely on a copy inside NCCL.

`NCCL_HOME` defaults to `third_party/nccl/build` automatically — no export
needed. To use your own NCCL build instead (e.g. a different version, or one
with custom patches), set `NCCL_HOME` to override the default:

```bash
export NCCL_HOME=/path/to/your/nccl/build
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

### Step 2: Build NCCL EP Library and Test

```bash
make -C nccl_ep MPI=1 \
       NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
```

Once `make` command is successfuly completed, the following files will be created:
- `${BUILDDIR}/lib/libnccl_ep.a` - Static library
- `${BUILDDIR}/lib/libnccl_ep.so` - Shared library (for Python bindings)
- `${BUILDDIR}/include/nccl_ep.h` - C API header
- `${BUILDDIR}/test/nccl_ep/ep_test` - Test application for both Low-Latency and High-Throughput modes
- `${BUILDDIR}/test/nccl_ep/ep_bench` - Benchmark application for both Low-Latency and High-Throughput modes

`BUILDDIR` defaults to `<repo>/build`. Set `BUILDDIR=/path/to/build` to place
all NCCL EP outputs elsewhere.

## Running

### Environment Setup

Make sure to set the generic environment according to the
[Set the environment](#set-the-environment) section.

```bash
# NCCL GIN configuration recommended for multi-node RDMA:
export NCCL_GIN_TYPE=3  # GDAKI - GPU Direct Async Kernel-Initiated
```

For debugging, the following variables can be set

```bash
export NCCL_DEBUG=INFO        # Enable NCCL debug output
export NCCL_DEBUG_SUBSYS=ALL  # All subsystems
export NCCL_EP_DEBUG=1        # NCCL-EP diagnostics, including zero-copy selection and fallback reasons
export NCCL_EP_ENV_VERBOSE=true  # Resolved NCCL-EP environment at group creation
export NCCL_EP_JIT_LOG=1      # Runtime kernel-compilation diagnostics (see docs/documentation/jit.md)
```

> **Runtime kernel compilation.** NCCL EP compiles some device kernels on first
> use, so a deployment needs a reachable `nvcc`, the kernel/NCCL/CUDA headers, and
> a writable cache directory (default `/tmp/nccl_ep/jit`). A build that stays
> where it was built needs no configuration; relocated installs, wheels,
> containers, and read-only filesystems do. See
> [JIT Kernel Compilation](docs/documentation/jit.md) for every knob, including `NCCL_EP_HOME`.

#### NCCL EP compatibility with newer NCCL runtimes

- Builds compiled against NCCL >= 2.31 use the runtime-reported DevComm size directly.
- Builds compiled against NCCL >= 2.30.5 and < 2.31 must enable
  `NCCL_DEV_API_JIT` when running with a newer NCCL runtime.
- Builds compiled against NCCL < 2.30.5 do not support this compatibility path.

```bash
export NCCL_HOME=/path/to/nccl-2.31
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_DEV_API_JIT=1
```

> The runtime lib and headers must come from the same NCCL installation; mixing Device API layouts is unsupported.

### High-Throughput tuning

```bash
# Override the HT dispatch/combine tokens-per-chunk. HT mode only; resolved once
# per group at ncclEpCreateGroup. Must be a multiple of 32 (non-conforming values
# are rounded up with a warning). When unset, the chunk size defaults to:
#   - RDMA / multi-node configs:  64
#   - LSA-only / single-node:     NUM_OF_TOKENS_PER_GROUP (4) * resolved SM count,
#                                 rounded up to a multiple of 32
export NCCL_EP_TOKENS_PER_CHUNK=128

# Dump every resolved NCCL EP environment variable (name + value, or "unset")
# at group creation, including NCCL_EP_TOKENS_PER_CHUNK.
export NCCL_EP_ENV_VERBOSE=true
```

HT dispatch and combine automatically adjust their pipeline configuration to
fit the device shared-memory (SMEM) limit. The following variables override the
pipeline defaults:

```bash
# Optional dispatch overrides with defaults
export NCCL_EP_DISPATCH_NUM_STAGES=12
export NCCL_EP_DISPATCH_NUM_PIPELINES=2

# Optional combine overrides with defaults
export NCCL_EP_COMBINE_NUM_STAGES_G2S=12  # 4 with multiple LSA teams
export NCCL_EP_COMBINE_NUM_STAGES_S2G=2
export NCCL_EP_COMBINE_NUM_PIPELINES=2    # 1 with multiple LSA teams
```

More stages or pipelines can improve overlap but consume more SMEM. Explicit
overrides are not reduced by the automatic tuner, and the operation returns
`ncclInvalidArgument` if the requested configuration cannot fit. When unset, the
tuner selects stage and pipeline counts that fit while preserving as much
overlap as possible. Leave these variables unset unless tuning a representative
workload. With `NCCL_EP_ENV_VERBOSE=true`, NCCL EP also prints the requested and
selected pipeline configuration, SMEM usage, and device limit when an HT kernel
configuration is first used.

Dispatch/combine SM-budget controls (LL and HT):

    # Legacy fallback: used by both operations unless overridden below
    export NCCL_EP_COMM_SMS=16

    # Independent operation overrides
    export NCCL_EP_DISPATCH_SMS=12
    export NCCL_EP_COMBINE_SMS=24

    # Shuffle and preprocessing default to all device SMs
    export NCCL_EP_SHUFFLE_SMS=<number_of_sms>
    export NCCL_EP_PREPROCESS_NUM_SMS=<number_of_sms>

Each operation resolves its budget independently. Precedence, highest first,
is its operation-specific environment variable, NCCL_EP_COMM_SMS, its
operation-specific ncclEpGroupConfig_t field, max_num_sms, then the
algorithm default (all device SMs for LL and 16 for HT). Leaving both new
fields and variables unset therefore preserves legacy behavior; setting one
operation does not alter the other. Values must be in [1, device_sm_count]
after resolution and must satisfy the LL warp-group geometry when LL is used.

The budgets do not need to be multiples of 4 or 8; any positive integer within
the device and LL geometry limits is supported. Combine commonly benefits from
a larger budget than dispatch.
An SM budget is the upper bound supplied to an operation launch-geometry

calculation. It is not always the literal CUDA grid size: LL rounds the grid to
its expert/warp-group geometry, while HT currently launches one communication
CTA per budgeted SM. Shuffle and preprocessing kernels have separate budgets.

By default EP guards its internal communication buffers so that neighboring
dispatch/combine calls cannot corrupt each other's data; this is safe and needs
no configuration. Advanced callers that can already guarantee consecutive EP
operations will not race on these buffers may disable the guard to reclaim its
overhead with `export NCCL_EP_DISABLE_GUARD=1`.


# Core Concepts

## Key Data Structures

### `ncclEpTensor_t` - Multi-dimensional Tensor Descriptor

A lightweight value-type struct that encapsulates tensor metadata and data layout.
Declare on the stack, as a struct member, or statically — no heap allocation needed for
the descriptor itself. Always zero-initialise with `NCCL_EP_TENSOR_INIT` and then fill
the fields directly:

```c
size_t dims[2] = { num_tokens, hidden };
ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
t.ndim = 2;
t.datatype = ncclFloat16;
t.data = my_device_ptr;   // device pointer (or set win_hdl/win_offset for windows)
t.sizes = dims;           // caller-owned array of length `ndim`

// No destroy needed — the descriptor holds no resources, but `dims` must
// outlive the descriptor for the duration of any library call that uses it.

// Access tensor properties directly:
data  = t.data;    // data pointer
ndim  = t.ndim;    // number of dimensions
sizes = t.sizes;   // sizes pointer (points to caller-owned storage)
```

### `ncclEpGroup_t` - EP Group Configuration

Created from an NCCL communicator, manages the distributed EP configuration across all ranks in the group:

```c
typedef struct {
    unsigned int size;                          // = sizeof(struct); ABI-size check
    unsigned int magic;                         // = NCCL_EP_MAGIC; type/init check
    unsigned int version;                       // = NCCL_EP_API_VERSION
    ncclEpAlgorithm_t algorithm;                // HT or LL mode
    unsigned int num_experts;                   // Total experts across all ranks
    unsigned int max_dispatch_tokens_per_rank;  // Max tokens any single rank dispatches
    unsigned int max_recv_tokens_per_rank;      // Max tokens any single rank receives
                                                //   HT: required (must be >= max_dispatch_tokens_per_rank)
                                                //   LL: unused (buffers always sized automatically); pass NCCL_EP_AUTO
    unsigned int max_token_bytes;               // Max token-row bytes. For quantized transmission, quantized
                                                //   token data and scales must fit within this budget; see docs/documentation/quantization.md.
    unsigned long int rdma_buffer_size;         // RDMA buffer size for LL mode.
                                                //   NCCL_EP_AUTO  → lazy: allocate on first ncclEpInitHandle, sized to that
                                                //                  handle's actual (layout, num_topk); collective re-grow
                                                //                  if a later handle needs more. See LL section for caveats.
                                                //   explicit > 0  → allocate exactly that many bytes at group time;
                                                //                  ncclEpInitHandle returns ncclInvalidUsage if a layout
                                                //                  doesn't fit. No reallocation ever performed.
    unsigned int num_qp_per_rank;               // Queue pairs per rank (NCCL_EP_AUTO for auto)
    unsigned int num_channels;                  // Channels per rank (NCCL_EP_AUTO for auto)
    unsigned int max_num_sms;                   // Legacy fallback for both dispatch and combine
    ncclEpAllocConfig_t alloc;                  // Custom device-memory allocator (zero-init → cudaMalloc/cudaFree)
    unsigned int enable_mask;                   // Enable active-mask fault tolerance (LL only)
    uint64_t timeout_ns;                        // GPU-side wait-loop timeout (0 = default)
    ncclEpZeroCopyMode_t zero_copy;             // Window-backed staging control (AUTO / OFF / ON);
                                                //   see docs/documentation/zero_copy.md.
    ncclEpOverflowPolicy_t overflow_policy;     // HT recv-overflow policy (AUTO → TRAP, or DROP);
                                                //   see docs/documentation/overflow_policy.md.
    unsigned int num_topk;                      // Upper bound on per-token top-k across the group's
                                                //   handles. Optional (0 = unset); required for HT
                                                //   eager mode with the expert-major layout;
                                                //   see docs/documentation/eager_mode.md.
    unsigned char padding_v2[4];                // Consumes V2 tail padding; future fields append after
    unsigned int dispatch_num_sms;              // Dispatch budget (AUTO inherits max_num_sms/default)
    unsigned int combine_num_sms;               // Combine budget (AUTO inherits max_num_sms/default)
} ncclEpGroupConfig_t;

// Use NCCL_EP_GROUP_CONFIG_INIT to pre-fill size/magic/version correctly.
```

Independently passed public structures start with the same `size`/`magic`
fields and are append-only. A newer library accepts a frozen older
prefix and supplies defaults for fields introduced later; an older library
accepts the known prefix of a larger future structure and ignores its unknown
tail. Callers that require a newly introduced field must compare the runtime
version returned by `ncclEpGetVersion` before relying on it. Historical
`NCCL_EP_*_Vn_SIZE` constants freeze each released boundary, while
`NCCL_EP_*_SIZE` describes the current struct size.
`ncclEpGroupConfig_t::version` records the caller's
`NCCL_EP_API_VERSION`; a mismatch is reported as a warning rather than
rejecting an otherwise size-compatible configuration.

### `ncclEpHandle_t` - Operation Handle

Maintains state for a sequence of related MoE operations, i.e. dispatch and combine pairs for forward and (optionally) backward passes. The handle encapsulates routing metadata and communication buffers.

## Algorithm-related configurations

### High Throughput (HT)

- Supports `NCCL_EP_LAYOUT_FLAT` and `NCCL_EP_LAYOUT_EXPERT_MAJOR` layouts.

- **`NCCL_EP_LAYOUT_FLAT`**: dispatch output is a contiguous 2D sequence `[N(r) x hidden]` where `N(r)` is the total number of tokens targeting this rank.
  - Static allocation: the output buffers are pre-allocated with capacity for `max_recv_tokens_per_rank` tokens. Required under CUDA Graph capture.
  - Query-then-allocate: supply `ncclEpLayoutInfo_t.recv_total_counter` to `ncclEpCreateHandle` / `ncclEpUpdateHandle`; the metadata kernel writes the actual N(r) there, and the caller copies it device-to-host and synchronizes before allocating. The count is readable in any mode; to size the dispatch outputs to it rather than to the worst case, create the group in eager mode (see below).
  - `dispatch_outputs.topk_idx` and `dispatch_outputs.topk_weights` carry per-slot routing metadata alongside the received tokens.
  - The caller uses `topk_idx` to route each slot to the correct local expert(s), applies the weighted reduction using `topk_weights`, and passes the pre-reduced `[N(r) x hidden]` tensor as `combine_inputs.tokens` to `ncclEpCombine`.

- **`NCCL_EP_LAYOUT_EXPERT_MAJOR`**: dispatch output is grouped by local expert. Each expert's slice is optionally padded to a multiple of `dispatch_output_per_expert_alignment` (set via `ncclEpHandleConfig_t`).
  - Tokens arrive pre-sorted by expert; the caller feeds each expert's slice directly without needing `topk_idx` for routing.
  - Set `ncclEpLayoutInfo_t.expert_counters` (1D tensor, length = `num_local_experts`) to receive per-expert received token counts.

### Low Latency (LL)

- Supports `NCCL_EP_LAYOUT_EXPERT_MAJOR` and `NCCL_EP_LAYOUT_RANK_MAJOR` layouts.
- Accepts `ncclInt32` or `ncclInt64` routing indices and supports up to 32 top-k
  entries.
- For rank-major `topk_idx` output (and HT flat), `layout_info.recv_topk_idx_kind`
  selects local or global expert IDs; `-1` marks a slot not routed locally.
  `AUTO` currently selects local IDs; choose `LOCAL` or `GLOBAL` to pin the contract.
- Output tokens are 3D:
  - expert-major: `[num_local_experts, num_ranks * max_dispatch_tokens_per_rank, hidden]`; `expert_counters[e]` gives the active rows for expert `e`.
  - rank-major:   `[num_ranks, max_dispatch_tokens_per_rank, hidden]`.
- Supports `send_only` (in `ncclEpDispatchConfig_t` / `ncclEpCombineConfig_t`) to enable computation/communication overlapping.
- `zero_copy` is dispatch-only in LL; combine always stages. See
  [Zero-Copy](docs/documentation/zero_copy.md).
- Does not support dynamic `max_dispatch_tokens_per_rank` detection.

## HT-specific modifiers

### Eager mode

A group normally commits to a fixed per-rank recv budget, and every dispatch recv
tensor is sized to it. Creating the group with
`max_recv_tokens_per_rank = NCCL_EP_AUTO` selects *eager mode* instead, where the
caller sizes those tensors per iteration to the actual recv count of the current
routing, read from `ncclEpLayoutInfo_t.recv_total_counter`.
Refer to the [Eager Mode](docs/documentation/eager_mode.md) documentation for more
details.

### Recv overflow policy

Routing is data-dependent, so a rank can be targeted by more tokens than the
`max_recv_tokens_per_rank` budget its group was created with.
`ncclEpGroupConfig_t::overflow_policy` selects what happens then: by default the
device traps and the process aborts, while `NCCL_EP_OVERFLOW_DROP` discards the
excess tokens and lets the pipeline continue. LL ignores the field.
Refer to the [Recv Overflow Policy](docs/documentation/overflow_policy.md)
documentation for more details.

## LL-specific modifiers

### `rdma_buffer_size` and lazy allocation (LL only)

The group's RDMA buffer is sized by `config.rdma_buffer_size`:

- **`NCCL_EP_AUTO`** (recommended for most users): the buffer is **not** allocated at `ncclEpCreateGroup` time. 
The first `ncclEpInitHandle` allocates it sized to that handle's actual `(layout, num_topk)`.
A later `ncclEpInitHandle` whose layout needs a larger buffer (for example, the first `EXPERT_MAJOR`
handle on a group that previously only hosted `RANK_MAJOR` handles, or a handle with a larger `num_topk`)
**collectively reallocates** the buffer: deregister window → free → `ncclMemAlloc` → register.
The recorded layout offsets on every live handle are pure offsets relative to the group's `rdma_buffer` and resolve correctly against the new base at use time, so existing handles remain valid.

  Two constraints follow from a reallocation:
  1. **All ranks must call `ncclEpInitHandle` in lockstep with the same `(layout, num_topk)`**. With AUTO sizing, `ncclEpInitHandle` is a conditionally collective call.
  2. **Reallocation drops the contents of the old RDMA buffer.** Any operation issued with `send_only = 1` that has staged data but is still awaiting its receive half via `ncclEpComplete` will lose its in-flight data. Drain all such operations before triggering a reallocation.
  3. **CUDA graph capture bakes the RDMA base pointer into the captured kernel parameters.** `ncclEpInitHandle` (in AUTO mode) must not be called between `cudaStreamBeginCapture` and `cudaStreamEndCapture`. Any previously captured graph containing EP kernels must be destroyed and re-captured after a reallocation.

- **Explicit `> 0`**: the buffer is allocated to exactly that size at `ncclEpCreateGroup` time. `ncclEpInitHandle` is purely local; it returns `ncclInvalidUsage` if the requested `(layout, num_topk)` does not fit. Use this mode if you need to avoid collective handle creation, mid-stream reallocation, or graph invalidation. There is currently no public API for computing the required size — one is planned for a future release. Until then, size conservatively: a buffer that is too small for the requested `(layout, num_topk)` is reported cleanly by `ncclEpInitHandle` rather than failing later.

## Zero-copy

Dispatch and combine normally move payloads through library-owned staging
buffers. A caller can enable direct peer access to input/output buffers by
attaching a NCCL window (`ncclCommWindowRegister`) to the respective tensor.
Refer to the [Zero-Copy](docs/documentation/zero_copy.md) documentation for more
details.

## Custom Allocators

EP allocates its internal device buffers with `cudaMalloc`/`cudaFree` by default.
A caller can route those allocations through its own memory pool or
framework-specific allocator by setting `ncclEpGroupConfig_t::alloc`
(`ncclEpAllocConfig_t`).
Refer to the [Custom Allocators](docs/documentation/custom_allocators.md)
documentation for more details.

## Elastic (GPU + CPU) receive buffers

Because the receive token count is data-dependent, a receive buffer sized for the
absolute worst case wastes scarce GPU memory. An *elastic buffer* is one
contiguous VA range backed partly by GPU memory and partly by host memory, so the
GPU segment covers the common case while the CPU segment absorbs rare outliers.
It is reference code rather than a shipped API, and only the Expert-Major kernel
path can drive its CPU segment safely.
Refer to the [Elastic Receive Buffers](docs/documentation/elastic_buffers.md)
documentation for more details.

## API Reference

The complete C API reference lives in **[docs/documentation/api_reference.md](docs/documentation/api_reference.md)**,
covering all 18 public entry points:

| Group                    | Functions                                                                                                    |
|--------------------------|--------------------------------------------------------------------------------------------------------------|
| Library                  | `ncclEpGetVersion`                                                                                           |
| Group Management         | `ncclEpCreateGroup`, `ncclEpGroupDestroy`                                                                    |
| Tensor Descriptors       | `ncclEpTensorAlloc`, `ncclEpTensorDestroy`                                                                   |
| Handle Management        | `ncclEpCreateHandle`, `ncclEpInitHandle`, `ncclEpUpdateHandle`, `ncclEpHandleMemSize`, `ncclEpHandleDestroy` |
| Communication Operations | `ncclEpDispatch`, `ncclEpCombine`, `ncclEpComplete`                                                          |
| Fault Tolerance (LL)     | `ncclEpMaskQuery`, `ncclEpMaskUpdate`, `ncclEpMaskClean`, `ncclEpGetAsyncError`, `ncclEpErrorClear`          |

# Execution Modes

Both `ncclEpDispatch()` and `ncclEpCombine()` support synchronous semantics, where
the call completes the whole operation, and — in Low Latency mode — staged
semantics, where `send_only` initiates the transfers and `ncclEpComplete()`
finishes them so the application can overlap computation in between.
Refer to the [Execution Modes](docs/documentation/execution_modes.md)
documentation for more details.

# Usage Examples

Complete, compilable walkthroughs for High Throughput (forward and backward pass)
and Low Latency (expert-major and rank-major) live in the
[Usage Examples](docs/documentation/usage_examples.md) documentation. For a
runnable version, see [`ep_test.cu`](ep_test.cu).
