# NCCL EP v0.2 — Release Notes

This release introduces quantization support, broadens high-throughput (HT)
operational modes, supports more use cases (i.e. 1F1B scheme, zero-copy for LL, etc.),
delivers performance optimizations and bug fixes, and lifts several hard-coded limits.

A versioned ABI keeps v0.1-compiled applications working against v0.2, and new
environment knobs expose SM allocation, HT pipelining, and build configuration.

## HT default flow compatibility

AUTO now selects pull dispatch / push combine for both flat and expert-major
when the group specifies top-k in [1, 32], at most 65535 experts, and one LSA
team of at most 128 ranks, without explicit EM duplication modes. Unspecified
top-k retains push/pull. Explicit flow and environment overrides remain supported.
Invalid flow environment values warn and use the AUTO policy.

MXFP8 combine is unsupported with pull/push and fails with an actionable error.
Use explicit `NCCL_EP_HT_FLOW_PUSH_PULL` with expert-major local-permute
(`zero_copy` AUTO/OFF, no duplication overrides) for this recipe. Pull/push
retains staging even with `zero_copy=ON`, which can increase group memory.
Input tokens, weights, and scales can independently use windows or staging;
the backing choice for each operand must match across ranks.

## HT communication SM defaults

Expert-major dispatch and combine default to 24 SMs on non-Rubin devices and
32 on Rubin SM107. Flat dispatch defaults to 16; flat combine defaults to 32
when group `max_dispatch_tokens_per_rank >= 8192`, otherwise 16. The threshold
uses configured capacity, not the current handle batch. Explicit SM overrides
retain their precedence.

Multi-LSA HT group resources require `num_qp_per_rank >= 1 + 2 * group dispatch
SMs`: automatic defaults require 49 contexts, or 65 on Rubin, even for flat-only
users. AUTO provisions them; smaller explicit budgets are rejected. The 24-SM
expert-major default produces a 96-token chunk, outside the adaptive dispatch
chunk policy's validated defaults; dispatch overrides must divide that chunk.

## HT benchmark defaults

`ep_bench` requests fused metadata by default where supported; use
`--unfused-meta-dispatch` for comparison. Random-routing receive capacity now
uses `ceil(1.15 * top_k * max_tokens_per_rank)`, plus expert-major padding.
This is a benchmark policy, not a worst-case guarantee. Explicit budgets and
legacy duplication staging requirements remain authoritative. Existing default
benchmark time series change with these defaults; explicit push/pull rows
retain coverage of the legacy implementation.

## New Features

### Quantization support

Dispatch-path quantization is supported for both LL and HT through two mechanisms.

`QUANT_FWD` is a generalized scale-forwarding mechanism, not a quantization recipe. It
byte-forwards two 2D inputs — tokens `[B x H]` and scales `[B x S]` — without interpreting
either; the scale-block size `S` is taken directly from the scale tensor. The
quantization format therefore lives entirely in the caller's tensors, not in an EP enum value,
so formats such as MXFP8 (block 32, E8M0) are expressed through the tensors rather than
requiring a dedicated recipe.

`DS_FP8E3M4` (DeepSeek recipe) is the internal, LL-only path where EP performs the
quantization internally rather than forwarding what the caller produced. The caller provides
BF16 tokens and no input scales; dispatch quantizes them on the fly and emits FP8 (E4M3) token
bytes plus generated FP32 scales.

Commands:
- Forward: `ep_bench -a ll -l [em | rm] --dispatch-quantization scales-forward`,
  `ep_bench -a ht -l [fl | em] --dispatch-quantization scales-forward`
- Internal (LL only): `ep_bench -a ll -l [em | rm] --dispatch-quantization ds-fp8e3m4`

**NOTE:** for quantized transmission, quantized token data and scales must together fit within the
configured `max_token_bytes` budget. HT quantized dispatch staging capacity was reduced
accordingly, lowering the memory footprint.

#### Combine/NVFP4 recipe — EXPERIMENTAL

`NCCL_EP_COMB_QUANT_NVFP4` enables quantized expert-output transport on the combine path.
LL-only; the caller supplies FP32 per-token global scales through the new `scales` field in
`ncclEpCombineInputs_t`, following the DeepEP-LL NVFP4 pack/dequantize contract. The API
contract, supported shapes, and numerical behavior may change before this graduates — do not
depend on it in production.

`QUANT_FWD` dispatch gained `ncclFloat4x2` as a packed-FP4 token wire type, which
carries its own shape and alignment rules — see the documentation for details.

### HT/EM modes

High-throughput mode now supports expert-mapping variants: `local_dup`, `nvlink_dup`, and
`local_permute`.

The mode is selected automatically from the zero-copy setting and the number of LSA teams:

| Configuration | Selected mode |
|---|---|
| Zero-copy off | `local_permute` (FLAT dispatch + permute kernels) |
| Zero-copy on, multiple LSA teams | `nvlink_dup` (sender duplicates per-expert over NVLink) |
| Zero-copy on, single LSA team | `local_dup` (receiver-side fan-out) |

Two environment flags override the automatic choice:
- `NCCL_EP_HT_EM_LOCAL_DUP` — force `local_dup`
- `NCCL_EP_HT_EM_NVLINK_DUP` — force `nvlink_dup`

The two are mutually exclusive; setting both fails group creation with `ncclInvalidUsage`.
There is no override flag for `local_permute` — it is the default whenever neither flag is set
and zero-copy is off.

Benchmarking:
`ep_bench -a ht -l em --ht-em-mode [local_dup | nvlink_dup | local_permute]`


### Zero-copy support

Zero-copy is enabled for the LL dispatch flow on a single LSA domain (i.e. NVL domain) and rank-major layout.

In addition, memory footprint reduction is introduced if the user guarantees
zero-copy-only operation via `EpGroupConfig` for HT mode.

### Query-and-allocate / dynamic output sizing

Receive buffers can be sized dynamically by setting `max_recv_tokens_per_rank=AUTO`; the
library reports the required size and allocates accordingly.

Because `AUTO` derives a theoretical worst-case budget, HT staging can be too large to allocate
even when real routing never approaches that bound. Allocation failures now report the derived
budget and point at setting an explicit `max_recv_tokens_per_rank`, instead of surfacing a bare
out-of-memory error.

### Receive-buffer overflow handling

When routing directs more tokens at a rank than `max_recv_tokens_per_rank` can hold, the
behavior is selected by `ncclEpGroupConfig_t::overflow_policy` (`ncclEpOverflowPolicy_t`):

- `NCCL_EP_OVERFLOW_AUTO` (default) — overflow is detected and the operation fails with a
  diagnostic that names the insufficient budget and points at the drop policy. Output is never
  silently corrupted.
- `NCCL_EP_OVERFLOW_DROP` — tokens that do not fit are dropped and the operation completes
  normally. Callers opting in must be able to tolerate token loss.

### Global vs. local expert IDs in `recv_topk_idx`

Callers can now choose whether received top-k indices are expressed in global or local expert
numbering.

### Versioned ABI and backward compatibility

Public structures now carry frozen per-version ABI ledgers, so applications compiled against
v0.1 continue to operate correctly with v0.2 and later releases. Structure validation is
size-tolerant: a caller-supplied structure is accepted if it matches a frozen released layout
and is normalized against current-version defaults. Invalid structures now return
`ncclInvalidArgument` rather than aborting the process. The API version is bumped from 1 to 2.

## Data Type Support

### int32 top-k index tensors

LL mode accepts int32 top-k tensors (`ep_bench -a ll -l * --topk-idx-int32`), and HT mode gained
the same support (covered by unit test).

### FP16 and FP32 payloads

Added FP16 and FP32 data type support.



## Relaxed Restrictions

- **Max batch size**: the 8K batch-size cap is removed.
- **64-slot alignment**: batch sizes no longer need to be 64-slot aligned.
- **LL top-k > 9**: previously capped, now supported. Group creation validates the warp-group
  geometry (derived from expert count and communication SM count) and reports an actionable
  error when a requested top-k cannot be satisfied.
- **LL combine shared memory**: the combine path is now fitted to the device shared-memory
  limit, allowing configurations that previously exceeded it.
- **LL combine token slots**: an outdated TMA alignment guard was removed; there is no minimum
  token-slot restriction.
- **Group creation**: the version constraint on `ncclEpCreateGroup()` is lifted.

## Fixes

- **int32 overflow in LL / rank-major** at large batch sizes (reproduced with 16K hidden, 8K
  batch size, 64 ranks).
- **int32 overflow in HT / combine cross-LSA RDMA offsets** — at 16K tokens and 16K hidden with
  MNNVL disabled across 16 nodes, the offset product wrapped and the combine kernel hit an
  illegal memory access.
- **HT scan out-of-bounds access** when the max receive buffer is insufficient.
- **Rank-major receive top-k sentinels** are now reset in dispatch.
- **HT zero receive in eager mode** — ranks receiving no tokens are now handled correctly.
- **HT `local_permute_dup` out-of-bounds write** on overflow.
- **LL combine stage reuse** now correctly fenced.
- **LL epoch ring is now group-owned** — concurrent LL handles on one group no longer race on
  the shared double-buffered RDMA banks, and CUDA-graph replay advances the bank selector.
- **HT sync guard for consecutive same-direction ops** — enabled by default, disable via env for
  testing.
- **Byte-padded HT routing map** to fix a scan layout mismatch.

## New Environment Controls

- **Per-stage SM count** — tune the number of SMs used by the communication
  (`NCCL_EP_COMM_SMS`), shuffle (`NCCL_EP_SHUFFLE_SMS`), and preprocessing
  (`NCCL_EP_PREPROCESS_NUM_SMS`) stages.
  QA: `ep_bench -a ll -l [em | rm] -b [128 | 256] -h [2K | 7K]`
- **Communication buffer guard** (`NCCL_EP_DISABLE_GUARD`) — EP guards its internal
  communication buffers so neighboring dispatch/combine calls cannot race. On by default and
  safe; advanced callers that can guarantee consecutive EP operations will not race on these
  buffers may set it to reclaim the overhead.
- **Custom NVCC compiler** — the JIT toolchain can be pointed at a specific NVCC binary
  (`NCCL_EP_JIT_NVCC`, or `NVCC`), built for an explicit target architecture instead of the
  detected one (`NVCC_ARCH_FLAGS`), and given additional compile flags
  (`NVCC_EXTRA_FLAGS`).
- **HT pipeline control knobs** — stage and pipeline counts are auto-fitted to the available
  shared memory by default, but can be pinned explicitly for dispatch
  (`NCCL_EP_DISPATCH_NUM_STAGES`, `NCCL_EP_DISPATCH_NUM_PIPELINES`) and combine
  (`NCCL_EP_COMBINE_NUM_STAGES_G2S`, `NCCL_EP_COMBINE_NUM_STAGES_S2G`,
  `NCCL_EP_COMBINE_NUM_PIPELINES`), along with the HT chunk size
  (`NCCL_EP_TOKENS_PER_CHUNK`). Pinned values are honored exactly and never reduced by the
  fitter, so the combination must be valid on its own — stages must be a multiple of pipelines,
  with at least 3 stages per pipeline.

## Experimental

### Elastic (GPU + CPU) receive buffers

A receive buffer spanning a GPU segment and a CPU (HOST_NUMA) segment in one contiguous
virtual-address range, so the GPU segment can be sized for the common case while the CPU
segment absorbs rare outliers instead of sizing GPU memory for the worst case.

NCCL EP supports only the restricted configuration described by the reference implementation
(`examples/nccl_ep_elastic_buffer.h`, header-only reference code rather than a shipped EP API):
HT with the Expert-Major layout. LL and zero-copy receive into the CPU segment are not
supported.

## Implementation Notes

Internal changes with no API impact, but worth knowing.

- **LL kernels migrated to JIT.** The low-latency path is now JIT-compiled, as HT already was.
  This is not user-visible at the API level, but it moves LL kernel compilation to runtime — so
  LL deployments may now need the NVCC toolchain tuned through the environment (see
  Environment Controls), and are subject to first-call compilation latency and the JIT cache.
- **HT auto-fit to shared-memory limits.** HT stage and pipeline counts are fitted automatically
  to the device shared-memory budget, and combine shared-memory usage is validated. The knobs
  under Environment Controls override this fitter; pinned values are honored exactly and are
  never reduced to fit.
- **Additional HT synchronization around dispatch and combine.** LSA syncs were added at the
  dispatch head and combine tail, plus a guard on the RDMA receive staging, so neighboring
  dispatch/combine calls cannot corrupt each other's data. This is a behavior change relative to
  v0.1 with a cost in back-to-back operations; it is on by default and can be disabled via
  `NCCL_EP_DISABLE_GUARD`.

## Known Limitations

### HT: at most 33 RDMA domains

The HT combine path parallelizes its RDMA transfers across LSA teams within a single warp pass,
one lane per remote domain, so the number of RDMA domains is capped at 33. This is enforced by
a compile-time assertion in the combine path rather than validated at group creation, so
exceeding it surfaces as a kernel build failure rather than an `ncclEpCreateGroup` error.
