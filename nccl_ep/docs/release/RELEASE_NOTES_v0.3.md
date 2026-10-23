# NCCL EP v0.3 — Release Notes

These notes cover NCCL EP **0.3.0**, relative to [v0.2](RELEASE_NOTES_v0.2.md).

This release adds optimized low-latency (LL) NVLink kernels, high-throughput (HT)
pull/push transport and fused metadata, MXFP8 combine, and Attention–FFN
Disaggregation (AFD). It also reduces memory use and expands tuning controls.

## New Features

### Attention–FFN Disaggregation

`num_attn_ranks` partitions a communicator into attention ranks followed by FFN
ranks. Attention ranks supply tokens and receive combined results; FFN ranks own
the experts. To place attention and FFN on separate RDMA-connected nodes, map
their rank ranges to GPUs on the respective nodes when creating the communicator.
Separate-node placement is optional.

The initial scope is universal LL, rank-major layout, equal numbers of attention
and FFN ranks, unquantized operations, and no CFT. Expert count must
be divisible by the FFN-rank count with at least one expert per FFN rank. All ranks
participate in each collective: attention ranks call `ncclEpAfDispatch` and
`ncclEpAfCombine`; FFN ranks call `ncclEpAfRecv` and `ncclEpAfReturn`. See the
[AFD API](../documentation/api_reference.md#afd-role-wrappers).

### LL kernels and zero-copy

Eligible rank-major operations within one Load-Store Accessible (LSA) domain now
use optimized dispatch and combine kernels. More hidden dimensions are supported.

Fused `DS_FP8E3M4` dispatch now supports zero-copy token and scale outputs.
Zero-copy support for LL combine may now require token inputs to be supported
by a NCCL window.

### Opt-in Compute Fabric Transport

`NCCL_EP_LL_CFT=1` enables Compute Fabric Transport (CFT) for LL dispatch within
one NVLink Domain. It requires Blackwell or later hardware, CUDA 13.3 or newer for
build/runtime/JIT, driver 610.43.02 or newer (driver API 13030), supported fabric
endpoints, and Fabric Manager. All ranks must agree on the setting; unsupported
platforms fail group creation.

CFT zero-copy supports unquantized rank-major BF16/FP16 output with a hidden
dimension divisible by 256.

### HT flow and metadata policies

Flat and Expert-Major default to **pull-based dispatch**/**push-based combine**
for eligible single-LSA configurations with a supported positive `num_topk`.
Other configurations retain **push-based dispatch**/**pull-based combine**.
`ncclEpGroupConfig_t::ht_flow` selects the flow explicitly. Pull/push supports
unquantized or scales-forward dispatch with unquantized combine.

Metadata fusion combines routing metadata exchange and receive-map construction in dispatch.
The new `meta_exchange_policy` selects automatic metadata fusion, requires fusion,
or forces scan-based exchange. Fusion requires an eligible single-LSA configuration
with a fixed receive budget; eager sizing and update-time count/offset queries
are incompatible with `REQUIRE_FUSED`. Explicit API selections override environment
controls, and all ranks must agree on the configuration. See
[HT flow and metadata](../../README.md#flat-pullpush) for limits and counter-query usage.

### MXFP8 combine

The new `NCCL_EP_COMB_QUANT_MXFP8` option enables transparent combine quantization.
It transports BF16 expert output as FP8 E4M3 with E8M0 block scales and returns
BF16 results. Scales are generated internally, so input `scales` must be `NULL`.
Supported for forward and backward HT Expert-Major
`local_permute` combine within one LSA domain, with CUDA 12.8 or newer and hidden
size divisible by 512.

MXFP8 combine is unsupported in pull/push mode, the default for eligible groups.
Explicitly select push/pull with `ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL`.

### Import a prepared HT handle

`ncclEpImportHandle` recreates an HT handle when its host object cannot be retained
between calls but its prepared device buffer can. Import uses the existing EP
group and matching host metadata without another update or device-state copy.
A snapshot is a saved copy of the prepared handle buffer.

Snapshot after update for forward dispatch; for count-mode backward dispatch or
combine, snapshot after the matching forward dispatch
completes. Imported state must match the group and handle configuration; operations
sharing group workspace still require ordered use. See the
[handle API](../documentation/api_reference.md#handle-management).

### Python and tools

- Python bindings expose the new C APIs. Select MXFP8 combine with
  `CombineConfig(quantization_recipe=CombineQuantizationRecipe.MXFP8)`.
- `ep_bench --cuda-graph` benchmarks captured dispatch/combine replay.
- `af_test` exercises AFD round trips.

## Resource Usage

- **HT routing and staging:** compact routing metadata reduces transfer volume
  and can improve HT dispatch performance. GB200 NVLink benchmarks of HT
  Expert-Major dispatch at 16–32 GPUs showed approximately 12–14% median reductions
  in host-observed dispatch time versus dense metadata. These are medians across
  six tested workloads; results at smaller EP sizes were mixed.
  Shared workspace, smaller Expert-Major staging, and reused scale capacity
  reduce memory use.
- **HT dispatch copies:** device-counted TMA/SIMT copies avoid host
  count readback in fixed-capacity mode; TMA is the default with copy-engine
  fallback.
- **HT chunk tuning:** HT push/pull supports separate dispatch and combine chunk
  sizes within a single LSA domain. `NCCL_EP_DISPATCH_TOKENS_PER_CHUNK` overrides
  dispatch chunk size; when unset, eligible configurations select it automatically
  based on token count.
- **LL launch overhead:** warm-cache launches reuse JIT preparation, and relative-rank
  staging distributes NVLink writes.
- **Independent SM budgets:** `dispatch_num_sms` and `combine_num_sms`, or
  `NCCL_EP_DISPATCH_SMS` and `NCCL_EP_COMBINE_SMS`, tune each operation separately.
  Unset controls retain the shared-budget behavior. See the
  [tuning guide](../../README.md#high-throughput-tuning) for precedence and chunk controls.
- **Optional NVLink scheduling:** `NCCL_EP_ENABLE_NUCS=1` enables utilization-centric
  scheduling; it requires CUDA 13.3 or newer build headers and driver support.

## Upgrade Notes

- **Backward compatibility:** applications compiled against v0.1 or v0.2 remain
  compatible with v0.3. Initialize public structures with their `NCCL_EP_*_INIT` macros.
- **HT compact receive ordering:** compact push dispatch now returns expert IDs
  and weights in source top-k order, visible in Flat receive tensors. Set
  `NCCL_EP_HT_RECV_TOPK_ORDER=expert-ascend` to retain ascending expert order.
  Dense fallback and pull/push ordering are unchanged.
- **HT staging alignment:** `max_token_bytes` must be a multiple of 16;
  misaligned configurations are rejected before TMA writes.
- **Python CUDA variants:** packaging supports CUDA 12 and CUDA 13 libraries,
  selected by the active `cuda.bindings` major. JIT compiler and headers must
  match that major. See the [Python guide](../../../python/README.md).

NCCL build/runtime compatibility is checked at group creation:

| NCCL used to build EP | Supported NCCL runtime |
| --- | --- |
| Earlier than 2.30.5 | Same version as the build. |
| 2.30.5 through 2.30.x | Same version, or newer with `NCCL_DEV_API_JIT=1` (enabled automatically if unset). |
| 2.31 or newer | Same version or newer. |

JIT headers must match the runtime NCCL installation.

## Fixes

- Improved stability for transported meta-data.
- Improve HT synchronization, including NUCS and multiple LSA teams on one physical node.
- Bound HT Flat conversion writes, report unpadded receive totals, and recount
  delivered Expert-Major tokens under `NCCL_EP_OVERFLOW_DROP`.
- Preserve per-handle pull/push routing snapshots for interleaved handles.
- Preserve sub-millisecond timeout precision and prevent cycle-conversion overflow.
- Install split LL JIT headers in CMake builds.

## Known Limitations

- **AFD expert count:** v0.3 additionally requires more experts than total
  communicator ranks (attention plus FFN).
- **HT supports at most 33 RDMA domains.** Exceeding the limit fails kernel JIT
  compilation, as in v0.2.
