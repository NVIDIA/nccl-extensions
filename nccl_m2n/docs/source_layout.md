# Source Layout (`src/`)

Canonical, maintainer-facing inventory of the library translation units under
`src/`. This is the **source of truth** for "which file does what"; `AGENTS.md`
links here instead of duplicating the list (the README keeps its own
customer-facing tree in the shipped drop). Keep this in sync when adding,
removing, or renaming a TU.

> Maintainer-only: this file is on the release denylist
> (`scripts/release/payload_lib.sh`) and does not ship to `nccl/nccl`.

## Headers

| File | Role |
|---|---|
| `nccl_m2n.h` | Public C API (the only header callers need). |
| `reshard_internal.h` | Cross-TU declarations + process-global runtime getters. Internal. |
| `reshard_types.h` | Internal struct and enum definitions. |
| `reshard_limits.h` | Internal compile-time constants (`DEFAULT_NUM_CTAS=8`, `CHUNK_SIZE_BYTES=256K`, `DEFAULT_ELEMENTS_PER_CHUNK=32`, `STAGING_DEFAULT_*`, etc.; tensor rank uses public `NCCL_RESHARD_MAX_TENSOR_DIMS`). |
| `reshard_call_setup.h` | Shared per-call setup types (`ReshardWorkStream`, `ReshardTensorSetup`) + declarations. |
| `reshard_split.h` | Split-comm RING path (QP-scalability) types and declarations. |
| `reshard_kernels.cuh` | Header-only device helpers shared by the RING/DIRECT kernels. |
| `staging_types.h` | Staging-path struct definitions. |
| `staging_primitives.cuh` | Header-only staging device helpers. |
| `staging_buffer.h` | Staging-buffer / bucket-pool lifecycle declarations. |
| `staging_profile.h` | Staging profiling-scope declarations. |
| `m2n_handle.h` | Opaque handle definition. |
| `m2n_log.h` | `RESHARD_{WARN,INFO,DEBUG,TRACE,FATAL}` macros, level-gated to stdout; FATAL always to stderr + abort. |
| `m2n_checks.h` | Error-check macros. |
| `m2n_checked_math.h` | Overflow-checked integer math helpers. |
| `m2n_env_parse.h` | Env-var parse helpers. |

## Host TUs — core

| File | Role |
|---|---|
| `m2n_init.cc` | M2N lifecycle APIs + process-epoch runtime initialization. |
| `m2n_group.cc` | Thread-local grouped-submission state, bucketing, replay, and group API entry points. |
| `m2n_config.cc` | Config validation/application + `applyReshardEnv`. |
| `reshard_cache.cc` | DevComm cache + Window cache + (comm,dev) stream pool. `cacheFinalize` releases all. |
| `reshard_mesh.cc` | Mesh group info, transfer-plan builder, overlap calc. |
| `reshard_loadbalance.cc` | Replication load balancer (`UNIFORM` vs `NODE_AWARE`). |
| `reshard_call_setup.cc` | Shared per-call setup: descriptor validation, active-buffer checks, local-byte + staging GIN-count computation. |
| `reshard_prepare.cc` | RING and DIRECT kernel-parameter builders. Hot path on the host side. |
| `reshard_quantize.cc` | On-the-fly wire compression: config validation, scratch sizing, and the `ncclReshardQuantized` entry point. Stages quantize/dequantize around a coupled reshard; adds no transport code. |
| `reshard_scale_plane.cc` | Coupled (payload, scales) reshard: scale-plane validation and the `ncclReshardScaled*` entry points. Submits both planes as one M2N group; adds no kernel or plan of its own. |
| `reshard_transpose.cc` | Owns the cross-dim transpose private buffer. |

## Host TUs — split-comm RING path (QP-scalability)

| File | Role |
|---|---|
| `reshard_split_comm.cu` | Split-comm creation/caching (`reshardGetOrCreateSplitComms`, `reshardSplitEnsureResources`): commA FULL + commB RAIL. |
| `reshard_split_prepare.cc` | Split-comm RING kernel-parameter builder (host). |

## Host TUs — staging / copy path (`ncclReshard`)

| File | Role |
|---|---|
| `staging_prepare.cc` | Staging transfer-descriptor builders. |
| `staging_buffer.cc` | Staging-buffer / bucket-pool lifecycle; reads the `NCCL_RESHARD_STAGING_*` env knobs. |
| `staging_profile.cc` | Staging profiling scopes. |

## Device TUs

| File | Role |
|---|---|
| `reshard_user_window.cu` | Window API entries + `ReshardKernelUserWindow` (RING) + `DirectReshardKernelUserWindow` (DIRECT). Algorithm dispatch + cross-dim transpose decision + kernel launch. |
| `reshard_split_user_window.cu` | Dual-DevComm RING kernel (`reshardKernelUserWindowSplit`) + launch for the split-comm path. |
| `reshard_staging.cu` | `ncclReshard` staging/copy-path entry + dispatch. |
| `staging_kernel.cu` | Staging CUDA kernels. |
