# NCCL M2N — Release Notes

NCCL M2N is an experimental, NCCL-based library for cross-group GPU data
movement. This preview scopes the surface to the **reshard** functionality: redistribute a
global tensor between two disjoint groups of GPU processes (the source group
holds one sharding / replication layout, the destination group holds
another). Future releases may extend the same library to other cross-group
transfer primitives under the same `NCCL M2N` umbrella. The reshard
functionality is built on NCCL's user-window API (`ncclWindow_t` +
`ncclMemAlloc`) and on the NCCL Device API (LSA load/store and GIN
put/signal), so transfers are zero-copy, one-sided, and have no host
involvement on the critical path.

Install artifacts are shared library `libnccl_m2n.so` and public header
`include/nccl_m2n.h`. v0.2 keeps `ncclReshardWithWindow` for the user-window
operation, adds `ncclReshard` for copy/staging transfers, moves generic
concepts to library-scope `ncclM2n*` names, and adds an explicit handle for
lifecycle-managed runtime state.

This is the first official release of NCCL M2N. The earlier v0.1 and v0.2 drops
were experimental and are not supported upgrade sources.

## v0.2

API-breaking cleanup for NCCL M2N naming, descriptors, and runtime ownership.
`NCCL_M2N_API_VERSION` is bumped from `1u` to `2u` for this API transition.
Existing binaries must be rebuilt against the updated header.

- **NCCL compatibility:** NCCL 2.30.5 or newer is required to build and use
  this release.
- **Release version macros:** `nccl_m2n.h` publishes `NCCL_M2N_MAJOR`,
  `NCCL_M2N_MINOR`, `NCCL_M2N_PATCH` and the packed `NCCL_M2N_VERSION_CODE`,
  matching the convention `nccl_ep.h` already uses. These carry the library
  release version and are distinct from `NCCL_M2N_API_VERSION`, which guards
  the public struct ABI. No build system consumes them yet.
- **Exported symbols:** the five public entry points carry default-visibility
  annotations, so a build that compiles hidden-by-default exports exactly those
  five. The Make build does not enable `-fvisibility=hidden`, so it additionally
  exports internal symbols; the CMake build compiles hidden-by-default and
  therefore exports the public API and nothing else.
- **Standalone CMake build:** from `nccl_m2n/`, configure with
  `cmake -B build -DNCCL_HOME=/path/to/nccl/build`. `NCCL_HOME` is required and
  validated at configure time. The build produces shared and static libraries
  and an installable package configuration, so consumers can use
  `find_package(NCCLM2N)` and link `NCCL::m2n` or `NCCL::m2n_static`.
  `NCCL_M2N_BUILD_BENCH` and `NCCL_M2N_BUILD_TESTS` enable the optional
  benchmark and test builds. There is deliberately no repository-root
  `CMakeLists.txt`.
- **Benchmark argument parsing:** All three public benchmark drivers —
  `reshard_bench`, `reshard_batch_bench_user_window`, and
  `reshard_model_bench` — now use a shared strict argument parser.
  Unknown options and unknown enum values are fatal. Previously, a stray or
  obsolete flag was silently ignored, so existing wrapper scripts that pass
  one may now fail.
- **Qwen3-32B benchmark configs:** Adds the Qwen3-32B model configuration and
  DP1/TP1-to-DP1/TP1, DP1/TP2-to-DP1/TP2, and DP1/TP4-to-DP1/TP4 system
  configurations.
- **Exported symbols:** the eight public entry points carry default-visibility
  annotations. The Make build does not enable `-fvisibility=hidden`, so it
  additionally exports internal symbols; the CMake build compiles
  hidden-by-default and therefore exports the public API and nothing else.
- **Grouped reshard submission:** `ncclM2nGroupStart()` records
  `ncclReshard` and `ncclReshardWithWindow` calls on the calling host thread,
  and the outermost `ncclM2nGroupEnd()` submits them by execution context while
  preserving original entry indices in deferred errors. Nested groups flatten
  into the outer group; `ncclM2nGroupAbort()` discards all active nested levels
  without submitting their recorded calls. Group submission stops at the first
  error and bounds the indexed diagnostic to the thread-local error buffer.
- **Public C API cleanup:**
  - `ncclMesh_t` carries topology only (`dims[]`, `startRank`).
  - Tensor placement moves from `mesh.placement[]` to
    `ncclDistTensor_t::placements[]`, so one mesh topology can describe
    multiple tensor layouts.
  - Placement helpers are `NCCL_RESHARD_REPLICATE` and
    `NCCL_RESHARD_SHARD(d)`.
  - `NCCL_RESHARD_MESH_NDIMS` is the mesh-axis array size, and
    `NCCL_RESHARD_MAX_TENSOR_DIMS` is the tensor-rank cap.
- **Configuration and lifecycle:**
  - `ncclM2nConfig_t` is the library configuration descriptor.
  - `NCCL_M2N_CONFIG_INITIALIZER` / `NCCL_M2N_CONFIG_UNDEF_INT` initialize
    and mark defaulted config fields.
  - `ncclM2nInit(&m2nHandle, config|NULL)` / `ncclM2nFinalize(m2nHandle)`
    manage explicit handles. The compatibility-only `ncclM2nHandleCreate` /
    `ncclM2nHandleDestroy` symbols are removed.
  - `ncclReshardWithWindow(m2nHandle, ...)` and
    `ncclReshard(m2nHandle, ...)` are the explicit-handle reshard operations.
    The window operation replaces the legacy explicit-handle entry point.
    Passing NULL lazily uses an internal default handle, released by
    `ncclM2nFinalize(NULL)`.
  - Internal runtime state is resolved by the first explicit init or lazy
    default-handle use in an epoch and released after the last handle is
    finalized. Finalized handles must not be reused.
  - Reshard operations enqueue asynchronous CUDA work. Callers must complete
    work submitted with a handle before finalizing it and must complete all M2N
    work in an epoch before finalizing the last explicit or default handle.
- **Copy/staging API:**
  - `ncclReshard` adds internally managed copy/staging transfers alongside the
    caller-provided user-window path.
  - `NCCL_RESHARD_COPY_ALGORITHM` selects the staging copy algorithm. `DIRECT`
    remains the default; `PACKWINDOW` packs each destination's bytes contiguously
    with CUDA copy engines, transfers them through the hierarchical user-window
    path, then unpacks them. Other values are rejected rather than silently ignored.
  - Every communicator rank participates and provides both descriptors. Active
    source/destination ranks require non-NULL buffers; inactive sides still
    provide local-shape metadata.
  - Both mesh intervals are checked against the communicator before rank
    planning, and descriptor/rank/count arithmetic rejects overflow.
  - Staging channel count, channel size, and chunk size are environment-tunable
    but must be rank-uniform. The default cached pool is 256 MiB per
    communicator (4 channels times 64 MiB); the configured 1 MiB chunk is kept
    when safe and otherwise reduced uniformly from shared geometry.
  - `NCCL_RESHARD_STAGING_BUCKETS` enables a bounded best-fit staging-buffer
    pool, and `NCCL_RESHARD_STAGING_WATERMARK_BYTES` sets the minimum per-comm
    staging allocation.
  - `ncclReshard` supports split communicators on the `PACKWINDOW`, `RING`,
    `NODE_AWARE` path. The library collectively forms a FULL communicator for
    source injection and a RAIL communicator for destination forwarding; the
    caller still invokes the operation collectively on the parent communicator
    with the same tensor descriptors.
  - Kernel launch is asynchronous. Callers complete the stream before
    finalizing M2N or releasing communicators, buffers, and streams used by the
    operation.
- **Runtime hardening:**
  - Integer and size environment values require a complete in-range decimal
    parse. Positive-only knobs reject zero and negative values; trailing
    characters and whitespace are rejected.
    Invalid tuning values are ignored, except invalid or non-positive stream
    pool sizes disable the pool and oversized values are capped.
  - On the RING path, `NCCL_RESHARD_DST_DOMAIN_SIZE` values larger than the
    NCCL DevComm LSA team size are rejected with `ncclInvalidArgument`; the
    diagnostic names both values. This rejects only configurations that could
    not previously have worked correctly, not valid setups. Non-RING paths and
    values within the LSA team size are unaffected.
  - The internal non-blocking stream pool preserves both readiness and
    completion ordering with the caller's default stream, including every
    error return after stream setup.
  - GIN signal counts reject overflow, and signal IDs are relative to the
    source mesh so meshes with a non-zero `startRank` use the allocated range.
- **Error reporting:**
  - `ncclM2nGetLastError()` exposes thread-local detail for the most recent M2N
    failure, and Python exceptions include that detail when available.
  - Calling a public reshard API from inside a CUDA graph capture now fails with
    `ncclInvalidUsage` and an explanatory message.
- **Documentation:**
  - Public header comments are now Doxygen-readable and document the handle,
    descriptor, stream, and window contracts in the installed header.

## v0.1

Initial preview of NCCL M2N — reshard functionality.

- **Public C API** in `src/nccl_m2n.h`:
  - `ncclReshardWithWindow` — single-shot reshard against a caller-registered
    `ncclWindow_t`.
  - `ncclMesh_t` descriptor — bundles each side's 2-D rank topology
    and per-mesh-axis placement (`NCCL_RESHARD_REPLICATE` or
    `NCCL_RESHARD_SHARD(d)`).
  - `ncclDistTensor_t` descriptor — bundles per-rank tile, dtype, and a
    pointer to the side's `ncclMesh_t`, modeled after PyTorch
    DTensor / JAX `NamedSharding`.
  - `ncclM2nConfig_t` with `NCCL_M2N_CONFIG_INITIALIZER`; currently exposes
    `maxCta`.
  - Lifecycle helpers `ncclM2nInit(config|NULL)` / `ncclM2nFinalize`.
    Algorithm, load-balance mode, stream-pool size,
    logging, and chunk sizing are env-driven.
- **Resharding kernels:**
  - **Ring** — hierarchical ring + intra-NVL fan-out via the user window.
  - **Direct** — per-rank GIN puts.
  - Transparent cross-dim transpose for 2-D and 3-D layouts when cross-dim
    sharding would otherwise create small innermost transfers.
- **Window contract:** participating pointers must be inside the supplied
  symmetric-memory window; source and destination pointers on the same rank
  must share a single window offset. Matching non-zero offsets are supported.
- **Default stream support:** callers may pass `NULL`, `cudaStreamLegacy`, or
  `cudaStreamPerThread`; the library uses an internal non-blocking stream pool
  and records a back-edge event so later default-stream work observes the
  result.
- **Benchmarks** under `benchmarks/`:
  - `reshard_bench` — single-layer bench (canonical worked example).
  - `reshard_batch_bench_user_window` — batched sequential vs concurrent
    comms sweep.
  - `reshard_model_bench` — config-driven model transfer bench.
- **Tests** under `tests/`:
  - `basic_api_test_mpi` and `basic_api_test_local` — C-level functional
    matrix mirroring an external pytest reference suite. Groups:
    `full_replication`, `full_sharding`, `2d_placement`, `uneven_ratio`,
    `tensor_size_sensitivity`, `nd_tensors`, `cross_dim_regression`, plus 1-D
    analogues and FP8 coverage where the NCCL enum is available.
  - `--list` / `--min-world` / `--max-world` introspection for binning a
    CI run into rank tiers.
- **Runtime environment variables:**
  - `NCCL_RESHARD_LOG_LEVEL` — `NONE` / `WARN` / `INFO` / `DEBUG` / `TRACE`.
  - `NCCL_RESHARD_ALGORITHM` — `AUTO`, `RING`, or `DIRECT`.
  - `NCCL_RESHARD_LB_MODE` — `UNIFORM` or `NODE_AWARE`.
  - `NCCL_RESHARD_NUM_CTAS` — directly overrides the CTA count resolved from
    `config.maxCta`.
  - `NCCL_RESHARD_USE_INTERNAL_STREAMS` — boolean; unset or `1` caches one
    internal stream per observed `(comm, device)` pair until runtime
    finalization, `0` keeps work on caller streams with ordered DevComm reuse.
  - `NCCL_RESHARD_CHUNK_SIZE` — override the default 256 KB byte-level
    chunk size in the RING prepare path.
  - `NCCL_RESHARD_SPLIT_COMM` — boolean override for split-communicator
    dispatch on the supported `ncclReshard` path; the adaptive default enables
    it whenever the selected copy path supports it.
  - `NCCL_RESHARD_SPLIT_AUTO_PARENT_THRESHOLD` — parent-communicator size
    threshold for adaptive single-replica injection; the default is `200`
    ranks.
  - `NCCL_RESHARD_SPLIT_SINGLE_REP_INJECT` — boolean override for adaptive
    single-replica injection; unset selects it above the adaptive threshold.
  - `NCCL_RESHARD_SPLIT_KERNEL_TRACE` — enables per-CTA split-kernel tracing.

## Breaking Changes

This is the first official release; the previous experimental drops are not
supported upgrade sources. Two runtime tuning variables changed with no
compatibility shim:

- `NCCL_RESHARD_MAX_CTA` is replaced by `NCCL_RESHARD_NUM_CTAS`, which directly
  overrides the CTA count resolved from `config.maxCta`.
- `NCCL_RESHARD_STREAM_POOL_SIZE` is removed and no longer parsed; setting it
  has no effect. Use the boolean `NCCL_RESHARD_USE_INTERNAL_STREAMS` instead:
  `0` maps the former `=0` behavior to ordered caller-stream execution, while
  unset or `1` uses internal streams. There is no internal-stream count cap.

## Known Limitations

This preview is experimental.

| Limitation | Description |
|---|---|
| **Limited QA coverage**           | Functional matrix is the C-level basic_api suite × {RING, DIRECT} × dtype mix. Large multi-node coverage is still cluster-limited and workload-specific. |
| **Tensor rank ≤ 3**               | `NCCL_RESHARD_MAX_TENSOR_DIMS = 3`. 4-D and higher are not supported. |
| **Both-REPLICATE meshes unsupported** | `placement = {REPLICATE, REPLICATE}` falls into a degenerate prepare-time branch that the test suite does not exercise. Encode full replication as a 1-shard layout (mesh axis of size 1). |
| **Single-offset window contract** | Per-rank source and destination pointers must have the same offset within the registered window when both are present; asymmetric source/destination offsets are rejected. |
| **Cross-rank offset symmetry trusted** | The API validates local offset consistency but does not currently perform a cross-rank collective check that every rank uses the same offset. |
| **Algorithm auto-select unimplemented** | `NCCL_RESHARD_ALGORITHM=AUTO` currently aliases to `RING`; no input/topology-aware algorithm picker in this build. |
| **Single in-flight reshard per `(comm, effective stream)`** | The internal DevComm/window/transpose caches are designed for sequential use on a comm. Use separate communicators for concurrent transfers, as in `reshard_batch_bench_user_window --num-comms`. |
| **Not thread-safe — process-wide single-thread access** | The init-time globals and the internal caches (DevComm cache, window cache, stream pool) are process-wide shared state. Caller is responsible for serializing every lifecycle call and every `ncclReshardWithWindow` / `ncclReshard` call on the host side — including calls on different `ncclComm_t` handles. Device-side concurrency (issuing successive reshards on separate CUDA streams from a single host thread) is supported. |
| **Caller-synchronized finalization** | `ncclM2nFinalize` does not synchronize caller streams. Callers must complete M2N work before finalizing the associated handle and before releasing communicators, windows, streams, or buffers used by that work. |
| **Static mesh-size caps**         | Compile-time array bounds in `src/reshard_limits.h` cap supported mesh sizes: `MAX_SOURCES = 16`, `MAX_TARGETS = 64`, `MAX_LOCAL_FOLLOWERS = 128` (RING); `MAX_DIRECT_SOURCES = 32`, `MAX_DIRECT_TARGETS = 64` (DIRECT). Larger meshes require recompiling. |
| **Copy/staging peer caps** | The initial DIRECT copy/staging path uses `MAX_SOURCES = 16`, `MAX_TARGETS = 64`, and at most `STAGING_MAX_CHANNELS = 32`. The rank-uniform preflight rejects plans that exceed these static arrays. |
| **Single-shot public API**        | The public API exposes one window or staging collective per call (`ncclReshardWithWindow` / `ncclReshard`). Batched/concurrent behavior is built by callers with multiple descriptors/comms, not by a persistent public API. |

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [GPU-Initiated Networking Paper](https://arxiv.org/abs/2511.15076)
- [NCCL M2N README](README.md)

## License

The NCCL contrib drop inherits the parent `nccl/nccl` license. Third-party
dependencies are listed in [`ThirdPartyNotices.txt`](ThirdPartyNotices.txt).
