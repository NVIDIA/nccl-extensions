# Fault-Tolerant Collectives — Implementation Overview

> **Intended audience:** This document is structured for conversion into a
> presentation.  Each top-level section maps to a slide group; sub-sections
> map to individual slides.

---

## 1. Motivation

### The Problem with Standard NCCL Collectives

Standard NCCL collectives require **all ranks to participate**.  If any single
rank crashes, hangs, or is preempted:

- Every surviving rank blocks indefinitely waiting for the missing peer.
- After a watchdog timeout, the entire process group is torn down.
- Training must be restarted from the last checkpoint — or abandoned.

This all-or-nothing guarantee is catastrophic in large-scale distributed
training where transient GPU or node failures are common.

### The Goal

Provide **fault-tolerant AllGather and AllReduce** primitives that:

1. **Never deadlock** — every participating rank makes forward progress
   regardless of what other ranks do.
2. **Return a result mask** — callers learn which ranks contributed data and
   which timed out.
3. **Self-heal** — after a timeout, the surviving ranks automatically exclude
   the dead rank from future collectives without restarting.
4. **Integrate with PyTorch** — expose fault-tolerance through the standard
   `torch.distributed` ProcessGroup API so existing training code needs
   minimal changes.

---

## 2. What We Built

### 2.1 C++ / CUDA Kernel Library (`ft_handle/`)

A shared library (`libft_collective.so`) implementing fault-tolerant
device-side kernels and a host-side coordination API.

**Key components:**

| Component | Description |
|---|---|
| `FTCollectiveHandle` | Per-communicator state: mask buffers, readywin, error status |
| `FTAllGatherCtx` | Context holding symmetric send/recv/flags buffers for AllGather |
| `FTAllReduceCtx` | Context holding symmetric send/recv/flags buffers for AllReduce |
| `FTAllGatherKernel` | LSA device kernel — NVLink symmetric direct-write AllGather |
| `FTAllGatherGINKernel` | GIN device kernel — RDMA-based AllGather fallback |
| `FTAllReduceKernel` | LSA device kernel — NVLink symmetric AllReduce with reduction |
| `FTAllReduceGINKernel` | GIN device kernel — RDMA-based AllReduce fallback |

**Two kernel backends, selected automatically at handle creation:**

- **LSA (Local Symmetric Access)** — preferred when `nLsaTeams >= 1`.  Uses
  NVLink symmetric memory (`ncclGetLsaPointer`) for direct GPU-to-GPU writes
  and a per-rank `readywin` barrier implemented with system-scope atomics.
- **GIN (GPU-Initiated Networking)** — fallback when LSA is unavailable (e.g.
  multi-node InfiniBand).  Uses `ncclGin` / RDMA for remote writes and
  `ncclGinBarrierSession` for the barrier phase.

**Kernel execution model (LSA AllGather):**

1. Rank signals its own `readywin` flag (system-scope atomic store).
2. Each rank polls peer `readywin` flags up to `timeout_cycles`; marks
   unreachable peers as inactive.
3. Each active peer DMA-writes its send data into the local rank's receive
   slot using `ncclGetLsaPointer`.
4. Each rank polls a per-peer completion flag.  Missing flags → timeout.
5. Thread 0 writes the output mask and, if any peer timed out, sets
   `d_error = FT_TIMEOUT` via a system-scope atomic release.

All launches are **asynchronous** — the kernel is queued on a CUDA stream and
the host call returns immediately.

### 2.2 Python ctypes Bindings (`ft_collective/ft_wrapper.py`)

Pure-Python wrappers that load `libft_collective.so` and `libnccl.so` at
runtime — no compilation needed on the Python side.

**Classes exposed:**

| Class | Purpose |
|---|---|
| `FTLibrary` | Wraps all `ftHandle*`, `ftAllGather*`, `ftAllReduce*`, `ftAllGather`, `ftAllReduce` C functions |
| `NCCLBaseLibrary` | Wraps `ncclCommInitRank`, `ncclMemAlloc`, `ncclCommWindowRegister`, etc. |
| `CUDARuntime` | Wraps `cudaMemset`, `cudaMemcpy`, `cudaStreamSynchronize`, `cudaStreamCreate` |

**Library search:**

- `libft_collective.so` — `FT_COLLECTIVE_LIB` env var, then `LD_LIBRARY_PATH`.
- `libnccl.so` — `NCCL_HOME/lib`, then `LD_LIBRARY_PATH`.

### 2.3 Python Package (`ft_collective/`)

The `ft_collective` Python package bundles all bindings and the torch backend:

```
ft_collective/
  __init__.py          # Public API surface
  _version.py          # Version string
  ft_wrapper.py        # ctypes bindings (FTLibrary, NCCLBaseLibrary, CUDARuntime)
  ft_process_group.py  # torch.distributed backend (FTProcessGroup)
```

Importing `ft_collective` automatically registers the `"ft_nccl"` backend
with `torch.distributed`.

### 2.4 `torch.distributed` Backend (`ft_process_group.py`)

`FTProcessGroup` — a `ProcessGroupNCCL` subclass that transparently intercepts
`allgather` and `allreduce` and routes them through the FT kernel when
eligible, or falls back to standard NCCL with a `UserWarning`.

### 2.5 Demo Applications

| Script | Description |
|---|---|
| `demo_ft_collective.py` | Low-level ctypes API demo — single-node AllGather and AllReduce |
| `demo_window_api.py` | Window-based API demo using nccl4py for symmetric memory management |
| `demo_fault_tolerance.py` | `torch.distributed` demo comparing FT backend vs. standard NCCL under rank failure |

---

## 3. The Fault-Tolerant Handle API

### 3.1 Core Concept

The `FTCollectiveHandle` is the central coordination object.  It is bound to a
single NCCL communicator and tracks:

- **Active mask** (input) — which ranks should participate in the next launch.
- **Result mask** (output) — which ranks actually delivered data.
- **Error status** — `FT_OK` or `FT_TIMEOUT`, written by the kernel into
  mapped pinned memory (zero-copy host read after stream synchronization).
- **Readywin** (LSA only) — a symmetric `uint64_t` per rank used as a
  liveness barrier.

The error status is **sticky**: once set to `FT_TIMEOUT` it remains until the
caller explicitly calls `ftHandleClearError()`.  The kernel never clears it.

### 3.2 Handle Lifecycle

```
ftHandleCreate(comm, nRanks, initial_mask, &handle)
    │
    ├── Queries ncclCommProperties
    │       nLsaTeams >= 1 → FT_KERNEL_LSA
    │       ginType != NONE → FT_KERNEL_GIN
    │
    ├── Allocates h_mask_in, d_mask_in, h_mask_out, d_mask_out
    ├── Allocates h_error / d_error (mapped pinned memory)
    └── Allocates d_ready / win_ready (LSA only)

[use handle for collectives]

ftHandleDestroy(handle)
```

### 3.3 AllGather Context Lifecycle

```
ftAllGatherCreate(handle, count, &ctx)
    │
    └── Allocates symmetric: d_send, d_recv, [d_flags (LSA only)]
        Registers windows:   win_send, win_recv, [win_flags (LSA only)]

ftAllGatherSetData(ctx, host_data)   // H2D copy into d_send
ftAllGatherLaunch(ctx, stream, timeout_us)  // ASYNC — queues kernel

cudaStreamSynchronize(stream)        // REQUIRED before reading results

ftHandleGetError(handle)             // FT_OK or FT_TIMEOUT
ftHandleGetResultMask(handle)        // bool[nRanks] — d→h copy

// On timeout, self-heal for next iteration:
ftHandleSetMask(handle, result_mask) // exclude dead ranks
ftHandleClearError(handle)           // reset sticky flag

ftAllGatherGetResult(ctx, host_out)  // D2H copy of gathered data

ftAllGatherDestroy(ctx)
ftHandleDestroy(handle)
```

### 3.4 Window-Based API

For callers that manage their own symmetric memory (e.g. via nccl4py), the
library exposes `ftAllGather` and `ftAllReduce` which accept pre-registered
`ncclWindow_t` handles directly:

```c
ncclResult_t ftAllGather(
    FTCollectiveHandle* h,
    ncclWindow_t win_send, size_t sendoffset,
    ncclWindow_t win_recv, size_t recvoffset,
    ncclWindow_t win_flags,      // ignored for GIN kernel
    size_t count,
    cudaStream_t stream,
    uint64_t timeout_us);
```

The handle owns: devComm, readywin, mask buffers, error status.
The caller owns: send/recv/flags windows — and must zero them before each call.

### 3.5 Key API Rules

| Rule | Detail |
|---|---|
| All launches are **async** | Call `cudaStreamSynchronize(stream)` before reading results or error status |
| Error status is **sticky** | Call `ftHandleClearError()` to reset; only safe when no kernel is in flight |
| `ncclMemAlloc` is **collective** | All ranks must call simultaneously with the same size |
| `ncclCommWindowRegister` is **collective** | All ranks must call simultaneously |
| `ftHandleSetMask` excludes dead ranks | Call after a timeout with the result mask to skip dead peers next iteration |
| Own rank is always set in result mask | A rank always considers itself ready; partial masks only clear *remote* peers |

---

## 4. Mapping the FT API to the PyTorch Backend

### 4.1 Architecture Overview

```
torch.distributed.init_process_group("ft_nccl", ...)
         │
         ▼
  FTProcessGroup (subclass of ProcessGroupNCCL)
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  _comm_world (ProcessGroupNCCL internal)            │
  │    └── Standard NCCL: broadcast, reduce,            │
  │        reduce_scatter, alltoall, send, recv, ...    │
  │        (each emits a UserWarning — not FT)          │
  │                                                     │
  │  _comm_ft  (bootstrapped separately)                │
  │    └── FTCollectiveHandle                           │
  │          └── allgather / allreduce                  │
  │                (FT kernel, fault-tolerant)          │
  └─────────────────────────────────────────────────────┘
```

### 4.2 Two Communicators, One Process Group

`FTProcessGroup` maintains two independent NCCL communicators:

**`_comm_world`** (managed by `ProcessGroupNCCL` parent)
- Created by calling `super().__init__(store, rank, world_size, nccl_opts)`.
- Used for all non-FT collectives (broadcast, reduce, etc.).
- Its watchdog timeout is set to `FT_timeout_s + 5 s` so NCCL does not abort
  while the FT kernel is exhausting its own timeout budget.

**`_comm_ft`** (bootstrapped manually)
- Created by rank 0 calling `ncclGetUniqueId()` and sharing the UID through
  the torch Store under the key `"ft_nccl_uid_ft"`.
- Passed to `ftHandleCreate()`, which attaches LSA or GIN resources.
- Owns all symmetric memory (`ncclMemAlloc`) and windows used by FT kernels.

The two communicators are kept separate to avoid resource conflicts between
device-side FT kernel state and host-side NCCL path bookkeeping.

### 4.3 Symmetric Tensor Allocation

The FT kernel requires **symmetric virtual addresses** — the same virtual
address on every rank — which is a property guaranteed by `ncclMemAlloc` but
not by `torch.zeros(..., device="cuda")`.

`FTProcessGroup.empty(*shape)` bridges this gap:

```
ncclMemAlloc(size_bytes)          →  raw device pointer (same VA on every rank)
ncclCommWindowRegister(comm, ptr) →  ncclWindow_t handle
_SymTensor(ptr, numel, device_id) →  DLPack producer
torch.from_dlpack(sym_tensor)     →  standard torch.Tensor
```

`_SymTensor` implements `__dlpack__` / `__dlpack_device__` by constructing a
`_DLManagedTensor` ctypes struct around the raw pointer.  `torch.from_dlpack`
consumes this capsule to produce a zero-copy tensor view.

The resulting `torch.Tensor` has a `data_ptr()` that lives in `_windows`, the
registry used by `_is_ft_eligible()` to decide which path to take.

### 4.4 FT Eligibility Check

On every `allgather` / `allreduce` call, eligibility checks that the tensor is
contiguous, fits its registered symmetric window and scratch limit, and has a
supported dtype (`float32`, `float16`, or `bfloat16`). AllGather also requires
the byte count to be divisible by four. A registered subregion carries its byte
offset into the parent window.

If any condition fails, the call falls back to `super().allgather()` /
`super().allreduce()` and emits a `UserWarning`.

### 4.5 FTWork and Stream Ordering

`FTWork` implements `torch.distributed.Work` following the same **stream-ordering
semantics** as standard `ProcessGroupNCCL` — it never blocks the host CPU:

- **`wait()`** — GPU-side ordering only.  Immediately after the FT kernel is
  enqueued, an event is recorded on the FT stream.  `wait()` calls
  `cudaStreamWaitEvent` to insert a GPU-side dependency into the calling
  (compute) stream and returns immediately on the host.  Future GPU work on
  the compute stream is automatically serialized after the FT collective.
  **No host sync, no FT error handling.**

- **`pg.get_result_mask()`** synchronizes the FT stream, clears the sticky error,
  and returns this rank's observed responder mask. It does not update membership.
  The process group disables local mask adaptation; `ft_converge()` agrees and
  installs membership across ranks.

`is_completed()` queries the recorded event. `is_success()` also reports CUDA
completion, including valid partial results after a peer timeout; inspect the
process-group status and result mask to determine peer participation. Each work
keeps its process group alive until the work is released.

### 4.6 Additional ProcessGroup API

Two lightweight methods expose the sticky error flag without forcing a stream sync:

| Method | Description |
|---|---|
| `pg.get_error()` | Read current sticky FT status (`FT_OK` / `FT_TIMEOUT`) directly from mapped pinned memory — no stream sync |
| `pg.clear_error()` | Reset sticky flag to `FT_OK` — only safe when no FT kernel is in flight |

These are useful for callers that want to poll the flag independently of
`get_result_mask()` or perform a manual reset at a known safe point.

### 4.7 Pre-Allocated Scratch Buffers

On initialization, `FTProcessGroup` pre-allocates two symmetric buffers
shared across all FT collectives:

| Buffer | Size | Purpose |
|---|---|---|
| `_d_recv` | `world_size × max_count × 4 B` | Symmetric gather scratch |
| `_d_flags` | `world_size × 8 B` | Per-rank completion flags (LSA only) |

The flags are zeroed asynchronously on `_stream` before each launch. Present
ranks overwrite their receive slots, so `_d_recv` is not cleared. The result
mask identifies valid slots.

### 4.8 Registration and `get_ft_process_group()`

Importing `ft_collective` triggers:

```python
dist.Backend.register_backend(
    "ft_nccl",
    _ft_nccl_pg_factory,
    devices=["cuda"],
)
```

After that, `dist.init_process_group("ft_nccl", ...)` creates an
`FTProcessGroup` exactly as it would any built-in backend.

`dist.distributed_c10d._get_default_group()` returns the C-level
`ProcessGroup` wrapper, which does not expose Python-only methods like
`empty()` or `get_result_mask()`.  The factory stores each new
`FTProcessGroup` in a module-level registry keyed by rank, which can be
retrieved with:

```python
from ft_collective import get_ft_process_group

pg = get_ft_process_group()   # returns the Python FTProcessGroup for this rank
```

---

## 5. Demonstrating Fault Tolerance

### 5.1 Demo Script: `demo_fault_tolerance.py`

The demo spawns one process per GPU rank using `multiprocessing.get_context("spawn")`.
Each rank is pinned to its own GPU by setting `CUDA_VISIBLE_DEVICES=<rank>`.

**Startup sequence per rank:**

1. Create a `TCPStore` (rank 0 is the master; all others connect).
2. Call `dist.init_process_group(backend, store=store, rank=rank, world_size=N)`.
3. Allocate send tensor (`pg.empty(count)` for `ft_nccl`; `torch.zeros` for `nccl`).
4. Print own PID so the user can target a specific rank with `kill <PID>`.

**Main loop (once per second):**

- Fill send tensor with `float(rank)`.
- Call `dist.all_gather(output_list, send_tensor, group=pg)`.
- On the FT backend: ranks call `pg.ft_converge()` to agree membership and
  `pg.get_result_mask()` to inspect each collective's responders. Rank 0 prints which
  ranks responded and their values, flagging any timeout.
- On the NCCL backend: print received values; catch `DistBackendError` on
  watchdog timeout and exit.

### 5.2 What Happens When You Kill a Rank

**`ft_nccl` backend (fault-tolerant):**

```
  [iter    1]  OK       active=[0,1,2,3]  values=['0','1','2','3']
  [iter    2]  OK       active=[0,1,2,3]  values=['0','1','2','3']
  # kill rank 2 here
  [iter    3]  TIMEOUT  active=[0,1,3]    values=['0','1','—','3']
  [iter    4]  OK       active=[0,1,3]    values=['0','1','—','3']
  [iter    5]  OK       active=[0,1,3]    values=['0','1','—','3']
```

- The first iteration after the kill times out (configurable via `--timeout-s`).
- `pg.ft_converge()` excludes the failed rank from the agreed active set.
  `pg.get_result_mask()` reports each rank's responders and clears the error.
- All subsequent iterations complete immediately — the dead rank is simply
  skipped, no restart required.

**`nccl` backend (standard):**

```
  [iter    1]  OK       active=[0,1,2,3]  values=['0','1','2','3']
  [iter    2]  OK       active=[0,1,2,3]  values=['0','1','2','3']
  # kill rank 2 here
  [iter   17]  rank 0  NCCL ERROR — watchdog timeout after 15 s:
                 <ncclRemoteError / ncclSystemError>
               The process group is now unusable.  Exiting.
```

- After the watchdog timeout (default `timeout_s + 5` seconds), all remaining
  ranks receive a `DistBackendError` and the process group is permanently
  torn down.

### 5.3 Key Flags

| Flag | Default | Description |
|---|---|---|
| `--backend` | `ft_nccl` | `ft_nccl` or `nccl` |
| `--nranks` | all visible GPUs | Number of ranks / GPUs |
| `--count` | `16` | Float32 elements per rank per AllGather |
| `--timeout-s` | `10` | FT kernel timeout (or NCCL watchdog base) |
| `--interval` | `1.0` | Seconds between iterations |

The `FT_TIMEOUT_US` environment variable (set internally from `--timeout-s`)
controls the FT kernel timeout without requiring changes to the factory
signature imposed by `torch.distributed`.

---

## 6. Summary

| Layer | What Was Built |
|---|---|
| **CUDA kernels** | LSA and GIN AllGather / AllReduce with per-rank timeout, readywin barrier, sticky error flag |
| **C host API** | `FTCollectiveHandle`, context create/destroy/launch, window-based API |
| **Python ctypes** | `FTLibrary`, `NCCLBaseLibrary`, `CUDARuntime` — no compilation needed from Python |
| **torch backend** | `FTProcessGroup(ProcessGroupNCCL)` — FT path for allgather/allreduce, NCCL fallback for everything else |
| **Membership** | `ft_converge()` agrees and installs the surviving set; the C handle also offers local self-adaptation for direct callers |
| **Demo** | Side-by-side comparison of FT vs. standard NCCL under live rank failure |

## 7. Known Limitations & Test Backlog

Remaining behaviors that need a recovery policy, measurement, or dedicated tests.

- **Barrier agreement at the failure boundary.** The collective membership
  barrier (`_ft_barrier_collective`) exchanges and intersects survivor masks over
  a *fixed* number of rounds. A failure or asymmetric timeout in the **final**
  round can leave survivors with different masks, because no later round
  propagates that observation — the fixed-round protocol does not prove agreement
  under arbitrary failure timing (see `FT_BARRIER_COLLECTIVE.md`). *Needed:* a
  defined recovery policy for ranks that exit with divergent masks, and an
  adversarial final-round-failure test.

- **Reduce-scatter pushes to expected-but-dead peers.** The scatter phase writes
  to peers in the expected active set even when the preceding readiness poll
  timed out. AllGather gates payload writes on its readiness result. A
  reduce-scatter-specific fault-injection test is needed before changing this
  CUDA path.

- **Error-clear ordering.** `check_and_clear_error()` and `clear_error()` require
  the caller to synchronize queued work before clearing the shared status.
  `work.wait()` only inserts a stream dependency. Use `get_error()` to poll
  pending work; its result may still change.

- **Cooperative-grid cap is shared across kernels.** `max_coop_blocks` is a single
  cap = the minimum occupancy across *all* cooperative kernels, so the
  lowest-occupancy kernel throttles the grid for every collective. Any
  performance-motivated CUDA simplification here (and to the GIN signal arrays)
  should be preceded by per-kernel occupancy and register-spill measurements to
  avoid a silent regression.
