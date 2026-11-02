# Multi-CTA Plan — Scaling FT LSA Kernels Past One SM

## Why

The LSA AllGather/AllReduce kernels launch **one CTA** (`FT_DEVICE_CTA_COUNT = 1`,
512 threads, one SM). At 8 GPUs that caps us at the throughput of a single SM's
NVLink/HBM traffic:

| | ft (round 3) | NCCL | gap |
|---|---|---|---|
| AllGather 4M | ~28 GB/s | 238 GB/s | ~8.5× |
| AllReduce 4M | ~2.4 GB/s | 95 GB/s | ~40× |

Small-count latency is already acceptable (AG p50 ~100 µs, AR ~91 µs), so this
plan **must not regress small counts** — it adds a multi-CTA path for large
counts and keeps the existing single-CTA kernel for small ones.

## Key observation: the data loops are already grid-ready

The copy/push loops (`ftCopy`) and the reduction loop already stride by the
**global** thread id / thread count:

```
int tid      = threadIdx.x + blockIdx.x * blockDim.x;
int nthreads = blockDim.x * gridDim.x;
```

So with `gridDim > 1` the data movement auto-partitions across all CTAs with
**no change**. The only thing that assumes a single block is the **control
protocol** — the readiness barrier, completion-flag signaling/polling, and the
shared-memory state (`s_can_push`, `s_result_mask`) coordinated with
`__syncthreads()`. Multi-CTA work is therefore almost entirely about making the
control protocol grid-wide, not about the data path.

## Design: cooperative-groups single kernel (recommended)

Launch the kernel cooperatively (`cudaLaunchCooperativeKernel`) with
`gridDim > 1` and replace the block-scope `__syncthreads()` phase boundaries
with **grid-wide** `cg::this_grid().sync()`. Cross-block state that was in shared
memory moves to a small global scratch buffer.

### Phase structure (AllGather)

```
cg::grid_group grid = cg::this_grid();
int gtid = global thread id;  int gnthreads = grid size in threads;

// P0  own readiness         (gtid == 0)        store d_ready = 1
// P1  ready-poll            (gtid < nRanks)     poll peer gtid's d_ready
//                                               -> d_can_push[gtid]   (GLOBAL)
grid.sync();                                  // all blocks see d_can_push
// P2  self-copy + push      (all blocks, global stride)   ftCopy(...)
grid.sync();                                  // all stores issued (device scope)
__threadfence_system();                       // ALL threads -> system scope
grid.sync();                                  // all threads passed the fence
// P3  flag-signal           (gtid < nRanks && d_can_push)  store peer flag (release)
// P4  flag-poll             (gtid < nRanks && d_can_push)  poll own flag (acquire)
//                                               -> active_mask_out[gtid], d_error
grid.sync();                                  // propagate arrived data to whole grid
// P5  reset readiness       (gtid == 0)        store d_ready = 0
```

AllReduce is identical through P4, then:

```
grid.sync();                                  // peer data now visible grid-wide
// P6  reduction             (all blocks, global stride)
//     sum over r where active_mask_out[r] is set; vectorized int4 path
```

### Cross-rank producer ordering (the correctness crux)

Peer *P* must see **all** of our pushed bytes before it observes our flag.
Because the push is partitioned across many threads in many blocks, a single
flag-signaling thread's fence is not enough. The safe sequence:

```
push  →  grid.sync()  →  __threadfence_system() [all threads]  →  grid.sync()  →  flag store (release)
```

After the second `grid.sync()` every thread has flushed its own NVLink stores to
**system scope**, so the subsequent release flag store is correctly ordered
after all data. Peer's acquire load on the flag then guarantees data-before-flag.

For the AllReduce reduction, the flag-poll's acquire establishes that arrived
peers' data is visible to the polling thread; the following `grid.sync()`
propagates that to every thread before the reduction reads our local recv
buffer (device scope is sufficient here — the reduction reads our own GPU).

### Global scratch (replaces shared cross-block state)

| was (shared, per-block) | becomes |
|---|---|
| `s_can_push[]` | `d_can_push[]` — new global buffer, `nRanks` bytes, in the handle |
| `s_result_mask[]` (AllReduce) | reuse `active_mask_out[]` (already global; holds arrived ranks after P4) |
| `s_active_mask[]` | keep per-block shared (read-only input, cheap to reload) |

### Launch configuration

- Compute the max cooperative grid once at handle create:
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpSM, kernel, threads, 0)`
  × `multiProcessorCount`. Store as `h->max_coop_blocks`.
- Per launch: `gridDim = min(max_coop_blocks, ceil(count / (threads * EL  )))`
  so we never launch more blocks than there is work. Large counts use the full
  grid (~hundreds of blocks on H100 → saturates NVLink/HBM).
- Launch via `cudaLaunchCooperativeKernel` (args passed as a `void*[]`).

### Small-vs-large dispatch (protects small-count latency)

```
if (count * sizeof(elem) >= FT_MULTICTA_THRESHOLD_BYTES)   // default e.g. 256 KiB, env-tunable
    cooperative multi-CTA launch
else
    existing single-CTA fused kernel  (unchanged)
```

The cooperative path adds a few `grid.sync()`s and a different launch API — both
negligible at large sizes, but not worth paying at small sizes where we already
match the target.

## Alternative: three-kernel pipeline (simpler, considered, not recommended)

Split into 3 ordinary kernels on the same stream — (1) ready-poll → `d_can_push`,
(2) push (many CTAs), (3) flag-signal/poll/reduce — relying on **stream
serialization** for the grid-wide barriers instead of `grid.sync()`. Pros: no
cooperative-launch grid-size limit, simpler kernels. Cons: 3 launch latencies
(~15 µs) per collective and more global round-trips. Acceptable only on the
large-count path. Falls out naturally as the fallback if cooperative launch is
unsupported on the device.

## Files / functions touched

- `ft_collective.cu`
  - Add `#include <cooperative_groups.h>`.
  - New `FTAllGatherKernelMultiCTA` / `FTAllReduceKernelMultiCTA` (or a
    `template<bool MultiCTA>` switch over the existing bodies — the data loops
    are shared; only the control phases differ).
  - `d_can_push` + `max_coop_blocks` added to `FTCollectiveHandle`; allocate
    `d_can_push` and query occupancy in `ftHandleCreate`.
  - `ftAllGather` / `ftAllReduce` (window API) and the `*Launch` (ctx API):
    size-threshold dispatch + `cudaLaunchCooperativeKernel`.
- `ft_collective.h` — handle struct fields only (no public API change).
- **No Python changes** — the C entry points are unchanged; dispatch is internal.
- GIN path untouched.

## Validation plan

1. Correctness: existing demos + a numeric check vs `torch.distributed` NCCL
   allgather/allreduce across the `--pow2` sweep (add a one-shot allclose).
2. Fault tolerance must still hold: run `collectives_bench` fault injection on
   the multi-CTA path — a dead peer must still produce `active_mask_out` with
   that peer cleared and `FT_TIMEOUT`, no hang (all *local* blocks always reach
   every `grid.sync()`, so peer death only trips the flag-poll timeout).
3. Perf: `--pow2` sweep into `results_round4.csv`; expect AllGather to climb from
   ~28 GB/s toward >150 GB/s and AllReduce well past the current ~2.4 GB/s once
   both gather and reduction span all SMs.

## Risks / open questions

- **grid.sync() ordering vs `__threadfence_system()`** — the double-sync + all-
  threads fence above is the conservative, provably-correct sequence; worth a
  focused review since it's the heart of the change.
- **Cooperative launch limits** — grid must be co-resident; cap to occupancy and
  keep a single-CTA fallback if `cudaLaunchCooperativeKernel` is unsupported.
- **Threshold tuning** — `FT_MULTICTA_THRESHOLD_BYTES` should be picked from the
  measured crossover where multi-CTA overtakes single-CTA (sweep will show it).
- **`devComm` in cooperative kernels** — LSA pointer access is per-thread and
  unaffected, but confirm no NCCL device-API call assumes a single block.

## Appendix: earlier multi-CTA attempt & obstacles (merged from MCTA_CHALLENGES.md)

Before the cooperative-groups design above, an earlier MCTA variant was tried and
**reverted** after measuring ~3× worse latency than single-CTA at large counts.
Its obstacles directly motivated the choices above; recorded here so they aren't
re-attempted:

- **Don't give each CTA its own inter-rank barrier.** The first attempt indexed a
  `ncclLsaBarrierSession` by `blockIdx.x`, creating 64 independent arrive+wait
  cycles (64× the NVLink sync traffic). Reducing to a single barrier at index 0
  removed the traffic but gave no speedup — the barrier was never the bottleneck.
  → This design **barriers only on CTA 0** (P3), which is why.
- **Avoid a global-flag thundering herd.** Having all 64×512 threads spin on one
  device-global `g_go` int (acquire loads) saturated L2 and stalled NVLink writes;
  restricting the spin to `tid==0` per CTA + caching `can_push` in shared memory
  did not help either. → Here the ready-poll result lives in `d_can_push[]` and is
  gated by a single `grid.sync()`, not a spun-on global flag.
- **The real large-count bottleneck was protocol latency, not data bandwidth.**
  Single-CTA at 1M floats (4 MB/rank) was ~2476 µs — ~60× the ~40 µs NVLink
  bandwidth bound — i.e. dominated by serial per-peer readywin/flagswin round
  trips, not by having too few threads pushing data. Adding CTAs for data movement
  alone therefore couldn't help; the win came from **parallelizing the data path
  *and* keeping the control protocol single-owner + grid-synced** (this plan),
  plus vectorized `int4` copies and the reduce-scatter AllReduce.
- **NCCL's multicast (STMC) path is unavailable to FT** because the destination
  set changes at runtime (failed ranks excluded). Vectorized writes and multi-CTA
  data movement were the adoptable pieces — both are in the shipped design.

The reverted attempt's "paths forward" (vectorized writes, two-kernel split,
persistent-thread polling) informed the final approach; vectorized writes and the
cooperative single-kernel grid-sync were adopted, the two-kernel split kept only as
the documented fallback (see "Alternative: three-kernel pipeline").
```
