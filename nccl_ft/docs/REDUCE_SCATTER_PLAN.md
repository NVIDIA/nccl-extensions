# Reduce-Scatter AllReduce Plan — Closing the Algorithm Gap

## Why

After multi-CTA (round 4), AllReduce sits at ~17 GB/s vs NCCL ~95 (≈18%), while
AllGather reaches ~128 GB/s (≈half-to-par with NCCL). The remaining AllReduce
gap is **algorithmic**, not a kernel-occupancy problem:

The current AllReduce is **gather-then-reduce** — every rank gathers all
`nRanks` chunks and reduces locally. Per-rank traffic is **O(nRanks)**:

| phase | per-rank bytes (nRanks = N) |
|---|---|
| gather (NVLink) | ~2·(N−1)·count |
| local reduce (HBM) | ~N·count read |

NCCL's bandwidth-optimal **reduce-scatter + allgather** is **O(1)** per element:
~2·(N−1)/N·count ≈ 2·count total, independent of N. At N=8 that's the ~7×
data-volume disadvantage we still see. This plan adopts the same algorithm.

## Algorithm (direct, single-node NVLink)

Because LSA is single-node with full NVSwitch connectivity, use **direct**
(not ring) reduce-scatter and allgather — lowest latency, NVSwitch carries the
all-to-all bandwidth.

Partition `count` into `nActive` shards of `shardSize` each (one shard per
**active** rank — see FT semantics below). Let shard *k* be owned by the *k*-th
active rank.

1. **Scatter-reduce.** Every rank pushes its `send[shard k]` slice into owner
   *k*'s scratch (a per-contributor slot). Each owner *k* then reduces all
   arrived contributions for shard *k* → the fully-reduced shard *k*, held
   locally. Per-rank push ≈ count·(N−1)/N; per-rank reduce ≈ count/N·N = count
   read (vs N·count before — an N× cut).
2. **Allgather.** Each owner *k* broadcasts its reduced shard *k* to every peer.
   This is exactly the existing multi-CTA AllGather primitive applied to shards.
   Per-rank push ≈ count·(N−1)/N.

Total per-rank ≈ 2·count NVLink — the same order as one AllGather of `count`,
which already runs at ~128 GB/s. **Expectation: AllReduce approaches AllGather
bandwidth (~100+ GB/s), i.e. near NCCL.**

### Worked walkthrough (8 ranks, count = 800)

`shardSize = ceil(800/8) = 100` (rounded up to a 16-byte multiple). Shard `s`
covers elements `[100s, 100s+100)` and is owned by rank `s` (all alive). The
in-place AllReduce tensor `t` (from `pg.empty`) is symmetric, so it is both
input and final output; `d_recv` is reused as the reduce-scatter scratch
(`nActive` contributor slots). One cooperative kernel, two phases:

```
# Phase 1 — scatter-reduce (each rank reduces ONLY its own shard)
for each owner k (= shard k):
    push my t[100k : 100k+100]  ->  rank k's rs_scratch[slot = me]
grid.sync; threadfence_system; grid.sync
signal round-1 flag[me] on each owner k      # "me delivered my shard-k slice"
poll  round-1 flag[0..nActive) on MYSELF     # I own my shard; wait for all contributors
reduce: t[myShard*100 + i] = Σ_r rs_scratch[r][i]   over arrived r   (vectorized int4)

# Phase 2 — allgather of the reduced shards (existing AllGather primitive, on shards)
grid.sync; threadfence_system; grid.sync
for each peer p:
    push my reduced t[myShard slice]  ->  peer p's t[myShard slice]
signal round-2 flag[myShard] on each peer
poll  round-2 flag[0..nActive) on MYSELF     # wait for every shard to arrive
# t now holds the full reduced result, in place — no extra copy
```

Reused from the round-4 multi-CTA kernels: the ready-barrier, the
`push → grid.sync → threadfence_system → grid.sync → flag(release)` ordering,
the timeout/mask logic, and the vectorized reduce. New: the active-set→shard
mapping (`activeRanks[]` from `active_mask_in`; my shard = my index in it), the
two-round flag window, the `d_recv` slot layout, and the small-count /
`nActive == 1` fallback. In-place works because Phase 1 finishes reading `t`
(grid.sync) before the reduce writes `t[myShard]`, and Phase 2 writes only
*other* shards into `t` after the global flag sync — no read/write hazard.

## Fault-tolerance semantics — the key decision

This changes the *partial-result* behavior under failure, and needs sign-off.

- **Shards are partitioned over the active set** (`active_mask_in`), so a stable
  survivor set is fully correct and bandwidth-optimal: after a rank is dropped
  from the mask, subsequent AllReduces re-partition over the survivors.
- **Gather-then-reduce (today):** if rank *k* dies mid-collective, the result is
  *uniformly degraded* — every element is the sum over survivors. Complete
  coverage, one missing term.
- **Reduce-scatter (proposed):** if owner *k* dies mid-collective, **shard *k*
  is missing for everyone** (k was its sole reducer/distributor), while other
  shards are correctly summed over survivors. This is a *non-uniform* partial
  result.

Both cases raise `FT_TIMEOUT` and update `active_mask_out`, so per the existing
contract ("timeout ⇒ partial data; caller consults the mask") the *failing*
iteration is already caller-handled. The change is only in *what the partial
data looks like* during the failure-transition iteration; steady-state over any
fixed survivor set is identical and correct.

### Worked failure example (8 ranks, count = 800, rank 3 dies mid-collective)

**Gather-then-reduce (today):** every rank gathers all 8 full chunks and
reduces element-wise; the reduction simply drops rank 3 from `s_result_mask`:

```
result[i] = Σ_{r ∈ {0,1,2,4,5,6,7}} send_r[i]      for ALL i ∈ [0,800)
```

Every element is present — uniformly the sum over survivors. The partial result
is itself a *valid allreduce over the 7 surviving ranks* (one fewer addend
everywhere).

**Reduce-scatter (proposed):** rank 3 is the **sole owner/reducer of shard 3**
(`[300,400)`):

```
elements [0,300) ∪ [400,800):  Σ over survivors    ✓  (rank 3's contribution just missing)
elements [300,400):            never reduced or broadcast → stale/garbage  ✗  (a "hole")
```

Phase-2's flag poll for shard 3 times out → `FT_TIMEOUT`, `active_mask_out`
drops rank 3. Most of the output is sum-over-survivors, but the dead owner's
shard is a hole no rank ever produces.

| (failure-transition iteration) | gather-then-reduce | reduce-scatter |
|---|---|---|
| coverage | all elements present | one shard is garbage (a hole) |
| meaning of the partial | clean allreduce over survivors | corrupt in the dead owner's region |
| caller can salvage it? | maybe (valid survivor-sum) | no — must discard + retry |

**Crucial mitigant — steady state is identical and correct.** Once the
framework drops rank 3 from the mask (`ftHandleSetMask`), the next launch reads
`active_mask_in` with `nActive = 7`, re-partitions into 7 shards each owned by a
*living* rank, and produces a complete, correct, fast allreduce over the 7
survivors. Only the single transition iteration has the hole; every iteration
before and after is clean — exactly the FT model's intent (the iteration during
which a rank dies is "lost"/flagged; you continue on the survivor set).

Secondary notes:
- A rank dying *between* Phase 1 and Phase 2 (after reducing its shard but
  before broadcasting it) produces the same hole — Phase-2's flag poll catches
  it, so both phases are timeout-protected.
- Masks can momentarily **diverge across ranks** during a partial failure (rank
  A saw rank 3's contribution, rank B did not). This already happens with
  gather-then-reduce — each rank independently decides who it heard from — so it
  is not new; the framework reconciles via the mask.

**Decision (chosen):** reduce-scatter becomes the **large-count multi-CTA**
AllReduce path (replacing the gather-then-reduce multi-CTA from round 4);
gather-then-reduce stays on the **small-count single-CTA** path (latency-bound;
the algorithm gap is irrelevant there, and it preserves uniform-degradation
semantics for tiny control-plane allreduces). The size-dependent failure
semantics are documented and considered acceptable: the crossover is the same
`FT_MULTICTA_THRESHOLD_BYTES` used for AllGather, so above it callers already
expect the multi-CTA path and must consult the mask/error on a partial result.

## Buffer layout (reuse existing symmetric scratch)

- `d_recv` (already `nRanks·paddedStride`, symmetric) is large enough to serve
  both phases:
  - **Scatter-reduce scratch:** owner *k*'s `d_recv` holds `nActive` contributor
    slots of `shardStride` bytes (`shardStride = round_up(shardSize·sizeof(T),16)`);
    contributor *r* writes slot *r*. Total ≤ nActive·shardStride ≈ count — fits.
  - **Allgather destination:** reduced shards are pushed into peers' `d_recv`
    laid out as `nActive` shard slots, then the kernel writes the assembled
    result to `output` (the in-place tensor). Two sub-regions of `d_recv`, or
    two passes, avoid aliasing the scatter-reduce scratch.
- **Two flag rounds** are needed (scatter-reduce completion, then allgather
  completion). Either a `2·nRanks` flag window or reset-and-reuse between phases
  via a `grid.sync()` (no host involvement). Reusing needs care: the round-2
  poll must not observe round-1 flags — simplest is a `2·nRanks` flag window
  (cheap) indexed `[phase·nRanks + rank]`.
- `output` 16-byte alignment already handled by the vectorized reduction path.

## Fused cooperative kernel phases

```
P0  own-ready (gtid 0);  P1 ready-poll -> d_can_push    grid.sync
// --- scatter-reduce ---
P2  for owner k (active): push send[shard k] -> rank k's rs_scratch[myslot]
    grid.sync; __threadfence_system() [all]; grid.sync
P3  signal rs-flags to owners (one thread/owner)
P4  poll rs-flags for MY shard's contributions; update mask/d_error   grid.sync
P5  reduce MY shard: reduced[i] = sum over arrived r of rs_scratch[r][i]  (vectorized)
    grid.sync; __threadfence_system() [all]; grid.sync
// --- allgather of reduced shards ---
P6  push my reduced shard -> every peer's ag slot[k]
    grid.sync; __threadfence_system() [all]; grid.sync
P7  signal ag-flags;  P8 poll ag-flags; assemble output + finalize mask   grid.sync
P9  reset ready (gtid 0)
```

Producer ordering (data-before-flag) uses the same
`push → grid.sync → threadfence_system(all) → grid.sync → flag(release)`
sequence proven out in the round-4 multi-CTA kernels.

## Shard sizing / alignment

- `shardSize = ceil(count / nActive)` rounded up to a multiple of `16/sizeof(T)`
  so every shard base is 16-byte aligned (keeps the vectorized `int4` push and
  reduce). The last shard is shorter (handle the `count % shardSize` tail).
- Guard tiny cases: if `nActive == 1` or `count < nActive·(16/sizeof(T))`,
  fall back to the existing gather-then-reduce path (reduce-scatter has no
  benefit and the shard bookkeeping gets fiddly).

## Files touched

- `ft_collective.cu`: new `FTAllReduceRSKernelMultiCTA<T>` (cooperative);
  shard-partition helpers; dispatch in the AllReduce launch paths
  (`ftAllReduce` window + `ftAllReduceLaunch` ctx) — reduce-scatter when LSA &&
  count ≥ multicta_threshold && nActive ≥ 2, else the round-4 multi-CTA gather-
  then-reduce (mid sizes) / single-CTA (small).
- Possibly a 2·nRanks flag window — either grow the existing `d_flags`
  allocation in the ctx/Python paths or allocate a dedicated reduce-scatter flag
  buffer in the handle.
- `ft_collective.h`: handle field if a dedicated flag/scratch buffer is added.
- **Python:** none expected (C entry points unchanged) unless the flag window
  must grow — then bump `_d_flags` size in `ft_process_group.py` (one line).
- GIN untouched.

## Validation

1. **Numeric correctness** vs `torch.distributed` NCCL AllReduce across the
   `--pow2` sweep, all dtypes (fp32/fp16/bf16), including non-power-of-2 counts
   to exercise the shard tail.
2. **Fault injection** (`collectives_bench`) at counts ≥ threshold — confirm:
   (a) steady-state survivor set gives the correct sum-over-survivors;
   (b) a kill mid-iteration raises `FT_TIMEOUT` with the dead rank cleared in
   `active_mask_out` and **no hang** (all local blocks still reach every
   `grid.sync()`); (c) the next iteration over the reduced mask is correct.
3. **Perf:** `--pow2` sweep into `results_round5.csv`; expect AllReduce large-
   count to jump from ~17 GB/s toward AllGather-class (~100+ GB/s).

## Risks / open questions

- **Failure semantics change** (above) — the main review item. Confirm the
  large-count = reduce-scatter / small-count = gather-then-reduce split is
  acceptable, or whether reduce-scatter should be opt-in.
- **Two-phase flag bookkeeping** — the `2·nRanks` flag window and the
  reset-free phase separation via `grid.sync()` need care to avoid round-1/round-2
  flag aliasing.
- **Shard tail + alignment** for non-power-of-2 counts and for fp16/bf16
  (VEC = 8) — the last shard and the per-shard stride must stay 16-byte aligned.
- **nActive changes between calls** (mask updated) re-partitions shards — verify
  the layout math reads `active_mask_in` consistently and that a single-survivor
  mask falls back cleanly.
```
