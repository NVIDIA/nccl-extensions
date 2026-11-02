# Collective-mode FT membership barrier — belief-exchange consensus

Design + correctness notes for the store-free membership barrier used by
`ft_converge()` under `FT_BARRIER_MODE=collective`. This is the **default** barrier (no TCPStore on the membership path).

## Problem: the epoch-boundary race

`ft_converge()` aims to make every survivor install the **same** surviving set
(*Agreement*). Both the store barrier and the *result-mask-only* collective
barrier decide membership from a **per-rank timeout**: rank *A* declares peer *C*
alive/dead based on whether *C*'s signal arrived before *A*'s own deadline. A
**slow-but-alive** *C* whose signal lands near the timeout is seen by survivors
with a later deadline and timed-out by those with an earlier one → **split
brain**: *A* keeps *C*, *B* drops it → the next collective deadlocks (*A* waits on
*C*, *B* does not).

This window is narrow in a healthy run (µs posts vs a multi-second timeout) but
opens up exactly when it hurts: **gray failures** (a degrading GPU/link that
straddles the timeout), **correlated delays during a real failure** (survivors
desynchronize, widening the skew), **aggressive/low `FT_TIMEOUT_US`**, and
**scale**. And a single-phase barrier cannot claim an Agreement property at all —
the counterexample exists.

An earlier fix shrank the poll set between rounds, which desynchronized
readiness generations and collapsed membership to singletons. The current
version keeps a fixed poll set.

## The algorithm: belief exchange + intersection, fixed poll set, fixed rounds

Each round every survivor **broadcasts its current belief** of the surviving set
(an `nRanks` mask) to every peer via one FT all-to-all, reads back the
responders' beliefs, and **intersects**:

```
belief'[r]    = responded(r) AND (every responder saw r)     for r != self
belief'[self] = True
```

"In the set iff *everyone who answered* saw it." This is what makes survivors
agree at the boundary: if *A* saw slow peer *C* but *B* did not, *B*'s belief
(excluding *C*) reaches *A* next round, so *A* intersects it away and both drop
*C*. This propagation works while the responder set remains stable long enough
for the changed belief to reach the other participants.

Two invariants preserve readiness-generation **lockstep** (what the store barrier
gives for free, and what the old fixpoint version lost):

- **Fixed poll set** — the handle mask stays the *entry* believed-alive set for
  the whole barrier; only the exchanged *belief* narrows, never the set of peers
  polled. Every rank polls the same peers each round.
- **Fixed round count** — no per-rank early break, so all ranks run the same
  number of collectives and exit at the same generation. Belief convergence
  happens *within* the fixed rounds, not by early termination (an early break
  would reintroduce generation drift).

### Implementation (`ft_process_group.py`)

The barrier buffers are registered during process-group initialization, while
all ranks are present. Each round uses the FT all-to-all path to exchange masks,
then reads the responder masks and intersects them. The implementation is
[`_ft_barrier_collective()`](../ft_handle/python/ft_collective/ft_process_group.py).

## Correctness properties (for the paper)

- **Agreement limit.** Once the same peers respond to each participant for
  enough rounds, exchanged beliefs can converge. A failure or asymmetric
  timeout in the last round can still leave different masks because no later
  round propagates that observation. The fixed-round protocol does not prove
  agreement under arbitrary failure timing.
- **Validity.** A crashed peer's signal reaches no one → excluded by all.
- **Generation progression.** Participants that enter with aligned counters
  advance by the same number of collectives. This does not repair a counter
  that was already behind; `ft_rejoin()` has a separate generation exchange.
- **Convergence limit.** More rounds can propagate earlier disagreement; no
  finite round count covers a new failure in its final round.
- **Liveness caveat (safety-over-liveness).** A slow-but-alive peer suspected by
  *any* responder is excluded from `M` (a false positive); it keeps only itself
  and re-converges to a singleton on its next collective (survivors are already
  consistent), then rejoins on recovery. Bounded, timeout-based detection cannot
  avoid this.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `FT_BARRIER_MODE` | `collective` | **default**; set `store` for the TCPStore barrier |
| `FT_BARRIER_ROUNDS` | `3` | fixed rounds; more rounds allow earlier disagreement to propagate |
| `FT_TIMEOUT_US` | 10000000 | per-round timeout; lower = faster failover but wider boundary window |

Cost: `FT_BARRIER_ROUNDS` all-to-alls of an `nRanks × nRanks` mask + one
device→host mask read per round. Fine for small `nRanks` (LSA, ≤ 8); the payload
is `O(nRanks²)`, so bit-pack the mask for very large groups.

## What this removes vs the store barrier

- **No epoch keys** → no wrap-around false positives → **no epoch-key GC**.
- **No TCPStore** on the membership path.

The fixed poll set addresses generation drift. Agreement at a failure boundary
still needs adversarial timing tests and a defined recovery policy for ranks
that leave with different masks.

## Proposed test suite

Agreement is the property; the tests inject the timing situations that break a
result-mask-only barrier and assert survivors still install one set. Cross-rank
comparison uses the TCPStore (off-GPU, valid mid-failure): each survivor posts its
mask and every believed-alive peer's mask must be identical to it.

### 1. Agreement under clean kill — IMPLEMENTED (`test_ft_converge.py`)

Healthy loop asserts all ranks agree on the full set; `--kill` a rank and assert
survivors converge to the identical surviving set (prints `AGREE FAIL <<< SPLIT
BRAIN` on any disagreement).

```
python test_ft_converge.py --nranks 8              # healthy: AGREE ok
python test_ft_converge.py --nranks 8 --kill --timeout-s 5   # kill -> survivors agree
```

### 2. Boundary straggler — PROPOSED (`--straggler`)

Inject a *slow-but-alive* rank so its signal straddles the timeout: one rank
`time.sleep(~timeout * f)` (f≈0.9–1.1) *before* entering `ft_converge()` on a
chosen iteration, creating deliberate skew. Assert: either everyone keeps it or
everyone drops it — never a split. This is the case a result-mask-only barrier
fails and belief-intersection converges. Sketch:

```python
if args.straggler and it == args.straggler_iter and rank == args.straggler_rank:
    time.sleep(args.timeout_s * args.straggler_frac)   # land near the deadline
ok, mask = _agree_check(pg, store, it, args.timeout_s)   # must AGREE regardless
```

Sweep `--straggler-frac` across 0.8..1.2 to walk the signal through the boundary.

### 3. Last-round death — PROPOSED

Kill a rank while the barrier is mid-flight so death lands in the final round.
Compare survivor masks to detect a split, then test the application's recovery
policy. This probes the agreement limit described above.

### 4. Cascading / correlated failures — PROPOSED

Kill two ranks staggered across rounds (e.g. one before round 1, one before the
last round). Compare masks and measure whether additional rounds help when
failures happen early enough to propagate.

### 5. Generation-lockstep invariant — PROPOSED

The collapse regression guard: repeatedly kill/replace ranks and assert the
surviving set never collapses to singletons and never grows (monotone until a
rejoin). Optionally read the device generation counter before/after to assert all
survivors advanced by the same amount (no drift).

### 6. Interleave barrier with data collectives — PROPOSED

`ft_converge()` between `allreduce`/`all_gatherv`/`reduce_scatterv` calls (they
share `win_ready`/`d_gen`); assert the data results stay correct and the barrier
still agrees — checks the shared readiness/generation state isn't corrupted.

---

Tests 2–6 are kill/timing-injection harnesses that must run on the multi-GPU box;
each reuses the `_agree_check` store-based agreement oracle from test 1.
