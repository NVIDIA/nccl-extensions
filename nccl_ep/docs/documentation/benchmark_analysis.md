# Reading ep_bench Results

`ep_bench` reports **High Throughput** bandwidth in three columns per direction —
`total_bw`, `nvl_bw`, `scale_out_bw` — because a token in HT mode can travel over two
different links, and one number cannot say whether a run saturated either of them. This
guide explains what each column counts, how to compare them against hardware limits, and
which invariants must hold in a healthy run.

Scope: **HT only.** Low Latency uses its own byte accounting (`calculateLowLatencyBytes`)
with different definitions, so an LL `nvl_bw` is not comparable to an HT one.

## The topology model

Ranks are grouped into nodes of `lsa_team_size` ranks. A rank reaches another node with a
**single put**, addressed to the one rank there that shares its own index within the node.
The ranks sharing an index across nodes form a **rail team**, and the receiving member is
the **landing rank**: it takes the token off the NIC on behalf of its whole node, then fans
it out over NVLink to whichever local experts want it.

Two ranks per node, two nodes, as used in the walkthrough below:

```
            NODE 0                          NODE 1
     ┌───────────────┐                ┌───────────────┐
     │  r0  ─────────┼══ rail 0 ══════┼─────────  r2  │
     │   ·  NVLink   │                │   NVLink  ·   │
     │  r1  ─────────┼══ rail 1 ══════┼─────────  r3  │
     └───────────────┘                └───────────────┘
      rail teams: {r0,r2}   {r1,r3}
```

Landing ranks follow from the rail teams: r2's puts land on r0, r3's puts land on r1, r0's
puts land on r2, r1's puts land on r3. A token never changes rails, and it crosses at most
two hops — one put to the destination node, then one local fan-out.

## The six counters

| counter | granularity | measures |
|---|---|---|
| `total_send` | per destination **rank** | the full logical payload this rank dispatches — every delivery, whichever path carries it: an **LSA** store to its own experts, an **LSA/NVLink** write to a peer on this node, or an **RDMA** put to another node |
| `nvl_send` | per destination **rank** | every NVLink write this rank issues: local destinations of its own tokens, plus the fan-out it performs as landing rank. These are **LSA** operations, and that includes the degenerate case of a rank storing to its own experts — a plain local store, no wire transfer, still counted here |
| `scale_out_send` | per destination **node** | one entry per destination node, **the local node included** unless `--ignore-local-traffic` — a token with destinations on both nodes contributes 2, one with destinations only on this node contributes 1 |
| `total_recv` | per **consumed** token | tokens this rank's experts consume, arriving over LSA or RDMA |
| `nvl_recv` | per consumed token | the subset of `total_recv` that arrived over LSA/NVLink |
| `scale_out_recv` | per **landed** token | tokens landed from a rail team member on another node, plus this rank's own tokens that **this node** wanted unless `--ignore-local-traffic`, whether consumed or only forwarded |

> **Transport terms.** **LSA** (load-store accessible) is the intra-node path — NVLink
> writes to peers on the same node, plus the degenerate case of a rank storing to its own
> experts, which moves no data over a wire but is still counted. **RDMA** (GIN,
> GPU-initiated networking) is the inter-node path over the NIC. `nvl_*` counts LSA traffic;
> `scale_out_*` counts RDMA puts, plus one entry for this rank's own node unless
> `--ignore-local-traffic` drops it.

> The six bandwidth columns are printed as **rates in GB/s**, not byte counts. The
> underlying byte totals appear separately in the `Logical payload bytes` line, in MB per
> rank; internally the counters are kept as both bytes and token counts.

A token delivered to a **peer** on this node is LSA traffic that never reaches the NIC, yet
with the flag off it still adds one entry to `scale_out_send` — the local node counts as a
destination node like any other. Only with `--ignore-local-traffic` does the column narrow to
puts that actually cross a NIC.

The recv counters answer two different questions, which is the most common source of
confusion. `total_recv` is *"what did my experts consume"*. `scale_out_recv` is *"what did I
receive on my rail"* — with `--ignore-local-traffic` that narrows to NIC arrivals only, but
with the flag off it also includes this rank's own tokens that any expert on this node
wanted, which never touch the NIC. A landing rank routinely receives tokens its own experts
never use.

## `scale_out_send` serves two different questions

The flag changes what the column *means*, and which one you want depends on who is reading:

- **Flag off** — counts the local node too, so the column is a clean per-node delivery count
  at one granularity. This answers *how effectively does this configuration dispatch data*,
  counting everything this rail position issues. That's the **benchmark / evaluation** view.
- **Flag on** — the self entry is dropped, leaving only what crosses the NIC. This is the
  number to compare against RDMA line rate, and it answers *how well are we driving the
  network*. That's the **network engineering** view.

Neither is more correct; they measure different things. Quoting a flag-off `scale_out_bw` as
an RDMA figure is the mistake to avoid.

## Worked example

Six tokens on the 2×2 topology above, processed in order. Every path is one of the shapes HT
can produce:

| # | token | path |
|---|---|---|
| A | r1 → {r0} | `r1 → NVLink → r0` |
| B | r2 → {r0} | `r2 → NIC → r0` — r0 is both landing rank and consumer |
| C | r3 → {r0} | `r3 → NIC → r1 → NVLink → r0` |
| D | r2 → {r1} | `r2 → NIC → r0 → NVLink → r1` |
| E | r0 → {r0} | `r0 → local store` |
| F | r0 → {r1, r3} | `r0 → NVLink → r1` and `r0 → NIC → r2 → NVLink → r3` |

Counters as **cumulative running totals**, all four ranks shown at every step. Bold =
changed by that token, `·` = zero.

> **This table is `--ignore-local-traffic` ON**, so every `so_*` cell is a NIC transfer and
> nothing else. It is also the variant that does not depend on how the local term is counted,
> which makes it the stable reference. With the flag **OFF** — the default — three local-node
> entries appear (A at r1, E at r0, and F's local half at r0), taking the cluster row to
> `so_send` 7 / `so_recv` 7 and per-rank `so_send` 3/1/2/1, `so_recv` 4/2/1/0. The other four
> columns are identical either way; the flag touches only `so_send` and `so_recv`.

| # | rank | `t_send` | `nvl_send` | `so_send` | `t_recv` | `nvl_recv` | `so_recv` |
|---|---|---|---|---|---|---|---|
| **A** | r0 | · | · | · | **1** | **1** | · |
| | r1 | **1** | **1** | · | · | · | · |
| | r2 | · | · | · | · | · | · |
| | r3 | · | · | · | · | · | · |
| **B** | r0 | · | · | · | **2** | 1 | **1** |
| | r1 | 1 | 1 | · | · | · | · |
| | r2 | **1** | · | **1** | · | · | · |
| | r3 | · | · | · | · | · | · |
| **C** | r0 | · | · | · | **3** | **2** | 1 |
| | r1 | 1 | **2** | · | · | · | **1** |
| | r2 | 1 | · | 1 | · | · | · |
| | r3 | **1** | · | **1** | · | · | · |
| **D** | r0 | · | **1** | · | 3 | 2 | **2** |
| | r1 | 1 | 2 | · | **1** | **1** | 1 |
| | r2 | **2** | · | **2** | · | · | · |
| | r3 | 1 | · | 1 | · | · | · |
| **E** | r0 | **1** | **2** | · | **4** | **3** | 2 |
| | r1 | 1 | 2 | · | 1 | 1 | 1 |
| | r2 | 2 | · | 2 | · | · | · |
| | r3 | 1 | · | 1 | · | · | · |
| **F** | r0 | **3** | **3** | **1** | 4 | 3 | 2 |
| | r1 | 1 | 2 | · | **2** | **2** | 1 |
| | r2 | 2 | **1** | 2 | · | · | **1** |
| | r3 | 1 | · | 1 | **1** | **1** | · |
| | **cluster** | **7** | **6** | **4** | **7** | **6** | **4** |

The cluster row is where the invariants land, and each total is checkable by hand: **7
deliveries** (1+1+1+1+1+2), **6 NVLink writes** (A `r1→r0`, C `r1→r0`, D `r0→r1`, E `r0`
self store, F `r0→r1`, F `r2→r3`), and **4 scale-out entries** — one per put: B, C, D, and F's
put to node 1.

Four things worth reading off the individual ranks:

**A produces no scale-out at all.** r1 → r0 stays inside node 0, so no put is issued and both
`so_*` columns are untouched. Only B, C, D and F put anything on a rail. Drop the flag and A
gains one `so_send` at r1 plus one `so_recv` there, since r1 is the landing rank for its own
node — the local node then counts as a destination node like any other.

**r1 ends with `t_send = 1` but `nvl_send = 2`.** It originated one token; the second NVLink
write is C's forward, issued on behalf of r3 — a rank on the other node. A rank's NVLink
send load is not proportional to its own dispatch.

**r2 ends with `t_recv = ·` yet is far from idle.** It dispatched two tokens, landed F's put
(`so_recv = 1`), and issued an NVLink write forwarding that token to r3 (`nvl_send = 1`).
Its own experts consumed nothing.

**Per-rank `so_send` and `so_recv` need not match** — r0 ends 1 against 2. Only the cluster
sums balance.

Without `--ignore-local-traffic` — the default — three local-node entries appear: A (at r1),
E (at r0) and F's local half (at r0). r0 gains 2 in each scale-out column and r1 gains 1,
giving per-rank `so_send` of 3/1/2/1 and `so_recv` of 4/2/1/0, and cluster totals of 7 and 7.
Every other cell is flag-independent, and `nvl_*` never moves.

## Why no counter is the difference of two others

It is tempting to write `nvl_send = total_send - scale_out_send`. That is wrong in both
directions:

- `total_send` counts destination **ranks**, `scale_out_send` counts destination **nodes**.
  For token F the subtraction gives 2 - 1 = 1 against a true `nvl_send` of 1 — right by luck
  with the flag on, and 2 - 2 = 0 with it off. It **understates** whenever a token reaches
  several ranks on this node and **overstates** whenever a token reaches several ranks on one
  remote node, since that node's fan-out belongs to its landing rank, not the sender.
- With the flag off it also subtracts the local-node entry, which is no RDMA put at all.

The recv side is worse: `total_recv - scale_out_recv` mixes *consumed* with *landed* tokens.
For a rank that lands much and consumes little — r2 above, with `t_recv = ·` and
`so_recv = 1` — the result goes **negative**.

## Invariants

Three equalities must hold when summed across ranks. Check them before reading any bandwidth
number:

| invariant | why |
|---|---|
| `total_send` = `total_recv` | every delivery is one send and one receive |
| `nvl_send` = `nvl_recv` | NVLink never leaves a node, so every byte written is read by a peer |
| `scale_out_send` = `scale_out_recv` | every put is issued by one rank and landed by exactly one other |

These are structural properties of the counters rather than a live health check: both sides
come from the same routing replay inside `calculateHighThroughputBytes`, so on a uniform
topology they hold by construction. State them anyway, for two reasons. They break when the
launch is not uniform — a node holding fewer ranks than the others has no member of some rail
teams, so puts get issued that nobody lands, and every column is then wrong by 15-25%. Equal
ranks per node is an unchecked precondition of these metrics. And they are what a future edit
to either loop must preserve: an earlier revision derived `nvl_send` as
`total_send - scale_out_send`, which understated true NVLink traffic by a factor of
`N - (N-1)/L` for `N` nodes of `L` ranks — 1.875x at 2x8, 3.625x at 4x8 — because the fan-out
a landing rank performs was counted on neither side.

Note that a symmetric error passes these checks: a bias that scales both directions equally,
such as the `nvlink_dup` case below, leaves all three equalities intact.

Note the invariants hold on the **sum**, not per rank. The printed token counts come from
rank 0 while the MB figures are cross-rank averages, so the two need not agree digit for
digit.

## `--ignore-local-traffic`

| | flag off | flag on |
|---|---|---|
| `scale_out_send` | every destination node, local included | remote nodes only |
| `scale_out_recv` | plus this rank's own tokens this node wanted | other nodes' rail members only |
| `nvl_send`, `nvl_recv` | real LSA traffic | **unchanged** |

Use it whenever you compare `scale_out_bw` against NIC line rate. With the flag off that
column is *not* RDMA traffic.

## Caveat: `nvl_bw` under EM `nvlink_dup`

`nvl_send` and `nvl_recv` count one NVLink write per destination **rank**, which is what the
kernels issue in every EM mode except one. Under `nvlink_dup` the sender stores once per
eligible **expert slot** — the same-destination dedup in `dispatch_s2g_resolve_dest` is skipped
— so a token wanting two experts on one peer produces two writes and `nvl_bw` reads low. The
gap is the ratio of eligible expert slots to distinct destination ranks on a node: about 1.23x,
or 23% low, at 16 ranks with top-k 8 and 16 experts per rank.

This matters because `nvlink_dup` is the **default for multi-node EM runs with `--zcopy`**
(`zero_copy=ON` and more than one node auto-selects it). Check the `HT EM mode:` line in the
bench's configuration banner before quoting `nvl_bw` from an EM run. `total_*` and
`scale_out_*` are unaffected — only the per-rank NVLink dedup is wrong — and because the bias
is symmetric, the invariants above still hold.

## Worked figures: a synthetic example

**The numbers in this section are synthetic. They are not a measured run, and no hardware
was involved in producing them.** They exist only so the relationships between the columns
can be checked by hand, and they are chosen to be round for exactly that reason. Do not
quote any of them as a performance result.

The example assumes a fabric giving each rank **50 GB/s of network bandwidth one way**, the
2-node topology above (2 nodes of 8 ranks, 16 ranks total), and a routing where every token
reaches 8 distinct destination ranks, 4 on each node. 12500 tokens per rank, 16.00 kB per
token. The figures are set so the NIC is exactly saturated, which is what makes the flag-off
column's arithmetic impossible to miss.

```
--- BW based on total time ---                  (--ignore-local-traffic OFF, the default)
Dispatch:    total=4000.00 us (min=3820.44, max=4311.07)
             recv: total_bw=400.00  nvl_bw=375.00  scale_out_bw=100.00 GB/s
             send: total_bw=400.00  nvl_bw=375.00  scale_out_bw=100.00 GB/s
Combine:     total=5000.00 us (min=4762.19, max=5388.63)
             send: total_bw=320.00  nvl_bw=300.00  scale_out_bw=80.00 GB/s
             recv: total_bw=320.00  nvl_bw=300.00  scale_out_bw=80.00 GB/s

Logical payload bytes (tokens + forwarded scales, per-rank avg): total_send=1600.00 MB
(100000 tokens), nvl_send=1500.00 MB (93750 tokens), scale_out_send=400.00 MB (25000 tokens),
total_recv=1600.00 MB (100000 tokens), nvl_recv=1500.00 MB (93750 tokens),
scale_out_recv=400.00 MB (25000 tokens)
```

**Check the invariants first.** `total_send` = `total_recv` = 1600.00 MB, `nvl_send` =
`nvl_recv` = 1500.00 MB, `scale_out_send` = `scale_out_recv` = 400.00 MB. All three hold, so
the accounting is self-consistent and the rates are worth reading. They are exactly equal
here because the example is idealised; in a real run the recv token counts differ slightly
from the send ones, because the recv side replays each source's routing from its seed rather
than reading the real `topk_idx`.

**Predict `nvl_bw` from the topology.** With N=2 nodes of L=8 and uniform routing, of all
deliveries 1/2 stay on the sender's node, 1/16 go to the sender's rail partner on the other
node, and the remaining 7/16 are two-hop. LSA carries the first and last, so it should see
1/2 + 7/16 = **15/16** of the payload:

```
predicted  nvl / total = 15/16   = 0.9375
example    1500.00 / 1600.00     = 0.9375        ok
           400.00 x 0.9375       = 375.00 GB/s   ok, matches the nvl_bw column
```

A ratio that drifts from `1 - (N-1)/(N*L)` means the routing was not uniform, not that the
counters are wrong. Two caveats on `nvl_send` worth stating, because both are easy to
misread. It is an **expectation**, not an identity: the 15/16 needs the destinations on a
node to be spread evenly across its ranks, and adversarial routings inside the same
constraints give 7 or 8 NVLink writes per token instead of 7.5, putting `nvl_bw` anywhere in
350-400 GB/s. And it is **not the cost of a rank's own tokens** — of the 1500.00 MB here,
only 800.00 MB is this rank dispatching; the other 700.00 MB is relay work it performs as
landing rank for its rail partner.

**The columns do not sum to `total_bw`.** 375.00 + 100.00 = 475.00 against 400.00. Two
separate effects produce that, and only the second is double counting:

- **Relay amplification.** Each token costs 4 local NVLink writes, 1 RDMA put, and 3.5 more
  NVLink writes at the landing rank: **8.5 physical transfers for 8 logical deliveries**. So
  `total_send` is a logical delivery rate, not a bound on wire bytes. Note that `nvl_bw` +
  `scale_out_bw` with the flag **on** is 375.00 + 50.00 = 425.00, already above 400.00, even
  though those two columns are **disjoint** — NVLink writes and RDMA puts share no bytes.
- **Double counting on top of that.** With the flag off, the local-node entry re-counts bytes
  already in `nvl_send`, taking the accounted total to 9.5 transfers per token and 475.00.

**The same figures with `--ignore-local-traffic`:**

```
Dispatch:    total=4000.00 us (min=3820.44, max=4311.07)
             recv: total_bw=400.00  nvl_bw=375.00  scale_out_bw=50.00 GB/s
             send: total_bw=400.00  nvl_bw=375.00  scale_out_bw=50.00 GB/s
Combine:     total=5000.00 us (min=4762.19, max=5388.63)
             send: total_bw=320.00  nvl_bw=300.00  scale_out_bw=40.00 GB/s
             recv: total_bw=320.00  nvl_bw=300.00  scale_out_bw=40.00 GB/s

scale_out_send=200.00 MB (12500 tokens)  scale_out_recv=200.00 MB (12500 tokens)
nvl_send=1500.00 MB                      nvl_recv=1500.00 MB
```

`nvl_*` is byte-identical to the flag-off figures, as it must be — the flag touches no LSA
traffic. `scale_out` halves, from 400.00 to 200.00 MB, because every token contributed two
entries (its own node and the remote one) and now contributes one.

**This is the clearest reason to use the flag.** Against a 50 GB/s fabric, the flag-on
column reads 50.00 GB/s: the NIC is saturated, which is as high as it can legitimately go.
The flag-off column reads 100.00 GB/s — **twice what the fabric can carry.** A column that
exceeds line rate cannot be measuring the wire. It is counting local NVLink deliveries
alongside the puts, and with two nodes the local and remote halves are near-symmetric, so it
lands at about twice the NIC figure; in general the overstatement is `N/(N-1)`.

Read the other way, the flag-on column shows the dedup the rail scheme buys: 4 of the 8
deliveries per token are remote, so 800.00 MB of logical remote payload is carried by
200.00 MB of puts — **4.0 deliveries per put**, one per destination rank on the remote node.
The saving is in NIC bytes only; those 4 deliveries still cost 1 put plus 3.5 NVLink writes.


## Summary: which number to use

Start by confirming the three invariants — `total_send` = `total_recv`, `nvl_send` =
`nvl_recv`, `scale_out_send` = `scale_out_recv`. If any of them fails, the rates mean nothing
and the counters need fixing before anything else is read.

| what you want to know | who asks | read | flag |
|---|---|---|---|
| How effectively does the system move data between ranks? | data scientist | `total_bw` | either |
| Are we saturating NVLink? | kernel / intra-node engineer | `nvl_bw` | either — flag-independent |
| Are we driving the RDMA network well? | network engineer | `scale_out_bw` | `--ignore-local-traffic` **on** |
| How effectively does the system transmit between rails? | benchmark engineer / evaluator | `scale_out_bw` | **off** |

**`total_bw` — the logical payload rate.** Every delivery the routing asked for, once per
destination rank, whichever path carried it. This is the **data scientist's** number: how
effectively is the system moving data between ranks for a given model shape. Use it to
compare configurations, routings, top-k values or token counts against each other. Do not
hold it against a hardware limit — no single link carries this traffic, and a local store
counts as one delivery just as a two-hop RDMA delivery does.

**`nvl_bw` — LSA traffic.** Everything that moved over NVLink, including the intra-node
fan-out a landing rank performs for other ranks and the degenerate local store a rank makes
to its own experts. This is the number to hold against NVLink peak bandwidth when tuning the
intra-node path, with two adjustments: it does not change with `--ignore-local-traffic`, so it
still includes local stores that touch no wire (6.7% of it at 16 ranks with 8 per node), and it
reads about 23% low under EM `nvlink_dup` — see the caveat above. The send and recv rows always
print the same value, since NVLink send and recv are the same bytes seen from two ends.

**`scale_out_bw` with `--ignore-local-traffic` — RDMA traffic.** Only what crosses the NIC:
one put per remote destination node. This is the **network engineer's** number, for
optimizing the network layer and holding against per-GPU NIC line rate — and the only form
of the column that may be quoted as an RDMA figure. Expect it to be a small fraction of
`total_bw`: one put serves every expert on the destination node, so the payload-to-wire ratio
improves as fan-out per node grows.

**`scale_out_bw` without the flag — per-node delivery count.** One entry per destination
node, the local one included. This is the **benchmark and evaluation** number: how effectively
is the system dispatching data, counted at one consistent granularity rather than only what
leaves the node. Its most common misuse is quoting it as network bandwidth; in the synthetic
figures above that would overstate RDMA by **2.0x**, since 400.00 MB includes 200.00 MB that
never touched a NIC — enough to put the column at twice the fabric's line rate.

**The columns are not additive**, for two reasons. They overlap — a two-hop delivery consumes
both an RDMA put and an NVLink write and is counted in both columns — and separately, relaying
means more physical transfers occur than there are logical deliveries, so even the disjoint
pair `nvl_bw + scale_out_bw` with the flag on exceeds `total_bw` (425.00 against 400.00 in the
figures above; 475.00 with the flag off). Read each column as an answer to its own question,
never as a share of a whole.
