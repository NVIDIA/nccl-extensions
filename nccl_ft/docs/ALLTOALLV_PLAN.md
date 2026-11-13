# FT AllToAllv — Phase 1 Plan (fused-count variable all-to-all)

## Low-level auto-recv-sizing API (for EP / MoE integrators)

The variable all-to-allv is exposed via `ftAllToAllv` (C) / `FTNccl.alltoallv`
(ctypes). There is intentionally **no** framework-opinionated `dispatch()` — an EP
library calls the primitive directly with its own device buffers. Two modes,
selected by `recv_capacity`:

- **Packed** (`recv_capacity == 0`): caller supplies `d_recv_counts` /
  `d_recv_displs`; the kernel packs each source into the caller's contiguous
  output. This is what `FTProcessGroup.alltoall_base` uses for
  `torch.all_to_all_single` with split sizes.
- **Auto-recv-sizing** (`recv_capacity > 0`): the caller supplies only the send
  side; the **receiver discovers** how much each source sent it (fused
  count-in-flag: the sender stores `count+1` in the completion flag) and writes
  it to `d_recv_counts_out` (device `int32[nRanks]`). Each source `s` is packed
  into a fixed slot `output[s*recv_capacity ...]`. No separate count exchange.

```c
ftAllToAllv(handle,
    win_send, sendoffset,          // symmetric send window
    win_recv, recvoffset,          // the handle's registered scratch (nRanks slots)
    win_flags,
    d_send_counts, d_send_displs,  // DEVICE int32[nRanks] / int64[nRanks] (native elems)
    recv_stride_bytes,             // fixed scratch slot stride (>= max_per_dest*sizeof)
    max_send_count,                // single- vs multi-CTA hint
    output,                        // DEVICE ptr; [nRanks*recv_capacity] in auto mode
    /*packed*/ NULL, NULL,         // d_recv_counts, d_recv_displs (packed mode only)
    /*auto*/   d_recv_counts_out, recv_capacity,   // discovered counts (device), slot stride
    dtype, stream, timeout_us);    // FTDataType; ASYNC on `stream`
```

Everything is **device-resident**: send counts/displs in, discovered recv counts
out — no host round-trip. On a rank timeout the source's `d_recv_counts_out[s]`
is `0`; consult `ftHandleGetResultMask` / the error flag for membership.

### EP-integration niceties

- **Counts-only** — pass `d_send_displs = NULL` and the API derives the
  displacements on-device (`ftExclusivePrefixSum` into per-handle scratch). No
  Python/host displacement math; explicit displs still accepted.
- **Row/token semantics** — `row_size` is the elements per row (hidden size).
  Counts, displacements, and `recv_capacity` are then in **rows/tokens**;
  lengths/offsets are scaled by `row_size` internally and discovered counts come
  back in rows. `row_size = 1` is the original per-element semantics. Exchange
  `[num_tokens, hidden]` records by passing token counts + `row_size = hidden`.
- **Caller-stream ordering** — the `FTProcessGroup` methods insert bidirectional
  stream ordering (FT stream waits for the caller's current stream via a
  persistent event; `FTWork.wait()` orders the caller's next op after the kernel)
  with **no host synchronize**. An EP library calling the primitive with its own
  stream gets natural ordering (single stream).
- **Preallocated workspace / CUDA graphs** — the communicator preallocates and
  registers its scratch (recv window, flags, readiness, work-steal counter,
  derived-displs scratch, and the metadata staging buffers) at init. The
  steady-state FT path is **allocation-free and host-sync-free** (kernel launches
  + stream-ordered async copies + event record/wait), so it is CUDA-graph
  capturable. The caller supplies a stable `output` and (for auto mode) a stable
  `d_recv_counts_out`.

### Reverse exchange (combine)

Combine is the same primitive **reversed**: rank *r* sends back to each source
*s* exactly what it received from *s*. So the forward's discovered
`d_recv_counts_out` **is the route handle** — pass it as `d_send_counts` for the
reverse `ftAllToAllv` (with `d_send_displs = NULL` to auto-derive offsets, same
`row_size`). No new collective, no re-exchange of sizes. The per-token reduction
/ top-k weight renormalization stays in the caller (application level).

## Implementation status (as-built on `ft-all-to-allv`)

Everything below is landed and pushed. Where the as-built design differs from the
original Phase-1 plan (further down), the as-built description here is
authoritative — the plan is kept as the design record.

- **Phase 0 — uniform FT all-to-all** (`93631866a`, `56ba654bc`):
  `FTAllToAllKernel` / `FTAllToAllKernelMultiCTA`, `ftAllToAll` (C) + ctypes
  binding, `FTProcessGroup.alltoall_base` uniform path.
- **Phase 1 — variable all-to-allv** (`805ad316a` single-CTA, `72ffb3e0f` Python
  + `test_alltoallv.py`): rank *r* pushes `send_counts[d]` rows to dest *d*'s
  fixed scratch slot[r]; **no per-slot header** — see count-in-flag below.
- **Multi-CTA + multi-dtype** (`bdb2a42e3`, `e65f6aa82`): cooperative
  `FTAllToAllvKernelMultiCTA`, `{single,multi}×{f32,f16,bf16}` dispatch, native-
  element counts, persistent pinned+device count/displ buffers (no per-call
  sync).
- **Fused compaction + balanced push** (`b1ea0967b`): compaction is **fused into
  the main kernel** (not a separate `FTAllToAllvCompactKernel`) — after the flag
  poll each source's padded slot is packed into `output`. CTA balancing is a
  **global-atomic work-steal over uniform token slices interleaved by peer**
  (`d_a2av_ctr`), not the prefix-sum→(peer,slice) assignment the plan sketched.
- **Auto-recv-sizing via count-in-flag** (`1d4f02127`): the sender stores
  `count+1` in the completion flag; the receiver derives `recv_counts = flag-1`
  during the flag poll. This **replaces the 16 B slot header** of the original
  plan — one flag round, no header, no separate count exchange. `recv_capacity>0`
  selects auto mode (fixed output slots); `==0` keeps packed mode.
- **Scope: no framework MoE** (`a9ad91c18`): the opinionated `dispatch()` method
  (and `test_dispatch.py`) were removed; the auto-recv-sizing *capability* stays,
  used via the low-level API. EP libraries own dispatch/combine policy.
- **EP-integration feedback** — counts-only NULL-displs on-device prefix-sum
  (`c75d42c35`), row/token semantics (`ea0e3fe4a`), upstream caller-stream
  ordering (`081845aef`), preallocated CUDA-graph-capturable workspace + doc
  (`4cbce0f21`), reverse-exchange route-handle (doc).
- **Benchmark + tests**: `benchmark_ft_collective.py --collective alltoallv`
  variable mode (`10c7b0d3b`); `test_alltoallv.py` (packed) strong correctness
  + `--kill`. (Correctness now consolidated in `ft_correctness_test.py`.)
- **Related**: the collective FT barrier reuses a fixed-round FT all-to-all
  (`07376a6f3`, `9e3802f71`) — documented in `CLAUDE.md` (`FT_BARRIER_MODE`).

The rest of this document is the **original Phase-1 design plan**. It predates the
count-in-flag / fused-compaction / work-steal decisions above; read §1–§4 as
design history, not as the shipped layout.

Branch: `ft-all-to-allv`. Phase 0 (uniform FT `all_to_all`) is implemented:
`FTAllToAllKernel` / `FTAllToAllKernelMultiCTA`, `ftAllToAll` (C), `ftAllToAll`
ctypes binding, and `FTProcessGroup.alltoall_base` for the uniform case. Phase 1
adds **variable per-destination counts** on top, using a **fused count header**
so there is no separate count-exchange collective.

## Core idea (recap)
We are push-based, so the sender already knows its `send_counts[j]` locally. Push
`[count header][payload]` into peer *j*'s slot and raise one flag. The receiver
polls the flag (timeout → mask), reads the count from the header, and knows how
much landed — **one FT collective, one flag round, no separate count exchange.**
Requires per-peer slots pre-sized to a bounded max (capacity); over-allocates
memory, not bandwidth. Dropless/unbounded is out of scope (needs a separate
count phase).

## What changes from Phase 0

### 1. Slot layout: header + padded body
Each recv slot for source *i* becomes `[header (16 B)][body (max_per_dest, padded)]`.
- Header word 0: `count` (elements actually sent by *i* to me).
- Header word 1: `generation` (staleness guard; matches the readiness model).
- `recv_stride = 16 + FT_RECV_STRIDE(max_per_dest * sizeof(T))`.
- Recv scratch = `nRanks * recv_stride`. Sized from `max_per_dest` (a new handle
  field / arg, analogous to `max_count`).

### 2. Kernel: variable push + count header (send side)
Replace the fixed-`count` push. Per destination *r*:
- write `header = {send_counts[r], generation}` into peer *r*'s slot,
- `ftCopy` `send_counts[r]` elements (the variable payload) into the slot body,
- fence, then flag (the flag still means "I spoke"; the header says what).
`send_counts[r] == 0` still writes the header + flag (resolves "0 vs dead").

CTA work assignment is now data-dependent: do a **device-side prefix-sum of
`send_counts`** (nRanks elements, one CTA) → assign `(peer, slice)` to CTAs from
the cumulative offsets so all egress links stay balanced (preserves the
peer-concurrency win). Send chunk *r* lives at `send_displs[r]` in the send
buffer (caller-provided or prefix-summed from `send_counts`).

### 3. Kernel: read counts (receive side)
The flag poll is unchanged (per-source, timeout → mask). On arrival, read the
slot header → `recv_counts[i]` (and check `generation`). Write `recv_counts[]`
and `active_mask_out[]` out for the host/compaction step. A masked/timed-out
source contributes `recv_counts[i] = 0`.

### 4. Local compaction (new, receiver-only, no comm)
Padded per-source slots → caller's packed output at the caller's displacements:
- prefix-sum `recv_counts` → `recv_displs` (device, nRanks).
- for each source *i*, gather its first `recv_counts[i]` elements from slot *i*'s
  body into `output[recv_displs[i] ...]`.
This is a segmented/variable-length copy kernel (vectorized bulk + scalar tail;
16 B-align each source's body start so the bulk stays `int4`). The packed output
is **local**, so it need not be a symmetric window — only the padded scratch is.

### 5. FT semantics
- Timeout on a source flag → mask that source; its `recv_counts[i] = 0` → it
  simply contributes nothing to the packed output (a clean hole, not garbage).
- Consensus: reconcile masks via `ft_converge()` / the collective FT barrier for
  consistent membership across ranks (same as the other collectives).
- Contract: on `FT_TIMEOUT`, the packed output is missing the dead source(s);
  caller consults the mask. For MoE dispatch this is a token drop (degraded but
  valid), not a discard.
- `generation` in the header rejects a stalled-then-resumed sender's stale
  payload.

## Implementation steps
1. **Handle:** add `max_per_dest` (elements) + recv scratch sizing from it
   (`ftHandleCreate` / a new create path). Env `FT_A2A_MAX_PER_DEST`.
2. **Kernels:** `FTAllToAllvKernel[MultiCTA]` from the Phase-0 kernels — add the
   header write/read, variable `len = send_counts[r]`, and the send-side
   prefix-sum for CTA assignment. Keep readiness/flag/timeout/mask verbatim.
3. **Compaction kernel:** `FTAllToAllvCompactKernel` (recv-side prefix-sum +
   segmented gather to packed output).
4. **C API:** `ftAllToAllv(handle, win_send, senddispls/sendcounts,
   win_recv, win_flags, d_recv_counts, d_output, max_per_dest, stream,
   timeout_us)`. Counts live in device memory (from the router/gating).
5. **Occupancy:** add the new multi-CTA kernel to the cooperative cap.
6. **Python:** extend `alltoall_base` to honor `input_splits`/`output_splits`
   (device or host); `_is_ft_eligible_a2av` checks `max(input_splits) <=
   max_per_dest`. Return discovered `output_splits` when the caller passes an
   empty output-splits list (auto recv-sizing — what MoE dispatch wants).

## Open questions / risks
- **Counts on host vs device.** torch `alltoall_base` split sizes are host
  lists; MoE routing produces device counts. Support both (host → H2D once).
- **`max_per_dest` selection.** Too small → drops (or fallback); too large →
  memory. For MoE this is the capacity factor.
- **Alignment.** Variable counts break 16 B alignment mid-buffer; pad each slot
  body and each compaction segment, scalar-tail the remainder (Phase 0 already
  falls back to scalar for unaligned `count`).
- **Two prefix-sums per call.** Small (nRanks), but new kernels; fold into the
  main kernel where possible (send-side) or a tiny separate launch.

## Reuse from Phase 0 / existing code
Readiness (push/generation), flag rounds, timeout+mask, `active_mask_out`,
symmetric windows, `FT_RECV_STRIDE`, the peer-concurrency reorder, self-adapt,
and the `ft_converge` consensus barrier all carry over unchanged. The genuinely new
code is: the count header, the two prefix-sums, and the compaction kernel.
