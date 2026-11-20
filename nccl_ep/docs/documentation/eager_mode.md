# HT Eager Mode

By default a High Throughput group commits up front to a fixed per-rank recv
budget, `ncclEpGroupConfig_t::max_recv_tokens_per_rank`, and every dispatch recv
tensor is sized to it. Because routing is data-dependent, that budget has to cover
the worst case, and most iterations leave most of it unused.

**Eager mode** trades that fixed reservation for per-iteration sizing. Create the
group with `max_recv_tokens_per_rank = NCCL_EP_AUTO` and the caller sizes dispatch
recv buffers to the actual recv count of the current routing instead.

Eager mode is **HT only**. Low Latency always sizes its buffers automatically and
ignores the field.

## Sizing recv buffers

Supply `ncclEpLayoutInfo_t::recv_total_counter` to `ncclEpCreateHandle` or
`ncclEpUpdateHandle`. The metadata kernel writes the rank's actual recv count
there; copy it device-to-host, synchronize, then allocate.

The counter is readable in **any** mode. What eager mode adds is permission to
*size the dispatch outputs to it* rather than to the worst case.

For `NCCL_EP_LAYOUT_EXPERT_MAJOR` with `dispatch_output_per_expert_alignment`,
the counter is the padded output-row extent: each local expert's token count
rounded up to the alignment, summed. Two local experts that receive 3 and 5
tokens at alignment 4 get zones at offsets 0 and 4 with padded counts 4 and 8,
so the counter reports 12 rows for 8 routed tokens. The padded count can exceed
the worst-case number of routed rows, `nRanks * max_dispatch_tokens_per_rank *
num_topk`. Size the recv tensors from the counter, not from that product. In the
local-permute and pull/push expert-major modes the eager bound adds
`num_local_experts * (alignment - 1)` rows of worst-case padding, and
`ncclEpCreateHandle`, `ncclEpInitHandle` and `ncclEpImportHandle` return
`ncclInvalidArgument` if that bound does not fit a signed 32-bit row count. The
remaining expert-major modes keep the unpadded bound and trap beyond it.

A rank may legitimately receive **zero** tokens under eager routing. Recv outputs
may then be empty (`data == nullptr`), but the token and weight outputs must agree
on emptiness — they describe rows of the same recv set, so one cannot be empty
while the other is not.

## Restrictions

1. **`num_topk` is required for `NCCL_EP_LAYOUT_EXPERT_MAJOR`.** Set
   `ncclEpGroupConfig_t::num_topk` to an upper bound on per-token top-k across the
   group's handles; expert-major expansion is unbounded without it.
   `ncclEpInitHandle` returns `ncclInvalidUsage` if it is missing, and validates
   each handle's `num_topk` against it. It is optional for `NCCL_EP_LAYOUT_FLAT`,
   whose deduplicated rows make the factor `1`.
2. **No CUDA Graph capture of `ncclEpDispatch`.** Sizing per routing requires the
   recv count on the host, which is unavailable mid-capture. Capturing dispatch
   under eager mode returns `ncclInvalidUsage`. Use a fixed
   `max_recv_tokens_per_rank` if you need graph capture.
3. **No `NCCL_EP_OVERFLOW_DROP`.** Eager mode relies on trap semantics, and
   `ncclEpCreateGroup` rejects the combination. See
   [Recv Overflow Policy](overflow_policy.md).
