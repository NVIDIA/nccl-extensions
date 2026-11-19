# NCCL EP v0.3.1 — Release Notes

These notes cover changes since [EP 0.3.0](RELEASE_NOTES_v0.3.md), as shipped
in the nccl-extensions 0.2.0 release line. EP 0.3.1 is included in the
nccl-extensions 0.3.0 package.

## Features and improvements

- Adaptive direct and hierarchical low-latency dispatch.
- BF16 routing weights in LL dispatch and combine, with FP32 reduction.
- MXFP8 combine for HT Expert-Major pull-dispatch/push-combine, extending
  the push-dispatch/pull-combine support in EP 0.3.0.
- Fused internal global-scale calculation for experimental NVFP4 combine.
- Dynamic work assignment in HT pull dispatch and push combine.
- Reduced LL host launch overhead by preserving host compiler optimization
  and avoiding JIT source construction on cache hits.
- Enabled forwarded scales in LL LSA STORE/TMA dispatch, including zero-copy,
  and corrected physical-row sizing and warp-tile selection for packed and
  one-byte data types.
- Fixed the LL combine BF16 16K hidden-size limitation through adaptive
  receive unrolling.
- Corrected eager Expert-Major pull/push count exchange and bounded HT
  dispatch GIN queue handling.
