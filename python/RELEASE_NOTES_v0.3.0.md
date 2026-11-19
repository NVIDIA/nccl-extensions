# nccl-extensions 0.3.0 — Release Notes

This release includes the main-branch feature set through `90c31fb8`, plus
release version metadata and these notes.

| Component | Version |
| --- | --- |
| Python package | 0.3.0 |
| NCCL-EP | 0.3.1 |
| NCCL-M2N | 0.3.1 |
| NCCL runtime requirement | >=2.32.3 |

## Changes from package 0.2.0

- EP: adaptive direct/hierarchical LL dispatch, BF16 LL routing weights,
  HT Expert-Major pull/push MXFP8 combine, fused NVFP4 global-scale
  calculation, dynamic HT work assignment, and correctness/performance fixes.
  See [EP 0.3.1 notes](../nccl_ep/docs/release/RELEASE_NOTES_v0.3.1.md).
- M2N: PIPE LSA copy alignment fixes, nonblocking benchmark communicator
  handling, and rejection of CPU tensors as device buffers.
- Packaging/build: NCCL submodule advances to 2.32.3; CUDA extras change from
  exactly NCCL 2.30.7 to NCCL >=2.32.3. CI build orchestration and NCCL
  build caching are reworked.

## Upgrade considerations

The NCCL dependency floor is higher. Experimental NVFP4 combine now documents
internally calculated scales (`inputs->scales = NULL`). Hierarchical LL can
require collective handle initialization and graph recapture after shared
buffer layout changes; consult the handle initialization contract in
`nccl_ep/include/nccl_ep.h` before upgrading graph-based callers.

## Release preparation

This branch is a release candidate, not a published or GPU-qualified
release. CUDA builds, GPU correctness/performance qualification, and final wheel
installation checks remain required before tagging and publishing.
