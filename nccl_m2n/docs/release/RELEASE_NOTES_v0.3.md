# NCCL M2N v0.3 Release Notes

NCCL M2N v0.3 extends resharding support and improves asynchronous execution.
It requires NCCL 2.30.5 or newer and a Linux build.

## New Features

- `ncclDistTensor_t` supports sharding across both mesh axes, including
  sharding the same tensor dimension on both axes.
- Reuse generator-side communicators on compatible split paths to improve
  cold-start overhead during weight synchronization.
- Python packaging supports CUDA 12 and CUDA 13 native libraries.
- A JAX example demonstrates eager weight transfer from row-sharded to
  column-sharded GPU groups with explicit synchronization.

## Fixes

- PIPE host-RMA establishes readiness for nonblocking communicators before
  dependent CUDA events.
- PIPE fixes address descriptor races, peer readiness, FIFO reuse, and signal
  ordering.
- A warning reports possible host submission overhead when NCCL operations
  return `ncclInProgress`.

## Known Limitations

| Limitation | Description |
|---|---|
| Tensor and mesh rank | Tensors support up to 3 dimensions. Meshes support 1 or 2 dimensions. |
| Window entry point | `ncclReshardWithWindow` currently ignores its window argument and follows `ncclReshard` transport selection. |
| Communicator concurrency | Sequential calls can be stream-ordered. Concurrent host calls on the same communicator are unsupported. |
| Finalization | `ncclM2nFinalize` does not synchronize CUDA streams. Complete submitted work before finalization or resource destruction. |
| Peer caps | Peer lists support at most 16 sources and 64 targets per rank on staging paths; other paths have separate limits. |
| Failure recovery | An error after PACK host-RMA protocol entry is fail-stop for its communicator and runtime epoch. |
