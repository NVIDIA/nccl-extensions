# NCCL M2N v0.3.1 Release Notes

## Fixes

- Fix nonblocking communicator handling in benchmarks.
- Check and reject PyTorch CPU tensors passed as device buffers in the Python API.

## Known Limitations

| Limitation | Description |
|---|---|
| Tensor and mesh rank | Tensors support up to 3 dimensions. Meshes support 1 or 2 dimensions. |
| Communicator concurrency | Concurrent calls on the same communicator using different CUDA streams is unsupported. |
| Peer caps | Peer lists support at most 16 sources and 64 targets per rank on staging paths; other paths have separate limits. |
| PIPE host-RMA | Requires NCCL 2.31.2 or newer. |
| Supported Buffer Types | CPU tensors, including pinned CPU tensors, are currently not supported. |
