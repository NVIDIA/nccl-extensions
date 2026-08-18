# NCCL M2N Python API

The `nccl.m2n` package provides Python access to the NCCL M2N resharding
library. See the [Python package overview](../../README.md) for installation
and packaging information.

## Entry points

`nccl.m2n.reshard` is the primary entry point and uses staging.
`nccl.m2n.reshard_with_window` is a compatibility alias that ignores the
supplied window; providing one does not guarantee zero-copy execution.

The package also provides explicit handles, grouped submission, mesh and
placement descriptors, and DTensor interop.

```python
import nccl.m2n as m2n

# Staging-backed reshard.
m2n.reshard(src, dst, comm, stream=stream, src_mesh=src_mesh,
            src_placements=src_placements, dst_mesh=dst_mesh,
            dst_placements=dst_placements)

# Compatibility entry point for a caller-registered window.
m2n.reshard_with_window(src, dst, comm, window, stream=stream,
                        src_mesh=src_mesh, src_placements=src_placements,
                        dst_mesh=dst_mesh, dst_placements=dst_placements)
```

## Mesh and placement

`Mesh` accepts flat rank lists and nested 2-D rank grids. A flat list keeps the
historical Python `Mesh.dims == (N, 1)` view while emitting a native 1-D C
descriptor. Use one placement for a flat list and two placements for a nested
2-D grid. The current Python API supports one or two mesh dimensions.

The C descriptor uses borrowed arrays for `dims`, `localShape`, and
`placements`; grouped calls snapshot those arrays before submission.

Existing high-level `nccl.m2n` call signatures remain source-compatible for
2-D callers. The low-level C ABI and generated Cython struct view intentionally
change from fixed inline arrays to versioned pointer-backed descriptors. Code
that initializes `ncclMesh_t` or `ncclDistTensor_t` directly must use the new
descriptor layout.

For example, this uses native 1-D source and destination meshes. `comm` must
cover all source and destination ranks, and `rank` is this process's rank in
that communicator:

```python
import torch
import nccl.m2n as m2n

src_size, dst_size = 4, 4
global_rows = 4096
src_mesh = m2n.Mesh([src_size], start_rank=0)
dst_mesh = m2n.Mesh([dst_size], start_rank=src_size)
stream = torch.cuda.current_stream()

src = (
    torch.empty((global_rows // src_size,), device="cuda")
    if rank < src_size
    else None
)
dst = (
    torch.empty((global_rows // dst_size,), device="cuda")
    if rank >= src_size
    else None
)
m2n.reshard(
    src,
    dst,
    comm,
    stream=stream,
    src_mesh=src_mesh,
    src_placements=[m2n.Shard(0)],
    dst_mesh=dst_mesh,
    dst_placements=[m2n.Shard(0)],
)
stream.synchronize()
```

## Grouping M2N calls

Use `m2n.group()` to submit several reshard calls together. The context ends
the group on success and aborts it if its body raises an exception.

Grouping defines a submission boundary and preserves abort and error
semantics. Grouping does not guarantee a performance benefit, and broader
grouping optimizations are future work.

```python
import nccl.m2n as m2n

# `comm`, `stream`, `src0`, `dst0`, `src1`, and `dst1` are prepared as for a
# normal M2N reshard call.
with m2n.init() as handle:
    with m2n.group():
        handle.reshard(comm, src0, dst0, stream=stream)
        handle.reshard(comm, src1, dst1, stream=stream)
    stream.synchronize()
```

For explicit control, pair `group_start()` with `group_end()` and abort on an
exception:

```python
with m2n.init() as handle:
    m2n.group_start()
    try:
        handle.reshard(comm, src0, dst0, stream=stream)
        handle.reshard(comm, src1, dst1, stream=stream)
    except BaseException:
        m2n.group_abort()
        raise
    else:
        m2n.group_end()
    stream.synchronize()
```

## Benchmark

Install the optional benchmark dependency before running the packaged M2N
benchmark:

```bash
pip install 'nccl-extensions[bench]'
python -m nccl.m2n.benchmarks.reshard_bench --help
```

## Native library loading

The M2N loader uses `NCCL_M2N_LIBRARY` when supplied. Otherwise, it prefers the
bundled `nccl/m2n/lib/libnccl_m2n.so` and retains the existing `NCCL_M2N_HOME`,
Conda, CUDA, and SONAME fallbacks.
