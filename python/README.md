# nccl-extensions (Python)

Python bindings for the [nccl-extensions](../README.md) communication
libraries.

## Package layout

This package installs into the **`nccl` namespace**, so the import path is
`nccl.ep` rather than a separate top-level package:

```python
import nccl.ep as ep
import nccl.m2n as m2n
```

It contributes exactly three directories to that namespace, and no
`nccl/__init__.py`:

| path | contents |
| --- | --- |
| `nccl/ep/` | public facade for nccl_ep, plus `lib/libnccl_ep.so` and headers |
| `nccl/m2n/` | public facade for NCCL M2N, plus `lib/libnccl_m2n.so` and `include/nccl_m2n.h` |
| `nccl/_extensions/` | internals shared by every extension library — the Cython bindings, `binding_dataclass`, the distribution version |

## Status

`nccl.ep` is complete: the Cython bindings under
`nccl/_extensions/bindings/` plus the Python facade (group/handle management,
dispatch/combine, tensors, torch interop) under `nccl/ep/`.

`nccl.m2n` is complete: the Cython bindings under
`nccl/_extensions/bindings/` plus the Python facade (meshes, placements,
handles, resharding, and DTensor interop) under `nccl/m2n/`.

`nccl.m2n.reshard` is the primary entry point and uses staging.
`nccl.m2n.reshard_with_window` is a compatibility alias that ignores the
supplied window; providing one does not guarantee zero-copy execution.

### Mesh rank compatibility

`Mesh` accepts flat rank lists and nested 2-D rank grids. A flat list keeps the
historical Python `Mesh.dims == (N, 1)` view while emitting a native 1-D C
descriptor. Use one placement for a flat list and two placements for a nested
2-D grid. The C descriptor uses borrowed arrays for `dims`, `localShape`, and
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

### Grouping M2N calls

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

Install the optional benchmark dependency before running the packaged M2N
benchmark:

```bash
pip install 'nccl-extensions[bench]'
python -m nccl.m2n.benchmarks.reshard_bench --help
```

## Install

```bash
CUDA_HOME=/usr/local/cuda pip install -e python/
```

Building requires a CUDA toolkit and a Cython toolchain. The generated M2N ABI
declarations do not require a separately installed `nccl_m2n.h` or `nccl.h`.

Stage native artifacts before building a distributable wheel:

```text
python/nccl/ep/lib/libnccl_ep.so
python/nccl/m2n/lib/libnccl_m2n.so
python/nccl/m2n/include/nccl_m2n.h
```

Missing shared libraries emit explicit build warnings. The resulting wheel is
not self-contained and needs a compatible external library at runtime. M2N
uses an explicit `NCCL_M2N_LIBRARY` override when supplied; otherwise it
prefers the bundled `nccl/m2n/lib/libnccl_m2n.so` and retains the existing
`NCCL_M2N_HOME`, Conda, CUDA, and SONAME fallbacks.

The sdist is source-only and excludes native shared libraries. Building a
wheel from it must stage the native libraries at the paths above to bundle
them, or provide compatible external libraries for runtime loading.

Pick a CUDA-variant extra to pull in the matching runtime stack (they forward
to nccl4py's `cu12` / `cu13` extras, and are mutually exclusive):

```bash
pip install -e 'python/[cu13]'
```

> **Do not run Python from inside `python/`.** There is no `nccl/__init__.py`
> there, so that directory resolves only as a namespace portion and these
> modules become invisible. Always go through the editable install.

## Regenerating the bindings

Everything under `nccl/_extensions/bindings/` is generated and checked in. Do
not edit it by hand — re-run
[`build_assets/generate_cython.py`](build_assets/README.md) after changing an
EP or M2N public header, config, or template and commit the result. The M2N
public API remains `nccl.m2n`; `nccl.m2n.bindings` is not a supported import
path.
