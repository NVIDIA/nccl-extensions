# NCCL M2N Python Binding

This directory is a standalone Python project for the `nccl.m2n` binding. It
wraps the public `libnccl_m2n.so` C API from `src/nccl_m2n.h` while depending
on `nccl4py>=0.3.1` for communicator, window, stream, and typing
conventions:

- `nccl.m2n.bindings.nccl_m2n`: low-level Cython structs and functions.
- `nccl.m2n`: Pythonic descriptors and reshard APIs.

When staged in the NCCL source tree, this project lives at:

```text
nccl_m2n/python
```

## Development Install

Install NCCL4Py first, then install this project. Published package flows can
let the `nccl-m2n` extras pull the matching NCCL4Py CUDA variant:

```bash
cd /path/to/nccl/nccl_m2n/python
python -m pip install -e ".[cu12]"  # or ".[cu13]" for CUDA 13
```

For source-tree development where NCCL4Py is checked out but not published,
install NCCL4Py from the NCCL checkout and then install `nccl-m2n` without
resolving dependencies again:

```bash
export NCCL_REPO_DIR=/path/to/nccl
python -m pip install -e "$NCCL_REPO_DIR/bindings/nccl4py[cu12]"
python -m pip install -e . --no-deps
```

The Cython build needs `nccl_m2n.h` and `nccl.h`. The setup script searches:

- `NCCL_M2N_INCLUDE_DIR` or `$NCCL_M2N_HOME/include`
- `NCCL_INCLUDE_DIR` or `$NCCL_HOME/include`
- `$NCCL_REPO_DIR/{build/include,src/include,include}`
- `../src` relative to this `python/` directory

`libnccl_m2n.so` is an external runtime prerequisite; it is not bundled in the
wheel or sdist. Build and install NCCL M2N first:

```bash
NCCL_HOME=/path/to/nccl/build make -C ..
export NCCL_M2N_HOME=/path/to/nccl/nccl_m2n/build
```

`NCCL_M2N_LIBRARY=/path/to/libnccl_m2n.so` can be used instead of
`NCCL_M2N_HOME` when the shared library is not under an install prefix.
The loader resolves the complete M2N v2 API from that one library and rejects
older or incomplete libraries.

## Minimal Usage

```python
import nccl.core as nccl
from nccl.m2n import reshard, Shard

reshard(
    src_buffer,         # None on destination-only ranks
    dst_buffer,         # None on source-only ranks
    comm,
    stream=stream,
    src_mesh=src_ranks,
    src_placements=[Shard(0)],
    dst_mesh=dst_ranks,
    dst_placements=[Shard(0)],
)
```

`reshard` accepts framework-neutral CUDA buffers or `None` for each side
and returns `None`. Buffers can be objects with `data_ptr()`, `shape`, and
`dtype`, CUDA Array Interface objects, or raw integer device pointers. Mesh
arguments can be `nccl.m2n.Mesh` or 1-D/2-D rank sequences; placements carry
the layout on each mesh axis. A flattened rank list remains valid and maps to a
1-D mesh with reshard mesh dims `(N, 1)`; pass nested rank rows when the caller needs to
preserve a 2-D mesh layout. If a role buffer is `None`, the corresponding mesh
and placements are still required because NCCL M2N uses both side layouts on
every rank. When exactly one role buffer is `None`, the missing side's
descriptor shape is derived from the participating side's local shape plus both
layouts. When both role buffers are `None`, pass explicit local shape and dtype
metadata for both layouts.

The high-level reshard APIs use the C library's lazily created internal default
handle when `handle=` is omitted, preserving M2N's internal warm caches across
operations. Python does not finalize the default handle at process exit and
does not destroy explicit handles from `Handle.__del__`; lifetime is always
caller-owned. Native calls are serialized with one process-wide Python lock to
match the M2N host-side concurrency restriction.

### Stream and memory lifetime

reshard enqueues asynchronous work on stream; every buffer — including PyTorch
tensors — must remain allocated and unmodified until that work completes. With
PyTorch's caching allocator, hold references until you synchronize, or call
tensor.record_stream(...) yourself with your torch stream.

CUDA work remains asynchronous. Synchronize the user stream before leaving a
handle context or calling `destroy()` / `finalize()`, then destroy communicators:

```python
from nccl.m2n import Handle, reshard, Shard

with Handle.create() as handle:
    reshard(
        src_buffer,
        dst_buffer,
        comm,
        stream=stream,
        src_mesh=src_ranks,
        src_placements=[Shard(0)],
        dst_mesh=dst_ranks,
        dst_placements=[Shard(0)],
        handle=handle,
    )
    stream.synchronize()
comm.destroy()
```

Raw pointer buffers need explicit local shape and dtype metadata:

```python
from nccl.m2n import reshard, Shard

reshard(
    src_ptr,
    None,
    comm,
    src_mesh=src_ranks,
    src_placements=[Shard(0)],
    src_local_shape=(256, 1024),
    src_dtype="float32",
    dst_mesh=dst_ranks,
    dst_placements=[Shard(0)],
)
```

For PyTorch DTensor inputs, `xdtensor_reshard` derives local tensors and
layout metadata. Explicit mesh or placement arguments override DTensor metadata
for that side, which is useful when the reshard communicator uses a subgroup-local
rank space:

```python
from nccl.m2n import xdtensor_reshard

xdtensor_reshard(
    src_dtensor,
    dst_dtensor,
    comm,
    stream=stream,
    src_mesh=src_comm_ranks,
    dst_mesh=dst_comm_ranks,
)
```

When a DTensor side is `None`, pass the missing side's layout explicitly:

```python
xdtensor_reshard(
    None,
    dst_dtensor,
    comm,
    src_mesh=src_ranks,
    src_placements=[Shard(0)],
)
```

For a DTensor-style wrapper where each rank owns only one local tensor,
adapt any framework-private mesh wrapper at the boundary, then call the real
NCCL path in a few lines; no golden broadcast helper or manual shape derivation
is needed:

```python
from nccl.m2n import Mesh, xdtensor_reshard

xdtensor_reshard(
    src_dtensor,
    dst_dtensor,
    process_group.nccl_communicator,
    src_mesh=Mesh.from_ranks(src_mesh.mesh),
    src_placements=src_placements,
    dst_mesh=Mesh.from_ranks(dst_mesh.mesh),
    dst_placements=dst_placements,
)
```

The lower-level descriptor API remains available for direct control:

```python
import nccl.core as nccl
from nccl.m2n.bindings import nccl_m2n as m2n_bindings
from nccl.m2n import Config, DistTensor, Handle, Mesh, Replicate, Shard

handle = Handle.create(Config(max_cta=8))
src_mesh = Mesh((4, 1), start_rank=0)
dst_mesh = Mesh((4, 1), start_rank=4)

src = DistTensor(
    src_tensor_or_none,
    local_shape=(256, 1024),
    dtype="float32",
    mesh=src_mesh,
    placements=[Replicate(), Shard(0)],
)
dst = DistTensor(
    dst_tensor_or_none,
    local_shape=(256, 1024),
    dtype="float32",
    mesh=dst_mesh,
    placements=[Replicate(), Shard(0)],
)

# Copy/staging-backed path. Does not require a registered window.
handle.reshard(comm, src, dst, stream=stream)

# Or use the zero-copy path when buffers are registered with NCCL windows.
# handle.reshard_with_window(comm, window, src, dst, stream=stream)
handle.destroy()
```

`nccl.m2n.init(config)` creates and returns an explicit handle. Keep the return
value alive, pass it as `handle=` to each high-level reshard call, and pass it
to `nccl.m2n.finalize(handle)` after synchronizing its user streams. A later
bare `reshard()` uses the separate C default handle without that configuration.
After synchronizing its user streams, call `nccl.m2n.finalize()` without a
handle to release only the C library's internal default handle.

The low-level Cython API can be used directly when callers need exact C ABI
control:

```python
cfg = m2n_bindings.Config()
handle = m2n_bindings.init(cfg.ptr)
# enqueue work, then synchronize the user stream
m2n_bindings.finalize(handle)
```

Passing `0` as the handle to a low-level reshard call uses the C library's
internal default handle; synchronize its streams and call
`m2n_bindings.finalize(0)` to release it.

`src_tensor_or_none` may be `None` on destination-only ranks and
`dst_tensor_or_none` may be `None` on source-only ranks.  Fully inactive ranks
may pass `None` for both tensors when they also provide explicit shape and dtype
metadata, so every rank can still describe the same source and destination
layouts.

### Error handling

Python argument and metadata validation raises `TypeError`, `ValueError`, or
`NcclInvalid`. Native M2N failures raise `NcclReshardError`.

reshard enqueues asynchronous work on stream; every buffer — including PyTorch
tensors — must remain allocated and unmodified until that work completes. With
PyTorch's caching allocator, hold references until you synchronize, or call
tensor.record_stream(...) yourself with your torch stream.

## Benchmark

`python/benchmarks/reshard_bench.py` mirrors the C++ single-layer benchmark
arguments and output.  Its default path reuses one `nccl.m2n.Handle` and calls
`Handle.reshard`, which matches `benchmarks/reshard_bench.cc --api default`:

```bash
cd python
mpirun -n 4 \
    "$PWD/.m2n-venv/bin/python" \
    benchmarks/reshard_bench.py \
    --api default \
    --python-api lowlevel \
    --src-mesh-dims 1,2 \
    --dst-mesh-dims 1,2 \
    --tensor-dims 8192,8192 \
    --src-shard-dim 0 \
    --dst-shard-dim 1 \
    --iterations 20 \
    --warmup 5 \
    --validate
```

Use `--buffer-provider raw --use-default-stream` to run the same low-level
default API benchmark with CUDA Runtime allocations and raw pointer metadata
instead of PyTorch tensors. Raw buffers do not support `--validate`.

Use `--python-api reshard` to measure the public `reshard` helper separately.
That mode includes the helper's default-handle path and is not the same
measurement as low-level binding overhead.

For parity comparisons, run `benchmarks/reshard_bench.cc --api default` and
this Python benchmark with matching mesh, tensor, algorithm, and iteration
arguments. The Python benchmark prints `CSV_RESULT` rows that can be compared
with the C++ benchmark's total throughput.
