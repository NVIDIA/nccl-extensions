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

`nccl.m2n.reshard` uses staging. `nccl.m2n.reshard_with_window` uses a
caller-registered window.

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

### Scale forwarding

Reshard a quantized payload together with its companion per-block scale plane
in one validated call. `block_size` is always caller-supplied; per-side scale
shapes can be derived from the payload descriptors or given explicitly.

```python
from nccl.m2n import DistTensor, ScalePlane, reshard_scaled

scales = ScalePlane(
    src_scale_buf, dst_scale_buf,
    dtype="float32",       # fp32/fp16/bf16/fp8/uint8 only
    block_size=128,
    block_dim=1,
    src_payload=src_desc,  # derive scale shapes from the payload...
    dst_payload=dst_desc,
)
# ...or pass src_local_shape=/dst_local_shape= explicitly instead.

reshard_scaled(src_buf, dst_buf, comm, scales, stream=stream, ...)
```

`Handle.reshard_scaled` and `Handle.reshard_scaled_with_window` are the
handle-scoped equivalents. The buffer-lifetime contract below applies to the
scale buffers as well as the payload.

Note: `ncclReshardScaled` / `ncclReshardScaledWithWindow` are post-v2 additions
and are resolved lazily. Against an older `libnccl_m2n.so` the package still
imports; calling them raises `FunctionNotFoundError`.

### On-the-fly quantization

Compress the payload on the wire as FP8 with generated per-block scales,
reconstructing the caller's dtype at the destination:

```python
from nccl.m2n import QUANT_MXFP8, QuantSpec, reshard_quantized

# Dequantizing: destination gets the original dtype back.
quant = QuantSpec(block_dim=1, block_size=128, round_scales=True)

# Keep-quantized: destination declares ncclFloat8e4m3 and receives the
# generated scales, for a consumer that computes in FP8.
quant = QuantSpec(block_dim=1, recipe=QUANT_MXFP8, round_scales=True,
                  dst_scales=dst_scale_buf)

reshard_quantized(src_buf, dst_buf, comm, quant, stream=stream, ...)
```

`Handle.reshard_quantized` is the handle-scoped equivalent. Payload dtype must
be bfloat16, float16, or float32, and the block dimension's extent must be a
multiple of `block_size` on both sides.

This is **lossy** — each block is reconstructed from one shared scale — and it
is **not always faster**, because it adds a pass over the tile on each side.
Measure before adopting it.

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
