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
