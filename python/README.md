# nccl-extensions (Python)

Python bindings for the [nccl-extensions](../README.md) communication
libraries.

## Package layout

This package installs into the **`nccl` namespace**, so the import path is
`nccl.ep` rather than a separate top-level package:

```python
import nccl.ep as ep
```

It contributes exactly two directories to that namespace, and no
`nccl/__init__.py`:

| path | contents |
| --- | --- |
| `nccl/ep/` | public facade for nccl_ep, plus `lib/libnccl_ep.so` and headers |
| `nccl/_extensions/` | internals shared by every extension library — the Cython bindings, `binding_dataclass`, the distribution version |

`nccl.m2n` will join as a second facade; its bindings go into the shared
`nccl/_extensions/bindings/` rather than a second copy of the machinery.

## Status

`nccl.ep` is complete: the Cython bindings under
`nccl/_extensions/bindings/` plus the Python facade (group/handle management,
dispatch/combine, tensors, torch interop) under `nccl/ep/`.

`nccl.m2n` bindings are not started here — they are being developed separately
and will be migrated into this package once ready.

## Install

```bash
CUDA_HOME=/usr/local/cuda pip install -e python/
```

Building requires a CUDA toolkit and a Cython toolchain. `libnccl_ep.so` must
be staged at `python/nccl/ep/lib/` before building a wheel; without it the
build still succeeds but `import nccl.ep` fails at runtime.

Pick a CUDA-variant extra to pull in the matching runtime stack (they forward
to nccl4py's `cu12` / `cu13` extras, and are mutually exclusive):

```bash
pip install -e 'python/[cu13]'
```

> **Do not run Python from inside `python/`.** There is no `nccl/__init__.py`
> there, so that directory resolves only as a namespace portion and these
> modules become invisible. Always go through the editable install.

## Regenerating the bindings

Everything under `nccl/_extensions/bindings/` is generated from
`nccl_ep/include/` and is checked in. Do not edit it by hand — re-run
[`build_assets/generate_cython.py`](build_assets/README.md) after changing the
nccl_ep public headers and commit the result.
