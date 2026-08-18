# nccl-extensions (Python)

Python bindings for the [nccl-extensions](../README.md) communication
libraries.

## Package layout

This package installs into the **`nccl` namespace**, so the import paths are
`nccl.ep` and `nccl.m2n`:

```python
import nccl.ep as ep
import nccl.m2n as m2n
```

It contributes exactly three directories to that namespace, and no
`nccl/__init__.py`:

| path | contents |
| --- | --- |
| `nccl/ep/` | public facade for nccl_ep, plus `lib/libnccl_ep.so` and headers |
| `nccl/m2n/` | public facade for NCCL M2N, plus `lib/libnccl_m2n.so` and headers. See the [M2N Python guide](nccl/m2n/README.md) for API usage and examples. |
| `nccl/_extensions/` | internals shared by every extension library — the Cython bindings, `binding_dataclass`, the distribution version |

## Install

```bash
CUDA_HOME=/usr/local/cuda pip install -e python/
```

Building requires a CUDA toolkit and a Cython toolchain.

Stage native artifacts before building a distributable wheel:

```text
python/nccl/ep/lib/libnccl_ep.so
python/nccl/m2n/lib/libnccl_m2n.so
python/nccl/m2n/include/nccl_m2n.h
```

Missing shared libraries emit explicit build warnings. The resulting wheel is
not self-contained and needs compatible external libraries at runtime.

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
[`build_assets/generate_cython.py`](build_assets/README.md) after changing a
public header, config, or template and commit the result.
