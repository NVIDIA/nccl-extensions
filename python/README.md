# nccl-extensions (Python)

Python bindings for the [nccl-extensions](../README.md) communication
libraries.

## Status

`nccl_extensions.ep` is complete: the Cython bindings under
`nccl_extensions/bindings/` plus the Python facade (group/handle management,
dispatch/combine, tensors, torch interop) under `nccl_extensions/ep/`.

`nccl_extensions.m2n` bindings are not started here — they are being
developed separately and will be migrated into this package once it's ready.

## Install

```bash
CUDA_HOME=/usr/local/cuda pip install -e python/
```

Building requires a CUDA toolkit and a Cython toolchain. `libnccl_ep.so` must
be staged at `python/nccl_extensions/ep/lib/` before building a wheel; without
it the build still succeeds but `import nccl_extensions.ep` fails at runtime.

Pick a CUDA-variant extra to pull in the matching runtime stack (they forward
to nccl4py's `cu12` / `cu13` extras, and are mutually exclusive):

```bash
pip install -e 'python/[cu13]'
```

## Regenerating the bindings

Everything under `nccl_extensions/bindings/` is generated from
`nccl_ep/include/` and is checked in. Do not edit it by hand — re-run
[`build_assets/generate_cython.py`](build_assets/README.md) after changing the
nccl_ep public headers and commit the result.
