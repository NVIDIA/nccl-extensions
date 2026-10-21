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
| `nccl/ep/` | public facade for nccl_ep, plus CUDA-specific native libraries and headers |
| `nccl/m2n/` | public facade for NCCL M2N, plus CUDA-specific native libraries and headers. See the [M2N Python guide](nccl/m2n/README.md) for API usage and examples. |
| `nccl/_extensions/` | internals shared by every extension library — the Cython bindings, `binding_dataclass`, the distribution version |

## Install

```bash
CUDA_HOME=/usr/local/cuda pip install -e python/
```

Building requires a CUDA toolkit and a Cython toolchain.

Stage native artifacts before building a distributable wheel:

```text
python/nccl/ep/lib/cu{12,13}/libnccl_ep.so
python/nccl/ep/include/**
python/nccl/m2n/lib/cu{12,13}/libnccl_m2n.so
python/nccl/m2n/include/**
```

For a complete Linux wheel build, the internal
[`build_assets/build_wheels.sh`](build_assets/README.md) script builds and
stages these artifacts in a temporary project copy before running
cibuildwheel. It does not stage artifacts into the source package tree.

Missing shared libraries emit explicit build warnings by default. Set
`NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1` to turn a missing library into a build
error; the production wheel script sets this automatically. With the default
value `0`, the resulting wheel is not self-contained and needs compatible
external libraries at runtime.

The sdist is source-only and excludes native shared libraries. Building a
wheel from it must stage the native libraries at the paths above to bundle
them, or provide compatible external libraries for runtime loading.

Pick a CUDA-variant extra to pull in the matching runtime stack (they forward
to nccl4py's `cu12` / `cu13` extras, and are mutually exclusive):

```bash
pip install -e 'python/[cu13]'
```

At runtime, the installed `cuda.bindings` major selects the matching bundled
`lib/cu12` or `lib/cu13` native libraries. NCCL EP compiles kernels at runtime
with `NCCL_EP_JIT_NVCC`, `NVCC`, or the compiler under `CUDA_HOME`; that compiler
and its headers must match the selected CUDA major.

> **Do not run Python from inside `python/`.** There is no `nccl/__init__.py`
> there, so that directory resolves only as a namespace portion and these
> modules become invisible. Always go through the editable install.

## NCCL EP handle lifecycle

`Group.create_handle(...)` remains the fused route. The v0.2 facade also
exposes the split and imported-handle C APIs, which let applications allocate
routing memory before CUDA graph capture or restore a prepared HT routing
buffer:

```python
import nccl.ep as ep
from nccl.core.typing import NcclDataType

# device_uint8_buffer must provide at least this many bytes.
num_bytes = group.handle_mem_size(ep.Layout.EXPERT_MAJOR, num_topk=-1)
handle_mem = ep.Tensor(device_uint8_buffer)

handle = group.init_handle(
    ep.Layout.EXPERT_MAJOR,
    num_topk=-1,
    handle_mem=handle_mem,
)
handle.update(topk_idx, layout_info=layout_info, stream=stream)

# After copying a prepared HT handle buffer, bind the snapshot locally.
state = ep.HandleState(
    num_tokens=num_tokens,
    num_topk=num_topk,
    topk_idx_datatype=NcclDataType.INT32,
)
restored = group.import_handle(
    ep.Layout.EXPERT_MAJOR,
    state,
    restored_handle_mem,
)
```

Combine-side MXFP8 is selected with
`CombineConfig(quantization_recipe=CombineQuantizationRecipe.MXFP8)`. It is an
HT expert-major, single-LSA-team path; input tokens are BF16 and
`CombineInputs.scales` must be `None` because NCCL EP generates E8M0 scales
internally.

## Generated bindings

Everything under `nccl/_extensions/bindings/` is generated and checked in. Do
not edit it by hand. Public builds use these checked-in sources directly.
