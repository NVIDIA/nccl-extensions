# Build Assets for the nccl-extensions Python package

> **⚠️ Internal Use Only**: This directory is **not released** to the public
> nccl-extensions GitHub repository.

## Purpose

The Cython bindings under `python/nccl/_extensions/bindings/` are **generated**,
not hand-written. This directory holds the tooling that generates them: the
cybind config, the Cython templates, and the driver script. The M2N low-level
surface is generated into the shared `nccl._extensions.bindings` package while
its public facade remains under `python/nccl/m2n/`.

The generated `.pyx`/`.pxd` files are checked into the repository so users can
build the package without internal tooling; the generation tooling here stays
internal.

## Why not public?

Generation relies on [cybind](https://gitlab-master.nvidia.com/leof/cybind), an
internal NVIDIA tool. To keep the package buildable without access to it, we:

1. Run `generate_cython.py` when a bound library's headers change
2. Commit the generated Cython sources
3. Exclude this `build_assets/` directory from public releases

Ported from nccl4py's `bindings/nccl4py/build_assets/`, trimmed to the targets
this repo owns. nccl4py's `generate_header.py` (which flattens the NCCL *device*
API headers) has no nccl-extensions equivalent and was not ported.

## Prerequisites

- CUDA installation, with `CUDA_HOME` or `CUDA_PATH` set (cybind needs `cuda.h`)
- [`uv`](https://docs.astral.sh/uv/) on `PATH`
- SSH access to the cybind repository (unless `--cybind-path` points at a local
  checkout)

## Usage

Re-run whenever `nccl_ep/include/nccl_ep.h`, `nccl_m2n/src/nccl_m2n.h`, their
templates/configs, or the pinned NCCL header changes:

```bash
python3 build_assets/generate_cython.py --verbose
```

The script clones cybind at the pinned `CYBIND_COMMIT`, stages our configs,
headers and templates into its `assets/`, then regenerates the
complete shared `python/nccl/_extensions/bindings/` package transactionally.
Commit the resulting diff.

See `generate_cython.py --help` for all options; `--cybind-path` reuses a local
cybind checkout instead of cloning.

## Headers

Two different policies, on purpose:

- **`nccl_ep` headers are not checked in here.** They live in this repo at
  `nccl_ep/include/`, and are staged straight from there. The bound version is
  read from `NCCL_EP_{MAJOR,MINOR,PATCH}` in `nccl_ep.h` and stamped into
  `cybind/configs/nccl_ep.cybind.yaml`, so bindings can never drift from the
  header they were generated against.
- **Headers this repo does not own are pinned** under
  `cybind/headers/<libname>/<version>/`, matching cybind's own layout.
  Currently just `nccl.h`, at the version in `NCCL_PIN`; it fixes the NCCL core
  ABI the generated bindings were built against. Bumping it means dropping the
  new `nccl.h` in place, updating `NCCL_PIN`, and regenerating.

## Contents

- `generate_cython.py` — driver: stages assets, runs cybind, installs output
- `cybind/configs/{nccl_ep,nccl_m2n}.cybind.yaml` — cybind configs for the
  bound libraries (including `AUTO_LOWPP_CLASS` struct overrides)
- `cybind/templates/nccl/_extensions/bindings/` — Cython templates, plus the
  static files (`_internal/utils.{pxd,pyx}`, `__init__.py`) that cybind does not
  process and `generate_cython.py` copies verbatim
- `cybind/headers/` — pinned third-party headers (see above)

## Generated binding conventions

All bound libraries share `nccl/_extensions/bindings/`, so common generated
support such as `_internal/utils.pyx` and `_binding_helpers.py` is built and
shipped once.

A library may provide dedicated templates when its ABI or native-loader
contract differs from the common case. Keep those differences inside the
generated binding layer:

- add the library configuration under `cybind/configs/`;
- add any library-specific templates under
  `cybind/templates/nccl/_extensions/bindings/`;
- keep public, framework-facing APIs in that library's facade package;
- preserve actionable loader errors and ensure native symbols resolve from the
  intended library handle.

After changing a bound header, configuration, or template, regenerate the
complete bindings package and commit the generated diff.
