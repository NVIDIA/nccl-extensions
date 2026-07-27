# Build Assets for the nccl-extensions Python package

> **⚠️ Internal Use Only**: This directory is **not released** to the public
> nccl-extensions GitHub repository.

## Purpose

The Cython bindings under `python/nccl_extensions/bindings/` are **generated**,
not hand-written. This directory holds the tooling that generates them: the
cybind config, the Cython templates, and the driver script.

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

Re-run whenever `nccl_ep/include/nccl_ep.h` (or `ep_enums.h`) changes, or when a
template or the cybind config here changes:

```bash
python3 build_assets/generate_cython.py --verbose
```

The script clones cybind at the pinned `CYBIND_COMMIT`, stages our config,
headers, and templates into its `assets/`, runs it, and copies the result over
`python/nccl_extensions/bindings/`. The existing bindings directory is backed up
and restored if generation fails. Commit the resulting diff.

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
- `cybind/configs/nccl_ep.cybind.yaml` — cybind config for `nccl_ep` (which
  functions/types to bind, and the `AUTO_LOWPP_CLASS` struct overrides)
- `cybind/templates/nccl_extensions/bindings/` — Cython templates, plus the
  static files (`_internal/utils.{pxd,pyx}`, `__init__.py`) that cybind does not
  process and `generate_cython.py` copies verbatim
- `cybind/headers/` — pinned third-party headers (see above)

## Adding a library

`nccl_m2n` bindings are not migrated yet. When they are:

1. Add `cybind/configs/nccl_m2n.cybind.yaml` with
   `module: nccl_extensions.bindings.nccl_m2n`
2. Add the per-library templates (`nccl_m2n.pyx/pxd`, `cynccl_m2n.pyx/pxd`,
   `_internal/nccl_m2n.pxd`, `_internal/nccl_m2n_linux.pyx`) under
   `cybind/templates/nccl_extensions/bindings/`
3. Add a `Target` factory and list it in `TARGETS` in `generate_cython.py`
4. Add `"nccl_m2n"` to `LIBNAMES` in `python/setup.py`
