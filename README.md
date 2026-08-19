# NCCL Extensions

NCCL Extensions is a repository of communication patterns for AI use cases,
built on top of NCCL device and host APIs. It speeds up tensor communication
for workloads like MoE token shuffle and reinforcement learning weight rollout.

This is an evolving space, and the content here is under constant development
and subject to change. We will continue exploring it and welcome your
contributions.

## What's Inside

### [`nccl_ep/`](nccl_ep/) — Expert Parallelism
Optimized dispatch and combine primitives for Mixture-of-Experts (MoE) token
routing, built on NCCL's Device API (LSA and GIN operations).

### [`nccl_m2n/`](nccl_m2n/) — Mesh-to-Mesh Rollout
Experimental library for resharding a tensor between two disjoint groups of
GPU processes (e.g. trainer and inference ranks) in a single, zero-copy call,
built on NCCL's window API.

### [`python/`](python/) — Python bindings
Python package (`nccl-extensions`) providing Pythonic wrappers for `nccl_ep`
and `nccl_m2n` as `nccl.ep` and `nccl.m2n`. See
[`python/README.md`](python/README.md) for details.

## Getting Started

This repo vendors NCCL as a git submodule. Clone with:

```bash
git clone --recursive <repo-url>
```

(or initialize and build the vendored NCCL after a normal clone):

```bash
make nccl-submodule
```

By default, NCCL keeps its native `third_party/nccl/build` output, while NCCL
EP and NCCL M2N place libraries under `build/lib` and headers under
`build/include` at the repository root.

Build either extension from the repository root. Both targets install their
artifacts into the shared build tree:

```bash
make nccl_ep.build   # build/{lib,include,...}
make nccl_m2n.build  # build/{lib,include,...}
```

Unless `NCCL_HOME` is set, plain `make` initializes and builds the vendored
NCCL before building both extension libraries. `make clean` cleans both
extensions without touching NCCL. Each library has an independent build-root
override, while `BUILDDIR` changes the shared default for both extensions:

```bash
make nccl-submodule NCCL_BUILDDIR=/path/to/nccl/build
make BUILDDIR=/path/to/extensions/build
make nccl_ep.build NCCL_EP_BUILDDIR=/path/to/nccl-ep/build
make nccl_m2n.build NCCL_M2N_BUILDDIR=/path/to/nccl-m2n/build
```

The root Makefile passes `NCCL_BUILDDIR` to NCCL as its native `BUILDDIR`
variable. See each subproject's README for library-specific build options.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) to get
started.

## License

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE.txt](LICENSE.txt) for details.
