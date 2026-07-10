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

## Getting Started

This repo vendors NCCL as a git submodule. Clone with:

```bash
git clone --recursive <repo-url>
```

(or `git submodule update --init` after a normal clone). See each
subproject's README for build instructions.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) to get
started.

## License

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE.txt](LICENSE.txt) for details.

