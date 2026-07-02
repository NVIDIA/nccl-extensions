# NCCL Extensions

NCCL Extensions is a repository of communication patterns for AI use cases,
built on top of NCCL device and host APIs. It speeds up tensor communication
for workloads like MoE token shuffling and reinforcement learning weight
rollout.

This is an evolving space, and the content here is under constant development
and subject to change. We will continue exploring it and welcome your
contributions.

## What's Inside

### [`nccl_ep/`](nccl_ep/) — Expert Parallelism (EP)
Optimized dispatch and combine primitives for Mixture-of-Experts (MoE) token
routing, built on NCCL's Device API (LSA and GIN operations).

### [`nccl_m2n/`](nccl_m2n/) — Mesh-to-Mesh (M2N)
Experimental library for resharding a tensor between two disjoint groups of
GPU processes (e.g. trainer and inference ranks) in a single, zero-copy call,
built on NCCL's window API.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) to get
started.

## License

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE.txt](LICENSE.txt) for details.

