# Fault-Tolerant Collectives

A fault-tolerant **AllGather**, **AllReduce**, **AllToAll**, and variable-size
collective library built on the NCCL Device API. Instead of waiting
indefinitely when a rank fails, the kernels use a
**timeout + active-mask** model: each kernel is given a `timeout_us` budget and an
`active_mask_in` bitmask of expected peers, and on timeout returns early with an
`active_mask_out` reflecting only the ranks that responded — the caller gets
partial data (and a clear survivor set) instead of a deadlock.

Two transports are implemented and auto-selected from the communicator at runtime:

- **LSA (Load/Store Access)** — single NVLink domain (spans hosts on GB200 NVL over
  the NVSwitch fabric). Lower latency; the primary path.
- **GIN (GPU-Initiated Networking)** — multi-node over InfiniBand (rail dimension).

GIN currently supports AllGather and AllReduce. AllToAll and variable-size
collectives use the LSA path.

A `torch.distributed` backend (`ft_nccl`) exposes the collectives to PyTorch.
Applications use symmetric tensors for the fault-tolerant path; unsupported
operations fall back to NCCL.

## Layout

```
nccl_ft/
├── README.md
├── Makefile                     # builds ft_handle, then the ft_handle_bench C++ example
├── docs/                        # design docs (see below)
├── ft_handle/                   # the library
│   ├── ft_collective.{cu,h}     # FTCollectiveHandle C/CUDA implementation
│   ├── Makefile                 # builds libft_collective.{so,a}
│   └── python/
│       ├── ft_collective/       # installable package (registers the "ft_nccl" backend)
│       ├── pyproject.toml
│       ├── benchmark_ft_collective.py   # single-host ft vs NCCL benchmark
│       ├── bench_ft_multinode.py        # multi-node (SLURM/torchrun) benchmark
│       ├── ft_launch.py                 # single-host / multi-node launcher
│       ├── ft_correctness_test.py       # numerical correctness sweep
│       ├── test_ft_converge.py          # membership-consensus barrier agreement
│       ├── test_fault_injection.py      # failures injected at each kernel stage
│       ├── test_ft_rejoin.py            # collective re-admission (GROW)
│       └── demo/                        # runnable demos
└── ft_handle_bench/             # C++ example of measuring the ft_handle C API
```

## Build & install

The library needs an NCCL with the Device API (2.30.7 or newer, the version
vendored in `third_party/nccl`). From the repository root:

```bash
# 1. Build the vendored NCCL, then libft_collective.{so,a} and ft_handle_bench
make nccl-submodule NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"  # sm_100 for GB200
make nccl_ft.build  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
#    (or point at your own NCCL build: make nccl_ft.build NCCL_HOME=/path/to/nccl/build)

# 2. Runtime env
export NCCL_HOME=$PWD/third_party/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export FT_COLLECTIVE_LIB=$PWD/nccl_ft/ft_handle/libft_collective.so

# 3. Install the Python package
pip install nccl_ft/ft_handle/python/

# 4. Verify
python3 -c "import ft_collective; print('OK')"
```

> **Build error** `ncclGinBarrierSession::sync cannot be called with given argument
> list` → `NCCL_HOME` points at an older NCCL whose Device API lacks the
> timeout-taking `sync()` overload; build against the vendored NCCL instead.

## Usage (PyTorch)

Importing `ft_collective` registers the `ft_nccl` backend. Tensors take the FT
kernel path when they are backed by a **symmetric** allocation (`pg.empty()`); other
tensors transparently fall back to stock NCCL.

```python
import torch, torch.distributed as dist
import ft_collective                      # registers the "ft_nccl" backend
from ft_collective import get_ft_process_group

dist.init_process_group("ft_nccl", rank=rank, world_size=world, device_id=dev)
pg = get_ft_process_group()

x = pg.empty(1 << 20)                      # symmetric tensor -> FT kernels
x.fill_(1.0)
pg.allreduce([x]).wait()                   # fault-tolerant AllReduce

mask = pg.get_result_mask()                # which ranks responded this collective
```

For AllGather, use the list form with a contiguous output chunked per rank — it
takes the fast packed path:

```python
out = torch.empty(count * world, device=dev)
dist.all_gather(list(out.chunk(world)), pg.empty(count).fill_(1.0), group=pg)
```

## Fault-tolerance model

- **Timeout + active mask** — every kernel takes a `timeout_us` and an
  `active_mask_in`; it returns `active_mask_out` with only the ranks whose data
  arrived in time. A dead peer trips the timeout instead of hanging.
- **Push-based generation counter** — readiness is a monotonic on-device
  generation counter, polled on *local* memory only, so a killed peer's freed
  memory is never read (safe under real `kill`, CUDA-graph friendly).
- **`ft_converge()`** — a timeout-bounded membership barrier (SHRINK). Default
  is a store-free belief-exchange collective; a TCPStore mode is available
  (`FT_BARRIER_MODE`). The [barrier notes](docs/FT_BARRIER_COLLECTIVE.md)
  describe the agreement limit when a failure lands in the final round.
- **`ft_rejoin()`** — the GROW sibling: re-admits a stalled-but-alive rank and
  re-synchronizes its drifted generation counter **without** rebuilding the
  communicator.
- **Membership override** — `set_active_mask()` installs a framework-managed
  surviving set directly (coordinated across ranks) for app-driven health checks.

## Tests & demos

Run under the launcher (single host via `mp.spawn`, or multi-node under
`srun`/`torchrun`):

```bash
cd nccl_ft/ft_handle/python
python3 ft_correctness_test.py  --nranks 8      # every collective, eager + CUDA graph
python3 test_ft_converge.py     --nranks 8      # membership-consensus agreement
python3 test_fault_injection.py --nranks 8      # failures injected at each kernel stage
python3 test_ft_rejoin.py       --nranks 8      # drop then rejoin a stalled rank
python3 demo/demo_fault_tolerance.py --nranks 8 # end-to-end real-kill demo
```

## Benchmarks

```bash
# Single host (8 GPUs), ft vs NCCL, both collectives
python3 benchmark_ft_collective.py --nranks 8 --backend both --collective both \
    --min-count 64 --max-count 4194304 --pow2 --csv results.csv

# Multi-node via SLURM (e.g. 8 nodes x 4 GPUs = 32 ranks); ft_launch derives
# rendezvous from the SLURM env.
srun --ntasks-per-node=4 python3 bench_ft_multinode.py --collective both --backend both \
    --min-count 64 --max-count 4194304 --pow2 --csv scale_n32.csv
```

Timing is apples-to-apples (timed region = `{collective + one device sync}`, no
per-iteration host barrier); multi-node timings are MAX-reduced across ranks
(slowest-rank collective time), algbw from p50.

## C++ API example

`ft_handle_bench/` shows how to drive the `FTCollectiveHandle` C API directly from
C++ (create/destroy a handle, launch collectives, read the result mask). Build with
`make` at the top level.

## Documentation

Design docs are in [`docs/`](docs/):

| Doc | Topic |
|---|---|
| `OVERVIEW.md` | Implementation overview |
| `MULTI_CTA_PLAN.md` | Cooperative multi-CTA kernels (large sizes) |
| `REDUCE_SCATTER_PLAN.md` | Reduce-scatter AllReduce |
| `FT_BARRIER_COLLECTIVE.md` | Collective-mode membership barrier |
| `FT_TP_ONLY_FT_NCCL.md` | Routing tensor-parallel AllReduce through `ft_nccl` |
| `ALLTOALLV_PLAN.md` | Variable all-to-all |

## Key environment variables

| Env var | Default | Description |
|---|---|---|
| `FT_NCCL_MAX_COUNT` | 4194304 | Max elements per rank for the FT kernel path |
| `FT_TIMEOUT_US` | 10000000 | FT kernel timeout (µs) |
| `FT_COLLECTIVE_LIB` | (`LD_LIBRARY_PATH`) | Explicit path to `libft_collective.so` |
| `FT_MULTICTA_THRESHOLD_BYTES` | 262144 | Per-rank size at/above which LSA uses the multi-CTA kernel |
| `FT_REDUCE_SCATTER_THRESHOLD_BYTES` | 2097152 | Per-rank size at/above which AllReduce uses reduce-scatter+allgather (full set only) |
| `FT_BARRIER_MODE` | `collective` | `ft_converge()` barrier: `collective` (store-free) or `store` (TCPStore) |
