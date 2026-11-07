# NCCL ND

NCCL ND provides Ulysses sequence/head redistribution on one NCCL communicator.
It combines peer transfer and the required layout transformation:

```text
Y[d] = concat([split(X[s], P, scatter_dim)[d] for s in ranks], gather_dim)
```

The initial scope is the two directions between sequence and head sharding.
`scatter_dim` and `gather_dim` refer to physical tensor dimensions, not direction
IDs. `Shard(i) -> Shard(j)` uses `gather_dim=i, scatter_dim=j`.

| Local input -> local output | Sharding change | scatter_dim | gather_dim |
| --- | --- | --- | --- |
| `[S, H] -> [P*S, H/P]` | `Shard(0) -> Shard(1)` | 1 | 0 |
| `[S, H] -> [S/P, P*H]` | `Shard(1) -> Shard(0)` | 0 | 1 |
| `[B, S, H, D] -> [B, P*S, H/P, D]` | `Shard(1) -> Shard(2)` | 2 | 1 |
| `[B, S, H, D] -> [B, S/P, P*H, D]` | `Shard(2) -> Shard(1)` | 1 | 2 |

Here S and H are the current local input extents; P is the communicator size.
In 4-D, B and D are unchanged. Batch and source-rank ordering are preserved;
flattening `B*S` would change the operation. Other dimensions and axis pairs
return an error. Generalized tensor-axis layouts are deferred to a follow-up.

This experimental module has one implementation: CUDA pitched copies over NCCL
LSA peer mappings. It does not select among SM, CE, or TMA backends. Its public
API is provisional and does not define a common ABI for other NCCL Extensions.

## Four-rank example

For a global tensor `[R, C]`, with both dimensions divisible by four,
redistribute from row sharding (`Shard(0)`) to column sharding (`Shard(1)`).
Each `Bij` is a `[R/4, C/4]` block sent from source rank `i` to destination
rank `j`.

```text
Input: Shard(0)                           Local shape: [R/4, C]

  rank 0  [ B00 | B01 | B02 | B03 ]
  rank 1  [ B10 | B11 | B12 | B13 ]
  rank 2  [ B20 | B21 | B22 | B23 ]
  rank 3  [ B30 | B31 | B32 | B33 ]

                       │
                 All-to-all
         scatter_dim=1, gather_dim=0
                       │
                       ▼

Output: Shard(1)                          Local shape: [R, C/4]

    rank 0      rank 1      rank 2      rank 3
   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
   │  B00  │   │  B01  │   │  B02  │   │  B03  │
   │  B10  │   │  B11  │   │  B12  │   │  B13  │
   │  B20  │   │  B21  │   │  B22  │   │  B23  │
   │  B30  │   │  B31  │   │  B32  │   │  B33  │
   └───────┘   └───────┘   └───────┘   └───────┘
```

Each destination concatenates its blocks in source-rank order along axis 0.
Elements within each block keep their row-major order; the blocks themselves
are not transposed. The global tensor is unchanged. The reverse redistribution
uses `scatter_dim=0, gather_dim=1`.

## Implementation

The production implementation consists of three files:

| File | Responsibility |
| --- | --- |
| `include/nccl_nd.h` | C API and ownership/capture contract |
| `nccl_nd_layout.hpp` | Checked shape normalization and pure byte-copy geometry |
| `nccl_nd.cu` | Handle lifetime, storage checks, NCCL device barriers and CUDA copies |

Each operation enqueues, in one explicit CUDA stream:

1. An NCCL device LSA barrier to publish input readiness.
2. One `cudaMemcpy3DAsync` call per peer, directly into the final output.
3. An NCCL device LSA barrier to finish all peer reads before input reuse.

Copy geometry comes from the axis pair and the dense shapes; there is no
direction flag, payload scratch, packing kernel or device allocation. Each
barrier kernel is one CTA of 32 threads per rank. The copies are host-issued
stream operations, so this is not a single fused kernel, and CUDA may run the
local self copy on a compute channel.

## Grouped calls

`ncclNdGroupStart` and `ncclNdGroupEnd` batch several redistributions, such as
Q, K and V, under one barrier pair:

```c
ncclNdGroupStart();
ncclNdAllToAll(handle, &q_in, &q_out, 2, 1, stream);
ncclNdAllToAll(handle, &k_in, &k_out, 2, 1, stream);
ncclNdAllToAll(handle, &v_in, &v_out, 2, 1, stream);
ncclNdGroupEnd();  // entry barrier, 3*P copies, exit barrier
```

Each call is validated and its copy descriptors are resolved when it is
recorded. The outermost `ncclNdGroupEnd` enqueues one entry barrier, every
recorded copy and one exit barrier. Copy order and synchronization within a
group are implementation details, not API guarantees; copies are currently
issued peer-major (for each peer in the rotated order `(rank + i) % P`, that
peer's copy of every call). `NCCL_ND_COPY_ORDER=tensor`, read at
`ncclNdCreate`, switches a handle to tensor-major order for benchmarking and
debugging only; it is not part of the API, and any value other than `peer` or
`tensor` prints a warning and keeps the default.

On 8x H100 (BF16, Q/K/V with Hq/Hkv = 64/8 at 1K-128K tokens, and 64/64 at
32K), a peer-major group was 1.06x-1.41x faster than three separate calls;
[`bench/qkv_bench.cu`](#benchmark) reproduces the comparison.

`include/nccl_nd.h` holds the full contract of each function. In summary:

- **Scope and binding.** Group state is local to the host thread. The first
  recorded call binds the handle, the stream and the capture state it sees
  (outside capture, or one capture session); later calls and the outermost end
  must match, and the outermost end runs with the handle's device current.
- **Errors.** A rejected `ncclNdAllToAll`, `ncclNdCreate` or `ncclNdDestroy`
  invalidates the whole local group. The first error is kept, and the
  outermost end returns it, enqueues nothing and clears the group. Start and
  end stay balanced after errors. A rejection submits nothing on this rank
  only; it is not a distributed rollback, so every rank must record the same
  calls in the same order.
- **Dependencies.** Grouped calls must be independent: no output may overlap
  another call's input or output (not checked), and no call may consume
  another's output.
- **Native NCCL groups.** ND and native NCCL groups keep independent state and
  do not group each other's operations.

## Requirements and supported scope

- Linux, C++17, CUDA Toolkit and runtime 12.5+, and a driver compatible with
  that CUDA/NCCL stack and the deployment GPU.
- NCCL headers and runtime with the device API, window API and communicator
  property query used here. Validated with NCCL 2.30.7; runtime NCCL must be
  at least the compile-time version.
- The entire communicator forms one LSA team, with at least two ranks. The
  current CUDA device must match the communicator.
- FP16, BF16 or FP32; dense row-major 2-D or 4-D tensors with the axis pairs
  listed above. Positive extents and equal splits are required.
- The output shape divides the scatter extent by the communicator size and
  multiplies the gather extent by it. All other extents are unchanged.
- Input/output pointers and the normalized per-peer copy width are 16-byte
  aligned. Both source and destination pitches must fit the device's maximum pitch.

Each copy is a pitched 3-D `cudaMemcpyDeviceToDevice` copy from the peer's LSA
mapping of the input to the local output.

Other dimensions and axis pairs, non-dense strides, uneven splits, in-place
operation and network fallback are unsupported. Explicit strides must match
canonical dense storage. Missing requirements return an error; there is no
fallback.

## Memory and execution contract

The caller owns the communicator, windows and storage. Register the input
collectively with `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`
(typically on `ncclMemAlloc` storage) and pass its window and byte offset; pass
the output as a plain device pointer. The library checks input window ranges
and local input/output overlap; matching metadata and registration across ranks
are the caller's responsibility.

Create and destroy handles outside capture. Use an explicit nondefault stream,
serialize calls and replays that share a handle, and keep the handle,
communicator, windows and storage alive until the last replay completes.
Captured addresses and shapes are fixed. Stream completion makes the output
ready and the input reusable.

A rejected call enqueues nothing on this rank. A submission failure
(`ncclUnhandledCudaError`) can leave partial collective work and is fatal for
the handle and the communicator; the failing rank prints `NCCL ND WARN ...` to
stderr. Recover on every rank:

1. Stop using the communicator and notify the other ranks out of band; peers
   may be stuck in a barrier, so a stream synchronize may never return.
2. Unless every rank is known to have failed the same way, call `ncclCommAbort`
   and abandon the handle and windows without calling `ncclNdDestroy`.
3. If every rank failed the same way, and the stream then synchronizes without
   error and the CUDA context is still usable, destroy the handle, deregister
   its windows, then abort the communicator. After a sticky CUDA error the
   context is unusable: abandon everything and restart the process.
4. Create a new communicator and handle.

After successful work, synchronize before destroying the handle, deregistering
windows or freeing memory.

## Build

The repository's default `make` builds ND alongside EP and M2N, and `make clean`
cleans all three modules. To build ND alone from the repository root, using an
existing compatible NCCL build:

```sh
make nccl_nd.build CUDA_HOME=/usr/local/cuda NCCL_HOME=/path/to/nccl
```

Set `NVCC_GENCODE` for the deployment GPU when using Make. The default targets
SM90, adding SM100 when the compiler supports it (CUDA 12.8+). CMake uses the
same defaults. Explicit architecture overrides must be supported by the selected
compiler. CMake is also available, for example:

```sh
cmake -S nccl_nd -B build-nd -DNCCL_HOME=/path/to/nccl \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-nd -j
```

No Python, FlashInfer, MPI or performance harness is a library dependency.
Performance experiments and their historical source remain outside this module;
old standalone measurements are not validation of this library revision.

## Tests

`nd_test` needs one node with two or more GPUs. It compares outputs bitwise
with an independently computed host reference: FP16/BF16/FP32, both
directions, 2-D and 4-D (B > 1) tensors, nonzero window offsets, forward +
inverse round trip, Q/K/V of different sizes, repeated eager calls and CUDA
Graph replays with new inputs, and rejected misuse. For groups it compares
grouped and separate Q/K/V, covers nested groups, repeated grouped calls,
grouped capture/replay with new inputs and tensor-major copy order, and checks
the group contract: binding, sticky first error, Create/Destroy rejection,
capture-boundary crossing and balanced start/end after errors.

`nd_fault_test` compiles the library source with test-only hooks that fail the
entry barrier launch, a copy after earlier copies were submitted, or the exit
barrier launch, on every rank. It checks the returned error and the cleared
group, then runs step 3 of the recovery procedure above. The same hooks record
the issued copy order, which it checks for both `NCCL_ND_COPY_ORDER` values.

```sh
make -C nccl_nd tests NCCL_HOME=/path/to/nccl      # or CMake with -DNCCL_ND_BUILD_TESTS=ON
export LD_LIBRARY_PATH=$PWD/build/lib:/path/to/nccl/lib:$LD_LIBRARY_PATH
nccl_nd/tests/run_tests.sh 2 8                     # rank counts; default: all GPUs
```

`run_tests.sh` starts one process per GPU for `nd_test` and, when built,
`nd_fault_test`; no MPI or job scheduler is needed.

## Benchmark

`bench/qkv_bench.cu` measures one Q/K/V set three ways on the same
communicator, stream and buffers: three separate `ncclNdAllToAll` calls, the
same calls grouped with tensor-major copy order, and grouped with the default
peer-major order. It links the library like any consumer.

- Sequence-sharded to head-sharded redistribution
  `[B, T/P, H, D] -> [B, T, H/P, D]` of Q (Hq heads), K and V (Hkv heads
  each); BF16, B=1, D=128. Inputs are symmetric windows; outputs are
  `ncclMemAlloc` storage.
- Each mode is captured once as a CUDA Graph of 20 Q/K/V sets; 5 warmup
  replays, then 15 samples per mode. A sample is one replay timed with CUDA
  events, divided by 20; its latency is the slowest rank's.
- Modes rotate order between samples. Inputs change every sample, and every
  sample's outputs are checked against an independent per-element reference.
  Each mode also runs once eagerly and is checked before capture.
- The table reports the median over samples and the min-max range.

```sh
make -C nccl_nd bench NCCL_HOME=/path/to/nccl      # or CMake with -DNCCL_ND_BUILD_BENCH=ON
export LD_LIBRARY_PATH=$PWD/build/lib:/path/to/nccl/lib:$LD_LIBRARY_PATH
nccl_nd/bench/run_bench.sh 8                       # --smoke: first case, 3 samples
```

The benchmark prints the GPU, NCCL, CUDA runtime and driver versions with a
table of the three modes.
