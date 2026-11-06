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

`cudaMemcpy3DAsync` is a host-side CUDA Runtime API call. NCCL Device API calls
execute inside the two barrier kernels; payload copies are separate stream
operations. Graph capture records this sequence for replay.

Copy geometry is derived from the supported sequence/head pair and the dense
input/output shapes. There is no forward/reverse layout flag. Copy width covers
one peer's contiguous head slice (including D in 4-D), height covers its sequence
slice, and depth is B (or 1 in 2-D). Source and destination pitches and plane
heights come from their respective shapes. Destination rank selects the input
split; source rank selects the output concat slice.

There is no payload scratch, packing kernel, input-to-window staging, or
device allocation inside `ncclNdAllToAll`. Host copy descriptors are temporary
and CUDA consumes them when the call returns (or captures their values into the
graph).

Each synchronization kernel launches one CTA of 32 threads per rank and uses
barrier slot 0. Stream ordering places the full copy sequence between the two
rank barriers. Serialized calls and graph replays reuse the slot with advancing
epochs. There is no public CTA setting and no payload CTA grid.

CUDA can execute the local self copy on a compute channel. This path is neither
a single fused kernel nor a zero-SM collective; API choice alone does not prove
the physical engine used for every copy.

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

The caller owns the communicator and tensor allocations. Register the input
collectively with `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` (typically
using `ncclMemAlloc` storage). Pass its window and byte offset in the input
descriptor; its `data` field must be null. A successful ordinary registration or
allocation alone is not the symmetric-window contract.

The output descriptor takes a device pointer and a null window. The output may
itself have a separately registered window for a subsequent inverse operation.
The caller guarantees output capacity, matching metadata and registration
contracts across ranks. The library checks input window ownership/ranges and
local input/output overlap, but cannot infer every caller-managed alias or verify
collective metadata agreement.

Create a handle with `ncclNdCreate(&handle, comm)` and destroy it with
`ncclNdDestroy(handle)`, both outside capture. Pass `scatter_dim` and
`gather_dim` directly to `ncclNdAllToAll`; no config struct or public CTA
setting is needed.

Use an explicit nondefault stream and serialize operations/replays sharing a
handle. Keep the handle, communicator, windows, and allocations alive until the
final replay has completed. The function
consumes host descriptors before returning. Captured addresses and shapes remain
fixed; changing host metadata does not update a graph. Stream completion makes
the output ready and the input reusable, including reads performed by peers.

An enqueue error can occur after partial collective submission. Do not retry it
as an eager operation or assume transactional rollback. Such an error is fatal
for the communicator: abort it with `ncclCommAbort` on every rank. The failing
rank prints the CUDA error and, for a copy, the peer to stderr as
`NCCL ND WARN ...`; the return code stays `ncclUnhandledCudaError`. After successful work,
synchronize before destroying the handle, deregistering windows or freeing memory.

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
Graph replays with new inputs, and rejected misuse.

```sh
make -C nccl_nd tests NCCL_HOME=/path/to/nccl      # or CMake with -DNCCL_ND_BUILD_TESTS=ON
export LD_LIBRARY_PATH=$PWD/build/lib:/path/to/nccl/lib:$LD_LIBRARY_PATH
nccl_nd/tests/run_tests.sh 2 8                     # rank counts; default: all GPUs
```

`run_tests.sh` starts one `nd_test` process per GPU; no MPI or job scheduler
is needed.
