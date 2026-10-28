# AWS EFA

NCCL EP moves inter-node traffic through NCCL's GPU-Initiated Networking (GIN)
API. On AWS Elastic Fabric Adapter (EFA) the GIN backend is provided by the
[aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) plugin, and the
configuration validated with NCCL EP is NCCL's **GIN CPU proxy**
(`NCCL_GIN_TYPE=2`): the GPU queues each network operation in host memory and a
CPU thread inside NCCL posts it to the NIC through the plugin and libfabric's
`efa` provider. The `NCCL_GIN_TYPE=3` (GDAKI) setting recommended under
[Environment Setup](../../README.md#environment-setup) was not validated on
EFA; see [Limitations](#limitations).

This page lists what the proxy configuration needs, how to set it up, how to
confirm that NCCL EP traffic is on EFA, and the exact configuration the
validation ran.

## Requirements

Everything NCCL EP itself needs still applies ([Prerequisites](../../README.md#prerequisites),
[JIT Kernel Compilation](jit.md)). On EFA, in addition:

| Component | Version | Notes |
|-----------|---------|-------|
| aws-ofi-nccl | 1.21.0+ | `libnccl-net-ofi.so` is both the NET and the GIN plugin. GIN support arrived in 1.21 ([GIN getting started](https://github.com/aws/aws-ofi-nccl/blob/68b016a24a4f0e69f6688d0e10ed886657af2d4a/doc/gin-getting-started.md)) |
| NCCL | 2.31.2+ | The GIN floor of aws-ofi-nccl. NCCL EP must be built against a NCCL the runtime accepts; see [NCCL EP compatibility with newer NCCL runtimes](../../README.md#nccl-ep-compatibility-with-newer-nccl-runtimes) |
| GDRCopy | 2.5+ | Both the runtime (`libgdrapi`) and the `gdrdrv` kernel module, in proxy mode too: aws-ofi-nccl's GIN initialization loads `libgdrapi.so` and opens a GDRCopy handle, and fails without them |
| EFA | kernel driver on the host, user space from the [EFA installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html) | The installer provides libfabric, aws-ofi-nccl and Open MPI. In a container, install the user space only (`--skip-kmod`) |
| Instance type | `p5en.48xlarge` | Validated: 8 H200 GPUs and 16 EFA devices per node. Other types are discussed under [Limitations](#limitations) |

Containers need `/dev/gdrdrv` and the EFA devices exposed. With the NVIDIA
container toolkit, `NVIDIA_GDRCOPY=enabled` in the container environment exposes
`/dev/gdrdrv` (the NVIDIA Kubernetes device plugin installs that path with
`gdrcopyEnabled=true`); the AWS EFA Kubernetes device plugin exposes the EFA
devices as the `vpc.amazonaws.com/efa` resource. The validated workers also had
the `IPC_LOCK` capability and a memory-backed `/dev/shm`.

## Environment setup

Set the GIN type and name the plugin. The validated run set both explicitly. The
aws-ofi-nccl guide notes that NCCL 2.31 defaults to the proxy for EFA; setting
`NCCL_GIN_TYPE=2` pins that choice and makes it visible in the log as
`NCCL_GIN_TYPE set by environment to 2`.

```bash
export NCCL_GIN_TYPE=2                                               # GIN CPU proxy
export NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so  # aws-ofi-nccl, NET + GIN
```

`/opt/amazon/ofi-nccl/lib` is where the EFA installer places the plugin; a
plugin built from source lives wherever it was installed.

The validated run also set the following. They are part of the validated
configuration, not known requirements:

```bash
export NCCL_GIN_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so  # the same library
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME=^lo,docker
```

The JIT variables from [JIT Kernel Compilation](jit.md) apply unchanged. The
validated container set `NCCL_EP_HOME` to its `BUILDDIR`, `NCCL_HOME` and
`CUDA_HOME` to the installed NCCL and CUDA, and `NCCL_EP_JIT_CACHE_DIR` to a
writable directory.

With an ssh-based `mpirun`, the ranks do not inherit the container image's
`ENV`; pass every variable with `-x`. To keep the EFA counters attributable to
NCCL EP, keep MPI's own traffic on TCP (`--mca pml ob1 --mca btl self,vader,tcp`).

## Checking the transport

Run with `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,ENV,NET`. Every rank
should log these lines (16 NICs on `p5en.48xlarge`):

```text
NCCL INFO NCCL_GIN_TYPE set by environment to 2.
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
NCCL INFO RMA/Plugin: Assigned plugin Libfabric to comm
NCCL INFO Using network Libfabric
NCCL INFO GIN Proxy will not be using GDRCopy
```

`GIN Proxy will not be using GDRCopy` is expected: NCCL's proxy copies through
GDRCopy only when `NCCL_GDRCOPY_ENABLE=1` (default 0). GDRCopy is still needed
for the plugin's GIN initialization. A rank that logs `NET/Socket` is not on EFA.

The log shows which transport NCCL selected, not that bytes moved. For that,
read the EFA hardware counters before and after a run, on every node:

```bash
for f in /sys/class/infiniband/rdmap*/ports/1/hw_counters/rdma_write_bytes; do
  echo "$(hostname) $f $(cat "$f")"
done
```

The increase summed over a node's devices is the wire volume. `ep_bench`'s own
`rdma_bw` and `rdma_send` figures count the logical payload (each destination
copy), which was about 3.2x the bytes the NICs carried in the validated run,
because a token crosses to the other node once and is forwarded over NVLink
there. Containers do not always see `hw_counters`; if the read prints
`No such file or directory`, read the same files on the node.

## Running `ep_test` and `ep_bench`

The validation used NCCL EP's own programs, launched with Open MPI, one rank per
GPU on two nodes. `ep_test` runs the built-in dispatch and combine checks;
`ep_bench --validate` times a fixed shape and checks its output at the end. Both
exit non-zero on a wrong result.

```bash
MPIRUN="mpirun -np 16 -N 8 --bind-to none \
  --mca pml ob1 --mca btl self,vader,tcp \
  -x PATH -x LD_LIBRARY_PATH -x NCCL_EP_HOME -x NCCL_HOME -x CUDA_HOME \
  -x NCCL_GIN_TYPE -x NCCL_NET_PLUGIN -x NCCL_GIN_PLUGIN \
  -x FI_PROVIDER -x FI_EFA_USE_DEVICE_RDMA -x NCCL_SOCKET_IFNAME \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_EP_JIT_CACHE_DIR"

# Correctness: HIGH_THROUGHPUT algorithm, FLAT layout.
${MPIRUN} ep_test -a ht -L fl

# Timing with validation. 61 passes: 10 warm-up, 50 timed, 1 validation.
${MPIRUN} ep_bench --algorithm ht --layout flat --tokens 4096 --hidden 7168 \
  --top-k 8 --experts 256 --validate
```

A passing `ep_test` prints, per rank, `HIGH_THROUGHPUT Dispatch flow passed
successfully`, `Combine verification PASSED!` and `[MPI Rank <n>] Success`. A
passing `ep_bench` ends with `Global validation: Dispatch=PASSED, Combine=PASSED`.

For the `ep_bench` shape above, with 256 experts split across two nodes and 8
experts chosen per token, nearly every token has at least one expert on the other
node, and NCCL EP sends each such token across once. A rank's 4096 tokens of 7168
BF16 values put about 58.5 MB on the wire per dispatch, and combine returns the
same amount, so the counters should rise by about 117 MB per rank per pass.

## Validated configuration

The validation is a two-node run packaged as a container recipe, with a pass/fail
checker, in awslabs/awsome-distributed-ai:
[`micro-benchmarks/expert-parallelism/nccl-ep-benchmark`](https://github.com/awslabs/awsome-distributed-ai/tree/main/micro-benchmarks/expert-parallelism/nccl-ep-benchmark)
(merged as [awslabs/awsome-distributed-ai#1283](https://github.com/awslabs/awsome-distributed-ai/pull/1283)).
It builds from a public NGC base and public sources, and launches one Kubeflow
MPIJob whose exit status is the result.

| Component | Validated |
|-----------|-----------|
| nccl-extensions | `901c4e65` (NCCL EP 0.2.0), built with `MPI=1` for sm_90, sm_100 and sm_103 |
| NCCL | 2.32.3 (`libnccl2` / `libnccl-dev` `2.32.3-1+cuda13.4`, the build and the runtime NCCL) |
| CUDA | 13.0, base image `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` |
| EFA installer | 1.50.0: libfabric 2.6.0amzn1.0, aws-ofi-nccl 1.21.1, Open MPI 4.1.7 |
| GDRCopy | v2.5.2 user space; `gdrdrv` from the host |
| Nodes | 2 x `p5en.48xlarge` (8 x H200, 16 EFA devices each), 16 ranks, Amazon EKS, Kubeflow MPI Operator 0.6.0 |

Results, in four launches of the same image:

- `ep_test -a ht -L fl`: `Success` on 16 of 16 ranks in every launch.
- `ep_bench` (shape above): `Dispatch=PASSED, Combine=PASSED` in every launch.
- Transport: every rank logged the lines under [Checking the transport](#checking-the-transport);
  no rank used `NET/Socket`; no `NCCL WARN` lines.
- Counters: `rdma_write_bytes` rose on every rank over `ep_bench`,
  119.5 MB per rank per pass against the expected 117 MB.

The timing `ep_bench` reported is descriptive, not a performance
characterization: CUDA-event averages across the 16 ranks from one launch, with
NCCL's default single proxy thread, were dispatch total 7790 us and combine total
2531 us. With `NCCL_GIN_PROXY_NTHREADS=2`, one launch gave dispatch 4621 us with
the same bytes on the wire; 4 threads gave the same as 2, and combine stayed at
2531-2599 us in all of them. Repeat on fresh nodes before comparing numbers.

## Limitations

- **GIN proxy only.** `NCCL_GIN_TYPE=3` (GDAKI) was not validated on EFA.
  aws-ofi-nccl's GPU-initiated backend, EFA-GDA (`NCCL_GIN_TYPE=5`), does not
  support strong or VA signals, which NCCL EP's device communicator requires:
  with NCCL's symmetric GIN kernels left enabled (the default),
  `ncclEpCreateGroup` fails with
  `GIN strong signals are required, but the GIN plugin does not support them`.
  The `NCCL_SYM_GIN_KERNELS_ENABLE=0` setting from the aws-ofi-nccl guide was
  not run.
- **High-Throughput with the FLAT layout only.** Low-Latency mode and the other
  layouts were not exercised.
- **Two `p5en.48xlarge` nodes only.** More nodes were not run. 32-NIC
  `p5.48xlarge` hits [NVIDIA/nccl#2160](https://github.com/NVIDIA/nccl/issues/2160)
  during topology detection with NCCL 2.32.3; the fix landed on NCCL's
  development branch after that release and was not tested here. Two
  `p6-b200.48xlarge` nodes passed `ep_test` and `ep_bench --validate` with an
  earlier revision of the recipe's scripts (built for sm_100), not the committed
  manifest.
- **Library drift.** `ep_test.cu` is unchanged since `901c4e65`; `ep_bench.cu`
  gained options the command above does not use. The library itself has changed
  since, and the configuration was not re-run on later commits.
- **Frameworks.** A framework process that uses NCCL EP must load the same
  `libnccl.so.2` that NCCL EP was built against. That is not covered here; a
  TensorRT-LLM example over the same transport, on an older NCCL EP, is proposed
  in [awslabs/awsome-distributed-ai#1240](https://github.com/awslabs/awsome-distributed-ai/pull/1240).
