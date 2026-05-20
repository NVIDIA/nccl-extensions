#!/usr/bin/env bash
# NCCL EP CI: run ep_test and ep_bench inside a Slurm allocation (srun --mpi=pmix, SRUN_MPI-style).
# Set OMPI_MCA_* in CI for EOS TCP BTL (e.g. btl_tcp_if_include) when applicable.
# NCCL_EP_BENCH_HT=1 enables ep_bench --algorithm high-throughput (optional; not all clusters pass it).

: "${NCCL_HOME:?NCCL_HOME must be set}"

EP_TEST="${NCCL_HOME}/test/nccl_ep/ep_test"
EP_BENCH="${NCCL_HOME}/test/nccl_ep/ep_bench"
if [[ ! -x "$EP_TEST" ]]; then
  echo "ERROR: ep_test not found or not executable: $EP_TEST" >&2
  exit 1
fi
if [[ ! -x "$EP_BENCH" ]]; then
  echo "ERROR: ep_bench not found or not executable: $EP_BENCH" >&2
  exit 1
fi

: "${NP:?NP must be set}"
# Set SLURM_PARTITION in CI for IPP6; leave unset on EOS (default partition / allocation).
PARTITION="${SLURM_PARTITION:-}"
TIME="${NCCL_EP_TEST_SLURM_TIME:-00:30:00}"
BENCH_TIME="${NCCL_EP_BENCH_SLURM_TIME:-$TIME}"

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
fi

# GDAKI (GPU Direct Async Kernel-Initiated) for multi-node RDMA / GIN
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}"

set -e

run_nccl_ep_srun() {
  local binary="$1"
  local walltime="$2"
  shift 2
  set -x
  # shellcheck disable=SC2086
  srun \
    ${PARTITION:+-p "${PARTITION}"} \
    ${SLURM_ACCOUNT:+--account="${SLURM_ACCOUNT}"} \
    -N 1 \
    -n "${NP}" \
    --exclusive \
    -t "${walltime}" \
    --mpi=pmix \
    --cpu-bind=none \
    --export=ALL \
    ${NCCL_EP_SRUN_EXTRA:-} \
    "$binary" "$@"
  set +x
}

# Run the two extra token-distribution variants for ep_bench: --tokens stays at
# $tokens (= buffer cap) across all calls so kernel iteration counts are identical
# and per-variant timings are directly comparable.
run_ep_bench_variants() {
  local algorithm="$1"
  local tokens="$2"
  local less_than=$((tokens / 2))
  run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" --algorithm "$algorithm" --tokens "$tokens" --hidden 7168 --top-k 8 --experts 256 --validate --dispatch-less-than-max-tokens "$less_than"
  run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" --algorithm "$algorithm" --tokens "$tokens" --hidden 7168 --top-k 8 --experts 256 --validate --non-uniform-tokens
}

# ep_test: low-latency / high-throughput
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ll -t 128 -d 7168
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ht -t 4096 -d 7168

# ep_bench: same shapes (override bench wall time with NCCL_EP_BENCH_SLURM_TIME if needed)
run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" --algorithm low-latency --tokens 128 --hidden 7168 --top-k 8 --experts 256 --validate
run_ep_bench_variants low-latency 128
# High-throughput ep_bench (set NCCL_EP_BENCH_HT=1 to run; off by default — cluster-dependent)
if [[ "${NCCL_EP_BENCH_HT:-0}" == "1" ]]; then
  run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" --algorithm high-throughput --tokens 4096 --hidden 7168 --top-k 8 --experts 256 --validate
  run_ep_bench_variants high-throughput 4096
fi
