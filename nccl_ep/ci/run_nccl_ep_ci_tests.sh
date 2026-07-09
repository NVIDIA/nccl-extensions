#!/usr/bin/env bash
# NCCL EP CI: run ep_test and ep_bench inside a Slurm allocation (srun --mpi=pmix, SRUN_MPI-style).
# Set OMPI_MCA_* in CI for EOS TCP BTL (e.g. btl_tcp_if_include) when applicable.
# NCCL_EP_BENCH_HT=1 enables ep_bench --algorithm high-throughput (optional; not all clusters pass it).

: "${NCCL_HOME:?NCCL_HOME must be set}"
NCCL_EP_HOME="${NCCL_EP_HOME:-${NCCL_HOME}}"

EP_TEST="${NCCL_EP_HOME}/test/nccl_ep/ep_test"
EP_BENCH="${NCCL_EP_HOME}/test/nccl_ep/ep_bench"
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

export LD_LIBRARY_PATH="${NCCL_EP_HOME}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
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

# ep_test: low-latency (expert-major, LL-hardcoded) / high-throughput × {flat, expert-major}
# ep_test is kept at one canonical batch size per algorithm (smoke coverage); batch-size
# sweep lives in ep_bench below.
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ll              -t 128  -d 7168
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ht -L fl        -t 4096 -d 7168
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ht -L em        -t 4096 -d 7168

# ep_bench: layout × batch-size × datatype cross-product (override bench wall time with NCCL_EP_BENCH_SLURM_TIME if needed)
# LL supports {expert-major, rank-major}; HT supports {flat, expert-major}.
# Batch sizes (tokens per rank): 128, 256, 1K, 4K, 8K.
# Datatypes: NONE-mode wire formats. bf16 is the baseline; fp16/fp32 exercise the
# distinct decode/encode (fp16) and larger-SMEM combine (fp32) paths.
# The datatype list and hidden dim are globals the caller sets before each sweep,
# so HT fp32 can run at a reduced hidden (see HT block below).
EP_BENCH_TOKEN_SIZES=(128 256 1024 4096)
EP_BENCH_DATATYPES=(bf16 fp16 fp32)
EP_BENCH_HIDDEN=7168

run_ep_bench_layout_size_sweep() {
  local algorithm="$1"; shift
  local layouts=("$@")
  local layout tokens dtype
  for layout in "${layouts[@]}"; do
    for tokens in "${EP_BENCH_TOKEN_SIZES[@]}"; do
      for dtype in "${EP_BENCH_DATATYPES[@]}"; do
        run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" \
          --algorithm "$algorithm" --layout "$layout" \
          --tokens "$tokens" --hidden "$EP_BENCH_HIDDEN" --top-k 8 --experts 256 --validate \
          --datatype "$dtype"
      done
    done
  done
}

# LL bf16/fp16: full hidden, full batch range.
EP_BENCH_DATATYPES=(bf16 fp16)
EP_BENCH_TOKEN_SIZES=(128 256 1024 4096)
EP_BENCH_HIDDEN=7168
run_ep_bench_layout_size_sweep low-latency em rm
# LL fp32: 4 B/elem -> cap the max batch at 2048 (half of 4096) to keep the
# per-rank byte footprint on par with the 16-bit runs.
EP_BENCH_DATATYPES=(fp32)
EP_BENCH_TOKEN_SIZES=(128 256 1024 2048)
run_ep_bench_layout_size_sweep low-latency em rm
# Token-distribution variants stay at the canonical LL batch size (cover the variant axis,
# not the size axis — already swept above).
run_ep_bench_variants low-latency 128

# SCALES_FORWARD is a byte-transport recipe. Keep a multi-token positive
# validation in CI so both token rows and FP32 scale rows are checked.
run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" \
  --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
  --validate --dispatch-only --dispatch-quantization scales-forward
run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" \
  --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
  --validate --dispatch-only --dispatch-quantization ds-fp8e3m4

# High-throughput ep_bench (set NCCL_EP_BENCH_HT=1 to run; off by default — cluster-dependent)
if [[ "${NCCL_EP_BENCH_HT:-0}" == "1" ]]; then
  # bf16/fp16 at the full hidden dim, full batch range.
  EP_BENCH_DATATYPES=(bf16 fp16)
  EP_BENCH_TOKEN_SIZES=(128 256 1024 4096)
  EP_BENCH_HIDDEN=7168
  run_ep_bench_layout_size_sweep high-throughput fl em
  run_ep_bench_variants high-throughput 4096
  run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" \
    --algorithm high-throughput --layout fl --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --dispatch-only --dispatch-quantization scales-forward

  # FOLLOW-UP: HT fp32 dispatch SMEM exceeds the device cap (~227KB on H100) at
  # hidden=7168 with the default stages/pipelines, and currently std::abort()s in
  # check_dispatch_smem_limit (device/ht_ep_adapter.cu:806). That abort should be
  # turned into a clean ncclInvalidArgument rejection (tracked separately). Until then,
  # exercise HT fp32 at half the hidden dim: fp32 is 4 B/elem vs 2 B for 16-bit, so
  # hidden=3584 gives the SAME per-token byte footprint as the 16-bit hidden=7168 runs
  # (3584*4 == 7168*2 == 14336 B) — both an apples-to-apples comparison and a guaranteed
  # SMEM fit. (Dispatch SMEM scales with hidden, not tokens, so hidden is the knob.)
  # fp32: hidden halved to 3584 (dispatch SMEM cap) and batch capped at 2048
  # (4 B/elem), matching the fp32 batch used everywhere else.
  EP_BENCH_DATATYPES=(fp32)
  EP_BENCH_TOKEN_SIZES=(128 256 1024 2048)
  EP_BENCH_HIDDEN=3584
  run_ep_bench_layout_size_sweep high-throughput fl em
fi
