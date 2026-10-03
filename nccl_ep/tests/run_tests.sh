#!/usr/bin/env bash
# Run ncclEp unit tests across multiple GPUs.
#
# Spawns one bash process per rank in the background. Each test binary picks
# the GPU via cudaSetDevice(rank % device_count). NCCL bootstrap uses a
# shared UID file: rank 0 writes, all other ranks poll for it (implemented
# in test_common.h::exchange_uid). No MPI runtime required.
#
# Usage:
#   NCCL_HOME=/path/to/nccl/build NCCL_EP_BUILDDIR=/path/to/build bash run_tests.sh [num_gpus]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NCCL_HOME="${NCCL_HOME:-${REPO_ROOT}/third_party/nccl/build}"
NCCL_EP_BUILDDIR="${NCCL_EP_BUILDDIR:-${REPO_ROOT}/build}"
NUM_GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

export LD_LIBRARY_PATH="${NCCL_EP_BUILDDIR}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"

GTEST_ARGS=()
[[ -n "${GTEST_FILTER:-}" ]] && GTEST_ARGS=(--gtest_filter="${GTEST_FILTER}")
TEST_SUITE="${TEST_SUITE:-}"
OVERALL_FAIL=0

run_suite() {
    local BINARY="$1"
    local SUITE_NAME="$2"
    local MIN_GPUS="${3:-4}"
    local FILTER="${4:-}"
    local RUN_GTEST_ARGS=()
    if [[ -n "${FILTER}" ]]; then
        RUN_GTEST_ARGS=(--gtest_filter="${FILTER}")
    else
        RUN_GTEST_ARGS=("${GTEST_ARGS[@]}")
    fi
    local TEST_BIN="${NCCL_EP_BUILDDIR}/test/nccl_ep/${BINARY}"

    if [[ ! -x "${TEST_BIN}" ]]; then
        echo "ERROR: binary not found: ${TEST_BIN}"
        echo "Build first:  make -C ${SCRIPT_DIR} NCCL_HOME=${NCCL_HOME} BUILDDIR=${NCCL_EP_BUILDDIR}"
        return 1
    fi

    if (( NUM_GPUS < MIN_GPUS )); then
        echo "${SUITE_NAME}: requires at least ${MIN_GPUS} GPUs, found ${NUM_GPUS}. Skipping."
        return 0
    fi

    local TMPDIR_L="${TMPDIR:-/tmp}"
    local UID_FILE="${TMPDIR_L}/te_ep_uid_${BINARY}_$$"
    rm -f "${UID_FILE}"
    trap "rm -f '${UID_FILE}'" EXIT INT TERM

    local LOG_DIR
    LOG_DIR=$(mktemp -d)
    local FAIL=0

    echo "=== ${SUITE_NAME} ==="
    echo "  GPUs: ${NUM_GPUS}   Binary: ${TEST_BIN}"
    echo

    # Spawn one process per rank. GPU binding is handled inside the test
    # binary via cudaSetDevice(rank % device_count).
    local PIDS=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        "${TEST_BIN}" \
            --rank="${i}" \
            --nranks="${NUM_GPUS}" \
            --uid-file="${UID_FILE}" \
            "${RUN_GTEST_ARGS[@]}" \
            > "${LOG_DIR}/rank_${i}.log" 2>&1 &
        PIDS+=($!)
    done
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        wait "${PIDS[$i]}" || FAIL=1
    done

    echo "--- Rank 0 output ---"
    cat "${LOG_DIR}/rank_0.log"

    if (( FAIL )); then
        for i in $(seq 1 $((NUM_GPUS - 1))); do
            echo "--- Rank ${i} output ---"
            cat "${LOG_DIR}/rank_${i}.log"
        done
        echo "=== ${SUITE_NAME}: FAILED ==="
        OVERALL_FAIL=1
    else
        echo "=== ${SUITE_NAME}: ALL PASSED ==="
    fi

    rm -rf "${LOG_DIR}"
}

# Suite list: bin|description|em_affected.
# em_affected suites are re-run under every HT-EM mode.
SUITES=(
    "test_ht_dispatch_async|EP HT Async Dispatch Tests|0"
    "test_output_layout|EP Output Layout Tests|1"
    "test_handle_maps|EP Handle Maps Tests|1"
    "test_lifecycle|EP Lifecycle Tests|1"
    "test_public_struct_abi|EP Public Struct ABI Tests|0"
    "test_ht_bwd|EP HT Backward Tests|1"
    "test_ht_stale_routing_map|EP HT Stale Routing Map Tests|1"
    "test_ht_combine_pp_interleave|EP HT Combine PP Interleave Tests|1"
    "test_ht_dispatch_pp_interleave|EP HT Dispatch PP Interleave Tests|1"
    "test_ht_overflow_drop|EP HT Overflow Drop Tests|0"
    "test_tensor_create|EP Tensor Create Tests|0"
    "test_quantization_recipe|EP Quantization Recipe Tests|1"
    "test_zero_copy|EP Zero-Copy forced|0"
    "test_recv_topk_idx_flags|EP Recv Topk Idx Flags Tests|0"
    "test_elastic_buffer|EP Elastic Buffer Tests|0"
    "test_mxfp8_combine|EP MXFP8 Combine Tests|0"
    "test_mxfp8_combine_conversion|EP MXFP8 Conversion Coverage Tests|0"
)

for entry in "${SUITES[@]}"; do
    IFS='|' read -r bin desc _ <<<"${entry}"
    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
    run_suite "${bin}" "${desc}"
done

# LL epoch-ring regression suites. These need an exact world size rather than
# one rank per GPU, so they are invoked here instead of from SUITES above:
#
#   test_ll_epoch_protocol  single process, no NCCL; checks the epoch state
#                           machine (bank parity, staged send/recv, rollover).
#   test_ll_group_epoch     two LL handles on one group; self-skips unless the
#                           world size is exactly 4. A skip exits 0, so running
#                           it at NUM_GPUS != 4 would report a pass without
#                           testing anything -- pin the rank count instead.
#
# Both are LL-only, so they are not re-run under the HT-EM modes below.
run_fixed_rank_suite() {
    local BINARY="$1"
    local SUITE_NAME="$2"
    local RANKS="$3"

    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${BINARY}" ]] || return 0
    if (( NUM_GPUS < RANKS )); then
        echo "${SUITE_NAME}: requires exactly ${RANKS} ranks, only ${NUM_GPUS} GPUs available. Skipping."
        return 0
    fi

    # run_suite spawns one rank per NUM_GPUS; scope the override to this call.
    local SAVED_NUM_GPUS="${NUM_GPUS}"
    NUM_GPUS="${RANKS}"
    run_suite "${BINARY}" "${SUITE_NAME} (${RANKS} ranks)" "${RANKS}"
    NUM_GPUS="${SAVED_NUM_GPUS}"
}

run_fixed_rank_suite test_ll_epoch_protocol "EP LL Epoch Protocol Tests" 1
run_fixed_rank_suite test_ll_group_epoch    "EP LL Group Epoch Tests"    4

for mode in LOCAL_DUP NVLINK_DUP; do
    label="$( [[ ${mode} == LOCAL_DUP ]] && echo 'Local Fanout' || echo 'NVLink Dup' )"
    export "NCCL_EP_HT_EM_${mode}=1"
    for entry in "${SUITES[@]}"; do
        IFS='|' read -r bin desc em <<<"${entry}"
        [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
        [[ ${em} == 1 ]] || continue
        run_suite "${bin}" "${desc} (${label})"
    done
    unset "NCCL_EP_HT_EM_${mode}"
done

# Scan mode (routing-map AllGather): re-run the EM output-layout suites here for
# count-vs-scan layout coverage.
export NCCL_EP_HT_EM_AG_SCAN_MODE=1
for entry in "${SUITES[@]}"; do
    IFS='|' read -r bin desc em <<<"${entry}"
    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
    [[ ${em} == 1 ]] || continue
    run_suite "${bin}" "${desc} (Scan)"
done
# DispatchCombineCrossBoundary only runs its body under scan/nvlink-dup/local-dup/pull-push, but
# it's not em_affected (those two dup modes need a bigger recv budget than it configures), so
# give it its own scan-only rerun for coverage.
[[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "test_elastic_buffer" ]] && run_suite "test_elastic_buffer" "EP Elastic Buffer Tests (Scan)"
# test_ht_overflow_drop is also not em_affected, so give it the same scan-only rerun: it's the
# default (count-mode) pass's only DROP coverage, and scan mode's overflow/drop clamp otherwise
# goes untested.
[[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "test_ht_overflow_drop" ]] && run_suite "test_ht_overflow_drop" "EP HT Overflow Drop Tests (Scan)"
unset NCCL_EP_HT_EM_AG_SCAN_MODE

# Unfused count mode: publishes the EM recv-count tables at UpdateHandle time like scan,
# so gets the same em_affected rerun plus test_elastic_buffer's explicit one.
export NCCL_EP_HT_EM_COUNT_UNFUSED=1
for entry in "${SUITES[@]}"; do
    IFS='|' read -r bin desc em <<<"${entry}"
    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
    [[ ${em} == 1 ]] || continue
    run_suite "${bin}" "${desc} (Unfused Count)"
done
[[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "test_elastic_buffer" ]] && run_suite "test_elastic_buffer" "EP Elastic Buffer Tests (Unfused Count)"
unset NCCL_EP_HT_EM_COUNT_UNFUSED

# Pull-dispatch / push-combine (single NVLink LSA team, expert-major only). Restricted to the
# suites with expert-major dispatch/combine coverage; the non-expert-major cases in them skip
# via SKIP_IF_PULL_PUSH. All ranks form one LSA team so the push combine's single-team path runs.
# Pull-count (receiver-pull map build, no AllGather/scan) is the default fused path here, same
# NCCL_EP_HT_EM_AG_SCAN_MODE fallback gate as EM local-permute. Also exercises the DROP/
# phantom-row overflow path via test_ht_overflow_drop's local-permute tests, which build their
# own expert-major group with no zero_copy override (the other tests in that binary force FLAT
# layout or nvlink_dup, neither of which pull-count supports, so they're filtered out here
# rather than run unguarded).
PULL_PUSH_SUITES="test_output_layout test_ht_bwd test_quantization_recipe test_ht_combine_pp_interleave test_ht_dispatch_pp_interleave"
export NCCL_EP_HT_EM_PULL_PUSH=1
export NCCL_LSA_TEAM_SIZE="${NUM_GPUS}"
for entry in "${SUITES[@]}"; do
    IFS='|' read -r bin desc _ <<<"${entry}"
    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
    [[ " ${PULL_PUSH_SUITES} " == *" ${bin} "* ]] || continue
    run_suite "${bin}" "${desc} (Pull-Push)"
done
OVERFLOW_DROP_PULL_COUNT_FILTER="HtOverflowDropTest.EmLocalPermuteAlignedDropNoOob:HtOverflowDropTest.EmLocalPermuteDeepFlatOverflowPhantomRowsZeroed"
[[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "test_ht_overflow_drop" ]] && \
    run_suite "test_ht_overflow_drop" "EP HT Overflow Drop Tests (Pull-Count)" 4 "${OVERFLOW_DROP_PULL_COUNT_FILTER}"
unset NCCL_EP_HT_EM_PULL_PUSH
unset NCCL_LSA_TEAM_SIZE

# Pull-push scan rerun: same NCCL_EP_HT_EM_AG_SCAN_MODE fallback that forces EM local-permute
# back to scan, applied here to confirm pull-push's scan path still matches.
export NCCL_EP_HT_EM_PULL_PUSH=1
export NCCL_EP_HT_EM_AG_SCAN_MODE=1
export NCCL_LSA_TEAM_SIZE="${NUM_GPUS}"
for entry in "${SUITES[@]}"; do
    IFS='|' read -r bin desc _ <<<"${entry}"
    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
    [[ " ${PULL_PUSH_SUITES} " == *" ${bin} "* ]] || continue
    run_suite "${bin}" "${desc} (Pull-Push, Scan)"
done
unset NCCL_EP_HT_EM_PULL_PUSH
unset NCCL_EP_HT_EM_AG_SCAN_MODE
unset NCCL_LSA_TEAM_SIZE

# Pull-push unfused-sync rerun: same suites, pull-count default (no AG_SCAN_MODE), with the
# head/tail LSA sync split into standalone kernels instead of fused into dispatch/combine.
export NCCL_EP_HT_EM_PULL_PUSH=1
export NCCL_EP_HT_UNFUSED_SYNC=1
export NCCL_LSA_TEAM_SIZE="${NUM_GPUS}"
for entry in "${SUITES[@]}"; do
    IFS='|' read -r bin desc _ <<<"${entry}"
    [[ -z "${TEST_SUITE}" || "${TEST_SUITE}" == "${bin}" ]] || continue
    [[ " ${PULL_PUSH_SUITES} " == *" ${bin} "* ]] || continue
    run_suite "${bin}" "${desc} (Pull-Push, Unfused Sync)"
done
unset NCCL_EP_HT_EM_PULL_PUSH
unset NCCL_EP_HT_UNFUSED_SYNC
unset NCCL_LSA_TEAM_SIZE

exit "${OVERALL_FAIL}"
