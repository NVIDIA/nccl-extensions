#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build and inspect the Python distribution with the M2N native artifact from
# the preceding child-pipeline build job. This intentionally tests the public
# package boundary rather than the source tree through PYTHONPATH.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <nccl_m2n build directory>" >&2
    exit 2
fi

: "${CUDA_HOME:?CUDA_HOME must be set}"
: "${NCCL_HOME:?NCCL_HOME must be set}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$(cd "$1" && pwd)"
PACKAGE_DIR="${ROOT}/python"
M2N_PACKAGE_DIR="${PACKAGE_DIR}/nccl/m2n"
DIST_DIR="${PACKAGE_DIR}/dist"
VENV_DIR="${PACKAGE_DIR}/.ci-m2n-wheel-venv"
BOOTSTRAP_DIR="${PACKAGE_DIR}/.ci-m2n-bootstrap"
TOOLS_VENV_DIR="${PACKAGE_DIR}/.ci-m2n-tools"
UV_CACHE_DIR="${PACKAGE_DIR}/.ci-m2n-uv-cache"
UV_PYTHON_INSTALL_DIR="${PACKAGE_DIR}/.ci-m2n-uv-python"

LIBRARY="${BUILD_DIR}/lib/libnccl_m2n.so"
HEADER="${BUILD_DIR}/include/nccl_m2n.h"
test -s "${LIBRARY}"
test -s "${HEADER}"
test -d "${NCCL_HOME}/lib"

cleanup() {
    rm -rf "${M2N_PACKAGE_DIR}/lib" "${M2N_PACKAGE_DIR}/include" "${VENV_DIR}" \
        "${BOOTSTRAP_DIR}" "${TOOLS_VENV_DIR}" "${UV_CACHE_DIR}" \
        "${UV_PYTHON_INSTALL_DIR}"
}
trap cleanup EXIT
rm -rf "${DIST_DIR}"
cleanup

mkdir -p "${M2N_PACKAGE_DIR}/lib" "${M2N_PACKAGE_DIR}/include" "${DIST_DIR}"
cp "${LIBRARY}" "${M2N_PACKAGE_DIR}/lib/libnccl_m2n.so"
cp "${HEADER}" "${M2N_PACKAGE_DIR}/include/nccl_m2n.h"

# The wheel build below Cython-compiles the checked-in generated sources for
# both EP and M2N. Regeneration is intentionally out of scope for this package
# gate: it needs private cybind source access unavailable in the build image.
# The build-tools image has Python 3.8 and a read-only home directory, while
# this package requires Python 3.10+. Bootstrap uv into the writable checkout,
# then let it provision an isolated supported interpreter and tool environment.
python3 -m pip install --disable-pip-version-check --target "${BOOTSTRAP_DIR}" uv
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
PYTHONPATH="${BOOTSTRAP_DIR}" python3 -m uv venv --python 3.12 --seed "${TOOLS_VENV_DIR}"
"${TOOLS_VENV_DIR}/bin/python" -m pip install --disable-pip-version-check build uv
export PATH="${TOOLS_VENV_DIR}/bin:${PATH}"
PYTHON="${TOOLS_VENV_DIR}/bin/python"

# Build the wheel from the staged checkout, then build the source-only sdist
# separately. The default `build` sequence derives its wheel from the sdist,
# which intentionally excludes the staged shared library.
"${PYTHON}" -m build --wheel --outdir "${DIST_DIR}" "${PACKAGE_DIR}"
"${PYTHON}" -m build --sdist --outdir "${DIST_DIR}" "${PACKAGE_DIR}"

WHEEL="$(find "${DIST_DIR}" -maxdepth 1 -name '*.whl' -print -quit)"
SDIST="$(find "${DIST_DIR}" -maxdepth 1 -name '*.tar.gz' -print -quit)"
test -n "${WHEEL}"
test -n "${SDIST}"

"${PYTHON}" - "${WHEEL}" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = wheel.namelist()
    for expected in (
        "nccl/m2n/lib/libnccl_m2n.so",
        "nccl/m2n/include/nccl_m2n.h",
    ):
        if expected not in names:
            raise SystemExit(f"wheel is missing {expected}")
    metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
    if "Provides-Extra: bench" not in wheel.read(metadata).decode():
        raise SystemExit("wheel metadata is missing the bench extra")
PY
tar -tzf "${SDIST}" | grep -E '/nccl/_extensions/bindings/(nccl_m2n|cynccl_m2n)\.pyx$'
tar -tzf "${SDIST}" | grep -E '/nccl/m2n/include/nccl_m2n\.h$'
if tar -tzf "${SDIST}" | grep -E '\.so$'; then
    echo "ERROR: the source distribution contains a shared library" >&2
    exit 1
fi

"${PYTHON}" -m venv "${VENV_DIR}"
# Match the CUDA 12 build image and install the benchmark dependency through
# the public extras, rather than relying on undeclared transitive packages.
"${VENV_DIR}/bin/pip" install --disable-pip-version-check "${WHEEL}[cu12,bench]"
LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${VENV_DIR}/bin/python" -c 'import nccl.m2n; from nccl._extensions.bindings import nccl_m2n'
LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${VENV_DIR}/bin/python" -m nccl.m2n.benchmarks.reshard_bench --help
