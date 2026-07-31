# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

from __future__ import annotations

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


def _candidate_include_dirs() -> list[str]:
    candidates: list[Path] = []

    for env_var in ("NCCL_M2N_INCLUDE_DIR", "NCCL_INCLUDE_DIR", "CUDA_INCLUDE_DIR"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    for env_var in ("NCCL_M2N_HOME", "NCCL_HOME", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value) / "include")

    nccl_repo_dir = os.environ.get("NCCL_REPO_DIR")
    if nccl_repo_dir:
        root = Path(nccl_repo_dir)
        candidates.extend([
            root / "build" / "include",
            root / "src" / "include",
            root / "include",
        ])

    candidates.extend([
        ROOT.parent / "src",
        ROOT / "include",
        Path("/usr/local/include"),
        Path("/usr/include"),
    ])

    seen: set[str] = set()
    include_dirs: list[str] = []
    for candidate in candidates:
        path = str(candidate)
        if path not in seen:
            seen.add(path)
            include_dirs.append(path)
    return include_dirs


def _has_header(include_dirs: list[str], header: str) -> bool:
    return any((Path(include_dir) / header).is_file() for include_dir in include_dirs)


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        include_dirs = _candidate_include_dirs()
        missing: list[str] = []
        if not _has_header(include_dirs, "nccl_m2n.h"):
            missing.append("nccl_m2n.h")
        if not _has_header(include_dirs, "nccl.h"):
            missing.append("nccl.h")
        if missing:
            searched = "\n  ".join(include_dirs)
            raise RuntimeError(
                "Cannot build nccl-m2n Python bindings; missing "
                + ", ".join(missing)
                + ". Set NCCL_M2N_INCLUDE_DIR/NCCL_M2N_HOME and "
                "NCCL_INCLUDE_DIR/NCCL_HOME, or set NCCL_REPO_DIR to an NCCL checkout.\n"
                f"Searched:\n  {searched}"
            )
        super().build_extensions()


def _extension(name: str, source: str) -> Extension:
    return Extension(
        name,
        sources=[source],
        include_dirs=_candidate_include_dirs(),
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )


ext_modules = [
    _extension("nccl.m2n.bindings.nccl_m2n", "nccl/m2n/bindings/nccl_m2n.pyx"),
    _extension("nccl.m2n.bindings.cynccl_m2n", "nccl/m2n/bindings/cynccl_m2n.pyx"),
    _extension(
        "nccl.m2n.bindings._internal.nccl_m2n",
        "nccl/m2n/bindings/_internal/nccl_m2n_linux.pyx",
    ),
]

compiler_directives = {
    "embedsignature": True,
    "show_performance_hints": True,
    "freethreading_compatible": True,
}

setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    packages=find_namespace_packages(where=".", include=["nccl", "nccl.*"]),
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
