# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import os
import sys
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
EP_PACKAGE = ROOT / "nccl" / "ep"
M2N_PACKAGE = ROOT / "nccl" / "m2n"
BUILDING_SDIST = "sdist" in sys.argv
REQUIRE_NATIVE_LIBS = (
    os.environ.get("NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS", "0") != "0"
)


def _check_staged_libraries(package_dir: Path, library: str, package: str) -> None:
    variants = tuple(package_dir / "lib" / f"cu{major}" / library for major in (12, 13))
    if REQUIRE_NATIVE_LIBS:
        missing = [path for path in variants if not path.is_file()]
        if not missing:
            return
        raise SystemExit(
            "Error: Production wheels require both CUDA variants; missing: "
            + ", ".join(str(path) for path in missing)
        )
    if any(path.is_file() for path in variants):
        return
    message = (
        f"No CUDA variant of the {package} shared library was found under "
        f"{package_dir / 'lib'}. The built wheel will require a compatible "
        "external library at runtime."
    )
    print(f"WARNING: {message}", file=sys.stderr)


if not BUILDING_SDIST:
    from Cython.Build import cythonize

    _check_staged_libraries(EP_PACKAGE, "libnccl_ep.so.0", "nccl.ep")
    _check_staged_libraries(M2N_PACKAGE, "libnccl_m2n.so", "nccl.m2n")

    CUDA_HOME = os.environ.get("CUDA_HOME")
    if not CUDA_HOME:
        raise SystemExit("Error: CUDA_HOME is not set")
    cuda_path = Path(CUDA_HOME)
    if not cuda_path.exists() or not cuda_path.is_dir():
        raise SystemExit(
            f"Error: CUDA_HOME does not exist or is not a directory: {CUDA_HOME}"
        )
    CUDA_INC = str(cuda_path / "include")


PACKAGE = "nccl._extensions.bindings"
LIBNAMES = ["nccl_ep", "nccl_m2n"]


def _ext(module: str, source: str, *, include_dirs: list[str] | None = None) -> Extension:
    return Extension(
        module,
        sources=[source],
        include_dirs=[CUDA_INC, *(include_dirs or [])],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )


def libname_extensions(libname: str) -> list[Extension]:
    """Three extensions per library: lowpp, cy variant, _internal loader.

    For libname="nccl_ep":
        nccl._extensions.bindings.nccl_ep           <- nccl/_extensions/bindings/nccl_ep.pyx
        nccl._extensions.bindings.cynccl_ep         <- nccl/_extensions/bindings/cynccl_ep.pyx
        nccl._extensions.bindings._internal.nccl_ep <- nccl/_extensions/bindings/_internal/nccl_ep_linux.pyx
    """
    pkg_dir = os.path.join(*PACKAGE.split("."))
    return [
        _ext(f"{PACKAGE}.{libname}", os.path.join(pkg_dir, f"{libname}.pyx")),
        _ext(f"{PACKAGE}.cy{libname}", os.path.join(pkg_dir, f"cy{libname}.pyx")),
        _ext(
            f"{PACKAGE}._internal.{libname}",
            os.path.join(pkg_dir, "_internal", f"{libname}_linux.pyx"),
        ),
    ]


ext_modules = []
if not BUILDING_SDIST:
    pkg_dir = os.path.join(*PACKAGE.split("."))
    ext_modules.append(
        _ext(
            f"{PACKAGE}._internal.utils",
            os.path.join(pkg_dir, "_internal", "utils.pyx"),
        )
    )
    for libname in LIBNAMES:
        ext_modules.extend(libname_extensions(libname))
    ext_modules = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives={
            "embedsignature": True,
            "show_performance_hints": True,
            "freethreading_compatible": True,
        },
    )

setup(
    ext_modules=ext_modules,
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
