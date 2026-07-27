# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup, Extension


_NCCL_EP_SO = Path(__file__).parent / "nccl" / "ep" / "lib" / "libnccl_ep.so"
if not _NCCL_EP_SO.exists():
    print(
        f"WARNING: {_NCCL_EP_SO} not found. The built wheel will not "
        "include the NCCL EP shared library, and `import nccl.ep` will fail at "
        "runtime. Drop a build of libnccl_ep.so at that path before "
        "building the wheel.",
        file=sys.stderr,
    )


CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    raise SystemExit("Error: CUDA_HOME is not set")

cuda_path = Path(CUDA_HOME)
if not cuda_path.exists() or not cuda_path.is_dir():
    raise SystemExit(f"Error: CUDA_HOME does not exist or is not a directory: {CUDA_HOME}")
CUDA_INC = str(cuda_path / "include")

PACKAGE = "nccl._extensions.bindings"
# One entry per bound library; nccl_m2n joins here once its bindings land.
LIBNAMES = ["nccl_ep"]


def _ext(module: str, source: str) -> Extension:
    return Extension(
        module,
        sources=[source],
        include_dirs=[CUDA_INC],
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


pkg_dir = os.path.join(*PACKAGE.split("."))
ext_modules = [
    _ext(f"{PACKAGE}._internal.utils", os.path.join(pkg_dir, "_internal", "utils.pyx"))
]
for libname in LIBNAMES:
    ext_modules.extend(libname_extensions(libname))

compiler_directives = {
    "embedsignature": True,
    "show_performance_hints": True,
    "freethreading_compatible": True,
}

setup(
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
