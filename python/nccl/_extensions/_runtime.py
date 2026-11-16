# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Process-wide runtime state shared by extension facades and bindings."""

from __future__ import annotations

import threading
from pathlib import Path

from cuda import bindings

NATIVE_CALL_LOCK = threading.RLock()


def cuda_major() -> int:
    """Return the CUDA userspace major selected by ``cuda.bindings``."""
    major = bindings.__version__.split(".", 1)[0]
    if major not in {"12", "13"}:
        raise RuntimeError("cuda.bindings 12.x or 13.x must be installed")
    return int(major)


def bundled_library(libname: str) -> str | None:
    """Return the bundled library matching the installed ``cuda.bindings``."""
    package_dir = Path(__file__).resolve().parent.parent / libname.removeprefix("nccl_")
    # EP's bundled ABI major is fixed by this package's native build.
    filename = "libnccl_ep.so.0" if libname == "nccl_ep" else f"lib{libname}.so"
    library = package_dir / "lib" / f"cu{cuda_major()}" / filename
    return str(library) if library.is_file() else None


__all__ = ["NATIVE_CALL_LOCK", "bundled_library", "cuda_major"]
