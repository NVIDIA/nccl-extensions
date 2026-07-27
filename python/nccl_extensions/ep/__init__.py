# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL EP: Pythonic API for the libnccl_ep.so extension.

The Cython bindings under :mod:`nccl_extensions.bindings.nccl_ep` are generated from
``nccl_ep/include/nccl_ep.h``. This package provides hand-written Pythonic
wrappers (:class:`Group`, :class:`Handle`, :class:`Tensor`) on top of those bindings.
"""

import os as _os
from pathlib import Path as _Path

try:
    from nccl_extensions.bindings import nccl_ep as _ep_bindings

    # Defaults for libnccl_ep.so's JIT runtime; either env var can be overridden
    # by setting it in the environment before importing nccl_extensions.ep.
    _PKG_DIR = _Path(__file__).parent
    if (_PKG_DIR / "include" / "nccl_ep").is_dir():
        _os.environ.setdefault("NCCL_EP_HOME", str(_PKG_DIR))

    # NCCL public headers (nccl.h, nccl_device/...).
    try:
        import nvidia.nccl as _nv_nccl

        _NCCL_HOME = _Path(_nv_nccl.__path__[0])
        if (_NCCL_HOME / "include").is_dir():
            _os.environ.setdefault("NCCL_HOME", str(_NCCL_HOME))
    except ImportError:
        pass

    _ep_bindings_available = True
except ImportError:
    _ep_bindings = None  # type: ignore[assignment]
    _ep_bindings_available = False

from nccl_extensions.ep.allocator import AllocConfig, AllocFn, FreeFn
from nccl_extensions.ep.enums import (
    Algorithm,
    CombineQuantizationRecipe,
    DispatchQuantizationRecipe,
    ExpertIdKind,
    Layout,
    OverflowPolicy,
    PassDir,
    ZeroCopyMode,
)

try:
    from nccl_extensions.ep.group import Group, GroupConfig
    from nccl_extensions.ep.handle import (
        CombineConfig,
        CombineInputs,
        CombineOutputs,
        DispatchConfig,
        DispatchInputs,
        DispatchOutputs,
        Handle,
        HandleConfig,
        LayoutInfo,
    )
    from nccl_extensions.ep.tensor import Tensor
except ImportError:
    pass


__all__ = [
    "Algorithm",
    "AllocConfig",
    "AllocFn",
    "CombineConfig",
    "CombineInputs",
    "CombineOutputs",
    "CombineQuantizationRecipe",
    "DispatchConfig",
    "DispatchInputs",
    "DispatchOutputs",
    "DispatchQuantizationRecipe",
    "ExpertIdKind",
    "FreeFn",
    "get_lib_path",
    "get_lib_version",
    "Group",
    "GroupConfig",
    "Handle",
    "HandleConfig",
    "Layout",
    "LayoutInfo",
    "OverflowPolicy",
    "PassDir",
    "Tensor",
    "ZeroCopyMode",
]


from packaging.version import Version as _Version


def _decode_version(v: int) -> _Version:
    """Decode NCCL_EP_VERSION_CODE (MAJOR*10000 + MINOR*100 + PATCH)."""
    return _Version(f"{v // 10000}.{(v % 10000) // 100}.{v % 100}")


def get_lib_version() -> _Version:
    """Release version of the loaded ``libnccl_ep.so`` (e.g. ``0.1.0``)."""
    if not _ep_bindings_available:
        raise ImportError("nccl_extensions.bindings.nccl_ep is not available (extension not built)")
    return _decode_version(_ep_bindings.get_version())


def get_lib_path() -> _Path | None:
    """Path of the loaded ``libnccl_ep.so``, or None if it cannot be determined."""
    if not _ep_bindings_available:
        raise ImportError("nccl_extensions.bindings.nccl_ep is not available (extension not built)")
    raw = _ep_bindings.get_library_path()
    return _Path(raw) if raw else None
