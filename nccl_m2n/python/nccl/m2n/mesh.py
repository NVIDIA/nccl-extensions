# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Mesh wrapper for ``ncclMesh_t``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nccl.m2n.bindings import nccl_m2n as _m2n_bindings
from nccl.m2n.constants import MESH_NDIMS


def _as_nested_list(ranks: object) -> list[list[int]]:
    tolist = getattr(ranks, "tolist", None)
    if callable(tolist):
        ranks = tolist()
    if not isinstance(ranks, Sequence) or isinstance(ranks, (str, bytes)):
        raise TypeError("mesh ranks must be a 1-D or 2-D sequence")
    if len(ranks) == 0:
        raise ValueError("mesh ranks must not be empty")

    first = ranks[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        rows = [list(row) for row in ranks]
    else:
        # Match PyTorch DeviceMesh: a 1-D mesh maps to dims=(N, 1).
        rows = [[rank] for rank in ranks]
    if any(len(row) == 0 for row in rows):
        raise ValueError("mesh rank rows must not be empty")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("mesh rank rows must have equal lengths")
    return [[int(rank) for rank in row] for row in rows]


@dataclass(frozen=True)
class Mesh:
    """2-D topology-only mesh descriptor.

    ``dims`` are the public ``ncclMesh_t.dims`` values and ``start_rank``
    is the first world rank in the contiguous interval owned by this side.
    """

    dims: tuple[int, int]
    start_rank: int = 0

    def __init__(self, dims: Sequence[int], start_rank: int = 0) -> None:
        if len(dims) != MESH_NDIMS:
            raise ValueError(f"expected {MESH_NDIMS} mesh dims, got {len(dims)}")
        norm_dims = tuple(int(d) for d in dims)
        if any(d <= 0 for d in norm_dims):
            raise ValueError(f"mesh dims must be positive, got {norm_dims}")
        start_rank = int(start_rank)
        if start_rank < 0:
            raise ValueError(f"mesh start_rank must be non-negative, got {start_rank}")
        object.__setattr__(self, "dims", norm_dims)
        object.__setattr__(self, "start_rank", start_rank)

    @classmethod
    def from_ranks(cls, ranks: object) -> "Mesh":
        rows = _as_nested_list(ranks)
        flat = [rank for row in rows for rank in row]
        start_rank = min(flat)
        expected = list(range(start_rank, start_rank + len(flat)))
        if flat != expected:
            raise ValueError(
                "ncclMesh_t requires a row-major contiguous rank interval; "
                f"got ranks={flat}, expected ranks={expected}"
            )
        return cls((len(rows), len(rows[0])), start_rank=start_rank)

    @classmethod
    def from_device_mesh(cls, device_mesh: object) -> "Mesh":
        ranks = getattr(device_mesh, "mesh", None)
        if ranks is None:
            raise TypeError("DeviceMesh object must expose public .mesh rank grid")
        return cls.from_ranks(ranks)

    @property
    def size(self) -> int:
        return self.dims[0] * self.dims[1]

    def to_binding(self) -> _m2n_bindings.Mesh:
        mesh = _m2n_bindings.Mesh()
        mesh.dims = (int(self.dims[0]), int(self.dims[1]))
        mesh.startRank = int(self.start_rank)
        return mesh


__all__ = ["Mesh"]
