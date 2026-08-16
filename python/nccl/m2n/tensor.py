# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Distributed tensor descriptor wrapper for NCCL M2N."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import nccl.core.typing as nccl_typing
from nccl._extensions.bindings import nccl_m2n as _m2n_bindings
from nccl.core.typing import NcclDataType, NcclInvalid
from nccl.m2n.constants import MAX_TENSOR_DIMS, MESH_NDIMS
from nccl.m2n.mesh import Mesh
from nccl.m2n.placement import normalize_placements


_DTYPE_NAMES: dict[str, NcclDataType] = {
    "int8": nccl_typing.INT8,
    "char": nccl_typing.INT8,
    "uint8": nccl_typing.UINT8,
    "byte": nccl_typing.UINT8,
    "int32": nccl_typing.INT32,
    "int": nccl_typing.INT32,
    "uint32": nccl_typing.UINT32,
    "int64": nccl_typing.INT64,
    "long": nccl_typing.INT64,
    "uint64": nccl_typing.UINT64,
    "float16": nccl_typing.FLOAT16,
    "half": nccl_typing.FLOAT16,
    "bfloat16": nccl_typing.BFLOAT16,
    "float32": nccl_typing.FLOAT32,
    "float": nccl_typing.FLOAT32,
    "float64": nccl_typing.FLOAT64,
    "double": nccl_typing.FLOAT64,
    "float8_e4m3fn": nccl_typing.FLOAT8E4M3,
    "float8_e4m3": nccl_typing.FLOAT8E4M3,
    "float8_e5m2": nccl_typing.FLOAT8E5M2,
}

_KIND_SIZE_TO_DTYPE: dict[tuple[str, int], NcclDataType] = {
    ("i", 1): nccl_typing.INT8,
    ("u", 1): nccl_typing.UINT8,
    ("i", 4): nccl_typing.INT32,
    ("u", 4): nccl_typing.UINT32,
    ("i", 8): nccl_typing.INT64,
    ("u", 8): nccl_typing.UINT64,
    ("f", 2): nccl_typing.FLOAT16,
    ("f", 4): nccl_typing.FLOAT32,
    ("f", 8): nccl_typing.FLOAT64,
}


def normalize_dtype(dtype: object) -> NcclDataType:
    if isinstance(dtype, NcclDataType):
        return dtype
    if isinstance(dtype, int):
        try:
            return NcclDataType(dtype)
        except ValueError as exc:
            raise NcclInvalid(f"unsupported NCCL Reshard dtype: {dtype!r}") from exc

    name = str(dtype).replace("torch.", "").lower()
    if name in _DTYPE_NAMES:
        return _DTYPE_NAMES[name]

    try:
        import numpy as np

        np_dtype = np.dtype(dtype)
        if np_dtype.name in _DTYPE_NAMES:
            return _DTYPE_NAMES[np_dtype.name]
        kind_size = (np_dtype.kind, np_dtype.itemsize)
        if kind_size in _KIND_SIZE_TO_DTYPE:
            return _KIND_SIZE_TO_DTYPE[kind_size]
    except (ImportError, TypeError, ValueError):
        pass

    raise NcclInvalid(f"unsupported NCCL Reshard dtype: {dtype!r}")


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(dim) for dim in shape)
    if not 1 <= len(out) <= MAX_TENSOR_DIMS:
        raise ValueError(f"tensor rank must be 1..{MAX_TENSOR_DIMS}, got shape={out}")
    if any(dim < 0 for dim in out):
        raise ValueError(f"local_shape values must be non-negative, got {out}")
    return out


def _cai_is_contiguous(shape: tuple[int, ...], strides: object, typestr: str) -> bool:
    if strides is None or any(dim == 0 for dim in shape):
        return True
    try:
        import numpy as np

        itemsize = int(np.dtype(typestr).itemsize)
        normalized_strides = tuple(int(stride) for stride in strides)
    except (TypeError, ValueError):
        return False
    if len(normalized_strides) != len(shape):
        return False
    expected = itemsize
    for dim, stride in zip(reversed(shape), reversed(normalized_strides), strict=True):
        if dim > 1 and stride != expected:
            return False
        expected *= dim
    return True


def _resolve_buffer(buffer: object) -> tuple[int, tuple[int, ...] | None, NcclDataType | None]:
    if buffer is None:
        return 0, None, None
    if isinstance(buffer, int):
        if buffer < 0:
            raise ValueError(f"raw device pointer must be non-negative, got {buffer}")
        return int(buffer), None, None

    data_ptr = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        is_contiguous = getattr(buffer, "is_contiguous", None)
        if callable(is_contiguous) and not is_contiguous():
            raise ValueError(
                f"{type(buffer).__name__} must be contiguous; "
                "ncclDistTensor_t does not carry stride metadata"
            )
        shape = _shape_tuple(tuple(int(dim) for dim in buffer.shape))
        dtype = normalize_dtype(buffer.dtype)
        ptr = int(data_ptr())
        if ptr < 0:
            raise ValueError(f"device pointer must be non-negative, got {ptr}")
        return ptr, shape, dtype

    cai = getattr(buffer, "__cuda_array_interface__", None)
    if cai is not None:
        ptr = int(cai["data"][0])
        if ptr < 0:
            raise ValueError(f"CUDA array device pointer must be non-negative, got {ptr}")
        shape = _shape_tuple(cai["shape"])
        typestr = cai.get("typestr")
        if typestr is None:
            raise ValueError(
                "__cuda_array_interface__ is missing typestr for "
                f"{type(buffer).__name__}"
            )
        if not _cai_is_contiguous(shape, cai.get("strides"), typestr):
            raise ValueError(
                f"{type(buffer).__name__} must expose a contiguous CUDA array; "
                "ncclDistTensor_t does not carry stride metadata"
            )
        dtype = normalize_dtype(typestr)
        return ptr, shape, dtype

    raise NcclInvalid(
        f"cannot resolve {type(buffer).__name__}: expected None, raw int "
        "device pointer, torch.Tensor, or CUDA Array Interface object"
    )


@dataclass
class _PreparedDistTensor:
    mesh: _m2n_bindings.Mesh
    struct: _m2n_bindings.DistTensor


class DistTensor:
    """Pythonic ``ncclDistTensor_t`` descriptor.

    ``buffer=None`` represents a non-participating side for this rank.  Mesh,
    placements, local shape, and dtype are still required because NCCL M2N
    reads both side descriptors on every rank.
    """

    def __init__(
        self,
        buffer: object,
        local_shape: Sequence[int] | None = None,
        dtype: object | None = None,
        *,
        mesh: Mesh,
        placements: Sequence[object],
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError(f"mesh must be nccl.m2n.Mesh, got {type(mesh).__name__}")
        ptr, inferred_shape, inferred_dtype = _resolve_buffer(buffer)
        if local_shape is None:
            if inferred_shape is None:
                raise ValueError("local_shape is required for None or raw-pointer buffers")
            local_shape = inferred_shape
        if dtype is None:
            if inferred_dtype is None:
                raise ValueError("dtype is required for None or raw-pointer buffers")
            dtype = inferred_dtype

        shape = _shape_tuple(local_shape)
        self._buffer = buffer
        self._data_ptr = int(ptr)
        self._local_shape = shape
        self._dtype = normalize_dtype(dtype)
        self._mesh = mesh
        self._placements = normalize_placements(placements)

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    @property
    def buffer(self) -> object:
        return self._buffer

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self._local_shape

    @property
    def ndims(self) -> int:
        return len(self._local_shape)

    @property
    def dtype(self) -> NcclDataType:
        return self._dtype

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def placements(self) -> tuple[int, int]:
        return self._placements

    @property
    def is_participating(self) -> bool:
        return self._data_ptr != 0

    def as_binding(self) -> _PreparedDistTensor:
        mesh_struct = self._mesh.to_binding()
        local_shape = list(self._local_shape) + [1] * (MAX_TENSOR_DIMS - self.ndims)
        struct = _m2n_bindings.DistTensor()
        struct.dataPtr = int(self._data_ptr)
        struct.localShape = tuple(local_shape)
        struct.ndims = self.ndims
        struct.dtype = int(self._dtype)
        struct.mesh = int(mesh_struct.ptr)
        struct.placements = tuple(int(p) for p in self._placements)
        return _PreparedDistTensor(mesh=mesh_struct, struct=struct)

    def __repr__(self) -> str:
        role = "data" if self.is_participating else "none"
        return (
            f"<DistTensor role={role} shape={self.local_shape} "
            f"dtype={self.dtype.name} mesh={self.mesh.dims}@{self.mesh.start_rank}>"
        )


SCALE_NONE = 0
SCALE_FWD = 1

# Deliberately narrower than the payload dtype set; mirrors the C validator.
_SCALE_DTYPES: frozenset[NcclDataType] = frozenset(
    {
        nccl_typing.FLOAT32,
        nccl_typing.FLOAT16,
        nccl_typing.BFLOAT16,
        nccl_typing.FLOAT8E4M3,
        nccl_typing.FLOAT8E5M2,
        nccl_typing.UINT8,
    }
)


@dataclass
class _PreparedDistTensor:
    mesh: _m2n_bindings.Mesh
    struct: _m2n_bindings.DistTensor


class DistTensor:
    """Pythonic ``ncclDistTensor_t`` descriptor.

    ``buffer=None`` represents a non-participating side for this rank.  Mesh,
    placements, local shape, and dtype are still required because NCCL M2N
    reads both side descriptors on every rank.
    """

    def __init__(
        self,
        buffer: object,
        local_shape: Sequence[int] | None = None,
        dtype: object | None = None,
        *,
        mesh: Mesh,
        placements: Sequence[object],
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError(f"mesh must be nccl.m2n.Mesh, got {type(mesh).__name__}")
        ptr, inferred_shape, inferred_dtype = _resolve_buffer(buffer)
        if local_shape is None:
            if inferred_shape is None:
                raise ValueError("local_shape is required for None or raw-pointer buffers")
            local_shape = inferred_shape
        if dtype is None:
            if inferred_dtype is None:
                raise ValueError("dtype is required for None or raw-pointer buffers")
            dtype = inferred_dtype

        shape = _shape_tuple(local_shape)
        self._buffer = buffer
        self._data_ptr = int(ptr)
        self._local_shape = shape
        self._dtype = normalize_dtype(dtype)
        self._mesh = mesh
        self._placements = normalize_placements(placements)

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    @property
    def buffer(self) -> object:
        return self._buffer

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self._local_shape

    @property
    def ndims(self) -> int:
        return len(self._local_shape)

    @property
    def dtype(self) -> NcclDataType:
        return self._dtype

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def placements(self) -> tuple[int, int]:
        return self._placements

    @property
    def is_participating(self) -> bool:
        return self._data_ptr != 0

    def as_binding(self) -> _PreparedDistTensor:
        mesh_struct = self._mesh.to_binding()
        local_shape = list(self._local_shape) + [1] * (MAX_TENSOR_DIMS - self.ndims)
        struct = _m2n_bindings.DistTensor()
        struct.dataPtr = int(self._data_ptr)
        struct.localShape = tuple(local_shape)
        struct.ndims = self.ndims
        struct.dtype = int(self._dtype)
        struct.mesh = int(mesh_struct.ptr)
        struct.placements = tuple(int(p) for p in self._placements)
        return _PreparedDistTensor(mesh=mesh_struct, struct=struct)

    def __repr__(self) -> str:
        role = "data" if self.is_participating else "none"
        return (
            f"<DistTensor role={role} shape={self.local_shape} "
            f"dtype={self.dtype.name} mesh={self.mesh.dims}@{self.mesh.start_rank}>"
        )


class ScalePlane:
    """Companion per-block scale plane for a coupled reshard.

    Wraps ``ncclReshardScalePlane_t``.  Topology is not restated: the plane
    reuses the payload tensors' meshes and placements, which is the invariant
    the C layer validates.  ``src``/``dst`` accept the same buffer forms as
    :class:`DistTensor` (``None`` for a non-participating side).

    ``block_size`` is required and never inferred, matching the C contract.
    Per-side scale shapes default to the payload shape with ``block_dim``
    divided by ``block_size``; supply them explicitly to have the C layer
    validate your own arithmetic instead.
    """

    def __init__(
        self,
        src: object,
        dst: object,
        *,
        dtype: object,
        block_size: int,
        block_dim: int = -1,
        src_local_shape: Sequence[int] | None = None,
        dst_local_shape: Sequence[int] | None = None,
        src_payload: DistTensor | None = None,
        dst_payload: DistTensor | None = None,
    ) -> None:
        if int(block_size) < 1:
            raise NcclInvalid(f"block_size must be at least 1, got {block_size}")

        resolved_dtype = normalize_dtype(dtype)
        if resolved_dtype not in _SCALE_DTYPES:
            raise NcclInvalid(
                f"unsupported scale dtype {resolved_dtype.name}; supported: "
                + ", ".join(sorted(d.name for d in _SCALE_DTYPES))
            )

        src_ptr, _, _ = _resolve_buffer(src)
        dst_ptr, _, _ = _resolve_buffer(dst)

        if src_local_shape is None or dst_local_shape is None:
            if src_payload is None or dst_payload is None:
                raise NcclInvalid(
                    "ScalePlane needs either explicit src/dst_local_shape or "
                    "src_payload/dst_payload to derive them from"
                )

        def _derive(payload: DistTensor, dim: int) -> tuple[int, ...]:
            shape = list(payload.local_shape)
            if shape[dim] % int(block_size) != 0:
                raise NcclInvalid(
                    f"block_size {block_size} does not divide payload extent "
                    f"{shape[dim]} on dim {dim}"
                )
            shape[dim] //= int(block_size)
            return tuple(shape)

        reference = src_payload if src_payload is not None else dst_payload
        if block_dim < 0:
            if reference is None:
                raise NcclInvalid("block_dim must be given when no payload tensor is supplied")
            block_dim = reference.ndims - 1

        if src_local_shape is None:
            src_local_shape = _derive(src_payload, block_dim)
        if dst_local_shape is None:
            dst_local_shape = _derive(dst_payload, block_dim)

        self._src_data_ptr = src_ptr
        self._dst_data_ptr = dst_ptr
        self._src_buffer = src
        self._dst_buffer = dst
        self._dtype = resolved_dtype
        self._block_dim = int(block_dim)
        self._block_size = int(block_size)
        self._src_local_shape = tuple(int(v) for v in src_local_shape)
        self._dst_local_shape = tuple(int(v) for v in dst_local_shape)

    @property
    def dtype(self) -> NcclDataType:
        return self._dtype

    @property
    def block_dim(self) -> int:
        return self._block_dim

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def src_local_shape(self) -> tuple[int, ...]:
        return self._src_local_shape

    @property
    def dst_local_shape(self) -> tuple[int, ...]:
        return self._dst_local_shape

    def as_binding(self) -> _m2n_bindings.ScalePlane:
        struct = _m2n_bindings.ScalePlane()
        struct.recipe = SCALE_FWD
        struct.srcDataPtr = int(self._src_data_ptr)
        struct.dstDataPtr = int(self._dst_data_ptr)
        struct.dtype = int(self._dtype)
        struct.blockDim = self._block_dim
        struct.blockSize = self._block_size
        pad = lambda s: tuple(list(s) + [0] * (MAX_TENSOR_DIMS - len(s)))
        struct.srcLocalShape = pad(self._src_local_shape)
        struct.dstLocalShape = pad(self._dst_local_shape)
        return struct

    def __repr__(self) -> str:
        return (
            f"<ScalePlane dtype={self._dtype.name} block_dim={self._block_dim} "
            f"block_size={self._block_size} src={self._src_local_shape} "
            f"dst={self._dst_local_shape}>"
        )


QUANT_NONE = 0
QUANT_FP8E4M3 = 1
QUANT_MXFP8 = 2

# The quantizer works through float, so integer payloads have no meaningful
# FP8 representation.  Mirrors the C validator.
_QUANT_PAYLOAD_DTYPES: frozenset[NcclDataType] = frozenset(
    {nccl_typing.BFLOAT16, nccl_typing.FLOAT16, nccl_typing.FLOAT32}
)


class QuantSpec:
    """On-the-fly wire compression settings for :func:`reshard_quantized`.

    Wraps ``ncclReshardQuantConfig_t``.  Carries no buffers: the FP8 payload
    and its generated scales live in library-owned scratch for the duration of
    the call.

    This is LOSSY — each block of ``block_size`` elements is reconstructed from
    one shared scale — and it is not unconditionally faster, since it adds a
    pass over the tile on each side.  See the README.
    """

    def __init__(
        self,
        *,
        block_dim: int,
        block_size: int | None = None,
        round_scales: bool = False,
        recipe: int = QUANT_FP8E4M3,
        dst_scales: object = None,
    ) -> None:
        if recipe not in (QUANT_FP8E4M3, QUANT_MXFP8):
            raise NcclInvalid(f"unsupported quantization recipe {recipe}")
        if recipe == QUANT_MXFP8:
            # MX defines a 32-element shared scale, and E8M0 can only hold a
            # power of two; mirror the C validator rather than silently fixing
            # up the caller's values.
            if block_size is None:
                block_size = 32
            if int(block_size) != 32:
                raise NcclInvalid(f"MXFP8 requires block_size == 32, got {block_size}")
            if not round_scales:
                raise NcclInvalid("MXFP8 scales are E8M0 (powers of two); set round_scales=True")
        if block_size is None:
            raise NcclInvalid("block_size is required")
        if int(block_size) < 1:
            raise NcclInvalid(f"block_size must be at least 1, got {block_size}")
        if int(block_dim) < 0:
            raise NcclInvalid(f"block_dim must be non-negative, got {block_dim}")
        self._recipe = int(recipe)
        self._block_size = int(block_size)
        self._block_dim = int(block_dim)
        self._round_scales = bool(round_scales)
        self._dst_scales_ptr, _, _ = _resolve_buffer(dst_scales)
        self._dst_scales = dst_scales

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_dim(self) -> int:
        return self._block_dim

    @property
    def round_scales(self) -> bool:
        return self._round_scales

    @property
    def recipe(self) -> int:
        return self._recipe

    @property
    def dst_scales(self) -> object:
        return self._dst_scales

    @staticmethod
    def check_payload_dtype(dtype: NcclDataType) -> None:
        if dtype not in _QUANT_PAYLOAD_DTYPES:
            raise NcclInvalid(
                f"unsupported quantized payload dtype {dtype.name}; supported: "
                + ", ".join(sorted(d.name for d in _QUANT_PAYLOAD_DTYPES))
            )

    def as_binding(self) -> _m2n_bindings.QuantConfig:
        struct = _m2n_bindings.QuantConfig()
        struct.recipe = self._recipe
        struct.blockDim = self._block_dim
        struct.blockSize = self._block_size
        struct.roundScales = 1 if self._round_scales else 0
        struct.dstScales = int(self._dst_scales_ptr)
        return struct

    def __repr__(self) -> str:
        name = "MXFP8" if self._recipe == QUANT_MXFP8 else "FP8E4M3"
        mode = "keep-quantized" if self._dst_scales_ptr else "dequantize"
        return (
            f"<QuantSpec {name} block_dim={self._block_dim} block_size={self._block_size} "
            f"round_scales={self._round_scales} {mode}>"
        )


__all__ = [
    "DistTensor",
    "QuantSpec",
    "QUANT_FP8E4M3",
    "QUANT_MXFP8",
    "QUANT_NONE",
    "SCALE_FWD",
    "SCALE_NONE",
    "ScalePlane",
    "normalize_dtype",
]
