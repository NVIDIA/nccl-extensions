# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Example of NCCL M2N reshard with JAX."""

from __future__ import annotations

import argparse
import multiprocessing
from multiprocessing.connection import wait
import socket
import sys
from collections.abc import Sequence
from typing import Any

import nccl.core as nccl
import numpy as np
from nccl.m2n import DistTensor, Handle, Mesh, Shard


def _parse_shape(value: str) -> tuple[int, int]:
    """Parse a valid global BF16 weight shape."""
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be '<rows>,<columns>'")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be positive, got {shape}"
        )
    return shape


def _select_cuda_device(device_id: int) -> Any:
    """Select and return this worker's CUDA device."""
    try:
        from cuda.core import Device
    except ImportError as exc:  # pragma: no cover - requires the CUDA runtime.
        raise SystemExit("ERROR: this sample requires cuda-core") from exc

    cuda_devices = Device.get_all_devices()
    if len(cuda_devices) <= device_id:
        raise RuntimeError(
            f"device {device_id} needs a CUDA device, but only "
            f"{len(cuda_devices)} device(s) are visible"
        )

    cuda_device = cuda_devices[device_id]
    cuda_device.set_current()
    return cuda_device


def _expected_shard(
    jnp: Any,
    shape: tuple[int, int],
    global_shape: tuple[int, int],
    row_start: int,
    column_start: int,
) -> Any:
    """Build the expected values for one global array shard."""
    rows = jnp.arange(row_start, row_start + shape[0], dtype=jnp.int32)[:, None]
    columns = jnp.arange(
        column_start, column_start + shape[1], dtype=jnp.int32
    )[None, :]
    return jnp.remainder(rows * global_shape[1] + columns, 251).astype(jnp.bfloat16)


def _validate_destination(
    jax: Any,
    jnp: Any,
    destination: Any,
    sharding: Any,
    global_shape: tuple[int, int],
    local_shape: tuple[int, int],
    shard_index: int,
) -> bool:
    """Check one destination shard against its expected layout and values."""
    if destination.shape != global_shape or destination.dtype != jnp.bfloat16:
        return False
    if not destination.sharding.is_equivalent_to(sharding, len(global_shape)):
        return False

    shards = destination.addressable_shards
    if len(shards) != 1:
        return False
    shard = shards[0]
    column_start = shard_index * local_shape[1]
    expected_index = (
        (0, global_shape[0]),
        (column_start, column_start + local_shape[1]),
    )
    for dim, (item, (expected_start, expected_stop)) in enumerate(
        zip(shard.index, expected_index)
    ):
        if not isinstance(item, slice):
            return False
        start = 0 if item.start is None else item.start
        stop = global_shape[dim] if item.stop is None else item.stop
        if (start, stop, item.step) != (expected_start, expected_stop, None):
            return False
    if tuple(shard.data.shape) != local_shape:
        return False

    expected = _expected_shard(jnp, local_shape, global_shape, 0, column_start)
    return bool(jax.device_get(jnp.all(shard.data == expected)))


def _restore_unique_id(value: bytes) -> object:
    """Restore an NCCL unique ID shared by the parent process."""
    unique_id = nccl.get_unique_id(empty=True)
    serialized = np.frombuffer(value, dtype=unique_id.as_ndarray.dtype)
    if serialized.shape != unique_id.as_ndarray.shape:
        raise RuntimeError(
            f"unexpected NCCL unique ID shape {serialized.shape}; "
            f"expected {unique_id.as_ndarray.shape}"
        )
    unique_id.as_ndarray[...] = serialized
    return unique_id


def _run_worker(
    rank: int,
    coordinator_address: str,
    unique_id_value: bytes,
    global_shape: tuple[int, int],
    source_ranks: int,
    destination_ranks: int,
    timeout_seconds: float,
    worker_barrier: Any,
) -> None:
    """Run one rank of the JAX row-to-column M2N reshard."""
    world_size = source_ranks + destination_ranks
    cuda_device = _select_cuda_device(rank)
    # JAX may own its own NCCL communicators. M2N needs this separate one
    # because the M2N APIs are outside JAX's distributed model.
    nccl_comm = nccl.Communicator.init(
        nranks=world_size,
        rank=rank,
        unique_id=_restore_unique_id(unique_id_value),
    )
    m2n_handle: Handle | None = None
    m2n_stream = None
    jax_initialized = False

    try:
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh as JaxMesh
        from jax.sharding import NamedSharding, PartitionSpec

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=world_size,
            process_id=rank,
            local_device_ids=[rank],
        )
        jax_initialized = True
        devices = jax.devices()
        local_devices = jax.local_devices()
        if len(devices) != world_size or len(local_devices) != 1:
            raise RuntimeError(
                f"expected {world_size} global and 1 local JAX device, got "
                f"{len(devices)} global and {len(local_devices)} local"
            )
        if devices[rank] != local_devices[0]:
            raise RuntimeError("JAX global device order does not match rank order")

        source_sharding = NamedSharding(
            JaxMesh(np.asarray(devices[:source_ranks], dtype=object), ("src",)),
            PartitionSpec("src", None),
        )
        destination_sharding = NamedSharding(
            JaxMesh(np.asarray(devices[source_ranks:], dtype=object), ("dst",)),
            PartitionSpec(None, "dst"),
        )
        source_local_shape = (global_shape[0] // source_ranks, global_shape[1])
        destination_local_shape = (
            global_shape[0],
            global_shape[1] // destination_ranks,
        )
        is_source = rank < source_ranks

        source_buffer = None
        if is_source:
            source_local = _expected_shard(
                jnp,
                source_local_shape,
                global_shape,
                rank * source_local_shape[0],
                0,
            )
            source_array = jax.make_array_from_single_device_arrays(
                global_shape, source_sharding, [source_local]
            )
            source_array.block_until_ready()
            addressable_source_shards = source_array.addressable_shards
            if len(addressable_source_shards) != 1:
                raise RuntimeError("source JAX array must have one addressable shard")
            source_buffer = int(
                addressable_source_shards[0].data.unsafe_buffer_pointer()
            )

        m2n_stream = cuda_device.create_stream()
        destination_pointer = 0
        if not is_source:
            # Use a mutable JAX Ref, then freeze it after M2N's stream completes.
            # The raw-pointer write is external to JAX, so stream sync is required.
            destination_local = jnp.empty(
                destination_local_shape, dtype=jnp.bfloat16, device=local_devices[0]
            )
            destination_local.block_until_ready()
            destination_ref = jax.new_ref(destination_local)
            destination_pointer = int(destination_ref.unsafe_buffer_pointer())

        # Describe the row-sharded JAX source buffer to M2N.
        source_descriptor = DistTensor(
            source_buffer if is_source else None,
            local_shape=source_local_shape,
            dtype="bfloat16",
            mesh=Mesh([source_ranks], start_rank=0),
            placements=[Shard(0)],
        )
        # Describe the JAX-owned column-sharded destination buffer.
        destination_descriptor = DistTensor(
            destination_pointer if not is_source else None,
            local_shape=destination_local_shape,
            dtype="bfloat16",
            mesh=Mesh([destination_ranks], start_rank=source_ranks),
            placements=[Shard(1)],
        )
        # Create the per-process M2N handle.
        m2n_handle = Handle.create()
        # Move the source row shards to the destination column shards.
        m2n_handle.reshard(
            nccl_comm,
            source_descriptor,
            destination_descriptor,
            stream=int(m2n_stream.handle),
        )
        m2n_stream.sync()
        if not is_source and (
            int(destination_ref.unsafe_buffer_pointer()) != destination_pointer
        ):
            raise RuntimeError("JAX destination buffer pointer changed during M2N")
        worker_barrier.wait(timeout=timeout_seconds)

        local_valid = True
        if not is_source:
            destination_local = jax.freeze(destination_ref)
            destination_array = jax.make_array_from_single_device_arrays(
                global_shape, destination_sharding, [destination_local]
            )
            local_valid = _validate_destination(
                jax,
                jnp,
                destination_array,
                destination_sharding,
                global_shape,
                destination_local_shape,
                rank - source_ranks,
            )
        worker_barrier.wait(timeout=timeout_seconds)
        raise SystemExit(0 if local_valid else 1)
    except Exception as exc:
        print(f"[Rank {rank}] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    finally:
        if m2n_stream is not None:
            m2n_stream.sync()
        if m2n_handle is not None:
            # Release this process's M2N handle after the transfer is complete.
            m2n_handle.destroy()
        nccl_comm.destroy()
        if m2n_stream is not None:
            m2n_stream.close()
        if jax_initialized:
            jax.distributed.shutdown()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments for the sample."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        default=(8, 8),
        help="global BF16 weight shape; use 4096,4096 for qualification",
    )
    parser.add_argument(
        "--source-ranks",
        type=int,
        default=2,
        help="number of row-sharded source ranks",
    )
    parser.add_argument(
        "--destination-ranks",
        type=int,
        default=2,
        help="number of column-sharded destination ranks",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300,
        help="timeout for worker synchronization",
    )
    args = parser.parse_args(argv)
    if args.source_ranks <= 0 or args.destination_ranks <= 0:
        parser.error("source and destination ranks must be positive")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.shape[0] % args.source_ranks or args.shape[1] % args.destination_ranks:
        parser.error(
            f"rows must be divisible by {args.source_ranks} and columns by "
            f"{args.destination_ranks}, got {args.shape}"
        )
    return args


def _coordinator_address() -> str:
    """Find a candidate local JAX coordinator address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def main(argv: Sequence[str] | None = None) -> int:
    """Launch and validate the local JAX worker processes."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    world_size = args.source_ranks + args.destination_ranks
    context = multiprocessing.get_context("spawn")
    coordinator_address = _coordinator_address()
    unique_id_value = nccl.get_unique_id().as_ndarray.tobytes()
    worker_barrier = context.Barrier(world_size)
    workers = [
        context.Process(
            target=_run_worker,
            args=(
                rank,
                coordinator_address,
                unique_id_value,
                args.shape,
                args.source_ranks,
                args.destination_ranks,
                args.timeout_seconds,
                worker_barrier,
            ),
            name=f"jax-m2n-rank-{rank}",
        )
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()

    remaining = {worker.sentinel: worker for worker in workers}
    while remaining:
        worker = remaining.pop(wait(remaining)[0])
        worker.join()
        if worker.exitcode != 0:
            for other in remaining.values():
                other.terminate()
            for other in remaining.values():
                other.join()
            return 1

    print("JAX M2N reshard validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
