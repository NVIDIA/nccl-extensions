#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-host (mp.spawn) OR multi-node (torchrun/srun) launcher for the FT tests
and benchmarks.

Why: on GB200 NVL nodes each host exposes only its local GPUs (~4), but the NVLink
domain — and therefore the LSA collectives — spans many hosts over the NVSwitch
fabric. So a 32-GPU FT run is 8 hosts x 4 GPUs, and the single-host mp.spawn path
cannot reach it. The FT collectives are unchanged (LSA over the whole domain); only
process launch + rendezvous differ.

Modes:
  - Launched (torchrun / srun set RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT):
    this process IS one rank. The worker runs directly with device = LOCAL_RANK and
    the store at MASTER_ADDR:MASTER_PORT.
  - Single host (no launcher env): mp.spawn `nranks` local processes on 127.0.0.1
    with device = rank (unchanged behavior).

Worker signature:
    worker(rank, world_size, local_rank, master_addr, master_port, *args)
It should end with sys.exit(code) (0 = pass). In single-host mode launch() returns
the list of per-rank exit codes; under a launcher it does not return (each process
is one rank and exits on its own).

Usage in a script's main():
    codes = ft_launch.launch(_rank_worker, nranks, args)
    if codes is not None:                    # single-host aggregation
        ok = all(c == 0 for c in codes)
        sys.exit(0 if ok else 1)
"""

import os
import socket
import sys
import multiprocessing as mp


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0)); port = s.getsockname()[1]; s.close()
    return port


def _torchrun_env() -> bool:
    return all(k in os.environ
               for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))


def _slurm_env() -> bool:
    return "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1


def launched() -> bool:
    """True if started under torchrun or srun (rendezvous env present)."""
    return _torchrun_env() or _slurm_env()


def _slurm_master_addr() -> str:
    """First hostname of the SLURM allocation (rank-0's node)."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST", "")
    try:
        import subprocess
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
        return out.split()[0].decode()
    except Exception:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")


def _context():
    """(rank, world, local_rank, master_addr, master_port) from the launcher env."""
    if _torchrun_env():
        return (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("LOCAL_RANK", 0)),
                os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]))
    # SLURM / srun
    rank  = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ.get("SLURM_LOCALID", 0))
    addr  = os.environ.get("MASTER_ADDR") or _slurm_master_addr()
    # All ranks in a job derive the SAME port (avoids collisions across jobs).
    port  = int(os.environ.get("MASTER_PORT",
                               str(29500 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000)))
    return rank, world, local, addr, port


def world_size_hint(default: int) -> int:
    """WORLD_SIZE under a launcher, else the caller's default (for banners)."""
    if _torchrun_env():
        return int(os.environ["WORLD_SIZE"])
    if _slurm_env():
        return int(os.environ["SLURM_NTASKS"])
    return default


def make_store(rank, world, addr, port, timeout_s=180):
    """TCPStore client/server at addr:port for this rank.

    Rank 0 hosts the store, except under torchrun: its elastic agent already
    serves a store on MASTER_PORT, so every rank connects as a client."""
    import datetime
    import torch.distributed as dist
    is_master = (rank == 0
                 and os.environ.get("TORCHELASTIC_USE_AGENT_STORE") != "True")
    return dist.TCPStore(addr, port, world, is_master=is_master,
                         timeout=datetime.timedelta(seconds=timeout_s))


def _entry(worker, rank, world, local, addr, port, wargs):
    # Runs one rank. The worker is expected to sys.exit() with its own code.
    worker(rank, world, local, addr, port, *wargs)


def launch(worker, nranks, *wargs):
    """Run `worker` across ranks. Returns per-rank exit codes in single-host mode;
    does not return under a launcher (this process is a single rank)."""
    if launched():
        rank, world, local, addr, port = _context()
        _entry(worker, rank, world, local, addr, port, wargs)
        sys.exit(0)   # only reached if the worker returned without sys.exit
    # single host
    world = nranks
    port  = _free_port()
    ctx   = mp.get_context("spawn")
    procs = [ctx.Process(target=_entry,
                         args=(worker, r, world, r, "127.0.0.1", port, wargs),
                         name=f"rank-{r}", daemon=False) for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return [p.exitcode for p in procs]
