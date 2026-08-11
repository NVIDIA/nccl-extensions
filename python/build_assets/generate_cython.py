#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
Generate Cython bindings for the nccl-extensions libraries using cybind.

Ported from nccl4py's ``build_assets/generate_cython.py``, trimmed to the
generated targets this repo owns (``nccl_ep`` and ``nccl_m2n``). Output goes
to ``python/nccl/_extensions/bindings/`` as flat sibling modules.

Unlike nccl4py, the bound headers are *not* checked in under
``cybind/headers/<libname>/``: nccl_ep's public header lives in this repo, so
it is staged straight from ``nccl_ep/include/`` and the version is read from
its ``NCCL_EP_{MAJOR,MINOR,PATCH}`` macros. Only headers this repo does not
own -- currently just ``nccl.h`` -- are pinned under ``cybind/headers/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from packaging.version import Version


# cybind repository configuration
CYBIND_COMMIT = "cde8bae486ff7ff88accf3cfff4e62527fd06199"
CYBIND_SSH_URL = "ssh://git@gitlab-master.nvidia.com:12051/xiakunl/cybind.git"

# Script directory for resolving default paths
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
REPO_ROOT = PYTHON_DIR.parent

# Shared paths: one cybind/ assets dir feeds every target; cybind emits into
# one bindings package (flat sibling layout).
ASSETS_DIR = SCRIPT_DIR / "cybind"
BINDINGS_DIR = PYTHON_DIR / "nccl" / "_extensions" / "bindings"

# Pinned copies of headers owned by other repos, laid out the way cybind's own
# assets/headers/ does: <libname>/<version>/. Bumping NCCL_PIN means dropping
# the matching nccl.h here and regenerating.
HEADERS_DIR = ASSETS_DIR / "headers"
NCCL_PIN = Version("2.30.4")

# Static files in templates/ that cybind doesn't process -- copied verbatim
# into BINDINGS_DIR after cybind finishes.
STATIC_FILES = (
    "__init__.py",
    "_internal/__init__.py",
    "_internal/utils.pxd",
    "_internal/utils.pyx",
)

# Every target emits into ``nccl._extensions.bindings.*``, so cybind looks up
# their templates under one shared subtree.
_TEMPLATES_RELPATH = Path("nccl", "_extensions", "bindings")

# Global logger - will be configured in main()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Target:
    """Per-binding-target configuration for cybind."""

    name: str  # cybind library name (top-level key in YAML)
    version: Version
    # Headers staged into cybind's assets/headers/<name>/<version>/, as
    # {path relative to that dir: source file}. Must cover the transitive
    # includes of the config's ``data.headers`` entry, laid out the way
    # those #include directives spell them.
    headers: dict[str, Path]


def _read_version(header: Path, prefix: str) -> Version:
    """Read ``<prefix>_{MAJOR,MINOR,PATCH}`` #defines out of a C header."""
    text = header.read_text()
    parts = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^#define\s+{prefix}_{component}\s+(\d+)", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"No {prefix}_{component} #define found in {header}")
        parts.append(match.group(1))
    return Version(".".join(parts))


def _nccl_ep_target() -> Target:
    """Bind nccl_ep against this repo's own headers.

    The staged layout mirrors what ``nccl_ep/CMakeLists.txt`` installs (public
    header at the root, everything else under ``nccl_ep/``), which is how
    ``nccl_ep.h``'s own #include directives spell them.
    """
    include_dir = REPO_ROOT / "nccl_ep" / "include"
    public_header = include_dir / "nccl_ep.h"
    if not public_header.is_file():
        raise RuntimeError(f"nccl_ep public header not found: {public_header}")
    return Target(
        name="nccl_ep",
        version=_read_version(public_header, "NCCL_EP"),
        headers={
            "nccl_ep.h": public_header,
            "nccl_ep/ep_enums.h": include_dir / "ep_enums.h",
            "nccl.h": HEADERS_DIR / "nccl" / str(NCCL_PIN) / "nccl.h",
        },
    )


def _nccl_m2n_target() -> Target:
    """Bind nccl_m2n against its public header.

    The generated declarations encode the public ABI, so downstream wheel
    builds do not need a separate M2N header installation.  ``nccl.h`` stays
    pinned with the other third-party headers because M2N imports its NCCL
    result and datatype definitions.
    """
    public_header = REPO_ROOT / "nccl_m2n" / "src" / "nccl_m2n.h"
    if not public_header.is_file():
        raise RuntimeError(f"nccl_m2n public header not found: {public_header}")
    return Target(
        name="nccl_m2n",
        version=_read_version(public_header, "NCCL_M2N"),
        headers={
            "nccl_m2n.h": public_header,
            "nccl.h": HEADERS_DIR / "nccl" / str(NCCL_PIN) / "nccl.h",
        },
    )


def _stamp_asset_yaml(yaml_path: Path, version: Version) -> None:
    """Stamp ``data.versions: - - X.Y.Z`` in the asset YAML in place.

    Text-level edits (vs. parse -> mutate -> dump) preserve the file's
    comments, ordering, and formatting that ``yaml.safe_dump`` would strip.
    The substitution is idempotent: the regex matches any prior value, so
    reruns at the same version are no-ops.
    """
    text = yaml_path.read_text()
    text, n_versions = re.subn(r"(versions:\n\s*- - )\S+", rf"\g<1>{version}", text)
    if n_versions != 1:
        raise RuntimeError(
            f"Expected exactly one `versions:` block in {yaml_path}, found {n_versions}"
        )
    yaml_path.write_text(text)


def clone_cybind(cybind_dir: Path) -> None:
    try:
        subprocess.run(
            ["git", "clone", CYBIND_SSH_URL, str(cybind_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "switch", "--detach", CYBIND_COMMIT],
            cwd=cybind_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(f"Cloned -> {cybind_dir}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone cybind: {e.stderr}") from e


def prepare_assets(cybind_dir: Path, targets: list[Target]) -> None:
    """Stage our targets' configs, headers, and templates into cybind's
    shared ``assets/`` -- only the slots we own (one ``configs/*.cybind.yaml``
    per target, per-target ``headers/<name>/<version>/``, and the shared
    ``templates/nccl/_extensions/bindings/`` subtree). Sibling files for
    other libs are left untouched."""
    cybind_assets = cybind_dir / "cybind" / "assets"

    for target in targets:
        config_filename = f"{target.name}.cybind.yaml"
        config_src = ASSETS_DIR / "configs" / config_filename
        if not config_src.exists():
            raise FileNotFoundError(f"Config file not found: {config_src}")
        _stamp_asset_yaml(config_src, target.version)
        config_dst = cybind_assets / "configs" / config_filename
        config_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_src, config_dst)
        logger.debug(f"Staged {config_filename} (version={target.version})")

        headers_dst = cybind_assets / "headers" / target.name / str(target.version)
        if headers_dst.exists():
            shutil.rmtree(headers_dst)
        for relpath, src in target.headers.items():
            if not src.is_file():
                raise FileNotFoundError(f"Header not found: {src}")
            dst = headers_dst / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logger.debug(f"Staged headers/{target.name}/{target.version}/{relpath} <- {src}")

    templates_src = ASSETS_DIR / "templates" / _TEMPLATES_RELPATH
    if not templates_src.is_dir():
        raise FileNotFoundError(f"Templates dir not found: {templates_src}")
    templates_dst = cybind_assets / "templates" / _TEMPLATES_RELPATH
    if templates_dst.exists():
        shutil.rmtree(templates_dst)
    templates_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(templates_src, templates_dst)
    logger.debug(f"Staged templates/{_TEMPLATES_RELPATH}/")

def run_cybind(cybind_dir: Path, libnames: list[str], output_dir: Path) -> None:
    """Run cybind once for all libraries in libnames.

    Cybind's ``--generate`` is multi-valued; passing all libnames in one
    invocation amortizes the venv setup. Each library's headers must live at
    ``cybind/assets/headers/<libname>/<version>/`` (cybind's default lookup
    when no ``--input-dir`` is passed); ``prepare_assets`` handles that.

    Args:
        - cybind_dir: Path to cybind repository.
        - libnames: Library keys to generate (top-level YAML keys).
        - output_dir: Cybind's ``--output`` value. Cybind emits each library
          into ``output_dir/<YAML module path with leaf stripped>/``.

    Note:
        - Requires CUDA_PATH for CUDA header resolution.
        - Uses uv to create an isolated venv and install cybind.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "--isolated",
        "--with",
        str(cybind_dir),
        "-m",
        "cybind",
        "--generate",
        *libnames,
        "--output",
        str(output_dir),
    ]

    logger.debug(f"Command: {' '.join(cmd)}")
    logger.debug(f"Working directory: {cybind_dir}")
    logger.debug(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'not set')}")

    try:
        result = subprocess.run(cmd, cwd=cybind_dir, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.debug("cybind execution stdout:")
            for line in result.stdout.splitlines():
                logger.debug(f"  {line}")
        if result.stderr:
            logger.debug("cybind execution stderr:")
            for line in result.stderr.splitlines():
                logger.debug(f"  {line}")
        logger.debug("Successfully generated bindings")
    except subprocess.CalledProcessError as e:
        logger.error(f"cybind failed with exit code {e.returncode}")
        if e.stdout:
            logger.error("=== stdout ===")
            for line in e.stdout.splitlines():
                logger.error(line)
        if e.stderr:
            logger.error("=== stderr ===")
            for line in e.stderr.splitlines():
                logger.error(line)
        raise RuntimeError("cybind execution failed") from e


@contextmanager
def backup_and_restore_on_failure(path: Path):
    """Restore the prior binding package if generation does not complete."""
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        shutil.rmtree(backup)
    if path.exists():
        logger.info(f">>> Backing up {path} -> {backup}")
        path.rename(backup)
    try:
        yield
    except BaseException:
        if path.exists():
            shutil.rmtree(path)
        if backup.exists():
            logger.error(f"Bindings generation failed; restoring {path}")
            backup.rename(path)
        raise
    else:
        if backup.exists():
            logger.info(f">>> Removing backup {backup}")
            shutil.rmtree(backup)


def verify_generated_m2n_loader_contract() -> None:
    """Check the generated M2N loader's atomic loading diagnostics contract."""
    loader = BINDINGS_DIR / "_internal" / "nccl_m2n_linux.pyx"
    text = loader.read_text()
    required = (
        'errors.append(f"{path}: not found")',
        'missing.append("ncclM2nGetLastError")',
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise RuntimeError(
            f"Generated {loader} does not preserve the M2N loader contract: {missing}"
        )


def verify_generated_m2n_import_layering() -> None:
    """Keep low-level bindings independent from the public M2N facade."""
    binding = BINDINGS_DIR / "nccl_m2n.pyx"
    text = binding.read_text()
    required = "from nccl._extensions._runtime import NATIVE_CALL_LOCK as _NATIVE_CALL_LOCK"
    if required not in text or "nccl.m2n" in text:
        raise RuntimeError(
            f"Generated {binding} imports public nccl.m2n state and can create an import cycle"
        )


# One entry per bound library. Each is a zero-arg factory so a missing header
# for one target raises with that target's own error message.
TARGETS = (_nccl_ep_target, _nccl_m2n_target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Cython bindings for nccl-extensions using cybind"
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=None,
        help="Path to CUDA installation (default: $CUDA_PATH or $CUDA_HOME)",
    )
    parser.add_argument(
        "--cybind-path",
        type=Path,
        default=None,
        help=(
            "Use a local cybind checkout at this path instead of cloning. "
            "Note: our targets' slots under the checkout's assets/ "
            "(configs/<name>.cybind.yaml, headers/<name>/<version>/, "
            "templates/nccl/_extensions/bindings/) are overwritten; sibling "
            "files for other libs are left alone. Default: clone "
            "CYBIND_SSH_URL at the pinned CYBIND_COMMIT into a temp dir."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (debug) output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Fail fast on missing tools rather than after the cybind clone.
    required_tools = ["uv"]
    if args.cybind_path is None:
        required_tools.append("git")
    for tool in required_tools:
        if shutil.which(tool) is None:
            logger.error(f"{tool} not found on PATH")
            return 1

    if args.cybind_path is not None and not args.cybind_path.is_dir():
        logger.error(f"--cybind-path is not a directory: {args.cybind_path}")
        return 1

    # CUDA_PATH (cybind uses it to find cuda.h)
    if args.cuda_home:
        if not args.cuda_home.exists():
            logger.error(f"CUDA home not found at {args.cuda_home}")
            return 1
        cuda_path = str(args.cuda_home)
    else:
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if not cuda_path:
            logger.error("Provide --cuda-home or set CUDA_PATH or CUDA_HOME")
            return 1
        if not Path(cuda_path).is_dir():
            logger.error(f"CUDA path is not a directory: {cuda_path}")
            return 1

    os.environ["CUDA_PATH"] = cuda_path
    logger.debug(f"CUDA: {cuda_path}")

    try:
        targets = [make_target() for make_target in TARGETS]
    except (RuntimeError, FileNotFoundError) as e:
        logger.error(str(e))
        return 1

    # cybind tree: use --cybind-path in place, or clone into a temp dir.
    if args.cybind_path is not None:
        logger.info(f">>> Using local cybind from {args.cybind_path} (in place)")
        cybind_ctx = nullcontext(args.cybind_path)
    else:
        cybind_ctx = tempfile.TemporaryDirectory(prefix="nccl_extensions_cybind_")

    with cybind_ctx as cybind_dir_:
        cybind_dir = Path(cybind_dir_)
        if args.cybind_path is None:
            logger.info(">>> Cloning cybind...")
            clone_cybind(cybind_dir)

        logger.info(">>> Preparing cybind assets")
        prepare_assets(cybind_dir, targets)

        # Back up the real bindings dir, then run cybind with --output pointing
        # at PYTHON_DIR so the emitted <output>/<YAML module path with leaf
        # stripped>/ path lands directly on BINDINGS_DIR. The context manager
        # removes the backup on success and restores it on failure.
        with backup_and_restore_on_failure(BINDINGS_DIR):
            logger.info(">>> Running cybind for all targets")
            run_cybind(cybind_dir, [t.name for t in targets], PYTHON_DIR)

            # Copy shared static template files cybind doesn't process.
            logger.debug("Copying static files from templates...")
            templates_root = ASSETS_DIR / "templates" / _TEMPLATES_RELPATH
            for rel in STATIC_FILES:
                src = templates_root / rel
                if not src.exists():
                    continue
                dst = BINDINGS_DIR / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"  {rel}")
                shutil.copy2(src, dst)

            verify_generated_m2n_loader_contract()
            verify_generated_m2n_import_layering()

    logger.info("=" * 60)
    logger.info(f"Bindings location: {BINDINGS_DIR}")
    logger.info(f"Generated: {', '.join(f'{t.name} {t.version}' for t in targets)}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
