"""Environment provenance capture."""

from __future__ import annotations

import datetime
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def capture_provenance(
    cwd: str | Path | None = None,
    include_pip: bool = True,
) -> dict[str, Any]:
    """Capture environment provenance for reproducibility.

    Collects git commit, Python version, platform info, GPU details
    (if available), installed pip packages, and a timestamp.

    Args:
        cwd: Working directory to check for git. Defaults to ``os.getcwd()``.
        include_pip: If ``True``, enumerate installed packages via
            ``importlib.metadata``.

    Returns:
        A dictionary suitable for serialising to JSON.
    """
    info: dict[str, Any] = {}

    # Timestamp
    info["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

    # Git commit hash
    info["git_commit_hash"] = _git_commit(cwd)

    # Python / platform
    info["python_version"] = sys.version.split()[0]
    info["platform"] = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    info["hostname"] = platform.node()

    # GPU
    info["gpu_info"] = _gpu_info()

    # Pip packages
    if include_pip:
        info["pip_packages"] = _pip_packages()

    return info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_commit(cwd: str | Path | None) -> str | None:
    """Return short git commit hash, or *None*."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _gpu_info() -> str | None:
    """Try to detect GPU via ``nvidia-smi``."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            return ", ".join(gpus) if gpus else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _pip_packages() -> list[dict[str, str]]:
    """Enumerate installed packages via importlib.metadata."""
    packages: list[dict[str, str]] = []
    seen: set[str] = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        packages.append({
            "name": name,
            "version": dist.version,
        })
    packages.sort(key=lambda p: p["name"].lower())
    return packages


__all__ = ["capture_provenance"]
