"""Environment fingerprint for one run.

Captures the dependency versions, git commit, host info, and dataset
content hashes that together make a run reproducible from its output
directory alone.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_version(modname: str) -> str | None:
    try:
        m = __import__(modname)
        return getattr(m, "__version__", None)
    except Exception:
        return None


def collect_environment(
    *,
    dataset_manifest_hashes: dict[str, str] | None = None,
    config_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single environment.json payload."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "packages": {
            "torch": _safe_version("torch"),
            "numpy": _safe_version("numpy"),
            "pytorch_lightning": _safe_version("pytorch_lightning"),
            "torchmetrics": _safe_version("torchmetrics"),
            "hydra-core": _safe_version("hydra"),
            "omegaconf": _safe_version("omegaconf"),
            "scikit-learn": _safe_version("sklearn"),
            "nptdms": _safe_version("nptdms"),
            "requests": _safe_version("requests"),
            "filelock": _safe_version("filelock"),
        },
        "git_sha": _git_sha(),
        "dataset_manifest_hashes": dataset_manifest_hashes or {},
        "config_hash": config_hash,
    }
    if extra:
        env["extra"] = extra
    return env


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def hash_config(config: Any) -> str:
    """Stable hash of a (possibly nested) config; canonical JSON."""
    import json

    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["collect_environment", "hash_config", "hash_file"]
