from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_csv(
    *,
    url: str,
    data_dir: str | Path,
    known_hash: Optional[str] = None,
    force: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / "df_chick.csv"
    sentinel_path = data_dir / ".df_chick.downloaded"

    if not force and sentinel_path.exists() and out_path.exists():
        if known_hash is None or _compute_sha256(out_path) == known_hash:
            return out_path
        print("Hash mismatch; re-downloading...")

    print(f"Downloading chick heart CSV from {url} ...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(resp, tmp)
            tmp_path = Path(tmp.name)

    tmp_path.replace(out_path)
    sentinel_path.write_text("ok\n")
    print(f"Saved to {out_path}")

    if known_hash is not None:
        actual = _compute_sha256(out_path)
        if actual != known_hash:
            os.remove(sentinel_path)
            raise RuntimeError(
                f"Hash mismatch for {out_path}\n"
                f"  expected: {known_hash}\n"
                f"  actual:   {actual}"
            )

    return out_path
