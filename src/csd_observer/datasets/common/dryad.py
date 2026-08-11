"""Token-aware Dryad v2 client with deterministic manifest resolution.

The client only handles transport and metadata.  Dataset-specific parsers do
not depend on Dryad response details and can therefore be run offline from a
manual drop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from .errors import DatasetError, DatasetErrorCode


@dataclass(frozen=True)
class DryadFile:
    path: str
    size: int
    md5: str
    download_url: str | None = None


class DryadClient:
    def __init__(self, *, base_url: str = "https://datadryad.org/api/v2", token: str | None = None,
                 session: requests.Session | None = None, retries: int = 3, backoff: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = session or requests.Session()
        self.retries = max(1, int(retries))
        self.backoff = max(0.0, float(backoff))

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, headers=headers, timeout=kwargs.pop("timeout", 30), **kwargs)
            except requests.RequestException as exc:
                if attempt + 1 == self.retries:
                    raise DatasetError(DatasetErrorCode.INGEST_NETWORK, str(exc)) from exc
                time.sleep(self.backoff * (2 ** attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < self.retries:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff * (2 ** attempt)
                    time.sleep(delay)
                    continue
            if response.status_code in (401, 403):
                raise DatasetError(DatasetErrorCode.INGEST_AUTH, f"Dryad returned HTTP {response.status_code}")
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise DatasetError(DatasetErrorCode.INGEST_NETWORK, str(exc)) from exc
            return response
        raise DatasetError(DatasetErrorCode.INGEST_NETWORK, f"failed to GET {url}")

    def files_for_doi(self, doi: str) -> list[DryadFile]:
        encoded = quote(f"doi:{doi}", safe="")
        dataset = self._get(f"{self.base_url}/datasets/{encoded}").json()
        version_href = dataset.get("_links", {}).get("stash:version", {}).get("href")
        if not version_href:
            raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "Dryad response has no version link")
        files_url = version_href.rstrip("/") + "/files"
        out: list[DryadFile] = []
        while files_url:
            page = self._get(files_url).json()
            for item in page.get("_embedded", {}).get("stash:files", page.get("files", [])):
                links = item.get("_links", {})
                download = links.get("stash:download", {}).get("href")
                digest = item.get("digest")
                digest_type = str(item.get("digestType") or "md5").lower()
                if not item.get("path") or digest is None:
                    raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "Dryad file entry lacks path/digest")
                if digest_type != "md5":
                    raise DatasetError(
                        DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
                        f"Dryad file {item['path']!r} uses unsupported digestType {digest_type!r} (only md5 is verifiable)",
                    )
                out.append(DryadFile(str(item["path"]), int(item.get("size", -1)), str(digest).lower(), download))
            next_link = page.get("_links", {}).get("next", {})
            files_url = next_link.get("href") if isinstance(next_link, dict) else None
        return out

    def download(self, file: DryadFile, destination: str) -> None:
        if not self.token:
            raise DatasetError(DatasetErrorCode.INGEST_AUTH, "DRYAD_API_TOKEN is required for downloads")
        if not file.download_url:
            raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"no download URL for {file.path}")
        response = self._get(file.download_url, stream=True, timeout=120)
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


__all__ = ["DryadClient", "DryadFile"]
