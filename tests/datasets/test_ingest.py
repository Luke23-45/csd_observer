"""Ingestion (L3.17) + provisioning tests with mocked Dryad transport."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.states import IngestState
from tests.datasets import _fixtures

PAYLOAD = b"fake dryad archive bytes for checksum verification"

NAME = "tac"
RAW_DIR = Path("raw") / "tac"


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _ingest_config(*, mode: str, md5: str, wait_minutes: float = 0.0) -> dict:
    return {
        "name": NAME,
        "source": "dryad",
        "doi": "10.5061/dryad.4cj4k",
        "expected_files": [{"path": "Experimental_time_traces_tdms.zip", "md5": md5}],
        "download": {
            "mode": mode,
            "env_token_var": "DRYAD_API_TOKEN",
            "retries": 3,
            "backoff": 0.001,
            "wait_minutes": wait_minutes,
            "poll_interval_sec": 0.005,
        },
    }


def _fake_dryad_client(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    """Replace the DryadClient bound in ``ingest``'s namespace (module-level
    ``from .dryad import DryadClient``) with an offline fake."""
    from csd_observer.datasets.common import ingest as ingest_module
    from csd_observer.datasets.common.dryad import DryadFile

    class FakeDryadClient:
        def __init__(self, *, token: str | None = None, retries: int = 3, backoff: float = 1.0) -> None:
            self.token = token

        def files_for_doi(self, doi: str) -> list[DryadFile]:
            return [DryadFile("Experimental_time_traces_tdms.zip", len(payload), _md5(payload), "https://example.invalid/dl")]

        def download(self, file: DryadFile, destination: str) -> None:
            Path(destination).write_bytes(payload)

    monkeypatch.setattr(ingest_module, "DryadClient", FakeDryadClient)


def test_ingest_auto_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from csd_observer.datasets.common.ingest import ingest_raw

    _fake_dryad_client(monkeypatch, PAYLOAD)
    monkeypatch.setenv("DRYAD_API_TOKEN", "tok")
    state = ingest_raw(_ingest_config(mode="auto", md5=_md5(PAYLOAD)), tmp_path, NAME)
    assert state is IngestState.READY_RAW
    raw_file = tmp_path / RAW_DIR / "Experimental_time_traces_tdms.zip"
    assert raw_file.read_bytes() == PAYLOAD


def test_ingest_auto_checksum_mismatch_quarantines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from csd_observer.datasets.common.ingest import ingest_raw

    _fake_dryad_client(monkeypatch, b"corrupted bytes")
    monkeypatch.setenv("DRYAD_API_TOKEN", "tok")
    with pytest.raises(DatasetError) as exc_info:
        ingest_raw(_ingest_config(mode="auto", md5=_md5(PAYLOAD)), tmp_path, NAME)
    assert exc_info.value.code == DatasetErrorCode.INGEST_CHECKSUM
    raw = tmp_path / RAW_DIR
    quarantined = list(raw.glob("*.bad-*"))
    assert len(quarantined) == 1
    assert not (raw / "Experimental_time_traces_tdms.zip").exists()


def test_ingest_auto_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from csd_observer.datasets.common.ingest import ingest_raw

    monkeypatch.delenv("DRYAD_API_TOKEN", raising=False)
    with pytest.raises(DatasetError) as exc_info:
        ingest_raw(_ingest_config(mode="auto", md5=_md5(PAYLOAD)), tmp_path, NAME)
    assert exc_info.value.code == DatasetErrorCode.INGEST_AUTH


def test_ingest_manual_success(tmp_path: Path) -> None:
    from csd_observer.datasets.common.ingest import ingest_raw

    raw = tmp_path / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "Experimental_time_traces_tdms.zip").write_bytes(PAYLOAD)
    state = ingest_raw(_ingest_config(mode="manual", md5=_md5(PAYLOAD)), tmp_path, NAME)
    assert state is IngestState.READY_RAW


def test_ingest_manual_timeout(tmp_path: Path) -> None:
    from csd_observer.datasets.common.ingest import ingest_raw

    with pytest.raises(DatasetError) as exc_info:
        ingest_raw(_ingest_config(mode="manual", md5=_md5(PAYLOAD), wait_minutes=0.01), tmp_path, NAME)
    assert exc_info.value.code == DatasetErrorCode.INGEST_CHECKSUM
    assert "not found" in str(exc_info.value)


def test_ingest_manifest_skip_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached processed manifest with valid split files must short-circuit
    ingest before any remote fetch."""
    import numpy as np

    from csd_observer.datasets.common.ingest import ingest_raw
    from csd_observer.datasets.common.manifest import write_manifest

    processed = tmp_path / "processed" / "tac"
    for split_name in ("train", "val", "test"):
        split_dir = processed / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            split_dir / f"{split_name}.npz",
            features=np.zeros((1, 32, 1), dtype=np.float32),
            seq_lengths=np.array([32], dtype=np.int64),
            bifurcation_times=np.array([33.0], dtype=np.float64),
            is_positive=np.array([True], dtype=bool),
        )
    write_manifest(processed / "manifest.json", {
        "schema_version": "1.0",
        "dataset": {"doi": "x"},
        "processing": {"params": {}},
        "gates": {"passed": ["nonfinite", "min_length", "split_balance"]},
        "split": {
            "policy": "replicate_based",
            "splits": {
                "train": {"file": "train.npz"},
                "val": {"file": "val.npz"},
                "test": {"file": "test.npz"},
            },
        },
        "content_hash": "abc",
    })
    monkeypatch.setenv("DRYAD_API_TOKEN", "tok")
    state = ingest_raw(_ingest_config(mode="auto", md5=_md5(PAYLOAD)), tmp_path, NAME)
    assert state is IngestState.READY_PROCESSED
    assert not (tmp_path / RAW_DIR / "Experimental_time_traces_tdms.zip").exists()


def test_ingest_lock_serializes_concurrent_runs(tmp_path: Path) -> None:
    """A second process must wait for the per-dataset lock; the pipeline
    never double-ingests."""
    from filelock import FileLock

    from csd_observer.datasets.common.ingest import ingest_raw

    raw = tmp_path / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "Experimental_time_traces_tdms.zip").write_bytes(PAYLOAD)
    lock = FileLock(str(tmp_path / f".ingest.{NAME}.lock"))
    lock.acquire(timeout=1)

    result: list[IngestState] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(ingest_raw(_ingest_config(mode="manual", md5=_md5(PAYLOAD)), tmp_path, NAME))
        except BaseException as exc:  # noqa: BLE001 - test thread boundary
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=0.5)
    assert thread.is_alive(), "ingest must block while the per-dataset lock is held"
    lock.release()
    thread.join(timeout=10)
    assert not errors
    assert result == [IngestState.READY_RAW]


def test_dryad_client_retries_rate_limit_then_succeeds() -> None:
    """Rate-limit (429) with Retry-After/backoff is retried; the second
    attempt succeeds."""
    from csd_observer.datasets.common.dryad import DryadClient

    calls: list[int] = []

    class StubResponse:
        status_code = 429
        headers: dict[str, str] = {}

        def __init__(self, status_code: int, payload: dict) -> None:
            self.status_code = status_code
            self.payload = payload

        def json(self) -> dict:
            return self.payload

        def raise_for_status(self) -> None:
            return

    class StubSession:
        def get(self, url: str, **kwargs):  # noqa: ANN001
            calls.append(1)
            if len(calls) == 1:
                return StubResponse(429, {})
            return StubResponse(200, {
                "_links": {"stash:version": {"href": "https://example.invalid/api/v2/versions/9"}},
            })

    client = DryadClient(session=StubSession(), retries=3, backoff=0.001)
    dataset = client._get("https://example.invalid/api/v2/datasets/x")
    assert len(calls) == 2
    assert dataset.status_code == 200


def test_provision_synthetic_short_circuits(tmp_path: Path) -> None:
    from csd_observer.datasets.provision import provision_dataset

    state = provision_dataset("synthetic_fold", {"source": "synthetic"}, root=tmp_path)
    assert state is IngestState.READY_PROCESSED
    assert not (tmp_path / "synthetic_fold").exists()


def test_provision_unknown_dataset(tmp_path: Path) -> None:
    from csd_observer.datasets.provision import provision_dataset

    with pytest.raises(DatasetError) as exc_info:
        provision_dataset("nope", {"source": "dryad"}, root=tmp_path)
    assert exc_info.value.code == DatasetErrorCode.INGEST_MANIFEST_MISMATCH


def test_provision_wires_processor_and_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from csd_observer.datasets import provision as provision_module
    from csd_observer.datasets.provision import provision_dataset

    captured: dict = {}

    def fake_run_pipeline(config, root, name, processor, *, token=None, extra_validator=None, force_rebuild=False):
        captured["config"] = config
        captured["root"] = root
        captured["name"] = name
        captured["processor"] = processor
        captured["validator"] = extra_validator
        captured["force_rebuild"] = force_rebuild
        return IngestState.READY_PROCESSED

    monkeypatch.setattr(provision_module, "run_pipeline", fake_run_pipeline)
    config = {"source": "dryad", "name": "tac"}
    state = provision_dataset("tac", config, root=tmp_path)
    assert state is IngestState.READY_PROCESSED
    assert captured["root"] == tmp_path
    assert captured["name"] == "tac"
    assert callable(captured["processor"])
    assert callable(captured["validator"])


def test_provision_tac_end_to_end(tmp_path: Path) -> None:
    """Full provisioning on a mocked extracted raw dir: ingest (manual)
    -> process -> gates -> manifest -> registry load."""
    from csd_observer.datasets.provision import provision_dataset
    from csd_observer.datasets.registry import get_dataset

    _fixtures.build_custom_tac_raw(tmp_path, n_stationary=3, n_ramp=4)
    config = _fixtures.tac_config(source="dryad")
    config["expected_files"] = [{"path": "Experimental_time_traces_tdms", "md5": ""}]
    config["download"] = {"mode": "manual", "wait_minutes": 0.0}

    root = tmp_path / "dataset"
    raw = tmp_path / RAW_DIR
    dataset_raw = root / RAW_DIR
    dataset_raw.mkdir(parents=True, exist_ok=True)
    (raw / "Experimental_time_traces_tdms").replace(dataset_raw / "Experimental_time_traces_tdms")

    state = provision_dataset("tac", config, root=root)
    assert state is IngestState.READY_PROCESSED

    bundle = get_dataset("tac", {"data_root": str(root)})
    assert bundle["meta"]["bif_type"] == "subcritical_hopf"
    assert bundle["meta"]["n_replicates"] == 7
    assert bundle["signal"]["features"].shape[0] == 4
    assert bundle["null"]["features"].shape[0] == 3
