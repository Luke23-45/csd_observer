"""DaphniaExt ingestion entry point; processing is intentionally separate."""
from csd_observer.datasets.common.ingest import ingest_raw


def ingest(config: dict, root: str, *, token: str | None = None):
    return ingest_raw(config, root, token=token)


__all__ = ["ingest"]
