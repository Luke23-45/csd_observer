"""Yaml file <-> store.py node alignment (R3.1 reversal).

The group *values* live in ``configs/<group>/<name>.yaml``; the
``store.py`` dataclasses are the in-code schema contract (they feed the
R0.1 defaults comparison and the validate-time schema check). These
tests pin the two together: every node must have a yaml file, and every
file must carry exactly the node's non-None values. A divergence here
would silently change R0.1's defaults source or weaken the schema check
without any run failing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from csd_observer.config.store import _GROUP_NODES, register_configs

_REPO = Path(__file__).resolve().parent.parent.parent
_CONFIGS = _REPO / "configs"

# Populated at import time so the parametrize below sees every node.
register_configs()


def _load_yaml(group: str, name: str) -> dict[str, Any]:
    path = _CONFIGS / group / f"{name}.yaml"
    assert path.is_file(), f"missing yaml file {path}"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _drop_empty(value: Any) -> Any:
    """Recursively drop None / empty containers so the yaml files (which
    omit them) compare equal to the node's ``asdict`` at every level."""
    if isinstance(value, dict):
        return {
            k: _drop_empty(v)
            for k, v in value.items()
            if v is not None and v != [] and v != {}
        }
    if isinstance(value, list):
        return [_drop_empty(v) for v in value if v is not None and v != [] and v != {}]
    return value


def _node_values(node: Any) -> dict[str, Any]:
    return _drop_empty(dataclasses.asdict(node))


def test_every_node_has_a_yaml_file() -> None:
    files = {
        (d.name, f.stem)
        for d in _CONFIGS.iterdir()
        if d.is_dir() and d.name != "hydra"
        for f in d.glob("*.yaml")
    }
    assert files == set(_GROUP_NODES)


@pytest.mark.parametrize("group, name", sorted(_GROUP_NODES))
def test_yaml_matches_store_node(group: str, name: str) -> None:
    node = _GROUP_NODES[(group, name)]
    assert _load_yaml(group, name) == _node_values(node), (
        f"{group}/{name}.yaml diverges from its store.py node"
    )
