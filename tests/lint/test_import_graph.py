"""L5.1: one-way import graph for the package layers (R5).

Enforces the documented dependency contract:

* ``cli`` → {``config``, ``orchestration``}
* ``orchestration`` → {``config``, ``datasets``, ``evaluation``,
  ``models``, ``outputs``, ``training``}
* ``training`` → {``models``, ``outputs``}
* ``evaluation`` → {``outputs``, ``config``}
* ``config`` → {``datasets``, ``models``}
* ``models``, ``datasets``, ``outputs`` → (nothing inside the project)

The check is a static AST walk of every ``csd_observer`` package
module: each intra-project import must be an allowed edge, same-package
imports are always fine, and anything outside ``csd_observer`` is
third-party. Keeps ``models`` free of ``training`` and
``orchestration`` free of a models→training back-edge forever.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "csd_observer"

_LAYERS = ("cli", "config", "datasets", "evaluation", "models",
           "orchestration", "outputs", "training")

#: allowed directed edges between top-level layers
_ALLOWED_EDGES = {
    ("cli", "config"),
    ("cli", "orchestration"),
    ("orchestration", "config"),
    ("orchestration", "datasets"),
    ("orchestration", "evaluation"),
    ("orchestration", "models"),
    ("orchestration", "outputs"),
    ("orchestration", "training"),
    ("training", "models"),
    ("training", "outputs"),
    ("evaluation", "outputs"),
    ("evaluation", "config"),
    ("config", "datasets"),
    ("config", "models"),
}


def _py_modules() -> list[Path]:
    return sorted(p for p in _SRC.rglob("*.py") if p.name != "__init__.py")


@pytest.fixture(scope="module", autouse=True)
def _module_paths() -> list[Path]:
    return _py_modules()


def test_every_layer_has_a_testable_module() -> None:
    """Each layer directory exists and contains at least one module."""
    for layer in _LAYERS:
        assert (_SRC / layer).is_dir(), f"layer {layer!r} missing"
        assert any((_SRC / layer).rglob("*.py")), f"layer {layer!r} has no modules"


def test_import_edges_respect_the_one_way_contract(_module_paths: list[Path]) -> None:
    violations: list[str] = []
    for path in _module_paths:
        rel = path.relative_to(_SRC)
        # ``__main__.py`` is the console entry point, part of the cli layer
        src_layer = "cli" if rel.parts[0] == "__main__.py" else rel.parts[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _check(alias.name, src_layer, rel, violations)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import: same package, always fine
                    continue
                if node.module:
                    _check(node.module, src_layer, rel, violations)
    assert not violations, (
        "one-way import violations:\n  " + "\n  ".join(sorted(violations))
    )


def _check(module: str, src_layer: str, rel: Path, violations: list[str]) -> None:
    if not module.startswith("csd_observer."):
        return  # third-party / stdlib
    target = module.split(".")[1]
    if target == src_layer:
        return
    if target not in _LAYERS:
        violations.append(f"{rel}: imports unknown layer {module!r}")
        return
    if (src_layer, target) not in _ALLOWED_EDGES:
        violations.append(
            f"{rel}: {src_layer} -> {target} is not an allowed edge "
            f"({module!r})"
        )


def test_no_executable_dependency_on_torch_in_indicator_path() -> None:
    """The indicator scoring path must stay torch-free (guards the
    ``models/common/indicators.py`` and ``models/common/mode.py``
    modules against a future accidental torch import)."""
    for name in ("indicators.py", "mode.py"):
        src = (_SRC / "models" / "common" / name).read_text(encoding="utf-8")
        assert "import torch" not in src
        assert "from torch" not in src
