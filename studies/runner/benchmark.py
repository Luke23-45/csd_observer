"""Benchmark entry point (historical path).

Delegates to ``csd_observer.benchmark`` — the suite logic lives in
``src/csd_observer/benchmark/`` (``methods.py`` catalog,
``experiments.py`` drivers, ``suite.py`` orchestration, ``__main__.py``
CLI). Kept as a shim so the documented commands
(``docs/notes/plan.md``) keep working; equivalent to
``python -m csd_observer.benchmark``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from csd_observer.benchmark.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
