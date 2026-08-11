"""Benchmark entry point (historical path, L7.3).

Shim to the Hydra CLI: ``python -m studies.runner.benchmark`` is
equivalent to ``python -m csd_observer.cli.main``. Kept so documented
study commands keep working until ``studies/`` is removed at L9.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from csd_observer.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
