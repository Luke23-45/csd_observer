"""Phase FINALIZE (R1): summarize, complete, ledger.

``summarize_run`` must finish before the ``.completed`` lifecycle
marker is written — the marker is the reliable signal that the
``tables/`` subtree and ``metrics/metrics.json`` exist (invariant the
ledger tests enforce). If summarize raises, the driver's failure
handler flips the marker to ``.failed`` instead.
"""

from __future__ import annotations

from csd_observer.orchestration.run_context import RunContext
from csd_observer.outputs.summarize import summarize_run


def finalize(ctx: RunContext) -> None:
    """Summarize the run, then mark completed in writer + ledger."""
    writer = ctx.writer
    if writer is None:
        raise RuntimeError("finalize() called before prepare()")
    summarize_run(writer.root)
    writer.mark_completed()
    if ctx.ledger is not None:
        ctx.ledger.update_status(ctx.run_id, "completed")


__all__ = ["finalize"]
