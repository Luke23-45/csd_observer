"""TAC dataset subpackage (Dryad subcritical-Hopf TDMS traces)."""
from csd_observer.datasets.tac.ingest import ingest
from csd_observer.datasets.tac.process import (
    auto_select_channel,
    bandpass_envelope,
    extract_channel,
    hilbert_envelope,
    inspect_tdms,
    process,
    validate_annotation,
)

__all__ = ["auto_select_channel", "bandpass_envelope", "extract_channel", "hilbert_envelope",
           "ingest", "inspect_tdms", "process", "validate_annotation"]
