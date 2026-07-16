"""
Publication-grade matplotlib style system for the Krishi YOLO paper.

Design principles:
- **Premium typography**: Serif fonts (Times New Roman / Times) suited for academic papers.
- **Harmonious whitespace**: Generous margins, constrained layouts, and clean axes.
- **Colorblind-safe**: Palette tailored for qualitative/quantitative visual contrasts.
- **Flexible outputs**: Presets for single-column (4.5") and double-column (6.5") graphics.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Column widths (inches) for academic standard format
_COL_WIDTH = 4.5     # single column
_TEXT_WIDTH = 6.5    # double column / full text width
_DPI = 300

# Font sizes (pt)
_FONT_SIZE_TITLE = 13
_FONT_SIZE_LABEL = 11
_FONT_SIZE_TICK = 10
_FONT_SIZE_LEGEND = 10
_FONT_SIZE_ANNOT = 9

# Colorblind-safe palettes (Wong & Tol)
PALETTE = {
    "indigo":   "#332288",
    "teal":     "#44AA99",
    "green":    "#117733",
    "olive":    "#999933",
    "sand":     "#DDCC77",
    "rose":     "#CC6677",
    "wine":     "#882255",
    "purple":   "#AA4499",
    "sky":      "#88CCEE",
    "pink":     "#EE3377",
    "grey":     "#BBBBBB",
    "dark":     "#333333",
}

# Standard style dictionary
THESIS_RC = {
    # Fonts
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":            _FONT_SIZE_LABEL,
    "mathtext.fontset":     "cm",

    # Axes
    "axes.titlesize":       _FONT_SIZE_TITLE,
    "axes.labelsize":       _FONT_SIZE_LABEL,
    "axes.titleweight":     "normal",
    "axes.titlepad":        8,
    "axes.labelpad":        5,
    "axes.linewidth":       0.6,
    "axes.edgecolor":       "#333333",
    "axes.facecolor":       "white",
    "axes.grid":            True,
    "axes.grid.which":      "major",
    "axes.axisbelow":       True,
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # Grid
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.7,
    "grid.linestyle":       "--",

    # Ticks
    "xtick.labelsize":      _FONT_SIZE_TICK,
    "ytick.labelsize":      _FONT_SIZE_TICK,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.pad":      4,
    "ytick.major.pad":      4,

    # Legend
    "legend.fontsize":      _FONT_SIZE_LEGEND,
    "legend.frameon":        True,
    "legend.framealpha":     0.92,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fancybox":      True,
    "legend.borderpad":     0.5,
    "legend.handlelength":  1.8,
    "legend.handletextpad": 0.5,

    # Lines & Markers
    "lines.linewidth":      1.5,
    "lines.markersize":     5,

    # Figure
    "figure.facecolor":     "white",
    "figure.dpi":           _DPI,
    "figure.constrained_layout.use": True,

    # Saving
    "savefig.dpi":          _DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.08,
    "savefig.facecolor":    "white",
    "savefig.transparent":  False,

    # PDF / PostScript Font embedding
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

@contextlib.contextmanager
def apply_thesis_style():
    """Context manager to temporarily apply the thesis publication style."""
    with mpl.rc_context(THESIS_RC):
        yield

def create_figure(
    width: str = "single",
    aspect: float = 0.618,
    nrows: int = 1,
    ncols: int = 1,
    height_override: Optional[float] = None,
    squeeze: bool = True,
) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, np.ndarray]]:
    """Create a figure with paper-appropriate dimensions.
    
    Parameters
    ----------
    width : {"single", "double"}
        Column width preset.
    aspect : float
        Height/width ratio (default: golden ratio 0.618).
    nrows, ncols : int
        Subplot grid dimensions.
    height_override : float, optional
        Explicit height in inches (overrides aspect).
    squeeze : bool
        Whether to squeeze singleton dimensions.
    """
    w = _TEXT_WIDTH if width == "double" else _COL_WIDTH
    h = height_override if height_override is not None else w * aspect

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(w, h),
        squeeze=squeeze,
    )
    return fig, axes

def save_figure(
    fig: plt.Figure,
    path: Path,
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
) -> None:
    """Save a figure to the target formats and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : Path
        Base output path without extension (e.g. outputs/my_figure).
    formats : sequence of str
        Target file formats to export (default: pdf, png).
    close : bool
        If True, closes the figure to reclaim memory.
    """
    for fmt in formats:
        out_path = path.parent / f"{path.name}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format=fmt)
        logger.info("Saved figure: %s", out_path)

    if close:
        plt.close(fig)
