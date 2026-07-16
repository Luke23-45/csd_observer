"""
LaTeX table generation utilities for the Krishi YOLO analysis framework.

Generates publication-ready booktabs tables with:
- Standard decimal formatting.
- Standalone or raw tabular inputs.
- Text character escaping for special characters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

def format_value(
    value: Union[float, int],
    *,
    precision: int = 4,
    use_math: bool = False,
) -> str:
    """Format a single numeric value for LaTeX.

    Parameters
    ----------
    value : float or int
        The number to format.
    precision : int
        Decimal places to retain.
    use_math : bool
        If True, wrap the returned string in LaTeX math mode ($...$).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    
    if isinstance(value, int):
        s = f"{value}"
    else:
        fmt = f"{{:.{precision}f}}"
        s = fmt.format(value)
        
    return f"${s}$" if use_math else s

def latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    if "\\" in text or "$" in text:
        return text  # assume already LaTeX formatted
    
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text

def save_latex_table(
    tex_str: str,
    path: Union[str, Path],
    *,
    standalone_header: bool = False,
) -> Path:
    """Save a LaTeX table to a .tex file.

    Parameters
    ----------
    tex_str : str
        LaTeX body text.
    path : str or Path
        Target file path.
    standalone_header : bool
        If True, wraps the table in a compilable LaTeX document template.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = tex_str
    if standalone_header:
        preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{margin=1in}\n"
            "\\begin{document}\n\n"
        )
        postamble = "\n\n\\end{document}\n"
        content = preamble + content + postamble

    path.write_text(content, encoding="utf-8")
    logger.info("Saved LaTeX table: %s", path)
    return path.resolve()
