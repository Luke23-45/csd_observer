"""
Data I/O utilities for the Krishi YOLO analysis framework.

Handles file loading, path resolution, and formatting safety.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file and return rows as list of dicts.
    
    Parameters
    ----------
    path : Path
        Path to a CSV file.
        
    Returns
    -------
    list of dict
        List containing each row parsed as a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_json(path: Path) -> Any:
    """Load a JSON file with safe parsing.
    
    Parameters
    ----------
    path : Path
        Path to a JSON file.
        
    Returns
    -------
    Any
        Parsed JSON data (dict, list, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file.
    
    Parameters
    ----------
    path : Path
        Path to a YAML file.
        
    Returns
    -------
    dict
        Parsed YAML dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_file(path: Path, content: str) -> None:
    """Write text content to a file, recursively creating parent directories.
    
    Parameters
    ----------
    path : Path
        Output file path.
    content : str
        Text content to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Written successfully: %s", path)
