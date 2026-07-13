from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class OutputWriter:
    def __init__(self, experiment_name: str, base_dir: str | Path = "outputs"):
        self.name = experiment_name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.root = Path(base_dir) / experiment_name / timestamp
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "configs").mkdir(exist_ok=True)
        (self.root / "metrics").mkdir(exist_ok=True)
        (self.root / "results").mkdir(exist_ok=True)

    def write_config(self, config: Dict[str, Any]) -> Path:
        path = self.root / "configs" / "resolved.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return path

    def write_metrics(self, metrics: Dict[str, Any]) -> Path:
        path = self.root / "metrics" / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def write_result_row(self, row: Dict[str, Any]) -> None:
        path = self.root / "results" / "results.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    @property
    def path(self) -> Path:
        return self.root
