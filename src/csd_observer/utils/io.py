from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        return path

    def write_metrics(self, metrics: Dict[str, Any]) -> Path:
        path = self.root / "metrics" / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def write_result_row(self, row: Dict[str, Any]) -> None:
        path = self.root / "results" / "results.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def write_epoch_log(self, system: str, method: str, seed: int, rows: List[Dict[str, Any]]) -> Path:
        dir_path = self.root / "results" / "epoch_logs"
        dir_path.mkdir(parents=True, exist_ok=True)
        safe_method = method.lower().replace("-", "_").replace(" ", "_")
        path = dir_path / f"{system}_{safe_method}_seed{seed}.csv"
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        else:
            path.write_text("")
        return path

    def write_trajectory_data(self, system: str, method: str, seed: int, **arrays: np.ndarray) -> Path:
        dir_path = self.root / "results" / "trajectories"
        dir_path.mkdir(parents=True, exist_ok=True)
        safe_method = method.lower().replace("-", "_").replace(" ", "_")
        path = dir_path / f"{system}_{safe_method}_seed{seed}.npz"
        np.savez_compressed(path, **arrays)
        return path

    @property
    def path(self) -> Path:
        return self.root
