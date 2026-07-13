"""Ablation studies for Kalman-LSTM.

Sweeps over hyperparameters:
    - spectral_radius_weight
    - latent_dim
    - lstm_dim
    - learning_rate
    - sigma (target width)

Usage:
    python studies/runner/ablation.py <run_name> [--sweep <key>=<val1>,<val2>,...]

Examples:
    python studies/runner/ablation.py default --sweep training.lr=0.0001,0.001,0.01

Output:
    outputs/ablation/<timestamp>/
        configs/resolved.yaml
        results/results.jsonl
        metrics/metrics.json
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    print(f"Ablation runner — run_name={sys.argv[1]}")
    print("Not yet implemented. Coming in next iteration.")


if __name__ == "__main__":
    main()
