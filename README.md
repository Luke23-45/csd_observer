# CSD Observer: Adaptive Critical-Slowing-Down Observer

Rao-Blackwellised particle filter over the spectral gap `c_k`, producing an
early-warning alarm from the posterior collapse probability `Pr(c_{t+1} < δ | y_{0:t})`.
Non-learned (no trainable parameters, buffers only).

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- numpy, scikit-learn, pyyaml, tqdm

Install: `pip install -e .`

## Project Structure

```
src/csd_observer/       # core package
├── config/load.py     # config loading (modular YAML merging)
├── models/            # spectral_drift/ package (observer, preprocess, grid search)
├── data/              # synthetic bifurcation generators
└── utils/             # metrics, OutputWriter
configs/               # YAML configs (data, model, run)
studies/runner/        # entry points (benchmark, diagnostics)
analysis/              # results analysis (classical baseline pipeline)
outputs/               # experiment results
```

## Usage

```bash
# Install
pip install -e .

# Default benchmark (patients_100..patients_500 and high_noise)
python studies/runner/benchmark.py

# Specific configs
python studies/runner/benchmark.py patients_100

# Spectral-drift diagnostics report
python studies/runner/diagnose_spectral_drift.py
```

## Evaluation

Per (system, patient-count): early-warning AUC (max collapse probability in
`[τ-50, τ-5)` vs null terminal windows), detection time at a validation-selected
threshold (fixed null FPR 0.05), and per-step null false-positive rate.
