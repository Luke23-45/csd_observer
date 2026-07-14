# CSD Observer: Adaptive Critical-Slowing-Down Observer

Replaces the per-step MLP head of a Kalman filter with a causal LSTM head to detect temporal CSD patterns (rising autocorrelation, slowing recovery) that a static per-step head cannot see.

The chick-heart benchmark is treated separately as a period-doubling / alternans problem, so it uses lag-2 and alternans-aware feature augmentation and reports a `Lag2-CSD` baseline in addition to the generic CSD metrics.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- numpy, scikit-learn, pyyaml, tqdm

Install: `pip install -e .`

## Project Structure

```
src/csd_observer/       # core package
├── config/load.py     # config loading (modular YAML merging)
├── models/            # CSDKalmanObserver
├── data/              # synthetic bifurcation generators
├── training/          # training loop, TensorizedDataset
└── utils/             # losses, metrics, OutputWriter
configs/               # YAML configs (data, model, training, run)
studies/runner/        # entry points (benchmark, ablation)
outputs/               # experiment results
```

## Usage

```bash
# Install
pip install -e .

# Default benchmark (noise=0.15, 500 patients, 30 epochs)
python studies/runner/benchmark.py default

# Stress tests
python studies/runner/benchmark.py high_noise
python studies/runner/benchmark.py low_data
```

## Verdict Criteria

A system **passes** when:
1. **DT**: Kalman-LSTM detection time <= 15 steps (DT gain >= 15 vs BCE)
2. **EW-AUC**: Early-warning AUC gain >= 0.05 vs BCE
3. **FPR**: False-positive ratio not > 1.5× BCE + 0.05

**Overall GO**: >= 2/3 systems pass.
