# Experiment Plan

```bash
# Install
pip install -e .

# Smoke test
python -m pytest -v --tb=short

# Benchmark (no args = all three runs)
python studies/runner/benchmark.py

# Or individually:
python studies/runner/benchmark.py default
python studies/runner/benchmark.py high_noise
python studies/runner/benchmark.py low_data

# Outputs in outputs/benchmark/<run_name>/<timestamp>/
```
python studies/runner/benchmark.py default n_seeds=1 methods=Raw-CSD,RunningVar,Lag2-CSD

python studies/runner/benchmark.py default n_seeds=1 methods=Lag2-CSD-detrended,Kalman-Lag2,Kalman-BCE

python studies/runner/benchmark.py default n_seeds=1 methods=Kalman-LSTM,Kalman-LSTM-Spec,Kalman-Lag2-Net,Kalman-ACKO