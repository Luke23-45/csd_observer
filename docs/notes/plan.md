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

# Final training


# 1.   100 patients, 10 seeds
python studies/runner/benchmark.py patients_100 n_seeds=10 methods="Kalman-BCE, Kalman-LSTM-Spec"

# 2.   200 patients, 10 seeds
python studies/runner/benchmark.py patients_200 n_seeds=10 methods="Kalman-BCE, Kalman-LSTM-Spec"

# 3.   300 patients, 10 seeds
python studies/runner/benchmark.py patients_300 n_seeds=10 methods="Kalman-BCE, Kalman-LSTM-Spec"

# 4.   400 patients, 10 seeds
python studies/runner/benchmark.py patients_400 n_seeds=10 methods="Kalman-BCE, Kalman-LSTM-Spec"

# 5.   500 patients, 10 seeds
python studies/runner/benchmark.py patients_500 n_seeds=10 methods="Kalman-BCE, Kalman-LSTM-Spec"