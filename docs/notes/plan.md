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


# Run the chick_heart benchmark (5 seeds × 3 methods)
python studies/runner/benchmark.py chick_heart

# Or clear cached data and re-run:
rm -rf data/chick_heart/
python studies/runner/benchmark.py chick_heart n_seeds=1


# Run only new methods (fast iteration):
python studies/runner/benchmark.py chick_heart methods=Kalman-Lag2,Kalman-Lag2-Net,Lag2-CSD-detrended

# Run a single baseline:
python studies/runner/benchmark.py chick_heart methods=Lag2-CSD-detrended

# Run all learned methods, skip non-learned:
python studies/runner/benchmark.py chick_heart methods=Kalman-BCE,Kalman-LSTM,Kalman-LSTM-Spec,Kalman-LSTM-Aux,Kalman-Lag2-Net

# Run everything (current behavior — same as omitting methods=):
python studies/runner/benchmark.py chick_heart methods=all