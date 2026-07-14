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