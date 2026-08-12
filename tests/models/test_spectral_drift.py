"""Tests for the spectral-drift Bayesian early warning observer.

Covers ``csd_observer.models.spectral_drift``: the scalar-mode extraction,
running-mean centring, the Rao-Blackwellised particle filter, and the
validation-based grid search over ``Q_drift``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _make_ou_signal(
    c: float,
    sigma_u: float,
    r: float,
    t: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a stationary OU mode ``du = -c u dt + sigma_u dW`` with
    observation noise ``r`` (variance)."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-c)
    var_eps = sigma_u ** 2 / (2.0 * c) * (1.0 - phi ** 2)
    u = np.zeros(t, dtype=np.float64)
    y = np.zeros(t, dtype=np.float64)
    for k in range(1, t):
        u[k] = phi * u[k - 1] + rng.normal(0.0, np.sqrt(var_eps))
        y[k] = u[k] + rng.normal(0.0, np.sqrt(r))
    return y.astype(np.float32)


def test_extract_mode_fold() -> None:
    from csd_observer.models.spectral_drift import extract_mode
    features = np.random.randn(4, 50, 1).astype(np.float32)
    mode = extract_mode(features, "fold")
    assert mode.shape == (4, 50)
    assert np.allclose(mode, features[..., 0])


def test_extract_mode_logistic() -> None:
    from csd_observer.models.spectral_drift import extract_mode
    features = np.random.randn(3, 40, 1).astype(np.float32)
    mode = extract_mode(features, "logistic")
    assert mode.shape == (3, 40)
    assert np.allclose(mode, features[..., 0])


def test_extract_mode_hopf_radius() -> None:
    from csd_observer.models.spectral_drift import extract_mode
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, (5, 60))
    radius = rng.uniform(0.5, 1.5, (5, 60))
    x1 = radius * np.cos(theta)
    x2 = radius * np.sin(theta)
    features = np.stack([x1, x2], axis=-1).astype(np.float32)
    mode = extract_mode(features, "hopf")
    assert mode.shape == (5, 60)
    assert np.allclose(mode ** 2, x1 ** 2 + x2 ** 2, atol=1e-5)


def test_extract_mode_invalid() -> None:
    import pytest

    from csd_observer.models.spectral_drift import extract_mode
    with pytest.raises(ValueError, match="Unknown system"):
        extract_mode(np.random.randn(2, 10, 1).astype(np.float32), "nope")


def test_extract_mode_hopf_single_channel_raises() -> None:
    import pytest

    from csd_observer.models.spectral_drift import extract_mode
    with pytest.raises(ValueError, match="at least 2 channels"):
        extract_mode(np.random.randn(2, 10, 1).astype(np.float32), "hopf")


def test_running_mean_center_constant() -> None:
    from csd_observer.models.spectral_drift import running_mean_center
    seq = np.ones((3, 50), dtype=np.float32) * 5.0
    lens = np.full(3, 50, dtype=np.int64)
    centered = running_mean_center(seq, lens, window=10)
    assert centered.shape == (3, 50)
    assert np.allclose(centered, 0.0, atol=1e-5)


def test_running_mean_center_removes_step() -> None:
    from csd_observer.models.spectral_drift import running_mean_center
    seq = np.zeros((1, 100), dtype=np.float32)
    seq[0, 50:] = 10.0
    lens = np.full(1, 100, dtype=np.int64)
    centered = running_mean_center(seq, lens, window=10)
    # before the step the centred series is exactly zero
    assert centered[0, 10] == 0.0
    # in the mixed window right after the step, the centred value is positive
    # (window spans both the low and high regime)
    assert centered[0, 52] > 5.0
    # once the window lies fully past the step it returns to zero
    assert np.allclose(centered[0, 75:], 0.0, atol=1e-4)


def test_running_mean_center_respects_lengths() -> None:
    from csd_observer.models.spectral_drift import running_mean_center
    seq = np.random.randn(2, 50).astype(np.float32)
    lens = np.array([50, 20], dtype=np.int64)
    centered = running_mean_center(seq, lens, window=5)
    assert np.allclose(centered[1, 20:], 0.0)
    assert not np.allclose(centered[1, :20], 0.0)


def test_observer_creation() -> None:
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    obs = SpectralDriftObserver(sigma_u=0.15, r=0.01, q_drift=1e-3)
    assert obs.count_parameters() == 0
    assert obs.n_particles == 500
    assert obs.c_min == 1e-3
    assert obs.delta == 0.05


def test_observer_creation_validation() -> None:
    import pytest

    from csd_observer.models.spectral_drift import SpectralDriftObserver
    with pytest.raises(ValueError, match="c_min"):
        SpectralDriftObserver(sigma_u=0.15, r=0.01, q_drift=1e-3, c_min=0.0)
    with pytest.raises(ValueError, match="q_drift"):
        SpectralDriftObserver(sigma_u=0.15, r=0.01, q_drift=0.0)
    with pytest.raises(ValueError, match="c_init"):
        SpectralDriftObserver(sigma_u=0.15, r=0.01, q_drift=1e-3, c_init=1e-4)


def test_observer_forward_shapes() -> None:
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    obs = SpectralDriftObserver(
        sigma_u=0.15, r=0.01, q_drift=1e-3, n_particles=200,
    )
    y = torch.randn(4, 100)
    out = obs(y)
    assert out["collapse_prob"].shape == (4, 100)
    assert out["c_hat"].shape == (4, 100)
    assert out["c_std"].shape == (4, 100)
    assert out["log_lik"].shape == (4,)


def test_observer_collapse_prob_bounds() -> None:
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    obs = SpectralDriftObserver(
        sigma_u=0.15, r=0.01, q_drift=1e-3, n_particles=200,
    )
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, (6, 150)).astype(np.float32)
    out = obs(torch.from_numpy(y))
    probs = out["collapse_prob"].numpy()
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert np.all(np.isfinite(probs))


def test_observer_c_hat_bounded_below() -> None:
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    obs = SpectralDriftObserver(
        sigma_u=0.15, r=0.01, q_drift=1e-1, n_particles=200, c_min=1e-3,
    )
    # high Q_drift drives c into the clip; c_hat must never go below c_min
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.05, (4, 200)).astype(np.float32)
    out = obs(torch.from_numpy(y))
    c_hat = out["c_hat"].numpy()
    assert c_hat.min() >= obs.c_min - 1e-4


def test_observer_tracks_slowly_decaying_gap() -> None:
    """When the true gap decays below delta, collapse probability rises."""
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    obs = SpectralDriftObserver(
        sigma_u=0.3, r=0.0025, q_drift=1e-3, n_particles=500,
        c_min=1e-3, c_init=0.5, c_init_std=0.05, delta=0.05, seed=7,
    )
    # true c: held at 0.5 for 60% of the horizon, then drops to 0.01
    T = 300
    rng = np.random.default_rng(3)
    u = np.zeros(T, dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    for k in range(1, T):
        ck = 0.5 if k < T * 0.6 else 0.01
        phi = np.exp(-ck)
        var_eps = 0.3 ** 2 / (2.0 * ck) * (1.0 - phi ** 2)
        u[k] = phi * u[k - 1] + rng.normal(0.0, np.sqrt(var_eps))
        y[k] = u[k] + rng.normal(0.0, 0.05)
    out = obs(torch.from_numpy(y.astype(np.float32).reshape(1, -1)))
    probs = out["collapse_prob"].numpy()[0]
    c_hat = out["c_hat"].numpy()[0]
    # posterior mean of c should decline as the true gap declines
    assert c_hat[-1] < 0.5 * c_hat[10]
    # collapse probability should be substantially higher at the end
    assert probs[-1] > probs[10] + 0.1


def test_observer_stationary_gap_low_collapse() -> None:
    """A stationary OU (fixed c >> delta) should keep collapse low."""
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    obs = SpectralDriftObserver(
        sigma_u=0.15, r=0.01, q_drift=1e-4, n_particles=300,
        c_min=1e-3, c_init=0.2, c_init_std=0.05, delta=0.05,
    )
    y = _make_ou_signal(c=0.2, sigma_u=0.15, r=0.01, t=300, seed=11)
    out = obs(torch.from_numpy(y.reshape(1, -1)))
    probs = out["collapse_prob"].numpy()[0]
    assert np.mean(probs[-50:]) < 0.25


def test_observer_deterministic_given_seed() -> None:
    from csd_observer.models.spectral_drift import SpectralDriftObserver
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, (2, 100)).astype(np.float32)
    y_t = torch.from_numpy(y)
    o1 = SpectralDriftObserver(sigma_u=0.15, r=0.01, q_drift=1e-3, n_particles=200, seed=42)
    o2 = SpectralDriftObserver(sigma_u=0.15, r=0.01, q_drift=1e-3, n_particles=200, seed=42)
    p1 = o1(y_t)["collapse_prob"].numpy()
    p2 = o2(y_t)["collapse_prob"].numpy()
    assert np.allclose(p1, p2)


def test_grid_search_q_drift_returns_grid_value() -> None:
    from csd_observer.models.spectral_drift import grid_search_q_drift
    B, T = 10, 100
    rng = np.random.default_rng(42)
    y_sig = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    y_null = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    bifs = np.full(B, 70.0, dtype=np.float32)
    lens = np.full(B, T, dtype=np.int64)
    grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    best_q = grid_search_q_drift(
        y_sig, y_null, bifs, lens, lens,
        sigma_u=0.15, r=0.01, q_grid=grid, n_particles=100,
    )
    assert best_q in grid


def test_grid_search_q_drift_prefers_working_q() -> None:
    """On a slowly-decaying signal vs stationary null, the grid search should
    return a q that yields finite, sensible EW-AUC (smoke-level check)."""
    from csd_observer.models.spectral_drift import grid_search_q_drift
    B, T = 8, 120
    rng = np.random.default_rng(7)
    y_sig = rng.normal(0, 0.05, (B, T)).astype(np.float32)
    # plant an amplitude ramp in the signal so its late-window variance rises
    ramp = np.linspace(1.0, 3.0, T).astype(np.float32)
    y_sig = y_sig * ramp[None, :]
    y_null = rng.normal(0, 0.05, (B, T)).astype(np.float32)
    bifs = np.full(B, T - 20.0, dtype=np.float32)
    lens = np.full(B, T, dtype=np.int64)
    best_q = grid_search_q_drift(
        y_sig, y_null, bifs, lens, lens,
        sigma_u=0.15, r=0.01, q_grid=[1e-4, 1e-3, 1e-2],
        n_particles=100,
    )
    assert best_q in [1e-4, 1e-3, 1e-2]


def test_grid_search_sigma_u_q_drift_returns_grid_values() -> None:
    """Joint (sigma_u, q) search must return members of both grids."""
    from csd_observer.models.spectral_drift import grid_search_sigma_u_q_drift
    B, T = 10, 100
    rng = np.random.default_rng(42)
    y_sig = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    y_null = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    bifs = np.full(B, 70.0, dtype=np.float32)
    lens = np.full(B, T, dtype=np.int64)
    s_grid = [0.15, 0.3, 0.6]
    q_grid = [1e-6, 1e-4, 1e-2]
    best_sigma_u, best_q = grid_search_sigma_u_q_drift(
        y_sig, y_null, bifs, lens, lens,
        sigma_u_grid=s_grid, r=0.01, q_grid=q_grid, n_particles=100,
    )
    assert best_sigma_u in s_grid
    assert best_q in q_grid


def test_grid_search_sigma_u_q_drift_single_sigma_matches_q_only() -> None:
    """With a one-element sigma grid the joint search must equal the
    q-only search (they share the same implementation)."""
    from csd_observer.models.spectral_drift import (
        grid_search_q_drift,
        grid_search_sigma_u_q_drift,
    )
    B, T = 8, 120
    rng = np.random.default_rng(7)
    y_sig = rng.normal(0, 0.05, (B, T)).astype(np.float32)
    ramp = np.linspace(1.0, 3.0, T).astype(np.float32)
    y_sig = y_sig * ramp[None, :]
    y_null = rng.normal(0, 0.05, (B, T)).astype(np.float32)
    bifs = np.full(B, T - 20.0, dtype=np.float32)
    lens = np.full(B, T, dtype=np.int64)
    q_grid = [1e-4, 1e-3, 1e-2]
    best_q = grid_search_q_drift(
        y_sig, y_null, bifs, lens, lens,
        sigma_u=0.15, r=0.01, q_grid=q_grid, n_particles=100,
    )
    best_sigma_u, best_q_joint = grid_search_sigma_u_q_drift(
        y_sig, y_null, bifs, lens, lens,
        sigma_u_grid=[0.15], r=0.01, q_grid=q_grid, n_particles=100,
    )
    assert best_sigma_u == 0.15
    assert best_q_joint == best_q


def test_grid_search_sigma_u_q_drift_rejects_bad_grids() -> None:
    """Empty or non-positive sigma grids must raise ValueError."""
    from csd_observer.models.spectral_drift import grid_search_sigma_u_q_drift
    B, T = 4, 50
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    bifs = np.full(B, 40.0, dtype=np.float32)
    lens = np.full(B, T, dtype=np.int64)
    for bad in ([], [0.0], [-0.1, 0.2], [0.15, 0.0]):
        with pytest.raises(ValueError):
            grid_search_sigma_u_q_drift(
                y, y, bifs, lens, lens,
                sigma_u_grid=bad, r=0.01, q_grid=[1e-3], n_particles=50,
            )


def test_grid_search_sigma_u_q_drift_prefers_matched_noise_scale() -> None:
    """On the real logistic data the model's per-step noise sigma_u is ~2x
    smaller than the true innovation scale: the mis-specified scale makes
    null trajectories look collapsed (variance misattributed to a small
    gap c), which depresses the validation EW-AUC. The joint search must
    pick the matched scale (0.3) over the mis-specified one (0.15)."""
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    from csd_observer.models.spectral_drift import (
        extract_mode,
        grid_search_sigma_u_q_drift,
        running_mean_center,
    )
    sig = build_dataset(
        "logistic", n_trajectories=100, max_length=200, noise_scale=0.15,
        obs_noise_scale=None, seed=101, null=False,
    )
    nul = build_dataset(
        "logistic", n_trajectories=100, max_length=200, noise_scale=0.15,
        obs_noise_scale=None, seed=202, null=True,
    )
    vs = sig["split_indices"]["val"]
    vn = nul["split_indices"]["val"]
    ms = running_mean_center(
        extract_mode(sig["features"], "logistic"), sig["seq_lengths"], 50
    )
    mn = running_mean_center(
        extract_mode(nul["features"], "logistic"), nul["seq_lengths"], 50
    )
    best_sigma_u, best_q = grid_search_sigma_u_q_drift(
        ms[vs], mn[vn],
        sig["bifurcation_times"][vs], sig["seq_lengths"][vs],
        nul["seq_lengths"][vn],
        sigma_u_grid=[0.15, 0.3], r=0.05 ** 2, q_grid=[1e-3],
        n_particles=200, c_min=1e-3, delta=0.05,
    )
    assert best_sigma_u == 0.3
    assert best_q == 1e-3


def test_spectral_method_accepts_real_systems() -> None:
    """G1: the spectral adapter must construct for the real-dataset
    bif_types (previously ``ValueError: Unknown system``)."""
    from csd_observer.models.common.systems import SUPPORTED_SYSTEMS
    from csd_observer.models.spectral_drift.method import SpectralDriftMethod

    for system in ("subcritical_hopf", "transcritical"):
        m = SpectralDriftMethod("kalman-spectral-drift", system)
        assert m._system == system
        assert m.meta.bif_types_supported == list(SUPPORTED_SYSTEMS)
    with pytest.raises(ValueError, match="Unknown system"):
        SpectralDriftMethod("kalman-spectral-drift", "nonexistent")


def test_obs_noise_defaults_cover_all_systems() -> None:
    """G2: every supported system has an observation-noise default, so
    ``fit`` can never KeyError for a plan-verified bif_type."""
    from csd_observer.models.common.systems import SUPPORTED_SYSTEMS
    from csd_observer.models.spectral_drift.method import _OBS_NOISE_DEFAULT

    assert set(_OBS_NOISE_DEFAULT) == set(SUPPORTED_SYSTEMS)
