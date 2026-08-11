"""Diagnostics for the Kalman-Spectral-Drift observer.

Usage:
    python studies/runner/diagnose_spectral_drift.py

Reproduces the exact benchmark data pipeline (same seeds, same split
indices) on the fast ``patients_100`` configuration and measures, per
system, the model-vs-data fit of the spectral-drift observer:

1. empirical rolling lag-1 AR coefficient and rolling variance of the
   centred mode (signal vs null) -- the "true" indicator paths;
2. per-step innovation variance of the data vs the variance implied by
   the model at a grid of gaps c (likelihood-based model fit: which c
   does the data actually support?);
3. observer outputs (c_hat, collapse probability) vs those data facts;
4. EW-AUC as a function of the warning window (where the discrimination
   actually lives);
5. operating curve: DT/FPR at several thresholds vs the fixed 5% rule;
6. counterfactual ablations: delta, c_init, sigma_u, R -- does a
   better-matched prior/scale fix the performance?

Writes a markdown report to ``docs/notes/spectral_drift_diagnosis.md``
and prints a condensed version to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for p in (str(_SRC), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from csd_observer.datasets.synthetic.common.generators import build_dataset  # noqa: E402
from csd_observer.evaluation.common.metrics import (  # noqa: E402
    W_LABEL,
    compute_detection_time,
    compute_early_warning_auc,
    compute_false_positive_rate,
)
from csd_observer.models.spectral_drift import (  # noqa: E402
    SpectralDriftObserver,
    extract_mode,
    grid_search_q_drift,
    running_mean_center,
)


def _youden_threshold(probs: np.ndarray, bifs: np.ndarray,
                      is_pos: np.ndarray, seq_lens: np.ndarray) -> float:
    """Youden's-J threshold over labelled steps (study-local; legacy
    ``utils.metrics.select_threshold`` was removed with the legacy
    package at L7.4)."""
    from sklearn.metrics import roc_curve

    positives = is_pos & (bifs > 0)
    if not positives.any():
        return 0.5
    scores: list[float] = []
    labels: list[int] = []
    for i in np.where(positives)[0]:
        tau = int(bifs[i])
        T = int(seq_lens[i])
        window = max(0, int(tau - W_LABEL))
        for t in range(window, min(T, tau)):
            scores.append(float(probs[i, t]))
            labels.append(1)
        far_window = max(0, int(tau - 2 * W_LABEL))
        for t in range(0, max(far_window - 1, 0)):
            scores.append(float(probs[i, t]))
            labels.append(0)
    if len(set(labels)) < 2 or np.std(np.array(scores)) < 1e-6:
        return 0.5
    _fpr, tpr, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(tpr - _fpr))
    return float(thresholds[best])

OBS_NOISE = {"fold": 0.10, "hopf": 0.15, "logistic": 0.05}
N_PARTICLES = 300
Q_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
C_GRID = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
                   0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
WINDOW = 30


# --------------------------------------------------------------------- #
# empirical indicator paths
# --------------------------------------------------------------------- #
def rolling_ar1(seq: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Causal rolling lag-1 AR coefficient of each (B, T) row."""
    B, T = seq.shape
    out = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        for t in range(window, T):
            seg = seq[b, t - window : t]
            seg_c = seg - seg.mean()
            num = np.sum(seg_c[:-1] * seg_c[1:])
            denom = np.sum(seg_c ** 2) + 1e-8
            out[b, t] = num / denom
    return out


def rolling_var(seq: np.ndarray, window: int = WINDOW) -> np.ndarray:
    B, T = seq.shape
    out = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        for t in range(window, T):
            out[b, t] = float(np.var(seq[b, t - window : t]))
    return out


# --------------------------------------------------------------------- #
# model-fit diagnostics (static-c likelihood)
# --------------------------------------------------------------------- #
def static_c_nll(y: np.ndarray, c_grid: np.ndarray, *, sigma_u: float,
                 r_var: float, dt: float = 1.0) -> tuple[np.ndarray, float]:
    """Per-step NLL of the data under the model with FIXED gap c.

    Runs a scalar Kalman filter over ``u`` for each candidate c and
    returns (nll[c], c_argmin). ``y`` is (B, T) centred mode.
    """
    nll = np.zeros(len(c_grid))
    for ci, c in enumerate(c_grid):
        phi = np.exp(-c * dt)
        q_c = sigma_u ** 2 / (2.0 * c) * (1.0 - phi ** 2)
        loglike = 0.0
        for b in range(y.shape[0]):
            m = 0.0
            p = sigma_u ** 2 / (2.0 * c)
            for t in range(y.shape[1]):
                m = phi * m
                p = phi * phi * p + q_c
                s = p + r_var
                e = y[b, t] - m
                loglike += -0.5 * (e * e / s + np.log(s) + np.log(2 * np.pi))
                k = p / s
                m = m + k * e
                p = (1.0 - k) * p
        nll[ci] = -loglike / (y.shape[0] * y.shape[1])
    return nll, float(c_grid[int(np.argmin(nll))])


# --------------------------------------------------------------------- #
# metric helpers (mirror the benchmark definitions)
# --------------------------------------------------------------------- #
def ew_auc_window(probs_sig: np.ndarray, bifs: np.ndarray, lens_sig: np.ndarray,
                  probs_null: np.ndarray, lens_null: np.ndarray,
                  *, w: float) -> float:
    """EW-AUC using max over [tau - w, tau - 5] (benchmark definition)."""
    scores: list[float] = []
    labels: list[int] = []
    for i in range(len(probs_sig)):
        tau = bifs[i]
        t_start = max(0, int(tau - w))
        t_end = max(0, int(tau - 5.0))
        window = probs_sig[i, t_start:t_end]
        if len(window) > 0:
            scores.append(float(np.max(window)))
            labels.append(1)
    for i in range(len(probs_null)):
        T = int(lens_null[i])
        t_start = max(0, int(T - w))
        t_end = max(0, int(T - 5.0))
        window = probs_null[i, t_start:t_end]
        if len(window) > 0:
            scores.append(float(np.max(window)))
            labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auc_of_scores(sig_score: np.ndarray, null_score: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(sig_score)), np.zeros(len(null_score))])
    scores = np.concatenate([sig_score, null_score])
    return float(roc_auc_score(labels, scores))


def _tracking_corr(c_hat: np.ndarray, ar1: np.ndarray) -> float:
    """Median per-patient correlation between c_hat and rolling AR1."""
    corrs: list[float] = []
    for i in range(c_hat.shape[0]):
        mask = ~np.isnan(ar1[i])
        if mask.sum() < 50:
            continue
        corrs.append(float(np.corrcoef(c_hat[i, mask], ar1[i, mask])[0, 1]))
    return float(np.nanmedian(corrs)) if corrs else float("nan")


# --------------------------------------------------------------------- #
# per-system diagnostics
# --------------------------------------------------------------------- #
def diagnose_system(system: str, n_patients: int, device: torch.device,
                    report: list[str]) -> None:
    sep = "=" * 74
    report.append(f"\n{sep}\n## SYSTEM: {system}\n{sep}\n")

    arrays_signal = build_dataset(
        system, n_trajectories=n_patients, max_length=200,
        noise_scale=0.15, obs_noise_scale=None, seed=0 + 101, null=False,
    )
    arrays_null = build_dataset(
        system, n_trajectories=n_patients, max_length=200,
        noise_scale=0.15, obs_noise_scale=None, seed=0 + 202, null=True,
    )
    val_s = arrays_signal["split_indices"]["val"]
    test_s = arrays_signal["split_indices"]["test"]
    val_n = arrays_null["split_indices"]["val"]
    test_n = arrays_null["split_indices"]["test"]
    bifs = arrays_signal["bifurcation_times"]
    lens_s = arrays_signal["seq_lengths"]
    lens_n = arrays_null["seq_lengths"]
    tau = float(np.median(bifs[test_s]))

    mode_sig = running_mean_center(
        extract_mode(arrays_signal["features"], system), lens_s, 50)
    mode_null = running_mean_center(
        extract_mode(arrays_null["features"], system), lens_n, 50)

    sigma_u = 0.15
    r_var = OBS_NOISE[system] ** 2

    # --- 1. empirical indicator paths ---------------------------------
    ar_sig = rolling_ar1(mode_sig)
    ar_null = rolling_ar1(mode_null)
    var_sig = rolling_var(mode_sig)
    var_null = rolling_var(mode_null)
    mean_ar_sig = np.nanmean(ar_sig[test_s], axis=0)
    mean_ar_null = np.nanmean(ar_null[test_n], axis=0)
    mean_var_sig = np.nanmean(var_sig[test_s], axis=0)
    mean_var_null = np.nanmean(var_null[test_n], axis=0)
    report.append("### 1. Empirical dynamics of the centred mode (test set)")
    report.append("| t | AR(sig) | AR(null) | Var(sig) | Var(null) |")
    report.append("|---|---------|----------|----------|-----------|")
    for t in [35, 50, 66, 83, 100, 116, 128, 133, 150, 180]:
        report.append(
            f"| {t} | {mean_ar_sig[t]:.3f} | {mean_ar_null[t]:.3f} | "
            f"{mean_var_sig[t]:.4f} | {mean_var_null[t]:.4f} |"
        )
    report.append(
        f"  (tau={tau:.1f}; null AR stays {np.nanmedian(mean_ar_null[60:]):.3f}, "
        f"null Var stays {np.nanmedian(mean_var_null[60:]):.4f})"
    )

    # --- 2. static-c model fit (which c does the data support?) --------
    report.append("\n### 2. Static-gap model fit (per-step NLL, lower is better)")
    sig_early = mode_sig[test_s][:, : int(tau * 0.6)]
    sig_late = mode_sig[test_s][:, int(tau * 0.8) : int(tau) - 5]
    nll_null, c_null = static_c_nll(mode_null[test_n], C_GRID, sigma_u=sigma_u, r_var=r_var)
    nll_sig_early, c_early = static_c_nll(sig_early, C_GRID, sigma_u=sigma_u, r_var=r_var)
    nll_sig_late, c_late = static_c_nll(sig_late, C_GRID, sigma_u=sigma_u, r_var=r_var)
    report.append("| c | NLL(null) | NLL(sig early) | NLL(sig late) |")
    report.append("|---|-----------|----------------|---------------|")
    for ci, c in enumerate(C_GRID):
        report.append(
            f"| {c:g} | {nll_null[ci]:.3f} | {nll_sig_early[ci]:.3f} | {nll_sig_late[ci]:.3f} |"
        )
    report.append(
        f"  argmin c: null={c_null:g}, sig-early={c_early:g}, sig-late={c_late:g}"
    )
    obs_var_emp = np.mean(np.var(mode_null[test_n], axis=1))
    report.append(
        f"  empirical per-trajectory variance (null) = {obs_var_emp:.4f}; "
        f"model stationary Var(u|{c_null:g}) = "
        f"{sigma_u**2/(2*c_null):.4f} vs R={r_var:.4f}"
    )

    # --- 3. observer outputs -------------------------------------------
    report.append("\n### 3. Observer on validation split (grid-search Q_drift)")
    best_q = grid_search_q_drift(
        mode_sig[val_s], mode_null[val_n], bifs[val_s], lens_s[val_s],
        lens_n[val_n], sigma_u=sigma_u, r=r_var, q_grid=Q_GRID,
        n_particles=N_PARTICLES, c_min=1e-3, delta=0.05, device=device,
    )
    report.append(f"  best Q_drift = {best_q:g}")

    obs = SpectralDriftObserver(
        sigma_u=sigma_u, r=r_var, q_drift=best_q, n_particles=N_PARTICLES,
        c_min=1e-3, c_init=0.1, c_init_std=0.05, delta=0.05,
    ).to(device)
    obs.eval()

    def run(y: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
        with torch.no_grad():
            out = obs(x)
        return {k: v.cpu().numpy() for k, v in out.items()}

    out_test = run(mode_sig[test_s])
    out_null = run(mode_null[test_n])
    probs_test = out_test["collapse_prob"]
    probs_null = out_null["collapse_prob"]
    c_hat_test = out_test["c_hat"]
    c_hat_null = out_null["c_hat"]

    report.append(
        "  collapse_prob at t=0 (prior floor): "
        f"signal={np.mean(probs_test[:, 0]):.3f}, null={np.mean(probs_null[:, 0]):.3f}"
    )
    report.append(
        f"  c_hat path: null [t35,t80,t120,t180] = "
        f"[{np.median(c_hat_null[:, 35]):.3f}, {np.median(c_hat_null[:, 80]):.3f}, "
        f"{np.median(c_hat_null[:, 120]):.3f}, {np.median(c_hat_null[:, 180]):.3f}]"
    )
    report.append(
        f"  c_hat path: sig  [t35,t80,t120,t180] = "
        f"[{np.median(c_hat_test[:, 35]):.3f}, {np.median(c_hat_test[:, 80]):.3f}, "
        f"{np.median(c_hat_test[:, 120]):.3f}, {np.median(c_hat_test[:, 180]):.3f}]"
    )
    report.append(
        f"  correlation(c_hat_null, rolling AR1(null)) = "
        f"{_tracking_corr(c_hat_null, ar_null):.3f}"
    )

    # null collapse-step percentiles (the calibration distribution)
    null_steps = np.concatenate([probs_null[i, : int(lens_n[test_n[i]])] for i in range(len(test_n))])
    report.append(
        f"  null collapse-step percentiles: "
        f"p50={np.percentile(null_steps, 50):.3f}, p90={np.percentile(null_steps, 90):.3f}, "
        f"p95={np.percentile(null_steps, 95):.3f}, p99={np.percentile(null_steps, 99):.3f}"
    )

    # --- 4. AUC vs warning-window width ---------------------------------
    report.append("\n### 4. EW-AUC vs warning window [tau - w, tau - 5]")
    report.append("| w | AUC |")
    report.append("|---|-----|")
    for w in [15.0, 25.0, 40.0, 60.0, 90.0]:
        a = ew_auc_window(probs_test, bifs[test_s], lens_s[test_s],
                          probs_null, lens_n[test_n], w=w)
        report.append(f"| {w:.0f} | {a:.3f} |")

    # --- 5. operating curve (DT vs FPR vs threshold) ---------------------
    report.append("\n### 5. Operating curve on test split")
    report.append("| threshold | DT | FPR |")
    report.append("|-----------|----|-----|")
    for p in [0.80, 0.90, 0.95, 0.99]:
        th = float(np.percentile(null_steps, 100.0 * p))
        dt = compute_detection_time(probs_test, bifs[test_s],
                                    arrays_signal["is_positive"][test_s],
                                    lens_s[test_s], th)
        fpr = compute_false_positive_rate(probs_null, lens_n[test_n], th)
        report.append(f"| p{p*100:.0f} = {th:.3f} | {dt:.1f} | {fpr:.4f} |")
    th_youden = _youden_threshold(
        probs_test, bifs[test_s], arrays_signal["is_positive"][test_s],
        lens_s[test_s],
    )
    dt_y = compute_detection_time(probs_test, bifs[test_s],
                                  arrays_signal["is_positive"][test_s],
                                  lens_s[test_s], th_youden)
    fpr_y = compute_false_positive_rate(probs_null, lens_n[test_n], th_youden)
    report.append(f"| Youden = {th_youden:.3f} | {dt_y:.1f} | {fpr_y:.4f} |")

    # --- 6. ablations ----------------------------------------------------
    report.append("\n### 6. Ablations (fixed-FPR threshold at p95, test split)")
    report.append("| variant | threshold | DT | AUC | FPR |")
    report.append("|---------|-----------|----|-----|-----|")

    def eval_variant(variant: dict[str, float], label: str) -> None:
        o = SpectralDriftObserver(
            sigma_u=variant.get("sigma_u", sigma_u),
            r=variant.get("r", r_var),
            q_drift=best_q,
            n_particles=N_PARTICLES,
            c_min=1e-3,
            c_init=variant.get("c_init", 0.1),
            c_init_std=variant.get("c_init_std", 0.05),
            delta=variant.get("delta", 0.05),
        ).to(device)
        o.eval()
        def r_obs(y: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
            with torch.no_grad():
                return o(x)["collapse_prob"].cpu().numpy()
        p_test = r_obs(mode_sig[test_s])
        p_null = r_obs(mode_null[test_n])
        ns = np.concatenate([p_null[i, : int(lens_n[test_n[i]])] for i in range(len(test_n))])
        th = float(np.percentile(ns, 95.0))
        dt = compute_detection_time(p_test, bifs[test_s],
                                    arrays_signal["is_positive"][test_s],
                                    lens_s[test_s], th)
        auc = compute_early_warning_auc(
            p_test, bifs[test_s], arrays_signal["is_positive"][test_s],
            lens_s[test_s], p_null, lens_n[test_n])
        fpr = compute_false_positive_rate(p_null, lens_n[test_n], th)
        report.append(f"| {label} | {th:.3f} | {dt:.1f} | {auc:.3f} | {fpr:.4f} |")

    eval_variant({}, "default")
    for d in [0.15, 0.3, 0.5]:
        eval_variant({"delta": d}, f"delta={d}")
    for ci in [0.3, 1.0]:
        eval_variant({"c_init": ci, "c_init_std": 0.3}, f"c_init={ci},std=0.3")
    for su in [0.5, 1.5]:
        eval_variant({"sigma_u": su}, f"sigma_u={su}")
    eval_variant({"r": obs_var_emp}, f"R=empirical var ({obs_var_emp:.4f})")

    # --- 7. simple baseline: rolling AR1 and variance --------------------
    report.append("\n### 7. Cheap indicators (same EW-AUC metric)")
    ar_auc = auc_of_scores(
        np.nanmax(ar_sig[test_s][:, int(tau - 50) : int(tau - 5)], axis=1),
        np.nanmax(ar_null[test_n][:, 150:195], axis=1),
    )
    var_auc = auc_of_scores(
        np.nanmax(var_sig[test_s][:, int(tau - 50) : int(tau - 5)], axis=1),
        np.nanmax(var_null[test_n][:, 150:195], axis=1),
    )
    report.append(f"| rolling AR1 | {ar_auc:.3f} |")
    report.append(f"| rolling Var | {var_auc:.3f} |")


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n_patients = 100
    report: list[str] = ["# Spectral-Drift Diagnostics (patients_100)\n"]
    for system in ["fold", "hopf", "logistic"]:
        diagnose_system(system, n_patients, device, report)

    out_path = _ROOT / "docs" / "notes" / "spectral_drift_diagnosis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
