"""Dominant-eigenvalue (DMD) critical-slowing-down baseline.

Definition (benchmark plan §3, row DMD): the score is the
**largest-modulus eigenvalue** ``max_j |lambda_j|`` of the linear
operator ``A`` estimated by dynamic mode decomposition (DMD) on the
within-window linearly detrended causal window. The window is
time-delay (Hankel) embedded with ``m = 6`` delay coordinates — the
standard adaptation of the spatial DMD method to a univariate series
(plan §3 / kalman plan §3.1, documented adaptation) — and ``A`` is
fitted by a rank-``r`` truncated SVD: with snapshot matrices
``X1 = [x_1 ... x_{n-1}]`` and ``X2 = [x_2 ... x_n]`` (``m x (n-1)``
each, overlapping by one step) and truncated SVD
``X1 = U_r S_r V_r^T``, the projected operator is
``A_tilde = U_r^T X2 V_r S_r^{-1}`` (equivalently ``A = X2 X1^+``,
plan §3: "rank-r SVD <= m-1"); the alarm score is the dominant
eigenvalue magnitude, which rises toward 1 as critical slowing down
increases — the leading eigenvalue of the *local Jacobian* of the
window dynamics, the same direction as AC1/RETRATE (plan §3:
"approaching 1").

Protocol (Donovan & Brand 2022, *Physica A* 596:127152, DOI
10.1016/j.physa.2022.127152): for a sliding window of ``n``
observations of the state vector, ``X`` and ``X'`` are the
``m x (n-1)`` snapshot matrices shifted by one time step, the SVD
``X = U S V^T`` gives ``S_tilde = U^T X' V S^{-1}``, and the EWS is
the leading eigenvalue modulus ``|lambda_max|`` (§2.2–§2.3; sliding
window of 10 observations in the paper's figures). The 2024
corrigendum (*Physica A* 639:129694, DOI 10.1016/j.physa.2024.129694)
corrects a typographical sign error in Eq. (1) of the paper's
lung-ventilation *model* only; the DMD method is unaffected (the
protocol above stands). The time-delay embedding with ``m = 6`` and
the linear detrending are our documented adaptations for univariate
series (plan §3 global convention). Grziwotz et al. (2023) provide
the same dynamic-eigenvalue idea via local linear Jacobians (cited in
the plan's DMD row).

Rank choice: the default ``rank = 2`` is deliberate. For a
short-window *univariate* delay embedding (``m = 6``, window 30) the
full-rank operator is noise-dominated — the 6 noisy modes put
``max_j |lambda_j|`` near the unit disk for *any* process, collapsing
the signal (verified: full-rank/``rank >= 3`` means over white noise
~0.85-0.88 vs ~0.93 at ``phi = 0.9``, non-monotone in ``phi``; the
plan's "rank-r SVD <= m-1" latitude exists for exactly this). Rank 2
is the minimum that represents a complex-conjugate (oscillatory) mode
pair: it reproduces ``|lambda| ~ 1`` for a pure sinusoid, is
monotone-increasing in the AR(1) coefficient, and rises with the pole
modulus of an AR(2) (Hopf-type) window (all verified). Rank 1
collapses oscillatory dynamics and is exposed only via the
``rank`` parameter.

Direction convention: raw score = dominant eigenvalue modulus,
higher score => higher alarm (rises as the system slows). The
indicator is deterministic and has no trainable parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.utils.metrics import _linear_detrend

_EMBEDDING_DIM = 6
_SV_FLOOR = 1e-6  # relative singular-value cutoff: drops noise-floor modes


def raw_dmd_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
    embedding_dim: int = 6,
    rank: int | None = 2,
) -> np.ndarray:
    """Dominant DMD eigenvalue magnitude of causal linearly detrended windows.

    For each trajectory ``b`` and step ``t`` the score is
    ``max_j |lambda_j|`` of the DMD operator estimated on the causal
    window ``x[max(0, t-W+1) : t+1]`` after within-window linear
    detrending (``_linear_detrend``, ``metrics.py``): the window is
    time-delay embedded with ``embedding_dim`` delay coordinates, the
    snapshot pair ``X1``/``X2`` is formed (overlap one step), and the
    projected operator ``U_r^T X2 V_r S_r^{-1}`` is built from the
    truncated SVD ``X1 = U_r S_r V_r^T`` with rank
    ``min(rank, embedding_dim, w - embedding_dim)`` (default
    ``rank = 2``, see the module docstring) and singular values below
    ``1e-6 * s[0]`` dropped. The window ends at ``t`` and has
    effective size ``min(W, t+1)``, i.e. every step gets a score once
    enough data has accumulated (plan §4 — same causal convention as
    ``running_mean_center``).

    Scores are undefined (``NaN``) when fewer than
    ``embedding_dim + 1`` points are available in the causal window
    (the minimum that yields at least one ``X1``/``X2`` snapshot
    pair), when the detrended window is degenerate (zero variance —
    constant or perfectly linear windows), when all singular values
    are below the relative cutoff, or when ``t >= seq_lengths[b]``
    (beyond the valid prefix).

    For multi-channel inputs the channel-wise scores are combined by
    taking the maximum over the *finite* channel scores at each step.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        seq_lengths: ``(B,)`` integer array of valid prefix lengths.
        window_size: causal window size (default 30).
        embedding_dim: delay-embedding dimension ``m`` (default 6;
            the plan fixes ``m`` at 6, with validation-tuning over
            ``{4, 6, 8}`` left as an open question).
        rank: SVD truncation rank (default 2 — the minimum that
            captures a conjugate oscillatory mode pair; ``None``
            selects full rank ``min(m, w-m)``, which is
            noise-dominated on short univariate windows).

    Returns:
        ``(B, T)`` float32 array of alarm scores (``NaN`` where
        undefined).
    """
    features = np.asarray(features)
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    if features.ndim != 3:
        raise ValueError(f"features must be (B, T, C), got shape {features.shape}")
    if seq_lengths.shape[0] != features.shape[0]:
        raise ValueError(
            f"seq_lengths length {seq_lengths.shape[0]} != B {features.shape[0]}"
        )
    B, T, C = features.shape
    W = max(1, min(int(window_size), T))
    m = max(1, int(embedding_dim))
    scores = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        L = min(int(seq_lengths[b]), T)
        if L <= 0:
            continue
        for c in range(C):
            seq = np.asarray(features[b, :, c], dtype=np.float64)
            for t in range(L):
                seg = seq[max(0, t - W + 1) : t + 1]
                if len(seg) < m + 1:
                    continue
                s = _window_dmd(seg, m, rank)
                if not np.isfinite(s):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = s
                else:
                    scores[b, t] = max(float(scores[b, t]), s)
    return scores


def _window_dmd(seg: np.ndarray, m: int, rank: int | None) -> float:
    """``max_j |lambda_j|`` of the DMD operator of the detrended window.

    The linear detrend is applied inside the window first (plan §3).
    The delay-embedded snapshots ``x_k = seg[k : k+m]`` form the
    ``m x (n-1)`` pair ``X1`` (snapshots ``0 .. n-2``) and ``X2``
    (snapshots ``1 .. n-1``), ``n = len(seg) - m + 1``. With the
    truncated SVD ``X1 = U_r S_r V_r^T`` (rank capped at
    ``min(rank, m, n-1)``; singular values below ``1e-6 * s[0]``
    dropped) the projected operator is ``A = U_r^T X2 V_r S_r^{-1}``
    (Donovan & Brand 2022 §2.2); the score is the largest eigenvalue
    modulus. Returns ``NaN`` for a degenerate (zero-variance) window
    or an empty mode set.
    """
    seg = _linear_detrend(seg)
    if float(np.dot(seg, seg)) <= 1e-12:
        return float("nan")
    n_snap = len(seg) - m + 1
    snap = np.stack([seg[k : k + m] for k in range(n_snap)])  # (n_snap, m)
    x1 = snap[:-1].T  # (m, n_snap-1)
    x2 = snap[1:].T  # (m, n_snap-1)
    u, s, vt = np.linalg.svd(x1, full_matrices=False)
    keep = s > _SV_FLOOR * s[0]
    r = int(np.count_nonzero(keep))
    if r == 0:
        return float("nan")
    if rank is not None:
        r = min(r, max(1, int(rank)))
    u, s, vt = u[:, :r], s[:r], vt[:r]
    a_proj = (u.T @ x2 @ vt.T) / s[None, :]
    eigvals = np.linalg.eigvals(a_proj)
    if not np.all(np.isfinite(eigvals)):
        return float("nan")
    return float(np.max(np.abs(eigvals)))


__all__ = ["raw_dmd_indicator"]
