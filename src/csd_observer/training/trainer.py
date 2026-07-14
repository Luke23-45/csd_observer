from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch

from csd_observer.models.csd_observer import CSDKalmanObserver
from csd_observer.utils.losses import SpectralRadiusLoss


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TensorizedDataset:
    features: torch.Tensor
    masks: torch.Tensor
    seq_lengths: torch.Tensor
    bifurcation_times: torch.Tensor
    is_positive: torch.Tensor


def _compute_ews(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B, T, _ = features.shape
    W = min(window_size, T)
    csd = np.zeros((B, T), dtype=np.float32)
    rvar = np.zeros((B, T), dtype=np.float32)
    lag2 = np.zeros((B, T), dtype=np.float32)
    alternans = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        L = int(seq_lengths[b])
        seq = features[b, :, 0]
        for t in range(W, L):
            seg = seq[t - W : t]
            seg_c = seg - seg.mean()
            num = np.sum(seg_c[:-1] * seg_c[1:])
            denom = np.sum(seg_c ** 2) + 1e-8
            csd[b, t] = num / denom
            rvar[b, t] = np.var(seg)
            if len(seg) >= 4:
                seg_lag2_a = seg[:-2] - seg[:-2].mean()
                seg_lag2_b = seg[2:] - seg[2:].mean()
                lag2[b, t] = np.sum(seg_lag2_a * seg_lag2_b) / (np.sum(seg_lag2_a ** 2) + 1e-8)
                even = seg[::2]
                odd = seg[1::2]
                m = min(len(even), len(odd))
                if m > 0:
                    alternans[b, t] = float(np.mean(np.abs(even[:m] - odd[:m])))
        if W < L:
            csd[b, :W] = csd[b, W]
            rvar[b, :W] = rvar[b, W]
            lag2[b, :W] = lag2[b, W]
            alternans[b, :W] = alternans[b, W]
    return csd, rvar, lag2, alternans


def tensorize(dataset: Dict, device: torch.device) -> TensorizedDataset:
    features = dataset["features"]
    B, T, D = features.shape

    if dataset.get("augment_features", False):
        seq_lens_np = dataset["seq_lengths"]
        csd, rvar, lag2, alternans = _compute_ews(features, seq_lens_np, window_size=60)
        features = np.concatenate([
            features,
            csd.reshape(B, T, 1),
            rvar.reshape(B, T, 1),
            lag2.reshape(B, T, 1),
            alternans.reshape(B, T, 1),
        ], axis=-1)
        D = features.shape[-1]

    seq_lens = torch.tensor(dataset["seq_lengths"], dtype=torch.long, device=device)
    t = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1).expand(B, T, D)
    masks = (t < seq_lens.unsqueeze(1).unsqueeze(-1)).float()
    return TensorizedDataset(
        features=torch.tensor(features, dtype=torch.float32, device=device),
        masks=masks,
        seq_lengths=seq_lens,
        bifurcation_times=torch.tensor(dataset["bifurcation_times"], dtype=torch.float32, device=device),
        is_positive=torch.tensor(dataset["is_positive"], dtype=torch.bool, device=device),
    )


def _make_targets(
    bifurcation_times: torch.Tensor,
    seq_lengths: torch.Tensor,
    device: torch.device,
    max_length: int = 200,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    if sigma is None:
        sigma = 60.0
    B = bifurcation_times.shape[0]
    t = torch.arange(max_length, device=device).float().unsqueeze(0).expand(B, -1)
    tau = bifurcation_times.unsqueeze(1).float()
    dist = torch.clamp(tau - t, min=0.0)
    target = torch.exp(-(dist / sigma) ** 2)
    valid_tau = tau > 0
    target = target * valid_tau.float()
    is_before_bifurcation = (t <= tau).float()
    target = target * is_before_bifurcation
    valid_len = t < seq_lengths.unsqueeze(1).float()
    return target * valid_len.float()


def train_kalman(
    tensors: TensorizedDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    loss_type: Literal["bce", "lstm", "lstm_spec"],
    seed: int,
    config: dict,
    device: torch.device,
    val_arrays: Optional[dict] = None,
) -> CSDKalmanObserver:
    use_lstm = loss_type in ("lstm", "lstm_spec")
    use_spec = loss_type == "lstm_spec"
    n_features = tensors.features.shape[-1]

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    _seed_all(seed)

    x_train = tensors.features[train_idx]
    x_val = tensors.features[val_idx]
    m_train = tensors.masks[train_idx]
    m_val = tensors.masks[val_idx]
    lens_train = tensors.seq_lengths[train_idx]
    lens_val = tensors.seq_lengths[val_idx]
    bif_train = tensors.bifurcation_times[train_idx]
    bif_val = tensors.bifurcation_times[val_idx]

    model = CSDKalmanObserver(
        input_dim=n_features,
        latent_dim=model_cfg.get("latent_dim", 4),
        lstm_head=use_lstm,
        lstm_dim=model_cfg.get("lstm_dim", 8),
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 64)
    patience = train_cfg.get("patience", 5)
    spec_weight = train_cfg.get("spectral_radius_weight", 0.1)
    spec_threshold = train_cfg.get("spectral_threshold", 0.95)
    scheduler_eta_min = train_cfg.get("scheduler_eta_min", 1e-6)
    target_sigma = train_cfg.get("target_sigma", None)
    max_length = x_train.shape[1]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=scheduler_eta_min,
    )

    spec_loss_fn = None
    if use_spec:
        spec_loss_fn = SpectralRadiusLoss(weight=spec_weight, threshold=spec_threshold)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    use_val_auc = val_arrays is not None
    if use_val_auc:
        best_val_metric = -float("inf")
    else:
        best_val_metric = float("inf")
    stale_epochs = 0
    train_size = len(train_idx)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits, zs, A, K, C = model(x_train[batch_ids], m_train[batch_ids])

            targets = _make_targets(
                bif_train[batch_ids], lens_train[batch_ids], device,
                max_length=max_length, sigma=target_sigma,
            )
            valid_mask = (
                torch.arange(max_length, device=device).unsqueeze(0)
                < lens_train[batch_ids].unsqueeze(1)
            ).float()

            bce_per_step = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none",
            )
            loss = (bce_per_step * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

            if use_spec and spec_loss_fn is not None:
                loss = loss + spec_loss_fn(A, K, C)["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_val, zs_val, A_val, K_val, C_val = model(x_val, m_val)
            probs_val = torch.sigmoid(logits_val).cpu().numpy()

        if use_val_auc:
            from csd_observer.utils.metrics import compute_early_warning_auc as _ewa
            is_pos_val = val_arrays["is_positive"][val_idx]
            bif_t_val = val_arrays["bifurcation_times"][val_idx]
            seq_l_val = val_arrays["seq_lengths"][val_idx]
            sig_mask = is_pos_val & (bif_t_val > 0)
            null_mask = ~is_pos_val
            if sig_mask.any() and null_mask.any():
                val_metric = _ewa(
                    probs_val[sig_mask], bif_t_val[sig_mask], is_pos_val[sig_mask], seq_l_val[sig_mask],
                    probs_val[null_mask], seq_l_val[null_mask],
                )
            else:
                val_metric = float("nan")
            improved = np.isfinite(val_metric) and val_metric > best_val_metric + 1e-4
        else:
            targets_val = _make_targets(bif_val, lens_val, device, max_length=max_length, sigma=target_sigma)
            valid_mask_val = (
                torch.arange(max_length, device=device).unsqueeze(0)
                < lens_val.unsqueeze(1)
            ).float()
            bce_per_step_val = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_val, targets_val, reduction="none",
            )
            val_metric = (bce_per_step_val * valid_mask_val).sum() / valid_mask_val.sum().clamp(min=1.0)
            if use_spec and spec_loss_fn is not None:
                val_metric = val_metric + spec_loss_fn(A_val, K_val, C_val)["loss"]
            val_metric = val_metric.item()
            improved = val_metric < best_val_metric - 1e-6

        if improved:
            best_val_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_probs(
    model: CSDKalmanObserver,
    tensors: TensorizedDataset,
    indices: np.ndarray,
) -> np.ndarray:
    x = tensors.features[indices]
    m = tensors.masks[indices]
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = model(x, m)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs
