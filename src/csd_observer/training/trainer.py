from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

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


def tensorize(dataset: Dict, device: torch.device) -> TensorizedDataset:
    B, T, D = dataset["features"].shape
    seq_lens = torch.tensor(dataset["seq_lengths"], dtype=torch.long, device=device)
    t = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1).expand(B, T, D)
    masks = (t < seq_lens.unsqueeze(1).unsqueeze(-1)).float()
    return TensorizedDataset(
        features=torch.tensor(dataset["features"], dtype=torch.float32, device=device),
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
    sigma: float = 60.0,
) -> torch.Tensor:
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
    ).to(device)

    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 64)
    patience = train_cfg.get("patience", 5)
    spec_weight = train_cfg.get("spectral_radius_weight", 0.1)
    spec_threshold = train_cfg.get("spectral_threshold", 0.95)
    scheduler_eta_min = train_cfg.get("scheduler_eta_min", 1e-6)
    max_length = x_train.shape[1]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=scheduler_eta_min,
    )

    spec_loss_fn = None
    if use_spec:
        spec_loss_fn = SpectralRadiusLoss(weight=spec_weight, threshold=spec_threshold)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
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
                max_length=max_length,
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
            targets_val = _make_targets(bif_val, lens_val, device, max_length=max_length)
            valid_mask_val = (
                torch.arange(max_length, device=device).unsqueeze(0)
                < lens_val.unsqueeze(1)
            ).float()

            bce_per_step_val = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_val, targets_val, reduction="none",
            )
            val_loss = (bce_per_step_val * valid_mask_val).sum() / valid_mask_val.sum().clamp(min=1.0)

            if use_spec and spec_loss_fn is not None:
                val_loss = val_loss + spec_loss_fn(A_val, K_val, C_val)["loss"]

            val_loss = val_loss.item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
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
