"""Torch data plumbing for neural baseline training.

The neural methods consume the exact same dataset-registry bundles as
the indicators (fairness contract, plan A6.4): ``(B, T, C)`` features,
per-trajectory ``seq_lengths``, and the alarm labels derived from
``bifurcation_times``/``is_positive``. Sequences are padded within a
batch to the batch maximum length; the loss mask (see ``labels.py``)
ignores padding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from csd_observer.models.neural.base.labels import build_alarm_labels


class BundleDataset(Dataset):
    """Torch dataset over registry bundles with derived alarm labels.

    Args:
        bundles: one or more registry bundles (signal and/or null);
            rows are concatenated in order.
        label_window: late-window size for :func:`build_alarm_labels`.
    """

    def __init__(self, bundles: list[dict[str, Any]], label_window: int = 60) -> None:
        if not bundles:
            raise ValueError("BundleDataset requires at least one bundle")
        feats = [np.asarray(b["features"]) for b in bundles]
        lens = [np.asarray(b["seq_lengths"], dtype=np.int64) for b in bundles]
        bifs = [np.asarray(b["bifurcation_times"], dtype=np.float64) for b in bundles]
        pos = [np.asarray(b["is_positive"], dtype=bool) for b in bundles]
        for arr in feats:
            if arr.ndim != 3:
                raise ValueError(
                    f"features must be (B, T, C), got shape {arr.shape}"
                )
        self.features = np.concatenate(feats, axis=0).astype(np.float32)
        self.seq_lengths = np.concatenate(lens, axis=0)
        self.bifurcation_times = np.concatenate(bifs, axis=0)
        self.is_positive = np.concatenate(pos, axis=0)
        self.targets = build_alarm_labels(
            {
                "features": self.features,
                "seq_lengths": self.seq_lengths,
                "bifurcation_times": self.bifurcation_times,
                "is_positive": self.is_positive,
            },
            label_window=label_window,
        )
        self.label_window = int(label_window)
        self.B = self.features.shape[0]

    def __len__(self) -> int:
        return self.B

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx])  # (T, C)
        L = torch.tensor(int(self.seq_lengths[idx]), dtype=torch.int64)
        y = torch.from_numpy(self.targets[idx])  # (T,)
        return x, L, y


def collate_bundles(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences to the batch maximum and stack.

    Returns ``(features (B, T_max, C), seq_lengths (B,), targets (B, T_max))``.
    """
    xs, lens, ys = zip(*batch, strict=True)
    T_max = max(int(L) for L in lens)
    C = xs[0].shape[-1]
    B = len(xs)
    feats = torch.zeros(B, T_max, C, dtype=torch.float32)
    targets = torch.zeros(B, T_max, dtype=torch.float32)
    lens_out = torch.zeros(B, dtype=torch.int64)
    for i, (x, L, y) in enumerate(batch):
        n = int(L)
        feats[i, :n] = x[:n]
        targets[i, :n] = y[:n]
        lens_out[i] = n
    return feats, lens_out, targets


def bundle_split_indices(
    bundles: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bundle start offsets and row counts in the concatenated set.

    Row ``i`` of the concatenated dataset belongs to bundle ``j`` when
    ``offsets[j] <= i < offsets[j] + counts[j]``.
    """
    offsets: list[int] = []
    counts: list[int] = []
    acc = 0
    for b in bundles:
        n = np.asarray(b["features"]).shape[0]
        offsets.append(acc)
        counts.append(n)
        acc += n
    return np.array(offsets, dtype=np.int64), np.array(counts, dtype=np.int64)


def build_dataloaders(
    train_bundles: list[dict[str, Any]],
    val_bundles: list[dict[str, Any]],
    *,
    batch_size: int,
    label_window: int = 60,
    num_workers: int = 0,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Training + validation loaders over concatenated bundles.

    ``num_workers=0`` by default (determinism contract, plan A7); a
    fixed generator seed makes shuffling reproducible.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    train_ds = BundleDataset(train_bundles, label_window=label_window)
    val_ds = BundleDataset(val_bundles, label_window=label_window)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_bundles,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_bundles,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def predict_loader(
    model: torch.nn.Module,
    features: np.ndarray,
    seq_lengths: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Full-array alarm probabilities ``(B, T)`` float32 (no labels).

    Padded cells beyond ``seq_lengths`` are forced to ``NaN`` in the
    output so downstream evaluation never treats them as alarms.
    """
    model = model.to(device).eval()
    feats = np.asarray(features, dtype=np.float32)
    lens = np.asarray(seq_lengths, dtype=np.int64)
    B, T = feats.shape[0], feats.shape[1]
    out = np.full((B, T), np.nan, dtype=np.float32)
    ds = BundleDataset(
        [
            {
                "features": feats,
                "seq_lengths": lens,
                "bifurcation_times": np.zeros(B, dtype=np.float64),
                "is_positive": np.zeros(B, dtype=bool),
            }
        ],
        label_window=60,
    )
    loader = DataLoader(
        ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate_bundles
    )
    row = 0
    with torch.no_grad():
        for feats_b, lens_b, _targets in loader:
            b = feats_b.shape[0]
            probs = torch.sigmoid(model(feats_b.to(device))).cpu().numpy()
            for i in range(b):
                n = int(lens_b[i])
                out[row + i, :n] = probs[i, :n]
            row += b
    return out


__all__ = [
    "BundleDataset",
    "build_dataloaders",
    "bundle_split_indices",
    "collate_bundles",
    "predict_loader",
]
