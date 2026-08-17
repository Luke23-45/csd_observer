"""One-method training entry point for learned baselines (R0.3).

Extracted from the legacy runner: the orchestrator delegates to
``train_method`` with plain arguments (never module imports of
``orchestration``), and ``models`` stays import-free of ``training`` —
the wiring runs through the registry's ``get_lit_module`` and the
injected ``loss_fn``.

The persistence knob ``evaluation.k_persist`` is folded into the alarm
loss at training time (``partial(alarm_bce, k_persist=...)``), so the
loss the network optimizes matches the evaluation protocol's definition
of a "persistent alarm" (plan §8: ``r_t >= k_persist``).
"""

from __future__ import annotations

from functools import partial
from typing import Any

from csd_observer.models.common.registry import get_lit_module
from csd_observer.outputs.writer import OutputWriter
from csd_observer.training.common.losses import alarm_bce
from csd_observer.training.common.trainer_factory import fit_model, seed_everything


def train_method(
    method: Any,
    bundle: dict[str, Any],
    config: dict[str, Any],
    training: dict[str, Any],
    writer: OutputWriter,
    *,
    seed_index: int,
    run_seed: int,
) -> dict[str, Any]:
    """Train one learned method on the run's seed schedule.

    Args:
        method: the method instance (``meta.is_learned`` decides whether
            anything happens).
        bundle: the dataset-registry bundle (``signal``/``null``
            subsets with ``split_indices``).
        config: full composed config (``model`` blocks, ``evaluation``).
        training: the ``training`` group block (dict form).
        writer: the run's :class:`OutputWriter` (artifacts/checkpoints).
        seed_index: 0-based seed index (ledger/logging only).
        run_seed: the per-seed schedule signal seed (applied before
            module construction so weight init is deterministic).

    Returns:
        the per-method config copy with the best-checkpoint path and
        the pinned ``in_channels`` injected into the method's model
        block (so ``method.fit`` loads the trained weights at score
        time). Indicator methods return the input config unchanged.
    """
    if not method.meta.is_learned:
        return config
    key = method.meta.config_path[0]
    model_cfg = dict(config.get("model", {}).get(key, {}) or {})
    if not training.get("enabled", True):
        if model_cfg.get("checkpoint"):
            return config
        raise ValueError(
            f"method {method.meta.name!r} is learned but training is disabled and no checkpoint is provided"
        )
    lit_cls = get_lit_module(method.meta.name)
    if lit_cls is None:
        raise ValueError(f"no Lightning module registered for method {method.meta.name!r}")

    evaluation = dict(config.get("evaluation", {}) or {})
    evaluation_name = str(evaluation.get("name", "persistenceaware"))
    k_persist = 1 if evaluation_name == "baseline_classic" else int(evaluation.get("k_persist", 5))
    if k_persist < 1:
        raise ValueError("evaluation.k_persist must be >= 1")

    seed_everything(
        int(run_seed),
        deterministic=bool(training.get("deterministic", True)),
    )

    model_cfg = dict(config.get("model", {}).get(key, {}) or {})
    # Channel count is dataset-determined: synthetic hopf emits a
    # 2-channel bundle; real datasets are single channel. ``None`` in
    # the schema means "auto" — hard-coding 1 here silently built a
    # mismatched net.
    if not model_cfg.get("in_channels"):
        model_cfg["in_channels"] = int(bundle["signal"]["features"].shape[-1])
    module = lit_cls(
        model_cfg,
        training,
        partial(alarm_bce, k_persist=k_persist),
    )

    train_bundles = _split_bundle(bundle["signal"], "train") + _split_bundle(bundle["null"], "train")
    val_bundles = _split_bundle(bundle["signal"], "val") + _split_bundle(bundle["null"], "val")
    trainer = fit_model(
        module, train_bundles, val_bundles,
        writer=writer, method=method.meta.name, seed=int(run_seed), config=config,
    )
    checkpoint = getattr(trainer, "checkpoint_callback", None)
    ckpt_path = checkpoint.best_model_path if checkpoint is not None else ""
    if not ckpt_path:
        raise RuntimeError(f"training {method.meta.name!r} produced no checkpoint")

    per_method = dict(config)
    model_blocks = {k: dict(v) if isinstance(v, dict) else v for k, v in per_method.get("model", {}).items()}
    block = dict(model_blocks.get(key, {}) or {})
    block["checkpoint"] = str(ckpt_path)
    # Keep the scoring-time fit consistent with the trained module: the
    # composed block may carry ``in_channels: null`` (auto).
    if not block.get("in_channels"):
        block["in_channels"] = int(bundle["signal"]["features"].shape[-1])
    model_blocks[key] = block
    per_method["model"] = model_blocks
    return per_method


def _split_bundle(bundle: dict[str, Any], part: str) -> list[dict[str, Any]]:
    """Slice one registry bundle to a split part (drop split_indices/meta)."""
    idx = bundle.get("split_indices", {}).get(part)
    if idx is None:
        return []
    return [
        {k: v[idx] for k, v in bundle.items() if k not in {"split_indices", "meta"}}
    ]


__all__ = ["train_method"]
