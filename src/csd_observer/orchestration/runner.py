"""Thin orchestration layer for one governed benchmark run (Â§11).

Runner contains pipeline order only: registry resolution â†’ per-method
governance via the MethodInterface â†’ ledger. Dataset generation, method
training and scoring are delegated; the runner never reimplements any
of them.

For learned methods the runner trains the Lightning module once per
seed (same train/val subsets the indicators calibrate on â€” fairness
contract Â§6.4) and hands the best checkpoint to the method's ``fit``,
which loads it before scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from csd_observer.config.validate import seed_schedule
from csd_observer.datasets.registry import get_dataset
from csd_observer.evaluation.persistence.governance import evaluate_method
from csd_observer.models.common.registry import get_method, list_families
from csd_observer.outputs.ledger import LedgerRow, RunLedger
from csd_observer.outputs.metadata import collect_environment, hash_config
from csd_observer.outputs.summarize import summarize_run
from csd_observer.outputs.writer import OutputWriter

_LEARNED_LIT_MODULES: dict[str, Any] = {
    "lstm": None,  # resolved lazily (torch import)
    "tcn": None,
    "patchtst": None,
}


def run_benchmark(
    config: dict[str, Any],
    *,
    cli_overrides: Sequence[str] = (),
) -> OutputWriter:
    """Run every (seed, method) pair of a composed, validated config.

    Args:
        config: full composed config dict (see ``config/store.py``):
            ``dataset`` (group block), ``models`` (registry display
            names), ``seed``/``seed_offset``/``n_seeds`` (Â§13 schedule),
            ``evaluation``, ``training``, ``output``,
            ``dataset_overrides`` (whitelisted keys only).
        cli_overrides: raw Hydra override strings, recorded verbatim.

    Returns:
        the run's :class:`OutputWriter` (output tree already finalized).
    """
    dataset_cfg = dict(config.get("dataset", {}) or {})
    dataset = str(dataset_cfg.get("name", ""))
    if not dataset:
        raise ValueError("config.dataset.name is required")
    methods = [str(m) for m in config.get("models", []) or []]
    if not methods:
        raise ValueError("config.models must list at least one method")

    evaluation = dict(config.get("evaluation", {}) or {})
    training = config.get("training", {}) or {}
    # ``training="none"`` is the CLI shortcut for the ``configs/training/none``
    # group; the validator treats the string as ``enabled=false``. The runner
    # must respect the same semantic so a string here is normalised to a
    # dict with ``enabled=False`` rather than coerced via ``dict(str)``.
    if isinstance(training, str):
        training = {"enabled": training != "none"}
    training = dict(training or {})
    output = dict(config.get("output", {}) or {})
    base_dir = str(output.get("base_dir", "outputs"))
    k_persist = int(evaluation.get("k_persist", 5))
    fpr_target = float(evaluation.get("fpr_target", 0.05))
    evaluation_name = str(evaluation.get("name", "persistenceaware"))
    store_trajectories = bool(output.get("store_trajectories", False))
    seed_offset = int(config.get("seed_offset", 0))
    n_seeds = int(config.get("n_seeds", 1))
    seed_base = int(config.get("seed", 42))
    overrides = dict(config.get("dataset_overrides", {}) or {})

    # Real datasets must be provisioned (ingest -> process -> gates ->
    # manifest) before anything else; synthetic datasets short-circuit
    # (in-memory registry fast path). Fails before any output is written.
    if str(dataset_cfg.get("source", "synthetic")) != "synthetic":
        from csd_observer.datasets.provision import provision_dataset

        provision_dataset(
            dataset,
            dataset_cfg,
            root=overrides.get("data_root", "final_data"),
        )

    writer = OutputWriter(dataset, base_dir)
    resolved = dict(config)
    resolved.setdefault("dataset_overrides", {})
    config_hash = hash_config(resolved)
    writer.write_resolved_config(resolved)
    writer.write_cli_overrides(list(cli_overrides))
    writer.write_environment(
        collect_environment(config_hash=config_hash)
    )

    ledger = RunLedger(Path(base_dir) / "_ledger")
    run_id = f"{dataset}-{writer.timestamp}"
    # ``meta`` is captured from the first pre-loop bundle load. Real
    # datasets pin ``bif_type`` in their manifest; synthetic generators
    # emit ``bif_type`` from the constructor. ``meta`` is therefore
    # stable across seeds and is not re-read inside the loop. We pass
    # the user ``overrides`` here for full reproducibility of any
    # dataset-side knobs (e.g. generator noise scale); the per-seed
    # schedule (``seed``, ``null_seed``) is added inside the loop.
    bundle = _load_bundle(dataset, overrides, seed_base)
    meta = bundle["meta"]
    families = list_families()
    learned = [m for m in methods if families.get(m) == "neural"]
    ledger.append(
        LedgerRow(
            run_id=run_id, timestamp=writer.timestamp, run_name=dataset,
            dataset=dataset, methods=list(methods), k_persist=k_persist,
            config_hash=config_hash, git_sha=collect_environment()["git_sha"],
            status="pending", path=str(writer.root), bif_types=[meta["bif_type"]],
            is_learned_methods=learned,
        )
    )

    try:
        for s in range(n_seeds):
            per_seed_overrides = dict(overrides)
            schedule = seed_schedule(seed_offset, s)
            per_seed_overrides["seed"] = schedule["signal"]
            per_seed_overrides["null_seed"] = schedule["null"]
            bundle = _load_bundle(dataset, per_seed_overrides, seed_base)
            for name in methods:
                method = get_method(name, meta["bif_type"])
                per_method_cfg = _train_if_learned(
                    method, bundle, writer, s, config, training
                )
                # Stash the per-seed run seed in a reserved config key so
                # stochastic methods (e.g. LSTM-AlarmNet) can seed their
                # internal RNG from the run schedule.
                per_method_cfg = dict(per_method_cfg)
                per_method_cfg["__run_seed__"] = schedule["signal"]
                evaluate_method(
                    method, bundle["signal"], bundle["null"],
                    system=meta["bif_type"], bif_type=meta["bif_type"],
                    dataset=dataset, run_name=dataset, run_id=run_id,
                    timestamp=writer.timestamp, seed=s,
                    replicate=f"s{s}",
                    config=per_method_cfg, writer=writer,
                    k_persist=k_persist, fpr_target=fpr_target,
                    evaluation=evaluation_name,
                    git_sha=collect_environment()["git_sha"],
                    config_hash=config_hash,
                    store_trajectories=store_trajectories,
                )
    except BaseException as exc:
        # Partial rows written before the failure are intentional: a
        # multi-method run that succeeds for VAR-CSD and fails on
        # LSTM-AlarmNet leaves VAR-CSD's row on disk so the
        # partial result is still analysable. Downstream consumers
        # must filter by ``status`` in the ledger / lifecycle marker.
        writer.mark_failed(exc)
        ledger.update_status(run_id, "failed")
        raise
    # Summarize before marking complete so the ``.completed`` lifecycle
    # marker is a reliable signal that the tables subtree is present.
    # If ``summarize_run`` itself raises, the ``except`` clause above
    # flips the marker to ``.failed``.
    summarize_run(writer.root)
    writer.mark_completed()
    ledger.update_status(run_id, "completed")
    return writer


def _load_bundle(dataset: str, overrides: dict[str, Any], seed_base: int) -> dict[str, Any]:
    effective = dict(overrides)
    effective.setdefault("seed", seed_base)
    bundle = get_dataset(dataset, effective)
    meta = bundle.get("meta", {})
    if not meta.get("bif_type"):
        raise ValueError(f"dataset {dataset!r} returned no meta.bif_type")
    return bundle


def _train_if_learned(
    method: Any,
    bundle: dict[str, Any],
    writer: OutputWriter,
    seed_index: int,
    config: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    """Train learned methods; return the per-method config dict.

    The checkpoint path is injected into the method's model block
    (``cfg["model"][<key>]["checkpoint"]``) so ``method.fit`` loads the
    trained weights — without this, fit would build a randomly
    initialised net and the run would silently score noise.

    Seeding order: the per-seed run seed is applied *before* the
    Lightning module is constructed so that ``nn.Linear`` / ``nn.LSTM``
    weight initialisation is deterministic. Seeding after construction
    (as the previous ordering did) left the init weights tied to the
    ambient RNG state.
    """
    if not method.meta.is_learned:
        return config
    if not training.get("enabled", True):
        raise ValueError(
            f"method {method.meta.name!r} is learned but training is disabled"
        )
    key = method.meta.config_path[0]
    _ensure_learned_modules()
    from csd_observer.training.common import fit_model
    from csd_observer.training.common.losses import alarm_bce
    from csd_observer.training.common.trainer_factory import seed_everything

    run_seed = seed_schedule(config.get("seed_offset", 0), seed_index)
    seed_everything(
        int(run_seed["signal"]),
        deterministic=bool(training.get("deterministic", True)),
    )

    lit_cls = _LEARNED_LIT_MODULES[key]
    if lit_cls is None:
        raise ValueError(f"no Lightning module registered for method key {key!r}")
    model_cfg = dict(config.get("model", {}).get(key, {}) or {})
    # Channel count is dataset-determined: synthetic hopf emits a
    # 2-channel (radius + frequency) bundle; real datasets are single
    # channel. ``None`` in the schema means "auto" — hard-coding 1 here
    # silently built a mismatched net.
    if not model_cfg.get("in_channels"):
        model_cfg["in_channels"] = int(bundle["signal"]["features"].shape[-1])
    module = lit_cls(model_cfg, training, alarm_bce)

    train_bundles = _split_bundle(bundle["signal"], "train") + _split_bundle(bundle["null"], "train")
    val_bundles = _split_bundle(bundle["signal"], "val") + _split_bundle(bundle["null"], "val")
    trainer = fit_model(
        module, train_bundles, val_bundles,
        writer=writer, method=method.meta.name, seed=run_seed["signal"], config=config,
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


def _ensure_learned_modules() -> None:
    """Import the Lightning wrappers once (torch-dependent)."""
    if _LEARNED_LIT_MODULES["lstm"] is None or _LEARNED_LIT_MODULES["tcn"] is None or _LEARNED_LIT_MODULES["patchtst"] is None:
        from csd_observer.models.neural.lstm.lit_module import LstmAlarmLitModule
        from csd_observer.models.neural.patchtst.lit_module import PatchTstAlarmLitModule
        from csd_observer.models.neural.tcn.lit_module import TcnAlarmLitModule

        _LEARNED_LIT_MODULES["lstm"] = LstmAlarmLitModule
        _LEARNED_LIT_MODULES["tcn"] = TcnAlarmLitModule
        _LEARNED_LIT_MODULES["patchtst"] = PatchTstAlarmLitModule


__all__ = ["run_benchmark"]
