"""Pure argv builders for the CLI commands a plan step executes.

Kept side-effect free and deterministic: the runner's correctness hinges on
these exact overrides (dataset, model, models list, seed_offset, n_seeds,
outputs base dir, evaluation mode, and checkpoint path).
"""

from __future__ import annotations

from pathlib import Path

from csd_observer.runner.protocol import Step

_MODEL_KEY_BY_NAME = {
    "LSTM-AlarmNet": "lstm",
    "TCN-AlarmNet": "tcn",
    "PatchTST-AlarmNet": "patchtst",
}


def _model_key(method_name: str, model_config: str) -> str:
    if method_name in _MODEL_KEY_BY_NAME:
        return _MODEL_KEY_BY_NAME[method_name]
    if model_config and model_config != "default":
        return model_config
    return method_name.lower().replace("-", "_")


def train_command(
    step: Step,
    *,
    outputs_base: Path,
    defaults: tuple[str, ...],
    ckpt_path: Path | None = None,
) -> list[str]:
    """Build ``csd-observer`` argv for a training step."""
    method = step.method
    cmd = [
        "csd-observer",
        f"dataset={method.data}",
        f"model={method.model}",
        f"models=[{method.name}]",
        f"seed_offset={step.seed}",
        "n_seeds=1",
        f"output.base_dir={outputs_base.as_posix()}",
        "training=default",
        "training.enabled=true",
    ]
    # For short datasets (daphnia_ext), default training.label_window=10 to respect min_length
    if method.data == "daphnia_ext" and not any("training.label_window" in d for d in defaults):
        cmd.append("training.label_window=10")

    if ckpt_path is not None:
        key = _model_key(method.name, method.model)
        cmd.append(f"model.{key}.checkpoint={ckpt_path.as_posix()}")
    cmd.extend(defaults)
    return cmd


def eval_command(
    step: Step,
    *,
    outputs_base: Path,
    defaults: tuple[str, ...],
    ckpt_path: Path | None = None,
) -> list[str]:
    """Build ``csd-observer`` argv for an evaluation step."""
    method = step.method
    cmd = [
        "csd-observer",
        f"dataset={method.data}",
        f"model={method.model}",
        f"models=[{method.name}]",
        f"seed_offset={step.seed}",
        "n_seeds=1",
        f"output.base_dir={outputs_base.as_posix()}",
        f"evaluation={method.evaluate_mode}",
        "training=none",
    ]
    if ckpt_path is not None:
        key = _model_key(method.name, method.model)
        cmd.append(f"model.{key}.checkpoint={ckpt_path.as_posix()}")
    cmd.extend(defaults)
    return cmd


__all__ = [
    "eval_command",
    "train_command",
]
