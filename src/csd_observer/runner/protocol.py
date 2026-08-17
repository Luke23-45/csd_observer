"""Frozen experiment protocol: load + validate the JSON matrix, build the plan.

The protocol manifest (e.g. ``experiments/synthetic_matrix.json``) is the single source
of truth describing every baseline method, its training stages, its Stage 1
source dependency (if any), and whether a complete run includes the persistence-aware
evaluation. ``load_protocol`` validates it loudly so a malformed matrix
never silently produces a partial sweep.

A *plan* is the ordered list of concrete steps produced by
:func:`build_plan`: for every selected method and seed, all training stages
in order, then the evaluation of the final-stage checkpoint, with any
required pretraining stages injected first when ``with_dependencies`` is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VALID_STAGES = frozenset({1, 2})
# Valid dataset config names matching files under ``configs/dataset/``
_VALID_DATA = frozenset(
    {
        "synthetic_fold",
        "synthetic_hopf",
        "synthetic_logistic",
        "tac",
        "daphnia_ext",
    }
)
_VALID_EVAL_MODES = frozenset({"persistenceaware", "baseline_classic"})


class ProtocolError(ValueError):
    """Raised when a protocol manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class Method:
    """One baseline method in the protocol."""

    index: int
    name: str
    role: str
    model: str
    data: str
    stages: tuple[int, ...]
    stage2_source: str | None
    evaluate: bool
    tag: str | None = None
    evaluate_mode: str = "persistenceaware"
    task: str | None = None

    @property
    def model_name(self) -> str:
        """Filesystem/config model name (e.g. ``lstm``, ``tcn``, ``default``)."""
        return self.model.rsplit("/", 1)[-1]

    @property
    def final_stage(self) -> int:
        """The stage whose checkpoint a complete run evaluates."""
        return self.stages[-1] if self.stages else 1

    @property
    def phase_key(self) -> str:
        """Registry key prefix for this method's phases (stable identity).

        For multi-dataset protocols, methods on different tasks share
        the same ``name`` but must have distinct output trees. Including the
        task in the phase_key prevents cross-task artifact collisions.
        """
        if self.task:
            return f"{self.task}/{self.name}"
        return self.name

    @property
    def output_tag(self) -> str | None:
        """Tag used to isolate this method's artifacts from other tasks."""
        parts: list[str] = []
        if self.task:
            parts.append(self.task)
        if self.tag:
            parts.append(self.tag)
        return "__".join(parts) if parts else None


@dataclass(frozen=True)
class Protocol:
    """The validated protocol document."""

    name: str
    task: str
    description: str
    seeds: tuple[int, ...]
    defaults: tuple[str, ...]
    methods: tuple[Method, ...]

    def method_by_name(self, name: str, *, task: str | None = None) -> Method | None:
        for m in self.methods:
            if m.name == name and (task is None or m.task == task):
                return m
        return None

    def method_by_index(self, index: int, *, task: str | None = None) -> Method | None:
        for m in self.methods:
            if m.index == index and (task is None or m.task == task):
                return m
        return None

    def select_methods(self, refs: list[str]) -> list[Method]:
        """Resolve a list of user references to ordered :class:`Method`s.

        Each reference may be a method name or its numeric index. Unknown
        references raise a :class:`ProtocolError` listing every valid option.
        """
        selected: list[Method] = []
        seen: set[tuple[str | None, str]] = set()
        for ref in refs:
            method: Method | None = None
            if ref.isdigit():
                method = self.method_by_index(int(ref))
                if method is None:
                    raise ProtocolError(
                        f"Unknown method index {ref!r}. Valid indices: "
                        f"{[m.index for m in self.methods]}."
                    )
            else:
                matches = [m for m in self.methods if m.name == ref]
                if not matches:
                    raise ProtocolError(
                        f"Unknown method {ref!r}. Valid names: {[m.name for m in self.methods]}."
                    )
                if len(matches) > 1:
                    tasks = [m.task or "default" for m in matches]
                    raise ProtocolError(
                        f"Method name {ref!r} is ambiguous across tasks {tasks}; "
                        "select it by manifest index."
                    )
                method = matches[0]
            identity = (method.task, method.name)
            if identity not in seen:
                selected.append(method)
                seen.add(identity)
        return sorted(selected, key=lambda m: m.index)


def _require(data: dict[str, Any], key: str, label: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise ProtocolError(f"Protocol method is missing {label} ({key!r}).")
    return data[key]


def _parse_method(raw: dict[str, Any]) -> Method:
    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol method must be a JSON object, got {type(raw).__name__}.")

    index = raw.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ProtocolError(f"Method {raw.get('name', '?')!r}: 'index' must be an int >= 1.")
    name = _require(raw, "name", "a name")
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ProtocolError(f"Method {index}: 'name' must be [a-zA-Z0-9_-]+ (got {name!r}).")
    role = str(_require(raw, "role", "a role description"))
    model = _require(raw, "model", "a model config path")
    if not isinstance(model, str) or not model:
        raise ProtocolError(f"Method {name!r}: 'model' must be a non-empty string.")
    data = str(raw.get("data", raw.get("dataset", "synthetic_fold")))
    if data not in _VALID_DATA:
        raise ProtocolError(
            f"Method {name!r}: 'data'/'dataset' must be one of {sorted(_VALID_DATA)}, got {data!r}."
        )

    stages_raw = raw.get("stages", [])
    if not isinstance(stages_raw, list):
        raise ProtocolError(f"Method {name!r}: 'stages' must be a list.")
    stages: list[int] = []
    for s in stages_raw:
        if isinstance(s, bool) or not isinstance(s, int) or s not in _VALID_STAGES:
            raise ProtocolError(
                f"Method {name!r}: stage {s!r} must be one of {sorted(_VALID_STAGES)}."
            )
        if s not in stages:
            stages.append(s)
    if stages != sorted(stages):
        raise ProtocolError(f"Method {name!r}: 'stages' must be ascending and unique.")

    stage2_source = raw.get("stage2_source")
    if stage2_source is not None:
        if 2 not in stages:
            raise ProtocolError(f"Method {name!r}: 'stage2_source' set but method has no stage 2.")

    evaluate = raw.get("evaluate", True)
    if not isinstance(evaluate, bool):
        raise ProtocolError(f"Method {name!r}: 'evaluate' must be a boolean.")

    tag = raw.get("tag")
    if tag is not None and (not isinstance(tag, str) or not tag):
        raise ProtocolError(f"Method {name!r}: 'tag' must be a non-empty string or null.")

    evaluate_mode = str(raw.get("evaluate_mode", "persistenceaware"))
    if evaluate_mode not in _VALID_EVAL_MODES:
        raise ProtocolError(
            f"Method {name!r}: 'evaluate_mode' must be one of "
            f"{sorted(_VALID_EVAL_MODES)}, got {evaluate_mode!r}."
        )

    task = raw.get("task")
    if task is not None and (not isinstance(task, str) or not task):
        raise ProtocolError(f"Method {name!r}: 'task' must be a non-empty string or null.")

    return Method(
        index=index,
        name=name,
        role=role,
        model=model,
        data=data,
        stages=tuple(stages),
        stage2_source=stage2_source,
        evaluate=evaluate,
        tag=tag,
        evaluate_mode=evaluate_mode,
        task=task,
    )


def load_protocol(path: str | Path) -> Protocol:
    """Load and strictly validate a protocol manifest.

    Raises:
        ProtocolError: The document is malformed (bad JSON, missing keys,
            duplicate indices, non-int seeds, ...).
        OSError: The file cannot be read.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Protocol {p} is not valid JSON: {exc.msg}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol {p} must be a JSON object.")

    name = _require(raw, "name", "a protocol name")
    task = str(_require(raw, "task", "a task name"))
    description = str(raw.get("description", ""))

    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ProtocolError(f"Protocol {name!r}: 'seeds' must be a non-empty list of ints.")
    seeds: list[int] = []
    for s in seeds_raw:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ProtocolError(f"Protocol {name!r}: seed {s!r} must be an int.")
        if s not in seeds:
            seeds.append(s)

    defaults_raw = raw.get("defaults", [])
    if not isinstance(defaults_raw, list) or not all(
        isinstance(d, str) and d for d in defaults_raw
    ):
        raise ProtocolError(f"Protocol {name!r}: 'defaults' must be a list of override strings.")
    defaults = tuple(defaults_raw)

    methods_raw = raw.get("methods")
    if not isinstance(methods_raw, list) or not methods_raw:
        raise ProtocolError(f"Protocol {name!r}: 'methods' must be a non-empty list.")

    methods: list[Method] = [_parse_method(m) for m in methods_raw]
    seen_indices: dict[tuple[str | None, int], str] = {}
    seen_names: set[tuple[str | None, str]] = set()
    for m in methods:
        task_key: str | None = m.task
        identity: tuple[str | None, str] = (task_key, m.name)
        index_key: tuple[str | None, int] = (task_key, m.index)
        if index_key in seen_indices:
            raise ProtocolError(
                f"Duplicate method index {m.index} for task {task_key!r} "
                f"({m.name!r} vs {seen_indices[index_key]!r})."
            )
        if identity in seen_names:
            raise ProtocolError(
                f"Duplicate method name {m.name!r}"
                + (f" for task {task_key!r}" if task_key else ".")
            )
        seen_indices[index_key] = m.name
        seen_names.add(identity)
    methods.sort(key=lambda m: (m.task is None, m.task or "", m.index))

    # Cross-checks for multi-stage dependencies
    for m in methods:
        if m.stage2_source is not None:
            provider = next(
                (p for p in methods if p.name == m.stage2_source and p.task == m.task),
                None,
            )
            if provider is None:
                raise ProtocolError(
                    f"Method {m.name!r} (task={m.task!r}): 'stage2_source' "
                    f"{m.stage2_source!r} is not a method in this task."
                )
            if 1 not in provider.stages:
                raise ProtocolError(
                    f"Method {m.name!r}: provider {provider.name!r} has no stage 1."
                )

    return Protocol(
        name=str(name),
        task=task,
        description=description,
        seeds=tuple(seeds),
        defaults=defaults,
        methods=tuple(methods),
    )


@dataclass(frozen=True)
class Step:
    """One concrete run step: a training stage or the evaluation of a method."""

    kind: str
    method: Method
    seed: int
    stage: int | None = None
    dependency: bool = False

    @property
    def label(self) -> str:
        task_prefix = f"[{self.method.task}] " if self.method.task else ""
        if self.kind == "eval":
            return f"{task_prefix}{self.method.name} seed={self.seed} eval"
        return f"{task_prefix}{self.method.name} seed={self.seed} stage{self.stage}"

    @property
    def registry_phase(self) -> str:
        """Registry phase name (``stage1``/``stage2``/``eval``)."""
        if self.kind == "eval":
            return "eval"
        return f"stage{self.stage}"

    @property
    def phase_key(self) -> str:
        """Registry phase key (stable identity for resume/skip logic)."""
        return f"{self.method.phase_key}/{self.seed}/{self.registry_phase}"

    def required_checkpoint(self) -> tuple[str, int] | None:
        """The ``(model, stage)`` artifact this step must load, or ``None``.

        * A stage-2 step loads the Stage 1 checkpoint of its provider
          (``"self"`` -> this method's own stage 1).
        * An eval step loads the method's final-stage checkpoint (if learned/trained).
        * A stage-1 step loads nothing.
        """
        if self.kind == "eval" and self.method.stages:
            return (self.method.model_name, self.method.final_stage)
        if self.kind == "train" and self.stage == 2:
            source = self.method.stage2_source
            if source == "self":
                return (self.method.model_name, 1)
            if source:
                return (source, 1)
        return None


def _validate_seed_filter(protocol: Protocol, seeds_filter: list[int] | None) -> list[int]:
    if not seeds_filter:
        return list(protocol.seeds)
    for s in seeds_filter:
        if s not in protocol.seeds:
            raise ProtocolError(
                f"Seed {s} is not in the protocol ({list(protocol.seeds)}). "
                "Adjust the protocol manifest or choose a protocol seed."
            )
    return list(seeds_filter)


def build_plan(
    protocol: Protocol,
    methods: list[Method],
    *,
    seeds: list[int] | None = None,
    stage: int | None = None,
    eval_only: bool = False,
    skip_eval: bool = False,
    with_dependencies: bool = False,
) -> list[Step]:
    """Build the ordered plan of steps for the selected methods and seeds.

    Ordering: methods by manifest index; within a method, seeds in protocol order and
    each seed's training stages ascending, followed by the method's eval.
    When ``with_dependencies`` is set and a selected method's Stage 1
    provider is not itself selected, the provider's Stage 1 (train only,
    marked ``dependency``) is prepended.
    """
    if eval_only and stage is not None:
        raise ProtocolError("--stage and --eval-only are mutually exclusive.")
    if eval_only and skip_eval:
        raise ProtocolError("--eval-only and --skip-eval are mutually exclusive.")
    if stage is not None and stage not in (1, 2):
        raise ProtocolError(f"--stage must be 1 or 2, got {stage}.")

    seed_list = _validate_seed_filter(protocol, seeds)
    selected_identities = {(m.task, m.name) for m in methods}

    deps: list[Step] = []
    if with_dependencies and not eval_only and stage is None:
        for m in methods:
            if m.stage2_source and m.stage2_source != "self":
                provider_identity = (m.task, m.stage2_source)
                if provider_identity not in selected_identities:
                    provider = protocol.method_by_name(m.stage2_source, task=m.task)
                    if provider is None:
                        raise ProtocolError(
                            f"Method {m.name!r}: provider {m.stage2_source!r} is missing "
                            "from the protocol."
                        )
                    for seed in seed_list:
                        deps.append(
                            Step(kind="train", method=provider, seed=seed, stage=1, dependency=True)
                        )
        seen: set[tuple[str | None, str, int]] = set()
        deduped: list[Step] = []
        for step in deps:
            key = (step.method.task, step.method.name, step.seed)
            if key not in seen:
                seen.add(key)
                deduped.append(step)
        deps = sorted(deduped, key=lambda s: (s.method.index, seed_list.index(s.seed)))

    steps: list[Step] = []
    for method in methods:
        for seed in seed_list:
            if eval_only:
                if method.evaluate:
                    steps.append(Step(kind="eval", method=method, seed=seed))
                continue
            for stage_n in method.stages:
                if stage is not None and stage_n != stage:
                    continue
                steps.append(Step(kind="train", method=method, seed=seed, stage=stage_n))
            if method.evaluate and not skip_eval and stage is None:
                steps.append(Step(kind="eval", method=method, seed=seed))

    return deps + steps


__all__ = [
    "Method",
    "Protocol",
    "ProtocolError",
    "Step",
    "build_plan",
    "load_protocol",
]
