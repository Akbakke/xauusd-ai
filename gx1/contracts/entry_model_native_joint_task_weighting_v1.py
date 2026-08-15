"""Immutable contract for learned model-native multi-task scalarization."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "entry_model_native_joint_task_weighting_v3"
MECHANISM = "trainable_homoscedastic_log_variance"
FORMULA = "sum(exp(-s_i) * L_i + s_i for active exact-label tasks i)"
REFERENCE = (
    "Kendall, Gal & Cipolla, Multi-Task Learning Using Uncertainty to Weigh "
    "Losses for Scene Geometry and Semantics, CVPR 2018"
)
NEUTRAL_INITIAL_LOG_VARIANCE = 0.0

# Every member is a genuine supervised task. Redundant ranking/composite
# penalties and cooperation-gate shaping are deliberately absent.
JOINT_TASK_NAMES = (
    "entry_action_q",
    "unified_exit_action",
    "side_mae_bps",
    "trendline_event",
    "position_size",
    "dip_bps",
    "forecast_return_bps",
    "dip_timing_fraction",
    "tail_risk_bps",
    "forward_volatility_bps",
)
JOINT_TASK_STATE_KEYS = tuple(
    f"task_log_variances.{name}" for name in JOINT_TASK_NAMES
)


def _canonical_sha256(value: Mapping[str, float]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def joint_task_weighting_objective_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mechanism": MECHANISM,
        "formula": FORMULA,
        "reference": REFERENCE,
        "task_names": list(JOINT_TASK_NAMES),
        "neutral_initial_log_variance": NEUTRAL_INITIAL_LOG_VARIANCE,
        "fixed_relative_task_weights": False,
        "handwritten_rank_losses": False,
        "handwritten_composite_weights": False,
        "handwritten_gate_regularization": False,
        "raw_bps_targets": True,
    }


def joint_task_weighting_metadata(
    selected_log_variances: Mapping[str, float],
    *,
    supervision_observed: Mapping[str, bool],
    gradient_observed: Mapping[str, bool],
) -> dict[str, Any]:
    selected = {name: float(selected_log_variances[name]) for name in JOINT_TASK_NAMES}
    tasks: dict[str, dict[str, Any]] = {}
    for name in JOINT_TASK_NAMES:
        try:
            precision = math.exp(-selected[name])
        except OverflowError as exc:
            raise RuntimeError(
                f"[ENTRY_EXPORT_JOINT_TASK_WEIGHTING_STATE_INVALID] task={name}"
            ) from exc
        tasks[name] = {
            "state_dict_key": f"task_log_variances.{name}",
            "initial_log_variance": NEUTRAL_INITIAL_LOG_VARIANCE,
            "selected_log_variance": selected[name],
            "effective_precision": precision,
            "supervision_observed": bool(supervision_observed.get(name, False)),
            "gradient_observed": bool(gradient_observed.get(name, False)),
            "moved_from_neutral": selected[name] != NEUTRAL_INITIAL_LOG_VARIANCE,
        }
    payload = {
        **joint_task_weighting_objective_contract(),
        "selected_log_variances": selected,
        "selected_log_variances_sha256": _canonical_sha256(selected),
        "tasks": tasks,
        "all_tasks_supervised": all(
            bool(supervision_observed.get(name, False)) for name in JOINT_TASK_NAMES
        ),
        "all_tasks_received_gradient": all(
            bool(gradient_observed.get(name, False)) for name in JOINT_TASK_NAMES
        ),
        "all_tasks_moved_from_neutral": all(
            selected[name] != NEUTRAL_INITIAL_LOG_VARIANCE
            for name in JOINT_TASK_NAMES
        ),
    }
    return require_joint_task_weighting_metadata(payload, context="ENTRY_EXPORT")


def require_joint_task_weighting_metadata(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected_objective = joint_task_weighting_objective_contract()
    for key, expected in expected_objective.items():
        if value.get(key) != expected:
            raise RuntimeError(
                f"[{context}_JOINT_TASK_WEIGHTING_CONTRACT_INVALID] field={key}"
            )
    selected = value.get("selected_log_variances")
    tasks = value.get("tasks")
    if not isinstance(selected, Mapping) or set(selected) != set(JOINT_TASK_NAMES):
        raise RuntimeError(f"[{context}_JOINT_TASK_WEIGHTING_STATE_INVALID]")
    if not isinstance(tasks, Mapping) or set(tasks) != set(JOINT_TASK_NAMES):
        raise RuntimeError(f"[{context}_JOINT_TASK_WEIGHTING_TASKS_INVALID]")
    normalized: dict[str, float] = {}
    normalized_tasks: dict[str, Any] = {}
    failures: list[str] = []
    for name in JOINT_TASK_NAMES:
        try:
            log_variance = float(selected[name])
        except (TypeError, ValueError):
            failures.append(f"{name}:non_numeric")
            continue
        if not math.isfinite(log_variance):
            failures.append(f"{name}:non_finite")
            continue
        task = tasks.get(name)
        try:
            expected_precision = math.exp(-log_variance)
        except OverflowError:
            failures.append(f"{name}:precision_overflow")
            continue
        if not isinstance(task, Mapping):
            failures.append(f"{name}:task_missing")
            continue
        if task.get("state_dict_key") != f"task_log_variances.{name}":
            failures.append(f"{name}:state_key")
        if task.get("initial_log_variance") != NEUTRAL_INITIAL_LOG_VARIANCE:
            failures.append(f"{name}:initial_not_neutral")
        if task.get("selected_log_variance") != log_variance:
            failures.append(f"{name}:selected_split_brain")
        if task.get("supervision_observed") is not True:
            failures.append(f"{name}:never_supervised")
        try:
            observed_precision = float(task.get("effective_precision"))
        except (TypeError, ValueError):
            observed_precision = float("nan")
        if not math.isfinite(observed_precision) or not math.isclose(
            observed_precision,
            expected_precision,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            failures.append(f"{name}:precision")
        if task.get("gradient_observed") is not True:
            failures.append(f"{name}:no_gradient")
        if task.get("moved_from_neutral") is not True or log_variance == 0.0:
            failures.append(f"{name}:no_movement")
        normalized[name] = log_variance
        normalized_tasks[name] = dict(task)
    if failures:
        raise RuntimeError(
            f"[{context}_JOINT_TASK_WEIGHTING_EVIDENCE_INVALID] "
            + "; ".join(failures)
        )
    if value.get("selected_log_variances_sha256") != _canonical_sha256(normalized):
        raise RuntimeError(f"[{context}_JOINT_TASK_WEIGHTING_HASH_INVALID]")
    if value.get("all_tasks_supervised") is not True:
        raise RuntimeError(f"[{context}_JOINT_TASK_WEIGHTING_SUPERVISION_UNPROVEN]")
    if value.get("all_tasks_received_gradient") is not True:
        raise RuntimeError(f"[{context}_JOINT_TASK_WEIGHTING_GRADIENTS_UNPROVEN]")
    if value.get("all_tasks_moved_from_neutral") is not True:
        raise RuntimeError(f"[{context}_JOINT_TASK_WEIGHTING_MOVEMENT_UNPROVEN]")
    return {
        **expected_objective,
        "selected_log_variances": normalized,
        "selected_log_variances_sha256": _canonical_sha256(normalized),
        "tasks": normalized_tasks,
        "all_tasks_supervised": True,
        "all_tasks_received_gradient": True,
        "all_tasks_moved_from_neutral": True,
    }
