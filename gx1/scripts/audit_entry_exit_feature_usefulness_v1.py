#!/usr/bin/env python3
"""VAL-only paired permutation audit for learned Entry/Exit usefulness.

The executable core accepts a frozen candidate predictor and immutable VAL
state tensors.  It swaps complete feature trajectories between genuine rows
according to a label-independent whole-block donor plan.  It does not retrain,
select a checkpoint, change model outputs or inspect TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    DECISION,
    DONOR_PLAN_SCHEMA_VERSION,
    PERTURBATION_POLICY,
    POLICY,
    SCHEMA_VERSION,
    SIDE_PAIR_PLAN_SCHEMA_VERSION,
    SPLIT,
    TASKS,
    TASK_CLASS_ORDER,
    canonical_json_sha256,
    feature_usefulness_layout,
    require_feature_usefulness_identity,
    require_feature_usefulness_report,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_SCHEMA_VERSION,
    entry_fitted_q_contract,
    require_entry_fitted_q_iteration_state,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
    require_unified_exit_fitted_q_iteration_state,
    unified_exit_fitted_q_contract,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


Predictor = Callable[[Mapping[str, np.ndarray]], np.ndarray]


def _array_sha256(value: np.ndarray, *, domain: bytes) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_structure_preserving_donor_plan(
    *,
    block_ids: Sequence[Any],
    within_block_positions: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate whole equal-geometry blocks; never shuffle individual rows."""

    raw_blocks = np.asarray(block_ids)
    positions = np.asarray(within_block_positions)
    if (
        raw_blocks.ndim != 1
        or positions.ndim != 1
        or raw_blocks.shape != positions.shape
        or raw_blocks.size < 2
        or positions.dtype.kind not in "iu"
    ):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_INPUT_INVALID")
    block_tokens = [str(value) for value in raw_blocks.tolist()]
    if any(not value or "\x00" in value for value in block_tokens):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_BLOCK_ID_INVALID")
    indices_by_block: dict[str, list[int]] = {}
    order: list[str] = []
    previous: str | None = None
    closed: set[str] = set()
    for index, token in enumerate(block_tokens):
        if token != previous:
            if token in closed:
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_BLOCK_NOT_CONTIGUOUS")
            if previous is not None:
                closed.add(previous)
            order.append(token)
            indices_by_block[token] = []
            previous = token
        indices_by_block[token].append(index)
    signatures: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for token in order:
        block_indices = indices_by_block[token]
        signature = tuple(int(positions[index]) for index in block_indices)
        if len(signature) != len(set(signature)):
            raise RuntimeError("FEATURE_USEFULNESS_DONOR_WITHIN_POSITION_DUPLICATE")
        signatures[signature].append(token)
    singleton_signatures = [signature for signature, rows in signatures.items() if len(rows) < 2]
    if singleton_signatures:
        raise RuntimeError(
            "FEATURE_USEFULNESS_DONOR_STRUCTURE_HAS_NO_PEER: "
            f"signature_count={len(singleton_signatures)}"
        )
    donor = np.full(raw_blocks.shape[0], -1, dtype=np.int64)
    block_mapping: dict[str, str] = {}
    for signature, tokens in signatures.items():
        for source_position, source_token in enumerate(tokens):
            donor_token = tokens[(source_position + 1) % len(tokens)]
            block_mapping[source_token] = donor_token
            source_rows = indices_by_block[source_token]
            donor_rows = indices_by_block[donor_token]
            if len(source_rows) != len(signature) or len(donor_rows) != len(signature):
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_BLOCK_GEOMETRY_INVALID")
            donor[np.asarray(source_rows, dtype=np.int64)] = np.asarray(
                donor_rows, dtype=np.int64
            )
    if (
        (donor < 0).any()
        or np.array_equal(donor, np.arange(len(donor), dtype=np.int64))
        or (donor == np.arange(len(donor), dtype=np.int64)).any()
        or sorted(donor.tolist()) != list(range(len(donor)))
    ):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_NOT_DERANGED_PERMUTATION")
    block_id_sha = canonical_json_sha256(block_tokens)
    positions_sha = _array_sha256(
        positions.astype("<i8", copy=False),
        domain=b"feature_usefulness_within_block_positions_v1\0",
    )
    donor_sha = _array_sha256(
        donor.astype("<i8", copy=False),
        domain=b"feature_usefulness_donor_indices_v1\0",
    )
    plan: dict[str, Any] = {
        "schema_version": DONOR_PLAN_SCHEMA_VERSION,
        "row_count": int(len(donor)),
        "block_count": len(order),
        "signature_group_count": len(signatures),
        "label_independent": True,
        "source_fields": ["structure_block_id", "within_block_position"],
        "block_ids_sha256": block_id_sha,
        "within_block_positions_sha256": positions_sha,
        "donor_indices_sha256": donor_sha,
        "all_rows_deranged": True,
        "whole_equal_geometry_blocks_preserved": True,
        "block_mapping_sha256": canonical_json_sha256(block_mapping),
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return donor, plan


def _require_state_surfaces(
    states: Mapping[str, Any],
    *,
    task: str,
    task_layout: Mapping[str, Any],
    row_count: int,
    timeframes: Sequence[str],
) -> dict[str, np.ndarray]:
    required = {
        "seq_signal",
        "snap_signal",
        "ctx_cont",
        "ctx_cat",
        *(f"seq_{timeframe.lower()}" for timeframe in timeframes),
    }
    if task == "exit":
        required.update(
            {
                "entry_decision_representation",
                "exit_path",
                "exit_path_lengths",
                "exit_side_index",
                "exit_episode_index",
                "exit_state_index",
            }
        )
    if not isinstance(states, Mapping) or not required.issubset(states):
        raise RuntimeError("FEATURE_USEFULNESS_STATE_SURFACES_MISSING")
    arrays = {str(name): np.asarray(value) for name, value in states.items()}
    if any(array.ndim < 1 or array.shape[0] != row_count for array in arrays.values()):
        raise RuntimeError("FEATURE_USEFULNESS_STATE_ROW_COUNT_INVALID")
    exact_shapes = {
        "seq_signal": (3, MODEL_NATIVE_SIGNAL_DIM),
        "snap_signal": (2, MODEL_NATIVE_SIGNAL_DIM),
        "ctx_cont": (2, len(MODEL_NATIVE_CTX_CONT_FIELDS)),
        "ctx_cat": (2, len(MODEL_NATIVE_CTX_CAT_FIELDS)),
    }
    for surface, (ndim, width) in exact_shapes.items():
        array = arrays[surface]
        if array.ndim != ndim or array.shape[-1] != width:
            raise RuntimeError(f"FEATURE_USEFULNESS_{surface.upper()}_SHAPE_INVALID")
    for timeframe in timeframes:
        surface = f"seq_{timeframe.lower()}"
        array = arrays[surface]
        if array.ndim != 3 or array.shape[-1] != len(MULTI_TF_PER_BAR_FEATURES_V4):
            raise RuntimeError(f"FEATURE_USEFULNESS_{surface.upper()}_SHAPE_INVALID")
    if task == "exit":
        episode_layout = task_layout["exit_episode_effects"]
        token_width = len(
            episode_layout[0]["targets"][0]["source_indices"]
        )
        path_width = len(episode_layout[1]["targets"][0]["source_indices"])
        if (
            arrays["entry_decision_representation"].ndim != 2
            or arrays["entry_decision_representation"].shape[1] != token_width
            or arrays["exit_path"].ndim != 3
            or arrays["exit_path"].shape[2] != path_width
            or arrays["exit_path_lengths"].shape != (row_count,)
            or arrays["exit_side_index"].shape != (row_count,)
            or arrays["exit_episode_index"].shape != (row_count,)
            or arrays["exit_state_index"].shape != (row_count,)
        ):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_EPISODE_SHAPE_INVALID")
        for name in (
            "exit_path_lengths", "exit_side_index", "exit_episode_index",
            "exit_state_index",
        ):
            if arrays[name].dtype.kind not in "iu":
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{name.upper()}_NOT_INTEGER"
                )
        if (
            (arrays["exit_path_lengths"] < 1).any()
            or (arrays["exit_path_lengths"] > arrays["exit_path"].shape[1]).any()
            or not np.isin(arrays["exit_side_index"], (0, 1)).all()
            or (arrays["exit_episode_index"] < 0).any()
            or (arrays["exit_state_index"] < 0).any()
        ):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_EPISODE_DOMAIN_INVALID")
    for name in required:
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"FEATURE_USEFULNESS_{name.upper()}_NONFINITE")
    ctx_cat = arrays["ctx_cat"]
    if not np.equal(ctx_cat, np.rint(ctx_cat)).all():
        raise RuntimeError("FEATURE_USEFULNESS_CTX_CAT_NOT_INTEGER")
    for index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
        if not np.isin(ctx_cat[:, index], MODEL_NATIVE_CTX_CAT_DOMAINS[field]).all():
            raise RuntimeError(f"FEATURE_USEFULNESS_CTX_CAT_DOMAIN_INVALID: {field}")
    for field, domain in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS.items():
        index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
        if not np.isin(arrays["ctx_cont"][:, index], domain).all():
            raise RuntimeError(f"FEATURE_USEFULNESS_CTX_CONT_DOMAIN_INVALID: {field}")
    mtf_index = {name: index for index, name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4)}
    for timeframe in timeframes:
        values = arrays[f"seq_{timeframe.lower()}"]
        for field, domain in MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
            if not np.isin(values[..., mtf_index[field]], domain).all():
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_MTF_DOMAIN_INVALID: {timeframe}:{field}"
                )
    return arrays


def _build_exit_side_pair_plan(
    states: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    episode = np.asarray(states["exit_episode_index"], dtype=np.int64)
    state = np.asarray(states["exit_state_index"], dtype=np.int64)
    side = np.asarray(states["exit_side_index"], dtype=np.int64)
    lookup: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(zip(episode, state, side, strict=True)):
        normalized = tuple(int(value) for value in key)
        if normalized in lookup:
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_DUPLICATE")
        lookup[normalized] = index
    pair = np.empty(len(episode), dtype=np.int64)
    for index, (episode_id, state_id, side_id) in enumerate(
        zip(episode, state, side, strict=True)
    ):
        peer = lookup.get((int(episode_id), int(state_id), 1 - int(side_id)))
        if peer is None:
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_MISSING")
        pair[index] = peer
    if (
        not np.array_equal(pair[pair], np.arange(len(pair), dtype=np.int64))
        or not np.array_equal(episode[pair], episode)
        or not np.array_equal(state[pair], state)
        or not np.array_equal(side[pair], 1 - side)
    ):
        raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_INVALID")
    token = states["entry_decision_representation"]
    for episode_id in np.unique(episode):
        rows = np.flatnonzero(episode == episode_id)
        if not np.array_equal(token[rows], np.broadcast_to(token[rows[0]], token[rows].shape)):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_FROZEN_TOKEN_NOT_EPISODE_IMMUTABLE")
    plan: dict[str, Any] = {
        "schema_version": SIDE_PAIR_PLAN_SCHEMA_VERSION,
        "row_count": len(pair),
        "source_fields": [
            "exit_episode_index", "exit_state_index", "exit_side_index"
        ],
        "pair_indices_sha256": _array_sha256(
            pair.astype("<i8", copy=False), domain=b"exit_side_pair_indices_v1\0"
        ),
        "episode_indices_sha256": _array_sha256(
            episode.astype("<i8", copy=False), domain=b"exit_episode_indices_v1\0"
        ),
        "state_indices_sha256": _array_sha256(
            state.astype("<i8", copy=False), domain=b"exit_state_indices_v1\0"
        ),
        "side_indices_sha256": _array_sha256(
            side.astype("<i8", copy=False), domain=b"exit_side_indices_v1\0"
        ),
        "involutive": True,
        "same_episode_state": True,
        "opposite_side": True,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return pair, plan


def _require_alias_manifold(
    states: Mapping[str, np.ndarray],
    perturbations: Sequence[Mapping[str, Any]],
) -> None:
    for spec in perturbations:
        signal_index = spec.get("alias_signal_index")
        ctx_index = spec.get("alias_ctx_cont_index")
        if signal_index is None and ctx_index is None:
            continue
        if signal_index is None or ctx_index is None:
            raise RuntimeError("FEATURE_USEFULNESS_ALIAS_OWNER_INCOMPLETE")
        seq = states["seq_signal"][:, -1, int(signal_index)]
        snap = states["snap_signal"][:, int(signal_index)]
        ctx = states["ctx_cont"][:, int(ctx_index)]
        if not np.array_equal(seq, snap) or not np.array_equal(snap, ctx):
            raise RuntimeError("FEATURE_USEFULNESS_ALIAS_SOURCE_OFF_MANIFOLD")


def _slice_states(states: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: array[indices] for name, array in states.items()}


def _apply_perturbation(
    baseline: Mapping[str, np.ndarray],
    donor: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    output = dict(baseline)
    cloned: dict[str, np.ndarray] = {}
    for target in spec["targets"]:
        surface = str(target["surface"])
        whole_surface = target.get("whole_surface") is True
        indices = (
            []
            if whole_surface
            else [int(index) for index in target["source_indices"]]
        )
        if surface not in output or (not whole_surface and not indices):
            raise RuntimeError("FEATURE_USEFULNESS_PERTURBATION_TARGET_INVALID")
        if surface not in cloned:
            cloned[surface] = np.array(output[surface], copy=True)
            output[surface] = cloned[surface]
        if whole_surface:
            output[surface][...] = donor[surface]
        else:
            output[surface][..., indices] = donor[surface][..., indices]
    signal_index = spec.get("alias_signal_index")
    ctx_index = spec.get("alias_ctx_cont_index")
    if signal_index is not None:
        if not np.array_equal(
            output["seq_signal"][:, -1, int(signal_index)],
            output["snap_signal"][:, int(signal_index)],
        ) or not np.array_equal(
            output["snap_signal"][:, int(signal_index)],
            output["ctx_cont"][:, int(ctx_index)],
        ):
            raise RuntimeError("FEATURE_USEFULNESS_ALIAS_PERTURBATION_OFF_MANIFOLD")
    return output


def _predict(
    *,
    states: Mapping[str, np.ndarray],
    predictor: Predictor,
    class_count: int,
    batch_rows: int,
    donor_indices: np.ndarray | None = None,
    perturbation: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    row_count = next(iter(states.values())).shape[0]
    if batch_rows < 1:
        raise RuntimeError("FEATURE_USEFULNESS_BATCH_ROWS_INVALID")
    parts: list[np.ndarray] = []
    calls = 0
    for start in range(0, row_count, batch_rows):
        stop = min(row_count, start + batch_rows)
        rows = np.arange(start, stop, dtype=np.int64)
        batch = _slice_states(states, rows)
        if perturbation is not None:
            if donor_indices is None:
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_REQUIRED")
            donor = _slice_states(states, donor_indices[rows])
            batch = _apply_perturbation(batch, donor, perturbation)
        outputs = np.asarray(predictor(batch), dtype=np.float64)
        calls += 1
        if outputs.shape != (len(rows), class_count) or not np.isfinite(outputs).all():
            raise RuntimeError("FEATURE_USEFULNESS_PREDICTOR_OUTPUTS_INVALID")
        parts.append(outputs)
    return np.concatenate(parts, axis=0), calls


def _fitted_q_loss_and_unique_target_margin(
    predicted_q_bps: np.ndarray,
    *,
    q_targets_bps: np.ndarray,
    action_valid_mask: np.ndarray,
    action_equivalence_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_q_bps, dtype=np.float64)
    if predicted.ndim != 2:
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_PREDICTION_INVALID")
    row_count, action_count = predicted.shape
    valid = np.asarray(action_valid_mask, dtype=np.bool_)
    equivalent = np.asarray(action_equivalence_mask, dtype=np.bool_)
    q_targets = np.asarray(q_targets_bps, dtype=np.float64)
    if (
        action_count < 2
        or q_targets.shape != predicted.shape
        or valid.shape != predicted.shape
        or equivalent.shape != predicted.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(q_targets).all()
        or not valid.any(axis=1).all()
        or not equivalent.any(axis=1).all()
        or (equivalent & ~valid).any()
    ):
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_SUPERVISION_INVALID")
    masked_q = np.where(valid, q_targets, -np.inf)
    expected_equivalence = valid & np.equal(
        q_targets, np.max(masked_q, axis=1, keepdims=True)
    )
    if not np.array_equal(equivalent, expected_equivalence):
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_EQUIVALENCE_INVALID")
    squared_error = np.square(predicted - q_targets)
    loss = np.where(valid, squared_error, 0.0).sum(axis=1) / valid.sum(axis=1)
    alternative = valid & ~equivalent
    margin_mask = (equivalent.sum(axis=1) == 1) & alternative.any(axis=1)
    unique_target_score = np.max(
        np.where(equivalent[margin_mask], predicted[margin_mask], -np.inf),
        axis=1,
    )
    alternative_score = np.max(
        np.where(
            alternative[margin_mask], predicted[margin_mask], -np.inf
        ),
        axis=1,
    )
    margin = unique_target_score - alternative_score
    if not np.isfinite(loss).all() or not np.isfinite(margin).all():
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_METRIC_NONFINITE")
    return loss, margin, margin_mask


def _require_exit_fitted_q_state_binding(
    *,
    states: Mapping[str, np.ndarray],
    action_valid_mask: np.ndarray,
    terminal_mask: np.ndarray,
) -> None:
    """Bind terminal action masks to the exact full-state episode surface."""

    valid = np.asarray(action_valid_mask, dtype=np.bool_)
    terminal = np.asarray(terminal_mask, dtype=np.bool_)
    episode = np.asarray(states["exit_episode_index"], dtype=np.int64)
    side = np.asarray(states["exit_side_index"], dtype=np.int64)
    state_index = np.asarray(states["exit_state_index"], dtype=np.int64)
    row_count = len(state_index)
    if (
        valid.shape != (row_count, 2)
        or terminal.shape != (row_count,)
        or episode.shape != (row_count,)
        or side.shape != (row_count,)
        or np.any(state_index < 0)
        or not np.array_equal(terminal, valid[:, 1] & ~valid[:, 0])
        or np.any(~valid[:, 1])
    ):
        raise RuntimeError(
            "FEATURE_USEFULNESS_EXIT_FITTED_Q_STATE_BINDING_INVALID"
        )
    pairs = np.column_stack([episode, side])
    for pair in np.unique(pairs, axis=0):
        rows = np.flatnonzero((episode == pair[0]) & (side == pair[1]))
        terminal_rows = rows[terminal[rows]]
        if (
            terminal_rows.size != 1
            or state_index[terminal_rows[0]] != np.max(state_index[rows])
            or np.unique(state_index[rows]).size != rows.size
        ):
            raise RuntimeError(
                "FEATURE_USEFULNESS_EXIT_FITTED_Q_STATE_BINDING_INVALID"
            )


def _paired_summary(values: np.ndarray, *, domain: bytes) -> dict[str, Any]:
    raw = np.ascontiguousarray(np.asarray(values, dtype="<f8").reshape(-1))
    if raw.size < 1 or not np.isfinite(raw).all():
        raise RuntimeError("FEATURE_USEFULNESS_PAIRED_VECTOR_INVALID")
    count = int(raw.size)
    variance = float(raw.var(ddof=1)) if count > 1 else 0.0
    return {
        "count": count,
        "sum": float(raw.sum(dtype=np.float64)),
        "mean": float(raw.mean(dtype=np.float64)),
        "sample_variance": variance,
        "standard_error": math.sqrt(variance / count),
        "minimum": float(raw.min()),
        "maximum": float(raw.max()),
        "positive_count": int((raw > 0.0).sum()),
        "zero_count": int((raw == 0.0).sum()),
        "negative_count": int((raw < 0.0).sum()),
        "paired_vector_sha256": _array_sha256(raw, domain=domain),
    }


def _effect(
    *,
    physical_id: str,
    baseline_loss: np.ndarray,
    baseline_margin: np.ndarray,
    perturbed_loss: np.ndarray,
    perturbed_margin: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    loss_delta = perturbed_loss - baseline_loss
    margin_delta = baseline_margin - perturbed_margin
    loss_summary = _paired_summary(
        loss_delta,
        domain=f"feature_usefulness_loss:{physical_id}\0".encode("utf-8"),
    )
    margin_summary = _paired_summary(
        margin_delta,
        domain=f"feature_usefulness_margin:{physical_id}\0".encode("utf-8"),
    )
    interpretation = (
        "non_positive_mean_on_both_raw_paired_metrics"
        if loss_summary["mean"] <= 0.0 and margin_summary["mean"] <= 0.0
        else "mixed_or_positive_raw_paired_evidence"
    )
    return (
        {
            "physical_id": physical_id,
            "paired_loss_delta": loss_summary,
            "paired_margin_delta": margin_summary,
            "interpretation": interpretation,
        },
        loss_delta,
        margin_delta,
    )


def audit_task_feature_usefulness(
    *,
    task: str,
    ordered_signal_names: Sequence[str],
    identity: Mapping[str, Any],
    states: Mapping[str, Any],
    row_times: Sequence[Any],
    row_splits: Sequence[str],
    block_ids: Sequence[Any],
    within_block_positions: Sequence[int],
    predictor: Predictor,
    entry_action_q_target_bps: Any | None = None,
    entry_action_valid_mask: Any | None = None,
    entry_action_equivalence_mask: Any | None = None,
    entry_fitted_q_iteration_state: Mapping[str, Any] | None = None,
    exit_fitted_q_iteration_state: Mapping[str, Any] | None = None,
    exit_action_q_target_bps: Any | None = None,
    exit_action_valid_mask: Any | None = None,
    exit_action_equivalence_mask: Any | None = None,
    exit_terminal_mask: Any | None = None,
    batch_rows: int = 256,
) -> dict[str, Any]:
    """Measure one task on complete immutable VAL rows without selection."""

    if task not in TASKS:
        raise RuntimeError("FEATURE_USEFULNESS_TASK_INVALID")
    checked_identity = require_feature_usefulness_identity(identity)
    layout = feature_usefulness_layout(ordered_signal_names)
    task_layout = layout["tasks"][task]
    times = pd.DatetimeIndex(pd.to_datetime(row_times, utc=True, errors="raise"))
    row_count = len(times)
    if (
        row_count < 2
        or row_count
        != checked_identity[f"{task}_val_population_row_count"]
    ):
        raise RuntimeError("FEATURE_USEFULNESS_ROW_CLOCK_INVALID")
    if len(row_splits) != row_count or any(str(value) != SPLIT for value in row_splits):
        raise RuntimeError("FEATURE_USEFULNESS_NON_VAL_ROW_FORBIDDEN")
    val_start = pd.Timestamp(checked_identity["val_start_utc"]).tz_convert("UTC")
    val_end = pd.Timestamp(checked_identity["val_end_utc"]).tz_convert("UTC")
    if (times < val_start).any() or (times > val_end).any():
        raise RuntimeError("FEATURE_USEFULNESS_FUTURE_OR_OUTSIDE_VAL_ROW_FORBIDDEN")
    class_count = len(TASK_CLASS_ORDER[task])
    arrays = _require_state_surfaces(
        states,
        task=task,
        task_layout=task_layout,
        row_count=row_count,
        timeframes=task_layout["timeframes"],
    )
    _require_alias_manifold(
        arrays,
        task_layout["physical_field_perturbations"],
    )
    donor, donor_plan = build_structure_preserving_donor_plan(
        block_ids=block_ids,
        within_block_positions=within_block_positions,
    )
    side_pair_indices: np.ndarray | None = None
    side_pair_plan: dict[str, Any] | None = None
    if task == "exit":
        side_pair_indices, side_pair_plan = _build_exit_side_pair_plan(arrays)
    baseline_outputs, forward_calls = _predict(
        states=arrays,
        predictor=predictor,
        class_count=class_count,
        batch_rows=batch_rows,
    )
    if task == "entry":
        if any(
            value is not None
            for value in (
                exit_action_q_target_bps, exit_action_valid_mask,
                exit_action_equivalence_mask, exit_terminal_mask,
            )
        ):
            raise RuntimeError("FEATURE_USEFULNESS_ENTRY_EXIT_TARGETS_FORBIDDEN")
        q_targets = np.asarray(entry_action_q_target_bps, dtype=np.float64)
        action_valid = np.asarray(entry_action_valid_mask, dtype=np.bool_)
        action_equivalent = np.asarray(
            entry_action_equivalence_mask, dtype=np.bool_
        )
        if not isinstance(entry_fitted_q_iteration_state, Mapping) or not isinstance(
            exit_fitted_q_iteration_state, Mapping
        ):
            raise RuntimeError(
                "FEATURE_USEFULNESS_ENTRY_FITTED_Q_ITERATION_REQUIRED"
            )
        exit_iteration = require_unified_exit_fitted_q_iteration_state(
            exit_fitted_q_iteration_state,
            context="FEATURE_USEFULNESS_ENTRY_EXIT_TEACHER",
        )
        entry_iteration = require_entry_fitted_q_iteration_state(
            entry_fitted_q_iteration_state,
            exit_fitted_q_iteration_state=exit_iteration,
            context="FEATURE_USEFULNESS_ENTRY",
        )
        baseline_loss, baseline_margin, margin_mask = (
            _fitted_q_loss_and_unique_target_margin(
                baseline_outputs,
                q_targets_bps=q_targets,
                action_valid_mask=action_valid,
                action_equivalence_mask=action_equivalent,
            )
        )
        if not action_valid[:, 2].all():
            raise RuntimeError("FEATURE_USEFULNESS_ENTRY_FLAT_MASK_INVALID")
        loss_mask = np.ones(row_count, dtype=np.bool_)
        supervision: dict[str, Any] = {
            "schema_version": ENTRY_FITTED_Q_SCHEMA_VERSION,
            "fitted_q_contract": entry_fitted_q_contract(),
            "fitted_q_iteration_state": entry_iteration,
            "fitted_q_iteration_state_sha256": canonical_json_sha256(
                entry_iteration
            ),
            "exit_fitted_q_iteration_state": exit_iteration,
            "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(
                exit_iteration
            ),
            "q_targets_bps_sha256": _array_sha256(
                np.ascontiguousarray(q_targets, dtype="<f8"),
                domain=b"feature_usefulness_entry_fitted_q_target_v1\0",
            ),
            "action_valid_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_valid, dtype="u1"),
                domain=b"feature_usefulness_entry_action_valid_v1\0",
            ),
            "action_equivalence_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_equivalent, dtype="u1"),
                domain=b"feature_usefulness_entry_action_equivalence_v1\0",
            ),
            "loss_valid_row_count": row_count,
            "margin_valid_row_count": int(margin_mask.sum()),
            "action_valid_cell_count": int(action_valid.sum()),
            "target_tied_row_count": int(
                (action_equivalent.sum(axis=1) > 1).sum()
            ),
            "single_valid_action_row_count": int(
                (action_valid.sum(axis=1) == 1).sum()
            ),
        }
    else:
        if any(
            value is not None
            for value in (
                entry_action_q_target_bps,
                entry_action_valid_mask,
                entry_action_equivalence_mask,
                entry_fitted_q_iteration_state,
            )
        ):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_ENTRY_TARGETS_FORBIDDEN")
        if not isinstance(exit_fitted_q_iteration_state, Mapping):
            raise RuntimeError(
                "FEATURE_USEFULNESS_EXIT_FITTED_Q_ITERATION_REQUIRED"
            )
        exit_iteration = require_unified_exit_fitted_q_iteration_state(
            exit_fitted_q_iteration_state,
            context="FEATURE_USEFULNESS_EXIT",
        )
        q_targets = np.asarray(exit_action_q_target_bps, dtype=np.float64)
        action_valid = np.asarray(exit_action_valid_mask, dtype=np.bool_)
        action_equivalent = np.asarray(
            exit_action_equivalence_mask, dtype=np.bool_
        )
        terminal = np.asarray(exit_terminal_mask, dtype=np.bool_)
        _require_exit_fitted_q_state_binding(
            states=arrays,
            action_valid_mask=action_valid,
            terminal_mask=terminal,
        )
        baseline_loss, baseline_margin, margin_mask = (
            _fitted_q_loss_and_unique_target_margin(
                baseline_outputs,
                q_targets_bps=q_targets,
                action_valid_mask=action_valid,
                action_equivalence_mask=action_equivalent,
            )
        )
        loss_mask = np.ones(row_count, dtype=np.bool_)
        supervision = {
            "schema_version": UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
            "fitted_q_contract": unified_exit_fitted_q_contract(),
            "fitted_q_iteration_state": exit_iteration,
            "fitted_q_iteration_state_sha256": canonical_json_sha256(
                exit_iteration
            ),
            "q_targets_bps_sha256": _array_sha256(
                np.ascontiguousarray(q_targets, dtype="<f8"),
                domain=b"feature_usefulness_exit_fitted_q_bellman_target_v1\0",
            ),
            "action_valid_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_valid, dtype="u1"),
                domain=b"feature_usefulness_exit_action_valid_v1\0",
            ),
            "action_equivalence_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_equivalent, dtype="u1"),
                domain=b"feature_usefulness_exit_action_equivalence_v1\0",
            ),
            "terminal_mask_sha256": _array_sha256(
                np.ascontiguousarray(terminal, dtype="u1"),
                domain=b"feature_usefulness_exit_terminal_v1\0",
            ),
            "loss_valid_row_count": row_count,
            "margin_valid_row_count": int(margin_mask.sum()),
            "action_valid_cell_count": int(action_valid.sum()),
            "target_tied_row_count": int(
                (action_equivalent.sum(axis=1) > 1).sum()
            ),
            "single_valid_action_row_count": int(
                (action_valid.sum(axis=1) == 1).sum()
            ),
            "terminal_row_count": int(terminal.sum()),
        }

    def evaluate_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        nonlocal forward_calls
        donor_kind = str(spec.get("donor_kind", "structure_block"))
        selected_donor = donor
        if donor_kind == "same_state_opposite_side":
            if side_pair_indices is None:
                raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_REQUIRED")
            selected_donor = side_pair_indices
        elif donor_kind != "structure_block":
            raise RuntimeError("FEATURE_USEFULNESS_DONOR_KIND_INVALID")
        perturbed_outputs, calls = _predict(
            states=arrays,
            predictor=predictor,
            class_count=class_count,
            batch_rows=batch_rows,
            donor_indices=selected_donor,
            perturbation=spec,
        )
        forward_calls += calls
        loss_all, margin_values, perturbed_margin_mask = (
            _fitted_q_loss_and_unique_target_margin(
                perturbed_outputs,
                q_targets_bps=q_targets,
                action_valid_mask=action_valid,
                action_equivalence_mask=action_equivalent,
            )
        )
        if not np.array_equal(perturbed_margin_mask, margin_mask):
            raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_MARGIN_MASK_CHANGED")
        perturbed_loss = loss_all[loss_mask]
        perturbed_margin = margin_values
        return _effect(
            physical_id=str(spec["physical_id"]),
            baseline_loss=baseline_loss,
            baseline_margin=baseline_margin,
            perturbed_loss=perturbed_loss,
            perturbed_margin=perturbed_margin,
        )

    physical_metrics: dict[str, dict[str, Any]] = {}
    for spec in task_layout["physical_field_perturbations"]:
        metric, _loss, _margin = evaluate_spec(spec)
        physical_metrics[str(spec["physical_id"])] = metric

    logical_metrics: dict[str, dict[str, Any]] = {}
    for group, rows in task_layout["logical_fields"].items():
        logical_metrics[group] = {
            str(row["token"]): dict(physical_metrics[str(row["physical_id"])])
            for row in rows
        }

    route_metrics: dict[str, dict[str, Any]] = {}
    component_metrics: dict[str, dict[str, Any]] = {}
    component_vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in task_layout["family_tf_routes"]:
        metric, loss_delta, margin_delta = evaluate_spec(spec)
        route_metrics[str(spec["token"])] = metric
        component_metrics[str(spec["physical_id"])] = metric
        component_vectors[str(spec["physical_id"])] = (
            loss_delta, margin_delta
        )

    for section in ("local_family_effects", "joint_effects"):
        for spec in task_layout[section]:
            metric, loss_delta, margin_delta = evaluate_spec(spec)
            physical_id = str(spec["physical_id"])
            component_metrics[physical_id] = metric
            component_vectors[physical_id] = (loss_delta, margin_delta)

    synergy_metrics: dict[str, dict[str, Any]] = {}
    for row in task_layout["interaction_synergy"]:
        left_id = str(row["left_effect_id"])
        right_id = str(row["right_effect_id"])
        joint_id = str(row["joint_effect_id"])
        left_loss, left_margin = component_vectors[left_id]
        right_loss, right_margin = component_vectors[right_id]
        joint_loss, joint_margin = component_vectors[joint_id]
        synergy_loss = joint_loss - left_loss - right_loss
        synergy_margin = joint_margin - left_margin - right_margin
        token = str(row["token"])
        synergy_metrics[token] = {
            "kind": row["kind"],
            "formula": row["formula"],
            "left_effect_id": left_id,
            "right_effect_id": right_id,
            "joint_effect_id": joint_id,
            "left_effect": component_metrics[left_id],
            "right_effect": component_metrics[right_id],
            "joint_effect": component_metrics[joint_id],
            "paired_loss_delta": _paired_summary(
                synergy_loss,
                domain=f"feature_usefulness_synergy_loss:{task}:{token}\0".encode("utf-8"),
            ),
            "paired_margin_delta": _paired_summary(
                synergy_margin,
                domain=f"feature_usefulness_synergy_margin:{task}:{token}\0".encode("utf-8"),
            ),
        }

    episode_metrics: dict[str, dict[str, Any]] = {}
    for spec in task_layout["exit_episode_effects"]:
        metric, _loss, _margin = evaluate_spec(spec)
        episode_metrics[str(spec["token"])] = metric

    logical_count = sum(len(rows) for rows in task_layout["logical_fields"].values())
    coverage = {
        **task_layout["coverage_counts"],
        "reported_logical_fields": logical_count,
        "reported_family_tf_routes": len(route_metrics),
        "reported_exit_episode_effects": len(episode_metrics),
        "reported_interaction_synergy": len(synergy_metrics),
        "omitted_tokens": [],
        "complete": True,
    }
    return {
        "ordered_signal_names": list(ordered_signal_names),
        "row_count": row_count,
        "class_order": list(TASK_CLASS_ORDER[task]),
        "comparison_surface": (
            "raw_entry_action_q_bps_valid_action_masked_mse_and_unique_target_q_margin"
            if task == "entry"
            else "raw_exit_action_q_bps_frozen_fitted_q_bellman_target_masked_mse_and_unique_target_q_margin"
        ),
        "row_times_sha256": _array_sha256(
            np.ascontiguousarray(times.asi8, dtype="<i8"),
            domain=f"feature_usefulness_times:{task}\0".encode("utf-8"),
        ),
        "supervision": supervision,
        "baseline_outputs_sha256": _array_sha256(
            np.ascontiguousarray(baseline_outputs, dtype="<f8"),
            domain=f"feature_usefulness_baseline_outputs:{task}\0".encode("utf-8"),
        ),
        "frozen_entry_decision_token_sha256": (
            None
            if task == "entry"
            else _array_sha256(
                np.ascontiguousarray(
                    arrays["entry_decision_representation"], dtype="<f8"
                ),
                domain=b"feature_usefulness_exit_frozen_entry_token_v1\0",
            )
        ),
        "donor_plan": donor_plan,
        "side_pair_plan": side_pair_plan,
        "forward_variant_count": (
            1
            + len(task_layout["physical_field_perturbations"])
            + len(task_layout["family_tf_routes"])
            + len(task_layout["local_family_effects"])
            + len(task_layout["joint_effects"])
            + len(task_layout["exit_episode_effects"])
        ),
        "logical_field_metrics": logical_metrics,
        "family_tf_route_metrics": route_metrics,
        "exit_episode_effect_metrics": episode_metrics,
        "interaction_synergy_metrics": synergy_metrics,
        "coverage": coverage,
        "_forward_batch_calls": forward_calls,
    }


def build_feature_usefulness_report(
    *,
    identity: Mapping[str, Any],
    ordered_signal_names: Sequence[str],
    entry_task: Mapping[str, Any],
    exit_task: Mapping[str, Any],
    created: datetime | None = None,
) -> dict[str, Any]:
    checked_identity = require_feature_usefulness_identity(identity)
    layout = feature_usefulness_layout(ordered_signal_names)
    tasks: dict[str, Any] = {}
    for task, raw in (("entry", entry_task), ("exit", exit_task)):
        row = dict(raw)
        row.pop("_forward_batch_calls", None)
        tasks[task] = row
    created = created or datetime.now(timezone.utc)
    if created.tzinfo is None or created.utcoffset() is None:
        raise RuntimeError("FEATURE_USEFULNESS_CREATED_UTC_INVALID")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created.astimezone(timezone.utc).isoformat(),
        "decision": DECISION,
        "split": SPLIT,
        "policy": dict(POLICY),
        "perturbation_policy": dict(PERTURBATION_POLICY),
        "identity": checked_identity,
        "identity_sha256": canonical_json_sha256(checked_identity),
        "layout_sha256": layout["layout_sha256"],
        "layout_counts": {
            task: layout["tasks"][task]["coverage_counts"] for task in TASKS
        },
        "tasks": tasks,
        "test_rows_read": False,
        "test_artifacts_read": [],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return require_feature_usefulness_report(report)


def write_immutable_feature_usefulness_report(
    path: Path,
    report: Mapping[str, Any],
) -> Path:
    checked = require_feature_usefulness_report(report)
    out = path.expanduser().resolve()
    if out.exists() or out.is_symlink():
        raise RuntimeError(f"FEATURE_USEFULNESS_OUTPUT_EXISTS: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(checked, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(out, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one immutable VAL-only Entry/Exit usefulness report"
    )
    parser.add_argument("--validate-json", type=Path, required=True)
    args = parser.parse_args()
    path = args.validate_json.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    require_feature_usefulness_report(value)
    print(json.dumps({"decision": DECISION, "path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "audit_task_feature_usefulness",
    "build_feature_usefulness_report",
    "build_structure_preserving_donor_plan",
    "write_immutable_feature_usefulness_report",
]
