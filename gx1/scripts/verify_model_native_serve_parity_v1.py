#!/usr/bin/env python3
"""TRAIN==SERVE parity gate for the LIVE model-native Entry serving path — MANDATORY
before anything downstream of the serving wave opens (vedtak
SMART_JOINT_POLICY_PROMOTION_20260708, serving-wave gap: parity proof).

What it proves, bar-for-bar, on exactly 256 deterministic positions spanning
the complete, hash-bound offline TEST split:

  LEG 1 (STATE, hard tolerance 1e-5): the LIVE state builder
  (gx1.execution.v12_model_native_state_live.ModelNativeStateBuilder) run on the LIVE
  cv3/BASE28 prebuilts reproduces the OFFLINE dataset rows named by
  --dataset-dir and the active bundle's model_native_state_contract exactly:
  seq (96,513) + snap (513) + ctx_cont (142) + ctx_cat (5). FAIL-LOUD per
  column name on any deviation.
  The frame-end decision bar is held to the same hard contract as every other
  bar. Any offline feature that needs future bars is leakage and fails parity.

  LEG 2 (FORWARD, hard tolerance 1e-3): the LIVE adapter
  (gx1.execution.v12_smart_entry_live.SmartEntryLiveInference) forwards those
  live states through the contract-resolved calibrated bundle and must
  reproduce fresh-event-pinned model_direction_argmax predictions: the final
  direction_logits, their canonical public trade/flat pair, the direct
  LONG/SHORT/FLAT argmax, probabilities, and auxiliary path heads. Legacy
  expected-utility or edge/session decision evidence is rejected before any
  live state/model work begins.
  NOTE: the pinned predictions were computed on CUDA; this gate runs the live
  CPU path, so LEG 2 tolerance covers numeric-backend drift ONLY — LEG 1 is the
  bit-level state proof.

Run under the capped runner (heavy: full prebuilt load + augmenters ~7 min):
  scripts/gx1_capped_run.sh --mem 10G --swap 512M -- .venv/bin/python -m \
      gx1.scripts.verify_model_native_serve_parity_v1 \
      --dataset-dir /absolute/model_native_dataset \
      --pair-manifest-path /absolute/CANONICAL_V3_BASE28_CURRENT_PAIR_MANIFEST.json \
      --pair-generation-root /absolute/CANONICAL_V3_BASE28_GENERATIONS \
      --pinned-predictions /absolute/selective_edge_predictions_<microstamp>.parquet \
      --prediction-report-json /absolute/ENTRY_CANDIDATE_SELECTIVE_EDGE_<microstamp>.json \
      --out-dir /absolute/immutable/parity/events

Writes an immutable MODEL_NATIVE_SERVE_PARITY_<microstamp>.json event under
--out-dir and exits non-zero on FAIL. No mutable authority mirror is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.model_native_serve_gate_v1 import (
    MODEL_NATIVE_REQUIRED_MODEL_NAME,
    MODEL_NATIVE_REQUIRED_TEST_SPLIT,
    MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION,
    MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION,
    SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS,
    SERVE_PARITY_CALIBRATION_EQUATION,
    SERVE_PARITY_CALIBRATION_TOL,
    SERVE_PARITY_ENV_PINS,
    SERVE_PARITY_FORWARD_FIELD_WIDTHS,
    SERVE_PARITY_FORWARD_HEADS,
    SERVE_PARITY_FORWARD_TOL,
    SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS,
    SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS,
    SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE,
    SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE,
    SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE,
    SERVE_PARITY_FUSION_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_FUSION_INFLUENCE_ABLATION,
    SERVE_PARITY_FUSION_INFLUENCE_EPSILON,
    SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_FUSION_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS,
    SERVE_PARITY_FUSION_DERIVED_ABLATION_SURFACES,
    SERVE_PARITY_FUSION_DERIVED_REFERENCE_INPUTS,
    SERVE_PARITY_FUSION_DERIVED_RELATION_ATOL,
    SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON,
    SERVE_PARITY_FUSION_REFERENCE_AGGREGATION,
    SERVE_PARITY_FUSION_REFERENCE_SPLIT,
    SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON,
    SERVE_PARITY_INDIVIDUAL_INPUT_COMPARISON_SURFACE,
    SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON,
    SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
    SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS,
    SERVE_PARITY_HEAD_VARIATION_EPSILON,
    SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
    SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES,
    SERVE_PARITY_SAMPLE_COUNT,
    SERVE_PARITY_SAMPLING_CONTRACT,
    SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY,
    SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT,
    SERVE_PARITY_SPECIALIST_GATE_MIN_STD,
    SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT,
    SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR,
    SERVE_PARITY_SPECIALIST_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON,
    SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS,
    SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_PARITY_STATE_TOL,
    SERVE_PARITY_UPSTREAM_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
    SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS,
    SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLING_CONTRACT,
    UTC_TIME_COVERAGE_SCHEMA_VERSION,
    build_serve_source_identity,
    serve_gate_event_contract_failures,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_FUSION_INPUTS,
    INPUTS_SHA256 as DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
    INPUT_DIM as DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
    ORDERED_INPUT_LAYOUT as DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT,
    direction_evidence_fusion_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4

from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_ACTION_BY_INDEX,
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
    PUBLIC_FLAT_INDEX,
    PUBLIC_TRADE_INDEX,
    require_model_direction_decision_contract,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    resolve_and_validate_prediction_evidence,
    sha256_file,
)

_TIMESTAMPED_PREDICTION_RE = re.compile(
    r"selective_edge_predictions_\d{8}T\d{12}Z\.parquet"
)

# Runtime parity proves the serving path itself, so it consumes the
# runtime-authoritative prediction event, never the pre-calibration one.
MODEL_NATIVE_REQUIRED_EVIDENCE_STAGE = "runtime_authoritative"
STATE_BLOCKS = ("seq", "snap", "ctx_cont", "ctx_cat")
_FORWARD_LIVE_KEY_OVERRIDES = {
    "public_trade_probability": "p_trade",
    "public_flat_probability": "p_flat_hier",
    "tf_agreement_prob": "tf_agreement_pred",
}
FORWARD_FIELD_MAP = {
    column: _FORWARD_LIVE_KEY_OVERRIDES.get(column, column)
    for column in SERVE_PARITY_FORWARD_FIELD_WIDTHS
}
FORWARD_SCALAR_MAP = {
    column: FORWARD_FIELD_MAP[column]
    for column, width in SERVE_PARITY_FORWARD_FIELD_WIDTHS.items()
    if width == 1
}
FORWARD_VECTOR_COLS = tuple(
    column
    for column, width in SERVE_PARITY_FORWARD_FIELD_WIDTHS.items()
    if width > 1
)
FORWARD_COLS = tuple(FORWARD_FIELD_MAP)
if FORWARD_COLS != SERVE_PARITY_FORWARD_HEADS:
    raise RuntimeError("SERVE_PARITY_FORWARD_HEAD_CONTRACT_MISMATCH")
FULL_STACK_REQUIRED_PREDICTION_COLS = tuple(
    dict.fromkeys(
        column
        for fields in SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS.values()
        for column in fields
    )
) + (
    "specialist_gate",
    "tf_gate",
    "family_tf_cooperation_gate",
    "family_tf_feature_gate",
)
PINNED_REQUIRED_COLS = tuple(
    dict.fromkeys(
        (
            "time",
            "split",
            "model",
            "selection_score_mode",
            "selection_score",
            "pred_direction",
            "trade_side",
            "public_trade_flat_hard_decision",
            *FORWARD_COLS,
            *FULL_STACK_REQUIRED_PREDICTION_COLS,
        )
    )
)
FORBIDDEN_LEGACY_DECISION_COLS = (
    "selection_score_threshold",
    "expected_utility_long_bps",
    "expected_utility_short_bps",
    "anchor_logits",
    "delta_logits",
    "anchor_gate",
)
MODEL_DIRECTION_ACTIONS = MODEL_DIRECTION_ACTION_BY_INDEX


def _apply_exact_env_pins() -> None:
    # Assignment is deliberate: inherited conflicts cannot weaken parity.
    for name, value in SERVE_PARITY_ENV_PINS.items():
        os.environ[name] = value


def _time_coverage_contract(values: object, *, label: str) -> dict[str, object]:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    except Exception as exc:
        raise RuntimeError(f"{label} contains invalid UTC times") from exc
    if index.empty:
        raise RuntimeError(f"{label} time coverage is empty")
    index = index.sort_values()
    if index.has_duplicates:
        raise RuntimeError(f"{label} time coverage contains duplicates")
    utc_ns = np.asarray(index.asi8, dtype="<i8")
    return {
        "schema_version": UTC_TIME_COVERAGE_SCHEMA_VERSION,
        "rows": int(len(index)),
        "first_utc": index[0].isoformat(),
        "last_utc": index[-1].isoformat(),
        "utc_ns_sha256": hashlib.sha256(utc_ns.tobytes()).hexdigest(),
    }


def _deterministic_sample_positions(row_count: int) -> np.ndarray:
    if row_count < SERVE_PARITY_SAMPLE_COUNT:
        raise RuntimeError(
            f"TEST coverage has {row_count} rows; exact parity contract requires at "
            f"least {SERVE_PARITY_SAMPLE_COUNT}"
        )
    numerator = np.arange(SERVE_PARITY_SAMPLE_COUNT, dtype=np.int64) * (row_count - 1)
    positions = numerator // (SERVE_PARITY_SAMPLE_COUNT - 1)
    if len(np.unique(positions)) != SERVE_PARITY_SAMPLE_COUNT:
        raise RuntimeError("deterministic parity sampling produced duplicate positions")
    return positions


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subset_state_rows(
    states: dict[str, object], positions: np.ndarray
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for key in ("seq", "snap", "ctx_cont", "ctx_cat"):
        values = np.asarray(states[key])
        rows[key] = values[positions].copy()
    times = np.asarray(states["times"], dtype=object)
    rows["times"] = times[positions].copy()
    return rows


def _batched_direction_logits(
    adapter: object,
    states: dict[str, object],
    *,
    hook_specialist: str | None = None,
    fusion_input_replacement: tuple[str, dict[str, np.ndarray]] | None = None,
    zero_mtf_key: str | None = None,
    zero_mtf_indices: tuple[str, tuple[int, ...]] | None = None,
    ctx_cat_perturb_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and calibrated logits for one audit-only perturbation batch."""

    import torch

    model = adapter._model
    if model is None:
        raise RuntimeError("specialist influence audit model is unavailable")
    device = adapter.device
    seq_t = torch.from_numpy(np.asarray(states["seq"], dtype=np.float32)).to(device)
    snap_t = torch.from_numpy(np.asarray(states["snap"], dtype=np.float32)).to(device)
    ctx_cont_t = torch.from_numpy(
        np.asarray(states["ctx_cont"], dtype=np.float32)
    ).to(device)
    ctx_cat_t = torch.from_numpy(np.asarray(states["ctx_cat"], dtype=np.int64)).to(
        device
    )
    if ctx_cat_perturb_index is not None:
        embeddings = getattr(model, "ctx_cat_embeddings", None)
        if (
            embeddings is None
            or not 0 <= ctx_cat_perturb_index < len(embeddings)
            or int(embeddings[ctx_cat_perturb_index].num_embeddings) < 2
        ):
            raise RuntimeError(
                "categorical influence perturbation lacks a valid embedding domain: "
                f"index={ctx_cat_perturb_index}"
            )
        ctx_cat_t = ctx_cat_t.clone()
        domain = int(embeddings[ctx_cat_perturb_index].num_embeddings)
        ctx_cat_t[:, ctx_cat_perturb_index] = (
            ctx_cat_t[:, ctx_cat_perturb_index] + 1
        ) % domain
    per_row_mtf = [
        adapter._multi_tf_window_tensors(pd.Timestamp(ts))
        for ts in np.asarray(states["times"], dtype=object)
    ]
    if not per_row_mtf:
        raise RuntimeError("specialist influence audit state subset is empty")
    mtf_kwargs = {
        key: torch.cat([row[key] for row in per_row_mtf], dim=0)
        for key in per_row_mtf[0]
    }
    if zero_mtf_key is not None:
        if zero_mtf_key not in mtf_kwargs:
            raise RuntimeError(f"multi-TF audit key is missing: {zero_mtf_key}")
        mtf_kwargs[zero_mtf_key] = torch.zeros_like(mtf_kwargs[zero_mtf_key])
    if zero_mtf_indices is not None:
        key, indices = zero_mtf_indices
        if key not in mtf_kwargs:
            raise RuntimeError(f"multi-TF family audit key is missing: {key}")
        if (
            not indices
            or min(indices) < 0
            or max(indices) >= int(mtf_kwargs[key].shape[-1])
        ):
            raise RuntimeError(
                f"multi-TF family audit indices are invalid: {key}={indices}"
            )
        mtf_kwargs[key] = mtf_kwargs[key].clone()
        mtf_kwargs[key][..., list(indices)] = 0.0
    hooks = []
    if hook_specialist is not None:
        encoders = getattr(model, "specialist_encoder", None)
        if encoders is None or hook_specialist not in encoders:
            raise RuntimeError(
                f"specialist encoder hook target is missing: {hook_specialist}"
            )

        def _zero_encoder_output(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"specialist encoder {hook_specialist} emitted a non-tensor"
                )
            return torch.zeros_like(output)

        hooks.append(
            encoders[hook_specialist].register_forward_hook(_zero_encoder_output)
        )
    if fusion_input_replacement is not None:
        replacement_name, replacement_means = fusion_input_replacement
        fusion_norm = getattr(model, "evidence_fusion_norm", None)
        if fusion_norm is None:
            raise RuntimeError("direction evidence fusion LayerNorm is missing")
        layouts = {
            str(row["name"]): row
            for row in DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT
        }
        if replacement_name not in layouts:
            raise RuntimeError(
                f"direction evidence fusion replacement is unknown: {replacement_name}"
            )
        required_names = (
            SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS
            if replacement_name in SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS
            else (replacement_name,)
        )
        replacement_tensors: dict[str, object] = {}
        for name in required_names:
            layout = layouts[name]
            replacement_array = np.asarray(
                replacement_means[name], dtype=np.float32
            ).reshape(-1)
            if replacement_array.shape != (int(layout["width"]),):
                raise RuntimeError(
                    "direction evidence fusion replacement width mismatch: "
                    f"{name}={replacement_array.shape}"
                )
            replacement_tensors[name] = torch.from_numpy(
                replacement_array
            ).to(device)

        def _replace_fusion_slice(_module, inputs):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("direction evidence fusion input hook shape invalid")
            value = inputs[0]
            if value.ndim != 2 or int(value.shape[1]) != DIRECTION_EVIDENCE_FUSION_INPUT_DIM:
                raise RuntimeError(
                    "direction evidence fusion hook width mismatch: "
                    f"got={tuple(value.shape)}"
                )
            replaced = value.clone()
            if replacement_name not in SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS:
                layout = layouts[replacement_name]
                replaced[:, int(layout["start"]):int(layout["stop"])] = (
                    replacement_tensors[replacement_name].to(
                        device=value.device, dtype=value.dtype
                    ).reshape(1, -1)
                )
            else:
                q_layout = layouts["action_value"]
                v_layout = layouts["expectile_value"]
                a_layout = layouts["action_advantage"]
                q = value[:, int(q_layout["start"]):int(q_layout["stop"])]
                v = value[:, int(v_layout["start"]):int(v_layout["stop"])]
                if replacement_name in ("action_value", "action_advantage"):
                    q = replacement_tensors["action_value"].to(
                        device=value.device, dtype=value.dtype
                    ).reshape(1, -1).expand_as(q)
                if replacement_name in ("expectile_value", "action_advantage"):
                    v = replacement_tensors["expectile_value"].to(
                        device=value.device, dtype=value.dtype
                    ).reshape(1, -1).expand_as(v)
                advantage = (q.reshape(-1, 3, 3) - v.unsqueeze(1)).reshape(
                    -1, 9
                )
                replaced[:, int(q_layout["start"]):int(q_layout["stop"])] = q
                replaced[:, int(v_layout["start"]):int(v_layout["stop"])] = v
                replaced[:, int(a_layout["start"]):int(a_layout["stop"])] = (
                    advantage
                )
            return (replaced,)

        hooks.append(fusion_norm.register_forward_pre_hook(_replace_fusion_slice))
    try:
        with torch.no_grad():
            out = model(
                seq_t,
                snap_t,
                ctx_cat=ctx_cat_t,
                ctx_cont=ctx_cont_t,
                **mtf_kwargs,
            )
            raw_t = out.get("raw_direction_logits")
            final_t = out.get("direction_logits")
            if not torch.is_tensor(raw_t) or not torch.is_tensor(final_t):
                raise RuntimeError(
                    "decision influence audit lacks raw/final direction logits"
                )
            raw_logits = raw_t.detach().cpu().to(torch.float64).numpy()
            final_logits = final_t.detach().cpu().to(torch.float64).numpy()
    finally:
        for hook in reversed(hooks):
            hook.remove()
    expected = (len(np.asarray(states["times"])), 3)
    for name, logits in (("raw_direction_logits", raw_logits), ("direction_logits", final_logits)):
        if logits.shape != expected or not np.isfinite(logits).all():
            raise RuntimeError(
                f"decision influence {name} must be finite shape {expected}; "
                f"got {logits.shape}"
            )
    return raw_logits, final_logits


def _specialist_model_input_indices(
    adapter: object,
) -> tuple[dict[str, list[int]], bool, bool, list[str]]:
    failures: list[str] = []
    model = adapter._model
    metadata = adapter._meta
    specialists = tuple(MODEL_NATIVE_REQUIRED_SPECIALISTS)
    metadata_specialist = metadata.get("specialist_fusion")
    metadata_indices = (
        metadata_specialist.get("input_indices")
        if isinstance(metadata_specialist, dict)
        else None
    )
    if not isinstance(metadata_indices, dict):
        metadata_indices = {}
        failures.append("bundle specialist_fusion.input_indices is missing")
    if tuple(getattr(model, "_specialist_names", ())) != specialists:
        failures.append("model specialist name/order mismatch")
    encoders = getattr(model, "specialist_encoder", None)
    if encoders is None or tuple(encoders) != specialists:
        failures.append("model specialist encoder set/order mismatch")
    projections = getattr(model, "specialist_proj", None)
    if projections is None or tuple(projections) != specialists:
        failures.append("model specialist projection set/order mismatch")

    indices: dict[str, list[int]] = {}
    buffers_valid = True
    for specialist in specialists:
        value = getattr(model, f"specialist_idx_{specialist}", None)
        try:
            row = [int(item) for item in value.detach().cpu().tolist()]
        except Exception:
            row = []
        if (
            not row
            or row != sorted(row)
            or len(row) != len(set(row))
            or min(row, default=-1) < 0
            or max(row, default=-1) >= int(np.asarray(adapter._meta["ordered_signal_names"]).size)
        ):
            failures.append(f"model specialist index buffer invalid: {specialist}")
            buffers_valid = False
        if projections is not None and specialist in projections:
            if int(projections[specialist].in_features) != len(row):
                failures.append(
                    f"model specialist projection width mismatch: {specialist}"
                )
                buffers_valid = False
        indices[specialist] = row
    metadata_exact = set(metadata_indices) == set(specialists) and all(
        metadata_indices.get(name) == indices[name] for name in specialists
    )
    if not metadata_exact:
        failures.append("bundle metadata specialist indices differ from model buffers")
    return indices, metadata_exact, buffers_valid, failures


def _specialist_decision_influence_contract(
    *,
    adapter: object,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
) -> dict[str, object]:
    """Prove both evidence-family and isolated specialist decision influence."""

    positions = np.asarray(
        SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS, dtype=np.int64
    )
    failures: list[str] = []
    if len(np.asarray(states["times"])) != SERVE_PARITY_SAMPLE_COUNT:
        failures.append("specialist audit did not receive the exact parity state set")
    if len(parity_targets) != SERVE_PARITY_SAMPLE_COUNT:
        failures.append("specialist audit did not receive the exact parity target set")
    subset = _subset_state_rows(states, positions)
    sampled_targets = pd.DatetimeIndex(parity_targets[positions])
    state_times = pd.DatetimeIndex(pd.to_datetime(subset["times"], utc=True))
    if not state_times.equals(sampled_targets):
        failures.append("specialist audit state times differ from deterministic targets")

    indices, metadata_exact, buffers_valid, index_failures = (
        _specialist_model_input_indices(adapter)
    )
    failures.extend(index_failures)
    try:
        baseline_raw, baseline_final = _batched_direction_logits(adapter, subset)
        baseline_raw_centered = baseline_raw - baseline_raw.mean(axis=1, keepdims=True)
        baseline_final_centered = (
            baseline_final - baseline_final.mean(axis=1, keepdims=True)
        )
    except RuntimeError as exc:
        failures.append(str(exc))
        baseline_raw_centered = np.zeros(
            (SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT, 3),
            dtype=np.float64,
        )
        baseline_final_centered = np.zeros(
            (SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT, 3),
            dtype=np.float64,
        )

    method_reports: dict[str, object] = {}
    for method in SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS:
        method_failures: list[str] = []
        specialist_reports: dict[str, object] = {}
        for specialist in MODEL_NATIVE_REQUIRED_SPECIALISTS:
            specialist_failures: list[str] = []
            try:
                if method == "input_family_mask":
                    ablated_states = _subset_state_rows(
                        subset,
                        np.arange(
                            SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
                            dtype=np.int64,
                        ),
                    )
                    row_indices = indices[specialist]
                    np.asarray(ablated_states["seq"])[:, :, row_indices] = 0.0
                    np.asarray(ablated_states["snap"])[:, row_indices] = 0.0
                    ablated_raw, ablated_final = _batched_direction_logits(
                        adapter, ablated_states
                    )
                    target = f"signal_indices:{specialist}"
                else:
                    ablated_raw, ablated_final = _batched_direction_logits(
                        adapter,
                        subset,
                        hook_specialist=specialist,
                    )
                    target = f"model.specialist_encoder.{specialist}"
                raw_centered = ablated_raw - ablated_raw.mean(axis=1, keepdims=True)
                final_centered = (
                    ablated_final - ablated_final.mean(axis=1, keepdims=True)
                )
                raw_per_row_delta = np.max(
                    np.abs(raw_centered - baseline_raw_centered), axis=1
                )
                final_per_row_delta = np.max(
                    np.abs(final_centered - baseline_final_centered), axis=1
                )
                raw_max_delta = float(np.max(raw_per_row_delta))
                final_max_delta = float(np.max(final_per_row_delta))
                raw_changed_rows = int(
                    np.count_nonzero(
                        raw_per_row_delta
                        > SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON
                    )
                )
                final_changed_rows = int(
                    np.count_nonzero(
                        final_per_row_delta
                        > SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON
                    )
                )
                if raw_max_delta <= SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON:
                    specialist_failures.append(
                        "class-centered raw logits did not move > epsilon"
                    )
                if final_max_delta <= SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON:
                    specialist_failures.append(
                        "class-centered final calibrated logits did not move > epsilon"
                    )
                if raw_changed_rows < SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS:
                    specialist_failures.append(
                        f"raw_changed_rows={raw_changed_rows} below "
                        f"{SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS}"
                    )
                if final_changed_rows < SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS:
                    specialist_failures.append(
                        f"final_changed_rows={final_changed_rows} below "
                        f"{SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS}"
                    )
            except Exception as exc:
                raw_max_delta = 0.0
                final_max_delta = 0.0
                raw_changed_rows = 0
                final_changed_rows = 0
                target = (
                    f"signal_indices:{specialist}"
                    if method == "input_family_mask"
                    else f"model.specialist_encoder.{specialist}"
                )
                specialist_failures.append(f"ablation failed: {exc}")
            row = {
                "decision": "PASS" if not specialist_failures else "FAIL",
                "failures": specialist_failures,
                "target": target,
                "input_indices_sha256": _canonical_sha256(indices[specialist]),
                "max_abs_class_centered_raw_logit_delta": raw_max_delta,
                "raw_changed_rows": raw_changed_rows,
                "max_abs_class_centered_logit_delta": final_max_delta,
                "changed_rows": final_changed_rows,
                "total_rows": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
            }
            specialist_reports[specialist] = row
            method_failures.extend(
                f"{specialist}: {failure}" for failure in specialist_failures
            )
        method_reports[method] = {
            "decision": "PASS" if not method_failures else "FAIL",
            "failures": method_failures,
            "ablation_surface": (
                "seq_and_snap_exact_specialist_input_indices"
                if method == "input_family_mask"
                else "specialist_encoder_output_zero_hook"
            ),
            "specialists": specialist_reports,
        }
        failures.extend(f"{method}: {failure}" for failure in method_failures)

    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": _time_coverage_contract(
            sampled_targets,
            label="specialist influence sampled TEST parity positions",
        ),
        "comparison_surface": SERVE_PARITY_SPECIALIST_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS,
        "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "specialist_input_indices": indices,
        "specialist_input_indices_sha256": _canonical_sha256(indices),
        "model_metadata_indices_exact_match": metadata_exact,
        "model_buffer_indices_exact_match": buffers_valid,
        "methods": method_reports,
    }


def _decision_influence_subset(
    *,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
    positions: tuple[int, ...],
    expected_count: int,
    audit_name: str,
) -> tuple[dict[str, object], pd.DatetimeIndex, list[str]]:
    """Select and time-bind one exact candidate-specific influence sample."""

    failures: list[str] = []
    if len(positions) != expected_count:
        failures.append(
            f"{audit_name} sample position count={len(positions)} expected={expected_count}"
        )
    if len(np.asarray(states["times"])) != SERVE_PARITY_SAMPLE_COUNT:
        failures.append(f"{audit_name} did not receive the exact parity state set")
    if len(parity_targets) != SERVE_PARITY_SAMPLE_COUNT:
        failures.append(f"{audit_name} did not receive the exact parity target set")
    position_array = np.asarray(positions, dtype=np.int64)
    subset = _subset_state_rows(states, position_array)
    sampled_targets = pd.DatetimeIndex(parity_targets[position_array])
    state_times = pd.DatetimeIndex(pd.to_datetime(subset["times"], utc=True))
    if not state_times.equals(sampled_targets):
        failures.append(
            f"{audit_name} state times differ from deterministic target positions"
        )
    return subset, sampled_targets, failures


def _class_centered_delta_metrics(
    *,
    baseline: np.ndarray,
    ablated: np.ndarray,
    epsilon: float,
) -> tuple[float, int]:
    baseline_centered = baseline - baseline.mean(axis=1, keepdims=True)
    ablated_centered = ablated - ablated.mean(axis=1, keepdims=True)
    per_row = np.max(np.abs(ablated_centered - baseline_centered), axis=1)
    return float(np.max(per_row)), int(np.count_nonzero(per_row > epsilon))


def _batched_direction_input_margin_gradients(
    adapter: object,
    states: dict[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    """Return max absolute pairwise-logit gradients for every numeric input."""

    import torch

    model = adapter._model
    if model is None:
        raise RuntimeError("individual input influence model is unavailable")
    device = adapter.device
    seq_t = (
        torch.from_numpy(np.asarray(states["seq"], dtype=np.float32))
        .to(device)
        .requires_grad_(True)
    )
    snap_t = (
        torch.from_numpy(np.asarray(states["snap"], dtype=np.float32))
        .to(device)
        .requires_grad_(True)
    )
    ctx_cont_t = (
        torch.from_numpy(np.asarray(states["ctx_cont"], dtype=np.float32))
        .to(device)
        .requires_grad_(True)
    )
    ctx_cat_t = torch.from_numpy(
        np.asarray(states["ctx_cat"], dtype=np.int64)
    ).to(device)
    per_row_mtf = [
        adapter._multi_tf_window_tensors(pd.Timestamp(ts))
        for ts in np.asarray(states["times"], dtype=object)
    ]
    if not per_row_mtf:
        raise RuntimeError("individual input influence state subset is empty")
    mtf_kwargs = {
        key: torch.cat([row[key] for row in per_row_mtf], dim=0)
        .detach()
        .clone()
        .requires_grad_(True)
        for key in per_row_mtf[0]
    }
    out = model(
        seq_t,
        snap_t,
        ctx_cat=ctx_cat_t,
        ctx_cont=ctx_cont_t,
        **mtf_kwargs,
    )
    raw = out.get("raw_direction_logits")
    final = out.get("direction_logits")
    if (
        not torch.is_tensor(raw)
        or not torch.is_tensor(final)
        or tuple(raw.shape) != (len(np.asarray(states["times"])), 3)
        or tuple(final.shape) != tuple(raw.shape)
    ):
        raise RuntimeError("individual input influence direction logits are invalid")

    input_names = (
        "seq",
        "snap",
        "ctx_cont",
        *tuple(mtf_kwargs),
    )
    input_tensors = (
        seq_t,
        snap_t,
        ctx_cont_t,
        *tuple(mtf_kwargs.values()),
    )

    def _surface(logits: object) -> dict[str, np.ndarray]:
        maxima = {
            name: np.zeros(int(tensor.shape[-1]), dtype=np.float64)
            for name, tensor in zip(input_names, input_tensors)
        }
        for left, right in ((0, 1), (0, 2), (1, 2)):
            objective = (logits[:, left] - logits[:, right]).sum()
            gradients = torch.autograd.grad(
                objective,
                input_tensors,
                retain_graph=True,
                allow_unused=True,
            )
            for name, tensor, gradient in zip(
                input_names, input_tensors, gradients
            ):
                if gradient is None:
                    continue
                values = gradient.detach().abs()
                reduce_dims = tuple(range(values.ndim - 1))
                reduced = (
                    values.amax(dim=reduce_dims)
                    if reduce_dims
                    else values
                )
                array = reduced.cpu().to(torch.float64).numpy().reshape(-1)
                if array.shape != maxima[name].shape or not np.isfinite(array).all():
                    raise RuntimeError(
                        "individual input influence gradient shape/nonfinite: "
                        f"{name}={array.shape}"
                    )
                maxima[name] = np.maximum(maxima[name], array)
        return maxima

    raw_gradients = _surface(raw)
    final_gradients = _surface(final)
    return {
        "raw": {
            "seq_signal": raw_gradients.pop("seq"),
            "snap_signal": raw_gradients.pop("snap"),
            **raw_gradients,
        },
        "final": {
            "seq_signal": final_gradients.pop("seq"),
            "snap_signal": final_gradients.pop("snap"),
            **final_gradients,
        },
    }


def _individual_input_decision_influence_contract(
    *,
    adapter: object,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
) -> dict[str, object]:
    """Prove every retained numeric field and categorical input reaches margins."""

    subset, sampled_targets, failures = _decision_influence_subset(
        states=states,
        parity_targets=parity_targets,
        positions=SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS,
        expected_count=SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        audit_name="individual input influence audit",
    )
    signal_names = [str(item) for item in adapter._meta.get(
        "ordered_signal_names", ()
    )]
    ctx_cont_names = [str(item) for item in adapter._meta.get(
        "ordered_ctx_cont_names", ()
    )]
    ctx_cat_names = [str(item) for item in adapter._meta.get(
        "ordered_ctx_cat_names", ()
    )]
    if (
        len(signal_names) != MODEL_NATIVE_SIGNAL_DIM
        or len(signal_names) != len(set(signal_names))
    ):
        failures.append("bundle ordered_signal_names is not exact unique seq513")
    if ctx_cont_names != list(MODEL_NATIVE_CTX_CONT_FIELDS):
        failures.append("bundle ordered_ctx_cont_names contract mismatch")
    if ctx_cat_names != list(MODEL_NATIVE_CTX_CAT_FIELDS):
        failures.append("bundle ordered_ctx_cat_names contract mismatch")

    expected_numeric_tokens = {
        "seq_signal": signal_names,
        "snap_signal": signal_names,
        "ctx_cont": ctx_cont_names,
        **{
            f"seq_{timeframe.lower()}": [
                f"{timeframe.lower()}:{name}"
                for name in MULTI_TF_PER_BAR_FEATURES_V4
            ]
            for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
        },
    }
    numeric_metrics: dict[str, object] = {}
    try:
        gradient_surfaces = _batched_direction_input_margin_gradients(
            adapter, subset
        )
    except Exception as exc:
        failures.append(f"input margin gradient execution failed: {exc}")
        gradient_surfaces = {
            surface: {
                key: np.zeros(len(tokens), dtype=np.float64)
                for key, tokens in expected_numeric_tokens.items()
            }
            for surface in ("raw", "final")
        }
    for surface_key, tokens in expected_numeric_tokens.items():
        raw_values = np.asarray(
            gradient_surfaces["raw"].get(surface_key, ()), dtype=np.float64
        ).reshape(-1)
        final_values = np.asarray(
            gradient_surfaces["final"].get(surface_key, ()), dtype=np.float64
        ).reshape(-1)
        if raw_values.shape != (len(tokens),) or final_values.shape != (
            len(tokens),
        ):
            failures.append(f"{surface_key}: gradient width mismatch")
            raw_values = np.zeros(len(tokens), dtype=np.float64)
            final_values = np.zeros(len(tokens), dtype=np.float64)
        metrics: dict[str, object] = {}
        for index, token in enumerate(tokens):
            raw_value = float(raw_values[index])
            final_value = float(final_values[index])
            row_failures: list[str] = []
            if (
                not np.isfinite(raw_value)
                or raw_value
                <= SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON
            ):
                row_failures.append("raw class-margin gradient is dead")
            if (
                not np.isfinite(final_value)
                or final_value
                <= SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON
            ):
                row_failures.append("final class-margin gradient is dead")
            metrics[token] = {
                "decision": "PASS" if not row_failures else "FAIL",
                "failures": row_failures,
                "max_abs_raw_class_margin_gradient": raw_value,
                "max_abs_final_class_margin_gradient": final_value,
            }
            failures.extend(
                f"{surface_key}/{token}: {failure}"
                for failure in row_failures
            )
        numeric_metrics[surface_key] = {
            "tokens": tokens,
            "metrics": metrics,
        }

    categorical_metrics: dict[str, object] = {}
    try:
        baseline_raw, baseline_final = _batched_direction_logits(
            adapter, subset
        )
    except Exception as exc:
        failures.append(f"categorical baseline forward failed: {exc}")
        baseline_raw = np.zeros(
            (SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT, 3),
            dtype=np.float64,
        )
        baseline_final = np.zeros_like(baseline_raw)
    for index, name in enumerate(ctx_cat_names):
        row_failures: list[str] = []
        try:
            perturbed_raw, perturbed_final = _batched_direction_logits(
                adapter,
                subset,
                ctx_cat_perturb_index=index,
            )
            raw_delta, raw_changed = _class_centered_delta_metrics(
                baseline=baseline_raw,
                ablated=perturbed_raw,
                epsilon=SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON,
            )
            final_delta, final_changed = _class_centered_delta_metrics(
                baseline=baseline_final,
                ablated=perturbed_final,
                epsilon=SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON,
            )
            if raw_changed < 1:
                row_failures.append("raw categorical counterfactual is dead")
            if final_changed < 1:
                row_failures.append("final categorical counterfactual is dead")
        except Exception as exc:
            raw_delta = 0.0
            raw_changed = 0
            final_delta = 0.0
            final_changed = 0
            row_failures.append(f"categorical counterfactual failed: {exc}")
        categorical_metrics[name] = {
            "decision": "PASS" if not row_failures else "FAIL",
            "failures": row_failures,
            "counterfactual": "next_valid_embedding_category_modulo_domain",
            "max_abs_class_centered_raw_logit_delta": raw_delta,
            "raw_changed_rows": raw_changed,
            "max_abs_class_centered_logit_delta": final_delta,
            "changed_rows": final_changed,
            "total_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        }
        failures.extend(f"ctx_cat/{name}: {failure}" for failure in row_failures)

    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        "sample_positions": list(
            SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS
        ),
        "sampled_test_coverage": _time_coverage_contract(
            sampled_targets,
            label="individual input influence sampled TEST positions",
        ),
        "comparison_surface": (
            SERVE_PARITY_INDIVIDUAL_INPUT_COMPARISON_SURFACE
        ),
        "gradient_epsilon": (
            SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON
        ),
        "categorical_delta_epsilon": (
            SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON
        ),
        "numeric_input_count": sum(
            len(tokens) for tokens in expected_numeric_tokens.values()
        ),
        "categorical_input_count": len(ctx_cat_names),
        "signal_names_sha256": _canonical_sha256(signal_names),
        "ctx_cont_names_sha256": _canonical_sha256(ctx_cont_names),
        "ctx_cat_names_sha256": _canonical_sha256(ctx_cat_names),
        "numeric": numeric_metrics,
        "categorical": categorical_metrics,
    }


def _multi_tf_decision_influence_contract(
    *,
    adapter: object,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
) -> dict[str, object]:
    """Prove every retained timeframe moves candidate raw and final margins."""

    subset, sampled_targets, failures = _decision_influence_subset(
        states=states,
        parity_targets=parity_targets,
        positions=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS,
        expected_count=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        audit_name="multi-TF influence audit",
    )
    try:
        baseline_raw, baseline_final = _batched_direction_logits(adapter, subset)
    except Exception as exc:
        failures.append(f"baseline forward failed: {exc}")
        baseline_raw = np.zeros(
            (SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT, 3), dtype=np.float64
        )
        baseline_final = np.zeros(
            (SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT, 3), dtype=np.float64
        )

    timeframe_reports: dict[str, object] = {}
    for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES:
        row_failures: list[str] = []
        try:
            ablated_raw, ablated_final = _batched_direction_logits(
                adapter,
                subset,
                zero_mtf_key=f"seq_{timeframe.lower()}",
            )
            raw_max_delta, raw_changed_rows = _class_centered_delta_metrics(
                baseline=baseline_raw,
                ablated=ablated_raw,
                epsilon=SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
            )
            max_delta, changed_rows = _class_centered_delta_metrics(
                baseline=baseline_final,
                ablated=ablated_final,
                epsilon=SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
            )
            if raw_max_delta <= SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON:
                row_failures.append("class-centered raw logits did not move > epsilon")
            if raw_changed_rows < SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS:
                row_failures.append(
                    f"raw_changed_rows={raw_changed_rows} below "
                    f"{SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS}"
                )
            if max_delta <= SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON:
                row_failures.append(
                    "class-centered final calibrated logits did not move > epsilon"
                )
            if changed_rows < SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS:
                row_failures.append(
                    f"changed_rows={changed_rows} below "
                    f"{SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS}"
                )
        except Exception as exc:
            raw_max_delta = 0.0
            raw_changed_rows = 0
            max_delta = 0.0
            changed_rows = 0
            row_failures.append(f"zero ablation failed: {exc}")
        timeframe_reports[timeframe] = {
            "decision": "PASS" if not row_failures else "FAIL",
            "failures": row_failures,
            "target": f"model.input.seq_{timeframe.lower()}",
            "ablation_surface": "full_tensor_zero_mask",
            "max_abs_class_centered_raw_logit_delta": raw_max_delta,
            "raw_changed_rows": raw_changed_rows,
            "max_abs_class_centered_logit_delta": max_delta,
            "changed_rows": changed_rows,
            "total_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        }
        failures.extend(f"{timeframe}: {failure}" for failure in row_failures)

    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": _time_coverage_contract(
            sampled_targets,
            label="multi-TF influence sampled TEST parity positions",
        ),
        "comparison_surface": SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
        "timeframes": list(SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES),
        "ablation": "candidate_specific_full_tensor_zero_ablation_v1",
        "metrics": timeframe_reports,
    }


def _family_tf_decision_influence_contract(
    *,
    adapter: object,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
) -> dict[str, object]:
    """Prove every one of the 5×8 family routes changes direction margins."""

    subset, sampled_targets, failures = _decision_influence_subset(
        states=states,
        parity_targets=parity_targets,
        positions=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS,
        expected_count=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        audit_name="family×timeframe influence audit",
    )
    routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    try:
        baseline_raw, baseline_final = _batched_direction_logits(adapter, subset)
    except Exception as exc:
        failures.append(f"baseline forward failed: {exc}")
        baseline_raw = np.zeros(
            (SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT, 3),
            dtype=np.float64,
        )
        baseline_final = np.zeros_like(baseline_raw)

    metrics: dict[str, object] = {}
    for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES:
        key = f"seq_{timeframe.lower()}"
        for specialist, indices in routing.items():
            token = f"{timeframe.lower()}:{specialist}"
            row_failures: list[str] = []
            try:
                ablated_raw, ablated_final = _batched_direction_logits(
                    adapter,
                    subset,
                    zero_mtf_indices=(key, tuple(indices)),
                )
                raw_max_delta, raw_changed_rows = _class_centered_delta_metrics(
                    baseline=baseline_raw,
                    ablated=ablated_raw,
                    epsilon=SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
                )
                max_delta, changed_rows = _class_centered_delta_metrics(
                    baseline=baseline_final,
                    ablated=ablated_final,
                    epsilon=SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
                )
                if raw_max_delta <= SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON:
                    row_failures.append(
                        "class-centered raw logits did not move > epsilon"
                    )
                if (
                    raw_changed_rows
                    < SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS
                ):
                    row_failures.append(
                        f"raw_changed_rows={raw_changed_rows} below "
                        f"{SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS}"
                    )
                if max_delta <= SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON:
                    row_failures.append(
                        "class-centered final calibrated logits did not move > epsilon"
                    )
                if changed_rows < SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS:
                    row_failures.append(
                        f"changed_rows={changed_rows} below "
                        f"{SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS}"
                    )
            except Exception as exc:
                raw_max_delta = 0.0
                raw_changed_rows = 0
                max_delta = 0.0
                changed_rows = 0
                row_failures.append(f"family zero ablation failed: {exc}")
            metrics[token] = {
                "decision": "PASS" if not row_failures else "FAIL",
                "failures": row_failures,
                "target": (
                    f"model.input.{key}[{','.join(str(i) for i in indices)}]"
                ),
                "ablation_surface": "exact_family_feature_indices_zero_mask",
                "max_abs_class_centered_raw_logit_delta": raw_max_delta,
                "raw_changed_rows": raw_changed_rows,
                "max_abs_class_centered_logit_delta": max_delta,
                "changed_rows": changed_rows,
                "total_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
            }
            failures.extend(
                f"{token}: {failure}" for failure in row_failures
            )
    tokens = [
        f"{timeframe.lower()}:{specialist}"
        for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
        for specialist in routing
    ]
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(
            SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS
        ),
        "sampled_test_coverage": _time_coverage_contract(
            sampled_targets,
            label="family×timeframe influence sampled TEST parity positions",
        ),
        "comparison_surface": SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
        "family_timeframe_tokens": tokens,
        "ablation": "candidate_specific_family_tensor_index_zero_ablation_v1",
        "metrics": metrics,
    }


def _upstream_context_decision_influence_contract(
    *,
    adapter: object,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
) -> dict[str, object]:
    """Prove continuous and categorical context moves raw and final margins."""

    subset, sampled_targets, failures = _decision_influence_subset(
        states=states,
        parity_targets=parity_targets,
        positions=SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS,
        expected_count=SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
        audit_name="upstream context influence audit",
    )
    try:
        baseline_raw, baseline_final = _batched_direction_logits(adapter, subset)
    except Exception as exc:
        failures.append(f"baseline forward failed: {exc}")
        baseline_raw = np.zeros(
            (SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT, 3), dtype=np.float64
        )
        baseline_final = np.zeros(
            (SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT, 3), dtype=np.float64
        )

    method_reports: dict[str, object] = {}
    for method in SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS:
        row_failures: list[str] = []
        target = "ctx_cont" if method == "ctx_cont_zero_mask" else "ctx_cat"
        try:
            ablated_states = _subset_state_rows(
                subset,
                np.arange(SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT, dtype=np.int64),
            )
            np.asarray(ablated_states[target])[...] = 0
            ablated_raw, ablated_final = _batched_direction_logits(
                adapter, ablated_states
            )
            raw_max_delta, raw_changed_rows = _class_centered_delta_metrics(
                baseline=baseline_raw,
                ablated=ablated_raw,
                epsilon=SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
            )
            max_delta, changed_rows = _class_centered_delta_metrics(
                baseline=baseline_final,
                ablated=ablated_final,
                epsilon=SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
            )
            if raw_max_delta <= SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON:
                row_failures.append("class-centered raw logits did not move > epsilon")
            if raw_changed_rows < SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS:
                row_failures.append(
                    f"raw_changed_rows={raw_changed_rows} below "
                    f"{SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS}"
                )
            if max_delta <= SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON:
                row_failures.append(
                    "class-centered final calibrated logits did not move > epsilon"
                )
            if changed_rows < SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS:
                row_failures.append(
                    f"changed_rows={changed_rows} below "
                    f"{SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS}"
                )
        except Exception as exc:
            raw_max_delta = 0.0
            raw_changed_rows = 0
            max_delta = 0.0
            changed_rows = 0
            row_failures.append(f"zero mask failed: {exc}")
        method_reports[method] = {
            "decision": "PASS" if not row_failures else "FAIL",
            "failures": row_failures,
            "target": f"model.input.{target}",
            "ablation_surface": "full_tensor_zero_mask",
            "max_abs_class_centered_raw_logit_delta": raw_max_delta,
            "raw_changed_rows": raw_changed_rows,
            "max_abs_class_centered_logit_delta": max_delta,
            "changed_rows": changed_rows,
            "total_rows": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
        }
        failures.extend(f"{method}: {failure}" for failure in row_failures)

    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": _time_coverage_contract(
            sampled_targets,
            label="upstream context influence sampled TEST parity positions",
        ),
        "comparison_surface": SERVE_PARITY_UPSTREAM_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS,
        "methods": list(SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS),
        "metrics": method_reports,
    }


def _finite_vector(value: object, *, name: str, size: int, row_label: object) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"pinned {name} is not numeric at {row_label}") from exc
    if vector.shape != (size,):
        raise RuntimeError(
            f"pinned {name} must have shape ({size},) at {row_label}; got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise RuntimeError(f"pinned {name} contains non-finite values at {row_label}")
    return vector


def _numeric_prediction_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"prediction evidence missing scalar column {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise RuntimeError(
            f"prediction evidence {column} must be finite shape ({len(frame)},)"
        )
    return values.reshape(-1, 1)


def _vector_prediction_column(
    frame: pd.DataFrame,
    column: str,
    width: int,
) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"prediction evidence missing vector column {column}")
    try:
        values = np.stack(
            [np.asarray(row, dtype=np.float64).reshape(-1) for row in frame[column]]
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"prediction evidence {column} is not a dense vector") from exc
    if values.shape != (len(frame), width) or not np.isfinite(values).all():
        raise RuntimeError(
            f"prediction evidence {column} must be finite shape ({len(frame)},{width})"
        )
    return values


def _validate_fusion_reference_prediction_contract(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return the immutable candidate VAL rows used only as ablation references."""

    required = {
        "time",
        "split",
        "model",
        *(name for name, _width in DIRECTION_EVIDENCE_FUSION_INPUTS),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "fresh prediction evidence lacks candidate VAL fusion reference columns: "
            + ",".join(missing)
        )
    candidate_val = frame.loc[
        (frame["model"].astype(str) == MODEL_NATIVE_REQUIRED_MODEL_NAME)
        & (frame["split"].astype(str) == SERVE_PARITY_FUSION_REFERENCE_SPLIT)
    ].copy()
    if candidate_val.empty:
        raise RuntimeError(
            "fresh prediction evidence contains no candidate VAL rows for fusion influence"
        )
    try:
        candidate_val["time"] = pd.to_datetime(
            candidate_val["time"], utc=True, errors="raise"
        )
    except Exception as exc:
        raise RuntimeError(
            "candidate VAL fusion reference contains invalid UTC times"
        ) from exc
    if candidate_val["time"].duplicated().any():
        raise RuntimeError("candidate VAL fusion reference contains duplicate times")
    for name, width in DIRECTION_EVIDENCE_FUSION_INPUTS:
        if width == 1:
            _numeric_prediction_column(candidate_val, name)
        else:
            _vector_prediction_column(candidate_val, name, width)
    _require_action_value_manifold(candidate_val)
    return candidate_val.set_index("time").sort_index()


def _require_action_value_manifold(frame: pd.DataFrame) -> float:
    """Reject impossible Q/V/A evidence states before any fusion audit."""

    action_value = _vector_prediction_column(frame, "action_value", 9)
    expectile_value = _vector_prediction_column(frame, "expectile_value", 3)
    action_advantage = _vector_prediction_column(
        frame, "action_advantage", 9
    )
    expected = (
        action_value.reshape(len(frame), 3, 3)
        - expectile_value[:, None, :]
    ).reshape(len(frame), 9)
    max_abs_error = float(np.max(np.abs(action_advantage - expected)))
    if (
        not np.isfinite(max_abs_error)
        or max_abs_error > SERVE_PARITY_FUSION_DERIVED_RELATION_ATOL
    ):
        raise RuntimeError(
            "candidate fusion reference violates action_advantage=Q-V: "
            f"max_abs_error={max_abs_error:.12g}"
        )
    return max_abs_error


def _direction_evidence_fusion_reference_contract(
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Materialize exact finite candidate-VAL column means in fusion order."""

    relation_error = _require_action_value_manifold(frame)
    mean_by_input: dict[str, list[float]] = {}
    ordered: list[float] = []
    for name, width in DIRECTION_EVIDENCE_FUSION_INPUTS:
        values = (
            _numeric_prediction_column(frame, name)
            if width == 1
            else _vector_prediction_column(frame, name, width)
        )
        mean = np.mean(values, axis=0, dtype=np.float64)
        if mean.shape != (width,) or not np.isfinite(mean).all():
            raise RuntimeError(f"candidate VAL fusion mean is invalid for {name}")
        row = [float(item) for item in mean]
        mean_by_input[name] = row
        ordered.extend(row)
    if len(ordered) != DIRECTION_EVIDENCE_FUSION_INPUT_DIM:
        raise RuntimeError(
            "candidate VAL fusion reference width mismatch: "
            f"{len(ordered)} != {DIRECTION_EVIDENCE_FUSION_INPUT_DIM}"
        )
    return {
        "split": SERVE_PARITY_FUSION_REFERENCE_SPLIT,
        "aggregation": SERVE_PARITY_FUSION_REFERENCE_AGGREGATION,
        "coverage": _time_coverage_contract(
            frame.index, label="candidate VAL direction fusion reference"
        ),
        "input_dim": DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
        "inputs_sha256": DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
        "derived_relation": {
            "equation": "action_advantage=action_value-expectile_value_by_horizon",
            "max_abs_error": relation_error,
            "atol": SERVE_PARITY_FUSION_DERIVED_RELATION_ATOL,
        },
        "mean_by_input": mean_by_input,
        "ordered_mean_sha256": _canonical_sha256(ordered),
    }


def _batched_fusion_input_margin_gradients(
    adapter: object,
    states: dict[str, object],
) -> dict[str, np.ndarray]:
    """Measure each learned fusion coordinate on real, manifold-valid states."""

    import torch

    model = adapter._model
    if model is None:
        raise RuntimeError("fusion input influence model is unavailable")
    fusion_norm = getattr(model, "evidence_fusion_norm", None)
    if fusion_norm is None:
        raise RuntimeError("direction evidence fusion LayerNorm is missing")
    device = adapter.device
    seq_t = (
        torch.from_numpy(np.asarray(states["seq"], dtype=np.float32))
        .to(device)
        .requires_grad_(True)
    )
    snap_t = torch.from_numpy(
        np.asarray(states["snap"], dtype=np.float32)
    ).to(device)
    ctx_cont_t = torch.from_numpy(
        np.asarray(states["ctx_cont"], dtype=np.float32)
    ).to(device)
    ctx_cat_t = torch.from_numpy(
        np.asarray(states["ctx_cat"], dtype=np.int64)
    ).to(device)
    per_row_mtf = [
        adapter._multi_tf_window_tensors(pd.Timestamp(ts))
        for ts in np.asarray(states["times"], dtype=object)
    ]
    if not per_row_mtf:
        raise RuntimeError("fusion input influence state subset is empty")
    mtf_kwargs = {
        key: torch.cat([row[key] for row in per_row_mtf], dim=0)
        for key in per_row_mtf[0]
    }
    captured: list[object] = []

    def _capture_fusion_input(_module, inputs):
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError("direction evidence fusion input capture invalid")
        captured.append(inputs[0])

    hook = fusion_norm.register_forward_pre_hook(_capture_fusion_input)
    try:
        out = model(
            seq_t,
            snap_t,
            ctx_cat=ctx_cat_t,
            ctx_cont=ctx_cont_t,
            **mtf_kwargs,
        )
    finally:
        hook.remove()
    if len(captured) != 1 or not torch.is_tensor(captured[0]):
        raise RuntimeError("direction evidence fusion input was not captured once")
    fusion_input = captured[0]
    if tuple(fusion_input.shape) != (
        len(np.asarray(states["times"])),
        DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
    ):
        raise RuntimeError(
            "direction evidence fusion captured input shape mismatch: "
            f"{tuple(fusion_input.shape)}"
        )

    def _surface(logits: object) -> np.ndarray:
        if (
            not torch.is_tensor(logits)
            or tuple(logits.shape)
            != (len(np.asarray(states["times"])), 3)
        ):
            raise RuntimeError("fusion input direction logits are invalid")
        maximum = np.zeros(
            DIRECTION_EVIDENCE_FUSION_INPUT_DIM, dtype=np.float64
        )
        for left, right in ((0, 1), (0, 2), (1, 2)):
            gradient = torch.autograd.grad(
                (logits[:, left] - logits[:, right]).sum(),
                fusion_input,
                retain_graph=True,
            )[0]
            values = (
                gradient.detach()
                .abs()
                .amax(dim=0)
                .cpu()
                .to(torch.float64)
                .numpy()
            )
            if (
                values.shape != maximum.shape
                or not np.isfinite(values).all()
            ):
                raise RuntimeError("fusion input gradient is invalid")
            maximum = np.maximum(maximum, values)
        return maximum

    return {
        "raw": _surface(out.get("raw_direction_logits")),
        "final": _surface(out.get("direction_logits")),
    }


def _direction_evidence_fusion_influence_contract(
    *,
    adapter: object,
    states: dict[str, object],
    parity_targets: pd.DatetimeIndex,
    val_reference_frame: pd.DataFrame,
) -> dict[str, object]:
    """Prove all 26 exact learned-fusion slices move raw and final margins."""

    subset, sampled_targets, failures = _decision_influence_subset(
        states=states,
        parity_targets=parity_targets,
        positions=SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS,
        expected_count=SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
        audit_name="direction evidence fusion influence audit",
    )
    reference = _direction_evidence_fusion_reference_contract(val_reference_frame)
    expected_metadata = direction_evidence_fusion_metadata()
    bundle_dir = Path(adapter.bundle_dir).expanduser().resolve()
    metadata_path = bundle_dir / "bundle_metadata.json"
    lock_path = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    metadata_exact = False
    lock_exact = False
    try:
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_exact = (
            metadata_payload.get("model_native_direction_evidence_fusion")
            == expected_metadata
            == adapter._meta.get("model_native_direction_evidence_fusion")
        )
        if not metadata_exact:
            failures.append("bundle metadata direction fusion contract mismatch")
    except Exception as exc:
        failures.append(f"bundle metadata direction fusion binding failed: {exc}")
    try:
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_exact = (
            lock_payload.get("model_native_direction_evidence_fusion")
            == expected_metadata
        )
        if not lock_exact:
            failures.append("MASTER_TRANSFORMER_LOCK direction fusion contract mismatch")
    except Exception as exc:
        failures.append(f"MASTER_TRANSFORMER_LOCK direction fusion binding failed: {exc}")
    metadata_sha = sha256_file(metadata_path) if metadata_path.is_file() else ""
    lock_sha = sha256_file(lock_path) if lock_path.is_file() else ""

    try:
        baseline_raw, baseline_final = _batched_direction_logits(adapter, subset)
    except Exception as exc:
        failures.append(f"baseline forward failed: {exc}")
        baseline_raw = np.zeros(
            (SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT, 3), dtype=np.float64
        )
        baseline_final = np.zeros(
            (SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT, 3), dtype=np.float64
        )
    try:
        fusion_gradients = _batched_fusion_input_margin_gradients(
            adapter, subset
        )
    except Exception as exc:
        failures.append(f"fusion input gradient execution failed: {exc}")
        fusion_gradients = {
            "raw": np.zeros(
                DIRECTION_EVIDENCE_FUSION_INPUT_DIM, dtype=np.float64
            ),
            "final": np.zeros(
                DIRECTION_EVIDENCE_FUSION_INPUT_DIM, dtype=np.float64
            ),
        }

    groups: dict[str, object] = {}
    means = reference["mean_by_input"]
    for layout in DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT:
        name = str(layout["name"])
        start = int(layout["start"])
        stop = int(layout["stop"])
        width = int(layout["width"])
        row_failures: list[str] = []
        if name in SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS:
            reference_inputs = list(
                SERVE_PARITY_FUSION_DERIVED_REFERENCE_INPUTS[name]
            )
            ablation_surface = (
                SERVE_PARITY_FUSION_DERIVED_ABLATION_SURFACES[name]
            )
            target = (
                "model.evidence_fusion_norm.input["
                + "+".join(
                    (
                        "action_value",
                        "action_advantage",
                    )
                    if name == "action_value"
                    else (
                        ("expectile_value", "action_advantage")
                        if name == "expectile_value"
                        else (
                            "action_value",
                            "expectile_value",
                            "action_advantage",
                        )
                    )
                )
                + "]"
            )
        else:
            reference_inputs = [name]
            ablation_surface = "exact_fusion_slice_val_mean_replacement"
            target = f"model.evidence_fusion_norm.input[{start}:{stop}]"
        reference_values = [
            item
            for reference_name in reference_inputs
            for item in means[reference_name]
        ]
        raw_gradient = float(
            np.max(np.asarray(fusion_gradients["raw"])[start:stop])
        )
        final_gradient = float(
            np.max(np.asarray(fusion_gradients["final"])[start:stop])
        )
        if (
            not np.isfinite(raw_gradient)
            or raw_gradient <= SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON
        ):
            row_failures.append("raw fusion input class-margin gradient is dead")
        if (
            not np.isfinite(final_gradient)
            or final_gradient <= SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON
        ):
            row_failures.append(
                "final fusion input class-margin gradient is dead"
            )
        try:
            ablated_raw, ablated_final = _batched_direction_logits(
                adapter,
                subset,
                fusion_input_replacement=(
                    name,
                    {
                        reference_name: np.asarray(
                            means[reference_name], dtype=np.float64
                        )
                        for reference_name in (
                            SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS
                            if name
                            in SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS
                            else (name,)
                        )
                    },
                ),
            )
            raw_max_delta, raw_changed_rows = _class_centered_delta_metrics(
                baseline=baseline_raw,
                ablated=ablated_raw,
                epsilon=SERVE_PARITY_FUSION_INFLUENCE_EPSILON,
            )
            max_delta, changed_rows = _class_centered_delta_metrics(
                baseline=baseline_final,
                ablated=ablated_final,
                epsilon=SERVE_PARITY_FUSION_INFLUENCE_EPSILON,
            )
            if raw_max_delta <= SERVE_PARITY_FUSION_INFLUENCE_EPSILON:
                row_failures.append(
                    "class-centered raw logits did not move > epsilon"
                )
            if raw_changed_rows < SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS:
                row_failures.append(
                    f"raw_changed_rows={raw_changed_rows} below "
                    f"{SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS}"
                )
            if max_delta <= SERVE_PARITY_FUSION_INFLUENCE_EPSILON:
                row_failures.append(
                    "class-centered final calibrated logits did not move > epsilon"
                )
            if changed_rows < SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS:
                row_failures.append(
                    f"changed_rows={changed_rows} below "
                    f"{SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS}"
                )
        except Exception as exc:
            raw_max_delta = 0.0
            raw_changed_rows = 0
            max_delta = 0.0
            changed_rows = 0
            row_failures.append(f"VAL-mean slice replacement failed: {exc}")
        groups[name] = {
            "decision": "PASS" if not row_failures else "FAIL",
            "failures": row_failures,
            "target": target,
            "ablation_surface": ablation_surface,
            "start": start,
            "stop": stop,
            "width": width,
            "reference_inputs": reference_inputs,
            "reference_values_sha256": _canonical_sha256(reference_values),
            "max_abs_raw_class_margin_input_gradient": raw_gradient,
            "max_abs_final_class_margin_input_gradient": final_gradient,
            "max_abs_class_centered_raw_logit_delta": raw_max_delta,
            "raw_changed_rows": raw_changed_rows,
            "max_abs_class_centered_logit_delta": max_delta,
            "changed_rows": changed_rows,
            "total_rows": SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
        }
        failures.extend(f"{name}: {failure}" for failure in row_failures)

    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_FUSION_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": _time_coverage_contract(
            sampled_targets,
            label="direction fusion influence sampled TEST parity positions",
        ),
        "comparison_surface": SERVE_PARITY_FUSION_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_FUSION_INFLUENCE_EPSILON,
        "fusion_input_gradient_epsilon": (
            SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON
        ),
        "min_changed_rows": SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS,
        "ablation": SERVE_PARITY_FUSION_INFLUENCE_ABLATION,
        "fusion_metadata": expected_metadata,
        "ordered_input_layout": DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT,
        "inputs_sha256": DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
        "input_dim": DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
        "bundle_metadata_path": str(metadata_path),
        "bundle_metadata_sha256": metadata_sha,
        "bundle_metadata_exact_match": metadata_exact,
        "master_transformer_lock_path": str(lock_path),
        "master_transformer_lock_sha256": lock_sha,
        "master_transformer_lock_exact_match": lock_exact,
        "reference": reference,
        "groups": groups,
    }


def _cooperation_gate_liveness_contract(
    frame: pd.DataFrame,
    *,
    column: str,
    token_names: tuple[str, ...],
) -> dict[str, object]:
    failures: list[str] = []
    try:
        gate = _vector_prediction_column(frame, column, len(token_names))
        finite = bool(np.isfinite(gate).all())
        row_sum_error = float(np.max(np.abs(gate.sum(axis=1) - 1.0)))
        mean_weight = gate.mean(axis=0)
        std_weight = gate.std(axis=0)
        clipped = np.clip(gate, 1e-12, 1.0)
        entropy_mean = float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))
        top_rank_count = np.bincount(
            np.argmax(gate, axis=1), minlength=len(token_names)
        )
        if bool(np.any(gate < 0.0)) or row_sum_error > (
            SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
        ):
            failures.append(f"{column} is not a normalized probability simplex")
        if entropy_mean < SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY:
            failures.append(f"{column} entropy_mean={entropy_mean:.12g} below contract")
        for index, token in enumerate(token_names):
            if mean_weight[index] <= SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT:
                failures.append(f"{column}/{token} mean weight is dead")
            if std_weight[index] <= SERVE_PARITY_SPECIALIST_GATE_MIN_STD:
                failures.append(f"{column}/{token} is constant")
            if top_rank_count[index] < SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT:
                failures.append(f"{column}/{token} is never top-ranked")
    except RuntimeError as exc:
        failures.append(str(exc))
        finite = False
        row_sum_error = 1e30
        entropy_mean = -1e30
        mean_weight = np.zeros(len(token_names), dtype=np.float64)
        std_weight = np.zeros(len(token_names), dtype=np.float64)
        top_rank_count = np.zeros(len(token_names), dtype=np.int64)
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": int(len(frame)),
        "finite": finite,
        "tokens": list(token_names),
        "row_sum_max_abs_error": row_sum_error,
        "entropy_mean": entropy_mean,
        "mean_weight": {
            name: float(mean_weight[index]) for index, name in enumerate(token_names)
        },
        "std_weight": {
            name: float(std_weight[index]) for index, name in enumerate(token_names)
        },
        "top_rank_count": {
            name: int(top_rank_count[index]) for index, name in enumerate(token_names)
        },
        "thresholds": {
            "row_sum_max_abs_error": (
                SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
            ),
            "min_mean_weight_exclusive": (
                SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT
            ),
            "min_entropy_inclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY,
            "min_std_exclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_STD,
            "min_top_rank_count_inclusive": (
                SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT
            ),
        },
    }


def _feature_gate_liveness_contract(
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Prove every learned feature×timeframe gate is finite and state-varying."""

    tokens = SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
    failures: list[str] = []
    try:
        gate = _vector_prediction_column(
            frame,
            "family_tf_feature_gate",
            len(tokens),
        )
        finite = bool(np.isfinite(gate).all())
        mean_weight = gate.mean(axis=0)
        std_weight = gate.std(axis=0)
        min_observed = gate.min(axis=0)
        max_observed = gate.max(axis=0)
        for index, token in enumerate(tokens):
            if (
                min_observed[index]
                <= SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE
                or max_observed[index]
                >= SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE
            ):
                failures.append(f"{token} feature gate is saturated/outside (0,2)")
            if std_weight[index] <= SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE:
                failures.append(f"{token} feature gate is constant/dead")
    except RuntimeError as exc:
        failures.append(str(exc))
        finite = False
        mean_weight = np.zeros(len(tokens), dtype=np.float64)
        std_weight = np.zeros(len(tokens), dtype=np.float64)
        min_observed = np.zeros(len(tokens), dtype=np.float64)
        max_observed = np.zeros(len(tokens), dtype=np.float64)
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": int(len(frame)),
        "finite": finite,
        "tokens": list(tokens),
        "mean_weight": {
            token: float(mean_weight[index]) for index, token in enumerate(tokens)
        },
        "std_weight": {
            token: float(std_weight[index]) for index, token in enumerate(tokens)
        },
        "min_observed": {
            token: float(min_observed[index]) for index, token in enumerate(tokens)
        },
        "max_observed": {
            token: float(max_observed[index]) for index, token in enumerate(tokens)
        },
        "thresholds": {
            "min_weight_exclusive": SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE,
            "max_weight_exclusive": SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE,
            "min_std_exclusive": SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE,
        },
    }


def _test_prediction_liveness_contract(frame: pd.DataFrame) -> dict[str, object]:
    """Audit every head plus all 8, 5, 40 and 555 learned gates on TEST."""

    failures: list[str] = []
    active_head_evidence: dict[str, object] = {}
    for head, field_contract in SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS.items():
        head_failures: list[str] = []
        fields: dict[str, object] = {}
        for column, width in field_contract.items():
            try:
                values = (
                    _numeric_prediction_column(frame, column)
                    if width == 1
                    else _vector_prediction_column(frame, column, width)
                )
                component_std = np.std(values, axis=0)
                min_component_std = float(np.min(component_std))
                finite = bool(np.isfinite(values).all())
                if min_component_std <= SERVE_PARITY_HEAD_VARIATION_EPSILON:
                    head_failures.append(
                        f"{head}/{column}: min_component_std={min_component_std:.12g} "
                        f"must be > {SERVE_PARITY_HEAD_VARIATION_EPSILON:.12g}"
                    )
            except RuntimeError as exc:
                values = np.empty((0, width), dtype=np.float64)
                component_std = np.zeros(width, dtype=np.float64)
                min_component_std = 0.0
                finite = False
                head_failures.append(f"{head}/{column}: {exc}")
            fields[column] = {
                "width": int(width),
                "rows": int(len(frame)),
                "finite": finite,
                "component_std": [float(item) for item in component_std],
                "min_component_std": min_component_std,
            }
        active_head_evidence[head] = {
            "decision": "PASS" if not head_failures else "FAIL",
            "failures": head_failures,
            "fields": fields,
        }
        failures.extend(head_failures)

    gate_failures: list[str] = []
    try:
        gate = _vector_prediction_column(
            frame,
            "specialist_gate",
            len(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        )
        finite = bool(np.isfinite(gate).all())
        row_sum_error = float(np.max(np.abs(gate.sum(axis=1) - 1.0)))
        mean_weight = gate.mean(axis=0)
        std_weight = gate.std(axis=0)
        clipped = np.clip(gate, 1e-12, 1.0)
        entropy_mean = float(
            np.mean(-np.sum(clipped * np.log(clipped), axis=1))
        )
        top_rank_count = np.bincount(
            np.argmax(gate, axis=1),
            minlength=len(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        )
        if bool(np.any(gate < 0.0)) or row_sum_error > (
            SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
        ):
            gate_failures.append("specialist_gate is not a normalized probability simplex")
        if entropy_mean < SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY:
            gate_failures.append(
                f"specialist_gate entropy_mean={entropy_mean:.12g} below contract"
            )
        for index, specialist in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS):
            if mean_weight[index] <= SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT:
                gate_failures.append(f"{specialist} mean specialist gate weight is dead")
            if std_weight[index] <= SERVE_PARITY_SPECIALIST_GATE_MIN_STD:
                gate_failures.append(f"{specialist} specialist gate is constant")
            if top_rank_count[index] < SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT:
                gate_failures.append(f"{specialist} is never top-ranked by specialist gate")
    except RuntimeError as exc:
        gate_failures.append(str(exc))
        finite = False
        row_sum_error = 1e30
        entropy_mean = -1e30
        mean_weight = np.zeros(len(MODEL_NATIVE_REQUIRED_SPECIALISTS), dtype=np.float64)
        std_weight = np.zeros(len(MODEL_NATIVE_REQUIRED_SPECIALISTS), dtype=np.float64)
        top_rank_count = np.zeros(len(MODEL_NATIVE_REQUIRED_SPECIALISTS), dtype=np.int64)
    gate_report = {
        "decision": "PASS" if not gate_failures else "FAIL",
        "failures": gate_failures,
        "rows": int(len(frame)),
        "finite": finite,
        "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "row_sum_max_abs_error": row_sum_error,
        "entropy_mean": entropy_mean,
        "mean_weight": {
            name: float(mean_weight[index])
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
        "std_weight": {
            name: float(std_weight[index])
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
        "top_rank_count": {
            name: int(top_rank_count[index])
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
        "thresholds": {
            "row_sum_max_abs_error": (
                SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
            ),
            "min_mean_weight_exclusive": (
                SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT
            ),
            "min_entropy_inclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY,
            "min_std_exclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_STD,
            "min_top_rank_count_inclusive": (
                SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT
            ),
        },
    }
    failures.extend(gate_failures)
    tf_gate_report = _cooperation_gate_liveness_contract(
        frame,
        column="tf_gate",
        token_names=SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES,
    )
    family_tf_gate_report = _cooperation_gate_liveness_contract(
        frame,
        column="family_tf_cooperation_gate",
        token_names=SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS,
    )
    feature_gate_report = _feature_gate_liveness_contract(frame)
    failures.extend(str(item) for item in tf_gate_report["failures"])
    failures.extend(str(item) for item in family_tf_gate_report["failures"])
    failures.extend(str(item) for item in feature_gate_report["failures"])
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": int(len(frame)),
        "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
        "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
        "head_variation_epsilon": SERVE_PARITY_HEAD_VARIATION_EPSILON,
        "active_head_evidence": active_head_evidence,
        "specialist_gate": gate_report,
        "tf_gate": tf_gate_report,
        "family_tf_cooperation_gate": family_tf_gate_report,
        "family_tf_feature_gate": feature_gate_report,
    }


def _softmax(vector: np.ndarray) -> np.ndarray:
    shifted = vector - float(np.max(vector))
    exp = np.exp(shifted)
    return exp / float(exp.sum())


def _require_timestamped_prediction_identity(pinned_path: Path) -> None:
    pinned = pinned_path.expanduser().resolve()
    if _TIMESTAMPED_PREDICTION_RE.fullmatch(pinned.name) is None:
        raise RuntimeError(
            "serve parity requires an exact microsecond-stamped immutable prediction parquet; "
            f"got {pinned}"
        )


def _strict_class_index(value: object, *, name: str, size: int, row_label: object) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"pinned {name} is invalid at {row_label}") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"pinned {name} is not an exact integer at {row_label}")
    index = int(numeric)
    if index < 0 or index >= size:
        raise RuntimeError(f"pinned {name} is outside [0,{size - 1}] at {row_label}")
    return index


def _validate_pinned_prediction_contract(
    frame: pd.DataFrame,
    *,
    dataset_dir: Path,
    pinned_path: Path,
) -> pd.DataFrame:
    """Return indexed candidate rows after strict model-direction SSOT validation."""
    _require_timestamped_prediction_identity(pinned_path)
    if frame.empty:
        raise RuntimeError("fresh event-pinned predictions are empty")
    missing = [column for column in PINNED_REQUIRED_COLS if column not in frame.columns]
    if missing:
        raise RuntimeError(
            "fresh serve-parity predictions are missing required model-direction columns: "
            + ",".join(missing)
        )
    legacy = [column for column in FORBIDDEN_LEGACY_DECISION_COLS if column in frame.columns]
    legacy.extend(
        column
        for column in frame.columns
        if any(blocked in str(column) for blocked in MODEL_NATIVE_BLOCKED_HEADS)
    )
    if legacy:
        raise RuntimeError(
            "legacy expected-utility decision columns are forbidden in serve parity: "
            + ",".join(legacy)
        )

    candidate = frame.loc[
        (frame["model"].astype(str) == MODEL_NATIVE_REQUIRED_MODEL_NAME)
        & (frame["split"].astype(str) == MODEL_NATIVE_REQUIRED_TEST_SPLIT)
    ].copy()
    if candidate.empty:
        raise RuntimeError(
            "fresh event-pinned predictions contain no candidate TEST rows"
        )
    observed_modes = set(candidate["selection_score_mode"].astype(str))
    if observed_modes != {MODEL_DIRECTION_SELECTION_MODE}:
        raise RuntimeError(
            "pinned predictions selection_score_mode must be exactly "
            f"{MODEL_DIRECTION_SELECTION_MODE!r}; got {sorted(observed_modes)}"
        )
    try:
        candidate["time"] = pd.to_datetime(candidate["time"], utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError("fresh event-pinned predictions contain invalid UTC times") from exc
    if candidate["time"].duplicated().any():
        duplicates = candidate.loc[candidate["time"].duplicated(), "time"].astype(str).tolist()
        raise RuntimeError(f"fresh event-pinned candidate times are duplicated: {duplicates[:3]}")

    for row_number, row in candidate.iterrows():
        row_label = row.get("time", row_number)
        direction_logits = _finite_vector(
            row["direction_logits"], name="direction_logits", size=3, row_label=row_label
        )
        public_logits = _finite_vector(
            row["public_trade_flat_decision_logits"],
            name="public_trade_flat_decision_logits",
            size=2,
            row_label=row_label,
        )
        canonical_public = np.asarray(
            [
                max(
                    float(direction_logits[MODEL_DIRECTION_LONG_INDEX]),
                    float(direction_logits[MODEL_DIRECTION_SHORT_INDEX]),
                ),
                float(direction_logits[MODEL_DIRECTION_FLAT_INDEX]),
            ],
            dtype=np.float64,
        )
        if not np.array_equal(public_logits, canonical_public):
            raise RuntimeError(
                "pinned public_trade_flat_decision_logits are not the canonical pair "
                f"[max(direction LONG,SHORT), direction FLAT] at {row_label}"
            )

        direction_probs = _softmax(direction_logits)
        public_probs = _softmax(public_logits)
        pinned_direction_probs = np.asarray(
            [row["p_long"], row["p_short"], row["p_flat"]], dtype=np.float64
        )
        pinned_public_probs = np.asarray(
            [row["public_trade_probability"], row["public_flat_probability"]],
            dtype=np.float64,
        )
        if not np.isfinite(pinned_direction_probs).all() or not np.allclose(
            pinned_direction_probs, direction_probs, rtol=0.0, atol=2e-6
        ):
            raise RuntimeError(f"pinned p_long/p_short/p_flat do not match direction_logits at {row_label}")
        if not np.isfinite(pinned_public_probs).all() or not np.allclose(
            pinned_public_probs, public_probs, rtol=0.0, atol=2e-6
        ):
            raise RuntimeError(
                "pinned public trade/flat probabilities do not match canonical logits "
                f"at {row_label}"
            )

        if int(np.count_nonzero(direction_logits == np.max(direction_logits))) != 1:
            raise RuntimeError(
                f"pinned direction_logits have no unique top class at {row_label}"
            )
        direction_index = int(np.argmax(direction_logits))
        public_index = int(np.argmax(public_logits))
        pred_direction = _strict_class_index(
            row["pred_direction"], name="pred_direction", size=3, row_label=row_label
        )
        trade_side = _strict_class_index(
            row["trade_side"], name="trade_side", size=3, row_label=row_label
        )
        pinned_public_index = _strict_class_index(
            row["public_trade_flat_hard_decision"],
            name="public_trade_flat_hard_decision",
            size=2,
            row_label=row_label,
        )
        if pred_direction != direction_index or trade_side != direction_index:
            raise RuntimeError(
                "pinned pred_direction/trade_side do not equal final direction argmax "
                f"at {row_label}"
            )
        if pinned_public_index != public_index:
            raise RuntimeError(
                "pinned public_trade_flat_hard_decision does not equal canonical pair argmax "
                f"at {row_label}"
            )
        expected_public_index = (
            PUBLIC_FLAT_INDEX
            if direction_index == MODEL_DIRECTION_FLAT_INDEX
            else PUBLIC_TRADE_INDEX
        )
        if public_index != expected_public_index:
            raise RuntimeError(
                f"pinned direction/public decisions disagree at {row_label}: "
                f"direction={direction_index}, public={public_index}"
            )
        for column in (*FORWARD_SCALAR_MAP, "selection_score"):
            try:
                value = float(row[column])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"pinned {column} is not numeric at {row_label}") from exc
            if not np.isfinite(value):
                raise RuntimeError(f"pinned {column} is non-finite at {row_label}")
        expected_edge = max(
            float(direction_probs[MODEL_DIRECTION_LONG_INDEX]),
            float(direction_probs[MODEL_DIRECTION_SHORT_INDEX]),
        ) - float(direction_probs[MODEL_DIRECTION_FLAT_INDEX])
        if not np.isclose(float(row["edge_score"]), expected_edge, rtol=0.0, atol=2e-6):
            raise RuntimeError(f"pinned edge_score does not match direction probabilities at {row_label}")

    return candidate.set_index("time").sort_index()


def _load_pinned_predictions(
    *,
    dataset_dir: Path,
    pinned_path: Path,
    prediction_report_path: Path,
    expected_predictions_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, dict, dict, dict[str, str]]:
    """Resolve and revalidate one explicitly named immutable prediction event.

    The fixed ``selective_edge_predictions.parquet`` mirror is deliberately not
    accepted here: a mutable locator cannot decide which evidence launch proves.
    The caller must name the timestamped parquet, whose timestamped report binds
    bundle, dataset, physical schema, semantics, and exact hashes.
    """

    requested = pinned_path.expanduser().resolve()
    requested_report = prediction_report_path.expanduser().resolve()
    authoritative, prediction_report, prediction_evidence = (
        resolve_and_validate_prediction_evidence(
            requested,
            prediction_report_path=requested_report,
            bundle_dir=None,
            dataset_dir=dataset_dir,
            expected_sha256=expected_predictions_sha256,
            expected_stage=MODEL_NATIVE_REQUIRED_EVIDENCE_STAGE,
            expected_splits=(MODEL_NATIVE_REQUIRED_TEST_SPLIT,),
            expected_model=MODEL_NATIVE_REQUIRED_MODEL_NAME,
        )
    )
    _require_timestamped_prediction_identity(authoritative)
    report_evidence = {
        "json_path": str(requested_report),
        "sha256": sha256_file(requested_report),
    }
    complete_frame = pd.read_parquet(authoritative)
    frame = _validate_pinned_prediction_contract(
        complete_frame,
        dataset_dir=dataset_dir,
        pinned_path=authoritative,
    )
    val_reference_frame = _validate_fusion_reference_prediction_contract(
        complete_frame
    )
    return (
        frame,
        val_reference_frame,
        authoritative,
        prediction_report,
        prediction_evidence,
        report_evidence,
    )


def _forward_row_deltas(head: dict[str, object], pinned: pd.Series) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for pinned_column, live_key in FORWARD_FIELD_MAP.items():
        if live_key not in head:
            raise RuntimeError(f"live forward is missing required field {live_key!r}")
        width = SERVE_PARITY_FORWARD_FIELD_WIDTHS[pinned_column]
        live_vector = _finite_vector(
            head[live_key],
            name=f"live {live_key}",
            size=width,
            row_label=head.get("time"),
        )
        pinned_vector = _finite_vector(
            pinned[pinned_column],
            name=pinned_column,
            size=width,
            row_label=pinned.name,
        )
        deltas[pinned_column] = float(
            np.max(np.abs(live_vector - pinned_vector))
        )
    return deltas


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def _prediction_report_test_parquet(
    prediction_report: dict[str, Any],
    dataset_dir: Path,
) -> Path:
    contract = prediction_report.get("dataset_signal_contract")
    rows = contract.get("splits") if isinstance(contract, dict) else None
    row = rows.get(MODEL_NATIVE_REQUIRED_TEST_SPLIT) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise RuntimeError(
            "[parity] prediction report lacks exact TEST dataset artifact binding"
        )
    manifest_path: Path | None = None
    parquet_path: Path | None = None
    for kind, suffix in (
        ("manifest", f"_{MODEL_NATIVE_REQUIRED_TEST_SPLIT}.manifest.json"),
        ("parquet", f"_{MODEL_NATIVE_REQUIRED_TEST_SPLIT}.parquet"),
    ):
        path = Path(str(row.get(f"{kind}_path") or "")).expanduser()
        expected_sha = str(row.get(f"{kind}_sha256") or "").strip().lower()
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or path.parent != dataset_dir
            or not path.name.endswith(suffix)
            or any("latest" in part.lower() for part in path.parts)
        ):
            raise RuntimeError(
                f"[parity] prediction report TEST {kind} identity is invalid: {path}"
            )
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise RuntimeError(
                f"[parity] prediction report TEST {kind} lacks SHA-256"
            )
        if sha256_file(path) != expected_sha:
            raise RuntimeError(
                f"[parity] prediction report TEST {kind} hash mismatch"
            )
        if kind == "manifest":
            manifest_path = path
        else:
            parquet_path = path
    if manifest_path is None or parquet_path is None:
        raise RuntimeError("[parity] TEST dataset identity is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(str(manifest.get("output_data_path") or "")).expanduser() != parquet_path:
        raise RuntimeError("[parity] TEST manifest output_data_path mismatch")
    if sha256_file(manifest_path) != str(row["manifest_sha256"]).strip().lower():
        raise RuntimeError("[parity] TEST manifest changed during validation")
    return parquet_path


def _load_offline_rows(parquet_path: Path, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Stream the split parquet batch-wise and keep ONLY the target rows.
    (A one-shot filtered to_table materializes every row group's nested
    (96,513) seq lists ≈ 14+GB — OOM'd the 34G-capped gate 2026-07-08.)"""
    import pyarrow.parquet as pq
    want = set(times.tz_convert("UTC").asi8)
    pf = pq.ParquetFile(parquet_path)
    kept: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=512, columns=["time", "seq", "snap", "ctx_cont", "ctx_cat"]):
        ts = pd.to_datetime(pd.Series(batch.column("time").to_pandas()), utc=True)
        mask = ts.astype("int64").isin(want)
        if mask.any():
            df_b = batch.to_pandas()
            df_b["time"] = ts
            kept.append(df_b.loc[mask.to_numpy()])
    if not kept:
        return pd.DataFrame(columns=["seq", "snap", "ctx_cont", "ctx_cat"]).set_index(
            pd.DatetimeIndex([], tz="UTC"))
    df = pd.concat(kept, ignore_index=True)
    return df.set_index("time").sort_index()


def main() -> int:
    _apply_exact_env_pins()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--pair-manifest-path", type=Path, required=True)
    ap.add_argument("--pair-generation-root", type=Path, required=True)
    ap.add_argument(
        "--pinned-predictions",
        type=Path,
        required=True,
        help="explicit timestamped authoritative selective_edge_predictions_<stamp>.parquet",
    )
    ap.add_argument(
        "--pinned-predictions-sha256",
        type=str,
        required=True,
        help="exact sha256 of the pinned authoritative predictions parquet",
    )
    ap.add_argument(
        "--prediction-report-json",
        type=Path,
        required=True,
        help="matching newest immutable ENTRY_CANDIDATE_SELECTIVE_EDGE_<stamp>.json",
    )
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="explicit strict pre-launch candidate bundle bound by prediction evidence",
    )
    ap.add_argument(
        "--max-trades",
        type=int,
        required=True,
        help="explicit execution exposure cap for the rule-free operating point",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="explicit output directory for one timestamped immutable parity event",
    )
    args = ap.parse_args()

    # Reject stale identity/schema before loading the active bundle or live
    # prebuilts. There is deliberately no legacy pinned-prediction fallback.
    dataset_dir = args.dataset_dir.expanduser().resolve()
    requested_pinned_path = args.pinned_predictions.expanduser().resolve()
    (
        pinned,
        val_reference_frame,
        pinned_path,
        prediction_report,
        prediction_evidence,
        prediction_report_evidence,
    ) = _load_pinned_predictions(
        dataset_dir=dataset_dir,
        pinned_path=args.pinned_predictions,
        prediction_report_path=args.prediction_report_json,
        expected_predictions_sha256=str(args.pinned_predictions_sha256),
    )
    dataset_parquet = _prediction_report_test_parquet(
        prediction_report,
        dataset_dir,
    )

    t0 = time.time()
    failures: list[str] = []
    report: dict = {
        "schema_version": MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION,
        "contract_version": MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_dir": str(dataset_dir),
        "split": MODEL_NATIVE_REQUIRED_TEST_SPLIT,
        "model_name": MODEL_NATIVE_REQUIRED_MODEL_NAME,
        "pinned_predictions": str(pinned_path),
        "requested_pinned_predictions": str(requested_pinned_path),
        "requested_prediction_report_json": str(
            args.prediction_report_json.expanduser().resolve()
        ),
        "prediction_report_evidence": prediction_report_evidence,
        "prediction_evidence": prediction_evidence,
        "state_tol": SERVE_PARITY_STATE_TOL,
        "forward_tol": SERVE_PARITY_FORWARD_TOL,
        "sampling_contract": SERVE_PARITY_SAMPLING_CONTRACT,
        "pinned_prediction_contract": {
            "dataset_dir": str(dataset_dir),
            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
            "candidate_test_rows": int(len(pinned)),
            "canonical_public_pair": True,
        },
        "env_pins": {name: os.environ[name] for name in SERVE_PARITY_ENV_PINS},
        "serve_source_identity": build_serve_source_identity(
            Path(__file__).resolve().parents[2]
        ),
    }

    # ── explicit pre-launch candidate through the shared live adapter ────────
    from gx1.execution.v12_smart_entry_live import SmartEntryLiveInference
    from gx1.execution.v12_model_native_state_live import SEQ_LEN_MODEL_NATIVE
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_FIELDS, MODEL_NATIVE_CTX_CAT_FIELDS,
    )
    requested_bundle_dir = args.bundle_dir.expanduser().resolve()
    prediction_bundle_dir = Path(
        str(prediction_report.get("bundle_dir") or "")
    ).expanduser().resolve()
    if prediction_bundle_dir != requested_bundle_dir:
        raise RuntimeError(
            "[parity] prediction evidence bundle does not equal the explicit "
            f"candidate bundle: prediction={prediction_bundle_dir} "
            f"candidate={requested_bundle_dir}"
        )
    operating_point = {
        "selection_score": MODEL_DIRECTION_SELECTION_MODE,
        "max_trades": args.max_trades,
    }
    adapter = SmartEntryLiveInference.load_candidate_for_parity(
        bundle_dir=requested_bundle_dir,
        operating_point=operating_point,
        device="cpu",
    )
    report["bundle_dir"] = str(adapter.bundle_dir)
    report["operating_point"] = adapter.operating_point
    report["runtime_device"] = adapter.device
    direction_decision_contract = require_model_direction_decision_contract(
        adapter._meta,
        context="[parity] explicit candidate bundle",
    )
    report["direction_decision_contract"] = direction_decision_contract
    selection_mode = adapter.operating_point.get("selection_score")
    if selection_mode != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError(
            "[parity] operating_point.selection_score must be exactly "
            f"{MODEL_DIRECTION_SELECTION_MODE!r}; got {selection_mode!r}"
        )
    report["selection_score_mode"] = selection_mode
    if adapter._state_contract is None:
        raise RuntimeError("[parity] model-native state contract missing from adapter")
    report["model_native_state_contract"] = adapter._state_contract.as_report()
    report["feature_history_start_utc"] = str(
        adapter._state_contract.feature_history_start_utc
    )
    signal_names = adapter._builder.ordered_signal_names

    # ── live prebuilts (frozen snapshot — deterministic) ─────────────────────
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
    pair_manifest_path = args.pair_manifest_path.expanduser().resolve()
    pair_generation_root = args.pair_generation_root.expanduser().resolve()
    loader = PrebuiltStateLoader(
        pair_manifest_path=pair_manifest_path,
        generation_root=pair_generation_root,
    )
    loader.load()
    loader._refresh_enabled = False
    cutoff = loader.cutoff_ts
    report["pair_manifest_path"] = str(pair_manifest_path)
    report["pair_generation_root"] = str(pair_generation_root)
    report["pair_generation_id"] = loader.pair_generation_id
    report["canonical_v3_path"] = str(loader.canonical_v3_path)
    report["base28_path"] = str(loader.base28_path)
    report["live_prebuilt_cutoff"] = str(cutoff)
    print(f"[parity] live prebuilts loaded (cutoff={cutoff}, {time.time()-t0:.0f}s)", flush=True)

    # ── exact TEST coverage + deterministic representative positions ─────────
    all_times = pd.to_datetime(
        pd.read_parquet(dataset_parquet, columns=["time"])["time"],
        utc=True,
        errors="raise",
    ).sort_values()
    if all_times.empty:
        raise RuntimeError("TEST dataset time coverage is empty")
    if bool(all_times.duplicated().any()):
        raise RuntimeError("TEST dataset time coverage contains duplicate rows")
    all_times = all_times.reset_index(drop=True)
    dataset_coverage = _time_coverage_contract(all_times, label="TEST dataset")
    prediction_coverage = _time_coverage_contract(
        pinned.index, label="candidate TEST predictions"
    )
    if prediction_coverage != dataset_coverage:
        raise RuntimeError(
            "candidate prediction time coverage does not exactly equal the complete "
            f"TEST dataset: dataset={dataset_coverage} predictions={prediction_coverage}"
        )
    if pd.Timestamp(all_times.iloc[-1]) > pd.Timestamp(cutoff):
        raise RuntimeError(
            "live prebuilt cutoff does not cover the complete TEST dataset: "
            f"test_end={all_times.iloc[-1]} cutoff={cutoff}"
        )
    pick = _deterministic_sample_positions(len(all_times))
    targets = pd.DatetimeIndex(all_times.iloc[pick])
    if len(targets) != SERVE_PARITY_SAMPLE_COUNT:
        raise RuntimeError("parity target count violates the exact 256-row contract")
    report["dataset_parquet"] = str(dataset_parquet)
    report["dataset_parquet_sha256"] = sha256_file(dataset_parquet)
    report["test_coverage"] = {
        "dataset": dataset_coverage,
        "predictions": prediction_coverage,
        "exact_match": True,
    }
    test_prediction_liveness = _test_prediction_liveness_contract(pinned)
    report["test_prediction_liveness"] = test_prediction_liveness
    failures.extend(
        f"TEST prediction liveness: {failure}"
        for failure in test_prediction_liveness["failures"]
    )
    report["sampled_test_coverage"] = _time_coverage_contract(
        targets, label="sampled TEST parity positions"
    )
    report["n_bars"] = int(len(targets))
    report["target_range"] = [str(targets[0]), str(targets[-1])]
    print(f"[parity] {len(targets)} target bars: {targets[0]} .. {targets[-1]}", flush=True)

    # ── LIVE common-history frame + states (shared one-truth adapter path) ─────
    t_max = targets[-1]
    frame = adapter.build_common_history_frame(loader, t_max)
    report["live_frame"] = {"rows": int(len(frame)),
                            "start": str(frame["time"].iloc[0]), "end": str(frame["time"].iloc[-1])}
    print(f"[parity] common-history frame prepared: {len(frame)} rows ({time.time()-t0:.0f}s)", flush=True)
    states = adapter._builder.build_states(frame, list(targets))
    print(f"[parity] live states built ({time.time()-t0:.0f}s)", flush=True)

    # ── LEG 1: state parity vs offline dataset rows ──────────────────────────
    off = _load_offline_rows(dataset_parquet, targets)
    missing = [str(t) for t in targets if t not in off.index]
    if missing:
        failures.append(f"offline rows missing for {len(missing)} targets: {missing[:3]}")
    block_max: dict[str, float] = {b: 0.0 for b in STATE_BLOCKS}
    worst: dict[str, dict] = {}
    per_bar_rows = []
    # per-COLUMN max diff across bars (diagnostic: which families skew)
    col_max_snap = np.zeros(len(signal_names))
    col_max_seq = np.zeros(len(signal_names))
    col_max_ctx = np.zeros(len(MODEL_NATIVE_CTX_CONT_FIELDS))
    state_rows_compared = 0
    for k, ts in enumerate(targets):
        if ts not in off.index:
            continue
        row = off.loc[ts]
        o_seq = np.asarray([np.asarray(x, dtype=np.float64) for x in row["seq"]])
        o_snap = np.asarray(row["snap"], dtype=np.float64)
        o_ctx = np.asarray(row["ctx_cont"], dtype=np.float64)
        o_cat = np.asarray(row["ctx_cat"], dtype=np.int64)
        l_seq = states["seq"][k].astype(np.float64)
        l_snap = states["snap"][k].astype(np.float64)
        l_ctx = states["ctx_cont"][k].astype(np.float64)
        l_cat = states["ctx_cat"][k]
        shape_mismatches = [
            f"{name} live={live_values.shape} offline={offline_values.shape}"
            for name, live_values, offline_values in (
                ("seq", l_seq, o_seq),
                ("snap", l_snap, o_snap),
                ("ctx_cont", l_ctx, o_ctx),
                ("ctx_cat", l_cat, o_cat),
            )
            if live_values.shape != offline_values.shape
        ]
        if shape_mismatches:
            failures.append(f"{ts}: state shape mismatch: {'; '.join(shape_mismatches)}")
            continue
        nonfinite_blocks = [
            name
            for name, values in (
                ("offline.seq", o_seq),
                ("offline.snap", o_snap),
                ("offline.ctx_cont", o_ctx),
                ("live.seq", l_seq),
                ("live.snap", l_snap),
                ("live.ctx_cont", l_ctx),
            )
            if not np.isfinite(values).all()
        ]
        if nonfinite_blocks:
            failures.append(
                f"{ts}: non-finite state evidence in {','.join(nonfinite_blocks)}"
            )
            continue
        state_rows_compared += 1
        d_seq = np.abs(l_seq - o_seq)
        d_snap = np.abs(l_snap - o_snap)
        d_ctx = np.abs(l_ctx - o_ctx)
        d_cat = np.abs(l_cat - o_cat).max()
        col_max_snap = np.maximum(col_max_snap, d_snap)
        col_max_seq = np.maximum(col_max_seq, d_seq.max(axis=0))
        col_max_ctx = np.maximum(col_max_ctx, d_ctx)
        vals = {"seq": float(d_seq.max()), "snap": float(d_snap.max()),
                "ctx_cont": float(d_ctx.max()), "ctx_cat": float(d_cat)}
        per_bar_rows.append({"time": str(ts), **vals})
        for b, v in vals.items():
            if v > block_max[b]:
                block_max[b] = v
                if b == "seq":
                    r, c = np.unravel_index(int(d_seq.argmax()), d_seq.shape)
                    worst[b] = {"time": str(ts), "row_offset": int(r) - (SEQ_LEN_MODEL_NATIVE - 1),
                                "col": signal_names[int(c)], "diff": float(d_seq.max())}
                elif b == "snap":
                    c = int(d_snap.argmax())
                    worst[b] = {"time": str(ts), "col": signal_names[c], "diff": float(d_snap.max())}
                elif b == "ctx_cont":
                    c = int(d_ctx.argmax())
                    worst[b] = {"time": str(ts), "col": MODEL_NATIVE_CTX_CONT_FIELDS[c],
                                "diff": float(d_ctx.max())}
                else:
                    worst[b] = {"time": str(ts),
                                "col": MODEL_NATIVE_CTX_CAT_FIELDS[int(np.abs(l_cat - o_cat).argmax())],
                                "diff": float(d_cat)}
    def _top(names, arr, k=25):
        order = np.argsort(arr)[::-1][:k]
        return [{"col": str(names[i]), "max_abs_diff": float(arr[i])}
                for i in order if arr[i] > SERVE_PARITY_STATE_TOL]
    report["state_parity"] = {
        "n_compared": state_rows_compared,
        "block_max_abs_diff": block_max,
        "worst": worst,
        "tolerance": SERVE_PARITY_STATE_TOL,
        "top_offenders": {
            "snap": _top(signal_names, col_max_snap),
            "seq": _top(signal_names, col_max_seq),
            "ctx_cont": _top(list(MODEL_NATIVE_CTX_CONT_FIELDS), col_max_ctx),
        },
    }
    if state_rows_compared != SERVE_PARITY_SAMPLE_COUNT:
        failures.append(
            f"STATE compared {state_rows_compared} rows; exact contract requires "
            f"{SERVE_PARITY_SAMPLE_COUNT}"
        )
    for b, v in block_max.items():
        tol = 0.0 if b == "ctx_cat" else SERVE_PARITY_STATE_TOL
        if v > tol:
            failures.append(
                f"STATE block '{b}' max_abs_diff={v:.3e} > tol={tol:.0e} (worst: {worst.get(b)})"
            )
    print(f"[parity] LEG1 state: {json.dumps(block_max)} ({time.time()-t0:.0f}s)", flush=True)

    # ── LEG 2: forward parity vs fresh model-direction predictions ──────────
    heads = adapter.forward_states(states)
    print(f"[parity] LEG2 forward done ({time.time()-t0:.0f}s)", flush=True)
    if len(heads) != len(targets):
        failures.append(
            f"FORWARD emitted {len(heads)} rows for {len(targets)} requested target bars"
        )
    fwd_max: dict[str, float] = {c: 0.0 for c in FORWARD_COLS}
    fwd_worst: dict[str, str] = {}
    n_fwd = 0
    direction_calibration = adapter._meta.get("direction_calibration")
    if not isinstance(direction_calibration, dict) or direction_calibration.get("enabled") is not True:
        raise RuntimeError("serve parity requires enabled direction calibration metadata")
    calibration_temperature = float(direction_calibration.get("temperature"))
    calibration_bias = np.asarray(
        direction_calibration.get("bias"), dtype=np.float64
    ).reshape(-1)
    if (
        not np.isfinite(calibration_temperature)
        or calibration_temperature <= 0.0
        or calibration_bias.shape != (3,)
        or not np.isfinite(calibration_bias).all()
    ):
        raise RuntimeError("serve parity direction calibration parameters are invalid")
    calibration_max_abs_diff = 0.0
    calibration_worst_ts: str | None = None
    atr_bps_values: list[float] = []
    direction_mismatch: list[str] = []
    public_pair_mismatch: list[str] = []
    action_mismatch: list[str] = []
    target_positions = {pd.Timestamp(ts): index for index, ts in enumerate(targets)}
    atr_ctx_index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index("atr_bps")
    for h in heads:
        ts = pd.Timestamp(h["time"])
        if ts not in pinned.index:
            failures.append(f"pinned prediction missing for {ts}")
            continue
        if ts not in target_positions:
            failures.append(f"FORWARD emitted an unrequested timestamp {ts}")
            continue
        p = pinned.loc[ts]
        deltas = _forward_row_deltas(h, p)
        raw_logits = _finite_vector(
            h.get("raw_direction_logits"),
            name="live raw_direction_logits",
            size=3,
            row_label=ts,
        )
        final_logits = _finite_vector(
            h.get("direction_logits"),
            name="live direction_logits",
            size=3,
            row_label=ts,
        )
        calibration_delta = float(
            np.max(
                np.abs(
                    final_logits
                    - (raw_logits / calibration_temperature + calibration_bias)
                )
            )
        )
        if calibration_delta > calibration_max_abs_diff:
            calibration_max_abs_diff = calibration_delta
            calibration_worst_ts = str(ts)
        atr_bps = float(
            states["ctx_cont"][target_positions[ts]][atr_ctx_index]
        )
        if not np.isfinite(atr_bps) or atr_bps <= 0.0:
            raise RuntimeError(
                f"{ts}: live state ctx_cont.atr_bps must be finite and positive; "
                f"got {atr_bps!r}"
            )
        atr_bps_values.append(atr_bps)
        decision = adapter.decide_direction(h)
        if decision.get("selection_score_mode") != MODEL_DIRECTION_SELECTION_MODE:
            raise RuntimeError(
                f"{ts}: live decision selection_score_mode="
                f"{decision.get('selection_score_mode')!r}"
            )
        pinned_direction = int(p["pred_direction"])
        pinned_public = int(p["public_trade_flat_hard_decision"])
        pinned_action = MODEL_DIRECTION_ACTIONS[pinned_direction]
        n_fwd += 1
        for c, d in deltas.items():
            if d > fwd_max[c]:
                fwd_max[c] = d
                fwd_worst[c] = str(ts)
        if int(decision["model_direction_index"]) != pinned_direction:
            direction_mismatch.append(str(ts))
        if int(decision["public_trade_flat_decision_index"]) != pinned_public:
            public_pair_mismatch.append(str(ts))
        if decision["action"] != pinned_action:
            action_mismatch.append(str(ts))
    if n_fwd != SERVE_PARITY_SAMPLE_COUNT:
        failures.append(
            f"FORWARD compared {n_fwd} rows; exact contract requires "
            f"{SERVE_PARITY_SAMPLE_COUNT}"
        )
    per_head_tol = {c: SERVE_PARITY_FORWARD_TOL for c in fwd_max}
    report["forward_parity"] = {
        "n_compared": n_fwd, "max_abs_diff": fwd_max, "worst_ts": fwd_worst,
        "model_direction_mismatches": direction_mismatch,
        "public_trade_flat_decision_mismatches": public_pair_mismatch,
        "model_argmax_action_mismatches": action_mismatch,
        "selection_score_mode": selection_mode,
        "direction_decision": direction_decision_contract["direction_decision"],
        "canonical_public_pair": direction_decision_contract["public_trade_flat_formula"],
        "tolerance": SERVE_PARITY_FORWARD_TOL,
        "per_head_tolerance": per_head_tol,
        "atr_bps_source": "live_state.ctx_cont.atr_bps",
        "atr_bps_min": float(min(atr_bps_values)) if atr_bps_values else None,
        "atr_bps_max": float(max(atr_bps_values)) if atr_bps_values else None,
        "note": "pinned=CUDA fp32 evidence run; live=CPU fp32 — LEG2 bounds backend drift only",
    }
    report["direction_calibration_parity"] = {
        "decision": (
            "PASS"
            if n_fwd == SERVE_PARITY_SAMPLE_COUNT
            and calibration_max_abs_diff <= SERVE_PARITY_CALIBRATION_TOL
            else "FAIL"
        ),
        "failures": (
            []
            if n_fwd == SERVE_PARITY_SAMPLE_COUNT
            and calibration_max_abs_diff <= SERVE_PARITY_CALIBRATION_TOL
            else ["raw-to-final direction calibration equation mismatch"]
        ),
        "n_compared": n_fwd,
        "equation": SERVE_PARITY_CALIBRATION_EQUATION,
        "enabled": True,
        "temperature": calibration_temperature,
        "bias": calibration_bias.tolist(),
        "tolerance": SERVE_PARITY_CALIBRATION_TOL,
        "max_abs_diff": calibration_max_abs_diff,
        "worst_ts": calibration_worst_ts,
    }
    if calibration_max_abs_diff > SERVE_PARITY_CALIBRATION_TOL:
        failures.append(
            "DIRECTION CALIBRATION max_abs_diff="
            f"{calibration_max_abs_diff:.3e} > {SERVE_PARITY_CALIBRATION_TOL:.3e}"
        )
    for c, v in fwd_max.items():
        if v > per_head_tol[c]:
            failures.append(f"FORWARD '{c}' max_abs_diff={v:.3e} > tol={per_head_tol[c]:.3e} "
                            f"(worst ts {fwd_worst.get(c)})")
    if direction_mismatch:
        failures.append(
            f"FORWARD model direction argmax mismatches: {len(direction_mismatch)} "
            f"({direction_mismatch[:3]})"
        )
    if public_pair_mismatch:
        failures.append(
            f"FORWARD public trade/flat decision mismatches: {len(public_pair_mismatch)} "
            f"({public_pair_mismatch[:3]})"
        )
    if action_mismatch:
        failures.append(
            f"FORWARD model-argmax action mismatches: {len(action_mismatch)} ({action_mismatch[:3]})"
        )
    print(f"[parity] LEG2 forward: {json.dumps({k: round(v, 9) for k, v in fwd_max.items()})}", flush=True)

    # ── LEG 3: exact full-stack specialist decision influence ───────────────
    specialist_influence = _specialist_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=targets,
    )
    report["specialist_decision_influence"] = specialist_influence
    failures.extend(
        f"specialist decision influence: {failure}"
        for failure in specialist_influence["failures"]
    )
    print(
        "[parity] LEG3 specialist influence: "
        f"{specialist_influence['decision']} ({time.time()-t0:.0f}s)",
        flush=True,
    )

    individual_input_influence = (
        _individual_input_decision_influence_contract(
            adapter=adapter,
            states=states,
            parity_targets=targets,
        )
    )
    report["individual_input_decision_influence"] = (
        individual_input_influence
    )
    failures.extend(
        f"individual input decision influence: {failure}"
        for failure in individual_input_influence["failures"]
    )
    print(
        "[parity] LEG3b all individual inputs: "
        f"{individual_input_influence['decision']} "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    # ── LEG 4: upstream, all-TF and exact fusion influence ──────────────────
    upstream_influence = _upstream_context_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=targets,
    )
    report["upstream_context_decision_influence"] = upstream_influence
    failures.extend(
        f"upstream context decision influence: {failure}"
        for failure in upstream_influence["failures"]
    )
    multi_tf_influence = _multi_tf_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=targets,
    )
    report["multi_tf_decision_influence"] = multi_tf_influence
    failures.extend(
        f"multi-TF decision influence: {failure}"
        for failure in multi_tf_influence["failures"]
    )
    family_tf_influence = _family_tf_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=targets,
    )
    report["family_tf_decision_influence"] = family_tf_influence
    failures.extend(
        f"family×timeframe decision influence: {failure}"
        for failure in family_tf_influence["failures"]
    )
    fusion_influence = _direction_evidence_fusion_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=targets,
        val_reference_frame=val_reference_frame,
    )
    report["direction_evidence_fusion_influence"] = fusion_influence
    failures.extend(
        f"direction evidence fusion influence: {failure}"
        for failure in fusion_influence["failures"]
    )
    print(
        "[parity] LEG4 upstream/TF/fusion influence: "
        f"{upstream_influence['decision']}/"
        f"{multi_tf_influence['decision']}/"
        f"{family_tf_influence['decision']}/"
        f"{fusion_influence['decision']} ({time.time()-t0:.0f}s)",
        flush=True,
    )

    # ── verdict ───────────────────────────────────────────────────────────────
    report["per_bar_state_diffs"] = per_bar_rows
    report["failures"] = list(failures)
    report["decision"] = "PASS" if not failures else "FAIL"
    if not failures:
        contract_failures = serve_gate_event_contract_failures(
            report,
            evidence_name="model_native_serve_parity",
        )
        if contract_failures:
            failures.extend(contract_failures)
            report["failures"] = list(failures)
            report["decision"] = "FAIL"
    report["runtime_s"] = round(time.time() - t0, 1)
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    out_path, report = write_immutable_json_event(
        args.out_dir,
        "MODEL_NATIVE_SERVE_PARITY",
        report,
    )
    print(json.dumps({
        "decision": report["decision"],
        "n_bars": report["n_bars"],
        "state_block_max_abs_diff": block_max,
        "forward_max_abs_diff": fwd_max,
        "failures": failures,
        "report": str(out_path),
        "runtime_s": report["runtime_s"],
    }, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
