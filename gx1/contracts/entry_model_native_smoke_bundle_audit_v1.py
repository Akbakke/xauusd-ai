"""Exact immutable smoke-bundle proof consumed by candidate readiness."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_SMOKE_SPLITS,
    FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    foundation_audit_policy_binding,
    foundation_audit_policy_metadata,
    require_foundation_audit_policy_binding,
    require_foundation_audit_report_policy,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    require_model_native_readiness_contract,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ACTION_ORDER,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    require_training_objective_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (
    require_model_direction_decision_contract,
)
SCHEMA_VERSION = "entry_foundation_smoke_bundle_audit_v7"
PASS_DECISION = "PASS"
DATA_SPLITS = FOUNDATION_AUDIT_SMOKE_SPLITS
PREDICTION_EVIDENCE_SCHEMA_VERSION = (
    "entry_candidate_model_direction_prediction_evidence_v3"
)
BUNDLE_ARTIFACT_KEYS = (
    "bundle_commit",
    "bundle_metadata",
    "master_transformer_lock",
    "model_state_dict",
)
# One owner for the pretrain audit schema. It used to be restated in four
# places; the producer moved to v5 while three consumers stayed on v4, which
# made the post-rebuild readiness event and the smoke bundle audit both
# unproducible. This module owns it because train_launch already imports
# from here, so the dependency can only point one way.
PRETRAIN_AUDIT_SCHEMA = "xau_direction_repair_pretrain_audit_v6"
INPUT_AUDIT_SCHEMAS = {
    "target": FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    "specialist": "entry_specialist_feature_group_audit_v1",
    "pretrain": PRETRAIN_AUDIT_SCHEMA,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLASS_NAMES = ENTRY_FITTED_Q_ACTION_ORDER
_SMOKE_EDGE_POLICY = foundation_audit_policy_metadata()["smoke_edge_pockets"]
_DIRECTION_METRIC_KEYS = {
    "decision",
    "failures",
    "rows",
    "accuracy",
    "majority_baseline_accuracy",
    "beats_majority_baseline",
    "balanced_accuracy",
    "support_scope",
    "wilson_confidence_level",
    "wilson_z_score",
    "trade_rows",
    "trade_successes",
    "minimum_trade_rows",
    "trade_coverage",
    "trade_direction_precision",
    "trade_direction_precision_wilson_lower",
    "minimum_prediction_rows_per_class",
    "log_loss",
    "label_counts",
    "prediction_counts",
    "precision",
    "precision_successes",
    "precision_wilson_lower",
    "recall",
    "confusion_matrix",
}
_CONTEXT_SLICE_KEYS = {
    "decision",
    "failures",
    "minimum_rows_per_slice",
    "minimum_trade_rows_per_slice",
    "fields",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _zero_failure(
    value: Any,
    *,
    context: str,
    exact_keys: set[str],
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"[{context}_MISSING]")
    _require(set(value) == exact_keys, f"[{context}_KEY_SET_INVALID]")
    _require(value.get("decision") == PASS_DECISION, f"[{context}_DECISION_INVALID]")
    _require(value.get("failures") == [], f"[{context}_FAILURES_NOT_EMPTY]")
    return value


def _exact_int(value: Any, *, context: str, minimum: int = 0) -> int:
    _require(type(value) is int, f"[{context}_INTEGER_INVALID]")
    normalized = int(value)
    _require(normalized >= minimum, f"[{context}_INTEGER_BELOW_MINIMUM]")
    return normalized


def _finite_float(value: Any, *, context: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"[{context}_NUMBER_INVALID]",
    )
    normalized = float(value)
    _require(math.isfinite(normalized), f"[{context}_NUMBER_NOT_FINITE]")
    return normalized


def _exact_policy_float(value: Any, expected: float, *, context: str) -> float:
    normalized = _finite_float(value, context=context)
    _require(normalized == float(expected), f"[{context}_POLICY_VALUE_INVALID]")
    return normalized


def _metric_equal(value: float, expected: float, *, context: str) -> None:
    _require(
        math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"[{context}_METRIC_INCONSISTENT]",
    )


def _wilson_lower(successes: int, trials: int, *, z_score: float) -> float:
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    radius = z_score * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return float(min(1.0, max(0.0, (centre - radius) / denominator)))


def _class_int_mapping(value: Any, *, context: str) -> dict[str, int]:
    _require(isinstance(value, Mapping), f"[{context}_MAPPING_MISSING]")
    _require(set(value) == set(_CLASS_NAMES), f"[{context}_CLASS_SET_INVALID]")
    return {
        name: _exact_int(value[name], context=f"{context}_{name}")
        for name in _CLASS_NAMES
    }


def _class_float_mapping(value: Any, *, context: str) -> dict[str, float]:
    _require(isinstance(value, Mapping), f"[{context}_MAPPING_MISSING]")
    _require(set(value) == set(_CLASS_NAMES), f"[{context}_CLASS_SET_INVALID]")
    return {
        name: _finite_float(value[name], context=f"{context}_{name}")
        for name in _CLASS_NAMES
    }


def _direction_wilson_contract(
    value: Any,
    *,
    context: str,
    support_scope: str,
) -> dict[str, Any]:
    direction = _zero_failure(
        value,
        context=context,
        exact_keys=_DIRECTION_METRIC_KEYS,
    )
    _require(
        direction.get("support_scope") == support_scope,
        f"[{context}_SUPPORT_SCOPE_INVALID]",
    )
    rows = _exact_int(direction.get("rows"), context=f"{context}_ROWS", minimum=1)
    confidence = _exact_policy_float(
        direction.get("wilson_confidence_level"),
        float(_SMOKE_EDGE_POLICY["wilson_confidence_level"]),
        context=f"{context}_WILSON_CONFIDENCE_LEVEL",
    )
    z_score = _exact_policy_float(
        direction.get("wilson_z_score"),
        float(_SMOKE_EDGE_POLICY["wilson_z_score"]),
        context=f"{context}_WILSON_Z_SCORE",
    )
    _require(0.0 < confidence < 1.0, f"[{context}_WILSON_CONFIDENCE_INVALID]")
    _require(z_score > 0.0, f"[{context}_WILSON_Z_INVALID]")

    if support_scope == "global":
        minimum_trade_rows = int(_SMOKE_EDGE_POLICY["min_trade_rows"])
        minimum_prediction_rows: int | None = int(
            _SMOKE_EDGE_POLICY["min_prediction_rows_per_class"]
        )
    else:
        _require(support_scope == "context", f"[{context}_SUPPORT_SCOPE_UNKNOWN]")
        minimum_trade_rows = int(_SMOKE_EDGE_POLICY["min_context_trade_rows"])
        minimum_prediction_rows = None

    _require(
        direction.get("minimum_trade_rows") == minimum_trade_rows,
        f"[{context}_MINIMUM_TRADE_ROWS_POLICY_INVALID]",
    )
    _require(
        direction.get("minimum_prediction_rows_per_class")
        == minimum_prediction_rows,
        f"[{context}_MINIMUM_CLASS_SUPPORT_POLICY_INVALID]",
    )
    for field, expected in ():
        value_at_field = direction.get(field)
        if expected is None:
            _require(value_at_field is None, f"[{context}_{field.upper()}_INVALID]")
        else:
            _exact_policy_float(
                value_at_field,
                expected,
                context=f"{context}_{field.upper()}",
            )

    label_counts = _class_int_mapping(
        direction.get("label_counts"), context=f"{context}_LABEL_COUNTS"
    )
    prediction_counts = _class_int_mapping(
        direction.get("prediction_counts"),
        context=f"{context}_PREDICTION_COUNTS",
    )
    precision_successes = _class_int_mapping(
        direction.get("precision_successes"),
        context=f"{context}_PRECISION_SUCCESSES",
    )
    precision = _class_float_mapping(
        direction.get("precision"), context=f"{context}_PRECISION"
    )
    precision_wilson = _class_float_mapping(
        direction.get("precision_wilson_lower"),
        context=f"{context}_PRECISION_WILSON_LOWER",
    )
    recall = _class_float_mapping(
        direction.get("recall"), context=f"{context}_RECALL"
    )
    _require(sum(label_counts.values()) == rows, f"[{context}_LABEL_ROWS_INVALID]")
    _require(
        sum(prediction_counts.values()) == rows,
        f"[{context}_PREDICTION_ROWS_INVALID]",
    )

    confusion_raw = direction.get("confusion_matrix")
    _require(
        isinstance(confusion_raw, list) and len(confusion_raw) == len(_CLASS_NAMES),
        f"[{context}_CONFUSION_MATRIX_INVALID]",
    )
    confusion: list[list[int]] = []
    for row_index, raw_row in enumerate(confusion_raw):
        _require(
            isinstance(raw_row, list) and len(raw_row) == len(_CLASS_NAMES),
            f"[{context}_CONFUSION_MATRIX_ROW_INVALID]",
        )
        confusion.append(
            [
                _exact_int(
                    cell,
                    context=f"{context}_CONFUSION_{row_index}_{column_index}",
                )
                for column_index, cell in enumerate(raw_row)
            ]
        )
    for index, name in enumerate(_CLASS_NAMES):
        _require(
            sum(confusion[index]) == label_counts[name],
            f"[{context}_{name}_LABEL_COUNT_INCONSISTENT]",
        )
        _require(
            sum(row[index] for row in confusion) == prediction_counts[name],
            f"[{context}_{name}_PREDICTION_COUNT_INCONSISTENT]",
        )
        _require(
            confusion[index][index] == precision_successes[name],
            f"[{context}_{name}_SUCCESS_COUNT_INCONSISTENT]",
        )
        _require(
            prediction_counts[name] > 0 and label_counts[name] > 0,
            f"[{context}_{name}_CLASS_SUPPORT_MISSING]",
        )
        if minimum_prediction_rows is not None:
            _require(
                prediction_counts[name] >= minimum_prediction_rows,
                f"[{context}_{name}_PREDICTION_SUPPORT_BELOW_POLICY]",
            )
        expected_precision = precision_successes[name] / prediction_counts[name]
        expected_recall = precision_successes[name] / label_counts[name]
        expected_wilson = _wilson_lower(
            precision_successes[name], prediction_counts[name], z_score=z_score
        )
        _metric_equal(
            precision[name], expected_precision, context=f"{context}_{name}_PRECISION"
        )
        _metric_equal(recall[name], expected_recall, context=f"{context}_{name}_RECALL")
        _metric_equal(
            precision_wilson[name],
            expected_wilson,
            context=f"{context}_{name}_PRECISION_WILSON_LOWER",
        )

    trade_rows = _exact_int(
        direction.get("trade_rows"), context=f"{context}_TRADE_ROWS"
    )
    trade_successes = _exact_int(
        direction.get("trade_successes"), context=f"{context}_TRADE_SUCCESSES"
    )
    _require(
        trade_rows == prediction_counts["LONG"] + prediction_counts["SHORT"],
        f"[{context}_TRADE_ROWS_INCONSISTENT]",
    )
    _require(
        trade_successes
        == precision_successes["LONG"] + precision_successes["SHORT"],
        f"[{context}_TRADE_SUCCESSES_INCONSISTENT]",
    )
    _require(
        minimum_trade_rows <= trade_rows <= rows,
        f"[{context}_TRADE_SUPPORT_BELOW_POLICY]",
    )
    _require(trade_successes <= trade_rows, f"[{context}_TRADE_SUCCESSES_INVALID]")
    expected_trade_precision = trade_successes / trade_rows
    expected_trade_wilson = _wilson_lower(
        trade_successes, trade_rows, z_score=z_score
    )
    trade_precision = _finite_float(
        direction.get("trade_direction_precision"),
        context=f"{context}_TRADE_DIRECTION_PRECISION",
    )
    trade_wilson = _finite_float(
        direction.get("trade_direction_precision_wilson_lower"),
        context=f"{context}_TRADE_DIRECTION_PRECISION_WILSON_LOWER",
    )
    _metric_equal(
        trade_precision,
        expected_trade_precision,
        context=f"{context}_TRADE_DIRECTION_PRECISION",
    )
    _metric_equal(
        trade_wilson,
        expected_trade_wilson,
        context=f"{context}_TRADE_DIRECTION_PRECISION_WILSON_LOWER",
    )

    accuracy = _finite_float(direction.get("accuracy"), context=f"{context}_ACCURACY")
    majority = _finite_float(
        direction.get("majority_baseline_accuracy"),
        context=f"{context}_MAJORITY_BASELINE_ACCURACY",
    )
    balanced = _finite_float(
        direction.get("balanced_accuracy"),
        context=f"{context}_BALANCED_ACCURACY",
    )
    trace = sum(confusion[index][index] for index in range(len(_CLASS_NAMES)))
    _metric_equal(accuracy, trace / rows, context=f"{context}_ACCURACY")
    _metric_equal(
        majority,
        max(label_counts.values()) / rows,
        context=f"{context}_MAJORITY_BASELINE_ACCURACY",
    )
    expected_balanced = sum(recall.values()) / len(_CLASS_NAMES)
    _metric_equal(
        balanced, expected_balanced, context=f"{context}_BALANCED_ACCURACY"
    )
    _require(
        direction.get("beats_majority_baseline") is True and accuracy > majority,
        f"[{context}_MAJORITY_BASELINE_EDGE_UNPROVEN]",
    )
    trade_coverage = _finite_float(
        direction.get("trade_coverage"), context=f"{context}_TRADE_COVERAGE"
    )
    _metric_equal(trade_coverage, trade_rows / rows, context=f"{context}_TRADE_COVERAGE")
    _require(
        _finite_float(direction.get("log_loss"), context=f"{context}_LOG_LOSS")
        >= 0.0,
        f"[{context}_LOG_LOSS_INVALID]",
    )
    return dict(direction)


def _context_wilson_contract(
    value: Any,
    *,
    context: str,
    expected_rows: int,
) -> dict[str, Any]:
    summary = _zero_failure(
        value,
        context=context,
        exact_keys=_CONTEXT_SLICE_KEYS,
    )
    minimum_rows = int(_SMOKE_EDGE_POLICY["min_rows_per_context_slice"])
    minimum_trade_rows = int(_SMOKE_EDGE_POLICY["min_context_trade_rows"])
    _require(
        summary.get("minimum_rows_per_slice") == minimum_rows,
        f"[{context}_MINIMUM_ROWS_POLICY_INVALID]",
    )
    _require(
        summary.get("minimum_trade_rows_per_slice") == minimum_trade_rows,
        f"[{context}_MINIMUM_TRADE_ROWS_POLICY_INVALID]",
    )
    fields = summary.get("fields")
    expected_fields = tuple(_SMOKE_EDGE_POLICY["context_fields"])
    _require(isinstance(fields, Mapping), f"[{context}_FIELDS_MISSING]")
    _require(set(fields) == set(expected_fields), f"[{context}_FIELD_SET_INVALID]")
    for field in expected_fields:
        field_row = fields[field]
        _require(isinstance(field_row, Mapping), f"[{context}_{field}_FIELD_INVALID]")
        _require(
            set(field_row) == {"values", "slices"},
            f"[{context}_{field}_FIELD_KEYS_INVALID]",
        )
        values = field_row.get("values")
        slices = field_row.get("slices")
        _require(
            isinstance(values, list)
            and all(isinstance(item, str) and item for item in values),
            f"[{context}_{field}_VALUES_INVALID]",
        )
        _require(
            values == sorted(set(values)), f"[{context}_{field}_VALUES_NOT_CANONICAL]"
        )
        if field == "session":
            _require(
                values == sorted(_SMOKE_EDGE_POLICY["expected_sessions"]),
                f"[{context}_SESSION_SET_INVALID]",
            )
        if field == "vol_regime":
            _require(len(values) >= 2, f"[{context}_VOL_REGIME_SUPPORT_INVALID]")
        _require(isinstance(slices, Mapping), f"[{context}_{field}_SLICES_MISSING]")
        _require(set(slices) == set(values), f"[{context}_{field}_SLICE_SET_INVALID]")
        field_rows = 0
        for slice_name in values:
            slice_contract = _direction_wilson_contract(
                slices[slice_name],
                context=f"{context}_{field}_{slice_name}",
                support_scope="context",
            )
            slice_rows = _exact_int(
                slice_contract.get("rows"),
                context=f"{context}_{field}_{slice_name}_ROWS",
                minimum=minimum_rows,
            )
            field_rows += slice_rows
        _require(field_rows == expected_rows, f"[{context}_{field}_ROW_TOTAL_INVALID]")
    return dict(summary)


def _turning_point_contract(value: Any, *, context: str) -> dict[str, Any]:
    expected_keys = {
        "decision",
        "failures",
        "policy",
        "layout",
        "target_alignment",
        "near_turn_pockets",
        "live_direction_rule_authority",
    }
    proof = _zero_failure(value, context=context, exact_keys=expected_keys)
    policy = _SMOKE_EDGE_POLICY["turning_point_evidence"]
    _require(proof.get("policy") == policy, f"[{context}_POLICY_INVALID]")
    _require(
        proof.get("live_direction_rule_authority") is False,
        f"[{context}_LIVE_RULE_AUTHORITY_INVALID]",
    )
    expected_layout = model_native_aux_target_contract_metadata()[
        "turning_point_timing"
    ]["layout"]
    _require(proof.get("layout") == expected_layout, f"[{context}_LAYOUT_INVALID]")

    alignment = proof.get("target_alignment")
    _require(
        isinstance(alignment, list) and len(alignment) == len(expected_layout),
        f"[{context}_ALIGNMENT_INVALID]",
    )
    min_spearman = float(policy["min_prediction_target_spearman"])
    max_mae = float(policy["max_prediction_target_mae"])
    for index, expected in enumerate(expected_layout):
        row = alignment[index]
        _require(isinstance(row, Mapping), f"[{context}_ALIGNMENT_ROW_INVALID]")
        _require(
            set(row) == set(expected) | {"spearman", "mae", "decision", "failures"},
            f"[{context}_ALIGNMENT_ROW_KEYS_INVALID]",
        )
        for key, expected_value in expected.items():
            _require(
                row.get(key) == expected_value,
                f"[{context}_ALIGNMENT_LAYOUT_INVALID]",
            )
        _require(
            row.get("decision") == PASS_DECISION and row.get("failures") == [],
            f"[{context}_ALIGNMENT_NOT_PASS]",
        )
        _require(
            _finite_float(row.get("spearman"), context=f"{context}_SPEARMAN")
            >= min_spearman,
            f"[{context}_SPEARMAN_BELOW_POLICY]",
        )
        _require(
            0.0
            <= _finite_float(row.get("mae"), context=f"{context}_MAE")
            <= max_mae,
            f"[{context}_MAE_ABOVE_POLICY]",
        )

    pockets = proof.get("near_turn_pockets")
    _require(
        isinstance(pockets, Mapping) and set(pockets) == {"BOTTOM", "TOP"},
        f"[{context}_POCKET_SET_INVALID]",
    )
    pocket_keys = {
        "decision",
        "failures",
        "model_direction",
        "timing_output_index",
        "evaluation_horizon_bars",
        "near_turn_max_fraction",
        "rows",
        "direction_successes",
        "direction_precision",
        "direction_precision_wilson_lower",
        "timing_successes",
        "timing_precision",
        "timing_precision_wilson_lower",
    }
    z_score = float(_SMOKE_EDGE_POLICY["wilson_z_score"])
    for turn, direction in (("BOTTOM", "LONG"), ("TOP", "SHORT")):
        row = _zero_failure(
            pockets[turn],
            context=f"{context}_{turn}",
            exact_keys=pocket_keys,
        )
        _require(
            row.get("model_direction") == direction,
            f"[{context}_{turn}_DIRECTION_INVALID]",
        )
        _require(
            row.get("evaluation_horizon_bars")
            == int(policy["evaluation_horizon_bars"]),
            f"[{context}_{turn}_HORIZON_INVALID]",
        )
        _exact_policy_float(
            row.get("near_turn_max_fraction"),
            float(policy["near_turn_max_fraction"]),
            context=f"{context}_{turn}_MAX_FRACTION",
        )
        rows = _exact_int(
            row.get("rows"),
            context=f"{context}_{turn}_ROWS",
            minimum=int(policy["min_near_turn_trade_rows_per_side"]),
        )
        direction_successes = _exact_int(
            row.get("direction_successes"),
            context=f"{context}_{turn}_DIRECTION_SUCCESSES",
        )
        timing_successes = _exact_int(
            row.get("timing_successes"),
            context=f"{context}_{turn}_TIMING_SUCCESSES",
        )
        _require(
            direction_successes <= rows and timing_successes <= rows,
            f"[{context}_{turn}_SUCCESSES_INVALID]",
        )
        direction_precision = direction_successes / rows
        timing_precision = timing_successes / rows
        direction_wilson = _wilson_lower(
            direction_successes, rows, z_score=z_score
        )
        timing_wilson = _wilson_lower(timing_successes, rows, z_score=z_score)
        for name, expected_value in (
            ("direction_precision", direction_precision),
            ("timing_precision", timing_precision),
            ("direction_precision_wilson_lower", direction_wilson),
            ("timing_precision_wilson_lower", timing_wilson),
        ):
            _metric_equal(
                _finite_float(row.get(name), context=f"{context}_{turn}_{name}"),
                expected_value,
                context=f"{context}_{turn}_{name}",
            )
        _require(
            direction_precision >= float(policy["min_near_turn_direction_precision"])
            and direction_wilson
            >= float(policy["min_near_turn_precision_wilson_lower"]),
            f"[{context}_{turn}_DIRECTION_EDGE_UNPROVEN]",
        )
        _require(
            timing_precision >= float(policy["min_near_turn_timing_precision"])
            and timing_wilson
            >= float(policy["min_near_turn_timing_precision_wilson_lower"]),
            f"[{context}_{turn}_TIMING_EDGE_UNPROVEN]",
        )
    return dict(proof)



def _split_wilson_contract(value: Any, *, split: str, context: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"[{context}_{split}_SPLIT_MISSING]")
    _require(value.get("decision") == PASS_DECISION, f"[{context}_{split}_DECISION_INVALID]")
    _require(value.get("failures") == [], f"[{context}_{split}_FAILURES_NOT_EMPTY]")
    rows = _exact_int(value.get("rows"), context=f"{context}_{split}_ROWS", minimum=1)
    direction = _direction_wilson_contract(
        value.get("direction"),
        context=f"{context}_{split}_GLOBAL_DIRECTION",
        support_scope="global",
    )
    _require(direction.get("rows") == rows, f"[{context}_{split}_DIRECTION_ROWS_INVALID]")
    context_slices = _context_wilson_contract(
        value.get("context_slice_contract"),
        context=f"{context}_{split}_CONTEXT",
        expected_rows=rows,
    )
    turning_point = _turning_point_contract(
        value.get("turning_point_evidence"),
        context=f"{context}_{split}_TURNING_POINT",
    )
    return {
        "decision": PASS_DECISION,
        "failures": [],
        "rows": rows,
        "direction": direction,
        "context_slice_contract": context_slices,
        "turning_point_evidence": turning_point,
    }


def _artifact_binding(value: Any, *, context: str) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"[{context}_ARTIFACT_BINDING_MISSING]")
    _require(set(value) == {"path", "sha256"}, f"[{context}_ARTIFACT_BINDING_KEYS_INVALID]")
    path = Path(str(value.get("path") or "")).expanduser()
    digest = str(value.get("sha256") or "").lower()
    _require(path.is_absolute(), f"[{context}_ARTIFACT_PATH_NOT_ABSOLUTE]")
    _require(_SHA256_RE.fullmatch(digest) is not None, f"[{context}_ARTIFACT_SHA256_INVALID]")
    return {"path": str(path), "sha256": digest}


def _input_audit_binding(value: Any, *, name: str, context: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"[{context}_{name.upper()}_AUDIT_MISSING]")
    expected_keys = {"path", "sha256", "schema_version", "decision", "failures"}
    if name in {"target", "specialist"}:
        expected_keys.update(foundation_audit_policy_binding())
        expected_keys.add("data_splits")
        expected_keys.add("foundation_audit_policy_enforcement")
    _require(
        set(value) == expected_keys,
        f"[{context}_{name.upper()}_AUDIT_KEYS_INVALID]",
    )
    binding = _artifact_binding(
        {"path": value.get("path"), "sha256": value.get("sha256")},
        context=f"{context}_{name.upper()}_AUDIT",
    )
    _require(
        value.get("schema_version") == INPUT_AUDIT_SCHEMAS[name],
        f"[{context}_{name.upper()}_AUDIT_SCHEMA_INVALID]",
    )
    _require(
        value.get("decision") == PASS_DECISION and value.get("failures") == [],
        f"[{context}_{name.upper()}_AUDIT_NOT_PASS]",
    )
    normalized = {
        **binding,
        "schema_version": INPUT_AUDIT_SCHEMAS[name],
        "decision": PASS_DECISION,
        "failures": [],
    }
    if name in {"target", "specialist"}:
        normalized.update(
            require_foundation_audit_report_policy(
                value,
                audit_kind=name,
                context=f"{context}_{name.upper()}_AUDIT",
            )
        )
    return normalized


def require_smoke_bundle_audit_contract(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Require one exact seq513 PASS proof; no historical aliases are read."""

    _require(isinstance(value, Mapping), f"[{context}_SMOKE_AUDIT_MISSING]")
    report = dict(value)
    _require(report.get("schema_version") == SCHEMA_VERSION, f"[{context}_SCHEMA_INVALID]")
    _require(report.get("decision") == PASS_DECISION, f"[{context}_DECISION_INVALID]")
    _require(report.get("failures") == [], f"[{context}_FAILURES_NOT_EMPTY]")
    policy_binding = require_foundation_audit_policy_binding(
        report,
        context=f"{context}_SMOKE_AUDIT",
    )
    _require(
        report.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE,
        f"[{context}_CONTRACT_MODE_INVALID]",
    )
    _require(
        report.get("sequence_length") == MODEL_NATIVE_SEQ_LEN,
        f"[{context}_SEQUENCE_LENGTH_INVALID]",
    )
    _require(
        report.get("signal_dim") == MODEL_NATIVE_SIGNAL_DIM,
        f"[{context}_SIGNAL_DIM_INVALID]",
    )
    _require(
        tuple(report.get("data_splits") or ()) == DATA_SPLITS,
        f"[{context}_DATA_SPLITS_INVALID]",
    )
    bundle_dir = Path(str(report.get("bundle_dir") or "")).expanduser()
    dataset_dir = Path(str(report.get("dataset_dir") or "")).expanduser()
    _require(bundle_dir.is_absolute(), f"[{context}_BUNDLE_DIR_INVALID]")
    _require(dataset_dir.is_absolute(), f"[{context}_DATASET_DIR_INVALID]")

    readiness = require_model_native_readiness_contract(
        report.get("model_native_readiness_contract"),
        context=f"{context}_SMOKE_AUDIT",
    )
    direction = require_model_direction_decision_contract(
        {"direction_decision_contract": report.get("direction_decision_contract")},
        context=f"{context} smoke audit",
    )

    artifacts_raw = report.get("bundle_artifacts")
    _require(isinstance(artifacts_raw, Mapping), f"[{context}_BUNDLE_ARTIFACTS_MISSING]")
    _require(
        set(artifacts_raw) == set(BUNDLE_ARTIFACT_KEYS),
        f"[{context}_BUNDLE_ARTIFACT_SET_INVALID]",
    )
    artifacts = {
        name: _artifact_binding(artifacts_raw[name], context=f"{context}_{name.upper()}")
        for name in BUNDLE_ARTIFACT_KEYS
    }
    expected_paths = {
        "bundle_commit": bundle_dir / "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json",
        "bundle_metadata": bundle_dir / "bundle_metadata.json",
        "master_transformer_lock": bundle_dir / "MASTER_TRANSFORMER_LOCK.json",
        "model_state_dict": bundle_dir / "model_state_dict.pt",
    }
    for name, expected in expected_paths.items():
        _require(
            Path(artifacts[name]["path"]).resolve() == expected.resolve(),
            f"[{context}_{name.upper()}_PATH_INVALID]",
        )

    input_audits_raw = report.get("input_audits")
    _require(isinstance(input_audits_raw, Mapping), f"[{context}_INPUT_AUDITS_MISSING]")
    _require(
        set(input_audits_raw) == set(INPUT_AUDIT_SCHEMAS),
        f"[{context}_INPUT_AUDIT_SET_INVALID]",
    )
    input_audits = {
        name: _input_audit_binding(
            input_audits_raw[name],
            name=name,
            context=context,
        )
        for name in INPUT_AUDIT_SCHEMAS
    }

    objective_proof = _zero_failure(
        report.get("model_native_training_objective_contract"),
        context=f"{context}_TRAINING_OBJECTIVE_PROOF",
        exact_keys={
            "decision",
            "failures",
            "meta_lock_exact",
            "objective",
            "metadata_path",
            "metadata_sha256",
            "lock_path",
            "lock_sha256",
        },
    )
    _require(
        objective_proof.get("meta_lock_exact") is True,
        f"[{context}_TRAINING_OBJECTIVE_SPLIT_BRAIN]",
    )
    objective = require_training_objective_contract(
        objective_proof.get("objective"),
        context=f"{context}_SMOKE_AUDIT",
    )
    _require(
        objective_proof.get("metadata_path") == artifacts["bundle_metadata"]["path"]
        and objective_proof.get("metadata_sha256")
        == artifacts["bundle_metadata"]["sha256"],
        f"[{context}_TRAINING_OBJECTIVE_METADATA_BINDING_INVALID]",
    )
    _require(
        objective_proof.get("lock_path")
        == artifacts["master_transformer_lock"]["path"]
        and objective_proof.get("lock_sha256")
        == artifacts["master_transformer_lock"]["sha256"],
        f"[{context}_TRAINING_OBJECTIVE_LOCK_BINDING_INVALID]",
    )

    head = _zero_failure(
        report.get("head_contract"),
        context=f"{context}_HEAD_PROOF",
        exact_keys={"decision", "failures", "active_heads", "blocked_heads"},
    )
    _require(
        tuple(head.get("active_heads") or ()) == MODEL_NATIVE_ACTIVE_HEADS,
        f"[{context}_ACTIVE_HEAD_SET_INVALID]",
    )
    _require(
        tuple(head.get("blocked_heads") or ()) == MODEL_NATIVE_BLOCKED_HEADS,
        f"[{context}_BLOCKED_HEAD_SET_INVALID]",
    )

    specialist = _zero_failure(
        report.get("specialist_contract"),
        context=f"{context}_SPECIALIST_PROOF",
        exact_keys={
            "decision",
            "failures",
            "specialists",
            "gate_liveness_proven",
        },
    )
    _require(
        tuple(specialist.get("specialists") or ())
        == MODEL_NATIVE_REQUIRED_SPECIALISTS,
        f"[{context}_SPECIALIST_SET_INVALID]",
    )
    _require(
        specialist.get("gate_liveness_proven") is True,
        f"[{context}_SPECIALIST_LIVENESS_UNPROVEN]",
    )

    liveness = _zero_failure(
        report.get("liveness_contract"),
        context=f"{context}_LIVENESS_PROOF",
        exact_keys={
            "decision",
            "failures",
            "all_active_head_predictions_live",
            "all_specialist_gates_live",
            "strict_bundle_components_live",
        },
    )
    for key in (
        "all_active_head_predictions_live",
        "all_specialist_gates_live",
        "strict_bundle_components_live",
    ):
        _require(liveness.get(key) is True, f"[{context}_{key.upper()}_UNPROVEN]")

    # The six per-family `*_edge_proven` booleans are retired with the
    # handwritten edge scorebook, and `offline_rl_edge_proven` named a contract
    # that no longer exists. The sole producer
    # (`audit_entry_foundation_smoke_bundle_v1`) now emits the fitted-Q edge
    # block, so the validator binds exactly that.
    edge = _zero_failure(
        report.get("edge_contract"),
        context=f"{context}_EDGE_PROOF",
        exact_keys={
            "decision",
            "failures",
            "raw_entry_q_structure_proven",
            "production_economics_ready",
            "edge_claim_allowed",
        },
    )
    _require(
        edge.get("raw_entry_q_structure_proven") is True,
        f"[{context}_RAW_ENTRY_Q_STRUCTURE_UNPROVEN]",
    )

    splits_raw = report.get("splits")
    _require(isinstance(splits_raw, Mapping), f"[{context}_SPLITS_MISSING]")
    _require(set(splits_raw) == set(DATA_SPLITS), f"[{context}_SPLIT_SET_INVALID]")
    splits = {
        split: _split_wilson_contract(
            splits_raw[split], split=split, context=f"{context}_SMOKE_AUDIT"
        )
        for split in DATA_SPLITS
    }

    prediction = report.get("prediction_evidence")
    _require(isinstance(prediction, Mapping), f"[{context}_PREDICTION_EVIDENCE_MISSING]")
    _require(
        prediction.get("schema_version") == PREDICTION_EVIDENCE_SCHEMA_VERSION
        and prediction.get("evidence_stage") == "pre_calibration"
        and prediction.get("authoritative") is False
        and prediction.get("runtime_head_evidence_authoritative") is False,
        f"[{context}_PREDICTION_EVIDENCE_INVALID]",
    )
    _require(
        prediction.get("splits") == list(DATA_SPLITS),
        f"[{context}_PREDICTION_EVIDENCE_SPLITS_INVALID]",
    )
    prediction_models = prediction.get("models")
    _require(
        isinstance(prediction_models, list)
        and len(prediction_models) == 1
        and isinstance(prediction_models[0], str)
        and bool(prediction_models[0]),
        f"[{context}_PREDICTION_EVIDENCE_MODEL_INVALID]",
    )
    _require(
        _SHA256_RE.fullmatch(str(prediction.get("sha256") or "")) is not None,
        f"[{context}_PREDICTION_EVIDENCE_SHA256_INVALID]",
    )
    prediction_path = Path(str(prediction.get("path") or "")).expanduser()
    _require(
        prediction_path.is_absolute(),
        f"[{context}_PREDICTION_EVIDENCE_PATH_INVALID]",
    )
    _require(
        report.get("prediction_evidence_stage") == "pre_calibration",
        f"[{context}_PREDICTION_EVIDENCE_STAGE_INVALID]",
    )
    prediction_report = Path(
        str(report.get("prediction_report_json") or "")
    ).expanduser()
    _require(
        prediction_report.is_absolute(),
        f"[{context}_PREDICTION_REPORT_PATH_INVALID]",
    )
    _require(
        _SHA256_RE.fullmatch(str(report.get("prediction_report_sha256") or ""))
        is not None,
        f"[{context}_PREDICTION_REPORT_SHA256_INVALID]",
    )
    _require(
        report.get("promotion_shadow_live_allowed") is False
        and report.get("activation_authority") is False,
        f"[{context}_ACTIVATION_AUTHORITY_FORBIDDEN]",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        **policy_binding,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "sequence_length": MODEL_NATIVE_SEQ_LEN,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_dir": str(bundle_dir),
        "dataset_dir": str(dataset_dir),
        "data_splits": list(DATA_SPLITS),
        "model_native_readiness_contract": readiness,
        "direction_decision_contract": direction,
        "bundle_artifacts": artifacts,
        "input_audits": input_audits,
        "model_native_training_objective_contract": dict(objective_proof),
        "model_native_training_objective": objective,
        "head_contract": dict(head),
        "specialist_contract": dict(specialist),
        "liveness_contract": dict(liveness),
        "edge_contract": dict(edge),
        "splits": splits,
        "prediction_evidence": dict(prediction),
        "prediction_evidence_stage": "pre_calibration",
        "prediction_report_json": str(prediction_report),
        "prediction_report_sha256": str(report["prediction_report_sha256"]),
    }
