"""Exact immutable smoke-bundle proof consumed by candidate readiness."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_SMOKE_SPLITS,
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
SCHEMA_VERSION = "entry_foundation_smoke_bundle_audit_v1"
PASS_DECISION = "PASS"
DATA_SPLITS = FOUNDATION_AUDIT_SMOKE_SPLITS
PREDICTION_EVIDENCE_SCHEMA_VERSION = (
    "entry_candidate_model_direction_prediction_evidence_v1"
)
BUNDLE_ARTIFACT_KEYS = (
    "bundle_metadata",
    "master_transformer_lock",
    "model_state_dict",
)
INPUT_AUDIT_SCHEMAS = {
    "target": "entry_target_foundation_audit_v1",
    "specialist": "entry_specialist_feature_group_audit_v1",
    "pretrain": "xau_direction_repair_pretrain_audit_v1",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLASS_NAMES = ("LONG", "SHORT", "FLAT")
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
    "minimum_trade_direction_precision",
    "trade_direction_precision_wilson_lower",
    "minimum_trade_precision_wilson_lower",
    "minimum_prediction_rows_per_class",
    "minimum_class_precision_wilson_lower",
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
    "minimum_trade_direction_precision",
    "minimum_trade_precision_wilson_lower",
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
        minimum_trade_precision = float(
            _SMOKE_EDGE_POLICY["min_trade_direction_precision"]
        )
        minimum_trade_wilson = float(
            _SMOKE_EDGE_POLICY["min_trade_precision_wilson_lower"]
        )
        minimum_prediction_rows: int | None = int(
            _SMOKE_EDGE_POLICY["min_prediction_rows_per_class"]
        )
        minimum_class_wilson: float | None = float(
            _SMOKE_EDGE_POLICY["min_class_precision_wilson_lower"]
        )
    else:
        _require(support_scope == "context", f"[{context}_SUPPORT_SCOPE_UNKNOWN]")
        minimum_trade_rows = int(_SMOKE_EDGE_POLICY["min_context_trade_rows"])
        minimum_trade_precision = float(
            _SMOKE_EDGE_POLICY["min_context_trade_direction_precision"]
        )
        minimum_trade_wilson = float(
            _SMOKE_EDGE_POLICY["min_context_trade_precision_wilson_lower"]
        )
        minimum_prediction_rows = None
        minimum_class_wilson = None

    _require(
        direction.get("minimum_trade_rows") == minimum_trade_rows,
        f"[{context}_MINIMUM_TRADE_ROWS_POLICY_INVALID]",
    )
    _exact_policy_float(
        direction.get("minimum_trade_direction_precision"),
        minimum_trade_precision,
        context=f"{context}_MINIMUM_TRADE_DIRECTION_PRECISION",
    )
    _exact_policy_float(
        direction.get("minimum_trade_precision_wilson_lower"),
        minimum_trade_wilson,
        context=f"{context}_MINIMUM_TRADE_WILSON_LOWER",
    )
    _require(
        direction.get("minimum_prediction_rows_per_class")
        == minimum_prediction_rows,
        f"[{context}_MINIMUM_CLASS_SUPPORT_POLICY_INVALID]",
    )
    _require(
        direction.get("minimum_class_precision_wilson_lower")
        == minimum_class_wilson,
        f"[{context}_MINIMUM_CLASS_WILSON_POLICY_INVALID]",
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
        _require(
            precision[name] >= float(_SMOKE_EDGE_POLICY["min_class_precision"]),
            f"[{context}_{name}_PRECISION_BELOW_POLICY]",
        )
        if minimum_class_wilson is not None:
            _require(
                precision_wilson[name] >= minimum_class_wilson,
                f"[{context}_{name}_WILSON_BELOW_POLICY]",
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
    _require(
        trade_precision >= minimum_trade_precision,
        f"[{context}_TRADE_PRECISION_BELOW_POLICY]",
    )
    _require(
        trade_wilson >= minimum_trade_wilson,
        f"[{context}_TRADE_WILSON_BELOW_POLICY]",
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
    _require(
        accuracy >= float(_SMOKE_EDGE_POLICY["min_direction_accuracy"]),
        f"[{context}_ACCURACY_BELOW_POLICY]",
    )
    _require(
        balanced >= float(_SMOKE_EDGE_POLICY["min_balanced_accuracy"]),
        f"[{context}_BALANCED_ACCURACY_BELOW_POLICY]",
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
    minimum_trade_precision = float(
        _SMOKE_EDGE_POLICY["min_context_trade_direction_precision"]
    )
    minimum_trade_wilson = float(
        _SMOKE_EDGE_POLICY["min_context_trade_precision_wilson_lower"]
    )
    _require(
        summary.get("minimum_rows_per_slice") == minimum_rows,
        f"[{context}_MINIMUM_ROWS_POLICY_INVALID]",
    )
    _require(
        summary.get("minimum_trade_rows_per_slice") == minimum_trade_rows,
        f"[{context}_MINIMUM_TRADE_ROWS_POLICY_INVALID]",
    )
    _exact_policy_float(
        summary.get("minimum_trade_direction_precision"),
        minimum_trade_precision,
        context=f"{context}_MINIMUM_TRADE_DIRECTION_PRECISION",
    )
    _exact_policy_float(
        summary.get("minimum_trade_precision_wilson_lower"),
        minimum_trade_wilson,
        context=f"{context}_MINIMUM_TRADE_WILSON_LOWER",
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
    return {
        "decision": PASS_DECISION,
        "failures": [],
        "rows": rows,
        "direction": direction,
        "context_slice_contract": context_slices,
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
            "all_active_heads_live",
            "all_specialists_live",
            "full_stack_live",
            "zero_init_pass_through_absent",
        },
    )
    for key in (
        "all_active_heads_live",
        "all_specialists_live",
        "full_stack_live",
        "zero_init_pass_through_absent",
    ):
        _require(liveness.get(key) is True, f"[{context}_{key.upper()}_UNPROVEN]")

    edge = _zero_failure(
        report.get("edge_contract"),
        context=f"{context}_EDGE_PROOF",
        exact_keys={
            "decision",
            "failures",
            "direction_edge_proven",
            "context_slice_edge_proven",
            "path_quality_edge_proven",
            "bad_path_edge_proven",
        },
    )
    for key in (
        "direction_edge_proven",
        "context_slice_edge_proven",
        "path_quality_edge_proven",
        "bad_path_edge_proven",
    ):
        _require(edge.get(key) is True, f"[{context}_{key.upper()}_UNPROVEN]")

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
        and prediction.get("authoritative") is True,
        f"[{context}_PREDICTION_EVIDENCE_INVALID]",
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
        "prediction_report_json": str(prediction_report),
        "prediction_report_sha256": str(report["prediction_report_sha256"]),
    }
