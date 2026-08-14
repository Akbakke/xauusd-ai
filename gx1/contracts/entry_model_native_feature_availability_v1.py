"""TRAIN-only availability evidence for the model-native candidate pool.

This contract deliberately does not estimate feature usefulness. Every
code-owned candidate remains available to the learned model. Exact TRAIN
constants or duplicate columns invalidate the artifact so their owner must be
repaired or the field explicitly retired in code; they are never silently
trimmed from one dataset.

Target association is persisted as read-only diagnostics and cannot affect
availability, ordering or exclusion.  A downstream signal/model dimension
contract must separately prove that every available field is reachable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
)


SCHEMA_VERSION = "entry_model_native_feature_availability_v1"
FIT_CLOCK = "M5"
CONSUMER_CLOCKS = ("M1", "M5")
SEMANTIC_TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4", "D1")
GLOBAL_TIMEFRAME_TAG = "LOCAL_OR_CROSS_CLOCK"

SELECTION_POLICY = {
    "authority": "train_feature_values_only",
    "available_rule": "all_code_owned_candidates_jointly_available",
    "constant_policy": "fail_closed_owner_repair_or_code_retirement_required",
    "duplicate_policy": "fail_closed_owner_repair_or_code_retirement_required",
    "fixed_top_k": False,
    "score_cutoff": False,
    "family_quota": False,
    "target_scores_affect_selection": False,
    "rare_event_frequency_cutoff": False,
}

SEMANTICS = {
    "availability": "eligible_for_joint_model_input",
    "reachability": "requires_exact_manifest_and_model_dimension_binding",
    "learned_usefulness": "downstream_model_training_and_held_out_ablation_only",
    "removal": "explicit_code_contract_change_only_not_dataset_selection",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASONS = {"available"}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray, *, domain: bytes) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _semantic_timeframes(name: str) -> list[str]:
    lowered = name.lower()
    tags: list[str] = []
    for timeframe in SEMANTIC_TIMEFRAMES:
        token = timeframe.lower()
        optional_legacy_prefix = "(?:v1)?" if token in {"h1", "h4"} else ""
        if re.search(
            rf"(?:^|[._]){optional_legacy_prefix}{token}(?=$|[._])",
            lowered,
        ):
            tags.append(timeframe)
    return tags or [GLOBAL_TIMEFRAME_TAG]


def _abs_spearman(values: np.ndarray, target: np.ndarray) -> float | None:
    both = np.isfinite(values) & np.isfinite(target)
    if int(both.sum()) < 2:
        return None
    feature = pd.Series(values[both]).rank(method="average").to_numpy()
    outcome = pd.Series(target[both]).rank(method="average").to_numpy()
    if np.all(feature == feature[0]) or np.all(outcome == outcome[0]):
        return None
    with np.errstate(invalid="ignore"):
        rho = float(np.corrcoef(feature, outcome)[0, 1])
    return abs(rho) if math.isfinite(rho) else None


def _coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    values: Sequence[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in values:
        matching = [
            row
            for row in rows
            if (
                row[key] == value
                if isinstance(row[key], str)
                else value in row[key]
            )
        ]
        result[value] = {
            "candidate": len(matching),
            "available": sum(row["decision"] == "available" for row in matching),
            "excluded_constant": sum(
                row["decision"] == "train_constant" for row in matching
            ),
            "excluded_exact_duplicate": sum(
                row["decision"] == "exact_duplicate" for row in matching
            ),
        }
    return result


def fit_feature_availability_contract(
    *,
    matrix: np.ndarray,
    names: Sequence[str],
    times: Sequence[Any],
    train_start: Any,
    train_end: Any,
    diagnostic_target: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit exact availability from ordered TRAIN feature values only."""

    ordered_names = [str(name) for name in names]
    if (
        not ordered_names
        or len(ordered_names) != len(set(ordered_names))
        or any(not name or name != name.strip() for name in ordered_names)
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_CANDIDATE_POOL_INVALID")
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(ordered_names):
        raise RuntimeError("FEATURE_AVAILABILITY_MATRIX_SHAPE_INVALID")
    time_index = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="raise"))
    if (
        len(time_index) != values.shape[0]
        or not time_index.is_monotonic_increasing
        or not time_index.is_unique
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_CLOCK_INVALID")
    fit_start = pd.Timestamp(train_start)
    fit_end = pd.Timestamp(train_end)
    if fit_start.tzinfo is None or fit_end.tzinfo is None:
        raise RuntimeError("FEATURE_AVAILABILITY_TRAIN_WINDOW_NOT_UTC")
    fit_start = fit_start.tz_convert("UTC")
    fit_end = fit_end.tz_convert("UTC")
    if fit_start >= fit_end or (time_index > fit_end).any():
        raise RuntimeError("FEATURE_AVAILABILITY_FUTURE_OR_WINDOW_INVALID")
    fit_mask = np.asarray(
        (time_index >= fit_start) & (time_index <= fit_end), dtype=np.bool_
    )
    if int(fit_mask.sum()) < 2:
        raise RuntimeError("FEATURE_AVAILABILITY_TRAIN_ROWS_INSUFFICIENT")
    train_values = np.ascontiguousarray(values[fit_mask], dtype=np.float32)
    if not bool(np.isfinite(train_values).all()):
        raise RuntimeError("FEATURE_AVAILABILITY_TRAIN_VALUES_NONFINITE")

    if diagnostic_target is None:
        target = np.full(values.shape[0], np.nan, dtype=np.float64)
    else:
        target = np.asarray(diagnostic_target, dtype=np.float64)
        if target.shape != (values.shape[0],):
            raise RuntimeError("FEATURE_AVAILABILITY_DIAGNOSTIC_TARGET_SHAPE_INVALID")
    train_target = target[fit_mask]
    target_finite = np.isfinite(train_target)
    target_sha256 = None
    if bool(target_finite.any()):
        target_digest = hashlib.sha256()
        target_digest.update(b"entry_feature_availability_diagnostic_target_v1\0")
        target_digest.update(
            np.ascontiguousarray(
                time_index[fit_mask].asi8[target_finite], dtype=np.int64
            ).tobytes()
        )
        target_digest.update(
            np.ascontiguousarray(
                train_target[target_finite], dtype=np.float64
            ).tobytes()
        )
        target_sha256 = target_digest.hexdigest()

    column_values: dict[str, np.ndarray] = {}
    column_hashes: dict[str, str] = {}
    hash_representatives: dict[str, str] = {}
    duplicate_of: dict[str, str] = {}
    for index, name in enumerate(ordered_names):
        column = np.ascontiguousarray(train_values[:, index], dtype=np.float32).copy()
        column[column == 0.0] = 0.0  # canonicalize signed zero
        column_values[name] = column
        value_sha256 = _array_sha256(
            column,
            domain=b"entry_feature_availability_column_v1\0",
        )
        column_hashes[name] = value_sha256
        representative = hash_representatives.get(value_sha256)
        if representative is None:
            hash_representatives[value_sha256] = name
        elif np.array_equal(column, column_values[representative]):
            duplicate_of[name] = representative
        else:
            raise RuntimeError("FEATURE_AVAILABILITY_COLUMN_HASH_COLLISION")

    constant_names = [
        name
        for name in ordered_names
        if bool(np.all(column_values[name] == column_values[name][0]))
    ]
    if constant_names or duplicate_of:
        evidence = {
            "train_constant": constant_names,
            "exact_duplicate": [
                {"name": name, "exact_duplicate_of": representative}
                for name, representative in duplicate_of.items()
            ],
        }
        raise RuntimeError(
            "FEATURE_AVAILABILITY_NONLIVE_CODE_OWNED_CANDIDATES: "
            + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        )

    feature_rows: list[dict[str, Any]] = []
    for name in ordered_names:
        column = column_values[name]
        decision = "available"
        exact_duplicate_of = None
        owner = classify_entry_specialist_feature(name)
        if owner not in MODEL_NATIVE_TRAINING_SPECIALISTS:
            raise RuntimeError(
                f"FEATURE_AVAILABILITY_OWNER_INVALID: feature={name} owner={owner}"
            )
        feature_rows.append(
            {
                "name": name,
                "owner": owner,
                "semantic_timeframes": _semantic_timeframes(name),
                "decision": decision,
                "exclusion_reason": None if decision == "available" else decision,
                "exact_duplicate_of": exact_duplicate_of,
                "train_value_sha256": column_hashes[name],
                "train_rows": int(column.size),
                "finite_rows": int(np.isfinite(column).sum()),
                "distinct_value_count": int(np.unique(column).size),
                "nonzero_rows": int(np.count_nonzero(column)),
                "observed_min": float(column.min()),
                "observed_max": float(column.max()),
                "diagnostic_abs_spearman": _abs_spearman(column, train_target),
            }
        )

    family_coverage = _coverage(
        feature_rows,
        key="owner",
        values=MODEL_NATIVE_TRAINING_SPECIALISTS,
    )
    starved = [name for name, row in family_coverage.items() if row["candidate"] == 0]
    if starved:
        raise RuntimeError(
            f"FEATURE_AVAILABILITY_CANDIDATE_FAMILY_STARVED: {starved}"
        )
    timeframe_coverage = _coverage(
        feature_rows,
        key="semantic_timeframes",
        values=(*SEMANTIC_TIMEFRAMES, GLOBAL_TIMEFRAME_TAG),
    )
    available = [row["name"] for row in feature_rows if row["decision"] == "available"]
    excluded = [
        {
            "name": row["name"],
            "reason": row["decision"],
            "exact_duplicate_of": row["exact_duplicate_of"],
        }
        for row in feature_rows
        if row["decision"] != "available"
    ]
    train_times_ns = np.ascontiguousarray(time_index[fit_mask].asi8, dtype=np.int64)
    matrix_digest = hashlib.sha256()
    matrix_digest.update(b"entry_feature_availability_train_matrix_v1\0")
    matrix_digest.update(train_times_ns.tobytes())
    for name in ordered_names:
        matrix_digest.update(name.encode("utf-8") + b"\0")
        matrix_digest.update(bytes.fromhex(column_hashes[name]))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fit_scope": "train_only",
        "fit_clock": FIT_CLOCK,
        "consumer_clocks": list(CONSUMER_CLOCKS),
        "train_start_utc": fit_start.isoformat(),
        "train_end_utc": fit_end.isoformat(),
        "train_row_count": int(fit_mask.sum()),
        "history_rows_excluded_from_fit": int((~fit_mask).sum()),
        "source_time_max_utc": time_index.max().isoformat(),
        "validation_rows_used": False,
        "test_rows_used": False,
        "future_rows_used": False,
        "selection_policy": dict(SELECTION_POLICY),
        "semantics": dict(SEMANTICS),
        "candidate_pool": ordered_names,
        "candidate_pool_count": len(ordered_names),
        "candidate_pool_sha256": _json_sha256(ordered_names),
        "train_matrix_sha256": matrix_digest.hexdigest(),
        "diagnostic_target_sha256": target_sha256,
        "diagnostic_target_affects_selection": False,
        "features": feature_rows,
        "available_features": available,
        "available_feature_count": len(available),
        "excluded_features": excluded,
        "excluded_feature_count": len(excluded),
        "family_coverage": family_coverage,
        "semantic_timeframe_coverage": timeframe_coverage,
        "selection_sha256": _json_sha256(
            {"available": available, "excluded": excluded}
        ),
        "reachability_decision": "REQUIRES_EXACT_SIGNAL_DIMENSION_BINDING",
    }
    payload["contract_sha256"] = _json_sha256(payload)
    return require_feature_availability_contract(payload)


def require_feature_availability_contract(
    value: Mapping[str, Any],
    *,
    expected_candidate_pool_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate exact internal lineage without trusting summary fields."""

    if not isinstance(value, Mapping):
        raise RuntimeError("FEATURE_AVAILABILITY_CONTRACT_MISSING")
    exact_keys = {
        "schema_version",
        "fit_scope",
        "fit_clock",
        "consumer_clocks",
        "train_start_utc",
        "train_end_utc",
        "train_row_count",
        "history_rows_excluded_from_fit",
        "source_time_max_utc",
        "validation_rows_used",
        "test_rows_used",
        "future_rows_used",
        "selection_policy",
        "semantics",
        "candidate_pool",
        "candidate_pool_count",
        "candidate_pool_sha256",
        "train_matrix_sha256",
        "diagnostic_target_sha256",
        "diagnostic_target_affects_selection",
        "features",
        "available_features",
        "available_feature_count",
        "excluded_features",
        "excluded_feature_count",
        "family_coverage",
        "semantic_timeframe_coverage",
        "selection_sha256",
        "reachability_decision",
        "contract_sha256",
    }
    if set(value) != exact_keys:
        raise RuntimeError("FEATURE_AVAILABILITY_CONTRACT_SURFACE_INVALID")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "fit_scope": "train_only",
        "fit_clock": FIT_CLOCK,
        "consumer_clocks": list(CONSUMER_CLOCKS),
        "validation_rows_used": False,
        "test_rows_used": False,
        "future_rows_used": False,
        "selection_policy": SELECTION_POLICY,
        "semantics": SEMANTICS,
        "diagnostic_target_affects_selection": False,
        "reachability_decision": "REQUIRES_EXACT_SIGNAL_DIMENSION_BINDING",
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise RuntimeError("FEATURE_AVAILABILITY_CONTRACT_POLICY_INVALID")
    names = value.get("candidate_pool")
    rows = value.get("features")
    if (
        not isinstance(names, list)
        or len(names) != len(set(names))
        or not isinstance(rows, list)
        or len(rows) != len(names)
        or [row.get("name") for row in rows if isinstance(row, Mapping)] != names
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_CONTRACT_CANDIDATES_INVALID")
    pool_sha = _json_sha256(names)
    if (
        value.get("candidate_pool_count") != len(names)
        or value.get("candidate_pool_sha256") != pool_sha
        or (
            expected_candidate_pool_sha256 is not None
            and pool_sha != expected_candidate_pool_sha256
        )
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_CANDIDATE_POOL_BINDING_INVALID")
    feature_keys = {
        "name",
        "owner",
        "semantic_timeframes",
        "decision",
        "exclusion_reason",
        "exact_duplicate_of",
        "train_value_sha256",
        "train_rows",
        "finite_rows",
        "distinct_value_count",
        "nonzero_rows",
        "observed_min",
        "observed_max",
        "diagnostic_abs_spearman",
    }
    available: list[str] = []
    excluded: list[dict[str, Any]] = []
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != feature_keys:
            raise RuntimeError("FEATURE_AVAILABILITY_FEATURE_ROW_INVALID")
        name = str(row["name"])
        decision = row.get("decision")
        owner = row.get("owner")
        value_sha = row.get("train_value_sha256")
        if (
            decision not in _REASONS
            or owner not in MODEL_NATIVE_TRAINING_SPECIALISTS
            or not isinstance(value_sha, str)
            or not _SHA256_RE.fullmatch(value_sha)
            or row.get("train_rows") != value.get("train_row_count")
            or row.get("finite_rows") != value.get("train_row_count")
            or not isinstance(row.get("distinct_value_count"), int)
            or int(row["distinct_value_count"]) < 1
            or not isinstance(row.get("nonzero_rows"), int)
            or not 0 <= int(row["nonzero_rows"]) <= int(row["train_rows"])
            or not math.isfinite(float(row.get("observed_min")))
            or not math.isfinite(float(row.get("observed_max")))
        ):
            raise RuntimeError(f"FEATURE_AVAILABILITY_FEATURE_EVIDENCE_INVALID: {name}")
        score = row.get("diagnostic_abs_spearman")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise RuntimeError(f"FEATURE_AVAILABILITY_DIAGNOSTIC_INVALID: {name}")
        duplicate = row.get("exact_duplicate_of")
        if decision == "available":
            if row.get("exclusion_reason") is not None or duplicate is not None:
                raise RuntimeError(f"FEATURE_AVAILABILITY_DECISION_INVALID: {name}")
            available.append(name)
        elif decision == "train_constant":
            if (
                row.get("exclusion_reason") != "train_constant"
                or duplicate is not None
                or int(row["distinct_value_count"]) != 1
            ):
                raise RuntimeError(f"FEATURE_AVAILABILITY_CONSTANT_INVALID: {name}")
            excluded.append(
                {"name": name, "reason": decision, "exact_duplicate_of": None}
            )
        else:
            representative = by_name.get(str(duplicate))
            if (
                row.get("exclusion_reason") != "exact_duplicate"
                or representative is None
                or representative.get("decision") != "available"
                or representative.get("train_value_sha256") != value_sha
                or int(row["distinct_value_count"]) <= 1
            ):
                raise RuntimeError(f"FEATURE_AVAILABILITY_DUPLICATE_INVALID: {name}")
            excluded.append(
                {"name": name, "reason": decision, "exact_duplicate_of": duplicate}
            )
        by_name[name] = row
    if (
        value.get("available_features") != available
        or value.get("available_feature_count") != len(available)
        or value.get("excluded_features") != excluded
        or value.get("excluded_feature_count") != len(excluded)
        or value.get("selection_sha256")
        != _json_sha256({"available": available, "excluded": excluded})
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_SELECTION_BINDING_INVALID")
    expected_family = _coverage(
        rows,
        key="owner",
        values=MODEL_NATIVE_TRAINING_SPECIALISTS,
    )
    expected_tf = _coverage(
        rows,
        key="semantic_timeframes",
        values=(*SEMANTIC_TIMEFRAMES, GLOBAL_TIMEFRAME_TAG),
    )
    if value.get("family_coverage") != expected_family or any(
        row["candidate"] == 0 for row in expected_family.values()
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_FAMILY_COVERAGE_INVALID")
    if value.get("semantic_timeframe_coverage") != expected_tf:
        raise RuntimeError("FEATURE_AVAILABILITY_TIMEFRAME_COVERAGE_INVALID")
    for field in ("candidate_pool_sha256", "train_matrix_sha256", "selection_sha256"):
        raw = value.get(field)
        if not isinstance(raw, str) or not _SHA256_RE.fullmatch(raw):
            raise RuntimeError("FEATURE_AVAILABILITY_HASH_INVALID")
    target_hash = value.get("diagnostic_target_sha256")
    if target_hash is not None and (
        not isinstance(target_hash, str) or not _SHA256_RE.fullmatch(target_hash)
    ):
        raise RuntimeError("FEATURE_AVAILABILITY_TARGET_HASH_INVALID")
    unsigned = dict(value)
    declared_contract_sha = unsigned.pop("contract_sha256")
    if declared_contract_sha != _json_sha256(unsigned):
        raise RuntimeError("FEATURE_AVAILABILITY_CONTRACT_SHA256_INVALID")
    return dict(value)


__all__ = [
    "CONSUMER_CLOCKS",
    "FIT_CLOCK",
    "GLOBAL_TIMEFRAME_TAG",
    "SCHEMA_VERSION",
    "SELECTION_POLICY",
    "SEMANTIC_TIMEFRAMES",
    "SEMANTICS",
    "fit_feature_availability_contract",
    "require_feature_availability_contract",
]
