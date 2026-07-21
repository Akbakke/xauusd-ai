"""Fail-closed liveness contract for the complete smart Entry model input.

The artifact described here is deliberately self-contained: every signal,
continuous-context and categorical-context field has one status per split.
Consumers must validate the artifact bytes, policy, exact field order, source
manifest bindings and ATR shift observation instead of trusting a report-level
PASS.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)

SCHEMA_VERSION = "entry_full_input_liveness_contract_v3"
POLICY_VERSION = "entry_full_input_liveness_policy_v3"
PASS_DECISION = "PASS"
FAIL_DECISION = "FAIL"
SPLITS = ("train", "val", "test")
SURFACES = ("signal", "ctx_cont", "ctx_cat")
EXPECTED_FIELD_COUNTS = {"signal": MODEL_NATIVE_SIGNAL_DIM, "ctx_cont": 142, "ctx_cat": 5}

NEAR_CONSTANT_STD = 1e-9
MIN_ACTIVE_RATE = 0.01
INTEGER_TOLERANCE = 1e-6
ATR_MAX_STANDARDIZED_MEAN_SHIFT = 1.0
ATR_MIN_STD_RATIO = 0.25
ATR_MAX_STD_RATIO = 4.0

# There is no constant pass-through surface in the model-native contract.  A
# field must be learnable on TRAIN.  VAL/TEST are untouched chronological
# observations and may legitimately contain one regime state for an entire
# short split; their job here is exact coverage/finiteness, not synthetic
# variation.  Direction edge is decided later by the OOS performance gates.
CONSTANT_ALLOWLIST: dict[tuple[str, str], tuple[str, ...]] = {}

# Semantically sparse impulses bypass the generic one-percent TRAIN activity
# rule only after meeting an explicit support floor.  OOS event counts are
# reported exactly but never manufactured or used to invalidate a genuine
# chronological market window.
RARE_EVENT_MINIMUMS: dict[tuple[str, str], dict[str, int]] = {
    ("signal", "smc_choch"): {"train": 32},
    ("signal", "candle.pattern_outside_after_inside_bull_breakout_score"): {
        "train": 16,
    },
    ("signal", "candle.pattern_outside_after_inside_bear_breakout_score"): {
        "train": 16,
    },
    ("signal", "chart.m5_ema50_200_cross_up"): {"train": 128},
    ("signal", "chart.m5_ema50_200_cross_down"): {"train": 128},
    ("ctx_cont", "d1_regime_changed_flag_v3"): {"train": 32},
}

ATR_OOD_FIELDS = (
    ("signal", "ctx_cont.d1_atr14_canon_v2"),
    ("signal", "ctx_cont._v1h4_atr"),
    ("ctx_cont", "d1_atr14_canon_v2"),
    ("ctx_cont", "_v1h4_atr"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def field_order_sha256(fields: Sequence[str]) -> str:
    return _canonical_json_sha256([str(field) for field in fields])


def field_order_hashes(field_order: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {surface: field_order_sha256(field_order.get(surface, ())) for surface in SURFACES}


def canonical_policy() -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "numeric": {
            "near_constant_std": NEAR_CONSTANT_STD,
            "near_constant_range": NEAR_CONSTANT_STD,
            "min_active_rate": MIN_ACTIVE_RATE,
            "active_abs_threshold": 1e-7,
        },
        "categorical": {
            "min_unique_count": 2,
            "integer_tolerance": INTEGER_TOLERANCE,
        },
        "constant_allowlist": [
            {"surface": surface, "field": field, "splits": list(splits)}
            for (surface, field), splits in sorted(CONSTANT_ALLOWLIST.items())
        ],
        "rare_event_minimums": [
            {
                "surface": surface,
                "field": field,
                "minimum_active_count": {
                    split: int(minimums[split]) for split in sorted(minimums)
                },
            }
            for (surface, field), minimums in sorted(RARE_EVENT_MINIMUMS.items())
        ],
        "atr_ood": {
            "fields": [
                {"surface": surface, "field": field}
                for surface, field in ATR_OOD_FIELDS
            ],
            "reference_split": "train",
            "comparison_splits": ["val", "test"],
            "max_standardized_mean_shift": ATR_MAX_STANDARDIZED_MEAN_SHIFT,
            "min_std_ratio": ATR_MIN_STD_RATIO,
            "max_std_ratio": ATR_MAX_STD_RATIO,
            "decision_role": "diagnostic_only_direction_edge_requires_oos_gates",
        },
        "split_roles": {
            "train": "strict_learnability_and_support",
            "val": "untouched_oos_coverage_and_finiteness",
            "test": "untouched_oos_coverage_and_finiteness",
        },
    }


def _normalized_path(value: Any) -> str:
    text = str(value or "").strip()
    return str(Path(text).expanduser().resolve()) if text else ""


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_stats(raw: Mapping[str, Any] | None, *, surface: str) -> dict[str, Any]:
    raw = raw if isinstance(raw, Mapping) else {}
    rows = _int(raw.get("row_count", raw.get("rows")))
    finite = _int(raw.get("finite_count"))
    nonfinite = _int(raw.get("nonfinite_count"), max(rows - finite, 0))
    active_count = _int(raw.get("active_count"))
    active_rate = _float(raw.get("active_rate"), active_count / rows if rows > 0 else 0.0)
    row = {
        "row_count": rows,
        "finite_count": finite,
        "nonfinite_count": nonfinite,
        "mean": _float(raw.get("mean")),
        "std": max(_float(raw.get("std")), 0.0),
        "min": _float(raw.get("min")),
        "max": _float(raw.get("max")),
        "value_range": max(_float(raw.get("value_range")), 0.0),
        "active_count": active_count,
        "active_rate": active_rate,
    }
    if surface == "ctx_cat":
        row["unique_count"] = _int(raw.get("unique_count"))
        row["integer_like_count"] = _int(raw.get("integer_like_count"))
        values = raw.get("unique_values")
        row["unique_values"] = (
            sorted({_int(value) for value in values})
            if isinstance(values, list)
            else []
        )
    return row


def classify_field_status(
    *,
    split: str,
    surface: str,
    field: str,
    stats: Mapping[str, Any],
) -> tuple[str, str]:
    rows = _int(stats.get("row_count"))
    finite = _int(stats.get("finite_count"))
    nonfinite = _int(stats.get("nonfinite_count"))
    if rows <= 0:
        return "FAIL", "no_rows"
    if finite != rows or nonfinite != 0:
        return "FAIL", "nonfinite"
    if surface == "ctx_cat":
        if _int(stats.get("integer_like_count")) != rows:
            return "FAIL", "categorical_non_integer"
        if split == "train" and _int(stats.get("unique_count")) < 2:
            return "FAIL", "categorical_cardinality_below_two"
        if split == "train":
            return "LIVE", "categorical_train_cardinality"
        if _int(stats.get("unique_count")) < 1:
            return "FAIL", "categorical_oos_empty"
        if _int(stats.get("unique_count")) == 1:
            return "OBSERVED_SINGLE_STATE", "categorical_oos_single_state"
        return "OBSERVED_VARIABLE", "categorical_oos_cardinality"

    std = _float(stats.get("std"))
    value_range = _float(stats.get("value_range"))
    active_count = _int(stats.get("active_count"))
    active_rate = _float(stats.get("active_rate"))
    variable = std > NEAR_CONSTANT_STD and value_range > NEAR_CONSTANT_STD
    if split != "train":
        if not variable:
            return "OBSERVED_SINGLE_STATE", "numeric_oos_single_state"
        if active_rate < MIN_ACTIVE_RATE:
            return "OBSERVED_RARE_EVENT", "numeric_oos_below_training_activity_floor"
        return "OBSERVED_VARIABLE", "numeric_oos_variability_and_activity"
    if variable and active_rate >= MIN_ACTIVE_RATE:
        return "LIVE", "numeric_variability_and_activity"
    rare_minimum = RARE_EVENT_MINIMUMS.get((surface, field), {}).get(split)
    if rare_minimum is not None and variable and active_count >= rare_minimum:
        return "ALLOWED_RARE_EVENT", "exact_rare_event_support_floor"
    if not variable:
        return "FAIL", "unallowed_near_constant"
    if rare_minimum is not None:
        return "FAIL", "rare_event_support_below_minimum"
    return "FAIL", "active_rate_below_minimum"


def _drift_rows(field_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row.get("split")), str(row.get("surface")), str(row.get("field"))): row
        for row in field_rows
    }
    rows: list[dict[str, Any]] = []
    for surface, field in ATR_OOD_FIELDS:
        train = index.get(("train", surface, field), {})
        train_mean = _float(train.get("mean"))
        train_std = _float(train.get("std"))
        for split in ("val", "test"):
            observed = index.get((split, surface, field), {})
            observed_mean = _float(observed.get("mean"))
            observed_std = _float(observed.get("std"))
            standardized_mean_shift = (
                abs(observed_mean - train_mean) / train_std
                if train_std > NEAR_CONSTANT_STD
                else math.inf
            )
            std_ratio = observed_std / train_std if train_std > NEAR_CONSTANT_STD else math.inf
            green = (
                train.get("status") in {"LIVE", "ALLOWED_RARE_EVENT"}
                and observed.get("status")
                in {"OBSERVED_VARIABLE", "OBSERVED_RARE_EVENT"}
                and math.isfinite(standardized_mean_shift)
                and standardized_mean_shift <= ATR_MAX_STANDARDIZED_MEAN_SHIFT
                and math.isfinite(std_ratio)
                and ATR_MIN_STD_RATIO <= std_ratio <= ATR_MAX_STD_RATIO
            )
            rows.append(
                {
                    "surface": surface,
                    "field": field,
                    "reference_split": "train",
                    "split": split,
                    "reference_mean": train_mean,
                    "reference_std": train_std,
                    "observed_mean": observed_mean,
                    "observed_std": observed_std,
                    "standardized_mean_shift": (
                        standardized_mean_shift if math.isfinite(standardized_mean_shift) else None
                    ),
                    "std_ratio": std_ratio if math.isfinite(std_ratio) else None,
                    "status": "STABLE" if green else "SHIFT_OBSERVED",
                }
            )
    return rows


def build_full_input_liveness_artifact(
    *,
    dataset_dir: str | Path,
    contract_mode: str,
    field_order: Mapping[str, Sequence[str]],
    stats_by_split: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    manifest_bindings: Mapping[str, Mapping[str, Any]],
    scan_proof_by_split: Mapping[str, Mapping[str, Any]],
    created_utc: str,
) -> dict[str, Any]:
    normalized_order = {
        surface: [str(field) for field in field_order.get(surface, ())]
        for surface in SURFACES
    }
    field_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if str(contract_mode) != MODEL_NATIVE_CONTRACT_MODE:
        failures.append(
            {
                "code": "contract_mode_mismatch",
                "expected": MODEL_NATIVE_CONTRACT_MODE,
                "observed": str(contract_mode),
            }
        )

    for surface, expected_count in EXPECTED_FIELD_COUNTS.items():
        fields = normalized_order[surface]
        if len(fields) != expected_count:
            failures.append(
                {
                    "code": "field_count_mismatch",
                    "surface": surface,
                    "expected": expected_count,
                    "observed": len(fields),
                }
            )
        if len(set(fields)) != len(fields):
            failures.append({"code": "duplicate_fields", "surface": surface})
    forbidden_present = sorted(
        set(normalized_order["signal"]) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    )
    if forbidden_present:
        failures.append(
            {
                "code": "forbidden_legacy_bridge_fields_present",
                "fields": forbidden_present,
            }
        )

    required_policy_fields = set(RARE_EVENT_MINIMUMS) | set(ATR_OOD_FIELDS)
    for surface, field in sorted(required_policy_fields):
        if field not in normalized_order.get(surface, []):
            failures.append(
                {"code": "required_policy_field_missing", "surface": surface, "field": field}
            )

    for split in SPLITS:
        split_stats = stats_by_split.get(split, {})
        for surface in SURFACES:
            surface_stats = split_stats.get(surface, {})
            for field in normalized_order[surface]:
                stats = _normalized_stats(surface_stats.get(field), surface=surface)
                status, reason = classify_field_status(
                    split=split,
                    surface=surface,
                    field=field,
                    stats=stats,
                )
                row = {
                    "split": split,
                    "surface": surface,
                    "field": field,
                    **stats,
                    "status": status,
                    "status_reason": reason,
                }
                field_rows.append(row)
                if status == "FAIL":
                    failures.append(
                        {
                            "code": "field_liveness_fail",
                            "split": split,
                            "surface": surface,
                            "field": field,
                            "reason": reason,
                        }
                    )

    # Categorical OOS values must remain inside the exact TRAIN vocabulary.
    # A one-state OOS split is a valid regime observation; an unseen category
    # is a schema/state-contract breach and therefore remains fail-closed.
    row_index = {
        (row["split"], row["surface"], row["field"]): row for row in field_rows
    }
    for field in normalized_order["ctx_cat"]:
        train_values = set(
            row_index.get(("train", "ctx_cat", field), {}).get("unique_values", [])
        )
        for split in ("val", "test"):
            observed_values = set(
                row_index.get((split, "ctx_cat", field), {}).get("unique_values", [])
            )
            unseen = sorted(observed_values - train_values)
            if unseen:
                failures.append(
                    {
                        "code": "categorical_oos_value_outside_train_support",
                        "split": split,
                        "surface": "ctx_cat",
                        "field": field,
                        "unseen_values": unseen,
                    }
                )

    bindings: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        raw = manifest_bindings.get(split, {})
        path = _normalized_path(raw.get("path"))
        recorded_sha = str(raw.get("sha256") or "")
        exists = bool(path and Path(path).exists())
        observed_sha = sha256_file(Path(path)) if exists else ""
        bindings[split] = {
            "path": path,
            "sha256": recorded_sha,
            "exists": exists,
            "observed_sha256": observed_sha,
        }
        if not exists or not _is_sha256(recorded_sha) or observed_sha != recorded_sha:
            failures.append(
                {
                    "code": "manifest_binding_invalid",
                    "split": split,
                    "path": path,
                    "recorded_sha256": recorded_sha,
                    "observed_sha256": observed_sha,
                }
            )

    scan_proof: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        raw = scan_proof_by_split.get(split, {})
        parquet_path = _normalized_path(raw.get("parquet_path"))
        parquet = Path(parquet_path) if parquet_path else None
        exists = bool(parquet is not None and parquet.exists() and parquet.is_file())
        size_bytes = _int(raw.get("size_bytes"))
        mtime_ns = _int(raw.get("mtime_ns"))
        total_rows = _int(raw.get("total_rows"))
        scanned_rows = _int(raw.get("scanned_rows"))
        fullscan = raw.get("fullscan") is True
        scan_complete = raw.get("scan_complete") is True
        observed_size = int(parquet.stat().st_size) if exists else -1
        observed_mtime_ns = int(parquet.stat().st_mtime_ns) if exists else -1
        stat_identity = {
            "parquet_path": parquet_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
        }
        scan_proof[split] = {
            **stat_identity,
            "stat_identity_sha256": _canonical_json_sha256(stat_identity),
            "total_rows": total_rows,
            "scanned_rows": scanned_rows,
            "fullscan": fullscan,
            "scan_complete": scan_complete,
        }
        if (
            not exists
            or size_bytes != observed_size
            or mtime_ns != observed_mtime_ns
            or total_rows <= 0
            or scanned_rows != total_rows
            or not fullscan
            or not scan_complete
        ):
            failures.append(
                {
                    "code": "fullscan_proof_invalid",
                    "split": split,
                    "parquet_path": parquet_path,
                    "total_rows": total_rows,
                    "scanned_rows": scanned_rows,
                    "fullscan": fullscan,
                    "scan_complete": scan_complete,
                    "recorded_size_bytes": size_bytes,
                    "observed_size_bytes": observed_size,
                    "recorded_mtime_ns": mtime_ns,
                    "observed_mtime_ns": observed_mtime_ns,
                }
            )

    drift_rows = _drift_rows(field_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": str(created_utc),
        "decision": PASS_DECISION if not failures else FAIL_DECISION,
        "dataset_dir": _normalized_path(dataset_dir),
        "contract_mode": str(contract_mode),
        "splits": list(SPLITS),
        "expected_field_counts": dict(EXPECTED_FIELD_COUNTS),
        "field_order": normalized_order,
        "field_order_sha256": field_order_hashes(normalized_order),
        "policy": canonical_policy(),
        "input_bindings": {
            "split_manifests": bindings,
            "fullscan_proof": scan_proof,
        },
        "field_status": field_rows,
        "atr_ood_drift": {
            "status": (
                "STABLE"
                if all(row["status"] == "STABLE" for row in drift_rows)
                else "SHIFT_OBSERVED"
            ),
            "decision_role": "diagnostic_only_direction_edge_requires_oos_gates",
            "rows": drift_rows,
        },
        "failures": failures,
    }


def _append_failure(failures: list[dict[str, Any]], code: str, **details: Any) -> None:
    failures.append({"code": code, **details})


def validate_full_input_liveness_artifact(
    artifact_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_dataset_dir: str | Path | None = None,
    expected_contract_mode: str = MODEL_NATIVE_CONTRACT_MODE,
    expected_field_order: Mapping[str, Sequence[str]] | None = None,
    expected_field_order_sha256: Mapping[str, str] | None = None,
    expected_manifest_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path).expanduser().resolve()
    failures: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "sha256": "",
            "schema_version": "",
            "decision": "",
            "field_order_sha256": {},
            "failures": [{"code": "artifact_missing", "path": str(path)}],
        }
    observed_sha = sha256_file(path)
    if expected_sha256 is not None and observed_sha != str(expected_sha256):
        _append_failure(
            failures,
            "artifact_sha256_mismatch",
            expected=str(expected_sha256),
            observed=observed_sha,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "path": str(path),
            "sha256": observed_sha,
            "schema_version": "",
            "decision": "",
            "field_order_sha256": {},
            "failures": [{"code": "artifact_json_invalid", "error": f"{type(exc).__name__}: {exc}"}],
        }
    if not isinstance(payload, dict):
        payload = {}
        _append_failure(failures, "artifact_root_not_object")

    if payload.get("schema_version") != SCHEMA_VERSION:
        _append_failure(failures, "schema_version_mismatch", observed=payload.get("schema_version"))
    if payload.get("contract_mode") != expected_contract_mode:
        _append_failure(failures, "contract_mode_mismatch", observed=payload.get("contract_mode"))
    if payload.get("splits") != list(SPLITS):
        _append_failure(failures, "split_contract_mismatch", observed=payload.get("splits"))
    if payload.get("expected_field_counts") != EXPECTED_FIELD_COUNTS:
        _append_failure(
            failures,
            "expected_field_counts_mismatch",
            observed=payload.get("expected_field_counts"),
        )
    if payload.get("policy") != canonical_policy():
        _append_failure(failures, "policy_mismatch_or_noncanonical_allowlist")
    if payload.get("decision") != PASS_DECISION or payload.get("failures"):
        _append_failure(
            failures,
            "artifact_decision_not_pass",
            decision=payload.get("decision"),
            artifact_failure_count=len(payload.get("failures") or []),
        )
    if expected_dataset_dir is not None and payload.get("dataset_dir") != _normalized_path(expected_dataset_dir):
        _append_failure(
            failures,
            "dataset_identity_mismatch",
            expected=_normalized_path(expected_dataset_dir),
            observed=payload.get("dataset_dir"),
        )

    raw_order = payload.get("field_order") if isinstance(payload.get("field_order"), dict) else {}
    normalized_order = {
        surface: [str(field) for field in raw_order.get(surface, [])]
        if isinstance(raw_order.get(surface), list)
        else []
        for surface in SURFACES
    }
    observed_order_hashes = field_order_hashes(normalized_order)
    if payload.get("field_order_sha256") != observed_order_hashes:
        _append_failure(failures, "field_order_hash_mismatch")
    for surface, expected_count in EXPECTED_FIELD_COUNTS.items():
        fields = normalized_order[surface]
        if len(fields) != expected_count or len(set(fields)) != expected_count:
            _append_failure(
                failures,
                "field_set_count_or_uniqueness_mismatch",
                surface=surface,
                expected=expected_count,
                observed=len(fields),
                unique=len(set(fields)),
            )
    forbidden_present = sorted(
        set(normalized_order["signal"]) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    )
    if forbidden_present:
        _append_failure(
            failures,
            "forbidden_legacy_bridge_fields_present",
            fields=forbidden_present,
        )
    if expected_field_order is not None:
        for surface in SURFACES:
            expected = [str(field) for field in expected_field_order.get(surface, ())]
            if normalized_order[surface] != expected:
                _append_failure(failures, "exact_field_order_mismatch", surface=surface)
    if expected_field_order_sha256 is not None:
        for surface in SURFACES:
            if observed_order_hashes[surface] != str(expected_field_order_sha256.get(surface) or ""):
                _append_failure(failures, "expected_field_order_hash_mismatch", surface=surface)
    for surface, field in sorted(set(RARE_EVENT_MINIMUMS) | set(ATR_OOD_FIELDS)):
        if field not in normalized_order[surface]:
            _append_failure(failures, "required_policy_field_missing", surface=surface, field=field)

    raw_rows = payload.get("field_status") if isinstance(payload.get("field_status"), list) else []
    expected_keys = {
        (split, surface, field)
        for split in SPLITS
        for surface in SURFACES
        for field in normalized_order[surface]
    }
    row_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    duplicate_keys: list[tuple[str, str, str]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        key = (str(raw.get("split")), str(raw.get("surface")), str(raw.get("field")))
        if key in row_index:
            duplicate_keys.append(key)
        row_index[key] = raw
    observed_keys = set(row_index)
    if duplicate_keys:
        _append_failure(failures, "duplicate_field_status_rows", count=len(duplicate_keys))
    if observed_keys != expected_keys:
        _append_failure(
            failures,
            "field_status_exact_coverage_mismatch",
            missing_count=len(expected_keys - observed_keys),
            extra_count=len(observed_keys - expected_keys),
        )
    for key in sorted(expected_keys & observed_keys):
        split, surface, field = key
        row = row_index[key]
        invalid_stats: list[str] = []
        required_counts = ("row_count", "finite_count", "nonfinite_count")
        for name in required_counts:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                invalid_stats.append(name)
        row_count = _int(row.get("row_count"))
        finite_count = _int(row.get("finite_count"))
        nonfinite_count = _int(row.get("nonfinite_count"))
        if finite_count + nonfinite_count != row_count:
            invalid_stats.append("finite_plus_nonfinite")
        if surface == "ctx_cat":
            for name in ("unique_count", "integer_like_count"):
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    invalid_stats.append(name)
            unique_values = row.get("unique_values")
            if (
                not isinstance(unique_values, list)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in unique_values)
                or unique_values != sorted(set(unique_values))
                or len(unique_values) != _int(row.get("unique_count"))
            ):
                invalid_stats.append("unique_values")
            if _int(row.get("unique_count")) > finite_count:
                invalid_stats.append("unique_count_exceeds_finite")
            if _int(row.get("integer_like_count")) > finite_count:
                invalid_stats.append("integer_like_count_exceeds_finite")
        else:
            for name in ("mean", "std", "min", "max", "value_range", "active_rate"):
                value = row.get(name)
                try:
                    finite_number = not isinstance(value, bool) and math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite_number = False
                if not finite_number:
                    invalid_stats.append(name)
            active_count = row.get("active_count")
            if isinstance(active_count, bool) or not isinstance(active_count, int) or not 0 <= active_count <= finite_count:
                invalid_stats.append("active_count")
            observed_active_rate = _float(row.get("active_rate"))
            expected_active_rate = _int(active_count) / row_count if row_count > 0 else 0.0
            if not math.isclose(observed_active_rate, expected_active_rate, rel_tol=1e-9, abs_tol=1e-12):
                invalid_stats.append("active_rate_consistency")
            minimum = _float(row.get("min"))
            maximum = _float(row.get("max"))
            mean = _float(row.get("mean"))
            value_range = _float(row.get("value_range"))
            if minimum > maximum:
                invalid_stats.append("min_max_order")
            if mean < minimum - 1e-9 or mean > maximum + 1e-9:
                invalid_stats.append("mean_outside_min_max")
            if not math.isclose(value_range, max(maximum - minimum, 0.0), rel_tol=1e-9, abs_tol=1e-12):
                invalid_stats.append("value_range_consistency")
        if invalid_stats:
            _append_failure(
                failures,
                "field_stats_invalid",
                split=split,
                surface=surface,
                field=field,
                invalid=sorted(set(invalid_stats)),
            )
        normalized_stats = _normalized_stats(row, surface=surface)
        expected_status, expected_reason = classify_field_status(
            split=split,
            surface=surface,
            field=field,
            stats=normalized_stats,
        )
        if row.get("status") != expected_status or row.get("status_reason") != expected_reason:
            _append_failure(
                failures,
                "field_status_recalculation_mismatch",
                split=split,
                surface=surface,
                field=field,
                expected_status=expected_status,
                observed_status=row.get("status"),
            )
        if expected_status == "FAIL":
            _append_failure(
                failures,
                "field_liveness_fail",
                split=split,
                surface=surface,
                field=field,
                reason=expected_reason,
            )

    for field in normalized_order["ctx_cat"]:
        train_values = set(
            row_index.get(("train", "ctx_cat", field), {}).get("unique_values", [])
        )
        for split in ("val", "test"):
            observed_values = set(
                row_index.get((split, "ctx_cat", field), {}).get("unique_values", [])
            )
            unseen = sorted(observed_values - train_values)
            if unseen:
                _append_failure(
                    failures,
                    "categorical_oos_value_outside_train_support",
                    split=split,
                    surface="ctx_cat",
                    field=field,
                    unseen_values=unseen,
                )

    bindings_root = payload.get("input_bindings") if isinstance(payload.get("input_bindings"), dict) else {}
    bindings = (
        bindings_root.get("split_manifests")
        if isinstance(bindings_root.get("split_manifests"), dict)
        else {}
    )
    if set(bindings) != set(SPLITS):
        _append_failure(failures, "split_manifest_binding_set_mismatch", observed=sorted(bindings))
    for split in SPLITS:
        binding = bindings.get(split) if isinstance(bindings.get(split), Mapping) else {}
        binding_path = _normalized_path(binding.get("path"))
        recorded_sha = str(binding.get("sha256") or "")
        exists = bool(binding_path and Path(binding_path).exists())
        actual_sha = sha256_file(Path(binding_path)) if exists else ""
        if not exists or not _is_sha256(recorded_sha) or actual_sha != recorded_sha:
            _append_failure(
                failures,
                "split_manifest_binding_invalid",
                split=split,
                path=binding_path,
                recorded_sha256=recorded_sha,
                actual_sha256=actual_sha,
            )
        if expected_manifest_bindings is not None:
            expected_binding = expected_manifest_bindings.get(split, {})
            if (
                binding_path != _normalized_path(expected_binding.get("path"))
                or recorded_sha != str(expected_binding.get("sha256") or "")
            ):
                _append_failure(failures, "expected_split_manifest_binding_mismatch", split=split)

    scan_proof = (
        bindings_root.get("fullscan_proof")
        if isinstance(bindings_root.get("fullscan_proof"), dict)
        else {}
    )
    if set(scan_proof) != set(SPLITS):
        _append_failure(failures, "fullscan_proof_split_set_mismatch", observed=sorted(scan_proof))
    for split in SPLITS:
        proof = scan_proof.get(split) if isinstance(scan_proof.get(split), Mapping) else {}
        parquet_path = _normalized_path(proof.get("parquet_path"))
        parquet = Path(parquet_path) if parquet_path else None
        exists = bool(parquet is not None and parquet.exists() and parquet.is_file())
        size_bytes = _int(proof.get("size_bytes"))
        mtime_ns = _int(proof.get("mtime_ns"))
        stat_identity = {
            "parquet_path": parquet_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
        }
        observed_size = int(parquet.stat().st_size) if exists else -1
        observed_mtime_ns = int(parquet.stat().st_mtime_ns) if exists else -1
        total_rows = _int(proof.get("total_rows"))
        scanned_rows = _int(proof.get("scanned_rows"))
        if (
            proof.get("stat_identity_sha256") != _canonical_json_sha256(stat_identity)
            or not exists
            or size_bytes != observed_size
            or mtime_ns != observed_mtime_ns
            or total_rows <= 0
            or scanned_rows != total_rows
            or proof.get("fullscan") is not True
            or proof.get("scan_complete") is not True
        ):
            _append_failure(
                failures,
                "fullscan_proof_invalid",
                split=split,
                parquet_path=parquet_path,
                total_rows=total_rows,
                scanned_rows=scanned_rows,
                fullscan=proof.get("fullscan"),
                scan_complete=proof.get("scan_complete"),
                recorded_size_bytes=size_bytes,
                observed_size_bytes=observed_size,
                recorded_mtime_ns=mtime_ns,
                observed_mtime_ns=observed_mtime_ns,
            )
        split_row_count_mismatches = sum(
            1
            for (row_split, _, _), row in row_index.items()
            if row_split == split and _int(row.get("row_count")) != total_rows
        )
        if split_row_count_mismatches:
            _append_failure(
                failures,
                "field_stats_row_count_vs_fullscan_mismatch",
                split=split,
                expected_row_count=total_rows,
                mismatch_count=split_row_count_mismatches,
            )

    recomputed_drift = _drift_rows(list(row_index.values()))
    drift_root = payload.get("atr_ood_drift") if isinstance(payload.get("atr_ood_drift"), dict) else {}
    raw_drift = drift_root.get("rows") if isinstance(drift_root.get("rows"), list) else []
    drift_index = {
        (str(row.get("surface")), str(row.get("field")), str(row.get("split"))): row
        for row in raw_drift
        if isinstance(row, Mapping)
    }
    expected_drift_keys = {
        (row["surface"], row["field"], row["split"])
        for row in recomputed_drift
    }
    if set(drift_index) != expected_drift_keys:
        _append_failure(failures, "atr_ood_row_set_mismatch")
    for expected_row in recomputed_drift:
        key = (expected_row["surface"], expected_row["field"], expected_row["split"])
        observed = drift_index.get(key, {})
        if observed.get("status") != expected_row["status"]:
            _append_failure(failures, "atr_ood_status_mismatch", key=list(key))
        for metric in ("standardized_mean_shift", "std_ratio"):
            expected_value = expected_row[metric]
            observed_value = observed.get(metric)
            if expected_value is None:
                equal = observed_value is None
            else:
                try:
                    equal = math.isclose(float(observed_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-12)
                except (TypeError, ValueError):
                    equal = False
            if not equal:
                _append_failure(failures, "atr_ood_metric_mismatch", key=list(key), metric=metric)
    expected_drift_status = (
        "STABLE"
        if all(row["status"] == "STABLE" for row in recomputed_drift)
        else "SHIFT_OBSERVED"
    )
    if drift_root.get("status") != expected_drift_status:
        _append_failure(
            failures,
            "atr_ood_summary_status_mismatch",
            expected=expected_drift_status,
            observed=drift_root.get("status"),
        )
    if drift_root.get("decision_role") != "diagnostic_only_direction_edge_requires_oos_gates":
        _append_failure(failures, "atr_ood_decision_role_mismatch")

    return {
        "ok": not failures,
        "path": str(path),
        "sha256": observed_sha,
        "schema_version": payload.get("schema_version"),
        "decision": payload.get("decision"),
        "dataset_dir": payload.get("dataset_dir"),
        "contract_mode": payload.get("contract_mode"),
        "field_order_sha256": observed_order_hashes,
        "field_counts": {surface: len(normalized_order[surface]) for surface in SURFACES},
        "field_status_row_count": len(raw_rows),
        "atr_ood_status": drift_root.get("status"),
        "failures": failures,
    }
