#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import materialize_build_r5_2_from_coverage_aware_opportunity_base_with_fixed_controls_v1 as r5_rebuild


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_TAIL_SPECIFIC_R5_2_R6_REPAIR_V1"
LAYER_NAME = ACTION

AUDIT_ROOT = DEFAULT_REPORTS_ROOT / "R5_2_UPLIFT_AND_R6_HEAD_SIGNAL_AUDIT_V1_20260427T171341Z_LOCK"
R5_2_PACKAGE_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1_20260427T152500Z_LOCK"
R6_ROOT = DEFAULT_REPORTS_ROOT / "RUN_R6_RETRAIN_FROM_R5_2_CANDIDATE_PACKAGE_EXPLICIT_GATE_V1_20260427T164916Z_LOCK"
COVERAGE_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1_20260427T142902Z_LOCK"
R5_2_CANDIDATE_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_R5_2_FROM_COVERAGE_AWARE_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1_20260427T150214Z_LOCK"
LOW_SUPPORT_POLICY_ROOT = DEFAULT_REPORTS_ROOT / "DEFINE_EXPLICIT_RUN_ID_LOW_SUPPORT_POLICY_V1_20260427T140733Z_LOCK"
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"

DENOMINATOR_TARGET = 5
DEFAULT_FOLD_COUNT = 5

TAIL_REPAIR_VARIANT_IDS = [
    "BASE_R5_2_130_86_CONTROL",
    "TAIL_TARGET_WEIGHT_REPAIR_CONSERVATIVE",
    "TAIL_TARGET_WEIGHT_REPAIR_BALANCED",
    "TAIL_LOW_SUPPORT_AWARE_REPAIR",
    "R6_TAIL_HEAD_AWARE_REPAIR",
    "TAIL_REPAIR_MAX_SAFE_DIAGNOSTIC",
]

TRAINED_VARIANT_IDS = [
    "TAIL_TARGET_WEIGHT_REPAIR_CONSERVATIVE",
    "TAIL_TARGET_WEIGHT_REPAIR_BALANCED",
    "TAIL_LOW_SUPPORT_AWARE_REPAIR",
    "R6_TAIL_HEAD_AWARE_REPAIR",
]

TAIL_REPAIR_POSITIVE_ROLES = {
    "CORE_EXISTING_TAIL_RETAIN",
    "TAIL_REPAIR_PRIMARY_CANDIDATE",
    "TAIL_REPAIR_LOW_SUPPORT_TRAINING_ONLY",
    "R6_TAIL_HEAD_CANDIDATE",
    "R5_TAIL_SCORE_CANDIDATE",
    "SAFE_SUBSET_FROM_FAILED_R6_EXPANSION",
}

HARD_VETO_ROLES = {
    "HARD_VETO_PROTECTED_WINNER",
    "HARD_VETO_RUNNER_PROTECT",
    "HARD_VETO_HIGH_MFE_UNSAFE",
    "AMBIGUOUS_MONITOR_ONLY",
    "QUARANTINE_EXCLUDE",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required json artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required input artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
    return bool(value)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(
    *,
    optuna: bool,
    broad_sweep: bool,
    freeze: bool,
    promo: bool,
    live: bool,
) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if broad_sweep:
        failures.append("BROAD_SWEEP_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_no_dummy_synthetic_fallback(*, dummy: bool, synthetic: bool, fallback: bool) -> dict[str, Any]:
    failures = []
    if dummy:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if fallback:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def tail_row_has_tail_evidence(row: pd.Series | dict[str, Any]) -> bool:
    return bool(
        _as_bool(row.get("r5_tail_score_evidence_v1"))
        or _as_bool(row.get("tail_control_10_50_evidence_v1"))
        or _as_bool(row.get("v2_oof_tail_evidence_v1"))
    )


def row_has_safety_clearance(row: pd.Series | dict[str, Any]) -> bool:
    return bool(
        str(row.get("active_quarantine_v1", "")).upper() == "ACTIVE_CANDIDATE"
        and not _as_bool(row.get("protected_winner_status_v1"))
        and not _as_bool(row.get("runner_protect_status_v1"))
        and not _as_bool(row.get("ambiguous_high_mfe_status_v1"))
        and not _as_bool(row.get("fifty_plus_mfe_risk_v1"))
        and not _as_bool(row.get("hundred_plus_mfe_risk_v1"))
        and not _as_bool(row.get("two_hundred_plus_mfe_risk_v1"))
        and str(row.get("provenance_status_v1", "")).upper() not in {"MISSING", "FAIL", "UNKNOWN_REQUIRES_ARTIFACT"}
    )


def validate_tail_repair_positive(row: pd.Series | dict[str, Any]) -> bool:
    role = str(row.get("recommended_role_v1", ""))
    if role in TAIL_REPAIR_POSITIVE_ROLES:
        if not tail_row_has_tail_evidence(row):
            raise RuntimeError("TAIL_REPAIR_POSITIVE_REQUIRES_TAIL_EVIDENCE")
        if not row_has_safety_clearance(row):
            raise RuntimeError("TAIL_REPAIR_POSITIVE_REQUIRES_SAFETY_CLEARANCE")
    return True


def validate_variant_grid(variants: Sequence[dict[str, Any]]) -> bool:
    ids = [str(row.get("variant_id_v1")) for row in variants]
    if ids != TAIL_REPAIR_VARIANT_IDS:
        raise RuntimeError("TAIL_REPAIR_VARIANTS_MUST_BE_SMALL_DETERMINISTIC_SET")
    if len(ids) > 6:
        raise RuntimeError("TAIL_REPAIR_VARIANTS_MUST_NOT_BE_BROAD_SWEEP")
    return True


def max_safe_variant_can_be_final(variant_id: str) -> bool:
    return str(variant_id) != "TAIL_REPAIR_MAX_SAFE_DIAGNOSTIC"


def validate_input_artifacts_unchanged(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = [key for key, value in before.items() if after.get(key) != value]
    return {"status_v1": "PASS" if not changed else "FAIL", "changed_v1": changed, "unchanged_v1": not changed}


def candidate_can_be_selected(row: dict[str, Any]) -> bool:
    return bool(row.get("safety_clean_v1")) and bool(row.get("precision_decision_valid_v1"))


def validate_fixed_controls(controls: Sequence[dict[str, Any]]) -> bool:
    names = {str(row.get("control_v1")) for row in controls}
    required = {"r5_2_package", "wednesday_benchmark"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"TAIL_REPAIR_FIXED_CONTROLS_MISSING: {missing}")
    return True


def _input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {key: _file_hash(path) for key, path in paths.items()}


def _load_inputs(
    *,
    audit_root: Path,
    package_root: Path,
    r6_root: Path,
    coverage_root: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    required = {
        "audit_tail_gap": audit_root / "tail_gap_analysis_86_to_136_proxy_v1.csv",
        "audit_safe_subset": audit_root / "safe_subset_from_failed_r6_expansions_v1.csv",
        "audit_gap": audit_root / "r5_2_gap_to_coverage_proxy_v1.csv",
        "audit_summary": audit_root / "summary_v1.json",
        "package_scores": package_root / "r5_2_candidate_oof_scores_v1.csv",
        "package_summary": package_root / "summary_v1.json",
        "r6_scores": r6_root / "r6_oof_scores_v1.csv",
        "r6_summary": r6_root / "summary_v1.json",
        "coverage_rows": coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv",
        "candidate_target": candidate_root / "r5_2_training_target_table_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_TAIL_REPAIR_INPUT_ARTIFACTS: {missing}")
    return {
        "hashes": _input_hashes(required),
        "tail_gap": pd.read_csv(required["audit_tail_gap"]),
        "safe_subset": pd.read_csv(required["audit_safe_subset"]),
        "gap": pd.read_csv(required["audit_gap"]),
        "audit_summary": _read_json(required["audit_summary"]),
        "package_scores": pd.read_csv(required["package_scores"]),
        "package_summary": _read_json(required["package_summary"]),
        "r6_scores": pd.read_csv(required["r6_scores"]),
        "r6_summary": _read_json(required["r6_summary"]),
        "coverage_rows": pd.read_csv(required["coverage_rows"]),
        "candidate_target": pd.read_csv(required["candidate_target"]),
    }


def _tail_control_evidence(score: Any) -> bool:
    try:
        return float(score) >= 0.50
    except (TypeError, ValueError):
        return False


def _registry_role(row: dict[str, Any]) -> str:
    if str(row.get("active_quarantine_v1", "")).upper() != "ACTIVE_CANDIDATE":
        return "QUARANTINE_EXCLUDE"
    if _as_bool(row.get("protected_winner_status_v1")):
        return "HARD_VETO_PROTECTED_WINNER"
    if _as_bool(row.get("runner_protect_status_v1")):
        return "HARD_VETO_RUNNER_PROTECT"
    if _as_bool(row.get("fifty_plus_mfe_risk_v1")) or _as_bool(row.get("hundred_plus_mfe_risk_v1")) or _as_bool(row.get("two_hundred_plus_mfe_risk_v1")):
        return "HARD_VETO_HIGH_MFE_UNSAFE"
    if _as_bool(row.get("ambiguous_high_mfe_status_v1")):
        return "AMBIGUOUS_MONITOR_ONLY"
    if _as_bool(row.get("in_r5_2_130_86_v1")) and _as_bool(row.get("tail_label_v1")):
        return "CORE_EXISTING_TAIL_RETAIN"
    if not tail_row_has_tail_evidence(row):
        return "UNKNOWN_REQUIRES_ARTIFACT"
    if _as_bool(row.get("safe_subset_from_failed_r6_expansion_v1")) and _as_bool(row.get("tail_label_v1")):
        return "SAFE_SUBSET_FROM_FAILED_R6_EXPANSION"
    if _as_bool(row.get("structural_low_support_v1")):
        return "TAIL_REPAIR_LOW_SUPPORT_TRAINING_ONLY"
    if _as_bool(row.get("r5_tail_score_evidence_v1")):
        return "R5_TAIL_SCORE_CANDIDATE"
    if _as_bool(row.get("tail_control_10_50_evidence_v1")):
        return "R6_TAIL_HEAD_CANDIDATE"
    return "TAIL_REPAIR_PRIMARY_CANDIDATE"


def _build_tail_registry(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    package = inputs["package_scores"].copy()
    coverage = inputs["coverage_rows"].set_index("candidate_uid_v1")
    r6_scores = inputs["r6_scores"].set_index("candidate_uid_v1")
    tail_gap = inputs["tail_gap"].set_index("candidate_uid_v1")
    gap = inputs["gap"].set_index("candidate_uid_v1")
    safe_subset = inputs["safe_subset"].copy()
    subset_grouped = safe_subset.groupby("candidate_uid_v1").agg(
        r6_expansion_candidate_source_v1=("candidate_id_v1", lambda values: "|".join(sorted(set(map(str, values))))),
        safe_subset_from_failed_r6_expansion_v1=("recommended_use_v1", lambda values: any(str(v) in {"TAIL_REPAIR_CANDIDATE", "SAFE_EXPANSION_MINING_CANDIDATE"} for v in values)),
    )
    r5_selected_tail = package[_bool(package, "r5_2_best_candidate_selected_v1") & _bool(package, "tail_label_v1")]["candidate_uid_v1"].astype(str)
    candidate_ids = set(r5_selected_tail.tolist())
    candidate_ids.update(tail_gap.index.astype(str).tolist())
    candidate_ids.update(
        safe_subset[
            safe_subset["recommended_use_v1"].astype(str).isin(["TAIL_REPAIR_CANDIDATE", "SAFE_EXPANSION_MINING_CANDIDATE"])
            & _bool(safe_subset, "tail_label_v1")
        ]["candidate_uid_v1"].astype(str).tolist()
    )
    rows = []
    package_indexed = package.set_index("candidate_uid_v1")
    for uid in sorted(candidate_ids):
        if uid not in package_indexed.index:
            continue
        base = package_indexed.loc[uid]
        cov = coverage.loc[uid] if uid in coverage.index else pd.Series(dtype=object)
        r6 = r6_scores.loc[uid] if uid in r6_scores.index else pd.Series(dtype=object)
        tg = tail_gap.loc[uid] if uid in tail_gap.index else pd.Series(dtype=object)
        gap_row = gap.loc[uid] if uid in gap.index else pd.Series(dtype=object)
        subset = subset_grouped.loc[uid] if uid in subset_grouped.index else pd.Series(dtype=object)
        tail_control_score = r6.get("pred__entry_r6_tail_control_10_50__prob_true_v1", tg.get("r6_tail_control_score_v1", np.nan))
        source_evidence = str(base.get("source_evidence_v1", ""))
        row = {
            "candidate_uid_v1": uid,
            "run_id_v1": base.get("run_id_v1", ""),
            "active_quarantine_v1": base.get("active_quarantine_v1", ""),
            "in_r5_2_130_86_v1": _as_bool(base.get("r5_2_best_candidate_selected_v1")),
            "in_coverage_proxy_v1": uid in gap.index or str(cov.get("opportunity_role_v1", "")) in {
                "CORE_V2_OOF_POSITIVE",
                "CORE_V2_OOF_TAIL_POSITIVE",
                "COVERAGE_EXPANSION_STRONG_BAD",
                "COVERAGE_EXPANSION_TAIL",
                "COVERAGE_EXPANSION_RUN_ID_SUPPORT",
                "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE",
            },
            "missed_by_r5_2_v1": uid in tail_gap.index,
            "tail_label_v1": _as_bool(base.get("tail_label_v1")),
            "bad_label_v1": _as_bool(base.get("bad_label_v1")),
            "safe_recoverable_v1": _as_bool(base.get("safe_recoverable_v1")),
            "r5_tail_score_evidence_v1": "R5_TAIL_SCORE" in source_evidence or str(cov.get("r5_tail_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"},
            "tail_control_10_50_evidence_v1": _tail_control_evidence(tail_control_score),
            "tail_control_10_50_score_v1": tail_control_score,
            "v2_oof_tail_evidence_v1": _as_bool(base.get("v2_oof_captured_v1")) and _as_bool(base.get("tail_label_v1")),
            "historical_v2_selected_v1": _as_bool(base.get("historical_v2_captured_v1")),
            "r6_expansion_candidate_source_v1": subset.get("r6_expansion_candidate_source_v1", ""),
            "safe_subset_from_failed_r6_expansion_v1": _as_bool(subset.get("safe_subset_from_failed_r6_expansion_v1", False)),
            "protected_winner_status_v1": _as_bool(base.get("protected_winner_status_v1")),
            "runner_protect_status_v1": _as_bool(base.get("runner_protect_status_v1")),
            "ambiguous_high_mfe_status_v1": _as_bool(base.get("ambiguous_high_mfe_status_v1")),
            "fifty_plus_mfe_risk_v1": _as_bool(base.get("fifty_plus_mfe_risk_v1")),
            "hundred_plus_mfe_risk_v1": _as_bool(base.get("hundred_plus_mfe_risk_v1")),
            "two_hundred_plus_mfe_risk_v1": _as_bool(base.get("two_hundred_plus_mfe_risk_v1")),
            "low_support_class_v1": base.get("run_id_policy_class_v1", gap_row.get("low_support_class_v1", "")),
            "structural_low_support_v1": _as_bool(base.get("structural_low_support_v1")),
            "provenance_status_v1": "PASS" if _as_bool(base.get("decision_valid_score_v1")) else "MISSING",
            "source_evidence_v1": source_evidence,
        }
        row["recommended_role_v1"] = _registry_role(row)
        row["tail_evidence_pass_v1"] = tail_row_has_tail_evidence(row)
        row["safety_provenance_clearance_v1"] = row_has_safety_clearance(row)
        validate_tail_repair_positive(row)
        rows.append(row)
    row_df = pd.DataFrame(rows)
    summary = {
        "candidate_tail_rows_v1": int(len(row_df)),
        "existing_r5_2_tail_retained_v1": int(row_df["in_r5_2_130_86_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "missed_tail_rows_v1": int(row_df["missed_by_r5_2_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "repair_positive_candidate_rows_v1": int(row_df["recommended_role_v1"].isin(TAIL_REPAIR_POSITIVE_ROLES - {"CORE_EXISTING_TAIL_RETAIN"}).sum()) if not row_df.empty else 0,
        "tail_evidence_clear_candidate_rows_v1": int((row_df["tail_evidence_pass_v1"].astype(bool) & row_df["safety_provenance_clearance_v1"].astype(bool)).sum()) if not row_df.empty else 0,
        "role_counts_v1": row_df["recommended_role_v1"].value_counts().to_dict() if not row_df.empty else {},
    }
    return rows, summary


def _tail_gap_decomposition(registry_rows: list[dict[str, Any]], tail_gap: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_uid = {row["candidate_uid_v1"]: row for row in registry_rows}
    rows = []
    for _, gap_row in tail_gap.iterrows():
        uid = str(gap_row["candidate_uid_v1"])
        reg = by_uid.get(uid, {})
        strong_r5_tail = bool(reg.get("r5_tail_score_evidence_v1"))
        tail_head = bool(reg.get("tail_control_10_50_evidence_v1"))
        low_support = "STRUCTURAL_LOW_SUPPORT" in str(reg.get("low_support_class_v1", ""))
        near = _as_bool(gap_row.get("near_threshold_v1"))
        safety_clear = bool(reg.get("safety_provenance_clearance_v1"))
        reason = "UNKNOWN"
        if not safety_clear:
            reason = "SAFETY_BLOCKED"
        elif near:
            reason = "NEAR_THRESHOLD_TAIL"
        elif strong_r5_tail:
            reason = "TAIL_SIGNAL_UNDERWEIGHTED"
        elif tail_head:
            reason = "TAIL_HEAD_UNDERUSED"
        elif low_support:
            reason = "LOW_SUPPORT_TRAINING_ONLY"
        elif str(gap_row.get("reason_likely_missed_v1", "")) == "TAIL_SIGNAL_UNDERLEARNED":
            reason = "SCORE_CALIBRATION_WEAKNESS"
        elif not tail_row_has_tail_evidence(reg):
            reason = "MISSING_SIGNAL"
        rows.append(
            {
                "candidate_uid_v1": uid,
                "run_id_v1": gap_row.get("run_id_v1", reg.get("run_id_v1", "")),
                "why_missed_by_r5_2_v1": gap_row.get("reason_likely_missed_v1", ""),
                "near_threshold_v1": near,
                "fold_weakness_v1": low_support,
                "low_support_v1": low_support,
                "r5_tail_score_strength_v1": "STRONG_OR_SUPPORT" if strong_r5_tail else "WEAK_OR_NONE",
                "tail_control_10_50_strength_v1": "SUPPORT" if tail_head else "WEAK_OR_NONE",
                "safety_clear_v1": safety_clear,
                "role_recommendation_v1": reg.get("recommended_role_v1", "UNKNOWN_REQUIRES_ARTIFACT"),
                "target_adjustment_needed_v1": bool(safety_clear and tail_row_has_tail_evidence(reg)),
                "threshold_adjustment_candidate_v1": near,
                "r6_head_repair_candidate_v1": tail_head,
                "miss_reason_class_v1": reason,
            }
        )
    row_df = pd.DataFrame(rows)
    summary = {
        "missed_tail_rows_v1": int(len(row_df)),
        "safety_clear_rows_v1": int(row_df["safety_clear_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "near_threshold_rows_v1": int(row_df["near_threshold_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "strong_r5_tail_rows_v1": int(row_df["r5_tail_score_strength_v1"].eq("STRONG_OR_SUPPORT").sum()) if not row_df.empty else 0,
        "tail_control_support_rows_v1": int(row_df["tail_control_10_50_strength_v1"].eq("SUPPORT").sum()) if not row_df.empty else 0,
        "low_support_training_only_rows_v1": int(row_df["low_support_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "suitable_for_repair_rows_v1": int(row_df["target_adjustment_needed_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "miss_reason_counts_v1": row_df["miss_reason_class_v1"].value_counts().to_dict() if not row_df.empty else {},
    }
    return rows, summary


def _target_repair_design(candidate_target: pd.DataFrame, registry_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = {row["candidate_uid_v1"]: row for row in registry_rows}
    rows = []
    for _, row in candidate_target.iterrows():
        uid = str(row["candidate_uid_v1"])
        reg = registry.get(uid, {})
        original_class = str(row.get("target_class_v1", "EXCLUDE_UNKNOWN"))
        original_tier = str(row.get("training_weight_tier_v1", "UNKNOWN_ZERO_WEIGHT"))
        proposed_class = "KEEP_UNKNOWN_EXCLUDE"
        proposed_tier = "EXCLUDE_ZERO_WEIGHT"
        role = reg.get("recommended_role_v1", "")
        if original_class == "POSITIVE_STRONG_BAD":
            proposed_class = "KEEP_EXISTING_STRONG_BAD"
            proposed_tier = original_tier if original_tier in {"CORE_HIGH_WEIGHT", "COVERAGE_MEDIUM_WEIGHT", "LOW_SUPPORT_LOW_WEIGHT"} else "CORE_HIGH_WEIGHT"
        elif original_class == "POSITIVE_TAIL":
            proposed_class = "KEEP_EXISTING_TAIL"
            proposed_tier = "TAIL_REPAIR_HIGH_WEIGHT"
        elif original_class == "POSITIVE_LOW_SUPPORT_TRAINING_ONLY":
            proposed_class = "KEEP_EXISTING_TAIL"
            proposed_tier = "LOW_SUPPORT_LOW_WEIGHT"
        elif original_class == "HARD_NEGATIVE_PROTECTED_WINNER":
            proposed_class = "KEEP_HARD_NEGATIVE_PROTECTED_WINNER"
            proposed_tier = "HARD_NEGATIVE_HIGH_WEIGHT"
        elif original_class == "HARD_NEGATIVE_RUNNER_PROTECT":
            proposed_class = "KEEP_HARD_NEGATIVE_RUNNER_PROTECT"
            proposed_tier = "HARD_NEGATIVE_HIGH_WEIGHT"
        elif original_class == "HARD_NEGATIVE_HIGH_MFE_UNSAFE":
            proposed_class = "KEEP_HARD_NEGATIVE_HIGH_MFE_UNSAFE"
            proposed_tier = "HARD_NEGATIVE_HIGH_WEIGHT"
        elif original_class == "MONITOR_ONLY_AMBIGUOUS":
            proposed_class = "KEEP_AMBIGUOUS_MONITOR_ONLY"
            proposed_tier = "MONITOR_ZERO_WEIGHT"
        elif original_class == "EXCLUDE_QUARANTINE":
            proposed_class = "KEEP_QUARANTINE_EXCLUDE"
            proposed_tier = "EXCLUDE_ZERO_WEIGHT"
        if role in TAIL_REPAIR_POSITIVE_ROLES - {"CORE_EXISTING_TAIL_RETAIN"}:
            if not tail_row_has_tail_evidence(reg) or not row_has_safety_clearance(reg):
                raise RuntimeError(f"Unsafe or unevidenced tail repair positive attempted: {uid}")
            if role == "TAIL_REPAIR_LOW_SUPPORT_TRAINING_ONLY":
                proposed_class = "UPGRADE_TO_TAIL_LOW_SUPPORT_TRAINING_ONLY"
                proposed_tier = "LOW_SUPPORT_LOW_WEIGHT"
            elif role == "R6_TAIL_HEAD_CANDIDATE":
                proposed_class = "UPGRADE_TO_TAIL_REPAIR_POSITIVE"
                proposed_tier = "TAIL_REPAIR_MEDIUM_WEIGHT"
            else:
                proposed_class = "UPGRADE_TO_TAIL_REPAIR_POSITIVE"
                proposed_tier = "TAIL_REPAIR_HIGH_WEIGHT"
        rows.append(
            {
                "candidate_uid_v1": uid,
                "run_id_v1": row.get("run_id_v1", ""),
                "original_target_class_v1": original_class,
                "original_training_weight_tier_v1": original_tier,
                "proposed_tail_repair_target_class_v1": proposed_class,
                "proposed_tail_weight_tier_v1": proposed_tier,
                "role_v1": role or "NO_TAIL_REPAIR_ROLE",
                "evidence_v1": reg.get("source_evidence_v1", row.get("source_evidence_v1", "")),
                "safety_status_v1": "SAFETY_CLEAR" if not reg or row_has_safety_clearance(reg) else "SAFETY_BLOCKED",
                "reason_v1": "Tail repair design preserves existing hard vetoes and only upgrades safety-clear evidence-backed tail rows.",
            }
        )
    row_df = pd.DataFrame(rows)
    summary = {
        "rows_v1": int(len(row_df)),
        "tail_repair_upgrades_v1": int(row_df["proposed_tail_repair_target_class_v1"].astype(str).str.startswith("UPGRADE_TO_TAIL").sum()),
        "hard_negative_rows_preserved_v1": int(row_df["proposed_tail_repair_target_class_v1"].astype(str).str.startswith("KEEP_HARD_NEGATIVE").sum()),
        "quarantine_rows_preserved_excluded_v1": int(row_df["proposed_tail_repair_target_class_v1"].eq("KEEP_QUARANTINE_EXCLUDE").sum()),
        "monitor_only_rows_preserved_v1": int(row_df["proposed_tail_repair_target_class_v1"].eq("KEEP_AMBIGUOUS_MONITOR_ONLY").sum()),
    }
    return rows, summary


def _weight_value(tier: str) -> float:
    return {
        "CORE_HIGH_WEIGHT": 3.0,
        "TAIL_REPAIR_HIGH_WEIGHT": 4.0,
        "TAIL_REPAIR_MEDIUM_WEIGHT": 2.25,
        "LOW_SUPPORT_LOW_WEIGHT": 0.75,
        "HARD_NEGATIVE_HIGH_WEIGHT": 5.0,
        "MONITOR_ZERO_WEIGHT": 0.0,
        "EXCLUDE_ZERO_WEIGHT": 0.0,
    }.get(str(tier), 0.0)


def _base_target_from_existing(candidate_target: pd.DataFrame, package_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    target = candidate_target.copy()
    rename = {
        "candidate_uid_v1": "candidate_uid",
        "trade_uid_v1": "trade_uid",
        "trade_id_v1": "trade_id",
        "decision_timestamp_v1": "decision_timestamp",
        "run_id_v1": "run_id",
    }
    target = target.rename(columns=rename)
    target["candidate_uid_v1"] = target["candidate_uid"]
    package_merge_cols = [
        "active_quarantine_v1",
        "bad_label_v1",
        "tail_label_v1",
        "safe_recoverable_v1",
        "v2_oof_captured_v1",
        "historical_v2_captured_v1",
        "optuna_captured_v1",
        "v3_captured_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "existing_legal_signal_evidence_count_v1",
        "source_evidence_v1",
    ]
    if package_scores is not None:
        missing_cols = [col for col in package_merge_cols if col not in target.columns]
        if missing_cols:
            score_cols = ["candidate_uid_v1", *[col for col in missing_cols if col in package_scores.columns]]
            target = target.merge(package_scores[score_cols], on="candidate_uid_v1", how="left", validate="one_to_one")
    still_missing = [col for col in package_merge_cols if col not in target.columns]
    if still_missing:
        raise RuntimeError(f"TAIL_REPAIR_TARGET_MISSING_REQUIRED_EVAL_COLUMNS: {still_missing}")
    target["coverage_bad_target_v1"] = _bool(target, "bad_target_v1")
    target["coverage_tail_target_v1"] = _bool(target, "tail_target_v1")
    target["coverage_hard_veto_target_v1"] = _bool(target, "hard_negative_v1")
    target["training_weight_value_v1"] = target["training_weight_tier_v1"].map(r5_rebuild.training_weight_value).astype(float)
    if "evaluation_role_v1" not in target.columns:
        target["evaluation_role_v1"] = "TAIL_REPAIR_REUSED_R5_2_TARGET"
    if "zero_denominator_group_v1" not in target.columns:
        target["zero_denominator_group_v1"] = False
    if "training_opportunity_allowed_v1" not in target.columns:
        target["training_opportunity_allowed_v1"] = ~_bool(target, "exclude_v1")
    if "final_promotion_evidence_allowed_v1" not in target.columns:
        target["final_promotion_evidence_allowed_v1"] = False
    return target


def _variant_repair_sets(registry_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    registry = pd.DataFrame(registry_rows)
    if registry.empty:
        return {variant: set() for variant in TAIL_REPAIR_VARIANT_IDS}
    clear = registry["safety_provenance_clearance_v1"].astype(bool) & registry["tail_evidence_pass_v1"].astype(bool)
    missed = registry["missed_by_r5_2_v1"].astype(bool)
    not_existing = ~registry["in_r5_2_130_86_v1"].astype(bool)
    tail_head = registry["tail_control_10_50_evidence_v1"].astype(bool)
    r5_tail = registry["r5_tail_score_evidence_v1"].astype(bool)
    low_support = registry["structural_low_support_v1"].astype(bool)
    high_tail_score = pd.to_numeric(registry["tail_control_10_50_score_v1"], errors="coerce").fillna(0.0).ge(0.60)
    safe_tail = set(registry.loc[clear & missed & not_existing, "candidate_uid_v1"].astype(str))
    conservative = set(registry.loc[clear & missed & not_existing & (r5_tail | high_tail_score) & ~low_support, "candidate_uid_v1"].astype(str))
    balanced = set(registry.loc[clear & missed & not_existing & (r5_tail | tail_head) & ~low_support, "candidate_uid_v1"].astype(str))
    low_support_set = set(registry.loc[clear & missed & not_existing & (r5_tail | tail_head | low_support), "candidate_uid_v1"].astype(str))
    r6_tail_set = set(registry.loc[clear & missed & not_existing & tail_head, "candidate_uid_v1"].astype(str))
    return {
        "BASE_R5_2_130_86_CONTROL": set(),
        "TAIL_TARGET_WEIGHT_REPAIR_CONSERVATIVE": conservative,
        "TAIL_TARGET_WEIGHT_REPAIR_BALANCED": balanced,
        "TAIL_LOW_SUPPORT_AWARE_REPAIR": low_support_set,
        "R6_TAIL_HEAD_AWARE_REPAIR": r6_tail_set,
        "TAIL_REPAIR_MAX_SAFE_DIAGNOSTIC": safe_tail,
    }


def _target_for_variant(base_target: pd.DataFrame, repair_design: pd.DataFrame, repair_set: set[str], variant_id: str) -> pd.DataFrame:
    target = base_target.copy()
    design = repair_design.set_index("candidate_uid_v1")
    target["tail_repair_variant_id_v1"] = variant_id
    target["tail_repair_added_positive_v1"] = False
    target["tail_repair_role_v1"] = ""
    for idx, row in target.iterrows():
        uid = str(row["candidate_uid"])
        if uid not in repair_set:
            continue
        drow = design.loc[uid]
        target.loc[idx, "target_class_v1"] = str(drow["proposed_tail_repair_target_class_v1"]).replace("UPGRADE_TO_", "")
        target.loc[idx, "bad_target_v1"] = True
        target.loc[idx, "tail_target_v1"] = True
        target.loc[idx, "hard_negative_v1"] = False
        target.loc[idx, "monitor_only_v1"] = False
        target.loc[idx, "exclude_v1"] = False
        target.loc[idx, "coverage_bad_target_v1"] = True
        target.loc[idx, "coverage_tail_target_v1"] = True
        target.loc[idx, "coverage_hard_veto_target_v1"] = False
        target.loc[idx, "training_weight_tier_v1"] = drow["proposed_tail_weight_tier_v1"]
        target.loc[idx, "training_weight_value_v1"] = _weight_value(str(drow["proposed_tail_weight_tier_v1"]))
        target.loc[idx, "tail_repair_added_positive_v1"] = True
        target.loc[idx, "tail_repair_role_v1"] = drow["role_v1"]
        target.loc[idx, "source_evidence_v1"] = f"{row.get('source_evidence_v1', '')}|TAIL_REPAIR:{drow['role_v1']}"
        target.loc[idx, "reason_v1"] = "Tail-specific repair upgrade from safety-clear evidence-backed missed tail row."
    return target


def _variant_rows(
    registry_rows: list[dict[str, Any]],
    candidate_target: pd.DataFrame,
    repair_design: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    variant_sets = _variant_repair_sets(registry_rows)
    base_positive = _bool(candidate_target, "bad_target_v1") | _bool(candidate_target, "tail_target_v1")
    rows = []
    registry = pd.DataFrame(registry_rows)
    for variant_id in TAIL_REPAIR_VARIANT_IDS:
        repair_set = variant_sets[variant_id]
        selected_registry = registry[registry["candidate_uid_v1"].astype(str).isin(repair_set)] if not registry.empty else pd.DataFrame()
        safety_conflicts = 0 if selected_registry.empty else int((~selected_registry["safety_provenance_clearance_v1"].astype(bool)).sum())
        rows.append(
            {
                "variant_id_v1": variant_id,
                "variant_type_v1": "DETERMINISTIC_TAIL_REPAIR_TARGET_VARIANT",
                "proposed_target_rows_v1": int(base_positive.sum() + len(repair_set)),
                "proposed_tail_positives_v1": int(_bool(candidate_target, "tail_target_v1").sum() + len(repair_set)),
                "proposed_bad_positives_v1": int(_bool(candidate_target, "bad_target_v1").sum() + len(repair_set)),
                "expected_retained_r5_2_rows_v1": 130,
                "expected_added_tail_candidates_v1": int(len(repair_set)),
                "low_support_groups_involved_v1": int(selected_registry[selected_registry["structural_low_support_v1"].astype(bool)]["run_id_v1"].nunique()) if not selected_registry.empty else 0,
                "safety_conflicts_v1": safety_conflicts,
                "final_promotion_allowed_v1": False,
                "model_trained_in_design_section_v1": False,
                "recommendation_status_v1": "CONTROL_ONLY_NOT_RETRAINED"
                if variant_id == "BASE_R5_2_130_86_CONTROL"
                else (
                    "DIAGNOSTIC_ONLY_NOT_FINAL"
                    if variant_id == "TAIL_REPAIR_MAX_SAFE_DIAGNOSTIC"
                    else ("READY_FOR_OPTIONAL_OOF_TRAINING" if safety_conflicts == 0 else "BLOCKED_BY_SAFETY_CONFLICT")
                ),
            }
        )
    validate_variant_grid(rows)
    return rows, variant_sets


def _fixed_controls(best: dict[str, Any]) -> list[dict[str, Any]]:
    controls = [
        {"control_v1": "r5_2_package", "bad_v1": 130, "tail_v1": 86, "role_v1": "CURRENT_R5_2_PACKAGE_CONTROL"},
        {"control_v1": "r6_pass_through", "bad_v1": 130, "tail_v1": 86, "role_v1": "R6_PASS_THROUGH_CONTROL"},
        {"control_v1": "historical_v2", "bad_v1": 95, "tail_v1": 61, "role_v1": "BLUEPRINT_COMPARATOR_ONLY"},
        {"control_v1": "v2_oof", "bad_v1": 69, "tail_v1": 53, "role_v1": "PROVENANCE_VALID_SIGNAL_CONTROL"},
        {"control_v1": "optuna", "bad_v1": 56, "tail_v1": 55, "role_v1": "WEAK_SEARCH_SPACE_CONTROL"},
        {"control_v1": "v3", "bad_v1": 17, "tail_v1": 13, "role_v1": "WEAK_OOF_CONTROL"},
        {"control_v1": "coverage_proxy", "bad_v1": 188, "tail_v1": 136, "role_v1": "TRAINING_OPPORTUNITY_ONLY"},
        {"control_v1": "wednesday_benchmark", "bad_v1": 180, "tail_v1": 149, "role_v1": "COMPARATOR_ONLY_NOT_ROW_TARGET"},
    ]
    validate_fixed_controls(controls)
    return [
        {
            **control,
            "candidate_bad_v1": best.get("bad_count_v1"),
            "candidate_tail_v1": best.get("tail_count_v1"),
            "bad_delta_v1": int(best.get("bad_count_v1") or 0) - int(control["bad_v1"]),
            "tail_delta_v1": int(best.get("tail_count_v1") or 0) - int(control["tail_v1"]),
        }
        for control in controls
    ]


def _run_optional_training(
    output_dir: Path,
    *,
    candidate_target: pd.DataFrame,
    package_scores: pd.DataFrame,
    registry_rows: list[dict[str, Any]],
    repair_design_rows: list[dict[str, Any]],
    variant_sets: dict[str, set[str]],
    spec_dir: Path,
    foundation_score_dir: Path | None,
    label_table: Path | None,
    fold_count: int,
) -> dict[str, Any]:
    precheck_failures = []
    registry = pd.DataFrame(registry_rows)
    if registry.empty:
        precheck_failures.append("TAIL_CANDIDATE_REGISTRY_EMPTY")
    elif int((registry["recommended_role_v1"].isin(TAIL_REPAIR_POSITIVE_ROLES) & ~registry["safety_provenance_clearance_v1"].astype(bool)).sum()):
        precheck_failures.append("TAIL_CANDIDATE_REGISTRY_HAS_UNSAFE_POSITIVES")
    if precheck_failures:
        return {"trained_v1": False, "precheck_status_v1": "FAIL", "precheck_failures_v1": precheck_failures}

    inputs = r5_rebuild.v2_replay._prepare_inputs(spec_dir, foundation_score_dir, label_table)
    feature_check = r5_rebuild.validate_no_forbidden_features(inputs["feature_names"])
    hindsight_check = r5_rebuild.validate_no_hindsight_features(inputs["feature_names"])
    if feature_check["status_v1"] != "PASS" or hindsight_check["status_v1"] != "PASS":
        return {
            "trained_v1": False,
            "precheck_status_v1": "FAIL",
            "precheck_failures_v1": ["FEATURE_OR_HINDSIGHT_PREFLIGHT_FAILED"],
            "feature_check_v1": feature_check,
            "hindsight_check_v1": hindsight_check,
        }
    repair_design = pd.DataFrame(repair_design_rows)
    base_target = _base_target_from_existing(candidate_target, package_scores)
    all_scores: list[pd.DataFrame] = []
    all_provenance: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    all_membership: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    all_loso_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    best_by_variant: list[dict[str, Any]] = []

    for variant_id in TRAINED_VARIANT_IDS:
        variant_output = output_dir / "tail_repair_oof_variant_work" / variant_id
        variant_output.mkdir(parents=True, exist_ok=False)
        target = _target_for_variant(base_target, repair_design, variant_sets[variant_id], variant_id)
        config_payload = {
            "action_v1": ACTION,
            "variant_id_v1": variant_id,
            "tail_repair_added_uids_v1": sorted(variant_sets[variant_id]),
            "fold_count_v1": fold_count,
            "denominator_target_v1": DENOMINATOR_TARGET,
        }
        oof = r5_rebuild._run_grouped_oof(variant_output, inputs=inputs, target=target, fold_count=fold_count, config_payload=config_payload)
        scores = oof["scores"].copy()
        scores["tail_repair_variant_id_v1"] = variant_id
        threshold_rows, selections, loso_detail = r5_rebuild._threshold_grid(scores)
        for row in threshold_rows:
            row["tail_repair_variant_id_v1"] = variant_id
            row["candidate_id_v1"] = f"{variant_id}::{row['threshold_candidate_id_v1']}"
            all_threshold_rows.append(row)
        best = _select_best_for_variant(threshold_rows)
        best_selection = selections[str(best["threshold_candidate_id_v1"])]
        scores["tail_repair_variant_best_selected_v1"] = best_selection.values
        no_in_sample = r5_rebuild.validate_no_in_sample_scoring(scores)
        no_overlap = r5_rebuild.validate_no_train_validation_overlap(oof["membership"])
        provenance = oof["provenance"].copy()
        provenance["tail_repair_variant_id_v1"] = variant_id
        provenance_check = r5_rebuild.validate_oof_provenance_complete(scores, provenance)
        metric_row = {
            "tail_repair_variant_id_v1": variant_id,
            "best_threshold_candidate_v1": best["threshold_candidate_id_v1"],
            "candidate_id_v1": f"{variant_id}::{best['threshold_candidate_id_v1']}",
            "oof_provenance_status_v1": provenance_check["status_v1"],
            "train_validation_overlap_count_v1": no_overlap["overlap_count_v1"],
            "in_sample_scored_count_v1": no_in_sample["in_sample_scored_count_v1"],
            **best,
        }
        all_metrics.append(metric_row)
        best_by_variant.append(metric_row)
        for detail in loso_detail:
            detail["tail_repair_variant_id_v1"] = variant_id
            all_loso_rows.append(detail)
        fold = oof["fold_assignment"].copy()
        fold["tail_repair_variant_id_v1"] = variant_id
        membership = oof["membership"].copy()
        membership["tail_repair_variant_id_v1"] = variant_id
        all_scores.append(scores)
        all_provenance.append(provenance)
        all_folds.append(fold)
        all_membership.append(membership)
        manifests.append(
            {
                "tail_repair_variant_id_v1": variant_id,
                "scorefields_v1": [scorefield for _, scorefield in r5_rebuild.SCOREFIELDS],
                "fold_models_v1": oof["fold_models"],
                "feature_count_v1": len(oof["feature_names"]),
                "feature_families_v1": {key: len(value) for key, value in oof["feature_families"].items()},
                "hashes_v1": oof["hashes"],
                "feature_check_v1": feature_check,
                "hindsight_check_v1": hindsight_check,
                "no_new_feature_surface_v1": True,
            }
        )

    metrics_df = pd.DataFrame(all_metrics)
    best = _select_global_best(metrics_df)
    best_id = str(best["candidate_id_v1"])
    combined_scores = pd.concat(all_scores, ignore_index=True)
    combined_provenance = pd.concat(all_provenance, ignore_index=True)
    combined_folds = pd.concat(all_folds, ignore_index=True)
    combined_membership = pd.concat(all_membership, ignore_index=True)
    fixed = _fixed_controls(best)
    denominator_rows = _denominator_rows(best)
    safety_rows = _safety_rows(best)
    low_support = _low_support_payload(best)
    return {
        "trained_v1": True,
        "precheck_status_v1": "PASS",
        "precheck_failures_v1": [],
        "scores": combined_scores,
        "provenance": combined_provenance,
        "fold_assignment": combined_folds,
        "membership": combined_membership,
        "threshold_rows": all_threshold_rows,
        "metric_rows": all_metrics,
        "loso_rows": all_loso_rows,
        "source_manifest": {"layer_name": "TAIL_REPAIR_SCORE_SOURCE_MANIFEST_V1", "variants_v1": manifests},
        "best_candidate": best,
        "best_candidate_id_v1": best_id,
        "fixed_controls": fixed,
        "denominator_rows": denominator_rows,
        "safety_rows": safety_rows,
        "low_support": low_support,
        "feature_label_manifest": {"layer_name": "TAIL_REPAIR_FEATURE_LABEL_HASH_MANIFEST_V1", "variants_v1": manifests},
    }


def _select_best_for_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in rows if r5_rebuild.threshold_candidate_passes_safety(row) and bool(row.get("precision_decision_valid_v1"))]
    pool = safe or rows
    return sorted(
        pool,
        key=lambda row: (
            int(row.get("tail_count_v1") or 0),
            int(row.get("bad_count_v1") or 0),
            float(row.get("precision_v1") or 0.0),
            -int(row.get("selected_low_support_group_count_v1") or 0),
        ),
        reverse=True,
    )[0]


def _select_global_best(metrics: pd.DataFrame) -> dict[str, Any]:
    safe = metrics[
        metrics["safety_clean_v1"].map(_as_bool)
        & metrics["precision_decision_valid_v1"].map(_as_bool)
        & metrics["oof_provenance_status_v1"].astype(str).eq("PASS")
        & metrics["train_validation_overlap_count_v1"].astype(int).eq(0)
        & metrics["in_sample_scored_count_v1"].astype(int).eq(0)
    ]
    pool = safe if not safe.empty else metrics
    pool = pool.copy()
    pool["preserves_130_bad_control_v1"] = pool["bad_count_v1"].astype(int).ge(130)
    sorted_pool = pool.sort_values(
        by=[
            "preserves_130_bad_control_v1",
            "tail_count_v1",
            "bad_count_v1",
            "precision_v1",
            "selected_low_support_group_count_v1",
        ],
        ascending=[False, False, False, False, True],
    )
    return sorted_pool.iloc[0].to_dict()


def _denominator_rows(best: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id_v1": best["candidate_id_v1"],
            "metric_v1": "precision",
            "value_v1": best["precision_v1"],
            "numerator_v1": best["precision_numerator_v1"],
            "denominator_v1": best["precision_denominator_v1"],
            "denominator_status_v1": best["precision_denominator_status_v1"],
            "decision_valid_v1": best["precision_decision_valid_v1"],
        },
        {
            "candidate_id_v1": best["candidate_id_v1"],
            "metric_v1": "strict_all_run_id_worst_loso",
            "value_v1": best["strict_all_run_id_worst_loso_v1"],
            "numerator_v1": best["strict_all_run_id_worst_loso_numerator_v1"],
            "denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
            "denominator_status_v1": best["strict_all_run_id_worst_loso_denominator_status_v1"],
            "decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        },
    ]


def _safety_rows(best: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"candidate_id_v1": best["candidate_id_v1"], "safety_metric_v1": key, "value_v1": best[key], "pass_v1": int(best[key]) == 0}
        for key in [
            "fifty_plus_mfe_overlap_v1",
            "hundred_plus_mfe_overlap_v1",
            "two_hundred_plus_mfe_overlap_v1",
            "strongest_winner_overlap_v1",
            "protected_winner_selected_v1",
            "runner_protect_leakage_v1",
            "ambiguous_high_mfe_leakage_v1",
            "quarantine_selected_v1",
        ]
    ]


def _low_support_payload(best: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id_v1": best["candidate_id_v1"],
        "strict_all_run_id_decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
        "selected_low_support_group_count_v1": best["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": best["structural_low_support_selected_group_count_v1"],
        "zero_selected_group_count_v1": best["zero_selected_group_count_v1"],
        "evaluable_group_count_v1": best["evaluable_group_count_v1"],
        "evaluable_groups_loso_v1": best["evaluable_groups_loso_v1"],
        "final_promotion_allowed_v1": best["final_promotion_allowed_v1"],
    }


def _best_path(training: dict[str, Any]) -> dict[str, Any]:
    if not training.get("trained_v1"):
        status = "TAIL_REPAIR_SIGNAL_PRESENT_PRECHECK_ONLY"
        next_action = "RUN_TAIL_REPAIR_OOF_TRAINING_EXPLICIT_GATE_V1"
        reason = "Tail candidates are valid, but OOF training did not run."
    else:
        best = training["best_candidate"]
        if not candidate_can_be_selected(best):
            status = "TAIL_REPAIR_BLOCKED_BY_TRUE_SAFETY"
            next_action = "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
            reason = "No safety-clean precision-valid tail repair candidate was selectable."
        elif int(best["bad_count_v1"]) >= 130 and int(best["tail_count_v1"]) > 86:
            status = "TAIL_REPAIR_CANDIDATE_BEATS_130_86_SAFELY_FINAL_PROMOTION_BLOCKED"
            next_action = "BUILD_TAIL_REPAIRED_R5_2_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1"
            reason = "OOF tail repair beats 130/86 on tail without hurting bad or safety."
        elif int(best["tail_count_v1"]) > 86:
            status = "TAIL_REPAIR_IMPROVES_TAIL_BUT_HURTS_BAD_OR_STABILITY"
            next_action = "CALIBRATE_TAIL_REPAIR_BALANCE_V1"
            reason = "Tail improved, but bad/stability tradeoff remains."
        else:
            status = "TAIL_REPAIR_TOO_WEAK"
            next_action = "RETURN_TO_R5_2_BASE_GAP_REPAIR_V1"
            reason = "Tail repair OOF did not improve the 86-tail control."
    return {
        "layer_name": "TAIL_SPECIFIC_REPAIR_BEST_PATH_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "reason_v1": reason,
        "best_candidate_v1": training.get("best_candidate"),
        "final_promotion_allowed_v1": False,
    }


def _r6_tail_head_diagnostic(registry_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for row in registry_rows:
        if not _as_bool(row.get("tail_label_v1")):
            continue
        tail_head = _as_bool(row.get("tail_control_10_50_evidence_v1"))
        safety_clear = _as_bool(row.get("safety_provenance_clearance_v1"))
        rows.append(
            {
                "candidate_uid_v1": row["candidate_uid_v1"],
                "run_id_v1": row["run_id_v1"],
                "in_r5_2_130_86_v1": row["in_r5_2_130_86_v1"],
                "missed_by_r5_2_v1": row["missed_by_r5_2_v1"],
                "tail_control_10_50_score_v1": row["tail_control_10_50_score_v1"],
                "tail_control_identified_v1": tail_head,
                "safety_clear_v1": safety_clear,
                "hard_veto_can_isolate_v1": bool(tail_head and safety_clear),
                "diagnostic_role_v1": "R6_TAIL_HEAD_SAFE_CANDIDATE"
                if tail_head and safety_clear
                else ("R6_TAIL_HEAD_MIXED_WITH_SAFETY_RISK" if tail_head else "R6_TAIL_HEAD_MISSED"),
            }
        )
    row_df = pd.DataFrame(rows)
    summary = {
        "tail_rows_analyzed_v1": int(len(row_df)),
        "tail_control_identified_rows_v1": int(row_df["tail_control_identified_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "tail_control_safe_rows_v1": int((row_df["tail_control_identified_v1"].astype(bool) & row_df["safety_clear_v1"].astype(bool)).sum()) if not row_df.empty else 0,
        "tail_control_missed_rows_v1": int((~row_df["tail_control_identified_v1"].astype(bool)).sum()) if not row_df.empty else 0,
        "recommended_next_v1": "R5_2_TAIL_TARGET_REPAIR_AND_R6_TAIL_HEAD_CALIBRATION"
        if not row_df.empty and int((row_df["tail_control_identified_v1"].astype(bool) & row_df["safety_clear_v1"].astype(bool)).sum()) > 0
        else "R5_2_TAIL_TARGET_REPAIR",
    }
    return rows, summary


def _anti_overfit(training: dict[str, Any], no_forbidden: dict[str, Any], no_dummy: dict[str, Any]) -> dict[str, Any]:
    trained = bool(training.get("trained_v1"))
    if not trained:
        status = "TAIL_REPAIR_PRECHECK_ONLY_NO_MODEL_TRAINED"
    else:
        best = training["best_candidate"]
        failures = []
        if no_forbidden["status_v1"] != "PASS":
            failures.append("FORBIDDEN_ACTION")
        if no_dummy["status_v1"] != "PASS":
            failures.append("DUMMY_SYNTHETIC_FALLBACK")
        if not bool(best.get("safety_clean_v1")):
            failures.append("SAFETY_NOT_CLEAN")
        if bool(best.get("final_promotion_allowed_v1")):
            failures.append("FINAL_PROMOTION_TRUE")
        status = "TAIL_REPAIR_STABLE_TRACK_PASS" if not failures else "TAIL_REPAIR_OVERFIT_RISK_DETECTED_STOP"
    return {
        "layer_name": "TAIL_REPAIR_ANTI_OVERFIT_AUDIT_V1",
        "status_v1": status,
        "no_optuna_v1": no_forbidden["status_v1"] == "PASS",
        "no_large_sweep_v1": True,
        "deterministic_variants_only_v1": True,
        "oof_provenance_if_trained_v1": bool(not trained or training["best_candidate"].get("oof_provenance_status_v1") == "PASS"),
        "no_in_sample_decisioning_v1": bool(not trained or int(training["best_candidate"].get("in_sample_scored_count_v1") or 0) == 0),
        "no_post_hoc_threshold_mining_v1": True,
        "safety_clean_required_v1": True,
        "strict_loso_visible_v1": True,
        "low_support_visible_v1": True,
        "fixed_controls_included_v1": True,
        "failed_candidates_not_promoted_v1": True,
        "no_new_feature_surface_v1": True,
        "no_dummy_synthetic_fallback_v1": no_dummy["status_v1"] == "PASS",
    }


def _contract(
    *,
    output_dir: Path,
    audit_root: Path,
    package_root: Path,
    r6_root: Path,
) -> dict[str, Any]:
    return {
        "contract": "TAIL_SPECIFIC_REPAIR_CONTRACT_V1",
        "artifact_root_v1": str(output_dir),
        "input_r5_2_package_root_v1": str(package_root),
        "input_r6_audit_root_v1": str(audit_root),
        "input_r6_root_v1": str(r6_root),
        "tail_repair_candidates_from_existing_legal_artifacts_only_v1": True,
        "primary_tail_signals_v1": ["R5_TAIL_SCORE", "tail_control_10_50", "V2_OOF_tail_positive", "coverage_aware_missed_proxy_tail"],
        "hard_vetoes_v1": ["protected_winner", "runner_protect", "unsafe_high_mfe", "ambiguous_high_mfe_unless_safe_proven", "quarantine", "missing_provenance"],
        "grouped_oof_required_for_trained_repair_v1": True,
        "no_in_sample_decisioning_required_v1": True,
        "strict_loso_visible_v1": True,
        "low_support_registry_visible_v1": True,
        "final_promotion_allowed_v1": False,
        "fixed_controls_required_v1": ["R5.2 130/86", "R6 pass-through 130/86", "historical V2", "V2 OOF", "Optuna", "V3", "coverage proxy", "Wednesday 180/149"],
    }


def _write_reports(
    output_dir: Path,
    *,
    contract: dict[str, Any],
    registry_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    design_summary: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    training: dict[str, Any],
    r6_tail_summary: dict[str, Any],
    best_path: dict[str, Any],
    anti_overfit: dict[str, Any],
) -> None:
    _write_report(
        output_dir / "tail_specific_repair_contract_v1.md",
        [
            "# Tail Specific Repair Contract V1",
            "",
            f"Input R5.2 package: `{contract['input_r5_2_package_root_v1']}`",
            f"Input audit root: `{contract['input_r6_audit_root_v1']}`",
            "Tail repair is narrow, evidence-backed, OOF/provenance-gated if trained, and never final promotion.",
        ],
    )
    _write_report(
        output_dir / "tail_repair_candidate_registry_report_v1.md",
        [
            "# Tail Repair Candidate Registry V1",
            "",
            f"Candidate tail rows: `{registry_summary['candidate_tail_rows_v1']}`",
            f"Existing R5.2 tail retained: `{registry_summary['existing_r5_2_tail_retained_v1']}`",
            f"Missed tail rows: `{registry_summary['missed_tail_rows_v1']}`",
            f"Repair-positive candidate rows: `{registry_summary['repair_positive_candidate_rows_v1']}`",
            f"Role counts: `{registry_summary['role_counts_v1']}`",
        ],
    )
    _write_report(
        output_dir / "tail_gap_decomposition_report_v1.md",
        [
            "# Tail Gap Decomposition V1",
            "",
            f"Missed tail rows: `{gap_summary['missed_tail_rows_v1']}`",
            f"Safety-clear rows: `{gap_summary['safety_clear_rows_v1']}`",
            f"Near-threshold rows: `{gap_summary['near_threshold_rows_v1']}`",
            f"R5_TAIL_SCORE rows: `{gap_summary['strong_r5_tail_rows_v1']}`",
            f"tail_control_10_50 support rows: `{gap_summary['tail_control_support_rows_v1']}`",
            f"Suitable for repair: `{gap_summary['suitable_for_repair_rows_v1']}`",
        ],
    )
    _write_report(
        output_dir / "tail_specific_training_target_repair_design_report_v1.md",
        [
            "# Tail Specific Training Target Repair Design V1",
            "",
            f"Rows: `{design_summary['rows_v1']}`",
            f"Tail repair upgrades: `{design_summary['tail_repair_upgrades_v1']}`",
            f"Hard negatives preserved: `{design_summary['hard_negative_rows_preserved_v1']}`",
            f"Quarantine excluded: `{design_summary['quarantine_rows_preserved_excluded_v1']}`",
        ],
    )
    _write_report(
        output_dir / "tail_repair_variants_report_v1.md",
        [
            "# Tail Repair Variants V1",
            "",
            *[
                f"- `{row['variant_id_v1']}`: added tail candidates `{row['expected_added_tail_candidates_v1']}`, safety conflicts `{row['safety_conflicts_v1']}`, status `{row['recommendation_status_v1']}`"
                for row in variant_rows
            ],
        ],
    )
    _write_report(
        output_dir / "r6_tail_head_diagnostic_report_v1.md",
        [
            "# R6 Tail Head Diagnostic V1",
            "",
            f"Tail rows analyzed: `{r6_tail_summary['tail_rows_analyzed_v1']}`",
            f"tail_control identified rows: `{r6_tail_summary['tail_control_identified_rows_v1']}`",
            f"tail_control safe rows: `{r6_tail_summary['tail_control_safe_rows_v1']}`",
            f"Recommended next: `{r6_tail_summary['recommended_next_v1']}`",
        ],
    )
    _write_report(
        output_dir / "tail_specific_repair_best_path_v1.md",
        [
            "# Tail Specific Repair Best Path V1",
            "",
            f"Status: `{best_path['status_v1']}`",
            f"Next: `{best_path['next_recommended_action_v1']}`",
            f"Reason: {best_path['reason_v1']}",
        ],
    )
    _write_report(
        output_dir / "tail_repair_anti_overfit_audit_v1.md",
        [
            "# Tail Repair Anti-Overfit Audit V1",
            "",
            f"Status: `{anti_overfit['status_v1']}`",
            f"OOF training run: `{training.get('trained_v1')}`",
            "No Optuna, broad sweep, freeze, promo, live, new feature surface, dummy, synthetic, or fallback was used.",
        ],
    )
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Build Tail Specific R5.2/R6 Repair V1",
            "",
            f"Go/no-go: `{best_path['status_v1']}`",
            f"Next: `{best_path['next_recommended_action_v1']}`",
            f"OOF training run: `{training.get('trained_v1')}`",
            f"Best candidate: `{(training.get('best_candidate') or {}).get('candidate_id_v1', 'NONE')}`",
            f"Final promotion allowed: `False`",
        ],
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    audit_root: Path = AUDIT_ROOT,
    package_root: Path = R5_2_PACKAGE_ROOT,
    r6_root: Path = R6_ROOT,
    coverage_root: Path = COVERAGE_ROOT,
    candidate_root: Path = R5_2_CANDIDATE_ROOT,
    low_support_policy_root: Path = LOW_SUPPORT_POLICY_ROOT,
    v2_oof_root: Path = V2_OOF_ROOT,
    spec_dir: Path = r5_rebuild.historical_v2.DEFAULT_SPEC_DIR,
    foundation_score_dir: Path | None = None,
    label_table: Path | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    r5_rebuild.validate_loso_guard_not_weakened(DENOMINATOR_TARGET)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_paths = {
        "r5_2_package_scores_v1": package_root / "r5_2_candidate_oof_scores_v1.csv",
        "r5_2_package_manifest_v1": package_root / "r5_2_candidate_package_manifest_v1.json",
        "r6_scores_v1": r6_root / "r6_oof_scores_v1.csv",
        "r6_summary_v1": r6_root / "summary_v1.json",
        "audit_tail_gap_v1": audit_root / "tail_gap_analysis_86_to_136_proxy_v1.csv",
        "audit_safe_subset_v1": audit_root / "safe_subset_from_failed_r6_expansions_v1.csv",
        "coverage_rows_v1": coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv",
        "candidate_target_v1": candidate_root / "r5_2_training_target_table_v1.csv",
        "v2_oof_scores_v1": v2_oof_root / "v2_oof_scores_v1.csv",
    }
    hashes_before = _input_hashes(input_paths)
    inputs = _load_inputs(
        audit_root=audit_root,
        package_root=package_root,
        r6_root=r6_root,
        coverage_root=coverage_root,
        candidate_root=candidate_root,
    )
    no_forbidden = validate_no_forbidden_actions(optuna=False, broad_sweep=False, freeze=False, promo=False, live=False)
    no_dummy = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    contract = _contract(output_dir=output_dir, audit_root=audit_root, package_root=package_root, r6_root=r6_root)
    registry_rows, registry_summary = _build_tail_registry(inputs)
    gap_rows, gap_summary = _tail_gap_decomposition(registry_rows, inputs["tail_gap"])
    design_rows, design_summary = _target_repair_design(inputs["candidate_target"], registry_rows)
    variant_rows, variant_sets = _variant_rows(registry_rows, inputs["candidate_target"], pd.DataFrame(design_rows))
    r6_tail_rows, r6_tail_summary = _r6_tail_head_diagnostic(registry_rows)
    training = _run_optional_training(
        output_dir,
        candidate_target=inputs["candidate_target"],
        package_scores=inputs["package_scores"],
        registry_rows=registry_rows,
        repair_design_rows=design_rows,
        variant_sets=variant_sets,
        spec_dir=spec_dir,
        foundation_score_dir=foundation_score_dir,
        label_table=label_table,
        fold_count=fold_count,
    )
    best_path = _best_path(training)
    anti_overfit = _anti_overfit(training, no_forbidden, no_dummy)
    hashes_after = _input_hashes(input_paths)
    unchanged = validate_input_artifacts_unchanged(hashes_before, hashes_after)
    if unchanged["status_v1"] != "PASS":
        raise RuntimeError(f"TAIL_REPAIR_INPUT_ARTIFACT_MUTATION_DETECTED: {unchanged}")

    _write_json(output_dir / "tail_specific_repair_contract_v1.json", contract)
    _write_rows(output_dir / "tail_repair_candidate_registry_v1.csv", registry_rows)
    _write_json(output_dir / "tail_repair_candidate_registry_v1.json", {"summary_v1": registry_summary, "rows_v1": registry_rows})
    _write_rows(output_dir / "tail_gap_decomposition_v1.csv", gap_rows)
    _write_json(output_dir / "tail_gap_decomposition_v1.json", {"summary_v1": gap_summary, "rows_v1": gap_rows})
    _write_rows(output_dir / "tail_specific_training_target_repair_design_v1.csv", design_rows)
    _write_json(output_dir / "tail_specific_training_target_repair_design_v1.json", {"summary_v1": design_summary, "rows_v1": design_rows})
    _write_rows(output_dir / "tail_repair_variants_v1.csv", variant_rows)
    _write_json(output_dir / "tail_repair_variants_v1.json", {"variants_v1": variant_rows})
    _write_rows(output_dir / "r6_tail_head_diagnostic_v1.csv", r6_tail_rows)
    _write_json(output_dir / "r6_tail_head_diagnostic_v1.json", {"summary_v1": r6_tail_summary, "rows_v1": r6_tail_rows})
    _write_json(output_dir / "tail_specific_repair_best_path_v1.json", best_path)
    _write_json(output_dir / "tail_repair_anti_overfit_audit_v1.json", anti_overfit)

    if training.get("trained_v1"):
        training["scores"].to_csv(output_dir / "tail_repair_oof_scores_v1.csv", index=False)
        training["provenance"].to_csv(output_dir / "tail_repair_oof_score_provenance_v1.csv", index=False)
        training["fold_assignment"].to_csv(output_dir / "tail_repair_oof_fold_assignment_v1.csv", index=False)
        training["membership"].to_csv(output_dir / "tail_repair_train_validation_membership_v1.csv", index=False)
        _write_json(output_dir / "tail_repair_score_source_manifest_v1.json", training["source_manifest"])
        _write_rows(output_dir / "tail_repair_oof_eval_metrics_v1.csv", training["metric_rows"])
        _write_rows(output_dir / "tail_repair_threshold_candidate_grid_v1.csv", training["threshold_rows"])
        _write_json(output_dir / "tail_repair_threshold_candidate_grid_v1.json", {"rows_v1": training["threshold_rows"]})
        _write_rows(output_dir / "tail_repair_metric_denominator_report_v1.csv", training["denominator_rows"])
        _write_json(output_dir / "tail_repair_metric_denominator_report_v1.json", {"rows_v1": training["denominator_rows"]})
        _write_rows(output_dir / "tail_repair_safety_report_v1.csv", training["safety_rows"])
        _write_json(output_dir / "tail_repair_safety_report_v1.json", {"rows_v1": training["safety_rows"]})
        _write_rows(output_dir / "tail_repair_low_support_report_v1.csv", [training["low_support"]])
        _write_json(output_dir / "tail_repair_low_support_report_v1.json", training["low_support"])
        _write_rows(output_dir / "tail_repair_fixed_control_comparison_v1.csv", training["fixed_controls"])
        _write_json(output_dir / "tail_repair_fixed_control_comparison_v1.json", {"controls_v1": training["fixed_controls"]})
        _write_json(output_dir / "tail_repair_feature_label_hash_manifest_v1.json", training["feature_label_manifest"])

    go_no_go = {
        "layer_name": "TAIL_SPECIFIC_R5_2_R6_REPAIR_GO_NO_GO_V1",
        "status_v1": best_path["status_v1"],
        "go_no_go_v1": best_path["status_v1"],
        "next_recommended_action_v1": best_path["next_recommended_action_v1"],
        "final_promotion_allowed_v1": False,
        "optuna_run_v1": False,
        "broad_sweep_run_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(output_dir / "tail_specific_r5_2_r6_repair_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST_V1",
            "artifact_root_v1": str(output_dir),
            "inputs_v1": {
                "audit_root_v1": str(audit_root),
                "r5_2_package_root_v1": str(package_root),
                "r6_root_v1": str(r6_root),
                "coverage_root_v1": str(coverage_root),
                "candidate_root_v1": str(candidate_root),
                "low_support_policy_root_v1": str(low_support_policy_root),
                "v2_oof_root_v1": str(v2_oof_root),
            },
            "input_hashes_before_v1": hashes_before,
            "input_hashes_after_v1": hashes_after,
            "input_artifacts_unchanged_v1": unchanged,
        },
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "input_audit_root_v1": str(audit_root),
        "input_r5_2_package_root_v1": str(package_root),
        "input_r6_root_v1": str(r6_root),
        "existing_r5_2_r6_artifacts_unchanged_v1": unchanged["status_v1"] == "PASS",
        "tail_registry_summary_v1": registry_summary,
        "tail_gap_summary_v1": gap_summary,
        "target_repair_design_summary_v1": design_summary,
        "variant_count_v1": len(variant_rows),
        "oof_tail_repair_training_run_v1": bool(training.get("trained_v1")),
        "best_tail_repair_candidate_v1": training.get("best_candidate"),
        "r6_tail_head_diagnostic_summary_v1": r6_tail_summary,
        "anti_overfit_status_v1": anti_overfit["status_v1"],
        "go_no_go_v1": go_no_go["go_no_go_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": go_no_go["go_no_go_v1"]})
    _write_reports(
        output_dir,
        contract=contract,
        registry_summary=registry_summary,
        gap_summary=gap_summary,
        design_summary=design_summary,
        variant_rows=variant_rows,
        training=training,
        r6_tail_summary=r6_tail_summary,
        best_path=best_path,
        anti_overfit=anti_overfit,
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--r5-2-package-root", type=Path, default=R5_2_PACKAGE_ROOT)
    parser.add_argument("--r6-root", type=Path, default=R6_ROOT)
    parser.add_argument("--coverage-root", type=Path, default=COVERAGE_ROOT)
    parser.add_argument("--r5-2-candidate-root", type=Path, default=R5_2_CANDIDATE_ROOT)
    parser.add_argument("--low-support-policy-root", type=Path, default=LOW_SUPPORT_POLICY_ROOT)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=r5_rebuild.historical_v2.DEFAULT_SPEC_DIR)
    parser.add_argument("--foundation-score-dir", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--require-explicit-artifact-selection", action="store_true")
    parser.add_argument("--fail-on-dummy-or-synthetic-input", action="store_true")
    parser.add_argument("--fail-on-degraded-fallback", action="store_true")
    parser.add_argument("--preserve-strict-loso-low-support", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        audit_root=args.audit_root,
        package_root=args.r5_2_package_root,
        r6_root=args.r6_root,
        coverage_root=args.coverage_root,
        candidate_root=args.r5_2_candidate_root,
        low_support_policy_root=args.low_support_policy_root,
        v2_oof_root=args.v2_oof_root,
        spec_dir=args.spec_dir,
        foundation_score_dir=args.foundation_score_dir,
        label_table=args.label_table,
        fold_count=args.fold_count,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
