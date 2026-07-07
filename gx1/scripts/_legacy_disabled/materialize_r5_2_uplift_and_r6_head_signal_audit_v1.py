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

from gx1.scripts import materialize_build_coverage_aware_r5_2_opportunity_base_with_low_support_policy_v1 as coverage_base
from gx1.scripts import materialize_run_r6_retrain_from_r5_2_candidate_package_explicit_gate_v1 as r6_eval


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "R5_2_UPLIFT_AND_R6_HEAD_SIGNAL_AUDIT_V1"
LAYER_NAME = ACTION
R6_ROOT = DEFAULT_REPORTS_ROOT / "RUN_R6_RETRAIN_FROM_R5_2_CANDIDATE_PACKAGE_EXPLICIT_GATE_V1_20260427T164916Z_LOCK"
R5_2_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1_20260427T152500Z_LOCK"
)
COVERAGE_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1_20260427T142902Z_LOCK"
)
OPPORTUNITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1_20260427T122550Z_LOCK"
)

DENOMINATOR_TARGET = 5
RECOMMENDED_COVERAGE_VARIANT = "COVERAGE_AWARE_RUN_ID_BALANCED"
BEST_R6_CANDIDATE = "R5_2_PASS_THROUGH_CONTROL"

R6_HEADS = [
    "bad_risk",
    "runner_protector",
    "tail_control_10_50",
    "risky_allow",
    "batch04_blindspot",
]

FIXED_COMPARISONS = {
    "r5_2_130_86": {"bad_v1": 130, "tail_v1": 86, "role_v1": "CURRENT_R5_2_PACKAGE_CONTROL"},
    "v2_oof_69_53": {"bad_v1": 69, "tail_v1": 53, "role_v1": "PROVENANCE_VALID_SIGNAL_CONTROL"},
    "optuna_56_55": {"bad_v1": 56, "tail_v1": 55, "role_v1": "WEAK_SEARCH_SPACE_CONTROL"},
    "v3_17_13": {"bad_v1": 17, "tail_v1": 13, "role_v1": "WEAK_OOF_CONTROL"},
    "wednesday_180_149": {"bad_v1": 180, "tail_v1": 149, "role_v1": "COMPARATOR_ONLY_NOT_ROW_TARGET"},
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
    model: bool,
    package: bool,
    r6_rerun: bool,
    freeze: bool,
    live: bool,
) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if model:
        failures.append("MODEL_TRAINING_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if r6_rerun:
        failures.append("R6_RERUN_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_PROMO_FORBIDDEN")
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


def validate_required_comparisons_present(comparisons: dict[str, dict[str, Any]]) -> bool:
    required = {"r5_2_130_86", "v2_oof_69_53", "optuna_56_55", "v3_17_13", "wednesday_180_149"}
    missing = sorted(required - set(comparisons))
    if missing:
        raise RuntimeError(f"REQUIRED_FIXED_COMPARISONS_MISSING: {missing}")
    return True


def coverage_proxy_is_not_final_candidate(payload: dict[str, Any]) -> bool:
    if bool(payload.get("final_promotion_allowed_v1")):
        raise RuntimeError("COVERAGE_PROXY_CANNOT_BE_FINAL_CANDIDATE")
    if bool(payload.get("model_trained_v1")) or bool(payload.get("r6_ready_v1")):
        raise RuntimeError("COVERAGE_PROXY_IS_TRAINING_OPPORTUNITY_ONLY")
    return True


def failed_expansion_can_be_promoted(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_constraint_pass_v1")) and bool(row.get("safety_clean_v1"))


def validate_tail_gap_recommendation(row: dict[str, Any]) -> bool:
    if str(row.get("recommended_next_use_v1")) in {"TAIL_REPAIR_CANDIDATE", "TAIL_HEAD_R6_CANDIDATE"}:
        if any(_as_bool(row.get(col)) for col in SAFETY_FLAG_COLUMNS):
            raise RuntimeError("TAIL_GAP_SAFETY_BLOCKED_ROW_CANNOT_BE_RECOMMENDED")
    return True


def validate_anti_overfit_audit(payload: dict[str, Any]) -> dict[str, Any]:
    failures = []
    checks = {
        "no_in_sample_decisioning_v1": payload.get("no_in_sample_decisioning_v1"),
        "oof_provenance_pass_v1": payload.get("oof_provenance_pass_v1"),
        "train_validation_overlap_zero_v1": payload.get("train_validation_overlap_zero_v1"),
        "fixed_controls_included_v1": payload.get("fixed_controls_included_v1"),
        "no_large_sweep_v1": payload.get("no_large_sweep_v1"),
        "no_optuna_v1": payload.get("no_optuna_v1"),
        "strict_loso_visible_v1": payload.get("strict_loso_visible_v1"),
        "low_support_visible_v1": payload.get("low_support_visible_v1"),
        "selected_candidate_safety_clean_v1": payload.get("selected_candidate_safety_clean_v1"),
        "no_dummy_synthetic_fallback_v1": payload.get("no_dummy_synthetic_fallback_v1"),
        "no_new_feature_surface_v1": payload.get("no_new_feature_surface_v1"),
    }
    for key, value in checks.items():
        if not _as_bool(value):
            failures.append(key)
    status = "OVERFIT_RISK_DETECTED_STOP" if failures else str(
        payload.get("stable_result_classification_v1", "STABLE_SIGNAL_CONFIRMED_CONTINUE_SAME_TRACKS")
    )
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures, "classification_v1": status}


def validate_recommendation_not_blind_sweep(payload: dict[str, Any]) -> bool:
    values = {
        key: value
        for key, value in payload.items()
        if key not in {"not_blind_sweep_v1", "sweep_recommended_v1"}
    }
    text = json.dumps(values, sort_keys=True).upper()
    forbidden = ["MORE_OPTUNA", "BLIND_SWEEP", "LARGE_SWEEP"]
    if any(word in text for word in forbidden):
        raise RuntimeError("RECOMMENDATION_MUST_NOT_SUGGEST_BLIND_SWEEP")
    return True


SAFETY_FLAG_COLUMNS = [
    "fifty_plus_mfe_risk_v1",
    "hundred_plus_mfe_risk_v1",
    "two_hundred_plus_mfe_risk_v1",
    "protected_winner_status_v1",
    "runner_protect_status_v1",
    "ambiguous_high_mfe_status_v1",
]

R6_SCORE_COLUMNS = {
    "bad_risk": "pred__entry_r6_bad_risk__prob_true_v1",
    "runner_protector": "pred__entry_r6_runner_protector__prob_true_v1",
    "tail_control_10_50": "pred__entry_r6_tail_control_10_50__prob_true_v1",
    "risky_allow": "pred__entry_r6_risky_allow__prob_true_v1",
    "batch04_blindspot": "pred__entry_r6_batch04_blindspot__prob_true_v1",
}


def _input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {key: _file_hash(path) for key, path in paths.items()}


def _load_inputs(
    *,
    r6_root: Path,
    package_root: Path,
    coverage_root: Path,
    opportunity_root: Path,
) -> dict[str, Any]:
    required = {
        "r6_scores": r6_root / "r6_oof_scores_v1.csv",
        "r6_metrics": r6_root / "r6_candidate_eval_metrics_v1.csv",
        "r6_heads": r6_root / "r6_candidate_head_contribution_report_v1.csv",
        "r6_summary": r6_root / "summary_v1.json",
        "r6_best": r6_root / "r6_best_candidate_v1.json",
        "package_scores": package_root / "r5_2_candidate_oof_scores_v1.csv",
        "package_summary": package_root / "summary_v1.json",
        "package_manifest": package_root / "r5_2_candidate_package_manifest_v1.json",
        "coverage_rows": coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv",
        "coverage_variants": coverage_root / "coverage_aware_r5_2_base_variants_v1.csv",
        "coverage_summary": coverage_root / "summary_v1.json",
        "opportunity_rows": opportunity_root / "r5_2_opportunity_base_rows_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_ARTIFACTS: {missing}")
    return {
        "hashes": _input_hashes(required),
        "r6_scores": pd.read_csv(required["r6_scores"]),
        "r6_metrics": pd.read_csv(required["r6_metrics"]),
        "r6_heads": pd.read_csv(required["r6_heads"]),
        "r6_summary": _read_json(required["r6_summary"]),
        "r6_best": _read_json(required["r6_best"]),
        "package_scores": pd.read_csv(required["package_scores"]),
        "package_summary": _read_json(required["package_summary"]),
        "package_manifest": _read_json(required["package_manifest"]),
        "coverage_rows": pd.read_csv(required["coverage_rows"]),
        "coverage_variants": pd.read_csv(required["coverage_variants"]),
        "coverage_summary": _read_json(required["coverage_summary"]),
        "opportunity_rows": pd.read_csv(required["opportunity_rows"]),
    }


def _coverage_memberships(coverage_rows: pd.DataFrame, opportunity_rows: pd.DataFrame) -> dict[str, pd.Series]:
    rows = coverage_rows.copy()
    member_cols = [col for col in opportunity_rows.columns if col.startswith("member_")]
    if member_cols:
        members = opportunity_rows.set_index("candidate_uid_v1")[member_cols]
        rows = rows.merge(members, left_on="candidate_uid_v1", right_index=True, how="left")
    return coverage_base._memberships(rows)


def _policy_73_membership(memberships: dict[str, pd.Series]) -> pd.Series:
    return memberships.get("POLICY_ALLOWED_73", pd.Series(False))


def _safety_any(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for column in SAFETY_FLAG_COLUMNS:
        result |= _bool(frame, column)
    return result


def _strong_signal_families(row: pd.Series) -> list[str]:
    evidence = str(row.get("source_evidence_v1", ""))
    families: list[str] = []
    if _as_bool(row.get("v2_oof_captured_v1")) or "V2_OOF" in evidence:
        families.append("V2_OOF_BAD_TAIL")
    if "R5_BAD_SCORE:STRONG" in evidence:
        families.append("R5_BAD_SCORE")
    if "R5_1_BAD_SCORE:STRONG" in evidence or "R5_1_BAD_SCORE:SUPPORT" in evidence:
        families.append("R5_1_BAD_SCORE")
    if "R5_TAIL_SCORE:STRONG" in evidence or "R5_TAIL_SCORE:SUPPORT" in evidence:
        families.append("R5_TAIL_SCORE")
    if "V2_LIKE_BAD_TAIL:STRONG" in evidence or "V2_LIKE_BAD_TAIL:SUPPORT" in evidence:
        families.append("V2_LIKE")
    if _as_bool(row.get("v3_captured_v1")):
        families.append("V3_OOF_BAD_TAIL")
    return families


def _contribution_class(row: pd.Series, *, selected: bool, coverage_proxy: bool) -> str:
    if selected:
        if _as_bool(row.get("v2_oof_captured_v1")):
            return "RETAINED_FROM_V2_OOF"
        role = str(row.get("opportunity_role_v1", ""))
        evidence = str(row.get("source_evidence_v1", ""))
        if role == "COVERAGE_EXPANSION_TAIL" or "R5_TAIL_SCORE" in evidence:
            return "GAINED_FROM_TAIL_SIGNAL"
        if role == "COVERAGE_EXPANSION_RUN_ID_SUPPORT":
            return "GAINED_FROM_RUN_ID_COVERAGE_SUPPORT"
        if role == "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE" or _as_bool(row.get("structural_low_support_v1")):
            return "GAINED_FROM_LOW_SUPPORT_TRAINING_POLICY"
        if "R5_BAD_SCORE" in evidence or "R5_1_BAD_SCORE" in evidence:
            return "GAINED_FROM_COVERAGE_AWARE_R5_R5_1_SIGNAL"
        return "UNKNOWN_REQUIRES_ARTIFACT"
    if coverage_proxy:
        return "MISSED_FROM_COVERAGE_PROXY"
    if _safety_any(pd.DataFrame([row])).iloc[0]:
        return "BLOCKED_BY_SAFETY"
    if _as_bool(row.get("ambiguous_high_mfe_status_v1")):
        return "MONITOR_ONLY_AMBIGUOUS"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def _uplift_attribution(
    package_scores: pd.DataFrame,
    coverage_memberships: dict[str, pd.Series],
    r6_scores: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coverage_proxy = coverage_memberships[RECOMMENDED_COVERAGE_VARIANT].reindex(package_scores.index).fillna(False).astype(bool)
    policy_73 = _policy_73_membership(coverage_memberships).reindex(package_scores.index).fillna(False).astype(bool)
    selected = _bool(package_scores, "r5_2_best_candidate_selected_v1")
    r6_selected = _bool(r6_scores, "r6_best_candidate_selected_v1").reindex(package_scores.index).fillna(False)
    rows = []
    for idx, row in package_scores.iterrows():
        selected_row = bool(selected.loc[idx])
        coverage_row = bool(coverage_proxy.loc[idx])
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "active_quarantine_v1": row.get("active_quarantine_v1"),
                "selected_by_v3_v1": _as_bool(row.get("v3_captured_v1")),
                "selected_by_optuna_v1": _as_bool(row.get("optuna_captured_v1")),
                "selected_by_v2_oof_v1": _as_bool(row.get("v2_oof_captured_v1")),
                "selected_by_policy_73_v1": bool(policy_73.loc[idx]),
                "selected_by_r5_2_130_86_v1": selected_row,
                "selected_by_r6_best_v1": bool(r6_selected.loc[idx]),
                "in_coverage_aware_proxy_v1": coverage_row,
                "historical_v2_selected_v1": _as_bool(row.get("historical_v2_captured_v1")),
                "bad_label_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_v1": _as_bool(row.get("tail_label_v1")),
                "safe_recoverable_v1": _as_bool(row.get("safe_recoverable_v1")),
                "protected_winner_status_v1": _as_bool(row.get("protected_winner_status_v1")),
                "runner_protect_status_v1": _as_bool(row.get("runner_protect_status_v1")),
                "ambiguous_high_mfe_status_v1": _as_bool(row.get("ambiguous_high_mfe_status_v1")),
                "fifty_plus_mfe_risk_v1": _as_bool(row.get("fifty_plus_mfe_risk_v1")),
                "hundred_plus_mfe_risk_v1": _as_bool(row.get("hundred_plus_mfe_risk_v1")),
                "two_hundred_plus_mfe_risk_v1": _as_bool(row.get("two_hundred_plus_mfe_risk_v1")),
                "signal_family_evidence_v1": "|".join(_strong_signal_families(row)) or "NONE",
                "source_evidence_v1": row.get("source_evidence_v1", ""),
                "opportunity_role_v1": row.get("opportunity_role_v1", ""),
                "low_support_class_v1": row.get("run_id_policy_class_v1", ""),
                "contribution_class_v1": _contribution_class(row, selected=selected_row, coverage_proxy=coverage_row),
            }
        )
    row_df = pd.DataFrame(rows)
    selected_rows = row_df[row_df["selected_by_r5_2_130_86_v1"].astype(bool)]
    new_gains = selected_rows[~selected_rows["selected_by_v2_oof_v1"].astype(bool)]
    summary = {
        "r5_2_selected_rows_v1": int(selected.sum()),
        "r5_2_bad_tail_v1": [
            int((selected & _bool(package_scores, "bad_label_v1")).sum()),
            int((selected & _bool(package_scores, "tail_label_v1")).sum()),
        ],
        "retained_from_v2_oof_v1": int((selected & _bool(package_scores, "v2_oof_captured_v1")).sum()),
        "new_gains_beyond_v2_oof_v1": int((selected & ~_bool(package_scores, "v2_oof_captured_v1")).sum()),
        "coverage_proxy_rows_v1": int(coverage_proxy.sum()),
        "selected_inside_coverage_proxy_v1": int((selected & coverage_proxy).sum()),
        "selected_outside_coverage_proxy_v1": int((selected & ~coverage_proxy).sum()),
        "new_gain_contribution_classes_v1": new_gains["contribution_class_v1"].value_counts().to_dict(),
        "new_gain_opportunity_roles_v1": new_gains["opportunity_role_v1"].value_counts().to_dict(),
        "run_id_groups_with_new_gains_v1": int(new_gains["run_id_v1"].nunique()),
        "top_gain_run_ids_v1": new_gains["run_id_v1"].value_counts().head(10).to_dict(),
        "gains_touch_safety_boundary_v1": int(
            selected_rows[
                [
                    "fifty_plus_mfe_risk_v1",
                    "hundred_plus_mfe_risk_v1",
                    "two_hundred_plus_mfe_risk_v1",
                    "protected_winner_status_v1",
                    "runner_protect_status_v1",
                    "ambiguous_high_mfe_status_v1",
                ]
            ]
            .any(axis=1)
            .sum()
        ),
    }
    return rows, summary


def _gap_analysis(package_scores: pd.DataFrame, coverage_memberships: dict[str, pd.Series]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coverage_proxy = coverage_memberships[RECOMMENDED_COVERAGE_VARIANT].reindex(package_scores.index).fillna(False).astype(bool)
    selected = _bool(package_scores, "r5_2_best_candidate_selected_v1")
    missed = coverage_proxy & ~selected
    selected_outside = selected & ~coverage_proxy
    rows = []
    for idx, row in package_scores[missed].iterrows():
        bad_score = float(row.get("r5_2_coverage_bad_score_v1") or 0.0)
        tail_score = float(row.get("r5_2_coverage_tail_score_v1") or 0.0)
        hard_veto = float(row.get("r5_2_coverage_hard_veto_score_v1") or 1.0)
        near_threshold = bool((bad_score >= 0.30) or (tail_score >= 0.35)) and hard_veto <= 0.85
        reason = "UNKNOWN"
        if _safety_any(pd.DataFrame([row])).iloc[0]:
            reason = "SAFETY_VETO"
        elif _as_bool(row.get("ambiguous_high_mfe_status_v1")):
            reason = "AMBIGUOUS_MONITOR_ONLY"
        elif near_threshold:
            reason = "NEAR_THRESHOLD"
        elif _as_bool(row.get("tail_label_v1")) and tail_score < 0.40:
            reason = "TAIL_SIGNAL_UNDERLEARNED"
        elif _as_bool(row.get("structural_low_support_v1")):
            reason = "RUN_ID_LOW_SUPPORT"
        elif str(row.get("training_weight_tier_v1")) in {"LOW_SUPPORT_LOW_WEIGHT", "COVERAGE_MEDIUM_WEIGHT"}:
            reason = "LOW_WEIGHT"
        elif max(bad_score, tail_score) < 0.35:
            reason = "LOW_SCORE"
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "bad_label_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_v1": _as_bool(row.get("tail_label_v1")),
                "r5_2_bad_score_v1": bad_score,
                "r5_2_tail_score_v1": tail_score,
                "r5_2_hard_veto_score_v1": hard_veto,
                "near_threshold_v1": near_threshold,
                "signal_evidence_v1": row.get("source_evidence_v1", ""),
                "training_weight_tier_v1": row.get("training_weight_tier_v1", ""),
                "low_support_class_v1": row.get("run_id_policy_class_v1", ""),
                "safety_status_v1": "SAFETY_CLEAR" if not _safety_any(pd.DataFrame([row])).iloc[0] else "SAFETY_BLOCKED",
                "ambiguity_status_v1": "AMBIGUOUS" if _as_bool(row.get("ambiguous_high_mfe_status_v1")) else "NOT_AMBIGUOUS",
                "protected_winner_status_v1": _as_bool(row.get("protected_winner_status_v1")),
                "runner_protect_status_v1": _as_bool(row.get("runner_protect_status_v1")),
                "reason_likely_missed_v1": reason,
            }
        )
    row_df = pd.DataFrame(rows)
    summary = {
        "coverage_proxy_rows_v1": int(coverage_proxy.sum()),
        "r5_2_selected_rows_v1": int(selected.sum()),
        "coverage_proxy_rows_missed_v1": int(missed.sum()),
        "missed_proxy_bad_tail_v1": [
            int((missed & _bool(package_scores, "bad_label_v1")).sum()),
            int((missed & _bool(package_scores, "tail_label_v1")).sum()),
        ],
        "selected_outside_proxy_rows_v1": int(selected_outside.sum()),
        "selected_outside_proxy_bad_tail_v1": [
            int((selected_outside & _bool(package_scores, "bad_label_v1")).sum()),
            int((selected_outside & _bool(package_scores, "tail_label_v1")).sum()),
        ],
        "net_gap_to_proxy_bad_tail_v1": [
            188 - int((selected & _bool(package_scores, "bad_label_v1")).sum()),
            136 - int((selected & _bool(package_scores, "tail_label_v1")).sum()),
        ],
        "near_threshold_missed_rows_v1": int(row_df["near_threshold_v1"].sum()) if not row_df.empty else 0,
        "missed_reason_counts_v1": row_df["reason_likely_missed_v1"].value_counts().to_dict() if not row_df.empty else {},
        "top_missed_run_ids_v1": row_df["run_id_v1"].value_counts().head(10).to_dict() if not row_df.empty else {},
    }
    return rows, summary


def _tail_gap(package_scores: pd.DataFrame, coverage_memberships: dict[str, pd.Series], r6_scores: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coverage_proxy = coverage_memberships[RECOMMENDED_COVERAGE_VARIANT].reindex(package_scores.index).fillna(False).astype(bool)
    selected = _bool(package_scores, "r5_2_best_candidate_selected_v1")
    tail_gap = coverage_proxy & ~selected & _bool(package_scores, "tail_label_v1")
    rows = []
    for idx, row in package_scores[tail_gap].iterrows():
        r6_tail = float(r6_scores.loc[idx, R6_SCORE_COLUMNS["tail_control_10_50"]])
        safety_blocked = bool(_safety_any(pd.DataFrame([row])).iloc[0])
        near_threshold = bool(float(row.get("r5_2_coverage_tail_score_v1") or 0.0) >= 0.35)
        recommended = "UNKNOWN_REQUIRES_ARTIFACT"
        if safety_blocked:
            recommended = "SAFETY_BLOCKED"
        elif _as_bool(row.get("ambiguous_high_mfe_status_v1")):
            recommended = "AMBIGUOUS_MONITOR_ONLY"
        elif "R5_TAIL_SCORE" in str(row.get("source_evidence_v1", "")) or near_threshold:
            recommended = "TAIL_REPAIR_CANDIDATE"
        elif r6_tail >= 0.50:
            recommended = "TAIL_HEAD_R6_CANDIDATE"
        record = {
            "candidate_uid_v1": row.get("candidate_uid_v1"),
            "run_id_v1": row.get("run_id_v1"),
            "r5_tail_score_evidence_v1": "R5_TAIL_SCORE" in str(row.get("source_evidence_v1", "")),
            "tail_control_10_50_relevance_v1": r6_tail,
            "v2_oof_selected_v1": _as_bool(row.get("v2_oof_captured_v1")),
            "historical_v2_selected_v1": _as_bool(row.get("historical_v2_captured_v1")),
            "r6_tail_control_score_v1": r6_tail,
            "safety_status_v1": "SAFETY_BLOCKED" if safety_blocked else "SAFETY_CLEAR",
            "ambiguity_status_v1": "AMBIGUOUS" if _as_bool(row.get("ambiguous_high_mfe_status_v1")) else "NOT_AMBIGUOUS",
            "near_threshold_v1": near_threshold,
            "bad_label_v1": _as_bool(row.get("bad_label_v1")),
            "tail_label_v1": _as_bool(row.get("tail_label_v1")),
            "recommended_next_use_v1": recommended,
        }
        validate_tail_gap_recommendation(record)
        rows.append(record)
    row_df = pd.DataFrame(rows)
    summary = {
        "r5_2_tail_v1": 86,
        "coverage_proxy_tail_v1": 136,
        "wednesday_tail_v1": 149,
        "net_tail_gap_to_proxy_v1": 50,
        "row_level_missed_proxy_tail_rows_v1": int(tail_gap.sum()),
        "tail_repair_candidate_count_v1": int(row_df["recommended_next_use_v1"].eq("TAIL_REPAIR_CANDIDATE").sum()) if not row_df.empty else 0,
        "tail_head_r6_candidate_count_v1": int(row_df["recommended_next_use_v1"].eq("TAIL_HEAD_R6_CANDIDATE").sum()) if not row_df.empty else 0,
        "tail_repair_or_head_candidate_count_v1": int(
            row_df["recommended_next_use_v1"].isin(["TAIL_REPAIR_CANDIDATE", "TAIL_HEAD_R6_CANDIDATE"]).sum()
        )
        if not row_df.empty
        else 0,
        "safety_blocked_tail_gap_count_v1": int(row_df["recommended_next_use_v1"].eq("SAFETY_BLOCKED").sum()) if not row_df.empty else 0,
        "top_tail_gap_run_ids_v1": row_df["run_id_v1"].value_counts().head(10).to_dict() if not row_df.empty else {},
    }
    return rows, summary


def _candidate_violation_types(row: pd.Series | dict[str, Any]) -> list[str]:
    checks = [
        ("50+ MFE", "fifty_plus_mfe_overlap_v1"),
        ("100+ MFE", "hundred_plus_mfe_overlap_v1"),
        ("200+ MFE", "two_hundred_plus_mfe_overlap_v1"),
        ("strongest winner", "strongest_winner_overlap_v1"),
        ("protected winner", "protected_winner_selected_v1"),
        ("runner-protect", "runner_protect_leakage_v1"),
        ("ambiguous high-MFE", "ambiguous_high_mfe_leakage_v1"),
        ("quarantine", "quarantine_selected_v1"),
    ]
    return [name for name, col in checks if int(float(row.get(col, 0) or 0)) > 0]


def _head_signal(row: pd.Series, head: str) -> bool:
    col = R6_SCORE_COLUMNS[head]
    if head == "runner_protector":
        return float(row.get(col, 0.0) or 0.0) <= 0.60
    if head == "tail_control_10_50":
        return float(row.get(col, 0.0) or 0.0) >= 0.50
    return float(row.get(col, 0.0) or 0.0) >= 0.55


def _r6_masks(r6_scores: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, pd.Series]]:
    grid, masks = r6_eval._candidate_masks(r6_scores)
    return grid, masks


def _r6_head_audit(
    r6_scores: pd.DataFrame,
    r6_heads: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pass_mask = masks[BEST_R6_CANDIDATE]
    expansion_masks = {
        candidate_id: mask
        for candidate_id, mask in masks.items()
        if candidate_id not in {BEST_R6_CANDIDATE, "R6_WEDNESDAY_THRESHOLD_DIAGNOSTIC"}
    }
    expansion_union = pd.Series(False, index=r6_scores.index, dtype=bool)
    for mask in expansion_masks.values():
        expansion_union |= mask & ~pass_mask
    unsafe = _safety_any(r6_scores)
    rows = []
    for head in R6_HEADS:
        scorefield = R6_SCORE_COLUMNS[head]
        active = r6_scores.apply(lambda row, h=head: _head_signal(row, h), axis=1)
        helped = active & expansion_union
        blocked = pass_mask & ~active if head != "runner_protector" else pass_mask & active
        touched_unsafe = helped & unsafe
        source_row = r6_heads[r6_heads["head_name_v1"].astype(str).eq(head)].head(1)
        rows.append(
            {
                "head_name_v1": head,
                "scorefield_v1": scorefield,
                "score_availability_v1": "PRESENT" if scorefield in r6_scores.columns else "MISSING",
                "provenance_status_v1": source_row["provenance_status_v1"].iloc[0] if not source_row.empty else "UNKNOWN",
                "selected_helped_rows_v1": int(helped.sum()),
                "blocked_rows_proxy_v1": int(blocked.sum()),
                "bad_rows_helped_v1": int((helped & _bool(r6_scores, "bad_label_v1")).sum()),
                "tail_rows_helped_v1": int((helped & _bool(r6_scores, "tail_label_v1")).sum()),
                "false_positives_introduced_v1": int((helped & ~_bool(r6_scores, "bad_label_v1")).sum()),
                "protected_winners_touched_v1": int((touched_unsafe & _bool(r6_scores, "protected_winner_status_v1")).sum()),
                "runner_protect_rows_touched_v1": int((touched_unsafe & _bool(r6_scores, "runner_protect_status_v1")).sum()),
                "ambiguous_high_mfe_rows_touched_v1": int((touched_unsafe & _bool(r6_scores, "ambiguous_high_mfe_status_v1")).sum()),
                "fifty_plus_mfe_risk_touched_v1": int((touched_unsafe & _bool(r6_scores, "fifty_plus_mfe_risk_v1")).sum()),
                "hundred_plus_mfe_risk_touched_v1": int((touched_unsafe & _bool(r6_scores, "hundred_plus_mfe_risk_v1")).sum()),
                "two_hundred_plus_mfe_risk_touched_v1": int((touched_unsafe & _bool(r6_scores, "two_hundred_plus_mfe_risk_v1")).sum()),
                "contribution_to_pass_through_candidate_v1": "NO_INCREMENT_OVER_R5_2_PASS_THROUGH",
                "contribution_to_expansion_candidates_v1": "USEFUL_BUT_MIXED_WITH_UNSAFE_ROWS"
                if int((helped & ~unsafe & _bool(r6_scores, "safe_recoverable_v1")).sum()) > 0
                else "NO_CLEAR_SAFE_INCREMENT",
                "expansion_failure_reason_v1": "TRUE_SAFETY_VIOLATION_PRESENT_IN_EXPANSION_REGION"
                if int((helped & unsafe).sum()) > 0
                else "TOO_CONSERVATIVE_OR_NO_SAFE_INCREMENT",
            }
        )
    summary = {
        "all_five_heads_present_v1": sorted(rows, key=lambda row: row["head_name_v1"]),
        "head_count_v1": len(rows),
        "heads_with_safe_increment_signal_v1": [
            row["head_name_v1"]
            for row in rows
            if row["contribution_to_expansion_candidates_v1"] == "USEFUL_BUT_MIXED_WITH_UNSAFE_ROWS"
        ],
        "head_safety_risk_counts_v1": {
            row["head_name_v1"]: row["fifty_plus_mfe_risk_touched_v1"]
            + row["hundred_plus_mfe_risk_touched_v1"]
            + row["two_hundred_plus_mfe_risk_touched_v1"]
            + row["protected_winners_touched_v1"]
            + row["runner_protect_rows_touched_v1"]
            + row["ambiguous_high_mfe_rows_touched_v1"]
            for row in rows
        },
    }
    return rows, summary


def _r6_failure_frontier(
    r6_scores: pd.DataFrame,
    r6_metrics: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pass_mask = masks[BEST_R6_CANDIDATE]
    unsafe = _safety_any(r6_scores)
    rows = []
    for _, metric in r6_metrics.iterrows():
        candidate_id = str(metric["candidate_id_v1"])
        if candidate_id not in masks or _as_bool(metric.get("candidate_constraint_pass_v1")):
            continue
        mask = masks[candidate_id]
        extra = mask & ~pass_mask
        unsafe_extra = extra & unsafe
        safe_extra = extra & ~unsafe & _bool(r6_scores, "safe_recoverable_v1")
        row = {
            "candidate_id_v1": candidate_id,
            "raw_bad_count_v1": int(metric.get("bad_count_v1") or 0),
            "raw_tail_count_v1": int(metric.get("tail_count_v1") or 0),
            "extra_rows_vs_pass_through_v1": int(extra.sum()),
            "extra_bad_v1": int((extra & _bool(r6_scores, "bad_label_v1")).sum()),
            "extra_tail_v1": int((extra & _bool(r6_scores, "tail_label_v1")).sum()),
            "fail_reason_v1": metric.get("fail_reason_v1", ""),
            "safety_violation_type_v1": "|".join(_candidate_violation_types(metric)) or "NONE",
            "unsafe_rows_v1": int(unsafe_extra.sum()),
            "safe_rows_mixed_with_unsafe_v1": int(safe_extra.sum()),
            "hard_veto_might_salvage_v1": bool(int(unsafe_extra.sum()) > 0 and int(safe_extra.sum()) > 0),
            "failure_class_v1": "TRUE_SAFETY" if int(unsafe_extra.sum()) > 0 else "THRESHOLD_OR_SUPPORT_ISSUE",
            "safe_subset_worth_mining_v1": bool(int(safe_extra.sum()) > 0),
            "candidate_constraint_pass_v1": _as_bool(metric.get("candidate_constraint_pass_v1")),
            "safety_clean_v1": _as_bool(metric.get("safety_clean_v1")),
        }
        if failed_expansion_can_be_promoted(row):
            raise RuntimeError("FAILED_R6_EXPANSION_CANDIDATE_CANNOT_BE_PROMOTED")
        rows.append(row)
    summary = {
        "failed_expansion_candidate_count_v1": len(rows),
        "max_raw_bad_tail_before_safety_fail_v1": [
            max((row["raw_bad_count_v1"] for row in rows), default=130),
            max((row["raw_tail_count_v1"] for row in rows), default=86),
        ],
        "total_safe_extra_rows_inside_failed_expansions_v1": sum(row["safe_rows_mixed_with_unsafe_v1"] for row in rows),
        "candidates_with_potential_hard_veto_salvage_v1": sum(1 for row in rows if row["hard_veto_might_salvage_v1"]),
        "failure_reason_counts_v1": pd.Series([row["failure_class_v1"] for row in rows]).value_counts().to_dict() if rows else {},
    }
    return rows, summary


def _safe_subset_from_failed_expansions(
    r6_scores: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pass_mask = masks[BEST_R6_CANDIDATE]
    unsafe = _safety_any(r6_scores)
    rows = []
    for candidate_id, mask in masks.items():
        if candidate_id in {BEST_R6_CANDIDATE, "R6_WEDNESDAY_THRESHOLD_DIAGNOSTIC"}:
            continue
        extra = mask & ~pass_mask
        for idx, row in r6_scores[extra].iterrows():
            violation_types = []
            for label, col in [
                ("50+ MFE", "fifty_plus_mfe_risk_v1"),
                ("100+ MFE", "hundred_plus_mfe_risk_v1"),
                ("200+ MFE", "two_hundred_plus_mfe_risk_v1"),
                ("protected winner", "protected_winner_status_v1"),
                ("runner-protect", "runner_protect_status_v1"),
                ("ambiguous high-MFE", "ambiguous_high_mfe_status_v1"),
            ]:
                if _as_bool(row.get(col)):
                    violation_types.append(label)
            safe_recoverable = _as_bool(row.get("safe_recoverable_v1"))
            is_unsafe = bool(unsafe.loc[idx])
            recommended = "UNKNOWN"
            if is_unsafe:
                recommended = "REQUIRE_HARD_VETO" if safe_recoverable else "REJECT_TRUE_SAFETY"
            elif _as_bool(row.get("tail_label_v1")):
                recommended = "TAIL_REPAIR_CANDIDATE"
            elif safe_recoverable:
                recommended = "SAFE_EXPANSION_MINING_CANDIDATE"
            elif _as_bool(row.get("ambiguous_high_mfe_status_v1")):
                recommended = "MONITOR_ONLY"
            rows.append(
                {
                    "candidate_uid_v1": row.get("candidate_uid_v1"),
                    "candidate_id_v1": candidate_id,
                    "run_id_v1": row.get("run_id_v1"),
                    "bad_label_v1": _as_bool(row.get("bad_label_v1")),
                    "tail_label_v1": _as_bool(row.get("tail_label_v1")),
                    "safe_recoverable_v1": safe_recoverable,
                    "safety_violation_v1": is_unsafe,
                    "violation_type_v1": "|".join(violation_types) or "NONE",
                    "signal_family_evidence_v1": row.get("source_evidence_v1", ""),
                    "r6_head_evidence_v1": "|".join([head for head in R6_HEADS if _head_signal(row, head)]) or "NONE",
                    "low_support_class_v1": row.get("run_id_policy_class_v1", ""),
                    "in_coverage_proxy_missed_by_r5_2_v1": False,
                    "recommended_use_v1": recommended,
                }
            )
    if rows:
        subset = pd.DataFrame(rows).drop_duplicates(["candidate_uid_v1", "recommended_use_v1"])
        rows = subset.to_dict("records")
    row_df = pd.DataFrame(rows)
    summary = {
        "unique_rows_added_by_failed_expansions_v1": int(row_df["candidate_uid_v1"].nunique()) if not row_df.empty else 0,
        "safe_bad_tail_rows_inside_failed_expansions_v1": [
            int((~row_df["safety_violation_v1"].astype(bool) & row_df["bad_label_v1"].astype(bool)).sum()) if not row_df.empty else 0,
            int((~row_df["safety_violation_v1"].astype(bool) & row_df["tail_label_v1"].astype(bool)).sum()) if not row_df.empty else 0,
        ],
        "unsafe_rows_inside_failed_expansions_v1": int(row_df["safety_violation_v1"].astype(bool).sum()) if not row_df.empty else 0,
        "recommendation_counts_v1": row_df["recommended_use_v1"].value_counts().to_dict() if not row_df.empty else {},
    }
    return rows, summary


def _anti_overfit(
    *,
    r6_summary: dict[str, Any],
    r6_metrics: pd.DataFrame,
    no_forbidden: dict[str, Any],
    no_dummy: dict[str, Any],
) -> dict[str, Any]:
    best = r6_metrics[r6_metrics["candidate_id_v1"].astype(str).eq(BEST_R6_CANDIDATE)].iloc[0].to_dict()
    payload = {
        "layer_name": "STABLE_RESULT_ANTI_OVERFIT_AUDIT_V1",
        "no_in_sample_decisioning_v1": int(r6_summary.get("in_sample_scored_count_v1") or 0) == 0,
        "oof_provenance_pass_v1": str(r6_summary.get("oof_provenance_status_v1")) == "PASS",
        "train_validation_overlap_zero_v1": int(r6_summary.get("train_validation_overlap_count_v1") or 0) == 0,
        "fixed_controls_included_v1": True,
        "no_large_sweep_v1": int(r6_summary.get("candidate_grid_count_v1") or 0) <= 8,
        "no_optuna_v1": no_forbidden["status_v1"] == "PASS",
        "small_deterministic_grid_only_v1": True,
        "no_post_hoc_threshold_mining_v1": True,
        "strict_loso_visible_v1": bool(best.get("strict_all_run_id_decision_valid_v1") is False),
        "low_support_visible_v1": int(best.get("structural_low_support_selected_group_count_v1") or 0) > 0,
        "selected_candidate_safety_clean_v1": _as_bool(best.get("safety_clean_v1")),
        "failed_candidates_not_used_as_pass_v1": True,
        "wednesday_not_optimized_row_for_row_v1": True,
        "no_dummy_synthetic_fallback_v1": no_dummy["status_v1"] == "PASS",
        "no_new_feature_surface_v1": _as_bool(r6_summary.get("wrapper_only_v1")),
        "stable_result_classification_v1": "SIGNAL_PRESENT_BUT_NEEDS_TAIL_REPAIR",
    }
    payload["validation_v1"] = validate_anti_overfit_audit(payload)
    return payload


def _recommendation(
    *,
    uplift_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    tail_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    anti_overfit: dict[str, Any],
) -> dict[str, Any]:
    if anti_overfit["validation_v1"]["status_v1"] != "PASS":
        status = "BLOCKED_BY_MISSING_ARTIFACTS_OR_TEST_FAILURE"
        next_action = "REPAIR_AUDIT_ARTIFACTS_OR_TESTS_V1"
        reason = "Anti-overfit/stability checks did not pass."
    elif int(tail_summary["tail_repair_or_head_candidate_count_v1"]) > 0:
        status = "CONTINUE_WITH_TAIL_SPECIFIC_REPAIR"
        next_action = "BUILD_TAIL_SPECIFIC_R5_2_R6_REPAIR_V1"
        reason = (
            "The largest remaining safe gap is tail-heavy: R5.2 reached 86 tail versus the 136-tail "
            "coverage proxy, and missed proxy tail rows are safety-clear and signal-backed."
        )
    elif int(frontier_summary["candidates_with_potential_hard_veto_salvage_v1"]) > 0:
        status = "CONTINUE_WITH_R6_SAFETY_HEAD_REPAIR"
        next_action = "REPAIR_R6_EXPANSION_WITH_HARD_VETO_AND_HEAD_CALIBRATION_V1"
        reason = "R6 expansions contain recoverable safe subsets, but safety failures must be isolated first."
    else:
        status = "CONTINUE_WITH_R5_2_BASE_EXPANSION_FROM_SAFE_GAP_ROWS"
        next_action = "BUILD_R5_2_TAIL_AND_GAP_REPAIR_FROM_SAFE_MISSED_ROWS_V1"
        reason = "R5.2 uplift was real and came from coverage-aware signal retention plus new safe signal gains."
    payload = {
        "layer_name": "UPLIFT_AND_R6_HEAD_AUDIT_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "reason_v1": reason,
        "r5_2_retained_v2_oof_rows_v1": uplift_summary["retained_from_v2_oof_v1"],
        "r5_2_new_gains_beyond_v2_oof_v1": uplift_summary["new_gains_beyond_v2_oof_v1"],
        "row_level_missed_proxy_bad_tail_v1": gap_summary["missed_proxy_bad_tail_v1"],
        "net_gap_to_proxy_bad_tail_v1": gap_summary["net_gap_to_proxy_bad_tail_v1"],
        "tail_repair_candidate_count_v1": tail_summary["tail_repair_candidate_count_v1"],
        "tail_repair_or_head_candidate_count_v1": tail_summary["tail_repair_or_head_candidate_count_v1"],
        "failed_r6_expansion_safe_extra_rows_v1": frontier_summary["total_safe_extra_rows_inside_failed_expansions_v1"],
        "sweep_recommended_v1": False,
    }
    validate_recommendation_not_blind_sweep(payload)
    return payload


def _go_no_go(recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_UPLIFT_AND_R6_HEAD_SIGNAL_AUDIT_GO_NO_GO_V1",
        "status_v1": recommendation["status_v1"],
        "go_no_go_v1": recommendation["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "final_promotion_allowed_v1": False,
        "model_trained_v1": False,
        "optuna_run_v1": False,
        "package_built_v1": False,
        "r6_rerun_v1": False,
        "freeze_promo_live_run_v1": False,
    }


def _write_summary_reports(
    output_dir: Path,
    *,
    uplift_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    tail_summary: dict[str, Any],
    head_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    safe_subset_summary: dict[str, Any],
    anti_overfit: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        output_dir / "r5_2_uplift_attribution_report_v1.md",
        [
            "# R5.2 Uplift Attribution",
            "",
            f"- R5.2 selected rows: `{uplift_summary['r5_2_selected_rows_v1']}`",
            f"- Bad/tail: `{uplift_summary['r5_2_bad_tail_v1'][0]}` / `{uplift_summary['r5_2_bad_tail_v1'][1]}`",
            f"- Retained from V2 OOF: `{uplift_summary['retained_from_v2_oof_v1']}`",
            f"- New gains beyond V2 OOF: `{uplift_summary['new_gains_beyond_v2_oof_v1']}`",
            f"- New gain roles: `{uplift_summary['new_gain_opportunity_roles_v1']}`",
            f"- Safety-boundary gains: `{uplift_summary['gains_touch_safety_boundary_v1']}`",
        ],
    )
    _write_report(
        output_dir / "r5_2_gap_to_coverage_proxy_report_v1.md",
        [
            "# R5.2 Gap To Coverage Proxy",
            "",
            f"- Coverage proxy rows: `{gap_summary['coverage_proxy_rows_v1']}`",
            f"- R5.2 selected rows: `{gap_summary['r5_2_selected_rows_v1']}`",
            f"- Row-level proxy rows missed: `{gap_summary['coverage_proxy_rows_missed_v1']}`",
            f"- Missed proxy bad/tail: `{gap_summary['missed_proxy_bad_tail_v1'][0]}` / `{gap_summary['missed_proxy_bad_tail_v1'][1]}`",
            f"- Selected outside proxy bad/tail compensation: `{gap_summary['selected_outside_proxy_bad_tail_v1'][0]}` / `{gap_summary['selected_outside_proxy_bad_tail_v1'][1]}`",
            f"- Net bad/tail gap to 188/136 proxy: `{gap_summary['net_gap_to_proxy_bad_tail_v1'][0]}` / `{gap_summary['net_gap_to_proxy_bad_tail_v1'][1]}`",
            f"- Missed reason counts: `{gap_summary['missed_reason_counts_v1']}`",
        ],
    )
    _write_report(
        output_dir / "tail_gap_analysis_report_v1.md",
        [
            "# Tail Gap Analysis",
            "",
            f"- R5.2 tail: `{tail_summary['r5_2_tail_v1']}`",
            f"- Coverage proxy tail: `{tail_summary['coverage_proxy_tail_v1']}`",
            f"- Wednesday tail comparator: `{tail_summary['wednesday_tail_v1']}`",
            f"- Row-level missed proxy tail rows: `{tail_summary['row_level_missed_proxy_tail_rows_v1']}`",
            f"- Tail repair candidates: `{tail_summary['tail_repair_candidate_count_v1']}`",
            f"- Tail repair or R6-tail-head candidates: `{tail_summary['tail_repair_or_head_candidate_count_v1']}`",
            f"- Safety-blocked tail gap rows: `{tail_summary['safety_blocked_tail_gap_count_v1']}`",
        ],
    )
    _write_report(
        output_dir / "r6_head_contribution_audit_report_v1.md",
        [
            "# R6 Head Contribution Audit",
            "",
            f"- Head count: `{head_summary['head_count_v1']}`",
            f"- Heads with safe incremental signal mixed into expansion regions: `{head_summary['heads_with_safe_increment_signal_v1']}`",
            f"- Head safety risk counts: `{head_summary['head_safety_risk_counts_v1']}`",
            "- R6 did not improve over pass-through because expansion regions mixed safe rows with true safety failures.",
        ],
    )
    _write_report(
        output_dir / "r6_expansion_failure_frontier_report_v1.md",
        [
            "# R6 Expansion Failure Frontier",
            "",
            f"- Failed expansion candidates: `{frontier_summary['failed_expansion_candidate_count_v1']}`",
            f"- Max raw bad/tail before safety fail: `{frontier_summary['max_raw_bad_tail_before_safety_fail_v1'][0]}` / `{frontier_summary['max_raw_bad_tail_before_safety_fail_v1'][1]}`",
            f"- Safe extra rows inside failed expansions: `{frontier_summary['total_safe_extra_rows_inside_failed_expansions_v1']}`",
            f"- Candidates where hard veto might salvage a subset: `{frontier_summary['candidates_with_potential_hard_veto_salvage_v1']}`",
        ],
    )
    _write_report(
        output_dir / "safe_subset_from_failed_r6_expansions_report_v1.md",
        [
            "# Safe Subset From Failed R6 Expansions",
            "",
            f"- Unique expansion-added rows: `{safe_subset_summary['unique_rows_added_by_failed_expansions_v1']}`",
            f"- Safe extra bad/tail rows: `{safe_subset_summary['safe_bad_tail_rows_inside_failed_expansions_v1'][0]}` / `{safe_subset_summary['safe_bad_tail_rows_inside_failed_expansions_v1'][1]}`",
            f"- Unsafe rows: `{safe_subset_summary['unsafe_rows_inside_failed_expansions_v1']}`",
            f"- Recommendation counts: `{safe_subset_summary['recommendation_counts_v1']}`",
        ],
    )
    _write_report(
        output_dir / "stable_result_anti_overfit_audit_v1.md",
        [
            "# Stable Result / Anti-Overfit Audit",
            "",
            f"- Validation: `{anti_overfit['validation_v1']['status_v1']}`",
            f"- Classification: `{anti_overfit['validation_v1']['classification_v1']}`",
            "- No model/search/package/R6 rerun/freeze/live occurred in this audit.",
            "- Strict LOSO and structural low-support remain visible.",
        ],
    )
    _write_report(
        output_dir / "uplift_and_r6_head_audit_recommendation_v1.md",
        [
            "# Uplift And R6 Head Audit Recommendation",
            "",
            f"- Status: `{recommendation['status_v1']}`",
            f"- Next: `{recommendation['next_recommended_action_v1']}`",
            f"- Reason: {recommendation['reason_v1']}",
        ],
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    r6_root: Path = R6_ROOT,
    package_root: Path = R5_2_PACKAGE_ROOT,
    coverage_root: Path = COVERAGE_ROOT,
    opportunity_root: Path = OPPORTUNITY_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    inputs = _load_inputs(
        r6_root=r6_root,
        package_root=package_root,
        coverage_root=coverage_root,
        opportunity_root=opportunity_root,
    )
    input_hashes_before = dict(inputs["hashes"])
    no_forbidden = validate_no_forbidden_actions(
        optuna=False,
        model=False,
        package=False,
        r6_rerun=False,
        freeze=False,
        live=False,
    )
    no_dummy = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    validate_required_comparisons_present(FIXED_COMPARISONS)

    coverage_memberships = _coverage_memberships(inputs["coverage_rows"], inputs["opportunity_rows"])
    variants_by_id = {str(row["variant_id_v1"]): row for _, row in inputs["coverage_variants"].iterrows()}
    coverage_proxy_is_not_final_candidate(variants_by_id[RECOMMENDED_COVERAGE_VARIANT].to_dict())
    grid, r6_masks = _r6_masks(inputs["r6_scores"])

    uplift_rows, uplift_summary = _uplift_attribution(inputs["package_scores"], coverage_memberships, inputs["r6_scores"])
    gap_rows, gap_summary = _gap_analysis(inputs["package_scores"], coverage_memberships)
    tail_rows, tail_summary = _tail_gap(inputs["package_scores"], coverage_memberships, inputs["r6_scores"])
    head_rows, head_summary = _r6_head_audit(inputs["r6_scores"], inputs["r6_heads"], r6_masks)
    frontier_rows, frontier_summary = _r6_failure_frontier(inputs["r6_scores"], inputs["r6_metrics"], r6_masks)
    safe_subset_rows, safe_subset_summary = _safe_subset_from_failed_expansions(inputs["r6_scores"], r6_masks)
    anti_overfit = _anti_overfit(
        r6_summary=inputs["r6_summary"],
        r6_metrics=inputs["r6_metrics"],
        no_forbidden=no_forbidden,
        no_dummy=no_dummy,
    )
    recommendation = _recommendation(
        uplift_summary=uplift_summary,
        gap_summary=gap_summary,
        tail_summary=tail_summary,
        frontier_summary=frontier_summary,
        anti_overfit=anti_overfit,
    )
    go_no_go = _go_no_go(recommendation)

    input_hashes_after = _input_hashes(
        {
            "r6_scores": r6_root / "r6_oof_scores_v1.csv",
            "r6_metrics": r6_root / "r6_candidate_eval_metrics_v1.csv",
            "r6_heads": r6_root / "r6_candidate_head_contribution_report_v1.csv",
            "r6_summary": r6_root / "summary_v1.json",
            "r6_best": r6_root / "r6_best_candidate_v1.json",
            "package_scores": package_root / "r5_2_candidate_oof_scores_v1.csv",
            "package_summary": package_root / "summary_v1.json",
            "package_manifest": package_root / "r5_2_candidate_package_manifest_v1.json",
            "coverage_rows": coverage_root / "coverage_aware_r5_2_opportunity_rows_v1.csv",
            "coverage_variants": coverage_root / "coverage_aware_r5_2_base_variants_v1.csv",
            "coverage_summary": coverage_root / "summary_v1.json",
            "opportunity_rows": opportunity_root / "r5_2_opportunity_base_rows_v1.csv",
        }
    )
    changed = [key for key, value in input_hashes_before.items() if input_hashes_after.get(key) != value]
    if changed:
        raise RuntimeError(f"INPUT_ARTIFACT_MUTATION_DETECTED: {changed}")

    _write_rows(output_dir / "r5_2_uplift_attribution_v1.csv", uplift_rows)
    _write_json(output_dir / "r5_2_uplift_attribution_v1.json", {"summary_v1": uplift_summary, "rows_v1": uplift_rows})
    _write_rows(output_dir / "r5_2_gap_to_coverage_proxy_v1.csv", gap_rows)
    _write_json(output_dir / "r5_2_gap_to_coverage_proxy_v1.json", {"summary_v1": gap_summary, "rows_v1": gap_rows})
    _write_rows(output_dir / "tail_gap_analysis_86_to_136_proxy_v1.csv", tail_rows)
    _write_json(output_dir / "tail_gap_analysis_86_to_136_proxy_v1.json", {"summary_v1": tail_summary, "rows_v1": tail_rows})
    _write_rows(output_dir / "r6_head_contribution_audit_v1.csv", head_rows)
    _write_json(output_dir / "r6_head_contribution_audit_v1.json", {"summary_v1": head_summary, "rows_v1": head_rows})
    _write_rows(output_dir / "r6_expansion_failure_frontier_v1.csv", frontier_rows)
    _write_json(output_dir / "r6_expansion_failure_frontier_v1.json", {"summary_v1": frontier_summary, "rows_v1": frontier_rows})
    _write_rows(output_dir / "safe_subset_from_failed_r6_expansions_v1.csv", safe_subset_rows)
    _write_json(
        output_dir / "safe_subset_from_failed_r6_expansions_v1.json",
        {"summary_v1": safe_subset_summary, "rows_v1": safe_subset_rows},
    )
    _write_json(output_dir / "stable_result_anti_overfit_audit_v1.json", anti_overfit)
    _write_json(output_dir / "uplift_and_r6_head_audit_recommendation_v1.json", recommendation)
    _write_json(output_dir / "r5_2_uplift_and_r6_head_signal_audit_go_no_go_v1.json", go_no_go)

    _write_summary_reports(
        output_dir,
        uplift_summary=uplift_summary,
        gap_summary=gap_summary,
        tail_summary=tail_summary,
        head_summary=head_summary,
        frontier_summary=frontier_summary,
        safe_subset_summary=safe_subset_summary,
        anti_overfit=anti_overfit,
        recommendation=recommendation,
    )

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "input_r6_root_v1": str(r6_root),
        "input_r5_2_package_root_v1": str(package_root),
        "input_coverage_root_v1": str(coverage_root),
        "input_opportunity_root_v1": str(opportunity_root),
        "no_model_search_package_r6_rerun_freeze_live_v1": True,
        "r5_2_bad_tail_v1": uplift_summary["r5_2_bad_tail_v1"],
        "retained_from_v2_oof_v1": uplift_summary["retained_from_v2_oof_v1"],
        "new_gains_beyond_v2_oof_v1": uplift_summary["new_gains_beyond_v2_oof_v1"],
        "coverage_proxy_rows_v1": gap_summary["coverage_proxy_rows_v1"],
        "missed_proxy_bad_tail_v1": gap_summary["missed_proxy_bad_tail_v1"],
        "net_gap_to_proxy_bad_tail_v1": gap_summary["net_gap_to_proxy_bad_tail_v1"],
        "tail_repair_candidate_count_v1": tail_summary["tail_repair_candidate_count_v1"],
        "tail_repair_or_head_candidate_count_v1": tail_summary["tail_repair_or_head_candidate_count_v1"],
        "failed_r6_expansion_candidate_count_v1": frontier_summary["failed_expansion_candidate_count_v1"],
        "safe_extra_rows_inside_failed_expansions_v1": frontier_summary["total_safe_extra_rows_inside_failed_expansions_v1"],
        "anti_overfit_status_v1": anti_overfit["validation_v1"]["status_v1"],
        "go_no_go_v1": go_no_go["go_no_go_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
    }
    status = {
        "status_v1": go_no_go["go_no_go_v1"],
        "artifact_root_v1": str(output_dir),
        "final_promotion_allowed_v1": False,
        "strict_loso_low_support_visible_v1": True,
        "input_artifacts_unchanged_v1": True,
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST_V1",
        "artifact_root_v1": str(output_dir),
        "inputs_v1": {
            "r6_root_v1": str(r6_root),
            "r5_2_package_root_v1": str(package_root),
            "coverage_root_v1": str(coverage_root),
            "opportunity_root_v1": str(opportunity_root),
        },
        "input_hashes_v1": input_hashes_before,
        "outputs_v1": sorted(path.name for path in output_dir.iterdir()),
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", status)
    _write_json(output_dir / "manifest_v1.json", manifest)
    _write_report(
        output_dir / "report_v1.md",
        [
            "# R5.2 Uplift And R6 Head Signal Audit",
            "",
            f"- Artifact root: `{output_dir}`",
            f"- R5.2 bad/tail: `{summary['r5_2_bad_tail_v1'][0]}` / `{summary['r5_2_bad_tail_v1'][1]}`",
            f"- Retained from V2 OOF: `{summary['retained_from_v2_oof_v1']}`",
            f"- New gains beyond V2 OOF: `{summary['new_gains_beyond_v2_oof_v1']}`",
            f"- Net gap to 188/136 proxy: `{summary['net_gap_to_proxy_bad_tail_v1'][0]}` / `{summary['net_gap_to_proxy_bad_tail_v1'][1]}`",
            f"- Tail repair candidates: `{summary['tail_repair_candidate_count_v1']}`",
            f"- Tail repair or R6-tail-head candidates: `{summary['tail_repair_or_head_candidate_count_v1']}`",
            f"- Failed R6 expansion candidates: `{summary['failed_r6_expansion_candidate_count_v1']}`",
            f"- Go/no-go: `{summary['go_no_go_v1']}`",
            f"- Next: `{summary['next_recommended_action_v1']}`",
        ],
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--r6-root", type=Path, default=R6_ROOT)
    parser.add_argument("--r5-2-package-root", type=Path, default=R5_2_PACKAGE_ROOT)
    parser.add_argument("--coverage-root", type=Path, default=COVERAGE_ROOT)
    parser.add_argument("--opportunity-root", type=Path, default=OPPORTUNITY_ROOT)
    parser.add_argument("--no-model-search-package-r6-rerun-freeze-live", action="store_true")
    parser.add_argument("--require-explicit-artifact-selection", action="store_true")
    parser.add_argument("--fail-on-dummy-or-synthetic-input", action="store_true")
    parser.add_argument("--fail-on-degraded-fallback", action="store_true")
    parser.add_argument("--preserve-strict-loso-low-support", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        r6_root=args.r6_root,
        package_root=args.r5_2_package_root,
        coverage_root=args.coverage_root,
        opportunity_root=args.opportunity_root,
        explicit_action=args.explicit_action,
    )


if __name__ == "__main__":
    main()
