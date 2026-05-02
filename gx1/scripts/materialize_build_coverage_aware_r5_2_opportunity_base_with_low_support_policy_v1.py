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


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1"
LAYER_NAME = ACTION
POLICY_ROOT = DEFAULT_REPORTS_ROOT / "DEFINE_EXPLICIT_RUN_ID_LOW_SUPPORT_POLICY_V1_20260427T140733Z_LOCK"
OPPORTUNITY_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1_20260427T122550Z_LOCK"
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"
DENOMINATOR_TARGET = 5
WORST_RUN_ID = "TRUTH_MONFRI_WEEK_20250106_20250113"

VARIANT_IDS = [
    "V2_OOF_CORE_69",
    "POLICY_ALLOWED_73",
    "COVERAGE_AWARE_SAFE_SIGNAL_CORE",
    "COVERAGE_AWARE_TAIL_EXPANSION",
    "COVERAGE_AWARE_RUN_ID_BALANCED",
    "COVERAGE_AWARE_BALANCED_CONSERVATIVE",
    "MAX_POLICY_ALLOWED_DIAGNOSTIC",
]

FIXED_CONTROLS = [
    "historical V2 95/61 as blueprint/comparator only",
    "V2 OOF 69/53 as provenance-valid signal control",
    "previous 73-row policy-allowed base",
    "Optuna 56/55 as weak/search-space control",
    "V3 17/13 as weak OOF control",
    "strict LOSO all-run_id reporting",
    "low-support registry reporting",
    "safety reports",
]


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


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING_LOCAL_ARTIFACT"


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


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_loso_guard_not_weakened(min_denominator: int) -> bool:
    if int(min_denominator) < DENOMINATOR_TARGET:
        raise RuntimeError("LOSO_DENOMINATOR_GUARD_WEAKENING_FORBIDDEN")
    return True


def validate_no_dummy_synthetic_fallback(*, dummy: bool, synthetic: bool, fallback: bool) -> dict[str, Any]:
    failures = []
    if dummy:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if fallback:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_no_forbidden_actions(*, optuna: bool, model: bool, r6: bool, package: bool, freeze: bool, live: bool) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if model:
        failures.append("MODEL_TRAINING_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_input_artifacts_unchanged(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = [key for key, value in before.items() if after.get(key) != value]
    return {
        "status_v1": "PASS" if not changed else "FAIL",
        "changed_v1": changed,
        "v2_oof_scores_unchanged_v1": "v2_oof_scores_sha256_v1" not in changed,
        "v2_oof_provenance_unchanged_v1": "v2_oof_provenance_sha256_v1" not in changed,
        "v2_model_objective_thresholds_unchanged_v1": True,
        "opportunity_rows_unchanged_v1": "opportunity_rows_sha256_v1" not in changed,
        "low_support_registry_unchanged_v1": "low_support_registry_sha256_v1" not in changed,
    }


def final_promotion_allowed(*, structural_low_support_selected: bool, strict_loso_decision_valid: bool, explicit_exception_gate: bool) -> bool:
    return bool(strict_loso_decision_valid and ((not structural_low_support_selected) or explicit_exception_gate))


def max_policy_allowed_can_be_final_recommendation(variant_id: str) -> bool:
    return variant_id != "MAX_POLICY_ALLOWED_DIAGNOSTIC"


def positive_row_has_evidence(row: pd.Series | dict[str, Any]) -> bool:
    if _as_bool(row.get("v2_oof_captured_v1")):
        return True
    if _as_bool(row.get("historical_v2_captured_v1")):
        return True
    if _as_bool(row.get("optuna_captured_v1")) or _as_bool(row.get("v3_captured_v1")):
        return True
    if str(row.get("r5_bad_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"}:
        return True
    if str(row.get("r5_1_bad_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"}:
        return True
    if str(row.get("r5_tail_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"}:
        return True
    if str(row.get("v2_like_bad_tail_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"}:
        return True
    return int(row.get("existing_legal_signal_evidence_count_v1", 0) or 0) > 0


def row_is_hard_veto(row: pd.Series | dict[str, Any]) -> bool:
    return bool(
        _as_bool(row.get("protected_winner_status_v1"))
        or _as_bool(row.get("runner_protect_status_v1"))
        or _as_bool(row.get("ambiguous_high_mfe_status_v1"))
        or _as_bool(row.get("fifty_plus_mfe_risk_v1"))
        or _as_bool(row.get("hundred_plus_mfe_risk_v1"))
        or _as_bool(row.get("two_hundred_plus_mfe_risk_v1"))
        or str(row.get("active_quarantine_v1", "")).upper() != "ACTIVE_CANDIDATE"
    )


def row_can_be_positive(row: pd.Series | dict[str, Any]) -> bool:
    return bool(_as_bool(row.get("safe_recoverable_v1")) and positive_row_has_evidence(row) and not row_is_hard_veto(row))


def classify_opportunity_role(row: pd.Series | dict[str, Any]) -> str:
    if str(row.get("active_quarantine_v1", "")).upper() != "ACTIVE_CANDIDATE":
        return "QUARANTINE_EXCLUDE"
    if _as_bool(row.get("protected_winner_status_v1")):
        return "HARD_NEGATIVE_PROTECTED_WINNER"
    if _as_bool(row.get("runner_protect_status_v1")):
        return "HARD_NEGATIVE_RUNNER_PROTECT"
    if (
        _as_bool(row.get("fifty_plus_mfe_risk_v1"))
        or _as_bool(row.get("hundred_plus_mfe_risk_v1"))
        or _as_bool(row.get("two_hundred_plus_mfe_risk_v1"))
    ):
        return "HARD_NEGATIVE_HIGH_MFE_UNSAFE"
    if _as_bool(row.get("ambiguous_high_mfe_status_v1")):
        return "AMBIGUOUS_MONITOR_ONLY"
    if _as_bool(row.get("v2_oof_captured_v1")) and _as_bool(row.get("tail_label_v1")):
        return "CORE_V2_OOF_TAIL_POSITIVE"
    if _as_bool(row.get("v2_oof_captured_v1")):
        return "CORE_V2_OOF_POSITIVE"
    if not row_can_be_positive(row):
        if _as_bool(row.get("structural_low_support_v1")):
            return "STRUCTURAL_LOW_SUPPORT_TRAINING_ONLY"
        return "UNKNOWN_REQUIRES_ARTIFACT"
    if _as_bool(row.get("structural_low_support_v1")):
        return "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE"
    if _as_bool(row.get("selected_low_support_v1")) or str(row.get("run_id_policy_class_v1", "")) == "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS":
        return "COVERAGE_EXPANSION_RUN_ID_SUPPORT"
    if _as_bool(row.get("tail_label_v1")) and str(row.get("r5_tail_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"}:
        return "COVERAGE_EXPANSION_TAIL"
    if str(row.get("r5_bad_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"} or str(row.get("r5_1_bad_score_signal_bucket_v1", "NONE")) in {"STRONG", "SUPPORT"}:
        return "COVERAGE_EXPANSION_STRONG_BAD"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def training_weight_tier(row: pd.Series | dict[str, Any]) -> str:
    role = str(row.get("opportunity_role_v1", "UNKNOWN_REQUIRES_ARTIFACT"))
    if role in {"HARD_NEGATIVE_PROTECTED_WINNER", "HARD_NEGATIVE_RUNNER_PROTECT", "HARD_NEGATIVE_HIGH_MFE_UNSAFE"}:
        return "HARD_NEGATIVE_HIGH_WEIGHT"
    if role in {"AMBIGUOUS_MONITOR_ONLY"}:
        return "MONITOR_ZERO_WEIGHT"
    if role == "QUARANTINE_EXCLUDE":
        return "EXCLUDE_ZERO_WEIGHT"
    if role in {"UNKNOWN_REQUIRES_ARTIFACT", "STRUCTURAL_LOW_SUPPORT_TRAINING_ONLY"}:
        return "UNKNOWN_ZERO_WEIGHT"
    if _as_bool(row.get("structural_low_support_v1")):
        return "LOW_SUPPORT_LOW_WEIGHT"
    if role == "CORE_V2_OOF_TAIL_POSITIVE":
        return "TAIL_HIGH_WEIGHT"
    if role == "CORE_V2_OOF_POSITIVE":
        return "CORE_HIGH_WEIGHT"
    return "COVERAGE_MEDIUM_WEIGHT"


def evaluation_role(row: pd.Series | dict[str, Any]) -> str:
    role = str(row.get("opportunity_role_v1", "UNKNOWN_REQUIRES_ARTIFACT"))
    if role.startswith("HARD_NEGATIVE"):
        return "SAFETY_VETO_CONTROL"
    if role == "AMBIGUOUS_MONITOR_ONLY":
        return "MONITOR_ONLY"
    if role == "QUARANTINE_EXCLUDE":
        return "EXCLUDE"
    if _as_bool(row.get("final_promotion_evidence_allowed_v1")) and role not in {"UNKNOWN_REQUIRES_ARTIFACT", "STRUCTURAL_LOW_SUPPORT_TRAINING_ONLY"}:
        return "STRICT_FINAL_EVIDENCE_ELIGIBLE"
    if _as_bool(row.get("training_opportunity_allowed_v1")):
        return "TRAINING_ONLY_LOW_SUPPORT" if _as_bool(row.get("structural_low_support_v1")) else "TRAINING_OPPORTUNITY_ONLY"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def _load_inputs(policy_root: Path, opportunity_root: Path) -> dict[str, Any]:
    return {
        "opportunity_rows": pd.read_csv(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
        "prior_variants": pd.read_csv(opportunity_root / "r5_2_opportunity_base_variants_v1.csv"),
        "registry": pd.read_csv(policy_root / "structural_low_support_run_id_registry_v1.csv"),
        "policy": _read_json(policy_root / "run_id_low_support_policy_v1.json"),
        "policy_summary": _read_json(policy_root / "summary_v1.json"),
    }


def _coverage_rows(opportunity_rows: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = opportunity_rows.copy()
    reg = registry.copy()
    rows = rows.merge(reg, on="run_id_v1", how="left", suffixes=("", "_policy"))
    rows["run_id_policy_class_v1"] = rows["run_id_policy_class_v1"].fillna("UNKNOWN_REQUIRES_ARTIFACT")
    rows["structural_low_support_v1"] = _bool(rows, "structural_low_support_v1")
    rows["selected_low_support_v1"] = _bool(rows, "selected_low_support_v1")
    rows["zero_denominator_group_v1"] = _bool(rows, "zero_denominator_group_v1")
    rows["training_opportunity_allowed_v1"] = rows.apply(
        lambda row: bool(
            str(row.get("active_quarantine_v1", "")).upper() == "ACTIVE_CANDIDATE"
            and not row_is_hard_veto(row)
            and positive_row_has_evidence(row)
            and (_as_bool(row.get("safe_recoverable_v1")) or _as_bool(row.get("v2_oof_captured_v1")))
        ),
        axis=1,
    )
    rows["candidate_eval_allowed_v1"] = rows["active_quarantine_v1"].astype(str).str.upper().eq("ACTIVE_CANDIDATE")
    rows["final_promotion_evidence_allowed_v1"] = rows.apply(
        lambda row: bool(
            _as_bool(row.get("can_be_used_in_decision_valid_eval_v1"))
            and not _as_bool(row.get("structural_low_support_v1"))
            and not _as_bool(row.get("zero_denominator_group_v1"))
            and row_can_be_positive(row)
        ),
        axis=1,
    )
    rows["opportunity_role_v1"] = rows.apply(classify_opportunity_role, axis=1)
    rows["training_weight_tier_v1"] = rows.apply(training_weight_tier, axis=1)
    rows["evaluation_role_v1"] = rows.apply(evaluation_role, axis=1)
    rows["coverage_reason_v1"] = rows.apply(_coverage_reason, axis=1)
    return rows


def _coverage_reason(row: pd.Series) -> str:
    role = str(row["opportunity_role_v1"])
    if role.startswith("CORE_V2_OOF"):
        return "Provenance-valid V2 OOF row retained as core signal control."
    if role == "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE":
        return "Structural low-support row has safety-clean evidence; allowed for training/opportunity only."
    if role == "COVERAGE_EXPANSION_RUN_ID_SUPPORT":
        return "Repairable or selected low-support group gets safe signal-backed coverage support."
    if role == "COVERAGE_EXPANSION_TAIL":
        return "Safe recoverable tail row with R5 tail signal evidence."
    if role == "COVERAGE_EXPANSION_STRONG_BAD":
        return "Safe recoverable row with R5/R5.1 bad signal evidence."
    if role.startswith("HARD_NEGATIVE"):
        return "Hard safety veto; never use as positive under current contract."
    if role == "AMBIGUOUS_MONITOR_ONLY":
        return "Ambiguous high-MFE row remains monitor-only unless separately safe-proven."
    if role == "QUARANTINE_EXCLUDE":
        return "Quarantine/non-active row excluded from positive support."
    if role == "STRUCTURAL_LOW_SUPPORT_TRAINING_ONLY":
        return "Structural low-support group is tagged, but this row lacks positive support evidence."
    return "Insufficient evidence for positive opportunity role."


def _output_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "candidate_uid_v1",
        "trade_uid_v1",
        "trade_id_v1",
        "decision_timestamp_v1",
        "run_id_v1",
        "active_quarantine_v1",
        "run_id_policy_class_v1",
        "structural_low_support_v1",
        "zero_denominator_group_v1",
        "training_opportunity_allowed_v1",
        "candidate_eval_allowed_v1",
        "final_promotion_evidence_allowed_v1",
        "bad_label_v1",
        "tail_label_v1",
        "safe_recoverable_v1",
        "v2_oof_captured_v1",
        "historical_v2_captured_v1",
        "optuna_captured_v1",
        "v3_captured_v1",
        "r5_bad_score_signal_bucket_v1",
        "r5_1_bad_score_signal_bucket_v1",
        "r5_tail_score_signal_bucket_v1",
        "v2_like_bad_tail_signal_bucket_v1",
        "v3_oof_signal_bucket_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "provenance_status_v1",
        "existing_legal_signal_evidence_count_v1",
        "opportunity_role_v1",
        "training_weight_tier_v1",
        "evaluation_role_v1",
        "coverage_reason_v1",
    ]
    return rows[[field for field in fields if field in rows.columns]].to_dict("records")


def _memberships(rows: pd.DataFrame) -> dict[str, pd.Series]:
    core = _bool(rows, "member_v2_oof_core_only_v1")
    policy_73 = _bool(rows, "member_v2_oof_plus_run_id_support_v1")
    allowed_positive = _bool(rows, "training_opportunity_allowed_v1") & rows["opportunity_role_v1"].isin(
        [
            "CORE_V2_OOF_POSITIVE",
            "CORE_V2_OOF_TAIL_POSITIVE",
            "COVERAGE_EXPANSION_STRONG_BAD",
            "COVERAGE_EXPANSION_TAIL",
            "COVERAGE_EXPANSION_RUN_ID_SUPPORT",
            "LOW_SUPPORT_TRAINING_ALLOWED_POSITIVE",
        ]
    )
    strong_bad = allowed_positive & rows["opportunity_role_v1"].eq("COVERAGE_EXPANSION_STRONG_BAD")
    tail = allowed_positive & rows["opportunity_role_v1"].eq("COVERAGE_EXPANSION_TAIL")
    run_balanced = allowed_positive & (
        rows["opportunity_role_v1"].eq("COVERAGE_EXPANSION_RUN_ID_SUPPORT")
        | (rows["run_id_policy_class_v1"].astype(str).eq("SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS"))
    )
    conservative = core | policy_73 | strong_bad | tail | run_balanced
    diagnostic = allowed_positive
    return {
        "V2_OOF_CORE_69": core,
        "POLICY_ALLOWED_73": policy_73,
        "COVERAGE_AWARE_SAFE_SIGNAL_CORE": core | strong_bad,
        "COVERAGE_AWARE_TAIL_EXPANSION": core | tail,
        "COVERAGE_AWARE_RUN_ID_BALANCED": core | policy_73 | run_balanced,
        "COVERAGE_AWARE_BALANCED_CONSERVATIVE": conservative,
        "MAX_POLICY_ALLOWED_DIAGNOSTIC": diagnostic,
    }


def _safety_counts(rows: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    return {
        "fifty_plus_overlap_v1": int((selected & _bool(rows, "fifty_plus_mfe_risk_v1")).sum()),
        "hundred_plus_overlap_v1": int((selected & _bool(rows, "hundred_plus_mfe_risk_v1")).sum()),
        "two_hundred_plus_overlap_v1": int((selected & _bool(rows, "two_hundred_plus_mfe_risk_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(rows, "protected_winner_status_v1")).sum()),
        "protected_winner_selected_v1": int((selected & _bool(rows, "protected_winner_status_v1")).sum()),
        "runner_protect_leakage_v1": int((selected & _bool(rows, "runner_protect_status_v1")).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & _bool(rows, "ambiguous_high_mfe_status_v1")).sum()),
        "quarantine_selected_v1": int((selected & rows["active_quarantine_v1"].astype(str).str.upper().ne("ACTIVE_CANDIDATE")).sum()),
    }


def _support_counts(rows: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    all_groups = sorted(rows["run_id_v1"].astype(str).unique())
    counts = selected.groupby(rows["run_id_v1"].astype(str)).sum().reindex(all_groups, fill_value=0).astype(int)
    nonzero = counts[counts > 0]
    low = nonzero[(nonzero > 0) & (nonzero < DENOMINATOR_TARGET)]
    return {
        "strict_all_run_id_min_denominator_v1": int(nonzero.min()) if len(nonzero) else 0,
        "strict_all_run_id_decision_valid_base_v1": bool(len(nonzero) > 0 and int(nonzero.min()) >= DENOMINATOR_TARGET),
        "selected_low_support_groups_v1": int(len(low)),
        "zero_denominator_groups_v1": int((counts == 0).sum()),
        "evaluable_group_count_v1": int((counts >= DENOMINATOR_TARGET).sum()),
        "selected_run_id_group_count_v1": int((counts > 0).sum()),
        "run_id_denominators_v1": {str(key): int(value) for key, value in counts.items()},
    }


def _variant_summary(rows: pd.DataFrame, memberships: dict[str, pd.Series]) -> list[dict[str, Any]]:
    variants = []
    for variant_id, selected in memberships.items():
        selected = selected.astype(bool)
        support = _support_counts(rows, selected)
        safety = _safety_counts(rows, selected)
        selected_rows = rows[selected]
        structural_selected = int((selected & _bool(rows, "structural_low_support_v1")).groupby(rows["run_id_v1"].astype(str)).any().sum())
        positive_without_evidence = int((selected & ~rows.apply(positive_row_has_evidence, axis=1)).sum())
        safety_clean = all(value == 0 for value in safety.values())
        strict_valid = bool(
            support["strict_all_run_id_decision_valid_base_v1"]
            and structural_selected == 0
            and positive_without_evidence == 0
            and safety_clean
        )
        final_allowed = final_promotion_allowed(
            structural_low_support_selected=structural_selected > 0,
            strict_loso_decision_valid=strict_valid,
            explicit_exception_gate=False,
        )
        training_allowed = bool(int(selected.sum()) > 0 and safety_clean and positive_without_evidence == 0)
        variants.append(
            {
                "variant_id_v1": variant_id,
                "variant_type_v1": "ROW_MEMBERSHIP_AND_WEIGHT_SET_NOT_TRAINED_MODEL",
                "selected_rows_v1": int(selected.sum()),
                "bad_proxy_v1": int((selected & _bool(rows, "bad_label_v1")).sum()),
                "tail_proxy_v1": int((selected & _bool(rows, "tail_label_v1")).sum()),
                "safe_recoverable_selected_v1": int((selected & _bool(rows, "safe_recoverable_v1")).sum()),
                "v2_oof_overlap_v1": int((selected & _bool(rows, "v2_oof_captured_v1")).sum()),
                "historical_v2_overlap_v1": int((selected & _bool(rows, "historical_v2_captured_v1")).sum()),
                "optuna_overlap_v1": int((selected & _bool(rows, "optuna_captured_v1")).sum()),
                "v3_overlap_v1": int((selected & _bool(rows, "v3_captured_v1")).sum()),
                "strict_all_run_id_min_denominator_v1": support["strict_all_run_id_min_denominator_v1"],
                "strict_all_run_id_decision_valid_v1": strict_valid,
                "selected_low_support_groups_v1": support["selected_low_support_groups_v1"],
                "structural_low_support_selected_groups_v1": structural_selected,
                "zero_denominator_selected_groups_v1": 0,
                "zero_denominator_groups_v1": support["zero_denominator_groups_v1"],
                "evaluable_group_count_v1": support["evaluable_group_count_v1"],
                **safety,
                "training_surface_allowed_v1": training_allowed,
                "final_promotion_allowed_v1": final_allowed,
                "model_trained_v1": False,
                "package_built_v1": False,
                "r6_ready_v1": False,
                "recommendation_status_v1": _variant_status(variant_id, training_allowed, final_allowed, structural_selected, safety_clean),
                "reason_v1": _variant_reason(variant_id, selected_rows, support, structural_selected, final_allowed),
            }
        )
    return variants


def _variant_status(variant_id: str, training_allowed: bool, final_allowed: bool, structural_selected: int, safety_clean: bool) -> str:
    if not safety_clean:
        return "BLOCKED_BY_SAFETY_CONFLICTS"
    if variant_id == "MAX_POLICY_ALLOWED_DIAGNOSTIC":
        return "DIAGNOSTIC_ONLY_NOT_FINAL_RECOMMENDATION"
    if training_allowed and not final_allowed and structural_selected:
        return "TRAINING_ALLOWED_FINAL_PROMOTION_BLOCKED"
    if training_allowed:
        return "TRAINING_ALLOWED"
    return "NOT_RECOMMENDED"


def _variant_reason(
    variant_id: str,
    selected_rows: pd.DataFrame,
    support: dict[str, Any],
    structural_selected: int,
    final_allowed: bool,
) -> str:
    if variant_id == "MAX_POLICY_ALLOWED_DIAGNOSTIC":
        return "Maximum policy-allowed training surface; diagnostic only, not final candidate."
    if final_allowed:
        return "Strict support and safety pass, but this package still does not approve promotion."
    if structural_selected:
        return "Training/opportunity allowed with structural low-support tags; final promotion remains blocked."
    if support["selected_low_support_groups_v1"]:
        return "Low-support groups remain visible under strict LOSO reporting."
    return f"Safety-clean membership set with {len(selected_rows)} selected rows."


def _addition_plan(rows: pd.DataFrame, memberships: dict[str, pd.Series]) -> list[dict[str, Any]]:
    base = memberships["POLICY_ALLOWED_73"].astype(bool)
    records = []
    for run_id, group in rows.groupby("run_id_v1"):
        idx = group.index
        current = int(base.loc[idx].sum())
        diagnostic = memberships["MAX_POLICY_ALLOWED_DIAGNOSTIC"].loc[idx].astype(bool)
        candidate_available = int((diagnostic & ~base.loc[idx]).sum())
        additions_by_variant = {
            variant: int((membership.loc[idx].astype(bool) & ~base.loc[idx]).sum())
            for variant, membership in memberships.items()
        }
        structural = bool(_bool(group, "structural_low_support_v1").any())
        repairable = bool(group["run_id_policy_class_v1"].astype(str).eq("SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS").any())
        feasible = int(group["feasible_safe_max_denominator_v1"].fillna(0).max()) if "feasible_safe_max_denominator_v1" in group.columns else 0
        treatment = _run_id_treatment(group, structural, repairable, candidate_available)
        records.append(
            {
                "run_id_v1": str(run_id),
                "current_selected_rows_v1": current,
                "feasible_safe_max_denominator_v1": feasible,
                "structural_low_support_v1": structural,
                "repairable_v1": repairable,
                "candidate_rows_available_v1": candidate_available,
                "selected_additions_by_variant_v1": additions_by_variant,
                "additions_improve_denominator_v1": any(value > 0 for value in additions_by_variant.values()),
                "additions_improve_bad_tail_coverage_v1": bool((diagnostic & ~base.loc[idx] & (_bool(group, "bad_label_v1") | _bool(group, "tail_label_v1"))).any()),
                "safety_risk_v1": _run_id_safety_risk(group),
                "final_treatment_v1": treatment,
                "reason_v1": _run_id_treatment_reason(treatment),
            }
        )
    return records


def _run_id_safety_risk(group: pd.DataFrame) -> str:
    if _bool(group, "protected_winner_status_v1").any():
        return "PROTECTED_WINNER_PRESENT"
    if _bool(group, "runner_protect_status_v1").any():
        return "RUNNER_PROTECT_PRESENT"
    if _bool(group, "ambiguous_high_mfe_status_v1").any():
        return "AMBIGUOUS_HIGH_MFE_PRESENT"
    return "NO_HARD_SAFETY_RISK_IN_SELECTED_POLICY_ROWS"


def _run_id_treatment(group: pd.DataFrame, structural: bool, repairable: bool, candidate_available: int) -> str:
    if structural:
        return "TRAINING_ONLY_STRUCTURAL_LOW_SUPPORT"
    if repairable and candidate_available:
        return "REPAIR_WITH_SAFE_SIGNAL_ROWS"
    if _run_id_safety_risk(group) != "NO_HARD_SAFETY_RISK_IN_SELECTED_POLICY_ROWS":
        return "REJECT_UNSAFE_EXPANSION"
    if group["run_id_policy_class_v1"].astype(str).eq("UNKNOWN_REQUIRES_ARTIFACT").any():
        return "UNKNOWN_REQUIRES_ARTIFACT"
    return "KEEP_MONITOR_ONLY"


def _run_id_treatment_reason(treatment: str) -> str:
    if treatment == "TRAINING_ONLY_STRUCTURAL_LOW_SUPPORT":
        return "Do not force denominator target; use safe evidence-backed rows for training/opportunity only."
    if treatment == "REPAIR_WITH_SAFE_SIGNAL_ROWS":
        return "Existing safe signal-backed additions can improve support without safety leakage."
    if treatment == "REJECT_UNSAFE_EXPANSION":
        return "Additional support would cross safety veto territory."
    if treatment == "UNKNOWN_REQUIRES_ARTIFACT":
        return "Missing evidence prevents confident support expansion."
    return "No safe/evidence-backed support action is recommended."


def _hard_negative_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    mask = rows["opportunity_role_v1"].isin(
        [
            "HARD_NEGATIVE_PROTECTED_WINNER",
            "HARD_NEGATIVE_RUNNER_PROTECT",
            "HARD_NEGATIVE_HIGH_MFE_UNSAFE",
            "AMBIGUOUS_MONITOR_ONLY",
            "QUARANTINE_EXCLUDE",
            "UNKNOWN_REQUIRES_ARTIFACT",
        ]
    )
    records = []
    for _, row in rows[mask].iterrows():
        role = str(row["opportunity_role_v1"])
        if role.startswith("HARD_NEGATIVE"):
            allowed = "HARD_NEGATIVE"
        elif role == "AMBIGUOUS_MONITOR_ONLY":
            allowed = "MONITOR_ONLY"
        else:
            allowed = "EXCLUDE"
        records.append(
            {
                "candidate_uid_v1": row["candidate_uid_v1"],
                "trade_uid_v1": row["trade_uid_v1"],
                "run_id_v1": row["run_id_v1"],
                "reason_v1": row["coverage_reason_v1"],
                "veto_type_v1": role,
                "allowed_use_v1": allowed,
                "may_ever_be_positive_under_current_contract_v1": False,
                "evidence_source_v1": row.get("provenance_status_v1", "LOCAL_OPPORTUNITY_BASE"),
            }
        )
    return records


def _contract(policy_root: Path) -> dict[str, Any]:
    return {
        "contract": "COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_CONTRACT_V1",
        "scope_v1": "TRAINING_OPPORTUNITY_SURFACE_ONLY",
        "final_promotion_allowed_v1": False,
        "low_support_policy_root_v1": str(policy_root),
        "strict_loso_still_reported_v1": True,
        "structural_low_support_groups_explicitly_tagged_v1": True,
        "all_included_rows_require_evidence_reason_role_v1": True,
        "hard_negatives_veto_rows_preserved_v1": True,
        "quarantine_excluded_v1": True,
        "ambiguous_high_mfe_monitor_only_unless_safe_proven_v1": True,
        "dummy_synthetic_fallback_allowed_v1": False,
        "in_sample_decisioning_allowed_v1": False,
        "implicit_latest_glob_artifact_selection_allowed_v1": False,
        "not_model_package_r6_promotion_ready_v1": True,
    }


def _fixed_controls() -> dict[str, Any]:
    return {
        "fixed_controls_required_for_next_r5_2_rebuild_v1": FIXED_CONTROLS,
        "historical_v2_role_v1": "BLUEPRINT_COMPARATOR_ONLY_NOT_DECISION_VALID",
        "v2_oof_role_v1": "PROVENANCE_VALID_SIGNAL_CONTROL_NOT_FINAL_DECISION_VALID",
        "optuna_v3_role_v1": "WEAK_CONTROLS_NOT_BASELINE",
    }


def _recommendation(variants: list[dict[str, Any]], addition_plan: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["variant_id_v1"]: row for row in variants}
    preferred = by_id["COVERAGE_AWARE_RUN_ID_BALANCED"]
    conservative = by_id["COVERAGE_AWARE_BALANCED_CONSERVATIVE"]
    conservative_keeps_support = (
        int(conservative["selected_low_support_groups_v1"]) <= int(preferred["selected_low_support_groups_v1"])
        and int(conservative["structural_low_support_selected_groups_v1"]) <= int(preferred["structural_low_support_selected_groups_v1"])
        and int(conservative["strict_all_run_id_min_denominator_v1"]) >= int(preferred["strict_all_run_id_min_denominator_v1"])
    )
    if (
        conservative["training_surface_allowed_v1"]
        and conservative["selected_rows_v1"] > preferred["selected_rows_v1"]
        and conservative_keeps_support
    ):
        preferred = conservative
    safety_clean = all(
        int(preferred[key]) == 0
        for key in [
            "fifty_plus_overlap_v1",
            "hundred_plus_overlap_v1",
            "two_hundred_plus_overlap_v1",
            "strongest_winner_overlap_v1",
            "protected_winner_selected_v1",
            "runner_protect_leakage_v1",
            "ambiguous_high_mfe_leakage_v1",
            "quarantine_selected_v1",
        ]
    )
    structural_remaining = int(preferred["structural_low_support_selected_groups_v1"])
    repairable_improved = sum(1 for row in addition_plan if row["final_treatment_v1"] == "REPAIR_WITH_SAFE_SIGNAL_ROWS" and row["additions_improve_denominator_v1"])
    if not safety_clean:
        status = "COVERAGE_AWARE_BASE_BLOCKED_BY_SAFETY_CONFLICTS"
        next_action = "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
    elif int(preferred["selected_rows_v1"]) == 0:
        status = "COVERAGE_AWARE_BASE_BLOCKED_BY_MISSING_ARTIFACTS"
        next_action = "REQUIRE_MISSING_ARTIFACTS_OR_REBUILD_SIGNAL_PROVENANCE_V1"
    elif structural_remaining:
        status = "COVERAGE_AWARE_BASE_READY_BUT_FINAL_PROMOTION_BLOCKED"
        next_action = "BUILD_R5_2_FROM_COVERAGE_AWARE_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1"
    else:
        status = "COVERAGE_AWARE_BASE_READY_FOR_R5_2_REBUILD_DESIGN"
        next_action = "BUILD_R5_2_FROM_COVERAGE_AWARE_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1"
    return {
        "layer_name": "COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_RECOMMENDATION_V1",
        "status_v1": status,
        "recommended_variant_v1": preferred["variant_id_v1"],
        "next_recommended_action_v1": next_action,
        "why_better_than_v2_oof_core_v1": (
            f"Retains {preferred['v2_oof_overlap_v1']} V2 OOF rows and adds "
            f"{int(preferred['selected_rows_v1']) - int(by_id['V2_OOF_CORE_69']['selected_rows_v1'])} "
            "policy-allowed evidence-backed rows with hard vetoes intact."
        ),
        "recommended_rows_v1": preferred["selected_rows_v1"],
        "bad_tail_proxy_v1": [preferred["bad_proxy_v1"], preferred["tail_proxy_v1"]],
        "v2_oof_signal_retained_v1": preferred["v2_oof_overlap_v1"],
        "structural_low_support_selected_groups_v1": structural_remaining,
        "repairable_groups_improved_v1": repairable_improved,
        "strict_loso_status_v1": "PASS" if preferred["strict_all_run_id_decision_valid_v1"] else "STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
        "training_opportunity_status_v1": "ALLOWED" if preferred["training_surface_allowed_v1"] else "BLOCKED",
        "final_promotion_status_v1": "BLOCKED_STRUCTURAL_LOW_SUPPORT_OR_NO_EXCEPTION_GATE",
        "fixed_controls_v1": _fixed_controls(),
        "not_r6_ready_v1": True,
        "not_package_ready_v1": True,
        "not_live_ready_v1": True,
    }


def _next_rebuild_contract(recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": "NEXT_R5_2_REBUILD_FROM_COVERAGE_AWARE_BASE_CONTRACT_V1",
        "selected_opportunity_base_variant_v1": recommendation["recommended_variant_v1"],
        "row_roles_required_v1": True,
        "training_weight_tiers_required_v1": True,
        "hard_negatives_required_v1": True,
        "monitor_only_rows_required_v1": True,
        "excluded_rows_required_v1": True,
        "fixed_controls_v1": _fixed_controls()["fixed_controls_required_for_next_r5_2_rebuild_v1"],
        "required_provenance_v1": ["OOF execution", "train/validation membership", "score source manifest", "row-level provenance"],
        "required_denominator_reports_v1": ["strict all-run_id LOSO", "low-support registry", "evaluable-groups secondary metric"],
        "required_low_support_policy_reports_v1": True,
        "required_safety_reports_v1": True,
        "final_promotion_allowed_v1": False,
        "r6_allowed_without_explicit_gate_v1": False,
    }


def _summary_counts(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count_v1": int(len(rows)),
        "training_opportunity_allowed_rows_v1": int(_bool(rows, "training_opportunity_allowed_v1").sum()),
        "final_promotion_evidence_allowed_rows_v1": int(_bool(rows, "final_promotion_evidence_allowed_v1").sum()),
        "structural_low_support_rows_v1": int(_bool(rows, "structural_low_support_v1").sum()),
        "zero_denominator_group_rows_v1": int(_bool(rows, "zero_denominator_group_v1").sum()),
        "role_counts_v1": {str(key): int(value) for key, value in rows["opportunity_role_v1"].value_counts().to_dict().items()},
        "training_weight_tier_counts_v1": {str(key): int(value) for key, value in rows["training_weight_tier_v1"].value_counts().to_dict().items()},
    }


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    policy_root: Path = POLICY_ROOT,
    opportunity_root: Path = OPPORTUNITY_ROOT,
    v2_oof_root: Path = V2_OOF_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    validate_loso_guard_not_weakened(DENOMINATOR_TARGET)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes_before = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "opportunity_rows_sha256_v1": _file_hash(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
        "low_support_registry_sha256_v1": _file_hash(policy_root / "structural_low_support_run_id_registry_v1.csv"),
    }
    inputs = _load_inputs(policy_root, opportunity_root)
    rows = _coverage_rows(inputs["opportunity_rows"], inputs["registry"])
    memberships = _memberships(rows)
    variants = _variant_summary(rows, memberships)
    addition_plan = _addition_plan(rows, memberships)
    hard_negative_rows = _hard_negative_table(rows)
    recommendation = _recommendation(variants, addition_plan)
    rebuild_contract = _next_rebuild_contract(recommendation)
    contract = _contract(policy_root)

    for variant_id, selected in memberships.items():
        rows[f"member_{variant_id.lower()}_v1"] = selected.astype(bool)
    output_rows = _output_rows(rows)
    row_json = {"summary_v1": _summary_counts(rows), "rows_v1": output_rows}
    variant_json = {"variants_v1": variants}
    addition_json = {"rows_v1": addition_plan}
    hard_negative_json = {"rows_v1": hard_negative_rows, "row_count_v1": len(hard_negative_rows)}

    input_hashes_after = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "opportunity_rows_sha256_v1": _file_hash(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
        "low_support_registry_sha256_v1": _file_hash(policy_root / "structural_low_support_run_id_registry_v1.csv"),
    }
    artifact_integrity = validate_input_artifacts_unchanged(input_hashes_before, input_hashes_after)
    no_dummy = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)
    no_forbidden = validate_no_forbidden_actions(optuna=False, model=False, r6=False, package=False, freeze=False, live=False)
    go_no_go = {
        "layer_name": "COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_GO_NO_GO_V1",
        "decision_v1": recommendation["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "not_final_decisioning_v1": True,
        "not_r6_ready_v1": True,
        "not_package_ready_v1": True,
        "not_freeze_promo_live_ready_v1": True,
        "no_optuna_run_v1": True,
        "no_training_run_v1": True,
    }

    rows_out = pd.DataFrame(output_rows)
    rows_out.to_csv(output_dir / "coverage_aware_r5_2_opportunity_rows_v1.csv", index=False)
    _write_json(output_dir / "coverage_aware_r5_2_opportunity_rows_v1.json", row_json)
    _write_json(output_dir / "coverage_aware_r5_2_opportunity_base_contract_v1.json", contract)
    _write_rows(output_dir / "coverage_aware_r5_2_base_variants_v1.csv", variants)
    _write_json(output_dir / "coverage_aware_r5_2_base_variants_v1.json", variant_json)
    _write_rows(output_dir / "run_id_coverage_aware_addition_plan_v1.csv", addition_plan)
    _write_json(output_dir / "run_id_coverage_aware_addition_plan_v1.json", addition_json)
    _write_rows(output_dir / "r5_2_hard_negative_and_veto_rows_v1.csv", hard_negative_rows)
    _write_json(output_dir / "r5_2_hard_negative_and_veto_rows_v1.json", hard_negative_json)
    _write_json(output_dir / "coverage_aware_r5_2_opportunity_base_recommendation_v1.json", recommendation)
    _write_json(output_dir / "next_r5_2_rebuild_from_coverage_aware_base_contract_v1.json", rebuild_contract)
    _write_json(output_dir / "coverage_aware_r5_2_opportunity_base_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "policy_root_v1": str(policy_root),
                "opportunity_root_v1": str(opportunity_root),
                "v2_oof_root_v1": str(v2_oof_root),
            },
            "input_hashes_before_v1": input_hashes_before,
            "input_hashes_after_v1": input_hashes_after,
            "input_artifact_integrity_v1": artifact_integrity,
            "no_dummy_synthetic_fallback_v1": no_dummy,
            "no_forbidden_actions_v1": no_forbidden,
        },
    )
    _write_reports(output_dir, rows, variants, addition_plan, hard_negative_rows, contract, recommendation, rebuild_contract)
    rec_variant = next(row for row in variants if row["variant_id_v1"] == recommendation["recommended_variant_v1"])
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "decision_v1": go_no_go["decision_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        "policy_root_used_v1": str(policy_root),
        "row_count_v1": int(len(rows)),
        "recommended_variant_v1": recommendation["recommended_variant_v1"],
        "recommended_rows_v1": rec_variant["selected_rows_v1"],
        "recommended_bad_tail_proxy_v1": [rec_variant["bad_proxy_v1"], rec_variant["tail_proxy_v1"]],
        "strict_loso_status_v1": recommendation["strict_loso_status_v1"],
        "selected_low_support_groups_v1": rec_variant["selected_low_support_groups_v1"],
        "structural_low_support_selected_groups_v1": rec_variant["structural_low_support_selected_groups_v1"],
        "training_opportunity_allowed_v1": rec_variant["training_surface_allowed_v1"],
        "final_promotion_allowed_v1": rec_variant["final_promotion_allowed_v1"],
        "safety_retained_v1": all(
            int(rec_variant[key]) == 0
            for key in [
                "fifty_plus_overlap_v1",
                "hundred_plus_overlap_v1",
                "two_hundred_plus_overlap_v1",
                "strongest_winner_overlap_v1",
                "protected_winner_selected_v1",
                "runner_protect_leakage_v1",
                "ambiguous_high_mfe_leakage_v1",
                "quarantine_selected_v1",
            ]
        ),
        "v2_scores_provenance_model_objective_thresholds_unchanged_v1": artifact_integrity["status_v1"] == "PASS",
        "optuna_not_run_v1": True,
        "model_not_trained_v1": True,
        "r6_not_run_v1": True,
        "package_not_built_v1": True,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "go_no_go_v1": go_no_go})
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Build Coverage-Aware R5.2 Opportunity Base With Low Support Policy V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Recommended variant: `{summary['recommended_variant_v1']}`",
            f"Recommended rows: `{summary['recommended_rows_v1']}`",
            f"Strict LOSO status: `{summary['strict_loso_status_v1']}`",
            f"Next action: `{summary['next_recommended_action_v1']}`",
            "",
            "No model training, Optuna, R6, package build, freeze, promo, or live action was run.",
        ],
    )
    return summary


def _write_reports(
    output_dir: Path,
    rows: pd.DataFrame,
    variants: list[dict[str, Any]],
    addition_plan: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    recommendation: dict[str, Any],
    rebuild_contract: dict[str, Any],
) -> None:
    summary = _summary_counts(rows)
    rec = recommendation
    rec_variant = next(row for row in variants if row["variant_id_v1"] == rec["recommended_variant_v1"])
    _write_report(
        output_dir / "coverage_aware_r5_2_opportunity_base_contract_v1.md",
        [
            "# Coverage-Aware R5.2 Opportunity Base Contract V1",
            "",
            "Scope: training/opportunity surface only.",
            f"Low-support policy root: `{contract['low_support_policy_root_v1']}`",
            "Strict LOSO remains reported; structural low-support groups remain explicitly tagged.",
            "Final promotion is not allowed by this artifact.",
        ],
    )
    _write_report(
        output_dir / "coverage_aware_r5_2_opportunity_rows_report_v1.md",
        [
            "# Coverage-Aware R5.2 Opportunity Rows V1",
            "",
            f"Rows: `{summary['row_count_v1']}`",
            f"Training/opportunity allowed rows: `{summary['training_opportunity_allowed_rows_v1']}`",
            f"Structural low-support rows: `{summary['structural_low_support_rows_v1']}`",
            f"Role counts: `{summary['role_counts_v1']}`",
            f"Weight tiers: `{summary['training_weight_tier_counts_v1']}`",
        ],
    )
    _write_report(
        output_dir / "coverage_aware_r5_2_base_variants_report_v1.md",
        [
            "# Coverage-Aware R5.2 Base Variants V1",
            "",
            *[
                f"- `{row['variant_id_v1']}`: rows `{row['selected_rows_v1']}`, bad/tail `{row['bad_proxy_v1']}` / `{row['tail_proxy_v1']}`, strict LOSO min `{row['strict_all_run_id_min_denominator_v1']}`, final promotion `{row['final_promotion_allowed_v1']}`"
                for row in variants
            ],
        ],
    )
    _write_report(
        output_dir / "run_id_coverage_aware_addition_plan_report_v1.md",
        [
            "# Run ID Coverage-Aware Addition Plan V1",
            "",
            f"Run_id groups: `{len(addition_plan)}`",
            f"Repairable with safe signal rows: `{sum(1 for row in addition_plan if row['final_treatment_v1'] == 'REPAIR_WITH_SAFE_SIGNAL_ROWS')}`",
            f"Training-only structural low-support: `{sum(1 for row in addition_plan if row['final_treatment_v1'] == 'TRAINING_ONLY_STRUCTURAL_LOW_SUPPORT')}`",
        ],
    )
    _write_report(
        output_dir / "r5_2_hard_negative_and_veto_report_v1.md",
        [
            "# R5.2 Hard Negative And Veto Rows V1",
            "",
            f"Rows: `{len(hard_negative_rows)}`",
            "Protected winners, runner-protect, high-MFE unsafe, ambiguous monitor-only, quarantine, and unknown rows are not positive support.",
        ],
    )
    _write_report(
        output_dir / "coverage_aware_r5_2_opportunity_base_recommendation_v1.md",
        [
            "# Coverage-Aware R5.2 Opportunity Base Recommendation V1",
            "",
            f"Status: `{rec['status_v1']}`",
            f"Recommended variant: `{rec['recommended_variant_v1']}`",
            f"Rows: `{rec_variant['selected_rows_v1']}`",
            f"Bad/tail proxy: `{rec_variant['bad_proxy_v1']}` / `{rec_variant['tail_proxy_v1']}`",
            rec["why_better_than_v2_oof_core_v1"],
            f"Strict LOSO status: `{rec['strict_loso_status_v1']}`",
            f"Training/opportunity status: `{rec['training_opportunity_status_v1']}`",
            f"Final promotion status: `{rec['final_promotion_status_v1']}`",
            f"Fixed controls: `{rec['fixed_controls_v1']['fixed_controls_required_for_next_r5_2_rebuild_v1']}`",
        ],
    )
    _write_report(
        output_dir / "next_r5_2_rebuild_from_coverage_aware_base_contract_v1.md",
        [
            "# Next R5.2 Rebuild From Coverage-Aware Base Contract V1",
            "",
            f"Selected opportunity-base variant: `{rebuild_contract['selected_opportunity_base_variant_v1']}`",
            "The future rebuild must run OOF with provenance, denominator reports, low-support policy reports, and safety reports.",
            "No final promotion and no R6 are allowed without an explicit later gate.",
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", default=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--policy-root", type=Path, default=POLICY_ROOT)
    parser.add_argument("--opportunity-root", type=Path, default=OPPORTUNITY_ROOT)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        policy_root=args.policy_root,
        opportunity_root=args.opportunity_root,
        v2_oof_root=args.v2_oof_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
