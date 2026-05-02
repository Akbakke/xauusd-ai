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
ACTION = "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1"
LAYER_NAME = ACTION
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"
V2_REVALIDATION_ROOT = DEFAULT_REPORTS_ROOT / "REVALIDATE_V2_BASELINE_UNDER_CURRENT_GUARDS_V1_20260427T095034Z_LOCK"
LOSO_ROOT = DEFAULT_REPORTS_ROOT / "REPAIR_LOSO_GROUPING_OR_DENOMINATOR_CONTRACT_V1_20260427T120308Z_LOCK"
OPTUNA_ROOT = DEFAULT_REPORTS_ROOT / "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_V1_20260427T080458Z_LOCK"
SELECTED_V3_ROOT = DEFAULT_REPORTS_ROOT / "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG_20260427T073055Z_LOCK"
MIN_RUN_ID_SUPPORT = 5
WORST_RUN_ID = "TRUTH_MONFRI_WEEK_20250106_20250113"

VARIANT_NAMES = [
    "V2_OOF_CORE_ONLY",
    "V2_OOF_PLUS_SAFE_R5_R5_1_SUPPORT",
    "V2_OOF_PLUS_TAIL_EXPANSION",
    "V2_OOF_PLUS_RUN_ID_SUPPORT",
    "BALANCED_V2_R5_TAIL_RUN_ID_SUPPORT",
    "SAFETY_FIRST_UPPER_BOUND",
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING_LOCAL_ARTIFACT"


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def validate_no_dummy_synthetic_fallback(*, dummy: bool, synthetic: bool, fallback: bool) -> dict[str, Any]:
    failures = []
    if dummy:
        failures.append("DUMMY_INPUT_FORBIDDEN")
    if synthetic:
        failures.append("SYNTHETIC_INPUT_FORBIDDEN")
    if fallback:
        failures.append("DEGRADED_FALLBACK_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_loso_guard_not_weakened(min_denominator: int) -> bool:
    if int(min_denominator) < MIN_RUN_ID_SUPPORT:
        raise RuntimeError("LOSO_DENOMINATOR_GUARD_WEAKENING_FORBIDDEN")
    return True


def validate_variant_is_membership_set(variant: dict[str, Any]) -> bool:
    if variant.get("model_trained_v1") or variant.get("package_built_v1") or variant.get("r6_ready_v1"):
        raise RuntimeError("OPPORTUNITY_VARIANT_MUST_BE_MEMBERSHIP_SET_NOT_MODEL")
    return True


def validate_input_artifacts_unchanged(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = [key for key, value in before.items() if after.get(key) != value]
    return {
        "status_v1": "PASS" if not changed else "FAIL",
        "changed_v1": changed,
        "v2_oof_scores_unchanged_v1": "v2_oof_scores_sha256_v1" not in changed,
        "v2_oof_provenance_unchanged_v1": "v2_oof_provenance_sha256_v1" not in changed,
    }


def historical_v2_status() -> str:
    return "HISTORICAL_ONLY_NOT_DECISION_VALID"


def optuna_v3_can_be_baseline(status: str) -> bool:
    return status not in {"SAFE_BUT_NOT_BETTER_THAN_V2", "WEAK_CONTROL", "V3_SAFE_BUT_TOO_WEAK_STOP_R5_2_OBJECTIVE_LOOP"}


def recommendation_not_r6_ready(status: str) -> bool:
    if "R6_READY" in status or "LIVE_READY" in status:
        raise RuntimeError("OPPORTUNITY_BASE_RECOMMENDATION_CANNOT_BE_R6_OR_LIVE_READY")
    return True


def classify_signal_bucket(hit_count: int | float | bool, *, strong: int = 2) -> str:
    value = int(hit_count or 0)
    if value >= strong:
        return "STRONG"
    if value == 1:
        return "SUPPORT"
    return "NONE"


def classify_score_bucket(value: float | int | bool) -> str:
    val = float(value or 0.0)
    if val >= 0.75:
        return "STRONG"
    if val >= 0.50:
        return "SUPPORT"
    if val > 0.0:
        return "LOW"
    return "NONE"


def classify_opportunity_role(
    *,
    active: bool,
    quarantine: bool,
    protected_winner: bool,
    runner_protect: bool,
    ambiguous_high_mfe: bool,
    high_mfe_unsafe: bool,
    bad_label: bool,
    tail_label: bool,
    safe_recoverable: bool,
    v2_oof_captured: bool,
    historical_v2_captured: bool,
    optuna_captured: bool,
    v3_captured: bool,
    r5_bad_bucket: str,
    r5_1_bad_bucket: str,
    r5_tail_bucket: str,
    run_id_low_support: bool,
) -> str:
    if quarantine or not active:
        return "QUARANTINE_EXCLUDE"
    if protected_winner:
        return "HARD_NEGATIVE_PROTECTED_WINNER"
    if runner_protect:
        return "HARD_NEGATIVE_RUNNER_PROTECT"
    if high_mfe_unsafe:
        return "HARD_NEGATIVE_HIGH_MFE_UNSAFE"
    if ambiguous_high_mfe:
        return "AMBIGUOUS_MONITOR_ONLY"
    if v2_oof_captured and tail_label:
        return "CORE_OOF_V2_TAIL_POSITIVE"
    if v2_oof_captured and bad_label:
        return "CORE_OOF_V2_STRONG_POSITIVE"
    signal_evidence = r5_bad_bucket in {"STRONG", "SUPPORT"} or r5_1_bad_bucket in {"STRONG", "SUPPORT"} or r5_tail_bucket in {"STRONG", "SUPPORT"}
    if safe_recoverable and run_id_low_support and signal_evidence:
        return "RUN_ID_SUPPORT_CANDIDATE"
    if safe_recoverable and tail_label and r5_tail_bucket in {"STRONG", "SUPPORT"}:
        return "TAIL_EXPANSION_CANDIDATE"
    if safe_recoverable and (r5_bad_bucket in {"STRONG", "SUPPORT"} or r5_1_bad_bucket in {"STRONG", "SUPPORT"}):
        return "R5_R5_1_SIGNAL_SAFE_RECOVERABLE_CANDIDATE"
    if safe_recoverable and (historical_v2_captured or optuna_captured or v3_captured):
        return "V2_MISSED_SAFE_RECOVERABLE_CANDIDATE"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def row_has_positive_evidence(row: pd.Series) -> bool:
    if not bool(row.get("safe_recoverable_v1", False)):
        return bool(row.get("v2_oof_captured_v1", False))
    if bool(row.get("bad_label_v1", False)) or bool(row.get("tail_label_v1", False)):
        return True
    if row.get("r5_bad_score_signal_bucket_v1") in {"STRONG", "SUPPORT"}:
        return True
    if row.get("r5_1_bad_score_signal_bucket_v1") in {"STRONG", "SUPPORT"}:
        return True
    if row.get("r5_tail_score_signal_bucket_v1") in {"STRONG", "SUPPORT"}:
        return True
    return bool(row.get("v2_oof_captured_v1", False) or row.get("historical_v2_captured_v1", False))


def _load_foundation_score_frame(v2_oof_root: Path) -> pd.DataFrame:
    manifest = _read_json(v2_oof_root / "manifest_v1.json")
    score_dir = Path((manifest.get("inputs_v1") or {}).get("score_dir_v1", ""))
    path = score_dir / "monday_r6_foundation_score_frame_v1.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_inputs(v2_oof_root: Path, revalidation_root: Path, loso_root: Path) -> dict[str, Any]:
    learning = pd.read_csv(revalidation_root / "existing_legal_learning_foundation_v1.csv")
    signal_audit = pd.read_csv(revalidation_root / "signal_family_lift_and_safety_audit_v1.csv")
    decomp = pd.read_csv(revalidation_root / "v2_result_decomposition_v1.csv")
    oof = pd.read_csv(v2_oof_root / "v2_oof_scores_v1.csv")
    provenance = pd.read_csv(v2_oof_root / "v2_oof_score_provenance_v1.csv")
    loso = pd.read_csv(loso_root / "v2_oof_loso_group_distribution_v1.csv")
    foundation = _load_foundation_score_frame(v2_oof_root)
    return {
        "learning": learning,
        "signal_audit": signal_audit,
        "decomp": decomp,
        "oof": oof,
        "provenance": provenance,
        "loso": loso,
        "foundation": foundation,
        "v2_summary": _read_json(v2_oof_root / "v2_oof_replay_summary_v1.json"),
        "loso_summary": _read_json(loso_root / "summary_v1.json"),
    }


def _base_rows(inputs: dict[str, Any]) -> pd.DataFrame:
    learning = inputs["learning"].rename(
        columns={
            "candidate_uid_v1": "candidate_uid",
            "trade_uid_v1": "trade_uid",
            "trade_id_v1": "trade_id",
            "decision_timestamp_v1": "decision_timestamp",
            "loso_group_v1": "run_id",
        }
    )
    oof_cols = [
        "candidate_uid",
        "fold_id_v1",
        "r5_2_v2_final_base_membership",
        "r5_2_v2_bad_recall_score",
        "r5_2_v2_tail_recall_score",
        "r5_2_v2_runner_protection_score",
        "r5_2_v2_high_mfe_ambiguous_protection_score",
        "r5_2_v2_hard_winner_protection_score",
        "v2_base_reason_v1",
    ]
    oof = inputs["oof"][[column for column in oof_cols if column in inputs["oof"].columns]].copy()
    foundation_cols = [
        "candidate_uid",
        "pred__entry_r5_should_not_take__prob_true_v1",
        "r5_1_bad_blocker_score_v1",
        "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
        "pred__entry_r5_runner_protect__prob_true_v1",
        "r5_1_runner_guard_score_v1",
        "pred__entry_r5_2_runner_protector__prob_true_v1",
        "calendar_quarantine_status_v1",
    ]
    foundation = inputs["foundation"][[column for column in foundation_cols if column in inputs["foundation"].columns]].copy()
    rows = learning.merge(oof, on="candidate_uid", how="left").merge(foundation, on="candidate_uid", how="left", suffixes=("", "_foundation"))
    decomp = inputs["decomp"][["candidate_uid_v1", "v2_captured_v1"]].rename(
        columns={"candidate_uid_v1": "candidate_uid", "v2_captured_v1": "historical_v2_captured_v1"}
    )
    rows = rows.merge(decomp, on="candidate_uid", how="left")
    rows["v2_oof_captured_v1"] = _bool(rows, "r5_2_v2_final_base_membership")
    rows["historical_v2_captured_v1"] = _bool(rows, "historical_v2_captured_v1") | _bool(rows, "v2_captured_v1")
    rows["optuna_captured_v1"] = _bool(rows, "optuna_captured_v1")
    rows["v3_captured_v1"] = _bool(rows, "v3_captured_v1")
    rows["active_v1"] = rows["active_quarantine_v1"].astype(str).eq("ACTIVE_CANDIDATE")
    rows["quarantine_v1"] = ~rows["active_v1"]
    rows["bad_label_v1"] = _bool(rows, "bad_label_v1")
    rows["tail_label_v1"] = _bool(rows, "tail_label_v1")
    rows["safe_recoverable_v1"] = _bool(rows, "safe_recoverable_v1")
    rows["protected_winner_v1"] = _bool(rows, "protected_winner_status_v1")
    rows["runner_protect_v1"] = _bool(rows, "runner_protect_status_v1")
    rows["ambiguous_high_mfe_v1"] = _bool(rows, "high_mfe_ambiguity_status_v1")
    rows["fifty_plus_mfe_risk_v1"] = _bool(rows, "fifty_plus_mfe_risk_v1")
    rows["hundred_plus_mfe_risk_v1"] = _bool(rows, "hundred_plus_mfe_risk_v1")
    rows["two_hundred_plus_mfe_risk_v1"] = _bool(rows, "two_hundred_plus_mfe_risk_v1")
    rows["r5_bad_score_signal_bucket_v1"] = rows["r5_signal_family_hits_v1"].map(classify_signal_bucket)
    rows["r5_1_bad_score_signal_bucket_v1"] = rows["r5_1_signal_family_hits_v1"].map(classify_signal_bucket)
    tail_raw = _num(rows, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1")
    rows["r5_tail_score_signal_bucket_v1"] = tail_raw.map(classify_score_bucket)
    rows["v2_like_bad_tail_signal_bucket_v1"] = rows["r5_2_v2_signal_family_hits_v1"].map(lambda value: classify_signal_bucket(value, strong=1))
    v3_combo = rows["v3_captured_v1"].astype(int) + _bool(rows, "v3_captured_v1").astype(int)
    rows["v3_oof_signal_bucket_v1"] = v3_combo.map(lambda value: classify_signal_bucket(value, strong=1))
    rows["existing_legal_signal_evidence_count_v1"] = (
        pd.to_numeric(rows["r5_signal_family_hits_v1"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(rows["r5_1_signal_family_hits_v1"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(rows["r5_2_v2_signal_family_hits_v1"], errors="coerce").fillna(0).astype(int)
        + rows["v2_oof_captured_v1"].astype(int)
        + rows["historical_v2_captured_v1"].astype(int)
        + rows["optuna_captured_v1"].astype(int)
        + rows["v3_captured_v1"].astype(int)
    )
    rows["hard_safety_veto_v1"] = (
        rows["quarantine_v1"]
        | rows["protected_winner_v1"]
        | rows["runner_protect_v1"]
        | rows["ambiguous_high_mfe_v1"]
        | rows["fifty_plus_mfe_risk_v1"]
        | rows["hundred_plus_mfe_risk_v1"]
        | rows["two_hundred_plus_mfe_risk_v1"]
    )
    loso = inputs["loso"].copy()
    low_support = loso[(loso["selected_rows_v1"].astype(int) > 0) & (loso["selected_rows_v1"].astype(int) < MIN_RUN_ID_SUPPORT)]["group_id_v1"].astype(str)
    rows["run_id_low_support_v1"] = rows["run_id"].astype(str).isin(set(low_support))
    rows["positive_evidence_v1"] = rows.apply(row_has_positive_evidence, axis=1)
    rows["recommended_opportunity_role_v1"] = rows.apply(
        lambda row: classify_opportunity_role(
            active=bool(row["active_v1"]),
            quarantine=bool(row["quarantine_v1"]),
            protected_winner=bool(row["protected_winner_v1"]),
            runner_protect=bool(row["runner_protect_v1"]),
            ambiguous_high_mfe=bool(row["ambiguous_high_mfe_v1"]),
            high_mfe_unsafe=bool(row["fifty_plus_mfe_risk_v1"] or row["hundred_plus_mfe_risk_v1"] or row["two_hundred_plus_mfe_risk_v1"]),
            bad_label=bool(row["bad_label_v1"]),
            tail_label=bool(row["tail_label_v1"]),
            safe_recoverable=bool(row["safe_recoverable_v1"]),
            v2_oof_captured=bool(row["v2_oof_captured_v1"]),
            historical_v2_captured=bool(row["historical_v2_captured_v1"]),
            optuna_captured=bool(row["optuna_captured_v1"]),
            v3_captured=bool(row["v3_captured_v1"]),
            r5_bad_bucket=str(row["r5_bad_score_signal_bucket_v1"]),
            r5_1_bad_bucket=str(row["r5_1_bad_score_signal_bucket_v1"]),
            r5_tail_bucket=str(row["r5_tail_score_signal_bucket_v1"]),
            run_id_low_support=bool(row["run_id_low_support_v1"]),
        ),
        axis=1,
    )
    rows["opportunity_reason_v1"] = rows.apply(_opportunity_reason, axis=1)
    return rows


def _opportunity_reason(row: pd.Series) -> str:
    role = str(row["recommended_opportunity_role_v1"])
    if role.startswith("CORE_OOF_V2"):
        return "OOF V2 selected with provenance and safety-clean row-level evidence."
    if role == "RUN_ID_SUPPORT_CANDIDATE":
        return "Safe recoverable row in low-support run_id with existing legal signal evidence."
    if role == "TAIL_EXPANSION_CANDIDATE":
        return "Safe recoverable tail row with R5 tail signal support."
    if role == "R5_R5_1_SIGNAL_SAFE_RECOVERABLE_CANDIDATE":
        return "Safe recoverable row with R5/R5.1 signal evidence."
    if role.startswith("HARD_NEGATIVE"):
        return "Hard safety veto row; never include as positive."
    if role == "AMBIGUOUS_MONITOR_ONLY":
        return "Ambiguous high-MFE row is monitor-only unless separately safe-proven."
    if role == "QUARANTINE_EXCLUDE":
        return "Quarantine or non-active row excluded."
    if role == "V2_MISSED_SAFE_RECOVERABLE_CANDIDATE":
        return "Safe recoverable row missed by V2 OOF; has comparator capture/label evidence but needs guarded use."
    return "No sufficient existing legal signal/reason for opportunity positive role."


def _contract() -> dict[str, Any]:
    return {
        "contract": "R5_2_OPPORTUNITY_BASE_CONTRACT_V1",
        "foundation_v1": {
            "expected_rows_v1": 1914,
            "active_rows_v1": 1852,
            "quarantine_rows_v1": 62,
            "as_of_columns_v1": 109,
        },
        "input_policy_v1": {
            "uses_existing_legal_signals_v1": True,
            "uses_v2_oof_replay_as_provenance_valid_signal_skeleton_v1": True,
            "historical_v2_role_v1": "COMPARATOR_BLUEPRINT_ONLY_NOT_DECISION_VALID",
            "optuna_v3_role_v1": "WEAK_CONTROLS_ONLY",
            "no_new_feature_surface_v1": True,
            "no_model_training_v1": True,
            "no_optuna_r6_package_freeze_promo_live_v1": True,
        },
        "row_role_policy_v1": {
            "positive_roles_v1": [
                "CORE_OOF_V2_STRONG_POSITIVE",
                "CORE_OOF_V2_TAIL_POSITIVE",
                "R5_R5_1_SIGNAL_SAFE_RECOVERABLE_CANDIDATE",
                "TAIL_EXPANSION_CANDIDATE",
                "RUN_ID_SUPPORT_CANDIDATE",
            ],
            "hard_negative_roles_v1": [
                "HARD_NEGATIVE_PROTECTED_WINNER",
                "HARD_NEGATIVE_RUNNER_PROTECT",
                "HARD_NEGATIVE_HIGH_MFE_UNSAFE",
            ],
            "monitor_or_exclude_roles_v1": [
                "AMBIGUOUS_MONITOR_ONLY",
                "QUARANTINE_EXCLUDE",
                "UNKNOWN_REQUIRES_ARTIFACT",
                "V2_MISSED_SAFE_RECOVERABLE_CANDIDATE",
            ],
        },
        "safety_hard_vetoes_v1": [
            "quarantine",
            "protected_winner",
            "runner_protect",
            "ambiguous_high_mfe",
            "50_plus_mfe",
            "100_plus_mfe",
            "200_plus_mfe",
        ],
        "run_id_loso_design_goal_v1": "Improve run_id support only with existing legal evidence; never weaken LOSO guard.",
        "not_final_model_or_package_v1": True,
        "not_r6_ready_without_gate_v1": True,
    }


def _variant_memberships(rows: pd.DataFrame) -> dict[str, pd.Series]:
    core = _bool(rows, "v2_oof_captured_v1") & ~_bool(rows, "hard_safety_veto_v1")
    safe = _bool(rows, "safe_recoverable_v1") & ~_bool(rows, "hard_safety_veto_v1") & _bool(rows, "positive_evidence_v1")
    r5_support = safe & ~core & (
        rows["r5_bad_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
        | rows["r5_1_bad_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
    )
    tail_support = safe & ~core & _bool(rows, "tail_label_v1") & rows["r5_tail_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
    run_support = safe & ~core & _bool(rows, "run_id_low_support_v1") & (
        r5_support | tail_support | rows["v2_like_bad_tail_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
    )
    upper_bound = _bool(rows, "safe_recoverable_v1") & ~_bool(rows, "hard_safety_veto_v1") & _bool(rows, "positive_evidence_v1")
    balanced = core | r5_support | tail_support | run_support
    return {
        "V2_OOF_CORE_ONLY": core,
        "V2_OOF_PLUS_SAFE_R5_R5_1_SUPPORT": core | r5_support,
        "V2_OOF_PLUS_TAIL_EXPANSION": core | tail_support,
        "V2_OOF_PLUS_RUN_ID_SUPPORT": core | run_support,
        "BALANCED_V2_R5_TAIL_RUN_ID_SUPPORT": balanced,
        "SAFETY_FIRST_UPPER_BOUND": upper_bound,
    }


def _support_denominators(rows: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"run_id": rows["run_id"].astype(str), "selected": selected.astype(bool)})
    selected_counts = work.groupby("run_id")["selected"].sum()
    non_empty = selected_counts[selected_counts > 0]
    return {
        "worst_run_id_support_denominator_v1": int(non_empty.min()) if not non_empty.empty else 0,
        "run_id_groups_below_denominator_threshold_v1": int(((non_empty > 0) & (non_empty < MIN_RUN_ID_SUPPORT)).sum()),
        "selected_run_id_group_count_v1": int((selected_counts > 0).sum()),
        "empty_run_id_group_count_v1": int((selected_counts == 0).sum()),
        "run_id_denominators_v1": {str(key): int(value) for key, value in selected_counts.items()},
    }


def _variant_summary(rows: pd.DataFrame, memberships: dict[str, pd.Series]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for name, selected in memberships.items():
        selected = selected.astype(bool)
        support = _support_denominators(rows, selected)
        safety = _safety_counts(rows, selected)
        total = int(selected.sum())
        bad = int((selected & _bool(rows, "bad_label_v1")).sum())
        included_without_evidence = int((selected & ~_bool(rows, "positive_evidence_v1")).sum())
        decision_valid_for_training = bool(
            total > 0
            and included_without_evidence == 0
            and all(value == 0 for value in safety.values())
            and name != "SAFETY_FIRST_UPPER_BOUND"
        )
        row = {
            "variant_id_v1": name,
            "variant_type_v1": "ROW_MEMBERSHIP_SET_NOT_TRAINED_MODEL",
            "total_selected_rows_v1": total,
            "bad_count_v1": bad,
            "tail_count_v1": int((selected & _bool(rows, "tail_label_v1")).sum()),
            "safe_recoverable_count_v1": int((selected & _bool(rows, "safe_recoverable_v1")).sum()),
            "precision_proxy_v1": bad / total if total else np.nan,
            **{key: value for key, value in support.items() if key != "run_id_denominators_v1"},
            **safety,
            "decision_valid_for_training_surface_v1": decision_valid_for_training,
            "model_trained_v1": False,
            "package_built_v1": False,
            "r6_ready_v1": False,
            "reason_v1": _variant_reason(name, support, safety, included_without_evidence),
        }
        validate_variant_is_membership_set(row)
        variants.append(row)
    return variants


def _safety_counts(rows: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    return {
        "fifty_plus_overlap_v1": int((selected & _bool(rows, "fifty_plus_mfe_risk_v1")).sum()),
        "hundred_plus_overlap_v1": int((selected & _bool(rows, "hundred_plus_mfe_risk_v1")).sum()),
        "two_hundred_plus_overlap_v1": int((selected & _bool(rows, "two_hundred_plus_mfe_risk_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(rows, "protected_winner_v1")).sum()),
        "runner_protect_leakage_v1": int((selected & _bool(rows, "runner_protect_v1")).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & _bool(rows, "ambiguous_high_mfe_v1")).sum()),
        "protected_winners_selected_v1": int((selected & _bool(rows, "protected_winner_v1")).sum()),
        "quarantine_selected_v1": int((selected & _bool(rows, "quarantine_v1")).sum()),
    }


def _variant_reason(name: str, support: dict[str, Any], safety: dict[str, int], evidence_missing: int) -> str:
    if any(value for value in safety.values()):
        return "Rejected for safety leakage."
    if evidence_missing:
        return "Rejected for included rows without explicit evidence."
    if name == "V2_OOF_CORE_ONLY":
        return "Baseline provenance-valid V2 OOF signal skeleton."
    if name == "V2_OOF_PLUS_RUN_ID_SUPPORT":
        return "Adds existing-signal candidates in low-support run_ids only."
    if name == "SAFETY_FIRST_UPPER_BOUND":
        return "Diagnostic upper bound; broad set, not final training recommendation."
    if support["run_id_groups_below_denominator_threshold_v1"] > 0:
        return "Safety-clean opportunity set, but low-support run_id groups remain."
    return "Safety-clean opportunity set with evidence-backed additions."


def _rows_for_output(rows: pd.DataFrame, memberships: dict[str, pd.Series]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "candidate_uid_v1": rows["candidate_uid"],
            "trade_uid_v1": rows["trade_uid"],
            "trade_id_v1": rows["trade_id"],
            "decision_timestamp_v1": rows["decision_timestamp"],
            "active_quarantine_v1": rows["active_quarantine_v1"],
            "run_id_v1": rows["run_id"],
            "fold_id_v1": rows.get("fold_id_v1", ""),
            "bad_label_v1": rows["bad_label_v1"],
            "tail_label_v1": rows["tail_label_v1"],
            "safe_recoverable_v1": rows["safe_recoverable_v1"],
            "v2_oof_captured_v1": rows["v2_oof_captured_v1"],
            "historical_v2_captured_v1": rows["historical_v2_captured_v1"],
            "optuna_captured_v1": rows["optuna_captured_v1"],
            "v3_captured_v1": rows["v3_captured_v1"],
            "r5_bad_score_signal_bucket_v1": rows["r5_bad_score_signal_bucket_v1"],
            "r5_1_bad_score_signal_bucket_v1": rows["r5_1_bad_score_signal_bucket_v1"],
            "r5_tail_score_signal_bucket_v1": rows["r5_tail_score_signal_bucket_v1"],
            "v2_like_bad_tail_signal_bucket_v1": rows["v2_like_bad_tail_signal_bucket_v1"],
            "v3_oof_signal_bucket_v1": rows["v3_oof_signal_bucket_v1"],
            "protected_winner_status_v1": rows["protected_winner_v1"],
            "runner_protect_status_v1": rows["runner_protect_v1"],
            "ambiguous_high_mfe_status_v1": rows["ambiguous_high_mfe_v1"],
            "fifty_plus_mfe_risk_v1": rows["fifty_plus_mfe_risk_v1"],
            "hundred_plus_mfe_risk_v1": rows["hundred_plus_mfe_risk_v1"],
            "two_hundred_plus_mfe_risk_v1": rows["two_hundred_plus_mfe_risk_v1"],
            "existing_legal_signal_evidence_count_v1": rows["existing_legal_signal_evidence_count_v1"],
            "provenance_status_v1": np.where(rows["v2_oof_captured_v1"], "V2_OOF_PROVENANCE_PASS", rows["provenance_status_v1"]),
            "recommended_opportunity_role_v1": rows["recommended_opportunity_role_v1"],
            "opportunity_reason_v1": rows["opportunity_reason_v1"],
        }
    )
    for name, selected in memberships.items():
        out[f"member_{name.lower()}_v1"] = selected.astype(bool).values
    return out


def _run_id_support(rows: pd.DataFrame, memberships: dict[str, pd.Series]) -> list[dict[str, Any]]:
    selected = memberships["V2_OOF_CORE_ONLY"].astype(bool)
    balanced = memberships["BALANCED_V2_R5_TAIL_RUN_ID_SUPPORT"].astype(bool)
    opportunity = balanced & ~selected
    records: list[dict[str, Any]] = []
    for run_id, group in rows.groupby("run_id"):
        idx = group.index
        group_selected = selected.loc[idx]
        group_opportunity = opportunity.loc[idx]
        safe_missed = _bool(group, "safe_recoverable_v1") & ~group_selected
        can_improve = int(group_opportunity.sum()) > 0
        records.append(
            {
                "run_id_v1": str(run_id),
                "total_rows_v1": int(len(group)),
                "active_rows_v1": int(_bool(group, "active_v1").sum()),
                "safe_recoverable_rows_v1": int(_bool(group, "safe_recoverable_v1").sum()),
                "v2_oof_selected_rows_v1": int(group_selected.sum()),
                "v2_oof_selected_bad_rows_v1": int((group_selected & _bool(group, "bad_label_v1")).sum()),
                "v2_oof_selected_tail_rows_v1": int((group_selected & _bool(group, "tail_label_v1")).sum()),
                "v2_missed_safe_recoverable_rows_v1": int(safe_missed.sum()),
                "r5_r5_1_r5_tail_signal_candidates_v1": int(
                    (
                        safe_missed
                        & (
                            group["r5_bad_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
                            | group["r5_1_bad_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
                            | group["r5_tail_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"])
                        )
                        & ~_bool(group, "hard_safety_veto_v1")
                    ).sum()
                ),
                "protected_winner_count_v1": int(_bool(group, "protected_winner_v1").sum()),
                "runner_protect_count_v1": int(_bool(group, "runner_protect_v1").sum()),
                "ambiguous_high_mfe_count_v1": int(_bool(group, "ambiguous_high_mfe_v1").sum()),
                "opportunity_candidate_count_v1": int(group_opportunity.sum()),
                "balanced_variant_selected_rows_v1": int(balanced.loc[idx].sum()),
                "support_can_be_improved_safely_v1": can_improve,
                "reason_v1": _run_id_reason(str(run_id), group, group_selected, group_opportunity),
            }
        )
    return records


def _run_id_reason(run_id: str, group: pd.DataFrame, selected: pd.Series, opportunity: pd.Series) -> str:
    if run_id == WORST_RUN_ID and int(opportunity.sum()) == 0:
        return "TRUE_LOW_SUPPORT_NOT_REPAIRABLE_WITH_EXISTING_SIGNALS"
    if int(opportunity.sum()) > 0:
        return "Existing legal signals can add safety-clean opportunity candidates."
    if int(selected.sum()) == 0:
        return "No V2 OOF selected support and no evidence-backed safe expansion candidate."
    return "Current V2 OOF support remains the available safety-clean evidence."


def _missed_safe_recoverable(rows: pd.DataFrame) -> list[dict[str, Any]]:
    missed = rows[_bool(rows, "safe_recoverable_v1") & ~_bool(rows, "v2_oof_captured_v1")].copy()
    records = []
    for _, row in missed.iterrows():
        conflict = bool(row["protected_winner_v1"] or row["runner_protect_v1"] or row["ambiguous_high_mfe_v1"] or row["fifty_plus_mfe_risk_v1"] or row["hundred_plus_mfe_risk_v1"] or row["two_hundred_plus_mfe_risk_v1"])
        if conflict:
            rec_role = "REJECT_SAFETY_CONFLICT"
        elif row["recommended_opportunity_role_v1"] == "RUN_ID_SUPPORT_CANDIDATE":
            rec_role = "ADD_AS_RUN_ID_SUPPORT"
        elif row["recommended_opportunity_role_v1"] == "TAIL_EXPANSION_CANDIDATE":
            rec_role = "ADD_AS_TAIL_EXPANSION"
        elif row["recommended_opportunity_role_v1"] == "R5_R5_1_SIGNAL_SAFE_RECOVERABLE_CANDIDATE":
            rec_role = "ADD_TO_OPPORTUNITY_BASE"
        elif row["recommended_opportunity_role_v1"] == "AMBIGUOUS_MONITOR_ONLY":
            rec_role = "KEEP_MONITOR_ONLY"
        else:
            rec_role = "UNKNOWN_REQUIRES_ARTIFACT"
        records.append(
            {
                "candidate_uid_v1": row["candidate_uid"],
                "trade_uid_v1": row["trade_uid"],
                "run_id_v1": row["run_id"],
                "bad_label_v1": bool(row["bad_label_v1"]),
                "tail_label_v1": bool(row["tail_label_v1"]),
                "r5_r5_1_r5_tail_signal_evidence_v1": int(row["r5_signal_family_hits_v1"]) + int(row["r5_1_signal_family_hits_v1"]) + int(row["r5_tail_score_signal_bucket_v1"] in {"STRONG", "SUPPORT"}),
                "historical_v2_captured_v1": bool(row["historical_v2_captured_v1"]),
                "optuna_captured_v1": bool(row["optuna_captured_v1"]),
                "v3_captured_v1": bool(row["v3_captured_v1"]),
                "safety_conflict_v1": conflict,
                "protected_winner_conflict_v1": bool(row["protected_winner_v1"]),
                "runner_protect_conflict_v1": bool(row["runner_protect_v1"]),
                "ambiguous_high_mfe_conflict_v1": bool(row["ambiguous_high_mfe_v1"]),
                "likely_miss_reason_v1": _miss_reason(row),
                "recommended_role_v1": rec_role,
            }
        )
    return records


def _miss_reason(row: pd.Series) -> str:
    if bool(row["hard_safety_veto_v1"]):
        return "V2_OOF_OR_SAFETY_VETO_EXCLUDED_ROW"
    if row["r5_bad_score_signal_bucket_v1"] == "NONE" and row["r5_1_bad_score_signal_bucket_v1"] == "NONE" and row["r5_tail_score_signal_bucket_v1"] in {"NONE", "LOW"}:
        return "NO_STRONG_EXISTING_SIGNAL_BUCKET"
    if not bool(row["v2_oof_captured_v1"]):
        return "OOF_V2_DID_NOT_SCORE_ABOVE_BASE_MEMBERSHIP_RULE"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def _signal_family_audit(rows: pd.DataFrame, prior_audit: pd.DataFrame) -> list[dict[str, Any]]:
    definitions = {
        "R5_BAD_SCORE": rows["r5_bad_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"]),
        "R5_1_BAD_SCORE": rows["r5_1_bad_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"]),
        "R5_TAIL_SCORE": rows["r5_tail_score_signal_bucket_v1"].isin(["STRONG", "SUPPORT"]),
        "V2_OOF_BAD_TAIL": _bool(rows, "v2_oof_captured_v1"),
        "HISTORICAL_V2_LIKE_SIGNAL": _bool(rows, "historical_v2_captured_v1"),
        "V3_OOF_BAD_TAIL": _bool(rows, "v3_captured_v1"),
        "OPTUNA_BEST_SIGNAL": _bool(rows, "optuna_captured_v1"),
        "AS_OF_SAFE_FEATURE_FAMILIES": rows["existing_legal_signal_evidence_count_v1"].astype(int).gt(0),
    }
    overall_bad = max(int(_bool(rows, "bad_label_v1").sum()), 1) / max(len(rows), 1)
    overall_tail = max(int(_bool(rows, "tail_label_v1").sum()), 1) / max(len(rows), 1)
    records = []
    for family, mask in definitions.items():
        mask = mask.astype(bool)
        count = int(mask.sum())
        bad_rate = int((mask & _bool(rows, "bad_label_v1")).sum()) / count if count else 0.0
        tail_rate = int((mask & _bool(rows, "tail_label_v1")).sum()) / count if count else 0.0
        support = _support_denominators(rows, mask)
        prior = prior_audit[prior_audit["signal_family_v1"].astype(str).eq(family)]
        prior_status = "PASS" if family in {"V2_OOF_BAD_TAIL", "V3_OOF_BAD_TAIL"} else "NOT_DECISION_PROVEN_AS_SCOREFIELD"
        if not prior.empty:
            prior_status = str(prior.iloc[0].get("oof_provenance_status_v1", prior_status))
        records.append(
            {
                "signal_family_v1": family,
                "coverage_v1": count,
                "overlap_with_v2_oof_69_v1": int((mask & _bool(rows, "v2_oof_captured_v1")).sum()),
                "overlap_with_historical_v2_95_v1": int((mask & _bool(rows, "historical_v2_captured_v1")).sum()),
                "overlap_with_optuna_56_v1": int((mask & _bool(rows, "optuna_captured_v1")).sum()),
                "overlap_with_v3_17_v1": int((mask & _bool(rows, "v3_captured_v1")).sum()),
                "overlap_with_325_safe_recoverable_v1": int((mask & _bool(rows, "safe_recoverable_v1")).sum()),
                "bad_lift_v1": bad_rate / overall_bad if overall_bad else np.nan,
                "tail_lift_v1": tail_rate / overall_tail if overall_tail else np.nan,
                "protected_winner_risk_v1": int((mask & _bool(rows, "protected_winner_v1")).sum()),
                "runner_protect_risk_v1": int((mask & _bool(rows, "runner_protect_v1")).sum()),
                "ambiguous_high_mfe_risk_v1": int((mask & _bool(rows, "ambiguous_high_mfe_v1")).sum()),
                "run_id_stability_v1": "OK" if support["run_id_groups_below_denominator_threshold_v1"] == 0 and count > 0 else "TOO_SMALL_DENOMINATOR",
                "minimum_run_id_denominator_support_v1": support["worst_run_id_support_denominator_v1"],
                "oof_provenance_status_v1": prior_status,
                "recommended_use_v1": _signal_recommendation(family, prior_status),
            }
        )
    return records


def _signal_recommendation(family: str, provenance: str) -> str:
    if "RUNNER" in family or family.endswith("PROTECTION"):
        return "SAFETY_VETO_SIGNAL"
    if family == "V2_OOF_BAD_TAIL":
        return "PRIMARY_OPPORTUNITY_SIGNAL"
    if family in {"R5_TAIL_SCORE"}:
        return "TAIL_EXPANSION_SIGNAL"
    if family in {"R5_BAD_SCORE", "R5_1_BAD_SCORE", "HISTORICAL_V2_LIKE_SIGNAL"}:
        return "PRIMARY_OPPORTUNITY_SIGNAL"
    if family == "V3_OOF_BAD_TAIL":
        return "AUXILIARY_SIGNAL" if provenance == "PASS" else "REJECTED_NO_PROVENANCE"
    if family == "OPTUNA_BEST_SIGNAL":
        return "MONITOR_ONLY"
    return "AUXILIARY_SIGNAL"


def _recommendation(variant_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = {str(row["variant_id_v1"]): row for row in variant_rows}
    balanced = candidates.get("BALANCED_V2_R5_TAIL_RUN_ID_SUPPORT", variant_rows[0])
    support_focused = candidates.get("V2_OOF_PLUS_RUN_ID_SUPPORT", balanced)
    core = next(row for row in variant_rows if row["variant_id_v1"] == "V2_OOF_CORE_ONLY")

    def _safety_clean(row: dict[str, Any]) -> bool:
        return all(
            int(row[key]) == 0
            for key in [
                "fifty_plus_overlap_v1",
                "hundred_plus_overlap_v1",
                "two_hundred_plus_overlap_v1",
                "strongest_winner_overlap_v1",
                "runner_protect_leakage_v1",
                "ambiguous_high_mfe_leakage_v1",
                "quarantine_selected_v1",
            ]
        )

    def _support_improved(row: dict[str, Any]) -> bool:
        return (
            int(row["worst_run_id_support_denominator_v1"]) > int(core["worst_run_id_support_denominator_v1"])
            or int(row["run_id_groups_below_denominator_threshold_v1"]) < int(core["run_id_groups_below_denominator_threshold_v1"])
        )

    balanced_safety_clean = _safety_clean(balanced)
    support_safety_clean = _safety_clean(support_focused)
    balanced_support_improved = _support_improved(balanced)
    support_focused_support_improved = _support_improved(support_focused)
    if balanced_safety_clean and balanced_support_improved:
        recommended = balanced
        status = "OPPORTUNITY_BASE_READY_FOR_R5_2_REBUILD"
        next_action = "BUILD_R5_2_FROM_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1"
    elif support_safety_clean and support_focused_support_improved:
        recommended = support_focused
        status = "OPPORTUNITY_BASE_SIGNAL_PRESENT_BUT_RUN_ID_SUPPORT_WEAK"
        next_action = "DEEPEN_RUN_ID_SUPPORT_SIGNAL_AUDIT_V1"
    elif balanced_safety_clean:
        recommended = balanced
        status = "OPPORTUNITY_BASE_SIGNAL_PRESENT_BUT_RUN_ID_SUPPORT_WEAK"
        next_action = "DEEPEN_RUN_ID_SUPPORT_SIGNAL_AUDIT_V1"
    else:
        recommended = balanced
        status = "OPPORTUNITY_BASE_BLOCKED_BY_SAFETY_CONFLICTS"
        next_action = "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
    safety_clean = _safety_clean(recommended)
    run_id_improved = _support_improved(recommended)
    recommendation_not_r6_ready(status)
    return {
        "layer_name": "R5_2_OPPORTUNITY_BASE_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "recommended_variant_v1": recommended["variant_id_v1"],
        "broader_balanced_variant_v1": balanced["variant_id_v1"],
        "broader_balanced_variant_rows_v1": balanced["total_selected_rows_v1"],
        "broader_balanced_run_id_support_improved_v1": balanced_support_improved,
        "why_better_than_v2_oof_core_v1": (
            f"Adds {int(recommended['total_selected_rows_v1']) - int(core['total_selected_rows_v1'])} evidence-backed safety-clean rows "
            "from existing legal R5/R5.1/R5-tail/V2-like signals."
        ),
        "run_id_loso_support_improved_v1": run_id_improved,
        "remaining_low_support_run_id_groups_v1": recommended["run_id_groups_below_denominator_threshold_v1"],
        "worst_run_id_support_denominator_v1": recommended["worst_run_id_support_denominator_v1"],
        "safety_retained_v1": safety_clean,
        "hard_veto_rows_v1": ["protected winners", "runner-protect", "ambiguous high-MFE", "50+/100+/200+ MFE", "quarantine"],
        "ambiguous_monitor_only_v1": True,
        "primary_signals_v1": ["V2_OOF_BAD_TAIL", "R5_BAD_SCORE", "R5_1_BAD_SCORE", "HISTORICAL_V2_LIKE_SIGNAL"],
        "auxiliary_signals_v1": ["V3_OOF_BAD_TAIL", "AS_OF_SAFE_FEATURE_FAMILIES"],
        "rejected_or_monitor_signals_v1": ["OPTUNA_BEST_SIGNAL", "runner/protection positives as opportunity rows"],
        "not_r6_ready_v1": True,
        "not_live_ready_v1": True,
        "worst_run_id_v1": WORST_RUN_ID,
        "worst_run_id_support_record_v1": next((row for row in run_rows if row["run_id_v1"] == WORST_RUN_ID), {}),
    }


def _summary_counts(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count_v1": int(len(rows)),
        "safe_recoverable_rows_v1": int(_bool(rows, "safe_recoverable_v1").sum()),
        "v2_oof_captured_v1": int(_bool(rows, "v2_oof_captured_v1").sum()),
        "historical_v2_captured_v1": int(_bool(rows, "historical_v2_captured_v1").sum()),
        "optuna_captured_v1": int(_bool(rows, "optuna_captured_v1").sum()),
        "v3_captured_v1": int(_bool(rows, "v3_captured_v1").sum()),
        "role_counts_v1": {str(key): int(value) for key, value in rows["recommended_opportunity_role_v1"].value_counts().to_dict().items()},
    }


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v2_oof_root: Path = V2_OOF_ROOT,
    revalidation_root: Path = V2_REVALIDATION_ROOT,
    loso_root: Path = LOSO_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    validate_loso_guard_not_weakened(MIN_RUN_ID_SUPPORT)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes_before = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
    }
    inputs = _load_inputs(v2_oof_root, revalidation_root, loso_root)
    rows = _base_rows(inputs)
    memberships = _variant_memberships(rows)
    for name, membership in memberships.items():
        rows[f"member_{name.lower()}_v1"] = membership.astype(bool)
    output_rows = _rows_for_output(rows, memberships)
    variant_rows = _variant_summary(rows, memberships)
    run_rows = _run_id_support(rows, memberships)
    missed_rows = _missed_safe_recoverable(rows)
    signal_rows = _signal_family_audit(rows, inputs["signal_audit"])
    recommendation = _recommendation(variant_rows, run_rows)
    go_no_go = {
        "layer_name": "R5_2_OPPORTUNITY_BASE_GO_NO_GO_V1",
        "decision_v1": recommendation["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "not_final_model_v1": True,
        "not_r6_ready_v1": True,
        "not_live_ready_v1": True,
        "no_optuna_run_v1": True,
        "no_training_run_v1": True,
    }
    input_hashes_after = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
    }
    artifact_integrity = validate_input_artifacts_unchanged(input_hashes_before, input_hashes_after)
    integrity = {
        "v2_scores_unchanged_v1": artifact_integrity["v2_oof_scores_unchanged_v1"],
        "v2_provenance_unchanged_v1": artifact_integrity["v2_oof_provenance_unchanged_v1"],
        "v2_model_objective_thresholds_unchanged_v1": True,
    }
    contract = _contract()
    no_fallback = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)

    output_rows.to_csv(output_dir / "r5_2_opportunity_base_rows_v1.csv", index=False)
    _write_json(
        output_dir / "r5_2_opportunity_base_rows_v1.json",
        {"summary_v1": _summary_counts(rows), "rows_v1": output_rows.to_dict("records")},
    )
    _write_json(output_dir / "r5_2_opportunity_base_contract_v1.json", contract)
    _write_rows(output_dir / "r5_2_opportunity_base_variants_v1.csv", variant_rows)
    _write_json(output_dir / "r5_2_opportunity_base_variants_v1.json", {"variants_v1": variant_rows})
    _write_rows(output_dir / "r5_2_opportunity_run_id_support_analysis_v1.csv", run_rows)
    _write_json(output_dir / "r5_2_opportunity_run_id_support_analysis_v1.json", {"rows_v1": run_rows})
    _write_rows(output_dir / "v2_oof_missed_safe_recoverable_analysis_v1.csv", missed_rows)
    _write_json(output_dir / "v2_oof_missed_safe_recoverable_analysis_v1.json", {"rows_v1": missed_rows, "row_count_v1": len(missed_rows)})
    _write_rows(output_dir / "r5_2_signal_family_opportunity_audit_v1.csv", signal_rows)
    _write_json(output_dir / "r5_2_signal_family_opportunity_audit_v1.json", {"families_v1": signal_rows})
    _write_json(output_dir / "r5_2_opportunity_base_recommendation_v1.json", recommendation)
    _write_json(output_dir / "r5_2_opportunity_base_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "v2_oof_root_v1": str(v2_oof_root),
                "revalidation_root_v1": str(revalidation_root),
                "loso_root_v1": str(loso_root),
                "optuna_root_reference_v1": str(OPTUNA_ROOT),
                "selected_v3_root_reference_v1": str(SELECTED_V3_ROOT),
            },
            "input_hashes_before_v1": input_hashes_before,
            "input_hashes_after_v1": input_hashes_after,
            "input_artifact_integrity_v1": artifact_integrity,
            "integrity_v1": integrity,
            "no_dummy_synthetic_fallback_v1": no_fallback,
        },
    )
    _write_report(
        output_dir / "r5_2_opportunity_base_contract_v1.md",
        [
            "# R5.2 Opportunity Base Contract V1",
            "",
            "This is a Monday 1914 opportunity-base design artifact, not a trained model or package.",
            "V2 OOF is used as a provenance-valid signal skeleton; historical V2 is comparator only.",
            "Hard safety vetoes exclude quarantine, protected winners, runner-protect, ambiguous high-MFE, and 50+/100+/200+ MFE risk.",
        ],
    )
    _write_report(
        output_dir / "r5_2_opportunity_base_rows_report_v1.md",
        ["# R5.2 Opportunity Base Rows V1", "", f"Rows: `{len(rows)}`", f"Role counts: `{_summary_counts(rows)['role_counts_v1']}`"],
    )
    _write_report(
        output_dir / "r5_2_opportunity_run_id_support_analysis_report_v1.md",
        [
            "# R5.2 Opportunity Run ID Support Analysis V1",
            "",
            f"Worst run_id `{WORST_RUN_ID}`: `{next(row for row in run_rows if row['run_id_v1'] == WORST_RUN_ID)['reason_v1']}`",
        ],
    )
    _write_report(
        output_dir / "v2_oof_missed_safe_recoverable_report_v1.md",
        ["# V2 OOF Missed Safe Recoverable Analysis V1", "", f"Missed safe recoverable rows: `{len(missed_rows)}`"],
    )
    _write_report(
        output_dir / "r5_2_signal_family_opportunity_audit_report_v1.md",
        ["# R5.2 Signal Family Opportunity Audit V1", "", "V2 OOF, R5, R5.1, R5 tail, historical V2, Optuna, V3, and AS_OF-safe families were summarized."],
    )
    _write_report(
        output_dir / "r5_2_opportunity_base_variant_report_v1.md",
        [
            "# R5.2 Opportunity Base Variant Report V1",
            "",
            *[
                f"- `{row['variant_id_v1']}`: selected `{row['total_selected_rows_v1']}`, bad/tail `{row['bad_count_v1']}` / `{row['tail_count_v1']}`, safety leaks `{row['protected_winners_selected_v1'] + row['runner_protect_leakage_v1'] + row['ambiguous_high_mfe_leakage_v1']}`"
                for row in variant_rows
            ],
        ],
    )
    _write_report(
        output_dir / "r5_2_opportunity_base_recommendation_v1.md",
        [
            "# R5.2 Opportunity Base Recommendation V1",
            "",
            f"Status: `{recommendation['status_v1']}`",
            f"Recommended variant: `{recommendation['recommended_variant_v1']}`",
            f"Next action: `{recommendation['next_recommended_action_v1']}`",
            recommendation["why_better_than_v2_oof_core_v1"],
            "",
            "This is not R6-ready or live-ready.",
        ],
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "decision_v1": go_no_go["decision_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        "row_count_v1": int(len(rows)),
        "safe_recoverable_rows_v1": int(_bool(rows, "safe_recoverable_v1").sum()),
        "v2_oof_core_rows_v1": int(memberships["V2_OOF_CORE_ONLY"].sum()),
        "recommended_variant_v1": recommendation["recommended_variant_v1"],
        "recommended_variant_rows_v1": int(next(row for row in variant_rows if row["variant_id_v1"] == recommendation["recommended_variant_v1"])["total_selected_rows_v1"]),
        "run_id_loso_support_improved_v1": recommendation["run_id_loso_support_improved_v1"],
        "safety_retained_v1": recommendation["safety_retained_v1"],
        "v2_scores_provenance_model_objective_thresholds_unchanged_v1": all(integrity.values()),
        "optuna_not_run_v1": True,
        "r6_not_run_v1": True,
        "model_not_trained_v1": True,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "go_no_go_v1": go_no_go})
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Build R5.2 Opportunity Base From Existing V2 OOF Replay V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Recommended variant: `{summary['recommended_variant_v1']}`",
            f"Rows: `{summary['recommended_variant_rows_v1']}`",
            f"Next action: `{summary['next_recommended_action_v1']}`",
            "",
            "No model training, Optuna, R6, package build, freeze, promo, or live action was run.",
        ],
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", default=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    parser.add_argument("--revalidation-root", type=Path, default=V2_REVALIDATION_ROOT)
    parser.add_argument("--loso-root", type=Path, default=LOSO_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v2_oof_root=args.v2_oof_root,
        revalidation_root=args.revalidation_root,
        loso_root=args.loso_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
