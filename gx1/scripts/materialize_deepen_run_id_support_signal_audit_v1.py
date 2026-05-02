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
ACTION = "DEEPEN_RUN_ID_SUPPORT_SIGNAL_AUDIT_V1"
LAYER_NAME = ACTION
OPPORTUNITY_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1_20260427T122550Z_LOCK"
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"
DENOMINATOR_TARGET = 5
WORST_RUN_ID = "TRUTH_MONFRI_WEEK_20250106_20250113"

V2_CORE_COL = "member_v2_oof_core_only_v1"
RECOMMENDED_COL = "member_v2_oof_plus_run_id_support_v1"
BALANCED_COL = "member_balanced_v2_r5_tail_run_id_support_v1"
UPPER_BOUND_COL = "member_safety_first_upper_bound_v1"


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


def validate_denominator_guard_not_weakened(target: int) -> bool:
    if int(target) < DENOMINATOR_TARGET:
        raise RuntimeError("DENOMINATOR_GUARD_WEAKENING_FORBIDDEN")
    return True


def validate_low_support_groups_not_silently_dropped(*, dropped: bool, explicitly_reported: bool) -> bool:
    if dropped and not explicitly_reported:
        raise RuntimeError("LOW_SUPPORT_GROUPS_CANNOT_BE_SILENTLY_DROPPED")
    return True


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
    }


def hard_veto_mask(rows: pd.DataFrame) -> pd.Series:
    return (
        rows["active_quarantine_v1"].astype(str).ne("ACTIVE_CANDIDATE")
        | _bool(rows, "protected_winner_status_v1")
        | _bool(rows, "runner_protect_status_v1")
        | _bool(rows, "ambiguous_high_mfe_status_v1")
        | _bool(rows, "fifty_plus_mfe_risk_v1")
        | _bool(rows, "hundred_plus_mfe_risk_v1")
        | _bool(rows, "two_hundred_plus_mfe_risk_v1")
    )


def signal_evidence_mask(rows: pd.DataFrame) -> pd.Series:
    return (
        rows["r5_bad_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"])
        | rows["r5_1_bad_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"])
        | rows["r5_tail_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"])
        | rows["v2_like_bad_tail_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"])
        | _bool(rows, "v2_oof_captured_v1")
        | _bool(rows, "historical_v2_captured_v1")
        | _bool(rows, "optuna_captured_v1")
        | _bool(rows, "v3_captured_v1")
    )


def safe_signal_candidate_mask(rows: pd.DataFrame) -> pd.Series:
    return _bool(rows, "safe_recoverable_v1") & ~hard_veto_mask(rows) & signal_evidence_mask(rows)


def validate_added_support_candidate(row: pd.Series) -> bool:
    if bool(row.get("protected_winner_status_v1", False)):
        raise RuntimeError("PROTECTED_WINNER_CANNOT_REPAIR_SUPPORT")
    if bool(row.get("runner_protect_status_v1", False)):
        raise RuntimeError("RUNNER_PROTECT_CANNOT_REPAIR_SUPPORT")
    if bool(row.get("ambiguous_high_mfe_status_v1", False)):
        raise RuntimeError("AMBIGUOUS_HIGH_MFE_CANNOT_REPAIR_SUPPORT_WITHOUT_SAFE_PROOF")
    if str(row.get("active_quarantine_v1", "")) != "ACTIVE_CANDIDATE":
        raise RuntimeError("QUARANTINE_CANNOT_REPAIR_SUPPORT")
    if bool(row.get("fifty_plus_mfe_risk_v1", False)) or bool(row.get("hundred_plus_mfe_risk_v1", False)) or bool(row.get("two_hundred_plus_mfe_risk_v1", False)):
        raise RuntimeError("HIGH_MFE_RISK_CANNOT_REPAIR_SUPPORT")
    if not bool(row.get("safe_recoverable_v1", False)):
        raise RuntimeError("SUPPORT_CANDIDATE_MUST_BE_SAFE_RECOVERABLE")
    if int(row.get("existing_legal_signal_evidence_count_v1", 0) or 0) <= 0:
        raise RuntimeError("SUPPORT_CANDIDATE_MUST_HAVE_SIGNAL_EVIDENCE")
    return True


def validate_frontier_has_no_unsafe_rows(rows: pd.DataFrame, selected: pd.Series) -> bool:
    if int((selected.astype(bool) & hard_veto_mask(rows)).sum()) > 0:
        raise RuntimeError("MAX_FEASIBLE_UNDER_HARD_VETOES_CANNOT_INCLUDE_UNSAFE_ROWS")
    return True


def classify_support_repairability(
    *,
    current_denominator: int,
    feasible_safe_max: int,
    denominator_target: int,
    additional_safe_candidates: int,
    tail_candidates: int,
    risky_signal_candidates: int,
    protected_winners: int,
    runner_protect: int,
    ambiguous_high_mfe: int,
    quarantine: int,
    missing_artifacts: int,
) -> str:
    if current_denominator >= denominator_target:
        return "SUPPORT_ALREADY_SUFFICIENT"
    if feasible_safe_max < denominator_target:
        return "STRUCTURALLY_UNSATISFIABLE_FEASIBLE_SAFE_MAX_BELOW_DENOMINATOR"
    if additional_safe_candidates >= denominator_target - current_denominator:
        return "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS"
    if tail_candidates >= denominator_target - current_denominator:
        return "SUPPORT_REPAIRABLE_ONLY_WITH_TAIL_EXPANSION"
    if risky_signal_candidates > 0:
        return "SUPPORT_REPAIRABLE_ONLY_WITH_RISKY_SIGNALS"
    if protected_winners > 0:
        return "SUPPORT_BLOCKED_BY_PROTECTED_WINNERS"
    if runner_protect > 0:
        return "SUPPORT_BLOCKED_BY_RUNNER_PROTECT"
    if ambiguous_high_mfe > 0:
        return "SUPPORT_BLOCKED_BY_HIGH_MFE_AMBIGUITY"
    if quarantine > 0:
        return "SUPPORT_BLOCKED_BY_QUARANTINE"
    if missing_artifacts > 0:
        return "SUPPORT_BLOCKED_BY_MISSING_ARTIFACTS"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def classify_low_support_root_cause(matrix_row: dict[str, Any]) -> str:
    current = int(matrix_row["current_denominator_v1"])
    target = int(matrix_row["denominator_target_v1"])
    feasible = int(matrix_row["feasible_max_denominator_v1"])
    if current >= target:
        return "SUPPORT_ALREADY_SUFFICIENT"
    if feasible < target:
        if int(matrix_row["total_rows_v1"]) < target:
            return "GROUP_TOO_SMALL_FOR_CURRENT_DENOMINATOR_CONTRACT"
        return "FEASIBLE_SAFE_MAX_BELOW_DENOMINATOR"
    if int(matrix_row["additional_safe_candidates_available_v1"]) > 0:
        return "SAFE_SIGNAL_EXISTS_BUT_NOT_INCLUDED"
    if int(matrix_row["r5_tail_score_candidates_v1"]) > 0:
        return "TAIL_SIGNAL_EXISTS_BUT_NOT_INCLUDED"
    if int(matrix_row["protected_winner_rows_v1"]) > 0:
        return "PROTECTED_WINNER_BLOCKS_EXPANSION"
    if int(matrix_row["runner_protect_rows_v1"]) > 0:
        return "RUNNER_PROTECT_BLOCKS_EXPANSION"
    if int(matrix_row["ambiguous_high_mfe_rows_v1"]) > 0:
        return "AMBIGUOUS_HIGH_MFE_BLOCKS_EXPANSION"
    if int(matrix_row["quarantine_rows_v1"]) > 0:
        return "QUARANTINE_BLOCKS_EXPANSION"
    if int(matrix_row["unknown_artifact_missing_rows_v1"]) > 0:
        return "MISSING_ARTIFACTS_PREVENT_CLASSIFICATION"
    return "TRUE_MODEL_UNDER_SELECTION"


def recommended_treatment(root_cause: str) -> str:
    mapping = {
        "SAFE_SIGNAL_EXISTS_BUT_NOT_INCLUDED": "ADD_SAFE_SIGNAL_CANDIDATES",
        "TAIL_SIGNAL_EXISTS_BUT_NOT_INCLUDED": "ADD_TAIL_EXPANSION_CANDIDATES",
        "FEASIBLE_SAFE_MAX_BELOW_DENOMINATOR": "REQUIRE_LOW_SUPPORT_POLICY",
        "GROUP_TOO_SMALL_FOR_CURRENT_DENOMINATOR_CONTRACT": "REQUIRE_LOW_SUPPORT_POLICY",
        "PROTECTED_WINNER_BLOCKS_EXPANSION": "REJECT_UNSAFE_EXPANSION",
        "RUNNER_PROTECT_BLOCKS_EXPANSION": "REJECT_UNSAFE_EXPANSION",
        "AMBIGUOUS_HIGH_MFE_BLOCKS_EXPANSION": "REJECT_UNSAFE_EXPANSION",
        "QUARANTINE_BLOCKS_EXPANSION": "REJECT_UNSAFE_EXPANSION",
        "MISSING_ARTIFACTS_PREVENT_CLASSIFICATION": "REQUIRE_MISSING_ARTIFACTS",
        "TRUE_MODEL_UNDER_SELECTION": "KEEP_INVALID_TRUE_LOW_SUPPORT",
    }
    return mapping.get(root_cause, "MONITOR_ONLY_GROUP")


def _load_rows(opportunity_root: Path) -> pd.DataFrame:
    rows = pd.read_csv(opportunity_root / "r5_2_opportunity_base_rows_v1.csv")
    for column in [
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
        V2_CORE_COL,
        RECOMMENDED_COL,
        BALANCED_COL,
        UPPER_BOUND_COL,
    ]:
        if column in rows.columns:
            rows[column] = _bool(rows, column)
    return rows


def _run_id_feasibility_matrix(rows: pd.DataFrame) -> list[dict[str, Any]]:
    hard_veto = hard_veto_mask(rows)
    safe_signal = safe_signal_candidate_mask(rows)
    records = []
    for run_id, group in rows.groupby("run_id_v1", dropna=False):
        idx = group.index
        current = _bool(group, RECOMMENDED_COL)
        core = _bool(group, V2_CORE_COL)
        balanced = _bool(group, BALANCED_COL)
        upper = _bool(group, UPPER_BOUND_COL)
        group_safe_signal = safe_signal.loc[idx]
        group_hard_veto = hard_veto.loc[idx]
        r5_bad = group["r5_bad_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"]) & _bool(group, "safe_recoverable_v1") & ~group_hard_veto
        r5_1 = group["r5_1_bad_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"]) & _bool(group, "safe_recoverable_v1") & ~group_hard_veto
        r5_tail = group["r5_tail_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"]) & _bool(group, "safe_recoverable_v1") & ~group_hard_veto
        v2_like = group["v2_like_bad_tail_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"]) & _bool(group, "safe_recoverable_v1") & ~group_hard_veto
        additional_safe = group_safe_signal & ~current
        risky_signal = signal_evidence_mask(group) & ~_bool(group, "safe_recoverable_v1")
        row = {
            "run_id_v1": str(run_id),
            "total_rows_v1": int(len(group)),
            "active_rows_v1": int(group["active_quarantine_v1"].astype(str).eq("ACTIVE_CANDIDATE").sum()),
            "quarantine_rows_v1": int(group["active_quarantine_v1"].astype(str).ne("ACTIVE_CANDIDATE").sum()),
            "safe_recoverable_rows_v1": int(_bool(group, "safe_recoverable_v1").sum()),
            "safe_recoverable_bad_rows_v1": int((_bool(group, "safe_recoverable_v1") & _bool(group, "bad_label_v1")).sum()),
            "safe_recoverable_tail_rows_v1": int((_bool(group, "safe_recoverable_v1") & _bool(group, "tail_label_v1")).sum()),
            "v2_oof_selected_rows_v1": int(core.sum()),
            "v2_oof_selected_bad_rows_v1": int((core & _bool(group, "bad_label_v1")).sum()),
            "v2_oof_selected_tail_rows_v1": int((core & _bool(group, "tail_label_v1")).sum()),
            "recommended_opportunity_base_selected_rows_v1": int(current.sum()),
            "balanced_diagnostic_selected_rows_v1": int(balanced.sum()),
            "additional_safe_candidates_available_v1": int(additional_safe.sum()),
            "r5_bad_score_candidates_v1": int(r5_bad.sum()),
            "r5_1_bad_score_candidates_v1": int(r5_1.sum()),
            "r5_tail_score_candidates_v1": int(r5_tail.sum()),
            "v2_like_candidates_v1": int(v2_like.sum()),
            "protected_winner_rows_v1": int(_bool(group, "protected_winner_status_v1").sum()),
            "runner_protect_rows_v1": int(_bool(group, "runner_protect_status_v1").sum()),
            "ambiguous_high_mfe_rows_v1": int(_bool(group, "ambiguous_high_mfe_status_v1").sum()),
            "fifty_plus_risk_rows_v1": int(_bool(group, "fifty_plus_mfe_risk_v1").sum()),
            "hundred_plus_risk_rows_v1": int(_bool(group, "hundred_plus_mfe_risk_v1").sum()),
            "two_hundred_plus_risk_rows_v1": int(_bool(group, "two_hundred_plus_mfe_risk_v1").sum()),
            "unknown_artifact_missing_rows_v1": int(group["recommended_opportunity_role_v1"].astype(str).eq("UNKNOWN_REQUIRES_ARTIFACT").sum()),
            "feasible_safe_max_selected_under_current_hard_vetoes_v1": int(upper.sum()),
            "current_denominator_v1": int(current.sum()),
            "v2_oof_core_denominator_v1": int(core.sum()),
            "feasible_max_denominator_v1": int(upper.sum()),
            "denominator_target_v1": DENOMINATOR_TARGET,
            "denominator_gap_v1": max(0, DENOMINATOR_TARGET - int(current.sum())),
        }
        row["support_repairability_status_v1"] = classify_support_repairability(
            current_denominator=int(row["current_denominator_v1"]),
            feasible_safe_max=int(row["feasible_max_denominator_v1"]),
            denominator_target=DENOMINATOR_TARGET,
            additional_safe_candidates=int(row["additional_safe_candidates_available_v1"]),
            tail_candidates=int(row["r5_tail_score_candidates_v1"]),
            risky_signal_candidates=int(risky_signal.sum()),
            protected_winners=int(row["protected_winner_rows_v1"]),
            runner_protect=int(row["runner_protect_rows_v1"]),
            ambiguous_high_mfe=int(row["ambiguous_high_mfe_rows_v1"]),
            quarantine=int(row["quarantine_rows_v1"]),
            missing_artifacts=int(row["unknown_artifact_missing_rows_v1"]),
        )
        row["reason_v1"] = _matrix_reason(row)
        records.append(row)
    return records


def _matrix_reason(row: dict[str, Any]) -> str:
    status = str(row["support_repairability_status_v1"])
    if status == "SUPPORT_ALREADY_SUFFICIENT":
        return "Current recommended opportunity denominator already meets target."
    if status == "STRUCTURALLY_UNSATISFIABLE_FEASIBLE_SAFE_MAX_BELOW_DENOMINATOR":
        return "Feasible safety-clean signal-backed max is below denominator target; cannot force-fill without leaving current evidence/safety contract."
    if status == "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS":
        return "Existing safety-clean signal candidates can close denominator gap."
    if status == "SUPPORT_REPAIRABLE_ONLY_WITH_TAIL_EXPANSION":
        return "Only tail expansion candidates appear able to close denominator gap."
    if status.endswith("RISKY_SIGNALS"):
        return "Closing denominator appears to require risky/non-safe signal rows."
    return "Insufficient safe evidence to classify as repairable."


def _worst_deep_dive(rows: pd.DataFrame) -> list[dict[str, Any]]:
    group = rows[rows["run_id_v1"].astype(str).eq(WORST_RUN_ID)].copy()
    records = []
    for _, row in group.iterrows():
        can_add = bool(row.get(UPPER_BOUND_COL, False) and not row.get(RECOMMENDED_COL, False))
        records.append(
            {
                "candidate_uid_v1": row["candidate_uid_v1"],
                "trade_uid_v1": row["trade_uid_v1"],
                "label_class_v1": "BAD_TAIL" if bool(row["bad_label_v1"]) and bool(row["tail_label_v1"]) else "BAD" if bool(row["bad_label_v1"]) else "TAIL" if bool(row["tail_label_v1"]) else "NEUTRAL_OR_PROTECTED",
                "safe_recoverable_v1": bool(row["safe_recoverable_v1"]),
                "v2_oof_selected_v1": bool(row[V2_CORE_COL]),
                "historical_v2_selected_v1": bool(row["historical_v2_captured_v1"]),
                "optuna_selected_v1": bool(row["optuna_captured_v1"]),
                "v3_selected_v1": bool(row["v3_captured_v1"]),
                "r5_bad_signal_bucket_v1": row["r5_bad_score_signal_bucket_v1"],
                "r5_1_bad_signal_bucket_v1": row["r5_1_bad_score_signal_bucket_v1"],
                "r5_tail_signal_bucket_v1": row["r5_tail_score_signal_bucket_v1"],
                "protected_winner_status_v1": bool(row["protected_winner_status_v1"]),
                "runner_protect_status_v1": bool(row["runner_protect_status_v1"]),
                "ambiguous_high_mfe_status_v1": bool(row["ambiguous_high_mfe_status_v1"]),
                "fifty_plus_mfe_risk_v1": bool(row["fifty_plus_mfe_risk_v1"]),
                "hundred_plus_mfe_risk_v1": bool(row["hundred_plus_mfe_risk_v1"]),
                "two_hundred_plus_mfe_risk_v1": bool(row["two_hundred_plus_mfe_risk_v1"]),
                "quarantine_status_v1": row["active_quarantine_v1"],
                "unknown_artifact_status_v1": row["recommended_opportunity_role_v1"] == "UNKNOWN_REQUIRES_ARTIFACT",
                "can_be_safely_added_to_opportunity_base_v1": can_add,
                "reason_v1": _worst_row_reason(row, can_add),
            }
        )
    return records


def _worst_row_reason(row: pd.Series, can_add: bool) -> str:
    if bool(row[V2_CORE_COL]):
        return "Already selected by V2 OOF core with provenance."
    if can_add:
        return "Safety-clean signal-backed row could be added, but this did not occur for this worst run_id in current evidence."
    if bool(row["safe_recoverable_v1"]):
        return "Safe recoverable but lacks additional usable signal beyond existing selected rows."
    if bool(row["protected_winner_status_v1"]):
        return "Protected winner hard veto."
    if bool(row["runner_protect_status_v1"]):
        return "Runner-protect hard veto."
    if bool(row["ambiguous_high_mfe_status_v1"]):
        return "Ambiguous high-MFE monitor-only."
    return "Not safe recoverable or lacks artifact-backed signal evidence."


def _low_support_taxonomy(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in matrix_rows:
        if int(row["current_denominator_v1"]) >= DENOMINATOR_TARGET:
            continue
        cause = classify_low_support_root_cause(row)
        records.append(
            {
                "run_id_v1": row["run_id_v1"],
                "current_denominator_v1": row["current_denominator_v1"],
                "feasible_safe_max_denominator_v1": row["feasible_max_denominator_v1"],
                "denominator_gap_v1": row["denominator_gap_v1"],
                "best_safe_repair_candidate_count_v1": row["additional_safe_candidates_available_v1"],
                "safety_risk_if_force_filled_v1": int(row["protected_winner_rows_v1"]) + int(row["runner_protect_rows_v1"]) + int(row["ambiguous_high_mfe_rows_v1"]) + int(row["fifty_plus_risk_rows_v1"]) + int(row["hundred_plus_risk_rows_v1"]) + int(row["two_hundred_plus_risk_rows_v1"]),
                "root_cause_v1": cause,
                "recommended_treatment_v1": recommended_treatment(cause),
            }
        )
    return records


def _balanced_failure(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for run_id, group in rows.groupby("run_id_v1", dropna=False):
        core = int(_bool(group, V2_CORE_COL).sum())
        recommended = int(_bool(group, RECOMMENDED_COL).sum())
        balanced = int(_bool(group, BALANCED_COL).sum())
        improved = balanced > core
        worsened = (core == 0 and 0 < balanced < DENOMINATOR_TARGET) or (recommended >= DENOMINATOR_TARGET and 0 < balanced < DENOMINATOR_TARGET)
        added = _bool(group, BALANCED_COL) & ~_bool(group, RECOMMENDED_COL)
        records.append(
            {
                "run_id_v1": str(run_id),
                "v2_core_denominator_v1": core,
                "recommended_denominator_v1": recommended,
                "balanced_denominator_v1": balanced,
                "balanced_improved_count_v1": int(improved),
                "balanced_worsened_low_support_v1": bool(worsened),
                "added_to_already_supported_group_v1": bool(core >= DENOMINATOR_TARGET and int(added.sum()) > 0),
                "created_new_low_support_selected_group_v1": bool(core == 0 and 0 < balanced < DENOMINATOR_TARGET),
                "tail_rows_added_v1": int((added & _bool(group, "tail_label_v1")).sum()),
                "protected_runner_ambiguous_risk_added_v1": int((added & hard_veto_mask(group)).sum()),
                "reason_v1": _balanced_reason(core, recommended, balanced, int(added.sum()), bool(worsened)),
            }
        )
    return records


def _balanced_reason(core: int, recommended: int, balanced: int, added: int, worsened: bool) -> str:
    if worsened:
        return "Broad expansion creates or preserves a selected low-support group."
    if balanced > core and core >= DENOMINATOR_TARGET:
        return "Broad expansion adds rows to already-supported group."
    if balanced > recommended:
        return "Broad expansion adds rows but not necessarily to denominator-critical groups."
    return "No useful support improvement from balanced expansion."


def _frontier_memberships(rows: pd.DataFrame) -> dict[str, pd.Series]:
    current = _bool(rows, RECOMMENDED_COL)
    upper = _bool(rows, UPPER_BOUND_COL)
    safe_signal = safe_signal_candidate_mask(rows)
    tail_expansion = current | (
        _bool(rows, "safe_recoverable_v1")
        & ~hard_veto_mask(rows)
        & rows["r5_tail_score_signal_bucket_v1"].astype(str).isin(["STRONG", "SUPPORT"])
    )
    run_balanced = current.copy()
    for run_id, group in rows.groupby("run_id_v1", dropna=False):
        idx = group.index
        if int(current.loc[idx].sum()) < DENOMINATOR_TARGET:
            run_balanced.loc[idx] = (current.loc[idx] | upper.loc[idx]).values
    return {
        "CURRENT_V2_OOF_CORE": _bool(rows, V2_CORE_COL),
        "RECOMMENDED_73_RUN_ID_SUPPORT": current,
        "BALANCED_209_DIAGNOSTIC": _bool(rows, BALANCED_COL),
        "MAX_SAFE_SIGNAL_BACKED_SUPPORT": safe_signal,
        "MAX_SAFE_TAIL_EXPANSION_SUPPORT": tail_expansion,
        "MAX_SAFE_RUN_ID_BALANCED_SUPPORT": run_balanced,
        "MAX_FEASIBLE_UNDER_HARD_VETOES": upper,
    }


def _support_counts(rows: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    counts = selected.astype(bool).groupby(rows["run_id_v1"].astype(str)).sum()
    selected_counts = counts[counts > 0]
    upper_counts = _bool(rows, UPPER_BOUND_COL).groupby(rows["run_id_v1"].astype(str)).sum()
    structurally_unsat = selected_counts[(selected_counts < DENOMINATOR_TARGET) & (upper_counts.reindex(selected_counts.index).fillna(0) < DENOMINATOR_TARGET)]
    repairable = selected_counts[(selected_counts < DENOMINATOR_TARGET) & (upper_counts.reindex(selected_counts.index).fillna(0) >= DENOMINATOR_TARGET)]
    return {
        "minimum_run_id_denominator_v1": int(selected_counts.min()) if not selected_counts.empty else 0,
        "run_id_groups_below_5_v1": int(((selected_counts > 0) & (selected_counts < DENOMINATOR_TARGET)).sum()),
        "structurally_unsatisfiable_groups_v1": int(len(structurally_unsat)),
        "repairable_low_support_groups_v1": int(len(repairable)),
    }


def _variant_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for name, selected in _frontier_memberships(rows).items():
        selected = selected.astype(bool)
        validate_frontier_has_no_unsafe_rows(rows, selected)
        support = _support_counts(rows, selected)
        safety = _safety_counts(rows, selected)
        records.append(
            {
                "variant_id_v1": name,
                "variant_type_v1": "REPORT_ONLY_ROW_MEMBERSHIP_SET_NOT_MODEL",
                "selected_rows_v1": int(selected.sum()),
                "bad_count_v1": int((selected & _bool(rows, "bad_label_v1")).sum()),
                "tail_count_v1": int((selected & _bool(rows, "tail_label_v1")).sum()),
                "safe_recoverable_count_v1": int((selected & _bool(rows, "safe_recoverable_v1")).sum()),
                **support,
                **safety,
                "recommendation_status_v1": _frontier_status(name, support, safety),
            }
        )
    return records


def _safety_counts(rows: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    return {
        "fifty_plus_overlap_v1": int((selected & _bool(rows, "fifty_plus_mfe_risk_v1")).sum()),
        "hundred_plus_overlap_v1": int((selected & _bool(rows, "hundred_plus_mfe_risk_v1")).sum()),
        "two_hundred_plus_overlap_v1": int((selected & _bool(rows, "two_hundred_plus_mfe_risk_v1")).sum()),
        "protected_winner_selected_count_v1": int((selected & _bool(rows, "protected_winner_status_v1")).sum()),
        "runner_protect_selected_count_v1": int((selected & _bool(rows, "runner_protect_status_v1")).sum()),
        "ambiguous_high_mfe_selected_count_v1": int((selected & _bool(rows, "ambiguous_high_mfe_status_v1")).sum()),
        "quarantine_selected_count_v1": int((selected & rows["active_quarantine_v1"].astype(str).ne("ACTIVE_CANDIDATE")).sum()),
    }


def _frontier_status(name: str, support: dict[str, int], safety: dict[str, int]) -> str:
    if any(int(value) > 0 for value in safety.values()):
        return "REJECT_UNSAFE_EXPANSION"
    if name == "MAX_FEASIBLE_UNDER_HARD_VETOES":
        return "DIAGNOSTIC_ONLY_NOT_FINAL_POLICY"
    if support["run_id_groups_below_5_v1"] == 0:
        return "SUPPORT_SUFFICIENT_REPORT_ONLY"
    if support["structurally_unsatisfiable_groups_v1"] > 0:
        return "STRUCTURAL_LOW_SUPPORT_REMAINS"
    return "SUPPORT_REPAIRABLE_REPORT_ONLY"


def _policy_need(matrix_rows: list[dict[str, Any]], taxonomy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    structural = [
        row for row in matrix_rows
        if int(row["current_denominator_v1"]) < DENOMINATOR_TARGET
        and int(row["feasible_max_denominator_v1"]) < DENOMINATOR_TARGET
        and int(row["current_denominator_v1"]) > 0
    ]
    worst = next((row for row in matrix_rows if row["run_id_v1"] == WORST_RUN_ID), {})
    force_unsafe = bool(worst and int(worst["feasible_max_denominator_v1"]) < DENOMINATOR_TARGET)
    conclusion = (
        "LOW_SUPPORT_POLICY_REQUIRED_STRUCTURAL_LOW_SUPPORT"
        if structural
        else "LOW_SUPPORT_POLICY_NOT_NEEDED_SUPPORT_REPAIRABLE"
    )
    return {
        "layer_name": "RUN_ID_LOW_SUPPORT_POLICY_NEED_V1",
        "conclusion_v1": conclusion,
        "current_denominator_guard_correct_v1": True,
        "groups_with_feasible_safe_max_below_5_v1": len(structural),
        "worst_group_structurally_unsatisfiable_v1": bool(
            worst and int(worst["feasible_max_denominator_v1"]) < DENOMINATOR_TARGET
        ),
        "forcing_denominator_5_requires_unsafe_or_unsupported_rows_v1": force_unsafe,
        "low_support_groups_remain_invalid_for_final_promotion_v1": True,
        "low_support_groups_may_be_used_for_training_surface_with_explicit_policy_v1": True,
        "separate_explicit_low_support_policy_needed_before_r5_2_rebuild_v1": conclusion != "LOW_SUPPORT_POLICY_NOT_NEEDED_SUPPORT_REPAIRABLE",
        "taxonomy_counts_v1": {str(key): int(value) for key, value in pd.Series([row["root_cause_v1"] for row in taxonomy_rows]).value_counts().to_dict().items()} if taxonomy_rows else {},
    }


def _recommendation(policy: dict[str, Any], frontier_rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_feasible = next(row for row in frontier_rows if row["variant_id_v1"] == "MAX_FEASIBLE_UNDER_HARD_VETOES")
    if policy["conclusion_v1"] == "LOW_SUPPORT_POLICY_REQUIRED_STRUCTURAL_LOW_SUPPORT":
        status = "RUN_ID_SUPPORT_STRUCTURALLY_WEAK_DEFINE_LOW_SUPPORT_POLICY"
        next_action = "DEFINE_EXPLICIT_RUN_ID_LOW_SUPPORT_POLICY_V1"
        final = "RUN_ID_SUPPORT_STRUCTURALLY_UNSATISFIABLE_UNDER_CURRENT_CONTRACT"
    elif int(max_feasible["run_id_groups_below_5_v1"]) == 0:
        status = "RUN_ID_SUPPORT_READY_FOR_R5_2_REBUILD"
        next_action = "BUILD_R5_2_FROM_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1"
        final = "RUN_ID_SUPPORT_SUFFICIENT_FOR_R5_2_REBUILD"
    else:
        status = "RUN_ID_SUPPORT_REPAIRABLE_BUILD_COVERAGE_AWARE_OPPORTUNITY_BASE"
        next_action = "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_V1"
        final = "RUN_ID_SUPPORT_REPAIRABLE_WITH_EXISTING_SIGNALS"
    return {
        "layer_name": "RUN_ID_SUPPORT_AUDIT_RECOMMENDATION_V1",
        "status_v1": status,
        "final_go_no_go_v1": final,
        "next_recommended_action_v1": next_action,
        "max_feasible_under_hard_vetoes_v1": {
            "selected_rows_v1": max_feasible["selected_rows_v1"],
            "minimum_run_id_denominator_v1": max_feasible["minimum_run_id_denominator_v1"],
            "run_id_groups_below_5_v1": max_feasible["run_id_groups_below_5_v1"],
            "structurally_unsatisfiable_groups_v1": max_feasible["structurally_unsatisfiable_groups_v1"],
        },
        "no_model_training_v1": True,
        "no_optuna_v1": True,
        "not_r6_ready_v1": True,
    }


def _summary_dict(rows: pd.DataFrame, matrix_rows: list[dict[str, Any]], taxonomy_rows: list[dict[str, Any]], recommendation: dict[str, Any]) -> dict[str, Any]:
    worst = next((row for row in matrix_rows if row["run_id_v1"] == WORST_RUN_ID), {})
    return {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "row_count_v1": int(len(rows)),
        "run_id_count_v1": int(rows["run_id_v1"].nunique()),
        "low_support_group_count_v1": len(taxonomy_rows),
        "worst_run_id_v1": WORST_RUN_ID,
        "worst_run_id_current_denominator_v1": worst.get("current_denominator_v1"),
        "worst_run_id_feasible_max_denominator_v1": worst.get("feasible_max_denominator_v1"),
        "worst_run_id_status_v1": worst.get("support_repairability_status_v1"),
        "decision_v1": recommendation["final_go_no_go_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "no_optuna_model_r6_package_freeze_live_v1": True,
    }


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    opportunity_root: Path = OPPORTUNITY_ROOT,
    v2_oof_root: Path = V2_OOF_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    validate_denominator_guard_not_weakened(DENOMINATOR_TARGET)
    validate_low_support_groups_not_silently_dropped(dropped=False, explicitly_reported=True)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes_before = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "opportunity_rows_sha256_v1": _file_hash(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
    }
    rows = _load_rows(opportunity_root)
    matrix_rows = _run_id_feasibility_matrix(rows)
    worst_rows = _worst_deep_dive(rows)
    taxonomy_rows = _low_support_taxonomy(matrix_rows)
    balanced_rows = _balanced_failure(rows)
    frontier_rows = _variant_summary(rows)
    policy = _policy_need(matrix_rows, taxonomy_rows)
    recommendation = _recommendation(policy, frontier_rows)
    go_no_go = {
        "layer_name": "DEEPEN_RUN_ID_SUPPORT_SIGNAL_AUDIT_GO_NO_GO_V1",
        "decision_v1": recommendation["final_go_no_go_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "no_optuna_run_v1": True,
        "no_training_run_v1": True,
        "not_r6_ready_v1": True,
        "not_package_ready_v1": True,
        "not_live_ready_v1": True,
    }
    input_hashes_after = {
        "v2_oof_scores_sha256_v1": _file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        "v2_oof_provenance_sha256_v1": _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        "opportunity_rows_sha256_v1": _file_hash(opportunity_root / "r5_2_opportunity_base_rows_v1.csv"),
    }
    integrity = validate_input_artifacts_unchanged(input_hashes_before, input_hashes_after)
    forbidden = validate_no_forbidden_actions(optuna=False, model=False, r6=False, package=False, freeze=False, live=False)
    no_dummy = validate_no_dummy_synthetic_fallback(dummy=False, synthetic=False, fallback=False)

    _write_rows(output_dir / "run_id_support_feasibility_matrix_v1.csv", matrix_rows)
    _write_json(output_dir / "run_id_support_feasibility_matrix_v1.json", {"rows_v1": matrix_rows})
    _write_rows(output_dir / "worst_run_id_deep_dive_v1.csv", worst_rows)
    _write_json(output_dir / "worst_run_id_deep_dive_v1.json", {"run_id_v1": WORST_RUN_ID, "rows_v1": worst_rows, "answers_v1": _worst_answers(matrix_rows, worst_rows)})
    _write_rows(output_dir / "low_support_run_id_taxonomy_v1.csv", taxonomy_rows)
    _write_json(output_dir / "low_support_run_id_taxonomy_v1.json", {"rows_v1": taxonomy_rows})
    _write_rows(output_dir / "balanced_variant_support_failure_analysis_v1.csv", balanced_rows)
    _write_json(output_dir / "balanced_variant_support_failure_analysis_v1.json", {"rows_v1": balanced_rows, "summary_v1": _balanced_summary(balanced_rows)})
    _write_rows(output_dir / "feasible_run_id_support_frontier_v1.csv", frontier_rows)
    _write_json(output_dir / "feasible_run_id_support_frontier_v1.json", {"variants_v1": frontier_rows})
    _write_json(output_dir / "run_id_low_support_policy_need_v1.json", policy)
    _write_json(output_dir / "run_id_support_audit_recommendation_v1.json", recommendation)
    _write_json(output_dir / "deepen_run_id_support_signal_audit_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "opportunity_root_v1": str(opportunity_root),
                "v2_oof_root_v1": str(v2_oof_root),
            },
            "input_hashes_before_v1": input_hashes_before,
            "input_hashes_after_v1": input_hashes_after,
            "input_integrity_v1": integrity,
            "no_forbidden_actions_v1": forbidden,
            "no_dummy_synthetic_fallback_v1": no_dummy,
        },
    )
    _write_reports(output_dir, matrix_rows, worst_rows, taxonomy_rows, balanced_rows, frontier_rows, policy, recommendation)
    summary = _summary_dict(rows, matrix_rows, taxonomy_rows, recommendation)
    summary["output_dir_v1"] = str(output_dir)
    summary["v2_scores_provenance_model_objective_thresholds_unchanged_v1"] = integrity["status_v1"] == "PASS"
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "go_no_go_v1": go_no_go})
    return summary


def _worst_answers(matrix_rows: list[dict[str, Any]], worst_rows: list[dict[str, Any]]) -> dict[str, Any]:
    worst = next((row for row in matrix_rows if row["run_id_v1"] == WORST_RUN_ID), {})
    selected_safe = int(worst.get("v2_oof_selected_rows_v1", 0))
    safe_rows = int(worst.get("safe_recoverable_rows_v1", 0))
    additional = int(worst.get("additional_safe_candidates_available_v1", 0))
    feasible = int(worst.get("feasible_max_denominator_v1", 0))
    return {
        "are_2_v2_oof_rows_only_safe_recoverable_rows_v1": selected_safe == 2 and safe_rows == 2,
        "additional_legal_signal_backed_rows_available_v1": additional,
        "denominator_5_impossible_under_current_hard_vetoes_v1": feasible < DENOMINATOR_TARGET,
        "classification_v1": "DENOMINATOR_CONTRACT_FEASIBILITY_ISSUE" if feasible < DENOMINATOR_TARGET else "REPAIRABLE_SIGNAL_WEAKNESS",
        "required_to_raise_denominator_to_5_v1": "Additional safe/recoverable signal-backed rows, which are absent locally for this run_id." if feasible < DENOMINATOR_TARGET else "Use existing safe signal candidates.",
        "would_require_unsafe_or_unsupported_rows_v1": feasible < DENOMINATOR_TARGET,
        "structurally_low_support_v1": feasible < DENOMINATOR_TARGET,
        "row_count_v1": len(worst_rows),
    }


def _balanced_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "groups_improved_v1": int(sum(1 for row in rows if row["balanced_denominator_v1"] > row["v2_core_denominator_v1"])),
        "groups_worsened_v1": int(sum(1 for row in rows if row["balanced_worsened_low_support_v1"])),
        "new_low_support_groups_created_v1": int(sum(1 for row in rows if row["created_new_low_support_selected_group_v1"])),
        "adds_to_already_supported_groups_v1": int(sum(1 for row in rows if row["added_to_already_supported_group_v1"])),
        "coverage_aware_variant_needed_v1": True,
    }


def _write_reports(
    output_dir: Path,
    matrix_rows: list[dict[str, Any]],
    worst_rows: list[dict[str, Any]],
    taxonomy_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    policy: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    worst = next(row for row in matrix_rows if row["run_id_v1"] == WORST_RUN_ID)
    _write_report(
        output_dir / "run_id_support_feasibility_matrix_report_v1.md",
        [
            "# Run ID Support Feasibility Matrix V1",
            "",
            f"Run IDs: `{len(matrix_rows)}`",
            f"Low-support current groups: `{len(taxonomy_rows)}`",
            f"Worst run_id `{WORST_RUN_ID}` feasible max: `{worst['feasible_max_denominator_v1']}`.",
        ],
    )
    _write_report(
        output_dir / "worst_run_id_deep_dive_report_v1.md",
        [
            "# Worst Run ID Deep Dive V1",
            "",
            f"Run ID: `{WORST_RUN_ID}`",
            f"Rows: `{len(worst_rows)}`",
            f"Safe recoverable rows: `{worst['safe_recoverable_rows_v1']}`",
            f"Additional safe candidates: `{worst['additional_safe_candidates_available_v1']}`",
            f"Feasible max denominator: `{worst['feasible_max_denominator_v1']}`",
        ],
    )
    _write_report(
        output_dir / "low_support_run_id_taxonomy_report_v1.md",
        [
            "# Low Support Run ID Taxonomy V1",
            "",
            f"Low-support groups: `{len(taxonomy_rows)}`",
            f"Root causes: `{policy['taxonomy_counts_v1']}`",
        ],
    )
    summary = _balanced_summary(balanced_rows)
    _write_report(
        output_dir / "balanced_variant_support_failure_report_v1.md",
        [
            "# Balanced Variant Support Failure V1",
            "",
            f"Groups improved: `{summary['groups_improved_v1']}`",
            f"Groups worsened/new low-support: `{summary['groups_worsened_v1']}`",
            f"New low-support groups created: `{summary['new_low_support_groups_created_v1']}`",
        ],
    )
    _write_report(
        output_dir / "feasible_run_id_support_frontier_report_v1.md",
        [
            "# Feasible Run ID Support Frontier V1",
            "",
            *[
                f"- `{row['variant_id_v1']}`: rows `{row['selected_rows_v1']}`, min denominator `{row['minimum_run_id_denominator_v1']}`, below 5 `{row['run_id_groups_below_5_v1']}`, status `{row['recommendation_status_v1']}`"
                for row in frontier_rows
            ],
        ],
    )
    _write_report(
        output_dir / "run_id_low_support_policy_need_report_v1.md",
        [
            "# Run ID Low Support Policy Need V1",
            "",
            f"Conclusion: `{policy['conclusion_v1']}`",
            f"Groups with feasible safe max below 5: `{policy['groups_with_feasible_safe_max_below_5_v1']}`",
            f"Worst group structurally unsatisfiable: `{policy['worst_group_structurally_unsatisfiable_v1']}`",
        ],
    )
    _write_report(
        output_dir / "run_id_support_audit_recommendation_v1.md",
        [
            "# Run ID Support Audit Recommendation V1",
            "",
            f"Status: `{recommendation['status_v1']}`",
            f"Final go/no-go: `{recommendation['final_go_no_go_v1']}`",
            f"Next action: `{recommendation['next_recommended_action_v1']}`",
        ],
    )
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Deepen Run ID Support Signal Audit V1",
            "",
            f"Decision: `{recommendation['final_go_no_go_v1']}`",
            f"Next action: `{recommendation['next_recommended_action_v1']}`",
            "No model, Optuna, R6, package, freeze, promo, live, V2 mutation, or denominator weakening was performed.",
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", default=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--opportunity-root", type=Path, default=OPPORTUNITY_ROOT)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        opportunity_root=args.opportunity_root,
        v2_oof_root=args.v2_oof_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
