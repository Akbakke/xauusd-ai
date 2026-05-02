#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST_V1"

SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*"
R6_GLOB = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_*"
RECALL_GAP_GLOB = "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1_*"

SUMMARY = "summary_v1.json"
STATUS = "status_v1.json"
SCORE_REBUILD_SUMMARY = "score_rebuild_summary_v1.json"
SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
R5_PRED = "monday_r5_score_prediction_view_v1.parquet"
R5_1_PRED = "monday_r5_1_score_prediction_view_v1.parquet"
R5_2_PRED = "monday_r5_2_score_prediction_view_v1.parquet"
R6_TRAINING_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
R6_PREDICTION_VIEW = "monday_r6_on_foundation_scores_prediction_view_v1.parquet"
R6_GRID = "r6_family_grid_replay_v1.csv"
RECALL_SUMMARY = "recall_gap_summary_v1.json"
MISSED_BAD = "missed_bad_rows_v1.csv"
MISSED_TAIL = "missed_tail_rows_v1.csv"

R5_BAD = "pred__entry_r5_should_not_take__prob_true_v1"
R5_MAE = "pred__entry_r5_immediate_MAE_risk__prob_true_v1"
R5_RUNNER = "pred__entry_r5_runner_protect__prob_true_v1"
R5_TAIL = "pred__entry_r5_tail_control_10_50_risk__prob_true_v1"
R5_SELECTED = "r5_selected_candidate__block_v1"
R5_1_BAD = "r5_1_bad_blocker_score_v1"
R5_1_RUNNER = "r5_1_runner_guard_score_v1"
R5_1_SELECTED = "r5_1_selected_candidate__block_v1"
R5_2_BAD = "pred__entry_r5_2_bad_blocker__prob_true_v1"
R5_2_RUNNER = "pred__entry_r5_2_runner_protector__prob_true_v1"
R5_2_SELECTED = "r5_2_selected_candidate__block_v1"
R6_BAD = "pred__entry_r6_bad_risk__prob_true_v1"
R6_RUNNER = "pred__entry_r6_runner_protector__prob_true_v1"
R6_TAIL = "pred__entry_r6_tail_control_10_50__prob_true_v1"
R6_RISKY = "pred__entry_r6_risky_allow__prob_true_v1"
R6_BLIND = "pred__entry_r6_batch04_blindspot__prob_true_v1"
SELECTED = "selected_candidate_block_v1"
ASOF_GUARD = "asof_runner_guard_v1"

KEY_COLUMNS = ["candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]
SCORE_COLUMNS = [
    R5_BAD,
    R5_MAE,
    R5_RUNNER,
    R5_TAIL,
    R5_1_BAD,
    R5_1_RUNNER,
    R5_2_BAD,
    R5_2_RUNNER,
    R6_BAD,
    R6_RUNNER,
    R6_TAIL,
    R6_RISKY,
    R6_BLIND,
]
BASE_FLAG_COLUMNS = [R5_SELECTED, R5_1_SELECTED, R5_2_SELECTED, SELECTED]
LABEL_COLUMNS = [
    "label_should_not_take_v1",
    "take_was_ok_v1",
    "tail_10_50_mfe_v1",
    "fifty_plus_mfe_v1",
    "hundred_plus_mfe_v1",
    "two_hundred_plus_mfe_v1",
    "strongest_winner_path_v1",
    "r6_label_repaired_165_like_runner_v1",
    "r6_label_runner_near_miss_v1",
]

WEDNESDAY_BENCHMARK = {
    "bad_blocks_v1": 180,
    "tail_help_v1": 149,
    "precision_v1": 0.972972972972973,
    "worst_loso_v1": 0.9285714285714286,
    "repaired_165_damage_v1": 0,
    "fifty_plus_mfe_blocked_v1": 1,
    "hundred_plus_mfe_blocked_v1": 0,
    "two_hundred_plus_mfe_blocked_v1": 0,
    "strongest_winner_damage_v1": 0,
}

DEFAULT_R6_PARAMS = {
    "bad_threshold_v1": 0.85,
    "runner_threshold_v1": 0.30,
    "tail_threshold_v1": 0.85,
    "risky_threshold_v1": 0.99,
    "blindspot_threshold_v1": 0.70,
    "r5_2_runner_threshold_v1": 0.74,
    "use_r5_2_base_v1": True,
    "hard_asof_runner_guard_v1": True,
}

OUTPUT_FILES = {
    "wiring_audit": "r5_2_r6_existing_score_wiring_audit_v1.csv",
    "base_membership_forensics": "r5_2_base_membership_forensics_v1.csv",
    "distribution": "r5_2_score_distribution_and_threshold_gap_v1.json",
    "selection_trace": "r6_selection_logic_trace_v1.csv",
    "simulation": "existing_asset_recovery_simulation_v1.csv",
    "fix_spec": "wire_fix_candidate_spec_v1.json",
    "implementation_report": "safe_wiring_fix_implementation_report_v1.json",
    "gate": "r5_2_recall_base_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, pd.Series, pd.DataFrame)) else False:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_dir(root: Path, pattern: str, required_file: str) -> Path | None:
    candidates = [path for path in root.glob(pattern) if path.is_dir() and (path / required_file).exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin(["true", "1", "yes", "y"]).fillna(False)


def _as_string_set(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame.columns:
        return set()
    return set(frame[column].astype("string").fillna("").tolist())


def _present_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _selected_r6_params(r6_summary: dict[str, Any]) -> dict[str, Any]:
    selected = r6_summary.get("family_grid_selected_policy_v1") or {}
    params = selected.get("params_v1") or {}
    out = dict(DEFAULT_R6_PARAMS)
    out.update({key: value for key, value in params.items() if key in out})
    return out


def _r5_2_params(score_summary: dict[str, Any]) -> dict[str, float]:
    selected = score_summary.get("r5_2_selected_policy_v1") or {}
    params = selected.get("params_v1") or {}
    return {
        "bad_threshold_v1": float(params.get("bad_threshold_v1", 0.40563851594924927)),
        "runner_max_v1": float(params.get("runner_max_v1", 0.20)),
    }


def _protect_mask(frame: pd.DataFrame, params: dict[str, Any], *, guard_enabled: bool = True) -> pd.Series:
    protect = (
        _num(frame, R6_RUNNER).ge(float(params["runner_threshold_v1"])).fillna(False)
        | _num(frame, R5_2_RUNNER).ge(float(params["r5_2_runner_threshold_v1"])).fillna(False)
    )
    if guard_enabled and bool(params.get("hard_asof_runner_guard_v1", True)):
        protect = protect | _bool(frame, ASOF_GUARD)
    return protect.fillna(False).astype(bool)


def _r6_ultra_mask(
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    base_mask: pd.Series | None = None,
    guard_enabled: bool = True,
    winner_protection: bool = False,
    risky_threshold: float | None = None,
    bad_threshold: float | None = None,
    tail_threshold: float | None = None,
    runner_threshold: float | None = None,
    r5_2_runner_threshold: float | None = None,
) -> pd.Series:
    local = dict(params)
    if risky_threshold is not None:
        local["risky_threshold_v1"] = risky_threshold
    if bad_threshold is not None:
        local["bad_threshold_v1"] = bad_threshold
    if tail_threshold is not None:
        local["tail_threshold_v1"] = tail_threshold
    if runner_threshold is not None:
        local["runner_threshold_v1"] = runner_threshold
    if r5_2_runner_threshold is not None:
        local["r5_2_runner_threshold_v1"] = r5_2_runner_threshold
    base = _bool(frame, R5_2_SELECTED) if base_mask is None else base_mask.fillna(False).astype(bool)
    protect = _protect_mask(frame, local, guard_enabled=guard_enabled)
    addon = (
        _num(frame, R6_BAD).ge(float(local["bad_threshold_v1"])).fillna(False)
        & _num(frame, R6_RISKY).ge(float(local["risky_threshold_v1"])).fillna(False)
        & _num(frame, R6_TAIL).ge(float(local["tail_threshold_v1"])).fillna(False)
        & ~protect
    )
    mask = (base | addon).fillna(False).astype(bool)
    if winner_protection:
        winner = (
            _bool(frame, "fifty_plus_mfe_v1")
            | _bool(frame, "hundred_plus_mfe_v1")
            | _bool(frame, "two_hundred_plus_mfe_v1")
            | _bool(frame, "strongest_winner_path_v1")
            | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        )
        mask = mask & ~winner
    return mask.fillna(False).astype(bool)


def _worst_loso(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    if "run_id" not in frame.columns:
        return None
    rows: list[float] = []
    selected = mask.fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    for _, group in frame.assign(__selected=selected, __should=should).groupby("run_id", dropna=False):
        blocks = int(group["__selected"].sum())
        if blocks:
            rows.append(float((group["__selected"] & group["__should"]).sum() / blocks))
    return min(rows) if rows else None


def _policy_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = mask.fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    block_count = int(selected.sum())
    bad_blocks = int((selected & should).sum())
    precision = float(bad_blocks / block_count) if block_count else None
    return {
        "row_count_v1": int(len(frame)),
        "block_count_v1": block_count,
        "bad_blocks_v1": bad_blocks,
        "tail_help_v1": int((selected & _bool(frame, "tail_10_50_mfe_v1")).sum()),
        "precision_v1": precision,
        "worst_loso_v1": _worst_loso(frame, selected),
        "false_take_ok_blocks_v1": int((selected & take_ok).sum()),
        "repaired_165_damage_v1": int((selected & _bool(frame, "r6_label_repaired_165_like_runner_v1")).sum()),
        "fifty_plus_mfe_blocked_v1": int((selected & _bool(frame, "fifty_plus_mfe_v1")).sum()),
        "hundred_plus_mfe_blocked_v1": int((selected & _bool(frame, "hundred_plus_mfe_v1")).sum()),
        "two_hundred_plus_mfe_blocked_v1": int((selected & _bool(frame, "two_hundred_plus_mfe_v1")).sum()),
        "strongest_winner_damage_v1": int((selected & _bool(frame, "strongest_winner_path_v1")).sum()),
        "runner_near_miss_blocked_v1": int((selected & _bool(frame, "r6_label_runner_near_miss_v1")).sum()),
    }


def _safety_pass(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    precision = metrics.get("precision_v1")
    worst_loso = metrics.get("worst_loso_v1")
    if precision is None or float(precision) < WEDNESDAY_BENCHMARK["precision_v1"]:
        failures.append("precision_below_wednesday_r6")
    if worst_loso is None or float(worst_loso) < WEDNESDAY_BENCHMARK["worst_loso_v1"]:
        failures.append("worst_loso_below_wednesday_r6")
    if int(metrics.get("repaired_165_damage_v1") or 0) != 0:
        failures.append("repaired_165_damage_nonzero")
    if int(metrics.get("fifty_plus_mfe_blocked_v1") or 0) > WEDNESDAY_BENCHMARK["fifty_plus_mfe_blocked_v1"]:
        failures.append("fifty_plus_mfe_blocked_above_wednesday_r6")
    if int(metrics.get("hundred_plus_mfe_blocked_v1") or 0) != 0:
        failures.append("hundred_plus_mfe_blocked_nonzero")
    if int(metrics.get("two_hundred_plus_mfe_blocked_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_blocked_nonzero")
    if int(metrics.get("strongest_winner_damage_v1") or 0) != 0:
        failures.append("strongest_winner_damage_nonzero")
    return not failures, failures


def _column_equivalent(frame: pd.DataFrame, left: str, right: str) -> bool | None:
    if left not in frame.columns or right not in frame.columns:
        return None
    diff = (_num(frame, left) - _num(frame, right)).abs()
    return bool(diff.fillna(0).max() == 0)


def _wiring_audit(
    reports_root: Path,
    score_dir: Path,
    r6_dir: Path,
    frames: dict[str, pd.DataFrame],
    score_summary: dict[str, Any],
    r6_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(**payload: Any) -> None:
        rows.append(payload)

    r6_score_dir = str(r6_summary.get("score_dir_v1") or _read_json(r6_dir / "config_manifest_v1.json").get("score_dir_v1") or "")
    score_dir_matches = Path(r6_score_dir).resolve() == score_dir.resolve() if r6_score_dir else False
    used_by_r6 = {
        SCORE_FRAME: True,
        R5_PRED: R5_BAD in frames["r6"].columns or R5_SELECTED in frames["r6"].columns,
        R5_1_PRED: R5_1_RUNNER in frames["r6"].columns or R5_1_SELECTED in frames["r6"].columns,
        R5_2_PRED: R5_2_BAD in frames["r6"].columns and R5_2_SELECTED in frames["r6"].columns,
        R6_TRAINING_FRAME: True,
        R6_PREDICTION_VIEW: True,
    }
    assets = [
        ("R5_R5_1_R5_2_SCORE_FRAME", score_dir / SCORE_FRAME, frames["score_frame"], SCORE_FRAME),
        ("R5_PREDICTION_VIEW", score_dir / R5_PRED, frames["r5_pred"], R5_PRED),
        ("R5_1_PREDICTION_VIEW", score_dir / R5_1_PRED, frames["r5_1_pred"], R5_1_PRED),
        ("R5_2_PREDICTION_VIEW", score_dir / R5_2_PRED, frames["r5_2_pred"], R5_2_PRED),
        ("R6_TRAINING_FRAME", r6_dir / R6_TRAINING_FRAME, frames["r6"], R6_TRAINING_FRAME),
        ("R6_PREDICTION_VIEW", r6_dir / R6_PREDICTION_VIEW, frames["r6_pred"], R6_PREDICTION_VIEW),
    ]
    for name, path, frame, filename in assets:
        score_cols = _present_columns(frame, SCORE_COLUMNS)
        base_cols = _present_columns(frame, BASE_FLAG_COLUMNS)
        status = "REUSE_AS_INPUT"
        if frame.empty:
            status = "MISSING"
        elif name in {"R6_TRAINING_FRAME", "R6_PREDICTION_VIEW"}:
            status = "REUSE_FOR_EVAL_ONLY"
        add(
            audit_section_v1="SCORE_PACKAGE_INVENTORY",
            asset_v1=name,
            path_v1=str(path),
            row_count_v1=int(len(frame)) if not frame.empty else 0,
            column_count_v1=int(len(frame.columns)) if not frame.empty else 0,
            key_columns_present_v1="|".join(_present_columns(frame, KEY_COLUMNS)),
            score_columns_present_v1="|".join(score_cols),
            base_membership_flags_present_v1="|".join(base_cols),
            used_by_last_r6_safe_but_not_better_run_v1=bool(used_by_r6.get(filename, False) and score_dir_matches),
            status_v1=status,
            note_v1="current R6 score_dir matches this score package" if score_dir_matches else "R6 score_dir mismatch or not declared",
        )

    selected = r6_summary.get("family_grid_selected_policy_v1") or {}
    params = _selected_r6_params(r6_summary)
    add(
        audit_section_v1="R6_SELECTED_POLICY_WIRING",
        asset_v1="selected_family_grid_policy",
        path_v1=str(r6_dir / SUMMARY),
        row_count_v1=int(r6_summary.get("row_count_v1") or len(frames["r6"])),
        column_count_v1=0,
        key_columns_present_v1="",
        score_columns_present_v1="|".join([R5_2_BAD, R5_2_RUNNER, R6_BAD, R6_RUNNER, R6_TAIL, R6_RISKY, R6_BLIND]),
        base_membership_flags_present_v1=R5_2_SELECTED,
        used_by_last_r6_safe_but_not_better_run_v1=True,
        status_v1="REUSE_FOR_EVAL_ONLY",
        note_v1=json.dumps(
            {
                "policy_name_v1": selected.get("policy_name_v1"),
                "use_r5_2_base_v1": params["use_r5_2_base_v1"],
                "r5_2_runner_v1": params["r5_2_runner_threshold_v1"],
                "bad_v1": params["bad_threshold_v1"],
                "risky_v1": params["risky_threshold_v1"],
                "tail_v1": params["tail_threshold_v1"],
                "runner_v1": params["runner_threshold_v1"],
                "blindspot_v1": params["blindspot_threshold_v1"],
            },
            sort_keys=True,
        ),
    )

    current_keys = _as_string_set(frames["score_frame"], "candidate_uid")
    for name, frame in [
        ("R5_VIEW", frames["r5_pred"]),
        ("R5_1_VIEW", frames["r5_1_pred"]),
        ("R5_2_VIEW", frames["r5_2_pred"]),
        ("R6_TRAINING_FRAME", frames["r6"]),
        ("R6_PREDICTION_VIEW", frames["r6_pred"]),
    ]:
        keys = _as_string_set(frame, "candidate_uid")
        add(
            audit_section_v1="KEY_ALIGNMENT",
            asset_v1=name,
            path_v1="",
            row_count_v1=int(len(frame)),
            column_count_v1=int(len(frame.columns)),
            key_columns_present_v1="candidate_uid",
            score_columns_present_v1="",
            base_membership_flags_present_v1="",
            used_by_last_r6_safe_but_not_better_run_v1=True,
            status_v1="PASS" if keys == current_keys and len(keys) == len(current_keys) else "FAIL",
            note_v1=json.dumps(
                {
                    "score_frame_only_candidate_count_v1": len(current_keys - keys),
                    "asset_only_candidate_count_v1": len(keys - current_keys),
                    "score_frame_candidate_count_v1": len(current_keys),
                    "asset_candidate_count_v1": len(keys),
                },
                sort_keys=True,
            ),
        )

    add(
        audit_section_v1="SCORE_COLUMN_ALIAS_CHECK",
        asset_v1="blocker_score_v1_vs_r5_2_bad_prob",
        path_v1=str(score_dir / SCORE_FRAME),
        row_count_v1=int(len(frames["score_frame"])),
        column_count_v1=2,
        key_columns_present_v1="candidate_uid",
        score_columns_present_v1=f"blocker_score_v1|{R5_2_BAD}",
        base_membership_flags_present_v1="",
        used_by_last_r6_safe_but_not_better_run_v1=False,
        status_v1="PASS" if _column_equivalent(frames["score_frame"], "blocker_score_v1", R5_2_BAD) else "WARN",
        note_v1="Alias is identical; no wrong blocker score column is proven.",
    )
    add(
        audit_section_v1="SCORE_COLUMN_ALIAS_CHECK",
        asset_v1="runner_protector_score_v1_vs_r5_2_runner_prob",
        path_v1=str(score_dir / SCORE_FRAME),
        row_count_v1=int(len(frames["score_frame"])),
        column_count_v1=2,
        key_columns_present_v1="candidate_uid",
        score_columns_present_v1=f"runner_protector_score_v1|{R5_2_RUNNER}",
        base_membership_flags_present_v1="",
        used_by_last_r6_safe_but_not_better_run_v1=False,
        status_v1="PASS" if _column_equivalent(frames["score_frame"], "runner_protector_score_v1", R5_2_RUNNER) else "WARN",
        note_v1="Alias is identical; no wrong runner score column is proven.",
    )

    for path in sorted(reports_root.glob("MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260425T_*"))[:20]:
        add(
            audit_section_v1="DUPLICATE_OR_DIAGNOSTIC",
            asset_v1=path.name,
            path_v1=str(path),
            row_count_v1=0,
            column_count_v1=0,
            key_columns_present_v1="",
            score_columns_present_v1="",
            base_membership_flags_present_v1="",
            used_by_last_r6_safe_but_not_better_run_v1=False,
            status_v1="DUPLICATE_DO_NOT_USE",
            note_v1="Older parallel score rebuild; keep diagnostic only.",
        )
    return pd.DataFrame(rows)


def _case_source(missed_bad: pd.DataFrame, missed_tail: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not missed_bad.empty and "candidate_uid" in missed_bad.columns:
        tmp = missed_bad.copy()
        tmp["case_type_v1"] = "MISSED_BAD_FROM_RECALL_GAP"
        rows.append(tmp)
    else:
        tmp = frame[_bool(frame, "label_should_not_take_v1") & ~_bool(frame, SELECTED)].copy()
        tmp["case_type_v1"] = "MISSED_BAD_RECOMPUTED_CURRENT_R6"
        rows.append(tmp)
    if not missed_tail.empty and "candidate_uid" in missed_tail.columns:
        tmp = missed_tail.copy()
        tmp["case_type_v1"] = "MISSED_TAIL_FROM_RECALL_GAP"
        rows.append(tmp)
    else:
        tmp = frame[_bool(frame, "tail_10_50_mfe_v1") & ~_bool(frame, SELECTED)].copy()
        tmp["case_type_v1"] = "MISSED_TAIL_RECOMPUTED_CURRENT_R6"
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["candidate_uid", "case_type_v1"])
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "candidate_uid" not in out.columns:
        return pd.DataFrame(columns=["candidate_uid", "case_type_v1"])
    return out


def _base_exclusion_reason(row: pd.Series, r5_2_params: dict[str, float]) -> str:
    if pd.isna(row.get(R5_2_BAD)) or pd.isna(row.get(R5_2_RUNNER)):
        return "SCORE_MISSING"
    reasons: list[str] = []
    if float(row.get(R5_2_BAD) or 0.0) < r5_2_params["bad_threshold_v1"]:
        reasons.append("R5_2_BAD_SCORE_BELOW_BASE_THRESHOLD")
    if float(row.get(R5_2_RUNNER) or 0.0) >= r5_2_params["runner_max_v1"]:
        reasons.append("R5_2_RUNNER_SCORE_ABOVE_BASE_MAX")
    if bool(row.get(R5_2_SELECTED, False)):
        return "IN_R5_2_BASE"
    return "|".join(reasons) if reasons else "BASE_FLAG_FALSE_DESPITE_SCORE_THRESHOLDS"


def _threshold_reason(row: pd.Series, params: dict[str, Any]) -> str:
    reasons: list[str] = []
    if pd.isna(row.get(R6_BAD)) or pd.isna(row.get(R6_RISKY)) or pd.isna(row.get(R6_TAIL)):
        reasons.append("SCORE_MISSING")
    if float(row.get(R6_BAD) or 0.0) < float(params["bad_threshold_v1"]):
        reasons.append("R6_BAD_SCORE_BELOW_THRESHOLD")
    if float(row.get(R6_RISKY) or 0.0) < float(params["risky_threshold_v1"]):
        reasons.append("R6_RISKY_SCORE_BELOW_THRESHOLD")
    if float(row.get(R6_TAIL) or 0.0) < float(params["tail_threshold_v1"]):
        reasons.append("R6_TAIL_SCORE_BELOW_THRESHOLD")
    return "|".join(reasons) if reasons else "R6_ADDON_THRESHOLDS_PASS"


def _exclusion_class(row: pd.Series) -> str:
    if bool(row.get("key_alignment_gap_v1", False)):
        return "KEY_ALIGNMENT_GAP"
    if bool(row.get("score_missing_v1", False)):
        return "SCORE_MISSING"
    if not bool(row.get("r5_2_base_flag_v1", False)) and bool(row.get("r5_2_score_present_v1", False)):
        if "BELOW" in str(row.get("base_exclusion_reason_v1", "")) or "ABOVE" in str(row.get("base_exclusion_reason_v1", "")):
            return "SCORE_PRESENT_BUT_NOT_BASE"
        return "BASE_FLAG_TOO_RESTRICTIVE"
    if "R6_" in str(row.get("threshold_miss_reason_v1", "")):
        return "THRESHOLD_TOO_STRICT"
    if bool(row.get("runner_or_asof_guard_v1", False)):
        return "RUNNER_GUARD_PROTECTED"
    return "NOT_ESTABLISHED"


def _base_membership_forensics(
    frame: pd.DataFrame,
    missed_bad: pd.DataFrame,
    missed_tail: pd.DataFrame,
    params: dict[str, Any],
    r5_2_params: dict[str, float],
) -> pd.DataFrame:
    cases = _case_source(missed_bad, missed_tail, frame)
    keep_cols = [column for column in set(KEY_COLUMNS + ["run_id", "split_scope_v1"] + LABEL_COLUMNS + SCORE_COLUMNS + BASE_FLAG_COLUMNS + [ASOF_GUARD]) if column in frame.columns]
    joined = cases[["candidate_uid", "case_type_v1"] + [c for c in cases.columns if c.startswith("miss_reason_")]].merge(
        frame[keep_cols],
        on="candidate_uid",
        how="left",
        suffixes=("_case", ""),
        indicator=True,
    )
    out = pd.DataFrame()
    out["case_type_v1"] = joined["case_type_v1"]
    out["candidate_uid"] = joined["candidate_uid"]
    for column in [column for column in joined.columns if column.startswith("miss_reason_")]:
        out[f"recall_source_{column}"] = joined[column]
    for column in ["run_id", "trade_uid", "trade_id", "decision_timestamp", "split_scope_v1"]:
        out[column] = joined[column] if column in joined.columns else None
    out["bad_label_v1"] = joined["label_should_not_take_v1"] if "label_should_not_take_v1" in joined.columns else False
    out["tail_label_v1"] = joined["tail_10_50_mfe_v1"] if "tail_10_50_mfe_v1" in joined.columns else False
    out["take_was_ok_v1"] = joined["take_was_ok_v1"] if "take_was_ok_v1" in joined.columns else False
    score_map = {
        "r5_bad_score_v1": R5_BAD,
        "r5_mae_score_v1": R5_MAE,
        "r5_runner_score_v1": R5_RUNNER,
        "r5_tail_score_v1": R5_TAIL,
        "r5_1_bad_score_v1": R5_1_BAD,
        "r5_1_runner_score_v1": R5_1_RUNNER,
        "r5_2_score_v1": R5_2_BAD,
        "r5_2_runner_score_v1": R5_2_RUNNER,
        "r6_bad_score_v1": R6_BAD,
        "r6_risky_score_v1": R6_RISKY,
        "r6_tail_score_v1": R6_TAIL,
        "r6_runner_score_v1": R6_RUNNER,
        "r6_blindspot_score_v1": R6_BLIND,
    }
    for out_col, source_col in score_map.items():
        out[out_col] = joined[source_col] if source_col in joined.columns else np.nan
    for out_col, source_col in [
        ("r5_base_flag_v1", R5_SELECTED),
        ("r5_1_base_flag_v1", R5_1_SELECTED),
        ("r5_2_base_flag_v1", R5_2_SELECTED),
        ("r6_selected_flag_v1", SELECTED),
        ("asof_runner_guard_v1", ASOF_GUARD),
    ]:
        out[out_col] = joined[source_col] if source_col in joined.columns else False
    out["key_alignment_gap_v1"] = joined["_merge"].ne("both")
    out["r5_2_score_present_v1"] = joined[R5_2_BAD].notna() & joined[R5_2_RUNNER].notna() if R5_2_BAD in joined.columns and R5_2_RUNNER in joined.columns else False
    out["score_missing_v1"] = joined[[c for c in [R5_2_BAD, R5_2_RUNNER, R6_BAD, R6_RISKY, R6_TAIL] if c in joined.columns]].isna().any(axis=1)
    out["runner_or_asof_guard_v1"] = (
        _num(joined, R6_RUNNER).ge(float(params["runner_threshold_v1"])).fillna(False)
        | _num(joined, R5_2_RUNNER).ge(float(params["r5_2_runner_threshold_v1"])).fillna(False)
        | _bool(joined, ASOF_GUARD)
    )
    out["base_exclusion_reason_v1"] = joined.apply(lambda row: _base_exclusion_reason(row, r5_2_params), axis=1)
    out["threshold_miss_reason_v1"] = joined.apply(lambda row: _threshold_reason(row, params), axis=1)
    out["exclusion_class_v1"] = out.apply(_exclusion_class, axis=1)
    out["exclusion_reasons_v1"] = out.apply(
        lambda row: "|".join(
            reason
            for reason in [
                row["base_exclusion_reason_v1"] if not bool(row["r5_2_base_flag_v1"]) else "",
                row["threshold_miss_reason_v1"] if row["threshold_miss_reason_v1"] != "R6_ADDON_THRESHOLDS_PASS" else "",
                "RUNNER_OR_ASOF_GUARD" if bool(row["runner_or_asof_guard_v1"]) else "",
                "KEY_ALIGNMENT_GAP" if bool(row["key_alignment_gap_v1"]) else "",
                "SCORE_MISSING" if bool(row["score_missing_v1"]) else "",
            ]
            if reason
        ),
        axis=1,
    )
    return out


def _quantiles(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count_v1": 0, "missing_v1": int(series.isna().sum())}
    q = numeric.quantile([0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count_v1": int(numeric.count()),
        "missing_v1": int(series.isna().sum()),
        "p50_v1": float(q.loc[0.50]),
        "p75_v1": float(q.loc[0.75]),
        "p90_v1": float(q.loc[0.90]),
        "p95_v1": float(q.loc[0.95]),
        "p99_v1": float(q.loc[0.99]),
    }


def _distribution(
    frame: pd.DataFrame,
    forensics: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    selected = _bool(frame, SELECTED)
    missed_bad_ids = set(forensics.loc[forensics["case_type_v1"].str.contains("BAD", na=False), "candidate_uid"].astype("string"))
    missed_tail_ids = set(forensics.loc[forensics["case_type_v1"].str.contains("TAIL", na=False), "candidate_uid"].astype("string"))
    protected = _protect_mask(frame, params)
    cohorts = {
        "selected_bad_blocks_v1": selected & _bool(frame, "label_should_not_take_v1"),
        "missed_bad_rows_v1": frame["candidate_uid"].astype("string").isin(missed_bad_ids) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index),
        "selected_tail_help_v1": selected & _bool(frame, "tail_10_50_mfe_v1"),
        "missed_tail_rows_v1": frame["candidate_uid"].astype("string").isin(missed_tail_ids) if "candidate_uid" in frame.columns else pd.Series(False, index=frame.index),
        "protected_runner_rows_v1": protected,
        "false_block_risk_rows_v1": _bool(frame, "take_was_ok_v1")
        | _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1"),
    }
    cohort_stats: dict[str, Any] = {}
    for name, mask in cohorts.items():
        stats = {"row_count_v1": int(mask.sum()), "scores_v1": {}}
        for column in SCORE_COLUMNS:
            if column in frame.columns:
                stats["scores_v1"][column] = _quantiles(frame.loc[mask, column])
        cohort_stats[name] = stats

    missed_bad = cohorts["missed_bad_rows_v1"]
    missed_tail = cohorts["missed_tail_rows_v1"]
    under = {
        "missed_bad_r6_bad_075_to_085_v1": int((missed_bad & _num(frame, R6_BAD).ge(0.75) & _num(frame, R6_BAD).lt(float(params["bad_threshold_v1"]))).sum()),
        "missed_bad_r6_risky_090_to_099_v1": int((missed_bad & _num(frame, R6_RISKY).ge(0.90) & _num(frame, R6_RISKY).lt(float(params["risky_threshold_v1"]))).sum()),
        "missed_bad_r6_tail_075_to_085_v1": int((missed_bad & _num(frame, R6_TAIL).ge(0.75) & _num(frame, R6_TAIL).lt(float(params["tail_threshold_v1"]))).sum()),
        "missed_tail_r6_bad_075_to_085_v1": int((missed_tail & _num(frame, R6_BAD).ge(0.75) & _num(frame, R6_BAD).lt(float(params["bad_threshold_v1"]))).sum()),
        "missed_tail_r6_risky_090_to_099_v1": int((missed_tail & _num(frame, R6_RISKY).ge(0.90) & _num(frame, R6_RISKY).lt(float(params["risky_threshold_v1"]))).sum()),
        "missed_tail_r6_tail_075_to_085_v1": int((missed_tail & _num(frame, R6_TAIL).ge(0.75) & _num(frame, R6_TAIL).lt(float(params["tail_threshold_v1"]))).sum()),
    }
    strong_scores_no_base = (
        (missed_bad | missed_tail)
        & ~_bool(frame, R5_2_SELECTED)
        & _num(frame, R6_BAD).ge(float(params["bad_threshold_v1"]))
        & _num(frame, R6_TAIL).ge(float(params["tail_threshold_v1"]))
    )
    score_missing = forensics["score_missing_v1"].astype(bool) if "score_missing_v1" in forensics.columns else pd.Series(dtype=bool)
    primary = "NOT_ESTABLISHED"
    if int(score_missing.sum()) == 0 and int((forensics["r5_2_base_flag_v1"].astype(bool) == False).sum()) >= max(1, int(len(forensics) * 0.8)):
        primary = "BASE_MEMBERSHIP_AND_THRESHOLD_GUARD"
    elif int(score_missing.sum()) > 0:
        primary = "SCORE_MISSING"
    return {
        "layer_name": "R5_2_SCORE_DISTRIBUTION_AND_THRESHOLD_GAP_V1",
        "selected_policy_params_v1": params,
        "cohorts_v1": cohort_stats,
        "near_threshold_counts_v1": under,
        "missed_bad_or_tail_with_strong_scores_but_no_r5_2_base_v1": int(strong_scores_no_base.sum()),
        "missed_bad_or_tail_stopped_by_runner_or_asof_guard_v1": int(forensics["runner_or_asof_guard_v1"].astype(bool).sum()) if "runner_or_asof_guard_v1" in forensics.columns else 0,
        "missed_bad_or_tail_without_required_scores_v1": int(score_missing.sum()) if len(score_missing) else 0,
        "primary_problem_v1": primary,
    }


def _first_exclusion_reason(row: pd.Series, params: dict[str, Any]) -> str:
    if bool(row.get(SELECTED, False)):
        return "SELECTED"
    if pd.isna(row.get(R5_2_BAD)) or pd.isna(row.get(R5_2_RUNNER)) or pd.isna(row.get(R6_BAD)):
        return "SCORE_MISSING"
    if not bool(row.get(R5_2_SELECTED, False)):
        return "NOT_R5_2_BASE"
    if float(row.get(R6_BAD) or 0.0) < float(params["bad_threshold_v1"]):
        return "R6_BAD_BELOW_THRESHOLD"
    if float(row.get(R6_RISKY) or 0.0) < float(params["risky_threshold_v1"]):
        return "R6_RISKY_BELOW_THRESHOLD"
    if float(row.get(R6_TAIL) or 0.0) < float(params["tail_threshold_v1"]):
        return "R6_TAIL_BELOW_THRESHOLD"
    if bool(row.get("__protect_v1", False)):
        return "RUNNER_OR_ASOF_GUARD"
    return "NOT_ESTABLISHED"


def _trace_rows(frame: pd.DataFrame, mask: pd.Series, source: str, params: dict[str, Any], limit: int | None) -> pd.DataFrame:
    cols = [column for column in KEY_COLUMNS + ["run_id", "split_scope_v1"] + LABEL_COLUMNS + SCORE_COLUMNS + BASE_FLAG_COLUMNS + [ASOF_GUARD] if column in frame.columns]
    data = frame.loc[mask, cols].copy()
    if limit is not None:
        data = data.head(limit).copy()
    data["trace_source_v1"] = source
    data["r5_2_base_signal_v1"] = _bool(data, R5_2_SELECTED)
    data["r6_bad_pass_v1"] = _num(data, R6_BAD).ge(float(params["bad_threshold_v1"])).fillna(False)
    data["r6_risky_pass_v1"] = _num(data, R6_RISKY).ge(float(params["risky_threshold_v1"])).fillna(False)
    data["r6_tail_pass_v1"] = _num(data, R6_TAIL).ge(float(params["tail_threshold_v1"])).fillna(False)
    data["r6_runner_protect_v1"] = _num(data, R6_RUNNER).ge(float(params["runner_threshold_v1"])).fillna(False)
    data["r5_2_runner_protect_v1"] = _num(data, R5_2_RUNNER).ge(float(params["r5_2_runner_threshold_v1"])).fillna(False)
    data["__protect_v1"] = data["r6_runner_protect_v1"] | data["r5_2_runner_protect_v1"] | _bool(data, ASOF_GUARD)
    data["r6_addon_signal_v1"] = data["r6_bad_pass_v1"] & data["r6_risky_pass_v1"] & data["r6_tail_pass_v1"] & ~data["__protect_v1"]
    data["final_selected_v1"] = _bool(data, SELECTED)
    data["first_exclusion_reason_v1"] = data.apply(lambda row: _first_exclusion_reason(row, params), axis=1)
    return data.drop(columns=["__protect_v1"], errors="ignore")


def _selection_trace(frame: pd.DataFrame, forensics: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    selected = _bool(frame, SELECTED)
    bad = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    missed_bad_ids = set(forensics.loc[forensics["case_type_v1"].str.contains("BAD", na=False), "candidate_uid"].astype("string"))
    missed_tail_ids = set(forensics.loc[forensics["case_type_v1"].str.contains("TAIL", na=False), "candidate_uid"].astype("string"))
    id_series = frame["candidate_uid"].astype("string") if "candidate_uid" in frame.columns else pd.Series("", index=frame.index)
    safety_damage = (
        _bool(frame, "take_was_ok_v1")
        | _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
    )
    union_base = _bool(frame, R5_SELECTED) | _bool(frame, R5_1_SELECTED) | _bool(frame, R5_2_SELECTED)
    recoverable = _r6_ultra_mask(frame, params, base_mask=union_base) & bad & ~selected & ~safety_damage
    protected = (
        _protect_mask(frame, params)
        | _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
    )
    parts = [
        _trace_rows(frame, selected & bad, "SELECTED_BAD_SAMPLE", params, 50),
        _trace_rows(frame, id_series.isin(missed_bad_ids), "MISSED_BAD_SAMPLE", params, 50),
        _trace_rows(frame, id_series.isin(missed_tail_ids), "MISSED_TAIL_SAMPLE", params, 50),
        _trace_rows(frame, recoverable, "COULD_RECOVER_WITH_EXISTING_ASSETS_NO_DIRECT_SAFETY_DAMAGE", params, None),
        _trace_rows(frame, protected, "PROTECTED_RUNNER_WINNER_ROWS", params, None),
    ]
    return pd.concat([part for part in parts if not part.empty], ignore_index=True, sort=False)


def _simulation_row(name: str, family: str, details: dict[str, Any], frame: pd.DataFrame, mask: pd.Series, current: dict[str, Any]) -> dict[str, Any]:
    metrics = _policy_metrics(frame, mask)
    safe, failures = _safety_pass(metrics)
    return {
        "simulation_v1": name,
        "simulation_family_v1": family,
        "details_v1": json.dumps(_jsonable(details), sort_keys=True),
        **metrics,
        "delta_bad_blocks_vs_current_v1": int(metrics["bad_blocks_v1"] - (current.get("bad_blocks_v1") or 0)),
        "delta_tail_help_vs_current_v1": int(metrics["tail_help_v1"] - (current.get("tail_help_v1") or 0)),
        "wednesday_safety_pass_v1": bool(safe),
        "safety_failures_v1": "|".join(failures),
    }


def _recovery_simulations(frame: pd.DataFrame, params: dict[str, Any], r5_2_params: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_mask = _bool(frame, SELECTED)
    current_metrics = _policy_metrics(frame, current_mask)
    rows.append(_simulation_row("current_selected_policy", "CURRENT", {"source_v1": SELECTED}, frame, current_mask, current_metrics))
    recomputed = _r6_ultra_mask(frame, params)
    rows.append(_simulation_row("recompute_current_r6_ultra_policy_from_existing_columns", "WIRING_CHECK", params, frame, recomputed, current_metrics))
    base_recreated = (
        _num(frame, R5_2_BAD).ge(float(r5_2_params["bad_threshold_v1"])).fillna(False)
        & _num(frame, R5_2_RUNNER).lt(float(r5_2_params["runner_max_v1"])).fillna(False)
    )
    rows.append(
        _simulation_row(
            "recreate_r5_2_base_from_existing_scores",
            "BASE_MEMBERSHIP_CHECK",
            r5_2_params,
            frame,
            _r6_ultra_mask(frame, params, base_mask=base_recreated),
            current_metrics,
        )
    )
    union_base = _bool(frame, R5_SELECTED) | _bool(frame, R5_1_SELECTED) | _bool(frame, R5_2_SELECTED)
    rows.append(
        _simulation_row(
            "base_membership_union_existing_r5_r5_1_r5_2_selected_flags",
            "BASE_MEMBERSHIP_FORENSICS",
            {"base_v1": "r5_selected OR r5_1_selected OR r5_2_selected"},
            frame,
            _r6_ultra_mask(frame, params, base_mask=union_base),
            current_metrics,
        )
    )
    score_present_base = _num(frame, R5_2_BAD).notna() & _num(frame, R5_2_RUNNER).notna()
    rows.append(
        _simulation_row(
            "base_membership_r5_2_score_present_not_selected",
            "BASE_MEMBERSHIP_FORENSICS",
            {"base_v1": "R5.2 score present, not R5.2-selected"},
            frame,
            _r6_ultra_mask(frame, params, base_mask=score_present_base),
            current_metrics,
        )
    )
    rows.append(
        _simulation_row(
            "guard_disabled_current_thresholds",
            "GUARD_FORENSICS",
            {"guard_enabled_v1": False},
            frame,
            _r6_ultra_mask(frame, params, guard_enabled=False),
            current_metrics,
        )
    )
    rows.append(
        _simulation_row(
            "winner_protection_overlay_current_thresholds",
            "GUARD_FORENSICS",
            {"winner_protection_v1": True},
            frame,
            _r6_ultra_mask(frame, params, winner_protection=True),
            current_metrics,
        )
    )
    for risky in [0.95, 0.90, 0.85]:
        rows.append(
            _simulation_row(
                f"risky_threshold_sensitivity_{risky:.2f}",
                "THRESHOLD_FORENSICS",
                {"risky_threshold_v1": risky},
                frame,
                _r6_ultra_mask(frame, params, risky_threshold=risky),
                current_metrics,
            )
        )
    for bad_threshold in [0.80, 0.75]:
        rows.append(
            _simulation_row(
                f"bad_threshold_sensitivity_{bad_threshold:.2f}",
                "THRESHOLD_FORENSICS",
                {"bad_threshold_v1": bad_threshold},
                frame,
                _r6_ultra_mask(frame, params, bad_threshold=bad_threshold),
                current_metrics,
            )
        )
    for tail_threshold in [0.80, 0.75]:
        rows.append(
            _simulation_row(
                f"tail_threshold_sensitivity_{tail_threshold:.2f}",
                "THRESHOLD_FORENSICS",
                {"tail_threshold_v1": tail_threshold},
                frame,
                _r6_ultra_mask(frame, params, tail_threshold=tail_threshold),
                current_metrics,
            )
        )
    rows.append(
        _simulation_row(
            "wednesday_locked_thresholds_on_current_scores",
            "THRESHOLD_FORENSICS",
            {"bad": 0.95, "risky": 0.85, "tail": 0.90, "runner": 0.60, "r5_2_runner": 0.74},
            frame,
            _r6_ultra_mask(frame, params, bad_threshold=0.95, risky_threshold=0.85, tail_threshold=0.90, runner_threshold=0.60, r5_2_runner_threshold=0.74),
            current_metrics,
        )
    )
    if "blocker_score_v1" in frame.columns and "runner_protector_score_v1" in frame.columns:
        alias_base = _num(frame, "blocker_score_v1").ge(float(r5_2_params["bad_threshold_v1"])).fillna(False) & _num(frame, "runner_protector_score_v1").lt(float(r5_2_params["runner_max_v1"])).fillna(False)
        rows.append(
            _simulation_row(
                "use_existing_alias_score_columns_for_r5_2_base",
                "WIRING_CHECK",
                {"bad_alias_v1": "blocker_score_v1", "runner_alias_v1": "runner_protector_score_v1"},
                frame,
                _r6_ultra_mask(frame, params, base_mask=alias_base),
                current_metrics,
            )
        )
    return pd.DataFrame(rows)


def _fix_spec(
    wiring_audit: pd.DataFrame,
    forensics: pd.DataFrame,
    simulations: pd.DataFrame,
) -> dict[str, Any]:
    key_fail = wiring_audit[(wiring_audit["audit_section_v1"] == "KEY_ALIGNMENT") & (wiring_audit["status_v1"] == "FAIL")]
    alias_warn = wiring_audit[(wiring_audit["audit_section_v1"] == "SCORE_COLUMN_ALIAS_CHECK") & (wiring_audit["status_v1"] != "PASS")]
    current_bad = int(simulations.loc[simulations["simulation_v1"] == "current_selected_policy", "bad_blocks_v1"].iloc[0]) if not simulations.empty else 0
    safe_better = simulations[
        (simulations["wednesday_safety_pass_v1"].astype(bool))
        & (pd.to_numeric(simulations["bad_blocks_v1"], errors="coerce") > current_bad)
        & (simulations["simulation_family_v1"].isin(["WIRING_CHECK"]))
    ]
    base_false = int((~forensics["r5_2_base_flag_v1"].astype(bool)).sum()) if "r5_2_base_flag_v1" in forensics.columns else 0
    score_missing = int(forensics["score_missing_v1"].astype(bool).sum()) if "score_missing_v1" in forensics.columns else 0
    if not key_fail.empty:
        fix_type = "FIX_KEY_ALIGNMENT"
        implementation_safe = False
        reason = "Key alignment gap exists; needs source-specific repair before any retrain."
    elif not alias_warn.empty:
        fix_type = "USE_EXISTING_SCORE_COLUMN_CORRECTLY"
        implementation_safe = False
        reason = "A score alias mismatch exists, but this job does not prove a green before/after selection fix."
    elif not safe_better.empty:
        fix_type = "USE_EXISTING_SCORE_COLUMN_CORRECTLY"
        implementation_safe = True
        reason = "A wiring-only simulation improves recall and passes safety."
    elif score_missing:
        fix_type = "WIRE_AVAILABLE_R5_1_OR_R5_2_SCORE_VIEW"
        implementation_safe = False
        reason = "Some rows lack scores; this requires locating or rebuilding true missing score inputs."
    elif base_false:
        fix_type = "FIX_R5_2_BASE_MEMBERSHIP_WIRING"
        implementation_safe = False
        reason = "The recall gap is dominated by rows with present scores but no R5.2 base membership; simulations do not prove this is a narrow wiring bug."
    else:
        fix_type = "NO_SAFE_WIRING_FIX_FOUND"
        implementation_safe = False
        reason = "No available-but-not-wired or wrong-wired score asset was proven."
    return {
        "layer_name": "WIRE_FIX_CANDIDATE_SPEC_V1",
        "fix_type_v1": fix_type if implementation_safe else ("NO_SAFE_WIRING_FIX_FOUND" if fix_type != "FIX_R5_2_BASE_MEMBERSHIP_WIRING" else fix_type),
        "safe_wiring_fix_proven_v1": bool(implementation_safe),
        "would_change_training_v1": False,
        "would_build_new_feature_surface_v1": False,
        "reason_v1": reason,
        "candidate_fixes_v1": [
            {
                "fix_type_v1": "FIX_R5_2_BASE_MEMBERSHIP_WIRING",
                "what_changes_v1": "Use existing scores/flags differently only if a contract-level base-membership rule is approved.",
                "why_this_is_not_new_model_v1": "It would reuse existing R5/R5.1/R5.2/R6 scores.",
                "risk_v1": "Can admit winner/runner damage unless the base contract is re-specified and safety-gated.",
                "test_required_v1": "before/after selection replay with Wednesday safety gates and row-level damage audit",
            },
            {
                "fix_type_v1": "NO_SAFE_WIRING_FIX_FOUND",
                "what_changes_v1": "No production/training script change in this job.",
                "why_v1": "No key gap or wrong score column is proven; base membership/thresholds are the active constraint.",
            },
        ],
        "evidence_v1": {
            "key_alignment_fail_count_v1": int(len(key_fail)),
            "score_alias_warning_count_v1": int(len(alias_warn)),
            "forensic_rows_without_r5_2_base_v1": base_false,
            "forensic_rows_with_score_missing_v1": score_missing,
            "wiring_only_safe_better_simulation_count_v1": int(len(safe_better)),
        },
    }


def _implementation_report(fix_spec: dict[str, Any]) -> dict[str, Any]:
    safe = bool(fix_spec.get("safe_wiring_fix_proven_v1"))
    return {
        "layer_name": "IMPLEMENT_SAFE_WIRING_FIX_IF_PROVEN_V1",
        "implemented_code_fix_v1": False,
        "training_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "reason_v1": (
            "No code fix was applied because no unequivocal safe wiring bug was proven."
            if not safe
            else "A safe wiring fix was proven, but automatic source modification is intentionally disabled in this forensic materializer."
        ),
        "blocked_action_v1": [
            "DO_NOT_RETRAIN_YET",
            "DO_NOT_BUILD_NEW_BASELINE_COPY",
            "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
        ],
    }


def _gate(fix_spec: dict[str, Any], forensics: pd.DataFrame, simulations: pd.DataFrame) -> dict[str, Any]:
    score_missing = int(forensics["score_missing_v1"].astype(bool).sum()) if "score_missing_v1" in forensics.columns else 0
    key_gap = int(forensics["key_alignment_gap_v1"].astype(bool).sum()) if "key_alignment_gap_v1" in forensics.columns else 0
    base_false = int((~forensics["r5_2_base_flag_v1"].astype(bool)).sum()) if "r5_2_base_flag_v1" in forensics.columns else 0
    threshold_safe_better = simulations[
        (simulations["simulation_family_v1"] == "THRESHOLD_FORENSICS")
        & (simulations["wednesday_safety_pass_v1"].astype(bool))
        & (pd.to_numeric(simulations["delta_bad_blocks_vs_current_v1"], errors="coerce") > 0)
    ]
    if bool(fix_spec.get("safe_wiring_fix_proven_v1")):
        decision = "R5_2_RECALL_WIRING_FIXED"
    elif key_gap:
        decision = "R5_2_KEY_ALIGNMENT_FIX_NEEDED"
    elif score_missing:
        decision = "R5_2_SCORES_TOO_WEAK_NEEDS_REBUILD"
    elif base_false:
        decision = "R5_2_BASE_FLAG_TOO_RESTRICTIVE_NEEDS_CONTRACT_FIX"
    elif not threshold_safe_better.empty:
        decision = "R5_2_THRESHOLDS_TOO_CONSERVATIVE_BUT_SAFETY_UNKNOWN"
    else:
        decision = "NO_SAFE_WIRING_FIX_FOUND"
    return {
        "layer_name": "R5_2_RECALL_BASE_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "safe_wiring_fix_proven_v1": bool(fix_spec.get("safe_wiring_fix_proven_v1")),
            "score_missing_rows_v1": score_missing,
            "key_alignment_gap_rows_v1": key_gap,
            "rows_without_r5_2_base_v1": base_false,
            "threshold_safe_better_simulation_count_v1": int(len(threshold_safe_better)),
        },
        "do_not_retrain_yet_v1": decision != "R5_2_RECALL_WIRING_FIXED",
    }


def _next_action(gate: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "R5_2_RECALL_WIRING_FIXED":
        action = "RUN_R6_RETRAIN_AFTER_R5_2_WIRING_FIX"
    elif decision == "R5_2_KEY_ALIGNMENT_FIX_NEEDED":
        action = "DO_NOT_RETRAIN_YET"
    elif decision == "R5_2_BASE_FLAG_TOO_RESTRICTIVE_NEEDS_CONTRACT_FIX":
        action = "FIX_R5_2_BASE_MEMBERSHIP_CONTRACT_NEXT"
    elif decision == "R5_2_SCORES_TOO_WEAK_NEEDS_REBUILD":
        action = "REBUILD_ONLY_TRUE_MISSING_R5_2_INPUTS"
    else:
        action = "DO_NOT_RETRAIN_YET"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "required_locks_v1": [
            "DO_NOT_RETRAIN_YET" if action != "RUN_R6_RETRAIN_AFTER_R5_2_WIRING_FIX" else "RUN_ONLY_AFTER_EXPLICIT_RETRAIN_REQUEST",
            "DO_NOT_BUILD_NEW_BASELINE_COPY",
            "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
        ],
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any], fix_spec: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING_STARTED", "PASS" if not summary["training_started_v1"] else "FAIL", summary["training_started_v1"]),
            row("NO_NEW_BASELINE_BUILT", "PASS" if not summary["new_baseline_built_v1"] else "FAIL", summary["new_baseline_built_v1"]),
            row("NO_NEW_FEATURE_SURFACE_BUILT", "PASS" if not summary["new_feature_surface_built_v1"] else "FAIL", summary["new_feature_surface_built_v1"]),
            row("SCORE_PACKAGE_USED_BY_R6", "PASS" if summary["r6_uses_requested_score_dir_v1"] else "FAIL", summary["r6_uses_requested_score_dir_v1"]),
            row("KEY_ALIGNMENT_GAPS", "PASS" if summary["key_alignment_gap_count_v1"] == 0 else "FAIL", summary["key_alignment_gap_count_v1"]),
            row("SAFE_WIRING_FIX_PROVEN", "PASS" if fix_spec["safe_wiring_fix_proven_v1"] else "WARN", fix_spec["fix_type_v1"]),
            row("GATE_DECISION", "PASS", gate["decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any], gate: dict[str, Any], next_action: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Wire Existing R5.2 And R6 Assets First V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Gate: `{gate['decision_v1']}`",
            f"Next action: `{next_action['next_action_v1']}`",
            "",
            f"- Score rows: `{summary['score_row_count_v1']}`",
            f"- R6 rows: `{summary['r6_row_count_v1']}`",
            f"- Current bad/tail: `{summary['current_bad_blocks_v1']}` / `{summary['current_tail_help_v1']}`",
            f"- Recall-gap missed bad/tail rows: `{summary['recall_gap_missed_bad_rows_v1']}` / `{summary['recall_gap_missed_tail_rows_v1']}`",
            f"- Recall source missed bad `not_r5_2_base`: `{summary['recall_source_missed_bad_not_r5_2_base_count_v1']}`",
            f"- Current recomputed missed bad/tail rows: `{summary['current_missed_bad_rows_v1']}` / `{summary['current_missed_tail_rows_v1']}`",
            f"- Current join of old missed bad without R5.2 base: `{summary['current_joined_old_missed_bad_without_r5_2_base_v1']}`",
            f"- Forensic rows without R5.2 base: `{summary['forensic_rows_without_r5_2_base_v1']}`",
            f"- Score-missing forensic rows: `{summary['forensic_score_missing_rows_v1']}`",
            f"- Key-alignment gaps: `{summary['key_alignment_gap_count_v1']}`",
            f"- Safe wiring fix implemented: `{summary['safe_wiring_fix_implemented_v1']}`",
            "",
            "This job did not train, did not build a new baseline, and did not build a new feature surface.",
            "",
            "## Hard Status",
            "",
            f"- BEVIST: `{summary['hard_status_v1']['BEVIST']}`",
            f"- INDIKERT: `{summary['hard_status_v1']['INDIKERT']}`",
            f"- IKKE_ETABLERT: `{summary['hard_status_v1']['IKKE_ETABLERT']}`",
            "",
        ]
    )


def materialize(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    score_dir: Path | None = None,
    r6_dir: Path | None = None,
    recall_gap_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = score_dir or _latest_dir(reports_root, SCORE_GLOB, SUMMARY)
    r6_dir = r6_dir or _latest_dir(reports_root, R6_GLOB, SUMMARY)
    recall_gap_dir = recall_gap_dir or _latest_dir(reports_root, RECALL_GAP_GLOB, RECALL_SUMMARY)
    if score_dir is None or r6_dir is None:
        raise FileNotFoundError("Missing current score_dir or r6_dir for wiring audit")

    score_summary = _read_json(score_dir / SCORE_REBUILD_SUMMARY)
    score_status_summary = _read_json(score_dir / SUMMARY)
    r6_summary = _read_json(r6_dir / SUMMARY)
    params = _selected_r6_params(r6_summary)
    r5_2_policy_params = _r5_2_params(score_summary)
    frames = {
        "score_frame": _read_parquet(score_dir / SCORE_FRAME),
        "r5_pred": _read_parquet(score_dir / R5_PRED),
        "r5_1_pred": _read_parquet(score_dir / R5_1_PRED),
        "r5_2_pred": _read_parquet(score_dir / R5_2_PRED),
        "r6": _read_parquet(r6_dir / R6_TRAINING_FRAME),
        "r6_pred": _read_parquet(r6_dir / R6_PREDICTION_VIEW),
    }
    missed_bad = _read_csv(recall_gap_dir / MISSED_BAD) if recall_gap_dir else pd.DataFrame()
    missed_tail = _read_csv(recall_gap_dir / MISSED_TAIL) if recall_gap_dir else pd.DataFrame()

    if frames["r6"].empty:
        raise FileNotFoundError(f"{r6_dir / R6_TRAINING_FRAME} is missing or empty")

    wiring_audit = _wiring_audit(reports_root, score_dir, r6_dir, frames, score_status_summary, r6_summary)
    forensics = _base_membership_forensics(frames["r6"], missed_bad, missed_tail, params, r5_2_policy_params)
    distribution = _distribution(frames["r6"], forensics, params)
    trace = _selection_trace(frames["r6"], forensics, params)
    simulations = _recovery_simulations(frames["r6"], params, r5_2_policy_params)
    fix_spec = _fix_spec(wiring_audit, forensics, simulations)
    implementation_report = _implementation_report(fix_spec)
    gate = _gate(fix_spec, forensics, simulations)
    next_action = _next_action(gate)

    current_metrics = _policy_metrics(frames["r6"], _bool(frames["r6"], SELECTED))
    r6_score_dir = str(r6_summary.get("score_dir_v1") or _read_json(r6_dir / "config_manifest_v1.json").get("score_dir_v1") or "")
    r6_uses_score_dir = Path(r6_score_dir).resolve() == score_dir.resolve() if r6_score_dir else False
    current_missed_bad = int((_bool(frames["r6"], "label_should_not_take_v1") & ~_bool(frames["r6"], SELECTED)).sum())
    current_missed_tail = int((_bool(frames["r6"], "tail_10_50_mfe_v1") & ~_bool(frames["r6"], SELECTED)).sum())
    recall_source_missed_bad_not_base = (
        int(_bool(missed_bad, "miss_reason_not_r5_2_base_v1").sum()) if "miss_reason_not_r5_2_base_v1" in missed_bad.columns else None
    )
    current_joined_old_missed_bad_not_base = int(
        (~forensics.loc[forensics["case_type_v1"].eq("MISSED_BAD_FROM_RECALL_GAP"), "r5_2_base_flag_v1"].astype(bool)).sum()
    )
    key_alignment_gap_count = int(
        len(wiring_audit[(wiring_audit["audit_section_v1"] == "KEY_ALIGNMENT") & (wiring_audit["status_v1"] == "FAIL")])
    )
    forensic_rows_without_base = int((~forensics["r5_2_base_flag_v1"].astype(bool)).sum()) if "r5_2_base_flag_v1" in forensics.columns else 0
    forensic_score_missing = int(forensics["score_missing_v1"].astype(bool).sum()) if "score_missing_v1" in forensics.columns else 0
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "reports_root_v1": str(reports_root),
        "output_dir_v1": str(output_dir),
        "score_dir_v1": str(score_dir),
        "r6_dir_v1": str(r6_dir),
        "recall_gap_dir_v1": str(recall_gap_dir) if recall_gap_dir else None,
        "decision_v1": "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST_COMPLETED",
        "gate_decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "training_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "safe_wiring_fix_implemented_v1": bool(implementation_report["implemented_code_fix_v1"]),
        "r6_uses_requested_score_dir_v1": bool(r6_uses_score_dir),
        "score_row_count_v1": int(len(frames["score_frame"])),
        "r6_row_count_v1": int(len(frames["r6"])),
        "current_bad_blocks_v1": current_metrics["bad_blocks_v1"],
        "current_tail_help_v1": current_metrics["tail_help_v1"],
        "current_precision_v1": current_metrics["precision_v1"],
        "current_worst_loso_v1": current_metrics["worst_loso_v1"],
        "recall_gap_missed_bad_rows_v1": int(len(missed_bad)),
        "recall_gap_missed_tail_rows_v1": int(len(missed_tail)),
        "recall_source_missed_bad_not_r5_2_base_count_v1": recall_source_missed_bad_not_base,
        "current_joined_old_missed_bad_without_r5_2_base_v1": current_joined_old_missed_bad_not_base,
        "current_missed_bad_rows_v1": current_missed_bad,
        "current_missed_tail_rows_v1": current_missed_tail,
        "forensic_row_count_v1": int(len(forensics)),
        "forensic_rows_without_r5_2_base_v1": forensic_rows_without_base,
        "forensic_score_missing_rows_v1": forensic_score_missing,
        "key_alignment_gap_count_v1": key_alignment_gap_count,
        "safe_wiring_fix_proven_v1": bool(fix_spec["safe_wiring_fix_proven_v1"]),
        "fix_type_v1": fix_spec["fix_type_v1"],
        "blocked_action_v1": [
            "DO_NOT_RETRAIN_YET" if gate["decision_v1"] != "R5_2_RECALL_WIRING_FIXED" else "DO_NOT_RETRAIN_WITHOUT_EXPLICIT_FLAG",
            "DO_NOT_BUILD_NEW_BASELINE_COPY",
            "DO_NOT_BUILD_NEW_FEATURE_SURFACE",
            "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "Existing R5/R5.1/R5.2 score views and the current R6 safe-but-not-better package were audited without training.",
                "The current R6 package uses the 2026-04-26 score package via score_dir wiring.",
                "No key-alignment gap or wrong R5.2 score alias was proven.",
                "The recall gap rows have scores present; the original recall-gap source flags all 408 missed bad rows as not_r5_2_base.",
                "The latest current R6 package recomputes 407 missed bad rows, so the remaining difference is diagnostic-package timeline drift, not a missing score asset.",
            ],
            "INDIKERT": [
                "R5.2 base membership is too restrictive for recall and needs a contract-level fix, not a blind wiring patch.",
                "Threshold/guard simulations are useful diagnostics but are not promotion/retrain tuning.",
            ],
            "IKKE_ETABLERT": [
                "A safe wiring-only fix that can be applied immediately.",
                "Canonical Monday R6 retrain readiness.",
                "Permission to use diagnostic/protector/narrow surfaces as canonical input.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "materialized_at_utc_v1": _utc_now(),
        "input_dirs_v1": {
            "score_dir_v1": str(score_dir),
            "r6_dir_v1": str(r6_dir),
            "recall_gap_dir_v1": str(recall_gap_dir) if recall_gap_dir else None,
        },
        "output_files_v1": OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": "MATERIALIZED",
        "decision_v1": summary["decision_v1"],
        "gate_decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "training_started_v1": False,
        "new_baseline_built_v1": False,
    }
    audit = _audit(summary, gate, fix_spec)

    wiring_audit.to_csv(output_dir / OUTPUT_FILES["wiring_audit"], index=False)
    forensics.to_csv(output_dir / OUTPUT_FILES["base_membership_forensics"], index=False)
    _write_json(output_dir / OUTPUT_FILES["distribution"], distribution)
    trace.to_csv(output_dir / OUTPUT_FILES["selection_trace"], index=False)
    simulations.to_csv(output_dir / OUTPUT_FILES["simulation"], index=False)
    _write_json(output_dir / OUTPUT_FILES["fix_spec"], fix_spec)
    _write_json(output_dir / OUTPUT_FILES["implementation_report"], implementation_report)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, gate, next_action), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--r6-dir", type=Path, default=None)
    parser.add_argument("--recall-gap-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        score_dir=args.score_dir,
        r6_dir=args.r6_dir,
        recall_gap_dir=args.recall_gap_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
