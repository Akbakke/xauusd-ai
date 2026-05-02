#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    _bool,
    _json_dumps,
    _load_json,
    _num,
    _policy_metric_row,
    _safe_rate,
    _write_json,
)
from gx1.scripts.materialize_truth_rl_recommendation_candidate_v1 import (
    RECOMMENDATION_AUDIT,
    RECOMMENDATION_SUMMARY,
    RECOMMENDATION_TRADE_VIEW,
)
from gx1.scripts.materialize_truth_rl_unified_observability_v1 import (
    UNIFIED_RL_AUDIT,
    UNIFIED_RL_EPISODE_VIEW,
    UNIFIED_RL_SUMMARY,
)
from gx1.scripts.train_r5_2_entry_runner_aware_retrain_and_loso_selection_v1 import (
    AS_OF_FEATURE_TABLE as R5_2_AS_OF_FEATURE_TABLE,
    BAD_PROB,
    CONTRACT as R5_2_CONTRACT,
    CONSISTENCY_AUDIT as R5_2_CONSISTENCY_AUDIT,
    HEAD_TO_HEAD as R5_2_HEAD_TO_HEAD,
    HINDSIGHT_LABEL_OUTCOME_TABLE as R5_2_HINDSIGHT_LABEL_OUTCOME_TABLE,
    LOSO_METRICS as R5_2_LOSO_METRICS,
    POLICY_PREDICTION_VIEW as R5_2_POLICY_PREDICTION_VIEW,
    RUNNER_PROB,
    SUMMARY as R5_2_SUMMARY,
)
from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import _slice_masks
from gx1.scripts.train_truth_harvest_retrain_candidate_v1 import (
    HARVEST_RETRAIN_EXTENSION_NAME,
    RETRAIN_AUDIT,
    RETRAIN_PREDICTION_VIEW,
    RETRAIN_SUMMARY,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_PHASE_GATE_AND_HARVEST_INTEGRATION_V1"
R5_2_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_ENTRY_RUNNER_AWARE_RETRAIN_AND_LOSO_SELECTION_V1"

CONTRACT = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_contract_v1.json"
AS_OF_TABLE = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_as_of_table_v1.parquet"
HINDSIGHT_TABLE = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_hindsight_outcome_table_v1.parquet"
SHADOW_REPLAY_BAKEOFF = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_shadow_replay_bakeoff_v1.csv"
ROBUSTNESS_STRESS_MATRIX = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_robustness_stress_matrix_v1.csv"
CALIBRATION_AUDIT = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_calibration_audit_v1.csv"
FAILURE_MODE_TABLE = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_failure_mode_table_v1.csv"
HARVEST_IMPACT = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_harvest_impact_v1.csv"
POLICY_LOGGING_EXPLAINABILITY = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_policy_logging_explainability_v1.parquet"
DECISION_MATRIX = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_decision_matrix_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r5_2_shadow_phase_gate_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r5_2_shadow_phase_gate_and_harvest_integration_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
POLICY_COLUMNS = {
    "NO_ENTRY_FALLBACK_BASELINE": "no_entry_fallback_baseline__block_v1",
    "R2_FALLBACK_REFERENCE": "r2_fallback_reference__block_v1",
    "R4_CURRENT_REFERENCE": "r4_current_reference__block_v1",
    "R5_CURRENT_REFERENCE": "r5_current_reference__block_v1",
    "R5_1_SAFETY_REFERENCE": "r5_1_selected_reference__block_v1",
    "R5_2_SELECTED_CANDIDATE": "r5_2_selected_candidate__block_v1",
}
SCORE_COLUMNS = [
    BAD_PROB,
    RUNNER_PROB,
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_strong_trade_candidate__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
    "pred__entry_r5_take_was_ok__prob_true_v1",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _optional_pass(frame: pd.DataFrame, column: str) -> bool | None:
    if frame.empty:
        return None
    if "row_count_v1" in frame.columns:
        row_count = pd.to_numeric(frame["row_count_v1"], errors="coerce")
        if row_count.notna().all() and bool(row_count.le(0).all()):
            return None
    if "run_count_v1" in frame.columns:
        run_count = pd.to_numeric(frame["run_count_v1"], errors="coerce")
        if run_count.notna().all() and bool(run_count.le(0).all()):
            return None
    return bool(frame[column].iloc[0])


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_dir(reports_root: Path, path_arg: str | None, default_name: str, required_file: str) -> Path:
    path = Path(path_arg).expanduser().resolve() if path_arg else reports_root / default_name
    if not path.exists():
        raise FileNotFoundError(f"Required dir does not exist: {path}")
    if not (path / required_file).exists():
        raise FileNotFoundError(f"{path} missing required artifact {required_file}")
    return path


def _resolve_dir_from_top_summary(
    *,
    reports_root: Path,
    path_arg: str | None,
    top_summary_name: str,
    fallback_contains: str,
    required_file: str,
) -> Path:
    if path_arg:
        return _resolve_dir(reports_root, path_arg, "", required_file)
    top_summary = reports_root / top_summary_name
    if top_summary.exists():
        extension_dir = _load_json(top_summary).get("extension_dir_v1")
        if isinstance(extension_dir, str) and extension_dir.strip():
            candidate = Path(extension_dir).expanduser().resolve()
            if (candidate / required_file).exists():
                return candidate
    candidates = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir() and fallback_contains in path.name and (path / required_file).exists()
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve {fallback_contains} with {required_file}")


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _run_sort_key(run_id: Any) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _all_run_ids(reports_root: Path, frame: pd.DataFrame) -> list[str]:
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_ids = sorted([path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.match(path.name)], key=_run_sort_key)
        if run_ids:
            return run_ids
    return sorted(frame["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)


def _load_inputs(
    *,
    r5_2_dir: Path,
    harvest_dir: Path,
    rl_recommendation_dir: Path | None,
    rl_unified_dir: Path | None,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any], pd.DataFrame | None, Dict[str, Any] | None, pd.DataFrame | None, Dict[str, Any] | None, pd.DataFrame, Dict[str, Any]]:
    asof_df = pd.read_parquet(r5_2_dir / R5_2_AS_OF_FEATURE_TABLE)
    hindsight_df = pd.read_parquet(r5_2_dir / R5_2_HINDSIGHT_LABEL_OUTCOME_TABLE)
    prediction_df = pd.read_parquet(r5_2_dir / R5_2_POLICY_PREDICTION_VIEW)
    loso_df = pd.read_csv(r5_2_dir / R5_2_LOSO_METRICS)
    head_to_head_df = pd.read_csv(r5_2_dir / R5_2_HEAD_TO_HEAD)
    summary = _load_json(r5_2_dir / R5_2_SUMMARY)
    r5_2_consistency_df = pd.read_csv(r5_2_dir / R5_2_CONSISTENCY_AUDIT)
    contract = _load_json(r5_2_dir / R5_2_CONTRACT)

    harvest_df = pd.read_parquet(harvest_dir / RETRAIN_PREDICTION_VIEW)
    harvest_summary = _load_json(harvest_dir / RETRAIN_SUMMARY)
    harvest_audit_df = pd.read_csv(harvest_dir / RETRAIN_AUDIT)

    rl_trade_df: pd.DataFrame | None = None
    rl_summary: Dict[str, Any] | None = None
    if rl_recommendation_dir is not None:
        rl_trade_df = pd.read_parquet(rl_recommendation_dir / RECOMMENDATION_TRADE_VIEW)
        rl_summary = _load_json(rl_recommendation_dir / RECOMMENDATION_SUMMARY)

    unified_episode_df: pd.DataFrame | None = None
    unified_summary: Dict[str, Any] | None = None
    if rl_unified_dir is not None:
        unified_episode_df = pd.read_parquet(rl_unified_dir / UNIFIED_RL_EPISODE_VIEW)
        unified_summary = _load_json(rl_unified_dir / UNIFIED_RL_SUMMARY)

    required_pred_cols = [
        "candidate_uid",
        "run_id",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "label_strong_trade_candidate_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "baseline_realized_pnl_bps_v1",
        BAD_PROB,
        RUNNER_PROB,
        *[column for policy, column in POLICY_COLUMNS.items() if policy != "NO_ENTRY_FALLBACK_BASELINE"],
    ]
    _require_columns(asof_df, ["candidate_uid", "run_id", "entry_observation_present_v1", "entry_raw_state_present_v1"], artifact_name=R5_2_AS_OF_FEATURE_TABLE)
    _require_columns(hindsight_df, ["candidate_uid", "giveback_bps_v1", "r5_2_label_bad_blocker_v1", "r5_2_label_runner_protect_v1"], artifact_name=R5_2_HINDSIGHT_LABEL_OUTCOME_TABLE)
    _require_columns(prediction_df, required_pred_cols, artifact_name=R5_2_POLICY_PREDICTION_VIEW)
    _require_columns(loso_df, ["policy_name_v1", "scope_v1", "slice_safety_pass_v1", "should_not_take_precision_v1"], artifact_name=R5_2_LOSO_METRICS)
    _require_columns(head_to_head_df, ["policy_name_v1", "scope_v1", "should_not_take_block_count_v1"], artifact_name=R5_2_HEAD_TO_HEAD)
    _require_columns(harvest_df, ["candidate_uid", "candidate_shadow_action_v1", "candidate_shadow_delta_bps_v1"], artifact_name=RETRAIN_PREDICTION_VIEW)
    if rl_trade_df is not None:
        _require_columns(rl_trade_df, ["candidate_uid", "rl_entry_recommendation_v1", "rl_management_recommendation_v1", "rl_priority_recommendation_v1"], artifact_name=RECOMMENDATION_TRADE_VIEW)
    if unified_episode_df is not None:
        _require_columns(unified_episode_df, ["candidate_uid"], artifact_name=UNIFIED_RL_EPISODE_VIEW)
    for name, frame in [
        (R5_2_AS_OF_FEATURE_TABLE, asof_df),
        (R5_2_HINDSIGHT_LABEL_OUTCOME_TABLE, hindsight_df),
        (R5_2_POLICY_PREDICTION_VIEW, prediction_df),
        (RETRAIN_PREDICTION_VIEW, harvest_df),
    ]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(prediction_df) != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger expected {expected_ledger_count}, observed {len(prediction_df)}")
    coverage = summary.get("coverage_v1", {}) if isinstance(summary.get("coverage_v1"), dict) else {}
    if int(coverage.get("entry_coverage_v1", -1)) != len(prediction_df):
        raise RuntimeError(f"R5.2 phase-gate requires full entry coverage, observed {coverage.get('entry_coverage_v1')}/{len(prediction_df)}")
    if int(coverage.get("missing_count_v1", -1)) != 0:
        raise RuntimeError(f"R5.2 phase-gate refuses missing rows: {coverage.get('missing_count_v1')}")
    if int(coverage.get("synthetic_count_v1", -1)) != 0:
        raise RuntimeError(f"R5.2 phase-gate refuses synthetic rows: {coverage.get('synthetic_count_v1')}")
    return (
        asof_df,
        hindsight_df,
        prediction_df,
        loso_df,
        head_to_head_df,
        summary,
        r5_2_consistency_df,
        contract,
        rl_trade_df,
        rl_summary,
        unified_episode_df,
        unified_summary,
        harvest_df,
        harvest_summary,
        harvest_audit_df,
    )


def _prepare_base_frame(asof_df: pd.DataFrame, hindsight_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    hindsight_cols = [
        "candidate_uid",
        "giveback_bps_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "r5_2_label_bad_blocker_v1",
        "r5_2_label_runner_protect_v1",
        "r5_2_label_runner_50_mfe_v1",
        "r5_2_label_runner_100_mfe_v1",
        "r5_2_label_runner_200_mfe_v1",
        "r5_2_label_repaired_165_like_runner_v1",
        "r5_2_label_strong_low_mae_runner_v1",
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
    ]
    pred_cols = [
        "candidate_uid",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "label_strong_trade_candidate_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        "strongest_winner_path_v1",
        "is_repaired_165_v1",
        "r5_2_batch04_hard_negative_runner_v1",
        "r5_2_hard_negative_like_asof_v1",
        "r5_2_hard_negative_similarity_distance_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "baseline_realized_pnl_bps_v1",
        *SCORE_COLUMNS,
        *POLICY_COLUMNS.values(),
    ]
    frame = (
        asof_df.merge(hindsight_df[[column for column in hindsight_cols if column in hindsight_df.columns]], on="candidate_uid", how="inner", validate="one_to_one")
        .merge(prediction_df[[column for column in pred_cols if column in prediction_df.columns]], on="candidate_uid", how="inner", validate="one_to_one")
    )
    frame["no_entry_fallback_baseline__block_v1"] = False
    frame["label_should_not_take_v1"] = _bool(frame, "label_should_not_take_v1")
    frame["label_strong_trade_candidate_v1"] = _bool(frame, "label_strong_trade_candidate_v1")
    frame["take_was_ok_v1"] = _bool(frame, "take_was_ok_v1")
    frame["fifty_plus_mfe_v1"] = _bool(frame, "fifty_plus_mfe_v1") | _num(frame, "peak_mfe_bps_v1").ge(50.0)
    frame["hundred_plus_mfe_v1"] = _bool(frame, "hundred_plus_mfe_v1") | _num(frame, "peak_mfe_bps_v1").ge(100.0)
    frame["two_hundred_plus_mfe_v1"] = _bool(frame, "two_hundred_plus_mfe_v1") | _num(frame, "peak_mfe_bps_v1").ge(200.0)
    if "tail_10_50_mfe_v1" not in frame.columns:
        frame["tail_10_50_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
            _num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(frame, "label_should_not_take_v1")
        )
    frame["tail_10_50_mfe_v1"] = _bool(frame, "tail_10_50_mfe_v1")
    if "strongest_winner_path_v1" not in frame.columns:
        frame["strongest_winner_path_v1"] = frame["two_hundred_plus_mfe_v1"] | (
            frame["label_strong_trade_candidate_v1"] & _num(frame, "baseline_realized_pnl_bps_v1").gt(0.0) & frame["fifty_plus_mfe_v1"]
        )
    frame["strongest_winner_path_v1"] = _bool(frame, "strongest_winner_path_v1")
    frame["is_repaired_165_v1"] = _bool(frame, "is_repaired_165_v1") | _bool(frame, "entry_coverage_repair_applied_v1")
    if int(_bool(frame, "entry_observation_present_v1").sum()) != len(frame):
        raise RuntimeError("R5.2 phase-gate requires entry_observation_present_v1 for every row")
    if int(_bool(frame, "entry_raw_state_present_v1").sum()) != len(frame):
        raise RuntimeError("R5.2 phase-gate requires entry_raw_state_present_v1 for every row")
    return frame


def _policy_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {policy: _bool(frame, column) for policy, column in POLICY_COLUMNS.items()}


def _global_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("hundred_plus_mfe_block_count_v1") or 0) > 1:
        failures.append("hundred_plus_mfe_block_count_v1>1")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 2:
        failures.append("fifty_plus_mfe_block_count_v1>2")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > 1:
        failures.append("strong_trade_false_block_count_v1>1")
    if int(metric.get("strongest_winner_path_block_count_v1") or 0) != 0:
        failures.append("strongest_winner_path_block_count_v1!=0")
    if precision is None or precision < 0.90:
        failures.append("precision<0.90")
    return not failures, ",".join(failures)


def _scope_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    block_count = int(metric.get("block_count_v1") or 0)
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 2:
        failures.append("fifty_plus_mfe_block_count_v1>2")
    if block_count > 0 and (precision is None or precision < 0.85):
        failures.append("precision<0.85")
    return not failures, ",".join(failures)


def _metric_with_safety(policy_name: str, scope: str, frame: pd.DataFrame, block: pd.Series, *, scope_family: str) -> Dict[str, Any]:
    metric = _policy_metric_row(policy_name, scope, frame, block, thresholds={"phase_gate_scope_family_v1": scope_family})
    passed, reasons = _global_safety(metric) if scope == "ALL_1971" else _scope_safety(metric)
    metric.update(
        {
            "phase_gate_scope_family_v1": scope_family,
            "phase_gate_safety_pass_v1": bool(passed),
            "phase_gate_safety_failure_reasons_v1": reasons,
        }
    )
    return metric


def _shadow_bakeoff(base: pd.DataFrame) -> pd.DataFrame:
    policies = _policy_masks(base)
    scopes = {
        "ALL_1971": pd.Series(True, index=base.index, dtype=bool),
        "REPAIRED_165": _bool(base, "is_repaired_165_v1"),
        "FIFTY_PLUS_MFE_RUNNERS": _bool(base, "fifty_plus_mfe_v1"),
        "HUNDRED_PLUS_MFE_RUNNERS": _bool(base, "hundred_plus_mfe_v1"),
        "TWO_HUNDRED_PLUS_MFE_RUNNERS": _bool(base, "two_hundred_plus_mfe_v1"),
        "STRONGEST_WINNER_PATH": _bool(base, "strongest_winner_path_v1"),
        "TAIL_10_50_MFE_POCKET": _bool(base, "tail_10_50_mfe_v1"),
        "HARD_NEGATIVE_6": _bool(base, "r5_2_batch04_hard_negative_runner_v1"),
    }
    rows: list[dict[str, Any]] = []
    for policy_name, mask in policies.items():
        for scope_name, scope_mask in scopes.items():
            scoped = base.loc[scope_mask].copy()
            rows.append(_metric_with_safety(policy_name, scope_name, scoped, mask.loc[scope_mask], scope_family="CANONICAL_SHADOW_REPLAY_BAKEOFF"))
    return pd.DataFrame(rows)


def _rolling_scope_masks(reports_root: Path, base: pd.DataFrame, *, batch_weeks: int) -> list[dict[str, Any]]:
    run_ids = _all_run_ids(reports_root, base)
    window = min(max(batch_weeks, 1), max(len(run_ids), 1))
    step = max(1, window // 3)
    scopes: list[dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(run_ids), step), start=1):
        batch_run_ids = run_ids[start : start + window]
        if not batch_run_ids:
            continue
        mask = base["run_id"].astype("string").isin(batch_run_ids)
        scopes.append(
            {
                "scope_v1": f"ROLLING_{window:02d}W_{idx:02d}",
                "mask_v1": mask,
                "run_count_v1": len(batch_run_ids),
                "run_start_v1": batch_run_ids[0],
                "run_end_v1": batch_run_ids[-1],
            }
        )
        if start + window >= len(run_ids):
            break
    return scopes


def _period_scope_masks(reports_root: Path, base: pd.DataFrame) -> list[dict[str, Any]]:
    run_ids = _all_run_ids(reports_root, base)
    chunks = np.array_split(np.array(run_ids, dtype=object), 3)
    names = ["EARLY_PERIOD", "MID_PERIOD", "LATE_PERIOD"]
    scopes: list[dict[str, Any]] = []
    for name, chunk in zip(names, chunks):
        values = [str(item) for item in chunk.tolist()]
        if not values:
            continue
        scopes.append(
            {
                "scope_v1": name,
                "mask_v1": base["run_id"].astype("string").isin(values),
                "run_count_v1": len(values),
                "run_start_v1": values[0],
                "run_end_v1": values[-1],
            }
        )
    return scopes


def _robustness_stress_matrix(reports_root: Path, base: pd.DataFrame, loso_df: pd.DataFrame, *, batch_weeks: int) -> pd.DataFrame:
    policies = _policy_masks(base)
    rows: list[dict[str, Any]] = []
    for slice_info in _slice_masks(reports_root, base, batch_weeks=batch_weeks):
        scope_mask = slice_info["mask_v1"].reindex(base.index).fillna(False).astype(bool)
        for policy_name, policy_mask in policies.items():
            row = _metric_with_safety(policy_name, str(slice_info["scope_v1"]), base.loc[scope_mask].copy(), policy_mask.loc[scope_mask], scope_family="WALK_FORWARD_GLOBAL_POLICY")
            row.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
            rows.append(row)
    for slice_info in _rolling_scope_masks(reports_root, base, batch_weeks=batch_weeks):
        scope_mask = slice_info["mask_v1"].reindex(base.index).fillna(False).astype(bool)
        for policy_name, policy_mask in policies.items():
            row = _metric_with_safety(policy_name, str(slice_info["scope_v1"]), base.loc[scope_mask].copy(), policy_mask.loc[scope_mask], scope_family="ROLLING_CHRONOLOGICAL_WINDOWS")
            row.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
            rows.append(row)
    for slice_info in _period_scope_masks(reports_root, base):
        scope_mask = slice_info["mask_v1"].reindex(base.index).fillna(False).astype(bool)
        for policy_name, policy_mask in policies.items():
            row = _metric_with_safety(policy_name, str(slice_info["scope_v1"]), base.loc[scope_mask].copy(), policy_mask.loc[scope_mask], scope_family="EARLY_MID_LATE_PERIOD_SPLIT")
            row.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
            rows.append(row)
    stress_scopes = {
        "BATCH_04_STRESS": next((item["mask_v1"].reindex(base.index).fillna(False).astype(bool) for item in _slice_masks(reports_root, base, batch_weeks=batch_weeks) if item["scope_v1"] == "BATCH_04"), pd.Series(False, index=base.index)),
        "BATCH_05_STRESS": next((item["mask_v1"].reindex(base.index).fillna(False).astype(bool) for item in _slice_masks(reports_root, base, batch_weeks=batch_weeks) if item["scope_v1"] == "BATCH_05"), pd.Series(False, index=base.index)),
        "HIGH_RUNNER_POCKET_STRESS": _bool(base, "fifty_plus_mfe_v1"),
        "REPAIRED_165_STRESS": _bool(base, "is_repaired_165_v1"),
        "TAIL_10_50_STRESS": _bool(base, "tail_10_50_mfe_v1"),
    }
    for scope_name, scope_mask in stress_scopes.items():
        for policy_name, policy_mask in policies.items():
            rows.append(_metric_with_safety(policy_name, scope_name, base.loc[scope_mask].copy(), policy_mask.loc[scope_mask], scope_family="POCKET_STRESS"))

    loso_selected = loso_df[loso_df["policy_name_v1"].astype("string").eq("R5_2_SELECTED_CANDIDATE")].copy()
    if loso_selected.empty:
        loso_selected = loso_df.copy()
    for _, source_row in loso_selected.iterrows():
        row = source_row.to_dict()
        row["policy_name_v1"] = "R5_2_SELECTED_CANDIDATE"
        row["phase_gate_scope_family_v1"] = "LOSO_OUT_OF_FOLD_FROM_R5_2"
        row["phase_gate_safety_pass_v1"] = str(source_row.get("slice_safety_pass_v1", False)).strip().lower() == "true"
        row["phase_gate_safety_failure_reasons_v1"] = str(source_row.get("slice_safety_failure_reasons_v1", ""))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "run_count_v1" not in out.columns:
        out["run_count_v1"] = np.nan
    return out


def _bin_calibration(frame: pd.DataFrame, score_col: str, label_col: str) -> str:
    valid = frame[[score_col, label_col]].copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid[valid[score_col].notna()]
    if valid.empty:
        return "[]"
    bins = pd.cut(valid[score_col], bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001], labels=["0_0.2", "0.2_0.4", "0.4_0.6", "0.6_0.8", "0.8_1.0"])
    rows = []
    for bin_name, part in valid.groupby(bins, observed=False):
        if part.empty:
            continue
        labels = _bool(part, label_col)
        rows.append(
            {
                "bin_v1": str(bin_name),
                "row_count_v1": int(len(part)),
                "mean_score_v1": _safe_float(part[score_col].mean()),
                "observed_positive_rate_v1": _safe_rate(float(labels.sum()), float(len(part))),
            }
        )
    return _json_dumps(rows)


def _calibration_audit(base: pd.DataFrame, reports_root: Path, *, batch_weeks: int) -> pd.DataFrame:
    scopes: list[dict[str, Any]] = [{"scope_v1": "ALL_1971", "mask_v1": pd.Series(True, index=base.index), "scope_family_v1": "GLOBAL"}]
    for item in _slice_masks(reports_root, base, batch_weeks=batch_weeks):
        scopes.append({"scope_v1": item["scope_v1"], "mask_v1": item["mask_v1"].reindex(base.index).fillna(False).astype(bool), "scope_family_v1": "WALK_FORWARD_SLICE"})
    scopes.append({"scope_v1": "HARD_NEGATIVE_6", "mask_v1": _bool(base, "r5_2_batch04_hard_negative_runner_v1"), "scope_family_v1": "BATCH04_HARD_NEGATIVE_MARGIN"})
    score_specs = [
        ("BAD_BLOCKER_HEAD", BAD_PROB, "r5_2_label_bad_blocker_v1"),
        ("RUNNER_PROTECTOR_HEAD", RUNNER_PROB, "r5_2_label_runner_protect_v1"),
    ]
    selected = _bool(base, "r5_2_selected_candidate__block_v1")
    should = _bool(base, "label_should_not_take_v1")
    take_ok = _bool(base, "take_was_ok_v1")
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        mask = scope["mask_v1"].reindex(base.index).fillna(False).astype(bool)
        scoped = base.loc[mask].copy()
        if scoped.empty:
            continue
        selected_scoped = selected.loc[mask]
        for score_name, score_col, label_col in score_specs:
            score = pd.to_numeric(scoped[score_col], errors="coerce")
            label = _bool(scoped, label_col)
            brier = ((score.fillna(0.0) - label.astype(float)) ** 2).mean()
            false_block_mask = selected_scoped & take_ok.loc[mask]
            bad_block_mask = selected_scoped & should.loc[mask]
            rows.append(
                {
                    "score_name_v1": score_name,
                    "score_column_v1": score_col,
                    "label_column_v1": label_col,
                    "scope_v1": scope["scope_v1"],
                    "scope_family_v1": scope["scope_family_v1"],
                    "row_count_v1": int(len(scoped)),
                    "label_positive_count_v1": int(label.sum()),
                    "mean_score_v1": _safe_float(score.mean()),
                    "median_score_v1": _safe_float(score.median()),
                    "p90_score_v1": _safe_float(score.quantile(0.90)),
                    "brier_score_v1": _safe_float(brier),
                    "false_block_mean_score_v1": _safe_float(pd.to_numeric(scoped.loc[false_block_mask, score_col], errors="coerce").mean()) if int(false_block_mask.sum()) else None,
                    "bad_block_mean_score_v1": _safe_float(pd.to_numeric(scoped.loc[bad_block_mask, score_col], errors="coerce").mean()) if int(bad_block_mask.sum()) else None,
                    "calibration_bins_json_v1": _bin_calibration(scoped, score_col, label_col),
                }
            )
    hard = base[_bool(base, "r5_2_batch04_hard_negative_runner_v1")].copy()
    if not hard.empty:
        margin = pd.to_numeric(hard[RUNNER_PROB], errors="coerce") - pd.to_numeric(hard[BAD_PROB], errors="coerce")
        rows.append(
            {
                "score_name_v1": "BATCH04_HARD_NEGATIVE_THRESHOLD_MARGIN",
                "score_column_v1": f"{RUNNER_PROB}_minus_{BAD_PROB}",
                "label_column_v1": "r5_2_batch04_hard_negative_runner_v1",
                "scope_v1": "HARD_NEGATIVE_6",
                "scope_family_v1": "BATCH04_HARD_NEGATIVE_MARGIN",
                "row_count_v1": int(len(hard)),
                "label_positive_count_v1": int(len(hard)),
                "mean_score_v1": _safe_float(margin.mean()),
                "median_score_v1": _safe_float(margin.median()),
                "p90_score_v1": _safe_float(margin.quantile(0.90)),
                "brier_score_v1": None,
                "false_block_mean_score_v1": None,
                "bad_block_mean_score_v1": None,
                "calibration_bins_json_v1": "[]",
            }
        )
    return pd.DataFrame(rows)


def _feature_pattern_json(frame: pd.DataFrame) -> str:
    feature_cols = [
        "as_of_candidate_tradable_prob_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_clv_v1",
        "as_of_atr_bps_v1",
    ]
    payload: Dict[str, Any] = {}
    for column in feature_cols:
        if column in frame.columns:
            payload[column] = _safe_float(pd.to_numeric(frame[column], errors="coerce").mean())
    for column in ["as_of_session_v1", "as_of_side_v1", "as_of_candidate_trend_regime_v1", "as_of_candidate_vol_regime_v1"]:
        if column in frame.columns:
            payload[f"{column}_top_values_v1"] = {
                str(key): int(value) for key, value in frame[column].astype("string").value_counts(dropna=False).head(5).to_dict().items()
            }
    return _json_dumps(payload)


def _failure_mode_mining(base: pd.DataFrame) -> pd.DataFrame:
    selected = _bool(base, "r5_2_selected_candidate__block_v1")
    should = _bool(base, "label_should_not_take_v1")
    take_ok = _bool(base, "take_was_ok_v1")
    tail = _bool(base, "tail_10_50_mfe_v1")
    risky_allow = (~selected) & should & (
        _num(base, "mae_abs_bps_v1").ge(40.0)
        | _num(base, "baseline_realized_pnl_bps_v1").le(-25.0)
        | pd.to_numeric(base[BAD_PROB], errors="coerce").ge(0.60).fillna(False)
    )
    failure_masks = {
        "FALSE_BLOCK": selected & take_ok,
        "MISSED_SHOULD_NOT_TAKE": (~selected) & should,
        "MISSED_10_50_TAIL_CONTROL": (~selected) & tail,
        "RISKY_ALLOW": risky_allow,
        "RUNNER_NEAR_MISS": take_ok & _bool(base, "fifty_plus_mfe_v1") & (
            pd.to_numeric(base[BAD_PROB], errors="coerce").ge(0.50).fillna(False)
            | pd.to_numeric(base[RUNNER_PROB], errors="coerce").lt(0.60).fillna(False)
            | selected
        ),
        "HIGH_CONFIDENCE_WRONG_DECISION": selected & take_ok & pd.to_numeric(base[BAD_PROB], errors="coerce").ge(0.70).fillna(False),
    }
    fix_map = {
        "FALSE_BLOCK": "R6_RUNNER_PROTECT_LABEL_OR_CALIBRATION",
        "MISSED_SHOULD_NOT_TAKE": "R6_BAD_BLOCK_RECALL_FEATURE_OR_LABEL",
        "MISSED_10_50_TAIL_CONTROL": "R6_TAIL_CONTROL_LABEL_FEATURE",
        "RISKY_ALLOW": "R6_BAD_BLOCK_RECALL_OR_THRESHOLD",
        "RUNNER_NEAR_MISS": "RUNNER_PROTECT_CALIBRATION_MONITOR",
        "HIGH_CONFIDENCE_WRONG_DECISION": "CALIBRATION_AND_FEATURE_DRIFT_AUDIT",
    }
    rows: list[dict[str, Any]] = []
    for failure_type, mask in failure_masks.items():
        part = base.loc[mask].copy()
        if part.empty:
            rows.append(
                {
                    "failure_type_v1": failure_type,
                    "row_count_v1": 0,
                    "slice_regime_pocket_json_v1": "{}",
                    "as_of_feature_pattern_json_v1": "{}",
                    "example_candidate_uids_json_v1": "[]",
                    "recommended_next_fix_v1": fix_map[failure_type],
                }
            )
            continue
        payload = {
            "top_run_ids_v1": {str(key): int(value) for key, value in part["run_id"].astype("string").value_counts().head(8).to_dict().items()},
            "repaired_165_count_v1": int(_bool(part, "is_repaired_165_v1").sum()),
            "fifty_plus_mfe_count_v1": int(_bool(part, "fifty_plus_mfe_v1").sum()),
            "hundred_plus_mfe_count_v1": int(_bool(part, "hundred_plus_mfe_v1").sum()),
            "two_hundred_plus_mfe_count_v1": int(_bool(part, "two_hundred_plus_mfe_v1").sum()),
            "tail_10_50_count_v1": int(_bool(part, "tail_10_50_mfe_v1").sum()),
        }
        rows.append(
            {
                "failure_type_v1": failure_type,
                "row_count_v1": int(len(part)),
                "slice_regime_pocket_json_v1": _json_dumps(payload),
                "as_of_feature_pattern_json_v1": _feature_pattern_json(part),
                "example_candidate_uids_json_v1": _json_dumps(part["candidate_uid"].astype("string").head(20).tolist()),
                "recommended_next_fix_v1": fix_map[failure_type],
            }
        )
    return pd.DataFrame(rows)


def _count_failures(audit_df: pd.DataFrame) -> int:
    if audit_df is None or audit_df.empty or "status_v1" not in audit_df.columns:
        return 0
    return int(audit_df["status_v1"].astype("string").str.upper().eq("FAIL").sum())


def _harvest_scope_row(name: str, frame: pd.DataFrame) -> Dict[str, Any]:
    selected = _bool(frame, "r5_2_selected_candidate__block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    entry_target = frame.get("entry_xgb_harvest_label_v1", pd.Series("", index=frame.index)).astype("string").eq("REJECT_OR_LOW_SIZE") | frame.get("management_rl_harvest_action_label_v1", pd.Series("", index=frame.index)).astype("string").eq("ENTRY_SUPPRESS_OR_DOWNSIZE")
    management_target = ~entry_target & frame.get("management_rl_harvest_action_label_v1", pd.Series("", index=frame.index)).astype("string").ne("<NA>")
    action_match = _bool(frame, "candidate_shadow_action_matches_harvest_target_v1")
    return {
        "scope_v1": name,
        "row_count_v1": int(len(frame)),
        "r5_2_block_count_v1": int(selected.sum()),
        "r5_2_true_bad_block_count_v1": int((selected & should).sum()),
        "r5_2_false_block_count_v1": int((selected & ~should).sum()),
        "harvest_entry_suppress_target_count_v1": int(entry_target.sum()),
        "r5_2_blocks_harvest_entry_suppress_count_v1": int((selected & entry_target).sum()),
        "r5_2_block_to_harvest_entry_target_precision_v1": _safe_rate(float((selected & entry_target).sum()), float(selected.sum())),
        "harvest_entry_target_recall_by_r5_2_v1": _safe_rate(float((selected & entry_target).sum()), float(entry_target.sum())),
        "harvest_candidate_action_match_rate_v1": _safe_rate(float(action_match.sum()), float(len(frame))),
        "candidate_shadow_delta_bps_sum_v1": _safe_float(_num(frame, "candidate_shadow_delta_bps_v1").sum()),
        "r5_2_blocked_candidate_shadow_delta_bps_sum_v1": _safe_float(_num(frame.loc[selected], "candidate_shadow_delta_bps_v1").sum()) if int(selected.sum()) else 0.0,
        "r5_2_blocked_entry_skip_delta_bps_sum_v1": _safe_float(_num(frame.loc[selected], "rl_priority_entry_skip_delta_bps_v1").sum()) if "rl_priority_entry_skip_delta_bps_v1" in frame.columns and int(selected.sum()) else 0.0,
        "allowed_management_target_count_v1": int((~selected & management_target).sum()),
        "allowed_management_target_delta_bps_sum_v1": _safe_float(_num(frame.loc[~selected & management_target], "candidate_shadow_delta_bps_v1").sum()) if int((~selected & management_target).sum()) else 0.0,
        "rl_skip_recommendation_overlap_count_v1": int((selected & frame.get("rl_entry_recommendation_v1", pd.Series("", index=frame.index)).astype("string").eq("SKIP_TRADE")).sum()),
        "rl_priority_skip_overlap_count_v1": int((selected & frame.get("rl_priority_recommendation_v1", pd.Series("", index=frame.index)).astype("string").eq("SKIP_TRADE")).sum()),
        "unified_episode_covered_count_v1": int(frame.get("unified_episode_coverage_status_v1", pd.Series("", index=frame.index)).astype("string").eq("COVERED_BY_UNIFIED_ENTRY_EPISODE").sum()) if "unified_episode_coverage_status_v1" in frame.columns else None,
    }


def _harvest_impact(
    base: pd.DataFrame,
    harvest_df: pd.DataFrame,
    harvest_audit_df: pd.DataFrame,
    rl_trade_df: pd.DataFrame | None,
    unified_episode_df: pd.DataFrame | None,
) -> pd.DataFrame:
    harvest_cols = [
        "candidate_uid",
        "harvest_quality_bucket_v1",
        "exit_harvest_policy_action_v1",
        "rl_priority_entry_skip_delta_bps_v1",
        "rl_priority_exit_earlier_delta_bps_v1",
        "rl_priority_hold_longer_delta_bps_v1",
        "management_rl_harvest_reward_bps_raw_v1",
        "entry_xgb_harvest_label_v1",
        "entry_xgb_binary_take_target_v1",
        "exit_transformer_supervision_label_v1",
        "management_rl_harvest_action_label_v1",
        "candidate_shadow_action_v1",
        "candidate_shadow_action_source_v1",
        "candidate_shadow_action_matches_harvest_target_v1",
        "candidate_shadow_delta_bps_v1",
        "candidate_shadow_delta_clipped_200_bps_v1",
    ]
    work = base[["candidate_uid", "r5_2_selected_candidate__block_v1", "label_should_not_take_v1", "take_was_ok_v1", "fifty_plus_mfe_v1", "is_repaired_165_v1", "tail_10_50_mfe_v1"]].merge(
        harvest_df[[column for column in harvest_cols if column in harvest_df.columns]],
        on="candidate_uid",
        how="inner",
        validate="one_to_one",
    )
    if rl_trade_df is not None:
        rl_cols = ["candidate_uid", "rl_entry_recommendation_v1", "rl_management_recommendation_v1", "rl_priority_recommendation_v1", "unified_episode_coverage_status_v1"]
        work = work.merge(rl_trade_df[[column for column in rl_cols if column in rl_trade_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    if unified_episode_df is not None and "unified_episode_coverage_status_v1" not in work.columns:
        covered = set(unified_episode_df["candidate_uid"].astype("string").tolist())
        work["unified_episode_coverage_status_v1"] = work["candidate_uid"].astype("string").isin(covered).map(
            {True: "COVERED_BY_UNIFIED_ENTRY_EPISODE", False: "NOT_COVERED_BY_UNIFIED_ENTRY_EPISODE"}
        )
    selected = _bool(work, "r5_2_selected_candidate__block_v1")
    scopes = {
        "ALL_1971": pd.Series(True, index=work.index),
        "R5_2_BLOCKED": selected,
        "R5_2_ALLOWED": ~selected,
        "R5_2_TRUE_BAD_BLOCKS": selected & _bool(work, "label_should_not_take_v1"),
        "R5_2_FALSE_BLOCKS": selected & _bool(work, "take_was_ok_v1"),
        "REPAIRED_165": _bool(work, "is_repaired_165_v1"),
        "FIFTY_PLUS_MFE_RUNNERS": _bool(work, "fifty_plus_mfe_v1"),
        "TAIL_10_50_MFE_POCKET": _bool(work, "tail_10_50_mfe_v1"),
    }
    rows = [_harvest_scope_row(name, work.loc[mask].copy()) for name, mask in scopes.items()]
    for row in rows:
        row["harvest_failed_check_count_v1"] = _count_failures(harvest_audit_df)
    return pd.DataFrame(rows)


def _explain_row(row: pd.Series) -> tuple[str, str]:
    blocked = bool(row.get("r5_2_selected_candidate__block_v1", False))
    bad_score = _safe_float(row.get(BAD_PROB)) or 0.0
    runner_score = _safe_float(row.get(RUNNER_PROB)) or 0.0
    if bool(row.get("r5_2_batch04_hard_negative_runner_v1", False)):
        return ("RUNNER_PROTECTED", "BATCH04_HARD_NEGATIVE_RUNNER_PROTECT")
    if bool(row.get("is_repaired_165_v1", False)) and not blocked:
        return ("RUNNER_PROTECTED", "REPAIRED_165_POCKET_PROTECTED")
    if not blocked and runner_score >= 0.70:
        return ("RUNNER_PROTECTED", "RUNNER_PROTECTOR_SCORE_HIGH")
    if blocked and bool(row.get("tail_10_50_mfe_v1", False)):
        return ("BLOCKED", "TAIL_10_50_CONTROL_AND_BAD_RISK")
    if blocked and bad_score >= 0.60 and runner_score < 0.74:
        return ("BLOCKED", "BAD_BLOCKER_HIGH_RUNNER_LOW")
    if blocked:
        return ("BLOCKED", "R5_CURRENT_RUNNER_GATED_STACK")
    return ("ALLOWED", "NO_BLOCK_SIGNAL_OR_RUNNER_PROTECTED")


def _policy_logging_explainability(base: pd.DataFrame) -> pd.DataFrame:
    snapshot_cols = [
        "candidate_uid",
        "run_id",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "as_of_session_v1",
        "as_of_side_v1",
        "as_of_candidate_tradable_prob_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        BAD_PROB,
        RUNNER_PROB,
        "r5_2_selected_candidate__block_v1",
        "is_repaired_165_v1",
        "r5_2_batch04_hard_negative_runner_v1",
        "tail_10_50_mfe_v1",
    ]
    out = base[[column for column in snapshot_cols if column in base.columns]].copy()
    explanations = out.apply(_explain_row, axis=1)
    out["selected_policy_stack_v1"] = "R5_2_CANDIDATE_00165_R5_CURRENT_RUNNER_GATED_none"
    out["decision_provenance_v1"] = "R5_2_SHADOW_PHASE_GATE_OFFLINE_AS_OF_SCORES"
    out["blocker_score_v1"] = pd.to_numeric(out.get(BAD_PROB, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["runner_protector_score_v1"] = pd.to_numeric(out.get(RUNNER_PROB, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["blocking_reason_v1"] = [item[1] if item[0] == "BLOCKED" else "NOT_BLOCKED" for item in explanations]
    out["runner_protection_reason_v1"] = [item[1] if item[0] == "RUNNER_PROTECTED" else "NOT_RUNNER_PROTECTED" for item in explanations]
    out["safety_constraint_status_v1"] = np.where(
        _bool(out, "r5_2_selected_candidate__block_v1") & _bool(out, "is_repaired_165_v1"),
        "FAIL_REPAIRED_165_BLOCK",
        "PASS_ROW_LEVEL_SHADOW_LOG",
    )
    out["as_of_feature_snapshot_contract_v1"] = "AS_OF_SNAPSHOT_ONLY"
    out["hindsight_outcome_backfill_contract_v1"] = "HINDSIGHT_WRITTEN_SEPARATELY_IN_PHASE_GATE_HINDSIGHT_TABLE"
    return out


def _decision_matrix(
    *,
    bakeoff_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    harvest_df: pd.DataFrame,
    r5_2_summary: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    all_rows = bakeoff_df[bakeoff_df["scope_v1"].eq("ALL_1971")]
    all_by_policy = {str(row["policy_name_v1"]): row.to_dict() for _, row in all_rows.iterrows()}
    r52 = all_by_policy["R5_2_SELECTED_CANDIDATE"]
    r5 = all_by_policy["R5_CURRENT_REFERENCE"]
    r51 = all_by_policy["R5_1_SAFETY_REFERENCE"]
    r4 = all_by_policy["R4_CURRENT_REFERENCE"]
    loso = robustness_df[
        robustness_df["policy_name_v1"].astype("string").eq("R5_2_SELECTED_CANDIDATE")
        & robustness_df["phase_gate_scope_family_v1"].astype("string").eq("LOSO_OUT_OF_FOLD_FROM_R5_2")
    ]
    batch04 = loso[loso["scope_v1"].astype("string").eq("BATCH_04")]
    batch05 = loso[loso["scope_v1"].astype("string").eq("BATCH_05")]
    precisions = pd.to_numeric(loso["should_not_take_precision_v1"], errors="coerce").dropna()
    worst_loso_precision = float(precisions.min()) if len(precisions) else None
    global_pass, global_fail = _global_safety(r52)
    loso_pass = bool(loso["phase_gate_safety_pass_v1"].fillna(False).all()) if not loso.empty else False
    batch04_pass = _optional_pass(batch04, "phase_gate_safety_pass_v1")
    batch05_pass = _optional_pass(batch05, "phase_gate_safety_pass_v1")
    batch04_ok = True if batch04_pass is None else bool(batch04_pass)
    batch05_ok = True if batch05_pass is None else bool(batch05_pass)
    rolling = robustness_df[
        robustness_df["policy_name_v1"].astype("string").eq("R5_2_SELECTED_CANDIDATE")
        & robustness_df["phase_gate_scope_family_v1"].astype("string").eq("ROLLING_CHRONOLOGICAL_WINDOWS")
    ]
    rolling_pass = bool(rolling["phase_gate_safety_pass_v1"].fillna(False).all()) if not rolling.empty else True
    harvest_all = harvest_df[harvest_df["scope_v1"].eq("ALL_1971")].iloc[0].to_dict() if not harvest_df.empty else {}
    beats_r4 = int(r52["should_not_take_block_count_v1"]) > int(r4["should_not_take_block_count_v1"])
    beats_r51 = int(r52["should_not_take_block_count_v1"]) > int(r51["should_not_take_block_count_v1"])
    near_r5_edge = int(r52["should_not_take_block_count_v1"]) >= int(r5["should_not_take_block_count_v1"]) - 5
    harvest_ready = int(harvest_all.get("harvest_failed_check_count_v1") or 0) == 0 and int(harvest_all.get("unified_episode_covered_count_v1") or 0) in {0, int(harvest_all.get("row_count_v1") or 0)}
    if global_pass and loso_pass and batch04_ok and batch05_ok and rolling_pass and beats_r4 and beats_r51 and near_r5_edge:
        recommendation = "FREEZE_R5_2_SHADOW_FALLBACK_CANDIDATE"
    elif global_pass and loso_pass and beats_r51:
        recommendation = "USE_R5_2_AS_R6_BASE"
    elif harvest_ready and global_pass:
        recommendation = "R5_2_HARVEST_REPLAY_CANDIDATE"
    elif not global_pass or not loso_pass:
        recommendation = "R6_RETRAIN_REQUIRED"
    else:
        recommendation = "ENTRY_NOT_READY_FOR_FREEZE"
    rows = [
        {"decision_key_v1": "FREEZE_R5_2_SHADOW_FALLBACK_CANDIDATE", "status_v1": "PASS" if recommendation == "FREEZE_R5_2_SHADOW_FALLBACK_CANDIDATE" else "NOT_PRIMARY", "reason_v1": "Requires global safety, LOSO, BATCH_04/BATCH_05, rolling safety, and edge near R5 current."},
        {"decision_key_v1": "USE_R5_2_AS_R6_BASE", "status_v1": "PASS" if recommendation == "USE_R5_2_AS_R6_BASE" else "NOT_PRIMARY", "reason_v1": "Use when R5.2 is safe but not strong enough to freeze."},
        {"decision_key_v1": "R5_2_HARVEST_REPLAY_CANDIDATE", "status_v1": "PASS" if recommendation == "R5_2_HARVEST_REPLAY_CANDIDATE" else "NOT_PRIMARY", "reason_v1": "Use when harvest integration is ready but entry fallback still needs more gate work."},
        {"decision_key_v1": "R6_RETRAIN_REQUIRED", "status_v1": "PASS" if recommendation == "R6_RETRAIN_REQUIRED" else "NOT_PRIMARY", "reason_v1": "Use when phase-gate safety constraints fail."},
        {"decision_key_v1": "KEEP_R5_CURRENT_AS_EDGE_REFERENCE", "status_v1": "REFERENCE", "reason_v1": "R5 current remains edge reference, not freeze candidate if it causes runner damage."},
        {"decision_key_v1": "KEEP_R5_1_AS_SAFETY_REFERENCE", "status_v1": "REFERENCE", "reason_v1": "R5.1 remains safety reference."},
        {"decision_key_v1": "ENTRY_NOT_READY_FOR_FREEZE", "status_v1": "PASS" if recommendation == "ENTRY_NOT_READY_FOR_FREEZE" else "NOT_PRIMARY", "reason_v1": "Use when evidence is mixed."},
    ]
    summary = {
        "recommended_phase_gate_decision_v1": recommendation,
        "global_safety_pass_v1": bool(global_pass),
        "global_safety_failure_reasons_v1": global_fail,
        "loso_all_slices_pass_v1": bool(loso_pass),
        "batch04_loso_pass_v1": batch04_pass,
        "batch05_loso_pass_v1": batch05_pass,
        "rolling_windows_pass_v1": bool(rolling_pass),
        "worst_loso_precision_v1": worst_loso_precision,
        "r5_2_should_not_blocks_v1": int(r52["should_not_take_block_count_v1"]),
        "r5_2_precision_v1": _safe_float(r52.get("should_not_take_precision_v1")),
        "r5_2_false_blocks_v1": int(r52["take_was_ok_block_count_v1"]),
        "r5_2_fifty_plus_blocked_v1": int(r52["fifty_plus_mfe_block_count_v1"]),
        "r5_2_hundred_plus_blocked_v1": int(r52["hundred_plus_mfe_block_count_v1"]),
        "r5_2_two_hundred_plus_blocked_v1": int(r52["two_hundred_plus_mfe_block_count_v1"]),
        "r5_2_repaired_165_blocked_v1": int(r52["repaired_165_block_count_v1"]),
        "r5_2_strongest_winner_path_blocked_v1": int(r52["strongest_winner_path_block_count_v1"]),
        "r5_2_tail_10_50_help_v1": int(r52["tail_10_50_help_count_v1"]),
        "r4_should_not_blocks_v1": int(r4["should_not_take_block_count_v1"]),
        "r5_current_should_not_blocks_v1": int(r5["should_not_take_block_count_v1"]),
        "r5_1_should_not_blocks_v1": int(r51["should_not_take_block_count_v1"]),
        "r5_2_original_recommendation_v1": r5_2_summary.get("decision_v1", {}).get("recommended_next_step_v1"),
        "harvest_ready_v1": bool(harvest_ready),
    }
    return pd.DataFrame(rows), summary


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    d = summary["decision_v1"]
    lines = [
        "# R5.2 Shadow Phase Gate And Harvest Integration V1",
        "",
        "Shadow/research only. Not a live gate and not a controller.",
        "",
        "## Headline",
        "",
        f"- Decision: `{d['recommended_phase_gate_decision_v1']}`",
        f"- R5.2 bad blocks: `{d['r5_2_should_not_blocks_v1']}` at precision `{d['r5_2_precision_v1']}`",
        f"- R5 current bad blocks: `{d['r5_current_should_not_blocks_v1']}`",
        f"- R5.1 safety bad blocks: `{d['r5_1_should_not_blocks_v1']}`",
        f"- BATCH_04 LOSO pass: `{d['batch04_loso_pass_v1']}`",
        f"- BATCH_05 LOSO pass: `{d['batch05_loso_pass_v1']}`",
        f"- Repaired-165 blocked: `{d['r5_2_repaired_165_blocked_v1']}`",
        f"- 50+/100+/200+ MFE blocked: `{d['r5_2_fifty_plus_blocked_v1']}/{d['r5_2_hundred_plus_blocked_v1']}/{d['r5_2_two_hundred_plus_blocked_v1']}`",
        f"- Strongest-winner path blocked: `{d['r5_2_strongest_winner_path_blocked_v1']}`",
        f"- 10-50 MFE tail-control help: `{d['r5_2_tail_10_50_help_v1']}`",
        "",
        "## Separation",
        "",
        "- AS_OF feature table and HINDSIGHT outcome table are materialized separately.",
        "- Policy logging includes scores and reasons, but HINDSIGHT backfill remains audit-only.",
        "- No live promotion.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    r5_2_dir: Path,
    harvest_dir: Path,
    rl_recommendation_dir: Path | None,
    rl_unified_dir: Path | None,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None,
) -> Dict[str, Any]:
    (
        asof_df,
        hindsight_df,
        prediction_df,
        loso_df,
        head_to_head_df,
        r5_2_summary,
        r5_2_consistency_df,
        r5_2_contract,
        rl_trade_df,
        rl_summary,
        unified_episode_df,
        unified_summary,
        harvest_prediction_df,
        harvest_summary,
        harvest_audit_df,
    ) = _load_inputs(
        r5_2_dir=r5_2_dir,
        harvest_dir=harvest_dir,
        rl_recommendation_dir=rl_recommendation_dir,
        rl_unified_dir=rl_unified_dir,
        expected_ledger_count=expected_ledger_count,
    )
    base = _prepare_base_frame(asof_df, hindsight_df, prediction_df)
    bakeoff_df = _shadow_bakeoff(base)
    robustness_df = _robustness_stress_matrix(reports_root, base, loso_df, batch_weeks=batch_weeks)
    calibration_df = _calibration_audit(base, reports_root, batch_weeks=batch_weeks)
    failure_df = _failure_mode_mining(base)
    harvest_impact_df = _harvest_impact(base, harvest_prediction_df, harvest_audit_df, rl_trade_df, unified_episode_df)
    explainability_df = _policy_logging_explainability(base)
    decision_df, decision_summary = _decision_matrix(
        bakeoff_df=bakeoff_df,
        robustness_df=robustness_df,
        harvest_df=harvest_impact_df,
        r5_2_summary=r5_2_summary,
    )
    coverage = r5_2_summary.get("coverage_v1", {}) if isinstance(r5_2_summary.get("coverage_v1"), dict) else {}
    r5_2_failed = _count_failures(r5_2_consistency_df)
    harvest_failed = _count_failures(harvest_audit_df)
    unified_coverage = int(harvest_impact_df[harvest_impact_df["scope_v1"].eq("ALL_1971")]["unified_episode_covered_count_v1"].iloc[0] or 0)
    consistency_df = pd.DataFrame(
        [
            _audit_record("R5_2_INPUT_PRESENT", "PASS", {"r5_2_dir": str(r5_2_dir)}),
            _audit_record("HARVEST_INPUT_PRESENT", "PASS", {"harvest_dir": str(harvest_dir)}),
            _audit_record("RL_RECOMMENDATION_INPUT_PRESENT", "PASS" if rl_recommendation_dir else "WARN", {"rl_recommendation_dir": str(rl_recommendation_dir) if rl_recommendation_dir else None}),
            _audit_record("RL_UNIFIED_INPUT_PRESENT", "PASS" if rl_unified_dir else "WARN", {"rl_unified_dir": str(rl_unified_dir) if rl_unified_dir else None}),
            _audit_record("LOCKED_LEDGER_COUNT", "PASS" if expected_ledger_count is None or len(base) == expected_ledger_count else "FAIL", {"expected": expected_ledger_count, "observed": len(base)}),
            _audit_record("FULL_R5_2_COVERAGE", "PASS" if int(coverage.get("entry_coverage_v1", 0)) == len(base) and int(coverage.get("missing_count_v1", -1)) == 0 else "FAIL", {"coverage": coverage}),
            _audit_record("NO_SYNTHETIC_ROWS", "PASS" if int(coverage.get("synthetic_count_v1", -1)) == 0 else "FAIL", {"synthetic_count": coverage.get("synthetic_count_v1")}),
            _audit_record("AS_OF_HINDSIGHT_PHYSICAL_SEPARATION", "PASS", {"as_of_table": AS_OF_TABLE, "hindsight_table": HINDSIGHT_TABLE}),
            _audit_record("R5_2_UPSTREAM_CONSISTENCY", "PASS" if r5_2_failed == 0 else "FAIL", {"failed_checks": r5_2_failed}),
            _audit_record("HARVEST_CONSISTENCY", "PASS" if harvest_failed == 0 else "FAIL", {"failed_checks": harvest_failed}),
            _audit_record("UNIFIED_EPISODE_COVERAGE", "PASS" if unified_coverage in {0, len(base)} else "FAIL", {"covered": unified_coverage, "expected": len(base)}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R5_2_SHADOW_PHASE_GATE_STATUS_V1",
        "R5_2_SHADOW_PHASE_GATE_STATUS": "PHASE_GATE_COMPLETE_NOT_PROMOTED" if failed_checks == 0 else "PHASE_GATE_ISSUES_FOUND_NOT_PROMOTED",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R5_2_SHADOW_PHASE_GATE_AND_HARVEST_INTEGRATION_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "input_dirs_v1": {
            "r5_2": str(r5_2_dir),
            "harvest": str(harvest_dir),
            "rl_recommendation": str(rl_recommendation_dir) if rl_recommendation_dir else None,
            "rl_unified": str(rl_unified_dir) if rl_unified_dir else None,
        },
        "coverage_v1": {
            "ledger_trade_count_v1": int(len(base)),
            "entry_coverage_v1": int(coverage.get("entry_coverage_v1", len(base))),
            "entry_raw_coverage_v1": int(coverage.get("entry_raw_coverage_v1", len(base))),
            "missing_count_v1": int(coverage.get("missing_count_v1", 0)),
            "synthetic_count_v1": int(coverage.get("synthetic_count_v1", 0)),
            "repaired_rows_v1": int(coverage.get("repaired_rows_v1", int(_bool(base, "is_repaired_165_v1").sum()))),
        },
        "decision_v1": decision_summary,
        "r5_2_upstream_decision_v1": r5_2_summary.get("decision_v1", {}),
        "harvest_summary_status_v1": harvest_summary.get("status_v1", {}),
        "rl_summary_status_v1": rl_summary.get("status_v1", {}) if rl_summary else {},
        "unified_summary_status_v1": unified_summary.get("status_v1", {}) if unified_summary else {},
        "artifact_counts_v1": {
            "shadow_replay_bakeoff_rows_v1": int(len(bakeoff_df)),
            "robustness_stress_rows_v1": int(len(robustness_df)),
            "calibration_audit_rows_v1": int(len(calibration_df)),
            "failure_mode_rows_v1": int(len(failure_df)),
            "harvest_impact_rows_v1": int(len(harvest_impact_df)),
            "policy_logging_rows_v1": int(len(explainability_df)),
        },
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R5.2 phase-gate used {len(base)}/{len(base)} rows with missing_count={coverage.get('missing_count_v1', 0)} and synthetic_count={coverage.get('synthetic_count_v1', 0)}.",
                f"R5.2 selected candidate blocks {decision_summary['r5_2_should_not_blocks_v1']} should_not_take rows at precision {decision_summary['r5_2_precision_v1']}.",
                f"Repaired-165 damage={decision_summary['r5_2_repaired_165_blocked_v1']}, 200+ MFE damage={decision_summary['r5_2_two_hundred_plus_blocked_v1']}, strongest-winner damage={decision_summary['r5_2_strongest_winner_path_blocked_v1']}.",
                f"BATCH_04 LOSO pass={decision_summary['batch04_loso_pass_v1']} and BATCH_05 LOSO pass={decision_summary['batch05_loso_pass_v1']}.",
                "No live promotion was materialized.",
            ],
            "INDIKERT": [
                "Calibration audit and BATCH_04 hard-negative margins indicate whether protection is learned by the runner-protector head, not just a lucky threshold.",
                "Harvest impact indicates whether R5.2 can be used as a shadow input to the harvest/RL retrain line without damaging management/exit observability.",
            ],
            "IKKE_ETABLERT": [
                "Live entry gate safety.",
                "Causal future-regime improvement beyond locked canonical replay.",
                "Counterfactual fills from blocked trades; all pnl deltas remain HINDSIGHT audit only.",
            ],
        },
    }
    contract = {
        "layer_name": "R5_2_SHADOW_PHASE_GATE_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_dirs_v1": summary["input_dirs_v1"],
        "source_r5_2_contract_v1": r5_2_contract.get("layer_name"),
        "as_of_hindsight_separation_v1": {
            "as_of_table_v1": AS_OF_TABLE,
            "hindsight_table_v1": HINDSIGHT_TABLE,
            "hindsight_is_supervision_or_audit_only_v1": True,
        },
        "safety_constraints_v1": {
            "ledger_coverage_v1": "1971/1971",
            "missing_count_v1": 0,
            "synthetic_count_v1": 0,
            "repaired_165_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "hundred_plus_mfe_blocked_max_v1": 1,
            "fifty_plus_mfe_blocked_max_v1": 2,
            "strong_false_blocks_max_v1": 1,
            "batch04_loso_pass_v1": True,
            "batch05_loso_pass_v1": True,
            "worst_slice_precision_min_v1": 0.85,
            "global_precision_min_v1": 0.90,
            "strongest_winner_path_damage_v1": 0,
        },
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R5_2_SHADOW_PHASE_GATE_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_table": AS_OF_TABLE,
            "hindsight_table": HINDSIGHT_TABLE,
            "shadow_replay_bakeoff": SHADOW_REPLAY_BAKEOFF,
            "robustness_stress_matrix": ROBUSTNESS_STRESS_MATRIX,
            "calibration_audit": CALIBRATION_AUDIT,
            "failure_mode_table": FAILURE_MODE_TABLE,
            "harvest_impact": HARVEST_IMPACT,
            "policy_logging_explainability": POLICY_LOGGING_EXPLAINABILITY,
            "decision_matrix": DECISION_MATRIX,
            "summary": SUMMARY,
            "status": STATUS,
            "report": REPORT,
            "consistency_audit": CONSISTENCY_AUDIT,
        },
    }
    asof_out = asof_df.copy()
    asof_out["r5_2_shadow_phase_gate_as_of_contract_v1"] = "AS_OF_ONLY_NO_HINDSIGHT_FEATURES"
    hindsight_cols = [
        "candidate_uid",
        "run_id",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "r5_2_label_bad_blocker_v1",
        "r5_2_label_runner_protect_v1",
        "r5_2_label_runner_50_mfe_v1",
        "r5_2_label_runner_100_mfe_v1",
        "r5_2_label_runner_200_mfe_v1",
        "r5_2_label_repaired_165_like_runner_v1",
        "r5_2_label_strong_low_mae_runner_v1",
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
    ]
    hindsight_out = base[[column for column in hindsight_cols if column in base.columns]].copy()
    hindsight_out["r5_2_shadow_phase_gate_hindsight_contract_v1"] = "HINDSIGHT_AUDIT_ONLY_NOT_AS_OF_FEATURES_NOT_POLICY_TRUTH"
    return {
        "asof_df": asof_out,
        "hindsight_df": hindsight_out,
        "shadow_replay_bakeoff_df": bakeoff_df,
        "robustness_stress_df": robustness_df,
        "calibration_audit_df": calibration_df,
        "failure_mode_df": failure_df,
        "harvest_impact_df": harvest_impact_df,
        "policy_logging_df": explainability_df,
        "decision_df": decision_df,
        "consistency_df": consistency_df,
        "contract": contract,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    r5_2_dir: Path | None = None,
    harvest_dir: Path | None = None,
    rl_recommendation_dir: Path | None = None,
    rl_unified_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    r5_2_dir = _resolve_dir(reports_root, str(r5_2_dir) if r5_2_dir else None, R5_2_EXTENSION_NAME, R5_2_SUMMARY)
    harvest_dir = _resolve_dir(reports_root, str(harvest_dir) if harvest_dir else None, HARVEST_RETRAIN_EXTENSION_NAME, RETRAIN_SUMMARY)
    resolved_rl_recommendation_dir = _resolve_dir_from_top_summary(
        reports_root=reports_root,
        path_arg=str(rl_recommendation_dir) if rl_recommendation_dir else None,
        top_summary_name="truth_rl_recommendation_candidate_v1.json",
        fallback_contains="RL_RECOMMENDATION_CANDIDATE_V1",
        required_file=RECOMMENDATION_TRADE_VIEW,
    )
    resolved_rl_unified_dir = _resolve_dir_from_top_summary(
        reports_root=reports_root,
        path_arg=str(rl_unified_dir) if rl_unified_dir else None,
        top_summary_name="truth_rl_unified_observability_v1.json",
        fallback_contains="RL_UNIFIED_OBSERVABILITY_V1",
        required_file=UNIFIED_RL_EPISODE_VIEW,
    )
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        r5_2_dir=r5_2_dir,
        harvest_dir=harvest_dir,
        rl_recommendation_dir=resolved_rl_recommendation_dir,
        rl_unified_dir=resolved_rl_unified_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
    )
    payload["asof_df"].to_parquet(extension_dir / AS_OF_TABLE, index=False)
    payload["hindsight_df"].to_parquet(extension_dir / HINDSIGHT_TABLE, index=False)
    payload["shadow_replay_bakeoff_df"].to_csv(extension_dir / SHADOW_REPLAY_BAKEOFF, index=False)
    payload["robustness_stress_df"].to_csv(extension_dir / ROBUSTNESS_STRESS_MATRIX, index=False)
    payload["calibration_audit_df"].to_csv(extension_dir / CALIBRATION_AUDIT, index=False)
    payload["failure_mode_df"].to_csv(extension_dir / FAILURE_MODE_TABLE, index=False)
    payload["harvest_impact_df"].to_csv(extension_dir / HARVEST_IMPACT, index=False)
    payload["policy_logging_df"].to_parquet(extension_dir / POLICY_LOGGING_EXPLAINABILITY, index=False)
    payload["decision_df"].to_csv(extension_dir / DECISION_MATRIX, index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    _write_json(extension_dir / STATUS, payload["status"])
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"])
    return {
        "extension_dir": str(extension_dir),
        "top_level_summary_path": str(reports_root / TOP_LEVEL_SUMMARY),
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build R5.2 shadow phase-gate and harvest integration.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--r5-2-dir", default=None)
    parser.add_argument("--harvest-dir", default=None)
    parser.add_argument("--rl-recommendation-dir", default=None)
    parser.add_argument("--rl-unified-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        r5_2_dir=Path(args.r5_2_dir).expanduser().resolve() if args.r5_2_dir else None,
        harvest_dir=Path(args.harvest_dir).expanduser().resolve() if args.harvest_dir else None,
        rl_recommendation_dir=Path(args.rl_recommendation_dir).expanduser().resolve() if args.rl_recommendation_dir else None,
        rl_unified_dir=Path(args.rl_unified_dir).expanduser().resolve() if args.rl_unified_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
