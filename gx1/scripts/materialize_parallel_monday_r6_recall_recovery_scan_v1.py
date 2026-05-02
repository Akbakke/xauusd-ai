#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    WEDNESDAY_R6_BENCHMARK,
    _asof_runner_guard,
    _r6_candidate_grid,
    _r6_policy_mask,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "PARALLEL_MONDAY_R6_RECALL_RECOVERY_SCAN_V1"

SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*CONTRACT_FIX_R5_R51_R52*"
R6_GLOB = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_*CONTRACT_FIX_R6_FROM_FIXED_R52*"

SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
SCORE_SUMMARY = "summary_v1.json"
R6_TRAINING_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
R6_PREDICTION_VIEW = "monday_r6_on_foundation_scores_prediction_view_v1.parquet"
R6_GRID = "r6_family_grid_replay_v1.csv"
R6_SUMMARY = "summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"

PARALLEL_SCAN_ORCHESTRATOR = "parallel_scan_orchestrator_v1.json"
LANE_01 = "lane_01_r6_threshold_frontier_scan_v1.csv"
LANE_02 = "lane_02_r5_2_base_extension_v2_scan_v1.csv"
LANE_03 = "lane_03_r5_r5_1_r5_2_union_with_strong_guard_scan_v1.csv"
LANE_04 = "lane_04_tail_control_10_50_recovery_scan_v1.csv"
LANE_05 = "lane_05_bad_risk_recall_recovery_scan_v1.csv"
LANE_06 = "lane_06_runner_guard_sensitivity_scan_v1.csv"
LANE_07 = "lane_07_candidate_grid_rejected_safe_recall_scan_v1.csv"
LANE_08 = "lane_08_batch_loso_stability_scan_v1.csv"
LANE_09 = "lane_09_score_calibration_diagnostic_scan_v1.csv"
LANE_10 = "lane_10_high_mfe_protection_stress_scan_v1.csv"
AGGREGATOR = "parallel_scan_aggregator_v1.json"
LEADERBOARD = "parallel_scan_leaderboard_v1.csv"
NEXT_ACTION = "next_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
AUDIT = "consistency_audit_v1.csv"

OUTPUT_FILES = {
    "parallel_scan_orchestrator": PARALLEL_SCAN_ORCHESTRATOR,
    "lane_01_r6_threshold_frontier_scan": LANE_01,
    "lane_02_r5_2_base_extension_v2_scan": LANE_02,
    "lane_03_r5_r5_1_r5_2_union_with_strong_guard_scan": LANE_03,
    "lane_04_tail_control_10_50_recovery_scan": LANE_04,
    "lane_05_bad_risk_recall_recovery_scan": LANE_05,
    "lane_06_runner_guard_sensitivity_scan": LANE_06,
    "lane_07_candidate_grid_rejected_safe_recall_scan": LANE_07,
    "lane_08_batch_loso_stability_scan": LANE_08,
    "lane_09_score_calibration_diagnostic_scan": LANE_09,
    "lane_10_high_mfe_protection_stress_scan": LANE_10,
    "parallel_scan_aggregator": AGGREGATOR,
    "parallel_scan_leaderboard": LEADERBOARD,
    "next_action_lock": NEXT_ACTION,
    "summary": SUMMARY,
    "report": REPORT,
    "manifest": MANIFEST,
    "status": STATUS,
    "audit": AUDIT,
}

EXPECTED_ROW_COUNT = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_AS_OF_COLUMNS = 109
FORBIDDEN_ROW_COUNTS = {1689, 1852}
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

SCORE_COLUMNS = [
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
    "r5_1_bad_blocker_score_v1",
    "r5_1_runner_guard_score_v1",
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    R6_RISKY_PROB,
    R6_BLINDSPOT_PROB,
]

LANE_FILES = {
    "LANE_01_R6_THRESHOLD_FRONTIER_SCAN_V1": LANE_01,
    "LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1": LANE_02,
    "LANE_03_R5_R5_1_R5_2_UNION_WITH_STRONG_GUARD_SCAN_V1": LANE_03,
    "LANE_04_TAIL_CONTROL_10_50_RECOVERY_SCAN_V1": LANE_04,
    "LANE_05_BAD_RISK_RECALL_RECOVERY_SCAN_V1": LANE_05,
    "LANE_06_RUNNER_GUARD_SENSITIVITY_SCAN_V1": LANE_06,
    "LANE_07_CANDIDATE_GRID_REJECTED_SAFE_RECALL_SCAN_V1": LANE_07,
    "LANE_08_BATCH_LOSO_STABILITY_SCAN_V1": LANE_08,
    "LANE_09_SCORE_CALIBRATION_DIAGNOSTIC_SCAN_V1": LANE_09,
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _latest_dir(reports_root: Path, pattern: str, required_file: str) -> Path:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    if not dirs:
        raise FileNotFoundError(f"No {pattern} with {required_file} under {reports_root}")
    return dirs[-1]


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes"]).fillna(default).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _nan_to_low(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=-1.0)


@dataclass(frozen=True)
class ScanContext:
    frame: pd.DataFrame
    current_mask: np.ndarray
    current_metrics: dict[str, Any]
    current_worst_loso: float | None
    current_runner_near_miss_limit: int
    forensic_present: bool
    should: np.ndarray
    take_ok: np.ndarray
    tail: np.ndarray
    fifty: np.ndarray
    hundred: np.ndarray
    two_hundred: np.ndarray
    strongest: np.ndarray
    repaired: np.ndarray
    repaired_any: np.ndarray
    near_miss: np.ndarray
    quarantine: np.ndarray
    forensic: np.ndarray
    run_ids: np.ndarray
    batch_ids: np.ndarray
    split_ids: np.ndarray
    r5_selected: np.ndarray
    r5_1_selected: np.ndarray
    r5_2_selected: np.ndarray
    asof_guard: np.ndarray
    scores: dict[str, np.ndarray]


def _worst_precision_from_arrays(ctx: ScanContext, mask: np.ndarray) -> float | None:
    values: list[float] = []
    for run_id in np.unique(ctx.run_ids):
        run_scope = ctx.run_ids == run_id
        selected = mask & run_scope
        block_count = int(selected.sum())
        if block_count:
            values.append(float((selected & ctx.should).sum() / block_count))
    return min(values) if values else None


def _metrics(ctx: ScanContext, mask: np.ndarray, *, compute_worst: bool = True) -> tuple[dict[str, Any], float | None]:
    selected = np.asarray(mask, dtype=bool)
    block_count = int(selected.sum())
    bad_blocks = int((selected & ctx.should).sum())
    precision = float(bad_blocks / block_count) if block_count else None
    metrics = {
        "row_count_v1": int(len(selected)),
        "block_count_v1": block_count,
        "bad_blocks_v1": bad_blocks,
        "tail_help_v1": int((selected & ctx.tail).sum()),
        "precision_v1": precision,
        "false_take_ok_blocks_v1": int((selected & ctx.take_ok).sum()),
        "fifty_plus_mfe_blocked_v1": int((selected & ctx.fifty).sum()),
        "hundred_plus_mfe_blocked_v1": int((selected & ctx.hundred).sum()),
        "two_hundred_plus_mfe_blocked_v1": int((selected & ctx.two_hundred).sum()),
        "strongest_winner_damage_v1": int((selected & ctx.strongest).sum()),
        "repaired_165_damage_v1": int((selected & ctx.repaired).sum()),
        "is_repaired_165_blocked_v1": int((selected & ctx.repaired_any).sum()),
        "forensic_repaired_trade_blocked_v1": int((selected & ctx.forensic).sum()),
        "quarantine_blocks_v1": int((selected & ctx.quarantine).sum()),
        "runner_near_miss_blocked_v1": int((selected & ctx.near_miss).sum()),
    }
    basic = block_count > 0 and precision is not None and precision >= WEDNESDAY_R6_BENCHMARK["precision_v1"]
    hard = _hard_safety_failures(metrics, ctx.current_runner_near_miss_limit)
    worst_loso = _worst_precision_from_arrays(ctx, selected) if compute_worst and basic and not hard else None
    return metrics, worst_loso


def _hard_safety_failures(metrics: dict[str, Any], runner_near_miss_limit: int) -> list[str]:
    failures: list[str] = []
    if int(metrics["repaired_165_damage_v1"]) > 0:
        failures.append("repaired_165_damage_v1>0")
    if int(metrics["forensic_repaired_trade_blocked_v1"]) > 0:
        failures.append("forensic_repaired_trade_blocked_v1>0")
    if int(metrics["fifty_plus_mfe_blocked_v1"]) > WEDNESDAY_R6_BENCHMARK["fifty_plus_mfe_blocked_v1"]:
        failures.append("fifty_plus_mfe_blocked_v1>wednesday")
    if int(metrics["hundred_plus_mfe_blocked_v1"]) > 0:
        failures.append("hundred_plus_mfe_blocked_v1>0")
    if int(metrics["two_hundred_plus_mfe_blocked_v1"]) > 0:
        failures.append("two_hundred_plus_mfe_blocked_v1>0")
    if int(metrics["strongest_winner_damage_v1"]) > 0:
        failures.append("strongest_winner_damage_v1>0")
    if int(metrics["runner_near_miss_blocked_v1"]) > runner_near_miss_limit:
        failures.append("runner_near_miss_regression")
    return failures


def _strict_safety_failures(ctx: ScanContext, metrics: dict[str, Any], worst_loso: float | None) -> list[str]:
    failures = _hard_safety_failures(metrics, ctx.current_runner_near_miss_limit)
    precision = metrics.get("precision_v1")
    if precision is None or float(precision) < WEDNESDAY_R6_BENCHMARK["precision_v1"]:
        failures.append("precision_below_wednesday_r6")
    if worst_loso is None or float(worst_loso) < WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]:
        failures.append("worst_loso_below_wednesday_r6")
    return failures


def _record(
    ctx: ScanContext,
    *,
    lane: str,
    rule_id: str,
    mask: np.ndarray,
    params: dict[str, Any],
    implementability: str,
    requires_retrain: bool = False,
    extra: dict[str, Any] | None = None,
    compute_worst: bool = True,
) -> dict[str, Any]:
    metrics, worst_loso = _metrics(ctx, mask, compute_worst=compute_worst)
    hard_failures = _hard_safety_failures(metrics, ctx.current_runner_near_miss_limit)
    strict_failures = _strict_safety_failures(ctx, metrics, worst_loso)
    payload = {
        "lane_v1": lane,
        "rule_id_v1": rule_id,
        "params_json_v1": json.dumps(_jsonable(params), sort_keys=True),
        "requires_retrain_v1": bool(requires_retrain),
        "implementability_v1": implementability,
        "hard_safety_pass_v1": not hard_failures,
        "wednesday_safety_pass_v1": not strict_failures,
        "hard_safety_failures_v1": "|".join(hard_failures),
        "strict_safety_failures_v1": "|".join(strict_failures),
        "worst_loso_v1": worst_loso,
        "bad_uplift_vs_current_v1": int(metrics["bad_blocks_v1"] - ctx.current_metrics["bad_blocks_v1"]),
        "tail_uplift_vs_current_v1": int(metrics["tail_help_v1"] - ctx.current_metrics["tail_help_v1"]),
        **metrics,
    }
    if extra:
        payload.update(extra)
    return payload


def _top_frontier(records: list[dict[str, Any]], *, limit: int = 1000) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["_safe_sort"] = frame["wednesday_safety_pass_v1"].astype(bool).astype(int)
    frame["_hard_sort"] = frame["hard_safety_pass_v1"].astype(bool).astype(int)
    ordered = frame.sort_values(
        [
            "_safe_sort",
            "_hard_sort",
            "bad_blocks_v1",
            "tail_help_v1",
            "precision_v1",
            "worst_loso_v1",
            "block_count_v1",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).head(limit)
    return ordered.drop(columns=["_safe_sort", "_hard_sort"])


def _make_context(frame: pd.DataFrame) -> ScanContext:
    current_mask = _bool(frame, "selected_candidate_block_v1").to_numpy(dtype=bool)
    empty_ctx = object.__new__(ScanContext)
    should = _bool(frame, "label_should_not_take_v1").to_numpy(dtype=bool)
    take_ok = _bool(frame, "take_was_ok_v1").to_numpy(dtype=bool)
    tail = _bool(frame, "tail_10_50_mfe_v1").to_numpy(dtype=bool)
    fifty = _bool(frame, "fifty_plus_mfe_v1").to_numpy(dtype=bool)
    hundred = _bool(frame, "hundred_plus_mfe_v1").to_numpy(dtype=bool)
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1").to_numpy(dtype=bool)
    strongest = _bool(frame, "strongest_winner_path_v1").to_numpy(dtype=bool)
    repaired = _bool(frame, "r6_label_repaired_165_like_runner_v1").to_numpy(dtype=bool)
    repaired_any = _bool(frame, "is_repaired_165_v1").to_numpy(dtype=bool)
    near_miss = _bool(frame, "r6_label_runner_near_miss_v1").to_numpy(dtype=bool)
    quarantine = ~frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE").to_numpy(dtype=bool)
    forensic = frame["candidate_uid"].astype("string").eq(FORENSIC_REPAIRED_CANDIDATE_UID).to_numpy(dtype=bool)
    asof_guard = _bool(frame, "asof_runner_guard_v1").to_numpy(dtype=bool)
    if not asof_guard.any():
        asof_guard = _asof_runner_guard(frame).to_numpy(dtype=bool)
    scores = {column: _nan_to_low(_num(frame, column)) for column in SCORE_COLUMNS}
    for column in ["blocker_score_v1", "runner_protector_score_v1", "mae_abs_bps_v1", "peak_mfe_bps_v1"]:
        scores[column] = _nan_to_low(_num(frame, column))
    ctx = ScanContext(
        frame=frame,
        current_mask=current_mask,
        current_metrics={},
        current_worst_loso=None,
        current_runner_near_miss_limit=0,
        forensic_present=bool(forensic.any()),
        should=should,
        take_ok=take_ok,
        tail=tail,
        fifty=fifty,
        hundred=hundred,
        two_hundred=two_hundred,
        strongest=strongest,
        repaired=repaired,
        repaired_any=repaired_any,
        near_miss=near_miss,
        quarantine=quarantine,
        forensic=forensic,
        run_ids=frame["run_id"].astype("string").fillna("").to_numpy(dtype=str),
        batch_ids=frame.get("batch_scope_v1", pd.Series("BATCH_UNKNOWN", index=frame.index)).astype("string").fillna("BATCH_UNKNOWN").to_numpy(dtype=str),
        split_ids=frame.get("split_scope_v1", pd.Series("UNKNOWN", index=frame.index)).astype("string").fillna("UNKNOWN").to_numpy(dtype=str),
        r5_selected=_bool(frame, "r5_selected_candidate__block_v1").to_numpy(dtype=bool),
        r5_1_selected=_bool(frame, "r5_1_selected_candidate__block_v1").to_numpy(dtype=bool),
        r5_2_selected=_bool(frame, "r5_2_selected_candidate__block_v1").to_numpy(dtype=bool),
        asof_guard=asof_guard,
        scores=scores,
    )
    current_metrics, current_worst = _metrics(ctx, current_mask, compute_worst=True)
    return ScanContext(
        **{
            **ctx.__dict__,
            "current_metrics": current_metrics,
            "current_worst_loso": current_worst,
            "current_runner_near_miss_limit": int(current_metrics["runner_near_miss_blocked_v1"]),
        }
    )


def _values(full: list[float], quick: list[float], quick_scan: bool) -> list[float]:
    return quick if quick_scan else full


def _lane_01(ctx: ScanContext, *, quick_scan: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    bad_values = _values([0.35, 0.50, 0.65, 0.80, 0.90, 0.97], [0.55, 0.85], quick_scan)
    risky_values = _values([0.50, 0.70, 0.85, 0.95], [0.65, 0.90], quick_scan)
    tail_values = _values([0.40, 0.65, 0.85, 0.95], [0.50, 0.90], quick_scan)
    runner_values = _values([0.30, 0.50, 0.70, 0.85], [0.45, 0.74], quick_scan)
    r5_2_runner_values = _values([0.35, 0.60, 0.74, 0.90], [0.45, 0.74], quick_scan)
    blind_values = _values([0.70, 0.90], [0.70], quick_scan)
    bad = ctx.scores[R6_BAD_PROB]
    risky = ctx.scores[R6_RISKY_PROB]
    tail = ctx.scores[R6_TAIL_PROB]
    runner = ctx.scores[R6_RUNNER_PROB]
    r5_2_runner = ctx.scores[R5_2_RUNNER_PROB]
    blind = ctx.scores[R6_BLINDSPOT_PROB]
    base = ctx.r5_2_selected
    formulas: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
        "BAD_RISKY_TAIL": lambda b, r, t, bl: b & r & t,
        "BAD_RISKY": lambda b, r, t, bl: b & r,
        "BAD_TAIL": lambda b, r, t, bl: b & t,
        "BAD_RISKY_TAIL_OR_BLINDSPOT": lambda b, r, t, bl: (b & r & t) | (b & bl),
    }
    for bad_t in bad_values:
        bad_signal = bad >= bad_t
        for risky_t in risky_values:
            risky_signal = risky >= risky_t
            for tail_t in tail_values:
                tail_signal = tail >= tail_t
                for runner_t in runner_values:
                    for r5_2_runner_t in r5_2_runner_values:
                        protect = (runner >= runner_t) | (r5_2_runner >= r5_2_runner_t) | ctx.asof_guard
                        for blind_t in blind_values:
                            blind_signal = blind >= blind_t
                            for formula_id, formula in formulas.items():
                                addon = formula(bad_signal, risky_signal, tail_signal, blind_signal) & ~protect
                                mask = base | addon
                                records.append(
                                    _record(
                                        ctx,
                                        lane="LANE_01_R6_THRESHOLD_FRONTIER_SCAN_V1",
                                        rule_id=formula_id,
                                        mask=mask,
                                        params={
                                            "bad_threshold_v1": bad_t,
                                            "risky_threshold_v1": risky_t,
                                            "tail_threshold_v1": tail_t,
                                            "runner_threshold_v1": runner_t,
                                            "r5_2_runner_threshold_v1": r5_2_runner_t,
                                            "blindspot_threshold_v1": blind_t,
                                        },
                                        implementability="THRESHOLD_OR_CANDIDATE_SELECTION_ONLY",
                                    )
                                )
    return _top_frontier(records, limit=1000)


def _lane_02(ctx: ScanContext, *, quick_scan: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    bad_sources = {
        "R5_2_BAD": ctx.scores[R5_2_BAD_PROB],
        "BLOCKER_SCORE": ctx.scores["blocker_score_v1"],
        "R5_1_BAD": ctx.scores["r5_1_bad_blocker_score_v1"],
        "R5_BAD": ctx.scores["pred__entry_r5_should_not_take__prob_true_v1"],
        "MAX_R5_R51_R52_BAD": np.maximum.reduce(
            [
                ctx.scores[R5_2_BAD_PROB],
                ctx.scores["r5_1_bad_blocker_score_v1"],
                ctx.scores["pred__entry_r5_should_not_take__prob_true_v1"],
            ]
        ),
    }
    bad_values = _values([0.35, 0.50, 0.65, 0.80, 0.92], [0.45, 0.75], quick_scan)
    mae_values = _values([0.50, 0.65, 0.75, 0.85], [0.50, 0.75], quick_scan)
    r5_runner_values = _values([0.45, 0.65, 0.85], [0.60, 0.85], quick_scan)
    r5_1_runner_values = _values([0.45, 0.65, 0.85], [0.60, 0.85], quick_scan)
    r5_2_runner_values = _values([0.35, 0.60, 0.74], [0.35, 0.74], quick_scan)
    immediate_mae = ctx.scores["pred__entry_r5_immediate_MAE_risk__prob_true_v1"]
    r5_runner = ctx.scores["pred__entry_r5_runner_protect__prob_true_v1"]
    r5_1_runner = ctx.scores["r5_1_runner_guard_score_v1"]
    r5_2_runner = ctx.scores[R5_2_RUNNER_PROB]
    base = ctx.r5_2_selected
    for source_name, bad_score in bad_sources.items():
        for bad_t in bad_values:
            for mae_t in mae_values:
                for r5_runner_t in r5_runner_values:
                    for r5_1_runner_t in r5_1_runner_values:
                        for r5_2_runner_t in r5_2_runner_values:
                            for exclude_asof in [True, False]:
                                extension = (
                                    (bad_score >= bad_t)
                                    & (immediate_mae >= mae_t)
                                    & (r5_runner < r5_runner_t)
                                    & (r5_1_runner < r5_1_runner_t)
                                    & (r5_2_runner < r5_2_runner_t)
                                )
                                if exclude_asof:
                                    extension = extension & ~ctx.asof_guard
                                added = extension & ~base
                                mask = base | extension
                                records.append(
                                    _record(
                                        ctx,
                                        lane="LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1",
                                        rule_id=f"BASE_EXTENSION_{source_name}",
                                        mask=mask,
                                        params={
                                            "bad_source_v1": source_name,
                                            "bad_threshold_v1": bad_t,
                                            "immediate_mae_threshold_v1": mae_t,
                                            "r5_runner_max_v1": r5_runner_t,
                                            "r5_1_runner_max_v1": r5_1_runner_t,
                                            "r5_2_runner_max_v1": r5_2_runner_t,
                                            "exclude_asof_runner_guard_v1": exclude_asof,
                                        },
                                        implementability="R5_2_BASE_MEMBERSHIP_CONTRACT_ONLY",
                                        extra={
                                            "rows_added_v1": int(added.sum()),
                                            "added_bad_rows_v1": int((added & ctx.should).sum()),
                                            "added_tail_rows_v1": int((added & ctx.tail).sum()),
                                            "added_take_ok_rows_v1": int((added & ctx.take_ok).sum()),
                                            "added_fifty_plus_rows_v1": int((added & ctx.fifty).sum()),
                                            "added_strongest_winner_rows_v1": int((added & ctx.strongest).sum()),
                                        },
                                    )
                                )
    return _top_frontier(records, limit=1000)


def _lane_03(ctx: ScanContext, *, quick_scan: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    union_sources = {
        "R5_OR_R51_OR_R52_SELECTED": ctx.r5_selected | ctx.r5_1_selected | ctx.r5_2_selected,
        "R51_OR_R52_SELECTED": ctx.r5_1_selected | ctx.r5_2_selected,
        "R52_SELECTED_PLUS_R5_BAD_SCORE": ctx.r5_2_selected | (ctx.scores["pred__entry_r5_should_not_take__prob_true_v1"] >= 0.85),
    }
    thresholds = _values([0.30, 0.45, 0.60, 0.75, 0.85], [0.45, 0.75], quick_scan)
    for source_name, union in union_sources.items():
        for r5_runner_t in thresholds:
            for r5_1_runner_t in thresholds:
                for r5_2_runner_t in thresholds:
                    for r6_runner_t in thresholds:
                        for use_asof in [True, False]:
                            protect = (
                                (ctx.scores["pred__entry_r5_runner_protect__prob_true_v1"] >= r5_runner_t)
                                | (ctx.scores["r5_1_runner_guard_score_v1"] >= r5_1_runner_t)
                                | (ctx.scores[R5_2_RUNNER_PROB] >= r5_2_runner_t)
                                | (ctx.scores[R6_RUNNER_PROB] >= r6_runner_t)
                            )
                            if use_asof:
                                protect = protect | ctx.asof_guard
                            stopped = union & protect
                            mask = union & ~protect
                            records.append(
                                _record(
                                    ctx,
                                    lane="LANE_03_R5_R5_1_R5_2_UNION_WITH_STRONG_GUARD_SCAN_V1",
                                    rule_id=f"UNION_STRICT_GUARD_{source_name}",
                                    mask=mask,
                                    params={
                                        "union_source_v1": source_name,
                                        "r5_runner_threshold_v1": r5_runner_t,
                                        "r5_1_runner_threshold_v1": r5_1_runner_t,
                                        "r5_2_runner_threshold_v1": r5_2_runner_t,
                                        "r6_runner_threshold_v1": r6_runner_t,
                                        "use_asof_guard_v1": use_asof,
                                    },
                                    implementability="BASE_UNION_CONTRACT_AND_GUARD_CHANGE",
                                    extra={
                                        "union_rows_v1": int(union.sum()),
                                        "guard_stopped_rows_v1": int(stopped.sum()),
                                        "guard_stopped_winner_rows_v1": int((stopped & (ctx.fifty | ctx.strongest)).sum()),
                                        "unsafe_rows_slipped_through_v1": int((mask & (ctx.hundred | ctx.two_hundred | ctx.strongest)).sum()),
                                    },
                                )
                            )
    return _top_frontier(records, limit=1000)


def _lane_04(ctx: ScanContext, *, quick_scan: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    tail_sources = {
        "R6_TAIL": ctx.scores[R6_TAIL_PROB],
        "R5_TAIL": ctx.scores["pred__entry_r5_tail_control_10_50_risk__prob_true_v1"],
        "MAX_R5_R6_TAIL": np.maximum(ctx.scores[R6_TAIL_PROB], ctx.scores["pred__entry_r5_tail_control_10_50_risk__prob_true_v1"]),
    }
    tail_values = _values([0.25, 0.40, 0.55, 0.70, 0.85, 0.95], [0.55, 0.85], quick_scan)
    bad_values = _values([0.25, 0.45, 0.65, 0.85], [0.45, 0.85], quick_scan)
    runner_values = _values([0.30, 0.45, 0.60, 0.74, 0.85], [0.45, 0.74], quick_scan)
    for source_name, tail_score in tail_sources.items():
        for tail_t in tail_values:
            tail_signal = tail_score >= tail_t
            for bad_t in bad_values:
                bad_signal = np.maximum(ctx.scores[R6_BAD_PROB], ctx.scores[R5_2_BAD_PROB]) >= bad_t
                for runner_t in runner_values:
                    for require_bad in [True, False]:
                        for use_asof in [True, False]:
                            protect = (ctx.scores[R6_RUNNER_PROB] >= runner_t) | (ctx.scores[R5_2_RUNNER_PROB] >= runner_t)
                            if use_asof:
                                protect = protect | ctx.asof_guard
                            signal = tail_signal & (bad_signal if require_bad else True)
                            mask = ctx.r5_2_selected | (signal & ~protect)
                            missed_tail = ctx.tail & ~ctx.current_mask
                            records.append(
                                _record(
                                    ctx,
                                    lane="LANE_04_TAIL_CONTROL_10_50_RECOVERY_SCAN_V1",
                                    rule_id=f"TAIL_RECOVERY_{source_name}",
                                    mask=mask,
                                    params={
                                        "tail_source_v1": source_name,
                                        "tail_threshold_v1": tail_t,
                                        "bad_threshold_v1": bad_t,
                                        "runner_threshold_v1": runner_t,
                                        "require_bad_signal_v1": require_bad,
                                        "use_asof_guard_v1": use_asof,
                                    },
                                    implementability="TAIL_THRESHOLD_OR_CONTRACT_CHANGE",
                                    extra={
                                        "missed_tail_population_v1": int(missed_tail.sum()),
                                        "missed_tail_recovered_v1": int((mask & missed_tail).sum()),
                                        "unsafe_tail_like_runner_rows_v1": int((signal & ~protect & ctx.fifty).sum()),
                                        "tail_rows_protected_by_guard_v1": int((signal & protect & ctx.tail).sum()),
                                    },
                                )
                            )
    return _top_frontier(records, limit=1000)


def _lane_05(ctx: ScanContext, *, quick_scan: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    bad_sources = {
        "R6_BAD": ctx.scores[R6_BAD_PROB],
        "R5_2_BAD": ctx.scores[R5_2_BAD_PROB],
        "R5_1_BAD": ctx.scores["r5_1_bad_blocker_score_v1"],
        "R5_BAD": ctx.scores["pred__entry_r5_should_not_take__prob_true_v1"],
        "MAX_BAD": np.maximum.reduce(
            [
                ctx.scores[R6_BAD_PROB],
                ctx.scores[R5_2_BAD_PROB],
                ctx.scores["r5_1_bad_blocker_score_v1"],
                ctx.scores["pred__entry_r5_should_not_take__prob_true_v1"],
            ]
        ),
    }
    bad_values = _values([0.35, 0.50, 0.65, 0.80, 0.92], [0.50, 0.90], quick_scan)
    runner_values = _values([0.35, 0.50, 0.74, 0.85], [0.45, 0.74], quick_scan)
    risky_values = _values([0.35, 0.65, 0.85], [0.50, 0.90], quick_scan)
    missed_bad = ctx.should & ~ctx.current_mask
    for source_name, bad_score in bad_sources.items():
        for bad_t in bad_values:
            signal_bad = bad_score >= bad_t
            for risky_t in risky_values:
                risky_signal = ctx.scores[R6_RISKY_PROB] >= risky_t
                for runner_t in runner_values:
                    for require_risky in [True, False]:
                        for use_asof in [True, False]:
                            protect = (
                                (ctx.scores[R6_RUNNER_PROB] >= runner_t)
                                | (ctx.scores[R5_2_RUNNER_PROB] >= runner_t)
                                | (ctx.scores["pred__entry_r5_runner_protect__prob_true_v1"] >= runner_t)
                            )
                            if use_asof:
                                protect = protect | ctx.asof_guard
                            signal = signal_bad & (risky_signal if require_risky else True)
                            mask = ctx.r5_2_selected | (signal & ~protect)
                            records.append(
                                _record(
                                    ctx,
                                    lane="LANE_05_BAD_RISK_RECALL_RECOVERY_SCAN_V1",
                                    rule_id=f"BAD_RISK_RECOVERY_{source_name}",
                                    mask=mask,
                                    params={
                                        "bad_source_v1": source_name,
                                        "bad_threshold_v1": bad_t,
                                        "risky_threshold_v1": risky_t,
                                        "runner_threshold_v1": runner_t,
                                        "require_risky_v1": require_risky,
                                        "use_asof_guard_v1": use_asof,
                                    },
                                    implementability="BAD_RISK_THRESHOLD_OR_CONTRACT_CHANGE",
                                    extra={
                                        "missed_bad_population_v1": int(missed_bad.sum()),
                                        "missed_bad_recovered_v1": int((mask & missed_bad).sum()),
                                        "missed_bad_below_signal_v1": int((missed_bad & ~signal).sum()),
                                        "missed_bad_protected_by_guard_v1": int((missed_bad & signal & protect).sum()),
                                        "dangerous_bad_like_rows_v1": int((signal & ~protect & (ctx.fifty | ctx.strongest)).sum()),
                                    },
                                )
                            )
    return _top_frontier(records, limit=1000)


def _selected_family_params(grid: pd.DataFrame) -> dict[str, Any]:
    if "wednesday_safety_pass_v1" in grid.columns:
        safe = grid[grid["wednesday_safety_pass_v1"].astype(bool)].copy()
        if not safe.empty:
            row = safe.sort_values(["bad_blocks_v1", "tail_help_v1", "precision_v1"], ascending=[False, False, False], na_position="last").iloc[0]
            return row.to_dict()
    row = grid.sort_values(["hard_damage_count_v1", "precision_v1", "bad_blocks_v1"], ascending=[True, False, False], na_position="last").iloc[0]
    return row.to_dict()


def _lane_06(ctx: ScanContext, grid: pd.DataFrame, *, quick_scan: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    selected = _selected_family_params(grid)
    bad_t = float(selected.get("bad_threshold_v1") or 0.85)
    risky_t = float(selected.get("risky_threshold_v1") or 0.85)
    tail_t = float(selected.get("tail_threshold_v1") or 0.85)
    base_signal = ctx.r5_2_selected | (
        (ctx.scores[R6_BAD_PROB] >= bad_t)
        & (ctx.scores[R6_RISKY_PROB] >= risky_t)
        & (ctx.scores[R6_TAIL_PROB] >= tail_t)
    )
    runner_values = _values([0.30, 0.45, 0.60, 0.74, 0.90], [0.45, 0.85], quick_scan)
    quality_values = _values([0.70, 0.80, 0.90], [0.70], quick_scan)
    tradable_values = _values([0.84, 0.90, 0.94], [0.84, 0.94], quick_scan)
    for r6_runner_t in runner_values:
        for r5_2_runner_t in runner_values:
            for tradable_t in tradable_values:
                for quality_t in quality_values:
                    for use_asof in [True, False]:
                        custom_asof = (
                            (_num(ctx.frame, "as_of_candidate_tradable_prob_v1").to_numpy(dtype=float) >= tradable_t)
                            & (_num(ctx.frame, "as_of_entry_candidate_path_quality_pred_v1").to_numpy(dtype=float) >= quality_t)
                            & (_num(ctx.frame, "as_of_candidate_mfe_first_n_pred_v1").to_numpy(dtype=float) >= 1.75)
                            & (_num(ctx.frame, "as_of_skip_candidate_p_flat_v1").to_numpy(dtype=float) <= 0.50)
                        )
                        protect = (ctx.scores[R6_RUNNER_PROB] >= r6_runner_t) | (ctx.scores[R5_2_RUNNER_PROB] >= r5_2_runner_t)
                        if use_asof:
                            protect = protect | custom_asof
                        stopped = base_signal & protect
                        mask = base_signal & ~protect
                        records.append(
                            _record(
                                ctx,
                                lane="LANE_06_RUNNER_GUARD_SENSITIVITY_SCAN_V1",
                                rule_id="RUNNER_GUARD_SENSITIVITY",
                                mask=mask,
                                params={
                                    "source_policy_v1": selected.get("policy_name_v1"),
                                    "bad_threshold_v1": bad_t,
                                    "risky_threshold_v1": risky_t,
                                    "tail_threshold_v1": tail_t,
                                    "r6_runner_threshold_v1": r6_runner_t,
                                    "r5_2_runner_threshold_v1": r5_2_runner_t,
                                    "asof_tradable_min_v1": tradable_t,
                                    "asof_quality_min_v1": quality_t,
                                    "use_asof_guard_v1": use_asof,
                                },
                                implementability="GUARD_CONTRACT_CHANGE",
                                extra={
                                    "guard_stopped_bad_rows_v1": int((stopped & ctx.should).sum()),
                                    "guard_stopped_tail_rows_v1": int((stopped & ctx.tail).sum()),
                                    "protected_winner_rows_v1": int((stopped & (ctx.fifty | ctx.strongest)).sum()),
                                    "unsafe_relaxation_cases_v1": int((mask & (ctx.hundred | ctx.two_hundred | ctx.strongest)).sum()),
                                },
                            )
                        )
    return _top_frontier(records, limit=1000)


def _lane_07(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["rejection_reason_v1"] = "NOT_REJECTED_SAFE_CANDIDATE"
    out.loc[~out.get("wednesday_basic_safety_pass_v1", pd.Series(False, index=out.index)).astype(bool), "rejection_reason_v1"] = "BASIC_SAFETY_FAIL"
    out.loc[(out["hard_damage_count_v1"] > 0), "rejection_reason_v1"] = "HARD_DAMAGE_FAIL"
    out.loc[pd.to_numeric(out.get("precision_v1"), errors="coerce").lt(WEDNESDAY_R6_BENCHMARK["precision_v1"]), "rejection_reason_v1"] = "PRECISION_FAIL"
    out.loc[pd.to_numeric(out.get("worst_loso_v1"), errors="coerce").lt(WEDNESDAY_R6_BENCHMARK["worst_loso_v1"]), "rejection_reason_v1"] = "WORST_LOSO_FAIL"
    out.loc[out.get("wednesday_safety_pass_v1", pd.Series(False, index=out.index)).astype(bool), "rejection_reason_v1"] = "SAFE_REJECTED_OR_NOT_SELECTED"
    out["lane_v1"] = "LANE_07_CANDIDATE_GRID_REJECTED_SAFE_RECALL_SCAN_V1"
    out["requires_retrain_v1"] = False
    out["implementability_v1"] = "CANDIDATE_SELECTION_ONLY"
    return out.sort_values(
        ["wednesday_safety_pass_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )


def _mask_for_candidate_name(frame: pd.DataFrame, policy_name: str) -> np.ndarray | None:
    for candidate in _r6_candidate_grid(compact=False):
        if candidate.policy_name == policy_name:
            return _r6_policy_mask(frame, candidate).to_numpy(dtype=bool)
    return None


def _slice_rows(ctx: ScanContext, mask: np.ndarray, *, policy_name: str, scope_name: str, ids: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(set(ids.tolist())):
        scope = ids == value
        scoped_mask = mask & scope
        block = int(scoped_mask.sum())
        bad = int((scoped_mask & ctx.should).sum())
        rows.append(
            {
                "lane_v1": "LANE_08_BATCH_LOSO_STABILITY_SCAN_V1",
                "policy_name_v1": policy_name,
                "scope_type_v1": scope_name,
                "scope_value_v1": str(value),
                "row_count_v1": int(scope.sum()),
                "bad_population_v1": int((scope & ctx.should).sum()),
                "tail_population_v1": int((scope & ctx.tail).sum()),
                "block_count_v1": block,
                "bad_blocks_v1": bad,
                "tail_help_v1": int((scoped_mask & ctx.tail).sum()),
                "precision_v1": float(bad / block) if block else None,
                "zero_selected_bad_blocks_v1": bool(int((scope & ctx.should).sum()) > 0 and bad == 0),
                "fifty_plus_mfe_blocked_v1": int((scoped_mask & ctx.fifty).sum()),
                "strongest_winner_damage_v1": int((scoped_mask & ctx.strongest).sum()),
            }
        )
    return rows


def _lane_08(ctx: ScanContext, grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_slice_rows(ctx, ctx.current_mask, policy_name="CURRENT_SELECTED_POLICY", scope_name="SPLIT", ids=ctx.split_ids))
    rows.extend(_slice_rows(ctx, ctx.current_mask, policy_name="CURRENT_SELECTED_POLICY", scope_name="BATCH", ids=ctx.batch_ids))
    rows.extend(_slice_rows(ctx, ctx.current_mask, policy_name="CURRENT_SELECTED_POLICY", scope_name="LOSO_RUN", ids=ctx.run_ids))
    for label, subset in {
        "TOP_WEDNESDAY_SAFE_GRID_POLICY": grid[grid["wednesday_safety_pass_v1"].astype(bool)].copy() if "wednesday_safety_pass_v1" in grid else pd.DataFrame(),
        "TOP_ALL_GRID_RECALL_POLICY": grid.copy(),
    }.items():
        if subset.empty:
            continue
        row = subset.sort_values(["bad_blocks_v1", "tail_help_v1", "precision_v1"], ascending=[False, False, False], na_position="last").iloc[0]
        mask = _mask_for_candidate_name(ctx.frame, str(row["policy_name_v1"]))
        if mask is None:
            continue
        rows.extend(_slice_rows(ctx, mask, policy_name=f"{label}:{row['policy_name_v1']}", scope_name="SPLIT", ids=ctx.split_ids))
        rows.extend(_slice_rows(ctx, mask, policy_name=f"{label}:{row['policy_name_v1']}", scope_name="BATCH", ids=ctx.batch_ids))
        rows.extend(_slice_rows(ctx, mask, policy_name=f"{label}:{row['policy_name_v1']}", scope_name="LOSO_RUN", ids=ctx.run_ids))
    return pd.DataFrame(rows)


def _lane_09(ctx: ScanContext) -> pd.DataFrame:
    groups = {
        "ALL": np.ones(len(ctx.frame), dtype=bool),
        "BAD": ctx.should,
        "MISSED_BAD": ctx.should & ~ctx.current_mask,
        "SELECTED_BAD": ctx.should & ctx.current_mask,
        "TAIL": ctx.tail,
        "MISSED_TAIL": ctx.tail & ~ctx.current_mask,
        "TAKE_OK": ctx.take_ok,
        "HIGH_MFE_50_PLUS": ctx.fifty,
        "STRONGEST_WINNER": ctx.strongest,
        "RUNNER_NEAR_MISS": ctx.near_miss,
    }
    rows: list[dict[str, Any]] = []
    for score_col in SCORE_COLUMNS:
        values = ctx.scores[score_col]
        for group_name, scope in groups.items():
            scoped = values[scope & np.isfinite(values) & (values >= 0)]
            rows.append(
                {
                    "lane_v1": "LANE_09_SCORE_CALIBRATION_DIAGNOSTIC_SCAN_V1",
                    "score_column_v1": score_col,
                    "group_v1": group_name,
                    "count_v1": int(len(scoped)),
                    "p50_v1": float(np.quantile(scoped, 0.50)) if len(scoped) else None,
                    "p75_v1": float(np.quantile(scoped, 0.75)) if len(scoped) else None,
                    "p90_v1": float(np.quantile(scoped, 0.90)) if len(scoped) else None,
                    "p95_v1": float(np.quantile(scoped, 0.95)) if len(scoped) else None,
                    "p99_v1": float(np.quantile(scoped, 0.99)) if len(scoped) else None,
                    "mean_v1": float(np.mean(scoped)) if len(scoped) else None,
                    "above_050_count_v1": int((scoped >= 0.50).sum()) if len(scoped) else 0,
                    "above_075_count_v1": int((scoped >= 0.75).sum()) if len(scoped) else 0,
                    "above_090_count_v1": int((scoped >= 0.90).sum()) if len(scoped) else 0,
                    "diagnostic_class_v1": "CALIBRATION_PROBLEM_CANDIDATE" if group_name.startswith("MISSED") and len(scoped) and float(np.quantile(scoped, 0.90)) < 0.50 else "SEPARATION_AVAILABLE_OR_NOT_ESTABLISHED",
                }
            )
    return pd.DataFrame(rows)


def _lane_10_from_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    candidates = leaderboard.sort_values(
        ["wednesday_safety_pass_v1", "bad_blocks_v1", "tail_help_v1", "hard_safety_pass_v1"],
        ascending=[False, False, False, False],
        na_position="last",
    ).head(200)
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        hard_overlap = int(row.get("fifty_plus_mfe_blocked_v1", 0) or 0) + int(row.get("hundred_plus_mfe_blocked_v1", 0) or 0) + int(row.get("two_hundred_plus_mfe_blocked_v1", 0) or 0)
        rows.append(
            {
                "lane_v1": "LANE_10_HIGH_MFE_PROTECTION_STRESS_SCAN_V1",
                "source_lane_v1": row.get("lane_v1"),
                "rule_id_v1": row.get("rule_id_v1", row.get("policy_name_v1")),
                "block_count_v1": row.get("block_count_v1"),
                "bad_blocks_v1": row.get("bad_blocks_v1"),
                "tail_help_v1": row.get("tail_help_v1"),
                "precision_v1": row.get("precision_v1"),
                "fifty_plus_mfe_blocked_v1": row.get("fifty_plus_mfe_blocked_v1"),
                "hundred_plus_mfe_blocked_v1": row.get("hundred_plus_mfe_blocked_v1"),
                "two_hundred_plus_mfe_blocked_v1": row.get("two_hundred_plus_mfe_blocked_v1"),
                "strongest_winner_damage_v1": row.get("strongest_winner_damage_v1"),
                "repaired_165_damage_v1": row.get("repaired_165_damage_v1"),
                "forensic_repaired_trade_blocked_v1": row.get("forensic_repaired_trade_blocked_v1", 0),
                "high_mfe_overlap_count_v1": hard_overlap,
                "safe_or_unsafe_classification_v1": "SAFE_STRESS_PASS" if bool(row.get("wednesday_safety_pass_v1")) else "UNSAFE_STRESS_FAIL",
                "failure_tags_v1": row.get("strict_safety_failures_v1", row.get("rejection_reason_v1", "")),
            }
        )
    return pd.DataFrame(rows)


def _lane_summary(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "lane_v1": name,
            "row_count_v1": 0,
            "safe_candidate_count_v1": 0,
            "best_safe_bad_blocks_v1": 0,
            "best_safe_tail_help_v1": 0,
        }
    safe_col = "wednesday_safety_pass_v1" if "wednesday_safety_pass_v1" in frame.columns else None
    safe = frame[frame[safe_col].astype(bool)].copy() if safe_col else pd.DataFrame()
    best = safe.sort_values(["bad_blocks_v1", "tail_help_v1", "precision_v1"], ascending=[False, False, False], na_position="last").iloc[0].to_dict() if not safe.empty and "bad_blocks_v1" in safe else None
    return {
        "lane_v1": name,
        "row_count_v1": int(len(frame)),
        "safe_candidate_count_v1": int(len(safe)),
        "best_safe_bad_blocks_v1": int(best.get("bad_blocks_v1")) if best else 0,
        "best_safe_tail_help_v1": int(best.get("tail_help_v1")) if best else 0,
        "best_safe_precision_v1": float(best.get("precision_v1")) if best and pd.notna(best.get("precision_v1")) else None,
        "best_safe_worst_loso_v1": float(best.get("worst_loso_v1")) if best and pd.notna(best.get("worst_loso_v1")) else None,
        "best_safe_rule_v1": best.get("rule_id_v1") or best.get("policy_name_v1") if best else None,
        "best_safe_implementability_v1": best.get("implementability_v1") if best else None,
    }


def _validate_inputs(score_dir: Path, r6_dir: Path, frame: pd.DataFrame, score_summary: dict[str, Any], r6_summary: dict[str, Any]) -> None:
    if len(frame) in FORBIDDEN_ROW_COUNTS:
        raise RuntimeError(f"Refuses forbidden R6 scan row count: {len(frame)}")
    if int(len(frame)) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_ROW_COUNT} Monday foundation rows, observed {len(frame)}")
    active = frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")
    if int(active.sum()) != EXPECTED_ACTIVE_ROWS or int((~active).sum()) != EXPECTED_QUARANTINE_ROWS:
        raise RuntimeError(f"Expected active/quarantine {EXPECTED_ACTIVE_ROWS}/{EXPECTED_QUARANTINE_ROWS}, observed {int(active.sum())}/{int((~active).sum())}")
    if int(r6_summary.get("as_of_column_count_v1") or EXPECTED_AS_OF_COLUMNS) != EXPECTED_AS_OF_COLUMNS:
        raise RuntimeError(f"Expected {EXPECTED_AS_OF_COLUMNS} AS_OF columns, observed {r6_summary.get('as_of_column_count_v1')}")
    if score_summary and score_summary.get("decision_v1") != "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED":
        raise RuntimeError(f"Score package is not completed: {score_summary.get('decision_v1')}")
    if score_summary and bool(score_summary.get("r6_heads_trained_v1")):
        raise RuntimeError("Score package must not contain R6 heads")
    if str(score_dir).upper().find("1689") >= 0 or str(r6_dir).upper().find("PROTECTOR") >= 0:
        raise RuntimeError("Refuses diagnostic/protector/1689 source path for canonical R6 scan")
    required = [
        "candidate_uid",
        "run_id",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "split_scope_v1",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "tail_10_50_mfe_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r5_2_selected_candidate__block_v1",
        "selected_candidate_block_v1",
        *SCORE_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"R6 training frame missing required columns: {missing}")


def _write_lane(output_dir: Path, lane_name: str, filename: str, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    lane_dir = output_dir / lane_name.lower()
    lane_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / filename, index=False)
    frame.to_csv(lane_dir / filename, index=False)
    _write_json(lane_dir / "summary_v1.json", summary)
    (lane_dir / "report_v1.md").write_text(
        "\n".join(
            [
                f"# {lane_name}",
                "",
                f"- Rows materialized: `{summary.get('row_count_v1')}`",
                f"- Safe candidates: `{summary.get('safe_candidate_count_v1')}`",
                f"- Best safe bad/tail: `{summary.get('best_safe_bad_blocks_v1')}` / `{summary.get('best_safe_tail_help_v1')}`",
                f"- Best safe rule: `{summary.get('best_safe_rule_v1')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING", "PASS" if summary["training_started_v1"] is False else "FAIL", summary["training_started_v1"]),
            row("NO_NEW_BASELINE", "PASS" if summary["new_baseline_built_v1"] is False else "FAIL", summary["new_baseline_built_v1"]),
            row("NO_NEW_FEATURE_SURFACE", "PASS" if summary["new_feature_surface_built_v1"] is False else "FAIL", summary["new_feature_surface_built_v1"]),
            row("ROW_COUNT_1914", "PASS" if summary["row_count_v1"] == EXPECTED_ROW_COUNT else "FAIL", summary["row_count_v1"]),
            row("ACTIVE_QUARANTINE_1852_62", "PASS" if [summary["active_rows_v1"], summary["quarantine_rows_v1"]] == [EXPECTED_ACTIVE_ROWS, EXPECTED_QUARANTINE_ROWS] else "FAIL", [summary["active_rows_v1"], summary["quarantine_rows_v1"]]),
            row("AS_OF_SCHEMA_109", "PASS" if summary["as_of_column_count_v1"] == EXPECTED_AS_OF_COLUMNS else "FAIL", summary["as_of_column_count_v1"]),
            row("FORENSIC_REPAIRED_TRADE_PRESENT", "PASS" if summary["forensic_repaired_trade_present_v1"] else "FAIL", summary["forensic_repaired_trade_present_v1"]),
            row("NO_PROTECTOR_OR_1689_SOURCE", "PASS" if summary["diagnostic_or_protector_source_used_v1"] is False else "FAIL", summary["diagnostic_or_protector_source_used_v1"]),
            row("ALL_LANES_MATERIALIZED", "PASS" if summary["lane_count_v1"] >= 10 else "FAIL", summary["lane_count_v1"]),
        ]
    )


def _report(summary: dict[str, Any], aggregator: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Parallel Monday R6 Recall Recovery Scan V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Source R6: `{summary['source_r6_dir_v1']}`",
            f"- Score package: `{summary['source_score_dir_v1']}`",
            f"- Rows: `{summary['row_count_v1']}` active/quarantine `{summary['active_rows_v1']}` / `{summary['quarantine_rows_v1']}`",
            f"- Current bad/tail: `{summary['current_bad_blocks_v1']}` / `{summary['current_tail_help_v1']}`",
            f"- Best safe bad/tail found: `{summary['best_safe_bad_blocks_v1']}` / `{summary['best_safe_tail_help_v1']}`",
            f"- Best safe source lane: `{summary['best_safe_lane_v1']}`",
            f"- Most dangerous tempting rule: `{aggregator.get('most_dangerous_tempting_rule_v1')}`",
            "",
            "No baseline, feature surface, protector-first path, freeze, promo, live gate, or controller mutation was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    score_dir: Path | None = None,
    r6_dir: Path | None = None,
    output_dir: Path | None = None,
    run_parallel_scan: bool = False,
    max_workers: int = 6,
    quick_scan: bool = False,
) -> dict[str, Any]:
    if not run_parallel_scan:
        raise RuntimeError("PARALLEL_MONDAY_R6_RECALL_RECOVERY_SCAN_V1 requires --run-parallel-scan")
    reports_root = reports_root.expanduser().resolve()
    score_dir = score_dir.expanduser().resolve() if score_dir else _latest_dir(reports_root, SCORE_GLOB, SCORE_FRAME)
    r6_dir = r6_dir.expanduser().resolve() if r6_dir else _latest_dir(reports_root, R6_GLOB, R6_TRAINING_FRAME)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(r6_dir / R6_TRAINING_FRAME)
    prediction = pd.read_parquet(r6_dir / R6_PREDICTION_VIEW)
    grid = pd.read_csv(r6_dir / R6_GRID)
    r6_summary = _read_json(r6_dir / R6_SUMMARY)
    r6_compare = _read_json(r6_dir / R6_COMPARE)
    score_summary = _read_json(score_dir / SCORE_SUMMARY)
    _validate_inputs(score_dir, r6_dir, frame, score_summary, r6_summary)
    ctx = _make_context(frame)

    orchestrator = {
        "layer_name": f"{LAYER_NAME}_ORCHESTRATOR",
        "materialized_at_utc_v1": _utc_now(),
        "execution_mode_v1": "PARALLEL_THREAD_LANES",
        "max_workers_v1": int(max_workers),
        "quick_scan_v1": bool(quick_scan),
        "source_score_dir_v1": str(score_dir),
        "source_r6_dir_v1": str(r6_dir),
        "same_foundation_and_score_package_for_all_lanes_v1": True,
        "new_baseline_allowed_v1": False,
        "new_feature_surface_allowed_v1": False,
        "protector_first_allowed_v1": False,
        "freeze_promo_live_allowed_v1": False,
        "lane_names_v1": list(LANE_FILES.keys()) + ["LANE_10_HIGH_MFE_PROTECTION_STRESS_SCAN_V1"],
    }

    lane_fns: dict[str, Callable[[], pd.DataFrame]] = {
        "LANE_01_R6_THRESHOLD_FRONTIER_SCAN_V1": lambda: _lane_01(ctx, quick_scan=quick_scan),
        "LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1": lambda: _lane_02(ctx, quick_scan=quick_scan),
        "LANE_03_R5_R5_1_R5_2_UNION_WITH_STRONG_GUARD_SCAN_V1": lambda: _lane_03(ctx, quick_scan=quick_scan),
        "LANE_04_TAIL_CONTROL_10_50_RECOVERY_SCAN_V1": lambda: _lane_04(ctx, quick_scan=quick_scan),
        "LANE_05_BAD_RISK_RECALL_RECOVERY_SCAN_V1": lambda: _lane_05(ctx, quick_scan=quick_scan),
        "LANE_06_RUNNER_GUARD_SENSITIVITY_SCAN_V1": lambda: _lane_06(ctx, grid, quick_scan=quick_scan),
        "LANE_07_CANDIDATE_GRID_REJECTED_SAFE_RECALL_SCAN_V1": lambda: _lane_07(grid),
        "LANE_08_BATCH_LOSO_STABILITY_SCAN_V1": lambda: _lane_08(ctx, grid),
        "LANE_09_SCORE_CALIBRATION_DIAGNOSTIC_SCAN_V1": lambda: _lane_09(ctx),
    }
    lane_frames: dict[str, pd.DataFrame] = {}
    worker_count = max(1, min(max_workers, len(lane_fns)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_lane = {executor.submit(func): lane for lane, func in lane_fns.items()}
        for future in as_completed(future_to_lane):
            lane = future_to_lane[future]
            lane_frames[lane] = future.result()

    lane_summaries: dict[str, dict[str, Any]] = {}
    for lane_name in LANE_FILES:
        lane_df = lane_frames[lane_name]
        lane_summary = _lane_summary(lane_name, lane_df)
        lane_summaries[lane_name] = lane_summary
        _write_lane(output_dir, lane_name, LANE_FILES[lane_name], lane_df, lane_summary)

    leaderboard_parts = []
    for lane_name in [
        "LANE_01_R6_THRESHOLD_FRONTIER_SCAN_V1",
        "LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1",
        "LANE_03_R5_R5_1_R5_2_UNION_WITH_STRONG_GUARD_SCAN_V1",
        "LANE_04_TAIL_CONTROL_10_50_RECOVERY_SCAN_V1",
        "LANE_05_BAD_RISK_RECALL_RECOVERY_SCAN_V1",
        "LANE_06_RUNNER_GUARD_SENSITIVITY_SCAN_V1",
        "LANE_07_CANDIDATE_GRID_REJECTED_SAFE_RECALL_SCAN_V1",
    ]:
        part = lane_frames[lane_name].copy()
        if "wednesday_safety_pass_v1" in part.columns and "bad_blocks_v1" in part.columns:
            leaderboard_parts.append(part)
    leaderboard = pd.concat(leaderboard_parts, ignore_index=True, sort=False) if leaderboard_parts else pd.DataFrame()
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["wednesday_safety_pass_v1", "bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )
    lane_10 = _lane_10_from_leaderboard(leaderboard)
    lane_10_summary = _lane_summary("LANE_10_HIGH_MFE_PROTECTION_STRESS_SCAN_V1", lane_10)
    lane_summaries["LANE_10_HIGH_MFE_PROTECTION_STRESS_SCAN_V1"] = lane_10_summary
    _write_lane(output_dir, "LANE_10_HIGH_MFE_PROTECTION_STRESS_SCAN_V1", LANE_10, lane_10, lane_10_summary)
    leaderboard.to_csv(output_dir / LEADERBOARD, index=False)

    safe_leaderboard = leaderboard[leaderboard["wednesday_safety_pass_v1"].astype(bool)].copy() if not leaderboard.empty else pd.DataFrame()
    best_safe = (
        safe_leaderboard.sort_values(["bad_blocks_v1", "tail_help_v1", "precision_v1", "worst_loso_v1"], ascending=[False, False, False, False], na_position="last").iloc[0].to_dict()
        if not safe_leaderboard.empty
        else None
    )
    best_tail = (
        safe_leaderboard.sort_values(["tail_help_v1", "bad_blocks_v1", "precision_v1", "worst_loso_v1"], ascending=[False, False, False, False], na_position="last").iloc[0].to_dict()
        if not safe_leaderboard.empty
        else None
    )
    dangerous = None
    if not leaderboard.empty:
        unsafe = leaderboard[~leaderboard["wednesday_safety_pass_v1"].astype(bool)].copy()
        if not unsafe.empty:
            dangerous = unsafe.sort_values(["bad_blocks_v1", "tail_help_v1", "precision_v1"], ascending=[False, False, False], na_position="last").iloc[0].to_dict()

    best_bad = int(best_safe.get("bad_blocks_v1")) if best_safe else int(ctx.current_metrics["bad_blocks_v1"])
    best_tail_help = int(best_tail.get("tail_help_v1")) if best_tail else int(ctx.current_metrics["tail_help_v1"])
    best_lane = str(best_safe.get("lane_v1")) if best_safe else None
    best_impl = str(best_safe.get("implementability_v1")) if best_safe and best_safe.get("implementability_v1") is not None else None
    uplift_found = best_bad > int(ctx.current_metrics["bad_blocks_v1"]) or best_tail_help > int(ctx.current_metrics["tail_help_v1"])
    if uplift_found and best_impl == "R5_2_BASE_MEMBERSHIP_CONTRACT_ONLY":
        next_action = "IMPLEMENT_SAFE_R5_2_BASE_EXTENSION_V2"
    elif uplift_found and best_impl == "GUARD_CONTRACT_CHANGE":
        next_action = "FIX_GUARD_OVERPROTECTION"
    elif uplift_found and best_impl == "CANDIDATE_SELECTION_ONLY":
        next_action = "ADJUST_R6_CANDIDATE_SELECTION"
    elif uplift_found:
        next_action = "RUN_R6_RETRAIN_WITH_BEST_SAFE_FRONTIER"
    else:
        next_action = "RUN_TARGETED_PARALLEL_SCAN_ROUND_2"

    decision = "SAFE_RECALL_UPLIFT_FOUND" if uplift_found else "NO_SAFE_RECALL_UPLIFT_FOUND"
    aggregator = {
        "layer_name": f"{LAYER_NAME}_AGGREGATOR",
        "decision_v1": decision,
        "next_action_v1": next_action,
        "current_metrics_v1": ctx.current_metrics,
        "current_worst_loso_v1": ctx.current_worst_loso,
        "best_safe_bad_candidate_v1": best_safe,
        "best_safe_tail_candidate_v1": best_tail,
        "best_safe_bad_blocks_v1": best_bad,
        "best_safe_tail_help_v1": best_tail_help,
        "best_safe_lane_v1": best_lane,
        "best_safe_implementability_v1": best_impl,
        "best_safe_bad_uplift_v1": int(best_bad - ctx.current_metrics["bad_blocks_v1"]),
        "best_safe_tail_uplift_v1": int(best_tail_help - ctx.current_metrics["tail_help_v1"]),
        "most_dangerous_tempting_rule_v1": dangerous,
        "lane_summaries_v1": lane_summaries,
        "no_go_rules_v1": [
            "Rules that block forensic repaired trade",
            "Rules with 100+/200+ MFE damage",
            "Rules with strongest-winner damage",
            "Rules with precision or worst LOSO below Wednesday-R6",
            "Rules relying on 1689/protector/diagnostic surfaces",
        ],
    }
    next_action_lock = {
        "layer_name": f"{LAYER_NAME}_NEXT_ACTION_LOCK",
        "decision_v1": decision,
        "next_action_v1": next_action,
        "blocked_actions_v1": [
            "DO_NOT_RETRAIN_YET" if next_action != "RUN_R6_RETRAIN_WITH_BEST_SAFE_FRONTIER" else "DO_NOT_FREEZE_OR_PROMOTE",
            "DO_NOT_BUILD_NEW_BASELINE_COPY",
            "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_CONTINUE_PROTECTOR_FIRST",
        ],
    }

    active = frame["calendar_quarantine_status_v1"].astype("string").eq("ACTIVE_CANDIDATE")
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "source_score_dir_v1": str(score_dir),
        "source_r6_dir_v1": str(r6_dir),
        "decision_v1": decision,
        "next_action_v1": next_action,
        "training_started_v1": False,
        "r6_training_started_v1": False,
        "freeze_promo_live_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "diagnostic_or_protector_source_used_v1": False,
        "row_count_v1": int(len(frame)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "as_of_column_count_v1": int(r6_summary.get("as_of_column_count_v1") or EXPECTED_AS_OF_COLUMNS),
        "prediction_view_rows_v1": int(len(prediction)),
        "current_bad_blocks_v1": int(ctx.current_metrics["bad_blocks_v1"]),
        "current_tail_help_v1": int(ctx.current_metrics["tail_help_v1"]),
        "current_precision_v1": ctx.current_metrics["precision_v1"],
        "current_worst_loso_v1": ctx.current_worst_loso,
        "best_safe_bad_blocks_v1": best_bad,
        "best_safe_tail_help_v1": best_tail_help,
        "best_safe_bad_uplift_v1": int(best_bad - ctx.current_metrics["bad_blocks_v1"]),
        "best_safe_tail_uplift_v1": int(best_tail_help - ctx.current_metrics["tail_help_v1"]),
        "best_safe_lane_v1": best_lane,
        "best_safe_implementability_v1": best_impl,
        "lane_count_v1": 10,
        "lane_summaries_v1": lane_summaries,
        "forensic_repaired_trade_present_v1": bool(ctx.forensic_present),
        "source_r6_compare_decision_v1": r6_compare.get("verdict_v1"),
        "hard_status_v1": {
            "BEVIST": [
                "All scans used the same fixed Monday R6 foundation and score package.",
                "No training, baseline build, feature-surface build, protector-first path, freeze, promo, or live/controller path ran.",
                "All lane outputs, leaderboard, aggregator, status, manifest and audit were materialized.",
            ],
            "INDIKERT": [
                "Any safe uplift found here is a read-only recovery candidate and still requires the locked next implementation action.",
            ],
            "IKKE_ETABLERT": [
                "No candidate from this scan is canonical Monday R6 until implemented and rerun through the R6 gate.",
            ],
        },
    }
    audit = _audit(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "source_score_dir_v1": str(score_dir),
        "source_r6_dir_v1": str(r6_dir),
        "training_started_v1": False,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": decision,
        "next_action_v1": next_action,
        "training_started_v1": False,
        "blocked_action_v1": next_action_lock["blocked_actions_v1"],
    }

    _write_json(output_dir / PARALLEL_SCAN_ORCHESTRATOR, orchestrator)
    _write_json(output_dir / AGGREGATOR, aggregator)
    _write_json(output_dir / NEXT_ACTION, next_action_lock)
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / MANIFEST, manifest)
    _write_json(output_dir / STATUS, status)
    audit.to_csv(output_dir / AUDIT, index=False)
    (output_dir / REPORT).write_text(_report(summary, aggregator), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--r6-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-parallel-scan", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--quick-scan-for-tests", action="store_true")
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        score_dir=args.score_dir,
        r6_dir=args.r6_dir,
        output_dir=args.output_dir,
        run_parallel_scan=args.run_parallel_scan,
        max_workers=args.max_workers,
        quick_scan=args.quick_scan_for_tests,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
