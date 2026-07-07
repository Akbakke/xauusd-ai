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
LAYER_NAME = "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1"
R6_REBUILD_GLOB = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_*"

TRAINING_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
PREDICTION_VIEW = "monday_r6_on_foundation_scores_prediction_view_v1.parquet"
R6_GRID = "r6_family_grid_replay_v1.csv"
R6_SUMMARY = "summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"

SUMMARY = "summary_v1.json"
RECALL_GAP_SUMMARY = "recall_gap_summary_v1.json"
SPLIT_RECALL_GAP = "split_recall_gap_v1.csv"
MISSED_BAD_ROWS = "missed_bad_rows_v1.csv"
MISSED_TAIL_ROWS = "missed_tail_rows_v1.csv"
SCORE_DISTRIBUTION_BY_SPLIT = "score_distribution_by_split_v1.csv"
CANDIDATE_GRID_RECALL_FRONTIER = "candidate_grid_recall_frontier_v1.csv"
TOP_SAFETY_TRADEOFF_CANDIDATES = "top_safety_tradeoff_candidates_v1.csv"
FALLBACK_WORD_AUDIT = "fallback_word_audit_v1.csv"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"
MANIFEST = "manifest_v1.json"
REPORT = "report_v1.md"

OUTPUT_FILES = {
    "summary": SUMMARY,
    "recall_gap_summary": RECALL_GAP_SUMMARY,
    "split_recall_gap": SPLIT_RECALL_GAP,
    "missed_bad_rows": MISSED_BAD_ROWS,
    "missed_tail_rows": MISSED_TAIL_ROWS,
    "score_distribution_by_split": SCORE_DISTRIBUTION_BY_SPLIT,
    "candidate_grid_recall_frontier": CANDIDATE_GRID_RECALL_FRONTIER,
    "top_safety_tradeoff_candidates": TOP_SAFETY_TRADEOFF_CANDIDATES,
    "fallback_word_audit": FALLBACK_WORD_AUDIT,
    "audit": CONSISTENCY_AUDIT,
    "manifest": MANIFEST,
    "report": REPORT,
}

WEDNESDAY_R6 = {
    "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
    "candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
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
EXPECTED_ROW_COUNT = 1914

SCORE_COLUMNS = [
    "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "pred__entry_r5_2_runner_protector__prob_true_v1",
    "pred__entry_r6_bad_risk__prob_true_v1",
    "pred__entry_r6_runner_protector__prob_true_v1",
    "pred__entry_r6_tail_control_10_50__prob_true_v1",
    "pred__entry_r6_risky_allow__prob_true_v1",
    "pred__entry_r6_batch04_blindspot__prob_true_v1",
]

ROW_COLUMNS = [
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "decision_timestamp",
    "split_scope_v1",
    "calendar_quarantine_status_v1",
    "label_should_not_take_v1",
    "tail_10_50_mfe_v1",
    "take_was_ok_v1",
    "fifty_plus_mfe_v1",
    "hundred_plus_mfe_v1",
    "two_hundred_plus_mfe_v1",
    "strongest_winner_path_v1",
    "r5_selected_candidate__block_v1",
    "r5_1_selected_candidate__block_v1",
    "r5_2_selected_candidate__block_v1",
    "selected_candidate_block_v1",
    *SCORE_COLUMNS,
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
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


def _validate(r6_dir: Path, frame: pd.DataFrame, grid: pd.DataFrame, summary: dict[str, Any]) -> None:
    missing = [path for path in [r6_dir / TRAINING_FRAME, r6_dir / PREDICTION_VIEW, r6_dir / R6_GRID, r6_dir / R6_SUMMARY] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R6 rebuild dir missing artifacts: {missing}")
    if int(len(frame)) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_ROW_COUNT}-row Monday R6 foundation frame, observed {len(frame)}")
    if summary.get("decision_v1") not in {
        "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER",
        "MONDAY_R6_ON_FOUNDATION_SCORES_RAN_BUT_FAILED_WEDNESDAY_SAFETY",
        "MONDAY_R6_ON_FOUNDATION_SCORES_IMPROVES_AND_HOLDS_WEDNESDAY_SAFETY",
    }:
        raise RuntimeError(f"Unexpected R6 rebuild decision: {summary.get('decision_v1')}")
    required = [
        "candidate_uid",
        "run_id",
        "split_scope_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "take_was_ok_v1",
        "selected_candidate_block_v1",
    ]
    missing_cols = [column for column in required if column not in frame.columns]
    if missing_cols:
        raise KeyError(f"Training frame missing required columns: {missing_cols}")
    grid_required = [
        "policy_name_v1",
        "family_v1",
        "wednesday_safety_pass_v1",
        "hard_damage_count_v1",
        "bad_blocks_v1",
        "tail_help_v1",
        "precision_v1",
    ]
    missing_grid = [column for column in grid_required if column not in grid.columns]
    if missing_grid:
        raise KeyError(f"R6 grid missing columns: {missing_grid}")


def _split_recall(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _bool(frame, "selected_candidate_block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    rows: list[dict[str, Any]] = []
    for split, group in frame.assign(_selected=selected, _should=should, _tail=tail, _take_ok=take_ok).groupby("split_scope_v1", dropna=False):
        gsel = group["_selected"].astype(bool)
        gshould = group["_should"].astype(bool)
        gtail = group["_tail"].astype(bool)
        bad_pop = int(gshould.sum())
        tail_pop = int(gtail.sum())
        bad_blocks = int((gsel & gshould).sum())
        tail_help = int((gsel & gtail).sum())
        rows.append(
            {
                "split_scope_v1": str(split),
                "row_count_v1": int(len(group)),
                "bad_population_v1": bad_pop,
                "tail_population_v1": tail_pop,
                "selected_blocks_v1": int(gsel.sum()),
                "selected_bad_blocks_v1": bad_blocks,
                "selected_tail_help_v1": tail_help,
                "bad_recall_v1": float(bad_blocks / bad_pop) if bad_pop else None,
                "tail_recall_v1": float(tail_help / tail_pop) if tail_pop else None,
                "false_take_ok_blocks_v1": int((gsel & group["_take_ok"].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("split_scope_v1")


def _score_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in frame.groupby("split_scope_v1", dropna=False):
        for column in SCORE_COLUMNS:
            if column not in group.columns:
                continue
            values = _num(group, column).dropna()
            rows.append(
                {
                    "split_scope_v1": str(split),
                    "score_column_v1": column,
                    "count_v1": int(values.shape[0]),
                    "mean_v1": float(values.mean()) if not values.empty else None,
                    "p50_v1": float(values.quantile(0.50)) if not values.empty else None,
                    "p75_v1": float(values.quantile(0.75)) if not values.empty else None,
                    "p90_v1": float(values.quantile(0.90)) if not values.empty else None,
                    "p95_v1": float(values.quantile(0.95)) if not values.empty else None,
                    "p99_v1": float(values.quantile(0.99)) if not values.empty else None,
                    "max_v1": float(values.max()) if not values.empty else None,
                }
            )
    return pd.DataFrame(rows)


def _grid_frontier(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    groups = {
        "ALL_GRID_TOP_BAD": grid,
        "ALL_GRID_TOP_TAIL": grid,
        "ZERO_HARD_DAMAGE_TOP_BAD": grid[(grid["hard_damage_count_v1"] == 0) & (grid["block_count_v1"] > 0)],
        "WEDNESDAY_SAFE_TOP_BAD": grid[grid["wednesday_safety_pass_v1"].astype(bool)],
    }
    for label, subset in groups.items():
        if subset.empty:
            continue
        if "TAIL" in label:
            ordered = subset.sort_values(["tail_help_v1", "bad_blocks_v1", "precision_v1"], ascending=[False, False, False], na_position="last").head(10)
        else:
            ordered = subset.sort_values(["bad_blocks_v1", "tail_help_v1", "precision_v1"], ascending=[False, False, False], na_position="last").head(10)
        for _, row in ordered.iterrows():
            payload = row.to_dict()
            payload["frontier_bucket_v1"] = label
            rows.append(payload)
    frontier = pd.DataFrame(rows)
    tradeoff_cols = [
        "policy_name_v1",
        "family_v1",
        "hard_damage_count_v1",
        "wednesday_safety_pass_v1",
        "wednesday_basic_safety_pass_v1",
        "worst_loso_v1",
        "block_count_v1",
        "bad_blocks_v1",
        "tail_help_v1",
        "precision_v1",
        "false_take_ok_blocks_v1",
        "fifty_plus_mfe_blocked_v1",
        "hundred_plus_mfe_blocked_v1",
        "two_hundred_plus_mfe_blocked_v1",
        "strongest_winner_damage_v1",
        "repaired_165_damage_v1",
    ]
    top_tradeoff = frontier[[column for column in ["frontier_bucket_v1", *tradeoff_cols] if column in frontier.columns]].copy()
    return frontier, top_tradeoff


def _missed_rows(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    selected = _bool(frame, "selected_candidate_block_v1")
    label = _bool(frame, label_column)
    missed = frame[label & ~selected].copy()
    if missed.empty:
        return missed
    missed["miss_reason_not_r5_2_base_v1"] = ~_bool(missed, "r5_2_selected_candidate__block_v1")
    missed["miss_reason_r6_bad_score_below_085_v1"] = _num(missed, "pred__entry_r6_bad_risk__prob_true_v1").lt(0.85)
    missed["miss_reason_r6_tail_score_below_085_v1"] = _num(missed, "pred__entry_r6_tail_control_10_50__prob_true_v1").lt(0.85)
    missed["miss_reason_r6_risky_score_below_099_v1"] = _num(missed, "pred__entry_r6_risky_allow__prob_true_v1").lt(0.99)
    missed["miss_reason_runner_or_asof_protected_v1"] = (
        _num(missed, "pred__entry_r6_runner_protector__prob_true_v1").ge(0.30)
        | _num(missed, "pred__entry_r5_2_runner_protector__prob_true_v1").ge(0.74)
        | _bool(missed, "asof_runner_guard_v1")
    )
    return missed[[column for column in [*ROW_COLUMNS, "miss_reason_not_r5_2_base_v1", "miss_reason_r6_bad_score_below_085_v1", "miss_reason_r6_tail_score_below_085_v1", "miss_reason_r6_risky_score_below_099_v1", "miss_reason_runner_or_asof_protected_v1"] if column in missed.columns]]


def _fallback_word_audit(summary: dict[str, Any]) -> pd.DataFrame:
    stale_artifact = "FALLBACK" in json.dumps(summary, sort_keys=True).upper()
    rows = [
        {
            "item_v1": "R6_FAMILY_GRID_SAFE_FALLBACK",
            "location_v1": "existing V2 generated artifact summary",
            "classification_v1": "STALE_NAME_IN_GENERATED_ARTIFACT",
            "canonical_use_v1": "DO_NOT_USE_AS_CANONICAL_LOCK_NAME",
            "evidence_v1": stale_artifact,
        },
        {
            "item_v1": "R6_FAMILY_GRID_SAFE_CANDIDATE",
            "location_v1": "current source runner",
            "classification_v1": "MODEL_SELECTION_CANDIDATE_NOT_RUNTIME_FALLBACK",
            "canonical_use_v1": "MAY_ANALYZE_BUT_NOT_CANONICAL_UNLESS_RECALL_GAP_RESOLVED",
            "evidence_v1": True,
        },
        {
            "item_v1": "R2_FALLBACK_REFERENCE / NO_ENTRY_FALLBACK_BASELINE",
            "location_v1": "legacy R5/R5.2/R6 evaluation references",
            "classification_v1": "REFERENCE_BASELINE",
            "canonical_use_v1": "NOT_MONDAY_R6_POLICY",
            "evidence_v1": True,
        },
        {
            "item_v1": "FROZEN_SHADOW_FALLBACK_CANDIDATE_NOT_LIVE_GATE",
            "location_v1": "older shadow-freeze naming",
            "classification_v1": "HISTORICAL_SHADOW_STATUS_NAME",
            "canonical_use_v1": "NOT_LIVE_FALLBACK",
            "evidence_v1": True,
        },
    ]
    return pd.DataFrame(rows)


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING_RUN", "PASS" if summary["training_started_v1"] is False else "FAIL", summary["training_started_v1"]),
            row("ROW_COUNT_1914", "PASS" if summary["row_count_v1"] == EXPECTED_ROW_COUNT else "FAIL", summary["row_count_v1"]),
            row("SELECTED_POLICY_SAFE", "PASS" if summary["selected_policy_safety_failures_v1"] == [] else "FAIL", summary["selected_policy_safety_failures_v1"]),
            row("RECALL_GAP_PRESENT", "PASS" if summary["bad_block_gap_vs_wednesday_v1"] > 0 and summary["tail_help_gap_vs_wednesday_v1"] > 0 else "FAIL", [summary["bad_block_gap_vs_wednesday_v1"], summary["tail_help_gap_vs_wednesday_v1"]]),
            row("SELECTED_BLOCKS_OUTSIDE_TRAIN_ZERO", "PASS" if summary["selected_blocks_outside_train_v1"] == 0 else "WARN", summary["selected_blocks_outside_train_v1"]),
            row("FALLBACK_NOT_CANONICAL_DECISION", "PASS", summary["fallback_word_decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Recall Gap Before Canonical Lock V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Rows: `{summary['row_count_v1']}`",
            f"- Selected bad blocks: `{summary['selected_bad_blocks_v1']}` vs Wednesday `{WEDNESDAY_R6['bad_blocks_v1']}`",
            f"- Selected tail help: `{summary['selected_tail_help_v1']}` vs Wednesday `{WEDNESDAY_R6['tail_help_v1']}`",
            f"- Selected precision: `{summary['selected_precision_v1']}`",
            f"- Selected worst LOSO: `{summary['selected_worst_loso_v1']}`",
            f"- Selected safety failures: `{summary['selected_policy_safety_failures_v1']}`",
            f"- Selected blocks outside TRAIN: `{summary['selected_blocks_outside_train_v1']}`",
            f"- Max safe family-grid bad/tail: `{summary['family_grid_safe_max_bad_blocks_v1']}` / `{summary['family_grid_safe_max_tail_help_v1']}`",
            f"- Max all-grid bad/tail: `{summary['family_grid_all_max_bad_blocks_v1']}` / `{summary['family_grid_all_max_tail_help_v1']}`",
            "",
            "The current Monday R6 is safety-green but recall-poor. It is not ready for canonical lock.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    r6_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    r6_dir = r6_dir.expanduser().resolve() if r6_dir else _latest_dir(reports_root, R6_REBUILD_GLOB, TRAINING_FRAME)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(r6_dir / TRAINING_FRAME)
    prediction_view = pd.read_parquet(r6_dir / PREDICTION_VIEW)
    grid = pd.read_csv(r6_dir / R6_GRID)
    r6_summary = _read_json(r6_dir / R6_SUMMARY)
    compare = _read_json(r6_dir / R6_COMPARE)
    _validate(r6_dir, frame, grid, r6_summary)

    selected = _bool(frame, "selected_candidate_block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    bad_blocks = int((selected & should).sum())
    tail_help = int((selected & tail).sum())
    block_count = int(selected.sum())
    selected_precision = float(bad_blocks / block_count) if block_count else None
    split_gap = _split_recall(frame)
    score_dist = _score_distribution(frame)
    frontier, tradeoff = _grid_frontier(grid)
    missed_bad = _missed_rows(frame, "label_should_not_take_v1")
    missed_tail = _missed_rows(frame, "tail_10_50_mfe_v1")
    fallback_audit = _fallback_word_audit(r6_summary)

    safe_grid = grid[grid["wednesday_safety_pass_v1"].astype(bool)]
    zero_hard_grid = grid[(grid["hard_damage_count_v1"] == 0) & (grid["block_count_v1"] > 0)]
    selected_by_split = frame.assign(_selected=selected).groupby("split_scope_v1")["_selected"].sum().to_dict()
    selected_outside_train = int(sum(int(value) for key, value in selected_by_split.items() if str(key) != "TRAIN"))
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "source_r6_dir_v1": str(r6_dir),
        "training_started_v1": False,
        "decision_v1": "MONDAY_R6_RECALL_GAP_CONFIRMED_BEFORE_CANONICAL_LOCK",
        "next_action_v1": "FIX_R5_2_BASE_AND_R6_RECALL_GENERALIZATION_BEFORE_CANONICAL_LOCK",
        "row_count_v1": int(len(frame)),
        "selected_block_count_v1": block_count,
        "selected_bad_blocks_v1": bad_blocks,
        "selected_tail_help_v1": tail_help,
        "selected_precision_v1": selected_precision,
        "selected_worst_loso_v1": compare.get("candidate_worst_loso_v1"),
        "selected_policy_safety_failures_v1": compare.get("safety_failures_v1", []),
        "bad_block_gap_vs_wednesday_v1": int(WEDNESDAY_R6["bad_blocks_v1"] - bad_blocks),
        "tail_help_gap_vs_wednesday_v1": int(WEDNESDAY_R6["tail_help_v1"] - tail_help),
        "bad_population_v1": int(should.sum()),
        "tail_population_v1": int(tail.sum()),
        "missed_bad_rows_v1": int((should & ~selected).sum()),
        "missed_tail_rows_v1": int((tail & ~selected).sum()),
        "selected_blocks_by_split_v1": {str(key): int(value) for key, value in selected_by_split.items()},
        "selected_blocks_outside_train_v1": selected_outside_train,
        "validation_bad_population_v1": int((frame["split_scope_v1"].astype("string").eq("VALIDATION") & should).sum()),
        "holdout_bad_population_v1": int((frame["split_scope_v1"].astype("string").eq("HOLDOUT") & should).sum()),
        "r5_2_selected_by_split_v1": {
            str(key): int(value)
            for key, value in frame.assign(_r5_2=_bool(frame, "r5_2_selected_candidate__block_v1")).groupby("split_scope_v1")["_r5_2"].sum().to_dict().items()
        },
        "family_grid_candidate_count_v1": int(len(grid)),
        "family_grid_safe_candidate_count_v1": int(len(safe_grid)),
        "family_grid_zero_hard_damage_candidate_count_v1": int(len(zero_hard_grid)),
        "family_grid_safe_max_bad_blocks_v1": int(safe_grid["bad_blocks_v1"].max()) if not safe_grid.empty else 0,
        "family_grid_safe_max_tail_help_v1": int(safe_grid["tail_help_v1"].max()) if not safe_grid.empty else 0,
        "family_grid_zero_hard_max_bad_blocks_v1": int(zero_hard_grid["bad_blocks_v1"].max()) if not zero_hard_grid.empty else 0,
        "family_grid_zero_hard_max_tail_help_v1": int(zero_hard_grid["tail_help_v1"].max()) if not zero_hard_grid.empty else 0,
        "family_grid_all_max_bad_blocks_v1": int(grid["bad_blocks_v1"].max()),
        "family_grid_all_max_tail_help_v1": int(grid["tail_help_v1"].max()),
        "fallback_word_decision_v1": "NO_FALLBACK_ALLOWED_FOR_CANONICAL_R6; SAFE_CANDIDATE_IS_DIAGNOSTIC_UNTIL_RECALL_GAP_FIXED",
        "hard_status_v1": {
            "BEVIST": [
                "The current selected Monday R6 policy is safety-green but recall-poor.",
                "Selected blocks are concentrated entirely in TRAIN.",
                "R5.2 base selection is also concentrated entirely in TRAIN.",
                "The broad R6 grid can increase recall only by accepting safety damage or precision loss.",
            ],
            "INDIKERT": [
                "The primary gap is score generalization/calibration around R5.2 base and R6 addon recall.",
            ],
            "IKKE_ETABLERT": [
                "Monday R6 canonical lock is not established.",
                "No fallback is accepted as canonical policy.",
            ],
        },
    }
    recall_summary = {
        "summary_v1": summary,
        "wednesday_reference_v1": WEDNESDAY_R6,
        "source_compare_v1": compare,
        "source_decision_v1": r6_summary.get("decision_v1"),
    }
    audit = _audit(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "source_r6_dir_v1": str(r6_dir),
        "training_started_v1": False,
    }

    split_gap.to_csv(output_dir / SPLIT_RECALL_GAP, index=False)
    missed_bad.to_csv(output_dir / MISSED_BAD_ROWS, index=False)
    missed_tail.to_csv(output_dir / MISSED_TAIL_ROWS, index=False)
    score_dist.to_csv(output_dir / SCORE_DISTRIBUTION_BY_SPLIT, index=False)
    frontier.to_csv(output_dir / CANDIDATE_GRID_RECALL_FRONTIER, index=False)
    tradeoff.to_csv(output_dir / TOP_SAFETY_TRADEOFF_CANDIDATES, index=False)
    fallback_audit.to_csv(output_dir / FALLBACK_WORD_AUDIT, index=False)
    audit.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / RECALL_GAP_SUMMARY, recall_summary)
    _write_json(output_dir / MANIFEST, manifest)
    (output_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--r6-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(reports_root=args.reports_root, r6_dir=args.r6_dir, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
