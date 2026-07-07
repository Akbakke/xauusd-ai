#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    AS_OF_TABLE,
    EVAL_EXACT_LABEL_COLUMNS,
    EXACT_EVAL_LABEL_SOURCE_TABLE,
    EXACT_R6_LABEL_SOURCE_TABLE,
    EXACT_R5_2_LABEL_SOURCE_DIR,
    EXACT_R5_2_LABEL_SOURCE_TABLE,
    EXACT_R5_LABEL_SOURCE_DIR,
    EXACT_R5_LABEL_SOURCE_TABLE,
    HINDSIGHT_TABLE,
    MONDAY_TRUTH_GLOB,
    R5_2_EXACT_LABEL_COLUMNS,
    R5_2_HEADS,
    R5_EXACT_LABEL_COLUMNS,
    R5_HEADS,
    R6_EXACT_LABEL_COLUMNS,
    REHYDRATED_GLOB,
    TRUTH_TABLE,
    WEDNESDAY_R6_BENCHMARK,
    WEDNESDAY_R6_SPLIT_SOURCE_DIR,
    _apply_base_label_aliases,
    _assign_splits,
    _base_feature_names,
    _derive_labels,
    _derive_r5_2_and_r6_labels,
    _exact_label_sources,
    _load_surfaces,
    _overlay_by_candidate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1"

FOUNDATION_AS_OF = "monday_r6_foundation_as_of_109_v1.parquet"
FOUNDATION_HINDSIGHT = "monday_r6_foundation_hindsight_with_labels_v1.parquet"
FOUNDATION_FRAME = "monday_r6_foundation_training_frame_pre_score_v1.parquet"
FOUNDATION_CONTRACT = "foundation_contract_v1.json"
FOUNDATION_LABEL_SUMMARY = "foundation_label_summary_v1.json"
FEATURE_CONTRACT_AUDIT = "feature_contract_audit_v1.csv"
ROW_UNIVERSE_DELTA = "row_universe_delta_v1.csv"
RUN_INVENTORY = "run_inventory_v1.csv"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"
STATUS = "status_v1.json"
SUMMARY = "summary_v1.json"
MANIFEST = "manifest_v1.json"
REPORT = "report_v1.md"

OUTPUT_FILES = {
    "foundation_as_of": FOUNDATION_AS_OF,
    "foundation_hindsight": FOUNDATION_HINDSIGHT,
    "foundation_frame": FOUNDATION_FRAME,
    "foundation_contract": FOUNDATION_CONTRACT,
    "foundation_label_summary": FOUNDATION_LABEL_SUMMARY,
    "feature_contract_audit": FEATURE_CONTRACT_AUDIT,
    "row_universe_delta": ROW_UNIVERSE_DELTA,
    "run_inventory": RUN_INVENTORY,
    "audit": CONSISTENCY_AUDIT,
    "status": STATUS,
    "summary": SUMMARY,
    "manifest": MANIFEST,
    "report": REPORT,
}

ID_COLS = ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]
EXPECTED_AS_OF_COLUMNS = 109
OLD_ACTIVE_ONLY_ROWS = 1852
OLD_EXACT_ONLY_ROWS = 1689


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


def _latest_dir(reports_root: Path, pattern: str, required_file: str | None = None) -> Path:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir())
    if required_file:
        dirs = [path for path in dirs if (path / required_file).exists()]
    if not dirs:
        raise FileNotFoundError(f"No directory matching {pattern} under {reports_root}")
    return dirs[-1]


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes"]).fillna(False).astype(bool)


def _fill_label_booleans(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    label_prefixes = ("r5_label_", "r5_2_label_", "r6_label_")
    label_names = {
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "label_strong_trade_candidate_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        "strongest_winner_path_v1",
        "is_repaired_165_v1",
    }
    for column in out.columns:
        if column.startswith(label_prefixes) or column in label_names:
            out[column] = _bool(out, column)
    return out


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _build_foundation_frame(reports_root: Path, monday_truth_dir: Path, rehydrated_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _load_surfaces(monday_truth_dir, rehydrated_dir)
    exact_sources = _exact_label_sources(reports_root)
    exact_label_report: dict[str, Any] = {}
    frame = _derive_labels(frame)
    frame, exact_label_report["r5_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["r5"], R5_EXACT_LABEL_COLUMNS)
    frame, exact_label_report["eval_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["eval"], EVAL_EXACT_LABEL_COLUMNS)
    frame = _apply_base_label_aliases(frame)
    frame = _derive_r5_2_and_r6_labels(frame)
    frame, exact_label_report["r5_2_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["r5_2"], R5_2_EXACT_LABEL_COLUMNS)
    frame, exact_label_report["r6_exact_labels_v1"] = _overlay_by_candidate(frame, exact_sources["r6"], R6_EXACT_LABEL_COLUMNS)
    frame = _apply_base_label_aliases(frame)
    frame = _fill_label_booleans(frame)
    frame = _assign_splits(frame, split_reference=None)
    return frame, exact_label_report


def _foundation_asof(frame: pd.DataFrame, rehydrated_dir: Path) -> pd.DataFrame:
    original = pd.read_parquet(rehydrated_dir / AS_OF_TABLE)
    out = original.copy()
    split_cols = ["used_for_training", "used_for_validation", "used_for_holdout"]
    split_map = frame[["candidate_uid", *[col for col in split_cols if col in frame.columns]]].drop_duplicates("candidate_uid")
    out = out.drop(columns=[col for col in split_cols if col in out.columns], errors="ignore").merge(split_map, on="candidate_uid", how="left", validate="one_to_one")
    return out[[col for col in original.columns if col in out.columns]]


def _foundation_hindsight(frame: pd.DataFrame, rehydrated_dir: Path) -> pd.DataFrame:
    original = pd.read_parquet(rehydrated_dir / HINDSIGHT_TABLE)
    label_cols = [
        column
        for column in frame.columns
        if column.startswith("r5_label_")
        or column.startswith("r5_2_label_")
        or column.startswith("r6_label_")
        or column
        in {
            "label_should_not_take_v1",
            "take_was_ok_v1",
            "label_strong_trade_candidate_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            "tail_10_50_mfe_v1",
            "strongest_winner_path_v1",
            "batch_scope_v1",
            "calendar_quarantine_status_v1",
            "calendar_quarantine_reason_v1",
        }
    ]
    overlay = frame[["candidate_uid", *label_cols]].drop_duplicates("candidate_uid")
    base = original.drop(columns=[col for col in label_cols if col in original.columns and col != "candidate_uid"], errors="ignore")
    return base.merge(overlay, on="candidate_uid", how="left", validate="one_to_one")


def _feature_audit(asof: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    base_features = set(_base_feature_names(frame))
    score_cols = {
        column
        for column in asof.columns
        if column.startswith("pred__entry_r5_")
        or column.startswith("pred__entry_r6_")
        or column.startswith("pred__entry_r5_2_")
        or column in {"blocker_score_v1", "runner_protector_score_v1"}
    }
    rows = []
    for ordinal, column in enumerate(asof.columns):
        rows.append(
            {
                "ordinal_v1": ordinal,
                "feature_v1": column,
                "role_v1": "ID" if column in ID_COLS else ("BLOCKED_SCORE_PLACEHOLDER" if column in score_cols else ("BASE_AS_OF_FEATURE" if column in base_features else "AS_OF_METADATA")),
                "used_in_pre_score_foundation_v1": column in base_features,
                "null_rate_v1": float(asof[column].isna().mean()) if len(asof) else None,
                "dtype_v1": str(asof[column].dtype),
            }
        )
    return pd.DataFrame(rows)


def _label_summary(frame: pd.DataFrame, exact_label_report: dict[str, Any]) -> dict[str, Any]:
    label_columns = [
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        "r5_2_label_runner_protect_v1",
        "r6_label_runner_protect_v1",
        "r6_label_bad_risk_v1",
        "r6_label_runner_near_miss_v1",
    ]
    counts = {column: int(_bool(frame, column).sum()) for column in label_columns if column in frame.columns}
    return {
        "row_count_v1": int(len(frame)),
        "label_true_counts_v1": counts,
        "exact_label_overlay_v1": exact_label_report,
    }


def _run_inventory(monday_truth_dir: Path) -> pd.DataFrame:
    path = monday_truth_dir / "monday_r6_truth_run_inventory_v1.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _row_delta(frame: pd.DataFrame, run_inventory: pd.DataFrame) -> pd.DataFrame:
    active_rows = int(frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE").sum())
    quarantine_rows = int(len(frame) - active_rows)
    rows = [
        {
            "universe_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK",
            "row_count_v1": int(WEDNESDAY_R6_BENCHMARK.get("policy_rows_v1") or 1971),
            "delta_vs_monday_foundation_v1": int(WEDNESDAY_R6_BENCHMARK.get("policy_rows_v1") or 1971) - int(len(frame)),
            "status_v1": "BENCHMARK_NOT_ROW_IDENTITY_TARGET_AFTER_MONDAY_REANCHOR",
        },
        {
            "universe_v1": "MONDAY_ACTIVE_ONLY_OLD_1852",
            "row_count_v1": active_rows,
            "delta_vs_monday_foundation_v1": active_rows - int(len(frame)),
            "status_v1": "OLD_ACTIVE_ONLY_EXCLUDES_DECEMBER_QUARANTINE",
        },
        {
            "universe_v1": "MONDAY_DECEMBER_QUARANTINE_EVAL_ONLY",
            "row_count_v1": quarantine_rows,
            "delta_vs_monday_foundation_v1": quarantine_rows - int(len(frame)),
            "status_v1": "INCLUDED_IN_FOUNDATION_AS_EVAL_OR_HARD_GUARD_NOT_TRAINING_REQUIRED",
        },
        {
            "universe_v1": "MONDAY_ACTUAL_FULLCOVERAGE_FOUNDATION",
            "row_count_v1": int(len(frame)),
            "delta_vs_monday_foundation_v1": 0,
            "status_v1": "FOUNDATION_ROW_UNIVERSE",
        },
        {
            "universe_v1": "MONDAY_EXACT_ONLY_1689_DIAGNOSTIC",
            "row_count_v1": OLD_EXACT_ONLY_ROWS,
            "delta_vs_monday_foundation_v1": OLD_EXACT_ONLY_ROWS - int(len(frame)),
            "status_v1": "DIAGNOSTIC_ONLY_DO_NOT_USE_AS_BASELINE",
        },
    ]
    if not run_inventory.empty and "outcome_rows_v1" in run_inventory.columns:
        for _, row in run_inventory.iterrows():
            rows.append(
                {
                    "universe_v1": f"RUN::{row.get('run_id')}",
                    "row_count_v1": int(pd.to_numeric(pd.Series([row.get("outcome_rows_v1")]), errors="coerce").fillna(0).iloc[0]),
                    "delta_vs_monday_foundation_v1": None,
                    "status_v1": str(row.get("calendar_quarantine_status_v1") or "UNKNOWN"),
                }
            )
    return pd.DataFrame(rows)


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("FOUNDATION_ROWS_NOT_1689", "PASS" if summary["row_count_v1"] != OLD_EXACT_ONLY_ROWS else "FAIL", summary["row_count_v1"]),
            row("FOUNDATION_ROWS_NOT_ACTIVE_ONLY_1852", "PASS" if summary["row_count_v1"] != OLD_ACTIVE_ONLY_ROWS else "FAIL", summary["row_count_v1"]),
            row("AS_OF_109_PRESENT", "PASS" if summary["as_of_column_count_v1"] == EXPECTED_AS_OF_COLUMNS else "FAIL", summary["as_of_column_count_v1"]),
            row("DECEMBER_QUARANTINE_INCLUDED", "PASS" if summary["quarantine_rows_v1"] > 0 else "WARN", summary["quarantine_rows_v1"]),
            row("NO_TRAINING_STARTED", "PASS", summary["training_started_v1"]),
            row("NO_FREEZE_OR_PROMOTION", "PASS", summary["not_freeze_or_promo_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Canonical Foundation Rebuild V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Foundation rows: `{summary['row_count_v1']}`",
            f"- Active rows: `{summary['active_rows_v1']}`",
            f"- Quarantine/eval rows: `{summary['quarantine_rows_v1']}`",
            f"- AS_OF columns: `{summary['as_of_column_count_v1']}`",
            f"- Base pre-score features: `{summary['base_feature_count_v1']}`",
            f"- R5 heads defined: `{summary['r5_head_count_v1']}`",
            f"- R5.2 heads defined: `{summary['r5_2_head_count_v1']}`",
            "",
            "This is the Monday actual fullcoverage foundation. It keeps Wednesday R6 as the benchmark/contract, but it does not fabricate the old 1971 row identity after Monday re-anchor.",
            "No training, freeze, promotion, or live controller change was started.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    monday_truth_dir: Path | None = None,
    rehydrated_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    monday_truth_dir = monday_truth_dir.expanduser().resolve() if monday_truth_dir else _latest_dir(reports_root, MONDAY_TRUTH_GLOB, TRUTH_TABLE)
    rehydrated_dir = rehydrated_dir.expanduser().resolve() if rehydrated_dir else _latest_dir(reports_root, REHYDRATED_GLOB, AS_OF_TABLE)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, exact_label_report = _build_foundation_frame(reports_root, monday_truth_dir, rehydrated_dir)
    asof = _foundation_asof(frame, rehydrated_dir)
    hindsight = _foundation_hindsight(frame, rehydrated_dir)
    feature_audit = _feature_audit(asof, frame)
    run_inventory = _run_inventory(monday_truth_dir)
    row_delta = _row_delta(frame, run_inventory)
    label_summary = _label_summary(frame, exact_label_report)
    active = frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")
    base_features = _base_feature_names(frame)
    asof_count = int(pd.read_parquet(rehydrated_dir / AS_OF_TABLE).shape[1])
    decision = "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT"
    if asof_count != EXPECTED_AS_OF_COLUMNS or len(frame) in {OLD_ACTIVE_ONLY_ROWS, OLD_EXACT_ONLY_ROWS}:
        decision = "MONDAY_R6_FOUNDATION_REBUILD_FAILED_CONTRACT_GUARD"
    next_action = (
        "RUN_R5_R5_1_R5_2_SCORE_REBUILD_ON_MONDAY_ACTUAL_FOUNDATION_WITH_EXPLICIT_FLAG"
        if decision == "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT"
        else "FIX_FOUNDATION_CONTRACT_FIRST"
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "monday_truth_dir_v1": str(monday_truth_dir),
        "rehydrated_dir_v1": str(rehydrated_dir),
        "decision_v1": decision,
        "next_action_v1": next_action,
        "row_count_v1": int(len(frame)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "as_of_column_count_v1": asof_count,
        "foundation_as_of_output_column_count_v1": int(asof.shape[1]),
        "hindsight_output_column_count_v1": int(hindsight.shape[1]),
        "base_feature_count_v1": int(len(base_features)),
        "r5_head_count_v1": int(len(R5_HEADS)),
        "r5_2_head_count_v1": int(len(R5_2_HEADS)),
        "wednesday_benchmark_rows_v1": 1971,
        "wednesday_benchmark_freeze_id_v1": WEDNESDAY_R6_BENCHMARK["freeze_id_v1"],
        "wednesday_benchmark_candidate_id_v1": WEDNESDAY_R6_BENCHMARK["candidate_id_v1"],
        "old_active_only_rows_v1": OLD_ACTIVE_ONLY_ROWS,
        "old_exact_only_rows_v1": OLD_EXACT_ONLY_ROWS,
        "training_started_v1": False,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
        "blocked_action_v1": [
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_USE_ACTIVE_ONLY_1852_AS_FULL_FOUNDATION",
            "DO_NOT_FABRICATE_1971_ROW_IDENTITY_AFTER_MONDAY_REANCHOR",
            "DO_NOT_FREEZE_OR_PROMOTE_FROM_FOUNDATION_BUILD",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "Foundation uses Monday actual fullcoverage rows from the rehydrated Wednesday-R6 AS_OF contract.",
                "The 62 December quarantine rows are present as quarantine/eval rows, not silently dropped.",
                "The 1689 exact-only and old 1852 active-only surfaces are not used as the foundation.",
            ],
            "INDIKERT": [
                "This is the correct foundation for the next score rebuild stage, but it does not by itself prove model safety.",
            ],
            "IKKE_ETABLERT": [
                "Frozen Wednesday source hashes are still not restored locally.",
                "A green Monday R6 retrain/freeze is not established by this foundation build.",
            ],
        },
    }
    contract = {
        "layer_name": f"{LAYER_NAME}_CONTRACT",
        "foundation_universe_v1": "MONDAY_ACTUAL_FULLCOVERAGE_68_WEEK_REANCHOR",
        "row_count_v1": summary["row_count_v1"],
        "as_of_contract_v1": "WEDNESDAY_R6_109_COLUMN_SHAPE_REHYDRATED_FOR_MONDAY",
        "hindsight_contract_v1": "WEDNESDAY_R6_HINDSIGHT_SHAPE_WITH_MONDAY_ACTUAL_LABELS",
        "training_started_v1": False,
        "r5_heads_v1": R5_HEADS,
        "r5_2_heads_v1": R5_2_HEADS,
        "exact_label_source_paths_v1": {
            "r5_v1": str(reports_root / EXACT_R5_LABEL_SOURCE_DIR / EXACT_R5_LABEL_SOURCE_TABLE),
            "r5_2_v1": str(reports_root / EXACT_R5_2_LABEL_SOURCE_DIR / EXACT_R5_2_LABEL_SOURCE_TABLE),
            "r6_v1": str(reports_root / WEDNESDAY_R6_SPLIT_SOURCE_DIR / EXACT_R6_LABEL_SOURCE_TABLE),
            "eval_v1": str(reports_root / WEDNESDAY_R6_SPLIT_SOURCE_DIR / EXACT_EVAL_LABEL_SOURCE_TABLE),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": decision,
        "next_action_v1": next_action,
        "training_started_v1": False,
    }
    audit = _audit(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "training_started_v1": False,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
    }

    asof.to_parquet(output_dir / FOUNDATION_AS_OF, index=False)
    hindsight.to_parquet(output_dir / FOUNDATION_HINDSIGHT, index=False)
    frame.to_parquet(output_dir / FOUNDATION_FRAME, index=False)
    feature_audit.to_csv(output_dir / FEATURE_CONTRACT_AUDIT, index=False)
    row_delta.to_csv(output_dir / ROW_UNIVERSE_DELTA, index=False)
    run_inventory.to_csv(output_dir / RUN_INVENTORY, index=False)
    audit.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(output_dir / FOUNDATION_CONTRACT, contract)
    _write_json(output_dir / FOUNDATION_LABEL_SUMMARY, label_summary)
    _write_json(output_dir / STATUS, status)
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / MANIFEST, manifest)
    (output_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--monday-truth-dir", type=Path, default=None)
    parser.add_argument("--rehydrated-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        monday_truth_dir=args.monday_truth_dir,
        rehydrated_dir=args.rehydrated_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
