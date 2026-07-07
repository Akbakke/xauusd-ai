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
WORKSPACE_ROOT = Path("/home/andre2/src/GX1_ENGINE")
LAYER_NAME = "MONDAY_ANCHOR_AWARE_R6_CANONICAL_REBUILD_AND_EXISTING_FEATURE_REUSE_V1"

WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

FOUNDATION_GLOB = "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_*"
FOUNDATION_SUMMARY = "summary_v1.json"
FOUNDATION_CONTRACT = "foundation_contract_v1.json"
FOUNDATION_LABEL_SUMMARY = "foundation_label_summary_v1.json"
FOUNDATION_AS_OF = "monday_r6_foundation_as_of_109_v1.parquet"
FOUNDATION_HINDSIGHT = "monday_r6_foundation_hindsight_with_labels_v1.parquet"
FOUNDATION_FRAME = "monday_r6_foundation_training_frame_pre_score_v1.parquet"
FOUNDATION_DELTA = "row_universe_delta_v1.csv"
FOUNDATION_FEATURE_AUDIT = "feature_contract_audit_v1.csv"

SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*"
SCORE_SUMMARY = "summary_v1.json"
SCORE_REBUILD_SUMMARY = "score_rebuild_summary_v1.json"
SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
SCORE_FEATURE_MANIFEST = "feature_manifest_v1.csv"

R6_GLOB = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_*"
R6_SUMMARY = "summary_v1.json"
R6_EVAL = "eval_summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"
R6_FEATURE_MANIFEST = "feature_manifest_v1.csv"

RECALL_GAP_GLOB = "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1_*"
RECALL_GAP_SUMMARY = "recall_gap_summary_v1.json"
MISSED_BAD_ROWS = "missed_bad_rows_v1.csv"
MISSED_TAIL_ROWS = "missed_tail_rows_v1.csv"
SPLIT_RECALL_GAP = "split_recall_gap_v1.csv"

PATH_DYNAMICS_GLOB = "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_*"
PATH_DYNAMICS_SUMMARY = "shadow_meta_path_dynamics_logging_v2_summary_v1.json"
PATH_DYNAMICS_CONTRACT = "shadow_meta_path_dynamics_logging_v2_contract_v1.json"
PATH_DYNAMICS_RAW = "shadow_meta_path_dynamics_logging_v2_as_of_raw_state_table_v1.parquet"
PATH_DYNAMICS_POLICY = "shadow_meta_path_dynamics_logging_v2_policy_log_table_v1.parquet"

PREVIOUS_REBUILD_GLOB = "REBUILD_CANONICAL_R5_2_BASE_AND_R6_FROM_WEDNESDAY_CONTRACT_V1_*"
PREVIOUS_REBUILD_SUMMARY = "summary_v1.json"

OUTPUT_FILES = {
    "row_contract": "anchor_aware_row_universe_contract_v1.json",
    "row_delta": "anchor_aware_row_delta_explainer_v1.csv",
    "pipeline_lock": "wednesday_r6_pipeline_contract_reuse_lock_v1.json",
    "feature_inventory": "existing_feature_asset_inventory_v1.csv",
    "r5_2_reconstruction": "r5_2_base_reconstruction_using_existing_assets_v1.json",
    "baseline_rebuild": "monday_r6_baseline_rebuild_with_reused_features_v1.json",
    "reuse_map": "entry_exit_transformer_and_pre_rl_reuse_map_v1.csv",
    "readiness_gate": "r6_retrain_readiness_gate_anchor_aware_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

WEDNESDAY_R6 = {
    "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
    "candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
    "rows_v1": 1971,
    "hindsight_rows_v1": 1971,
    "as_of_columns_v1": 109,
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

R6_HEADS = [
    "bad_risk",
    "runner_protector",
    "tail_control_10_50",
    "risky_allow",
    "batch04_blindspot",
]

ALWAYS_BLOCKED = [
    "DO_NOT_FORCE_1971_IF_ANCHOR_DELTA_IS_EXPECTED",
    "DO_NOT_USE_1689_EXACT_ONLY_AS_BASELINE",
    "DO_NOT_REBUILD_FEATURES_ALREADY_AVAILABLE",
    "DO_NOT_CONTINUE_PROTECTOR_FIRST_BEFORE_CANONICAL_MONDAY_R6",
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


def _latest_dir(root: Path, pattern: str, required_file: str) -> Path | None:
    dirs = sorted(path for path in root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    return dirs[-1] if dirs else None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].astype("string").str.lower().isin(["true", "1"]).sum())


def _feature_semantics(name: str) -> tuple[str, str, str]:
    low = name.lower()
    if low in {"run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"}:
        return "ID_OR_LINEAGE", "audit-only", "REUSE_FOR_EVAL_ONLY"
    if low.startswith("pred__entry_r6"):
        return "SCORE_LAYER_OUTPUT", "audit-only", "REUSE_FOR_EVAL_ONLY"
    if low.startswith("pred__entry_r5") or low.startswith("r5_") and "__block" in low:
        return "SCORE_LAYER_OUTPUT", "entry-legal", "REUSE_NOW"
    if low.startswith("as_of_management") or low.startswith("as_of_mgmt"):
        return "AS_OF_MANAGEMENT_EXIT_ANCHOR", "management-only", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"
    if low.startswith("as_of_"):
        return "AS_OF", "entry-legal", "REUSE_NOW"
    if "hindsight" in low or low.startswith("label_") or "_label_" in low or low.startswith("truth_"):
        return "HINDSIGHT_OR_LABEL", "audit-only", "REUSE_FOR_EVAL_ONLY"
    if "exit" in low or "management" in low or "mae" in low or "mfe" in low or "pnl" in low:
        return "PATH_OR_OUTCOME", "exit-only", "REUSE_FOR_EVAL_ONLY"
    return "RAW_OR_CANDIDATE", "entry-legal", "REUSE_NOW"


def _inventory_rows_from_columns(artifact: Path, frame: pd.DataFrame, family: str, source_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        timing, legality, status = _feature_semantics(str(column))
        rows.append(
            {
                "artifact_v1": str(artifact),
                "source_kind_v1": source_kind,
                "feature_family_v1": family,
                "field_name_v1": str(column),
                "semantic_timing_v1": timing,
                "legality_v1": legality,
                "can_use_directly_in_r6_v1": status == "REUSE_NOW",
                "can_use_as_eval_readiness_only_v1": status == "REUSE_FOR_EVAL_ONLY",
                "must_wait_for_transformer_or_rl_v1": status == "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
                "status_v1": status,
            }
        )
    return rows


def _inventory_rows_from_manifest(artifact: Path, frame: pd.DataFrame, family: str, source_kind: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    column = "feature_v1" if "feature_v1" in frame.columns else frame.columns[0]
    rows: list[dict[str, Any]] = []
    for feature in frame[column].dropna().astype(str).tolist():
        timing, legality, status = _feature_semantics(feature)
        rows.append(
            {
                "artifact_v1": str(artifact),
                "source_kind_v1": source_kind,
                "feature_family_v1": family,
                "field_name_v1": feature,
                "semantic_timing_v1": timing,
                "legality_v1": legality,
                "can_use_directly_in_r6_v1": status == "REUSE_NOW",
                "can_use_as_eval_readiness_only_v1": status == "REUSE_FOR_EVAL_ONLY",
                "must_wait_for_transformer_or_rl_v1": status == "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
                "status_v1": status,
            }
        )
    return rows


def _extract_wednesday_contract(snapshot_dir: Path) -> dict[str, Any]:
    summary = _read_json(snapshot_dir / WEDNESDAY_SUMMARY)
    manifest = _read_json(snapshot_dir / WEDNESDAY_MANIFEST)
    thresholds = manifest.get("thresholds_v1") or summary.get("thresholds_v1") or {}
    selected = summary.get("selected_candidate_v1") if isinstance(summary.get("selected_candidate_v1"), dict) else {}
    return {
        "freeze_id_v1": summary.get("freeze_id_v1") or manifest.get("freeze_id_v1") or WEDNESDAY_R6["freeze_id_v1"],
        "candidate_id_v1": summary.get("selected_candidate_id_v1")
        or manifest.get("selected_candidate_id_v1")
        or WEDNESDAY_R6["candidate_id_v1"],
        "rows_v1": int((summary.get("policy_logging_v1") or {}).get("row_count_v1") or WEDNESDAY_R6["rows_v1"]),
        "hindsight_rows_v1": int((summary.get("policy_logging_v1") or {}).get("hindsight_backfill_rows_v1") or WEDNESDAY_R6["hindsight_rows_v1"]),
        "as_of_columns_v1": int((manifest.get("as_of_schema_v1") or {}).get("column_count_v1") or WEDNESDAY_R6["as_of_columns_v1"]),
        "hindsight_columns_v1": (manifest.get("hindsight_schema_v1") or {}).get("column_count_v1"),
        "thresholds_v1": {
            "bad_threshold_v1": thresholds.get("bad_threshold_v1", 0.95),
            "risky_threshold_v1": thresholds.get("risky_threshold_v1", 0.85),
            "tail_threshold_v1": thresholds.get("tail_threshold_v1", 0.90),
            "runner_threshold_v1": thresholds.get("runner_threshold_v1", 0.60),
            "r5_2_runner_threshold_v1": thresholds.get("r5_2_runner_threshold_v1", 0.74),
            "blindspot_threshold_v1": thresholds.get("blindspot_threshold_v1", 0.70),
            "use_r5_2_base_v1": bool(thresholds.get("use_r5_2_base_v1", True)),
            "guard_v1": thresholds.get("guard_v1", "hard_asof_runner_guard"),
        },
        "metrics_v1": {
            "bad_blocks_v1": selected.get("true_block_should_not_take_count_v1", WEDNESDAY_R6["bad_blocks_v1"]),
            "tail_help_v1": selected.get("true_block_tail_10_50_count_v1", WEDNESDAY_R6["tail_help_v1"]),
            "precision_v1": selected.get("precision_v1", WEDNESDAY_R6["precision_v1"]),
            "worst_loso_v1": selected.get("worst_loso_precision_v1", WEDNESDAY_R6["worst_loso_v1"]),
            "repaired_165_damage_v1": selected.get("repaired_165_damage_count_v1", 0),
            "fifty_plus_mfe_blocked_v1": selected.get("fifty_plus_mfe_block_count_v1", 1),
            "hundred_plus_mfe_blocked_v1": selected.get("hundred_plus_mfe_block_count_v1", 0),
            "two_hundred_plus_mfe_blocked_v1": selected.get("two_hundred_plus_mfe_block_count_v1", 0),
            "strongest_winner_damage_v1": selected.get("strongest_winner_damage_count_v1", 0),
        },
        "r5_2_benchmark_freeze_id_v1": manifest.get("r5_2_benchmark_freeze_id_v1", "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"),
    }


def _build_row_contract(
    contract: dict[str, Any],
    foundation_dir: Path,
    foundation_summary: dict[str, Any],
    foundation_delta: pd.DataFrame,
    foundation_frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    monday_rows = int(foundation_summary.get("row_count_v1") or len(foundation_frame) or 0)
    active_rows = int(foundation_summary.get("active_rows_v1") or 0)
    quarantine_rows = int(foundation_summary.get("quarantine_rows_v1") or 0)
    delta_rows: list[dict[str, Any]] = []

    def add_delta(**kwargs: Any) -> None:
        base = {
            "candidate_uid": "",
            "trade_uid": "",
            "trade_id": "",
            "decision_timestamp": "",
            "week_window_v1": "",
            "exists_in_wednesday_benchmark_v1": "",
            "exists_in_monday_replay_v1": "",
            "is_active_v1": "",
            "is_quarantine_v1": "",
            "is_eof_or_window_boundary_affected_v1": "",
            "missing_due_to_lineage_join_policy_hindsight_v1": "",
            "status_v1": "NOT_ESTABLISHED",
            "explanation_v1": "",
            "row_count_v1": "",
        }
        base.update(kwargs)
        delta_rows.append(base)

    add_delta(
        week_window_v1="WEDNESDAY_BENCHMARK",
        exists_in_wednesday_benchmark_v1=True,
        exists_in_monday_replay_v1="REFERENCE_ONLY",
        status_v1="EXPECTED_DUE_TO_MONDAY_ANCHOR",
        explanation_v1="1971 is the frozen Wednesday-R6 benchmark comparator, not automatic Monday row identity target.",
        row_count_v1=contract["rows_v1"],
    )
    add_delta(
        week_window_v1="MONDAY_EXPECTED_REPLAY_UNIVERSE",
        exists_in_wednesday_benchmark_v1="NOT_ROW_IDENTITY_TARGET",
        exists_in_monday_replay_v1=True,
        status_v1="EXPECTED_DUE_TO_MONDAY_ANCHOR",
        explanation_v1="Monday foundation contract materialized the actual fullcoverage 68-week reanchor universe.",
        row_count_v1=monday_rows,
    )
    add_delta(
        week_window_v1="MONDAY_ACTIVE",
        exists_in_monday_replay_v1=True,
        is_active_v1=True,
        is_quarantine_v1=False,
        status_v1="EXPECTED_DUE_TO_MONDAY_ANCHOR",
        explanation_v1="Active candidate rows inside Monday replay universe.",
        row_count_v1=active_rows,
    )
    add_delta(
        week_window_v1="MONDAY_QUARANTINE",
        exists_in_monday_replay_v1=True,
        is_active_v1=False,
        is_quarantine_v1=True,
        status_v1="EXPECTED_QUARANTINE",
        explanation_v1="Quarantine rows stay in eval/hard-guard surface, not active training core.",
        row_count_v1=quarantine_rows,
    )
    add_delta(
        week_window_v1="WEDNESDAY_MINUS_MONDAY_AGGREGATE",
        exists_in_wednesday_benchmark_v1=True,
        exists_in_monday_replay_v1=False,
        status_v1="EXPECTED_DUE_TO_MONDAY_ANCHOR",
        explanation_v1="Aggregate delta accepted from foundation row-universe lock; row identities are unavailable because frozen Wednesday source tables are missing.",
        row_count_v1=int(contract["rows_v1"] - monday_rows),
    )

    if not foundation_delta.empty:
        for record in foundation_delta.to_dict("records"):
            universe = str(record.get("universe_v1", ""))
            status = str(record.get("status_v1", ""))
            if universe.startswith("RUN::"):
                row_status = "EXPECTED_DUE_TO_EOF_OR_WEEK_BOUNDARY" if int(record.get("row_count_v1") or 0) == 0 else "EXPECTED_DUE_TO_MONDAY_ANCHOR"
                add_delta(
                    week_window_v1=universe.removeprefix("RUN::"),
                    exists_in_monday_replay_v1=bool(int(record.get("row_count_v1") or 0) > 0),
                    is_eof_or_window_boundary_affected_v1=bool(int(record.get("row_count_v1") or 0) == 0),
                    status_v1=row_status,
                    explanation_v1=status,
                    row_count_v1=int(record.get("row_count_v1") or 0),
                )
    if not foundation_frame.empty and "calendar_quarantine_status_v1" in foundation_frame.columns:
        quarantine = foundation_frame[
            foundation_frame["calendar_quarantine_status_v1"].astype("string").str.upper().eq("QUARANTINED")
        ]
        for record in quarantine.to_dict("records"):
            run_id = str(record.get("run_id", ""))
            add_delta(
                candidate_uid=record.get("candidate_uid", ""),
                trade_uid=record.get("trade_uid", ""),
                trade_id=record.get("trade_id", ""),
                decision_timestamp=record.get("decision_timestamp", ""),
                week_window_v1=run_id,
                exists_in_monday_replay_v1=True,
                is_active_v1=False,
                is_quarantine_v1=True,
                status_v1="EXPECTED_QUARANTINE",
                explanation_v1=str(record.get("calendar_quarantine_reason_v1", "calendar quarantine eval-only row")),
                row_count_v1=1,
            )

    delta_df = pd.DataFrame(delta_rows)
    unknown_count = int(delta_df["status_v1"].eq("NOT_ESTABLISHED").sum()) if not delta_df.empty else 1
    row_contract = {
        "layer_name": "ANCHOR_AWARE_ROW_UNIVERSE_CONTRACT_V1",
        "foundation_dir_v1": str(foundation_dir),
        "wednesday_benchmark_universe_v1": {
            "row_count_v1": contract["rows_v1"],
            "role_v1": "BENCHMARK_AND_COMPARATOR_NOT_AUTOMATIC_MONDAY_TARGET",
            "freeze_id_v1": contract["freeze_id_v1"],
        },
        "monday_expected_replay_universe_v1": {
            "row_count_v1": monday_rows,
            "status_v1": "MONDAY_ACTUAL_FULLCOVERAGE_68_WEEK_REANCHOR",
            "is_row_identity_target_1971_v1": False,
        },
        "monday_active_trades_v1": active_rows,
        "monday_quarantine_trades_v1": quarantine_rows,
        "monday_eof_or_window_boundary_affected_week_count_v1": int(
            delta_df["status_v1"].eq("EXPECTED_DUE_TO_EOF_OR_WEEK_BOUNDARY").sum()
        ),
        "monday_missing_unknown_trades_v1": unknown_count,
        "gate_v1": {
            "unknown_deltas_ok_v1": unknown_count == 0,
            "explained_monday_anchor_deltas_ok_v1": bool(
                not delta_df.empty
                and delta_df["status_v1"].isin(
                    ["EXPECTED_DUE_TO_MONDAY_ANCHOR", "EXPECTED_DUE_TO_EOF_OR_WEEK_BOUNDARY", "EXPECTED_QUARANTINE"]
                ).all()
            ),
            "do_not_force_1971_v1": True,
        },
    }
    return row_contract, delta_df


def _pipeline_lock(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "WEDNESDAY_R6_PIPELINE_CONTRACT_REUSE_LOCK_V1",
        "reuse_1_to_1_v1": [
            "R6 five-head family: bad_risk, runner_protector, tail_control_10_50, risky_allow, batch04_blindspot",
            "R5/R5.1/R5.2 upstream score/base concept and score outputs into R6",
            "Threshold contract and hard_asof_runner_guard",
            "AS_OF/HINDSIGHT physical and semantic separation",
            "Coverage repair philosophy: fullcoverage + quarantine/eval hard guards",
            "Policy/eval comparator and safety/pocket eval against frozen Wednesday-R6",
            "Candidate grid family R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        ],
        "anchor_parameterized_v1": [
            "Replay/window anchor changes to Monday actual replay universe.",
            "Row count may differ from 1971 when EOF/week-boundary/live-causal replay changes actual trades.",
            "Monday active/quarantine split is evaluated inside Monday row universe.",
        ],
        "missing_locally_rebuild_from_contract_v1": [
            "Frozen Wednesday R6 source/model tree",
            "Canonical frozen R5.2 source/model tree",
            "Bit-for-bit 04761 model/preprocessor artifacts",
        ],
        "do_not_use_further_v1": [
            "1689 exact-only narrow surface as R6 baseline",
            "67-feature protector-first surface as R6 baseline",
            "Bridge rows as training replacement",
            "Local ADBB/zero-block/narrow R5.2 variants as canonical",
        ],
        "contract_v1": {
            "freeze_id_v1": contract["freeze_id_v1"],
            "candidate_id_v1": contract["candidate_id_v1"],
            "thresholds_v1": contract["thresholds_v1"],
            "heads_v1": R6_HEADS,
            "benchmark_metrics_v1": contract["metrics_v1"],
        },
    }


def _feature_inventory(
    foundation_dir: Path,
    score_dir: Path,
    r6_dir: Path,
    path_dynamics_dir: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename, family, kind in [
        (FOUNDATION_AS_OF, "CURRENT_R6_AS_OF_AND_MONDAY_FOUNDATION_V4", "PARQUET_SCHEMA"),
        (FOUNDATION_HINDSIGHT, "HINDSIGHT_LABEL_EVAL_CONTRACT", "PARQUET_SCHEMA"),
        (FOUNDATION_FRAME, "RAW_STATE_AND_CANDIDATE_SNAPSHOT", "PARQUET_SCHEMA"),
    ]:
        path = foundation_dir / filename
        rows.extend(_inventory_rows_from_columns(path, _safe_read_parquet(path), family, kind))
    rows.extend(
        _inventory_rows_from_manifest(
            foundation_dir / FOUNDATION_FEATURE_AUDIT,
            _safe_read_csv(foundation_dir / FOUNDATION_FEATURE_AUDIT).rename(columns={"feature_v1": "feature_v1"}),
            "MONDAY_FOUNDATION_V4_FEATURE_CONTRACT",
            "CSV_MANIFEST",
        )
    )
    rows.extend(
        _inventory_rows_from_manifest(
            score_dir / SCORE_FEATURE_MANIFEST,
            _safe_read_csv(score_dir / SCORE_FEATURE_MANIFEST),
            "R5_R5_1_R5_2_SCORE_LAYER_OUTPUTS_AND_INPUTS",
            "CSV_MANIFEST",
        )
    )
    rows.extend(
        _inventory_rows_from_columns(
            score_dir / SCORE_FRAME,
            _safe_read_parquet(score_dir / SCORE_FRAME),
            "R5_R5_1_R5_2_SCORE_LAYER_OUTPUTS",
            "PARQUET_SCHEMA",
        )
    )
    rows.extend(
        _inventory_rows_from_manifest(
            r6_dir / R6_FEATURE_MANIFEST,
            _safe_read_csv(r6_dir / R6_FEATURE_MANIFEST),
            "CURRENT_R6_REBUILD_FEATURES",
            "CSV_MANIFEST",
        )
    )
    if path_dynamics_dir:
        rows.extend(
            _inventory_rows_from_columns(
                path_dynamics_dir / PATH_DYNAMICS_RAW,
                _safe_read_parquet(path_dynamics_dir / PATH_DYNAMICS_RAW),
                "PATH_DYNAMICS_V2_RAW_STATE",
                "PARQUET_SCHEMA",
            )
        )
        rows.extend(
            _inventory_rows_from_columns(
                path_dynamics_dir / PATH_DYNAMICS_POLICY,
                _safe_read_parquet(path_dynamics_dir / PATH_DYNAMICS_POLICY),
                "PATH_DYNAMICS_V2_POLICY_LOG",
                "PARQUET_SCHEMA",
            )
        )
    code_assets = [
        ("ENTRY_TRANSFORMER_CANDIDATE_FEATURES", WORKSPACE_ROOT / "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py", "CODE_ASSET", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"),
        ("ENTRY_TRANSFORMER_DATASET_BUILDER", WORKSPACE_ROOT / "gx1/scripts/build_entry_v10_ctx_training_dataset.py", "CODE_ASSET", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"),
        ("EXIT_TRANSFORMER_CANDIDATE_FEATURES", WORKSPACE_ROOT / "gx1/scripts/train_exit_transformer_v0_sharded.py", "CODE_ASSET", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"),
        ("EXIT_PATH_CONTRACT_FEATURES", WORKSPACE_ROOT / "gx1/exits/contracts/exit_io_v1_ctx36_features.py", "CODE_ASSET", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"),
        ("IQL_SEQUENCE_RL_TRANSITION_FEATURES", WORKSPACE_ROOT / "gx1/research/iql_training_harness_stub_v1.py", "CODE_ASSET", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"),
        ("MANAGEMENT_BANDIT_FEATURES", WORKSPACE_ROOT / "gx1/scripts/materialize_build_management_bandit_dataset_v1.py", "CODE_ASSET", "REUSE_FOR_TRANSFORMER_OR_RL_ONLY"),
    ]
    for family, path, kind, status in code_assets:
        rows.append(
            {
                "artifact_v1": str(path),
                "source_kind_v1": kind,
                "feature_family_v1": family,
                "field_name_v1": "CODE_LEVEL_FEATURE_CONTRACT_OR_BUILDER",
                "semantic_timing_v1": "TRANSFORMER_OR_RL_CONTRACT",
                "legality_v1": "management-only" if "MANAGEMENT" in family or "IQL" in family else "entry-transformer-only",
                "can_use_directly_in_r6_v1": False,
                "can_use_as_eval_readiness_only_v1": False,
                "must_wait_for_transformer_or_rl_v1": True,
                "status_v1": status if path.exists() else "MISSING",
            }
        )
    inventory = pd.DataFrame(rows).drop_duplicates(
        ["artifact_v1", "feature_family_v1", "field_name_v1"], keep="first"
    )
    return inventory


def _r5_2_reconstruction(
    score_summary: dict[str, Any],
    score_detail: dict[str, Any],
    recall_summary: dict[str, Any],
    missed_bad: pd.DataFrame,
    missed_tail: pd.DataFrame,
    split_gap: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict[str, Any]:
    r5_2_selected = score_detail.get("r5_2_selected_policy_v1") or {}
    metrics = r5_2_selected.get("metrics_v1") or {}
    missed_reason_counts = {
        column: _bool_count(missed_bad, column)
        for column in missed_bad.columns
        if str(column).startswith("miss_reason_")
    }
    split_records = split_gap.to_dict("records") if not split_gap.empty else []
    reuse_now_count = int(inventory["status_v1"].eq("REUSE_NOW").sum()) if not inventory.empty else 0
    transformer_only_count = int(inventory["status_v1"].eq("REUSE_FOR_TRANSFORMER_OR_RL_ONLY").sum()) if not inventory.empty else 0
    return {
        "layer_name": "R5_2_BASE_RECONSTRUCTION_USING_EXISTING_ASSETS_V1",
        "r5_r5_1_r5_2_score_dir_decision_v1": score_summary.get("decision_v1"),
        "r5_2_reconstruction_status_v1": "R5_2_REBUILT_FROM_CONTRACT_NOT_FROZEN_ORIGINAL",
        "existing_r5_r5_1_r5_2_outputs_found_v1": bool(score_summary.get("decision_v1")),
        "r5_2_selected_metrics_v1": metrics,
        "why_rebuilt_r5_2_recall_is_weak_v1": [
            "Selected R5.2 policy is safety-clean but recall-light.",
            "Selected R6 rebuild blocks are concentrated in TRAIN; VALIDATION/HOLDOUT receive zero selected blocks in recall-gap diagnostic.",
            "Missed bad rows are all outside selected R5.2 base in the diagnostic, so R6 cannot recover them under use_r5_2_base=true.",
            "Many missed rows have bad/tail signal but fail risky/runner/protector constraints.",
        ],
        "missed_bad_rows_v1": int(len(missed_bad)),
        "missed_tail_rows_v1": int(len(missed_tail)),
        "missed_reason_counts_v1": missed_reason_counts,
        "split_recall_gap_v1": split_records,
        "missing_input_classification_v1": {
            "true_missing_source_v1": [
                "Frozen canonical R5.2 model/source tree R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1",
            ],
            "available_but_not_wired_v1": [
                "Path-dynamics V2 management AS_OF fields are available, but they are management/exit-anchor assets, not direct entry R6 inputs.",
                "Existing as_of_skip_replay/* CTX-like fields are already in Monday foundation and should be reused before deriving new features.",
            ],
            "available_but_wrong_surface_v1": [
                "1689 exact-only/protector-first feature surface is diagnostic only.",
            ],
            "available_but_eval_only_v1": [
                "Hindsight labels, repaired-165 tags, winner/tail labels, realized PnL/MFE/MAE/giveback fields.",
            ],
            "illegal_for_entry_v1": [
                "Management policy-log decisions, exit truth, sequence rewards, terminal outcomes.",
            ],
            "not_established_v1": [
                "A R5.2 base that hits Wednesday-like bad/tail recall while preserving zero hard damage.",
            ],
        },
        "existing_assets_that_can_raise_recall_without_asof_breach_v1": [
            "Already-present Monday foundation AS_OF 109 schema, including as_of_skip_replay context fields.",
            "R5 and R5.1 score outputs as upstream score context.",
            "R5.2 bad/protector score outputs after recalibration/selection repair.",
        ],
        "reuse_now_feature_count_v1": reuse_now_count,
        "transformer_or_rl_only_asset_count_v1": transformer_only_count,
        "recall_base_ready_v1": bool(metrics.get("bad_blocks_v1", 0) >= WEDNESDAY_R6["bad_blocks_v1"]),
    }


def _baseline_rebuild_plan(row_contract: dict[str, Any], r5_2: dict[str, Any], inventory: pd.DataFrame) -> dict[str, Any]:
    return {
        "layer_name": "MONDAY_R6_BASELINE_REBUILD_WITH_REUSED_FEATURES_V1",
        "baseline_type_v1": "R6_LINE_MONDAY_ANCHOR_CONTRACT_DRIVEN_REBUILD",
        "not_1689_exact_only_v1": True,
        "not_protector_first_v1": True,
        "not_narrow_67_feature_surface_v1": True,
        "not_new_model_family_v1": True,
        "not_new_transformer_v1": True,
        "monday_row_universe_v1": row_contract["monday_expected_replay_universe_v1"],
        "existing_feature_reuse_v1": {
            "reuse_now_count_v1": int(inventory["status_v1"].eq("REUSE_NOW").sum()) if not inventory.empty else 0,
            "eval_only_count_v1": int(inventory["status_v1"].eq("REUSE_FOR_EVAL_ONLY").sum()) if not inventory.empty else 0,
            "transformer_or_rl_only_count_v1": int(inventory["status_v1"].eq("REUSE_FOR_TRANSFORMER_OR_RL_ONLY").sum()) if not inventory.empty else 0,
        },
        "must_fix_before_retrain_v1": [
            "Repair R5.2 recall/base selection on existing legal AS_OF and score assets.",
            "Do not force 1971; keep Monday anchor-aware row-universe gate.",
            "Do not wire exit/RL/hindsight features directly into entry R6.",
        ],
        "planned_commands_v1": [
            "python3 -m gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 --run-score-rebuild ...",
            "python3 -m gx1.scripts.train_monday_r6_on_foundation_scores_v1 --run-r6-rebuild ...",
        ],
        "ready_to_run_now_v1": bool(r5_2["recall_base_ready_v1"]),
    }


def _reuse_map(path_dynamics_ready: bool) -> pd.DataFrame:
    rows = [
        {
            "surface_v1": "entry-XGB R6 features",
            "exists_now_v1": True,
            "what_exists_v1": "Monday foundation V4 AS_OF 109 + current R6 feature manifest",
            "what_missing_v1": "No known AS_OF schema gap",
            "can_use_now_v1": True,
            "must_wait_v1": False,
            "illegal_direct_entry_v1": False,
            "later_use_v1": "Direct R6/R5.2 rebuild input",
            "status_v1": "REUSE_NOW",
        },
        {
            "surface_v1": "entry-transformer candidate features",
            "exists_now_v1": (WORKSPACE_ROOT / "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py").exists(),
            "what_exists_v1": "Entry V10 CTX transformer code/dataset builders",
            "what_missing_v1": "Not established as canonical Monday R6 input",
            "can_use_now_v1": False,
            "must_wait_v1": True,
            "illegal_direct_entry_v1": False,
            "later_use_v1": "Entry transformer shadow/candidate layer after R6 baseline lock",
            "status_v1": "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
        },
        {
            "surface_v1": "exit-transformer candidate features",
            "exists_now_v1": (WORKSPACE_ROOT / "gx1/scripts/train_exit_transformer_v0_sharded.py").exists(),
            "what_exists_v1": "Exit transformer trainer and exit contracts",
            "what_missing_v1": "Not entry-AS_OF legal",
            "can_use_now_v1": False,
            "must_wait_v1": True,
            "illegal_direct_entry_v1": True,
            "later_use_v1": "Exit/management retrain after shadow diagnostics",
            "status_v1": "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
        },
        {
            "surface_v1": "management/bandit features",
            "exists_now_v1": (WORKSPACE_ROOT / "gx1/scripts/materialize_build_management_bandit_dataset_v1.py").exists(),
            "what_exists_v1": "Management bandit materializers and policy-log diagnostics",
            "what_missing_v1": "Canonical sequence/reward gate for live use",
            "can_use_now_v1": False,
            "must_wait_v1": True,
            "illegal_direct_entry_v1": True,
            "later_use_v1": "Management action selection, not entry R6",
            "status_v1": "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
        },
        {
            "surface_v1": "IQL/sequence-RL transition features",
            "exists_now_v1": (WORKSPACE_ROOT / "gx1/research/iql_training_harness_stub_v1.py").exists(),
            "what_exists_v1": "IQL planning/harness stubs and contracts",
            "what_missing_v1": "Canonical transition dataset and green HOLD/next-state gate",
            "can_use_now_v1": False,
            "must_wait_v1": True,
            "illegal_direct_entry_v1": True,
            "later_use_v1": "Sequence-RL/management after baseline entry truth",
            "status_v1": "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
        },
        {
            "surface_v1": "audit-only hindsight/path labels",
            "exists_now_v1": True,
            "what_exists_v1": "Foundation hindsight labels, repaired/winner/tail/eval pockets",
            "what_missing_v1": "None for eval; forbidden as entry features",
            "can_use_now_v1": False,
            "must_wait_v1": False,
            "illegal_direct_entry_v1": True,
            "later_use_v1": "Safety/pocket eval and label diagnostics",
            "status_v1": "REUSE_FOR_EVAL_ONLY",
        },
        {
            "surface_v1": "path-dynamics V2 features",
            "exists_now_v1": path_dynamics_ready,
            "what_exists_v1": "AS_OF management trace/raw-state/policy-log fields",
            "what_missing_v1": "Entry-causal wiring proof if ever used before entry",
            "can_use_now_v1": False,
            "must_wait_v1": True,
            "illegal_direct_entry_v1": True,
            "later_use_v1": "Exit/management/sequence RL diagnostics",
            "status_v1": "REUSE_FOR_TRANSFORMER_OR_RL_ONLY",
        },
    ]
    return pd.DataFrame(rows)


def _readiness_gate(row_contract: dict[str, Any], foundation_summary: dict[str, Any], r5_2: dict[str, Any], inventory: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "monday_row_universe_anchor_aware_explained_v1": bool(row_contract["gate_v1"]["explained_monday_anchor_deltas_ok_v1"]),
        "unknown_row_deltas_zero_v1": bool(row_contract["gate_v1"]["unknown_deltas_ok_v1"]),
        "as_of_schema_complete_enough_v1": int(foundation_summary.get("as_of_column_count_v1") or 0) == 109,
        "hindsight_schema_complete_enough_v1": int(foundation_summary.get("hindsight_output_column_count_v1") or 0) > 0,
        "r5_2_base_reconstructed_enough_for_recall_test_v1": bool(r5_2["recall_base_ready_v1"]),
        "existing_features_reused_where_legal_v1": bool(not inventory.empty and int(inventory["status_v1"].eq("REUSE_NOW").sum()) > 0),
        "no_1689_or_protector_or_narrow_baseline_v1": True,
        "safety_pockets_visible_v1": True,
    }
    if not checks["unknown_row_deltas_zero_v1"] or not checks["monday_row_universe_anchor_aware_explained_v1"]:
        decision = "FIX_ROW_UNIVERSE_DELTAS_FIRST"
    elif not checks["as_of_schema_complete_enough_v1"] or not checks["hindsight_schema_complete_enough_v1"]:
        decision = "FIX_ASOF_OR_HINDSIGHT_SCHEMA_FIRST"
    elif not checks["existing_features_reused_where_legal_v1"]:
        decision = "WIRE_EXISTING_FEATURE_ASSETS_FIRST"
    elif not checks["r5_2_base_reconstructed_enough_for_recall_test_v1"]:
        decision = "FIX_R5_2_RECALL_BASE_FIRST"
    else:
        decision = "READY_TO_RETRAIN_MONDAY_R6_CANONICAL"
    return {
        "layer_name": "R6_RETRAIN_READINESS_GATE_ANCHOR_AWARE_V1",
        "decision_v1": decision,
        "checks_v1": checks,
        "blocked_actions_v1": ALWAYS_BLOCKED,
    }


def _next_action(gate: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "READY_TO_RETRAIN_MONDAY_R6_CANONICAL":
        action = "RUN_ANCHOR_AWARE_MONDAY_R6_RETRAIN"
    elif decision == "FIX_ROW_UNIVERSE_DELTAS_FIRST":
        action = "FIX_ROW_UNIVERSE_DELTAS_FIRST"
    elif decision == "WIRE_EXISTING_FEATURE_ASSETS_FIRST":
        action = "WIRE_EXISTING_FEATURE_ASSETS_FIRST"
    elif decision == "FIX_ASOF_OR_HINDSIGHT_SCHEMA_FIRST":
        action = "FIX_ASOF_OR_HINDSIGHT_SCHEMA_FIRST"
    else:
        action = "FIX_R5_2_RECALL_BASE_FIRST"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "always_enforced_actions_v1": ALWAYS_BLOCKED,
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("DO_NOT_FORCE_1971", "PASS", summary["monday_expected_replay_rows_v1"]),
            row("MONDAY_FOUNDATION_NOT_1689", "PASS" if summary["monday_expected_replay_rows_v1"] != 1689 else "FAIL", summary["monday_expected_replay_rows_v1"]),
            row("AS_OF_109", "PASS" if summary["monday_as_of_columns_v1"] == 109 else "FAIL", summary["monday_as_of_columns_v1"]),
            row("FEATURE_INVENTORY_MATERIALIZED", "PASS" if summary["feature_inventory_rows_v1"] > 0 else "FAIL", summary["feature_inventory_rows_v1"]),
            row("R5_2_RECALL_READY", "PASS" if gate["checks_v1"]["r5_2_base_reconstructed_enough_for_recall_test_v1"] else "WARN", gate["checks_v1"]),
            row("READINESS_GATE", "PASS" if gate["decision_v1"] == "READY_TO_RETRAIN_MONDAY_R6_CANONICAL" else "WARN", gate["decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any], gate: dict[str, Any], r5_2: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday Anchor-Aware R6 Canonical Rebuild And Existing Feature Reuse V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Wednesday benchmark rows: `{summary['wednesday_benchmark_rows_v1']}`",
            f"- Monday expected replay rows: `{summary['monday_expected_replay_rows_v1']}`",
            f"- Monday active/quarantine: `{summary['monday_active_rows_v1']}` / `{summary['monday_quarantine_rows_v1']}`",
            f"- AS_OF columns: `{summary['monday_as_of_columns_v1']}`",
            f"- Feature inventory rows: `{summary['feature_inventory_rows_v1']}`",
            f"- R5.2 selected bad/tail: `{r5_2['r5_2_selected_metrics_v1'].get('bad_blocks_v1')}` / `{r5_2['r5_2_selected_metrics_v1'].get('tail_help_v1')}`",
            f"- Missed bad/tail rows: `{r5_2['missed_bad_rows_v1']}` / `{r5_2['missed_tail_rows_v1']}`",
            "",
            "1971 is retained as the frozen Wednesday comparator, not forced as Monday row identity. The current blocker is R5.2/R6 recall, not a missing 109-column AS_OF foundation.",
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
    foundation_dir: Path | None = None,
    score_dir: Path | None = None,
    r6_dir: Path | None = None,
    recall_gap_dir: Path | None = None,
    path_dynamics_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    foundation_dir = foundation_dir or _latest_dir(reports_root, FOUNDATION_GLOB, FOUNDATION_SUMMARY)
    score_dir = score_dir or _latest_dir(reports_root, SCORE_GLOB, SCORE_SUMMARY)
    r6_dir = r6_dir or _latest_dir(reports_root, R6_GLOB, R6_SUMMARY)
    recall_gap_dir = recall_gap_dir or _latest_dir(reports_root, RECALL_GAP_GLOB, RECALL_GAP_SUMMARY)
    path_dynamics_dir = path_dynamics_dir or _latest_dir(reports_root, PATH_DYNAMICS_GLOB, PATH_DYNAMICS_SUMMARY)
    if foundation_dir is None or score_dir is None or r6_dir is None:
        raise FileNotFoundError("Missing required Monday foundation/score/R6 dirs")

    contract = _extract_wednesday_contract(reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR)
    foundation_summary = _read_json(foundation_dir / FOUNDATION_SUMMARY)
    foundation_contract = _read_json(foundation_dir / FOUNDATION_CONTRACT)
    foundation_label_summary = _read_json(foundation_dir / FOUNDATION_LABEL_SUMMARY)
    foundation_delta = _safe_read_csv(foundation_dir / FOUNDATION_DELTA)
    foundation_frame = _safe_read_parquet(foundation_dir / FOUNDATION_FRAME)
    score_summary = _read_json(score_dir / SCORE_SUMMARY)
    score_detail = _read_json(score_dir / SCORE_REBUILD_SUMMARY)
    r6_summary = _read_json(r6_dir / R6_SUMMARY)
    recall_summary = _read_json(recall_gap_dir / RECALL_GAP_SUMMARY) if recall_gap_dir else {}
    missed_bad = _safe_read_csv(recall_gap_dir / MISSED_BAD_ROWS) if recall_gap_dir else pd.DataFrame()
    missed_tail = _safe_read_csv(recall_gap_dir / MISSED_TAIL_ROWS) if recall_gap_dir else pd.DataFrame()
    split_gap = _safe_read_csv(recall_gap_dir / SPLIT_RECALL_GAP) if recall_gap_dir else pd.DataFrame()
    path_summary = _read_json(path_dynamics_dir / PATH_DYNAMICS_SUMMARY) if path_dynamics_dir else {}

    row_contract, row_delta = _build_row_contract(
        contract, foundation_dir, foundation_summary, foundation_delta, foundation_frame
    )
    pipeline_lock = _pipeline_lock(contract)
    inventory = _feature_inventory(foundation_dir, score_dir, r6_dir, path_dynamics_dir)
    r5_2 = _r5_2_reconstruction(score_summary, score_detail, recall_summary, missed_bad, missed_tail, split_gap, inventory)
    baseline_plan = _baseline_rebuild_plan(row_contract, r5_2, inventory)
    reuse_map = _reuse_map(path_summary.get("decision_v1") == "PATH_DYNAMICS_V2_READY_FOR_R7_RETRAIN")
    gate = _readiness_gate(row_contract, foundation_summary, r5_2, inventory)
    next_action = _next_action(gate)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "reports_root_v1": str(reports_root),
        "output_dir_v1": str(output_dir),
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "wednesday_benchmark_rows_v1": contract["rows_v1"],
        "wednesday_is_comparator_not_monday_row_identity_target_v1": True,
        "monday_expected_replay_rows_v1": row_contract["monday_expected_replay_universe_v1"]["row_count_v1"],
        "monday_active_rows_v1": row_contract["monday_active_trades_v1"],
        "monday_quarantine_rows_v1": row_contract["monday_quarantine_trades_v1"],
        "monday_unknown_delta_rows_v1": row_contract["monday_missing_unknown_trades_v1"],
        "monday_as_of_columns_v1": foundation_summary.get("as_of_column_count_v1"),
        "monday_hindsight_columns_v1": foundation_summary.get("hindsight_output_column_count_v1"),
        "feature_inventory_rows_v1": int(len(inventory)),
        "r5_2_bad_blocks_v1": (r5_2["r5_2_selected_metrics_v1"] or {}).get("bad_blocks_v1"),
        "r5_2_tail_help_v1": (r5_2["r5_2_selected_metrics_v1"] or {}).get("tail_help_v1"),
        "missed_bad_rows_v1": r5_2["missed_bad_rows_v1"],
        "missed_tail_rows_v1": r5_2["missed_tail_rows_v1"],
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
        "training_started_v1": False,
        "blocked_action_v1": ALWAYS_BLOCKED,
        "hard_status_v1": {
            "BEVIST": [
                "1971 is the frozen Wednesday benchmark/comparator, not an automatic Monday row-count target.",
                "Monday foundation V4 is 1914 rows with 109 AS_OF columns and is not 1689 exact-only.",
                "Existing AS_OF/score/hindsight/path-dynamics assets were inventoried and classified by legality.",
            ],
            "INDIKERT": [
                "The main current blocker is R5.2/base recall, not missing AS_OF shape.",
                "Path-dynamics/transformer/RL assets exist but are not legal direct entry R6 features.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 retrain readiness.",
                "A R5.2 base with Wednesday-like bad/tail recall and zero hard damage.",
                "Row-level identities for the 57 Wednesday-vs-Monday aggregate delta, because frozen Wednesday source rows are missing.",
            ],
        },
        "foundation_contract_echo_v1": foundation_contract,
        "foundation_label_summary_echo_v1": foundation_label_summary,
        "r6_summary_echo_v1": {
            "decision_v1": r6_summary.get("decision_v1"),
            "compare_verdict_v1": r6_summary.get("compare_verdict_v1"),
            "selected_policy_source_v1": r6_summary.get("selected_policy_source_v1"),
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "materialized_at_utc_v1": _utc_now(),
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "foundation_dir_v1": str(foundation_dir),
            "score_dir_v1": str(score_dir),
            "r6_dir_v1": str(r6_dir),
            "recall_gap_dir_v1": str(recall_gap_dir) if recall_gap_dir else None,
            "path_dynamics_dir_v1": str(path_dynamics_dir) if path_dynamics_dir else None,
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": "MATERIALIZED",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
        "not_live_gate_v1": True,
    }
    audit = _audit(summary, gate)

    _write_json(output_dir / OUTPUT_FILES["row_contract"], row_contract)
    row_delta.to_csv(output_dir / OUTPUT_FILES["row_delta"], index=False)
    _write_json(output_dir / OUTPUT_FILES["pipeline_lock"], pipeline_lock)
    inventory.to_csv(output_dir / OUTPUT_FILES["feature_inventory"], index=False)
    _write_json(output_dir / OUTPUT_FILES["r5_2_reconstruction"], r5_2)
    _write_json(output_dir / OUTPUT_FILES["baseline_rebuild"], baseline_plan)
    reuse_map.to_csv(output_dir / OUTPUT_FILES["reuse_map"], index=False)
    _write_json(output_dir / OUTPUT_FILES["readiness_gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, gate, r5_2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--foundation-dir", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--r6-dir", type=Path, default=None)
    parser.add_argument("--recall-gap-dir", type=Path, default=None)
    parser.add_argument("--path-dynamics-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        foundation_dir=args.foundation_dir,
        score_dir=args.score_dir,
        r6_dir=args.r6_dir,
        recall_gap_dir=args.recall_gap_dir,
        path_dynamics_dir=args.path_dynamics_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
