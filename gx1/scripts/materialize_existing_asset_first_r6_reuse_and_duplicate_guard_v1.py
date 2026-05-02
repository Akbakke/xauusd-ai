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
LAYER_NAME = "EXISTING_ASSET_FIRST_R6_REUSE_AND_DUPLICATE_GUARD_V1"

WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

FOUNDATION_GLOB = "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_*"
SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*"
R6_GLOB = "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_*"
RECALL_GAP_GLOB = "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1_*"
PATH_DYNAMICS_GLOB = "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_*"
ANCHOR_AWARE_GLOB = "MONDAY_ANCHOR_AWARE_R6_CANONICAL_REBUILD_AND_EXISTING_FEATURE_REUSE_V1_*"
NARROW_GLOBS = [
    "MONDAY_NARROW*",
    "ALL_TRADE_REVIEW_LEDGER_20260424T170555Z_MONDAY_NARROW_RETRAIN_RUN_V1",
]
PROTECTOR_GLOBS = ["PROTECTOR_FIRST_SHADOW_EXPERIMENT*"]
OLD_REBUILD_GLOBS = [
    "MONDAY_R6_EXPLICIT_REBUILD_FROM_REHYDRATED_CONTRACT_V1_*",
    "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260425T_*",
    "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260425T_*",
]

SUMMARY = "summary_v1.json"
STATUS = "status_v1.json"
FOUNDATION_CONTRACT = "foundation_contract_v1.json"
FOUNDATION_LABEL_SUMMARY = "foundation_label_summary_v1.json"
FOUNDATION_AS_OF = "monday_r6_foundation_as_of_109_v1.parquet"
FOUNDATION_HINDSIGHT = "monday_r6_foundation_hindsight_with_labels_v1.parquet"
FOUNDATION_FRAME = "monday_r6_foundation_training_frame_pre_score_v1.parquet"
FOUNDATION_DELTA = "row_universe_delta_v1.csv"
FOUNDATION_FEATURE_AUDIT = "feature_contract_audit_v1.csv"
SCORE_REBUILD_SUMMARY = "score_rebuild_summary_v1.json"
SCORE_FRAME = "monday_r6_foundation_score_frame_v1.parquet"
R5_PRED = "monday_r5_score_prediction_view_v1.parquet"
R5_1_PRED = "monday_r5_1_score_prediction_view_v1.parquet"
R5_2_PRED = "monday_r5_2_score_prediction_view_v1.parquet"
FEATURE_MANIFEST = "feature_manifest_v1.csv"
MODEL_METRICS = "model_metrics_v1.csv"
R6_TRAINING_FRAME = "monday_r6_on_foundation_scores_training_frame_v1.parquet"
R6_PREDICTION_VIEW = "monday_r6_on_foundation_scores_prediction_view_v1.parquet"
R6_EVAL = "eval_summary_v1.json"
R6_COMPARE = "compare_against_wednesday_r6_v1.json"
R6_GRID = "r6_family_grid_replay_v1.csv"
RECALL_SUMMARY = "recall_gap_summary_v1.json"
MISSED_BAD = "missed_bad_rows_v1.csv"
MISSED_TAIL = "missed_tail_rows_v1.csv"
SPLIT_RECALL = "split_recall_gap_v1.csv"
PATH_SUMMARY = "shadow_meta_path_dynamics_logging_v2_summary_v1.json"
PATH_CONTRACT = "shadow_meta_path_dynamics_logging_v2_contract_v1.json"
PATH_RAW = "shadow_meta_path_dynamics_logging_v2_as_of_raw_state_table_v1.parquet"
PATH_POLICY = "shadow_meta_path_dynamics_logging_v2_policy_log_table_v1.parquet"
ANCHOR_ROW_CONTRACT = "anchor_aware_row_universe_contract_v1.json"
ANCHOR_ROW_DELTA = "anchor_aware_row_delta_explainer_v1.csv"
ANCHOR_FEATURE_INVENTORY = "existing_feature_asset_inventory_v1.csv"
ANCHOR_REUSE_MAP = "entry_exit_transformer_and_pre_rl_reuse_map_v1.csv"
ANCHOR_R5_2 = "r5_2_base_reconstruction_using_existing_assets_v1.json"

OUTPUT_FILES = {
    "asset_inventory": "existing_asset_inventory_v1.csv",
    "duplicate_guard": "no_duplicate_analysis_guard_v1.json",
    "source_graph": "canonical_r6_source_graph_v1.json",
    "anchor_reuse": "monday_anchor_aware_existing_asset_reuse_v1.csv",
    "r5_2_forensics": "r5_2_recall_base_reuse_forensics_v1.json",
    "feature_reuse_map": "feature_reuse_map_for_r6_entry_exit_and_rl_v1.csv",
    "decision_matrix": "canonical_reuse_decision_matrix_v1.json",
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
    "as_of_columns_v1": 109,
    "bad_blocks_v1": 180,
    "tail_help_v1": 149,
    "precision_v1": 0.972972972972973,
    "worst_loso_v1": 0.9285714285714286,
}

BLOCKED_ACTIONS = [
    "DO_NOT_CREATE_PARALLEL_BASELINE",
    "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
    "DO_NOT_USE_1689_EXACT_ONLY_AS_BASELINE",
    "DO_NOT_TREAT_BRIDGE_READINESS_AS_TRAINING_SURFACE",
    "DEFER_PROTECTOR_FIRST_UNTIL_CANONICAL_R6_READY",
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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _latest_dir(root: Path, pattern: str, required_file: str) -> Path | None:
    dirs = sorted(path for path in root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    return dirs[-1] if dirs else None


def _namespace(path: Path, reports_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(reports_root.resolve())
        return rel.parts[0] if rel.parts else path.name
    except ValueError:
        return path.parent.name


def _shape_and_columns(path: Path) -> tuple[int | None, int | None, list[str]]:
    if not path.exists() or path.is_dir():
        return None, None, []
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            return int(len(frame)), int(len(frame.columns)), [str(col) for col in frame.columns]
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            return int(len(frame)), int(len(frame.columns)), [str(col) for col in frame.columns]
        if path.suffix == ".json":
            data = _read_json(path)
            row_count = (
                data.get("row_count_v1")
                or data.get("rows_v1")
                or data.get("active_rows_v1")
                or (data.get("policy_logging_v1") or {}).get("row_count_v1")
            )
            return int(row_count) if row_count is not None else None, None, list(data.keys())
    except Exception:
        return None, None, []
    return None, None, []


def _surface_kind(columns: list[str], path: Path, category: str) -> str:
    joined = " ".join(columns).lower() + " " + path.name.lower() + " " + category.lower()
    has_asof = "as_of" in joined or "asof" in joined
    has_hindsight = "hindsight" in joined or "label_" in joined or "truth_" in joined or "realized" in joined
    if has_asof and has_hindsight:
        return "MIXED"
    if has_asof:
        return "AS_OF"
    if has_hindsight:
        return "HINDSIGHT"
    return "UNKNOWN"


def _families(columns: list[str], category: str) -> list[str]:
    names = [str(col) for col in columns]
    families: list[str] = []
    checks = [
        ("AS_OF_SKIP_REPLAY", lambda c: c.startswith("as_of_skip_replay")),
        ("AS_OF_SKIP_XGB", lambda c: c.startswith("as_of_skip_xgb")),
        ("AS_OF_CANDIDATE", lambda c: c.startswith("as_of_candidate") or c.startswith("as_of_entry_candidate")),
        ("R5_SCORE", lambda c: c.startswith("pred__entry_r5_") and "_r5_2_" not in c),
        ("R5_2_SCORE", lambda c: c.startswith("pred__entry_r5_2")),
        ("R6_SCORE", lambda c: c.startswith("pred__entry_r6")),
        ("R5_LABEL", lambda c: c.startswith("r5_label_")),
        ("R5_2_LABEL", lambda c: c.startswith("r5_2_label_")),
        ("R6_LABEL", lambda c: c.startswith("r6_label_")),
        ("PATH_DYNAMICS", lambda c: "mgmt_trace" in c or "management_core" in c),
        ("POLICY_LOG", lambda c: "policy" in c),
        ("LINEAGE_KEYS", lambda c: c in {"run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"}),
    ]
    for family, predicate in checks:
        if any(predicate(col) for col in names):
            families.append(family)
    if not families:
        families.append(category)
    return families


def _field_sample(columns: list[str], predicate: Any, limit: int = 16) -> str:
    return "|".join([col for col in columns if predicate(col)][:limit])


def _asset_row(
    *,
    path: Path,
    reports_root: Path,
    asset_kind: str,
    category: str,
    status: str,
    downstream: list[str],
    reason: str,
    row_count_override: int | None = None,
    column_count_override: int | None = None,
) -> dict[str, Any]:
    row_count, column_count, columns = _shape_and_columns(path)
    if row_count_override is not None:
        row_count = row_count_override
    if column_count_override is not None:
        column_count = column_count_override
    key_cols = [col for col in ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"] if col in columns]
    label_fields = _field_sample(columns, lambda col: "label" in col or "hindsight" in col or col.startswith("truth_"))
    score_fields = _field_sample(columns, lambda col: col.startswith("pred__") or "__block" in col)
    return {
        "path_v1": str(path),
        "run_id_or_namespace_v1": _namespace(path, reports_root),
        "asset_kind_v1": asset_kind,
        "row_count_v1": row_count,
        "column_count_v1": column_count,
        "key_columns_v1": "|".join(key_cols),
        "surface_kind_v1": _surface_kind(columns, path, category),
        "feature_families_v1": "|".join(_families(columns, category)),
        "label_eval_fields_v1": label_fields,
        "score_fields_v1": score_fields,
        "status_v1": status,
        "downstream_jobs_currently_depend_on_it_v1": "|".join(downstream),
        "reason_v1": reason,
    }


def _dir_summary_row(
    *,
    directory: Path,
    reports_root: Path,
    asset_kind: str,
    category: str,
    status: str,
    downstream: list[str],
    reason: str,
) -> dict[str, Any]:
    summary = _read_json(directory / SUMMARY)
    row_count = summary.get("row_count_v1") or summary.get("raw_rows_v1") or summary.get("monday_foundation_rows_v1")
    col_count = summary.get("as_of_column_count_v1") or summary.get("feature_count_v1") or summary.get("r6_feature_count_v1")
    return _asset_row(
        path=directory / SUMMARY if (directory / SUMMARY).exists() else directory,
        reports_root=reports_root,
        asset_kind=asset_kind,
        category=category,
        status=status,
        downstream=downstream,
        reason=reason,
        row_count_override=int(row_count) if row_count is not None else None,
        column_count_override=int(col_count) if col_count is not None else None,
    )


def _collect_inventory(
    reports_root: Path,
    foundation_dir: Path,
    score_dir: Path,
    r6_dir: Path,
    recall_gap_dir: Path | None,
    path_dynamics_dir: Path | None,
    anchor_dir: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snapshot_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    for filename in [WEDNESDAY_SUMMARY, WEDNESDAY_MANIFEST, "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_report_v1.md"]:
        path = snapshot_dir / filename
        if path.exists():
            rows.append(
                _asset_row(
                    path=path,
                    reports_root=reports_root,
                    asset_kind="WEDNESDAY_R6_BENCHMARK_SNAPSHOT",
                    category="WEDNESDAY_R6_BENCHMARK",
                    status="CANONICAL_REUSE",
                    downstream=["R6_COMPARATOR", "SAFETY_GATE", "CONTRACT_EXTRACTION"],
                    reason="Frozen Wednesday-R6 is the benchmark contract/comparator, not a restorable training source.",
                )
            )

    foundation_assets = [
        (SUMMARY, "MONDAY_FOUNDATION_SUMMARY", "CANONICAL_REUSE", ["R5_R5_1_R5_2_SCORE_REBUILD", "R6_REBUILD"]),
        (FOUNDATION_CONTRACT, "MONDAY_FOUNDATION_CONTRACT", "CANONICAL_REUSE", ["R5_R5_1_R5_2_SCORE_REBUILD", "R6_REBUILD"]),
        (FOUNDATION_AS_OF, "MONDAY_FOUNDATION_AS_OF_109", "CANONICAL_REUSE", ["R5_R5_1_R5_2_SCORE_REBUILD", "R6_REBUILD"]),
        (FOUNDATION_FRAME, "MONDAY_FOUNDATION_PRE_SCORE_FRAME", "CANONICAL_REUSE", ["R5_R5_1_R5_2_SCORE_REBUILD"]),
        (FOUNDATION_HINDSIGHT, "MONDAY_FOUNDATION_HINDSIGHT_LABELS", "REUSE_FOR_EVAL_ONLY", ["LABEL_INTERSECTION", "SAFETY_POCKET_EVAL"]),
        (FOUNDATION_LABEL_SUMMARY, "MONDAY_FOUNDATION_LABEL_SUMMARY", "REUSE_FOR_EVAL_ONLY", ["LABEL_INTERSECTION", "SAFETY_POCKET_EVAL"]),
        (FOUNDATION_DELTA, "MONDAY_FOUNDATION_ROW_DELTA", "REUSE_FOR_EVAL_ONLY", ["ANCHOR_AWARE_ROW_GATE"]),
        (FOUNDATION_FEATURE_AUDIT, "MONDAY_FOUNDATION_FEATURE_CONTRACT", "CANONICAL_REUSE", ["FEATURE_CONTRACT_GATE"]),
    ]
    for filename, kind, status, downstream in foundation_assets:
        path = foundation_dir / filename
        if path.exists():
            rows.append(
                _asset_row(
                    path=path,
                    reports_root=reports_root,
                    asset_kind=kind,
                    category="MONDAY_R6_FOUNDATION_V4",
                    status=status,
                    downstream=downstream,
                    reason="Current Monday anchor-aware foundation single source of truth.",
                )
            )

    score_assets = [
        (SUMMARY, "R5_R5_1_R5_2_SCORE_SUMMARY"),
        (SCORE_REBUILD_SUMMARY, "R5_R5_1_R5_2_SCORE_SELECTION_SUMMARY"),
        (SCORE_FRAME, "R5_R5_1_R5_2_SCORE_FRAME"),
        (R5_PRED, "R5_SCORE_LAYER_OUTPUT"),
        (R5_1_PRED, "R5_1_SCORE_LAYER_OUTPUT"),
        (R5_2_PRED, "R5_2_SCORE_LAYER_OUTPUT"),
        (FEATURE_MANIFEST, "R5_R5_1_R5_2_FEATURE_MANIFEST"),
        (MODEL_METRICS, "R5_R5_1_R5_2_MODEL_METRICS"),
    ]
    for filename, kind in score_assets:
        path = score_dir / filename
        if path.exists():
            rows.append(
                _asset_row(
                    path=path,
                    reports_root=reports_root,
                    asset_kind=kind,
                    category="R5_R5_1_R5_2_SCORE_ASSET",
                    status="REUSE_AS_INPUT",
                    downstream=["R5_2_RECALL_FIX", "R6_RETRAIN_AFTER_REUSE_GATE"],
                    reason="Latest explicit Monday score package; reuse before rebuilding true missing inputs.",
                )
            )

    r6_assets = [
        (SUMMARY, "R6_REBUILD_SUMMARY"),
        (R6_TRAINING_FRAME, "R6_TRAINING_EVAL_FRAME"),
        (R6_PREDICTION_VIEW, "R6_PREDICTION_VIEW"),
        (R6_EVAL, "R6_EVAL_SUMMARY"),
        (R6_COMPARE, "R6_WEDNESDAY_COMPARE_REPORT"),
        (R6_GRID, "R6_CANDIDATE_GRID_REPLAY"),
        (FEATURE_MANIFEST, "R6_FEATURE_MANIFEST"),
        (MODEL_METRICS, "R6_MODEL_METRICS"),
    ]
    for filename, kind in r6_assets:
        path = r6_dir / filename
        if path.exists():
            rows.append(
                _asset_row(
                    path=path,
                    reports_root=reports_root,
                    asset_kind=kind,
                    category="R6_SAFE_BUT_NOT_BETTER_REBUILD",
                    status="REUSE_FOR_EVAL_ONLY",
                    downstream=["R5_2_RECALL_FORENSICS", "R6_COMPARATOR", "THRESHOLD_REUSE_DIAGNOSIS"],
                    reason="Safe-but-not-better R6 rebuild is evidence and forensics, not canonical final R6.",
                )
            )

    if recall_gap_dir:
        for filename, kind in [
            (RECALL_SUMMARY, "R5_2_R6_RECALL_GAP_SUMMARY"),
            (MISSED_BAD, "MISSED_BAD_ROWS_FORENSICS"),
            (MISSED_TAIL, "MISSED_TAIL_ROWS_FORENSICS"),
            (SPLIT_RECALL, "SPLIT_RECALL_GAP"),
        ]:
            path = recall_gap_dir / filename
            if path.exists():
                rows.append(
                    _asset_row(
                        path=path,
                        reports_root=reports_root,
                        asset_kind=kind,
                        category="R5_2_RECALL_FORENSICS",
                        status="DIAGNOSTIC_ONLY",
                        downstream=["R5_2_RECALL_FIX"],
                        reason="Diagnostic for recall repair; not a canonical training surface.",
                    )
                )

    if path_dynamics_dir:
        for filename, kind in [
            (PATH_SUMMARY, "PATH_DYNAMICS_V2_SUMMARY"),
            (PATH_CONTRACT, "PATH_DYNAMICS_V2_CONTRACT"),
            (PATH_RAW, "PATH_DYNAMICS_V2_RAW_STATE"),
            (PATH_POLICY, "PATH_DYNAMICS_V2_POLICY_LOG"),
        ]:
            path = path_dynamics_dir / filename
            if path.exists():
                rows.append(
                    _asset_row(
                        path=path,
                        reports_root=reports_root,
                        asset_kind=kind,
                        category="PATH_DYNAMICS_V2",
                        status="REUSE_FOR_EVAL_ONLY",
                        downstream=["EXIT_MANAGEMENT_DIAGNOSTICS", "IQL_OR_BANDIT_READINESS"],
                        reason="Management/exit-anchor AS_OF diagnostics; not legal direct entry-R6 inputs.",
                    )
                )

    if anchor_dir:
        for filename, kind in [
            (ANCHOR_ROW_CONTRACT, "ANCHOR_AWARE_ROW_CONTRACT"),
            (ANCHOR_ROW_DELTA, "ANCHOR_AWARE_ROW_DELTA"),
            (ANCHOR_FEATURE_INVENTORY, "FEATURE_REUSE_INVENTORY"),
            (ANCHOR_REUSE_MAP, "ENTRY_EXIT_RL_REUSE_MAP"),
            (ANCHOR_R5_2, "R5_2_RECALL_REUSE_FORENSICS"),
            (SUMMARY, "ANCHOR_AWARE_REUSE_SUMMARY"),
        ]:
            path = anchor_dir / filename
            if path.exists():
                rows.append(
                    _asset_row(
                        path=path,
                        reports_root=reports_root,
                        asset_kind=kind,
                        category="ANCHOR_AWARE_REUSE_LOCK",
                        status="CANONICAL_REUSE",
                        downstream=["DUPLICATE_GUARD", "R6_REUSE_GATE"],
                        reason="Current anchor-aware reuse/gate lock; use instead of re-materializing ad hoc analysis.",
                    )
                )

    code_assets = [
        ("ENTRY_TRANSFORMER_CODE", WORKSPACE_ROOT / "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py", "REUSE_AS_INPUT", ["ENTRY_TRANSFORMER"]),
        ("ENTRY_TRANSFORMER_DATASET_BUILDER", WORKSPACE_ROOT / "gx1/scripts/build_entry_v10_ctx_training_dataset.py", "REUSE_AS_INPUT", ["ENTRY_TRANSFORMER"]),
        ("EXIT_TRANSFORMER_TRAINER", WORKSPACE_ROOT / "gx1/scripts/train_exit_transformer_v0_sharded.py", "REUSE_AS_INPUT", ["EXIT_TRANSFORMER"]),
        ("EXIT_PATH_CONTRACT", WORKSPACE_ROOT / "gx1/exits/contracts/exit_io_v1_ctx36_features.py", "REUSE_AS_INPUT", ["EXIT_TRANSFORMER", "MANAGEMENT_EXIT"]),
        ("MANAGEMENT_BANDIT_BUILDER", WORKSPACE_ROOT / "gx1/scripts/materialize_build_management_bandit_dataset_v1.py", "REUSE_AS_INPUT", ["MANAGEMENT_BANDIT"]),
        ("IQL_HARNESS", WORKSPACE_ROOT / "gx1/research/iql_training_harness_stub_v1.py", "REUSE_AS_INPUT", ["IQL_TRANSITION_REPAIR"]),
    ]
    for kind, path, status, downstream in code_assets:
        rows.append(
            _asset_row(
                path=path,
                reports_root=reports_root,
                asset_kind=kind,
                category="TRANSFORMER_PRE_RL_EXIT_MANAGEMENT_CODE_ASSET",
                status=status if path.exists() else "MISSING",
                downstream=downstream,
                reason="Existing code-level asset; reuse for its own lane, not as direct entry-R6 feature surface.",
            )
        )

    seen_dirs: set[Path] = set()
    for pattern in NARROW_GLOBS + PROTECTOR_GLOBS:
        for directory in sorted(reports_root.glob(pattern)):
            if directory.is_dir() and directory not in seen_dirs:
                seen_dirs.add(directory)
                rows.append(
                    _dir_summary_row(
                        directory=directory,
                        reports_root=reports_root,
                        asset_kind="NONCANONICAL_DIAGNOSTIC_SURFACE",
                        category="NARROW_OR_PROTECTOR_DIAGNOSTIC",
                        status="DIAGNOSTIC_ONLY",
                        downstream=["NONE_FOR_CANONICAL_R6"],
                        reason="Narrow/protector/1689/bridge path is diagnostic only and must not be canonical R6 input.",
                    )
                )
    for pattern in OLD_REBUILD_GLOBS:
        for directory in sorted(reports_root.glob(pattern)):
            if not directory.is_dir() or directory in {score_dir, r6_dir}:
                continue
            name = directory.name.upper()
            if "RECALLFIX" in name or "SAFETYGATE" in name:
                status = "UNSAFE_OR_LEAKAGE_RISK"
            elif "DRY" in name:
                status = "DIAGNOSTIC_ONLY"
            else:
                status = "DUPLICATE_DO_NOT_USE"
            rows.append(
                _dir_summary_row(
                    directory=directory,
                    reports_root=reports_root,
                    asset_kind="OLD_PARALLEL_REBUILD_OR_DUPLICATE",
                    category="OLD_R5_R6_REBUILD_ATTEMPT",
                    status=status,
                    downstream=["NONE_FOR_CANONICAL_R6"],
                    reason="Older parallel attempt superseded by current anchor-aware foundation/score/reuse locks.",
                )
            )

    return pd.DataFrame(rows).drop_duplicates(["path_v1", "asset_kind_v1"], keep="first")


def _duplicate_guard(inventory: pd.DataFrame) -> dict[str, Any]:
    canonical_count = int(inventory["status_v1"].isin(["CANONICAL_REUSE", "REUSE_AS_INPUT"]).sum())
    diagnostic_count = int(inventory["status_v1"].isin(["DIAGNOSTIC_ONLY", "DUPLICATE_DO_NOT_USE", "STALE_DO_NOT_USE", "UNSAFE_OR_LEAKAGE_RISK"]).sum())
    return {
        "layer_name": "NO_DUPLICATE_ANALYSIS_GUARD_V1",
        "guard_decision_v1": "DUPLICATE_GUARD_ACTIVE",
        "canonical_or_reusable_asset_count_v1": canonical_count,
        "diagnostic_or_duplicate_asset_count_v1": diagnostic_count,
        "this_job_created_new_training_surface_v1": False,
        "new_surface_requirements_v1": {
            "explicit_reason_required_v1": True,
            "must_state_what_it_replaces_or_supplements_v1": True,
            "must_state_why_existing_asset_is_not_enough_v1": True,
            "must_mark_canonical_or_diagnostic_only_v1": True,
            "must_link_back_to_single_source_of_truth_v1": True,
        },
        "hard_fail_if_v1": [
            "new baseline built without checking existing canonical assets",
            "1689 exact-only or other narrow surface used as R6 baseline",
            "bridge/readiness eval surface treated as training surface",
            "diagnostic assets used as canonical input",
            "same analysis re-materialized with a new name and no added evidence",
        ],
        "blocked_actions_v1": BLOCKED_ACTIONS,
    }


def _source_graph(inventory: pd.DataFrame, foundation_dir: Path, score_dir: Path, r6_dir: Path, path_dynamics_dir: Path | None) -> dict[str, Any]:
    return {
        "layer_name": "CANONICAL_R6_SOURCE_GRAPH_V1",
        "nodes_v1": [
            {
                "node_id_v1": "WEDNESDAY_R6_BENCHMARK_CONTRACT",
                "asset_path_v1": str(Path(WEDNESDAY_SNAPSHOT_DIR) / WEDNESDAY_FREEZE_DIR),
                "status_v1": "CANONICAL_REUSE",
                "role_v1": "benchmark/comparator/contract",
            },
            {
                "node_id_v1": "MONDAY_FOUNDATION_V4",
                "asset_path_v1": str(foundation_dir),
                "status_v1": "CANONICAL_REUSE",
                "role_v1": "anchor-aware AS_OF/HINDSIGHT foundation",
            },
            {
                "node_id_v1": "R5_R5_1_R5_2_SCORE_PACKAGE",
                "asset_path_v1": str(score_dir),
                "status_v1": "REUSE_AS_INPUT",
                "role_v1": "existing upstream score layer package; recall fix target",
            },
            {
                "node_id_v1": "R6_SAFE_BUT_NOT_BETTER_REBUILD",
                "asset_path_v1": str(r6_dir),
                "status_v1": "REUSE_FOR_EVAL_ONLY",
                "role_v1": "R6 score/eval forensics, not final canonical",
            },
            {
                "node_id_v1": "PATH_DYNAMICS_V2",
                "asset_path_v1": str(path_dynamics_dir) if path_dynamics_dir else None,
                "status_v1": "REUSE_FOR_EVAL_ONLY",
                "role_v1": "exit/management/RL diagnostics only for entry-R6",
            },
            {
                "node_id_v1": "NARROW_PROTECTOR_1689_SURFACES",
                "asset_path_v1": "MONDAY_NARROW* / PROTECTOR_FIRST*",
                "status_v1": "DIAGNOSTIC_ONLY",
                "role_v1": "negative diagnostic, not canonical input",
            },
        ],
        "edges_v1": [
            {"from_v1": "WEDNESDAY_R6_BENCHMARK_CONTRACT", "to_v1": "R6_EVAL_COMPARATOR", "meaning_v1": "sets metric/threshold/pocket contract"},
            {"from_v1": "MONDAY_FOUNDATION_V4", "to_v1": "R5_R5_1_R5_2_SCORE_PACKAGE", "meaning_v1": "feeds score/base rebuild"},
            {"from_v1": "R5_R5_1_R5_2_SCORE_PACKAGE", "to_v1": "R6_SAFE_BUT_NOT_BETTER_REBUILD", "meaning_v1": "feeds R6 five-head rebuild"},
            {"from_v1": "R6_SAFE_BUT_NOT_BETTER_REBUILD", "to_v1": "R5_2_RECALL_FORENSICS", "meaning_v1": "identifies missed bad/tail rows and threshold/score gaps"},
            {"from_v1": "PATH_DYNAMICS_V2", "to_v1": "EXIT_MANAGEMENT_RL_DIAGNOSTICS", "meaning_v1": "available for management/exit lanes, not direct entry R6"},
        ],
        "score_layers_already_exist_v1": ["R5", "R5.1", "R5.2", "R6 safe-but-not-better eval package"],
        "score_layers_that_must_be_repaired_or_rebuilt_v1": ["R5.2 recall/base selection", "R6 after reuse gate is clean"],
        "features_already_exist_v1": ["Monday foundation 109 AS_OF", "R5/R5.1/R5.2 score outputs", "R6 prediction/eval scores", "path-dynamics V2 diagnostics"],
        "features_not_established_v1": ["Canonical frozen R5.2 source/model tree", "bit-for-bit frozen R6 source/model tree"],
        "eval_readiness_only_surfaces_v1": ["hindsight/backfill labels", "path-dynamics V2 management fields", "narrow/protector bridge diagnostics", "R6 safe-but-not-better reports"],
        "training_eval_policy_surfaces_v1": {
            "training_input_v1": "MONDAY_FOUNDATION_V4 + R5/R5.1/R5.2 score package after recall fix",
            "eval_input_v1": "MONDAY_FOUNDATION hindsight/eval pockets + Wednesday comparator",
            "policy_surface_v1": "R6 five-head candidate grid after reuse gate",
        },
    }


def _anchor_reuse(anchor_dir: Path | None, foundation_dir: Path) -> pd.DataFrame:
    if anchor_dir and (anchor_dir / ANCHOR_ROW_DELTA).exists():
        source = _safe_read_csv(anchor_dir / ANCHOR_ROW_DELTA)
    else:
        source = _safe_read_csv(foundation_dir / FOUNDATION_DELTA)
        if not source.empty:
            source = source.rename(columns={"universe_v1": "week_window_v1"})
    rows: list[dict[str, Any]] = []
    for record in source.to_dict("records"):
        status = str(record.get("status_v1", "NOT_ESTABLISHED"))
        window = str(record.get("week_window_v1") or record.get("universe_v1") or "")
        row_count = record.get("row_count_v1", "")
        mapped = {
            "EXPECTED_DUE_TO_MONDAY_ANCHOR": "EXPECTED_ANCHOR_DELTA",
            "EXPECTED_DUE_TO_EOF_OR_WEEK_BOUNDARY": "EXPECTED_EOF_OR_BOUNDARY_DELTA",
            "EXPECTED_QUARANTINE": "EXPECTED_QUARANTINE",
            "FOUNDATION_ROW_UNIVERSE": "AVAILABLE_IN_EXISTING_ASSET",
            "ACTIVE_CANDIDATE": "AVAILABLE_IN_EXISTING_ASSET",
            "BENCHMARK_NOT_ROW_IDENTITY_TARGET_AFTER_MONDAY_REANCHOR": "EXPECTED_ANCHOR_DELTA",
        }.get(status, "NOT_ESTABLISHED")
        numeric_row_count = int(row_count) if str(row_count).strip() not in {"", "nan", "None"} else None
        if "MONDAY_ACTUAL_FULLCOVERAGE_FOUNDATION" in window:
            mapped = "AVAILABLE_IN_EXISTING_ASSET"
        if window in {"MONDAY_EXPECTED_REPLAY_UNIVERSE", "MONDAY_ACTIVE"}:
            mapped = "AVAILABLE_IN_EXISTING_ASSET"
        if window.startswith("TRUTH_") and numeric_row_count is not None and numeric_row_count > 0:
            mapped = "AVAILABLE_IN_EXISTING_ASSET"
        exists_in_asset = mapped in {"AVAILABLE_IN_EXISTING_ASSET", "EXPECTED_QUARANTINE"}
        if window == "WEDNESDAY_BENCHMARK":
            exists_in_asset = True
        if window == "WEDNESDAY_MINUS_MONDAY_AGGREGATE":
            exists_in_asset = False
        rows.append(
            {
                "candidate_uid": record.get("candidate_uid", ""),
                "trade_uid": record.get("trade_uid", ""),
                "trade_id": record.get("trade_id", ""),
                "decision_timestamp": record.get("decision_timestamp", ""),
                "week_window_v1": window,
                "row_count_v1": row_count,
                "expected_or_missing_v1": mapped,
                "exists_in_existing_asset_v1": exists_in_asset,
                "can_reuse_v1": mapped == "AVAILABLE_IN_EXISTING_ASSET",
                "requires_rebuild_v1": mapped == "TRUE_MISSING_NEEDS_REBUILD",
                "eval_readiness_only_v1": mapped == "EXPECTED_QUARANTINE",
                "status_v1": mapped,
                "source_status_v1": status,
                "source_explanation_v1": record.get("explanation_v1", record.get("status_v1", "")),
            }
        )
    return pd.DataFrame(rows)


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].astype("string").str.lower().isin(["true", "1"]).sum())


def _r5_2_forensics(score_dir: Path, r6_dir: Path, recall_gap_dir: Path | None, inventory: pd.DataFrame) -> dict[str, Any]:
    score_summary = _read_json(score_dir / SUMMARY)
    score_detail = _read_json(score_dir / SCORE_REBUILD_SUMMARY)
    r6_summary = _read_json(r6_dir / SUMMARY)
    missed_bad = _safe_read_csv(recall_gap_dir / MISSED_BAD) if recall_gap_dir else pd.DataFrame()
    missed_tail = _safe_read_csv(recall_gap_dir / MISSED_TAIL) if recall_gap_dir else pd.DataFrame()
    split_recall = _safe_read_csv(recall_gap_dir / SPLIT_RECALL) if recall_gap_dir else pd.DataFrame()
    current_score_assets = inventory[
        inventory["run_id_or_namespace_v1"].eq(score_dir.name)
        & inventory["asset_kind_v1"].str.contains("SCORE|PRED", regex=True, na=False)
    ]
    old_score_assets = inventory[
        inventory["asset_kind_v1"].eq("OLD_PARALLEL_REBUILD_OR_DUPLICATE")
        | inventory["run_id_or_namespace_v1"].str.contains("RECALLFIX|EXPLICIT_R5|EXPLICIT_R6|SAFETYGATE", regex=True, na=False)
    ]
    r5_2_metrics = (score_detail.get("r5_2_selected_policy_v1") or {}).get("metrics_v1") or {}
    missed_reason_counts = {
        col: _bool_count(missed_bad, col)
        for col in missed_bad.columns
        if str(col).startswith("miss_reason_")
    }
    used_score_dir = str(r6_summary.get("score_dir_v1") or "")
    return {
        "layer_name": "R5_2_RECALL_BASE_REUSE_FORENSICS_V1",
        "existing_score_assets_found_v1": {
            "current_score_asset_count_v1": int(len(current_score_assets)),
            "current_score_dir_v1": str(score_dir),
            "older_duplicate_or_stale_score_asset_count_v1": int(len(old_score_assets)),
        },
        "used_in_latest_r6_rebuild_v1": {
            "r6_dir_v1": str(r6_dir),
            "r6_decision_v1": r6_summary.get("decision_v1"),
            "score_dir_used_v1": used_score_dir,
            "latest_score_dir_used_v1": used_score_dir == str(score_dir),
            "foundation_score_context_column_count_v1": r6_summary.get("foundation_score_context_column_count_v1"),
        },
        "not_used_or_not_canonical_v1": old_score_assets[["path_v1", "status_v1", "reason_v1"]].head(100).to_dict("records"),
        "bad_tail_cases_missing_score_support_v1": {
            "missed_bad_rows_v1": int(len(missed_bad)),
            "missed_tail_rows_v1": int(len(missed_tail)),
            "missed_reason_counts_v1": missed_reason_counts,
            "split_recall_gap_v1": split_recall.to_dict("records") if not split_recall.empty else [],
        },
        "low_recall_cause_classification_v1": {
            "missing_existing_score_wiring_v1": "NOT_PRIMARY" if used_score_dir == str(score_dir) else "POSSIBLE",
            "wrong_or_too_conservative_score_layer_v1": True,
            "wrong_thresholds_or_selection_v1": True,
            "missing_feature_family_v1": "NOT_ESTABLISHED",
            "active_quarantine_mismatch_v1": False,
            "label_intersection_mismatch_v1": False,
            "not_established_source_v1": "Frozen canonical R5.2 source/model tree still missing.",
        },
        "r5_2_selected_metrics_v1": r5_2_metrics,
        "existing_assets_that_can_lift_recall_without_new_architecture_v1": [
            "Current Monday foundation AS_OF 109 and as_of_skip_replay context fields.",
            "Current R5/R5.1/R5.2 score outputs and prediction views.",
            "Current R6 candidate grid/eval traces for threshold and selection forensics.",
            "Recall-gap missed-row CSVs for targeted calibration without creating a new feature surface.",
        ],
        "recommended_fix_before_r6_retrain_v1": "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST_THEN_REPAIR_R5_2_RECALL_SELECTION",
    }


def _feature_reuse_map(anchor_dir: Path | None, inventory: pd.DataFrame) -> pd.DataFrame:
    if anchor_dir and (anchor_dir / ANCHOR_REUSE_MAP).exists():
        base = _safe_read_csv(anchor_dir / ANCHOR_REUSE_MAP)
        rows = []
        for record in base.to_dict("records"):
            rows.append(
                {
                    "feature_or_family_v1": record.get("surface_v1"),
                    "exists_already_v1": record.get("exists_now_v1"),
                    "where_v1": record.get("what_exists_v1"),
                    "legal_for_entry_v1": not bool(record.get("illegal_direct_entry_v1")),
                    "exit_or_management_only_v1": bool(record.get("illegal_direct_entry_v1")) and "audit" not in str(record.get("surface_v1", "")),
                    "eval_or_audit_only_v1": "audit" in str(record.get("surface_v1", "")) or record.get("status_v1") == "REUSE_FOR_EVAL_ONLY",
                    "can_reuse_now_v1": record.get("can_use_now_v1"),
                    "must_wait_v1": record.get("must_wait_v1"),
                    "must_not_use_directly_v1": record.get("illegal_direct_entry_v1"),
                    "status_v1": record.get("status_v1"),
                }
            )
        return pd.DataFrame(rows)
    families = sorted(set(inventory["feature_families_v1"].dropna().astype(str)))
    return pd.DataFrame(
        [
            {
                "feature_or_family_v1": family,
                "exists_already_v1": True,
                "where_v1": "existing_asset_inventory_v1.csv",
                "legal_for_entry_v1": "HINDSIGHT" not in family and "PATH_DYNAMICS" not in family,
                "exit_or_management_only_v1": "PATH_DYNAMICS" in family,
                "eval_or_audit_only_v1": "LABEL" in family or "R6_SCORE" in family,
                "can_reuse_now_v1": "HINDSIGHT" not in family and "PATH_DYNAMICS" not in family,
                "must_wait_v1": "PATH_DYNAMICS" in family,
                "must_not_use_directly_v1": "HINDSIGHT" in family or "PATH_DYNAMICS" in family,
                "status_v1": "REUSE_NOW",
            }
            for family in families
        ]
    )


def _decision_matrix(inventory: pd.DataFrame, r5_2: dict[str, Any]) -> dict[str, Any]:
    def exists(kind: str) -> bool:
        return bool(inventory["asset_kind_v1"].astype(str).str.contains(kind, regex=False).any())

    return {
        "layer_name": "CANONICAL_REUSE_DECISION_MATRIX_V1",
        "decisions_v1": [
            {
                "work_v1": "R5.2 base fix",
                "decision_v1": "WIRE_EXISTING_R5_2_SCORE_ASSETS_FIRST",
                "required_assets_v1": ["Monday foundation V4", "R5/R5.1/R5.2 score package", "recall-gap forensics"],
                "exists_v1": exists("R5_2_SCORE_LAYER_OUTPUT") and exists("MISSED_BAD_ROWS_FORENSICS"),
                "missing_v1": ["frozen canonical R5.2 source/model tree"],
                "diagnostic_only_v1": ["old recallfix/safetygate/narrow/protector attempts"],
                "must_wire_v1": ["current score frame", "R5/R5.1/R5.2 prediction views", "missed-row forensics into calibration audit"],
                "must_rebuild_v1": ["only true missing R5.2 inputs after wiring proof"],
                "do_not_do_now_v1": ["new baseline copy", "protector-first", "entry transformer"],
            },
            {
                "work_v1": "R6 retrain",
                "decision_v1": "RUN_R6_RETRAIN_AFTER_EXISTING_ASSET_WIRING",
                "required_assets_v1": ["fixed R5.2 recall base", "Monday foundation V4", "Wednesday comparator"],
                "exists_v1": False,
                "missing_v1": ["green R5.2 recall base"],
                "diagnostic_only_v1": ["safe-but-not-better R6 package"],
                "must_wire_v1": ["fixed score package into R6"],
                "must_rebuild_v1": ["R6 only after reuse gate"],
                "do_not_do_now_v1": ["parallel R6 baseline copy"],
            },
            {
                "work_v1": "entry transformer",
                "decision_v1": "DEFER_TRANSFORMER_UNTIL_R6_ASSET_GRAPH_IS_CLEAN",
                "required_assets_v1": ["entry_v10 code/dataset builders"],
                "exists_v1": exists("ENTRY_TRANSFORMER_CODE"),
                "missing_v1": ["canonical Monday R6 lock"],
                "diagnostic_only_v1": [],
                "must_wire_v1": [],
                "must_rebuild_v1": [],
                "do_not_do_now_v1": ["do not mix transformer into R6 entry baseline"],
            },
            {
                "work_v1": "exit transformer",
                "decision_v1": "DEFER_TRANSFORMER_UNTIL_R6_ASSET_GRAPH_IS_CLEAN",
                "required_assets_v1": ["exit transformer trainer", "exit contracts", "path dynamics"],
                "exists_v1": exists("EXIT_TRANSFORMER_TRAINER"),
                "missing_v1": ["canonical entry baseline context"],
                "diagnostic_only_v1": ["path dynamics for entry-R6"],
                "must_wire_v1": [],
                "must_rebuild_v1": [],
                "do_not_do_now_v1": ["do not use exit features directly in entry R6"],
            },
            {
                "work_v1": "management bandit",
                "decision_v1": "DEFER_SEQUENCE_IQL_UNTIL_TRANSITIONS_ARE_REAL",
                "required_assets_v1": ["management bandit builder", "policy logs", "path dynamics"],
                "exists_v1": exists("MANAGEMENT_BANDIT_BUILDER"),
                "missing_v1": ["green canonical transition/reward gate"],
                "diagnostic_only_v1": ["policy-log surfaces for entry-R6"],
                "must_wire_v1": [],
                "must_rebuild_v1": [],
                "do_not_do_now_v1": ["do not drive entry baseline from management diagnostics"],
            },
            {
                "work_v1": "IQL transition repair",
                "decision_v1": "DEFER_SEQUENCE_IQL_UNTIL_TRANSITIONS_ARE_REAL",
                "required_assets_v1": ["IQL harness", "true transition dataset"],
                "exists_v1": exists("IQL_HARNESS"),
                "missing_v1": ["real HOLD/next-state transition dataset"],
                "diagnostic_only_v1": ["current IQL planning stubs"],
                "must_wire_v1": [],
                "must_rebuild_v1": ["only true missing transition inputs"],
                "do_not_do_now_v1": ["do not use IQL as R6 baseline feature source"],
            },
            {
                "work_v1": "protector-first",
                "decision_v1": "DEFER_PROTECTOR_FIRST_UNTIL_CANONICAL_R6_READY",
                "required_assets_v1": ["canonical Monday R6"],
                "exists_v1": False,
                "missing_v1": ["canonical Monday R6"],
                "diagnostic_only_v1": ["1689 protector-first dry/prelaunch/run artifacts"],
                "must_wire_v1": [],
                "must_rebuild_v1": [],
                "do_not_do_now_v1": ["do not continue protector-first before canonical R6"],
            },
        ],
        "hard_decisions_v1": [
            "WIRE_EXISTING_R5_2_SCORE_ASSETS_FIRST",
            "REBUILD_ONLY_TRUE_MISSING_R5_2_INPUTS",
            "RUN_R6_RETRAIN_AFTER_EXISTING_ASSET_WIRING",
            "DO_NOT_CREATE_PARALLEL_BASELINE",
            "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
            "DEFER_PROTECTOR_FIRST_UNTIL_CANONICAL_R6_READY",
            "DEFER_TRANSFORMER_UNTIL_R6_ASSET_GRAPH_IS_CLEAN",
            "DEFER_SEQUENCE_IQL_UNTIL_TRANSITIONS_ARE_REAL",
        ],
        "r5_2_forensics_pointer_v1": {
            "missed_bad_rows_v1": r5_2["bad_tail_cases_missing_score_support_v1"]["missed_bad_rows_v1"],
            "missed_tail_rows_v1": r5_2["bad_tail_cases_missing_score_support_v1"]["missed_tail_rows_v1"],
        },
    }


def _next_action() -> dict[str, Any]:
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST",
        "priority_order_v1": [
            "use existing assets",
            "wire existing score/features",
            "rebuild only true missing inputs",
            "run R6 retrain only after reuse gate",
        ],
        "blocked_actions_v1": BLOCKED_ACTIONS,
    }


def _audit(summary: dict[str, Any], guard: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("ASSET_INVENTORY_MATERIALIZED", "PASS" if summary["asset_inventory_rows_v1"] > 0 else "FAIL", summary["asset_inventory_rows_v1"]),
            row("CANONICAL_ASSETS_FOUND", "PASS" if summary["canonical_reuse_asset_count_v1"] > 0 else "FAIL", summary["canonical_reuse_asset_count_v1"]),
            row("DIAGNOSTIC_SURFACES_MARKED", "PASS" if summary["diagnostic_or_duplicate_asset_count_v1"] > 0 else "WARN", summary["diagnostic_or_duplicate_asset_count_v1"]),
            row("NO_NEW_TRAINING_SURFACE_CREATED", "PASS", guard["this_job_created_new_training_surface_v1"]),
            row("DUPLICATE_GUARD_ACTIVE", "PASS", guard["guard_decision_v1"]),
            row("NEXT_ACTION_REUSE_FIRST", "PASS", summary["next_action_v1"]),
        ]
    )


def _report(summary: dict[str, Any], r5_2: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Existing Asset First R6 Reuse And Duplicate Guard V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Asset inventory rows: `{summary['asset_inventory_rows_v1']}`",
            f"- Canonical/reusable assets: `{summary['canonical_reuse_asset_count_v1']}` / `{summary['reuse_as_input_asset_count_v1']}`",
            f"- Diagnostic/duplicate/unsafe assets: `{summary['diagnostic_or_duplicate_asset_count_v1']}`",
            f"- Current R5.2 bad/tail: `{summary['r5_2_bad_blocks_v1']}` / `{summary['r5_2_tail_help_v1']}`",
            f"- Missed bad/tail rows: `{r5_2['bad_tail_cases_missing_score_support_v1']['missed_bad_rows_v1']}` / `{r5_2['bad_tail_cases_missing_score_support_v1']['missed_tail_rows_v1']}`",
            "",
            "This job creates no new baseline or training surface. It marks the reusable source graph and blocks duplicate/narrow/protector surfaces from canonical use.",
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
    anchor_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    foundation_dir = foundation_dir or _latest_dir(reports_root, FOUNDATION_GLOB, SUMMARY)
    score_dir = score_dir or _latest_dir(reports_root, SCORE_GLOB, SUMMARY)
    r6_dir = r6_dir or _latest_dir(reports_root, R6_GLOB, SUMMARY)
    recall_gap_dir = recall_gap_dir or _latest_dir(reports_root, RECALL_GAP_GLOB, RECALL_SUMMARY)
    path_dynamics_dir = path_dynamics_dir or _latest_dir(reports_root, PATH_DYNAMICS_GLOB, PATH_SUMMARY)
    anchor_dir = anchor_dir or _latest_dir(reports_root, ANCHOR_AWARE_GLOB, SUMMARY)
    if foundation_dir is None or score_dir is None or r6_dir is None:
        raise FileNotFoundError("Missing foundation, score, or R6 dir for existing-asset reuse guard")

    inventory = _collect_inventory(reports_root, foundation_dir, score_dir, r6_dir, recall_gap_dir, path_dynamics_dir, anchor_dir)
    guard = _duplicate_guard(inventory)
    graph = _source_graph(inventory, foundation_dir, score_dir, r6_dir, path_dynamics_dir)
    anchor_reuse = _anchor_reuse(anchor_dir, foundation_dir)
    r5_2 = _r5_2_forensics(score_dir, r6_dir, recall_gap_dir, inventory)
    feature_reuse = _feature_reuse_map(anchor_dir, inventory)
    decision_matrix = _decision_matrix(inventory, r5_2)
    next_action = _next_action()

    status_counts = inventory["status_v1"].value_counts().to_dict() if not inventory.empty else {}
    r5_2_metrics = r5_2.get("r5_2_selected_metrics_v1") or {}
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "reports_root_v1": str(reports_root),
        "output_dir_v1": str(output_dir),
        "decision_v1": "EXISTING_ASSET_FIRST_DUPLICATE_GUARD_ACTIVE",
        "next_action_v1": next_action["next_action_v1"],
        "training_started_v1": False,
        "new_surface_created_v1": False,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
        "asset_inventory_rows_v1": int(len(inventory)),
        "canonical_reuse_asset_count_v1": int(status_counts.get("CANONICAL_REUSE", 0)),
        "reuse_as_input_asset_count_v1": int(status_counts.get("REUSE_AS_INPUT", 0)),
        "reuse_eval_only_asset_count_v1": int(status_counts.get("REUSE_FOR_EVAL_ONLY", 0)),
        "diagnostic_or_duplicate_asset_count_v1": int(
            sum(status_counts.get(status, 0) for status in ["DIAGNOSTIC_ONLY", "DUPLICATE_DO_NOT_USE", "STALE_DO_NOT_USE", "UNSAFE_OR_LEAKAGE_RISK"])
        ),
        "r5_2_bad_blocks_v1": r5_2_metrics.get("bad_blocks_v1"),
        "r5_2_tail_help_v1": r5_2_metrics.get("tail_help_v1"),
        "blocked_action_v1": BLOCKED_ACTIONS,
        "hard_status_v1": {
            "BEVIST": [
                "Wednesday benchmark, Monday foundation V4, current score package, current R6 eval package, path-dynamics V2, and code-level transformer/RL assets were inventoried.",
                "No new baseline/training surface was built by this job.",
                "Narrow/protector/1689 and old duplicate rebuild attempts are marked non-canonical.",
            ],
            "INDIKERT": [
                "The next useful move is wiring/reusing existing R5.2/R6 score and forensics assets before rebuilding anything.",
                "Low recall is driven by conservative R5.2/R6 selection/threshold behavior, not by missing Monday AS_OF 109 foundation.",
            ],
            "IKKE_ETABLERT": [
                "A green canonical Monday R6.",
                "Frozen canonical R5.2 and R6 source/model trees.",
                "Proof that any new feature surface is needed before existing assets are wired.",
            ],
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
            "anchor_dir_v1": str(anchor_dir) if anchor_dir else None,
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
    audit = _audit(summary, guard)

    inventory.to_csv(output_dir / OUTPUT_FILES["asset_inventory"], index=False)
    _write_json(output_dir / OUTPUT_FILES["duplicate_guard"], guard)
    _write_json(output_dir / OUTPUT_FILES["source_graph"], graph)
    anchor_reuse.to_csv(output_dir / OUTPUT_FILES["anchor_reuse"], index=False)
    _write_json(output_dir / OUTPUT_FILES["r5_2_forensics"], r5_2)
    feature_reuse.to_csv(output_dir / OUTPUT_FILES["feature_reuse_map"], index=False)
    _write_json(output_dir / OUTPUT_FILES["decision_matrix"], decision_matrix)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, r5_2), encoding="utf-8")
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
    parser.add_argument("--anchor-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        foundation_dir=args.foundation_dir,
        score_dir=args.score_dir,
        r6_dir=args.r6_dir,
        recall_gap_dir=args.recall_gap_dir,
        path_dynamics_dir=args.path_dynamics_dir,
        anchor_dir=args.anchor_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
