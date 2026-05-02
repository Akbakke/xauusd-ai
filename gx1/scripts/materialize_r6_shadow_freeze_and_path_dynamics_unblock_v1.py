#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    _bool,
    _json_dumps,
    _load_json,
    _num,
    _safe_rate,
    _write_json,
)
from gx1.scripts.materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import (
    FREEZE_MANIFEST as R5_2_FREEZE_MANIFEST,
    EXTENSION_NAME as R5_2_FREEZE_EXTENSION_NAME,
)
from gx1.scripts.train_r6_entry_runner_first_retrain_v1 import (
    AS_OF_FEATURE_TABLE as R6_AS_OF_FEATURE_TABLE,
    BAD_RISK_LABEL_AUDIT as R6_BAD_RISK_LABEL_AUDIT,
    CONSISTENCY_AUDIT as R6_CONSISTENCY_AUDIT,
    CONTRACT as R6_CONTRACT,
    EXTENSION_NAME as R6_EXTENSION_NAME,
    FEATURE_PATH_DYNAMICS_AUDIT as R6_FEATURE_PATH_DYNAMICS_AUDIT,
    HEAD_TO_HEAD as R6_HEAD_TO_HEAD,
    HINDSIGHT_LABEL_OUTCOME_TABLE as R6_HINDSIGHT_LABEL_OUTCOME_TABLE,
    LOSO_METRICS as R6_LOSO_METRICS,
    MODEL_FAMILY_BAKEOFF as R6_MODEL_FAMILY_BAKEOFF,
    POLICY_PREDICTION_VIEW as R6_POLICY_PREDICTION_VIEW,
    R5_2_BASELINE,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    ROLLING_WINDOW_METRICS as R6_ROLLING_WINDOW_METRICS,
    RUNNER_LABEL_AUDIT as R6_RUNNER_LABEL_AUDIT,
    SUMMARY as R6_SUMMARY,
    TAIL_CONTROL_AUDIT as R6_TAIL_CONTROL_AUDIT,
    THRESHOLD_CALIBRATION as R6_THRESHOLD_CALIBRATION,
    WALKFORWARD_METRICS as R6_WALKFORWARD_METRICS,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"

FREEZE_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"
CONTRACT_LOCK_TABLE = "shadow_meta_all_trade_review_r6_contract_lock_v1.csv"
POLICY_LOGGING_LOCK = "shadow_meta_all_trade_review_r6_policy_logging_lock_v1.parquet"
HINDSIGHT_BACKFILL_LOCK = "shadow_meta_all_trade_review_r6_hindsight_outcome_backfill_lock_v1.parquet"
BATCH05_MARGIN_MONITOR = "shadow_meta_all_trade_review_r6_batch05_margin_monitor_v1.csv"
PATH_DYNAMICS_BLOCKER_AUDIT = "shadow_meta_all_trade_review_path_dynamics_blocker_audit_v1.csv"
PATH_DYNAMICS_INSTRUMENTATION_SPEC = "shadow_meta_path_dynamics_instrumentation_spec_v2.json"
R7_BACKLOG_TABLE = "shadow_meta_all_trade_review_r7_backlog_from_r6_failures_v1.csv"
NEXT_STEP_DECISION_MATRIX = "shadow_meta_all_trade_review_r6_next_step_decision_matrix_v1.csv"
ARTIFACT_HASH_TABLE = "shadow_meta_all_trade_review_r6_shadow_freeze_artifact_hashes_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_report_v1.md"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_consistency_audit_v1.csv"
TOP_LEVEL_SUMMARY = "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"

MODEL_VERSION_ID = "R6_ENTRY_RUNNER_FIRST_GLOBAL_FIVE_HEAD_20260422_V1"
THRESHOLD_VERSION_ID = "R6_THRESHOLDS_20260422T_SELECTED_CANDIDATE_04761_V1"
SELECTED_POLICY_STACK = "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON"
RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")

LOCKED_THRESHOLDS = {
    "bad_threshold_v1": 0.95,
    "risky_threshold_v1": 0.85,
    "tail_threshold_v1": 0.90,
    "runner_threshold_v1": 0.60,
    "r5_2_runner_threshold_v1": 0.74,
    "blindspot_threshold_v1": 0.70,
    "guard_v1": "hard_asof_runner_guard",
    "use_r5_2_base_v1": True,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_r6_dir(reports_root: Path, r6_dir_arg: str | None) -> Path:
    path = Path(r6_dir_arg).expanduser().resolve() if r6_dir_arg else reports_root / R6_EXTENSION_NAME
    if not path.exists():
        raise FileNotFoundError(f"R6 dir does not exist: {path}")
    for artifact in [
        R6_SUMMARY,
        R6_AS_OF_FEATURE_TABLE,
        R6_HINDSIGHT_LABEL_OUTCOME_TABLE,
        R6_POLICY_PREDICTION_VIEW,
        R6_HEAD_TO_HEAD,
        R6_LOSO_METRICS,
        R6_MODEL_FAMILY_BAKEOFF,
    ]:
        if not (path / artifact).exists():
            raise FileNotFoundError(f"{path} missing required R6 artifact {artifact}")
    return path


def _resolve_r5_2_freeze_dir(reports_root: Path, r5_2_freeze_arg: str | None) -> Path:
    path = Path(r5_2_freeze_arg).expanduser().resolve() if r5_2_freeze_arg else reports_root / R5_2_FREEZE_EXTENSION_NAME
    if not path.exists():
        raise FileNotFoundError(f"R5.2 freeze dir does not exist: {path}")
    if not (path / R5_2_FREEZE_MANIFEST).exists():
        raise FileNotFoundError(f"{path} missing R5.2 freeze manifest {R5_2_FREEZE_MANIFEST}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_json_ready(payload), ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _hash_rows(paths: Iterable[Path], *, root: Path, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        filename = path.name
        if filename == "model.joblib":
            hash_kind = "model_hash"
        elif filename == "feature_preprocessor.joblib":
            hash_kind = "preprocessor_hash"
        elif filename == "metadata.json":
            hash_kind = "metadata_hash"
        else:
            hash_kind = "artifact_hash"
        rows.append(
            {
                "artifact_role_v1": role,
                "hash_kind_v1": hash_kind,
                "relative_path_v1": rel,
                "absolute_path_v1": str(path),
                "byte_size_v1": int(path.stat().st_size),
                "sha256_v1": _sha256_file(path),
            }
        )
    return rows


def _schema_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [{"name_v1": str(column), "dtype_v1": str(dtype)} for column, dtype in frame.dtypes.items()]
    return {
        "column_count_v1": int(len(columns)),
        "columns_v1": columns,
        "schema_sha256_v1": _json_hash({"columns": columns}),
    }


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


def _batch_map(reports_root: Path, frame: pd.DataFrame, *, batch_weeks: int) -> dict[str, str]:
    out: dict[str, str] = {}
    run_ids = _all_run_ids(reports_root, frame)
    for batch_idx, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        for run_id in run_ids[start : start + batch_weeks]:
            out[str(run_id)] = f"BATCH_{batch_idx:02d}"
    return out


def _load_inputs(
    *,
    reports_root: Path,
    r6_dir: Path,
    r5_2_freeze_dir: Path,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = _load_json(r6_dir / R6_SUMMARY)
    contract = _load_json(r6_dir / R6_CONTRACT)
    asof = pd.read_parquet(r6_dir / R6_AS_OF_FEATURE_TABLE)
    hindsight = pd.read_parquet(r6_dir / R6_HINDSIGHT_LABEL_OUTCOME_TABLE)
    policy = pd.read_parquet(r6_dir / R6_POLICY_PREDICTION_VIEW)
    head = pd.read_csv(r6_dir / R6_HEAD_TO_HEAD)
    loso = pd.read_csv(r6_dir / R6_LOSO_METRICS)
    rolling = pd.read_csv(r6_dir / R6_ROLLING_WINDOW_METRICS)
    bakeoff = pd.read_csv(r6_dir / R6_MODEL_FAMILY_BAKEOFF)
    feature_audit = pd.read_csv(r6_dir / R6_FEATURE_PATH_DYNAMICS_AUDIT)
    runner_audit = pd.read_csv(r6_dir / R6_RUNNER_LABEL_AUDIT)
    bad_audit = pd.read_csv(r6_dir / R6_BAD_RISK_LABEL_AUDIT)
    tail_audit = pd.read_csv(r6_dir / R6_TAIL_CONTROL_AUDIT)
    threshold_calibration = pd.read_csv(r6_dir / R6_THRESHOLD_CALIBRATION)
    r5_2_manifest = _load_json(r5_2_freeze_dir / R5_2_FREEZE_MANIFEST)
    _require_columns(asof, ["candidate_uid", "run_id"], artifact_name=R6_AS_OF_FEATURE_TABLE)
    _require_columns(hindsight, ["candidate_uid"], artifact_name=R6_HINDSIGHT_LABEL_OUTCOME_TABLE)
    _require_columns(
        policy,
        [
            "candidate_uid",
            "run_id",
            "label_should_not_take_v1",
            "take_was_ok_v1",
            "label_strong_trade_candidate_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            "tail_10_50_mfe_v1",
            "is_repaired_165_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "baseline_realized_pnl_bps_v1",
            R6_BAD_PROB,
            R6_RISKY_PROB,
            R6_TAIL_PROB,
            R6_RUNNER_PROB,
            R5_2_RUNNER_PROB,
            "r5_2_frozen_reference__block_v1",
            "r6_selected_candidate__block_v1",
        ],
        artifact_name=R6_POLICY_PREDICTION_VIEW,
    )
    for name, frame in [(R6_AS_OF_FEATURE_TABLE, asof), (R6_HINDSIGHT_LABEL_OUTCOME_TABLE, hindsight), (R6_POLICY_PREDICTION_VIEW, policy)]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(policy) != expected_ledger_count:
        raise RuntimeError(f"Expected {expected_ledger_count} R6 policy rows, observed {len(policy)}")
    if summary.get("decision_v1", {}).get("r6_beats_r5_2_contract_v1") is not True and expected_ledger_count == 1971:
        raise RuntimeError("R6 freeze requires canonical R6 build that beats R5.2 contract")
    return (
        asof,
        hindsight,
        policy,
        summary,
        contract,
        head,
        loso,
        rolling,
        bakeoff,
        feature_audit,
        runner_audit,
        bad_audit,
        tail_audit,
        threshold_calibration,
        r5_2_manifest,
    )


def _asof_runner_guard(frame: pd.DataFrame) -> pd.Series:
    return (
        _num(frame, "as_of_candidate_tradable_prob_v1").ge(0.94)
        & _num(frame, "as_of_entry_candidate_path_quality_pred_v1").ge(0.70)
        & _num(frame, "as_of_candidate_mfe_first_n_pred_v1").ge(1.75)
        & _num(frame, "as_of_skip_candidate_p_flat_v1").le(0.50)
    )


def _join_policy(asof: pd.DataFrame, hindsight: pd.DataFrame, policy: pd.DataFrame, reports_root: Path, *, batch_weeks: int) -> pd.DataFrame:
    asof_cols = [
        column
        for column in asof.columns
        if column not in {"run_id", "trade_uid", "trade_id", "decision_timestamp"}
        and (column == "candidate_uid" or column not in policy.columns)
    ]
    hindsight_cols = [
        "candidate_uid",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        *[column for column in hindsight.columns if column.startswith("r6_label_")],
    ]
    frame = policy.merge(asof[[column for column in asof_cols if column in asof.columns]], on="candidate_uid", how="left", validate="one_to_one").merge(
        hindsight[[column for column in dict.fromkeys(hindsight_cols) if column in hindsight.columns]],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
        suffixes=("", "_hindsight"),
    )
    mapping = _batch_map(reports_root, frame, batch_weeks=batch_weeks)
    frame["batch_scope_v1"] = frame["run_id"].astype("string").map(mapping).fillna("BATCH_UNKNOWN")
    return frame


def _selected_thresholds(summary: dict[str, Any]) -> dict[str, Any]:
    selected = summary.get("selected_candidate_v1", {}) if isinstance(summary.get("selected_candidate_v1"), dict) else {}
    thresholds = selected.get("selected_thresholds_v1")
    if not isinstance(thresholds, dict):
        thresholds = dict(LOCKED_THRESHOLDS)
    return {**LOCKED_THRESHOLDS, **thresholds}


def _verify_policy_mask(frame: pd.DataFrame, thresholds: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    r5_2_base = _bool(frame, "r5_2_frozen_reference__block_v1")
    protect = (
        pd.to_numeric(frame[R6_RUNNER_PROB], errors="coerce").ge(float(thresholds["runner_threshold_v1"])).fillna(False)
        | pd.to_numeric(frame[R5_2_RUNNER_PROB], errors="coerce").ge(float(thresholds["r5_2_runner_threshold_v1"])).fillna(False)
        | _asof_runner_guard(frame)
    )
    addon = (
        pd.to_numeric(frame[R6_BAD_PROB], errors="coerce").ge(float(thresholds["bad_threshold_v1"])).fillna(False)
        & pd.to_numeric(frame[R6_RISKY_PROB], errors="coerce").ge(float(thresholds["risky_threshold_v1"])).fillna(False)
        & pd.to_numeric(frame[R6_TAIL_PROB], errors="coerce").ge(float(thresholds["tail_threshold_v1"])).fillna(False)
        & ~protect
    )
    computed = (r5_2_base | addon).fillna(False).astype(bool)
    materialized = _bool(frame, "r6_selected_candidate__block_v1")
    mismatch = computed.ne(materialized)
    return computed, addon, protect, mismatch


def _metric(head: pd.DataFrame, policy_name: str, column: str) -> Any:
    row = head[head["scope_v1"].astype("string").eq("ALL_1971") & head["policy_name_v1"].astype("string").eq(policy_name)]
    if row.empty:
        return None
    return row.iloc[0].get(column)


def _contract_lock(summary: dict[str, Any], head: pd.DataFrame, loso: pd.DataFrame) -> pd.DataFrame:
    selected = summary["selected_candidate_v1"]
    rows: list[dict[str, Any]] = []

    def add(name: str, r5: Any, r6: Any, requirement: str, passed: bool, margin: float | None, note: str) -> None:
        if margin is None:
            margin_status = "PASS" if passed else "FAIL"
        elif abs(float(margin)) <= 1e-12:
            margin_status = "EXACT_PASS_MONITOR" if passed else "FAIL"
        elif 0.0 < float(margin) < 0.01 and isinstance(r6, float):
            margin_status = "THIN_PASS_MONITOR"
        elif 0.0 < float(margin) < 2.0 and not isinstance(r6, float):
            margin_status = "THIN_PASS_MONITOR"
        else:
            margin_status = "CLEAR_PASS" if passed else "FAIL"
        rows.append(
            {
                "requirement_v1": name,
                "r5_2_benchmark_value_v1": r5,
                "r6_value_v1": r6,
                "required_comparison_v1": requirement,
                "pass_v1": bool(passed),
                "margin_v1": margin,
                "margin_status_v1": margin_status,
                "monitoring_required_v1": margin_status in {"EXACT_PASS_MONITOR", "THIN_PASS_MONITOR"} or name == "batch05_loso_pass",
                "notes_v1": note,
            }
        )

    add("bad_blocks", R5_2_BASELINE["bad_blocks_v1"], selected["should_not_take_block_count_v1"], "> R5.2", selected["should_not_take_block_count_v1"] > R5_2_BASELINE["bad_blocks_v1"], float(selected["should_not_take_block_count_v1"] - R5_2_BASELINE["bad_blocks_v1"]), "Clear recall improvement.")
    add("tail_help", R5_2_BASELINE["tail_help_v1"], selected["tail_10_50_help_count_v1"], "> R5.2", selected["tail_10_50_help_count_v1"] > R5_2_BASELINE["tail_help_v1"], float(selected["tail_10_50_help_count_v1"] - R5_2_BASELINE["tail_help_v1"]), "Clear tail-control improvement.")
    add("global_precision", R5_2_BASELINE["global_precision_v1"], selected["should_not_take_precision_v1"], ">= R5.2", selected["should_not_take_precision_v1"] >= R5_2_BASELINE["global_precision_v1"], float(selected["should_not_take_precision_v1"] - R5_2_BASELINE["global_precision_v1"]), "Very small positive margin; monitor.")
    add("worst_loso_precision", R5_2_BASELINE["worst_loso_precision_v1"], selected["worst_loso_precision_v1"], ">= R5.2", selected["worst_loso_precision_v1"] >= R5_2_BASELINE["worst_loso_precision_v1"], float(selected["worst_loso_precision_v1"] - R5_2_BASELINE["worst_loso_precision_v1"]), "Exact threshold pass; monitor BATCH_05.")
    add("repaired_165_damage", 0, selected["repaired_165_block_count_v1"], "= 0", selected["repaired_165_block_count_v1"] == 0, float(0 - selected["repaired_165_block_count_v1"]), "No repaired-pocket damage.")
    add("strong_false_blocks", 0, selected["strong_trade_false_block_count_v1"], "= 0", selected["strong_trade_false_block_count_v1"] == 0, float(0 - selected["strong_trade_false_block_count_v1"]), "No strong-trade false blocks.")
    add("hundred_plus_mfe_blocked", 0, selected["hundred_plus_mfe_block_count_v1"], "= 0", selected["hundred_plus_mfe_block_count_v1"] == 0, float(0 - selected["hundred_plus_mfe_block_count_v1"]), "No 100+ runner damage.")
    add("two_hundred_plus_mfe_blocked", 0, selected["two_hundred_plus_mfe_block_count_v1"], "= 0", selected["two_hundred_plus_mfe_block_count_v1"] == 0, float(0 - selected["two_hundred_plus_mfe_block_count_v1"]), "No 200+ runner damage.")
    add("fifty_plus_mfe_blocked", R5_2_BASELINE["fifty_plus_mfe_blocked_v1"], selected["fifty_plus_mfe_block_count_v1"], "<= R5.2", selected["fifty_plus_mfe_block_count_v1"] <= R5_2_BASELINE["fifty_plus_mfe_blocked_v1"], float(R5_2_BASELINE["fifty_plus_mfe_blocked_v1"] - selected["fifty_plus_mfe_block_count_v1"]), "At allowed ceiling; monitor.")
    add("strongest_winner_path_damage", 0, selected["strongest_winner_path_block_count_v1"], "= 0", selected["strongest_winner_path_block_count_v1"] == 0, float(0 - selected["strongest_winner_path_block_count_v1"]), "No strongest-winner damage.")
    add("batch04_loso_pass", True, selected["batch04_loso_pass_v1"], "is True", bool(selected["batch04_loso_pass_v1"]), None, f"BATCH_04 precision={selected.get('batch04_precision_v1')}.")
    add("batch05_loso_pass", True, selected["batch05_loso_pass_v1"], "is True", bool(selected["batch05_loso_pass_v1"]), None, f"BATCH_05 precision={selected.get('batch05_precision_v1')}; monitor thin margin.")
    add("no_live_promotion", True, True, "is True", True, None, "Freeze remains shadow/research only.")
    out = pd.DataFrame(rows)
    out["contract_pass_all_v1"] = bool(out["pass_v1"].all())
    out["source_policy_name_v1"] = str(selected.get("selected_policy_name_v1", SELECTED_POLICY_STACK))
    out["head_to_head_r6_blocks_v1"] = _metric(head, "R6_SELECTED_CANDIDATE", "block_count_v1")
    out["selected_loso_row_count_v1"] = int(len(loso))
    return out


def _snapshot_json(row: pd.Series, columns: Sequence[str]) -> str:
    payload = {column: _json_ready(row.get(column)) for column in columns}
    return _json_dumps(payload)


def _policy_logging_lock(frame: pd.DataFrame, thresholds: dict[str, Any], freeze_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    computed, addon, protect, mismatch = _verify_policy_mask(frame, thresholds)
    selected = _bool(frame, "r6_selected_candidate__block_v1")
    r5_2_base = _bool(frame, "r5_2_frozen_reference__block_v1")
    asof_snapshot_cols = [
        column
        for column in frame.columns
        if column.startswith("as_of_")
        or column
        in {
            R6_BAD_PROB,
            R6_RISKY_PROB,
            R6_TAIL_PROB,
            R6_RUNNER_PROB,
            R5_2_RUNNER_PROB,
            "entry_observation_present_v1",
            "entry_raw_state_present_v1",
            "entry_coverage_repair_applied_v1",
            "entry_coverage_repair_source_v1",
        }
    ]
    blocker = pd.to_numeric(frame[R6_BAD_PROB], errors="coerce")
    risky = pd.to_numeric(frame[R6_RISKY_PROB], errors="coerce")
    tail = pd.to_numeric(frame[R6_TAIL_PROB], errors="coerce")
    runner = pd.to_numeric(frame[R6_RUNNER_PROB], errors="coerce")
    r5_runner = pd.to_numeric(frame[R5_2_RUNNER_PROB], errors="coerce")
    block_reason = np.select(
        [r5_2_base, addon],
        ["R5_2_PRESERVED_BLOCK", "R6_ULTRA_SAFE_TAIL_RISKY_ADDON"],
        default="NOT_BLOCKED",
    )
    runner_reason = np.select(
        [
            runner.ge(float(thresholds["runner_threshold_v1"])).fillna(False),
            r5_runner.ge(float(thresholds["r5_2_runner_threshold_v1"])).fillna(False),
            _asof_runner_guard(frame),
        ],
        ["R6_RUNNER_SCORE_PROTECT", "R5_2_RUNNER_SCORE_PROTECT", "HARD_ASOF_RUNNER_GUARD"],
        default="NOT_RUNNER_PROTECTED",
    )
    allow_reason = np.select(
        [selected, protect, blocker.lt(float(thresholds["bad_threshold_v1"])).fillna(False), risky.lt(float(thresholds["risky_threshold_v1"])).fillna(False), tail.lt(float(thresholds["tail_threshold_v1"])).fillna(False)],
        ["NOT_ALLOWED_BLOCKED_SHADOW_ONLY", runner_reason, "BAD_SCORE_BELOW_THRESHOLD", "RISKY_SCORE_BELOW_THRESHOLD", "TAIL_SCORE_BELOW_THRESHOLD"],
        default="NO_BLOCK_SIGNAL",
    )
    safety_fail = (
        (selected & _bool(frame, "is_repaired_165_v1"))
        | (selected & _bool(frame, "hundred_plus_mfe_v1"))
        | (selected & _bool(frame, "two_hundred_plus_mfe_v1"))
        | (selected & _bool(frame, "label_strong_trade_candidate_v1"))
    )
    out = pd.DataFrame(
        {
            "candidate_uid": frame["candidate_uid"].astype("string"),
            "candidate_uid_exact_v1": frame["candidate_uid"].astype("string"),
            "run_id": frame["run_id"].astype("string"),
            "model_version_id_v1": MODEL_VERSION_ID,
            "threshold_version_id_v1": THRESHOLD_VERSION_ID,
            "freeze_id_v1": freeze_id,
            "selected_policy_stack_v1": SELECTED_POLICY_STACK,
            "blocker_score_v1": blocker,
            "risky_score_v1": risky,
            "tail_score_v1": tail,
            "runner_score_v1": runner,
            "r5_2_runner_score_v1": r5_runner,
            "batch04_blindspot_score_v1": pd.to_numeric(frame.get(R6_BLINDSPOT_PROB, pd.Series(np.nan, index=frame.index)), errors="coerce"),
            "r5_2_base_block_v1": r5_2_base.to_numpy(dtype=bool),
            "r6_addon_block_v1": addon.to_numpy(dtype=bool),
            "runner_guard_active_v1": protect.to_numpy(dtype=bool),
            "guard_status_v1": np.where(protect, runner_reason, "GUARD_NOT_TRIGGERED"),
            "block_decision_v1": selected.to_numpy(dtype=bool),
            "selected_action_v1": np.where(selected, "ENTRY_FALLBACK_BLOCK_SHADOW_ONLY", "KEEP_ENTRY_BASELINE_SHADOW_ONLY"),
            "block_reason_v1": block_reason,
            "allow_reason_v1": allow_reason,
            "runner_protection_reason_v1": runner_reason,
            "safety_constraint_status_v1": np.where(safety_fail, "ROW_SAFETY_REVIEW_REQUIRED", "PASS"),
            "policy_mask_matches_materialized_v1": ~mismatch.to_numpy(dtype=bool),
            "thresholds_json_v1": _json_dumps(thresholds),
            "as_of_feature_snapshot_json_v1": [
                _snapshot_json(row, asof_snapshot_cols)
                for _, row in frame.iterrows()
            ],
            "hindsight_backfill_join_key_v1": frame["candidate_uid"].astype("string"),
            "policy_logging_lock_contract_v1": "AS_OF_DECISION_PROVENANCE_ONLY_HINDSIGHT_BACKFILL_SEPARATE_NOT_LIVE_GATE",
        }
    )
    hindsight_cols = [
        "candidate_uid",
        "run_id",
        "label_should_not_take_v1",
        "take_was_ok_v1",
        "label_strong_trade_candidate_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "tail_10_50_mfe_v1",
        "is_repaired_165_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "baseline_realized_pnl_bps_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        *[column for column in frame.columns if column.startswith("r6_label_")],
    ]
    backfill = frame[[column for column in dict.fromkeys(hindsight_cols) if column in frame.columns]].copy()
    backfill["hindsight_backfill_contract_v1"] = "HINDSIGHT_OUTCOME_ONLY_NOT_AS_OF_NOT_POLICY_TRUTH"
    return out, backfill


def _batch05_margin_monitor(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    batch05 = frame[frame["batch_scope_v1"].astype("string").eq("BATCH_05")].copy()
    if batch05.empty:
        return pd.DataFrame()
    selected = _bool(batch05, "r6_selected_candidate__block_v1")
    bad = pd.to_numeric(batch05[R6_BAD_PROB], errors="coerce")
    risky = pd.to_numeric(batch05[R6_RISKY_PROB], errors="coerce")
    tail = pd.to_numeric(batch05[R6_TAIL_PROB], errors="coerce")
    runner = pd.to_numeric(batch05[R6_RUNNER_PROB], errors="coerce")
    r5_runner = pd.to_numeric(batch05[R5_2_RUNNER_PROB], errors="coerce")
    bad_margin = bad - float(thresholds["bad_threshold_v1"])
    risky_margin = risky - float(thresholds["risky_threshold_v1"])
    tail_margin = tail - float(thresholds["tail_threshold_v1"])
    runner_margin = pd.concat(
        [
            runner - float(thresholds["runner_threshold_v1"]),
            r5_runner - float(thresholds["r5_2_runner_threshold_v1"]),
        ],
        axis=1,
    ).max(axis=1)
    addon_margin = pd.concat([bad_margin, risky_margin, tail_margin, -runner_margin], axis=1).min(axis=1)
    near = (
        bad_margin.abs().le(0.05).fillna(False)
        | risky_margin.abs().le(0.05).fillna(False)
        | tail_margin.abs().le(0.05).fillna(False)
        | runner_margin.abs().le(0.05).fillna(False)
    )
    out = pd.DataFrame(
        {
            "candidate_uid": batch05["candidate_uid"].astype("string"),
            "run_id": batch05["run_id"].astype("string"),
            "block_decision_v1": selected.to_numpy(dtype=bool),
            "bad_block_v1": (selected & _bool(batch05, "label_should_not_take_v1")).to_numpy(dtype=bool),
            "false_block_risk_v1": (selected & _bool(batch05, "take_was_ok_v1")).to_numpy(dtype=bool),
            "runner_risk_v1": (
                _bool(batch05, "fifty_plus_mfe_v1")
                | _bool(batch05, "hundred_plus_mfe_v1")
                | _bool(batch05, "two_hundred_plus_mfe_v1")
                | runner.ge(float(thresholds["runner_threshold_v1"])).fillna(False)
                | r5_runner.ge(float(thresholds["r5_2_runner_threshold_v1"])).fillna(False)
            ).to_numpy(dtype=bool),
            "bad_score_v1": bad,
            "risky_score_v1": risky,
            "tail_score_v1": tail,
            "runner_score_v1": runner,
            "r5_2_runner_score_v1": r5_runner,
            "bad_margin_v1": bad_margin,
            "risky_margin_v1": risky_margin,
            "tail_margin_v1": tail_margin,
            "runner_protect_margin_v1": runner_margin,
            "addon_decision_margin_v1": addon_margin,
            "near_decision_boundary_v1": near.to_numpy(dtype=bool),
            "peak_mfe_bps_v1": _num(batch05, "peak_mfe_bps_v1"),
            "mae_abs_bps_v1": _num(batch05, "mae_abs_bps_v1"),
            "baseline_realized_pnl_bps_v1": _num(batch05, "baseline_realized_pnl_bps_v1"),
            "batch05_monitor_verdict_v1": np.where(near | (selected & _bool(batch05, "take_was_ok_v1")), "MONITOR_REQUIRED", "OK"),
        }
    )
    return out.sort_values(["block_decision_v1", "near_decision_boundary_v1", "addon_decision_margin_v1"], ascending=[False, False, True]).reset_index(drop=True)


def _path_dynamics_blocker_audit(frame: pd.DataFrame, feature_audit: pd.DataFrame) -> pd.DataFrame:
    expected = [
        {
            "field_name_v1": "last_peak_ts",
            "target_as_of_field_v1": "as_of_last_peak_ts_utc_v1",
            "dtype_v1": "timestamp_utc_string",
            "upstream_trace_field_v1": "last_peak_ts_utc / as_of_mgmt_trace_last_peak_ts_utc_v1",
            "derived_features_v1": "minutes_since_last_peak,bars_since_last_peak,peak_recency_bucket",
            "expected_r6_r7_help_v1": "runner near-miss protection; exit-too-early/too-late separation; BATCH_05 margin diagnosis",
            "source_exists_note_v1": "exit_manager.py emits last_peak_ts_utc to EXIT_EVAL_TRACE; shadow_meta maps management anchor fields, but R6 entry AS_OF does not carry it.",
        },
        {
            "field_name_v1": "last_mfe_ts",
            "target_as_of_field_v1": "as_of_last_mfe_ts_utc_v1",
            "dtype_v1": "timestamp_utc_string",
            "upstream_trace_field_v1": "last_mfe_ts_utc / as_of_mgmt_trace_last_mfe_ts_utc_v1",
            "derived_features_v1": "minutes_since_last_mfe,mfe_recency_bucket,stale_mfe_flag",
            "expected_r6_r7_help_v1": "detect giveback/tail leakage and hold-longer/exit-earlier conflicts",
            "source_exists_note_v1": "exit_manager.py emits last_mfe_ts_utc to EXIT_EVAL_TRACE; not flowed into R6 entry AS_OF feature table.",
        },
        {
            "field_name_v1": "last_peak_mfe",
            "target_as_of_field_v1": "as_of_last_peak_mfe_bps_v1",
            "dtype_v1": "float64_bps",
            "upstream_trace_field_v1": "mfe_bps_so_far / distance_from_peak_mfe_bps plus peak state",
            "derived_features_v1": "mfe_peak_slope,runner_strength_at_anchor,peak_mfe_bucket",
            "expected_r6_r7_help_v1": "separate true runners from low-value 10-50 tail cases",
            "source_exists_note_v1": "MFE so far exists in management trace; explicit last_peak_mfe_bps is not locked in R6 AS_OF.",
        },
        {
            "field_name_v1": "max_mfe_without_mae",
            "target_as_of_field_v1": "as_of_max_mfe_without_mae_bps_v1",
            "dtype_v1": "float64_bps",
            "upstream_trace_field_v1": "requires incremental trade state: max favorable excursion before first adverse excursion threshold",
            "derived_features_v1": "clean_runner_score,adverse_first_penalty,entry_quality_path_score",
            "expected_r6_r7_help_v1": "immediate MAE risk and should-not-take vs clean-entry separation",
            "source_exists_note_v1": "not found as an upstream locked field; requires trade state or trace instrumentation.",
        },
        {
            "field_name_v1": "mfe_mae_sequence_order",
            "target_as_of_field_v1": "as_of_mfe_mae_sequence_order_v1",
            "dtype_v1": "categorical_enum",
            "upstream_trace_field_v1": "requires first_mfe_ts_utc and first_mae_ts_utc or equivalent state",
            "derived_features_v1": "MFE_FIRST,MAE_FIRST,SAME_BAR,ONLY_MFE,ONLY_MAE,NEITHER",
            "expected_r6_r7_help_v1": "detect bad trades that go straight to MAE vs entries that breathe before running",
            "source_exists_note_v1": "not found as a locked upstream field; requires trace/policy-log extension.",
        },
    ]
    rows: list[dict[str, Any]] = []
    for item in expected:
        field = item["target_as_of_field_v1"]
        present = field in frame.columns
        feature_status = "AVAILABLE_IN_R6_AS_OF" if present else "LOGGING_BLOCKED_FOR_R6_ENTRY_AS_OF"
        audit_row = feature_audit[feature_audit.get("top_features_json_v1", pd.Series(dtype=str)).astype("string").str.contains(field, regex=False, na=False)]
        rows.append(
            {
                **item,
                "present_in_r6_as_of_v1": bool(present),
                "null_rate_in_r6_as_of_v1": _safe_rate(float(frame[field].isna().sum()), float(len(frame))) if present else None,
                "blocker_status_v1": feature_status,
                "why_blocked_v1": "Not present in R6 AS_OF feature table; cannot be used in R6/R7 without leakage-safe raw-state/policy-log flow." if not present else "Available; must still pass null/leakage audit.",
                "upstream_source_exists_v1": item["field_name_v1"] in {"last_peak_ts", "last_mfe_ts", "last_peak_mfe"},
                "as_of_legal_at_decision_point_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                "requires_exit_manager_change_v1": item["field_name_v1"] in {"max_mfe_without_mae", "mfe_mae_sequence_order", "last_peak_mfe"},
                "requires_trade_state_change_v1": item["field_name_v1"] in {"max_mfe_without_mae", "mfe_mae_sequence_order", "last_peak_mfe"},
                "requires_trace_change_v1": True,
                "requires_raw_state_change_v1": True,
                "requires_policy_log_change_v1": True,
                "feature_audit_seen_v1": bool(not audit_row.empty),
            }
        )
    return pd.DataFrame(rows)


def _instrumentation_spec(blocker_df: pd.DataFrame) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for row in blocker_df.to_dict(orient="records"):
        fields.append(
            {
                "field_name_v1": row["target_as_of_field_v1"],
                "source_trace_field_v1": row["upstream_trace_field_v1"],
                "dtype_v1": row["dtype_v1"],
                "as_of_semantics_v1": row["as_of_legal_at_decision_point_v1"],
                "set_in_flow_v1": "exit_manager.py::_emit_exit_eval_trace at every exit/management decision anchor",
                "write_to_v1": "EXIT_EVAL_TRACE.csv",
                "raw_state_mapping_v1": f"shadow_meta_v1 maps trace field to {row['target_as_of_field_v1']} and policy logging carries same value",
                "leakage_guard_v1": "timestamp/value must be <= decision_anchor_timestamp_utc; pre-entry models must not consume in-trade path fields for the same trade",
                "derived_features_enabled_v1": str(row["derived_features_v1"]).split(","),
            }
        )
    return {
        "layer_name": "PATH_DYNAMICS_INSTRUMENTATION_SPEC_V2",
        "mode_v1": "LOGGING_SPEC_ONLY_NOT_IMPLEMENTED_IN_THIS_FREEZE",
        "flow_v1": "exit_manager.py -> EXIT_EVAL_TRACE.csv -> shadow_meta raw-state -> policy-log -> audit",
        "fields_v1": fields,
        "coverage_null_rate_audit_after_replay_v1": {
            "required_rows_v1": "all non-zero management/exit anchors; entry pre-trade use must remain separate",
            "required_null_rate_v1": 0.0,
            "required_coverage_v1": "match non-zero trace coverage",
            "audit_artifact_v1": "PATH_DYNAMICS_COVERAGE_NULL_RATE_AUDIT_V1",
        },
        "leakage_checks_v1": [
            "field timestamp <= decision_anchor_timestamp_utc",
            "no same-trade in-trade path field in pre-entry AS_OF model input",
            "HINDSIGHT labels remain physically separate from AS_OF features",
            "raw-state values must come from EXIT_EVAL_TRACE row at or before decision anchor",
        ],
        "compile_and_test_requirements_v1": [
            "python -m py_compile gx1/execution/exit_manager.py gx1/analysis/shadow_meta_v1.py",
            "pytest tests/test_shadow_meta_replay_market_pressure_fields.py tests/test_replay_merge_shadow_meta_outcome_fallback.py",
            "full replay coverage/null-rate audit before any R7 retrain consumes these fields",
        ],
        "not_live_gate_v1": True,
    }


def _r7_backlog(frame: pd.DataFrame, batch05_monitor: pd.DataFrame, blocker_df: pd.DataFrame) -> pd.DataFrame:
    selected = _bool(frame, "r6_selected_candidate__block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    risky = _bool(frame, "r6_label_risky_allow_v1") if "r6_label_risky_allow_v1" in frame.columns else (should & _num(frame, "mae_abs_bps_v1").ge(40.0))
    runner_near = _bool(frame, "r6_label_runner_near_miss_v1") if "r6_label_runner_near_miss_v1" in frame.columns else (_bool(frame, "take_was_ok_v1") & _bool(frame, "fifty_plus_mfe_v1"))
    rows = [
        {
            "backlog_item_v1": "R7_TAIL_RISKY_ADDON_MARGIN_HARDENING",
            "affected_count_v1": int((tail & ~selected).sum()),
            "expected_utility_rank_v1": 1,
            "expected_utility_v1": "HIGH",
            "runner_damage_risk_v1": "MEDIUM_UNLESS_R6_PROTECTOR_PRESERVED",
            "requires_new_labels_v1": True,
            "requires_new_features_v1": False,
            "requires_path_dynamics_logging_first_v1": False,
            "reason_v1": "R6 lifted tail help to 149, but remaining tail cases still exist; tune only with R6 runner-first constraints.",
        },
        {
            "backlog_item_v1": "R7_BATCH05_MARGIN_AND_ROLLING_STABILITY",
            "affected_count_v1": int(batch05_monitor["near_decision_boundary_v1"].fillna(False).sum()) if not batch05_monitor.empty else 0,
            "expected_utility_rank_v1": 2,
            "expected_utility_v1": "HIGH",
            "runner_damage_risk_v1": "LOW_MEDIUM",
            "requires_new_labels_v1": False,
            "requires_new_features_v1": False,
            "requires_path_dynamics_logging_first_v1": False,
            "reason_v1": "BATCH_05 passes exactly at worst-LOSO precision threshold; monitor and harden calibration before harvest/RL.",
        },
        {
            "backlog_item_v1": "R7_MISSED_SHOULD_NOT_TAKE_BLINDSPOTS",
            "affected_count_v1": int((should & ~selected).sum()),
            "expected_utility_rank_v1": 3,
            "expected_utility_v1": "MEDIUM_HIGH",
            "runner_damage_risk_v1": "HIGH_IF_RECALL_ONLY",
            "requires_new_labels_v1": True,
            "requires_new_features_v1": True,
            "requires_path_dynamics_logging_first_v1": True,
            "reason_v1": "321 should-not-take rows remain unblocked; path sequence/adverse-first fields may separate safe recall from runner damage.",
        },
        {
            "backlog_item_v1": "R7_RISKY_ALLOW_RESIDUALS",
            "affected_count_v1": int((risky & ~selected).sum()),
            "expected_utility_rank_v1": 4,
            "expected_utility_v1": "MEDIUM",
            "runner_damage_risk_v1": "MEDIUM_HIGH",
            "requires_new_labels_v1": True,
            "requires_new_features_v1": True,
            "requires_path_dynamics_logging_first_v1": True,
            "reason_v1": "R6 catches 74 risky allows but residuals likely need path/order features, not just threshold expansion.",
        },
        {
            "backlog_item_v1": "R7_RUNNER_NEAR_MISS_PROTECTION",
            "affected_count_v1": int(runner_near.sum()),
            "expected_utility_rank_v1": 5,
            "expected_utility_v1": "MEDIUM",
            "runner_damage_risk_v1": "REDUCES_RISK",
            "requires_new_labels_v1": True,
            "requires_new_features_v1": True,
            "requires_path_dynamics_logging_first_v1": False,
            "reason_v1": "52 runner near-misses remain protected by R6; preserve them before any recall expansion.",
        },
        {
            "backlog_item_v1": "R7_PATH_DYNAMICS_UNBLOCK",
            "affected_count_v1": int(blocker_df["blocker_status_v1"].astype("string").eq("LOGGING_BLOCKED_FOR_R6_ENTRY_AS_OF").sum()),
            "expected_utility_rank_v1": 6,
            "expected_utility_v1": "FOUNDATIONAL",
            "runner_damage_risk_v1": "LOW_IF_SHADOW_ONLY",
            "requires_new_labels_v1": False,
            "requires_new_features_v1": True,
            "requires_path_dynamics_logging_first_v1": True,
            "reason_v1": "Five path-dynamics fields are still blocked; unlock them before R7 consumes path sequence features.",
        },
    ]
    return pd.DataFrame(rows).sort_values("expected_utility_rank_v1").reset_index(drop=True)


def _decision_matrix(contract_df: pd.DataFrame, batch05_monitor: pd.DataFrame, blocker_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    freeze_ok = bool(contract_df["pass_v1"].all())
    blocked_count = int(blocker_df["blocker_status_v1"].astype("string").eq("LOGGING_BLOCKED_FOR_R6_ENTRY_AS_OF").sum())
    batch05_monitor_required = bool(not batch05_monitor.empty and batch05_monitor["batch05_monitor_verdict_v1"].astype("string").eq("MONITOR_REQUIRED").any())
    next_after_freeze = "IMPROVE_PATH_DYNAMICS_LOGGING_FIRST" if blocked_count else "RUN_R6_SHADOW_HARVEST_REPLAY_NEXT"
    current_decision = "FREEZE_R6_SHADOW_CANDIDATE_DONE" if freeze_ok else "KEEP_R5_2_AS_REFERENCE_ONLY"
    rows = [
        {
            "decision_key_v1": "FREEZE_R6_SHADOW_CANDIDATE_DONE",
            "status_v1": "PRIMARY" if current_decision == "FREEZE_R6_SHADOW_CANDIDATE_DONE" else "FAIL",
            "reason_v1": "R6 passes locked R5.2 contract and remains NOT_LIVE_GATE." if freeze_ok else "R6 contract did not pass.",
        },
        {
            "decision_key_v1": "RUN_R6_SHADOW_HARVEST_REPLAY_NEXT",
            "status_v1": "DEFER_UNTIL_PATH_DYNAMICS" if blocked_count else "NEXT",
            "reason_v1": "Do after path-dynamics unblock unless explicitly choosing to replay without new fields.",
        },
        {
            "decision_key_v1": "IMPROVE_PATH_DYNAMICS_LOGGING_FIRST",
            "status_v1": "RECOMMENDED_NEXT" if next_after_freeze == "IMPROVE_PATH_DYNAMICS_LOGGING_FIRST" else "NOT_NEEDED",
            "reason_v1": f"{blocked_count} path-dynamics fields remain blocked; BATCH_05 monitor required={batch05_monitor_required}.",
        },
        {
            "decision_key_v1": "START_R7_RETRAIN",
            "status_v1": "DEFER",
            "reason_v1": "Do not retrain R7 until freeze is locked and path-dynamics/logging readiness is decided.",
        },
        {
            "decision_key_v1": "KEEP_R5_2_AS_REFERENCE_ONLY",
            "status_v1": "REFERENCE",
            "reason_v1": "R5.2 remains benchmark/reference, but R6 is the stronger frozen shadow candidate.",
        },
    ]
    return pd.DataFrame(rows), current_decision, next_after_freeze


def _freeze_manifest(
    *,
    reports_root: Path,
    r6_dir: Path,
    r5_2_freeze_manifest: dict[str, Any],
    summary: dict[str, Any],
    asof: pd.DataFrame,
    hindsight: pd.DataFrame,
    artifact_hash_df: pd.DataFrame,
    contract_df: pd.DataFrame,
    test_status: str,
) -> dict[str, Any]:
    selected = summary["selected_candidate_v1"]
    basis = {
        "selected_candidate": selected.get("selected_policy_name_v1", SELECTED_POLICY_STACK),
        "thresholds": LOCKED_THRESHOLDS,
        "metrics": {key: selected.get(key) for key in sorted(selected)},
        "model_hashes": artifact_hash_df[artifact_hash_df["artifact_role_v1"].eq("R6_MODEL_ARTIFACT")][["relative_path_v1", "sha256_v1"]].to_dict(orient="records"),
    }
    freeze_id = f"R6_SHADOW_FREEZE_{_json_hash(basis)[:16].upper()}_V1"
    return {
        "layer_name": "R6_FREEZE_MANIFEST_V1",
        "freeze_id_v1": freeze_id,
        "freeze_status_v1": "FROZEN_SHADOW_RESEARCH_CANDIDATE_NOT_LIVE_GATE",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "r6_source_dir_v1": str(r6_dir),
        "r5_2_benchmark_freeze_id_v1": r5_2_freeze_manifest.get("freeze_id_v1", "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"),
        "model_version_id_v1": MODEL_VERSION_ID,
        "threshold_version_id_v1": THRESHOLD_VERSION_ID,
        "selected_candidate_id_v1": selected.get("selected_policy_name_v1", SELECTED_POLICY_STACK),
        "selected_policy_stack_v1": SELECTED_POLICY_STACK,
        "thresholds_v1": LOCKED_THRESHOLDS,
        "score_head_names_v1": {
            "blocker_score_v1": R6_BAD_PROB,
            "risky_score_v1": R6_RISKY_PROB,
            "tail_score_v1": R6_TAIL_PROB,
            "runner_score_v1": R6_RUNNER_PROB,
            "r5_2_runner_score_v1": R5_2_RUNNER_PROB,
            "batch04_blindspot_score_v1": R6_BLINDSPOT_PROB,
        },
        "hashes_v1": {
            "model_hashes_v1": artifact_hash_df[artifact_hash_df["hash_kind_v1"].eq("model_hash")].to_dict(orient="records"),
            "preprocessor_hashes_v1": artifact_hash_df[artifact_hash_df["hash_kind_v1"].eq("preprocessor_hash")].to_dict(orient="records"),
            "metadata_hashes_v1": artifact_hash_df[artifact_hash_df["hash_kind_v1"].eq("metadata_hash")].to_dict(orient="records"),
        },
        "as_of_schema_v1": _schema_payload(asof),
        "hindsight_schema_v1": _schema_payload(hindsight),
        "training_artifact_lineage_v1": {
            "r6_summary": str(r6_dir / R6_SUMMARY),
            "r6_model_family_bakeoff": str(r6_dir / R6_MODEL_FAMILY_BAKEOFF),
            "r6_loso_metrics": str(r6_dir / R6_LOSO_METRICS),
            "artifact_hash_table": ARTIFACT_HASH_TABLE,
        },
        "contract_lock_summary_v1": {
            "all_pass_v1": bool(contract_df["pass_v1"].all()),
            "monitoring_required_count_v1": int(contract_df["monitoring_required_v1"].sum()),
            "thin_or_exact_passes_v1": contract_df.loc[contract_df["monitoring_required_v1"], "requirement_v1"].astype("string").tolist(),
        },
        "test_status_v1": test_status,
        "not_live_gate_status_v1": {
            "not_controller": True,
            "not_live_gate": True,
            "not_policy_truth": True,
            "promotion_status_v1": "NOT_PROMOTED_NOT_LIVE_GATE",
        },
    }


def _audit_record(name: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: dict[str, Any]) -> str:
    selected = summary["selected_candidate_v1"]
    lines = [
        "# R6 Shadow Freeze And Path Dynamics Unblock V1",
        "",
        "Shadow/research only. Not a live gate.",
        "",
        "## Freeze",
        "",
        f"- Freeze id: `{summary['freeze_id_v1']}`",
        f"- Model id: `{summary['model_version_id_v1']}`",
        f"- Candidate: `{summary['selected_candidate_id_v1']}`",
        f"- Decision: `{summary['current_decision_v1']}`",
        f"- Next after freeze: `{summary['recommended_next_after_freeze_v1']}`",
        "",
        "## R6 vs R5.2",
        "",
        f"- Bad blocks: `{selected['should_not_take_block_count_v1']}` vs R5.2 `106`",
        f"- Tail help: `{selected['tail_10_50_help_count_v1']}` vs R5.2 `82`",
        f"- Precision: `{selected['should_not_take_precision_v1']}`",
        f"- Worst LOSO precision: `{selected['worst_loso_precision_v1']}`",
        "",
        "## Path Dynamics",
        "",
        f"- Blocked fields: `{summary['path_dynamics_v1']['blocked_field_count_v1']}`",
        "- Instrumentation flow: `exit_manager.py -> EXIT_EVAL_TRACE.csv -> raw-state -> policy-log -> audit`",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    r6_dir: Path,
    r5_2_freeze_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None,
    test_status: str,
) -> dict[str, Any]:
    (
        asof,
        hindsight,
        policy,
        r6_summary,
        r6_contract,
        head,
        loso,
        rolling,
        bakeoff,
        feature_audit,
        runner_audit,
        bad_audit,
        tail_audit,
        threshold_calibration,
        r5_2_manifest,
    ) = _load_inputs(reports_root=reports_root, r6_dir=r6_dir, r5_2_freeze_dir=r5_2_freeze_dir, expected_ledger_count=expected_ledger_count)
    frame = _join_policy(asof, hindsight, policy, reports_root, batch_weeks=batch_weeks)
    thresholds = _selected_thresholds(r6_summary)
    model_files = list((r6_dir / "models").rglob("*")) if (r6_dir / "models").exists() else []
    input_files = [
        r6_dir / artifact
        for artifact in [
            R6_SUMMARY,
            R6_CONTRACT,
            R6_AS_OF_FEATURE_TABLE,
            R6_HINDSIGHT_LABEL_OUTCOME_TABLE,
            R6_POLICY_PREDICTION_VIEW,
            R6_HEAD_TO_HEAD,
            R6_LOSO_METRICS,
            R6_ROLLING_WINDOW_METRICS,
            R6_MODEL_FAMILY_BAKEOFF,
            R6_FEATURE_PATH_DYNAMICS_AUDIT,
            R6_RUNNER_LABEL_AUDIT,
            R6_BAD_RISK_LABEL_AUDIT,
            R6_TAIL_CONTROL_AUDIT,
            R6_THRESHOLD_CALIBRATION,
        ]
    ]
    artifact_hash_df = pd.DataFrame(_hash_rows(input_files, root=reports_root, role="R6_FREEZE_INPUT_ARTIFACT") + _hash_rows(model_files, root=reports_root, role="R6_MODEL_ARTIFACT"))
    contract_df = _contract_lock(r6_summary, head, loso)
    freeze_manifest = _freeze_manifest(
        reports_root=reports_root,
        r6_dir=r6_dir,
        r5_2_freeze_manifest=r5_2_manifest,
        summary=r6_summary,
        asof=asof,
        hindsight=hindsight,
        artifact_hash_df=artifact_hash_df,
        contract_df=contract_df,
        test_status=test_status,
    )
    freeze_id = str(freeze_manifest["freeze_id_v1"])
    policy_lock_df, hindsight_backfill_df = _policy_logging_lock(frame, thresholds, freeze_id)
    batch05_df = _batch05_margin_monitor(frame, thresholds)
    blocker_df = _path_dynamics_blocker_audit(frame, feature_audit)
    spec = _instrumentation_spec(blocker_df)
    backlog_df = _r7_backlog(frame, batch05_df, blocker_df)
    decision_df, current_decision, next_after_freeze = _decision_matrix(contract_df, batch05_df, blocker_df)
    computed, _addon, _protect, mismatch = _verify_policy_mask(frame, thresholds)
    r6_consistency = pd.read_csv(r6_dir / R6_CONSISTENCY_AUDIT)
    consistency_df = pd.DataFrame(
        [
            _audit_record("R6_INPUT_PRESENT", "PASS", {"r6_dir": str(r6_dir)}),
            _audit_record("R5_2_BENCHMARK_PRESENT", "PASS", {"r5_2_freeze_id": r5_2_manifest.get("freeze_id_v1")}),
            _audit_record("FULL_COVERAGE_LOCK", "PASS" if expected_ledger_count is None or len(policy) == expected_ledger_count else "FAIL", {"observed": len(policy), "expected": expected_ledger_count}),
            _audit_record(
                "R6_CONTRACT_BEATS_R5_2",
                "PASS" if expected_ledger_count != 1971 or bool(r6_summary.get("decision_v1", {}).get("r6_beats_r5_2_contract_v1")) else "FAIL",
                {"decision": r6_summary.get("decision_v1", {}), "canonical_gate_enforced_v1": expected_ledger_count == 1971},
            ),
            _audit_record(
                "CONTRACT_LOCK_ALL_PASS",
                "PASS" if expected_ledger_count != 1971 or bool(contract_df["pass_v1"].all()) else "FAIL",
                {"failed_requirements": contract_df.loc[~contract_df["pass_v1"], "requirement_v1"].astype("string").tolist(), "canonical_gate_enforced_v1": expected_ledger_count == 1971},
            ),
            _audit_record("POLICY_MASK_MATCHES", "PASS" if not bool(mismatch.any()) else "FAIL", {"mismatch_count": int(mismatch.sum())}),
            _audit_record("POLICY_LOCK_ROW_COUNT", "PASS" if len(policy_lock_df) == len(policy) else "FAIL", {"policy_lock_rows": len(policy_lock_df), "policy_rows": len(policy)}),
            _audit_record("AS_OF_HINDSIGHT_SEPARATED", "PASS" if not any(column.startswith("r6_label_") for column in asof.columns) else "FAIL", {"r6_label_columns_in_asof": [column for column in asof.columns if column.startswith("r6_label_")]}),
            _audit_record(
                "MODEL_HASHES_PRESENT",
                "PASS" if expected_ledger_count != 1971 or int(artifact_hash_df["hash_kind_v1"].eq("model_hash").sum()) >= 5 else "FAIL",
                {"model_hash_count": int(artifact_hash_df["hash_kind_v1"].eq("model_hash").sum()), "canonical_gate_enforced_v1": expected_ledger_count == 1971},
            ),
            _audit_record("PATH_DYNAMICS_BLOCKER_AUDIT_PRESENT", "PASS" if len(blocker_df) == 5 else "FAIL", {"rows": len(blocker_df)}),
            _audit_record("UPSTREAM_R6_CONSISTENCY", "PASS" if not r6_consistency["status_v1"].astype("string").eq("FAIL").any() else "FAIL", {"r6_failed_checks": int(r6_consistency["status_v1"].astype("string").eq("FAIL").sum())}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_controller": True, "not_live_gate": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    selected = r6_summary["selected_candidate_v1"]
    batch05_precision = selected.get("batch05_precision_v1")
    batch05_monitor_required = bool(not batch05_df.empty and batch05_df["batch05_monitor_verdict_v1"].astype("string").eq("MONITOR_REQUIRED").any())
    status = {
        "layer_name": "R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_STATUS_V1",
        "FREEZE_STATUS": "FROZEN_SHADOW_RESEARCH_CANDIDATE_NOT_LIVE_GATE" if failed_checks == 0 else "FREEZE_ISSUES_FOUND_NOT_PROMOTED",
        "PATH_DYNAMICS_STATUS": "UNBLOCK_PLAN_MATERIALIZED_LOGGING_STILL_REQUIRED",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "r6_source_dir_v1": str(r6_dir),
        "r5_2_freeze_dir_v1": str(r5_2_freeze_dir),
        "freeze_id_v1": freeze_id,
        "model_version_id_v1": MODEL_VERSION_ID,
        "threshold_version_id_v1": THRESHOLD_VERSION_ID,
        "selected_candidate_id_v1": selected.get("selected_policy_name_v1", SELECTED_POLICY_STACK),
        "selected_policy_stack_v1": SELECTED_POLICY_STACK,
        "thresholds_v1": LOCKED_THRESHOLDS,
        "current_decision_v1": current_decision,
        "recommended_next_after_freeze_v1": next_after_freeze,
        "selected_candidate_v1": selected,
        "contract_v1": {
            "all_requirements_pass_v1": bool(contract_df["pass_v1"].all()),
            "clearly_beaten_requirements_v1": contract_df.loc[contract_df["margin_status_v1"].eq("CLEAR_PASS"), "requirement_v1"].astype("string").tolist(),
            "thin_or_exact_requirements_v1": contract_df.loc[contract_df["monitoring_required_v1"], "requirement_v1"].astype("string").tolist(),
        },
        "batch05_v1": {
            "precision_v1": batch05_precision,
            "monitor_required_v1": batch05_monitor_required,
            "near_boundary_count_v1": int(batch05_df["near_decision_boundary_v1"].fillna(False).sum()) if not batch05_df.empty else 0,
            "false_block_risk_count_v1": int(batch05_df["false_block_risk_v1"].fillna(False).sum()) if not batch05_df.empty else 0,
        },
        "policy_logging_v1": {
            "row_count_v1": int(len(policy_lock_df)),
            "mask_mismatch_count_v1": int(mismatch.sum()),
            "hindsight_backfill_rows_v1": int(len(hindsight_backfill_df)),
        },
        "path_dynamics_v1": {
            "blocked_field_count_v1": int(blocker_df["blocker_status_v1"].astype("string").eq("LOGGING_BLOCKED_FOR_R6_ENTRY_AS_OF").sum()),
            "blocked_fields_v1": blocker_df.loc[blocker_df["blocker_status_v1"].astype("string").eq("LOGGING_BLOCKED_FOR_R6_ENTRY_AS_OF"), "field_name_v1"].astype("string").tolist(),
            "instrumentation_spec_v1": PATH_DYNAMICS_INSTRUMENTATION_SPEC,
        },
        "r7_backlog_top3_v1": backlog_df.head(3).to_dict(orient="records"),
        "artifact_counts_v1": {
            "artifact_hash_rows_v1": int(len(artifact_hash_df)),
            "policy_logging_rows_v1": int(len(policy_lock_df)),
            "batch05_monitor_rows_v1": int(len(batch05_df)),
            "r7_backlog_rows_v1": int(len(backlog_df)),
        },
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R6 freeze materialized with freeze_id={freeze_id}.",
                f"R6 selected candidate={selected.get('selected_policy_name_v1', SELECTED_POLICY_STACK)}.",
                f"R6 beats R5.2: bad_blocks={selected.get('should_not_take_block_count_v1')} > 106, tail_help={selected.get('tail_10_50_help_count_v1')} > 82.",
                f"Policy logging lock rows={len(policy_lock_df)} and mask mismatches={int(mismatch.sum())}.",
                "No live promotion was materialized.",
            ],
            "INDIKERT": [
                "BATCH_05 passes, but margin is thin enough to monitor before harvest/RL replay.",
                "Path-dynamics source exists partially in management trace for last_peak/last_mfe, but R6 entry AS_OF remains blocked.",
                "R7 should start with tail-risky margin hardening and BATCH_05 stability before broad recall expansion.",
            ],
            "IKKE_ETABLERT": [
                "Live gate readiness.",
                "R7 improvement with path-dynamics consumed as features.",
                "Counterfactual live fill quality for newly blocked rows.",
            ],
        },
    }
    manifest = {
        "layer_name": "R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_MANIFEST_V1",
        "artifacts_v1": {
            "freeze_manifest": FREEZE_MANIFEST,
            "contract_lock_table": CONTRACT_LOCK_TABLE,
            "policy_logging_lock": POLICY_LOGGING_LOCK,
            "hindsight_backfill_lock": HINDSIGHT_BACKFILL_LOCK,
            "batch05_margin_monitor": BATCH05_MARGIN_MONITOR,
            "path_dynamics_blocker_audit": PATH_DYNAMICS_BLOCKER_AUDIT,
            "path_dynamics_instrumentation_spec": PATH_DYNAMICS_INSTRUMENTATION_SPEC,
            "r7_backlog_table": R7_BACKLOG_TABLE,
            "next_step_decision_matrix": NEXT_STEP_DECISION_MATRIX,
            "artifact_hash_table": ARTIFACT_HASH_TABLE,
            "summary": SUMMARY,
            "status": STATUS,
            "report": REPORT,
            "consistency_audit": CONSISTENCY_AUDIT,
        },
    }
    return {
        "freeze_manifest": _json_ready(freeze_manifest),
        "contract_df": contract_df,
        "policy_lock_df": policy_lock_df,
        "hindsight_backfill_df": hindsight_backfill_df,
        "batch05_df": batch05_df,
        "blocker_df": blocker_df,
        "instrumentation_spec": _json_ready(spec),
        "backlog_df": backlog_df,
        "decision_df": decision_df,
        "artifact_hash_df": artifact_hash_df,
        "summary": _json_ready(summary),
        "status": _json_ready(status),
        "manifest": _json_ready(manifest),
        "consistency_df": consistency_df,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    r6_dir: Path | None = None,
    r5_2_freeze_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
    test_status: str = "NOT_EXECUTED_INSIDE_MATERIALIZER",
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_r6_dir = _resolve_r6_dir(reports_root, str(r6_dir) if r6_dir else None)
    resolved_r5_2_freeze_dir = _resolve_r5_2_freeze_dir(reports_root, str(r5_2_freeze_dir) if r5_2_freeze_dir else None)
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        r6_dir=resolved_r6_dir,
        r5_2_freeze_dir=resolved_r5_2_freeze_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
        test_status=test_status,
    )
    _write_json(extension_dir / FREEZE_MANIFEST, payload["freeze_manifest"])
    payload["contract_df"].to_csv(extension_dir / CONTRACT_LOCK_TABLE, index=False)
    payload["policy_lock_df"].to_parquet(extension_dir / POLICY_LOGGING_LOCK, index=False)
    payload["hindsight_backfill_df"].to_parquet(extension_dir / HINDSIGHT_BACKFILL_LOCK, index=False)
    payload["batch05_df"].to_csv(extension_dir / BATCH05_MARGIN_MONITOR, index=False)
    payload["blocker_df"].to_csv(extension_dir / PATH_DYNAMICS_BLOCKER_AUDIT, index=False)
    _write_json(extension_dir / PATH_DYNAMICS_INSTRUMENTATION_SPEC, payload["instrumentation_spec"])
    payload["backlog_df"].to_csv(extension_dir / R7_BACKLOG_TABLE, index=False)
    payload["decision_df"].to_csv(extension_dir / NEXT_STEP_DECISION_MATRIX, index=False)
    payload["artifact_hash_df"].to_csv(extension_dir / ARTIFACT_HASH_TABLE, index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
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
    parser = argparse.ArgumentParser(description="Freeze R6 shadow candidate and materialize path-dynamics unblock plan.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--r6-dir", default=None)
    parser.add_argument("--r5-2-freeze-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    parser.add_argument("--test-status", default="NOT_EXECUTED_INSIDE_MATERIALIZER")
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        r6_dir=Path(args.r6_dir).expanduser().resolve() if args.r6_dir else None,
        r5_2_freeze_dir=Path(args.r5_2_freeze_dir).expanduser().resolve() if args.r5_2_freeze_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
        test_status=args.test_status,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
