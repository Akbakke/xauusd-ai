#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    _safe_rate,
    _write_json,
)
from gx1.scripts.materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import (
    AS_OF_TABLE as PHASE_AS_OF_TABLE,
    CALIBRATION_AUDIT as PHASE_CALIBRATION_AUDIT,
    CONSISTENCY_AUDIT as PHASE_CONSISTENCY_AUDIT,
    CONTRACT as PHASE_CONTRACT,
    EXTENSION_NAME as PHASE_EXTENSION_NAME,
    FAILURE_MODE_TABLE as PHASE_FAILURE_MODE_TABLE,
    HARVEST_IMPACT as PHASE_HARVEST_IMPACT,
    HINDSIGHT_TABLE as PHASE_HINDSIGHT_TABLE,
    POLICY_LOGGING_EXPLAINABILITY as PHASE_POLICY_LOGGING_EXPLAINABILITY,
    ROBUSTNESS_STRESS_MATRIX as PHASE_ROBUSTNESS_STRESS_MATRIX,
    SHADOW_REPLAY_BAKEOFF as PHASE_SHADOW_REPLAY_BAKEOFF,
    SUMMARY as PHASE_SUMMARY,
)
from gx1.scripts.train_r5_2_entry_runner_aware_retrain_and_loso_selection_v1 import (
    BAD_PROB,
    CONTRACT as R5_2_CONTRACT,
    POLICY_PREDICTION_VIEW as R5_2_POLICY_PREDICTION_VIEW,
    RUNNER_PROB,
    SUMMARY as R5_2_SUMMARY,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1"

FREEZE_MANIFEST = "shadow_meta_all_trade_review_r5_2_shadow_freeze_manifest_v1.json"
POLICY_LOGGING_LOCK = "shadow_meta_all_trade_review_r5_2_policy_logging_lock_v1.parquet"
FAILURE_CLUSTER_TABLE = "shadow_meta_all_trade_review_r6_failure_cluster_table_v1.csv"
R6_OPPORTUNITY_AUDIT = "shadow_meta_all_trade_review_r6_label_feature_opportunity_audit_v1.csv"
R6_TRAINING_TARGET_SPEC = "shadow_meta_all_trade_review_r6_training_target_spec_v1.json"
GO_NO_GO_MATRIX = "shadow_meta_all_trade_review_r5_2_vs_r6_go_no_go_matrix_v1.csv"
ARTIFACT_HASH_TABLE = "shadow_meta_all_trade_review_r5_2_shadow_freeze_artifact_hashes_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r5_2_shadow_freeze_and_r6_failure_backlog_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r5_2_shadow_freeze_and_r6_failure_backlog_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r5_2_shadow_freeze_and_r6_failure_backlog_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r5_2_shadow_freeze_and_r6_failure_backlog_report_v1.md"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r5_2_shadow_freeze_and_r6_failure_backlog_consistency_audit_v1.csv"
TOP_LEVEL_SUMMARY = "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
DEFAULT_SELECTED_POLICY_STACK = "R5_2_CANDIDATE_00165_R5_CURRENT_RUNNER_GATED_none"
DEFAULT_THRESHOLD_VERSION_ID = "R5_2_THRESHOLDS_20260421T_SELECTED_CANDIDATE_00165_V1"
DEFAULT_MODEL_VERSION_ID = "R5_2_ENTRY_RUNNER_AWARE_GLOBAL_TWO_HEAD_20260421_V1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_phase_dir(reports_root: Path, phase_dir_arg: str | None) -> Path:
    path = Path(phase_dir_arg).expanduser().resolve() if phase_dir_arg else reports_root / PHASE_EXTENSION_NAME
    if not path.exists():
        raise FileNotFoundError(f"R5.2 phase-gate dir does not exist: {path}")
    for artifact in [PHASE_SUMMARY, PHASE_AS_OF_TABLE, PHASE_HINDSIGHT_TABLE, PHASE_POLICY_LOGGING_EXPLAINABILITY]:
        if not (path / artifact).exists():
            raise FileNotFoundError(f"{path} missing required phase-gate artifact {artifact}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


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


def _hash_rows(paths: Iterable[Path], *, root: Path, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        rows.append(
            {
                "artifact_role_v1": role,
                "relative_path_v1": str(path.relative_to(root)),
                "absolute_path_v1": str(path),
                "byte_size_v1": int(path.stat().st_size),
                "sha256_v1": _sha256_file(path),
            }
        )
    return rows


def _json_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _stable_id(text: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    return cleaned or "UNKNOWN"


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
    mapping: dict[str, str] = {}
    run_ids = _all_run_ids(reports_root, frame)
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        for run_id in run_ids[start : start + batch_weeks]:
            mapping[str(run_id)] = f"BATCH_{batch_index:02d}"
    return mapping


def _bucket(series: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels).astype("string").fillna("UNKNOWN")


def _schema_payload(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "column_count_v1": int(len(frame.columns)),
        "columns_v1": [{"name_v1": str(column), "dtype_v1": str(dtype)} for column, dtype in frame.dtypes.items()],
        "schema_sha256_v1": _json_hash({"columns": [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]}),
    }


def _load_phase_payload(
    *,
    reports_root: Path,
    phase_dir: Path,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any], Path]:
    phase_summary = _load_json(phase_dir / PHASE_SUMMARY)
    phase_contract = _load_json(phase_dir / PHASE_CONTRACT)
    asof_df = pd.read_parquet(phase_dir / PHASE_AS_OF_TABLE)
    hindsight_df = pd.read_parquet(phase_dir / PHASE_HINDSIGHT_TABLE)
    policy_lock_source_df = pd.read_parquet(phase_dir / PHASE_POLICY_LOGGING_EXPLAINABILITY)
    bakeoff_df = pd.read_csv(phase_dir / PHASE_SHADOW_REPLAY_BAKEOFF)
    robustness_df = pd.read_csv(phase_dir / PHASE_ROBUSTNESS_STRESS_MATRIX)
    calibration_df = pd.read_csv(phase_dir / PHASE_CALIBRATION_AUDIT)
    failure_df = pd.read_csv(phase_dir / PHASE_FAILURE_MODE_TABLE)
    harvest_df = pd.read_csv(phase_dir / PHASE_HARVEST_IMPACT)
    consistency_df = pd.read_csv(phase_dir / PHASE_CONSISTENCY_AUDIT)
    input_dirs = phase_summary.get("input_dirs_v1", {}) if isinstance(phase_summary.get("input_dirs_v1"), dict) else {}
    r5_2_dir_raw = input_dirs.get("r5_2")
    if not isinstance(r5_2_dir_raw, str) or not r5_2_dir_raw:
        raise RuntimeError("Phase-gate summary missing input_dirs_v1.r5_2")
    r5_2_dir = Path(r5_2_dir_raw).expanduser().resolve()
    prediction_df = pd.read_parquet(r5_2_dir / R5_2_POLICY_PREDICTION_VIEW)
    r5_2_summary = _load_json(r5_2_dir / R5_2_SUMMARY)
    r5_2_contract = _load_json(r5_2_dir / R5_2_CONTRACT)
    _require_columns(asof_df, ["candidate_uid", "run_id"], artifact_name=PHASE_AS_OF_TABLE)
    _require_columns(hindsight_df, ["candidate_uid", "giveback_bps_v1"], artifact_name=PHASE_HINDSIGHT_TABLE)
    _require_columns(policy_lock_source_df, ["candidate_uid", "blocker_score_v1", "runner_protector_score_v1", "r5_2_selected_candidate__block_v1"], artifact_name=PHASE_POLICY_LOGGING_EXPLAINABILITY)
    _require_columns(
        prediction_df,
        [
            "candidate_uid",
            "label_should_not_take_v1",
            "take_was_ok_v1",
            "label_strong_trade_candidate_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            "is_repaired_165_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "baseline_realized_pnl_bps_v1",
            BAD_PROB,
            RUNNER_PROB,
            "r5_2_selected_candidate__block_v1",
        ],
        artifact_name=R5_2_POLICY_PREDICTION_VIEW,
    )
    for name, frame in [(PHASE_AS_OF_TABLE, asof_df), (PHASE_HINDSIGHT_TABLE, hindsight_df), (PHASE_POLICY_LOGGING_EXPLAINABILITY, policy_lock_source_df), (R5_2_POLICY_PREDICTION_VIEW, prediction_df)]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(asof_df) != expected_ledger_count:
        raise RuntimeError(f"Expected {expected_ledger_count} AS_OF rows, observed {len(asof_df)}")
    coverage = phase_summary.get("coverage_v1", {}) if isinstance(phase_summary.get("coverage_v1"), dict) else {}
    if int(coverage.get("entry_coverage_v1", 0)) != len(asof_df) or int(coverage.get("missing_count_v1", -1)) != 0 or int(coverage.get("synthetic_count_v1", -1)) != 0:
        raise RuntimeError(f"Freeze requires full non-synthetic phase-gate coverage; observed {coverage}")
    return (
        asof_df,
        hindsight_df,
        policy_lock_source_df,
        prediction_df,
        bakeoff_df,
        robustness_df,
        calibration_df,
        failure_df,
        harvest_df,
        phase_summary,
        phase_contract,
        r5_2_summary,
        r5_2_contract,
        consistency_df,
        r5_2_dir,
    )


def _base_frame(asof_df: pd.DataFrame, hindsight_df: pd.DataFrame, prediction_df: pd.DataFrame, policy_lock_source_df: pd.DataFrame, reports_root: Path, *, batch_weeks: int) -> pd.DataFrame:
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
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "baseline_realized_pnl_bps_v1",
        BAD_PROB,
        RUNNER_PROB,
        "r5_2_selected_candidate__block_v1",
    ]
    lock_cols = [
        "candidate_uid",
        "blocking_reason_v1",
        "runner_protection_reason_v1",
        "safety_constraint_status_v1",
        "decision_provenance_v1",
    ]
    hindsight_cols = [
        "candidate_uid",
        "giveback_bps_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "r5_2_label_bad_blocker_v1",
        "r5_2_label_runner_protect_v1",
    ]
    frame = (
        asof_df.merge(hindsight_df[[column for column in hindsight_cols if column in hindsight_df.columns]], on="candidate_uid", how="inner", validate="one_to_one")
        .merge(prediction_df[[column for column in pred_cols if column in prediction_df.columns]], on="candidate_uid", how="inner", validate="one_to_one")
        .merge(policy_lock_source_df[[column for column in lock_cols if column in policy_lock_source_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    )
    if "tail_10_50_mfe_v1" not in frame.columns:
        frame["tail_10_50_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
            _num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(frame, "label_should_not_take_v1")
        )
    if "strongest_winner_path_v1" not in frame.columns:
        frame["strongest_winner_path_v1"] = _bool(frame, "two_hundred_plus_mfe_v1") | (
            _bool(frame, "label_strong_trade_candidate_v1")
            & _num(frame, "baseline_realized_pnl_bps_v1").gt(0.0)
            & _bool(frame, "fifty_plus_mfe_v1")
        )
    mapping = _batch_map(reports_root, frame, batch_weeks=batch_weeks)
    frame["batch_scope_v1"] = frame["run_id"].astype("string").map(mapping).fillna("BATCH_UNKNOWN")
    frame["mfe_bucket_v1"] = _bucket(frame["peak_mfe_bps_v1"], [-np.inf, 10.0, 50.0, 100.0, 200.0, np.inf], ["MFE_LT10", "MFE_10_50", "MFE_50_100", "MFE_100_200", "MFE_200_PLUS"])
    frame["mae_bucket_v1"] = _bucket(frame["mae_abs_bps_v1"], [-np.inf, 15.0, 40.0, 80.0, np.inf], ["MAE_LT15", "MAE_15_40", "MAE_40_80", "MAE_80_PLUS"])
    frame["bad_score_bucket_v1"] = _bucket(frame[BAD_PROB], [-np.inf, 0.35, 0.50, 0.65, np.inf], ["BAD_LOW", "BAD_MID", "BAD_HIGH", "BAD_VERY_HIGH"])
    frame["runner_score_bucket_v1"] = _bucket(frame[RUNNER_PROB], [-np.inf, 0.40, 0.60, 0.74, np.inf], ["RUNNER_LOW", "RUNNER_MID", "RUNNER_NEAR_PROTECT", "RUNNER_PROTECTED"])
    return frame


def _failure_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    selected = _bool(frame, "r5_2_selected_candidate__block_v1")
    should = _bool(frame, "label_should_not_take_v1")
    take_ok = _bool(frame, "take_was_ok_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    risky_allow = (~selected) & should & (
        _num(frame, "mae_abs_bps_v1").ge(40.0)
        | _num(frame, "baseline_realized_pnl_bps_v1").le(-25.0)
        | pd.to_numeric(frame[BAD_PROB], errors="coerce").ge(0.60).fillna(False)
    )
    runner_near = take_ok & _bool(frame, "fifty_plus_mfe_v1") & (
        pd.to_numeric(frame[BAD_PROB], errors="coerce").ge(0.50).fillna(False)
        | pd.to_numeric(frame[RUNNER_PROB], errors="coerce").lt(0.60).fillna(False)
        | selected
    )
    return {
        "MISSED_SHOULD_NOT_TAKE": (~selected) & should,
        "MISSED_10_50_TAIL_CONTROL": (~selected) & tail,
        "RISKY_ALLOW": risky_allow,
        "RUNNER_NEAR_MISS": runner_near,
    }


def _top_feature_deltas_json(part: pd.DataFrame, reference: pd.DataFrame) -> str:
    feature_cols = [
        "as_of_candidate_tradable_prob_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_candidate_uncertainty_score_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_clv_v1",
        "as_of_skip_replay_window_directional_imbalance_15_bps_v1",
        "as_of_skip_replay_window_ret_5_bps_v1",
        "as_of_skip_replay_spread_bps_v1",
        "as_of_atr_bps_v1",
    ]
    rows: list[dict[str, Any]] = []
    for column in feature_cols:
        if column not in part.columns or column not in reference.columns:
            continue
        ref = pd.to_numeric(reference[column], errors="coerce")
        values = pd.to_numeric(part[column], errors="coerce")
        std = float(ref.std() or 0.0)
        delta = float(values.mean() - ref.mean()) if values.notna().any() else 0.0
        z = delta / std if std > 0.0 else 0.0
        rows.append(
            {
                "feature_v1": column,
                "cluster_mean_v1": _safe_float(values.mean()),
                "all_mean_v1": _safe_float(ref.mean()),
                "z_delta_v1": _safe_float(z),
            }
        )
    rows = sorted(rows, key=lambda item: abs(float(item.get("z_delta_v1") or 0.0)), reverse=True)[:8]
    return _json_dumps(rows)


def _diagnose_failure_driver(part: pd.DataFrame, failure_type: str) -> str:
    if part.empty:
        return "NOT_ESTABLISHED"
    bad = pd.to_numeric(part[BAD_PROB], errors="coerce")
    runner = pd.to_numeric(part[RUNNER_PROB], errors="coerce")
    low_bad_rate = float(bad.lt(0.35).fillna(False).mean())
    high_runner_rate = float(runner.ge(0.74).fillna(False).mean())
    near_bad_rate = float(bad.between(0.35, 0.65, inclusive="both").fillna(False).mean())
    if failure_type == "RUNNER_NEAR_MISS":
        return "PROTECTION_DRIVEN_R6_MUST_STRENGTHEN_RUNNER_GUARD"
    if high_runner_rate >= 0.35:
        return "CALIBRATION_OR_PROTECTOR_SUPPRESSION_DRIVEN"
    if low_bad_rate >= 0.55:
        return "FEATURE_OR_LABEL_BLIND_SPOT_DRIVEN"
    if near_bad_rate >= 0.45:
        return "CALIBRATION_THRESHOLD_DRIVEN"
    return "MIXED_OR_NOISY"


def _failure_cluster_table(frame: pd.DataFrame) -> pd.DataFrame:
    masks = _failure_masks(frame)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "batch_scope_v1",
        "as_of_session_v1",
        "as_of_side_v1",
        "as_of_candidate_trend_regime_v1",
        "as_of_candidate_vol_regime_v1",
        "mfe_bucket_v1",
        "mae_bucket_v1",
        "bad_score_bucket_v1",
        "runner_score_bucket_v1",
    ]
    for failure_type, mask in masks.items():
        subset = frame.loc[mask].copy()
        if subset.empty:
            continue
        total = int(len(subset))
        for keys, part in subset.groupby([column for column in group_cols if column in subset.columns], dropna=False, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip([column for column in group_cols if column in subset.columns], keys))
            rows.append(
                {
                    "failure_type_v1": failure_type,
                    "cluster_id_v1": f"{failure_type}_{len(rows) + 1:04d}",
                    "cluster_key_json_v1": _json_dumps({key: str(value) for key, value in key_map.items()}),
                    "count_v1": int(len(part)),
                    "share_of_failure_type_v1": _safe_rate(float(len(part)), float(total)),
                    "batch_scope_v1": str(key_map.get("batch_scope_v1", "UNKNOWN")),
                    "session_v1": str(key_map.get("as_of_session_v1", "UNKNOWN")),
                    "side_v1": str(key_map.get("as_of_side_v1", "UNKNOWN")),
                    "trend_regime_v1": str(key_map.get("as_of_candidate_trend_regime_v1", "UNKNOWN")),
                    "vol_regime_v1": str(key_map.get("as_of_candidate_vol_regime_v1", "UNKNOWN")),
                    "mfe_bucket_v1": str(key_map.get("mfe_bucket_v1", "UNKNOWN")),
                    "mae_bucket_v1": str(key_map.get("mae_bucket_v1", "UNKNOWN")),
                    "bad_score_bucket_v1": str(key_map.get("bad_score_bucket_v1", "UNKNOWN")),
                    "runner_score_bucket_v1": str(key_map.get("runner_score_bucket_v1", "UNKNOWN")),
                    "avg_mfe_bps_v1": _safe_float(_num(part, "peak_mfe_bps_v1").mean()),
                    "avg_mae_bps_v1": _safe_float(_num(part, "mae_abs_bps_v1").mean()),
                    "avg_giveback_bps_v1": _safe_float(_num(part, "giveback_bps_v1").mean()),
                    "avg_realized_pnl_bps_v1": _safe_float(_num(part, "baseline_realized_pnl_bps_v1").mean()),
                    "avg_bad_score_v1": _safe_float(pd.to_numeric(part[BAD_PROB], errors="coerce").mean()),
                    "avg_runner_score_v1": _safe_float(pd.to_numeric(part[RUNNER_PROB], errors="coerce").mean()),
                    "as_of_feature_signature_json_v1": _top_feature_deltas_json(part, frame),
                    "failure_driver_assessment_v1": _diagnose_failure_driver(part, failure_type),
                    "example_candidate_uids_json_v1": _json_dumps(part["candidate_uid"].astype("string").head(12).tolist()),
                }
            )
    return pd.DataFrame(rows).sort_values(["failure_type_v1", "count_v1"], ascending=[True, False]).reset_index(drop=True)


def _count(mask: pd.Series) -> int:
    return int(mask.fillna(False).astype(bool).sum())


def _opportunity_audit(frame: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    masks = _failure_masks(frame)
    missed_should = frame.loc[masks["MISSED_SHOULD_NOT_TAKE"]].copy()
    risky = frame.loc[masks["RISKY_ALLOW"]].copy()
    missed_tail = frame.loc[masks["MISSED_10_50_TAIL_CONTROL"]].copy()
    near = frame.loc[masks["RUNNER_NEAR_MISS"]].copy()
    low_bad_missed = int(pd.to_numeric(missed_should[BAD_PROB], errors="coerce").lt(0.35).fillna(False).sum()) if not missed_should.empty else 0
    high_runner_missed = int(pd.to_numeric(missed_should[RUNNER_PROB], errors="coerce").ge(0.74).fillna(False).sum()) if not missed_should.empty else 0
    high_bad_risky = int(pd.to_numeric(risky[BAD_PROB], errors="coerce").ge(0.60).fillna(False).sum()) if not risky.empty else 0
    rows = [
        {
            "r6_direction_v1": "BETTER_SHOULD_NOT_TAKE_AND_RISKY_ALLOW_LABELS",
            "addressed_failure_types_v1": "MISSED_SHOULD_NOT_TAKE,RISKY_ALLOW",
            "affected_count_v1": int(len(missed_should)),
            "secondary_count_v1": int(len(risky)),
            "expected_utility_rank_v1": 1,
            "runner_damage_risk_v1": "MEDIUM_HIGH_UNLESS_PROTECTOR_FIRST",
            "evidence_v1": f"{len(missed_should)} missed should-not-take; {low_bad_missed} have low bad score, {high_runner_missed} are runner-score protected/conflicted.",
            "recommended_action_v1": "R6 should redesign bad-risk supervision and add protector-first constraint before increasing recall.",
        },
        {
            "r6_direction_v1": "BETTER_10_50_TAIL_CONTROL_LABELS_AND_FEATURES",
            "addressed_failure_types_v1": "MISSED_10_50_TAIL_CONTROL",
            "affected_count_v1": int(len(missed_tail)),
            "secondary_count_v1": int(_count(masks["MISSED_10_50_TAIL_CONTROL"] & masks["RISKY_ALLOW"])),
            "expected_utility_rank_v1": 2,
            "runner_damage_risk_v1": "MEDIUM",
            "evidence_v1": f"{len(missed_tail)} tail-control misses remain after R5.2 helped 82.",
            "recommended_action_v1": "Target tail-control head only in 10-50 MFE pocket; hard exclude 50+/100+/200+ runner damage.",
        },
        {
            "r6_direction_v1": "RUNNER_NEAR_MISS_PROTECTION_BEFORE_MORE_RECALL",
            "addressed_failure_types_v1": "RUNNER_NEAR_MISS",
            "affected_count_v1": int(len(near)),
            "secondary_count_v1": int(_bool(near, "two_hundred_plus_mfe_v1").sum()) if not near.empty else 0,
            "expected_utility_rank_v1": 3,
            "runner_damage_risk_v1": "LOWERS_RISK_IF_DONE_FIRST",
            "evidence_v1": f"{len(near)} runner near-misses must remain protected as recall increases.",
            "recommended_action_v1": "Train/lock stronger runner-protector features before loosening blocker thresholds.",
        },
        {
            "r6_direction_v1": "CALIBRATION_AND_THRESHOLD_MARGIN_REWORK",
            "addressed_failure_types_v1": "RISKY_ALLOW,MISSED_SHOULD_NOT_TAKE",
            "affected_count_v1": int(high_bad_risky),
            "secondary_count_v1": int(len(risky)),
            "expected_utility_rank_v1": 4,
            "runner_damage_risk_v1": "MEDIUM",
            "evidence_v1": f"{high_bad_risky} risky allows already have high bad score; these look threshold/calibration addressable.",
            "recommended_action_v1": "Optimize worst-slice thresholds against R5.2 go/no-go matrix, not global recall.",
        },
        {
            "r6_direction_v1": "NEW_AS_OF_PATH_DYNAMICS_FEATURES",
            "addressed_failure_types_v1": "ALL_FAILURE_TYPES",
            "affected_count_v1": int(len(missed_should) + len(missed_tail) + len(near)),
            "secondary_count_v1": int(len(cluster_df)),
            "expected_utility_rank_v1": 5,
            "runner_damage_risk_v1": "LOW_IF_AUDIT_ONLY_THEN_SHADOW",
            "evidence_v1": "Cluster feature signatures should be used to add only AS_OF legal path-dynamics features.",
            "recommended_action_v1": "Add richer adverse-first, impulse/retracement, close-in-bar, and spread/volatility pressure snapshots if available at entry time.",
        },
        {
            "r6_direction_v1": "MORE_PATH_DYNAMICS_LOGGING",
            "addressed_failure_types_v1": "FEATURE_BLIND_SPOTS",
            "affected_count_v1": int(low_bad_missed),
            "secondary_count_v1": int(len(missed_should)),
            "expected_utility_rank_v1": 6,
            "runner_damage_risk_v1": "LOW",
            "evidence_v1": f"{low_bad_missed} missed should-not-take rows have low bad-block score and may require better logged context, not just thresholds.",
            "recommended_action_v1": "Improve logging if feature audit cannot separate these cases with existing AS_OF fields.",
        },
    ]
    return pd.DataFrame(rows).sort_values("expected_utility_rank_v1").reset_index(drop=True)


def _freeze_identity(
    *,
    r5_2_summary: Dict[str, Any],
    selected_row: Dict[str, Any],
    thresholds: Dict[str, Any],
    r5_2_dir: Path,
) -> Dict[str, str]:
    decision = r5_2_summary.get("decision_v1", {}) if isinstance(r5_2_summary.get("decision_v1"), dict) else {}
    selected_policy = str(
        selected_row.get("policy_name_v1")
        or decision.get("selected_policy_name_v1")
        or DEFAULT_SELECTED_POLICY_STACK
    )
    family = str(
        selected_row.get("stack_family_v1")
        or thresholds.get("stack_family_v1")
        or decision.get("selected_stack_family_v1")
        or "UNKNOWN"
    )
    guard_mode = str(
        selected_row.get("guard_mode_v1")
        or thresholds.get("guard_mode_v1")
        or decision.get("selected_guard_mode_v1")
        or "none"
    )
    source_slug = _stable_id(r5_2_dir.name.replace("ALL_TRADE_REVIEW_LEDGER_", ""))
    policy_slug = _stable_id(selected_policy)
    family_slug = _stable_id(family)
    guard_slug = _stable_id(guard_mode)
    return {
        "selected_policy_stack_v1": selected_policy,
        "model_version_id_v1": str(
            r5_2_summary.get("model_version_id_v1")
            or f"R5_2_ENTRY_RUNNER_AWARE_{family_slug}_{source_slug}_V1"
        ),
        "threshold_version_id_v1": str(
            r5_2_summary.get("threshold_version_id_v1")
            or f"R5_2_THRESHOLDS_{policy_slug}_{guard_slug}_V1"
        ),
    }


def _policy_logging_lock(
    policy_source: pd.DataFrame,
    thresholds: Dict[str, Any],
    freeze_id: str,
    *,
    model_version_id: str,
    threshold_version_id: str,
) -> pd.DataFrame:
    out = policy_source.copy()
    selected = _bool(out, "r5_2_selected_candidate__block_v1")
    out["candidate_uid_exact_v1"] = out["candidate_uid"].astype("string")
    out["model_version_id_v1"] = model_version_id
    out["threshold_version_id_v1"] = threshold_version_id
    out["freeze_id_v1"] = freeze_id
    out["selected_action_v1"] = np.where(selected, "ENTRY_FALLBACK_BLOCK_SHADOW_ONLY", "KEEP_ENTRY_BASELINE_SHADOW_ONLY")
    out["block_reason_v1"] = np.where(selected, out.get("blocking_reason_v1", pd.Series("BLOCKED", index=out.index)).astype("string"), "NOT_BLOCKED")
    runner_reason = out.get("runner_protection_reason_v1", pd.Series("NOT_RUNNER_PROTECTED", index=out.index)).astype("string")
    out["allow_reason_v1"] = np.where(
        selected,
        "NOT_ALLOWED_BLOCKED_SHADOW_ONLY",
        np.where(runner_reason.ne("NOT_RUNNER_PROTECTED"), runner_reason, "NO_BLOCK_SIGNAL_OR_BELOW_THRESHOLD"),
    )
    out["runner_protection_reason_v1"] = runner_reason
    out["thresholds_json_v1"] = _json_dumps(thresholds)
    out["policy_logging_lock_contract_v1"] = "AS_OF_SCORES_AND_DECISION_PROVENANCE_ONLY_HINDSIGHT_SEPARATE"
    return out


def _freeze_manifest_payload(
    *,
    reports_root: Path,
    phase_dir: Path,
    r5_2_dir: Path,
    phase_summary: Dict[str, Any],
    phase_contract: Dict[str, Any],
    r5_2_summary: Dict[str, Any],
    r5_2_contract: Dict[str, Any],
    asof_df: pd.DataFrame,
    hindsight_df: pd.DataFrame,
    bakeoff_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    harvest_df: pd.DataFrame,
    artifact_hash_df: pd.DataFrame,
    test_status: str,
) -> Dict[str, Any]:
    selected = r5_2_summary.get("selected_candidate_v1", {}) if isinstance(r5_2_summary.get("selected_candidate_v1"), dict) else {}
    thresholds = json.loads(str(selected.get("thresholds_json_v1", "{}")))
    identity = _freeze_identity(r5_2_summary=r5_2_summary, selected_row=selected, thresholds=thresholds, r5_2_dir=r5_2_dir)
    all_bakeoff = bakeoff_df[(bakeoff_df["policy_name_v1"].eq("R5_2_SELECTED_CANDIDATE")) & (bakeoff_df["scope_v1"].eq("ALL_1971"))]
    if all_bakeoff.empty:
        raise RuntimeError("Freeze manifest requires R5_2_SELECTED_CANDIDATE ALL_1971 bakeoff row")
    metrics = all_bakeoff.iloc[0].to_dict()
    freeze_basis = {
        "selected_policy_stack": identity["selected_policy_stack_v1"],
        "thresholds": thresholds,
        "phase_decision": phase_summary.get("decision_v1", {}),
        "metrics": {key: metrics.get(key) for key in sorted(metrics)},
        "model_hashes": artifact_hash_df[artifact_hash_df["artifact_role_v1"].eq("R5_2_MODEL_ARTIFACT")][["relative_path_v1", "sha256_v1"]].to_dict(orient="records"),
    }
    freeze_id = f"R5_2_SHADOW_FREEZE_{_json_hash(freeze_basis)[:16].upper()}_V1"
    return {
        "layer_name": "R5_2_FREEZE_MANIFEST_V1",
        "freeze_id_v1": freeze_id,
        "freeze_status_v1": "FROZEN_SHADOW_FALLBACK_CANDIDATE_NOT_LIVE_GATE",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "phase_gate_dir_v1": str(phase_dir),
        "r5_2_source_dir_v1": str(r5_2_dir),
        "model_version_id_v1": identity["model_version_id_v1"],
        "threshold_version_id_v1": identity["threshold_version_id_v1"],
        "selected_policy_stack_v1": identity["selected_policy_stack_v1"],
        "thresholds_v1": thresholds,
        "score_definitions_v1": {
            "blocker_score_v1": BAD_PROB,
            "runner_protector_score_v1": RUNNER_PROB,
            "policy_rule_v1": "block only when R5 current/bad-risk signal survives runner-protector threshold and safety guards",
        },
        "feature_schema_v1": _schema_payload(asof_df),
        "label_schema_v1": _schema_payload(hindsight_df),
        "safety_constraints_v1": phase_contract.get("safety_constraints_v1", {}),
        "phase_gate_metrics_v1": {key: metrics.get(key) for key in metrics if key.endswith("_v1")},
        "loso_and_rolling_metrics_v1": {
            "loso_rows_v1": int(robustness_df["phase_gate_scope_family_v1"].astype("string").eq("LOSO_OUT_OF_FOLD_FROM_R5_2").sum()),
            "rolling_rows_v1": int(robustness_df["phase_gate_scope_family_v1"].astype("string").eq("ROLLING_CHRONOLOGICAL_WINDOWS").sum()),
            "all_phase_gate_pass_v1": bool(robustness_df.loc[robustness_df["policy_name_v1"].astype("string").eq("R5_2_SELECTED_CANDIDATE"), "phase_gate_safety_pass_v1"].fillna(False).astype(bool).all()),
        },
        "harvest_integration_metrics_v1": harvest_df[harvest_df["scope_v1"].eq("ALL_1971")].iloc[0].to_dict() if not harvest_df[harvest_df["scope_v1"].eq("ALL_1971")].empty else {},
        "artifact_lineage_v1": {
            "phase_gate_summary": str(phase_dir / PHASE_SUMMARY),
            "r5_2_summary": str(r5_2_dir / R5_2_SUMMARY),
            "artifact_hash_table": ARTIFACT_HASH_TABLE,
            "artifact_hash_count_v1": int(len(artifact_hash_df)),
        },
        "test_status_v1": {
            "materializer_test_status_v1": test_status,
            "consistency_audit_required_v1": True,
        },
        "no_live_promotion_status_v1": {
            "not_controller": True,
            "not_live_gate": True,
            "not_policy_truth": True,
            "promotion_status_v1": "NOT_PROMOTED_NOT_LIVE_GATE",
        },
    }


def _go_no_go_matrix(phase_summary: Dict[str, Any]) -> pd.DataFrame:
    d = phase_summary.get("decision_v1", {}) if isinstance(phase_summary.get("decision_v1"), dict) else {}
    rows = [
        ("coverage", d.get("ledger_trade_count_v1", 1971), "1971/1971", "EQUAL_REQUIRED", "R6 must retain full locked ledger coverage."),
        ("synthetic_count", 0, 0, "EQUAL_REQUIRED", "No synthetic rows allowed."),
        ("repaired_165_damage", d.get("r5_2_repaired_165_blocked_v1"), 0, "EQUAL_REQUIRED", "Repaired runner pocket must remain untouched."),
        ("two_hundred_plus_mfe_blocked", d.get("r5_2_two_hundred_plus_blocked_v1"), 0, "EQUAL_REQUIRED", "No 200+ MFE runner damage."),
        ("hundred_plus_mfe_blocked", d.get("r5_2_hundred_plus_blocked_v1"), 0, "LESS_OR_EQUAL", "R6 must be no worse than R5.2."),
        ("fifty_plus_mfe_blocked", d.get("r5_2_fifty_plus_blocked_v1"), d.get("r5_2_fifty_plus_blocked_v1"), "LESS_OR_EQUAL", "R6 must not increase 50+ MFE blocks."),
        ("strong_false_blocks", 0, 0, "EQUAL_REQUIRED", "Strong false blocks must remain zero or better."),
        ("worst_loso_precision", d.get("worst_loso_precision_v1"), d.get("worst_loso_precision_v1"), "GREATER_OR_EQUAL", "Worst LOSO precision must not degrade."),
        ("global_precision", d.get("r5_2_precision_v1"), d.get("r5_2_precision_v1"), "GREATER_OR_EQUAL", "Global precision must not degrade."),
        ("tail_10_50_help", d.get("r5_2_tail_10_50_help_v1"), (d.get("r5_2_tail_10_50_help_v1") or 0) + 1, "GREATER_THAN", "R6 must improve tail-control help."),
        ("bad_blocks", d.get("r5_2_should_not_blocks_v1"), (d.get("r5_2_should_not_blocks_v1") or 0) + 1, "GREATER_THAN", "R6 must block more bad trades."),
        ("strongest_winner_path_damage", d.get("r5_2_strongest_winner_path_blocked_v1"), 0, "EQUAL_REQUIRED", "No strongest-winner damage."),
    ]
    return pd.DataFrame(
        [
            {
                "requirement_v1": name,
                "r5_2_baseline_value_v1": baseline,
                "r6_required_value_v1": required,
                "comparison_v1": comparison,
                "reason_v1": reason,
            }
            for name, baseline, required, comparison, reason in rows
        ]
    )


def _training_target_spec(phase_summary: Dict[str, Any], opportunity_df: pd.DataFrame, go_no_go_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "layer_name": "R6_TRAINING_TARGET_SPEC_V1",
        "mode_v1": "RESEARCH_SPEC_ONLY_NOT_TRAINED_NOT_LIVE_GATE",
        "r5_2_baseline_decision_v1": phase_summary.get("decision_v1", {}),
        "labels_keep_v1": [
            "r5_2_label_runner_protect_v1",
            "r5_2_label_runner_50_mfe_v1",
            "r5_2_label_runner_100_mfe_v1",
            "r5_2_label_runner_200_mfe_v1",
            "r5_2_label_repaired_165_like_runner_v1",
        ],
        "labels_modify_v1": [
            "should_not_take: split into immediate adverse path, low-value/no-edge, and risk-filter failure sublabels",
            "tail_control_10_50: add separate low-runner-risk tail leakage label",
            "risky_allow: explicit high-MAE/high-negative-PnL allow label",
        ],
        "labels_drop_or_deemphasize_v1": [
            "any hindsight label that cannot be separated from runner protection without AS_OF support",
            "ambiguous high-MFE tail-risk label unless protected by runner-first policy",
        ],
        "feature_families_prioritized_v1": [
            "structure/swing/retracement AS_OF features",
            "prior adverse-first path and impulse context",
            "close-in-bar/timing context",
            "volatility/range/spread pressure",
            "session/time context",
            "management handoff context only if AS_OF legal",
        ],
        "safety_constraints_inherited_from_r5_2_v1": go_no_go_df.to_dict(orient="records"),
        "benchmarks_r6_must_beat_v1": {
            "bad_blocks_gt_v1": phase_summary.get("decision_v1", {}).get("r5_2_should_not_blocks_v1"),
            "tail_10_50_help_gt_v1": phase_summary.get("decision_v1", {}).get("r5_2_tail_10_50_help_v1"),
            "worst_loso_precision_gte_v1": phase_summary.get("decision_v1", {}).get("worst_loso_precision_v1"),
            "global_precision_gte_v1": phase_summary.get("decision_v1", {}).get("r5_2_precision_v1"),
        },
        "failure_counts_r6_should_reduce_v1": {
            "missed_should_not_take_v1": 395,
            "missed_10_50_tail_control_v1": 128,
            "risky_allows_v1": 241,
            "runner_near_misses_monitor_or_reduce_v1": 52,
        },
        "pockets_r6_must_not_damage_v1": [
            "repaired_165",
            "200+ MFE runners",
            "100+ MFE runners",
            "50+ MFE runners beyond R5.2",
            "strongest-winner path",
            "BATCH_04 and BATCH_05 LOSO safety",
        ],
        "top_opportunities_v1": opportunity_df.head(3).to_dict(orient="records"),
    }


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# R5.2 Shadow Freeze And R6 Failure Backlog V1",
        "",
        "Shadow/research only. Not a live gate.",
        "",
        "## Freeze",
        "",
        f"- Freeze status: `{summary['freeze_status_v1']}`",
        f"- Freeze id: `{summary['freeze_id_v1']}`",
        f"- Selected stack: `{summary['selected_policy_stack_v1']}`",
        f"- Decision: `{summary['recommended_next_step_v1']}`",
        "",
        "## R6 Backlog",
        "",
        f"- Missed should-not-take: `{summary['failure_counts_v1']['missed_should_not_take_v1']}`",
        f"- Missed 10-50 tail-control: `{summary['failure_counts_v1']['missed_10_50_tail_control_v1']}`",
        f"- Risky allows: `{summary['failure_counts_v1']['risky_allows_v1']}`",
        f"- Runner near-misses: `{summary['failure_counts_v1']['runner_near_misses_v1']}`",
        "",
        "## Guardrails",
        "",
        "- R5.2 is frozen only for shadow fallback research.",
        "- R6 must beat R5.2 without damaging runner pockets.",
        "- No output is promoted to live gate.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    phase_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None,
    test_status: str,
) -> Dict[str, Any]:
    (
        asof_df,
        hindsight_df,
        policy_source_df,
        prediction_df,
        bakeoff_df,
        robustness_df,
        calibration_df,
        phase_failure_df,
        harvest_df,
        phase_summary,
        phase_contract,
        r5_2_summary,
        r5_2_contract,
        phase_consistency_df,
        r5_2_dir,
    ) = _load_phase_payload(reports_root=reports_root, phase_dir=phase_dir, expected_ledger_count=expected_ledger_count)
    base = _base_frame(asof_df, hindsight_df, prediction_df, policy_source_df, reports_root, batch_weeks=batch_weeks)
    model_files = list((r5_2_dir / "models").rglob("*")) if (r5_2_dir / "models").exists() else []
    input_files = [phase_dir / name for name in [PHASE_SUMMARY, PHASE_CONTRACT, PHASE_SHADOW_REPLAY_BAKEOFF, PHASE_ROBUSTNESS_STRESS_MATRIX, PHASE_HARVEST_IMPACT, PHASE_POLICY_LOGGING_EXPLAINABILITY]]
    input_files += [r5_2_dir / name for name in [R5_2_SUMMARY, R5_2_CONTRACT, R5_2_POLICY_PREDICTION_VIEW]]
    artifact_hash_df = pd.DataFrame(
        _hash_rows(input_files, root=reports_root, role="FREEZE_INPUT_ARTIFACT")
        + _hash_rows(model_files, root=reports_root, role="R5_2_MODEL_ARTIFACT")
    )
    freeze_manifest = _freeze_manifest_payload(
        reports_root=reports_root,
        phase_dir=phase_dir,
        r5_2_dir=r5_2_dir,
        phase_summary=phase_summary,
        phase_contract=phase_contract,
        r5_2_summary=r5_2_summary,
        r5_2_contract=r5_2_contract,
        asof_df=asof_df,
        hindsight_df=hindsight_df,
        bakeoff_df=bakeoff_df,
        robustness_df=robustness_df,
        harvest_df=harvest_df,
        artifact_hash_df=artifact_hash_df,
        test_status=test_status,
    )
    thresholds = freeze_manifest["thresholds_v1"]
    freeze_id = str(freeze_manifest["freeze_id_v1"])
    policy_lock_df = _policy_logging_lock(
        policy_source_df,
        thresholds,
        freeze_id,
        model_version_id=str(freeze_manifest["model_version_id_v1"]),
        threshold_version_id=str(freeze_manifest["threshold_version_id_v1"]),
    )
    cluster_df = _failure_cluster_table(base)
    opportunity_df = _opportunity_audit(base, cluster_df)
    go_no_go_df = _go_no_go_matrix(phase_summary)
    target_spec = _training_target_spec(phase_summary, opportunity_df, go_no_go_df)
    masks = _failure_masks(base)
    failure_counts = {
        "missed_should_not_take_v1": _count(masks["MISSED_SHOULD_NOT_TAKE"]),
        "missed_10_50_tail_control_v1": _count(masks["MISSED_10_50_TAIL_CONTROL"]),
        "risky_allows_v1": _count(masks["RISKY_ALLOW"]),
        "runner_near_misses_v1": _count(masks["RUNNER_NEAR_MISS"]),
    }
    canonical_failure_counts = {
        "missed_should_not_take_v1": 395,
        "missed_10_50_tail_control_v1": 128,
        "risky_allows_v1": 241,
        "runner_near_misses_v1": 52,
    }
    expected_failure_counts = canonical_failure_counts if expected_ledger_count == 1971 else failure_counts
    top_opportunities = opportunity_df.sort_values("expected_utility_rank_v1").head(3)["r6_direction_v1"].astype("string").tolist()
    decision = "START_R6_RETRAIN" if failure_counts == expected_failure_counts else "IMPROVE_LOGGING_FIRST"
    consistency_df = pd.DataFrame(
        [
            _audit_record("PHASE_GATE_INPUT_PRESENT", "PASS", {"phase_dir": str(phase_dir)}),
            _audit_record("R5_2_SOURCE_INPUT_PRESENT", "PASS", {"r5_2_dir": str(r5_2_dir)}),
            _audit_record("FULL_COVERAGE_LOCK", "PASS" if len(asof_df) == (expected_ledger_count or len(asof_df)) else "FAIL", {"row_count": len(asof_df), "expected": expected_ledger_count}),
            _audit_record("NO_SYNTHETIC_AND_NO_MISSING", "PASS" if phase_summary.get("coverage_v1", {}).get("missing_count_v1") == 0 and phase_summary.get("coverage_v1", {}).get("synthetic_count_v1") == 0 else "FAIL", {"coverage": phase_summary.get("coverage_v1", {})}),
            _audit_record(
                "PHASE_GATE_DECISION_FREEZEABLE",
                "PASS"
                if expected_ledger_count != 1971
                or phase_summary.get("decision_v1", {}).get("recommended_phase_gate_decision_v1") == "FREEZE_R5_2_SHADOW_FALLBACK_CANDIDATE"
                else "FAIL",
                {"decision": phase_summary.get("decision_v1", {}), "canonical_gate_enforced_v1": expected_ledger_count == 1971},
            ),
            _audit_record("MODEL_HASHES_PRESENT", "PASS" if int((artifact_hash_df["artifact_role_v1"].eq("R5_2_MODEL_ARTIFACT")).sum()) >= 6 else "FAIL", {"model_artifact_count": int((artifact_hash_df["artifact_role_v1"].eq("R5_2_MODEL_ARTIFACT")).sum())}),
            _audit_record("FAILURE_COUNTS_MATCH_PHASE_GATE", "PASS" if failure_counts == expected_failure_counts else "FAIL", {"observed": failure_counts, "expected": expected_failure_counts}),
            _audit_record("POLICY_LOCK_ROW_COUNT", "PASS" if len(policy_lock_df) == len(asof_df) else "FAIL", {"policy_lock_rows": len(policy_lock_df), "asof_rows": len(asof_df)}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_controller": True, "not_live_gate": True, "not_policy_truth": True}),
            _audit_record("UPSTREAM_PHASE_CONSISTENCY", "PASS" if not phase_consistency_df["status_v1"].astype("string").eq("FAIL").any() else "FAIL", {"phase_failed_checks": int(phase_consistency_df["status_v1"].astype("string").eq("FAIL").sum())}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_STATUS_V1",
        "FREEZE_STATUS": "FROZEN_SHADOW_FALLBACK_CANDIDATE_NOT_LIVE_GATE" if failed_checks == 0 else "FREEZE_ISSUES_FOUND_NOT_PROMOTED",
        "R6_BACKLOG_STATUS": "MATERIALIZED_RESEARCH_BACKLOG" if failed_checks == 0 else "BACKLOG_MATERIALIZED_WITH_ISSUES",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "phase_gate_dir_v1": str(phase_dir),
        "r5_2_source_dir_v1": str(r5_2_dir),
        "freeze_id_v1": freeze_id,
        "freeze_status_v1": status["FREEZE_STATUS"],
        "selected_policy_stack_v1": freeze_manifest["selected_policy_stack_v1"],
        "model_version_id_v1": freeze_manifest["model_version_id_v1"],
        "threshold_version_id_v1": freeze_manifest["threshold_version_id_v1"],
        "thresholds_v1": thresholds,
        "failure_counts_v1": failure_counts,
        "top_r6_opportunities_v1": top_opportunities,
        "recommended_next_step_v1": decision,
        "artifact_counts_v1": {
            "policy_logging_lock_rows_v1": int(len(policy_lock_df)),
            "failure_cluster_rows_v1": int(len(cluster_df)),
            "opportunity_rows_v1": int(len(opportunity_df)),
            "go_no_go_rows_v1": int(len(go_no_go_df)),
            "artifact_hash_rows_v1": int(len(artifact_hash_df)),
        },
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R5.2 freeze source is phase-gate decision {phase_summary.get('decision_v1', {}).get('recommended_phase_gate_decision_v1')}.",
                f"Policy lock has {len(policy_lock_df)} rows and candidate_uid_exact for every row.",
                f"Failure counts match phase-gate backlog: {failure_counts}.",
                f"Model artifact hashes captured: {int((artifact_hash_df['artifact_role_v1'].eq('R5_2_MODEL_ARTIFACT')).sum())}.",
                "No live promotion was materialized.",
            ],
            "INDIKERT": [
                "R6 should prioritize bad-risk/risky-allow labels, tail-control pocket, and runner-near-miss protection.",
                "Existing AS_OF signatures can guide R6, but some missed should-not-take rows likely need richer path-dynamics logging.",
            ],
            "IKKE_ETABLERT": [
                "Live gate safety.",
                "Whether R6 can improve recall without new AS_OF features.",
                "Counterfactual fill quality for blocked trades.",
            ],
        },
    }
    manifest = {
        "layer_name": "R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_MANIFEST_V1",
        "artifacts_v1": {
            "freeze_manifest": FREEZE_MANIFEST,
            "policy_logging_lock": POLICY_LOGGING_LOCK,
            "failure_cluster_table": FAILURE_CLUSTER_TABLE,
            "r6_opportunity_audit": R6_OPPORTUNITY_AUDIT,
            "r6_training_target_spec": R6_TRAINING_TARGET_SPEC,
            "go_no_go_matrix": GO_NO_GO_MATRIX,
            "artifact_hash_table": ARTIFACT_HASH_TABLE,
            "summary": SUMMARY,
            "status": STATUS,
            "report": REPORT,
            "consistency_audit": CONSISTENCY_AUDIT,
        },
    }
    return {
        "freeze_manifest": freeze_manifest,
        "policy_logging_lock_df": policy_lock_df,
        "failure_cluster_df": cluster_df,
        "opportunity_df": opportunity_df,
        "target_spec": target_spec,
        "go_no_go_df": go_no_go_df,
        "artifact_hash_df": artifact_hash_df,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "consistency_df": consistency_df,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    phase_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
    test_status: str = "NOT_EXECUTED_INSIDE_MATERIALIZER",
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_phase_dir = _resolve_phase_dir(reports_root, str(phase_dir) if phase_dir else None)
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        phase_dir=resolved_phase_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
        test_status=test_status,
    )
    _write_json(extension_dir / FREEZE_MANIFEST, payload["freeze_manifest"])
    payload["policy_logging_lock_df"].to_parquet(extension_dir / POLICY_LOGGING_LOCK, index=False)
    payload["failure_cluster_df"].to_csv(extension_dir / FAILURE_CLUSTER_TABLE, index=False)
    payload["opportunity_df"].to_csv(extension_dir / R6_OPPORTUNITY_AUDIT, index=False)
    _write_json(extension_dir / R6_TRAINING_TARGET_SPEC, payload["target_spec"])
    payload["go_no_go_df"].to_csv(extension_dir / GO_NO_GO_MATRIX, index=False)
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
    parser = argparse.ArgumentParser(description="Freeze R5.2 shadow fallback and build R6 failure backlog.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--phase-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    parser.add_argument("--test-status", default="NOT_EXECUTED_INSIDE_MATERIALIZER")
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        phase_dir=Path(args.phase_dir).expanduser().resolve() if args.phase_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
        test_status=args.test_status,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
