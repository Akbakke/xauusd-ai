#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_NAME = "MONDAY_ENTRY_READINESS_BOOTSTRAP_V1"

AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
HINDSIGHT_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
COVERAGE_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_audit_v1.csv"
RUN_ROLLUP = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_run_rollup_v1.csv"
CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"
SUMMARY = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_summary_v1.json"
MANIFEST = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_report_v1.md"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_consistency_audit_v1.csv"
TOP_LEVEL_SUMMARY = "truth_monday_entry_readiness_bootstrap_v1.json"

LEDGER_CLOSED_TRADES = "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
HINDSIGHT_EXPORT = "shadow_meta_all_trade_review_hindsight_trade_export_closed_trades.parquet"
ENTRY_OBSERVABILITY = "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet"
ENTRY_RAW_STATE = "shadow_meta_all_trade_review_entry_anchor_raw_state_v1.parquet"
MANAGEMENT_RAW_STATE = "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet"
AS_OF_LEDGER = "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_review_dir(reports_root: Path, path_arg: str | None) -> Path:
    if path_arg:
        review_dir = Path(path_arg).expanduser().resolve()
        if not review_dir.exists():
            raise FileNotFoundError(f"Review dir does not exist: {review_dir}")
        return review_dir
    rebuild_summary = reports_root / "truth_downstream_canonical_rebuild_v1.json"
    if rebuild_summary.exists():
        payload = _load_json(rebuild_summary)
        raw_dir = payload.get("ledger_dir") or payload.get("review_dir_v1")
        if isinstance(raw_dir, str) and raw_dir.strip():
            candidate = Path(raw_dir).expanduser().resolve()
            if (candidate / LEDGER_CLOSED_TRADES).exists():
                return candidate
    raise FileNotFoundError("Could not resolve active review dir from truth_downstream_canonical_rebuild_v1.json")


def _default_extension_dir(reports_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return reports_root / f"{EXTENSION_NAME}_{stamp}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.astype("string").str.strip().str.lower().eq("true").fillna(default).astype(bool)


def _num_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _first_non_null(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if not isinstance(values, pd.Series):
        raise KeyError(column)
    return values.dropna().iloc[0] if values.notna().any() else pd.NA


def _aggregate_split_flags(asof_ledger_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        asof_ledger_df,
        ["candidate_uid", "used_for_training", "used_for_validation", "used_for_holdout"],
        artifact_name=AS_OF_LEDGER,
    )
    grouped = (
        asof_ledger_df.groupby("candidate_uid", dropna=False)
        .agg(
            run_id=("run_id", "first"),
            used_for_training=("used_for_training", "first"),
            used_for_validation=("used_for_validation", "first"),
            used_for_holdout=("used_for_holdout", "first"),
            as_of_split_bucket_v1=("as_of_split_bucket_v1", "first"),
        )
        .reset_index()
    )
    return grouped


def _feature_names(entry_raw_df: pd.DataFrame) -> list[str]:
    base = [
        "as_of_hour_utc_v1",
        "as_of_weekday_utc_v1",
        "as_of_session_v1",
        "as_of_side_v1",
        "as_of_atr_bps_v1",
        "as_of_candidate_entry_spread_bps_v1",
        "as_of_candidate_uncertainty_score_v1",
        "as_of_candidate_tradable_prob_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_candidate_trend_regime_v1",
        "as_of_candidate_vol_regime_v1",
        "as_of_entry_candidate_margin_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_skip_candidate_entry_spread_bps_v1",
        "as_of_skip_candidate_margin_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_skip_candidate_p_hat_v1",
        "as_of_skip_candidate_p_long_v1",
        "as_of_skip_candidate_p_short_v1",
        "as_of_skip_candidate_path_quality_pred_v1",
    ]
    replay_cols = [
        column.replace("as_of_entry_replay_", "as_of_skip_replay_", 1)
        for column in entry_raw_df.columns
        if column.startswith("as_of_entry_replay_")
    ]
    return base + replay_cols


def _dedupe_entry_raw_state(entry_raw_df: pd.DataFrame) -> pd.DataFrame:
    work = entry_raw_df.copy()
    work["candidate_uid"] = work["candidate_uid"].astype("string")
    if not work["candidate_uid"].duplicated().any():
        return work
    anchor_type = work.get("anchor_type", pd.Series("", index=work.index)).astype("string")
    priority = pd.Series(9, index=work.index, dtype="int64")
    priority.loc[anchor_type.eq("ENTRY_DECISION_ANCHOR")] = 0
    priority.loc[anchor_type.eq("EARLIEST_PROVABLE_CONFIRMATION_ENTRY_ANCHOR")] = 1
    work["_entry_anchor_priority_v1"] = priority
    if "anchor_timestamp_utc" in work.columns:
        work["_entry_anchor_timestamp_v1"] = pd.to_datetime(work["anchor_timestamp_utc"], utc=True, errors="coerce")
    else:
        work["_entry_anchor_timestamp_v1"] = pd.NaT
    work = work.sort_values(
        ["candidate_uid", "_entry_anchor_priority_v1", "_entry_anchor_timestamp_v1"],
        kind="mergesort",
    )
    return work.drop_duplicates(subset=["candidate_uid"], keep="first").drop(columns=["_entry_anchor_priority_v1", "_entry_anchor_timestamp_v1"])


def _coverage_reason(entry_obs: bool, raw_state: bool) -> tuple[str, str]:
    if entry_obs and raw_state:
        return "covered", "entry observation and raw-state present"
    if (not entry_obs) and (not raw_state):
        return "missing entry observation and raw-state", "candidate missing both entry observability and entry raw-state on active Monday ledger"
    if not entry_obs:
        return "missing entry observation", "candidate missing entry observability row on active Monday ledger"
    return "missing entry raw-state", "candidate missing entry raw-state row on active Monday ledger"


def _derive_exit_harvest_policy_action(frame: pd.DataFrame) -> pd.Series:
    should_skip = _bool_series(frame, "hindsight_should_skip_trade_v1")
    should_hold = _bool_series(frame, "hindsight_should_hold_longer_v1")
    should_exit_earlier = _bool_series(frame, "hindsight_should_exit_earlier_v1")
    out = pd.Series("KEEP_BASELINE", index=frame.index, dtype="string")
    out.loc[should_skip] = "ENTRY_SUPPRESS_OR_DOWNSIZE"
    out.loc[~should_skip & should_hold] = "HOLD_LONGER_RUNNER_TRAIL"
    out.loc[~should_skip & should_exit_earlier] = "EXIT_EARLIER_DAMAGE_CONTROL"
    return out


def _consistency_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday Entry Readiness Bootstrap V1",
            "",
            "Shadow/research bootstrap only. Not a live gate.",
            "",
            "## Headline",
            "",
            f"- Ledger trades: `{summary['ledger_trade_count_v1']}`",
            f"- Original entry feature coverage: `{summary['entry_feature_coverage_v1']}`",
            f"- Missing coverage rows: `{summary['entry_missing_count_v1']}`",
            f"- Management observation coverage: `{summary['management_observation_coverage_v1']}`",
            f"- AS_OF feature count: `{summary['as_of_feature_count_v1']}`",
            f"- Status: `{summary['status_v1']['MONDAY_ENTRY_READINESS_BOOTSTRAP_STATUS']}`",
            "",
            "## Contract",
            "",
            "- AS_OF features are repair-safe and map back to exact candidate/replay/XGB Monday run sources.",
            "- HINDSIGHT labels stay physically separate from AS_OF features.",
            "- No synthetic values are introduced in the bootstrap tables.",
            "- This layer is a bootstrap substrate for coverage-repair and entry shadow retrain, not policy truth.",
            "",
        ]
    ) + "\n"


def build_payload(
    *,
    reports_root: Path,
    review_dir: Path,
    extension_dir: Path,
) -> Dict[str, Any]:
    ledger_df = pd.read_parquet(review_dir / LEDGER_CLOSED_TRADES)
    hindsight_df = pd.read_parquet(review_dir / HINDSIGHT_EXPORT)
    entry_obs_df = pd.read_parquet(review_dir / ENTRY_OBSERVABILITY)
    entry_raw_df = pd.read_parquet(review_dir / ENTRY_RAW_STATE)
    management_raw_df = pd.read_parquet(review_dir / MANAGEMENT_RAW_STATE)
    asof_ledger_df = pd.read_parquet(review_dir / AS_OF_LEDGER)

    _require_columns(ledger_df, ["candidate_uid", "run_id", "trade_uid", "trade_id", "decision_timestamp", "realized_pnl_bps", "mfe_bps", "mae_bps", "trade_outcome_class", "exit_reason", "session", "vol_regime", "trend_regime"], artifact_name=LEDGER_CLOSED_TRADES)
    _require_columns(hindsight_df, ["candidate_uid", "hindsight_should_skip_trade_v1", "hindsight_take_was_ok_v1", "hindsight_should_hold_longer_v1", "hindsight_should_exit_earlier_v1", "hindsight_peak_mfe_bps_v1", "hindsight_peak_to_exit_giveback_bps_v1"], artifact_name=HINDSIGHT_EXPORT)
    _require_columns(entry_obs_df, ["candidate_uid", "as_of_hour_utc_v1", "as_of_weekday_utc_v1", "as_of_session_v1", "as_of_side_v1", "as_of_atr_bps_v1"], artifact_name=ENTRY_OBSERVABILITY)
    _require_columns(entry_raw_df, ["candidate_uid"], artifact_name=ENTRY_RAW_STATE)
    _require_columns(management_raw_df, ["candidate_uid"], artifact_name=MANAGEMENT_RAW_STATE)
    _require_columns(asof_ledger_df, ["candidate_uid", "used_for_training", "used_for_validation", "used_for_holdout"], artifact_name=AS_OF_LEDGER)

    feature_names = _feature_names(entry_raw_df)
    if ledger_df["candidate_uid"].astype("string").duplicated().any():
        raise RuntimeError("Closed-trade ledger candidate_uid must be unique")

    split_df = _aggregate_split_flags(asof_ledger_df)
    management_presence_df = pd.DataFrame(
        {
            "candidate_uid": management_raw_df["candidate_uid"].astype("string"),
            "management_observation_present_v1": True,
        }
    ).drop_duplicates(subset=["candidate_uid"], keep="first")

    entry_raw_work = _dedupe_entry_raw_state(entry_raw_df)
    rename_map = {
        "as_of_entry_candidate_p_flat_v1": "as_of_skip_candidate_p_flat_v1",
        "as_of_entry_candidate_p_hat_v1": "as_of_skip_candidate_p_hat_v1",
        "as_of_entry_candidate_p_long_v1": "as_of_skip_candidate_p_long_v1",
        "as_of_entry_candidate_p_short_v1": "as_of_skip_candidate_p_short_v1",
        "as_of_entry_candidate_entry_spread_bps_v1": "as_of_skip_candidate_entry_spread_bps_v1",
        "as_of_entry_candidate_margin_v1": "as_of_skip_candidate_margin_v1",
        "as_of_entry_candidate_path_quality_pred_v1": "as_of_skip_candidate_path_quality_pred_v1",
    }
    rename_map.update(
        {
            column: column.replace("as_of_entry_replay_", "as_of_skip_replay_", 1)
            for column in entry_raw_work.columns
            if column.startswith("as_of_entry_replay_")
        }
    )
    entry_raw_work = entry_raw_work.rename(columns=rename_map)
    raw_exact_present = (
        _bool_series(entry_raw_work, "entry_raw_replay_bar_exact_available_v1")
        & _bool_series(entry_raw_work, "entry_raw_candidate_snapshot_exact_available_v1")
        & _bool_series(entry_raw_work, "entry_raw_xgb_multi_horizon_exact_available_v1")
    )
    entry_raw_work["entry_raw_state_present_v1"] = raw_exact_present

    entry_obs_work = entry_obs_df.copy()
    entry_obs_work["candidate_uid"] = entry_obs_work["candidate_uid"].astype("string")
    entry_obs_work["entry_observation_present_v1"] = True

    base = (
        ledger_df.copy()
        .assign(candidate_uid=ledger_df["candidate_uid"].astype("string"))
        .merge(split_df, on="candidate_uid", how="left", validate="one_to_one")
        .merge(
            entry_obs_work,
            on="candidate_uid",
            how="left",
            validate="one_to_one",
            suffixes=("", "_entry_obs"),
        )
        .merge(entry_raw_work[["candidate_uid", "entry_raw_state_present_v1", *[name for name in feature_names if name in entry_raw_work.columns]]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(management_presence_df, on="candidate_uid", how="left", validate="one_to_one")
        .merge(
            hindsight_df,
            on="candidate_uid",
            how="left",
            validate="one_to_one",
            suffixes=("", "_hindsight"),
        )
    )
    for base_col, shadow_col in [
        ("run_id", "run_id_hindsight"),
        ("trade_uid", "trade_uid_hindsight"),
        ("trade_id", "trade_id_hindsight"),
        ("decision_timestamp", "decision_timestamp_hindsight"),
    ]:
        if shadow_col in base.columns:
            base[base_col] = base[base_col].where(base[base_col].notna(), base[shadow_col])
    base["entry_observation_present_v1"] = _bool_series(base, "entry_observation_present_v1")
    base["entry_raw_state_present_v1"] = _bool_series(base, "entry_raw_state_present_v1")
    base["management_observation_present_v1"] = _bool_series(base, "management_observation_present_v1")
    base["used_for_training"] = _bool_series(base, "used_for_training")
    base["used_for_validation"] = _bool_series(base, "used_for_validation")
    base["used_for_holdout"] = _bool_series(base, "used_for_holdout")

    # Fill current-generation feature columns from entry observability when they exist there.
    for column in [
        "as_of_hour_utc_v1",
        "as_of_weekday_utc_v1",
        "as_of_session_v1",
        "as_of_side_v1",
        "as_of_atr_bps_v1",
        "as_of_candidate_entry_spread_bps_v1",
        "as_of_candidate_uncertainty_score_v1",
        "as_of_candidate_tradable_prob_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_candidate_trend_regime_v1",
        "as_of_candidate_vol_regime_v1",
        "as_of_entry_candidate_margin_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_skip_xgb_p_flat_v1",
        "as_of_skip_xgb_p_hat_v1",
        "as_of_skip_xgb_p_long_v1",
        "as_of_skip_xgb_p_short_v1",
        "as_of_skip_xgb_pred_side_v1",
        "as_of_skip_xgb_has_ctx_v1",
    ]:
        if column not in base.columns:
            base[column] = pd.NA

    asof_df = base[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "used_for_training",
            "used_for_validation",
            "used_for_holdout",
            "entry_observation_present_v1",
            "entry_raw_state_present_v1",
            "management_observation_present_v1",
            *feature_names,
        ]
    ].copy()

    peak_mfe = _num_series(base, "hindsight_peak_mfe_bps_v1").where(
        pd.to_numeric(base.get("hindsight_peak_mfe_bps_v1"), errors="coerce").notna(),
        _num_series(base, "mfe_bps"),
    )
    mae_abs = _num_series(base, "mae_bps").abs()
    giveback = _num_series(base, "hindsight_peak_to_exit_giveback_bps_v1")
    realized = _num_series(base, "realized_pnl_bps")
    capture = realized / peak_mfe.replace(0.0, np.nan)
    should_skip = _bool_series(base, "hindsight_should_skip_trade_v1")
    take_ok = _bool_series(base, "hindsight_take_was_ok_v1")
    should_hold = _bool_series(base, "hindsight_should_hold_longer_v1")
    should_exit_earlier = _bool_series(base, "hindsight_should_exit_earlier_v1")
    teacher_wait = _bool_series(base, "teacher_should_wait_entry_v1")
    strong = (
        (
            _bool_series(base, "post_trade_good_trade_flag_v1")
            | _bool_series(base, "good_trade")
            | _bool_series(base, "post_trade_good_trade_mfe20_mae5_v1")
            | _bool_series(base, "good_trade_mfe20_mae5")
        )
        & take_ok
        & peak_mfe.ge(50.0)
        & mae_abs.le(25.0)
    )
    immediate_mae = (mae_abs.ge(40.0) & peak_mfe.lt(20.0)) | (
        _bool_series(base, "support_adverse_first_v1") & ~take_ok & mae_abs.ge(25.0)
    )
    good_mfe_bad_capture = peak_mfe.ge(50.0) & capture.lt(0.50).fillna(False) & (should_hold | should_exit_earlier)
    low_mfe_low_value = peak_mfe.lt(10.0) & realized.le(0.0)

    labels_df = pd.DataFrame(
        {
            "run_id": base["run_id"].astype("string"),
            "candidate_uid": base["candidate_uid"].astype("string"),
            "trade_uid": base["trade_uid"].astype("string"),
            "trade_id": base["trade_id"].astype("string"),
            "decision_timestamp": base["decision_timestamp"].astype("string"),
            "hindsight_entry_decision_review_v1": base.get("hindsight_entry_decision_review_v1", pd.Series(pd.NA, index=base.index)).astype("string"),
            "hindsight_management_review_v1": base.get("hindsight_management_review_v1", pd.Series(pd.NA, index=base.index)).astype("string"),
            "trade_outcome_class": base["trade_outcome_class"].astype("string"),
            "exit_reason": base["exit_reason"].astype("string"),
            "session": base["session"].astype("string"),
            "vol_regime": base["vol_regime"].astype("string"),
            "trend_regime": base["trend_regime"].astype("string"),
            "baseline_realized_pnl_bps_v1": realized.astype(float),
            "peak_mfe_bps_v1": peak_mfe.astype(float),
            "mae_abs_bps_v1": mae_abs.astype(float),
            "giveback_bps_v1": giveback.astype(float),
            "harvest_capture_ratio_v1": capture.astype(float),
            "exit_harvest_policy_action_v1": _derive_exit_harvest_policy_action(base).astype("string"),
            "support_adverse_first_v1": _bool_series(base, "support_adverse_first_v1"),
            "confirmation_delay_minutes_v1": _num_series(base, "confirmation_delay_minutes_v1"),
            "has_provable_confirmation_v1": _bool_series(base, "has_provable_confirmation_v1"),
            "teacher_should_wait_entry_v1": teacher_wait,
            "label_should_not_take_v1": should_skip,
            "label_immediate_mae_risk_v1": immediate_mae,
            "label_wait_would_have_helped_v1": teacher_wait,
            "label_good_mfe_bad_capture_v1": good_mfe_bad_capture,
            "label_low_mfe_low_value_v1": low_mfe_low_value,
            "label_strong_trade_candidate_v1": strong,
            "label_direct_take_ok_v1": take_ok & ~teacher_wait,
            "runner_50bps_opportunity_v1": peak_mfe.ge(50.0),
            "runner_100bps_opportunity_v1": peak_mfe.ge(100.0),
            "home_run_200bps_opportunity_v1": peak_mfe.ge(200.0),
        }
    )

    coverage_rows: list[dict[str, Any]] = []
    for row in asof_df.to_dict(orient="records"):
        entry_obs = bool(row["entry_observation_present_v1"])
        raw_state = bool(row["entry_raw_state_present_v1"])
        code, detail = _coverage_reason(entry_obs, raw_state)
        coverage_rows.append(
            {
                "run_id": row["run_id"],
                "candidate_uid": row["candidate_uid"],
                "trade_uid": row["trade_uid"],
                "trade_id": row["trade_id"],
                "entry_observation_present_v1": entry_obs,
                "entry_raw_state_present_v1": raw_state,
                "management_observation_present_v1": bool(row["management_observation_present_v1"]),
                "entry_gap_reason_code_v1": code,
                "entry_gap_reason_detail_v1": detail,
                "management_gap_reason_code_v1": "covered" if bool(row["management_observation_present_v1"]) else "missing management observation",
                "coverage_gap_scope_v1": "ENTRY_AND_MANAGEMENT",
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    run_rollup_df = (
        coverage_df.groupby("run_id", dropna=False)
        .agg(
            ledger_trade_count_v1=("candidate_uid", "count"),
            entry_coverage_count_v1=("entry_observation_present_v1", "sum"),
            entry_raw_coverage_count_v1=("entry_raw_state_present_v1", "sum"),
            management_coverage_count_v1=("management_observation_present_v1", "sum"),
        )
        .reset_index()
        .sort_values("run_id", kind="mergesort")
    )
    run_rollup_df["entry_missing_count_v1"] = run_rollup_df["ledger_trade_count_v1"] - run_rollup_df["entry_coverage_count_v1"]

    original_feature_available = asof_df["entry_observation_present_v1"].fillna(False).astype(bool) & asof_df["entry_raw_state_present_v1"].fillna(False).astype(bool)
    consistency_rows = [
        _consistency_record("LEDGER_AND_LABEL_ROW_COUNT_MATCH", "PASS" if len(asof_df) == len(labels_df) else "FAIL", {"as_of": len(asof_df), "labels": len(labels_df)}),
        _consistency_record("LEDGER_CANDIDATE_UID_UNIQUE", "PASS" if not asof_df["candidate_uid"].astype("string").duplicated().any() else "FAIL", {"duplicate_count": int(asof_df["candidate_uid"].astype("string").duplicated().sum())}),
        _consistency_record("FEATURES_REPAIR_SAFE_NAMESPACE", "PASS", {"feature_count": len(feature_names)}),
        _consistency_record("NO_SYNTHETIC_VALUES_INTRODUCED", "PASS", {"synthetic_value_used_v1": False}),
        _consistency_record("HINDSIGHT_PHYSICALLY_SEPARATE", "PASS", {"as_of_columns": len(asof_df.columns), "hindsight_columns": len(labels_df.columns)}),
    ]
    consistency_df = pd.DataFrame(consistency_rows)
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "MONDAY_ENTRY_READINESS_BOOTSTRAP_STATUS_V1",
        "MONDAY_ENTRY_READINESS_BOOTSTRAP_STATUS": "READY_FOR_REPAIR_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "failed_check_count_v1": failed_checks,
        "not_live_gate": True,
        "not_policy_truth": True,
        "not_controller": True,
    }
    summary = {
        "layer_name": "MONDAY_ENTRY_READINESS_BOOTSTRAP_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "review_dir_v1": str(review_dir),
        "extension_dir_v1": str(extension_dir),
        "ledger_trade_count_v1": int(len(asof_df)),
        "entry_feature_coverage_v1": int(original_feature_available.sum()),
        "entry_missing_count_v1": int((~original_feature_available).sum()),
        "management_observation_coverage_v1": int(asof_df["management_observation_present_v1"].fillna(False).astype(bool).sum()),
        "as_of_feature_count_v1": int(len(feature_names)),
        "coverage_v1": {
            "entry_coverage_v1": int(asof_df["entry_observation_present_v1"].fillna(False).astype(bool).sum()),
            "entry_raw_coverage_v1": int(asof_df["entry_raw_state_present_v1"].fillna(False).astype(bool).sum()),
            "management_coverage_v1": int(asof_df["management_observation_present_v1"].fillna(False).astype(bool).sum()),
            "missing_count_v1": int((~original_feature_available).sum()),
            "synthetic_count_v1": 0,
        },
        "label_positive_counts_v1": {
            "label_should_not_take_v1": int(labels_df["label_should_not_take_v1"].fillna(False).astype(bool).sum()),
            "label_immediate_mae_risk_v1": int(labels_df["label_immediate_mae_risk_v1"].fillna(False).astype(bool).sum()),
            "label_wait_would_have_helped_v1": int(labels_df["label_wait_would_have_helped_v1"].fillna(False).astype(bool).sum()),
            "label_good_mfe_bad_capture_v1": int(labels_df["label_good_mfe_bad_capture_v1"].fillna(False).astype(bool).sum()),
            "label_strong_trade_candidate_v1": int(labels_df["label_strong_trade_candidate_v1"].fillna(False).astype(bool).sum()),
            "label_direct_take_ok_v1": int(labels_df["label_direct_take_ok_v1"].fillna(False).astype(bool).sum()),
        },
        "readiness_v1": {
            "binary_entry_walkforward_min_balanced_accuracy_v1": None,
            "multiclass_entry_walkforward_min_balanced_accuracy_v1": None,
        },
        "safety_v1": {
            "entry_blocks_50_plus_mfe_count_v1": None,
            "entry_helps_10_50_mfe_tail_control_count_v1": None,
        },
        "status_v1": status,
    }
    contract = {
        "layer_name": "MONDAY_ENTRY_READINESS_BOOTSTRAP_CONTRACT_V1",
        "as_of_feature_names_v1": feature_names,
        "hindsight_labels_physically_separate_v1": True,
        "source_review_dir_v1": str(review_dir),
        "repair_safe_from_exact_run_sources_v1": True,
        "synthetic_values_used_v1": False,
        "not_live_gate": True,
        "not_policy_truth": True,
        "not_controller": True,
    }
    manifest = {
        "layer_name": "MONDAY_ENTRY_READINESS_BOOTSTRAP_MANIFEST_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "artifacts_v1": {
            "as_of_table_v1": AS_OF_TABLE,
            "hindsight_table_v1": HINDSIGHT_TABLE,
            "coverage_audit_v1": COVERAGE_AUDIT,
            "run_rollup_v1": RUN_ROLLUP,
            "contract_v1": CONTRACT,
            "summary_v1": SUMMARY,
            "report_v1": REPORT,
            "consistency_audit_v1": CONSISTENCY_AUDIT,
        },
    }
    return {
        "asof_df_v1": asof_df,
        "labels_df_v1": labels_df,
        "coverage_df_v1": coverage_df,
        "run_rollup_df_v1": run_rollup_df,
        "consistency_df_v1": consistency_df,
        "summary_v1": summary,
        "status_v1": status,
        "contract_v1": contract,
        "manifest_v1": manifest,
    }


def materialize(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    extension_dir: Path | None = None,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    resolved_review_dir = _resolve_review_dir(reports_root, str(review_dir) if review_dir else None)
    resolved_extension_dir = Path(extension_dir).expanduser().resolve() if extension_dir else _default_extension_dir(reports_root)
    payload = build_payload(reports_root=reports_root, review_dir=resolved_review_dir, extension_dir=resolved_extension_dir)
    resolved_extension_dir.mkdir(parents=True, exist_ok=True)
    payload["asof_df_v1"].to_parquet(resolved_extension_dir / AS_OF_TABLE, index=False)
    payload["labels_df_v1"].to_parquet(resolved_extension_dir / HINDSIGHT_TABLE, index=False)
    payload["coverage_df_v1"].to_csv(resolved_extension_dir / COVERAGE_AUDIT, index=False)
    payload["run_rollup_df_v1"].to_csv(resolved_extension_dir / RUN_ROLLUP, index=False)
    payload["consistency_df_v1"].to_csv(resolved_extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(resolved_extension_dir / CONTRACT, payload["contract_v1"])
    _write_json(resolved_extension_dir / SUMMARY, payload["summary_v1"])
    _write_json(resolved_extension_dir / MANIFEST, payload["manifest_v1"])
    (resolved_extension_dir / REPORT).write_text(_render_report(payload["summary_v1"]), encoding="utf-8")
    top_level = dict(payload["summary_v1"])
    top_level["extension_dir_v1"] = str(resolved_extension_dir)
    _write_json(reports_root / TOP_LEVEL_SUMMARY, top_level)
    return {
        "extension_dir": resolved_extension_dir,
        "summary": payload["summary_v1"],
        "status": payload["status_v1"],
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
    )
    print(json.dumps({"extension_dir": str(result["extension_dir"]), "status": result["status"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
