from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1"
MONDAY_TRUTH_GLOB = "MONDAY_R6_CANONICAL_TRUTH_V1_*"
WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

AS_OF_TABLE = "monday_r6_entry_runner_first_as_of_feature_table_v1.parquet"
HINDSIGHT_TABLE = "monday_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
POLICY_VIEW = "monday_r6_policy_prediction_view_scaffold_v1.parquet"
FEATURE_SOURCE_AUDIT = "monday_r6_rehydration_feature_source_audit_v1.csv"
BLOCKED_FIELDS = "monday_r6_rehydration_blocked_fields_v1.csv"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"
SUMMARY = "summary_v1.json"
MANIFEST = "manifest_v1.json"
REPORT = "report_v1.md"

OUTPUT_FILES = {
    "as_of_table": AS_OF_TABLE,
    "hindsight_table": HINDSIGHT_TABLE,
    "policy_view": POLICY_VIEW,
    "feature_source_audit": FEATURE_SOURCE_AUDIT,
    "blocked_fields": BLOCKED_FIELDS,
    "audit": CONSISTENCY_AUDIT,
    "summary": SUMMARY,
    "manifest": MANIFEST,
    "report": REPORT,
}

ID_COLS = ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]
SCORE_OR_POLICY_COLS = {
    "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "pred__entry_r5_2_runner_protector__prob_true_v1",
    "blocker_score_v1",
    "runner_protector_score_v1",
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_strong_trade_candidate__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
    "pred__entry_r5_take_was_ok__prob_true_v1",
    "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1",
    "pred__entry_r5_wait_or_delay_advisory__prob_true_v1",
}

REPLAY_DIRECT_RENAME = {
    "spread_bps": "as_of_skip_replay_spread_bps_v1",
    "_v1_body_share_1": "as_of_skip_replay_body_share_v1",
    "_v1_clv": "as_of_skip_replay_clv_v1",
    "_v1_cost_bps_dyn": "as_of_skip_replay_cost_bps_dyn_v1",
    "_v1_cost_bps_est": "as_of_skip_replay_cost_bps_est_v1",
    "_v1_bb_squeeze_20_2": "as_of_skip_replay_bb_squeeze_20_2_v1",
    "_v1_bb_bandwidth_delta_10": "as_of_skip_replay_bb_bandwidth_delta_10_v1",
    "_v1_kama_slope_30": "as_of_skip_replay_kama_slope_30_v1",
    "micro_momentum_3": "as_of_skip_replay_micro_momentum_3_v1",
    "micro_momentum_5": "as_of_skip_replay_micro_momentum_5_v1",
    "micro_acceleration": "as_of_skip_replay_micro_acceleration_v1",
    "wick_ratio": "as_of_skip_replay_wick_ratio_v1",
    "distance_ema_fast": "as_of_skip_replay_distance_ema_fast_v1",
    "dist_last_swing_high_atr": "as_of_skip_replay_dist_last_swing_high_atr_v1",
    "dist_last_swing_low_atr": "as_of_skip_replay_dist_last_swing_low_atr_v1",
    "bars_since_swing_high": "as_of_skip_replay_bars_since_swing_high_v1",
    "bars_since_swing_low": "as_of_skip_replay_bars_since_swing_low_v1",
    "retracement_from_last_impulse": "as_of_skip_replay_retracement_from_last_impulse_v1",
    "minutes_since_session_open": "as_of_skip_replay_minutes_since_session_open_v1",
    "minutes_to_next_session_boundary": "as_of_skip_replay_minutes_to_next_session_boundary_v1",
    "session_change_flag": "as_of_skip_replay_session_change_flag_v1",
    "session_tradable": "as_of_skip_replay_session_tradable_v1",
    "H1_range_compression_ratio": "as_of_skip_replay_h1_range_compression_ratio_v1",
    "M15_range_compression_ratio": "as_of_skip_replay_m15_range_compression_ratio_v1",
    "D1_atr_percentile_252": "as_of_skip_replay_d1_atr_percentile_252_v1",
    "D1_dist_from_ema200_atr": "as_of_skip_replay_d1_dist_from_ema200_atr_v1",
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
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_monday_truth(reports_root: Path) -> Path:
    dirs = sorted(path for path in reports_root.glob(MONDAY_TRUTH_GLOB) if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No Monday truth package found under {reports_root}")
    return dirs[-1]


def _bool_col(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
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


def _candidate_source(truth: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=truth.index)
    out["as_of_hour_utc_v1"] = _num(truth, "entry_candidate_hour_utc_v1")
    out["as_of_weekday_utc_v1"] = _num(truth, "entry_candidate_weekday_utc_v1").astype("Int64")
    out["as_of_session_v1"] = truth.get("entry_candidate_session_v1", pd.Series(pd.NA, index=truth.index)).astype("string")
    out["as_of_side_v1"] = truth.get("entry_candidate_side_v1", pd.Series(pd.NA, index=truth.index)).astype("string")
    mapping = {
        "as_of_atr_bps_v1": "entry_candidate_atr_bps_v1",
        "as_of_candidate_entry_spread_bps_v1": "entry_candidate_entry_spread_bps_v1",
        "as_of_candidate_uncertainty_score_v1": "entry_candidate_uncertainty_score_v1",
        "as_of_candidate_tradable_prob_v1": "entry_candidate_tradable_prob_v1",
        "as_of_candidate_mfe_first_n_pred_v1": "entry_candidate_mfe_first_n_pred_v1",
        "as_of_candidate_trend_regime_v1": "entry_candidate_trend_regime_v1",
        "as_of_candidate_vol_regime_v1": "entry_candidate_vol_regime_v1",
        "as_of_entry_candidate_margin_v1": "entry_candidate_margin_v1",
        "as_of_entry_candidate_path_quality_pred_v1": "entry_candidate_path_quality_pred_v1",
        "as_of_skip_candidate_entry_spread_bps_v1": "entry_candidate_entry_spread_bps_v1",
        "as_of_skip_candidate_margin_v1": "entry_candidate_margin_v1",
        "as_of_skip_candidate_p_flat_v1": "entry_candidate_p_flat_v1",
        "as_of_skip_candidate_p_hat_v1": "entry_candidate_p_hat_v1",
        "as_of_skip_candidate_p_long_v1": "entry_candidate_p_long_v1",
        "as_of_skip_candidate_p_short_v1": "entry_candidate_p_short_v1",
        "as_of_skip_candidate_path_quality_pred_v1": "entry_candidate_path_quality_pred_v1",
        "as_of_skip_xgb_p_flat_v1": "entry_xgb_p_flat_v1",
        "as_of_skip_xgb_p_hat_v1": "entry_xgb_p_hat_v1",
        "as_of_skip_xgb_p_long_v1": "entry_xgb_p_long_v1",
        "as_of_skip_xgb_p_short_v1": "entry_xgb_p_short_v1",
        "as_of_skip_xgb_pred_side_v1": "entry_xgb_pred_side_v1",
        "as_of_skip_xgb_has_ctx_v1": "entry_xgb_has_ctx_v1",
    }
    for target, source in mapping.items():
        if source in truth.columns:
            out[target] = truth[source]
        else:
            out[target] = pd.NA
    return out


def _market_pressure_fields(run_bar: pd.DataFrame, windows: tuple[int, ...] = (15, 60, 240)) -> pd.DataFrame:
    close = pd.to_numeric(run_bar.get("close"), errors="coerce")
    high = pd.to_numeric(run_bar.get("high"), errors="coerce")
    low = pd.to_numeric(run_bar.get("low"), errors="coerce")
    base = close.replace(0, np.nan)
    payload: dict[str, Any] = {}
    for window in windows:
        horizon = int(window)
        min_obs = min(3, horizon)
        rolling_high = high.rolling(horizon, min_periods=min_obs).max()
        rolling_low = low.rolling(horizon, min_periods=min_obs).min()
        rolling_range = (rolling_high - rolling_low).astype(float)
        up_move = (close - rolling_low) / base * 1e4
        down_move = (rolling_high - close) / base * 1e4
        payload[f"as_of_skip_replay_window_up_move_{horizon}_bps_v1"] = up_move
        payload[f"as_of_skip_replay_window_down_move_{horizon}_bps_v1"] = down_move
        payload[f"as_of_skip_replay_window_range_{horizon}_bps_v1"] = rolling_range / base * 1e4
        payload[f"as_of_skip_replay_window_directional_imbalance_{horizon}_bps_v1"] = up_move - down_move
        payload[f"as_of_skip_replay_window_close_in_range_{horizon}_v1"] = (
            close - rolling_low
        ) / rolling_range.replace(0, np.nan)
    return pd.DataFrame(payload, index=run_bar.index)


def _replay_source(bar: pd.DataFrame) -> pd.DataFrame:
    if bar.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for run_id, run_bar in bar.groupby(bar["run_id"].astype("string"), dropna=False):
        run = run_bar.copy()
        run["_bar_ts"] = pd.to_datetime(run["time"], utc=True, errors="coerce")
        run = run.sort_values("_bar_ts", kind="mergesort").reset_index(drop=True)
        close = pd.to_numeric(run["close"], errors="coerce")
        open_ = pd.to_numeric(run["open"], errors="coerce")
        high = pd.to_numeric(run["high"], errors="coerce")
        low = pd.to_numeric(run["low"], errors="coerce")
        range_abs = (high - low).astype(float)
        range_bps = range_abs / close.replace(0, np.nan) * 1e4
        ret1 = close.pct_change() * 1e4
        spread = pd.to_numeric(run.get("spread_bps"), errors="coerce")
        spread_median_5 = spread.rolling(5, min_periods=3).median()
        range_mean_5 = range_bps.rolling(5, min_periods=3).mean()
        out = pd.DataFrame({"run_id": str(run_id), "decision_timestamp": run["_bar_ts"]})
        for source, target in REPLAY_DIRECT_RENAME.items():
            out[target] = run[source] if source in run.columns else pd.NA
        out["as_of_skip_replay_range_bps_v1"] = range_bps
        out["as_of_skip_replay_body_bps_v1"] = (close - open_).abs() / close.replace(0, np.nan) * 1e4
        out["as_of_skip_replay_upper_wick_share_v1"] = (
            high - pd.concat([open_, close], axis=1).max(axis=1)
        ) / range_abs.replace(0, np.nan)
        out["as_of_skip_replay_lower_wick_share_v1"] = (
            pd.concat([open_, close], axis=1).min(axis=1) - low
        ) / range_abs.replace(0, np.nan)
        out["as_of_skip_replay_close_in_bar_v1"] = (close - low) / range_abs.replace(0, np.nan)
        out["as_of_skip_replay_window_ret_1_bps_v1"] = ret1
        out["as_of_skip_replay_window_ret_3_bps_v1"] = close.pct_change(3) * 1e4
        out["as_of_skip_replay_window_ret_5_bps_v1"] = close.pct_change(5) * 1e4
        out["as_of_skip_replay_window_realized_vol_3_bps_v1"] = ret1.rolling(3, min_periods=2).std()
        out["as_of_skip_replay_window_realized_vol_5_bps_v1"] = ret1.rolling(5, min_periods=3).std()
        out["as_of_skip_replay_window_spread_minus_median_5_bps_v1"] = spread - spread_median_5
        out["as_of_skip_replay_window_spread_ratio_median_5_v1"] = spread / spread_median_5.replace(0, np.nan)
        out["as_of_skip_replay_window_range_minus_mean_5_bps_v1"] = range_bps - range_mean_5
        out["as_of_skip_replay_window_range_ratio_mean_5_v1"] = range_bps / range_mean_5.replace(0, np.nan)
        pressure = _market_pressure_fields(run)
        for column in pressure.columns:
            out[column] = pressure[column]
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _run_order_map(run_ids: pd.Series, batch_weeks: int = 15) -> dict[str, str]:
    ordered = sorted(run_ids.astype("string").dropna().unique().tolist())
    out: dict[str, str] = {}
    for idx, run_id in enumerate(ordered):
        out[run_id] = f"BATCH_{idx // batch_weeks + 1:02d}"
    return out


def _build_asof(
    truth: pd.DataFrame,
    bar: pd.DataFrame,
    expected_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = truth.copy()
    base["decision_timestamp"] = pd.to_datetime(base["decision_timestamp_v1"], utc=True, errors="coerce")
    asof = base[["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]].copy()
    # Keep splits closed for safety until the exact Monday R6 training plan is explicitly rebuilt.
    asof["used_for_training"] = False
    asof["used_for_validation"] = False
    asof["used_for_holdout"] = False
    asof = pd.concat([asof, _candidate_source(base)], axis=1)

    replay = _replay_source(bar)
    if not replay.empty:
        asof = asof.merge(replay, on=["run_id", "decision_timestamp"], how="left", validate="many_to_one")

    asof["entry_observation_present_v1"] = asof["candidate_uid"].notna()
    asof["entry_raw_state_present_v1"] = asof[[c for c in asof.columns if c.startswith("as_of_skip_replay_")]].notna().any(axis=1)
    asof["management_observation_present_v1"] = base["journal_exit_reason_v1"].notna() if "journal_exit_reason_v1" in base.columns else True
    asof["entry_coverage_original_entry_observation_present_v1"] = asof["entry_observation_present_v1"]
    asof["entry_coverage_original_entry_raw_state_present_v1"] = asof["entry_raw_state_present_v1"]
    asof["entry_coverage_repair_applied_v1"] = False
    asof["entry_coverage_repair_source_v1"] = "MONDAY_REPLAY_EXACT_ENTRY_SOURCE_NO_REPAIR"
    asof["r6_as_of_feature_contract_v1"] = "MONDAY_R6_REHYDRATED_FROM_WEDNESDAY_CONTRACT|SPLITS_CLOSED|SCORE_COLUMNS_BLOCKED_UNTIL_CANONICAL_SOURCE"

    blocked_rows: list[dict[str, Any]] = []
    for column in expected_columns:
        if column in SCORE_OR_POLICY_COLS:
            asof[column] = pd.Series(pd.NA, index=asof.index, dtype="Float64")
            blocked_rows.append(
                {
                    "field_v1": column,
                    "surface_v1": "AS_OF",
                    "blocked_reason_v1": "CANONICAL_WEDNESDAY_R5_R5_2_SCORE_SOURCE_NOT_AVAILABLE_LOCALLY",
                    "status_v1": "BLOCKED_NOT_FILLED",
                }
            )
        elif column not in asof.columns:
            asof[column] = pd.NA
            blocked_rows.append(
                {
                    "field_v1": column,
                    "surface_v1": "AS_OF",
                    "blocked_reason_v1": "NO_MONDAY_SOURCE_OR_DERIVATION_IMPLEMENTED",
                    "status_v1": "MISSING_FILLED_NULL",
                }
            )
    asof = asof[expected_columns].copy()
    feature_audit = []
    for column in expected_columns:
        if column in ID_COLS:
            family = "identity"
        elif column in SCORE_OR_POLICY_COLS:
            family = "blocked_canonical_score_source_missing"
        elif column.startswith("as_of_skip_replay_"):
            family = "rehydrated_from_monday_bar_surface"
        elif column.startswith("as_of_skip_xgb_"):
            family = "rehydrated_from_monday_xgb_exact_entry"
        elif column.startswith("as_of_skip_candidate_") or column.startswith("as_of_candidate_") or column.startswith("as_of_entry_candidate_"):
            family = "rehydrated_from_monday_candidate_surface"
        elif column.startswith("entry_coverage_") or column in {"entry_observation_present_v1", "entry_raw_state_present_v1", "management_observation_present_v1"}:
            family = "derived_monday_coverage_metadata"
        else:
            family = "derived_or_contract_metadata"
        feature_audit.append(
            {
                "field_v1": column,
                "surface_v1": "AS_OF",
                "source_family_v1": family,
                "non_null_rows_v1": int(asof[column].notna().sum()),
                "row_count_v1": int(len(asof)),
                "null_rate_v1": float(asof[column].isna().mean()) if len(asof) else None,
            }
        )
    return asof, pd.DataFrame(feature_audit), pd.DataFrame(blocked_rows)


def _build_hindsight(truth: pd.DataFrame, expected_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = truth[["candidate_uid", "run_id", "trade_uid", "trade_id"]].copy()
    out["decision_timestamp"] = pd.to_datetime(truth["decision_timestamp_v1"], utc=True, errors="coerce")
    pnl = _num(truth, "canonical_pnl_bps_v1")
    mfe = _num(truth, "canonical_mfe_bps_v1")
    mae_abs = _num(truth, "canonical_mae_bps_v1").abs()
    giveback = (mfe - pnl).clip(lower=0.0)
    if "journal_dd_from_mfe_bps_exit_v1" in truth.columns:
        giveback = _num(truth, "journal_dd_from_mfe_bps_exit_v1").where(
            _num(truth, "journal_dd_from_mfe_bps_exit_v1").notna(), giveback
        )
    out["baseline_realized_pnl_bps_v1"] = pnl
    out["peak_mfe_bps_v1"] = mfe
    out["mae_abs_bps_v1"] = mae_abs
    out["giveback_bps_v1"] = giveback
    should = ((pnl <= 0.0) & (mfe < 50.0)) | ((mae_abs >= 40.0) & (pnl <= 0.0)) | _bool_col(
        truth, "truth_cata_or_friday_flat_damage_v1"
    )
    take_ok = (~should) & (pnl > 0.0) & (mfe >= 20.0) & (mae_abs <= 50.0)
    strong = (~should) & (mfe >= 50.0) & (mae_abs <= 25.0) & (pnl > 0.0)
    fifty = mfe >= 50.0
    hundred = mfe >= 100.0
    two_hundred = mfe >= 200.0
    tail_10_50 = mfe.between(10.0, 50.0, inclusive="left") & ((pnl <= 0.0) | should)
    out["hindsight_entry_decision_review_v1"] = np.where(take_ok, "TAKE_WAS_OK", np.where(should, "SHOULD_NOT_TAKE", "NEUTRAL_REVIEW_PROXY"))
    out["hindsight_management_review_v1"] = np.where(
        _bool_col(truth, "truth_exit_too_early_regret_replay_end_v1"),
        "EXIT_TOO_EARLY_REGRET_PROXY",
        "NO_REPLAY_END_REGRET_PROXY",
    )
    out["r6_label_runner_50_mfe_v1"] = take_ok & fifty
    out["r6_label_runner_100_mfe_v1"] = take_ok & hundred
    out["r6_label_runner_200_mfe_v1"] = take_ok & two_hundred
    out["r6_label_repaired_165_like_runner_v1"] = False
    out["r6_label_strong_low_mae_runner_v1"] = take_ok & strong
    out["r6_label_high_mfe_low_giveback_v1"] = take_ok & fifty & ((giveback <= 25.0) | (giveback <= mfe * 0.25))
    out["r6_label_runner_near_miss_v1"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["r6_label_runner_protect_v1"] = (
        out["r6_label_runner_50_mfe_v1"]
        | out["r6_label_runner_100_mfe_v1"]
        | out["r6_label_runner_200_mfe_v1"]
        | out["r6_label_repaired_165_like_runner_v1"]
        | out["r6_label_strong_low_mae_runner_v1"]
        | out["r6_label_high_mfe_low_giveback_v1"]
    )
    out["r6_label_missed_should_not_take_v1"] = should
    out["r6_label_risky_allow_v1"] = should & ((mae_abs >= 40.0) | (pnl <= -25.0))
    out["r6_label_high_mae_low_mfe_v1"] = should & (mae_abs >= 40.0) & (mfe < 50.0)
    out["r6_label_low_mfe_low_value_v1"] = should & (mfe < 10.0) & (pnl <= 0.0)
    out["r6_label_early_adverse_excursion_v1"] = should & (mae_abs >= 40.0) & (mfe < 50.0)
    session = truth.get("canonical_session_v1", pd.Series("", index=truth.index)).astype("string").str.upper()
    vol = truth.get("entry_candidate_vol_regime_v1", pd.Series("", index=truth.index)).astype("string").str.upper()
    trend = truth.get("entry_candidate_trend_regime_v1", pd.Series("", index=truth.index)).astype("string").str.upper()
    out["r6_label_bad_trade_overlap_extreme_vol_v1"] = should & session.eq("OVERLAP") & vol.eq("EXTREME")
    batch_map = _run_order_map(truth["run_id"])
    batch_scope = truth["run_id"].astype("string").map(batch_map).fillna("BATCH_UNKNOWN")
    out["r6_label_batch04_blindspot_v1"] = should & batch_scope.eq("BATCH_04")
    out["r6_label_trend_neutral_extreme_vol_risk_v1"] = should & trend.eq("TREND_NEUTRAL") & vol.eq("EXTREME")
    out["r6_label_bad_risk_v1"] = should
    out["r6_label_tail_control_10_50_v1"] = tail_10_50
    out["r6_hindsight_contract_v1"] = "MONDAY_R6_REHYDRATED_HINDSIGHT_PROXY|NOT_WEDNESDAY_EXACT_LABEL_SOURCE|NOT_POLICY_TRUTH"

    blocked_rows = [
        {
            "field_v1": "r6_label_runner_near_miss_v1",
            "surface_v1": "HINDSIGHT",
            "blocked_reason_v1": "EXACT_RUNNER_NEAR_MISS_REQUIRES_CANONICAL_R5_2_SCORE_AND_SELECTED_POLICY_CONTEXT",
            "status_v1": "NULL_NOT_EXACT",
        },
        {
            "field_v1": "hindsight_entry_decision_review_v1",
            "surface_v1": "HINDSIGHT",
            "blocked_reason_v1": "SOURCE_REVIEW_LABEL_NOT_AVAILABLE_REBUILT_AS_PROXY_FROM_MONDAY_TRUTH",
            "status_v1": "PROXY_NOT_EXACT",
        },
        {
            "field_v1": "hindsight_management_review_v1",
            "surface_v1": "HINDSIGHT",
            "blocked_reason_v1": "SOURCE_MANAGEMENT_REVIEW_NOT_AVAILABLE_REBUILT_AS_PROXY_FROM_EXIT_REGRET",
            "status_v1": "PROXY_NOT_EXACT",
        },
    ]
    for column in expected_columns:
        if column not in out.columns:
            out[column] = pd.NA
            blocked_rows.append(
                {
                    "field_v1": column,
                    "surface_v1": "HINDSIGHT",
                    "blocked_reason_v1": "NO_MONDAY_SOURCE_OR_DERIVATION_IMPLEMENTED",
                    "status_v1": "MISSING_FILLED_NULL",
                }
            )
    out = out[expected_columns].copy()
    audit = [
        {
            "field_v1": column,
            "surface_v1": "HINDSIGHT",
            "source_family_v1": "derived_from_monday_realized_trade_truth"
            if column not in ID_COLS
            else "identity",
            "non_null_rows_v1": int(out[column].notna().sum()),
            "row_count_v1": int(len(out)),
            "null_rate_v1": float(out[column].isna().mean()) if len(out) else None,
        }
        for column in expected_columns
    ]
    return out, pd.DataFrame(audit), pd.DataFrame(blocked_rows)


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("AS_OF_SCHEMA_109_MATERIALIZED", "PASS" if summary["as_of_column_count_v1"] == 109 else "FAIL", summary["as_of_column_count_v1"]),
            row("HINDSIGHT_SCHEMA_30_MATERIALIZED", "PASS" if summary["hindsight_column_count_v1"] == 30 else "FAIL", summary["hindsight_column_count_v1"]),
            row("MONDAY_ROWS_NOT_1689_OR_1852", "PASS" if summary["row_count_v1"] not in (1689, 1852) else "FAIL", summary["row_count_v1"]),
            row("NO_TRAINING_STARTED", "PASS", summary["training_started_v1"]),
            row("CANONICAL_SCORE_SOURCES_AVAILABLE", "PASS" if summary["blocked_score_column_count_v1"] == 0 else "FAIL", summary["blocked_score_column_count_v1"]),
            row("HINDSIGHT_EXACT_LABEL_SOURCE_AVAILABLE", "PASS" if summary["hindsight_proxy_column_count_v1"] == 0 else "FAIL", summary["hindsight_proxy_column_count_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Rehydrated Wednesday Contract V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Rows: `{summary['row_count_v1']}`",
            f"- AS_OF columns: `{summary['as_of_column_count_v1']}`",
            f"- Hindsight columns: `{summary['hindsight_column_count_v1']}`",
            f"- Blocked score columns: `{summary['blocked_score_column_count_v1']}`",
            f"- Proxy hindsight fields: `{summary['hindsight_proxy_column_count_v1']}`",
            f"- Training started: `{summary['training_started_v1']}`",
            "",
            "The table shape is restored, but canonical training remains blocked until score/model and exact label sources are restored or explicitly rebuilt.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    monday_truth_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    monday_truth_dir = monday_truth_dir.expanduser().resolve() if monday_truth_dir else _latest_monday_truth(reports_root)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    wednesday_summary = _read_json(freeze_dir / WEDNESDAY_SUMMARY)
    wednesday_manifest = _read_json(freeze_dir / WEDNESDAY_MANIFEST)
    asof_cols = [str(row["name_v1"]) for row in wednesday_manifest["as_of_schema_v1"]["columns_v1"]]
    hindsight_cols = [str(row["name_v1"]) for row in wednesday_manifest["hindsight_schema_v1"]["columns_v1"]]

    truth = pd.read_parquet(monday_truth_dir / "monday_r6_trade_truth_v1.parquet")
    bar = pd.read_parquet(monday_truth_dir / "monday_r6_bar_feature_surface_v1.parquet")
    truth = truth.sort_values(["run_id", "canonical_entry_ts_utc_v1", "candidate_uid"], kind="mergesort").reset_index(drop=True)

    asof, asof_audit, asof_blocked = _build_asof(truth, bar, asof_cols)
    hindsight, hindsight_audit, hindsight_blocked = _build_hindsight(truth, hindsight_cols)
    blocked = pd.concat([asof_blocked, hindsight_blocked], ignore_index=True, sort=False)
    source_audit = pd.concat([asof_audit, hindsight_audit], ignore_index=True, sort=False)

    policy_view_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "r6_label_runner_50_mfe_v1",
        "r6_label_runner_100_mfe_v1",
        "r6_label_runner_200_mfe_v1",
        "r6_label_bad_risk_v1",
        "r6_label_tail_control_10_50_v1",
    ]
    policy_view = hindsight[[column for column in policy_view_cols if column in hindsight.columns]].copy()
    for column in (wednesday_manifest.get("score_head_names_v1") or {}).values():
        policy_view[str(column)] = pd.NA

    blocked_score_count = int(asof_blocked["field_v1"].isin(SCORE_OR_POLICY_COLS).sum()) if not asof_blocked.empty else 0
    hindsight_proxy_count = int(hindsight_blocked["status_v1"].astype("string").str.contains("PROXY|NULL_NOT_EXACT", regex=True).sum()) if not hindsight_blocked.empty else 0
    if blocked_score_count or hindsight_proxy_count:
        decision = "MONDAY_R6_WEDNESDAY_CONTRACT_SHAPE_REHYDRATED_BUT_NOT_TRAINING_READY"
        next_action = "RESTORE_OR_REBUILD_CANONICAL_SCORE_AND_EXACT_LABEL_SOURCES"
    else:
        decision = "MONDAY_R6_WEDNESDAY_CONTRACT_REHYDRATED_AND_READY_FOR_EXPLICIT_TRAINING"
        next_action = "RUN_MONDAY_R6_TRAINING_WITH_EXPLICIT_FLAG"
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "monday_truth_dir_v1": str(monday_truth_dir),
        "wednesday_freeze_id_v1": wednesday_summary.get("freeze_id_v1"),
        "wednesday_candidate_id_v1": wednesday_summary.get("selected_candidate_id_v1"),
        "row_count_v1": int(len(asof)),
        "as_of_column_count_v1": int(asof.shape[1]),
        "hindsight_column_count_v1": int(hindsight.shape[1]),
        "policy_view_column_count_v1": int(policy_view.shape[1]),
        "blocked_field_count_v1": int(len(blocked)),
        "blocked_score_column_count_v1": blocked_score_count,
        "hindsight_proxy_column_count_v1": hindsight_proxy_count,
        "training_started_v1": False,
        "decision_v1": decision,
        "next_action_v1": next_action,
        "blocked_action_v1": "DO_NOT_TRAIN_UNTIL_CANONICAL_SCORE_AND_EXACT_LABEL_SOURCES_PASS",
    }
    audit = _audit(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "training_started_v1": False,
    }

    asof.to_parquet(output_dir / AS_OF_TABLE, index=False)
    hindsight.to_parquet(output_dir / HINDSIGHT_TABLE, index=False)
    policy_view.to_parquet(output_dir / POLICY_VIEW, index=False)
    source_audit.to_csv(output_dir / FEATURE_SOURCE_AUDIT, index=False)
    blocked.to_csv(output_dir / BLOCKED_FIELDS, index=False)
    audit.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(output_dir / SUMMARY, summary)
    _write_json(output_dir / MANIFEST, manifest)
    (output_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--monday-truth-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(reports_root=args.reports_root, monday_truth_dir=args.monday_truth_dir, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
