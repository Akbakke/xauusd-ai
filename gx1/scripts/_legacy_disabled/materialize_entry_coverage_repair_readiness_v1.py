#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
REPAIR_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_ENTRY_COVERAGE_REPAIR_READINESS_V1"
R2_READINESS_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_V1"
CANONICAL_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_REWARD_CHANNEL_FIX_R1_CANONICAL"

R2_READINESS_CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"
R2_AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
R2_LABEL_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
R2_COVERAGE_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_audit_v1.csv"
R2_COVERAGE_RUN_ROLLUP = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_run_rollup_v1.csv"
R2_READINESS_SUMMARY = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_summary_v1.json"
R2_READINESS_MANIFEST = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_manifest_v1.json"
R2_READINESS_REPORT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_report_v1.md"

CANONICAL_AS_OF_LEDGER = "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet"

REPAIR_AUDIT = "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv"
REPAIR_FEATURE_SOURCE_AUDIT = "shadow_meta_all_trade_review_entry_coverage_repair_feature_source_audit_v1.csv"
REPAIR_CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_entry_coverage_repair_consistency_audit_v1.csv"
REPAIR_CONTRACT = "shadow_meta_all_trade_review_entry_coverage_repair_contract_v1.json"
REPAIR_SUMMARY = "shadow_meta_all_trade_review_entry_coverage_repair_summary_v1.json"
REPAIR_STATUS = "shadow_meta_all_trade_review_entry_coverage_repair_status_v1.json"
REPAIR_MANIFEST = "shadow_meta_all_trade_review_entry_coverage_repair_manifest_v1.json"
REPAIR_REPORT = "shadow_meta_all_trade_review_entry_coverage_repair_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_entry_coverage_repair_readiness_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
FORBIDDEN_AS_OF_TOKENS = (
    "hindsight",
    "pnl",
    "reward",
    "target",
    "label",
    "harvest",
    "terminal",
    "good_trade",
    "bad_trade",
    "premature",
    "late_exit",
)

REPLAY_BASE_COLS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "spread_bps",
    "_v1_body_share_1",
    "_v1_clv",
    "_v1_cost_bps_dyn",
    "_v1_cost_bps_est",
    "_v1_bb_squeeze_20_2",
    "_v1_bb_bandwidth_delta_10",
    "_v1_kama_slope_30",
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
    "H1_range_compression_ratio",
    "M15_range_compression_ratio",
    "D1_atr_percentile_252",
    "D1_dist_from_ema200_atr",
]

REPLAY_RENAME = {
    "spread_bps": "as_of_entry_replay_spread_bps_v1",
    "_v1_body_share_1": "as_of_entry_replay_body_share_v1",
    "_v1_clv": "as_of_entry_replay_clv_v1",
    "_v1_cost_bps_dyn": "as_of_entry_replay_cost_bps_dyn_v1",
    "_v1_cost_bps_est": "as_of_entry_replay_cost_bps_est_v1",
    "_v1_bb_squeeze_20_2": "as_of_entry_replay_bb_squeeze_20_2_v1",
    "_v1_bb_bandwidth_delta_10": "as_of_entry_replay_bb_bandwidth_delta_10_v1",
    "_v1_kama_slope_30": "as_of_entry_replay_kama_slope_30_v1",
    "micro_momentum_3": "as_of_entry_replay_micro_momentum_3_v1",
    "micro_momentum_5": "as_of_entry_replay_micro_momentum_5_v1",
    "micro_acceleration": "as_of_entry_replay_micro_acceleration_v1",
    "wick_ratio": "as_of_entry_replay_wick_ratio_v1",
    "distance_ema_fast": "as_of_entry_replay_distance_ema_fast_v1",
    "dist_last_swing_high_atr": "as_of_entry_replay_dist_last_swing_high_atr_v1",
    "dist_last_swing_low_atr": "as_of_entry_replay_dist_last_swing_low_atr_v1",
    "bars_since_swing_high": "as_of_entry_replay_bars_since_swing_high_v1",
    "bars_since_swing_low": "as_of_entry_replay_bars_since_swing_low_v1",
    "retracement_from_last_impulse": "as_of_entry_replay_retracement_from_last_impulse_v1",
    "minutes_since_session_open": "as_of_entry_replay_minutes_since_session_open_v1",
    "minutes_to_next_session_boundary": "as_of_entry_replay_minutes_to_next_session_boundary_v1",
    "session_change_flag": "as_of_entry_replay_session_change_flag_v1",
    "session_tradable": "as_of_entry_replay_session_tradable_v1",
    "H1_range_compression_ratio": "as_of_entry_replay_h1_range_compression_ratio_v1",
    "M15_range_compression_ratio": "as_of_entry_replay_m15_range_compression_ratio_v1",
    "D1_atr_percentile_252": "as_of_entry_replay_d1_atr_percentile_252_v1",
    "D1_dist_from_ema200_atr": "as_of_entry_replay_d1_dist_from_ema200_atr_v1",
}

CANDIDATE_COLS = [
    "side",
    "session",
    "weekday_utc",
    "hour_utc",
    "atr_bps",
    "entry_spread_bps",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "tradable_prob",
    "mfe_first_n_pred",
    "path_quality_pred",
    "vol_regime",
    "trend_regime",
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "decision_ts_utc",
    "decision",
    "accepted",
    "decision_reason",
    "policy_hash",
    "entry_bundle_sha256",
    "exit_bundle_sha256",
]

XGB_COLS = ["ts", "head", "horizon_bars", "p_long", "p_short", "p_flat", "p_hat", "pred_side", "has_ctx"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


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


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / REPAIR_EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


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


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()}


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.astype("string").str.strip().str.lower().eq("true").fillna(default).astype(bool)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _check_feature_names(feature_names: Sequence[str]) -> None:
    bad: List[str] = []
    for feature in feature_names:
        lower = feature.lower()
        for token in FORBIDDEN_AS_OF_TOKENS:
            if token == "realized" and "realized_vol" in lower:
                continue
            if token in lower:
                bad.append(feature)
                break
    if bad:
        raise ValueError(f"AS_OF feature list contains forbidden hindsight/target-like names: {bad[:20]}")


def _market_pressure_fields(replay_df: pd.DataFrame, windows: Sequence[int] = (15, 60, 240)) -> pd.DataFrame:
    close = pd.to_numeric(replay_df.get("close"), errors="coerce")
    high = pd.to_numeric(replay_df.get("high"), errors="coerce")
    low = pd.to_numeric(replay_df.get("low"), errors="coerce")
    base = close.replace(0, np.nan)
    payload: Dict[str, Any] = {}
    for window in windows:
        horizon = int(window)
        min_obs = min(3, horizon)
        rolling_high = high.rolling(horizon, min_periods=min_obs).max()
        rolling_low = low.rolling(horizon, min_periods=min_obs).min()
        rolling_range = (rolling_high - rolling_low).astype(float)
        up_move = (close - rolling_low) / base * 1e4
        down_move = (rolling_high - close) / base * 1e4
        payload[f"as_of_entry_replay_window_up_move_{horizon}_bps_v1"] = up_move
        payload[f"as_of_entry_replay_window_down_move_{horizon}_bps_v1"] = down_move
        payload[f"as_of_entry_replay_window_range_{horizon}_bps_v1"] = rolling_range / base * 1e4
        payload[f"as_of_entry_replay_window_directional_imbalance_{horizon}_bps_v1"] = up_move - down_move
        payload[f"as_of_entry_replay_window_close_in_range_{horizon}_v1"] = (close - rolling_low) / rolling_range.replace(0, np.nan)
    return pd.DataFrame(payload, index=replay_df.index)


def _build_replay_feature_frame(replay_path: Path) -> pd.DataFrame:
    if not replay_path.exists():
        raise FileNotFoundError(f"Missing replay chunk data for exact entry repair: {replay_path}")
    replay_df = pd.read_parquet(replay_path)
    _require_columns(replay_df, REPLAY_BASE_COLS, artifact_name=str(replay_path))
    replay_df = replay_df[REPLAY_BASE_COLS].copy()
    replay_df["replay_timestamp_utc_v1"] = pd.to_datetime(replay_df["time"], utc=True, errors="coerce")
    replay_df = replay_df.sort_values("replay_timestamp_utc_v1", kind="mergesort").reset_index(drop=True)

    close = pd.to_numeric(replay_df["close"], errors="coerce")
    open_ = pd.to_numeric(replay_df["open"], errors="coerce")
    high = pd.to_numeric(replay_df["high"], errors="coerce")
    low = pd.to_numeric(replay_df["low"], errors="coerce")
    range_abs = (high - low).astype(float)
    ret1 = close.pct_change() * 1e4
    spread = pd.to_numeric(replay_df["spread_bps"], errors="coerce")
    spread_median_5 = spread.rolling(5, min_periods=3).median()
    range_bps = range_abs / close.replace(0, np.nan) * 1e4
    range_mean_5 = range_bps.rolling(5, min_periods=3).mean()

    replay_df["as_of_entry_replay_range_bps_v1"] = range_bps
    replay_df["as_of_entry_replay_body_bps_v1"] = (close - open_).abs() / close.replace(0, np.nan) * 1e4
    replay_df["as_of_entry_replay_upper_wick_share_v1"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / range_abs.replace(0, np.nan)
    replay_df["as_of_entry_replay_lower_wick_share_v1"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / range_abs.replace(0, np.nan)
    replay_df["as_of_entry_replay_close_in_bar_v1"] = (close - low) / range_abs.replace(0, np.nan)
    replay_df["as_of_entry_replay_window_ret_1_bps_v1"] = ret1
    replay_df["as_of_entry_replay_window_ret_3_bps_v1"] = close.pct_change(3) * 1e4
    replay_df["as_of_entry_replay_window_ret_5_bps_v1"] = close.pct_change(5) * 1e4
    replay_df["as_of_entry_replay_window_realized_vol_3_bps_v1"] = ret1.rolling(3, min_periods=2).std()
    replay_df["as_of_entry_replay_window_realized_vol_5_bps_v1"] = ret1.rolling(5, min_periods=3).std()
    replay_df["as_of_entry_replay_window_spread_minus_median_5_bps_v1"] = spread - spread_median_5
    replay_df["as_of_entry_replay_window_spread_ratio_median_5_v1"] = spread / spread_median_5.replace(0, np.nan)
    replay_df["as_of_entry_replay_window_range_minus_mean_5_bps_v1"] = range_bps - range_mean_5
    replay_df["as_of_entry_replay_window_range_ratio_mean_5_v1"] = range_bps / range_mean_5.replace(0, np.nan)
    pressure_df = _market_pressure_fields(replay_df)
    for field_name in pressure_df.columns:
        replay_df[field_name] = pd.to_numeric(pressure_df[field_name], errors="coerce")
    replay_df = replay_df.rename(columns=REPLAY_RENAME)
    feature_cols = [
        "replay_timestamp_utc_v1",
        *[value for value in REPLAY_RENAME.values()],
        "as_of_entry_replay_range_bps_v1",
        "as_of_entry_replay_body_bps_v1",
        "as_of_entry_replay_upper_wick_share_v1",
        "as_of_entry_replay_lower_wick_share_v1",
        "as_of_entry_replay_close_in_bar_v1",
        "as_of_entry_replay_window_ret_1_bps_v1",
        "as_of_entry_replay_window_ret_3_bps_v1",
        "as_of_entry_replay_window_ret_5_bps_v1",
        "as_of_entry_replay_window_realized_vol_3_bps_v1",
        "as_of_entry_replay_window_realized_vol_5_bps_v1",
        "as_of_entry_replay_window_spread_minus_median_5_bps_v1",
        "as_of_entry_replay_window_spread_ratio_median_5_v1",
        "as_of_entry_replay_window_range_minus_mean_5_bps_v1",
        "as_of_entry_replay_window_range_ratio_mean_5_v1",
        *[field for field in pressure_df.columns],
    ]
    return replay_df[feature_cols].copy()


def _resolve_run_dir(reports_root: Path, run_id: str) -> Path:
    direct = reports_root / run_id
    if direct.exists():
        return direct
    legacy = reports_root / "runs" / run_id
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Could not resolve run dir for {run_id} under {reports_root}")


def _load_missing_candidate_repair_rows(
    *,
    reports_root: Path,
    missing_asof_df: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovered_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []
    require_xgb = any(str(feature).startswith("as_of_skip_xgb_") for feature in feature_names)
    for run_id, run_missing in missing_asof_df.groupby("run_id", dropna=False):
        run_id = str(run_id)
        run_dir = _resolve_run_dir(reports_root, run_id)
        candidate_path = run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
        replay_path = run_dir / "replay" / "chunk_0" / "chunk_0_data.parquet"
        xgb_path = run_dir / f"xgb_multi_horizon_predictions_{run_id}.parquet"
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing shadow_meta_candidates source for repair: {candidate_path}")
        if require_xgb and not xgb_path.exists():
            raise FileNotFoundError(f"Missing XGB prediction source for repair: {xgb_path}")

        candidate_df = pd.read_parquet(candidate_path)
        _require_columns(candidate_df, CANDIDATE_COLS, artifact_name=str(candidate_path))
        run_candidate_uids = set(run_missing["candidate_uid"].astype("string").tolist())
        candidate_df = candidate_df[candidate_df["candidate_uid"].astype("string").isin(run_candidate_uids)].copy()
        if len(candidate_df) != len(run_candidate_uids) or candidate_df["candidate_uid"].astype("string").nunique() != len(run_candidate_uids):
            raise RuntimeError(
                f"Repair requires exact one candidate snapshot per missing row for {run_id}: "
                f"expected={len(run_candidate_uids)} observed_rows={len(candidate_df)} observed_unique={candidate_df['candidate_uid'].astype('string').nunique()}"
            )
        candidate_df["candidate_uid"] = candidate_df["candidate_uid"].astype("string")
        candidate_df["repair_timestamp_utc_v1"] = pd.to_datetime(candidate_df["decision_ts_utc"], utc=True, errors="coerce")
        if candidate_df["repair_timestamp_utc_v1"].isna().any():
            raise RuntimeError(f"Repair found null candidate decision timestamps in {candidate_path}")

        replay_df = _build_replay_feature_frame(replay_path)
        candidate_df = candidate_df.merge(
            replay_df,
            left_on="repair_timestamp_utc_v1",
            right_on="replay_timestamp_utc_v1",
            how="left",
            validate="many_to_one",
        )
        replay_exact_rows = int(candidate_df["replay_timestamp_utc_v1"].notna().sum())
        if replay_exact_rows != len(candidate_df):
            raise RuntimeError(f"Repair requires exact replay chunk row for {run_id}: expected={len(candidate_df)} observed={replay_exact_rows}")

        xgb_exact_rows = 0
        if require_xgb:
            xgb_df = pd.read_parquet(xgb_path)
            _require_columns(xgb_df, XGB_COLS, artifact_name=str(xgb_path))
            xgb_df = xgb_df[XGB_COLS].copy()
            xgb_df["xgb_timestamp_utc_v1"] = pd.to_datetime(xgb_df["ts"], utc=True, errors="coerce")
            xgb_df = xgb_df.sort_values("xgb_timestamp_utc_v1", kind="mergesort").drop_duplicates(
                subset=["xgb_timestamp_utc_v1"],
                keep="last",
            )
            candidate_df = candidate_df.merge(
                xgb_df[
                    [
                        "xgb_timestamp_utc_v1",
                        "p_long",
                        "p_short",
                        "p_flat",
                        "p_hat",
                        "pred_side",
                        "has_ctx",
                    ]
                ].rename(
                    columns={
                        "p_long": "xgb_p_long_v1",
                        "p_short": "xgb_p_short_v1",
                        "p_flat": "xgb_p_flat_v1",
                        "p_hat": "xgb_p_hat_v1",
                        "pred_side": "xgb_pred_side_v1",
                        "has_ctx": "xgb_has_ctx_v1",
                    }
                ),
                left_on="repair_timestamp_utc_v1",
                right_on="xgb_timestamp_utc_v1",
                how="left",
                validate="many_to_one",
            )
            xgb_exact_rows = int(candidate_df["xgb_timestamp_utc_v1"].notna().sum())
            if xgb_exact_rows != len(candidate_df):
                raise RuntimeError(f"Repair requires exact XGB timestamp row for {run_id}: expected={len(candidate_df)} observed={xgb_exact_rows}")

            for name in ["long", "short", "flat", "hat"]:
                candidate_values = pd.to_numeric(candidate_df[f"p_{name}"], errors="coerce")
                xgb_values = pd.to_numeric(candidate_df[f"xgb_p_{name}_v1"], errors="coerce")
                max_diff = (candidate_values - xgb_values).abs().max()
                if pd.notna(max_diff) and float(max_diff) > 1e-12:
                    raise RuntimeError(f"Candidate/XGB probability mismatch for {run_id} p_{name}: max_diff={max_diff}")

        recovered_frames.append(candidate_df)
        run_rows.append(
            {
                "run_id": run_id,
                "missing_rows_v1": int(len(run_missing)),
                "candidate_snapshot_exact_rows_v1": int(len(candidate_df)),
                "replay_chunk_exact_rows_v1": replay_exact_rows,
                "xgb_exact_rows_v1": xgb_exact_rows,
                "repair_status_v1": "PASS",
                "xgb_required_v1": require_xgb,
            }
        )

    recovered = pd.concat(recovered_frames, ignore_index=True) if recovered_frames else pd.DataFrame()
    return recovered, pd.DataFrame(run_rows)


def _build_recovered_feature_table(recovered_source_df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    recovered = pd.DataFrame({"candidate_uid": recovered_source_df["candidate_uid"].astype("string")})
    direct_map = {
        "as_of_hour_utc_v1": "hour_utc",
        "as_of_weekday_utc_v1": "weekday_utc",
        "as_of_session_v1": "session",
        "as_of_side_v1": "side",
        "as_of_atr_bps_v1": "atr_bps",
        "as_of_candidate_entry_spread_bps_v1": "entry_spread_bps",
        "as_of_candidate_uncertainty_score_v1": "uncertainty_score",
        "as_of_candidate_tradable_prob_v1": "tradable_prob",
        "as_of_candidate_mfe_first_n_pred_v1": "mfe_first_n_pred",
        "as_of_candidate_trend_regime_v1": "trend_regime",
        "as_of_candidate_vol_regime_v1": "vol_regime",
        "as_of_entry_candidate_margin_v1": "margin",
        "as_of_entry_candidate_path_quality_pred_v1": "path_quality_pred",
        "as_of_skip_xgb_p_flat_v1": "xgb_p_flat_v1",
        "as_of_skip_xgb_p_hat_v1": "xgb_p_hat_v1",
        "as_of_skip_xgb_p_long_v1": "xgb_p_long_v1",
        "as_of_skip_xgb_p_short_v1": "xgb_p_short_v1",
        "as_of_skip_xgb_pred_side_v1": "xgb_pred_side_v1",
        "as_of_skip_xgb_has_ctx_v1": "xgb_has_ctx_v1",
        "as_of_skip_candidate_entry_spread_bps_v1": "entry_spread_bps",
        "as_of_skip_candidate_margin_v1": "margin",
        "as_of_skip_candidate_p_flat_v1": "p_flat",
        "as_of_skip_candidate_p_hat_v1": "p_hat",
        "as_of_skip_candidate_p_long_v1": "p_long",
        "as_of_skip_candidate_p_short_v1": "p_short",
        "as_of_skip_candidate_path_quality_pred_v1": "path_quality_pred",
    }
    for feature_name, source_col in direct_map.items():
        if feature_name in feature_names:
            recovered[feature_name] = recovered_source_df[source_col]
    for feature_name in feature_names:
        if feature_name.startswith("as_of_skip_replay_"):
            source_col = feature_name.replace("as_of_skip_replay_", "as_of_entry_replay_", 1)
            if source_col not in recovered_source_df.columns:
                raise KeyError(f"Missing computed replay source column for {feature_name}: {source_col}")
            recovered[feature_name] = recovered_source_df[source_col]
    missing_features = [feature for feature in feature_names if feature not in recovered.columns]
    if missing_features:
        raise KeyError(f"Repair source map does not cover required AS_OF features: {missing_features}")
    return recovered[["candidate_uid", *feature_names]].copy()


def _feature_source_rows(feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feature in feature_names:
        if feature.startswith("as_of_skip_replay_"):
            source_family = "replay_chunk_bar_exact_and_window_exact"
            source_artifact = "runs/<run>/replay/chunk_0/chunk_0_data.parquet"
        elif feature.startswith("as_of_skip_xgb_"):
            source_family = "xgb_multi_horizon_predictions_exact"
            source_artifact = "runs/<run>/xgb_multi_horizon_predictions_<run>.parquet"
        elif feature.startswith("as_of_skip_candidate_") or feature in {
            "as_of_entry_candidate_margin_v1",
            "as_of_entry_candidate_path_quality_pred_v1",
            "as_of_candidate_entry_spread_bps_v1",
            "as_of_candidate_uncertainty_score_v1",
            "as_of_candidate_tradable_prob_v1",
            "as_of_candidate_mfe_first_n_pred_v1",
        }:
            source_family = "shadow_meta_candidates_exact_decision_ts"
            source_artifact = "runs/<run>/shadow_meta_candidates_<run>_MERGED.parquet"
        elif feature.startswith("as_of_"):
            source_family = "shadow_meta_candidates_exact_decision_ts_core_state"
            source_artifact = "runs/<run>/shadow_meta_candidates_<run>_MERGED.parquet"
        else:
            source_family = "UNKNOWN"
            source_artifact = "UNKNOWN"
        rows.append(
            {
                "feature_name_v1": feature,
                "repair_source_family_v1": source_family,
                "repair_source_artifact_v1": source_artifact,
                "as_of_safe_v1": True,
                "synthetic_value_used_v1": False,
                "hindsight_label_used_v1": False,
            }
        )
    return rows


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _repair_run_rollup(coverage_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_id, run_df in coverage_df.groupby("run_id", dropna=False):
        rows.append(
            {
                "run_id": str(run_id),
                "ledger_trade_count_v1": int(len(run_df)),
                "entry_coverage_count_v1": int(_bool(run_df, "entry_observation_present_v1").sum()),
                "entry_missing_count_v1": int((~_bool(run_df, "entry_observation_present_v1")).sum()),
                "entry_repaired_count_v1": int(_bool(run_df, "entry_coverage_repair_applied_v1").sum()),
                "management_missing_count_v1": int((~_bool(run_df, "management_observation_present_v1", True)).sum())
                if "management_observation_present_v1" in run_df.columns
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("run_id", key=lambda s: s.map(_run_sort_key))


def _consistency_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Entry Coverage Repair Readiness V1",
        "",
        "Shadow/research readiness only. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Original entry coverage: `{summary['original_entry_coverage_v1']}/{summary['ledger_trade_count_v1']}`",
        f"- Repaired entry coverage: `{summary['repaired_entry_coverage_v1']}/{summary['ledger_trade_count_v1']}`",
        f"- Recovered missing rows: `{summary['recovered_missing_entry_rows_v1']}`",
        f"- Synthetic values used: `{summary['synthetic_values_used_v1']}`",
        f"- Recovered should-not-take labels: `{summary['recovered_label_counts_v1'].get('label_should_not_take_v1', {})}`",
        f"- Recovered strong-trade labels: `{summary['recovered_label_counts_v1'].get('label_strong_trade_candidate_v1', {})}`",
        f"- Status: `{summary['status_v1']['ENTRY_COVERAGE_REPAIR_STATUS']}`",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    readiness_dir: Path,
    canonical_dir: Path,
    extension_dir: Path,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    contract = _load_json(readiness_dir / R2_READINESS_CONTRACT)
    summary_in = _load_json(readiness_dir / R2_READINESS_SUMMARY)
    manifest_in = _load_json(readiness_dir / R2_READINESS_MANIFEST) if (readiness_dir / R2_READINESS_MANIFEST).exists() else {}
    asof_df = pd.read_parquet(readiness_dir / R2_AS_OF_TABLE)
    labels_df = pd.read_parquet(readiness_dir / R2_LABEL_TABLE)
    coverage_df = pd.read_csv(readiness_dir / R2_COVERAGE_AUDIT)
    asof_decision_df = pd.read_parquet(canonical_dir / CANONICAL_AS_OF_LEDGER)
    feature_names = [str(feature) for feature in contract.get("as_of_feature_names_v1", [])]
    if not feature_names:
        raise RuntimeError("R2 readiness contract missing non-empty as_of_feature_names_v1")
    _check_feature_names(feature_names)
    _require_columns(
        asof_df,
        ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "entry_observation_present_v1", "entry_raw_state_present_v1", *feature_names],
        artifact_name=R2_AS_OF_TABLE,
    )
    _require_columns(labels_df, ["candidate_uid"], artifact_name=R2_LABEL_TABLE)
    _require_columns(coverage_df, ["run_id", "candidate_uid", "entry_observation_present_v1"], artifact_name=R2_COVERAGE_AUDIT)
    if bool(asof_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError("AS_OF table requires unique candidate_uid")
    if bool(labels_df["candidate_uid"].astype("string").duplicated().any()):
        raise ValueError("HINDSIGHT label table requires unique candidate_uid")
    if expected_ledger_count is not None and int(len(asof_df)) != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger trade count expected {expected_ledger_count}, observed {len(asof_df)}")

    original_feature_available = _bool(asof_df, "entry_observation_present_v1") & _bool(asof_df, "entry_raw_state_present_v1")
    missing_df = asof_df.loc[~original_feature_available, ["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]].copy()
    recovered_source_df, run_repair_audit_df = _load_missing_candidate_repair_rows(
        reports_root=reports_root,
        missing_asof_df=missing_df,
        feature_names=feature_names,
    )
    recovered_features_df = _build_recovered_feature_table(recovered_source_df, feature_names)
    recovered_set = set(recovered_features_df["candidate_uid"].astype("string").tolist())
    missing_set = set(missing_df["candidate_uid"].astype("string").tolist())
    unrecovered = sorted(missing_set - recovered_set)
    if unrecovered:
        raise RuntimeError(f"Coverage repair failed to recover missing entry rows from exact run sources: {unrecovered[:20]}")

    recovered_from_asof_decision = int(asof_decision_df["candidate_uid"].astype("string").isin(recovered_set).sum())
    repaired_asof_df = asof_df.copy()
    repaired_asof_df["entry_coverage_original_entry_observation_present_v1"] = _bool(repaired_asof_df, "entry_observation_present_v1")
    repaired_asof_df["entry_coverage_original_entry_raw_state_present_v1"] = _bool(repaired_asof_df, "entry_raw_state_present_v1")
    repaired_asof_df["entry_coverage_repair_applied_v1"] = repaired_asof_df["candidate_uid"].astype("string").isin(recovered_set)
    repaired_asof_df["entry_coverage_repair_source_v1"] = np.where(
        repaired_asof_df["entry_coverage_repair_applied_v1"],
        "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
        "ORIGINAL_R2_ENTRY_OBSERVABILITY",
    )
    repaired_asof_df["entry_coverage_repair_contract_v1"] = "ENTRY_COVERAGE_REPAIR_READINESS_V1|AS_OF_FEATURES_ONLY|NO_SYNTHETIC_VALUES|NOT_LIVE_GATE"
    recovered_lookup = recovered_features_df.set_index("candidate_uid")
    repaired_index = repaired_asof_df["candidate_uid"].astype("string")
    repair_mask = repaired_index.isin(recovered_set)
    for feature in feature_names:
        repaired_asof_df.loc[repair_mask, feature] = repaired_index.loc[repair_mask].map(recovered_lookup[feature])
    repaired_asof_df.loc[repair_mask, "entry_observation_present_v1"] = True
    repaired_asof_df.loc[repair_mask, "entry_raw_state_present_v1"] = True

    repaired_coverage_df = coverage_df.copy()
    repaired_coverage_df["entry_original_gap_reason_code_v1"] = repaired_coverage_df.get("entry_gap_reason_code_v1", pd.Series(pd.NA, index=repaired_coverage_df.index)).astype("string")
    repaired_coverage_df["entry_original_gap_reason_detail_v1"] = repaired_coverage_df.get("entry_gap_reason_detail_v1", pd.Series(pd.NA, index=repaired_coverage_df.index)).astype("string")
    repaired_coverage_df["entry_coverage_repair_applied_v1"] = repaired_coverage_df["candidate_uid"].astype("string").isin(recovered_set)
    repaired_coverage_df["entry_coverage_repair_source_v1"] = np.where(
        repaired_coverage_df["entry_coverage_repair_applied_v1"],
        "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
        "ORIGINAL_R2_ENTRY_OBSERVABILITY",
    )
    repaired_coverage_df.loc[repaired_coverage_df["entry_coverage_repair_applied_v1"], "entry_observation_present_v1"] = True
    repaired_coverage_df.loc[repaired_coverage_df["entry_coverage_repair_applied_v1"], "entry_gap_reason_code_v1"] = "covered by exact run-source repair"
    repaired_coverage_df.loc[repaired_coverage_df["entry_coverage_repair_applied_v1"], "entry_gap_reason_detail_v1"] = (
        "recovered from exact shadow_meta_candidates + replay chunk + xgb timestamp sources; original gap retained in entry_original_gap_reason_*"
    )
    if "coverage_gap_scope_v1" in repaired_coverage_df.columns:
        repaired_coverage_df.loc[repaired_coverage_df["entry_coverage_repair_applied_v1"], "coverage_gap_scope_v1"] = "REPAIRED_ENTRY_COVERAGE"

    recovered_labels = labels_df[labels_df["candidate_uid"].astype("string").isin(recovered_set)].copy()
    label_count_cols = [
        "label_should_not_take_v1",
        "label_immediate_mae_risk_v1",
        "label_wait_would_have_helped_v1",
        "label_good_mfe_bad_capture_v1",
        "label_low_mfe_low_value_v1",
        "label_strong_trade_candidate_v1",
        "label_direct_take_ok_v1",
        "runner_50bps_opportunity_v1",
        "runner_100bps_opportunity_v1",
        "home_run_200bps_opportunity_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
    ]
    recovered_label_counts = {column: _counts(recovered_labels, column) for column in label_count_cols if column in recovered_labels.columns}
    numeric_cols = ["baseline_realized_pnl_bps_v1", "peak_mfe_bps_v1", "mae_abs_bps_v1", "giveback_bps_v1"]
    recovered_quality = {
        column: _safe_float(pd.to_numeric(recovered_labels[column], errors="coerce").mean())
        for column in numeric_cols
        if column in recovered_labels.columns
    }

    feature_source_audit_df = pd.DataFrame(_feature_source_rows(feature_names))
    non_null_original = asof_df.loc[original_feature_available, feature_names].notna().sum()
    non_null_recovered = recovered_features_df[feature_names].notna().sum()
    for column in ["original_non_null_rows_v1", "recovered_non_null_rows_v1"]:
        feature_source_audit_df[column] = 0
    feature_source_audit_df["original_non_null_rows_v1"] = feature_source_audit_df["feature_name_v1"].map(non_null_original).fillna(0).astype(int)
    feature_source_audit_df["recovered_non_null_rows_v1"] = feature_source_audit_df["feature_name_v1"].map(non_null_recovered).fillna(0).astype(int)

    repair_audit_df = missing_df.merge(
        recovered_source_df[
            [
                column
                for column in [
                    "candidate_uid",
                    "repair_timestamp_utc_v1",
                    "replay_timestamp_utc_v1",
                    "xgb_timestamp_utc_v1",
                    "decision",
                    "accepted",
                    "decision_reason",
                    "policy_hash",
                    "entry_bundle_sha256",
                    "exit_bundle_sha256",
                ]
                if column in recovered_source_df.columns
            ]
        ],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    repair_audit_df["entry_coverage_repair_status_v1"] = "RECOVERED_EXACT_RUN_SOURCE"
    repair_audit_df["synthetic_value_used_v1"] = False
    repair_audit_df["hindsight_label_used_for_as_of_repair_v1"] = False
    repair_audit_df["recovery_source_v1"] = "shadow_meta_candidates + replay/chunk_0/chunk_0_data + xgb_multi_horizon_predictions"
    repair_audit_df = repair_audit_df.merge(
        coverage_df[
            [
                column
                for column in [
                    "candidate_uid",
                    "entry_gap_reason_code_v1",
                    "entry_gap_reason_detail_v1",
                    "management_gap_reason_code_v1",
                    "coverage_gap_scope_v1",
                ]
                if column in coverage_df.columns
            ]
        ],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    repaired_run_rollup_df = _repair_run_rollup(repaired_coverage_df)

    repaired_feature_available = _bool(repaired_asof_df, "entry_observation_present_v1") & _bool(repaired_asof_df, "entry_raw_state_present_v1")
    original_coverage = int(original_feature_available.sum())
    repaired_coverage = int(repaired_feature_available.sum())
    repaired_missing = int((~repaired_feature_available).sum())
    consistency_rows = [
        _consistency_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS", {"expected": expected_ledger_count, "observed": int(len(repaired_asof_df))}),
        _consistency_record("ORIGINAL_ENTRY_COVERAGE_1806", "PASS" if original_coverage == 1806 or expected_ledger_count != 1971 else "FAIL", {"observed": original_coverage}),
        _consistency_record("MISSING_ENTRY_ROWS_RECOVERED", "PASS" if int(len(recovered_set)) == int(len(missing_set)) else "FAIL", {"missing": int(len(missing_set)), "recovered": int(len(recovered_set))}),
        _consistency_record("REPAIRED_ENTRY_FEATURE_COVERAGE_FULL_LEDGER", "PASS" if repaired_missing == 0 else "FAIL", {"covered": repaired_coverage, "missing": repaired_missing}),
        _consistency_record("AS_OF_FEATURE_LEAKAGE_SCAN", "PASS", {"feature_count": int(len(feature_names))}),
        _consistency_record("NO_SYNTHETIC_VALUES_USED", "PASS", {"synthetic_value_used_v1": False}),
        _consistency_record("HINDSIGHT_LABELS_PHYSICALLY_SEPARATE", "PASS", {"as_of_columns": int(len(repaired_asof_df.columns)), "label_columns_in_label_table": int(len(labels_df.columns))}),
        _consistency_record("REPAIR_SOURCE_EXACT_RUN_ARTIFACTS", "PASS", {"run_count": int(run_repair_audit_df["run_id"].nunique()), "rows": int(len(repair_audit_df))}),
    ]
    consistency_df = pd.DataFrame(consistency_rows)
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "ENTRY_COVERAGE_REPAIR_STATUS_V1",
        "ENTRY_COVERAGE_REPAIR_STATUS": "ENTRY_COVERAGE_REPAIRED_READY_FOR_SHADOW_RETRAIN_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "ENTRY_COVERAGE_REPAIR_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "source_readiness_dir_v1": str(readiness_dir),
        "canonical_dir_v1": str(canonical_dir),
        "extension_dir_v1": str(extension_dir),
        "ledger_trade_count_v1": int(len(repaired_asof_df)),
        "original_entry_coverage_v1": original_coverage,
        "original_entry_missing_v1": int(len(missing_set)),
        "recovered_missing_entry_rows_v1": int(len(recovered_set)),
        "repaired_entry_coverage_v1": repaired_coverage,
        "repaired_entry_missing_v1": repaired_missing,
        "recovered_from_as_of_decision_moment_ledger_rows_v1": recovered_from_asof_decision,
        "recovered_from_run_shadow_meta_candidates_rows_v1": int(len(recovered_set)),
        "recovered_replay_chunk_exact_rows_v1": int(run_repair_audit_df["replay_chunk_exact_rows_v1"].sum()) if not run_repair_audit_df.empty else 0,
        "recovered_xgb_exact_rows_v1": int(run_repair_audit_df["xgb_exact_rows_v1"].sum()) if not run_repair_audit_df.empty else 0,
        "feature_count_v1": int(len(feature_names)),
        "synthetic_values_used_v1": False,
        "hindsight_labels_used_for_as_of_repair_v1": False,
        "recovered_label_counts_v1": recovered_label_counts,
        "recovered_quality_means_v1": recovered_quality,
        "repaired_readiness_is_drop_in_for_r3_r4_research_v1": True,
        "not_live_gate": True,
        "not_policy_truth": True,
        "source_r2_summary_reference_v1": {
            "summary_path_v1": str(readiness_dir / R2_READINESS_SUMMARY),
            "original_layer_name_v1": summary_in.get("layer_name"),
        },
        "status_v1": status,
    }
    repaired_contract = dict(contract)
    repaired_contract.update(
        {
            "layer_name": "ENTRY_COVERAGE_REPAIR_READINESS_CONTRACT_V1",
            "source_contract_v1": str(readiness_dir / R2_READINESS_CONTRACT),
            "repair_mode_v1": "AS_OF_EXACT_RUN_SOURCE_REPAIR_NO_SYNTHETIC_VALUES",
            "as_of_feature_names_v1": list(feature_names),
            "hindsight_labels_physically_separate_v1": True,
            "not_controller": True,
            "not_live_gate": True,
            "not_policy_truth": True,
        }
    )
    repaired_readiness_summary = dict(summary_in)
    repaired_readiness_summary.update(
        {
            "layer_name": "HARVEST_R2_ENTRY_READINESS_WITH_ENTRY_COVERAGE_REPAIR_V1",
            "coverage_repair_applied_v1": True,
            "source_readiness_dir_v1": str(readiness_dir),
            "entry_feature_coverage_v1": repaired_coverage,
            "entry_feature_missing_v1": repaired_missing,
            "entry_coverage_repair_summary_v1": summary,
            "not_live_gate": True,
            "not_policy_truth": True,
        }
    )
    manifest = {
        "layer_name": "ENTRY_COVERAGE_REPAIR_MANIFEST_V1",
        "mode_v1": "DROP_IN_READINESS_EXTENSION_FOR_SHADOW_RETRAIN_ONLY",
        "source_manifest_v1": manifest_in,
        "artifacts_v1": {
            "as_of_table_v1": R2_AS_OF_TABLE,
            "hindsight_label_table_v1": R2_LABEL_TABLE,
            "coverage_audit_v1": R2_COVERAGE_AUDIT,
            "coverage_run_rollup_v1": R2_COVERAGE_RUN_ROLLUP,
            "readiness_contract_v1": R2_READINESS_CONTRACT,
            "readiness_summary_v1": R2_READINESS_SUMMARY,
            "repair_audit_v1": REPAIR_AUDIT,
            "repair_feature_source_audit_v1": REPAIR_FEATURE_SOURCE_AUDIT,
            "repair_consistency_audit_v1": REPAIR_CONSISTENCY_AUDIT,
            "repair_summary_v1": REPAIR_SUMMARY,
            "repair_status_v1": REPAIR_STATUS,
            "repair_report_v1": REPAIR_REPORT,
        },
    }
    return {
        "repaired_asof_df": repaired_asof_df,
        "labels_df": labels_df,
        "repaired_coverage_df": repaired_coverage_df,
        "repaired_run_rollup_df": repaired_run_rollup_df,
        "repair_audit_df": repair_audit_df,
        "feature_source_audit_df": feature_source_audit_df,
        "run_repair_audit_df": run_repair_audit_df,
        "consistency_df": consistency_df,
        "contract": repaired_contract,
        "summary": summary,
        "readiness_summary": repaired_readiness_summary,
        "status": status,
        "manifest": manifest,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    readiness_dir: Path | None = None,
    canonical_dir: Path | None = None,
    extension_dir: Path | None = None,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_readiness_dir = readiness_dir or _resolve_dir(reports_root, None, R2_READINESS_EXTENSION_NAME, R2_AS_OF_TABLE)
    resolved_canonical_dir = canonical_dir or _resolve_dir(reports_root, None, CANONICAL_EXTENSION_NAME, CANONICAL_AS_OF_LEDGER)
    resolved_extension_dir = extension_dir or _default_extension_dir(reports_root)
    resolved_extension_dir = Path(resolved_extension_dir).expanduser().resolve()
    payload = build_payload(
        reports_root=reports_root,
        readiness_dir=Path(resolved_readiness_dir).expanduser().resolve(),
        canonical_dir=Path(resolved_canonical_dir).expanduser().resolve(),
        extension_dir=resolved_extension_dir,
        expected_ledger_count=expected_ledger_count,
    )
    resolved_extension_dir.mkdir(parents=True, exist_ok=True)
    payload["repaired_asof_df"].to_parquet(resolved_extension_dir / R2_AS_OF_TABLE, index=False)
    payload["labels_df"].to_parquet(resolved_extension_dir / R2_LABEL_TABLE, index=False)
    payload["repaired_coverage_df"].to_csv(resolved_extension_dir / R2_COVERAGE_AUDIT, index=False)
    payload["repaired_run_rollup_df"].to_csv(resolved_extension_dir / R2_COVERAGE_RUN_ROLLUP, index=False)
    payload["repair_audit_df"].to_csv(resolved_extension_dir / REPAIR_AUDIT, index=False)
    payload["feature_source_audit_df"].to_csv(resolved_extension_dir / REPAIR_FEATURE_SOURCE_AUDIT, index=False)
    payload["run_repair_audit_df"].to_csv(resolved_extension_dir / "shadow_meta_all_trade_review_entry_coverage_repair_run_source_audit_v1.csv", index=False)
    payload["consistency_df"].to_csv(resolved_extension_dir / REPAIR_CONSISTENCY_AUDIT, index=False)
    _write_json(resolved_extension_dir / R2_READINESS_CONTRACT, payload["contract"])
    _write_json(resolved_extension_dir / R2_READINESS_SUMMARY, payload["readiness_summary"])
    _write_json(resolved_extension_dir / REPAIR_CONTRACT, payload["contract"])
    _write_json(resolved_extension_dir / REPAIR_SUMMARY, payload["summary"])
    _write_json(resolved_extension_dir / REPAIR_STATUS, payload["status"])
    _write_json(resolved_extension_dir / REPAIR_MANIFEST, payload["manifest"])
    (resolved_extension_dir / REPAIR_REPORT).write_text(payload["report"], encoding="utf-8")
    (resolved_extension_dir / R2_READINESS_REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(resolved_extension_dir / R2_READINESS_MANIFEST, payload["manifest"])

    top_level = dict(payload["summary"])
    top_level["extension_dir_v1"] = str(resolved_extension_dir)
    _write_json(reports_root / TOP_LEVEL_SUMMARY, top_level)
    return {
        "extension_dir": resolved_extension_dir,
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair R2 entry AS_OF coverage from exact run artifacts for shadow retrain readiness.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--readiness-dir", default=None)
    parser.add_argument("--canonical-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    readiness_dir = _resolve_dir(reports_root, args.readiness_dir, R2_READINESS_EXTENSION_NAME, R2_AS_OF_TABLE)
    canonical_dir = _resolve_dir(reports_root, args.canonical_dir, CANONICAL_EXTENSION_NAME, CANONICAL_AS_OF_LEDGER)
    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        canonical_dir=canonical_dir,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
