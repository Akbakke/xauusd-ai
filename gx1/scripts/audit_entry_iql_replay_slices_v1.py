#!/usr/bin/env python3
"""Audit Entry-IQL replay edge across slices and tail-risk buckets.

This gate is report-only. It reads already-materialized candidate and IQL
replay evidence, compares supported session/regime/side slices, and keeps
shadow/live/promotion closed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT
from gx1.scripts.verify_entry_replay_readiness_v1 import DEFAULT_REPLAY_DIR as DEFAULT_CANDIDATE_REPLAY_DIR


DEFAULT_IQL_REPLAY_DIR = REPORTS_ROOT / "entry_iql_distillation_replay_20260628_v1"
DEFAULT_COMPARISON_JSON = (
    REPORTS_ROOT / "entry_iql_replay_comparison_20260628_v1/ENTRY_IQL_REPLAY_COMPARISON_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_iql_replay_slice_audit_20260628_v1"

EDGE_CUBES = ("session", "vol_regime", "side")
DIAGNOSTIC_CUBES = ("direction", "exit_reason", "bad_path", "mae_tail", "held_bars")
CUBE_COLUMNS = {
    "session": "session_slice",
    "vol_regime": "vol_regime_slice",
    "side": "side_slice",
    "direction": "direction_slice",
    "exit_reason": "exit_reason_slice",
    "bad_path": "bad_path_slice",
    "mae_tail": "mae_tail_slice",
    "held_bars": "held_bars_slice",
}
TAIL_PATH_REQUIRED_COLUMNS = {
    "net_pnl_bps",
    "exit_reason",
    "mfe_bps",
    "mae_bps",
    "held_bars",
    "bad_path_prob",
    "path_quality_pred",
}
TAIL_PATH_NUMERIC_COLUMNS = {
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
    "bad_path_prob",
    "path_quality_pred",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _path_or_default(value: Any, fallback: Path) -> Path:
    raw = str(value or "").strip()
    return Path(raw).expanduser().resolve() if raw else fallback.expanduser().resolve()


def _manifest_trades_path(manifest: dict[str, Any], replay_dir: Path) -> Path:
    for key in ("trades_path", "input_trades_path", "source_trades_path"):
        raw = str(manifest.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return (replay_dir / "replay_policy_trades.csv").expanduser().resolve()


def _clean_category(value: Any, *, upper: bool = True) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if not text:
        return "UNKNOWN"
    return text.upper() if upper else text


def _category_series(df: pd.DataFrame, *names: str, upper: bool = True) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].map(lambda value: _clean_category(value, upper=upper))
    return pd.Series(["UNKNOWN"] * len(df), index=df.index)


def _direction_series(df: pd.DataFrame) -> pd.Series:
    if "direction_correct" not in df.columns:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)

    def normalize(value: Any) -> str:
        if value is None or pd.isna(value):
            return "UNKNOWN"
        if isinstance(value, (bool, np.bool_)):
            return "TRUE" if bool(value) else "FALSE"
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return "TRUE"
        if text in {"false", "0", "no", "n"}:
            return "FALSE"
        return "UNKNOWN"

    return df["direction_correct"].map(normalize)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _tail_thresholds(candidate: pd.DataFrame, iql: pd.DataFrame, column: str) -> dict[str, float]:
    values = pd.concat([_numeric(candidate, column), _numeric(iql, column)], ignore_index=True).dropna()
    if values.empty:
        return {}
    return {
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def _three_bucket(values: pd.Series, *, p75: float | None, p90: float | None, prefix: str) -> pd.Series:
    if p75 is None or p90 is None:
        return pd.Series(["UNKNOWN"] * len(values), index=values.index)
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series([f"{prefix}_below_p75"] * len(values), index=values.index)
    out[numeric.isna()] = "UNKNOWN"
    out[numeric >= float(p75)] = f"{prefix}_p75_p90"
    out[numeric >= float(p90)] = f"{prefix}_p90_plus"
    return out


def _held_bars_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(["UNKNOWN"] * len(values), index=values.index)
    out[(numeric >= 0) & (numeric <= 6)] = "held_0_6"
    out[(numeric >= 7) & (numeric <= 12)] = "held_7_12"
    out[(numeric >= 13) & (numeric <= 24)] = "held_13_24"
    out[numeric >= 25] = "held_25_plus"
    return out


def _normalize_trades(
    trades: pd.DataFrame,
    *,
    model: str,
    bad_path_thresholds: dict[str, float],
    mae_thresholds: dict[str, float],
) -> pd.DataFrame:
    out = trades.copy()
    out["model"] = model
    out["entry_time_ts"] = pd.to_datetime(out.get("entry_time"), errors="coerce", utc=True)
    out["session_slice"] = _category_series(out, "session", "state_session")
    out["vol_regime_slice"] = _category_series(out, "vol_regime", "state_vol_regime", upper=False)
    out["side_slice"] = _category_series(out, "side")
    out["direction_slice"] = _direction_series(out)
    out["exit_reason_slice"] = _category_series(out, "exit_reason")
    out["bad_path_slice"] = _three_bucket(
        _numeric(out, "bad_path_prob"),
        p75=bad_path_thresholds.get("p75"),
        p90=bad_path_thresholds.get("p90"),
        prefix="bad_path",
    )
    out["mae_tail_slice"] = _three_bucket(
        _numeric(out, "mae_bps"),
        p75=mae_thresholds.get("p75"),
        p90=mae_thresholds.get("p90"),
        prefix="mae",
    )
    out["held_bars_slice"] = _held_bars_bucket(_numeric(out, "held_bars"))
    return out


def _profit_factor(values: pd.Series) -> float | None:
    pnl = pd.to_numeric(values, errors="coerce").dropna()
    if pnl.empty:
        return None
    wins = float(pnl[pnl > 0.0].sum())
    losses = float(-pnl[pnl < 0.0].sum())
    if losses <= 0.0:
        return None
    return wins / losses


def _pf_for_check(value: Any, net_sum_bps: Any) -> float:
    try:
        pf = float(value)
    except Exception:
        pf = float("inf") if float(net_sum_bps or 0.0) > 0.0 else 0.0
    return pf if np.isfinite(pf) else float("inf")


def _max_drawdown(values: pd.Series) -> float:
    pnl = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    peak = equity.cummax()
    return float((peak - equity).max())


def _metric_row(model: str, cube: str, slice_name: str, group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("entry_time_ts", na_position="last")
    pnl = pd.to_numeric(ordered.get("net_pnl_bps"), errors="coerce").dropna()
    mae = pd.to_numeric(ordered.get("mae_bps"), errors="coerce").dropna()
    mfe = pd.to_numeric(ordered.get("mfe_bps"), errors="coerce").dropna()
    return {
        "model": model,
        "cube": cube,
        "slice": str(slice_name),
        "n_trades": int(len(pnl)),
        "net_sum_bps": float(pnl.sum()) if not pnl.empty else 0.0,
        "net_mean_bps": float(pnl.mean()) if not pnl.empty else 0.0,
        "profit_factor": _profit_factor(pnl),
        "max_drawdown_bps": _max_drawdown(pnl),
        "max_loss_bps": float(pnl.min()) if not pnl.empty else 0.0,
        "win_rate": float((pnl > 0.0).mean()) if not pnl.empty else 0.0,
        "p10_net_bps": float(pnl.quantile(0.10)) if not pnl.empty else 0.0,
        "p90_net_bps": float(pnl.quantile(0.90)) if not pnl.empty else 0.0,
        "mean_mae_bps": float(mae.mean()) if not mae.empty else None,
        "p90_mae_bps": float(mae.quantile(0.90)) if not mae.empty else None,
        "max_mae_bps": float(mae.max()) if not mae.empty else None,
        "mean_mfe_bps": float(mfe.mean()) if not mfe.empty else None,
        "p90_mfe_bps": float(mfe.quantile(0.90)) if not mfe.empty else None,
    }


def _slice_metrics(candidate: pd.DataFrame, iql: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, df in (("candidate", candidate), ("iql", iql)):
        for cube, column in CUBE_COLUMNS.items():
            for slice_name, group in df.groupby(column, dropna=False, sort=True):
                rows.append(_metric_row(model, cube, str(slice_name), group))
    return pd.DataFrame(rows)


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _num_allow_inf(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key))
    except Exception:
        return default
    return value if np.isfinite(value) or np.isinf(value) else default


def _comparison(metrics: pd.DataFrame, *, min_slice_trades: int) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    candidate = metrics[metrics["model"] == "candidate"].copy()
    iql = metrics[metrics["model"] == "iql"].copy()
    candidate = candidate.rename(columns={col: f"candidate_{col}" for col in candidate.columns})
    iql = iql.rename(columns={col: f"iql_{col}" for col in iql.columns})
    candidate["cube"] = candidate["candidate_cube"]
    candidate["slice"] = candidate["candidate_slice"]
    iql["cube"] = iql["iql_cube"]
    iql["slice"] = iql["iql_slice"]
    merged = candidate.merge(iql, on=["cube", "slice"], how="outer")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        cand_n = int(_num(row, "candidate_n_trades"))
        iql_n = int(_num(row, "iql_n_trades"))
        cand_pf = _pf_for_check(row.get("candidate_profit_factor"), row.get("candidate_net_sum_bps"))
        iql_pf = _pf_for_check(row.get("iql_profit_factor"), row.get("iql_net_sum_bps"))
        rows.append(
            {
                "cube": str(row.get("cube")),
                "slice": str(row.get("slice")),
                "class": "edge" if str(row.get("cube")) in EDGE_CUBES else "diagnostic",
                "candidate_n_trades": cand_n,
                "iql_n_trades": iql_n,
                "supported": bool(cand_n >= min_slice_trades and iql_n >= min_slice_trades),
                "low_support_reason": (
                    ""
                    if cand_n >= min_slice_trades and iql_n >= min_slice_trades
                    else f"candidate_n={cand_n}, iql_n={iql_n}, min={min_slice_trades}"
                ),
                "candidate_net_sum_bps": _num(row, "candidate_net_sum_bps"),
                "iql_net_sum_bps": _num(row, "iql_net_sum_bps"),
                "net_sum_delta_bps": _num(row, "iql_net_sum_bps") - _num(row, "candidate_net_sum_bps"),
                "candidate_net_mean_bps": _num(row, "candidate_net_mean_bps"),
                "iql_net_mean_bps": _num(row, "iql_net_mean_bps"),
                "net_mean_delta_bps": _num(row, "iql_net_mean_bps") - _num(row, "candidate_net_mean_bps"),
                "candidate_profit_factor_for_check": cand_pf,
                "iql_profit_factor_for_check": iql_pf,
                "profit_factor_delta": iql_pf - cand_pf if np.isfinite(cand_pf) and np.isfinite(iql_pf) else None,
                "candidate_max_drawdown_bps": _num(row, "candidate_max_drawdown_bps"),
                "iql_max_drawdown_bps": _num(row, "iql_max_drawdown_bps"),
                "max_drawdown_delta_bps": _num(row, "iql_max_drawdown_bps") - _num(row, "candidate_max_drawdown_bps"),
                "candidate_max_loss_bps": _num(row, "candidate_max_loss_bps"),
                "iql_max_loss_bps": _num(row, "iql_max_loss_bps"),
                "max_loss_worsening_bps": _num(row, "candidate_max_loss_bps") - _num(row, "iql_max_loss_bps"),
                "candidate_p10_net_bps": _num(row, "candidate_p10_net_bps"),
                "iql_p10_net_bps": _num(row, "iql_p10_net_bps"),
                "p10_net_delta_bps": _num(row, "iql_p10_net_bps") - _num(row, "candidate_p10_net_bps"),
                "candidate_mean_mae_bps": _num(row, "candidate_mean_mae_bps"),
                "iql_mean_mae_bps": _num(row, "iql_mean_mae_bps"),
                "mean_mae_delta_bps": _num(row, "iql_mean_mae_bps") - _num(row, "candidate_mean_mae_bps"),
                "candidate_p90_mae_bps": _num(row, "candidate_p90_mae_bps"),
                "iql_p90_mae_bps": _num(row, "iql_p90_mae_bps"),
                "p90_mae_delta_bps": _num(row, "iql_p90_mae_bps") - _num(row, "candidate_p90_mae_bps"),
            }
        )
    return pd.DataFrame(rows)


def _coverage(model: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cube, column in CUBE_COLUMNS.items():
        counts = df[column].value_counts(dropna=False).to_dict() if column in df.columns else {}
        unknown = int(counts.get("UNKNOWN", 0))
        rows.append(
            {
                "model": model,
                "cube": cube,
                "total": int(len(df)),
                "unknown": unknown,
                "non_unknown": int(len(df) - unknown),
                "distinct_non_unknown": int(len([key for key in counts if str(key) != "UNKNOWN"])),
                "counts": {str(key): int(value) for key, value in counts.items()},
            }
        )
    return rows


def _exit_opportunity_row(model: str, scope: str, value: str, group: pd.DataFrame) -> dict[str, Any]:
    pnl = _numeric(group, "net_pnl_bps").fillna(0.0)
    mfe = _numeric(group, "mfe_bps").fillna(0.0)
    mae = _numeric(group, "mae_bps").fillna(0.0)
    held = _numeric(group, "held_bars").dropna()
    positive_mfe = mfe > 0.0
    capture_ratio = (pnl.clip(lower=0.0) / mfe.where(positive_mfe)).replace([np.inf, -np.inf], np.nan)
    giveback = (mfe - pnl).clip(lower=0.0)
    peak_oracle_lift = giveback
    capture_75_lift = ((0.75 * mfe) - pnl).clip(lower=0.0)
    capture_50_lift = ((0.50 * mfe) - pnl).clip(lower=0.0)
    exit_reason = _category_series(group, "exit_reason")
    stop_loss = exit_reason == "STOP_LOSS"
    horizon = exit_reason == "HORIZON"
    take_profit = exit_reason == "TAKE_PROFIT"
    return {
        "model": model,
        "scope": scope,
        "value": value,
        "n_trades": int(len(group)),
        "net_sum_bps": float(pnl.sum()),
        "mfe_sum_bps": float(mfe.sum()),
        "mae_sum_bps": float(mae.sum()),
        "mean_mfe_capture_ratio": float(capture_ratio.mean()) if capture_ratio.notna().any() else None,
        "median_mfe_capture_ratio": float(capture_ratio.median()) if capture_ratio.notna().any() else None,
        "mean_giveback_bps": float(giveback.mean()) if len(giveback) else 0.0,
        "p75_giveback_bps": float(giveback.quantile(0.75)) if len(giveback) else 0.0,
        "p90_giveback_bps": float(giveback.quantile(0.90)) if len(giveback) else 0.0,
        "peak_oracle_lift_sum_bps": float(peak_oracle_lift.sum()),
        "capture_75_lift_sum_bps": float(capture_75_lift.sum()),
        "capture_50_lift_sum_bps": float(capture_50_lift.sum()),
        "stop_loss_count": int(stop_loss.sum()),
        "stop_loss_with_positive_mfe_count": int((stop_loss & positive_mfe).sum()),
        "horizon_count": int(horizon.sum()),
        "horizon_p75_giveback_bps": float(giveback[horizon].quantile(0.75)) if bool(horizon.any()) else 0.0,
        "take_profit_count": int(take_profit.sum()),
        "take_profit_p75_mfe_slack_bps": (
            float((mfe[take_profit] - pnl[take_profit]).clip(lower=0.0).quantile(0.75))
            if bool(take_profit.any())
            else 0.0
        ),
        "mean_held_bars": float(held.mean()) if not held.empty else None,
    }


def _exit_opportunity(trades_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, df in trades_by_model.items():
        rows.append(_exit_opportunity_row(model, "ALL", "ALL", df))
        for scope, column in (
            ("exit_reason", "exit_reason_slice"),
            ("session", "session_slice"),
            ("vol_regime", "vol_regime_slice"),
            ("side", "side_slice"),
        ):
            for value, group in df.groupby(column, dropna=False, sort=True):
                rows.append(_exit_opportunity_row(model, scope, str(value), group))
    return pd.DataFrame(rows)


def _tail_path_summary(model: str, trades: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    missing_columns = sorted(TAIL_PATH_REQUIRED_COLUMNS - set(str(column) for column in trades.columns))
    numeric_nonfinite: dict[str, int] = {}
    for column in sorted(TAIL_PATH_NUMERIC_COLUMNS & set(str(column) for column in trades.columns)):
        values = pd.to_numeric(trades[column], errors="coerce")
        numeric_nonfinite[column] = int(values.isna().sum())

    pnl = _numeric(trades, "net_pnl_bps").fillna(0.0)
    mfe = _numeric(trades, "mfe_bps").fillna(0.0)
    exit_reason = _category_series(trades, "exit_reason")
    stop_loss = exit_reason == "STOP_LOSS"
    positive_mfe = mfe > 0.0
    stop_loss_count = int(stop_loss.sum())
    trade_count = int(len(trades))
    stop_loss_rate = float(stop_loss_count / trade_count) if trade_count else 0.0
    stop_loss_positive_mfe_count = int((stop_loss & positive_mfe).sum())
    stop_loss_positive_mfe_rate = (
        float(stop_loss_positive_mfe_count / stop_loss_count) if stop_loss_count else 0.0
    )
    tail_count = max(1, int(np.ceil(trade_count * 0.05))) if trade_count else 0
    tail_p05 = pnl.sort_values(kind="mergesort").head(tail_count) if tail_count else pd.Series(dtype="float64")
    supported_slice_failures: list[dict[str, Any]] = []
    for scope, column in (
        ("session", "session_slice"),
        ("vol_regime", "vol_regime_slice"),
        ("side", "side_slice"),
    ):
        if column not in trades.columns:
            continue
        for value, group in trades.groupby(column, dropna=False, sort=True):
            if len(group) < int(args.min_slice_trades):
                continue
            group_exit = _category_series(group, "exit_reason")
            group_stop_rate = float((group_exit == "STOP_LOSS").mean()) if len(group) else 0.0
            if group_stop_rate > float(args.max_supported_slice_stop_loss_rate):
                supported_slice_failures.append(
                    {
                        "scope": scope,
                        "slice": str(value),
                        "n_trades": int(len(group)),
                        "stop_loss_rate": group_stop_rate,
                        "threshold": float(args.max_supported_slice_stop_loss_rate),
                    }
                )

    failures: list[dict[str, Any]] = []
    if missing_columns:
        failures.append({"reason": "missing_tail_path_columns", "columns": missing_columns})
    bad_numeric = {key: value for key, value in numeric_nonfinite.items() if value > 0}
    if bad_numeric:
        failures.append({"reason": "nonfinite_tail_path_columns", "columns": bad_numeric})
    max_loss_bps = float(pnl.min()) if len(pnl) else None
    tail_p05_mean_bps = float(tail_p05.mean()) if len(tail_p05) else None
    if max_loss_bps is None or max_loss_bps < -float(args.max_abs_replay_loss_bps):
        failures.append(
            {
                "reason": "max_loss_bps",
                "max_loss_bps": max_loss_bps,
                "threshold": -float(args.max_abs_replay_loss_bps),
            }
        )
    if stop_loss_rate > float(args.max_total_stop_loss_rate):
        failures.append(
            {
                "reason": "stop_loss_rate",
                "stop_loss_rate": stop_loss_rate,
                "threshold": float(args.max_total_stop_loss_rate),
            }
        )
    if stop_loss_count and stop_loss_positive_mfe_rate > float(args.max_stop_loss_positive_mfe_rate):
        failures.append(
            {
                "reason": "stop_loss_positive_mfe_rate",
                "stop_loss_positive_mfe_rate": stop_loss_positive_mfe_rate,
                "threshold": float(args.max_stop_loss_positive_mfe_rate),
            }
        )
    if tail_p05_mean_bps is None or tail_p05_mean_bps < -float(args.max_tail_loss_p05_abs_mean_bps):
        failures.append(
            {
                "reason": "tail_loss_p05_mean_bps",
                "tail_loss_p05_mean_bps": tail_p05_mean_bps,
                "threshold": -float(args.max_tail_loss_p05_abs_mean_bps),
            }
        )
    if supported_slice_failures:
        failures.append(
            {
                "reason": "supported_slice_stop_loss_rate",
                "failures": supported_slice_failures[:25],
                "failure_count": len(supported_slice_failures),
            }
        )
    return {
        "model": model,
        "n_trades": trade_count,
        "missing_columns": missing_columns,
        "numeric_nonfinite": numeric_nonfinite,
        "max_loss_bps": max_loss_bps,
        "tail_loss_p05_count": int(len(tail_p05)),
        "tail_loss_p05_mean_bps": tail_p05_mean_bps,
        "stop_loss_count": stop_loss_count,
        "stop_loss_rate": stop_loss_rate,
        "stop_loss_with_positive_mfe_count": stop_loss_positive_mfe_count,
        "stop_loss_with_positive_mfe_rate": stop_loss_positive_mfe_rate,
        "supported_slice_stop_loss_failures": supported_slice_failures,
        "failures": failures,
        "ready": not failures,
    }


def _path_signal_calibration(trades_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, trades in trades_by_model.items():
        if trades.empty:
            continue
        exit_reason = _category_series(trades, "exit_reason")
        enriched = trades.copy()
        enriched["_is_stop_loss"] = (exit_reason == "STOP_LOSS").astype(float)
        for signal in ("path_quality_pred", "bad_path_prob"):
            if signal not in enriched.columns:
                continue
            signal_values = pd.to_numeric(enriched[signal], errors="coerce")
            valid = enriched.loc[signal_values.notna()].copy()
            if valid.empty:
                continue
            valid["_signal"] = pd.to_numeric(valid[signal], errors="coerce")
            bins = min(10, int(len(valid)))
            valid["_decile"] = pd.qcut(
                valid["_signal"].rank(method="first"),
                q=bins,
                labels=False,
                duplicates="drop",
            ).astype(int) + 1
            for decile, group in valid.groupby("_decile", sort=True):
                pnl = _numeric(group, "net_pnl_bps")
                mae = _numeric(group, "mae_bps")
                mfe = _numeric(group, "mfe_bps")
                rows.append(
                    {
                        "model": model,
                        "signal": signal,
                        "decile": int(decile),
                        "n_trades": int(len(group)),
                        "signal_min": float(group["_signal"].min()),
                        "signal_max": float(group["_signal"].max()),
                        "signal_mean": float(group["_signal"].mean()),
                        "net_sum_bps": float(pnl.sum()),
                        "net_mean_bps": float(pnl.mean()) if len(pnl) else None,
                        "stop_loss_rate": float(group["_is_stop_loss"].mean()) if len(group) else None,
                        "mean_mae_bps": float(mae.mean()) if len(mae) else None,
                        "mean_mfe_bps": float(mfe.mean()) if len(mfe) else None,
                    }
                )
    return pd.DataFrame(rows)


def _path_signal_calibration_summary(calibration: pd.DataFrame) -> list[dict[str, Any]]:
    if calibration.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (model, signal), group in calibration.groupby(["model", "signal"], sort=True):
        ordered = group.sort_values("decile")
        decile = pd.to_numeric(ordered["decile"], errors="coerce")
        stop_loss = pd.to_numeric(ordered["stop_loss_rate"], errors="coerce")
        net_mean = pd.to_numeric(ordered["net_mean_bps"], errors="coerce")
        stop_corr = (
            float(decile.corr(stop_loss, method="spearman"))
            if stop_loss.notna().sum() >= 2 and stop_loss.nunique(dropna=True) >= 2 and decile.nunique(dropna=True) >= 2
            else None
        )
        net_corr = (
            float(decile.corr(net_mean, method="spearman"))
            if net_mean.notna().sum() >= 2 and net_mean.nunique(dropna=True) >= 2 and decile.nunique(dropna=True) >= 2
            else None
        )
        expected_stop_corr_sign = "negative" if signal == "path_quality_pred" else "positive"
        expected_net_corr_sign = "positive" if signal == "path_quality_pred" else "negative"
        stop_direction_ok = (
            stop_corr is not None
            and (stop_corr <= 0.0 if expected_stop_corr_sign == "negative" else stop_corr >= 0.0)
        )
        net_direction_ok = (
            net_corr is not None
            and (net_corr >= 0.0 if expected_net_corr_sign == "positive" else net_corr <= 0.0)
        )
        rows.append(
            {
                "model": str(model),
                "signal": str(signal),
                "decile_count": int(len(ordered)),
                "stop_loss_rate_spearman_vs_decile": stop_corr,
                "net_mean_spearman_vs_decile": net_corr,
                "expected_stop_corr_sign": expected_stop_corr_sign,
                "expected_net_corr_sign": expected_net_corr_sign,
                "stop_direction_ok": bool(stop_direction_ok),
                "net_direction_ok": bool(net_direction_ok),
            }
        )
    return rows


def _path_signal_calibration_failures(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    expected = {
        ("candidate", "bad_path_prob"),
        ("candidate", "path_quality_pred"),
        ("iql", "bad_path_prob"),
        ("iql", "path_quality_pred"),
    }
    seen = {(str(row.get("model")), str(row.get("signal"))) for row in summary}
    for model, signal in sorted(expected - seen):
        failures.append({"model": model, "signal": signal, "reason": "missing_calibration_summary"})

    for row in summary:
        decile_count = int(row.get("decile_count") or 0)
        if decile_count < 3:
            failures.append(
                {
                    "model": row.get("model"),
                    "signal": row.get("signal"),
                    "reason": "insufficient_calibration_deciles",
                    "decile_count": decile_count,
                }
            )
            continue
        if not bool(row.get("net_direction_ok")):
            failures.append(
                {
                    "model": row.get("model"),
                    "signal": row.get("signal"),
                    "reason": "net_direction_calibration_wrong_sign",
                    "net_mean_spearman_vs_decile": row.get("net_mean_spearman_vs_decile"),
                    "expected_net_corr_sign": row.get("expected_net_corr_sign"),
                }
            )
        if row.get("stop_loss_rate_spearman_vs_decile") is not None and not bool(row.get("stop_direction_ok")):
            failures.append(
                {
                    "model": row.get("model"),
                    "signal": row.get("signal"),
                    "reason": "stop_loss_calibration_wrong_sign",
                    "stop_loss_rate_spearman_vs_decile": row.get("stop_loss_rate_spearman_vs_decile"),
                    "expected_stop_corr_sign": row.get("expected_stop_corr_sign"),
                }
            )
    return failures


def _edge_failures(comparison: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if comparison.empty:
        return failures
    supported = comparison[(comparison["class"] == "edge") & (comparison["supported"])]
    for _, row in supported.iterrows():
        reasons: list[str] = []
        if _num(row, "iql_net_sum_bps") <= float(args.min_iql_edge_net_bps):
            reasons.append("net_sum")
        if _num_allow_inf(row, "iql_profit_factor_for_check") < float(args.min_iql_edge_profit_factor):
            reasons.append("profit_factor")
        if _num(row, "max_drawdown_delta_bps") > float(args.max_slice_drawdown_worsening_bps):
            reasons.append("drawdown")
        if _num(row, "max_loss_worsening_bps") > float(args.max_max_loss_worsening_bps):
            reasons.append("max_loss")
        if reasons:
            failures.append({"cube": row["cube"], "slice": row["slice"], "reasons": reasons, "row": row.to_dict()})
    return failures


def _diagnostic_failures(comparison: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if comparison.empty:
        return failures
    supported = comparison[(comparison["class"] == "diagnostic") & (comparison["supported"])]
    for _, row in supported.iterrows():
        reasons: list[str] = []
        if _num(row, "net_mean_delta_bps") < -float(args.max_diagnostic_mean_degradation_bps):
            reasons.append("mean_net_degradation")
        if _num(row, "p10_net_delta_bps") < -float(args.max_tail_p10_degradation_bps):
            reasons.append("p10_tail_degradation")
        if _num(row, "max_loss_worsening_bps") > float(args.max_diagnostic_max_loss_worsening_bps):
            reasons.append("max_loss")
        if reasons:
            failures.append({"cube": row["cube"], "slice": row["slice"], "reasons": reasons, "row": row.to_dict()})
    return failures


def _candidate_vs_iql_regression_disclosure(comparison: pd.DataFrame) -> dict[str, Any]:
    if comparison.empty:
        return {
            "diagnostic_only_not_gate": True,
            "supported_edge_regression_count": 0,
            "supported_diagnostic_regression_count": 0,
            "worst_supported_edge_net_regressions": [],
            "worst_supported_diagnostic_net_regressions": [],
            "worst_supported_drawdown_regressions": [],
            "worst_supported_mae_regressions": [],
        }

    supported = comparison[comparison["supported"]].copy()
    edge = supported[supported["class"] == "edge"].copy()
    diagnostic = supported[supported["class"] == "diagnostic"].copy()
    edge_net = edge[edge["net_sum_delta_bps"] < 0.0].sort_values("net_sum_delta_bps", kind="mergesort")
    diagnostic_net = diagnostic[diagnostic["net_sum_delta_bps"] < 0.0].sort_values(
        "net_sum_delta_bps",
        kind="mergesort",
    )
    drawdown = supported[supported["max_drawdown_delta_bps"] > 0.0].sort_values(
        "max_drawdown_delta_bps",
        ascending=False,
        kind="mergesort",
    )
    mae = supported[supported["p90_mae_delta_bps"] > 0.0].sort_values(
        "p90_mae_delta_bps",
        ascending=False,
        kind="mergesort",
    )

    keep_columns = [
        "cube",
        "slice",
        "class",
        "candidate_n_trades",
        "iql_n_trades",
        "candidate_net_sum_bps",
        "iql_net_sum_bps",
        "net_sum_delta_bps",
        "candidate_net_mean_bps",
        "iql_net_mean_bps",
        "net_mean_delta_bps",
        "candidate_profit_factor_for_check",
        "iql_profit_factor_for_check",
        "profit_factor_delta",
        "candidate_max_drawdown_bps",
        "iql_max_drawdown_bps",
        "max_drawdown_delta_bps",
        "candidate_p90_mae_bps",
        "iql_p90_mae_bps",
        "p90_mae_delta_bps",
    ]

    def records(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
        return frame[[column for column in keep_columns if column in frame.columns]].head(limit).to_dict("records")

    return {
        "diagnostic_only_not_gate": True,
        "interpretation": (
            "PASS means IQL kept required supported edge/tail gates alive. These rows disclose where IQL "
            "still underperforms candidate on supported slices so broad averages cannot hide weak behavior."
        ),
        "supported_edge_regression_count": int(len(edge_net)),
        "supported_diagnostic_regression_count": int(len(diagnostic_net)),
        "supported_drawdown_regression_count": int(len(drawdown)),
        "supported_p90_mae_regression_count": int(len(mae)),
        "worst_supported_edge_net_regressions": records(edge_net),
        "worst_supported_diagnostic_net_regressions": records(diagnostic_net),
        "worst_supported_drawdown_regressions": records(drawdown),
        "worst_supported_mae_regressions": records(mae),
    }


def _write_markdown(path: Path, report: dict[str, Any], comparison: pd.DataFrame, exit_opportunity: pd.DataFrame) -> None:
    lines = [
        "# Entry IQL Replay Slice Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        f"- Candidate trades: `{report['candidate_trade_count']}`",
        f"- IQL trades: `{report['iql_trade_count']}`",
        "",
        "## Supported Edge Slices",
        "",
    ]
    if comparison.empty:
        lines.append("- None")
    else:
        edge = comparison[(comparison["class"] == "edge") & (comparison["supported"])].sort_values(
            ["cube", "slice"]
        )
        for _, row in edge.iterrows():
            lines.append(
                "- "
                f"`{row['cube']}={row['slice']}` "
                f"iql_net=`{row['iql_net_sum_bps']}` "
                f"iql_pf=`{row['iql_profit_factor_for_check']}` "
                f"dd_delta=`{row['max_drawdown_delta_bps']}`"
            )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Candidate-vs-IQL Regression Disclosure", ""])
    disclosure = report.get("candidate_vs_iql_regression_disclosure") or {}
    lines.append(
        "- "
        f"Supported edge regressions: `{disclosure.get('supported_edge_regression_count', 0)}`; "
        f"diagnostic regressions: `{disclosure.get('supported_diagnostic_regression_count', 0)}`; "
        f"drawdown regressions: `{disclosure.get('supported_drawdown_regression_count', 0)}`; "
        f"p90 MAE regressions: `{disclosure.get('supported_p90_mae_regression_count', 0)}`"
    )
    for row in disclosure.get("worst_supported_edge_net_regressions", [])[:10]:
        lines.append(
            "- "
            f"`{row.get('cube')}={row.get('slice')}` "
            f"net_delta=`{row.get('net_sum_delta_bps')}` "
            f"iql_net=`{row.get('iql_net_sum_bps')}` "
            f"candidate_net=`{row.get('candidate_net_sum_bps')}`"
        )
    lines.extend(["", "## Exit Opportunity", ""])
    if exit_opportunity.empty:
        lines.append("- None")
    else:
        all_rows = exit_opportunity[exit_opportunity["scope"] == "ALL"].sort_values("model")
        for _, row in all_rows.iterrows():
            lines.append(
                "- "
                f"`{row['model']}` "
                f"capture_mean=`{row['mean_mfe_capture_ratio']}` "
                f"giveback_p90=`{row['p90_giveback_bps']}` "
                f"peak_oracle_lift=`{row['peak_oracle_lift_sum_bps']}`"
            )
    lines.extend(["", "## Weakest Supported IQL Means", ""])
    if not comparison.empty:
        weakest = comparison[comparison["supported"]].sort_values("iql_net_mean_bps").head(10)
        for _, row in weakest.iterrows():
            lines.append(
                "- "
                f"`{row['cube']}={row['slice']}` "
                f"iql_mean=`{row['iql_net_mean_bps']}` "
                f"candidate_mean=`{row['candidate_net_mean_bps']}` "
                f"delta=`{row['net_mean_delta_bps']}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_replay_dir = Path(args.candidate_replay_dir).expanduser().resolve()
    iql_replay_dir = Path(args.iql_replay_dir).expanduser().resolve()
    candidate_manifest_path = candidate_replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    iql_manifest_path = iql_replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    comparison_path = Path(args.comparison_json).expanduser().resolve()
    candidate_manifest = _read_json_or_empty(candidate_manifest_path)
    iql_manifest = _read_json_or_empty(iql_manifest_path)
    comparison_report = _read_json_or_empty(comparison_path)

    candidate_trades_path = _path_or_default(args.candidate_trades_path, _manifest_trades_path(candidate_manifest, candidate_replay_dir))
    iql_trades_path = _path_or_default(args.iql_trades_path, _manifest_trades_path(iql_manifest, iql_replay_dir))
    candidate_raw = _read_csv_or_empty(candidate_trades_path)
    iql_raw = _read_csv_or_empty(iql_trades_path)

    bad_path_thresholds = _tail_thresholds(candidate_raw, iql_raw, "bad_path_prob")
    mae_thresholds = _tail_thresholds(candidate_raw, iql_raw, "mae_bps")
    candidate = _normalize_trades(
        candidate_raw,
        model="candidate",
        bad_path_thresholds=bad_path_thresholds,
        mae_thresholds=mae_thresholds,
    )
    iql = _normalize_trades(
        iql_raw,
        model="iql",
        bad_path_thresholds=bad_path_thresholds,
        mae_thresholds=mae_thresholds,
    )
    metrics = _slice_metrics(candidate, iql)
    comparison = _comparison(metrics, min_slice_trades=int(args.min_slice_trades))
    exit_opportunity = _exit_opportunity({"candidate": candidate, "iql": iql})
    path_signal_calibration = _path_signal_calibration({"candidate": candidate, "iql": iql})
    path_signal_calibration_summary = _path_signal_calibration_summary(path_signal_calibration)
    path_signal_calibration_failures = _path_signal_calibration_failures(path_signal_calibration_summary)
    tail_path_quality = {
        "candidate": _tail_path_summary("candidate", candidate, args),
        "iql": _tail_path_summary("iql", iql, args),
    }
    tail_path_failures = [
        {"model": model, "failures": summary["failures"]}
        for model, summary in tail_path_quality.items()
        if summary["failures"]
    ]
    edge_failures = _edge_failures(comparison, args)
    diagnostic_failures = _diagnostic_failures(comparison, args)
    candidate_vs_iql_regression_disclosure = _candidate_vs_iql_regression_disclosure(comparison)
    coverage = _coverage("candidate", candidate) + _coverage("iql", iql)
    coverage_failures = [
        row
        for row in coverage
        if row["non_unknown"] <= 0 or row["distinct_non_unknown"] <= 0
    ]
    supported_edge_counts = {
        cube: int(len(comparison[(comparison["cube"] == cube) & (comparison["supported"])]))
        for cube in EDGE_CUBES
    } if not comparison.empty else {cube: 0 for cube in EDGE_CUBES}
    low_support = comparison[~comparison["supported"]].to_dict("records") if not comparison.empty else []

    checks = [
        _check(
            "candidate replay manifest is PASS",
            str(candidate_manifest.get("decision")) == "PASS",
            {"path": str(candidate_manifest_path), "decision": candidate_manifest.get("decision")},
        ),
        _check(
            "IQL replay manifest is PASS",
            str(iql_manifest.get("decision")) == "PASS",
            {"path": str(iql_manifest_path), "decision": iql_manifest.get("decision")},
        ),
        _check(
            "IQL comparison gate is ready before slice audit",
            str(comparison_report.get("decision")) == "READY_FOR_PROMOTION_REVIEW_VEDTAK",
            {"path": str(comparison_path), "decision": comparison_report.get("decision")},
        ),
        _check("candidate replay trades are present", not candidate_raw.empty, {"path": str(candidate_trades_path)}),
        _check("IQL replay trades are present", not iql_raw.empty, {"path": str(iql_trades_path)}),
        _check("slice metrics were produced", not metrics.empty, {"rows": int(len(metrics))}),
        _check("slice comparison rows were produced", not comparison.empty, {"rows": int(len(comparison))}),
        _check(
            "exit opportunity diagnostics were produced from replay MFE/MAE/held bars",
            not exit_opportunity.empty,
            {"rows": int(len(exit_opportunity))},
        ),
        _check(
            "candidate and IQL tail/path quality hard checks pass",
            not tail_path_failures,
            {
                "failures": tail_path_failures,
                "thresholds": {
                    "max_total_stop_loss_rate": float(args.max_total_stop_loss_rate),
                    "max_supported_slice_stop_loss_rate": float(args.max_supported_slice_stop_loss_rate),
                    "max_stop_loss_positive_mfe_rate": float(args.max_stop_loss_positive_mfe_rate),
                    "max_abs_replay_loss_bps": float(args.max_abs_replay_loss_bps),
                    "max_tail_loss_p05_abs_mean_bps": float(args.max_tail_loss_p05_abs_mean_bps),
                },
            },
        ),
        _check(
            "candidate and IQL path signal calibration has expected direction",
            not path_signal_calibration_failures,
            {
                "failures": path_signal_calibration_failures[:25],
                "failure_count": len(path_signal_calibration_failures),
            },
        ),
        _check("session/regime/side/direction/bad-path/tail coverage is live", not coverage_failures, {"failures": coverage_failures}),
        _check(
            "supported edge cubes exist for session/regime/side",
            all(count > 0 for count in supported_edge_counts.values()),
            {"supported_edge_counts": supported_edge_counts, "min_slice_trades": int(args.min_slice_trades)},
        ),
        _check(
            "IQL supported edge slices keep positive net/PF/drawdown/max-loss",
            not edge_failures,
            {"failures": edge_failures[:25], "failure_count": len(edge_failures)},
        ),
        _check(
            "IQL diagnostic slices do not materially worsen tails vs candidate",
            not diagnostic_failures,
            {"failures": diagnostic_failures[:25], "failure_count": len(diagnostic_failures)},
        ),
        _check(
            "slice audit never trains, replays, builds adapters, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "adapter_built": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
        _check(
            "upstream replay evidence keeps promotion/shadow/live closed",
            bool(candidate_manifest.get("promotion_shadow_live_allowed")) is False
            and bool(iql_manifest.get("promotion_shadow_live_allowed")) is False
            and bool(comparison_report.get("promotion_shadow_live_allowed")) is False,
            {
                "candidate": candidate_manifest.get("promotion_shadow_live_allowed"),
                "iql": iql_manifest.get("promotion_shadow_live_allowed"),
                "comparison": comparison_report.get("promotion_shadow_live_allowed"),
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_csv = out_dir / "entry_iql_replay_slice_metrics.csv"
    comparison_csv = out_dir / "entry_iql_replay_slice_comparison.csv"
    exit_opportunity_csv = out_dir / "entry_iql_replay_exit_opportunity.csv"
    path_signal_calibration_csv = out_dir / "entry_iql_replay_path_signal_calibration.csv"
    metrics.to_csv(metrics_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)
    exit_opportunity.to_csv(exit_opportunity_csv, index=False)
    path_signal_calibration.to_csv(path_signal_calibration_csv, index=False)
    json_path = out_dir / f"ENTRY_IQL_REPLAY_SLICE_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_IQL_REPLAY_SLICE_AUDIT_{timestamp}.md"
    report = {
        "schema_version": "entry_iql_replay_slice_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if ready else "FAIL",
        "candidate_replay_dir": str(candidate_replay_dir),
        "iql_replay_dir": str(iql_replay_dir),
        "candidate_replay_manifest_json": str(candidate_manifest_path),
        "iql_replay_manifest_json": str(iql_manifest_path),
        "comparison_json": str(comparison_path),
        "candidate_trades_path": str(candidate_trades_path),
        "iql_trades_path": str(iql_trades_path),
        "candidate_trade_count": int(len(candidate_raw)),
        "iql_trade_count": int(len(iql_raw)),
        "tail_thresholds": {
            "bad_path_prob": bad_path_thresholds,
            "mae_bps": mae_thresholds,
        },
        "min_slice_trades": int(args.min_slice_trades),
        "thresholds": {
            "min_iql_edge_net_bps": float(args.min_iql_edge_net_bps),
            "min_iql_edge_profit_factor": float(args.min_iql_edge_profit_factor),
            "max_slice_drawdown_worsening_bps": float(args.max_slice_drawdown_worsening_bps),
            "max_max_loss_worsening_bps": float(args.max_max_loss_worsening_bps),
            "max_diagnostic_mean_degradation_bps": float(args.max_diagnostic_mean_degradation_bps),
            "max_tail_p10_degradation_bps": float(args.max_tail_p10_degradation_bps),
            "max_diagnostic_max_loss_worsening_bps": float(args.max_diagnostic_max_loss_worsening_bps),
            "max_total_stop_loss_rate": float(args.max_total_stop_loss_rate),
            "max_supported_slice_stop_loss_rate": float(args.max_supported_slice_stop_loss_rate),
            "max_stop_loss_positive_mfe_rate": float(args.max_stop_loss_positive_mfe_rate),
            "max_abs_replay_loss_bps": float(args.max_abs_replay_loss_bps),
            "max_tail_loss_p05_abs_mean_bps": float(args.max_tail_loss_p05_abs_mean_bps),
        },
        "tail_path_quality": tail_path_quality,
        "tail_path_failures": tail_path_failures,
        "supported_edge_counts": supported_edge_counts,
        "coverage": coverage,
        "low_support_slice_count": int(len(low_support)),
        "low_support_slices": low_support[:50],
        "exit_opportunity_summary": {
            "csv": str(exit_opportunity_csv),
            "iql_all": (
                exit_opportunity[
                    (exit_opportunity["model"] == "iql") & (exit_opportunity["scope"] == "ALL")
                ].to_dict("records")[:1]
            ),
            "candidate_all": (
                exit_opportunity[
                    (exit_opportunity["model"] == "candidate") & (exit_opportunity["scope"] == "ALL")
                ].to_dict("records")[:1]
            ),
            "top_iql_peak_oracle_lift": (
                exit_opportunity[exit_opportunity["model"] == "iql"]
                .sort_values("peak_oracle_lift_sum_bps", ascending=False)
                .head(10)
                .to_dict("records")
            ),
            "interpretation": (
                "Hindsight opportunity only. These numbers estimate where an exit/hazard model might add value; "
                "they are not deployable policy evidence."
            ),
        },
        "path_signal_calibration": {
            "csv": str(path_signal_calibration_csv),
            "rows": int(len(path_signal_calibration)),
            "summary": path_signal_calibration_summary,
            "failures": path_signal_calibration_failures,
            "ready": not path_signal_calibration_failures,
            "diagnostic_only_not_gate": False,
            "promotion_review_gate": True,
        },
        "candidate_vs_iql_regression_disclosure": candidate_vs_iql_regression_disclosure,
        "edge_failures": edge_failures,
        "diagnostic_failures": diagnostic_failures,
        "checks": checks,
        "failures": failures,
        "slice_metrics_csv": str(metrics_csv),
        "slice_comparison_csv": str(comparison_csv),
        "exit_opportunity_csv": str(exit_opportunity_csv),
        "json_path": str(json_path),
        "md_path": str(md_path),
        "trainer_started": False,
        "replay_started": False,
        "adapter_built": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "manual review of slice/tail audit; shadow/live remain blocked"
            if ready
            else "repair weak slices or path-signal calibration before promotion review; shadow/live remain blocked"
        ),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report, comparison, exit_opportunity)
    (out_dir / "ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "json_path": str(json_path),
                    "slice_comparison_csv": str(comparison_csv),
                    "next_required_gate": report["next_required_gate"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-replay-dir", default=str(DEFAULT_CANDIDATE_REPLAY_DIR))
    ap.add_argument("--iql-replay-dir", default=str(DEFAULT_IQL_REPLAY_DIR))
    ap.add_argument("--comparison-json", default=str(DEFAULT_COMPARISON_JSON))
    ap.add_argument("--candidate-trades-path", default="")
    ap.add_argument("--iql-trades-path", default="")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-slice-trades", type=int, default=20)
    ap.add_argument("--min-iql-edge-net-bps", type=float, default=0.0)
    ap.add_argument("--min-iql-edge-profit-factor", type=float, default=1.0)
    ap.add_argument("--max-slice-drawdown-worsening-bps", type=float, default=120.0)
    ap.add_argument("--max-max-loss-worsening-bps", type=float, default=0.0)
    ap.add_argument("--max-diagnostic-mean-degradation-bps", type=float, default=10.0)
    ap.add_argument("--max-tail-p10-degradation-bps", type=float, default=20.0)
    ap.add_argument("--max-diagnostic-max-loss-worsening-bps", type=float, default=10.0)
    ap.add_argument("--max-total-stop-loss-rate", type=float, default=0.25)
    ap.add_argument("--max-supported-slice-stop-loss-rate", type=float, default=0.40)
    ap.add_argument("--max-stop-loss-positive-mfe-rate", type=float, default=0.70)
    ap.add_argument("--max-abs-replay-loss-bps", type=float, default=90.0)
    ap.add_argument("--max-tail-loss-p05-abs-mean-bps", type=float, default=90.0)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
