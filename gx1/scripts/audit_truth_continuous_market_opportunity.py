#!/usr/bin/env python3
"""
Audit truth replay roots for continuous market opportunity pressure.

This uses the per-bar replay eval logs to measure how much up/down opportunity
the market offered independent of whether the system actually took a trade.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
SUPPORTED_RUN_PREFIXES = ("E2E_SANITY_ORDERFIX_", "TRUTH_MONFRI_WEEK_")
RUN_ID_RE = re.compile(r"^(?:E2E_SANITY_ORDERFIX|TRUTH_MONFRI_WEEK)_(\d{8})_(\d{8})$")
DEFAULT_HORIZONS_BARS = (15, 60, 240)
DEFAULT_THRESHOLDS_BPS = (10.0, 25.0, 50.0, 100.0, 200.0)


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _runs_root(reports_root: Path) -> Path:
    candidate = reports_root / "runs"
    return candidate if candidate.exists() else reports_root


def _stats(series_like: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(series_like), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _parse_run_dates(run_id: str) -> Dict[str, Optional[str]]:
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None:
        return {"start_date": None, "end_date": None}
    start_ts = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(match.group(2), format="%Y%m%d", errors="coerce")
    return {
        "start_date": None if pd.isna(start_ts) else start_ts.date().isoformat(),
        "end_date": None if pd.isna(end_ts) else end_ts.date().isoformat(),
    }


def _load_eval_price_frame(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("replay/chunk_*/logs/eval_log_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                ts_utc = payload.get("ts_utc")
                price = payload.get("price")
                if ts_utc is None or price is None:
                    continue
                rows.append(
                    {
                        "ts_utc": ts_utc,
                        "price": price,
                        "session": payload.get("session"),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["ts_utc", "price", "session"])
    frame = pd.DataFrame(rows)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["ts_utc", "price"]).sort_values("ts_utc", kind="mergesort")
    frame = frame.drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
    return frame


def _forward_opportunity(price: pd.Series, horizon_bars: int) -> tuple[pd.Series, pd.Series]:
    rev = price.iloc[::-1]
    future_max = rev.shift(1).rolling(window=horizon_bars, min_periods=1).max().iloc[::-1]
    future_min = rev.shift(1).rolling(window=horizon_bars, min_periods=1).min().iloc[::-1]
    up_move_bps = ((future_max - price) / price * 1e4).clip(lower=0.0)
    down_move_bps = ((price - future_min) / price * 1e4).clip(lower=0.0)
    return up_move_bps, down_move_bps


def _backward_pressure(price: pd.Series, horizon_bars: int) -> tuple[pd.Series, pd.Series]:
    rolling_max = price.rolling(window=horizon_bars, min_periods=min(3, int(horizon_bars))).max()
    rolling_min = price.rolling(window=horizon_bars, min_periods=min(3, int(horizon_bars))).min()
    up_move_bps = ((price - rolling_min) / price * 1e4).clip(lower=0.0)
    down_move_bps = ((rolling_max - price) / price * 1e4).clip(lower=0.0)
    return up_move_bps, down_move_bps


def _group_summary(frame: pd.DataFrame, thresholds_bps: Sequence[float]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "rows": int(len(frame)),
        "up_move_bps": _stats(frame.get("up_move_bps", pd.Series(dtype="float64"))),
        "down_move_bps": _stats(frame.get("down_move_bps", pd.Series(dtype="float64"))),
        "best_move_bps": _stats(frame.get("best_move_bps", pd.Series(dtype="float64"))),
        "range_bps": _stats(frame.get("range_bps", pd.Series(dtype="float64"))),
        "backward_up_move_bps": _stats(frame.get("backward_up_move_bps", pd.Series(dtype="float64"))),
        "backward_down_move_bps": _stats(frame.get("backward_down_move_bps", pd.Series(dtype="float64"))),
        "backward_range_bps": _stats(frame.get("backward_range_bps", pd.Series(dtype="float64"))),
        "backward_directional_imbalance_bps": _stats(
            frame.get("backward_directional_imbalance_bps", pd.Series(dtype="float64"))
        ),
        "threshold_rates_bps": {},
    }
    best_move = pd.to_numeric(frame.get("best_move_bps"), errors="coerce")
    for threshold in thresholds_bps:
        payload["threshold_rates_bps"][str(int(threshold))] = float((best_move >= float(threshold)).mean()) if len(best_move) else 0.0
    return payload


def build_continuous_market_opportunity_summary(
    reports_root: Path,
    *,
    horizons_bars: Sequence[int] = DEFAULT_HORIZONS_BARS,
    thresholds_bps: Sequence[float] = DEFAULT_THRESHOLDS_BPS,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    runs_root = _runs_root(reports_root)
    run_dirs = sorted(
        [path for path in runs_root.iterdir() if path.is_dir() and RUN_ID_RE.fullmatch(path.name)],
        key=lambda path: path.name,
    )

    bar_level_rows: dict[int, list[pd.DataFrame]] = {int(h): [] for h in horizons_bars}
    per_run_rows: list[dict[str, Any]] = []
    missing_eval_runs: list[str] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        trade_path = run_dir / f"trade_outcomes_{run_id}_MERGED.parquet"
        trade_count = int(len(pd.read_parquet(trade_path, columns=["trade_id"]))) if trade_path.exists() else 0
        completed = (run_dir / "RUN_COMPLETED.json").exists()

        price_frame = _load_eval_price_frame(run_dir)
        if price_frame.empty:
            missing_eval_runs.append(run_id)
            continue

        price_series = price_frame["price"].astype("float64")
        for horizon_bars in horizons_bars:
            horizon = int(horizon_bars)
            up_move_bps, down_move_bps = _forward_opportunity(price_series, horizon)
            backward_up_move_bps, backward_down_move_bps = _backward_pressure(price_series, horizon)
            best_move_bps = np.maximum(up_move_bps, down_move_bps)
            range_bps = up_move_bps + down_move_bps
            backward_range_bps = backward_up_move_bps + backward_down_move_bps
            backward_directional_imbalance_bps = backward_up_move_bps - backward_down_move_bps

            horizon_frame = pd.DataFrame(
                {
                    "run_id": run_id,
                    "trade_count": trade_count,
                    "completed": completed,
                    "session": price_frame.get("session", pd.Series(dtype="string")).astype("string"),
                    "up_move_bps": up_move_bps,
                    "down_move_bps": down_move_bps,
                    "best_move_bps": best_move_bps,
                    "range_bps": range_bps,
                    "backward_up_move_bps": backward_up_move_bps,
                    "backward_down_move_bps": backward_down_move_bps,
                    "backward_range_bps": backward_range_bps,
                    "backward_directional_imbalance_bps": backward_directional_imbalance_bps,
                }
            )
            bar_level_rows[horizon].append(horizon_frame)

            run_row: dict[str, Any] = {
                "run_id": run_id,
                "completed": completed,
                "trade_count": trade_count,
                "price_rows": int(len(price_frame)),
                "horizon_bars": horizon,
                "up_move_mean_bps": float(pd.Series(up_move_bps).mean()),
                "down_move_mean_bps": float(pd.Series(down_move_bps).mean()),
                "best_move_mean_bps": float(pd.Series(best_move_bps).mean()),
                "best_move_median_bps": float(pd.Series(best_move_bps).median()),
                "best_move_p90_bps": float(pd.Series(best_move_bps).quantile(0.90)),
                "range_mean_bps": float(pd.Series(range_bps).mean()),
                "backward_range_mean_bps": float(pd.Series(backward_range_bps).mean()),
                "backward_directional_imbalance_mean_bps": float(pd.Series(backward_directional_imbalance_bps).mean()),
            }
            for threshold in thresholds_bps:
                run_row[f"rate_ge_{int(threshold)}bps"] = float((pd.Series(best_move_bps) >= float(threshold)).mean())
            run_row.update(_parse_run_dates(run_id))
            per_run_rows.append(run_row)

    per_run_df = pd.DataFrame(per_run_rows)
    completed_df = per_run_df.loc[per_run_df["completed"].fillna(False)].copy() if not per_run_df.empty else pd.DataFrame()
    completed_zero_df = completed_df.loc[completed_df["trade_count"].fillna(0).eq(0)].copy() if not completed_df.empty else pd.DataFrame()
    completed_nonzero_df = completed_df.loc[completed_df["trade_count"].fillna(0).gt(0)].copy() if not completed_df.empty else pd.DataFrame()

    overall_by_horizon: dict[str, Any] = {}
    session_summary_by_horizon: dict[str, Any] = {}
    for horizon in horizons_bars:
        horizon_key = str(int(horizon))
        frame = pd.concat(bar_level_rows[int(horizon)], ignore_index=True) if bar_level_rows[int(horizon)] else pd.DataFrame()
        overall_by_horizon[horizon_key] = _group_summary(frame, thresholds_bps)
        session_rows = []
        if not frame.empty:
            for session, part in frame.groupby("session", dropna=False):
                session_payload = _group_summary(part, thresholds_bps)
                session_payload["session"] = str(session)
                session_rows.append(session_payload)
        session_rows.sort(key=lambda row: (-row["rows"], row["session"]))
        session_summary_by_horizon[horizon_key] = session_rows

    anchor_horizon = 60 if 60 in [int(item) for item in horizons_bars] else int(horizons_bars[0])
    anchor_df = completed_df.loc[completed_df["horizon_bars"].eq(anchor_horizon)].copy() if not completed_df.empty else pd.DataFrame()
    anchor_zero_df = completed_zero_df.loc[completed_zero_df["horizon_bars"].eq(anchor_horizon)].copy() if not completed_zero_df.empty else pd.DataFrame()
    anchor_nonzero_df = completed_nonzero_df.loc[completed_nonzero_df["horizon_bars"].eq(anchor_horizon)].copy() if not completed_nonzero_df.empty else pd.DataFrame()

    zero_trade_anchor_comparison: Dict[str, Any] = {}
    opportunity_rich_zero_trade_runs: list[str] = []
    if not anchor_nonzero_df.empty:
        nonzero_median_best_mean = float(anchor_nonzero_df["best_move_mean_bps"].median())
        nonzero_median_rate_50 = float(anchor_nonzero_df["rate_ge_50bps"].median())
        zero_work = anchor_zero_df.copy() if not anchor_zero_df.empty else pd.DataFrame()
        if not zero_work.empty:
            zero_work["best_move_mean_ratio_to_nonzero_median"] = zero_work["best_move_mean_bps"] / nonzero_median_best_mean
            zero_work["rate_ge_50bps_ratio_to_nonzero_median"] = zero_work["rate_ge_50bps"] / nonzero_median_rate_50 if nonzero_median_rate_50 > 0 else np.nan
            opportunity_rich_zero_trade_runs = zero_work.loc[
                (zero_work["best_move_mean_ratio_to_nonzero_median"] >= 1.0)
                | (zero_work["rate_ge_50bps_ratio_to_nonzero_median"] >= 1.0),
                "run_id",
            ].astype("string").tolist()
            anchor_zero_df = zero_work

        zero_trade_anchor_comparison = {
            "anchor_horizon_bars": int(anchor_horizon),
            "nonzero_median_best_move_mean_bps": nonzero_median_best_mean,
            "nonzero_median_rate_ge_50bps": nonzero_median_rate_50,
            "zero_trade_best_move_mean_stats": _stats(anchor_zero_df.get("best_move_mean_bps", pd.Series(dtype="float64"))),
            "zero_trade_rate_ge_50bps_stats": _stats(anchor_zero_df.get("rate_ge_50bps", pd.Series(dtype="float64"))),
            "nonzero_best_move_mean_stats": _stats(anchor_nonzero_df.get("best_move_mean_bps", pd.Series(dtype="float64"))),
            "nonzero_rate_ge_50bps_stats": _stats(anchor_nonzero_df.get("rate_ge_50bps", pd.Series(dtype="float64"))),
        }

    outlook_v1 = "NO_MARKET_DATA"
    if overall_by_horizon:
        if opportunity_rich_zero_trade_runs:
            outlook_v1 = "LOWER_MEDIAN_OPPORTUNITY_WITH_MISSED_OUTLIERS"
        else:
            outlook_v1 = "LOWER_MEDIAN_OPPORTUNITY_IN_ZERO_TRADE_WEEKS"

    recommendations_v1: list[str] = []
    if opportunity_rich_zero_trade_runs:
        recommendations_v1.append(
            "Promote continuous market opportunity into the canonical entry scorecard so opportunity-rich zero-trade weeks are flagged even when trade_count stays at zero."
        )
    recommendations_v1.append(
        "Keep anchor-horizon opportunity comparisons at fixed horizons (15/60/240 bars) to avoid tuning to a specific bad week."
    )
    recommendations_v1.append(
        "Use continuous opportunity pressure together with hold-longer pressure so low capture can be separated from genuinely low-motion market windows."
    )

    summary = {
        "reports_root": str(reports_root),
        "runs_root": str(runs_root),
        "run_dir_count": int(len(run_dirs)),
        "completed_runs_with_market_data": int(completed_df["run_id"].nunique()) if not completed_df.empty else 0,
        "missing_eval_price_runs": missing_eval_runs,
        "horizons_bars": [int(item) for item in horizons_bars],
        "thresholds_bps": [float(item) for item in thresholds_bps],
        "outlook_v1": outlook_v1,
        "overall_by_horizon": overall_by_horizon,
        "session_summary_by_horizon": session_summary_by_horizon,
        "zero_trade_anchor_comparison_v1": zero_trade_anchor_comparison,
        "opportunity_rich_zero_trade_runs_anchor": opportunity_rich_zero_trade_runs,
        "opportunity_rich_zero_trade_run_details_anchor": (
            anchor_zero_df.loc[
                anchor_zero_df["run_id"].astype("string").isin(opportunity_rich_zero_trade_runs),
                [
                    "run_id",
                    "trade_count",
                    "best_move_mean_bps",
                    "best_move_median_bps",
                    "best_move_p90_bps",
                    "rate_ge_50bps",
                    "rate_ge_100bps",
                    "best_move_mean_ratio_to_nonzero_median",
                    "rate_ge_50bps_ratio_to_nonzero_median",
                    "start_date",
                    "end_date",
                ],
            ].sort_values(["best_move_mean_bps", "run_id"], ascending=[False, True], kind="mergesort").to_dict(orient="records")
            if opportunity_rich_zero_trade_runs and not anchor_zero_df.empty
            else []
        ),
        "top_zero_trade_runs_anchor": (
            anchor_zero_df.sort_values(["best_move_mean_bps", "run_id"], ascending=[False, True], kind="mergesort")
            .head(sample_limit)
            .to_dict(orient="records")
            if not anchor_zero_df.empty
            else []
        ),
        "top_nonzero_runs_anchor": (
            anchor_nonzero_df.sort_values(["best_move_mean_bps", "run_id"], ascending=[False, True], kind="mergesort")
            .head(sample_limit)
            .to_dict(orient="records")
            if not anchor_nonzero_df.empty
            else []
        ),
        "verdicts": {
            "continuous_market_data_status": "PASS" if len(missing_eval_runs) == 0 else "FAIL",
            "zero_trade_median_opportunity_gap_status": (
                "PASS"
                if not anchor_zero_df.empty
                and not anchor_nonzero_df.empty
                and float(anchor_zero_df["best_move_mean_bps"].median()) < float(anchor_nonzero_df["best_move_mean_bps"].median())
                else "FAIL"
            ),
            "zero_trade_opportunity_rich_outlier_status": "FAIL" if opportunity_rich_zero_trade_runs else "PASS",
        },
        "recommendations_v1": recommendations_v1,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a truth replay root for continuous market opportunity pressure.")
    parser.add_argument("--reports-root", help="Path to the truth replay root. Defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--sample-limit", type=int, default=10, help="How many top runs to keep in sample sections.")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    summary = build_continuous_market_opportunity_summary(
        reports_root,
        sample_limit=max(1, int(args.sample_limit)),
    )
    payload = json.dumps(summary, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
