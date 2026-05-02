#!/usr/bin/env python3
"""
Audit truth replay roots for trade foundation quality.

This summarizes trade-level quality without tuning to a specific week subset.
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


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _runs_root(reports_root: Path) -> Path:
    candidate = reports_root / "runs"
    return candidate if candidate.exists() else reports_root


def _stats(series_like: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(series_like), dtype="float64").dropna()
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


def _ordered_trade_frame(reports_root: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(
        [path for path in _runs_root(reports_root).iterdir() if path.is_dir() and RUN_ID_RE.fullmatch(path.name)]
    ):
        run_id = run_dir.name
        trade_path = run_dir / f"trade_outcomes_{run_id}_MERGED.parquet"
        if not trade_path.exists():
            continue
        df = pd.read_parquet(trade_path)
        if df.empty:
            continue
        df = df.copy()
        df["run_id"] = run_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    trade_df = pd.concat(frames, ignore_index=True)
    trade_df["close_ts_utc"] = pd.to_datetime(trade_df.get("close_ts_utc"), utc=True, errors="coerce")
    trade_df["open_ts_utc"] = pd.to_datetime(trade_df.get("open_ts_utc"), utc=True, errors="coerce")
    trade_df = trade_df.sort_values(
        ["close_ts_utc", "open_ts_utc", "run_id", "trade_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return trade_df


def _profit_factor(pnl: pd.Series) -> Optional[float]:
    wins = pnl.loc[pnl > 0.0].sum()
    losses = pnl.loc[pnl <= 0.0].sum()
    if losses == 0:
        return None
    return float(wins / abs(losses))


def _payoff_ratio(pnl: pd.Series) -> Optional[float]:
    wins = pnl.loc[pnl > 0.0]
    losses = pnl.loc[pnl <= 0.0]
    if wins.empty or losses.empty:
        return None
    mean_loss_abs = abs(float(losses.mean()))
    if mean_loss_abs == 0.0:
        return None
    return float(float(wins.mean()) / mean_loss_abs)


def build_trade_foundation_quality_summary(reports_root: Path, *, sample_limit: int = 10) -> Dict[str, Any]:
    trade_df = _ordered_trade_frame(reports_root)
    if trade_df.empty:
        return {
            "reports_root": str(reports_root),
            "trade_count": 0,
            "outlook_v1": "NO_TRADES",
            "verdicts": {
                "profitability_status": "FAIL",
                "drawdown_cover_status": "FAIL",
                "exit_efficiency_status": "FAIL",
            },
        }

    pnl = pd.to_numeric(trade_df.get("pnl_bps"), errors="coerce").fillna(0.0)
    mfe = pd.to_numeric(trade_df.get("mfe_bps"), errors="coerce")
    mae = pd.to_numeric(trade_df.get("mae_bps"), errors="coerce")
    post_exit_mfe = pd.to_numeric(trade_df.get("post_exit_mfe_bps"), errors="coerce")
    early_exit_regret = trade_df.get("early_exit_regret", pd.Series(False, index=trade_df.index)).fillna(False).astype(bool)
    capture_ratio = (pnl / mfe.where(mfe > 0.0)).replace([np.inf, -np.inf], np.nan)
    potential_hold_longer_total = pnl.fillna(0.0) + post_exit_mfe.fillna(0.0)
    winner_mask = pnl > 0.0
    loser_mask = ~winner_mask

    equity_curve = pnl.cumsum()
    running_max = equity_curve.cummax()
    drawdown_curve = equity_curve - running_max
    max_drawdown_bps = float(drawdown_curve.min()) if len(drawdown_curve) else 0.0
    total_pnl_bps = float(pnl.sum())
    profit_factor = _profit_factor(pnl)
    payoff_ratio = _payoff_ratio(pnl)
    calmar_like = float(total_pnl_bps / abs(max_drawdown_bps)) if max_drawdown_bps < 0.0 else None

    per_session_rows = []
    for session, part in trade_df.groupby("session", dropna=False):
        session_pnl = pd.to_numeric(part.get("pnl_bps"), errors="coerce").fillna(0.0)
        session_mfe = pd.to_numeric(part.get("mfe_bps"), errors="coerce")
        session_mae = pd.to_numeric(part.get("mae_bps"), errors="coerce")
        session_post_exit = pd.to_numeric(part.get("post_exit_mfe_bps"), errors="coerce")
        session_regret = part.get("early_exit_regret", pd.Series(False, index=part.index)).fillna(False).astype(bool)
        session_equity = session_pnl.cumsum()
        session_dd = float((session_equity - session_equity.cummax()).min()) if len(session_equity) else 0.0
        session_capture = (session_pnl / session_mfe.where(session_mfe > 0.0)).replace([np.inf, -np.inf], np.nan)
        per_session_rows.append(
            {
                "session": str(session),
                "trade_count": int(len(part)),
                "total_pnl_bps": float(session_pnl.sum()),
                "avg_pnl_bps": float(session_pnl.mean()),
                "median_pnl_bps": float(session_pnl.median()),
                "win_rate": float((session_pnl > 0.0).mean()),
                "profit_factor": _profit_factor(session_pnl),
                "payoff_ratio": _payoff_ratio(session_pnl),
                "max_drawdown_bps": session_dd,
                "mfe_mean_bps": float(session_mfe.mean()),
                "mae_mean_bps": float(session_mae.mean()),
                "post_exit_mfe_mean_bps": float(session_post_exit.mean()),
                "post_exit_mfe_median_bps": float(session_post_exit.median()),
                "hold_longer_rate_25bps": float((session_post_exit >= 25.0).mean()),
                "hold_longer_rate_50bps": float((session_post_exit >= 50.0).mean()),
                "regret_rate": float(session_regret.mean()),
                "capture_median": (float(session_capture.dropna().median()) if not session_capture.dropna().empty else None),
            }
        )
    per_session_rows.sort(key=lambda row: row["trade_count"], reverse=True)

    per_run_rows = []
    for run_id, part in trade_df.groupby("run_id", dropna=False):
        run_pnl = pd.to_numeric(part.get("pnl_bps"), errors="coerce").fillna(0.0)
        run_equity = run_pnl.cumsum()
        row = {
            "run_id": str(run_id),
            "trade_count": int(len(part)),
            "total_pnl_bps": float(run_pnl.sum()),
            "avg_pnl_bps": float(run_pnl.mean()),
            "max_drawdown_bps": float((run_equity - run_equity.cummax()).min()) if len(run_equity) else 0.0,
        }
        row.update(_parse_run_dates(str(run_id)))
        per_run_rows.append(row)
    per_run_df = pd.DataFrame(per_run_rows)

    clean_good_20_5 = (mfe >= 20.0) & (mae > -5.0)
    home_run_200 = mfe >= 200.0
    adverse_gt_favorable = mae.abs() > mfe
    mae_le_minus20 = mae <= -20.0
    mfe_lt_10 = mfe < 10.0
    post_exit_ge_10 = post_exit_mfe >= 10.0
    post_exit_ge_25 = post_exit_mfe >= 25.0
    post_exit_ge_50 = post_exit_mfe >= 50.0
    post_exit_ge_100 = post_exit_mfe >= 100.0
    post_exit_ge_200 = post_exit_mfe >= 200.0

    outlook_v1 = "MIXED_OR_WEAK"
    if total_pnl_bps > 0.0 and (profit_factor or 0.0) > 1.0:
        outlook_v1 = "POSITIVE_EDGE_HIGH_REGRET" if (calmar_like is None or calmar_like < 1.0 or float(early_exit_regret.mean()) >= 0.5) else "POSITIVE_EDGE_BALANCED"

    summary = {
        "reports_root": str(reports_root),
        "trade_count": int(len(trade_df)),
        "outlook_v1": outlook_v1,
        "profitability": {
            "total_pnl_bps": total_pnl_bps,
            "avg_pnl_bps": float(pnl.mean()),
            "median_pnl_bps": float(pnl.median()),
            "win_rate": float((pnl > 0.0).mean()),
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "avg_win_bps": (float(pnl.loc[pnl > 0.0].mean()) if (pnl > 0.0).any() else None),
            "avg_loss_bps": (float(pnl.loc[pnl <= 0.0].mean()) if (pnl <= 0.0).any() else None),
            "max_drawdown_bps": max_drawdown_bps,
            "calmar_like_total_over_abs_dd": calmar_like,
        },
        "trade_shape": {
            "pnl_bps": _stats(pnl),
            "mfe_bps": _stats(mfe),
            "mae_bps": _stats(mae),
            "post_exit_mfe_bps": _stats(post_exit_mfe),
        },
        "capture": {
            "all_capture_ratio": _stats(capture_ratio),
            "winner_capture_ratio": _stats(capture_ratio.loc[winner_mask]),
            "winner_mfe_bps": _stats(mfe.loc[winner_mask]),
            "loser_mfe_bps": _stats(mfe.loc[loser_mask]),
            "loser_mae_abs_bps": _stats(mae.loc[loser_mask].abs()),
        },
        "quality_flags": {
            "clean_good_trade_mfe20_mae5_count": int(clean_good_20_5.sum()),
            "clean_good_trade_mfe20_mae5_rate": float(clean_good_20_5.mean()),
            "home_run_200bps_count": int(home_run_200.sum()),
            "home_run_200bps_rate": float(home_run_200.mean()),
            "adverse_gt_favorable_count": int(adverse_gt_favorable.sum()),
            "adverse_gt_favorable_rate": float(adverse_gt_favorable.mean()),
            "mae_le_minus20_count": int(mae_le_minus20.sum()),
            "mfe_lt_10_count": int(mfe_lt_10.sum()),
        },
        "exit_efficiency": {
            "early_exit_regret_count": int(early_exit_regret.sum()),
            "early_exit_regret_rate": float(early_exit_regret.mean()),
            "winner_regret_rate": float(early_exit_regret.loc[winner_mask].mean()) if winner_mask.any() else None,
            "loser_regret_rate": float(early_exit_regret.loc[loser_mask].mean()) if loser_mask.any() else None,
        },
        "hold_longer_pressure": {
            "extra_value_bps": _stats(post_exit_mfe),
            "potential_total_bps_if_held": _stats(potential_hold_longer_total),
            "winner_extra_value_bps": _stats(post_exit_mfe.loc[winner_mask]),
            "loser_extra_value_bps": _stats(post_exit_mfe.loc[loser_mask]),
            "meaningful_extra_value_10bps_count": int(post_exit_ge_10.sum()),
            "meaningful_extra_value_10bps_rate": float(post_exit_ge_10.mean()),
            "meaningful_extra_value_25bps_count": int(post_exit_ge_25.sum()),
            "meaningful_extra_value_25bps_rate": float(post_exit_ge_25.mean()),
            "large_extra_value_50bps_count": int(post_exit_ge_50.sum()),
            "large_extra_value_50bps_rate": float(post_exit_ge_50.mean()),
            "extreme_extra_value_100bps_count": int(post_exit_ge_100.sum()),
            "extreme_extra_value_100bps_rate": float(post_exit_ge_100.mean()),
            "elite_extra_value_200bps_count": int(post_exit_ge_200.sum()),
            "elite_extra_value_200bps_rate": float(post_exit_ge_200.mean()),
        },
        "session_summary": per_session_rows,
        "worst_weeks_top10": (
            per_run_df.sort_values(["total_pnl_bps", "run_id"], ascending=[True, True], kind="mergesort")
            .head(sample_limit)
            .to_dict(orient="records")
            if not per_run_df.empty
            else []
        ),
        "best_weeks_top10": (
            per_run_df.sort_values(["total_pnl_bps", "run_id"], ascending=[False, True], kind="mergesort")
            .head(sample_limit)
            .to_dict(orient="records")
            if not per_run_df.empty
            else []
        ),
        "verdicts": {
            "profitability_status": ("PASS" if total_pnl_bps > 0.0 and (profit_factor or 0.0) > 1.0 else "FAIL"),
            "drawdown_cover_status": ("PASS" if calmar_like is not None and calmar_like >= 1.0 else "FAIL"),
            "exit_efficiency_status": ("PASS" if float(early_exit_regret.mean()) < 0.5 else "FAIL"),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a truth replay root for trade foundation quality.")
    parser.add_argument("--reports-root", help="Path to the truth replay root. Defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--sample-limit", type=int, default=10, help="How many best/worst weeks to keep.")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    summary = build_trade_foundation_quality_summary(reports_root, sample_limit=max(1, int(args.sample_limit)))
    payload = json.dumps(summary, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        Path(args.output).expanduser().resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
