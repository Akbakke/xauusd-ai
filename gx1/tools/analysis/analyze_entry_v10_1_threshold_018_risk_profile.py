#!/usr/bin/env python3
"""
ENTRY_V10.1 Threshold 0.18 Risk Profile Analysis

Analyzes risk profile details for ENTRY_V10.1 with threshold=0.18 (FULLYEAR 2025).
Identifies where losses come from (session, regime, time-of-day, spread/vol).

Usage:
    python -m gx1.tools.analysis.analyze_entry_v10_1_threshold_018_risk_profile \
        --trade-journal-dir data/replay/sniper/entry_v10_1_flat_threshold0_18/2025/FLAT_THRESHOLD_0_18_*/trade_journal \
        --output-report reports/rl/entry_v10/ENTRY_V10_1_THRESHOLD_0_18_RISK_PROFILE_2025.md
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _first_non_null(*values: Any) -> Optional[Any]:
    """Return first non-None value."""
    for v in values:
        if v is not None:
            return v
    return None


def load_trade_journal(trade_journal_path: Path) -> pd.DataFrame:
    """
    Load trade journal (JSON files or parquet) and extract analysis fields.
    
    Returns DataFrame with:
    - trade_id, entry_time, exit_time
    - pnl_bps
    - session, trend_regime, vol_regime
    - spread_bps, atr_bps
    - p_long_v10_1
    - exit_reason
    """
    log.info(f"Loading trade journal from: {trade_journal_path}")
    
    trades_data = []
    
    # Check if it's a directory
    if trade_journal_path.is_dir():
        # Try merged parquet first
        merged_parquet = trade_journal_path / "merged_trade_journal.parquet"
        if merged_parquet.exists():
            log.info(f"Found merged parquet: {merged_parquet}")
            df = pd.read_parquet(merged_parquet)
            # Extract fields from parquet if needed
            if "pnl_bps" not in df.columns and "realized_pnl_bps" in df.columns:
                df["pnl_bps"] = df["realized_pnl_bps"]
            return df
        
        # Try trade_journal_index.csv
        index_csv = trade_journal_path / "trade_journal_index.csv"
        if index_csv.exists():
            log.info(f"Found trade journal index CSV: {index_csv}")
            df = pd.read_csv(index_csv)
            # Extract fields if needed
            if "pnl_bps" not in df.columns and "realized_pnl_bps" in df.columns:
                df["pnl_bps"] = df["realized_pnl_bps"]
            return df
        
        # Load from JSON files
        trades_dir = trade_journal_path / "trades"
        if not trades_dir.exists():
            trades_dir = trade_journal_path  # Fallback: try the directory itself
        
        log.info(f"Loading from JSON files in: {trades_dir}")
        json_files = list(trades_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {trades_dir}")
        
        log.info(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    trade = json.load(f)
                
                # Extract fields using _first_non_null pattern
                entry = trade.get("entry_snapshot") or {}
                exit_summary = trade.get("exit_summary") or {}
                feature_ctx = trade.get("feature_context") or {}  # feature_context is at top level, not nested in entry
                extra = trade.get("extra") or {}
                
                # PnL
                pnl_bps = _first_non_null(
                    exit_summary.get("realized_pnl_bps"),
                    exit_summary.get("pnl_bps"),
                    trade.get("pnl_bps"),
                )
                
                # Skip trades without PnL
                if pnl_bps is None:
                    continue
                
                # Timestamps
                entry_time = _first_non_null(
                    entry.get("entry_time"),
                    trade.get("entry_time"),
                )
                exit_time = _first_non_null(
                    exit_summary.get("exit_time"),
                    trade.get("exit_time"),
                )
                
                # Regime fields
                session = _first_non_null(
                    entry.get("session"),
                    feature_ctx.get("session"),
                    trade.get("session"),
                )
                trend_regime = _first_non_null(
                    entry.get("trend_regime"),
                    feature_ctx.get("trend_regime"),
                    trade.get("trend_regime"),
                )
                vol_regime = _first_non_null(
                    entry.get("vol_regime"),
                    feature_ctx.get("vol_regime"),
                    trade.get("vol_regime"),
                )
                
                # Spread and ATR
                spread_bps = _first_non_null(
                    entry.get("spread_bps"),
                    feature_ctx.get("spread_bps"),
                    trade.get("spread_bps"),
                )
                atr_bps = _first_non_null(
                    entry.get("atr_bps"),
                    feature_ctx.get("atr_bps"),
                    trade.get("atr_bps"),
                )
                
                # p_long_v10_1
                entry_score = entry.get("entry_score") or {}
                p_long_v10_1 = _first_non_null(
                    entry_score.get("p_long_v10_1"),
                    entry.get("p_long_v10_1"),
                    extra.get("p_long_v10_1"),
                    trade.get("p_long_v10_1"),
                )
                
                # Exit reason
                exit_reason = _first_non_null(
                    exit_summary.get("exit_reason"),
                    trade.get("exit_reason"),
                )
                
                trades_data.append({
                    "trade_id": trade.get("trade_id"),
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "pnl_bps": float(pnl_bps) if pnl_bps is not None else None,
                    "session": session,
                    "trend_regime": trend_regime,
                    "vol_regime": vol_regime,
                    "spread_bps": float(spread_bps) if spread_bps is not None else None,
                    "atr_bps": float(atr_bps) if atr_bps is not None else None,
                    "p_long_v10_1": float(p_long_v10_1) if p_long_v10_1 is not None else None,
                    "exit_reason": exit_reason,
                })
                
            except json.JSONDecodeError as e:
                log.warning(f"Failed to decode JSON from {json_file}: {e}")
            except Exception as e:
                log.warning(f"Error processing {json_file}: {e}")
    
    else:
        # Assume it's a parquet or CSV file
        if trade_journal_path.suffix == ".parquet":
            df = pd.read_parquet(trade_journal_path)
            if "pnl_bps" not in df.columns and "realized_pnl_bps" in df.columns:
                df["pnl_bps"] = df["realized_pnl_bps"]
            return df
        elif trade_journal_path.suffix == ".csv":
            df = pd.read_csv(trade_journal_path)
            if "pnl_bps" not in df.columns and "realized_pnl_bps" in df.columns:
                df["pnl_bps"] = df["realized_pnl_bps"]
            return df
        else:
            raise ValueError(f"Unsupported file format: {trade_journal_path.suffix}")
    
    if not trades_data:
        raise ValueError(f"No valid trades found in {trade_journal_path}")
    
    df = pd.DataFrame(trades_data)
    
    # Convert timestamps
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    
    log.info(f"Loaded {len(df)} trades")
    return df


def compute_equity_curve(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute max drawdown from equity curve.
    
    Returns: (max_drawdown_bps, max_drawdown_pct)
    """
    if len(df) == 0:
        return 0.0, 0.0
    
    df_sorted = df.sort_values("entry_time").copy()
    df_sorted["cumulative_pnl"] = df_sorted["pnl_bps"].cumsum()
    df_sorted["running_max"] = df_sorted["cumulative_pnl"].expanding().max()
    df_sorted["drawdown"] = df_sorted["cumulative_pnl"] - df_sorted["running_max"]
    
    max_drawdown_bps = df_sorted["drawdown"].min()
    
    return max_drawdown_bps, 0.0  # pct not used for now


def analyze_global_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute global overview metrics."""
    n_trades = len(df)
    mean_pnl = df["pnl_bps"].mean()
    median_pnl = df["pnl_bps"].median()
    winrate = (df["pnl_bps"] > 0).sum() / n_trades if n_trades > 0 else 0.0
    
    p01 = df["pnl_bps"].quantile(0.01)
    p05 = df["pnl_bps"].quantile(0.05)
    p95 = df["pnl_bps"].quantile(0.95)
    p99 = df["pnl_bps"].quantile(0.99)
    
    max_dd_bps, _ = compute_equity_curve(df)
    
    return {
        "n_trades": n_trades,
        "mean_pnl_bps": float(mean_pnl),
        "median_pnl_bps": float(median_pnl),
        "winrate": float(winrate),
        "p01_bps": float(p01),
        "p05_bps": float(p05),
        "p95_bps": float(p95),
        "p99_bps": float(p99),
        "max_drawdown_bps": float(max_dd_bps),
    }


def analyze_worst_trades(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Extract worst N trades sorted by pnl_bps."""
    worst = df.nsmallest(n, "pnl_bps").copy()
    
    # Select relevant columns
    cols = [
        "trade_id",
        "entry_time",
        "session",
        "trend_regime",
        "vol_regime",
        "spread_bps",
        "atr_bps",
        "p_long_v10_1",
        "exit_reason",
        "pnl_bps",
    ]
    available_cols = [c for c in cols if c in worst.columns]
    
    return worst[available_cols].sort_values("pnl_bps")


def analyze_regime_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze PnL per (session, trend_regime, vol_regime) combination."""
    # Filter out rows with missing regime data
    df_clean = df[
        df["session"].notna() &
        df["trend_regime"].notna() &
        df["vol_regime"].notna()
    ].copy()
    
    if len(df_clean) == 0:
        log.warning("No trades with complete regime data for heatmap analysis")
        return pd.DataFrame()
    
    grouped = df_clean.groupby(["session", "trend_regime", "vol_regime"], dropna=False).agg({
        "pnl_bps": ["count", "mean", "median", lambda x: x.quantile(0.05)],
    }).reset_index()
    
    grouped.columns = ["session", "trend_regime", "vol_regime", "n_trades", "mean_pnl_bps", "median_pnl_bps", "p05_pnl_bps"]
    
    return grouped.sort_values("mean_pnl_bps")


def analyze_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze PnL per hour of day."""
    df_clean = df[df["entry_time"].notna()].copy()
    
    if len(df_clean) == 0:
        log.warning("No trades with entry_time for time-of-day analysis")
        return pd.DataFrame()
    
    df_clean["hour"] = df_clean["entry_time"].dt.hour
    
    grouped = df_clean.groupby("hour", dropna=False).agg({
        "pnl_bps": ["count", "mean", lambda x: x.quantile(0.05)],
    }).reset_index()
    
    grouped.columns = ["hour", "n_trades", "mean_pnl_bps", "p05_pnl_bps"]
    
    return grouped.sort_values("hour")


def analyze_spread_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze PnL per spread_bps bin."""
    df_clean = df[df["spread_bps"].notna()].copy()
    
    if len(df_clean) == 0:
        log.warning("No trades with spread_bps for spread bin analysis")
        return pd.DataFrame()
    
    # Define bins: [0-5, 5-10, 10-20, 20+]
    bins = [0, 5, 10, 20, float("inf")]
    labels = ["0-5", "5-10", "10-20", "20+"]
    
    df_clean["spread_bin"] = pd.cut(df_clean["spread_bps"], bins=bins, labels=labels, include_lowest=True)
    
    grouped = df_clean.groupby("spread_bin", dropna=False).agg({
        "pnl_bps": ["count", "mean", "median", lambda x: x.quantile(0.05)],
    }).reset_index()
    
    grouped.columns = ["spread_bin", "n_trades", "mean_pnl_bps", "median_pnl_bps", "p05_pnl_bps"]
    
    return grouped.sort_values("spread_bin")


def analyze_atr_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze PnL per atr_bps bin."""
    df_clean = df[df["atr_bps"].notna()].copy()
    
    if len(df_clean) == 0:
        log.warning("No trades with atr_bps for ATR bin analysis")
        return pd.DataFrame()
    
    # Use quartiles for ATR bins
    quartiles = df_clean["atr_bps"].quantile([0.25, 0.5, 0.75]).values
    
    if len(quartiles) < 3:
        # Fallback to simple bins: [0-20, 20-40, 40-80, 80+]
        bins = [0, 20, 40, 80, float("inf")]
        labels = ["0-20", "20-40", "40-80", "80+"]
    else:
        bins = [0, quartiles[0], quartiles[1], quartiles[2], float("inf")]
        labels = [f"Q1 (≤{quartiles[0]:.1f})", f"Q2 ({quartiles[0]:.1f}-{quartiles[1]:.1f})", 
                  f"Q3 ({quartiles[1]:.1f}-{quartiles[2]:.1f})", f"Q4 (>{quartiles[2]:.1f})"]
    
    df_clean["atr_bin"] = pd.cut(df_clean["atr_bps"], bins=bins, labels=labels, include_lowest=True)
    
    grouped = df_clean.groupby("atr_bin", dropna=False).agg({
        "pnl_bps": ["count", "mean", "median", lambda x: x.quantile(0.05)],
    }).reset_index()
    
    grouped.columns = ["atr_bin", "n_trades", "mean_pnl_bps", "median_pnl_bps", "p05_pnl_bps"]
    
    return grouped.sort_values("atr_bin")


def flag_risky_regimes(df_regime: pd.DataFrame, min_trades: int = 30, p05_threshold: float = -100.0, mean_threshold: float = -10.0) -> pd.DataFrame:
    """Flag risky regime combinations."""
    if df_regime.empty:
        return df_regime
    
    df_regime["flag"] = "ok"
    
    risky_mask = (
        (df_regime["n_trades"] >= min_trades) &
        (
            (df_regime["p05_pnl_bps"] < p05_threshold) |
            (df_regime["mean_pnl_bps"] < mean_threshold)
        )
    )
    
    df_regime.loc[risky_mask, "flag"] = "risky"
    
    return df_regime


def generate_hotspots_json(
    df_regime: pd.DataFrame,
    df_hour: pd.DataFrame,
    df_spread: pd.DataFrame,
    df_atr: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate JSON hotspots structure."""
    hotspots = {
        "by_regime": [],
        "by_hour": [],
        "by_spread_bin": [],
        "by_atr_bin": [],
    }
    
    # Regime hotspots
    if not df_regime.empty:
        for _, row in df_regime.iterrows():
            hotspots["by_regime"].append({
                "session": str(row["session"]) if pd.notna(row["session"]) else None,
                "trend_regime": str(row["trend_regime"]) if pd.notna(row["trend_regime"]) else None,
                "vol_regime": str(row["vol_regime"]) if pd.notna(row["vol_regime"]) else None,
                "n_trades": int(row["n_trades"]),
                "mean_pnl_bps": float(row["mean_pnl_bps"]),
                "p05_pnl_bps": float(row["p05_pnl_bps"]),
                "flag": str(row.get("flag", "ok")),
            })
    
    # Hour hotspots
    if not df_hour.empty:
        for _, row in df_hour.iterrows():
            flag = "risky" if row["p05_pnl_bps"] < -100.0 or row["mean_pnl_bps"] < -10.0 else "ok"
            hotspots["by_hour"].append({
                "hour": int(row["hour"]),
                "n_trades": int(row["n_trades"]),
                "mean_pnl_bps": float(row["mean_pnl_bps"]),
                "p05_pnl_bps": float(row["p05_pnl_bps"]),
                "flag": flag,
            })
    
    # Spread bin hotspots
    if not df_spread.empty:
        for _, row in df_spread.iterrows():
            flag = "risky" if row["p05_pnl_bps"] < -100.0 or row["mean_pnl_bps"] < -10.0 else "ok"
            hotspots["by_spread_bin"].append({
                "spread_bin": str(row["spread_bin"]),
                "n_trades": int(row["n_trades"]),
                "mean_pnl_bps": float(row["mean_pnl_bps"]),
                "median_pnl_bps": float(row["median_pnl_bps"]),
                "p05_pnl_bps": float(row["p05_pnl_bps"]),
                "flag": flag,
            })
    
    # ATR bin hotspots
    if not df_atr.empty:
        for _, row in df_atr.iterrows():
            flag = "risky" if row["p05_pnl_bps"] < -100.0 or row["mean_pnl_bps"] < -10.0 else "ok"
            hotspots["by_atr_bin"].append({
                "atr_bin": str(row["atr_bin"]),
                "n_trades": int(row["n_trades"]),
                "mean_pnl_bps": float(row["mean_pnl_bps"]),
                "median_pnl_bps": float(row["median_pnl_bps"]),
                "p05_pnl_bps": float(row["p05_pnl_bps"]),
                "flag": flag,
            })
    
    return hotspots


def generate_report(
    global_metrics: Dict[str, Any],
    worst_trades: pd.DataFrame,
    df_regime: pd.DataFrame,
    df_hour: pd.DataFrame,
    df_spread: pd.DataFrame,
    df_atr: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate Markdown report."""
    lines = [
        "# ENTRY_V10.1 Threshold 0.18 Risk Profile Analysis 2025",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Global Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Trades | {global_metrics['n_trades']:,} |",
        f"| Mean PnL | {global_metrics['mean_pnl_bps']:.2f} bps |",
        f"| Median PnL | {global_metrics['median_pnl_bps']:.2f} bps |",
        f"| Win Rate | {global_metrics['winrate']*100:.1f}% |",
        f"| P01 (Tail Risk) | {global_metrics['p01_bps']:.2f} bps |",
        f"| P05 (Tail Risk) | {global_metrics['p05_bps']:.2f} bps |",
        f"| P95 | {global_metrics['p95_bps']:.2f} bps |",
        f"| P99 | {global_metrics['p99_bps']:.2f} bps |",
        f"| Max Drawdown | {global_metrics['max_drawdown_bps']:.2f} bps |",
        "",
        "## Worst Trades (Top 50)",
        "",
    ]
    
    # Worst trades table
    if not worst_trades.empty:
        lines.append("| Trade ID | Entry Time | Session | Trend | Vol | Spread (bps) | ATR (bps) | p_long_v10_1 | Exit Reason | PnL (bps) |")
        lines.append("|----------|------------|---------|-------|-----|--------------|-----------|--------------|-------------|-----------|")
        
        for _, row in worst_trades.head(50).iterrows():
            entry_time = row.get("entry_time")
            entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M") if pd.notna(entry_time) and entry_time is not None else "N/A"
            spread_bps = row.get('spread_bps', None)
            atr_bps = row.get('atr_bps', None)
            p_long_v10_1 = row.get('p_long_v10_1', None)
            spread_str = f"{spread_bps:.2f}" if pd.notna(spread_bps) and spread_bps is not None else "N/A"
            atr_str = f"{atr_bps:.2f}" if pd.notna(atr_bps) and atr_bps is not None else "N/A"
            p_long_str = f"{p_long_v10_1:.3f}" if pd.notna(p_long_v10_1) and p_long_v10_1 is not None else "N/A"
            lines.append(
                f"| {row.get('trade_id', 'N/A')} | {entry_time_str} | "
                f"{row.get('session', 'N/A')} | {row.get('trend_regime', 'N/A')} | {row.get('vol_regime', 'N/A')} | "
                f"{spread_str} | {atr_str} | "
                f"{p_long_str} | {row.get('exit_reason', 'N/A')} | "
                f"{row['pnl_bps']:.2f} |"
            )
    else:
        lines.append("No worst trades data available.")
    
    lines.extend([
        "",
        "## Regime Heatmap",
        "",
        "PnL statistics per (session, trend_regime, vol_regime) combination.",
        "",
    ])
    
    # Regime heatmap table
    if not df_regime.empty:
        lines.append("| Session | Trend | Vol | N Trades | Mean PnL | Median PnL | P05 | Flag |")
        lines.append("|---------|-------|-----|----------|----------|------------|-----|------|")
        
        for _, row in df_regime.iterrows():
            flag = row.get("flag", "ok")
            flag_emoji = "🔴" if flag == "risky" else "🟢"
            lines.append(
                f"| {row['session']} | {row['trend_regime']} | {row['vol_regime']} | "
                f"{int(row['n_trades'])} | {row['mean_pnl_bps']:.2f} | {row['median_pnl_bps']:.2f} | "
                f"{row['p05_pnl_bps']:.2f} | {flag_emoji} {flag} |"
            )
    else:
        lines.append("No regime data available.")
    
    lines.extend([
        "",
        "## Time-of-Day Analysis",
        "",
        "PnL statistics per hour of day (UTC).",
        "",
    ])
    
    # Time-of-day table
    if not df_hour.empty:
        lines.append("| Hour (UTC) | N Trades | Mean PnL | P05 |")
        lines.append("|------------|----------|----------|-----|")
        
        for _, row in df_hour.iterrows():
            flag_emoji = "🔴" if row["p05_pnl_bps"] < -100.0 or row["mean_pnl_bps"] < -10.0 else "🟢"
            lines.append(
                f"| {int(row['hour']):02d}:00 | {int(row['n_trades'])} | "
                f"{row['mean_pnl_bps']:.2f} | {row['p05_pnl_bps']:.2f} | {flag_emoji}"
            )
    else:
        lines.append("No time-of-day data available.")
    
    lines.extend([
        "",
        "## Spread Bins Analysis",
        "",
        "PnL statistics per spread_bps bin.",
        "",
    ])
    
    # Spread bins table
    if not df_spread.empty:
        lines.append("| Spread Bin (bps) | N Trades | Mean PnL | Median PnL | P05 |")
        lines.append("|------------------|----------|----------|------------|-----|")
        
        for _, row in df_spread.iterrows():
            flag_emoji = "🔴" if row["p05_pnl_bps"] < -100.0 or row["mean_pnl_bps"] < -10.0 else "🟢"
            lines.append(
                f"| {row['spread_bin']} | {int(row['n_trades'])} | "
                f"{row['mean_pnl_bps']:.2f} | {row['median_pnl_bps']:.2f} | "
                f"{row['p05_pnl_bps']:.2f} | {flag_emoji}"
            )
    else:
        lines.append("No spread bin data available.")
    
    lines.extend([
        "",
        "## ATR Bins Analysis",
        "",
        "PnL statistics per atr_bps bin (quartiles).",
        "",
    ])
    
    # ATR bins table
    if not df_atr.empty:
        lines.append("| ATR Bin (bps) | N Trades | Mean PnL | Median PnL | P05 |")
        lines.append("|---------------|----------|----------|------------|-----|")
        
        for _, row in df_atr.iterrows():
            flag_emoji = "🔴" if row["p05_pnl_bps"] < -100.0 or row["mean_pnl_bps"] < -10.0 else "🟢"
            lines.append(
                f"| {row['atr_bin']} | {int(row['n_trades'])} | "
                f"{row['mean_pnl_bps']:.2f} | {row['median_pnl_bps']:.2f} | "
                f"{row['p05_pnl_bps']:.2f} | {flag_emoji}"
            )
    else:
        lines.append("No ATR bin data available.")
    
    lines.extend([
        "",
        "## Interpretation",
        "",
        "**Flag Legend:**",
        "- 🟢 ok: Risk profile acceptable",
        "- 🔴 risky: High tail risk (P05 < -100 bps) or negative mean PnL (< -10 bps) with sufficient sample size (n >= 30)",
        "",
        "**Next Steps:**",
        "- Identify regime combinations with 🔴 risky flags",
        "- Consider simple regime/spread gates based on risk hotspots",
        "- This analysis provides factual basis before implementing gates or TimingCritic",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ENTRY_V10.1 threshold 0.18 risk profile")
    parser.add_argument(
        "--trade-journal-dir",
        type=Path,
        required=True,
        help="Path to trade journal directory (or merged parquet/CSV file)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/rl/entry_v10/ENTRY_V10_1_THRESHOLD_0_18_RISK_PROFILE_2025.md"),
        help="Path to output Markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/entry_v10/entry_v10_1_threshold_0_18_risk_hotspots_2025.json"),
        help="Path to output JSON hotspots file",
    )
    
    args = parser.parse_args()
    
    if not args.trade_journal_dir.exists():
        log.error(f"❌ ERROR: Trade journal directory not found: {args.trade_journal_dir}")
        return 1
    
    # Load trade journal
    try:
        df = load_trade_journal(args.trade_journal_dir)
    except Exception as e:
        log.error(f"❌ ERROR loading trade journal: {e}")
        return 1
    
    if len(df) == 0:
        log.error("❌ ERROR: No trades found in trade journal")
        return 1
    
    log.info(f"✅ Loaded {len(df)} trades")
    
    # Run analyses
    log.info("Computing global overview...")
    global_metrics = analyze_global_overview(df)
    
    log.info("Extracting worst trades...")
    worst_trades = analyze_worst_trades(df, n=50)
    
    log.info("Computing regime heatmap...")
    df_regime = analyze_regime_heatmap(df)
    df_regime = flag_risky_regimes(df_regime, min_trades=30, p05_threshold=-100.0, mean_threshold=-10.0)
    
    log.info("Computing time-of-day analysis...")
    df_hour = analyze_time_of_day(df)
    
    log.info("Computing spread bins analysis...")
    df_spread = analyze_spread_bins(df)
    
    log.info("Computing ATR bins analysis...")
    df_atr = analyze_atr_bins(df)
    
    # Generate hotspots JSON
    log.info("Generating hotspots JSON...")
    hotspots = generate_hotspots_json(df_regime, df_hour, df_spread, df_atr)
    
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(hotspots, f, indent=2)
    
    log.info(f"✅ Hotspots JSON saved to: {args.output_json}")
    
    # Generate report
    log.info("Generating Markdown report...")
    generate_report(
        global_metrics,
        worst_trades,
        df_regime,
        df_hour,
        df_spread,
        df_atr,
        args.output_report,
    )
    
    log.info("✅ Risk profile analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

