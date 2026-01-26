#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Shadow & Trade Journals → RL Dataset Builder

Reads shadow-journal (shadow_hits.jsonl) and trade-journal (trades/*.json)
from a selected run directory, joins them on timestamp, and generates:
1. Summary analysis (histograms, hit-rates, average PnL per bucket)
2. RL-ready dataset (Parquet + CSV)

Usage:
    python gx1/scripts/analyze_shadow_and_trades.py --run_dir runs/live_demo/SNIPER_20251225_204700
    python gx1/scripts/analyze_shadow_and_trades.py  # Uses latest SNIPER run
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def find_latest_sniper_run(base_dir: Path = Path("runs/live_demo")) -> Optional[Path]:
    """Find latest SNIPER run directory."""
    if not base_dir.exists():
        return None
    
    sniper_runs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("SNIPER_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    
    if sniper_runs:
        return sniper_runs[0]
    return None


def load_shadow_journal(shadow_path: Path) -> pd.DataFrame:
    """Load shadow journal from JSONL file."""
    if not shadow_path.exists():
        log.warning(f"Shadow journal not found: {shadow_path}")
        return pd.DataFrame()
    
    records = []
    with open(shadow_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse shadow line: {e}")
                continue
    
    if not records:
        log.warning("No shadow records found")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Convert ts to datetime
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
    # Normalize shadow_hits dict to columns
    if "shadow_hits" in df.columns:
        shadow_thresholds = ["0.55", "0.58", "0.60", "0.62", "0.65"]
        for thr in shadow_thresholds:
            df[f"shadow_hit_{thr}"] = df["shadow_hits"].apply(
                lambda x: x.get(thr, False) if isinstance(x, dict) else False
            )
    
    log.info(f"Loaded {len(df)} shadow records from {shadow_path}")
    return df


def load_trade_journal(run_dir: Path) -> pd.DataFrame:
    """Load trade journal from JSON files and index CSV."""
    trade_dir = run_dir / "trade_journal" / "trades"
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    
    if not trade_dir.exists():
        log.warning(f"Trade journal directory not found: {trade_dir}")
        return pd.DataFrame()
    
    trades = []
    
    # Try to load from index CSV first (faster)
    if index_path.exists():
        try:
            index_df = pd.read_csv(index_path)
            log.info(f"Loaded {len(index_df)} trades from index CSV")
            
            # Load full JSON for each trade to get complete data
            for _, row in index_df.iterrows():
                trade_id = row.get("trade_id")
                if not trade_id:
                    continue
                
                trade_json_path = trade_dir / f"{trade_id}.json"
                if trade_json_path.exists():
                    try:
                        with open(trade_json_path, "r") as f:
                            trade_data = json.load(f)
                            trades.append(trade_data)
                    except Exception as e:
                        log.warning(f"Failed to load trade {trade_id}: {e}")
        except Exception as e:
            log.warning(f"Failed to load index CSV: {e}")
    
    # Fallback: load all JSON files directly
    if not trades:
        json_files = list(trade_dir.glob("*.json"))
        log.info(f"Loading {len(json_files)} trade JSON files directly")
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    trade_data = json.load(f)
                    trades.append(trade_data)
            except Exception as e:
                log.warning(f"Failed to load {json_file}: {e}")
    
    if not trades:
        log.warning("No trades found")
        return pd.DataFrame()
    
    # Flatten trade data to DataFrame
    trade_rows = []
    for trade in trades:
        # Handle None entry_snapshot (fallback to top-level or empty dict)
        entry_snapshot_raw = trade.get("entry_snapshot")
        if entry_snapshot_raw is None:
            entry_snapshot = {}
        elif isinstance(entry_snapshot_raw, dict):
            entry_snapshot = entry_snapshot_raw
        else:
            entry_snapshot = {}
        
        exit_summary = trade.get("exit_summary", {}) or {}
        feature_context = trade.get("feature_context", {}) or {}
        
        # Extract p_long from entry_score or direct field
        p_long = None
        entry_score = entry_snapshot.get("entry_score", {})
        if isinstance(entry_score, dict):
            p_long = entry_score.get("p_long")
        if p_long is None:
            p_long = entry_snapshot.get("p_long")
        # Fallback to top-level entry_score if still None
        if p_long is None:
            p_long = trade.get("entry_score", 0.0)
        
        # Calculate holding_time_bars from entry/exit times
        holding_time_bars = None
        entry_time_str = entry_snapshot.get("entry_time") or trade.get("entry_time")
        exit_time_str = exit_summary.get("exit_time") or trade.get("exit_time")
        if entry_time_str and exit_time_str:
            try:
                entry_ts = pd.to_datetime(entry_time_str, utc=True)
                exit_ts = pd.to_datetime(exit_time_str, utc=True)
                holding_time_bars = int((exit_ts - entry_ts).total_seconds() / 300)  # 5 min bars
            except Exception:
                pass
        
        # Extract Entry Critic score from entry_snapshot.entry_critic if available
        entry_critic_score = None
        entry_critic_data = entry_snapshot.get("entry_critic")
        if isinstance(entry_critic_data, dict):
            entry_critic_score = entry_critic_data.get("score_v1")
        
        row = {
            "trade_id": trade.get("trade_id"),
            "entry_time": entry_time_str,
            "exit_time": exit_time_str,
            "entry_price": entry_snapshot.get("entry_price") or trade.get("entry_price"),
            "exit_price": exit_summary.get("exit_price") or trade.get("exit_price"),
            "side": entry_snapshot.get("side") or trade.get("side"),
            "session": entry_snapshot.get("session") or trade.get("session", "UNKNOWN"),
            "trend_regime": entry_snapshot.get("trend_regime") or trade.get("trend_regime", "UNKNOWN"),
            "vol_regime": entry_snapshot.get("vol_regime") or entry_snapshot.get("regime") or trade.get("vol_regime", "UNKNOWN"),
            "p_long": p_long or 0.0,
            "entry_reason": entry_snapshot.get("reason") or trade.get("entry_reason"),
            "exit_reason": exit_summary.get("exit_reason") or trade.get("exit_reason"),
            "pnl_bps": exit_summary.get("realized_pnl_bps") or trade.get("realized_pnl_bps"),
            "max_mfe_bps": exit_summary.get("max_mfe_bps") or trade.get("max_mfe_bps"),
            "max_mae_bps": exit_summary.get("max_mae_bps") or trade.get("max_mae_bps"),
            "intratrade_drawdown_bps": exit_summary.get("intratrade_drawdown_bps") or trade.get("intratrade_drawdown_bps"),
            "holding_time_bars": holding_time_bars,
            "atr_bps": feature_context.get("atr_bps") or entry_snapshot.get("atr_bps") or trade.get("atr_bps"),
            "spread_bps": feature_context.get("spread_bps") or entry_snapshot.get("spread_bps") or trade.get("spread_bps"),
            "range_pos": feature_context.get("range_pos") or trade.get("range_pos"),
            "distance_to_range": feature_context.get("distance_to_range") or trade.get("distance_to_range"),
            "range_edge_dist_atr": feature_context.get("range_edge_dist_atr") or trade.get("range_edge_dist_atr"),
            "units": entry_snapshot.get("units") or trade.get("units"),
            "base_units": entry_snapshot.get("base_units") or trade.get("base_units"),
            "entry_critic_score_v1": entry_critic_score,  # Entry Critic V1 score from trade journal
        }
        trade_rows.append(row)
    
    df = pd.DataFrame(trade_rows)
    
    # Convert timestamps
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    
    log.info(f"Loaded {len(df)} trades")
    return df


def round_to_m5(ts: pd.Timestamp) -> pd.Timestamp:
    """Round timestamp to nearest M5 (00:00, 00:05, 00:10, etc.)."""
    if pd.isna(ts):
        return ts
    
    # Round to nearest 5 minutes
    minutes = ts.minute
    rounded_minutes = (minutes // 5) * 5
    return ts.replace(minute=rounded_minutes, second=0, microsecond=0)


def join_shadow_and_trades(
    shadow_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    tolerance_bars: int = 1,
) -> pd.DataFrame:
    """
    Join shadow and trades on M5 candle timestamp.
    
    Args:
        shadow_df: Shadow journal DataFrame
        trades_df: Trade journal DataFrame
        tolerance_bars: Number of M5 bars tolerance for matching (±1 = ±5 minutes)
    
    Returns:
        Joined DataFrame with action_taken (1=trade, 0=skip)
    """
    if shadow_df.empty:
        log.warning("Shadow DataFrame is empty")
        return pd.DataFrame()
    
    # Round timestamps to M5
    shadow_df = shadow_df.copy()
    shadow_df["candle_time"] = shadow_df["ts"].apply(round_to_m5)
    
    if trades_df.empty:
        # No trades: all shadow entries are negative examples
        shadow_df["action_taken"] = 0
        shadow_df["trade_id"] = None
        shadow_df["pnl_bps"] = None
        shadow_df["exit_reason"] = None
        shadow_df["holding_time_bars"] = None
        return shadow_df
    
    trades_df = trades_df.copy()
    trades_df["candle_time"] = trades_df["entry_time"].apply(round_to_m5)
    
    # Merge on candle_time
    merged = shadow_df.merge(
        trades_df,
        on="candle_time",
        how="left",
        suffixes=("_shadow", "_trade"),
    )
    
    # Set action_taken: 1 if trade matched, 0 otherwise
    merged["action_taken"] = merged["trade_id"].notna().astype(int)
    
    # For rows with trades, use trade data; for shadow-only, use shadow data
    # Rename columns to avoid conflicts
    if "p_long_trade" in merged.columns:
        merged["p_long"] = merged["p_long_trade"].fillna(merged["p_long_shadow"])
    if "atr_bps_trade" in merged.columns:
        merged["atr_bps"] = merged["atr_bps_trade"].fillna(merged["atr_bps_shadow"])
    if "spread_bps_trade" in merged.columns:
        merged["spread_bps"] = merged["spread_bps_trade"].fillna(merged["spread_bps_shadow"])
    if "session_trade" in merged.columns:
        merged["session"] = merged["session_trade"].fillna(merged["session_shadow"])
    if "trend_regime_trade" in merged.columns:
        merged["trend_regime"] = merged["trend_regime_trade"].fillna(merged["trend_regime_shadow"])
    if "vol_regime_trade" in merged.columns:
        merged["vol_regime"] = merged["vol_regime_trade"].fillna(merged["vol_regime_shadow"])
    
    # Entry Critic score: prefer shadow (always present), fallback to trade if available
    if "entry_critic_score_v1_shadow" in merged.columns:
        merged["entry_critic_score_v1"] = merged["entry_critic_score_v1_shadow"]
    elif "entry_critic_score_v1_trade" in merged.columns:
        merged["entry_critic_score_v1"] = merged["entry_critic_score_v1_trade"]
    # Also check entry_snapshot.entry_critic.score_v1 from trades
    if "entry_critic_score_v1" not in merged.columns or merged["entry_critic_score_v1"].isna().all():
        # Try to extract from trade entry_snapshot if available
        if "entry_snapshot" in trades_df.columns:
            # This would require flattening entry_snapshot first, but for now use shadow data
            pass
    
    log.info(f"Joined {len(merged)} records: {merged['action_taken'].sum()} trades, {len(merged) - merged['action_taken'].sum()} shadow-only")
    
    return merged


def build_rl_dataset(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Build RL-ready dataset from joined shadow+trades DataFrame."""
    if joined_df.empty:
        return pd.DataFrame()
    
    rl_df = joined_df.copy()
    
    # State fields (already present)
    # - candle_time, p_long, spread_bps, atr_bps, trend_regime, vol_regime, session
    
    # Action fields
    # - action_taken (already set in join)
    # - threshold_slot (bucket for p_long)
    rl_df["threshold_slot"] = rl_df["p_long"].apply(
        lambda x: "0.67+" if pd.isna(x) or x >= 0.67
        else "0.65-0.67" if x >= 0.65
        else "0.62-0.65" if x >= 0.62
        else "0.60-0.62" if x >= 0.60
        else "0.58-0.60" if x >= 0.58
        else "0.55-0.58" if x >= 0.55
        else "<0.55"
    )
    
    # Real threshold (from shadow data)
    rl_df["real_threshold"] = rl_df.get("real_threshold_shadow", rl_df.get("real_threshold", 0.67))
    
    # Outcome fields
    # - pnl_bps (already present from trades)
    # - reward (use pnl_bps as reward for now)
    rl_df["reward"] = rl_df["pnl_bps"].fillna(0.0).astype(float)
    
    # Labels
    rl_df["label_profitable"] = (rl_df["pnl_bps"] > 0).astype(int).fillna(0)
    rl_df["label_profitable_10bps"] = (rl_df["pnl_bps"] > 10).astype(int).fillna(0)
    rl_df["label_profitable_20bps"] = (rl_df["pnl_bps"] > 20).astype(int).fillna(0)
    rl_df["label_profitable_50bps"] = (rl_df["pnl_bps"] > 50).astype(int).fillna(0)
    
    # One-hot encode regimes
    rl_df["regime_trend_up"] = (rl_df["trend_regime"] == "TREND_UP").astype(int)
    rl_df["regime_trend_down"] = (rl_df["trend_regime"] == "TREND_DOWN").astype(int)
    rl_df["regime_vol_low"] = (rl_df["vol_regime"] == "LOW").astype(int)
    rl_df["regime_vol_high"] = (rl_df["vol_regime"] == "HIGH").astype(int)
    
    # One-hot encode sessions
    rl_df["session_eu"] = (rl_df["session"] == "EU").astype(int)
    rl_df["session_overlap"] = (rl_df["session"] == "OVERLAP").astype(int)
    rl_df["session_us"] = (rl_df["session"] == "US").astype(int)
    
    # Select final columns for RL dataset
    rl_columns = [
        "candle_time",
        "instrument",
        "session",
        "p_long",
        "spread_bps",
        "atr_bps",
        "trend_regime",
        "vol_regime",
        "range_pos",
        "distance_to_range",
        "range_edge_dist_atr",
        # Regime tags (one-hot)
        "regime_trend_up",
        "regime_trend_down",
        "regime_vol_low",
        "regime_vol_high",
        # Session tags (one-hot)
        "session_eu",
        "session_overlap",
        "session_us",
        # Action
        "action_taken",
        "threshold_slot",
        "real_threshold",
        "shadow_hit_0.55",
        "shadow_hit_0.58",
        "shadow_hit_0.60",
        "shadow_hit_0.62",
        "shadow_hit_0.65",
        # Outcome
        "pnl_bps",
        "reward",
        "max_mfe_bps",
        "max_mae_bps",
        "intratrade_drawdown_bps",
        "holding_time_bars",
        "exit_reason",
        # Labels
        "label_profitable",
        "label_profitable_10bps",
        "label_profitable_20bps",
        "label_profitable_50bps",
        # Metadata
        "trade_id",
        "entry_time",
        "exit_time",
    ]
    
    # Filter to available columns
    available_columns = [col for col in rl_columns if col in rl_df.columns]
    rl_df = rl_df[available_columns].copy()
    
    # Fill missing values
    if "instrument" not in rl_df.columns:
        rl_df["instrument"] = "XAU_USD"
    if "pnl_bps" in rl_df.columns:
        rl_df["pnl_bps"] = rl_df["pnl_bps"].fillna(0.0).astype(float)
    if "reward" in rl_df.columns:
        rl_df["reward"] = rl_df["reward"].fillna(0.0).astype(float)
    
    log.info(f"Built RL dataset with {len(rl_df)} records and {len(available_columns)} columns")
    
    return rl_df


def generate_analysis_report(
    shadow_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    run_dir: Path,
) -> str:
    """Generate markdown analysis report."""
    lines = []
    
    lines.append("# SNIPER Shadow & Trade Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Run Directory:** {run_dir}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    total_shadow = len(shadow_df)
    total_trades = len(trades_df)
    shadow_only = len(joined_df) - joined_df["action_taken"].sum() if not joined_df.empty else 0
    
    lines.append(f"- **Shadow samples:** {total_shadow}")
    lines.append(f"- **Trades executed:** {total_trades}")
    lines.append(f"- **Shadow-only signals:** {shadow_only}")
    lines.append("")
    
    # Reward statistics
    if not rl_df.empty and "reward" in rl_df.columns:
        trades_mask = rl_df["action_taken"] == 1
        trades_reward = rl_df[trades_mask]["reward"]
        shadow_reward = rl_df[~trades_mask]["reward"]
        
        lines.append("## Reward Statistics")
        lines.append("")
        if len(trades_reward) > 0:
            lines.append(f"- **Mean reward (trades):** {trades_reward.mean():.2f} bps")
            lines.append(f"- **Median reward (trades):** {trades_reward.median():.2f} bps")
            lines.append(f"- **Std reward (trades):** {trades_reward.std():.2f} bps")
            lines.append(f"- **Min reward (trades):** {trades_reward.min():.2f} bps")
            lines.append(f"- **Max reward (trades):** {trades_reward.max():.2f} bps")
        else:
            lines.append("- **Mean reward (trades):** N/A (no trades)")
        lines.append(f"- **Mean reward (shadow skipped):** {shadow_reward.mean():.2f} bps (always 0)")
        lines.append("")
    
    # p_long distribution
    if not rl_df.empty and "p_long" in rl_df.columns:
        lines.append("## p_long Distribution")
        lines.append("")
        lines.append("### All Signals")
        lines.append("")
        p_long_stats = rl_df["p_long"].describe()
        lines.append(f"- **Mean:** {p_long_stats.get('mean', 0):.3f}")
        lines.append(f"- **Median:** {p_long_stats.get('50%', 0):.3f}")
        lines.append(f"- **Min:** {p_long_stats.get('min', 0):.3f}")
        lines.append(f"- **Max:** {p_long_stats.get('max', 0):.3f}")
        lines.append(f"- **Std:** {p_long_stats.get('std', 0):.3f}")
        lines.append("")
        
        lines.append("### Trades Only")
        lines.append("")
        trades_p_long = rl_df[rl_df["action_taken"] == 1]["p_long"]
        if len(trades_p_long) > 0:
            trades_stats = trades_p_long.describe()
            lines.append(f"- **Mean:** {trades_stats.get('mean', 0):.3f}")
            lines.append(f"- **Median:** {trades_stats.get('50%', 0):.3f}")
            lines.append(f"- **Min:** {trades_stats.get('min', 0):.3f}")
            lines.append(f"- **Max:** {trades_stats.get('max', 0):.3f}")
        else:
            lines.append("- **N/A** (no trades)")
        lines.append("")
    
    # Entry Critic V1 Evaluation (if available)
    if not rl_df.empty and "entry_critic_score_v1" in rl_df.columns:
        lines.append("## Entry Critic V1 Evaluation")
        lines.append("")
        
        critic_scores = rl_df["entry_critic_score_v1"].dropna()
        if len(critic_scores) > 0:
            lines.append("### Score Statistics")
            lines.append("")
            critic_stats = critic_scores.describe()
            lines.append(f"- **Mean:** {critic_stats.get('mean', 0):.3f}")
            lines.append(f"- **Median:** {critic_stats.get('50%', 0):.3f}")
            lines.append(f"- **Min:** {critic_stats.get('min', 0):.3f}")
            lines.append(f"- **Max:** {critic_stats.get('max', 0):.3f}")
            lines.append(f"- **Std:** {critic_stats.get('std', 0):.3f}")
            lines.append("")
            
            # Correlation with reward
            if "reward" in rl_df.columns:
                critic_reward_corr = rl_df[["entry_critic_score_v1", "reward"]].corr().iloc[0, 1]
                lines.append(f"- **Correlation with reward:** {critic_reward_corr:.3f}")
                lines.append("")
            
            # Score by trade outcome
            if "pnl_bps" in rl_df.columns and "action_taken" in rl_df.columns:
                trades_mask = rl_df["action_taken"] == 1
                profitable_mask = (rl_df["pnl_bps"] >= 10.0) & trades_mask
                unprofitable_mask = (rl_df["pnl_bps"] < 10.0) & trades_mask
                
                if profitable_mask.sum() > 0:
                    profitable_scores = rl_df[profitable_mask]["entry_critic_score_v1"].dropna()
                    if len(profitable_scores) > 0:
                        lines.append("### Score by Trade Outcome")
                        lines.append("")
                        lines.append(f"- **Profitable trades (≥10 bps):** {len(profitable_scores)}")
                        lines.append(f"  - Mean score: {profitable_scores.mean():.3f}")
                        lines.append(f"  - Median score: {profitable_scores.median():.3f}")
                        lines.append("")
                
                if unprofitable_mask.sum() > 0:
                    unprofitable_scores = rl_df[unprofitable_mask]["entry_critic_score_v1"].dropna()
                    if len(unprofitable_scores) > 0:
                        lines.append(f"- **Unprofitable trades (<10 bps):** {len(unprofitable_scores)}")
                        lines.append(f"  - Mean score: {unprofitable_scores.mean():.3f}")
                        lines.append(f"  - Median score: {unprofitable_scores.median():.3f}")
                        lines.append("")
            
            # AUC comparison: p_long vs entry_critic_score_v1
            if "pnl_bps" in rl_df.columns and "p_long" in rl_df.columns and "action_taken" in rl_df.columns:
                trades_mask = rl_df["action_taken"] == 1
                if trades_mask.sum() > 0:
                    trades_subset = rl_df[trades_mask].copy()
                    trades_subset["profitable"] = (trades_subset["pnl_bps"] >= 10.0).astype(int)
                    
                    if trades_subset["profitable"].nunique() == 2:  # Both classes present
                        try:
                            from sklearn.metrics import roc_auc_score
                            
                            p_long_auc = roc_auc_score(
                                trades_subset["profitable"],
                                trades_subset["p_long"]
                            )
                            critic_auc = roc_auc_score(
                                trades_subset["profitable"],
                                trades_subset["entry_critic_score_v1"]
                            )
                            
                            lines.append("### AUC Comparison (p_long vs Entry Critic)")
                            lines.append("")
                            lines.append(f"- **p_long AUC:** {p_long_auc:.3f}")
                            lines.append(f"- **Entry Critic AUC:** {critic_auc:.3f}")
                            lines.append(f"- **Improvement:** {critic_auc - p_long_auc:+.3f}")
                            lines.append("")
                        except Exception as e:
                            log.warning(f"Failed to calculate AUC: {e}")
        else:
            lines.append("- **No Entry Critic scores available**")
            lines.append("")
    
    # Threshold performance
    if not rl_df.empty and "shadow_hit_0.55" in rl_df.columns:
        lines.append("## Threshold Performance")
        lines.append("")
        lines.append("| Threshold | Hit Count | Trades | Avg PnL (bps) | Hit Rate (%) |")
        lines.append("|-----------|-----------|--------|---------------|--------------|")
        
        thresholds = [0.55, 0.58, 0.60, 0.62, 0.65]
        for thr in thresholds:
            thr_col = f"shadow_hit_{thr:.2f}"
            if thr_col in rl_df.columns:
                hits = rl_df[thr_col].sum()
                trades_at_thr = rl_df[(rl_df[thr_col] == True) & (rl_df["action_taken"] == 1)]
                trade_count = len(trades_at_thr)
                avg_pnl = trades_at_thr["pnl_bps"].mean() if trade_count > 0 else 0.0
                hit_rate = (trade_count / hits * 100) if hits > 0 else 0.0
                lines.append(f"| {thr:.2f} | {hits} | {trade_count} | {avg_pnl:.2f} | {hit_rate:.1f} |")
        lines.append("")
    
    # Top 10 p_long ranges by reward
    if not rl_df.empty and "p_long" in rl_df.columns and "reward" in rl_df.columns:
        lines.append("## Top 10 p_long Ranges by Reward")
        lines.append("")
        trades_rl = rl_df[rl_df["action_taken"] == 1].copy()
        if len(trades_rl) > 0:
            # Create bins
            trades_rl["p_long_bin"] = pd.cut(trades_rl["p_long"], bins=10, precision=2)
            bin_stats = trades_rl.groupby("p_long_bin")["reward"].agg(["mean", "count"]).sort_values("mean", ascending=False)
            
            lines.append("| p_long Range | Mean Reward (bps) | Count |")
            lines.append("|--------------|-------------------|-------|")
            for idx, (bin_range, stats) in enumerate(bin_stats.head(10).iterrows()):
                lines.append(f"| {bin_range} | {stats['mean']:.2f} | {int(stats['count'])} |")
        else:
            lines.append("- **N/A** (no trades)")
        lines.append("")
    
    # Regime distribution
    if not rl_df.empty and "trend_regime" in rl_df.columns and "vol_regime" in rl_df.columns:
        lines.append("## Regime Distribution")
        lines.append("")
        lines.append("### Trades vs Non-Trades")
        lines.append("")
        
        regime_cross = pd.crosstab(
            [rl_df["trend_regime"], rl_df["vol_regime"]],
            rl_df["action_taken"],
            margins=True,
        )
        # Convert to markdown manually (avoid tabulate dependency)
        lines.append("| Trend Regime | Vol Regime | 0 (Skip) | 1 (Trade) | All |")
        lines.append("|--------------|------------|----------|-----------|-----|")
        for idx, row in regime_cross.iterrows():
            trend, vol = idx if isinstance(idx, tuple) else (idx, "")
            lines.append(f"| {trend} | {vol} | {int(row.get(0, 0))} | {int(row.get(1, 0))} | {int(row.get('All', 0))} |")
        lines.append("")
    
    # Exit reasons
    if not trades_df.empty and "exit_reason" in trades_df.columns:
        lines.append("## Exit Reasons")
        lines.append("")
        exit_reasons = trades_df["exit_reason"].value_counts()
        lines.append("| Exit Reason | Count |")
        lines.append("|-------------|-------|")
        for reason, count in exit_reasons.items():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    
    # Session distribution
    if not rl_df.empty and "session" in rl_df.columns:
        lines.append("## Session Distribution")
        lines.append("")
        session_cross = pd.crosstab(rl_df["session"], rl_df["action_taken"], margins=True)
        # Convert to markdown manually (avoid tabulate dependency)
        lines.append("| Session | 0 (Skip) | 1 (Trade) | All |")
        lines.append("|---------|----------|-----------|-----|")
        for session, row in session_cross.iterrows():
            lines.append(f"| {session} | {int(row.get(0, 0))} | {int(row.get(1, 0))} | {int(row.get('All', 0))} |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/analyze_shadow_and_trades.py`*")
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Shadow & Trade Journals → RL Dataset Builder"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Run directory path (default: latest SNIPER run)",
    )
    parser.add_argument(
        "--export_rl_dataset",
        action="store_true",
        default=True,
        help="Export RL dataset to Parquet and CSV (default: True)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/sniper/auto_analysis"),
        help="Output directory for reports (default: reports/sniper/auto_analysis)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/rl"),
        help="Output directory for RL datasets (default: data/rl)",
    )
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_sniper_run()
        if not run_dir:
            log.error("No SNIPER run found. Specify --run_dir or ensure runs/live_demo/SNIPER_* exists")
            return 1
        log.info(f"Using latest SNIPER run: {run_dir}")
    
    if not run_dir.exists():
        log.error(f"Run directory not found: {run_dir}")
        return 1
    
    log.info(f"Analyzing run: {run_dir}")
    
    # Load shadow journal
    shadow_path = run_dir / "shadow" / "shadow_hits.jsonl"
    shadow_df = load_shadow_journal(shadow_path)
    
    if shadow_df.empty:
        log.warning("No shadow data found. Analysis will be limited.")
    
    # Load trade journal
    trades_df = load_trade_journal(run_dir)
    
    if trades_df.empty:
        log.warning("No trades found. Analysis will show shadow-only signals.")
    
    # Join shadow and trades
    joined_df = join_shadow_and_trades(shadow_df, trades_df)
    
    if joined_df.empty:
        log.error("No joined data available. Cannot generate analysis.")
        return 1
    
    # Build RL dataset
    rl_df = build_rl_dataset(joined_df)
    
    # Generate analysis report
    report = generate_analysis_report(shadow_df, trades_df, joined_df, rl_df, run_dir)
    
    # Save report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"SNIPER_SHADOW_REPORT_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Analysis report saved: {report_path}")
    
    # Export RL dataset
    if args.export_rl_dataset and not rl_df.empty:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Parquet
        parquet_path = args.data_dir / f"sniper_shadow_rl_dataset_{timestamp}.parquet"
        rl_df.to_parquet(parquet_path, index=False)
        log.info(f"RL dataset (Parquet) saved: {parquet_path}")
        
        # CSV
        csv_path = args.data_dir / f"sniper_shadow_rl_dataset_{timestamp}.csv"
        rl_df.to_csv(csv_path, index=False)
        log.info(f"RL dataset (CSV) saved: {csv_path}")
    
    log.info("✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())

