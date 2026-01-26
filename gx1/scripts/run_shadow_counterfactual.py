#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shadow Counterfactual Backtesting

Simulates exit logic on shadow-only entries to create hypothetical trades.
This is purely offline analysis - no runtime code is modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import exit logic (read-only, no modifications)
from gx1.policy.exit_farm_v2_rules import ExitFarmV2Rules, ExitDecision
from gx1.policy.exit_farm_v2_rules_adaptive import ExitFarmV2RulesAdaptive
from gx1.utils.pnl import compute_pnl_bps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Constants for timing quality analysis
TIMING_QUALITY_MAE_HEAVY_BPS = 150.0  # MAE threshold for "heavy" drawdown (Y_MAE_HEAVY) - for AVOID_TRADE
TIMING_QUALITY_MAE_DELAY_BPS = 50.0  # MAE threshold for "delay better" (lower threshold for profitable trades)
TIMING_QUALITY_BETTER_ENTRY_BPS = 75.0  # Threshold for "clearly better entry possible" (Y_BETTER_ENTRY)


def load_raw_candles(candle_path: Path) -> pd.DataFrame:
    """Load raw M5 candle data."""
    if not candle_path.exists():
        raise FileNotFoundError(f"Raw candle data not found: {candle_path}")
    
    log.info(f"Loading raw candles: {candle_path}")
    df = pd.read_parquet(candle_path)
    
    # Normalize timestamp column
    if "ts" not in df.columns and "time" in df.columns:
        df["ts"] = pd.to_datetime(df["time"])
    elif "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["ts"] = df.index
        else:
            raise ValueError("Cannot find timestamp column in candle data")
    
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    
    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in candle data: {missing}")
    
    # Add bid/ask if available, otherwise use mid
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    elif "mid" not in df.columns:
        df["mid"] = df["close"]
    
    df = df.sort_values("ts").reset_index(drop=True)
    log.info(f"Loaded {len(df)} candles from {df['ts'].min()} to {df['ts'].max()}")
    
    return df


def get_price_series_after_entry(
    candles: pd.DataFrame,
    entry_ts: pd.Timestamp,
    max_bars: int = 100,
) -> Optional[pd.DataFrame]:
    """Get price series after entry timestamp."""
    # Find entry bar (first bar >= entry_ts)
    entry_idx = candles[candles["ts"] >= entry_ts].index
    if len(entry_idx) == 0:
        return None
    
    entry_idx = entry_idx[0]
    end_idx = min(entry_idx + max_bars, len(candles))
    
    price_series = candles.iloc[entry_idx:end_idx].copy()
    return price_series


def simulate_exit_rule5(
    price_series: pd.DataFrame,
    entry_price: float,
    side: str = "long",
    config: Optional[Dict[str, Any]] = None,
    track_path: bool = False,
) -> Tuple[ExitDecision, Optional[List[float]]]:
    """
    Simulate RULE5 exit logic on price series using actual ExitFarmV2Rules.
    
    Uses SNIPER_EXIT_RULES_A config by default.
    """
    if config is None:
        config = {
            "enable_rule_a": True,
            "enable_rule_b": False,
            "enable_rule_c": False,
            "rule_a_adaptive_bars": 3,
            "rule_a_adaptive_threshold_bps": 4.0,
            "rule_a_profit_max_bps": 9.0,
            "rule_a_profit_min_bps": 5.0,
            "rule_a_trailing_stop_bps": 2.0,
            "force_exit_bars": 100,  # Max hold time
        }
    
    # Initialize exit policy
    exit_policy = ExitFarmV2Rules(**config)
    
    # Get entry bar
    entry_bar = price_series.iloc[0]
    entry_bid = entry_bar.get("bid", entry_bar["mid"])
    entry_ask = entry_bar.get("ask", entry_bar["mid"])
    entry_ts = entry_bar["ts"]
    
    # Reset exit policy for entry
    exit_policy.reset_on_entry(
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        entry_ts=entry_ts,
        side=side,
    )
    
    # Track PnL path if requested
    pnl_path_bps: Optional[List[float]] = [] if track_path else None
    
    # Process each bar
    for bar_idx, bar in price_series.iterrows():
        bar_bid = bar.get("bid", bar["mid"])
        bar_ask = bar.get("ask", bar["mid"])
        bar_ts = bar["ts"]
        
        # Calculate current PnL for path tracking
        if track_path:
            current_pnl = compute_pnl_bps(
                entry_bid=exit_policy.entry_bid,
                entry_ask=exit_policy.entry_ask,
                exit_bid=bar_bid,
                exit_ask=bar_ask,
                side=side,
            )
            pnl_path_bps.append(current_pnl)
        
        # Check for exit
        exit_decision = exit_policy.on_bar(
            price_bid=bar_bid,
            price_ask=bar_ask,
            ts=bar_ts,
        )
        
        if exit_decision is not None:
            return exit_decision, pnl_path_bps
    
    # Force exit at end of series
    final_bar = price_series.iloc[-1]
    final_bid = final_bar.get("bid", final_bar["mid"])
    final_ask = final_bar.get("ask", final_bar["mid"])
    final_ts = final_bar["ts"]
    
    # Calculate final PnL
    final_pnl = compute_pnl_bps(
        entry_bid=exit_policy.entry_bid,
        entry_ask=exit_policy.entry_ask,
        exit_bid=final_bid,
        exit_ask=final_ask,
        side=side,
    )
    exit_price = final_bid if side == "long" else final_ask
    
    exit_decision = ExitDecision(
        exit_price=exit_price,
        reason="FORCE_CLOSE_END_OF_DATA",
        bars_held=exit_policy.bars_held,
        pnl_bps=final_pnl,
        mae_bps=exit_policy.mae_bps,
        mfe_bps=exit_policy.mfe_bps,
    )
    
    return exit_decision, pnl_path_bps


def simulate_exit_rule6a(
    price_series: pd.DataFrame,
    entry_price: float,
    side: str = "long",
    config: Optional[Dict[str, Any]] = None,
    track_path: bool = False,
) -> Tuple[ExitDecision, Optional[List[float]]]:
    """
    Simulate RULE6A (adaptive) exit logic on price series.
    
    Uses ExitFarmV2RulesAdaptive - simplified for now.
    """
    # For SNIPER, RULE6A is similar to RULE5 but with adaptive parameters
    # Use RULE5 simulation for now (can be enhanced later)
    return simulate_exit_rule5(price_series, entry_price, side, config, track_path=track_path)


def determine_exit_rule(
    atr_bps: Optional[float],
    spread_bps: Optional[float],
    session: str,
    trend_regime: Optional[str],
    vol_regime: Optional[str],
) -> str:
    """
    Determine which exit rule to use based on context.
    
    Simplified version - actual logic is in exit_hybrid_controller.py
    """
    # For SNIPER, default to RULE5 (SNIPER_EXIT_RULES_A)
    # In production, this would use ExitModeSelector
    return "RULE5"


def calculate_timing_metrics(
    pnl_path_bps: List[float],
    price_series: pd.DataFrame,
    entry_price: float,
    side: str = "long",
    label_profitable_10bps: int = 0,
) -> Dict[str, Any]:
    """
    Calculate timing metrics from PnL path.
    
    Returns:
        Dictionary with timing metrics
    """
    if not pnl_path_bps:
        return {
            "mae_bps": 0.0,
            "mfe_bps": 0.0,
            "bars_to_first_profit": None,
            "mae_before_first_profit_bps": None,
            "bars_to_mfe_peak": None,
            "better_entry_offset_bps": 0.0,
            "better_entry_possible": False,
            "timing_quality": "IMMEDIATE_OK",
        }
    
    # MAE and MFE from path
    mae_bps = min(pnl_path_bps)
    mfe_bps = max(pnl_path_bps)
    
    # Find first profit bar
    bars_to_first_profit = None
    mae_before_first_profit_bps = None
    for i, pnl in enumerate(pnl_path_bps):
        if pnl >= 0:
            bars_to_first_profit = i + 1  # 1-indexed
            if i > 0:
                mae_before_first_profit_bps = min(pnl_path_bps[:i])
            else:
                mae_before_first_profit_bps = 0.0
            break
    
    # Find bar with max MFE
    bars_to_mfe_peak = None
    if mfe_bps > 0:
        for i, pnl in enumerate(pnl_path_bps):
            if pnl >= mfe_bps - 0.01:  # Allow small floating point tolerance
                bars_to_mfe_peak = i + 1
                break
    
    # Better entry offset (for long: lowest price after entry)
    if side == "long":
        lowest_price = min(price_series["low"].min(), price_series["mid"].min())
        better_entry_offset_bps = (entry_price - lowest_price) / entry_price * 10000.0
    else:
        # For short: highest price after entry
        highest_price = max(price_series["high"].max(), price_series["mid"].max())
        better_entry_offset_bps = (highest_price - entry_price) / entry_price * 10000.0
    
    # Better entry possible
    better_entry_possible = (
        label_profitable_10bps == 1
        and better_entry_offset_bps > TIMING_QUALITY_BETTER_ENTRY_BPS
    )
    
    # Timing quality
    timing_quality = "IMMEDIATE_OK"
    
    if label_profitable_10bps == 1:
        # Trade ends profitable, but took heat before profit
        # Use mae_bps (overall MAE) if mae_before_first_profit_bps is not available or 0
        effective_mae = mae_before_first_profit_bps if (mae_before_first_profit_bps is not None and mae_before_first_profit_bps < 0) else mae_bps
        if (
            effective_mae is not None
            and effective_mae <= -TIMING_QUALITY_MAE_DELAY_BPS  # Lower threshold for profitable trades
            and better_entry_offset_bps is not None
            and better_entry_offset_bps >= TIMING_QUALITY_BETTER_ENTRY_BPS
        ):
            timing_quality = "DELAY_BETTER"
    elif label_profitable_10bps == 0:
        # Trade ends unprofitable and had heavy MAE
        if mae_bps is not None and mae_bps <= -TIMING_QUALITY_MAE_HEAVY_BPS:
            timing_quality = "AVOID_TRADE"
    
    return {
        "mae_bps": mae_bps,
        "mfe_bps": mfe_bps,
        "bars_to_first_profit": bars_to_first_profit,
        "mae_before_first_profit_bps": mae_before_first_profit_bps,
        "bars_to_mfe_peak": bars_to_mfe_peak,
        "better_entry_offset_bps": better_entry_offset_bps,
        "better_entry_possible": better_entry_possible,
        "timing_quality": timing_quality,
    }


def simulate_shadow_trade(
    shadow_row: pd.Series,
    candles: pd.DataFrame,
    candle_path: Path,
    track_path: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Simulate a trade for a shadow-only entry.
    
    Returns a dictionary with simulated trade results, or None if simulation fails.
    """
    try:
        # Get entry timestamp
        entry_ts = shadow_row["candle_time"]
        if pd.isna(entry_ts):
            return None
        
        # Get price series after entry
        price_series = get_price_series_after_entry(candles, entry_ts, max_bars=100)
        if price_series is None or len(price_series) == 0:
            return None
        
        # Determine entry price (use mid of entry bar)
        entry_bar = price_series.iloc[0]
        entry_price = entry_bar["mid"]
        
        # Determine side (assume long for SNIPER - could be extracted from context)
        side = "long"
        
        # Determine exit rule
        exit_rule = determine_exit_rule(
            atr_bps=shadow_row.get("atr_bps"),
            spread_bps=shadow_row.get("spread_bps"),
            session=shadow_row.get("session", "EU"),
            trend_regime=shadow_row.get("trend_regime"),
            vol_regime=shadow_row.get("vol_regime"),
        )
        
        # Simulate exit
        if exit_rule == "RULE5":
            exit_decision, pnl_path_bps = simulate_exit_rule5(
                price_series, entry_price, side, track_path=track_path
            )
        elif exit_rule == "RULE6A":
            exit_decision, pnl_path_bps = simulate_exit_rule6a(
                price_series, entry_price, side, track_path=track_path
            )
        else:
            # Fallback to RULE5
            exit_decision, pnl_path_bps = simulate_exit_rule5(
                price_series, entry_price, side, track_path=track_path
            )
        
        label_profitable_10bps = 1 if exit_decision.pnl_bps >= 10.0 else 0
        
        # Build result
        result = {
            "ts": entry_ts,
            "p_long": shadow_row.get("p_long"),
            "spread_bps": shadow_row.get("spread_bps"),
            "atr_bps": shadow_row.get("atr_bps"),
            "sim_entry_price": entry_price,
            "sim_exit_price": exit_decision.exit_price,
            "pnl_bps": exit_decision.pnl_bps,
            "max_mfe": exit_decision.mfe_bps,
            "max_mae": exit_decision.mae_bps,
            "hold_time_bars": exit_decision.bars_held,
            "exit_rule": exit_rule,
            "exit_reason": exit_decision.reason,
            "label_profitable_10bps": label_profitable_10bps,
            "session": shadow_row.get("session"),
            "trend_regime": shadow_row.get("trend_regime"),
            "vol_regime": shadow_row.get("vol_regime"),
        }
        
        # Add Entry Critic score if available
        if "entry_critic_score_v1" in shadow_row.index:
            result["entry_critic_score_v1"] = shadow_row.get("entry_critic_score_v1")
        
        # Add V2 timing metrics if tracking path
        if track_path and pnl_path_bps is not None:
            timing_metrics = calculate_timing_metrics(
                pnl_path_bps, price_series, entry_price, side, label_profitable_10bps
            )
            result.update(timing_metrics)
            # Optionally serialize PnL path (can be large, so make it optional)
            # result["pnl_path_bps_serialized"] = json.dumps(pnl_path_bps)
        
        return result
    
    except Exception as e:
        log.warning(f"Failed to simulate shadow trade for {shadow_row.get('candle_time')}: {e}")
        return None


def generate_report_v2(
    real_trades: pd.DataFrame,
    shadow_trades: pd.DataFrame,
    dataset_path: Path,
    report_path: Path,
) -> None:
    """Generate V2 markdown analysis report with timing & MAE/MFE analysis."""
    lines = []
    
    lines.append("# SHADOW COUNTERFACTUAL – FULLYEAR 2025 (V2 – Timing & MAE/MFE)")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Dataset path:** {dataset_path}")
    lines.append(f"- **Shadow trades simulated:** {len(shadow_trades):,}")
    lines.append(f"- **Real trades:** {len(real_trades):,}")
    lines.append(f"- **Period:** {shadow_trades['ts'].min()} to {shadow_trades['ts'].max()}" if len(shadow_trades) > 0 else "- **Period:** N/A")
    lines.append("- **Exit engine:** RULE5 (SNIPER_EXIT_RULES_A)")
    lines.append("")
    
    # Del 1 – Real vs Shadow (recap)
    lines.append("## Del 1 – Real vs Shadow (Recap)")
    lines.append("")
    lines.append("| Metric | Real Trades | Shadow Simulated |")
    lines.append("|--------|-------------|------------------|")
    
    if len(real_trades) > 0:
        lines.append(f"| Count | {len(real_trades):,} | {len(shadow_trades):,} |")
        lines.append(f"| Win Rate | {real_trades['label_profitable_10bps'].mean():.2%} | {shadow_trades['label_profitable_10bps'].mean():.2%} |")
        lines.append(f"| Avg PnL (bps) | {real_trades['pnl_bps'].mean():.2f} | {shadow_trades['pnl_bps'].mean():.2f} |")
        lines.append(f"| Median PnL (bps) | {real_trades['pnl_bps'].median():.2f} | {shadow_trades['pnl_bps'].median():.2f} |")
        lines.append(f"| P95 PnL (bps) | {real_trades['pnl_bps'].quantile(0.95):.2f} | {shadow_trades['pnl_bps'].quantile(0.95):.2f} |")
        lines.append(f"| P05 PnL (bps) | {real_trades['pnl_bps'].quantile(0.05):.2f} | {shadow_trades['pnl_bps'].quantile(0.05):.2f} |")
    else:
        lines.append(f"| Count | 0 | {len(shadow_trades):,} |")
        lines.append(f"| Win Rate | N/A | {shadow_trades['label_profitable_10bps'].mean():.2%} |")
        lines.append(f"| Avg PnL (bps) | N/A | {shadow_trades['pnl_bps'].mean():.2f} |")
    
    lines.append("")
    
    # Del 2 – MAE/MFE profiles
    if len(shadow_trades) > 0 and "mae_bps" in shadow_trades.columns:
        lines.append("## Del 2 – MAE/MFE Profiles")
        lines.append("")
        
        # Overall MAE/MFE stats
        lines.append("### Overall MAE/MFE Statistics (Shadow Simulated)")
        lines.append("")
        lines.append(f"- **Avg MAE (bps):** {shadow_trades['mae_bps'].mean():.2f}")
        lines.append(f"- **Avg MFE (bps):** {shadow_trades['mfe_bps'].mean():.2f}")
        
        profitable = shadow_trades[shadow_trades['label_profitable_10bps'] == 1]
        if len(profitable) > 0 and "mae_before_first_profit_bps" in profitable.columns:
            mae_before_profit = profitable['mae_before_first_profit_bps'].dropna()
            if len(mae_before_profit) > 0:
                lines.append(f"- **Avg MAE before first profit (profitable trades):** {mae_before_profit.mean():.2f} bps")
        
        if "bars_to_first_profit" in shadow_trades.columns:
            bars_to_profit = shadow_trades['bars_to_first_profit'].dropna()
            if len(bars_to_profit) > 0:
                lines.append(f"- **Avg bars to first profit:** {bars_to_profit.mean():.1f}")
        
        if "bars_to_mfe_peak" in shadow_trades.columns:
            bars_to_mfe = shadow_trades['bars_to_mfe_peak'].dropna()
            if len(bars_to_mfe) > 0:
                lines.append(f"- **Avg bars to MFE peak:** {bars_to_mfe.mean():.1f}")
        
        lines.append("")
        
        # MAE buckets
        lines.append("### MAE Buckets")
        lines.append("")
        lines.append("| MAE Bucket (bps) | Count | Win Rate | Avg PnL (bps) |")
        lines.append("|------------------|-------|----------|---------------|")
        
        buckets = [
            (0, -100, "0 to -100"),
            (-100, -500, "-100 to -500"),
            (-500, -1000, "-500 to -1000"),
            (-1000, float('-inf'), "< -1000"),
        ]
        
        for lower, upper, label in buckets:
            if upper == float('-inf'):
                mask = shadow_trades['mae_bps'] <= lower
            else:
                mask = (shadow_trades['mae_bps'] > lower) & (shadow_trades['mae_bps'] <= upper)
            bucket_trades = shadow_trades[mask]
            
            if len(bucket_trades) > 0:
                win_rate = bucket_trades['label_profitable_10bps'].mean()
                avg_pnl = bucket_trades['pnl_bps'].mean()
                lines.append(f"| {label} | {len(bucket_trades):,} | {win_rate:.2%} | {avg_pnl:.2f} |")
        
        lines.append("")
    
    # Del 3 – Timing Quality
    if len(shadow_trades) > 0 and "timing_quality" in shadow_trades.columns:
        lines.append("## Del 3 – Timing Quality")
        lines.append("")
        
        timing_counts = shadow_trades['timing_quality'].value_counts()
        lines.append("### Timing Quality Distribution")
        lines.append("")
        lines.append("| Quality | Count | Percentage |")
        lines.append("|---------|-------|------------|")
        for quality, count in timing_counts.items():
            pct = count / len(shadow_trades) * 100
            lines.append(f"| {quality} | {count:,} | {pct:.1f}% |")
        lines.append("")
        
        # Stats per timing quality
        lines.append("### Statistics by Timing Quality")
        lines.append("")
        for quality in ["IMMEDIATE_OK", "DELAY_BETTER", "AVOID_TRADE"]:
            if quality in timing_counts.index:
                quality_trades = shadow_trades[shadow_trades['timing_quality'] == quality]
                lines.append(f"#### {quality}")
                lines.append("")
                lines.append(f"- **Count:** {len(quality_trades):,}")
                lines.append(f"- **Win Rate:** {quality_trades['label_profitable_10bps'].mean():.2%}")
                lines.append(f"- **Avg PnL (bps):** {quality_trades['pnl_bps'].mean():.2f}")
                
                if "mae_before_first_profit_bps" in quality_trades.columns:
                    mae_before = quality_trades['mae_before_first_profit_bps'].dropna()
                    if len(mae_before) > 0:
                        lines.append(f"- **Avg MAE before first profit (bps):** {mae_before.mean():.2f}")
                
                if "better_entry_offset_bps" in quality_trades.columns:
                    better_offset = quality_trades['better_entry_offset_bps'].mean()
                    lines.append(f"- **Avg better entry offset (bps):** {better_offset:.2f}")
                
                lines.append("")
        
        # Top 10 DELAY_BETTER examples
        if "DELAY_BETTER" in timing_counts.index:
            delay_better = shadow_trades[shadow_trades['timing_quality'] == 'DELAY_BETTER']
            if len(delay_better) > 0:
                top_delay = delay_better.nlargest(10, "pnl_bps")
                lines.append("### Top 10 DELAY_BETTER Examples")
                lines.append("")
                lines.append("| Entry Time | p_long | MAE before profit | PnL (bps) | Better entry offset |")
                lines.append("|------------|--------|-------------------|-----------|---------------------|")
                for _, row in top_delay.iterrows():
                    mae_before = row.get('mae_before_first_profit_bps', 'N/A')
                    if mae_before != 'N/A' and pd.notna(mae_before):
                        mae_before = f"{mae_before:.2f}"
                    better_offset = row.get('better_entry_offset_bps', 'N/A')
                    if better_offset != 'N/A' and pd.notna(better_offset):
                        better_offset = f"{better_offset:.2f}"
                    lines.append(
                        f"| {row['ts']} | {row['p_long']:.3f} | {mae_before} | "
                        f"{row['pnl_bps']:.2f} | {better_offset} |"
                    )
                lines.append("")
    
    # Del 4 – Potential improvement
    if len(shadow_trades) > 0 and "timing_quality" in shadow_trades.columns:
        lines.append("## Del 4 – Potential Improvement on Baseline")
        lines.append("")
        
        # Scenario A: Only trades with timing_quality != AVOID_TRADE
        scenario_a = shadow_trades[shadow_trades['timing_quality'] != 'AVOID_TRADE']
        if len(scenario_a) > 0:
            lines.append("### Scenario A: Only trades with timing_quality != AVOID_TRADE")
            lines.append("")
            lines.append(f"- **Count:** {len(scenario_a):,} (vs {len(shadow_trades):,} total)")
            lines.append(f"- **Win Rate:** {scenario_a['label_profitable_10bps'].mean():.2%}")
            lines.append(f"- **Total PnL (bps):** {scenario_a['pnl_bps'].sum():.2f}")
            lines.append(f"- **Avg PnL (bps):** {scenario_a['pnl_bps'].mean():.2f}")
            lines.append("")
        
        # DELAY_BETTER trades
        delay_better = shadow_trades[shadow_trades['timing_quality'] == 'DELAY_BETTER']
        if len(delay_better) > 0:
            lines.append("### DELAY_BETTER Trades Summary")
            lines.append("")
            lines.append(f"- **Count:** {len(delay_better):,}")
            lines.append(f"- **Total hypotetisk ekstra PnL:** {delay_better['pnl_bps'].sum():.2f} bps")
            lines.append(f"- **Avg PnL:** {delay_better['pnl_bps'].mean():.2f} bps")
            lines.append("")
            lines.append("*Note: These trades would have been profitable with better entry timing.*")
            lines.append("")
        
        # Real trades with AVOID_TRADE characteristics (if we had timing_quality for real trades)
        if len(real_trades) > 0 and "mae_bps" in real_trades.columns:
            real_avoid = real_trades[real_trades['mae_bps'] < -TIMING_QUALITY_MAE_HEAVY_BPS]
            if len(real_avoid) > 0:
                lines.append("### Real Trades with AVOID_TRADE Characteristics")
                lines.append("")
                lines.append(f"- **Count:** {len(real_avoid):,} (out of {len(real_trades):,} real trades)")
                lines.append(f"- **Win Rate:** {real_avoid['label_profitable_10bps'].mean():.2%}")
                lines.append(f"- **Total PnL (bps):** {real_avoid['pnl_bps'].sum():.2f}")
                lines.append("")
    
    # Notes
    lines.append("---")
    lines.append("")
    lines.append("## Notater")
    lines.append("")
    lines.append("- **Ingen runtime-kode er endret.**")
    lines.append("- Dette er en ren \"alternate universe\"-analyse.")
    lines.append("- Resultatene skal brukes til design av fremtidig Entry Timing-modell")
    lines.append("  (ikke direkte endring av policy nå).")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/run_shadow_counterfactual.py` (V2)*")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"V2 report saved: {report_path}")


def generate_report(
    real_trades: pd.DataFrame,
    shadow_trades: pd.DataFrame,
    dataset_path: Path,
    report_path: Path,
) -> None:
    """Generate markdown analysis report."""
    lines = []
    
    lines.append("# Shadow Counterfactual Backtesting Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Dataset path:** {dataset_path}")
    lines.append(f"- **Real trades:** {len(real_trades):,}")
    lines.append(f"- **Shadow trades (simulated):** {len(shadow_trades):,}")
    lines.append("")
    
    # Overall statistics
    lines.append("## Overall Statistics")
    lines.append("")
    lines.append("### Real Trades")
    lines.append("")
    if len(real_trades) > 0:
        lines.append(f"- **Count:** {len(real_trades):,}")
        lines.append(f"- **Win Rate:** {real_trades['label_profitable_10bps'].mean():.2%}")
        lines.append(f"- **Avg PnL (bps):** {real_trades['pnl_bps'].mean():.2f}")
        lines.append(f"- **Total PnL (bps):** {real_trades['pnl_bps'].sum():.2f}")
        lines.append(f"- **Median PnL (bps):** {real_trades['pnl_bps'].median():.2f}")
        lines.append("")
    else:
        lines.append("- No real trades in dataset")
        lines.append("")
    
    lines.append("### Shadow Trades (Simulated)")
    lines.append("")
    if len(shadow_trades) > 0:
        lines.append(f"- **Count:** {len(shadow_trades):,}")
        lines.append(f"- **Win Rate:** {shadow_trades['label_profitable_10bps'].mean():.2%}")
        lines.append(f"- **Avg PnL (bps):** {shadow_trades['pnl_bps'].mean():.2f}")
        lines.append(f"- **Total PnL (bps):** {shadow_trades['pnl_bps'].sum():.2f}")
        lines.append(f"- **Median PnL (bps):** {shadow_trades['pnl_bps'].median():.2f}")
        lines.append("")
    else:
        lines.append("- No shadow trades simulated")
        lines.append("")
    
    # Best missed opportunities
    if len(shadow_trades) > 0:
        lines.append("## Top 100 Missed Opportunities")
        lines.append("")
        top_shadow = shadow_trades.nlargest(100, "pnl_bps")
        lines.append("| Rank | Entry Time | p_long | PnL (bps) | Max MFE | Exit Reason |")
        lines.append("|------|------------|--------|-----------|---------|-------------|")
        for idx, (_, row) in enumerate(top_shadow.iterrows(), 1):
            lines.append(
                f"| {idx} | {row['ts']} | {row['p_long']:.3f} | "
                f"{row['pnl_bps']:.2f} | {row['max_mfe']:.2f} | {row['exit_reason']} |"
            )
        lines.append("")
    
    # Combined analysis
    if len(real_trades) > 0 and len(shadow_trades) > 0:
        lines.append("## Combined Analysis")
        lines.append("")
        combined_pnl = real_trades['pnl_bps'].sum() + shadow_trades['pnl_bps'].sum()
        lines.append(f"- **Total PnL (Real + Shadow):** {combined_pnl:.2f} bps")
        lines.append(f"- **Shadow PnL as % of Real:** {(shadow_trades['pnl_bps'].sum() / real_trades['pnl_bps'].sum() * 100):.1f}%")
        lines.append("")
    
    # Notes
    lines.append("---")
    lines.append("")
    lines.append("## Notater")
    lines.append("")
    lines.append("- **Ingen runtime-kode er endret.**")
    lines.append("- Dette er ren offline-simulering.")
    lines.append("- Shadow trades er hypotetiske - de ble ikke faktisk utført.")
    lines.append("- Exit-logikken er forenklet - faktisk exit-logikk kan være mer kompleks.")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/run_shadow_counterfactual.py`*")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"Report saved: {report_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shadow Counterfactual Backtesting"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_PARALLEL.parquet"),
        help="Path to RL dataset Parquet file",
    )
    parser.add_argument(
        "--candles",
        type=Path,
        default=None,
        help="Path to raw M5 candle data (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rl/shadow_counterfactual_FULLYEAR_2025.parquet"),
        help="Output path for counterfactual dataset (V1)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/rl/SHADOW_COUNTERFACTUAL_REPORT_FULLYEAR_2025.md"),
        help="Output path for markdown report (V1)",
    )
    parser.add_argument(
        "--output_v2",
        type=Path,
        default=Path("data/rl/shadow_counterfactual_FULLYEAR_2025_V2.parquet"),
        help="Output path for V2 counterfactual dataset (with timing metrics)",
    )
    parser.add_argument(
        "--report_v2",
        type=Path,
        default=Path("reports/rl/SHADOW_COUNTERFACTUAL_REPORT_FULLYEAR_2025_V2.md"),
        help="Output path for V2 markdown report",
    )
    parser.add_argument(
        "--max_shadow",
        type=int,
        default=None,
        help="Maximum number of shadow entries to simulate (for testing)",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Enable V2 mode (timing & MAE/MFE analysis)",
    )
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        log.info(f"Loading dataset: {args.dataset}")
        df = pd.read_parquet(args.dataset)
        log.info(f"Loaded {len(df)} rows")
        
        # Split real trades and shadow-only
        real_trades = df[df["action_taken"] == 1].copy()
        shadow_only = df[df["action_taken"] == 0].copy()
        
        log.info(f"Real trades: {len(real_trades):,}")
        log.info(f"Shadow-only entries: {len(shadow_only):,}")
        
        # Limit shadow entries if specified
        if args.max_shadow is not None:
            shadow_only = shadow_only.head(args.max_shadow)
            log.info(f"Limited to {len(shadow_only)} shadow entries for testing")
        
        # Find raw candle data
        if args.candles is None:
            # Auto-detect candle data
            candle_candidates = [
                Path("data/raw/xauusd_m5_2025_bid_ask.parquet"),
                Path("data/raw/xauusd_m5_2025.parquet"),
            ]
            # Also search for any M5 2025 parquet file
            for candidate in Path("data/raw").glob("*m5*2025*.parquet"):
                candle_candidates.append(candidate)
            
            candle_path = None
            for candidate in candle_candidates:
                if candidate.exists():
                    candle_path = candidate
                    break
            
            if candle_path is None:
                raise FileNotFoundError(
                    f"Could not find raw candle data. Tried: {candle_candidates}. "
                    "Please specify --candles path."
                )
            log.info(f"Auto-detected candle data: {candle_path}")
        else:
            candle_path = args.candles
        
        # Load raw candles
        candles = load_raw_candles(candle_path)
        
        # Determine if V2 mode is enabled
        use_v2 = args.v2 or args.output_v2 != Path("data/rl/shadow_counterfactual_FULLYEAR_2025_V2.parquet") or args.report_v2 != Path("reports/rl/SHADOW_COUNTERFACTUAL_REPORT_FULLYEAR_2025_V2.md")
        
        # Simulate shadow trades
        log.info(f"Simulating shadow trades... (V2 mode: {use_v2})")
        shadow_results = []
        
        for idx, (_, shadow_row) in enumerate(shadow_only.iterrows(), 1):
            if idx % 100 == 0:
                log.info(f"Processed {idx}/{len(shadow_only)} shadow entries...")
            
            result = simulate_shadow_trade(
                shadow_row, candles, args.candles, track_path=use_v2
            )
            if result is not None:
                shadow_results.append(result)
        
        log.info(f"Simulated {len(shadow_results)} shadow trades")
        
        # Convert to DataFrame
        shadow_trades_df = pd.DataFrame(shadow_results)
        
        if len(shadow_trades_df) == 0:
            log.warning("No shadow trades simulated - check candle data availability")
            return 1
        
        # V1 output (always generated)
        combined_df_v1 = pd.concat([real_trades, shadow_trades_df], ignore_index=True)
        log.info(f"Saving V1 counterfactual dataset: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined_df_v1.to_parquet(args.output)
        log.info(f"Saved {len(combined_df_v1)} rows to {args.output}")
        
        # Generate V1 report
        generate_report(real_trades, shadow_trades_df, args.dataset, args.report)
        
        # V2 output (if enabled)
        if use_v2:
            # Add trade_type column for V2
            real_trades_v2 = real_trades.copy()
            real_trades_v2["trade_type"] = "REAL"
            
            shadow_trades_v2 = shadow_trades_df.copy()
            shadow_trades_v2["trade_type"] = "SHADOW_SIM"
            
            # Combine for V2
            combined_df_v2 = pd.concat([real_trades_v2, shadow_trades_v2], ignore_index=True)
            
            # Save V2 output
            log.info(f"Saving V2 counterfactual dataset: {args.output_v2}")
            args.output_v2.parent.mkdir(parents=True, exist_ok=True)
            combined_df_v2.to_parquet(args.output_v2)
            log.info(f"Saved {len(combined_df_v2)} rows to {args.output_v2}")
            
            # Also save shadow-only V2 dataset
            shadow_v2_path = args.output_v2.parent / f"{args.output_v2.stem}_shadow_only.parquet"
            shadow_trades_v2.to_parquet(shadow_v2_path)
            log.info(f"Saved {len(shadow_trades_v2)} shadow trades to {shadow_v2_path}")
            
            # Generate V2 report
            generate_report_v2(real_trades, shadow_trades_v2, args.dataset, args.report_v2)
        
        log.info("✅ Counterfactual backtesting complete!")
        return 0
    
    except Exception as e:
        log.error(f"❌ Counterfactual backtesting failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

