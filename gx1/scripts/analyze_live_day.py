#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER Daily Report V2

Generates a daily analysis report for SNIPER live trading runs, including:
- Trade logging sanity checks
- Real trades analysis
- Shadow & counterfactual analysis
- EntryCritic & TimingCritic scoring and analysis
- Top lists (missed opportunities, AVOID_TRADE trades, etc.)

This is offline analysis only - no runtime code is modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import runtime helpers for feature preparation
from gx1.rl.entry_critic_runtime import (
    load_entry_critic_v1,
    prepare_entry_critic_features,
    score_entry_critic,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_timing_critic(
    model_path: Path,
    meta_path: Path,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[List[str]]]:
    """Load Entry Timing Critic V1 model."""
    if not model_path.exists():
        log.warning(f"Timing Critic model not found: {model_path}")
        return None, None, None
    if not meta_path.exists():
        log.warning(f"Timing Critic metadata not found: {meta_path}")
        return None, None, None
    
    try:
        log.info(f"Loading Entry Timing Critic V1: {model_path}")
        model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        feature_order = meta.get("features", [])
        log.info(f"Timing Critic loaded: {meta.get('model_type')}, {len(feature_order)} features")
        
        return model, meta, feature_order
    except Exception as e:
        log.error(f"Failed to load Timing Critic: {e}", exc_info=True)
        return None, None, None


def prepare_timing_features(
    df: pd.DataFrame,
    feature_order: List[str],
    encoder_info: Dict[str, Any],
) -> pd.DataFrame:
    """Prepare features for Timing Critic."""
    feature_data = {}
    
    for feat in feature_order:
        if feat in df.columns:
            feature_data[feat] = df[feat].fillna(0.0)
        else:
            # Create default value
            feature_data[feat] = 0.0
    
    feature_df = pd.DataFrame(feature_data)
    
    # Encode categorical features if needed
    if "encoders" in encoder_info:
        for col, enc_info in encoder_info["encoders"].items():
            if col in feature_df.columns and enc_info.get("type") == "LabelEncoder":
                # Map values to encoded integers
                classes = enc_info.get("classes", [])
                if len(classes) > 0:
                    # Create mapping
                    mapping = {cls: idx for idx, cls in enumerate(classes)}
                    # Map with default to 0 for unknown
                    feature_df[col] = df[col].fillna("UNKNOWN").map(mapping).fillna(0)
    
    # Ensure all features are present
    feature_df = feature_df.reindex(columns=feature_order, fill_value=0.0)
    
    return feature_df


def score_critics(
    df: pd.DataFrame,
    score_entry: bool = True,
    score_timing: bool = True,
) -> pd.DataFrame:
    """
    Apply EntryCritic and TimingCritic scoring to dataset.
    
    Returns DataFrame with added columns:
    - entry_critic_score_v1 (if score_entry=True)
    - entry_critic_score_bucket (if score_entry=True)
    - timing_critic_pred (if score_timing=True)
    - timing_critic_p_avoid (if score_timing=True)
    """
    result_df = df.copy()
    
    # Score Entry Critic
    if score_entry:
        entry_model_path = Path("gx1/models/entry_critic_v1.joblib")
        entry_meta_path = Path("gx1/models/entry_critic_v1_meta.json")
        
        entry_model, entry_meta, entry_feature_order = load_entry_critic_v1(
            entry_model_path, entry_meta_path
        )
        
        if entry_model is not None and entry_meta is not None:
            log.info("Calculating Entry Critic scores...")
            entry_scores = []
            entry_buckets = []
            
            for idx, (_, row) in enumerate(result_df.iterrows()):
                if idx % 100 == 0 and idx > 0:
                    log.info(f"  Processed {idx}/{len(result_df)} entries...")
                
                try:
                    # Prepare features for Entry Critic
                    shadow_hits = {}
                    for thr in [0.55, 0.58, 0.60, 0.62, 0.65]:
                        col = f"shadow_hit_{thr:.2f}".replace('.', '')
                        if col in row.index:
                            shadow_hits[thr] = bool(row[col])
                    
                    feature_vector = prepare_entry_critic_features(
                        p_long=row.get("p_long", 0.0),
                        spread_bps=row.get("spread_bps"),
                        atr_bps=row.get("atr_bps"),
                        trend_regime=row.get("trend_regime"),
                        vol_regime=row.get("vol_regime"),
                        session=row.get("session"),
                        shadow_hits=shadow_hits,
                        real_threshold=row.get("real_threshold", 0.67),
                        feature_order=entry_feature_order,
                    )
                    
                    if feature_vector is not None:
                        score = score_entry_critic(entry_model, feature_vector)
                        entry_scores.append(score)
                        
                        # Bucket score
                        if score < 0.5:
                            bucket = "[0.0-0.5)"
                        elif score < 0.6:
                            bucket = "[0.5-0.6)"
                        elif score < 0.7:
                            bucket = "[0.6-0.7)"
                        elif score < 0.8:
                            bucket = "[0.7-0.8)"
                        elif score < 0.9:
                            bucket = "[0.8-0.9)"
                        else:
                            bucket = "[0.9-1.0]"
                        entry_buckets.append(bucket)
                    else:
                        entry_scores.append(None)
                        entry_buckets.append(None)
                except Exception as e:
                    log.debug(f"Failed to calculate Entry Critic score for row {idx}: {e}")
                    entry_scores.append(None)
                    entry_buckets.append(None)
            
            result_df["entry_critic_score_v1"] = entry_scores
            result_df["entry_critic_score_bucket"] = entry_buckets
            log.info(f"Entry Critic scoring complete: {len([s for s in entry_scores if s is not None])} scores calculated")
        else:
            log.warning("Entry Critic scoring skipped - model/feature mismatch")
    
    # Score Timing Critic
    if score_timing:
        timing_model_path = Path("gx1/models/entry_timing_critic_v1.joblib")
        timing_meta_path = Path("gx1/models/entry_timing_critic_v1_meta.json")
        
        timing_model, timing_meta, timing_feature_order = load_timing_critic(
            timing_model_path, timing_meta_path
        )
        
        if timing_model is not None and timing_meta is not None:
            log.info("Calculating Timing Critic predictions...")
            
            # Prepare features
            timing_features = prepare_timing_features(
                result_df, timing_feature_order, timing_meta
            )
            
            # Get predictions and probabilities
            timing_pred = timing_model.predict(timing_features)
            timing_proba = timing_model.predict_proba(timing_features)
            
            # Map integer predictions to class labels
            classes = timing_meta.get("classes", ["IMMEDIATE_OK", "DELAY_BETTER", "AVOID_TRADE"])
            if len(classes) == 3:
                class_map = {0: classes[0], 1: classes[1], 2: classes[2]}
            else:
                # Fallback mapping
                class_map = {0: "IMMEDIATE_OK", 1: "DELAY_BETTER", 2: "AVOID_TRADE"}
            
            # Handle both integer and string predictions
            timing_pred_str = []
            for p in timing_pred:
                if isinstance(p, (int, np.integer)):
                    timing_pred_str.append(class_map.get(int(p), "UNKNOWN"))
                elif isinstance(p, str):
                    # Already a string, use as-is if it's valid
                    timing_pred_str.append(p if p in classes else "UNKNOWN")
                else:
                    timing_pred_str.append("UNKNOWN")
            
            # Get probability of AVOID_TRADE
            avoid_idx = None
            for idx, cls in enumerate(classes):
                if cls == "AVOID_TRADE":
                    avoid_idx = idx
                    break
            
            if avoid_idx is not None:
                timing_p_avoid = timing_proba[:, avoid_idx]
            else:
                timing_p_avoid = np.zeros(len(result_df))
            
            result_df["timing_critic_pred"] = timing_pred_str
            result_df["timing_critic_p_avoid"] = timing_p_avoid
            
            log.info(f"Timing Critic scoring complete: {len(result_df)} predictions")
        else:
            log.warning("Timing Critic scoring skipped - model/feature mismatch")
    
    return result_df


def load_trade_journal_index(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load trade journal index CSV."""
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        return None
    
    try:
        df = pd.read_csv(index_path)
        return df
    except Exception as e:
        log.warning(f"Failed to load trade journal index: {e}")
        return None


def generate_report(
    rl_df: pd.DataFrame,
    cf_df: Optional[pd.DataFrame],
    trade_journal_df: Optional[pd.DataFrame],
    run_dir: Path,
    date: str,
    critics_scored: bool,
) -> str:
    """Generate markdown report."""
    lines = []
    
    lines.append(f"# SNIPER Daily Report V2 – {date}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"**Run Directory:** {run_dir}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Section 1: Trade Logging Sanity
    lines.append("## 1. Trade Logging Sanity")
    lines.append("")
    
    if trade_journal_df is not None:
        journal_count = len(trade_journal_df)
        lines.append(f"- **Trades in journal index:** {journal_count}")
        
        if "entry_time" in trade_journal_df.columns:
            journal_df = trade_journal_df.copy()
            journal_df["entry_time"] = pd.to_datetime(journal_df["entry_time"])
            journal_df = journal_df[journal_df["entry_time"].dt.date == pd.to_datetime(date).date()]
            lines.append(f"- **Trades on {date}:** {len(journal_df)}")
    else:
        lines.append("- **Trade journal index:** Not found")
    
    real_trades = rl_df[rl_df["action_taken"] == 1]
    lines.append(f"- **Real trades in RL dataset:** {len(real_trades)}")
    lines.append("")
    
    # Section 2: Real Trades
    lines.append("## 2. Real Trades")
    lines.append("")
    
    if len(real_trades) > 0:
        lines.append(f"**Count:** {len(real_trades)}")
        lines.append("")
        
        if "pnl_bps" in real_trades.columns:
            pnl_col = "pnl_bps"
        elif "reward" in real_trades.columns:
            pnl_col = "reward"
        else:
            pnl_col = None
        
        if pnl_col is not None:
            pnl_values = real_trades[pnl_col].dropna()
            if len(pnl_values) > 0:
                lines.append(f"- **Avg PnL (bps):** {pnl_values.mean():.2f}")
                lines.append(f"- **Median PnL (bps):** {pnl_values.median():.2f}")
                lines.append(f"- **Win Rate:** {(pnl_values > 0).sum() / len(pnl_values) * 100:.1f}%")
                lines.append(f"- **P95 PnL (bps):** {pnl_values.quantile(0.95):.2f}")
                lines.append(f"- **P05 PnL (bps):** {pnl_values.quantile(0.05):.2f}")
    else:
        lines.append("**No real trades found.**")
    lines.append("")
    
    # Section 3: Shadow & Counterfactual
    lines.append("## 3. Shadow & Counterfactual")
    lines.append("")
    
    shadow_trades = rl_df[rl_df["action_taken"] == 0]
    lines.append(f"**Shadow entries:** {len(shadow_trades)}")
    
    if cf_df is not None:
        lines.append(f"**Counterfactual trades simulated:** {len(cf_df)}")
        
        if "pnl_bps" in cf_df.columns:
            cf_pnl = cf_df["pnl_bps"].dropna()
            if len(cf_pnl) > 0:
                lines.append(f"- **Avg PnL (bps):** {cf_pnl.mean():.2f}")
                lines.append(f"- **Win Rate:** {(cf_pnl > 0).sum() / len(cf_pnl) * 100:.1f}%")
    else:
        lines.append("**Counterfactual dataset:** Not loaded")
    lines.append("")
    
    # Section 4: Timing & Dips (if CF available)
    if cf_df is not None and "mae_bps" in cf_df.columns:
        lines.append("## 4. Timing & Dips")
        lines.append("")
        
        mae_values = cf_df["mae_bps"].dropna()
        if len(mae_values) > 0:
            lines.append(f"- **Avg MAE (bps):** {mae_values.mean():.2f}")
            lines.append(f"- **P05 MAE (bps):** {mae_values.quantile(0.05):.2f}")
        
        if "mfe_bps" in cf_df.columns:
            mfe_values = cf_df["mfe_bps"].dropna()
            if len(mfe_values) > 0:
                lines.append(f"- **Avg MFE (bps):** {mfe_values.mean():.2f}")
        lines.append("")
    
    # Section 5: Shadow vs Real Summary
    lines.append("## 5. Shadow vs Real Summary")
    lines.append("")
    
    if len(real_trades) > 0 and cf_df is not None:
        real_pnl = real_trades.get("pnl_bps", pd.Series()).dropna()
        cf_pnl = cf_df.get("pnl_bps", pd.Series()).dropna()
        
        if len(real_pnl) > 0 and len(cf_pnl) > 0:
            lines.append(f"- **Real trades avg PnL:** {real_pnl.mean():.2f} bps")
            lines.append(f"- **Shadow sim avg PnL:** {cf_pnl.mean():.2f} bps")
            lines.append(f"- **Difference:** {cf_pnl.mean() - real_pnl.mean():.2f} bps")
    lines.append("")
    
    # Section 6: EntryCritic vs Real Trades (V2)
    if critics_scored and "entry_critic_score_v1" in rl_df.columns:
        lines.append("## 6. EntryCritic vs Real Trades")
        lines.append("")
        
        real_with_scores = real_trades[real_trades["entry_critic_score_v1"].notna()]
        
        if len(real_with_scores) > 0:
            avg_score = real_with_scores["entry_critic_score_v1"].mean()
            lines.append(f"- **Avg EntryCritic score (real trades):** {avg_score:.3f}")
            lines.append("")
            
            # Score buckets
            if "entry_critic_score_bucket" in real_with_scores.columns:
                bucket_counts = real_with_scores["entry_critic_score_bucket"].value_counts().sort_index()
                lines.append("**Score Distribution:**")
                lines.append("")
                lines.append("| Bucket | Count |")
                lines.append("|--------|-------|")
                for bucket, count in bucket_counts.items():
                    lines.append(f"| {bucket} | {count} |")
                lines.append("")
            
            # Win rate by score bucket
            if "pnl_bps" in real_with_scores.columns or "reward" in real_with_scores.columns:
                pnl_col = "pnl_bps" if "pnl_bps" in real_with_scores.columns else "reward"
                pnl_values = real_with_scores[pnl_col].dropna()
                
                if len(pnl_values) > 0:
                    high_score = real_with_scores[real_with_scores["entry_critic_score_v1"] >= 0.8]
                    if len(high_score) > 0:
                        high_pnl = high_score[pnl_col].dropna()
                        if len(high_pnl) > 0:
                            win_rate = (high_pnl > 0).sum() / len(high_pnl) * 100
                            avg_pnl = high_pnl.mean()
                            lines.append(f"- **Trades med score ≥ 0.8:** {len(high_score)} trades, win_rate={win_rate:.1f}%, avg_pnl={avg_pnl:.2f} bps")
        else:
            lines.append("**No real trades with EntryCritic scores.**")
        lines.append("")
    
    # Section 7: TimingCritic vs Counterfactual (V2)
    if critics_scored and "timing_critic_pred" in rl_df.columns and cf_df is not None:
        lines.append("## 7. TimingCritic vs Counterfactual")
        lines.append("")
        
        # Merge timing predictions with CF data
        # Use candle_time as both datasets have it
        if "candle_time" in rl_df.columns and "candle_time" in cf_df.columns:
            # Normalize timestamps
            rl_df_ts = pd.to_datetime(rl_df["candle_time"])
            cf_df_ts = pd.to_datetime(cf_df["candle_time"])
            
            # Create merge key (floor to 5min)
            rl_df_copy = rl_df.copy()
            cf_df_copy = cf_df.copy()
            if hasattr(rl_df_ts, 'dt'):
                rl_df_copy["merge_ts"] = rl_df_ts.dt.floor("5min")
            else:
                rl_df_copy["merge_ts"] = pd.to_datetime(rl_df_ts).floor("5min")
            
            if hasattr(cf_df_ts, 'dt'):
                cf_df_copy["merge_ts"] = cf_df_ts.dt.floor("5min")
            else:
                cf_df_copy["merge_ts"] = pd.to_datetime(cf_df_ts).floor("5min")
            
            # Merge
            cf_with_timing = cf_df_copy.merge(
                rl_df_copy[["merge_ts", "timing_critic_pred", "timing_critic_p_avoid"]],
                on="merge_ts",
                how="left",
                suffixes=("", "_rl")
            )
            
            if "timing_critic_pred" in cf_with_timing.columns:
                timing_col = "timing_critic_pred"
            else:
                timing_col = "timing_critic_pred_rl"
            
            if timing_col in cf_with_timing.columns:
                lines.append("**Timing Quality Distribution (Shadow Simulated):**")
                lines.append("")
                lines.append("| Quality | Count | Avg PnL (bps) | Avg MAE (bps) | Avg MFE (bps) |")
                lines.append("|---------|-------|---------------|---------------|---------------|")
                
                for quality in ["IMMEDIATE_OK", "DELAY_BETTER", "AVOID_TRADE"]:
                    quality_trades = cf_with_timing[cf_with_timing[timing_col] == quality]
                    if len(quality_trades) > 0:
                        count = len(quality_trades)
                        avg_pnl = quality_trades.get("pnl_bps", pd.Series()).mean() if "pnl_bps" in quality_trades.columns else 0.0
                        avg_mae = quality_trades.get("mae_bps", pd.Series()).mean() if "mae_bps" in quality_trades.columns else 0.0
                        avg_mfe = quality_trades.get("mfe_bps", pd.Series()).mean() if "mfe_bps" in quality_trades.columns else 0.0
                        lines.append(f"| {quality} | {count} | {avg_pnl:.2f} | {avg_mae:.2f} | {avg_mfe:.2f} |")
                
                lines.append("")
                
                # Interpretation
                avoid_trades = cf_with_timing[cf_with_timing[timing_col] == "AVOID_TRADE"]
                if len(avoid_trades) > 0:
                    avoid_pnl = avoid_trades.get("pnl_bps", pd.Series()).dropna()
                    if len(avoid_pnl) > 0 and avoid_pnl.mean() < 0:
                        lines.append("✅ **Hvis modellen sier AVOID_TRADE og PnL er konsekvent negativ, det er bra.**")
                        lines.append("")
                
                delay_trades = cf_with_timing[cf_with_timing[timing_col] == "DELAY_BETTER"]
                if len(delay_trades) > 0:
                    delay_pnl = delay_trades.get("pnl_bps", pd.Series()).dropna()
                    delay_mae = delay_trades.get("mae_bps", pd.Series()).dropna()
                    if len(delay_pnl) > 0 and len(delay_mae) > 0 and delay_pnl.mean() > 0 and delay_mae.mean() < -50:
                        lines.append("⚠️ **Hvis modellen sier DELAY_BETTER og PnL er positiv med høy MAE, dette peker på timing-forbedring.**")
                        lines.append("")
        else:
            lines.append("**Cannot merge timing predictions with counterfactual data.**")
        lines.append("")
    
    # Section 8: Top Lists (V2)
    if critics_scored:
        lines.append("## 8. Top Lists – Critic Perspective")
        lines.append("")
        
        # 8.1: Trades modellen hater (AVOID_TRADE på real trades)
        lines.append("### 8.1 Trades modellen hater (AVOID_TRADE på real trades)")
        lines.append("")
        
        if "timing_critic_pred" in rl_df.columns:
            avoid_real = rl_df[
                (rl_df["action_taken"] == 1) &
                (rl_df["timing_critic_pred"] == "AVOID_TRADE")
            ]
            
            if len(avoid_real) > 0:
                # Sort by worst PnL
                if "pnl_bps" in avoid_real.columns:
                    avoid_real = avoid_real.sort_values("pnl_bps", ascending=True)
                elif "reward" in avoid_real.columns:
                    avoid_real = avoid_real.sort_values("reward", ascending=True)
                
                top_avoid = avoid_real.head(10)
                
                lines.append("| Time | p_long | EntryCritic Score | Timing Pred | PnL (bps) |")
                lines.append("|------|--------|-------------------|-------------|-----------|")
                
                for _, row in top_avoid.iterrows():
                    ts = row.get("ts", row.get("candle_time", "N/A"))
                    if isinstance(ts, pd.Timestamp):
                        ts_str = ts.strftime("%Y-%m-%d %H:%M")
                    else:
                        ts_str = str(ts)
                    
                    p_long = row.get("p_long", "N/A")
                    entry_score = row.get("entry_critic_score_v1", "N/A")
                    timing_pred = row.get("timing_critic_pred", "N/A")
                    pnl = row.get("pnl_bps", row.get("reward", "N/A"))
                    
                    lines.append(f"| {ts_str} | {p_long:.3f} | {entry_score:.3f} | {timing_pred} | {pnl:.2f} |")
            else:
                lines.append("**Ingen real-trades merket AVOID_TRADE av modellen i dag.**")
        else:
            lines.append("**TimingCritic predictions not available.**")
        lines.append("")
        
        # 8.2: Missed opportunities modellen elsket
        lines.append("### 8.2 Missed opportunities modellen elsket")
        lines.append("")
        
        if cf_df is not None:
            # Merge with RL dataset to get critic scores
            if "candle_time" in rl_df.columns and "candle_time" in cf_df.columns:
                rl_df_ts = pd.to_datetime(rl_df["candle_time"])
                cf_df_ts = pd.to_datetime(cf_df["candle_time"])
                
                rl_df_copy = rl_df.copy()
                cf_df_copy = cf_df.copy()
                if hasattr(rl_df_ts, 'dt'):
                    rl_df_copy["merge_ts"] = rl_df_ts.dt.floor("5min")
                else:
                    rl_df_copy["merge_ts"] = pd.to_datetime(rl_df_ts).floor("5min")
                
                if hasattr(cf_df_ts, 'dt'):
                    cf_df_copy["merge_ts"] = cf_df_ts.dt.floor("5min")
                else:
                    cf_df_copy["merge_ts"] = pd.to_datetime(cf_df_ts).floor("5min")
                
                cf_with_scores = cf_df_copy.merge(
                    rl_df_copy[["merge_ts", "entry_critic_score_v1", "timing_critic_pred"]],
                    on="merge_ts",
                    how="left"
                )
                
                # Filter: profitable + high score
                missed = cf_with_scores[
                    (cf_with_scores.get("pnl_bps", 0) > 0) &
                    (cf_with_scores.get("entry_critic_score_v1", 0) >= 0.8)
                ]
                
                if len(missed) > 0:
                    missed = missed.sort_values("pnl_bps", ascending=False)
                    top_missed = missed.head(10)
                    
                    lines.append("| Time | p_long | EntryCritic Score | PnL (bps) | MAE (bps) | Timing Pred |")
                    lines.append("|------|--------|-------------------|-----------|-----------|-------------|")
                    
                    for _, row in top_missed.iterrows():
                        ts = row.get("ts", "N/A")
                        if isinstance(ts, pd.Timestamp):
                            ts_str = ts.strftime("%Y-%m-%d %H:%M")
                        else:
                            ts_str = str(ts)
                        
                        p_long = row.get("p_long", "N/A")
                        entry_score = row.get("entry_critic_score_v1", "N/A")
                        pnl = row.get("pnl_bps", "N/A")
                        mae = row.get("mae_bps", "N/A")
                        timing_pred = row.get("timing_critic_pred", "N/A")
                        
                        lines.append(f"| {ts_str} | {p_long:.3f} | {entry_score:.3f} | {pnl:.2f} | {mae:.2f} | {timing_pred} |")
                else:
                    lines.append("**Ingen profitable shadow trades med EntryCritic score ≥ 0.8.**")
            else:
                lines.append("**Cannot merge counterfactual with critic scores.**")
        else:
            lines.append("**Counterfactual dataset not available.**")
        lines.append("")
        
        # 8.3: Dype dips men positiv
        lines.append("### 8.3 Dype dips men positiv")
        lines.append("")
        
        if cf_df is not None:
            deep_dips = cf_df[
                (cf_df.get("pnl_bps", 0) > 0) &
                (cf_df.get("mae_bps", 0) < -50)
            ]
            
            if len(deep_dips) > 0:
                deep_dips = deep_dips.sort_values("mae_bps", ascending=True)
                top_dips = deep_dips.head(10)
                
                # Merge with RL for critic scores
                if "candle_time" in rl_df.columns and "candle_time" in top_dips.columns:
                    rl_df_ts = pd.to_datetime(rl_df["candle_time"])
                    top_dips_ts = pd.to_datetime(top_dips["candle_time"])
                    
                    rl_df_copy = rl_df.copy()
                    top_dips_copy = top_dips.copy()
                    if hasattr(rl_df_ts, 'dt'):
                        rl_df_copy["merge_ts"] = rl_df_ts.dt.floor("5min")
                    else:
                        rl_df_copy["merge_ts"] = pd.to_datetime(rl_df_ts).floor("5min")
                    
                    if hasattr(top_dips_ts, 'dt'):
                        top_dips_copy["merge_ts"] = top_dips_ts.dt.floor("5min")
                    else:
                        top_dips_copy["merge_ts"] = pd.to_datetime(top_dips_ts).floor("5min")
                    
                    top_dips = top_dips_copy.merge(
                        rl_df_copy[["merge_ts", "entry_critic_score_v1", "timing_critic_pred"]],
                        on="merge_ts",
                        how="left"
                    )
                
                lines.append("| Time | p_long | PnL (bps) | MAE (bps) | Bars to Profit | EntryCritic Score | Timing Pred |")
                lines.append("|------|--------|-----------|-----------|----------------|-------------------|-------------|")
                
                for _, row in top_dips.iterrows():
                    ts = row.get("ts", "N/A")
                    if isinstance(ts, pd.Timestamp):
                        ts_str = ts.strftime("%Y-%m-%d %H:%M")
                    else:
                        ts_str = str(ts)
                    
                    p_long = row.get("p_long", "N/A")
                    pnl = row.get("pnl_bps", "N/A")
                    mae = row.get("mae_bps", "N/A")
                    bars_to_profit = row.get("bars_to_first_profit", "N/A")
                    entry_score = row.get("entry_critic_score_v1", "N/A")
                    timing_pred = row.get("timing_critic_pred", "N/A")
                    
                    lines.append(f"| {ts_str} | {p_long:.3f} | {pnl:.2f} | {mae:.2f} | {bars_to_profit} | {entry_score:.3f} | {timing_pred} |")
            else:
                lines.append("**Ingen profitable trades med MAE < -50 bps.**")
        else:
            lines.append("**Counterfactual dataset not available.**")
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("**Note:** This is offline analysis only. No runtime code or policies were modified.")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/analyze_live_day.py` (V2)*")
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SNIPER Daily Report V2 - Analyze live trading day with critic scoring"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("runs/live_demo/SNIPER_20251226_113527"),
        help="SNIPER run directory",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to analyze (YYYY-MM-DD, optional - inferred from run_dir if not provided)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/rl/sniper_shadow_rl_dataset_LIVE_DAY1_WITH_TRADES.parquet"),
        help="Path to RL dataset Parquet file",
    )
    parser.add_argument(
        "--cf_dataset",
        type=Path,
        default=Path("data/rl/shadow_counterfactual_LIVE_DAY1_V2.parquet"),
        help="Path to shadow counterfactual V2 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/live"),
        help="Output directory for report",
    )
    parser.add_argument(
        "--score_critics",
        action="store_true",
        default=True,
        help="Apply EntryCritic and TimingCritic scoring (default: True)",
    )
    parser.add_argument(
        "--no-score-critics",
        dest="score_critics",
        action="store_false",
        help="Skip critic scoring",
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.date is None:
        # Try to infer from run_dir name
        date_str = args.run_dir.name.split("_")[-1] if "_" in args.run_dir.name else None
        if date_str and len(date_str) >= 8:
            try:
                date_obj = datetime.strptime(date_str[:8], "%Y%m%d")
                args.date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                args.date = datetime.now().strftime("%Y-%m-%d")
        else:
            args.date = datetime.now().strftime("%Y-%m-%d")
    
    log.info(f"Analyzing live day: {args.date}")
    log.info(f"Run directory: {args.run_dir}")
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Counterfactual dataset: {args.cf_dataset}")
    log.info(f"Score critics: {args.score_critics}")
    
    # Load RL dataset
    if not args.dataset.exists():
        log.error(f"RL dataset not found: {args.dataset}")
        return 1
    
    log.info(f"Loading RL dataset: {args.dataset}")
    rl_df = pd.read_parquet(args.dataset)
    log.info(f"Loaded {len(rl_df)} rows from RL dataset")
    
    # Load counterfactual dataset
    cf_df = None
    if args.cf_dataset.exists():
        log.info(f"Loading counterfactual dataset: {args.cf_dataset}")
        cf_df = pd.read_parquet(args.cf_dataset)
        log.info(f"Loaded {len(cf_df)} rows from counterfactual dataset")
    else:
        log.warning(f"Counterfactual dataset not found: {args.cf_dataset}")
    
    # Load trade journal index
    trade_journal_df = load_trade_journal_index(args.run_dir)
    if trade_journal_df is not None:
        log.info(f"Loaded {len(trade_journal_df)} trades from journal index")
    
    # Score critics if requested
    critics_scored = False
    if args.score_critics:
        try:
            rl_df = score_critics(rl_df, score_entry=True, score_timing=True)
            critics_scored = True
            log.info("Critic scoring complete")
        except Exception as e:
            log.error(f"Failed to score critics: {e}", exc_info=True)
            log.warning("Continuing without critic scores")
    
    # Generate report
    report = generate_report(
        rl_df=rl_df,
        cf_df=cf_df,
        trade_journal_df=trade_journal_df,
        run_dir=args.run_dir,
        date=args.date,
        critics_scored=critics_scored,
    )
    
    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"DAILY_REPORT_{args.date.replace('-', '_')}_V2.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    log.info(f"Report saved: {report_path}")
    log.info("✅ Daily report generation complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())

