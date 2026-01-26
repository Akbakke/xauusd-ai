#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry Policy Evaluation V1

Evaluates different entry policies using EntryCritic V1 + TimingCritic V1
combined with shadow counterfactual V2.
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


def load_entry_critic(model_path: Path, meta_path: Path) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load Entry Critic V1 model."""
    log.info(f"Loading Entry Critic V1: {model_path}")
    model, meta, feature_order = load_entry_critic_v1(model_path, meta_path)
    log.info(f"Entry Critic loaded: {meta.get('model_type')}")
    return model, meta, feature_order


def load_timing_critic(model_path: Path, meta_path: Path) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load Entry Timing Critic V1 model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Timing Critic model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Timing Critic metadata not found: {meta_path}")
    
    log.info(f"Loading Entry Timing Critic V1: {model_path}")
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    feature_order = meta.get("features", [])
    log.info(f"Timing Critic loaded: {meta.get('model_type')}, {len(feature_order)} features")
    
    return model, meta, feature_order


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
        from sklearn.preprocessing import LabelEncoder
        
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


def evaluate_policy(
    df: pd.DataFrame,
    policy_name: str,
    policy_func,
) -> Dict[str, Any]:
    """
    Evaluate a policy function on the dataset.
    
    policy_func should be a function that takes a row and returns True/False.
    """
    # Apply policy
    mask = df.apply(policy_func, axis=1)
    policy_trades = df[mask].copy()
    
    if len(policy_trades) == 0:
        return {
            "policy_name": policy_name,
            "n_trades": 0,
            "real_frac": 0.0,
            "shadow_frac": 0.0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "median_reward": 0.0,
            "sum_reward": 0.0,
            "reward_per_1k": 0.0,
            "p05_reward": 0.0,
            "p95_reward": 0.0,
        }
    
    # Determine reward column
    reward_col = None
    if "reward" in policy_trades.columns:
        reward_col = "reward"
    elif "pnl_bps" in policy_trades.columns:
        reward_col = "pnl_bps"
    else:
        log.warning(f"No reward column found for policy {policy_name}")
        return {
            "policy_name": policy_name,
            "n_trades": len(policy_trades),
            "real_frac": 0.0,
            "shadow_frac": 0.0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "median_reward": 0.0,
            "sum_reward": 0.0,
            "reward_per_1k": 0.0,
            "p05_reward": 0.0,
            "p95_reward": 0.0,
        }
    
    # Calculate statistics
    real_count = (policy_trades["trade_type"] == "REAL").sum() if "trade_type" in policy_trades.columns else len(policy_trades)
    shadow_count = (policy_trades["trade_type"] == "SHADOW_SIM").sum() if "trade_type" in policy_trades.columns else 0
    
    win_rate = policy_trades["label_profitable_10bps"].mean() if "label_profitable_10bps" in policy_trades.columns else 0.0
    avg_reward = policy_trades[reward_col].mean()
    median_reward = policy_trades[reward_col].median()
    sum_reward = policy_trades[reward_col].sum()
    reward_per_1k = (sum_reward / len(policy_trades)) * 1000 if len(policy_trades) > 0 else 0.0
    p05_reward = policy_trades[reward_col].quantile(0.05)
    p95_reward = policy_trades[reward_col].quantile(0.95)
    
    return {
        "policy_name": policy_name,
        "n_trades": len(policy_trades),
        "real_frac": real_count / len(policy_trades) if len(policy_trades) > 0 else 0.0,
        "shadow_frac": shadow_count / len(policy_trades) if len(policy_trades) > 0 else 0.0,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "median_reward": median_reward,
        "sum_reward": sum_reward,
        "reward_per_1k": reward_per_1k,
        "p05_reward": p05_reward,
        "p95_reward": p95_reward,
    }


def generate_report(
    policy_results: List[Dict[str, Any]],
    rl_dataset_path: Path,
    cf_dataset_path: Path,
    entry_model_path: Path,
    timing_model_path: Path,
    report_path: Path,
) -> None:
    """Generate markdown evaluation report."""
    lines = []
    
    lines.append("# ENTRY POLICY EVAL V1 – FULLYEAR 2025")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **RL Dataset:** {rl_dataset_path}")
    lines.append(f"- **Counterfactual V2 Dataset:** {cf_dataset_path}")
    lines.append(f"- **Entry Critic Model:** {entry_model_path}")
    lines.append(f"- **Timing Critic Model:** {timing_model_path}")
    lines.append("")
    
    # Policy results table
    lines.append("## Policy Results")
    lines.append("")
    lines.append(
        "| Policy | n_trades | real_frac | shadow_frac | win_rate | "
        "avg_reward | reward_per_1k | p05_reward | p95_reward |"
    )
    lines.append("|--------|----------|-----------|-------------|----------|------------|---------------|------------|------------|")
    
    for result in policy_results:
        lines.append(
            f"| {result['policy_name']} | {result['n_trades']:,} | "
            f"{result['real_frac']:.1%} | {result['shadow_frac']:.1%} | "
            f"{result['win_rate']:.2%} | {result['avg_reward']:.2f} | "
            f"{result['reward_per_1k']:.2f} | {result['p05_reward']:.2f} | "
            f"{result['p95_reward']:.2f} |"
        )
    lines.append("")
    
    # Observations
    lines.append("## Observations")
    lines.append("")
    
    # Find best policies by different metrics
    if len(policy_results) > 0:
        best_ev = max(policy_results, key=lambda x: x["reward_per_1k"])
        best_winrate = max(policy_results, key=lambda x: x["win_rate"])
        best_p05 = max(policy_results, key=lambda x: x["p05_reward"])
        
        lines.append("### Best Policies by Metric")
        lines.append("")
        lines.append(f"- **Highest EV (reward_per_1k):** {best_ev['policy_name']}")
        lines.append(f"  - reward_per_1k: {best_ev['reward_per_1k']:.2f}")
        lines.append(f"  - win_rate: {best_ev['win_rate']:.2%}")
        lines.append(f"  - n_trades: {best_ev['n_trades']:,}")
        lines.append("")
        lines.append(f"- **Highest Win Rate:** {best_winrate['policy_name']}")
        lines.append(f"  - win_rate: {best_winrate['win_rate']:.2%}")
        lines.append(f"  - reward_per_1k: {best_winrate['reward_per_1k']:.2f}")
        lines.append("")
        lines.append(f"- **Best Risk Profile (p05):** {best_p05['policy_name']}")
        lines.append(f"  - p05_reward: {best_p05['p05_reward']:.2f}")
        lines.append(f"  - reward_per_1k: {best_p05['reward_per_1k']:.2f}")
        lines.append("")
        
        # Shadow contribution
        policies_with_shadow = [r for r in policy_results if r["shadow_frac"] > 0]
        if policies_with_shadow:
            lines.append("### Shadow Contribution")
            lines.append("")
            for result in policies_with_shadow:
                shadow_pnl = result["sum_reward"] * result["shadow_frac"]
                lines.append(
                    f"- **{result['policy_name']}:** "
                    f"{result['shadow_frac']:.1%} shadow trades, "
                    f"contributed {shadow_pnl:.2f} bps to total PnL"
                )
            lines.append("")
    
    # Notes
    lines.append("---")
    lines.append("")
    lines.append("## Viktig Notat")
    lines.append("")
    lines.append("- **Ingen av disse policyene er aktivert i live.**")
    lines.append("- De er kandidater til CANARY/EXPERIMENT-modus senere.")
    lines.append("- Baseline SNIPER runtime er uendret.")
    lines.append("- Dette er ren offline-analyse og simulering.")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/eval_entry_policies_v1.py`*")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"Report saved: {report_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Entry Policy Evaluation V1"
    )
    parser.add_argument(
        "--rl_dataset",
        type=Path,
        default=Path("data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_PARALLEL.parquet"),
        help="Path to RL dataset Parquet file",
    )
    parser.add_argument(
        "--cf_dataset_v2",
        type=Path,
        default=Path("data/rl/shadow_counterfactual_FULLYEAR_2025_V2.parquet"),
        help="Path to shadow counterfactual V2 dataset",
    )
    parser.add_argument(
        "--entry_model",
        type=Path,
        default=Path("gx1/models/entry_critic_v1.joblib"),
        help="Path to Entry Critic V1 model",
    )
    parser.add_argument(
        "--entry_meta",
        type=Path,
        default=Path("gx1/models/entry_critic_v1_meta.json"),
        help="Path to Entry Critic V1 metadata",
    )
    parser.add_argument(
        "--timing_model",
        type=Path,
        default=Path("gx1/models/entry_timing_critic_v1.joblib"),
        help="Path to Entry Timing Critic V1 model",
    )
    parser.add_argument(
        "--timing_meta",
        type=Path,
        default=Path("gx1/models/entry_timing_critic_v1_meta.json"),
        help="Path to Entry Timing Critic V1 metadata",
    )
    parser.add_argument(
        "--report_out",
        type=Path,
        default=Path("reports/rl/ENTRY_POLICY_EVAL_V1_FULLYEAR_2025.md"),
        help="Output path for evaluation report",
    )
    
    args = parser.parse_args()
    
    try:
        # Load RL dataset
        log.info(f"Loading RL dataset: {args.rl_dataset}")
        rl_df = pd.read_parquet(args.rl_dataset)
        log.info(f"Loaded {len(rl_df)} rows from RL dataset")
        
        # Load counterfactual V2 dataset
        log.info(f"Loading counterfactual V2 dataset: {args.cf_dataset_v2}")
        cf_df = pd.read_parquet(args.cf_dataset_v2)
        log.info(f"Loaded {len(cf_df)} rows from CF V2 dataset")
        
        # Load models
        entry_model, entry_meta, entry_feature_order = load_entry_critic(args.entry_model, args.entry_meta)
        timing_model, timing_meta, timing_feature_order = load_timing_critic(
            args.timing_model, args.timing_meta
        )
        
        # Build evaluation dataset
        log.info("Building policy evaluation dataset...")
        
        # Start with RL dataset (all candidates)
        eval_df = rl_df.copy()
        
        # Add trade_type
        eval_df["trade_type"] = eval_df["action_taken"].map({1: "REAL", 0: "SHADOW_SIM"})
        
        # Calculate Entry Critic scores
        log.info("Calculating Entry Critic scores...")
        entry_scores = []
        for idx, (_, row) in enumerate(eval_df.iterrows()):
            if idx % 1000 == 0:
                log.info(f"  Processed {idx}/{len(eval_df)} entries...")
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
                else:
                    entry_scores.append(None)
            except Exception as e:
                log.debug(f"Failed to calculate Entry Critic score for row {idx}: {e}")
                entry_scores.append(None)
        
        eval_df["entry_critic_score_v1"] = entry_scores
        
        # Calculate Timing Critic predictions
        log.info("Calculating Timing Critic predictions...")
        timing_features = prepare_timing_features(
            eval_df, timing_feature_order, timing_meta
        )
        timing_pred = timing_model.predict(timing_features)
        eval_df["timing_pred"] = timing_pred
        
        # Merge with CF V2 for shadow rewards
        log.info("Merging with counterfactual V2 for shadow rewards...")
        
        # Normalize timestamps
        if "candle_time" in eval_df.columns:
            eval_df["ts"] = pd.to_datetime(eval_df["candle_time"])
        elif "ts" not in eval_df.columns:
            raise ValueError("Cannot find timestamp column in RL dataset")
        
        if eval_df["ts"].dt.tz is None:
            eval_df["ts"] = eval_df["ts"].dt.tz_localize("UTC")
        
        cf_df["ts"] = pd.to_datetime(cf_df["ts"])
        if cf_df["ts"].dt.tz is None:
            cf_df["ts"] = cf_df["ts"].dt.tz_localize("UTC")
        
        # Merge CF rewards for shadow trades
        # Handle duplicate timestamps by taking first match
        cf_df_unique = cf_df.drop_duplicates(subset=["ts"], keep="first")
        cf_rewards_dict = {}
        for _, cf_row in cf_df_unique.iterrows():
            ts = cf_row["ts"]
            if ts not in cf_rewards_dict:
                cf_rewards_dict[ts] = {
                    "pnl_bps": cf_row.get("pnl_bps", 0.0),
                    "timing_quality": cf_row.get("timing_quality"),
                }
        
        # Add reward column (use CF for shadow, RL for real)
        rewards = []
        for _, row in eval_df.iterrows():
            if row["trade_type"] == "SHADOW_SIM" and row["ts"] in cf_rewards_dict:
                rewards.append(cf_rewards_dict[row["ts"]]["pnl_bps"])
            elif row["trade_type"] == "REAL" and "reward" in row.index:
                rewards.append(row["reward"])
            elif row["trade_type"] == "REAL" and "pnl_bps" in row.index:
                rewards.append(row["pnl_bps"])
            else:
                rewards.append(0.0)
        
        eval_df["reward"] = rewards
        
        # Add timing_quality from CF for shadow trades
        timing_qualities = []
        for _, row in eval_df.iterrows():
            if row["trade_type"] == "SHADOW_SIM" and row["ts"] in cf_rewards_dict:
                timing_qualities.append(cf_rewards_dict[row["ts"]]["timing_quality"])
            else:
                timing_qualities.append(None)
        
        eval_df["timing_quality"] = timing_qualities
        
        log.info(f"Evaluation dataset: {len(eval_df)} rows")
        log.info(f"  - REAL: {(eval_df['trade_type'] == 'REAL').sum():,}")
        log.info(f"  - SHADOW_SIM: {(eval_df['trade_type'] == 'SHADOW_SIM').sum():,}")
        
        # Define policies
        log.info("Evaluating policies...")
        
        policies = []
        
        # Policy A: Baseline (only real trades)
        def policy_a(row):
            return row["trade_type"] == "REAL"
        policies.append(("A (Baseline)", policy_a))
        
        # Policy B: Strict p_long
        def policy_b(row):
            return row.get("p_long", 0.0) >= 0.80
        policies.append(("B (p_long >= 0.80)", policy_b))
        
        # Policy C: EntryCritic gate
        def policy_c(row):
            return (
                row.get("p_long", 0.0) >= 0.67
                and row.get("entry_critic_score_v1") is not None
                and row["entry_critic_score_v1"] >= 0.60
            )
        policies.append(("C (EntryCritic >= 0.60)", policy_c))
        
        # Policy D: TimingCritic gate
        def policy_d(row):
            return (
                row.get("p_long", 0.0) >= 0.67
                and row.get("timing_pred") is not None
                and row["timing_pred"] != "AVOID_TRADE"
            )
        policies.append(("D (TimingCritic != AVOID_TRADE)", policy_d))
        
        # Policy E: Combined (strict)
        def policy_e(row):
            return (
                row.get("p_long", 0.0) >= 0.67
                and row.get("entry_critic_score_v1") is not None
                and row["entry_critic_score_v1"] >= 0.60
                and row.get("timing_pred") is not None
                and row["timing_pred"] == "IMMEDIATE_OK"
            )
        policies.append(("E (Combined strict)", policy_e))
        
        # Policy F: Aggressive exploitation
        def policy_f(row):
            return (
                row.get("p_long", 0.0) >= 0.67
                and row.get("timing_pred") is not None
                and row["timing_pred"] in ["IMMEDIATE_OK", "DELAY_BETTER"]
            )
        policies.append(("F (Aggressive)", policy_f))
        
        # Evaluate all policies
        policy_results = []
        for policy_name, policy_func in policies:
            result = evaluate_policy(eval_df, policy_name, policy_func)
            policy_results.append(result)
            log.info(
                f"{policy_name}: {result['n_trades']:,} trades, "
                f"win_rate={result['win_rate']:.2%}, "
                f"reward_per_1k={result['reward_per_1k']:.2f}"
            )
        
        # Generate report
        generate_report(
            policy_results,
            args.rl_dataset,
            args.cf_dataset_v2,
            args.entry_model,
            args.timing_model,
            args.report_out,
        )
        
        log.info("✅ Entry policy evaluation complete!")
        return 0
    
    except Exception as e:
        log.error(f"❌ Policy evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

