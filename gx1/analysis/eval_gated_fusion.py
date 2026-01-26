#!/usr/bin/env python3
"""
Evaluate Gated Fusion

Phase 2: Evaluate gated fusion performance (tail risk, gate responsiveness, regime stability).

Usage:
    python gx1/analysis/eval_gated_fusion.py \
        --replay_dir runs/replay_shadow/... \
        --output_dir reports/fusion
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def compute_tail_risk_metrics(pnl_bps: np.ndarray) -> Dict[str, float]:
    """
    Compute tail risk metrics.
    
    Returns:
        Dict with max_drawdown, var_95, var_99, etc.
    """
    if len(pnl_bps) == 0:
        return {
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "max_loss": 0.0,
        }
    
    # Cumulative PnL
    cum_pnl = np.cumsum(pnl_bps)
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = float(np.min(drawdown))
    
    # VaR (Value at Risk)
    var_95 = float(np.percentile(pnl_bps, 5))  # 5th percentile (loss)
    var_99 = float(np.percentile(pnl_bps, 1))  # 1st percentile (loss)
    
    # Max loss
    max_loss = float(np.min(pnl_bps))
    
    return {
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "var_99": var_99,
        "max_loss": max_loss,
    }


def compute_regime_stability_score(
    pnl_by_regime: Dict[str, np.ndarray],
) -> float:
    """
    Compute regime stability score (0-1, higher is better).
    
    Returns:
        Stability score based on performance consistency across regimes
    """
    if not pnl_by_regime:
        return 0.0
    
    # Compute mean PnL per regime
    regime_means = {k: float(np.mean(v)) for k, v in pnl_by_regime.items() if len(v) > 0}
    
    if not regime_means:
        return 0.0
    
    # Coefficient of variation (lower is better)
    mean_values = list(regime_means.values())
    if np.std(mean_values) == 0:
        return 1.0  # Perfect stability
    
    cv = np.std(mean_values) / (abs(np.mean(mean_values)) + 1e-6)
    stability_score = 1.0 / (1.0 + cv)  # Normalize to [0, 1]
    
    return float(stability_score)


def compute_gate_responsiveness(
    gate_values: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute gate responsiveness metrics.
    
    Returns:
        Dict with correlation, gate_variance, etc.
    """
    if len(gate_values) == 0 or len(uncertainty_scores) == 0:
        return {
            "correlation": 0.0,
            "gate_variance": 0.0,
            "gate_mean": 0.0,
            "gate_p95": 0.0,
            "gate_std": 0.0,
        }
    
    # Correlation: high uncertainty → low gate (trust XGB less)
    # Expected: negative correlation between uncertainty_score and gate
    correlation = float(np.corrcoef(uncertainty_scores, 1.0 - gate_values)[0, 1])
    
    # Gate statistics
    gate_mean = float(np.mean(gate_values))
    gate_std = float(np.std(gate_values))
    gate_p95 = float(np.percentile(gate_values, 95))
    gate_variance = float(np.var(gate_values))
    
    return {
        "correlation": correlation,
        "gate_variance": gate_variance,
        "gate_mean": gate_mean,
        "gate_p95": gate_p95,
        "gate_std": gate_std,
    }


def evaluate_gated_fusion(
    replay_dir: Path,
) -> Dict[str, any]:
    """
    Evaluate gated fusion from replay results.
    
    Args:
        replay_dir: Directory containing replay results (trade log, telemetry, etc.)
    
    Returns:
        Dict with evaluation metrics
    """
    log.info(f"[FUSION_EVAL] Loading replay results from {replay_dir}")
    
    # Load trade log
    trade_log_path = replay_dir / "trade_log.csv"
    if not trade_log_path.exists():
        log.warning(f"[FUSION_EVAL] Trade log not found: {trade_log_path}")
        return {}
    
    trades_df = pd.read_csv(trade_log_path)
    if len(trades_df) == 0:
        log.warning("[FUSION_EVAL] No trades in trade log")
        return {}
    
    # Load telemetry (if available)
    telemetry_path = replay_dir / "telemetry.json"
    telemetry = {}
    if telemetry_path.exists():
        with open(telemetry_path, "r") as f:
            telemetry = json.load(f)
    
    # Extract gate values from telemetry
    gate_values = []
    uncertainty_scores = []
    gate_vs_uncertainty = []
    
    if "entry_telemetry" in telemetry:
        entry_telemetry = telemetry["entry_telemetry"]
        
        if "gate_values" in entry_telemetry:
            gate_values = entry_telemetry["gate_values"]
        
        if "gate_vs_uncertainty" in entry_telemetry:
            gate_vs_uncertainty = entry_telemetry["gate_vs_uncertainty"]
            gate_values = [g["gate"] for g in gate_vs_uncertainty]
            uncertainty_scores = [g["uncertainty_score"] for g in gate_vs_uncertainty]
    
    # Compute PnL metrics
    if "pnl_bps" in trades_df.columns:
        pnl_bps = trades_df["pnl_bps"].values
        tail_risk = compute_tail_risk_metrics(pnl_bps)
    else:
        tail_risk = {}
    
    # Compute performance by uncertainty buckets
    performance_by_uncertainty = {}
    if "uncertainty_score" in trades_df.columns and "pnl_bps" in trades_df.columns:
        uncertainty_buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(uncertainty_buckets) - 1):
            bucket_low = uncertainty_buckets[i]
            bucket_high = uncertainty_buckets[i + 1]
            bucket_name = f"{bucket_low:.1f}-{bucket_high:.1f}"
            
            mask = (trades_df["uncertainty_score"] >= bucket_low) & (trades_df["uncertainty_score"] < bucket_high)
            bucket_trades = trades_df[mask]
            
            if len(bucket_trades) > 0:
                performance_by_uncertainty[bucket_name] = {
                    "n_trades": int(len(bucket_trades)),
                    "mean_pnl": float(bucket_trades["pnl_bps"].mean()),
                    "win_rate": float((bucket_trades["pnl_bps"] > 0).mean()),
                    "max_loss": float(bucket_trades["pnl_bps"].min()),
                }
    
    # Compute performance by gate buckets
    performance_by_gate = {}
    if "gate" in trades_df.columns and "pnl_bps" in trades_df.columns:
        gate_buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(gate_buckets) - 1):
            bucket_low = gate_buckets[i]
            bucket_high = gate_buckets[i + 1]
            bucket_name = f"{bucket_low:.1f}-{bucket_high:.1f}"
            
            mask = (trades_df["gate"] >= bucket_low) & (trades_df["gate"] < bucket_high)
            bucket_trades = trades_df[mask]
            
            if len(bucket_trades) > 0:
                performance_by_gate[bucket_name] = {
                    "n_trades": int(len(bucket_trades)),
                    "mean_pnl": float(bucket_trades["pnl_bps"].mean()),
                    "win_rate": float((bucket_trades["pnl_bps"] > 0).mean()),
                    "max_loss": float(bucket_trades["pnl_bps"].min()),
                }
    
    # Compute regime stability
    pnl_by_regime = {}
    if "regime" in trades_df.columns and "pnl_bps" in trades_df.columns:
        for regime in trades_df["regime"].unique():
            regime_trades = trades_df[trades_df["regime"] == regime]
            if len(regime_trades) > 0:
                pnl_by_regime[regime] = regime_trades["pnl_bps"].values
    
    regime_stability_score = compute_regime_stability_score(pnl_by_regime)
    
    # Compute gate responsiveness
    gate_metrics = {}
    if len(gate_values) > 0 and len(uncertainty_scores) > 0:
        gate_metrics = compute_gate_responsiveness(
            np.array(gate_values),
            np.array(uncertainty_scores),
        )
    
    # Gate histogram per regime
    gate_histogram_per_regime = {}
    if "gate_per_regime" in entry_telemetry:
        for regime_key, gates in entry_telemetry["gate_per_regime"].items():
            if len(gates) > 0:
                gate_histogram_per_regime[regime_key] = {
                    "mean": float(np.mean(gates)),
                    "std": float(np.std(gates)),
                    "p95": float(np.percentile(gates, 95)),
                    "min": float(np.min(gates)),
                    "max": float(np.max(gates)),
                    "n_samples": int(len(gates)),
                }
    
    return {
        "tail_risk": tail_risk,
        "performance_by_uncertainty": performance_by_uncertainty,
        "performance_by_gate": performance_by_gate,
        "regime_stability_score": regime_stability_score,
        "gate_metrics": gate_metrics,
        "gate_histogram_per_regime": gate_histogram_per_regime,
        "n_trades": int(len(trades_df)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gated fusion")
    parser.add_argument("--replay_dir", type=str, required=True, help="Replay directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reports")
    parser.add_argument("--baseline_replay_dir", type=str, help="Baseline replay directory (for comparison)")
    
    args = parser.parse_args()
    
    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        raise FileNotFoundError(f"Replay directory not found: {replay_dir}")
    
    # Evaluate gated fusion
    results = evaluate_gated_fusion(replay_dir)
    
    # Evaluate baseline (if provided)
    baseline_results = {}
    if args.baseline_replay_dir:
        baseline_replay_dir = Path(args.baseline_replay_dir)
        if baseline_replay_dir.exists():
            baseline_results = evaluate_gated_fusion(baseline_replay_dir)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"GATED_FUSION_REPORT_{timestamp}.md"
    
    # Generate markdown report
    with open(report_path, "w") as f:
        f.write(f"# Gated Fusion Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Replay Directory:** {args.replay_dir}\n\n")
        
        # Tail Risk
        f.write("## Tail Risk Metrics\n\n")
        if "tail_risk" in results:
            tail_risk = results["tail_risk"]
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Max Drawdown | {tail_risk.get('max_drawdown', 0.0):.2f} bps |\n")
            f.write(f"| VaR (95th) | {tail_risk.get('var_95', 0.0):.2f} bps |\n")
            f.write(f"| VaR (99th) | {tail_risk.get('var_99', 0.0):.2f} bps |\n")
            f.write(f"| Max Loss | {tail_risk.get('max_loss', 0.0):.2f} bps |\n")
        
        # Gate Responsiveness
        f.write("\n## Gate Responsiveness\n\n")
        if "gate_metrics" in results:
            gate_metrics = results["gate_metrics"]
            f.write("| Metric | Value | Target |\n")
            f.write("|--------|-------|--------|\n")
            f.write(f"| Correlation (uncertainty vs 1-gate) | {gate_metrics.get('correlation', 0.0):.4f} | > 0.3 |\n")
            f.write(f"| Gate Variance | {gate_metrics.get('gate_variance', 0.0):.4f} | > 0.05 |\n")
            f.write(f"| Gate Mean | {gate_metrics.get('gate_mean', 0.0):.4f} | - |\n")
            f.write(f"| Gate P95 | {gate_metrics.get('gate_p95', 0.0):.4f} | - |\n")
            f.write(f"| Gate Std | {gate_metrics.get('gate_std', 0.0):.4f} | - |\n")
        
        # Regime Stability
        f.write("\n## Regime Stability\n\n")
        f.write(f"**Stability Score:** {results.get('regime_stability_score', 0.0):.4f} (target: > 0.8)\n\n")
        
        if "gate_histogram_per_regime" in results:
            f.write("### Gate Distribution per Regime\n\n")
            f.write("| Regime | Mean | Std | P95 | N Samples |\n")
            f.write("|--------|------|-----|-----|-----------|\n")
            for regime, stats in results["gate_histogram_per_regime"].items():
                f.write(f"| {regime} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                       f"{stats['p95']:.4f} | {stats['n_samples']} |\n")
        
        # Performance by Uncertainty
        f.write("\n## Performance by Uncertainty Buckets\n\n")
        if "performance_by_uncertainty" in results:
            f.write("| Uncertainty Bucket | N Trades | Mean PnL | Win Rate | Max Loss |\n")
            f.write("|-------------------|----------|----------|----------|----------|\n")
            for bucket, perf in results["performance_by_uncertainty"].items():
                f.write(f"| {bucket} | {perf['n_trades']} | {perf['mean_pnl']:.2f} bps | "
                       f"{perf['win_rate']:.2%} | {perf['max_loss']:.2f} bps |\n")
        
        # Performance by Gate
        f.write("\n## Performance by Gate Buckets\n\n")
        if "performance_by_gate" in results:
            f.write("| Gate Bucket | N Trades | Mean PnL | Win Rate | Max Loss |\n")
            f.write("|-------------|----------|----------|----------|----------|\n")
            for bucket, perf in results["performance_by_gate"].items():
                f.write(f"| {bucket} | {perf['n_trades']} | {perf['mean_pnl']:.2f} bps | "
                       f"{perf['win_rate']:.2%} | {perf['max_loss']:.2f} bps |\n")
        
        # GO/NO-GO
        f.write("\n## GO/NO-GO Criteria\n\n")
        
        all_pass = True
        checks = []
        
        # Tail risk check
        if "tail_risk" in results:
            tail_risk = results["tail_risk"]
            max_dd = tail_risk.get("max_drawdown", 0.0)
            checks.append(("Max Drawdown < -200 bps", max_dd > -200, max_dd))
        
        # Gate responsiveness
        if "gate_metrics" in results:
            gate_metrics = results["gate_metrics"]
            corr = gate_metrics.get("correlation", 0.0)
            variance = gate_metrics.get("gate_variance", 0.0)
            checks.append(("Gate responds to uncertainty (corr > 0.3)", corr > 0.3, corr))
            checks.append(("Gate is not constant (variance > 0.05)", variance > 0.05, variance))
        
        # Regime stability
        stability = results.get("regime_stability_score", 0.0)
        checks.append(("Regime stability score > 0.8", stability > 0.8, stability))
        
        for check_name, passed, value in checks:
            status = "✅" if passed else "❌"
            f.write(f"- {status} {check_name}: {value:.4f}\n")
            if not passed:
                all_pass = False
        
        f.write(f"\n## Overall Status\n\n")
        f.write(f"{'✅ GO' if all_pass else '❌ NO-GO'}\n")
    
    # Save JSON results
    json_path = output_dir / f"GATED_FUSION_RESULTS_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"[FUSION_EVAL] ✅ Evaluation complete. Report: {report_path}")


if __name__ == "__main__":
    main()
