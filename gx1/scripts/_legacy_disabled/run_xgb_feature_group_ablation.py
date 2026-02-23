#!/usr/bin/env python3
"""
XGB Feature Group Ablation Harness.

Tests different feature group configurations to find which groups
reduce drift without collapsing signal quality.

Usage:
    python3 gx1/scripts/run_xgb_feature_group_ablation.py
    python3 gx1/scripts/run_xgb_feature_group_ablation.py --groups "RSI,MACD,EMA"
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


# Feature group definitions (regex patterns)
FEATURE_GROUPS = {
    "RSI": r".*rsi.*",
    "MACD": r".*macd.*",
    "EMA": r".*ema.*",
    "ATR": r".*atr.*",
    "REGIME": r".*(regime|session).*",
    "RETURNS": r".*(ret|roc|r1|r3|r5|r8|r12|r24|r48).*",
    "VOL": r".*(vol|std|rvol).*",
    "VWAP": r".*vwap.*",
    "RANGE": r".*range.*",
    "BB": r".*(bb|bandwidth|squeeze).*",
    "SLOPE": r".*slope.*",
    "H1H4": r".*_v1h[14]_.*",
    "INTERACTION": r".*_int_.*",
}


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def resolve_prebuilt_for_year(year: int, gx1_data: Path) -> Optional[Path]:
    """Resolve prebuilt parquet path for a given year."""
    candidates = [
        gx1_data / "data" / "data" / "prebuilt" / "TRIAL160" / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_feature_contract() -> List[str]:
    """Load feature contract."""
    contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    with open(contract_path, "r") as f:
        contract = json.load(f)
    return contract.get("features", [])


def get_group_features(group_name: str, all_features: List[str]) -> List[str]:
    """Get features matching a group pattern."""
    pattern = FEATURE_GROUPS.get(group_name, group_name)
    regex = re.compile(pattern, re.IGNORECASE)
    return [f for f in all_features if regex.match(f)]


def compute_ks_statistic(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Compute KS statistic between two distributions."""
    stat, _ = stats.ks_2samp(dist_a, dist_b)
    return float(stat)


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index."""
    # Create bins from expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    expected_pct = expected_hist / (len(expected) + eps)
    actual_pct = actual_hist / (len(actual) + eps)
    
    # Avoid log(0)
    expected_pct = np.clip(expected_pct, eps, 1)
    actual_pct = np.clip(actual_pct, eps, 1)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def run_xgb_inference(X: np.ndarray, gx1_data: Path, session: str = "EU") -> np.ndarray:
    """Run XGB inference and return p_long."""
    try:
        from joblib import load as joblib_load
    except ImportError:
        import joblib
        joblib_load = joblib.load
    
    model_path = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION" / f"xgb_{session}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"XGB model not found: {model_path}")
    
    model = joblib_load(model_path)
    
    # Check feature count
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
        if X.shape[1] != expected:
            X = X[:, :expected]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] >= 2:
            return proba[:, 1]
        return proba[:, 0]
    return model.predict(X)


def main():
    parser = argparse.ArgumentParser(
        description="XGB Feature Group Ablation Harness"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to test"
    )
    parser.add_argument(
        "--n-bars-per-year",
        type=int,
        default=20000,
        help="Number of bars per year"
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated list of groups to ablate (default: all)"
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2025,
        help="Reference year for drift calculation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling"
    )
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Load feature contract
    all_features = load_feature_contract()
    print(f"Total features: {len(all_features)}")
    
    # Determine groups to test
    if args.groups:
        groups_to_test = [g.strip() for g in args.groups.split(",")]
    else:
        groups_to_test = list(FEATURE_GROUPS.keys())
    
    print(f"\nGroups to ablate: {groups_to_test}")
    
    # Show group sizes
    print("\nGroup sizes:")
    for group in groups_to_test:
        group_features = get_group_features(group, all_features)
        print(f"  {group}: {len(group_features)} features")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "xgb_ablation" / f"ABLATION_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")
    
    # Load data for all years
    print(f"\nLoading data...")
    year_data: Dict[int, pd.DataFrame] = {}
    for year in args.years:
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  WARNING: No prebuilt for {year}")
            continue
        
        df = pd.read_parquet(prebuilt_path)
        if len(df) > args.n_bars_per_year:
            step = len(df) // args.n_bars_per_year
            df = df.iloc[::step][:args.n_bars_per_year]
        
        year_data[year] = df
        print(f"  {year}: {len(df)} bars")
    
    # Run baseline (all features)
    print(f"\nRunning baseline (ARM_A: all features)...")
    baseline_results: Dict[int, np.ndarray] = {}
    for year, df in year_data.items():
        # Extract features (use all available)
        available_features = [f for f in all_features if f in df.columns]
        X = df[available_features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            p_long = run_xgb_inference(X, gx1_data)
            baseline_results[year] = p_long
            print(f"  {year}: p_long mean={p_long.mean():.4f}")
        except Exception as e:
            print(f"  {year}: ERROR - {e}")
    
    # Run ablations
    ablation_results: Dict[str, Dict[int, np.ndarray]] = {}
    
    for group in groups_to_test:
        print(f"\nRunning ablation ARM_B: minus {group}...")
        group_features = set(get_group_features(group, all_features))
        remaining_features = [f for f in all_features if f not in group_features]
        print(f"  Removed {len(group_features)} features, {len(remaining_features)} remaining")
        
        ablation_results[group] = {}
        for year, df in year_data.items():
            available_features = [f for f in remaining_features if f in df.columns]
            X = df[available_features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                p_long = run_xgb_inference(X, gx1_data)
                ablation_results[group][year] = p_long
                print(f"  {year}: p_long mean={p_long.mean():.4f}")
            except Exception as e:
                print(f"  {year}: ERROR - {e}")
    
    # Compute metrics
    print(f"\nComputing drift metrics...")
    ref_year = args.reference_year
    if ref_year not in baseline_results:
        ref_year = max(baseline_results.keys())
    ref_baseline = baseline_results[ref_year]
    
    metrics: List[Dict[str, Any]] = []
    
    # Baseline drift
    for year in sorted(baseline_results.keys()):
        if year == ref_year:
            continue
        ks = compute_ks_statistic(ref_baseline, baseline_results[year])
        psi = compute_psi(ref_baseline, baseline_results[year])
        metrics.append({
            "arm": "BASELINE",
            "group": "none",
            "year": year,
            "ks_vs_ref": ks,
            "psi_vs_ref": psi,
            "mean_p_long": float(baseline_results[year].mean()),
            "std_p_long": float(baseline_results[year].std()),
        })
    
    # Ablation drift
    for group, group_results in ablation_results.items():
        for year in sorted(group_results.keys()):
            if year == ref_year:
                continue
            if ref_year not in group_results:
                continue
            ref_ablation = group_results[ref_year]
            ks = compute_ks_statistic(ref_ablation, group_results[year])
            psi = compute_psi(ref_ablation, group_results[year])
            metrics.append({
                "arm": f"MINUS_{group}",
                "group": group,
                "year": year,
                "ks_vs_ref": ks,
                "psi_vs_ref": psi,
                "mean_p_long": float(group_results[year].mean()),
                "std_p_long": float(group_results[year].std()),
            })
    
    # Write metrics CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_dir / "KS_PSI_BY_ARM.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path}")
    
    # Generate summary
    summary_path = output_dir / "ABLATION_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# XGB Feature Group Ablation Summary\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Reference year: {ref_year}\n")
        f.write(f"- Years tested: {args.years}\n")
        f.write(f"- Bars per year: {args.n_bars_per_year}\n")
        f.write(f"- Groups tested: {groups_to_test}\n")
        f.write(f"- Seed: {args.seed}\n\n")
        
        f.write("## Baseline Drift\n\n")
        f.write("| Year | KS vs Ref | PSI vs Ref | Mean p_long |\n")
        f.write("|------|-----------|------------|-------------|\n")
        for m in metrics:
            if m["arm"] == "BASELINE":
                f.write(f"| {m['year']} | {m['ks_vs_ref']:.4f} | {m['psi_vs_ref']:.4f} | {m['mean_p_long']:.4f} |\n")
        
        f.write("\n## Ablation Results\n\n")
        
        # Compare arms
        f.write("### Drift Reduction by Removing Group\n\n")
        f.write("Lower KS/PSI = less drift = potentially better stability.\n\n")
        f.write("| Group Removed | Avg KS | Avg PSI | Verdict |\n")
        f.write("|---------------|--------|---------|--------|\n")
        
        baseline_ks = metrics_df[metrics_df["arm"] == "BASELINE"]["ks_vs_ref"].mean()
        baseline_psi = metrics_df[metrics_df["arm"] == "BASELINE"]["psi_vs_ref"].mean()
        
        verdicts = []
        for group in groups_to_test:
            arm_name = f"MINUS_{group}"
            arm_metrics = metrics_df[metrics_df["arm"] == arm_name]
            if len(arm_metrics) > 0:
                avg_ks = arm_metrics["ks_vs_ref"].mean()
                avg_psi = arm_metrics["psi_vs_ref"].mean()
                
                if avg_ks < baseline_ks * 0.9 and avg_psi < baseline_psi * 0.9:
                    verdict = "✅ REDUCES DRIFT"
                    verdicts.append((group, avg_ks, avg_psi, True))
                elif avg_ks > baseline_ks * 1.1 or avg_psi > baseline_psi * 1.1:
                    verdict = "❌ INCREASES DRIFT"
                    verdicts.append((group, avg_ks, avg_psi, False))
                else:
                    verdict = "➖ NEUTRAL"
                    verdicts.append((group, avg_ks, avg_psi, None))
                
                f.write(f"| {group} | {avg_ks:.4f} | {avg_psi:.4f} | {verdict} |\n")
        
        f.write(f"\n*Baseline: KS={baseline_ks:.4f}, PSI={baseline_psi:.4f}*\n")
        
        f.write("\n## GO/NO-GO Hints\n\n")
        helpful = [v for v in verdicts if v[3] is True]
        harmful = [v for v in verdicts if v[3] is False]
        
        if helpful:
            f.write("### Groups that REDUCE drift (consider removing):\n\n")
            for group, ks, psi, _ in sorted(helpful, key=lambda x: x[1]):
                f.write(f"- **{group}**: KS={ks:.4f}, PSI={psi:.4f}\n")
        else:
            f.write("No groups clearly reduce drift.\n")
        
        f.write("\n")
        
        if harmful:
            f.write("### Groups that INCREASE drift (keep these):\n\n")
            for group, ks, psi, _ in sorted(harmful, key=lambda x: -x[1]):
                f.write(f"- **{group}**: KS={ks:.4f}, PSI={psi:.4f}\n")
    
    print(f"Wrote: {summary_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
