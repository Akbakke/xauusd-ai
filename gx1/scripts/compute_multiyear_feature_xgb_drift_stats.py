#!/usr/bin/env python3
"""
Multiyear Feature & XGB Drift Statistics

Computes feature statistics and drift metrics across years 2020-2025,
with focus on detecting OOD patterns and prebuilt mismatches.

Usage:
    python gx1/scripts/compute_multiyear_feature_xgb_drift_stats.py \
        --years 2020 2021 2022 2023 2024 2025 \
        --reference-year 2025 \
        --n-bars 20000
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import pandas as pd
    from scipy import stats
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install pandas scipy")
    sys.exit(1)

# Default paths
DEFAULT_GX1_DATA = Path("/Users/andrekildalbakke/Desktop/GX1_DATA")
DEFAULT_PREBUILT_ROOT = DEFAULT_GX1_DATA / "data" / "prebuilt" / "TRIAL160"
DEFAULT_BUNDLE = DEFAULT_GX1_DATA / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"

# XGB feature names (from feature contract)
XGB_OUTPUT_NAMES = ["p_long_xgb", "p_hat_xgb", "uncertainty_score"]

# Session definitions (UTC hours)
SESSION_DEFINITIONS = {
    "ASIA": (0, 7),
    "EU": (7, 13),
    "US": (13, 21),
    "OVERLAP": (12, 16),  # EU/US overlap
}


def get_prebuilt_path(year: int, prebuilt_root: Path) -> Path:
    """Get prebuilt path for a year."""
    return prebuilt_root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"


def compute_schema_fingerprint(columns: List[str]) -> str:
    """Compute fingerprint of column schema."""
    schema_str = ",".join(sorted(columns))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def get_session(hour: int) -> str:
    """Get session name from hour (UTC)."""
    if 12 <= hour < 16:
        return "OVERLAP"
    elif 0 <= hour < 7:
        return "ASIA"
    elif 7 <= hour < 13:
        return "EU"
    elif 13 <= hour < 21:
        return "US"
    else:
        return "ASIA"  # Late night maps to ASIA


def load_prebuilt_sample(
    year: int,
    prebuilt_root: Path,
    n_bars: int,
    warmup: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Load a deterministic subsample from prebuilt parquet.
    
    Returns: (df, schema_fingerprint, error_message)
    """
    path = get_prebuilt_path(year, prebuilt_root)
    
    if not path.exists():
        return None, None, f"Prebuilt file not found: {path}"
    
    try:
        # Load full dataframe
        df = pd.read_parquet(path)
        
        # Validate schema
        columns = list(df.columns)
        fingerprint = compute_schema_fingerprint(columns)
        
        # HARD GATE: Check for CLOSE column
        if "CLOSE" in columns:
            return None, fingerprint, f"HARD_FAIL: CLOSE column found in prebuilt for year {year}"
        
        # Handle index
        if "ts" in df.columns:
            df = df.set_index("ts")
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, fingerprint, f"Index is not DatetimeIndex for year {year}"
        
        # HARD GATE: Check UTC
        if df.index.tz is None:
            return None, fingerprint, f"HARD_FAIL: Timestamp not timezone-aware for year {year}"
        if str(df.index.tz) not in ["UTC", "utc"]:
            return None, fingerprint, f"HARD_FAIL: Timestamp not UTC for year {year}, got {df.index.tz}"
        
        # HARD GATE: Check monotonic
        if not df.index.is_monotonic_increasing:
            return None, fingerprint, f"HARD_FAIL: Timestamps not monotonic increasing for year {year}"
        
        # Skip warmup and take subsample
        total_bars = len(df)
        if total_bars <= warmup:
            return None, fingerprint, f"Not enough bars after warmup for year {year}"
        
        # Deterministic subsample: take evenly spaced bars
        available_bars = total_bars - warmup
        sample_size = min(n_bars, available_bars)
        
        # Use step sampling for determinism
        step = max(1, available_bars // sample_size)
        indices = list(range(warmup, total_bars, step))[:sample_size]
        
        df_sample = df.iloc[indices].copy()
        
        return df_sample, fingerprint, None
        
    except Exception as e:
        return None, None, f"Error loading {year}: {str(e)}"


def compute_feature_stats(df: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
    """Compute statistics for a single feature."""
    if feature_name not in df.columns:
        return {
            "count": 0,
            "missing_rate": 1.0,
            "nan_rate": 0.0,
            "inf_rate": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p1": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    
    values = df[feature_name].values
    count = len(values)
    
    # Missing/NaN/Inf rates
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)
    missing_rate = np.sum(nan_mask) / count if count > 0 else 0
    nan_rate = np.sum(nan_mask) / count if count > 0 else 0
    inf_rate = np.sum(inf_mask) / count if count > 0 else 0
    
    # Filter valid values
    valid_mask = ~(nan_mask | inf_mask)
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return {
            "count": count,
            "missing_rate": missing_rate,
            "nan_rate": nan_rate,
            "inf_rate": inf_rate,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p1": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    
    return {
        "count": count,
        "missing_rate": missing_rate,
        "nan_rate": nan_rate,
        "inf_rate": inf_rate,
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values)),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "p1": float(np.percentile(valid_values, 1)),
        "p5": float(np.percentile(valid_values, 5)),
        "p50": float(np.percentile(valid_values, 50)),
        "p95": float(np.percentile(valid_values, 95)),
        "p99": float(np.percentile(valid_values, 99)),
    }


def compute_ks_statistic(
    ref_values: np.ndarray,
    test_values: np.ndarray,
) -> Tuple[float, float]:
    """Compute KS statistic and p-value."""
    # Filter valid values
    ref_valid = ref_values[~(np.isnan(ref_values) | np.isinf(ref_values))]
    test_valid = test_values[~(np.isnan(test_values) | np.isinf(test_values))]
    
    if len(ref_valid) < 10 or len(test_valid) < 10:
        return 0.0, 1.0
    
    try:
        ks_stat, p_value = stats.ks_2samp(ref_valid, test_valid)
        return float(ks_stat), float(p_value)
    except Exception:
        return 0.0, 1.0


def compute_psi(
    ref_values: np.ndarray,
    test_values: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI)."""
    # Filter valid values
    ref_valid = ref_values[~(np.isnan(ref_values) | np.isinf(ref_values))]
    test_valid = test_values[~(np.isnan(test_values) | np.isinf(test_values))]
    
    if len(ref_valid) < n_bins or len(test_valid) < n_bins:
        return 0.0
    
    try:
        # Create bins from reference distribution
        _, bin_edges = np.histogram(ref_valid, bins=n_bins)
        
        # Compute distributions
        ref_counts, _ = np.histogram(ref_valid, bins=bin_edges)
        test_counts, _ = np.histogram(test_valid, bins=bin_edges)
        
        # Convert to proportions (with small epsilon to avoid division by zero)
        eps = 1e-10
        ref_props = (ref_counts + eps) / (len(ref_valid) + n_bins * eps)
        test_props = (test_counts + eps) / (len(test_valid) + n_bins * eps)
        
        # PSI formula
        psi = np.sum((test_props - ref_props) * np.log(test_props / ref_props))
        
        return float(psi)
    except Exception:
        return 0.0


def compute_session_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    """Compute bars per session."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return {}
    
    breakdown = {session: 0 for session in SESSION_DEFINITIONS.keys()}
    
    for ts in df.index:
        hour = ts.hour
        session = get_session(hour)
        breakdown[session] += 1
    
    return breakdown


def run_xgb_inference_pass(
    df: pd.DataFrame,
    bundle_dir: Path,
    year: int,
) -> Optional[pd.DataFrame]:
    """
    Run XGB-only inference to get output columns.
    
    Returns DataFrame with XGB outputs added.
    """
    # Check if XGB outputs already exist
    existing_outputs = [col for col in XGB_OUTPUT_NAMES if col in df.columns]
    if len(existing_outputs) == len(XGB_OUTPUT_NAMES):
        return df
    
    # Try to load XGB model
    try:
        import joblib
        
        # Find XGB model for session (use EU as default)
        xgb_path = bundle_dir / "xgb_EU.pkl"
        if not xgb_path.exists():
            # Try alternative paths
            for session in ["US", "OVERLAP"]:
                alt_path = bundle_dir / f"xgb_{session}.pkl"
                if alt_path.exists():
                    xgb_path = alt_path
                    break
        
        if not xgb_path.exists():
            print(f"  WARNING: No XGB model found in bundle, skipping XGB inference for {year}")
            return df
        
        xgb_model = joblib.load(xgb_path)
        
        # Get feature columns from model
        if hasattr(xgb_model, "feature_cols"):
            feature_cols = xgb_model.feature_cols
        elif hasattr(xgb_model, "feature_names_in_"):
            feature_cols = list(xgb_model.feature_names_in_)
        else:
            print(f"  WARNING: Cannot determine XGB feature columns for {year}")
            return df
        
        # Check if all features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"  WARNING: Missing {len(missing_features)} XGB features for {year}")
            return df
        
        # Run inference
        X = df[feature_cols].values
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get predictions
        if hasattr(xgb_model, "predict_proba"):
            proba = xgb_model.predict_proba(X)
            p_long = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            p_long = xgb_model.predict(X)
        
        # Add outputs
        df = df.copy()
        df["p_long_xgb"] = p_long
        df["p_hat_xgb"] = p_long  # Same as p_long if no calibration
        
        # Compute uncertainty (simple heuristic: distance from 0.5)
        df["uncertainty_score"] = 1.0 - 2 * np.abs(p_long - 0.5)
        
        print(f"  XGB inference completed for {year}: {len(df)} rows")
        
        return df
        
    except Exception as e:
        print(f"  WARNING: XGB inference failed for {year}: {e}")
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Multiyear Feature & XGB Drift Statistics"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to analyze"
    )
    parser.add_argument(
        "--prebuilt-root",
        type=Path,
        default=DEFAULT_PREBUILT_ROOT,
        help="Prebuilt root directory"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Bundle directory for XGB models"
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2025,
        help="Reference year for drift calculation"
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=20000,
        help="Number of bars to sample per year"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup bars to skip"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["all", "xgb_outputs"],
        help="Feature sets to analyze"
    )
    parser.add_argument(
        "--include-session-breakdown",
        action="store_true",
        default=True,
        help="Include session breakdown"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = WORKSPACE_ROOT / "reports" / "pipeline_audit" / f"V10_HYBRID_DRIFT_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MULTIYEAR FEATURE & XGB DRIFT STATISTICS")
    print("=" * 60)
    print(f"Years: {args.years}")
    print(f"Reference year: {args.reference_year}")
    print(f"Bars per year: {args.n_bars}")
    print(f"Output: {output_dir}")
    print()
    
    # Load samples for each year
    samples = {}
    fingerprints = {}
    session_breakdowns = {}
    
    print("Loading prebuilt samples...")
    for year in args.years:
        print(f"  Loading {year}...")
        df, fingerprint, error = load_prebuilt_sample(
            year=year,
            prebuilt_root=args.prebuilt_root,
            n_bars=args.n_bars,
            warmup=args.warmup,
        )
        
        if error:
            if error.startswith("HARD_FAIL"):
                print(f"    {error}")
                return 1
            print(f"    WARNING: {error}")
            continue
        
        # Run XGB inference if needed
        df = run_xgb_inference_pass(df, args.bundle_dir, year)
        
        # HARD GATE: Check XGB outputs for NaN/Inf
        for xgb_col in XGB_OUTPUT_NAMES:
            if xgb_col in df.columns:
                nan_rate = df[xgb_col].isna().mean()
                inf_rate = np.isinf(df[xgb_col]).mean()
                if nan_rate > 0 or inf_rate > 0:
                    print(f"    HARD_FAIL: {xgb_col} has NaN rate={nan_rate:.4f}, Inf rate={inf_rate:.4f}")
                    return 1
        
        samples[year] = df
        fingerprints[year] = fingerprint
        session_breakdowns[year] = compute_session_breakdown(df)
        
        print(f"    Loaded {len(df)} bars, fingerprint={fingerprint}")
    
    if not samples:
        print("ERROR: No samples loaded")
        return 1
    
    # Get reference sample
    if args.reference_year not in samples:
        print(f"ERROR: Reference year {args.reference_year} not loaded")
        return 1
    
    ref_df = samples[args.reference_year]
    
    # Determine features to analyze
    all_features = list(ref_df.columns)
    xgb_output_features = [f for f in XGB_OUTPUT_NAMES if f in all_features]
    
    # Get XGB input features from bundle if available
    xgb_input_features = []
    try:
        import joblib
        xgb_path = args.bundle_dir / "xgb_EU.pkl"
        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)
            if hasattr(xgb_model, "feature_cols"):
                xgb_input_features = [f for f in xgb_model.feature_cols if f in all_features]
            elif hasattr(xgb_model, "feature_names_in_"):
                xgb_input_features = [f for f in xgb_model.feature_names_in_ if f in all_features]
    except Exception:
        pass
    
    if not xgb_input_features:
        # Fallback: use all features except XGB outputs
        xgb_input_features = [f for f in all_features if f not in XGB_OUTPUT_NAMES]
    
    print(f"\nFeatures to analyze:")
    print(f"  All features: {len(all_features)}")
    print(f"  XGB input features: {len(xgb_input_features)}")
    print(f"  XGB output features: {len(xgb_output_features)}")
    
    # Compute feature statistics
    print("\nComputing feature statistics...")
    feature_stats = []
    
    for year, df in samples.items():
        for feature in all_features:
            stats_dict = compute_feature_stats(df, feature)
            stats_dict["year"] = year
            stats_dict["feature"] = feature
            stats_dict["feature_type"] = (
                "xgb_output" if feature in XGB_OUTPUT_NAMES else
                "xgb_input" if feature in xgb_input_features else
                "other"
            )
            feature_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(feature_stats)
    
    # Save feature stats
    stats_csv_path = output_dir / "FEATURE_STATS_BY_YEAR.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Written: {stats_csv_path}")
    
    # Compute drift metrics
    print("\nComputing drift metrics...")
    drift_metrics = []
    
    for year, df in samples.items():
        if year == args.reference_year:
            continue
        
        for feature in all_features:
            if feature not in ref_df.columns or feature not in df.columns:
                continue
            
            ref_values = ref_df[feature].values
            test_values = df[feature].values
            
            ks_stat, ks_pval = compute_ks_statistic(ref_values, test_values)
            psi = compute_psi(ref_values, test_values)
            
            drift_metrics.append({
                "year": year,
                "reference_year": args.reference_year,
                "feature": feature,
                "feature_type": (
                    "xgb_output" if feature in XGB_OUTPUT_NAMES else
                    "xgb_input" if feature in xgb_input_features else
                    "other"
                ),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "psi": psi,
            })
    
    drift_df = pd.DataFrame(drift_metrics)
    
    # Save drift metrics
    drift_csv_path = output_dir / "DRIFT_METRICS.csv"
    drift_df.to_csv(drift_csv_path, index=False)
    print(f"Written: {drift_csv_path}")
    
    # Save session breakdown
    if args.include_session_breakdown:
        session_rows = []
        for year, breakdown in session_breakdowns.items():
            for session, count in breakdown.items():
                session_rows.append({
                    "year": year,
                    "session": session,
                    "bars": count,
                })
        
        session_df = pd.DataFrame(session_rows)
        session_csv_path = output_dir / "SESSION_BREAKDOWN.csv"
        session_df.to_csv(session_csv_path, index=False)
        print(f"Written: {session_csv_path}")
    
    # Save fingerprints
    fingerprint_path = output_dir / "SCHEMA_FINGERPRINTS.json"
    with open(fingerprint_path, "w") as f:
        json.dump(fingerprints, f, indent=2)
    print(f"Written: {fingerprint_path}")
    
    # Generate DRIFT_SUMMARY.md
    print("\nGenerating drift summary...")
    
    lines = [
        "# Feature & XGB Drift Summary",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        f"**Reference Year:** {args.reference_year}",
        f"**Bars per Year:** {args.n_bars}",
        "",
        "---",
        "",
        "## Schema Fingerprints",
        "",
        "| Year | Fingerprint | Match Reference |",
        "|------|-------------|-----------------|",
    ]
    
    ref_fingerprint = fingerprints.get(args.reference_year, "")
    for year, fp in sorted(fingerprints.items()):
        match = "✅" if fp == ref_fingerprint else "❌"
        lines.append(f"| {year} | `{fp}` | {match} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Top 20 Features by KS Statistic (XGB Outputs)",
        "",
        "| Year | Feature | KS Stat | P-Value | PSI |",
        "|------|---------|---------|---------|-----|",
    ])
    
    # Top 20 XGB output drift
    xgb_output_drift = drift_df[drift_df["feature_type"] == "xgb_output"].copy()
    xgb_output_drift = xgb_output_drift.sort_values("ks_statistic", ascending=False).head(20)
    
    for _, row in xgb_output_drift.iterrows():
        lines.append(
            f"| {row['year']} | {row['feature']} | {row['ks_statistic']:.4f} | {row['ks_pvalue']:.4f} | {row['psi']:.4f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Top 20 Features by KS Statistic (XGB Inputs)",
        "",
        "| Year | Feature | KS Stat | P-Value | PSI |",
        "|------|---------|---------|---------|-----|",
    ])
    
    # Top 20 XGB input drift
    xgb_input_drift = drift_df[drift_df["feature_type"] == "xgb_input"].copy()
    xgb_input_drift = xgb_input_drift.sort_values("ks_statistic", ascending=False).head(20)
    
    for _, row in xgb_input_drift.iterrows():
        lines.append(
            f"| {row['year']} | {row['feature']} | {row['ks_statistic']:.4f} | {row['ks_pvalue']:.4f} | {row['psi']:.4f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Session Distribution",
        "",
        "| Year | ASIA | EU | US | OVERLAP |",
        "|------|------|----|----|---------|",
    ])
    
    for year in sorted(session_breakdowns.keys()):
        breakdown = session_breakdowns[year]
        lines.append(
            f"| {year} | {breakdown.get('ASIA', 0):,} | {breakdown.get('EU', 0):,} | "
            f"{breakdown.get('US', 0):,} | {breakdown.get('OVERLAP', 0):,} |"
        )
    
    # Session distribution deviation check
    ref_session = session_breakdowns.get(args.reference_year, {})
    session_deviations = []
    for year, breakdown in session_breakdowns.items():
        if year == args.reference_year:
            continue
        for session, count in breakdown.items():
            ref_count = ref_session.get(session, 0)
            if ref_count > 0:
                pct_diff = abs(count - ref_count) / ref_count * 100
                if pct_diff > 10:
                    session_deviations.append(f"{year}/{session}: {pct_diff:.1f}% off")
    
    lines.extend([
        "",
        "---",
        "",
        "## Verdict Hints",
        "",
    ])
    
    # Analyze and generate verdict hints
    verdicts = []
    
    # Check schema mismatch
    schema_mismatches = [y for y, fp in fingerprints.items() if fp != ref_fingerprint]
    if schema_mismatches:
        verdicts.append(f"**(A) PREBUILT MISMATCH:** Schema fingerprints differ for years: {schema_mismatches}")
    
    # Check XGB OOD
    high_ks_xgb = xgb_output_drift[xgb_output_drift["ks_statistic"] > 0.3]
    if len(high_ks_xgb) > 0:
        years_affected = sorted(high_ks_xgb["year"].unique())
        verdicts.append(f"**(B) XGB OOD:** High KS (>0.3) in XGB outputs for years: {years_affected}")
    
    # Check session deviation
    if session_deviations:
        verdicts.append(f"**(C) SESSION/TIMEZONE MAPPING:** Deviations: {session_deviations[:5]}")
    
    if not verdicts:
        lines.append("✅ No major drift detected across years.")
    else:
        for verdict in verdicts:
            lines.append(f"- {verdict}")
    
    lines.extend([
        "",
        "---",
        "",
        "## XGB Output Statistics by Year",
        "",
        "| Year | Feature | Mean | Std | Min | Max | P50 |",
        "|------|---------|------|-----|-----|-----|-----|",
    ])
    
    for feature in XGB_OUTPUT_NAMES:
        feature_stats_subset = stats_df[stats_df["feature"] == feature]
        for _, row in feature_stats_subset.iterrows():
            if row["mean"] is not None:
                lines.append(
                    f"| {row['year']} | {feature} | {row['mean']:.4f} | {row['std']:.4f} | "
                    f"{row['min']:.4f} | {row['max']:.4f} | {row['p50']:.4f} |"
                )
    
    lines.append("")
    
    summary_path = output_dir / "DRIFT_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {summary_path}")
    
    # Summary
    print()
    print("=" * 60)
    print("DRIFT ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Output files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    print()
    print(f"Output directory: {output_dir}")
    
    if verdicts:
        print()
        print("⚠️  VERDICT HINTS:")
        for v in verdicts:
            print(f"  {v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
