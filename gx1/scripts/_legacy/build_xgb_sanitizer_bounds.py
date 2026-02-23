#!/usr/bin/env python3
"""
Build XGB Sanitizer Bounds from multiyear stats.

Reads the XGB input contract report and generates sanitizer config
with quantile bounds for all features and fixed bounds for known-bad features.

Usage:
    python3 gx1/scripts/build_xgb_sanitizer_bounds.py
    python3 gx1/scripts/build_xgb_sanitizer_bounds.py --quantiles 0.005 0.995
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


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
        gx1_data / "data" / "data" / "features" / f"xauusd_m5_{year}_features_v10_ctx.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_feature_contract() -> Tuple[List[str], str]:
    """Load feature contract and return features + schema hash."""
    contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Feature contract not found: {contract_path}")
    
    with open(contract_path, "r") as f:
        contract = json.load(f)
    
    features = contract.get("features", [])
    schema_hash = contract.get("schema_hash", "unknown")
    
    return features, schema_hash


# Known problematic features that need fixed bounds
KNOWN_BAD_FEATURES = {
    "_v1_rsi14": (0.0, 100.0),  # RSI is 0-100 by definition
    "_v1_rsi14_z": (-10.0, 10.0),  # Z-score should be small
    "_v1_rsi2": (0.0, 100.0),
    "_v1h1_rsi14_z": (-10.0, 10.0),
    "_v1h4_rsi14_z": (-10.0, 10.0),
}


def compute_quantile_bounds(
    years: List[int],
    features: List[str],
    gx1_data: Path,
    n_bars_per_year: int = 50000,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> Dict[str, Tuple[float, float]]:
    """Compute quantile bounds from multiyear data."""
    
    # Collect samples from all years
    all_samples = {f: [] for f in features}
    
    for year in years:
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  WARNING: No prebuilt for {year}, skipping")
            continue
        
        print(f"  Loading {year} from {prebuilt_path.name}...")
        df = pd.read_parquet(prebuilt_path)
        
        # Sample if needed
        if len(df) > n_bars_per_year:
            step = len(df) // n_bars_per_year
            df = df.iloc[::step][:n_bars_per_year]
        
        # Collect samples for each feature
        for feature in features:
            if feature in df.columns:
                values = df[feature].dropna().values
                # Filter out extreme values for quantile calculation
                values = values[np.isfinite(values)]
                all_samples[feature].extend(values.tolist())
    
    # Compute bounds
    bounds = {}
    for feature in features:
        samples = np.array(all_samples[feature])
        if len(samples) > 0:
            lower = float(np.percentile(samples, lower_q * 100))
            upper = float(np.percentile(samples, upper_q * 100))
            
            # Apply fixed bounds for known-bad features
            if feature in KNOWN_BAD_FEATURES:
                fixed_lower, fixed_upper = KNOWN_BAD_FEATURES[feature]
                lower = max(lower, fixed_lower)
                upper = min(upper, fixed_upper)
            
            bounds[feature] = (lower, upper)
        else:
            print(f"  WARNING: No samples for {feature}")
    
    return bounds


def main():
    parser = argparse.ArgumentParser(
        description="Build XGB Sanitizer Bounds from multiyear stats"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to use for quantile calculation"
    )
    parser.add_argument(
        "--n-bars-per-year",
        type=int,
        default=50000,
        help="Number of bars to sample per year"
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=2,
        default=[0.005, 0.995],
        help="Lower and upper quantiles (default: 0.005 0.995)"
    )
    parser.add_argument(
        "--hard-fail-abs-max",
        type=float,
        default=1e9,
        help="Hard fail threshold for absolute values"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: gx1/xgb/contracts/)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Load feature contract
    print("Loading feature contract...")
    features, schema_hash = load_feature_contract()
    print(f"  Features: {len(features)}")
    print(f"  Schema hash: {schema_hash}")
    
    # Compute bounds
    print(f"\nComputing quantile bounds from {args.years}...")
    lower_q, upper_q = args.quantiles
    bounds = compute_quantile_bounds(
        years=args.years,
        features=features,
        gx1_data=gx1_data,
        n_bars_per_year=args.n_bars_per_year,
        lower_q=lower_q,
        upper_q=upper_q,
    )
    print(f"  Computed bounds for {len(bounds)} features")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build sanitizer config
    sanitizer_config = {
        "version": "v1",
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "schema_hash": schema_hash,
        "clip_method": "quantile",
        "quantiles": [lower_q, upper_q],
        "hard_fail_on_nan": True,
        "hard_fail_on_inf": True,
        "hard_fail_abs_max": args.hard_fail_abs_max,
        "max_clip_rate_pct": 10.0,
        "feature_contract_path": "gx1/xgb/contracts/xgb_input_features_v1.json",
        "feature_list": features,
        "bounds": {f: list(b) for f, b in bounds.items()},
        "known_bad_features": {f: list(b) for f, b in KNOWN_BAD_FEATURES.items()},
        "provenance": {
            "years_used": args.years,
            "n_bars_per_year": args.n_bars_per_year,
        },
    }
    
    # Compute SHA256 of config
    config_json = json.dumps(sanitizer_config, sort_keys=True)
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    sanitizer_config["config_sha256"] = config_sha256
    
    # Write sanitizer config
    config_path = output_dir / "xgb_input_sanitizer_v1.json"
    with open(config_path, "w") as f:
        json.dump(sanitizer_config, f, indent=2)
    print(f"\nWrote: {config_path}")
    
    # Write summary markdown
    summary_path = output_dir / "SANITIZER_BOUNDS_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# XGB Input Sanitizer Bounds Summary\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Schema hash: `{schema_hash}`\n")
        f.write(f"- Config SHA256: `{config_sha256[:16]}...`\n")
        f.write(f"- Clip method: {sanitizer_config['clip_method']}\n")
        f.write(f"- Quantiles: {lower_q}, {upper_q}\n")
        f.write(f"- Hard fail abs max: {args.hard_fail_abs_max:.2e}\n")
        f.write(f"- Years used: {args.years}\n")
        f.write(f"- Bars per year: {args.n_bars_per_year}\n\n")
        
        f.write(f"## Known Bad Features (Fixed Bounds)\n\n")
        f.write("| Feature | Lower | Upper | Reason |\n")
        f.write("|---------|-------|-------|--------|\n")
        for feature, (lower, upper) in KNOWN_BAD_FEATURES.items():
            f.write(f"| {feature} | {lower} | {upper} | RSI/Z-score constraint |\n")
        f.write("\n")
        
        f.write(f"## All Bounds (Quantile-based)\n\n")
        f.write(f"Total features: {len(bounds)}\n\n")
        f.write("| Feature | Lower | Upper | Range |\n")
        f.write("|---------|-------|-------|-------|\n")
        for feature in sorted(bounds.keys()):
            lower, upper = bounds[feature]
            range_val = upper - lower
            f.write(f"| {feature} | {lower:.4f} | {upper:.4f} | {range_val:.4f} |\n")
    
    print(f"Wrote: {summary_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SANITIZER BOUNDS COMPLETE")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Summary: {summary_path}")
    print(f"Features: {len(features)}")
    print(f"Bounds: {len(bounds)}")
    print(f"Known bad: {len(KNOWN_BAD_FEATURES)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
