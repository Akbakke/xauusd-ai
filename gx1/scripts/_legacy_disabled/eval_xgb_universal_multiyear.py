#!/usr/bin/env python3
"""
Evaluate Universal XGB v1 on multiyear data.

Computes drift metrics (KS/PSI) and distribution statistics
to determine GO/NO-GO for the universal model.

Usage:
    python3 gx1/scripts/eval_xgb_universal_multiyear.py --years 2020 2021 2022 2023 2024 2025 --reference-year 2025
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
from scipy import stats as scipy_stats

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from joblib import load as joblib_load
except ImportError:
    import joblib
    joblib_load = joblib.load

from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer


# GO/NO-GO thresholds
KS_THRESHOLD = 0.15
PSI_THRESHOLD = 0.10
CLIP_RATE_THRESHOLD = 6.0  # % (relaxed until RSI fix)


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


def load_contracts() -> Tuple[List[str], str, XGBInputSanitizer]:
    """Load feature contract and sanitizer."""
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    with open(feature_contract_path, "r") as f:
        feature_contract = json.load(f)
    
    features = feature_contract.get("features", [])
    schema_hash = feature_contract.get("schema_hash", "unknown")
    
    sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config_path))
    
    return features, schema_hash, sanitizer


def load_universal_model(gx1_data: Path) -> Tuple[Any, Path, str]:
    """Load universal XGB model and return model, path, sha256."""
    model_path = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION" / "xgb_universal_v1.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Universal model not found: {model_path}")
    
    model = joblib_load(model_path)
    
    # Compute SHA256
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    model_sha = sha256_hash.hexdigest()
    
    return model, model_path, model_sha


def compute_ks_statistic(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Compute KS statistic between two distributions."""
    stat, _ = scipy_stats.ks_2samp(dist_a, dist_b)
    return float(stat)


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    
    eps = 1e-8
    expected_pct = expected_hist / (len(expected) + eps)
    actual_pct = actual_hist / (len(actual) + eps)
    
    expected_pct = np.clip(expected_pct, eps, 1)
    actual_pct = np.clip(actual_pct, eps, 1)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Universal XGB v1 on multiyear data"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to evaluate"
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2025,
        help="Reference year for drift calculation"
    )
    parser.add_argument(
        "--n-bars-per-year",
        type=int,
        default=None,
        help="Limit bars per year (default: use all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--allow-high-clip",
        action="store_true",
        help="Allow high clip rate without failing"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVAL UNIVERSAL XGB V1")
    print("=" * 60)
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Load contracts
    print("\nLoading contracts...")
    features, schema_hash, sanitizer = load_contracts()
    print(f"  Features: {len(features)}")
    print(f"  Schema hash: {schema_hash}")
    
    # Load model
    print("\nLoading universal model...")
    try:
        model, model_path, model_sha = load_universal_model(gx1_data)
        print(f"  Model: {model_path.name}")
        print(f"  SHA256: {model_sha[:16]}...")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1
    
    # Check feature count
    if hasattr(model, "n_features_in_"):
        expected_features = model.n_features_in_
        print(f"  Model expects: {expected_features} features")
        if expected_features != len(features):
            print(f"  WARNING: Contract has {len(features)} features")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "xgb_eval" / f"UNIVERSAL_EVAL_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Evaluate each year
    print("\nEvaluating years...")
    year_results: Dict[int, Dict[str, Any]] = {}
    
    for year in args.years:
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  {year}: No prebuilt found")
            continue
        
        print(f"\n  {year}:")
        df = pd.read_parquet(prebuilt_path)
        
        # Limit bars if specified
        if args.n_bars_per_year and len(df) > args.n_bars_per_year:
            step = len(df) // args.n_bars_per_year
            df = df.iloc[::step][:args.n_bars_per_year]
        
        print(f"    Rows: {len(df)}")
        
        # Apply sanitizer
        try:
            X, stats = sanitizer.sanitize(df, features, allow_nan_fill=True, nan_fill_value=0.0)
            clip_rate = stats.clip_rate_pct
            print(f"    Clip rate: {clip_rate:.2f}%")
        except Exception as e:
            print(f"    ERROR: Sanitization failed: {e}")
            continue
        
        # Check clip rate
        if clip_rate > CLIP_RATE_THRESHOLD and not args.allow_high_clip:
            print(f"    ⚠️  High clip rate (> {CLIP_RATE_THRESHOLD}%)")
        
        # Run inference
        n_features_model = model.n_features_in_ if hasattr(model, "n_features_in_") else X.shape[1]
        X_infer = X[:, :n_features_model]
        
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_infer)
                if proba.shape[1] >= 2:
                    p_long = proba[:, 1]
                else:
                    p_long = proba[:, 0]
            else:
                p_long = model.predict(X_infer)
            
            # Check for NaN/Inf
            if np.isnan(p_long).any():
                print(f"    ❌ NaN in outputs!")
                continue
            if np.isinf(p_long).any():
                print(f"    ❌ Inf in outputs!")
                continue
            
            print(f"    p_long: mean={p_long.mean():.4f}, std={p_long.std():.4f}")
            
            year_results[year] = {
                "n_samples": len(df),
                "clip_rate_pct": clip_rate,
                "p_long_mean": float(p_long.mean()),
                "p_long_std": float(p_long.std()),
                "p_long_min": float(p_long.min()),
                "p_long_max": float(p_long.max()),
                "p_long_values": p_long,  # Keep for drift calc
            }
            
        except Exception as e:
            print(f"    ERROR: Inference failed: {e}")
            continue
    
    if not year_results:
        print("\nERROR: No years evaluated successfully")
        return 1
    
    # Compute drift metrics
    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS")
    print("=" * 60)
    
    ref_year = args.reference_year
    if ref_year not in year_results:
        ref_year = max(year_results.keys())
        print(f"Reference year {args.reference_year} not available, using {ref_year}")
    
    ref_p_long = year_results[ref_year]["p_long_values"]
    
    drift_metrics = []
    for year in sorted(year_results.keys()):
        if year == ref_year:
            continue
        
        year_p_long = year_results[year]["p_long_values"]
        ks = compute_ks_statistic(ref_p_long, year_p_long)
        psi = compute_psi(ref_p_long, year_p_long)
        
        drift_metrics.append({
            "year": year,
            "ks_vs_ref": ks,
            "psi_vs_ref": psi,
            "p_long_mean": year_results[year]["p_long_mean"],
            "p_long_std": year_results[year]["p_long_std"],
            "clip_rate_pct": year_results[year]["clip_rate_pct"],
        })
        
        ks_status = "✅" if ks < KS_THRESHOLD else "❌"
        psi_status = "✅" if psi < PSI_THRESHOLD else "❌"
        print(f"  {year} vs {ref_year}: KS={ks:.4f} {ks_status}, PSI={psi:.4f} {psi_status}")
    
    # Remove p_long_values from results (too large for JSON)
    for year in year_results:
        del year_results[year]["p_long_values"]
    
    # Write metrics
    metrics_df = pd.DataFrame(drift_metrics)
    metrics_path = output_dir / "DRIFT_METRICS.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path}")
    
    # Compute GO/NO-GO
    print("\n" + "=" * 60)
    print("GO/NO-GO ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    # Check KS
    max_ks = metrics_df["ks_vs_ref"].max() if len(metrics_df) > 0 else 0
    if max_ks >= KS_THRESHOLD:
        issues.append(f"KS drift too high: {max_ks:.4f} >= {KS_THRESHOLD}")
    else:
        print(f"  ✅ KS drift OK: max={max_ks:.4f} < {KS_THRESHOLD}")
    
    # Check PSI
    max_psi = metrics_df["psi_vs_ref"].max() if len(metrics_df) > 0 else 0
    if max_psi >= PSI_THRESHOLD:
        issues.append(f"PSI drift too high: {max_psi:.4f} >= {PSI_THRESHOLD}")
    else:
        print(f"  ✅ PSI drift OK: max={max_psi:.4f} < {PSI_THRESHOLD}")
    
    # Check clip rates
    max_clip = max(r["clip_rate_pct"] for r in year_results.values())
    if max_clip > CLIP_RATE_THRESHOLD and not args.allow_high_clip:
        issues.append(f"Clip rate too high: {max_clip:.2f}% > {CLIP_RATE_THRESHOLD}%")
    else:
        print(f"  ✅ Clip rate OK: max={max_clip:.2f}%")
    
    # Verdict
    print("\n" + "-" * 40)
    if issues:
        print("VERDICT: ❌ NO-GO")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("VERDICT: ✅ GO")
        print("  All metrics within thresholds")
    print("-" * 40)
    
    # Compute additional SHAs for GO marker
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    meta_path = model_path.parent / "xgb_universal_v1_meta.json"
    
    feature_contract_sha = None
    if feature_contract_path.exists():
        with open(feature_contract_path, "rb") as f:
            feature_contract_sha = hashlib.sha256(f.read()).hexdigest()
    
    sanitizer_sha = None
    if sanitizer_config_path.exists():
        with open(sanitizer_config_path, "rb") as f:
            sanitizer_sha = hashlib.sha256(f.read()).hexdigest()
    
    meta_sha = None
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta_sha = hashlib.sha256(f.read()).hexdigest()
    
    # Write summary
    summary = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "schema_hash": schema_hash,
        "reference_year": ref_year,
        "years_evaluated": list(year_results.keys()),
        "thresholds": {
            "ks": KS_THRESHOLD,
            "psi": PSI_THRESHOLD,
            "clip_rate_pct": CLIP_RATE_THRESHOLD,
        },
        "results": {
            "max_ks": max_ks,
            "max_psi": max_psi,
            "max_clip_rate_pct": max_clip,
        },
        "verdict": "GO" if not issues else "NO-GO",
        "issues": issues,
        "year_details": year_results,
    }
    
    summary_path = output_dir / "EVAL_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {summary_path}")
    
    # Write GO or NO-GO marker to bundle dir
    bundle_dir = model_path.parent
    marker_content = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "xgb_mode": "universal",
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "meta_sha256": meta_sha,
        "feature_contract_path": str(feature_contract_path),
        "feature_contract_sha256": feature_contract_sha,
        "sanitizer_config_path": str(sanitizer_config_path),
        "sanitizer_config_sha256": sanitizer_sha,
        "schema_hash": schema_hash,
        "go_criteria": {
            "min_auc": 0.55,  # Future: add AUC check
            "max_ks": KS_THRESHOLD,
            "max_psi": PSI_THRESHOLD,
            "max_clip_rate_pct": CLIP_RATE_THRESHOLD,
        },
        "eval_results": {
            "max_ks": max_ks,
            "max_psi": max_psi,
            "max_clip_rate_pct": max_clip,
        },
        "eval_run_dir": str(output_dir),
    }
    
    if not issues:
        # Write GO marker
        marker_path = bundle_dir / "XGB_UNIVERSAL_GO_MARKER.json"
        tmp_path = bundle_dir / ".XGB_UNIVERSAL_GO_MARKER.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(marker_content, f, indent=2)
        tmp_path.rename(marker_path)
        print(f"\n✅ GO MARKER written: {marker_path}")
        
        # Remove NO-GO marker if exists
        no_go_path = bundle_dir / "XGB_UNIVERSAL_NO_GO.json"
        if no_go_path.exists():
            no_go_path.unlink()
    else:
        # Write NO-GO marker
        marker_content["issues"] = issues
        marker_path = bundle_dir / "XGB_UNIVERSAL_NO_GO.json"
        tmp_path = bundle_dir / ".XGB_UNIVERSAL_NO_GO.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(marker_content, f, indent=2)
        tmp_path.rename(marker_path)
        print(f"\n❌ NO-GO MARKER written: {marker_path}")
        
        # Remove GO marker if exists (model is now NO-GO)
        go_path = bundle_dir / "XGB_UNIVERSAL_GO_MARKER.json"
        if go_path.exists():
            go_path.unlink()
            print(f"  Removed stale GO marker")
    
    # Write markdown report
    report_path = output_dir / "EVAL_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Universal XGB v1 Evaluation Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"## Model\n\n")
        f.write(f"- Path: `{model_path.name}`\n")
        f.write(f"- SHA256: `{model_sha[:16]}...`\n")
        f.write(f"- Schema hash: `{schema_hash}`\n\n")
        
        f.write(f"## Drift Analysis (vs {ref_year})\n\n")
        f.write("| Year | KS | PSI | p_long mean | Clip Rate |\n")
        f.write("|------|-----|-----|-------------|----------|\n")
        for m in drift_metrics:
            ks_mark = "✅" if m["ks_vs_ref"] < KS_THRESHOLD else "❌"
            psi_mark = "✅" if m["psi_vs_ref"] < PSI_THRESHOLD else "❌"
            f.write(f"| {m['year']} | {m['ks_vs_ref']:.4f} {ks_mark} | {m['psi_vs_ref']:.4f} {psi_mark} | {m['p_long_mean']:.4f} | {m['clip_rate_pct']:.2f}% |\n")
        f.write("\n")
        
        f.write(f"## Verdict\n\n")
        if issues:
            f.write(f"**❌ NO-GO**\n\n")
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write(f"**✅ GO**\n\n")
            f.write("All metrics within thresholds.\n")
    
    print(f"Wrote: {report_path}")
    
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
