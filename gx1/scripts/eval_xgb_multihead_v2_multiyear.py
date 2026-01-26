#!/usr/bin/env python3
"""
Evaluate Universal Multi-head XGB v2 on multiyear data.

Computes drift metrics (KS/PSI) per output, per head, per year.
Writes GO/NO-GO marker based on evaluation results.

Usage:
    python3 gx1/scripts/eval_xgb_multihead_v2_multiyear.py --years 2020 2021 2022 2023 2024 2025 --reference-year 2025
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

from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
from gx1.time.session_detector import get_session_vectorized, get_session_stats


# GO/NO-GO thresholds
KS_THRESHOLD = 0.15
PSI_THRESHOLD = 0.10
CLIP_RATE_THRESHOLD = 6.0
MIN_CLASS_RATE = 0.02  # No class below 2%


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


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_sanitizer() -> Tuple[XGBInputSanitizer, List[str], str]:
    """Load sanitizer and feature list."""
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    with open(feature_contract_path, "r") as f:
        contract = json.load(f)
    features = contract.get("features", [])
    schema_hash = contract.get("schema_hash", "unknown")
    
    sanitizer_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_path))
    
    return sanitizer, features, schema_hash


def load_multihead_model(gx1_data: Path) -> Tuple[XGBMultiheadModel, Path, str]:
    """Load multihead model."""
    model_path = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION" / "xgb_universal_multihead_v2.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Multihead model not found: {model_path}")
    
    model = XGBMultiheadModel.load(str(model_path))
    model_sha = compute_file_sha256(model_path)
    
    return model, model_path, model_sha


def compute_ks_statistic(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Compute KS statistic."""
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
        description="Evaluate Universal Multi-head XGB v2"
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
        help="Reference year for drift"
    )
    parser.add_argument(
        "--n-bars-per-year",
        type=int,
        default=None,
        help="Limit bars per year"
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
        help="Allow high clip rate"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVAL UNIVERSAL MULTI-HEAD XGB V2")
    print("=" * 60)
    
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Load model
    print("\nLoading multihead model...")
    try:
        model, model_path, model_sha = load_multihead_model(gx1_data)
        print(f"  Model: {model_path.name}")
        print(f"  SHA256: {model_sha[:16]}...")
        print(f"  Sessions: {model.sessions}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1
    
    # Load sanitizer
    print("\nLoading sanitizer...")
    sanitizer, features, schema_hash = load_sanitizer()
    print(f"  Features: {len(features)}")
    print(f"  Schema hash: {schema_hash}")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "xgb_eval" / f"MULTIHEAD_EVAL_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Evaluate each year - per head, per session
    print("\nEvaluating years (per-head, per-session)...")
    year_results: Dict[int, Dict[str, Any]] = {}
    
    for year in args.years:
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  {year}: No prebuilt found")
            continue
        
        print(f"\n  {year}:")
        df = pd.read_parquet(prebuilt_path)
        
        if args.n_bars_per_year and len(df) > args.n_bars_per_year:
            step = len(df) // args.n_bars_per_year
            df = df.iloc[::step][:args.n_bars_per_year]
        
        print(f"    Rows: {len(df)}")
        
        # Compute session for each row
        df["_session"] = get_session_vectorized(df.get("timestamp", df.get("ts", df.index)))
        session_stats = get_session_stats(df["_session"])
        print(f"    Session distribution: {session_stats['percentages']}")
        
        # Sanitize
        try:
            X, stats = sanitizer.sanitize(df, features, allow_nan_fill=True, nan_fill_value=0.0)
            df_sanitized = df.copy()
            for i, f in enumerate(features):
                if i < X.shape[1]:
                    df_sanitized[f] = X[:, i]
            clip_rate = stats.clip_rate_pct
            print(f"    Clip rate: {clip_rate:.2f}%")
        except Exception as e:
            print(f"    ERROR: Sanitization failed: {e}")
            continue
        
        year_results[year] = {
            "clip_rate_pct": clip_rate,
            "n_samples": len(df),
            "session_distribution": session_stats["percentages"],
            "heads": {},
        }
        
        # Evaluate each head on its session-filtered data
        for session in model.sessions:
            # Filter to session rows only
            session_mask = df["_session"] == session
            n_session_rows = session_mask.sum()
            
            if n_session_rows == 0:
                print(f"    {session}: No rows in this session")
                continue
            
            df_session = df_sanitized[session_mask]
            
            try:
                outputs = model.predict_proba(df_session, session, features)
                
                year_results[year]["heads"][session] = {
                    "n_rows": n_session_rows,
                    "p_long_mean": float(outputs.p_long.mean()),
                    "p_short_mean": float(outputs.p_short.mean()),
                    "p_flat_mean": float(outputs.p_flat.mean()),
                    "uncertainty_mean": float(outputs.uncertainty.mean()),
                    "p_long_values": outputs.p_long,
                    "p_short_values": outputs.p_short,
                    "p_flat_values": outputs.p_flat,
                    "uncertainty_values": outputs.uncertainty,
                }
                
                print(f"    {session} ({n_session_rows} bars): "
                      f"p_long={outputs.p_long.mean():.3f}, "
                      f"p_short={outputs.p_short.mean():.3f}, "
                      f"p_flat={outputs.p_flat.mean():.3f}, "
                      f"unc={outputs.uncertainty.mean():.3f}")
                
            except Exception as e:
                print(f"    {session}: ERROR - {e}")
    
    if not year_results:
        print("\nERROR: No years evaluated")
        return 1
    
    # Compute drift metrics
    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS (per head, per session)")
    print("=" * 60)
    
    ref_year = args.reference_year
    if ref_year not in year_results:
        ref_year = max(year_results.keys())
        print(f"Reference year {args.reference_year} not available, using {ref_year}")
    
    drift_metrics = []
    max_ks = 0.0
    max_psi = 0.0
    max_ks_by_head: Dict[str, float] = {}
    max_psi_by_head: Dict[str, float] = {}
    worst_offenders = []
    
    for session in model.sessions:
        if session not in year_results[ref_year]["heads"]:
            continue
        
        ref_outputs = year_results[ref_year]["heads"][session]
        max_ks_by_head[session] = 0.0
        max_psi_by_head[session] = 0.0
        
        for year in sorted(year_results.keys()):
            if year == ref_year:
                continue
            if session not in year_results[year]["heads"]:
                continue
            
            year_outputs = year_results[year]["heads"][session]
            
            # Compute KS/PSI for each output
            for output_name in ["p_long", "p_short", "p_flat", "uncertainty"]:
                ref_values = ref_outputs[f"{output_name}_values"]
                year_values = year_outputs[f"{output_name}_values"]
                
                ks = compute_ks_statistic(ref_values, year_values)
                psi = compute_psi(ref_values, year_values)
                
                max_ks = max(max_ks, ks)
                max_psi = max(max_psi, psi)
                max_ks_by_head[session] = max(max_ks_by_head[session], ks)
                max_psi_by_head[session] = max(max_psi_by_head[session], psi)
                
                drift_metrics.append({
                    "session": session,
                    "output": output_name,
                    "year": year,
                    "ks_vs_ref": ks,
                    "psi_vs_ref": psi,
                })
                
                # Track worst offenders
                if ks > KS_THRESHOLD or psi > PSI_THRESHOLD:
                    worst_offenders.append({
                        "year": year,
                        "session": session,
                        "output": output_name,
                        "ks": ks,
                        "psi": psi,
                    })
    
    # Sort worst offenders
    worst_offenders = sorted(worst_offenders, key=lambda x: -x["ks"])[:10]
    
    # Print summary
    print(f"\nMax KS overall: {max_ks:.4f} (threshold: {KS_THRESHOLD})")
    print(f"Max PSI overall: {max_psi:.4f} (threshold: {PSI_THRESHOLD})")
    print(f"\nMax KS by head:")
    for session, ks_val in max_ks_by_head.items():
        status = "✅" if ks_val < KS_THRESHOLD else "❌"
        print(f"  {session}: {ks_val:.4f} {status}")
    print(f"\nMax PSI by head:")
    for session, psi_val in max_psi_by_head.items():
        status = "✅" if psi_val < PSI_THRESHOLD else "❌"
        print(f"  {session}: {psi_val:.4f} {status}")
    
    if worst_offenders:
        print(f"\nWorst offenders:")
        for wo in worst_offenders[:5]:
            print(f"  {wo['year']}/{wo['session']}/{wo['output']}: KS={wo['ks']:.4f}, PSI={wo['psi']:.4f}")
    
    # Write drift metrics
    metrics_df = pd.DataFrame(drift_metrics)
    metrics_path = output_dir / "DRIFT_METRICS.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path}")
    
    # Remove values from year_results for JSON
    for year in year_results:
        for session in list(year_results[year].get("heads", {}).keys()):
            for key in list(year_results[year]["heads"][session].keys()):
                if key.endswith("_values"):
                    del year_results[year]["heads"][session][key]
    
    # Check class distribution
    class_issues = []
    for year in year_results:
        for session in year_results[year].get("heads", {}):
            head = year_results[year]["heads"][session]
            for class_name in ["p_long", "p_short", "p_flat"]:
                rate = head[f"{class_name}_mean"]
                if rate < MIN_CLASS_RATE:
                    class_issues.append(f"{year}/{session}: {class_name} = {rate:.2%} < {MIN_CLASS_RATE:.0%}")
    
    # GO/NO-GO analysis
    print("\n" + "=" * 60)
    print("GO/NO-GO ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    if max_ks >= KS_THRESHOLD:
        issues.append(f"KS drift too high: {max_ks:.4f} >= {KS_THRESHOLD}")
    else:
        print(f"  ✅ KS drift OK: max={max_ks:.4f} < {KS_THRESHOLD}")
    
    if max_psi >= PSI_THRESHOLD:
        issues.append(f"PSI drift too high: {max_psi:.4f} >= {PSI_THRESHOLD}")
    else:
        print(f"  ✅ PSI drift OK: max={max_psi:.4f} < {PSI_THRESHOLD}")
    
    max_clip = max(r["clip_rate_pct"] for r in year_results.values())
    if max_clip > CLIP_RATE_THRESHOLD and not args.allow_high_clip:
        issues.append(f"Clip rate too high: {max_clip:.2f}% > {CLIP_RATE_THRESHOLD}%")
    else:
        print(f"  ✅ Clip rate OK: max={max_clip:.2f}%")
    
    if class_issues:
        issues.extend(class_issues[:3])  # Limit to first 3
        print(f"  ⚠️  Class distribution issues: {len(class_issues)}")
    else:
        print(f"  ✅ Class distribution OK")
    
    print("\n" + "-" * 40)
    if issues:
        print("VERDICT: ❌ NO-GO")
        for issue in issues[:5]:
            print(f"  - {issue}")
    else:
        print("VERDICT: ✅ GO")
    print("-" * 40)
    
    # Load contract SHAs
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    sanitizer_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"
    meta_path = model_path.parent / "xgb_universal_multihead_v2_meta.json"
    
    feature_contract_sha = compute_file_sha256(feature_contract_path) if feature_contract_path.exists() else None
    sanitizer_sha = compute_file_sha256(sanitizer_path) if sanitizer_path.exists() else None
    output_contract_sha = compute_file_sha256(output_contract_path) if output_contract_path.exists() else None
    meta_sha = compute_file_sha256(meta_path) if meta_path.exists() else None
    
    # Write marker
    bundle_dir = model_path.parent
    marker_content = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "xgb_mode": "universal_multihead",
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "meta_sha256": meta_sha,
        "feature_contract_sha256": feature_contract_sha,
        "sanitizer_sha256": sanitizer_sha,
        "output_contract_sha256": output_contract_sha,
        "schema_hash": schema_hash,
        "sessions": model.sessions,
        "go_criteria": {
            "max_ks": KS_THRESHOLD,
            "max_psi": PSI_THRESHOLD,
            "max_clip_rate_pct": CLIP_RATE_THRESHOLD,
            "min_class_rate": MIN_CLASS_RATE,
        },
        "eval_results": {
            "max_ks": max_ks,
            "max_psi": max_psi,
            "max_ks_by_head": max_ks_by_head,
            "max_psi_by_head": max_psi_by_head,
            "max_clip_rate_pct": max_clip,
        },
        "worst_offenders": worst_offenders[:5],
        "eval_run_dir": str(output_dir),
    }
    
    if not issues:
        marker_path = bundle_dir / "XGB_MULTIHEAD_GO_MARKER.json"
        tmp_path = bundle_dir / ".XGB_MULTIHEAD_GO_MARKER.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(marker_content, f, indent=2)
        tmp_path.rename(marker_path)
        print(f"\n✅ GO MARKER written: {marker_path}")
        
        no_go_path = bundle_dir / "XGB_MULTIHEAD_NO_GO.json"
        if no_go_path.exists():
            no_go_path.unlink()
    else:
        marker_content["issues"] = issues
        marker_path = bundle_dir / "XGB_MULTIHEAD_NO_GO.json"
        tmp_path = bundle_dir / ".XGB_MULTIHEAD_NO_GO.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(marker_content, f, indent=2)
        tmp_path.rename(marker_path)
        print(f"\n❌ NO-GO MARKER written: {marker_path}")
        
        go_path = bundle_dir / "XGB_MULTIHEAD_GO_MARKER.json"
        if go_path.exists():
            go_path.unlink()
    
    # Convert numpy types to Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Write summary
    summary = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "model_sha256": model_sha,
        "sessions": model.sessions,
        "reference_year": ref_year,
        "max_ks": max_ks,
        "max_psi": max_psi,
        "max_ks_by_head": max_ks_by_head,
        "max_psi_by_head": max_psi_by_head,
        "max_clip_rate_pct": max_clip,
        "verdict": "GO" if not issues else "NO-GO",
        "issues": issues,
        "year_summary": year_results,
    }
    
    summary_path = output_dir / "EVAL_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    print(f"\nWrote: {summary_path}")
    
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
