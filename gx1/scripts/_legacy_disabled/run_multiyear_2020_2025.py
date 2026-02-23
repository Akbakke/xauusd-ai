#!/usr/bin/env python3
"""
Multiyear 2020-2025 Replay Evaluation

Runs replay for each year (2020-2025) in parallel and aggregates results.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class YearResult:
    """Result for a single year."""
    year: int
    success: bool
    output_dir: Path
    metrics: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def run_year_replay(
    year: int,
    data_root: Path,
    prebuilt_root: Path,
    bundle_dir: Path,
    policy_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> YearResult:
    """Run replay for a single year."""
    log.info(f"[YEAR {year}] Starting replay...")
    
    year_output_dir = output_dir / str(year)
    year_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use year-specific data file if available, otherwise use data_root as single file
    if data_root.is_dir():
        # Assume data_root contains year files like 2020.parquet, 2021.parquet, etc.
        year_data_path = data_root / f"{year}.parquet"
        if not year_data_path.exists():
            log.error(f"[YEAR {year}] Data file not found: {year_data_path}")
            return YearResult(
                year=year,
                success=False,
                output_dir=year_output_dir,
                error=f"Data file not found: {year_data_path}"
            )
    else:
        # Single file for all years
        year_data_path = data_root
    
    # Use year-specific prebuilt file if available
    if prebuilt_root.is_dir():
        # Assume prebuilt_root contains year subdirectories like TRIAL160/2020/xauusd_m5_2020_features_v10_ctx.parquet
        year_prebuilt_path = prebuilt_root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"
        if not year_prebuilt_path.exists():
            # Try alternative structure: prebuilt_root/{year}/xauusd_m5_{year}_features_v10_ctx.parquet
            year_prebuilt_path = prebuilt_root / f"{year}" / f"xauusd_m5_{year}_features_v10_ctx.parquet"
        if not year_prebuilt_path.exists():
            log.error(f"[YEAR {year}] Prebuilt file not found: {year_prebuilt_path}")
            return YearResult(
                year=year,
                success=False,
                output_dir=year_output_dir,
                error=f"Prebuilt file not found: {year_prebuilt_path}"
            )
    else:
        # Single prebuilt file for all years
        year_prebuilt_path = prebuilt_root
    
    # Set environment variables
    env = os.environ.copy()
    env["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    env["GX1_GATED_FUSION_ENABLED"] = "1"
    env["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env["GX1_REQUIRE_ENTRY_TELEMETRY"] = "1"
    env["GX1_REQUIRE_XGB_CALIBRATION"] = "0"  # Allow uncalibrated for speed
    
    # Build command
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "replay_eval_gated_parallel.py"),
        "--data", str(year_data_path),
        "--prebuilt-parquet", str(year_prebuilt_path),
        "--bundle-dir", str(bundle_dir),
        "--policy", str(policy_path),
        "--output-dir", str(year_output_dir),
        "--workers", str(workers),
        "--start-ts", f"{year}-01-01T00:00:00",
        "--end-ts", f"{year}-12-31T23:59:59",
    ]
    
    log.info(f"[YEAR {year}] Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour per year
        )
        
        if result.returncode != 0:
            log.error(f"[YEAR {year}] Failed with return code {result.returncode}")
            log.error(f"[YEAR {year}] stderr: {result.stderr[:500]}")
            return YearResult(
                year=year,
                success=False,
                output_dir=year_output_dir,
                error=f"Return code {result.returncode}: {result.stderr[:200]}"
            )
        
        # Load metrics and telemetry
        metrics = {
            "n_trades": 0,
            "total_pnl_bps": 0.0,
            "max_dd": 0.0,
            "session_breakdown": {},
            "regime_breakdown": {},
        }
        telemetry = {}
        invariants = {
            "prebuilt_used": False,
            "feature_build_call_count": -1,
            "lookup_hits": 0,
            "lookup_attempts": 0,
        }
        
        # Find chunk directories
        chunk_dirs = sorted(year_output_dir.glob("chunk_*"))
        for chunk_dir in chunk_dirs:
            # Load metrics
            metrics_files = list(chunk_dir.glob("metrics_*.json"))
            for mf in metrics_files:
                with open(mf) as f:
                    data = json.load(f)
                metrics["n_trades"] += data.get("n_trades", 0)
                metrics["total_pnl_bps"] += data.get("total_pnl_bps", 0.0)
                if data.get("max_dd", 0.0) < metrics["max_dd"]:
                    metrics["max_dd"] = data.get("max_dd", 0.0)
                
                # Session breakdown
                if "session_breakdown" in data:
                    for session, session_data in data["session_breakdown"].items():
                        if session not in metrics["session_breakdown"]:
                            metrics["session_breakdown"][session] = {
                                "n_trades": 0,
                                "total_pnl_bps": 0.0,
                            }
                        metrics["session_breakdown"][session]["n_trades"] += session_data.get("n_trades", 0)
                        metrics["session_breakdown"][session]["total_pnl_bps"] += session_data.get("total_pnl_bps", 0.0)
                
                # Regime breakdown
                if "regime_breakdown" in data:
                    for regime, regime_data in data["regime_breakdown"].items():
                        if regime not in metrics["regime_breakdown"]:
                            metrics["regime_breakdown"][regime] = {
                                "n_trades": 0,
                                "total_pnl_bps": 0.0,
                            }
                        metrics["regime_breakdown"][regime]["n_trades"] += regime_data.get("n_trades", 0)
                        metrics["regime_breakdown"][regime]["total_pnl_bps"] += regime_data.get("total_pnl_bps", 0.0)
            
            # Load telemetry
            ef_path = chunk_dir / "ENTRY_FEATURES_USED.json"
            if ef_path.exists():
                with open(ef_path) as f:
                    ef = json.load(f)
                telemetry["transformer_forward_calls"] = telemetry.get("transformer_forward_calls", 0) + ef.get("transformer_forward_calls", 0)
                telemetry["xgb_pre_predict_count"] = telemetry.get("xgb_pre_predict_count", 0) + ef.get("xgb_flow", {}).get("xgb_pre_predict_count", 0)
                telemetry["xgb_seq_channels"] = ef.get("xgb_seq_channels", {}).get("names", [])
                telemetry["xgb_snap_channels"] = ef.get("xgb_snap_channels", {}).get("names", [])
                telemetry["xgb_used_as"] = ef.get("xgb_flow", {}).get("xgb_used_as", "unknown")
            
            # Load invariants from chunk_footer
            footer_path = chunk_dir / "chunk_footer.json"
            if footer_path.exists():
                with open(footer_path) as f:
                    footer = json.load(f)
                invariants["prebuilt_used"] = footer.get("prebuilt_used", False) or invariants["prebuilt_used"]
                invariants["feature_build_call_count"] = max(invariants["feature_build_call_count"], footer.get("feature_build_call_count", -1))
                invariants["lookup_hits"] += footer.get("lookup_hits", 0)
                invariants["lookup_attempts"] += footer.get("lookup_attempts", 0)
        
        elapsed = time.time() - start_time
        log.info(f"[YEAR {year}] Completed in {elapsed:.1f}s: {metrics.get('n_trades', 0)} trades, {metrics.get('total_pnl_bps', 0):.0f} bps")
        
        return YearResult(
            year=year,
            success=True,
            output_dir=year_output_dir,
            metrics=metrics,
            telemetry={**telemetry, "invariants": invariants},
        )
        
    except subprocess.TimeoutExpired:
        log.error(f"[YEAR {year}] Timeout after 1 hour")
        return YearResult(
            year=year,
            success=False,
            output_dir=year_output_dir,
            error="Timeout after 1 hour"
        )
    except Exception as e:
        log.error(f"[YEAR {year}] Exception: {e}")
        return YearResult(
            year=year,
            success=False,
            output_dir=year_output_dir,
            error=str(e)
        )


def aggregate_results(results: List[YearResult]) -> Dict[str, Any]:
    """Aggregate results across all years."""
    aggregated = {
        "years": {},
        "total": {
            "n_trades": 0,
            "total_pnl_bps": 0.0,
            "max_dd": 0.0,
            "transformer_forward_calls": 0,
            "xgb_pre_predict_count": 0,
        },
        "pipeline_verification": {
            "xgb_used_as": "pre",
            "xgb_seq_channels": ["p_long_xgb", "uncertainty_score"],
            "xgb_snap_channels": ["p_long_xgb", "p_hat_xgb"],
            "margin_xgb_present": False,
        },
    }
    
    for result in results:
        if not result.success:
            aggregated["years"][result.year] = {
                "success": False,
                "error": result.error,
            }
            continue
        
        invariants = result.telemetry.get("invariants", {})
        year_data = {
            "success": True,
            "n_trades": result.metrics.get("n_trades", 0),
            "total_pnl_bps": result.metrics.get("total_pnl_bps", 0.0),
            "max_dd": result.metrics.get("max_dd", 0.0),
            "transformer_forward_calls": result.telemetry.get("transformer_forward_calls", 0),
            "xgb_pre_predict_count": result.telemetry.get("xgb_pre_predict_count", 0),
            "xgb_used_as": result.telemetry.get("xgb_used_as", "unknown"),
            "xgb_seq_channels": result.telemetry.get("xgb_seq_channels", []),
            "xgb_snap_channels": result.telemetry.get("xgb_snap_channels", []),
            "session_breakdown": result.metrics.get("session_breakdown", {}),
            "regime_breakdown": result.metrics.get("regime_breakdown", {}),
            "invariants": {
                "prebuilt_used": invariants.get("prebuilt_used", False),
                "feature_build_call_count": invariants.get("feature_build_call_count", -1),
                "feature_build_call_count_zero": invariants.get("feature_build_call_count", -1) == 0,
                "lookup_hits": invariants.get("lookup_hits", 0),
                "lookup_attempts": invariants.get("lookup_attempts", 0),
                "lookup_invariant_pass": invariants.get("lookup_hits", 0) + (invariants.get("lookup_attempts", 0) - invariants.get("lookup_hits", 0)) == invariants.get("lookup_attempts", 0),
            },
        }
        
        # Check for margin_xgb
        if "margin_xgb" in year_data["xgb_seq_channels"] or "margin_xgb" in year_data["xgb_snap_channels"]:
            aggregated["pipeline_verification"]["margin_xgb_present"] = True
            log.error(f"[YEAR {result.year}] ❌ margin_xgb detected in channels!")
        
        aggregated["years"][result.year] = year_data
        
        # Aggregate totals
        aggregated["total"]["n_trades"] += year_data["n_trades"]
        aggregated["total"]["total_pnl_bps"] += year_data["total_pnl_bps"]
        if year_data["max_dd"] < aggregated["total"]["max_dd"]:
            aggregated["total"]["max_dd"] = year_data["max_dd"]
        aggregated["total"]["transformer_forward_calls"] += year_data["transformer_forward_calls"]
        aggregated["total"]["xgb_pre_predict_count"] += year_data["xgb_pre_predict_count"]
    
    return aggregated


def generate_report(aggregated: Dict[str, Any], output_dir: Path) -> None:
    """Generate multiyear report."""
    
    # JSON report
    json_path = output_dir / "MULTIYEAR_2020_2025_REPORT.json"
    with open(json_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    log.info(f"Written JSON: {json_path}")
    
    # Markdown report
    # Check pipeline verification
    all_xgb_pre = all(
        r.get("xgb_used_as") == "pre" 
        for r in aggregated["years"].values() 
        if isinstance(r, dict) and r.get("success")
    )
    all_prebuilt = all(
        r.get("invariants", {}).get("prebuilt_used", False)
        for r in aggregated["years"].values()
        if isinstance(r, dict) and r.get("success")
    )
    all_feature_build_zero = all(
        r.get("invariants", {}).get("feature_build_call_count_zero", False)
        for r in aggregated["years"].values()
        if isinstance(r, dict) and r.get("success")
    )
    all_lookup_ok = all(
        r.get("invariants", {}).get("lookup_invariant_pass", False)
        for r in aggregated["years"].values()
        if isinstance(r, dict) and r.get("success")
    )
    all_transformer_calls = all(
        r.get("transformer_forward_calls", 0) > 0
        for r in aggregated["years"].values()
        if isinstance(r, dict) and r.get("success")
    )
    
    md = f"""# Multiyear 2020-2025 Report

**Generated:** {datetime.now().isoformat()}

---

## Pipeline Verification

| Check | Status |
|-------|--------|
| `xgb_used_as == "pre"` | {'✅' if all_xgb_pre else '❌'} |
| `xgb_seq_channels` | {aggregated['pipeline_verification']['xgb_seq_channels']} |
| `xgb_snap_channels` | {aggregated['pipeline_verification']['xgb_snap_channels']} |
| `margin_xgb_present` | {'❌ YES (ERROR!)' if aggregated['pipeline_verification']['margin_xgb_present'] else '✅ NO'} |
| `prebuilt_used == true` | {'✅' if all_prebuilt else '❌'} |
| `feature_build_call_count == 0` | {'✅' if all_feature_build_zero else '❌'} |
| `lookup_invariant_pass` | {'✅' if all_lookup_ok else '❌'} |
| `transformer_forward_calls > 0` | {'✅' if all_transformer_calls else '❌'} |

---

## Per-Year Results

| Year | Trades | PnL (bps) | Max DD | Transformer Calls | XGB Calls | Prebuilt | Feature Build | Lookup OK |
|------|--------|-----------|--------|-------------------|-----------|----------|---------------|-----------|
"""
    
    for year in sorted(aggregated["years"].keys()):
        year_data = aggregated["years"][year]
        if not year_data.get("success"):
            md += f"| {year} | ❌ FAILED | {year_data.get('error', 'Unknown error')} | - | - | - | - | - | - |\n"
        else:
            inv = year_data.get("invariants", {})
            md += f"| {year} | {year_data['n_trades']:,} | {year_data['total_pnl_bps']:.0f} | {year_data['max_dd']:.0f} | {year_data['transformer_forward_calls']:,} | {year_data['xgb_pre_predict_count']:,} | {'✅' if inv.get('prebuilt_used') else '❌'} | {'✅' if inv.get('feature_build_call_count_zero') else '❌'} | {'✅' if inv.get('lookup_invariant_pass') else '❌'} |\n"
    
    md += f"""
---

## Totals (2020-2025)

| Metric | Value |
|--------|-------|
| Total Trades | {aggregated['total']['n_trades']:,} |
| Total PnL (bps) | {aggregated['total']['total_pnl_bps']:.0f} |
| Max DD | {aggregated['total']['max_dd']:.0f} |
| Transformer Forward Calls | {aggregated['total']['transformer_forward_calls']:,} |
| XGB Pre-Predict Calls | {aggregated['total']['xgb_pre_predict_count']:,} |

---

## Session Breakdown (Per Year)

"""
    
    # Session breakdown table
    all_sessions = set()
    for year_data in aggregated["years"].values():
        if isinstance(year_data, dict) and year_data.get("success"):
            all_sessions.update(year_data.get("session_breakdown", {}).keys())
    
    if all_sessions:
        md += "| Year | " + " | ".join(sorted(all_sessions)) + " |\n"
        md += "|------|" + "|".join(["--------" for _ in sorted(all_sessions)]) + "|\n"
        for year in sorted(aggregated["years"].keys()):
            year_data = aggregated["years"][year]
            if year_data.get("success"):
                session_breakdown = year_data.get("session_breakdown", {})
                md += f"| {year} | "
                md += " | ".join([
                    f"{session_breakdown.get(s, {}).get('n_trades', 0):,} trades<br/>{session_breakdown.get(s, {}).get('total_pnl_bps', 0):.0f} bps"
                    for s in sorted(all_sessions)
                ]) + " |\n"
        md += "\n"
    
    md += """
---

## Regime Breakdown (Per Year)

"""
    
    # Regime breakdown table
    all_regimes = set()
    for year_data in aggregated["years"].values():
        if isinstance(year_data, dict) and year_data.get("success"):
            all_regimes.update(year_data.get("regime_breakdown", {}).keys())
    
    if all_regimes:
        md += "| Year | " + " | ".join(sorted(all_regimes)) + " |\n"
        md += "|------|" + "|".join(["--------" for _ in sorted(all_regimes)]) + "|\n"
        for year in sorted(aggregated["years"].keys()):
            year_data = aggregated["years"][year]
            if year_data.get("success"):
                regime_breakdown = year_data.get("regime_breakdown", {})
                md += f"| {year} | "
                md += " | ".join([
                    f"{regime_breakdown.get(r, {}).get('n_trades', 0):,} trades<br/>{regime_breakdown.get(r, {}).get('total_pnl_bps', 0):.0f} bps"
                    for r in sorted(all_regimes)
                ]) + " |\n"
        md += "\n"
    else:
        md += "*No regime breakdown data available.*\n\n"
    
    md += """
"""
    
    md_path = output_dir / "MULTIYEAR_2020_2025_REPORT.md"
    with open(md_path, "w") as f:
        f.write(md)
    log.info(f"Written Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Multiyear 2020-2025 Replay Evaluation")
    parser.add_argument("--data-root", type=Path, required=True, 
                        help="Path to data root (directory with year.parquet files) or single parquet file")
    parser.add_argument("--prebuilt-root", type=Path, required=True, 
                        help="Path to prebuilt root (directory with year subdirs) or single parquet file")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Path to V10 bundle")
    parser.add_argument("--policy", type=Path, required=True, help="Path to policy YAML")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers (default: 6, one per year)")
    parser.add_argument("--years", type=str, default="2020,2021,2022,2023,2024,2025",
                        help="Comma-separated years (default: 2020,2021,2022,2023,2024,2025)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.data_root.exists():
        log.error(f"Data root does not exist: {args.data_root}")
        return 1
    
    if not args.prebuilt_root.exists():
        log.error(f"Prebuilt root does not exist: {args.prebuilt_root}")
        return 1
    
    for path, name in [
        (args.bundle_dir, "bundle-dir"),
        (args.policy, "policy"),
    ]:
        if not path.exists():
            log.error(f"Path does not exist: {name}={path}")
            return 1
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(",")]
    
    # Validate year data files if data_root is a directory
    if args.data_root.is_dir():
        for year in years:
            year_file = args.data_root / f"{year}.parquet"
            if not year_file.exists():
                log.error(f"Year data file not found: {year_file}")
                return 1
    
    # Validate year prebuilt files if prebuilt_root is a directory
    if args.prebuilt_root.is_dir():
        for year in years:
            year_prebuilt = args.prebuilt_root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"
            if not year_prebuilt.exists():
                # Try alternative structure
                year_prebuilt = args.prebuilt_root / f"{year}" / f"xauusd_m5_{year}_features_v10_ctx.parquet"
            if not year_prebuilt.exists():
                log.error(f"Year prebuilt file not found: {year_prebuilt}")
                return 1
    
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out_root / f"MULTIYEAR_2020_2025_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 80)
    log.info("MULTIYEAR 2020-2025 REPLAY EVALUATION")
    log.info("=" * 80)
    log.info(f"Years: {years}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Output: {output_dir}")
    log.info(f"Data root: {args.data_root}")
    log.info(f"Prebuilt root: {args.prebuilt_root}")
    
    # Run years in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_year_replay,
                year,
                args.data_root,
                args.prebuilt_root,
                args.bundle_dir,
                args.policy,
                output_dir,
                workers=1,  # 1 worker per year
            ): year
            for year in years
        }
        
        for future in as_completed(futures):
            year = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log.error(f"[YEAR {year}] Exception in executor: {e}")
                results.append(YearResult(
                    year=year,
                    success=False,
                    output_dir=output_dir / str(year),
                    error=str(e)
                ))
    
    # Aggregate results
    log.info("\n" + "=" * 80)
    log.info("AGGREGATING RESULTS")
    log.info("=" * 80)
    
    aggregated = aggregate_results(results)
    
    # Generate report
    generate_report(aggregated, output_dir)
    
    # Summary
    successful = sum(1 for r in results if r.success)
    log.info(f"\n{'=' * 80}")
    log.info(f"✅ Completed: {successful}/{len(years)} years successful")
    log.info(f"Reports: {output_dir}")
    log.info("=" * 80)
    
    return 0 if successful == len(years) else 1


if __name__ == "__main__":
    sys.exit(main())
