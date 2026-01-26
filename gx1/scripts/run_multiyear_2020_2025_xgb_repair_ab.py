#!/usr/bin/env python3
"""
Multiyear 2020-2025 XGB Repair A/B Verification

Runs controlled A/B test:
  ARM_A: Current truth pipeline (no calibration)
  ARM_B: Calibrated + normalized XGB outputs

Usage:
    python gx1/scripts/run_multiyear_2020_2025_xgb_repair_ab.py \
        --years 2020 2021 2022 2023 2024 2025 \
        --calibrator-path ../GX1_DATA/models/calibrators/xgb_calibrator_platt_*.pkl \
        --clipper-path ../GX1_DATA/models/calibrators/xgb_clipper_*.pkl
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Tuple
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

# Default paths
DEFAULT_GX1_DATA = Path("/Users/andrekildalbakke/Desktop/GX1_DATA")
DEFAULT_PREBUILT_ROOT = DEFAULT_GX1_DATA / "data" / "prebuilt" / "TRIAL160"
DEFAULT_BUNDLE = DEFAULT_GX1_DATA / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
DEFAULT_POLICY = WORKSPACE_ROOT / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
DATA_FILE = DEFAULT_GX1_DATA / "data" / "data" / "entry_v9" / "full_2020_2025.parquet"


@dataclass
class ArmResult:
    """Results for a single arm-year combination."""
    arm: str
    year: int
    success: bool = False
    error: Optional[str] = None
    
    # Trading metrics
    trades: int = 0
    pnl_bps: float = 0.0
    max_dd_bps: float = 0.0
    
    # Pipeline metrics
    transformer_forward_calls: int = 0
    xgb_pre_predict_count: int = 0
    conversion_rate: float = 0.0
    
    # XGB output stats (raw)
    p_long_raw_mean: float = 0.0
    p_long_raw_std: float = 0.0
    p_long_raw_p95: float = 0.0
    p_long_raw_p99: float = 0.0
    
    # XGB output stats (calibrated)
    p_long_cal_mean: float = 0.0
    p_long_cal_std: float = 0.0
    p_long_cal_p95: float = 0.0
    p_long_cal_p99: float = 0.0
    
    # Uncertainty stats
    uncertainty_mean: float = 0.0
    uncertainty_std: float = 0.0
    
    # Calibration applied
    calibration_applied: bool = False
    calibrator_sha: str = ""
    clipper_sha: str = ""
    clipper_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Invariants
    prebuilt_used: bool = False
    xgb_used_as: str = ""


def get_prebuilt_path(year: int, prebuilt_root: Path) -> Path:
    """Get prebuilt path for a year."""
    return prebuilt_root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"


def run_arm_year(
    arm: str,
    year: int,
    output_dir: Path,
    policy: Path,
    bundle_dir: Path,
    prebuilt_root: Path,
    calibrator_path: Optional[Path],
    clipper_path: Optional[Path],
    workers: int,
) -> ArmResult:
    """Run replay for a single arm-year."""
    result = ArmResult(arm=arm, year=year)
    
    prebuilt_path = get_prebuilt_path(year, prebuilt_root)
    if not prebuilt_path.exists():
        result.error = f"Prebuilt not found: {prebuilt_path}"
        return result
    
    # Create output directory
    year_output = output_dir / arm / str(year)
    year_output.mkdir(parents=True, exist_ok=True)
    
    # Build environment
    env = os.environ.copy()
    env["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env["GX1_REQUIRE_ENTRY_TELEMETRY"] = "1"
    env["GX1_GATED_FUSION_ENABLED"] = "1"
    env["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    env["GX1_FEATURE_BUILD_DISABLED"] = "1"
    
    # ARM_B specific: calibration
    if arm == "ARM_B" and calibrator_path:
        env["GX1_XGB_CALIBRATOR_PATH"] = str(calibrator_path)
    if arm == "ARM_B" and clipper_path:
        env["GX1_XGB_CLIPPER_PATH"] = str(clipper_path)
    
    # Build command
    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "gx1" / "scripts" / "replay_eval_gated_parallel.py"),
        "--data", str(DATA_FILE),
        "--prebuilt-parquet", str(prebuilt_path),
        "--bundle-dir", str(bundle_dir),
        "--policy", str(policy),
        "--output-dir", str(year_output),
        "--workers", str(min(workers, 4)),
        "--start-ts", f"{year}-01-01T00:00:00",
        "--end-ts", f"{year}-12-31T23:59:59",
    ]
    
    print(f"  [{arm}] Running year {year}...")
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        
        if proc.returncode != 0:
            result.error = f"Replay failed: {proc.stderr[-500:] if proc.stderr else 'unknown'}"
            return result
        
    except subprocess.TimeoutExpired:
        result.error = "Timeout after 30 minutes"
        return result
    except Exception as e:
        result.error = str(e)
        return result
    
    # Parse results
    chunk_footer = year_output / "chunk_0" / "chunk_footer.json"
    if not chunk_footer.exists():
        for chunk_dir in year_output.glob("chunk_*"):
            footer = chunk_dir / "chunk_footer.json"
            if footer.exists():
                chunk_footer = footer
                break
    
    if not chunk_footer.exists():
        result.error = "No chunk_footer.json found"
        return result
    
    try:
        with open(chunk_footer) as f:
            footer = json.load(f)
    except Exception as e:
        result.error = f"Failed to parse chunk_footer.json: {e}"
        return result
    
    result.success = True
    result.trades = footer.get("n_trades_closed", 0)
    result.pnl_bps = footer.get("total_pnl_bps", 0.0)
    result.max_dd_bps = footer.get("max_dd_bps", 0.0)
    result.transformer_forward_calls = footer.get("transformer_forward_calls", 0)
    result.xgb_pre_predict_count = footer.get("xgb_pre_predict_count", 0)
    result.prebuilt_used = footer.get("prebuilt_used", False)
    result.xgb_used_as = footer.get("xgb_used_as", "")
    
    if result.transformer_forward_calls > 0:
        result.conversion_rate = result.trades / result.transformer_forward_calls
    
    # Try to get XGB output stats from telemetry
    master_path = year_output / "chunk_0" / "ENTRY_FEATURES_USED_MASTER.json"
    if master_path.exists():
        try:
            with open(master_path) as f:
                master = json.load(f)
            
            xgb_flow = master.get("xgb_flow", {})
            result.p_long_raw_mean = xgb_flow.get("p_long_raw_mean", xgb_flow.get("p_long_mean", 0.0))
            result.p_long_raw_std = xgb_flow.get("p_long_raw_std", xgb_flow.get("p_long_std", 0.0))
            result.p_long_raw_p95 = xgb_flow.get("p_long_raw_p95", 0.0)
            result.p_long_raw_p99 = xgb_flow.get("p_long_raw_p99", 0.0)
            
            result.p_long_cal_mean = xgb_flow.get("p_long_cal_mean", 0.0)
            result.p_long_cal_std = xgb_flow.get("p_long_cal_std", 0.0)
            result.p_long_cal_p95 = xgb_flow.get("p_long_cal_p95", 0.0)
            result.p_long_cal_p99 = xgb_flow.get("p_long_cal_p99", 0.0)
            
            result.uncertainty_mean = xgb_flow.get("uncertainty_mean", 0.0)
            result.uncertainty_std = xgb_flow.get("uncertainty_std", 0.0)
            
            result.calibration_applied = xgb_flow.get("calibration_applied", False)
            result.calibrator_sha = xgb_flow.get("calibrator_sha", "")
            result.clipper_sha = xgb_flow.get("clipper_sha", "")
            result.clipper_bounds = xgb_flow.get("clipper_bounds", {})
            
        except Exception:
            pass
    
    return result


def compute_ks_statistic(arm_a_values: List[float], arm_b_values: List[float]) -> float:
    """Compute KS statistic between two arms."""
    try:
        from scipy import stats
        if len(arm_a_values) < 10 or len(arm_b_values) < 10:
            return 0.0
        ks_stat, _ = stats.ks_2samp(arm_a_values, arm_b_values)
        return float(ks_stat)
    except Exception:
        return 0.0


def generate_reports(
    arm_a_results: List[ArmResult],
    arm_b_results: List[ArmResult],
    output_dir: Path,
    calibrator_path: Optional[Path],
    clipper_path: Optional[Path],
) -> None:
    """Generate comparison reports."""
    
    # MULTIYEAR_SUMMARY.json
    summary = {
        "arm_a": {
            "name": "current_truth",
            "calibration": False,
            "years": {},
            "total_trades": sum(r.trades for r in arm_a_results),
            "total_pnl_bps": sum(r.pnl_bps for r in arm_a_results),
        },
        "arm_b": {
            "name": "calibrated_normalized",
            "calibration": True,
            "calibrator_path": str(calibrator_path) if calibrator_path else None,
            "clipper_path": str(clipper_path) if clipper_path else None,
            "years": {},
            "total_trades": sum(r.trades for r in arm_b_results),
            "total_pnl_bps": sum(r.pnl_bps for r in arm_b_results),
        },
    }
    
    for r in arm_a_results:
        summary["arm_a"]["years"][r.year] = asdict(r)
    for r in arm_b_results:
        summary["arm_b"]["years"][r.year] = asdict(r)
    
    summary_path = output_dir / "MULTIYEAR_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {summary_path}")
    
    # ARM_COMPARE.md
    lines = [
        "# XGB Repair A/B Comparison",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Arms",
        "",
        "| Arm | Description | Calibration |",
        "|-----|-------------|-------------|",
        "| ARM_A | Current truth pipeline | ❌ |",
        "| ARM_B | Calibrated + normalized | ✅ |",
        "",
        "---",
        "",
        "## Per-Year Comparison",
        "",
        "| Year | ARM_A Trades | ARM_B Trades | Δ Trades | ARM_A PnL | ARM_B PnL | Δ PnL |",
        "|------|--------------|--------------|----------|-----------|-----------|-------|",
    ]
    
    total_delta_trades = 0
    total_delta_pnl = 0.0
    
    for a_result in arm_a_results:
        b_result = next((r for r in arm_b_results if r.year == a_result.year), None)
        if not b_result:
            continue
        
        delta_trades = b_result.trades - a_result.trades
        delta_pnl = b_result.pnl_bps - a_result.pnl_bps
        
        total_delta_trades += delta_trades
        total_delta_pnl += delta_pnl
        
        lines.append(
            f"| {a_result.year} | {a_result.trades:,} | {b_result.trades:,} | {delta_trades:+d} | "
            f"{a_result.pnl_bps:.0f} | {b_result.pnl_bps:.0f} | {delta_pnl:+.0f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Totals",
        "",
        f"- **ARM_A Total Trades:** {sum(r.trades for r in arm_a_results):,}",
        f"- **ARM_B Total Trades:** {sum(r.trades for r in arm_b_results):,}",
        f"- **Δ Trades:** {total_delta_trades:+d}",
        "",
        f"- **ARM_A Total PnL:** {sum(r.pnl_bps for r in arm_a_results):.0f} bps",
        f"- **ARM_B Total PnL:** {sum(r.pnl_bps for r in arm_b_results):.0f} bps",
        f"- **Δ PnL:** {total_delta_pnl:+.0f} bps",
        "",
        "---",
        "",
        "## GO Criteria",
        "",
    ])
    
    # Check GO criteria
    arm_b_2025 = next((r for r in arm_b_results if r.year == 2025), None)
    arm_a_2025 = next((r for r in arm_a_results if r.year == 2025), None)
    
    go_criteria = []
    
    # 2025 not collapsed (>80% of baseline PnL)
    if arm_a_2025 and arm_b_2025:
        if arm_a_2025.pnl_bps > 0:
            pnl_ratio = arm_b_2025.pnl_bps / arm_a_2025.pnl_bps
            if pnl_ratio >= 0.8:
                go_criteria.append(f"✅ 2025 PnL retained: {pnl_ratio*100:.1f}% of baseline")
            else:
                go_criteria.append(f"❌ 2025 PnL collapsed: {pnl_ratio*100:.1f}% of baseline (need >80%)")
    
    # 2020-2021 improved
    arm_a_early = sum(r.pnl_bps for r in arm_a_results if r.year in [2020, 2021])
    arm_b_early = sum(r.pnl_bps for r in arm_b_results if r.year in [2020, 2021])
    
    if arm_b_early > arm_a_early:
        go_criteria.append(f"✅ 2020-2021 improved: {arm_a_early:.0f} → {arm_b_early:.0f} bps")
    else:
        go_criteria.append(f"⚠️ 2020-2021 not improved: {arm_a_early:.0f} → {arm_b_early:.0f} bps")
    
    for criterion in go_criteria:
        lines.append(f"- {criterion}")
    
    lines.extend([
        "",
        "---",
        "",
        "## XGB Output Stats (raw vs calibrated)",
        "",
        "### ARM_A (no calibration)",
        "",
        "| Year | p_long_mean | p_long_std | p_long_p95 | p_long_p99 | uncertainty_mean |",
        "|------|-------------|------------|------------|------------|------------------|",
    ])
    
    for r in arm_a_results:
        lines.append(
            f"| {r.year} | {r.p_long_raw_mean:.4f} | {r.p_long_raw_std:.4f} | "
            f"{r.p_long_raw_p95:.4f} | {r.p_long_raw_p99:.4f} | {r.uncertainty_mean:.4f} |"
        )
    
    lines.extend([
        "",
        "### ARM_B (calibrated + clipped)",
        "",
        "| Year | p_long_raw_mean | p_long_cal_mean | p_long_cal_std | p_long_cal_p95 | p_long_cal_p99 |",
        "|------|-----------------|-----------------|----------------|----------------|----------------|",
    ])
    
    for r in arm_b_results:
        lines.append(
            f"| {r.year} | {r.p_long_raw_mean:.4f} | {r.p_long_cal_mean:.4f} | "
            f"{r.p_long_cal_std:.4f} | {r.p_long_cal_p95:.4f} | {r.p_long_cal_p99:.4f} |"
        )
    
    # Clipper bounds
    if arm_b_results and arm_b_results[0].clipper_bounds:
        lines.extend([
            "",
            "### Clipper Bounds Used",
            "",
            "| Channel | Lower (p1) | Upper (p99) |",
            "|---------|------------|-------------|",
        ])
        for channel, bounds in arm_b_results[0].clipper_bounds.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lines.append(f"| {channel} | {bounds[0]:.4f} | {bounds[1]:.4f} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Conversion Rates (trades / forward_calls)",
        "",
        "| Year | ARM_A Conv | ARM_B Conv | Δ Conv |",
        "|------|------------|------------|--------|",
    ])
    
    for a_result in arm_a_results:
        b_result = next((r for r in arm_b_results if r.year == a_result.year), None)
        if b_result:
            delta = b_result.conversion_rate - a_result.conversion_rate
            lines.append(
                f"| {a_result.year} | {a_result.conversion_rate:.6f} | "
                f"{b_result.conversion_rate:.6f} | {delta:+.6f} |"
            )
    
    lines.append("")
    
    compare_path = output_dir / "ARM_COMPARE.md"
    with open(compare_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {compare_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multiyear 2020-2025 XGB Repair A/B Verification"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to run"
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
        help="Bundle directory"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="Policy YAML"
    )
    parser.add_argument(
        "--calibrator-path",
        type=Path,
        default=None,
        help="Path to calibrator .pkl for ARM_B"
    )
    parser.add_argument(
        "--clipper-path",
        type=Path,
        default=None,
        help="Path to clipper .pkl for ARM_B"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Workers per year"
    )
    parser.add_argument(
        "--arm-a-only",
        action="store_true",
        help="Run only ARM_A"
    )
    parser.add_argument(
        "--arm-b-only",
        action="store_true",
        help="Run only ARM_B"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = DEFAULT_GX1_DATA / "reports" / "replay_eval" / f"XGB_REPAIR_AB_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_arm_a = not args.arm_b_only
    run_arm_b = not args.arm_a_only
    
    print("=" * 60)
    print("XGB REPAIR A/B VERIFICATION")
    print("=" * 60)
    print(f"Years: {args.years}")
    print(f"Output: {output_dir}")
    print(f"Calibrator: {args.calibrator_path}")
    print(f"Clipper: {args.clipper_path}")
    print()
    
    arm_a_results = []
    arm_b_results = []
    
    # Run ARM_A
    if run_arm_a:
        print("=" * 40)
        print("ARM_A: Current truth (no calibration)")
        print("=" * 40)
        
        for year in args.years:
            result = run_arm_year(
                arm="ARM_A",
                year=year,
                output_dir=output_dir,
                policy=args.policy,
                bundle_dir=args.bundle_dir,
                prebuilt_root=args.prebuilt_root,
                calibrator_path=None,
                clipper_path=None,
                workers=args.workers,
            )
            arm_a_results.append(result)
            
            if result.success:
                print(f"  ✅ {year}: {result.trades} trades, {result.pnl_bps:.0f} bps")
            else:
                print(f"  ❌ {year}: {result.error}")
    
    # Run ARM_B
    if run_arm_b:
        print()
        print("=" * 40)
        print("ARM_B: Calibrated + normalized")
        print("=" * 40)
        
        for year in args.years:
            result = run_arm_year(
                arm="ARM_B",
                year=year,
                output_dir=output_dir,
                policy=args.policy,
                bundle_dir=args.bundle_dir,
                prebuilt_root=args.prebuilt_root,
                calibrator_path=args.calibrator_path,
                clipper_path=args.clipper_path,
                workers=args.workers,
            )
            arm_b_results.append(result)
            
            if result.success:
                print(f"  ✅ {year}: {result.trades} trades, {result.pnl_bps:.0f} bps")
            else:
                print(f"  ❌ {year}: {result.error}")
    
    # Generate reports
    print()
    print("=" * 40)
    print("GENERATING REPORTS")
    print("=" * 40)
    
    generate_reports(
        arm_a_results=arm_a_results,
        arm_b_results=arm_b_results,
        output_dir=output_dir,
        calibrator_path=args.calibrator_path,
        clipper_path=args.clipper_path,
    )
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if arm_a_results:
        print(f"ARM_A: {sum(r.trades for r in arm_a_results):,} trades, {sum(r.pnl_bps for r in arm_a_results):.0f} bps")
    if arm_b_results:
        print(f"ARM_B: {sum(r.trades for r in arm_b_results):,} trades, {sum(r.pnl_bps for r in arm_b_results):.0f} bps")
    
    print()
    print(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
