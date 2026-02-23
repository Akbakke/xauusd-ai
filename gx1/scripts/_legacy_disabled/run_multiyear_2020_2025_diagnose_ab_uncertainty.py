#!/usr/bin/env python3
"""
Multiyear 2020-2025 Diagnostic A/B: Uncertainty Score Ablation

Builds a blame-matrix: year × session × regime with full metrics.
Runs controlled A/B:
  ARM_A = current truth pipeline (margin_xgb removed)
  ARM_B = identical, but uncertainty_score removed from XGB→Transformer injection

All other parameters identical: policy, temp scaling, thresholds, gates, exits, PREBUILT-only.

Usage:
    python gx1/scripts/run_multiyear_2020_2025_diagnose_ab_uncertainty.py \
        --years 2020 2021 2022 2023 2024 2025 \
        --workers 6 \
        --arm-b-drop-uncertainty
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

# Default paths
DEFAULT_GX1_DATA = Path("/Users/andrekildalbakke/Desktop/GX1_DATA")
DEFAULT_OUTPUT_ROOT = DEFAULT_GX1_DATA / "reports" / "replay_eval"
DEFAULT_POLICY = WORKSPACE_ROOT / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
DEFAULT_BUNDLE = DEFAULT_GX1_DATA / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"

# Prebuilt root (directory with year subdirs)
DEFAULT_PREBUILT_ROOT = DEFAULT_GX1_DATA / "data" / "prebuilt" / "TRIAL160"

def get_prebuilt_path(year: int, prebuilt_root: Path) -> Path:
    """Get prebuilt path for a year."""
    # Try structure: prebuilt_root/2020/xauusd_m5_2020_features_v10_ctx.parquet
    path = prebuilt_root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"
    if path.exists():
        return path
    # Try alternative: prebuilt_root/2020/xauusd_m5_2020_features_v10_ctx.parquet (same)
    return path

# Data file (single file for all years, sliced by date range)
DATA_FILE = DEFAULT_GX1_DATA / "data" / "data" / "entry_v9" / "full_2020_2025.parquet"


@dataclass
class SessionBreakdown:
    """Metrics per session."""
    trades: int = 0
    pnl_bps: float = 0.0
    max_dd_bps: float = 0.0
    forward_calls: int = 0
    xgb_calls: int = 0
    model_attempts: int = 0
    eligibility_blocks: int = 0
    conversion_rate: float = 0.0


@dataclass
class YearResult:
    """Results for a single year."""
    year: int
    arm: str
    success: bool = False
    error: Optional[str] = None
    
    # Trading metrics
    trades: int = 0
    pnl_bps: float = 0.0
    max_dd_bps: float = 0.0
    
    # Pipeline metrics
    transformer_forward_calls: int = 0
    xgb_pre_predict_count: int = 0
    model_attempts: int = 0
    eligibility_blocks: int = 0
    conversion_rate: float = 0.0
    
    # XGB channels
    xgb_seq_channels: List[str] = field(default_factory=list)
    xgb_snap_channels: List[str] = field(default_factory=list)
    
    # Session breakdown
    session_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Regime breakdown
    regime_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Invariants
    prebuilt_used: bool = False
    feature_build_call_count: int = -1
    lookup_invariant_pass: bool = False
    xgb_used_as: str = ""
    margin_xgb_present: bool = True
    
    # Data contract
    ts_utc_ok: bool = False
    monotonic_ok: bool = False
    bars_total: int = 0
    session_bar_sums_ok: bool = False


@dataclass
class RunManifest:
    """Manifest written before run starts."""
    run_id: str
    policy_id: str
    bundle_sha: str
    replay_mode: str
    temp_scaling_effective: bool
    arms: List[str]
    years: List[int]
    cmdline: List[str]
    sys_executable: str
    timestamp: str
    xgb_channels_seq_arm_a: List[str]
    xgb_channels_snap_arm_a: List[str]
    xgb_channels_seq_arm_b: List[str]
    xgb_channels_snap_arm_b: List[str]


def compute_bundle_sha(bundle_dir: Path) -> str:
    """Compute SHA256 of bundle metadata."""
    metadata_path = bundle_dir / "bundle_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return "UNKNOWN"


def run_year_arm(
    year: int,
    arm: str,
    output_dir: Path,
    policy: Path,
    bundle_dir: Path,
    prebuilt_root: Path,
    workers: int,
    drop_uncertainty: bool
) -> YearResult:
    """Run replay for a single year and arm."""
    result = YearResult(year=year, arm=arm)
    
    # Check data and prebuilt exist
    data_path = DATA_FILE
    prebuilt_path = get_prebuilt_path(year, prebuilt_root)
    
    if not data_path or not data_path.exists():
        result.error = f"Data file not found: {data_path}"
        return result
    
    if not prebuilt_path or not prebuilt_path.exists():
        result.error = f"Prebuilt file not found: {prebuilt_path}"
        return result
    
    # Create year-arm output directory
    year_arm_output = output_dir / arm / str(year)
    year_arm_output.mkdir(parents=True, exist_ok=True)
    
    # Build environment
    env = os.environ.copy()
    env["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env["GX1_REQUIRE_ENTRY_TELEMETRY"] = "1"
    env["GX1_GATED_FUSION_ENABLED"] = "1"
    env["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    env["GX1_FEATURE_BUILD_DISABLED"] = "1"
    
    if drop_uncertainty:
        env["GX1_XGB_CHANNEL_MASK"] = "uncertainty_score"
    
    # Build command
    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "gx1" / "scripts" / "replay_eval_gated_parallel.py"),
        "--data", str(data_path),
        "--prebuilt-parquet", str(prebuilt_path),
        "--bundle-dir", str(bundle_dir),
        "--policy", str(policy),
        "--output-dir", str(year_arm_output),
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
            timeout=1800  # 30 min timeout per year
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
    
    # Parse results from chunk_footer.json
    chunk_footer = year_arm_output / "chunk_0" / "chunk_footer.json"
    if not chunk_footer.exists():
        # Try to find any chunk footer
        for chunk_dir in year_arm_output.glob("chunk_*"):
            footer = chunk_dir / "chunk_footer.json"
            if footer.exists():
                chunk_footer = footer
                break
    
    if not chunk_footer.exists():
        result.error = "No chunk_footer.json found"
        return result
    
    try:
        with open(chunk_footer) as f:
            footer_data = json.load(f)
    except Exception as e:
        result.error = f"Failed to parse chunk_footer.json: {e}"
        return result
    
    result.success = True
    
    # Extract trading metrics
    result.trades = footer_data.get("n_trades_closed", 0)
    result.pnl_bps = footer_data.get("total_pnl_bps", 0.0)
    result.max_dd_bps = footer_data.get("max_dd_bps", 0.0)
    
    # Extract pipeline metrics
    result.transformer_forward_calls = footer_data.get("transformer_forward_calls", 0)
    result.xgb_pre_predict_count = footer_data.get("xgb_pre_predict_count", 0)
    result.model_attempts = footer_data.get("model_attempt_calls", 0)
    result.eligibility_blocks = footer_data.get("bars_blocked_hard_eligibility", 0)
    
    if result.transformer_forward_calls > 0:
        result.conversion_rate = result.trades / result.transformer_forward_calls
    
    # Extract XGB channels
    result.xgb_seq_channels = footer_data.get("xgb_seq_channel_names", [])
    result.xgb_snap_channels = footer_data.get("xgb_snap_channel_names", [])
    result.xgb_used_as = footer_data.get("xgb_used_as", "unknown")
    
    # Check margin_xgb presence
    all_channels = result.xgb_seq_channels + result.xgb_snap_channels
    result.margin_xgb_present = "margin_xgb" in all_channels
    
    # Extract invariants
    result.prebuilt_used = footer_data.get("prebuilt_used", False)
    result.feature_build_call_count = footer_data.get("feature_build_call_count", -1)
    
    lookup = footer_data.get("lookup_accounting", {})
    if lookup:
        hits = lookup.get("lookup_hits", 0)
        attempts = lookup.get("lookup_attempts", 0)
        misses = lookup.get("lookup_misses", 0)
        result.lookup_invariant_pass = (hits + misses == attempts) if attempts > 0 else True
    else:
        result.lookup_invariant_pass = True
    
    # Extract session breakdown
    session_data = footer_data.get("session_breakdown", {})
    for session, metrics in session_data.items():
        result.session_breakdown[session] = {
            "trades": metrics.get("trades", 0),
            "pnl_bps": metrics.get("pnl_bps", 0.0),
            "max_dd_bps": metrics.get("max_dd_bps", 0.0),
            "forward_calls": metrics.get("forward_calls", 0),
            "xgb_calls": metrics.get("xgb_calls", 0),
            "model_attempts": metrics.get("model_attempts", 0),
            "eligibility_blocks": metrics.get("eligibility_blocks", 0),
        }
    
    # Extract regime breakdown
    regime_data = footer_data.get("regime_breakdown", {})
    for regime, metrics in regime_data.items():
        result.regime_breakdown[regime] = {
            "trades": metrics.get("trades", 0),
            "pnl_bps": metrics.get("pnl_bps", 0.0),
            "max_dd_bps": metrics.get("max_dd_bps", 0.0),
            "forward_calls": metrics.get("forward_calls", 0),
        }
    
    # Data contract checks
    result.ts_utc_ok = footer_data.get("ts_utc_ok", True)
    result.monotonic_ok = footer_data.get("monotonic_ok", True)
    result.bars_total = footer_data.get("bars_reaching_entry_stage", 0)
    
    # Session bar sums check
    session_bars_sum = sum(
        s.get("forward_calls", 0) for s in result.session_breakdown.values()
    )
    result.session_bar_sums_ok = (
        session_bars_sum == result.transformer_forward_calls or 
        len(result.session_breakdown) == 0
    )
    
    # Write YEAR_PROOF.json
    proof = {
        "year": year,
        "arm": arm,
        "success": result.success,
        "trades": result.trades,
        "pnl_bps": result.pnl_bps,
        "max_dd_bps": result.max_dd_bps,
        "transformer_forward_calls": result.transformer_forward_calls,
        "xgb_pre_predict_count": result.xgb_pre_predict_count,
        "xgb_used_as": result.xgb_used_as,
        "xgb_seq_channels": result.xgb_seq_channels,
        "xgb_snap_channels": result.xgb_snap_channels,
        "margin_xgb_present": result.margin_xgb_present,
        "prebuilt_used": result.prebuilt_used,
        "feature_build_call_count": result.feature_build_call_count,
        "lookup_invariant_pass": result.lookup_invariant_pass,
        "conversion_rate": result.conversion_rate,
        "invariant_checks": {
            "prebuilt_only": result.prebuilt_used,
            "xgb_pre": result.xgb_used_as == "pre",
            "no_margin_xgb": not result.margin_xgb_present,
            "transformer_calls_gt_0": result.transformer_forward_calls > 0,
            "lookup_ok": result.lookup_invariant_pass,
        }
    }
    
    proof_path = year_arm_output / "YEAR_PROOF.json"
    with open(proof_path, "w") as f:
        json.dump(proof, f, indent=2)
    
    return result


def check_invariants(result: YearResult) -> List[str]:
    """Check hard invariants and return list of failures."""
    failures = []
    
    if not result.success:
        failures.append(f"RUN_FAILED: {result.error}")
        return failures
    
    if not result.prebuilt_used:
        failures.append("PREBUILT_NOT_USED")
    
    if result.feature_build_call_count != 0 and result.feature_build_call_count != -1:
        failures.append(f"FEATURE_BUILD_CALL_COUNT={result.feature_build_call_count}")
    
    if result.xgb_used_as != "pre":
        failures.append(f"XGB_USED_AS={result.xgb_used_as}")
    
    if result.margin_xgb_present:
        failures.append("MARGIN_XGB_PRESENT")
    
    if result.transformer_forward_calls == 0:
        failures.append("TRANSFORMER_FORWARD_CALLS=0")
    
    if not result.lookup_invariant_pass:
        failures.append("LOOKUP_INVARIANT_FAILED")
    
    return failures


def generate_blame_matrix(results: List[YearResult], output_path: Path):
    """Generate BLAME_MATRIX.csv with year × arm × session × regime breakdown."""
    rows = []
    
    for result in results:
        # Base row for year total
        base_row = {
            "year": result.year,
            "arm": result.arm,
            "session": "ALL",
            "regime": "ALL",
            "trades": result.trades,
            "pnl_bps": round(result.pnl_bps, 2),
            "maxdd_bps": round(result.max_dd_bps, 2),
            "forward_calls": result.transformer_forward_calls,
            "xgb_calls": result.xgb_pre_predict_count,
            "model_attempts": result.model_attempts,
            "eligibility_blocks": result.eligibility_blocks,
            "conversion": round(result.conversion_rate, 6),
        }
        rows.append(base_row)
        
        # Session breakdown rows
        for session, metrics in result.session_breakdown.items():
            session_row = {
                "year": result.year,
                "arm": result.arm,
                "session": session,
                "regime": "ALL",
                "trades": metrics.get("trades", 0),
                "pnl_bps": round(metrics.get("pnl_bps", 0.0), 2),
                "maxdd_bps": round(metrics.get("max_dd_bps", 0.0), 2),
                "forward_calls": metrics.get("forward_calls", 0),
                "xgb_calls": metrics.get("xgb_calls", 0),
                "model_attempts": metrics.get("model_attempts", 0),
                "eligibility_blocks": metrics.get("eligibility_blocks", 0),
                "conversion": round(
                    metrics.get("trades", 0) / metrics.get("forward_calls", 1) 
                    if metrics.get("forward_calls", 0) > 0 else 0.0, 6
                ),
            }
            rows.append(session_row)
        
        # Regime breakdown rows
        for regime, metrics in result.regime_breakdown.items():
            regime_row = {
                "year": result.year,
                "arm": result.arm,
                "session": "ALL",
                "regime": regime,
                "trades": metrics.get("trades", 0),
                "pnl_bps": round(metrics.get("pnl_bps", 0.0), 2),
                "maxdd_bps": round(metrics.get("max_dd_bps", 0.0), 2),
                "forward_calls": metrics.get("forward_calls", 0),
                "xgb_calls": 0,
                "model_attempts": 0,
                "eligibility_blocks": 0,
                "conversion": round(
                    metrics.get("trades", 0) / metrics.get("forward_calls", 1)
                    if metrics.get("forward_calls", 0) > 0 else 0.0, 6
                ),
            }
            rows.append(regime_row)
    
    # Write CSV
    fieldnames = [
        "year", "arm", "session", "regime", "trades", "pnl_bps", 
        "maxdd_bps", "forward_calls", "xgb_calls", "model_attempts",
        "eligibility_blocks", "conversion"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  Written: {output_path}")


def generate_sanity_data_contract(results: List[YearResult], output_path: Path):
    """Generate SANITY_DATA_CONTRACT.json with per-year data contract checks."""
    contract = {}
    
    for result in results:
        key = f"{result.year}_{result.arm}"
        contract[key] = {
            "year": result.year,
            "arm": result.arm,
            "ts_utc_ok": result.ts_utc_ok,
            "monotonic_ok": result.monotonic_ok,
            "bars_total": result.bars_total,
            "session_bar_sums_ok": result.session_bar_sums_ok,
            "notes": [] if result.success else [result.error],
        }
    
    with open(output_path, "w") as f:
        json.dump(contract, f, indent=2)
    
    print(f"  Written: {output_path}")


def generate_arm_compare(
    arm_a_results: List[YearResult], 
    arm_b_results: List[YearResult], 
    output_path: Path
):
    """Generate ARM_COMPARE.md with deltas between arms."""
    lines = [
        "# ARM Compare: A (baseline) vs B (drop uncertainty_score)",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Summary Deltas (B - A)",
        "",
        "| Year | Δ Trades | Δ PnL (bps) | Δ MaxDD | Δ Forward Calls | Verdict |",
        "|------|----------|-------------|---------|-----------------|---------|",
    ]
    
    total_delta_trades = 0
    total_delta_pnl = 0.0
    
    for a_result in arm_a_results:
        b_result = next((r for r in arm_b_results if r.year == a_result.year), None)
        if not b_result:
            continue
        
        delta_trades = b_result.trades - a_result.trades
        delta_pnl = b_result.pnl_bps - a_result.pnl_bps
        delta_maxdd = b_result.max_dd_bps - a_result.max_dd_bps
        delta_forward = b_result.transformer_forward_calls - a_result.transformer_forward_calls
        
        total_delta_trades += delta_trades
        total_delta_pnl += delta_pnl
        
        # Determine verdict
        if delta_pnl > 50:
            verdict = "✅ BETTER"
        elif delta_pnl < -50:
            verdict = "❌ WORSE"
        else:
            verdict = "➖ NEUTRAL"
        
        lines.append(
            f"| {a_result.year} | {delta_trades:+d} | {delta_pnl:+.0f} | {delta_maxdd:+.0f} | {delta_forward:+d} | {verdict} |"
        )
    
    lines.extend([
        "",
        f"**Total Δ Trades:** {total_delta_trades:+d}",
        f"**Total Δ PnL:** {total_delta_pnl:+.0f} bps",
        "",
        "---",
        "",
        "## XGB Channels",
        "",
        "### ARM A (baseline)",
        f"- SEQ: {arm_a_results[0].xgb_seq_channels if arm_a_results else []}",
        f"- SNAP: {arm_a_results[0].xgb_snap_channels if arm_a_results else []}",
        "",
        "### ARM B (drop uncertainty_score)",
        f"- SEQ: {arm_b_results[0].xgb_seq_channels if arm_b_results else []}",
        f"- SNAP: {arm_b_results[0].xgb_snap_channels if arm_b_results else []}",
        "",
        "---",
        "",
        "## Per-Year Details",
        "",
    ])
    
    for a_result in arm_a_results:
        b_result = next((r for r in arm_b_results if r.year == a_result.year), None)
        if not b_result:
            continue
        
        lines.extend([
            f"### {a_result.year}",
            "",
            "| Metric | ARM A | ARM B | Delta |",
            "|--------|-------|-------|-------|",
            f"| Trades | {a_result.trades:,} | {b_result.trades:,} | {b_result.trades - a_result.trades:+d} |",
            f"| PnL (bps) | {a_result.pnl_bps:.0f} | {b_result.pnl_bps:.0f} | {b_result.pnl_bps - a_result.pnl_bps:+.0f} |",
            f"| MaxDD | {a_result.max_dd_bps:.0f} | {b_result.max_dd_bps:.0f} | {b_result.max_dd_bps - a_result.max_dd_bps:+.0f} |",
            f"| Forward Calls | {a_result.transformer_forward_calls:,} | {b_result.transformer_forward_calls:,} | {b_result.transformer_forward_calls - a_result.transformer_forward_calls:+d} |",
            "",
        ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"  Written: {output_path}")


def generate_multiyear_summary(
    arm_a_results: List[YearResult],
    arm_b_results: List[YearResult],
    output_path: Path
):
    """Generate MULTIYEAR_SUMMARY.json with per-arm totals."""
    
    def aggregate_arm(results: List[YearResult]) -> Dict:
        total = {
            "trades": sum(r.trades for r in results),
            "pnl_bps": sum(r.pnl_bps for r in results),
            "max_dd_bps": min(r.max_dd_bps for r in results) if results else 0,
            "transformer_forward_calls": sum(r.transformer_forward_calls for r in results),
            "xgb_pre_predict_count": sum(r.xgb_pre_predict_count for r in results),
            "years": {r.year: asdict(r) for r in results},
        }
        return total
    
    summary = {
        "arm_a": aggregate_arm(arm_a_results),
        "arm_b": aggregate_arm(arm_b_results),
        "delta": {
            "trades": sum(r.trades for r in arm_b_results) - sum(r.trades for r in arm_a_results),
            "pnl_bps": sum(r.pnl_bps for r in arm_b_results) - sum(r.pnl_bps for r in arm_a_results),
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multiyear 2020-2025 Diagnostic A/B: Uncertainty Score Ablation"
    )
    parser.add_argument(
        "--years", 
        nargs="+", 
        type=int, 
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to run"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="Policy YAML path"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Bundle directory"
    )
    parser.add_argument(
        "--prebuilt-root",
        type=Path,
        default=DEFAULT_PREBUILT_ROOT,
        help="Prebuilt root directory (with year subdirs, e.g., GX1_DATA/data/prebuilt/TRIAL160)"
    )
    parser.add_argument(
        "--arm-b-drop-uncertainty",
        action="store_true",
        default=True,
        help="Drop uncertainty_score in ARM B (default: True)"
    )
    parser.add_argument(
        "--arm-a-only",
        action="store_true",
        help="Run only ARM A (baseline)"
    )
    parser.add_argument(
        "--arm-b-only",
        action="store_true",
        help="Run only ARM B (ablation)"
    )
    
    args = parser.parse_args()
    
    # Create run ID and output directory
    run_id = f"DIAGNOSE_AB_UNCERTAINTY_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"MULTIYEAR DIAGNOSTIC A/B: UNCERTAINTY ABLATION")
    print(f"=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_dir}")
    print(f"Years: {args.years}")
    print(f"Workers: {args.workers}")
    print(f"Prebuilt root: {args.prebuilt_root}")
    print()
    
    # Validate prebuilt root exists
    if not args.prebuilt_root.exists():
        print(f"ERROR: Prebuilt root does not exist: {args.prebuilt_root}")
        return 1
    
    # Validate prebuilt files for each year
    missing_prebuilt = []
    for year in args.years:
        prebuilt_path = get_prebuilt_path(year, args.prebuilt_root)
        if not prebuilt_path.exists():
            missing_prebuilt.append((year, prebuilt_path))
    
    if missing_prebuilt:
        print("ERROR: Missing prebuilt files:")
        for year, path in missing_prebuilt:
            print(f"  {year}: {path}")
        print(f"\nHint: Prebuilt files should be at: {args.prebuilt_root}/<year>/xauusd_m5_<year>_features_v10_ctx.parquet")
        return 1
    
    print("✅ All prebuilt files found")
    print()
    
    # Determine which arms to run
    run_arm_a = not args.arm_b_only
    run_arm_b = not args.arm_a_only and args.arm_b_drop_uncertainty
    
    arms = []
    if run_arm_a:
        arms.append("ARM_A")
    if run_arm_b:
        arms.append("ARM_B")
    
    # Write RUN_MANIFEST.json
    manifest = RunManifest(
        run_id=run_id,
        policy_id=str(args.policy.name),
        bundle_sha=compute_bundle_sha(args.bundle_dir),
        replay_mode="PREBUILT_ONLY",
        temp_scaling_effective=True,
        arms=arms,
        years=args.years,
        cmdline=sys.argv,
        sys_executable=sys.executable,
        timestamp=datetime.datetime.now().isoformat(),
        xgb_channels_seq_arm_a=["p_long_xgb", "uncertainty_score"],
        xgb_channels_snap_arm_a=["p_long_xgb", "p_hat_xgb"],
        xgb_channels_seq_arm_b=["p_long_xgb"],  # uncertainty_score dropped
        xgb_channels_snap_arm_b=["p_long_xgb", "p_hat_xgb"],
    )
    
    manifest_path = output_dir / "RUN_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    print(f"Written: {manifest_path}")
    print()
    
    # Run ARM A (baseline)
    arm_a_results = []
    if run_arm_a:
        print("=" * 40)
        print("ARM A: Baseline (current truth pipeline)")
        print("=" * 40)
        
        for year in args.years:
            result = run_year_arm(
                year=year,
                arm="ARM_A",
                output_dir=output_dir,
                policy=args.policy,
                bundle_dir=args.bundle_dir,
                prebuilt_root=args.prebuilt_root,
                workers=args.workers,
                drop_uncertainty=False
            )
            arm_a_results.append(result)
            
            # Check invariants
            failures = check_invariants(result)
            if failures:
                print(f"  ⚠️  {year} invariant failures: {failures}")
            else:
                print(f"  ✅ {year}: {result.trades} trades, {result.pnl_bps:.0f} bps")
    
    # Run ARM B (drop uncertainty_score)
    arm_b_results = []
    if run_arm_b:
        print()
        print("=" * 40)
        print("ARM B: Drop uncertainty_score")
        print("=" * 40)
        
        for year in args.years:
            result = run_year_arm(
                year=year,
                arm="ARM_B",
                output_dir=output_dir,
                policy=args.policy,
                bundle_dir=args.bundle_dir,
                prebuilt_root=args.prebuilt_root,
                workers=args.workers,
                drop_uncertainty=True
            )
            arm_b_results.append(result)
            
            # Check invariants
            failures = check_invariants(result)
            if failures:
                print(f"  ⚠️  {year} invariant failures: {failures}")
            else:
                print(f"  ✅ {year}: {result.trades} trades, {result.pnl_bps:.0f} bps")
    
    # Generate outputs
    print()
    print("=" * 40)
    print("GENERATING REPORTS")
    print("=" * 40)
    
    all_results = arm_a_results + arm_b_results
    
    # BLAME_MATRIX.csv
    generate_blame_matrix(all_results, output_dir / "BLAME_MATRIX.csv")
    
    # SANITY_DATA_CONTRACT.json
    generate_sanity_data_contract(all_results, output_dir / "SANITY_DATA_CONTRACT.json")
    
    # MULTIYEAR_SUMMARY.json
    generate_multiyear_summary(arm_a_results, arm_b_results, output_dir / "MULTIYEAR_SUMMARY.json")
    
    # ARM_COMPARE.md
    if arm_a_results and arm_b_results:
        generate_arm_compare(arm_a_results, arm_b_results, output_dir / "ARM_COMPARE.md")
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if arm_a_results:
        total_a_trades = sum(r.trades for r in arm_a_results)
        total_a_pnl = sum(r.pnl_bps for r in arm_a_results)
        print(f"ARM A: {total_a_trades:,} trades, {total_a_pnl:.0f} bps")
    
    if arm_b_results:
        total_b_trades = sum(r.trades for r in arm_b_results)
        total_b_pnl = sum(r.pnl_bps for r in arm_b_results)
        print(f"ARM B: {total_b_trades:,} trades, {total_b_pnl:.0f} bps")
        
        if arm_a_results:
            delta_trades = total_b_trades - total_a_trades
            delta_pnl = total_b_pnl - total_a_pnl
            print(f"DELTA: {delta_trades:+d} trades, {delta_pnl:+.0f} bps")
    
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Files generated:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
