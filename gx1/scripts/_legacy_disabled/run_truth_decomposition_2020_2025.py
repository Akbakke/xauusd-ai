#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Truth Decomposition 2020-2025 - Entrypoint Runner

Runs all decomposition analyses and generates reports.

================================================================================
LOCKED_TRUTH_BASELINE: TRIAL160 (2020-2025)
================================================================================
This script generates the canonical truth baseline for Trial160 across 2020-2025.
The output (reports/truth_decomp/) is the SINGLE SOURCE OF TRUTH for:
- Edge bins (session × regime combinations with positive edge)
- Poison bins (session × regime combinations with negative edge)
- Stable edge bins (edge bins that persist across 2020-2025)
- Winner/loser separation patterns
- Payoff shapes (delayed edge vs instant fail)

⚠️  DO NOT MODIFY TRADING LOGIC BASED ON THIS ANALYSIS WITHOUT:
    1. Explicit hypothesis flag (GX1_HYPOTHESIS_*)
    2. A/B test design
    3. Monster-PC validation

See: docs/TRUTH_BASELINE_LOCK.md
================================================================================
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    log.info(f"Running: {description}")
    log.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        log.info(f"✅ {description} completed")
        if result.stdout:
            log.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"❌ {description} failed")
        log.error(f"Return code: {e.returncode}")
        if e.stdout:
            log.error(f"Stdout: {e.stdout}")
        if e.stderr:
            log.error(f"Stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Truth Decomposition 2020-2025 Runner")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing multiyear baseline output",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        # DEL 4A: Use GX1_DATA env vars for default paths
        default=Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports")) / "truth_decomp",
        help="Output root directory (default: reports/truth_decomp)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2020,2021,2022,2023,2024,2025",
        help="Comma-separated list of years",
    )
    parser.add_argument(
        "--skip-trade-table",
        action="store_true",
        help="Skip building trade table (use existing)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if not args.input_root.is_absolute():
        args.input_root = workspace_root / args.input_root
    if not args.out_root.is_absolute():
        args.out_root = workspace_root / args.out_root
    
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    # Paths
    trade_table_path = args.out_root / "trades_baseline_2020_2025.parquet"
    json_output_path = args.out_root / "truth_decomp_2020_2025.json"
    
    log.info("=" * 60)
    log.info("TRUTH DECOMPOSITION 2020-2025")
    log.info("=" * 60)
    log.info(f"Input root: {args.input_root}")
    log.info(f"Output root: {args.out_root}")
    log.info(f"Years: {args.years}")
    log.info("")
    
    # DEL 0: Audit input root (fail-fast)
    log.info("DEL 0: Auditing input root...")
    audit_cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "audit_truth_input_root.py"),
        "--input-root", str(args.input_root),
    ]
    if not run_command(audit_cmd, "Audit input root"):
        log.error("Failed to audit input root")
        return 1
    
    # DEL 1: Build trade table
    if not args.skip_trade_table or not trade_table_path.exists():
        log.info("DEL 1: Building canonical trade table...")
        cmd = [
            sys.executable,
            str(workspace_root / "gx1" / "scripts" / "build_truth_decomp_trade_table.py"),
            "--input-root", str(args.input_root),
            "--output-path", str(trade_table_path),
            "--years", args.years,
        ]
        if not run_command(cmd, "Build trade table"):
            log.error("Failed to build trade table")
            return 1
    else:
        log.info("DEL 1: Using existing trade table")
    
    # DEL 2: Session × Regime Matrix
    log.info("DEL 2: Building session × regime matrix...")
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "report_truth_decomp_session_regime.py"),
        "--trade-table", str(trade_table_path),
        "--output-dir", str(args.out_root),
        "--json-output", str(json_output_path),
    ]
    if not run_command(cmd, "Session × Regime Matrix"):
        log.error("Failed to build session × regime matrix")
        return 1
    
    # DEL 3: Winner vs Loser Separation
    log.info("DEL 3: Analyzing winner vs loser separation...")
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "report_truth_decomp_winner_loser.py"),
        "--trade-table", str(trade_table_path),
        "--output-dir", str(args.out_root),
        "--json-output", str(json_output_path),
    ]
    if not run_command(cmd, "Winner vs Loser Separation"):
        log.error("Failed to analyze winner/loser separation")
        return 1
    
    # DEL 4: Payoff Shapes
    log.info("DEL 4: Analyzing payoff shapes...")
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "report_truth_decomp_payoff_shapes.py"),
        "--trade-table", str(trade_table_path),
        "--output-dir", str(args.out_root),
        "--json-output", str(json_output_path),
    ]
    if not run_command(cmd, "Payoff Shapes"):
        log.error("Failed to analyze payoff shapes")
        return 1
    
    # DEL 5: Stability Test
    log.info("DEL 5: Running stability test (2020 vs 2025)...")
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "report_truth_decomp_stability.py"),
        "--trade-table", str(trade_table_path),
        "--output-dir", str(args.out_root),
        "--json-output", str(json_output_path),
    ]
    if not run_command(cmd, "Stability Test"):
        log.error("Failed to run stability test")
        return 1
    
    # DEL 6: Executive Summary
    log.info("DEL 6: Generating executive summary...")
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "report_truth_decomp_executive.py"),
        "--trade-table", str(trade_table_path),
        "--json-data", str(json_output_path),
        "--output-dir", str(args.out_root),
    ]
    if not run_command(cmd, "Executive Summary"):
        log.error("Failed to generate executive summary")
        return 1
    
    # Generate INDEX.md
    log.info("Generating INDEX.md...")
    index_path = args.out_root / "INDEX.md"
    with open(index_path, "w") as f:
        f.write("# Truth Decomposition 2020-2025 - Index\n\n")
        f.write(f"**Generated:** {Path(__file__).stat().st_mtime}\n\n")
        f.write("## Reports\n\n")
        f.write("1. [Executive Summary](TRUTH_DECOMP_EXECUTIVE.md)\n")
        f.write("2. [Session × Regime Matrix](TRUTH_DECOMP_SESSION_REGIME_MATRIX.md)\n")
        f.write("3. [Winner vs Loser Separation](TRUTH_DECOMP_WINNER_LOSER_SEPARATION.md)\n")
        f.write("4. [Payoff Shapes](TRUTH_DECOMP_PAYOFF_SHAPES.md)\n")
        f.write("5. [Stability Test (2020 vs 2025)](TRUTH_DECOMP_STABILITY_2020_vs_2025.md)\n\n")
        f.write("## Data Files\n\n")
        f.write("- [Trade Table (Parquet)](trades_baseline_2020_2025.parquet)\n")
        f.write("- [Decomposition Data (JSON)](truth_decomp_2020_2025.json)\n")
        f.write("- [Coverage Stats](trades_baseline_2020_2025_coverage.json)\n\n")
    
    log.info(f"✅ Wrote INDEX.md: {index_path}")
    
    log.info("")
    log.info("=" * 60)
    log.info("✅ TRUTH DECOMPOSITION COMPLETE")
    log.info("=" * 60)
    log.info(f"Reports: {args.out_root}")
    log.info(f"Index: {index_path}")
    log.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
