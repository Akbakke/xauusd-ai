#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determinism gate test runner.

Runs multiple replays with identical policy to detect:
- Non-determinism in entry logic
- Chunk/state leakage in parallel replay
- Data parity issues

Usage:
    python gx1/analysis/determinism_gate.py \
        --policy gx1/configs/policies/.../PROD_BASELINE.yaml \
        --start 2025-01-01 --end 2025-01-15 \
        --out gx1/wf_runs/DETERMINISM_GATE_2025Q1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of file."""
    if not file_path.exists():
        return None
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_replay(
    policy_path: Path,
    start_date: str,
    end_date: str,
    n_workers: int,
    output_tag: str,
    base_output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single replay and return metadata.
    
    Returns dict with:
    - run_dir: Path to output directory
    - price_data_hash: SHA256 of price_data_filtered.parquet
    - trade_count: Number of trades
    - run_header: run_header.json content if exists
    """
    run_dir = base_output_dir / output_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running replay: {output_tag} (workers={n_workers})")
    
    # Run replay script
    cmd = [
        "bash",
        "scripts/run_replay.sh",
        str(policy_path),
        start_date,
        end_date,
        str(n_workers),
        str(run_dir),
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    
    if result.returncode != 0:
        logger.error(f"Replay failed for {output_tag}: {result.stderr}")
        return {
            "run_dir": run_dir,
            "success": False,
            "error": result.stderr,
        }
    
    # Collect metadata
    metadata = {
        "run_dir": run_dir,
        "success": True,
        "n_workers": n_workers,
    }
    
    # Price data hash
    price_data_path = run_dir / "price_data_filtered.parquet"
    if price_data_path.exists():
        metadata["price_data_hash"] = compute_file_hash(price_data_path)
        
        # Price data stats
        try:
            df_price = pd.read_parquet(price_data_path)
            metadata["price_data_rows"] = len(df_price)
            if len(df_price) > 0:
                metadata["price_data_first_ts"] = str(df_price.index[0])
                metadata["price_data_last_ts"] = str(df_price.index[-1])
                # Check for duplicate timestamps
                duplicates = df_price.index.duplicated().sum()
                metadata["price_data_duplicates"] = duplicates
        except Exception as e:
            logger.warning(f"Could not read price data stats: {e}")
    
    # Trade log
    trade_log_path = run_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not trade_log_path.exists():
        trade_log_path = run_dir / "trade_log.csv"
    
    if trade_log_path.exists():
        try:
            df_trades = pd.read_csv(trade_log_path)
            metadata["trade_count"] = len(df_trades)
            metadata["trade_log_path"] = str(trade_log_path)
            
            # Trade log hash
            metadata["trade_log_hash"] = compute_file_hash(trade_log_path)
        except Exception as e:
            logger.warning(f"Could not read trade log: {e}")
    
    # Run header
    run_header_path = run_dir / "run_header.json"
    if run_header_path.exists():
        try:
            with open(run_header_path) as f:
                metadata["run_header"] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read run_header.json: {e}")
    
    logger.info(f"✅ {output_tag} complete: {metadata.get('trade_count', 0)} trades")
    
    return metadata


def match_trades(
    run1_metadata: Dict[str, Any],
    run2_metadata: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Match trades between two runs.
    
    Returns match statistics.
    """
    run1_dir = run1_metadata["run_dir"]
    run2_dir = run2_metadata["run_dir"]
    
    # Load trade logs
    trade_log1_path = Path(run1_metadata.get("trade_log_path", ""))
    if not trade_log1_path.exists():
        trade_log1_path = run1_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not trade_log1_path.exists():
        trade_log1_path = run1_dir / "trade_log.csv"
    
    trade_log2_path = Path(run2_metadata.get("trade_log_path", ""))
    if not trade_log2_path.exists():
        trade_log2_path = run2_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not trade_log2_path.exists():
        trade_log2_path = run2_dir / "trade_log.csv"
    
    if not trade_log1_path.exists() or not trade_log2_path.exists():
        return {
            "match_rate": None,
            "error": "Trade logs not found",
        }
    
    df1 = pd.read_csv(trade_log1_path)
    df2 = pd.read_csv(trade_log2_path)
    
    df1["entry_time"] = pd.to_datetime(df1["entry_time"])
    df2["entry_time"] = pd.to_datetime(df2["entry_time"])
    
    # Filter by date
    df1_filtered = df1[
        (df1["entry_time"] >= start_date) &
        (df1["entry_time"] <= end_date)
    ].copy()
    df2_filtered = df2[
        (df2["entry_time"] >= start_date) &
        (df2["entry_time"] <= end_date)
    ].copy()
    
    # Create trade keys
    df1_filtered["trade_key"] = (
        df1_filtered["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S") + "_" +
        df1_filtered["entry_price"].astype(str) + "_" +
        df1_filtered["side"].astype(str)
    )
    df2_filtered["trade_key"] = (
        df2_filtered["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S") + "_" +
        df2_filtered["entry_price"].astype(str) + "_" +
        df2_filtered["side"].astype(str)
    )
    
    keys1 = set(df1_filtered["trade_key"])
    keys2 = set(df2_filtered["trade_key"])
    matched_keys = keys1 & keys2
    
    match_rate = len(matched_keys) / max(len(keys1), len(keys2)) if max(len(keys1), len(keys2)) > 0 else 0.0
    
    # Get unmatched trades
    unmatched1 = df1_filtered[~df1_filtered["trade_key"].isin(matched_keys)]
    unmatched2 = df2_filtered[~df2_filtered["trade_key"].isin(matched_keys)]
    
    return {
        "match_rate": match_rate,
        "run1_trades": len(df1_filtered),
        "run2_trades": len(df2_filtered),
        "matched_trades": len(matched_keys),
        "unmatched_run1": len(unmatched1),
        "unmatched_run2": len(unmatched2),
        "unmatched_run1_samples": unmatched1[["entry_time", "entry_price", "side"]].head(10).to_dict("records") if len(unmatched1) > 0 else [],
        "unmatched_run2_samples": unmatched2[["entry_time", "entry_price", "side"]].head(10).to_dict("records") if len(unmatched2) > 0 else [],
    }


def generate_report(
    results: Dict[str, Dict[str, Any]],
    parity_checks: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate determinism report."""
    
    report_path = output_dir / "determinism_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Determinism Gate Report\n\n")
        f.write("## Test Configuration\n\n")
        f.write(f"- **Policy:** `{results['SINGLE'].get('run_dir', 'UNKNOWN')}`\n")
        f.write(f"- **Period:** {results['SINGLE'].get('start_date', 'UNKNOWN')} to {results['SINGLE'].get('end_date', 'UNKNOWN')}\n")
        f.write(f"- **Runs:** SINGLE (n_workers=1), PARALLEL (n_workers=7), SINGLE_REPEAT (n_workers=1)\n\n")
        f.write("---\n\n")
        
        # Run results
        f.write("## Run Results\n\n")
        for run_name, metadata in results.items():
            f.write(f"### {run_name}\n\n")
            f.write(f"- **Workers:** {metadata.get('n_workers', 'UNKNOWN')}\n")
            f.write(f"- **Success:** {metadata.get('success', False)}\n")
            f.write(f"- **Trade count:** {metadata.get('trade_count', 0)}\n")
            f.write(f"- **Price data hash:** `{metadata.get('price_data_hash', 'N/A')[:16]}...`\n")
            if metadata.get("price_data_rows"):
                f.write(f"- **Price data rows:** {metadata['price_data_rows']}\n")
                f.write(f"- **First timestamp:** {metadata.get('price_data_first_ts', 'N/A')}\n")
                f.write(f"- **Last timestamp:** {metadata.get('price_data_last_ts', 'N/A')}\n")
                f.write(f"- **Duplicate timestamps:** {metadata.get('price_data_duplicates', 0)}\n")
            f.write("\n")
        
        # Data parity
        f.write("## Data Parity\n\n")
        single_hash = results["SINGLE"].get("price_data_hash")
        parallel_hash = results["PARALLEL"].get("price_data_hash")
        single_repeat_hash = results["SINGLE_REPEAT"].get("price_data_hash")
        
        f.write("### Price Data Hashes\n\n")
        f.write("| Run | Hash (first 16 chars) | Match SINGLE |\n")
        f.write("|-----|----------------------|-------------|\n")
        f.write(f"| SINGLE | `{single_hash[:16] if single_hash else 'N/A'}...` | ✅ |\n")
        f.write(f"| PARALLEL | `{parallel_hash[:16] if parallel_hash else 'N/A'}...` | {'✅' if parallel_hash == single_hash else '❌'} |\n")
        f.write(f"| SINGLE_REPEAT | `{single_repeat_hash[:16] if single_repeat_hash else 'N/A'}...` | {'✅' if single_repeat_hash == single_hash else '❌'} |\n")
        f.write("\n")
        
        if single_hash != parallel_hash:
            f.write("⚠️  **Price data hash mismatch** between SINGLE and PARALLEL\n\n")
            f.write("Possible causes:\n")
            f.write("- Different filtering logic\n")
            f.write("- Different rounding/timezone handling\n")
            f.write("- Chunk boundary effects\n\n")
        
        if single_hash != single_repeat_hash:
            f.write("⚠️  **Price data hash mismatch** between SINGLE and SINGLE_REPEAT\n\n")
            f.write("This indicates non-deterministic data loading/filtering.\n\n")
        
        # Parity checks
        f.write("## Parity Checks\n\n")
        
        # SINGLE vs SINGLE_REPEAT
        if "SINGLE_vs_SINGLE_REPEAT" in parity_checks:
            pc = parity_checks["SINGLE_vs_SINGLE_REPEAT"]
            f.write("### SINGLE vs SINGLE_REPEAT\n\n")
            f.write(f"- **Match rate:** {pc.get('match_rate', 0.0):.1%}\n")
            f.write(f"- **Trade count diff:** {pc.get('run1_trades', 0)} vs {pc.get('run2_trades', 0)} (diff: {pc.get('run2_trades', 0) - pc.get('run1_trades', 0):+d})\n")
            f.write(f"- **Matched trades:** {pc.get('matched_trades', 0)}\n")
            f.write(f"- **Unmatched SINGLE:** {pc.get('unmatched_run1', 0)}\n")
            f.write(f"- **Unmatched SINGLE_REPEAT:** {pc.get('unmatched_run2', 0)}\n\n")
            
            if pc.get("match_rate", 1.0) < 0.95:
                f.write("⚠️  **Low match rate** - Non-determinism detected in entry logic\n\n")
                if pc.get("unmatched_run1_samples"):
                    f.write("Sample unmatched trades (SINGLE):\n")
                    for trade in pc["unmatched_run1_samples"][:5]:
                        f.write(f"- {trade}\n")
                    f.write("\n")
            else:
                f.write("✅ **High match rate** - Entry logic is deterministic\n\n")
        
        # SINGLE vs PARALLEL
        if "SINGLE_vs_PARALLEL" in parity_checks:
            pc = parity_checks["SINGLE_vs_PARALLEL"]
            f.write("### SINGLE vs PARALLEL\n\n")
            f.write(f"- **Match rate:** {pc.get('match_rate', 0.0):.1%}\n")
            f.write(f"- **Trade count diff:** {pc.get('run1_trades', 0)} vs {pc.get('run2_trades', 0)} (diff: {pc.get('run2_trades', 0) - pc.get('run1_trades', 0):+d})\n")
            f.write(f"- **Matched trades:** {pc.get('matched_trades', 0)}\n")
            f.write(f"- **Unmatched SINGLE:** {pc.get('unmatched_run1', 0)}\n")
            f.write(f"- **Unmatched PARALLEL:** {pc.get('unmatched_run2', 0)}\n\n")
            
            if pc.get("match_rate", 1.0) < 0.95:
                f.write("⚠️  **Low match rate** - Chunk/state leakage detected\n\n")
                if pc.get("unmatched_run2_samples"):
                    f.write("Sample unmatched trades (PARALLEL):\n")
                    for trade in pc["unmatched_run2_samples"][:5]:
                        f.write(f"- {trade}\n")
                    f.write("\n")
            else:
                f.write("✅ **High match rate** - No chunk/state leakage\n\n")
        
        # Root cause analysis
        f.write("## Root Cause Analysis\n\n")
        
        single_repeat_match = parity_checks.get("SINGLE_vs_SINGLE_REPEAT", {}).get("match_rate", 1.0)
        single_parallel_match = parity_checks.get("SINGLE_vs_PARALLEL", {}).get("match_rate", 1.0)
        
        if single_repeat_match < 0.95:
            f.write("**Root Cause: NON-DETERMINISM**\n\n")
            f.write("Entry logic is non-deterministic (SINGLE vs SINGLE_REPEAT mismatch).\n\n")
            f.write("**Fix:**\n")
            f.write("1. Ensure torch eval mode\n")
            f.write("2. Disable dropout\n")
            f.write("3. Set seeds (python/numpy/torch)\n")
            f.write("4. Use torch.use_deterministic_algorithms(True) if possible\n\n")
        elif single_parallel_match < 0.95:
            f.write("**Root Cause: CHUNK STATE LEAKAGE**\n\n")
            f.write("Parallel replay has chunk/state leakage (SINGLE is stable but PARALLEL differs).\n\n")
            f.write("**Fix:**\n")
            f.write("1. Implement chunk warmup overlap (prepend warmup window per chunk)\n")
            f.write("2. Or run canary always in single-worker mode for decision testing\n")
            f.write("3. Document in RUNBOOK: 'For parity testing, use n_workers=1'\n\n")
        elif single_hash != parallel_hash:
            f.write("**Root Cause: DATA MISMATCH**\n\n")
            f.write("Price data differs between SINGLE and PARALLEL runs.\n\n")
            f.write("**Fix:**\n")
            f.write("1. Check filtering logic consistency\n")
            f.write("2. Verify timezone/rounding handling\n")
            f.write("3. Check chunk boundary effects\n\n")
        else:
            f.write("✅ **No issues detected** - All runs match\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if single_repeat_match < 0.95:
            f.write("1. **Immediate:** Fix non-determinism in entry logic\n")
            f.write("2. **Re-run:** Run determinism gate again after fix\n")
        elif single_parallel_match < 0.95:
            f.write("1. **Immediate:** Use n_workers=1 for canary testing\n")
            f.write("2. **Long-term:** Implement chunk warmup overlap\n")
            f.write("3. **Documentation:** Update RUNBOOK with parity testing guidelines\n")
        elif single_hash != parallel_hash:
            f.write("1. **Immediate:** Fix data loading/filtering consistency\n")
            f.write("2. **Re-run:** Run determinism gate again after fix\n")
        else:
            f.write("✅ No action needed - system is deterministic\n\n")
        
        # Re-run plan
        f.write("## Re-run Plan\n\n")
        f.write("After fixes, re-run determinism gate:\n\n")
        f.write("```bash\n")
        f.write("python gx1/analysis/determinism_gate.py \\\n")
        f.write("  --policy <policy_path> \\\n")
        f.write("  --start <start_date> --end <end_date> \\\n")
        f.write("  --out gx1/wf_runs/DETERMINISM_GATE_RETEST\n")
        f.write("```\n\n")
    
    logger.info(f"✅ Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Determinism gate test runner")
    parser.add_argument("--policy", required=True, type=Path, help="Policy YAML path")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    if not args.policy.exists():
        logger.error(f"Policy file not found: {args.policy}")
        sys.exit(1)
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run SINGLE (n_workers=1)
    logger.info("=" * 80)
    logger.info("Running SINGLE (n_workers=1)")
    logger.info("=" * 80)
    results["SINGLE"] = run_replay(
        args.policy,
        args.start,
        args.end,
        n_workers=1,
        output_tag="SINGLE",
        base_output_dir=args.out,
    )
    results["SINGLE"]["start_date"] = args.start
    results["SINGLE"]["end_date"] = args.end
    
    # Run PARALLEL (n_workers=7)
    logger.info("=" * 80)
    logger.info("Running PARALLEL (n_workers=7)")
    logger.info("=" * 80)
    results["PARALLEL"] = run_replay(
        args.policy,
        args.start,
        args.end,
        n_workers=7,
        output_tag="PARALLEL",
        base_output_dir=args.out,
    )
    results["PARALLEL"]["start_date"] = args.start
    results["PARALLEL"]["end_date"] = args.end
    
    # Run SINGLE_REPEAT (n_workers=1, repeat)
    logger.info("=" * 80)
    logger.info("Running SINGLE_REPEAT (n_workers=1, repeat)")
    logger.info("=" * 80)
    results["SINGLE_REPEAT"] = run_replay(
        args.policy,
        args.start,
        args.end,
        n_workers=1,
        output_tag="SINGLE_REPEAT",
        base_output_dir=args.out,
    )
    results["SINGLE_REPEAT"]["start_date"] = args.start
    results["SINGLE_REPEAT"]["end_date"] = args.end
    
    # Parity checks
    logger.info("=" * 80)
    logger.info("Running parity checks")
    logger.info("=" * 80)
    
    parity_checks = {}
    
    # SINGLE vs SINGLE_REPEAT
    if results["SINGLE"].get("success") and results["SINGLE_REPEAT"].get("success"):
        logger.info("Checking SINGLE vs SINGLE_REPEAT...")
        parity_checks["SINGLE_vs_SINGLE_REPEAT"] = match_trades(
            results["SINGLE"],
            results["SINGLE_REPEAT"],
            args.start,
            args.end,
        )
    
    # SINGLE vs PARALLEL
    if results["SINGLE"].get("success") and results["PARALLEL"].get("success"):
        logger.info("Checking SINGLE vs PARALLEL...")
        parity_checks["SINGLE_vs_PARALLEL"] = match_trades(
            results["SINGLE"],
            results["PARALLEL"],
            args.start,
            args.end,
        )
    
    # Generate report
    logger.info("=" * 80)
    logger.info("Generating report")
    logger.info("=" * 80)
    generate_report(results, parity_checks, args.out)
    
    logger.info("=" * 80)
    logger.info("✅ Determinism gate complete")
    logger.info(f"Report: {args.out / 'determinism_report.md'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

