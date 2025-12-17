#!/usr/bin/env python3
"""
Pipeline Hygiene Audit: Runs Inventory Scanner

Scans all run directories and creates an inventory of:
- Run metadata (dates, sizes, fingerprints)
- Config/policy references
- Baseline alignment
- Obsolete candidates

Outputs:
- gx1/wf_runs/_inventory.json (detailed)
- gx1/wf_runs/_inventory.csv (summary)
- Terminal summary with actionable insights
"""
from __future__ import annotations

import json
import hashlib
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv

import pandas as pd
import yaml


def get_git_commit(path: Path) -> Optional[str]:
    """Get git commit hash for a path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path.parent.parent.parent if "wf_runs" in str(path) else path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_file_hash(file_path: Path) -> Optional[str]:
    """Get SHA256 hash of a file."""
    try:
        if file_path.exists():
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        pass
    return None


def load_yaml_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load YAML file."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return None


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def extract_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Extract metadata from a run directory."""
    metadata = {
        "run_path": str(run_dir),
        "run_name": run_dir.name,
        "mtime": None,
        "size_bytes": 0,
        "size_mb": 0.0,
        "n_trades": 0,
        "n_journal_files": 0,
        "has_trade_journal": False,
        "has_reconciliation": False,
        "has_run_header": False,
        "git_commit": None,
        "config_path": None,
        "config_hash": None,
        "policy_name": None,
        "policy_hash": None,
        "policy_role": None,
        "guardrail_params": {},
        "instrument": None,
        "timeframe": None,
        "date_range": None,
        "run_tag": None,
        "is_prod_baseline": False,
        "is_canary": False,
        "v3_range_edge_cutoff": None,
        "router_version": None,
        "entry_config": None,
        "exit_config": None,
    }
    
    if not run_dir.exists() or not run_dir.is_dir():
        return metadata
    
    # Get mtime and size
    try:
        stat = run_dir.stat()
        metadata["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        metadata["size_bytes"] = get_dir_size(run_dir)
        metadata["size_mb"] = round(metadata["size_bytes"] / (1024 * 1024), 2)
    except Exception:
        pass
    
    # Load run_header.json
    run_header_path = run_dir / "run_header.json"
    if run_header_path.exists():
        metadata["has_run_header"] = True
        run_header = load_yaml_safe(run_header_path)
        if run_header:
            metadata["run_tag"] = run_header.get("run_tag")
            metadata["git_commit"] = run_header.get("git_commit")
            metadata["config_path"] = run_header.get("config_path")
            metadata["policy_name"] = run_header.get("policy_name")
            metadata["policy_role"] = run_header.get("meta", {}).get("role")
            metadata["instrument"] = run_header.get("instrument")
            metadata["timeframe"] = run_header.get("timeframe")
            
            # Extract guardrail params
            hybrid_exit = run_header.get("hybrid_exit_router", {})
            metadata["v3_range_edge_cutoff"] = hybrid_exit.get("v3_range_edge_cutoff")
            metadata["router_version"] = hybrid_exit.get("version")
            metadata["guardrail_params"] = {
                "v3_range_edge_cutoff": metadata["v3_range_edge_cutoff"],
                "router_version": metadata["router_version"],
            }
            
            # Check if PROD_BASELINE or CANARY
            metadata["is_prod_baseline"] = (metadata["policy_role"] == "PROD_BASELINE")
            metadata["is_canary"] = (metadata["policy_role"] == "CANARY")
            
            # Extract config paths
            metadata["entry_config"] = run_header.get("entry_config")
            metadata["exit_config"] = run_header.get("exit_config")
    
    # Count trade journal files
    trade_journal_dir = run_dir / "trade_journal" / "trades"
    if trade_journal_dir.exists():
        metadata["has_trade_journal"] = True
        try:
            journal_files = list(trade_journal_dir.glob("*.json"))
            metadata["n_journal_files"] = len(journal_files)
            # Count trades from index if available
            index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
            if index_path.exists():
                try:
                    df = pd.read_csv(index_path)
                    metadata["n_trades"] = len(df)
                except Exception:
                    pass
        except Exception:
            pass
    
    # Check for reconciliation report
    recon_path = run_dir / "reconciliation_report.md"
    if recon_path.exists():
        metadata["has_reconciliation"] = True
    
    # Hash config if path exists
    if metadata["config_path"]:
        config_path = Path(metadata["config_path"])
        if config_path.exists():
            metadata["config_hash"] = get_file_hash(config_path)
        elif (Path("gx1/configs/policies") / metadata["config_path"]).exists():
            metadata["config_hash"] = get_file_hash(Path("gx1/configs/policies") / metadata["config_path"])
    
    # Hash policy if path exists
    if metadata["policy_name"]:
        policy_path = Path(f"gx1/configs/policies/active/{metadata['policy_name']}.yaml")
        if not policy_path.exists():
            policy_path = Path(f"gx1/configs/policies/prod_snapshot/{metadata['policy_name']}.yaml")
        if policy_path.exists():
            metadata["policy_hash"] = get_file_hash(policy_path)
    
    return metadata


def find_prod_baseline() -> Dict[str, Any]:
    """Find current PROD_BASELINE configuration."""
    baseline = {
        "policy_path": None,
        "policy_hash": None,
        "entry_config": None,
        "exit_config": None,
        "guardrail_params": {},
        "v3_range_edge_cutoff": None,
        "router_version": None,
    }
    
    # Check prod_snapshot for V3_RANGE baseline
    prod_snapshot_path = Path("gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml")
    if prod_snapshot_path.exists():
        baseline["policy_path"] = str(prod_snapshot_path)
        baseline["policy_hash"] = get_file_hash(prod_snapshot_path)
        policy = load_yaml_safe(prod_snapshot_path)
        if policy:
            baseline["entry_config"] = policy.get("entry_config")
            baseline["exit_config"] = policy.get("exit_config")
            hybrid_exit = policy.get("hybrid_exit_router", {})
            baseline["v3_range_edge_cutoff"] = hybrid_exit.get("v3_range_edge_cutoff")
            baseline["router_version"] = hybrid_exit.get("version")
            baseline["guardrail_params"] = {
                "v3_range_edge_cutoff": baseline["v3_range_edge_cutoff"],
                "router_version": baseline["router_version"],
            }
    
    return baseline


def scan_runs() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan all run directories."""
    runs_dir = Path("gx1/wf_runs")
    if not runs_dir.exists():
        return [], {}
    
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("_"):
            metadata = extract_run_metadata(run_dir)
            runs.append(metadata)
    
    # Sort by mtime (newest first)
    runs.sort(key=lambda x: x.get("mtime") or "", reverse=True)
    
    # Find current baseline
    baseline = find_prod_baseline()
    
    return runs, baseline


def generate_summary(runs: List[Dict[str, Any]], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics."""
    summary = {
        "total_runs": len(runs),
        "total_size_mb": sum(r.get("size_mb", 0) for r in runs),
        "runs_with_journal": sum(1 for r in runs if r.get("has_trade_journal")),
        "runs_with_reconciliation": sum(1 for r in runs if r.get("has_reconciliation")),
        "prod_baseline_runs": sum(1 for r in runs if r.get("is_prod_baseline")),
        "canary_runs": sum(1 for r in runs if r.get("is_canary")),
        "runs_with_v3_range": sum(1 for r in runs if r.get("v3_range_edge_cutoff") is not None),
        "runs_matching_baseline": [],
        "obsolete_candidates": [],
    }
    
    # Find runs matching current baseline
    if baseline.get("policy_hash"):
        for run in runs:
            if run.get("policy_hash") == baseline.get("policy_hash"):
                summary["runs_matching_baseline"].append(run["run_name"])
    
    # Find obsolete candidates (no journal, old, not referenced)
    for run in runs:
        if not run.get("has_trade_journal") and run.get("size_mb", 0) < 1.0:
            summary["obsolete_candidates"].append({
                "run_name": run["run_name"],
                "size_mb": run.get("size_mb", 0),
                "mtime": run.get("mtime"),
                "reason": "No trade journal, small size",
            })
    
    return summary


def main():
    """Main entry point."""
    print("=" * 80)
    print("GX1 Pipeline Hygiene Audit: Runs Inventory Scanner")
    print("=" * 80)
    print()
    
    # Scan runs
    print("Scanning run directories...")
    runs, baseline = scan_runs()
    print(f"Found {len(runs)} run directories")
    print()
    
    # Generate summary
    summary = generate_summary(runs, baseline)
    
    # Print baseline info
    print("=" * 80)
    print("CURRENT PROD_BASELINE")
    print("=" * 80)
    if baseline.get("policy_path"):
        print(f"Policy: {baseline['policy_path']}")
        print(f"Hash: {baseline.get('policy_hash', 'N/A')}")
        print(f"Entry Config: {baseline.get('entry_config', 'N/A')}")
        print(f"Exit Config: {baseline.get('exit_config', 'N/A')}")
        print(f"Router Version: {baseline.get('router_version', 'N/A')}")
        print(f"V3 Range Edge Cutoff: {baseline.get('v3_range_edge_cutoff', 'N/A')}")
    else:
        print("WARNING: No PROD_BASELINE found in prod_snapshot!")
    print()
    
    # Print top 10 newest runs
    print("=" * 80)
    print("TOP 10 NEWEST RUNS")
    print("=" * 80)
    for i, run in enumerate(runs[:10], 1):
        print(f"{i:2d}. {run['run_name']}")
        print(f"    Size: {run.get('size_mb', 0):.2f} MB | Trades: {run.get('n_trades', 0)} | "
              f"Role: {run.get('policy_role', 'N/A')} | "
              f"Date: {run.get('mtime', 'N/A')[:10] if run.get('mtime') else 'N/A'}")
        if run.get("v3_range_edge_cutoff") is not None:
            print(f"    V3 Range Cutoff: {run.get('v3_range_edge_cutoff')}")
    print()
    
    # Print runs matching baseline
    print("=" * 80)
    print(f"RUNS MATCHING CURRENT BASELINE ({len(summary['runs_matching_baseline'])} found)")
    print("=" * 80)
    for run_name in summary["runs_matching_baseline"][:10]:
        print(f"  - {run_name}")
    if len(summary["runs_matching_baseline"]) > 10:
        print(f"  ... and {len(summary['runs_matching_baseline']) - 10} more")
    print()
    
    # Print obsolete candidates
    print("=" * 80)
    print(f"OBSOLETE CANDIDATES ({len(summary['obsolete_candidates'])} found)")
    print("=" * 80)
    for candidate in summary["obsolete_candidates"][:20]:
        print(f"  - {candidate['run_name']}: {candidate['size_mb']:.2f} MB "
              f"({candidate.get('mtime', 'N/A')[:10] if candidate.get('mtime') else 'N/A'}) - {candidate['reason']}")
    if len(summary["obsolete_candidates"]) > 20:
        print(f"  ... and {len(summary['obsolete_candidates']) - 20} more")
    print()
    
    # Save inventory
    output_dir = Path("gx1/wf_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    inventory_json = {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "summary": summary,
        "runs": runs,
    }
    json_path = output_dir / "_inventory.json"
    with open(json_path, "w") as f:
        json.dump(inventory_json, f, indent=2)
    print(f"Saved detailed inventory: {json_path}")
    
    # Save CSV
    csv_path = output_dir / "_inventory.csv"
    if runs:
        df = pd.DataFrame(runs)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV inventory: {csv_path}")
    
    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Total runs: {summary['total_runs']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")
    print(f"Runs with journal: {summary['runs_with_journal']}")
    print(f"PROD_BASELINE runs: {summary['prod_baseline_runs']}")
    print(f"CANARY runs: {summary['canary_runs']}")
    print()


if __name__ == "__main__":
    main()

