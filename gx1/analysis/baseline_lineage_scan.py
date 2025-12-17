#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Lineage Scanner.

Scans entire repository to identify authoritative PROD_BASELINE bundles
and maps all runs to their bundles. Generates a shortlist of production
candidates with hashes, usage history, and recommendations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute hash for {file_path}: {e}")
        return None


def normalize_config_for_hash(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for hashing by removing run-specific fields."""
    import copy
    
    normalized = copy.deepcopy(config)
    
    # Remove date fields
    for key in ["start_date", "end_date", "start", "end", "period"]:
        if key in normalized:
            del normalized[key]
    
    # Remove output/logging paths
    for key in ["output_dir", "log_dir", "trade_log_path", "run_id", "run_tag"]:
        if key in normalized:
            del normalized[key]
    
    # Remove from meta
    if "meta" in normalized:
        meta = normalized["meta"]
        for key in ["run_tag", "output_dir", "start_date", "end_date"]:
            if key in meta:
                del meta[key]
    
    # Normalize paths (make relative, normalize separators)
    def normalize_paths(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and ("/" in v or "\\" in v):
                    # Normalize separators
                    obj[k] = v.replace("\\", "/")
                else:
                    normalize_paths(v)
        elif isinstance(obj, list):
            for item in obj:
                normalize_paths(item)
    
    normalize_paths(normalized)
    
    return normalized


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA256 hash of normalized config."""
    normalized = normalize_config_for_hash(config)
    config_str = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def load_yaml_config(path: Path) -> Optional[Dict[str, Any]]:
    """Load YAML config file."""
    if not path.exists():
        return None
    
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load YAML from {path}: {e}")
        return None


def build_effective_config(policy: Dict[str, Any], policy_path: Path) -> Dict[str, Any]:
    """Build effective config by merging policy + entry_config + exit_config."""
    effective = {"policy": policy}
    
    # Load entry config
    entry_config_path = policy.get("entry_config")
    if entry_config_path:
        entry_path = Path(entry_config_path)
        if not entry_path.is_absolute():
            entry_path = policy_path.parent / entry_path
        
        if entry_path.exists():
            entry_config = load_yaml_config(entry_path)
            if entry_config:
                effective["entry_config"] = entry_config
    
    # Load exit config
    exit_config_path = policy.get("exit_config")
    if exit_config_path:
        exit_path = Path(exit_config_path)
        if not exit_path.is_absolute():
            exit_path = policy_path.parent / exit_path
        
        if exit_path.exists():
            exit_config = load_yaml_config(exit_path)
            if exit_config:
                effective["exit_config"] = exit_config
    
    return effective


def build_bundle_fingerprint(
    policy_path: Path,
    policy: Dict[str, Any],
    effective_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build complete bundle fingerprint.
    
    Returns:
        Dict with all fingerprint components
    """
    fingerprint = {
        "bundle_id": None,  # Will be computed from all components
        "policy_path": str(policy_path),
        "policy_effective_sha256": None,
        "entry_config_sha256": None,
        "exit_config_sha256": None,
        "router_model_sha256": None,
        "feature_manifest_sha256": None,
        "entry_model_metadata": {},
        "guardrail_cutoff": None,
        "router_version": None,
        "price_source_contract": {},
        "role": None,
        "source": None,  # Will be set by caller
    }
    
    # Policy effective hash
    fingerprint["policy_effective_sha256"] = compute_config_hash(policy)
    
    # Entry config hash
    entry_config = effective_config.get("entry_config")
    if entry_config:
        fingerprint["entry_config_sha256"] = compute_config_hash(entry_config)
    
    # Exit config hash
    exit_config = effective_config.get("exit_config")
    if exit_config:
        fingerprint["exit_config_sha256"] = compute_config_hash(exit_config)
    
    # Router info
    router_cfg = policy.get("hybrid_exit_router", {})
    fingerprint["router_version"] = router_cfg.get("version")
    fingerprint["guardrail_cutoff"] = router_cfg.get("v3_range_edge_cutoff")
    
    # Router model hash
    router_model_path = router_cfg.get("model_path")
    if router_model_path:
        router_path = Path(router_model_path)
        if not router_path.is_absolute():
            router_path = policy_path.parent / router_path
        
        if router_path.exists():
            fingerprint["router_model_sha256"] = compute_file_hash(router_path)
    
    # Feature manifest hash
    # Try to find feature manifest (usually near entry models or in prod/current)
    manifest_candidates = [
        policy_path.parent / "feature_manifest.json",
        Path("gx1/models/feature_manifest.json"),
        Path("gx1/prod/current/feature_manifest.json"),
    ]
    
    for manifest_path in manifest_candidates:
        if manifest_path.exists():
            fingerprint["feature_manifest_sha256"] = compute_file_hash(manifest_path)
            break
    
    # Entry model metadata
    metadata_path = Path("gx1/models/GX1_entry_session_metadata.json")
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                fingerprint["entry_model_metadata"] = {
                    "feature_cols_hash": metadata.get("feature_cols_hash"),
                    "n_features_in_": metadata.get("n_features_in_"),
                    "entry_model_version": metadata.get("entry_model_version", "ENTRY_V9"),
                }
        except Exception as e:
            logger.warning(f"Failed to load entry model metadata: {e}")
    
    # Role
    meta = policy.get("meta", {})
    fingerprint["role"] = meta.get("role")
    
    # Price source contract (infer from policy/runtime patterns)
    # This is a simplified inference - full analysis would require code scanning
    fingerprint["price_source_contract"] = {
        "entry_features": "inferred_bid_ask_mid",  # Default assumption
        "range_features": "inferred_bid_ask_mid",
        "spread": "bid_ask",
    }
    
    # Compute bundle_id from all components
    bundle_components = [
        fingerprint["policy_effective_sha256"],
        fingerprint["entry_config_sha256"] or "",
        fingerprint["exit_config_sha256"] or "",
        fingerprint["router_model_sha256"] or "",
        fingerprint["feature_manifest_sha256"] or "",
        fingerprint["router_version"] or "",
        str(fingerprint["guardrail_cutoff"] or ""),
    ]
    bundle_str = "|".join(bundle_components)
    fingerprint["bundle_id"] = hashlib.sha256(bundle_str.encode("utf-8")).hexdigest()[:16]
    
    return fingerprint


def scan_prod_snapshots() -> List[Dict[str, Any]]:
    """Scan prod_snapshot directory for baseline candidates."""
    bundles = []
    
    snapshot_dir = Path("gx1/configs/policies/prod_snapshot")
    if not snapshot_dir.exists():
        return bundles
    
    for snapshot_subdir in snapshot_dir.iterdir():
        if not snapshot_subdir.is_dir():
            continue
        
        # Look for policy YAML
        policy_candidates = [
            snapshot_subdir / "policy.yaml",
            snapshot_subdir / "policy.yml",
            snapshot_subdir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml",
        ]
        
        for policy_candidate in policy_candidates:
            if policy_candidate.exists():
                policy = load_yaml_config(policy_candidate)
                if policy:
                    effective_config = build_effective_config(policy, policy_candidate)
                    fingerprint = build_bundle_fingerprint(policy_candidate, policy, effective_config)
                    fingerprint["source"] = "prod_snapshot"
                    fingerprint["snapshot_name"] = snapshot_subdir.name
                    bundles.append(fingerprint)
                    break
    
    return bundles


def scan_active_policies() -> List[Dict[str, Any]]:
    """Scan active policies directory."""
    bundles = []
    
    active_dir = Path("gx1/configs/policies/active")
    if not active_dir.exists():
        return bundles
    
    for policy_file in active_dir.glob("*.yaml"):
        policy = load_yaml_config(policy_file)
        if policy:
            effective_config = build_effective_config(policy, policy_file)
            fingerprint = build_bundle_fingerprint(policy_file, policy, effective_config)
            fingerprint["source"] = "active"
            fingerprint["policy_name"] = policy_file.stem
            bundles.append(fingerprint)
    
    return bundles


def scan_wf_runs() -> List[Dict[str, Any]]:
    """Scan wf_runs for actual executed bundles."""
    bundles = []
    run_mappings = []  # Will store run -> bundle mappings
    
    wf_runs_dir = Path("gx1/wf_runs")
    if not wf_runs_dir.exists():
        return bundles, run_mappings
    
    for run_dir in wf_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_info = {
            "run_tag": run_dir.name,
            "run_dir": str(run_dir),
            "bundle_id": None,
            "mode": None,
            "period": None,
            "n_workers": None,
            "trades_per_day": None,
            "n_trades": None,
        }
        
        # Try to load run_header.json
        header_path = run_dir / "run_header.json"
        if header_path.exists():
            try:
                with open(header_path, "r") as f:
                    header = json.load(f)
                    artifacts = header.get("artifacts", {})
                    meta = header.get("meta", {})
                    
                    run_info["mode"] = meta.get("mode")
                    run_info["period"] = f"{meta.get('start_date', 'N/A')} to {meta.get('end_date', 'N/A')}"
                    run_info["n_workers"] = meta.get("n_workers")
                    
                    # Try to match bundle from hashes
                    policy_hash = artifacts.get("policy", {}).get("sha256")
                    router_hash = artifacts.get("router_model", {}).get("sha256")
                    manifest_hash = artifacts.get("feature_manifest", {}).get("sha256")
                    
                    # We'll match this later
                    run_info["policy_hash"] = policy_hash
                    run_info["router_hash"] = router_hash
                    run_info["manifest_hash"] = manifest_hash
            except Exception as e:
                logger.warning(f"Failed to load run_header from {run_dir}: {e}")
        
        # Try to find policy from parallel_chunks
        chunks_dir = run_dir / "parallel_chunks"
        policy_path = None
        if chunks_dir.exists():
            chunk_policies = list(chunks_dir.glob("policy_chunk_*.yaml"))
            if chunk_policies:
                policy_path = chunk_policies[0]
        
        # If no policy found, try to infer from logs
        if not policy_path:
            log_files = list(run_dir.glob("*.log"))
            for log_file in log_files[:1]:  # Check first log file
                try:
                    with open(log_file, "r") as f:
                        content = f.read()
                        # Look for policy path
                        match = re.search(r'policy[_-]?path[:\s=]+([^\s\n]+)', content, re.IGNORECASE)
                        if match:
                            policy_path = Path(match.group(1))
                            if policy_path.exists():
                                break
                except Exception:
                    continue
        
        # Build bundle from policy if found
        if policy_path and policy_path.exists():
            policy = load_yaml_config(policy_path)
            if policy:
                effective_config = build_effective_config(policy, policy_path)
                fingerprint = build_bundle_fingerprint(policy_path, policy, effective_config)
                fingerprint["source"] = "wf_run"
                fingerprint["run_tag"] = run_dir.name
                bundles.append(fingerprint)
                run_info["bundle_id"] = fingerprint["bundle_id"]
        
        # Try to get run stats
        results_path = run_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    run_info["trades_per_day"] = results.get("trades_per_day")
                    run_info["n_trades"] = results.get("n_trades")
            except Exception:
                pass
        
        run_mappings.append(run_info)
    
    return bundles, run_mappings


def score_bundle(bundle: Dict[str, Any], run_mappings: List[Dict[str, Any]]) -> int:
    """Score bundle based on production readiness criteria."""
    score = 0
    
    # A) PROD_BASELINE role
    if bundle.get("role") == "PROD_BASELINE":
        score += 100
    
    # B) prod_snapshot location
    if bundle.get("source") == "prod_snapshot":
        score += 50
    
    # C) Safety hooks (infer from policy structure)
    # Check for policy lock, feature manifest validation, router fail-closed
    # This is simplified - would need deeper analysis
    if bundle.get("feature_manifest_sha256"):
        score += 10  # Has manifest
    
    # D) Practice-live usage
    for run in run_mappings:
        if run.get("bundle_id") == bundle.get("bundle_id"):
            mode = run.get("mode") or ""
            if "live" in mode.lower() or "practice" in mode.lower():
                score += 30
                break
    
    # E) FULLYEAR usage with trade journal
    for run in run_mappings:
        if run.get("bundle_id") == bundle.get("bundle_id"):
            if "FULLYEAR" in run.get("run_tag", "").upper():
                score += 10
                break
    
    return score


def find_conflicts(bundles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find bundles that claim PROD_BASELINE but have different fingerprints."""
    conflicts = []
    
    baseline_bundles = [b for b in bundles if b.get("role") == "PROD_BASELINE"]
    
    if len(baseline_bundles) <= 1:
        return conflicts
    
    # Compare all pairs
    for i, bundle1 in enumerate(baseline_bundles):
        for bundle2 in baseline_bundles[i+1:]:
            if bundle1["bundle_id"] != bundle2["bundle_id"]:
                # Find differences
                differences = []
                if bundle1.get("policy_effective_sha256") != bundle2.get("policy_effective_sha256"):
                    differences.append("policy_effective")
                if bundle1.get("entry_config_sha256") != bundle2.get("entry_config_sha256"):
                    differences.append("entry_config")
                if bundle1.get("exit_config_sha256") != bundle2.get("exit_config_sha256"):
                    differences.append("exit_config")
                if bundle1.get("router_model_sha256") != bundle2.get("router_model_sha256"):
                    differences.append("router_model")
                if bundle1.get("feature_manifest_sha256") != bundle2.get("feature_manifest_sha256"):
                    differences.append("feature_manifest")
                if bundle1.get("guardrail_cutoff") != bundle2.get("guardrail_cutoff"):
                    differences.append(f"guardrail_cutoff ({bundle1.get('guardrail_cutoff')} vs {bundle2.get('guardrail_cutoff')})")
                
                conflicts.append({
                    "bundle1": bundle1["bundle_id"],
                    "bundle1_path": bundle1["policy_path"],
                    "bundle2": bundle2["bundle_id"],
                    "bundle2_path": bundle2["policy_path"],
                    "differences": differences,
                })
    
    return conflicts


def generate_report(
    bundles: List[Dict[str, Any]],
    run_mappings: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
) -> str:
    """Generate markdown baseline lineage report."""
    lines = []
    
    lines.append("# Baseline Lineage Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    unique_bundles = len(set(b["bundle_id"] for b in bundles))
    baseline_count = sum(1 for b in bundles if b.get("role") == "PROD_BASELINE")
    
    lines.append(f"- **Total Unique Bundles Found:** {unique_bundles}")
    lines.append(f"- **Bundles with role=PROD_BASELINE:** {baseline_count}")
    lines.append("")
    
    # Score and rank bundles
    scored_bundles = []
    for bundle in bundles:
        score = score_bundle(bundle, run_mappings)
        bundle["score"] = score
        scored_bundles.append(bundle)
    
    scored_bundles.sort(key=lambda x: x["score"], reverse=True)
    top_3 = scored_bundles[:3]
    
    if top_3:
        recommended = top_3[0]
        lines.append(f"- **Recommended One True Baseline:** `{recommended['bundle_id']}`")
        lines.append(f"  - Source: {recommended['source']}")
        lines.append(f"  - Policy Path: `{recommended['policy_path']}`")
        lines.append(f"  - Score: {recommended['score']}")
    else:
        lines.append("- **Recommended One True Baseline:** None found")
    lines.append("")
    
    # Top 3 Candidates
    lines.append("## Top 3 Candidates")
    lines.append("")
    lines.append("| Bundle ID | Source | Role | Policy Hash | Router | Entry Model | Router Hash | Manifest Hash | Last Run | Mode | Trades/Day |")
    lines.append("|-----------|--------|------|-------------|--------|-------------|-------------|---------------|----------|------|------------|")
    
    for bundle in top_3:
        bundle_id = bundle.get("bundle_id", "N/A")
        source = bundle.get("source", "N/A")
        role = bundle.get("role", "N/A")
        policy_hash = bundle.get("policy_effective_sha256", "N/A")[:12] if bundle.get("policy_effective_sha256") else "N/A"
        router = f"{bundle.get('router_version', 'N/A')}"
        if bundle.get("guardrail_cutoff"):
            router += f" (cutoff={bundle.get('guardrail_cutoff')})"
        entry_model = bundle.get("entry_model_metadata", {}).get("entry_model_version", "N/A")
        router_hash = bundle.get("router_model_sha256", "N/A")[:12] if bundle.get("router_model_sha256") else "N/A"
        manifest_hash = bundle.get("feature_manifest_sha256", "N/A")[:12] if bundle.get("feature_manifest_sha256") else "N/A"
        
        # Find last run
        last_run = None
        last_mode = None
        last_trades_per_day = None
        for run in run_mappings:
            if run.get("bundle_id") == bundle_id:
                last_run = run.get("run_tag", "N/A")
                last_mode = run.get("mode", "N/A")
                last_trades_per_day = run.get("trades_per_day")
                break
        
        trades_str = f"{last_trades_per_day:.2f}" if last_trades_per_day else "N/A"
        
        lines.append(f"| `{bundle_id}` | {source} | {role} | `{policy_hash}...` | {router} | {entry_model} | `{router_hash}...` | `{manifest_hash}...` | {last_run} | {last_mode} | {trades_str} |")
    
    lines.append("")
    
    # Conflicts
    if conflicts:
        lines.append("## Conflicts")
        lines.append("")
        lines.append("Multiple bundles claim PROD_BASELINE but have different fingerprints:")
        lines.append("")
        for conflict in conflicts:
            lines.append(f"### Conflict: `{conflict['bundle1']}` vs `{conflict['bundle2']}`")
            lines.append(f"- Bundle 1: `{conflict['bundle1_path']}`")
            lines.append(f"- Bundle 2: `{conflict['bundle2_path']}`")
            lines.append(f"- Differences: {', '.join(conflict['differences'])}")
            lines.append("")
    else:
        lines.append("## Conflicts")
        lines.append("")
        lines.append("✅ No conflicts found - all PROD_BASELINE bundles have matching fingerprints.")
        lines.append("")
    
    # Run Lineage
    lines.append("## Run Lineage")
    lines.append("")
    
    # Group runs by bundle
    bundle_runs = defaultdict(list)
    for run in run_mappings:
        bundle_id = run.get("bundle_id")
        if bundle_id:
            bundle_runs[bundle_id].append(run)
    
    for bundle in scored_bundles[:5]:  # Top 5 bundles
        bundle_id = bundle.get("bundle_id")
        runs = bundle_runs.get(bundle_id, [])
        
        if runs:
            lines.append(f"### Bundle `{bundle_id}` ({bundle.get('source', 'unknown')})")
            lines.append("")
            lines.append("| Run Tag | Mode | Period | N Workers | Trades/Day |")
            lines.append("|---------|------|--------|-----------|------------|")
            
            for run in sorted(runs, key=lambda x: x.get("run_tag", ""), reverse=True)[:10]:  # Latest 10
                run_tag = run.get("run_tag", "N/A")
                mode = run.get("mode", "N/A")
                period = run.get("period", "N/A")
                n_workers = run.get("n_workers", "N/A")
                trades_per_day = f"{run.get('trades_per_day', 0):.2f}" if run.get("trades_per_day") else "N/A"
                lines.append(f"| `{run_tag}` | {mode} | {period} | {n_workers} | {trades_per_day} |")
            
            lines.append("")
    
    # Action Plan
    lines.append("## Action Plan")
    lines.append("")
    
    if top_3:
        recommended = top_3[0]
        lines.append("### Freeze Baseline")
        lines.append("")
        lines.append(f"Recommended baseline: `{recommended['bundle_id']}`")
        lines.append(f"")
        lines.append(f"```yaml")
        lines.append(f"Policy Path: {recommended['policy_path']}")
        lines.append(f"Policy Hash: {recommended.get('policy_effective_sha256', 'N/A')}")
        lines.append(f"Router Model Hash: {recommended.get('router_model_sha256', 'N/A')}")
        lines.append(f"Feature Manifest Hash: {recommended.get('feature_manifest_sha256', 'N/A')}")
        lines.append(f"```")
        lines.append("")
        
        lines.append("### Fix Parity")
        lines.append("")
        lines.append("To achieve parity between FULLYEAR and CANARY:")
        lines.append("")
        lines.append("1. Use the same policy bundle for both runs")
        lines.append("2. Ensure same entry/exit configs")
        lines.append("3. Use same router model and feature manifest")
        lines.append("4. Run both with `n_workers=1` for determinism")
        lines.append("")
        
        lines.append("### Deprecate")
        lines.append("")
        lines.append("Consider archiving or marking as obsolete:")
        lines.append("- Policies not in prod_snapshot")
        lines.append("- Bundles with score < 50")
        lines.append("- Bundles with no runs")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `gx1/analysis/baseline_lineage_scan.py`*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Lineage Scanner - Identify authoritative PROD_BASELINE"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gx1/analysis/baseline_lineage_report.md"),
        help="Output report path",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting baseline lineage scan...")
    
    # Scan all sources
    logger.info("Scanning prod_snapshots...")
    prod_bundles = scan_prod_snapshots()
    logger.info(f"Found {len(prod_bundles)} bundles in prod_snapshots")
    
    logger.info("Scanning active policies...")
    active_bundles = scan_active_policies()
    logger.info(f"Found {len(active_bundles)} bundles in active policies")
    
    logger.info("Scanning wf_runs...")
    wf_bundles, run_mappings = scan_wf_runs()
    logger.info(f"Found {len(wf_bundles)} bundles in wf_runs")
    logger.info(f"Found {len(run_mappings)} runs")
    
    # Combine all bundles
    all_bundles = prod_bundles + active_bundles + wf_bundles
    
    # Deduplicate by bundle_id
    unique_bundles = {}
    for bundle in all_bundles:
        bundle_id = bundle.get("bundle_id")
        if bundle_id:
            if bundle_id not in unique_bundles:
                unique_bundles[bundle_id] = bundle
            else:
                # Merge sources
                existing = unique_bundles[bundle_id]
                if bundle.get("source") == "prod_snapshot":
                    existing["source"] = "prod_snapshot"  # Prefer prod_snapshot
                elif existing.get("source") != "prod_snapshot" and bundle.get("source") == "active":
                    existing["source"] = "active"
    
    bundles_list = list(unique_bundles.values())
    
    # Find conflicts
    conflicts = find_conflicts(bundles_list)
    
    # Generate report
    logger.info("Generating report...")
    report = generate_report(bundles_list, run_mappings, conflicts)
    
    # Write report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report written to: {args.out}")
    
    # Write JSON
    json_out = args.out.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "bundles": bundles_list,
            "run_mappings": run_mappings,
            "conflicts": conflicts,
            "generated": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    
    logger.info(f"JSON data written to: {json_out}")
    
    # Write CSV
    csv_out = args.out.with_suffix(".csv").with_name(args.out.stem + "_bundles.csv")
    if bundles_list:
        bundles_df = pd.DataFrame(bundles_list)
        bundles_df.to_csv(csv_out, index=False)
        logger.info(f"CSV written to: {csv_out}")
    
    return 0


if __name__ == "__main__":
    exit(main())

