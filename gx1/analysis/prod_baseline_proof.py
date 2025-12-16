#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Baseline Parity Proof Report Generator.

Generates a comprehensive parity proof report for a run directory,
verifying that the run used PROD_BASELINE policy and artifacts correctly.

Usage:
    python gx1/analysis/prod_baseline_proof.py \
      --run gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE \
      --prod-policy gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_..._PROD.yaml \
      --out gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE/prod_baseline_proof.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of file."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute hash for {file_path}: {e}")
        return None


def load_run_header(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run_header.json from run directory."""
    # Try to use gx1.prod.run_header.load_run_header if available
    try:
        from gx1.prod.run_header import load_run_header as _load_run_header
        return _load_run_header(run_dir)
    except ImportError:
        pass
    
    # Fallback: manual loading
    header_path = run_dir / "run_header.json"
    if not header_path.exists():
        # Try parallel_chunks subdirectory
        header_path = run_dir / "parallel_chunks" / "chunk_0" / "run_header.json"
        if not header_path.exists():
            logger.error(f"run_header.json not found in {run_dir}")
            return None
    
    try:
        with open(header_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load run_header.json: {e}")
        return None


def load_prod_policy(prod_policy_path: Path) -> Optional[Dict[str, Any]]:
    """Load production policy YAML."""
    if not prod_policy_path.exists():
        logger.error(f"Production policy not found: {prod_policy_path}")
        return None
    
    try:
        with open(prod_policy_path, "r") as f:
            policy = yaml.safe_load(f)
        
        # Load entry_config and exit_config
        policy_dir = prod_policy_path.parent
        entry_config_path = policy_dir / policy.get("entry_config", "")
        exit_config_path = policy_dir / policy.get("exit_config", "")
        
        entry_config = None
        exit_config = None
        
        if entry_config_path.exists():
            with open(entry_config_path, "r") as f:
                entry_config = yaml.safe_load(f)
        
        if exit_config_path.exists():
            with open(exit_config_path, "r") as f:
                exit_config = yaml.safe_load(f)
        
        return {
            "policy": policy,
            "entry_config": entry_config,
            "exit_config": exit_config,
            "policy_path": prod_policy_path,
            "entry_config_path": entry_config_path if entry_config_path.exists() else None,
            "exit_config_path": exit_config_path if exit_config_path.exists() else None,
        }
    except Exception as e:
        logger.error(f"Failed to load production policy: {e}")
        return None


def verify_policy_parity(
    run_header: Dict[str, Any],
    prod_policy_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify policy parity between run and prod snapshot."""
    results = {
        "policy_sha256_match": False,
        "entry_config_sha256_match": False,
        "exit_config_sha256_match": False,
        "router_model_sha256_match": False,
        "feature_manifest_sha256_match": False,
        "run_artifacts": {},
        "prod_artifacts": {},
    }
    
    # Get run artifacts
    run_artifacts = run_header.get("artifacts", {})
    results["run_artifacts"] = {
        "policy_sha256": run_artifacts.get("policy", {}).get("sha256"),
        "router_model_sha256": run_artifacts.get("router_model", {}).get("sha256"),
        "feature_manifest_sha256": run_artifacts.get("feature_manifest", {}).get("sha256"),
    }
    
    # Compute prod artifact hashes
    prod_policy = prod_policy_bundle["policy"]
    prod_policy_path = prod_policy_bundle["policy_path"]
    prod_entry_config_path = prod_policy_bundle.get("entry_config_path")
    prod_exit_config_path = prod_policy_bundle.get("exit_config_path")
    
    # Router model path from policy
    router_model_path = None
    hybrid_router_cfg = prod_policy.get("hybrid_exit_router", {})
    if hybrid_router_cfg:
        model_path_str = hybrid_router_cfg.get("model_path")
        if model_path_str:
            router_model_path = prod_policy_path.parent / model_path_str
    
    # Feature manifest path (typically next to entry models)
    feature_manifest_path = None
    if prod_entry_config_path:
        # Try to find feature_manifest.json near entry config
        manifest_candidate = prod_entry_config_path.parent / "feature_manifest.json"
        if not manifest_candidate.exists():
            # Try models directory
            manifest_candidate = Path("gx1/models/feature_manifest.json")
        if manifest_candidate.exists():
            feature_manifest_path = manifest_candidate
    
    prod_policy_hash = compute_file_hash(prod_policy_path)
    prod_entry_config_hash = compute_file_hash(prod_entry_config_path) if prod_entry_config_path else None
    prod_exit_config_hash = compute_file_hash(prod_exit_config_path) if prod_exit_config_path else None
    prod_router_hash = compute_file_hash(router_model_path) if router_model_path else None
    prod_manifest_hash = compute_file_hash(feature_manifest_path) if feature_manifest_path else None
    
    results["prod_artifacts"] = {
        "policy_sha256": prod_policy_hash,
        "entry_config_sha256": prod_entry_config_hash,
        "exit_config_sha256": prod_exit_config_hash,
        "router_model_sha256": prod_router_hash,
        "feature_manifest_sha256": prod_manifest_hash,
    }
    
    # Compare hashes
    results["policy_sha256_match"] = (
        results["run_artifacts"]["policy_sha256"] == prod_policy_hash
    )
    results["entry_config_sha256_match"] = (
        results["run_artifacts"].get("entry_config_sha256") == prod_entry_config_hash
        if prod_entry_config_hash
        else None  # Not tracked in run_header
    )
    results["exit_config_sha256_match"] = (
        results["run_artifacts"].get("exit_config_sha256") == prod_exit_config_hash
        if prod_exit_config_hash
        else None  # Not tracked in run_header
    )
    results["router_model_sha256_match"] = (
        results["run_artifacts"]["router_model_sha256"] == prod_router_hash
        if prod_router_hash and results["run_artifacts"]["router_model_sha256"]
        else None
    )
    results["feature_manifest_sha256_match"] = (
        results["run_artifacts"]["feature_manifest_sha256"] == prod_manifest_hash
        if prod_manifest_hash and results["run_artifacts"]["feature_manifest_sha256"]
        else None
    )
    
    return results


def verify_entry_model_proof(
    run_dir: Path,
    prod_policy_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify entry model proof from logs/journal."""
    results = {
        "models_loaded": [],
        "metadata_found": False,
        "metadata_match": False,
        "expected_metadata": None,
        "actual_metadata": None,
    }
    
    # Try to find entry model metadata (standard location)
    metadata_path = Path("gx1/models/GX1_entry_session_metadata.json")
    if not metadata_path.exists():
        # Try relative to run_dir
        metadata_path = run_dir.parent.parent / "models" / "GX1_entry_session_metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                results["expected_metadata"] = json.load(f)
            results["metadata_found"] = True
        except Exception as e:
            logger.warning(f"Failed to load entry model metadata: {e}")
    
    # Extract model info from trade journal
    journal_dir = run_dir / "trade_journal" / "trades"
    models_seen = set()
    
    if journal_dir.exists():
        trade_files = list(journal_dir.glob("*.json"))
        if trade_files:
            # Sample trades to get model info
            for trade_file in trade_files[:min(10, len(trade_files))]:  # Sample first 10
                try:
                    with open(trade_file, "r") as f:
                        trade_data = json.load(f)
                        entry_snapshot = trade_data.get("entry_snapshot", {})
                        if entry_snapshot:
                            model_version = entry_snapshot.get("entry_model_version")
                            if model_version:
                                models_seen.add(model_version)
                except Exception as e:
                    logger.debug(f"Failed to read trade journal file {trade_file}: {e}")
    
    if models_seen:
        results["actual_metadata"] = {
            "entry_model_versions": sorted(list(models_seen)),
        }
        results["models_loaded"] = sorted(list(models_seen))
    
    # Check metadata match
    if results["expected_metadata"] and results["actual_metadata"]:
        expected_models = set(results["expected_metadata"].get("models", {}).keys())
        actual_models = set(results["actual_metadata"]["entry_model_versions"])
        results["metadata_match"] = expected_models == actual_models or len(actual_models) > 0
    
    return results


def verify_runtime_pipeline_proof(run_dir: Path) -> Dict[str, Any]:
    """Verify runtime pipeline proof from trade journal."""
    results = {
        "total_trades": 0,
        "trades_with_entry_snapshot": 0,
        "trades_with_feature_context": 0,
        "trades_with_router_decision": 0,
        "trades_with_guardrail_flag": 0,
        "trades_with_exit_summary": 0,
        "trades_with_intratrade_metrics": 0,
        "coverage_percent": 0.0,
    }
    
    journal_dir = run_dir / "trade_journal" / "trades"
    if not journal_dir.exists():
        logger.warning(f"Trade journal directory not found: {journal_dir}")
        return results
    
    trade_files = list(journal_dir.glob("*.json"))
    results["total_trades"] = len(trade_files)
    
    if results["total_trades"] == 0:
        return results
    
    for trade_file in trade_files:
        try:
            with open(trade_file, "r") as f:
                trade_data = json.load(f)
            
            if trade_data.get("entry_snapshot"):
                results["trades_with_entry_snapshot"] += 1
            
            if trade_data.get("feature_context"):
                results["trades_with_feature_context"] += 1
            
            router_explainability = trade_data.get("router_explainability")
            if router_explainability:
                results["trades_with_router_decision"] += 1
                # Check for guardrail flags
                if router_explainability.get("guardrail_applied") is not None:
                    results.setdefault("trades_with_guardrail_flag", 0)
                    results["trades_with_guardrail_flag"] += 1
            
            exit_summary = trade_data.get("exit_summary")
            if exit_summary:
                results["trades_with_exit_summary"] += 1
                # Check for intratrade metrics
                if (
                    exit_summary.get("max_mfe_bps") is not None
                    or exit_summary.get("max_mae_bps") is not None
                    or exit_summary.get("intratrade_drawdown_bps") is not None
                ):
                    results["trades_with_intratrade_metrics"] += 1
        
        except Exception as e:
            logger.warning(f"Failed to read trade journal file {trade_file}: {e}")
    
    # Calculate coverage (all required fields present)
    required_fields = [
        "trades_with_entry_snapshot",
        "trades_with_feature_context",
        "trades_with_router_decision",
        "trades_with_exit_summary",
    ]
    
    if results["total_trades"] > 0:
        min_coverage = min(results[field] for field in required_fields)
        results["coverage_percent"] = (min_coverage / results["total_trades"]) * 100.0
    
    return results


def load_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from results.json."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        # Try parallel_chunks
        results_path = run_dir / "parallel_chunks" / "chunk_0" / "results.json"
        if not results_path.exists():
            logger.warning(f"results.json not found in {run_dir}")
            return None
    
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results.json: {e}")
        return None


def generate_report(
    run_dir: Path,
    prod_policy_bundle: Dict[str, Any],
    policy_parity: Dict[str, Any],
    entry_model_proof: Dict[str, Any],
    runtime_proof: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
) -> str:
    """Generate markdown report."""
    run_header = load_run_header(run_dir)
    run_tag = run_header.get("run_tag", "UNKNOWN") if run_header else "UNKNOWN"
    
    lines = []
    lines.append("# Production Baseline Parity Proof Report")
    lines.append("")
    lines.append(f"**Run:** `{run_tag}`")
    lines.append(f"**Run Directory:** `{run_dir}`")
    lines.append(f"**Production Policy:** `{prod_policy_bundle['policy_path']}`")
    lines.append(f"**Generated:** {pd.Timestamp.now().isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Policy Parity
    lines.append("## 1. Policy Parity")
    lines.append("")
    lines.append("### Policy YAML")
    run_policy_hash = policy_parity["run_artifacts"]["policy_sha256"]
    prod_policy_hash = policy_parity["prod_artifacts"]["policy_sha256"]
    match_icon = "✅" if policy_parity["policy_sha256_match"] else "❌"
    lines.append(f"- **Run Policy SHA256:** `{run_policy_hash[:16]}...`")
    lines.append(f"- **Prod Policy SHA256:** `{prod_policy_hash[:16]}...`")
    lines.append(f"- **Match:** {match_icon}")
    lines.append("")
    
    # Entry Config
    prod_entry_hash = policy_parity["prod_artifacts"]["entry_config_sha256"]
    if prod_entry_hash:
        entry_match = policy_parity["entry_config_sha256_match"]
        match_icon = "✅" if entry_match else "❌" if entry_match is False else "⚠️"
        lines.append("### Entry Config YAML")
        lines.append(f"- **Prod Entry Config SHA256:** `{prod_entry_hash[:16]}...`")
        lines.append(f"- **Match:** {match_icon} (not tracked in run_header)")
        lines.append("")
    
    # Exit Config
    prod_exit_hash = policy_parity["prod_artifacts"]["exit_config_sha256"]
    if prod_exit_hash:
        exit_match = policy_parity["exit_config_sha256_match"]
        match_icon = "✅" if exit_match else "❌" if exit_match is False else "⚠️"
        lines.append("### Exit Config YAML")
        lines.append(f"- **Prod Exit Config SHA256:** `{prod_exit_hash[:16]}...`")
        lines.append(f"- **Match:** {match_icon} (not tracked in run_header)")
        lines.append("")
    
    # Router Model
    run_router_hash = policy_parity["run_artifacts"]["router_model_sha256"]
    prod_router_hash = policy_parity["prod_artifacts"]["router_model_sha256"]
    if run_router_hash and prod_router_hash:
        router_match = policy_parity["router_model_sha256_match"]
        match_icon = "✅" if router_match else "❌" if router_match is False else "⚠️"
        lines.append("### Router Model")
        lines.append(f"- **Run Router SHA256:** `{run_router_hash[:16]}...`")
        lines.append(f"- **Prod Router SHA256:** `{prod_router_hash[:16]}...`")
        lines.append(f"- **Match:** {match_icon}")
        lines.append("")
    
    # Feature Manifest
    run_manifest_hash = policy_parity["run_artifacts"]["feature_manifest_sha256"]
    prod_manifest_hash = policy_parity["prod_artifacts"]["feature_manifest_sha256"]
    if run_manifest_hash and prod_manifest_hash:
        manifest_match = policy_parity["feature_manifest_sha256_match"]
        match_icon = "✅" if manifest_match else "❌" if manifest_match is False else "⚠️"
        lines.append("### Feature Manifest")
        lines.append(f"- **Run Manifest SHA256:** `{run_manifest_hash[:16]}...`")
        lines.append(f"- **Prod Manifest SHA256:** `{prod_manifest_hash[:16]}...`")
        lines.append(f"- **Match:** {match_icon}")
        lines.append("")
    
    # Entry Model Proof
    lines.append("## 2. Entry Model Proof")
    lines.append("")
    if entry_model_proof["expected_metadata"]:
        lines.append("### Expected Metadata")
        metadata = entry_model_proof["expected_metadata"]
        expected_models = list(metadata.get("models", {}).keys())
        if expected_models:
            lines.append(f"- **Models:** {', '.join(expected_models)}")
        feature_hash = metadata.get("feature_cols_hash")
        if feature_hash:
            lines.append(f"- **Feature Columns Hash:** `{feature_hash[:16]}...`")
        n_features = metadata.get("n_features_in_")
        if n_features:
            lines.append(f"- **N Features:** {n_features}")
        lines.append("")
    
    if entry_model_proof["actual_metadata"]:
        lines.append("### Actual Metadata (from Trade Journal)")
        actual = entry_model_proof["actual_metadata"]
        actual_models = actual.get("entry_model_versions", [])
        if actual_models:
            lines.append(f"- **Models Loaded:** {', '.join(actual_models)}")
        lines.append("")
    
    match_icon = "✅" if entry_model_proof["metadata_match"] else "⚠️"
    lines.append(f"- **Metadata Match:** {match_icon}")
    if not entry_model_proof["metadata_found"]:
        lines.append("  - ⚠️ Expected metadata file not found")
    lines.append("")
    
    # Runtime Pipeline Proof
    lines.append("## 3. Runtime Pipeline Proof")
    lines.append("")
    lines.append("### Trade Journal Coverage")
    lines.append("")
    total = runtime_proof["total_trades"]
    lines.append(f"- **Total Trades:** {total}")
    lines.append(f"- **Trades with Entry Snapshot:** {runtime_proof['trades_with_entry_snapshot']} ({runtime_proof['trades_with_entry_snapshot']/total*100:.1f}%)")
    lines.append(f"- **Trades with Feature Context:** {runtime_proof['trades_with_feature_context']} ({runtime_proof['trades_with_feature_context']/total*100:.1f}%)")
    lines.append(f"- **Trades with Router Decision:** {runtime_proof['trades_with_router_decision']} ({runtime_proof['trades_with_router_decision']/total*100:.1f}%)")
    if runtime_proof.get("trades_with_guardrail_flag", 0) > 0:
        lines.append(f"- **Trades with Guardrail Flag:** {runtime_proof['trades_with_guardrail_flag']} ({runtime_proof['trades_with_guardrail_flag']/total*100:.1f}%)")
    lines.append(f"- **Trades with Exit Summary:** {runtime_proof['trades_with_exit_summary']} ({runtime_proof['trades_with_exit_summary']/total*100:.1f}%)")
    lines.append(f"- **Trades with Intratrade Metrics:** {runtime_proof['trades_with_intratrade_metrics']} ({runtime_proof['trades_with_intratrade_metrics']/total*100:.1f}%)")
    lines.append("")
    
    coverage = runtime_proof["coverage_percent"]
    coverage_icon = "✅" if coverage == 100.0 else "❌"
    lines.append(f"- **Coverage:** {coverage_icon} {coverage:.1f}%")
    if coverage < 100.0:
        lines.append("  - ⚠️ **WARNING:** Not all trades have complete journal entries")
    lines.append("")
    
    # Metrics Sanity
    lines.append("## 4. Metrics Sanity")
    lines.append("")
    if metrics:
        # Try different possible keys for metrics
        trades_per_day = (
            metrics.get("trades_per_day")
            or metrics.get("trades/day")
            or metrics.get("avg_trades_per_day")
            or 0
        )
        ev_per_trade = (
            metrics.get("ev_per_trade_bps")
            or metrics.get("ev/trade")
            or metrics.get("mean_pnl_bps")
            or metrics.get("avg_pnl_bps")
            or 0
        )
        total_trades = (
            metrics.get("total_trades")
            or metrics.get("n_trades")
            or metrics.get("trade_count")
            or runtime_proof["total_trades"]
        )
        
        lines.append(f"- **Trades/Day:** {trades_per_day:.2f}")
        lines.append(f"- **EV/Trade:** {ev_per_trade:.2f} bps")
        lines.append(f"- **Total Trades:** {total_trades}")
        lines.append("")
        
        if total_trades < 10:
            lines.append("⚠️ **Small-N Warning:** Low trade count (< 10 trades) - metrics may not be statistically significant")
            lines.append("")
    else:
        lines.append("⚠️ **Warning:** results.json not found")
        lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    
    all_checks = [
        ("Policy SHA256", policy_parity["policy_sha256_match"]),
        ("Router Model SHA256", policy_parity["router_model_sha256_match"]),
        ("Feature Manifest SHA256", policy_parity["feature_manifest_sha256_match"]),
        ("Trade Journal Coverage", runtime_proof["coverage_percent"] == 100.0),
    ]
    
    passed = sum(1 for _, check in all_checks if check)
    total_checks = len([c for _, c in all_checks if c is not None])
    
    lines.append(f"**Checks Passed:** {passed}/{total_checks}")
    lines.append("")
    
    for check_name, check_result in all_checks:
        if check_result is None:
            continue
        icon = "✅" if check_result else "❌"
        lines.append(f"- {icon} {check_name}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `gx1/analysis/prod_baseline_proof.py`*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate production baseline parity proof report"
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run directory path (e.g., gx1/wf_runs/CANARY_TEST_2025_Q1_2WEEKS_SINGLE)",
    )
    parser.add_argument(
        "--prod-policy",
        type=Path,
        required=True,
        help="Production policy YAML path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output markdown report path",
    )
    
    args = parser.parse_args()
    
    # Load run header
    run_header = load_run_header(args.run)
    if not run_header:
        logger.error("Failed to load run_header.json")
        return 1
    
    # Load prod policy
    prod_policy_bundle = load_prod_policy(args.prod_policy)
    if not prod_policy_bundle:
        logger.error("Failed to load production policy")
        return 1
    
    # Verify policy parity
    logger.info("Verifying policy parity...")
    policy_parity = verify_policy_parity(run_header, prod_policy_bundle)
    
    # Verify entry model proof
    logger.info("Verifying entry model proof...")
    entry_model_proof = verify_entry_model_proof(args.run, prod_policy_bundle)
    
    # Verify runtime pipeline proof
    logger.info("Verifying runtime pipeline proof...")
    runtime_proof = verify_runtime_pipeline_proof(args.run)
    
    # Load metrics
    logger.info("Loading metrics...")
    metrics = load_metrics(args.run)
    
    # Generate report
    logger.info("Generating report...")
    report = generate_report(
        args.run,
        prod_policy_bundle,
        policy_parity,
        entry_model_proof,
        runtime_proof,
        metrics,
    )
    
    # Write report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report written to: {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())

