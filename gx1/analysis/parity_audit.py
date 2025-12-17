#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parity Audit Tool.

Verifies that FULLYEAR, CANARY, and LIVE runs use identical:
- Effective config (policy, entry, exit, router)
- Entry model + feature set
- Price source (bid/ask vs mid vs OHLC)
- Warmup/state semantics
- Gate/throttle settings

Produces a comprehensive audit report with parity verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_run_header(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run_header.json."""
    header_path = run_dir / "run_header.json"
    if not header_path.exists():
        logger.warning(f"run_header.json not found in {run_dir}")
        return None
    
    try:
        with open(header_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load run_header.json: {e}")
        return None


def find_policy_file(run_dir: Path) -> Optional[Path]:
    """Find policy YAML file from run directory."""
    # Try run_header.json first
    header = load_run_header(run_dir)
    if header:
        artifacts = header.get("artifacts", {})
        policy_info = artifacts.get("policy", {})
        policy_path = policy_info.get("path")
        if policy_path:
            policy_path_obj = Path(policy_path)
            if policy_path_obj.exists():
                return policy_path_obj
            # Try relative to run_dir
            policy_path_obj = run_dir / policy_path
            if policy_path_obj.exists():
                return policy_path_obj
    
    # Try parallel_chunks
    chunks_dir = run_dir / "parallel_chunks"
    if chunks_dir.exists():
        chunk_policies = list(chunks_dir.glob("policy_chunk_*.yaml"))
        if chunk_policies:
            return chunk_policies[0]  # Use first chunk policy
    
    # Try logs
    log_files = list(run_dir.glob("*.log"))
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()
                # Look for policy path in logs
                match = re.search(r'policy[_-]?path[:\s=]+([^\s\n]+)', content, re.IGNORECASE)
                if match:
                    policy_path = Path(match.group(1))
                    if policy_path.exists():
                        return policy_path
        except Exception:
            continue
    
    return None


def load_yaml_config(path: Path) -> Optional[Dict[str, Any]]:
    """Load YAML config file."""
    if not path.exists():
        return None
    
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML from {path}: {e}")
        return None


def normalize_config_for_hash(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config for hashing by removing run-specific fields.
    
    Removes:
    - start/end dates
    - output paths
    - run tags
    - logging dirs
    """
    # Custom JSON encoder for dates and other non-serializable types
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):  # datetime, date
            return obj.isoformat()
        elif isinstance(obj, (Path, type(Path()))):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    try:
        normalized = json.loads(json.dumps(config, default=json_serializer))  # Deep copy
    except (TypeError, ValueError):
        # Fallback: manual deep copy with date handling
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
        for key in ["run_tag", "output_dir"]:
            if key in meta:
                del meta[key]
    
    # Normalize paths in config (make relative)
    def normalize_paths(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and ("/" in v or "\\" in v):
                    # Try to make relative
                    if Path(v).is_absolute():
                        # Keep relative paths as-is, but normalize separators
                        obj[k] = v.replace("\\", "/")
                else:
                    normalize_paths(v)
        elif isinstance(obj, list):
            for item in obj:
                normalize_paths(item)
    
    normalize_paths(normalized)
    
    return normalized


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


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA256 hash of normalized config."""
    normalized = normalize_config_for_hash(config)
    config_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def load_effective_config_bundle(run_dir: Path) -> Dict[str, Any]:
    """
    Load effective config bundle for a run.
    
    Returns:
        Dict with: policy, entry_config, exit_config, policy_hash, entry_config_hash, exit_config_hash
    """
    result = {
        "policy": None,
        "entry_config": None,
        "exit_config": None,
        "policy_path": None,
        "entry_config_path": None,
        "exit_config_path": None,
        "policy_hash": None,
        "entry_config_hash": None,
        "exit_config_hash": None,
    }
    
    # Find and load policy
    policy_path = find_policy_file(run_dir)
    if not policy_path:
        logger.warning(f"Could not find policy file for {run_dir}")
        return result
    
    result["policy_path"] = str(policy_path)
    policy = load_yaml_config(policy_path)
    if not policy:
        return result
    
    result["policy"] = policy
    result["policy_hash"] = compute_config_hash(policy)
    
    # Load entry config
    entry_config_path = policy.get("entry_config")
    if entry_config_path:
        entry_path = Path(entry_config_path)
        if not entry_path.is_absolute():
            # Try relative to policy path
            entry_path = policy_path.parent / entry_path
        
        if entry_path.exists():
            result["entry_config_path"] = str(entry_path)
            entry_config = load_yaml_config(entry_path)
            if entry_config:
                result["entry_config"] = entry_config
                result["entry_config_hash"] = compute_config_hash(entry_config)
    
    # Load exit config
    exit_config_path = policy.get("exit_config")
    if exit_config_path:
        exit_path = Path(exit_config_path)
        if not exit_path.is_absolute():
            exit_path = policy_path.parent / exit_path
        
        if exit_path.exists():
            result["exit_config_path"] = str(exit_path)
            exit_config = load_yaml_config(exit_path)
            if exit_config:
                result["exit_config"] = exit_config
                result["exit_config_hash"] = compute_config_hash(exit_config)
    
    return result


def extract_entry_model_truth(run_dir: Path) -> Dict[str, Any]:
    """Extract entry model truth from run artifacts."""
    result = {
        "entry_model_version": None,
        "feature_cols_hash": None,
        "n_features_in_": None,
        "model_paths": [],
        "session_metadata": None,
    }
    
    # Try to load from trade journal
    journal_index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if journal_index_path.exists():
        try:
            df = pd.read_csv(journal_index_path)
            # Check if we have entry model info in trade journals
            trade_json_dir = run_dir / "trade_journal" / "trades"
            if trade_json_dir.exists():
                trade_jsons = list(trade_json_dir.glob("*.json"))
                if trade_jsons:
                    # Load first trade JSON
                    with open(trade_jsons[0], "r") as f:
                        trade_data = json.load(f)
                        entry_snapshot = trade_data.get("entry_snapshot", {})
                        result["entry_model_version"] = entry_snapshot.get("entry_model_version")
        except Exception as e:
            logger.warning(f"Failed to extract from trade journal: {e}")
    
    # Try to load from session metadata
    metadata_path = Path("gx1/models/GX1_entry_session_metadata.json")
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                result["session_metadata"] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load session metadata: {e}")
    
    return result


def extract_router_truth(run_dir: Path, policy: Dict[str, Any]) -> Dict[str, Any]:
    """Extract router truth from run and policy."""
    result = {
        "router_version": None,
        "model_path": None,
        "model_hash": None,
        "guardrail_cutoff": None,
    }
    
    router_cfg = policy.get("hybrid_exit_router", {})
    result["router_version"] = router_cfg.get("version")
    result["model_path"] = router_cfg.get("model_path")
    result["guardrail_cutoff"] = router_cfg.get("v3_range_edge_cutoff")
    
    # Try to get model hash from run_header
    header = load_run_header(run_dir)
    if header:
        artifacts = header.get("artifacts", {})
        router_info = artifacts.get("router_model", {})
        result["model_hash"] = router_info.get("sha256")
    
    return result


def scan_price_source_logic() -> List[Dict[str, Any]]:
    """
    Scan codebase for price source selection logic.
    
    Returns list of findings with file, line, function, and description.
    """
    findings = []
    
    # Key patterns to search for with context
    patterns = [
        (r"mid_price|midPrice|\(bid.*\+.*ask.*\)\s*/\s*2", "mid price calculation"),
        (r"bid_high|ask_high|bid_low|ask_low|bid_close|ask_close", "bid/ask OHLC usage"),
        (r"use_mid|useMid", "mid price flag"),
        (r"price_ref\s*=", "price reference assignment"),
        (r"build_live_entry_features", "live entry features"),
        (r"range_hi|range_lo|_compute_range_features", "range feature calculation"),
        (r"_collect_price_trace|price_trace", "price trace collection"),
        (r"entry_price\s*=|exit_price\s*=", "entry/exit price assignment"),
        (r"spread_raw|spread_price|spread_pct", "spread calculation"),
        (r"candles\.iloc\[-1\]|candles\.iloc\[-2\]|\.tail\(|\.head\(", "bar close semantics"),
        (r"has_bid_ask|has_direct", "price source detection"),
        (r"close_vals\[-1\]|close_vals\.iloc\[-1\]", "last closed bar price"),
    ]
    
    # Search in key directories
    search_dirs = [
        Path("gx1/execution"),
        Path("gx1/features"),
        Path("gx1/policy"),
    ]
    
    cwd = Path.cwd()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Resolve to absolute path
        search_dir_abs = search_dir.resolve()
        
        for py_file in search_dir_abs.rglob("*.py"):
            try:
                # Try to get relative path
                try:
                    file_rel = str(py_file.relative_to(cwd))
                except ValueError:
                    # If not relative, use absolute
                    file_rel = str(py_file)
                
                with open(py_file, "r") as f:
                    lines = f.readlines()
                    for line_num, line in enumerate(lines, 1):
                        for pattern, description in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Get function context
                                func_name = "unknown"
                                for i in range(max(0, line_num - 20), line_num):
                                    if re.match(r"^\s*def\s+\w+", lines[i]):
                                        func_match = re.match(r"^\s*def\s+(\w+)", lines[i])
                                        if func_match:
                                            func_name = func_match.group(1)
                                        break
                                
                                findings.append({
                                    "file": file_rel,
                                    "line": line_num,
                                    "function": func_name,
                                    "code": line.strip()[:100],  # Truncate long lines
                                    "description": description,
                                    "pattern": pattern,
                                })
            except Exception as e:
                logger.warning(f"Failed to scan {py_file}: {e}")
    
    return findings


def analyze_price_source_contract(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze price source contract from findings.
    
    Determines what price source is used for:
    - Entry features
    - Spread calculation
    - Range features
    - Exit evaluation
    - Intratrade metrics
    """
    contract = {
        "entry_features": "UNKNOWN",
        "spread": "UNKNOWN",
        "range_features": "UNKNOWN",
        "exit_evaluation": "UNKNOWN",
        "intratrade_metrics": "UNKNOWN",
        "bar_close_semantics": "UNKNOWN",
        "mixed_sources": False,
    }
    
    # Analyze findings
    has_bid_ask_usage = any("bid_" in f.get("code", "") or "ask_" in f.get("code", "") for f in findings)
    has_direct_ohlc = any("'high'" in f.get("code", "") or "'low'" in f.get("code", "") or "'close'" in f.get("code", "") for f in findings)
    has_mid_calc = any("mid" in f.get("code", "").lower() or "(bid" in f.get("code", "") for f in findings)
    
    # Entry features
    entry_findings = [f for f in findings if "entry" in f.get("file", "").lower() or "entry" in f.get("function", "").lower()]
    if entry_findings:
        if any("bid_high" in f.get("code", "") for f in entry_findings):
            contract["entry_features"] = "bid/ask mid"
        elif any("'close'" in f.get("code", "") for f in entry_findings):
            contract["entry_features"] = "OHLC close"
    
    # Spread
    spread_findings = [f for f in findings if "spread" in f.get("description", "").lower()]
    if spread_findings:
        contract["spread"] = "bid/ask"  # Spread always uses bid/ask
    
    # Range features
    range_findings = [f for f in findings if "range" in f.get("description", "").lower() or "_compute_range" in f.get("function", "")]
    if range_findings:
        if any("bid_high" in f.get("code", "") for f in range_findings):
            contract["range_features"] = "bid/ask mid"
        elif any("'high'" in f.get("code", "") for f in range_findings):
            contract["range_features"] = "OHLC"
    
    # Bar close semantics
    close_findings = [f for f in findings if "iloc[-1]" in f.get("code", "") or "close_vals[-1]" in f.get("code", "")]
    if close_findings:
        contract["bar_close_semantics"] = "last closed bar (iloc[-1] or close_vals[-1])"
    
    # Check for mixed sources
    sources_used = set([contract["entry_features"], contract["range_features"]])
    sources_used.discard("UNKNOWN")
    if len(sources_used) > 1:
        contract["mixed_sources"] = True
    
    return contract


def analyze_price_data_parity(run_dir: Path) -> Dict[str, Any]:
    """Analyze price data parity (bid/ask vs OHLC)."""
    result = {
        "has_bid_ask": False,
        "has_ohlc": False,
        "sample_diffs": [],
        "price_data_hash": None,
    }
    
    # Try to find price_data_filtered.parquet
    price_data_path = run_dir / "price_data_filtered.parquet"
    if not price_data_path.exists():
        # Try parallel_chunks
        chunks_dir = run_dir / "parallel_chunks"
        if chunks_dir.exists():
            chunk_data = list(chunks_dir.glob("price_data_chunk_*.parquet"))
            if chunk_data:
                price_data_path = chunk_data[0]
    
    if price_data_path.exists():
        result["price_data_hash"] = compute_file_hash(price_data_path)
        
        try:
            df = pd.read_parquet(price_data_path)
            
            # Check columns
            result["has_bid_ask"] = all(col in df.columns for col in ['bid_close', 'ask_close'])
            result["has_ohlc"] = all(col in df.columns for col in ['high', 'low', 'close'])
            
            # Sample differences
            if result["has_bid_ask"] and result["has_ohlc"] and len(df) > 0:
                sample = df.head(5)
                for idx, row in sample.iterrows():
                    if pd.notna(row.get("close")) and pd.notna(row.get("bid_close")) and pd.notna(row.get("ask_close")):
                        mid_close = (row["bid_close"] + row["ask_close"]) / 2.0
                        diff = abs(row["close"] - mid_close)
                        result["sample_diffs"].append({
                            "timestamp": str(idx) if hasattr(idx, "isoformat") else str(idx),
                            "close": float(row["close"]),
                            "mid_close": float(mid_close),
                            "diff": float(diff),
                            "diff_bps": float((diff / row["close"]) * 10000) if row["close"] > 0 else 0.0,
                        })
        except Exception as e:
            logger.warning(f"Failed to analyze price data: {e}")
    
    return result


def extract_warmup_state(run_dir: Path, policy: Dict[str, Any]) -> Dict[str, Any]:
    """Extract warmup/state configuration."""
    result = {
        "warmup_bars": None,
        "warmup_floor": None,
        "n_workers": None,
        "parallel_chunking": False,
        "chunk_overlap": None,
    }
    
    # From policy
    result["warmup_bars"] = policy.get("warmup_bars")
    
    # From logs or run_header
    header = load_run_header(run_dir)
    if header:
        result["n_workers"] = header.get("meta", {}).get("n_workers")
    
    # Check for parallel chunks
    chunks_dir = run_dir / "parallel_chunks"
    if chunks_dir.exists():
        result["parallel_chunking"] = True
        # Try to infer overlap from chunk policies
        chunk_policies = list(chunks_dir.glob("policy_chunk_*.yaml"))
        if chunk_policies:
            # Check if chunks have overlapping warmup
            result["chunk_overlap"] = "unknown"  # Would need deeper analysis
    
    return result


def extract_gate_throttle_config(policy: Dict[str, Any], entry_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract gate and throttle configuration."""
    result = {
        "min_time_between_trades_sec": None,
        "max_open_trades": None,
        "session_allowlist": None,
        "regime_allowlist": None,
        "spread_filter_threshold": None,
        "live_only_latch": None,
    }
    
    # From policy
    result["max_open_trades"] = policy.get("max_open_trades")
    
    # From entry config
    if entry_config:
        gates = entry_config.get("gates", {})
        result["spread_filter_threshold"] = gates.get("max_spread_pct")
        result["session_allowlist"] = gates.get("session_allowlist")
        result["regime_allowlist"] = gates.get("regime_allowlist")
    
    # From execution config
    exec_cfg = policy.get("execution", {})
    result["min_time_between_trades_sec"] = exec_cfg.get("min_time_between_trades_sec")
    
    return result


def compute_trades_per_day(run_dir: Path, session_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute trades per day statistics.
    
    Args:
        run_dir: Run directory
        session_filter: Optional session filter (e.g., "ASIA")
    
    Returns:
        Dict with trades/day stats
    """
    result = {
        "total_trades": 0,
        "trades_per_day": None,
        "session_trades": 0,
        "session_trades_per_day": None,
        "period_days": None,
        "start_date": None,
        "end_date": None,
    }
    
    # Try trade journal index
    journal_index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if journal_index_path.exists():
        try:
            df = pd.read_csv(journal_index_path)
            result["total_trades"] = len(df)
            
            if len(df) > 0 and "entry_time" in df.columns:
                entry_times = pd.to_datetime(df["entry_time"], errors="coerce")
                result["start_date"] = entry_times.min().isoformat() if not entry_times.isna().all() else None
                result["end_date"] = entry_times.max().isoformat() if not entry_times.isna().all() else None
                
                if result["start_date"] and result["end_date"]:
                    start = pd.to_datetime(result["start_date"])
                    end = pd.to_datetime(result["end_date"])
                    days = (end - start).days + 1
                    result["period_days"] = days
                    if days > 0:
                        result["trades_per_day"] = result["total_trades"] / days
                
                # Filter by session if requested
                if session_filter and "session" in df.columns:
                    session_df = df[df["session"] == session_filter]
                    result["session_trades"] = len(session_df)
                    if result["period_days"] and result["period_days"] > 0:
                        result["session_trades_per_day"] = result["session_trades"] / result["period_days"]
        except Exception as e:
            logger.warning(f"Failed to compute trades/day from journal: {e}")
    
    # Fallback to trade_log_merged.csv
    if result["total_trades"] == 0:
        trade_log_path = run_dir / "trade_log_merged.csv"
        if trade_log_path.exists():
            try:
                df = pd.read_csv(trade_log_path)
                result["total_trades"] = len(df)
                # Similar date calculation...
            except Exception as e:
                logger.warning(f"Failed to compute trades/day from trade log: {e}")
    
    return result


def find_historical_claim(run_tag_pattern: str = "FULLYEAR") -> List[Dict[str, Any]]:
    """Find historical runs that might have produced '3 trades/day' claim."""
    findings = []
    
    wf_runs_dir = Path("gx1/wf_runs")
    if not wf_runs_dir.exists():
        return findings
    
    for run_dir in wf_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        if run_tag_pattern not in run_dir.name.upper():
            continue
        
        # Check results.json
        results_path = run_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    trades_per_day = results.get("trades_per_day")
                    if trades_per_day and 2.5 <= trades_per_day <= 3.5:
                        findings.append({
                            "run_tag": run_dir.name,
                            "trades_per_day": trades_per_day,
                            "source": "results.json",
                        })
            except Exception:
                pass
    
    return findings


def generate_parity_verdict(
    fullyear_data: Dict[str, Any],
    canary_data: Dict[str, Any],
    live_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate final parity verdict."""
    verdict = {
        "status": "UNKNOWN",
        "primary_cause": None,
        "secondary_causes": [],
        "details": {},
    }
    
    mismatches = []
    
    # Check config hashes
    if fullyear_data.get("config", {}).get("policy_hash") != canary_data.get("config", {}).get("policy_hash"):
        mismatches.append("Policy config hash mismatch")
    
    if fullyear_data.get("config", {}).get("entry_config_hash") != canary_data.get("config", {}).get("entry_config_hash"):
        mismatches.append("Entry config hash mismatch")
    
    if fullyear_data.get("config", {}).get("exit_config_hash") != canary_data.get("config", {}).get("exit_config_hash"):
        mismatches.append("Exit config hash mismatch")
    
    # Check router
    if fullyear_data.get("router", {}).get("model_hash") != canary_data.get("router", {}).get("model_hash"):
        mismatches.append("Router model hash mismatch")
    
    # Check warmup
    if fullyear_data.get("warmup", {}).get("warmup_bars") != canary_data.get("warmup", {}).get("warmup_bars"):
        mismatches.append("Warmup bars mismatch")
    
    # Check gates
    fullyear_gates = fullyear_data.get("gates", {})
    canary_gates = canary_data.get("gates", {})
    for key in ["max_open_trades", "spread_filter_threshold"]:
        if fullyear_gates.get(key) != canary_gates.get(key):
            mismatches.append(f"Gate {key} mismatch: {fullyear_gates.get(key)} vs {canary_gates.get(key)}")
    
    if len(mismatches) == 0:
        verdict["status"] = "FULL PARITY"
    else:
        verdict["status"] = "NOT PARITY"
        verdict["primary_cause"] = mismatches[0] if mismatches else "Unknown"
        verdict["secondary_causes"] = mismatches[1:4]  # Max 3 secondary
    
    return verdict


def generate_report(
    fullyear_data: Dict[str, Any],
    canary_data: Dict[str, Any],
    live_data: Optional[Dict[str, Any]],
    price_findings: List[Dict[str, Any]],
    price_contract: Dict[str, Any],
    fullyear_price_data: Dict[str, Any],
    canary_price_data: Dict[str, Any],
    live_price_data: Optional[Dict[str, Any]],
    verdict: Dict[str, Any],
    historical_claims: List[Dict[str, Any]],
) -> str:
    """Generate markdown audit report."""
    lines = []
    
    lines.append("# Parity Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    
    # Verdict
    status_icon = "✅" if verdict["status"] == "FULL PARITY" else "❌"
    lines.append(f"## Verdict: {status_icon} {verdict['status']}")
    lines.append("")
    
    if verdict["primary_cause"]:
        lines.append(f"**Primary Cause:** {verdict['primary_cause']}")
        lines.append("")
    
    if verdict["secondary_causes"]:
        lines.append("**Secondary Causes:**")
        for cause in verdict["secondary_causes"]:
            lines.append(f"- {cause}")
        lines.append("")
    
    # Run information
    lines.append("## Run Information")
    lines.append("")
    lines.append(f"- **FULLYEAR Run:** `{fullyear_data.get('run_dir', 'N/A')}`")
    lines.append(f"- **CANARY Run:** `{canary_data.get('run_dir', 'N/A')}`")
    if live_data:
        lines.append(f"- **LIVE Run:** `{live_data.get('run_dir', 'N/A')}`")
    lines.append("")
    
    # Config parity
    lines.append("## Config Parity")
    lines.append("")
    lines.append("| Config Item | FULLYEAR | CANARY | Match |")
    lines.append("|-------------|----------|--------|-------|")
    
    fullyear_config = fullyear_data.get("config", {})
    canary_config = canary_data.get("config", {})
    
    fullyear_policy_hash = fullyear_config.get("policy_hash") or "N/A"
    canary_policy_hash = canary_config.get("policy_hash") or "N/A"
    policy_match = "✅" if fullyear_policy_hash == canary_policy_hash and fullyear_policy_hash != "N/A" else "❌"
    fullyear_hash_str = f"{fullyear_policy_hash[:16]}..." if fullyear_policy_hash != "N/A" else "N/A"
    canary_hash_str = f"{canary_policy_hash[:16]}..." if canary_policy_hash != "N/A" else "N/A"
    lines.append(f"| Policy Hash | `{fullyear_hash_str}` | `{canary_hash_str}` | {policy_match} |")
    
    fullyear_entry_hash = fullyear_config.get("entry_config_hash") or "N/A"
    canary_entry_hash = canary_config.get("entry_config_hash") or "N/A"
    entry_match = "✅" if fullyear_entry_hash == canary_entry_hash and fullyear_entry_hash != "N/A" else "❌"
    fullyear_entry_str = f"{fullyear_entry_hash[:16]}..." if fullyear_entry_hash != "N/A" else "N/A"
    canary_entry_str = f"{canary_entry_hash[:16]}..." if canary_entry_hash != "N/A" else "N/A"
    lines.append(f"| Entry Config Hash | `{fullyear_entry_str}` | `{canary_entry_str}` | {entry_match} |")
    
    fullyear_exit_hash = fullyear_config.get("exit_config_hash") or "N/A"
    canary_exit_hash = canary_config.get("exit_config_hash") or "N/A"
    exit_match = "✅" if fullyear_exit_hash == canary_exit_hash and fullyear_exit_hash != "N/A" else "❌"
    fullyear_exit_str = f"{fullyear_exit_hash[:16]}..." if fullyear_exit_hash != "N/A" else "N/A"
    canary_exit_str = f"{canary_exit_hash[:16]}..." if canary_exit_hash != "N/A" else "N/A"
    lines.append(f"| Exit Config Hash | `{fullyear_exit_str}` | `{canary_exit_str}` | {exit_match} |")
    
    lines.append("")
    
    # Price source audit
    lines.append("## Price Source Audit (P0)")
    lines.append("")
    
    lines.append("### Effective Price Source Contract")
    lines.append("")
    lines.append("| Component | Price Source |")
    lines.append("|-----------|--------------|")
    lines.append(f"| Entry Features | {price_contract.get('entry_features', 'UNKNOWN')} |")
    lines.append(f"| Spread Calculation | {price_contract.get('spread', 'UNKNOWN')} |")
    lines.append(f"| Range Features | {price_contract.get('range_features', 'UNKNOWN')} |")
    lines.append(f"| Exit Evaluation | {price_contract.get('exit_evaluation', 'UNKNOWN')} |")
    lines.append(f"| Intratrade Metrics | {price_contract.get('intratrade_metrics', 'UNKNOWN')} |")
    lines.append(f"| Bar Close Semantics | {price_contract.get('bar_close_semantics', 'UNKNOWN')} |")
    lines.append("")
    
    if price_contract.get("mixed_sources"):
        lines.append("⚠️ **WARNING:** Mixed price sources detected! This may cause parity issues.")
        lines.append("")
    
    lines.append("### Price Data Parity")
    lines.append("")
    lines.append("| Run | Has Bid/Ask | Has OHLC | Price Data Hash |")
    lines.append("|-----|-------------|----------|-----------------|")
    fullyear_hash = fullyear_price_data.get("price_data_hash") or "N/A"
    canary_hash = canary_price_data.get("price_data_hash") or "N/A"
    fullyear_hash_str = f"{fullyear_hash[:16]}..." if fullyear_hash != "N/A" else "N/A"
    canary_hash_str = f"{canary_hash[:16]}..." if canary_hash != "N/A" else "N/A"
    lines.append(f"| FULLYEAR | {'✅' if fullyear_price_data.get('has_bid_ask') else '❌'} | {'✅' if fullyear_price_data.get('has_ohlc') else '❌'} | `{fullyear_hash_str}` |")
    lines.append(f"| CANARY | {'✅' if canary_price_data.get('has_bid_ask') else '❌'} | {'✅' if canary_price_data.get('has_ohlc') else '❌'} | `{canary_hash_str}` |")
    if live_price_data:
        live_hash = live_price_data.get("price_data_hash") or "N/A"
        live_hash_str = f"{live_hash[:16]}..." if live_hash != "N/A" else "N/A"
        lines.append(f"| LIVE | {'✅' if live_price_data.get('has_bid_ask') else '❌'} | {'✅' if live_price_data.get('has_ohlc') else '❌'} | `{live_hash_str}` |")
    lines.append("")
    
    # Sample price differences
    if fullyear_price_data.get("sample_diffs"):
        lines.append("### Price Data Sample Differences (OHLC close vs bid/ask mid)")
        lines.append("")
        lines.append("| Timestamp | OHLC Close | Mid Close | Diff | Diff (bps) |")
        lines.append("|-----------|------------|-----------|------|------------|")
        for diff in fullyear_price_data["sample_diffs"][:5]:
            ts = diff.get("timestamp", "N/A")[:19]  # Truncate timestamp
            lines.append(f"| {ts} | {diff.get('close', 0):.5f} | {diff.get('mid_close', 0):.5f} | {diff.get('diff', 0):.5f} | {diff.get('diff_bps', 0):.2f} |")
        lines.append("")
    
    lines.append("### Price Selection Logic Findings")
    lines.append("")
    if price_findings:
        lines.append("| File | Line | Function | Description | Code Snippet |")
        lines.append("|------|------|----------|-------------|--------------|")
        for finding in price_findings[:30]:  # Limit to first 30
            file = finding.get("file", "N/A")
            line = finding.get("line", "N/A")
            func = finding.get("function", "N/A")
            desc = finding.get("description", "N/A")
            code = finding.get("code", "N/A")[:80]  # Truncate
            lines.append(f"| `{file}` | {line} | `{func}` | {desc} | `{code}...` |")
    else:
        lines.append("No price selection logic found.")
    lines.append("")
    
    # Warmup/State parity
    lines.append("## Warmup/State Parity (P0)")
    lines.append("")
    lines.append("| Setting | FULLYEAR | CANARY | Match |")
    lines.append("|---------|----------|--------|-------|")
    
    fullyear_warmup = fullyear_data.get("warmup", {})
    canary_warmup = canary_data.get("warmup", {})
    
    warmup_match = "✅" if fullyear_warmup.get("warmup_bars") == canary_warmup.get("warmup_bars") else "❌"
    lines.append(f"| Warmup Bars | {fullyear_warmup.get('warmup_bars', 'N/A')} | {canary_warmup.get('warmup_bars', 'N/A')} | {warmup_match} |")
    
    n_workers_match = "✅" if fullyear_warmup.get("n_workers") == canary_warmup.get("n_workers") else "❌"
    lines.append(f"| N Workers | {fullyear_warmup.get('n_workers', 'N/A')} | {canary_warmup.get('n_workers', 'N/A')} | {n_workers_match} |")
    
    parallel_match = "✅" if fullyear_warmup.get("parallel_chunking") == canary_warmup.get("parallel_chunking") else "❌"
    lines.append(f"| Parallel Chunking | {fullyear_warmup.get('parallel_chunking', 'N/A')} | {canary_warmup.get('parallel_chunking', 'N/A')} | {parallel_match} |")
    lines.append("")
    
    if fullyear_warmup.get("parallel_chunking") and not canary_warmup.get("parallel_chunking"):
        lines.append("⚠️ **WARNING:** FULLYEAR uses parallel chunking but CANARY does not. This may cause parity issues.")
        lines.append("")
    
    # Gate/Throttle parity
    lines.append("## Gate & Throttle Parity (P0)")
    lines.append("")
    lines.append("| Gate/Throttle | FULLYEAR | CANARY | Match |")
    lines.append("|---------------|----------|--------|-------|")
    
    fullyear_gates = fullyear_data.get("gates", {})
    canary_gates = canary_data.get("gates", {})
    
    gate_keys = ["max_open_trades", "spread_filter_threshold", "min_time_between_trades_sec", "session_allowlist", "regime_allowlist"]
    for key in gate_keys:
        fullyear_val = fullyear_gates.get(key, "N/A")
        canary_val = canary_gates.get(key, "N/A")
        match = "✅" if fullyear_val == canary_val else "❌"
        lines.append(f"| {key} | {fullyear_val} | {canary_val} | {match} |")
    lines.append("")
    
    # Trades per day
    lines.append("## Trades Per Day Analysis (P0)")
    lines.append("")
    fullyear_trades = fullyear_data.get("trades_per_day", {})
    canary_trades = canary_data.get("trades_per_day", {})
    
    lines.append(f"### FULLYEAR")
    lines.append(f"- Total Trades: {fullyear_trades.get('total_trades', 'N/A')}")
    if fullyear_trades.get('trades_per_day'):
        lines.append(f"- Trades/Day: {fullyear_trades.get('trades_per_day'):.2f}")
    else:
        lines.append("- Trades/Day: N/A")
    if fullyear_trades.get('session_trades_per_day'):
        lines.append(f"- ASIA Trades/Day: {fullyear_trades.get('session_trades_per_day'):.2f}")
    else:
        lines.append("- ASIA Trades/Day: N/A")
    if fullyear_trades.get('start_date') and fullyear_trades.get('end_date'):
        lines.append(f"- Period: {fullyear_trades.get('start_date')} to {fullyear_trades.get('end_date')}")
    lines.append("")
    
    lines.append(f"### CANARY")
    lines.append(f"- Total Trades: {canary_trades.get('total_trades', 'N/A')}")
    if canary_trades.get('trades_per_day'):
        lines.append(f"- Trades/Day: {canary_trades.get('trades_per_day'):.2f}")
    else:
        lines.append("- Trades/Day: N/A")
    if canary_trades.get('start_date') and canary_trades.get('end_date'):
        lines.append(f"- Period: {canary_trades.get('start_date')} to {canary_trades.get('end_date')}")
    lines.append("")
    
    # Entry model parity
    lines.append("## Entry Model Parity")
    lines.append("")
    fullyear_entry = fullyear_data.get("entry_model", {})
    canary_entry = canary_data.get("entry_model", {})
    
    entry_model_match = "✅" if fullyear_entry.get("entry_model_version") == canary_entry.get("entry_model_version") else "❌"
    lines.append(f"- Entry Model Version Match: {entry_model_match}")
    lines.append(f"  - FULLYEAR: {fullyear_entry.get('entry_model_version', 'N/A')}")
    lines.append(f"  - CANARY: {canary_entry.get('entry_model_version', 'N/A')}")
    lines.append("")
    
    # Router parity
    lines.append("## Router Parity")
    lines.append("")
    fullyear_router = fullyear_data.get("router", {})
    canary_router = canary_data.get("router", {})
    
    router_version_match = "✅" if fullyear_router.get("router_version") == canary_router.get("router_version") else "❌"
    lines.append(f"- Router Version Match: {router_version_match}")
    lines.append(f"  - FULLYEAR: {fullyear_router.get('router_version', 'N/A')}")
    lines.append(f"  - CANARY: {canary_router.get('router_version', 'N/A')}")
    lines.append("")
    
    fullyear_router_hash = fullyear_router.get("model_hash") or "N/A"
    canary_router_hash = canary_router.get("model_hash") or "N/A"
    router_hash_match = "✅" if fullyear_router_hash == canary_router_hash and fullyear_router_hash != "N/A" else "❌"
    lines.append(f"- Router Model Hash Match: {router_hash_match}")
    fullyear_router_str = f"{fullyear_router_hash[:16]}..." if fullyear_router_hash != "N/A" else "N/A"
    canary_router_str = f"{canary_router_hash[:16]}..." if canary_router_hash != "N/A" else "N/A"
    lines.append(f"  - FULLYEAR: `{fullyear_router_str}`")
    lines.append(f"  - CANARY: `{canary_router_str}`")
    lines.append("")
    
    guardrail_match = "✅" if fullyear_router.get("guardrail_cutoff") == canary_router.get("guardrail_cutoff") else "❌"
    lines.append(f"- Guardrail Cutoff Match: {guardrail_match}")
    lines.append(f"  - FULLYEAR: {fullyear_router.get('guardrail_cutoff', 'N/A')}")
    lines.append(f"  - CANARY: {canary_router.get('guardrail_cutoff', 'N/A')}")
    lines.append("")
    
    # Historical claims
    if historical_claims:
        lines.append("## Where Did '3 Trades/Day in Asia' Come From?")
        lines.append("")
        for claim in historical_claims:
            run_tag = claim.get('run_tag', 'N/A')
            trades_per_day = claim.get('trades_per_day', 'N/A')
            lines.append(f"### Run: `{run_tag}`")
            lines.append(f"- **Trades/Day:** {trades_per_day}")
            lines.append(f"- **Source:** {claim.get('source', 'N/A')}")
            
            # Try to get more details from run
            run_dir = Path("gx1/wf_runs") / run_tag
            if run_dir.exists():
                header = load_run_header(run_dir)
                if header:
                    meta = header.get("meta", {})
                    lines.append(f"- **N Workers:** {meta.get('n_workers', 'N/A')}")
                    lines.append(f"- **Period:** {meta.get('start_date', 'N/A')} to {meta.get('end_date', 'N/A')}")
                
                # Get policy hash
                config_bundle = load_effective_config_bundle(run_dir)
                if config_bundle.get("policy_hash"):
                    lines.append(f"- **Policy Hash:** `{config_bundle.get('policy_hash', 'N/A')[:16]}...`")
            
            lines.append("")
    else:
        lines.append("## Where Did '3 Trades/Day in Asia' Come From?")
        lines.append("")
        lines.append("No historical runs found with ~3 trades/day. The claim may be:")
        lines.append("- From a different run tag pattern")
        lines.append("- From manual calculation")
        lines.append("- From a different time period")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `gx1/analysis/parity_audit.py`*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parity Audit Tool - Verify run parity"
    )
    parser.add_argument(
        "--fullyear-run",
        type=Path,
        required=True,
        help="FULLYEAR run directory",
    )
    parser.add_argument(
        "--canary-run",
        type=Path,
        required=True,
        help="CANARY run directory",
    )
    parser.add_argument(
        "--live-run",
        type=Path,
        default=None,
        help="LIVE/PRACTICE run directory (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gx1/analysis/parity_audit_report.md"),
        help="Output report path",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting parity audit...")
    
    # Extract data for each run
    fullyear_data = {
        "run_dir": str(args.fullyear_run),
        "config": load_effective_config_bundle(args.fullyear_run),
        "entry_model": extract_entry_model_truth(args.fullyear_run),
        "router": extract_router_truth(args.fullyear_run, load_effective_config_bundle(args.fullyear_run).get("policy", {})),
        "warmup": extract_warmup_state(args.fullyear_run, load_effective_config_bundle(args.fullyear_run).get("policy", {})),
        "gates": extract_gate_throttle_config(
            load_effective_config_bundle(args.fullyear_run).get("policy", {}),
            load_effective_config_bundle(args.fullyear_run).get("entry_config"),
        ),
        "trades_per_day": compute_trades_per_day(args.fullyear_run, session_filter="ASIA"),
    }
    
    canary_data = {
        "run_dir": str(args.canary_run),
        "config": load_effective_config_bundle(args.canary_run),
        "entry_model": extract_entry_model_truth(args.canary_run),
        "router": extract_router_truth(args.canary_run, load_effective_config_bundle(args.canary_run).get("policy", {})),
        "warmup": extract_warmup_state(args.canary_run, load_effective_config_bundle(args.canary_run).get("policy", {})),
        "gates": extract_gate_throttle_config(
            load_effective_config_bundle(args.canary_run).get("policy", {}),
            load_effective_config_bundle(args.canary_run).get("entry_config"),
        ),
        "trades_per_day": compute_trades_per_day(args.canary_run),
    }
    
    live_data = None
    if args.live_run:
        live_data = {
            "run_dir": str(args.live_run),
            "config": load_effective_config_bundle(args.live_run),
            "entry_model": extract_entry_model_truth(args.live_run),
            "router": extract_router_truth(args.live_run, load_effective_config_bundle(args.live_run).get("policy", {})),
            "warmup": extract_warmup_state(args.live_run, load_effective_config_bundle(args.live_run).get("policy", {})),
            "gates": extract_gate_throttle_config(
                load_effective_config_bundle(args.live_run).get("policy", {}),
                load_effective_config_bundle(args.live_run).get("entry_config"),
            ),
            "trades_per_day": compute_trades_per_day(args.live_run),
        }
    
    # Scan price source logic
    logger.info("Scanning price source logic...")
    price_findings = scan_price_source_logic()
    price_contract = analyze_price_source_contract(price_findings)
    
    # Analyze price data parity
    logger.info("Analyzing price data parity...")
    fullyear_price_data = analyze_price_data_parity(args.fullyear_run)
    canary_price_data = analyze_price_data_parity(args.canary_run)
    if args.live_run:
        live_price_data = analyze_price_data_parity(args.live_run)
    else:
        live_price_data = None
    
    # Find historical claims
    historical_claims = find_historical_claim()
    
    # Generate verdict
    verdict = generate_parity_verdict(fullyear_data, canary_data, live_data)
    
    # Generate report
    logger.info("Generating report...")
    report = generate_report(
        fullyear_data,
        canary_data,
        live_data,
        price_findings,
        price_contract,
        fullyear_price_data,
        canary_price_data,
        live_price_data,
        verdict,
        historical_claims,
    )
    
    # Write report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report written to: {args.out}")
    
    # Also write JSON
    json_out = args.out.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "fullyear": fullyear_data,
            "canary": canary_data,
            "live": live_data,
            "verdict": verdict,
            "price_findings": price_findings,
            "price_contract": price_contract,
            "fullyear_price_data": fullyear_price_data,
            "canary_price_data": canary_price_data,
            "live_price_data": live_price_data,
            "historical_claims": historical_claims,
        }, f, indent=2, default=str)
    
    logger.info(f"JSON data written to: {json_out}")
    
    # Write CSV findings
    csv_out = args.out.with_suffix(".csv").with_name(args.out.stem + "_findings.csv")
    if price_findings:
        findings_df = pd.DataFrame(price_findings)
        findings_df.to_csv(csv_out, index=False)
        logger.info(f"CSV findings written to: {csv_out}")
    
    # Write mismatch CSV
    mismatches = []
    if fullyear_data.get("config", {}).get("policy_hash") != canary_data.get("config", {}).get("policy_hash"):
        mismatches.append({"category": "config", "item": "policy_hash", "fullyear": fullyear_data.get("config", {}).get("policy_hash"), "canary": canary_data.get("config", {}).get("policy_hash")})
    if fullyear_data.get("config", {}).get("entry_config_hash") != canary_data.get("config", {}).get("entry_config_hash"):
        mismatches.append({"category": "config", "item": "entry_config_hash", "fullyear": fullyear_data.get("config", {}).get("entry_config_hash"), "canary": canary_data.get("config", {}).get("entry_config_hash")})
    if fullyear_data.get("warmup", {}).get("warmup_bars") != canary_data.get("warmup", {}).get("warmup_bars"):
        mismatches.append({"category": "warmup", "item": "warmup_bars", "fullyear": fullyear_data.get("warmup", {}).get("warmup_bars"), "canary": canary_data.get("warmup", {}).get("warmup_bars")})
    
    if mismatches:
        mismatch_df = pd.DataFrame(mismatches)
        mismatch_csv = args.out.with_suffix(".csv").with_name(args.out.stem + "_mismatches.csv")
        mismatch_df.to_csv(mismatch_csv, index=False)
        logger.info(f"Mismatch CSV written to: {mismatch_csv}")
    
    return 0


if __name__ == "__main__":
    exit(main())

