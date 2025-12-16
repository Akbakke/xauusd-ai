#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff two GX1 runs to identify configuration differences.

Compares:
- Policy YAML files
- Entry/exit configs
- Runtime settings
- Model paths
- Effective configuration (including runtime overrides)

Usage:
    python gx1/analysis/diff_runs.py \
        --baseline-run gx1/wf_runs/BASELINE_TAG \
        --canary-run gx1/wf_runs/CANARY_TAG \
        --out gx1/wf_runs/CANARY_TAG/run_diff_report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load YAML from {path}: {e}")
        raise


def find_policy_file(run_dir: Path) -> Optional[Path]:
    """
    Find policy YAML file used for this run.
    
    Checks:
    1. run_header.json -> artifacts.policy.path
    2. parallel_chunks/policy_chunk_*.yaml
    3. logs/*.log for policy path references
    """
    # Check run_header.json
    run_header = run_dir / "run_header.json"
    if run_header.exists():
        try:
            with open(run_header) as f:
                header = json.load(f)
                policy_path = header.get("artifacts", {}).get("policy", {}).get("path")
                if policy_path:
                    policy_path_obj = Path(policy_path)
                    if policy_path_obj.exists():
                        return policy_path_obj
                    # Try relative to run_dir
                    rel_path = run_dir.parent.parent / policy_path_obj
                    if rel_path.exists():
                        return rel_path
        except Exception as e:
            logger.warning(f"Could not read policy path from run_header.json: {e}")
    
    # Check parallel_chunks for policy files
    chunks_dir = run_dir / "parallel_chunks"
    if chunks_dir.exists():
        policy_chunks = list(chunks_dir.glob("policy_chunk_*.yaml"))
        if policy_chunks:
            # Use first chunk policy (should be same across chunks)
            return policy_chunks[0]
    
    # Try to find policy in logs
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Look for policy path patterns
                    matches = re.findall(r'policy[_\s]*path[:\s=]+([^\s\n]+\.yaml)', content, re.IGNORECASE)
                    for match in matches:
                        policy_path = Path(match)
                        if policy_path.exists():
                            return policy_path
                        # Try relative paths
                        rel_path = run_dir.parent.parent / policy_path
                        if rel_path.exists():
                            return rel_path
            except Exception:
                continue
    
    return None


def load_policy_bundle(policy_yaml_path: Path) -> Dict[str, Any]:
    """
    Load policy bundle including entry_config and exit_config.
    
    Returns effective config dict with all configs inlined.
    """
    policy = load_yaml_config(policy_yaml_path)
    effective_config = {
        "policy_path": str(policy_yaml_path.resolve()),
        "policy": policy,
    }
    
    # Load entry_config if specified
    entry_config_path_str = policy.get("entry_config")
    if entry_config_path_str:
        entry_config_path = Path(entry_config_path_str)
        if not entry_config_path.is_absolute():
            # Try multiple locations:
            # 1. Relative to policy file
            # 2. Relative to repo root (if policy is in parallel_chunks)
            # 3. Relative to current working directory
            candidates = [
                policy_yaml_path.parent / entry_config_path,
                Path.cwd() / entry_config_path,
            ]
            # If policy is in parallel_chunks, try repo root
            if "parallel_chunks" in str(policy_yaml_path):
                repo_root = policy_yaml_path.parent.parent.parent
                candidates.insert(0, repo_root / entry_config_path)
            
            entry_config_path = None
            for candidate in candidates:
                if candidate.exists():
                    entry_config_path = candidate
                    break
            
            if entry_config_path:
                effective_config["entry_config"] = load_yaml_config(entry_config_path)
                effective_config["entry_config_path"] = str(entry_config_path.resolve())
            else:
                logger.warning(f"Entry config not found: {entry_config_path_str} (tried: {candidates})")
    
    # Load exit_config if specified
    exit_config_path_str = policy.get("exit_config")
    if exit_config_path_str:
        exit_config_path = Path(exit_config_path_str)
        if not exit_config_path.is_absolute():
            # Try multiple locations:
            # 1. Relative to policy file
            # 2. Relative to repo root (if policy is in parallel_chunks)
            # 3. Relative to current working directory
            candidates = [
                policy_yaml_path.parent / exit_config_path,
                Path.cwd() / exit_config_path,
            ]
            # If policy is in parallel_chunks, try repo root
            if "parallel_chunks" in str(policy_yaml_path):
                repo_root = policy_yaml_path.parent.parent.parent
                candidates.insert(0, repo_root / exit_config_path)
            
            exit_config_path_resolved = None
            for candidate in candidates:
                if candidate.exists():
                    exit_config_path_resolved = candidate
                    break
            
            if exit_config_path_resolved:
                effective_config["exit_config"] = load_yaml_config(exit_config_path_resolved)
                effective_config["exit_config_path"] = str(exit_config_path_resolved.resolve())
            else:
                logger.warning(f"Exit config not found: {exit_config_path_str} (tried: {candidates})")
    
    return effective_config


def normalize_config_for_diff(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config for comparison (remove paths, dates, etc.).
    
    Keeps structure but removes:
    - Absolute paths (normalize to relative)
    - Dates/timestamps
    - Output directories
    - Run IDs
    """
    normalized = {}
    
    # Keys that should be ignored in diff (allowed differences)
    ignore_keys = {
        "output_dir", "log_dir", "run_id", "start_date", "end_date",
        "policy_path", "entry_config_path", "exit_config_path",
    }
    
    for key, value in config.items():
        if key in ignore_keys:
            continue
        
        if isinstance(value, dict):
            normalized[key] = normalize_config_for_diff(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                normalize_config_for_diff(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str) and Path(value).is_absolute():
            # Normalize absolute paths to relative
            try:
                rel_path = Path(value).relative_to(Path.cwd())
                normalized[key] = f"<RELATIVE_PATH>/{rel_path}"
            except ValueError:
                normalized[key] = f"<ABSOLUTE_PATH>/{Path(value).name}"
        else:
            normalized[key] = value
    
    return normalized


def categorize_diff(key_path: str) -> str:
    """
    Categorize diff as ENTRY-driving or EXIT-driving.
    
    Returns: "ENTRY", "EXIT", or "OTHER"
    """
    entry_keywords = {
        "entry_config", "entry_model", "warmup", "spread", "atr",
        "session", "regime", "min_margin", "cooldown", "max_open_trades",
        "entry_threshold", "entry_gate", "entry_policy",
    }
    exit_keywords = {
        "exit_config", "exit_policy", "router", "guardrail", "v3_range_edge",
        "rule5", "rule6a", "hybrid_exit", "exit_profile",
    }
    
    key_lower = key_path.lower()
    if any(kw in key_lower for kw in entry_keywords):
        return "ENTRY"
    elif any(kw in key_lower for kw in exit_keywords):
        return "EXIT"
    else:
        return "OTHER"


def diff_dicts(
    baseline: Dict[str, Any],
    canary: Dict[str, Any],
    path: str = "",
    allowed_diffs: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, Any, Any]], Dict[str, List[Tuple[str, Any, Any]]]]:
    """
    Recursively diff two dictionaries.
    
    Returns list of (key_path, baseline_value, canary_value) tuples.
    """
    if allowed_diffs is None:
        allowed_diffs = {
            "output_dir", "log_dir", "run_id", "start_date", "end_date",
            "mode", "dry_run", "policy_path", "entry_config_path", "exit_config_path",
        }
    
    differences = []
    categorized = {"ENTRY": [], "EXIT": [], "OTHER": []}
    all_keys = set(baseline.keys()) | set(canary.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key in allowed_diffs:
            continue  # Skip allowed differences
        
        baseline_val = baseline.get(key)
        canary_val = canary.get(key)
        
        diff_tuple = None
        if key not in baseline:
            diff_tuple = (current_path, "<MISSING>", canary_val)
        elif key not in canary:
            diff_tuple = (current_path, baseline_val, "<MISSING>")
        elif isinstance(baseline_val, dict) and isinstance(canary_val, dict):
            sub_diffs, sub_categorized = diff_dicts(baseline_val, canary_val, current_path, allowed_diffs)
            differences.extend(sub_diffs)
            for cat, cat_diffs in sub_categorized.items():
                categorized[cat].extend(cat_diffs)
        elif baseline_val != canary_val:
            diff_tuple = (current_path, baseline_val, canary_val)
        
        if diff_tuple:
            differences.append(diff_tuple)
            category = categorize_diff(current_path)
            categorized[category].append(diff_tuple)
    
    return differences, categorized


def extract_runtime_overrides(run_dir: Path) -> Dict[str, Any]:
    """
    Extract runtime overrides from logs.
    
    Looks for:
    - dry_run, CANARY mode
    - max_open_trades
    - warmup_bars
    - Entry thresholds/gates
    - Exit settings
    """
    overrides = {}
    logs_dir = run_dir / "logs"
    
    if not logs_dir.exists():
        return overrides
    
    # Patterns to search for
    patterns = {
        "dry_run": r"dry_run[:\s=]+(True|False)",
        "mode": r"mode[:\s=]+(\w+)",
        "max_open_trades": r"max_open_trades[:\s=]+(\d+)",
        "warmup_bars": r"warmup[_\s]*bars[:\s=]+(\d+)",
        "canary_mode": r"CANARY.*mode",
    }
    
    for log_file in logs_dir.glob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                for key, pattern in patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        if key == "canary_mode":
                            overrides["canary_mode"] = True
                        elif key == "dry_run":
                            overrides["dry_run"] = matches[-1].lower() == "true"
                        elif key == "mode":
                            overrides["mode"] = matches[-1]
                        elif key in ("max_open_trades", "warmup_bars"):
                            overrides[key] = int(matches[-1])
        except Exception as e:
            logger.warning(f"Could not parse log file {log_file}: {e}")
    
    return overrides


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of file."""
    if not file_path.exists():
        return None
    import hashlib
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def baseline_parity_check(
    baseline_run_dir: Path,
    canary_run_dir: Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Check baseline parity by matching trades.
    
    Matches trades on stable key (entry_time + entry_price + side).
    """
    import pandas as pd
    
    # Load trade logs
    baseline_trade_log = baseline_run_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not baseline_trade_log.exists():
        baseline_trade_log = baseline_run_dir / "trade_log.csv"
    
    canary_trade_log = canary_run_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not canary_trade_log.exists():
        canary_trade_log = canary_run_dir / "trade_log.csv"
    
    result = {
        "baseline_trades": None,
        "canary_trades": None,
        "matched_trades": None,
        "match_rate": None,
        "unmatched_baseline": None,
        "unmatched_canary": None,
        "metrics": {},
    }
    
    if not baseline_trade_log.exists() or not canary_trade_log.exists():
        return result
    
    # Load and filter by date
    df_baseline = pd.read_csv(baseline_trade_log)
    df_canary = pd.read_csv(canary_trade_log)
    
    df_baseline["entry_time"] = pd.to_datetime(df_baseline["entry_time"])
    df_canary["entry_time"] = pd.to_datetime(df_canary["entry_time"])
    
    df_baseline_filtered = df_baseline[
        (df_baseline["entry_time"] >= start_date) &
        (df_baseline["entry_time"] <= end_date)
    ].copy()
    
    df_canary_filtered = df_canary[
        (df_canary["entry_time"] >= start_date) &
        (df_canary["entry_time"] <= end_date)
    ].copy()
    
    result["baseline_trades"] = len(df_baseline_filtered)
    result["canary_trades"] = len(df_canary_filtered)
    
    if len(df_baseline_filtered) == 0 or len(df_canary_filtered) == 0:
        return result
    
    # Create stable trade keys
    df_baseline_filtered["trade_key"] = (
        df_baseline_filtered["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S") + "_" +
        df_baseline_filtered["entry_price"].astype(str) + "_" +
        df_baseline_filtered["side"].astype(str)
    )
    df_canary_filtered["trade_key"] = (
        df_canary_filtered["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S") + "_" +
        df_canary_filtered["entry_price"].astype(str) + "_" +
        df_canary_filtered["side"].astype(str)
    )
    
    # Match trades
    baseline_keys = set(df_baseline_filtered["trade_key"])
    canary_keys = set(df_canary_filtered["trade_key"])
    matched_keys = baseline_keys & canary_keys
    
    result["matched_trades"] = len(matched_keys)
    result["match_rate"] = len(matched_keys) / max(len(baseline_keys), len(canary_keys)) if max(len(baseline_keys), len(canary_keys)) > 0 else 0.0
    result["unmatched_baseline"] = len(baseline_keys - canary_keys)
    result["unmatched_canary"] = len(canary_keys - baseline_keys)
    
    # Compare metrics for matched trades
    if matched_keys:
        df_baseline_matched = df_baseline_filtered[df_baseline_filtered["trade_key"].isin(matched_keys)]
        df_canary_matched = df_canary_filtered[df_canary_filtered["trade_key"].isin(matched_keys)]
        
        result["metrics"] = {
            "baseline_ev_trade": df_baseline_matched["pnl_bps"].mean(),
            "canary_ev_trade": df_canary_matched["pnl_bps"].mean(),
            "ev_trade_diff": df_canary_matched["pnl_bps"].mean() - df_baseline_matched["pnl_bps"].mean(),
            "baseline_trades_per_day": len(df_baseline_matched) / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1),
            "canary_trades_per_day": len(df_canary_matched) / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1),
        }
    
    return result


def compare_period_subset(
    baseline_run_dir: Path,
    canary_run_dir: Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Compare FULLYEAR subset period with CANARY period.
    
    Loads trade logs and filters to same period.
    """
    import pandas as pd
    
    # Load baseline trade log
    baseline_trade_log = baseline_run_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not baseline_trade_log.exists():
        baseline_trade_log = baseline_run_dir / "trade_log.csv"
    
    # Load canary trade log
    canary_trade_log = canary_run_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    if not canary_trade_log.exists():
        canary_trade_log = canary_run_dir / "trade_log.csv"
    
    results = {
        "baseline_subset": None,
        "canary": None,
        "comparison": None,
    }
    
    if baseline_trade_log.exists():
        df_baseline = pd.read_csv(baseline_trade_log)
        df_baseline["entry_time"] = pd.to_datetime(df_baseline["entry_time"])
        df_subset = df_baseline[
            (df_baseline["entry_time"] >= start_date) &
            (df_baseline["entry_time"] <= end_date)
        ]
        
        if len(df_subset) > 0:
            results["baseline_subset"] = {
                "n_trades": len(df_subset),
                "ev_trade": df_subset["pnl_bps"].mean(),
                "trades_per_day": len(df_subset) / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1),
                "ev_per_day": df_subset["pnl_bps"].sum() / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1),
            }
    
    if canary_trade_log.exists():
        df_canary = pd.read_csv(canary_trade_log)
        df_canary["entry_time"] = pd.to_datetime(df_canary["entry_time"])
        
        if len(df_canary) > 0:
            results["canary"] = {
                "n_trades": len(df_canary),
                "ev_trade": df_canary["pnl_bps"].mean(),
                "trades_per_day": len(df_canary) / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1),
                "ev_per_day": df_canary["pnl_bps"].sum() / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1),
            }
    
    # Compare
    if results["baseline_subset"] and results["canary"]:
        baseline_metrics = results["baseline_subset"]
        canary_metrics = results["canary"]
        
        results["comparison"] = {
            "trades_diff": canary_metrics["n_trades"] - baseline_metrics["n_trades"],
            "ev_trade_diff": canary_metrics["ev_trade"] - baseline_metrics["ev_trade"],
            "trades_per_day_diff": canary_metrics["trades_per_day"] - baseline_metrics["trades_per_day"],
            "ev_per_day_diff": canary_metrics["ev_per_day"] - baseline_metrics["ev_per_day"],
            "is_period_effect": abs(canary_metrics["ev_trade"] - baseline_metrics["ev_trade"]) < 5.0,  # Within 5 bps
        }
    
    return results


def generate_report(
    baseline_run_dir: Path,
    canary_run_dir: Path,
    baseline_config: Dict[str, Any],
    canary_config: Dict[str, Any],
    differences: List[Tuple[str, Any, Any]],
    categorized_diffs: Dict[str, List[Tuple[str, Any, Any]]],
    runtime_overrides: Dict[str, Any],
    period_comparison: Dict[str, Any],
    parity_check: Dict[str, Any],
    data_hashes: Dict[str, Dict[str, Optional[str]]],
    output_path: Path,
) -> None:
    """Generate markdown diff report."""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Run Diff Report\n\n")
        f.write(f"**Baseline Run:** `{baseline_run_dir.name}`\n")
        f.write(f"**Canary Run:** `{canary_run_dir.name}`\n\n")
        f.write("---\n\n")
        
        # Policy paths
        f.write("## Policy Files\n\n")
        f.write(f"- **Baseline:** `{baseline_config.get('policy_path', 'UNKNOWN')}`\n")
        f.write(f"- **Canary:** `{canary_config.get('policy_path', 'UNKNOWN')}`\n\n")
        
        # Runtime overrides
        if runtime_overrides:
            f.write("## Runtime Overrides Detected\n\n")
            for key, value in runtime_overrides.items():
                f.write(f"- `{key}`: `{value}`\n")
            f.write("\n")
        
        # Data Hashes
        f.write("## Data Parity\n\n")
        f.write("### File Hashes (SHA256)\n\n")
        f.write("| File | Baseline | Canary | Match |\n")
        f.write("|------|----------|--------|-------|\n")
        
        for file_type, hashes in data_hashes.items():
            baseline_hash = hashes.get("baseline")
            canary_hash = hashes.get("canary")
            match = "✅" if baseline_hash == canary_hash and baseline_hash else "❌"
            baseline_short = (baseline_hash[:16] + "...") if baseline_hash else "N/A"
            canary_short = (canary_hash[:16] + "...") if canary_hash else "N/A"
            f.write(f"| `{file_type}` | `{baseline_short}` | `{canary_short}` | {match} |\n")
        f.write("\n")
        
        # Differences - categorized
        f.write("## Configuration Differences\n\n")
        
        if not differences:
            f.write("✅ **No differences found** (excluding allowed differences)\n\n")
        else:
            # Entry-driving diffs
            if categorized_diffs.get("ENTRY"):
                f.write("### Entry-Driving Differences (Affects Trade Count)\n\n")
                f.write("| Key Path | Baseline | Canary |\n")
                f.write("|----------|----------|--------|\n")
                for key_path, baseline_val, canary_val in categorized_diffs["ENTRY"]:
                    baseline_str = str(baseline_val)[:50] if baseline_val != "<MISSING>" else "<MISSING>"
                    canary_str = str(canary_val)[:50] if canary_val != "<MISSING>" else "<MISSING>"
                    f.write(f"| `{key_path}` | `{baseline_str}` | `{canary_str}` |\n")
                f.write("\n")
            
            # Exit-driving diffs
            if categorized_diffs.get("EXIT"):
                f.write("### Exit-Driving Differences (Affects PnL/Hold Time)\n\n")
                f.write("| Key Path | Baseline | Canary |\n")
                f.write("|----------|----------|--------|\n")
                for key_path, baseline_val, canary_val in categorized_diffs["EXIT"]:
                    baseline_str = str(baseline_val)[:50] if baseline_val != "<MISSING>" else "<MISSING>"
                    canary_str = str(canary_val)[:50] if canary_val != "<MISSING>" else "<MISSING>"
                    f.write(f"| `{key_path}` | `{baseline_str}` | `{canary_str}` |\n")
                f.write("\n")
            
            # Other diffs
            if categorized_diffs.get("OTHER"):
                f.write("### Other Differences\n\n")
                f.write("| Key Path | Baseline | Canary |\n")
                f.write("|----------|----------|--------|\n")
                for key_path, baseline_val, canary_val in categorized_diffs["OTHER"]:
                    baseline_str = str(baseline_val)[:50] if baseline_val != "<MISSING>" else "<MISSING>"
                    canary_str = str(canary_val)[:50] if canary_val != "<MISSING>" else "<MISSING>"
                    f.write(f"| `{key_path}` | `{baseline_str}` | `{canary_str}` |\n")
                f.write("\n")
        
        # Baseline Parity Check
        f.write("## Baseline Parity Check\n\n")
        if parity_check.get("baseline_trades") is not None:
            f.write(f"- **Baseline trades:** {parity_check['baseline_trades']}\n")
            f.write(f"- **Canary trades:** {parity_check['canary_trades']}\n")
            f.write(f"- **Matched trades:** {parity_check.get('matched_trades', 0)}\n")
            f.write(f"- **Match rate:** {parity_check.get('match_rate', 0.0):.1%}\n")
            f.write(f"- **Unmatched baseline:** {parity_check.get('unmatched_baseline', 0)}\n")
            f.write(f"- **Unmatched canary:** {parity_check.get('unmatched_canary', 0)}\n\n")
            
            if parity_check.get("metrics"):
                metrics = parity_check["metrics"]
                f.write("### Matched Trades Metrics\n\n")
                f.write(f"- Baseline EV/trade: {metrics.get('baseline_ev_trade', 0):.2f} bps\n")
                f.write(f"- Canary EV/trade: {metrics.get('canary_ev_trade', 0):.2f} bps\n")
                f.write(f"- EV/trade diff: {metrics.get('ev_trade_diff', 0):+.2f} bps\n\n")
            
            match_rate = parity_check.get("match_rate", 0.0)
            if match_rate < 0.5:
                f.write("⚠️  **Low match rate** - Trades do not match. Possible causes:\n")
                f.write("- Different entry config/model\n")
                f.write("- Different price data\n")
                f.write("- Different warmup period\n")
                f.write("- Non-deterministic entry logic\n\n")
            elif match_rate < 0.9:
                f.write("⚠️  **Partial match** - Some trades differ. Check unmatched trades.\n\n")
            else:
                f.write("✅ **High match rate** - Trades match well.\n\n")
        else:
            f.write("⚠️  Could not perform parity check (missing trade logs)\n\n")
        
        # Period comparison
        if period_comparison.get("comparison"):
            f.write("## Period Comparison\n\n")
            comp = period_comparison["comparison"]
            baseline_subset = period_comparison["baseline_subset"]
            canary_metrics = period_comparison["canary"]
            
            f.write("### FULLYEAR Subset (same period as CANARY)\n\n")
            f.write(f"- Trades: {baseline_subset['n_trades']}\n")
            f.write(f"- EV/trade: {baseline_subset['ev_trade']:.2f} bps\n")
            f.write(f"- Trades/day: {baseline_subset['trades_per_day']:.2f}\n")
            f.write(f"- EV/day: {baseline_subset['ev_per_day']:.2f} bps\n\n")
            
            f.write("### CANARY\n\n")
            f.write(f"- Trades: {canary_metrics['n_trades']}\n")
            f.write(f"- EV/trade: {canary_metrics['ev_trade']:.2f} bps\n")
            f.write(f"- Trades/day: {canary_metrics['trades_per_day']:.2f}\n")
            f.write(f"- EV/day: {canary_metrics['ev_per_day']:.2f} bps\n\n")
            
            f.write("### Difference\n\n")
            f.write(f"- Trades diff: {comp['trades_diff']:+d}\n")
            f.write(f"- EV/trade diff: {comp['ev_trade_diff']:+.2f} bps\n")
            f.write(f"- Trades/day diff: {comp['trades_per_day_diff']:+.2f}\n")
            f.write(f"- EV/day diff: {comp['ev_per_day_diff']:+.2f} bps\n\n")
            
            if comp["is_period_effect"]:
                f.write("✅ **Conclusion:** Difference is likely period/regime effect, not config.\n\n")
            else:
                f.write("⚠️  **Conclusion:** Difference is likely config/runtime, not period effect.\n\n")
        
        # Root cause
        f.write("## Root Cause Analysis\n\n")
        
        # Check trade count difference
        trades_diff_pct = 0.0
        if period_comparison.get("comparison"):
            comp = period_comparison["comparison"]
            baseline_trades = period_comparison.get("baseline_subset", {}).get("n_trades", 0)
            if baseline_trades > 0:
                trades_diff_pct = abs(comp.get("trades_diff", 0)) / baseline_trades
        
        # Entry vs Exit analysis
        entry_diffs = categorized_diffs.get("ENTRY", [])
        exit_diffs = categorized_diffs.get("EXIT", [])
        
        f.write("### Trade Count Analysis\n\n")
        if trades_diff_pct > 0.2:
            f.write(f"⚠️  **Large trade count difference** ({trades_diff_pct:.1%}) - Likely **ENTRY/GATES mismatch**\n\n")
            if entry_diffs:
                f.write("**Entry-driving differences found:**\n")
                for key_path, _, _ in entry_diffs[:5]:  # Show first 5
                    f.write(f"- `{key_path}`\n")
                if len(entry_diffs) > 5:
                    f.write(f"- ... and {len(entry_diffs) - 5} more\n")
                f.write("\n")
            else:
                f.write("⚠️  **No entry differences found** - Possible causes:\n")
                f.write("- Different price data (check Data Parity section)\n")
                f.write("- Different warmup period\n")
                f.write("- Non-deterministic entry logic\n")
                f.write("- Hidden config differences\n\n")
        else:
            f.write("✅ Trade count difference is small (<20%) - likely not entry-related.\n\n")
        
        f.write("### PnL/Hold Time Analysis\n\n")
        if exit_diffs:
            f.write("**Exit-driving differences found:**\n")
            for key_path, baseline_val, canary_val in exit_diffs:
                f.write(f"- `{key_path}`: Baseline=`{baseline_val}`, Canary=`{canary_val}`\n")
            f.write("\n")
        else:
            f.write("✅ No exit differences found.\n\n")
        
        # Conclusion
        f.write("### Conclusion\n\n")
        if trades_diff_pct > 0.2 and entry_diffs:
            f.write("**Årsak (konkret):** Trade count difference er **ENTRY/GATES mismatch**. ")
            f.write(f"Funn: {len(entry_diffs)} entry-driving config differences. ")
            f.write("Dette forklarer hvorfor CANARY genererer flere/færre trades enn baseline.\n\n")
        elif trades_diff_pct > 0.2 and not entry_diffs:
            f.write("**Årsak (konkret):** Trade count difference er stor, men ingen entry-differences funnet. ")
            f.write("Sannsynlige årsaker: forskjellig price data, warmup, eller non-determinisme.\n\n")
        elif exit_diffs:
            f.write("**Årsak (konkret):** Exit differences funnet. ")
            f.write("Dette kan forklare forskjeller i PnL/hold time, men ikke trade count.\n\n")
        else:
            f.write("Ingen tydelig root cause identifisert. Forskjellen kan være periode/regime-effekt.\n\n")
        
        # Fix recommendations
        f.write("## Fix Recommendations\n\n")
        
        fixes = []
        if any("router_version" in d[0] or "v3_range_edge_cutoff" in d[0] for d in differences):
            fixes.append("**Use PROD_BASELINE policy**: CANARY skal bruke samme policy som PROD_BASELINE (V3_RANGE med guardrail)")
            fixes.append("**Verify policy path**: Sjekk at `gx1/prod/current/policy.yaml` eller `policy_canary.yaml` peker til riktig frozen policy")
        if differences:
            fixes.append("Ensure CANARY uses same policy YAML as PROD_BASELINE")
            fixes.append("Verify entry_config and exit_config paths match")
        if runtime_overrides.get("canary_mode"):
            fixes.append("CANARY mode is expected (dry_run=True) - ikke endre dette")
        
        if fixes:
            for fix in fixes:
                f.write(f"- {fix}\n")
            f.write("\n")
        
        # Specific fix
        if any("router_version" in d[0] for d in differences):
            f.write("### Specific Fix\n\n")
            f.write("1. Sjekk hvilken policy CANARY faktisk bruker:\n")
            f.write("   ```bash\n")
            f.write(f"   cat {canary_config.get('policy_path', 'UNKNOWN')}\n")
            f.write("   ```\n\n")
            f.write("2. Sjekk PROD_BASELINE policy:\n")
            f.write("   ```bash\n")
            f.write("   cat gx1/prod/current/policy.yaml\n")
            f.write("   ```\n\n")
            f.write("3. Oppdater CANARY policy til å bruke samme policy som PROD_BASELINE\n\n")
        
        # Re-run plan
        f.write("## Re-run Plan\n\n")
        
        baseline_policy_path = baseline_config.get("policy_path", "UNKNOWN")
        canary_policy_path = canary_config.get("policy_path", "UNKNOWN")
        
        f.write("### Same Policy, Same Period Sanity Check\n\n")
        f.write("Kjør begge runs med samme policy og periode:\n\n")
        
        f.write("**Baseline (PROD_BASELINE policy):**\n")
        f.write("```bash\n")
        f.write("export OANDA_ENV=practice\n")
        f.write("export OANDA_API_TOKEN=<token>\n")
        f.write("export OANDA_ACCOUNT_ID=<account_id>\n")
        f.write(f"bash scripts/run_replay.sh {baseline_policy_path} 2025-01-01 2025-01-15 7 gx1/wf_runs/BASELINE_SANITY_CHECK\n")
        f.write("```\n\n")
        
        f.write("**Canary (same policy):**\n")
        f.write("```bash\n")
        f.write("export OANDA_ENV=practice\n")
        f.write("export OANDA_API_TOKEN=<token>\n")
        f.write("export OANDA_ACCOUNT_ID=<account_id>\n")
        f.write(f"bash scripts/run_replay.sh {canary_policy_path} 2025-01-01 2025-01-15 7 gx1/wf_runs/CANARY_SANITY_CHECK\n")
        f.write("```\n\n")
        
        f.write("**Then compare:**\n")
        f.write("```bash\n")
        f.write("python gx1/analysis/diff_runs.py \\\n")
        f.write("  --baseline-run gx1/wf_runs/BASELINE_SANITY_CHECK \\\n")
        f.write("  --canary-run gx1/wf_runs/CANARY_SANITY_CHECK \\\n")
        f.write("  --out gx1/wf_runs/CANARY_SANITY_CHECK/sanity_diff_report.md \\\n")
        f.write("  --start-date 2025-01-01 --end-date 2025-01-15\n")
        f.write("```\n\n")


def main():
    parser = argparse.ArgumentParser(description="Diff two GX1 runs")
    parser.add_argument("--baseline-run", required=True, type=Path, help="Baseline run directory")
    parser.add_argument("--canary-run", required=True, type=Path, help="Canary run directory")
    parser.add_argument("--out", required=True, type=Path, help="Output report path")
    parser.add_argument("--start-date", default="2025-01-01", help="Start date for period comparison")
    parser.add_argument("--end-date", default="2025-01-15", help="End date for period comparison")
    
    args = parser.parse_args()
    
    # Find policy files
    logger.info(f"Finding policy files...")
    baseline_policy_path = find_policy_file(args.baseline_run)
    canary_policy_path = find_policy_file(args.canary_run)
    
    if not baseline_policy_path:
        raise FileNotFoundError(f"Could not find policy file for baseline run: {args.baseline_run}")
    if not canary_policy_path:
        raise FileNotFoundError(f"Could not find policy file for canary run: {args.canary_run}")
    
    logger.info(f"Baseline policy: {baseline_policy_path}")
    logger.info(f"Canary policy: {canary_policy_path}")
    
    # Load policy bundles
    logger.info("Loading policy bundles...")
    baseline_config = load_policy_bundle(baseline_policy_path)
    canary_config = load_policy_bundle(canary_policy_path)
    
    # Normalize for diff
    baseline_normalized = normalize_config_for_diff(baseline_config)
    canary_normalized = normalize_config_for_diff(canary_config)
    
    # Diff configs
    logger.info("Diffing configurations...")
    differences, categorized_diffs = diff_dicts(baseline_normalized, canary_normalized)
    
    # Extract runtime overrides
    logger.info("Extracting runtime overrides...")
    runtime_overrides = extract_runtime_overrides(args.canary_run)
    
    # Period comparison
    logger.info("Comparing period subset...")
    period_comparison = compare_period_subset(
        args.baseline_run,
        args.canary_run,
        args.start_date,
        args.end_date,
    )
    
    # Baseline parity check
    logger.info("Performing baseline parity check...")
    parity_check = baseline_parity_check(
        args.baseline_run,
        args.canary_run,
        args.start_date,
        args.end_date,
    )
    
    # Data hashes
    logger.info("Computing data hashes...")
    data_hashes = {}
    
    # Price data
    baseline_price_data = args.baseline_run / "price_data_filtered.parquet"
    canary_price_data = args.canary_run / "price_data_filtered.parquet"
    data_hashes["price_data_filtered.parquet"] = {
        "baseline": compute_file_hash(baseline_price_data),
        "canary": compute_file_hash(canary_price_data),
    }
    
    # Policy files
    baseline_policy_path = Path(baseline_config.get("policy_path", ""))
    canary_policy_path = Path(canary_config.get("policy_path", ""))
    data_hashes["policy.yaml"] = {
        "baseline": compute_file_hash(baseline_policy_path),
        "canary": compute_file_hash(canary_policy_path),
    }
    
    # Entry config
    baseline_entry_path = Path(baseline_config.get("entry_config_path", ""))
    canary_entry_path = Path(canary_config.get("entry_config_path", ""))
    if baseline_entry_path.exists() or canary_entry_path.exists():
        data_hashes["entry_config.yaml"] = {
            "baseline": compute_file_hash(baseline_entry_path) if baseline_entry_path.exists() else None,
            "canary": compute_file_hash(canary_entry_path) if canary_entry_path.exists() else None,
        }
    
    # Exit config
    baseline_exit_path = Path(baseline_config.get("exit_config_path", ""))
    canary_exit_path = Path(canary_config.get("exit_config_path", ""))
    if baseline_exit_path.exists() or canary_exit_path.exists():
        data_hashes["exit_config.yaml"] = {
            "baseline": compute_file_hash(baseline_exit_path) if baseline_exit_path.exists() else None,
            "canary": compute_file_hash(canary_exit_path) if canary_exit_path.exists() else None,
        }
    
    # Router model (from run_header if available)
    baseline_run_header = args.baseline_run / "run_header.json"
    canary_run_header = args.canary_run / "run_header.json"
    
    for run_header, run_type in [(baseline_run_header, "baseline"), (canary_run_header, "canary")]:
        if run_header.exists():
            try:
                with open(run_header) as f:
                    header = json.load(f)
                    router_model = header.get("artifacts", {}).get("router_model", {})
                    if router_model.get("sha256"):
                        if "router_model.pkl" not in data_hashes:
                            data_hashes["router_model.pkl"] = {"baseline": None, "canary": None}
                        data_hashes["router_model.pkl"][run_type] = router_model["sha256"]
            except Exception as e:
                logger.warning(f"Could not read router model hash from {run_header}: {e}")
    
    # Generate report
    logger.info(f"Generating report: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate_report(
        args.baseline_run,
        args.canary_run,
        baseline_config,
        canary_config,
        differences,
        categorized_diffs,
        runtime_overrides,
        period_comparison,
        parity_check,
        data_hashes,
        args.out,
    )
    
    logger.info("✅ Diff report generated")


if __name__ == "__main__":
    main()

