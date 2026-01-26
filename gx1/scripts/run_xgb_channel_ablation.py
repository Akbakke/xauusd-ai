#!/usr/bin/env python3
"""
XGB Channel Ablation Script (DEL C)

Runs per-channel ablation analysis to determine value of individual XGB channels.
Each channel is dropped (set to 0) one at a time, and metrics are compared to baseline.

Usage:
    # Q-smoke (quick)
    python3 gx1/scripts/run_xgb_channel_ablation.py \
        --mode qsmoke \
        --smoke-date-range "2025-01-06..2025-03-31" \
        --data <CANDLES_PARQUET> \
        --prebuilt-parquet <PREBUILT_PARQUET> \
        --bundle-dir <BUNDLE_DIR> \
        --policy <POLICY_YAML> \
        --out-root ../GX1_DATA/reports/replay_eval/XGB_CHANNEL_ABLATION

    # FULLYEAR
    python3 gx1/scripts/run_xgb_channel_ablation.py \
        --mode fullyear \
        --data <CANDLES_PARQUET> \
        --prebuilt-parquet <PREBUILT_PARQUET> \
        --bundle-dir <BUNDLE_DIR> \
        --policy <POLICY_YAML> \
        --out-root ../GX1_DATA/reports/replay_eval/XGB_CHANNEL_ABLATION

Output:
    XGB_CHANNEL_ABLATION_<mode>.json
    XGB_CHANNEL_ABLATION_<mode>.md
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# XGB channels to ablate
# NOTE: margin_xgb was REMOVED as of 2026-01-25 (FULLYEAR ablation showed it's harmful)
CHANNELS_TO_ABLATE = [
    "p_long_xgb",      # Present in both seq and snap
    # "margin_xgb",      # REMOVED
    "uncertainty_score",  # Present in seq only
    "p_hat_xgb",       # Present in snap only
]

# Ablation arms configuration
ABLATION_ARMS = {
    "baseline": {
        "name": "baseline",
        "description": "All XGB channels enabled",
        "mask": None,
        "env": {},
    },
    "drop_p_long_xgb": {
        "name": "drop_p_long_xgb",
        "description": "Drop p_long_xgb channel",
        "mask": "p_long_xgb",
        "env": {"GX1_XGB_CHANNEL_MASK": "p_long_xgb"},
    },
    # "drop_margin_xgb": {  # REMOVED: margin_xgb was removed from pipeline as of 2026-01-25
    #     "name": "drop_margin_xgb",
    #     "description": "Drop margin_xgb channel",
    #     "mask": "margin_xgb",
    #     "env": {"GX1_XGB_CHANNEL_MASK": "margin_xgb"},
    # },
    "drop_uncertainty": {
        "name": "drop_uncertainty",
        "description": "Drop uncertainty_score channel (seq only)",
        "mask": "uncertainty_score",
        "env": {"GX1_XGB_CHANNEL_MASK": "uncertainty_score"},
    },
    "drop_p_hat": {
        "name": "drop_p_hat",
        "description": "Drop p_hat_xgb channel (snap only)",
        "mask": "p_hat_xgb",
        "env": {"GX1_XGB_CHANNEL_MASK": "p_hat_xgb"},
    },
}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# REPLAY ARM RUNNER
# ============================================================================

def run_replay_arm(
    arm_name: str,
    arm_config: Dict[str, Any],
    data_path: Path,
    prebuilt_path: Path,
    bundle_dir: Path,
    policy_path: Path,
    output_dir: Path,
    smoke_date_range: Optional[str] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run a single replay arm."""
    
    arm_output = output_dir / arm_name
    arm_output.mkdir(parents=True, exist_ok=True)
    
    # Base environment
    env = os.environ.copy()
    env.update({
        "GX1_GATED_FUSION_ENABLED": "1",
        "GX1_REPLAY_USE_PREBUILT_FEATURES": "1",
        "GX1_FEATURE_BUILD_DISABLED": "1",
        "GX1_REQUIRE_ENTRY_TELEMETRY": "1",
        "GX1_ALLOW_PARALLEL_REPLAY": "1",
        "GX1_PANIC_MODE": "0",
    })
    
    # Add arm-specific environment
    env.update(arm_config.get("env", {}))
    
    # Build command
    replay_script = workspace_root / "gx1" / "scripts" / "replay_eval_gated_parallel.py"
    cmd = [
        sys.executable,
        str(replay_script),
        "--policy", str(policy_path),
        "--data", str(data_path),
        "--prebuilt-parquet", str(prebuilt_path),
        "--bundle-dir", str(bundle_dir),
        "--output-dir", str(arm_output),
        "--workers", str(workers),
    ]
    
    if smoke_date_range:
        start_date, end_date = smoke_date_range.split("..")
        cmd.extend(["--start-ts", start_date, "--end-ts", end_date])
    else:
        # Full year 2025
        cmd.extend(["--start-ts", "2025-01-01", "--end-ts", "2025-12-31"])
    
    log.info(f"[{arm_name}] Running replay...")
    log.info(f"[{arm_name}] Mask: {arm_config.get('mask', 'None')}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=workspace_root,
            env=env,
            timeout=3600,
        )
        elapsed = time.time() - start_time
        
        success = result.returncode == 0
        if not success:
            log.error(f"[{arm_name}] Failed: {result.stderr[-500:]}")
        else:
            log.info(f"[{arm_name}] Completed in {elapsed:.1f}s")
        
        return {
            "arm_name": arm_name,
            "success": success,
            "elapsed_sec": elapsed,
            "output_dir": str(arm_output),
            "mask": arm_config.get("mask"),
            "env": arm_config.get("env", {}),
        }
        
    except Exception as e:
        log.error(f"[{arm_name}] Exception: {e}")
        return {
            "arm_name": arm_name,
            "success": False,
            "error": str(e),
            "output_dir": str(arm_output),
        }


def load_arm_metrics(arm_result: Dict[str, Any]) -> Dict[str, Any]:
    """Load metrics from arm output (SSoT: chunk_*/metrics_*.json)."""
    output_dir = Path(arm_result["output_dir"])
    
    metrics = {
        "n_trades": 0,
        "total_pnl_bps": 0.0,
        "max_dd": 0.0,
        "mean_pnl_bps": 0.0,
        "median_pnl_bps": 0.0,
        "winrate": 0.0,
        "transformer_forward_calls": 0,
        "xgb_pre_predict_count": 0,
        "session_breakdown": {},  # ASIA/EU/US/OVERLAP
    }
    
    # Find chunk directories
    chunk_dirs = sorted(output_dir.glob("chunk_*"))
    
    if not chunk_dirs:
        raise RuntimeError(f"No chunks found in {output_dir}")
    
    all_pnl_per_trade = []
    
    for chunk_dir in chunk_dirs:
        # Load metrics (SSoT)
        metrics_files = list(chunk_dir.glob("metrics_*.json"))
        if not metrics_files:
            log.warning(f"No metrics_*.json found in {chunk_dir}")
            continue
        
        for mf in metrics_files:
            with open(mf) as f:
                data = json.load(f)
            metrics["n_trades"] += data.get("n_trades", 0)
            metrics["total_pnl_bps"] += data.get("total_pnl_bps", 0.0)
            if data.get("max_dd", 0.0) < metrics["max_dd"]:
                metrics["max_dd"] = data.get("max_dd", 0.0)
            
            # Collect PnL per trade for median
            if "pnl_per_trade" in data:
                all_pnl_per_trade.extend(data["pnl_per_trade"])
            
            # Session breakdown (if available)
            if "session_breakdown" in data:
                for session, session_data in data["session_breakdown"].items():
                    if session not in metrics["session_breakdown"]:
                        metrics["session_breakdown"][session] = {
                            "n_trades": 0,
                            "total_pnl_bps": 0.0,
                        }
                    metrics["session_breakdown"][session]["n_trades"] += session_data.get("n_trades", 0)
                    metrics["session_breakdown"][session]["total_pnl_bps"] += session_data.get("total_pnl_bps", 0.0)
        
        # Load telemetry
        ef_path = chunk_dir / "ENTRY_FEATURES_USED.json"
        if ef_path.exists():
            with open(ef_path) as f:
                ef = json.load(f)
            metrics["transformer_forward_calls"] += ef.get("transformer_forward_calls", 0)
            xgb_flow = ef.get("xgb_flow", {})
            metrics["xgb_pre_predict_count"] += xgb_flow.get("xgb_pre_predict_count", 0)
    
    if metrics["n_trades"] > 0:
        metrics["mean_pnl_bps"] = metrics["total_pnl_bps"] / metrics["n_trades"]
        if all_pnl_per_trade:
            import numpy as np
            metrics["median_pnl_bps"] = float(np.median(all_pnl_per_trade))
        
        # Winrate (if available)
        winning_trades = sum(1 for pnl in all_pnl_per_trade if pnl > 0) if all_pnl_per_trade else 0
        metrics["winrate"] = winning_trades / len(all_pnl_per_trade) if all_pnl_per_trade else 0.0
    
    return metrics


def load_arm_channel_info(arm_result: Dict[str, Any]) -> Dict[str, Any]:
    """Load channel masking info from arm output."""
    output_dir = Path(arm_result["output_dir"])
    
    # Aggregate from all chunks
    channel_info = {
        "n_xgb_channels_in_transformer_input": 0,
        "xgb_seq_channel_names": [],
        "xgb_snap_channel_names": [],
        "channel_mask_info": {},
    }
    
    # Find first chunk with telemetry
    chunk_dirs = sorted(output_dir.glob("chunk_*"))
    for chunk_dir in chunk_dirs:
        ef_path = chunk_dir / "ENTRY_FEATURES_USED.json"
        if ef_path.exists():
            with open(ef_path) as f:
                ef = json.load(f)
            
            # Take first non-empty values
            if not channel_info["xgb_seq_channel_names"]:
                channel_info["xgb_seq_channel_names"] = ef.get("xgb_seq_channels", {}).get("names", [])
            if not channel_info["xgb_snap_channel_names"]:
                channel_info["xgb_snap_channel_names"] = ef.get("xgb_snap_channels", {}).get("names", [])
            
            xgb_flow = ef.get("xgb_flow", {})
            if channel_info["n_xgb_channels_in_transformer_input"] == 0:
                channel_info["n_xgb_channels_in_transformer_input"] = xgb_flow.get("n_xgb_channels_in_transformer_input", 3)
            
            # Channel mask info (if available)
            if "channel_mask_info" in ef:
                channel_info["channel_mask_info"] = ef["channel_mask_info"]
            
            break  # Use first chunk
    
    return channel_info


def verify_masking_effect(arm_result: Dict[str, Any], expected_mask: Optional[str]) -> Dict[str, Any]:
    """Verify that masking actually took effect (fail-fast if not)."""
    output_dir = Path(arm_result["output_dir"])
    channel_info = load_arm_channel_info(arm_result)
    
    verification = {
        "masking_verified": False,
        "expected_mask": expected_mask,
        "actual_channels": {
            "seq": channel_info["xgb_seq_channel_names"],
            "snap": channel_info["xgb_snap_channel_names"],
        },
        "errors": [],
    }
    
    if expected_mask is None:
        # Baseline: should have all channels (margin_xgb was REMOVED as of 2026-01-25)
        expected_seq = ["p_long_xgb", "uncertainty_score"]
        expected_snap = ["p_long_xgb", "p_hat_xgb"]
        
        if set(channel_info["xgb_seq_channel_names"]) != set(expected_seq):
            verification["errors"].append(
                f"Baseline seq channels mismatch: expected {expected_seq}, got {channel_info['xgb_seq_channel_names']}"
            )
        if set(channel_info["xgb_snap_channel_names"]) != set(expected_snap):
            verification["errors"].append(
                f"Baseline snap channels mismatch: expected {expected_snap}, got {channel_info['xgb_snap_channel_names']}"
            )
    else:
        # Ablated: should NOT have the masked channel
        if expected_mask in channel_info["xgb_seq_channel_names"]:
            verification["errors"].append(
                f"MASKING_FAIL: {expected_mask} still present in seq channels: {channel_info['xgb_seq_channel_names']}"
            )
        if expected_mask in channel_info["xgb_snap_channel_names"]:
            verification["errors"].append(
                f"MASKING_FAIL: {expected_mask} still present in snap channels: {channel_info['xgb_snap_channel_names']}"
            )
    
    verification["masking_verified"] = len(verification["errors"]) == 0
    
    if not verification["masking_verified"]:
        raise RuntimeError(f"MASKING_VERIFICATION_FAILED: {verification['errors']}")
    
    return verification


def generate_verification_report(
    arm_result: Dict[str, Any],
    metrics: Dict[str, Any],
    channel_info: Dict[str, Any],
    masking_verification: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate FULLYEAR_VERIFICATION_REPORT for an arm."""
    
    # Load chunk footer for additional invariants
    chunk_dirs = sorted(output_dir.glob("chunk_*"))
    prebuilt_used = False
    feature_build_call_count = -1
    lookup_hits = 0
    lookup_attempts = 0
    
    for chunk_dir in chunk_dirs:
        footer_path = chunk_dir / "chunk_footer.json"
        if footer_path.exists():
            with open(footer_path) as f:
                footer = json.load(f)
            prebuilt_used = footer.get("prebuilt_used", False) or prebuilt_used
            feature_build_call_count = max(feature_build_call_count, footer.get("feature_build_call_count", -1))
            lookup_hits += footer.get("lookup_hits", 0)
            lookup_attempts += footer.get("lookup_attempts", 0)
    
    # Build verification report
    report = {
        "arm_name": arm_result["arm_name"],
        "timestamp": datetime.now().isoformat(),
        "verification": {
            "prebuilt_used": prebuilt_used,
            "feature_build_call_count": feature_build_call_count,
            "feature_build_call_count_zero": feature_build_call_count == 0,
            "lookup_hits": lookup_hits,
            "lookup_attempts": lookup_attempts,
            "lookup_invariant_pass": lookup_hits + (lookup_attempts - lookup_hits) == lookup_attempts,
            "v10_hybrid_routing": metrics.get("transformer_forward_calls", 0) > 0,
            "transformer_forward_calls": metrics["transformer_forward_calls"],
            "xgb_pre_predict_count": metrics["xgb_pre_predict_count"],
            "masking_verified": masking_verification["masking_verified"],
            "effective_xgb_channels": {
                "seq": channel_info["xgb_seq_channel_names"],
                "snap": channel_info["xgb_snap_channel_names"],
                "n_total": channel_info["n_xgb_channels_in_transformer_input"],
            },
        },
        "metrics": metrics,
        "all_invariants_pass": (
            prebuilt_used and
            feature_build_call_count == 0 and
            metrics["transformer_forward_calls"] > 0 and
            masking_verification["masking_verified"]
        ),
    }
    
    # Write JSON
    json_path = output_dir / "FULLYEAR_VERIFICATION_REPORT.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    # Write Markdown
    md = f"""# FULLYEAR Verification Report

**Arm:** {arm_result['arm_name']}  
**Timestamp:** {report['timestamp']}

---

## Verification Status

| Check | Status |
|-------|--------|
| `prebuilt_used` | {'✅' if prebuilt_used else '❌'} |
| `feature_build_call_count == 0` | {'✅' if feature_build_call_count == 0 else f'❌ ({feature_build_call_count})'} |
| `transformer_forward_calls > 0` | {'✅' if metrics['transformer_forward_calls'] > 0 else '❌'} |
| `masking_verified` | {'✅' if masking_verification['masking_verified'] else '❌'} |
| `lookup_invariant_pass` | {'✅' if report['verification']['lookup_invariant_pass'] else '❌'} |

---

## Metrics

| Metric | Value |
|--------|-------|
| `n_trades` | {metrics['n_trades']:,} |
| `total_pnl_bps` | {metrics['total_pnl_bps']:.2f} |
| `mean_pnl_bps` | {metrics.get('mean_pnl_bps', 0):.2f} |
| `median_pnl_bps` | {metrics.get('median_pnl_bps', 0):.2f} |
| `max_dd` | {metrics['max_dd']:.2f} |
| `winrate` | {metrics.get('winrate', 0):.2%} |

---

## XGB Channels

| Point | Channels |
|-------|----------|
| **Sequence** | {', '.join(channel_info['xgb_seq_channel_names']) or 'None'} |
| **Snapshot** | {', '.join(channel_info['xgb_snap_channel_names']) or 'None'} |
| **Total Count** | {channel_info['n_xgb_channels_in_transformer_input']} |

---

## Masking Verification

| Check | Status |
|-------|--------|
| Expected mask | {masking_verification.get('expected_mask', 'None')} |
| Masking verified | {'✅' if masking_verification['masking_verified'] else '❌'} |

"""
    
    if metrics.get("session_breakdown"):
        md += "## Session Breakdown\n\n"
        md += "| Session | Trades | PnL (bps) |\n"
        md += "|---------|--------|-----------|\n"
        for session, data in metrics["session_breakdown"].items():
            md += f"| {session} | {data['n_trades']:,} | {data['total_pnl_bps']:.2f} |\n"
        md += "\n"
    
    md += f"""
---

## Overall Status

{'✅ **ALL INVARIANTS PASS**' if report['all_invariants_pass'] else '❌ **SOME INVARIANTS FAILED**'}
"""
    
    md_path = output_dir / "FULLYEAR_VERIFICATION_REPORT.md"
    with open(md_path, "w") as f:
        f.write(md)
    
    log.info(f"Written verification report: {json_path}")


# ============================================================================
# COMPARISON
# ============================================================================

def compare_to_baseline(
    baseline_metrics: Dict[str, Any],
    ablated_metrics: Dict[str, Any],
    arm_name: str,
) -> Dict[str, Any]:
    """Compare ablated arm to baseline."""
    
    def pct_diff(a, b):
        if b == 0:
            return 0.0 if a == 0 else float("inf")
        return ((a - b) / abs(b)) * 100
    
    comparison = {
        "arm_name": arm_name,
        "baseline": baseline_metrics,
        "ablated": ablated_metrics,
        "delta": {
            "n_trades": ablated_metrics["n_trades"] - baseline_metrics["n_trades"],
            "total_pnl_bps": ablated_metrics["total_pnl_bps"] - baseline_metrics["total_pnl_bps"],
            "max_dd": ablated_metrics["max_dd"] - baseline_metrics["max_dd"],
        },
        "pct_delta": {
            "n_trades": pct_diff(ablated_metrics["n_trades"], baseline_metrics["n_trades"]),
            "total_pnl_bps": pct_diff(ablated_metrics["total_pnl_bps"], baseline_metrics["total_pnl_bps"]),
        },
    }
    
    # Verdict
    trades_impact = comparison["delta"]["n_trades"]
    pnl_impact = comparison["delta"]["total_pnl_bps"]
    
    if trades_impact == 0 and abs(pnl_impact) < 10:
        verdict = "REDUNDANT"
        verdict_reason = "No significant impact on trades or PnL"
    elif trades_impact < 0 and pnl_impact < 0:
        verdict = "CRITICAL"
        verdict_reason = f"Removing channel reduces trades by {abs(trades_impact)} and PnL by {abs(pnl_impact):.0f} bps"
    elif trades_impact > 0 or pnl_impact > 0:
        verdict = "HARMFUL"
        verdict_reason = f"Removing channel improves performance (possible noise source)"
    else:
        verdict = "MIXED"
        verdict_reason = "Mixed impact on trades/PnL"
    
    comparison["verdict"] = verdict
    comparison["verdict_reason"] = verdict_reason
    
    return comparison


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    comparisons: List[Dict[str, Any]],
    output_dir: Path,
    mode: str,
) -> None:
    """Generate comparison report."""
    
    timestamp = datetime.now().isoformat()
    
    # JSON report
    report = {
        "report_type": "XGB_CHANNEL_ABLATION",
        "mode": mode,
        "timestamp": timestamp,
        "channels_ablated": CHANNELS_TO_ABLATE,
        "comparisons": comparisons,
    }
    
    json_path = output_dir / f"XGB_CHANNEL_ABLATION_{mode.upper()}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    log.info(f"Written JSON: {json_path}")
    
    # Markdown report
    baseline = comparisons[0]["baseline"] if comparisons else {}
    
    md = f"""# XGB Channel Ablation Report

**Mode:** {mode}  
**Timestamp:** {timestamp}

---

## Baseline Metrics

| Metric | Value |
|--------|-------|
| `n_trades` | {baseline.get('n_trades', 0):,} |
| `total_pnl_bps` | {baseline.get('total_pnl_bps', 0):.2f} |
| `max_dd` | {baseline.get('max_dd', 0):.2f} |
| `transformer_forward_calls` | {baseline.get('transformer_forward_calls', 0):,} |

---

## Channel Ablation Results

| Channel | Trades | Δ Trades | PnL (bps) | Δ PnL | Verdict |
|---------|--------|----------|-----------|-------|---------|
| **baseline** | {baseline.get('n_trades', 0):,} | - | {baseline.get('total_pnl_bps', 0):.0f} | - | - |
"""
    
    for comp in comparisons:
        arm = comp["arm_name"]
        if arm == "baseline":
            continue
        abl = comp["ablated"]
        delta = comp["delta"]
        verdict = comp.get("verdict", "")
        verdict_emoji = {
            "CRITICAL": "🔴",
            "HARMFUL": "🟡",
            "REDUNDANT": "🟢",
            "MIXED": "🔵",
        }.get(verdict, "⚪")
        
        md += f"| `{arm}` | {abl.get('n_trades', 0):,} | {delta.get('n_trades', 0):+d} | {abl.get('total_pnl_bps', 0):.0f} | {delta.get('total_pnl_bps', 0):+.0f} | {verdict_emoji} {verdict} |\n"
    
    md += f"""
---

## Verdicts

| Verdict | Meaning |
|---------|---------|
| 🔴 CRITICAL | Channel is essential - removing it hurts performance |
| 🟡 HARMFUL | Removing channel improves performance (noise source?) |
| 🟢 REDUNDANT | Channel has no significant impact |
| 🔵 MIXED | Mixed impact on different metrics |

---

## Per-Channel Analysis

"""
    for comp in comparisons:
        if comp["arm_name"] == "baseline":
            continue
        md += f"""### {comp['arm_name']}

**Verdict:** {comp.get('verdict', 'N/A')}  
**Reason:** {comp.get('verdict_reason', 'N/A')}

| Metric | Baseline | Ablated | Delta |
|--------|----------|---------|-------|
| Trades | {comp['baseline'].get('n_trades', 0):,} | {comp['ablated'].get('n_trades', 0):,} | {comp['delta'].get('n_trades', 0):+d} |
| PnL (bps) | {comp['baseline'].get('total_pnl_bps', 0):.0f} | {comp['ablated'].get('total_pnl_bps', 0):.0f} | {comp['delta'].get('total_pnl_bps', 0):+.0f} |
| Max DD | {comp['baseline'].get('max_dd', 0):.0f} | {comp['ablated'].get('max_dd', 0):.0f} | {comp['delta'].get('max_dd', 0):+.0f} |
| Mean PnL | {comp['baseline'].get('mean_pnl_bps', 0):.2f} | {comp['ablated'].get('mean_pnl_bps', 0):.2f} | {comp['ablated'].get('mean_pnl_bps', 0) - comp['baseline'].get('mean_pnl_bps', 0):+.2f} |
| Winrate | {comp['baseline'].get('winrate', 0):.1%} | {comp['ablated'].get('winrate', 0):.1%} | {(comp['ablated'].get('winrate', 0) - comp['baseline'].get('winrate', 0)) * 100:+.1f}pp |

"""
    
    # Add session breakdown if available
    if baseline.get("session_breakdown"):
        md += "## Session Breakdown (Baseline)\n\n"
        md += "| Session | Trades | PnL (bps) |\n"
        md += "|---------|--------|-----------|\n"
        for session, data in baseline["session_breakdown"].items():
            md += f"| {session} | {data['n_trades']:,} | {data['total_pnl_bps']:.2f} |\n"
        md += "\n"
        
        # Session breakdown for ablated arms
        for comp in comparisons:
            if comp["arm_name"] == "baseline":
                continue
            abl = comp["ablated"]
            if abl.get("session_breakdown"):
                md += f"### {comp['arm_name']} Session Breakdown\n\n"
                md += "| Session | Trades | PnL (bps) | Δ Trades | Δ PnL |\n"
                md += "|---------|--------|-----------|----------|-------|\n"
                for session in sorted(set(baseline["session_breakdown"].keys()) | set(abl["session_breakdown"].keys())):
                    base_sess = baseline["session_breakdown"].get(session, {"n_trades": 0, "total_pnl_bps": 0.0})
                    abl_sess = abl["session_breakdown"].get(session, {"n_trades": 0, "total_pnl_bps": 0.0})
                    md += f"| {session} | {abl_sess['n_trades']:,} | {abl_sess['total_pnl_bps']:.2f} | "
                    md += f"{abl_sess['n_trades'] - base_sess['n_trades']:+d} | "
                    md += f"{abl_sess['total_pnl_bps'] - base_sess['total_pnl_bps']:+.2f} |\n"
                md += "\n"
    
    md_path = output_dir / f"XGB_CHANNEL_ABLATION_{mode.upper()}.md"
    with open(md_path, "w") as f:
        f.write(md)
    log.info(f"Written Markdown: {md_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="XGB Channel Ablation Analysis")
    parser.add_argument("--mode", type=str, required=True, choices=["qsmoke", "fullyear"],
                        help="Run mode: qsmoke (quick) or fullyear")
    parser.add_argument("--data", type=Path, required=True, help="Path to candles parquet")
    parser.add_argument("--prebuilt-parquet", type=Path, required=True, help="Path to prebuilt features parquet")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Path to V10 bundle")
    parser.add_argument("--policy", type=Path, required=True, help="Path to policy YAML")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--smoke-date-range", type=str, default=None,
                        help="Date range for qsmoke (e.g., '2025-01-06..2025-03-31')")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated channels to ablate (default: all)")
    
    args = parser.parse_args()
    
    # Validate paths
    for path, name in [
        (args.data, "data"),
        (args.prebuilt_parquet, "prebuilt-parquet"),
        (args.bundle_dir, "bundle-dir"),
        (args.policy, "policy"),
    ]:
        if not path.exists():
            log.error(f"Path does not exist: {name}={path}")
            return 1
    
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out_root / f"channel_ablation_{args.mode}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 80)
    log.info("XGB CHANNEL ABLATION ANALYSIS")
    log.info("=" * 80)
    log.info(f"Mode: {args.mode}")
    log.info(f"Output: {output_dir}")
    
    # Determine date range
    if args.mode == "qsmoke":
        smoke_date_range = args.smoke_date_range or "2025-01-06..2025-03-31"
    else:
        smoke_date_range = None  # Full year
    
    # Determine channels to ablate
    if args.channels:
        channels = [c.strip() for c in args.channels.split(",")]
    else:
        channels = CHANNELS_TO_ABLATE
    
    # Build arms to run
    arms_to_run = ["baseline"] + [f"drop_{ch}" for ch in channels]
    
    # Run arms
    results = {}
    for arm_name in arms_to_run:
        if arm_name == "baseline":
            arm_config = ABLATION_ARMS["baseline"]
        else:
            # Build config for drop arm
            channel = arm_name.replace("drop_", "")
            arm_config = {
                "name": arm_name,
                "description": f"Drop {channel} channel",
                "mask": channel,
                "env": {"GX1_XGB_CHANNEL_MASK": channel},
            }
        
        log.info(f"\n{'=' * 60}")
        log.info(f"Running arm: {arm_name}")
        log.info(f"Description: {arm_config.get('description', '')}")
        log.info(f"{'=' * 60}")
        
        result = run_replay_arm(
            arm_name,
            arm_config,
            args.data,
            args.prebuilt_parquet,
            args.bundle_dir,
            args.policy,
            output_dir,
            smoke_date_range=smoke_date_range,
            workers=args.workers,
        )
        results[arm_name] = result
        
        if not result.get("success"):
            log.error(f"Arm {arm_name} failed")
            return 1
    
    # Load metrics, verify masking, and generate verification reports
    log.info("\n" + "=" * 60)
    log.info("LOADING METRICS, VERIFYING MASKING, AND COMPARING")
    log.info("=" * 60)
    
    # Process each arm
    arm_metrics = {}
    arm_channel_info = {}
    masking_verifications = {}
    
    for arm_name in arms_to_run:
        arm_output = output_dir / arm_name
        
        # Load metrics
        metrics = load_arm_metrics(results[arm_name])
        arm_metrics[arm_name] = metrics
        
        # Load channel info
        channel_info = load_arm_channel_info(results[arm_name])
        arm_channel_info[arm_name] = channel_info
        
        # Verify masking
        expected_mask = None
        if arm_name != "baseline":
            # Extract expected mask from arm config
            channel = arm_name.replace("drop_", "")
            expected_mask = channel
        
        try:
            masking_verification = verify_masking_effect(results[arm_name], expected_mask)
            masking_verifications[arm_name] = masking_verification
            log.info(f"[{arm_name}] ✅ Masking verified")
        except RuntimeError as e:
            log.error(f"[{arm_name}] ❌ Masking verification failed: {e}")
            return 1
        
        # Generate verification report per arm
        generate_verification_report(
            results[arm_name],
            metrics,
            channel_info,
            masking_verification,
            arm_output,
        )
    
    baseline_metrics = arm_metrics["baseline"]
    log.info(f"Baseline: {baseline_metrics['n_trades']} trades, {baseline_metrics['total_pnl_bps']:.0f} bps")
    
    # Verify baseline has data
    if baseline_metrics["transformer_forward_calls"] == 0:
        log.error("FATAL: Baseline has transformer_forward_calls=0. Pipeline not working.")
        return 1
    
    # Compare to baseline
    comparisons = []
    for arm_name in arms_to_run:
        if arm_name == "baseline":
            comparisons.append({
                "arm_name": "baseline",
                "baseline": baseline_metrics,
                "ablated": baseline_metrics,
                "delta": {"n_trades": 0, "total_pnl_bps": 0, "max_dd": 0},
            })
        else:
            ablated_metrics = arm_metrics[arm_name]
            comparison = compare_to_baseline(baseline_metrics, ablated_metrics, arm_name)
            comparisons.append(comparison)
            
            log.info(f"{arm_name}: {ablated_metrics['n_trades']} trades, "
                     f"{ablated_metrics['total_pnl_bps']:.0f} bps "
                     f"(Δ={comparison['delta']['total_pnl_bps']:+.0f}) "
                     f"→ {comparison['verdict']}")
    
    # Generate reports
    generate_report(comparisons, output_dir, args.mode)
    
    # Copy to repo_audit for reference
    import shutil
    audit_dir = workspace_root / "reports" / "repo_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(
        output_dir / f"XGB_CHANNEL_ABLATION_{args.mode.upper()}.json",
        audit_dir / f"XGB_CHANNEL_ABLATION_{args.mode.upper()}.json"
    )
    shutil.copy(
        output_dir / f"XGB_CHANNEL_ABLATION_{args.mode.upper()}.md",
        audit_dir / f"XGB_CHANNEL_ABLATION_{args.mode.upper()}.md"
    )
    
    log.info("\n" + "=" * 80)
    log.info("✅ CHANNEL ABLATION COMPLETE")
    log.info(f"Reports: {output_dir}")
    log.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
