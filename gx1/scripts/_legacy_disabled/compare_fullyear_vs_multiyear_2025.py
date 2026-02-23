#!/usr/bin/env python3
"""
Compare FULLYEAR 2025 vs MULTIYEAR 2025 Identity and Counters

Verifies that multiyear 2025 runs identical pipeline/config as fullyear 2025.

Usage:
    python gx1/scripts/compare_fullyear_vs_multiyear_2025.py \
        --fullyear-dir /path/to/fullyear/2025/truth/run \
        --multiyear-dir /path/to/MULTIYEAR_2020_2025_YYYYMMDD_HHMMSS \
        --year 2025
"""

import argparse
import json
import hashlib
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


@dataclass
class RunIdentity:
    """Identity fields for a run."""
    source: str = ""  # "fullyear" or "multiyear"
    run_dir: str = ""
    
    # Core identity
    policy_id: str = ""
    bundle_sha: str = ""
    replay_mode: str = ""
    temperature_scaling_effective: bool = False
    
    # XGB channels
    xgb_channels_seq: List[str] = field(default_factory=list)
    xgb_channels_snap: List[str] = field(default_factory=list)
    xgb_used_as: str = ""
    
    # Feature contract
    feature_fingerprint: str = ""
    feature_build_call_count: int = -1
    
    # Universe
    start_ts: str = ""
    end_ts: str = ""
    bars_total: int = 0
    
    # Session/timezone
    session_definition: str = ""
    timezone: str = ""
    
    # Thresholds/gates (serialized)
    thresholds_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunCounters:
    """Counter fields for a run."""
    source: str = ""
    
    # Bar funnel
    bars_seen: int = 0
    bars_processed: int = 0
    bars_reaching_entry_stage: int = 0
    eligibility_blocks: int = 0
    
    # Model calls
    model_attempts: int = 0
    transformer_forward_calls: int = 0
    xgb_pre_predict_calls: int = 0
    
    # Trades
    trades: int = 0
    total_pnl_bps: float = 0.0
    max_dd_bps: float = 0.0
    
    # Conversion
    conversion_rate: float = 0.0
    
    # Prebuilt
    prebuilt_used: bool = False
    lookup_hits: int = 0
    lookup_attempts: int = 0


def find_chunk_footer(run_dir: Path) -> Optional[Path]:
    """Find chunk_footer.json in a run directory."""
    # Try direct path
    direct = run_dir / "chunk_0" / "chunk_footer.json"
    if direct.exists():
        return direct
    
    # Try any chunk
    for chunk_dir in run_dir.glob("chunk_*"):
        footer = chunk_dir / "chunk_footer.json"
        if footer.exists():
            return footer
    
    return None


def find_run_identity(run_dir: Path) -> Optional[Path]:
    """Find RUN_IDENTITY.json or RUN_CTX.json."""
    for name in ["RUN_IDENTITY.json", "RUN_CTX.json", "run_identity.json"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def find_entry_features_master(run_dir: Path) -> Optional[Path]:
    """Find ENTRY_FEATURES_USED_MASTER.json."""
    # Try direct
    direct = run_dir / "ENTRY_FEATURES_USED_MASTER.json"
    if direct.exists():
        return direct
    
    # Try in chunk_0
    chunk = run_dir / "chunk_0" / "ENTRY_FEATURES_USED_MASTER.json"
    if chunk.exists():
        return chunk
    
    # Try ENTRY_FEATURES_USED.json
    for name in ["ENTRY_FEATURES_USED.json", "entry_features_used.json"]:
        for path in [run_dir / name, run_dir / "chunk_0" / name]:
            if path.exists():
                return path
    
    return None


def extract_identity(run_dir: Path, source: str) -> RunIdentity:
    """Extract identity from a run directory."""
    identity = RunIdentity(source=source, run_dir=str(run_dir))
    
    # Load chunk_footer
    footer_path = find_chunk_footer(run_dir)
    if footer_path:
        with open(footer_path) as f:
            footer = json.load(f)
        
        identity.xgb_channels_seq = footer.get("xgb_seq_channel_names", [])
        identity.xgb_channels_snap = footer.get("xgb_snap_channel_names", [])
        identity.xgb_used_as = footer.get("xgb_used_as", "")
        identity.feature_build_call_count = footer.get("feature_build_call_count", -1)
        identity.bars_total = footer.get("bars_seen", 0)
        identity.replay_mode = "PREBUILT" if footer.get("prebuilt_used", False) else "LIVE"
        identity.timezone = footer.get("prebuilt_gate_dump", {}).get("prebuilt_features_df_index_tz", "")
    
    # Load RUN_IDENTITY / RUN_CTX
    identity_path = find_run_identity(run_dir)
    if identity_path:
        with open(identity_path) as f:
            run_ctx = json.load(f)
        
        identity.policy_id = run_ctx.get("policy_id", run_ctx.get("policy", ""))
        identity.bundle_sha = run_ctx.get("bundle_sha", run_ctx.get("bundle_sha256", ""))
        identity.temperature_scaling_effective = run_ctx.get("temperature_scaling_effective", False)
        identity.start_ts = run_ctx.get("start_ts", "")
        identity.end_ts = run_ctx.get("end_ts", "")
        identity.feature_fingerprint = run_ctx.get("feature_fingerprint", "")
        
        # Extract thresholds if present
        thresholds = {}
        for key in ["entry_threshold", "exit_threshold", "confidence_threshold", "gates_config"]:
            if key in run_ctx:
                thresholds[key] = run_ctx[key]
        identity.thresholds_snapshot = thresholds
    
    # Load ENTRY_FEATURES_USED_MASTER for additional identity
    master_path = find_entry_features_master(run_dir)
    if master_path:
        with open(master_path) as f:
            master = json.load(f)
        
        if not identity.xgb_channels_seq:
            identity.xgb_channels_seq = master.get("xgb_seq_channel_names", [])
        if not identity.xgb_channels_snap:
            identity.xgb_channels_snap = master.get("xgb_snap_channel_names", [])
        if not identity.xgb_used_as:
            identity.xgb_used_as = master.get("xgb_used_as", "")
    
    return identity


def extract_counters(run_dir: Path, source: str) -> RunCounters:
    """Extract counters from a run directory."""
    counters = RunCounters(source=source)
    
    # Load chunk_footer
    footer_path = find_chunk_footer(run_dir)
    if footer_path:
        with open(footer_path) as f:
            footer = json.load(f)
        
        counters.bars_seen = footer.get("bars_seen", 0)
        counters.bars_processed = footer.get("bars_processed", 0)
        counters.bars_reaching_entry_stage = footer.get("bars_reaching_entry_stage", 0)
        counters.eligibility_blocks = footer.get("bars_skipped_pregate", 0) + footer.get("bars_skipped_warmup", 0)
        counters.transformer_forward_calls = footer.get("transformer_forward_calls", 0)
        counters.xgb_pre_predict_calls = footer.get("xgb_pre_predict_count", 0)
        counters.prebuilt_used = footer.get("prebuilt_used", False)
        counters.lookup_hits = footer.get("lookup_hits", 0)
        counters.lookup_attempts = footer.get("lookup_attempts", 0)
    
    # Load metrics
    metrics_patterns = [
        run_dir / "chunk_0" / "metrics_*.json",
        run_dir / "metrics_*.json",
    ]
    for pattern in metrics_patterns:
        for metrics_path in run_dir.parent.glob(str(pattern.relative_to(run_dir.parent))) if "*" in str(pattern) else ([pattern] if pattern.exists() else []):
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                counters.trades = metrics.get("n_trades", metrics.get("trades", 0))
                counters.total_pnl_bps = metrics.get("total_pnl_bps", 0.0)
                counters.max_dd_bps = metrics.get("max_dd_bps", metrics.get("maxdd_bps", 0.0))
                break
    
    # Try to get trades from footer if not in metrics
    if counters.trades == 0 and footer_path:
        with open(footer_path) as f:
            footer = json.load(f)
        counters.trades = footer.get("n_trades_closed", 0)
    
    # Load ENTRY_FEATURES_USED_MASTER for model attempts
    master_path = find_entry_features_master(run_dir)
    if master_path:
        with open(master_path) as f:
            master = json.load(f)
        
        model_entry = master.get("model_entry", {})
        counters.model_attempts = model_entry.get("total", 0)
        
        if not counters.transformer_forward_calls:
            counters.transformer_forward_calls = master.get("transformer_forward_calls", 0)
        if not counters.xgb_pre_predict_calls:
            counters.xgb_pre_predict_calls = master.get("xgb_pre_predict_count", 0)
    
    # Compute conversion rate
    if counters.transformer_forward_calls > 0:
        counters.conversion_rate = counters.trades / counters.transformer_forward_calls
    
    return counters


def compare_identities(fullyear: RunIdentity, multiyear: RunIdentity) -> Tuple[bool, Dict[str, Any]]:
    """Compare two run identities and return (match, diff)."""
    diff = {}
    hard_fail = False
    hard_fail_reasons = []
    
    # Critical fields (hard fail if mismatch)
    critical_fields = [
        ("policy_id", fullyear.policy_id, multiyear.policy_id),
        ("bundle_sha", fullyear.bundle_sha, multiyear.bundle_sha),
        ("replay_mode", fullyear.replay_mode, multiyear.replay_mode),
        ("xgb_used_as", fullyear.xgb_used_as, multiyear.xgb_used_as),
    ]
    
    for field_name, fy_val, my_val in critical_fields:
        if fy_val != my_val:
            diff[field_name] = {"fullyear": fy_val, "multiyear": my_val, "critical": True}
            hard_fail = True
            hard_fail_reasons.append(f"{field_name}: {fy_val} != {my_val}")
    
    # Non-critical fields (warn but don't fail)
    non_critical_fields = [
        ("temperature_scaling_effective", fullyear.temperature_scaling_effective, multiyear.temperature_scaling_effective),
        ("xgb_channels_seq", fullyear.xgb_channels_seq, multiyear.xgb_channels_seq),
        ("xgb_channels_snap", fullyear.xgb_channels_snap, multiyear.xgb_channels_snap),
        ("feature_fingerprint", fullyear.feature_fingerprint, multiyear.feature_fingerprint),
        ("timezone", fullyear.timezone, multiyear.timezone),
    ]
    
    for field_name, fy_val, my_val in non_critical_fields:
        if fy_val != my_val:
            diff[field_name] = {"fullyear": fy_val, "multiyear": my_val, "critical": False}
    
    # Check feature_build_call_count
    if fullyear.feature_build_call_count != 0 and fullyear.feature_build_call_count != -1:
        hard_fail = True
        hard_fail_reasons.append(f"fullyear.feature_build_call_count={fullyear.feature_build_call_count}")
    if multiyear.feature_build_call_count != 0 and multiyear.feature_build_call_count != -1:
        hard_fail = True
        hard_fail_reasons.append(f"multiyear.feature_build_call_count={multiyear.feature_build_call_count}")
    
    # Check replay_mode is PREBUILT
    if fullyear.replay_mode != "PREBUILT":
        hard_fail = True
        hard_fail_reasons.append(f"fullyear.replay_mode={fullyear.replay_mode}")
    if multiyear.replay_mode != "PREBUILT":
        hard_fail = True
        hard_fail_reasons.append(f"multiyear.replay_mode={multiyear.replay_mode}")
    
    # Check xgb_used_as is "pre"
    if fullyear.xgb_used_as != "pre" and fullyear.xgb_used_as:
        hard_fail = True
        hard_fail_reasons.append(f"fullyear.xgb_used_as={fullyear.xgb_used_as}")
    if multiyear.xgb_used_as != "pre" and multiyear.xgb_used_as:
        hard_fail = True
        hard_fail_reasons.append(f"multiyear.xgb_used_as={multiyear.xgb_used_as}")
    
    return not hard_fail, {
        "match": not hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "diff": diff,
    }


def compare_counters(fullyear: RunCounters, multiyear: RunCounters) -> Dict[str, Any]:
    """Compare two run counters and return diff."""
    diff = {}
    
    counter_fields = [
        "bars_seen",
        "bars_processed",
        "bars_reaching_entry_stage",
        "eligibility_blocks",
        "model_attempts",
        "transformer_forward_calls",
        "xgb_pre_predict_calls",
        "trades",
        "total_pnl_bps",
        "max_dd_bps",
        "conversion_rate",
        "lookup_hits",
        "lookup_attempts",
    ]
    
    for field in counter_fields:
        fy_val = getattr(fullyear, field)
        my_val = getattr(multiyear, field)
        
        if isinstance(fy_val, float) or isinstance(my_val, float):
            delta = my_val - fy_val if fy_val is not None and my_val is not None else 0
            pct = (delta / fy_val * 100) if fy_val and fy_val != 0 else 0
        else:
            delta = my_val - fy_val
            pct = (delta / fy_val * 100) if fy_val and fy_val != 0 else 0
        
        diff[field] = {
            "fullyear": fy_val,
            "multiyear": my_val,
            "delta": delta,
            "pct": round(pct, 2),
        }
    
    return diff


def generate_diff_md(
    identity_result: Dict[str, Any],
    counters_diff: Dict[str, Any],
    fullyear_identity: RunIdentity,
    multiyear_identity: RunIdentity,
    output_path: Path
):
    """Generate COMPARE_2025_DIFF.md."""
    lines = [
        "# FULLYEAR 2025 vs MULTIYEAR 2025 Comparison",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "---",
        "",
    ]
    
    # Verdict
    if identity_result["match"]:
        lines.append("## ✅ VERDICT: MATCH")
        lines.append("")
        lines.append("Identity fields match. Pipelines are comparable.")
    else:
        lines.append("## ❌ VERDICT: MISMATCH")
        lines.append("")
        lines.append("**Hard fail reasons:**")
        for reason in identity_result["hard_fail_reasons"]:
            lines.append(f"- {reason}")
    
    lines.extend([
        "",
        "---",
        "",
        "## Identity Comparison",
        "",
        "| Field | FULLYEAR | MULTIYEAR | Match |",
        "|-------|----------|-----------|-------|",
    ])
    
    # Identity table
    identity_fields = [
        ("policy_id", fullyear_identity.policy_id, multiyear_identity.policy_id),
        ("bundle_sha", fullyear_identity.bundle_sha[:16] if fullyear_identity.bundle_sha else "", 
         multiyear_identity.bundle_sha[:16] if multiyear_identity.bundle_sha else ""),
        ("replay_mode", fullyear_identity.replay_mode, multiyear_identity.replay_mode),
        ("xgb_used_as", fullyear_identity.xgb_used_as, multiyear_identity.xgb_used_as),
        ("xgb_channels_seq", str(fullyear_identity.xgb_channels_seq), str(multiyear_identity.xgb_channels_seq)),
        ("xgb_channels_snap", str(fullyear_identity.xgb_channels_snap), str(multiyear_identity.xgb_channels_snap)),
        ("feature_build_call_count", str(fullyear_identity.feature_build_call_count), str(multiyear_identity.feature_build_call_count)),
        ("bars_total", str(fullyear_identity.bars_total), str(multiyear_identity.bars_total)),
    ]
    
    for field_name, fy_val, my_val in identity_fields:
        match = "✅" if fy_val == my_val else "❌"
        lines.append(f"| {field_name} | {fy_val} | {my_val} | {match} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Counter Comparison",
        "",
        "| Counter | FULLYEAR | MULTIYEAR | Delta | Δ% |",
        "|---------|----------|-----------|-------|-----|",
    ])
    
    # Counter table
    for field, data in counters_diff.items():
        fy_val = data["fullyear"]
        my_val = data["multiyear"]
        delta = data["delta"]
        pct = data["pct"]
        
        if isinstance(fy_val, float):
            fy_str = f"{fy_val:.2f}"
            my_str = f"{my_val:.2f}"
            delta_str = f"{delta:+.2f}"
        else:
            fy_str = f"{fy_val:,}" if isinstance(fy_val, int) else str(fy_val)
            my_str = f"{my_val:,}" if isinstance(my_val, int) else str(my_val)
            delta_str = f"{delta:+,}" if isinstance(delta, int) else f"{delta:+.2f}"
        
        lines.append(f"| {field} | {fy_str} | {my_str} | {delta_str} | {pct:+.1f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Paths",
        "",
        f"- FULLYEAR: `{fullyear_identity.run_dir}`",
        f"- MULTIYEAR: `{multiyear_identity.run_dir}`",
        "",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare FULLYEAR 2025 vs MULTIYEAR 2025 Identity and Counters"
    )
    parser.add_argument(
        "--fullyear-dir",
        type=Path,
        required=True,
        help="Path to FULLYEAR 2025 truth run output"
    )
    parser.add_argument(
        "--multiyear-dir",
        type=Path,
        required=True,
        help="Path to MULTIYEAR_2020_2025_YYYYMMDD_HHMMSS output"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to compare (default: 2025)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: multiyear-dir)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.fullyear_dir.exists():
        print(f"ERROR: FULLYEAR dir does not exist: {args.fullyear_dir}")
        return 1
    
    if not args.multiyear_dir.exists():
        print(f"ERROR: MULTIYEAR dir does not exist: {args.multiyear_dir}")
        return 1
    
    # Find multiyear year subdir
    multiyear_year_dir = args.multiyear_dir / str(args.year)
    if not multiyear_year_dir.exists():
        # Try direct (if multiyear_dir IS the year dir)
        multiyear_year_dir = args.multiyear_dir
    
    # Output directory
    output_dir = args.output_dir or args.multiyear_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COMPARE FULLYEAR vs MULTIYEAR 2025")
    print("=" * 60)
    print(f"FULLYEAR: {args.fullyear_dir}")
    print(f"MULTIYEAR: {multiyear_year_dir}")
    print()
    
    # Extract identities
    print("Extracting identities...")
    fullyear_identity = extract_identity(args.fullyear_dir, "fullyear")
    multiyear_identity = extract_identity(multiyear_year_dir, "multiyear")
    
    # Extract counters
    print("Extracting counters...")
    fullyear_counters = extract_counters(args.fullyear_dir, "fullyear")
    multiyear_counters = extract_counters(multiyear_year_dir, "multiyear")
    
    # Compare identities
    print("Comparing identities...")
    match, identity_result = compare_identities(fullyear_identity, multiyear_identity)
    
    # Compare counters
    print("Comparing counters...")
    counters_diff = compare_counters(fullyear_counters, multiyear_counters)
    
    # Write COMPARE_2025_IDENTITY.json
    identity_output = {
        "fullyear": asdict(fullyear_identity),
        "multiyear": asdict(multiyear_identity),
        "comparison": identity_result,
    }
    identity_path = output_dir / "COMPARE_2025_IDENTITY.json"
    with open(identity_path, "w") as f:
        json.dump(identity_output, f, indent=2, default=str)
    print(f"Written: {identity_path}")
    
    # Write COMPARE_2025_COUNTERS.json
    counters_output = {
        "fullyear": asdict(fullyear_counters),
        "multiyear": asdict(multiyear_counters),
        "diff": counters_diff,
    }
    counters_path = output_dir / "COMPARE_2025_COUNTERS.json"
    with open(counters_path, "w") as f:
        json.dump(counters_output, f, indent=2, default=str)
    print(f"Written: {counters_path}")
    
    # Write COMPARE_2025_DIFF.md
    diff_path = output_dir / "COMPARE_2025_DIFF.md"
    generate_diff_md(identity_result, counters_diff, fullyear_identity, multiyear_identity, diff_path)
    
    # Print verdict
    print()
    print("=" * 60)
    if match:
        print("✅ VERDICT: MATCH — Pipelines are comparable")
    else:
        print("❌ VERDICT: MISMATCH")
        print("Reasons:")
        for reason in identity_result["hard_fail_reasons"]:
            print(f"  - {reason}")
    print("=" * 60)
    
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
