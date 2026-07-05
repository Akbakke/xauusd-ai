#!/usr/bin/env python3
"""
Print Entry Feature Flow

Standalone script to audit entry feature flow telemetry from replay output.
Reads ENTRY_FEATURES_USED.json, FEATURE_MASK_APPLIED.json, and ENTRY_FEATURES_TELEMETRY.json
and prints a human-readable summary.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return None


def print_entry_features_used(data: Dict[str, Any]) -> None:
    """Print entry features used summary."""
    print("\n" + "=" * 80)
    print("ENTRY FEATURES USED")
    print("=" * 80)
    
    seq_features = data.get("seq_features", {})
    snap_features = data.get("snap_features", {})
    xgb_seq = data.get("xgb_seq_channels", {})
    xgb_snap = data.get("xgb_snap_channels", {})
    
    print(f"\nSequence Features: {seq_features.get('count', 0)}")
    for i, name in enumerate(seq_features.get("names", [])[:20], 1):
        print(f"  {i:2d}. {name}")
    if len(seq_features.get("names", [])) > 20:
        print(f"  ... and {len(seq_features.get('names', [])) - 20} more")
    
    print(f"\nSnapshot Features: {snap_features.get('count', 0)}")
    for i, name in enumerate(snap_features.get("names", [])[:20], 1):
        print(f"  {i:2d}. {name}")
    if len(snap_features.get("names", [])) > 20:
        print(f"  ... and {len(snap_features.get('names', [])) - 20} more")
    
    print(f"\nXGB Sequence Channels: {xgb_seq.get('count', 0)}")
    for i, name in enumerate(xgb_seq.get("names", []), 1):
        print(f"  {i}. {name}")
    
    print(f"\nXGB Snapshot Channels: {xgb_snap.get('count', 0)}")
    for i, name in enumerate(xgb_snap.get("names", []), 1):
        print(f"  {i}. {name}")


def print_gate_stats(data: Dict[str, Any]) -> None:
    """Print gate statistics."""
    gate_stats = data.get("gate_stats", {})
    if not gate_stats:
        return
    
    print("\n" + "=" * 80)
    print("GATE STATISTICS")
    print("=" * 80)
    print(f"\n{'Gate Name':<30} {'Executed':<12} {'Blocked':<12} {'Passed':<12}")
    print("-" * 80)
    
    for gate_name, stats in sorted(gate_stats.items()):
        executed = stats.get("executed", 0)
        blocked = stats.get("blocked", 0)
        passed = stats.get("passed", 0)
        print(f"{gate_name:<30} {executed:<12} {blocked:<12} {passed:<12}")


def print_xgb_stats(data: Dict[str, Any]) -> None:
    """Print XGB statistics."""
    xgb_stats = data.get("xgb_stats", {})
    if not xgb_stats:
        return
    
    print("\n" + "=" * 80)
    print("XGB FLOW STATISTICS")
    print("=" * 80)
    
    for metric, count in sorted(xgb_stats.items()):
        print(f"  {metric}: {count}")


def print_mask_info(data: Dict[str, Any]) -> None:
    """Print feature masking information."""
    mask_telemetry = data.get("mask_telemetry")
    if not mask_telemetry or not mask_telemetry.get("mask_enabled"):
        return
    
    print("\n" + "=" * 80)
    print("FEATURE MASKING")
    print("=" * 80)
    
    print(f"\nMask Enabled: {mask_telemetry.get('mask_enabled')}")
    print(f"Mask Families: {', '.join(mask_telemetry.get('mask_families', []))}")
    print(f"Mask Strategy: {mask_telemetry.get('mask_strategy', 'N/A')}")
    
    masked_features = mask_telemetry.get("masked_features", [])
    print(f"\nMasked Features: {len(masked_features)}")
    for i, feat_name in enumerate(masked_features[:20], 1):
        print(f"  {i:2d}. {feat_name}")
    if len(masked_features) > 20:
        print(f"  ... and {len(masked_features) - 20} more")
    
    sample_values = mask_telemetry.get("sample_masked_values", {})
    if sample_values:
        print(f"\nSample Masked Values (first 10):")
        for feat_name, value in list(sample_values.items())[:10]:
            print(f"  {feat_name}: {value}")


def print_transformer_input_samples(data: Dict[str, Any]) -> None:
    """Print transformer input samples."""
    transformer_inputs = data.get("transformer_inputs", [])
    if not transformer_inputs:
        return
    
    print("\n" + "=" * 80)
    print("TRANSFORMER INPUT SAMPLES (first 3)")
    print("=" * 80)
    
    for i, sample in enumerate(transformer_inputs[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Sequence shape: {sample.get('seq_shape')}")
        print(f"  Snapshot shape: {sample.get('snap_shape')}")
        print(f"  XGB Seq Values: {sample.get('xgb_seq_values', {})}")
        print(f"  XGB Snap Values: {sample.get('xgb_snap_values', {})}")


def main():
    parser = argparse.ArgumentParser(
        description="Print entry feature flow telemetry from replay output"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Replay output directory (chunk directory or run directory)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        help="Chunk index (if reading from chunk directory)",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    if args.chunk is not None:
        telemetry_dir = args.output_dir / f"chunk_{args.chunk}"
    else:
        # Try to find chunk directories
        chunk_dirs = sorted(args.output_dir.glob("chunk_*"))
        if chunk_dirs:
            telemetry_dir = chunk_dirs[0]  # Use first chunk
            print(f"Found {len(chunk_dirs)} chunk directories, using {telemetry_dir.name}")
        else:
            telemetry_dir = args.output_dir
    
    # Load files
    entry_features_used = load_json(telemetry_dir / "ENTRY_FEATURES_USED.json")
    mask_applied = load_json(telemetry_dir / "FEATURE_MASK_APPLIED.json")
    telemetry = load_json(telemetry_dir / "ENTRY_FEATURES_TELEMETRY.json")
    
    if not entry_features_used and not telemetry:
        print(f"Error: No telemetry files found in {telemetry_dir}", file=sys.stderr)
        print(f"Expected: ENTRY_FEATURES_USED.json or ENTRY_FEATURES_TELEMETRY.json", file=sys.stderr)
        sys.exit(1)
    
    # Print summaries
    if entry_features_used:
        print_entry_features_used(entry_features_used)
        print_gate_stats(entry_features_used)
        print_xgb_stats(entry_features_used)
    
    if mask_applied:
        print_mask_info({"mask_telemetry": mask_applied})
    
    if telemetry:
        print_transformer_input_samples(telemetry)
        if not entry_features_used:
            # Fallback: use telemetry data
            print_gate_stats(telemetry)
            print_xgb_stats(telemetry)
            if telemetry.get("mask_telemetry"):
                print_mask_info(telemetry)
    
    print("\n" + "=" * 80)
    print("END")
    print("=" * 80)


if __name__ == "__main__":
    main()
