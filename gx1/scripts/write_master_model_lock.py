#!/usr/bin/env python3
"""
Write MASTER_MODEL_LOCK.json - the single source of truth for model authorization.

Usage:
    python3 gx1/scripts/write_master_model_lock.py --bundle-dir <path> --xgb-mode universal_multihead
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


def compute_file_sha256(filepath: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not filepath.exists():
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Write MASTER_MODEL_LOCK.json"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Bundle directory (default: GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/)"
    )
    parser.add_argument(
        "--xgb-mode",
        type=str,
        default="universal_multihead",
        choices=["universal_multihead", "universal", "session"],
        help="XGB mode"
    )
    parser.add_argument(
        "--require-go-marker",
        type=int,
        default=1,
        help="Require GO marker (default: 1)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WRITE MASTER MODEL LOCK")
    print("=" * 60)
    
    # Resolve bundle dir
    gx1_data = resolve_gx1_data_dir()
    if args.bundle_dir:
        bundle_dir = args.bundle_dir
    else:
        bundle_dir = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
    
    print(f"Bundle dir: {bundle_dir}")
    print(f"XGB mode: {args.xgb_mode}")
    
    if not bundle_dir.exists():
        print(f"ERROR: Bundle dir not found: {bundle_dir}")
        return 1
    
    # Determine model files based on mode
    if args.xgb_mode == "universal_multihead":
        model_filename = "xgb_universal_multihead_v2.joblib"
        meta_filename = "xgb_universal_multihead_v2_meta.json"
        go_marker_filename = "XGB_MULTIHEAD_GO_MARKER.json"
        no_go_marker_filename = "XGB_MULTIHEAD_NO_GO.json"
    elif args.xgb_mode == "universal":
        model_filename = "xgb_universal_v1.joblib"
        meta_filename = "xgb_universal_v1_meta.json"
        go_marker_filename = "XGB_UNIVERSAL_GO_MARKER.json"
        no_go_marker_filename = "XGB_UNIVERSAL_NO_GO.json"
    else:
        print(f"ERROR: Session mode not supported for master lock")
        return 1
    
    model_path = bundle_dir / model_filename
    meta_path = bundle_dir / meta_filename
    go_marker_path = bundle_dir / go_marker_filename
    
    # Check model exists
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1
    
    # Compute SHAs
    print("\nComputing SHAs...")
    model_sha = compute_file_sha256(model_path)
    meta_sha = compute_file_sha256(meta_path)
    go_marker_sha = compute_file_sha256(go_marker_path)
    
    print(f"  Model: {model_sha[:16]}...")
    print(f"  Meta: {meta_sha[:16] if meta_sha else 'NOT FOUND'}...")
    if go_marker_sha:
        print(f"  GO Marker: {go_marker_sha[:16]}... ✅")
    else:
        print(f"  GO Marker: NOT FOUND (will be set to null)")
    
    # Load meta to get schema hash
    schema_hash = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        schema_hash = meta.get("schema_hash")
        print(f"  Schema hash: {schema_hash}")
    
    # Contract files
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"
    
    # Load output contract to extract injection channels
    injection_channels = []
    injection_mode = "current_session"
    if output_contract_path.exists():
        with open(output_contract_path, "r") as f:
            output_contract = json.load(f)
        injection_mode = output_contract.get("injection_mode", "current_session")
        template = output_contract.get("injection_channels_template", [])
        sessions = output_contract.get("sessions", ["EU", "US", "OVERLAP"])
        
        # Build actual channel names based on injection mode
        if injection_mode == "current_session":
            # Phase 1: inject only current session (e.g., p_long_xgb_EU)
            for session in sessions:
                for tmpl in template:
                    channel = tmpl.replace("{session}", session)
                    injection_channels.append(channel)
        else:
            # Phase 2: inject all sessions (future)
            for tmpl in template:
                for session in sessions:
                    channel = tmpl.replace("{session}", session)
                    injection_channels.append(channel)
    
    contracts = {
        "feature_contract": {
            "path": str(feature_contract_path),
            "sha256": compute_file_sha256(feature_contract_path),
        },
        "sanitizer_config": {
            "path": str(sanitizer_config_path),
            "sha256": compute_file_sha256(sanitizer_config_path),
        },
        "output_contract": {
            "path": str(output_contract_path),
            "sha256": compute_file_sha256(output_contract_path),
        },
    }
    
    print("\nContract SHAs:")
    for name, info in contracts.items():
        sha = info["sha256"]
        print(f"  {name}: {sha[:16] if sha else 'NOT FOUND'}...")
    
    print(f"\nInjection contract:")
    print(f"  Mode: {injection_mode}")
    print(f"  Channels ({len(injection_channels)}): {injection_channels[:4]}...")
    
    # Build lock
    lock = {
        "version": "v1",
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "description": "Master model lock - single source of truth for authorized model",
        "xgb_mode": args.xgb_mode,
        "model_path_relative": model_filename,
        "model_sha256": model_sha,
        "meta_path_relative": meta_filename,
        "meta_sha256": meta_sha,
        "schema_hash": schema_hash,
        "go_marker_filename": go_marker_filename,
        "go_marker_sha256": go_marker_sha,  # Set when GO marker exists
        "no_go_marker_filename": no_go_marker_filename,
        "require_go_marker": bool(args.require_go_marker),
        "contracts": contracts,
        "injection_contract": {
            "mode": injection_mode,
            "channels": injection_channels,
            "expected_channels_count": len(injection_channels),
        },
        "invariants": {
            "no_fallback": True,
            "no_legacy_paths": True,
            "no_auto_resolve": True,
            "require_feature_names": True,
        },
    }
    
    # Check GO marker status
    if args.require_go_marker:
        if go_marker_sha:
            print(f"\n✅ GO marker found")
            lock["promotion_status"] = "GO"
        else:
            no_go_path = bundle_dir / no_go_marker_filename
            if no_go_path.exists():
                print(f"\n⚠️  NO-GO marker found (model not promoted)")
                lock["promotion_status"] = "NO-GO"
            else:
                print(f"\n⚠️  No promotion marker found (model not evaluated)")
                lock["promotion_status"] = "NOT_EVALUATED"
    
    # Write lock
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    tmp_path = lock_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(lock, f, indent=2)
    tmp_path.rename(lock_path)
    
    print(f"\n✅ MASTER_MODEL_LOCK.json written: {lock_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("LOCK SUMMARY")
    print("=" * 60)
    print(f"XGB Mode: {args.xgb_mode}")
    print(f"Model: {model_filename}")
    print(f"Model SHA: {model_sha}")
    print(f"Schema hash: {schema_hash}")
    print(f"Promotion: {lock.get('promotion_status', 'UNKNOWN')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
