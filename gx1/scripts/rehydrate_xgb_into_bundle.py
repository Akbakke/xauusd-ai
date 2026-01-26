#!/usr/bin/env python3
"""
rehydrate_xgb_into_bundle.py — Copy XGB models from a source bundle to a target bundle.

This script safely copies XGB models (xgb_EU.pkl, xgb_US.pkl, xgb_OVERLAP.pkl) from
a source bundle that has them to a target bundle that is missing them.

Safety checks:
  - feature_contract_hash must match between source and target
  - Source bundle must have all requested XGB sessions
  - Target bundle must have model_state_dict.pt and bundle_metadata.json

Usage:
    # Dry run (default)
    python gx1/scripts/rehydrate_xgb_into_bundle.py \\
        --source-bundle /path/to/source_bundle \\
        --target-bundle /path/to/FULLYEAR_2025_GATED_FUSION \\
        --sessions EU,US,OVERLAP

    # Apply changes
    python gx1/scripts/rehydrate_xgb_into_bundle.py \\
        --source-bundle /path/to/source_bundle \\
        --target-bundle /path/to/FULLYEAR_2025_GATED_FUSION \\
        --sessions EU,US,OVERLAP \\
        --apply
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure gx1 is importable
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_contract_hash(bundle_dir: Path) -> Optional[str]:
    """Get feature contract hash from bundle."""
    # Try file first
    hash_file = bundle_dir / "feature_contract_hash.txt"
    if hash_file.exists():
        return hash_file.read_text().strip()
    
    # Try metadata
    meta_file = bundle_dir / "bundle_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            return meta.get("feature_contract_hash")
        except Exception:
            pass
    
    return None


def validate_source_bundle(source_dir: Path, sessions: List[str]) -> Dict:
    """Validate source bundle has all required XGB models."""
    issues = []
    xgb_files = {}
    
    # Check model_state_dict.pt
    if not (source_dir / "model_state_dict.pt").exists():
        issues.append("missing model_state_dict.pt")
    
    # Check bundle_metadata.json
    if not (source_dir / "bundle_metadata.json").exists():
        issues.append("missing bundle_metadata.json")
    
    # Check XGB models for each session
    for session in sessions:
        xgb_path = source_dir / f"xgb_{session}.pkl"
        if xgb_path.exists():
            xgb_files[session] = {
                "path": str(xgb_path),
                "sha256": compute_file_sha256(xgb_path),
                "size": xgb_path.stat().st_size,
            }
        else:
            issues.append(f"missing xgb_{session}.pkl")
    
    # Get contract hash
    contract_hash = get_contract_hash(source_dir)
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "xgb_files": xgb_files,
        "contract_hash": contract_hash,
    }


def validate_target_bundle(target_dir: Path) -> Dict:
    """Validate target bundle structure."""
    issues = []
    
    # Check model_state_dict.pt
    if not (target_dir / "model_state_dict.pt").exists():
        issues.append("missing model_state_dict.pt")
    
    # Check bundle_metadata.json
    if not (target_dir / "bundle_metadata.json").exists():
        issues.append("missing bundle_metadata.json")
    
    # Get contract hash
    contract_hash = get_contract_hash(target_dir)
    
    # Check for existing XGB files
    existing_xgb = []
    for f in target_dir.glob("xgb_*.pkl"):
        existing_xgb.append(f.name)
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "contract_hash": contract_hash,
        "existing_xgb": existing_xgb,
    }


def rehydrate_xgb(
    source_dir: Path,
    target_dir: Path,
    sessions: List[str],
    dry_run: bool = True,
) -> Dict:
    """Copy XGB models from source to target bundle."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "source_bundle": str(source_dir),
        "target_bundle": str(target_dir),
        "sessions": sessions,
        "dry_run": dry_run,
        "success": False,
        "actions": [],
        "errors": [],
    }
    
    # Validate source bundle
    print(f"[VALIDATE] Checking source bundle: {source_dir}")
    source_info = validate_source_bundle(source_dir, sessions)
    if not source_info["valid"]:
        report["errors"].append(f"Source bundle validation failed: {source_info['issues']}")
        print(f"[ERROR] Source bundle validation failed: {source_info['issues']}")
        return report
    print(f"[VALIDATE] Source bundle OK, XGB sessions: {list(source_info['xgb_files'].keys())}")
    
    # Validate target bundle
    print(f"[VALIDATE] Checking target bundle: {target_dir}")
    target_info = validate_target_bundle(target_dir)
    if not target_info["valid"]:
        report["errors"].append(f"Target bundle validation failed: {target_info['issues']}")
        print(f"[ERROR] Target bundle validation failed: {target_info['issues']}")
        return report
    print(f"[VALIDATE] Target bundle OK, existing XGB: {target_info['existing_xgb']}")
    
    # Check contract hash compatibility
    source_hash = source_info["contract_hash"]
    target_hash = target_info["contract_hash"]
    
    if source_hash and target_hash and source_hash != target_hash:
        report["errors"].append(
            f"Feature contract hash mismatch: source={source_hash[:16]}... target={target_hash[:16]}..."
        )
        print(f"[ERROR] Feature contract hash mismatch!")
        print(f"        Source: {source_hash}")
        print(f"        Target: {target_hash}")
        return report
    
    if source_hash and target_hash:
        print(f"[VALIDATE] Contract hash match: {source_hash[:16]}...")
    elif not source_hash:
        print(f"[WARN] Source bundle has no contract hash - proceeding with caution")
    elif not target_hash:
        print(f"[WARN] Target bundle has no contract hash - proceeding with caution")
    
    # Copy XGB files
    xgb_copies = []
    for session in sessions:
        src_path = source_dir / f"xgb_{session}.pkl"
        dst_path = target_dir / f"xgb_{session}.pkl"
        
        action = {
            "session": session,
            "source": str(src_path),
            "target": str(dst_path),
            "sha256": source_info["xgb_files"][session]["sha256"],
            "size": source_info["xgb_files"][session]["size"],
        }
        
        if dry_run:
            print(f"[DRY-RUN] Would copy: {src_path.name} -> {dst_path}")
            action["status"] = "would_copy"
        else:
            # Atomic copy: tmp -> rename
            tmp_path = dst_path.with_suffix(".pkl.tmp")
            try:
                shutil.copy2(src_path, tmp_path)
                tmp_path.rename(dst_path)
                print(f"[COPY] Copied: {src_path.name} -> {dst_path}")
                action["status"] = "copied"
            except Exception as e:
                print(f"[ERROR] Failed to copy {src_path.name}: {e}")
                action["status"] = "failed"
                action["error"] = str(e)
                report["errors"].append(f"Failed to copy {src_path.name}: {e}")
        
        xgb_copies.append(action)
    
    report["actions"] = xgb_copies
    
    # Update bundle_metadata.json
    if not dry_run and not report["errors"]:
        meta_path = target_dir / "bundle_metadata.json"
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            
            # Add xgb_models_by_session mapping
            metadata["xgb_models_by_session"] = {
                session: f"xgb_{session}.pkl"
                for session in sessions
            }
            
            # Add rehydration info
            metadata["xgb_rehydrated"] = {
                "timestamp": datetime.now().isoformat(),
                "source_bundle": str(source_dir),
                "sessions": sessions,
                "xgb_hashes": {
                    session: source_info["xgb_files"][session]["sha256"]
                    for session in sessions
                },
            }
            
            # Atomic write
            tmp_meta = meta_path.with_suffix(".json.tmp")
            with open(tmp_meta, "w") as f:
                json.dump(metadata, f, indent=2)
            tmp_meta.rename(meta_path)
            
            print(f"[UPDATE] Updated bundle_metadata.json with xgb_models_by_session")
            report["actions"].append({
                "type": "metadata_update",
                "status": "updated",
            })
        except Exception as e:
            print(f"[ERROR] Failed to update bundle_metadata.json: {e}")
            report["errors"].append(f"Failed to update metadata: {e}")
    
    # Write rehydration report
    if not dry_run:
        rehydrate_report_path = target_dir / "BUNDLE_REHYDRATE_REPORT.json"
        with open(rehydrate_report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[REPORT] Wrote: {rehydrate_report_path}")
    
    report["success"] = len(report["errors"]) == 0
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Copy XGB models from source bundle to target bundle"
    )
    parser.add_argument(
        "--source-bundle",
        type=str,
        required=True,
        help="Path to source bundle (must have XGB models)"
    )
    parser.add_argument(
        "--target-bundle",
        type=str,
        required=True,
        help="Path to target bundle (missing XGB models)"
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="EU,US,OVERLAP",
        help="Comma-separated list of XGB sessions to copy (default: EU,US,OVERLAP)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run (default: true)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (disables dry-run)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    source_dir = Path(args.source_bundle).resolve()
    target_dir = Path(args.target_bundle).resolve()
    sessions = [s.strip() for s in args.sessions.split(",")]
    
    # Check paths exist
    if not source_dir.exists():
        print(f"[ERROR] Source bundle does not exist: {source_dir}")
        sys.exit(1)
    if not target_dir.exists():
        print(f"[ERROR] Target bundle does not exist: {target_dir}")
        sys.exit(1)
    
    # Determine dry-run mode
    dry_run = not args.apply
    
    print("=" * 80)
    print("XGB REHYDRATION")
    print("=" * 80)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Sessions: {sessions}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}")
    print("=" * 80)
    print("")
    
    # Run rehydration
    report = rehydrate_xgb(source_dir, target_dir, sessions, dry_run=dry_run)
    
    print("")
    print("=" * 80)
    if report["success"]:
        if dry_run:
            print("✅ DRY-RUN SUCCESS - Run with --apply to actually copy files")
        else:
            print("✅ REHYDRATION SUCCESS")
    else:
        print(f"❌ REHYDRATION FAILED: {report['errors']}")
    print("=" * 80)
    
    sys.exit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
