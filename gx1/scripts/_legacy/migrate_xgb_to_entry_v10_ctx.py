#!/usr/bin/env python3
"""
Migrate legacy entry_v10 XGB models to entry_v10_ctx bundle.

This script copies XGB models from entry_v10/ to entry_v10_ctx/<bundle>/
and updates the entry config to point to the new paths.

Usage:
    python3 gx1/scripts/migrate_xgb_to_entry_v10_ctx.py --dry-run
    python3 gx1/scripts/migrate_xgb_to_entry_v10_ctx.py --execute
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    
    # Fallback
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy entry_v10 XGB models to entry_v10_ctx bundle"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration"
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default="FULLYEAR_2025_GATED_FUSION",
        help="Target bundle name (default: FULLYEAR_2025_GATED_FUSION)"
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("ERROR: Must specify either --dry-run or --execute")
        return 1
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Source directory (legacy)
    legacy_dir = gx1_data / "models" / "models" / "entry_v10"
    if not legacy_dir.exists():
        print(f"ERROR: Legacy directory not found: {legacy_dir}")
        return 1
    
    print(f"Legacy directory: {legacy_dir}")
    
    # Target directory (entry_v10_ctx bundle)
    target_bundle_dir = gx1_data / "models" / "models" / "entry_v10_ctx" / args.bundle_name
    if not target_bundle_dir.exists():
        print(f"ERROR: Target bundle directory not found: {target_bundle_dir}")
        return 1
    
    print(f"Target bundle: {target_bundle_dir}")
    
    # Find legacy XGB models
    legacy_models = list(legacy_dir.glob("xgb_entry_*.joblib"))
    if not legacy_models:
        print("ERROR: No legacy XGB models found")
        return 1
    
    print(f"\nFound {len(legacy_models)} legacy XGB models:")
    for model in sorted(legacy_models):
        size_kb = model.stat().st_size / 1024
        sha256 = compute_file_sha256(model)[:16]
        print(f"  {model.name} ({size_kb:.1f} KB, SHA256: {sha256}...)")
    
    # Define migration mapping
    migration_map = {}
    for model in legacy_models:
        # Extract session from filename (xgb_entry_EU_v10.joblib -> EU)
        name_parts = model.stem.split("_")  # ['xgb', 'entry', 'EU', 'v10']
        if len(name_parts) >= 3:
            session = name_parts[2]  # EU, US, OVERLAP, ASIA
            # New name format: xgb_{session}.joblib
            new_name = f"xgb_{session}.joblib"
            migration_map[model] = target_bundle_dir / new_name
    
    print(f"\n{'='*60}")
    print("Migration Plan")
    print(f"{'='*60}")
    
    for src, dst in sorted(migration_map.items()):
        print(f"  {src.name}")
        print(f"    -> {dst}")
        if dst.exists():
            print(f"       (WARNING: destination exists, will overwrite)")
    
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - No changes made")
        print(f"{'='*60}")
        print("\nTo execute, run with --execute")
        return 0
    
    # Execute migration
    print(f"\n{'='*60}")
    print("Executing Migration")
    print(f"{'='*60}")
    
    migrated = []
    for src, dst in sorted(migration_map.items()):
        print(f"  Copying {src.name} -> {dst.name}")
        shutil.copy2(src, dst)
        
        # Verify copy
        src_sha = compute_file_sha256(src)
        dst_sha = compute_file_sha256(dst)
        if src_sha != dst_sha:
            print(f"    ERROR: SHA256 mismatch after copy!")
            return 1
        
        migrated.append({
            "src": str(src),
            "dst": str(dst),
            "sha256": src_sha,
        })
        print(f"    ✅ Verified (SHA256: {src_sha[:16]}...)")
    
    # Write migration manifest
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "legacy_dir": str(legacy_dir),
        "target_bundle": str(target_bundle_dir),
        "migrated": migrated,
    }
    
    manifest_path = target_bundle_dir / "XGB_MIGRATION_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Wrote manifest: {manifest_path}")
    
    # Print next steps
    print(f"\n{'='*60}")
    print("Migration Complete")
    print(f"{'='*60}")
    print(f"Migrated {len(migrated)} XGB models to {target_bundle_dir}")
    print("\nNext steps:")
    print("1. Update entry_config to point to new paths:")
    print(f"   eu_model_path: models/entry_v10_ctx/{args.bundle_name}/xgb_EU.joblib")
    print(f"   us_model_path: models/entry_v10_ctx/{args.bundle_name}/xgb_US.joblib")
    print(f"   overlap_model_path: models/entry_v10_ctx/{args.bundle_name}/xgb_OVERLAP.joblib")
    print("\n2. Re-run write_model_used_capsule.py to verify non-legacy paths")
    print("\n3. Run audit to confirm TRUTH_USED_SET has no legacy paths")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
