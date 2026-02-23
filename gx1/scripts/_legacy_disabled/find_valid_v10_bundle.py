#!/usr/bin/env python3
"""
find_valid_v10_bundle.py — Scan for valid V10 hybrid bundles with XGB models.

A bundle is "VALID_V10_HYBRID" if ALL of these exist:
  a) model_state_dict.pt
  b) bundle_metadata.json
  c) feature_contract_hash.txt (or contract hash field in metadata)
  d) xgb_EU.pkl, xgb_US.pkl, xgb_OVERLAP.pkl
  e) bundle_metadata has xgb_models_by_session OR we can deterministically map files to sessions

Classification:
  - OK: All requirements met
  - NO_XGB: Missing one or more xgb_*.pkl
  - LEGACY: v9/entry_v9/"OANDA_DEMO_V9" in path or metadata
  - PARTIAL: Missing critical files (model_state_dict.pt, bundle_metadata.json)

Usage:
    python gx1/scripts/find_valid_v10_bundle.py --gx1-data-root ../GX1_DATA
    python gx1/scripts/find_valid_v10_bundle.py --expect-sessions "EU,US,OVERLAP" --print-top 10
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure gx1 is importable
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


@dataclass
class BundleScanResult:
    """Result of scanning a single bundle directory."""
    path: str
    bundle_class: str = "UNKNOWN"  # OK, NO_XGB, LEGACY, PARTIAL
    reasons: List[str] = field(default_factory=list)
    has_model_state_dict: bool = False
    has_bundle_metadata: bool = False
    has_feature_contract_hash: bool = False
    xgb_sessions_found: List[str] = field(default_factory=list)
    xgb_sessions_missing: List[str] = field(default_factory=list)
    has_xgb_models_by_session_in_meta: bool = False
    modified_time: Optional[str] = None
    modified_timestamp: float = 0.0
    contract_hash: Optional[str] = None
    is_legacy: bool = False
    legacy_markers: List[str] = field(default_factory=list)


LEGACY_MARKERS = [
    "v9", "entry_v9", "V9", "OANDA_DEMO_V9", "sniper/NY", "gated_old", "legacy"
]


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_legacy_markers(path: Path, metadata: Optional[dict]) -> Tuple[bool, List[str]]:
    """Check if bundle path or metadata contains legacy markers."""
    markers_found = []
    path_str = str(path).lower()
    
    for marker in LEGACY_MARKERS:
        if marker.lower() in path_str:
            markers_found.append(f"path contains '{marker}'")
    
    if metadata:
        meta_str = json.dumps(metadata).lower()
        for marker in LEGACY_MARKERS:
            if marker.lower() in meta_str:
                markers_found.append(f"metadata contains '{marker}'")
                break  # Only report once for metadata
    
    return len(markers_found) > 0, markers_found


def scan_bundle(bundle_dir: Path, expected_sessions: List[str]) -> BundleScanResult:
    """Scan a single bundle directory and classify it."""
    result = BundleScanResult(path=str(bundle_dir))
    reasons = []
    
    # Get modification time
    try:
        mtime = bundle_dir.stat().st_mtime
        result.modified_timestamp = mtime
        result.modified_time = datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        result.modified_time = None
        result.modified_timestamp = 0.0
    
    # Check model_state_dict.pt
    model_state_path = bundle_dir / "model_state_dict.pt"
    result.has_model_state_dict = model_state_path.exists()
    if not result.has_model_state_dict:
        reasons.append("missing model_state_dict.pt")
    
    # Check bundle_metadata.json
    metadata_path = bundle_dir / "bundle_metadata.json"
    metadata = None
    result.has_bundle_metadata = metadata_path.exists()
    if result.has_bundle_metadata:
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            # Check for xgb_models_by_session in metadata
            if "xgb_models_by_session" in metadata:
                result.has_xgb_models_by_session_in_meta = True
        except Exception as e:
            reasons.append(f"failed to parse bundle_metadata.json: {e}")
    else:
        reasons.append("missing bundle_metadata.json")
    
    # Check feature_contract_hash.txt or metadata field
    contract_hash_path = bundle_dir / "feature_contract_hash.txt"
    if contract_hash_path.exists():
        result.has_feature_contract_hash = True
        try:
            result.contract_hash = contract_hash_path.read_text().strip()
        except Exception:
            pass
    elif metadata and "feature_contract_hash" in metadata:
        result.has_feature_contract_hash = True
        result.contract_hash = metadata.get("feature_contract_hash")
    else:
        reasons.append("missing feature_contract_hash")
    
    # Check XGB models for each expected session
    # Support multiple naming conventions:
    # - xgb_{session}.pkl (preferred)
    # - xgb_{session}.joblib
    # - xgb_entry_{session}_v10.joblib (v10 format)
    for session in expected_sessions:
        xgb_patterns = [
            bundle_dir / f"xgb_{session}.pkl",
            bundle_dir / f"xgb_{session}.joblib",
            bundle_dir / f"xgb_entry_{session}_v10.joblib",
            bundle_dir / f"xgb_entry_{session}_v10.pkl",
        ]
        found = False
        for xgb_path in xgb_patterns:
            if xgb_path.exists():
                result.xgb_sessions_found.append(session)
                found = True
                break
        if not found:
            result.xgb_sessions_missing.append(session)
            reasons.append(f"missing xgb_{session}.pkl/joblib")
    
    # Check for legacy markers
    is_legacy, legacy_markers = check_legacy_markers(bundle_dir, metadata)
    result.is_legacy = is_legacy
    result.legacy_markers = legacy_markers
    if is_legacy:
        reasons.extend(legacy_markers)
    
    # Classify bundle
    if is_legacy:
        result.bundle_class = "LEGACY"
    elif not result.has_model_state_dict or not result.has_bundle_metadata:
        result.bundle_class = "PARTIAL"
    elif len(result.xgb_sessions_missing) > 0:
        result.bundle_class = "NO_XGB"
    else:
        result.bundle_class = "OK"
    
    result.reasons = reasons
    return result


def find_v10_bundles(gx1_data_root: Path, expected_sessions: List[str]) -> List[BundleScanResult]:
    """Find and scan all V10 bundles under GX1_DATA/models/models/."""
    bundles = []
    
    # Scan under models/models/**/ for v10_ctx bundles
    models_root = gx1_data_root / "models" / "models"
    if not models_root.exists():
        print(f"[WARN] Models root does not exist: {models_root}")
        return bundles
    
    # Find all directories that might be bundles
    # Look for directories containing model_state_dict.pt or bundle_metadata.json
    for root, dirs, files in os.walk(models_root):
        root_path = Path(root)
        
        # Skip if this is clearly not a bundle directory
        if "model_state_dict.pt" in files or "bundle_metadata.json" in files:
            # This is a potential bundle
            # Check if it's a v10_ctx bundle (path contains v10_ctx or entry_v10_ctx)
            path_str = str(root_path).lower()
            if "v10_ctx" in path_str or "entry_v10" in path_str:
                result = scan_bundle(root_path, expected_sessions)
                bundles.append(result)
    
    # Sort: OK first, then by modified time (newest first)
    def sort_key(b: BundleScanResult) -> Tuple[int, float]:
        class_order = {"OK": 0, "NO_XGB": 1, "PARTIAL": 2, "LEGACY": 3}
        return (class_order.get(b.bundle_class, 99), -b.modified_timestamp)
    
    bundles.sort(key=sort_key)
    return bundles


def write_reports(bundles: List[BundleScanResult], output_dir: Path) -> Tuple[Path, Path]:
    """Write JSON and MD reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_path = output_dir / "V10_BUNDLE_SCAN.json"
    json_data = {
        "scan_time": datetime.now().isoformat(),
        "total_bundles": len(bundles),
        "ok_count": sum(1 for b in bundles if b.bundle_class == "OK"),
        "no_xgb_count": sum(1 for b in bundles if b.bundle_class == "NO_XGB"),
        "partial_count": sum(1 for b in bundles if b.bundle_class == "PARTIAL"),
        "legacy_count": sum(1 for b in bundles if b.bundle_class == "LEGACY"),
        "bundles": [asdict(b) for b in bundles],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    # MD report
    md_path = output_dir / "V10_BUNDLE_SCAN.md"
    lines = [
        "# V10 Bundle Scan Report",
        "",
        f"**Scan Time:** {datetime.now().isoformat()}",
        f"**Total Bundles:** {len(bundles)}",
        "",
        "## Summary",
        "",
        f"- **OK:** {json_data['ok_count']}",
        f"- **NO_XGB:** {json_data['no_xgb_count']}",
        f"- **PARTIAL:** {json_data['partial_count']}",
        f"- **LEGACY:** {json_data['legacy_count']}",
        "",
        "## Bundle Details",
        "",
    ]
    
    for b in bundles:
        status_emoji = {"OK": "✅", "NO_XGB": "⚠️", "PARTIAL": "❌", "LEGACY": "🚫"}.get(b.bundle_class, "❓")
        lines.append(f"### {status_emoji} {b.bundle_class}: `{b.path}`")
        lines.append("")
        lines.append(f"- **Modified:** {b.modified_time or 'unknown'}")
        lines.append(f"- **Contract Hash:** {b.contract_hash or 'N/A'}")
        lines.append(f"- **XGB Sessions Found:** {', '.join(b.xgb_sessions_found) or 'none'}")
        lines.append(f"- **XGB Sessions Missing:** {', '.join(b.xgb_sessions_missing) or 'none'}")
        if b.reasons:
            lines.append(f"- **Reasons:** {'; '.join(b.reasons)}")
        lines.append("")
    
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    
    return json_path, md_path


def get_best_ok_bundle(bundles: List[BundleScanResult]) -> Optional[BundleScanResult]:
    """Get the best OK bundle (first in sorted list, which is newest by mtime)."""
    for b in bundles:
        if b.bundle_class == "OK":
            return b
    return None


def main():
    parser = argparse.ArgumentParser(description="Find valid V10 hybrid bundles with XGB models")
    parser.add_argument(
        "--gx1-data-root",
        type=str,
        default="../GX1_DATA",
        help="Root directory for GX1_DATA (default: ../GX1_DATA)"
    )
    parser.add_argument(
        "--expect-sessions",
        type=str,
        default="EU,US,OVERLAP",
        help="Comma-separated list of expected XGB sessions (default: EU,US,OVERLAP)"
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=10,
        help="Number of top bundles to print (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: <workspace>/reports/repo_audit)"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output only JSON to stdout (for programmatic use)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    gx1_data_root = Path(args.gx1_data_root).resolve()
    if not gx1_data_root.exists():
        print(f"[ERROR] GX1_DATA root does not exist: {gx1_data_root}", file=sys.stderr)
        sys.exit(1)
    
    expected_sessions = [s.strip() for s in args.expect_sessions.split(",")]
    
    output_dir = Path(args.output_dir) if args.output_dir else WORKSPACE_ROOT / "reports" / "repo_audit"
    
    # Scan bundles
    print(f"[SCAN] Scanning for V10 bundles under: {gx1_data_root / 'models' / 'models'}")
    print(f"[SCAN] Expected XGB sessions: {expected_sessions}")
    
    bundles = find_v10_bundles(gx1_data_root, expected_sessions)
    
    if args.json_output:
        # Output only JSON for programmatic use
        result = {
            "ok_bundles": [asdict(b) for b in bundles if b.bundle_class == "OK"],
            "best_ok_bundle": None,
        }
        best = get_best_ok_bundle(bundles)
        if best:
            result["best_ok_bundle"] = best.path
        print(json.dumps(result, indent=2))
        sys.exit(0)
    
    # Print summary
    print("")
    print("=" * 80)
    print("V10 BUNDLE SCAN RESULTS")
    print("=" * 80)
    print(f"Total bundles found: {len(bundles)}")
    print(f"  OK: {sum(1 for b in bundles if b.bundle_class == 'OK')}")
    print(f"  NO_XGB: {sum(1 for b in bundles if b.bundle_class == 'NO_XGB')}")
    print(f"  PARTIAL: {sum(1 for b in bundles if b.bundle_class == 'PARTIAL')}")
    print(f"  LEGACY: {sum(1 for b in bundles if b.bundle_class == 'LEGACY')}")
    print("")
    
    # Print top bundles
    print(f"Top {args.print_top} bundles:")
    print("-" * 80)
    for i, b in enumerate(bundles[:args.print_top]):
        status_emoji = {"OK": "✅", "NO_XGB": "⚠️", "PARTIAL": "❌", "LEGACY": "🚫"}.get(b.bundle_class, "❓")
        print(f"{i+1}. {status_emoji} [{b.bundle_class}] {b.path}")
        if b.reasons:
            print(f"      Reasons: {'; '.join(b.reasons[:3])}")
        print(f"      XGB: found={b.xgb_sessions_found}, missing={b.xgb_sessions_missing}")
        print(f"      Modified: {b.modified_time}")
    print("")
    
    # Get best OK bundle
    best = get_best_ok_bundle(bundles)
    if best:
        print(f"✅ BEST OK BUNDLE: {best.path}")
        print(f"   Selection reason: OK class, newest modified ({best.modified_time})")
    else:
        print("❌ NO OK BUNDLES FOUND!")
        print("   Consider using rehydrate_xgb_into_bundle.py to add XGB models to a NO_XGB bundle")
    print("")
    
    # Write reports
    json_path, md_path = write_reports(bundles, output_dir)
    print(f"[REPORT] JSON: {json_path}")
    print(f"[REPORT] MD: {md_path}")
    
    # Exit with appropriate code
    if best:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
