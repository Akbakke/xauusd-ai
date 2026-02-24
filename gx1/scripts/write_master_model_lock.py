#!/usr/bin/env python3
"""
Write MASTER_MODEL_LOCK.json - the single source of truth for model authorization.

TRUTH XGB LANE (CANONICAL)
- This script is intended for BASE28_CANONICAL + xgb_universal_multihead_v2__CANONICAL.
- Read: gx1/scripts/README_TRUTH_XGB.md
- Anything else is legacy / wrong lane.

Usage:
    python3 gx1/scripts/write_master_model_lock.py \
      --bundle-dir <path> \
      --xgb-mode universal_multihead_v2 \
      --require-go-marker 1
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


# --- TRUTH XGB guard (inline, no external imports) ----------------------------
def truth_readme_path() -> Path:
    return WORKSPACE_ROOT / "gx1" / "scripts" / "README_TRUTH_XGB.md"


def require_truth_xgb_lane(script_path: str) -> Path:
    """
    Minimal, robust guard:
    - Ensures README exists (so we always have a single source of truth).
    - Ensures GX1_DATA_ROOT is set and points to a real directory (canonical behavior).
    Returns resolved GX1_DATA path.
    """
    readme = truth_readme_path()
    if not readme.exists():
        raise RuntimeError(
            "TRUTH_XGB_README_MISSING: expected gx1/scripts/README_TRUTH_XGB.md\n"
            f"Script: {script_path}\n"
            f"Expected: {readme}"
        )

    gx1_data_root = os.environ.get("GX1_DATA_ROOT")
    if not gx1_data_root:
        raise RuntimeError(
            "GX1_DATA_ROOT_NOT_SET: canonical XGB lane requires GX1_DATA_ROOT.\n"
            f"Read: {readme}"
        )
    gx1_data = Path(gx1_data_root)
    if not gx1_data.exists():
        raise RuntimeError(
            f"GX1_DATA_ROOT_NOT_FOUND: {gx1_data}\n"
            f"Read: {readme}"
        )

    return gx1_data


GX1_DATA_REQUIRED = require_truth_xgb_lane(__file__)
# -----------------------------------------------------------------------------


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
    """Resolve GX1_DATA directory (canonical: env var required by guard)."""
    return GX1_DATA_REQUIRED


def _pick_marker_paths(
    bundle_dir: Path,
    promotion_lane: str,
) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Decide which GO/NO-GO marker filenames to use for universal_multihead_v2.

    promotion_lane:
      - "NORM": requires normalized lane markers
      - "RAW": requires raw lane markers
      - "ANY": prefer NORM if present, else RAW if present

    Returns: (go_path, no_go_path, lane_used)
    """
    norm_go = bundle_dir / "XGB_MULTIHEAD_V2_NORM_GO_MARKER.json"
    norm_no = bundle_dir / "XGB_MULTIHEAD_V2_NORM_NO_GO_MARKER.json"

    raw_go = bundle_dir / "XGB_MULTIHEAD_V2_RAW_GO_MARKER.json"
    raw_no = bundle_dir / "XGB_MULTIHEAD_V2_RAW_NO_GO_MARKER.json"

    # Backward-compat legacy names (treat as RAW lane)
    legacy_go_1 = bundle_dir / "XGB_MULTIHEAD_V2_GO_MARKER.json"
    legacy_go_2 = bundle_dir / "XGB_MULTIHEAD_V2_GO.json"
    legacy_no_1 = bundle_dir / "XGB_MULTIHEAD_V2_NO_GO.json"
    legacy_no_2 = bundle_dir / "XGB_MULTIHEAD_V2_NO_GO_MARKER.json"

    def exists(p: Path) -> bool:
        return p.exists()

    lane = promotion_lane.upper()

    if lane == "NORM":
        return (norm_go if exists(norm_go) else None, norm_no if exists(norm_no) else None, "NORM")

    if lane == "RAW":
        go = raw_go if exists(raw_go) else (legacy_go_2 if exists(legacy_go_2) else (legacy_go_1 if exists(legacy_go_1) else None))
        no = raw_no if exists(raw_no) else (legacy_no_1 if exists(legacy_no_1) else (legacy_no_2 if exists(legacy_no_2) else None))
        return (go, no, "RAW")

    # ANY: prefer NORM first
    if exists(norm_go) or exists(norm_no):
        return (norm_go if exists(norm_go) else None, norm_no if exists(norm_no) else None, "NORM")

    # then RAW
    if exists(raw_go) or exists(raw_no):
        return (raw_go if exists(raw_go) else None, raw_no if exists(raw_no) else None, "RAW")

    # then legacy (RAW)
    if exists(legacy_go_2) or exists(legacy_go_1) or exists(legacy_no_1) or exists(legacy_no_2):
        go = legacy_go_2 if exists(legacy_go_2) else (legacy_go_1 if exists(legacy_go_1) else None)
        no = legacy_no_1 if exists(legacy_no_1) else (legacy_no_2 if exists(legacy_no_2) else None)
        return (go, no, "RAW")

    return (None, None, "UNKNOWN")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write MASTER_MODEL_LOCK.json (CANONICAL)")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Bundle directory (default: GX1_DATA/models/models/xgb_universal_multihead_v2__CANONICAL/)",
    )
    parser.add_argument(
        "--xgb-mode",
        type=str,
        default="universal_multihead_v2",
        choices=["universal_multihead_v2", "universal", "session"],
        help="XGB mode",
    )
    parser.add_argument(
        "--promotion-lane",
        type=str,
        default="ANY",
        choices=["ANY", "NORM", "RAW"],
        help="Which promotion lane marker to honor (default: ANY prefers NORM then RAW).",
    )
    parser.add_argument(
        "--require-go-marker",
        type=int,
        default=1,
        help="Require GO marker (default: 1)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WRITE MASTER MODEL LOCK (CANONICAL)")
    print("=" * 60)

    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA:   {gx1_data}")
    print(f"TRUTH:      {truth_readme_path()}")

    # Resolve bundle dir
    if args.bundle_dir:
        bundle_dir = args.bundle_dir
    else:
        # Canonical default for this lane
        bundle_dir = gx1_data / "models" / "models" / "xgb_universal_multihead_v2__CANONICAL"

    print(f"Bundle dir: {bundle_dir}")
    print(f"XGB mode:   {args.xgb_mode}")
    if args.xgb_mode == "universal_multihead_v2":
        print(f"Lane:       {args.promotion_lane}")

    if not bundle_dir.exists():
        print(f"ERROR: Bundle dir not found: {bundle_dir}")
        return 1

    # Determine model files based on mode
    if args.xgb_mode == "universal_multihead_v2":
        model_filename = "xgb_universal_multihead_v2.joblib"
        meta_filename = "xgb_universal_multihead_v2_meta.json"
    elif args.xgb_mode == "universal":
        model_filename = "xgb_universal_v1.joblib"
        meta_filename = "xgb_universal_v1_meta.json"
    else:
        print("ERROR: Session mode not supported for master lock")
        return 1

    model_path = bundle_dir / model_filename
    meta_path = bundle_dir / meta_filename

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1

    print("\nComputing SHAs...")
    model_sha = compute_file_sha256(model_path)
    meta_sha = compute_file_sha256(meta_path)

    print(f"  Model:      {model_sha[:16] if model_sha else 'NOT FOUND'}...")
    print(f"  Meta:       {meta_sha[:16] if meta_sha else 'NOT FOUND'}...")

    # Contracts (BASE28 canonical)
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_base28_v1.json"
    sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base28_v1.json"
    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"

    contracts: Dict[str, Dict[str, Any]] = {
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

    # Compute schema_hash deterministically (no nulls)
    if not contracts["feature_contract"]["sha256"]:
        raise RuntimeError(f"FEATURE_CONTRACT_MISSING: {feature_contract_path}")
    if not contracts["sanitizer_config"]["sha256"]:
        raise RuntimeError(f"SANITIZER_CONTRACT_MISSING: {sanitizer_config_path}")
    if not contracts["output_contract"]["sha256"]:
        raise RuntimeError(f"OUTPUT_CONTRACT_MISSING: {output_contract_path}")

    with open(feature_contract_path, "r", encoding="utf-8") as f:
        feature_contract_obj = json.load(f)
    features = feature_contract_obj.get("features", [])
    if not isinstance(features, list) or not features:
        raise RuntimeError("FEATURE_CONTRACT_INVALID: features missing or empty")

    schema_payload = {
        "features": features,
        "sanitizer_sha256": contracts["sanitizer_config"]["sha256"],
        "output_contract_sha256": contracts["output_contract"]["sha256"],
    }
    schema_hash = hashlib.sha256(json.dumps(schema_payload, sort_keys=True).encode("utf-8")).hexdigest()

    # Injection contract extraction (best-effort)
    injection_channels = []
    injection_mode = "current_session"
    if output_contract_path.exists():
        with open(output_contract_path, "r", encoding="utf-8") as f:
            output_contract = json.load(f)
        injection_mode = output_contract.get("injection_mode", "current_session")
        template = output_contract.get("injection_channels_template", [])
        sessions = output_contract.get("sessions", ["EU", "US", "OVERLAP"])

        if injection_mode == "current_session":
            for session in sessions:
                for tmpl in template:
                    injection_channels.append(tmpl.replace("{session}", session))
        else:
            for tmpl in template:
                for session in sessions:
                    injection_channels.append(tmpl.replace("{session}", session))

    print(f"\nInjection contract:")
    print(f"  Mode: {injection_mode}")
    print(f"  Channels ({len(injection_channels)}): {injection_channels[:4]}...")

    # Promotion markers
    go_marker_path: Optional[Path] = None
    no_go_marker_path: Optional[Path] = None
    lane_used = "UNKNOWN"

    if args.xgb_mode == "universal_multihead_v2":
        go_marker_path, no_go_marker_path, lane_used = _pick_marker_paths(bundle_dir, args.promotion_lane)
    else:
        go_marker_path = bundle_dir / "XGB_UNIVERSAL_GO_MARKER.json"
        no_go_marker_path = bundle_dir / "XGB_UNIVERSAL_NO_GO.json"
        lane_used = "UNIVERSAL"

    go_marker_sha = compute_file_sha256(go_marker_path) if go_marker_path else None
    no_go_marker_sha = compute_file_sha256(no_go_marker_path) if no_go_marker_path else None

    print("\nPromotion markers:")
    print(f"  Lane used:  {lane_used}")
    print(f"  GO marker:  {str(go_marker_path) if go_marker_path else 'NOT FOUND'}")
    print(f"  NO-GO:      {str(no_go_marker_path) if no_go_marker_path else 'NOT FOUND'}")
    if go_marker_sha:
        print(f"  GO sha:     {go_marker_sha[:16]}...")
    if no_go_marker_sha:
        print(f"  NO-GO sha:  {no_go_marker_sha[:16]}...")

    if bool(args.require_go_marker) and not go_marker_sha:
        print("\nERROR: require-go-marker=1 but GO marker is missing.")
        print("  Expected (lane-dependent):")
        print("    NORM: XGB_MULTIHEAD_V2_NORM_GO_MARKER.json")
        print("    RAW:  XGB_MULTIHEAD_V2_RAW_GO_MARKER.json (or legacy XGB_MULTIHEAD_V2_GO*.json)")
        if no_go_marker_path and no_go_marker_path.exists():
            print(f"  Found NO-GO marker: {no_go_marker_path}")
        return 1

    lock: Dict[str, Any] = {
        "version": "v1",
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "description": "Master model lock - single source of truth for authorized model (CANONICAL)",
        "truth_readme": str(truth_readme_path()),
        "xgb_mode": args.xgb_mode,
        "promotion_lane": lane_used,
        "model_path_relative": model_filename,
        "model_sha256": model_sha,
        "meta_path_relative": meta_filename,
        "meta_sha256": meta_sha,
        "schema_hash": schema_hash,
        "require_go_marker": bool(args.require_go_marker),
        "promotion_markers": {
            "go_marker_filename": go_marker_path.name if go_marker_path else None,
            "go_marker_sha256": go_marker_sha,
            "no_go_marker_filename": no_go_marker_path.name if no_go_marker_path else None,
            "no_go_marker_sha256": no_go_marker_sha,
        },
        "contracts": contracts,
        "injection_contract": {
            "mode": injection_mode,
            "channels": injection_channels,
            "expected_channels_count": len(injection_channels),
            "auxiliary_channels": ["uncertainty_xgb_*"],
        },
        "invariants": {
            "no_fallback": True,
            "no_legacy_paths": True,
            "no_auto_resolve": True,
            "require_feature_names": True,
        },
    }

    if go_marker_sha:
        lock["promotion_status"] = "GO"
    elif no_go_marker_sha:
        lock["promotion_status"] = "NO-GO"
    else:
        lock["promotion_status"] = "NOT_EVALUATED"

    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    tmp_path = lock_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
    tmp_path.rename(lock_path)

    print(f"\n✅ MASTER_MODEL_LOCK.json written: {lock_path}")

    print("\n" + "=" * 60)
    print("LOCK SUMMARY")
    print("=" * 60)
    print(f"XGB Mode:    {args.xgb_mode}")
    print(f"Lane:        {lane_used}")
    print(f"Model:       {model_filename}")
    print(f"Model SHA:   {model_sha}")
    print(f"Schema hash: {schema_hash}")
    print(f"Promotion:   {lock.get('promotion_status', 'UNKNOWN')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())