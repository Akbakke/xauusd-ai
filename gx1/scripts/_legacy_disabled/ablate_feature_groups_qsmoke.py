#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Group Ablation Smoke Runner

Runs Q1/Q2 smoke evaluations with baseline + masked feature groups.
Masking is deterministic, logged in RUN_IDENTITY, and does NOT affect universe-fingerprint.

Usage:
    python3 gx1/scripts/ablate_feature_groups_qsmoke.py \
        --bundle-dir ../GX1_DATA/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
        --data-root ../GX1_DATA/data/oanda/years \
        --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --policy policies/sniper_trial160_prod.json \
        --manifest-json gx1/feature_manifest_v1.json \
        --out-root ../GX1_DATA/reports/feature_ablation \
        --quarters Q1,Q2
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_git_sha() -> Optional[str]:
    """Get git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return None


def load_feature_manifest(manifest_json_path: Path) -> Dict[str, Any]:
    """Load feature manifest."""
    with open(manifest_json_path, "r") as f:
        return json.load(f)


def get_features_by_family(manifest: Dict[str, Any], family: str) -> Set[str]:
    """Get all features in a family."""
    features = set()
    for feat in manifest.get("features", []):
        if feat.get("family") == family:
            features.add(feat["name"])
    return features


def create_feature_mask_spec(
    manifest: Dict[str, Any],
    mask_families: List[str],
) -> Dict[str, Any]:
    """
    Create feature mask specification.
    
    Args:
        manifest: Feature manifest
        mask_families: List of families to mask (e.g., ["basic_v1", "sequence"])
    
    Returns:
        Mask spec dict with feature names and masking strategy
    """
    mask_spec = {
        "masked_families": mask_families,
        "masked_features": set(),
        "mask_strategy": "zero_numeric_keep_flags",  # Default strategy
    }
    
    for family in mask_families:
        features = get_features_by_family(manifest, family)
        mask_spec["masked_features"].update(features)
    
    mask_spec["masked_features"] = sorted(list(mask_spec["masked_features"]))
    
    return mask_spec


def run_ablation_smoke(
    quarter: str,
    mask_families: List[str],
    bundle_dir: Path,
    data_root: Path,
    prebuilt_parquet: Path,
    policy_path: Path,
    manifest_json_path: Path,
    output_dir: Path,
    workspace_root: Path,
) -> Dict[str, Any]:
    """
    Run ablation smoke for a quarter with masked feature groups.
    
    Args:
        quarter: Quarter identifier (Q1, Q2, Q3, Q4)
        mask_families: List of families to mask
        bundle_dir: Path to bundle directory
        data_root: Path to data root
        prebuilt_parquet: Path to prebuilt features parquet
        policy_path: Path to policy YAML
        manifest_json_path: Path to feature manifest JSON
        output_dir: Output directory
        workspace_root: Workspace root
    
    Returns:
        Run metadata dict
    """
    log.info("=" * 60)
    log.info(f"ABLATION SMOKE: {quarter} with mask_families={mask_families}")
    log.info("=" * 60)
    
    # Load manifest
    manifest = load_feature_manifest(manifest_json_path)
    
    # Create mask spec
    mask_spec = create_feature_mask_spec(manifest, mask_families)
    
    log.info(f"Masking {len(mask_spec['masked_features'])} features from families: {mask_families}")
    
    # Determine date range for quarter
    quarter_ranges = {
        "Q1": ("2025-01-01", "2025-03-31"),
        "Q2": ("2025-04-01", "2025-06-30"),
        "Q3": ("2025-07-01", "2025-09-30"),
        "Q4": ("2025-10-01", "2025-12-31"),
    }
    
    if quarter not in quarter_ranges:
        raise ValueError(f"Invalid quarter: {quarter}. Must be one of: {list(quarter_ranges.keys())}")
    
    start_date, end_date = quarter_ranges[quarter]
    smoke_date_range = f"{start_date}..{end_date}"
    
    # Create output directory for this ablation run
    mask_name = "_".join(mask_families) if mask_families else "baseline"
    run_id = f"ABLATION_{quarter}_{mask_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    quarter_output_dir = output_dir / run_id / quarter
    
    log.info(f"Output directory: {quarter_output_dir}")
    
    # Set environment variables for masking
    env = os.environ.copy()
    env["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env["GX1_REPLAY_PREBUILT_FEATURES_PATH"] = str(prebuilt_parquet.resolve())
    env["GX1_FEATURE_BUILD_DISABLED"] = "1"
    env["GX1_REPLAY_MODE"] = "PREBUILT"
    env["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    
    # NEW: Feature masking environment variables
    env["GX1_FEATURE_MASK_ENABLED"] = "1" if mask_families else "0"
    env["GX1_FEATURE_MASK_FAMILIES"] = ",".join(mask_families) if mask_families else ""
    env["GX1_FEATURE_MASK_STRATEGY"] = "zero_numeric_keep_flags"
    env["GX1_FEATURE_MANIFEST_JSON"] = str(manifest_json_path.resolve())
    
    # Write mask spec to output directory
    quarter_output_dir.mkdir(parents=True, exist_ok=True)
    mask_spec_path = quarter_output_dir / "FEATURE_MASK_SPEC.json"
    with open(mask_spec_path, "w") as f:
        json.dump(mask_spec, f, indent=2)
    
    # Import and call run_depth_ladder_eval_multiyear's run_smoke_eval
    # We'll use the same infrastructure but with masking enabled
    from gx1.scripts.run_depth_ladder_eval_multiyear import run_smoke_eval
    
    # Load bundle metadata
    bundle_metadata_path = bundle_dir / "bundle_metadata.json"
    bundle_metadata = {}
    if bundle_metadata_path.exists():
        with open(bundle_metadata_path, "r") as f:
            bundle_metadata = json.load(f)
    
    log.info(f"Running smoke eval for {quarter} with mask_families={mask_families}...")
    
    try:
        result = run_smoke_eval(
            arm="baseline",  # Use baseline arm (masking is separate)
            bundle_dir=bundle_dir,
            data_path=data_root / "2025.parquet",  # Assume 2025 data
            prebuilt_path=prebuilt_parquet,
            policy_path=policy_path,
            output_dir=quarter_output_dir,
            workspace_root=workspace_root,
            bundle_metadata=bundle_metadata,
            baseline_reference_path=None,  # No baseline reference for ablation
            smoke_date_range=smoke_date_range,
            smoke_bars=None,
            safety_timeout_seconds=600,
        )
        
        # Add mask spec to result
        result["feature_mask_spec"] = mask_spec
        
        # Write RUN_IDENTITY with mask spec
        run_identity_path = quarter_output_dir / "RUN_IDENTITY.json"
        if run_identity_path.exists():
            with open(run_identity_path, "r") as f:
                run_identity = json.load(f)
        else:
            run_identity = {}
        
        run_identity["feature_mask_spec"] = mask_spec
        run_identity["feature_mask_enabled"] = True
        run_identity["masked_families"] = mask_families
        
        with open(run_identity_path, "w") as f:
            json.dump(run_identity, f, indent=2)
        
        log.info(f"✅ Ablation smoke complete: {quarter_output_dir}")
        return result
    
    except Exception as e:
        log.error(f"❌ Ablation smoke failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Feature Group Ablation Smoke Runner")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to bundle directory",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to data root directory",
    )
    parser.add_argument(
        "--prebuilt-parquet",
        type=Path,
        required=True,
        help="Path to prebuilt features parquet",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to policy YAML",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("gx1/feature_manifest_v1.json"),
        help="Path to feature manifest JSON",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: GX1_REPORTS_ROOT/feature_ablation)",
    )
    parser.add_argument(
        "--quarters",
        type=str,
        default="Q1,Q2",
        help="Comma-separated quarters to run (default: Q1,Q2)",
    )
    
    args = parser.parse_args()
    
    # Resolve output directory
    if args.out_root is None:
        default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports"))
        args.out_root = default_reports_root / "feature_ablation"
    
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest = load_feature_manifest(args.manifest_json)
    
    # Define feature groups to mask
    mask_groups = [
        [],  # Baseline (no mask)
        ["basic_v1"],
        ["sequence"],
        ["htf"],
        ["microstructure"],
        ["session"],
    ]
    
    quarters = [q.strip() for q in args.quarters.split(",")]
    
    log.info("=" * 60)
    log.info("FEATURE GROUP ABLATION SMOKE RUNNER")
    log.info("=" * 60)
    log.info(f"Quarters: {quarters}")
    log.info(f"Mask groups: {[g if g else 'baseline' for g in mask_groups]}")
    
    # Run baseline first
    log.info("\n" + "=" * 60)
    log.info("RUNNING BASELINE (no mask)")
    log.info("=" * 60)
    
    baseline_results = {}
    for quarter in quarters:
        log.info(f"\n--- Baseline {quarter} ---")
        try:
            result = run_ablation_smoke(
                quarter=quarter,
                mask_families=[],
                bundle_dir=args.bundle_dir,
                data_root=args.data_root,
                prebuilt_parquet=args.prebuilt_parquet,
                policy_path=args.policy,
                manifest_json_path=args.manifest_json,
                output_dir=args.out_root,
                workspace_root=workspace_root,
            )
            baseline_results[quarter] = result
        except Exception as e:
            log.error(f"❌ Baseline {quarter} failed: {e}", exc_info=True)
            baseline_results[quarter] = {"error": str(e)}
    
    # Run masked groups
    ablation_results = {}
    for mask_families in mask_groups[1:]:  # Skip baseline (already run)
        mask_name = "_".join(mask_families)
        log.info("\n" + "=" * 60)
        log.info(f"RUNNING MASKED GROUP: {mask_name}")
        log.info("=" * 60)
        
        ablation_results[mask_name] = {}
        for quarter in quarters:
            log.info(f"\n--- {mask_name} {quarter} ---")
            try:
                result = run_ablation_smoke(
                    quarter=quarter,
                    mask_families=mask_families,
                    bundle_dir=args.bundle_dir,
                    data_root=args.data_root,
                    prebuilt_parquet=args.prebuilt_parquet,
                    policy_path=args.policy,
                    manifest_json_path=args.manifest_json,
                    output_dir=args.out_root,
                    workspace_root=workspace_root,
                )
                ablation_results[mask_name][quarter] = result
            except Exception as e:
                log.error(f"❌ {mask_name} {quarter} failed: {e}", exc_info=True)
                ablation_results[mask_name][quarter] = {"error": str(e)}
    
    # Generate comparison report
    log.info("\n" + "=" * 60)
    log.info("GENERATING ABLATION COMPARISON REPORT")
    log.info("=" * 60)
    
    # Write summary
    summary_path = args.out_root / "ABLATION_SUMMARY.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "sys_executable": sys.executable,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git_sha": get_git_sha(),
        "baseline_results": baseline_results,
        "ablation_results": ablation_results,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"✅ Ablation complete: {args.out_root}")
    log.info(f"   Summary: {summary_path}")


if __name__ == "__main__":
    main()
