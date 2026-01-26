#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRY_V10_CTX Bundle Loader Selftest

Minimal test script to reproduce and diagnose bundle loading failures.

Usage:
    python gx1/scripts/selftest_entry_v10_bundle_load.py \
        --bundle-dir <PATH> \
        [--policy <PATH>]
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def list_bundle_directory(bundle_dir: Path) -> Dict[str, Any]:
    """List all files in bundle directory and subdirectories."""
    result = {
        "bundle_dir": str(bundle_dir.resolve()),
        "exists": bundle_dir.exists(),
        "is_dir": bundle_dir.is_dir() if bundle_dir.exists() else False,
        "top_level_files": [],
        "subdirectories": {},
    }
    
    if not bundle_dir.exists():
        return result
    
    try:
        for item in sorted(bundle_dir.iterdir()):
            if item.is_file():
                result["top_level_files"].append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "path": str(item.resolve()),
                })
            elif item.is_dir():
                subdir_files = []
                try:
                    for subitem in sorted(item.iterdir()):
                        if subitem.is_file():
                            subdir_files.append({
                                "name": subitem.name,
                                "size": subitem.stat().st_size,
                                "path": str(subitem.resolve()),
                            })
                except Exception as e:
                    subdir_files = [{"error": str(e)}]
                result["subdirectories"][item.name] = subdir_files
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_expected_bundle_files(bundle_dir: Path) -> Dict[str, Any]:
    """Check for expected bundle files and return status."""
    expected_files = {
        "bundle_metadata.json": bundle_dir / "bundle_metadata.json",
        "model_state_dict.pt": bundle_dir / "model_state_dict.pt",
        "feature_contract_hash.txt": bundle_dir / "feature_contract_hash.txt",
    }
    
    result = {
        "bundle_dir": str(bundle_dir.resolve()),
        "expected_files": {},
        "all_present": True,
    }
    
    for name, path in expected_files.items():
        exists = path.exists()
        result["expected_files"][name] = {
            "path": str(path.resolve()),
            "exists": exists,
            "size": path.stat().st_size if exists else None,
        }
        if not exists:
            result["all_present"] = False
    
    return result


def load_bundle_metadata(bundle_dir: Path) -> Dict[str, Any]:
    """Load and validate bundle metadata."""
    metadata_path = bundle_dir / "bundle_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"bundle_metadata.json not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Check required fields
    required_fields = [
        "model_variant",
        "supports_context_features",
    ]
    
    missing_fields = [f for f in required_fields if f not in metadata]
    if missing_fields:
        raise ValueError(f"Missing required fields in bundle_metadata.json: {missing_fields}")
    
    # Check depth ladder fields (if present)
    depth_ladder_fields = {
        "transformer_layers": metadata.get("transformer_layers"),
        "transformer_layers_baseline": metadata.get("transformer_layers_baseline"),
        "depth_ladder_delta": metadata.get("depth_ladder_delta"),
    }
    
    return {
        "metadata": metadata,
        "depth_ladder_fields": depth_ladder_fields,
    }


def test_bundle_load(bundle_dir: Path, policy_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Test bundle loading and return result.
    
    Returns:
        Dict with either BUNDLE_LOAD_OK or BUNDLE_LOAD_FAIL structure
    """
    bundle_dir = bundle_dir.resolve()
    
    result = {
        "bundle_dir": str(bundle_dir),
        "test_timestamp": None,
        "status": None,
    }
    
    try:
        # Check expected files
        file_check = check_expected_bundle_files(bundle_dir)
        if not file_check["all_present"]:
            missing = [name for name, info in file_check["expected_files"].items() if not info["exists"]]
            raise FileNotFoundError(f"Missing bundle files: {missing}")
        
        # Load metadata
        metadata_result = load_bundle_metadata(bundle_dir)
        metadata = metadata_result["metadata"]
        depth_ladder_fields = metadata_result["depth_ladder_fields"]
        
        # Try to load bundle
        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        
        # Get feature_meta_path from policy or use default
        feature_meta_path = None
        if policy_path and policy_path.exists():
            import yaml
            with open(policy_path, "r") as f:
                policy = yaml.safe_load(f)
            if "entry_models" in policy and "v10_ctx" in policy["entry_models"]:
                feature_meta_path_str = policy["entry_models"]["v10_ctx"].get("feature_meta_path")
                if feature_meta_path_str:
                    feature_meta_path = (workspace_root / feature_meta_path_str).resolve()
        
        if feature_meta_path is None:
            # Use default
            feature_meta_path = workspace_root / "gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"
        
        if not feature_meta_path.exists():
            raise FileNotFoundError(f"feature_meta_path not found: {feature_meta_path}")
        
        log.info(f"Loading bundle from {bundle_dir}")
        log.info(f"Using feature_meta_path: {feature_meta_path}")
        
        bundle = load_entry_v10_ctx_bundle(
            bundle_dir=bundle_dir,
            feature_meta_path=feature_meta_path,
            device=None,  # Will use CPU
            is_replay=True,
        )
        
        # Extract bundle info
        bundle_info = {
            "bundle_dir": str(bundle_dir),
            "model_variant": metadata.get("model_variant"),
            "transformer_layers": depth_ladder_fields["transformer_layers"],
            "transformer_layers_baseline": depth_ladder_fields["transformer_layers_baseline"],
            "depth_ladder_delta": depth_ladder_fields["depth_ladder_delta"],
            "seq_input_dim": metadata.get("seq_input_dim"),
            "snap_input_dim": metadata.get("snap_input_dim"),
            "supports_context_features": metadata.get("supports_context_features", False),
        }
        
        # Try to compute bundle_sha256
        try:
            import hashlib
            model_path = bundle_dir / "model_state_dict.pt"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    bundle_sha256 = hashlib.sha256(f.read()).hexdigest()
                bundle_info["bundle_sha256"] = bundle_sha256
        except Exception as e:
            log.warning(f"Could not compute bundle_sha256: {e}")
        
        result["status"] = "OK"
        result["bundle_info"] = bundle_info
        result["file_check"] = file_check
        
        log.info("✅ Bundle load successful")
        return result
        
    except Exception as e:
        # Capture full error info
        result["status"] = "FAIL"
        result["exception"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        result["directory_listing"] = list_bundle_directory(bundle_dir)
        result["file_check"] = check_expected_bundle_files(bundle_dir)
        
        log.error(f"❌ Bundle load failed: {e}")
        log.error(traceback.format_exc())
        
        # Re-raise to ensure we don't hide the error
        raise


def main():
    parser = argparse.ArgumentParser(description="ENTRY_V10_CTX Bundle Loader Selftest")
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="Path to bundle directory")
    parser.add_argument("--policy", type=Path, default=None,
                        help="Optional policy YAML path (for feature_meta_path)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for result JSON (default: bundle_dir)")
    
    args = parser.parse_args()
    
    bundle_dir = args.bundle_dir.resolve()
    if not bundle_dir.exists():
        log.error(f"Bundle directory not found: {bundle_dir}")
        sys.exit(1)
    
    output_dir = args.output if args.output else bundle_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = test_bundle_load(bundle_dir, args.policy)
        
        # Write success result
        output_path = output_dir / "BUNDLE_LOAD_OK.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"✅ Written: {output_path}")
        
        return 0
        
    except Exception as e:
        # Write failure result
        result = {
            "bundle_dir": str(bundle_dir),
            "status": "FAIL",
            "exception": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
            "directory_listing": list_bundle_directory(bundle_dir),
            "file_check": check_expected_bundle_files(bundle_dir),
        }
        
        output_path = output_dir / "BUNDLE_LOAD_FAIL.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        log.error(f"❌ Written: {output_path}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
