#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify PROD freeze structure.

Checks that all required artifacts are present and valid.
"""
from pathlib import Path
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_prod_freeze(prod_dir: Path = Path("gx1/prod/current")) -> bool:
    """
    Verify PROD freeze structure.
    
    Args:
        prod_dir: Path to PROD freeze directory
        
    Returns:
        True if all artifacts are present and valid
    """
    logger.info(f"[VERIFY] Checking PROD freeze: {prod_dir}")
    
    required_files = [
        "policy.yaml",
        "exit_router_v3_tree.pkl",
    ]
    
    optional_files = [
        "feature_manifest.json",
    ]
    
    all_present = True
    
    # Check required files
    for filename in required_files:
        file_path = prod_dir / filename
        if file_path.exists():
            file_hash = compute_file_hash(file_path)
            file_size = file_path.stat().st_size
            logger.info(f"  ✅ {filename}: {file_size} bytes, sha256={file_hash[:16]}...")
        else:
            logger.error(f"  ❌ {filename}: MISSING")
            all_present = False
    
    # Check optional files
    for filename in optional_files:
        file_path = prod_dir / filename
        if file_path.exists():
            file_hash = compute_file_hash(file_path)
            file_size = file_path.stat().st_size
            logger.info(f"  ✅ {filename}: {file_size} bytes, sha256={file_hash[:16]}...")
        else:
            logger.warning(f"  ⚠️  {filename}: NOT PRESENT (optional)")
    
    # Check entry_models directory
    entry_models_dir = prod_dir / "entry_models"
    if entry_models_dir.exists():
        model_files = list(entry_models_dir.glob("*.joblib")) + list(entry_models_dir.glob("*.pkl"))
        if model_files:
            logger.info(f"  ✅ entry_models/: {len(model_files)} model files")
        else:
            logger.warning(f"  ⚠️  entry_models/: No model files found")
    else:
        logger.warning(f"  ⚠️  entry_models/: Directory not present (optional)")
    
    if all_present:
        logger.info("[VERIFY] ✅ PROD freeze structure is valid")
    else:
        logger.error("[VERIFY] ❌ PROD freeze structure is INVALID")
    
    return all_present


if __name__ == "__main__":
    import sys
    success = verify_prod_freeze()
    sys.exit(0 if success else 1)

