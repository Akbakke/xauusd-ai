#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate run_header.json artifact at startup.

Computes SHA256 hashes for:
- Policy file
- Entry model(s)
- Router model
- Feature manifest

Includes git commit hash if available.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    if not file_path.exists():
        return None
    
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def generate_run_header(
    policy_path: Path,
    router_model_path: Optional[Path] = None,
    entry_model_paths: Optional[list[Path]] = None,
    feature_manifest_path: Optional[Path] = None,
    output_dir: Path = Path("gx1/wf_runs"),
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate run_header.json artifact.
    
    Args:
        policy_path: Path to policy YAML
        router_model_path: Path to router model (optional)
        entry_model_paths: List of entry model paths (optional)
        feature_manifest_path: Path to feature manifest (optional)
        output_dir: Output directory for run_header.json
        run_tag: Run tag/identifier (optional)
        
    Returns:
        Run header dict
    """
    header = {
        "timestamp": datetime.now().isoformat(),
        "run_tag": run_tag,
        "artifacts": {},
        "git_commit": get_git_commit_hash(),
    }
    
    # Policy hash
    if policy_path.exists():
        policy_hash = compute_file_hash(policy_path)
        policy_size = policy_path.stat().st_size
        header["artifacts"]["policy"] = {
            "path": str(policy_path.resolve()),
            "sha256": policy_hash,
            "size_bytes": policy_size,
        }
        logger.info("[RUN_HEADER] Policy: %s (%d bytes, sha256=%s...)", policy_path.name, policy_size, policy_hash[:16])
    
    # Router model hash
    if router_model_path and router_model_path.exists():
        router_hash = compute_file_hash(router_model_path)
        router_size = router_model_path.stat().st_size
        header["artifacts"]["router_model"] = {
            "path": str(router_model_path.resolve()),
            "sha256": router_hash,
            "size_bytes": router_size,
        }
        logger.info("[RUN_HEADER] Router model: %s (%d bytes, sha256=%s...)", router_model_path.name, router_size, router_hash[:16])
    
    # Entry model hashes
    if entry_model_paths:
        header["artifacts"]["entry_models"] = []
        for model_path in entry_model_paths:
            if model_path.exists():
                model_hash = compute_file_hash(model_path)
                model_size = model_path.stat().st_size
                header["artifacts"]["entry_models"].append({
                    "path": str(model_path.resolve()),
                    "sha256": model_hash,
                    "size_bytes": model_size,
                })
                logger.info("[RUN_HEADER] Entry model: %s (%d bytes, sha256=%s...)", model_path.name, model_size, model_hash[:16])
    
    # Feature manifest hash
    if feature_manifest_path and feature_manifest_path.exists():
        manifest_hash = compute_file_hash(feature_manifest_path)
        manifest_size = feature_manifest_path.stat().st_size
        header["artifacts"]["feature_manifest"] = {
            "path": str(feature_manifest_path.resolve()),
            "sha256": manifest_hash,
            "size_bytes": manifest_size,
        }
        logger.info("[RUN_HEADER] Feature manifest: %s (%d bytes, sha256=%s...)", feature_manifest_path.name, manifest_size, manifest_hash[:16])
    
    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / "run_header.json"
    with open(header_path, "w") as f:
        json.dump(header, f, indent=2)
    
    logger.info("[RUN_HEADER] Run header saved to: %s", header_path)
    
    return header


def load_run_header(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load run_header.json from run directory.
    
    Args:
        run_dir: Run directory path
        
    Returns:
        Run header dict, or None if not found
    """
    header_path = run_dir / "run_header.json"
    if not header_path.exists():
        # Try parallel_chunks subdirectory
        header_path = run_dir / "parallel_chunks" / "chunk_0" / "run_header.json"
        if not header_path.exists():
            return None
    
    try:
        with open(header_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load run_header.json: {e}")
        return None

