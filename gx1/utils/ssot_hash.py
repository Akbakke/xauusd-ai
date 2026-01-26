"""
SSoT (Single Source of Truth) hash utilities for deterministic bundle_sha256 computation.

This module provides functions to compute SHA256 hashes of files and bundles
for audit trail and reproducibility in replay runs.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def sha256_file(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hex digest
        
    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"[SSOT_FAIL] File not found: {file_path}")
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """
    Compute SHA256 hash of bytes.
    
    Args:
        data: Bytes to hash
        
    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(data).hexdigest()


def resolve_artifact_paths_from_policy(
    policy_path: Path,
    bundle_dir_override: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    """
    Resolve all artifact paths from policy YAML.
    
    Args:
        policy_path: Path to policy YAML file
        bundle_dir_override: Optional bundle_dir override (CLI or ENV, highest priority)
        
    Returns:
        Dict with keys: bundle_dir, feature_meta_path, seq_scaler_path, snap_scaler_path, xgb_models
        Values are resolved Path objects or None if not specified
        
    Raises:
        FileNotFoundError: If policy file does not exist
        ValueError: If policy structure is invalid
    """
    import os
    import yaml
    
    if not policy_path.exists():
        raise FileNotFoundError(f"[SSOT_FAIL] Policy file not found: {policy_path}")
    
    with open(policy_path, "r") as f:
        policy = yaml.safe_load(f)
    
    entry_models = policy.get("entry_models", {})
    v10_ctx_cfg = entry_models.get("v10_ctx", {})
    
    if not v10_ctx_cfg:
        raise ValueError(f"[SSOT_FAIL] entry_models.v10_ctx not found in policy: {policy_path}")
    
    # Resolve bundle_dir with priority: CLI override > ENV override > Policy
    bundle_dir = None
    bundle_dir_source = None
    
    # Priority A: CLI override (bundle_dir_override parameter)
    if bundle_dir_override:
        bundle_dir = Path(bundle_dir_override).resolve()
        bundle_dir_source = "cli"
    # Priority B: ENV override (GX1_BUNDLE_DIR)
    elif os.getenv("GX1_BUNDLE_DIR"):
        bundle_dir = Path(os.getenv("GX1_BUNDLE_DIR")).resolve()
        bundle_dir_source = "env"
    # Priority C: Policy (resolve relative to policy file's directory, not cwd)
    else:
        bundle_dir_str = v10_ctx_cfg.get("bundle_dir")
        if bundle_dir_str:
            bundle_dir_path = Path(bundle_dir_str)
            if bundle_dir_path.is_absolute():
                bundle_dir = bundle_dir_path.resolve()
            else:
                # Resolve relative to policy file's directory (not cwd)
                policy_dir = policy_path.resolve().parent
                bundle_dir = (policy_dir / bundle_dir_path).resolve()
            bundle_dir_source = "policy"
    
    # Find repo root for other paths (relative to repo root, not policy dir)
    repo_root = policy_path.resolve().parent
    max_levels = 10
    for _ in range(max_levels):
        if (repo_root / ".git").exists() or (repo_root / "Makefile").exists() or (repo_root / "setup.py").exists():
            break
        if repo_root.parent == repo_root:  # Reached filesystem root
            break
        repo_root = repo_root.parent
    else:
        # If we didn't find repo root, assume current working directory is repo root
        repo_root = Path(os.getcwd()).resolve()
    
    # Log bundle_dir resolution
    if bundle_dir:
        log.info(f"[SSOT] bundle_dir resolved from {bundle_dir_source}: {bundle_dir}")
        if not bundle_dir.exists():
            raise FileNotFoundError(
                f"[SSOT_FAIL] bundle_dir does not exist: {bundle_dir} "
                f"(resolved from {bundle_dir_source} in policy {policy_path})"
            )
    
    feature_meta_str = v10_ctx_cfg.get("feature_meta_path")
    feature_meta_path = (repo_root / feature_meta_str).resolve() if feature_meta_str else None
    
    seq_scaler_str = v10_ctx_cfg.get("seq_scaler_path")
    seq_scaler_path = (repo_root / seq_scaler_str).resolve() if seq_scaler_str else None
    
    snap_scaler_str = v10_ctx_cfg.get("snap_scaler_path")
    snap_scaler_path = (repo_root / snap_scaler_str).resolve() if snap_scaler_str else None
    
    # XGB models are in entry_config, not entry_models
    entry_config_str = policy.get("entry_config", "")
    xgb_models = None
    if entry_config_str:
        entry_config_path = (repo_root / entry_config_str).resolve()
        if entry_config_path.exists():
            # XGB models are referenced in entry_config YAML
            # For now, we'll hash the entry_config file itself
            # TODO: If XGB models are separate files, resolve them here
            xgb_models = entry_config_path
    
    return {
        "bundle_dir": bundle_dir,
        "bundle_dir_source": bundle_dir_source,  # "cli" | "env" | "policy" | None
        "feature_meta_path": feature_meta_path,
        "seq_scaler_path": seq_scaler_path,
        "snap_scaler_path": snap_scaler_path,
        "xgb_models": xgb_models,
    }


def compute_bundle_sha256(
    policy_path: Path,
    resolved_artifact_paths: Optional[Dict[str, Optional[Path]]] = None,
    bundle_dir_override: Optional[Path] = None,
) -> str:
    """
    Compute deterministic bundle_sha256 from policy + artifacts.
    
    The hash is computed over a canonical JSON representation:
    {
        "policy_sha256": "...",
        "artifacts": [
            {"path": "relative/path", "sha256": "..."},
            ...
        ]
    }
    
    Args:
        policy_path: Path to policy YAML file (must exist)
        resolved_artifact_paths: Optional dict from resolve_artifact_paths_from_policy()
                                 If None, will be computed automatically
        
    Returns:
        SHA256 hex digest of bundle
        
    Raises:
        FileNotFoundError: If any required artifact is missing
        RuntimeError: If bundle_sha256 cannot be computed
    """
    # Resolve artifact paths if not provided
    if resolved_artifact_paths is None:
        resolved_artifact_paths = resolve_artifact_paths_from_policy(policy_path, bundle_dir_override)
    
    # Hash policy file
    policy_sha256 = sha256_file(policy_path)
    
    # Build canonical artifact list (sorted by path for determinism)
    artifacts = []
    
    # Required artifacts
    required_artifacts = {
        "bundle_dir": "bundle",
        "feature_meta_path": "feature_meta.json",
    }
    
    for key, artifact_name in required_artifacts.items():
        artifact_path = resolved_artifact_paths.get(key)
        if not artifact_path:
            raise RuntimeError(
                f"[SSOT_FAIL] Required artifact missing: {key}. "
                f"Ensure policy YAML specifies entry_models.v10_ctx.{key}. "
                f"Policy: {policy_path}"
            )
        
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"[SSOT_FAIL] Artifact file/directory not found: {artifact_path} "
                f"(resolved from {key} in policy {policy_path})"
            )
        
        # For bundle_dir, hash bundle_metadata.json if it exists, else model_state_dict.pt
        if key == "bundle_dir":
            bundle_metadata_path = artifact_path / "bundle_metadata.json"
            model_state_path = artifact_path / "model_state_dict.pt"
            
            if bundle_metadata_path.exists():
                artifact_sha256 = sha256_file(bundle_metadata_path)
                artifacts.append({
                    "path": f"bundle/bundle_metadata.json",
                    "sha256": artifact_sha256,
                })
            elif model_state_path.exists():
                artifact_sha256 = sha256_file(model_state_path)
                artifacts.append({
                    "path": f"bundle/model_state_dict.pt",
                    "sha256": artifact_sha256,
                })
            else:
                raise FileNotFoundError(
                    f"[SSOT_FAIL] Bundle directory missing required file: "
                    f"neither bundle_metadata.json nor model_state_dict.pt found in {artifact_path}"
                )
        else:
            # Regular file
            artifact_sha256 = sha256_file(artifact_path)
            # Use relative path from repo root for canonical identifier
            repo_root = policy_path.parent
            # Find repo root (same logic as in resolve_artifact_paths_from_policy)
            for _ in range(5):
                if (repo_root / ".git").exists() or (repo_root / "setup.py").exists() or (repo_root / "Makefile").exists():
                    break
                if repo_root.parent == repo_root:
                    break
                repo_root = repo_root.parent
            else:
                repo_root = policy_path.parent
            
            try:
                rel_path = artifact_path.relative_to(repo_root)
            except ValueError:
                # If not relative, use absolute path as fallback (but log warning)
                log.warning(f"[SSOT] Artifact path not relative to repo root: {artifact_path}")
                rel_path = Path(artifact_path.name)
            
            artifacts.append({
                "path": str(rel_path),
                "sha256": artifact_sha256,
            })
    
    # Optional artifacts (scaler paths, xgb models)
    # HARD-FAIL: If specified in policy but missing, fail immediately
    optional_artifacts = {
        "seq_scaler_path": "seq_scaler",
        "snap_scaler_path": "snap_scaler",
        "xgb_models": "entry_config.yaml",  # XGB models are in entry_config
    }
    
    # Find repo root (same logic as in resolve_artifact_paths_from_policy)
    repo_root = policy_path.resolve().parent
    max_levels = 10
    for _ in range(max_levels):
        if (repo_root / ".git").exists() or (repo_root / "Makefile").exists() or (repo_root / "setup.py").exists():
            break
        if repo_root.parent == repo_root:
            break
        repo_root = repo_root.parent
    else:
        import os
        repo_root = Path(os.getcwd()).resolve()
    
    for key, artifact_name in optional_artifacts.items():
        artifact_path = resolved_artifact_paths.get(key)
        if artifact_path:
            # If path is specified in policy, it MUST exist (hard-fail if missing)
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"[SSOT_FAIL] Optional artifact specified in policy but not found: {artifact_path} "
                    f"(resolved from {key} in policy {policy_path}). "
                    f"HARD-FAIL: Artifact must exist if specified in policy."
                )
            # Artifact exists, hash it
            artifact_sha256 = sha256_file(artifact_path)
            try:
                rel_path = artifact_path.relative_to(repo_root)
            except ValueError:
                rel_path = Path(artifact_path.name)
            
            artifacts.append({
                "path": str(rel_path),
                "sha256": artifact_sha256,
            })
    
    # Sort artifacts by path for determinism
    artifacts.sort(key=lambda x: x["path"])
    
    # Build canonical JSON (sorted keys)
    bundle_manifest = {
        "policy_sha256": policy_sha256,
        "artifacts": artifacts,
    }
    
    # Serialize to JSON with sorted keys
    bundle_manifest_json = json.dumps(bundle_manifest, sort_keys=True, separators=(",", ":"))
    bundle_manifest_bytes = bundle_manifest_json.encode("utf-8")
    
    # Compute final SHA256
    bundle_sha256 = sha256_bytes(bundle_manifest_bytes)
    
    log.info(f"[SSOT] Computed bundle_sha256={bundle_sha256[:16]}... from {len(artifacts)} artifacts")
    
    return bundle_sha256
