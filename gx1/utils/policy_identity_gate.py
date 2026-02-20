#!/usr/bin/env python3
"""
Policy Identity Gate for GX1 (TRUTH/SMOKE).

Purpose:
- Enforce that the replay is using the canonical policy file.
- Verify that the policy's SHA256 matches the expected value.
- Fail early with a clear capsule if there is a mismatch.

Contract (when truth_or_smoke=True):
- Requires policy_path to be provided and exist.
- Computes the policy's SHA256.
- Reads the expected policy SHA256 from canonical policy file (if available).
- If SHA256 mismatch: write POLICY_IDENTITY_FATAL.json capsule and exit code 2.
- Always write POLICY_IDENTITY.md atomically to output_dir (or /tmp fallback).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# --- Utility functions (copied from env_identity_gate for self-containment) ---
def _now_utc_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _fallback_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "gx1_policy_identity_gate"
    if not _safe_mkdir(root):
        root = Path("/tmp")
    run_dir = root / f"run_{_now_utc_compact()}_{os.getpid()}"
    _safe_mkdir(run_dir)
    return run_dir


def _choose_report_dir(output_dir: Path) -> Path:
    try:
        if output_dir is not None and _safe_mkdir(output_dir):
            return output_dir
    except Exception:
        pass
    return _fallback_dir()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_canonical_policy_path() -> Optional[Path]:
    """Get canonical policy path from environment or default location."""
    # Check environment variable first
    canonical_path_str = os.getenv("GX1_CANONICAL_POLICY_PATH")
    if canonical_path_str:
        canonical_path = Path(canonical_path_str).resolve()
        if canonical_path.exists():
            return canonical_path
    
    # Default canonical location
    default_path = Path("/home/andre2/GX1_DATA/configs/policies/canonical/TRUTH_BASELINE_V12AB91.yaml")
    if default_path.exists():
        return default_path.resolve()
    
    return None


def _read_policy_id(policy_path: Path) -> Optional[str]:
    """Extract policy_id from policy YAML file."""
    try:
        import yaml
        with open(policy_path, "r") as f:
            policy_data = yaml.safe_load(f)
        
        # Try to extract policy_id from various possible locations
        if isinstance(policy_data, dict):
            # Check replay_config.policy_module
            replay_config = policy_data.get("replay_config", {})
            if isinstance(replay_config, dict):
                policy_module = replay_config.get("policy_module")
                if policy_module:
                    # Extract policy_id from module name (e.g., "gx1.policy.entry_policy_sniper_v10_ctx" -> "entry_policy_sniper_v10_ctx")
                    if "." in policy_module:
                        return policy_module.split(".")[-1]
                    return policy_module
            
            # Fallback: use filename stem
            return policy_path.stem
    except Exception:
        pass
    
    # Ultimate fallback: filename stem
    return policy_path.stem


def _render_md(
    policy_path: Path,
    policy_sha256: str,
    canonical_policy_path: Optional[Path],
    canonical_sha256: Optional[str],
    policy_id: Optional[str],
    git_head_sha: Optional[str],
) -> str:
    """Render the POLICY_IDENTITY report in Markdown format."""
    lines = [
        f"# POLICY_IDENTITY",
        f"",
        f"## Policy Path",
        f"",
        f"- **Absolute Path:** `{policy_path.resolve()}`",
        f"- **Policy ID:** `{policy_id or 'N/A'}`",
        f"",
        f"## Policy SHA256",
        f"",
        f"- **Computed:** `{policy_sha256}`",
    ]
    
    if canonical_policy_path:
        lines.extend([
            f"- **Canonical Path:** `{canonical_policy_path}`",
            f"- **Canonical SHA256:** `{canonical_sha256 or 'N/A'}`",
            f"- **Match:** {'✅ YES' if canonical_sha256 and policy_sha256 == canonical_sha256 else '❌ NO'}",
        ])
    else:
        lines.append(f"- **Canonical Path:** Not found (using provided policy)")
    
    if git_head_sha:
        lines.extend([
            f"",
            f"## Git Metadata",
            f"",
            f"- **Git HEAD SHA:** `{git_head_sha}`",
        ])
    
    lines.extend([
        f"",
        f"---",
        f"*Report generated by policy_identity_gate.py at {datetime.utcnow().isoformat()}Z*",
    ])
    
    return "\n".join(lines)


def _get_git_head_sha(repo_path: Path) -> Optional[str]:
    """Get git HEAD SHA if available."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def run_policy_identity_gate_or_fatal(
    output_dir: Path,
    truth_or_smoke: bool,
    policy_path: Path,
) -> None:
    """
    Run policy identity gate and hard-fail (exit code 2) in TRUTH/SMOKE if mismatch.
    
    Args:
        output_dir: Output directory for reports
        truth_or_smoke: Whether this is a TRUTH/SMOKE run
        policy_path: Path to policy YAML file
    """
    if not truth_or_smoke:
        return
    
    report_dir = _choose_report_dir(output_dir)
    fatal_path = report_dir / "POLICY_IDENTITY_FATAL.json"
    md_path = report_dir / "POLICY_IDENTITY.md"
    
    # Verify policy_path exists
    policy_path_resolved = Path(policy_path).resolve()
    if not policy_path_resolved.exists():
        msg = f"Policy file does not exist: {policy_path_resolved}"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "policy_path": str(policy_path_resolved),
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    # Compute policy SHA256
    try:
        policy_sha256 = _sha256_file(policy_path_resolved)
    except Exception as e:
        msg = f"Failed to compute SHA256 for policy file: {e}"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "policy_path": str(policy_path_resolved),
            "error": str(e),
            "sys.executable": sys.executable,
            "cwd": str(Path.cwd()),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    # Get canonical policy path and SHA256
    canonical_policy_path = _get_canonical_policy_path()
    canonical_sha256 = None
    if canonical_policy_path:
        try:
            canonical_sha256 = _sha256_file(canonical_policy_path)
        except Exception as e:
            # Log warning but don't fail - canonical might not exist yet
            sys.stderr.write(f"[POLICY_IDENTITY_GATE] WARNING: Failed to read canonical policy {canonical_policy_path}: {e}\n")
    
    # Extract policy_id
    policy_id = _read_policy_id(policy_path_resolved)
    
    # Get git HEAD SHA (if available)
    git_head_sha = None
    workspace_root = Path(__file__).resolve().parents[2]
    if (workspace_root / ".git").exists():
        git_head_sha = _get_git_head_sha(workspace_root)
    
    # In TRUTH/SMOKE, require canonical policy match if canonical exists
    if canonical_policy_path and canonical_sha256:
        if policy_sha256 != canonical_sha256:
            msg = f"Policy SHA256 mismatch: actual={policy_sha256}, canonical={canonical_sha256}"
            capsule = {
                "status": "FAIL",
                "message": msg,
                "policy_path": str(policy_path_resolved),
                "policy_sha256": policy_sha256,
                "canonical_policy_path": str(canonical_policy_path),
                "canonical_sha256": canonical_sha256,
                "policy_id": policy_id,
                "sys.executable": sys.executable,
                "cwd": str(Path.cwd()),
            }
            _write_json_atomic(fatal_path, capsule)
            raise SystemExit(2)
    
    # Write POLICY_IDENTITY.md
    md_content = _render_md(
        policy_path=policy_path_resolved,
        policy_sha256=policy_sha256,
        canonical_policy_path=canonical_policy_path,
        canonical_sha256=canonical_sha256,
        policy_id=policy_id,
        git_head_sha=git_head_sha,
    )
    _write_text_atomic(md_path, md_content)
