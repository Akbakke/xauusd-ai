#!/usr/bin/env python3
"""
Model Artifact Hygiene - 3-Phase Cleanup

FASE A: Kartlegging (read-only)
FASE B: Quarantine (reversibelt)
FASE C: Delete (irreversibelt)

Usage:
    # FASE A: List all models
    python3 gx1/scripts/audit_model_artifacts.py --mode list
    
    # FASE B: Quarantine (dry-run)
    python3 gx1/scripts/audit_model_artifacts.py --mode quarantine --dry-run 1
    
    # FASE B: Quarantine (execute)
    python3 gx1/scripts/audit_model_artifacts.py --mode quarantine --dry-run 0
    
    # FASE C: Delete (requires allow-delete)
    python3 gx1/scripts/audit_model_artifacts.py --mode delete --allow-delete 1 --dry-run 0
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

# Model file extensions
MODEL_EXTENSIONS = {".joblib", ".pkl", ".json", ".bst", ".bin", ".onnx"}

# Keywords for model identification
MODEL_KEYWORDS = {
    "xgb", "xgboost", "entry", "transformer", "bundle", "v10", "v9",
    "sniper", "farm", "ctx", "hybrid", "fusion"
}

# Safety window: don't touch files modified in last 24 hours
SAFETY_WINDOW_HOURS = 24

# Don't quarantine files that match locked SHA (even if path differs)
def is_locked_sha(filepath: Path, truth_set: Dict[str, Dict[str, Any]]) -> bool:
    """Check if file SHA matches any locked SHA in truth set."""
    try:
        file_sha = compute_file_sha256(filepath)
        for truth_info in truth_set.values():
            if truth_info.get("sha256") == file_sha:
                return True
    except:
        pass
    return False


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    
    if "GX1_DATA_DATA" in os.environ:
        path = Path(os.environ["GX1_DATA_DATA"]).parent
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


def guess_model_type(filepath: Path) -> str:
    """Guess model type from path and content."""
    path_str = str(filepath).lower()
    
    if "xgb" in path_str or "xgboost" in path_str:
        return "xgb"
    elif "transformer" in path_str or "model_state_dict" in path_str:
        return "transformer"
    elif "bundle" in path_str or "metadata" in path_str:
        return "bundle"
    else:
        return "other"


def tag_model(filepath: Path) -> List[str]:
    """Tag model based on path patterns."""
    tags = []
    path_str = str(filepath).lower()
    
    # Version tags
    if "/v9/" in path_str or "v9" in filepath.name.lower():
        tags.append("v9")
    if "/v10/" in path_str or "v10" in filepath.name.lower():
        tags.append("v10")
    if "/v10_ctx/" in path_str or "v10_ctx" in filepath.name.lower():
        tags.append("v10_ctx")
    
    # Session tags
    if "_eu_" in path_str or "_eu." in path_str or filepath.name.lower().startswith("xgb_eu"):
        tags.append("eu-only")
    
    # Legacy tags
    if "/entry_v10/" in path_str and "/entry_v10_ctx/" not in path_str:
        tags.append("legacy-entry-v10")
    if "sniper" in path_str or "farm" in path_str:
        tags.append("sniper-farm")
    
    # Location tags
    if "/archive/" in path_str or "/reports/" in path_str:
        tags.append("non-canonical")
    
    return tags


def scan_model_artifacts(
    models_root: Path,
    since_days: int = 60,
) -> List[Dict[str, Any]]:
    """Scan for model artifacts."""
    artifacts = []
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=since_days)
    
    print(f"Scanning {models_root} for model artifacts...")
    print(f"  Extensions: {MODEL_EXTENSIONS}")
    print(f"  Since: {since_days} days ago")
    
    for root, dirs, files in os.walk(models_root):
        root_path = Path(root)
        
        # Skip quarantine and archive directories
        if "quarantine" in root_path.parts or "archive" in root_path.parts:
            continue
        
        for file in files:
            filepath = root_path / file
            
            # Check extension
            if filepath.suffix.lower() not in MODEL_EXTENSIONS:
                continue
            
            # Check keywords in path
            path_str = str(filepath).lower()
            if not any(kw in path_str for kw in MODEL_KEYWORDS):
                continue
            
            try:
                stat = filepath.stat()
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                
                # Skip if too old (optional filter)
                if mtime < cutoff_time:
                    continue
                
                # Compute SHA256
                sha256 = compute_file_sha256(filepath)
                
                # Guess type and tag
                model_type = guess_model_type(filepath)
                tags = tag_model(filepath)
                
                artifact = {
                    "path": str(filepath),
                    "size_bytes": stat.st_size,
                    "mtime": mtime.isoformat(),
                    "sha256": sha256,
                    "type": model_type,
                    "tags": tags,
                }
                
                artifacts.append(artifact)
                
            except Exception as e:
                print(f"  WARNING: Failed to process {filepath}: {e}")
    
    print(f"Found {len(artifacts)} model artifacts")
    return artifacts


def find_truth_used_set(
    reports_root: Path,
    gx1_data_dir: Path,
    since_days: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """
    Find truth used set by parsing:
    1. MASTER_MODEL_LOCK.json (authority #1)
    2. MODEL_USED_CAPSULE.json files
    3. Bundle metadata and configs
    
    Returns dict mapping sha256 -> artifact info
    """
    truth_set = {}
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=since_days)
    
    print(f"Finding truth used set...")
    
    # 0. PRIORITY #1: Check for MASTER_MODEL_LOCK.json (SSoT authority)
    print(f"  Checking for MASTER_MODEL_LOCK.json...")
    bundle_dirs = [
        gx1_data_dir / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION",
    ]
    
    for bundle_dir in bundle_dirs:
        lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
        if lock_path.exists():
            print(f"    Found: {lock_path}")
            with open(lock_path, "r") as f:
                lock = json.load(f)
            
            # Add model from lock
            model_sha = lock.get("model_sha256")
            model_rel = lock.get("model_path_relative")
            if model_sha and model_rel:
                model_path = bundle_dir / model_rel
                truth_set[model_sha] = {
                    "path": str(model_path),
                    "sha256": model_sha,
                    "type": "xgb_model",
                    "source": "MASTER_MODEL_LOCK",
                    "xgb_mode": lock.get("xgb_mode"),
                }
                print(f"      + {model_rel} (MASTER LOCK)")
            
            # Add meta from lock
            meta_sha = lock.get("meta_sha256")
            meta_rel = lock.get("meta_path_relative")
            if meta_sha and meta_rel:
                meta_path = bundle_dir / meta_rel
                truth_set[meta_sha] = {
                    "path": str(meta_path),
                    "sha256": meta_sha,
                    "type": "xgb_meta",
                    "source": "MASTER_MODEL_LOCK",
                }
                print(f"      + {meta_rel} (MASTER LOCK)")
            
            # Add GO marker if present
            go_marker_sha = lock.get("go_marker_sha256")
            go_marker_name = lock.get("go_marker_filename")
            if go_marker_sha and go_marker_name:
                go_path = bundle_dir / go_marker_name
                truth_set[go_marker_sha] = {
                    "path": str(go_path),
                    "sha256": go_marker_sha,
                    "type": "go_marker",
                    "source": "MASTER_MODEL_LOCK",
                }
    
    print(f"  MASTER_MODEL_LOCK added {len(truth_set)} items to truth set")
    
    print(f"  Scanning for MODEL_USED_CAPSULE.json files...")
    
    # 1. Scan for MODEL_USED_CAPSULE.json files (primary SSoT)
    # Use newest capsule per SHA256 to handle capsule updates
    print(f"  Scanning for MODEL_USED_CAPSULE.json files...")
    capsule_count = 0
    capsule_timestamps: Dict[str, datetime.datetime] = {}  # sha256 -> timestamp of newest capsule
    capsule_data: Dict[str, Dict[str, Any]] = {}  # sha256 -> capsule info
    
    def process_capsule_file(capsule_file: Path) -> int:
        """Process a single capsule file, tracking by SHA256 and timestamp."""
        nonlocal capsule_count
        count = 0
        try:
            # Check mtime
            if capsule_file.stat().st_mtime < cutoff_time.timestamp():
                return 0
            
            with open(capsule_file, "r") as f:
                capsule = json.load(f)
            
            capsule_ts_str = capsule.get("timestamp", "")
            try:
                capsule_ts = datetime.datetime.fromisoformat(capsule_ts_str.replace("Z", "+00:00"))
            except:
                capsule_ts = datetime.datetime.fromtimestamp(capsule_file.stat().st_mtime)
            
            # Extract XGB model info - track newest capsule per SHA256
            xgb_models_info = capsule.get("xgb_models", {})
            for session, model_info in xgb_models_info.items():
                model_path_str = model_info.get("selected_xgb_model_path")
                model_sha256 = model_info.get("selected_xgb_model_sha256")
                
                if model_path_str and model_sha256:
                    # Check if this is newer than any existing entry for this SHA
                    existing_ts = capsule_timestamps.get(model_sha256)
                    if existing_ts is None or capsule_ts > existing_ts:
                        model_path = Path(model_path_str)
                        if model_path.exists():
                            # Verify SHA256 matches
                            actual_sha256 = compute_file_sha256(model_path)
                            if actual_sha256 == model_sha256:
                                capsule_timestamps[model_sha256] = capsule_ts
                                capsule_data[model_sha256] = {
                                    "path": str(model_path),
                                    "source": "MODEL_USED_CAPSULE",
                                    "run_id": capsule.get("run_id", "unknown"),
                                    "policy_id": capsule.get("policy_id", "unknown"),
                                    "bundle_sha256": capsule.get("bundle_sha256"),
                                    "capsule_path": str(capsule_file),
                                    "capsule_timestamp": capsule_ts_str,
                                }
                                count += 1
            
            return 1  # Processed 1 capsule
            
        except Exception as e:
            print(f"    WARNING: Failed to parse {capsule_file}: {e}")
            return 0
    
    # Check replay_eval directories
    replay_eval_root = reports_root / "replay_eval"
    if replay_eval_root.exists():
        for capsule_file in replay_eval_root.rglob("MODEL_USED_CAPSULE.json"):
            capsule_count += process_capsule_file(capsule_file)
    
    # Also check gx1/wf_runs for capsules
    wf_runs_root = Path("gx1/wf_runs")
    if wf_runs_root.exists():
        for capsule_file in wf_runs_root.rglob("MODEL_USED_CAPSULE.json"):
            capsule_count += process_capsule_file(capsule_file)
    
    # Also check GX1_DATA/reports/repo_audit for capsules (from write_model_used_capsule.py)
    repo_audit_root = reports_root / "repo_audit"
    if repo_audit_root.exists():
        for capsule_file in repo_audit_root.rglob("MODEL_USED_CAPSULE.json"):
            capsule_count += process_capsule_file(capsule_file)
    
    # Add capsule data to truth_set (newest capsule per SHA256 wins)
    for sha256, info in capsule_data.items():
        truth_set[sha256] = info
    
    print(f"    Found {capsule_count} MODEL_USED_CAPSULE.json files")
    
    # 2. Scan RUN_IDENTITY.json files (fallback)
    if replay_eval_root.exists():
        print(f"  Scanning RUN_IDENTITY.json files...")
        for replay_dir in replay_eval_root.iterdir():
            if not replay_dir.is_dir():
                continue
            
            for identity_file in replay_dir.rglob("RUN_IDENTITY.json"):
                try:
                    if identity_file.stat().st_mtime < cutoff_time.timestamp():
                        continue
                    
                    with open(identity_file, "r") as f:
                        identity = json.load(f)
                    
                    # Extract bundle_dir
                    bundle_dir_str = identity.get("bundle_dir_resolved")
                    if bundle_dir_str:
                        bundle_dir = Path(bundle_dir_str)
                        if bundle_dir.exists():
                            # Scan bundle directory for models
                            for model_file in bundle_dir.rglob("*"):
                                if model_file.is_file() and model_file.suffix.lower() in MODEL_EXTENSIONS:
                                    sha256 = compute_file_sha256(model_file)
                                    # Only add if not already in truth_set (capsule takes priority)
                                    if sha256 not in truth_set:
                                        truth_set[sha256] = {
                                            "path": str(model_file),
                                            "source": "RUN_IDENTITY",
                                            "run_id": identity.get("run_id", "unknown"),
                                            "policy_id": identity.get("policy_id", "unknown"),
                                        }
                    
                except Exception as e:
                    print(f"    WARNING: Failed to parse {identity_file}: {e}")
    
    # 3. Scan bundle_metadata.json files
    models_root = gx1_data_dir / "models"
    if models_root.exists():
        print(f"  Scanning bundle_metadata.json files...")
        for meta_file in models_root.rglob("bundle_metadata.json"):
            try:
                if meta_file.stat().st_mtime < cutoff_time.timestamp():
                    continue
                
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                
                bundle_dir = meta_file.parent
                
                # Check xgb_models_by_session
                xgb_models = metadata.get("xgb_models_by_session", {})
                for session, model_name in xgb_models.items():
                    model_path = bundle_dir / model_name
                    if model_path.exists():
                        sha256 = compute_file_sha256(model_path)
                        truth_set[sha256] = {
                            "path": str(model_path),
                            "source": "bundle_metadata",
                            "bundle_dir": str(bundle_dir),
                        }
                
                # Check model_state_dict.pt (transformer)
                model_state = bundle_dir / "model_state_dict.pt"
                if model_state.exists():
                    sha256 = compute_file_sha256(model_state)
                    truth_set[sha256] = {
                        "path": str(model_state),
                        "source": "bundle_metadata",
                        "bundle_dir": str(bundle_dir),
                    }
                
            except Exception as e:
                print(f"    WARNING: Failed to parse {meta_file}: {e}")
    
    print(f"  Found {len(truth_set)} artifacts in truth used set")
    return truth_set


def classify_candidates(
    artifacts: List[Dict[str, Any]],
    truth_set: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Classify artifacts into used and candidates for removal.
    
    An artifact is "used" only if BOTH:
    1. Its SHA256 matches an entry in truth_set
    2. Its path matches the truth_set path (to avoid duplicates from copied files)
    """
    used = []
    candidates = []
    
    truth_shas = set(truth_set.keys())
    
    # Build path->sha256 map for exact path matching
    truth_paths = {truth_set[sha]["path"]: sha for sha in truth_shas}
    
    for artifact in artifacts:
        sha256 = artifact["sha256"]
        artifact_path = artifact["path"]
        
        # Check if this exact path is in truth_set
        if artifact_path in truth_paths:
            artifact["in_truth_set"] = True
            artifact["truth_source"] = truth_set[sha256].get("source", "unknown")
            used.append(artifact)
        elif sha256 in truth_shas:
            # SHA matches but path is different (duplicate/legacy copy)
            # Mark as candidate for removal, not "used"
            artifact["in_truth_set"] = False
            artifact["risk_reason"] = "DUPLICATE_SHA256_DIFFERENT_PATH"
            candidates.append(artifact)
        else:
            artifact["in_truth_set"] = False
            artifact["risk_reason"] = determine_risk_reason(artifact)
            candidates.append(artifact)
    
    return used, candidates


def determine_risk_reason(artifact: Dict[str, Any]) -> str:
    """Determine why artifact is a candidate for removal."""
    tags = artifact.get("tags", [])
    path = artifact.get("path", "")
    
    reasons = []
    
    if "v9" in tags:
        reasons.append("v9 legacy")
    if "eu-only" in tags:
        reasons.append("EU-only (not universal)")
    if "legacy-entry-v10" in tags:
        reasons.append("legacy entry_v10 (not v10_ctx)")
    if "sniper-farm" in tags:
        reasons.append("sniper/farm (not universal)")
    if "non-canonical" in tags:
        reasons.append("non-canonical location")
    if not artifact.get("in_truth_set", False):
        reasons.append("not referenced in recent runs")
    
    return "; ".join(reasons) if reasons else "unknown"


def write_reports(
    output_dir: Path,
    artifacts: List[Dict[str, Any]],
    used: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    truth_set: Dict[str, Dict[str, Any]],
) -> None:
    """Write all reports to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MODEL_INDEX.csv
    csv_path = output_dir / "MODEL_INDEX.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "size_bytes", "mtime", "sha256", "type", "tags", "in_truth_set"
        ])
        writer.writeheader()
        for artifact in artifacts:
            row = {
                "path": artifact.get("path", ""),
                "size_bytes": artifact.get("size_bytes", 0),
                "mtime": artifact.get("mtime", ""),
                "sha256": artifact.get("sha256", ""),
                "type": artifact.get("type", ""),
                "tags": ",".join(artifact.get("tags", [])),
                "in_truth_set": str(artifact.get("in_truth_set", False)),
            }
            writer.writerow(row)
    print(f"  Wrote: {csv_path}")
    
    # MODEL_INDEX.json
    json_path = output_dir / "MODEL_INDEX.json"
    with open(json_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"  Wrote: {json_path}")
    
    # TRUTH_USED_SET.json
    truth_path = output_dir / "TRUTH_USED_SET.json"
    truth_list = [
        {
            "path": a["path"],
            "sha256": a["sha256"],
            "source": a.get("truth_source", "unknown"),
        }
        for a in used
    ]
    with open(truth_path, "w") as f:
        json.dump(truth_list, f, indent=2)
    print(f"  Wrote: {truth_path}")
    
    # CANDIDATES_TO_REMOVE.csv
    candidates_path = output_dir / "CANDIDATES_TO_REMOVE.csv"
    with open(candidates_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "size_bytes", "mtime", "sha256", "type", "tags", "risk_reason"
        ])
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda x: x.get("size_bytes", 0), reverse=True):
            row = {
                "path": candidate.get("path", ""),
                "size_bytes": candidate.get("size_bytes", 0),
                "mtime": candidate.get("mtime", ""),
                "sha256": candidate.get("sha256", ""),
                "type": candidate.get("type", ""),
                "tags": ",".join(candidate.get("tags", [])),
                "risk_reason": candidate.get("risk_reason", ""),
            }
            writer.writerow(row)
    print(f"  Wrote: {candidates_path}")
    
    # RISK_REPORT.md
    risk_path = output_dir / "RISK_REPORT.md"
    with open(risk_path, "w") as f:
        f.write("# Model Hygiene Risk Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"Total artifacts: {len(artifacts)}\n")
        f.write(f"In truth set: {len(used)}\n")
        f.write(f"Candidates for removal: {len(candidates)}\n\n")
        
        # Top 20 largest candidates
        f.write("## Top 20 Largest Candidates\n\n")
        f.write("| Path | Size (MB) | Risk Reason |\n")
        f.write("|------|-----------|-------------|\n")
        
        top_candidates = sorted(candidates, key=lambda x: x.get("size_bytes", 0), reverse=True)[:20]
        for candidate in top_candidates:
            size_mb = candidate.get("size_bytes", 0) / (1024 * 1024)
            path = candidate.get("path", "")
            reason = candidate.get("risk_reason", "unknown")
            # Truncate path if too long
            if len(path) > 80:
                path = "..." + path[-77:]
            f.write(f"| `{path}` | {size_mb:.2f} | {reason} |\n")
        
        # Summary by tag
        f.write("\n## Summary by Tag\n\n")
        tag_counts = {}
        for candidate in candidates:
            for tag in candidate.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- `{tag}`: {count} artifacts\n")
    
    print(f"  Wrote: {risk_path}")


def quarantine_artifacts(
    candidates: List[Dict[str, Any]],
    gx1_data_dir: Path,
    truth_set: Optional[Dict[str, Dict[str, Any]]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Quarantine candidate artifacts.
    
    Safety rules:
    - Don't quarantine files modified in last 24 hours
    - Don't quarantine files with SHA matching locked SHA (even if path differs)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_dir = gx1_data_dir / "quarantine" / f"MODEL_HYGIENE_{timestamp}"
    
    manifest = {
        "timestamp": timestamp,
        "quarantine_dir": str(quarantine_dir),
        "dry_run": dry_run,
        "actions": [],
        "skipped": {
            "safety_window": 0,
            "locked_sha": 0,
        },
    }
    
    print(f"Quarantine target: {quarantine_dir}")
    print(f"Dry run: {dry_run}")
    
    if not dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    # Safety check: don't touch files modified in last 24 hours
    safety_cutoff = datetime.datetime.now() - datetime.timedelta(hours=SAFETY_WINDOW_HOURS)
    
    for candidate in candidates:
        src_path = Path(candidate["path"])
        
        # Safety check 1: mtime < 24h
        mtime = datetime.datetime.fromisoformat(candidate["mtime"])
        if mtime > safety_cutoff:
            print(f"  SKIP (safety window < 24h): {src_path}")
            manifest["skipped"]["safety_window"] += 1
            continue
        
        # Safety check 2: SHA matches locked SHA (even if path differs)
        if truth_set and is_locked_sha(src_path, truth_set):
            print(f"  SKIP (matches locked SHA): {src_path}")
            manifest["skipped"]["locked_sha"] += 1
            continue
        
        # Preserve directory structure
        try:
            # Make path relative to GX1_DATA
            rel_path = src_path.relative_to(gx1_data_dir)
            dst_path = quarantine_dir / rel_path
        except ValueError:
            # If not under GX1_DATA, use full path structure
            dst_path = quarantine_dir / "external" / src_path.name
        
        action = {
            "src": str(src_path),
            "dst": str(dst_path),
            "sha256": candidate["sha256"],
            "size_bytes": candidate["size_bytes"],
        }
        
        if dry_run:
            print(f"  [DRY-RUN] Would move: {src_path} -> {dst_path}")
        else:
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                print(f"  MOVED: {src_path} -> {dst_path}")
                action["status"] = "moved"
            except Exception as e:
                print(f"  ERROR moving {src_path}: {e}")
                action["status"] = "error"
                action["error"] = str(e)
        
        manifest["actions"].append(action)
    
    # Write manifest
    manifest_path = quarantine_dir / "QUARANTINE_MANIFEST.json"
    if not dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Wrote: {manifest_path}")
        
        # Write undo script
        undo_script = quarantine_dir / "UNDO_QUARANTINE.sh"
        with open(undo_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Undo quarantine - move files back\n\n")
            for action in manifest["actions"]:
                if action.get("status") == "moved":
                    f.write(f'mv "{action["dst"]}" "{action["src"]}"\n')
        undo_script.chmod(0o755)
        print(f"  Wrote: {undo_script}")
    
    return manifest


def delete_quarantined(
    quarantine_manifest_path: Path,
    allow_delete: bool = False,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Delete quarantined artifacts."""
    if not allow_delete:
        raise ValueError("--allow-delete 1 required for delete mode")
    
    if dry_run:
        raise ValueError("--dry-run 0 required for delete mode")
    
    if not quarantine_manifest_path.exists():
        raise FileNotFoundError(f"Quarantine manifest not found: {quarantine_manifest_path}")
    
    with open(quarantine_manifest_path, "r") as f:
        manifest = json.load(f)
    
    delete_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "quarantine_manifest": str(quarantine_manifest_path),
        "deleted": [],
    }
    
    print(f"Deleting from quarantine: {manifest['quarantine_dir']}")
    
    for action in manifest.get("actions", []):
        if action.get("status") != "moved":
            continue
        
        dst_path = Path(action["dst"])
        if dst_path.exists():
            try:
                dst_path.unlink()
                print(f"  DELETED: {dst_path}")
                delete_log["deleted"].append({
                    "path": str(dst_path),
                    "sha256": action.get("sha256"),
                })
            except Exception as e:
                print(f"  ERROR deleting {dst_path}: {e}")
    
    # Write delete log
    quarantine_dir = Path(manifest["quarantine_dir"])
    delete_log_path = quarantine_dir / "DELETE_LOG.json"
    with open(delete_log_path, "w") as f:
        json.dump(delete_log, f, indent=2)
    print(f"  Wrote: {delete_log_path}")
    
    # Write summary
    summary_path = quarantine_dir / "POST_DELETE_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Post-Delete Summary\n\n")
        f.write(f"Deleted: {len(delete_log['deleted'])} files\n")
        f.write(f"Timestamp: {delete_log['timestamp']}\n")
    
    print(f"  Wrote: {summary_path}")
    
    return delete_log


def main():
    parser = argparse.ArgumentParser(
        description="Model Artifact Hygiene - 3-Phase Cleanup"
    )
    parser.add_argument(
        "--gx1-data-root",
        type=Path,
        default=None,
        help="GX1_DATA root directory (auto-resolved if not provided)"
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=None,
        help="Reports root directory (default: <GX1_DATA>/reports)"
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Models root directory (default: <GX1_DATA>/models)"
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=60,
        help="Only consider artifacts modified in last N days (default: 60)"
    )
    parser.add_argument(
        "--mode",
        choices=["list", "quarantine", "delete"],
        default="list",
        help="Operation mode (default: list)"
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=1,
        help="Dry run mode (default: 1 for quarantine/delete, 0 to execute)"
    )
    parser.add_argument(
        "--allow-delete",
        type=int,
        default=0,
        help="Allow delete operations (default: 0, must be 1 for delete mode)"
    )
    parser.add_argument(
        "--quarantine-manifest",
        type=Path,
        default=None,
        help="Path to quarantine manifest (required for delete mode)"
    )
    parser.add_argument(
        "--require-truth-set",
        type=int,
        default=0,
        help="Hard fail if TRUTH_USED_SET is empty (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Resolve directories
    gx1_data = args.gx1_data_root or resolve_gx1_data_dir()
    reports_root = args.reports_root or (gx1_data / "reports")
    models_root = args.models_root or (gx1_data / "models")
    
    print(f"GX1_DATA: {gx1_data}")
    print(f"Reports root: {reports_root}")
    print(f"Models root: {models_root}")
    print()
    
    if args.mode == "list":
        # FASE A: Kartlegging
        print("=" * 60)
        print("FASE A: KARTLEGGING")
        print("=" * 60)
        
        # Scan artifacts
        artifacts = scan_model_artifacts(models_root, since_days=args.since_days)
        
        # Find truth used set
        truth_set = find_truth_used_set(reports_root, gx1_data, since_days=args.since_days)
        
        # Classify
        used, candidates = classify_candidates(artifacts, truth_set)
        
        # Check if truth set is required
        if args.require_truth_set and len(truth_set) == 0:
            print()
            print("=" * 60)
            print("FATAL: TRUTH_USED_SET is empty")
            print("=" * 60)
            print("Cannot prove model usage. Run a smoke test to generate MODEL_USED_CAPSULE.json:")
            print("  python3 gx1/scripts/train_xgb_calibrator_multiyear.py --smoke-xgb-infer")
            print("  # Or run a replay to generate capsules")
            return 1
        
        # Write reports
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "repo_audit" / f"MODEL_HYGIENE_{timestamp}"
        write_reports(output_dir, artifacts, used, candidates, truth_set)
        
        print()
        print(f"Reports written to: {output_dir}")
        print(f"  - MODEL_INDEX.csv")
        print(f"  - MODEL_INDEX.json")
        print(f"  - TRUTH_USED_SET.json ({len(truth_set)} artifacts)")
        print(f"  - CANDIDATES_TO_REMOVE.csv ({len(candidates)} candidates)")
        print(f"  - RISK_REPORT.md")
        
    elif args.mode == "quarantine":
        # FASE B: Quarantine
        print("=" * 60)
        print("FASE B: QUARANTINE")
        print("=" * 60)
        
        # Load candidates from latest report
        audit_dirs = sorted(
            (WORKSPACE_ROOT / "reports" / "repo_audit").glob("MODEL_HYGIENE_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not audit_dirs:
            print("ERROR: No audit reports found. Run --mode list first.")
            return 1
        
        latest_audit = audit_dirs[0]
        candidates_path = latest_audit / "CANDIDATES_TO_REMOVE.csv"
        
        if not candidates_path.exists():
            print(f"ERROR: CANDIDATES_TO_REMOVE.csv not found in {latest_audit}")
            return 1
        
        # Load candidates
        candidates = []
        with open(candidates_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append({
                    "path": row["path"],
                    "size_bytes": int(row["size_bytes"]),
                    "mtime": row["mtime"],
                    "sha256": row["sha256"],
                    "tags": row["tags"].split(",") if row["tags"] else [],
                    "risk_reason": row["risk_reason"],
                })
        
        print(f"Loaded {len(candidates)} candidates from {latest_audit}")
        
        # Quarantine
        # Load truth set for safety checks
        truth_set = {}
        if args.require_truth_set:
            truth_set = find_truth_used_set(reports_root, gx1_data, since_days=60)
        
        manifest = quarantine_artifacts(
            candidates,
            gx1_data,
            truth_set=truth_set,
            dry_run=bool(args.dry_run),
        )
        
        if not args.dry_run:
            print()
            print(f"Quarantine complete: {manifest['quarantine_dir']}")
            print(f"  - QUARANTINE_MANIFEST.json")
            print(f"  - UNDO_QUARANTINE.sh")
        
    elif args.mode == "delete":
        # FASE C: Delete
        print("=" * 60)
        print("FASE C: DELETE")
        print("=" * 60)
        
        if not args.quarantine_manifest:
            # Find latest quarantine manifest
            quarantine_dirs = sorted(
                (gx1_data / "quarantine").glob("MODEL_HYGIENE_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if not quarantine_dirs:
                print("ERROR: No quarantine directories found. Run --mode quarantine first.")
                return 1
            
            args.quarantine_manifest = quarantine_dirs[0] / "QUARANTINE_MANIFEST.json"
        
        delete_log = delete_quarantined(
            args.quarantine_manifest,
            allow_delete=bool(args.allow_delete),
            dry_run=bool(args.dry_run),
        )
        
        print()
        print(f"Delete complete: {len(delete_log['deleted'])} files deleted")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
