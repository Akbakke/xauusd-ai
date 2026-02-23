#!/usr/bin/env python3
"""
Canonicalize exit router artifacts.

This script:
1. Finds all exit_router*.pkl files
2. Computes SHA256 for each
3. Identifies duplicates
4. Moves canonical copies to gx1/models/exit_router/
5. Updates all policy references
6. Generates cleanup report

Usage:
    python -m gx1.scripts.canonicalize_exit_router --repo_root . --out_dir reports/cleanup
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import yaml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Canonical directory
CANONICAL_EXIT_ROUTER_DIR = "gx1/models/exit_router/"


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not file_path.exists() or not file_path.is_file():
        return None
    
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    except Exception as e:
        log.warning(f"Failed to compute hash for {file_path}: {e}")
        return None


def find_all_exit_router_files(repo_root: Path) -> List[Dict[str, Any]]:
    """Find all exit_router*.pkl files in repository."""
    exit_router_files = []
    repo_root = repo_root.resolve()
    
    log.info("Scanning for exit_router*.pkl files...")
    
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        
        # Skip excluded directories
        if any(exclude in str(root_path.relative_to(repo_root)).split(os.sep) 
               for exclude in ["__pycache__", ".git", ".pytest_cache", "_archive_artifacts"]):
            dirs[:] = []
            continue
        
        for file in files:
            if file.startswith("exit_router") and file.endswith(".pkl"):
                file_path = root_path / file
                
                # Compute hash
                sha256 = compute_file_hash(file_path)
                if sha256 is None:
                    continue
                
                # Get relative path
                try:
                    rel_path = file_path.relative_to(repo_root)
                except ValueError:
                    continue
                
                # Get file stats
                stat = file_path.stat()
                
                exit_router_files.append({
                    "path": str(rel_path),
                    "abs_path": str(file_path.resolve()),
                    "size_bytes": stat.st_size,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "mtime": stat.st_mtime,
                    "sha256": sha256,
                    "filename": file,
                })
    
    log.info(f"Found {len(exit_router_files)} exit_router files")
    return exit_router_files


def find_hash_duplicates(files: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Find files with duplicate hashes."""
    hash_to_files = defaultdict(list)
    
    for file_info in files:
        hash_to_files[file_info["sha256"]].append(file_info)
    
    # Filter to only duplicates (2+ files with same hash)
    duplicates = {sha256: file_list for sha256, file_list in hash_to_files.items() 
                  if len(file_list) > 1}
    
    return duplicates


def select_canonical_file(file_list: List[Dict[str, Any]], repo_root: Path) -> Dict[str, Any]:
    """Select canonical file from list (prefer canonical dir, then newest)."""
    # Prefer files already in canonical dir
    canonical_files = [f for f in file_list if CANONICAL_EXIT_ROUTER_DIR in f["path"]]
    if canonical_files:
        return max(canonical_files, key=lambda x: x["mtime"])
    
    # Prefer files in prod_snapshot (current active)
    prod_files = [f for f in file_list if "prod_snapshot" in f["path"]]
    if prod_files:
        return max(prod_files, key=lambda x: x["mtime"])
    
    # Otherwise, prefer newest
    return max(file_list, key=lambda x: x["mtime"])


def find_policy_references(repo_root: Path) -> Dict[str, List[str]]:
    """Find all policy references to exit_router files."""
    references = defaultdict(list)
    
    # Search in configs
    config_patterns = [
        "gx1/configs/**/*.yaml",
        "gx1/configs/policies/**/*.yaml",
    ]
    
    for pattern in config_patterns:
        for config_path in repo_root.glob(pattern):
            if not config_path.is_file():
                continue
            
            try:
                with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                    # Search for exit_router references
                    if "exit_router" in content.lower():
                        # Try to parse YAML to find model_path
                        try:
                            with open(config_path, "r") as f2:
                                config = yaml.safe_load(f2)
                                
                            # Recursively search for model_path
                            def extract_paths(obj, prefix=""):
                                if isinstance(obj, dict):
                                    for key, value in obj.items():
                                        if key == "model_path" and isinstance(value, str) and "exit_router" in value.lower():
                                            references[value].append(str(config_path.relative_to(repo_root)))
                                        else:
                                            extract_paths(value, f"{prefix}.{key}")
                                elif isinstance(obj, list):
                                    for item in obj:
                                        extract_paths(item, prefix)
                            
                            extract_paths(config)
                        except Exception:
                            pass
            except Exception:
                pass
    
    return dict(references)


def update_policy_references(
    old_path: str,
    new_path: str,
    repo_root: Path,
    dry_run: bool = False
) -> int:
    """Update all policy references from old_path to new_path."""
    updated_count = 0
    
    # Find all YAML files
    for yaml_path in repo_root.glob("gx1/configs/**/*.yaml"):
        if not yaml_path.is_file():
            continue
        
        try:
            with open(yaml_path, "r") as f:
                content = f.read()
            
            if old_path in content:
                # Parse and update
                with open(yaml_path, "r") as f:
                    config = yaml.safe_load(f)
                
                file_updated = False
                
                # Recursively update model_path
                def update_paths(obj):
                    nonlocal file_updated
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if key == "model_path" and isinstance(value, str) and value == old_path:
                                obj[key] = new_path
                                file_updated = True
                                log.info(f"  Updating {yaml_path.relative_to(repo_root)}: {old_path} -> {new_path}")
                            else:
                                update_paths(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            update_paths(item)
                
                update_paths(config)
                
                if file_updated:
                    updated_count += 1
                    if not dry_run:
                        with open(yaml_path, "w") as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            log.warning(f"Failed to update {yaml_path}: {e}")
    
    return updated_count


def main():
    parser = argparse.ArgumentParser(description="Canonicalize exit router artifacts")
    parser.add_argument("--repo_root", type=str, default=".", help="Repository root")
    parser.add_argument("--out_dir", type=str, default="reports/cleanup", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Find all exit_router files
    exit_router_files = find_all_exit_router_files(repo_root)
    
    if not exit_router_files:
        log.warning("No exit_router files found!")
        return 0
    
    # Step 2: Compute hashes and find duplicates
    duplicates = find_hash_duplicates(exit_router_files)
    log.info(f"Found {len(duplicates)} hash duplicate groups")
    
    # Step 3: Find policy references
    references = find_policy_references(repo_root)
    log.info(f"Found {len(references)} unique exit_router paths referenced in policies")
    
    # Step 4: Select canonical files and plan moves
    canonical_dir = repo_root / CANONICAL_EXIT_ROUTER_DIR
    canonical_dir.mkdir(parents=True, exist_ok=True)
    
    moves = []
    archives = []
    
    # Group by hash
    hash_to_files = defaultdict(list)
    for file_info in exit_router_files:
        hash_to_files[file_info["sha256"]].append(file_info)
    
    for sha256, file_list in hash_to_files.items():
        canonical_file = select_canonical_file(file_list, repo_root)
        canonical_filename = canonical_file["filename"]
        canonical_target = canonical_dir / canonical_filename
        
        # Check if already in canonical
        if CANONICAL_EXIT_ROUTER_DIR in canonical_file["path"]:
            log.info(f"✅ {canonical_file['filename']} already in canonical dir")
            continue
        
        # Plan move to canonical
        moves.append({
            "source": canonical_file,
            "target": canonical_target,
            "target_rel": str(canonical_target.relative_to(repo_root)),
        })
        
        # Plan archive for duplicates
        for file_info in file_list:
            if file_info["path"] != canonical_file["path"]:
                archives.append({
                    "file": file_info,
                    "canonical": canonical_file,
                })
    
    # Step 5: Generate report
    report_path = out_dir / f"EXIT_ROUTER_CANONICALIZATION_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(f"# Exit Router Canonicalization Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total exit_router files found:** {len(exit_router_files)}\n")
        f.write(f"- **Hash duplicate groups:** {len(duplicates)}\n")
        f.write(f"- **Files to move to canonical:** {len(moves)}\n")
        f.write(f"- **Files to archive (duplicates):** {len(archives)}\n\n")
        
        f.write("## Files Found\n\n")
        f.write("| Path | Size (MB) | SHA256 | References |\n")
        f.write("|------|-----------|--------|------------|\n")
        for file_info in exit_router_files:
            refs = references.get(file_info["path"], [])
            ref_count = len(refs)
            sha256_short = file_info["sha256"][:16] + "..."
            f.write(f"| `{file_info['path']}` | {file_info['size_mb']:.2f} | `{sha256_short}` | {ref_count} |\n")
        
        f.write("\n## Canonicalization Plan\n\n")
        for move in moves:
            source = move["source"]
            target = move["target_rel"]
            f.write(f"### {source['filename']}\n\n")
            f.write(f"- **Source:** `{source['path']}`\n")
            f.write(f"- **Target:** `{target}`\n")
            f.write(f"- **SHA256:** `{source['sha256']}`\n")
            refs = references.get(source["path"], [])
            if refs:
                f.write(f"- **Referenced in:** {len(refs)} policies\n")
            f.write("\n")
    
    log.info(f"Report written: {report_path}")
    
    # Step 6: Execute moves (if not dry run)
    if not args.dry_run:
        for move in moves:
            source_path = Path(move["source"]["abs_path"])
            target_path = move["target"]
            
            if target_path.exists():
                log.warning(f"Target already exists: {target_path}, skipping")
                continue
            
            # Copy to canonical
            shutil.copy2(source_path, target_path)
            log.info(f"✅ Copied {source_path.name} to {target_path}")
            
            # Update policy references
            old_path = move["source"]["path"]
            new_path = move["target_rel"]
            updated = update_policy_references(old_path, new_path, repo_root, dry_run=False)
            log.info(f"  Updated {updated} policy references")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
