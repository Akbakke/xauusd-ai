#!/usr/bin/env python3
"""
Comprehensive repository overview and cleanup inventory.

Scans entire repo, classifies files, traces artifact references,
and generates safe archive scripts.

Usage:
    python -m gx1.scripts.repo_real_overview \
        --repo_root . \
        --out_dir reports/cleanup
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import yaml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Canonical directories for ACTIVE artifacts
CANONICAL_DIRS = [
    "models/entry_v10/",
    "models/entry_v10_ctx/",
    "models/xgb_calibration/",
    "gx1/models/entry_v9/",
    "gx1/models/entry_v10/",
]

# Artifact extensions to scan
ARTIFACT_EXTENSIONS = [
    ".joblib",
    ".pt",
    ".pth",
    ".pkl",
    ".onnx",
    ".bin",
]

# Metadata files
METADATA_PATTERNS = [
    "bundle_metadata.json",
    "feature_contract_hash.txt",
    "*.manifest.json",
]

# Directories to exclude from hashing (but count for stats)
EXCLUDE_FROM_HASH = [
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "venv",
    "env",
    ".venv",
    "runs/",
    "data/replay/",
    "gx1/wf_runs/",
    "gx1/live/",
    "_archive_artifacts/",
]

# Script patterns
SCRIPT_PATTERNS = [
    "**/scripts/**/*.py",
    "gx1/scripts/**/*.py",
    "**/train*.py",
    "**/eval*.py",
    "**/build*.py",
    "**/debug*.py",
]


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        log.warning(f"Failed to hash {filepath}: {e}")
        return ""


def should_exclude_from_hash(filepath: Path, repo_root: Path) -> bool:
    """Check if file should be excluded from hashing."""
    rel_path = str(filepath.relative_to(repo_root))
    return any(exclude in rel_path for exclude in EXCLUDE_FROM_HASH)


def get_file_counts(repo_root: Path) -> Dict:
    """Get file count statistics."""
    log.info("[OVERVIEW] Counting files...")
    
    file_counts = defaultdict(int)
    dir_counts = defaultdict(int)
    total_files = 0
    total_dirs = 0
    
    for root, dirs, files in os.walk(repo_root):
        # Skip .git
        if ".git" in root:
            continue
        
        rel_root = str(Path(root).relative_to(repo_root))
        if rel_root == ".":
            rel_root = "."
        
        # Count files per top-level directory
        top_dir = rel_root.split("/")[0] if "/" in rel_root else rel_root
        file_counts[top_dir] += len(files)
        dir_counts[top_dir] += len(dirs)
        total_files += len(files)
        total_dirs += len(dirs)
    
    return {
        "total_files": total_files,
        "total_dirs": total_dirs,
        "file_counts_by_dir": dict(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:50]),
        "dir_counts_by_dir": dict(sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:50]),
    }


def get_largest_files(repo_root: Path, top_n: int = 50) -> List[Dict]:
    """Get largest files in repo."""
    log.info(f"[OVERVIEW] Finding {top_n} largest files...")
    
    largest = []
    
    for root, dirs, files in os.walk(repo_root):
        # Skip .git and excluded dirs
        if ".git" in root or any(exclude in root for exclude in EXCLUDE_FROM_HASH):
            continue
        
        for file in files:
            filepath = Path(root) / file
            try:
                size = filepath.stat().st_size
                rel_path = str(filepath.relative_to(repo_root))
                largest.append({
                    "path": rel_path,
                    "size_bytes": size,
                    "size_mb": size / (1024**2),
                })
            except:
                pass
    
    return sorted(largest, key=lambda x: x["size_bytes"], reverse=True)[:top_n]


def get_largest_dirs(repo_root: Path, top_n: int = 20) -> List[Dict]:
    """Get largest directories using du (if available) or manual calculation."""
    log.info(f"[OVERVIEW] Finding {top_n} largest directories...")
    
    try:
        # Try using du command (faster)
        result = subprocess.run(
            ["du", "-sk"] + [str(repo_root / d) for d in ["gx1", "models", "data", "runs", "docs", "scripts", "reports"] if (repo_root / d).exists()],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        dirs = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    size_kb = int(parts[0])
                    path = parts[1]
                    rel_path = str(Path(path).relative_to(repo_root))
                    dirs.append({
                        "path": rel_path,
                        "size_kb": size_kb,
                        "size_mb": size_kb / 1024,
                    })
        
        return sorted(dirs, key=lambda x: x["size_kb"], reverse=True)[:top_n]
    except:
        log.warning("[OVERVIEW] du command failed, skipping directory sizes")
        return []


def scan_artifacts(repo_root: Path, hash_extensions: List[str]) -> Dict[str, Dict]:
    """Scan repository for all artifact files."""
    log.info("[OVERVIEW] Scanning for artifacts...")
    
    artifacts = {}
    repo_root = Path(repo_root).resolve()
    
    # Add metadata patterns
    all_patterns = hash_extensions + ["bundle_metadata.json", "feature_contract_hash.txt"]
    
    for root, dirs, files in os.walk(repo_root):
        # Skip excluded dirs
        if any(exclude in root for exclude in EXCLUDE_FROM_HASH):
            continue
        
        for file in files:
            filepath = Path(root) / file
            rel_path = str(filepath.relative_to(repo_root))
            
            # Check if matches artifact extension
            matches = False
            inferred_type = "unknown"
            
            if any(file.endswith(ext) for ext in hash_extensions):
                matches = True
                if file.endswith(".joblib"):
                    inferred_type = "xgb_model" if "xgb" in rel_path.lower() else "scaler_or_unknown"
                elif file.endswith((".pt", ".pth")):
                    inferred_type = "transformer_model" if "transformer" in rel_path.lower() or "model_state_dict" in file else "unknown"
                elif file.endswith(".pkl"):
                    inferred_type = "pickle_artifact"
                elif file.endswith(".onnx"):
                    inferred_type = "onnx_model"
            elif file == "bundle_metadata.json":
                matches = True
                inferred_type = "bundle_metadata"
            elif file == "feature_contract_hash.txt":
                matches = True
                inferred_type = "contract_hash"
            elif file.endswith(".manifest.json"):
                matches = True
                inferred_type = "manifest"
            
            if matches:
                try:
                    stat = filepath.stat()
                    hash_val = ""
                    if not should_exclude_from_hash(filepath, repo_root):
                        hash_val = compute_file_hash(filepath)
                    
                    artifacts[rel_path] = {
                        "path": rel_path,
                        "realpath": str(filepath.resolve()),
                        "size_bytes": stat.st_size,
                        "mtime": stat.st_mtime,
                        "hash": hash_val,
                        "inferred_type": inferred_type,
                    }
                except Exception as e:
                    log.warning(f"Failed to process {filepath}: {e}")
    
    log.info(f"[OVERVIEW] Found {len(artifacts)} artifacts")
    return artifacts


def find_references(repo_root: Path, artifacts: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Find all references to artifacts in configs and code."""
    log.info("[OVERVIEW] Tracing references...")
    
    references = defaultdict(list)
    repo_root = Path(repo_root).resolve()
    
    # Config patterns
    config_patterns = [
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
    ]
    
    # Code patterns (only for default paths/constants)
    code_patterns = [
        "gx1/**/*.py",
    ]
    
    all_patterns = config_patterns + code_patterns
    
    for pattern in all_patterns:
        for config_file in repo_root.glob(pattern):
            # Skip excluded dirs
            if any(exclude in str(config_file) for exclude in EXCLUDE_FROM_HASH):
                continue
            
            # Skip cleanup reports (they reference themselves)
            if "reports/cleanup" in str(config_file):
                continue
            
            # Skip if too large (likely not a config)
            if config_file.stat().st_size > 1_000_000:  # 1MB
                continue
            
            try:
                if config_file.suffix in [".yaml", ".yml"]:
                    with open(config_file) as f:
                        content = yaml.safe_load(f)
                        content_str = str(content)
                elif config_file.suffix == ".json":
                    with open(config_file) as f:
                        content = json.load(f)
                        content_str = json.dumps(content)
                elif config_file.suffix == ".py":
                    with open(config_file) as f:
                        content_str = f.read()
                else:
                    continue
                
                # Find artifact references
                for artifact_path, artifact_info in artifacts.items():
                    # Check if artifact path appears in content
                    if artifact_path in content_str or artifact_info["realpath"] in content_str:
                        references[artifact_path].append(str(config_file.relative_to(repo_root)))
                
            except Exception as e:
                log.debug(f"Failed to parse {config_file}: {e}")
    
    log.info(f"[OVERVIEW] Found references for {len(references)} artifacts")
    return dict(references)


def classify_artifacts(
    artifacts: Dict[str, Dict],
    references: Dict[str, List[str]],
    repo_root: Path,
) -> Dict[str, Dict]:
    """Classify artifacts as ACTIVE/REFERENCED/ORPHAN/UNKNOWN."""
    log.info("[OVERVIEW] Classifying artifacts...")
    
    repo_root = Path(repo_root).resolve()
    canonical_dirs_resolved = [repo_root / d for d in CANONICAL_DIRS]
    
    classified = {}
    
    for artifact_path, artifact_info in artifacts.items():
        artifact_full_path = Path(artifact_info["realpath"])
        rel_path = artifact_info["path"]
        
        # Check if in canonical directory
        in_canonical = any(
            artifact_full_path.is_relative_to(canonical_dir)
            for canonical_dir in canonical_dirs_resolved
        )
        
        # Check if referenced
        refs = references.get(artifact_path, [])
        is_referenced = len(refs) > 0
        
        # Check if referenced by canonical policy
        canonical_refs = [r for r in refs if "policies" in r and "2025_SNIPER_V1" in r]
        is_canonical_ref = len(canonical_refs) > 0
        
        # Classify
        if in_canonical and is_canonical_ref:
            classification = "ACTIVE"
        elif in_canonical and is_referenced:
            classification = "REFERENCED"
        elif is_referenced:
            classification = "REFERENCED"
        elif in_canonical:
            classification = "UNKNOWN"  # In canonical but not referenced - needs investigation
        else:
            classification = "ORPHAN"
        
        classified[artifact_path] = {
            **artifact_info,
            "classification": classification,
            "in_canonical_dir": in_canonical,
            "reference_count": len(refs),
            "references": refs,
            "canonical_refs": canonical_refs,
        }
    
    counts = defaultdict(int)
    for a in classified.values():
        counts[a["classification"]] += 1
    
    log.info(f"[OVERVIEW] Classified: {dict(counts)}")
    return classified


def find_duplicates(classified: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Find duplicate artifacts by hash."""
    log.info("[OVERVIEW] Finding duplicates...")
    
    hash_to_paths = defaultdict(list)
    for artifact_path, artifact_info in classified.items():
        if artifact_info["hash"]:
            hash_to_paths[artifact_info["hash"]].append(artifact_path)
    
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    
    log.info(f"[OVERVIEW] Found {len(duplicates)} duplicate groups")
    return duplicates


def scan_scripts(repo_root: Path) -> List[Dict]:
    """Scan for training/build/eval/debug scripts."""
    log.info("[OVERVIEW] Scanning scripts...")
    
    scripts = []
    seen_paths = set()
    repo_root = Path(repo_root).resolve()
    
    # Find all Python scripts matching patterns
    for pattern in SCRIPT_PATTERNS:
        for script_path in repo_root.glob(pattern):
            if any(exclude in str(script_path) for exclude in EXCLUDE_FROM_HASH):
                continue
            
            rel_path = str(script_path.relative_to(repo_root))
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            
            try:
                # Read first 50 lines to guess purpose
                with open(script_path) as f:
                    lines = [f.readline() for _ in range(50)]
                    content_preview = "".join(lines)
                
                # Guess purpose from filename and content
                purpose = "unknown"
                if "train" in script_path.name.lower():
                    purpose = "training"
                elif "eval" in script_path.name.lower() or "evaluate" in script_path.name.lower():
                    purpose = "evaluation"
                elif "build" in script_path.name.lower():
                    purpose = "dataset_building"
                elif "debug" in script_path.name.lower():
                    purpose = "debugging"
                elif "test" in script_path.name.lower():
                    purpose = "testing"
                
                # Check if referenced in docs/Makefile
                is_referenced = False
                # Simple check: look for script name in docs
                for doc_file in repo_root.glob("docs/**/*.md"):
                    try:
                        with open(doc_file) as f:
                            if script_path.name in f.read():
                                is_referenced = True
                                break
                    except:
                        pass
                
                # Check Makefile
                makefile = repo_root / "Makefile"
                if makefile.exists():
                    try:
                        with open(makefile) as f:
                            if script_path.name in f.read() or rel_path in f.read():
                                is_referenced = True
                    except:
                        pass
                
                scripts.append({
                    "path": rel_path,
                    "name": script_path.name,
                    "purpose": purpose,
                    "is_referenced": is_referenced,
                    "size_bytes": script_path.stat().st_size,
                })
            except Exception as e:
                log.warning(f"Failed to process script {script_path}: {e}")
    
    log.info(f"[OVERVIEW] Found {len(scripts)} scripts")
    return scripts


def generate_archive_script(
    classified: Dict[str, Dict],
    scripts: List[Dict],
    repo_root: Path,
) -> Path:
    """Generate shell script to archive ORPHAN artifacts and LEGACY scripts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create scripts/cleanup directory
    scripts_cleanup_dir = repo_root / "scripts" / "cleanup"
    scripts_cleanup_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_cleanup_dir / f"ARCHIVE_CANDIDATES_{timestamp}.sh"
    
    orphans = [path for path, info in classified.items() if info["classification"] == "ORPHAN"]
    legacy_scripts = [s for s in scripts if s["purpose"] in ["training", "evaluation", "dataset_building"] and not s["is_referenced"]]
    
    if not orphans and not legacy_scripts:
        log.info("[OVERVIEW] No candidates for archive")
        return script_path
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Archive ORPHAN artifacts and LEGACY scripts\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total orphans: {len(orphans)}\n")
        f.write(f"# Total legacy scripts: {len(legacy_scripts)}\n\n")
        
        f.write("ARCHIVE_DIR=\"_archive_artifacts/${timestamp}\"\n".replace("${timestamp}", timestamp))
        f.write("mkdir -p \"$ARCHIVE_DIR\"\n\n")
        
        f.write("echo \"Archiving ORPHAN artifacts...\"\n\n")
        
        for orphan_path in sorted(orphans):
            f.write(f"# {orphan_path}\n")
            f.write(f"mkdir -p \"$ARCHIVE_DIR/$(dirname '{orphan_path}')\"\n")
            f.write(f"mv '{orphan_path}' \"$ARCHIVE_DIR/{orphan_path}\"\n")
            f.write(f"echo \"Archived: {orphan_path}\"\n\n")
        
        if legacy_scripts:
            f.write("echo \"Archiving LEGACY scripts...\"\n\n")
            for script in sorted(legacy_scripts, key=lambda x: x["path"]):
                f.write(f"# {script['path']}\n")
                f.write(f"mkdir -p \"$ARCHIVE_DIR/$(dirname '{script['path']}')\"\n")
                f.write(f"mv '{script['path']}' \"$ARCHIVE_DIR/{script['path']}\"\n")
                f.write(f"echo \"Archived: {script['path']}\"\n\n")
        
        f.write("echo \"Archive complete. Review $ARCHIVE_DIR before deleting.\"\n")
    
    script_path.chmod(0o755)
    log.info(f"[OVERVIEW] ✅ Archive script: {script_path}")
    return script_path


def generate_reports(
    file_counts: Dict,
    largest_files: List[Dict],
    largest_dirs: List[Dict],
    classified: Dict[str, Dict],
    duplicates: Dict[str, List[str]],
    scripts: List[Dict],
    archive_script: Path,
    repo_root: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Generate markdown, JSON, and CSV reports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    json_path = output_dir / f"REPO_REAL_OVERVIEW_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "file_counts": file_counts,
            "largest_files": largest_files,
            "largest_dirs": largest_dirs,
            "classified_artifacts": classified,
            "duplicates": duplicates,
            "scripts": scripts,
            "canonical_dirs": CANONICAL_DIRS,
            "timestamp": timestamp,
        }, f, indent=2)
    
    # CSV report
    csv_path = output_dir / f"REPO_REAL_OVERVIEW_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path", "classification", "in_canonical_dir", "reference_count",
            "inferred_type", "size_bytes", "size_mb", "hash", "mtime"
        ])
        for artifact_path, artifact_info in sorted(classified.items()):
            writer.writerow([
                artifact_path,
                artifact_info["classification"],
                artifact_info["in_canonical_dir"],
                artifact_info["reference_count"],
                artifact_info["inferred_type"],
                artifact_info["size_bytes"],
                artifact_info["size_bytes"] / (1024**2),
                artifact_info["hash"][:16] + "..." if artifact_info["hash"] else "",
                datetime.fromtimestamp(artifact_info["mtime"]).isoformat(),
            ])
    
    # Markdown report
    md_path = output_dir / f"REPO_REAL_OVERVIEW_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Repository Real Overview\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # A) Repo file counts overview
        f.write("## A) Repo File Counts Overview\n\n")
        f.write(f"- **Total Files:** {file_counts['total_files']:,}\n")
        f.write(f"- **Total Directories:** {file_counts['total_dirs']:,}\n\n")
        
        f.write("### Top 50 Directories by File Count\n\n")
        f.write("| Directory | File Count |\n")
        f.write("|-----------|------------|\n")
        for dir_name, count in list(file_counts["file_counts_by_dir"].items())[:50]:
            f.write(f"| `{dir_name}` | {count:,} |\n")
        f.write("\n")
        
        f.write("### Top 50 Largest Files\n\n")
        f.write("| Path | Size (MB) |\n")
        f.write("|------|-----------|\n")
        for file_info in largest_files[:50]:
            f.write(f"| `{file_info['path']}` | {file_info['size_mb']:.2f} |\n")
        f.write("\n")
        
        f.write("### Top 20 Largest Directories\n\n")
        f.write("| Path | Size (MB) |\n")
        f.write("|------|-----------|\n")
        for dir_info in largest_dirs[:20]:
            f.write(f"| `{dir_info['path']}` | {dir_info['size_mb']:.2f} |\n")
        f.write("\n")
        
        # B) System files vs runtime artifacts vs models
        f.write("## B) System Files vs Runtime Artifacts vs Models\n\n")
        active_count = sum(1 for a in classified.values() if a["classification"] == "ACTIVE")
        referenced_count = sum(1 for a in classified.values() if a["classification"] == "REFERENCED")
        orphan_count = sum(1 for a in classified.values() if a["classification"] == "ORPHAN")
        unknown_count = sum(1 for a in classified.values() if a["classification"] == "UNKNOWN")
        
        f.write(f"- **ACTIVE Artifacts:** {active_count}\n")
        f.write(f"- **REFERENCED Artifacts:** {referenced_count}\n")
        f.write(f"- **ORPHAN Artifacts:** {orphan_count}\n")
        f.write(f"- **UNKNOWN Artifacts:** {unknown_count}\n\n")
        
        # C) Artifact inventory
        f.write("## C) Artifact Inventory\n\n")
        f.write(f"**Total Artifacts Scanned:** {len(classified)}\n\n")
        
        f.write("### By Classification\n\n")
        f.write("| Classification | Count | Total Size (MB) |\n")
        f.write("|----------------|-------|-----------------|\n")
        for classification in ["ACTIVE", "REFERENCED", "ORPHAN", "UNKNOWN"]:
            artifacts_of_type = [a for a in classified.values() if a["classification"] == classification]
            total_size = sum(a["size_bytes"] for a in artifacts_of_type)
            f.write(f"| {classification} | {len(artifacts_of_type)} | {total_size / (1024**2):.2f} |\n")
        f.write("\n")
        
        # D) Reference tracing
        f.write("## D) Reference Tracing\n\n")
        f.write("### Artifacts Outside Canonical Dirs\n\n")
        outside_canonical = [a for a in classified.values() if not a["in_canonical_dir"]]
        f.write(f"**Count:** {len(outside_canonical)}\n\n")
        if outside_canonical:
            f.write("| Path | Classification | References |\n")
            f.write("|------|----------------|------------|\n")
            for artifact in sorted(outside_canonical, key=lambda x: x["path"])[:20]:
                refs_str = ", ".join(artifact["references"][:3])
                if len(artifact["references"]) > 3:
                    refs_str += f" ... (+{len(artifact['references']) - 3} more)"
                f.write(f"| `{artifact['path']}` | {artifact['classification']} | {refs_str} |\n")
        f.write("\n")
        
        # E) Canonical directory enforcement
        f.write("## E) Canonical Directory Enforcement Check\n\n")
        f.write("### Canonical Directories\n\n")
        for canonical_dir in CANONICAL_DIRS:
            f.write(f"- `{canonical_dir}`\n")
        f.write("\n")
        
        f.write("### Artifacts in Canonical Dirs but Not Referenced\n\n")
        in_canonical_unreferenced = [a for a in classified.values() if a["in_canonical_dir"] and a["reference_count"] == 0]
        f.write(f"**Count:** {len(in_canonical_unreferenced)}\n\n")
        if in_canonical_unreferenced:
            f.write("| Path | Type |\n")
            f.write("|------|------|\n")
            for artifact in sorted(in_canonical_unreferenced, key=lambda x: x["path"])[:20]:
                f.write(f"| `{artifact['path']}` | {artifact['inferred_type']} |\n")
        f.write("\n")
        
        # Duplicates
        f.write("### Duplicate Artifacts (by hash)\n\n")
        duplicate_sizes = sorted(
            [(h, paths, sum(classified[p]["size_bytes"] for p in paths)) for h, paths in duplicates.items()],
            key=lambda x: x[2],
            reverse=True,
        )[:20]
        
        f.write("| Hash (first 16) | Paths | Total Size (MB) |\n")
        f.write("|-----------------|-------|-----------------|\n")
        for hash_val, paths, total_size_mb in duplicate_sizes:
            f.write(f"| `{hash_val[:16]}...` | {len(paths)} | {total_size_mb / (1024**2):.2f} |\n")
            for path in paths[:3]:
                f.write(f"  - `{path}`\n")
            if len(paths) > 3:
                f.write(f"  - ... and {len(paths) - 3} more\n")
        f.write("\n")
        
        # F) Script inventory
        f.write("## F) Script Inventory\n\n")
        f.write(f"**Total Scripts:** {len(scripts)}\n\n")
        
        f.write("### By Purpose\n\n")
        purpose_counts = defaultdict(int)
        for script in scripts:
            purpose_counts[script["purpose"]] += 1
        
        f.write("| Purpose | Count |\n")
        f.write("|---------|-------|\n")
        for purpose, count in sorted(purpose_counts.items()):
            f.write(f"| {purpose} | {count} |\n")
        f.write("\n")
        
        f.write("### Legacy Scripts (Not Referenced)\n\n")
        legacy = [s for s in scripts if not s["is_referenced"]]
        f.write(f"**Count:** {len(legacy)}\n\n")
        if legacy:
            f.write("| Path | Purpose | Size (KB) |\n")
            f.write("|------|---------|-----------|\n")
            for script in sorted(legacy, key=lambda x: x["path"])[:20]:
                f.write(f"| `{script['path']}` | {script['purpose']} | {script['size_bytes'] / 1024:.1f} |\n")
        f.write("\n")
        
        # G) Safe cleanup plan
        f.write("## G) Safe Cleanup Plan\n\n")
        f.write(f"- **ORPHAN Artifacts:** {orphan_count} (candidates for archive)\n")
        f.write(f"- **Legacy Scripts:** {len([s for s in scripts if not s['is_referenced']])} (candidates for archive)\n\n")
        
        orphan_size = sum(a["size_bytes"] for a in classified.values() if a["classification"] == "ORPHAN")
        f.write(f"- **Total ORPHAN Size:** {orphan_size / (1024**3):.2f} GB\n\n")
        
        f.write("### Archive Script Generated\n\n")
        archive_rel = str(archive_script) if archive_script.is_relative_to(repo_root) else str(Path(archive_script).relative_to(Path.cwd()))
        f.write(f"See: `{archive_rel}`\n\n")
        f.write("**Note:** Archive script only MOVES files to `_archive_artifacts/`, never deletes.\n\n")
    
    log.info(f"[OVERVIEW] ✅ Reports saved: {md_path}, {json_path}, {csv_path}")
    return md_path, json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Repository real overview")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="reports/cleanup")
    parser.add_argument("--exclude_dirs", type=str, default="", help="Comma-separated list of dirs to exclude")
    parser.add_argument("--hash_extensions", type=str, default=".joblib,.pt,.pth,.pkl,.onnx,.bin", help="Comma-separated extensions to hash")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode (print what would be archived)")
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse extensions
    hash_extensions = [ext.strip() for ext in args.hash_extensions.split(",") if ext.strip()]
    
    log.info(f"[OVERVIEW] Starting repository overview for {repo_root}")
    
    # A) File counts
    file_counts = get_file_counts(repo_root)
    
    # Largest files
    largest_files = get_largest_files(repo_root, top_n=50)
    
    # Largest dirs
    largest_dirs = get_largest_dirs(repo_root, top_n=20)
    
    # C) Artifact inventory
    artifacts = scan_artifacts(repo_root, hash_extensions)
    
    # D) Reference tracing
    references = find_references(repo_root, artifacts)
    
    # Classify
    classified = classify_artifacts(artifacts, references, repo_root)
    
    # Duplicates
    duplicates = find_duplicates(classified)
    
    # F) Script inventory
    scripts = scan_scripts(repo_root)
    
    # Generate archive script first
    archive_script = generate_archive_script(classified, scripts, repo_root)
    
    # Generate reports
    md_path, json_path, csv_path = generate_reports(
        file_counts,
        largest_files,
        largest_dirs,
        classified,
        duplicates,
        scripts,
        archive_script,
        repo_root,
        output_dir,
    )
    
    # Summary
    active_count = sum(1 for a in classified.values() if a["classification"] == "ACTIVE")
    referenced_count = sum(1 for a in classified.values() if a["classification"] == "REFERENCED")
    orphan_count = sum(1 for a in classified.values() if a["classification"] == "ORPHAN")
    unknown_count = sum(1 for a in classified.values() if a["classification"] == "UNKNOWN")
    
    orphan_size = sum(a["size_bytes"] for a in classified.values() if a["classification"] == "ORPHAN")
    
    log.info("=" * 80)
    log.info("[OVERVIEW] ✅ SUMMARY")
    log.info("=" * 80)
    log.info(f"Total artifacts scanned: {len(classified)}")
    log.info(f"  ACTIVE: {active_count}")
    log.info(f"  REFERENCED: {referenced_count}")
    log.info(f"  ORPHAN: {orphan_count}")
    log.info(f"  UNKNOWN: {unknown_count}")
    log.info(f"Archive candidates: {orphan_count} artifacts ({orphan_size / (1024**3):.2f} GB)")
    log.info(f"Reports: {md_path}, {json_path}, {csv_path}")
    log.info(f"Archive script: {archive_script}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
