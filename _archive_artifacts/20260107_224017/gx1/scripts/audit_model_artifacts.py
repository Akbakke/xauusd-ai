#!/usr/bin/env python3
"""
Audit model artifacts in repository.

Scans for all model artifacts, computes hashes, finds references in configs,
and classifies as ACTIVE/REFERENCED/ORPHAN.

Usage:
    python -m gx1.scripts.audit_model_artifacts \
        --repo_root . \
        --output_dir reports/cleanup
"""

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
    ".json",  # model metadata
    ".manifest.json",
]

# Directories to exclude from scan
EXCLUDE_DIRS = [
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


def scan_artifacts(repo_root: Path) -> Dict[str, Dict]:
    """Scan repository for all artifact files."""
    log.info("[AUDIT] Scanning for artifacts...")
    
    artifacts = {}
    repo_root = Path(repo_root).resolve()
    
    for ext in ARTIFACT_EXTENSIONS:
        for filepath in repo_root.rglob(f"*{ext}"):
            # Skip excluded directories
            if any(exclude in str(filepath.relative_to(repo_root)) for exclude in EXCLUDE_DIRS):
                continue
            
            rel_path = str(filepath.relative_to(repo_root))
            artifacts[rel_path] = {
                "path": str(filepath.resolve()),
                "rel_path": rel_path,
                "size_bytes": filepath.stat().st_size,
                "extension": ext,
                "hash": compute_file_hash(filepath),
            }
    
    log.info(f"[AUDIT] Found {len(artifacts)} artifacts")
    return artifacts


def find_references(repo_root: Path) -> Dict[str, List[str]]:
    """Find all references to artifacts in config files."""
    log.info("[AUDIT] Scanning for references...")
    
    references = defaultdict(list)
    repo_root = Path(repo_root).resolve()
    
    # Config patterns to search
    config_patterns = [
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
    ]
    
    # Also search in markdown docs (optional)
    doc_patterns = [
        "docs/**/*.md",
    ]
    
    all_patterns = config_patterns + doc_patterns
    
    for pattern in all_patterns:
        for config_file in repo_root.glob(pattern):
            # Skip excluded directories
            if any(exclude in str(config_file.relative_to(repo_root)) for exclude in EXCLUDE_DIRS):
                continue
            
            try:
                if config_file.suffix in [".yaml", ".yml"]:
                    with open(config_file) as f:
                        content = yaml.safe_load(f)
                        # Convert to string for pattern matching
                        content_str = str(content)
                elif config_file.suffix == ".json":
                    with open(config_file) as f:
                        content = json.load(f)
                        content_str = json.dumps(content)
                elif config_file.suffix == ".md":
                    with open(config_file) as f:
                        content_str = f.read()
                else:
                    continue
                
                # Find artifact paths in content
                for artifact_path, artifact_info in artifacts.items():
                    # Check if artifact path appears in content
                    if artifact_path in content_str or artifact_info["path"] in content_str:
                        references[artifact_path].append(str(config_file.relative_to(repo_root)))
                
            except Exception as e:
                log.warning(f"Failed to parse {config_file}: {e}")
    
    log.info(f"[AUDIT] Found references for {len(references)} artifacts")
    return dict(references)


def classify_artifacts(
    artifacts: Dict[str, Dict],
    references: Dict[str, List[str]],
    repo_root: Path,
) -> Dict[str, Dict]:
    """Classify artifacts as ACTIVE/REFERENCED/ORPHAN."""
    log.info("[AUDIT] Classifying artifacts...")
    
    repo_root = Path(repo_root).resolve()
    canonical_dirs_resolved = [repo_root / d for d in CANONICAL_DIRS]
    
    classified = {}
    
    for artifact_path, artifact_info in artifacts.items():
        artifact_full_path = Path(artifact_info["path"])
        rel_path = artifact_info["rel_path"]
        
        # Check if in canonical directory
        in_canonical = any(
            artifact_full_path.is_relative_to(canonical_dir)
            for canonical_dir in canonical_dirs_resolved
        )
        
        # Check if referenced
        refs = references.get(artifact_path, [])
        is_referenced = len(refs) > 0
        
        # Classify
        if in_canonical and is_referenced:
            classification = "ACTIVE"
        elif is_referenced:
            classification = "REFERENCED"
        else:
            classification = "ORPHAN"
        
        classified[artifact_path] = {
            **artifact_info,
            "classification": classification,
            "in_canonical_dir": in_canonical,
            "reference_count": len(refs),
            "references": refs,
        }
    
    log.info(f"[AUDIT] Classified: {sum(1 for a in classified.values() if a['classification'] == 'ACTIVE')} ACTIVE, "
             f"{sum(1 for a in classified.values() if a['classification'] == 'REFERENCED')} REFERENCED, "
             f"{sum(1 for a in classified.values() if a['classification'] == 'ORPHAN')} ORPHAN")
    
    return classified


def find_duplicates(classified: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Find duplicate artifacts by hash."""
    log.info("[AUDIT] Finding duplicates...")
    
    hash_to_paths = defaultdict(list)
    for artifact_path, artifact_info in classified.items():
        if artifact_info["hash"]:
            hash_to_paths[artifact_info["hash"]].append(artifact_path)
    
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    
    log.info(f"[AUDIT] Found {len(duplicates)} duplicate groups")
    return duplicates


def generate_archive_script(
    classified: Dict[str, Dict],
    output_dir: Path,
) -> Path:
    """Generate shell script to archive ORPHAN artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = output_dir / f"ARCHIVE_ORPHANS_{timestamp}.sh"
    
    orphans = [path for path, info in classified.items() if info["classification"] == "ORPHAN"]
    
    if not orphans:
        log.info("[AUDIT] No orphans to archive")
        return script_path
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Archive ORPHAN artifacts\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total orphans: {len(orphans)}\n\n")
        
        f.write("ARCHIVE_DIR=\"_archive_artifacts/${timestamp}\"\n".replace("${timestamp}", timestamp))
        f.write("mkdir -p \"$ARCHIVE_DIR\"\n\n")
        
        f.write("echo \"Archiving ORPHAN artifacts...\"\n\n")
        
        for orphan_path in sorted(orphans):
            f.write(f"# {orphan_path}\n")
            f.write(f"mkdir -p \"$ARCHIVE_DIR/$(dirname '{orphan_path}')\"\n")
            f.write(f"mv '{orphan_path}' \"$ARCHIVE_DIR/{orphan_path}\"\n")
            f.write(f"echo \"Archived: {orphan_path}\"\n\n")
        
        f.write("echo \"Archive complete. Review $ARCHIVE_DIR before deleting.\"\n")
    
    script_path.chmod(0o755)
    log.info(f"[AUDIT] ✅ Archive script: {script_path}")
    return script_path


def generate_report(
    classified: Dict[str, Dict],
    duplicates: Dict[str, List[str]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate markdown and JSON reports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    json_path = output_dir / f"artifacts_inventory_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "classified": classified,
            "duplicates": duplicates,
            "canonical_dirs": CANONICAL_DIRS,
            "timestamp": timestamp,
        }, f, indent=2)
    
    # Markdown report
    md_path = output_dir / f"MODEL_ARTIFACT_INVENTORY_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Model Artifact Inventory\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        active_count = sum(1 for a in classified.values() if a["classification"] == "ACTIVE")
        referenced_count = sum(1 for a in classified.values() if a["classification"] == "REFERENCED")
        orphan_count = sum(1 for a in classified.values() if a["classification"] == "ORPHAN")
        total_size = sum(a["size_bytes"] for a in classified.values())
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Artifacts:** {len(classified)}\n")
        f.write(f"- **ACTIVE:** {active_count}\n")
        f.write(f"- **REFERENCED:** {referenced_count}\n")
        f.write(f"- **ORPHAN:** {orphan_count}\n")
        f.write(f"- **Total Size:** {total_size / (1024**3):.2f} GB\n\n")
        
        # Canonical directories
        f.write("## Canonical Directories\n\n")
        for canonical_dir in CANONICAL_DIRS:
            f.write(f"- `{canonical_dir}`\n")
        f.write("\n")
        
        # Top duplicates
        f.write("## Top Duplicates (by hash)\n\n")
        duplicate_sizes = sorted(
            [(h, paths, sum(classified[p]["size_bytes"] for p in paths)) for h, paths in duplicates.items()],
            key=lambda x: x[2],
            reverse=True,
        )[:20]
        
        f.write("| Hash (first 16) | Paths | Total Size (MB) |\n")
        f.write("|-----------------|-------|-----------------|\n")
        for hash_val, paths, total_size_mb in duplicate_sizes:
            f.write(f"| `{hash_val[:16]}...` | {len(paths)} | {total_size_mb / (1024**2):.2f} |\n")
            for path in paths[:3]:  # Show first 3 paths
                f.write(f"  - `{path}`\n")
            if len(paths) > 3:
                f.write(f"  - ... and {len(paths) - 3} more\n")
        f.write("\n")
        
        # Top largest artifacts
        f.write("## Top Largest Artifacts\n\n")
        largest = sorted(
            classified.items(),
            key=lambda x: x[1]["size_bytes"],
            reverse=True,
        )[:20]
        
        f.write("| Path | Size (MB) | Classification |\n")
        f.write("|-----|-----------|---------------|\n")
        for artifact_path, artifact_info in largest:
            f.write(f"| `{artifact_path}` | {artifact_info['size_bytes'] / (1024**2):.2f} | {artifact_info['classification']} |\n")
        f.write("\n")
        
        # ACTIVE artifacts
        f.write("## ACTIVE Artifacts\n\n")
        active = [(p, i) for p, i in classified.items() if i["classification"] == "ACTIVE"]
        f.write(f"**Count:** {len(active)}\n\n")
        f.write("| Path | Size (MB) | References |\n")
        f.write("|-----|-----------|------------|\n")
        for artifact_path, artifact_info in sorted(active, key=lambda x: x[1]["size_bytes"], reverse=True):
            ref_count = artifact_info["reference_count"]
            f.write(f"| `{artifact_path}` | {artifact_info['size_bytes'] / (1024**2):.2f} | {ref_count} |\n")
        f.write("\n")
        
        # ORPHAN artifacts
        f.write("## ORPHAN Artifacts (Candidates for Archive)\n\n")
        orphans = [(p, i) for p, i in classified.items() if i["classification"] == "ORPHAN"]
        f.write(f"**Count:** {len(orphans)}\n\n")
        f.write("| Path | Size (MB) |\n")
        f.write("|-----|-----------|\n")
        for artifact_path, artifact_info in sorted(orphans, key=lambda x: x[1]["size_bytes"], reverse=True):
            f.write(f"| `{artifact_path}` | {artifact_info['size_bytes'] / (1024**2):.2f} |\n")
        f.write("\n")
        
        # Risky directories
        f.write("## Risky Directories (Excluded from Backup)\n\n")
        f.write("These directories are excluded from artifact scanning and should be in `.gitignore`:\n\n")
        for exclude_dir in EXCLUDE_DIRS:
            f.write(f"- `{exclude_dir}`\n")
        f.write("\n")
    
    log.info(f"[AUDIT] ✅ Reports saved: {md_path}, {json_path}")
    return md_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Audit model artifacts")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan artifacts
    artifacts = scan_artifacts(repo_root)
    
    # Find references
    references = find_references(repo_root)
    
    # Classify
    classified = classify_artifacts(artifacts, references, repo_root)
    
    # Find duplicates
    duplicates = find_duplicates(classified)
    
    # Generate reports
    md_path, json_path = generate_report(classified, duplicates, output_dir)
    
    # Generate archive script
    archive_script = generate_archive_script(classified, output_dir)
    
    log.info(f"[AUDIT] ✅ Audit complete. Reports: {md_path}, {json_path}")
    log.info(f"[AUDIT] ✅ Archive script: {archive_script}")


if __name__ == "__main__":
    main()
