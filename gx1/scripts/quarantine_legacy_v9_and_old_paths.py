#!/usr/bin/env python3
"""
Quarantine Legacy V9 and Old Paths

Scans repository, classifies legacy files, and moves them to quarantine.
Reversible operation - all files moved with manifest.

Usage:
    python3 gx1/scripts/quarantine_legacy_v9_and_old_paths.py [--dry-run]
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Load V10 dependency graph
V10_DEPGRAPH_PATH = workspace_root / "reports" / "repo_audit" / "V10_DEPGRAPH.json"

# Directories to exclude from scanning
EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    '.pytest_cache',
    'node_modules',
    '_archive_artifacts',
    '_archive_v9',
    'docs/legacy',
    'tools/legacy',
    'reports',  # Historical reports
    '_quarantine_',  # Quarantine dirs
}

# Legacy patterns to detect
LEGACY_PATTERNS = [
    (r'\bentry_v9\b', 'ENTRY_V9_REFERENCE', 'BLOCKER'),
    (r'\bv9\b', 'V9_REFERENCE', 'WARN'),  # May be in comments/docs
    (r'sniper/NY', 'SNIPER_NY_LEGACY', 'BLOCKER'),
    (r'V9_FARM', 'V9_FARM_REFERENCE', 'BLOCKER'),
    (r'OANDA_DEMO_V9', 'OANDA_DEMO_V9', 'BLOCKER'),
    (r'trial.*_v9', 'TRIAL_V9_REFERENCE', 'BLOCKER'),
    (r'entry_models\.v9', 'POLICY_ENTRY_V9', 'BLOCKER'),
    (r'reports/replay_eval', 'REPORTS_UNDER_ENGINE', 'BLOCKER'),
    (r'models/models/', 'DOUBLE_MODELS_PATH', 'BLOCKER'),
]

def load_v10_depgraph() -> Dict[str, Any]:
    """Load V10 dependency graph."""
    if not V10_DEPGRAPH_PATH.exists():
        print(f"[WARN] V10_DEPGRAPH.json not found: {V10_DEPGRAPH_PATH}")
        print("[WARN] Run: python3 gx1/scripts/build_v10_depgraph.py")
        return {
            "canonical_entrypoints": [],
            "all_imported_modules": [],
            "runtime_artifact_files": [],
        }
    
    with open(V10_DEPGRAPH_PATH, 'r') as f:
        return json.load(f)

def is_excluded(path: Path) -> bool:
    """Check if path should be excluded."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith('_quarantine_'):
            return True
    return False

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""

def scan_file_for_legacy(file_path: Path) -> List[Dict[str, Any]]:
    """Scan a file for legacy patterns."""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return findings
    
    for line_no, line in enumerate(lines, 1):
        for pattern, category, severity in LEGACY_PATTERNS:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)
                context = line.strip()[:200]
                
                findings.append({
                    'file': str(file_path),
                    'line_no': line_no,
                    'matched_text': matched_text,
                    'context': context,
                    'category': category,
                    'severity': severity,
                })
    
    return findings

def check_policy_for_v9(policy_path: Path) -> bool:
    """Check if policy has entry_models.v9."""
    try:
        import yaml
        with open(policy_path, 'r') as f:
            policy_data = yaml.safe_load(f)
        
        entry_models = policy_data.get('entry_models', {})
        return 'v9' in entry_models
    except Exception:
        return False

def classify_file(
    file_path: Path,
    depgraph: Dict[str, Any],
    all_findings: List[Dict[str, Any]],
) -> Tuple[str, str, List[str]]:
    """
    Classify a file: KEEP, SAFE_TO_QUARANTINE, or BLOCKER.
    
    Returns: (classification, reason, evidence)
    """
    file_rel = str(file_path.relative_to(workspace_root))
    
    # Check if in V10 dependency graph
    canonical_entrypoints = depgraph.get("canonical_entrypoints", [])
    all_imported_modules = depgraph.get("all_imported_modules", [])
    runtime_artifact_files = depgraph.get("runtime_artifact_files", [])
    
    # KEEP: Canonical entrypoints
    if file_rel in canonical_entrypoints:
        return ("KEEP", "Canonical v10 entrypoint", [])
    
    # KEEP: Imported modules
    if file_rel in all_imported_modules:
        return ("KEEP", "Imported by v10 entrypoint", [])
    
    # KEEP: Runtime artifacts
    if file_rel in runtime_artifact_files:
        return ("KEEP", "Runtime artifact used by v10", [])
    
    # KEEP: V10-specific patterns (conservative - don't quarantine v10 files)
    v10_keep_patterns = [
        'v10_ctx',
        'entry_v10',
        'entry_policy_sniper_v10',
        'runtime_v10',
        'entry_v10_bundle',
        'entry_v10_ctx',
        'verify_entry_flow_gap',
        'legacy_guard',  # Keep legacy guard itself
    ]
    for pattern in v10_keep_patterns:
        if pattern in file_rel.lower():
            return ("KEEP", f"V10 pattern detected: {pattern}", [])
    
    # Check for legacy patterns in this file
    file_findings = [f for f in all_findings if f['file'] == str(file_path)]
    
    if not file_findings:
        # No legacy patterns found - keep it
        return ("KEEP", "No legacy patterns detected", [])
    
    # Check severity
    blockers = [f for f in file_findings if f['severity'] == 'BLOCKER']
    warnings = [f for f in file_findings if f['severity'] == 'WARN']
    
    # Check if it's a policy with entry_models.v9
    if file_path.suffix in ['.yaml', '.yml', '.json']:
        if check_policy_for_v9(file_path):
            return ("BLOCKER", "Policy has entry_models.v9", [f['category'] for f in blockers])
    
    # Check if it's a script with hardcoded reports/ path
    if file_path.suffix in ['.py', '.sh']:
        for finding in file_findings:
            if finding['category'] == 'REPORTS_UNDER_ENGINE':
                return ("BLOCKER", "Hardcoded reports/ path under engine", [finding['category']])
    
    # If has blockers, classify as BLOCKER
    if blockers:
        return ("BLOCKER", "Legacy patterns detected", [f['category'] for f in blockers])
    
    # If only warnings, classify as SAFE_TO_QUARANTINE
    if warnings:
        return ("SAFE_TO_QUARANTINE", "Legacy references in comments/docs", [f['category'] for f in warnings])
    
    return ("KEEP", "No actionable legacy patterns", [])

def scan_repository(depgraph: Dict[str, Any]) -> Dict[str, Any]:
    """Scan repository and classify files."""
    print("[SCAN] Scanning repository...")
    
    all_findings = []
    files_to_classify = []
    
    # Scan Python files
    for py_file in workspace_root.rglob('*.py'):
        if is_excluded(py_file):
            continue
        findings = scan_file_for_legacy(py_file)
        all_findings.extend(findings)
        if findings:
            files_to_classify.append(py_file)
    
    # Scan shell scripts
    for sh_file in workspace_root.rglob('*.sh'):
        if is_excluded(sh_file):
            continue
        findings = scan_file_for_legacy(sh_file)
        all_findings.extend(findings)
        if findings:
            files_to_classify.append(sh_file)
    
    # Scan YAML/JSON files (policies)
    for config_file in workspace_root.rglob('*.yaml'):
        if is_excluded(config_file):
            continue
        findings = scan_file_for_legacy(config_file)
        all_findings.extend(findings)
        if config_file not in files_to_classify:
            files_to_classify.append(config_file)
    
    for config_file in workspace_root.rglob('*.yml'):
        if is_excluded(config_file):
            continue
        findings = scan_file_for_legacy(config_file)
        all_findings.extend(findings)
        if config_file not in files_to_classify:
            files_to_classify.append(config_file)
    
    for json_file in workspace_root.rglob('*.json'):
        if is_excluded(json_file):
            continue
        if 'policy' in json_file.name.lower() or 'policies' in str(json_file.parent):
            findings = scan_file_for_legacy(json_file)
            all_findings.extend(findings)
            if json_file not in files_to_classify:
                files_to_classify.append(json_file)
    
    # Classify files
    print("[CLASSIFY] Classifying files...")
    classifications = {}
    
    for file_path in files_to_classify:
        classification, reason, evidence = classify_file(file_path, depgraph, all_findings)
        classifications[str(file_path)] = {
            "classification": classification,
            "reason": reason,
            "evidence": evidence,
        }
    
    # Group by classification
    by_classification = defaultdict(list)
    for file_path_str, info in classifications.items():
        by_classification[info["classification"]].append({
            "file": file_path_str,
            "reason": info["reason"],
            "evidence": info["evidence"],
        })
    
    return {
        "all_findings": all_findings,
        "classifications": classifications,
        "by_classification": dict(by_classification),
        "summary": {
            "total_scanned": len(files_to_classify),
            "keep": len(by_classification.get("KEEP", [])),
            "safe_to_quarantine": len(by_classification.get("SAFE_TO_QUARANTINE", [])),
            "blocker": len(by_classification.get("BLOCKER", [])),
        },
    }

def quarantine_files(
    scan_results: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Move files to quarantine."""
    if dry_run:
        print("[DRY_RUN] Would quarantine files (not actually moving)")
    else:
        print("[QUARANTINE] Moving files to quarantine...")
    
    # Create quarantine directory
    quarantine_dir = workspace_root / f"_quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        print(f"[QUARANTINE] Created: {quarantine_dir}")
    
    manifest = []
    
    # Get files to quarantine
    to_quarantine = []
    to_quarantine.extend(scan_results["by_classification"].get("SAFE_TO_QUARANTINE", []))
    to_quarantine.extend(scan_results["by_classification"].get("BLOCKER", []))
    
    for item in to_quarantine:
        file_path = Path(item["file"])
        
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue
        
        # Compute hash
        file_hash = compute_file_hash(file_path)
        
        # Determine quarantine path (preserve structure)
        rel_path = file_path.relative_to(workspace_root)
        quarantine_path = quarantine_dir / rel_path
        
        manifest_entry = {
            "original_path": str(rel_path),
            "new_path": str(quarantine_path) if not dry_run else f"[DRY_RUN]{quarantine_path}",
            "sha256": file_hash,
            "reason": item["reason"],
            "evidence": item["evidence"],
            "classification": "SAFE_TO_QUARANTINE" if item in scan_results["by_classification"].get("SAFE_TO_QUARANTINE", []) else "BLOCKER",
        }
        manifest.append(manifest_entry)
        
        if dry_run:
            print(f"[DRY_RUN] Would move: {rel_path} -> {quarantine_path}")
        else:
            # Create parent directory
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(file_path), str(quarantine_path))
            print(f"[QUARANTINE] Moved: {rel_path}")
    
    return {
        "quarantine_dir": str(quarantine_dir),
        "manifest": manifest,
        "dry_run": dry_run,
    }

def write_reports(scan_results: Dict[str, Any], quarantine_results: Dict[str, Any]) -> None:
    """Write reports."""
    output_dir = workspace_root / "reports" / "repo_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write candidates JSON
    candidates_path = output_dir / "LEGACY_QUARANTINE_CANDIDATES.json"
    with open(candidates_path, 'w') as f:
        json.dump(scan_results, f, indent=2)
    print(f"✅ Written: {candidates_path}")
    
    # Write candidates MD
    candidates_md_path = output_dir / "LEGACY_QUARANTINE_CANDIDATES.md"
    with open(candidates_md_path, 'w') as f:
        f.write("# Legacy Quarantine Candidates\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        summary = scan_results["summary"]
        f.write("## Summary\n\n")
        f.write(f"- **Total Scanned**: {summary['total_scanned']}\n")
        f.write(f"- **KEEP**: {summary['keep']}\n")
        f.write(f"- **SAFE_TO_QUARANTINE**: {summary['safe_to_quarantine']}\n")
        f.write(f"- **BLOCKER**: {summary['blocker']}\n\n")
        
        f.write("## By Classification\n\n")
        for classification in ["BLOCKER", "SAFE_TO_QUARANTINE", "KEEP"]:
            items = scan_results["by_classification"].get(classification, [])
            if items:
                f.write(f"### {classification} ({len(items)} files)\n\n")
                for item in items[:50]:
                    f.write(f"- `{item['file']}` - {item['reason']}\n")
                    if item['evidence']:
                        f.write(f"  - Evidence: {', '.join(item['evidence'])}\n")
                if len(items) > 50:
                    f.write(f"\n*... and {len(items) - 50} more*\n\n")
    
    print(f"✅ Written: {candidates_md_path}")
    
    # Write manifest
    manifest_path = output_dir / "QUARANTINE_MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(quarantine_results, f, indent=2)
    print(f"✅ Written: {manifest_path}")

def update_gitignore(quarantine_dir: Path) -> None:
    """Update .gitignore to exclude quarantine directories."""
    gitignore_path = workspace_root / ".gitignore"
    
    if not gitignore_path.exists():
        return
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    if '_quarantine_' not in content:
        with open(gitignore_path, 'a') as f:
            f.write("\n# Quarantine directories (legacy cleanup)\n")
            f.write("_quarantine_*/\n")
        print(f"✅ Updated: {gitignore_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quarantine legacy v9 and old paths")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't actually move files)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Legacy Quarantine Tool")
    print("=" * 80)
    print()
    
    # Load V10 dependency graph
    depgraph = load_v10_depgraph()
    print(f"[LOAD] Loaded V10 dependency graph")
    print(f"  - Entrypoints: {len(depgraph.get('canonical_entrypoints', []))}")
    print(f"  - Imported modules: {len(depgraph.get('all_imported_modules', []))}")
    print(f"  - Runtime artifacts: {len(depgraph.get('runtime_artifact_files', []))}")
    print()
    
    # Scan repository
    scan_results = scan_repository(depgraph)
    
    summary = scan_results["summary"]
    print()
    print("=" * 80)
    print("Scan Results")
    print("=" * 80)
    print(f"Total scanned: {summary['total_scanned']}")
    print(f"KEEP: {summary['keep']}")
    print(f"SAFE_TO_QUARANTINE: {summary['safe_to_quarantine']}")
    print(f"BLOCKER: {summary['blocker']}")
    print()
    
    # Quarantine files
    quarantine_results = quarantine_files(scan_results, dry_run=args.dry_run)
    
    if not args.dry_run:
        # Update .gitignore
        update_gitignore(Path(quarantine_results["quarantine_dir"]))
    
    # Write reports
    write_reports(scan_results, quarantine_results)
    
    print()
    print("=" * 80)
    if args.dry_run:
        print("DRY RUN COMPLETE - No files moved")
    else:
        print(f"QUARANTINE COMPLETE - {len(quarantine_results['manifest'])} files moved")
        print(f"Quarantine directory: {quarantine_results['quarantine_dir']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
