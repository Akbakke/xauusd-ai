#!/usr/bin/env python3
"""
Resolve UNKNOWN artifacts to KEEP/ARCHIVE/DELETE_LATER/NEEDS_DECISION.

Usage:
    python -m gx1.scripts.resolve_unknown_artifacts \
        --inventory reports/cleanup/REPO_REAL_OVERVIEW_20260107_223110.json \
        --out_dir reports/cleanup
"""

import argparse
import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Canonical policy configs (currently used)
CANONICAL_POLICIES = [
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml",
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml",
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml",
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY.yaml",
]

# Entrypoint patterns
ENTRYPOINT_PATTERNS = [
    "run_",
    "main",
    "entry",
    "live",
    "replay",
    "train",
    "eval",
]


def load_inventory(inventory_path: Path) -> Dict:
    """Load inventory JSON."""
    with open(inventory_path) as f:
        return json.load(f)


def refsweep(artifact_path: str, repo_root: Path) -> Tuple[List[str], str]:
    """Search for references using ripgrep. Returns (references, verdict)."""
    artifact_name = Path(artifact_path).name
    artifact_basename = Path(artifact_path).stem
    artifact_parent = str(Path(artifact_path).parent)
    
    references = []
    
    # Search patterns
    search_patterns = [
        artifact_path,
        artifact_name,
        artifact_basename,
        Path(artifact_path).parent.name,  # parent dir name
    ]
    
    for pattern in search_patterns:
        if not pattern or pattern == ".":
            continue
        
        try:
            # Use ripgrep if available, fallback to grep
            try:
                result = subprocess.run(
                    ["rg", "-n", "--hidden", "--no-ignore-vcs", "-S", pattern,
                     str(repo_root / "gx1"),
                     str(repo_root / "configs"),
                     str(repo_root / "scripts"),
                     str(repo_root / "docs"),
                     str(repo_root / "models"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=repo_root,
                )
                output = result.stdout
            except FileNotFoundError:
                # Fallback to grep
                result = subprocess.run(
                    ["grep", "-rn", "--include=*.py", "--include=*.yaml", "--include=*.yml",
                     "--include=*.md", "--include=*.sh", pattern,
                     str(repo_root / "gx1"),
                     str(repo_root / "configs"),
                     str(repo_root / "scripts"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=repo_root,
                )
                output = result.stdout
            
            for line in output.strip().split("\n"):
                if line and ":" in line:
                    file_path = line.split(":")[0]
                    if file_path and file_path not in references:
                        # Skip if it's the artifact itself
                        if artifact_path not in file_path:
                            references.append(file_path)
        except Exception as e:
            log.debug(f"Failed to search pattern {pattern}: {e}")
    
    # Classify references
    canonical_refs = [r for r in references if "policies" in r and "2025_SNIPER_V1" in r]
    entrypoint_refs = [r for r in references if any(p in r for p in ENTRYPOINT_PATTERNS)]
    report_refs = [r for r in references if "reports" in r or "cleanup" in r]
    doc_refs = [r for r in references if "docs" in r and r not in entrypoint_refs]
    
    # Verdict
    if canonical_refs or entrypoint_refs:
        verdict = "REFERENCED_CANONICAL"
    elif report_refs or doc_refs:
        verdict = "REFERENCED_OLD_DOCS_ONLY"
    elif references:
        verdict = "REFERENCED_UNKNOWN"
    else:
        verdict = "NO_REFS"
    
    return references, verdict


def classify_unknown(artifact_path: str, artifact_info: Dict, repo_root: Path) -> Dict:
    """Classify UNKNOWN artifact into KEEP/ARCHIVE/DELETE_LATER/NEEDS_DECISION."""
    path = artifact_path
    name = Path(path).name
    parent = Path(path).parent.name
    inferred_type = artifact_info.get("inferred_type", "unknown")
    in_canonical = artifact_info.get("in_canonical_dir", False)
    
    # Search for references
    references, ref_verdict = refsweep(path, repo_root)
    
    classification = {
        "path": path,
        "inferred_type": inferred_type,
        "in_canonical_dir": in_canonical,
        "references": references,
        "ref_verdict": ref_verdict,
        "size_mb": artifact_info.get("size_bytes", 0) / (1024**2),
    }
    
    # Classification logic
    if "checkpoint" in path.lower() or "checkpoint_epoch" in name:
        # Checkpoint file
        if ref_verdict == "REFERENCED_CANONICAL":
            classification["verdict"] = "KEEP"
            classification["reason"] = "Referenced in canonical policies/configs"
        else:
            classification["verdict"] = "NEEDS_DECISION"
            classification["reason"] = "Checkpoint file. Decide: resume training? If no resume, archive. If resume, keep latest 1-2."
            classification["recommended_action"] = "Archive if no resume training. Keep only latest 2 if resume."
    
    elif "calibrator" in path.lower() or "calibration" in path.lower():
        # Calibrator
        if in_canonical:
            classification["verdict"] = "KEEP"
            classification["reason"] = "Calibrator in canonical dir. Keep until runtime usage stats prove hierarchy usage."
            classification["recommended_action"] = "Log calibrator_usage_stats in runtime. Verify session+bucket → session-only hierarchy."
        else:
            classification["verdict"] = "ARCHIVE"
            classification["reason"] = "Calibrator outside canonical dirs"
    
    elif "SMOKE" in path or "BASELINE_NO_GATE" in path or "BASELINE" in path:
        # SMOKE or BASELINE variant
        if ref_verdict == "REFERENCED_CANONICAL":
            classification["verdict"] = "KEEP"
            classification["reason"] = "Referenced in canonical policies"
        elif ref_verdict == "NO_REFS":
            classification["verdict"] = "ARCHIVE"
            classification["reason"] = "Not referenced in any runnable policy or eval pipeline"
        else:
            classification["verdict"] = "NEEDS_DECISION"
            classification["reason"] = "May be used for comparison in eval. Verify eval pipelines."
            classification["recommended_action"] = "Check if used in eval_gated_fusion_offline.py or similar. Archive if not."
    
    elif "bundle_metadata" in path or "feature_contract_hash" in path:
        # Metadata files
        if ref_verdict == "REFERENCED_CANONICAL":
            classification["verdict"] = "KEEP"
            classification["reason"] = "Bundle metadata referenced in canonical policies"
        else:
            # Check if bundle dir is referenced
            bundle_dir = str(Path(path).parent)
            bundle_refs, _ = refsweep(bundle_dir, repo_root)
            if any("policies" in r and "2025_SNIPER_V1" in r for r in bundle_refs):
                classification["verdict"] = "KEEP"
                classification["reason"] = "Bundle dir referenced in canonical policies"
            else:
                classification["verdict"] = "ARCHIVE"
                classification["reason"] = "Metadata for unreferenced bundle"
    
    elif "model.pt" in name or "model_state_dict.pt" in name:
        # Model files
        if ref_verdict == "REFERENCED_CANONICAL":
            classification["verdict"] = "KEEP"
            classification["reason"] = "Model referenced in canonical policies"
        elif in_canonical:
            classification["verdict"] = "NEEDS_DECISION"
            classification["reason"] = "Model in canonical dir but not referenced. May be legacy training run."
            classification["recommended_action"] = "Archive if not needed for training resume or eval comparisons."
        else:
            classification["verdict"] = "ARCHIVE"
            classification["reason"] = "Model outside canonical dirs"
    
    elif "scaler" in path.lower():
        # Scaler files
        if ref_verdict == "REFERENCED_CANONICAL":
            classification["verdict"] = "KEEP"
            classification["reason"] = "Scaler referenced in canonical policies"
        elif in_canonical:
            # Check if parent model dir is referenced
            model_dir = str(Path(path).parent)
            model_refs, _ = refsweep(model_dir, repo_root)
            if any("policies" in r and "2025_SNIPER_V1" in r for r in model_refs):
                classification["verdict"] = "KEEP"
                classification["reason"] = "Parent model dir referenced in canonical policies"
            else:
                classification["verdict"] = "ARCHIVE"
                classification["reason"] = "Scaler in canonical dir but parent not referenced"
        else:
            classification["verdict"] = "ARCHIVE"
            classification["reason"] = "Scaler outside canonical dirs"
    
    else:
        # Unknown type
        if ref_verdict == "REFERENCED_CANONICAL":
            classification["verdict"] = "KEEP"
            classification["reason"] = "Referenced in canonical policies/configs"
        elif ref_verdict == "NO_REFS" and not in_canonical:
            classification["verdict"] = "ARCHIVE"
            classification["reason"] = "No references found, outside canonical dirs"
        elif ref_verdict == "NO_REFS" and in_canonical:
            classification["verdict"] = "NEEDS_DECISION"
            classification["reason"] = "No references but in canonical dir. May be needed for training/eval."
            classification["recommended_action"] = "Manual review required"
        else:
            classification["verdict"] = "NEEDS_DECISION"
            classification["reason"] = f"References found but unclear if canonical ({ref_verdict})"
            classification["recommended_action"] = "Review references and decide"
    
    return classification


def generate_unknown_verdict_report(
    classifications: List[Dict],
    repo_root: Path,
    output_dir: Path,
) -> Path:
    """Generate UNKNOWN verdict report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"UNKNOWN_VERDICT_{timestamp}.md"
    
    # Group by verdict
    by_verdict = defaultdict(list)
    for cls in classifications:
        by_verdict[cls["verdict"]].append(cls)
    
    with open(report_path, "w") as f:
        f.write("# UNKNOWN Artifacts Verdict Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Verdict | Count | Action |\n")
        f.write("|---------|-------|--------|\n")
        f.write(f"| KEEP | {len(by_verdict['KEEP'])} | Do not archive |\n")
        f.write(f"| ARCHIVE | {len(by_verdict['ARCHIVE'])} | Archive now |\n")
        f.write(f"| DELETE LATER | {len(by_verdict['DELETE LATER'])} | Archive now, delete after 7 days |\n")
        f.write(f"| NEEDS DECISION | {len(by_verdict['NEEDS_DECISION'])} | Manual review required |\n")
        f.write("\n")
        
        # KEEP
        if by_verdict["KEEP"]:
            f.write("## KEEP\n\n")
            f.write("**Rationale:** Referenced in canonical policies/configs or essential for runtime.\n\n")
            f.write("| Path | Reason | References |\n")
            f.write("|------|--------|------------|\n")
            for cls in sorted(by_verdict["KEEP"], key=lambda x: x["path"]):
                refs_str = ", ".join([Path(r).name for r in cls["references"][:2]])
                if len(cls["references"]) > 2:
                    refs_str += f" ... (+{len(cls['references']) - 2} more)"
                f.write(f"| `{cls['path']}` | {cls['reason']} | {refs_str} |\n")
            f.write("\n")
        
        # ARCHIVE
        if by_verdict["ARCHIVE"]:
            f.write("## ARCHIVE (Safe to Archive Now)\n\n")
            f.write("**Rationale:** No references in canonical policies/configs, safe to archive.\n\n")
            f.write("| Path | Reason | Size (MB) |\n")
            f.write("|------|--------|-----------|\n")
            for cls in sorted(by_verdict["ARCHIVE"], key=lambda x: x["path"]):
                f.write(f"| `{cls['path']}` | {cls['reason']} | {cls['size_mb']:.2f} |\n")
            f.write("\n")
        
        # DELETE LATER
        if by_verdict["DELETE LATER"]:
            f.write("## DELETE LATER (Archive Now, Delete After 7 Days)\n\n")
            f.write("**Rationale:** Archived, safe to delete after verification period.\n\n")
            f.write("| Path | Reason |\n")
            f.write("|------|--------|\n")
            for cls in sorted(by_verdict["DELETE LATER"], key=lambda x: x["path"]):
                f.write(f"| `{cls['path']}` | {cls['reason']} |\n")
            f.write("\n")
        
        # NEEDS DECISION
        if by_verdict["NEEDS_DECISION"]:
            f.write("## NEEDS DECISION (Manual Review Required)\n\n")
            f.write("| Path | Reason | Recommended Action | References |\n")
            f.write("|------|--------|-------------------|------------|\n")
            for cls in sorted(by_verdict["NEEDS_DECISION"], key=lambda x: x["path"]):
                refs_str = ", ".join([Path(r).name for r in cls["references"][:2]]) if cls["references"] else "None"
                action = cls.get("recommended_action", "Manual review")
                f.write(f"| `{cls['path']}` | {cls['reason']} | {action} | {refs_str} |\n")
            f.write("\n")
        
        # Next steps
        f.write("## Next Steps\n\n")
        f.write("1. Review NEEDS_DECISION items and make decisions\n")
        f.write("2. Execute archive script for ARCHIVE items\n")
        f.write("3. Verify runtime after archiving\n")
        f.write("4. Wait 7 days, then delete DELETE LATER items from archive\n\n")
    
    return report_path


def generate_unknown_archive_script(
    archive_items: List[Dict],
    repo_root: Path,
    output_dir: Path,
) -> Path:
    """Generate archive script for UNKNOWN items marked as ARCHIVE."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scripts_cleanup_dir = repo_root / "scripts" / "cleanup"
    scripts_cleanup_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_cleanup_dir / f"ARCHIVE_UNKNOWN_{timestamp}.sh"
    
    archive_dir = f"_archive_artifacts/{timestamp}"
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Archive UNKNOWN items marked as ARCHIVE\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total items: {len(archive_items)}\n")
        f.write("# Usage: ./script.sh [--dry-run]\n\n")
        
        f.write("# Parse --dry-run flag\n")
        f.write("DRY_RUN=false\n")
        f.write("if [ \"$1\" = \"--dry-run\" ]; then\n")
        f.write("  DRY_RUN=true\n")
        f.write("  echo '=== DRY RUN MODE ==='\n")
        f.write("  echo 'No files will be moved'\n")
        f.write("  echo ''\n")
        f.write("fi\n\n")
        
        f.write(f"ARCHIVE_DIR=\"{archive_dir}\"\n")
        f.write("if [ \"$DRY_RUN\" = \"false\" ]; then\n")
        f.write("  mkdir -p \"$ARCHIVE_DIR\"\n")
        f.write("fi\n\n")
        
        f.write("TOTAL_MOVED=0\n")
        f.write("TOTAL_SIZE=0\n\n")
        
        f.write("echo \"Archiving UNKNOWN items marked as ARCHIVE...\"\n\n")
        
        for item in sorted(archive_items, key=lambda x: x["path"]):
            path = item["path"]
            size = int(item["size_mb"] * 1024 * 1024)
            reason = item["reason"]
            
            f.write(f"# {path}\n")
            f.write(f"# Reason: {reason}\n")
            if "DRY_RUN" in locals():
                f.write(f"if [ \"$DRY_RUN\" = \"true\" ]; then\n")
                f.write(f"  echo \"[DRY RUN] Would archive: {path} ({item['size_mb']:.2f} MB)\"\n")
                f.write(f"else\n")
                f.write(f"  mkdir -p \"$ARCHIVE_DIR/$(dirname '{path}')\"\n")
                f.write(f"  if [ -f '{path}' ] || [ -d '{path}' ]; then\n")
                f.write(f"    mv '{path}' \"$ARCHIVE_DIR/{path}\"\n")
                f.write(f"    echo \"Archived: {path}\"\n")
                f.write(f"    TOTAL_MOVED=$((TOTAL_MOVED + 1))\n")
                f.write(f"    TOTAL_SIZE=$((TOTAL_SIZE + {size}))\n")
                f.write(f"  else\n")
                f.write(f"    echo \"WARNING: {path} not found, skipping\"\n")
                f.write(f"  fi\n")
                f.write(f"fi\n")
            else:
                f.write(f"mkdir -p \"$ARCHIVE_DIR/$(dirname '{path}')\"\n")
                f.write(f"if [ -f '{path}' ] || [ -d '{path}' ]; then\n")
                f.write(f"  mv '{path}' \"$ARCHIVE_DIR/{path}\"\n")
                f.write(f"  echo \"Archived: {path}\"\n")
                f.write(f"  TOTAL_MOVED=$((TOTAL_MOVED + 1))\n")
                f.write(f"  TOTAL_SIZE=$((TOTAL_SIZE + {size}))\n")
                f.write(f"else\n")
                f.write(f"  echo \"WARNING: {path} not found, skipping\"\n")
                f.write(f"fi\n")
            f.write("\n")
        
        f.write("echo \"\"\n")
        f.write("if [ \"$DRY_RUN\" = \"true\" ]; then\n")
        f.write("  echo \"=== DRY RUN SUMMARY ===\"\n")
        f.write("  echo \"Would move: $TOTAL_MOVED files\"\n")
        f.write("  echo \"Total size: $(echo \"scale=2; $TOTAL_SIZE / (1024*1024)\" | bc) MB\"\n")
        f.write("else\n")
        f.write("  echo \"=== Archive Summary ===\"\n")
        f.write("  echo \"Total files moved: $TOTAL_MOVED\"\n")
        f.write("  echo \"Total size: $(echo \"scale=2; $TOTAL_SIZE / (1024*1024)\" | bc) MB\"\n")
        f.write("  echo \"Archive location: $ARCHIVE_DIR\"\n")
        f.write("  echo \"Review $ARCHIVE_DIR before deleting.\"\n")
        f.write("fi\n")
    
    script_path.chmod(0o755)
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Resolve UNKNOWN artifacts")
    parser.add_argument("--inventory", type=str, required=True, help="Path to inventory JSON")
    parser.add_argument("--out_dir", type=str, default="reports/cleanup", help="Output directory")
    parser.add_argument("--repo_root", type=str, default=".", help="Repository root")
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).resolve()
    inventory_path = Path(args.inventory)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Loading inventory from {inventory_path}")
    inventory = load_inventory(inventory_path)
    
    # Get UNKNOWN artifacts from execution plan classification
    # We'll extract from the inventory directly
    artifacts = inventory.get("classified_artifacts", {})
    unknown_items = []
    
    for artifact_path, artifact_info in artifacts.items():
        classification = artifact_info.get("classification", "UNKNOWN")
        if classification == "UNKNOWN":
            unknown_items.append((artifact_path, artifact_info))
    
    log.info(f"Found {len(unknown_items)} UNKNOWN artifacts")
    
    log.info("Classifying UNKNOWN artifacts...")
    classifications = []
    for artifact_path, artifact_info in unknown_items:
        log.info(f"Processing: {artifact_path}")
        classification = classify_unknown(artifact_path, artifact_info, repo_root)
        classifications.append(classification)
    
    # Generate report
    log.info("Generating verdict report...")
    report_path = generate_unknown_verdict_report(classifications, repo_root, output_dir)
    
    # Generate archive script for ARCHIVE items
    archive_items = [c for c in classifications if c["verdict"] == "ARCHIVE"]
    if archive_items:
        log.info(f"Generating archive script for {len(archive_items)} ARCHIVE items...")
        archive_script = generate_unknown_archive_script(archive_items, repo_root, output_dir)
    else:
        archive_script = None
    
    # Summary
    by_verdict = defaultdict(int)
    for cls in classifications:
        by_verdict[cls["verdict"]] += 1
    
    log.info("=" * 80)
    log.info("✅ UNKNOWN RESOLUTION COMPLETE")
    log.info("=" * 80)
    log.info(f"Total UNKNOWN items: {len(classifications)}")
    log.info(f"  KEEP: {by_verdict['KEEP']}")
    log.info(f"  ARCHIVE: {by_verdict['ARCHIVE']}")
    log.info(f"  DELETE LATER: {by_verdict['DELETE LATER']}")
    log.info(f"  NEEDS DECISION: {by_verdict['NEEDS_DECISION']}")
    log.info(f"Report: {report_path}")
    if archive_script:
        log.info(f"Archive script: {archive_script}")
        log.info(f"  Usage: {archive_script.name} [--dry-run]")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
