#!/usr/bin/env python3
"""
Build V10 Dependency Graph

Identifies what v10 actually uses:
- Canonical entrypoints and their imports
- Runtime artifact patterns (bundle files, models, etc.)
- Required modules and files

Output: V10_DEPGRAPH.json and V10_DEPGRAPH.md
"""

import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Canonical v10 entrypoints
CANONICAL_ENTRYPOINTS = [
    "gx1/scripts/replay_eval_gated_parallel.py",
    "gx1/scripts/run_depth_ladder_eval_multiyear.py",
    "gx1/scripts/run_xgb_flow_ablation_qsmoke.py",
    "gx1/scripts/run_verify_entry_flow_gap_smoke.py",
    "gx1/scripts/verify_entry_flow_gap.py",
    "gx1/scripts/preflight_prebuilt_import_check.py",
]

# Runtime artifact patterns (files that v10 loads at runtime)
RUNTIME_ARTIFACT_PATTERNS = [
    r'bundle_metadata\.json',
    r'feature_meta\.json',
    r'.*\.pkl',
    r'.*xgb.*\.(pkl|model|json)',
    r'seq_scaler.*\.pkl',
    r'snap_scaler.*\.pkl',
    r'.*calibration.*\.(pkl|json)',
    r'entry_v10.*\.json',
    r'v10_ctx.*\.json',
]

def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file using AST."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"[WARN] Failed to parse {file_path}: {e}")
    
    return imports

def find_imported_modules(start_file: Path, visited: Optional[Set[Path]] = None) -> Set[str]:
    """Recursively find all imported modules starting from a file."""
    if visited is None:
        visited = set()
    
    if start_file in visited:
        return set()
    
    visited.add(start_file)
    
    imports = extract_imports_from_file(start_file)
    modules = set()
    
    for imp in imports:
        # Try to find the module file
        if imp.startswith('gx1.'):
            # gx1 module
            module_path = imp.replace('.', '/')
            potential_paths = [
                workspace_root / f"{module_path}.py",
                workspace_root / f"{module_path}/__init__.py",
            ]
            
            for path in potential_paths:
                if path.exists() and path not in visited:
                    modules.add(str(path.relative_to(workspace_root)))
                    modules.update(find_imported_modules(path, visited))
        elif imp == 'gx1':
            # gx1 package
            gx1_init = workspace_root / "gx1" / "__init__.py"
            if gx1_init.exists() and gx1_init not in visited:
                modules.add(str(gx1_init.relative_to(workspace_root)))
                modules.update(find_imported_modules(gx1_init, visited))
    
    return modules

def grep_runtime_artifacts(file_path: Path, patterns: List[str]) -> List[str]:
    """Grep for runtime artifact patterns in a file."""
    matches = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(content):
                matches.append(match.group(0))
    except Exception:
        pass
    
    return matches

def scan_entrypoint(entrypoint_path: Path) -> Dict[str, Any]:
    """Scan a single entrypoint for dependencies."""
    print(f"[SCAN] {entrypoint_path}")
    
    # Extract imports
    direct_imports = extract_imports_from_file(entrypoint_path)
    
    # Find imported modules (recursive)
    imported_modules = find_imported_modules(entrypoint_path)
    
    # Grep for runtime artifacts
    runtime_artifacts = grep_runtime_artifacts(entrypoint_path, RUNTIME_ARTIFACT_PATTERNS)
    
    # Grep for v10-specific patterns
    v10_patterns = [
        r'entry_v10',
        r'v10_ctx',
        r'entry_models\.v10_ctx',
        r'GX1_BUNDLE_DIR',
        r'feature_meta',
        r'seq_scaler',
        r'snap_scaler',
        r'xgb_calibration',
    ]
    v10_refs = grep_runtime_artifacts(entrypoint_path, v10_patterns)
    
    return {
        "entrypoint": str(entrypoint_path.relative_to(workspace_root)),
        "direct_imports": sorted(list(direct_imports)),
        "imported_modules": sorted(list(imported_modules)),
        "runtime_artifacts": sorted(list(set(runtime_artifacts))),
        "v10_references": sorted(list(set(v10_refs))),
    }

def find_runtime_artifact_files(base_dir: Path, patterns: List[str]) -> List[Path]:
    """Find actual files matching runtime artifact patterns."""
    found = []
    
    for pattern in patterns:
        # Convert pattern to glob
        if '.*' in pattern:
            # Flexible pattern - try common extensions
            base_pattern = pattern.replace('.*', '')
            for ext in ['', '.json', '.pkl', '.model']:
                for path in base_dir.rglob(f"*{base_pattern}*{ext}"):
                    if path.is_file():
                        found.append(path)
        else:
            # Exact pattern
            for path in base_dir.rglob(pattern):
                if path.is_file():
                    found.append(path)
    
    return sorted(set(found))

def main():
    """Main entry point."""
    print("=" * 80)
    print("V10 Dependency Graph Builder")
    print("=" * 80)
    print()
    
    all_modules = set()
    all_runtime_artifacts = set()
    all_v10_refs = set()
    entrypoint_scans = []
    
    # Scan each entrypoint
    for entrypoint_rel in CANONICAL_ENTRYPOINTS:
        entrypoint_path = workspace_root / entrypoint_rel
        if not entrypoint_path.exists():
            print(f"[WARN] Entrypoint not found: {entrypoint_path}")
            continue
        
        scan_result = scan_entrypoint(entrypoint_path)
        entrypoint_scans.append(scan_result)
        
        all_modules.update(scan_result["imported_modules"])
        all_runtime_artifacts.update(scan_result["runtime_artifacts"])
        all_v10_refs.update(scan_result["v10_references"])
    
    # Find actual runtime artifact files
    print()
    print("[SCAN] Finding runtime artifact files...")
    runtime_files = find_runtime_artifact_files(workspace_root / "gx1", RUNTIME_ARTIFACT_PATTERNS)
    runtime_files_rel = [str(f.relative_to(workspace_root)) for f in runtime_files]
    
    # Build dependency graph
    depgraph = {
        "canonical_entrypoints": CANONICAL_ENTRYPOINTS,
        "entrypoint_scans": entrypoint_scans,
        "all_imported_modules": sorted(list(all_modules)),
        "runtime_artifact_patterns": RUNTIME_ARTIFACT_PATTERNS,
        "runtime_artifact_references": sorted(list(all_runtime_artifacts)),
        "runtime_artifact_files": sorted(runtime_files_rel),
        "v10_references": sorted(list(all_v10_refs)),
        "required_patterns": {
            "bundle_metadata": "bundle_metadata.json",
            "feature_meta": "feature_meta.json",
            "scalers": ["seq_scaler", "snap_scaler"],
            "xgb_models": ["xgb", "calibration"],
            "v10_configs": ["entry_v10", "v10_ctx"],
        },
    }
    
    # Write JSON
    output_dir = workspace_root / "reports" / "repo_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "V10_DEPGRAPH.json"
    with open(json_path, 'w') as f:
        json.dump(depgraph, f, indent=2)
    print(f"✅ Written: {json_path}")
    
    # Write Markdown summary
    md_path = output_dir / "V10_DEPGRAPH.md"
    with open(md_path, 'w') as f:
        f.write("# V10 Dependency Graph\n\n")
        f.write("## Canonical Entrypoints\n\n")
        for ep in CANONICAL_ENTRYPOINTS:
            f.write(f"- `{ep}`\n")
        f.write("\n")
        
        f.write("## Entrypoint Scans\n\n")
        for scan in entrypoint_scans:
            f.write(f"### {scan['entrypoint']}\n\n")
            f.write(f"- **Direct Imports:** {len(scan['direct_imports'])} modules\n")
            f.write(f"- **Imported Modules:** {len(scan['imported_modules'])} files\n")
            f.write(f"- **Runtime Artifacts:** {len(scan['runtime_artifacts'])} patterns\n")
            f.write(f"- **V10 References:** {len(scan['v10_references'])} patterns\n")
            f.write("\n")
        
        f.write("## All Imported Modules\n\n")
        f.write(f"Total: {len(all_modules)} modules\n\n")
        for mod in sorted(all_modules)[:50]:
            f.write(f"- `{mod}`\n")
        if len(all_modules) > 50:
            f.write(f"\n*... and {len(all_modules) - 50} more*\n")
        
        f.write("\n## Runtime Artifact Patterns\n\n")
        for pattern in RUNTIME_ARTIFACT_PATTERNS:
            f.write(f"- `{pattern}`\n")
        
        f.write("\n## Runtime Artifact Files Found\n\n")
        f.write(f"Total: {len(runtime_files_rel)} files\n\n")
        for file in sorted(runtime_files_rel)[:50]:
            f.write(f"- `{file}`\n")
        if len(runtime_files_rel) > 50:
            f.write(f"\n*... and {len(runtime_files_rel) - 50} more*\n")
        
        f.write("\n## V10 References\n\n")
        for ref in sorted(all_v10_refs):
            f.write(f"- `{ref}`\n")
    
    print(f"✅ Written: {md_path}")
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Entrypoints scanned: {len(entrypoint_scans)}")
    print(f"Total imported modules: {len(all_modules)}")
    print(f"Runtime artifact files: {len(runtime_files_rel)}")
    print(f"V10 references: {len(all_v10_refs)}")

if __name__ == "__main__":
    main()
