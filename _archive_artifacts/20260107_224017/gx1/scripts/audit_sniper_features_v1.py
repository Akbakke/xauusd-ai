#!/usr/bin/env python3
"""
SNIPER Feature Audit V1

Kartlegger alle features/inputs som har med SNIPER å gjøre:
- Runtime (entry_manager, policy, overlays, regime, osv.)
- Datasett (RL, shadow, timing, counterfactual)
- Modeller (EntryCritic, TimingCritic)

Kun kartlegging og rapportering - ingen endringer.
"""

import ast
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("reports/feature_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_python_files(root_dir: Path, patterns: List[str]) -> List[Path]:
    """Find Python files matching patterns."""
    files = []
    for pattern in patterns:
        for path in root_dir.glob(pattern):
            if path.is_file() and path.suffix == ".py":
                files.append(path)
    return sorted(set(files))


def extract_features_from_code(file_path: Path) -> Set[str]:
    """Extract feature names from Python code."""
    features = set()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        log.warning(f"Could not read {file_path}: {e}")
        return features
    
    # Pattern 1: Dictionary assignments like features["key"] = ...
    pattern1 = r'["\']([^"\']+)["\']\s*[:=]'
    matches1 = re.findall(pattern1, content)
    features.update(matches1)
    
    # Pattern 2: Dictionary access like ctx["key"] or features["key"]
    pattern2 = r'\[["\']([^"\']+)["\']\]'
    matches2 = re.findall(pattern2, content)
    features.update(matches2)
    
    # Pattern 3: .get("key") or .get('key')
    pattern3 = r'\.get\(["\']([^"\']+)["\']\)'
    matches3 = re.findall(pattern3, content)
    features.update(matches3)
    
    # Pattern 4: Feature context assignments (entry_manager.py style)
    pattern4 = r'feature_context\[["\']([^"\']+)["\']\]'
    matches4 = re.findall(pattern4, content)
    features.update(matches4)
    
    # Pattern 5: Policy state assignments
    pattern5 = r'policy_state\[["\']([^"\']+)["\']\]'
    matches5 = re.findall(pattern5, content)
    features.update(matches5)
    
    # Pattern 6: Trade extra assignments
    pattern6 = r'\.extra\[["\']([^"\']+)["\']\]'
    matches6 = re.findall(pattern6, content)
    features.update(matches6)
    
    # Filter out common non-feature keys
    exclude = {
        "self", "type", "id", "name", "value", "key", "keys", "items", "values",
        "get", "set", "update", "pop", "clear", "copy", "len", "str", "int",
        "float", "bool", "list", "dict", "tuple", "None", "True", "False",
        "if", "else", "elif", "for", "while", "def", "class", "import", "from",
        "return", "pass", "break", "continue", "try", "except", "finally",
        "raise", "assert", "with", "as", "in", "is", "not", "and", "or",
        "print", "log", "logger", "logging", "sys", "os", "path", "Path",
    }
    
    # Filter features - more aggressive filtering
    filtered = set()
    for feat in features:
        # Skip if too short or contains special chars
        if len(feat) < 2:
            continue
        if feat in exclude:
            continue
        if feat.startswith("_") or feat.startswith("."):
            continue
        if feat.isdigit():
            continue
        # Skip if contains operators or special formatting
        if any(c in feat for c in "()[]{}%+*/=<>!@#$^&|\\"):
            continue
        # Skip common string formatting patterns
        if feat in ["%s", "%d", "%f", "%r", " + ", ", ", ".json", ".parquet", ".yaml"]:
            continue
        # Skip if looks like a format string
        if "%" in feat or "{" in feat or "}" in feat:
            continue
        # Must contain at least one letter
        if not any(c.isalpha() for c in feat):
            continue
        filtered.add(feat)
    
    return filtered


def analyze_runtime_features() -> Set[str]:
    """Analyze runtime code to extract feature names."""
    log.info("Analyzing runtime code for features...")
    
    root_dir = Path("gx1")
    runtime_features = set()
    
    # Files to analyze
    patterns = [
        "sniper/**/*.py",
        "execution/entry_manager.py",
        "execution/**/*.py",
        "features/**/*.py",
        "prod/run_header.py",
        "prod/path_resolver.py",
        "tools/verify_freeze.py",
    ]
    
    files = find_python_files(root_dir, patterns)
    
    # Also check config files
    config_dir = Path("gx1/configs/policies/sniper_snapshot")
    if config_dir.exists():
        for yaml_file in config_dir.rglob("*.yaml"):
            files.append(yaml_file)
    
    log.info(f"Found {len(files)} files to analyze")
    
    for file_path in files:
        # Skip if file doesn't contain SNIPER/sniper references (for execution files)
        if "execution" in str(file_path) or "prod" in str(file_path) or "tools" in str(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    if "sniper" not in content and "entry_manager" not in str(file_path):
                        continue
            except:
                continue
        
        features = extract_features_from_code(file_path)
        runtime_features.update(features)
        if features:
            log.debug(f"  {file_path.name}: {len(features)} features")
    
    log.info(f"Extracted {len(runtime_features)} unique runtime features")
    return runtime_features


def analyze_dataset_columns(dataset_path: Path) -> List[str]:
    """Read dataset and return column names."""
    if not dataset_path.exists():
        log.warning(f"Dataset not found: {dataset_path}")
        return []
    
    try:
        df = pd.read_parquet(dataset_path)
        return sorted(df.columns.tolist())
    except Exception as e:
        log.warning(f"Could not read {dataset_path}: {e}")
        return []


def analyze_model_features(meta_path: Path) -> Dict:
    """Read model metadata and return features."""
    if not meta_path.exists():
        log.warning(f"Model meta not found: {meta_path}")
        return {}
    
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta
    except Exception as e:
        log.warning(f"Could not read {meta_path}: {e}")
        return {}


def save_feature_list(features: Set[str], output_path: Path, prefix: str = ""):
    """Save feature list to text file."""
    sorted_features = sorted(features)
    with open(output_path, "w") as f:
        for feat in sorted_features:
            if prefix:
                f.write(f"{prefix}:{feat}\n")
            else:
                f.write(f"{feat}\n")
    log.info(f"Saved {len(sorted_features)} features to {output_path}")


def build_venn_csv(
    all_features: Set[str],
    runtime_features: Set[str],
    rl_features: Set[str],
    cf_v2_features: Set[str],
    timing_features: Set[str],
    entry_critic_features: Set[str],
    timing_critic_features: Set[str],
    output_path: Path,
):
    """Build CSV with feature presence across all sources."""
    rows = []
    for feat in sorted(all_features):
        rows.append({
            "feature_name": feat,
            "in_runtime": 1 if feat in runtime_features else 0,
            "in_rl_fullyear": 1 if feat in rl_features else 0,
            "in_shadow_cf_v2": 1 if feat in cf_v2_features else 0,
            "in_timing_dataset": 1 if feat in timing_features else 0,
            "in_entry_critic": 1 if feat in entry_critic_features else 0,
            "in_timing_critic": 1 if feat in timing_critic_features else 0,
        })
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    log.info(f"Saved Venn CSV with {len(rows)} features to {output_path}")


def generate_main_report(
    runtime_features: Set[str],
    rl_features: Set[str],
    cf_v2_features: Set[str],
    timing_features: Set[str],
    entry_critic_features: Set[str],
    timing_critic_features: Set[str],
    runtime_files: List[Path],
    dataset_info: Dict,
    model_info: Dict,
    output_path: Path,
):
    """Generate main markdown report."""
    all_features = (
        runtime_features
        | rl_features
        | cf_v2_features
        | timing_features
        | entry_critic_features
        | timing_critic_features
    )
    
    # Find full-chain features (runtime + RL + EntryCritic)
    full_chain_features = (
        runtime_features & rl_features & entry_critic_features
    )
    
    # Features only in runtime
    only_runtime = runtime_features - (rl_features | entry_critic_features | timing_critic_features)
    
    # Features only in RL/AI
    only_ai = (rl_features | entry_critic_features | timing_critic_features) - runtime_features
    
    lines = []
    lines.append("# SNIPER Feature Audit V1")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Kilder")
    lines.append("")
    lines.append("### Runtime filer analysert")
    lines.append(f"- **Antall filer:** {len(runtime_files)}")
    lines.append("- **Hovedfiler:**")
    main_files = [
        "gx1/execution/entry_manager.py",
        "gx1/sniper/**/*.py",
        "gx1/features/**/*.py",
        "gx1/configs/policies/sniper_snapshot/**/*.yaml",
    ]
    for f in main_files:
        lines.append(f"  - `{f}`")
    lines.append("")
    
    lines.append("### Datasett")
    for name, info in dataset_info.items():
        lines.append(f"- **{name}:**")
        lines.append(f"  - Path: `{info['path']}`")
        lines.append(f"  - Rows: {info['rows']:,}")
        lines.append(f"  - Columns: {info['cols']:,}")
    lines.append("")
    
    lines.append("### Modeller")
    for name, info in model_info.items():
        lines.append(f"- **{name}:**")
        lines.append(f"  - Features: {info['n_features']}")
        lines.append(f"  - Target: `{info['target']}`")
        lines.append(f"  - Path: `{info['path']}`")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Feature-sets (stikkord)")
    lines.append("")
    lines.append(f"- **Runtime features:** N = {len(runtime_features):,}")
    lines.append(f"- **RL FULLYEAR features:** N = {len(rl_features):,}")
    lines.append(f"- **Shadow CF V2 features:** N = {len(cf_v2_features):,}")
    lines.append(f"- **Timing dataset features:** N = {len(timing_features):,}")
    lines.append(f"- **EntryCritic features:** N = {len(entry_critic_features):,}")
    lines.append(f"- **TimingCritic features:** N = {len(timing_critic_features):,}")
    lines.append(f"- **Total unique features:** N = {len(all_features):,}")
    lines.append("")
    
    lines.append("### Eksempler (første 20 features per kategori)")
    lines.append("")
    
    for name, features in [
        ("Runtime", sorted(runtime_features)[:20]),
        ("RL FULLYEAR", sorted(rl_features)[:20]),
        ("Shadow CF V2", sorted(cf_v2_features)[:20]),
        ("Timing Dataset", sorted(timing_features)[:20]),
        ("EntryCritic", sorted(entry_critic_features)[:20]),
        ("TimingCritic", sorted(timing_critic_features)[:20]),
    ]:
        lines.append(f"#### {name}")
        for feat in features:
            lines.append(f"- `{feat}`")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Kryss-sjekk (høy nivå)")
    lines.append("")
    
    lines.append(f"### Features KUN i runtime (ikke i RL/AI)")
    lines.append(f"- **Antall:** {len(only_runtime):,}")
    lines.append("- **Topp 20:**")
    for feat in sorted(only_runtime)[:20]:
        lines.append(f"  - `{feat}`")
    lines.append("")
    
    lines.append(f"### Features KUN i RL/AI (ikke i runtime)")
    lines.append(f"- **Antall:** {len(only_ai):,}")
    lines.append("- **Topp 20:**")
    for feat in sorted(only_ai)[:20]:
        lines.append(f"  - `{feat}`")
    lines.append("")
    
    lines.append(f"### Features som er full-chain (runtime + RL + EntryCritic)")
    lines.append(f"- **Antall:** {len(full_chain_features):,}")
    lines.append("- **Eksempler:**")
    for feat in sorted(full_chain_features)[:20]:
        lines.append(f"  - `{feat}`")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Observasjoner / TODO-ideer")
    lines.append("")
    lines.append("### Features som burde med i neste EntryCritic-versjon")
    # Suggest features from runtime that are not in EntryCritic
    candidates = sorted(only_runtime & rl_features)[:5]
    if candidates:
        for feat in candidates:
            lines.append(f"- `{feat}` (finnes i runtime og RL, men ikke i EntryCritic)")
    else:
        lines.append("- (Ingen åpenbare kandidater funnet)")
    lines.append("")
    
    lines.append("### Potensielle mismatch mellom modell og datasett")
    # Check if model features exist in datasets
    entry_missing = entry_critic_features - (rl_features | timing_features)
    timing_missing = timing_critic_features - (rl_features | timing_features)
    if entry_missing:
        lines.append(f"- **EntryCritic features ikke i datasett:** {len(entry_missing)}")
        for feat in sorted(entry_missing)[:5]:
            lines.append(f"  - `{feat}`")
    if timing_missing:
        lines.append(f"- **TimingCritic features ikke i datasett:** {len(timing_missing)}")
        for feat in sorted(timing_missing)[:5]:
            lines.append(f"  - `{feat}`")
    if not entry_missing and not timing_missing:
        lines.append("- (Ingen mismatch funnet)")
    lines.append("")
    
    lines.append("### Mistanker om navn-mismatch")
    # Look for similar feature names that might be the same
    lines.append("- (Manuell inspeksjon av `sniper_feature_venn.csv` anbefales)")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `gx1/scripts/audit_sniper_features_v1.py`*")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"Saved main report to {output_path}")


def main():
    """Main function."""
    log.info("=== SNIPER Feature Audit V1 ===")
    log.info("")
    
    # (1) Analyze runtime features
    log.info("(1) Analyzing runtime features...")
    runtime_features = analyze_runtime_features()
    runtime_files = find_python_files(Path("gx1"), ["sniper/**/*.py", "execution/entry_manager.py"])
    
    # Save runtime features
    save_feature_list(
        runtime_features,
        OUTPUT_DIR / "sniper_runtime_features.txt",
    )
    
    # (2) Analyze datasets
    log.info("")
    log.info("(2) Analyzing datasets...")
    
    datasets = {
        "RL FULLYEAR": Path("data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_PARALLEL.parquet"),
        "Shadow CF V2": Path("data/rl/shadow_counterfactual_FULLYEAR_2025_V2.parquet"),
        "Timing Dataset": Path("data/rl/entry_timing_dataset_FULLYEAR_2025_V1.parquet"),
    }
    
    dataset_info = {}
    rl_features = set()
    cf_v2_features = set()
    timing_features = set()
    
    for name, path in datasets.items():
        cols = analyze_dataset_columns(path)
        if cols:
            dataset_info[name] = {
                "path": str(path),
                "rows": len(pd.read_parquet(path)) if path.exists() else 0,
                "cols": len(cols),
            }
            
            if name == "RL FULLYEAR":
                rl_features = set(cols)
                save_feature_list(
                    rl_features,
                    OUTPUT_DIR / "sniper_rl_fullyear_columns.txt",
                )
            elif name == "Shadow CF V2":
                cf_v2_features = set(cols)
                save_feature_list(
                    cf_v2_features,
                    OUTPUT_DIR / "sniper_shadow_cf_v2_columns.txt",
                )
            elif name == "Timing Dataset":
                timing_features = set(cols)
                save_feature_list(
                    timing_features,
                    OUTPUT_DIR / "sniper_entry_timing_columns.txt",
                )
    
    # (3) Analyze models
    log.info("")
    log.info("(3) Analyzing models...")
    
    models = {
        "EntryCritic V1": Path("gx1/models/entry_critic_v1_meta.json"),
        "TimingCritic V1": Path("gx1/models/entry_timing_critic_v1_meta.json"),
    }
    
    model_info = {}
    entry_critic_features = set()
    timing_critic_features = set()
    
    for name, path in models.items():
        meta = analyze_model_features(path)
        if meta:
            features = meta.get("features", [])
            model_info[name] = {
                "path": str(path),
                "n_features": len(features),
                "target": meta.get("target", "unknown"),
            }
            
            if name == "EntryCritic V1":
                entry_critic_features = set(features)
            elif name == "TimingCritic V1":
                timing_critic_features = set(features)
            
            # Save model features as JSON
            output_name = f"sniper_model_{name.lower().replace(' ', '_')}_features.json"
            with open(OUTPUT_DIR / output_name, "w") as f:
                json.dump(meta, f, indent=2)
            log.info(f"Saved model features to {OUTPUT_DIR / output_name}")
    
    # (4) Build Venn CSV
    log.info("")
    log.info("(4) Building Venn CSV...")
    all_features = (
        runtime_features
        | rl_features
        | cf_v2_features
        | timing_features
        | entry_critic_features
        | timing_critic_features
    )
    
    build_venn_csv(
        all_features,
        runtime_features,
        rl_features,
        cf_v2_features,
        timing_features,
        entry_critic_features,
        timing_critic_features,
        OUTPUT_DIR / "sniper_feature_venn.csv",
    )
    
    # (5) Generate main report
    log.info("")
    log.info("(5) Generating main report...")
    generate_main_report(
        runtime_features,
        rl_features,
        cf_v2_features,
        timing_features,
        entry_critic_features,
        timing_critic_features,
        runtime_files,
        dataset_info,
        model_info,
        OUTPUT_DIR / "SNIPER_FEATURE_AUDIT_V1.md",
    )
    
    log.info("")
    log.info("✅ Feature audit complete!")
    log.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

