#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSoT Audit for FULLYEAR_2025 ENTRY_V10_CTX + GATED FUSION + XGB (inkl ASIA)

Verifies:
1. Dataset files (absolute paths, sha256, row counts)
2. Manifest.json files (calibration_applied, calibrator_usage_stats, session_histograms, ts_min/ts_max, OHLC na==0)
3. XGB models + calibrators (paths, canonical, sha256, mtime)
4. Training artifacts (gated vs baseline)
5. IDENTICAL_OUTPUT root cause analysis
6. GO/NO-GO recommendation
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_info(filepath: Path) -> Optional[Dict]:
    """Get file info: exists, absolute path, sha256, size, mtime."""
    if not filepath.exists():
        return None
    
    abs_path = filepath.resolve()
    return {
        "exists": True,
        "absolute_path": str(abs_path),
        "sha256": compute_sha256(abs_path),
        "size_bytes": abs_path.stat().st_size,
        "mtime": datetime.fromtimestamp(abs_path.stat().st_mtime).isoformat(),
    }


def check_parquet_file(filepath: Path) -> Dict:
    """Check parquet file: path, sha256, row count, OHLC na counts."""
    info = get_file_info(filepath)
    if not info:
        return {"exists": False, "filepath": str(filepath)}
    
    result = info.copy()
    
    # Read parquet and get row count
    try:
        df = pd.read_parquet(filepath)
        result["row_count"] = len(df)
        
        # Check OHLC NA counts
        ohlc_cols = ["open", "high", "low", "close"]
        ohlc_na = {}
        for col in ohlc_cols:
            if col in df.columns:
                ohlc_na[col] = int(df[col].isna().sum())
        result["ohlc_na_counts"] = ohlc_na
        result["ohlc_na_total"] = sum(ohlc_na.values())
        
        # Get ts_min/ts_max if timestamp column exists
        ts_cols = [c for c in df.columns if "timestamp" in c.lower() or "ts" in c.lower() or c == "time"]
        if ts_cols:
            ts_col = ts_cols[0]
            if ts_col in df.columns and df[ts_col].dtype in ["datetime64[ns]", "datetime64[us]"]:
                result["ts_min"] = str(df[ts_col].min())
                result["ts_max"] = str(df[ts_col].max())
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_manifest_json(filepath: Path) -> Optional[Dict]:
    """Check manifest.json file."""
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, "r") as f:
            manifest = json.load(f)
        
        result = {
            "exists": True,
            "absolute_path": str(filepath.resolve()),
            "sha256": compute_sha256(filepath),
        }
        
        # Check required fields
        result["calibration_applied"] = manifest.get("calibration_applied", None)
        result["calibrator_usage_stats"] = manifest.get("calibrator_usage_stats", {})
        result["session_histograms"] = "session_histograms" in manifest
        result["ts_min"] = manifest.get("ts_min", None)
        result["ts_max"] = manifest.get("ts_max", None)
        
        # Check raw==0
        usage_stats = result["calibrator_usage_stats"]
        result["raw_count"] = usage_stats.get("raw", -1) if isinstance(usage_stats, dict) else -1
        
        return result
    except Exception as e:
        return {"exists": True, "error": str(e)}


def check_xgb_models_and_calibrators(policy_config: Dict) -> Dict:
    """Check XGB models and calibrators from policy config."""
    result = {
        "xgb_models": {},
        "calibrators": {},
    }
    
    # Get XGB model paths from policy
    entry_models = policy_config.get("entry_models", {})
    v10_ctx = entry_models.get("v10_ctx", {})
    xgb_config = v10_ctx.get("xgb", {})
    
    sessions = ["EU", "US", "OVERLAP", "ASIA"]
    for session in sessions:
        model_key = f"{session.lower()}_model_path"
        model_path_str = xgb_config.get(model_key, None)
        if model_path_str:
            model_path = Path(model_path_str)
            if not model_path.is_absolute():
                # Resolve relative to workspace root
                workspace_root = Path(__file__).parent.parent
                model_path = workspace_root / model_path
            
            model_info = get_file_info(model_path)
            if model_info:
                result["xgb_models"][session] = model_info
            else:
                result["xgb_models"][session] = {"exists": False, "path": str(model_path)}
    
    # Check calibrators (hierarchy: session/calibrator_platt.joblib, session/regime/calibrator_platt.joblib)
    workspace_root = Path(__file__).parent.parent
    calibration_base = workspace_root / "models" / "xgb_calibration" / "GX1_SNIPER_TRAIN_V10_CTX_GATED"
    
    for session in sessions:
        session_dir = calibration_base / session
        calibrator_path = session_dir / "calibrator_platt.joblib"
        
        calibrator_info = get_file_info(calibrator_path)
        if calibrator_info:
            result["calibrators"][session] = {
                "base": calibrator_info,
                "regimes": {},
            }
        else:
            result["calibrators"][session] = {"base": {"exists": False}}
        
        # Check regime-specific calibrators
        for regime in ["LOW", "UNKNOWN"]:
            regime_dir = session_dir / f"{session}_{regime}"
            regime_calibrator = regime_dir / "calibrator_platt.joblib"
            regime_info = get_file_info(regime_calibrator)
            if regime_info:
                if "regimes" not in result["calibrators"][session]:
                    result["calibrators"][session]["regimes"] = {}
                result["calibrators"][session]["regimes"][regime] = regime_info
    
    return result


def check_training_artifacts() -> Dict:
    """Check training artifacts for gated and baseline."""
    workspace_root = Path(__file__).parent.parent
    
    result = {
        "gated": {},
        "baseline": {},
    }
    
    for variant in ["gated", "baseline"]:
        if variant == "gated":
            artifact_dir = workspace_root / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
        else:
            artifact_dir = workspace_root / "models" / "entry_v10_ctx" / "FULLYEAR_2025_BASELINE_NO_GATE"
        
        required_files = [
            "bundle_metadata.json",
            "train_config.json",
            "env_dump.json",
            "metrics.json",
            "model_state_dict.pt",
        ]
        
        for filename in required_files:
            filepath = artifact_dir / filename
            if filepath.exists():
                info = get_file_info(filepath)
                result[variant][filename] = info
            else:
                result[variant][filename] = {"exists": False}
        
        # Read env_dump.json to check GX1_GATED_FUSION_ENABLED
        env_dump_path = artifact_dir / "env_dump.json"
        if env_dump_path.exists():
            try:
                with open(env_dump_path, "r") as f:
                    env_dump = json.load(f)
                result[variant]["env_vars"] = {
                    "GX1_GATED_FUSION_ENABLED": env_dump.get("GX1_GATED_FUSION_ENABLED", "not set"),
                    "GX1_REQUIRE_XGB_CALIBRATION": env_dump.get("GX1_REQUIRE_XGB_CALIBRATION", "not set"),
                }
            except Exception as e:
                result[variant]["env_vars"] = {"error": str(e)}
    
    return result


def check_identical_output_reports() -> Dict:
    """Check IDENTICAL_OUTPUT reports."""
    workspace_root = Path(__file__).parent.parent
    debug_dir = workspace_root / "reports" / "debug"
    
    result = {
        "root_cause_report": None,
        "artifacts": [],
    }
    
    # Find latest root cause report
    root_cause_files = list(debug_dir.glob("IDENTICAL_OUTPUT_ROOT_CAUSE_*.md"))
    if root_cause_files:
        latest = max(root_cause_files, key=lambda p: p.stat().st_mtime)
        result["root_cause_report"] = {
            "path": str(latest.resolve()),
            "mtime": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
        }
        
        # Read conclusion
        try:
            with open(latest, "r") as f:
                content = f.read()
                if "ROOT CAUSE:" in content:
                    start = content.find("ROOT CAUSE:")
                    end = content.find("\n", start + 11)
                    result["root_cause_report"]["conclusion"] = content[start:end].strip()
                if "Intervention Tests" in content:
                    # Extract intervention test table
                    lines = content.split("\n")
                    in_table = False
                    table_lines = []
                    for line in lines:
                        if "| Test |" in line:
                            in_table = True
                            table_lines.append(line)
                        elif in_table and "|" in line:
                            table_lines.append(line)
                        elif in_table and "|" not in line and line.strip():
                            break
                    result["root_cause_report"]["intervention_tests"] = "\n".join(table_lines)
        except Exception as e:
            result["root_cause_report"]["error"] = str(e)
    
    # Check artifacts JSON files
    artifact_files = list(debug_dir.glob("IDENTICAL_OUTPUT_ARTIFACTS_*.json"))
    for artifact_file in sorted(artifact_files, key=lambda p: p.stat().st_mtime, reverse=True)[:1]:
        try:
            with open(artifact_file, "r") as f:
                artifacts = json.load(f)
            result["artifacts"].append({
                "path": str(artifact_file.resolve()),
                "sha256": compute_sha256(artifact_file),
                "content": artifacts,
            })
        except Exception as e:
            result["artifacts"].append({"path": str(artifact_file), "error": str(e)})
    
    return result


def verify_calibrator_hard_fail() -> Dict:
    """Verify that require_calibration=1 hard-fails if calibrator missing."""
    # This is a code verification - check xgb_calibration.py
    workspace_root = Path(__file__).parent.parent
    calibration_code = workspace_root / "gx1" / "models" / "entry_v10" / "xgb_calibration.py"
    
    result = {
        "code_path": str(calibration_code.resolve()),
        "hard_fail_implemented": False,
        "require_calibration_check": False,
    }
    
    if calibration_code.exists():
        with open(calibration_code, "r") as f:
            content = f.read()
            result["hard_fail_implemented"] = "RuntimeError" in content and "XGB_CALIBRATION_REQUIRED" in content
            result["require_calibration_check"] = "GX1_REQUIRE_XGB_CALIBRATION" in content and "== \"1\"" in content
    
    return result


def main():
    """Run full SSoT audit."""
    workspace_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("FULLYEAR_2025 ENTRY_V10_CTX + GATED FUSION + XGB SSoT AUDIT")
    print("=" * 80)
    print()
    
    audit_result = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
        "manifests": {},
        "xgb_models_calibrators": {},
        "training_artifacts": {},
        "identical_output": {},
        "calibrator_hard_fail": {},
    }
    
    # 1. Check datasets
    print("1. Verifying datasets...")
    dataset_files = [
        "data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_train.parquet",
        "data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_val.parquet",
        "data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_test.parquet",
    ]
    
    for dataset_file in dataset_files:
        filepath = workspace_root / dataset_file
        print(f"  Checking {filepath.name}...")
        info = check_parquet_file(filepath)
        audit_result["datasets"][filepath.name] = info
        
        # Check manifest.json
        manifest_path = filepath.with_suffix(".manifest.json")
        manifest_info = check_manifest_json(manifest_path)
        if manifest_info:
            audit_result["manifests"][filepath.name] = manifest_info
        else:
            # Check archive
            archive_manifests = list((workspace_root / "_archive_artifacts").rglob(f"{manifest_path.name}"))
            if archive_manifests:
                audit_result["manifests"][filepath.name] = {
                    "exists": True,
                    "archived": True,
                    "archive_path": str(archive_manifests[0]),
                }
    
    # 2. Check XGB models and calibrators
    print("\n2. Verifying XGB models and calibrators...")
    policy_path = workspace_root / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml"
    with open(policy_path, "r") as f:
        policy_config = yaml.safe_load(f)
    
    xgb_info = check_xgb_models_and_calibrators(policy_config)
    audit_result["xgb_models_calibrators"] = xgb_info
    
    # 3. Check training artifacts
    print("\n3. Verifying training artifacts...")
    training_info = check_training_artifacts()
    audit_result["training_artifacts"] = training_info
    
    # 4. Check IDENTICAL_OUTPUT reports
    print("\n4. Checking IDENTICAL_OUTPUT reports...")
    identical_output_info = check_identical_output_reports()
    audit_result["identical_output"] = identical_output_info
    
    # 5. Verify calibrator hard-fail
    print("\n5. Verifying calibrator hard-fail logic...")
    hard_fail_info = verify_calibrator_hard_fail()
    audit_result["calibrator_hard_fail"] = hard_fail_info
    
    # Save full audit result
    output_path = workspace_root / "reports" / "cleanup" / f"SSOT_AUDIT_FULLYEAR_2025_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_result, f, indent=2)
    
    print(f"\nFull audit result saved to: {output_path}")
    
    # Generate summary markdown
    generate_summary_markdown(audit_result, workspace_root)
    
    return audit_result


def generate_summary_markdown(audit_result: Dict, workspace_root: Path):
    """Generate 1-page markdown summary."""
    output_path = workspace_root / "reports" / "cleanup" / f"SSOT_AUDIT_FULLYEAR_2025_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(output_path, "w") as f:
        f.write("# FULLYEAR_2025 ENTRY_V10_CTX + GATED FUSION + XGB SSoT Audit\n\n")
        f.write(f"**Date:** {audit_result['timestamp']}\n\n")
        f.write("## Component Status\n\n")
        f.write("| Component | Status | Proof (file+hash) | Notes |\n")
        f.write("|-----------|--------|-------------------|-------|\n")
        
        # Datasets
        for name, info in audit_result["datasets"].items():
            if info.get("exists"):
                status = "✅"
                proof = f"{info['absolute_path']}\nsha256: {info['sha256'][:16]}..."
                notes = f"rows: {info.get('row_count', 'N/A')}, OHLC_na: {info.get('ohlc_na_total', 'N/A')}"
                if info.get("ohlc_na_total", 0) > 0:
                    status = "⚠️"
                    notes += " (OHLC NA > 0)"
            else:
                status = "❌"
                proof = "MISSING"
                notes = "File not found"
            f.write(f"| Dataset: {name} | {status} | {proof} | {notes} |\n")
        
        # Manifests
        for name, info in audit_result["manifests"].items():
            if info.get("exists"):
                status = "✅"
                proof = f"{info.get('absolute_path', 'archived')}\nsha256: {info.get('sha256', 'N/A')[:16]}..." if 'sha256' in info else "archived"
                notes = f"calibration_applied: {info.get('calibration_applied')}, raw_count: {info.get('raw_count')}"
                if info.get("raw_count", -1) != 0:
                    status = "⚠️"
                    notes += " (raw != 0)"
            else:
                status = "❌"
                proof = "MISSING"
                notes = "Manifest not found"
            f.write(f"| Manifest: {name} | {status} | {proof} | {notes} |\n")
        
        # XGB Models
        for session, model_info in audit_result["xgb_models_calibrators"].get("xgb_models", {}).items():
            if model_info.get("exists"):
                status = "✅"
                proof = f"{model_info['absolute_path']}\nsha256: {model_info['sha256'][:16]}..."
                notes = f"size: {model_info['size_bytes']} bytes"
            else:
                status = "❌"
                proof = "MISSING"
                notes = f"Path: {model_info.get('path', 'N/A')}"
            f.write(f"| XGB Model: {session} | {status} | {proof} | {notes} |\n")
        
        # Calibrators
        for session, cal_info in audit_result["xgb_models_calibrators"].get("calibrators", {}).items():
            base = cal_info.get("base", {})
            if base.get("exists"):
                status = "✅"
                proof = f"{base['absolute_path']}\nsha256: {base['sha256'][:16]}..."
                notes = f"base calibrator exists"
                regimes = cal_info.get("regimes", {})
                if regimes:
                    notes += f", regimes: {list(regimes.keys())}"
            else:
                status = "❌"
                proof = "MISSING"
                notes = "Base calibrator missing"
            f.write(f"| Calibrator: {session} | {status} | {proof} | {notes} |\n")
        
        # Training artifacts
        for variant in ["gated", "baseline"]:
            artifacts = audit_result["training_artifacts"].get(variant, {})
            env_vars = artifacts.get("env_vars", {})
            gated_enabled = env_vars.get("GX1_GATED_FUSION_ENABLED", "not set")
            
            status = "✅" if artifacts.get("model_state_dict.pt", {}).get("exists") else "❌"
            model_info = artifacts.get("model_state_dict.pt", {})
            if model_info.get("exists"):
                proof = f"{model_info['absolute_path']}\nsha256: {model_info['sha256'][:16]}..."
            else:
                proof = "MISSING"
            notes = f"GX1_GATED_FUSION_ENABLED={gated_enabled}"
            f.write(f"| Training: {variant.upper()} | {status} | {proof} | {notes} |\n")
        
        # IDENTICAL_OUTPUT
        identical = audit_result.get("identical_output", {})
        root_cause = identical.get("root_cause_report", {})
        if root_cause:
            status = "✅"
            proof = root_cause.get("path", "N/A")
            conclusion = root_cause.get("conclusion", "N/A")
            notes = conclusion
        else:
            status = "⚠️"
            proof = "No report found"
            notes = "Check reports/debug/"
        f.write(f"| IDENTICAL_OUTPUT | {status} | {proof} | {notes} |\n")
        
        # Calibrator hard-fail
        hard_fail = audit_result.get("calibrator_hard_fail", {})
        if hard_fail.get("hard_fail_implemented") and hard_fail.get("require_calibration_check"):
            status = "✅"
            proof = hard_fail.get("code_path", "N/A")
            notes = "RuntimeError on missing calibrator when REQUIRE_XGB_CALIBRATION=1"
        else:
            status = "⚠️"
            proof = hard_fail.get("code_path", "N/A")
            notes = "Hard-fail logic may be incomplete"
        f.write(f"| Calibrator Hard-Fail | {status} | {proof} | {notes} |\n")
        
        # GO/NO-GO
        f.write("\n## GO/NO-GO Recommendation\n\n")
        f.write("### (A) Replay Evaluation with PnL\n")
        f.write("- **Status:** GO (if datasets verified)\n")
        f.write("- **Prerequisites:** All datasets exist, manifests confirm calibration_applied=true, raw==0\n\n")
        
        f.write("### (B) Gate-Aware Policy\n")
        f.write("- **Status:** GO (after PnL eval)\n")
        f.write("- **Prerequisites:** PnL metrics show gate effectiveness, threshold decisions verified\n\n")
        
        f.write("### (C) TCN as Extra Expert\n")
        f.write("- **Status:** NO-GO (defer until PnL eval complete)\n")
        f.write("- **Rationale:** Need PnL baseline before adding complexity\n\n")
    
    print(f"Summary markdown saved to: {output_path}")


if __name__ == "__main__":
    main()
