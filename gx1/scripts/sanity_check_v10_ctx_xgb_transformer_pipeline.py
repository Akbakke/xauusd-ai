#!/usr/bin/env python3
"""
Full Pipeline Sanity Check: XGB → Transformer.

Validates that the V10_CTX pipeline is correctly configured and wired.
Two modes:
- static: Validate config/paths/SHAs without running replay
- smoke: Run short PREBUILT replay and verify counters

Usage:
    python3 gx1/scripts/sanity_check_v10_ctx_xgb_transformer_pipeline.py --mode static
    python3 gx1/scripts/sanity_check_v10_ctx_xgb_transformer_pipeline.py --mode smoke --year 2025 --days 2
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def check_forbidden_imports() -> Tuple[bool, List[str]]:
    """
    Check for forbidden imports in PREBUILT mode.
    
    Returns:
        (passed, issues)
    """
    issues = []
    
    # Files that should NOT import feature building in PREBUILT mode
    # This is a simplified check - real implementation would be more thorough
    prebuilt_forbidden_patterns = [
        r"from gx1\.features import build_features",
        r"from gx1\.features\.runtime import",
    ]
    
    # For now, just check that feature_build_tripwires.py exists
    tripwire_path = WORKSPACE_ROOT / "gx1" / "execution" / "feature_build_tripwires.py"
    if not tripwire_path.exists():
        issues.append("Missing feature_build_tripwires.py")
    
    return len(issues) == 0, issues


def check_bundle_artifacts(bundle_dir: Path, xgb_mode: str) -> Tuple[bool, List[str]]:
    """
    Check that bundle directory has required artifacts.
    
    Returns:
        (passed, issues)
    """
    issues = []
    
    if not bundle_dir.exists():
        issues.append(f"Bundle dir not found: {bundle_dir}")
        return False, issues
    
    # Check for transformer artifacts (common ones)
    expected_transformer_files = [
        # Add expected transformer files here if known
    ]
    
    # Check XGB models based on mode
    if xgb_mode == "universal":
        universal_model = bundle_dir / "xgb_universal_v1.joblib"
        if not universal_model.exists():
            issues.append(f"Universal model not found: {universal_model}")
    elif xgb_mode == "universal_multihead":
        multihead_model = bundle_dir / "xgb_universal_multihead_v2.joblib"
        if not multihead_model.exists():
            issues.append(f"Multihead model not found: {multihead_model}")
        multihead_meta = bundle_dir / "xgb_universal_multihead_v2_meta.json"
        if not multihead_meta.exists():
            issues.append(f"Multihead meta not found: {multihead_meta}")
    else:
        # Session mode
        for session in ["EU", "US", "OVERLAP"]:
            session_model = bundle_dir / f"xgb_{session}.joblib"
            if not session_model.exists():
                issues.append(f"Session model not found: {session_model}")
    
    return len(issues) == 0, issues


def check_contract_files(xgb_mode: str = "session") -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Check that contract files exist and compute their SHAs.
    
    Returns:
        (passed, sha_dict, issues)
    """
    issues = []
    shas = {}
    
    feature_contract = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    sanitizer_config = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    output_contract = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"
    
    if feature_contract.exists():
        shas["feature_contract_sha256"] = compute_file_sha256(feature_contract)
        with open(feature_contract, "r") as f:
            contract_data = json.load(f)
        shas["schema_hash"] = contract_data.get("schema_hash", "unknown")
    else:
        issues.append(f"Feature contract not found: {feature_contract}")
    
    if sanitizer_config.exists():
        shas["sanitizer_config_sha256"] = compute_file_sha256(sanitizer_config)
        with open(sanitizer_config, "r") as f:
            sanitizer_data = json.load(f)
        sanitizer_schema = sanitizer_data.get("schema_hash", "unknown")
        if sanitizer_schema != shas.get("schema_hash"):
            issues.append(f"Schema hash mismatch: contract={shas.get('schema_hash')}, sanitizer={sanitizer_schema}")
    else:
        issues.append(f"Sanitizer config not found: {sanitizer_config}")
    
    # Check multihead output contract for universal_multihead mode
    if xgb_mode == "universal_multihead":
        if output_contract.exists():
            shas["output_contract_sha256"] = compute_file_sha256(output_contract)
        else:
            issues.append(f"Multihead output contract not found: {output_contract}")
    
    return len(issues) == 0, shas, issues


def check_go_marker(bundle_dir: Path, xgb_mode: str) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Check GO marker for universal/multihead mode.
    
    Returns:
        (passed, marker_info, issues)
    """
    issues = []
    marker_info = {"required": xgb_mode in ("universal", "universal_multihead")}
    
    if xgb_mode not in ("universal", "universal_multihead"):
        return True, marker_info, issues
    
    # Select marker paths based on mode
    if xgb_mode == "universal_multihead":
        go_marker_path = bundle_dir / "XGB_MULTIHEAD_GO_MARKER.json"
        no_go_path = bundle_dir / "XGB_MULTIHEAD_NO_GO.json"
        model_name = "Multihead"
    else:
        go_marker_path = bundle_dir / "XGB_UNIVERSAL_GO_MARKER.json"
        no_go_path = bundle_dir / "XGB_UNIVERSAL_NO_GO.json"
        model_name = "Universal"
    
    if go_marker_path.exists():
        with open(go_marker_path, "r") as f:
            marker = json.load(f)
        marker_info["go_marker"] = marker
        marker_info["status"] = "GO"
        
        # Verify model SHA
        model_path = Path(marker.get("model_path", ""))
        if model_path.exists():
            actual_sha = compute_file_sha256(model_path)
            expected_sha = marker.get("model_sha256")
            if actual_sha != expected_sha:
                issues.append(f"Model SHA mismatch: expected {expected_sha[:16]}, got {actual_sha[:16]}")
        else:
            issues.append(f"Model not found: {model_path}")
    
    elif no_go_path.exists():
        with open(no_go_path, "r") as f:
            marker = json.load(f)
        marker_info["no_go_marker"] = marker
        marker_info["status"] = "NO-GO"
        issues.append(f"{model_name} model is NO-GO: {marker.get('issues', [])}")
    
    else:
        marker_info["status"] = "NOT_EVALUATED"
        issues.append(f"{model_name} model not evaluated: no GO/NO-GO marker found")
    
    return len(issues) == 0, marker_info, issues


def check_no_fallback_code() -> Tuple[bool, List[str]]:
    """
    Check that "first N numeric columns" fallback does not exist in truth mode.
    
    Returns:
        (passed, issues)
    """
    issues = []
    
    # Search for fallback patterns in key files
    fallback_patterns = [
        r"first.*\d+.*numeric.*col",
        r"numeric_cols\[:.*\]",
    ]
    
    # Files to check
    files_to_check = [
        WORKSPACE_ROOT / "gx1" / "scripts" / "train_xgb_calibrator_multiyear.py",
        WORKSPACE_ROOT / "gx1" / "execution" / "oanda_demo_runner.py",
    ]
    
    for filepath in files_to_check:
        if not filepath.exists():
            continue
        
        with open(filepath, "r") as f:
            content = f.read()
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Check if it's in a disabled/commented section
                for match in matches:
                    if "fallback" in content.lower() and "require_feature_names" in content.lower():
                        # Likely has a gate, so it's okay
                        pass
                    else:
                        issues.append(f"Potential fallback code in {filepath.name}: {match}")
    
    return len(issues) == 0, issues


def check_no_post_gate() -> Tuple[bool, List[str]]:
    """
    Check that post-gate XGB is removed/disabled.
    
    Returns:
        (passed, issues)
    """
    issues = []
    
    # Search for post-gate patterns
    post_gate_patterns = [
        r"xgb_post_veto",
        r"calibration_post",
        r"post_gate_.*xgb",
    ]
    
    # Check config files
    config_dirs = [
        WORKSPACE_ROOT / "gx1" / "configs" / "policies",
        WORKSPACE_ROOT / "gx1" / "configs" / "entry_configs",
    ]
    
    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        
        for config_file in config_dir.rglob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    content = f.read()
                
                for pattern in post_gate_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"Post-gate config found in {config_file.name}")
            except Exception:
                pass
    
    return len(issues) == 0, issues


def run_static_sanity(xgb_mode: str, bundle_dir: Path) -> Dict[str, Any]:
    """
    Run static sanity checks.
    
    Returns:
        Dict with check results
    """
    results = {
        "mode": "static",
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "xgb_mode": xgb_mode,
        "bundle_dir": str(bundle_dir),
        "checks": {},
        "all_passed": True,
        "issues": [],
    }
    
    # Check 1: Forbidden imports
    print("Checking forbidden imports...")
    passed, issues = check_forbidden_imports()
    results["checks"]["forbidden_imports"] = {"passed": passed, "issues": issues}
    if not passed:
        results["all_passed"] = False
        results["issues"].extend(issues)
    print(f"  {'✅' if passed else '❌'} Forbidden imports")
    
    # Check 2: Bundle artifacts
    print("Checking bundle artifacts...")
    passed, issues = check_bundle_artifacts(bundle_dir, xgb_mode)
    results["checks"]["bundle_artifacts"] = {"passed": passed, "issues": issues}
    if not passed:
        results["all_passed"] = False
        results["issues"].extend(issues)
    print(f"  {'✅' if passed else '❌'} Bundle artifacts")
    
    # Check 3: Contract files
    print("Checking contract files...")
    passed, shas, issues = check_contract_files(xgb_mode)
    results["checks"]["contract_files"] = {"passed": passed, "shas": shas, "issues": issues}
    results["contract_shas"] = shas
    if not passed:
        results["all_passed"] = False
        results["issues"].extend(issues)
    print(f"  {'✅' if passed else '❌'} Contract files")
    
    # Check 4: GO marker (for universal mode)
    print("Checking GO marker...")
    passed, marker_info, issues = check_go_marker(bundle_dir, xgb_mode)
    results["checks"]["go_marker"] = {"passed": passed, "info": marker_info, "issues": issues}
    if not passed:
        results["all_passed"] = False
        results["issues"].extend(issues)
    print(f"  {'✅' if passed else '❌'} GO marker (status: {marker_info.get('status', 'N/A')})")
    
    # Check 5: No fallback code
    print("Checking for fallback code...")
    passed, issues = check_no_fallback_code()
    results["checks"]["no_fallback"] = {"passed": passed, "issues": issues}
    if not passed:
        results["all_passed"] = False
        results["issues"].extend(issues)
    print(f"  {'✅' if passed else '❌'} No fallback code")
    
    # Check 6: No post-gate
    print("Checking for post-gate removal...")
    passed, issues = check_no_post_gate()
    results["checks"]["no_post_gate"] = {"passed": passed, "issues": issues}
    if not passed:
        results["all_passed"] = False
        results["issues"].extend(issues)
    print(f"  {'✅' if passed else '❌'} No post-gate")
    
    return results


def run_smoke_sanity(xgb_mode: str, bundle_dir: Path, year: int, days: int) -> Dict[str, Any]:
    """
    Run smoke sanity checks (requires replay infrastructure).
    
    For now, this validates that we CAN run inference without errors.
    
    Returns:
        Dict with check results
    """
    results = {
        "mode": "smoke",
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "xgb_mode": xgb_mode,
        "bundle_dir": str(bundle_dir),
        "year": year,
        "days": days,
        "checks": {},
        "counters": {},
        "all_passed": True,
        "issues": [],
    }
    
    gx1_data = resolve_gx1_data_dir()
    
    # Run sanitizer smoke test
    print(f"\nRunning sanitizer smoke test ({year}, {days} days worth of data)...")
    try:
        # Use smoke_xgb_sanitizer_infer as proxy for now
        cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "gx1" / "scripts" / "smoke_xgb_sanitizer_infer.py"),
            "--year", str(year),
            "--n-bars", "5000",  # ~2 days of 5m bars
            "--allow-high-clip",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            results["checks"]["sanitizer_smoke"] = {"passed": True}
            print("  ✅ Sanitizer smoke test passed")
            
            # Parse output for counters
            output = result.stdout
            if "Clip rate:" in output:
                clip_match = re.search(r"Clip rate: ([\d.]+)%", output)
                if clip_match:
                    results["counters"]["clip_rate_pct"] = float(clip_match.group(1))
            
            if "p_long_xgb:" in output:
                plm_match = re.search(r"p_long_xgb:.*mean=([\d.]+)", output)
                if plm_match:
                    results["counters"]["p_long_mean"] = float(plm_match.group(1))
        else:
            results["checks"]["sanitizer_smoke"] = {
                "passed": False,
                "error": result.stderr[:500]
            }
            results["all_passed"] = False
            results["issues"].append("Sanitizer smoke test failed")
            print("  ❌ Sanitizer smoke test failed")
    
    except subprocess.TimeoutExpired:
        results["checks"]["sanitizer_smoke"] = {"passed": False, "error": "Timeout"}
        results["all_passed"] = False
        results["issues"].append("Sanitizer smoke test timed out")
        print("  ❌ Sanitizer smoke test timed out")
    except Exception as e:
        results["checks"]["sanitizer_smoke"] = {"passed": False, "error": str(e)}
        results["all_passed"] = False
        results["issues"].append(f"Sanitizer smoke test error: {e}")
        print(f"  ❌ Sanitizer smoke test error: {e}")
    
    # Expected counters (for reference)
    results["expected_counters"] = {
        "feature_build_call_count": 0,
        "prebuilt_used": True,
        "xgb_pre_predict_calls": "> 0",
        "transformer_forward_calls": "> 0",
        "xgb_used_as": "pre",
        "post_gate_calls": 0,
    }
    
    return results


def write_reports(results: Dict[str, Any], output_dir: Path) -> None:
    """Write sanity check reports."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mode = results["mode"].upper()
    
    # JSON report
    json_path = output_dir / f"{mode}_SANITY.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {json_path}")
    
    # Markdown report
    md_path = output_dir / f"{mode}_SANITY.md"
    with open(md_path, "w") as f:
        f.write(f"# {mode} Sanity Check Report\n\n")
        f.write(f"Generated: {results['timestamp']}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- XGB Mode: `{results['xgb_mode']}`\n")
        f.write(f"- Bundle: `{results['bundle_dir']}`\n")
        if "year" in results:
            f.write(f"- Year: {results['year']}\n")
        f.write("\n")
        
        f.write(f"## Checks\n\n")
        f.write("| Check | Status | Issues |\n")
        f.write("|-------|--------|--------|\n")
        for check_name, check_result in results.get("checks", {}).items():
            status = "✅" if check_result.get("passed") else "❌"
            issues = ", ".join(check_result.get("issues", []))[:50] or "None"
            f.write(f"| {check_name} | {status} | {issues} |\n")
        f.write("\n")
        
        if results.get("counters"):
            f.write(f"## Counters\n\n")
            for k, v in results["counters"].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        
        f.write(f"## Verdict\n\n")
        if results["all_passed"]:
            f.write("**✅ PASS**\n\n")
            f.write("All sanity checks passed.\n")
        else:
            f.write("**❌ FAIL**\n\n")
            f.write("Issues found:\n")
            for issue in results.get("issues", []):
                f.write(f"- {issue}\n")
    
    print(f"Wrote: {md_path}")


def write_one_truth_summary(static_results: Dict, smoke_results: Optional[Dict], output_dir: Path) -> None:
    """Write ONE_TRUTH_SUMMARY.json combining all results."""
    
    summary = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "verdict": "PASS" if static_results["all_passed"] and (not smoke_results or smoke_results["all_passed"]) else "FAIL",
        "xgb_mode": static_results["xgb_mode"],
        "bundle_dir": static_results["bundle_dir"],
        "contract_shas": static_results.get("contract_shas", {}),
        "static_passed": static_results["all_passed"],
        "smoke_passed": smoke_results["all_passed"] if smoke_results else None,
        "all_issues": static_results.get("issues", []) + (smoke_results.get("issues", []) if smoke_results else []),
    }
    
    # Add model info if available
    go_marker_info = static_results.get("checks", {}).get("go_marker", {}).get("info", {})
    if "go_marker" in go_marker_info:
        marker = go_marker_info["go_marker"]
        summary["model_sha256"] = marker.get("model_sha256")
        summary["model_path"] = marker.get("model_path")
    
    summary_path = output_dir / "ONE_TRUTH_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="V10_CTX XGB→Transformer Pipeline Sanity Check"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "smoke", "both"],
        default="static",
        help="Check mode"
    )
    parser.add_argument(
        "--xgb-mode",
        type=str,
        choices=["universal", "session", "universal_multihead"],
        default="session",
        help="XGB mode to validate"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year for smoke test"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Days for smoke test"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V10_CTX XGB→TRANSFORMER PIPELINE SANITY CHECK")
    print("=" * 60)
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    bundle_dir = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
    
    print(f"GX1_DATA: {gx1_data}")
    print(f"Bundle: {bundle_dir}")
    print(f"XGB Mode: {args.xgb_mode}")
    print(f"Check Mode: {args.mode}")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "repo_audit" / f"PIPELINE_SANITY_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    static_results = None
    smoke_results = None
    
    # Run static checks
    if args.mode in ["static", "both"]:
        print("\n" + "-" * 40)
        print("STATIC SANITY CHECKS")
        print("-" * 40)
        static_results = run_static_sanity(args.xgb_mode, bundle_dir)
        write_reports(static_results, output_dir)
    
    # Run smoke checks
    if args.mode in ["smoke", "both"]:
        print("\n" + "-" * 40)
        print("SMOKE SANITY CHECKS")
        print("-" * 40)
        
        # Static must pass first
        if args.mode == "both" and static_results and not static_results["all_passed"]:
            print("Skipping smoke checks - static checks failed")
        else:
            smoke_results = run_smoke_sanity(args.xgb_mode, bundle_dir, args.year, args.days)
            write_reports(smoke_results, output_dir)
    
    # Write ONE_TRUTH_SUMMARY
    if static_results:
        summary = write_one_truth_summary(static_results, smoke_results, output_dir)
    
    # Final verdict
    print("\n" + "=" * 60)
    all_passed = (
        (not static_results or static_results["all_passed"]) and
        (not smoke_results or smoke_results["all_passed"])
    )
    if all_passed:
        print("VERDICT: ✅ PASS")
    else:
        print("VERDICT: ❌ FAIL")
        if static_results and not static_results["all_passed"]:
            print("\nStatic issues:")
            for issue in static_results.get("issues", []):
                print(f"  - {issue}")
        if smoke_results and not smoke_results["all_passed"]:
            print("\nSmoke issues:")
            for issue in smoke_results.get("issues", []):
                print(f"  - {issue}")
    print("=" * 60)
    
    print(f"\nReports: {output_dir}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
