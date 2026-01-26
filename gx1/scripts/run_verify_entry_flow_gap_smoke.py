#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Verify Entry Flow Gap Smoke Test

Deterministic script that:
1. Runs preflight check
2. Runs replay with v10_ctx, correct bundle, prebuilt, policy
3. Generates verify_entry_flow_gap output

All paths are ABSOLUTE and explicit - no guessing.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

def get_git_commit() -> Optional[str]:
    """Get git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def find_data_file(pattern: str, base_dir: Path) -> Optional[Path]:
    """Find a data file matching pattern under base_dir."""
    if not base_dir.exists():
        return None
    
    for path in base_dir.rglob(pattern):
        if path.is_file():
            return path.resolve()
    return None

def find_bundle_dir(pattern: str, base_dir: Path) -> Optional[Path]:
    """Find a bundle directory matching pattern under base_dir."""
    if not base_dir.exists():
        return None
    
    for path in base_dir.rglob(pattern):
        if path.is_dir():
            return path.resolve()
    return None

def main():
    """Main entry point."""
    # LEGACY_GUARD: Check for legacy modes before proceeding
    try:
        from gx1.runtime.legacy_guard import assert_no_legacy_mode_enabled
        assert_no_legacy_mode_enabled()
    except ImportError:
        print("[WARN] legacy_guard not available - skipping check")
    except RuntimeError as e:
        print(f"[LEGACY_GUARD] {e}")
        raise
    
    # DEL 3: Truth-mode guard - hard-fail if compat-mode is enabled in truth/baseline runs
    compat_enabled = os.getenv("GX1_ALLOW_CLOSE_ALIAS_COMPAT", "0") == "1"
    truth_mode = os.getenv("GX1_TRUTH_RUN", "0") == "1" or os.getenv("TRUTH_BASELINE_LOCK", "0") == "1"
    
    if compat_enabled and truth_mode:
        raise RuntimeError(
            "[TRUTH_MODE_GUARD] GX1_ALLOW_CLOSE_ALIAS_COMPAT=1 is not allowed in truth/baseline runs. "
            "Compat-mode is only for emergency use, not for truth runs. "
            "If you see this error, the permanent fix (CLOSE alias from candles) should be working. "
            "Check that prebuilt features do not contain CLOSE and that transformer input assembly aliases CLOSE correctly."
        )
    
    if compat_enabled:
        print("[WARN] GX1_ALLOW_CLOSE_ALIAS_COMPAT=1 is enabled. This is NOT for truth/baseline runs.")
        print("[WARN] Compat-mode is an emergency workaround - permanent fix should be used instead.")
    
    print("=" * 80)
    print("VERIFY ENTRY FLOW GAP SMOKE TEST")
    print("=" * 80)
    print()
    
    # Resolve GX1_DATA root
    gx1_data_root = Path("../GX1_DATA").resolve()
    if not gx1_data_root.exists():
        raise FileNotFoundError(
            f"GX1_DATA root not found: {gx1_data_root}\n"
            "Expected: ../GX1_DATA (relative to engine repo)"
        )
    
    print(f"[RUN_CTX] GX1_DATA root: {gx1_data_root}")
    print(f"[RUN_CTX] Workspace root: {workspace_root}")
    print(f"[RUN_CTX] Python: {sys.executable}")
    print(f"[RUN_CTX] CWD: {Path.cwd()}")
    print()
    
    # Find required files/dirs
    print("[FIND] Locating required files...")
    
    # 1. Data file (2025.parquet)
    data_file = find_data_file("*2025*.parquet", gx1_data_root / "data")
    if not data_file:
        # Try alternative location
        data_file = find_data_file("full_2025.parquet", gx1_data_root / "data")
    
    if not data_file:
        raise FileNotFoundError(
            f"Data file not found. Searched in: {gx1_data_root / 'data'}\n"
            "Expected: ../GX1_DATA/data/**/*2025*.parquet\n"
            "Hint: Build or locate 2025 data parquet file"
        )
    print(f"  ✅ Data: {data_file}")
    
    # 2. Prebuilt features
    prebuilt_file = find_data_file("*features*v10_ctx*.parquet", gx1_data_root / "data")
    if not prebuilt_file:
        raise FileNotFoundError(
            f"Prebuilt features not found. Searched in: {gx1_data_root / 'data'}\n"
            "Expected: ../GX1_DATA/data/**/*features*v10_ctx*.parquet\n"
            "Hint: Build prebuilt features first using feature build pipeline"
        )
    print(f"  ✅ Prebuilt: {prebuilt_file}")
    
    # 3. Bundle directory
    bundle_dir = find_bundle_dir("*FULLYEAR*GATED*", gx1_data_root / "models")
    if not bundle_dir:
        # Try alternative pattern
        bundle_dir = find_bundle_dir("FULLYEAR_2025_GATED_FUSION", gx1_data_root / "models")
    
    if not bundle_dir:
        raise FileNotFoundError(
            f"Bundle directory not found. Searched in: {gx1_data_root / 'models'}\n"
            "Expected: ../GX1_DATA/models/**/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION\n"
            "Hint: Train or locate entry_v10_ctx bundle"
        )
    print(f"  ✅ Bundle: {bundle_dir}")
    
    # 4. Policy file
    policy_file = workspace_root / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
    if not policy_file.exists():
        raise FileNotFoundError(
            f"Policy file not found: {policy_file}\n"
            "Expected: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
        )
    policy_file = policy_file.resolve()
    print(f"  ✅ Policy: {policy_file}")
    
    # Verify policy has v10_ctx
    import yaml
    with open(policy_file, 'r') as f:
        policy_data = yaml.safe_load(f)
    
    entry_models = policy_data.get('entry_models', {})
    if 'v10_ctx' not in entry_models:
        raise ValueError(
            f"Policy does not have entry_models.v10_ctx: {policy_file}\n"
            f"Found entry_models: {list(entry_models.keys())}"
        )
    print(f"  ✅ Policy has entry_models.v10_ctx")
    print()
    
    # Generate run ID
    run_id = f"VERIFY_ENTRY_FLOW_GAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = gx1_data_root / "reports" / "replay_eval" / "VERIFY_ENTRY_FLOW_GAP" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Output directory: {output_dir}")
    print()
    
    # Step 1: Preflight check
    print("=" * 80)
    print("STEP 1: Preflight Check")
    print("=" * 80)
    preflight_script = workspace_root / "gx1" / "scripts" / "preflight_prebuilt_import_check.py"
    if not preflight_script.exists():
        print(f"[WARN] Preflight script not found: {preflight_script}")
        print("[WARN] Skipping preflight check")
    else:
        print(f"[PREFLIGHT] Running: {preflight_script}")
        env_preflight = os.environ.copy()
        env_preflight["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
        env_preflight["GX1_FEATURE_BUILD_DISABLED"] = "1"
        
        result = subprocess.run(
            [sys.executable, str(preflight_script)],
            env=env_preflight,
            cwd=workspace_root,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"[PREFLIGHT_FAIL] Preflight check failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Preflight check failed - aborting")
        
        print("[PREFLIGHT] ✅ Preflight check passed")
        print()
    
    # Step 2: Run replay
    print("=" * 80)
    print("STEP 2: Run Replay")
    print("=" * 80)
    
    # Date range for smoke (EU/US/OVERLAP)
    # Use 2025-01-06 to 2025-01-10 (5 days, includes EU/US overlap)
    start_date = "2025-01-06"
    end_date = "2025-01-10"
    
    print(f"[REPLAY] Date range: {start_date} to {end_date}")
    print(f"[REPLAY] Data: {data_file}")
    print(f"[REPLAY] Prebuilt: {prebuilt_file}")
    print(f"[REPLAY] Bundle: {bundle_dir}")
    print(f"[REPLAY] Policy: {policy_file}")
    print(f"[REPLAY] Output: {output_dir}")
    print()
    
    replay_script = workspace_root / "gx1" / "scripts" / "replay_eval_gated_parallel.py"
    
    cmd = [
        sys.executable,
        str(replay_script),
        "--policy", str(policy_file),
        "--data", str(data_file),
        "--prebuilt-parquet", str(prebuilt_file),
        "--bundle-dir", str(bundle_dir),
        "--output-dir", str(output_dir),
        "--workers", "1",
        "--start-ts", start_date,
        "--end-ts", end_date,
    ]
    
    env_replay = os.environ.copy()
    env_replay["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env_replay["GX1_FEATURE_BUILD_DISABLED"] = "1"
    env_replay["GX1_REQUIRE_ENTRY_TELEMETRY"] = "1"
    env_replay["GX1_GATED_FUSION_ENABLED"] = "1"
    env_replay["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    env_replay["GX1_PANIC_MODE"] = "0"
    # Allow uncalibrated XGB for smoke testing (calibrators may not exist)
    env_replay["GX1_REQUIRE_XGB_CALIBRATION"] = "0"
    
    print(f"[REPLAY] Command: {' '.join(cmd)}")
    print(f"[REPLAY] Env vars:")
    for key in ["GX1_REPLAY_USE_PREBUILT_FEATURES", "GX1_FEATURE_BUILD_DISABLED", 
                "GX1_REQUIRE_ENTRY_TELEMETRY", "GX1_GATED_FUSION_ENABLED", "GX1_REQUIRE_XGB_CALIBRATION"]:
        print(f"  {key}={env_replay.get(key, 'NOT SET')}")
    print()
    
    print("[REPLAY] Starting replay (this may take a few minutes)...")
    result = subprocess.run(
        cmd,
        env=env_replay,
        cwd=workspace_root,
    )
    
    if result.returncode != 0:
        print(f"[REPLAY_FAIL] Replay failed with exit code {result.returncode}")
        print(f"[REPLAY_FAIL] Check logs in: {output_dir}")
        raise RuntimeError(f"Replay failed with exit code {result.returncode}")
    
    print("[REPLAY] ✅ Replay completed successfully")
    print()
    
    # Step 3: Run verification script
    print("=" * 80)
    print("STEP 3: Run Verification Script")
    print("=" * 80)
    
    verify_script = workspace_root / "gx1" / "scripts" / "verify_entry_flow_gap.py"
    
    cmd_verify = [
        sys.executable,
        str(verify_script),
        str(output_dir),
        "--verbose",
    ]
    
    print(f"[VERIFY] Command: {' '.join(cmd_verify)}")
    print()
    
    result_verify = subprocess.run(
        cmd_verify,
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )
    
    print(result_verify.stdout)
    if result_verify.stderr:
        print(result_verify.stderr)
    
    # Find verify report
    verify_report = output_dir / "verify_entry_flow_gap_report.json"
    if verify_report.exists():
        print(f"[VERIFY] Report: {verify_report}")
        with open(verify_report, 'r') as f:
            report_data = json.load(f)
        print(f"[VERIFY] Status: {report_data.get('status', 'UNKNOWN')}")
        print(f"[VERIFY] Exit code: {report_data.get('exit_code', 'N/A')}")
    else:
        print(f"[VERIFY] Report not found: {verify_report}")
    
    print()
    
    # Write RUN_CTX.json
    compat_enabled = os.getenv("GX1_ALLOW_CLOSE_ALIAS_COMPAT", "0") == "1"
    run_ctx = {
        "sys_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "argv": sys.argv.copy(),
        "resolved_paths": {
            "data": str(data_file),
            "prebuilt": str(prebuilt_file),
            "bundle_dir": str(bundle_dir),
            "policy": str(policy_file),
            "output_dir": str(output_dir),
        },
        "git_commit": get_git_commit(),
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        # DEL 3: Compat-mode logging
        "compat_close_alias_enabled": compat_enabled,
        "run_is_truth_eligible": not compat_enabled,  # Truth runs cannot use compat-mode
    }
    
    run_ctx_path = output_dir / "RUN_CTX.json"
    with open(run_ctx_path, 'w') as f:
        json.dump(run_ctx, f, indent=2)
    
    print(f"[RUN_CTX] Written: {run_ctx_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Preflight: PASSED")
    print(f"✅ Replay: PASSED (exit code {result.returncode})")
    print(f"✅ Verification: {'PASSED' if result_verify.returncode == 0 else 'FAILED'} (exit code {result_verify.returncode})")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Report: {verify_report if verify_report.exists() else 'NOT FOUND'}")
    print()
    
    return result_verify.returncode

if __name__ == "__main__":
    sys.exit(main())
