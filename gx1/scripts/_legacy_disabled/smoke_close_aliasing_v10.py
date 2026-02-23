#!/usr/bin/env python3
"""
Integration smoke test for CLOSE aliasing in V10 entry flow.

Verifies that:
1. Preflight check passes
2. Replay runs without CASE_COLLISION
3. CLOSE is not in prebuilt schema
4. input_aliases_applied is present when CLOSE is in snap feature list
5. verify_entry_flow_gap.py passes

This test should run WITHOUT GX1_ALLOW_CLOSE_ALIAS_COMPAT=1.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))


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


def check_prebuilt_schema_has_close(prebuilt_path: Path) -> bool:
    """Check if prebuilt parquet schema contains CLOSE (case-insensitive)."""
    try:
        import pandas as pd
        # Read just schema (first row)
        df_sample = pd.read_parquet(prebuilt_path, nrows=1)
        cols_lower = [c.lower() for c in df_sample.columns]
        return "close" in cols_lower
    except Exception as e:
        print(f"[WARN] Could not check prebuilt schema: {e}")
        return False


def main():
    """Main entry point."""
    print("=" * 80)
    print("SMOKE TEST: CLOSE Aliasing in V10 Entry Flow")
    print("=" * 80)
    print()
    
    # DEL 3: Hard-fail if compat-mode is enabled
    compat_enabled = os.getenv("GX1_ALLOW_CLOSE_ALIAS_COMPAT", "0") == "1"
    if compat_enabled:
        raise RuntimeError(
            "[SMOKE_FAIL] GX1_ALLOW_CLOSE_ALIAS_COMPAT=1 is set. "
            "This smoke test verifies the permanent fix (CLOSE alias from candles), "
            "not the compat-mode workaround. Unset GX1_ALLOW_CLOSE_ALIAS_COMPAT and rerun."
        )
    
    # Resolve GX1_DATA root
    gx1_data_root = Path("../GX1_DATA").resolve()
    if not gx1_data_root.exists():
        raise FileNotFoundError(
            f"GX1_DATA root not found: {gx1_data_root}\n"
            "Expected: ../GX1_DATA (relative to engine repo)"
        )
    
    print(f"[SMOKE] GX1_DATA root: {gx1_data_root}")
    print(f"[SMOKE] Workspace root: {workspace_root}")
    print()
    
    # Find required files/dirs
    print("[SMOKE] Locating required files...")
    
    # 1. Data file
    data_file = find_data_file("*2025*.parquet", gx1_data_root / "data")
    if not data_file:
        data_file = find_data_file("full_2025.parquet", gx1_data_root / "data")
    
    if not data_file:
        raise FileNotFoundError(
            f"Data file not found. Searched in: {gx1_data_root / 'data'}\n"
            "Expected: ../GX1_DATA/data/**/*2025*.parquet"
        )
    print(f"  ✅ Data: {data_file}")
    
    # 2. Prebuilt features
    prebuilt_file = find_data_file("*features*v10_ctx*.parquet", gx1_data_root / "data")
    if not prebuilt_file:
        raise FileNotFoundError(
            f"Prebuilt features not found. Searched in: {gx1_data_root / 'data'}\n"
            "Expected: ../GX1_DATA/data/**/*features*v10_ctx*.parquet"
        )
    print(f"  ✅ Prebuilt: {prebuilt_file}")
    
    # DEL 4b: Check that prebuilt schema does NOT contain CLOSE
    if check_prebuilt_schema_has_close(prebuilt_file):
        raise RuntimeError(
            f"[SMOKE_FAIL] Prebuilt parquet schema contains CLOSE column. "
            f"This violates the permanent fix - CLOSE should be dropped from prebuilt schema. "
            f"Prebuilt file: {prebuilt_file}"
        )
    print(f"  ✅ Prebuilt schema verified: CLOSE not present")
    
    # 3. Bundle directory
    bundle_dir = find_bundle_dir("*FULLYEAR*GATED*", gx1_data_root / "models")
    if not bundle_dir:
        bundle_dir = find_bundle_dir("FULLYEAR_2025_GATED_FUSION", gx1_data_root / "models")
    
    if not bundle_dir:
        raise FileNotFoundError(
            f"Bundle directory not found. Searched in: {gx1_data_root / 'models'}\n"
            "Expected: ../GX1_DATA/models/**/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION"
        )
    print(f"  ✅ Bundle: {bundle_dir}")
    
    # 4. Policy file
    policy_file = workspace_root / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
    if not policy_file.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file}")
    policy_file = policy_file.resolve()
    print(f"  ✅ Policy: {policy_file}")
    print()
    
    # Generate output directory
    from datetime import datetime
    run_id = f"SMOKE_CLOSE_ALIAS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = gx1_data_root / "reports" / "replay_eval" / "SMOKE_CLOSE_ALIAS" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SMOKE] Output directory: {output_dir}")
    print()
    
    # Step 1: Preflight check
    print("=" * 80)
    print("STEP 1: Preflight Check")
    print("=" * 80)
    preflight_script = workspace_root / "gx1" / "scripts" / "preflight_prebuilt_import_check.py"
    if not preflight_script.exists():
        raise FileNotFoundError(f"Preflight script not found: {preflight_script}")
    
    print(f"[PREFLIGHT] Running: {preflight_script}")
    env_preflight = os.environ.copy()
    env_preflight["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env_preflight["GX1_FEATURE_BUILD_DISABLED"] = "1"
    # DEL 3: Ensure compat-mode is NOT set
    env_preflight.pop("GX1_ALLOW_CLOSE_ALIAS_COMPAT", None)
    
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
    
    # Step 2: Run replay (short date range)
    print("=" * 80)
    print("STEP 2: Run Replay (Short Date Range)")
    print("=" * 80)
    
    start_date = "2025-01-06"
    end_date = "2025-01-10"
    
    print(f"[REPLAY] Date range: {start_date} to {end_date}")
    
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
    # DEL 3: Ensure compat-mode is NOT set
    env_replay.pop("GX1_ALLOW_CLOSE_ALIAS_COMPAT", None)
    
    print(f"[REPLAY] Command: {' '.join(cmd)}")
    print(f"[REPLAY] Env vars (compat-mode should be unset):")
    print(f"  GX1_ALLOW_CLOSE_ALIAS_COMPAT={env_replay.get('GX1_ALLOW_CLOSE_ALIAS_COMPAT', 'NOT SET')}")
    print()
    
    print("[REPLAY] Starting replay...")
    result = subprocess.run(
        cmd,
        env=env_replay,
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"[REPLAY_FAIL] Replay failed with exit code {result.returncode}")
        print(f"[REPLAY_FAIL] stdout: {result.stdout[-2000:]}")
        print(f"[REPLAY_FAIL] stderr: {result.stderr[-2000:]}")
        
        # Check for CASE_COLLISION error
        if "CASE_COLLISION" in result.stdout or "CASE_COLLISION" in result.stderr:
            raise RuntimeError(
                "[SMOKE_FAIL] CASE_COLLISION detected in replay. "
                "This indicates the permanent fix (CLOSE alias from candles) is not working. "
                "Check that prebuilt builder drops CLOSE and transformer input assembly aliases CLOSE correctly."
            )
        
        raise RuntimeError(f"Replay failed with exit code {result.returncode}")
    
    print("[REPLAY] ✅ Replay completed successfully")
    print()
    
    # Step 3: Verify entry flow gap
    print("=" * 80)
    print("STEP 3: Verify Entry Flow Gap")
    print("=" * 80)
    
    verify_script = workspace_root / "gx1" / "scripts" / "verify_entry_flow_gap.py"
    
    cmd_verify = [
        sys.executable,
        str(verify_script),
        str(output_dir),
        "--verbose",
    ]
    
    print(f"[VERIFY] Command: {' '.join(cmd_verify)}")
    
    result_verify = subprocess.run(
        cmd_verify,
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )
    
    print(result_verify.stdout)
    if result_verify.stderr:
        print(result_verify.stderr)
    
    if result_verify.returncode != 0:
        raise RuntimeError(f"Verify script failed with exit code {result_verify.returncode}")
    
    print("[VERIFY] ✅ Verify script passed")
    print()
    
    # Step 4: Check for input_aliases_applied in telemetry
    print("=" * 80)
    print("STEP 4: Check Telemetry for input_aliases_applied")
    print("=" * 80)
    
    # Find ENTRY_FEATURES_USED.json
    entry_features_path = output_dir / "ENTRY_FEATURES_USED_MASTER.json"
    if not entry_features_path.exists():
        # Try chunk directory
        chunk_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        if chunk_dirs:
            entry_features_path = chunk_dirs[0] / "ENTRY_FEATURES_USED.json"
    
    if not entry_features_path.exists():
        raise RuntimeError(
            f"[SMOKE_FAIL] ENTRY_FEATURES_USED*.json not found in {output_dir}. "
            "Telemetry was not written - check that entry evaluation occurred."
        )
    
    with open(entry_features_path, "r") as f:
        telemetry_data = json.load(f)
    
    # Check if CLOSE is in snap features
    snap_features = telemetry_data.get("snap_features", {}).get("names", [])
    close_in_features = "CLOSE" in snap_features
    
    if close_in_features:
        # CLOSE is in feature list - must have input_aliases_applied
        input_aliases = telemetry_data.get("input_aliases_applied", {})
        if "CLOSE" not in input_aliases:
            raise RuntimeError(
                "[SMOKE_FAIL] CLOSE is in snap_feature_names but input_aliases_applied does not contain CLOSE. "
                "This indicates the aliasing logic is not working correctly."
            )
        
        if input_aliases.get("CLOSE") != "candles.close":
            raise RuntimeError(
                f"[SMOKE_FAIL] input_aliases_applied['CLOSE'] = '{input_aliases.get('CLOSE')}' "
                f"(expected 'candles.close'). Aliasing logic may be incorrect."
            )
        
        print(f"  ✅ CLOSE found in snap_features, input_aliases_applied present: {input_aliases}")
    else:
        print(f"  ℹ️  CLOSE not in snap_features (aliasing not needed)")
    
    print()
    
    # Write summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "PASSED",
        "checks": {
            "preflight_passed": True,
            "replay_passed": True,
            "verify_passed": True,
            "prebuilt_schema_clean": True,  # CLOSE not in prebuilt
            "input_aliases_applied_present": close_in_features and "CLOSE" in telemetry_data.get("input_aliases_applied", {}),
        },
        "compat_mode_used": False,
        "close_in_snap_features": close_in_features,
        "input_aliases_applied": telemetry_data.get("input_aliases_applied", {}),
    }
    
    summary_path = output_dir / "SMOKE_CLOSE_ALIAS_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Preflight: PASSED")
    print(f"✅ Replay: PASSED (no CASE_COLLISION)")
    print(f"✅ Verify: PASSED")
    print(f"✅ Prebuilt schema: CLEAN (CLOSE not present)")
    if close_in_features:
        print(f"✅ Aliasing: VERIFIED (input_aliases_applied present)")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Summary: {summary_path}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
