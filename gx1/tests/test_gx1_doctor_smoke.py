#!/usr/bin/env python3
"""
Smoke test for gx1_doctor.

Tests that the script can be imported and that main(["--json"]) returns
a structure with expected keys (not dependent on actual GX1_DATA in CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))


def test_doctor_import():
    """Test that gx1_doctor can be imported."""
    try:
        from gx1.tools.gx1_doctor import main, CheckResult, run_checks
        return True
    except ImportError as e:
        print(f"FAIL: Could not import gx1_doctor: {e}")
        return False


def test_doctor_json_output():
    """Test that --json flag produces valid JSON with expected keys."""
    try:
        from gx1.tools.gx1_doctor import main
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["--json"])
        
        json_str = output.getvalue()
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"FAIL: Invalid JSON output: {e}")
            print(f"Output: {json_str[:200]}")
            return False
        
        # Check expected keys
        required_keys = ["checks", "has_failures", "exit_code"]
        for key in required_keys:
            if key not in data:
                print(f"FAIL: Missing key in JSON output: {key}")
                return False
        
        # Check checks structure
        if not isinstance(data["checks"], list):
            print(f"FAIL: 'checks' should be a list, got {type(data['checks'])}")
            return False
        
        # Each check should have expected keys
        for check in data["checks"]:
            required_check_keys = ["name", "status", "message", "blocking"]
            for key in required_check_keys:
                if key not in check:
                    print(f"FAIL: Check missing key: {key}")
                    return False
        
        print(f"✅ JSON output valid: {len(data['checks'])} checks, exit_code={data['exit_code']}")
        return True
    
    except Exception as e:
        print(f"FAIL: Error testing JSON output: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_check_result():
    """Test CheckResult class."""
    try:
        from gx1.tools.gx1_doctor import CheckResult
        
        result = CheckResult(
            name="Test check",
            status="OK",
            message="Test message",
            blocking=False,
            fix="Test fix"
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["name"] == "Test check"
        assert result_dict["status"] == "OK"
        assert result_dict["message"] == "Test message"
        assert result_dict["blocking"] is False
        assert result_dict["fix"] == "Test fix"
        
        print("✅ CheckResult class works")
        return True
    
    except Exception as e:
        print(f"FAIL: CheckResult test failed: {e}")
        return False


def test_preflight_doctor_fatal():
    """Test that run_gx1_doctor_or_fatal raises on FAIL."""
    try:
        from gx1.utils.preflight_doctor import run_gx1_doctor_or_fatal
        from pathlib import Path
        import tempfile
        import os
        
        # Test with invalid GX1_DATA_DIR (should fail)
        test_output = Path(tempfile.mkdtemp())
        try:
            old_env = os.environ.get("GX1_DATA_DIR")
            os.environ["GX1_DATA_DIR"] = "/nonexistent/path/GX1_DATA"
            
            try:
                run_gx1_doctor_or_fatal(strict=True, truth_or_smoke=True, output_dir=test_output)
                print("FAIL: Should have raised RuntimeError")
                return False
            except RuntimeError as e:
                if "[DOCTOR_FATAL]" in str(e):
                    # Check that capsule was written
                    capsule = test_output / "DOCTOR_FATAL.json"
                    if capsule.exists():
                        print("✅ PASS: run_gx1_doctor_or_fatal correctly raises and writes capsule")
                        return True
                    else:
                        print(f"FAIL: DOCTOR_FATAL.json not found at {capsule}")
                        return False
                else:
                    print(f"FAIL: RuntimeError without [DOCTOR_FATAL] prefix: {e}")
                    return False
            finally:
                if old_env:
                    os.environ["GX1_DATA_DIR"] = old_env
                elif "GX1_DATA_DIR" in os.environ:
                    del os.environ["GX1_DATA_DIR"]
        finally:
            import shutil
            shutil.rmtree(test_output, ignore_errors=True)
    
    except Exception as e:
        print(f"FAIL: Error testing preflight_doctor: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running gx1_doctor smoke tests...")
    print()
    
    tests = [
        ("Import test", test_doctor_import),
        ("CheckResult test", test_check_result),
        ("JSON output test", test_doctor_json_output),
        ("Preflight doctor fatal test", test_preflight_doctor_fatal),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"Test: {name}")
        try:
            if test_func():
                passed += 1
                print("✅ PASS")
            else:
                failed += 1
                print("❌ FAIL")
        except Exception as e:
            failed += 1
            print(f"❌ FAIL: {e}")
        print()
    
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
