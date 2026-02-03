#!/usr/bin/env python3
"""
GX1 Doctor - Quick sanity checks for GX1 environment.

Usage:
    python -m gx1.tools.gx1_doctor
    python -m gx1.tools.gx1_doctor --strict
    python -m gx1.tools.gx1_doctor --json

Exit codes:
    0 = OK
    2 = FAIL (blocking issues found)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add workspace root to path
_workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_workspace_root))


class CheckResult:
    """Result of a single check."""
    
    def __init__(self, name: str, status: str, message: str, blocking: bool = False, fix: Optional[str] = None):
        self.name = name
        self.status = status  # "OK", "WARN", "FAIL"
        self.message = message
        self.blocking = blocking
        self.fix = fix
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "blocking": self.blocking,
            "fix": self.fix,
        }


def check_canonical_python() -> CheckResult:
    """Check: canonical Python identity (BLOCKING in TRUTH/SMOKE)."""
    canonical_python = os.getenv("CANONICAL_PYTHON", "/home/andre2/venvs/gx1/bin/python")
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    truth_or_smoke = run_mode in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"

    exe = sys.executable
    if exe == canonical_python:
        return CheckResult(
            name="Canonical Python",
            status="OK",
            message=f"sys.executable matches CANONICAL_PYTHON: {exe}",
            blocking=False,
        )

    fix = (
        "Wrong Python/venv. Do NOT install packages blindly. Activate correct env.\n"
        f"Use full path:\n  {canonical_python} -m gx1.tools.gx1_doctor --strict\n"
        f"Or set:\n  export CANONICAL_PYTHON={canonical_python}"
    )
    return CheckResult(
        name="Canonical Python",
        status="FAIL" if truth_or_smoke else "WARN",
        message=f"sys.executable={exe} does not match canonical_python={canonical_python}",
        blocking=truth_or_smoke,
        fix=fix,
    )


def resolve_gx1_data_root() -> Tuple[Optional[Path], Optional[str]]:
    """Resolve GX1_DATA root using same logic as scripts. Returns (path, error)."""
    gx1_data_env = os.environ.get("GX1_DATA_DIR") or os.environ.get("GX1_DATA_ROOT")
    gx1_data = Path(gx1_data_env) if gx1_data_env else Path.home() / "GX1_DATA"
    gx1_data = gx1_data.expanduser().resolve()
    
    if not gx1_data.exists():
        return None, f"Path does not exist: {gx1_data}"
    
    if gx1_data.name != "GX1_DATA":
        return None, f"Basename must be 'GX1_DATA', got: {gx1_data.name}"
    
    return gx1_data, None


def find_engine_root() -> Tuple[Optional[Path], Optional[str]]:
    """Find GX1_ENGINE root. Returns (path, error)."""
    # Try to find from __file__ (go up 2 levels from gx1/tools/gx1_doctor.py)
    current_file = Path(__file__).resolve()
    candidate = current_file.parent.parent.parent  # gx1/tools -> gx1 -> GX1_ENGINE
    
    # Check for markers
    markers = [".git", "setup.py", "pyproject.toml", "Makefile"]
    for marker in markers:
        if (candidate / marker).exists():
            return candidate, None
    
    # Fallback: go up until we find .git or setup.py
    for parent in candidate.parents:
        for marker in markers:
            if (parent / marker).exists():
                return parent, None
    
    return None, "Could not find ENGINE root (no .git, setup.py, pyproject.toml, or Makefile found)"


def check_gx1_data_root() -> CheckResult:
    """Check 1: GX1_DATA root sanity (BLOCKING)."""
    path, error = resolve_gx1_data_root()
    
    if error:
        return CheckResult(
            name="GX1_DATA root",
            status="FAIL",
            message=error,
            blocking=True,
            fix=f"export GX1_DATA_DIR=/path/to/GX1_DATA (or set GX1_DATA_ROOT, or create ~/GX1_DATA)"
        )
    
    return CheckResult(
        name="GX1_DATA root",
        status="OK",
        message=f"Resolved: {path}",
        blocking=False,
    )


def check_engine_root() -> CheckResult:
    """Check 2: ENGINE root sanity (BLOCKING)."""
    path, error = find_engine_root()
    
    if error:
        return CheckResult(
            name="ENGINE root",
            status="FAIL",
            message=error,
            blocking=True,
            fix="Ensure you're running from GX1_ENGINE checkout with .git or setup.py"
        )
    
    return CheckResult(
        name="ENGINE root",
        status="OK",
        message=f"Resolved: {path}",
        blocking=False,
    )


def check_dual_data_root(strict: bool = False) -> CheckResult:
    """Check 3: Dual data-root detector (WARN or BLOCKING in TRUTH)."""
    # Check if /home/andre2/src/GX1_DATA exists (should never happen)
    src_data_path = Path("/home/andre2/src/GX1_DATA")
    if src_data_path.exists():
        is_truth = os.getenv("GX1_REPLAY_MODE") in ("TRUTH", "SMOKE")
        blocking = is_truth or strict
        
        return CheckResult(
            name="Dual data-root",
            status="FAIL" if blocking else "WARN",
            message=f"Found legacy data root: {src_data_path} (should not exist)",
            blocking=blocking,
            fix=f"Remove or move {src_data_path} - GX1_DATA should only be at ~/GX1_DATA or GX1_DATA_DIR"
        )
    
    return CheckResult(
        name="Dual data-root",
        status="OK",
        message="No legacy /home/andre2/src/GX1_DATA found",
        blocking=False,
    )


def check_prebuilt_existence() -> CheckResult:
    """Check 4: Prebuilt existence (SOFT/BLOCKING with flag)."""
    replay_mode = os.getenv("GX1_REPLAY_MODE", "").upper()
    use_prebuilt = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "").lower() in ("1", "true", "yes")
    prebuilt_path_env = os.getenv("GX1_REPLAY_PREBUILT_FEATURES_PATH")
    
    # Check if we're in PREBUILT mode (either via env or flag)
    is_prebuilt_mode = replay_mode == "PREBUILT" or use_prebuilt
    
    if is_prebuilt_mode:
        if not prebuilt_path_env:
            return CheckResult(
                name="Prebuilt existence",
                status="WARN",
                message="PREBUILT mode but no prebuilt path set",
                blocking=False,
                fix="Set GX1_REPLAY_PREBUILT_FEATURES_PATH=/path/to/prebuilt.parquet"
            )
        
        try:
            prebuilt_path = Path(prebuilt_path_env).expanduser().resolve()
            if not prebuilt_path.exists():
                return CheckResult(
                    name="Prebuilt existence",
                    status="FAIL",
                    message=f"Prebuilt path does not exist: {prebuilt_path}",
                    blocking=True,
                    fix=f"Ensure prebuilt file exists: {prebuilt_path}"
                )
            
            return CheckResult(
                name="Prebuilt existence",
                status="OK",
                message=f"Prebuilt found: {prebuilt_path}",
                blocking=False,
            )
        except Exception as e:
            return CheckResult(
                name="Prebuilt existence",
                status="WARN",
                message=f"Prebuilt path invalid: {prebuilt_path_env} ({e})",
                blocking=False,
                fix="Set GX1_REPLAY_PREBUILT_FEATURES_PATH to a valid path"
            )
    
    # Not PREBUILT mode - just info
    if prebuilt_path_env:
        try:
            prebuilt_path = Path(prebuilt_path_env).expanduser().resolve()
            exists = prebuilt_path.exists()
            return CheckResult(
                name="Prebuilt existence",
                status="OK" if exists else "WARN",
                message=f"Prebuilt path set but not required: {prebuilt_path} (exists: {exists})",
                blocking=False,
            )
        except Exception:
            # Invalid path but not required, so just OK
            pass
    
    return CheckResult(
        name="Prebuilt existence",
        status="OK",
        message="No prebuilt required (not in PREBUILT mode)",
        blocking=False,
    )


def check_output_policy() -> CheckResult:
    """Check 5: Output policy sanity (INFO)."""
    output_mode = os.getenv("GX1_OUTPUT_MODE", "MINIMAL").upper()
    
    # Get limits from reports_cleanup
    try:
        from gx1.scripts.reports_cleanup import FILE_COUNT_LIMITS
        limits = FILE_COUNT_LIMITS
    except ImportError:
        limits = {"MINIMAL": 2000, "DEBUG": 20000, "TRUTH": None}
    
    limit = limits.get(output_mode, "Unknown")
    limit_str = str(limit) if limit is not None else "No limit"
    
    return CheckResult(
        name="Output policy",
        status="OK",
        message=f"GX1_OUTPUT_MODE={output_mode}, file limit={limit_str}",
        blocking=False,
    )


def check_imports() -> CheckResult:
    """Check 6: Quick import check (BLOCKING)."""
    try:
        import gx1
        import gx1.scripts.replay_eval_gated_parallel
        return CheckResult(
            name="Import check",
            status="OK",
            message="gx1 and gx1.scripts.replay_eval_gated_parallel imported successfully",
            blocking=False,
        )
    except ImportError as e:
        return CheckResult(
            name="Import check",
            status="FAIL",
            message=f"Import failed: {e}",
            blocking=True,
            fix="Run: pip install -e . (or ensure gx1 package is in PYTHONPATH)"
        )
    except Exception as e:
        return CheckResult(
            name="Import check",
            status="FAIL",
            message=f"Unexpected error during import: {e}",
            blocking=True,
            fix="Check Python environment and gx1 package installation"
        )


def run_checks(strict: bool = False) -> Tuple[List[CheckResult], bool]:
    """Run all checks. Returns (results, has_failures)."""
    checks = [
        check_canonical_python(),
        check_gx1_data_root(),
        check_engine_root(),
        check_dual_data_root(strict=strict),
        check_prebuilt_existence(),
        check_output_policy(),
        check_imports(),
    ]
    
    has_failures = any(c.status == "FAIL" or (c.blocking and c.status == "WARN") for c in checks)
    
    return checks, has_failures


def print_results(results: List[CheckResult], debug: bool = False):
    """Print results in human-readable format."""
    print("GX1 Doctor - Environment Checks")
    print("=" * 60)
    
    for result in results:
        if result.status == "OK":
            icon = "✅"
        elif result.status == "WARN":
            icon = "⚠️"
        else:
            icon = "❌"
        
        print(f"{icon} {result.name}: {result.message}")
        
        if result.fix:
            print(f"   Fix: {result.fix}")
        
        if debug and result.status != "OK":
            # In debug mode, we could print more details
            pass
    
    print("=" * 60)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GX1 Doctor - Quick sanity checks")
    parser.add_argument("--strict", action="store_true", help="Make warnings into failures")
    parser.add_argument("--debug", action="store_true", help="Print tracebacks on errors")
    parser.add_argument("--json", action="store_true", help="Output JSON report to stdout")
    parsed_args = parser.parse_args(args)
    
    try:
        results, has_failures = run_checks(strict=parsed_args.strict)
        
        if parsed_args.json:
            json_output = {
                "checks": [r.to_dict() for r in results],
                "has_failures": has_failures,
                "exit_code": 2 if has_failures else 0,
            }
            print(json.dumps(json_output, indent=2))
        else:
            print_results(results, debug=parsed_args.debug)
        
        return 2 if has_failures else 0
    
    except Exception as e:
        if parsed_args.debug:
            traceback.print_exc()
        else:
            print(f"❌ FATAL: {e}", file=sys.stderr)
            print("Run with --debug for traceback", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
