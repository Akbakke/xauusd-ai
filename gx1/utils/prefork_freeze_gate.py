#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRE-FORK FREEZE GATE (TRUTH/SMOKE ONLY)

Verifies that all required code fixes/guards are present BEFORE any workers
are spawned. This prevents mixed-logic runs where some chunks start before
fixes are applied.

Required guards:
1. compute_pnl_bps import path resolved (no local shadowing)
2. Exit duplicate guards enabled for:
   - REPLAY_END
   - REPLAY_EOF
   - EXIT_FARM_V2_RULES (exit_manager.py)
"""

import ast
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def verify_compute_pnl_bps_import() -> tuple[bool, str]:
    """
    Verify that compute_pnl_bps is imported globally (not locally shadowed).
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from gx1.execution import oanda_demo_runner
        
        # Check if compute_pnl_bps is imported at module level
        source_file = Path(inspect.getfile(oanda_demo_runner))
        with open(source_file, "r") as f:
            source_code = f.read()
        
        # Parse AST to find imports
        tree = ast.parse(source_code, filename=str(source_file))
        
        # Check for global import: from gx1.utils.pnl import compute_pnl_bps
        has_global_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "gx1.utils.pnl":
                    for alias in node.names:
                        if alias.name == "compute_pnl_bps":
                            has_global_import = True
                            break
                if has_global_import:
                    break
        
        if not has_global_import:
            return False, "compute_pnl_bps is not imported globally from gx1.utils.pnl"
        
        # Verify it's actually available in the module
        if not hasattr(oanda_demo_runner, "compute_pnl_bps"):
            # Try to import it to verify
            try:
                from gx1.utils.pnl import compute_pnl_bps
                # Check if it's accessible
                if compute_pnl_bps is None:
                    return False, "compute_pnl_bps import resolved to None"
            except ImportError as e:
                return False, f"compute_pnl_bps cannot be imported: {e}"
        
        return True, "compute_pnl_bps import verified"
    except Exception as e:
        return False, f"Failed to verify compute_pnl_bps import: {e}"


def verify_replay_end_guard() -> tuple[bool, str]:
    """
    Verify that REPLAY_END has duplicate exit guard.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from gx1.execution import oanda_demo_runner
        
        source_file = Path(inspect.getfile(oanda_demo_runner))
        with open(source_file, "r") as f:
            source_code = f.read()
        
        # Look for REPLAY_END guard pattern
        # Should have: if hasattr(self, "_closing_trades") and trade.trade_id in self._closing_trades:
        # and: if hasattr(self, "_exited_trade_ids") and trade.trade_id in self._exited_trade_ids:
        
        has_closing_trades_check = False
        has_exited_trade_ids_check = False
        
        # Find REPLAY_END section (around line 14690-14720)
        lines = source_code.split("\n")
        in_replay_end_section = False
        for i, line in enumerate(lines):
            if "REPLAY_END" in line and "reason=" in line:
                in_replay_end_section = True
            if in_replay_end_section:
                if "_closing_trades" in line and "trade.trade_id in" in line:
                    has_closing_trades_check = True
                if "_exited_trade_ids" in line and "trade.trade_id in" in line:
                    has_exited_trade_ids_check = True
                # Stop after finding both or moving too far
                if has_closing_trades_check and has_exited_trade_ids_check:
                    break
                if i > 0 and "def " in line and "REPLAY_END" not in line:
                    # Moved to next function
                    break
        
        if not has_closing_trades_check:
            return False, "REPLAY_END missing _closing_trades guard"
        if not has_exited_trade_ids_check:
            return False, "REPLAY_END missing _exited_trade_ids guard"
        
        return True, "REPLAY_END guards verified"
    except Exception as e:
        return False, f"Failed to verify REPLAY_END guard: {e}"


def verify_replay_eof_guard() -> tuple[bool, str]:
    """
    Verify that REPLAY_EOF has duplicate exit guard.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from gx1.execution import oanda_demo_runner
        
        source_file = Path(inspect.getfile(oanda_demo_runner))
        with open(source_file, "r") as f:
            source_code = f.read()
        
        # Look for REPLAY_EOF guard pattern
        # Guards should appear before source="REPLAY_EOF" call
        has_closing_trades_check = False
        has_exited_trade_ids_check = False
        
        lines = source_code.split("\n")
        # Find all lines with REPLAY_EOF (in source="REPLAY_EOF" or log messages)
        replay_eof_line_indices = [i for i, line in enumerate(lines) if "REPLAY_EOF" in line]
        
        if not replay_eof_line_indices:
            return False, "REPLAY_EOF not found in source code"
        
        # Check for guards near REPLAY_EOF (within 20 lines before each REPLAY_EOF occurrence)
        for eof_idx in replay_eof_line_indices:
            # Look backwards from REPLAY_EOF for guards
            start_idx = max(0, eof_idx - 20)
            for i in range(start_idx, eof_idx + 1):
                line = lines[i]
                # Check if this line has guard pattern AND is in REPLAY_EOF context
                # (either has REPLAY_EOF in same line, or is within REPLAY_EOF function)
                if "_closing_trades" in line and "trade.trade_id in" in line:
                    # Verify it's in REPLAY_EOF context by checking nearby lines
                    context_lines = " ".join(lines[max(0, i-5):min(len(lines), i+5)])
                    if "REPLAY_EOF" in context_lines:
                        has_closing_trades_check = True
                if "_exited_trade_ids" in line and "trade.trade_id in" in line:
                    context_lines = " ".join(lines[max(0, i-5):min(len(lines), i+5)])
                    if "REPLAY_EOF" in context_lines:
                        has_exited_trade_ids_check = True
                if has_closing_trades_check and has_exited_trade_ids_check:
                    break
            if has_closing_trades_check and has_exited_trade_ids_check:
                break
        
        if not has_closing_trades_check:
            return False, "REPLAY_EOF missing _closing_trades guard"
        if not has_exited_trade_ids_check:
            return False, "REPLAY_EOF missing _exited_trade_ids guard"
        
        return True, "REPLAY_EOF guards verified"
    except Exception as e:
        return False, f"Failed to verify REPLAY_EOF guard: {e}"


def verify_exit_farm_v2_rules_guard() -> tuple[bool, str]:
    """
    Verify that EXIT_FARM_V2_RULES has duplicate exit guard in exit_manager.py.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from gx1.execution import exit_manager
        
        source_file = Path(inspect.getfile(exit_manager))
        with open(source_file, "r") as f:
            source_code = f.read()
        
        # Look for EXIT_FARM_V2_RULES guard pattern
        has_closing_trades_check = False
        has_exited_trade_ids_check = False
        
        lines = source_code.split("\n")
        in_exit_farm_section = False
        for i, line in enumerate(lines):
            if "EXIT_FARM_V2_RULES" in line:
                in_exit_farm_section = True
            if in_exit_farm_section:
                if "_closing_trades" in line and "trade.trade_id in" in line:
                    has_closing_trades_check = True
                if "_exited_trade_ids" in line and "trade.trade_id in" in line:
                    has_exited_trade_ids_check = True
                if has_closing_trades_check and has_exited_trade_ids_check:
                    break
                if i > 0 and "def " in line and "EXIT_FARM_V2_RULES" not in line:
                    break
        
        if not has_closing_trades_check:
            return False, "EXIT_FARM_V2_RULES missing _closing_trades guard"
        if not has_exited_trade_ids_check:
            return False, "EXIT_FARM_V2_RULES missing _exited_trade_ids guard"
        
        return True, "EXIT_FARM_V2_RULES guards verified"
    except Exception as e:
        return False, f"Failed to verify EXIT_FARM_V2_RULES guard: {e}"


def run_prefork_freeze_gate_or_fatal(
    output_dir: Path,
    truth_or_smoke: bool,
    git_head_sha: Optional[str] = None,
    policy_sha: Optional[str] = None,
    bundle_sha: Optional[str] = None,
) -> None:
    """
    Run PRE-FORK FREEZE gate and write PRE_FORK_FREEZE.json if all checks pass.
    On failure, write PRE_FORK_FREEZE_FATAL.json and exit with code 2.
    
    Args:
        output_dir: Output directory for the run
        truth_or_smoke: Whether this is a TRUTH/SMOKE run
        git_head_sha: Git HEAD SHA (optional)
        policy_sha: Policy file SHA256 (optional)
        bundle_sha: Bundle SHA256 (optional)
    """
    if not truth_or_smoke:
        log.info("[PRE_FORK_FREEZE] Skipping (not TRUTH/SMOKE mode)")
        return
    
    log.info("[PRE_FORK_FREEZE] Verifying required code fixes...")
    
    # Verify all required guards (exit_farm_v2_rules_guard removed: ONE PATH, no EXIT_FARM_V2_RULES in exit_manager)
    checks = [
        ("compute_pnl_bps_import", verify_compute_pnl_bps_import),
        ("replay_end_guard", verify_replay_end_guard),
        ("replay_eof_guard", verify_replay_eof_guard),
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        is_valid, message = check_func()
        results[check_name] = {
            "present": is_valid,
            "message": message,
        }
        if not is_valid:
            all_passed = False
            log.error(f"[PRE_FORK_FREEZE] ❌ {check_name}: {message}")
        else:
            log.info(f"[PRE_FORK_FREEZE] ✅ {check_name}: {message}")
    
    if not all_passed:
        # Write FATAL capsule
        fatal_capsule = {
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "fatal_reason": "PRE_FORK_FREEZE_FAIL",
            "error_message": "One or more required guards are missing",
            "output_dir": str(output_dir),
            "checks": results,
            "git_head_sha": git_head_sha,
            "policy_sha": policy_sha,
            "bundle_sha": bundle_sha,
        }
        fatal_path = output_dir / "PRE_FORK_FREEZE_FATAL.json"
        try:
            fatal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fatal_path, "w") as f:
                json.dump(fatal_capsule, f, indent=2)
            log.error(f"[PRE_FORK_FREEZE] FATAL capsule written: {fatal_path}")
        except Exception as e:
            log.error(f"[PRE_FORK_FREEZE] Failed to write FATAL capsule: {e}")
        
        print(f"[PRE_FORK_FREEZE] FATAL: Required guards missing. See {fatal_path}", file=sys.stderr, flush=True)
        sys.exit(2)
    
    # All checks passed - write PRE_FORK_FREEZE.json
    freeze_capsule = {
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "git_head_sha": git_head_sha,
        "policy_sha": policy_sha,
        "bundle_sha": bundle_sha,
        "required_guards": results,
        "all_guards_present": True,
    }
    freeze_path = output_dir / "PRE_FORK_FREEZE.json"
    try:
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file, then rename
        temp_path = freeze_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(freeze_capsule, f, indent=2)
        temp_path.replace(freeze_path)
        log.info(f"[PRE_FORK_FREEZE] ✅ Freeze capsule written: {freeze_path}")
    except Exception as e:
        log.error(f"[PRE_FORK_FREEZE] Failed to write freeze capsule: {e}")
        raise RuntimeError(f"Failed to write PRE_FORK_FREEZE.json: {e}") from e
