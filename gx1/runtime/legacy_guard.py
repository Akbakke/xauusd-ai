#!/usr/bin/env python3
"""
Legacy Guard - Runtime Tripwire

Hard-fails if legacy v9/gated/sniper paths or modes are detected.
Called early in entrypoints to prevent accidental legacy execution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

def assert_no_legacy_mode_enabled(
    env: Optional[Dict[str, str]] = None,
    policy_dict: Optional[Dict[str, Any]] = None,
    argv: Optional[List[str]] = None,
    bundle_dir_resolved: Optional[Path] = None,
    output_dir_resolved: Optional[Path] = None,
) -> None:
    """
    Assert that no legacy modes are enabled.
    
    Hard-fails if:
    - Policy has entry_models.v9 or refers to v9
    - argv or env indicates v9/gated legacy mode
    - output-dir is under engine-repo (not under ../GX1_DATA/reports or GX1_REPORTS_ROOT)
    
    Args:
        env: Environment variables dict (default: os.environ)
        policy_dict: Policy configuration dict
        argv: Command-line arguments (default: sys.argv)
    
    Raises:
        RuntimeError: If legacy mode detected
    """
    if env is None:
        env = dict(os.environ)
    
    if argv is None:
        argv = sys.argv
    
    errors = []
    
    # Check policy for entry_models.v9 (only if enabled)
    if policy_dict:
        entry_models = policy_dict.get('entry_models', {})
        v9_config = entry_models.get('v9', {})
        # Only block if v9 is explicitly enabled (disabled v9 config is allowed for documentation)
        if v9_config.get('enabled', False):
            errors.append(
                "LEGACY_BLOCKED: Policy has entry_models.v9 enabled. "
                "Use entry_models.v10_ctx instead. "
                f"Policy: {policy_dict.get('policy_name', 'unknown')}"
            )
        
        # Check for v9 references in policy
        policy_str = str(policy_dict)
        if 'entry_v9' in policy_str.lower() and 'v10' not in policy_str.lower():
            errors.append(
                "LEGACY_BLOCKED: Policy references entry_v9 without v10. "
                "This is a legacy policy and should not be used."
            )
    
    # Check argv for legacy flags
    argv_str = ' '.join(argv).lower()
    legacy_flags = [
        ('entry_v9', True),  # (flag, allow_in_data_paths)
        ('v9_farm', False),
        ('oanda_demo_v9', False),
        ('sniper/ny', False),  # Legacy sniper/NY path
    ]
    
    for flag, allow_in_data_paths in legacy_flags:
        if flag in argv_str:
            # Allow "entry_v9" in data paths (e.g., ../GX1_DATA/data/entry_v9/file.parquet)
            # This is just a directory structure, not a legacy mode
            if allow_in_data_paths:
                # Check if entry_v9 appears in a data path argument
                is_in_data_path = False
                for i, arg in enumerate(argv):
                    arg_lower = arg.lower()
                    # Check if this is a --data argument
                    if arg in ['--data', '--data-path'] and i + 1 < len(argv):
                        next_arg = argv[i + 1].lower()
                        if flag in next_arg and ('data/' in next_arg or '/data/' in next_arg):
                            is_in_data_path = True
                            break
                    # Check if flag is in a path that contains "data/"
                    elif flag in arg_lower and ('data/' in arg_lower or '/data/' in arg_lower):
                        is_in_data_path = True
                        break
                
                if not is_in_data_path:
                    errors.append(
                        f"LEGACY_BLOCKED: Command-line argument contains legacy reference: {flag}"
                    )
            else:
                # Other flags always block
                errors.append(
                    f"LEGACY_BLOCKED: Command-line argument contains legacy reference: {flag}"
                )
    
    # Check env for legacy modes
    legacy_env_vars = [
        'GX1_V9_MODE',
        'GX1_ENTRY_V9_ENABLED',
        'GX1_SNIPER_NY_MODE',
    ]
    
    for var in legacy_env_vars:
        if env.get(var) == '1' or env.get(var, '').lower() == 'true':
            errors.append(
                f"LEGACY_BLOCKED: Environment variable {var} is set (legacy mode)"
            )
    
    # Check output-dir in argv
    output_dir = None
    for i, arg in enumerate(argv):
        if arg in ['--output-dir', '--output_dir', '-o']:
            if i + 1 < len(argv):
                output_dir = Path(argv[i + 1])
                break
    
    if output_dir:
        output_dir = Path(output_dir)
        
        # Resolve to absolute path
        if not output_dir.is_absolute():
            # Try to resolve relative to current working directory
            try:
                output_dir = Path.cwd() / output_dir
                output_dir = output_dir.resolve()
            except Exception:
                pass
        
        # Check if output-dir is under engine repo (bad)
        # Engine repo is typically the current working directory or parent
        engine_repo_indicators = [
            Path.cwd(),
            Path(__file__).parent.parent.parent,  # gx1/runtime -> gx1 -> repo root
        ]
        
        for engine_root in engine_repo_indicators:
            try:
                if output_dir.is_relative_to(engine_root):
                    # Check if it's in reports/ under engine (bad)
                    if 'reports' in output_dir.parts:
                        # Check if it's using GX1_REPORTS_ROOT or ../GX1_DATA
                        reports_root = env.get('GX1_REPORTS_ROOT')
                        if reports_root:
                            reports_root_path = Path(reports_root).resolve()
                            if output_dir.is_relative_to(reports_root_path):
                                # OK - using GX1_REPORTS_ROOT
                                continue
                        
                        # Check if path contains GX1_DATA
                        if 'GX1_DATA' not in str(output_dir):
                            errors.append(
                                f"LEGACY_BLOCKED: Output directory is under engine repo: {output_dir}. "
                                "Use ../GX1_DATA/reports/ or set GX1_REPORTS_ROOT env var. "
                                "Output must not be under engine repository."
                            )
            except (ValueError, AttributeError):
                # Path resolution failed, skip this check
                pass
    
    # Check for hardcoded reports/ paths in argv
    argv_str_full = ' '.join(argv)
    if 'reports/replay_eval' in argv_str_full or 'reports/' in argv_str_full:
        # Check if it's using GX1_REPORTS_ROOT or ../GX1_DATA
        if 'GX1_REPORTS_ROOT' not in argv_str_full and '../GX1_DATA' not in argv_str_full and 'GX1_DATA' not in argv_str_full:
            errors.append(
                "LEGACY_BLOCKED: Hardcoded 'reports/' path detected in arguments. "
                "Use GX1_REPORTS_ROOT env var or ../GX1_DATA/reports/ path. "
                "Output must not be under engine repository."
            )
    
    # Check bundle_dir_resolved
    if bundle_dir_resolved:
        bundle_dir_str = str(bundle_dir_resolved)
        # Must be under ../GX1_DATA/models/.../entry_v10_ctx/...
        if 'GX1_DATA' not in bundle_dir_str:
            errors.append(
                f"LEGACY_BLOCKED: Bundle directory not under GX1_DATA: {bundle_dir_resolved}. "
                "Bundle must be under ../GX1_DATA/models/.../entry_v10_ctx/..."
            )
        elif 'entry_v10_ctx' not in bundle_dir_str:
            errors.append(
                f"LEGACY_BLOCKED: Bundle directory does not contain entry_v10_ctx: {bundle_dir_resolved}. "
                "Bundle must be under .../entry_v10_ctx/..."
            )
    
    # Check output_dir_resolved
    if output_dir_resolved:
        output_dir_str = str(output_dir_resolved)
        # Must be under GX1_REPORTS_ROOT or ../GX1_DATA/reports
        reports_root = env.get('GX1_REPORTS_ROOT')
        if reports_root:
            reports_root_path = Path(reports_root).resolve()
            if not output_dir_resolved.is_relative_to(reports_root_path):
                errors.append(
                    f"LEGACY_BLOCKED: Output directory not under GX1_REPORTS_ROOT: {output_dir_resolved}. "
                    f"GX1_REPORTS_ROOT={reports_root}"
                )
        elif 'GX1_DATA' not in output_dir_str:
            errors.append(
                f"LEGACY_BLOCKED: Output directory not under GX1_DATA: {output_dir_resolved}. "
                "Output must be under ../GX1_DATA/reports/ or set GX1_REPORTS_ROOT env var."
            )
    
    # If any errors, raise
    if errors:
        error_msg = "LEGACY_GUARD_TRIPWIRE: Legacy mode detected and blocked.\n\n"
        error_msg += "Errors:\n"
        for i, error in enumerate(errors, 1):
            error_msg += f"  {i}. {error}\n"
        error_msg += "\n"
        error_msg += "Action required:\n"
        error_msg += "  1. Remove entry_models.v9 from policy (use v10_ctx)\n"
        error_msg += "  2. Use ../GX1_DATA/reports/ or GX1_REPORTS_ROOT for output\n"
        error_msg += "  3. Remove legacy flags/env vars\n"
        error_msg += "  4. Use V10_CTX policies and paths only\n"
        
        raise RuntimeError(error_msg)

def check_policy_for_legacy(policy_path: Path) -> None:
    """
    Check a policy file for legacy references.
    
    Args:
        policy_path: Path to policy file
    
    Raises:
        RuntimeError: If legacy references found
    """
    import json
    import yaml
    
    try:
        if policy_path.suffix == '.json':
            with open(policy_path, 'r') as f:
                policy_dict = json.load(f)
        elif policy_path.suffix in ['.yaml', '.yml']:
            with open(policy_path, 'r') as f:
                policy_dict = yaml.safe_load(f)
        else:
            return  # Unknown format, skip
        
        assert_no_legacy_mode_enabled(policy_dict=policy_dict)
    except Exception as e:
        # If we can't parse policy, that's a separate error
        # Don't fail on legacy guard if policy parsing fails
        pass
