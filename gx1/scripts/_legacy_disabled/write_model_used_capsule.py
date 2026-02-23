#!/usr/bin/env python3
"""
Write MODEL_USED_CAPSULE.json - SSoT for XGB model usage.

This script loads policy/bundle config the same way as replay would,
loads XGB pre-models, and writes a capsule file proving which models
were actually used.

Usage:
    python3 gx1/scripts/write_model_used_capsule.py
    python3 gx1/scripts/write_model_used_capsule.py --bundle-dir <path> --policy-id <id>
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import yaml
    from joblib import load as joblib_load
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    
    if "GX1_DATA_DATA" in os.environ:
        path = Path(os.environ["GX1_DATA_DATA"]).parent
        if path.exists():
            return path
    
    # Fallback
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_xgb_model(
    session: str,
    model_path: str,
    workspace_root: Path,
    source: str,
) -> tuple[Optional[Path], Optional[Any]]:
    """
    Load XGB model and return path and model.
    
    Returns: (model_path_resolved, model) or (None, None) if not found
    """
    if not model_path:
        return None, None
    
    model_path_obj = Path(model_path)
    # Resolve paths relative to workspace root
    if not model_path_obj.is_absolute():
        model_path_resolved = workspace_root / model_path_obj
    else:
        model_path_resolved = model_path_obj
    
    if model_path_resolved.exists():
        model = joblib_load(model_path_resolved)
        print(f"  ✅ Loaded XGB model for {session} from {source}: {model_path_resolved}")
        return model_path_resolved, model
    else:
        print(f"  ⚠️  XGB model not found for {session} from {source}: {model_path_resolved}")
        return None, None


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Write MODEL_USED_CAPSULE.json - SSoT for XGB model usage"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Bundle directory override (default: from policy)"
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default=None,
        help="Policy ID override (default: from policy file)"
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Policy YAML path (default: find active policy)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: GX1_DATA/reports/repo_audit/MODEL_USED_CAPSULE_<timestamp>)"
    )
    parser.add_argument(
        "--require-universal",
        type=int,
        default=1,
        help="Hard fail if XGB model path is under /entry_v10/ (default: 1 for universal mode)"
    )
    parser.add_argument(
        "--xgb-mode",
        type=str,
        choices=["universal", "session"],
        default="session",
        help="XGB mode: 'universal' (single model) or 'session' (EU/US/OVERLAP)"
    )
    
    args = parser.parse_args()
    
    # Resolve GX1_DATA
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA resolved to: {gx1_data}")
    
    # Resolve workspace root
    workspace_root = WORKSPACE_ROOT
    
    # Find policy path if not provided
    if args.policy_path is None:
        # Try to find active policy with v10_ctx config
        policy_candidates = [
            WORKSPACE_ROOT / "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml",
            WORKSPACE_ROOT / "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1" / "GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml",
            WORKSPACE_ROOT / "gx1/configs/policies/active" / "GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml",
            WORKSPACE_ROOT / "gx1/configs/policies" / "sniper_trial160_prod.json",
        ]
        
        for candidate in policy_candidates:
            if candidate.exists():
                args.policy_path = candidate
                break
        
        if args.policy_path is None:
            print("ERROR: No policy path provided and no default policy found")
            print("  Use --policy-path to specify policy file")
            print("  Example: --policy-path gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml")
            return 1
    
    print(f"Loading policy: {args.policy_path}")
    policy = load_yaml_config(args.policy_path)
    
    # Get policy_id
    policy_id = args.policy_id or policy.get("policy_id") or "unknown"
    print(f"Policy ID: {policy_id}")
    
    # Get entry_models.v10_ctx config
    entry_models_cfg = policy.get("entry_models", {})
    entry_v10_ctx_cfg = entry_models_cfg.get("v10_ctx", {})
    
    if not entry_v10_ctx_cfg:
        print("ERROR: entry_models.v10_ctx config not found in policy")
        return 1
    
    # Resolve bundle_dir
    bundle_dir = args.bundle_dir
    if bundle_dir is None:
        # Priority: ENV > Policy
        if os.getenv("GX1_BUNDLE_DIR"):
            bundle_dir = Path(os.getenv("GX1_BUNDLE_DIR")).resolve()
        else:
            bundle_dir_str = entry_v10_ctx_cfg.get("bundle_dir")
            if not bundle_dir_str:
                print("ERROR: bundle_dir is required in entry_models.v10_ctx config or GX1_BUNDLE_DIR env var")
                return 1
            
            bundle_dir = Path(bundle_dir_str)
            if not bundle_dir.is_absolute():
                # Try multiple resolution strategies:
                # 1. Relative to GX1_DATA/models
                candidate1 = gx1_data / "models" / bundle_dir
                if candidate1.exists():
                    bundle_dir = candidate1.resolve()
                else:
                    # 2. Relative to workspace root
                    candidate2 = workspace_root / bundle_dir
                    if candidate2.exists():
                        bundle_dir = candidate2.resolve()
                    else:
                        # 3. Relative to policy file's directory (last resort)
                        policy_dir = args.policy_path.resolve().parent
                        bundle_dir = (policy_dir / bundle_dir).resolve()
            else:
                bundle_dir = bundle_dir.resolve()
    
    if not bundle_dir.exists():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        print(f"  Tried resolving from: {entry_v10_ctx_cfg.get('bundle_dir')}")
        print(f"  GX1_DATA: {gx1_data}")
        print(f"  Workspace: {workspace_root}")
        return 1
    
    print(f"Bundle directory: {bundle_dir}")
    
    # Load XGB models (same logic as oanda_demo_runner)
    xgb_models = {}
    xgb_model_paths = {}
    
    # Check entry_config first
    entry_config_path = policy.get("entry_config", "")
    if entry_config_path:
        try:
            entry_config = load_yaml_config(Path(entry_config_path))
            entry_models_cfg_in_entry = entry_config.get("entry_models", {})
            xgb_cfg_from_entry = entry_models_cfg_in_entry.get("xgb", {})
            if xgb_cfg_from_entry:
                print("Loading XGB models from entry_config...")
                for session in ["EU", "US", "OVERLAP"]:
                    model_path_str = xgb_cfg_from_entry.get(f"{session.lower()}_model_path")
                    if model_path_str:
                        path, model = load_xgb_model(session, model_path_str, workspace_root, "entry_config")
                        if path and model:
                            xgb_models[session] = model
                            xgb_model_paths[session] = str(path.resolve())
        except Exception as e:
            print(f"  WARNING: Failed to load XGB models from entry_config: {e}")
    
    # Check v10_ctx config
    if not xgb_models:
        xgb_cfg_from_v10_ctx = entry_v10_ctx_cfg.get("xgb", {})
        if xgb_cfg_from_v10_ctx:
            print("Loading XGB models from v10_ctx config...")
            for session in ["EU", "US", "OVERLAP"]:
                model_path_str = xgb_cfg_from_v10_ctx.get(f"{session.lower()}_model_path")
                if model_path_str:
                    path, model = load_xgb_model(session, model_path_str, workspace_root, "v10_ctx")
                    if path and model:
                        xgb_models[session] = model
                        xgb_model_paths[session] = str(path.resolve())
    
    # Check policy config
    if not xgb_models:
        xgb_cfg_from_policy = entry_models_cfg.get("xgb", {})
        if xgb_cfg_from_policy:
            print("Loading XGB models from policy config...")
            for session in ["EU", "US", "OVERLAP"]:
                model_path_str = xgb_cfg_from_policy.get(f"{session.lower()}_model_path")
                if model_path_str:
                    path, model = load_xgb_model(session, model_path_str, workspace_root, "policy")
                    if path and model:
                        xgb_models[session] = model
                        xgb_model_paths[session] = str(path.resolve())
    
    # Hard fail if no models loaded
    if not xgb_models:
        print()
        print("=" * 60)
        print("FATAL: No XGB models loaded")
        print("=" * 60)
        print("Cannot write capsule without XGB models.")
        print("Check policy/entry_config for XGB model paths.")
        return 1
    
    print(f"\nLoaded {len(xgb_models)} XGB models: {list(xgb_models.keys())}")
    
    # Check universal requirement
    if args.require_universal:
        for session, model_path_str in xgb_model_paths.items():
            if "/entry_v10/" in model_path_str and "/entry_v10_ctx/" not in model_path_str:
                print()
                print("=" * 60)
                print("FATAL: Legacy XGB model path detected")
                print("=" * 60)
                print(f"Session: {session}")
                print(f"Path: {model_path_str}")
                print("Universal mode requires entry_v10_ctx bundles, not entry_v10.")
                return 1
    
    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = gx1_data / "reports" / "repo_audit" / f"MODEL_USED_CAPSULE_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Compute SHA256 for each XGB model
    xgb_models_info = {}
    for session, model_path_str in xgb_model_paths.items():
        model_path = Path(model_path_str)
        if model_path.exists():
            model_sha256 = compute_file_sha256(model_path)
            xgb_models_info[session] = {
                "selected_xgb_model_path": str(model_path.resolve()),
                "selected_xgb_model_sha256": model_sha256,
                "selected_model_kind": "xgb_pre",
            }
            print(f"  {session}: {model_path.name} (SHA256: {model_sha256[:16]}...)")
    
    # Get bundle SHA256 if available
    bundle_sha256 = None
    model_state_path = bundle_dir / "model_state_dict.pt"
    if model_state_path.exists():
        bundle_sha256 = compute_file_sha256(model_state_path)
        print(f"Bundle SHA256: {bundle_sha256[:16]}...")
    
    # Get policy SHA256
    policy_sha256 = compute_file_sha256(args.policy_path)
    print(f"Policy SHA256: {policy_sha256[:16]}...")
    
    # Get git commit
    git_commit = None
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:12]
    except Exception:
        pass
    
    # Load contract SHAs for provenance
    feature_contract_sha = None
    sanitizer_sha = None
    try:
        feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
        if feature_contract_path.exists():
            feature_contract_sha = compute_file_sha256(feature_contract_path)
        
        sanitizer_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
        if sanitizer_path.exists():
            sanitizer_sha = compute_file_sha256(sanitizer_path)
    except Exception:
        pass
    
    # Build capsule
    capsule = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "run_id": f"capsule_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "bundle_dir": str(bundle_dir.resolve()),
        "bundle_sha256": bundle_sha256,
        "xgb_mode": args.xgb_mode,
        "xgb_models": xgb_models_info,
        "contracts": {
            "feature_contract_sha256": feature_contract_sha,
            "sanitizer_sha256": sanitizer_sha,
        },
        "provenance": {
            "git_commit": git_commit,
            "worker_pid": os.getpid(),
        },
    }
    
    # Atomic write
    capsule_path = output_dir / "MODEL_USED_CAPSULE.json"
    tmp_path = capsule_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(capsule, f, indent=2)
    tmp_path.rename(capsule_path)
    
    print()
    print("=" * 60)
    print("✅ MODEL_USED_CAPSULE.json written")
    print("=" * 60)
    print(f"Path: {capsule_path}")
    print(f"XGB models: {list(xgb_models_info.keys())}")
    print()
    print("To verify, run:")
    print(f"  find \"{gx1_data}\" -name \"MODEL_USED_CAPSULE.json\" | tail -n 5")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
