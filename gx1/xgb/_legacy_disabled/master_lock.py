"""
Master Model Lock - SSoT enforcement for XGB models.

Ensures that only the authorized model artifacts can be loaded.
Hard-fails with diagnostic capsule on any deviation.

Usage:
    from gx1.xgb.master_lock import verify_master_model_lock, MasterLockViolation
    
    # Before loading any XGB model:
    verify_master_model_lock(
        bundle_dir=Path("/path/to/bundle"),
        model_path=Path("/path/to/model.joblib"),
        xgb_mode="universal_multihead",
    )
"""

import datetime
import hashlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

MASTER_LOCK_FILENAME = "MASTER_MODEL_LOCK.json"
VIOLATION_CAPSULE_FILENAME = "MASTER_MODEL_VIOLATION_CAPSULE.json"

# Run modes
RUN_MODE_DEV_EVAL = "DEV_EVAL"  # Allows NO-GO for development/evaluation
RUN_MODE_TRUTH = "TRUTH"  # Requires GO marker (replay/multiyear)
RUN_MODE_PROD = "PROD"  # Requires GO marker (production)


def get_run_mode() -> str:
    """
    Get current run mode from environment.
    
    Defaults to TRUTH (strict) if not set.
    """
    mode = os.environ.get("GX1_RUN_MODE", "TRUTH").upper()
    if mode not in [RUN_MODE_DEV_EVAL, RUN_MODE_TRUTH, RUN_MODE_PROD]:
        log.warning(f"Unknown GX1_RUN_MODE={mode}, defaulting to TRUTH")
        return RUN_MODE_TRUTH
    return mode


class MasterLockViolation(Exception):
    """Raised when master model lock verification fails."""
    pass


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def write_violation_capsule(
    bundle_dir: Path,
    violation_type: str,
    expected: Any,
    actual: Any,
    context: Dict[str, Any],
) -> Path:
    """
    Write violation capsule for diagnostics.
    
    Returns:
        Path to capsule file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    capsule_dir = bundle_dir / "violations"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    
    capsule_path = capsule_dir / f"VIOLATION_{timestamp}.json"
    
    run_mode = context.get("run_mode") or get_run_mode()
    
    capsule = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "violation_type": violation_type,
        "expected": str(expected),
        "actual": str(actual),
        "bundle_dir": str(bundle_dir),
        "run_mode": run_mode,
        "gx1_run_mode_env": os.environ.get("GX1_RUN_MODE", "NOT_SET"),
        "context": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "executable": sys.executable,
            "pid": os.getpid(),
        },
        "stack_trace": traceback.format_stack(),
        **{k: v for k, v in context.items() if k != "run_mode"},
    }
    
    tmp_path = capsule_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(capsule, f, indent=2, default=str)
    tmp_path.rename(capsule_path)
    
    log.error(f"[MASTER_LOCK] Violation capsule written: {capsule_path}")
    
    return capsule_path


def load_master_lock(bundle_dir: Path) -> Dict[str, Any]:
    """
    Load MASTER_MODEL_LOCK.json from bundle directory.
    
    Raises:
        MasterLockViolation: If lock file not found
    """
    lock_path = bundle_dir / MASTER_LOCK_FILENAME
    
    if not lock_path.exists():
        write_violation_capsule(
            bundle_dir=bundle_dir,
            violation_type="MASTER_LOCK_NOT_FOUND",
            expected=str(lock_path),
            actual="File does not exist",
            context={},
        )
        raise MasterLockViolation(
            f"MASTER_LOCK_NOT_FOUND: {lock_path}\n\n"
            f"HOW TO FIX:\n"
            f"1. Generate lock: python3 gx1/scripts/write_master_model_lock.py --bundle-dir {bundle_dir}\n"
            f"2. Ensure the correct model is trained and evaluated"
        )
    
    with open(lock_path, "r") as f:
        return json.load(f)


def verify_master_model_lock(
    bundle_dir: Path,
    model_path: Path,
    xgb_mode: str,
    run_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify that model matches master lock requirements.
    
    Args:
        bundle_dir: Bundle directory containing MASTER_MODEL_LOCK.json
        model_path: Path to the model being loaded
        xgb_mode: XGB mode (must match lock)
        run_mode: Run mode (DEV_EVAL/TRUTH/PROD). If None, reads from GX1_RUN_MODE env.
    
    Returns:
        Dict with verification results including run_mode
    
    Raises:
        MasterLockViolation: On any mismatch
    """
    if run_mode is None:
        run_mode = get_run_mode()
    
    result = {
        "verified": False,
        "lock_path": None,
        "model_sha256": None,
        "run_mode": run_mode,
    }
    
    # Determine GO marker requirement based on run mode
    require_go_marker = run_mode in [RUN_MODE_TRUTH, RUN_MODE_PROD]
    allow_no_go_dev = run_mode == RUN_MODE_DEV_EVAL
    
    log.info(f"[MASTER_LOCK] Run mode: {run_mode} (require_go={require_go_marker}, allow_no_go_dev={allow_no_go_dev})")
    
    # Load master lock
    lock = load_master_lock(bundle_dir)
    result["lock_path"] = str(bundle_dir / MASTER_LOCK_FILENAME)
    
    # Verify xgb_mode
    expected_mode = lock.get("xgb_mode")
    if xgb_mode != expected_mode:
        write_violation_capsule(
            bundle_dir=bundle_dir,
            violation_type="XGB_MODE_MISMATCH",
            expected=expected_mode,
            actual=xgb_mode,
            context={"lock": lock},
        )
        raise MasterLockViolation(
            f"XGB_MODE_MISMATCH: Expected '{expected_mode}', got '{xgb_mode}'\n\n"
            f"HOW TO FIX: Update config to use xgb_mode: {expected_mode}"
        )
    
    # Verify model path
    expected_model_rel = lock.get("model_path_relative")
    actual_model_rel = str(model_path.relative_to(bundle_dir)) if bundle_dir in model_path.parents or bundle_dir == model_path.parent else str(model_path.name)
    
    if expected_model_rel and actual_model_rel != expected_model_rel:
        write_violation_capsule(
            bundle_dir=bundle_dir,
            violation_type="MODEL_PATH_MISMATCH",
            expected=expected_model_rel,
            actual=actual_model_rel,
            context={"lock": lock, "model_path": str(model_path)},
        )
        raise MasterLockViolation(
            f"MODEL_PATH_MISMATCH:\n"
            f"  Expected: {expected_model_rel}\n"
            f"  Actual:   {actual_model_rel}\n\n"
            f"HOW TO FIX: Use the correct model path from MASTER_MODEL_LOCK.json"
        )
    
    # Verify model exists
    if not model_path.exists():
        write_violation_capsule(
            bundle_dir=bundle_dir,
            violation_type="MODEL_NOT_FOUND",
            expected=str(model_path),
            actual="File does not exist",
            context={"lock": lock},
        )
        raise MasterLockViolation(
            f"MODEL_NOT_FOUND: {model_path}\n\n"
            f"HOW TO FIX:\n"
            f"1. Ensure model is trained: python3 gx1/scripts/train_xgb_universal_multihead_v2.py\n"
            f"2. Ensure model is in correct location"
        )
    
    # Verify model SHA256
    actual_sha = compute_file_sha256(model_path)
    expected_sha = lock.get("model_sha256")
    result["model_sha256"] = actual_sha
    
    if expected_sha and actual_sha != expected_sha:
        write_violation_capsule(
            bundle_dir=bundle_dir,
            violation_type="MODEL_SHA256_MISMATCH",
            expected=expected_sha,
            actual=actual_sha,
            context={"lock": lock, "model_path": str(model_path)},
        )
        raise MasterLockViolation(
            f"MODEL_SHA256_MISMATCH:\n"
            f"  Expected: {expected_sha[:16]}...\n"
            f"  Actual:   {actual_sha[:16]}...\n\n"
            f"HOW TO FIX:\n"
            f"1. Model has changed - regenerate lock:\n"
            f"   python3 gx1/scripts/write_master_model_lock.py --bundle-dir {bundle_dir}\n"
            f"2. Or restore the correct model version"
        )
    
    # Verify meta SHA256
    meta_path = model_path.parent / lock.get("meta_path_relative", "xgb_universal_multihead_v2_meta.json")
    if meta_path.exists():
        actual_meta_sha = compute_file_sha256(meta_path)
        expected_meta_sha = lock.get("meta_sha256")
        
        if expected_meta_sha and actual_meta_sha != expected_meta_sha:
            write_violation_capsule(
                bundle_dir=bundle_dir,
                violation_type="META_SHA256_MISMATCH",
                expected=expected_meta_sha,
                actual=actual_meta_sha,
                context={"lock": lock, "meta_path": str(meta_path)},
            )
            raise MasterLockViolation(
                f"META_SHA256_MISMATCH:\n"
                f"  Expected: {expected_meta_sha[:16]}...\n"
                f"  Actual:   {actual_meta_sha[:16]}...\n\n"
                f"HOW TO FIX: Regenerate lock or restore correct meta file"
            )
    
    # Verify schema hash
    expected_schema = lock.get("schema_hash")
    if expected_schema:
        # Load meta to check schema
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            actual_schema = meta.get("schema_hash")
            if actual_schema != expected_schema:
                write_violation_capsule(
                    bundle_dir=bundle_dir,
                    violation_type="SCHEMA_HASH_MISMATCH",
                    expected=expected_schema,
                    actual=actual_schema,
                    context={"lock": lock},
                )
                raise MasterLockViolation(
                    f"SCHEMA_HASH_MISMATCH:\n"
                    f"  Expected: {expected_schema}\n"
                    f"  Actual:   {actual_schema}\n\n"
                    f"HOW TO FIX: Ensure model was trained with correct feature contract"
                )
    
    # Verify GO marker if required
    if require_go_marker:
        go_marker_name = lock.get("go_marker_filename", "XGB_MULTIHEAD_GO_MARKER.json")
        go_marker_path = bundle_dir / go_marker_name
        no_go_path = bundle_dir / lock.get("no_go_marker_filename", "XGB_MULTIHEAD_NO_GO.json")
        
        if not go_marker_path.exists():
            if no_go_path.exists() and allow_no_go_dev:
                log.warning(
                    f"[MASTER_LOCK] ⚠️  Running with NO-GO model (DEV_EVAL mode allowed)"
                )
                result["go_marker_status"] = "NO-GO_ALLOWED_DEV"
            elif no_go_path.exists():
                with open(no_go_path, "r") as f:
                    no_go_data = json.load(f)
                issues = no_go_data.get("issues", [])
                write_violation_capsule(
                    bundle_dir=bundle_dir,
                    violation_type="MODEL_NOT_PROMOTED",
                    expected="GO marker",
                    actual=f"NO-GO: {issues}",
                    context={
                        "lock": lock,
                        "no_go_data": no_go_data,
                        "run_mode": run_mode,
                    },
                )
                raise MasterLockViolation(
                    f"MODEL_NOT_PROMOTED: Model has NO-GO marker (run_mode={run_mode})\n"
                    f"Issues: {issues}\n\n"
                    f"HOW TO FIX:\n"
                    f"1. Improve model and re-evaluate\n"
                    f"2. Or set GX1_RUN_MODE=DEV_EVAL for development only"
                )
            else:
                write_violation_capsule(
                    bundle_dir=bundle_dir,
                    violation_type="GO_MARKER_NOT_FOUND",
                    expected=str(go_marker_path),
                    actual="Not found",
                    context={"lock": lock, "run_mode": run_mode},
                )
                raise MasterLockViolation(
                    f"GO_MARKER_NOT_FOUND: {go_marker_path} (run_mode={run_mode})\n\n"
                    f"HOW TO FIX:\n"
                    f"1. Evaluate model: python3 gx1/scripts/eval_xgb_multihead_v2_multiyear.py\n"
                    f"2. If GO, marker will be created automatically\n"
                    f"3. Or set GX1_RUN_MODE=DEV_EVAL for development"
                )
        else:
            # Verify GO marker SHA if locked
            expected_go_sha = lock.get("go_marker_sha256")
            if expected_go_sha:
                actual_go_sha = compute_file_sha256(go_marker_path)
                if actual_go_sha != expected_go_sha:
                    write_violation_capsule(
                        bundle_dir=bundle_dir,
                        violation_type="GO_MARKER_SHA256_MISMATCH",
                        expected=expected_go_sha,
                        actual=actual_go_sha,
                        context={"lock": lock, "run_mode": run_mode},
                    )
                    raise MasterLockViolation(
                        f"GO_MARKER_SHA256_MISMATCH:\n"
                        f"  Expected: {expected_go_sha[:16]}...\n"
                        f"  Actual:   {actual_go_sha[:16]}...\n\n"
                        f"HOW TO FIX: Regenerate lock or restore correct GO marker"
                    )
            result["go_marker_status"] = "GO_VERIFIED"
    else:
        result["go_marker_status"] = "NOT_REQUIRED"
    
    result["verified"] = True
    log.info(
        f"[MASTER_LOCK] ✅ Verified: {model_path.name} "
        f"(SHA: {actual_sha[:16]}..., run_mode={run_mode}, go_status={result.get('go_marker_status')})"
    )
    
    return result


def get_truth_set_from_lock(bundle_dir: Path) -> Dict[str, Any]:
    """
    Build truth set from MASTER_MODEL_LOCK.json.
    
    Returns:
        Dict with truth set files and their expected SHAs
    """
    lock = load_master_lock(bundle_dir)
    
    truth_set = {
        "source": "MASTER_MODEL_LOCK",
        "bundle_dir": str(bundle_dir),
        "files": {},
    }
    
    # Model file
    model_rel = lock.get("model_path_relative")
    if model_rel:
        model_path = bundle_dir / model_rel
        truth_set["files"][str(model_path)] = {
            "expected_sha256": lock.get("model_sha256"),
            "type": "xgb_model",
        }
    
    # Meta file
    meta_rel = lock.get("meta_path_relative")
    if meta_rel:
        meta_path = bundle_dir / meta_rel
        truth_set["files"][str(meta_path)] = {
            "expected_sha256": lock.get("meta_sha256"),
            "type": "xgb_meta",
        }
    
    # Contract files
    for contract_key in ["feature_contract", "sanitizer_config", "output_contract"]:
        contract_info = lock.get("contracts", {}).get(contract_key, {})
        if contract_info.get("path"):
            truth_set["files"][contract_info["path"]] = {
                "expected_sha256": contract_info.get("sha256"),
                "type": f"contract_{contract_key}",
            }
    
    # GO marker
    go_marker = lock.get("go_marker_filename")
    if go_marker:
        go_path = bundle_dir / go_marker
        truth_set["files"][str(go_path)] = {
            "expected_sha256": lock.get("go_marker_sha256"),
            "type": "go_marker",
        }
    
    return truth_set
