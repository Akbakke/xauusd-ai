"""
XGB Universal Model Promotion Gate.

Enforces that xgb_mode: universal can only run if the model
has been evaluated and received a GO marker with matching SHA256.

Usage:
    from gx1.xgb.promotion_gate import verify_xgb_promotion_gate
    
    # In runtime/replay before loading XGB model:
    verify_xgb_promotion_gate(
        xgb_mode="universal",
        model_path=Path("/path/to/xgb_universal_v1.joblib"),
        bundle_dir=Path("/path/to/bundle"),
    )
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Legacy patterns that are ALWAYS forbidden
LEGACY_PATH_PATTERNS = [
    r"/entry_v10/",  # Old entry_v10 dir (not entry_v10_ctx)
    r"xgb_entry_.*_v10\.joblib$",  # Legacy session model naming
]


class PromotionGateError(Exception):
    """Raised when promotion gate check fails."""
    pass


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def is_legacy_path(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Check if a path matches legacy patterns.
    
    Returns:
        (is_legacy, matched_pattern)
    """
    path_str = str(path)
    for pattern in LEGACY_PATH_PATTERNS:
        if re.search(pattern, path_str):
            return True, pattern
    return False, None


def verify_legacy_path_invariant(model_path: Path) -> None:
    """
    Hard-fail if model path is legacy.
    
    This is a universal invariant that applies regardless of xgb_mode.
    """
    is_legacy, pattern = is_legacy_path(model_path)
    if is_legacy:
        raise PromotionGateError(
            f"LEGACY_PATH_INVARIANT_FAIL: Model path '{model_path}' "
            f"matches forbidden legacy pattern: {pattern}\n\n"
            f"HOW TO FIX:\n"
            f"1. Use models from entry_v10_ctx bundle\n"
            f"2. Ensure path is under .../entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/\n"
            f"3. Do NOT use xgb_entry_*_v10.joblib naming"
        )


def verify_xgb_promotion_gate(
    xgb_mode: str,
    model_path: Path,
    bundle_dir: Path,
    feature_contract_path: Optional[Path] = None,
    sanitizer_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Verify XGB promotion gate requirements.
    
    Args:
        xgb_mode: "universal" or "session"
        model_path: Path to the XGB model file
        bundle_dir: Path to the bundle directory
        feature_contract_path: Optional path to feature contract
        sanitizer_config_path: Optional path to sanitizer config
    
    Returns:
        Dict with verification results
    
    Raises:
        PromotionGateError: If verification fails
    """
    result = {
        "xgb_mode": xgb_mode,
        "model_path": str(model_path),
        "bundle_dir": str(bundle_dir),
        "legacy_check_passed": False,
        "go_marker_required": xgb_mode == "universal",
        "go_marker_found": False,
        "sha_verified": False,
    }
    
    # Step 1: Always check legacy path invariant
    verify_legacy_path_invariant(model_path)
    result["legacy_check_passed"] = True
    
    # Step 2: For universal mode, require GO marker
    if xgb_mode == "universal":
        go_marker_path = bundle_dir / "XGB_UNIVERSAL_GO_MARKER.json"
        
        if not go_marker_path.exists():
            # Check if NO-GO marker exists
            no_go_path = bundle_dir / "XGB_UNIVERSAL_NO_GO.json"
            if no_go_path.exists():
                with open(no_go_path, "r") as f:
                    no_go_data = json.load(f)
                issues = no_go_data.get("issues", ["Unknown"])
                raise PromotionGateError(
                    f"XGB_UNIVERSAL_NOT_PROMOTED: Model has NO-GO marker\n"
                    f"Issues: {issues}\n\n"
                    f"HOW TO FIX:\n"
                    f"1. Improve model training (better labels, hyperparams)\n"
                    f"2. Run: python3 gx1/scripts/eval_xgb_universal_multiyear.py\n"
                    f"3. Fix issues until eval returns GO verdict\n"
                    f"4. Retry with xgb_mode: universal"
                )
            else:
                raise PromotionGateError(
                    f"XGB_UNIVERSAL_NOT_PROMOTED: No GO marker found at {go_marker_path}\n\n"
                    f"HOW TO FIX:\n"
                    f"1. Train universal model: python3 gx1/scripts/train_xgb_universal_v1_multiyear.py\n"
                    f"2. Evaluate model: python3 gx1/scripts/eval_xgb_universal_multiyear.py\n"
                    f"3. If GO verdict: marker will be written automatically\n"
                    f"4. If NO-GO: fix issues and re-train"
                )
        
        result["go_marker_found"] = True
        
        # Load and verify GO marker
        with open(go_marker_path, "r") as f:
            marker = json.load(f)
        
        # Verify model SHA256
        if not model_path.exists():
            raise PromotionGateError(
                f"XGB_MODEL_NOT_FOUND: {model_path}"
            )
        
        actual_model_sha = compute_file_sha256(model_path)
        expected_model_sha = marker.get("model_sha256")
        
        if actual_model_sha != expected_model_sha:
            raise PromotionGateError(
                f"XGB_SHA256_MISMATCH: Model has changed since GO marker was written\n"
                f"Expected: {expected_model_sha[:16]}...\n"
                f"Actual:   {actual_model_sha[:16]}...\n\n"
                f"HOW TO FIX:\n"
                f"1. Re-run eval: python3 gx1/scripts/eval_xgb_universal_multiyear.py\n"
                f"2. If GO: new marker will be written with updated SHA\n"
                f"3. If NO-GO: fix issues before using universal mode"
            )
        
        result["sha_verified"] = True
        result["model_sha256"] = actual_model_sha
        result["go_marker"] = marker
        
        # Verify contract/sanitizer SHA if provided
        if feature_contract_path and feature_contract_path.exists():
            actual_fc_sha = compute_file_sha256(feature_contract_path)
            expected_fc_sha = marker.get("feature_contract_sha256")
            if expected_fc_sha and actual_fc_sha != expected_fc_sha:
                log.warning(
                    f"Feature contract SHA mismatch: {actual_fc_sha[:16]} vs {expected_fc_sha[:16]}"
                )
        
        if sanitizer_config_path and sanitizer_config_path.exists():
            actual_san_sha = compute_file_sha256(sanitizer_config_path)
            expected_san_sha = marker.get("sanitizer_config_sha256")
            if expected_san_sha and actual_san_sha != expected_san_sha:
                log.warning(
                    f"Sanitizer config SHA mismatch: {actual_san_sha[:16]} vs {expected_san_sha[:16]}"
                )
        
        log.info(
            f"[PROMOTION_GATE] ✅ Universal XGB verified: {model_path.name} "
            f"(SHA: {actual_model_sha[:16]}...)"
        )
    
    else:
        # Session mode - just verify not legacy
        log.info(f"[PROMOTION_GATE] Session mode, legacy check passed for {model_path.name}")
    
    return result


def get_go_marker_status(bundle_dir: Path, xgb_mode: str = "universal") -> Dict[str, Any]:
    """
    Get current GO/NO-GO marker status for a bundle.
    
    Args:
        bundle_dir: Bundle directory path
        xgb_mode: "universal" or "universal_multihead"
    
    Returns:
        Dict with status info
    """
    if xgb_mode == "universal_multihead":
        go_marker_path = bundle_dir / "XGB_MULTIHEAD_GO_MARKER.json"
        no_go_path = bundle_dir / "XGB_MULTIHEAD_NO_GO.json"
    else:
        go_marker_path = bundle_dir / "XGB_UNIVERSAL_GO_MARKER.json"
        no_go_path = bundle_dir / "XGB_UNIVERSAL_NO_GO.json"
    
    status = {
        "bundle_dir": str(bundle_dir),
        "xgb_mode": xgb_mode,
        "go_marker_exists": go_marker_path.exists(),
        "no_go_marker_exists": no_go_path.exists(),
        "status": "unknown",
    }
    
    if go_marker_path.exists():
        with open(go_marker_path, "r") as f:
            marker = json.load(f)
        status["status"] = "GO"
        status["go_marker"] = marker
    elif no_go_path.exists():
        with open(no_go_path, "r") as f:
            marker = json.load(f)
        status["status"] = "NO-GO"
        status["no_go_marker"] = marker
        status["issues"] = marker.get("issues", [])
    else:
        status["status"] = "NOT_EVALUATED"
    
    return status


def verify_xgb_multihead_promotion_gate(
    model_path: Path,
    bundle_dir: Path,
    feature_contract_path: Optional[Path] = None,
    sanitizer_config_path: Optional[Path] = None,
    output_contract_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Verify XGB multihead promotion gate requirements.
    
    Args:
        model_path: Path to the multihead model file
        bundle_dir: Path to the bundle directory
        feature_contract_path: Optional path to feature contract
        sanitizer_config_path: Optional path to sanitizer config
        output_contract_path: Optional path to multihead output contract
    
    Returns:
        Dict with verification results
    
    Raises:
        PromotionGateError: If verification fails
    """
    result = {
        "xgb_mode": "universal_multihead",
        "model_path": str(model_path),
        "bundle_dir": str(bundle_dir),
        "legacy_check_passed": False,
        "go_marker_required": True,
        "go_marker_found": False,
        "sha_verified": False,
    }
    
    # Step 1: Check legacy path invariant
    verify_legacy_path_invariant(model_path)
    result["legacy_check_passed"] = True
    
    # Step 2: Require GO marker
    go_marker_path = bundle_dir / "XGB_MULTIHEAD_GO_MARKER.json"
    no_go_path = bundle_dir / "XGB_MULTIHEAD_NO_GO.json"
    
    if not go_marker_path.exists():
        if no_go_path.exists():
            with open(no_go_path, "r") as f:
                no_go_data = json.load(f)
            issues = no_go_data.get("issues", ["Unknown"])
            raise PromotionGateError(
                f"XGB_MULTIHEAD_NOT_PROMOTED: Model has NO-GO marker\n"
                f"Issues: {issues}\n\n"
                f"HOW TO FIX:\n"
                f"1. Improve model (better labels, threshold, class balance)\n"
                f"2. Run: python3 gx1/scripts/eval_xgb_multihead_v2_multiyear.py\n"
                f"3. Fix issues until eval returns GO verdict"
            )
        else:
            raise PromotionGateError(
                f"XGB_MULTIHEAD_NOT_PROMOTED: No GO marker found at {go_marker_path}\n\n"
                f"HOW TO FIX:\n"
                f"1. Train: python3 gx1/scripts/train_xgb_universal_multihead_v2.py\n"
                f"2. Evaluate: python3 gx1/scripts/eval_xgb_multihead_v2_multiyear.py\n"
                f"3. If GO verdict: marker will be written automatically"
            )
    
    result["go_marker_found"] = True
    
    # Load and verify GO marker
    with open(go_marker_path, "r") as f:
        marker = json.load(f)
    
    # Verify model SHA256
    if not model_path.exists():
        raise PromotionGateError(f"XGB_MODEL_NOT_FOUND: {model_path}")
    
    actual_model_sha = compute_file_sha256(model_path)
    expected_model_sha = marker.get("model_sha256")
    
    if actual_model_sha != expected_model_sha:
        raise PromotionGateError(
            f"XGB_SHA256_MISMATCH: Model has changed since GO marker was written\n"
            f"Expected: {expected_model_sha[:16]}...\n"
            f"Actual:   {actual_model_sha[:16]}...\n\n"
            f"HOW TO FIX:\n"
            f"Re-run eval: python3 gx1/scripts/eval_xgb_multihead_v2_multiyear.py"
        )
    
    result["sha_verified"] = True
    result["model_sha256"] = actual_model_sha
    result["go_marker"] = marker
    result["sessions"] = marker.get("sessions", [])
    
    log.info(
        f"[PROMOTION_GATE] ✅ Multihead XGB verified: {model_path.name} "
        f"(SHA: {actual_model_sha[:16]}..., sessions: {result['sessions']})"
    )
    
    return result
