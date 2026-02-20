"""
Bundle Identity Gate for TRUTH/SMOKE

Enforces canonical bundle directory in TRUTH/SMOKE mode and verifies bundle SHA256 matches expected.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from gx1.utils.ssot_hash import compute_bundle_sha256, resolve_artifact_paths_from_policy, sha256_file

log = logging.getLogger(__name__)


def _choose_report_dir(output_dir: Path) -> Path:
    """Choose report directory (output_dir or /tmp fallback)."""
    if output_dir.exists() or output_dir.parent.exists():
        return output_dir
    return Path("/tmp")


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON file atomically."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    """Write text file atomically."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(content)
    tmp_path.replace(path)


def _render_md(
    canonical_bundle_dir: Path,
    actual_bundle_dir: Path,
    policy_path: Path,
    computed_sha256: str,
    expected_sha256: Optional[str],
    lock_sha256: Optional[str],
) -> str:
    """Render BUNDLE_IDENTITY.md."""
    lines = []
    lines.append("# BUNDLE_IDENTITY")
    lines.append("")
    lines.append("## Canonical Bundle Directory")
    lines.append("")
    lines.append(f"- **Canonical:** `{canonical_bundle_dir}`")
    lines.append(f"- **Actual Used:** `{actual_bundle_dir}`")
    lines.append(f"- **Match:** {'✅ YES' if actual_bundle_dir.resolve() == canonical_bundle_dir.resolve() else '❌ NO'}")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(f"- **Path:** `{policy_path}`")
    lines.append("")
    lines.append("## Bundle SHA256")
    lines.append("")
    lines.append(f"- **Computed:** `{computed_sha256}`")
    if expected_sha256:
        lines.append(f"- **Expected (from lock):** `{expected_sha256}`")
        lines.append(f"- **Match:** {'✅ YES' if computed_sha256 == expected_sha256 else '❌ NO'}")
    if lock_sha256:
        lines.append(f"- **Lock SHA256:** `{lock_sha256}`")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **GX1_CANONICAL_BUNDLE_DIR:** `{os.getenv('GX1_CANONICAL_BUNDLE_DIR', 'NOT SET')}`")
    lines.append("")
    return "\n".join(lines)


def _read_lock_sha256(bundle_dir: Path) -> Optional[str]:
    """Read expected SHA256 from MASTER_MODEL_LOCK.json if it exists.
    
    Returns model_sha256 from lock (which is the XGB joblib SHA256).
    Note: This is different from compute_bundle_sha256() which computes a hash over policy + artifacts.
    For verification, we compare computed bundle_sha256 with what's in lock, but if lock doesn't have
    bundle_sha256, we fall back to comparing joblib SHA256 with model_sha256.
    """
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        return None
    
    try:
        with open(lock_path) as f:
            lock_data = json.load(f)
            # Return model_sha256 (XGB joblib SHA256) for comparison
            return lock_data.get("model_sha256")
    except Exception as e:
        log.warning(f"Failed to read lock file {lock_path}: {e}")
        return None


def run_bundle_identity_gate_or_fatal(
    output_dir: Path,
    truth_or_smoke: bool,
    policy_path: Path,
    bundle_dir_override: Optional[Path] = None,
) -> None:
    """
    Run bundle identity gate over bundle directory and hard-fail (exit code 2) in TRUTH/SMOKE if mismatch.
    
    Args:
        output_dir: Output directory for reports
        truth_or_smoke: Whether running in TRUTH/SMOKE mode
        policy_path: Path to policy YAML
        bundle_dir_override: Optional bundle_dir override (CLI or ENV)
    """
    if not truth_or_smoke:
        return
    
    # Get canonical bundle dir from env
    canonical_bundle_dir_str = os.getenv("GX1_CANONICAL_BUNDLE_DIR")
    if not canonical_bundle_dir_str:
        msg = "GX1_CANONICAL_BUNDLE_DIR must be set in TRUTH/SMOKE mode"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(output_dir),
        }
        report_dir = _choose_report_dir(output_dir)
        fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    canonical_bundle_dir = Path(canonical_bundle_dir_str).resolve()
    if not canonical_bundle_dir.exists():
        msg = f"Canonical bundle directory does not exist: {canonical_bundle_dir}"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "canonical_bundle_dir": str(canonical_bundle_dir),
            "sys.executable": sys.executable,
            "cwd": str(Path.cwd()),
        }
        report_dir = _choose_report_dir(output_dir)
        fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    # Resolve actual bundle dir.
    #
    # Signal-only TRUTH: policy bundle_dir may be stale or intentionally non-existent; the canonical truth
    # is provided via GX1_CANONICAL_BUNDLE_DIR (and/or GX1_CANONICAL_TRUTH_FILE in the caller).
    # In TRUTH/SMOKE we therefore accept canonical as "actual used" when a canonical override is present.
    resolved_paths: Dict[str, Any] = {}
    actual_bundle_dir: Optional[Path] = None
    if bundle_dir_override is not None:
        actual_bundle_dir = Path(bundle_dir_override).resolve()
        resolved_paths["bundle_dir"] = actual_bundle_dir
    else:
        truth_file = os.getenv("GX1_CANONICAL_TRUTH_FILE") or ""
        if truth_file:
            actual_bundle_dir = canonical_bundle_dir.resolve()
            resolved_paths["bundle_dir"] = actual_bundle_dir
        else:
            try:
                resolved_paths = resolve_artifact_paths_from_policy(policy_path, bundle_dir_override)
                actual_bundle_dir = resolved_paths.get("bundle_dir")
                if not actual_bundle_dir:
                    msg = "bundle_dir not found in policy or overrides"
                    capsule = {
                        "status": "FAIL",
                        "message": msg,
                        "policy_path": str(policy_path),
                        "canonical_bundle_dir": str(canonical_bundle_dir),
                    }
                    report_dir = _choose_report_dir(output_dir)
                    fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
                    _write_json_atomic(fatal_path, capsule)
                    raise SystemExit(2)
                actual_bundle_dir = actual_bundle_dir.resolve()
            except Exception as e:
                msg = f"Failed to resolve bundle_dir from policy: {e}"
                capsule = {
                    "status": "FAIL",
                    "message": msg,
                    "policy_path": str(policy_path),
                    "canonical_bundle_dir": str(canonical_bundle_dir),
                    "error": str(e),
                }
                report_dir = _choose_report_dir(output_dir)
                fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
                _write_json_atomic(fatal_path, capsule)
                raise SystemExit(2)
    if actual_bundle_dir is None:
        msg = "actual_bundle_dir is None after resolution"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "policy_path": str(policy_path),
            "canonical_bundle_dir": str(canonical_bundle_dir),
        }
        report_dir = _choose_report_dir(output_dir)
        fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    # Verify actual bundle dir is under canonical
    if actual_bundle_dir.resolve() != canonical_bundle_dir.resolve():
        msg = f"Bundle directory mismatch: actual={actual_bundle_dir}, canonical={canonical_bundle_dir}"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "canonical_bundle_dir": str(canonical_bundle_dir),
            "actual_bundle_dir": str(actual_bundle_dir),
            "policy_path": str(policy_path),
            "bundle_dir_override": str(bundle_dir_override) if bundle_dir_override else None,
        }
        report_dir = _choose_report_dir(output_dir)
        fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    # SIGNAL-ONLY TRUTH (dual bundles): canonical XGB bundle dir + canonical Transformer bundle dir.
    truth_file = os.getenv("GX1_CANONICAL_TRUTH_FILE") or ""
    transformer_bundle_dir_env = os.getenv("GX1_CANONICAL_TRANSFORMER_BUNDLE_DIR") or ""
    if truth_file and transformer_bundle_dir_env:
        transformer_bundle_dir = Path(transformer_bundle_dir_env).resolve()
        if not transformer_bundle_dir.exists():
            capsule = {
                "status": "FAIL",
                "message": "Canonical transformer bundle directory does not exist",
                "canonical_bundle_dir": str(canonical_bundle_dir),
                "canonical_transformer_bundle_dir": str(transformer_bundle_dir),
            }
            report_dir = _choose_report_dir(output_dir)
            fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
            _write_json_atomic(fatal_path, capsule)
            raise SystemExit(2)

        # A) Verify XGB lock vs joblib sha
        expected_sha256 = _read_lock_sha256(canonical_bundle_dir)
        if expected_sha256:
            xgb_joblib_path = canonical_bundle_dir / "xgb_universal_multihead_v2.joblib"
            if not xgb_joblib_path.exists():
                capsule = {
                    "status": "FAIL",
                    "message": "XGB joblib missing for canonical bundle",
                    "canonical_bundle_dir": str(canonical_bundle_dir),
                    "xgb_joblib_path": str(xgb_joblib_path),
                }
                report_dir = _choose_report_dir(output_dir)
                fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
                _write_json_atomic(fatal_path, capsule)
                raise SystemExit(2)
            joblib_sha256 = sha256_file(xgb_joblib_path)
            if joblib_sha256 != expected_sha256:
                capsule = {
                    "status": "FAIL",
                    "message": "XGB joblib SHA256 mismatch vs MASTER_MODEL_LOCK.model_sha256",
                    "canonical_bundle_dir": str(canonical_bundle_dir),
                    "joblib_sha256": joblib_sha256,
                    "expected_sha256": expected_sha256,
                }
                report_dir = _choose_report_dir(output_dir)
                fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
                _write_json_atomic(fatal_path, capsule)
                raise SystemExit(2)

        # B) Verify Transformer lock vs model_state_dict sha
        t_lock = transformer_bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
        t_model = transformer_bundle_dir / "model_state_dict.pt"
        if not t_lock.exists() or not t_model.exists():
            capsule = {
                "status": "FAIL",
                "message": "Transformer bundle missing required files",
                "canonical_transformer_bundle_dir": str(transformer_bundle_dir),
                "missing_master_transformer_lock": not t_lock.exists(),
                "missing_model_state_dict": not t_model.exists(),
            }
            report_dir = _choose_report_dir(output_dir)
            fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
            _write_json_atomic(fatal_path, capsule)
            raise SystemExit(2)
        try:
            t_lock_obj = json.loads(t_lock.read_text())
        except Exception as e:
            capsule = {
                "status": "FAIL",
                "message": f"Failed to parse transformer lock: {e}",
                "path": str(t_lock),
            }
            report_dir = _choose_report_dir(output_dir)
            fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
            _write_json_atomic(fatal_path, capsule)
            raise SystemExit(2)
        t_expected = str(t_lock_obj.get("model_sha256") or "")
        t_actual = sha256_file(t_model)
        if t_expected and t_actual != t_expected:
            capsule = {
                "status": "FAIL",
                "message": "Transformer model SHA256 mismatch vs MASTER_TRANSFORMER_LOCK.model_sha256",
                "canonical_transformer_bundle_dir": str(transformer_bundle_dir),
                "model_sha256": t_actual,
                "expected_sha256": t_expected,
            }
            report_dir = _choose_report_dir(output_dir)
            fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
            _write_json_atomic(fatal_path, capsule)
            raise SystemExit(2)

        # Write success MD (minimal; canonical truth is enforced by caller)
        report_dir = _choose_report_dir(output_dir)
        md_path = report_dir / "BUNDLE_IDENTITY.md"
        md = _render_md(
            canonical_bundle_dir=canonical_bundle_dir,
            actual_bundle_dir=actual_bundle_dir,
            policy_path=policy_path,
            computed_sha256="SIGNAL_ONLY_DUAL_BUNDLE_V1",
            expected_sha256=expected_sha256,
            lock_sha256=None,
        )
        md += "\n" + "\n".join(
            [
                "## Transformer Bundle (signal-only)",
                "",
                f"- **canonical_transformer_bundle_dir**: `{transformer_bundle_dir}`",
                f"- **transformer_model_sha256**: `{t_actual}`",
            ]
        )
        _write_text_atomic(md_path, md)
        return
    
    # Compute bundle SHA256
    try:
        computed_sha256 = compute_bundle_sha256(
            policy_path=policy_path,
            resolved_artifact_paths=resolved_paths,
            bundle_dir_override=bundle_dir_override,
        )
    except Exception as e:
        msg = f"Failed to compute bundle_sha256: {e}"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "canonical_bundle_dir": str(canonical_bundle_dir),
            "actual_bundle_dir": str(actual_bundle_dir),
            "policy_path": str(policy_path),
            "error": str(e),
        }
        report_dir = _choose_report_dir(output_dir)
        fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    # Read expected SHA256 from lock if available
    expected_sha256 = _read_lock_sha256(canonical_bundle_dir)
    lock_sha256 = None
    lock_path = canonical_bundle_dir / "MASTER_MODEL_LOCK.json"
    if lock_path.exists():
        try:
            with open(lock_path, "rb") as f:
                import hashlib
                lock_sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
    
    # Verify SHA256 match if expected is available
    # Note: expected_sha256 from lock is model_sha256 (joblib SHA256), while computed_sha256
    # is from compute_bundle_sha256 (policy + artifacts). These are different hashes.
    # We verify joblib SHA256 matches model_sha256 from lock.
    if expected_sha256:
        # Compare joblib SHA256 with model_sha256 from lock
        xgb_joblib_path = canonical_bundle_dir / "xgb_universal_multihead_v2.joblib"
        if xgb_joblib_path.exists():
            try:
                joblib_sha256 = sha256_file(xgb_joblib_path)
                if joblib_sha256 != expected_sha256:
                    msg = f"XGB joblib SHA256 mismatch: actual={joblib_sha256}, expected (from lock)={expected_sha256}"
                    capsule = {
                        "status": "FAIL",
                        "message": msg,
                        "canonical_bundle_dir": str(canonical_bundle_dir),
                        "actual_bundle_dir": str(actual_bundle_dir),
                        "joblib_sha256": joblib_sha256,
                        "expected_sha256": expected_sha256,
                        "computed_bundle_sha256": computed_sha256,
                        "policy_path": str(policy_path),
                    }
                    report_dir = _choose_report_dir(output_dir)
                    fatal_path = report_dir / "BUNDLE_IDENTITY_FATAL.json"
                    _write_json_atomic(fatal_path, capsule)
                    raise SystemExit(2)
            except Exception as e:
                log.warning(f"Failed to verify joblib SHA256: {e}")
    
    # Write BUNDLE_IDENTITY.md
    report_dir = _choose_report_dir(output_dir)
    md_path = report_dir / "BUNDLE_IDENTITY.md"
    md_content = _render_md(
        canonical_bundle_dir=canonical_bundle_dir,
        actual_bundle_dir=actual_bundle_dir,
        policy_path=policy_path,
        computed_sha256=computed_sha256,
        expected_sha256=expected_sha256,
        lock_sha256=lock_sha256,
    )
    _write_text_atomic(md_path, md_content)
    
    log.info(f"[BUNDLE_IDENTITY_GATE] ✅ PASS: canonical={canonical_bundle_dir}, computed_sha256={computed_sha256[:16]}...")
