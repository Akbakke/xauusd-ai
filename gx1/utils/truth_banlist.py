"""
TRUTH banlist (SSoT) — forbid legacy truth contracts and fallback mechanisms.

Non-negotiable:
- ONE truth only.
- No fallback / no auto-discovery / no ambiguity in TRUTH/SMOKE.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TruthBanlist:
    banned_modules: List[str]
    banned_env_vars: List[str]


BANLIST = TruthBanlist(
    banned_modules=[
        # Legacy feature truth contracts
        "gx1.contracts.feature_contract_v13_core",
        "gx1.features.feature_contract_v10_ctx",
        # ONE UNIVERSE: legacy 16/88 entry transformer (session/vol/trend as separate args)
        "gx1.models.entry_v10.entry_v10_hybrid_transformer",
        # ONE UNIVERSE: rule-based exit (quarantined); ML exit only
        "gx1.policy.exit_master_v1",
        "gx1.policy.exit_farm_v2_rules",
        "gx1.policy.exit_farm_v2_rules_adaptive",
        "gx1.policy.exit_fixed_bar",
        # Ghost purge: legacy replay script must not be importable in TRUTH
        "gx1.scripts.replay_eval_gated_parallel",
    ],
    banned_env_vars=[
        # Explicitly forbidden fallback selector for prebuilt
        "GX1_REPLAY_PREBUILT_FEATURES_PATH",
    ],
)


def is_truth_or_smoke() -> bool:
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or run_mode in {"TRUTH", "SMOKE"} or os.getenv("GX1_SMOKE", "0") == "1"


def _write_capsule(output_dir: Optional[Path], payload: Dict[str, Any]) -> None:
    if output_dir is None:
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        p = output_dir / "TRUTH_BANLIST_HIT.json"
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def assert_truth_banlist_clean(*, output_dir: Optional[Path], stage: str) -> None:
    """
    In TRUTH/SMOKE, hard-fail if banned legacy modules are already imported or forbidden fallback env vars are set.
    """
    if not is_truth_or_smoke():
        return

    imported_hits = sorted([m for m in BANLIST.banned_modules if m in sys.modules])
    env_hits = sorted([k for k in BANLIST.banned_env_vars if os.getenv(k)])

    if imported_hits or env_hits:
        payload = {
            "status": "FAIL",
            "error": "TRUTH_BANLIST_HIT",
            "stage": stage,
            "utc_ts": datetime.now(timezone.utc).isoformat(),
            "imported_banned_modules": imported_hits,
            "forbidden_env_vars_set": {k: os.getenv(k) for k in env_hits},
            "GX1_RUN_MODE": os.getenv("GX1_RUN_MODE"),
            "GX1_TRUTH_MODE": os.getenv("GX1_TRUTH_MODE"),
        }
        _write_capsule(output_dir, payload)
        raise RuntimeError(f"[TRUTH_BANLIST_HIT] stage={stage} imported={imported_hits} env={env_hits}")


def assert_truth_exit_policy_clean(
    policy: Dict[str, Any],
    *,
    output_dir: Optional[Path] = None,
    resolved_exit_type: Optional[str] = None,
) -> None:
    """
    In TRUTH/SMOKE, hard-fail if policy contains forbidden exit keys or exit.type is not EXIT_TRANSFORMER_V0.
    ONE UNIVERSE: ML exit only; no router, no exit_policies, no exit_critic, no rule-based exit.
    resolved_exit_type: pass the exit type from loaded exit YAML (runner sets this after loading exit_config).
    """
    if not is_truth_or_smoke():
        return
    if not isinstance(policy, dict):
        return
    errors: List[str] = []
    if policy.get("hybrid_exit_router") is not None:
        errors.append("policy must not contain 'hybrid_exit_router'")
    if policy.get("exit_policies") is not None:
        errors.append("policy must not contain 'exit_policies'")
    if policy.get("exit_critic") is not None:
        errors.append("policy must not contain 'exit_critic'")
    exit_type = resolved_exit_type
    if exit_type is None:
        exit_block = policy.get("exit") if isinstance(policy.get("exit"), dict) else None
        if exit_block:
            exit_type = exit_block.get("type")
    if exit_type is not None and str(exit_type).strip() != "EXIT_TRANSFORMER_V0":
        errors.append(f"exit.type must be exactly 'EXIT_TRANSFORMER_V0' in TRUTH/SMOKE (ONE UNIVERSE ML-only), got: {exit_type!r}")
    if not errors:
        return
    payload = {
        "status": "FAIL",
        "error": "TRUTH_EXIT_POLICY_VIOLATION",
        "utc_ts": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "GX1_RUN_MODE": os.getenv("GX1_RUN_MODE"),
    }
    _write_capsule(output_dir, payload)
    raise RuntimeError(f"[TRUTH_EXIT_POLICY_VIOLATION] {'; '.join(errors)}")


# Canonical policy path (relative to engine/repo root). In TRUTH/SMOKE only this file may be used.
TRUTH_CANONICAL_POLICY_RELATIVE = "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"


_CAPSULE_FILENAME = "TRUTH_POLICY_PATH_MISMATCH_CAPSULE.json"


def _write_policy_path_capsule(output_dir: Optional[Path], payload: Dict[str, Any]) -> None:
    """Write policy-path mismatch capsule atomically (best-effort)."""
    if output_dir is None:
        return
    try:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        p = output_dir / _CAPSULE_FILENAME
        tmp = output_dir / f"{_CAPSULE_FILENAME}.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def _policy_path_capsule_payload(
    error: str,
    expected_resolved: Path,
    loaded_resolved: Path,
    engine_root: Path,
    *,
    loaded_path_raw: Optional[str] = None,
) -> Dict[str, Any]:
    """Build capsule payload with expected/loaded/engine_root/cwd, env, timestamp_utc, raw inputs for debugging."""
    out: Dict[str, Any] = {
        "status": "FAIL",
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_policy_path_resolved": str(expected_resolved),
        "expected_policy_relative": TRUTH_CANONICAL_POLICY_RELATIVE,
        "loaded_policy_path_resolved": str(loaded_resolved),
        "engine_root_resolved": str(engine_root),
        "cwd": str(Path.cwd()),
        "GX1_CANONICAL_POLICY_PATH": os.getenv("GX1_CANONICAL_POLICY_PATH"),
        "GX1_CANONICAL_TRUTH_FILE": os.getenv("GX1_CANONICAL_TRUTH_FILE"),
        "module": "gx1.utils.truth_banlist",
        "function": "assert_truth_policy_path_canonical",
    }
    if loaded_path_raw is not None:
        out["loaded_path_raw"] = loaded_path_raw
    return out


def _path_is_relative_to(path: Path, other: Path) -> bool:
    """True if path is under other (Python 3.9+ has path.is_relative_to). Uses resolved paths so symlinks are normalized."""
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except ValueError:
        return False


def assert_truth_policy_path_canonical(
    loaded_path: Path,
    *,
    engine_root: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """
    In TRUTH/SMOKE, hard-fail if the policy path used does not match the canonical policy file exactly.
    Resolves engine_root and loaded_path; compares loaded_path.resolve() to (engine_root/TRUTH_CANONICAL_POLICY_RELATIVE).resolve().
    No globs, no fuzzy matching.
    """
    if not is_truth_or_smoke():
        return
    loaded_path_raw = str(loaded_path)
    engine_root = Path(engine_root).resolve()
    loaded_norm = Path(loaded_path).expanduser()
    expected_resolved = (engine_root / TRUTH_CANONICAL_POLICY_RELATIVE).resolve()
    loaded_resolved = loaded_norm.resolve()

    if not loaded_norm.exists():
        payload = _policy_path_capsule_payload(
            "TRUTH_POLICY_PATH_LOADED_MISSING",
            expected_resolved, loaded_resolved, engine_root, loaded_path_raw=loaded_path_raw,
        )
        _write_policy_path_capsule(output_dir, payload)
        raise RuntimeError(
            f"[TRUTH_POLICY_PATH] Loaded policy path does not exist: {loaded_norm}. "
            f"TRUTH/SMOKE may only use the canonical policy file."
        )
    if not loaded_norm.is_file():
        payload = _policy_path_capsule_payload(
            "TRUTH_POLICY_PATH_LOADED_NOT_FILE",
            expected_resolved, loaded_resolved, engine_root, loaded_path_raw=loaded_path_raw,
        )
        _write_policy_path_capsule(output_dir, payload)
        raise RuntimeError(
            f"[TRUTH_POLICY_PATH] Loaded policy path is not a file: {loaded_norm}. "
            f"TRUTH/SMOKE may only use the canonical policy file."
        )
    if not _path_is_relative_to(loaded_resolved, engine_root):
        payload = _policy_path_capsule_payload(
            "TRUTH_POLICY_PATH_OUTSIDE_ENGINE",
            expected_resolved, loaded_resolved, engine_root, loaded_path_raw=loaded_path_raw,
        )
        _write_policy_path_capsule(output_dir, payload)
        raise RuntimeError(
            f"[TRUTH_POLICY_PATH] Loaded policy path is outside engine root: {loaded_resolved} | engine_root={engine_root}. "
            f"TRUTH/SMOKE may only use the canonical policy file under engine."
        )
    # Check canonical file exists before equality so failure points to root cause (canonical missing), not mismatch.
    if not expected_resolved.exists() or not expected_resolved.is_file():
        payload = _policy_path_capsule_payload(
            "TRUTH_CANONICAL_POLICY_MISSING",
            expected_resolved, loaded_resolved, engine_root, loaded_path_raw=loaded_path_raw,
        )
        _write_policy_path_capsule(output_dir, payload)
        raise RuntimeError(
            f"[TRUTH_CANONICAL_POLICY] Canonical policy file not found: {expected_resolved}. "
            f"TRUTH/SMOKE may only use this single policy file."
        )
    if loaded_resolved != expected_resolved:
        payload = _policy_path_capsule_payload(
            "TRUTH_POLICY_PATH_VIOLATION",
            expected_resolved, loaded_resolved, engine_root, loaded_path_raw=loaded_path_raw,
        )
        _write_policy_path_capsule(output_dir, payload)
        raise RuntimeError(
            f"[TRUTH_POLICY_PATH_VIOLATION] In TRUTH/SMOKE only the canonical policy file is allowed. "
            f"Loaded: {loaded_resolved} | Required: {expected_resolved}. "
            f"Do not use another policy file or duplicate; use exactly: {TRUTH_CANONICAL_POLICY_RELATIVE}"
        )


__all__ = [
    "BANLIST",
    "is_truth_or_smoke",
    "assert_truth_banlist_clean",
    "assert_truth_exit_policy_clean",
    "assert_truth_policy_path_canonical",
    "TRUTH_CANONICAL_POLICY_RELATIVE",
]

