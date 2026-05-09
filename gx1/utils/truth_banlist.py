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


def assert_truth_policy_path_canonical(policy_path: Path, *, engine_root: Path, output_dir: Optional[Path]) -> None:
    """
    TRUTH gate: policy_path must be in the canonical allowlist for TRUTH/SMOKE runs.
    """
    if not is_truth_or_smoke():
        return
    sweep_root_env = os.getenv("GX1_TRUTH_POLICY_SWEEP_ROOT", "").strip()
    sweep_root = None
    if sweep_root_env:
        try:
            sweep_root = Path(sweep_root_env).expanduser().resolve()
        except Exception:
            sweep_root = None
    allowlist = {
        (engine_root / "gx1" / "configs" / "policies" / "canonical_truth" / "GX1_TRUTH_REPLAY_V10_CTX.yaml").resolve(),
        (engine_root / "gx1" / "configs" / "policies" / "canonical_truth" / "GX1_TRUTH_REPLAY_V10_CTX_V2.yaml").resolve(),
        (engine_root / "gx1" / "configs" / "policies" / "phase5_exit_verification" / "PHASE5_EXIT_VERIFICATION_V10_CTX__R5C.yaml").resolve(),
        (engine_root / "gx1" / "configs" / "policies" / "phase5_exit_verification" / "PHASE5_EXIT_VERIFICATION_V10_CTX__R5D.yaml").resolve(),
    }
    policy_resolved = policy_path.resolve()
    sweep_allowed = False
    if sweep_root is not None:
        try:
            sweep_allowed = policy_resolved.is_relative_to(sweep_root)
        except Exception:
            sweep_allowed = False
    if policy_resolved not in allowlist and not sweep_allowed:
        payload = {
            "status": "FAIL",
            "error": "TRUTH_POLICY_PATH_NOT_CANONICAL",
            "allowed": sorted([str(p) for p in allowlist]),
            "policy_path": str(policy_resolved),
            "sweep_root": str(sweep_root) if sweep_root is not None else "",
        }
        _write_capsule(output_dir, payload)
        raise RuntimeError(
            f"[TRUTH_POLICY_PATH_NOT_CANONICAL] policy_path={policy_resolved} not in allowlist"
        )


__all__ = ["BANLIST", "is_truth_or_smoke", "assert_truth_banlist_clean", "assert_truth_policy_path_canonical"]
