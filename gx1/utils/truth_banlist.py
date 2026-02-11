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


__all__ = ["BANLIST", "is_truth_or_smoke", "assert_truth_banlist_clean"]

