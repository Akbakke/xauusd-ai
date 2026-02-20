"""
TRUTH/SMOKE import proof: evidence of loaded modules and hard-fail on forbidden imports.

Collects sys.modules snapshot; checks for forbidden substrings; writes proof JSON
atomically. Used by runner init and E2E post-run gate.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Conservative list: modules that must not appear in sys.modules in TRUTH/SMOKE (ONE EXIT PATH, no ghost).
# Substring match: any key in sys.modules containing one of these fails.
FORBIDDEN_IMPORT_SUBSTRINGS: List[str] = [
    "replay_eval_gated_parallel",
    "exit_master_v1",
    "exit_farm_v2_rules",
    "exit_farm_v2_rules_adaptive",
    "exit_fixed_bar",
    "FARM_V2",
]


def collect_import_proof() -> Dict[str, Any]:
    """
    Snapshot current process: python_exe, cwd, pid, sys_path, modules_loaded, forbidden_hits, timestamp_utc.
    """
    modules_loaded = sorted(sys.modules.keys())
    forbidden_hits: List[str] = []
    for mod in modules_loaded:
        for sub in FORBIDDEN_IMPORT_SUBSTRINGS:
            if sub in mod:
                forbidden_hits.append(mod)
                break
    return {
        "python_exe": getattr(sys, "executable", ""),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "sys_path": list(sys.path),
        "modules_loaded": modules_loaded,
        "forbidden_hits": forbidden_hits,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_import_proof(path: Path, proof: Dict[str, Any]) -> None:
    """
    Write proof dict to path as JSON, atomically (tmp -> os.replace). TRUTH-safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def assert_no_forbidden_imports(proof: Dict[str, Any], context: str) -> None:
    """
    If proof["forbidden_hits"] is non-empty, raise RuntimeError with [TRUTH_IMPORT_GHOST] and list.
    """
    hits = proof.get("forbidden_hits") or []
    if hits:
        raise RuntimeError(
            f"[TRUTH_IMPORT_GHOST] Forbidden modules loaded in {context}: {hits}. "
            "ONE EXIT PATH: remove ghost imports (replay_eval_gated_parallel, exit_master_v1, "
            "exit_farm_v2_rules, exit_fixed_bar, FARM_V2, etc.)."
        )
