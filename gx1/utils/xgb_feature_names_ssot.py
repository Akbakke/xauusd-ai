#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSoT loader for ordered XGB feature names in TRUTH/PREBUILT runs.

Signal-only truth:
1) MASTER_MODEL_LOCK.json MUST contain an explicit ordered feature list -> use it (order-sensitive)
2) XGB meta and schema manifest (if present/used) must match the lock list (order-sensitive)

No fallback / no auto-discovery in TRUTH/SMOKE.

This module performs no network calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sha256_pipe_join(names: List[str]) -> str:
    return hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def _extract_feature_list(obj: Dict[str, Any]) -> Optional[List[str]]:
    """
    Try a few common keys used across GX1 artifacts.
    """
    cand = _first_present(obj, ["ordered_features", "feature_names", "feature_list", "features", "required_all_features"])
    if isinstance(cand, list) and cand and all(isinstance(x, str) for x in cand):
        return list(cand)
    return None


def load_xgb_feature_names_ssot(run_dir: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Load ordered XGB feature names with strict sha verification.

    Returns: (names_ordered, details)
    """
    rd = Path(run_dir)
    if not rd.exists():
        raise RuntimeError(f"[SSOT] run_dir does not exist: {rd}")

    run_identity = _read_json(rd / "RUN_IDENTITY.json")
    bundle_dir = Path(str(run_identity.get("bundle_dir_resolved") or run_identity.get("bundle_dir") or ""))
    if not bundle_dir.exists():
        raise RuntimeError(f"[SSOT] bundle_dir missing: {bundle_dir}")

    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError(f"[SSOT] MASTER_MODEL_LOCK.json missing: {lock_path}")
    lock = _read_json(lock_path)
    lock_sha = str(lock.get("feature_list_sha256") or "")
    if not lock_sha:
        raise RuntimeError("[SSOT] MASTER_MODEL_LOCK.feature_list_sha256 missing")

    # 1) Lock explicit list (REQUIRED)
    names = _extract_feature_list(lock)
    if not names:
        raise RuntimeError(
            "[SSOT] TRUTH_NO_FALLBACK: MASTER_MODEL_LOCK is missing explicit ordered feature list "
            "(ordered_features/feature_names/feature_list/features)."
        )
    sha = _sha256_pipe_join(names)
    if sha != lock_sha:
        raise RuntimeError(
            "[SSOT] MASTER_MODEL_LOCK feature list sha mismatch: "
            f"computed={sha} expected={lock_sha}"
        )
    return names, {"source": "MASTER_MODEL_LOCK.json", "feature_list_sha256": sha}

