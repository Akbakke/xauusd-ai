#!/usr/bin/env python3
"""
Canonical ENTRY_V10_CTX trainer.

ONE UNIVERSE (STRICT):
- Signal bridge: XGB_SIGNAL_BRIDGE_V1 (7-dim)
- Context: CTX6CAT6 base (ctx_cat=6 fixed; ctx_cont base=6 with optional micro extensions)
- No RL
- No legacy
- No fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from gx1.contracts.signal_bridge_v1 import (
    ORDERED_FIELDS as SIGNAL_FIELDS,
    CONTRACT_SHA256 as SIGNAL_BRIDGE_CONTRACT_SHA256,
    SEQ_SIGNAL_DIM,
    SNAP_SIGNAL_DIM,
)
from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
from gx1.time.session_detector import (
    get_session_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)

# -----------------------------------------------------------------------------
# RL / legacy guard (fail-fast)
# -----------------------------------------------------------------------------
def _guard_no_rl() -> None:
    """Hard-fail if gx1.rl or legacy was imported."""
    for mod in list(sys.modules.keys()):
        if mod == "gx1.rl" or mod.startswith("gx1.rl."):
            raise RuntimeError(
                "[ENTRY_V10_CTX_RL_FORBIDDEN] gx1.rl must not be imported. "
                f"Found: {mod}"
            )
        if "legacy" in mod and mod.startswith("gx1."):
            raise RuntimeError(
                "[ENTRY_V10_CTX_LEGACY_FORBIDDEN] gx1 legacy must not be imported. "
                f"Found: {mod}"
            )

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()

# -----------------------------------------------------------------------------
# SHORT collapse countermeasures (training-only)
# Canonical lane keeps these parked unless we have clear evidence they help more
# than they complicate the recipe.
# -----------------------------------------------------------------------------
SHORT_CLASS_WEIGHT = float(_env_str("ENTRY_SHORT_CLASS_WEIGHT", "0.90"))
XGB_SHORT_LEAD_MARGIN = float(_env_str("ENTRY_XGB_SHORT_LEAD_MARGIN", "0.0"))
XGB_SHORT_LONG_PENALTY = float(_env_str("ENTRY_XGB_SHORT_LONG_PENALTY", "0.0"))

# -----------------------------------------------------------------------------
# Cost-sensitive loss (ENTRY 3-class)
# -----------------------------------------------------------------------------
# Defaults are deliberately moderate: wrong-side (LONG<->SHORT) costs clearly more
# than LONG/SHORT->FLAT, while FLAT->LONG/SHORT remains moderate.
ENTRY_COST_SENSITIVE_ENABLED = int(_env_str("ENTRY_COST_SENSITIVE_LOSS", "0"))
ENTRY_COST_SENSITIVE_SCALE = float(_env_str("ENTRY_COST_SENSITIVE_SCALE", "0.0"))
ENTRY_COST_LONG_TO_SHORT = float(_env_str("ENTRY_COST_LONG_TO_SHORT", "1.50"))
ENTRY_COST_LONG_TO_FLAT = float(_env_str("ENTRY_COST_LONG_TO_FLAT", "0.60"))
ENTRY_COST_SHORT_TO_LONG = float(_env_str("ENTRY_COST_SHORT_TO_LONG", "1.50"))
ENTRY_COST_SHORT_TO_FLAT = float(_env_str("ENTRY_COST_SHORT_TO_FLAT", "0.60"))
ENTRY_COST_FLAT_TO_LONG = float(_env_str("ENTRY_COST_FLAT_TO_LONG", "0.80"))
ENTRY_COST_FLAT_TO_SHORT = float(_env_str("ENTRY_COST_FLAT_TO_SHORT", "0.80"))

# -----------------------------------------------------------------------------
# Prediction balance regularizer (anti-collapse; training/eval loss only)
# -----------------------------------------------------------------------------
# Default is mild and label-aligned: nudges mean predicted distribution toward
# the batch label distribution (not uniform), to reduce single-side collapse.
ENTRY_PRED_BALANCE_ALPHA = float(_env_str("ENTRY_PRED_BALANCE_ALPHA", "0.0"))
ENTRY_PRED_BALANCE_TARGET = _env_str("ENTRY_PRED_BALANCE_TARGET", "label").lower()
ENTRY_RESIDUAL_SIDE_BIAS_ALPHA = float(_env_str("ENTRY_RESIDUAL_SIDE_BIAS_ALPHA", "0.0"))
ENTRY_DIRECTION_CE_SCALE = float(_env_str("ENTRY_DIRECTION_CE_SCALE", "1.0"))

# -----------------------------------------------------------------------------
# Timing loss (early adverse move penalty)
# -----------------------------------------------------------------------------
ENTRY_TIMING_TARGET_BPS = float(_env_str("ENTRY_TIMING_TARGET_BPS", "3.0"))
ENTRY_TIMING_LOSS_SCALE = float(_env_str("ENTRY_TIMING_LOSS_SCALE", "0.0"))

# -----------------------------------------------------------------------------
# Auxiliary losses (use existing dataset targets)
# -----------------------------------------------------------------------------
# Canonical lane keeps only the auxiliary heads that directly support runtime
# gates (tradable, mfe_first_n, path_quality). Early/quality-score/bad-path stay parked.
ENTRY_AUX_EARLY_WEIGHT = float(_env_str("ENTRY_AUX_EARLY_WEIGHT", "0.0"))
ENTRY_AUX_QUALITY_WEIGHT = float(_env_str("ENTRY_AUX_QUALITY_WEIGHT", "0.0"))
ENTRY_AUX_PATH_WEIGHT = float(_env_str("ENTRY_AUX_PATH_WEIGHT", "0.35"))
ENTRY_AUX_MFE_WEIGHT = float(_env_str("ENTRY_AUX_MFE_WEIGHT", "0.25"))
ENTRY_AUX_TRADABLE_WEIGHT = float(_env_str("ENTRY_AUX_TRADABLE_WEIGHT", "0.50"))
# Canonical lane keeps bad-path parked until it shows clean incremental value.
ENTRY_AUX_BAD_PATH_WEIGHT = float(_env_str("ENTRY_AUX_BAD_PATH_WEIGHT", "0.0"))
ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP", "20.0"))
# Scale bps targets to keep regression losses in a stable range
ENTRY_AUX_QUALITY_SCALE_BPS = float(_env_str("ENTRY_AUX_QUALITY_SCALE_BPS", "50.0"))
ENTRY_AUX_PATH_SCALE_BPS = float(_env_str("ENTRY_AUX_PATH_SCALE_BPS", "50.0"))
ENTRY_AUX_MFE_SCALE_BPS = float(_env_str("ENTRY_AUX_MFE_SCALE_BPS", "20.0"))

# -----------------------------------------------------------------------------
# Micro features (ctx_cont extension)
# -----------------------------------------------------------------------------
MICRO_FEATURE_NAMES = [
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
]
SWING_FEATURE_NAMES = [
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
]
EXT_CTX_FEATURE_NAMES = list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES)

# -----------------------------------------------------------------------------
# V_NEXT session-context extension (ctx_cont +5)
# Canonical default is V_NEXT (CTX21). Explicitly set GX1_CTX_CONTRACT=V_CURRENT
# only for legacy/debug use. Training without this env set now defaults to CTX21.
# -----------------------------------------------------------------------------
_CTX_CONTRACT_MODE = _env_str("GX1_CTX_CONTRACT", "V_NEXT").upper()
V_NEXT_EXTRA_CTX_CONT = [
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
]

def _is_vnext() -> bool:
    return _CTX_CONTRACT_MODE == "V_NEXT"

def _expected_ctx_cont_dim() -> int:
    return 21 if _is_vnext() else 16

def _build_ordered_ctx_cont_names(ctx_cont_dim: int, base_names: List[str]) -> List[str]:
    ordered = list(base_names)
    if ctx_cont_dim > len(ordered):
        ordered = ordered + list(EXT_CTX_FEATURE_NAMES)
    if _is_vnext() and ctx_cont_dim > len(ordered):
        ordered = ordered + list(V_NEXT_EXTRA_CTX_CONT)
    return ordered

# -----------------------------------------------------------------------------
# Anchored ENTRY (residual over XGB signal7 probs)
# -----------------------------------------------------------------------------
# Keep the canonical anchor mix unchanged for this bad-path candidate so replay
# deltas reflect the new adverse-first supervision rather than a different anchor.
ENTRY_RESIDUAL_SCALE = float(_env_str("ENTRY_RESIDUAL_SCALE", "0.35"))
ENTRY_ANCHOR_EPS = float(_env_str("ENTRY_ANCHOR_EPS", "1e-6"))

_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS: Dict[str, str] = {
    "ENTRY_SHORT_CLASS_WEIGHT": "0.90",
    "ENTRY_XGB_SHORT_LEAD_MARGIN": "0.0",
    "ENTRY_XGB_SHORT_LONG_PENALTY": "0.0",
    "ENTRY_COST_SENSITIVE_LOSS": "0",
    "ENTRY_COST_SENSITIVE_SCALE": "0.0",
    "ENTRY_COST_LONG_TO_SHORT": "1.50",
    "ENTRY_COST_LONG_TO_FLAT": "0.60",
    "ENTRY_COST_SHORT_TO_LONG": "1.50",
    "ENTRY_COST_SHORT_TO_FLAT": "0.60",
    "ENTRY_COST_FLAT_TO_LONG": "0.80",
    "ENTRY_COST_FLAT_TO_SHORT": "0.80",
    "ENTRY_PRED_BALANCE_ALPHA": "0.0",
    "ENTRY_PRED_BALANCE_TARGET": "label",
    "ENTRY_RESIDUAL_SIDE_BIAS_ALPHA": "0.0",
    "ENTRY_DIRECTION_CE_SCALE": "1.0",
    "ENTRY_TIMING_TARGET_BPS": "3.0",
    "ENTRY_TIMING_LOSS_SCALE": "0.0",
    "ENTRY_AUX_EARLY_WEIGHT": "0.0",
    "ENTRY_AUX_QUALITY_WEIGHT": "0.0",
    "ENTRY_AUX_PATH_WEIGHT": "0.35",
    "ENTRY_AUX_MFE_WEIGHT": "0.25",
    "ENTRY_AUX_TRADABLE_WEIGHT": "0.50",
    "ENTRY_AUX_BAD_PATH_WEIGHT": "0.0",
    "ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP": "20.0",
    "ENTRY_AUX_QUALITY_SCALE_BPS": "50.0",
    "ENTRY_AUX_PATH_SCALE_BPS": "50.0",
    "ENTRY_AUX_MFE_SCALE_BPS": "20.0",
    "GX1_CTX_CONTRACT": "V_NEXT",
    "ENTRY_RESIDUAL_SCALE": "0.35",
    "ENTRY_ANCHOR_EPS": "1e-6",
}


def _enforce_canonical_train_env_contract() -> None:
    """
    Canonical ENTRY training must not be silently steered by ad-hoc env knobs.
    Non-canonical experimentation may opt in explicitly.
    """
    allow_noncanonical = _env_str("GX1_NON_CANONICAL_DIAGNOSTIC", "0") in {"1", "true", "yes", "on"} or _env_str(
        "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES", "0"
    ) in {"1", "true", "yes", "on"}
    if allow_noncanonical:
        log.warning("[ENTRY_CANONICAL_TRAIN_ENV] non-canonical env overrides explicitly enabled")
        return

    tripped: List[str] = []
    for name, default in _CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS.items():
        if name not in os.environ:
            continue
        actual = str(os.environ.get(name, "")).strip()
        if actual != str(default):
            tripped.append(f"{name}={actual!r} (default={default!r})")
    if tripped:
        raise RuntimeError(
            "[ENTRY_CANONICAL_TRAIN_ENV_FORBIDDEN] canonical entry training forbids non-default env overrides: "
            + ", ".join(sorted(tripped))
        )

# --- Determinism utilities (SSoT training API) ---------------------------------
# NOTE: These functions are part of the stable training-module API.
# The wrapper (gx1/scripts/train_entry_v10_ctx_depth_ladder.py) expects them.
# No fallbacks, no silent behavior changes—just deterministic seeding + thread limits.
import os as _os_det
import random as _random_det
from typing import Optional as _OptionalDet

import numpy as _np_det
import torch as _torch_det


def set_seed(seed: int) -> None:
    """
    Deterministic seeding for ENTRY_V10_CTX training.

    This is intentionally minimal and stable. Do not add "smart defaults" here.
    The caller decides the seed value.
    """
    if seed is None:
        raise ValueError("seed must be an int (not None)")

    # Python / NumPy
    _os_det.environ["PYTHONHASHSEED"] = str(int(seed))
    _random_det.seed(int(seed))
    _np_det.random.seed(int(seed))

    # Torch
    _torch_det.manual_seed(int(seed))
    if _torch_det.cuda.is_available():
        _torch_det.cuda.manual_seed_all(int(seed))

    # Determinism knobs (best-effort; no silent fallback logic)
    try:
        _torch_det.backends.cudnn.deterministic = True
        _torch_det.backends.cudnn.benchmark = False
    except Exception:
        # Some builds may not expose these flags; do not hard-fail.
        pass

    # PyTorch deterministic algorithms (may raise if op not supported; keep best-effort)
    try:
        _torch_det.use_deterministic_algorithms(True)
    except Exception:
        pass


def set_thread_limits(threads: int = 1) -> None:
    """
    Limit CPU thread usage for deterministic / reproducible runs (TRUTH doctrine).

    Best-effort: do not hard-fail on environments that do not support all settings.
    """
    if threads is None:
        raise ValueError("threads must be an int (not None)")
    t = int(threads)
    if t <= 0:
        raise ValueError(f"threads must be >= 1, got {t}")

    # Common BLAS/OpenMP knobs
    _os_det.environ["OMP_NUM_THREADS"] = str(t)
    _os_det.environ["MKL_NUM_THREADS"] = str(t)
    _os_det.environ["NUMEXPR_NUM_THREADS"] = str(t)
    _os_det.environ["OPENBLAS_NUM_THREADS"] = str(t)
    _os_det.environ["VECLIB_MAXIMUM_THREADS"] = str(t)

    # Torch thread limits
    try:
        _torch_det.set_num_threads(t)
    except Exception:
        pass

    try:
        _torch_det.set_num_interop_threads(t)
    except Exception:
        pass
# --- end determinism utilities -------------------------------------------------

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)

def _require_nonneg(name: str, v: float) -> None:
    if float(v) < 0.0:
        raise RuntimeError(f"[ENTRY_COST_INVALID] {name} must be >= 0.0, got {v}")

def _resolve_gx1_data(override: str = "") -> Path:
    base = Path(override or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(f"GX1_DATA invalid or missing: {base}")
    return base

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("[CUDA_NOT_AVAILABLE] requested cuda but torch.cuda.is_available() is False")
    return torch.device(device_str)

def _set_deterministic(seed: int, device: torch.device, deterministic: bool) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# -----------------------------------------------------------------------------
# Dataset resolution (manifest or dir)
# -----------------------------------------------------------------------------
def _resolve_train_val_parquets(
    dataset_manifest: Optional[Path],
    dataset_dir: Optional[Path],
    gx1_data: Path,
    train_parquet_hint: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve (train_parquet, val_parquet). Exactly one of dataset_manifest or dataset_dir must be set.
    When dataset_dir is set, train/val are matched by strict suffix *_train.parquet / *_val.parquet.
    If train_parquet_hint is provided, that path is used as train and val is inferred (same stem, _val.parquet).
    """
    if dataset_manifest is not None and dataset_dir is not None:
        raise RuntimeError(
            "[ENTRY_V10_CTX_DATASET_ARGS] Use only one of --dataset_manifest or --dataset_dir"
        )
    if dataset_manifest is None and dataset_dir is None:
        raise RuntimeError(
            "[ENTRY_V10_CTX_DATASET_ARGS] Provide --dataset_manifest or --dataset_dir"
        )

    if dataset_manifest is not None:
        p = Path(dataset_manifest).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_MISSING] {p}")
        if p.suffix.lower() != ".json":
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_NOT_JSON] {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        train_path = Path(data.get("output_data_path", "")).expanduser().resolve()
        if not train_path.is_absolute():
            train_path = (p.parent / train_path).resolve()
        if not train_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_TRAIN_PARQUET_MISSING] {train_path}")
        # Val: same dir, stem with _train -> _val
        stem = train_path.stem
        if stem.endswith("_train"):
            val_stem = stem[: -len("_train")] + "_val"
        else:
            val_stem = stem.replace("train", "val", 1) if "train" in stem else stem + "_val"
        val_path = train_path.parent / f"{val_stem}.parquet"
        if not val_path.exists():
            raise RuntimeError(
                f"[ENTRY_V10_CTX_VAL_PARQUET_MISSING] {val_path} (inferred from train)"
            )
        return train_path, val_path

    # dataset_dir: strict suffix match _train.parquet / _val.parquet only
    d = Path(dataset_dir).expanduser().resolve()
    if not d.is_dir():
        raise RuntimeError(f"[ENTRY_V10_CTX_DATASET_DIR_MISSING] {d}")
    parquets = list(d.glob("*.parquet"))
    train_candidates = [f for f in parquets if f.stem.endswith("_train")]
    val_candidates = [f for f in parquets if f.stem.endswith("_val")]

    if train_parquet_hint is not None:
        train_path = Path(train_parquet_hint).expanduser().resolve()
        if not train_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_TRAIN_PARQUET_MISSING] {train_path}")
        if not train_path.stem.endswith("_train"):
            raise RuntimeError(
                f"[ENTRY_V10_CTX_TRAIN_STEM] train_parquet_hint stem must end with _train, got {train_path.stem}"
            )
        val_stem = train_path.stem[: -len("_train")] + "_val"
        val_path = train_path.parent / f"{val_stem}.parquet"
        if not val_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_VAL_PARQUET_MISSING] {val_path} (inferred from train)")
        log.info("[DATASET_RESOLVE] train=%s val=%s", train_path, val_path)
        return train_path, val_path

    if len(train_candidates) != 1:
        raise RuntimeError(
            f"[ENTRY_V10_CTX_NO_TRAIN_PARQUET] expected exactly one *_train.parquet in {d}, got {len(train_candidates)}"
        )
    if len(val_candidates) != 1:
        raise RuntimeError(
            f"[ENTRY_V10_CTX_NO_VAL_PARQUET] expected exactly one *_val.parquet in {d}, got {len(val_candidates)}"
        )
    train_path = train_candidates[0].resolve()
    val_path = val_candidates[0].resolve()
    log.info("[DATASET_RESOLVE] train=%s val=%s", train_path, val_path)
    return train_path, val_path


def _log_manifest_proof(dataset_manifest: Optional[Path]) -> None:
    if dataset_manifest is None:
        return
    p = Path(dataset_manifest).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_MISSING] {p}")
    if p.suffix.lower() != ".json":
        raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_NOT_JSON] {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    inputs = data.get("inputs") or {}
    fc = data.get("feature_contract") or {}
    xgb_bundle = str(inputs.get("xgb_bundle") or "")
    xgb_model_sha256 = str(inputs.get("xgb_model_sha256") or "")
    bridge_id = str(fc.get("signal_bridge_id") or "")
    bridge_sha = str(fc.get("signal_bridge_contract_sha256") or "")

    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        override_path = str(Path(xgb_override).expanduser().resolve())
        if xgb_bundle and Path(xgb_bundle).expanduser().resolve() != Path(override_path).expanduser().resolve():
            raise RuntimeError(
                "[ENTRY_V10_CTX_XGB_OVERRIDE_MISMATCH] "
                f"GX1_XGB_BUNDLE_DIR={override_path} dataset_manifest.xgb_bundle={xgb_bundle}"
            )

    log.info(
        "[ENTRY_DATASET_MANIFEST_PROOF] manifest=%s xgb_bundle=%s xgb_model_sha256=%s signal_bridge_id=%s signal_bridge_sha256=%s",
        p,
        xgb_bundle,
        xgb_model_sha256,
        bridge_id,
        bridge_sha,
    )


def _resolve_test_parquet(
    dataset_manifest: Optional[Path],
    dataset_dir: Optional[Path],
    test_parquet: Optional[Path],
    gx1_data: Path,
    bundle_dir: Optional[Path] = None,
) -> Path:
    """
    Resolve test parquet. Priority:
    1) Explicit --test_parquet
    2) From --dataset_dir: single *test*.parquet
    3) From --dataset_manifest: infer _test.parquet from train stem
    """
    if test_parquet is not None:
        p = Path(test_parquet).expanduser().resolve()
        _require(p.exists(), f"[ENTRY_V10_CTX_TEST_PARQUET_MISSING] {p}")
        return p

    if dataset_dir is not None:
        d = Path(dataset_dir).expanduser().resolve()
        _require(d.is_dir(), f"[ENTRY_V10_CTX_DATASET_DIR_MISSING] {d}")
        parquets = list(d.glob("*.parquet"))
        test_candidates = [f for f in parquets if "test" in f.stem.lower()]
        if len(test_candidates) == 1:
            return test_candidates[0]
        if len(test_candidates) > 1 and bundle_dir is not None:
            meta_path = Path(bundle_dir).expanduser() / "bundle_metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    train_path = Path(meta.get("train_data", "")).expanduser()
                    if not train_path.is_absolute():
                        train_path = (d / train_path).resolve()
                    stem = train_path.stem
                    if stem.endswith("_train"):
                        test_stem = stem[: -len("_train")] + "_test"
                    else:
                        test_stem = stem.replace("train", "test", 1) if "train" in stem else stem + "_test"
                    inferred = train_path.parent / f"{test_stem}.parquet"
                    if inferred.exists():
                        return inferred
                except Exception:
                    pass
        raise RuntimeError(
            f"[ENTRY_V10_CTX_TEST_AMBIGUOUS] expected exactly one *test*.parquet in {d}, got {len(test_candidates)}"
        )

    if dataset_manifest is not None:
        p = Path(dataset_manifest).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_MISSING] {p}")
        if p.suffix.lower() != ".json":
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_NOT_JSON] {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        train_path = Path(data.get("output_data_path", "")).expanduser()
        if not train_path.is_absolute():
            train_path = (p.parent / train_path).resolve()
        if not train_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_TRAIN_PARQUET_MISSING] {train_path}")
        stem = train_path.stem
        if stem.endswith("_train"):
            test_stem = stem[: -len("_train")] + "_test"
        else:
            test_stem = stem.replace("train", "test", 1) if "train" in stem else stem + "_test"
        test_path = train_path.parent / f"{test_stem}.parquet"
        if not test_path.exists():
            raise RuntimeError(
                f"[ENTRY_V10_CTX_TEST_PARQUET_MISSING] {test_path} (inferred from train)"
            )
        return test_path

    raise RuntimeError(
        "[ENTRY_V10_CTX_TEST_RESOLVE_FAIL] provide --test_parquet or dataset manifest/dir"
    )


def _log_label_distribution(parquet_path: Path, split: str) -> None:
    p = Path(parquet_path).expanduser().resolve()
    if not p.exists():
        log.warning("[ENTRY_LABEL_DISTRIBUTION] split=%s status=missing path=%s", split, p)
        return
    try:
        df = pd.read_parquet(p, columns=["y_direction", "ctx_cat"])
    except Exception:
        df = pd.read_parquet(p, columns=["y_direction"])
    if "y_direction" not in df.columns:
        log.warning("[ENTRY_LABEL_DISTRIBUTION] split=%s status=no_y_direction path=%s", split, p)
        return
    y = df["y_direction"].astype(int)
    n = int(len(y))
    if n == 0:
        log.warning("[ENTRY_LABEL_DISTRIBUTION] split=%s status=empty path=%s", split, p)
        return
    long_c = int((y == 0).sum())
    short_c = int((y == 1).sum())
    flat_c = int((y == 2).sum())
    long_rate = long_c / n
    short_rate = short_c / n
    flat_rate = flat_c / n
    log.info(
        "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s n=%d long=%d (%.6f) short=%d (%.6f) flat=%d (%.6f) path=%s",
        split,
        n,
        long_c,
        long_rate,
        short_c,
        short_rate,
        flat_c,
        flat_rate,
        p,
    )
    log.info(
        "[ENTRY_FLAT_LABEL_PROOF] split=%s flat=%d flat_rate=%.6f status=%s path=%s",
        split,
        flat_c,
        flat_rate,
        "OK" if flat_c > 0 else "EMPTY",
        p,
    )

    if "ctx_cat" in df.columns:
        try:
            sess_ids = df["ctx_cat"].apply(lambda v: int(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else None)
            df_s = pd.DataFrame({"y": y, "session_id": sess_ids}).dropna(subset=["session_id"])
            if not df_s.empty:
                for sid, grp in df_s.groupby("session_id"):
                    n_s = int(len(grp))
                    long_rate_s = float((grp["y"] == 0).mean())
                    short_rate_s = float((grp["y"] == 1).mean())
                    flat_rate_s = float((grp["y"] == 2).mean())
                    session_name = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}.get(int(sid), "UNKNOWN")
                    log.info(
                        "[ENTRY_LABEL_BY_SESSION_PROOF] split=%s session=%s session_id=%s n=%d long_rate=%.6f short_rate=%.6f flat_rate=%.6f",
                        split,
                        session_name,
                        int(sid),
                        n_s,
                        long_rate_s,
                        short_rate_s,
                        flat_rate_s,
                    )
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class EntryV10CtxDataset(Dataset):
    """
    Builds rolling-window samples from canonical ENTRY_V10_CTX parquet.
    ctx_cont / ctx_cat are per-sample (B, N), not per-timestep.
    """

    def __init__(
        self,
        parquet_path: Path,
        seq_len: int,
        allow_constant_labels: bool,
    ):
        self.parquet_path = Path(parquet_path)
        self.seq_len = int(seq_len)

        if not self.parquet_path.exists():
            raise FileNotFoundError(self.parquet_path)

        df = pd.read_parquet(self.parquet_path)

        ctx = get_canonical_ctx_contract()
        ctx_cont = list(ctx["ctx_cont_names"])
        ctx_cat = list(ctx["ctx_cat_names"])

        if "seq" in df.columns:
            # ---- advanced schema: builder has prebuilt samples
            required_advanced = [
                "time",
                "seq",
                "snap",
                "ctx_cont",
                "ctx_cat",
                "y_direction",
                "mae_first_n_bps",
                "y_early_move",
                "y_quality_score",
                "y_tradable",
                "mfe_first_n_bps",
                "path_quality_bps",
                "y_bad_path",
            ]
            missing = [c for c in required_advanced if c not in df.columns]
            _require(not missing, f"[ENTRY_V10_CTX_SCHEMA_MISSING] advanced {missing}")

            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            _require(not df["time"].isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
            df = df.sort_values("time").reset_index(drop=True)

            self.df = df
            self._advanced = True
            self.signal_cols = None
            self.ctx_cont_cols = None
            self.ctx_cat_cols = None
            # Infer ctx dims from first row (advanced schema)
            def _to_arr(x, dtype):
                if hasattr(x, "tolist"):
                    x = x.tolist()
                return np.array(x, dtype=dtype)
            first = df.iloc[0]
            self.ctx_cont_dim = int(_to_arr(first["ctx_cont"], np.float32).shape[0])
            self.ctx_cat_dim = int(_to_arr(first["ctx_cat"], np.int64).shape[0])
            self._ctx_vnext_extra = None
            if _is_vnext():
                if self.ctx_cont_dim == 21:
                    self._ctx_vnext_extra = None
                    log.info(
                        "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=prebuilt ctx_cont_dim=%d status=present",
                        self.ctx_cont_dim,
                    )
                elif self.ctx_cont_dim == 16:
                    ts = pd.to_datetime(self.df["time"], utc=True, errors="coerce")
                    _require(not ts.isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
                    sessions = get_session_vectorized(ts)
                    sess_id_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
                    sess_id = sessions.map(sess_id_map).fillna(0).astype("int32")
                    mins_since = get_session_minutes_since_open_vectorized(ts).astype("float32")
                    mins_to = get_session_minutes_to_next_boundary_vectorized(ts).astype("float32")
                    sess_change = sess_id.diff().fillna(0).ne(0).astype("int8")
                    sess_tradable = (sess_id != 0).astype("int8")
                    is_asia = (sess_id == 0).astype("int8")
                    self._ctx_vnext_extra = np.column_stack(
                        [is_asia.values, mins_since.values, mins_to.values, sess_change.values, sess_tradable.values]
                    ).astype(np.float32)
                    self.ctx_cont_dim = self.ctx_cont_dim + 5
                    uniq, cnt = np.unique(sess_id.values, return_counts=True)
                    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
                    log.info(
                        "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=session_detector ctx_cont_dim_base=16 added=5 "
                        "session_id_dist=%s mins_since_range=[%.1f, %.1f] mins_to_range=[%.1f, %.1f]",
                        dist,
                        float(np.nanmin(mins_since.values)),
                        float(np.nanmax(mins_since.values)),
                        float(np.nanmin(mins_to.values)),
                        float(np.nanmax(mins_to.values)),
                    )
                else:
                    raise RuntimeError(
                        f"[ENTRY_V10_CTX_VNEXT_DIM] expected ctx_cont_dim 16 or 21, got {self.ctx_cont_dim}"
                    )

            y = df["y_direction"].astype(int).values
            if not allow_constant_labels:
                if len(np.unique(y)) < 2:
                    raise RuntimeError(
                        "[ENTRY_V10_CTX_LABELS_CONSTANT] "
                        "All y_direction identical. Use --allow-constant-labels only for smoke/plumbing."
                    )

            self.indices = np.arange(len(df))
            _require(len(self.indices) > 0, "[ENTRY_V10_CTX_NO_SAMPLES]")

            log.info(
                f"[DATASET_SCHEMA] advanced | rows={len(df)} samples={len(self.indices)} "
                f"time=[{df['time'].min()} .. {df['time'].max()}]"
            )
        else:
            # ---- flat columns (rolling-window); canary/smoke
            required_signal = list(SIGNAL_FIELDS)
            required = ["time"] + required_signal + ctx_cont + ctx_cat + ["y_direction"]
            missing = [c for c in required if c not in df.columns]
            _require(not missing, f"[ENTRY_V10_CTX_SCHEMA_MISSING] {missing}")

            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            _require(not df["time"].isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
            df = df.sort_values("time").reset_index(drop=True)

            self.df = df
            self._advanced = False
            self.signal_cols = required_signal
            self.ctx_cont_cols = ctx_cont
            self.ctx_cat_cols = ctx_cat
            self.ctx_cont_dim = int(len(ctx_cont))
            self.ctx_cat_dim = int(len(ctx_cat))
            self._ctx_vnext_extra = None
            if _is_vnext():
                ts = pd.to_datetime(self.df["time"], utc=True, errors="coerce")
                _require(not ts.isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
                sessions = get_session_vectorized(ts)
                sess_id_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
                sess_id = sessions.map(sess_id_map).fillna(0).astype("int32")
                mins_since = get_session_minutes_since_open_vectorized(ts).astype("float32")
                mins_to = get_session_minutes_to_next_boundary_vectorized(ts).astype("float32")
                sess_change = sess_id.diff().fillna(0).ne(0).astype("int8")
                sess_tradable = (sess_id != 0).astype("int8")
                is_asia = (sess_id == 0).astype("int8")
                self._ctx_vnext_extra = np.column_stack(
                    [is_asia.values, mins_since.values, mins_to.values, sess_change.values, sess_tradable.values]
                ).astype(np.float32)
                self.ctx_cont_dim = self.ctx_cont_dim + 5
                uniq, cnt = np.unique(sess_id.values, return_counts=True)
                dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
                log.info(
                    "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=session_detector ctx_cont_dim_base=%d added=5 "
                    "session_id_dist=%s mins_since_range=[%.1f, %.1f] mins_to_range=[%.1f, %.1f]",
                    int(len(ctx_cont)),
                    dist,
                    float(np.nanmin(mins_since.values)),
                    float(np.nanmax(mins_since.values)),
                    float(np.nanmin(mins_to.values)),
                    float(np.nanmax(mins_to.values)),
                )

            y = df["y_direction"].astype(int).values
            if not allow_constant_labels:
                if len(np.unique(y)) < 2:
                    raise RuntimeError(
                        "[ENTRY_V10_CTX_LABELS_CONSTANT] "
                        "All y_direction identical. Use --allow-constant-labels only for smoke/plumbing."
                    )

            self.indices = np.arange(self.seq_len - 1, len(df))
            _require(len(self.indices) > 0, "[ENTRY_V10_CTX_NO_SAMPLES] after seq_len windowing")

            log.info(
                f"[DATASET_SCHEMA] flat | rows={len(df)} samples={len(self.indices)} "
                f"time=[{df['time'].min()} .. {df['time'].max()}]"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self._advanced:
            row = self.df.iloc[i]
            # Parquet/pandas may return nested lists or object arrays; force via Python list
            def _to_arr(x, dtype):
                if hasattr(x, "tolist"):
                    x = x.tolist()
                return np.array(x, dtype=dtype)

            seq = _to_arr(row["seq"], np.float32)
            snap = _to_arr(row["snap"], np.float32)
            ctx_cont = _to_arr(row["ctx_cont"], np.float32)
            if _is_vnext() and self._ctx_vnext_extra is not None:
                extra = self._ctx_vnext_extra[i]
                ctx_cont = np.concatenate([ctx_cont, extra.astype(np.float32)], axis=0)
            ctx_cat = _to_arr(row["ctx_cat"], np.int64)
            y = int(np.asarray(row["y_direction"]).ravel()[0])
            if y not in (0, 1, 2):
                raise RuntimeError(f"[ENTRY_V10_CTX_LABEL_INVALID] y_direction={y} expected 0/1/2")

            if seq.shape != (self.seq_len, 7):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] seq shape {seq.shape} expected ({self.seq_len}, 7)"
                )
            if snap.shape != (7,):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] snap shape {snap.shape} expected (7,)"
                )
            if ctx_cont.shape != (self.ctx_cont_dim,):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] ctx_cont shape {ctx_cont.shape} expected ({self.ctx_cont_dim},)"
                )
            if ctx_cat.shape != (self.ctx_cat_dim,):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] ctx_cat shape {ctx_cat.shape} expected ({self.ctx_cat_dim},)"
                )

            return {
                "seq_x": torch.tensor(seq),
                "snap_x": torch.tensor(snap),
                "ctx_cont": torch.tensor(ctx_cont),
                "ctx_cat": torch.tensor(ctx_cat),
                "y": torch.tensor(y, dtype=torch.long),
                "y_tradable": torch.tensor(float(row["y_tradable"]), dtype=torch.float32),
                "mfe_first_n_bps": torch.tensor(float(row["mfe_first_n_bps"]), dtype=torch.float32),
                "path_quality_bps": torch.tensor(float(row["path_quality_bps"]), dtype=torch.float32),
            }
        else:
            t = self.indices[i]
            start = t - self.seq_len + 1

            seq = self.df.iloc[start : t + 1][self.signal_cols].values.astype(np.float32)
            snap = self.df.iloc[t][self.signal_cols].values.astype(np.float32)
            ctx_cont = self.df.iloc[t][self.ctx_cont_cols].values.astype(np.float32)
            if _is_vnext() and self._ctx_vnext_extra is not None:
                extra = self._ctx_vnext_extra[t]
                ctx_cont = np.concatenate([ctx_cont, extra.astype(np.float32)], axis=0)
            ctx_cat = self.df.iloc[t][self.ctx_cat_cols].values.astype(np.int64)
            y = int(self.df.iloc[t]["y_direction"])
            if y not in (0, 1, 2):
                raise RuntimeError(f"[ENTRY_V10_CTX_LABEL_INVALID] y_direction={y} expected 0/1/2")

            return {
                "seq_x": torch.tensor(seq),
                "snap_x": torch.tensor(snap),
                "ctx_cont": torch.tensor(ctx_cont),
                "ctx_cat": torch.tensor(ctx_cat),
                "y": torch.tensor(y, dtype=torch.long),
                "y_tradable": torch.tensor(0.0, dtype=torch.float32),
                "mfe_first_n_bps": torch.tensor(0.0, dtype=torch.float32),
                "path_quality_bps": torch.tensor(0.0, dtype=torch.float32),
            }

# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------
class CostSensitiveCrossEntropyLoss(nn.Module):
    """
    Cost-sensitive cross-entropy for ENTRY 3-class (0=LONG, 1=SHORT, 2=FLAT).
    Base CE uses optional class weights; expected misclassification cost is added
    using a fixed cost matrix indexed by true class.
    """

    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor],
        cost_matrix: torch.Tensor,
        cost_scale: float = 1.0,
        enabled: bool = True,
        balance_alpha: float = 0.0,
        balance_target: str = "label",
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.cost_scale = float(cost_scale)
        self.balance_alpha = float(balance_alpha)
        self.balance_target = str(balance_target).strip().lower()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.register_buffer("cost_matrix", cost_matrix.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)  # (B,)
        if not self.enabled:
            return ce.mean()

        probs = torch.softmax(logits, dim=1)
        cost = self.cost_matrix.to(dtype=logits.dtype)[targets]  # (B,3)
        expected_cost = (cost * probs).sum(dim=1)
        loss = ce + (self.cost_scale * expected_cost)

        if self.balance_alpha > 0.0:
            mean_probs = probs.mean(dim=0)
            if self.balance_target == "uniform":
                target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
            else:
                counts = torch.bincount(targets, minlength=mean_probs.numel()).float()
                denom = counts.sum().clamp(min=1.0)
                target = counts / denom
            balance_loss = torch.mean((mean_probs - target) ** 2)
            loss = loss + (self.balance_alpha * balance_loss)
        return loss.mean()


def _build_cost_sensitive_criterion(
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    cost_long_to_short: float,
    cost_long_to_flat: float,
    cost_short_to_long: float,
    cost_short_to_flat: float,
    cost_flat_to_long: float,
    cost_flat_to_short: float,
    cost_scale: float,
    enabled: bool,
    balance_alpha: float,
    balance_target: str,
) -> Tuple[CostSensitiveCrossEntropyLoss, torch.Tensor]:
    cost_matrix = torch.tensor(
        [
            [0.0, float(cost_long_to_short), float(cost_long_to_flat)],
            [float(cost_short_to_long), 0.0, float(cost_short_to_flat)],
            [float(cost_flat_to_long), float(cost_flat_to_short), 0.0],
        ],
        device=device,
    )
    criterion = CostSensitiveCrossEntropyLoss(
        class_weights=class_weights,
        cost_matrix=cost_matrix,
        cost_scale=float(cost_scale),
        enabled=bool(enabled),
        balance_alpha=float(balance_alpha),
        balance_target=str(balance_target),
    )
    return criterion, cost_matrix


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    short_lead_margin: float,
    long_penalty_weight: float,
    residual_side_bias_alpha: float,
    timing_target_bps: float,
    timing_loss_scale: float,
    aux_early_weight: float,
    aux_quality_weight: float,
    aux_path_weight: float,
    aux_mfe_weight: float,
    aux_tradable_weight: float,
    aux_quality_scale_bps: float,
    aux_path_scale_bps: float,
    aux_mfe_scale_bps: float,
    bad_path_pos_weight: float,
):
    model.train()
    total = 0.0
    total_ce = 0.0
    total_cost = 0.0
    total_balance = 0.0
    total_timing = 0.0
    total_aux_early = 0.0
    total_aux_quality = 0.0
    total_aux_path = 0.0
    total_aux_mfe = 0.0
    total_aux_tradable = 0.0
    total_aux_bad_path = 0.0
    n = 0
    short_total = 0
    short_pred_long = 0
    short_lead_count = 0
    short_lead_long_prob_sum = 0.0
    anchor_abs_sum = 0.0
    delta_abs_sum = 0.0
    scaled_delta_abs_sum = 0.0
    final_minus_anchor_abs_sum = 0.0
    timing_mae_sum = 0.0
    timing_penalty_sum = 0.0
    early_loss_sum = 0.0
    quality_loss_sum = 0.0
    path_loss_sum = 0.0
    mfe_loss_sum = 0.0
    tradable_loss_sum = 0.0
    bad_path_loss_sum = 0.0

    for batch in loader:
        non_blocking = device.type == "cuda"
        seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
        snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
        ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
        ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
        y = batch["y"].to(device, non_blocking=non_blocking)
        y_mfe_first = batch["mfe_first_n_bps"].to(device, non_blocking=non_blocking)
        y_path_quality = batch["path_quality_bps"].to(device, non_blocking=non_blocking)
        y_tradable = batch["y_tradable"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
        logits = out["direction_logits"]
        path_pred = out.get("path_quality")
        mfe_pred = out.get("mfe_first_n")
        tradable_logit = out.get("tradable_logit")
        anchor_logits = out.get("anchor_logits")
        delta_logits = out.get("delta_logits")

        ce_loss_raw = criterion.ce(logits, y).mean()
        ce_loss = float(ENTRY_DIRECTION_CE_SCALE) * ce_loss_raw
        probs = torch.softmax(logits, dim=1)

        cost_term = 0.0
        balance_term = 0.0
        if bool(getattr(criterion, "enabled", False)):
            cost = criterion.cost_matrix.to(dtype=logits.dtype)[y]
            expected_cost = (cost * probs).sum(dim=1)
            cost_term = float(getattr(criterion, "cost_scale", 1.0)) * expected_cost.mean()
            if float(getattr(criterion, "balance_alpha", 0.0)) > 0.0:
                mean_probs = probs.mean(dim=0)
                if str(getattr(criterion, "balance_target", "label")).strip().lower() == "uniform":
                    target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
                else:
                    counts = torch.bincount(y, minlength=mean_probs.numel()).float()
                    denom = counts.sum().clamp(min=1.0)
                    target = counts / denom
                balance_loss = torch.mean((mean_probs - target) ** 2)
                balance_term = float(getattr(criterion, "balance_alpha", 0.0)) * balance_loss

        loss = ce_loss + cost_term + balance_term
        if residual_side_bias_alpha > 0.0 and delta_logits is not None:
            residual_gap = delta_logits[:, 0] - delta_logits[:, 1]
            residual_side_bias_loss = residual_gap.mean().pow(2)
            loss = loss + (float(residual_side_bias_alpha) * residual_side_bias_loss)

        tradable_loss = torch.tensor(0.0, device=device)
        if aux_path_weight > 0.0 and path_pred is not None:
            p_scale = max(1.0, float(aux_path_scale_bps))
            # Runtime only uses path_quality as a positive minimum gate, so train
            # this head on the non-negative quality floor rather than signed path.
            path_target = (y_path_quality / p_scale).clamp(min=0.0)
            path_loss = nn.functional.smooth_l1_loss(
                path_pred.squeeze(1), path_target.float()
            )
            path_loss = float(aux_path_weight) * path_loss
            loss = loss + path_loss
            path_loss_sum += float(path_loss.item()) * y.shape[0]
        if aux_mfe_weight > 0.0 and mfe_pred is not None:
            m_scale = max(1.0, float(aux_mfe_scale_bps))
            mfe_target = (y_mfe_first / m_scale).clamp(min=0.0)
            mfe_loss = nn.functional.smooth_l1_loss(
                mfe_pred.squeeze(1), mfe_target.float()
            )
            mfe_loss = float(aux_mfe_weight) * mfe_loss
            loss = loss + mfe_loss
            mfe_loss_sum += float(mfe_loss.item()) * y.shape[0]
        if aux_tradable_weight > 0.0 and tradable_logit is not None:
            tradable_loss = nn.functional.binary_cross_entropy_with_logits(
                tradable_logit.squeeze(1), y_tradable.float()
            )
            tradable_loss = float(aux_tradable_weight) * tradable_loss
            loss = loss + tradable_loss
            tradable_loss_sum += float(tradable_loss.item()) * y.shape[0]
        if long_penalty_weight > 0.0:
            short_lead_mask = (snap_x[:, 1] - snap_x[:, 0]) >= float(short_lead_margin)
            if short_lead_mask.any():
                short_lead_count += int(short_lead_mask.sum().item())
                short_lead_long_prob = probs[short_lead_mask, 0].mean()
                short_lead_long_prob_sum += float(short_lead_long_prob.item()) * int(short_lead_mask.sum().item())
                loss = loss + float(long_penalty_weight) * short_lead_long_prob

        preds = torch.argmax(probs, dim=1)
        short_mask = y == 1
        if short_mask.any():
            short_total += int(short_mask.sum().item())
            short_pred_long += int(((preds == 0) & short_mask).sum().item())
        if anchor_logits is not None and delta_logits is not None:
            residual_scale = float(getattr(model, "residual_scale", 1.0))
            scaled_delta = delta_logits * residual_scale
            anchor_abs_sum += float(anchor_logits.abs().mean().item()) * y.shape[0]
            delta_abs_sum += float(delta_logits.abs().mean().item()) * y.shape[0]
            scaled_delta_abs_sum += float(scaled_delta.abs().mean().item()) * y.shape[0]
            final_minus_anchor_abs_sum += float((logits - anchor_logits).abs().mean().item()) * y.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = y.shape[0]
        total += float(loss) * bs
        total_ce += float(ce_loss) * bs
        total_cost += float(cost_term) * bs
        total_balance += float(balance_term) * bs
        if aux_path_weight > 0.0:
            total_aux_path += float(path_loss) * bs
        if aux_mfe_weight > 0.0:
            total_aux_mfe += float(mfe_loss) * bs
        if aux_tradable_weight > 0.0:
            total_aux_tradable += float(tradable_loss) * bs
        n += bs

    return total / max(1, n), {
        "ce_loss_mean": (total_ce / max(1, n)),
        "cost_loss_mean": (total_cost / max(1, n)),
        "balance_loss_mean": (total_balance / max(1, n)),
        "aux_path_loss_mean": (total_aux_path / max(1, n)),
        "aux_mfe_loss_mean": (total_aux_mfe / max(1, n)),
        "aux_tradable_loss_mean": (total_aux_tradable / max(1, n)),
        "short_pred_long_rate": (short_pred_long / short_total if short_total > 0 else 0.0),
        "short_lead_count": short_lead_count,
        "short_lead_long_prob_mean": (short_lead_long_prob_sum / short_lead_count if short_lead_count > 0 else 0.0),
        "anchor_abs_mean": (anchor_abs_sum / max(1, n)),
        "delta_abs_mean": (delta_abs_sum / max(1, n)),
        "scaled_delta_abs_mean": (scaled_delta_abs_sum / max(1, n)),
        "final_minus_anchor_abs_mean": (final_minus_anchor_abs_sum / max(1, n)),
        "aux_path_loss_mean": (path_loss_sum / max(1, n)),
        "aux_mfe_loss_mean": (mfe_loss_sum / max(1, n)),
        "aux_tradable_loss_mean": (tradable_loss_sum / max(1, n)),
    }

def validate(
    model,
    loader,
    criterion,
    device,
    residual_side_bias_alpha: float,
    aux_early_weight: float,
    aux_quality_weight: float,
    aux_path_weight: float,
    aux_mfe_weight: float,
    aux_tradable_weight: float,
    aux_quality_scale_bps: float,
    aux_path_scale_bps: float,
    aux_mfe_scale_bps: float,
    bad_path_pos_weight: float,
):
    model.eval()
    total = 0.0
    total_ce = 0.0
    total_cost = 0.0
    total_balance = 0.0
    n = 0
    preds, targets = [], []
    short_total = 0
    short_pred_long = 0
    anchor_abs_sum = 0.0
    delta_abs_sum = 0.0
    scaled_delta_abs_sum = 0.0
    final_minus_anchor_abs_sum = 0.0
    early_loss_sum = 0.0
    quality_loss_sum = 0.0
    path_loss_sum = 0.0
    mfe_loss_sum = 0.0
    tradable_loss_sum = 0.0
    bad_path_loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
            snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
            ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
            ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)
            y_mfe_first = batch["mfe_first_n_bps"].to(device, non_blocking=non_blocking)
            y_path_quality = batch["path_quality_bps"].to(device, non_blocking=non_blocking)
            y_tradable = batch["y_tradable"].to(device, non_blocking=non_blocking)

            out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
            logits = out["direction_logits"]
            path_pred = out.get("path_quality")
            mfe_pred = out.get("mfe_first_n")
            tradable_logit = out.get("tradable_logit")
            anchor_logits = out.get("anchor_logits")
            delta_logits = out.get("delta_logits")

            ce_loss_raw = criterion.ce(logits, y).mean()
            ce_loss = float(ENTRY_DIRECTION_CE_SCALE) * ce_loss_raw
            probs = torch.softmax(logits, dim=1)

            cost_term = 0.0
            balance_term = 0.0
            if bool(getattr(criterion, "enabled", False)):
                cost = criterion.cost_matrix.to(dtype=logits.dtype)[y]
                expected_cost = (cost * probs).sum(dim=1)
                cost_term = float(getattr(criterion, "cost_scale", 1.0)) * expected_cost.mean()
                if float(getattr(criterion, "balance_alpha", 0.0)) > 0.0:
                    mean_probs = probs.mean(dim=0)
                    if str(getattr(criterion, "balance_target", "label")).strip().lower() == "uniform":
                        target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
                    else:
                        counts = torch.bincount(y, minlength=mean_probs.numel()).float()
                        denom = counts.sum().clamp(min=1.0)
                        target = counts / denom
                    balance_loss = torch.mean((mean_probs - target) ** 2)
                    balance_term = float(getattr(criterion, "balance_alpha", 0.0)) * balance_loss

            loss = ce_loss + cost_term + balance_term
            if residual_side_bias_alpha > 0.0 and delta_logits is not None:
                residual_gap = delta_logits[:, 0] - delta_logits[:, 1]
                residual_side_bias_loss = residual_gap.mean().pow(2)
                loss = loss + (float(residual_side_bias_alpha) * residual_side_bias_loss)

            tradable_loss = torch.tensor(0.0, device=device)
            if aux_path_weight > 0.0 and path_pred is not None:
                p_scale = max(1.0, float(aux_path_scale_bps))
                path_target = (y_path_quality / p_scale).clamp(min=0.0)
                path_loss = nn.functional.smooth_l1_loss(
                    path_pred.squeeze(1), path_target.float()
                )
                loss = loss + (float(aux_path_weight) * path_loss)
                path_loss_sum += float(path_loss.item()) * y.shape[0]
            if aux_mfe_weight > 0.0 and mfe_pred is not None:
                m_scale = max(1.0, float(aux_mfe_scale_bps))
                mfe_target = (y_mfe_first / m_scale).clamp(min=0.0)
                mfe_loss = nn.functional.smooth_l1_loss(
                    mfe_pred.squeeze(1), mfe_target.float()
                )
                loss = loss + (float(aux_mfe_weight) * mfe_loss)
                mfe_loss_sum += float(mfe_loss.item()) * y.shape[0]
            if aux_tradable_weight > 0.0 and tradable_logit is not None:
                tradable_loss = nn.functional.binary_cross_entropy_with_logits(
                    tradable_logit.squeeze(1), y_tradable.float()
                )
                loss = loss + (float(aux_tradable_weight) * tradable_loss)
                tradable_loss_sum += float(tradable_loss.item()) * y.shape[0]
            bs = y.shape[0]
            total += float(loss) * bs
            total_ce += float(ce_loss) * bs
            total_cost += float(cost_term) * bs
            total_balance += float(balance_term) * bs
            n += bs

            p = probs.cpu().numpy()
            preds.extend(np.argmax(p, axis=1).tolist())
            targets.extend(y.cpu().numpy().tolist())
            y_np = y.cpu().numpy()
            pred_np = np.argmax(p, axis=1)
            short_total += int((y_np == 1).sum())
            if short_total > 0:
                short_pred_long += int(((pred_np == 0) & (y_np == 1)).sum())
            if anchor_logits is not None and delta_logits is not None:
                residual_scale = float(getattr(model, "residual_scale", 1.0))
                scaled_delta = delta_logits * residual_scale
                anchor_abs_sum += float(anchor_logits.abs().mean().item()) * bs
                delta_abs_sum += float(delta_logits.abs().mean().item()) * bs
                scaled_delta_abs_sum += float(scaled_delta.abs().mean().item()) * bs
                final_minus_anchor_abs_sum += float((logits - anchor_logits).abs().mean().item()) * bs

    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)

    acc = float(accuracy_score(targets_np.astype(int), preds_np.astype(int)))
    short_pred_long_rate = (short_pred_long / short_total if short_total > 0 else 0.0)
    stats = {
        "anchor_abs_mean": (anchor_abs_sum / max(1, n)),
        "delta_abs_mean": (delta_abs_sum / max(1, n)),
        "scaled_delta_abs_mean": (scaled_delta_abs_sum / max(1, n)),
        "final_minus_anchor_abs_mean": (final_minus_anchor_abs_sum / max(1, n)),
        "aux_path_loss_mean": (path_loss_sum / max(1, n)),
        "aux_mfe_loss_mean": (mfe_loss_sum / max(1, n)),
        "aux_tradable_loss_mean": (tradable_loss_sum / max(1, n)),
        "ce_loss_mean": (total_ce / max(1, n)),
        "cost_loss_mean": (total_cost / max(1, n)),
        "balance_loss_mean": (total_balance / max(1, n)),
    }
    # AUC is intentionally disabled for this 3-class path (previously hardcoded 0.0)
    return total / max(1, n), float("nan"), acc, short_pred_long_rate, stats


def _validate_eval(model, loader, criterion, device, residual_side_bias_alpha: float):
    """
    Eval with non-finite guard; returns loss/acc and raises on NaN/Inf.
    """
    model.eval()
    total = 0.0
    n = 0
    preds, targets = [], []
    session_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    anchor_counts = {"long": 0, "short": 0, "flat": 0}
    final_counts = {"long": 0, "short": 0, "flat": 0}
    flip_counts = {
        "anchor_short_to_short": 0,
        "anchor_short_to_long": 0,
        "anchor_long_to_short": 0,
        "anchor_long_to_long": 0,
        "anchor_flat_to_long": 0,
        "anchor_flat_to_short": 0,
    }
    anchor_counts_by_session = {name: {"long": 0, "short": 0, "flat": 0} for name in session_map.values()}
    final_counts_by_session = {name: {"long": 0, "short": 0, "flat": 0} for name in session_map.values()}
    flip_counts_by_session = {
        name: {k: 0 for k in flip_counts.keys()} for name in session_map.values()
    }
    residual_gap_chunks: List[np.ndarray] = []
    anchor_gap_chunks: List[np.ndarray] = []
    ratio_chunks: List[np.ndarray] = []
    ratio_eps = 1e-8

    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
            snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
            ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
            ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)

            out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
            logits = out["direction_logits"]
            anchor_logits = out.get("anchor_logits")
            delta_logits = out.get("delta_logits")

            if not torch.isfinite(logits).all():
                raise RuntimeError("[EVAL_NON_FINITE] logits contain non-finite values")

            loss = float(ENTRY_DIRECTION_CE_SCALE) * criterion(logits, y)
            if residual_side_bias_alpha > 0.0 and delta_logits is not None:
                residual_gap = delta_logits[:, 0] - delta_logits[:, 1]
                residual_side_bias_loss = residual_gap.mean().pow(2)
                loss = loss + (float(residual_side_bias_alpha) * residual_side_bias_loss)
            bs = y.shape[0]
            total += float(loss) * bs
            n += bs

            prob = torch.softmax(logits, dim=1)
            if not torch.isfinite(prob).all():
                raise RuntimeError("[EVAL_NON_FINITE] probs contain non-finite values")

            preds.extend(np.argmax(prob.cpu().numpy(), axis=1).tolist())
            targets.extend(y.cpu().numpy().tolist())

            if anchor_logits is not None:
                anchor_side = torch.argmax(anchor_logits, dim=1)
                final_side = torch.argmax(logits, dim=1)

                anchor_counts["long"] += int((anchor_side == 0).sum().item())
                anchor_counts["short"] += int((anchor_side == 1).sum().item())
                anchor_counts["flat"] += int((anchor_side == 2).sum().item())
                final_counts["long"] += int((final_side == 0).sum().item())
                final_counts["short"] += int((final_side == 1).sum().item())
                final_counts["flat"] += int((final_side == 2).sum().item())

                flip_counts["anchor_short_to_short"] += int(((anchor_side == 1) & (final_side == 1)).sum().item())
                flip_counts["anchor_short_to_long"] += int(((anchor_side == 1) & (final_side == 0)).sum().item())
                flip_counts["anchor_long_to_short"] += int(((anchor_side == 0) & (final_side == 1)).sum().item())
                flip_counts["anchor_long_to_long"] += int(((anchor_side == 0) & (final_side == 0)).sum().item())
                flip_counts["anchor_flat_to_long"] += int(((anchor_side == 2) & (final_side == 0)).sum().item())
                flip_counts["anchor_flat_to_short"] += int(((anchor_side == 2) & (final_side == 1)).sum().item())

                sessions = ctx_cat[:, 0].cpu().numpy()
                for sess_id, sess_name in session_map.items():
                    sess_mask = sessions == sess_id
                    if not np.any(sess_mask):
                        continue
                    sess_anchor = anchor_side[sess_mask]
                    sess_final = final_side[sess_mask]
                    anchor_counts_by_session[sess_name]["long"] += int((sess_anchor == 0).sum().item())
                    anchor_counts_by_session[sess_name]["short"] += int((sess_anchor == 1).sum().item())
                    anchor_counts_by_session[sess_name]["flat"] += int((sess_anchor == 2).sum().item())
                    final_counts_by_session[sess_name]["long"] += int((sess_final == 0).sum().item())
                    final_counts_by_session[sess_name]["short"] += int((sess_final == 1).sum().item())
                    final_counts_by_session[sess_name]["flat"] += int((sess_final == 2).sum().item())
                    flip_counts_by_session[sess_name]["anchor_short_to_short"] += int(
                        ((sess_anchor == 1) & (sess_final == 1)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_short_to_long"] += int(
                        ((sess_anchor == 1) & (sess_final == 0)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_long_to_short"] += int(
                        ((sess_anchor == 0) & (sess_final == 1)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_long_to_long"] += int(
                        ((sess_anchor == 0) & (sess_final == 0)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_flat_to_long"] += int(
                        ((sess_anchor == 2) & (sess_final == 0)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_flat_to_short"] += int(
                        ((sess_anchor == 2) & (sess_final == 1)).sum().item()
                    )

                if delta_logits is not None:
                    residual_scale = float(getattr(model, "residual_scale", 1.0))
                    residual_gap = residual_scale * (delta_logits[:, 0] - delta_logits[:, 1])
                    anchor_gap = anchor_logits[:, 1] - anchor_logits[:, 0]
                    ratio = torch.abs(residual_gap) / (torch.abs(anchor_gap) + ratio_eps)
                    residual_gap_chunks.append(residual_gap.detach().cpu().numpy())
                    anchor_gap_chunks.append(anchor_gap.detach().cpu().numpy())
                    ratio_chunks.append(ratio.detach().cpu().numpy())

    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)

    acc = float(accuracy_score(targets_np.astype(int), preds_np.astype(int)))

    def _stat_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
        if values.size == 0:
            return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    total_anchor = sum(anchor_counts.values())
    total_final = sum(final_counts.values())
    if total_anchor > 0:
        if total_final > 0:
            log.info(
                "[ENTRY_PRED_DIST_PROOF] pred_long_pct=%.6f pred_short_pct=%.6f pred_flat_pct=%.6f",
                final_counts["long"] / total_final,
                final_counts["short"] / total_final,
                final_counts["flat"] / total_final,
            )
            for sess_name in session_map.values():
                sess_total = sum(final_counts_by_session[sess_name].values())
                if sess_total > 0:
                    log.info(
                        "[ENTRY_PRED_DIST_PROOF] session=%s pred_long_pct=%.6f pred_short_pct=%.6f pred_flat_pct=%.6f",
                        sess_name,
                        final_counts_by_session[sess_name]["long"] / sess_total,
                        final_counts_by_session[sess_name]["short"] / sess_total,
                        final_counts_by_session[sess_name]["flat"] / sess_total,
                    )
        log.info(
            "[ENTRY_ANCHOR_DIST_PROOF] anchor_long_pct=%.6f anchor_short_pct=%.6f anchor_flat_pct=%.6f",
            anchor_counts["long"] / total_anchor,
            anchor_counts["short"] / total_anchor,
            anchor_counts["flat"] / total_anchor,
        )
        for sess_name in session_map.values():
            sess_total = sum(anchor_counts_by_session[sess_name].values())
            if sess_total > 0:
                log.info(
                    "[ENTRY_ANCHOR_DIST_PROOF] session=%s anchor_long_pct=%.6f anchor_short_pct=%.6f anchor_flat_pct=%.6f",
                    sess_name,
                    anchor_counts_by_session[sess_name]["long"] / sess_total,
                    anchor_counts_by_session[sess_name]["short"] / sess_total,
                    anchor_counts_by_session[sess_name]["flat"] / sess_total,
                )

        log.info(
            "[ENTRY_ANCHOR_FLIP_PROOF] anchor_short_to_short=%d anchor_short_to_long=%d anchor_long_to_short=%d "
            "anchor_long_to_long=%d anchor_flat_to_long=%d anchor_flat_to_short=%d",
            flip_counts["anchor_short_to_short"],
            flip_counts["anchor_short_to_long"],
            flip_counts["anchor_long_to_short"],
            flip_counts["anchor_long_to_long"],
            flip_counts["anchor_flat_to_long"],
            flip_counts["anchor_flat_to_short"],
        )
        for sess_name in session_map.values():
            sess_counts = flip_counts_by_session[sess_name]
            if sum(sess_counts.values()) > 0:
                log.info(
                    "[ENTRY_ANCHOR_FLIP_PROOF] session=%s anchor_short_to_short=%d anchor_short_to_long=%d "
                    "anchor_long_to_short=%d anchor_long_to_long=%d anchor_flat_to_long=%d anchor_flat_to_short=%d",
                    sess_name,
                    sess_counts["anchor_short_to_short"],
                    sess_counts["anchor_short_to_long"],
                    sess_counts["anchor_long_to_short"],
                    sess_counts["anchor_long_to_long"],
                    sess_counts["anchor_flat_to_long"],
                    sess_counts["anchor_flat_to_short"],
                )

    if residual_gap_chunks and anchor_gap_chunks and ratio_chunks:
        residual_gap_all = np.concatenate(residual_gap_chunks, axis=0)
        anchor_gap_all = np.concatenate(anchor_gap_chunks, axis=0)
        ratio_all = np.concatenate(ratio_chunks, axis=0)

        res_stats = _stat_summary(residual_gap_all)
        anc_stats = _stat_summary(anchor_gap_all)
        ratio_stats = _stat_summary(ratio_all)

        log.info(
            "[ENTRY_RESIDUAL_GAP_PROOF] mean_gap=%.6f median_gap=%.6f p90_gap=%.6f p95_gap=%.6f max_gap=%.6f",
            res_stats["mean"] or 0.0,
            res_stats["median"] or 0.0,
            res_stats["p90"] or 0.0,
            res_stats["p95"] or 0.0,
            res_stats["max"] or 0.0,
        )
        log.info(
            "[ENTRY_ANCHOR_GAP_PROOF] mean_anchor_gap=%.6f median_anchor_gap=%.6f p90_anchor_gap=%.6f",
            anc_stats["mean"] or 0.0,
            anc_stats["median"] or 0.0,
            anc_stats["p90"] or 0.0,
        )
        log.info(
            "[ENTRY_RESIDUAL_VS_ANCHOR_PROOF] mean_ratio=%.6f p90_ratio=%.6f p95_ratio=%.6f",
            ratio_stats["mean"] or 0.0,
            ratio_stats["p90"] or 0.0,
            ratio_stats["p95"] or 0.0,
        )

    # AUC disabled for this 3-class path
    return total / max(1, n), float("nan"), acc

# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
def run_sanity_check(
    seq_len: int,
    seed: int,
    device: torch.device,
    out_bundle_dir: Path,
    dataset_manifest: Optional[Path] = None,
    deterministic: bool = True,
) -> None:
    """
    Contract + dummy forward + write minimal bundle + reload with runtime loader (strict).
    Fail-fast with clear error labels.
    """
    _guard_no_rl()

    if dataset_manifest is not None:
        p = Path(dataset_manifest).expanduser().resolve()
        _require(p.exists(), f"[SANITY_MANIFEST_MISSING] {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        fc = data.get("feature_contract") or {}
        fc_ctx_cont_dim = int(fc.get("ctx_cont_dim") or -1)
        fc_ctx_cat_dim = int(fc.get("ctx_cat_dim") or -1)
        _require(
            fc.get("ctx_tag") == "CTX6CAT6"
            and fc_ctx_cont_dim >= 6
            and fc_ctx_cat_dim == 6,
            f"[SANITY_MANIFEST_CONTRACT] manifest feature_contract must be CTX6CAT6 ctx_cont_dim>=6 ctx_cat_dim=6, got {fc}",
        )
        if fc.get("ctx_cont_base_dim") is not None:
            _require(
                int(fc.get("ctx_cont_base_dim")) == 6,
                f"[SANITY_MANIFEST_CTX_BASE] expected ctx_cont_base_dim=6, got {fc.get('ctx_cont_base_dim')}",
            )
        _require(
            fc.get("signal_bridge_id") == "XGB_SIGNAL_BRIDGE_V1",
            f"[SANITY_MANIFEST_SIGNAL] expected XGB_SIGNAL_BRIDGE_V1, got {fc.get('signal_bridge_id')}",
        )
        log.info(f"[SANITY] manifest contract OK: {p}")

    ctx = get_canonical_ctx_contract()
    _require(ctx["tag"] == "CTX6CAT6", "[SANITY_CTX_SPLIT_BRAIN] expected CTX6CAT6")
    if not _is_vnext():
        _require(
            ctx.get("ctx_cont_dim") == 6 and ctx.get("ctx_cat_dim") == 6,
            "[SANITY_CTX_DIM_MISMATCH] expected ctx_cont_base=6 ctx_cat_dim=6",
        )
    _require(SEQ_SIGNAL_DIM == 7 and SNAP_SIGNAL_DIM == 7, "[SANITY_SIGNAL_DIM] expected 7/7")

    if dataset_manifest is not None:
        ctx_cont_dim = int(fc_ctx_cont_dim)
        ctx_cat_dim = int(fc_ctx_cat_dim)
    else:
        ctx_cont_dim = int(ctx.get("ctx_cont_dim") or 6)
        ctx_cat_dim = int(ctx.get("ctx_cat_dim") or 6)
    if _is_vnext():
        ctx_cont_dim = max(ctx_cont_dim, 21)

    log.info(
        f"[SANITY] seed={seed} device={device} "
        f"signal_bridge=7 ctx_cont={ctx_cont_dim} ctx_cat={ctx_cat_dim} seq_len={seq_len}"
    )

    _set_deterministic(seed, device, deterministic=deterministic)

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_SIGNAL_DIM,
        snap_input_dim=SNAP_SIGNAL_DIM,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        residual_scale=float(ENTRY_RESIDUAL_SCALE),
        anchor_eps=float(ENTRY_ANCHOR_EPS),
    ).to(device)
    try:
        head_out = int(getattr(model.head_direction, "out_features", -1))
    except Exception:
        head_out = -1
    log.info("[ENTRY_3CLASS_PROOF] head_direction_out=%s", head_out)
    try:
        head_out = int(getattr(model.head_direction, "out_features", -1))
    except Exception:
        head_out = -1
    log.info("[ENTRY_3CLASS_PROOF] head_direction_out=%s", head_out)

    # Dummy batch: per-sample ctx (B, ctx_*) as in dataset/trening
    B, T = 4, seq_len
    dummy_seq = torch.randn(B, T, 7, device=device, dtype=torch.float32)
    dummy_snap = torch.randn(B, 7, device=device, dtype=torch.float32)
    dummy_ctx_cont = torch.randn(B, ctx_cont_dim, device=device, dtype=torch.float32)
    dummy_ctx_cat = torch.randint(0, 256, (B, ctx_cat_dim), device=device, dtype=torch.int64)

    with torch.no_grad():
        out = model(
            dummy_seq,
            dummy_snap,
            ctx_cat=dummy_ctx_cat,
            ctx_cont=dummy_ctx_cont,
        )

    direction_logits = out["direction_logits"]
    _require(
        direction_logits.dim() == 2 and direction_logits.shape[1] == 3,
        f"[SANITY_OUTPUT_SHAPE] expected (B,3) got {tuple(direction_logits.shape)}",
    )
    _require(
        direction_logits.dtype == torch.float32,
        f"[SANITY_OUTPUT_DTYPE] expected float32 got {direction_logits.dtype}",
    )
    if torch.isnan(direction_logits).any() or torch.isinf(direction_logits).any():
        raise RuntimeError("[SANITY_NAN_INF] direction_logits contains NaN/Inf")

    log.info(
        f"[SANITY] forward OK shapes seq={dummy_seq.shape} snap={dummy_snap.shape} "
        f"ctx_cont=(B,{ctx_cont_dim}) ctx_cat=(B,{ctx_cat_dim}) out={direction_logits.shape}"
    )

    # Write minimal sanity bundle
    out_bundle_dir = Path(out_bundle_dir).expanduser().resolve()
    out_bundle_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_bundle_dir / "model_state_dict.pt"
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state_dict, state_path)

    ordered_ctx_cont_names = _build_ordered_ctx_cont_names(ctx_cont_dim, list(ctx.get("ctx_cont_names") or []))

    lock = {
        "version": "entry_v10_ctx_lock_v1",
        "created_at_utc": _utc_now(),
        "signal_bridge_contract_sha256": SIGNAL_BRIDGE_CONTRACT_SHA256,
        "seq_input_dim": SEQ_SIGNAL_DIM,
        "snap_input_dim": SNAP_SIGNAL_DIM,
        "seq_len": seq_len,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": _sha256_file(state_path),
    }
    (out_bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(lock, indent=2)
    )

    feature_meta_path = out_bundle_dir / "feature_meta.json"
    feature_meta_path.write_text(json.dumps({"sanity": True, "placeholder": True}))

    meta = {
        "created_at_utc": _utc_now(),
        "seq_input_dim": SEQ_SIGNAL_DIM,
        "snap_input_dim": SNAP_SIGNAL_DIM,
        "seq_len": seq_len,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "supports_context_features": True,
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "ctx_tag": "CTX6CAT6",
        "ordered_ctx_cont_names": ordered_ctx_cont_names,
        "ordered_ctx_cat_names": list(ctx.get("ctx_cat_names") or []),
        "feature_meta_path": str(feature_meta_path.name),
        "sanity_bundle": True,
    }
    (out_bundle_dir / "bundle_metadata.json").write_text(json.dumps(meta, indent=2))

    # Reload with runtime loader (strict=True in loader)
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle

    bundle = load_entry_v10_ctx_bundle(
        bundle_dir=out_bundle_dir,
        feature_meta_path=feature_meta_path,
        device="cpu",
        xgb_models=None,
    )
    with torch.no_grad():
        _ = bundle.transformer_model(
            dummy_seq.cpu(),
            dummy_snap.cpu(),
            ctx_cat=dummy_ctx_cat.cpu(),
            ctx_cont=dummy_ctx_cont.cpu(),
        )
    log.info("[SANITY] strict load + forward OK")

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def run_train(
    train_parquet: Path,
    val_parquet: Path,
    seq_len: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    out_bundle_dir: Path,
    gx1_data_override: str,
    allow_constant_labels: bool,
    num_workers: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    deterministic: bool = True,
) -> None:
    _guard_no_rl()

    ctx = get_canonical_ctx_contract()
    _require(ctx["tag"] == "CTX6CAT6", "[CTX_SPLIT_BRAIN]")
    _require(SEQ_SIGNAL_DIM == 7 and SNAP_SIGNAL_DIM == 7, "[SIGNAL_DIM_SPLIT_BRAIN]")

    log.info(
        f"[TRAIN] seed={seed} device={device} batch_size={batch_size} epochs={epochs} lr={lr} "
        f"signal=7 ctx_cont=dynamic ctx_cat=6 early_stop_patience={early_stopping_patience} "
        f"early_stop_min_delta={early_stopping_min_delta}"
    )

    _set_deterministic(seed, device, deterministic=deterministic)

    _log_label_distribution(train_parquet, split="train")
    _log_label_distribution(val_parquet, split="val")

    train_ds = EntryV10CtxDataset(
        train_parquet,
        seq_len=seq_len,
        allow_constant_labels=allow_constant_labels,
    )
    val_ds = EntryV10CtxDataset(
        val_parquet,
        seq_len=seq_len,
        allow_constant_labels=True,
    )
    train_bad_path_rate = float(train_ds.df["y_bad_path"].astype(float).mean()) if "y_bad_path" in train_ds.df.columns else 0.0
    val_bad_path_rate = float(val_ds.df["y_bad_path"].astype(float).mean()) if "y_bad_path" in val_ds.df.columns else 0.0
    if train_bad_path_rate > 0.0:
        raw_bad_path_pos_weight = (1.0 - train_bad_path_rate) / max(train_bad_path_rate, 1e-9)
    else:
        raw_bad_path_pos_weight = 1.0
    bad_path_pos_weight = float(
        min(float(ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP), max(1.0, raw_bad_path_pos_weight))
    )
    log.info(
        "[ENTRY_BAD_PATH_BALANCE_PROOF] train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f capped_pos_weight=%.6f cap=%.3f",
        train_bad_path_rate,
        val_bad_path_rate,
        raw_bad_path_pos_weight,
        bad_path_pos_weight,
        float(ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP),
    )

    use_cuda = device.type == "cuda"
    if use_cuda and num_workers == 0:
        num_workers = max(2, min(8, (os.cpu_count() or 4)))
    pin_memory = bool(use_cuda)
    persistent_workers = bool(num_workers > 0)
    prefetch_factor = 2 if num_workers > 0 else None
    log.info(
        "[DATALOADER_CONFIG] num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        num_workers,
        pin_memory,
        persistent_workers,
        str(prefetch_factor),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Before first epoch: log sample shapes and confirm contract signal=7, ctx_cat=6, ctx_cont>=6
    sample = next(iter(train_loader))
    ctx_cont_dim = int(sample["ctx_cont"].shape[1])
    ctx_cat_dim = int(sample["ctx_cat"].shape[1])
    base_ctx_cont_names = list(ctx.get("ctx_cont_names") or [])
    ordered_ctx_cont_names = _build_ordered_ctx_cont_names(ctx_cont_dim, base_ctx_cont_names)
    ordered_ctx_cat_names = list(ctx.get("ctx_cat_names") or [])
    log.info(
        f"[TRAIN_CONTRACT] seq_x={sample['seq_x'].shape} snap_x={sample['snap_x'].shape} "
        f"ctx_cont={sample['ctx_cont'].shape} ctx_cat={sample['ctx_cat'].shape}"
    )
    log.info(
        "[ENTRY_INPUT_SCHEMA_PROOF] signal_dim=7 ctx_cont_dim=%d ctx_cat_dim=%d",
        ctx_cont_dim,
        ctx_cat_dim,
    )
    expected_ctx_cont_dim = _expected_ctx_cont_dim()
    _require(
        ctx_cont_dim == expected_ctx_cont_dim,
        f"[ENTRY_CTX_CONT_DIM_MISMATCH] expected ctx_cont_dim={expected_ctx_cont_dim} got={ctx_cont_dim}",
    )
    _require(ctx_cat_dim == 6, f"[ENTRY_CTX_CAT_DIM_MISMATCH] expected ctx_cat_dim=6 got={ctx_cat_dim}")
    if ctx_cont_dim > 6:
        log.info(
            "[ENTRY_MICRO_FEATURES_PROOF] names=%s count=%d",
            list(MICRO_FEATURE_NAMES),
            len(MICRO_FEATURE_NAMES),
        )
        log.info(
            "[ENTRY_SWING_FEATURES_PROOF] names=%s count=%d",
            list(SWING_FEATURE_NAMES),
            len(SWING_FEATURE_NAMES),
        )
    _require(
        sample["seq_x"].shape[2] == 7
        and sample["ctx_cont"].shape[1] == ctx_cont_dim
        and sample["ctx_cat"].shape[1] == ctx_cat_dim,
        "[TRAIN_CONTRACT_MISMATCH] expected signal=7 ctx_cont=dynamic ctx_cat=dynamic",
    )

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_SIGNAL_DIM,
        snap_input_dim=SNAP_SIGNAL_DIM,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
    ).to(device)
    try:
        head_out = int(getattr(model.head_direction, "out_features", -1))
    except Exception:
        head_out = -1
    log.info("[ENTRY_3CLASS_PROOF] head_direction_out=%s", head_out)
    log.info(
        "[ENTRY_CTX_SCALE] ctx_cat_scale=%.3f ctx_cont_scale=%.3f",
        float(model.cfg.ctx_cat_scale),
        float(model.cfg.ctx_cont_scale),
    )
    log.info(
        "[ENTRY_ANCHORED_PROOF] enabled=1 residual_scale=%.3f anchor_source=signal7_p_long_short_flat anchor_eps=%.8f",
        float(ENTRY_RESIDUAL_SCALE),
        float(ENTRY_ANCHOR_EPS),
    )

    _require_nonneg("ENTRY_COST_SENSITIVE_SCALE", ENTRY_COST_SENSITIVE_SCALE)
    _require_nonneg("ENTRY_COST_LONG_TO_SHORT", ENTRY_COST_LONG_TO_SHORT)
    _require_nonneg("ENTRY_COST_LONG_TO_FLAT", ENTRY_COST_LONG_TO_FLAT)
    _require_nonneg("ENTRY_COST_SHORT_TO_LONG", ENTRY_COST_SHORT_TO_LONG)
    _require_nonneg("ENTRY_COST_SHORT_TO_FLAT", ENTRY_COST_SHORT_TO_FLAT)
    _require_nonneg("ENTRY_COST_FLAT_TO_LONG", ENTRY_COST_FLAT_TO_LONG)
    _require_nonneg("ENTRY_COST_FLAT_TO_SHORT", ENTRY_COST_FLAT_TO_SHORT)
    _require_nonneg("ENTRY_PRED_BALANCE_ALPHA", ENTRY_PRED_BALANCE_ALPHA)
    _require_nonneg("ENTRY_RESIDUAL_SCALE", ENTRY_RESIDUAL_SCALE)
    _require_nonneg("ENTRY_ANCHOR_EPS", ENTRY_ANCHOR_EPS)
    _require_nonneg("ENTRY_RESIDUAL_SIDE_BIAS_ALPHA", ENTRY_RESIDUAL_SIDE_BIAS_ALPHA)
    if ENTRY_PRED_BALANCE_TARGET not in ("label", "uniform"):
        raise RuntimeError(
            f"[ENTRY_BALANCE_TARGET_INVALID] ENTRY_PRED_BALANCE_TARGET={ENTRY_PRED_BALANCE_TARGET!r} "
            "expected 'label' or 'uniform'"
        )

    class_weights = torch.tensor([1.0, float(SHORT_CLASS_WEIGHT), 1.0], device=device)
    criterion, cost_matrix = _build_cost_sensitive_criterion(
        device=device,
        class_weights=class_weights,
        cost_long_to_short=float(ENTRY_COST_LONG_TO_SHORT),
        cost_long_to_flat=float(ENTRY_COST_LONG_TO_FLAT),
        cost_short_to_long=float(ENTRY_COST_SHORT_TO_LONG),
        cost_short_to_flat=float(ENTRY_COST_SHORT_TO_FLAT),
        cost_flat_to_long=float(ENTRY_COST_FLAT_TO_LONG),
        cost_flat_to_short=float(ENTRY_COST_FLAT_TO_SHORT),
        cost_scale=float(ENTRY_COST_SENSITIVE_SCALE),
        enabled=bool(ENTRY_COST_SENSITIVE_ENABLED),
        balance_alpha=float(ENTRY_PRED_BALANCE_ALPHA),
        balance_target=str(ENTRY_PRED_BALANCE_TARGET),
    )
    log.info(
        "[ENTRY_TRAIN_RECIPE] direction_ce_scale=%.3f residual_scale=%.3f tradable_w=%.3f path_w=%.3f mfe_w=%.3f",
        float(ENTRY_DIRECTION_CE_SCALE),
        float(ENTRY_RESIDUAL_SCALE),
        float(ENTRY_AUX_TRADABLE_WEIGHT),
        float(ENTRY_AUX_PATH_WEIGHT),
        float(ENTRY_AUX_MFE_WEIGHT),
    )
    log.info(
        "[ENTRY_TRAIN_PARKED] cost_sensitive=%d cost_scale=%.3f pred_balance_alpha=%.3f residual_side_bias_alpha=%.3f "
        "timing_scale=%.3f early_w=%.3f quality_w=%.3f bad_path_w=%.3f xgb_short_penalty=%.3f short_class_weight=%.3f",
        int(bool(ENTRY_COST_SENSITIVE_ENABLED)),
        float(ENTRY_COST_SENSITIVE_SCALE),
        float(ENTRY_PRED_BALANCE_ALPHA),
        float(ENTRY_RESIDUAL_SIDE_BIAS_ALPHA),
        float(ENTRY_TIMING_LOSS_SCALE),
        float(ENTRY_AUX_EARLY_WEIGHT),
        float(ENTRY_AUX_QUALITY_WEIGHT),
        float(ENTRY_AUX_BAD_PATH_WEIGHT),
        float(XGB_SHORT_LONG_PENALTY),
        float(SHORT_CLASS_WEIGHT),
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    last_epoch = 0
    early_stopped = False

    for epoch in range(epochs):
        last_epoch = epoch + 1
        tr_loss, tr_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            short_lead_margin=XGB_SHORT_LEAD_MARGIN,
            long_penalty_weight=XGB_SHORT_LONG_PENALTY,
            residual_side_bias_alpha=ENTRY_RESIDUAL_SIDE_BIAS_ALPHA,
            timing_target_bps=ENTRY_TIMING_TARGET_BPS,
            timing_loss_scale=ENTRY_TIMING_LOSS_SCALE,
            aux_early_weight=ENTRY_AUX_EARLY_WEIGHT,
            aux_quality_weight=ENTRY_AUX_QUALITY_WEIGHT,
            aux_path_weight=ENTRY_AUX_PATH_WEIGHT,
            aux_mfe_weight=ENTRY_AUX_MFE_WEIGHT,
            aux_tradable_weight=ENTRY_AUX_TRADABLE_WEIGHT,
            aux_quality_scale_bps=ENTRY_AUX_QUALITY_SCALE_BPS,
            aux_path_scale_bps=ENTRY_AUX_PATH_SCALE_BPS,
            aux_mfe_scale_bps=ENTRY_AUX_MFE_SCALE_BPS,
            bad_path_pos_weight=bad_path_pos_weight,
        )
        va_loss, auc, acc, val_short_to_long, val_stats = validate(
            model,
            val_loader,
            criterion,
            device,
            residual_side_bias_alpha=ENTRY_RESIDUAL_SIDE_BIAS_ALPHA,
            aux_early_weight=ENTRY_AUX_EARLY_WEIGHT,
            aux_quality_weight=ENTRY_AUX_QUALITY_WEIGHT,
            aux_path_weight=ENTRY_AUX_PATH_WEIGHT,
            aux_mfe_weight=ENTRY_AUX_MFE_WEIGHT,
            aux_tradable_weight=ENTRY_AUX_TRADABLE_WEIGHT,
            aux_quality_scale_bps=ENTRY_AUX_QUALITY_SCALE_BPS,
            aux_path_scale_bps=ENTRY_AUX_PATH_SCALE_BPS,
            aux_mfe_scale_bps=ENTRY_AUX_MFE_SCALE_BPS,
            bad_path_pos_weight=bad_path_pos_weight,
        )
        auc_display = "DISABLED" if not np.isfinite(auc) else f"{auc:.4f}"
        log.info(
            f"[EPOCH {epoch+1}/{epochs}] "
            f"train={tr_loss:.6f} val={va_loss:.6f} auc={auc_display} acc={acc:.4f} "
            f"short_to_long_val={val_short_to_long:.6f}"
        )
        if tr_stats:
            anchor_abs_mean = float(tr_stats.get("anchor_abs_mean") or 0.0)
            delta_abs_mean = float(tr_stats.get("delta_abs_mean") or 0.0)
            scaled_delta_abs_mean = float(tr_stats.get("scaled_delta_abs_mean") or 0.0)
            final_minus_anchor_abs_mean = float(tr_stats.get("final_minus_anchor_abs_mean") or 0.0)
            ratio = (scaled_delta_abs_mean / max(anchor_abs_mean, 1e-12))
            log.info(
                "[ENTRY_RESIDUAL_MAG_PROOF] split=train epoch=%d "
                "anchor_abs_mean=%.6f delta_abs_mean=%.6f scaled_delta_abs_mean=%.6f "
                "final_minus_anchor_abs_mean=%.6f scaled_delta_to_anchor_ratio=%.6f",
                epoch + 1,
                anchor_abs_mean,
                delta_abs_mean,
                scaled_delta_abs_mean,
                final_minus_anchor_abs_mean,
                ratio,
            )
        if val_stats:
            anchor_abs_mean = float(val_stats.get("anchor_abs_mean") or 0.0)
            delta_abs_mean = float(val_stats.get("delta_abs_mean") or 0.0)
            scaled_delta_abs_mean = float(val_stats.get("scaled_delta_abs_mean") or 0.0)
            final_minus_anchor_abs_mean = float(val_stats.get("final_minus_anchor_abs_mean") or 0.0)
            ratio = (scaled_delta_abs_mean / max(anchor_abs_mean, 1e-12))
            log.info(
                "[ENTRY_RESIDUAL_MAG_PROOF] split=val epoch=%d "
                "anchor_abs_mean=%.6f delta_abs_mean=%.6f scaled_delta_abs_mean=%.6f "
                "final_minus_anchor_abs_mean=%.6f scaled_delta_to_anchor_ratio=%.6f",
                epoch + 1,
                anchor_abs_mean,
                delta_abs_mean,
                scaled_delta_abs_mean,
                final_minus_anchor_abs_mean,
                ratio,
            )
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=val epoch=%d ce=%.6f path=%.6f mfe=%.6f tradable=%.6f total=%.6f",
                epoch + 1,
                float(val_stats.get("ce_loss_mean", 0.0)),
                float(val_stats.get("aux_path_loss_mean", 0.0)),
                float(val_stats.get("aux_mfe_loss_mean", 0.0)),
                float(val_stats.get("aux_tradable_loss_mean", 0.0)),
                float(va_loss),
            )
        log.info(
            "[SHORT_TO_LONG_TRAIN] rate=%.6f short_lead_count=%d short_lead_long_prob_mean=%.6f",
            float(tr_stats.get("short_pred_long_rate", 0.0)),
            int(tr_stats.get("short_lead_count", 0)),
            float(tr_stats.get("short_lead_long_prob_mean", 0.0)),
        )
        if tr_stats:
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=train epoch=%d ce=%.6f path=%.6f mfe=%.6f tradable=%.6f total=%.6f",
                epoch + 1,
                float(tr_stats.get("ce_loss_mean", 0.0)),
                float(tr_stats.get("aux_path_loss_mean", 0.0)),
                float(tr_stats.get("aux_mfe_loss_mean", 0.0)),
                float(tr_stats.get("aux_tradable_loss_mean", 0.0)),
                float(tr_loss),
            )
        if (best_val - va_loss) > float(early_stopping_min_delta):
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_since_improve = 0
            log.info(
                "[BEST_CHECKPOINT] epoch=%d val=%.6f",
                best_epoch,
                best_val,
            )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= int(early_stopping_patience):
                early_stopped = True
                log.info(
                    "[EARLY_STOP] epoch=%d best_epoch=%d best_val=%.6f patience=%d min_delta=%.6f",
                    epoch + 1,
                    best_epoch,
                    best_val,
                    int(early_stopping_patience),
                    float(early_stopping_min_delta),
                )
                break

    _require(best_state is not None, "[TRAIN_FAIL_NO_BEST_STATE]")

    # Resolve output bundle dir (under GX1_DATA if relative)
    out_bundle_dir = Path(out_bundle_dir).expanduser().resolve()
    if not out_bundle_dir.is_absolute():
        gx1_data = _resolve_gx1_data(gx1_data_override)
        out_bundle_dir = gx1_data / out_bundle_dir
    out_bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_bundle_dir / "model_state_dict.pt"
    torch.save(best_state, model_path)
    state_dict_sha256 = _sha256_file(model_path)

    lock = {
        "version": "entry_v10_ctx_lock_v1",
        "created_at_utc": _utc_now(),
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "signal_bridge_contract_sha256": SIGNAL_BRIDGE_CONTRACT_SHA256,
        "ctx_tag": "CTX6CAT6",
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "seq_input_dim": SEQ_SIGNAL_DIM,
        "snap_input_dim": SNAP_SIGNAL_DIM,
        "seq_len": seq_len,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": state_dict_sha256,
    }
    (out_bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(lock, indent=2)
    )

    meta = {
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "train_data": str(train_parquet),
        "val_data": str(val_parquet),
        "train_data_sha256": _sha256_file(Path(train_parquet)),
        "val_data_sha256": _sha256_file(Path(val_parquet)),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "early_stopped": bool(early_stopped),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "seq_input_dim": SEQ_SIGNAL_DIM,
        "snap_input_dim": SNAP_SIGNAL_DIM,
        "seq_len": seq_len,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "expected_ctx_cont_dim": ctx_cont_dim,
        "expected_ctx_cat_dim": ctx_cat_dim,
        "supports_context_features": True,
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "ctx_tag": "CTX6CAT6",
        "model_class": "EntryV10CtxHybridTransformer",
        "arch_id": "entry_v10_ctx_hybrid_transformer",
        "state_dict_sha256": state_dict_sha256,
        "anchored_entry_enabled": True,
        "anchor_source": "signal7_p_long_short_flat",
        "residual_scale": float(ENTRY_RESIDUAL_SCALE),
        "anchor_eps": float(ENTRY_ANCHOR_EPS),
        "train_recipe": {
            "direction_ce_scale": float(ENTRY_DIRECTION_CE_SCALE),
            "residual_scale": float(ENTRY_RESIDUAL_SCALE),
            "tradable_weight": float(ENTRY_AUX_TRADABLE_WEIGHT),
            "path_weight": float(ENTRY_AUX_PATH_WEIGHT),
            "mfe_weight": float(ENTRY_AUX_MFE_WEIGHT),
            "active_heads": [
                "direction",
                "tradable",
                "path_quality",
                "mfe_first_n",
            ],
        },
        "lane_contract": {
            "entry_admission_policy": "OVERLAP_LONG_REPLACES_OLDEST_OVERLAP_SHORT_WHEN_FULL",
            "max_open_trades": 10,
        },
        "parked_features": {
            "cost_sensitive_loss_enabled": bool(ENTRY_COST_SENSITIVE_ENABLED),
            "pred_balance_alpha": float(ENTRY_PRED_BALANCE_ALPHA),
            "residual_side_bias_alpha": float(ENTRY_RESIDUAL_SIDE_BIAS_ALPHA),
            "timing_loss_scale": float(ENTRY_TIMING_LOSS_SCALE),
            "aux_early_weight": float(ENTRY_AUX_EARLY_WEIGHT),
            "aux_quality_weight": float(ENTRY_AUX_QUALITY_WEIGHT),
            "aux_bad_path_weight": float(ENTRY_AUX_BAD_PATH_WEIGHT),
            "xgb_short_penalty_weight": float(XGB_SHORT_LONG_PENALTY),
        },
    }
    (out_bundle_dir / "bundle_metadata.json").write_text(json.dumps(meta, indent=2))

    # Post-export verify: strict load
    model2 = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_SIGNAL_DIM,
        snap_input_dim=SNAP_SIGNAL_DIM,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
    )
    model2.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model2.eval()
    with torch.no_grad():
        B = 2
        dummy_seq = torch.zeros(B, seq_len, 7)
        dummy_snap = torch.zeros(B, 7)
        dummy_cat = torch.zeros(B, ctx_cat_dim, dtype=torch.long)
        dummy_cont = torch.zeros(B, ctx_cont_dim)
        _ = model2(dummy_seq, dummy_snap, ctx_cat=dummy_cat, ctx_cont=dummy_cont)
    log.info(f"[DONE] Bundle OK strict load verified: {out_bundle_dir}")

    # Bundle load proof via runtime loader (strict)
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
    _ = load_entry_v10_ctx_bundle(
        bundle_dir=out_bundle_dir,
        device="cpu",
        xgb_models=None,
    )


def run_eval(
    bundle_dir: Path,
    train_parquet: Optional[Path],
    val_parquet: Optional[Path],
    test_parquet: Path,
    seq_len: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    gx1_data_override: str,
) -> None:
    """
    Deterministic eval of an existing bundle on a test parquet.
    No bundle mutation; writes EVAL_TEST.json alongside the bundle.
    """
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)
    set_thread_limits(1)

    bd = Path(bundle_dir).expanduser()
    if not bd.is_absolute():
        gx1_data = _resolve_gx1_data(gx1_data_override)
        bd = (gx1_data / bd).resolve()
    _require(bd.is_dir(), f"[ENTRY_V10_CTX_BUNDLE_DIR_MISSING] {bd}")

    model_path = bd / "model_state_dict.pt"
    meta_path = bd / "bundle_metadata.json"
    _require(model_path.exists(), f"[ENTRY_V10_CTX_MODEL_MISSING] {model_path}")
    _require(meta_path.exists(), f"[ENTRY_V10_CTX_META_MISSING] {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _require(meta.get("signal_bridge_id") == "XGB_SIGNAL_BRIDGE_V1", "[EVAL_CONTRACT_BRIDGE]")
    _require(meta.get("ctx_tag") == "CTX6CAT6", "[EVAL_CONTRACT_CTX_TAG]")
    ctx_cont_dim = int(meta.get("ctx_cont_dim") or -1)
    ctx_cat_dim = int(meta.get("ctx_cat_dim") or -1)
    _require(ctx_cont_dim >= 6, "[EVAL_CONTRACT_CTX_CONT]")
    _require(ctx_cat_dim == 6, "[EVAL_CONTRACT_CTX_CAT]")
    if meta.get("seq_input_dim") is not None:
        _require(int(meta.get("seq_input_dim")) == 7, "[EVAL_CONTRACT_SEQ_DIM]")
    if meta.get("snap_input_dim") is not None:
        _require(int(meta.get("snap_input_dim")) == 7, "[EVAL_CONTRACT_SNAP_DIM]")

    state_dict_sha = _sha256_file(model_path)

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_SIGNAL_DIM,
        snap_input_dim=SNAP_SIGNAL_DIM,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    dataset = EntryV10CtxDataset(
        parquet_path=test_parquet,
        seq_len=seq_len,
        allow_constant_labels=True,
    )
    _require(len(dataset) > 0, "[EVAL_NO_SAMPLES]")
    sample = dataset[0]
    _require(
        sample["seq_x"].shape[0] == seq_len,
        f"[EVAL_SEQ_LEN_MISMATCH] dataset seq_len {sample['seq_x'].shape[0]} != {seq_len}",
    )
    _require(
        sample["seq_x"].shape[1] == 7
        and sample["snap_x"].shape[0] == 7
        and sample["ctx_cont"].shape[0] == ctx_cont_dim
        and sample["ctx_cat"].shape[0] == ctx_cat_dim,
        "[EVAL_CONTRACT_MISMATCH] expected signal=7 ctx_cont=dynamic ctx_cat=6",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    meta_cost = meta.get("cost_matrix") or {}
    meta_cw = meta.get("class_weights") or {}
    cost_enabled = bool(meta.get("cost_sensitive_loss_enabled", ENTRY_COST_SENSITIVE_ENABLED))
    cost_scale = float(meta.get("cost_sensitive_loss_scale", ENTRY_COST_SENSITIVE_SCALE))
    _require_nonneg("EVAL_COST_SENSITIVE_SCALE", cost_scale)
    balance_alpha = float(meta.get("pred_balance_alpha", ENTRY_PRED_BALANCE_ALPHA))
    balance_target = str(meta.get("pred_balance_target", ENTRY_PRED_BALANCE_TARGET)).strip().lower()
    _require_nonneg("EVAL_PRED_BALANCE_ALPHA", balance_alpha)
    residual_side_bias_alpha = float(
        meta.get("residual_side_bias_alpha", ENTRY_RESIDUAL_SIDE_BIAS_ALPHA)
    )
    _require_nonneg("EVAL_RESIDUAL_SIDE_BIAS_ALPHA", residual_side_bias_alpha)
    if balance_target not in ("label", "uniform"):
        raise RuntimeError(
            f"[EVAL_BALANCE_TARGET_INVALID] pred_balance_target={balance_target!r} expected 'label' or 'uniform'"
        )

    cw_long = float(meta_cw.get("long", 1.0))
    cw_short = float(meta_cw.get("short", SHORT_CLASS_WEIGHT))
    cw_flat = float(meta_cw.get("flat", 1.0))
    class_weights = torch.tensor([cw_long, cw_short, cw_flat], device=device)

    cost_long_to_short = float(meta_cost.get("long_to_short", ENTRY_COST_LONG_TO_SHORT))
    cost_long_to_flat = float(meta_cost.get("long_to_flat", ENTRY_COST_LONG_TO_FLAT))
    cost_short_to_long = float(meta_cost.get("short_to_long", ENTRY_COST_SHORT_TO_LONG))
    cost_short_to_flat = float(meta_cost.get("short_to_flat", ENTRY_COST_SHORT_TO_FLAT))
    cost_flat_to_long = float(meta_cost.get("flat_to_long", ENTRY_COST_FLAT_TO_LONG))
    cost_flat_to_short = float(meta_cost.get("flat_to_short", ENTRY_COST_FLAT_TO_SHORT))
    _require_nonneg("EVAL_COST_LONG_TO_SHORT", cost_long_to_short)
    _require_nonneg("EVAL_COST_LONG_TO_FLAT", cost_long_to_flat)
    _require_nonneg("EVAL_COST_SHORT_TO_LONG", cost_short_to_long)
    _require_nonneg("EVAL_COST_SHORT_TO_FLAT", cost_short_to_flat)
    _require_nonneg("EVAL_COST_FLAT_TO_LONG", cost_flat_to_long)
    _require_nonneg("EVAL_COST_FLAT_TO_SHORT", cost_flat_to_short)

    criterion, _ = _build_cost_sensitive_criterion(
        device=device,
        class_weights=class_weights,
        cost_long_to_short=cost_long_to_short,
        cost_long_to_flat=cost_long_to_flat,
        cost_short_to_long=cost_short_to_long,
        cost_short_to_flat=cost_short_to_flat,
        cost_flat_to_long=cost_flat_to_long,
        cost_flat_to_short=cost_flat_to_short,
        cost_scale=cost_scale,
        enabled=cost_enabled,
        balance_alpha=balance_alpha,
        balance_target=balance_target,
    )
    test_loss, test_auc, test_acc = _validate_eval(
        model,
        loader,
        criterion,
        device,
        residual_side_bias_alpha=residual_side_bias_alpha,
    )

    eval_artifact = {
        "created_at_utc": _utc_now(),
        "bundle_dir": str(bd),
        "bundle_state_dict_sha256": state_dict_sha,
        "test_parquet": str(test_parquet),
        "test_parquet_sha256": _sha256_file(Path(test_parquet)),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "device": str(device),
        "seed": seed,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_auc_status": "DISABLED",
        "test_acc": test_acc,
        "n_test_samples": len(dataset),
    }
    eval_path = bd / "EVAL_TEST.json"
    eval_path.write_text(json.dumps(eval_artifact, indent=2), encoding="utf-8")
    auc_display = "DISABLED" if not np.isfinite(test_auc) else f"{test_auc:.4f}"
    log.info(
        f"[EVAL_DONE] {eval_path} loss={test_loss:.6f} auc={auc_display} acc={test_acc:.4f}"
    )

    _run_entry_training_bias_audit(
        bundle_dir=bd,
        model=model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_parquet,
    )


def _mean_median(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _init_confusion_bucket() -> Dict[str, Dict[str, List[float]]]:
    return {
        "label_SHORT__pred_LONG": {"p_long": [], "p_short": [], "p_flat": []},
        "label_SHORT__pred_SHORT": {"p_long": [], "p_short": [], "p_flat": []},
        "label_LONG__pred_LONG": {"p_long": [], "p_short": [], "p_flat": []},
        "label_LONG__pred_SHORT": {"p_long": [], "p_short": [], "p_flat": []},
    }


def _finalize_confusion(conf: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    out = {}
    for key, vals in conf.items():
        out[key] = {
            "p_long": _mean_median(vals["p_long"]),
            "p_short": _mean_median(vals["p_short"]),
            "p_flat": _mean_median(vals["p_flat"]),
            "count": len(vals["p_long"]),
        }
    return out


def _compute_bias_stats(
    model: nn.Module,
    dataset: EntryV10CtxDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    session_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    conf_global = _init_confusion_bucket()
    conf_by_session: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    totals_by_session: Dict[str, int] = {}
    total_samples = 0
    total_label_long_short = 0
    label_short_total = 0
    label_short_pred_long = 0
    label_short_margin_ge_002 = 0
    label_short_margin_ge_005 = 0

    label_short_by_session: Dict[str, Dict[str, int]] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cont = batch["ctx_cont"].to(device)
            ctx_cat = batch["ctx_cat"].to(device)
            y = batch["y"].to(device)

            out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
            logits = out["direction_logits"]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            p_long = probs[:, 0].cpu().numpy()
            p_short = probs[:, 1].cpu().numpy()
            p_flat = probs[:, 2].cpu().numpy()
            labels = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            sessions = ctx_cat[:, 0].cpu().numpy()

            total_samples += len(labels)

            for i in range(len(labels)):
                label = int(labels[i])
                pred = int(preds_np[i])
                sess_id = int(sessions[i]) if sessions is not None else -1
                sess_name = session_map.get(sess_id, f"UNKNOWN_{sess_id}")

                totals_by_session[sess_name] = totals_by_session.get(sess_name, 0) + 1
                if sess_name not in conf_by_session:
                    conf_by_session[sess_name] = _init_confusion_bucket()
                    label_short_by_session[sess_name] = {
                        "total": 0,
                        "pred_long": 0,
                        "pred_short": 0,
                        "pred_flat": 0,
                        "margin_ge_002": 0,
                        "margin_ge_005": 0,
                    }

                if label in (0, 1):
                    total_label_long_short += 1
                    if label == 1:
                        label_short_total += 1
                        label_short_by_session[sess_name]["total"] += 1
                        if pred == 0:
                            label_short_pred_long += 1
                            label_short_by_session[sess_name]["pred_long"] += 1
                        if pred == 1:
                            label_short_by_session[sess_name]["pred_short"] += 1
                        if pred == 2:
                            label_short_by_session[sess_name]["pred_flat"] += 1
                        margin = float(p_long[i] - p_short[i])
                        if margin >= 0.02:
                            label_short_margin_ge_002 += 1
                            label_short_by_session[sess_name]["margin_ge_002"] += 1
                        if margin >= 0.05:
                            label_short_margin_ge_005 += 1
                            label_short_by_session[sess_name]["margin_ge_005"] += 1

                    if label == 1 and pred == 0:
                        key = "label_SHORT__pred_LONG"
                    elif label == 1 and pred == 1:
                        key = "label_SHORT__pred_SHORT"
                    elif label == 0 and pred == 0:
                        key = "label_LONG__pred_LONG"
                    elif label == 0 and pred == 1:
                        key = "label_LONG__pred_SHORT"
                    else:
                        key = None

                    if key:
                        conf_global[key]["p_long"].append(float(p_long[i]))
                        conf_global[key]["p_short"].append(float(p_short[i]))
                        conf_global[key]["p_flat"].append(float(p_flat[i]))
                        conf_by_session[sess_name][key]["p_long"].append(float(p_long[i]))
                        conf_by_session[sess_name][key]["p_short"].append(float(p_short[i]))
                        conf_by_session[sess_name][key]["p_flat"].append(float(p_flat[i]))

    confusion_counts = {k: len(v["p_long"]) for k, v in conf_global.items()}
    confusion_rates = {
        k: (count / total_label_long_short if total_label_long_short > 0 else 0.0)
        for k, count in confusion_counts.items()
    }

    session_stats = {}
    for sess_name, conf in conf_by_session.items():
        session_label_short = label_short_by_session.get(sess_name, {})
        session_total = totals_by_session.get(sess_name, 0)
        session_conf_counts = {k: len(v["p_long"]) for k, v in conf.items()}
        session_conf_rates = {
            k: (count / sum(session_conf_counts.values()) if sum(session_conf_counts.values()) > 0 else 0.0)
            for k, count in session_conf_counts.items()
        }
        short_total = session_label_short.get("total", 0)
        pred_long = session_label_short.get("pred_long", 0)
        pred_short = session_label_short.get("pred_short", 0)
        pred_flat = session_label_short.get("pred_flat", 0)
        session_stats[sess_name] = {
            "total_samples": session_total,
            "confusion_counts": session_conf_counts,
            "confusion_rates": session_conf_rates,
            "prob_stats": _finalize_confusion(conf),
            "label_short": {
                "total": short_total,
                "pred_long_count": pred_long,
                "pred_short_count": pred_short,
                "pred_flat_count": pred_flat,
                "pred_long_rate": (pred_long / short_total if short_total > 0 else 0.0),
                "pred_short_rate": (pred_short / short_total if short_total > 0 else 0.0),
                "pred_flat_rate": (pred_flat / short_total if short_total > 0 else 0.0),
                "p_long_minus_p_short_ge_0.02_count": session_label_short.get("margin_ge_002", 0),
                "p_long_minus_p_short_ge_0.02_rate": (
                    session_label_short.get("margin_ge_002", 0) / short_total if short_total > 0 else 0.0
                ),
                "p_long_minus_p_short_ge_0.05_count": session_label_short.get("margin_ge_005", 0),
                "p_long_minus_p_short_ge_0.05_rate": (
                    session_label_short.get("margin_ge_005", 0) / short_total if short_total > 0 else 0.0
                ),
            },
        }

    return {
        "total_samples": total_samples,
        "label_long_short_total": total_label_long_short,
        "confusion_counts": confusion_counts,
        "confusion_rates": confusion_rates,
        "prob_stats": _finalize_confusion(conf_global),
        "label_short": {
            "total": label_short_total,
            "pred_long_count": label_short_pred_long,
            "pred_long_rate": (label_short_pred_long / label_short_total if label_short_total > 0 else 0.0),
            "p_long_minus_p_short_ge_0.02_count": label_short_margin_ge_002,
            "p_long_minus_p_short_ge_0.02_rate": (
                label_short_margin_ge_002 / label_short_total if label_short_total > 0 else 0.0
            ),
            "p_long_minus_p_short_ge_0.05_count": label_short_margin_ge_005,
            "p_long_minus_p_short_ge_0.05_rate": (
                label_short_margin_ge_005 / label_short_total if label_short_total > 0 else 0.0
            ),
        },
        "sessions": session_stats,
    }


def _run_entry_training_bias_audit(
    bundle_dir: Path,
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    train_parquet: Optional[Path],
    val_parquet: Optional[Path],
    test_parquet: Optional[Path],
) -> None:
    splits = {
        "train": train_parquet,
        "val": val_parquet,
        "test": test_parquet,
    }

    results = {
        "created_at_utc": _utc_now(),
        "bundle_dir": str(bundle_dir),
        "splits": {},
    }

    for split_name, parquet_path in splits.items():
        if parquet_path is None:
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] split=%s status=skip reason=missing_parquet", split_name)
            continue
        if not Path(parquet_path).exists():
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] split=%s status=skip reason=missing_file path=%s", split_name, parquet_path)
            continue

        dataset = EntryV10CtxDataset(
            parquet_path=Path(parquet_path),
            seq_len=seq_len,
            allow_constant_labels=True,
        )
        if len(dataset) == 0:
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] split=%s status=skip reason=empty_dataset", split_name)
            continue

        stats = _compute_bias_stats(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        results["splits"][split_name] = stats

    audit_path = Path(bundle_dir) / "ENTRY_TRAINING_BIAS_AUDIT.json"
    audit_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    session_bias = {"created_at_utc": _utc_now(), "bundle_dir": str(bundle_dir), "splits": {}}
    for split_name, split_stats in results.get("splits", {}).items():
        split_sessions = split_stats.get("sessions", {})
        session_bias["splits"][split_name] = {}
        for session_name in ["EU", "OVERLAP", "US", "US_LATE"]:
            sess_stats = split_sessions.get(session_name, {})
            label_short = sess_stats.get("label_short", {})
            session_bias["splits"][split_name][session_name] = {
                "label_short_total": label_short.get("total", 0),
                "pred_long_rate": label_short.get("pred_long_rate", 0.0),
                "pred_short_rate": label_short.get("pred_short_rate", 0.0),
                "pred_flat_rate": label_short.get("pred_flat_rate", 0.0),
                "margin_ge_0.02_rate": label_short.get("p_long_minus_p_short_ge_0.02_rate", 0.0),
                "margin_ge_0.05_rate": label_short.get("p_long_minus_p_short_ge_0.05_rate", 0.0),
            }

    session_bias_path = Path(bundle_dir) / "ENTRY_SESSION_BIAS_AUDIT.json"
    session_bias_path.write_text(json.dumps(session_bias, indent=2), encoding="utf-8")

    log.info("[ENTRY_TRAINING_BIAS_AUDIT]")
    log.info("bundle_dir=%s", bundle_dir)
    log.info("splits=%s", json.dumps(list(results.get("splits", {}).keys())))
    for split_name, split_stats in results.get("splits", {}).items():
        short_stats = split_stats.get("label_short", {})
        log.info(
            "[ENTRY_TRAINING_BIAS_AUDIT] split=%s label_short_total=%s pred_long_rate=%.6f margin_ge_0.02_rate=%.6f margin_ge_0.05_rate=%.6f",
            split_name,
            short_stats.get("total"),
            float(short_stats.get("pred_long_rate") or 0.0),
            float(short_stats.get("p_long_minus_p_short_ge_0.02_rate") or 0.0),
            float(short_stats.get("p_long_minus_p_short_ge_0.05_rate") or 0.0),
        )

    log.info("[ENTRY_SESSION_BIAS_AUDIT]")
    log.info("bundle_dir=%s", bundle_dir)
    for split_name, split_sessions in session_bias.get("splits", {}).items():
        for session_name, metrics in split_sessions.items():
            log.info(
                "[ENTRY_SESSION_BIAS_AUDIT] split=%s session=%s label_short_total=%s pred_long_rate=%.6f pred_short_rate=%.6f pred_flat_rate=%.6f margin_ge_0.02_rate=%.6f margin_ge_0.05_rate=%.6f",
                split_name,
                session_name,
                metrics.get("label_short_total"),
                float(metrics.get("pred_long_rate") or 0.0),
                float(metrics.get("pred_short_rate") or 0.0),
                float(metrics.get("pred_flat_rate") or 0.0),
                float(metrics.get("margin_ge_0.02_rate") or 0.0),
                float(metrics.get("margin_ge_0.05_rate") or 0.0),
            )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    _enforce_canonical_train_env_contract()

    parser = argparse.ArgumentParser("ENTRY_V10_CTX canonical trainer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity", action="store_true", help="Run sanity check only and exit 0/1")
    mode.add_argument("--train", action="store_true", help="Run training")
    mode.add_argument("--eval", action="store_true", help="Run eval on test split")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs (used with early stopping)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--dataset_manifest", type=Path, default=None, help="Path to dataset .manifest.json (train parquet from output_data_path; val = same dir stem_val.parquet)")
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Directory with *_train.parquet and *_val.parquet")
    parser.add_argument("--dataset_train_parquet", type=Path, default=None, help="Optional: explicit train parquet path when dataset_dir has multiple pairs")
    parser.add_argument("--out_bundle_dir", type=Path, required=False, help="Output bundle directory (under GX1_DATA if relative for train/sanity)")
    parser.add_argument("--bundle_dir", type=Path, default=None, help="Existing bundle directory for eval mode")
    parser.add_argument("--test_parquet", type=Path, default=None, help="Explicit test parquet path (optional)")
    parser.add_argument("--gx1-data", type=str, default="")
    parser.add_argument("--allow-constant-labels", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster (non-deterministic) training: cudnn benchmark on, deterministic off",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    args = parser.parse_args()

    _guard_no_rl()

    device = _resolve_device(args.device)
    log.info(f"[CONFIG] seed={args.seed} device={device} deterministic={not args.fast}")
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        log.info(
            "[CUDA_PROOF] cuda_available=%s device=%s device_name=%s",
            True,
            device,
            name,
        )
    else:
        log.info("[CUDA_PROOF] cuda_available=%s device=%s", False, device)

    if args.sanity:
        if args.out_bundle_dir is None:
            parser.error("--out_bundle_dir is required for --sanity")
        _log_manifest_proof(args.dataset_manifest)
        run_sanity_check(
            seq_len=args.seq_len,
            seed=args.seed,
            device=device,
            out_bundle_dir=args.out_bundle_dir,
            dataset_manifest=args.dataset_manifest,
            deterministic=not args.fast,
        )
        return

    if args.train:
        if args.out_bundle_dir is None:
            parser.error("--out_bundle_dir is required for --train")
        _log_manifest_proof(args.dataset_manifest)
        gx1_data = _resolve_gx1_data(args.gx1_data)
        train_parquet, val_parquet = _resolve_train_val_parquets(
            args.dataset_manifest,
            args.dataset_dir,
            gx1_data,
            train_parquet_hint=args.dataset_train_parquet,
        )
        try:
            test_parquet = _resolve_test_parquet(
                args.dataset_manifest,
                args.dataset_dir,
                args.test_parquet,
                gx1_data,
            )
            _log_label_distribution(test_parquet, split="test")
        except Exception as e:
            log.warning("[ENTRY_LABEL_DISTRIBUTION] split=test status=skip reason=%s", e)
        run_train(
            train_parquet=train_parquet,
            val_parquet=val_parquet,
            seq_len=args.seq_len,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            out_bundle_dir=args.out_bundle_dir,
            gx1_data_override=args.gx1_data,
            allow_constant_labels=args.allow_constant_labels,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            deterministic=not args.fast,
        )
        return

    if args.eval:
        _require(args.bundle_dir is not None, "[ENTRY_V10_CTX_EVAL_BUNDLE_REQUIRED]")
        _log_manifest_proof(args.dataset_manifest)
        gx1_data = _resolve_gx1_data(args.gx1_data)
        test_parquet = _resolve_test_parquet(
            args.dataset_manifest,
            args.dataset_dir,
            args.test_parquet,
            gx1_data,
            bundle_dir=args.bundle_dir,
        )
        # Resolve train/val from bundle metadata if available
        train_parquet = None
        val_parquet = None
        try:
            meta_path = Path(args.bundle_dir).expanduser() / "bundle_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("train_data"):
                    train_parquet = Path(meta["train_data"])
                if meta.get("val_data"):
                    val_parquet = Path(meta["val_data"])
        except Exception as e:
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] failed to resolve train/val from bundle metadata: %s", e)

        run_eval(
            bundle_dir=args.bundle_dir,
            train_parquet=train_parquet,
            val_parquet=val_parquet,
            test_parquet=test_parquet,
            seq_len=args.seq_len,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gx1_data_override=args.gx1_data,
        )
        return

if __name__ == "__main__":
    main()
