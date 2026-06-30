#!/usr/bin/env python3
"""
Build ENTRY_V10_CTX training dataset V2 — for the 2026-Q2 stack rebuild.

DELTAS vs v1 (build_entry_v10_ctx_training_dataset.py):
  α  ctx_cont 21 → 43 (added explicit H1/H4/D1/M15 indicators from canonical_v2)
  β  per-bar seq features 7 → 30 (XGB-bridge 7 + 23 price-state from canonical_v2)
  γ  seq_len 30 → 96 (8h M5, full session — symmetric with V3's M1L512)
  +  BIDIRECTIONAL: CANONICAL_PREMIUM_LONG_ONLY = False — emits both long+short labels
  +  XGB v3 default bundle (canonical_v2-trained, 87 features, 4-session-head bidir)

SSoT / ONE UNIVERSE:
- ctx contract: actual emitted CTX6CAT5/6 + extended ctx_cont contract (signal_bridge_v3)
- signal bridge: XGB_SIGNAL_BRIDGE_V3 plus optional sequence-structure extension
- Canonical v2 features parquet: /home/andre2/GX1_DATA/reports/.../CANONICAL_FEATURES_V2/canonical_features_v2.parquet
- Inputs must be canonical:
  - BASE28 prebuilt via CURRENT_MANIFEST.json (still used for ctx scaffold; v2 features joined on top)
  - canonical XGB bundle V3 (universal multihead v2 trained on canonical_v2; 87-feature contract)
  - canonical market tape lane (bid/ask) for deterministic label building

Outputs:
- time: tz-aware UTC timestamp
- seq: ndarray shaped [seq_len=96, 30]  (signal_bridge_v3 sequence)
- snap: ndarray shaped [30]             (signal_bridge_v3 snapshot)
- ctx_cont: ndarray shaped [len(ORDERED_CTX_CONT_NAMES_V3)]
- ctx_cat: ndarray shaped [len(ORDERED_CTX_CAT_NAMES_V3)]
- y_direction, y_early_move, y_quality_score, y_bad_path, y_*_long, y_*_short, ... (full label set)

NO FALLBACKS unless explicitly allowed by CLI flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import hashlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
from gx1.features.swing_structure_v1 import compute_swing_structure_features
# V2: switch from signal_bridge_v1 (7 fields) to signal_bridge_v3 (30 per-bar fields).
# SIGNAL_FIELDS now refers to the FULL 30-field per-bar feature vector (XGB-bridge 7 + price-state 23).
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_SEQ_FIELDS_V3 as SIGNAL_FIELDS,
    ORDERED_BRIDGE_FIELDS_V3 as XGB_BRIDGE_FIELDS,
    PER_BAR_PRICE_STATE_FIELDS_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_CTX_CAT_NAMES_V3,
    SEQ_SIGNAL_DIM_V3,
    SNAP_SIGNAL_DIM_V3,
    CTX_CONT_DIM_V3,
    CTX_CAT_DIM_V3,
    DEFAULT_SEQ_LEN_V3,
    SIGNAL_BRIDGE_ID_V3,
    CONTRACT_SHA256_V3 as SIGNAL_CONTRACT_SHA256,
)
from gx1.time.session_detector import (
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_vectorized,
)
from gx1.utils.canonical_prebuilt_resolver import resolve_base28_canonical_from_manifest
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel, proba_to_signal_bridge_v1
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PATH_QUALITY_HORIZON_BARS = 10
BAD_PATH_HORIZON_BARS = PATH_QUALITY_HORIZON_BARS
BAD_PATH_MAE_THRESHOLD_BPS = 6.0
BAD_PATH_MFE_THRESHOLD_BPS = 4.0
STRICT_TRADABLE_MFE_MIN_BPS = 22.0   # V2 (was 28.0): looser → ~5% tradable for richer IQL candidate stream
STRICT_TRADABLE_MAE_MAX_BPS = 3.0    # V2 (was 2.0)
STRICT_TRADABLE_PATH_MIN_BPS = 11.0  # V2 (was 14.0)
STRICT_TRADABLE_PATH_LEAD_MIN_BPS = 7.0
STRICT_TRADABLE_MFE_LEAD_MIN_BPS = 6.0
# 2026-05-26 — entry direction/tradable label re-tuned (user: "89% flat uaktuelt").
# The directional label = (pnl_at_horizon >= V11_TRADABLE_PNL_MIN_BPS) over
# V11_DIRECTION_HORIZON_BARS. Lowered 30→15 bps + horizon 10→24 (2h) → ~60% flat
# (was ~89%) so the model has real directional signal to learn + calibrate on.
# 15 bps ≈ 11 bps net after ~2.25 bps spread — a solid edge, not noise. bad_path /
# path_quality keep their own (10-bar) horizon — only the direction label changed.
V11_TRADABLE_PNL_MIN_BPS = 15.0
V11_DIRECTION_HORIZON_BARS = 24
HARD_NEG_LONG_MIN_XGB_P_LONG = 0.30
HARD_NEG_LONG_MIN_MFE_BPS = 10.0
HARD_NEG_LONG_MIN_MAE_BPS = 6.0
HARD_NEG_LONG_MAX_PATH_BPS = 8.0
DEAD_LONG_MAX_MFE_BPS = 0.5
DEAD_LONG_MIN_MAE_BPS = 6.0
TEASER_LONG_MIN_MFE_BPS = 0.5
TEASER_LONG_MAX_MFE_BPS = 10.0
TEASER_LONG_MIN_MAE_BPS = 6.0
TEASER_LONG_MAX_PATH_BPS = 4.0
CLEAN_EDGE_LONG_MFE_MIN_BPS = 14.0
CLEAN_EDGE_LONG_MAE_MAX_BPS = 4.0
CLEAN_EDGE_LONG_PATH_MIN_BPS = 16.0
SURVIVAL_LONG_MFE_MIN_BPS = 8.0
SURVIVAL_LONG_MAE_MAX_BPS = 6.0
SURVIVAL_LONG_PATH_MIN_BPS = 8.0
CANONICAL_PREMIUM_LONG_ONLY = False  # V2: BIDIRECTIONAL — emit both long and short labels


def final_direction_label_horizon_bars() -> int:
    """Return the actual horizon used by the emitted final y_direction label."""
    return int(V11_DIRECTION_HORIZON_BARS)


def final_direction_label_horizon_array(n_rows: int) -> np.ndarray:
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    return np.full(int(n_rows), final_direction_label_horizon_bars(), dtype=np.int32)


def direction_label_contract() -> Dict[str, Any]:
    return {
        "direction_label_source": "v11_spread_aware_final_pnl_at_direction_horizon",
        "direction_label_horizon_bars": final_direction_label_horizon_bars(),
        "direction_tradable_pnl_min_bps": float(V11_TRADABLE_PNL_MIN_BPS),
    }

# Active teacher runs used by the builder.
ACTIVE_V12_STRONG_RUN_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ENTRY_WEEKLY_AUDIT_V12_STRONG_20250512_20260408")
ACTIVE_V12_NORMAL_RUN_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ENTRY_WEEKLY_AUDIT_V12_NORMAL_20250604_20260408")
ACTIVE_LONG_WINDOW_TEACHER_RUN_SPECS = [
    {
        "label": "v12_strong_weekly",
        "source_model": "V12",
        "run_root": ACTIVE_V12_STRONG_RUN_ROOT,
    },
    {
        "label": "v12_normal_weekly",
        "source_model": "V12",
        "run_root": ACTIVE_V12_NORMAL_RUN_ROOT,
    },
]
# Frozen legacy benchmark runs kept for comparison/provenance only.
LEGACY_BENCHMARK_V6_STRONG_RUN_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ENTRY_WEEKLY_AUDIT_V6_STRONG_20250512_20260408")
LEGACY_BENCHMARK_V6_NORMAL_RUN_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ENTRY_WEEKLY_AUDIT_V6_NORMAL_20250604_20260408")
LEGACY_LONG_WINDOW_TEACHER_RUN_SPECS = [
    {
        "label": "legacy_benchmark_strong_weekly_v6",
        "source_model": "legacy_benchmark_v6",
        "run_root": LEGACY_BENCHMARK_V6_STRONG_RUN_ROOT,
    },
    {
        "label": "legacy_benchmark_normal_weekly_v6",
        "source_model": "legacy_benchmark_v6",
        "run_root": LEGACY_BENCHMARK_V6_NORMAL_RUN_ROOT,
    },
]
# Combined set preserved for the existing builder path.
LONG_WINDOW_TEACHER_RUN_SPECS = [
    *ACTIVE_LONG_WINDOW_TEACHER_RUN_SPECS,
    *LEGACY_LONG_WINDOW_TEACHER_RUN_SPECS,
]
LONG_WINDOW_TEACHER_BAD_EXIT_REASONS = {"CATASTROPHIC_GUARD"}
LONG_WINDOW_TEACHER_MEANINGFUL_MFE_BPS = 8.0
LONG_WINDOW_TEACHER_COLLAPSE_MIN_MFE_BPS = 18.0
LONG_WINDOW_TEACHER_COLLAPSE_MAX_RETAIN_FRAC = 0.25
LONG_WINDOW_TEACHER_COLLAPSE_MAX_PNL_BPS = 5.0
LONG_WINDOW_TEACHER_AGED_ROT_MIN_BARS = 96
LONG_WINDOW_TEACHER_AGED_ROT_MIN_TIME_SINCE_MFE_BARS = 36
LONG_WINDOW_TEACHER_AGED_ROT_MIN_DD_FROM_MFE_BPS = 18.0
LONG_WINDOW_TEACHER_AGED_ROT_MAX_RETAIN_FRAC = 0.35
LONG_WINDOW_TEACHER_AGED_ROT_MAX_PNL_BPS = 8.0
LONG_WINDOW_TEACHER_PREMIUM_MIN_PNL_BPS = 20.0
LONG_WINDOW_TEACHER_PREMIUM_MIN_MFE_BPS = 25.0
LONG_WINDOW_TEACHER_PREMIUM_MIN_RETAIN_FRAC = 0.45
LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_PNL_BPS = 18.0
LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_MFE_BPS = 22.0
LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_RETAIN_FRAC = 0.45
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
SESSION_CTX_CONT_NAMES = [
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
]
EXPECTED_SESSION_TAGS = ("ASIA", "EU", "OVERLAP", "US", "UNKNOWN")
SWING_ATR_PERIOD = 14

# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------
def compute_session_histogram(
    df: pd.DataFrame,
    ts_col: str = "ts",
    session_col: str = "session_id",
    session_override: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Compute a robust session histogram for dataset/audit logging."""
    n_rows = int(len(df))
    out: Dict[str, Any] = {"n_rows": n_rows}

    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        out["n_ts_na"] = int(ts.isna().sum())
        out["ts_min"] = None if ts.dropna().empty else str(ts.dropna().min())
        out["ts_max"] = None if ts.dropna().empty else str(ts.dropna().max())
    else:
        ts = None
        out["n_ts_na"] = n_rows
        out["ts_min"] = None
        out["ts_max"] = None

    unexpected_values: List[str] = []

    def normalize_tag(value: Any) -> str:
        try:
            tag = str(value).upper()
        except Exception:
            return "UNKNOWN"
        if tag in {"ASIA", "EU", "OVERLAP", "US"}:
            return tag
        return "UNKNOWN"

    def infer_from_timestamp() -> pd.Series:
        if ts is None:
            return pd.Series(["UNKNOWN"] * n_rows, index=df.index)
        inferred = get_session_vectorized(ts)
        return inferred.reindex(df.index).fillna("UNKNOWN").astype(str).str.upper()

    if session_override is not None:
        tags = pd.Series(session_override, index=df.index).map(normalize_tag)
    elif session_col in df.columns:
        ser = df[session_col]
        if pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            tags = ser.map(normalize_tag)
            unexpected_values = [
                str(v)
                for v in ser.dropna().astype(str).unique().tolist()
                if str(v).upper() not in {"ASIA", "EU", "OVERLAP", "US"}
            ][:5]
        else:
            ser_num = pd.to_numeric(ser, errors="coerce")
            non_na = ser_num.dropna()
            numeric_values = non_na.to_numpy(dtype=float)
            looks_like_session_id = (
                len(numeric_values) > 0
                and np.isfinite(numeric_values).all()
                and np.isclose(numeric_values, np.round(numeric_values)).all()
                and set(np.round(numeric_values).astype(int).tolist()).issubset({0, 1, 2, 3})
            )
            if looks_like_session_id:
                mapping = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
                tags = ser_num.map(lambda v: mapping.get(int(v), "UNKNOWN") if pd.notna(v) else "UNKNOWN")
            else:
                unexpected_values = [str(v) for v in ser.dropna().unique().tolist()[:5]]
                tags = infer_from_timestamp()
    else:
        tags = infer_from_timestamp()

    counts = {tag: 0 for tag in EXPECTED_SESSION_TAGS}
    for key, value in tags.value_counts(dropna=False).to_dict().items():
        tag = str(key).upper()
        counts[tag if tag in counts else "UNKNOWN"] += int(value)

    out["counts"] = counts
    out["pct"] = {tag: (count / n_rows * 100.0) if n_rows > 0 else 0.0 for tag, count in counts.items()}
    if unexpected_values:
        out["unexpected_session_values_top5"] = unexpected_values
    out["counts_sum"] = int(sum(counts.values()))
    return out


def log_session_histogram(
    df: pd.DataFrame,
    label: str,
    ts_col: str = "ts",
    session_col: str = "session_id",
    session_override: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Log and return the session histogram used by dataset build audits."""
    histogram = compute_session_histogram(
        df=df,
        ts_col=ts_col,
        session_col=session_col,
        session_override=session_override,
    )
    log.info(
        "[SESSION_HIST] %s: n_rows=%s n_ts_na=%s ts_min=%s ts_max=%s counts=%s",
        label,
        histogram["n_rows"],
        histogram["n_ts_na"],
        histogram["ts_min"],
        histogram["ts_max"],
        histogram["counts"],
    )
    return histogram


def get_git_commit() -> str:
    """Get current git commit hash (best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _hard_gate_ctx6cat6() -> Dict[str, Any]:
    """Fail-fast: ensure the canonical ctx base contract is CTX6CAT6 (ctx_cont_base=6, ctx_cat=6)."""
    ctx = get_canonical_ctx_contract()
    tag = str(ctx.get("tag", ""))
    cont = int(ctx.get("ctx_cont_dim", -1))
    cat = int(ctx.get("ctx_cat_dim", -1))
    if tag != "CTX6CAT6" or cont != 6 or cat != 6:
        raise RuntimeError(
            f"CTX_CONTRACT_SPLIT_BRAIN: expected CTX6CAT6 base (ctx_cont_base=6, ctx_cat=6) "
            f"but got tag={tag} cont={cont} cat={cat}"
        )
    # Names must exist for stable column mapping
    if "ctx_cont_names" not in ctx or "ctx_cat_names" not in ctx:
        raise RuntimeError("CTX_CONTRACT_INVALID: missing ctx_cont_names/ctx_cat_names in contract")
    if len(ctx["ctx_cont_names"]) != 6 or len(ctx["ctx_cat_names"]) != 6:
        raise RuntimeError("CTX_CONTRACT_INVALID: ctx base names length must be 6/6")
    return ctx


def _ensure_inputs_exist(base28_manifest: Path, xgb_bundle: Path) -> None:
    if not base28_manifest.exists():
        raise RuntimeError(f"INPUT_MISSING: base28_manifest not found: {base28_manifest}")
    if not xgb_bundle.exists():
        raise RuntimeError(f"INPUT_MISSING: xgb_bundle not found: {xgb_bundle}")
    if base28_manifest.suffix.lower() != ".json":
        raise RuntimeError(f"INPUT_INVALID: base28_manifest must be a .json manifest file: {base28_manifest}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_feature_prefix(name: str) -> str:
    out = str(name)
    for prefix in ("chart.", "ctx_cont.", "snap.", "seq."):
        if out.startswith(prefix):
            return out[len(prefix):]
    return out


def _resolve_seq_structure_extension(
    *,
    parquet_path: Optional[Path],
    manifest_path: Optional[Path],
    allow_manifest_only_inline: bool = False,
) -> Tuple[Optional[Path], List[str], Dict[str, Any]]:
    if parquet_path is None and manifest_path is None:
        return None, [], {}

    manifest: Dict[str, Any] = {}
    features: List[str] = []
    if manifest_path is not None:
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.exists():
            raise RuntimeError(f"SEQ_STRUCTURE_MANIFEST_MISSING: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        features = [str(x) for x in manifest.get("selected_features", []) if str(x).strip()]
        if parquet_path is None:
            raw_parquet = str(manifest.get("parquet_path") or "").strip()
            if not raw_parquet and not (allow_manifest_only_inline and features):
                raise RuntimeError(f"SEQ_STRUCTURE_MANIFEST_NO_PARQUET: {manifest_path}")
            parquet_path = Path(raw_parquet).expanduser().resolve() if raw_parquet else None

    if parquet_path is None and not (allow_manifest_only_inline and features):
        raise RuntimeError("SEQ_STRUCTURE_EXTENSION_PARQUET_UNRESOLVED")

    actual_sha: Optional[str] = None
    if parquet_path is not None:
        parquet_path = Path(parquet_path).expanduser().resolve()
        if not parquet_path.exists():
            raise RuntimeError(f"SEQ_STRUCTURE_PARQUET_MISSING: {parquet_path}")

        expected_sha = str(manifest.get("parquet_sha256") or "").strip()
        actual_sha = _sha256_file(parquet_path)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                f"SEQ_STRUCTURE_PARQUET_SHA_MISMATCH: got={actual_sha} expected={expected_sha} path={parquet_path}"
            )

    meta = {
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "parquet_path": str(parquet_path) if parquet_path is not None else None,
        "parquet_sha256": actual_sha,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_selected_feature_count": len(features) if features else None,
        "manifest_only": bool(manifest.get("manifest_only")) if manifest else False,
        "foundation_structure_feature_version": manifest.get("foundation_structure_feature_version"),
        "foundation_structure_feature_count": manifest.get("foundation_structure_feature_count"),
        "foundation_structure_missing_feature_count": manifest.get("foundation_structure_missing_feature_count"),
        "foundation_structure_all_required_selected": manifest.get("foundation_structure_all_required_selected"),
        "manifest": manifest,
    }
    return parquet_path, features, meta


def _join_seq_structure_extension(
    merged3: pd.DataFrame,
    *,
    parquet_path: Path,
    requested_features: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    ext = pd.read_parquet(parquet_path)
    if "time" not in ext.columns:
        raise RuntimeError(f"SEQ_STRUCTURE_TIME_COL_MISSING: {parquet_path}")
    ext = _normalize_time_utc(ext.reset_index(drop=True), "time")
    ext = ext.drop_duplicates(subset=["time"], keep="last")

    if requested_features:
        features = list(requested_features)
    else:
        features = [c for c in ext.columns if c not in {"time", "session", "source_split"}]
    if not features:
        raise RuntimeError("SEQ_STRUCTURE_FEATURES_EMPTY")
    missing = [f for f in features if f not in ext.columns]
    if missing:
        raise RuntimeError(f"SEQ_STRUCTURE_FEATURES_MISSING_IN_PARQUET: {missing[:30]} total={len(missing)}")

    overlap = [f for f in features if f in merged3.columns]
    if overlap:
        raise RuntimeError(f"SEQ_STRUCTURE_FEATURE_NAME_COLLISION: {overlap[:30]} total={len(overlap)}")

    before = int(len(merged3))
    cols = ["time"] + features
    ext_small = ext[cols].copy()
    for f in features:
        ext_small[f] = pd.to_numeric(ext_small[f], errors="raise").astype(np.float32)
    joined = merged3.merge(ext_small, on="time", how="inner", validate="one_to_one")
    after = int(len(joined))
    if after != before:
        raise RuntimeError(
            f"SEQ_STRUCTURE_JOIN_ROW_MISMATCH: before={before} after={after} parquet={parquet_path}"
        )
    mat = joined[features].astype(np.float32).to_numpy()
    if not np.isfinite(mat).all():
        raise RuntimeError("SEQ_STRUCTURE_NONFINITE_VALUES")

    normalized_existing_seq_overlap = [
        f for f in features if _strip_feature_prefix(f) in set(SIGNAL_FIELDS)
    ]
    meta = {
        "features": list(features),
        "feature_count": int(len(features)),
        "rows_joined": after,
        "normalized_existing_seq_overlap": normalized_existing_seq_overlap,
        "mode": "parquet_join",
    }
    return joined, features, meta


def _build_inline_seq_structure_extension(
    merged3: pd.DataFrame,
    *,
    requested_features: List[str],
    ctx_cont_names: List[str],
    source_parquet: Optional[Path],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if not requested_features:
        raise RuntimeError("SEQ_STRUCTURE_INLINE_REQUESTED_FEATURES_EMPTY")

    from gx1.scripts.experiment_entry_chart_structure_ablation_v1 import (
        _build_candlestick_derived_layer,
        _build_chart_layer,
        _build_deep_interaction_layer,
        _build_price_derived_layer,
    )

    base_blocks: List[np.ndarray] = []
    base_names: List[str] = []
    for field in SIGNAL_FIELDS:
        if field not in merged3.columns:
            raise RuntimeError(f"SEQ_STRUCTURE_INLINE_MISSING_SIGNAL_FIELD: {field}")
        base_blocks.append(merged3[field].astype(np.float32).to_numpy().reshape(-1, 1))
        base_names.append(f"snap.{field}")
    for field in ctx_cont_names:
        if field not in merged3.columns:
            raise RuntimeError(f"SEQ_STRUCTURE_INLINE_MISSING_CTX_FIELD: {field}")
        base_blocks.append(merged3[field].astype(np.float32).to_numpy().reshape(-1, 1))
        base_names.append(f"ctx_cont.{field}")
    base_x = np.concatenate(base_blocks, axis=1).astype(np.float32, copy=False)

    chart_x, chart_names = _build_chart_layer(base_x, base_names)
    if source_parquet is not None:
        price_x, price_names = _build_price_derived_layer(merged3[["time"]].copy(), Path(source_parquet))
        if price_x.shape[1]:
            chart_x = (
                np.concatenate([chart_x, price_x], axis=1).astype(np.float32, copy=False)
                if chart_x.shape[1]
                else price_x
            )
            chart_names = list(chart_names) + list(price_names)
        candle_x, candle_names = _build_candlestick_derived_layer(merged3[["time"]].copy(), Path(source_parquet))
        if candle_x.shape[1]:
            chart_x = (
                np.concatenate([chart_x, candle_x], axis=1).astype(np.float32, copy=False)
                if chart_x.shape[1]
                else candle_x
            )
            chart_names = list(chart_names) + list(candle_names)
    chart_all_x = (
        np.concatenate([base_x, chart_x], axis=1).astype(np.float32, copy=False)
        if chart_x.shape[1]
        else base_x
    )
    chart_all_names = list(base_names) + list(chart_names)
    deep_x, deep_names = _build_deep_interaction_layer(chart_all_x, chart_all_names, merged3[["time"]].copy())

    all_pieces = [base_x]
    all_names = list(base_names)
    if chart_x.shape[1]:
        all_pieces.append(chart_x)
        all_names.extend(chart_names)
    if deep_x.shape[1]:
        all_pieces.append(deep_x)
        all_names.extend(deep_names)
    all_x = np.concatenate(all_pieces, axis=1).astype(np.float32, copy=False)
    index = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in requested_features if name not in index]
    if missing:
        raise RuntimeError(f"SEQ_STRUCTURE_INLINE_FEATURES_MISSING: {missing[:30]} total={len(missing)}")

    selected = [name for name in requested_features]
    selected_cols: List[np.ndarray] = []
    for name in selected:
        selected_cols.append(all_x[:, index[name]].astype(np.float32, copy=False))
    out = np.column_stack(selected_cols).astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise RuntimeError("SEQ_STRUCTURE_INLINE_NONFINITE_VALUES")

    meta = {
        "mode": "inline_from_merged3",
        "features": list(selected),
        "feature_count": int(len(selected)),
        "base_matrix_dim": int(base_x.shape[1]),
        "chart_generated_dim": int(chart_x.shape[1]),
        "deep_generated_dim": int(deep_x.shape[1]),
        "available_feature_count": int(len(all_names)),
        "source_parquet_for_price_features": str(source_parquet) if source_parquet is not None else None,
        "missing_generated_features": [],
    }
    return out, selected, meta


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _split_min_max_from_ts_series(ts: pd.Series) -> Dict[str, Optional[str]]:
    t = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    if t.empty:
        return {"ts_min": None, "ts_max": None}
    return {"ts_min": str(t.min()), "ts_max": str(t.max())}


def _detect_time_col(df: pd.DataFrame) -> str:
    if "time" in df.columns:
        return "time"
    if "ts" in df.columns:
        return "ts"
    # Sometimes parquet index is time
    if "index" in df.columns:
        return "index"
    raise RuntimeError(
        "TIME_COLUMN_MISSING: canonical builder requires tz-aware UTC time column (time or ts)."
    )


def _normalize_time_utc(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError("TIME_PARSE_FAIL: time column could not be parsed to tz-aware UTC")
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out[~out["time"].duplicated()].copy()
    if len(out) == 0:
        raise RuntimeError("EMPTY_AFTER_TIME_NORMALIZATION")
    return out


def _resolve_gx1_data_root() -> Path:
    gx1_data = os.environ.get("GX1_DATA", "").strip()
    if not gx1_data:
        raise RuntimeError("GX1_DATA not set; required to resolve canonical market tape lane")
    root = Path(gx1_data).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"GX1_DATA invalid dir: {root}")
    return root


def _resolve_single_existing(paths: List[Path], *, label: str) -> Path:
    existing = [p for p in paths if p.exists()]
    if len(existing) != 1:
        raise RuntimeError(
            f"[ENTRY_LONG_WINDOW_TEACHER_SOURCE_INVALID] label={label} expected exactly one existing path, got={existing}"
        )
    return existing[0]


def _resolve_teacher_journal(run_root: Path, *, label: str) -> Path:
    merged = sorted(run_root.glob("trade_journal_*_MERGED.parquet"))
    if merged:
        return _resolve_single_existing(merged, label=f"{label}:trade_journal_merged")
    chunk = sorted((run_root / "replay" / "chunk_0").glob("trade_journal_*.parquet"))
    return _resolve_single_existing(chunk, label=f"{label}:trade_journal_chunk")


def _require_completed_teacher_run(run_root: Path, *, label: str) -> None:
    if not (run_root / "RUN_COMPLETED.json").exists():
        raise RuntimeError(
            f"[ENTRY_LONG_WINDOW_TEACHER_INCOMPLETE] label={label} run_root={run_root} missing RUN_COMPLETED.json"
        )


def _load_long_window_teacher_long_m5_labels() -> pd.DataFrame:
    """V2 stub: returns empty teacher labels.

    The legacy v1 teacher mining required RUN_COMPLETED.json markers in
    long-window teacher runs (V12_STRONG/V12_NORMAL etc.). For the 2026-Q2
    BIDIRECTIONAL rebuild we don't depend on these long-window LONG-only
    teacher labels — the bidirectional reward design (with explicit
    y_bad_path / y_clean_edge per side from the path-quality computation)
    provides equivalent supervision symmetrically for long+short.

    Returns an empty DataFrame with the schema downstream code expects.
    Downstream merge uses how="left" so empty teacher = NaN columns =
    treated as zero by the .get(..., default_zeros) fallback.
    """
    return pd.DataFrame({
        "time": pd.Series(dtype="datetime64[ns, UTC]"),
        "teacher_bad_long": pd.Series(dtype=bool),
        "teacher_winner_long": pd.Series(dtype=bool),
    })


def _load_long_window_teacher_long_m5_labels_legacy() -> pd.DataFrame:
    """
    Legacy v1 implementation — preserved for reference. Mine accepted long
    bars as teacher supervision on the same M5 grid as the entry dataset.

    Teacher bad:
    - CATA exits
    - EOF-negative
    - meaningful-edge longs that still collapse back to non-positive realized outcome

    Teacher winner:
    - positive closes with healthy retained edge
    - EOF-positive only when retention is strong enough to look harvestable on a longer window
    """

    def _load_run(run_root: Path, *, label: str, source_model: str) -> pd.DataFrame:
        _require_completed_teacher_run(run_root, label=label)
        eval_path = _resolve_single_existing(
            list((run_root / "replay" / "chunk_0" / "logs").glob("eval_log_*.jsonl")),
            label=f"{label}:eval_log",
        )
        journal_path = _resolve_teacher_journal(run_root, label=label)
        eval_df = pd.read_json(eval_path, lines=True)
        required_eval = {"ts_utc", "decision", "trade_id"}
        missing_eval = sorted(required_eval - set(eval_df.columns))
        if missing_eval:
            raise RuntimeError(
                f"[ENTRY_LONG_WINDOW_TEACHER_EVAL_SCHEMA] label={label} missing_eval_cols={missing_eval}"
            )
        accepted = eval_df.loc[eval_df["decision"] == "LONG", ["ts_utc", "trade_id"]].copy()
        if accepted.empty:
            raise RuntimeError(f"[ENTRY_LONG_WINDOW_TEACHER_EMPTY] label={label} no LONG decisions in {eval_path}")
        accepted["time"] = pd.to_datetime(accepted["ts_utc"], utc=True, errors="coerce").dt.floor("5min")
        if accepted["time"].isna().any():
            raise RuntimeError(f"[ENTRY_LONG_WINDOW_TEACHER_TIME_PARSE] label={label} eval ts parse failed")

        journal_df = pd.read_parquet(journal_path)
        required_journal = {"trade_id", "exit_reason", "pnl_bps"}
        missing_journal = sorted(required_journal - set(journal_df.columns))
        if missing_journal:
            raise RuntimeError(
                f"[ENTRY_LONG_WINDOW_TEACHER_JOURNAL_SCHEMA] label={label} missing_journal_cols={missing_journal}"
            )
        merged = accepted.merge(
            journal_df[
                [
                    c
                    for c in [
                        "trade_id",
                        "exit_reason",
                        "pnl_bps",
                        "mfe_bps",
                        "mae_bps",
                        "tradable_prob",
                        "mfe_first_n_pred",
                        "path_quality_pred",
                        "bars_in_trade",
                        "bars_held_exit_state",
                        "first_meaningful_mfe_bar_index",
                        "peak_mfe_bar_index",
                        "time_since_mfe_bars_exit",
                        "dd_from_mfe_bps_exit",
                        "distance_from_peak_mfe_bps_exit",
                    ]
                    if c in journal_df.columns
                ]
            ],
            on="trade_id",
            how="left",
            validate="one_to_one",
        )
        missing_join = merged["exit_reason"].isna() | merged["pnl_bps"].isna()
        if bool(missing_join.any()):
            missing_count = int(missing_join.sum())
            log.warning(
                "[ENTRY_LONG_WINDOW_TEACHER_JOIN_PARTIAL] label=%s missing_journal_rows=%d action=drop_non_realized_accepts",
                label,
                missing_count,
            )
            merged = merged.loc[~missing_join].copy()
        if merged.empty:
            raise RuntimeError(f"[ENTRY_LONG_WINDOW_TEACHER_EMPTY_AFTER_JOIN] label={label}")
        pnl = merged["pnl_bps"].astype(float)
        mfe = merged.get("mfe_bps", pd.Series(np.zeros(len(merged), dtype=float))).astype(float)
        bars_in_trade = merged.get("bars_in_trade", pd.Series(np.zeros(len(merged), dtype=float))).astype(float)
        time_since_mfe_bars_exit = (
            merged.get("time_since_mfe_bars_exit", pd.Series(np.zeros(len(merged), dtype=float))).astype(float)
        )
        dd_from_mfe_bps_exit = (
            merged.get("dd_from_mfe_bps_exit", pd.Series(np.zeros(len(merged), dtype=float))).astype(float)
        )
        retained_frac = np.where(
            mfe > 1e-6,
            pnl / np.maximum(mfe, 1e-6),
            np.where(pnl > 0.0, 1.0, -1.0),
        )
        meaningful_edge = mfe >= float(LONG_WINDOW_TEACHER_MEANINGFUL_MFE_BPS)
        eof_positive = merged["exit_reason"].eq("REPLAY_EOF") & (pnl > 0.0)
        eof_negative = merged["exit_reason"].eq("REPLAY_EOF") & (pnl < 0.0)
        collapse_bad = (
            (mfe >= float(LONG_WINDOW_TEACHER_COLLAPSE_MIN_MFE_BPS))
            & (retained_frac <= float(LONG_WINDOW_TEACHER_COLLAPSE_MAX_RETAIN_FRAC))
            & (pnl <= float(LONG_WINDOW_TEACHER_COLLAPSE_MAX_PNL_BPS))
        )
        aged_rot_bad = (
            meaningful_edge
            & (bars_in_trade >= float(LONG_WINDOW_TEACHER_AGED_ROT_MIN_BARS))
            & (time_since_mfe_bars_exit >= float(LONG_WINDOW_TEACHER_AGED_ROT_MIN_TIME_SINCE_MFE_BARS))
            & (dd_from_mfe_bps_exit >= float(LONG_WINDOW_TEACHER_AGED_ROT_MIN_DD_FROM_MFE_BPS))
            & (
                (retained_frac <= float(LONG_WINDOW_TEACHER_AGED_ROT_MAX_RETAIN_FRAC))
                | (pnl <= float(LONG_WINDOW_TEACHER_AGED_ROT_MAX_PNL_BPS))
            )
        )
        premium_eof_positive = eof_positive & (
            (mfe >= float(LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_MFE_BPS))
            & (pnl >= float(LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_PNL_BPS))
            & (retained_frac >= float(LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_RETAIN_FRAC))
        )
        premium_closed_positive = (
            ~merged["exit_reason"].isin(LONG_WINDOW_TEACHER_BAD_EXIT_REASONS)
            & ~merged["exit_reason"].eq("REPLAY_EOF")
            & (pnl >= float(LONG_WINDOW_TEACHER_PREMIUM_MIN_PNL_BPS))
            & (mfe >= float(LONG_WINDOW_TEACHER_PREMIUM_MIN_MFE_BPS))
            & (retained_frac >= float(LONG_WINDOW_TEACHER_PREMIUM_MIN_RETAIN_FRAC))
        )
        merged["teacher_bad_long"] = (
            merged["exit_reason"].isin(LONG_WINDOW_TEACHER_BAD_EXIT_REASONS)
            | eof_negative
            | collapse_bad
            | aged_rot_bad
        )
        merged["teacher_winner_long"] = premium_closed_positive | premium_eof_positive
        merged["teacher_eof_positive_long"] = premium_eof_positive
        merged["teacher_negative_eof_long"] = eof_negative
        merged["teacher_collapse_long"] = collapse_bad
        merged["teacher_aged_rot_long"] = aged_rot_bad
        grouped = (
            merged.groupby("time", as_index=False)
            .agg(
                teacher_bad_long=("teacher_bad_long", "max"),
                teacher_winner_long=("teacher_winner_long", "max"),
                teacher_trade_count=("trade_id", "count"),
                teacher_mean_pnl_bps=("pnl_bps", "mean"),
                teacher_eof_positive_long=("teacher_eof_positive_long", "max"),
                teacher_negative_eof_long=("teacher_negative_eof_long", "max"),
                teacher_collapse_long=("teacher_collapse_long", "max"),
                teacher_aged_rot_long=("teacher_aged_rot_long", "max"),
            )
            .sort_values("time")
            .reset_index(drop=True)
        )
        overlap = grouped["teacher_bad_long"] & grouped["teacher_winner_long"]
        if bool(overlap.any()):
            conflict_count = int(overlap.sum())
            log.warning(
                "[ENTRY_LONG_WINDOW_TEACHER_CONFLICT_DROP] label=%s conflicting_good_bad_rows=%d action=drop_ambiguous_teacher_rows",
                label,
                conflict_count,
            )
            grouped = grouped.loc[~overlap].copy()
            if grouped.empty:
                raise RuntimeError(
                    f"[ENTRY_LONG_WINDOW_TEACHER_EMPTY_AFTER_CONFLICT_DROP] label={label}"
                )
        grouped["teacher_label"] = label
        grouped["teacher_source_model"] = source_model
        log.info(
            "[ENTRY_LONG_WINDOW_TEACHER_PROOF] label=%s source_model=%s run_root=%s eval_path=%s journal_path=%s long_rows=%d unique_m5=%d bad_m5=%d good_m5=%d eof_pos_m5=%d eof_neg_m5=%d collapse_m5=%d aged_rot_m5=%d",
            label,
            source_model,
            run_root,
            eval_path,
            journal_path,
            int(len(accepted)),
            int(grouped["time"].nunique()),
            int(grouped["teacher_bad_long"].sum()),
            int(grouped["teacher_winner_long"].sum()),
            int(grouped["teacher_eof_positive_long"].sum()),
            int(grouped["teacher_negative_eof_long"].sum()),
            int(grouped["teacher_collapse_long"].sum()),
            int(grouped["teacher_aged_rot_long"].sum()),
        )
        return grouped

    teacher_parts: List[pd.DataFrame] = []
    for spec in LONG_WINDOW_TEACHER_RUN_SPECS:
        teacher_parts.append(
            _load_run(
                Path(spec["run_root"]),
                label=str(spec["label"]),
                source_model=str(spec["source_model"]),
            )
        )
    teacher = pd.concat(teacher_parts, ignore_index=True)
    teacher = (
        teacher.groupby("time", as_index=False)
        .agg(
            teacher_bad_long=("teacher_bad_long", "max"),
            teacher_winner_long=("teacher_winner_long", "max"),
            teacher_trade_count=("teacher_trade_count", "sum"),
            teacher_mean_pnl_bps=("teacher_mean_pnl_bps", "mean"),
            teacher_eof_positive_long=("teacher_eof_positive_long", "max"),
            teacher_negative_eof_long=("teacher_negative_eof_long", "max"),
            teacher_collapse_long=("teacher_collapse_long", "max"),
            teacher_aged_rot_long=("teacher_aged_rot_long", "max"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    overlap = teacher["teacher_bad_long"] & teacher["teacher_winner_long"]
    if bool(overlap.any()):
        conflict_count = int(overlap.sum())
        log.warning(
            "[ENTRY_LONG_WINDOW_TEACHER_COMBINED_CONFLICT_DROP] conflicting_good_bad_rows=%d action=drop_ambiguous_teacher_rows",
            conflict_count,
        )
        teacher = teacher.loc[~overlap].copy()
        if teacher.empty:
            raise RuntimeError("[ENTRY_LONG_WINDOW_TEACHER_EMPTY_AFTER_COMBINED_CONFLICT_DROP]")
    return teacher


# -----------------------------------------------------------------------------
# Market tape loading (canonical lane)
# -----------------------------------------------------------------------------
def _load_canonical_tape(
    *,
    tape_root: Path,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Load canonical M5 tape for [t_min, t_max] from a partitioned parquet dataset:
      .../xauusd_m5_bid_ask__CANONICAL/year=YYYY/part-000.parquet

    We avoid depending on manifest schema here; we trust the canonical lane path and parquet partitioning.
    """
    tape_root = tape_root.expanduser().resolve()
    if not tape_root.exists():
        raise RuntimeError(f"TAPE_ROOT_MISSING: {tape_root}")
    if not tape_root.is_dir():
        raise RuntimeError(f"TAPE_ROOT_NOT_DIR: {tape_root}")

    # Pull only years that intersect range
    y0 = int(pd.Timestamp(t_min).year)
    y1 = int(pd.Timestamp(t_max).year)
    files: List[Path] = []
    for y in range(y0, y1 + 1):
        p = tape_root / f"year={y}"
        if p.exists() and p.is_dir():
            files.extend(sorted(p.glob("*.parquet")))
            files.extend(sorted(p.glob("part-*.parquet")))
    # If layout differs, fall back to recursive parquet scan (still deterministic)
    if not files:
        files = sorted(tape_root.rglob("*.parquet"))

    if not files:
        raise RuntimeError(f"TAPE_NO_FILES: no parquet files found under {tape_root}")

    # Read and filter
    df_list: List[pd.DataFrame] = []
    for fp in files:
        dfi = pd.read_parquet(fp, columns=list(set(["time"] + required_cols)))
        if "time" not in dfi.columns:
            # Some tape uses "ts"
            if "ts" in dfi.columns:
                dfi = dfi.rename(columns={"ts": "time"})
            else:
                raise RuntimeError(f"TAPE_TIME_MISSING: {fp}")
        dfi["time"] = pd.to_datetime(dfi["time"], utc=True, errors="coerce")
        dfi = dfi.dropna(subset=["time"])
        dfi = dfi[(dfi["time"] >= t_min) & (dfi["time"] <= t_max)]
        if len(dfi):
            df_list.append(dfi)

    if not df_list:
        raise RuntimeError("TAPE_EMPTY_IN_RANGE")

    tape = pd.concat(df_list, ignore_index=True)
    tape = tape.sort_values("time")
    tape = tape[~tape["time"].duplicated()].copy()

    missing = [c for c in required_cols if c not in tape.columns]
    if missing:
        raise RuntimeError(f"TAPE_REQUIRED_COLS_MISSING: {missing}")

    if tape["time"].dtype != "datetime64[ns, UTC]":
        # pandas sometimes shows tz-aware as dtype object, normalize again
        tape["time"] = pd.to_datetime(tape["time"], utc=True, errors="coerce")
        tape = tape.dropna(subset=["time"])

    if len(tape) == 0:
        raise RuntimeError("TAPE_EMPTY_AFTER_NORMALIZATION")

    return tape


# -----------------------------------------------------------------------------
# Labels (simple fixed-hold exit, consistent with the "close after N bars" sanity exit)
# -----------------------------------------------------------------------------
def _compute_labels_fixed_hold(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
    early_move_threshold_bps: float,
    flat_threshold_bps: float,
) -> pd.DataFrame:
    """
    Labels are computed from bid/ask close (or bid/ask) with a fixed hold:
    - y_direction: 0=LONG, 1=SHORT, 2=FLAT
      LONG  if long_edge_bps  >= +flat_threshold_bps and long_edge_bps >= short_edge_bps
      SHORT if short_edge_bps >= +flat_threshold_bps and short_edge_bps >  long_edge_bps
      FLAT  otherwise (no clear edge or abs return below flat threshold)
    - y_early_move: 1 if max favorable move within horizon >= threshold (direction-aware)
    - y_quality_score: pnl_bps clipped for chosen direction (direction-aware)
    """
    if horizon_bars < 1:
        raise RuntimeError("HORIZON_INVALID")

    # We need bid/ask "close" like columns. Common names:
    # - bid_close / ask_close, or bid / ask
    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"TAPE_BID_ASK_COLS_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("TAPE_TOO_SHORT_FOR_HORIZON")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]
    exit_bid = bid[horizon_bars:]
    exit_ask = ask[horizon_bars:]

    # pnl in bps (spread-aware)
    # LONG: buy at ask, sell at bid
    pnl_long_bps = (exit_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4
    # SHORT: sell at bid, buy at ask
    pnl_short_bps = (entry_bid - exit_ask) / np.clip(entry_bid, 1e-12, None) * 1e4

    # Early move (direction-aware):
    # LONG: max bid over horizon vs entry_ask
    # SHORT: min ask over horizon vs entry_bid
    max_fav_bid = np.empty(n - horizon_bars, dtype=np.float64)
    min_fav_ask = np.empty(n - horizon_bars, dtype=np.float64)
    for i in range(0, n - horizon_bars):
        bid_window = bid[i : i + horizon_bars + 1]
        ask_window = ask[i : i + horizon_bars + 1]
        max_fav_bid[i] = float(np.max(bid_window))
        min_fav_ask[i] = float(np.min(ask_window))
    mfe_long_bps = (max_fav_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4
    mfe_short_bps = (entry_bid - min_fav_ask) / np.clip(entry_bid, 1e-12, None) * 1e4

    if flat_threshold_bps < 0:
        raise RuntimeError("FLAT_THRESHOLD_INVALID")
    y_direction = np.full_like(pnl_long_bps, 2, dtype=np.int32)
    long_edge = pnl_long_bps
    short_edge = pnl_short_bps
    thr = float(flat_threshold_bps)
    long_mask = (long_edge >= thr) & (long_edge >= short_edge)
    short_mask = (short_edge >= thr) & (short_edge > long_edge)
    y_direction[long_mask] = 0
    y_direction[short_mask] = 1
    thr_early = float(early_move_threshold_bps)
    y_early = np.zeros_like(pnl_long_bps, dtype=np.float32)
    y_early[long_mask] = (mfe_long_bps[long_mask] >= thr_early).astype(np.float32)
    y_early[short_mask] = (mfe_short_bps[short_mask] >= thr_early).astype(np.float32)

    y_quality = np.zeros_like(pnl_long_bps, dtype=np.float32)
    y_quality[long_mask] = np.clip(pnl_long_bps[long_mask], 0.0, 5000.0).astype(np.float32)
    y_quality[short_mask] = np.clip(pnl_short_bps[short_mask], 0.0, 5000.0).astype(np.float32)

    out = pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "y_direction": y_direction.astype(np.int32),
            "y_early_move": y_early,
            "y_quality_score": y_quality,
            "label_horizon_bars": np.int32(horizon_bars),
        }
    )
    return out


# -----------------------------------------------------------------------------
# Path quality (first N bars)
# -----------------------------------------------------------------------------
def _compute_path_quality_first_n(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
) -> pd.DataFrame:
    if horizon_bars < 1:
        raise RuntimeError("PATH_QUALITY_HORIZON_INVALID")

    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"PATH_QUALITY_BID_ASK_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("PATH_QUALITY_TAPE_TOO_SHORT")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]

    mfe_long = np.empty(n - horizon_bars, dtype=np.float64)
    mae_long = np.empty(n - horizon_bars, dtype=np.float64)
    mfe_short = np.empty(n - horizon_bars, dtype=np.float64)
    mae_short = np.empty(n - horizon_bars, dtype=np.float64)

    for i in range(0, n - horizon_bars):
        w_bid = bid[i : i + horizon_bars + 1]
        w_ask = ask[i : i + horizon_bars + 1]
        max_bid = float(np.max(w_bid))
        min_bid = float(np.min(w_bid))
        max_ask = float(np.max(w_ask))
        min_ask = float(np.min(w_ask))

        mfe_long[i] = (max_bid - entry_ask[i]) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        mae_long[i] = (entry_ask[i] - min_bid) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        mfe_short[i] = (entry_bid[i] - min_ask) / np.clip(entry_bid[i], 1e-12, None) * 1e4
        mae_short[i] = (max_ask - entry_bid[i]) / np.clip(entry_bid[i], 1e-12, None) * 1e4

    out = pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "mfe_long_first_n_bps": mfe_long,
            "mae_long_first_n_bps": mae_long,
            "mfe_short_first_n_bps": mfe_short,
            "mae_short_first_n_bps": mae_short,
            "path_quality_horizon_bars": np.int32(horizon_bars),
        }
    )
    return out


def _compute_bad_path_first_n(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
    adverse_threshold_bps: float,
    favorable_threshold_bps: float,
) -> pd.DataFrame:
    """V11 redesign: bad_path predicts ACTUAL LOSER, not trajectory shape.

    Old (V10): bad_path = MAE-threshold-hit BEFORE MFE-threshold-hit during trade.
              → predicts volatility, not outcome (anti-calibrated for trending markets).

    New (V11): bad_path = (final_PnL_at_horizon < 0).
              → predicts whether trade ends up a loser at the K-horizon.
              Direct outcome target, no path-shape confounder.
    """
    if horizon_bars < 1:
        raise RuntimeError("BAD_PATH_HORIZON_INVALID")

    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"BAD_PATH_BID_ASK_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("BAD_PATH_TAPE_TOO_SHORT")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]
    horizon_bid = bid[horizon_bars:]
    horizon_ask = ask[horizon_bars:]
    pnl_long_at_horizon_bps = (horizon_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4
    pnl_short_at_horizon_bps = (entry_bid - horizon_ask) / np.clip(entry_bid, 1e-12, None) * 1e4

    out_long = (pnl_long_at_horizon_bps < 0).astype(np.float32)
    out_short = (pnl_short_at_horizon_bps < 0).astype(np.float32)

    return pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "bad_path_long_first_n": out_long,
            "bad_path_short_first_n": out_short,
            "bad_path_horizon_bars": np.int32(horizon_bars),
            "bad_path_mae_threshold_bps": np.float32(adverse_threshold_bps),  # kept for compat
            "bad_path_mfe_threshold_bps": np.float32(favorable_threshold_bps),  # kept for compat
            "v11_pnl_long_at_horizon_bps": pnl_long_at_horizon_bps.astype(np.float32),
            "v11_pnl_short_at_horizon_bps": pnl_short_at_horizon_bps.astype(np.float32),
        }
    )


# -----------------------------------------------------------------------------
# Manifest writing
# -----------------------------------------------------------------------------
def write_manifest(
    *,
    output_path: Path,
    build_command: List[str],
    base28_manifest: Path,
    xgb_bundle: Path,
    tape_root: Optional[Path],
    splits: Optional[Dict[str, Any]] = None,
    ts_min_max_by_split: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ctx = _hard_gate_ctx6cat6()
    extra_ctx = (extra or {}).get("ctx_contract") or {}
    extra_signal = (extra or {}).get("signal_bridge") or {}
    ctx_cont_dim = int(extra_ctx.get("ctx_cont_dim") or ctx.get("ctx_cont_dim") or 6)
    # R4: ctx_cat is contract-driven (5/6) — default to the v3 contract len, never the v1 anchor's 6.
    ctx_cat_dim = int(extra_ctx.get("ctx_cat_dim") or len(ORDERED_CTX_CAT_NAMES_V3))
    ctx_cont_base_dim = int(extra_ctx.get("ctx_cont_base_dim") or ctx.get("ctx_cont_dim") or 6)
    ctx_cont_micro = list(extra_ctx.get("ctx_cont_micro_features") or [])
    ctx_cont_swing = list(extra_ctx.get("ctx_cont_swing_features") or [])
    ctx_tag = str(extra_ctx.get("tag") or f"CTX6CAT{ctx_cat_dim}")
    signal_bridge_id = str(extra_signal.get("signal_bridge_id") or extra_signal.get("id") or SIGNAL_BRIDGE_ID_V3)
    signal_bridge_sha = str(extra_signal.get("contract_sha256") or SIGNAL_CONTRACT_SHA256)
    signal_bridge_fields = list(extra_signal.get("fields") or SIGNAL_FIELDS)

    manifest: Dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "git_commit": get_git_commit(),
        "output_data_path": str(output_path),
        "build_command": build_command,
        "inputs": {
            "base28_manifest": str(base28_manifest),
            "xgb_bundle": str(xgb_bundle),
            "xgb_model_file": str((xgb_bundle / "xgb_universal_multihead_v2.joblib").resolve()),
            "xgb_model_sha256": extra.get("xgb_model_sha256") if extra else None,
            "neutral_xgb_bridge": bool((extra or {}).get("neutral_xgb_bridge", False)),
            "xgb_bridge_source": (
                "neutral_uniform_proba"
                if bool((extra or {}).get("neutral_xgb_bridge", False))
                else "xgb_bundle_inference"
            ),
            "tape_root": str(tape_root) if tape_root is not None else None,
        },
        "feature_contract": {
            "ctx_tag": ctx_tag,
            "ctx_cont_dim": int(ctx_cont_dim),
            "ctx_cat_dim": int(ctx_cat_dim),
            "ctx_cont_base_dim": int(ctx_cont_base_dim),
            "ctx_cont_micro_features": list(ctx_cont_micro),
            "ctx_cont_swing_features": list(ctx_cont_swing),
            "signal_bridge_id": signal_bridge_id,
            "signal_bridge_contract_sha256": signal_bridge_sha,
            "signal_bridge_fields": signal_bridge_fields,
        },
        "splits": splits,
        "ts_min_max_by_split": ts_min_max_by_split or {},
        "notes": notes,
    }
    if extra:
        manifest["extra"] = extra

    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    log.info(f"MANIFEST WRITTEN: {manifest_path}")
    return manifest_path


# -----------------------------------------------------------------------------
# Label proof
# -----------------------------------------------------------------------------
_SESSION_ID_TO_NAME = {
    0: "ASIA",
    1: "EU",
    2: "OVERLAP",
    3: "US",
}


def _log_label_distribution_proof(df: pd.DataFrame, split: str) -> None:
    if "y_direction" not in df.columns:
        log.warning("[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s status=no_y_direction", split)
        return
    y = df["y_direction"].astype(int)
    n = int(len(y))
    if n == 0:
        log.warning("[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s status=empty", split)
        return
    long_c = int((y == 0).sum())
    short_c = int((y == 1).sum())
    flat_c = int((y == 2).sum())
    log.info(
        "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s n=%d long=%d (%.4f) short=%d (%.4f) flat=%d (%.4f)",
        split,
        n,
        long_c,
        long_c / n,
        short_c,
        short_c / n,
        flat_c,
        flat_c / n,
    )
    log.info(
        "[ENTRY_FLAT_LABEL_PROOF] split=%s flat=%d flat_rate=%.4f status=%s",
        split,
        flat_c,
        flat_c / n,
        "OK" if flat_c > 0 else "EMPTY",
    )
    if "ctx_cat" not in df.columns:
        return
    try:
        sess_ids = df["ctx_cat"].apply(
            lambda v: int(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else None
        )
        df_s = pd.DataFrame({"y": y, "session_id": sess_ids}).dropna(subset=["session_id"])
        if df_s.empty:
            return
        for sid, grp in df_s.groupby("session_id"):
            sid_int = int(sid)
            n_s = int(len(grp))
            long_s = int((grp["y"] == 0).sum())
            short_s = int((grp["y"] == 1).sum())
            flat_s = int((grp["y"] == 2).sum())
            log.info(
                "[ENTRY_LABEL_BY_SESSION_PROOF] split=%s session=%s session_id=%d n=%d long=%d (%.4f) short=%d (%.4f) flat=%d (%.4f)",
                split,
                _SESSION_ID_TO_NAME.get(sid_int, "UNKNOWN"),
                sid_int,
                n_s,
                long_s,
                long_s / n_s,
                short_s,
                short_s / n_s,
                flat_s,
                flat_s / n_s,
            )
    except Exception:
        return


# -----------------------------------------------------------------------------
# Core builder
# -----------------------------------------------------------------------------
def build_dataset_canonical(
    *,
    base28_manifest_path: Path,
    xgb_bundle_path: Path,
    tape_root: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_rows: Optional[int],
    seq_len: int,
    horizon_bars: int,
    early_move_threshold_bps: float,
    flat_threshold_bps: float,
    allow_zero_ctx: bool,
    split_name: Optional[str] = None,
    canonical_v2_parquet: Optional[Path] = None,
    output_path: Optional[Path] = None,  # V2 streaming-write target
    streaming_batch_size: int = 5000,     # V2 batch rows per ParquetWriter flush
    source_parquet_override: Optional[Path] = None,
    xgb_feature_contract_path: Optional[Path] = None,
    xgb_sanitizer_config_path: Optional[Path] = None,
    seq_structure_features_parquet: Optional[Path] = None,
    seq_structure_manifest_path: Optional[Path] = None,
    seq_structure_compute_inline: bool = False,
    neutral_xgb_bridge: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ctx = _hard_gate_ctx6cat6()
    if source_parquet_override is None:
        _ensure_inputs_exist(base28_manifest_path, xgb_bundle_path)
    elif not xgb_bundle_path.exists():
        raise RuntimeError(f"INPUT_MISSING: xgb_bundle not found: {xgb_bundle_path}")

    if seq_len < 2:
        raise RuntimeError("SEQ_LEN_INVALID: must be >=2")
    if horizon_bars < 1:
        raise RuntimeError("HORIZON_INVALID: must be >=1")

    # 1) Resolve source: either BASE28 manifest or single-parquet override (BASE76 mode).
    if source_parquet_override is not None:
        parquet_path = Path(source_parquet_override).expanduser().resolve()
        if not parquet_path.exists():
            raise RuntimeError(f"SOURCE_PARQUET_OVERRIDE_MISSING: {parquet_path}")
        parquet_sha = _sha256_file(parquet_path)
        log.info("[SOURCE_PARQUET_OVERRIDE] %s sha256=%s", parquet_path, parquet_sha)
    else:
        manifest_info = resolve_base28_canonical_from_manifest(str(base28_manifest_path))
        parquet_path = Path(manifest_info["parquet_path"]).expanduser().resolve()
        parquet_sha = manifest_info["parquet_sha256"]
        if not parquet_path.exists():
            raise RuntimeError(f"BASE28_PARQUET_MISSING: {parquet_path}")

    # 2) Load parquet
    df = pd.read_parquet(parquet_path)
    df = df.reset_index(drop=False)
    time_col = _detect_time_col(df)
    df = _normalize_time_utc(df, time_col)

    # Canonical BASE28 may be expanded to raw M1 rows with an explicit model-bar marker.
    # Entry training remains defined on the M5 model-bar lane, so filter here instead of
    # letting the later tape join fail on a mixed-granularity input. Skip when override
    # source is already M5-cadence.
    if "is_model_bar" in df.columns and source_parquet_override is None:
        model_bar_mask = pd.Series(df["is_model_bar"]).fillna(False).astype(bool)
        rows_before_model_bar = int(len(df))
        rows_model_bar = int(model_bar_mask.sum())
        df = df.loc[model_bar_mask].copy()
        log.info(
            "[BASE28_MODEL_BAR_FILTER] rows_before=%d rows_after=%d dropped=%d",
            rows_before_model_bar,
            rows_model_bar,
            rows_before_model_bar - rows_model_bar,
        )

    # filter by start/end
    if start is not None:
        df = df[df["time"] >= start]
    if end is not None:
        df = df[df["time"] <= end]

    if len(df) == 0:
        raise RuntimeError("NO_ROWS_AFTER_FILTERS")

    # deterministic head
    if max_rows and len(df) > max_rows:
        df = df.head(int(max_rows)).copy()

    # 3) Enforce XGB feature contract order. Default: base80 (the ONE active
    # contract). Override via --xgb-feature-contract-path.
    if xgb_feature_contract_path is not None:
        contract_path = Path(xgb_feature_contract_path).expanduser().resolve()
    else:
        contract_path = project_root / "gx1" / "xgb" / "contracts" / "xgb_input_features_base80_v1.json"
    if not contract_path.exists():
        raise RuntimeError(f"XGB_FEATURE_CONTRACT_MISSING: {contract_path}")
    log.info("[XGB_FEATURE_CONTRACT] %s", contract_path)
    contract_obj = json.loads(contract_path.read_text(encoding="utf-8"))
    features = contract_obj.get("features") or contract_obj.get("ordered_features") or []
    if len(features) < 28:
        raise RuntimeError("FEATURE_CONTRACT_INVALID_LEN")

    # Derive session-context features from canonical UTC timestamps when absent or degenerate.
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if ts.isna().any():
        raise RuntimeError("TIME_PARSE_FAIL_FOR_SESSION_CONTEXT")
    session_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
    session_recompute = False
    if "session_id" not in df.columns:
        session_recompute = True
    else:
        # Degenerate if < 2 unique non-null values
        n_unique = int(pd.Series(df["session_id"]).dropna().nunique())
        if n_unique < 2:
            session_recompute = True
    if session_recompute:
        df["session_id"] = get_session_vectorized(ts).map(session_map).astype(np.int32)
        log.info("[SESSION_ID_RECOMPUTE] recomputed session_id from timestamp (reason=%s)", "missing" if "session_id" not in df.columns else "degenerate")
    # Hard-fail if still degenerate after recompute
    n_unique_post = int(pd.Series(df["session_id"]).dropna().nunique())
    if n_unique_post < 2:
        raise RuntimeError(
            f"SESSION_ID_DEGENERATE_AFTER_RECOMPUTE: unique={n_unique_post} "
            f"(expected >=2). Check prebuilt time/session pipeline."
        )
    # Log distribution proof
    _sess_counts = pd.Series(df["session_id"]).value_counts(dropna=False).sort_index()
    log.info("[SESSION_ID_DISTRIBUTION_PROOF] %s", _sess_counts.to_dict())
    if "is_ASIA" not in df.columns:
        df["is_ASIA"] = (df["session_id"].astype(int) == 0).astype(np.int8)
    if "minutes_since_session_open" not in df.columns:
        df["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(ts).astype(np.float32)
    if "minutes_to_next_session_boundary" not in df.columns:
        df["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    if "session_change_flag" not in df.columns:
        sid = df["session_id"].astype(np.int32)
        df["session_change_flag"] = (sid.diff().fillna(0) != 0).astype(np.int8)
    if "session_tradable" not in df.columns:
        df["session_tradable"] = (df["session_id"].astype(int) != 0).astype(np.int8)

    # BASE76 derived-features parity with XGB trainer (canonical_v3_FULL dropped these
    # as redundant; re-expose under historical names so BASE76 contract is satisfied).
    sid = df["session_id"].astype(np.int32)
    if "_v1_is_EU" not in df.columns:
        df["_v1_is_EU"] = (sid == 1).astype(np.int8)
    if "_v1_is_US" not in df.columns:
        df["_v1_is_US"] = (sid == 3).astype(np.int8)
    is_us_f = df["_v1_is_US"].to_numpy(dtype=np.float64)
    if "_v1_body_tr" not in df.columns and "_v1_body_share_1" in df.columns:
        df["_v1_body_tr"] = df["_v1_body_share_1"].to_numpy(dtype=np.float64)
    if "_v1_int_clv_atr" not in df.columns and "_v1_clv" in df.columns:
        df["_v1_int_clv_atr"] = df["_v1_clv"].to_numpy(dtype=np.float64)
    if "_v1_int_r5_atr" not in df.columns and "_v1_r5" in df.columns:
        df["_v1_int_r5_atr"] = df["_v1_r5"].to_numpy(dtype=np.float64)
    if "_v1_int_slope_h4_atr" not in df.columns and "_v1h4_slope5" in df.columns:
        df["_v1_int_slope_h4_atr"] = df["_v1h4_slope5"].to_numpy(dtype=np.float64)
    if "_v1_int_ema_us" not in df.columns and "_v1_ema_diff" in df.columns:
        df["_v1_int_ema_us"] = df["_v1_ema_diff"].to_numpy(dtype=np.float64) * is_us_f
    if "_v1_int_range_us" not in df.columns and "_v1_range_z" in df.columns:
        df["_v1_int_range_us"] = df["_v1_range_z"].to_numpy(dtype=np.float64) * is_us_f
    if "_v1_int_slope_h1_us" not in df.columns and "_v1h1_slope3" in df.columns:
        df["_v1_int_slope_h1_us"] = df["_v1h1_slope3"].to_numpy(dtype=np.float64) * is_us_f
    # Derive atr_bps if missing (needed by canonical_v3 contract; BASE76 doesn't require it).
    if "atr_bps" not in df.columns and "_v1_atr14" in df.columns and "close" in df.columns:
        df["atr_bps"] = (df["_v1_atr14"].astype(float) / df["close"].astype(float).clip(lower=1e-6) * 1e4).clip(0, 500).fillna(0.0)

    bridge_all = np.zeros((len(df), 7), dtype=np.float64)
    xgb_model_sha256: Optional[str] = None
    if neutral_xgb_bridge:
        neutral_proba = np.full((len(df), 3), 1.0 / 3.0, dtype=np.float64)
        bridge_all[:, :] = proba_to_signal_bridge_v1(neutral_proba)
        log.info(
            "[NEUTRAL_XGB_BRIDGE] enabled rows=%d bridge_vector=%s",
            len(df),
            bridge_all[0].tolist() if len(df) else [],
        )
    else:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise RuntimeError(f"SOURCE_FEATURES_MISSING: {missing}")

        df_features = df[features].copy()
        if len(df_features) == 0:
            raise RuntimeError("NO_ROWS_AFTER_FEATURE_SELECT")

        # 4) Load canonical XGB bundle + sanitizer. Sanitizer matches contract.
        model_path = Path(xgb_bundle_path) / "xgb_universal_multihead_v2.joblib"
        if xgb_sanitizer_config_path is not None:
            sanitizer_cfg = Path(xgb_sanitizer_config_path).expanduser().resolve()
        else:
            # X2 (2026-06-04 parity): default to the BASE80 sanitizer the bundle was TRAINED with (empty
            # bounds -> NO clipping, matching serve), NOT canonical_v3 (34 populated bounds -> clips 34
            # _v1_*/session XGB inputs the serve path never clips). The two sibling builders
            # (materialize_inference_batch_candidates_v3_v1.py, materialize_build_v3_training_dataset_v2.py)
            # already hard-default base80; this entry builder silently diverged -> different XGB probs/bridge.
            sanitizer_cfg = project_root / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base80_v1.json"
        if not model_path.exists():
            raise RuntimeError(f"XGB_MODEL_MISSING: {model_path}")
        if not sanitizer_cfg.exists():
            raise RuntimeError(f"SANITIZER_CONFIG_MISSING: {sanitizer_cfg}")
        # X2 fail-closed (one-truth): the sanitizer used here MUST be the exact one the bundle was trained
        # with — assert its SHA == the bundle meta's sanitizer_sha256, hard-fail on mismatch (would catch a
        # wrong --xgb-sanitizer-config-path or a future default drift statically).
        _xgb_meta_path = Path(xgb_bundle_path) / "xgb_universal_multihead_v2_meta.json"
        if _xgb_meta_path.exists():
            _expected_san_sha = (json.loads(_xgb_meta_path.read_text(encoding="utf-8")) or {}).get("sanitizer_sha256")
            if _expected_san_sha:
                _actual_san_sha = _sha256_file(sanitizer_cfg)
                if _actual_san_sha != _expected_san_sha:
                    raise RuntimeError(
                        f"XGB_SANITIZER_SHA_MISMATCH: {sanitizer_cfg.name} sha={_actual_san_sha} != bundle "
                        f"meta.sanitizer_sha256={_expected_san_sha} — train would clip differently than the "
                        f"model was trained with. Pass the matching --xgb-sanitizer-config-path."
                    )
        log.info("[XGB_SANITIZER_CONFIG] %s", sanitizer_cfg)

        xgb_model_sha256 = _sha256_file(model_path)
        log.info("[XGB_MODEL_SHA256] %s %s", model_path, xgb_model_sha256)

        model = XGBMultiheadModel.load(str(model_path))
        model_features = list(getattr(model, "feature_list", []) or [])
        if model_features != features:
            raise RuntimeError(
                "XGB_FEATURE_CONTRACT_MISMATCH: contract features != model.feature_list "
                f"(contract_len={len(features)} model_len={len(model_features)})"
            )
        sanitizer = XGBInputSanitizer.from_config(str(sanitizer_cfg))

        # sanitize (contract-ordered)
        x_array, _stats = sanitizer.sanitize(df_features, feature_list=features, allow_nan_fill=False)
        if x_array is None or len(df_features) != len(df):
            raise RuntimeError("SANITIZER_OUTPUT_INVALID")
        if np.isnan(x_array).any() or np.isinf(x_array).any():
            raise RuntimeError("SANITIZER_FAIL_NONFINITE: sanitized features contain NaN/Inf")
        df_features_sanitized = pd.DataFrame(
            x_array,
            columns=features,
            index=df_features.index,
        )

        # 5) Predict per session head. 2026-06-04 (audit R1): route ASIA(0)->ASIA head. The COSTFIX
        # XGB bundle HAS a real ASIA head (xgb_universal_multihead_v2_feature_importance_ASIA.csv), and
        # serve (v12_xgb_live.py:65), candidate-gen (materialize_inference_batch_candidates_v3_v1.py:172)
        # and the V3 builder (materialize_build_v3_training_dataset_v2.py:239) all route {0:ASIA}. The old
        # {0:OVERLAP} fallback (assumed no ASIA head) trained the V10 dataset's ASIA bars (37.7% of all
        # bars) on the WRONG head's bridge -> ~40% argmax disagreement vs serve. ONE TRUTH = ASIA->ASIA.
        # Missing session_id -> 0 (ASIA), mirroring serve's .get(s, "ASIA") default.
        session_series = df["session_id"].fillna(0).astype(int) if "session_id" in df.columns else None
        session_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}

        def _run_for_session(sess_name: str, idx: np.ndarray) -> None:
            if idx.size == 0:
                return
            probs = model.predict_proba(
                df_features_sanitized.iloc[idx],
                session=sess_name,
                feature_list=features,
            )
            # Expect attributes or dict-like; support both
            if hasattr(probs, "p_long"):
                pl = np.asarray(probs.p_long, dtype=np.float64)
                ps = np.asarray(probs.p_short, dtype=np.float64)
                pf = np.asarray(probs.p_flat, dtype=np.float64)
            else:
                pl = np.asarray(probs["p_long"], dtype=np.float64)
                ps = np.asarray(probs["p_short"], dtype=np.float64)
                pf = np.asarray(probs["p_flat"], dtype=np.float64)
            bridge_input = np.column_stack([pl, ps, pf])
            bridge = proba_to_signal_bridge_v1(bridge_input)
            if bridge.shape[1] != 7:
                raise RuntimeError(f"[BRIDGE_DIM_MISMATCH] expected bridge_dim=7, got shape={bridge.shape}")
            bridge_all[idx, :] = bridge

        if session_series is not None:
            for sid, name in session_map.items():
                mask = session_series.values == sid
                idx = np.where(mask)[0]
                _run_for_session(name, idx)
        else:
            _run_for_session("US", np.arange(len(df), dtype=np.int64))

    # Bridge proof log/meta
    log.info("[BRIDGE_PROOF] proba_dim=%d -> bridge_dim=%d rows=%d neutral=%s", 3, 7, bridge_all.shape[0], bool(neutral_xgb_bridge))

    # 6) Build ctx features
    ctx_cont_names = list(ctx["ctx_cont_names"])
    # R4 (2026-06-04): ctx_cat is contract-driven from signal_bridge_v3 (5 when GX1_REGIME_V4=1
    # — trend_regime_id dropped; 6 cement), NOT the stale v1 base anchor. This also fixes the
    # H4 case: the v3 contract uses capital "H4_trend_sign_cat" exactly as add_ctx_cont emits it
    # (the v1 base used lowercase "h4_trend_sign_cat", which mismatched the prebuilt column).
    # Symmetric with the ctx_cont_names = ORDERED_CTX_CONT_NAMES_V3 upgrade later in this builder.
    ctx_cat_names = list(ORDERED_CTX_CAT_NAMES_V3)

    # 7) Assemble per-bar signal dataframe (time aligned)
    # V2: bridge_all is shape (N, 7) — XGB-bridge fields only.
    # The remaining 23 price-state fields in SIGNAL_FIELDS come from canonical_v2 (joined later).
    df_sig = pd.DataFrame({"time": df["time"].to_numpy()})
    for i, field in enumerate(XGB_BRIDGE_FIELDS):
        df_sig[field] = bridge_all[:, i]

    # 8) Labels from canonical tape lane (join by time)
    t_min = pd.Timestamp(df_sig["time"].min()).tz_convert("UTC")
    t_max = pd.Timestamp(df_sig["time"].max()).tz_convert("UTC")

    # V10-AUX-05 (2026-06-03 cross-model scan): spread-aware relabel of the DIRECTIONAL
    # risk targets (y_dip_mae / y_dip_mfe / y_tail_mae). Mid-vs-mid excursions understate a
    # position's adverse move and overstate its favorable move by ~half a spread on each
    # leg (~0.7-1.8 bps risk-optimistic). When enabled we load bid/ask high/low and measure
    # the round-trip-honest excursion (long: enter ASK, mark BID; short: enter BID, mark
    # ASK). Default OFF = bit-identical to the cemented dataset; the regime-robust retrain
    # sets GX1_V10_SPREAD_AWARE_RISK_TARGETS=1. forecast_ret / vol_fwd are direction-agnostic
    # and are NEVER touched. Timing fractions stay mid-defined (only magnitudes change).
    _spread_aware_risk = os.environ.get(
        "GX1_V10_SPREAD_AWARE_RISK_TARGETS", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    _risk_tape_cols = ["bid_close", "ask_close", "open", "high", "low", "close"]
    if _spread_aware_risk:
        _risk_tape_cols = _risk_tape_cols + ["bid_high", "bid_low", "ask_high", "ask_low"]
    tape = _load_canonical_tape(
        tape_root=tape_root,
        t_min=t_min,
        t_max=t_max,
        required_cols=_risk_tape_cols,
    )

    # Inner join tape to BASE28 by time.
    # Determinism requirement is "every BASE28 model-bar timestamp must resolve to exactly one
    # tape row". Tape is allowed to contain extra timestamps outside the BASE28 model-bar lane.
    merged = df_sig.merge(tape, on="time", how="inner", validate="one_to_one")
    rows_base28 = int(len(df_sig))
    rows_tape = int(len(tape))
    rows_joined = int(len(merged))
    full_base28_match = int(rows_base28 == rows_joined)
    log.info(
        "[ENTRY_TAPE_JOIN_PROOF] rows_base28=%d rows_tape=%d rows_joined=%d full_base28_match=%d",
        rows_base28,
        rows_tape,
        rows_joined,
        full_base28_match,
    )
    if not full_base28_match:
        raise RuntimeError(
            f"TAPE_JOIN_STRICT_FAIL: rows_base28={rows_base28} rows_tape={rows_tape} rows_joined={rows_joined}"
        )

    # Micro-structure features (canonical tape OHLC)
    eps = 1e-9
    if not all(c in merged.columns for c in ("close", "high", "low")):
        raise RuntimeError("MICRO_FEATURES_MISSING: require close/high/low in canonical tape")
    tape_feat = merged[["time", "close", "high", "low"]].copy().sort_values("time")
    close = tape_feat["close"].astype(float)
    high = tape_feat["high"].astype(float)
    low = tape_feat["low"].astype(float)

    tape_feat["micro_momentum_3"] = close - close.shift(3)
    tape_feat["micro_momentum_5"] = close - close.shift(5)
    tape_feat["micro_acceleration"] = (close - close.shift(1)) - (close.shift(1) - close.shift(2))
    tape_feat["wick_ratio"] = (high - close) / (high - low + eps)
    ema_fast = close.ewm(span=5, adjust=False).mean()
    tape_feat["distance_ema_fast"] = close - ema_fast

    for name in MICRO_FEATURE_NAMES:
        if name not in tape_feat.columns:
            raise RuntimeError(f"MICRO_FEATURE_MISSING: {name}")

    # Swing-structure features (ATR-normalized) — ONE TRUTH: gx1.features.swing_structure_v1
    # (lookahead-safe confirmation lag). Shared with the live serve augmenter
    # (v12_ctx_augment_live._add_swing_features) so train == serve bit-for-bit; do NOT
    # re-implement the math here (2026-06-24 unification — was a 2nd, edge-divergent copy).
    _swing = compute_swing_structure_features(
        high.to_numpy(dtype=np.float64),
        low.to_numpy(dtype=np.float64),
        close.to_numpy(dtype=np.float64),
        lookback=2,
        atr_period=SWING_ATR_PERIOD,
    )
    for _name, _arr in _swing.items():
        tape_feat[_name] = _arr
    log.info(
        "[ENTRY_SWING_PIVOT_PROOF] swing_resets_high=%d swing_resets_low=%d",
        int((np.diff(_swing["bars_since_swing_high"]) < 0).sum()),
        int((np.diff(_swing["bars_since_swing_low"]) < 0).sum()),
    )

    for name in SWING_FEATURE_NAMES:
        if name not in tape_feat.columns:
            raise RuntimeError(f"SWING_FEATURE_MISSING: {name}")

    # Attach micro features to BASE28 rows (strict 1:1 time alignment)
    df = df.merge(
        tape_feat[["time"] + list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES)],
        on="time",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_tape"),
    )
    if len(df) != rows_base28:
        raise RuntimeError(
            f"MICRO_FEATURE_JOIN_FAIL: rows_base28={rows_base28} rows_after={len(df)}"
        )
    for name in list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES):
        tape_name = f"{name}_tape"
        if name not in df.columns and tape_name in df.columns:
            df[name] = df[tape_name]
            df.drop(columns=[tape_name], inplace=True)

    ctx_cont_names = (
        ctx_cont_names
        + list(MICRO_FEATURE_NAMES)
        + list(SWING_FEATURE_NAMES)
        + list(SESSION_CTX_CONT_NAMES)
    )
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
    log.info(
        "[ENTRY_SESSION_CTX_PROOF] names=%s count=%d",
        list(SESSION_CTX_CONT_NAMES),
        len(SESSION_CTX_CONT_NAMES),
    )

    for name in ctx_cont_names:
        if name not in df.columns:
            if allow_zero_ctx:
                df[name] = 0.0
            else:
                raise RuntimeError(f"CTX_CONT_MISSING_IN_BASE28: '{name}' not found (use --allow_zero_ctx to force zeros)")
    for name in ctx_cat_names:
        if name not in df.columns:
            if allow_zero_ctx:
                df[name] = 0
            else:
                raise RuntimeError(f"CTX_CAT_MISSING_IN_BASE28: '{name}' not found (use --allow_zero_ctx to force zeros)")

    # Normalize ctx dtypes
    df_ctx_cont = df[ctx_cont_names].astype(np.float32)
    df_ctx_cat = df[ctx_cat_names].astype(np.int64)

    # Compute labels on merged tape
    labels = _compute_labels_fixed_hold(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=horizon_bars,
        early_move_threshold_bps=early_move_threshold_bps,
        flat_threshold_bps=flat_threshold_bps,
    )
    path_quality = _compute_path_quality_first_n(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=PATH_QUALITY_HORIZON_BARS,
    )
    bad_path = _compute_bad_path_first_n(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=BAD_PATH_HORIZON_BARS,
        adverse_threshold_bps=BAD_PATH_MAE_THRESHOLD_BPS,
        favorable_threshold_bps=BAD_PATH_MFE_THRESHOLD_BPS,
    )

    # Align signals to labels (labels are shorter by horizon_bars)
    merged2 = merged.merge(
        labels[["time", "y_direction", "y_early_move", "y_quality_score", "label_horizon_bars"]],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    merged2 = merged2.merge(
        path_quality[
            [
                "time",
                "mfe_long_first_n_bps",
                "mae_long_first_n_bps",
                "mfe_short_first_n_bps",
                "mae_short_first_n_bps",
                "path_quality_horizon_bars",
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    merged2 = merged2.merge(
        bad_path[
            [
                "time",
                "bad_path_long_first_n",
                "bad_path_short_first_n",
                "bad_path_horizon_bars",
                "bad_path_mae_threshold_bps",
                "bad_path_mfe_threshold_bps",
                # NOTE: the H=10 v11_pnl_*_at_horizon_bps are NOT merged — the direction
                # label uses the dedicated H=24 v11_pnl_*_at_dir_horizon_bps below
                # (2026-05-26). bad_path head still uses its own bad_path_*_first_n.
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    # 2026-05-26: dedicated H=24 (2h) pnl-at-horizon for the DIRECTION/tradable label
    # (decoupled from bad_path's 10-bar horizon). Reuses the same spread-aware pnl fn.
    _dir_pnl = _compute_bad_path_first_n(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=V11_DIRECTION_HORIZON_BARS,
        adverse_threshold_bps=BAD_PATH_MAE_THRESHOLD_BPS,
        favorable_threshold_bps=BAD_PATH_MFE_THRESHOLD_BPS,
    )[["time", "v11_pnl_long_at_horizon_bps", "v11_pnl_short_at_horizon_bps"]].rename(
        columns={
            "v11_pnl_long_at_horizon_bps": "v11_pnl_long_at_dir_horizon_bps",
            "v11_pnl_short_at_horizon_bps": "v11_pnl_short_at_dir_horizon_bps",
        }
    )
    merged2 = merged2.merge(_dir_pnl, on="time", how="inner", validate="one_to_one")
    if len(merged2) == 0:
        raise RuntimeError("LABEL_JOIN_EMPTY")

    # Select MAE/MFE based on true direction (flat -> 0)
    y_dir = merged2["y_direction"].to_numpy()
    mae_first_n = np.where(
        y_dir == 0,
        merged2["mae_long_first_n_bps"].to_numpy(),
        np.where(y_dir == 1, merged2["mae_short_first_n_bps"].to_numpy(), 0.0),
    )
    mfe_first_n = np.where(
        y_dir == 0,
        merged2["mfe_long_first_n_bps"].to_numpy(),
        np.where(y_dir == 1, merged2["mfe_short_first_n_bps"].to_numpy(), 0.0),
    )
    merged2["mae_first_n_bps"] = mae_first_n.astype(np.float32)
    merged2["mfe_first_n_bps"] = mfe_first_n.astype(np.float32)
    merged2["path_quality_bps"] = (merged2["mfe_first_n_bps"] - merged2["mae_first_n_bps"]).astype(np.float32)
    bad_path_dir = np.where(
        y_dir == 0,
        merged2["bad_path_long_first_n"].to_numpy(),
        np.where(y_dir == 1, merged2["bad_path_short_first_n"].to_numpy(), 0.0),
    )
    merged2["y_bad_path"] = bad_path_dir.astype(np.float32)

    # Re-attach ctx to merged2 (align by time)
    df_ctx = pd.DataFrame({"time": df["time"].to_numpy()})
    for i, name in enumerate(ctx_cont_names):
        df_ctx[name] = df_ctx_cont.iloc[:, i].to_numpy()
    for i, name in enumerate(ctx_cat_names):
        df_ctx[name] = df_ctx_cat.iloc[:, i].to_numpy()

    merged3 = merged2.merge(df_ctx, on="time", how="inner", validate="one_to_one")
    if len(merged3) == 0:
        raise RuntimeError("CTX_JOIN_EMPTY")

    # 8b) V2: JOIN canonical_features_v2 to bring in 23 per-bar price-state features
    # (for the extended seq dim 7→30) and 22 new ctx_cont features (D1/M15/H1/H4) for
    # the extended ctx_cont dim 21→43.
    if canonical_v2_parquet is None:
        raise RuntimeError("V2_BUILDER_MISSING_CANONICAL_V2_PARQUET")
    if not Path(canonical_v2_parquet).exists():
        raise RuntimeError(f"CANONICAL_V2_PARQUET_NOT_FOUND: {canonical_v2_parquet}")

    log.info("[V2_CANONICAL_JOIN] loading canonical_v2 from %s", canonical_v2_parquet)
    # Volume features (vol_z_20 …) and GROUP-A market-parity features are DERIVED
    # below (volume from raw `volume`; parity via augment_candidate one-truth w/
    # IQL) — they do NOT exist in any source parquet, so exclude them from the
    # load requirement and instead pull raw `volume`.
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES as _VOLUME_FEAT_NAMES
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_GROUP_A_PARITY as _GROUP_A_PARITY,
        ORDERED_CTX_CONT_DIP_STRUCT as _DIP_STRUCT_PARITY,
        ORDERED_CTX_CONT_ENTRY_SMART_DERIVED as _ENTRY_SMART_DERIVED,
    )
    _computed_not_loaded = set(_VOLUME_FEAT_NAMES) | set(_GROUP_A_PARITY) | set(_DIP_STRUCT_PARITY) | set(_ENTRY_SMART_DERIVED)
    cv2_needed = list((set(
        list(PER_BAR_PRICE_STATE_FIELDS_V3)
        + list(ORDERED_CTX_CONT_NAMES_V3)
        + ["volume"]
    )) - _computed_not_loaded)
    cv2_already_have = set(merged3.columns)
    cv2_to_load = [c for c in cv2_needed if c not in cv2_already_have]
    # When source-parquet-override is used (canonical_v3_FULL_PLUS_CTX has 249 cols),
    # pull what's available from there first, then fall back to canonical_v2 for the rest.
    src_supplied: List[str] = []
    if source_parquet_override is not None:
        src_cols = set(df.columns)
        src_supplied = [c for c in cv2_to_load if c in src_cols]
        if src_supplied:
            log.info("[SOURCE_PARQUET_SUPPLIED] %d cols from source parquet (skipping cv2 for those)", len(src_supplied))
            src_extra = df[["time"] + src_supplied].drop_duplicates(subset=["time"])
            merged3 = merged3.merge(src_extra, on="time", how="inner", validate="one_to_one")
            cv2_to_load = [c for c in cv2_to_load if c not in src_supplied]
    log.info(
        "[V2_CANONICAL_JOIN] need=%d in_merged3=%d to_load_from_cv2=%d (src_supplied=%d)",
        len(cv2_needed), len(cv2_needed) - len(cv2_to_load) - len(src_supplied), len(cv2_to_load), len(src_supplied),
    )
    # Filter cv2_to_load to only what canonical_v2 actually has, hard-fail if any
    # truly missing across all sources (instead of obscure pyarrow error).
    import pyarrow.parquet as _pq_chk
    cv2_available = set(_pq_chk.read_schema(str(canonical_v2_parquet)).names) if cv2_to_load else set()
    cv2_truly_missing = [c for c in cv2_to_load if c not in cv2_available]
    if cv2_truly_missing:
        raise RuntimeError(
            f"V2_CANONICAL_FIELDS_MISSING_EVERYWHERE: {cv2_truly_missing} "
            f"(not in source_parquet, not in canonical_v2 schema)"
        )
    cv2_df = pd.read_parquet(canonical_v2_parquet, columns=["time"] + cv2_to_load)
    cv2_df["time"] = pd.to_datetime(cv2_df["time"], utc=True)
    log.info(
        "[V2_CANONICAL_JOIN] cv2 loaded: %d rows × %d cols (time + %d new)",
        len(cv2_df), len(cv2_df.columns), len(cv2_to_load),
    )
    rows_pre = len(merged3)
    merged3 = merged3.merge(cv2_df, on="time", how="inner", validate="one_to_one")
    rows_post = len(merged3)
    log.info(
        "[V2_CANONICAL_JOIN] merged: rows_pre=%d rows_post=%d lost=%d",
        rows_pre, rows_post, rows_pre - rows_post,
    )
    if rows_post == 0:
        raise RuntimeError("V2_CANONICAL_JOIN_EMPTY: no time overlap between merged3 and canonical_v2")

    # ── GROUP-A market-parity (24) — 2026-05-26 ──────────────────────────────
    # Compute the 24 ctx_cont parity features (dip-distance, pivots, vol-term,
    # vol-percentile, session-overlap) the SAME way the Entry/Exit-IQL data does
    # (augment_forward_outcome_v2.augment_candidate) → identical values = ONE
    # TRUTH (V10 sees exactly what the IQL acts on). atr is derived from the M5
    # multi-TF cache inside augment_candidate (matches IQL). Portfolio feats are
    # excluded (journal_dir nonexistent → 0; they're IQL-only state).
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_GROUP_A_PARITY, ORDERED_CTX_CONT_DIP_STRUCT,
    )
    # struct_smc_swing_x_dip_v3 is computed downstream (needs SMC×dip_proximity_m5);
    # the other 35 dip/struct come straight from augment_candidate._dip_struct_5tf.
    _DIP_STRUCT_FROM_AUG = [f for f in ORDERED_CTX_CONT_DIP_STRUCT if f != "struct_smc_swing_x_dip_v3"]
    _AUG_EXTRACT = list(ORDERED_CTX_CONT_GROUP_A_PARITY) + _DIP_STRUCT_FROM_AUG + ["dip_proximity_m5_v3"]
    if any(f not in merged3.columns for f in (list(ORDERED_CTX_CONT_GROUP_A_PARITY) + list(ORDERED_CTX_CONT_DIP_STRUCT))):
        from gx1.scripts.augment_forward_outcome_v2 import build_context as _ga_ctx, augment_candidate as _ga_aug
        from gx1.features.htf_features import load_multi_tf_v2_cache as _ga_load_cache
        _cache_dir = os.environ.get("GX1_V10_MULTI_TF_V2_CACHE_DIR",
            "/home/andre2/GX1_DATA/data/data/prebuilt/MULTI_TF_V2_CACHE")
        _m5 = merged3[["high", "low", "close"]].copy()
        _m5.index = pd.to_datetime(merged3["time"], utc=True)
        _m5 = _m5.sort_index()
        _ctx = _ga_ctx(_m5, _ga_load_cache(_cache_dir), journal_dir=Path("/nonexistent_v10_build_journal"))
        _cols = {k: np.zeros(len(merged3), dtype=np.float32) for k in _AUG_EXTRACT}
        for _i, _ts in enumerate(pd.DatetimeIndex(pd.to_datetime(merged3["time"], utc=True))):
            _feat = _ga_aug(_ctx, _ts)
            for _k in _AUG_EXTRACT:
                _cols[_k][_i] = float(_feat.get(_k, 0.0))
        for _k, _arr in _cols.items():
            if _k in ORDERED_CTX_CONT_GROUP_A_PARITY or _k in _DIP_STRUCT_FROM_AUG:
                merged3[_k] = _arr
        # struct_smc_swing_x_dip_v3 = norm(smc_swing_state) × dip_proximity_m5_v3
        # (mirrors augment_forward_outcome_v2.py:548 / IQL state builder — one truth).
        _sw_col = "smc_swing_state" if "smc_swing_state" in merged3.columns else None
        if _sw_col is not None:
            _sw = pd.to_numeric(merged3[_sw_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
            _max_abs = float(np.abs(_sw).max()) if len(_sw) else 1.0
            _sw_norm = np.clip(_sw / max(_max_abs, 1.0), -1.0, 1.0)
            merged3["struct_smc_swing_x_dip_v3"] = (_sw_norm * _cols["dip_proximity_m5_v3"]).astype(np.float32)
        else:
            merged3["struct_smc_swing_x_dip_v3"] = np.zeros(len(merged3), dtype=np.float32)
        log.info("[V10_GROUP_A_PARITY] computed %d parity + %d dip/struct ctx_cont features (one-truth w/ IQL)",
                 len(ORDERED_CTX_CONT_GROUP_A_PARITY), len(ORDERED_CTX_CONT_DIP_STRUCT))

    # ── Volume / order-flow per-bar features (2026-05-26) ────────────────────
    # Derived from raw `volume` (+ `close`) via the SAME helper the live ctx
    # augmenter calls (gx1.features.volume_features) → identical train/serve
    # values. Adds vol_z_20 / vol_ratio_5_20 / vol_pct_96 / signed_vol_z_20 to
    # the per-bar seq (PER_BAR_PRICE_STATE_FIELDS_V3 tail).
    if any(f not in merged3.columns for f in _VOLUME_FEAT_NAMES):
        from gx1.features.volume_features import add_volume_features as _add_vol
        if "volume" not in merged3.columns:
            raise RuntimeError("V10_VOLUME_FEATURES: raw `volume` column missing from merged3 — cannot derive")
        merged3 = merged3.sort_values("time").reset_index(drop=True)
        _add_vol(merged3)
        log.info("[V10_VOLUME_FEATURES] computed %d volume/order-flow seq features (one-truth w/ serving)",
                 len(_VOLUME_FEAT_NAMES))

    # ── Dip + forecast + timing + tail-risk + vol HEAD TARGETS (2026-05-26) ───
    # Dip-head (18): for {long,short} × K{12,48,96} we label mae_before_mfe (the
    # adverse drawdown BEFORE the favorable peak — same definition as the IQL's
    # take_now_*_mae_before_mfe) and mfe (recovery). Computed O(K)-vectorized on
    # M5 high/low (close-entry). Pinball loss turns dip_p50/p90 ← mae, recovery_p50
    # ← mfe. Forecast-head (4): cum future return @ K{1,5,12,24}. Self-supervised.
    #
    # NEW heads (2026-05-26, per user go-ahead):
    #   • dip-timing (6): y_dip_bottom_frac_{side}_K = bar-of-dip-bottom / K ∈[0,1]
    #     → "don't enter at the TOP of a dip" needs WHEN, not just how-deep.
    #   • time-to-MFE (6): y_time_to_mfe_frac_{side}_K = bar-of-favorable-peak / K
    #     → entry urgency (move imminent vs hours away).
    #   • tail-risk (6): y_tail_mae_{side}_K = worst adverse over FULL K horizon
    #     (regardless of mfe ordering) → stop placement / risk sizing. p90 via pinball.
    #   • vol-forecast (3): y_vol_fwd_K = forward realized vol (std of 1-bar ret, bps)
    #     over next K — direction-agnostic. → sizing + regime.
    # All come (almost) free from the same M5 high/low loop; vol from close returns.
    if "y_dip_mae_long_K12" not in merged3.columns:
        _close = merged3["close"].to_numpy(np.float64)
        _high = merged3["high"].to_numpy(np.float64)
        _low = merged3["low"].to_numpy(np.float64)
        _n = len(_close); _BPS = 1e4
        # V10-AUX-05: spread-aware price lanes for the directional risk targets (only used
        # when GX1_V10_SPREAD_AWARE_RISK_TARGETS=1). Fail-closed: assert the bid/ask high/low
        # carried through every join before we rely on them.
        if _spread_aware_risk:
            _need_sa = ("bid_close", "ask_close", "bid_high", "bid_low", "ask_high", "ask_low")
            _missing_sa = [c for c in _need_sa if c not in merged3.columns]
            if _missing_sa:
                raise RuntimeError(
                    f"GX1_V10_SPREAD_AWARE_RISK_TARGETS=1 but spread cols missing from "
                    f"merged3 (lost in a join?): {_missing_sa}"
                )
            _ask_close = merged3["ask_close"].to_numpy(np.float64)  # long entry (buy at ask)
            _bid_close = merged3["bid_close"].to_numpy(np.float64)  # short entry (sell at bid)
            _bid_high = merged3["bid_high"].to_numpy(np.float64)
            _bid_low = merged3["bid_low"].to_numpy(np.float64)
            _ask_high = merged3["ask_high"].to_numpy(np.float64)
            _ask_low = merged3["ask_low"].to_numpy(np.float64)
        for _K in (1, 5, 12, 24):  # forecast horizons (M5)
            _f = np.zeros(_n, dtype=np.float64)
            if _n > _K:
                _f[: _n - _K] = (_close[_K:] - _close[: _n - _K]) / _close[: _n - _K] * _BPS
            # clip gap/bad-print outliers (weekend gaps) so the forecast loss
            # isn't dominated by a few extreme bars; real M5 moves are well within.
            merged3[f"y_forecast_ret_K{_K}"] = np.clip(np.nan_to_num(_f), -1000.0, 1000.0).astype(np.float32)
        # forward realized-vol (std of 1-bar pct returns over next K), direction-agnostic
        _ret1 = pd.Series(_close).pct_change()
        for _K in (12, 48, 96):
            _fwd_std = (_ret1.rolling(_K).std().shift(-_K) * _BPS)
            merged3[f"y_vol_fwd_K{_K}"] = np.clip(
                np.nan_to_num(_fwd_std.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1000.0
            ).astype(np.float32)
        for _side in ("long", "short"):
            for _K in (12, 48, 96):  # dip horizons (M5)
                _mfe = np.full(_n, -1e18); _mae_before = np.zeros(_n); _run_adv = np.zeros(_n)
                _run_adv_bar = np.zeros(_n); _dip_bottom_bar = np.zeros(_n); _mfe_bar = np.zeros(_n)
                # V10-AUX-05: parallel spread-aware magnitude trackers. The MID trackers above
                # still drive peak/bottom detection (timing fractions stay mid-defined); these
                # only feed the three directional risk MAGNITUDES when the flag is ON.
                if _spread_aware_risk:
                    _mfe_sa = np.full(_n, -1e18); _mae_before_sa = np.zeros(_n); _run_adv_sa = np.zeros(_n)
                for _k in range(1, _K + 1):
                    _hi = np.full(_n, np.nan); _lo = np.full(_n, np.nan)
                    if _n > _k:
                        _hi[: _n - _k] = _high[_k:]; _lo[: _n - _k] = _low[_k:]
                    if _side == "long":
                        _fav = np.nan_to_num((_hi - _close) / _close * _BPS, nan=-1e18)
                        _adv = np.nan_to_num((_lo - _close) / _close * _BPS, nan=0.0)
                    else:  # short: profit when price falls
                        _fav = np.nan_to_num((_close - _lo) / _close * _BPS, nan=-1e18)
                        _adv = np.nan_to_num((_close - _hi) / _close * _BPS, nan=0.0)
                    if _spread_aware_risk:
                        _bhi = np.full(_n, np.nan); _blo = np.full(_n, np.nan)
                        _ahi = np.full(_n, np.nan); _alo = np.full(_n, np.nan)
                        if _n > _k:
                            _bhi[: _n - _k] = _bid_high[_k:]; _blo[: _n - _k] = _bid_low[_k:]
                            _ahi[: _n - _k] = _ask_high[_k:]; _alo[: _n - _k] = _ask_low[_k:]
                        if _side == "long":
                            # enter long buying at ASK; mark/exit selling at BID
                            _fav_sa = np.nan_to_num((_bhi - _ask_close) / _ask_close * _BPS, nan=-1e18)
                            _adv_sa = np.nan_to_num((_blo - _ask_close) / _ask_close * _BPS, nan=0.0)
                        else:
                            # enter short selling at BID; mark/exit buying back at ASK (profit as ASK falls)
                            _fav_sa = np.nan_to_num((_bid_close - _alo) / _bid_close * _BPS, nan=-1e18)
                            _adv_sa = np.nan_to_num((_bid_close - _ahi) / _bid_close * _BPS, nan=0.0)
                    _new_worst = _adv < _run_adv                     # strictly new worst adverse
                    _run_adv = np.minimum(_run_adv, _adv)            # worst adverse so far
                    _run_adv_bar = np.where(_new_worst, _k, _run_adv_bar)
                    _new_peak = _fav > _mfe
                    _mae_before = np.where(_new_peak, -_run_adv, _mae_before)      # drawdown BEFORE peak
                    _dip_bottom_bar = np.where(_new_peak, _run_adv_bar, _dip_bottom_bar)  # WHEN that dip bottomed
                    _mfe_bar = np.where(_new_peak, _k, _mfe_bar)      # WHEN the favorable peak hit
                    _mfe = np.maximum(_mfe, _fav)
                    if _spread_aware_risk:
                        _run_adv_sa = np.minimum(_run_adv_sa, _adv_sa)
                        # mae_before captured at the SAME (mid-defined) favorable peak bar
                        _mae_before_sa = np.where(_new_peak, -_run_adv_sa, _mae_before_sa)
                        _mfe_sa = np.maximum(_mfe_sa, _fav_sa)
                _mfe = np.where(_mfe < -1e17, 0.0, _mfe)
                if _spread_aware_risk:
                    _mae_out = _mae_before_sa
                    _mfe_out = np.where(_mfe_sa < -1e17, 0.0, _mfe_sa)
                    _tail_out = -_run_adv_sa
                else:
                    _mae_out = _mae_before
                    _mfe_out = _mfe
                    _tail_out = -_run_adv
                merged3[f"y_dip_mae_{_side}_K{_K}"] = np.clip(_mae_out, 0.0, 1000.0).astype(np.float32)
                merged3[f"y_dip_mfe_{_side}_K{_K}"] = np.clip(_mfe_out, 0.0, 1000.0).astype(np.float32)
                # timing (normalized to [0,1] by horizon) + full-horizon tail risk
                merged3[f"y_dip_bottom_frac_{_side}_K{_K}"] = np.clip(_dip_bottom_bar / float(_K), 0.0, 1.0).astype(np.float32)
                merged3[f"y_time_to_mfe_frac_{_side}_K{_K}"] = np.clip(_mfe_bar / float(_K), 0.0, 1.0).astype(np.float32)
                merged3[f"y_tail_mae_{_side}_K{_K}"] = np.clip(_tail_out, 0.0, 1000.0).astype(np.float32)
        log.info(
            "[V10_DIP_FORECAST_TARGETS] computed 12 dip + 4 forecast + 12 timing "
            "(dip-bottom+time-to-mfe ×dir×K) + 6 tail-risk + 3 vol-forecast targets "
            "(risk_targets=%s)",
            "SPREAD_AWARE_bid/ask" if _spread_aware_risk else "mid",
        )

    # ── HEAD-TARGET column manifest (ONE place — emitted into every output row) ─
    # These regression targets feed V10's aux heads (dip / forecast / timing /
    # tail-risk / vol). They are emitted into the per-row dict below so the
    # trainer's row.get(col, 0.0) actually finds them (silent-zero gap fixed
    # 2026-05-26). Order is irrelevant (read by name downstream).
    _HEAD_TARGET_COLS = (
        [f"y_dip_mae_{_d}_K{_K}" for _d in ("long", "short") for _K in (12, 48, 96)]
        + [f"y_dip_mfe_{_d}_K{_K}" for _d in ("long", "short") for _K in (12, 48, 96)]
        + [f"y_forecast_ret_K{_K}" for _K in (1, 5, 12, 24)]
        + [f"y_dip_bottom_frac_{_d}_K{_K}" for _d in ("long", "short") for _K in (12, 48, 96)]
        + [f"y_time_to_mfe_frac_{_d}_K{_K}" for _d in ("long", "short") for _K in (12, 48, 96)]
        + [f"y_tail_mae_{_d}_K{_K}" for _d in ("long", "short") for _K in (12, 48, 96)]
        + [f"y_vol_fwd_K{_K}" for _K in (12, 48, 96)]
    )
    _missing_head_tgt = [c for c in _HEAD_TARGET_COLS if c not in merged3.columns]
    if _missing_head_tgt:
        raise RuntimeError(f"V10_HEAD_TARGETS_MISSING before emit: {_missing_head_tgt}")
    _head_target_arrays = {c: merged3[c].astype(np.float32).to_numpy() for c in _HEAD_TARGET_COLS}
    log.info("[V10_HEAD_TARGETS] %d head-target columns staged for emission", len(_HEAD_TARGET_COLS))

    # ENTRY smart-context features promoted from audit-only candidates. These are
    # computed after SMC + Group-A + dip/struct source columns exist, before the
    # ctx_cont contract gate below.
    from gx1.features.entry_smart_context import add_entry_smart_context_features as _add_entry_smart
    _add_entry_smart(merged3, strict=True)
    log.info("[V10_ENTRY_SMART_CTX] computed %d promoted ctx_cont features", len(_ENTRY_SMART_DERIVED))

    # Verify all contracted signal and ctx_cont names are present after join.
    missing_sig = [f for f in SIGNAL_FIELDS if f not in merged3.columns]
    if missing_sig:
        raise RuntimeError(f"V2_SIGNAL_FIELDS_MISSING after canonical_v2 join: {missing_sig}")
    missing_ctx = [f for f in ORDERED_CTX_CONT_NAMES_V3 if f not in merged3.columns]
    if missing_ctx:
        raise RuntimeError(f"V2_CTX_CONT_FIELDS_MISSING after canonical_v2 join: {missing_ctx}")

    # 9) Build advanced structure: seq + snap + ctx arrays per sample
    # V2: sig_mat is now (N, 30) per signal_bridge_v3 (XGB-bridge 7 + price-state 23).
    # V2: ctx_cont_names is upgraded from 21-feature v1 list to 43-feature v2 list here,
    # since merged3 now contains all 22 v2 extension features after the canonical_v2 join.
    ctx_cont_names = list(ORDERED_CTX_CONT_NAMES_V3)
    log.info("[V2_CTX_CONT_UPGRADE] ctx_cont_names_v2 count=%d", len(ctx_cont_names))
    seq_structure_parquet, seq_structure_requested, seq_structure_meta = _resolve_seq_structure_extension(
        parquet_path=seq_structure_features_parquet,
        manifest_path=seq_structure_manifest_path,
        allow_manifest_only_inline=bool(seq_structure_compute_inline),
    )
    seq_structure_feature_names: List[str] = []
    if seq_structure_parquet is not None or (seq_structure_compute_inline and seq_structure_requested):
        if seq_structure_compute_inline:
            ext_sig_mat, seq_structure_feature_names, _seq_join_meta = _build_inline_seq_structure_extension(
                merged3,
                requested_features=seq_structure_requested,
                ctx_cont_names=ctx_cont_names,
                source_parquet=canonical_v2_parquet,
            )
        else:
            merged3, seq_structure_feature_names, _seq_join_meta = _join_seq_structure_extension(
                merged3,
                parquet_path=seq_structure_parquet,
                requested_features=seq_structure_requested,
            )
            ext_sig_mat = merged3[seq_structure_feature_names].astype(np.float32).to_numpy()
        seq_structure_meta.update(_seq_join_meta)
        log.info(
            "[SEQ_STRUCTURE_EXTENSION_V1] enabled features=%d rows=%d parquet=%s",
            len(seq_structure_feature_names),
            len(merged3),
            seq_structure_parquet,
        )

    base_sig_mat = merged3[list(SIGNAL_FIELDS)].astype(np.float32).to_numpy()
    if seq_structure_feature_names:
        sig_mat = np.concatenate([base_sig_mat, ext_sig_mat], axis=1).astype(np.float32, copy=False)
    else:
        sig_mat = base_sig_mat
    signal_fields_emitted = list(SIGNAL_FIELDS) + list(seq_structure_feature_names)
    snap_fields_emitted = list(signal_fields_emitted)
    times = merged3["time"].to_numpy()

    ctx_cont_mat = merged3[ctx_cont_names].astype(np.float32).to_numpy()
    ctx_cat_mat = merged3[ctx_cat_names].astype(np.int64).to_numpy()

    y_dir = merged3["y_direction"].astype(np.int32).to_numpy()
    # Snapshot of original labels before any relabel rules for RAW pocket proofs.
    y_dir_raw = y_dir.copy()
    y_early = merged3["y_early_move"].astype(np.float32).to_numpy()
    y_qual = merged3["y_quality_score"].astype(np.float32).to_numpy()
    y_mae_first_n = merged3["mae_first_n_bps"].astype(np.float32).to_numpy()
    y_mfe_first_n = merged3["mfe_first_n_bps"].astype(np.float32).to_numpy()
    y_path_quality = merged3["path_quality_bps"].astype(np.float32).to_numpy()
    y_bad_path = merged3["y_bad_path"].astype(np.float32).to_numpy()
    fixed_hold_bootstrap_horizon = merged3["label_horizon_bars"].astype(np.int32).to_numpy()
    if len(fixed_hold_bootstrap_horizon):
        _bootstrap_unique = sorted({int(v) for v in fixed_hold_bootstrap_horizon.tolist()})
        log.info(
            "[ENTRY_FIXED_HOLD_BOOTSTRAP_HORIZON] split=%s values=%s final_direction_horizon=%d",
            split_name or "full",
            _bootstrap_unique,
            final_direction_label_horizon_bars(),
        )
    # The early fixed-hold labels are only a bootstrap surface. The final emitted
    # y_direction is overwritten below by the V11 spread-aware H=24 pnl label, so
    # label_horizon_bars must describe that final direction target.
    y_label_horizon = final_direction_label_horizon_array(len(merged3))
    y_path_horizon = merged3["path_quality_horizon_bars"].astype(np.int32).to_numpy()

    # V10 v3+ TARGET 1: multi-TF trend-agreement score.
    # Computed from D1/H4/H1/M15/M5 sign signals; fraction of non-D1 TFs
    # whose sign matches D1's. Aux label for training (loss-weighting when
    # direction prediction is wrong under high TF-disagreement).
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 1.
    try:
        from gx1.features.tf_agreement_score import compute_tf_agreement_score
        y_tf_agreement = compute_tf_agreement_score(merged3).astype(np.float32).to_numpy()
        log.info(
            "[V3_TF_AGREEMENT] computed n=%d  mean=%.3f  std=%.3f  frac_full=%.3f  frac_zero=%.3f",
            len(y_tf_agreement), float(y_tf_agreement.mean()), float(y_tf_agreement.std()),
            float((y_tf_agreement == 1.0).mean()), float((y_tf_agreement == 0.0).mean()),
        )
    except Exception as exc:
        log.warning(f"[V3_TF_AGREEMENT] compute failed ({exc}) — filling with neutral 0.5")
        y_tf_agreement = np.full(len(merged3), 0.5, dtype=np.float32)

    # V10 v3+ TARGET 4: realized hold-horizon ∈ [0,1] (bars / 1440).
    # Join V10 candidates (entry-side) with Exit-IQL per_bar dataset to
    # get max(bars_in_trade_v1) per candidate = realized hold. Non-trade
    # samples (y_tradable=0) get neutral 0.5 (=~720 bars).
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 4.
    try:
        from pathlib import Path as _P
        per_bar_root = _P("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
                           "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_V3PLUS_FULL_20260519T012648Z_LOCK_V3TRACKED_20260519T022946Z_LOCK/per_week")
        if not per_bar_root.exists():
            raise RuntimeError(f"Exit-IQL per_bar root missing: {per_bar_root}")
        # Build a candidate_uid → max_bars_in_trade map across all weeks.
        # We only need the columns; this is a cheap pass.
        hold_map: dict = {}  # decision_ts_utc → max_bars_in_trade
        for wp in sorted(per_bar_root.glob("*.parquet")):
            try:
                _df_pb = pd.read_parquet(wp, columns=["decision_ts_utc", "bars_in_trade_v1"])
                _agg = _df_pb.groupby("decision_ts_utc")["bars_in_trade_v1"].max()
                for ts, max_bars in _agg.items():
                    hold_map[ts] = float(max_bars)
            except Exception:
                continue
        log.info(f"[V3_HOLD_HORIZON] candidates with realized hold: {len(hold_map):,}")
        # Map onto merged3 by time (entry timestamp). Per_bar dataset stores
        # decision_ts_utc as ISO string ("2026-04-13T00:00:00+00:00") so we
        # must match that format — pandas .astype(str) gives space-separator
        # which won't match.
        time_series = merged3["time"]
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series, utc=True)
        time_keys = time_series.apply(lambda t: t.isoformat() if pd.notna(t) else "")
        realized_hold = time_keys.map(hold_map).fillna(720.0)  # 720 = neutral 12h default
        y_hold_horizon = (realized_hold.astype(np.float32) / 1440.0).clip(0.0, 1.0).to_numpy()
        # Non-tradable rows get neutral 0.5
        try:
            non_tradable_mask = merged3["y_tradable"].astype(np.float32).to_numpy() <= 0.5
            y_hold_horizon[non_tradable_mask] = 0.5
        except KeyError:
            pass
        n_with_real = int((~np.isclose(y_hold_horizon, 0.5)).sum())
        log.info(
            "[V3_HOLD_HORIZON] computed n=%d (with realized=%d)  mean=%.3f  p50=%.3f",
            len(y_hold_horizon), n_with_real, float(y_hold_horizon.mean()),
            float(np.percentile(y_hold_horizon, 50)),
        )
    except Exception as exc:
        log.warning(f"[V3_HOLD_HORIZON] compute failed ({exc}) — filling with neutral 0.5")
        y_hold_horizon = np.full(len(merged3), 0.5, dtype=np.float32)

    # V10 v3+ TARGET 3: position-size target ∈ [0,1].
    # Proxy: realized signed edge in ATR units, sigmoid-mapped.
    # target = sigmoid((mfe + mae) / (atr * 2))
    #   +2 ATR winner → ~0.88  (model should learn "bigger size here")
    #   -2 ATR loser  → ~0.12  ("smaller size here")
    #     0 (flat)    → 0.5    (neutral default)
    # Non-tradable rows get neutral 0.5 so they don't pull head astray.
    # Live runner maps prediction to {0.25, 0.5, 1.0, 2.0}× units.
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 3.
    try:
        atr_col = "atr_bps" if "atr_bps" in merged3.columns else None
        if atr_col is None:
            raise RuntimeError("atr_bps column missing — cannot compute position-size target")
        atr_arr = np.maximum(merged3[atr_col].astype(np.float32).to_numpy(), 1e-3)
        signed_edge_atr = (y_mfe_first_n + y_mae_first_n) / (atr_arr * 2.0)
        y_position_size = (1.0 / (1.0 + np.exp(-signed_edge_atr))).astype(np.float32)
        # Non-tradable rows → neutral 0.5
        try:
            non_tradable_mask = merged3["y_tradable"].astype(np.float32).to_numpy() <= 0.5
            y_position_size[non_tradable_mask] = 0.5
        except KeyError:
            pass
        log.info(
            "[V3_POSITION_SIZE] computed n=%d  mean=%.3f  p10=%.3f  p50=%.3f  p90=%.3f",
            len(y_position_size), float(y_position_size.mean()),
            float(np.percentile(y_position_size, 10)),
            float(np.percentile(y_position_size, 50)),
            float(np.percentile(y_position_size, 90)),
        )
    except Exception as exc:
        log.warning(f"[V3_POSITION_SIZE] compute failed ({exc}) — filling with neutral 0.5")
        y_position_size = np.full(len(merged3), 0.5, dtype=np.float32)

    # ── Relabel rules REMOVED 2026-05-26 (one truth, smart-AI) ───────────────
    # The 4 hand-tuned FLAT-relabel patches (POISON_SHORT / GROUP_B /
    # OVERLAP_SHORT_TAIL / OVERLAP_SHORT_RESIDUAL), the proof-only matrix /
    # OVERLAP_LONG / US_SHORT blocks, AND the GX1_V10_RELABEL_RULES switch were
    # DELETED. They were magic-number crutches (H4/momentum/ATR thresholds) for
    # model failure modes the current model (multi-TF×5 + dip/struct inputs +
    # dip/timing/tail heads) is meant to LEARN. y_direction now follows ONLY the
    # outcome-based label; any residual pocket regression is fixed with a LEARNED
    # signal, never a hardcoded patch. (The strict-tradable labeling below is the
    # legitimate outcome-based label definition, NOT a pocket patch — kept.)
    # relabel_veto below is therefore always all-False (harmless no-op so the
    # downstream tradability propagation stays unchanged).

    # ---------------------------------------------------------------------------
    # Quality/tradability targets (post-relabel):
    # - We now intentionally collapse weak/ambiguous directional labels into FLAT.
    # - Main direction label follows the strict tradable side, not raw directional truth.
    # - This makes the dataset teach "only obvious edge" instead of "direction exists
    #   but maybe don't take it", which was too permissive for the current goal.
    # ---------------------------------------------------------------------------
    # Relabel veto = any rule that flipped y_dir from raw label (all relabels only force FLAT)
    relabel_veto = (y_dir != y_dir_raw)
    y_dir_directional = y_dir.copy()

    _mfe_long = merged3["mfe_long_first_n_bps"].astype(np.float32).to_numpy()
    _mae_long = merged3["mae_long_first_n_bps"].astype(np.float32).to_numpy()
    _mfe_short = merged3["mfe_short_first_n_bps"].astype(np.float32).to_numpy()
    _mae_short = merged3["mae_short_first_n_bps"].astype(np.float32).to_numpy()
    _bad_path_long = (merged3["bad_path_long_first_n"].astype(np.float32).to_numpy() > 0.5)
    # V2: also extract bad_path_short for symmetric BIDIR auxiliary labels.
    _bad_path_short = (merged3["bad_path_short_first_n"].astype(np.float32).to_numpy() > 0.5)
    _path_long = (_mfe_long - _mae_long).astype(np.float32)
    _path_short = (_mfe_short - _mae_short).astype(np.float32)
    _path_lead_long = (_path_long - _path_short).astype(np.float32)
    _path_lead_short = (_path_short - _path_long).astype(np.float32)
    _mfe_lead_long = (_mfe_long - _mfe_short).astype(np.float32)
    _mfe_lead_short = (_mfe_short - _mfe_long).astype(np.float32)

    # V11 redesign: tradable target is OUTCOME-based (final PnL ≥ threshold).
    # Old (V10): tradable required 6 trajectory conditions (mfe/mae/path/lead/...).
    #            Resulted in head_tradable being saturated (median pred ≈ 0.91)
    #            despite only ~5% of dataset being tradable. Anti-calibrated head.
    # New (V11): tradable = (final_pnl_at_horizon >= V11_TRADABLE_PNL_MIN_BPS).
    #            Direct outcome target, head learns to predict P(profitable trade).
    # 2026-05-26: threshold 30→15 bps + dedicated H=24 (2h) direction horizon
    # (V11_TRADABLE_PNL_MIN_BPS / V11_DIRECTION_HORIZON_BARS module consts) → ~60%
    # flat (was ~89%). Uses the v11_pnl_*_at_DIR_horizon_bps columns (H=24), NOT the
    # bad_path H=10 columns. (rule: 89% flat uaktuelt — give the model real signal.)
    _dir_long_col = "v11_pnl_long_at_dir_horizon_bps"
    _dir_short_col = "v11_pnl_short_at_dir_horizon_bps"
    if _dir_long_col not in merged3.columns or _dir_short_col not in merged3.columns:
        raise RuntimeError(f"V11_DIRECTION_PNL_MISSING: need {_dir_long_col}/{_dir_short_col} (H={V11_DIRECTION_HORIZON_BARS})")
    _pnl_long_at_h = merged3[_dir_long_col].astype(np.float32).to_numpy()
    _pnl_short_at_h = merged3[_dir_short_col].astype(np.float32).to_numpy()

    _tradable_long = (_pnl_long_at_h >= V11_TRADABLE_PNL_MIN_BPS)
    _tradable_short = (_pnl_short_at_h >= V11_TRADABLE_PNL_MIN_BPS)
    if CANONICAL_PREMIUM_LONG_ONLY:
        _tradable_short = np.zeros_like(_tradable_short, dtype=bool)

    # V11 side selection: pick whichever side ends MORE profitable.
    _side = np.full(len(merged3), -1, dtype=np.int8)  # -1 none, 0 long, 1 short
    _only_long = _tradable_long & ~_tradable_short
    _only_short = _tradable_short & ~_tradable_long
    _both = _tradable_long & _tradable_short
    _side[_only_long] = 0
    _side[_only_short] = 1
    if _both.any():
        _prefer_long = _pnl_long_at_h >= _pnl_short_at_h
        _prefer_short = _pnl_short_at_h > _pnl_long_at_h
        _side[_both & _prefer_long] = 0
        _side[_both & _prefer_short] = 1

    # Apply relabel veto to tradability as well: explicit poison pockets stay non-tradable
    _side[relabel_veto] = -1
    _direction_side = _side.copy()

    teacher_df = _load_long_window_teacher_long_m5_labels()
    teacher_join = pd.DataFrame({"time": pd.to_datetime(times, utc=True)}).merge(
        teacher_df,
        on="time",
        how="left",
    )
    _teacher_bad_long = (
        teacher_join.get("teacher_bad_long", pd.Series(np.zeros(len(teacher_join), dtype=bool)))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    _teacher_winner_long = (
        teacher_join.get("teacher_winner_long", pd.Series(np.zeros(len(teacher_join), dtype=bool)))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    _teacher_known_long = _teacher_bad_long | _teacher_winner_long
    if bool((_teacher_bad_long & _teacher_winner_long).any()):
        raise RuntimeError("[ENTRY_LONG_WINDOW_TEACHER_LABEL_CONFLICT] same row marked both good and bad")

    _harvest_side = _direction_side.copy()
    _harvest_side[_teacher_bad_long & (_direction_side == 0)] = -1
    _harvest_side[_teacher_winner_long] = 0

    # Hard-negative longs:
    # XGB points long, but forward path is too weak / too adverse to be premium.
    # These rows are the important "say no" examples for the anchored residual model.
    _p_long = merged3["p_long"].astype(np.float32).to_numpy()
    _p_short = merged3["p_short"].astype(np.float32).to_numpy()
    _p_flat = merged3["p_flat"].astype(np.float32).to_numpy()
    _xgb_long_candidate = (
        (_p_long >= float(HARD_NEG_LONG_MIN_XGB_P_LONG))
        & (_p_long >= _p_short)
        & (_p_long >= _p_flat)
    )
    _dead_negative_long = (
        _xgb_long_candidate
        & (_side != 0)
        & (_mfe_long <= float(DEAD_LONG_MAX_MFE_BPS))
        & (_mae_long >= float(DEAD_LONG_MIN_MAE_BPS))
    )
    _teaser_negative_long = (
        _xgb_long_candidate
        & (_side != 0)
        & (_mfe_long > float(TEASER_LONG_MIN_MFE_BPS))
        & (_mfe_long <= float(TEASER_LONG_MAX_MFE_BPS))
        & (
            _bad_path_long
            | (_mae_long >= float(TEASER_LONG_MIN_MAE_BPS))
            | (_path_long <= float(TEASER_LONG_MAX_PATH_BPS))
        )
    )
    _hard_negative_long = (
        _xgb_long_candidate
        & (_side != 0)
        & ~_dead_negative_long
        & ~_teaser_negative_long
        & (
            _bad_path_long
            | (
                (_mfe_long >= float(HARD_NEG_LONG_MIN_MFE_BPS))
                & (
                    (_mae_long >= float(HARD_NEG_LONG_MIN_MAE_BPS))
                    | (_path_long <= float(HARD_NEG_LONG_MAX_PATH_BPS))
                )
            )
        )
    )
    _hard_negative_long = _hard_negative_long | _teacher_bad_long
    y_dead_negative_long = _dead_negative_long.astype(np.float32)
    y_teaser_negative_long = _teaser_negative_long.astype(np.float32)
    y_hard_negative_long = _hard_negative_long.astype(np.float32)
    _clean_edge_intrinsic = (
        (_direction_side == 0)
        & (_mfe_long >= float(CLEAN_EDGE_LONG_MFE_MIN_BPS))
        & (_mae_long <= float(CLEAN_EDGE_LONG_MAE_MAX_BPS))
        & (_path_long >= float(CLEAN_EDGE_LONG_PATH_MIN_BPS))
        & (~_bad_path_long)
    )
    y_clean_edge_long = _clean_edge_intrinsic.astype(np.float32)
    y_clean_edge_long[_teacher_bad_long] = 0.0
    y_clean_edge_long[_teacher_winner_long] = 1.0
    _survival_intrinsic = (
        (_direction_side == 0)
        & (_mfe_long >= float(SURVIVAL_LONG_MFE_MIN_BPS))
        & (_mae_long <= float(SURVIVAL_LONG_MAE_MAX_BPS))
        & (_path_long >= float(SURVIVAL_LONG_PATH_MIN_BPS))
    )
    y_survival_long = _survival_intrinsic.astype(np.float32)
    y_survival_long[_teacher_bad_long] = 0.0
    y_survival_long[_teacher_winner_long] = 1.0
    y_teacher_bad_long = _teacher_bad_long.astype(np.float32)
    y_teacher_winner_long = _teacher_winner_long.astype(np.float32)
    y_selector_long_mask = (
        _xgb_long_candidate
        | (_direction_side == 0)
        | _teacher_known_long
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # V2: SHORT-side auxiliary labels (mirror of LONG-side for BIDIR symmetry)
    # ------------------------------------------------------------------------
    # Same threshold-based logic as long-side, applied to mfe_short / mae_short / path_short.
    # These feed V10's auxiliary heads (head_bad_path / head_clean_edge / head_survival)
    # symmetrically — without these, short-side rows would only have direction-supervision.
    _xgb_short_candidate = (
        (_p_short >= float(HARD_NEG_LONG_MIN_XGB_P_LONG))  # same threshold (interpret as side-confidence)
        & (_p_short >= _p_long)
        & (_p_short >= _p_flat)
    )
    _dead_negative_short = (
        _xgb_short_candidate
        & (_side != 1)
        & (_mfe_short <= float(DEAD_LONG_MAX_MFE_BPS))
        & (_mae_short >= float(DEAD_LONG_MIN_MAE_BPS))
    )
    _teaser_negative_short = (
        _xgb_short_candidate
        & (_side != 1)
        & (_mfe_short > float(TEASER_LONG_MIN_MFE_BPS))
        & (_mfe_short <= float(TEASER_LONG_MAX_MFE_BPS))
        & (
            _bad_path_short
            | (_mae_short >= float(TEASER_LONG_MIN_MAE_BPS))
            | (_path_short <= float(TEASER_LONG_MAX_PATH_BPS))
        )
    )
    _hard_negative_short = (
        _xgb_short_candidate
        & (_side != 1)
        & ~_dead_negative_short
        & ~_teaser_negative_short
        & (
            _bad_path_short
            | (
                (_mfe_short >= float(HARD_NEG_LONG_MIN_MFE_BPS))
                & (
                    (_mae_short >= float(HARD_NEG_LONG_MIN_MAE_BPS))
                    | (_path_short <= float(HARD_NEG_LONG_MAX_PATH_BPS))
                )
            )
        )
    )
    y_dead_negative_short = _dead_negative_short.astype(np.float32)
    y_teaser_negative_short = _teaser_negative_short.astype(np.float32)
    y_hard_negative_short = _hard_negative_short.astype(np.float32)
    _clean_edge_short_intrinsic = (
        (_direction_side == 1)
        & (_mfe_short >= float(CLEAN_EDGE_LONG_MFE_MIN_BPS))
        & (_mae_short <= float(CLEAN_EDGE_LONG_MAE_MAX_BPS))
        & (_path_short >= float(CLEAN_EDGE_LONG_PATH_MIN_BPS))
        & (~_bad_path_short)
    )
    y_clean_edge_short = _clean_edge_short_intrinsic.astype(np.float32)
    _survival_short_intrinsic = (
        (_direction_side == 1)
        & (_mfe_short >= float(SURVIVAL_LONG_MFE_MIN_BPS))
        & (_mae_short <= float(SURVIVAL_LONG_MAE_MAX_BPS))
        & (_path_short >= float(SURVIVAL_LONG_PATH_MIN_BPS))
    )
    y_survival_short = _survival_short_intrinsic.astype(np.float32)
    y_selector_short_mask = (
        _xgb_short_candidate
        | (_direction_side == 1)
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # V2: BIDIR-COMBINED labels for V10's single auxiliary heads
    # ------------------------------------------------------------------------
    # V10 has ONE head_clean_edge, ONE head_survival, ONE head_bad_path (not split per side).
    # Bidir supervision = OR across long-side and short-side label.
    y_clean_edge_bidir = np.maximum(y_clean_edge_long, y_clean_edge_short).astype(np.float32)
    y_survival_bidir = np.maximum(y_survival_long, y_survival_short).astype(np.float32)
    # y_bad_path is already direction-aware via merged3["y_bad_path"] (computed earlier from
    # bad_path_long when y_dir==0, bad_path_short when y_dir==1, else 0). No change needed.

    # Final tradability / quality targets
    y_tradable = (_harvest_side != -1).astype(np.int32)
    y_dir = np.full(len(merged3), 2, dtype=np.int32)
    y_dir[_direction_side == 0] = 0
    y_dir[_direction_side == 1] = 1

    # Quality auxiliaries align to the strict tradable side. Non-obvious labels are
    # intentionally parked to zero/FLAT.
    _quality_side = _harvest_side.copy()

    y_mfe_first_n = np.zeros_like(y_mfe_first_n)
    y_mae_first_n = np.zeros_like(y_mae_first_n)
    y_path_quality = np.zeros_like(y_path_quality)
    y_mfe_first_n[_quality_side == 0] = _mfe_long[_quality_side == 0]
    y_mfe_first_n[_quality_side == 1] = _mfe_short[_quality_side == 1]
    y_mae_first_n[_quality_side == 0] = _mae_long[_quality_side == 0]
    y_mae_first_n[_quality_side == 1] = _mae_short[_quality_side == 1]
    y_path_quality[_quality_side == 0] = _path_long[_quality_side == 0]
    y_path_quality[_quality_side == 1] = _path_short[_quality_side == 1]

    # Early move: align to the directional-quality side instead of tradability side.
    y_early = np.zeros_like(y_early)
    y_early[_quality_side != -1] = (
        y_mfe_first_n[_quality_side != -1] >= float(early_move_threshold_bps)
    ).astype(np.float32)

    # Quality score stays non-negative but now reflects directional path quality.
    y_qual = np.zeros_like(y_qual)
    y_qual[_quality_side != -1] = np.maximum(0.0, y_path_quality[_quality_side != -1]).astype(np.float32)

    _split_tag = split_name or "full"
    _directional_long_rate = float(np.mean(y_dir == 0)) if len(y_dir) else 0.0
    _directional_short_rate = float(np.mean(y_dir == 1)) if len(y_dir) else 0.0
    _directional_flat_rate = float(np.mean(y_dir == 2)) if len(y_dir) else 0.0
    log.info(
        "[ENTRY_DIRECTION_TARGET_SEMANTICS] split=%s source=post_relabel_directional "
        "long_rate=%.6f short_rate=%.6f flat_rate=%.6f relabel_veto_rate=%.6f",
        _split_tag,
        _directional_long_rate,
        _directional_short_rate,
        _directional_flat_rate,
        float(np.mean(relabel_veto)) if len(relabel_veto) else 0.0,
    )
    log.info(
        "[ENTRY_STRICT_TRADABLE_RULES] split=%s mfe_min=%.2f mae_max=%.2f path_min=%.2f path_lead_min=%.2f mfe_lead_min=%.2f",
        _split_tag,
        float(STRICT_TRADABLE_MFE_MIN_BPS),
        float(STRICT_TRADABLE_MAE_MAX_BPS),
        float(STRICT_TRADABLE_PATH_MIN_BPS),
        float(STRICT_TRADABLE_PATH_LEAD_MIN_BPS),
        float(STRICT_TRADABLE_MFE_LEAD_MIN_BPS),
    )
    log.info(
        "[ENTRY_DIRECTION_LANE_PROOF] split=%s canonical_premium_long_only=%s",
        _split_tag,
        bool(CANONICAL_PREMIUM_LONG_ONLY),
    )
    log.info(
        "[ENTRY_DEAD_LONG_RULES] split=%s mfe_max=%.2f mae_min=%.2f rate=%.6f",
        _split_tag,
        float(DEAD_LONG_MAX_MFE_BPS),
        float(DEAD_LONG_MIN_MAE_BPS),
        float(np.mean(y_dead_negative_long)) if len(y_dead_negative_long) else 0.0,
    )
    log.info(
        "[ENTRY_TEASER_LONG_RULES] split=%s mfe_min=%.2f mfe_max=%.2f mae_min=%.2f path_max=%.2f rate=%.6f",
        _split_tag,
        float(TEASER_LONG_MIN_MFE_BPS),
        float(TEASER_LONG_MAX_MFE_BPS),
        float(TEASER_LONG_MIN_MAE_BPS),
        float(TEASER_LONG_MAX_PATH_BPS),
        float(np.mean(y_teaser_negative_long)) if len(y_teaser_negative_long) else 0.0,
    )
    log.info(
        "[ENTRY_HARD_NEG_LONG_RULES] split=%s xgb_p_long_min=%.2f mfe_min=%.2f mae_min=%.2f path_max=%.2f rate=%.6f",
        _split_tag,
        float(HARD_NEG_LONG_MIN_XGB_P_LONG),
        float(HARD_NEG_LONG_MIN_MFE_BPS),
        float(HARD_NEG_LONG_MIN_MAE_BPS),
        float(HARD_NEG_LONG_MAX_PATH_BPS),
        float(np.mean(y_hard_negative_long)) if len(y_hard_negative_long) else 0.0,
    )
    log.info(
        "[ENTRY_CLEAN_EDGE_LONG_RULES] split=%s mfe_min=%.2f mae_max=%.2f path_min=%.2f rate=%.6f",
        _split_tag,
        float(CLEAN_EDGE_LONG_MFE_MIN_BPS),
        float(CLEAN_EDGE_LONG_MAE_MAX_BPS),
        float(CLEAN_EDGE_LONG_PATH_MIN_BPS),
        float(np.mean(y_clean_edge_long)) if len(y_clean_edge_long) else 0.0,
    )
    log.info(
        "[ENTRY_SURVIVAL_LONG_RULES] split=%s mfe_min=%.2f mae_max=%.2f path_min=%.2f rate=%.6f",
        _split_tag,
        float(SURVIVAL_LONG_MFE_MIN_BPS),
        float(SURVIVAL_LONG_MAE_MAX_BPS),
        float(SURVIVAL_LONG_PATH_MIN_BPS),
        float(np.mean(y_survival_long)) if len(y_survival_long) else 0.0,
    )
    log.info(
        "[ENTRY_LONG_WINDOW_TEACHER_LABELS] split=%s bad_rate=%.6f winner_rate=%.6f selector_mask_rate=%.6f",
        _split_tag,
        float(np.mean(y_teacher_bad_long)) if len(y_teacher_bad_long) else 0.0,
        float(np.mean(y_teacher_winner_long)) if len(y_teacher_winner_long) else 0.0,
        float(np.mean(y_selector_long_mask)) if len(y_selector_long_mask) else 0.0,
    )

    # Tradable rate proof (split-aware)
    _tradable_rate = float(np.mean(y_tradable)) if len(y_tradable) else 0.0
    log.info(
        "[ENTRY_TRADABLE_RATE_PROOF] split=%s n_rows=%d tradable_rate=%.6f",
        _split_tag,
        len(y_tradable),
        _tradable_rate,
    )
    _quality_side_rate = float(np.mean(_quality_side != -1)) if len(_quality_side) else 0.0
    log.info(
        "[ENTRY_QUALITY_SIDE_RATE_PROOF] split=%s n_rows=%d quality_side_rate=%.6f",
        _split_tag,
        len(_quality_side),
        _quality_side_rate,
    )
    _side_long = (y_dir == 0)
    _side_short = (y_dir == 1)
    _long_rate = float(np.mean(y_tradable[_side_long])) if _side_long.any() else 0.0
    _short_rate = float(np.mean(y_tradable[_side_short])) if _side_short.any() else 0.0
    log.info(
        "[ENTRY_TRADABLE_RATE_BY_SIDE] split=%s long_rate=%.6f short_rate=%.6f",
        _split_tag,
        _long_rate,
        _short_rate,
    )
    if "session_id" in merged3.columns:
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        _sess_names = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
        for sid in sorted(set(_sess.tolist())):
            _mask = (_sess == sid)
            _rate = float(np.mean(y_tradable[_mask])) if _mask.any() else 0.0
            log.info(
                "[ENTRY_TRADABLE_RATE_BY_SESSION] split=%s session_id=%d session_name=%s rate=%.6f",
                _split_tag,
                int(sid),
                _sess_names.get(int(sid), "UNK"),
                _rate,
            )

    n = len(merged3)
    if n < (seq_len + 1):
        raise RuntimeError(f"TOO_FEW_ROWS_FOR_SEQ: rows={n} seq_len={seq_len}")

    # V2 STREAMING-WRITE: Build rows in batches, write each batch to parquet via
    # pyarrow.parquet.ParquetWriter, and free memory between batches. The previous
    # accumulate-then-DataFrame pattern OOM'd at ~14 GB on full 311k rows because
    # 96-bar × 37-feature seq + ctx + labels per row × 311k rows ~= 25 GB DataFrame
    # in the 16 GB sandbox.
    import pyarrow as _pa
    import pyarrow.parquet as _pq

    def _to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [_to_list(v) for v in x]
        return x

    streaming_active = output_path is not None
    if streaming_active:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "[V2_STREAMING_WRITE] enabled: output=%s batch_size=%d total_emitted_rows~=%d",
            str(output_path), int(streaming_batch_size), n - (seq_len - 1),
        )

    pq_writer = None
    pq_schema = None
    summary_rows: List[Dict[str, Any]] = []  # only identity + labels (no seq/snap/ctx) — small
    total_written = 0

    def _emit_batch(rows_batch: List[Dict[str, Any]]) -> None:
        nonlocal pq_writer, pq_schema, total_written
        if not rows_batch:
            return
        df_b = pd.DataFrame(rows_batch)
        df_b["seq"] = df_b["seq"].apply(_to_list)
        df_b["snap"] = df_b["snap"].apply(_to_list)
        df_b["ctx_cont"] = df_b["ctx_cont"].apply(_to_list)
        df_b["ctx_cat"] = df_b["ctx_cat"].apply(_to_list)
        if streaming_active:
            if pq_schema is None:
                table = _pa.Table.from_pandas(df_b, preserve_index=False)
                pq_schema = table.schema
                pq_writer = _pq.ParquetWriter(str(output_path), pq_schema, compression="snappy")
                pq_writer.write_table(table)
            else:
                table = _pa.Table.from_pandas(df_b, schema=pq_schema, preserve_index=False, safe=False)
                pq_writer.write_table(table)
        # Always keep tiny summary row (identity + labels) for downstream logging
        summary_rows.extend(
            df_b[
                [c for c in df_b.columns if c not in ("seq", "snap", "ctx_cont", "ctx_cat")]
            ].to_dict(orient="records")
        )
        total_written += len(df_b)
        del df_b, rows_batch[:]

    pending: List[Dict[str, Any]] = []
    # Start index at seq_len-1 so we have a full history ending at i
    for i in range(seq_len - 1, n):
        seq = sig_mat[i - (seq_len - 1) : i + 1]  # V2: [seq_len, 37]
        snap = sig_mat[i]  # V2: [37]
        pending.append(
            {
                "time": times[i],
                "seq": seq,
                "snap": snap,
                "ctx_cont": ctx_cont_mat[i],
                "ctx_cat": ctx_cat_mat[i],
                "y_direction": y_dir[i],
                "y_early_move": y_early[i],
                "y_quality_score": y_qual[i],
                "y_bad_path": y_bad_path[i],
                "y_tradable": y_tradable[i],
                "y_tf_agreement_score": y_tf_agreement[i],
                "y_position_size_target": y_position_size[i],
                "y_hold_horizon_target": y_hold_horizon[i],
                "mae_first_n_bps": y_mae_first_n[i],
                "mfe_first_n_bps": y_mfe_first_n[i],
                "path_quality_bps": y_path_quality[i],
                "y_dead_negative_long": y_dead_negative_long[i],
                "y_teaser_negative_long": y_teaser_negative_long[i],
                "y_hard_negative_long": y_hard_negative_long[i],
                "y_clean_edge_long": y_clean_edge_long[i],
                "y_survival_long": y_survival_long[i],
                "y_teacher_bad_long": y_teacher_bad_long[i],
                "y_teacher_winner_long": y_teacher_winner_long[i],
                "y_selector_long_mask": y_selector_long_mask[i],
                "y_dead_negative_short": y_dead_negative_short[i],
                "y_teaser_negative_short": y_teaser_negative_short[i],
                "y_hard_negative_short": y_hard_negative_short[i],
                "y_clean_edge_short": y_clean_edge_short[i],
                "y_survival_short": y_survival_short[i],
                "y_selector_short_mask": y_selector_short_mask[i],
                "y_clean_edge_bidir": y_clean_edge_bidir[i],
                "y_survival_bidir": y_survival_bidir[i],
                "label_horizon_bars": y_label_horizon[i],
                "path_quality_horizon_bars": y_path_horizon[i],
                # aux-head regression targets (dip/forecast/timing/tail/vol) — emit
                # so trainer's row.get(col) finds real values (not silent 0.0).
                **{_c: _head_target_arrays[_c][i] for _c in _HEAD_TARGET_COLS},
            }
        )
        if len(pending) >= streaming_batch_size:
            _emit_batch(pending)
            if total_written % (streaming_batch_size * 10) == 0:
                log.info("[V2_STREAMING_WRITE] flushed %d rows", total_written)
    _emit_batch(pending)

    if pq_writer is not None:
        pq_writer.close()
        log.info("[V2_STREAMING_WRITE] closed writer, total_rows=%d", total_written)

    df_out = pd.DataFrame(summary_rows)
    if len(df_out) == 0:
        raise RuntimeError("BUILD_EMPTY_OUTPUT")

    mae_missing = int(pd.isna(df_out["mae_first_n_bps"]).sum())
    mfe_missing = int(pd.isna(df_out["mfe_first_n_bps"]).sum())
    log.info(
        "[ENTRY_PATH_QUALITY_PROOF] rows=%d mae_missing=%d mfe_missing=%d",
        int(len(df_out)),
        mae_missing,
        mfe_missing,
    )
    log.info(
        "[ENTRY_INPUT_SCHEMA_PROOF] signal_dim=%d base_signal_dim=%d seq_structure_dim=%d ctx_cont_dim=%d ctx_cat_dim=%d",
        int(sig_mat.shape[1]),
        int(len(SIGNAL_FIELDS)),
        int(len(seq_structure_feature_names)),
        int(len(ctx_cont_names)),
        int(len(ctx_cat_names)),
    )

    # 10) Metadata
    _hold_bars = int(horizon_bars)
    meta: Dict[str, Any] = {
        "rows": int(len(df_out)),
        "seq_len": int(seq_len),
        "hold_bars": _hold_bars,
        "fixed_hold_bootstrap_bars": _hold_bars,
        **direction_label_contract(),
        "early_move_threshold_bps": float(early_move_threshold_bps),
        "flat_threshold_bps": float(flat_threshold_bps),
        "base28_manifest": {
            "path": str(base28_manifest_path),
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
        },
        "xgb_bundle": str(Path(xgb_bundle_path).resolve()),
        "xgb_model_sha256": xgb_model_sha256,
        "neutral_xgb_bridge": bool(neutral_xgb_bridge),
        "xgb_bridge_source": "neutral_uniform_proba" if neutral_xgb_bridge else "xgb_bundle_inference",
        "tape_root": str(Path(tape_root).resolve()),
        "join_ratio_tape": float(rows_joined / max(1, rows_base28)),
        "signal_bridge": {
            "id": SIGNAL_BRIDGE_ID_V3,
            "neutral_xgb_bridge": bool(neutral_xgb_bridge),
            "bridge_source": "neutral_uniform_proba" if neutral_xgb_bridge else "xgb_bundle_inference",
            "fields": list(signal_fields_emitted),
            "base_fields": list(SIGNAL_FIELDS),
            "snap_fields": list(snap_fields_emitted),
            "seq_input_dim": int(sig_mat.shape[1]),
            "snap_input_dim": int(sig_mat.shape[1]),
            "base_seq_input_dim": int(len(SIGNAL_FIELDS)),
            "seq_structure_extension_dim": int(len(seq_structure_feature_names)),
            "contract_sha256": SIGNAL_CONTRACT_SHA256,
            "proba_dim_seen": 3,
            "bridge_dim": 7,
            "seq_structure_extension_v1": {
                "enabled": bool(seq_structure_feature_names),
                **{k: v for k, v in seq_structure_meta.items() if k != "manifest"},
            },
        },
        "ctx_contract": {
            # R4: self-describing tag/dim from the ACTUAL emitted ctx_cat (5/6), not the v1 anchor.
            "tag": f"CTX6CAT{len(ctx_cat_names)}",
            "ctx_cont_dim": int(len(ctx_cont_names)),
            "ctx_cat_dim": int(len(ctx_cat_names)),
            "ctx_cont_names": list(ctx_cont_names),
            "ctx_cat_names": list(ctx_cat_names),
            "allow_zero_ctx": bool(allow_zero_ctx),
            "ctx_cont_base_dim": int(ctx["ctx_cont_dim"]),
            "ctx_cont_micro_features": list(MICRO_FEATURE_NAMES),
            "ctx_cont_swing_features": list(SWING_FEATURE_NAMES),
            "ctx_cont_session_features": list(SESSION_CTX_CONT_NAMES),
        },
        "lane_contract": {
            "direction_policy": "LONG_ONLY_PREMIUM" if CANONICAL_PREMIUM_LONG_ONLY else "BIDIRECTIONAL_PREMIUM",
            "entry_admission_policy": "OVERLAP_LONG_REPLACES_OLDEST_OVERLAP_SHORT_WHEN_FULL",
            "entry_runtime_gates": [
                "flat_veto",
                "tradable_gate",
                "quality_gate",
            ],
            "max_open_trades": 10,
        },
        "strict_entry_labels": {
            **direction_label_contract(),
            "fixed_hold_bootstrap_bars": _hold_bars,
            "tradable_mfe_min_bps": float(STRICT_TRADABLE_MFE_MIN_BPS),
            "tradable_mae_max_bps": float(STRICT_TRADABLE_MAE_MAX_BPS),
            "tradable_path_min_bps": float(STRICT_TRADABLE_PATH_MIN_BPS),
            "tradable_path_lead_min_bps": float(STRICT_TRADABLE_PATH_LEAD_MIN_BPS),
            "tradable_mfe_lead_min_bps": float(STRICT_TRADABLE_MFE_LEAD_MIN_BPS),
            "direction_follows_tradable_side": True,
            "canonical_premium_long_only": bool(CANONICAL_PREMIUM_LONG_ONLY),
            "hard_negative_long_xgb_p_long_min": float(HARD_NEG_LONG_MIN_XGB_P_LONG),
            "hard_negative_long_mfe_min_bps": float(HARD_NEG_LONG_MIN_MFE_BPS),
            "hard_negative_long_mae_min_bps": float(HARD_NEG_LONG_MIN_MAE_BPS),
            "hard_negative_long_path_max_bps": float(HARD_NEG_LONG_MAX_PATH_BPS),
            "dead_long_mfe_max_bps": float(DEAD_LONG_MAX_MFE_BPS),
            "dead_long_mae_min_bps": float(DEAD_LONG_MIN_MAE_BPS),
            "teaser_long_mfe_min_bps": float(TEASER_LONG_MIN_MFE_BPS),
            "teaser_long_mfe_max_bps": float(TEASER_LONG_MAX_MFE_BPS),
            "teaser_long_mae_min_bps": float(TEASER_LONG_MIN_MAE_BPS),
            "teaser_long_path_max_bps": float(TEASER_LONG_MAX_PATH_BPS),
            "clean_edge_long_mfe_min_bps": float(CLEAN_EDGE_LONG_MFE_MIN_BPS),
            "clean_edge_long_mae_max_bps": float(CLEAN_EDGE_LONG_MAE_MAX_BPS),
            "clean_edge_long_path_min_bps": float(CLEAN_EDGE_LONG_PATH_MIN_BPS),
            "survival_long_mfe_min_bps": float(SURVIVAL_LONG_MFE_MIN_BPS),
            "survival_long_mae_max_bps": float(SURVIVAL_LONG_MAE_MAX_BPS),
            "survival_long_path_min_bps": float(SURVIVAL_LONG_PATH_MIN_BPS),
            "tradable_excludes_bad_path": True,
            "teacher_source": {
                "run_specs": [
                    {
                        "label": str(spec["label"]),
                        "source_model": str(spec["source_model"]),
                        "run_root": str(spec["run_root"]),
                    }
                    for spec in LONG_WINDOW_TEACHER_RUN_SPECS
                ],
                "teacher_bad_exit_reasons": sorted(LONG_WINDOW_TEACHER_BAD_EXIT_REASONS),
                "meaningful_mfe_bps": float(LONG_WINDOW_TEACHER_MEANINGFUL_MFE_BPS),
                "collapse_min_mfe_bps": float(LONG_WINDOW_TEACHER_COLLAPSE_MIN_MFE_BPS),
                "collapse_max_retain_frac": float(LONG_WINDOW_TEACHER_COLLAPSE_MAX_RETAIN_FRAC),
                "collapse_max_pnl_bps": float(LONG_WINDOW_TEACHER_COLLAPSE_MAX_PNL_BPS),
                "aged_rot_min_bars": int(LONG_WINDOW_TEACHER_AGED_ROT_MIN_BARS),
                "aged_rot_min_time_since_mfe_bars": int(LONG_WINDOW_TEACHER_AGED_ROT_MIN_TIME_SINCE_MFE_BARS),
                "aged_rot_min_dd_from_mfe_bps": float(LONG_WINDOW_TEACHER_AGED_ROT_MIN_DD_FROM_MFE_BPS),
                "aged_rot_max_retain_frac": float(LONG_WINDOW_TEACHER_AGED_ROT_MAX_RETAIN_FRAC),
                "aged_rot_max_pnl_bps": float(LONG_WINDOW_TEACHER_AGED_ROT_MAX_PNL_BPS),
                "premium_min_pnl_bps": float(LONG_WINDOW_TEACHER_PREMIUM_MIN_PNL_BPS),
                "premium_min_mfe_bps": float(LONG_WINDOW_TEACHER_PREMIUM_MIN_MFE_BPS),
                "premium_min_retain_frac": float(LONG_WINDOW_TEACHER_PREMIUM_MIN_RETAIN_FRAC),
                "eof_premium_min_pnl_bps": float(LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_PNL_BPS),
                "eof_premium_min_mfe_bps": float(LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_MFE_BPS),
                "eof_premium_min_retain_frac": float(LONG_WINDOW_TEACHER_EOF_PREMIUM_MIN_RETAIN_FRAC),
                "teacher_bad_m5_rows": int(np.sum(y_teacher_bad_long)),
                "teacher_winner_m5_rows": int(np.sum(y_teacher_winner_long)),
            },
        },
        "parked_targets": {
            "bad_path": {
                "horizon_bars": int(BAD_PATH_HORIZON_BARS),
                "mae_threshold_bps": float(BAD_PATH_MAE_THRESHOLD_BPS),
                "mfe_threshold_bps": float(BAD_PATH_MFE_THRESHOLD_BPS),
            }
        },
        "base28_feature_contract": {
            "features": list(features),
            "contract_path": str(contract_path),
        },
    }

    return df_out, meta


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ENTRY_V10_CTX training dataset (canonical, CTX6CAT6-only; advanced seq/snap/ctx structure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--truth-config",
        type=str,
        required=False,
        help="Path to canonical truth config JSON. If provided, base28_manifest and xgb_bundle are resolved from it.",
    )
    parser.add_argument(
        "--base28_manifest",
        type=str,
        required=False,
        help="Path to BASE28_CANONICAL CURRENT_MANIFEST.json (manifest-only resolution). Optional when --truth-config is set.",
    )
    parser.add_argument(
        "--xgb_bundle",
        type=str,
        required=False,
        help="Path to canonical XGB bundle directory (universal multihead v2; locked). Optional when --truth-config is set.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output dataset path (.parquet).")

    # Deterministic filters
    parser.add_argument("--start", type=str, default=None, help="Start datetime (ISO; UTC recommended).")
    parser.add_argument("--end", type=str, default=None, help="End datetime (ISO; UTC recommended).")
    parser.add_argument("--max_rows", type=int, default=None, help="Deterministic: take first N rows after filtering.")

    # Advanced dataset structure (V2 default: seq_len=96 for 8h M5 window)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN_V3, help=f"Sequence length for seq feature (default: {DEFAULT_SEQ_LEN_V3}).")
    parser.add_argument(
        "--canonical_v2_parquet",
        type=str,
        required=True,
        help="Path to the canonical_v2 / FULL_PLUS_CTX parquet (per-bar price-state + ctx_cont). Pre-rebuild "
             "fail-close (2026-06-04, rule 4 'no silent defaults'): REQUIRED — the old default was a 10-day-stale "
             "(ends 05-25) canonical_v2 that would silently poison the V10 dataset; pass the regime-fresh rebuild "
             "output explicitly (V3's --canonical-v2 is already required; this mirrors it).",
    )

    # BASE76 propagation: when training transformers against a BASE76 XGB bundle.
    # Overrides BASE28 manifest source + canonical_v3 92-feat contract.
    parser.add_argument(
        "--source-parquet-override",
        type=str,
        default=None,
        help="Override BASE28 manifest with a single-parquet source (e.g. canonical_v3_FULL_PLUS_CTX). "
             "Skips is_model_bar filter (must already be M5-cadence) and uses the XGB feature contract "
             "from --xgb-feature-contract-path.",
    )
    parser.add_argument(
        "--xgb-feature-contract-path",
        type=str,
        default=None,
        help="Override default XGB feature contract. Default: "
             "gx1/xgb/contracts/xgb_input_features_base80_v1.json",
    )
    parser.add_argument(
        "--xgb-sanitizer-config-path",
        type=str,
        default=None,
        help="Override default canonical_v3 sanitizer config. Must match feature contract.",
    )
    parser.add_argument(
        "--neutral-xgb-bridge",
        action="store_true",
        help="Keep the 7 bridge fields for contract compatibility but fill them with neutral "
             "1/3,1/3,1/3 probabilities instead of running XGBoost inference.",
    )
    parser.add_argument(
        "--seq-structure-features-parquet",
        type=str,
        default=None,
        help="Optional per-bar sequence-structure extension parquet keyed by time. "
             "When supplied, selected fields are appended to seq/snap after signal_bridge_v3 fields.",
    )
    parser.add_argument(
        "--seq-structure-manifest",
        type=str,
        default=None,
        help="Optional sequence-structure feature-layer manifest. Supplies selected feature order, parquet path, and sha.",
    )
    parser.add_argument(
        "--seq-structure-compute-inline",
        action="store_true",
        help="Compute selected sequence-structure features inline from per-bar merged3 instead of joining the parquet. "
             "Use this for true temporal seq extension builds.",
    )

    # Labels (fixed-hold)
    parser.add_argument("--hold-bars", dest="hold_bars", type=int, default=3, help="Fixed-hold label horizon in M5 bars (default: 3). Must be between 1 and 50.")
    parser.add_argument("--early_move_threshold_bps", type=float, default=4.0, help="Early-move threshold in bps (default: 4.0).")
    parser.add_argument("--flat_threshold_bps", type=float, default=3.0, help="Flat threshold in bps for 3-class labels (default: 3.0).")

    # Tape lane
    parser.add_argument(
        "--tape_root",
        type=str,
        default="",
        help="Override canonical tape lane root. Default resolves from $GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
    )

    # CTX handling
    parser.add_argument(
        "--allow_zero_ctx",
        action="store_true",
        help="If ctx columns are missing in BASE28, fill zeros instead of hard-fail (NOT recommended for final).",
    )

    # Output splitting scaffolding (kept for parity)
    parser.add_argument("--time_split", action="store_true", help="Write train/val/test outputs (time-based).")
    parser.add_argument("--train_start", type=str, default="2025-01-01T00:00:00Z", help="Train split start (ISO).")
    parser.add_argument("--train_end", type=str, default="2025-09-30T23:59:59Z", help="Train split end (ISO).")
    parser.add_argument("--val_start", type=str, default="2025-10-01T00:00:00Z", help="Val split start (ISO).")
    parser.add_argument("--val_end", type=str, default="2025-11-30T23:59:59Z", help="Val split end (ISO).")
    parser.add_argument("--test_start", type=str, default="2025-12-01T00:00:00Z", help="Test split start (ISO).")
    parser.add_argument("--test_end", type=str, default="2025-12-31T23:59:59Z", help="Test split end (ISO).")

    parser.add_argument("--dry_run", action="store_true", help="Dry run: validate inputs/ctx, then exit.")

    args = parser.parse_args()
    from gx1.runtime.entry_next_edge_legacy_guard import enforce_legacy_entry_research_ack
    enforce_legacy_entry_research_ack("legacy Entry V10_CTX training dataset builder")
    build_command = sys.argv.copy()

    # Hard gate: ONE UNIVERSE
    ctx = _hard_gate_ctx6cat6()
    log.info(f"[CTX_CONTRACT] OK: tag={ctx['tag']} cont={ctx['ctx_cont_dim']} cat={ctx['ctx_cat_dim']}")

    hold_bars = int(args.hold_bars)
    if hold_bars < 1 or hold_bars > 50:
        raise ValueError(f"HOLD_BARS_INVALID: {hold_bars} (must be 1..50)")
    log.info("[LABEL_HOLD] hold_bars=%d", hold_bars)
    flat_threshold_bps = float(args.flat_threshold_bps)
    if flat_threshold_bps < 0:
        raise ValueError(f"FLAT_THRESHOLD_INVALID: {flat_threshold_bps} (must be >=0)")
    log.info("[LABEL_FLAT] flat_threshold_bps=%.4f", flat_threshold_bps)

    # Dataset build proof (will be written after output_path resolved)
    proof_payload = {
        "ctx_tag": f"CTX6CAT{len(ORDERED_CTX_CAT_NAMES_V3)}",
        "ctx_cont_dim": int(len(ORDERED_CTX_CONT_NAMES_V3)),
        "ctx_cat_dim": int(len(ORDERED_CTX_CAT_NAMES_V3)),
        "signal_bridge_id": SIGNAL_BRIDGE_ID_V3,
        "signal_bridge_contract_sha256": str(SIGNAL_CONTRACT_SHA256),
        "hold_bars": hold_bars,
        "fixed_hold_bootstrap_bars": hold_bars,
        **direction_label_contract(),
        "flat_threshold_bps": float(flat_threshold_bps),
        "neutral_xgb_bridge": bool(args.neutral_xgb_bridge),
        "seq_structure_extension_v1": {
            "manifest_path": str(Path(args.seq_structure_manifest).expanduser().resolve()) if args.seq_structure_manifest else None,
            "parquet_path": str(Path(args.seq_structure_features_parquet).expanduser().resolve()) if args.seq_structure_features_parquet else None,
            "compute_inline": bool(args.seq_structure_compute_inline),
        },
    }

    # Resolve SSoT inputs (truth-config or manual)
    truth_config_path: Optional[Path] = None
    truth_tape_root_model: Optional[Path] = None
    if args.truth_config:
        truth_config_path = Path(args.truth_config).expanduser().resolve()
        if args.base28_manifest or args.xgb_bundle:
            raise SystemExit("[SPLIT_BRAIN_ARGS] truth-config provided but base28_manifest/xgb_bundle also supplied")
        if not truth_config_path.exists():
            raise RuntimeError(f"TRUTH_CONFIG_MISSING: {truth_config_path}")
        truth_obj = json.loads(truth_config_path.read_text())
        base28_manifest_path = Path(
            truth_obj.get(
                "canonical_prebuilt_manifest",
                "/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json",
            )
        ).expanduser().resolve()
        canonical_xgb_bundle_dir = str(truth_obj.get("canonical_xgb_bundle_dir") or "").strip()
        if not canonical_xgb_bundle_dir:
            raise RuntimeError(
                f"TRUTH_CONFIG_MISSING_CANONICAL_XGB_BUNDLE: canonical_xgb_bundle_dir missing in {truth_config_path}"
            )
        canonical_tape_root_model = str(truth_obj.get("canonical_market_tape_root_model") or "").strip()
        if canonical_tape_root_model:
            truth_tape_root_model = Path(canonical_tape_root_model).expanduser().resolve()
        xgb_bundle_path = Path(canonical_xgb_bundle_dir).expanduser().resolve()
        xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
        if xgb_override:
            override_path = Path(xgb_override).expanduser().resolve()
            if override_path != xgb_bundle_path:
                log.info(
                    "[XGB_OVERRIDE] truth_config_bundle=%s override_bundle=%s",
                    xgb_bundle_path,
                    override_path,
                )
            xgb_bundle_path = override_path
        log.info(
            "[TRUTH_CONFIG] Using truth-config %s -> base28_manifest=%s xgb_bundle=%s tape_root_model=%s",
            truth_config_path,
            base28_manifest_path,
            xgb_bundle_path,
            truth_tape_root_model,
        )
        proof_payload.update(
            {
                "truth_config_path": str(truth_config_path),
                "truth_source": "truth_config",
                "truth_tape_root_model": str(truth_tape_root_model) if truth_tape_root_model is not None else None,
            }
        )
    else:
        if not args.xgb_bundle:
            raise SystemExit("--xgb_bundle required when --truth-config is not provided")
        if not args.source_parquet_override and not args.base28_manifest:
            raise SystemExit("Either --base28_manifest or --source-parquet-override required when --truth-config is not provided")
        # When --source-parquet-override is set, BASE28 manifest is bypassed entirely.
        base28_manifest_path = Path(args.base28_manifest).resolve() if args.base28_manifest else Path("/dev/null")
        xgb_bundle_path = Path(args.xgb_bundle).resolve()
        xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
        if xgb_override:
            override_path = Path(xgb_override).expanduser().resolve()
            if override_path != xgb_bundle_path:
                raise RuntimeError(
                    f"[XGB_OVERRIDE_MISMATCH] GX1_XGB_BUNDLE_DIR={override_path} != --xgb_bundle={xgb_bundle_path}"
                )
            xgb_bundle_path = override_path
        proof_payload.update({"truth_config_path": None, "truth_source": "manual_args"})
    if not args.source_parquet_override:
        _ensure_inputs_exist(base28_manifest_path, xgb_bundle_path)
    elif not xgb_bundle_path.exists():
        raise RuntimeError(f"INPUT_MISSING: xgb_bundle not found: {xgb_bundle_path}")

    output_path = Path(args.output).resolve()
    hold_suffix = f"HOLD_{hold_bars:02d}B"
    if hold_suffix not in output_path.stem:
        output_path = output_path.with_name(f"{output_path.stem}__{hold_suffix}{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write proof payload
    try:
        proof_path = output_path.parent / "DATASET_BUILD_PROOF.json"
        with open(proof_path, "w") as f:
            json.dump(proof_payload, f, indent=2)
        log.info("[DATASET_BUILD_PROOF] wrote %s", proof_path)
    except Exception as e:
        log.warning("[DATASET_BUILD_PROOF] failed to write proof file: %s", e)
    proof_payload.update(
        {
            "base28_manifest_path": str(base28_manifest_path),
            "xgb_bundle_path": str(xgb_bundle_path),
            "output_path": str(output_path),
        }
    )

    model_file = xgb_bundle_path / "xgb_universal_multihead_v2.joblib"
    if bool(args.neutral_xgb_bridge):
        xgb_model_sha256 = None
        log.info("[NEUTRAL_XGB_BRIDGE] main preflight skips XGB model sha/load")
    else:
        if not model_file.exists():
            raise RuntimeError(f"XGB_MODEL_MISSING: {model_file}")
        xgb_model_sha256 = _sha256_file(model_file)
    proof_payload["xgb_model_sha256"] = xgb_model_sha256
    log.info(
        "[XGB_BUNDLE_PROOF] xgb_bundle=%s model_file=%s model_sha256=%s neutral=%s",
        xgb_bundle_path,
        model_file,
        xgb_model_sha256,
        bool(args.neutral_xgb_bridge),
    )

    start = _parse_ts(args.start)
    end = _parse_ts(args.end)

    # Tape root resolution
    if args.tape_root.strip():
        tape_root = Path(args.tape_root).expanduser().resolve()
    elif truth_tape_root_model is not None:
        tape_root = truth_tape_root_model
    else:
        gx1_data = _resolve_gx1_data_root()
        tape_root = gx1_data / "data" / "oanda" / "canonical" / "xauusd_m5_bid_ask__CANONICAL"

    if args.dry_run:
        log.info("[DRY_RUN] Inputs exist and CTX contract is valid. Exiting.")
        write_manifest(
            output_path=output_path,
            build_command=build_command,
            base28_manifest=base28_manifest_path,
            xgb_bundle=xgb_bundle_path,
            tape_root=tape_root,
            notes="DRY_RUN only.",
            extra={
                "start": args.start,
                "end": args.end,
                "max_rows": args.max_rows,
                "time_split": bool(args.time_split),
                "seq_len": int(args.seq_len),
                "hold_bars": int(hold_bars),
                "fixed_hold_bootstrap_bars": int(hold_bars),
                **direction_label_contract(),
                "early_move_threshold_bps": float(args.early_move_threshold_bps),
                "allow_zero_ctx": bool(args.allow_zero_ctx),
                "xgb_model_sha256": xgb_model_sha256,
                "neutral_xgb_bridge": bool(args.neutral_xgb_bridge),
                "seq_structure_extension_v1": {
                    "manifest_path": str(Path(args.seq_structure_manifest).expanduser().resolve()) if args.seq_structure_manifest else None,
                    "parquet_path": str(Path(args.seq_structure_features_parquet).expanduser().resolve()) if args.seq_structure_features_parquet else None,
                    "compute_inline": bool(args.seq_structure_compute_inline),
                },
            },
        )
        return

    if args.time_split:
        train_start = _parse_ts(args.train_start)
        train_end = _parse_ts(args.train_end)
        val_start = _parse_ts(args.val_start)
        val_end = _parse_ts(args.val_end)
        test_start = _parse_ts(args.test_start)
        test_end = _parse_ts(args.test_end)

        splits = {
            "train": {"start": str(train_start), "end": str(train_end)},
            "val": {"start": str(val_start), "end": str(val_end)},
            "test": {"start": str(test_start), "end": str(test_end)},
        }

        base = output_path
        out_dir = base.parent
        stem = base.stem

        metas: Dict[str, Any] = {}
        ts_min_max_by_split: Dict[str, Dict[str, Optional[str]]] = {}

        for split_name, (s0, s1) in {
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        }.items():
            log.info(f"[BUILD] split={split_name} start={s0} end={s1}")
            out = out_dir / f"{stem}_{split_name}.parquet"
            df_built, meta = build_dataset_canonical(
                base28_manifest_path=base28_manifest_path,
                xgb_bundle_path=xgb_bundle_path,
                tape_root=tape_root,
                start=s0,
                end=s1,
                max_rows=args.max_rows,
                seq_len=int(args.seq_len),
                horizon_bars=int(hold_bars),
                early_move_threshold_bps=float(args.early_move_threshold_bps),
                flat_threshold_bps=float(flat_threshold_bps),
                allow_zero_ctx=bool(args.allow_zero_ctx),
                split_name=split_name,
                canonical_v2_parquet=Path(args.canonical_v2_parquet).expanduser().resolve(),
                output_path=out,  # V2 streaming-write target
                source_parquet_override=(Path(args.source_parquet_override).expanduser().resolve() if args.source_parquet_override else None),
                xgb_feature_contract_path=(Path(args.xgb_feature_contract_path).expanduser().resolve() if args.xgb_feature_contract_path else None),
                xgb_sanitizer_config_path=(Path(args.xgb_sanitizer_config_path).expanduser().resolve() if args.xgb_sanitizer_config_path else None),
                seq_structure_features_parquet=(Path(args.seq_structure_features_parquet).expanduser().resolve() if args.seq_structure_features_parquet else None),
                seq_structure_manifest_path=(Path(args.seq_structure_manifest).expanduser().resolve() if args.seq_structure_manifest else None),
                seq_structure_compute_inline=bool(args.seq_structure_compute_inline),
                neutral_xgb_bridge=bool(args.neutral_xgb_bridge),
            )
            _log_label_distribution_proof(df_built, split=split_name)
            # V2: parquet already written by streaming_write inside build_dataset_canonical
            metas[split_name] = deepcopy(meta)

            ts_min_max_by_split[split_name] = _split_min_max_from_ts_series(df_built["time"])

            write_manifest(
                output_path=out,
                build_command=build_command,
                base28_manifest=base28_manifest_path,
                xgb_bundle=xgb_bundle_path,
                tape_root=tape_root,
                splits=splits,
                ts_min_max_by_split=ts_min_max_by_split,
                notes=f"Canonical build completed for split={split_name}.",
                extra=metas[split_name],
            )

        log.info("[DATASET_BUILD] Time-split build complete!")
        return

    # Single output
    df_built, meta = build_dataset_canonical(
        base28_manifest_path=base28_manifest_path,
        xgb_bundle_path=xgb_bundle_path,
        tape_root=tape_root,
        start=start,
        end=end,
        max_rows=args.max_rows,
        seq_len=int(args.seq_len),
        horizon_bars=int(hold_bars),
        early_move_threshold_bps=float(args.early_move_threshold_bps),
        flat_threshold_bps=float(flat_threshold_bps),
        allow_zero_ctx=bool(args.allow_zero_ctx),
        split_name="full",
        canonical_v2_parquet=Path(args.canonical_v2_parquet).expanduser().resolve(),
        output_path=output_path,  # V2 streaming-write target
        source_parquet_override=(Path(args.source_parquet_override).expanduser().resolve() if args.source_parquet_override else None),
        xgb_feature_contract_path=(Path(args.xgb_feature_contract_path).expanduser().resolve() if args.xgb_feature_contract_path else None),
        xgb_sanitizer_config_path=(Path(args.xgb_sanitizer_config_path).expanduser().resolve() if args.xgb_sanitizer_config_path else None),
        seq_structure_features_parquet=(Path(args.seq_structure_features_parquet).expanduser().resolve() if args.seq_structure_features_parquet else None),
        seq_structure_manifest_path=(Path(args.seq_structure_manifest).expanduser().resolve() if args.seq_structure_manifest else None),
        seq_structure_compute_inline=bool(args.seq_structure_compute_inline),
        neutral_xgb_bridge=bool(args.neutral_xgb_bridge),
    )
    _log_label_distribution_proof(df_built, split="SINGLE")
    # V2: parquet already written by streaming_write inside build_dataset_canonical
    log.info(f"✅ Saved dataset: {output_path}")

    ts_min_max_by_split = {"SINGLE": _split_min_max_from_ts_series(df_built["time"])}

    write_manifest(
        output_path=output_path,
        build_command=build_command,
        base28_manifest=base28_manifest_path,
        xgb_bundle=xgb_bundle_path,
        tape_root=tape_root,
        splits=None,
        ts_min_max_by_split=ts_min_max_by_split,
        notes="Canonical build completed (single).",
        extra=meta,
    )

    log.info("[DATASET_BUILD] Dataset build complete!")


if __name__ == "__main__":
    main()
