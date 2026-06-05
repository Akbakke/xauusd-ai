#!/usr/bin/env python3
"""
Train Universal Multi-head XGB v2 on canonical BASE28 data.

Trains one XGB head per session (ASIA, EU, OVERLAP, US) with 3-class ATR
triple-barrier labels (LONG / SHORT / FLAT, multi:softprob):
- LONG:  future_return_bps >= +T
- SHORT: future_return_bps <= -T
Rows within (-T, +T) are dropped (no FLAT in this iteration).

All heads are stored in a single artifact bundle:
- xgb_universal_multihead_v2.joblib
- xgb_universal_multihead_v2_meta.json

Canonical defaults:
- Data root:   /home/andre2/GX1_DATA
- Prebuilt:    BASE28 canonical parquet
- Tape truth:  /home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL
- Output:      GX1_DATA/models/models/xgb_universal_multihead_v2__<timestamp>

This trainer is intended to be the active writer for the bundle family used by:
- XGB_SIGNAL_BRIDGE_V1
- gx1.xgb.multihead.xgb_multihead_model_v1
- replay / truth lanes

Example (full retrain):
    GX1_DATA=/home/andre2/GX1_DATA \
    /home/andre2/venvs/gx1/bin/python -m gx1.scripts.train_xgb_universal_multihead_v2 \
      --years 2020 2021 2022 2023 2024 2025 \
      --sessions ASIA EU OVERLAP US \
      --output-dir /home/andre2/GX1_DATA/models/models/xgb_universal_multihead_v2__RETRAIN_20260306_000000

Smoke example:
    GX1_DATA=/home/andre2/GX1_DATA \
    GX1_XGB_USE_CLASS_WEIGHTS=1 \
    GX1_XGB_LOG_MARGIN_METRICS=1 \
    /home/andre2/venvs/gx1/bin/python -m gx1.scripts.train_xgb_universal_multihead_v2 \
      --years 2025 \
      --sessions ASIA EU OVERLAP US \
      --n-bars 120000 \
      --min-bars-per-head 5000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Repo root: /.../GX1_ENGINE
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import xgboost as xgb
    from joblib import dump as joblib_dump
    from sklearn.metrics import accuracy_score, log_loss
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install xgboost scikit-learn joblib")
    sys.exit(1)

from gx1.time.session_detector import (
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_stats,
    get_session_vectorized,
)
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer


# -----------------------------
# Helpers
# -----------------------------

def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()


def _timestamp_tag() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _compute_sha256(filepath: Path) -> Optional[str]:
    if not filepath.exists() or not filepath.is_file():
        return None
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_list_sha256(features: Sequence[str]) -> str:
    return hashlib.sha256("|".join(str(f) for f in features).encode("utf-8")).hexdigest()


def _derive_session_context_features(df: pd.DataFrame, ts_col: str = "_ts_utc") -> pd.DataFrame:
    """
    Derive canonical session-context features directly from UTC timestamps.

    This keeps XGB feature sourcing deterministic and independent of optional
    precomputed session columns in prebuilt data.
    """
    if ts_col not in df.columns:
        raise KeyError(f"Missing timestamp column for session context: {ts_col}")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Timestamp conversion failed for session context derivation in {ts_col}")

    session = get_session_vectorized(ts)
    session_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
    session_id = session.map(session_map).astype(np.int32)
    out = df.copy()
    out["_session"] = session
    out["session_id"] = session_id
    out["is_ASIA"] = (session_id == 0).astype(np.int8)
    out["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(ts).astype(np.float32)
    out["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    out["session_change_flag"] = (session_id.diff().fillna(0) != 0).astype(np.int8)
    out["session_tradable"] = (session_id != 0).astype(np.int8)
    return out


def _resolve_gx1_data_dir() -> Path:
    for env_key in ("GX1_DATA", "GX1_DATA_ROOT"):
        env_val = os.environ.get(env_key)
        if env_val:
            p = Path(env_val).expanduser().resolve()
            if p.exists():
                return p
    default = Path("/home/andre2/GX1_DATA")
    if default.exists():
        return default
    # Fallback for odd environments
    fallback = WORKSPACE_ROOT.parent / "GX1_DATA"
    return fallback.resolve()


def _candidate_canonical_manifest_paths(gx1_data: Path) -> List[Path]:
    return [
        gx1_data / "data" / "data" / "prebuilt" / "BASE28_CANONICAL" / "CURRENT_MANIFEST.json",
        gx1_data / "data" / "prebuilt" / "BASE28_CANONICAL" / "CURRENT_MANIFEST.json",
    ]


def _candidate_canonical_parquet_paths(gx1_data: Path) -> List[Path]:
    return [
        gx1_data / "data" / "data" / "prebuilt" / "BASE28_CANONICAL" / "xauusd_m5_BASE28_2020_2025.parquet",
        gx1_data / "data" / "prebuilt" / "BASE28_CANONICAL" / "xauusd_m5_BASE28_2020_2025.parquet",
    ]


def _resolve_canonical_manifest_path(gx1_data: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        p = explicit.expanduser().resolve()
        return p if p.exists() else None
    for p in _candidate_canonical_manifest_paths(gx1_data):
        if p.exists():
            return p.resolve()
    return None


def _resolve_canonical_parquet_path(
    gx1_data: Path,
    explicit_parquet: Optional[Path],
    manifest_path: Optional[Path],
) -> Path:
    if explicit_parquet:
        p = explicit_parquet.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Explicit parquet not found: {p}")
        return p

    # Try manifest first
    if manifest_path and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            candidate_keys = [
                "canonical_prebuilt_parquet",
                "prebuilt_parquet",
                "parquet_path",
                "canonical_parquet_path",
                "resolved_parquet_path",
            ]
            for key in candidate_keys:
                if key in manifest:
                    p = Path(manifest[key]).expanduser().resolve()
                    if p.exists():
                        return p
        except Exception:
            pass

    for p in _candidate_canonical_parquet_paths(gx1_data):
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "Could not resolve canonical BASE28 parquet. "
        "Pass --canonical-prebuilt-parquet explicitly or ensure BASE28 canonical files exist."
    )


def _resolve_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    """
    Resolve timestamp source.

    Returns:
        - column name if timestamp lives in a column
        - None if timestamp should be taken from index
    """
    candidates = [
        "timestamp",
        "ts",
        "time",
        "datetime",
        "open_time",
        "open_ts",
        "close_time",
        "event_time",
        "bar_ts",
        "bar_time",
        "Date",
        "date",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: use index if it looks timestamp-like
    if isinstance(df.index, pd.DatetimeIndex):
        return None

    # Try coercing index to datetime
    idx_as_ts = pd.to_datetime(df.index, utc=True, errors="coerce")
    if not pd.isna(idx_as_ts).all():
        return None

    preview_cols = list(df.columns[:30])
    raise KeyError(
        "No timestamp column found and index is not timestamp-like. "
        f"Checked columns={candidates}. Preview first columns={preview_cols}"
    )


def _to_utc_timestamp_series(df: pd.DataFrame, ts_col: Optional[str]) -> pd.Series:
    """
    Convert timestamp source to UTC pandas Series.
    If ts_col is None, use dataframe index.
    """
    raw = df.index.to_series(index=df.index) if ts_col is None else df[ts_col]
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if ts.isna().any():
        n_bad = int(ts.isna().sum())
        raise ValueError(
            f"Timestamp conversion failed for {n_bad} rows "
            f"(source={'index' if ts_col is None else ts_col})"
        )
    return pd.Series(ts, index=df.index)


def _load_canonical_tape(tape_root: Path, years: Sequence[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    years_sorted = sorted({int(y) for y in years})
    if not years_sorted:
        raise RuntimeError("No years provided for tape load")

    for year in years_sorted:
        path = tape_root / f"year={year}" / "part-000.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Canonical tape parquet not found: {path}")

        tape = pd.read_parquet(path)
        required = {"time", "close"}
        missing = sorted(required - set(tape.columns))
        if missing:
            raise RuntimeError(f"Canonical tape missing required columns={missing} path={path}")

        tape = tape[["time", "close"]].copy()
        tape["_ts_utc"] = pd.to_datetime(tape["time"], utc=True, errors="coerce")
        if tape["_ts_utc"].isna().any():
            raise ValueError(f"Canonical tape has non-coercible timestamps in {path}")
        if tape["close"].isna().any():
            raise ValueError(f"Canonical tape has NaN close values in {path}")

        frames.append(tape[["_ts_utc", "close"]])

    tape_all = pd.concat(frames, ignore_index=True)
    if tape_all["_ts_utc"].duplicated().any():
        dup_count = int(tape_all["_ts_utc"].duplicated().sum())
        raise RuntimeError(f"Canonical tape has duplicate timestamps: {dup_count}")
    if tape_all["close"].isna().any():
        raise ValueError("Canonical tape has NaN close values after concat")

    return tape_all


def _load_feature_contracts(
    feature_contract_path: Optional[Path] = None,
    sanitizer_config_path: Optional[Path] = None,
) -> Tuple[
    List[str],
    str,
    XGBInputSanitizer,
    Optional[str],
    Optional[str],
    Optional[str],
    Path,
    Path,
    Path,
]:
    if feature_contract_path is None:
        # BASE76 is the ONE active contract (V10/V3 expect it). Default here so a
        # re-run without --feature-contract-path can never silently rebuild the
        # wrong (legacy base28) contract into the same bundle filename.
        feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_base80_v1.json"
    if not feature_contract_path.exists():
        raise FileNotFoundError(f"Feature contract not found: {feature_contract_path}")

    feature_contract = json.loads(feature_contract_path.read_text())
    features = feature_contract.get("features", [])
    schema_hash = feature_contract.get("schema_hash", "unknown")

    if sanitizer_config_path is None:
        sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base80_v1.json"
    if not sanitizer_config_path.exists():
        raise FileNotFoundError(f"Sanitizer config not found: {sanitizer_config_path}")

    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"

    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config_path))

    return (
        features,
        schema_hash,
        sanitizer,
        _compute_sha256(feature_contract_path),
        _compute_sha256(sanitizer_config_path),
        _compute_sha256(output_contract_path) if output_contract_path.exists() else None,
        feature_contract_path,
        sanitizer_config_path,
        output_contract_path,
    )


def _margin_top1_top2(y_pred_proba: np.ndarray) -> np.ndarray:
    if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] < 2:
        raise ValueError(f"Invalid y_pred_proba shape: {y_pred_proba.shape}")
    top2 = np.partition(y_pred_proba, -2, axis=1)
    return top2[:, -1] - top2[:, -2]


def _margin_stats(vals: np.ndarray) -> Dict[str, float]:
    if len(vals) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def eval_margin_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    side_from: str = "y_true",
    log_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Research metric:
      [XGB_MARGIN_METRIC] n=... margin_mean=... margin_median=... margin_p95=...
      [XGB_MARGIN_BY_SIDE] side=long ...
      [XGB_MARGIN_BY_SIDE] side=short ...
    """
    margin = _margin_top1_top2(y_pred_proba)
    overall = _margin_stats(margin)
    n = int(len(margin))

    if log_metrics:
        print(
            "[XGB_MARGIN_METRIC] "
            f"n={n} "
            f"margin_mean={overall['mean']:.6f} "
            f"margin_median={overall['median']:.6f} "
            f"margin_p95={overall['p95']:.6f}"
        )

    if side_from == "y_pred":
        side_labels = np.argmax(y_pred_proba, axis=1)
    else:
        side_labels = y_true.astype(int)

    out: Dict[str, Any] = {"overall": {"n": n, **overall}, "by_side": {}}

    for side_name, side_label in (("long", 0), ("short", 1)):
        mask = side_labels == side_label
        vals = margin[mask]
        stats = _margin_stats(vals)
        out["by_side"][side_name] = {"n": int(mask.sum()), **stats}
        if log_metrics:
            print(
                "[XGB_MARGIN_BY_SIDE] "
                f"side={side_name} "
                f"n={int(mask.sum())} "
                f"mean={stats['mean']:.6f} "
                f"median={stats['median']:.6f} "
                f"p95={stats['p95']:.6f}"
            )
    return out


def compute_class_weights(y_train: np.ndarray, n_classes: int = 3) -> Dict[int, float]:
    """
    Inverse-frequency class weights normalized so mean(sample_weight)=1.
    """
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return {i: 1.0 for i in range(n_classes)}

    freq = counts / total
    weights_raw = np.zeros_like(freq)
    for cls in range(n_classes):
        if counts[cls] > 0:
            weights_raw[cls] = 1.0 / freq[cls]
        else:
            weights_raw[cls] = 0.0

    row_weights = np.array([weights_raw[int(y)] for y in y_train], dtype=np.float64)
    if row_weights.sum() <= 0:
        return {i: 1.0 for i in range(n_classes)}

    scale = len(row_weights) / row_weights.sum()
    weights_norm = weights_raw * scale
    return {cls: float(weights_norm[cls]) for cls in range(n_classes)}


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    lookahead_bars: int,
    atr_col: str,
    session_col: str,
    session_multipliers: Dict[str, float],
    no_hit_return_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ATR-normalized triple-barrier labels using tape close.

    LONG  if +barrier hit first within horizon
    SHORT if -barrier hit first within horizon
    FLAT  if no barrier hit within horizon and terminal return is small
    (optional) No-hit promotion to LONG/SHORT if terminal return exceeds factor*barrier
    invalid (-1) for missing/invalid ATR or insufficient lookahead
    """
    if "close" not in df.columns:
        preview_cols = list(df.columns[:50])
        raise KeyError(
            "Required price column 'close' not found in dataframe after tape join. "
            f"Preview first columns={preview_cols}"
        )
    if atr_col not in df.columns:
        preview_cols = list(df.columns[:50])
        raise KeyError(
            f"Required ATR column '{atr_col}' not found in dataframe. "
            f"Preview first columns={preview_cols}"
        )
    if session_col not in df.columns:
        preview_cols = list(df.columns[:50])
        raise KeyError(
            f"Required session column '{session_col}' not found in dataframe. "
            f"Preview first columns={preview_cols}"
        )

    close = df["close"].to_numpy(dtype=np.float64)
    atr_bps = df[atr_col].to_numpy(dtype=np.float64)
    sessions = df[session_col].astype(str).to_numpy()

    n = len(df)
    labels = np.full(n, -1, dtype=np.int32)
    hit_code = np.full(n, -1, dtype=np.int32)  # 0=LONG,1=SHORT,2=NO_HIT,-1=INVALID
    valid_mask = np.ones(n, dtype=bool)
    if lookahead_bars > 0:
        valid_mask[-lookahead_bars:] = False

    for i in range(n - lookahead_bars):
        sess = sessions[i]
        if sess not in session_multipliers:
            raise KeyError(f"Session '{sess}' not in session_multipliers")

        atr_val = atr_bps[i]
        if not np.isfinite(atr_val) or atr_val <= 0.0:
            valid_mask[i] = False
            continue

        barrier = float(atr_val) * float(session_multipliers[sess])
        if not np.isfinite(barrier) or barrier <= 0.0:
            valid_mask[i] = False
            continue

        window = close[i + 1 : i + lookahead_bars + 1]
        if window.size == 0:
            valid_mask[i] = False
            continue

        rets_bps = (window - close[i]) * 10_000.0 / close[i]
        long_hits = np.where(rets_bps >= barrier)[0]
        short_hits = np.where(rets_bps <= -barrier)[0]

        if long_hits.size == 0 and short_hits.size == 0:
            # No barrier hit: optionally promote to LONG/SHORT based on terminal return
            terminal_ret = float(rets_bps[-1])
            promote_barrier = float(barrier) * float(no_hit_return_factor)
            if np.isfinite(promote_barrier) and promote_barrier > 0.0:
                if terminal_ret >= promote_barrier:
                    labels[i] = 0  # LONG (no-hit promotion)
                    hit_code[i] = 3
                    continue
                if terminal_ret <= -promote_barrier:
                    labels[i] = 1  # SHORT (no-hit promotion)
                    hit_code[i] = 4
                    continue
            labels[i] = 2  # FLAT
            hit_code[i] = 2
            continue
        if long_hits.size > 0 and short_hits.size > 0:
            if long_hits[0] < short_hits[0]:
                labels[i] = 0  # LONG
                hit_code[i] = 0
            elif short_hits[0] < long_hits[0]:
                labels[i] = 1  # SHORT
                hit_code[i] = 1
            else:
                # Same-bar hit: pick larger magnitude
                if abs(rets_bps[long_hits[0]]) >= abs(rets_bps[short_hits[0]]):
                    labels[i] = 0
                    hit_code[i] = 0
                else:
                    labels[i] = 1
                    hit_code[i] = 1
        elif long_hits.size > 0:
            labels[i] = 0
            hit_code[i] = 0
        else:
            labels[i] = 1
            hit_code[i] = 1

    return labels, valid_mask, hit_code


def _ensure_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ensure_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_ensure_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train Universal Multi-head XGB v2 on canonical BASE28")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--sessions", type=str, nargs="+", default=["ASIA", "EU", "OVERLAP", "US"])
    parser.add_argument("--canonical-prebuilt-manifest", type=Path, default=None)
    parser.add_argument("--canonical-prebuilt-parquet", type=Path, default=None)
    parser.add_argument(
        "--feature-contract-path", type=Path, default=None,
        help="Override feature contract path. Default: gx1/xgb/contracts/xgb_input_features_base80_v1.json",
    )
    parser.add_argument(
        "--sanitizer-config-path", type=Path, default=None,
        help="Override sanitizer config path. Default: gx1/xgb/contracts/xgb_input_sanitizer_base80_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-bars", type=int, default=None, help="Optional cap after year-filter, preserves chronological order")
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--lookahead-bars", type=int, default=24)  # BASE76 bundle was trained at 24 (~2h M5)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument(
        "--tape-root",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--min-bars-per-head", type=int, default=50000)
    parser.add_argument("--target-flat-min", type=float, default=0.70)
    parser.add_argument("--target-flat-max", type=float, default=0.90)
    parser.add_argument("--min-class-rate", type=float, default=0.02)
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    )
    parser.add_argument(
        "--use-class-weights",
        dest="use_class_weights",
        action="store_true",
        default=_bool_env("GX1_XGB_USE_CLASS_WEIGHTS", True),
    )
    parser.add_argument(
        "--no-use-class-weights",
        dest="use_class_weights",
        action="store_false",
    )
    parser.add_argument(
        "--log-margin-metrics",
        dest="log_margin_metrics",
        action="store_true",
        default=_bool_env("GX1_XGB_LOG_MARGIN_METRICS", True),
    )
    parser.add_argument(
        "--no-log-margin-metrics",
        dest="log_margin_metrics",
        action="store_false",
    )

    parser.add_argument(
        "--vedtak", type=str, default=None,
        help="Explicit retrain decision id — rule 3, NEVER auto-retrain (gx1_guards fail-closed).",
    )
    args = parser.parse_args()
    # NEVER auto-retrain — fail-closed unless an explicit --vedtak is passed (CLAUDE.md rule 3).
    from gx1_guards.gates import require_retrain_vedtak, GateError
    try:
        require_retrain_vedtak(args.vedtak)
    except GateError as e:
        parser.error(str(e))
    np.random.seed(args.seed)

    print("=" * 80)
    print("TRAIN UNIVERSAL MULTI-HEAD XGB V2 (CANONICAL BASE28)")
    print("=" * 80)

    gx1_data = _resolve_gx1_data_dir()
    # Pre-rebuild fail-close (2026-06-04, rule 4): an XGB RETRAIN requires an EXPLICIT M5 BASE80 training
    # parquet. The no-arg path resolves via BASE28_CANONICAL/CURRENT_MANIFEST.json — but that manifest is a
    # LIVE-M1-serve pointer (rewritten every 15s by v12_canonical_incremental, read by serve/bootstrap), NOT
    # a frozen M5 training source. Training off it = a moving/wrong-granularity target. Pass --canonical-prebuilt-parquet.
    if args.canonical_prebuilt_parquet is None:
        raise RuntimeError(
            "[XGB_TRAIN_NO_SILENT_DEFAULT] --canonical-prebuilt-parquet is REQUIRED for a retrain — refusing to "
            "auto-resolve the BASE28_CANONICAL manifest (a live-M1-serve pointer, not an M5 training source). "
            "Pass the freshly-built, pinned M5 BASE80 prebuilt explicitly."
        )
    manifest_path = _resolve_canonical_manifest_path(gx1_data, args.canonical_prebuilt_manifest)
    parquet_path = _resolve_canonical_parquet_path(gx1_data, args.canonical_prebuilt_parquet, manifest_path)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = (
            gx1_data
            / "models"
            / "models"
            / f"xgb_universal_multihead_v2__{_timestamp_tag()}"
        ).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GX1_DATA:            {gx1_data}")
    print(f"Manifest:            {manifest_path if manifest_path else 'NONE'}")
    print(f"Canonical parquet:   {parquet_path}")
    print(f"Output dir:          {output_dir}")
    print(f"Years:               {args.years}")
    canonical_sessions = ["ASIA", "EU", "OVERLAP", "US"]
    requested = [s.upper() for s in args.sessions]
    sessions = [s for s in canonical_sessions if s in requested]
    if sessions != canonical_sessions:
        raise RuntimeError(
            f"[XGB_SESSION_HEADS_INVALID] expected heads={canonical_sessions} got={sessions} "
            f"(requested={requested})"
        )
    print(f"Sessions:            {sessions}")
    print(f"Use class weights:   {args.use_class_weights}")
    print(f"Log margin metrics:  {args.log_margin_metrics}")

    (
        features,
        schema_hash,
        sanitizer,
        feature_contract_sha,
        sanitizer_sha,
        output_contract_sha,
        feature_contract_path,
        sanitizer_config_path,
        output_contract_path,
    ) = _load_feature_contracts(
        feature_contract_path=args.feature_contract_path,
        sanitizer_config_path=args.sanitizer_config_path,
    )

    print(f"Feature contract:    {feature_contract_path}")
    print(f"Sanitizer config:    {sanitizer_config_path}")
    print(f"Output contract:     {output_contract_path}")

    print("\nLoading canonical parquet...")
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise RuntimeError(f"Loaded empty parquet: {parquet_path}")

    ts_col = _resolve_timestamp_col(df)
    df["_ts_utc"] = _to_utc_timestamp_series(df, ts_col)
    df["_year"] = df["_ts_utc"].dt.year
    # Pre-rebuild fail-close (rule 4): assert the training parquet is M5 (median bar spacing ~300s), NOT the
    # M1 live tape — catches a wrong-granularity source before the expensive tape-join (base80/_v1 features are
    # M5-baked; an M1 input silently trains garbage).
    _dts = df["_ts_utc"].sort_values().diff().dropna().dt.total_seconds()
    if len(_dts) and abs(float(_dts.median()) - 300.0) > 1.0:
        raise RuntimeError(
            f"[XGB_TRAIN_GRANULARITY] training parquet median bar spacing = {float(_dts.median()):.0f}s, expected "
            f"300s (M5). Refusing to train XGB on a non-M5 source ({parquet_path})."
        )

    years_set = set(int(y) for y in args.years)
    df = df.loc[df["_year"].isin(years_set)].copy()
    if df.empty:
        raise RuntimeError(f"No rows remain after filtering to years={sorted(years_set)}")

    if args.n_bars is not None and len(df) > args.n_bars:
        df = df.iloc[: args.n_bars].copy()

    df = df.sort_values("_ts_utc", kind="stable").reset_index(drop=True)

    print(f"Rows after year filter/cap: {len(df):,}")
    print(f"Timestamp range:            {df['_ts_utc'].iloc[0]} -> {df['_ts_utc'].iloc[-1]}")

    year_counts = df["_year"].value_counts().sort_index()
    print("Rows by year:")
    for year, count in year_counts.items():
        print(f"  {int(year)}: {int(count):,}")

    tape_root = args.tape_root.expanduser().resolve()
    print(f"Tape root:           {tape_root}")

    tape_df = _load_canonical_tape(tape_root, years_set)
    # 2026-05-24 FIX: if rebuilt canonical contains "close" column, drop it before
    # merge so tape's close wins without suffix collision (pandas would rename to
    # close_x/close_y otherwise → KeyError downstream).
    if "close" in df.columns:
        df = df.drop(columns=["close"])
    rows_base28 = len(df)
    rows_tape = len(tape_df)
    joined_df = df.merge(
        tape_df,
        on="_ts_utc",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    rows_joined = len(joined_df)
    exact_match = int(rows_joined == rows_base28)
    print(
        "[XGB_TAPE_JOIN] "
        f"rows_base28={rows_base28} "
        f"rows_tape={rows_tape} "
        f"rows_joined={rows_joined} "
        f"exact_match={exact_match}"
    )
    if exact_match != 1:
        raise RuntimeError(
            "Tape join is not an exact 1:1 match with BASE28 rows. "
            f"rows_base28={rows_base28} rows_tape={rows_tape} rows_joined={rows_joined}"
        )
    if joined_df["close"].isna().any():
        raise RuntimeError("Tape join produced NaN close values")

    df = joined_df
    df = df.sort_values("_ts_utc", kind="stable").reset_index(drop=True)

    print("Computing sessions + session-context features...")
    df = _derive_session_context_features(df, ts_col="_ts_utc")
    if df["_session"].isna().any():
        raise RuntimeError("Session computation produced NaN values")
    required_session_ctx = [
        "session_id",
        "is_ASIA",
        "minutes_since_session_open",
        "minutes_to_next_session_boundary",
        "session_change_flag",
        "session_tradable",
    ]
    missing_session_ctx = [c for c in required_session_ctx if c not in df.columns]
    if missing_session_ctx:
        raise RuntimeError(f"Missing derived session context columns: {missing_session_ctx}")
    session_stats = get_session_stats(df["_session"])
    percentages = session_stats.get("percentages", {})
    print(
        "Session mix: "
        f"EU={percentages.get('EU', 0):.1f}% "
        f"OVERLAP={percentages.get('OVERLAP', 0):.1f}% "
        f"US={percentages.get('US', 0):.1f}% "
        f"ASIA={percentages.get('ASIA', 0):.1f}%"
    )

    # 2026-05-24 BASE76: derive features dropped by canonical_v3_augment as redundant.
    # Duplicates: re-expose under historical names so BASE76 contract is satisfied
    # without bloating canonical_v3 parquet.
    # Session-interaction (*_us): _v1_is_US gated derivatives; same formulas as
    # v12_ctx_augment_live._derive_us_interactions (parity with live).
    sid = df["session_id"].astype(np.int32)
    # X1 (2026-06-04 train==serve parity): PRESERVE the prebuilt's baked _v1_is_EU/_v1_is_US (+ the
    # *_us interactions derived from them) — they carry the SHIFTED prev-bar convention (basic_v1.py:
    # 981-988) the live serve path feeds. Overwriting unconditionally with the UNSHIFTED current-bar
    # session_id flipped ~0.7% of session-boundary bars (+3 interactions) -> trainer!=serve. Guard
    # `if col not in df.columns` (matches the if-missing pattern in the V10/V3 builders); only
    # synthesize when genuinely absent. Interactions use the (baked-or-fresh) is_US -> consistent basis.
    _is_eu_baked = "_v1_is_EU" in df.columns  # N2 (2026-06-05): track preserve-vs-synthesize for honest log
    _is_us_baked = "_v1_is_US" in df.columns
    if not _is_eu_baked:
        df["_v1_is_EU"] = (sid == 1).astype(np.int8)
    if not _is_us_baked:
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
    if _is_eu_baked and _is_us_baked:
        print("[XGB] derived BASE76 missing features; PRESERVED baked-shifted _v1_is_EU/_v1_is_US (train==serve X1 parity)")
    else:
        print(f"[XGB] derived BASE76 missing features; SYNTHESIZED UNSHIFTED _v1_is_EU(baked={_is_eu_baked})/"
              f"_v1_is_US(baked={_is_us_baked}) — WARNING: live serve feeds SHIFTED; bake shifted in base80 for train==serve")

    # ATR-normalized triple-barrier labels (session-aware, 3-class)
    # 2026-05-24 FIX: if atr_bps not in canonical, derive from _v1_atr14 / close * 1e4
    if "atr_bps" not in df.columns:
        if "_v1_atr14" in df.columns and "close" in df.columns:
            df["atr_bps"] = (df["_v1_atr14"].astype(float) / df["close"].astype(float).clip(lower=1e-6) * 1e4).clip(0, 500).fillna(0.0)
            print("[XGB] derived atr_bps from _v1_atr14 / close * 1e4 (no atr_bps in canonical)")
        else:
            raise KeyError("atr_bps missing and cannot derive from _v1_atr14 + close")
    atr_col = "atr_bps"
    session_multipliers = {
        "EU": 1.35,
        "OVERLAP": 1.10,
        "US": 1.0,
        "ASIA": 1.0,
    }
    # Reduce FLAT share moderately by promoting no-hit cases with a terminal return
    # that exceeds a fraction of the barrier.
    no_hit_return_factor = 0.50

    print("\nComputing triple-barrier labels (ATR-normalized)...")
    labels, valid_mask, hit_code = compute_triple_barrier_labels(
        df=df,
        lookahead_bars=args.lookahead_bars,
        atr_col=atr_col,
        session_col="_session",
        session_multipliers=session_multipliers,
        no_hit_return_factor=no_hit_return_factor,
    )
    df["_label"] = labels
    df["_valid_mask"] = valid_mask
    df["_label_valid"] = df["_valid_mask"] & (df["_label"] >= 0)
    df["_hit_code"] = hit_code

    n_total = int(len(df))
    n_valid = int(df["_label_valid"].sum())
    n_long = int((df["_label"] == 0).sum())
    n_short = int((df["_label"] == 1).sum())
    n_flat = int((df["_label"] == 2).sum())
    long_share = float(n_long / max(1, n_long + n_short + n_flat))
    print(
        "[XGB_LABELS] "
        f"total_rows={n_total:,} "
        f"valid_rows={n_valid:,} "
        f"LONG={n_long:,} "
        f"SHORT={n_short:,} "
        f"FLAT={n_flat:,} "
        f"long_share={long_share:.4f} "
        f"horizon_bars={args.lookahead_bars} "
        f"atr_col={atr_col} "
        f"session_multipliers={session_multipliers} "
        f"no_hit_return_factor={no_hit_return_factor}"
    )

    # Precompute barrier_bps per row for audit
    session_mult_series = df["_session"].map(session_multipliers).astype(float)
    barrier_bps = df[atr_col].astype(float) * session_mult_series
    df["_barrier_bps"] = barrier_bps

    print("[XGB_LABELS_BY_SESSION]")
    # XGB-10 (2026-06-03): capture the TRUE per-session barrier stats so head_metrics can
    # record honest provenance instead of the fabricated hardcoded threshold_bps=6.0.
    barrier_stats_by_session: Dict[str, Dict[str, float]] = {}
    for sess in sessions:
        mask = (df["_session"] == sess) & df["_label_valid"]
        n_sess = int(mask.sum())
        n_long_sess = int((df.loc[mask, "_label"] == 0).sum())
        n_short_sess = int((df.loc[mask, "_label"] == 1).sum())
        n_flat_sess = int((df.loc[mask, "_label"] == 2).sum())
        long_share_sess = float(n_long_sess / max(1, n_long_sess + n_short_sess + n_flat_sess))
        hit_long = int(((df["_session"] == sess) & (df["_hit_code"] == 0)).sum())
        hit_short = int(((df["_session"] == sess) & (df["_hit_code"] == 1)).sum())
        hit_no = int(((df["_session"] == sess) & (df["_hit_code"] == 2)).sum())
        hit_no_promote_long = int(((df["_session"] == sess) & (df["_hit_code"] == 3)).sum())
        hit_no_promote_short = int(((df["_session"] == sess) & (df["_hit_code"] == 4)).sum())
        dropped = int(((df["_session"] == sess) & (~df["_valid_mask"])).sum())
        atr_vals = df.loc[mask, atr_col].to_numpy(dtype=np.float64)
        barrier_vals = df.loc[mask, "_barrier_bps"].to_numpy(dtype=np.float64)
        atr_mean = float(np.nanmean(atr_vals)) if atr_vals.size else float("nan")
        atr_median = float(np.nanmedian(atr_vals)) if atr_vals.size else float("nan")
        barrier_mean = float(np.nanmean(barrier_vals)) if barrier_vals.size else float("nan")
        barrier_median = float(np.nanmedian(barrier_vals)) if barrier_vals.size else float("nan")
        print(
            f"  session={sess} "
            f"valid_rows={n_sess:,} "
            f"LONG={n_long_sess:,} "
            f"SHORT={n_short_sess:,} "
            f"FLAT={n_flat_sess:,} "
            f"long_share={long_share_sess:.4f} "
            f"first_hit_long={hit_long:,} "
            f"first_hit_short={hit_short:,} "
            f"no_hit={hit_no:,} "
            f"no_hit_promote_long={hit_no_promote_long:,} "
            f"no_hit_promote_short={hit_no_promote_short:,} "
            f"dropped={dropped:,} "
            f"atr_bps_mean={atr_mean:.4f} atr_bps_median={atr_median:.4f} "
            f"barrier_bps_mean={barrier_mean:.4f} barrier_bps_median={barrier_median:.4f}"
        )
        barrier_stats_by_session[sess] = {
            "session_multiplier": float(session_multipliers.get(sess, 1.0)),
            "atr_bps_mean": atr_mean,
            "atr_bps_median": atr_median,
            "barrier_bps_mean": barrier_mean,
            "barrier_bps_median": barrier_median,
        }

    print("\nSanitizing XGB inputs...")
    X_all, sanitize_stats = sanitizer.sanitize(df, features, allow_nan_fill=True, nan_fill_value=0.0)
    if X_all.shape[0] != len(df):
        raise RuntimeError(f"Sanitized X shape mismatch: {X_all.shape[0]} != {len(df)}")

    print(f"Sanitized shape:           {X_all.shape}")
    print(f"Sanitizer clip rate:       {_safe_float(getattr(sanitize_stats, 'clip_rate_pct', float('nan'))):.4f}%")

    heads: Dict[str, Any] = {}
    head_metrics: Dict[str, Dict[str, Any]] = {}
    head_signatures: Dict[str, Dict[str, float]] = {}
    head_eval: Dict[str, Dict[str, np.ndarray]] = {}

    xgb_params: Dict[str, Any] = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": args.seed,
        "n_jobs": -1,
        "early_stopping_rounds": args.early_stopping_rounds,
        "tree_method": "hist",
    }
    print(
        "[XGB_3CLASS_MODE] "
        "enabled=True "
        f"objective={xgb_params['objective']} "
        "expected_proba_dim=3 "
        "p_flat=from_model"
    )

    print("\nTraining heads...")
    print("-" * 80)

    for session in sessions:
        print(f"\n[HEAD] session={session}")

        session_mask = (df["_session"] == session) & df["_label_valid"]
        n_session = int(session_mask.sum())
        print(f"  Session rows (valid): {n_session:,}")

        if n_session < args.min_bars_per_head:
            raise RuntimeError(
                f"Too few bars for {session}: {n_session} < min-bars-per-head={args.min_bars_per_head}"
            )

        idx = np.where(session_mask.to_numpy(dtype=bool))[0]
        X_session = X_all[idx]
        y_session = df.loc[session_mask, "_label"].to_numpy(dtype=np.int32)
        ts_session = df.loc[session_mask, "_ts_utc"].reset_index(drop=True)
        year_session = df.loc[session_mask, "_year"].reset_index(drop=True)

        n_total = len(X_session)
        n_train = int(n_total * (1.0 - args.val_split))
        if n_train <= 0 or n_train >= n_total:
            raise RuntimeError(
                f"Invalid split for session={session}: n_total={n_total}, n_train={n_train}"
            )

        # XGB-3 (2026-06-03 cross-model scan): chronological EMBARGO between train and val.
        # The triple-barrier label at row i looks `lookahead_bars` rows into the future, so
        # the last `lookahead_bars` train rows have forward windows that overlap the first
        # val rows -> early-stopping and the held-out probability calibrator both leak. We
        # PURGE those train rows; the val block stays anchored at the original cut (n_train),
        # so val is unchanged and only the train tail shrinks (~lookahead_bars rows / session).
        # Default = lookahead_bars; set GX1_XGB_TRAIN_EMBARGO_BARS=0 to reproduce the
        # pre-2026-06-03 cement exactly (leaky, for bit-parity reproduction only).
        _embargo = int(os.environ.get("GX1_XGB_TRAIN_EMBARGO_BARS", str(int(args.lookahead_bars))))
        _embargo = max(0, _embargo)
        n_train_emb = max(0, n_train - _embargo)
        if n_train_emb <= 0:
            raise RuntimeError(
                f"Embargo too large for session={session}: n_train={n_train}, "
                f"embargo={_embargo} -> {n_train_emb} train rows"
            )
        if _embargo > 0:
            print(f"  Train embargo:         drop last {_embargo} rows "
                  f"({n_train} -> {n_train_emb}) so no fwd-label overlaps val")

        # Time-based split: no shuffle, no funny business. Train tail purged by embargo.
        X_train = X_session[:n_train_emb]
        X_val = X_session[n_train:]
        y_train = y_session[:n_train_emb]
        y_val = y_session[n_train:]
        ts_train = ts_session.iloc[:n_train_emb]
        ts_val = ts_session.iloc[n_train:]
        year_train = year_session.iloc[:n_train_emb]
        year_val = year_session.iloc[n_train:]

        print(f"  Train rows:            {len(X_train):,}")
        print(f"  Val rows:              {len(X_val):,}")
        print(f"  Train ts range:        {ts_train.iloc[0]} -> {ts_train.iloc[-1]}")
        print(f"  Val ts range:          {ts_val.iloc[0]} -> {ts_val.iloc[-1]}")

        train_year_counts = year_train.value_counts().sort_index().to_dict()
        val_year_counts = year_val.value_counts().sort_index().to_dict()
        print(f"  Train rows by year:    {train_year_counts}")
        print(f"  Val rows by year:      {val_year_counts}")

        train_counts = {
            0: int((y_train == 0).sum()),
            1: int((y_train == 1).sum()),
            2: int((y_train == 2).sum()),
        }
        val_counts = {
            0: int((y_val == 0).sum()),
            1: int((y_val == 1).sum()),
            2: int((y_val == 2).sum()),
        }

        print(
            "  Train class counts:    "
            f"LONG={train_counts[0]:,} SHORT={train_counts[1]:,} FLAT={train_counts[2]:,}"
        )
        print(
            "  Val class counts:      "
            f"LONG={val_counts[0]:,} SHORT={val_counts[1]:,} FLAT={val_counts[2]:,}"
        )

        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }

        class_weight: Dict[int, float] = {0: 1.0, 1: 1.0, 2: 1.0}
        if args.use_class_weights:
            class_weight = compute_class_weights(y_train, n_classes=3)
            sample_weight_train = np.array([class_weight[int(y)] for y in y_train], dtype=np.float32)
            sample_weight_val = np.array([class_weight[int(y)] for y in y_val], dtype=np.float32)

            fit_kwargs["sample_weight"] = sample_weight_train
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]

        if args.use_class_weights:
            print(
                "  Class weights:         "
                f"LONG={class_weight[0]:.6f} "
                f"SHORT={class_weight[1]:.6f} "
                f"FLAT={class_weight[2]:.6f}"
            )
        else:
            print("  Class weights:         DISABLED")

        print("  Training XGBoost...")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, **fit_kwargs)

        print("  Evaluating...")
        y_val_pred = model.predict_proba(X_val)
        if y_val_pred.ndim != 2 or y_val_pred.shape[1] != 3:
            raise RuntimeError(
                f"Invalid predict_proba shape for {session}: {y_val_pred.shape}; expected (n, 3)"
            )

        val_logloss = float(log_loss(y_val, y_val_pred, labels=[0, 1, 2]))
        y_val_hat = np.argmax(y_val_pred, axis=1)
        val_accuracy = float(accuracy_score(y_val, y_val_hat))

        margin_metrics = eval_margin_metrics(
            y_true=y_val,
            y_pred_proba=y_val_pred,
            side_from="y_true",
            log_metrics=args.log_margin_metrics,
        )

        y_train_pred = model.predict_proba(X_train)
        train_p_long_mean = float(np.mean(y_train_pred[:, 0]))
        train_p_short_mean = float(np.mean(y_train_pred[:, 1]))
        train_p_flat_mean = float(np.mean(y_train_pred[:, 2]))
        print(
            "  Train prob means:      "
            f"p_long_mean={train_p_long_mean:.6f} "
            f"p_short_mean={train_p_short_mean:.6f} "
            f"p_flat_mean={train_p_flat_mean:.6f}"
        )

        print(f"  Val LogLoss:           {val_logloss:.6f}")
        print(f"  Val Accuracy:          {val_accuracy:.2%}")

        heads[session] = model

        total_counts = {
            "LONG": int(((np.concatenate([y_train, y_val]) == 0)).sum()),
            "SHORT": int(((np.concatenate([y_train, y_val]) == 1)).sum()),
            "FLAT": int(((np.concatenate([y_train, y_val]) == 2)).sum()),
        }
        total_n = sum(total_counts.values())

        head_metrics[session] = {
            "n_samples": int(n_total),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "bars_per_year": {str(int(k)): int(v) for k, v in year_session.value_counts().sort_index().to_dict().items()},
            # XGB-10: honest barrier provenance. The label barrier is ATR-normalized and
            # session-scaled (barrier_bps = atr_bps * session_multiplier), NOT a fixed
            # threshold. threshold_bps now reports the REAL per-session median barrier in
            # bps (was a fabricated hardcoded 6.0 with no relation to the labeler).
            "threshold_bps": float(barrier_stats_by_session.get(session, {}).get("barrier_bps_median", float("nan"))),
            "barrier_params": barrier_stats_by_session.get(session, {}),
            "class_distribution_total": {
                "LONG": float(total_counts["LONG"] / total_n),
                "SHORT": float(total_counts["SHORT"] / total_n),
                "FLAT": float(total_counts["FLAT"] / total_n),
            },
            "class_distribution_train": {
                "LONG": float(train_counts[0] / len(y_train)),
                "SHORT": float(train_counts[1] / len(y_train)),
                "FLAT": float(train_counts[2] / len(y_train)),
            },
            "class_distribution_val": {
                "LONG": float(val_counts[0] / len(y_val)),
                "SHORT": float(val_counts[1] / len(y_val)),
                "FLAT": float(val_counts[2] / len(y_val)),
            },
            "class_weights": {
                "LONG": float(class_weight[0]),
                "SHORT": float(class_weight[1]),
                "FLAT": float(class_weight[2]),
            },
            "val_logloss": val_logloss,
            "val_accuracy": val_accuracy,
            "margin_metrics": margin_metrics,
        }

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=np.float64)
            imp_df = pd.DataFrame({
                "feature": features[: len(imp)],
                "gain": imp,
            }).sort_values("gain", ascending=False)
            imp_path = output_dir / f"xgb_universal_multihead_v2_feature_importance_{session}.csv"
            imp_df.to_csv(imp_path, index=False)
            print(f"  Feature importance:    {imp_path.name}")

        # Head signature for divergence sanity
        proba_mean = y_val_pred.mean(axis=0)
        head_signatures[session] = {
            "p_long_mean": float(proba_mean[0]),
            "p_short_mean": float(proba_mean[1]),
            "p_flat_mean": float(proba_mean[2]) if len(proba_mean) > 2 else 0.0,
        }
        head_eval[session] = {
            "y_val": y_val,
            "p_long": y_val_pred[:, 0],
            "p_short": y_val_pred[:, 1],
            "p_flat": y_val_pred[:, 2],
        }
        print(
            "  Head signature:        "
            f"p_long={proba_mean[0]:.6f} "
            f"p_short={proba_mean[1]:.6f} "
            f"p_flat={proba_mean[2]:.6f}"
        )

    if not heads:
        raise RuntimeError("No heads trained")

    print("\nChecking head divergence...")
    sessions_list = list(head_signatures.keys())
    for i in range(len(sessions_list)):
        for j in range(i + 1, len(sessions_list)):
            s1, s2 = sessions_list[i], sessions_list[j]
            diff_long = abs(head_signatures[s1]["p_long_mean"] - head_signatures[s2]["p_long_mean"])
            diff_short = abs(head_signatures[s1]["p_short_mean"] - head_signatures[s2]["p_short_mean"])
            diff_flat = abs(head_signatures[s1]["p_flat_mean"] - head_signatures[s2]["p_flat_mean"])
            print(
                f"  {s1} vs {s2}: "
                f"Δp_long={diff_long:.6f} "
                f"Δp_short={diff_short:.6f} "
                f"Δp_flat={diff_flat:.6f}"
            )

    calibration_meta = None
    # Optional head calibration report (gated)
    if _bool_env("GX1_XGB_HEAD_CALIBRATE", False):
        from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS
        from gx1.xgb.calibration.isotonic_scaler import IsotonicScaler
        from gx1.xgb.calibration.platt_scaler import PlattScaler

        def _entropy_from_probs(p_long: np.ndarray, p_short: np.ndarray, p_flat: np.ndarray) -> np.ndarray:
            eps = 1e-12
            probs = np.clip(np.column_stack([p_long, p_short, p_flat]), eps, 1.0)
            return -np.sum(probs * np.log(probs), axis=1)

        def _calibration_bins(p_hat: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            rows = []
            for i in range(n_bins):
                lo, hi = bins[i], bins[i + 1]
                mask = (p_hat >= lo) & (p_hat < hi) if i < n_bins - 1 else (p_hat >= lo) & (p_hat <= hi)
                if not np.any(mask):
                    rows.append({"bin": i, "count": 0, "avg_pred": float("nan"), "avg_outcome": float("nan")})
                    continue
                rows.append(
                    {
                        "bin": i,
                        "count": int(mask.sum()),
                        "avg_pred": float(np.mean(p_hat[mask])),
                        "avg_outcome": float(np.mean(y_true[mask])),
                    }
                )
            return rows

        def _ece(p_hat: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
            rows = _calibration_bins(p_hat, y_true, n_bins=n_bins)
            total = max(1, len(p_hat))
            ece = 0.0
            for r in rows:
                if r["count"] <= 0 or not np.isfinite(r["avg_pred"]) or not np.isfinite(r["avg_outcome"]):
                    continue
                weight = r["count"] / total
                ece += weight * abs(r["avg_pred"] - r["avg_outcome"])
            return float(ece)

        def _max_gap(p_hat: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
            rows = _calibration_bins(p_hat, y_true, n_bins=n_bins)
            gaps = []
            for r in rows:
                if r["count"] <= 0 or not np.isfinite(r["avg_pred"]) or not np.isfinite(r["avg_outcome"]):
                    continue
                gaps.append(abs(r["avg_pred"] - r["avg_outcome"]))
            return float(max(gaps)) if gaps else float("nan")

        report_lines = []
        report_lines.append("# ENTRY_XGB_SESSION_CONTEXT_REPORT")
        report_lines.append("")
        report_lines.append(f"- Bundle: {output_dir}")
        report_lines.append(f"- Signal bridge keys: {list(ORDERED_FIELDS)}")
        report_lines.append(f"- Signal bridge dim: {len(ORDERED_FIELDS)}")
        report_lines.append("")
        report_lines.append("## Per-head calibration (p_hat vs correctness)")

        bins_rows = []
        sample_rows = []
        calibration_summary = {}
        calibrators_state = {}

        for head, ev in head_eval.items():
            y_val = ev["y_val"]
            p_long = ev["p_long"]
            p_short = ev["p_short"]
            p_flat = ev["p_flat"]
            probs = np.column_stack([p_long, p_short, p_flat])
            y_pred = np.argmax(probs, axis=1)
            correct = (y_pred == y_val).astype(np.int32)
            p_hat = np.max(probs, axis=1)
            ent = _entropy_from_probs(p_long, p_short, p_flat)

            mean_p_hat = float(np.mean(p_hat))
            std_p_hat = float(np.std(p_hat))
            mean_ent = float(np.mean(ent))
            std_ent = float(np.std(ent))
            ece = _ece(p_hat, correct, n_bins=10)
            max_gap = _max_gap(p_hat, correct, n_bins=10)

            calibration_summary[head] = {
                "n_samples": int(len(p_hat)),
                "p_hat_mean": mean_p_hat,
                "p_hat_std": std_p_hat,
                "entropy_mean": mean_ent,
                "entropy_std": std_ent,
                "ece": ece,
                "max_bin_gap": max_gap,
            }

            report_lines.append(
                f"- {head}: n={len(p_hat)}, p_hat_mean={mean_p_hat:.4f}, p_hat_std={std_p_hat:.4f}, "
                f"entropy_mean={mean_ent:.4f}, entropy_std={std_ent:.4f}, ece={ece:.4f}, max_gap={max_gap:.4f}"
            )

            for row in _calibration_bins(p_hat, correct, n_bins=10):
                bins_rows.append(
                    {
                        "head": head,
                        "bin": row["bin"],
                        "count": row["count"],
                        "avg_pred": row["avg_pred"],
                        "avg_outcome": row["avg_outcome"],
                    }
                )

            sample_rows.append(
                pd.DataFrame(
                    {
                        "head": head,
                        "p_hat": p_hat,
                        "entropy": ent,
                        "y_true": y_val,
                        "y_pred": y_pred,
                        "correct": correct,
                    }
                )
            )

        asia = calibration_summary.get("ASIA", {})
        asia_ok = bool(asia) and (asia.get("ece", 1.0) <= 0.05) and (asia.get("max_bin_gap", 1.0) <= 0.10)
        report_lines.append("")
        report_lines.append(f"## ASIA calibration: {'OK' if asia_ok else 'NOT OK'} (criteria: ECE<=0.05 and max_bin_gap<=0.10)")

        if not asia_ok:
            report_lines.append("")
            report_lines.append("## Calibrator")
            method = os.getenv("GX1_XGB_CALIBRATOR", "isotonic").lower()
            report_lines.append(f"- method={method}")
            for head, ev in head_eval.items():
                y_val = ev["y_val"]
                p_long = ev["p_long"]
                p_short = ev["p_short"]
                p_flat = ev["p_flat"]
                probs = np.column_stack([p_long, p_short, p_flat])
                y_pred = np.argmax(probs, axis=1)
                correct = (y_pred == y_val).astype(np.int32)
                p_hat = np.max(probs, axis=1)
                calibrator = IsotonicScaler(name="isotonic") if method == "isotonic" else PlattScaler(name="platt")
                calibrator.fit(p_hat, correct)
                p_hat_cal = calibrator.transform(p_hat)
                ece_cal = _ece(p_hat_cal, correct, n_bins=10)
                max_gap_cal = _max_gap(p_hat_cal, correct, n_bins=10)
                if method == "isotonic":
                    state = calibrator.get_state()
                    iso_state = state.get("isotonic", {})
                    x = iso_state.get("X_thresholds_", [])
                    y = iso_state.get("y_thresholds_", [])
                    calibrators_state[head] = {"x": x, "y": y}
                else:
                    calibrators_state[head] = calibrator.get_state()
                report_lines.append(
                    f"- {head}: ece_pre={calibration_summary[head]['ece']:.4f} ece_post={ece_cal:.4f} "
                    f"max_gap_pre={calibration_summary[head]['max_bin_gap']:.4f} max_gap_post={max_gap_cal:.4f}"
                )

        pd.DataFrame(bins_rows).to_csv(output_dir / "entry_score_deciles_by_session.csv", index=False)
        pd.concat(sample_rows, ignore_index=True).to_csv(output_dir / "entry_score_vs_pnl.csv", index=False)
        (output_dir / "ENTRY_XGB_SESSION_CONTEXT_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
        if calibrators_state:
            method = os.getenv("GX1_XGB_CALIBRATOR", "isotonic").lower()
            if method == "isotonic":
                calib_payload = {"method": "isotonic_interp", "heads": calibrators_state}
            else:
                calib_payload = {"method": method, "heads": calibrators_state}
            (output_dir / "CALIBRATION.json").write_text(json.dumps(calib_payload, indent=2), encoding="utf-8")
            calibration_meta = {"method": calib_payload["method"], "heads": list(calibrators_state.keys())}

    print("\nSaving bundle...")
    model_data: Dict[str, Any] = {
        "heads": heads,
        "feature_list": features,
        "meta": {
            "version": "xgb_universal_multihead_v2",
            "feature_names_ordered": list(features),
            "feature_list_sha256": _feature_list_sha256(features),
            "signal_bridge_version": "signal_bridge_v1",
            "created_at": _utc_now_iso(),
            "workspace_root": str(WORKSPACE_ROOT),
            "gx1_data": str(gx1_data),
            "canonical_prebuilt_manifest": str(manifest_path) if manifest_path else None,
            "canonical_prebuilt_parquet": str(parquet_path),
            "canonical_prebuilt_manifest_sha256": _compute_sha256(manifest_path) if manifest_path else None,
            "canonical_prebuilt_parquet_sha256": _compute_sha256(parquet_path),
            "schema_hash": schema_hash,
            "feature_contract_sha256": feature_contract_sha,
            "sanitizer_sha256": sanitizer_sha,
            "output_contract_sha256": output_contract_sha,
            "sessions": list(heads.keys()),
            "n_features": int(len(features)),
            "head_sample_counts": {k: int(head_metrics[k]["n_samples"]) for k in head_metrics},
            "training": {
                "years": [int(y) for y in args.years],
                "lookahead_bars": int(args.lookahead_bars),
                # XGB-10: TRUE label provenance (replaces the fabricated threshold_bps=6.0).
                # Labels are ATR-normalized triple-barrier: barrier_bps = atr_bps *
                # session_multiplier; a no-hit terminal return >= no_hit_return_factor * barrier
                # is promoted to LONG/SHORT. There is NO fixed threshold_bps.
                "label_method": "atr_triple_barrier_session_scaled",
                "atr_col": str(atr_col),
                "session_multipliers": {str(k): float(v) for k, v in session_multipliers.items()},
                "no_hit_return_factor": float(no_hit_return_factor),
                "barrier_formula": "barrier_bps = atr_bps * session_multiplier",
                "train_embargo_bars": int(os.environ.get("GX1_XGB_TRAIN_EMBARGO_BARS", str(int(args.lookahead_bars)))),
                "val_split": float(args.val_split),
                "seed": int(args.seed),
                "n_bars": int(args.n_bars) if args.n_bars is not None else None,
                "min_bars_per_head": int(args.min_bars_per_head),
                "use_class_weights": bool(args.use_class_weights),
                "log_margin_metrics": bool(args.log_margin_metrics),
                "vedtak": str(args.vedtak) if args.vedtak else None,  # N6 (2026-06-05): retrain provenance in meta
            },
            "xgb_params": {
                "max_depth": int(args.max_depth),
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "random_state": int(args.seed),
                "n_jobs": -1,
                "tree_method": "hist",
            },
            "head_metrics": head_metrics,
            "head_signatures": head_signatures,
        },
    }
    if calibration_meta is not None:
        model_data["meta"]["calibration"] = calibration_meta

    model_path = output_dir / "xgb_universal_multihead_v2.joblib"
    meta_path = output_dir / "xgb_universal_multihead_v2_meta.json"

    joblib_dump(model_data, model_path)
    model_sha = _compute_sha256(model_path)

    meta_json = _ensure_serializable(model_data["meta"])
    meta_path.write_text(json.dumps(meta_json, indent=2))
    meta_sha = _compute_sha256(meta_path)

    print(f"  Model path:            {model_path}")
    print(f"  Model sha256:          {model_sha}")
    print(f"  Meta path:             {meta_path}")
    print(f"  Meta sha256:           {meta_sha}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Bundle dir: {output_dir}")
    print(f"Sessions:   {list(heads.keys())}")
    for session, metrics in head_metrics.items():
        print(
            f"  {session}: "
            f"logloss={metrics['val_logloss']:.6f} "
            f"acc={metrics['val_accuracy']:.2%} "
            f"thr_bps={metrics['threshold_bps']:.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
